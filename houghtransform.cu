// The following compiler flag will change the default usage of floats in this class to doubles
// Use, e.g., as follows: g++ hough.cpp AhHoughTransformation.o -DUSE_DOUBLES=1 
#ifdef USE_DOUBLES
	typedef double TYPE;
#else
	typedef float TYPE;
#endif

#include <iostream>
#include <fstream>
#include <algorithm> // to find minimum --> min_element
#include "stdio.h"
#include <vector>
#include <cuda.h>
#include "math.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>

#include "TH2D.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TMatrixD.h"
#include "TCanvas.h"
#include "TStopwatch.h"
#include "TStyle.h"

#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include "AhTwoArraysToMatrix.h"
#include "AhHoughTransformation.h"


/**
 * @mainpage Conformal Mapping and Hough Transformation in CUDA Thrust
 * 
 * <h1>HoughTrust</h1>
 * <h2>A PhD project</h2>
 * 
 * Using CUDA Thrust to conformal map and hough transform points of a tracker. Two dimensionally.
 * 
 * @author Andreas Herten
 */

/**
 * @file houghtransform.cu
 * 
 * @brief Everything is in that one file
 * 
 * Sorry for that, but if/when I'm going to need any special functions, I will extract them
 **/

/**
 * @brief Reads data from specific formated file into three given vectors
 * @param filename       Filename (include path if necessary)
 * @param x              x-value vector (1st row of file)
 * @param y              y-value vector (2nd row of file)
 * @param r              r-value vector (4th row of file)
 * @param upToLineNumber Up to with line (including that line) the file will be read. Starts at line 1.
 *
 * Uses just x, y and r at the moment, because that's all I need.
 */
void readPoints(std::string filename, std::vector<TYPE> &x, std::vector<TYPE> &y, std::vector<TYPE> &r, int upToLineNumber = 2) {
	std::ifstream file(filename.c_str());
	float tempX, tempY, tempZ, tempR, tempI;
	char tempChar[10];
	int i = 1;
	if (!file.good() || file.fail()) std::cout << "Failed opening file!" << std::endl;
	while(i <= upToLineNumber && file >> tempX >> tempY >> tempZ >> tempR >> tempI >> tempChar) {
		x.push_back(tempX);
		y.push_back(tempY);
		r.push_back(tempR);
		i++;
	}
	file.close();
}

/**
 * @brief Helper function which simply prints out the contents of a tuple
 * @param thatTuple A tuple of doubles to be printed
 * @return Nothing, it's a void.
 */
void printTuple (thrust::tuple<TYPE, TYPE> thatTuple) {
	std::cout << thrust::get<0>(thatTuple) << " - " << thrust::get<1>(thatTuple);
}

/**
 * @brief Prints out every element of a vector
 * @param Vector - and a type (this method is a template)
 */
template <class T>
void printVector (const T & v) {
	for (int i = 0; i < v.size(); ++i) {
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}

/**
 * @brief Adds up TH2D histograms
 * @param vHistograms A std vector with pointers to TH2D histograms
 * @return A pointer to a TH2D histogram
 */
TH2D * addVectorOfPToHistograms (std::vector<TH2D* > vHistograms) {
	TH2D * tempHist = new TH2D(*(vHistograms[0]));
	for (int i = 1; i < vHistograms.size(); i++) {
		tempHist->Add(vHistograms[i]);
	}
	return tempHist;
}

int main (int argc, char** argv) {
	gStyle->SetOptStat(0);
	int verbose = 1;
	
	//! fill original data
	std::vector<TYPE> x;
	std::vector<TYPE> y;
	std::vector<TYPE> r;
	// readPoints("data.dat", x, y, r, 18);

	std::string inputFileName = "real_data.txt";
	int readInUpToLineNumber = 20;
	if (argc > 2) readInUpToLineNumber = static_cast<int>(atof(argv[2]));
	if (argc > 3) inputFileName = argv[3];
	readPoints(inputFileName, x, y, r, readInUpToLineNumber);

	//! Change container from std::vector to thrust::host_vector
	thrust::host_vector<TYPE> h_x = x;
	thrust::host_vector<TYPE> h_y = y;
	thrust::host_vector<TYPE> h_r = r;
	
	TStopwatch myWatch;

	//! Setting parameters
	TYPE maxAngle = 180; //!< Hough transform ranges from 0 deg to 180 deg
	TYPE everyXDegrees = 30; //!< make a point every X degrees of alpha; default = 30
	if (argc > 1) everyXDegrees = (TYPE)atof(argv[1]); //!< overwrite default value to what was given by command line

	//! Simple (x,y) coordinates
	AhHoughTransformation * houghTrans = new AhHoughTransformation(h_x, h_y, maxAngle, everyXDegrees, true);

	//! Use isochrones - (x,y,r) coordinates
	// maxAngle *= 2; //!< for isochrones, hough transformation goes from 0 to 360
	// AhHoughTransformation * houghTrans = new AhHoughTransformation(h_x, h_y, h_r, maxAngle, everyXDegrees, true);

	thrust::device_vector<TYPE> alphas = houghTrans->GetAngles();
	std::vector<thrust::device_vector<TYPE> > transformedPoints = houghTrans->GetVectorOfTransformedPoints();

	// std::cout << "It took " << houghTrans->GetTimeHoughTransform()/1000 << "s for actual HT." << std::endl;

	/*
	 * ### Make CUSP Matrix ###
	 */
	//! Find upper and lower borders of histograms
	int nBinsX = (int) maxAngle/everyXDegrees;
	TYPE minValueX = 0;
	TYPE maxValueX = maxAngle;
	if (verbose > 0) std::cout << "nBinsX = " << nBinsX << ", minValueX = " << minValueX << ", maxValueX = " << maxValueX << std::endl;

	int nBinsY = maxAngle/everyXDegrees;
	// TYPE minValueY = -0.4;
	// TYPE maxValueY = 0.7;
	// if (verbose > 0) std::cout << "nBinsY = " << nBinsY << ", minValueY = " << minValueY << ", maxValueY " << maxValueY << std::endl;
	//! Automatically get y borders
	std::cout << transformedPoints.size() << std::endl;
	TYPE minValueY = transformedPoints[0][0];
	TYPE maxValueY = transformedPoints[0][0];
	std::cout << "minValueY = " << minValueY << ", maxValueY = " << maxValueY << std::endl;
	for (int i = 0; i < transformedPoints.size(); i++) {
		thrust::device_vector<TYPE> tempD(transformedPoints[i]);
		TYPE minimum = *(thrust::min_element(tempD.begin(), tempD.end()));
		TYPE maximum = *(thrust::max_element(tempD.begin(), tempD.end()));
		if (minimum < minValueY) minValueY = minimum;
		if (maximum > maxValueY) maxValueY = maximum;
		std::cout << "minimum = " << minimum << ", maximum = " << maximum << std::endl;
	}
	minValueY *= 1.1; //!< make edges a little smoother
	maxValueY *= 1.1; //!< make edges a little smoother
	std::cout << "minValueY = " << minValueY << ", maxValueY = " << maxValueY << std::endl;
	
	//! Create matrices
	std::vector<TH2D*> theHistograms;
	std::vector< cusp::coo_matrix<int, TYPE, cusp::device_memory> > theMatrices;
	std::vector<AhTwoArraysToMatrix> theObjects;
	for (int i = 0; i < transformedPoints.size(); i++) {
		AhTwoArraysToMatrix tempObject(
			thrust::device_vector<TYPE> (alphas), thrust::device_vector<TYPE> (transformedPoints[i]),
			nBinsX,
			minValueX,
			maxValueX,
			nBinsY,
			minValueY,
			maxValueY,
			true,
			true
		);

		if (verbose > 0) std::cout << "Some matrix parameters: " << tempObject.GetNBinsX() << " " << tempObject.GetXlow() << " " << tempObject.GetXup() << " "<< tempObject.GetNBinsY() <<  " " << tempObject.GetYlow() << " " << tempObject.GetYup() << std::endl;

		theHistograms.push_back(tempObject.GetHistogram());
		theMatrices.push_back(tempObject.GetCUSPMatrix());
		theObjects.push_back(tempObject);
		char tempchar[5];
		sprintf(tempchar, "%d", i);
		theHistograms[i]->SetName(tempchar);
	}


	/*
	 * ### DEBUG Output ###
	 */
	if (verbose > 5) cusp::print(theMatrices[0]);

	if (verbose > 0) { // Timings
		for (int i = 0; i < theObjects.size(); i++) {
			std::cout << "Timings for histogram " << i << std::endl;
			std::cout << "  T for translating values: " << theObjects[i].GetTimeTranslateValues() << std::endl;
			std::cout << "  T for sorting histogram vectors: " << theObjects[i].GetTimeHistSort() << std::endl;
			std::cout << "  T for summing histogram vectors: " << theObjects[i].GetTimeHistSum() << std::endl;
			std::cout << "  T for generating Matrix: " << theObjects[i].GetTimeCreateTMatrixD() << std::endl;
			std::cout << "  T for generating TH2D: " << theObjects[i].GetTimeCreateTH2D() << std::endl;
		}
	}
	
	if (verbose > 1) {
	for (int i = 0; i < transformedPoints.size(); i++) {
		std::cout << "transformedPoints[" << i << "].size() = " << transformedPoints[i].size() << std::endl;
		for (int j = 0; j < transformedPoints[i].size(); j++) {
			std::cout << "  transformedPoints[" << i << "][" << j << "] = " << transformedPoints[i][j] << std::endl;
		}
	}
	}
	
	myWatch.Stop();
	std::cout << "For operations, it took me ";
	myWatch.Print();
	std::cout << std::endl;
	
	/*
	 * ### ROOT VISUALIZATION ###
	 */
	
	bool doRoot = true;
	if (argc > 2) doRoot = (bool)atof(argv[2]);
	
	if (doRoot) {

		TH2D * thatHist = addVectorOfPToHistograms(theHistograms);
		thatHist->GetXaxis()->SetTitle("#alpha / #circ");
		thatHist->GetYaxis()->SetTitle("r / cm");
	
//		for (int i = 0; i < transformedPoints.size(); i++) {
//			for (int j = 0; j < transformedPoints[i].size(); j++) {
//				thatHist->Fill(alphas[j], transformedPoints[i][j]);
//			}
//		}
		
		TApplication *theApp = new TApplication("app", &argc, argv, 0, -1);
		TCanvas * c1 = new TCanvas("c1", "default", 100, 10, 800, 600);
		
		thatHist->Draw("COLZ");
		c1->Update();
		theApp->Run();
	}
	
	
	/*
	 * 
	 * ### TODO ###
	 * 
	 * make peak finder
	 * outline:
	 * 	create rough grid
	 * 		find max of trasnformedPoints via thrust lib to determine borders
	 * 	divide grid
	 * 	find max
	 * 	create fine grid
	 * 	find peaks
	
	
	LINKS FOR MATRIX STUFF:
	
	https://www.google.com/search?q=thrust+matrix
	https://www.google.com/search?q=thrust+matrix+multiplication&sugexp=chrome,mod=10&sourceid=chrome&ie=UTF-8
	http://stackoverflow.com/questions/618511/a-proper-way-to-create-a-matrix-in-c
	http://www.velocityreviews.com/forums/t281152-is-there-any-matrix-in-the-stl.html
	BLAS
	Blitz++
	
	 */
}
