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

#include "cuPrintf.cu"

#include "TH2D.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TMatrixD.h"
#include "TCanvas.h"
#include "TStopwatch.h"

#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include "AhTwoArraysToMatrix.h"
#include "AhHoughTransformation.h"

/**
 * @mainpage Conformal Mapping and Hough Transformation in CUDA Thrust
 * 
 * <h1>HoughTrust</h1>
 * <h2>A toy project during my PhD time</h2>
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
void readPoints(std::string filename, std::vector<double> &x, std::vector<double> &y, std::vector<double> &r, int upToLineNumber = 2) {
	std::ifstream file(filename.c_str());
	float tempX, tempY, tempZ, tempR, tempI;
	char tempChar[10];
	int i = 1;
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
void printTuple (thrust::tuple<double, double> thatTuple) {
	std::cout << thrust::get<0>(thatTuple) << " - " << thrust::get<1>(thatTuple);
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
	int verbose = 1;
	
	//! fill original data
	std::vector<double> x;
	std::vector<double> y;
	std::vector<double> r;

	readPoints("data.dat", x, y, r, 18);


	//! Preparations
	thrust::host_vector<double> h_x = x;
	thrust::host_vector<double> h_y = y;
	
	TStopwatch myWatch;

	//! Setting parameters
	double maxAngle = 180; //!< Hough transform ranges from 0 deg to 180 deg
	double everyXDegrees = 30; //!< make a point every X degrees of alpha; default = 30
	if (argc > 1) everyXDegrees = (double)atof(argv[1]); //!< overwrite default value to what was given by command line

	AhHoughTransformation * houghTrans = new AhHoughTransformation(h_x, h_y, maxAngle, everyXDegrees);
	thrust::device_vector<double> alphas = houghTrans->GetAngles();
	std::vector<thrust::device_vector<double> > transformedPoints = houghTrans->GetVectorOfTransformedPoints();

	/*
	 * ### Make CUSP Matrix ###
	 */
	int nBinsX = (int) maxAngle/everyXDegrees;
	double minValueX = 0;
	double maxValueX = maxAngle;
	if (verbose > 0) std::cout << "nBinsX = " << nBinsX << ", minValueX = " << minValueX << " ,maxValueX = " << maxValueX << std::endl;

	int nBinsY = maxAngle/everyXDegrees;
	double minValueY = -0.4;
	double maxValueY = 0.7;
	if (verbose > 0) std::cout << "nBinsY = " << nBinsY << ", minValueY = " << minValueY << ", maxValueY " << maxValueY << std::endl;
	// ALTERNATIVE DEFINITION OF M**VALUEY: using automated mins and maxes
//	double minValueY = transformedPoints[0][0];
//	double maxValueY = transformedPoints[0][0];
//	for (int i = 0; i < mappedData.size(); i++) {
//		thrust::device_vector<double> tempD(transformedPoints[i]);
//		double minimum = thrust::reduce(tempD.begin(), tempD.end(), std::numeric_limits<double>::max(), thrust::minimum<double>());
//		double maximum = thrust::reduce(tempD.begin(), tempD.end(), std::numeric_limits<double>::max(), thrust::maximum<double>());
//		if (minimum < minValue) minValueY = minimum;
//		if (maximum > maxValue) maxValueY = maximum;
//	}
//	minValueY -= everyXDegrees;
//	maxValueY -= everyXDegrees;
	
	// Now create the Matrices
	std::vector<TH2D*> theHistograms;
	std::vector< cusp::coo_matrix<int, double, cusp::device_memory> > theMatrices;
	std::vector<AhTwoArraysToMatrix> theObjects;
	for (int i = 0; i < transformedPoints.size(); i++) {
		AhTwoArraysToMatrix tempObject(
			thrust::device_vector<double> (alphas), thrust::device_vector<double> (transformedPoints[i]),
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
		thatHist->GetXaxis()->SetTitle("Angle / #circ");
		thatHist->GetYaxis()->SetTitle("Hough transformed");
	
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
	 * Get Matrix min max automatically
	 * MAKE CLASS OF THAT
	 * 
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
