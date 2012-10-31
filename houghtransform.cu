#include <iostream>
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
//	x.push_back(-3.89038); y.push_back(-3.14085);
//	x.push_back(-8.42652); y.push_back(-5.38459);
//	x.push_back(-13.4702); y.push_back(-6.59945);
//	x.push_back(-18.8446); y.push_back(-6.69932);
//	x.push_back(-24.3888); y.push_back(-5.49439);
//	x.push_back(-29.8641); y.push_back(-2.8527);

	x.push_back(-0.287274); y.push_back(-1.668553); // MVD POINT EVENT 1
	x.push_back(-0.378530); y.push_back(-2.244961); // MVD POINT EVENT 1
	x.push_back(-0.557000); y.push_back(-3.422990); // MVD POINT EVENT 1
// 	x.push_back(-0.763607); y.push_back(-4.893884); // MVD POINT EVENT 1
// 	x.push_back(1.3155927); y.push_back(-1.276497); // MVD POINT EVENT 1
// 	x.push_back(1.7415385); y.push_back(-1.742629); // MVD POINT EVENT 1
// 	x.push_back(2.5251355); y.push_back(-2.654892); // MVD POINT EVENT 1
// 	x.push_back(3.6173491); y.push_back(-4.050041); // MVD POINT EVENT 1

// 	x.push_back(8.6005649); y.push_back(-14.18077); // STT POINT EVENT 2
// 	x.push_back(8.8446016); y.push_back(-15.12664); // STT POINT EVENT 2
// 	x.push_back(9.0629711); y.push_back(-16.09391); // STT POINT EVENT 2
// 	x.push_back(9.1932706); y.push_back(-16.74574); // STT POINT EVENT 2
// 	x.push_back(9.3548145); y.push_back(-17.71571); // STT POINT EVENT 2
// 	x.push_back(9.4836196); y.push_back(-18.70058); // STT POINT EVENT 2
// 	x.push_back(9.5774650); y.push_back(-19.69842); // STT POINT EVENT 2
// 	x.push_back(9.6384601); y.push_back(-20.70427); // STT POINT EVENT 2
// 	x.push_back(9.6667957); y.push_back(-21.91839); // STT POINT EVENT 2
// 	x.push_back(9.6510515); y.push_back(-23.02375); // STT POINT EVENT 2


// 	x.push_back(-1.687104); y.push_back(-1.412673); // MVD POINT EVENT 2
// 	x.push_back(-3.564506); y.push_back(-3.116787); // MVD POINT EVENT 2
// 	x.push_back(-6.729347); y.push_back(-6.360077); // MVD POINT EVENT 2
// 	x.push_back(-8.854902); y.push_back(-8.857123); // MVD POINT EVENT 2
// 	x.push_back(3.6133863); y.push_back(3.8080084); // MVD POINT EVENT 2
// 	x.push_back(6.5745592); y.push_back(6.6399340); // MVD POINT EVENT 2
// 	x.push_back(8.9718790); y.push_back(8.7770128); // MVD POINT EVENT 2
// 	x.push_back(6.5918564); y.push_back(6.6557998); // MVD POINT EVENT 2
// 	x.push_back(6.3520546); y.push_back(7.0406150); // MVD POINT EVENT 2
// //
// 	x.push_back(-11.55082); y.push_back(-12.50229); // STT POINTS EVENT 2
// 	x.push_back(-12.05648); y.push_back(-13.25516); // STT POINTS EVENT 2
// 	x.push_back(-12.51873); y.push_back(-13.96542); // STT POINTS EVENT 2
// 	x.push_back(-12.99433); y.push_back(-14.72306); // STT POINTS EVENT 2
// 	x.push_back(-13.44450); y.push_back(-15.46083); // STT POINTS EVENT 2
// 	x.push_back(-13.89046); y.push_back(-16.22060); // STT POINTS EVENT 2
// 	x.push_back(-14.32617); y.push_back(-16.98514); // STT POINTS EVENT 2
// 	x.push_back(-14.74108); y.push_back(-17.74436); // STT POINTS EVENT 2
// 	x.push_back(-15.21368); y.push_back(-18.64175); // STT POINTS EVENT 2
// 	x.push_back(-15.62225); y.push_back(-19.44583); // STT POINTS EVENT 2
// 	x.push_back(-16.17100); y.push_back(-20.58612); // STT POINTS EVENT 2
// 	x.push_back(-16.51310); y.push_back(-21.33209); // STT POINTS EVENT 2
// 	x.push_back(-17.03618); y.push_back(-22.52636); // STT POINTS EVENT 2
// 	x.push_back(-17.34774); y.push_back(-23.26947); // STT POINTS EVENT 2
// 	x.push_back(-17.85295); y.push_back(-24.55199); // STT POINTS EVENT 2
// 	x.push_back(-18.12363); y.push_back(-25.27700); // STT POINTS EVENT 2
// 	x.push_back(-18.61602); y.push_back(-26.68180); // STT POINTS EVENT 2
// 	x.push_back(-18.85255); y.push_back(-27.39675); // STT POINTS EVENT 2
// 	x.push_back(-19.13898); y.push_back(-28.31199); // STT POINTS EVENT 2
// 	x.push_back(-19.41473); y.push_back(-29.24073); // STT POINTS EVENT 2
// 	x.push_back(12.852836); y.push_back(11.957662); // STT POINTS EVENT 2
// 	x.push_back(13.342592); y.push_back(12.337225); // STT POINTS EVENT 2
// 	x.push_back(14.137040); y.push_back(12.941991); // STT POINTS EVENT 2
// 	x.push_back(14.940003); y.push_back(13.539941); // STT POINTS EVENT 2
// 	x.push_back(15.750968); y.push_back(14.130979); // STT POINTS EVENT 2
// 	x.push_back(16.570022); y.push_back(14.714738); // STT POINTS EVENT 2
// 	x.push_back(17.396743); y.push_back(15.290876); // STT POINTS EVENT 2
// 	x.push_back(18.230621); y.push_back(15.859502); // STT POINTS EVENT 2
// 	x.push_back(19.105146); y.push_back(16.441886); // STT POINTS EVENT 2
// 	x.push_back(19.955823); y.push_back(16.995523); // STT POINTS EVENT 2
// 	x.push_back(20.869682); y.push_back(17.577196); // STT POINTS EVENT 2
// 	x.push_back(21.278230); y.push_back(17.831554); // STT POINTS EVENT 2
// 	x.push_back(21.718780); y.push_back(18.102874); // STT POINTS EVENT 2
// 	x.push_back(22.138130); y.push_back(18.357793); // STT POINTS EVENT 2
// 	x.push_back(22.929496); y.push_back(18.831468); // STT POINTS EVENT 2
// 	x.push_back(23.338378); y.push_back(19.071701); // STT POINTS EVENT 2
// 	x.push_back(23.803163); y.push_back(19.342851); // STT POINTS EVENT 2
// 	x.push_back(25.082901); y.push_back(20.070076); // STT POINTS EVENT 2
// 	x.push_back(25.542038); y.push_back(20.325212); // STT POINTS EVENT 2
// 	x.push_back(26.426181); y.push_back(20.807189); // STT POINTS EVENT 2
// 	x.push_back(27.338024); y.push_back(21.292156); // STT POINTS EVENT 2
// 	x.push_back(27.818302); y.push_back(21.541893); // STT POINTS EVENT 2

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
