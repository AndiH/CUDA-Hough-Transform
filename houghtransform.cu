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
 * @brief Struct for an operator to be used in a conf mapping transform call
 * 
 * @param data A tuple of doubles to be conformal mapped
 * @return A tuple of doubles which have been conformal mapped
 */
struct confMap {
	__device__ thrust::tuple<double, double> operator() (thrust::tuple<double, double> &data) {
		double x = thrust::get<0>(data);
		double y = thrust::get<1>(data);
		
		double x2y2 = x*x+y*y;
// 		printf("x2y2 = %f\n", x2y2); // ### DEBUG
		
		return thrust::make_tuple(x/x2y2, -y/x2y2);
	}
};

// struct transf {
// 	__device__ thrust::device_vector<thrust::tuple<double, double> > operator() (thrust::tuple<double, double> &data) {
// 		thrust::device_vector<thrust::tuple<double, double> > tempVec;
// 		for (int i = 0; i < 10; i++) {
// 			tempVec.push_back(thrust::make_tuple(i, 2*i));
// 		}
// 		return tempVec;
// 	}
// };

/**
 * @brief (Outsourced) Function which actually does the hough transformation
 * @param alpha Angle to be hough transformed with, in degrees
 * @param x,y Coordinates of the to be transformed point
 * @return The hough transformed point corresponding to a certain angle (as a double)
 * 
 * This actual hough transforming method has been outsourced in order to call different hough transform functions from the operator mother function
 */
__device__ double houghTransfFunction(double alpha, double x, double y) {
	return (cos(alpha*1.74532925199432955e-02) * x + sin(alpha*1.74532925199432955e-02) * y);
}

/**
 * @brief Struct for an operator which hough transforms its input due to an exchangeable hough transforming method
 * @param data A tuple of (an angle and a tuple of (to be hough transformed x and y coordinates))
 * @return A hough transformed point corresponding to a certain angle
 * 
 * This method only runs on the device
 */
struct transf {
	__device__ double operator() (thrust::tuple<double, thrust::tuple<double, double> > data) {
		double alpha = thrust::get<0>(data);
		thrust::tuple<double, double> xy = thrust::get<1>(data);
		double x = thrust::get<0>(xy);	

		double y = thrust::get<1>(xy);
		
// 		printf("alpha * x + alpha * y = %f * %f + %f * %f\n", alpha, x, alpha, y);
		
		return houghTransfFunction(alpha, x, y);
	}
};

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

//	x.push_back(-0.287274); y.push_back(-1.668553); // MVD POINT EVENT 1
//	x.push_back(-0.378530); y.push_back(-2.244961); // MVD POINT EVENT 1
//	x.push_back(-0.557000); y.push_back(-3.422990); // MVD POINT EVENT 1
//	x.push_back(-0.763607); y.push_back(-4.893884); // MVD POINT EVENT 1
//	x.push_back(1.3155927); y.push_back(-1.276497); // MVD POINT EVENT 1
//	x.push_back(1.7415385); y.push_back(-1.742629); // MVD POINT EVENT 1
//	x.push_back(2.5251355); y.push_back(-2.654892); // MVD POINT EVENT 1
//	x.push_back(3.6173491); y.push_back(-4.050041); // MVD POINT EVENT 1

	x.push_back(8.6005649); y.push_back(-14.18077); // STT POINT EVENT 2
	x.push_back(8.8446016); y.push_back(-15.12664); // STT POINT EVENT 2
	x.push_back(9.0629711); y.push_back(-16.09391); // STT POINT EVENT 2
	x.push_back(9.1932706); y.push_back(-16.74574); // STT POINT EVENT 2
	x.push_back(9.3548145); y.push_back(-17.71571); // STT POINT EVENT 2
	x.push_back(9.4836196); y.push_back(-18.70058); // STT POINT EVENT 2
	x.push_back(9.5774650); y.push_back(-19.69842); // STT POINT EVENT 2
	x.push_back(9.6384601); y.push_back(-20.70427); // STT POINT EVENT 2
	x.push_back(9.6667957); y.push_back(-21.91839); // STT POINT EVENT 2
	x.push_back(9.6510515); y.push_back(-23.02375); // STT POINT EVENT 2


//	x.push_back(-1.687104); y.push_back(-1.412673); // MVD POINT EVENT 2
//	x.push_back(-3.564506); y.push_back(-3.116787); // MVD POINT EVENT 2
//	x.push_back(-6.729347); y.push_back(-6.360077); // MVD POINT EVENT 2
//	x.push_back(-8.854902); y.push_back(-8.857123); // MVD POINT EVENT 2
//	x.push_back(3.6133863); y.push_back(3.8080084); // MVD POINT EVENT 2
//	x.push_back(6.5745592); y.push_back(6.6399340); // MVD POINT EVENT 2
//	x.push_back(8.9718790); y.push_back(8.7770128); // MVD POINT EVENT 2
//	x.push_back(6.5918564); y.push_back(6.6557998); // MVD POINT EVENT 2
//	x.push_back(6.3520546); y.push_back(7.0406150); // MVD POINT EVENT 2
////
//	x.push_back(-11.55082); y.push_back(-12.50229); // STT POINTS EVENT 2
//	x.push_back(-12.05648); y.push_back(-13.25516); // STT POINTS EVENT 2
//	x.push_back(-12.51873); y.push_back(-13.96542); // STT POINTS EVENT 2
//	x.push_back(-12.99433); y.push_back(-14.72306); // STT POINTS EVENT 2
//	x.push_back(-13.44450); y.push_back(-15.46083); // STT POINTS EVENT 2
//	x.push_back(-13.89046); y.push_back(-16.22060); // STT POINTS EVENT 2
//	x.push_back(-14.32617); y.push_back(-16.98514); // STT POINTS EVENT 2
//	x.push_back(-14.74108); y.push_back(-17.74436); // STT POINTS EVENT 2
//	x.push_back(-15.21368); y.push_back(-18.64175); // STT POINTS EVENT 2
//	x.push_back(-15.62225); y.push_back(-19.44583); // STT POINTS EVENT 2
//	x.push_back(-16.17100); y.push_back(-20.58612); // STT POINTS EVENT 2
//	x.push_back(-16.51310); y.push_back(-21.33209); // STT POINTS EVENT 2
//	x.push_back(-17.03618); y.push_back(-22.52636); // STT POINTS EVENT 2
//	x.push_back(-17.34774); y.push_back(-23.26947); // STT POINTS EVENT 2
//	x.push_back(-17.85295); y.push_back(-24.55199); // STT POINTS EVENT 2
//	x.push_back(-18.12363); y.push_back(-25.27700); // STT POINTS EVENT 2
//	x.push_back(-18.61602); y.push_back(-26.68180); // STT POINTS EVENT 2
//	x.push_back(-18.85255); y.push_back(-27.39675); // STT POINTS EVENT 2
//	x.push_back(-19.13898); y.push_back(-28.31199); // STT POINTS EVENT 2
//	x.push_back(-19.41473); y.push_back(-29.24073); // STT POINTS EVENT 2
//	x.push_back(12.852836); y.push_back(11.957662); // STT POINTS EVENT 2
//	x.push_back(13.342592); y.push_back(12.337225); // STT POINTS EVENT 2
//	x.push_back(14.137040); y.push_back(12.941991); // STT POINTS EVENT 2
//	x.push_back(14.940003); y.push_back(13.539941); // STT POINTS EVENT 2
//	x.push_back(15.750968); y.push_back(14.130979); // STT POINTS EVENT 2
//	x.push_back(16.570022); y.push_back(14.714738); // STT POINTS EVENT 2
//	x.push_back(17.396743); y.push_back(15.290876); // STT POINTS EVENT 2
//	x.push_back(18.230621); y.push_back(15.859502); // STT POINTS EVENT 2
//	x.push_back(19.105146); y.push_back(16.441886); // STT POINTS EVENT 2
//	x.push_back(19.955823); y.push_back(16.995523); // STT POINTS EVENT 2
//	x.push_back(20.869682); y.push_back(17.577196); // STT POINTS EVENT 2
//	x.push_back(21.278230); y.push_back(17.831554); // STT POINTS EVENT 2
//	x.push_back(21.718780); y.push_back(18.102874); // STT POINTS EVENT 2
//	x.push_back(22.138130); y.push_back(18.357793); // STT POINTS EVENT 2
//	x.push_back(22.929496); y.push_back(18.831468); // STT POINTS EVENT 2
//	x.push_back(23.338378); y.push_back(19.071701); // STT POINTS EVENT 2
//	x.push_back(23.803163); y.push_back(19.342851); // STT POINTS EVENT 2
//	x.push_back(25.082901); y.push_back(20.070076); // STT POINTS EVENT 2
//	x.push_back(25.542038); y.push_back(20.325212); // STT POINTS EVENT 2
//	x.push_back(26.426181); y.push_back(20.807189); // STT POINTS EVENT 2
//	x.push_back(27.338024); y.push_back(21.292156); // STT POINTS EVENT 2
//	x.push_back(27.818302); y.push_back(21.541893); // STT POINTS EVENT 2



	//x.push_back(); y.push_back();
	
	/** 
	 * ### CONFORMAL MAPPING PART ###
	 */
	TStopwatch myWatch;
	//! change container from vec_x and vec_y to host_vector<tuple<x, y> >
// 	thrust::host_vector<thrust::tuple<double, double> > originalData(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.end())), thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end()))); // should work, though
	thrust::host_vector<thrust::tuple<double, double> > originalData;
	for (int i = 0; i < x.size(); i++) {
		originalData.push_back(thrust::make_tuple(x[i], y[i]));
	}
	thrust::device_vector<thrust::tuple<double, double> > d_originalData = originalData; //!< copy onto device
	
	if (verbose > 1) {
		for (int i = 0; i < originalData.size(); i++) {
			printTuple(originalData[i]);
			std::cout << std::endl;
		}
	}	
	
	thrust::device_vector<thrust::tuple<double, double> > d_mappedData(originalData.size()); //!< create empty device_vector for mapped data
	thrust::transform(d_originalData.begin(), d_originalData.end(), d_mappedData.begin(), confMap()); //!< conformal map data
	thrust::host_vector<thrust::tuple<double, double> > mappedData = d_mappedData; //!< copy back to host
	
	if (verbose > 0) {
		for (int i = 0; i < mappedData.size(); i++) {
			printTuple(mappedData[i]);
			std::cout << std::endl;
		}
	}
	
	/** 
	 * ### HOUGH TRANSFORM PART ###
	 */
	double everyXDegrees = 30; //!< make a point every X degrees of alpha
	if (argc > 1) everyXDegrees = (double)atof(argv[1]); //!< overwrite default value to what was given by command line
	thrust::host_vector<double> alphas(360/everyXDegrees); //!< create host vector with enough space for 360/everyXDegrees elements
	thrust::sequence(alphas.begin(), alphas.end(), 0., everyXDegrees); //!< fill the just created vector with alphas -- this vector is the basis for the hough transformation
	if (verbose > 1) {
		for (int i = 0; i < alphas.size(); i++) {
			std::cout << "alpha[" << i << "] = " << alphas[i] << std::endl;
		}
	}
	
	std::vector<thrust::host_vector<double> > transformedPoints; //!< create empty vector for all the hough transformed points, has size of: x.size() * 360/everyXDegrees
	
	for (int i = 0; i < mappedData.size(); i++) { //!< ON HOST! loop over all (conf mapped) data points
		thrust::device_vector<double> d_tempAlphas = alphas;
		thrust::device_vector<double> d_tempD(d_tempAlphas.size()); //!< create temp (and empty) device side vector to store the results of the hough transform in
		thrust::constant_iterator<thrust::tuple<double, double> > currentData(d_mappedData[i]); //!< create constant iterator for the conf mapped data 2-tuples
		
		//! following transformation uses the operator of transf to run over all elements
		//!   elements being a iterator from alpha.start to alpha.end with each time the constant iterator with the conf mapped 2-tuple
		//!   the result of the calculation is written in to the d_tempD vector
		thrust::transform(
			thrust::make_zip_iterator(
				thrust::make_tuple(
					d_tempAlphas.begin(),
					currentData
  				)
			),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					d_tempAlphas.end(),
					currentData
				)
			),
			d_tempD.begin(),
			transf()
		);
		
		thrust::host_vector<double> tempD = d_tempD; //!< copy device side array to host
		transformedPoints.push_back(tempD); //!< push it back to global data vector
	}
	
	/*
	 * ### Make CUSP Matrix ###
	 */
	int nBinsX = (int) 360/everyXDegrees;
	double minValueX = 0;
	double maxValueX = 360;
	if (verbose > 0) std::cout << "nBinsX = " << nBinsX << ", minValueX = " << minValueX << " ,maxValueX = " << maxValueX << std::endl;

	int nBinsY = 360/everyXDegrees;
	double minValueY = -0.07;
	double maxValueY = 0.07;
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
	for (int i = 0; i < mappedData.size(); i++) {
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
		std::cout << "Timings for first histogram:" << std::endl;
		std::cout << "  T for translating values: ";
		theObjects[0].GetSwTranslateValues()->Print();
		std::cout << "  T for sorting histogram vectors: ";
		theObjects[0].GetSwHistSort()->Print();
		std::cout << "  T for summing histogram vectors: ";
		theObjects[0].GetSwHistSum()->Print();
		std::cout << "  T for generating Matrix: ";
		theObjects[0].GetSwCreateTMatrixD()->Print();
		std::cout << "  T for generating TH2D: ";
		theObjects[0].GetSwCreateTH2D()->Print();
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
	 * create matrix from point
	 * 	use cusp & cusp example: http://code.google.com/p/cusp-library/source/browse/examples/MatrixAssembly/unordered_triplets.cu
	 * 	V[n] = 1
	 * 	calculate I[n] & J[n] due to resolution of matrix grid
	 * 	TESTCODE
	 * 		int anzahlderpunkte = 6;
	 * 		int gesamtbreite = 3;
	 * 		double zellenbreite = gesamtbreite / anzahlderpunkte;
	 * 		double testZahl = 2.6; // muss in die zelle #3 (1 --> 1,5)
	 * 		int testZahlMussInZelle = (int) (testZahl/zellenbreite);
	 * 
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
