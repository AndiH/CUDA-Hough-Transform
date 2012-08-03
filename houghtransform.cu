#include <iostream>
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
#include "TCanvas.h"
#include "TStopwatch.h"

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

int main (int argc, char** argv) {
	int verbose = 0;
	
	//! fill original data
	std::vector<double> x;
	std::vector<double> y;
	x.push_back(-3.89038); y.push_back(-3.14085);
	x.push_back(-8.42652); y.push_back(-5.38459);
	x.push_back(-13.4702); y.push_back(-6.59945);
	x.push_back(-18.8446); y.push_back(-6.69932);
	x.push_back(-24.3888); y.push_back(-5.49439);
	x.push_back(-29.8641); y.push_back(-2.8527);
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
		transform(
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
	
	if (verbose > 0) {
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
		TH2D * thatHist = new TH2D("thatHist", "Hist", (int)360/everyXDegrees, 0, 180, (int)360, -0.2, 0.2);
		thatHist->GetXaxis()->SetTitle("Angle / #circ");
		thatHist->GetYaxis()->SetTitle("Hough transformed");
	
		for (int i = 0; i < transformedPoints.size(); i++) {
			for (int j = 0; j < transformedPoints[i].size(); j++) {
				thatHist->Fill(alphas[j], transformedPoints[i][j]);
			}
		}
		
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
