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

#include "../../CUDA/local/cuPrintf.cu"

struct confMap {
	__device__ thrust::tuple<double, double> operator() (thrust::tuple<double, double> &data) {
		double x = thrust::get<0>(data);
		double y = thrust::get<1>(data);
		
		double x2y2 = x*x+y*y;
// 		printf("x2y2 = %f\n", x2y2);
		
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

__device__ double houghTransfFunction(double alpha, double x, double y) {
	return (cos(alpha) * x + sin(alpha) * y);
}

struct transf {
	__device__ double operator() (thrust::tuple<double, thrust::tuple<double, double> > data) {
		double alpha = thrust::get<0>(data);
		thrust::tuple<double, double> xy = thrust::get<1>(data);
		double x = thrust::get<0>(xy);
		double y = thrust::get<1>(xy);
		
		printf("alpha * x + alpha * y = %f * %f + %f * %f\n", alpha, x, alpha, y);
		
		return houghTransfFunction(alpha, x, y);
	}
};

void printTuple (thrust::tuple<double, double> thatTuple) {
	std::cout << thrust::get<0>(thatTuple) << " - " << thrust::get<1>(thatTuple);
}

int main (int argc, char** argv) {
	int verbose = 2;
	
	// fill original data
	std::vector<double> x;
	std::vector<double> y;
	x.push_back(-3.89038); y.push_back(-3.14085);
	x.push_back(-8.42652); y.push_back(-5.38459);
	x.push_back(-13.4702); y.push_back(-6.59945);
	x.push_back(-18.8446); y.push_back(-6.69932);
	x.push_back(-24.3888); y.push_back(-5.49439);
	x.push_back(-29.8641); y.push_back(-2.8527);
	//x.push_back(); y.push_back();
	
	/* 
	 * ### CONFORMAL MAPPING PART ###
	 */
	
	// change container from vec_x and vec_y to host_vector<tuple<x, y> >
// 	thrust::host_vector<thrust::tuple<double, double> > originalData(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.end())), thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end()))); // should work, though
	thrust::host_vector<thrust::tuple<double, double> > originalData;
	for (int i = 0; i < x.size(); i++) {
		originalData.push_back(thrust::make_tuple(x[i], y[i]));
	}
	thrust::device_vector<thrust::tuple<double, double> > d_originalData = originalData; // copy on device
	
	if (verbose > 1) {
		for (int i = 0; i < originalData.size(); i++) {
			printTuple(originalData[i]);
			std::cout << std::endl;
		}
	}	
	
	thrust::device_vector<thrust::tuple<double, double> > d_mappedData(originalData.size()); // create empty device_vector for mapped data
	thrust::transform(d_originalData.begin(), d_originalData.end(), d_mappedData.begin(), confMap()); // map data
	thrust::host_vector<thrust::tuple<double, double> > mappedData = d_mappedData; // copy back to host
	
	if (verbose > 0) {
		for (int i = 0; i < mappedData.size(); i++) {
			printTuple(mappedData[i]);
			std::cout << std::endl;
		}
	}
	
	/* 
	 * ### HOUGH TRANSFORM PART ###
	 */
	double everyXDegrees = 30; // make a point every X degrees of alpha
	thrust::host_vector<double> alphas(360/everyXDegrees); // create host vector with enough space for 360/everyXDegrees elements
	thrust::sequence(alphas.begin(), alphas.end(), 0., everyXDegrees); // fill the just created vector with alphas -- this vector is the basis for the hough transformation
	if (verbose > 1) {
		for (int i = 0; i < alphas.size(); i++) {
			std::cout << "alpha[" << i << "] = " << alphas[i] << std::endl;
		}
	}
	
	std::vector<thrust::host_vector<double> > transformedPoints; // create empty vector for all the hough transformed points, has size of: x.size() * 360/everyXDegrees
	
	for (int i = 0; i < mappedData.size(); i++) { // ON HOST! loop over all (conf mapped) data points
		thrust::device_vector<double> d_tempAlphas = alphas;
		thrust::device_vector<double> d_tempD(d_tempAlphas.size()); // create temp (and empty) device side vector to store the results of the hough transform in
		thrust::constant_iterator<thrust::tuple<double, double> > currentData(d_mappedData[i]); // create constant iterator for the conf mapped data 2-tuples
		
		// following transformation uses the operator of transf to run over all elements
		//   elements being a iterator from alpha.start to alpha.end with each time the constant iterator with the conf mapped 2-tuple
		//   the result of the calculation is written in to the d_tempD vector
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
		
		thrust::host_vector<double> tempD = d_tempD; // copy device side array to host
		transformedPoints.push_back(tempD); // push it back to global data vector
	}
	
	for (int i = 0; i < transformedPoints.size(); i++) {
		std::cout << "transformedPoints[" << i << "].size() = " << transformedPoints[i].size() << std::endl;
		for (int j = 0; j < transformedPoints[i].size(); j++) {
			std::cout << "  transformedPoints[" << i << "] = " << transformedPoints[i][j] << std::endl;
		}
	}
	
	/*
	 * 
	 * ### TODO ###
	 * make peak finder
	 * outline:
	 * 	create rough grid
	 * 		find max of trasnformedPoints via thrust lib to determine borders
	 * 	divide grid
	 * 	find max
	 * 	create fine grid
	 * 	find peaks
	 */
}