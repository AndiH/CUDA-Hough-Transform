#ifndef AHHOUGHTRANSFORMATION_H
#define AHHOUGHTRANSFORMATION_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace my { // I don't know if this is a good idea are bad coding style
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
}

class AhHoughTransformation
{
public:
	AhHoughTransformation();
	AhHoughTransformation(thrust::host_vector<double>, thrust::host_vector<double>);
	
	virtual ~AhHoughTransformation();
protected:
	DoChangeContainerToTwoTuples();
	DoConformalMapping();
	DoHoughTransform();
private:
	thrust::device_vector<double> fXValues;
	thrust::device_vector<double> fYValues;
	thrust::device_vector<thrust::tuple<double, double> > fXYValues;

};
#endif //AHHOUGHTRANSFORMATION_H