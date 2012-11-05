/*
 * Ah2DPeakfinder.h
 *
 *  Created on: Sep 21, 2012
 *      Author: Andreas Herten
 *
 */

#ifndef AH2DPEAKFINDER_H_
#define AH2DPEAKFINDER_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>

//#include "cuPrintf.cuh"

//#include "AhTwoArraysToMatrix.h"

template <class InputType, class ReturnType>
struct returnElementOfTuple : public thrust::binary_function<InputType, int, ReturnType> {
	__host__ __device__ ReturnType operator() (const InputType &inputTuple, const int &returnIndex) const {
		if (0 == returnIndex) {
			return thrust::get<0>(inputTuple);
		} else if (1 == returnIndex) {
			return thrust::get<1>(inputTuple);
		} else { // if (2 == returnIndex)
			return thrust::get<2>(inputTuple);
		}
	}
};

// make a template out of it to get a proper comparison at return line?
struct isGreaterThan { // there's a Thrust version for this: http://docs.thrust.googlecode.com/hg/structthrust_1_1greater__equal.html - maybe implement this here as a wrapper?
	isGreaterThan(int _cutOff) : cutOff(_cutOff) {};
	__device__ bool operator() (const double x) const {
//		cuPrintf("x = %f, cutOff = %f\n", x, cutOff);
		return x > cutOff;
	}
private:
	int cutOff;
};

class Ah2DPeakfinder // : protected AhTwoArraysToMatrix
{
public:
	Ah2DPeakfinder();
	Ah2DPeakfinder(thrust::device_vector<thrust::tuple<int, int, double> >, int cutOff = 2); // cutOff can once be a double value, when weights are implemented
	Ah2DPeakfinder(thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<double>, int cutOff = 2);

	virtual ~Ah2DPeakfinder();

	thrust::device_vector<int> GetPlainXValues() { return fX; };
	thrust::device_vector<int> GetPlainYValues() { return fY; };
	thrust::device_vector<double> GetMultiplicities() { return fValues; };

protected:
	void DoEverything();
	void DoCutOff();
	void DoSortByMultiplicity();

private:
	thrust::device_vector<int> fX;
	thrust::device_vector<int> fY;
	thrust::device_vector<double> fValues;
	int fCutOff;
};


#endif /* AH2DPEAKFINDER_H_ */
