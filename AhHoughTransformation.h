#ifndef AHHOUGHTRANSFORMATION_H
#define AHHOUGHTRANSFORMATION_H

// The following compiler flag will change the default usage of floats in this class to doubles
// Use, e.g., as follows: g++ hough.cpp AhHoughTransformation.o -DUSE_DOUBLES=1 
#ifdef USE_DOUBLES
	typedef double TYPE;
#else
	typedef float TYPE;
#endif

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

namespace my { // I don't know if this is a good idea are bad coding style
	/**
	 * @brief Struct for an operator to be used in a conf mapping transform call
	 * 
	 * @param data A tuple of doubles to be conformal mapped
	 * @return A tuple of doubles which have been conformal mapped
	 */
	struct confMap {
		__device__ thrust::tuple<TYPE, TYPE> operator() (thrust::tuple<TYPE, TYPE> &data) {
			TYPE x = thrust::get<0>(data);
			TYPE y = thrust::get<1>(data);
			
			TYPE x2y2 = x*x+y*y;
		//	printf("x2y2 = %f\n", x2y2); // ### DEBUG
			
			return thrust::make_tuple(x/x2y2, -y/x2y2);
		}

		// Testing code for isochrones: 
		__device__ TYPE operator() (thrust::tuple<TYPE&, TYPE&, TYPE&> &data) {
			TYPE value = thrust::get<0>(data);
			TYPE x = thrust::get<1>(data);
			TYPE y = thrust::get<2>(data);

			TYPE x2y2 = x*x+y*y;
			
			return value/x2y2;
		}
	};

	/**
	* @brief Struct for an operator which hough transforms its input according to an exchangeable hough transforming method
	* 
	* This method only runs on the device
	*/
	struct htransf {
		/**
		* @brief (Outsourced) Function which actually does the hough transformation
		* @param alpha Angle to be hough transformed with, in degrees
		* @param x,y Coordinates of the to be transformed point
		* @return The hough transformed point corresponding to a certain angle (as a double)
		* 
		* This actual hough transforming method has been outsourced in order to call different hough transform functions from the operator mother function.
		* So, if you want to change the way the hough transform is done, just implement another houghTransFunction function.
		*/
		__device__ TYPE houghTransfFunction(TYPE alpha, TYPE x, TYPE y) {
			return (cos(alpha*1.74532925199432955e-02) * x + sin(alpha*1.74532925199432955e-02) * y);
		}

		/**
		* @param data A tuple of (an angle and a tuple of (to be hough transformed x and y coordinates))
		* @return A hough transformed point corresponding to a certain angle
		*/
		__device__ TYPE operator() (thrust::tuple<TYPE &, thrust::tuple<TYPE, TYPE> > data) {
			TYPE alpha = thrust::get<0>(data);
			thrust::tuple<TYPE, TYPE> xy = thrust::get<1>(data);
			TYPE x = thrust::get<0>(xy);	

			TYPE y = thrust::get<1>(xy);

	// 		printf("alpha * x + alpha * y = %f * %f + %f * %f\n", alpha, x, alpha, y);

			return houghTransfFunction(alpha, x, y);
		}

		// Testing code for isochrones:
		__device__ TYPE operator() (thrust::tuple<TYPE &, thrust::tuple<TYPE, TYPE, TYPE> > data) {
			TYPE alpha = thrust::get<0>(data);
			thrust::tuple<TYPE, TYPE, TYPE> xyr = thrust::get<1>(data);
			TYPE x = thrust::get<0>(xyr);
			TYPE y = thrust::get<1>(xyr);
			TYPE r = thrust::get<2>(xyr);

			return (houghTransfFunction(alpha, x, y) + r);
		}
	};
}

class AhHoughTransformation
{
public:
	AhHoughTransformation();
	AhHoughTransformation(thrust::host_vector<TYPE>, thrust::host_vector<TYPE>, TYPE maxAngle = 180., TYPE everyXDegrees = 30., bool doTiming = false);
	AhHoughTransformation(thrust::host_vector<TYPE>, thrust::host_vector<TYPE>, thrust::host_vector<TYPE>, TYPE maxAngle = 360., TYPE everyXDegrees = 30., bool doTiming = false);
	
	virtual ~AhHoughTransformation();

	thrust::device_vector<TYPE> GetAngles() { return fAngles; };
	std::vector<thrust::device_vector<TYPE> > GetVectorOfTransformedPoints() {return fTransformedPoints; };

	float GetTimeConfMap() { return fTimeConfMap; }; // in milli seconds
	float GetTimeGenAngles() { return fTimeGenAngles; }; // in milli seconds
	float GetTimeHoughTransform() { return fTimeHoughTransform; }; // in milli seconds

protected:
	void DoEverything();
	void DoConformalMapping();
	void DoGenerateAngles();
	template <class T>
	void DoHoughTransformOnePoint(thrust::constant_iterator<T>, thrust::device_vector<TYPE> &);
	void DoHoughTransform();

	void EventTiming_start();
	float EventTiming_stop();

private:
	// functions
	void ConformalMapOneVector(thrust::device_vector<TYPE> &, thrust::device_vector<TYPE> &); // TODO: Are the & here really working?

	// data containers
	thrust::device_vector<TYPE> fXValues, fCXValues;
	thrust::device_vector<TYPE> fYValues, fCYValues;
	thrust::device_vector<TYPE> fRValues, fCRValues;
	thrust::device_vector<TYPE> fAngles;
	std::vector<thrust::device_vector<TYPE> > fTransformedPoints;

	// needed values
	TYPE fMaxAngle; // (right border of) range of the angles to investigate, [0, maxAngle]
	TYPE fEveryXDegrees; // make a data point every x degrees

	// switches
	bool fUseIsochrones;
	bool fDoTiming;

	// for timing
	cudaEvent_t fEventStart, fEventStop;
	float fTimeConfMap;
	float fTimeGenAngles;
	float fTimeHoughTransform;

};
#endif //AHHOUGHTRANSFORMATION_H
