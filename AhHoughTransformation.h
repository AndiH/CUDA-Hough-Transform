#ifndef AHHOUGHTRANSFORMATION_H
#define AHHOUGHTRANSFORMATION_H

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
		__device__ thrust::tuple<double, double> operator() (thrust::tuple<double, double> &data) {
			double x = thrust::get<0>(data);
			double y = thrust::get<1>(data);
			
			double x2y2 = x*x+y*y;
		//	printf("x2y2 = %f\n", x2y2); // ### DEBUG
			
			return thrust::make_tuple(x/x2y2, -y/x2y2);
		}

		// Testing code for isochrones: 
		__device__ double operator() (thrust::tuple<double&, double&, double&> &data) {
			double value = thrust::get<0>(data);
			double x = thrust::get<1>(data);
			double y = thrust::get<2>(data);

			double x2y2 = x*x+y*y;
			
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
		__device__ double houghTransfFunction(double alpha, double x, double y) {
			return (cos(alpha*1.74532925199432955e-02) * x + sin(alpha*1.74532925199432955e-02) * y);
		}

		/**
		* @param data A tuple of (an angle and a tuple of (to be hough transformed x and y coordinates))
		* @return A hough transformed point corresponding to a certain angle
		*/
		__device__ double operator() (thrust::tuple<double &, thrust::tuple<double, double> > data) {
			double alpha = thrust::get<0>(data);
			thrust::tuple<double, double> xy = thrust::get<1>(data);
			double x = thrust::get<0>(xy);	

			double y = thrust::get<1>(xy);

	// 		printf("alpha * x + alpha * y = %f * %f + %f * %f\n", alpha, x, alpha, y);

			return houghTransfFunction(alpha, x, y);
		}

		// Testing code for isochrones:
		__device__ double operator() (thrust::tuple<double &, thrust::tuple<double, double, double> > data) {
			double alpha = thrust::get<0>(data);
			thrust::tuple<double, double, double> xyr = thrust::get<1>(data);
			double x = thrust::get<0>(xyr);
			double y = thrust::get<1>(xyr);
			double r = thrust::get<2>(xyr);

			return (houghTransfFunction(alpha, x, y) + r);
		}
	};
}

class AhHoughTransformation
{
public:
	AhHoughTransformation();
	AhHoughTransformation(thrust::host_vector<double>, thrust::host_vector<double>, double maxAngle = 180., double everyXDegrees = 30.);
	AhHoughTransformation(thrust::host_vector<double>, thrust::host_vector<double>, thrust::host_vector<double>, double maxAngle = 360., double everyXDegrees = 30.);
	
	virtual ~AhHoughTransformation();

	thrust::device_vector<double> GetAngles() { return fAngles; };
	std::vector<thrust::device_vector<double> > GetVectorOfTransformedPoints() {return fTransformedPoints; };
protected:
	void DoEverything();
	void DoConformalMapping();
	void DoGenerateAngles();
	void DoHoughTransform();
private:
	// functions
	void ConformalMapOneVector(thrust::device_vector<double> &, thrust::device_vector<double> &); // TODO: Are the & here really working?

	// data containers
	thrust::device_vector<double> fXValues, fCXValues;
	thrust::device_vector<double> fYValues, fCYValues;
	thrust::device_vector<double> fRValues, fCRValues;
	thrust::device_vector<double> fAngles;
	std::vector<thrust::device_vector<double> > fTransformedPoints;

	// needed values
	double fMaxAngle; // (right border of) range of the angles to investigate, [0, maxAngle]
	double fEveryXDegrees; // make a data point every x degrees

	// switches
	bool fUseIsochrones;

};
#endif //AHHOUGHTRANSFORMATION_H
