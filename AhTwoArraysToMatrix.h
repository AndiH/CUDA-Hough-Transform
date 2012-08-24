#ifndef TWOARRAYSTOMATRIX_H
#define TWOARRAYSTOMATRIX_H

#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>

#include "TH2D.h"
#include "TMatrixD.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TCanvas.h"

//struct AhTranslatorFunction {
//	double widthOfCell;
//	AhTranslatorFunction(double _widthOfCell) : widthOfCell(_widthOfCell) {}
//
//	__device__ int operator() (const double &value) {
//		return value/widthOfCell; // maybe +1 has to be applied? check that and remember that
//		// TODO: if returnValue == 0
//		// TODO: negative values?
//	}
//};

class AhTwoArraysToMatrix
{
public:
	AhTwoArraysToMatrix();
	AhTwoArraysToMatrix(thrust::host_vector<double>, thrust::host_vector<double>, int, double, int, double);
	
	virtual ~AhTwoArraysToMatrix();
	
 	void SetXValues(thrust::host_vector<double>);
 	void SetYValues(thrust::host_vector<double>);
 	void SetNBinsX(int);
 	void SetNBinsY(int);
 	void SetMaxX(double);
 	void SetMaxY(double);
	
	thrust::device_vector<int> TranslateValuesToMatrixCoordinates (thrust::device_vector<double>, int, double);
	
	void DoTranslations();
 	void CalculateHistogram();
	
	cusp::coo_matrix<int, float, cusp::device_memory> GetCUSPMatrix() {return fCUSPMatrix;}
	TMatrixD GetTMatrixD();
	
private:
	thrust::host_vector<double> fXValues;
	thrust::host_vector<double> fYValues;
	thrust::device_vector<double> fTranslatedXValues;
	thrust::device_vector<double> fTranslatedYValues;
	int fNBinsX;
	int fNBinsY;
	double fMaxX;
	double fMaxY;
	cusp::coo_matrix<int, float, cusp::device_memory> fCUSPMatrix;
};

// SetXValues
// SetYValues
// SetNBinsX
// SetNBinsY
// SetMaxX
// SetMaxY
// CalculateValuesToMatrixRange
// CalculateHistogram
// GetTMatrixD
// GetCUSPMatrix

#endif //TWOARRAYSTOMATRIX_H
