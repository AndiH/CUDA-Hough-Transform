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
	AhTwoArraysToMatrix(thrust::host_vector<double>, thrust::host_vector<double>, int, double, double, int, double, double);
	
	virtual ~AhTwoArraysToMatrix();
	
	void SetXValues(thrust::host_vector<double> _xvalues) { fXValues = _xvalues; };
	void SetYValues(thrust::host_vector<double> _yvalues) { fYValues = _yvalues; };
	void SetNBinsX(int _nbinsx) { fNBinsX = _nbinsx; };
	void SetNBinsY(int _nbinsy) { fNBinsY = _nbinsy; };
	void SetXlow(double _xlow) { fXlow = _xlow; };
	void SetXup(double _xup) { fXup = _xup; };
	void SetYlow(double _ylow) { fYlow = _ylow; };
	void SetYup(double _yup) { fYup = _yup; };
	
	thrust::device_vector<int> TranslateValuesToMatrixCoordinates (thrust::device_vector<double>, int, double);
	
	void DoTranslations();
 	void CalculateHistogram();
	
	cusp::coo_matrix<int, float, cusp::device_memory> GetCUSPMatrix() {return fCUSPMatrix;}
	TMatrixD GetTMatrixD();
	TH2D * GetHistogram();
	int GetNBinsX() { return fNBinsX; };
	int GetNBinsY() { return fNBinsY; };
	double GetXlow() { return fXlow; };
	double GetXup() { return fXup; };
	double GetYlow() { return fYlow; };
	double GetYup() { return fYup; };
	
private:
	thrust::host_vector<double> fXValues;
	thrust::host_vector<double> fYValues;
	thrust::device_vector<double> fTranslatedXValues;
	thrust::device_vector<double> fTranslatedYValues;
	int fNBinsX;
	int fNBinsY;
	double fXlow;
	double fXup;
	double fYlow;
	double fYup;
	cusp::coo_matrix<int, float, cusp::device_memory> fCUSPMatrix;
	
	double fXStepWidth;
	double fYStepWidth;
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