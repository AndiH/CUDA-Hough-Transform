#ifndef TWOARRAYSTOMATRIX_H
#define TWOARRAYSTOMATRIX_H

#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>

#include "TH2D.h"
#include "TMatrixD.h"

#include "TStopwatch.h"

class AhTwoArraysToMatrix
{
public:
	AhTwoArraysToMatrix();
	AhTwoArraysToMatrix(thrust::host_vector<double>, thrust::host_vector<double>, int, double, double, int, double, double, bool doTiming = false, bool doBoundaryCheck = false);
	
	virtual ~AhTwoArraysToMatrix();
	
	void SetXValues(thrust::host_vector<double> _xvalues) { fXValues = _xvalues; };
	void SetYValues(thrust::host_vector<double> _yvalues) { fYValues = _yvalues; };
	void SetNBinsX(int _nbinsx) { fNBinsX = _nbinsx; };
	void SetNBinsY(int _nbinsy) { fNBinsY = _nbinsy; };
	void SetXlow(double _xlow) { fXlow = _xlow; };
	void SetXup(double _xup) { fXup = _xup; };
	void SetYlow(double _ylow) { fYlow = _ylow; };
	void SetYup(double _yup) { fYup = _yup; };
	
	thrust::device_vector<int> TranslateValuesToMatrixCoordinates (const thrust::device_vector<double>&, double, double);
	
 	bool DoBoundaryCheck();
 	void DoTranslations();
 	void CalculateHistogram();
	
	cusp::coo_matrix<int, double, cusp::device_memory> GetCUSPMatrix() {return fCUSPMatrix;}
	TMatrixD GetTMatrixD();
	TH2D * GetHistogram();
	int GetNBinsX() { return fNBinsX; };
	int GetNBinsY() { return fNBinsY; };
	double GetXlow() { return fXlow; };
	double GetXup() { return fXup; };
	double GetYlow() { return fYlow; };
	double GetYup() { return fYup; };
	thrust::device_vector<int> GetPlainXValues();
	thrust::device_vector<int> GetPlainYValues();
	thrust::device_vector<double> GetMultiplicities();
	thrust::device_vector<double> GetBinContent() { return GetMultiplicities(); };
	
	TStopwatch * GetSwTranslateValues() { return fSwTranslateValues; }; // breaks if fDoTiming is not set to true
	TStopwatch * GetSwHistSort() { return fSwHistSort; }; // breaks if fDoTiming is not set to true
	TStopwatch * GetSwHistSum() { return fSwHistSum; }; // breaks if fDoTiming is not set to true
	TStopwatch * GetSwCreateTMatrixD() { return fSwCreateTMatrixD; }; // breaks if fDoTiming is not set to true
	TStopwatch * GetSwCreateTH2D() { return fSwCreateTH2D; }; // breaks if fDoTiming is not set to true
	
private:
	template <class T>
	thrust::device_vector<T> CuspVectorToDeviceVector(const cusp::array1d<T, cusp::device_memory> &cuspvec);
	
	bool fDoTiming;
	thrust::host_vector<double> fXValues;
	thrust::host_vector<double> fYValues;
	thrust::device_vector<int> fTranslatedXValues;
	thrust::device_vector<int> fTranslatedYValues;
	int fNBinsX;
	int fNBinsY;
	double fXlow;
	double fXup;
	double fYlow;
	double fYup;
	cusp::coo_matrix<int, double, cusp::device_memory> fCUSPMatrix;
	
	double fXStepWidth;
	double fYStepWidth;
	
	TStopwatch * fSwTranslateValues;
	TStopwatch * fSwHistSort;
	TStopwatch * fSwHistSum;
	TStopwatch * fSwCreateTMatrixD;
	TStopwatch * fSwCreateTH2D;
	
	cusp::array1d<int, cusp::device_memory> fI;
	cusp::array1d<int, cusp::device_memory> fJ;
	cusp::array1d<double, cusp::device_memory> fV;
};

#endif //TWOARRAYSTOMATRIX_H
