#ifndef TWOARRAYSTOMATRIX_H
#define TWOARRAYSTOMATRIX_H

#ifdef USE_DOUBLES
	typedef double TYPE;
#else
	typedef float TYPE;
#endif

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
	AhTwoArraysToMatrix(thrust::host_vector<TYPE>, thrust::host_vector<TYPE>, int, TYPE, TYPE, int, TYPE, TYPE, bool doTiming = false, bool doBoundaryCheck = false);
	
	virtual ~AhTwoArraysToMatrix();
	
	void SetXValues(thrust::host_vector<TYPE> _xvalues) { fXValues = _xvalues; };
	void SetYValues(thrust::host_vector<TYPE> _yvalues) { fYValues = _yvalues; };
	void SetNBinsX(int _nbinsx) { fNBinsX = _nbinsx; };
	void SetNBinsY(int _nbinsy) { fNBinsY = _nbinsy; };
	void SetXlow(TYPE _xlow) { fXlow = _xlow; };
	void SetXup(TYPE _xup) { fXup = _xup; };
	void SetYlow(TYPE _ylow) { fYlow = _ylow; };
	void SetYup(TYPE _yup) { fYup = _yup; };
	
	thrust::device_vector<int> TranslateValuesToMatrixCoordinates (const thrust::device_vector<TYPE>&, TYPE, TYPE);
	thrust::device_vector<TYPE> RetranslateValuesFromMatrixCoordinates (const thrust::device_vector<int> &values, TYPE inverseStepWidth, TYPE lowValue);
	
 	bool DoBoundaryCheck();
 	void DoTranslations();
	void DoRetranslations();
 	void CalculateHistogram();
	
	cusp::coo_matrix<int, TYPE, cusp::device_memory> GetCUSPMatrix() {return fCUSPMatrix;}
	TMatrixD GetTMatrixD();
	TH2D * GetHistogram();
	int GetNBinsX() { return fNBinsX; };
	int GetNBinsY() { return fNBinsY; };
	TYPE GetXlow() { return fXlow; };
	TYPE GetXup() { return fXup; };
	TYPE GetYlow() { return fYlow; };
	TYPE GetYup() { return fYup; };
	thrust::device_vector<int> GetPlainXValues();
	thrust::device_vector<int> GetPlainYValues();
	thrust::device_vector<TYPE> GetMultiplicities();
	thrust::device_vector<TYPE> GetBinContent() { return GetMultiplicities(); };
	thrust::device_vector<thrust::tuple<int, int, TYPE> > GetPlainMatrixValues();
	thrust::device_vector<thrust::tuple<TYPE, TYPE, TYPE> > GetRetranslatedMatrixValues();

	TYPE GetTimeTranslateValues() { return fTimeTranslateValues; }; // in MILLI seconds, see http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDART__EVENT_g14c387cc57ce2e328f6669854e6020a5.html
	TYPE GetTimeHistSort() { return fTimeHistSort; }; // in MILLI seconds
	TYPE GetTimeHistSum() { return fTimeHistSum; }; // in MILLI seconds
	TYPE GetTimeCreateTMatrixD() { return fSwCreateTMatrixD->CpuTime(); }; // in SECONDS
	TYPE GetTimeCreateTH2D() { return fSwCreateTH2D->CpuTime(); }; // in SECONDS

protected:
	// Not used, just kept here for ... well... I don't know for what
	TStopwatch * GetSwTranslateValues() { return fSwTranslateValues; }; // breaks if fDoTiming is not set to true
	TStopwatch * GetSwHistSort() { return fSwHistSort; }; // breaks if fDoTiming is not set to true
	TStopwatch * GetSwHistSum() { return fSwHistSum; }; // breaks if fDoTiming is not set to true
	TStopwatch * GetSwCreateTMatrixD() { return fSwCreateTMatrixD; }; // breaks if fDoTiming is not set to true
	TStopwatch * GetSwCreateTH2D() { return fSwCreateTH2D; }; // breaks if fDoTiming is not set to true
	
private:
	template <class T>
	thrust::device_vector<T> CuspVectorToDeviceVector(const cusp::array1d<T, cusp::device_memory> &cuspvec);
	
	bool fDoTiming;
	thrust::device_vector<TYPE> fXValues;
	thrust::device_vector<TYPE> fYValues;
	thrust::device_vector<int> fTranslatedXValues;
	thrust::device_vector<int> fTranslatedYValues;
	thrust::device_vector<TYPE> fRetranslatedXValues;
	thrust::device_vector<TYPE> fRetranslatedYValues;
	int fNBinsX;
	int fNBinsY;
	TYPE fXlow;
	TYPE fXup;
	TYPE fYlow;
	TYPE fYup;
	cusp::coo_matrix<int, TYPE, cusp::device_memory> fCUSPMatrix;
	
	TYPE fXStepWidth;
	TYPE fYStepWidth;
	
	bool fReTranslationHasBeenDone;
	
	// Stuff for timing purposes
	TStopwatch * fSwTranslateValues;
	TStopwatch * fSwHistSort;
	TStopwatch * fSwHistSum;
	TStopwatch * fSwCreateTMatrixD;
	TStopwatch * fSwCreateTH2D;

	cudaEvent_t start, stop;
	float fTimeTranslateValues;
	float fTimeHistSort;
	float fTimeHistSum;
	
	cusp::array1d<int, cusp::device_memory> fI;
	cusp::array1d<int, cusp::device_memory> fJ;
	cusp::array1d<TYPE, cusp::device_memory> fV;
};

#endif //TWOARRAYSTOMATRIX_H
