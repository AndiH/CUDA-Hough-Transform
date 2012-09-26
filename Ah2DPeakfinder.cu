/*
 * Ah2DPeakfinder.cu
 *
 *  Created on: Sep 21, 2012
 *      Author: Andreas Herten
 */

#include "Ah2DPeakfinder.h"


Ah2DPeakfinder::Ah2DPeakfinder()
{}

Ah2DPeakfinder::Ah2DPeakfinder(thrust::device_vector<thrust::tuple<int, int, double> > data, int cutOff) :
		fCutOff(cutOff)
{
	// http://code.google.com/p/thrust/source/browse/examples/sorting_aos_vs_soa.cu?r=a878c02883f1e7f8fa7c851c109b4855eb57056c says a structure of arrays is always faster than an array of structures - so we have to first convert and then go to the central constructor

	// CONVERT

	thrust::device_vector<int> xData(data.size());
	thrust::device_vector<int> yData(data.size());
	thrust::device_vector<double> valueData(data.size());

	typedef thrust::tuple<int, int, double> default_tuple;

	thrust::transform(
			data.begin(),
			data.end(),
			thrust::make_constant_iterator(0),
			xData.begin(),
			returnElementOfTuple<default_tuple, int>()
	);
	thrust::transform(data.begin(), data.end(), thrust::make_constant_iterator(1), yData.begin(), returnElementOfTuple<default_tuple, int>());
	thrust::transform(data.begin(), data.end(), thrust::make_constant_iterator(2), valueData.begin(), returnElementOfTuple<default_tuple, double>());

//	std::cout << "DEBUG: " << std::endl;
//	for (int i = 0; i < xData.size(); i++) std::cout << "  xData[" << i << "] = " << xData[i] << std::endl;
//	for (int i = 0; i < yData.size(); i++) std::cout << "  yData[" << i << "] = " << yData[i] << std::endl;
//	for (int i = 0; i < valueData.size(); i++) std::cout << "  valueData[" << i << "] = " << valueData[i] << std::endl;

	fX = xData;
	fY = yData;
	fValues = valueData;

//	std::cout << "fValues.size(): " << fValues.size() << std::endl;

	DoEverything();

//	std::cout << "fValues.size(): " << fValues.size() << std::endl;
}

Ah2DPeakfinder::Ah2DPeakfinder(thrust::device_vector<int> x, thrust::device_vector<int> y, thrust::device_vector<double> values, int cutOff) :
		fX(x),
		fY(y),
		fValues(values),
		fCutOff(cutOff)
{
//	std::cout << "fValues.size(): " << fValues.size() << std::endl;
//	std::cout << "DEBUG2: " << std::endl;
//	for (int i = 0; i < fX.size(); i++) std::cout << "  fX[" << i << "] = " << fX[i] << std::endl;
//	for (int i = 0; i < fY.size(); i++) std::cout << "  fY[" << i << "] = " << fY[i] << std::endl;
//	for (int i = 0; i < fValues.size(); i++) std::cout << "  fValues[" << i << "] = " << fValues[i] << std::endl;

	DoEverything();

//	std::cout << "DEBUG4: " << std::endl;
//	for (int i = 0; i < fX.size(); i++) std::cout << "  fX[" << i << "] = " << fX[i] << std::endl;
//	for (int i = 0; i < fY.size(); i++) std::cout << "  fY[" << i << "] = " << fY[i] << std::endl;
//	for (int i = 0; i < fValues.size(); i++) std::cout << "  fValues[" << i << "] = " << fValues[i] << std::endl;
}


Ah2DPeakfinder::~Ah2DPeakfinder()
{}

void Ah2DPeakfinder::DoEverything() {
	// Called by different constructors
	if (fCutOff > 0) DoCutOff();
}

void Ah2DPeakfinder::DoCutOff()
{
//	int * new_end = thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(fX.begin(), fY.begin())), thrust::make_zip_iterator(thrust::make_tuple(fX.end(),fY.end())), fValues.begin(), removeElementsStrictSmallerThan(fCutOff)); // instead of zip_iterator make two single removes for x and y
//	thrust::device_vector<double>::iterator * new_end2 = thrust::remove_if(fValues.begin(), fValues.end(), removeElementsStrictSmallerThan(fCutOff));
	// this answer suggests, that remove_it is serial and not parallel https://groups.google.com/forum/?fromgroups=#!topic/thrust-users/UF7YIj_rM1E  -use copy_if instead

	// Count number of elements over threshold to create equally sized new vectors to be copied into
	int nElementsOverCut = thrust::count_if(fValues.begin(), fValues.end(), isGreaterThan(fCutOff-1)); // -1: isGreaterThan --> isGreatherEqualThan
//	std::cout << "Ah2DPeakFinder::DoCutOff() -- nElementsOverCut = " << nElementsOverCut << std::endl;
	// create new vectors
	thrust::device_vector<int> newX(nElementsOverCut);
	thrust::device_vector<int> newY(nElementsOverCut);
	thrust::device_vector<double> newValues(nElementsOverCut);

	// Condition-based copy content of old vectors into new ones
	thrust::copy_if(fX.begin(), fX.end(), fValues.begin(), newX.begin(), isGreaterThan(fCutOff-1));
	thrust::copy_if(fY.begin(), fY.end(), fValues.begin(), newY.begin(), isGreaterThan(fCutOff-1));
	thrust::copy_if(fValues.begin(), fValues.end(), newValues.begin(), isGreaterThan(fCutOff-1));

	// Resize old vector to have enough space for new stuff
	fX.resize(nElementsOverCut);
	fY.resize(nElementsOverCut);
	fValues.resize(nElementsOverCut);

	// Well, this should be clear
	fX = newX;
	fY = newY;
	thrust::copy(newValues.begin(), newValues.end(), fValues.begin());
//	fValues = newValues;

//	std::cout << "DEBUG3: " << std::endl;
//	for (int i = 0; i < fX.size(); i++) std::cout << "  fX[" << i << "] = " << fX[i] << std::endl;
//	for (int i = 0; i < fY.size(); i++) std::cout << "  fY[" << i << "] = " << fY[i] << std::endl;
//	for (int i = 0; i < fValues.size(); i++) std::cout << "  fValues[" << i << "] = " << fValues[i] << std::endl;


}
