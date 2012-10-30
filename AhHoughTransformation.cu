#include "AhHoughTransformation.h"

AhHoughTransformation::AhHoughTransformation()
{}

AhHoughTransformation::AhHoughTransformation(thrust::host_vector<double> xValues, thrust::host_vector<double> yValues)
: fXValues(xValues),
  fYValues(yValues)
{
	DoChangeContainerToTwoTuples();
	DoConformalMapping();
	DoHoughTransform();
}

AhHoughTransformation::~AhHoughTransformation()
{}

AhHoughTransformation::DoChangeContainerToTwoTuples() {
	//! change container from vec<x> and vec<y> to vec<tuple<x, y> >
	thrust::copy(
		thrust::make_zip_iterator(
			fXValues.begin(), 
			fYValues.begin()
		), 
		thrust::make_zip_iterator(
			fXValues.end(), 
			fYValues.end()
		), 
		fXYValues.begin()
	);
}

AhHoughTransformation::DoConformalMapping() {
	//! conformal mapping
	thrust::transform(
		fXYValues.begin(),
		fXYValues.end(),
		fXYValues.begin(), // in place copy - maybe not so full of sense? think about it
		my::confMap()
	);
}