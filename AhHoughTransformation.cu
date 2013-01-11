#include "AhHoughTransformation.h"

AhHoughTransformation::AhHoughTransformation()
{}

AhHoughTransformation::AhHoughTransformation(thrust::host_vector<double> xValues, thrust::host_vector<double> yValues, double maxAngle, double everyXDegrees)
: fXValues(xValues),
  fYValues(yValues),
  fMaxAngle(maxAngle),
  fEveryXDegrees(everyXDegrees),
  fUseIsochrones(false)
{
	DoEverything();
}

AhHoughTransformation::AhHoughTransformation(thrust::host_vector<double> xValues, thrust::host_vector<double> yValues, thrust::host_vector<double> rValues, double maxAngle, double everyXDegrees)
: fXValues(xValues),
  fYValues(yValues),
  fRValues(rValues),
  fMaxAngle(maxAngle),
  fEveryXDegrees(everyXDegrees),
  fUseIsochrones(true)
{
	// std::cout << "fXValues.size() = " << fXValues.size() << ", fYValues.size()" << fYValues.size() << "fRValues.size() = " << fRValues.size() << ", fMaxAngle = " << fMaxAngle << ", fEveryXDegrees = " << fEveryXDegrees << ", fUseIsochrones = " << fUseIsochrones << std::endl;
	DoEverything();
}

AhHoughTransformation::~AhHoughTransformation()
{}

void AhHoughTransformation::DoEverything() {
	if (false == fUseIsochrones) DoChangeContainerToTwoTuples();
	// OLD_DoChangeContainerToTwoTuples();
	DoConformalMapping();
	DoGenerateAngles();
	DoHoughTransform();
}

void AhHoughTransformation::OLD_DoChangeContainerToTwoTuples() {
	thrust::host_vector<double> h_xValues = fXValues;
	thrust::host_vector<double> h_yValues = fYValues;
	thrust::host_vector<thrust::tuple<double, double> > h_xYValues(h_xValues.size());

	for (int i = 0; i < h_xValues.size(); i++) {
		h_xYValues.push_back(thrust::make_tuple(h_xValues[i], fYValues[i]));
	}
	fXYValues = h_xYValues; //!< copy onto device
}
void AhHoughTransformation::DoChangeContainerToTwoTuples() {
	//! change container from vec<x> and vec<y> to vec<tuple<x, y> >
	fXYValues.resize(fXValues.size());

	thrust::copy(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				fXValues.begin(), 
				fYValues.begin()
			)
		), 
		thrust::make_zip_iterator(
			thrust::make_tuple(
				fXValues.end(), 
				fYValues.end()
			)
		), 
		fXYValues.begin()
	);
	/* TODO
	 * is this at all necessary to use this stuff here - changing containers?
	 * performance(AoS) < performance(SoA)
	 */
}

void AhHoughTransformation::DoConformalMapping() {
	//! conformal mapping
	if (true == fUseIsochrones) {
		fCXValues.resize(fXValues.size());
		fCYValues.resize(fYValues.size());
		fCRValues.resize(fRValues.size());

		ConformalMapOneVector(fXValues, fCXValues);
		ConformalMapOneVector(fYValues, fCYValues);
		ConformalMapOneVector(fRValues, fCRValues);
	} else {
		thrust::transform(
			fXYValues.begin(),
			fXYValues.end(),
			fXYValues.begin(), // in place copy (== output vector = input vector) - maybe not so full of sense? think about it
			my::confMap()
		);
	}
}

void AhHoughTransformation::ConformalMapOneVector(thrust::device_vector<double> &originalData, thrust::device_vector<double> &mappedData) {
	thrust::transform(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				originalData.begin(),
				fXValues.begin(),
				fYValues.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				originalData.end(),
				fXValues.end(),
				fYValues.end()
			)
		),
		mappedData.begin(),
		my::confMap()
	);
	std::cout << "been here!" << std::endl;
}

void AhHoughTransformation::DoGenerateAngles() {
	fAngles.resize(fMaxAngle/fEveryXDegrees); //!< Resize angle vector to match the actual size
	//! Fill angle vector with angles in appropriate stepping
	thrust::sequence(
		fAngles.begin(), 
		fAngles.end(), 
		0., 
		fEveryXDegrees
	);
}

void AhHoughTransformation::DoHoughTransform() {
	/** Attention
	* For every (x*,y*) point a hough transform is done!
	* While this hough transform itself is done in parallel, all hough transforms in whole are done serial, one by one after an other.
	* This might be a point of huge improvements
	*/
	for (int iDataPoints = 0; iDataPoints < fXValues.size(); iDataPoints++) {
		thrust::device_vector<double> d_tempData(fAngles.size()); //!< Temp vector which is being filled and then pushed back to the main return vector. For every angle point theres a data point, so that's the size of it

		if (false == fUseIsochrones) {
			thrust::constant_iterator<thrust::tuple<double, double> > currentData(fXYValues[iDataPoints]); //!< create constant iterator for the conf mapped data 2-tuples

			//! following transformation uses the operator of htransf to run over all elements
			//!   elements being a iterator from angles.start to angles.end with each time the constant iterator with the 	conf mapped 2-tuple
			//!   the result of the calculation is written in to the d_tempData vector
			thrust::transform(
				thrust::make_zip_iterator(
					thrust::make_tuple(
						fAngles.begin(),
						currentData
					)
				),
				thrust::make_zip_iterator(
					thrust::make_tuple(
						fAngles.end(),
						currentData
					)
				),
				d_tempData.begin(),
				my::htransf()
			);
		} else { // (true == fUseIsochrones)
			thrust::constant_iterator<thrust::tuple<double, double, double> > currentData(thrust::make_tuple(fCXValues[iDataPoints], fCYValues[iDataPoints], fCRValues[iDataPoints]));
			//! following transformation uses the operator of htransf to run over all elements
			//!   elements being a iterator from angles.start to angles.end with each time the constant iterator with the 	conf mapped 2-tuple
			//!   the result of the calculation is written in to the d_tempData vector
			thrust::transform(
				thrust::make_zip_iterator(
					thrust::make_tuple(
						fAngles.begin(),
						currentData
					)
				),
				thrust::make_zip_iterator(
					thrust::make_tuple(
						fAngles.end(),
						currentData
					)
				),
				d_tempData.begin(),
				my::htransf()
			);
		}
		
		
		fTransformedPoints.push_back(d_tempData); //!< push it back to the main data stack vector

	}
}
