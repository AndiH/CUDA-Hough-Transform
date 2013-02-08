#include "AhHoughTransformation.h"

AhHoughTransformation::AhHoughTransformation()
{}

AhHoughTransformation::AhHoughTransformation(thrust::host_vector<TYPE> xValues, thrust::host_vector<TYPE> yValues, TYPE maxAngle, TYPE everyXDegrees, bool doTiming)
: fXValues(xValues),
  fYValues(yValues),
  fMaxAngle(maxAngle),
  fEveryXDegrees(everyXDegrees),
  fDoTiming(doTiming),
  fUseIsochrones(false)
{
	DoEverything();
}

AhHoughTransformation::AhHoughTransformation(thrust::host_vector<TYPE> xValues, thrust::host_vector<TYPE> yValues, thrust::host_vector<TYPE> rValues, TYPE maxAngle, TYPE everyXDegrees, bool doTiming)
: fXValues(xValues),
  fYValues(yValues),
  fRValues(rValues),
  fMaxAngle(maxAngle),
  fEveryXDegrees(everyXDegrees),
  fDoTiming(doTiming),
  fUseIsochrones(true)
{
	// std::cout << "fXValues.size() = " << fXValues.size() << ", fYValues.size()" << fYValues.size() << "fRValues.size() = " << fRValues.size() << ", fMaxAngle = " << fMaxAngle << ", fEveryXDegrees = " << fEveryXDegrees << ", fUseIsochrones = " << fUseIsochrones << std::endl;
	DoEverything();
}

AhHoughTransformation::~AhHoughTransformation()
{}

void AhHoughTransformation::EventTiming_start() {
	cudaEventCreate(&fEventStart);
	cudaEventCreate(&fEventStop);
	cudaThreadSynchronize();
	cudaEventRecord(fEventStart, 0);
}

float AhHoughTransformation::EventTiming_stop() {
	float currentTime;
	cudaEventRecord(fEventStop, 0);
	cudaEventSynchronize(fEventStop);
	cudaEventElapsedTime(&currentTime, fEventStart, fEventStop);
	cudaEventDestroy(fEventStart);
	cudaEventDestroy(fEventStop);

	return currentTime;
}

void AhHoughTransformation::DoEverything() {
	if (true == fDoTiming) EventTiming_start();
	DoConformalMapping();
	if (true == fDoTiming) fTimeConfMap = EventTiming_stop();

	if (true == fDoTiming) EventTiming_start();
	DoGenerateAngles();
	if (true == fDoTiming) fTimeGenAngles = EventTiming_stop();

	if (true == fDoTiming) EventTiming_start();
	DoHoughTransform();
	if (true == fDoTiming) fTimeHoughTransform = EventTiming_stop();
}

void AhHoughTransformation::DoConformalMapping() {
	//! conformal mapping

	fCXValues.resize(fXValues.size());
	fCYValues.resize(fYValues.size());

	ConformalMapOneVector(fXValues, fCXValues);
	ConformalMapOneVector(fYValues, fCYValues);

	if (true == fUseIsochrones) {
		fCRValues.resize(fRValues.size());

		ConformalMapOneVector(fRValues, fCRValues);
	}
}

void AhHoughTransformation::ConformalMapOneVector(thrust::device_vector<TYPE> &originalData, thrust::device_vector<TYPE> &mappedData) {
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
	std::cout << "--DEBUG-- AhHoughTransformation::ConformalMapOneVector ### Been here!" << std::endl;
}

void AhHoughTransformation::DoGenerateAngles() {
	fAngles.resize(fMaxAngle/fEveryXDegrees); //!< Resize angle vector to match the actual size
	//! Fill angle vector with angles in appropriate stepping
	TYPE zero = 0.;
	thrust::sequence(
		fAngles.begin(), 
		fAngles.end(), 
		zero, 
		fEveryXDegrees
	);
}

template <class T>
void AhHoughTransformation::DoHoughTransformOnePoint(thrust::constant_iterator<T> data, thrust::device_vector<TYPE> &d_tempData) {
	thrust::transform(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				fAngles.begin(),
				data
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				fAngles.end(),
				data
			)
		),
		d_tempData.begin(),
		my::htransf()
	);
}

void AhHoughTransformation::DoHoughTransform() {
	/** Attention
	* For every (x*,y*) point a hough transform is done!
	* While this hough transform itself is done in parallel, all hough transforms in whole are done serial, one by one after an other.
	* This might be a point of huge improvements
	*/
	for (int iDataPoints = 0; iDataPoints < fXValues.size(); iDataPoints++) {
		thrust::device_vector<TYPE> d_tempData(fAngles.size()); //!< Temp vector which is being filled and then pushed back to the main return vector. For every angle point theres a data point, so that's the size of it

		if (false == fUseIsochrones) {
			thrust::constant_iterator<thrust::tuple<TYPE, TYPE> > currentData(thrust::make_tuple(fCXValues[iDataPoints], fCYValues[iDataPoints])); //!< create constant iterator for the conf mapped data 2-tuples

			//! following transformation uses the operator of htransf to run over all elements
			//!   elements being a iterator from angles.start to angles.end with each time the constant iterator with the 	conf mapped 2-tuple
			//!   the result of the calculation is written in to the d_tempData vector
			DoHoughTransformOnePoint<thrust::tuple<TYPE, TYPE> >(currentData, d_tempData);
			// thrust::transform(
			// 	thrust::make_zip_iterator(
			// 		thrust::make_tuple(
			// 			fAngles.begin(),
			// 			currentData
			// 		)
			// 	),
			// 	thrust::make_zip_iterator(
			// 		thrust::make_tuple(
			// 			fAngles.end(),
			// 			currentData
			// 		)
			// 	),
			// 	d_tempData.begin(),
			// 	my::htransf()
			// );
		} else { // (true == fUseIsochrones)
			thrust::constant_iterator<thrust::tuple<TYPE, TYPE, TYPE> > currentData(thrust::make_tuple(fCXValues[iDataPoints], fCYValues[iDataPoints], fCRValues[iDataPoints]));
			//! following transformation uses the operator of htransf to run over all elements
			//!   elements being a iterator from angles.start to angles.end with each time the constant iterator with the 	conf mapped 2-tuple
			//!   the result of the calculation is written in to the d_tempData vector
			DoHoughTransformOnePoint<thrust::tuple<TYPE, TYPE, TYPE> >(currentData, d_tempData);
			// thrust::transform(
			// 	thrust::make_zip_iterator(
			// 		thrust::make_tuple(
			// 			fAngles.begin(),
			// 			currentData
			// 		)
			// 	),
			// 	thrust::make_zip_iterator(
			// 		thrust::make_tuple(
			// 			fAngles.end(),
			// 			currentData
			// 		)
			// 	),
			// 	d_tempData.begin(),
			// 	my::htransf()
			// );
		}
		
		
		fTransformedPoints.push_back(d_tempData); //!< push it back to the main data stack vector

	}
}
