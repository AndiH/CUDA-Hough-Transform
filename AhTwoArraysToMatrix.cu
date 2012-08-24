#include "AhTwoArraysToMatrix.h"
#include "AhTranslatorFunction.h"

AhTwoArraysToMatrix::AhTwoArraysToMatrix()
{}

AhTwoArraysToMatrix::AhTwoArraysToMatrix(thrust::host_vector<double> xValues, thrust::host_vector<double> yValues, int nbinsx, double maxx, int nbinsy, double maxy)
: fXValues(xValues),
  fYValues(yValues),
  fNBinsX(nbinsx),
  fNBinsY(nbinsy),
  fMaxX(maxx),
  fMaxY(maxy)
{
	DoTranslations();
	
 	CalculateHistogram();
}


AhTwoArraysToMatrix::~AhTwoArraysToMatrix()
{}

 void AhTwoArraysToMatrix::SetXValues(thrust::host_vector<double> xValues)
 {
 	fXValues = xValues;
 }
 void AhTwoArraysToMatrix::SetYValues(thrust::host_vector<double> yValues)
 {
 	fYValues = yValues;
 }
 void AhTwoArraysToMatrix::SetNBinsX(int nbinsx)
 {
 	fNBinsX = nbinsx;
 }
 void AhTwoArraysToMatrix::SetNBinsY(int nbinsy)
 {
 	fNBinsY = nbinsy;
 }
 void AhTwoArraysToMatrix::SetMaxX(double maxx)
 {
 	fMaxX = maxx;
 }
 void AhTwoArraysToMatrix::SetMaxY(double maxy)
 {
 	fMaxY = maxy;
 }

void AhTwoArraysToMatrix::DoTranslations()
{

	thrust::device_vector<double> d_tempDummyX(fXValues.size());
 	thrust::device_vector<double> d_tempfXValues = fXValues;

 	double widthOfCellX = fNBinsX / fMaxX;
 	thrust::transform(d_tempfXValues.begin(), d_tempfXValues.end(), d_tempDummyX.begin(), AhTranslatorFunction(widthOfCellX));

 	fTranslatedXValues = d_tempDummyX;

	thrust::device_vector<double> d_tempDummyY(fYValues.size());
 	thrust::device_vector<double> d_tempfYValues = fYValues;

 	double widthOfCellY = fNBinsY / fMaxY;
 	thrust::transform(d_tempfYValues.begin(), d_tempfYValues.end(), d_tempDummyY.begin(), AhTranslatorFunction(widthOfCellY));
	
 	fTranslatedYValues = d_tempDummyY;
	
 	// 	fTranslatedXValues = TranslateValuesToMatrixCoordinates(tempVecX, fNBinsX, fMaxX); // ## TODO: IMPLEMENT FUNCTIONS FOR THIS
 	// 	fTranslatedYValues = TranslateValuesToMatrixCoordinates(fYValues, fNBinsY, fMaxY);
}

//thrust::device_vector<int> AhTwoArraysToMatrix::TranslateValuesToMatrixCoordinates (thrust::device_vector<double> values, int nOfPoints, double dimension) {
// 	thrust::device_vector<int> tempVec(values.size());
//
// 	double widthOfCell = nOfPoints / dimension;
//
// 	thrust::transform(values.begin(), values.end(), tempVec.begin(), AhTranslatorFunction(widthOfCell));
//
// 	return tempVec;
// }

 void AhTwoArraysToMatrix::CalculateHistogram()
 {
     thrust::device_vector<int> weightVector(fTranslatedXValues.size(), 1); // just dummy -- each cell should have weight of 1 // TODO: This, once, of course can be changed to support different weightes

     // allocate storage for unordered triplets
     cusp::array1d<int,   cusp::device_memory> I(fTranslatedXValues.begin(), fTranslatedXValues.end());  // row indices
     cusp::array1d<int,   cusp::device_memory> J(fTranslatedYValues.begin(), fTranslatedYValues.end());  // column indices
     cusp::array1d<float, cusp::device_memory> V(weightVector.begin(), weightVector.end());  // values

     // sort triplets by (i,j) index using two stable sorts (first by J, then by I)
     thrust::stable_sort_by_key(J.begin(), J.end(), thrust::make_zip_iterator(thrust::make_tuple(I.begin(), V.begin())));
     thrust::stable_sort_by_key(I.begin(), I.end(), thrust::make_zip_iterator(thrust::make_tuple(J.begin(), V.begin())));

     // compute unique number of nonzeros in the output
     int num_entries = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                             thrust::make_zip_iterator(thrust::make_tuple(I.end (),  J.end()))   - 1,
                                             thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
                                             int(0),
                                             thrust::plus<int>(),
                                             thrust::not_equal_to< thrust::tuple<int,int> >()) + 1;

     // allocate output matrix
     cusp::coo_matrix<int, float, cusp::device_memory> A(fNBinsX, fNBinsY, num_entries);

     // sum values with the same (i,j) index
     thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                           thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
                           V.begin(),
                           thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                           A.values.begin(),
                           thrust::equal_to< thrust::tuple<int,int> >(),
                           thrust::plus<float>());
     fCUSPMatrix = A;
 }

TMatrixD AhTwoArraysToMatrix::GetTMatrixD()
{
	TMatrixD myMatrix(fNBinsX, fNBinsY);
	for (int i = 0; i < fCUSPMatrix.num_entries; i++) {
		myMatrix[fCUSPMatrix.row_indices[i]][fCUSPMatrix.column_indices[i]] = fCUSPMatrix.values[i];
	}
	return myMatrix;
}
