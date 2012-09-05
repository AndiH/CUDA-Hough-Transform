#include "AhTwoArraysToMatrix.h"
#include "AhTranslatorFunction.h"

/* TODO
 * 
 * check bounds: are all values in range of xylow xyup?
 * find out why old idea of functions doesnt work
 * tstopwatches
 * 
 * deliver two functions:
 * 	give translated value
 * 	give array translated VALUES
 * 
 * peak finder
 */


AhTwoArraysToMatrix::AhTwoArraysToMatrix()
{}

AhTwoArraysToMatrix::AhTwoArraysToMatrix(thrust::host_vector<double> xValues, thrust::host_vector<double> yValues, int nbinsx, double xlow, double xup, int nbinsy, double ylow, double yup)
: fXValues(xValues),
  fYValues(yValues),
  fNBinsX(nbinsx),
  fNBinsY(nbinsy),
  fXlow(xlow),
  fXup(xup),
  fYlow(ylow),
  fYup(yup)
{
	fXStepWidth = (fXup - fXlow)/(fNBinsX);
	fYStepWidth = (fYup - fYlow)/(fNBinsY);
	
	DoTranslations();
	
 	CalculateHistogram();
}


AhTwoArraysToMatrix::~AhTwoArraysToMatrix()
{}

void AhTwoArraysToMatrix::DoTranslations()
{

	thrust::device_vector<double> d_tempDummyX(fXValues.size());
 	thrust::device_vector<double> d_tempfXValues = fXValues;

 	thrust::transform(d_tempfXValues.begin(), d_tempfXValues.end(), d_tempDummyX.begin(), AhTranslatorFunction(fXStepWidth, fXlow));
	
 	fTranslatedXValues = d_tempDummyX;

	thrust::device_vector<double> d_tempDummyY(fYValues.size());
 	thrust::device_vector<double> d_tempfYValues = fYValues;

 	thrust::transform(d_tempfYValues.begin(), d_tempfYValues.end(), d_tempDummyY.begin(), AhTranslatorFunction(fYStepWidth, fYlow));
	
 	fTranslatedYValues = d_tempDummyY;
	
 	// 	fTranslatedXValues = TranslateValuesToMatrixCoordinates(tempVecX, fNBinsX, fMaxX); // ## TODO: IMPLEMENT FUNCTIONS FOR THIS
 	// 	fTranslatedYValues = TranslateValuesToMatrixCoordinates(fYValues, fNBinsY, fMaxY);
	
	// ### DEBUG OUTPUT
// 	for (int i = 0; i < fTranslatedXValues.size(); i++) {
// 		std::cout << "(x,y)_" << i << " = (" << fXValues[i] << "," << fYValues[i] << ") --> (" << fTranslatedXValues[i] << "," << fTranslatedYValues[i] << ")" << std::endl;	
// 	}
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
	TMatrixD myMatrix(fNBinsY, fNBinsX);
	std::cout << myMatrix.GetRowUpb() << " "<< myMatrix.GetColUpb() << std::endl;
	std::cout << fCUSPMatrix.num_rows << " " << fCUSPMatrix.num_cols << std::endl;
	for (int i = 0; i < fCUSPMatrix.num_entries; i++) {
		int column_index = fCUSPMatrix.column_indices[i];
		int row_index = fCUSPMatrix.row_indices[i];
		int value = fCUSPMatrix.values[i];

		std::cout << i << " - columindex = " << column_index << ", rowindex = " <<  row_index << ", value = " << value << std::endl;

		if (column_index <= fNBinsY && column_index >= 0 && row_index <= fNBinsX && row_index >= 0) {
			myMatrix[column_index][row_index] = value;
		} else {
			std::cout << "matrix element not filled" << std::endl;
		}
	}
	return myMatrix;
}

TH2D * AhTwoArraysToMatrix::GetHistogram()
{
	// Gives TH2D Histogram with proper, re-translated boundaries
	TH2D * tempHisto = new TH2D(GetTMatrixD());
	tempHisto->GetXaxis()->SetLimits(fXlow, fXlow + fNBinsX * fXStepWidth);
	tempHisto->GetYaxis()->SetLimits(fYlow, fYlow + fNBinsY * fYStepWidth);
	
	return tempHisto;
}
