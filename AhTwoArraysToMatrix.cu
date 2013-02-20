#include "AhTwoArraysToMatrix.h"
#include "AhTranslatorFunction.h"

/* TODO
 * 
 * find out why old idea of functions doesnt work
 * 
 * deliver two functions:
 * 	give translated value
 * 	give array translated VALUES
 * 
 * peak finder
 */


AhTwoArraysToMatrix::AhTwoArraysToMatrix()
{}

AhTwoArraysToMatrix::AhTwoArraysToMatrix(thrust::host_vector<TYPE> xValues, thrust::host_vector<TYPE> yValues, int nbinsx, TYPE xlow, TYPE xup, int nbinsy, TYPE ylow, TYPE yup, bool doTiming, bool doBoundaryCheck)
: fXValues(xValues),
  fYValues(yValues),
  fNBinsX(nbinsx),
  fNBinsY(nbinsy),
  fXlow(xlow),
  fXup(xup),
  fYlow(ylow),
  fYup(yup),
  fDoTiming(doTiming),
  fReTranslationHasBeenDone(false)
{
	bool dontbreak = true;

	fXStepWidth = (fXup - fXlow)/(fNBinsX);
	fYStepWidth = (fYup - fYlow)/(fNBinsY);
	
	if (doBoundaryCheck == true) dontbreak = DoBoundaryCheck();

	if (dontbreak == true) {
		DoTranslations();

		CalculateHistogram();
	} else {
		std::cout << "Did nothing due to some error." << std::endl;
	}
	
}


AhTwoArraysToMatrix::~AhTwoArraysToMatrix()
{}

bool AhTwoArraysToMatrix::DoBoundaryCheck() {
	bool everythingIsOk = true;

	TYPE minimumX = *(thrust::min_element(fXValues.begin(), fXValues.end()));
	TYPE maximumX = *(thrust::max_element(fXValues.begin(), fXValues.end()));
	TYPE minimumY = *(thrust::min_element(fYValues.begin(), fYValues.end()));
	TYPE maximumY = *(thrust::max_element(fYValues.begin(), fYValues.end()));

	if (minimumX < fXlow || maximumX > fXup || minimumY < fYlow || maximumY > fYup) {
		std::cout << "Values are out of boundaries! Breaking!" << std::endl;
		std::cout << "  " << minimumX << " !>= " << fXlow << ",   " << maximumX << "!<=" << fXup << std::endl;
		std::cout << "  " << minimumY << " !>= " << fYlow << ",   " << maximumY << "!<=" << fYup << std::endl;
		everythingIsOk = false;
	}
	
	return everythingIsOk;
}

void AhTwoArraysToMatrix::DoTranslations()
{
	if (true == fDoTiming) {
		fSwTranslateValues = new TStopwatch(); // Old version with TStopwatch, kept for backwards compability
		
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaThreadSynchronize(); // make sure everything is ready
		cudaEventRecord(start, 0); // create start point
	}
	
	fTranslatedXValues = TranslateValuesToMatrixCoordinates(fXValues, 1/fXStepWidth, fXlow);
	fTranslatedYValues = TranslateValuesToMatrixCoordinates(fYValues, 1/fYStepWidth, fYlow);
	
	if (true == fDoTiming) {
		fSwTranslateValues->Stop(); // Old
		cudaThreadSynchronize();
		
		cudaEventRecord(stop, 0); // create stop point
		cudaEventSynchronize(stop); // wait for stop to finish
		cudaEventElapsedTime(&fTimeTranslateValues, start, stop); // in mili seconds
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	// ### DEBUG OUTPUT
// 	for (int i = 0; i < fTranslatedXValues.size(); i++) {
// 		std::cout << "(x,y)_" << i << " = (" << fXValues[i] << "," << fYValues[i] << ") --> (" << fTranslatedXValues[i] << "," << fTranslatedYValues[i] << ")" << std::endl;	
// 	}
}

thrust::device_vector<int> AhTwoArraysToMatrix::TranslateValuesToMatrixCoordinates (const thrust::device_vector<TYPE> &values, TYPE inverseStepWidth, TYPE lowValue) {
	thrust::device_vector<int> tempVec(values.size());

	thrust::transform(values.begin(), values.end(), tempVec.begin(), AhTranslatorFunction(inverseStepWidth, lowValue));

	return tempVec;
}

thrust::device_vector<TYPE> AhTwoArraysToMatrix::RetranslateValuesFromMatrixCoordinates (const thrust::device_vector<int> &values, TYPE inverseStepWidth, TYPE lowValue) {
	thrust::device_vector<TYPE> tempVec(values.size());
	
	thrust::transform(values.begin(), values.end(), tempVec.begin(), AhTranslatorFunction(inverseStepWidth, lowValue)); // uses operator for INTs (which is the retranslation operator)
	
	return tempVec;
}

void AhTwoArraysToMatrix::CalculateHistogram()
{
	thrust::device_vector<int> weightVector(fTranslatedXValues.size(), 1); // just dummy -- each cell should have weight of 1 // TODO: This, once, of course can be changed to support different weightes
	
	// allocate storage for unordered triplets
	cusp::array1d<int, cusp::device_memory> I(fTranslatedXValues.begin(), fTranslatedXValues.end());  // row indices
	cusp::array1d<int, cusp::device_memory> J(fTranslatedYValues.begin(), fTranslatedYValues.end());  // column indices
	cusp::array1d<TYPE, cusp::device_memory> V(weightVector.begin(), weightVector.end());  // values
	
	if (true == fDoTiming) {
		fSwHistSort = new TStopwatch(); // old

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaThreadSynchronize();
		cudaEventRecord(start, 0);
	}
	// sort triplets by (i,j) index using two stable sorts (first by J, then by I)
	/** The Trick (of following line):
	 * 
	 * Generally speaking, sort_by_key sorts a vector A in ascending order, while also putting the corresponding (paired) element of a second vector B into the new position -- leading to vector A being sorted after your wish and a vector B, of which each element is 'chained' to the original one in A in therefore being sorted the way A is.
	 * 
	 * stable_sort_by_key will preserve the original order if two (or more) elements are equal (this is not important for this case, I guess)
	 * 
	 * So, in the first example
	 * 	J is sorted by 'operator<'
	 * 	At the same time the tuple of (I,V) is put in J's order
	 * 
	 * The second example sorts I, so that in the end I and J are both sorted ascending
	 * 
	 * Further read: http://docs.thrust.googlecode.com/hg/group__sorting.html#ga2bb765aeef19f6a04ca8b8ba11efff24
	 */
	thrust::stable_sort_by_key(
		J.begin(), 
		J.end(), 
		thrust::make_zip_iterator(
			thrust::make_tuple(
				I.begin(), 
				V.begin()
			)
		)
	);
	thrust::stable_sort_by_key(
		I.begin(), 
		I.end(), 
		thrust::make_zip_iterator(
			thrust::make_tuple(
				J.begin(), 
				V.begin()
			)
		)
	);
	if (true == fDoTiming) {
		fSwHistSort->Stop(); // old

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&fTimeHistSort, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	// compute unique number of nonzeros in the output
	int num_entries = thrust::inner_product(
		thrust::make_zip_iterator(
			thrust::make_tuple(I.begin(), J.begin())
		), 
		thrust::make_zip_iterator(
			thrust::make_tuple(I.end (),  J.end())
		) - 1, 
		thrust::make_zip_iterator(
			thrust::make_tuple(I.begin(), J.begin())
		) + 1, 
		int(0), 
		thrust::plus<int>(), 
		thrust::not_equal_to< thrust::tuple<int,int> >()
	) + 1;
	
	// allocate output matrix
	cusp::coo_matrix<int, TYPE, cusp::device_memory> A(fNBinsX, fNBinsY, num_entries);
	
	if (true == fDoTiming) {
		fSwHistSum = new TStopwatch(); // old

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaThreadSynchronize();
		cudaEventRecord(start, 0);
	} 
	// sum values with the same (i,j) index
	/** The Trick (of the following line):
	 * reduce_by_key sums up values V(I,J), if two (three...n-2) consecutive key_parameters (=tuples) (I_0..n,J_0..n) are equal
	 * if so, the matching key_tuples (I_0,J_0) are put into the fourth argument (==matrix indices) and the summed up values of V(I_0..n,J_0..n) are put into the fifth argument (==matrix values).
	 * 
	 * equalness (equality?) is tested by invoking the second-to-last argument,
	 * reduction is done by invoking th elast argument
	 * 
	 * See this example for clarification: http://docs.thrust.googlecode.com/hg/group__reductions.html#ga633d78d4cb2650624ec354c9abd0c97f
	 */
	thrust::reduce_by_key(
		thrust::make_zip_iterator(
			thrust::make_tuple(I.begin(), J.begin())
		), 
		thrust::make_zip_iterator(
			thrust::make_tuple(I.end(),   J.end())
		), 
		V.begin(), 
		thrust::make_zip_iterator(
			thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())
		), // TODO: maybe it makes sense to put a tuple of vectors in here? investigate for performance
		A.values.begin(), // TODO: maybe it makes sense to put a plain vector in here? then combine this one and the tuple from the line above into a matrix
		thrust::equal_to< thrust::tuple<int,int> >(), 
		thrust::plus<TYPE>()
	);
	if (true == fDoTiming) {
		fSwHistSum->Stop(); // old

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&fTimeHistSum, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	} 
	
	fCUSPMatrix = A;
	fI = I;
	fJ = J;
	fV = V;
}

TMatrixD AhTwoArraysToMatrix::GetTMatrixD()
{
	if (true == fDoTiming) fSwCreateTMatrixD = new TStopwatch();
	TMatrixD myMatrix(fNBinsY, fNBinsX);
//	std::cout << myMatrix.GetRowUpb() << " "<< myMatrix.GetColUpb() << std::endl;
//	std::cout << fCUSPMatrix.num_rows << " " << fCUSPMatrix.num_cols << std::endl;
	for (int i = 0; i < fCUSPMatrix.num_entries; i++) {
		int column_index = fCUSPMatrix.column_indices[i];
		int row_index = fCUSPMatrix.row_indices[i];
		int value = fCUSPMatrix.values[i];

//		std::cout << i << " - columindex = " << column_index << ", rowindex = " <<  row_index << ", value = " << value << std::endl;

		if (column_index <= fNBinsY && column_index >= 0 && row_index <= fNBinsX && row_index >= 0) {
			myMatrix[column_index][row_index] = value;
		} else {
//			std::cout << "matrix element not filled" << std::endl;
		}
	}
	if (true == fDoTiming) fSwCreateTMatrixD->Stop();
	
	return myMatrix;
}

TH2D * AhTwoArraysToMatrix::GetHistogram()
{
	// Gives a pointer to a TH2D histogram with proper, re-translated boundaries
	if (true == fDoTiming) fSwCreateTH2D = new TStopwatch();

	TH2D * tempHisto = new TH2D(GetTMatrixD());
	tempHisto->GetXaxis()->SetLimits(fXlow, fXlow + fNBinsX * fXStepWidth);
	tempHisto->GetYaxis()->SetLimits(fYlow, fYlow + fNBinsY * fYStepWidth);

	if (true == fDoTiming) fSwCreateTH2D->Stop();
	
	return tempHisto;
}

/**
 * @brief Return all important values of constructed matrix in a list-like fashion
 * @return A device vector of a thrust::tuple with x-value (int), y-value (int) and the multiplicity of that bin (double)
 * 
 * Example access: thrust::get<0>(this->GetPlainMatrixValues()[0])
 */
thrust::device_vector<thrust::tuple<int, int, TYPE> > AhTwoArraysToMatrix::GetPlainMatrixValues()
{
	thrust::device_vector< thrust::tuple<int, int, TYPE> > allStuff(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				fCUSPMatrix.row_indices.begin(), 
				fCUSPMatrix.column_indices.begin(),
				fCUSPMatrix.values.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				fCUSPMatrix.row_indices.end(),
				fCUSPMatrix.column_indices.end(),
				fCUSPMatrix.values.end()
			)
		)
	);
	return allStuff;
}

void AhTwoArraysToMatrix::DoRetranslations()
{
	fRetranslatedXValues = RetranslateValuesFromMatrixCoordinates(CuspVectorToDeviceVector(fCUSPMatrix.row_indices), 1/fXStepWidth, fXlow);
	fRetranslatedYValues = RetranslateValuesFromMatrixCoordinates(CuspVectorToDeviceVector(fCUSPMatrix.column_indices), 1/fYStepWidth, fYlow);
}

thrust::device_vector<thrust::tuple<TYPE, TYPE, TYPE> > AhTwoArraysToMatrix::GetRetranslatedMatrixValues()
{
	// First off, translate I and J from matrix int space into real double space
	if (false == fReTranslationHasBeenDone) {
		DoRetranslations();
		fReTranslationHasBeenDone = true;
	}
	
	// make dev vec tuple construct
	thrust::device_vector< thrust::tuple<TYPE, TYPE, TYPE> > allStuff(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				fRetranslatedXValues.begin(), 
				fRetranslatedYValues.begin(),
				fCUSPMatrix.values.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				fRetranslatedXValues.end(),
				fRetranslatedYValues.end(),
				fCUSPMatrix.values.end()
			)
		)
	);
	return allStuff;
}

template <class T>
thrust::device_vector<T> AhTwoArraysToMatrix::CuspVectorToDeviceVector(const cusp::array1d<T, cusp::device_memory> &cuspvec)
{
	thrust::device_vector<T> values(cuspvec.begin(), cuspvec.end());
	return values;
}

thrust::device_vector<int> AhTwoArraysToMatrix::GetPlainXValues()
{
	return CuspVectorToDeviceVector<int>(fJ);
}

thrust::device_vector<int> AhTwoArraysToMatrix::GetPlainYValues()
{
	return CuspVectorToDeviceVector<int>(fI);
}

thrust::device_vector<TYPE> AhTwoArraysToMatrix::GetMultiplicities()
{
	return CuspVectorToDeviceVector<TYPE>(fV);
}
