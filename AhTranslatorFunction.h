/*
 * AhTranslatorFunction.h
 *
 *  Created on: Aug 22, 2012
 *      Author: herten
 */

#ifndef AHTRANSLATORFUNCTION_H_
#define AHTRANSLATORFUNCTION_H_

#include <functional>
#include "cuPrintf.cuh"

class AhTranslatorFunction : public std::unary_function<int,int>  {
public:
	AhTranslatorFunction(double _inverseWidthOfCell, double _firstValue) : inverseWidthOfCell(_inverseWidthOfCell),
	firstValue(_firstValue) {};
	
	__device__ int operator() (const double &value) const {
		return ((value - firstValue) * inverseWidthOfCell);
//		if (result > 31100 && result < 31200){
//			printf("DEBUG - result = %.8f = (%.8f - %.8f)/%.8f\n", result, value, firstValue, inverseWidthOfCell);
//		}
// 		return (int)result; //casting a double to an int doesnt work, with float it does, hooray
	}

private:
	double inverseWidthOfCell;
	double firstValue;

};
#endif /* AHTRANSLATORFUNCTION_H_ */

