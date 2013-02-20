/*
 * AhTranslatorFunction.h
 *
 *  Created on: Aug 22, 2012
 *      Author: herten
 */

#ifndef AHTRANSLATORFUNCTION_H_
#define AHTRANSLATORFUNCTION_H_

#include <functional>
//#include "cuPrintf.cuh"

class AhTranslatorFunction : public std::unary_function<int,int>  {
public:
	AhTranslatorFunction(double _inverseWidthOfCell, double _firstValue) : inverseWidthOfCell(_inverseWidthOfCell),
	firstValue(_firstValue) {};
	
	// Overloaded operators, see each description
	__device__ int operator() (const double &value) const { //< operator to translate from continuous double space into matrix suitable int space
		return ((value - firstValue) * inverseWidthOfCell);
	}
	
	__device__ double operator() (const int &value) const { //< opoerator to REtranslate ready-calculated integer matrix values into old-bordered double space
		return ((value + firstValue) / inverseWidthOfCell);
	}

private:
	double inverseWidthOfCell;
	double firstValue;

};
#endif /* AHTRANSLATORFUNCTION_H_ */

