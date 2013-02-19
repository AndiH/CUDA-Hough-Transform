/*
 * AhTranslatorFunction.h
 *
 *  Created on: Aug 22, 2012
 *      Author: herten
 */

#ifndef AHTRANSLATORFUNCTION_H_
#define AHTRANSLATORFUNCTION_H_

#ifdef USE_DOUBLES
	typedef double TYPE;
#else
	typedef float TYPE;
#endif

#include <functional>

class AhTranslatorFunction : public std::unary_function<int,int>  {
public:
	AhTranslatorFunction(TYPE _inverseWidthOfCell, TYPE _firstValue) : inverseWidthOfCell(_inverseWidthOfCell),
	firstValue(_firstValue) {};
	
	// Overloaded operators, see each description
	__device__ int operator() (const TYPE &value) const { //< operator to translate from continuous double space into matrix suitable int space
		return ((value - firstValue) * inverseWidthOfCell);
	}
	
	__device__ TYPE operator() (const int &value) const { //< opoerator to REtranslate ready-calculated integer matrix values into old-bordered double space
		return ((value + firstValue) / inverseWidthOfCell);
	}

private:
	TYPE inverseWidthOfCell;
	TYPE firstValue;

};
#endif /* AHTRANSLATORFUNCTION_H_ */

