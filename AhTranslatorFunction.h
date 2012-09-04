/*
 * AhTranslatorFunction.h
 *
 *  Created on: Aug 22, 2012
 *      Author: herten
 */

#ifndef AHTRANSLATORFUNCTION_H_
#define AHTRANSLATORFUNCTION_H_

#include <functional>

class AhTranslatorFunction : public std::unary_function<int,int>  {
public:
	AhTranslatorFunction(double _widthOfCell, double _firstValue) : widthOfCell(_widthOfCell),
	firstValue(_firstValue) {};
	
	__device__ int operator() (const double &value) const {
		return ((value - firstValue)/widthOfCell);
	}

private:
	double widthOfCell;
	double firstValue;

};
#endif /* AHTRANSLATORFUNCTION_H_ */

