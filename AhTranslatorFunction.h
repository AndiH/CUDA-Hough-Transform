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
	AhTranslatorFunction(double _widthOfCell) : widthOfCell(_widthOfCell) {};
	__device__ int operator() (const double &value) const {
		return value/widthOfCell;
	}
private:
	double widthOfCell;

};
#endif /* AHTRANSLATORFUNCTION_H_ */

