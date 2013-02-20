#include <iostream>

#include "AhHoughTransformation.h"

int main (int argc, char** argv) {

	std::vector<double> x;
	std::vector<double> y;
	std::vector<double> r;

	x.push_back(0.1);
	y.push_back(0.2);
	r.push_back(0.3);

	AhHoughTransformation * houghTrans = new AhHoughTransformation(x, y, r, 360, 30, true);

	std::cout << "Some output to test if the object actually exists: " << houghTrans->GetTimeHoughTransform() << std::endl;
}
