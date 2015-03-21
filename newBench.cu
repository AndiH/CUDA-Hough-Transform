#ifdef USE_DOUBLES
	typedef double TYPE;
#else
	typedef float TYPE;
#endif
#include <iostream>
#include <fstream>
#include "stdio.h"
#include <vector>
#include "math.h"
#include <string>

#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>

//#include "cuPrintf.cu"

#include "TROOT.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TStopwatch.h"
// #include "TMath.h"
#include "TAxis.h"

#include "AhHoughTransformation.h"

/**
 * @brief Reads data from specific formated file into three given vectors
 * @param filename       Filename (include path if necessary)
 * @param x              x-value vector (1st row of file)
 * @param y              y-value vector (2nd row of file)
 * @param r              r-value vector (4th row of file)
 * @param upToLineNumber Up to with line (including that line) the file will be read. Starts at line 1.
 *
 * Uses just x, y and r at the moment, because that's all I need.
 */
void readPoints(std::string filename, std::vector<TYPE> &x, std::vector<TYPE> &y, std::vector<TYPE> &r, std::vector<int> &vI, int upToLineNumber = 2) {
	std::ifstream file(filename.c_str());
	TYPE tempX, tempY, tempZ, tempR, tempI;
	char tempChar[10];
	int i = 1;
	if (!file.good() || file.fail()) std::cout << "Failed opening file!" << std::endl;
	while(i <= upToLineNumber && file >> tempX >> tempY >> tempZ >> tempR >> tempI >> tempChar) {
		x.push_back(tempX);
		y.push_back(tempY);
		r.push_back(tempR);
		vI.push_back((int)tempI);
		i++;
	}
	file.close();
}

namespace my {
	void printType(float object) {
		std::cout << "I'm using floats!" << std::endl;
	}
	void printType(double object) {
		std::cout << "I'm using doubles!" << std::endl;
	}
	std::string returnTypeString(float object) {
		return "float";
	}
	std::string returnTypeString(double object) {
		return "double";
	}
}

void printCsvStyle(std::string deviceName, int nHits, TYPE angleStepSize, TYPE tConf, TYPE tAngles, TYPE tHough) {
	// structure:
	// devicename,datatype,nHits,angelStepSize,timeConformalMap,timeAngleGeneration,timeHough

	std::cout << deviceName << "," << my::returnTypeString(angleStepSize) << "," << nHits << "," << angleStepSize << "," << tConf << "," << tAngles << "," << tHough << std::endl;
}

int main (int argc, char** argv) {
	int verbose = getenv("LH_VERBOSE") != NULL ? atoi(getenv("LH_VERBOSE")) : 0;
	int useDeviceNumber = getenv("LH_DEVICE_NUMBER") != NULL ? atoi(getenv("LH_DEVICE_NUMBER")) : 0;
	// create context to initialize CUDA environment
	cudaFree(0);
	// cudaSetDevice(2);
	cudaSetDevice(useDeviceNumber);

	cudaDeviceProp devProps;
	cudaGetDeviceProperties(&devProps, useDeviceNumber);

	// beginning of profiling region
	cudaProfilerStart();

	// Get Environment Values
	int readInUpToLineNumber = getenv("LH_READ_N_LINES") != NULL ? atoi(getenv("LH_READ_N_LINES")) : 10;
	TYPE angleStepSize = getenv("LH_ANGLE_STEPSIZE") != NULL ? atof(getenv("LH_ANGLE_STEPSIZE")) : 30;

	//! fill original data
	std::vector<TYPE> x, y, r;
	std::vector<int> vI;
	// readPoints("data.dat", x, y, r, 18);
	readPoints("real_data.txt", x, y, r, vI, readInUpToLineNumber);
	
	thrust::host_vector<TYPE> h_x(x), h_y(y), h_r(r);

	TYPE maxAngle = 180*2;

	if (verbose > 0) {
		my::printType(x[0]);
	}

	// ##################
	// ### FIRST PART ###
	// ##################

	if (verbose > 0) {
		my::printType(h_x[0]);
	}

	TYPE timeConfMap, timeGenAngles, timeHoughTrans;

	AhHoughTransformation * houghTrans = new AhHoughTransformation(h_x, h_y, h_r, maxAngle, angleStepSize, true);

	timeConfMap = houghTrans->GetTimeConfMap();
	timeGenAngles = houghTrans->GetTimeGenAngles();
	timeHoughTrans = houghTrans->GetTimeHoughTransform();

	delete houghTrans;

	printCsvStyle((std::string) devProps.name, readInUpToLineNumber, angleStepSize, timeConfMap, timeGenAngles, timeHoughTrans);

	cudaDeviceReset();

}
