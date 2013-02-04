#include <iostream>
#include <fstream>
#include "stdio.h"
#include <vector>
#include <cuda.h>
#include "math.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>

#include "cuPrintf.cu"

#include "TROOT.h"
#include "TApplication.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TCanvas.h"
#include "TStopwatch.h"

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
void readPoints(std::string filename, std::vector<double> &x, std::vector<double> &y, std::vector<double> &r, std::vector<int> &vI, int upToLineNumber = 2) {
	std::ifstream file(filename.c_str());
	float tempX, tempY, tempZ, tempR, tempI;
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

int main (int argc, char** argv) {
	int verbose = 1;
	
	//! fill original data
	std::vector<double> x;
	std::vector<double> y;
	std::vector<double> r;
	std::vector<int> vI;
	// readPoints("data.dat", x, y, r, 18);
	readPoints("real_data.txt", x, y, r, vI, 10000);
	
	thrust::host_vector<double> h_x, h_xMid, h_xMax;
	thrust::host_vector<double> h_y, h_yMid, h_yMax;
	thrust::host_vector<double> h_r, h_rMid, h_rMax;

	//! Setting parameters
	double maxAngle = 180*2;

	// ##################
	// ### FIRST PART ###
	// ##################
	
	//! Change container from std::vector to thrust::host_vector
	int singleRange = 1;
	int midRangeMax = 50;
	int maxRangeMax = 500;
	for (int i = 0; i < x.size(); ++i) {
		if (vI[i] <= maxRangeMax) {
			h_xMax.push_back(x[i]);
			h_yMax.push_back(y[i]);
			h_rMax.push_back(r[i]);
			if (vI[i] <= midRangeMax) {
				h_xMid.push_back(x[i]);
				h_yMid.push_back(y[i]);
				h_rMid.push_back(r[i]);
				if (vI[i] <= singleRange) {
					h_x.push_back(x[i]);
					h_y.push_back(y[i]);
					h_r.push_back(r[i]);
				}
			}
		}
	}

	std::vector<double> timesConfMap, timesGenAngles, timesHoughTrans, timesAll, timesAllMid, timesAllMax;

// int currentEvent = 0;
// int nEvents = 10;
// for (int iEvents = 1; iEvents <= nEvents; ++iEvents) {
// 	while (vI[currentEvent] == iEvents) {
// 		currentEvent++;
// 	}
// }
	std::vector<double> vOfDegreeCellsizes;
	vOfDegreeCellsizes.push_back(30);
	vOfDegreeCellsizes.push_back(15);
	vOfDegreeCellsizes.push_back(10);
	vOfDegreeCellsizes.push_back(5);
	vOfDegreeCellsizes.push_back(2);
	vOfDegreeCellsizes.push_back(1);
	for (int i = 0; i < vOfDegreeCellsizes.size(); ++i) {
		AhHoughTransformation * houghTrans = new AhHoughTransformation(h_x, h_y, h_r, maxAngle, vOfDegreeCellsizes[i], true);
		AhHoughTransformation * houghTransMid = new AhHoughTransformation(h_xMid, h_yMid, h_rMid, maxAngle, vOfDegreeCellsizes[i], true);
		AhHoughTransformation * houghTransMax = new AhHoughTransformation(h_xMax, h_yMax, h_rMax, maxAngle, vOfDegreeCellsizes[i], true);

		timesConfMap.push_back(houghTrans->GetTimeConfMap());
		timesGenAngles.push_back(houghTrans->GetTimeGenAngles());
		timesHoughTrans.push_back(houghTrans->GetTimeHoughTransform());
		timesAll.push_back(houghTrans->GetTimeConfMap() + houghTrans->GetTimeGenAngles() + houghTrans->GetTimeHoughTransform());

		timesAllMid.push_back(houghTransMid->GetTimeConfMap() + houghTransMid->GetTimeGenAngles() + houghTransMid->GetTimeHoughTransform());
		timesAllMax.push_back(houghTransMax->GetTimeConfMap() + houghTransMax->GetTimeGenAngles() + houghTransMax->GetTimeHoughTransform());
		delete houghTrans;
		delete houghTransMid;
		delete houghTransMax;
	}

	TGraphErrors * graphConfMap = new TGraphErrors();
	TGraphErrors * graphGenAngles = new TGraphErrors();
	TGraphErrors * graphHoughTrans = new TGraphErrors();
	TGraphErrors * graphAll = new TGraphErrors();
	TGraphErrors * graphAllMid = new TGraphErrors();
	TGraphErrors * graphAllMax = new TGraphErrors();

	for (int i = 0; i < timesAll.size(); ++i) {
		graphConfMap->SetPoint(i, maxAngle/vOfDegreeCellsizes[i], timesConfMap[i]);
		graphGenAngles->SetPoint(i, maxAngle/vOfDegreeCellsizes[i], timesGenAngles[i]);
		graphHoughTrans->SetPoint(i, maxAngle/vOfDegreeCellsizes[i], timesHoughTrans[i]);
		graphAll->SetPoint(i, maxAngle/vOfDegreeCellsizes[i], timesAll[i]);
		graphAllMid->SetPoint(i, maxAngle/vOfDegreeCellsizes[i], timesAllMid[i]/midRangeMax);
		graphAllMax->SetPoint(i, maxAngle/vOfDegreeCellsizes[i], timesAllMax[i]/maxRangeMax);

	}
	int dotSize = 1;

	graphConfMap->SetLineColor(kOrange);
	graphConfMap->SetFillColor(graphConfMap->GetLineColor() - 10);
	graphConfMap->SetMarkerStyle(kFullDotLarge);
	graphConfMap->SetMarkerSize(dotSize);
	graphConfMap->SetMarkerColor(graphConfMap->GetLineColor() +2);
	graphConfMap->SetTitle("Conf Map");
	graphGenAngles->SetLineColor(kOrange+3);
	graphGenAngles->SetFillColor(graphGenAngles->GetLineColor() - 10);
	graphGenAngles->SetMarkerStyle(kFullDotLarge);
	graphGenAngles->SetMarkerSize(dotSize);
	graphGenAngles->SetMarkerColor(graphGenAngles->GetLineColor() +2);
	graphGenAngles->SetTitle("Gen Angles");
	graphHoughTrans->SetLineColor(kRed);
	graphHoughTrans->SetFillColor(graphHoughTrans->GetLineColor() - 10);
	graphHoughTrans->SetMarkerStyle(kFullDotLarge);
	graphHoughTrans->SetMarkerSize(dotSize);
	graphHoughTrans->SetMarkerColor(graphHoughTrans->GetLineColor() +2);
	graphHoughTrans->SetTitle("Hough Trans");
	graphAll->SetLineColor(kBlue);
	graphAll->SetFillColor(graphAll->GetLineColor() - 10);
	graphAll->SetMarkerStyle(kFullDotLarge);
	graphAll->SetMarkerSize(dotSize);
	graphAll->SetMarkerColor(graphAll->GetLineColor() +2);
	graphAll->SetTitle("All (1 Evt)");


	graphAllMid->SetLineColor(kCyan);
	graphAllMid->SetFillColor(graphAllMid->GetLineColor() - 10);
	graphAllMid->SetMarkerStyle(kFullDotLarge);
	graphAllMid->SetMarkerSize(dotSize);
	graphAllMid->SetMarkerColor(graphAllMid->GetLineColor() +2);
	graphAllMid->SetTitle("All (100 Evt)");
	graphAllMax->SetLineColor(kGreen);
	graphAllMax->SetFillColor(graphAllMax->GetLineColor() - 10);
	graphAllMax->SetMarkerStyle(kFullDotLarge);
	graphAllMax->SetMarkerSize(dotSize);
	graphAllMax->SetMarkerColor(graphAllMax->GetLineColor() +2);
	graphAllMax->SetTitle("All (1000 Evt)");

	TApplication *theApp = new TApplication("app", &argc, argv, 0, -1);
	TCanvas * c1 = new TCanvas("c1", "default", 100, 10, 800, 600);
	
	TMultiGraph * mg = new TMultiGraph();

	mg->Add(graphAll);
	mg->Add(graphHoughTrans);
	mg->Add(graphGenAngles);
	mg->Add(graphConfMap);
	
	mg->Draw("AP");
	mg->GetXaxis()->SetTitle("Grid Size (Number of Grid Points)/#");
	mg->GetYaxis()->SetTitle("Computation Time per Event/s");
	mg->GetYaxis()->SetTitleOffset(1.4);
	
// 	gPad->SetLogy();
	
	TLegend * leg = c1->BuildLegend(0.1,0.75,0.35,0.9);
	leg->SetFillColor(kWhite);
	// TPaveText * formulaLeg = new TPaveText(0.35,0.75,0.45,0.9,"blNDC");
	// formulaLeg->AddText("#sum^{N}_{i} = x^{2}");
	// formulaLeg->SetFillColor(kWhite);
	// formulaLeg->SetBorderSize(1);
	// formulaLeg->SetTextSize(formulaLeg->GetTextSize()*0.7);
	// formulaLeg->Draw();
	c1->Update();

	// ###################
	// ### SECOND PART ###
	// ###################
	TCanvas * c2 = new TCanvas("c2","default2",300,10,800,600);
	TMultiGraph * mgEvents = new TMultiGraph();
	
	mgEvents->Add(graphAll);
	mgEvents->Add(graphAllMid);
	mgEvents->Add(graphAllMax);

	mgEvents->Draw("AP");
	// mgEvents->GetXaxis()->SetTitle("Grid Size (Number of Grid Points)/#");
	// mgEvents->GetYaxis()->SetTitle("Time/s");
	// mgEvents->GetYaxis()->SetTitleOffset(1.4);

	TLegend * legEvents = c1->BuildLegend(0.1,0.75,0.35,0.9);
	legEvents->SetFillColor(kWhite);
	c2->Update();

	theApp->Run();

}
