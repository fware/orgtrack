/******
 *  InitialFrameProcessor.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#ifndef INITIALFRAMEPROCESSOR_HPP_
#define INITIALFRAMEPROCESSOR_HPP_
#pragma once

//#include <unistd.h>
#include <stdio.h>
// Includes:
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "BlockRectFinder.hpp"


using namespace std;
using namespace cv;

class BackboardFinder
{

public:
	BackboardFinder(int t, int file_number);
public:
	Rect process(const Mat& computeFrame);
	void getGray(const Mat& image, Mat& gray);
	Mat getGrayForRect();
	Rect getBBOffset();
private:
	logLevel_e extern_logLevel;
	int m_filenumber;
	int m_thresh;
	BlockRectFinder m_bRF;

};
#endif
