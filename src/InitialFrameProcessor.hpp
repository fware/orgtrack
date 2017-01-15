#pragma once
/******
 * Author:  Fred Ware
 *
 * WareShop Consulting LLC
 * Copyright 2016
 *
 */

//#include <unistd.h>
#include <stdio.h>
// Includes:
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "BlockRectFinder.hpp"


using namespace std;
using namespace cv;

class InitialFrameProcessor
{

public:
	InitialFrameProcessor(int t, int file_number);
public:
	Rect initialProcessFrame(const Mat& computeFrame);
	void getGray(const Mat& image, Mat& gray);
	Mat getGrayForRect();
	Rect getBBOffset();
private:
	int m_filenumber;
	int m_thresh;
	Mat grayForRect;
	BlockRectFinder m_bRF;

};
