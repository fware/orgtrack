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
	BackboardFinder(int t, int file_number, cv::Size imgSize);
public:
	Rect process(const Mat& grayImage, int leftBBRegionLimit, int rightBBRegionLimit, int bottomBBRegionLimit);
	void getGray(const Mat& image, Mat& gray);
	Mat getGrayForRect();
	Rect getBBOffset();
private:
	logLevel_e extern_logLevel;
	cv::Size m_imgsize;
    bool m_firstpass;
	int m_filenumber;
	int m_thresh;
	int m_backboardoffsetx;
	int m_backboardoffsety;
	int m_backboardcenterx;
	int m_backboardcentery;
	cv::Point m_bbcenterposit;
	cv::Rect m_offsetbackboard;
	BlockRectFinder m_bRF;
	cv::Rect m_backboard;

};
#endif
