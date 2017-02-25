/******
 *  BlockRectFinder.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#pragma once
#ifndef BLOCKRECTFINDER_HPP_
#define BLOCKRECTFINDER_HPP_

// Includes:
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <unistd.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "Logger.hpp"

using namespace std;
using namespace cv;

class BlockRectFinder 
{
public:
	BlockRectFinder();

	Rect getBoundRects(const Mat& imageFrame, Mat& threshold_output, vector<Vec4i> hierarchy, vector< vector<Point> > boardContours, int file_number);

	Rect findBackboard(const Mat& imgFrame, size_t bbContoursSize, vector<Rect> bound_Rects, int file_number);

	Rect getBBOffset();

private:
	logLevel_e extern_logLevel;
	bool mHaveBackboard;
	Rect mFreezeBB;
	Rect mOffsetBackboard;
	int mBackBoardOffsetX;
	int mBackBoardOffsetY;
	int mFreezeCenterX;
	int mFreezeCenterY;
	int mFileNumber;
};
#endif
 
