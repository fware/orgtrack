/******
 *  SecondFrameProcessor.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#ifndef SECONDFRAMEPROCESSOR_HPP_
#define SECONDFRAMEPROCESSOR_HPP_
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
#include <iostream>
// Includes:
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Utils.hpp"
#include "Logger.hpp"

using namespace std;
using namespace cv;

class SecondFrameProcessor
{
public:
	SecondFrameProcessor(const Mat& firstFrame);
public:
	Rect  secondProcessFrame(const Mat& computeFrame);
private:
	logLevel_e extern_logLevel;
	int thresh;
	float canny1;
	float canny2;
	RNG rng;
	Mat fgimg;
	Mat fgmask;
	Mat imgBball;
	Mat imgBballGray;
	Rect bballRect;
	Utils utils;
	Ptr<BackgroundSubtractor> bg_model;	
	vector<vector<Point> > bballContours;
	vector<Vec3f> basketballTracker;
};
#endif
