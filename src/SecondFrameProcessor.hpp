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
#include "opencv2/video/background_segm.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

class SecondFrameProcessor
{
public:
	SecondFrameProcessor(const Mat& firstFrame);
public:
	Rect  secondProcessFrame(const Mat& computeFrame);
	void getGray(const Mat& image, Mat& gray);
private:
	int thresh;
	RNG rng;
	Mat fgmask;
	Mat imgBball;
	Ptr<BackgroundSubtractor> bg_model;	
	vector<vector<Point> > bballContours;
	vector<Vec4i> hierarchy;
	vector<Vec3f> basketballTracker;
};
