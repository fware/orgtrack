/*
 * BallTracker.hpp
 *
 *  Created on: May 28, 2018
 *      Author: WareShop LLC
 */
#ifndef SRC_COMMON_BALLTRACKER_HPP_
#define SRC_COMMON_BALLTRACKER_HPP_

// Includes:
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Utils.hpp"
#include "Logger.hpp"

//Abstract base class for tracking a ball
class BallTracker
{
public:
	BallTracker();
	~BallTracker();

public:
	Rect  process(const Mat& computeFrame);

public:
	logLevel_e extern_logLevel;
	int thresh;
	float canny1;
	float canny2;
	Ptr<BackgroundSubtractor> bg_model;
	vector<vector<Point> > ballContours;
	RNG rng;
	Mat fgimg;
	Mat fgmask;
	Mat imgBall;
	Mat imgBallGray;
	Rect ballRect;
	Utils utils;
};



#endif /* SRC_COMMON_BALLTRACKER_HPP_ */
