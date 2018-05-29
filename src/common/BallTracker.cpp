/*
 * BallTracker.cpp
 *
 *  Created on: May 28, 2018
 *      Author: WareShop Consulting LLC
 */
#include "BallTracker.hpp"
#include "Logger.hpp"

using namespace std;
using namespace cv;

BallTracker::BallTracker() : thresh(255), canny1(100.0), canny2(14.0)
{
	bg_model = createBackgroundSubtractorMOG2(30, 16, false);
    rng(12345);
}

BallTracker::~BallTracker() {}

Rect BallTracker::process(const Mat &img)
{
	log(logINFO) << "Base class BallTracker::process is doing nothing.  You must declare from a specific ball tracker";
	Rect nothing;
	return nothing;
}


