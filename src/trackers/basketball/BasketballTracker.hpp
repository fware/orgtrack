/******
 *  BasketballTracker.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#pragma once
#ifndef BASKETBALLTRACKER_HPP_
#define BASKETBALLTRACKER_HPP_

// Includes:
#include "BallTracker.hpp"

using namespace std;
using namespace cv;

class BasketballTracker : public BallTracker
{
public:
	int getCoords();
	Rect  process(const Mat& img);
public:
	logLevel_e extern_logLevel;
};

#endif
