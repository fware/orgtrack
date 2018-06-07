/******
 *  PlayerInfo.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#ifndef PLAYERINFO_HPP_
#define PLAYERINFO_HPP_
//Standard Includes
#pragma once
#include <stdio.h>
//OpenCV Includes
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

class PlayerInfo 
{
public:
	PlayerInfo();
public:
	int		activeValue;
	int		radiusIdx;
	int 	placement;
	Point   position; 
	int 	frameCount;
};
#endif
