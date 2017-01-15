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

