
/******
 * Author:  Fred Ware
 *
 * WareShop Consulting LLC
 * Copyright 2016
 *
 */
#include <stdio.h>
	 // Includes:
#include "opencv2/core/core.hpp"
	 
	 
using namespace std;
using namespace cv;
	 
class Utils
{
public:
	Utils();

public:
	void generateCourtRings(Rect offsetBackboard);
	vector<int> getHTableRange();
	Point **getHeightRings();

private:
	vector<int> mHTableRange;
	Point** mHgtRings;

};

