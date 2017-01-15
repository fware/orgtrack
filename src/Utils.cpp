/******
 * Author:  Fred Ware
 *
 * WareShop Consulting LLC
 * Copyright 2016
 *
 */
#include "Utils.hpp"

Utils::Utils()
{
	mHgtRings = new Point* [50];
	for (int i = 0; i < 1200; i++)
		mHgtRings[i] = new Point[1200];
}

void Utils::generateCourtRings(Rect offsetBackboard)
{
	Point semiCircleCenterPt( (offsetBackboard.tl().x+offsetBackboard.width/2) , (offsetBackboard.tl().y + offsetBackboard.height/2) );
	Point bbCenterPosit = semiCircleCenterPt;
	
	int bCounter = 0;
	for (int radius=40; radius < 280; radius+= 20)	 //Radius for distFromBB
	{
		mHTableRange.push_back(radius);
		
		int temp1, temp2, temp3;
		int yval;
		for (int x=bbCenterPosit.x-radius; x<=bbCenterPosit.x+radius; x++) 
		{
			temp1 = radius * radius;
			temp2 = (x - bbCenterPosit.x) * (x - bbCenterPosit.x);
			temp3 = temp1 - temp2;
			yval = sqrt(temp3);
			yval += bbCenterPosit.y;
			Point ptTemp = Point(x, yval);
			mHgtRings[bCounter][x] = ptTemp;
		}
		
		bCounter++;
	}
}

vector<int> Utils::getHTableRange() 
{
	return mHTableRange;
}

Point **Utils::getHeightRings() 
{
	return mHgtRings;
}

