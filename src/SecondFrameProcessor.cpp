/*****************************************************************************
*   WareShop Consulting LLC
*   Copyright 2016.
*****************************************************************************/
#include "SecondFrameProcessor.hpp"

using namespace std;
using namespace cv;

SecondFrameProcessor::SecondFrameProcessor(const Mat& firstFrame) : thresh(255)
{
	bg_model = createBackgroundSubtractorMOG2(30, 16, false);
    imgBball = Mat::zeros(firstFrame.size(),CV_8UC1);
    rng(12345);
}

Rect  SecondFrameProcessor::secondProcessFrame(const Mat & img)
{
	bg_model->apply(img, fgmask);
	Canny(fgmask, fgmask, thresh, thresh*2, 3);	
	findContours(fgmask,bballContours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	
	for (int i = 0; i < bballContours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
		drawContours(imgBball,bballContours,i,color,2,8,hierarchy,0,Point());
	}
	
	
	//------------Track the basketball!!!!---------------
	float canny1 = 100;
	float canny2 = 14; //16;
	double minDist = imgBball.rows/4;  //8; //4;
	HoughCircles(imgBball, basketballTracker, CV_HOUGH_GRADIENT, 1, minDist, canny1, canny2, 1, 9 );

	Rect bballRect;
	if (basketballTracker.size() > 0)
	{
		for (size_t i = 0; i < basketballTracker.size(); i++)
		{
			Point bballCenter(cvRound(basketballTracker[i][0]), cvRound(basketballTracker[i][1]));
			double bballRadius = (double)cvRound(basketballTracker[i][2]);
			double bballDiameter = (double)cvRound(2 * bballRadius);

			int bballXtl = (int)(basketballTracker[i][0] - bballRadius);
			int bballYtl = (int)(basketballTracker[i][0] - bballRadius);
			bballRect = Rect(bballXtl, bballYtl, bballDiameter, bballDiameter);
		}
	}

	return bballRect;
}
