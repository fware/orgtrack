/******
 *  SecondFrameProcessor.cpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#include "SecondFrameProcessor.hpp"
#include "Logger.hpp"

using namespace std;
using namespace cv;

SecondFrameProcessor::SecondFrameProcessor(const Mat& firstFrame) : thresh(255), canny1(100.0), canny2(14.0)
{
	extern_logLevel = logDEBUG2;
	bg_model = createBackgroundSubtractorMOG2(30, 16, false);
    rng(12345);
}

Rect  SecondFrameProcessor::secondProcessFrame(const Mat & img)
{
	if (fgimg.empty())
		fgimg.create(img.size(), img.type());

	bg_model->apply(img, fgmask);				//Computes a foreground mask for the input video frame.

	fgimg = Scalar::all(0);
	img.copyTo(fgimg, fgmask);

	Canny(fgmask, fgmask, thresh, thresh*2, 3);		//Finds edges in an image.  Going to use it to help identify and track the basketball.
													//Also used in the processing pipeline to identify the person(i.e. human body) shooting the ball.

	vector<vector<Point> > pBallContours;
	vector<Vec4i> pHierarchy;
	findContours(fgmask,pBallContours,pHierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	//Finds contours in foreground mask image.
	
	imgBball = Mat::zeros(fgmask.size(),CV_8UC3);
	for (size_t i = 0; i < pBallContours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
		drawContours(imgBball,pBallContours,i,color,2,8,pHierarchy,0,Point());		//Draws contours onto output image, i.e. imgBball.
																					//The goal here is the find and track the basketball inside of imgBball image frames.
	}
	//rectangle(fgmask, bballRect.tl(), bballRect.br(), Scalar(60,180,255), 2, 8, 0 );
	
	
	//------------Track the basketball!!!!---------------
	vector<Vec3f> localBasketballTracker;
	utils.getGray(imgBball, imgBballGray);
	double minDist = imgBballGray.rows/8;  //8; //4;
	HoughCircles(imgBballGray, localBasketballTracker, CV_HOUGH_GRADIENT, 1, minDist, 100, 14,/*canny1, canny2,*/ 1, 9 );

	if (localBasketballTracker.size() > 0)
	{
		for (size_t i = 0; i < localBasketballTracker.size(); i++)
		{
			//Point bballCenter(cvRound(localBasketballTracker[i][0]), cvRound(localBasketballTracker[i][1]));
			//circle(img, bballCenter, 3, Scalar(0,255,0), -1);
			double bballRadius = (double)cvRound(localBasketballTracker[i][2]);
			double bballDiameter = (double)cvRound(2 * bballRadius);

			int bballXtl = (int)(localBasketballTracker[i][0] - bballRadius);
			int bballYtl = (int)(localBasketballTracker[i][1] - bballRadius);
			bballRect = Rect(bballXtl, bballYtl, bballDiameter, bballDiameter);
		}
	}

	log(logDEBUG) << "End of secondProcessFrame";

	return bballRect;
}
