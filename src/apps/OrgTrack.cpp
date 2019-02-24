/******
 *  OrgTrack.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "BackboardFinder.hpp"
#include "BasketballTracker.hpp"
#include "CourtPositionEstimator.hpp"
#include "ShotEstimator.hpp"
#include "PlayerInfo.hpp"
#include "Utils.hpp"
#include "Logger.hpp"
//#include <opencv2/legacy/compat.hpp>

using namespace std;
using namespace cv;

Rect findBackboard(BackboardFinder& backboardFinder, const Mat& frame, int leftBBRegionLimit, int rightBBRegionLimit, int bottomBBRegionLimit);
Rect getChartBBOffset(BackboardFinder& initialPipe);
//void logTest();

static void help()
{
	printf("\nUsing various functions in opencv to track a basketball.\n"
			"			./OrgTrack {file index number} (choose 1 thru 12)\n\n");
}

int main(int argc, const char** argv)
{
	logLevel_e extern_logLevel = logINFO;
	const string videoIdx 							= argc >= 2 ? argv[1] : "1";
	int fileNumber;
	string videofileName;

	if ( argc > 1 ) {
		fileNumber = atoi( argv[1] );
	}
	else {
		fileNumber = 1;
	}
	stringstream vSS;
	vSS << fileNumber;
	string vIdx 									= vSS.str();

	if ( fileNumber <= 5 )
	{
		videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + ".mp4";
	}
	else if ( fileNumber > 6 && fileNumber < 18 )
	{
		videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + ".MOV";
	}
	else
		videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + ".mp4";

	help();
	int frameCount 									= 0;
	const string bballFileName 						= "/home/fred/Pictures/OrgTrack_res/bball-half-court-vga.jpeg";
	Mat bbsrc 										= imread(bballFileName);
	PlayerInfo playerInfo;
	vector <int> hTableRange;
	namedWindow("halfcourt", WINDOW_NORMAL);
	Mat img;
	Scalar greenColor 								= Scalar (0, 215, 0);
	String body_cascade_name 						= "/home/fred/Pictures/OrgTrack_res/cascadeconfigs/haarcascade_fullbody.xml";
	String modelConfig								= "/home/fred/Dev/DNN_models/MadeShots/V1/made.cfg";
	String modelBinary								= "/home/fred/Dev/DNN_models/MadeShots/V1/made_8200.weights";
	String modelClasses								= "/home/fred/Dev/DNN_models/MadeShots/V1/made.names";
	CascadeClassifier body_cascade; 
	Mat firstFrame;
	Rect offsetBackboard;
	Rect Backboard;
	Rect chartBBOffsetRect;
	int leftActiveBoundary;
	int rightActiveBoundary;
	int topActiveBoundary;
	int bottomActiveBoundary;
	int leftBBRegionLimit;
	int rightBBRegionLimit;
	int bottomBBRegionLimit;
	bool sizeFlag = false;
	bool haveShotRings = false;	
	vector<int> radiusArray;
	Point courtArc[50][1200];
	Rect ballRect;
	const string OUTNAME = "v4_output_longversion.mp4";

	VideoCapture cap(videofileName);

	if( !cap.isOpened() )
	{
		cout << "can not open video file " << videofileName << endl;
		return -1;
	}

	cap >> firstFrame;
	if (firstFrame.empty())
	{
		cout << "Cannot retrieve first video capture frame." << std::endl;
		return -1;
	}

	//int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));   // Get Codec Type - Int form
	Size S = Size((int) cap.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
			(int) cap.get(CAP_PROP_FRAME_HEIGHT));

	log(logDEBUG4) << "first frame [" << S.width << "]";
	if (S.width > 640) 
	{
		sizeFlag = true;
		S = Size (640, 480);
		resize(firstFrame, firstFrame, S);
	}

	Mat finalImg(S.height, S.width+S.width, CV_8UC3);

	leftActiveBoundary 			= firstFrame.cols/4;  
	rightActiveBoundary			= firstFrame.cols*3/4;
	topActiveBoundary			= firstFrame.rows/4;
	bottomActiveBoundary		= firstFrame.rows*3/4;
	leftBBRegionLimit = (int) firstFrame.cols * 3 / 8;
	rightBBRegionLimit = (int) firstFrame.cols * 5 / 8;
	bottomBBRegionLimit = (int) leftActiveBoundary;

	Mat imgBball = Mat::zeros(firstFrame.size(),CV_8UC3);

	//Intialize the object used for initial and second phase frame processing.
	Utils utils;
	BackboardFinder backboardFinder(85, fileNumber, S);
	BasketballTracker basketballTracker;
	ShotEstimator shotEstimator(modelConfig, modelBinary, modelClasses);

	firstFrame.release();
	Mat grayImage;

	if( !body_cascade.load( body_cascade_name ) )
	{
		printf("--(!)Error loading body_cascade_name\n");
		return -1;
	}

	//Instantiate object to handle body tracking.
	CourtPositionEstimator courtEstimator(body_cascade, playerInfo, leftActiveBoundary, rightActiveBoundary);  //, radiusArray);

	for(;;)
	{
		stringstream ss;
		ss << frameCount;

		cap >> img;

		if (sizeFlag) {
			resize(img, img, S);
		}

        if (fileNumber > 10)
        {
        	flip(img, img, 0);
        }

		frameCount++;

		if( img.empty() )
			break;


		if (frameCount < 50 /*Backboard.area() == 0*/)
		{
			utils.getGray(img,grayImage);
			Backboard = findBackboard(backboardFinder, grayImage, leftBBRegionLimit, rightBBRegionLimit, bottomBBRegionLimit);
			Point tempPoint( (Backboard.tl().x + (Backboard.width/2)), (Backboard.tl().y + (Backboard.height/2)) );
			courtEstimator.setBackboardPoint(tempPoint);

			log(logDEBUG4) << frameCount << " End of findBackboard flow";
		}
		else 
		{
			//if (!haveShotRings)
			//{
				chartBBOffsetRect = getChartBBOffset(backboardFinder);
				Point semiCircleCenterPt( (chartBBOffsetRect.tl().x+chartBBOffsetRect.width/2) , (chartBBOffsetRect.tl().y + chartBBOffsetRect.height/2) );
				courtEstimator.setHalfCourtCenterPt(semiCircleCenterPt);

				int radiusIdx = 0;
				for (int radius=40; radius < 280; radius+= 20)	 //Radius for distFromBB
				{
					radiusArray.push_back(radius);

					int temp1, temp2, temp3;
					int yval;
					for (int x=semiCircleCenterPt.x-radius; x<=semiCircleCenterPt.x+radius; x++) 	//Using Pythagorean's theorem to find positions on the each court arc.
					{
						temp1 = radius * radius;
						temp2 = (x - semiCircleCenterPt.x) * (x - semiCircleCenterPt.x);
						temp3 = temp1 - temp2;
						yval = sqrt(temp3);
						yval += semiCircleCenterPt.y;
						Point ptTemp = Point(x, yval);
						courtArc[radiusIdx][x] = ptTemp;
						//circle(bbsrc, ptTemp, 1, Scalar(0,255,0), -1);   //Drawing the arcs on the chart.
					}

					radiusIdx++;
				}
				haveShotRings = true;

			//}
			courtEstimator.setRadiusArray(radiusArray);
			log(logDEBUG4) << frameCount << " End of building rings";
		}

		if (haveShotRings/* || fileNumber > 4*/) //Once true, this processing must happen every frame.
		{
			ballRect = basketballTracker.process(img);		//Tracking the basketball
			utils.getGray(img,grayImage);
			PlayerInfo estPlayerInfo = courtEstimator.findBody(frameCount, grayImage, img);	//method that finds the body of player on the court.


			log(logDEBUG4) << frameCount << " testing ballRect";
			if ((ballRect.x > leftActiveBoundary) 
					&& (ballRect.x < rightActiveBoundary)
					&& (ballRect.y > topActiveBoundary)
					&& (ballRect.y < bottomActiveBoundary))
			{
				/*circle(img,
						estPlayerInfo.position,
						5,
						Scalar(174,232,41));*/

				rectangle(img, ballRect.tl(), ballRect.br(), Scalar(255,180,0), 2, 8, 0 );

				Rect objIntersect = Backboard & ballRect;

				//***************************************************************************
				//TEST: Display detection outline of backboard. This is strictly for testing.
				rectangle(img, Backboard.tl(), Backboard.br(), Scalar(180, 50, 0), 2, 8, 0);
				//***************************************************************************

				if (objIntersect.area() > 0)
				{
					//Shot on basket region
					Mat basketRoI = img(Backboard).clone();
					resize(basketRoI, basketRoI, Size(416, 416));

					//object_results shot_results = shotEstimator.process(basketRoI);   //Detect if shot is made from dnn model.

					log(logDEBUG4) << "intersection:  estPlayerInfo.radiusIdx=" << estPlayerInfo.radiusIdx;
					log(logDEBUG4) << "intersection:  estPlayerInfo.placement=" << estPlayerInfo.placement;

					if (estPlayerInfo.radiusIdx > 49)  estPlayerInfo.radiusIdx = 4;
					circle(bbsrc, courtArc[estPlayerInfo.radiusIdx][estPlayerInfo.placement], 1, Scalar(0, 165, 255), 3);
				}
			}
		}

		//Create string of frame counter to display on video window.
		string str = "frame" + ss.str();		
		putText(img, str, Point(100, 100), FONT_HERSHEY_PLAIN, 2 , greenColor, 2);

		Mat left(finalImg, Rect(0, 0, img.cols, img.rows));
		img.copyTo(left);
		Mat right(finalImg, Rect(bbsrc.cols, 0, bbsrc.cols, bbsrc.rows));
		bbsrc.copyTo(right);		

		log(logDEBUG3) << frameCount << " About to imshow halfcourt image";
		imshow("halfcourt", finalImg);
		left.release();
		right.release();

		char k = (char)waitKey(30);
		if( k == 27 ) break;

		//outputVideo << finalImg;

	}

	return 0;
}

Rect findBackboard(BackboardFinder& backboardFinder, const Mat& grayImage, int leftBBRegionLimit, int rightBBRegionLimit, int bottomBBRegionLimit)
{
	Rect boundRect;

	boundRect = backboardFinder.process(grayImage, leftBBRegionLimit, rightBBRegionLimit, bottomBBRegionLimit);

	return boundRect;
}

Rect getChartBBOffset(BackboardFinder& initialPipe) {
	return initialPipe.getBBOffset();
}
