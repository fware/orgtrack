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
#include "DebugHelpers.hpp"
#include "InitialFrameProcessor.hpp"
#include "SecondFrameProcessor.hpp"
#include "CourtPositionEstimator.hpp"
#include "PlayerInfo.hpp"
#include "Utils.hpp"
//#include <opencv2/legacy/compat.hpp>


//#define IMG_DEBUG
#define SHOT_DEBUG

using namespace std;
using namespace cv;

Rect findBackboard(const Mat& frame, InitialFrameProcessor& initialPipe);
Rect getChartBBOffset(InitialFrameProcessor& initialPipe);

static void help()
{
 printf("\nUsing various functions in opencv to track a basketball.\n"
"			./OrgTrack {file index number} (choose 1 thru 8)\n\n");
}

int main(int argc, const char** argv)
{
	//const string videofileName = argc >= 2 ? argv[1] : "v1.mp4";
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

	if ( fileNumber < 7 ) 
	{
		videofileName 						= "/home/fred/Videos/testvideos/v" + vIdx + ".mp4";
	}
	else 
	{
		videofileName						= "/home/fred/Videos/testvideos/m" + vIdx + ".MOV";
	}


    help();
	int frameCount 									= 0;
	const string bballPatternFile 					= "/home/fred/Pictures/OrgTrack_res/bball3_vga.jpg";
	Mat patternImage 								= imread(bballPatternFile);
	const string bballFileName 						= "/home/fred/Pictures/OrgTrack_res/bball-half-court-vga.jpeg";
    Mat bbsrc 										= imread(bballFileName);	
	int newPlayerWindowSize 						= 50;
	//PlayerObs newPlayerWindow[newPlayerWindowSize];  
	PlayerInfo playerInfo;
	vector <int> hTableRange;
	//Point **courtArc;
	namedWindow("halfcourt", WINDOW_NORMAL);

    Mat img;
	Scalar redColor 								= Scalar (0, 20, 180);
	Scalar greenColor 								= Scalar (0, 215, 0);
	Scalar blueColor 								= Scalar (180, 0, 0);
	String body_cascade_name 						= "/home/fred/Pictures/OrgTrack_res/cascadeconfigs/haarcascade_fullbody.xml";
	CascadeClassifier body_cascade; 
	Mat firstFrame;
	Rect offsetBackboard;
	Rect Backboard;
	Rect chartBBOffsetRect;
	int leftActiveBoundary;
	int rightActiveBoundary;
	int topActiveBoundary;
	int bottomActiveBoundary;
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
        std::cout << "Cannot retrieve first video capture frame." << std::endl;
        return -1;
	}

    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
    Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	if (S.width > 640) 
	{
		sizeFlag = true;
		S = Size (640, 480);
	}
	
	Size outS = Size ((int) 2 * S.width, S.height);
	//VideoWriter outputVideo; 
	//outputVideo.open(OUTNAME, ex, cap.get(CV_CAP_PROP_FPS), outS, true);
	Mat finalImg(S.height, S.width+S.width, CV_8UC3);

	leftActiveBoundary 			= firstFrame.cols/4;  
	rightActiveBoundary			= firstFrame.cols*3/4;
	topActiveBoundary				= firstFrame.rows/4;
	bottomActiveBoundary			= firstFrame.rows*3/4;
	Mat imgBball = Mat::zeros(firstFrame.size(),CV_8UC3);


    //Intialize the object used for initial and second phase frame processing.
	Utils utils;
	InitialFrameProcessor initialPipe(85, fileNumber);
	SecondFrameProcessor secondPipe(firstFrame);
	
	firstFrame.release();
	Mat grayImage;
		
	if( !body_cascade.load( body_cascade_name ) )
	{
		printf("--(!)Error loading body_cascade_name\n");
		return -1;
	}

	for(;;)
    {
		stringstream ss;
		ss << frameCount;

        cap >> img; 	
		if (sizeFlag)
			resize(img, img, S);
		frameCount++;
		
        if( img.empty() )
            break;


		//Instantiate object to handle body tracking.
		CourtPositionEstimator courtEstimator(body_cascade, 
											  playerInfo,
											  leftActiveBoundary,
											  rightActiveBoundary,
											  radiusArray);
		
		if (Backboard.area() == 0) 
		{
			Backboard = findBackboard(img, initialPipe);
			Point tempPoint(
							(Backboard.tl().x + (Backboard.width/2)),
							(Backboard.tl().y + (Backboard.height/2))
							);
			courtEstimator.setFreezeBBPoint(tempPoint);			
		}
		else 
		{
			if (!haveShotRings) 
			{
				chartBBOffsetRect = getChartBBOffset(initialPipe);
				
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
						circle(bbsrc, ptTemp, 1, Scalar(0,255,0), -1);
					}
					
					radiusIdx++;
				}
							
				haveShotRings = true;

			}

			rectangle(img, Backboard.tl(), Backboard.br(), Scalar(180, 50, 0), 2, 8, 0);	//Display detection outline of backboard. This is strictly for testing.
		}

		if (haveShotRings) //Once true, this processing must happen every frame.
		{
			utils.getGray(img,grayImage);
			ballRect = secondPipe.secondProcessFrame(grayImage);

			PlayerInfo estPlayerInfo = courtEstimator.findBody(grayImage);

			if ((ballRect.x > leftActiveBoundary) 
							&& (ballRect.x < rightActiveBoundary)
							&& (ballRect.y > topActiveBoundary)
							&& (ballRect.y < bottomActiveBoundary)) 
			{
				Rect objIntersect = Backboard & ballRect;
			
				if (objIntersect.area() > 0) {
					cout << "We have an intersect!!" << endl;
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

		imshow("halfcourt", finalImg);
		left.release();
		right.release();

        char k = (char)waitKey(30);
        if( k == 27 ) break;

		//outputVideo << finalImg;

    }

    return 0;
}

Rect findBackboard(const Mat& frame, InitialFrameProcessor& initialPipe) 
{
	bool result;
    Rect boundRect;

	boundRect = initialPipe.initialProcessFrame(frame);
	
	return boundRect;
}

Rect getChartBBOffset(InitialFrameProcessor& initialPipe) {
	return initialPipe.getBBOffset();
}



