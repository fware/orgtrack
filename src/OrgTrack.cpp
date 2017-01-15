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
//#include <opencv2/legacy/compat.hpp>


//#define IMG_DEBUG
#define SHOT_DEBUG

using namespace std;
using namespace cv;

void getGray(const Mat& image, Mat& gray);
double DistanceToCamera(double knownWidth, double focalLength, double perWidth);
int getPlayerOffsetX(int frame_count, float player_pos_x, float player_pos_y, int player_hgt);
int getPlayerOffsetY(int frame_count, float player_pos_x, float player_pos_y, int player_hgt);
vector<Point> get_sliding_windows(Mat& image, int winWidth);
Mat drawSemiCircle(Mat& image, int radius, Point center);
//float euclideanDist(Point& p, Point& q);
double euclideanDist(double x1, double y1, double x2, double y2);
double oneDDist(double p1, double p2);
int findIndex_BSearch(const vector< int> &my_numbers, int key);
//void processFrames(const Mat& computeFrame, cv::VideoCapture& capture);
Rect findBackboard(const Mat& frame, InitialFrameProcessor& initialPipe);
Rect getChartBBOffset(InitialFrameProcessor& initialPipe);
void generateCourtRings(Rect offsetBackboard);

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
	Point **hgtRings;
	namedWindow("halfcourt", WINDOW_NORMAL);

    Mat img;
	Scalar redColor 								= Scalar (0, 20, 180);
	Scalar greenColor 								= Scalar (0, 215, 0);
	Scalar blueColor 								= Scalar (180, 0, 0);
	String body_cascade_name 						= "/home/fred/Pictures/OrgTrack_res/cascadeconfigs/haarcascade_fullbody.xml";
	CascadeClassifier body_cascade; 
	Mat firstFrame;
	Rect offsetBackboard;
	Rect freezeBB;
	Rect chartBBOffsetRect;
	int leftImgBoundary;
	int rightImgBoundary;
	int topImgBoundary;
	int bottomImgBoundary;
	bool sizeFlag = false;
	bool haveChart = false;	
	vector<int> mHTableRange;
	Point mHgtRings[50][1200];
	Rect bBallRect;

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

	leftImgBoundary 			= firstFrame.cols/4;  
	rightImgBoundary			= firstFrame.cols*3/4;
	topImgBoundary				= firstFrame.rows/4;
	bottomImgBoundary			= firstFrame.rows*3/4;
	Mat imgBball = Mat::zeros(firstFrame.size(),CV_8UC3);


    //Intialize the object used for initial and second phase frame processing.
	InitialFrameProcessor initialPipe(85, fileNumber);
	SecondFrameProcessor secondPipe(firstFrame);
	
	firstFrame.release();
	Mat grayForRect;
		
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
											  leftImgBoundary,
											  rightImgBoundary,
											  mHTableRange);
		
		if (freezeBB.area() == 0) 
		{
			freezeBB = findBackboard(img, initialPipe);
			Point tempPoint(
							(freezeBB.tl().x + (freezeBB.width/2)),
							(freezeBB.tl().y + (freezeBB.height/2))
							);
			courtEstimator.setFreezeBBPoint(tempPoint);			
		}
		else 
		{
			if (!haveChart) 
			{
				chartBBOffsetRect = getChartBBOffset(initialPipe);
				
				Point semiCircleCenterPt( (chartBBOffsetRect.tl().x+chartBBOffsetRect.width/2) , (chartBBOffsetRect.tl().y + chartBBOffsetRect.height/2) );
				courtEstimator.setHalfCourtCenterPt(semiCircleCenterPt);
				
				int bCounter = 0;
				for (int radius=40; radius < 280; radius+= 20)	 //Radius for distFromBB
				{
					mHTableRange.push_back(radius);
					
					int temp1, temp2, temp3;
					int yval;
					for (int x=semiCircleCenterPt.x-radius; x<=semiCircleCenterPt.x+radius; x++) 
					{
						temp1 = radius * radius;
						temp2 = (x - semiCircleCenterPt.x) * (x - semiCircleCenterPt.x);
						temp3 = temp1 - temp2;
						yval = sqrt(temp3);
						yval += semiCircleCenterPt.y;
						Point ptTemp = Point(x, yval);
						mHgtRings[bCounter][x] = ptTemp;
						circle(bbsrc, ptTemp, 1, Scalar(0,255,0), -1);
					}
					
					bCounter++;
				}
							
				haveChart = true;

			}

			rectangle(img, freezeBB.tl(), freezeBB.br(), Scalar(180, 50, 0), 2, 8, 0);
		}

		rectangle(img, bBallRect.tl(), bBallRect.br(), Scalar(180, 255, 0), 2, 8, 0);
		if (haveChart) //Once true, this processing must happen every frame.
		{
			bBallRect = secondPipe.secondProcessFrame(img);

			getGray(img,grayForRect);
			PlayerInfo estPlayerInfo = courtEstimator.findBody(grayForRect);

			if ((bBallRect.x > leftImgBoundary) 
							&& (bBallRect.x < rightImgBoundary)
							&& (bBallRect.y > topImgBoundary)
							&& (bBallRect.y < bottomImgBoundary)) 
			{
				//*********  For Testing only
				//rectangle(img, bBallRect.tl(), bBallRect.br(), Scalar(180, 255, 0), 2, 8, 0);
				//****************


				//The basketball on video frames.
				cout << frameCount << ": freezeBB= " << freezeBB
					 << "bBallRect= " << bBallRect 
					 << " is inside boundaries" << endl;
				
				Rect objIntersect = freezeBB & bBallRect;
			
				if (objIntersect.area() > 0) {
					cout << "We have an intersect!!" << endl;
					circle(bbsrc, hgtRings[estPlayerInfo.radiusIdx][estPlayerInfo.placement], 1, Scalar(0, 165, 255), 3); 
				}
			}				
		}


		
		
/*
		///*******Start of main code to detect Basketball*************************
        if( fgimg.empty() )
          fgimg.create(img.size(), img.type());

        bg_model->apply(img, fgmask);

        fgimg = Scalar::all(0);
        img.copyTo(fgimg, fgmask);

		Canny(fgmask, fgmask, thresh, thresh*2, 3);
		
		vector<vector<Point> > bballContours;
		vector<Vec4i> hierarchy;
		findContours(fgmask,bballContours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		
		for (int i = 0; i < bballContours.size(); i++ )
		{
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
			drawContours(imgBball,bballContours,i,color,2,8,hierarchy,0,Point());
		}


		//------------Track the basketball!!!!---------------
        vector<Vec3f> basketballTracker;
		getGray(imgBball, imgBBallGray);
		float canny1 = 100;
        float canny2 = 14; //16;
		double minDist = imgBBallGray.rows/4;  //8; //4;
		HoughCircles(imgBBallGray, basketballTracker, CV_HOUGH_GRADIENT, 1, minDist, canny1, canny2, 1, 9 );

		if (basketballTracker.size() > 0)
		{
			for (size_t i = 0; i < basketballTracker.size(); i++)
			{		
				Point bballCenter(cvRound(basketballTracker[i][0]), cvRound(basketballTracker[i][1]));
				double bballRadius = (double) cvRound(basketballTracker[i][2]);
				double bballDiameter = (double)(2*bballRadius);

				int bballXtl = (int)(basketballTracker[i][0]-bballRadius);
				int bballYtl = (int)(basketballTracker[i][1]-bballRadius);
				bballRect = Rect(bballXtl, bballYtl, bballDiameter, bballDiameter);

				//-------We chose what we think is the basketball and put it in basketballChoice!!!---------
				Rect basketballChoice = bballRect;
				
				if ((bballRect.x > leftImgBoundary) 
								&& (bballRect.x < rightImgBoundary)
								&& (bballRect.y > topImgBoundary)
								&& (bballRect.y < bottomImgBoundary)) {
					sleepDelay = 0;

					//The basketball on video frames.
					Scalar rngColor = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
					Rect objIntersect = freezeBB & bballRect;

					//---Start of the process of identifying a shot at the basket!!!------------
					if (objIntersect.area() > 0) {
						
						//---Start of using player position on halfcourt image to draw shot location-----
						int activeCount = 0;
						Point playerPositAvg = Point(0, 0);
						int playerHeightIndex = 0;
						int playerPlacement = 0;
						
						for (int k=30; k < newPlayerWindowSize; k++) 
						{
							activeCount += newPlayerWindow[k].activeValue;
							playerPositAvg += newPlayerWindow[k].position;
							playerHeightIndex += newPlayerWindow[k].radiusIdx;
							playerPlacement += newPlayerWindow[k].placement;
						}

						if (activeCount > 0) 
						{
							playerPositAvg /= activeCount;
							playerHeightIndex /= activeCount;
							playerPlacement /= activeCount;
						}

						if (playerPositAvg.x == 0 || playerPositAvg.y == 0) 
						{
							playerPositAvg = newPlayerWindow[newPlayerWindowSize-1].position;
							playerHeightIndex = newPlayerWindow[newPlayerWindowSize-1].radiusIdx;
							playerPlacement = newPlayerWindow[newPlayerWindowSize-1].placement;
							cout << "newPlayerWindow[0].frameCount=" << newPlayerWindow[0].frameCount
							     << "newPlayerWindow[0].position=" << newPlayerWindow[0].position
								 << "   newPlayerWindow[0].radiusIdx=" << newPlayerWindow[0].radiusIdx
								 << "   newPlayerWindow[0].placement=" << newPlayerWindow[0].placement
								 << endl;
						}

						if (frameCount > 220) 
						{
								if ((newPlayerWindow[0].position.x == 0) || (newPlayerWindow[0].position.y == 0))
								{
									cout << "newPlayerWindow[0].frameCount=" << newPlayerWindow[0].frameCount
										 << "	 freezeCenter="		   		<< Point(freezeCenterX,freezeCenterY)
										 << "	 Chosen  newPlayerWindow[0].position="	<<	newPlayerWindow[0].position
										 << "	 hgtRings["					<< newPlayerWindow[0].radiusIdx
										 << "][" 							<< newPlayerWindow[0].placement
										 << "]=" 							<< hgtRings[newPlayerWindow[0].radiusIdx][newPlayerWindow[0].placement]
										 << endl;
									circle(bbsrc, hgtRings[newPlayerWindow[0].radiusIdx][newPlayerWindow[0].placement], 1, Scalar(0, 165, 255), 3); 
								}
								else
								{
									cout << frameCount << ":"
										 << "	 freezeCenter="		   		<< Point(freezeCenterX,freezeCenterY)
										 << "	 Chosen  newPlayerWindow[0].position="	<< newPlayerWindow[0].position
										 << "	 hgtRings["					<< newPlayerWindow[0].radiusIdx
										 << "]["							<< newPlayerWindow[0].placement
										 << "]="							<< hgtRings[newPlayerWindow[0].radiusIdx][newPlayerWindow[0].placement]
										 << endl;
									circle(bbsrc, hgtRings[newPlayerWindow[0].radiusIdx][newPlayerWindow[0].placement], 1, Scalar(0, 165, 255), 3);
								}
						}

						Point semiCircleCenterPt( (offsetBackboard.tl().x+offsetBackboard.width/2) , (offsetBackboard.tl().y + offsetBackboard.height/2) );
						halfCourtCenterPt = semiCircleCenterPt;
						
						if (!semiCircleReady) {
							int bCounter = 0;
							for (int radius=40; radius < 280; radius+= 20)   //Radius for distFromBB
							{
								hTableRange.push_back(radius);
								
								int temp1, temp2, temp3;
								int yval;
								for (int x=halfCourtCenterPt.x-radius; x<=halfCourtCenterPt.x+radius; x++) 
								{
									temp1 = radius * radius;
									temp2 = (x - halfCourtCenterPt.x) * (x - halfCourtCenterPt.x);
									temp3 = temp1 - temp2;
									yval = sqrt(temp3);
									yval += halfCourtCenterPt.y;
									Point ptTemp = Point(x, yval);
									hgtRings[bCounter][x] = ptTemp;
								}
								
								bCounter++;
							}
							semiCircleReady = true;
						}
					}
					//---Start of using player position on halfcourt image to draw shot location-----
					//---End of the process of identifying a shot at the basket!!!------------
					
				}				
			}
			///*******End of code to detect & select Basketball*************************
    	}

	    //-- detect body 
	    body_cascade.detectMultiScale(grayForRect, bodys, 1.1, 2, 18|9, Size(3,7));

		stringstream bodyheight;
		stringstream bodycenterX;
		stringstream bodycenterY;
		string posit_on_video;
		string bb_initial_inputs;
		string posit_on_chart;
		stringstream playerXStr;
		stringstream playerYStr;

	    for( int j = 0; j < bodys.size(); j++ ) 
        {
        	//-----------Identifying player height and position!!--------------
			Point bodyCenter( bodys[j].x + bodys[j].width*0.5, bodys[j].y + bodys[j].height*0.5 ); 
			
			//Sliding window for finding the average position of the player.
			for (int k = newPlayerWindowSize; k > 1; k--) 
			{
				newPlayerWindow[k-1].activeValue = newPlayerWindow[k-2].activeValue;
				newPlayerWindow[k-1].radiusIdx = newPlayerWindow[k-2].radiusIdx;
				newPlayerWindow[k-1].placement = newPlayerWindow[k-2].placement;
				newPlayerWindow[k-1].position = newPlayerWindow[k-2].position;
				newPlayerWindow[k-1].frameCount = newPlayerWindow[k-2].frameCount;
			}
			newPlayerWindow[0].frameCount = frameCount;
			newPlayerWindow[0].activeValue = 1;
			newPlayerWindow[0].position = bodyCenter; //playerNewPosit;					

			double distFromBB = euclideanDist((double) freezeCenterX,(double) freezeCenterY,(double) bodyCenter.x, (double) bodyCenter.y);
			double xDistFromBB = oneDDist(freezeCenterX, bodyCenter.x);
			double yDistFromBB = oneDDist(freezeCenterY, bodyCenter.y);

#ifdef SHOT_DEBUG 
			cout << "   frameCount=" 		<< frameCount
				 << "   freezeCenter="	    << Point(freezeCenterX, freezeCenterY)
				 << "   bodyCenter=" 		<< bodyCenter
			 	 << "   bodys[j].height="   << bodys[j].height 
				 << "   distFromBB=" 		<< distFromBB
				 << "   xDistFromBB="		<< xDistFromBB
				 << "   yDistFromBB="		<< yDistFromBB
				 << endl;
#endif

			if (distFromBB > 135) 
			{
				newPlayerWindow[0].radiusIdx = hTableRange.size() * 0.99;

				distFromBB += 120;

				cout << "frameCount="										<< frameCount
					 << "    halfCourtCenterPt.x="								<< halfCourtCenterPt.x
					 << "    hTableRange[newPlayerWindow[0].radiusIdx]="	<< hTableRange[newPlayerWindow[0].radiusIdx]
					 << endl;

				int tempPlacement = (halfCourtCenterPt.x + hTableRange[newPlayerWindow[0].radiusIdx])
								- (halfCourtCenterPt.x - hTableRange[newPlayerWindow[0].radiusIdx]);
				
				if (bodyCenter.x > freezeCenterX) tempPlacement -= 1;
				else tempPlacement = 0;
				tempPlacement += (halfCourtCenterPt.x - hTableRange[newPlayerWindow[0].radiusIdx]);

				newPlayerWindow[0].placement = tempPlacement;
			}
			else if (distFromBB < 30) 
			{
				int tempPlacement;
				if (bodyCenter.x < freezeCenterX) tempPlacement = 0;
			    else tempPlacement = (halfCourtCenterPt.x + hTableRange[newPlayerWindow[0].radiusIdx])
								- (halfCourtCenterPt.x - hTableRange[newPlayerWindow[0].radiusIdx]) - 1;
				
				newPlayerWindow[0].placement = tempPlacement;
				newPlayerWindow[0].radiusIdx = hTableRange.size() * 0.01;
			}
			else 
			{
			    if (bodys[j].height < 170)    //NOTE:  If not true, then we have inaccurate calculation of body height from detectMultiscale method.  Do not estimate a player position for it. 
				{
					newPlayerWindow[0].radiusIdx = findIndex_BSearch(hTableRange, distFromBB);
					newPlayerWindow[0].radiusIdx += 5;
					if ((xDistFromBB < 51) && (yDistFromBB < 70)) newPlayerWindow[0].radiusIdx = 0;
					cout << "frameCount="	<< frameCount
						 << "	 radiusIdx=" << newPlayerWindow[0].radiusIdx
						 << endl;
					
					double percentPlacement = (double) (bodyCenter.x - leftImgBoundary) / (rightImgBoundary - leftImgBoundary);
					int leftRingBound		= halfCourtCenterPt.x - hTableRange[newPlayerWindow[0].radiusIdx];
					int rightRingBound		= halfCourtCenterPt.x + hTableRange[newPlayerWindow[0].radiusIdx];
					int chartPlacementTemp	= (rightRingBound - leftRingBound) * percentPlacement;
					int chartPlacement		= leftRingBound + chartPlacementTemp;

					newPlayerWindow[0].placement = chartPlacement;
				}
			}			
			//--- End of adjusting player position on image of half court!!!-----

        } 
*/		

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


void getGray(const Mat& image, Mat& gray)
{
    if (image.channels()  == 3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;
}

double DistanceToCamera(double knownWidth, double focalLength, double perWidth) {
	return ((knownWidth * focalLength)/ perWidth);
}

int getPlayerOffsetX(int frame_count, float player_pos_x, float player_pos_y, int player_hgt)
{
	float numer, denom, res;
    if (player_hgt <= 50) {
		return (42);
    }
	else if (player_hgt > 50 && player_hgt <= 60) {
		numer = (float)(player_hgt - 50) * (75 - 42);
		denom = (float)(60 - 50);
		res = numer / denom;
		res += 42;
		if (res < 42) res = 42;
		return (res);	
	}
	else if (player_hgt > 60 && player_hgt <= 70) {
		numer = (float)(player_hgt - 60) * (125 - 75);
		denom = (float)(70 - 60);
		res = numer / denom;
		res += 75;
		return (res);	
	}	
	else if (player_hgt > 70 && player_hgt <= 80) {
		numer = (float)(player_hgt - 70) * (185 - 125);
		denom = (float)(80 - 70);
		res = numer / denom;
		res += 125;
		return (res);	
	}
	else if (player_hgt > 80 && player_hgt <= 90) {
		numer = (float) (player_hgt - 80) * (230 - 185);
		denom = (float) (90 - 80);
		res = numer / denom;
		res += 185;
		return (res);
	}
	else if (player_hgt > 90 && player_hgt <= 100) {
		numer = (float) (player_hgt - 90) * (266 - 230);
		denom = (float) (100 - 90);
		res = numer / denom;
		res += 230;
		return(res);
	}
	else {
		return (270.0);
	}
}

int getPlayerOffsetY(int frame_count, float player_pos_x, float player_pos_y, int player_hgt)
{
	float numer, denom, res;
    if (player_hgt <= 50) {
		//printf("%d:Height Range[... 50]  Expect Y Range[... 42] res=42 \n", frame_count);
		return (42);
    }
	else if (player_hgt > 50 && player_hgt <= 60) {
		numer = (float)(player_hgt - 50) * (75 - 42);
		denom = (float)(60 - 50);
		res = numer / denom;
		res += 42;
		if (res < 42) res = 42;
		//printf("%d:Height Range[50 ... 60]  Expect Y Range[42 ... 75] res=%f \n", frame_count, res);
		return (res);	
	}
	else if (player_hgt > 60 && player_hgt <= 70) {
		numer = (float)(player_hgt - 60) * (125 - 75);
		denom = (float)(70 - 60);
		res = numer / denom;
		res += 75;
		//printf("%d:Height Range[60 ... 70]  Expect Y Range[75 ... 125] res=%f \n", frame_count, res);
		return (res);	
	}	
	else if (player_hgt > 70 && player_hgt <= 80) {
		numer = (float)(player_hgt - 70) * (185 - 125);
		denom = (float)(80 - 70);
		res = numer / denom;
		res += 125;
		//printf("%d:Height Range[70 ... 80]  Expect Y Range[125 ... 185] res=%f \n", frame_count, res);
		return (res);	
	}
	else if (player_hgt > 80 && player_hgt <= 90) {
		numer = (float) (player_hgt - 80) * (230 - 185);
		denom = (float) (90 - 80);
		res = numer / denom;
		res += 185;
		//printf("%d:Height Range[80 ... 90]  Expect Y Range[185 ... 230] res=%f \n", frame_count, res);
		return (res);
	}
	else if (player_hgt > 90 && player_hgt <= 100) {
		numer = (float) (player_hgt - 90) * (266 - 230);
		denom = (float) (100 - 90);
		res = numer / denom;
		res += 230;
		//printf("%d:Height Range[90 ... 100]  Expect Y Range[230 ... 266] res=%f \n", frame_count, res);
		return(res);
	}
	else if (player_hgt > 100 && player_hgt <= 110) {
		numer = (float) (player_hgt - 90) * (266 - 230);
		denom = (float) (100 - 90);
		res = numer / denom;
		res += 230;
		//printf("%d:Height Range[90 ... 100]  Expect Y Range[230 ... 266] res=%f \n", frame_count, res);
		return(res);
	}
	else {
		res = 270.0;
		//printf("%d:Height Range[100 ...]  Expect Y Range[266 ..0] res=%f \n", frame_count, res);
		return (res);
	}
}

vector<Rect> get_sliding_windows(Mat& image,int winWidth,int winHeight)
{
  vector<Rect> rects;
  int step = 16;
  for(int i=0;i<image.rows;i+=step)
  {
      if((i+winHeight)>image.rows){break;}
      for(int j=0;j< image.cols;j+=step)    
      {
          if((j+winWidth)>image.cols){break;}
          Rect rect(j,i,winWidth,winHeight);
          rects.push_back(rect);
      }
  } 
  return rects;
}
Mat drawSemiCircle(Mat& image, int radius, Point center) {
	int temp1, temp2, temp3;
	int yval;
	
	for (int x=center.x-radius; x<=center.x+radius; x++) 
	{
		temp1 = radius * radius;
		temp2 = (x - center.x) * (x - center.x);
		temp3 = temp1 - temp2;
		yval = sqrt(temp3);
		yval += center.y;			
		circle(image, Point(x, yval), 1, Scalar(0,255,0), -1);			
	}
	return image;
}

double euclideanDist(double x1, double y1, double x2, double y2)
{
	double x = x1 - x2; //calculating number to square in next step
	double y = y1 - y2;
	double dist;

	dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
	dist = sqrt(dist);                  

	return dist;
}

double oneDDist(double p1, double p2) {
	double dist;
	
	double p = p1 - p2;
	dist = pow(p, 2);
	dist = sqrt(dist);
	
	return dist;
}
int findIndex_BSearch(const vector< int> &numbersArray, int key) {

	int iteration = 0;
	int left = 0;
	int right = numbersArray.size()-1;
	int mid;

	while (left <= right) {
		iteration++;
		mid = (int) ((left + right) / 2);
		if (key <= numbersArray[mid]) 
		{
			right = mid - 1;
		}
		else if (key > numbersArray[mid])
		{
			left = mid + 1;
		}
	}
	return (mid);
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

void generateCourtRings(Rect offsetBackboard, vector<int> mHTableRange, Point** mHgtRings)
{
	Point semiCircleCenterPt( (offsetBackboard.tl().x+offsetBackboard.width/2) , (offsetBackboard.tl().y + offsetBackboard.height/2) );
	Point halfCourtCenterPt = semiCircleCenterPt;
	
	int bCounter = 0;
	for (int radius=40; radius < 280; radius+= 20)	 //Radius for distFromBB
	{
		mHTableRange.push_back(radius);
		
		int temp1, temp2, temp3;
		int yval;
		for (int x=halfCourtCenterPt.x-radius; x<=halfCourtCenterPt.x+radius; x++) 
		{
			temp1 = radius * radius;
			temp2 = (x - halfCourtCenterPt.x) * (x - halfCourtCenterPt.x);
			temp3 = temp1 - temp2;
			yval = sqrt(temp3);
			yval += halfCourtCenterPt.y;
			Point ptTemp = Point(x, yval);
			mHgtRings[bCounter][x] = ptTemp;
		}
		
		bCounter++;
	}
}

