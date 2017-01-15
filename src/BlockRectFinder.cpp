/*****************************************************************************
*   Copyright 2016.
*****************************************************************************/

////////////////////////////////////////////////////////////////////
// File includes:
#include "BlockRectFinder.hpp"

BlockRectFinder::BlockRectFinder() : mHaveBackboard(false)
{
}

Rect BlockRectFinder::getBoundRects(const Mat& imageFrame, 
										Mat& threshold_output, 
										vector<Vec4i> hierarchy, 
										vector< vector<Point> > boardContours,
										int file_number) {
	findContours( threshold_output, boardContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	
	vector< vector<Point> > contours_poly( boardContours.size() );
	vector<Rect> boundRects( boardContours.size() );
	//vector<Point2f> ccenter( boardContours.size() );
	//vector<float> rradius( boardContours.size() );
	
	///*************************Start of main code to detect BackBoard************************* 	
	for ( int i = 0; i < boardContours.size(); i++ ) {
		approxPolyDP(Mat(boardContours[i]),contours_poly[i],3,true);
		boundRects[i] = boundingRect(Mat(contours_poly[i]));
	}

	Rect backBoardRect = findBackboard(imageFrame, boardContours.size(), boundRects, file_number);

	return backBoardRect;
	
}

Rect BlockRectFinder::findBackboard(const Mat& imgFrame, 
										size_t bbContoursSize, 
										vector<Rect> bound_Rects,
										int file_number) {

	double bb_ratio = 0.0;
	double bb_w = 0.0;
	double bb_h = 0.0;
	double bb_area = 0.0;
	int bb_x, bb_y;

	//-----------Find the Backboard!!!-----------------
	for (size_t j = 0; j < bbContoursSize; j++) 
	{
		bb_w = (double) bound_Rects[j].size().width;
		bb_h = (double) bound_Rects[j].size().height;
		///cout << j << ": "  <<
		//	"bb_w= " << bb_w
		//	<< "  bb_h= " << bb_h << endl;
		bb_ratio = bb_w/bb_h;

		if((bound_Rects[j].area() > 700)
			&& (bound_Rects[j].area() < 900)
			&& (bb_ratio > 1.50) 
			&& (bb_ratio < 2.00)) {

			if (file_number <= 3) 
			{
				if (bound_Rects[j].tl().x < imgFrame.size().width/2) 
				{
					if (!mHaveBackboard) 
					{
						mBackBoardOffsetX = -bound_Rects[j].tl().x + imgFrame.size().width/2 - 13;
						mBackBoardOffsetY = -bound_Rects[j].tl().y + 30;
						mOffsetBackboard = Rect(bound_Rects[j].tl().x+mBackBoardOffsetX, 
										bound_Rects[j].tl().y+mBackBoardOffsetY, 
										bound_Rects[j].size().width,
										bound_Rects[j].size().height);
						mFreezeBB = bound_Rects[j];
						return mFreezeBB;
					}
					mHaveBackboard = true;
					mFreezeCenterX = (mFreezeBB.tl().x+(mFreezeBB.width/2));
					mFreezeCenterY = (mFreezeBB.tl().y+(mFreezeBB.height/2));
				}
			}
			else if (file_number == 4) 
			{
				if (bound_Rects[j].tl().x > imgFrame.size().width/2) 
				{
					if (!mHaveBackboard) 
					{
						//----------Compute the offset for backboard on shot chart!!!---------------
						mBackBoardOffsetX = -bound_Rects[j].tl().x + imgFrame.size().width/2 - 13;
						mBackBoardOffsetY = -bound_Rects[j].tl().y + 30;
						mOffsetBackboard = Rect(bound_Rects[j].tl().x+mBackBoardOffsetX, 
										bound_Rects[j].tl().y+mBackBoardOffsetY, 
										bound_Rects[j].size().width,
										bound_Rects[j].size().height);
						//----------We chose our background and put it in freezeBB!!!--------------
						mFreezeBB = bound_Rects[j];
						return mFreezeBB;
					}
					mHaveBackboard = true;
					mFreezeCenterX = (mFreezeBB.tl().x+(mFreezeBB.width/2));
					mFreezeCenterY = (mFreezeBB.tl().y+(mFreezeBB.height/2));
				}
			}
		}
	}
	
	return mFreezeBB;

}


Rect BlockRectFinder::getBBOffset()
{
	return mOffsetBackboard;
}

