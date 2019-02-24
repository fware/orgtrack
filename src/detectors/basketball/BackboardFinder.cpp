/******
 *  InitialFrameProcessor.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#include "BackboardFinder.hpp"

using namespace std;
using namespace cv;


BackboardFinder::BackboardFinder(int t, int file_number, cv::Size imgSize) : m_thresh(t), m_filenumber(file_number), m_imgsize(imgSize)//, m_bRF()
{
	extern_logLevel = logINFO;
	m_firstpass = true;
}

Rect BackboardFinder::process(const Mat& grayImage, int leftBBRegionLimit, int rightBBRegionLimit, int bottomBBRegionLimit)
{
	Mat threshold_output;
	vector<Vec4i> hierarchy;
	vector< vector<Point> > boardContours;
	
	blur(grayImage, grayImage, Size(3,3));								//Blurs, i.e. smooths, an image using the normalized box filter.  Used to reduce noise.
	//equalizeHist(grayImage, grayImage);									//Equalizes the histogram of the input image.  Normalizes the brightness and increases the contrast of the image.
	Canny(grayImage, grayImage, m_thresh, m_thresh*2, 3);
	findContours( grayImage, boardContours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
	vector< vector<Point> > contours_poly( boardContours.size() );
	vector<Rect> boundRect( boardContours.size() );
	for ( size_t i = 0; i < boardContours.size(); i++ )
	{
		approxPolyDP(Mat(boardContours[i]),contours_poly[i],3,true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));

		double bb_w = (double) boundRect[i].size().width;
		double bb_h = (double) boundRect[i].size().height;
		double bb_ratio = (double) bb_w / bb_h;
		if ( (boundRect[i].x > leftBBRegionLimit)
			  && (boundRect[i].x < rightBBRegionLimit)
			  && (boundRect[i].x + boundRect[i].width < rightBBRegionLimit)
			  && (boundRect[i].y < bottomBBRegionLimit)
			  && (boundRect[i].area() > 50)
			  && (bb_ratio < 1.3)
			  && (bb_w > (bb_h * 0.74) ) )
		{
			if (m_firstpass)
			{
				m_backboard = boundRect[i];
				m_firstpass = false;
			}
			else
			{
				m_backboard = m_backboard | boundRect[i];
			}
		}
	}


	/*
	threshold(grayImage, threshold_output, m_thresh, 255, THRESH_BINARY);	//Fixed-level thresholding.  Used here to produce a bi-level image.  Can also be used to remove noise.
	findContours( threshold_output, boardContours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );	//Finds contours in binary image. Contours are useful for shape analysis																													//and object detection & recognition.
	Rect backBoardRect = m_bRF.getBoundRects(grayImage, threshold_output, hierarchy, boardContours, m_filenumber);
	*/

	log(logDEBUG4) << "End of initialProcessFrame";

	return m_backboard;  //backBoardRect;//m_bRF.getBoundRect(threshold_output, hierarchy, boardContours);
}



Rect BackboardFinder::getBBOffset()
{
	//rectangle(img, unionRect.tl(), unionRect.br(), Scalar(0,255,0), 2, 8, 0);
	m_backboardoffsetx = -m_backboard.tl().x + m_imgsize.width/2 - 13;
	m_backboardoffsety = -m_backboard.tl().y + 30;
	m_offsetbackboard = Rect(m_backboard.tl().x+m_backboardoffsetx,
								m_backboard.tl().y+m_backboardoffsety,
								m_backboard.size().width,
								m_backboard.size().height);

	Point semiCircleCenterPt( (m_offsetbackboard.tl().x+m_offsetbackboard.width/2) , (m_offsetbackboard.tl().y + m_offsetbackboard.height/2) );
	m_bbcenterposit = semiCircleCenterPt;

	m_backboardcenterx = (m_backboard.tl().x+(m_backboard.width/2));
	m_backboardcentery = (m_backboard.tl().y+(m_backboard.height/2));

	return m_offsetbackboard; //m_bRF.getBBOffset();
}

void BackboardFinder::getGray(const Mat& image, Mat& gray)
{
    if (image.channels()  == 3)
        cv::cvtColor(image, gray, COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, COLOR_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;
}



