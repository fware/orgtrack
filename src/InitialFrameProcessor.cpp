/*****************************************************************************
*   WareShop Consulting LLC
*   Copyright 2016.
*****************************************************************************/
#include "InitialFrameProcessor.hpp"

using namespace std;
using namespace cv;


InitialFrameProcessor::InitialFrameProcessor(int t, int file_number) : m_thresh(t), m_filenumber(file_number) //, m_bRF()
{
}

Rect InitialFrameProcessor::initialProcessFrame(const Mat& img)
{
	Mat threshold_output;
	vector<Vec4i> hierarchy;
	vector< vector<Point> > boardContours;
	
	getGray(img,grayForRect);
	blur(grayForRect, grayForRect, Size(3,3));
	equalizeHist(grayForRect, grayForRect);
	threshold(grayForRect, threshold_output, m_thresh, 255, THRESH_BINARY);
	findContours( threshold_output, boardContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	Rect backBoardRect = m_bRF.getBoundRects(img, threshold_output, hierarchy, boardContours, m_filenumber);

	return backBoardRect;//m_bRF.getBoundRect(threshold_output, hierarchy, boardContours);
}

Mat InitialFrameProcessor::getGrayForRect()
{
	return grayForRect;
}

Rect InitialFrameProcessor::getBBOffset() 
{
	return m_bRF.getBBOffset();
}

void InitialFrameProcessor::getGray(const Mat& image, Mat& gray)
{
    if (image.channels()  == 3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;
}



