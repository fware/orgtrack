/******
 *  InitialFrameProcessor.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#include "InitialFrameProcessor.hpp"

using namespace std;
using namespace cv;


InitialFrameProcessor::InitialFrameProcessor(int t, int file_number) : m_thresh(t), m_filenumber(file_number) //, m_bRF()
{
	extern_logLevel = logDEBUG2;
}

Rect InitialFrameProcessor::initialProcessFrame(const Mat& grayImage)
{
	Mat threshold_output;
	vector<Vec4i> hierarchy;
	vector< vector<Point> > boardContours;
	
	blur(grayImage, grayImage, Size(3,3));								//Blurs, i.e. smooths, an image using the normalized box filter.  Used to reduce noise.
	equalizeHist(grayImage, grayImage);									//Equalizes the histogram of the input image.  Normalizes the brightness and increases the contrast of the image.
	threshold(grayImage, threshold_output, m_thresh, 255, THRESH_BINARY);	//Fixed-level thresholding.  Used here to produce a bi-level image.  Can also be used to remove noise.
	findContours( threshold_output, boardContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	//Finds contours in binary image. Contours are useful for shape analysis																													//and object detection & recognition.

	Rect backBoardRect = m_bRF.getBoundRects(grayImage, threshold_output, hierarchy, boardContours, m_filenumber);

	log(logDEBUG) << "End of initialProcessFrame";

	return backBoardRect;//m_bRF.getBoundRect(threshold_output, hierarchy, boardContours);
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



