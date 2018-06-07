/******
 *  CourtPositionEstimator.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#ifndef COURTPOSITIONESTIMATOR_HPP_
#define COURTPOSITIONESTIMATOR_HPP_
#pragma once
//Standard Includes
#include <stdio.h>
#include <iostream>
//OpenCV Includes
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "PlayerInfo.hpp"
#include "Utils.hpp"
#include "Logger.hpp"

using namespace std;
using namespace cv;

class CourtPositionEstimator
{
public:
	CourtPositionEstimator(
		CascadeClassifier& body_cascade, 
		PlayerInfo& pInfo,
		int& leftImgBoundary,
		int& rightImgBoundary);      //,vector<int>& hTableRange);
public:
	PlayerInfo findBody(int count, const Mat& computeFrame, Mat& image);
	double euclideanDist(double x1, double y1, double x2, double y2);
	double oneDDist(double p1, double p2);
	int findIndex_BSearch(const vector< int> &numbersArray, int key);
	void setBackboardPoint(Point freeze_point);
	void setHalfCourtCenterPt(Point halfcourt_centerpt);
	void setRadiusArray(vector<int>& hTableRange);
//	PlayerInfo estimateCourtPosition();
public:
	logLevel_e extern_logLevel;
	Mat grayFrame;
	Utils utils;
	CascadeClassifier mClassifier;
	Point mHalfCourtCenterPt;
	PlayerInfo mPlayerInfo;
	Point mBackboardCenterPt;
	int mLeftImgBoundary;
	int mRightImgBoundary;
	vector<int> mRadiusArray;
	vector<Rect> mVectorOfBodies;		
};
#endif
