#pragma once
//Standard Includes
#include <stdio.h>
//OpenCV Includes
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "PlayerInfo.hpp"

using namespace std;
using namespace cv;

class CourtPositionEstimator
{
public:
	CourtPositionEstimator(
		CascadeClassifier& body_cascade, 
		PlayerInfo& pInfo,
		int& leftImgBoundary,
		int& rightImgBoundary,
		vector<int>& hTableRange);
public:
	PlayerInfo findBody(const Mat& computeFrame);
	double euclideanDist(double x1, double y1, double x2, double y2);
	double oneDDist(double p1, double p2);
	int findIndex_BSearch(const vector< int> &numbersArray, int key);
	void setFreezeBBPoint(Point freeze_point);
	void setHalfCourtCenterPt(Point halfcourt_centerpt);
//	PlayerInfo estimateCourtPosition();
public:
	CascadeClassifier mClassifier;
	Point mHalfCourtCenterPt;
	PlayerInfo mPlayerInfo;
	Point mFreezeCenterPt;
	int mLeftImgBoundary;
	int mRightImgBoundary;
	vector<int> mHTableRange;
	vector<Rect> mVectorOfBodys;		
};

