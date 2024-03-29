/******
 *  CourtPositionEstimator.cpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#include "CourtPositionEstimator.hpp"
#include "PlayerInfo.hpp"
#include "Logger.hpp"

using namespace std;
using namespace cv;

string TAG = "CourtPositionEstimator";

CourtPositionEstimator::CourtPositionEstimator(
	CascadeClassifier& body_cascade, 
	PlayerInfo& pInfo,
	int& leftImgBoundary,
	int& rightImgBoundary)   //, vector<int>& radiusArray)
	: mClassifier(body_cascade), 
	  mPlayerInfo(pInfo),
	  mLeftImgBoundary(leftImgBoundary),
	  mRightImgBoundary(rightImgBoundary)  //,
//	  mRadiusArray(radiusArray)
{
	extern_logLevel = logINFO;
}

PlayerInfo CourtPositionEstimator::findBody(int count, const Mat & grayFrame, Mat& image)
{
	//-- detect body
	mClassifier.detectMultiScale(grayFrame, mVectorOfBodies, 1.1, 2, 18|9, Size(3,7));

	for( size_t j = 0; j < mVectorOfBodies.size(); j++ )
	{
		stringstream heightS;
		heightS << mVectorOfBodies[j].height;
		string heightStr = "mVectorOfBodys[j].height=" + heightS.str();
		//log(logINFO) << heightStr;
		stringstream widthS;
		widthS << mVectorOfBodies[j].width;
		string widthStr = "mVectorOfBodys[j].width=" + widthS.str();
		//log(logINFO) << widthStr;
		//-----------Identifying player height and position!!--------------
		Point bodyCenter( mVectorOfBodies[j].x + mVectorOfBodies[j].width*0.5, mVectorOfBodies[j].y + mVectorOfBodies[j].height*0.5 ); 
		
		mPlayerInfo.activeValue = 1;
		mPlayerInfo.position = bodyCenter;

		double distFromBB = euclideanDist((double) mBackboardCenterPt.x,(double) mBackboardCenterPt.y,(double) bodyCenter.x, (double) bodyCenter.y);
		double xDistFromBB = oneDDist(mBackboardCenterPt.x, bodyCenter.x);
		double yDistFromBB = oneDDist(mBackboardCenterPt.y, bodyCenter.y);

		//log(logINFO) << distFromBB;
		//log(logINFO) << xDistFromBB;
		//log(logINFO) << yDistFromBB;

		if (distFromBB > 135) 
		{
			//log(logINFO) << count;
			//log (logINFO) << "distFromBB > 135";
			mPlayerInfo.radiusIdx = mRadiusArray.size() * 0.99;

			distFromBB += 120;

			int tempPlacement = (mHalfCourtCenterPt.x + mRadiusArray[mPlayerInfo.radiusIdx])
							- (mHalfCourtCenterPt.x - mRadiusArray[mPlayerInfo.radiusIdx]);
			
			if (bodyCenter.x > mBackboardCenterPt.x) tempPlacement -= 1;
			else tempPlacement = 0;
			tempPlacement += (mHalfCourtCenterPt.x - mRadiusArray[mPlayerInfo.radiusIdx]);

			mPlayerInfo.placement = tempPlacement;
		}
		else if (distFromBB < 30) 
		{
			//log(logINFO) << "distFromBB < 30";
			int tempPlacement;
			if (bodyCenter.x < mBackboardCenterPt.x) tempPlacement = 0;
			else tempPlacement = (mHalfCourtCenterPt.x + mRadiusArray[mPlayerInfo.radiusIdx])
							- (mHalfCourtCenterPt.x - mRadiusArray[mPlayerInfo.radiusIdx]) - 1;
			
			mPlayerInfo.placement = tempPlacement;
			mPlayerInfo.radiusIdx = mRadiusArray.size() * 0.01;
		}
		else 
		{
			if (mVectorOfBodies[j].height < 170)	  //NOTE:  If not true, then we have inaccurate calculation of body height from detectMultiscale method.  Do not estimate a player position for it. 
			{
				//log("height is < 170");
				mPlayerInfo.radiusIdx = findIndex_BSearch(mRadiusArray, distFromBB);
				stringstream radiusS;
				radiusS << mPlayerInfo.radiusIdx;
				string radiusStr = "mPlayerInfo.radiusIdx=" + radiusS.str();
				//log(radiusStr);
				mPlayerInfo.radiusIdx += 5;
				if ((xDistFromBB < 51) && (yDistFromBB < 70)) mPlayerInfo.radiusIdx = 0;
				
				double percentPlacement = (double) (bodyCenter.x - mLeftImgBoundary) / (mRightImgBoundary - mLeftImgBoundary);
				int leftRingBound		= mHalfCourtCenterPt.x - mRadiusArray[mPlayerInfo.radiusIdx];
				int rightRingBound		= mHalfCourtCenterPt.x + mRadiusArray[mPlayerInfo.radiusIdx];
				int chartPlacementTemp	= (rightRingBound - leftRingBound) * percentPlacement;
				int chartPlacement		= leftRingBound + chartPlacementTemp;

				mPlayerInfo.placement = chartPlacement;
			}
		}			
		//--- End of adjusting player position on image of half court!!!-----

		log(logDEBUG4) << ": " << count << " return 1: End of findBody";
		return mPlayerInfo;

		ellipse( image, bodyCenter, Size( mVectorOfBodies[j].width*0.5, mVectorOfBodies[j].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
	}

	log(logDEBUG4) << ": " << count << " return 2: End of findBody";
	return mPlayerInfo;

	//return mVectorOfBodys;		
}

double CourtPositionEstimator::euclideanDist(double x1, double y1, double x2, double y2)
{
	double x = x1 - x2; //calculating number to square in next step
	double y = y1 - y2;
	double dist;

	dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
	dist = sqrt(dist);                  

	return dist;
}

double CourtPositionEstimator::oneDDist(double p1, double p2) {
	double dist;
	
	double p = p1 - p2;
	dist = pow(p, 2);
	dist = sqrt(dist);
	
	return dist;
}

int CourtPositionEstimator::findIndex_BSearch(const vector< int> &numbersArray, int value) {

	int iteration = 0;
	int left = 0;
	int right = numbersArray.size()-1;
	int mid;

	while (left <= right) {
		iteration++;
		mid = (int) ((left + right) / 2);
		if (value <= numbersArray[mid])
		{
			right = mid - 1;
		}
		else if (value > numbersArray[mid])
		{
			left = mid + 1;
		}
	}
	return (mid);
}

void CourtPositionEstimator::setBackboardPoint(Point freeze_point)
{
	mBackboardCenterPt = freeze_point;
}

void CourtPositionEstimator::setHalfCourtCenterPt(Point halfcourt_centerpt)
{
	mHalfCourtCenterPt = halfcourt_centerpt;
}

void CourtPositionEstimator::setRadiusArray(vector<int> &hTableRange)
{
	mRadiusArray = hTableRange;
}


/*
PlayerInfo CourtPositionEstimator::estimateCourtPosition()
{
	for( int j = 0; j < mVectorOfBodys.size(); j++ ) 
	{
		//-----------Identifying player height and position!!--------------
		Point bodyCenter( mVectorOfBodys[j].x + mVectorOfBodys[j].width*0.5, mVectorOfBodys[j].y + mVectorOfBodys[j].height*0.5 ); 
		
		mPlayerInfo.activeValue = 1;
		mPlayerInfo.position = bodyCenter;

		double distFromBB = euclideanDist((double) mFreezeCenterPt.x,(double) mFreezeCenterPt.y,(double) bodyCenter.x, (double) bodyCenter.y);
		double xDistFromBB = oneDDist(mFreezeCenterPt.x, bodyCenter.x);
		double yDistFromBB = oneDDist(mFreezeCenterPt.y, bodyCenter.y);

		if (distFromBB > 135) 
		{
			mPlayerInfo.radiusIdx = mHTableRange.size() * 0.99;

			distFromBB += 120;

			int tempPlacement = (mHalfCourtCenterPt.x + mHTableRange[mPlayerInfo.radiusIdx])
							- (mHalfCourtCenterPt.x - mHTableRange[mPlayerInfo.radiusIdx]);
			
			if (bodyCenter.x > mFreezeCenterPt.x) tempPlacement -= 1;
			else tempPlacement = 0;
			tempPlacement += (mHalfCourtCenterPt.x - mHTableRange[mPlayerInfo.radiusIdx]);

			mPlayerInfo.placement = tempPlacement;
		}
		else if (distFromBB < 30) 
		{
			int tempPlacement;
			if (bodyCenter.x < mFreezeCenterPt.x) tempPlacement = 0;
			else tempPlacement = (mHalfCourtCenterPt.x + mHTableRange[mPlayerInfo.radiusIdx])
							- (mHalfCourtCenterPt.x - mHTableRange[mPlayerInfo.radiusIdx]) - 1;
			
			mPlayerInfo.placement = tempPlacement;
			mPlayerInfo.radiusIdx = mHTableRange.size() * 0.01;
		}
		else 
		{
			if (mVectorOfBodys[j].height < 170)	  //NOTE:  If not true, then we have inaccurate calculation of body height from detectMultiscale method.  Do not estimate a player position for it. 
			{
				mPlayerInfo.radiusIdx = findIndex_BSearch(mHTableRange, distFromBB);
				mPlayerInfo.radiusIdx += 5;
				if ((xDistFromBB < 51) && (yDistFromBB < 70)) mPlayerInfo.radiusIdx = 0;
				
				double percentPlacement = (double) (bodyCenter.x - mLeftImgBoundary) / (mRightImgBoundary - mLeftImgBoundary);
				int leftRingBound		= mHalfCourtCenterPt.x - mHTableRange[mPlayerInfo.radiusIdx];
				int rightRingBound		= mHalfCourtCenterPt.x + mHTableRange[mPlayerInfo.radiusIdx];
				int chartPlacementTemp	= (rightRingBound - leftRingBound) * percentPlacement;
				int chartPlacement		= leftRingBound + chartPlacementTemp;

				mPlayerInfo.placement = chartPlacement;
			}
		}			
		//--- End of adjusting player position on image of half court!!!-----

		return mPlayerInfo;

		//ellipse( img, bodyCenter, Size( mVectorOfBodys[j].width*0.5, mVectorOfBodys[j].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
	} 
}
*/
