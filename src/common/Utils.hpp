/******
 *  Utils.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#ifndef UTILS_HPP_
#define UTILS_HPP_
#include <stdio.h>
	 // Includes:
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
	 
using namespace std;
using namespace cv;
	 
class Utils
{

public:
	double DistanceToCamera(double knownWidth, double focalLength, double perWidth);
	int getPlayerOffsetX(int frame_count, float player_pos_x, float player_pos_y, int player_hgt);
	int getPlayerOffsetY(int frame_count, float player_pos_x, float player_pos_y, int player_hgt);
	Mat drawSemiCircle(Mat& image, int radius, Point center);
	double euclideanDist(double x1, double y1, double x2, double y2);
	double oneDDist(double p1, double p2);
	int findIndex_BSearch(const vector< int> &my_numbers, int key);
	void getGray(const Mat& image, Mat& gray);
};
#endif

