/******
 *  ShotEstimator.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2019
 *
 */
#ifndef SHOTESTIMATOR_HPP_
#define SHOTESTIMATOR_HPP_

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdio.h>
// Includes:
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/shape_utils.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Logger.hpp"


struct object_results
{
	cv::String labelName;
	float confidence;
	int xLeftBottom;
    int yLeftBottom;
    int xRightTop;
    int yRightTop;
};

class ShotEstimator
{

public:
	ShotEstimator(cv::String modelConfig, cv::String modelBinary, cv::String modelClasses);
public:
	int initialize();
	object_results process(Mat& basket_region_image);
	//shot_results getResults();
private:
	logLevel_e extern_logLevel;
	cv::String m_modelconfig;
	cv::String m_modelbinary;
	cv::String m_modelclasses;
	dnn::Net m_net;
	vector<string> m_classnamesvec;
};

#endif
