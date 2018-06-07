/******
 *  Logger.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include <iostream>
#include <sstream>
#include <map>
#include <string>
#include <cstdio>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;

enum logLevel_e {
	logERROR, logWARN, logINFO, logDEBUG, logDEBUG1, logDEBUG2, logDEBUG3, logDEBUG4
};

class Logger
{
public:
	logLevel_e extern_logLevel;

	Logger(logLevel_e logLevel = logDEBUG2)
	{
		static map<logLevel_e, string> strings;
		if (strings.size() == 0){
#define INSERT_ELEMENT(p) strings[p] = #p
			INSERT_ELEMENT(logERROR);
			INSERT_ELEMENT(logWARN);
			INSERT_ELEMENT(logINFO);
			INSERT_ELEMENT(logDEBUG);
			INSERT_ELEMENT(logDEBUG1);
			INSERT_ELEMENT(logDEBUG2);
			INSERT_ELEMENT(logDEBUG3);
			INSERT_ELEMENT(logDEBUG4);
		}
		buffer << strings[logLevel] << " :" << string(logLevel > logDEBUG ? (logLevel - logDEBUG) * 4 : 1, ' ');

		//buffer << logLevel << " :" << string(logLevel > logDEBUG ? (logLevel - logDEBUG) * 4 : 1, ' ');
	}

	~Logger()
	{
		buffer << endl;
		cerr << buffer.str();
	}

	template <typename T>
	Logger & operator<<(T const & value)
	{
		buffer << value;
		return *this;
	}

	// Does lexical cast of the input argument to string
	template <typename T>
	std::string ToString(const T& value)
	{
	    std::ostringstream stream;
	    stream << value;
	    return stream.str();
	}

	// This function used to show and save the image to the disk (used for during chapter writing).
	inline void showAndSave(std::string name, const cv::Mat& m)
	{
		imshow(name, m);
		imwrite(name + ".png", m);
		//cv::waitKey(25);
	}

	// Draw matches between two images
	inline cv::Mat getMatchesImage(cv::Mat query, cv::Mat pattern, const std::vector<cv::KeyPoint>& queryKp, const std::vector<cv::KeyPoint>& trainKp, std::vector<cv::DMatch> matches, size_t maxMatchesDrawn)
	{
		cv::Mat outImg;

		if (matches.size() > maxMatchesDrawn)
		{
			matches.resize(maxMatchesDrawn);
		}

		cv::drawMatches
			(
			query,
			queryKp,
			pattern,
			trainKp,
			matches,
			outImg,
			cv::Scalar(0,200,0,255),
			cv::Scalar::all(-1),
			std::vector<char>(),
			cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
			);

		return outImg;
	}

private:
	ostringstream buffer;
};

extern logLevel_e extern_logLevel;

#define log(level) \
	if (level > extern_logLevel) ; \
	else Logger(level)

#endif
