/******
 *  InitialFrameProcessor.hpp
 *  Author:  WareShop Consulting LLC
 *
 *  Copyright 2016
 *
 */
#include "ShotEstimator.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

ShotEstimator::ShotEstimator(cv::String modelConfig, cv::String modelBinary, cv::String modelClasses)
{
	extern_logLevel = logINFO;
	m_modelconfig = modelConfig;
	m_modelbinary = modelBinary;
	m_modelclasses = modelClasses;
}

int ShotEstimator::initialize()
{
	m_net = readNetFromDarknet(m_modelconfig, m_modelbinary);
	if (m_net.empty())
	{
		log(logERROR) << "dnn model is empty.";
		return -1;
	}

    ifstream classNamesFile(m_modelclasses);
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            m_classnamesvec.push_back(className);
    }
	return 0;
}

object_results ShotEstimator::process(Mat& basket_region_image)
{
	log(logINFO) << __FILE__  << " " << __FUNCTION__ << " Bug 1";
	stringstream ss;

    //! [Prepare blob]
	log(logINFO) << __FILE__  << " basket_region_image[" << basket_region_image.cols << " x " << basket_region_image.rows <<  "] "<< __FUNCTION__ << " Bug 2";
    Mat inputBlob = blobFromImage(basket_region_image, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
    //! [Prepare blob]

    //! [Set input blob]
	log(logINFO) << __FILE__  << " " << __FUNCTION__ << " Bug 3";
    m_net.setInput(inputBlob, "data");                   //set the network input
    //! [Set input blob]

    //! [Make forward pass]
	log(logINFO) << __FILE__  << " " << __FUNCTION__ << " Bug 4";
    Mat detectionMat = m_net.forward("detection_out");   //compute output

	log(logINFO) << __FILE__  << " " << __FUNCTION__ << " Bug 5";
    object_results results;

	log(logINFO) << __FILE__  << " " << __FUNCTION__ << " Bug 6";
    for (int i = 0; i < detectionMat.rows; i++)
    {
        const int probability_index = 5;
        const int probability_size = detectionMat.cols - probability_index;
        float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

        size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
        float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

        if (confidence > 0.24)
        {
            float x = detectionMat.at<float>(i, 0);
            float y = detectionMat.at<float>(i, 1);
            float width = detectionMat.at<float>(i, 2);
            float height = detectionMat.at<float>(i, 3);
            int xLeftBottom = static_cast<int>((x - width / 2) * basket_region_image.cols);
            int yLeftBottom = static_cast<int>((y - height / 2) * basket_region_image.rows);
            int xRightTop = static_cast<int>((x + width / 2) * basket_region_image.cols);
            int yRightTop = static_cast<int>((y + height / 2) * basket_region_image.rows);

            Rect object(xLeftBottom, yLeftBottom,
                        xRightTop - xLeftBottom,
                        yRightTop - yLeftBottom);

            rectangle(basket_region_image, object, Scalar(0, 255, 0));

            if (objectClass < m_classnamesvec.size())
            {
                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = String(m_classnamesvec[objectClass]) + ": " + conf;
                results.labelName = label;

                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(basket_region_image, Rect(Point(xLeftBottom, yLeftBottom ),
                                      Size(labelSize.width, labelSize.height + baseLine)),
                          Scalar(255, 255, 255), CV_FILLED);
                putText(basket_region_image, label, Point(xLeftBottom, yLeftBottom+labelSize.height),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
            else
            {
            	log(logINFO) << "Class: " << objectClass;
            	log(logINFO) << "Confidence: " << confidence;
            	log(logINFO) << " " << xLeftBottom
                     << " " << yLeftBottom
                     << " " << xRightTop
                     << " " << yRightTop;
            }

            results.xLeftBottom = xLeftBottom;
            results.yLeftBottom = yLeftBottom;
            results.xRightTop = xRightTop;
            results.yRightTop = yRightTop;
        }

        results.confidence = confidence;
    }

	return results;
}



/*shot_results getResults()
{

	return results;
}*/

