#pragma once
#include "postprocessor.h"
#include "preprocessor.h"
class ALIGNMENT
{
public:
     explicit ALIGNMENT() {
     S.at<float>(0,0)=38.2946;
     S.at<float>(1,0)=51.6963;//左眼
     S.at<float>(2,0)=73.5318;
     S.at<float>(3,0)=51.5014;//右眼
     S.at<float>(4,0)=56.0252;
     S.at<float>(5,0)=71.7366;//鼻子
     S.at<float>(6,0)=41.5493;
     S.at<float>(7,0)=92.3655;//左嘴角
     S.at<float>(8,0)=70.7299;
     S.at<float>(9,0)=92.2041;//右嘴角
    }
     int Run(cv::Mat& img, std::vector<FACEPredictResult>& FACEres);
private:
     cv::Mat Q=cv::Mat(10,4,CV_32FC1);
     cv::Mat S=cv::Mat(10,1,CV_32FC1);

};

