#pragma once
#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

class RetinaNormalize
{
public:
	virtual void Run(cv::Mat* im, const std::vector<float>& mean);
};
class ARCNormalize
{
public:
	virtual void Run(cv::Mat* im);
};
class Resize {
public:
	virtual void Run(const cv::Mat& img, cv::Mat& resize_img, const int h,
		const int w);
};

class Permute {
public:
	virtual void Run(const cv::Mat* im, float* data);
};
class ResizeLetterBox {
public:
	virtual void Run(const cv::Mat& img ,cv::Mat& resize_img, const int size, float& ratio, int& x_off, int& y_off);
};
class PermuteBatch {
public:
	virtual void Run(const std::vector<cv::Mat> imgs, float* data);
};


