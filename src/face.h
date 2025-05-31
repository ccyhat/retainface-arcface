#pragma once
#include "face_det.h"
#include "face_rec.h"
#include "face_ali.h"
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

class FACE {
public:
    explicit FACE();
    ~FACE() = default;

    std::vector<std::vector<FACEPredictResult>> face(const std::vector<cv::Mat>& img_list);
    std::vector<FACEPredictResult> face(const cv::Mat& img);
    void init(const std::vector<std::string>& path);

protected:
    void det(const cv::Mat& img, std::vector<FACEPredictResult>& face_results, std::vector<cv::Mat>& aligned_faces);
    void rec(const std::vector<cv::Mat>& img_list, std::vector<FACEPredictResult>& face_results);

private:
    std::unique_ptr<RETINA> detector;
    std::unique_ptr<ALIGNMENT> aligner;
    std::unique_ptr<ARCFACE> recognizer;
    bool do_align_=false; // 是否进行人脸对齐
};

