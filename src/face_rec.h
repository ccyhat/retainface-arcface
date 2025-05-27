#pragma once
#include <onnxruntime_cxx_api.h>
#include "preprocessor.h"
#include "postprocessor.h"
#include "utility.h"
#include<list>
class ARCFACE
{
public:
    explicit ARCFACE(const std::string& model_dir,int img_size) {
        this->img_size = img_size;
        LoadModel(model_dir);
    }
    // Load Paddle inference model
    int LoadModel(const std::string& model_dir);

    // Run predictor
    int Run(std::vector<cv::Mat>& imgs, std::vector<FACEPredictResult>& FACEres,
        std::vector<double>& times);
    void GetFeature(std::vector<std::string>& path,std::vector<cv::Mat> imgs);
    
private:
    int img_size;

    std::vector<float> mean_ = { 123,117, 104 };//bgr
    //104, 117, 123 
    std::vector<float> scale_ = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };

    // pre-process
    ResizeLetterBox resize_op_;
    ARCNormalize normalize_op_;
    PermuteBatch permute_op_;
    std::list<FaceData>  feature;
    // post-process
    ARCProcessor post_processor_;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

};

