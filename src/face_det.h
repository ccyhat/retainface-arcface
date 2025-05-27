#pragma once
#include <onnxruntime_cxx_api.h>
#include "postprocessor.h"
#include "preprocessor.h"
class RETINA
{
public:
    explicit RETINA(const std::string& model_dir, float confidence, float NMSthreshold) {

        this->confidence = confidence;
        this->NMSthreshold = NMSthreshold;
        LoadModel(model_dir);
    }

    // Load Paddle inference model
    int LoadModel(const std::string& model_dir);

    // Run predictor
    int Run(cv::Mat& img, std::vector<FACEPredictResult>& FACEres,
        std::vector<double>& times);
private:
   

    std::vector<float> mean_ = { 123,117, 104 };//bgr
    //104, 117, 123 
   
 
    float confidence;
    float NMSthreshold;
    // pre-process

    RetinaNormalize normalize_op_;
    Permute permute_op_;

    // post-process
    RETINAProcessor post_processor_;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
};

