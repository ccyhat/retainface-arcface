#pragma once
#include <onnxruntime_cxx_api.h>
#include "postprocessor.h"
#include "preprocessor.h"
#include <vector>
#include <memory>

class RETINA {
public:
    RETINA(const std::string& model_dir, float confidence, float NMSthreshold);

    int LoadModel(const std::string& model_dir);
    int Run(cv::Mat& img, std::vector<FACEPredictResult>& FACEres, std::vector<double>& times);

private:
    std::vector<float> mean_ = {123, 117, 104}; // bgr
    float confidence_;
    float NMSthreshold_;

    RetinaNormalize normalize_op_;
    Permute permute_op_;
    RETINAProcessor post_processor_;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<std::string> input_name_strs_, output_name_strs_;
    std::vector<const char*> input_names_, output_names_;

    void prepareIONameCache();
};

