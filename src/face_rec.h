#pragma once
#include <onnxruntime_cxx_api.h>
#include "preprocessor.h"
#include "postprocessor.h"
#include "utility.h"
#include <vector>
#include <memory>

class ARCFACE {
public:
    ARCFACE(const std::string& model_path, int img_size);
    int LoadModel(const std::string& model_path);
    int Run(const std::vector<cv::Mat>& imgs, std::vector<FACEPredictResult>& FACEres);
    void GetFeature(const std::vector<std::string>& paths, const std::vector<cv::Mat>& imgs);

private:
    int img_size_;
    std::vector<float> mean_ = {123, 117, 104};
    std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};

    ResizeLetterBox resize_op_;
    ARCNormalize normalize_op_;
    PermuteBatch permute_op_;

    std::vector<FaceData> feature_db_;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<std::string> input_name_strs_, output_name_strs_;
    std::vector<const char*> input_names_, output_names_;

    void prepareIONameCache();
    void extractFeature(const std::vector<cv::Mat>& imgs, cv::Mat& out_feature);
};

