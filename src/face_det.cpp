#include "face_det.h"
#include <thread>
#include <numeric>
#include <iostream>

RETINA::RETINA(const std::string& model_dir, float confidence, float NMSthreshold)
    : confidence_(confidence), NMSthreshold_(NMSthreshold) {
    LoadModel(model_dir);
}

int RETINA::LoadModel(const std::string& model_dir) {
    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FaceDetection");
        session_options_ = Ort::SessionOptions();
        session_options_.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, model_dir.c_str(), session_options_);
        prepareIONameCache();
        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void RETINA::prepareIONameCache() {
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_strs_.clear();
    input_names_.clear();
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        auto name_alloc = session_->GetInputNameAllocated(i, allocator);
        input_name_strs_.push_back(name_alloc.get());
    }
    for (auto& s : input_name_strs_) input_names_.push_back(s.c_str());

    output_name_strs_.clear();
    output_names_.clear();
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        auto name_alloc = session_->GetOutputNameAllocated(i, allocator);
        output_name_strs_.push_back(name_alloc.get());
    }
    for (auto& s : output_name_strs_) output_names_.push_back(s.c_str());
}

int RETINA::Run(cv::Mat& img, std::vector<FACEPredictResult>& FACEres) {
    cv::Mat resize_img;
    img.copyTo(resize_img);
    normalize_op_.Run(&resize_img, mean_);
    std::vector<float> srcInputTensorValues(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
    permute_op_.Run(&resize_img, srcInputTensorValues.data());
    std::vector<int64_t> input_shape = {1, 3, resize_img.rows, resize_img.cols};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, srcInputTensorValues.data(), srcInputTensorValues.size(),
        input_shape.data(), input_shape.size()
    ));

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(),
                                       input_tensors.size(), output_names_.data(), output_names_.size());

    std::vector<cv::Mat> out_datas;
    for (size_t i = 0; i < output_names_.size(); i++) {
        auto output_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        float* pOutputData = output_tensors[i].GetTensorMutableData<float>();
        cv::Mat out_data(output_shape[1], output_shape[2], CV_32FC1, pOutputData);
        out_datas.push_back(out_data);
    }
    post_processor_.BoxesFromRETINA(out_datas, FACEres, confidence_, NMSthreshold_, img.cols, img.rows);
    return 0;
}