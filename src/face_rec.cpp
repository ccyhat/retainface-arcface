#include "face_rec.h"
#include <thread>
#include <numeric>
#include <iostream>

ARCFACE::ARCFACE(const std::string& model_path, int img_size)
    : img_size_(img_size) {
    LoadModel(model_path);
}

int ARCFACE::LoadModel(const std::string& model_path) {
    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FaceRecognization");
        session_options_ = Ort::SessionOptions();
        session_options_.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        prepareIONameCache();
        std::cout << "[INFO] ArcFace ONNXRuntime environment created successfully." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] ArcFace ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void ARCFACE::prepareIONameCache() {
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

void ARCFACE::extractFeature(const std::vector<cv::Mat>& imgs, cv::Mat& out_feature) {
    float ratio{};
    int x_off{}, y_off{};
    std::vector<cv::Mat> resize_imgs;
    for (const auto& img : imgs) {
        cv::Mat resize_img;
        img.copyTo(resize_img);
        resize_op_.Run(img, resize_img, img_size_, ratio, x_off, y_off);
        normalize_op_.Run(&resize_img);
        resize_imgs.push_back(resize_img);
    }
    std::vector<float> srcInputTensorValues(imgs.size() * 3 * resize_imgs[0].rows * resize_imgs[0].cols, 0.0f);
    permute_op_.Run(resize_imgs, srcInputTensorValues.data());
    std::vector<int64_t> input_shape = {(int)imgs.size(), 3, resize_imgs[0].rows, resize_imgs[0].cols};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, srcInputTensorValues.data(), srcInputTensorValues.size(),
        input_shape.data(), input_shape.size()
    ));

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(),
                                        input_tensors.size(), output_names_.data(), output_names_.size());
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float* pOutputData = output_tensors[0].GetTensorMutableData<float>();
    out_feature = cv::Mat(output_shape[0], output_shape[1], CV_32FC1, pOutputData).clone();
}

void ARCFACE::GetFeature(const std::vector<std::string>& paths, const std::vector<cv::Mat>& imgs) {
    cv::Mat features;
    extractFeature(imgs, features);
    feature_db_.clear();
    for (int i = 0; i < features.rows; ++i) {
        FaceData obj;
        obj.id = i;
        std::string temp = Utility::basename(paths[i]);
        size_t dotPos = temp.find_last_of('.');
        if (dotPos != std::string::npos) temp = temp.substr(0, dotPos);
        obj.name = temp;
        obj.feature.assign(features.ptr<float>(i), features.ptr<float>(i) + features.cols);
        feature_db_.push_back(obj);
    }
}

int ARCFACE::Run(const std::vector<cv::Mat>& imgs, std::vector<FACEPredictResult>& FACEres) {
    if (imgs.empty() || feature_db_.empty()) return -1;
    cv::Mat features;
    extractFeature(imgs, features);

    FACEres.resize(features.rows);
    for (int i = 0; i < features.rows; ++i) {
        cv::Mat tempfeature = features.row(i);
        cv::normalize(tempfeature, tempfeature);

        float best_score = -1.0f;
        std::string best_name = "unknown";
        for (const auto& it : feature_db_) {
            cv::Mat dbfeature(1, tempfeature.cols, CV_32FC1, const_cast<float*>(it.feature.data()));
            cv::normalize(dbfeature, dbfeature);
            float cosine = tempfeature.dot(dbfeature);
            if (cosine > best_score) {
                best_score = cosine;
                best_name = it.name;
            }
        }
        if (best_score > 0.3f) {
            FACEres[i].face_name = best_name;
            FACEres[i].score = best_score;
        } else {
            FACEres[i].face_name = "unknown";
            FACEres[i].score = best_score;
        }
    }
    return 0;
}