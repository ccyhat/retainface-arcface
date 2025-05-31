#include "face.h"

FACE::FACE()
{
    detector = std::make_unique<RETINA>("../model/mobilenet0.25_Final.onnx", 0.9, 0.3);
    recognizer = std::make_unique<ARCFACE>("../model/MFN.onnx", 112);
    aligner = std::make_unique<ALIGNMENT>();
}

std::vector<std::vector<FACEPredictResult>> FACE::face(const std::vector<cv::Mat>& img_list) {
    std::vector<std::vector<FACEPredictResult>> face_results;
    for (const auto& img : img_list) {
        face_results.push_back(face(img));
    }
    return face_results;
}

std::vector<FACEPredictResult> FACE::face(const cv::Mat& img) {
    std::vector<FACEPredictResult> face_result;
    std::vector<cv::Mat> aligned_faces;
    det(img, face_result, aligned_faces);
    if (face_result.empty()) return face_result;
    rec(aligned_faces, face_result);
    return face_result;
}

void FACE::det(const cv::Mat& img, std::vector<FACEPredictResult>& face_results, std::vector<cv::Mat>& aligned_faces) {
    face_results.clear();
    aligned_faces.clear();
    detector->Run(const_cast<cv::Mat&>(img), face_results);
    for (auto& res : face_results) {
        cv::Rect valid_box = res.box & cv::Rect(0, 0, img.cols, img.rows);
        if (valid_box.width > 0 && valid_box.height > 0) {
            cv::Mat face_img = img(valid_box).clone();
            if (do_align_) {
                std::vector<FACEPredictResult> tmp_res{res};
                aligner->Run(face_img, tmp_res);
            }
            aligned_faces.push_back(face_img);
        }
    }
}

void FACE::rec(const std::vector<cv::Mat>& img_list, std::vector<FACEPredictResult>& face_results) {
    recognizer->Run(img_list, face_results);
}

void FACE::init(const std::vector<std::string>& path) {
    std::vector<cv::Mat> aligned_imgs;
    std::vector<std::string> valid_paths;
    for (const auto& p : path) {
        cv::Mat img = cv::imread(p);
        if (img.empty()) continue;
        std::vector<FACEPredictResult> face_base;
        std::vector<cv::Mat> aligned_faces;
        det(img, face_base, aligned_faces);
        if (!aligned_faces.empty()) {
            aligned_imgs.push_back(aligned_faces[0]);
            valid_paths.push_back(p);
        }
    }
    recognizer->GetFeature(valid_paths, aligned_imgs);
}