#include "face.h"
FACE::FACE() {
    this->detector.reset(new RETINA("../model/mobilenet0.25_Final.onnx",  0.8, 0.3));
    this->recognizer.reset(new ARCFACE("../model/MFN.onnx", 112));
    this->aligner.reset(new ALIGNMENT());
}
std::vector<std::vector<FACEPredictResult>> FACE::face(std::vector<cv::Mat> img_list) {
    std::vector<std::vector<FACEPredictResult>> face_results;
    for (int i = 0; i < img_list.size(); ++i) {
        std::vector<FACEPredictResult> face_result = this->face(img_list[i]);
        face_results.push_back(face_result);
    }

    return face_results;
}
std::vector<FACEPredictResult> FACE::face(cv::Mat img) {

    std::vector<FACEPredictResult> face_result;
    std::vector<cv::Mat>faces;
    // det
    this->det(img, face_result);

    // align
    for(auto it:face_result){
        faces.push_back(img(it.box));
    }
   
    this->rec(faces,face_result);   
    
    return face_result;
}
void FACE::det(cv::Mat img, std::vector<FACEPredictResult>& face_results) {
    std::vector<double> det_times;
    this->detector->Run(img, face_results, det_times);
    this->time_info_det[0] += det_times[0];
    this->time_info_det[1] += det_times[1];
    this->time_info_det[2] += det_times[2];
}
void FACE::init(std::vector<std::string>& path) {
    std::vector<cv::Mat>imgs;
    for(auto it:path){
        std::vector<FACEPredictResult> face_base;
        cv::Mat img;
        img=cv::imread(it);
        this->det(img,face_base);
        img=img(face_base[0].box);
        imgs.push_back(img);
    }
    this->recognizer->GetFeature(path,imgs);
}
void FACE::rec(std::vector<cv::Mat> img_list,std::vector<FACEPredictResult>& face_results){
    std::vector<double> rec_times;
    this->recognizer->Run(img_list,face_results,rec_times);
}