#pragma once
#include"face_det.h"
#include"face_rec.h"
#include"face_ali.h"
class FACE
{
public:
    explicit FACE();
    ~FACE() = default;



    std::vector<std::vector<FACEPredictResult>> face(std::vector<cv::Mat> img_list);
    std::vector<FACEPredictResult> face(cv::Mat img);
    void init(std::vector<std::string>& path);

protected:
  

    void det(cv::Mat img, std::vector<FACEPredictResult>& face_results);
    void rec(std::vector<cv::Mat> img_list,std::vector<FACEPredictResult>& face_results);


private:
    std::unique_ptr<RETINA> detector;
     std::unique_ptr<ALIGNMENT> aligner;
    std::unique_ptr<ARCFACE> recognizer;

};

