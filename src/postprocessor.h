#pragma once
#include "utility.h"


class RETINAProcessor {
public:
    void BoxesFromRETINA(std::vector<cv::Mat>& output, std::vector<FACEPredictResult>& res, float confidence, float NMSthreshold, int img_w, int img_h);
   
private:
    void PriorBox(std::vector<float>& priorboxes,int img_w, int img_h);
    cv::Mat Decode(cv::Mat priors, cv::Mat loc);
    cv::Mat Decode_landm(cv::Mat priors, cv::Mat pts);

};
class ARCProcessor {
public:
    std::vector<int> GetFaceID(cv::Mat& output, std::vector<cv::Mat>& feature);
    void GetFaceName(std::vector<int> ID, std::vector<FACEPredictResult>& res);
   

};
