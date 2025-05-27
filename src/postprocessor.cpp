#include "postprocessor.h"
#include <opencv2/dnn.hpp>
//#include<algorithm>
void RETINAProcessor::BoxesFromRETINA(std::vector<cv::Mat>& output, std::vector<FACEPredictResult>& res, float confidence, float NMSthreshold,int img_w,int img_h) {
    //�����+΢������λbox
    std::vector<float> priorboxes;
    PriorBox(priorboxes, img_w, img_h);
    cv::Mat prior(priorboxes.size()/4, 4, CV_32FC1, (float*)priorboxes.data());
    cv::Mat boxes = Decode(prior, output[0]);
    //�����+΢������λface_pts
    cv::Mat pts = Decode_landm(prior, output[2]);
    //box
    float* pbox = (float*)boxes.data;
    float* pconf = (float*)output[1].data;
    float* ppts = (float*)output[2].data;
    std::vector<std::vector<float>> box_data(boxes.rows, std::vector<float>(boxes.cols));
    for (int i = 0; i < box_data.size(); i++) {
        memcpy(box_data[i].data(), pbox, boxes.cols * sizeof(float));
        pbox += boxes.cols;
    }
    //confidence
    std::vector<std::vector<float>> conf_data(output[1].rows, std::vector<float>(output[1].cols));
    for (int i = 0; i < conf_data.size(); i++) {
        memcpy(conf_data[i].data(), pconf, output[1].cols * sizeof(float));
        pconf += output[1].cols;
    }
    //face_pts
    std::vector<std::vector<float>> pts_data(pts.rows, std::vector<float>(pts.cols));
    for (int i = 0; i < pts_data.size(); i++) {
        memcpy(pts_data[i].data(), ppts, pts.cols * sizeof(float));
        ppts += pts.cols;
    }
    
    std::vector<float> PreConf;
    std::vector<cv::Rect> PreBox;
    std::vector<std::vector<cv::Point>> PrePts;
    for (int i = 0; i < conf_data.size(); i++) {
        if (conf_data[i][1] > confidence) {
            float x1 = std::max(box_data[i][0] * img_w,0.0f);
            float y1 = std::max(box_data[i][1] * img_h,0.0f);
            float w = box_data[i][2] * img_w;
            float h = box_data[i][3] * img_h;
            PreConf.push_back(conf_data[i][1]);
            PreBox.push_back(cv::Rect(x1, y1, w, h));
            std::vector<cv::Point > Pts;
            for (int k = 0; k < pts_data[i].size(); k+=2) {
                int x = pts_data[i][k] * img_w;
                int y = pts_data[i][k + 1] * img_h;
                Pts.push_back(cv::Point(x,y));
            }
         
            PrePts.push_back(Pts);
        }
    }
    std::vector<int> index;
    cv::dnn::NMSBoxes(PreBox, PreConf,confidence,NMSthreshold, index);
    for (auto it : index) {
        FACEPredictResult obj;
        obj.box = PreBox[it];
        obj.pts = PrePts[it];
        obj.score = PreConf[it];
        res.push_back(obj);
    }
}
void RETINAProcessor::PriorBox( std::vector<float> &priorboxes,int img_w, int img_h) {
    cfg_mnet cfg;
    std::vector<std::vector<int>> feature_maps;
    for (auto it : cfg.steps) {
        int w = std::ceil(img_w / ((float)it));
        int h = std::ceil(img_h / ((float)it));
        feature_maps.push_back({ h, w });
    }
    for (int k = 0; k < feature_maps.size(); ++k) {
        auto f = feature_maps[k];
        auto min_sizes = cfg.min_sizes[k];

        for (int i = 0; i < f[0]; ++i) {
            for (int j = 0; j < f[1]; ++j) {
                for (auto min_size : min_sizes) {
                    float s_kx = (float)min_size / img_w;
                    float s_ky = (float)min_size / img_h;

                    std::vector<float> dense_cx = { (j + 0.5f) * cfg.steps[k] / img_w };
                    std::vector<float> dense_cy = { (i + 0.5f) * cfg.steps[k] / img_h };
                    for (auto cy : dense_cy) {
                        for (auto cx : dense_cx) {
                            priorboxes.emplace_back(cx);
                            priorboxes.emplace_back(cy); 
                            priorboxes.emplace_back(s_kx);
                            priorboxes.emplace_back(s_ky); 
                        }
                    }
                }
            }
        }
    }
}
cv::Mat RETINAProcessor::Decode(cv::Mat priors,cv::Mat loc) {
    cfg_mnet cfg;
    cv::Mat result;

    cv::Mat cxy;
    cv::Mat sxy;
    cv::multiply(loc.colRange(0, 2), priors.colRange(2, 4), cxy, cfg.variance[0]);
    cv::add(priors.colRange(0, 2), cxy, cxy);

    cv::Mat exp;

    cv::exp(loc.colRange(2,4) * cfg.variance[1], exp);
    cv::multiply(priors.colRange(2, 4), exp, sxy);
    cv::subtract(cxy,sxy/2, cxy);
    cv::hconcat(cxy, sxy, result);

    return result;

}
cv::Mat RETINAProcessor::Decode_landm(cv::Mat priors, cv::Mat pts) {
    cfg_mnet cfg;
    cv::Mat result;
    cv::Mat temp1;
    cv::multiply(pts.colRange(0, 2), priors.colRange(2, 4), temp1, cfg.variance[0]);
    cv::add(priors.colRange(0, 2), temp1, temp1);
    temp1.copyTo(result);
    cv::multiply(pts.colRange(2, 4), priors.colRange(2, 4), temp1, cfg.variance[0]);
    cv::add(priors.colRange(0, 2), temp1, temp1);
    cv::hconcat(result, temp1, result);
    cv::multiply(pts.colRange(4, 6), priors.colRange(2, 4), temp1, cfg.variance[0]);
    cv::add(priors.colRange(0, 2), temp1, temp1);
    cv::hconcat(result, temp1, result);
    cv::multiply(pts.colRange(6, 8), priors.colRange(2, 4), temp1, cfg.variance[0]);
    cv::add(priors.colRange(0, 2), temp1, temp1);
    cv::hconcat(result, temp1, result);
    cv::multiply(pts.colRange(8, 10), priors.colRange(2, 4), temp1, cfg.variance[0]);
    cv::add(priors.colRange(0, 2), temp1, temp1);
    cv::hconcat(result, temp1, result);
    return result;
}