#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


struct FACEPredictResult {
    cv::Rect box;
    std::vector<cv::Point> pts;
    float score;
    std::string face_name = "";
};
struct FaceData {
    int id;
    cv::Mat feature;
    std::string name;
};

struct cfg_mnet {
  
    const std::vector<std::vector<int>> min_sizes = {
        {16, 32},
        {64, 128},
        {256, 512}
    };
    const int steps[3]={8,16,32};
    const float variance[2]={ 0.1,0.2 };
};

class Utility
{
public:
    static  void  print_result(const std::vector<FACEPredictResult>& face_result);
    static std::string basename(const std::string& filename);
   
    static void VisualizeBboxes(const cv::Mat& srcimg,
        const std::vector<FACEPredictResult>& yolo_result,
        const std::string& save_path);
   
   
    
    
private:
   
};


