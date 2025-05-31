// PaddleOCRonnx.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "face.h"
#include <fstream>
#include "ThreadPool.h"
#include "processor.h"

int main(int argc, char* argv[])
{
    int thread_num = 10;
    if (argc > 1) {
        try {
            thread_num = std::stoi(argv[1]);
        } catch (...) {
            std::cout << "Invalid thread number, use default 10." << std::endl;
            thread_num = 10;
        }
    }
    std::cout << "Thread pool size: " << thread_num << std::endl;
    std::vector<cv::String> path;
    path.push_back("../img/ccy.jpg");
    std::ifstream f(path[0].c_str());
    if (!f.good()) {
        std::cout << "no img in folder" << std::endl;
        return 0;
    }
    FACE face;
    face.init(path);
    std::cout << "init success" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "camera open failed" << std::endl;
        return -1;
    }

    Processor processor(face, cap, thread_num);
    processor.run();

    cap.release();
    return 0;
}
