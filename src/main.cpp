// PaddleOCRonnx.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "face.h"
#include <fstream>
#include "ThreadPool.h"
#include "processor.h"
#include <filesystem>

int main(int argc, char* argv[])
{
    int thread_num = 10;
    std::string video_path;
    if (argc > 1) {
        try {
            thread_num = std::stoi(argv[1]);
            if (argc > 2) {
                video_path = argv[2]; // 第二个参数为视频文件路径
            }
        } catch (...) {
            std::cout << "Invalid thread number, use default 10." << std::endl;
            thread_num = 10;
        }
    }
    std::cout << "Thread pool size: " << thread_num << std::endl;

    // 搜索img文件夹下所有图片文件
    std::vector<cv::String> path;
    std::string img_dir = "img";
    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            // 支持常见图片格式
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp") {
                path.push_back(entry.path().string());
            }
        }
    }

    if (path.empty()) {
        std::cout << "no img in folder" << std::endl;
        return 0;
    }
    FACE face;
    face.init(path);
    std::cout << "init success" << std::endl;

    cv::VideoCapture cap;
    if (!video_path.empty()) {
        cap.open(video_path); // 打开视频文件
        if (!cap.isOpened()) {
            std::cout << "video file open failed" << std::endl;
            return -1;
        }
        std::cout << "Open video file: " << video_path << std::endl;
    } else {
        cap.open(0); // 打开摄像头
        if (!cap.isOpened()) {
            std::cout << "camera open failed" << std::endl;
            return -1;
        }
        std::cout << "Open camera." << std::endl;
    }

    Processor processor(face, cap, thread_num);
    processor.run();

    cap.release();
    return 0;
}
