#pragma once
#include <opencv2/opencv.hpp>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "face.h"
#include "ThreadPool.h"

class Processor {
public:
    Processor(FACE& face, cv::VideoCapture& cap, int pool_size = 20)
        : face_(face), cap_(cap), pool_(pool_size), running_(true) {
        // 判断是否为摄像头（帧数为0或1通常为摄像头）
        is_camera_ = (cap_.get(cv::CAP_PROP_FRAME_COUNT) <= 1);
    }
    void run() {
        // 采集任务
        auto capture_task = [&]() {
            while (running_) {
                cv::Mat frame;
                cap_ >> frame;
                if (is_camera_) {
                    cv::flip(frame, frame, 1); // 仅摄像头镜像
                }
                if (frame.empty()) continue;
                {
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    if (frame_queue_.size() >= 5) frame_queue_.pop();
                    frame_queue_.push(frame.clone());
                }
                frame_cv_.notify_one();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        };
        auto capture_future = pool_.enqueue(capture_task);

        // 分发任务
        std::thread dispatch_thread([&]() {
            while (running_) {
                cv::Mat frame;
                {
                    std::unique_lock<std::mutex> lock(frame_mutex_);
                    frame_cv_.wait(lock, [&]() { return !frame_queue_.empty() || !running_; });
                    if (!running_) break;
                    frame = frame_queue_.front();
                    frame_queue_.pop();
                }
                auto fut = pool_.enqueue([frame, this]() mutable {
                    std::vector<FACEPredictResult> v = face_.face(frame);
                    return std::make_pair(frame, v);
                });
                {
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    if (result_queue_.size() < 5) {
                        result_queue_.push(std::move(fut));
                        result_cv_.notify_one();
                    }
                }
            }
        });

        // 显示
        cv::namedWindow("Video", 1);
        double last_time = (double)cv::getTickCount();
        double fps = 0.0;
        int frame_count = 0;

        while (running_) {
            std::future<std::pair<cv::Mat, std::vector<FACEPredictResult>>> fut;
            {
                std::unique_lock<std::mutex> lock(result_mutex_);
                result_cv_.wait(lock, [&]() { return !result_queue_.empty() || !running_; });
                if (!running_) break;
                fut = std::move(result_queue_.front());
                result_queue_.pop();
            }
            if (fut.valid()) {
                auto result = fut.get();
                cv::Mat show_frame = result.first;
                auto& v = result.second;
                for (int i = 0; i < v.size(); i++) {
                    cv::rectangle(show_frame, v[i].box, cv::Scalar(255, 0, 0));
                    if (v[i].face_name != "") {
                        int baseline = 0;
                        cv::Size textSize = cv::getTextSize(v[i].face_name + "_" + std::to_string(v[i].score), cv::FONT_HERSHEY_SIMPLEX, 1.2, 1, &baseline);
                        cv::rectangle(show_frame, v[i].box.tl() + cv::Point(0, baseline), v[i].box.tl() + cv::Point(textSize.width, -textSize.height), cv::Scalar(255, 0, 0), -1);
                        cv::putText(show_frame, v[i].face_name + "_" + std::to_string(v[i].score), v[i].box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 1);
                    }
                }
                // FPS
                frame_count++;
                double now = (double)cv::getTickCount();
                double elapsed = (now - last_time) / cv::getTickFrequency();
                if (elapsed >= 1.0) {
                    fps = frame_count / elapsed;
                    last_time = now;
                    frame_count = 0;
                }
                char fps_text[32];
                snprintf(fps_text, sizeof(fps_text), "FPS: %.2f", fps);
                cv::putText(show_frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

                imshow("Video", show_frame);
                if (cv::waitKey(1) >= 0) {
                    running_ = false;
                    frame_cv_.notify_all();
                    result_cv_.notify_all();
                    break;
                }
            }
        }
        capture_future.get();
        dispatch_thread.join();
    }

private:
    FACE& face_;
    cv::VideoCapture& cap_;
    ThreadPool pool_;
    std::queue<cv::Mat> frame_queue_;
    std::queue<std::future<std::pair<cv::Mat, std::vector<FACEPredictResult>>>> result_queue_;
    std::mutex frame_mutex_, result_mutex_;
    std::condition_variable frame_cv_, result_cv_;
    bool running_;
    bool is_camera_; // 新增成员变量

};