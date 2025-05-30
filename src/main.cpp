// PaddleOCRonnx.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "face.h"
#include <fstream>
#include "ThreadPool.h"
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <future>

int main()
{
    std::vector<cv::String> path;
    path.push_back("../img/ccy.jpg");
    std::ifstream f(path[0].c_str());
    if (!f.good())
    {
        std::cout << "no img in folder" << std::endl;
        return 0;
    }

    FACE face;
    face.init(path);
    std::cout << "init success" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "camera open failed" << std::endl;
        return -1;
    }

    // 队列和同步
    std::queue<cv::Mat> frame_queue;
    std::queue<std::future<std::pair<cv::Mat, std::vector<FACEPredictResult>>>> result_queue;
    std::mutex frame_mutex, result_mutex;
    std::condition_variable frame_cv, result_cv;
    bool running = true;

    // 线程池
    ThreadPool pool(20); // 线程数可根据实际情况调整

    // 采集任务（用线程池执行）
    auto capture_task = [&]() {
        while (running) {
            cv::Mat frame;
            cap >> frame;
            cv::flip(frame, frame, 1);
            if (frame.empty()) continue;
			if(result_queue.size() > 5) // 控制队列大小，避免内存溢出
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}
            {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame_queue.push(frame.clone());
            frame_cv.notify_one();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 控制采集帧率
        }
    };
    // 提交采集任务到线程池
    auto capture_future = pool.enqueue(capture_task);

    // 主线程负责分发任务到线程池
    std::thread dispatch_thread([&]()
    {
        while (running) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(frame_mutex);
                frame_cv.wait(lock, [&]() { return !frame_queue.empty() || !running; });
                if (!running) break;
                frame = frame_queue.front();
                frame_queue.pop();
            }
            // 提交检测任务到线程池
            auto fut = pool.enqueue([frame, &face]() mutable {
                std::vector<FACEPredictResult> v = face.face(frame);
                return std::make_pair(frame, v);
            });
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                result_queue.push(std::move(fut));
            }
            result_cv.notify_one();
        }
    });

    // 主线程显示
    cv::namedWindow("Video", 1);
    double last_time = (double)cv::getTickCount();
    double fps = 0.0;
    int frame_count = 0;

    while (running)
    {
        std::future<std::pair<cv::Mat, std::vector<FACEPredictResult>>> fut;
        {
            std::unique_lock<std::mutex> lock(result_mutex);
            result_cv.wait(lock, [&]()
                { return !result_queue.empty() || !running; });
            if (!running)
                break;
            fut = std::move(result_queue.front());
            result_queue.pop();
			//std::cout << "result_queue size: " << result_queue.size() << std::endl;
        }
        if (fut.valid())
        {
            auto result = fut.get();
            cv::Mat show_frame = result.first;
            auto &v = result.second;
            // 画框...
            for (int i = 0; i < v.size(); i++)
            {
                cv::rectangle(show_frame, v[i].box, cv::Scalar(255, 0, 0));
                if (v[i].face_name != "")
                {
                    int baseline = 0;
                    cv::Size textSize = cv::getTextSize(v[i].face_name + "_" + std::to_string(v[i].score), cv::FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseline);
                    cv::rectangle(show_frame, v[i].box.tl() + cv::Point(0, baseline), v[i].box.tl() + cv::Point(textSize.width, -textSize.height), cv::Scalar(255, 0, 0), -1);
                    cv::putText(show_frame, v[i].face_name + "_" + std::to_string(v[i].score), v[i].box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
                }
            }

            // 计算并显示帧率
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
            if (cv::waitKey(1) >= 0)
            {
                running = false;
                frame_cv.notify_all();
                result_cv.notify_all();
                break;
            }
        }
    }

    // 等待线程结束
    capture_future.get(); // 等待采集任务结束
    dispatch_thread.join();
    cap.release();
    return 0;
}
