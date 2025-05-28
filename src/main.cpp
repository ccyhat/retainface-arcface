// PaddleOCRonnx.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include"face.h"
#include<fstream>



int main()
{
    std::vector<cv::String> path;
    path.push_back("../img/ccy.jpg");
	std::ifstream f(path[0].c_str());
	if(!f.good()){
		std::cout<<"no img in folder"<<std::endl;
		return 0;
	}
	
    FACE face;
    face.init(path);
	std::cout<<"init success"<<std::endl;
    // 1.创建视频采集对象;
	 cv::VideoCapture cap;	
    // FACE face;

	 // 2.打开默认相机;
	 cap.open(0);

	 // 3.判断相机是否打开成功;
	 if (!cap.isOpened())
	 	return -1;

	 // 4.显示窗口命名;
	 cv::namedWindow("Video", 1);
	for (;;)
	{
		// 获取新的一帧;
		cv::Mat frame;
		cap >> frame; 
		cv::flip(frame, frame, 1);
		if (frame.empty()){
			std::cout << "No frame captured from camera." << std::endl;
			break;
		}
        std::vector<FACEPredictResult>v=face.face(frame);

		if(v.size()==0){
			std::cout<<"no face detected"<<std::endl;
			continue;
		}
        for(int i=0;i<v.size();i++){
            cv::rectangle(frame,v[i].box,cv::Scalar(255,0,0));
            if(v[i].face_name!=""){
                 int baseline=0;
                 cv::Size textSize = cv::getTextSize(v[i].face_name+"_"+std::to_string(v[i].score), cv::FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseline);
                 cv::rectangle(frame, v[i].box.tl() + cv::Point(0, baseline), v[i].box.tl() + cv::Point(textSize.width, -textSize.height), cv::Scalar(255, 0, 0), -1);
                 cv::putText(frame, v[i].face_name+"_"+std::to_string(v[i].score), v[i].box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
            }
        }
        
     

	// 	// 显示新的帧;
	 	imshow("Video", frame);
		
	// 	// 按键退出显示;
	 	if (cv::waitKey(30) >= 0) break;
	 }

	// // 5.释放视频采集对象;
	 cap.release();

    return 0;
}


