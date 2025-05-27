#include "utility.h"
#include <codecvt>
#include <locale>


void Utility::print_result(const std::vector<FACEPredictResult>& face_result) {
    for (int i = 0; i < face_result.size(); i++) {
        std::cout << i << "\t";
        // det
        if (!face_result[i].box.empty()) {
            std::cout << "det boxes: [";
            std::cout << face_result[i].box.br();
            std::cout << ",";
            std::cout << face_result[i].box.tl();
            std::cout << "] ";
            std::cout << "class: ";
            std::cout << face_result[i].face_name;

        }
     
        std::cout << std::endl;
    }
}
std::string Utility::basename(const std::string& filename) {
    if (filename.empty()) {
        return "";
    }

    auto len = filename.length();
    auto index = filename.find_last_of("/\\");

    if (index == std::string::npos) {
        return filename;
    }

    if (index + 1 >= len) {

        len--;
        index = filename.substr(0, len).find_last_of("/\\");

        if (len == 0) {
            return filename;
        }

        if (index == 0) {
            return filename.substr(1, len - 1);
        }

        if (index == std::string::npos) {
            return filename.substr(0, len);
        }

        return filename.substr(index + 1, len - index - 1);
    }

    return filename.substr(index + 1, len - index);
}

void Utility::VisualizeBboxes(const cv::Mat& srcimg,
    const std::vector<FACEPredictResult>& face_result,
    const std::string& save_path) {
    cv::Mat img_vis;
    srcimg.copyTo(img_vis);
    for (int n = 0; n < face_result.size(); n++) {
        cv::rectangle(img_vis, face_result[n].box, cv::Scalar(255, 0, 0), 2);

        int baseline = 0;
        if(face_result[n].face_name!=""){
            cv::Size textSize = cv::getTextSize(face_result[n].face_name, cv::FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseline);
            cv::rectangle(img_vis, face_result[n].box.tl() + cv::Point(0, baseline), face_result[n].box.tl() + cv::Point(textSize.width, -textSize.height), cv::Scalar(255, 0, 0), -1);
            cv::putText(img_vis, face_result[n].face_name, face_result[n].box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
        }
        //cv::Size textSize = cv::getTextSize(face_result[n].face_name, cv::FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseline);
        //cv::rectangle(img_vis, face_result[n].box.tl() + cv::Point(0, baseline), face_result[n].box.tl() + cv::Point(textSize.width, -textSize.height), cv::Scalar(255, 0, 0), -1);
        //cv::putText(img_vis, face_result[n].face_name, face_result[n].box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);

    }

    cv::imwrite(save_path, img_vis);
    std::cout << "The detection visualized image saved in " + save_path
        << std::endl;
}



