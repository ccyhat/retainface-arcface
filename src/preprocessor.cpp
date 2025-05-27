#include "preprocessor.h"
void RetinaNormalize::Run(cv::Mat* im, const std::vector<float>& mean) {
    double e = 1.0;
    (*im).convertTo(*im, CV_32FC3, e);
    std::vector<cv::Mat> bgr_channels(3);
    cv::split(*im, bgr_channels);
    for (auto i = 0; i < bgr_channels.size(); i++) {
            bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0,
                (0.0 - mean[i]));
    }
    cv::merge(bgr_channels, *im);   
  
}
void ARCNormalize::Run(cv::Mat* im) {
    double e = 1.0;
    e /= 255.0;
    (*im).convertTo(*im, CV_32FC3, e);  
}
void ResizeLetterBox::Run(const cv::Mat& img, cv::Mat& resize_img, const int size,float &ratio,int& x_off,int& y_off) {

   
    if (img.cols == img.rows) {
        ratio = ((float)img.cols) / size;
        cv::resize(img, resize_img,cv::Size(size, size),0, 0, cv::INTER_LINEAR);
    }
    else {
        int max_wh = std::max(img.cols, img.rows);
        ratio = ((float)max_wh) / size;

        int resize_h = int(float(img.rows) / ratio);
        int resize_w = int(float(img.cols) / ratio);
        cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
        //Ìî³ä
        cv::Mat padded_image(size, size, resize_img.type(), cv::Scalar(114, 114, 114));
        y_off = (size - resize_img.rows) / 2;
        x_off = (size - resize_img.cols) / 2;
        resize_img.copyTo(padded_image(cv::Rect(x_off, y_off, resize_img.cols, resize_img.rows)));
        padded_image.copyTo(resize_img);
    }
   
}


void Resize::Run(const cv::Mat& img, cv::Mat& resize_img, const int h,
    const int w) {
    cv::resize(img, resize_img, cv::Size(w, h));
}
void Permute::Run(const cv::Mat* im, float* data) {
    int rh = im->rows;
    int rw = im->cols;
    int rc = im->channels();
    for (int i = 0; i < rc; ++i) {
        cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
    }
}
void PermuteBatch::Run(const std::vector<cv::Mat> imgs, float* data) {
    for (int j = 0; j < imgs.size(); j++) {
        int rh = imgs[j].rows;
        int rw = imgs[j].cols;
        int rc = imgs[j].channels();
        for (int i = 0; i < rc; ++i) {
            cv::extractChannel(
                imgs[j], cv::Mat(rh, rw, CV_32FC1, data + (j * rc + i) * rh * rw), i);
        }
    }
}
