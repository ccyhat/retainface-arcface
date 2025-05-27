#include "face_ali.h"
int ALIGNMENT::Run(cv::Mat& img, std::vector<FACEPredictResult>& FACEres){
    
for(int i=0;i<Q.rows;i++){
  if(i%2){//odd
     Q.at<float>(i,0)=FACEres[0].pts[i/2].y;
     Q.at<float>(i,1)=FACEres[0].pts[i/2].x*(-1);
  }else{
     Q.at<float>(i,0)=FACEres[0].pts[i/2].x;
     Q.at<float>(i,1)=FACEres[0].pts[i/2].y;
  }
 
  Q.at<float>(i,2)=(i+1)%2;
  Q.at<float>(i,3)=i%2;
}
  cv::Mat M=(Q.t()*Q).inv()*Q.t()*S;

cv::Mat tran=cv::Mat(2,3,CV_32FC1);
tran.at<float>(0,0)=M.at<float>(0,0);
tran.at<float>(1,0)=-M.at<float>(1,0);
tran.at<float>(0,1)=M.at<float>(1,0);
tran.at<float>(1,1)=M.at<float>(0,0);
tran.at<float>(0,2)=M.at<float>(2,0);
tran.at<float>(1,2)=M.at<float>(3,0);

warpAffine(img, img,tran, img.size());
img=img(cv::Rect(0,0,112,112));
}