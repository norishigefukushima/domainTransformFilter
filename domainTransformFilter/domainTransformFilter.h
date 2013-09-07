#pragma once

#include <opencv2/opencv.hpp>
using namespace cv;
#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")

typedef enum
{
	DTF_RF=0,//Recursive Filtering
	DTF_NC=1,//Normalized Convolution
	DTF_IC=1,//Interpolated Convolution

}DTF_METHOD;

void domainTransformFilter(cv::Mat& img, cv::Mat& out, double sigma_s, double sigma_r, int maxiter, int method=DTF_RF);
void domainTransformFilterbase(cv::Mat& src, cv::Mat& dest, double sigma_s, double sigma_r, int maxiter);