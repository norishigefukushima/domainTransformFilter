#pragma once

#include <opencv2/opencv.hpp>
using namespace cv;
#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")


typedef enum
{
	DTF_L1=1,
	DTF_L2=2
}DTF_NORM;

typedef enum
{
	DTF_RF=0,//Recursive Filtering
	DTF_NC=1,//Normalized Convolution
	DTF_IC=2,//Interpolated Convolution

}DTF_METHOD;

typedef enum
{
	DTF_BGRA_SSE=0,
	DTF_BGRA_SSE_PARALLEL,
	DTF_SLOWEST
}DTF_IMPLEMENTATION;


void domainTransformFilter(InputArray srcImage, OutputArray destImage, const float sigma_r, const float sigma_s, const int maxiter, const int norm=DTF_L1, const int convolutionType=DTF_RF, const int implementation=DTF_SLOWEST);
void domainTransformFilter(InputArray srcImage, InputArray guideImage, OutputArray destImage, const float sigma_r, const float sigma_s, const int maxiter, const int norm=DTF_L1, const int convolutionType=DTF_RF, const int implementation=DTF_SLOWEST);


void cvtColorBGR2PLANE(const Mat& src, Mat& dest);
void cvtColorPLANE2BGR(const Mat& src, Mat& dest);
void cvtColorBGRA2BGR(const Mat& src, Mat& dest);
void cvtColorBGRA32f2BGR8u(const Mat& src, Mat& dest);
void cvtColorBGR2BGRA(const Mat& src, Mat& dest, const uchar alpha=255);
void cvtColorBGR8u2BGRA32f(const Mat& src, Mat& dest, const float alpha=255.f);