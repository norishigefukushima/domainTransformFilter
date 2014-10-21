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

void domainTransformFilterRF(const Mat& img, Mat& out, float sigma_r, float sigma_s, int maxiter, int norm=DTF_L1, int implementation=DTF_BGRA_SSE_PARALLEL);
void domainTransformFilterRF(const Mat& img, const Mat& guide, Mat& out, float sigma_r, float sigma_s,int maxiter, int norm=DTF_L1, int implementation=DTF_BGRA_SSE_PARALLEL);
void domainTransformFilterNC(const Mat& src, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm, int implementation=DTF_BGRA_SSE_PARALLEL);
void domainTransformFilterNC(const Mat& src, const Mat& guide, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm, int implementation=DTF_BGRA_SSE_PARALLEL);
void domainTransformFilterIC(const Mat& src, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm, int implementation=DTF_BGRA_SSE_PARALLEL);
void domainTransformFilterIC(const Mat& src, const Mat& guide, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm, int implementation=DTF_BGRA_SSE_PARALLEL);


void cvtColorBGR2PLANE(const Mat& src, Mat& dest);
void cvtColorPLANE2BGR(const Mat& src, Mat& dest);
void cvtColorBGRA2BGR(const Mat& src, Mat& dest);
void cvtColorBGRA32f2BGR8u(const Mat& src, Mat& dest);
void cvtColorBGR2BGRA(const Mat& src, Mat& dest, const uchar alpha=255);
void cvtColorBGR8u2BGRA32f(const Mat& src, Mat& dest, const float alpha=255.f);