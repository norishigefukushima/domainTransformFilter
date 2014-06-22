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


class Domain_Filter
{
    public:
        Mat ct_H, ct_V, horiz, vert, O, O_t, lower_idx, upper_idx;
        void init(const Mat &img, int flags, float sigma_s, float sigma_r);
        void getGradientx( const Mat &img, Mat &gx);
        void getGradienty( const Mat &img, Mat &gy);
        void diffx(const Mat &img, Mat &temp);
        void diffy(const Mat &img, Mat &temp);
        void find_magnitude(Mat &img, Mat &mag);
        void compute_boxfilter(Mat &output, Mat &hz, Mat &psketch, float radius);
        void compute_Rfilter(Mat &O, Mat &horiz, float sigma_h);
        void compute_NCfilter(Mat &O, Mat &horiz, Mat &psketch, float radius);
        void filter(const Mat &img, Mat &res, float sigma_s = 60, float sigma_r = 0.4, int flags = 1, int iter=3);
		 
        void pencil_sketch(const Mat &img, Mat &sketch, Mat &color_res, float sigma_s, float sigma_r, float shade_factor);
        void Depth_of_field(const Mat &img, Mat &img1, float sigma_s, float sigma_r);
};

class AdaptiveManifoldFilter : public Algorithm
{
public:
	/**
	* @brief Apply High-dimensional filtering using adaptive manifolds
	* @param src       Input image to be filtered.
	* @param dst       Adaptive-manifold filter response adjusted for outliers.
	* @param tilde_dst Adaptive-manifold filter response NOT adjusted for outliers.
	* @param src_joint Image for joint filtering (optional).
	*/
	virtual void apply(InputArray src, OutputArray dst, OutputArray tilde_dst = noArray(), InputArray src_joint = noArray()) = 0;

	virtual void collectGarbage() = 0;

	static Ptr<AdaptiveManifoldFilter> create();
};