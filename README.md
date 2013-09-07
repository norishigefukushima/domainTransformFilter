domainTransformFilter
=====================

The code is implimentation of domain transform filter for OpenCV.  
The code is parallelized by Intel TBB or OpenMP over c cv::ParallelLoopBody class in OpenCV.  
Also, the code is vectorized by SIMD SSE4.2.   

*Reference  
Eduardo S. L. Gastal and Manuel M. Oliveira. "Domain Transform for Edge-Aware Image and Video Processing". ACM Transactions on Graphics. Volume 30 (2011), Number 4, Proceedings of SIGGRAPH 2011, Article 69.
http://inf.ufrgs.br/~eslgastal/DomainTransform/


-----------------------------------------------
function document

void domainTransformFilter(cv::Mat& img, cv::Mat& out, double sigma_s, double sigma_r, int maxiter, int method=DTF_RF)
 img: src image  
 out: dest image  
 sigma_s: sigma of space  
 sigma_r: sigma of range[0:255]  
 maxiter: number of maximam iteration -1  
 method: filtering method for transformed domain. Only Recursive Filtering is implimented, now.  

typedef enum  
{  
	DTF_RF=0,//Recursive Filtering  
	DTF_NC=1,//Normalized Convolution  
	DTF_IC=1,//Interpolated Convolution  
}DTF_METHOD;  



------------------------------------------------------
This code is forked from https://github.com/tatsy/ImageProcessing .  
pow operations are optimized by fmath.hpp (https://github.com/herumi/fmath) .  


