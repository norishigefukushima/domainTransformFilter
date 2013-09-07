domainTransformFilter
=====================

domain transform filter for opencv

Eduardo S. L. Gastal and Manuel M. Oliveira. "Domain Transform for Edge-Aware Image and Video Processing". ACM Transactions on Graphics. Volume 30 (2011), Number 4, Proceedings of SIGGRAPH 2011, Article 69.
http://inf.ufrgs.br/~eslgastal/DomainTransform/

-----------------------------------------------

'void domainTransformFilter(cv::Mat& img, cv::Mat& out, double sigma_s, double sigma_r, int maxiter, int method=DTF_RF)'  
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

