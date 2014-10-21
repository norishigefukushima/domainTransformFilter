#include "domainTransformFilter.h"

#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
using namespace std;

#ifndef VK_ESCAPE
#define VK_ESCAPE 0x1B
#endif // VK_ESCAPE

 void smoothDemo(Mat& src)
 {
	 string wname = "smooth";
	namedWindow(wname);

	int sc = 50;
	int ss = 100;
	int iteration = 3;
	createTrackbar("sigma_color",wname,&sc,255);
	createTrackbar("sigma_space",wname,&ss,255);
	createTrackbar("iteration",wname,&iteration,10);

	int norm = 0;
	createTrackbar("normL1/L2",wname,&norm,1);
	int implimentation=2;
	createTrackbar("impliment",wname,&implimentation,2);

	int sw=2;
	createTrackbar("RF/NC/IC",wname,&sw,2);
	int key = 0;

	Mat show;
	while(key!='q' && key!=VK_ESCAPE)
	{
		int64 startTime = getTickCount();
		//domainTransformFilterRF(src, show,ss,sc,iteration);
		if(sw==0)
		{
			domainTransformFilter(src, show,ss,sc,iteration,norm+1,DTF_RF,implimentation);
		}
		else if(sw == 1)
		{
			domainTransformFilter(src, show,ss,sc,iteration,norm+1,DTF_NC,implimentation);
		}
		else
		{
			domainTransformFilter(src, show,ss,sc,iteration,norm+1,DTF_IC,implimentation);
		}
		
		double time = (getTickCount()-startTime)/(getTickFrequency());
		printf("domain transform filter: %f ms\n",time*1000.0);

		imshow(wname,show);
		key = waitKey(1);
	}

	destroyWindow(wname);
 }


void detailEnhancement(Mat& src)
{
	string wname = "detail enhancement";
	namedWindow(wname);

	int sc = 35;
	int ss = 100;
	int iteration = 2;
	int boost = 50;
	createTrackbar("sigma_color",wname,&sc,255);
	createTrackbar("sigma_space",wname,&ss,255);
	createTrackbar("iteration",wname,&iteration,255);
	createTrackbar("boost",wname,&boost,500);
	int key = 0;

	Mat show;
	Mat smooth;
	Mat sub;
	while(key!='q' && key!=VK_ESCAPE)
	{
		int64 startTime = getTickCount();

		domainTransformFilter(src, smooth,ss,sc,iteration,DTF_L1,DTF_RF,DTF_BGRA_SSE_PARALLEL);
		

		subtract(src,smooth,sub,noArray(),CV_32F);
		sub*=(boost*0.1);
		add(src,1*sub,sub,noArray(),CV_32F);
		sub.convertTo(show,CV_8U);

		double time = (getTickCount()-startTime)/(getTickFrequency());
		printf("domain transform filter with enhancement: %f ms\n",time*1000.0);

		imshow(wname,show);
		key = waitKey(1);
	}

	destroyWindow(wname);
}

int main(int argc, char** argv)
{
	Mat img = imread("statue.png");
	//Mat img = imread("kodim23.png");
	
	smoothDemo(img);

	detailEnhancement(img);

	return 0;
}