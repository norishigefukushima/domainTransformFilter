#include "domainTransformFilter.h"

#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")


#ifndef VK_ESCAPE
#define VK_ESCAPE 0x1B
#endif // VK_ESCAPE

 void smoothDemo(Mat& src)
 {
	 string wname = "smooth";
	namedWindow(wname);

	int sc = 50;
	int ss = 100;
	int iteration = 2;
	createTrackbar("sigma_color",wname,&sc,255);
	createTrackbar("sigma_space",wname,&ss,255);
	createTrackbar("iteration",wname,&iteration,255);
	int key = 0;

	Mat show;
	while(key!='q' && key!=VK_ESCAPE)
	{
		int64 startTime = getTickCount();
		domainTransformFilter(src, show,ss,sc,iteration);

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

		domainTransformFilter(src, smooth,ss,sc,iteration);

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