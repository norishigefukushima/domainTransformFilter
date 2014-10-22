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

	int sc = 500;
	int ss = 30;
	int iteration = 2;
	
	createTrackbar("sigma_color",wname,&sc,2550);
	createTrackbar("sigma_space",wname,&ss,100);
	createTrackbar("iteration",wname,&iteration,10);
	int norm = 0;
	createTrackbar("normL1/L2",wname,&norm,1);
	int implimentation=0;
	createTrackbar("impliment",wname,&implimentation,2);
	int sw=0;
	createTrackbar("RF/NC/IC",wname,&sw,2);
	int color = 0;
	createTrackbar("color",wname,&color,1);

	int key = 0;
	while(key!='q' && key!=VK_ESCAPE)
	{
		float scf = sc*0.1f;
		Mat show;
		Mat input;
		
		if(color==0) cvtColor(src,input,COLOR_BGR2GRAY);
		else input = src;
		
		int64 startTime = getTickCount();
		if(sw==0)
		{
			domainTransformFilter(input, show,scf,ss,iteration,norm+1,DTF_RF,implimentation);
		}
		else if(sw == 1)
		{
			domainTransformFilter(input, show,scf,ss,iteration,norm+1,DTF_NC,implimentation);
		}
		else if(sw == 2)
		{
			domainTransformFilter(input, show,scf,ss,iteration,norm+1,DTF_IC,implimentation);
		}

		double time = (getTickCount()-startTime)/(getTickFrequency());
		printf("domain transform filter: %f ms\n",time*1000.0);

		imshow(wname,show);
		key = waitKey(1);
	}

	destroyWindow(wname);
 }

 void jointSmoothDemo(Mat& src, Mat& guide)
 {
	 string wname = "smooth";
	 namedWindow(wname);

	 int sc = 500;
	 int ss = 30;
	 int iteration = 2;

	 createTrackbar("sigma_color",wname,&sc,2550);
	 createTrackbar("sigma_space",wname,&ss,100);
	 createTrackbar("iteration",wname,&iteration,10);
	 int norm = 0;
	 createTrackbar("normL1/L2",wname,&norm,1);
	 int implimentation=0;
	 createTrackbar("impliment",wname,&implimentation,2);
	 int sw=0;
	 createTrackbar("RF/NC/IC",wname,&sw,5);

	 int color = 0;
	 createTrackbar("color",wname,&color,1);

	 int key = 0;
	 while(key!='q' && key!=VK_ESCAPE)
	 {
		 float scf = sc*0.1f;
		 Mat show;
		 Mat input;

		 if(color==0) cvtColor(src,input,COLOR_BGR2GRAY);
		 else input = src;

		 int64 startTime = getTickCount();
		 if(sw==0)
		 {
			 domainTransformFilter(input,show,scf,ss,iteration,norm+1,DTF_RF,implimentation);
		 }
		 else if(sw == 2)
		 {
			 domainTransformFilter(input, show,scf,ss,iteration,norm+1,DTF_NC,implimentation);
		 }
		 else if(sw == 4)
		 {
			 domainTransformFilter(input, show,scf,ss,iteration,norm+1,DTF_IC,implimentation);
		 }
		 if(sw==1)
		 {
			 domainTransformFilter(input, guide,show,scf,ss,iteration,norm+1,DTF_RF,implimentation);
		 }
		 else if(sw == 3)
		 {
			 domainTransformFilter(input, guide, show,scf,ss,iteration,norm+1,DTF_NC,implimentation);
		 }
		 else if(sw == 5)
		 {
			 domainTransformFilter(input, guide, show,scf,ss,iteration,norm+1,DTF_IC,implimentation);
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
	Mat flash = imread("cave-flash.png");
	Mat noflash = imread("cave-noflash.png");
	
	smoothDemo(img);
	jointSmoothDemo(noflash,flash);

	detailEnhancement(img);

	return 0;
}