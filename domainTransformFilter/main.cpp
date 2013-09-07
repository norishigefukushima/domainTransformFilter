#include "domainTransformFilter.h"

#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")

int main(int argc, char** argv)
{
	Mat img = imread("statue.png");

	string wname = "domaintransform";
	namedWindow(wname);

	int sc = 50;
	int ss = 100;
	int iteration = 2;
	createTrackbar("sigma_color",wname,&sc,255);
	createTrackbar("sigma_space",wname,&ss,255);
	createTrackbar("iteration",wname,&iteration,255);
	int key = 0;

	Mat show;
	while(key!='q')
	{
		int64 startTime = getTickCount();
		domainTransformFilter(img, show,ss,sc,iteration);

		double time = (getTickCount()-startTime)/(getTickFrequency());
		printf("domain transform filter: %f ms\n",time*1000.0);

		imshow(wname,show);
		key = waitKey(1);
	}

	return 0;
}