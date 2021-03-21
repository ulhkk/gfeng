#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main(){

    cv::Mat src = cv::imread("/home/gfeng/gfeng_github_ws/image_processing/image_stitching/data/lake/1.jpg", cv::IMREAD_COLOR);
    std::cout<<src.rows<<" "<<src.cols;
    cv::cvtColor(src,src,cv::COLOR_BGR2GRAY);
    cv::Size s = {512,384};
    cv::resize(src,src,s);
    cv::imshow("img",src);
    cv::waitKey(0);
    return 0;
}