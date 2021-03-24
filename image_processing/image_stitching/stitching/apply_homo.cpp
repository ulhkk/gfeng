#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char** argv){

    std::string path = argv[1];
    std::vector<cv::Mat> imgs_;
    std::vector<cv::String> fileNames_;
    cv::glob(path, fileNames_, false);

    for (size_t i = 0; i < fileNames_.size(); i++){
        imgs_.push_back(cv::imread(fileNames_[i]));          
    }

    float homo_[3][3] = {{0.984923 ,-0.0107477 ,0.172663},
                         {0.0105862 ,0.999942, 0.00185575},
                         {-0.172673 ,-4.5675e-008, 0.984979}};
    cv::Mat h(3, 3, CV_32FC1, cv::Scalar(0));
    for(int i = 0;i < 3;i++){
        for(int j = 0;j < 3;j++){
            h.at<float>(i,j) = homo_[i][j];
        }
    }
    cv::Mat img;
    cvtColor(imgs_[0], img, cv::COLOR_BGR2GRAY);
    cv::Size pano_size_ = {2166,616};
    cv::Mat dst = img.clone();
    //cv::warpPerspective(img, dst, h, img.size());
    /*cv::Mat src = cv::imread("/home/gfeng/gfeng_github_ws/image_processing/image_stitching/data/lake/1.jpg", cv::IMREAD_COLOR);
    std::cout<<src.rows<<" "<<src.cols;
    cv::cvtColor(src,src,cv::COLOR_BGR2GRAY);
    cv::Size s = {512,384};
    cv::resize(src,src,s);
    cv::imshow("img",src);
    cv::waitKey(0);*/

    /*cv::Mat img(512,2048,CV_8UC1,cv::Scalar(0));
    cv::imshow("img", img);
    cv::waitKey(0);
    img.convertTo(img,CV_64F);*/
    //unsigned char a = 0;
    //std::cout<<!std::isprint(a)<<'\n';
    cv::imshow("pano",dst);
    cv::waitKey(0);
    return 0;
}