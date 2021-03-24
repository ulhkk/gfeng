#include "stitching.h"

void image_stitcher::loadImages(const std::string& path){
    std::vector<cv::String> fileNames_;
    cv::glob(path, fileNames_, false);

    for (size_t i = 0; i < fileNames_.size(); i++){
        imgs_.push_back(cv::imread(fileNames_[i]));          
    }
    return;
}

void image_stitcher::splitImages(){

    cv::Rect roiLeft_;
    roiLeft_.x = 1440;
    roiLeft_.y = 0;
    roiLeft_.width = imgs_[0].size().width - 1440;
    roiLeft_.height = imgs_[0].size().height;
    cv::Mat crop1 = imgs_[0](roiLeft_);
    cv::imshow("crop", crop1);
    cv::waitKey(0);
    imgs_[0] = crop1;

    cv::Rect roiRight_;
    roiRight_.x = 0;
    roiRight_.y = 0;
    roiRight_.width = imgs_[1].size().width - 1440;
    roiRight_.height = imgs_[1].size().height;
    cv::Mat crop2 = imgs_[1](roiRight_);
    cv::imshow("crop", crop2);
    cv::waitKey(0);
    imgs_[1] = crop2;
    
    return;
}

void image_stitcher::stitch_imgs(){

    cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
    cv::Ptr<cv::Stitcher> cur_stitcher = cv::Stitcher::create(mode);
    cv::Stitcher::Status status = cur_stitcher->stitch(imgs_, pano_);
    return;
}


int image_stitcher::getNum(){
    return imgs_.size();
}

cv::Mat image_stitcher::getPano(){
    return pano_;
}