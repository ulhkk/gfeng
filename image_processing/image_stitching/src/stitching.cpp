#include "stitching.h"

void image_stitcher::loadImages(const std::string& path){
    std::vector<cv::String> fileNames_;
    cv::glob(path, fileNames_, false);

    for (size_t i = 0; i < fileNames_.size(); i++){
        imgs_.push_back(cv::imread(fileNames_[i]));          
    }
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