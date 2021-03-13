#include "stitching.h"

void image_stitcher::loadImages(const std::string& path){
    std::vector<cv::String> fileNames_;
    cv::glob(path, fileNames_, false);

    for (size_t i = 0; i < fileNames_.size(); i++){
        imgs_.push_back(cv::imread(fileNames_[i]));          
    }
    return;
}

int image_stitcher::getNum(){
    return imgs_.size();
}