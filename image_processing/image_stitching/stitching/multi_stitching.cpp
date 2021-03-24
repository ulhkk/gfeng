#include "stitching.h"
#include <iostream>




int main(int argc, char **argv){
    std::string path = argv[1];
    std::unique_ptr<image_stitcher> stitcher = std::make_unique<image_stitcher>();
    stitcher->loadImages(path);

    std::cout<<"stitching "<<stitcher->getNum()<<" images\n";

    stitcher->splitImages();

    stitcher->stitch_imgs();
    cv::Mat pano = stitcher->getPano();
    if(!pano.data){
        std::cout<<"failed to stitch images\n";
        return 0;
    }
    cv::resize(pano,pano,{1696,410});
    cv::imshow("pano",pano);
    cv::waitKey(0);
    return 0;
}