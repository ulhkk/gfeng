#include "stitching.h"
#include <iostream>

int main(int argc, char **argv){
    std::string path = argv[1];
    std::unique_ptr<image_stitcher> stitcher = std::make_unique<image_stitcher>();
    stitcher->loadImages(path);
    std::cout<<"stitching "<<stitcher->getNum()<<" images\n";
    return 0;
}