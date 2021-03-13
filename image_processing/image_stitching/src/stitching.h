#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/core/cvstd.hpp>

class image_stitcher{
public:
    void loadImages(const std::string& path);
    void stitch_imgs();
    int getNum();
    cv::Mat getPano();
private:
    std::vector<cv::Mat> imgs_;
    cv::Mat pano_;
};
