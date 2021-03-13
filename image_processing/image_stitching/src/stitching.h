#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/core/utility.hpp>

class image_stitcher{
public:
    void loadImages(const std::string& path);
    int getNum();

private:
    std::vector<cv::Mat> imgs_;
};
