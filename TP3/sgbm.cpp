#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <filesystem>

void processStereoImages(const std::string& inputDir, const std::string& outputDir, int numImages) {

    
    int preFilterCap = 63;
    int sadWindowSize = 3;
    int p1 = sadWindowSize * sadWindowSize * 4;
    int p2 = sadWindowSize * sadWindowSize * 32;
    int minDisparity = 0;
    int numDisparities = 32;
    int uniquenessRatio = 10;
    int speckleWindowSize = 100;
    int speckleRange = 32;
    int dispMaxDiff = 1;

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        minDisparity,
        numDisparities,
        sadWindowSize
    );

    sgbm->setPreFilterCap(preFilterCap);
    sgbm->setP1(p1);
    sgbm->setP2(p2);
    sgbm->setUniquenessRatio(uniquenessRatio);
    sgbm->setSpeckleWindowSize(speckleWindowSize);
    sgbm->setSpeckleRange(speckleRange);
    sgbm->setDisp12MaxDiff(dispMaxDiff);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);


    for (int i = 1; i <= numImages; ++i) {
        std::string leftImagePath = inputDir + "/" + std::to_string(i) + "_iLeft.jpg";
        std::string rightImagePath = inputDir + "/" + std::to_string(i) + "_iRight.jpg";

        cv::Mat leftImage = cv::imread(leftImagePath, cv::IMREAD_COLOR);
        cv::Mat rightImage = cv::imread(rightImagePath, cv::IMREAD_COLOR);

        cv::Mat disparity;
        sgbm->compute(leftImage, rightImage, disparity);

        std::string disparityPath = outputDir + "/" + std::to_string(i) + "_disparity.jpg";
        cv::imwrite(disparityPath, disparity);
    }
}

int main() {
    std::string inputDir = "../kitti15/";
    std::string outputDir = "../sgbm/";
    int numImages = 200;

    processStereoImages(inputDir, outputDir, numImages);

    return 0;
}
