#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <Windows.h>

#include <dlib/clustering.h>
#include <dlib/rand.h>

using namespace dlib;

int const max_BINARY_value = 255;
int const cluster_number = 2;

/**** helper functions *****/
std::vector<int> clustersList(std::string metajson, int regionId, std::string modelType);
std::vector<string> get_all_files_names_within_folder(std::string folder);
cv::Mat facade_clustering_kkmeans(cv::Mat src_img, int clusters);

/**** steps *****/
bool chipping(std::string metajson, std::string modeljson, cv::Mat& croppedImage, bool bDebug, std::string img_filename);
cv::Mat crop_chip(cv::Mat src_chip, std::string modeljson, int type, bool bground, std::vector<double> facChip_size, double target_width, double target_height);
cv::Mat adjust_chip(cv::Mat src_chip, cv::Mat chip, int type, bool bground, std::vector<double> facChip_size, double target_width, double target_height);

bool segment_chip(cv::Mat croppedImage, cv::Mat& dnn_img, std::string metajson, std::string modeljson, bool bDebug, std::string img_filename);
float findSkewAngle(cv::Mat src_img);
cv::Mat cleanAlignedImage(cv::Mat src, float threshold);
cv::Mat deSkewImg(cv::Mat src_img);
void writeBackAvgColors(std::string metajson, cv::Scalar bg_avg_color, cv::Scalar win_avg_color);
cv::Rect findLargestRectangle(cv::Mat image);
bool findIntersection(cv::Rect a1, cv::Rect a2);
bool insideRect(cv::Rect a1, cv::Point p);

std::vector<double> feedDnn(cv::Mat dnn_img, std::string metajson, std::string modeljson, bool bDebug, std::string img_filename);

double compute_confidence(cv::Mat croppedImage, std::string modeljson, bool bDebug);

void synthesis(std::vector<double> predictions, cv::Size src_size, std::string dnnsOut_folder, cv::Scalar win_avg_color, cv::Scalar bg_avg_color, std::string img_filename, bool bDebug);
cv::Scalar readColor(std::string metajson, std::string color_name);
