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
std::vector<std::string> get_all_files_names_within_folder(std::string folder);
cv::Mat facade_clustering_kkmeans(cv::Mat src_img, int clusters);

/**** steps *****/
bool chipping(std::string metajson, std::string modeljson, cv::Mat& croppedImage, bool bMultipleChips, bool bDebug, std::string img_filename);
std::vector<cv::Mat> crop_chip(cv::Mat src_chip, std::string modeljson, int type, bool bground, std::vector<double> facChip_size, double target_width, double target_height, bool bMultipleChips);
cv::Mat adjust_chip(cv::Mat chip);
bool checkFacade(std::string facade_name);
void saveInvalidFacade(std::string metajson, std::string img_name, bool bDebug, std::string img_filename);

bool segment_chip(cv::Mat croppedImage, cv::Mat& dnn_img, std::string metajson, std::string modeljson, bool bDebug, std::string img_filename);
cv::Mat cleanAlignedImage(cv::Mat src, float threshold);
cv::Mat deSkewImg(cv::Mat src_img);
void writebackColor(std::string metajson, std::string attr, cv::Scalar color);
cv::Rect findLargestRectangle(cv::Mat image);
bool findIntersection(cv::Rect a1, cv::Rect a2);
bool insideRect(cv::Rect a1, cv::Point p);

std::vector<double> feedDnn(cv::Mat dnn_img, std::string metajson, std::string modeljson, bool bDebug, std::string img_filename);

std::vector<double> compute_confidence(cv::Mat croppedImage, std::string modeljson, bool bDebug);
std::vector<double> compute_door_paras(cv::Mat croppedImage, std::string modeljson, bool bDebug);

void synthesis(std::vector<double> predictions, cv::Size src_size, std::string dnnsOut_folder, cv::Scalar win_avg_color, cv::Scalar bg_avg_color, cv::Scalar win_histeq_color, cv::Scalar bg_histeq_color, std::string img_filename, bool bDebug);
cv::Scalar readColor(std::string metajson, std::string color_name);
bool readGround(std::string metajson);
double readScore(std::string metajson);
// For evaluating
void generateSegOutAndDnnOut(std::string chip_img_file, std::string modeljson, std::string segOut_file_name, std::string dnnOut_file_name, bool bDebug);

