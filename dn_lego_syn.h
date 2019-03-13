#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <Windows.h>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"

#include <dlib/clustering.h>
#include <dlib/rand.h>

using namespace dlib;

int const max_BINARY_value = 255;
int const cluster_number = 2;

double readNumber(const rapidjson::Value& node, const char* key, double default_value) {
	if (node.HasMember(key) && node[key].IsDouble()) {
		return node[key].GetDouble();
	}
	else if (node.HasMember(key) && node[key].IsInt()) {
		return node[key].GetInt();
	}
	else {
		return default_value;
	}
}

std::vector<double> read1DArray(const rapidjson::Value& node, const char* key) {
	std::vector<double> array_values;
	if (node.HasMember(key)) {
		const rapidjson::Value& data = node[key];
		array_values.resize(data.Size());
		for (int i = 0; i < data.Size(); i++)
			array_values[i] = data[i].GetDouble();
		return array_values;
	}
	else {
		return array_values;
	}
}

bool readBoolValue(const rapidjson::Value& node, const char* key, bool default_value) {
	if (node.HasMember(key) && node[key].IsBool()) {
		return node[key].GetBool();
	}
	else {
		return default_value;
	}
}

std::string readStringValue(const rapidjson::Value& node, const char* key) {
	if (node.HasMember(key) && node[key].IsString()) {
		return node[key].GetString();
	}
	else {
		throw "Could not read string from node";
	}
}

cv::Mat generateFacadeSynImage(int width, int height, int imageRows, int imageCols, int imageGroups, double imageRelativeWidth, double imageRelativeHeight);
cv::Mat generateFacadeSynImage(int width, int height, int imageRows, int imageCols, int imageGroups, int imageDoors, double imageRelativeWidth, double imageRelativeHeight, double imageRelativeDWidth, double imageRelativeDHeight);

/**** helper functions *****/
std::vector<int> clustersList(std::string metajson, int regionId, std::string modelType);
std::vector<string> get_all_files_names_within_folder(std::string folder);
cv::Mat facade_clustering_kkmeans(cv::Mat src_img, int clusters);

/**** steps *****/
bool chipping(std::string metajson, std::string modeljson, cv::Mat& croppedImage, bool bDebug, std::string img_filename);
cv::Mat crop_chip(cv::Mat src_chip, int type, bool bground, std::vector<double> facChip_size, double target_width, double target_height);
cv::Mat adjust_chip(cv::Mat src_chip, cv::Mat chip, int type, bool bground, std::vector<double> facChip_size, double target_width, double target_height);

bool segment_chip(cv::Mat croppedImage, cv::Mat& dnn_img, std::string metajson, std::string modeljson, bool bDebug, std::string img_filename);
float findSkewAngle(cv::Mat src_img);
cv::Mat cleanAlignedImage(cv::Mat src, float threshold);
void writeBackAvgColors(std::string metajson, bool bvalid, cv::Scalar bg_avg_color, cv::Scalar win_avg_color);

std::vector<double> feedDnn(cv::Mat dnn_img, std::string metajson, std::string modeljson, bool bDebug, std::string img_filename, int best_class);
std::vector<double> grammar1(std::string modeljson, std::vector<double> paras, bool bDebug);
std::vector<double> grammar2(std::string modeljson, std::vector<double> paras, bool bDebug);
std::vector<double> grammar3(std::string modeljson, std::vector<double> paras, bool bDebug);
std::vector<double> grammar4(std::string modeljson, std::vector<double> paras, bool bDebug);
std::vector<double> grammar5(std::string modeljson, std::vector<double> paras, bool bDebug);
std::vector<double> grammar6(std::string modeljson, std::vector<double> paras, bool bDebug);

void synthesis(std::vector<double> predictions, cv::Size src_size, std::string dnnsOut_folder, cv::Scalar win_avg_color, cv::Scalar bg_avg_color, std::string img_filename, bool bDebug);
cv::Scalar readColor(std::string metajson, std::string color_name);
