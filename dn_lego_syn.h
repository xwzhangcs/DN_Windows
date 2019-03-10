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

cv::Mat generateFacadeSynImage(int width, int height, int imageRows, int imageCols, int imageGroups, double imageRelativeWidth, double imageRelativeHeight);
cv::Mat generateFacadeSynImage(int width, int height, int imageRows, int imageCols, int imageGroups, int imageDoors, double imageRelativeWidth, double imageRelativeHeight, double imageRelativeDWidth, double imageRelativeDHeight);

int find_threshold(cv::Mat src, bool bground);

/// Function header
std::vector<string> get_all_files_names_within_folder(std::string folder);

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

void process_single_chip(std::string metajson, std::string modeljson);
cv::Mat facade_clustering_kkmeans(cv::Mat src_img,  int clusters);

float findSkewAngle(cv::Mat src_img);