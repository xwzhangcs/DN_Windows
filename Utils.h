#pragma once
#include <vector>
#include <map>
#include <tuple>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"

namespace util {
	double readNumber(const rapidjson::Value& node, const char* key, double default_value);
	std::vector<double> read1DArray(const rapidjson::Value& node, const char* key);
	bool readBoolValue(const rapidjson::Value& node, const char* key, bool default_value);
	std::string readStringValue(const rapidjson::Value& node, const char* key);
	//grammars
	std::vector<double> grammar1(std::string modeljson, std::vector<double> paras, bool bDebug);
	std::vector<double> grammar2(std::string modeljson, std::vector<double> paras, bool bDebug);
	std::vector<double> grammar3(std::string modeljson, std::vector<double> paras, bool bDebug);
	std::vector<double> grammar4(std::string modeljson, std::vector<double> paras, bool bDebug);
	std::vector<double> grammar5(std::string modeljson, std::vector<double> paras, bool bDebug);
	std::vector<double> grammar6(std::string modeljson, std::vector<double> paras, bool bDebug);
	//synthesis
	cv::Mat generateFacadeSynImage(int width, int height, int imageRows, int imageCols, int imageGroups, double imageRelativeWidth, double imageRelativeHeight);
	cv::Mat generateFacadeSynImage(int width, int height, int imageRows, int imageCols, int imageGroups, int imageDoors, double imageRelativeWidth, double imageRelativeHeight, double imageRelativeDWidth, double imageRelativeDHeight);

}