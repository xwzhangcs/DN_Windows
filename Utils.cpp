#include <iostream>
#include "Utils.h"

namespace util {

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

	std::vector<double> grammar1(std::string modeljson, std::vector<double> paras, bool bDebug) {
		FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
		char readBuffer[10240];
		rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
		rapidjson::Document docModel;
		docModel.ParseStream(isModel);
		rapidjson::Value& grammars = docModel["grammars"];
		rapidjson::Value& grammar = grammars["grammar1"];
		// range of Rows
		std::vector<double> tmp_array = read1DArray(grammar, "rangeOfRows");
		std::pair<int, int> imageRows(tmp_array[0], tmp_array[1]);
		// range of Cols
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "rangeOfCols");
		std::pair<int, int> imageCols(tmp_array[0], tmp_array[1]);
		// relativeWidth
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeWidth");
		std::pair<double, double> imageRelativeWidth(tmp_array[0], tmp_array[1]);
		// relativeHeight
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeHeight");
		std::pair<double, double> imageRelativeHeight(tmp_array[0], tmp_array[1]);
		if (bDebug) {
			std::cout << "imageRows is " << imageRows.first << ", " << imageRows.second << std::endl;
			std::cout << "imageCols is " << imageCols.first << ", " << imageCols.second << std::endl;
			std::cout << "imageRelativeWidth is " << imageRelativeWidth.first << ", " << imageRelativeWidth.second << std::endl;
			std::cout << "imageRelativeHeight is " << imageRelativeHeight.first << ", " << imageRelativeHeight.second << std::endl;
		}
		fclose(fp);
		int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
		if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.8)
			img_rows++;
		int img_cols = paras[1] * (imageCols.second - imageCols.first) + imageCols.first;
		if (paras[1] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.8)
			img_cols++;
		int img_groups = 1;
		double relative_width = paras[2] * (imageRelativeWidth.second - imageRelativeWidth.first) + imageRelativeWidth.first;
		double relative_height = paras[3] * (imageRelativeHeight.second - imageRelativeHeight.first) + imageRelativeHeight.first;
		std::vector<double> results;
		results.push_back(img_rows);
		results.push_back(img_cols);
		results.push_back(img_groups);
		results.push_back(relative_width);
		results.push_back(relative_height);
		return results;
	}

	std::vector<double> grammar2(std::string modeljson, std::vector<double> paras, bool bDebug) {
		FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
		char readBuffer[10240];
		rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
		rapidjson::Document docModel;
		docModel.ParseStream(isModel);
		rapidjson::Value& grammars = docModel["grammars"];
		rapidjson::Value& grammar = grammars["grammar2"];
		// range of Rows
		std::vector<double> tmp_array = read1DArray(grammar, "rangeOfRows");
		std::pair<int, int> imageRows(tmp_array[0], tmp_array[1]);
		// range of Cols
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "rangeOfCols");
		std::pair<int, int> imageCols(tmp_array[0], tmp_array[1]);
		// range of Doors
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "rangeOfDoors");
		std::pair<int, int> imageDoors(tmp_array[0], tmp_array[1]);
		// relativeWidth
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeWidth");
		std::pair<double, double> imageRelativeWidth(tmp_array[0], tmp_array[1]);
		// relativeHeight
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeHeight");
		std::pair<double, double> imageRelativeHeight(tmp_array[0], tmp_array[1]);
		// relativeDWidth
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeDWidth");
		std::pair<double, double> imageDRelativeWidth(tmp_array[0], tmp_array[1]);
		// relativeDHeight
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeDHeight");
		std::pair<double, double> imageDRelativeHeight(tmp_array[0], tmp_array[1]);
		if (bDebug) {
			std::cout << "imageRows is " << imageRows.first << ", " << imageRows.second << std::endl;
			std::cout << "imageCols is " << imageCols.first << ", " << imageCols.second << std::endl;
			std::cout << "imageDoors is " << imageDoors.first << ", " << imageDoors.second << std::endl;
			std::cout << "imageRelativeWidth is " << imageRelativeWidth.first << ", " << imageRelativeWidth.second << std::endl;
			std::cout << "imageRelativeHeight is " << imageRelativeHeight.first << ", " << imageRelativeHeight.second << std::endl;
			std::cout << "imageDRelativeWidth is " << imageDRelativeWidth.first << ", " << imageDRelativeWidth.second << std::endl;
			std::cout << "imageDRelativeHeight is " << imageDRelativeHeight.first << ", " << imageDRelativeHeight.second << std::endl;
		}
		fclose(fp);
		int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
		if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.8)
			img_rows++;
		int img_cols = paras[1] * (imageCols.second - imageCols.first) + imageCols.first;
		if (paras[1] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.8)
			img_cols++;
		int img_groups = 1;
		int img_doors = paras[2] * (imageDoors.second - imageDoors.first) + imageDoors.first;
		if (paras[2] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.8)
			img_doors++;
		double relative_width = paras[3] * (imageRelativeWidth.second - imageRelativeWidth.first) + imageRelativeWidth.first;
		double relative_height = paras[4] * (imageRelativeHeight.second - imageRelativeHeight.first) + imageRelativeHeight.first;
		double relative_door_width = paras[5] * (imageDRelativeWidth.second - imageDRelativeWidth.first) + imageDRelativeWidth.first;
		double relative_door_height = paras[6] * (imageDRelativeHeight.second - imageDRelativeHeight.first) + imageDRelativeHeight.first;
		std::vector<double> results;
		results.push_back(img_rows);
		results.push_back(img_cols);
		results.push_back(img_groups);
		results.push_back(img_doors);
		results.push_back(relative_width);
		results.push_back(relative_height);
		results.push_back(relative_door_width);
		results.push_back(relative_door_height);
		return results;
	}

	std::vector<double> grammar3(std::string modeljson, std::vector<double> paras, bool bDebug) {
		FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
		char readBuffer[10240];
		rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
		rapidjson::Document docModel;
		docModel.ParseStream(isModel);
		rapidjson::Value& grammars = docModel["grammars"];
		rapidjson::Value& grammar = grammars["grammar3"];
		// range of Cols
		std::vector<double> tmp_array = read1DArray(grammar, "rangeOfCols");
		std::pair<int, int> imageCols(tmp_array[0], tmp_array[1]);
		// relativeWidth
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeWidth");
		std::pair<double, double> imageRelativeWidth(tmp_array[0], tmp_array[1]);
		if (bDebug) {
			std::cout << "imageCols is " << imageCols.first << ", " << imageCols.second << std::endl;
			std::cout << "imageRelativeWidth is " << imageRelativeWidth.first << ", " << imageRelativeWidth.second << std::endl;
		}
		fclose(fp);
		int img_rows = 1;
		int img_cols = paras[0] * (imageCols.second - imageCols.first) + imageCols.first;
		if (paras[0] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.8)
			img_cols++;
		int img_groups = 1;
		double relative_width = paras[1] * (imageRelativeWidth.second - imageRelativeWidth.first) + imageRelativeWidth.first;
		double relative_height = 1.0;
		std::vector<double> results;
		results.push_back(img_rows);
		results.push_back(img_cols);
		results.push_back(img_groups);
		results.push_back(relative_width);
		results.push_back(relative_height);
		return results;
	}

	std::vector<double> grammar4(std::string modeljson, std::vector<double> paras, bool bDebug) {
		FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
		char readBuffer[10240];
		rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
		rapidjson::Document docModel;
		docModel.ParseStream(isModel);
		rapidjson::Value& grammars = docModel["grammars"];
		rapidjson::Value& grammar = grammars["grammar4"];
		// range of Rows
		std::vector<double> tmp_array = read1DArray(grammar, "rangeOfCols");
		std::pair<int, int> imageCols(tmp_array[0], tmp_array[1]);
		// range of Doors
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "rangeOfDoors");
		std::pair<int, int> imageDoors(tmp_array[0], tmp_array[1]);
		// relativeWidth
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeWidth");
		std::pair<double, double> imageRelativeWidth(tmp_array[0], tmp_array[1]);
		// relativeDWidth
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeDWidth");
		std::pair<double, double> imageDRelativeWidth(tmp_array[0], tmp_array[1]);
		// relativeDHeight
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeDHeight");
		std::pair<double, double> imageDRelativeHeight(tmp_array[0], tmp_array[1]);
		if (bDebug) {
			std::cout << "imageCols is " << imageCols.first << ", " << imageCols.second << std::endl;
			std::cout << "imageDoors is " << imageDoors.first << ", " << imageDoors.second << std::endl;
			std::cout << "imageRelativeWidth is " << imageRelativeWidth.first << ", " << imageRelativeWidth.second << std::endl;
			std::cout << "imageDRelativeWidth is " << imageDRelativeWidth.first << ", " << imageDRelativeWidth.second << std::endl;
			std::cout << "imageDRelativeHeight is " << imageDRelativeHeight.first << ", " << imageDRelativeHeight.second << std::endl;
		}
		fclose(fp);
		int img_rows = 1;;
		int img_cols = paras[0] * (imageCols.second - imageCols.first) + imageCols.first;
		if (paras[0] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.8)
			img_cols++;
		int img_groups = 1;
		int img_doors = paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first;
		if (paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.8)
			img_doors++;
		double relative_width = paras[2] * (imageRelativeWidth.second - imageRelativeWidth.first) + imageRelativeWidth.first;
		double relative_height = 1.0;
		double relative_door_width = paras[3] * (imageDRelativeWidth.second - imageDRelativeWidth.first) + imageDRelativeWidth.first;
		double relative_door_height = paras[4] * (imageDRelativeHeight.second - imageDRelativeHeight.first) + imageDRelativeHeight.first;
		std::vector<double> results;
		results.push_back(img_rows);
		results.push_back(img_cols);
		results.push_back(img_groups);
		results.push_back(img_doors);
		results.push_back(relative_width);
		results.push_back(relative_height);
		results.push_back(relative_door_width);
		results.push_back(relative_door_height);
		return results;
	}

	std::vector<double> grammar5(std::string modeljson, std::vector<double> paras, bool bDebug) {
		FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
		char readBuffer[10240];
		rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
		rapidjson::Document docModel;
		docModel.ParseStream(isModel);
		rapidjson::Value& grammars = docModel["grammars"];
		rapidjson::Value& grammar = grammars["grammar5"];
		// range of Rows
		std::vector<double> tmp_array = read1DArray(grammar, "rangeOfRows");
		std::pair<int, int> imageRows(tmp_array[0], tmp_array[1]);
		// relativeHeight
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeHeight");
		std::pair<double, double> imageRelativeHeight(tmp_array[0], tmp_array[1]);
		if (bDebug) {
			std::cout << "imageRows is " << imageRows.first << ", " << imageRows.second << std::endl;
			std::cout << "imageRelativeHeight is " << imageRelativeHeight.first << ", " << imageRelativeHeight.second << std::endl;
		}
		fclose(fp);
		int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
		if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.8)
			img_rows++;
		int img_cols = 1;
		int img_groups = 1;
		double relative_width = 1.0;
		double relative_height = paras[1] * (imageRelativeHeight.second - imageRelativeHeight.first) + imageRelativeHeight.first;
		std::vector<double> results;
		results.push_back(img_rows);
		results.push_back(img_cols);
		results.push_back(img_groups);
		results.push_back(relative_width);
		results.push_back(relative_height);
		return results;
	}

	std::vector<double> grammar6(std::string modeljson, std::vector<double> paras, bool bDebug) {
		FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
		char readBuffer[10240];
		rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
		rapidjson::Document docModel;
		docModel.ParseStream(isModel);
		rapidjson::Value& grammars = docModel["grammars"];
		rapidjson::Value& grammar = grammars["grammar6"];
		// range of Rows
		std::vector<double> tmp_array = read1DArray(grammar, "rangeOfRows");
		std::pair<int, int> imageRows(tmp_array[0], tmp_array[1]);
		// range of Doors
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "rangeOfDoors");
		std::pair<int, int> imageDoors(tmp_array[0], tmp_array[1]);
		// relativeHeight
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeHeight");
		std::pair<double, double> imageRelativeHeight(tmp_array[0], tmp_array[1]);
		// relativeDWidth
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeDWidth");
		std::pair<double, double> imageDRelativeWidth(tmp_array[0], tmp_array[1]);
		// relativeDHeight
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "relativeDHeight");
		std::pair<double, double> imageDRelativeHeight(tmp_array[0], tmp_array[1]);
		if (bDebug) {
			std::cout << "imageRows is " << imageRows.first << ", " << imageRows.second << std::endl;
			std::cout << "imageDoors is " << imageDoors.first << ", " << imageDoors.second << std::endl;
			std::cout << "imageRelativeHeight is " << imageRelativeHeight.first << ", " << imageRelativeHeight.second << std::endl;
			std::cout << "imageDRelativeWidth is " << imageDRelativeWidth.first << ", " << imageDRelativeWidth.second << std::endl;
			std::cout << "imageDRelativeHeight is " << imageDRelativeHeight.first << ", " << imageDRelativeHeight.second << std::endl;
		}
		fclose(fp);
		int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
		if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.8)
			img_rows++;
		int img_cols = 1;
		int img_groups = 1;
		int img_doors = paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first;
		if (paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.8)
			img_doors++;
		double relative_width = 1.0;
		double relative_height = paras[2] * (imageRelativeHeight.second - imageRelativeHeight.first) + imageRelativeHeight.first;
		double relative_door_width = paras[3] * (imageDRelativeWidth.second - imageDRelativeWidth.first) + imageDRelativeWidth.first;
		double relative_door_height = paras[4] * (imageDRelativeHeight.second - imageDRelativeHeight.first) + imageDRelativeHeight.first;
		std::vector<double> results;
		results.push_back(img_rows);
		results.push_back(img_cols);
		results.push_back(img_groups);
		results.push_back(img_doors);
		results.push_back(relative_width);
		results.push_back(relative_height);
		results.push_back(relative_door_width);
		results.push_back(relative_door_height);
		return results;
	}

	std::vector<double> eval_accuracy(const cv::Mat& seg_img, const cv::Mat& gt_img) {
		int gt_p = 0;
		int seg_tp = 0;
		int seg_fn = 0;
		int gt_n = 0;
		int seg_tn = 0;
		int seg_fp = 0;
		for (int i = 0; i < gt_img.size().height; i++) {
			for (int j = 0; j < gt_img.size().width; j++) {
				// wall
				if (gt_img.at<cv::Vec3b>(i, j)[0] == 0 && gt_img.at<cv::Vec3b>(i, j)[1] == 0 && gt_img.at<cv::Vec3b>(i, j)[2] == 255) {
					gt_p++;
					if (seg_img.at<cv::Vec3b>(i, j)[0] == 0 && seg_img.at<cv::Vec3b>(i, j)[1] == 0 && seg_img.at<cv::Vec3b>(i, j)[2] == 255) {
						seg_tp++;
					}
					else
						seg_fn++;
				}
				else {// non-wall
					gt_n++;
					if (seg_img.at<cv::Vec3b>(i, j)[0] == 255 && seg_img.at<cv::Vec3b>(i, j)[1] == 0 && seg_img.at<cv::Vec3b>(i, j)[2] == 0) {
						seg_tn++;
					}
					else
						seg_fp++;
				}
			}
		}
		// return pixel accuracy and class accuracy
		std::vector<double> eval_metrix;
		// accuracy 
		eval_metrix.push_back(1.0 * (seg_tp + seg_tn) / (gt_p + gt_n));
		// precision
		double precision = 1.0 * seg_tp / (seg_tp + seg_fp);
		// recall
		double recall = 1.0 * seg_tp / (seg_tp + seg_fn);
		eval_metrix.push_back(precision);
		eval_metrix.push_back(recall);
		/*std::cout << "P = " << gt_p << std::endl;
		std::cout << "N = " << gt_n << std::endl;
		std::cout << "TP = " << seg_tp << std::endl;
		std::cout << "FN = " << seg_fn << std::endl;
		std::cout << "TN = " << seg_tn << std::endl;
		std::cout << "FP = " << seg_fp << std::endl;*/
		return eval_metrix;
	}

	cv::Mat generateFacadeSynImage(int width, int height, int imageRows, int imageCols, int imageGroups, double imageRelativeWidth, double imageRelativeHeight) {
		cv::Scalar bg_color(255, 255, 255); // white back ground
		cv::Scalar window_color(0, 0, 0); // black for windows
		int NR = imageRows;
		int NC = imageCols;
		int NG = imageGroups;
		double ratioWidth = imageRelativeWidth;
		double ratioHeight = imageRelativeHeight;
		int thickness = -1;
		cv::Mat result(height, width, CV_8UC3, bg_color);
		double FH = height * 1.0 / NR;
		double FW = width * 1.0 / NC;
		double WH = FH * ratioHeight;
		double WW = FW * ratioWidth;
		/*std::cout << "NR is " << NR << std::endl;
		std::cout << "NC is " << NC << std::endl;
		std::cout << "FH is " << FH << std::endl;
		std::cout << "FW is " << FW << std::endl;
		std::cout << "ratioWidth is " << ratioWidth << std::endl;
		std::cout << "ratioHeight is " << ratioHeight << std::endl;
		std::cout << "WH is " << WH << std::endl;
		std::cout << "WW is " << WW << std::endl;*/
		// draw facade image
		if (NG == 1) {
			for (int i = 0; i < NR; ++i) {
				for (int j = 0; j < NC; ++j) {
					float x1 = (FW - WW) * 0.5 + FW * j;
					float y1 = (FH - WH) * 0.5 + FH * i;
					float x2 = x1 + WW;
					float y2 = y1 + WH;
					cv::rectangle(result, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), window_color, thickness);
				}
			}
		}
		else {
			double gap = 0.05 * WW; // Assume not too many windows in one group
			double GWW = (WW - gap * (NG - 1)) / NG;
			double GFW = GWW + gap;
			for (int i = 0; i < NR; ++i) {
				for (int j = 0; j < NC; ++j) {
					float x1 = (FW - WW) * 0.5 + FW * j;
					float y1 = (FH - WH) * 0.5 + FH * i;
					for (int k = 0; k < NG; k++) {
						float g_x1 = x1 + GFW * k;
						float g_y1 = y1;
						float g_x2 = g_x1 + GWW;
						float g_y2 = g_y1 + WH;

						cv::rectangle(result, cv::Point(std::round(g_x1), std::round(g_y1)), cv::Point(std::round(g_x2), std::round(g_y2)), window_color, thickness);
					}
				}
			}
		}
		return result;
	}

	cv::Mat generateFacadeSynImage_new(int width, int height, int imageRows, int imageCols, int imageGroups, double imageRelativeWidth, double imageRelativeHeight, double margin_t, double margin_b, double margin_l, double margin_r) {
		cv::Scalar bg_color(255, 255, 255); // white back ground
		cv::Scalar window_color(0, 0, 0); // black for windows
		int NR = imageRows;
		int NC = imageCols;
		int NG = imageGroups;
		double ratioWidth = imageRelativeWidth;
		double ratioHeight = imageRelativeHeight;
		int thickness = -1;
		cv::Mat result(height, width, CV_8UC3, bg_color);
		int height_valid = height - margin_t * height - margin_b * height;
		int width_valid = width - margin_l * width - margin_r * width;
		double FH = height_valid * 1.0 / NR;
		double FW = width_valid * 1.0 / NC;
		double WH = FH * ratioHeight;
		double WW = FW * ratioWidth;
		if (NC > 1)
			FW = WW + (width_valid - WW * NC) / (NC - 1);
		if (NR > 1)
			FH = WH + (height_valid - WH * NR) / (NR - 1);
		/*std::cout << "NR is " << NR << std::endl;
		std::cout << "NC is " << NC << std::endl;
		std::cout << "FH is " << FH << std::endl;
		std::cout << "FW is " << FW << std::endl;
		std::cout << "ratioWidth is " << ratioWidth << std::endl;
		std::cout << "ratioHeight is " << ratioHeight << std::endl;
		std::cout << "WH is " << WH << std::endl;
		std::cout << "WW is " << WW << std::endl;*/
		// draw facade image
		for (int i = 0; i < NR; ++i) {
			for (int j = 0; j < NC; ++j) {
				float x1 = FW * j + margin_l * width;
				float y1 = FH * i + margin_t * height;
				float x2 = x1 + WW;
				float y2 = y1 + WH;
				cv::rectangle(result, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), window_color, thickness);
			}
		}
		return result;
	}

	cv::Mat generateFacadeSynImage(int width, int height, int imageRows, int imageCols, int imageGroups, int imageDoors, double imageRelativeWidth, double imageRelativeHeight, double imageRelativeDWidth, double imageRelativeDHeight) {
		cv::Scalar bg_color(255, 255, 255); // white back ground
		cv::Scalar window_color(0, 0, 0); // black for windows
		int NR = imageRows;
		int NG = imageGroups;
		int NC = imageCols;
		int ND = imageDoors;
		double ratioWidth = imageRelativeWidth;
		double ratioHeight = imageRelativeHeight;
		double ratioDWidth = imageRelativeDWidth;
		double ratioDHeight = imageRelativeDHeight;
		int thickness = -1;
		cv::Mat result(height, width, CV_8UC3, bg_color);
		double DFW = width * 1.0 / ND;
		double DFH = height * ratioDHeight;
		double DW = DFW * ratioDWidth;
		double DH = DFH * 0.9;
		double FH = (height - DFH) * 1.0 / NR;
		double FW = width * 1.0 / NC;
		double WH = FH * ratioHeight;
		double WW = FW * ratioWidth;
		/*std::cout << "NR is " << NR << std::endl;
		std::cout << "NC is " << NC << std::endl;
		std::cout << "FH is " << FH << std::endl;
		std::cout << "FW is " << FW << std::endl;
		std::cout << "NG is " << NG << std::endl;
		std::cout << "ND is " << ND << std::endl;
		std::cout << "ratioWidth is " << ratioWidth << std::endl;
		std::cout << "ratioHeight is " << ratioHeight << std::endl;
		std::cout << "WH is " << WH << std::endl;
		std::cout << "WW is " << WW << std::endl;
		std::cout << "ratioDWidth is " << ratioDWidth << std::endl;
		std::cout << "ratioDHeight is " << ratioDHeight << std::endl;
		std::cout << "DH is " << DH << std::endl;
		std::cout << "DW is " << DW << std::endl;*/

		if (NG == 1) {
			// windows
			for (int i = 0; i < NR; ++i) {
				for (int j = 0; j < NC; ++j) {
					float x1 = (FW - WW) * 0.5 + FW * j;
					float y1 = (FH - WH) * 0.5 + FH * i;
					float x2 = x1 + WW;
					float y2 = y1 + WH;
					cv::rectangle(result, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), window_color, thickness);
				}
			}
			// doors
			for (int i = 0; i < ND; i++) {
				float x1 = (DFW - DW) * 0.5 + DFW * i;
				float y1 = height - DH;
				float x2 = x1 + DW;
				float y2 = y1 + DH;
				cv::rectangle(result, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), window_color, thickness);
			}
		}
		else {
			// windows
			double gap = 0.05 * WW; // Assume not too many windows in one group
			double GWW = (WW - gap * (NG - 1)) / NG;
			double GFW = GWW + gap;
			for (int i = 0; i < NR; ++i) {
				for (int j = 0; j < NC; ++j) {
					float x1 = (FW - WW) * 0.5 + FW * j;
					float y1 = (FH - WH) * 0.5 + FH * i;
					for (int k = 0; k < NG; k++) {
						float g_x1 = x1 + GFW * k;
						float g_y1 = y1;
						float g_x2 = g_x1 + GWW;
						float g_y2 = g_y1 + WH;
						cv::rectangle(result, cv::Point(std::round(g_x1), std::round(g_y1)), cv::Point(std::round(g_x2), std::round(g_y2)), window_color, thickness);
					}
				}
			}
			// doors
			for (int i = 0; i < ND; i++) {
				float x1 = (DFW - DW) * 0.5 + DFW * i;
				float y1 = height - DH;
				float x2 = x1 + DW;
				float y2 = y1 + DH;
				cv::rectangle(result, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), window_color, thickness);
			}
		}
		return result;
	}

	cv::Mat generateFacadeSynImage_new(int width, int height, int imageRows, int imageCols, int imageGroups, int imageDoors, double imageRelativeWidth, double imageRelativeHeight, double imageRelativeDWidth, double imageRelativeDHeight, double margin_t, double margin_b, double margin_l, double margin_r, double margin_d) {
		cv::Scalar bg_color(255, 255, 255); // white back ground
		cv::Scalar window_color(0, 0, 0); // black for windows
		int NR = imageRows;
		int NG = imageGroups;
		int NC = imageCols;
		int ND = imageDoors;
		double ratioWidth = imageRelativeWidth;
		double ratioHeight = imageRelativeHeight;
		double ratioDWidth = imageRelativeDWidth;
		double ratioDHeight = imageRelativeDHeight;
		int thickness = -1;
		cv::Mat result(height, width, CV_8UC3, bg_color);
		int width_valid = width - margin_l * width - margin_r * width;
		double DFW = width_valid * 1.0 / ND;
		double DFH = height * ratioDHeight;
		double DW = DFW * ratioDWidth;
		double DH = height * ratioDHeight;
		if (ND > 1)
			DFW = DW + (width_valid - DW * ND) / (ND - 1);
		int height_valid = height - margin_t * height - margin_b * height - margin_d * height - DFH;
		double FH = height_valid * 1.0 / NR;
		double FW = width_valid * 1.0 / NC;
		double WH = FH * ratioHeight;
		double WW = FW * ratioWidth;
		if (NC > 1)
			FW = WW + (width_valid - WW * NC) / (NC - 1);
		if (NR > 1)
			FH = WH + (height_valid - WH * NR) / (NR - 1);
		// windows
		for (int i = 0; i < NR; ++i) {
			for (int j = 0; j < NC; ++j) {
				float x1 = FW * j + margin_l * width;;
				float y1 = FH * i + margin_t * height;
				float x2 = x1 + WW;
				float y2 = y1 + WH;
				cv::rectangle(result, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), window_color, thickness);
			}
		}
		// doors
		for (int i = 0; i < ND; i++) {
			float x1 = DFW * i + margin_l * width;
			float y1 = height - DH - margin_b * height;
			float x2 = x1 + DW;
			float y2 = y1 + DH;
			cv::rectangle(result, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), window_color, thickness);
		}

		return result;
	}

}