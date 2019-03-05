#include <torch/script.h> // One-stop header.

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

void facade_clustering_kkmeans(std::string in_img_file, std::string seg_img_file, std::string color_img_file, int clusters);
void facade_clustering_spectral(std::string in_img_file, std::string seg_img_file, std::string color_img_file, int clusters);
void eval_dataset_postprocessing(std::string label_img);
std::vector<double> eval_segmented_gt(std::string seg_img_file, std::string gt_img_file);
std::vector<double> eval_synthesis_gt(std::string seg_img_file, std::string gt_img_file);

void eval_different_segs(std::string result_file);
void create_different_segs();

int main(int argc, const char* argv[]) {
	if (argc != 3) {
		std::cerr << "usage: app <path-to-metadata> <path-to-model-config-JSON-file>\n";
		return -1;
	}
	/// Load an image
	eval_different_segs("../data/results.txt");
	//eval_segmented_gt("../data/test_seg_thre.png", "../data/test_gt.png");
	//facade_clustering_spectral("../data/test.png", "../data/test_seg.png", "../data/test_color.png", 2);
	return 0;
	std::string path(argv[1]);
	std::vector<std::string> metaFiles = get_all_files_names_within_folder(path);
	for (int i = 0; i < metaFiles.size(); i++) {
		std::string metajason = path + "/" + metaFiles[i];
		std::cout << metajason << std::endl;
		process_single_chip(metajason, argv[2]);
	}
	return 0;
}

void process_single_chip(std::string metajson, std::string modeljson) {
	// read image json file
	FILE* fp = fopen(metajson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document doc;
	doc.ParseStream(is);
	// size of chip
	std::vector<double> facChip_size = read1DArray(doc, "size");
	// ground
	bool bground = readBoolValue(doc, "ground", false);
	// image file
	std::string img_name = readStringValue(doc, "imagename");
	// score
	double score = readNumber(doc, "score", 0.2);
	fclose(fp);
	// first decide whether it's a valid chip
	bool bvalid = false;
	int type = 0;
	if (facChip_size[0] < 30.0 && facChip_size[0] > 8.0 && facChip_size[1] < 30.0 && facChip_size[1] > 8.0 && score > 0.95) {
		type = 1;
		bvalid = true;
	}
	else if (facChip_size[0] > 30.0 && facChip_size[1] < 30.0 && facChip_size[1] > 8.0 && score > 0.9) {
		type = 2;
		bvalid = true;
	}
	else if (facChip_size[0] < 30.0 && facChip_size[0] > 8.0 && facChip_size[1] > 30.0 && score > 0.9) {
		type = 3;
		bvalid = true;
	}
	else if (facChip_size[0] > 30.0 && facChip_size[1] > 30.0 && score > 0.7) {
		type = 4;
		bvalid = true;
	}
	else {
		// do nothing
	}

	if (!bvalid) {
		// write back to json file
		fp = fopen(metajson.c_str(), "wb"); // non-Windows use "w"
		rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
		doc.AddMember("valid", bvalid, alloc);
		// compute avg color
		cv::Scalar avg_color(0, 0, 0);
		cv::Mat src = cv::imread(img_name, 1);
		for (int i = 0; i < src.size().height; i++) {
			for (int j = 0; j < src.size().width; j++) {
				avg_color.val[0] += src.at<cv::Vec3b>(i, j)[0];
				avg_color.val[1] += src.at<cv::Vec3b>(i, j)[1];
				avg_color.val[2] += src.at<cv::Vec3b>(i, j)[2];
			}
		}
		avg_color.val[0] = avg_color.val[0] / (src.size().height * src.size().width);
		avg_color.val[1] = avg_color.val[1] / (src.size().height * src.size().width);
		avg_color.val[2] = avg_color.val[2] / (src.size().height * src.size().width);

		rapidjson::Value avg_color_json(rapidjson::kArrayType);
		avg_color_json.PushBack(avg_color.val[0], alloc);
		avg_color_json.PushBack(avg_color.val[1], alloc);
		avg_color_json.PushBack(avg_color.val[2], alloc);
		doc.AddMember("bg_color", avg_color_json, alloc);

		char writeBuffer[10240];
		rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
		rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
		doc.Accept(writer);
		fclose(fp);
		return;
	}
	// read model config json file
	fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	memset(readBuffer, 0, sizeof(readBuffer));
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	std::string model_name;
	std::string grammar_name;
	// flag debug
	bool bDebug = readBoolValue(docModel, "debug", false);
	if (bground) { // choose grammar2
		grammar_name = "grammar2";
	}
	else { // choose grammar1
		grammar_name = "grammar1";
	}
	rapidjson::Value& grammar = docModel[grammar_name.c_str()];
	// path of DN model
	model_name = readStringValue(grammar, "model");
	if(bDebug)
		std::cout << "model_name is " << model_name << std::endl;
	// number of paras
	int num_paras = readNumber(grammar, "number_paras", 5);
	if (bDebug)
		std::cout << "num_paras is " << num_paras << std::endl;
	// range of Rows
	std::vector<double> tmp_array = read1DArray(grammar, "rangeOfRows");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the rangeOfRows member in the JSON file" << std::endl;
		return;
	}
	std::pair<int, int> imageRows(tmp_array[0], tmp_array[1]);
	if (bDebug)
		std::cout << "imageRows is " << imageRows.first << ", " << imageRows.second << std::endl;
	// range of Cols
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "rangeOfCols");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the rangeOfCols member in the JSON file" << std::endl;
		return;
	}
	std::pair<int, int> imageCols(tmp_array[0], tmp_array[1]);
	if (bDebug)
		std::cout << "imageCols is " << imageCols.first << ", " << imageCols.second << std::endl;
	// range of Grouping
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "rangeOfGrouping");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the rangeOfGrouping member in the JSON file" << std::endl;
		return;
	}
	std::pair<int, int> imageGroups(tmp_array[0], tmp_array[1]);
	if (bDebug)
		std::cout << "imageGroups is " << imageGroups.first << ", " << imageGroups.second << std::endl;
	// default size for NN
	int height = 224; // DNN image height
	int width = 224; // DNN image width
	tmp_array.empty();
	tmp_array = read1DArray(docModel, "defaultSize");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the defaultSize member in the JSON file" << std::endl;
		return;
	}
	width = tmp_array[0];
	height = tmp_array[1];
	std::pair<int, int> imageDoors(2, 6);
	if (bground) {
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "rangeOfDoors");
		if (tmp_array.size() != 2) {
			std::cout << "Please check the rangeOfDoors member in the JSON file" << std::endl;
			return;
		}
		imageDoors.first = tmp_array[0];
		imageDoors.second = tmp_array[1];
		if (bDebug)
			std::cout << "imageDoors is " << imageDoors.first << ", " << imageDoors.second << std::endl;
	}
	tmp_array.empty();
	tmp_array = read1DArray(docModel, "targetChipSize");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the targetChipSize member in the JSON file" << std::endl;
		return;
	}
	double target_width = tmp_array[0];
	double target_height = tmp_array[1];
	// get facade folder path
	std::string facades_folder = readStringValue(docModel, "facadesFolder");
	// get facadehist folder path
	std::string facadeshist_folder = readStringValue(docModel, "facadehistFolder");
	// get chips folder path
	std::string chips_folder = readStringValue(docModel, "chipsFolder");
	// get chiphist folder path
	std::string chipshist_folder = readStringValue(docModel, "chiphistFolder");
	// get segs folder path
	std::string segs_folder = readStringValue(docModel, "segsFolder");
	// get segs folder path
	std::string segs_color_folder = readStringValue(docModel, "segsColorFolder");
	// get dnn folder path
	std::string dnns_folder = readStringValue(docModel, "dnnsFolder");
	// get threshold path
	std::string thresholds_file = readStringValue(docModel, "thresholds");
	// get ground info
	std::string grounds_file = readStringValue(docModel, "grounds");
	fclose(fp);
	// Deserialize the ScriptModule from a file using torch::jit::load().
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_name);
	module->to(at::kCUDA);
	assert(module != nullptr);

	// reshape the chip and pick the representative one
	double ratio_width, ratio_height;
	// image relative name
	std::size_t found = img_name.find("image/");
	if (found < 0) {
		std::cout << "found failed!!!" << std::endl;
		return;
	}
	found = found + 6;
	cv::Mat src_chip, dst_chip, croppedImage;
	if (bDebug)
		std::cout << "type is " << type << std::endl;
	if (type == 1) {
		src_chip = cv::imread(img_name);
		ratio_width = target_width / facChip_size[0] - 1;
		ratio_height = target_height / facChip_size[1] - 1;
		if (bDebug) {
			std::cout << "ratio_width is " << ratio_width << std::endl;
			std::cout << "ratio_height is " << ratio_height << std::endl;
		}
		int top = (int)(ratio_height * src_chip.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src_chip.cols);
		int borderType = cv::BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src_chip, dst_chip, top, bottom, left, right, borderType, value);
		croppedImage = dst_chip;
		cv::imwrite(chips_folder + "/" + img_name.substr(found), croppedImage);
	}
	else if (type == 2) {
		src_chip = cv::imread(img_name);
		int times = ceil(facChip_size[0] / target_width);
		ratio_width = (times * target_width - facChip_size[0]) / facChip_size[0];
		ratio_height = target_height / facChip_size[1] - 1;
		if (bDebug) {
			std::cout << "ratio_width is " << ratio_width << std::endl;
			std::cout << "ratio_height is " << ratio_height << std::endl;
		}
		int top = (int)(ratio_height * src_chip.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src_chip.cols);
		int borderType = cv::BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src_chip, dst_chip, top, bottom, left, right, borderType, value);
		// crop 30 * 30
		croppedImage = dst_chip(cv::Rect(dst_chip.size().width * 0.1, 0, dst_chip.size().width / times, dst_chip.size().height));
		cv::imwrite(chips_folder + "/" + img_name.substr(found), croppedImage);
	}
	else if (type == 3) {
		src_chip = cv::imread(img_name);
		int times = ceil(facChip_size[1] / target_height);
		ratio_height = (times * target_height - facChip_size[1]) / facChip_size[1];
		ratio_width = target_width / facChip_size[0] - 1;
		if (bDebug) {
			std::cout << "ratio_width is " << ratio_width << std::endl;
			std::cout << "ratio_height is " << ratio_height << std::endl;
		}
		int top = (int)(ratio_height * src_chip.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src_chip.cols);
		int borderType = cv::BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src_chip, dst_chip, top, bottom, left, right, borderType, value);
		// crop 30 * 30
		croppedImage = dst_chip(cv::Rect(0, dst_chip.size().height * (times - 1) / times, dst_chip.size().width, dst_chip.size().height / times));
		cv::imwrite(chips_folder + "/" + img_name.substr(found), croppedImage);
	}
	else if (type == 4) {
		src_chip = cv::imread(img_name);
		/*int times_width = ceil(facChip_size[0] / target_width);
		int times_height = ceil(facChip_size[1] / target_height);
		ratio_width = (times_width * target_width - facChip_size[0]) / facChip_size[0];
		ratio_height = (times_height * target_height - facChip_size[1]) / facChip_size[1];
		if (bDebug) {
			std::cout << "ratio_width is " << ratio_width << std::endl;
			std::cout << "ratio_height is " << ratio_height << std::endl;
		}
		int top = (int)(ratio_height * src_chip.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src_chip.cols);
		int borderType = cv::BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src_chip, dst_chip, top, bottom, left, right, borderType, value);*/
		// crop 30 * 30
		if (!bground) {
			double target_ratio_width = target_width / facChip_size[0];
			double target_ratio_height = target_height / facChip_size[1];
			double padding_width_ratio = (1 - target_ratio_width) * 0.5;
			double padding_height_ratio = (1 - target_ratio_height) * 0.5;
			croppedImage = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			cv::imwrite(chips_folder + "/" + img_name.substr(found), croppedImage);
		}
		else {
			double target_ratio_width = target_width / facChip_size[0];
			double target_ratio_height = target_height / facChip_size[1];
			double padding_width_ratio = (1 - target_ratio_width) * 0.5;
			double padding_height_ratio = (1 - target_ratio_height);
			croppedImage = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			cv::imwrite(chips_folder + "/" + img_name.substr(found), croppedImage);
		}
	}
	else {
		// do nothing
	}
	// for debugging
	if (bDebug) {
		cv::imwrite(facades_folder + "/" + img_name.substr(found), src_chip);
		cv::Mat img_histeq = cv::imread("../histeq/"+ img_name.substr(found));
		cv::imwrite(facadeshist_folder + "/" + img_name.substr(found), img_histeq);
	}
	// load image
	cv::Mat src, dst_ehist, dst_classify;
	//src = cv::imread(img_name, 1);
	src = croppedImage;
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	dst_ehist = bgr[2];
	// threshold classification
	int threshold = find_threshold(src, bground);
	if (bDebug) {
		std::ofstream out_param(thresholds_file, std::ios::app);
		out_param << img_name.substr(found);
		out_param << ",";
		out_param << threshold;
		out_param << "\n";

		std::ofstream out_param_ground(grounds_file, std::ios::app);
		out_param_ground << img_name.substr(found);
		out_param_ground << ",";
		out_param_ground << bground;
		out_param_ground << "\n";

		cv::imwrite(chipshist_folder + "/" + img_name.substr(found), dst_ehist);
	}
	
	cv::threshold(dst_ehist, dst_classify, threshold, max_BINARY_value, cv::THRESH_BINARY);
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(dst_classify, scale_img, cv::Size(width, height));

	// correct the color
	for (int i = 0; i < scale_img.size().height; i++) {
		for (int j = 0; j < scale_img.size().width; j++) {
			//noise
			if ((int)scale_img.at<uchar>(i, j) < 255) {
				scale_img.at<uchar>(i, j) = 0;
			}
		}
	}
	// add padding
	int padding_size = 5;
	int borderType = cv::BORDER_CONSTANT;
	cv::Scalar value(255, 255, 255);
	cv::Mat grid_dst;
	cv::copyMakeBorder(scale_img, scale_img, padding_size, padding_size, padding_size, padding_size, borderType, value);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(scale_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<cv::RotatedRect> minRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		minRect[i] = minAreaRect(cv::Mat(contours[i]));
	}
	cv::Mat dnn_img(scale_img.size(), CV_8UC3, bg_color);
	for (int i = 1; i< contours.size(); i++)
	{
		//if (hierarchy[i][2] != -1) continue;
		if (hierarchy[i][3] != 0) continue;
		cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x + 1, boundRect[i].tl().y + 1), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
		//cv::Point2f rect_points[4];
		//minRect[i].points(rect_points);
		//cv::Point vertices[4];
		//for (int i = 0; i < 4; ++i) {
		//	vertices[i] = rect_points[i];
		//}
		//for (int j = 0; j < 4; j++)
		//line(dnn_img, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
		//cv::fillConvexPoly(dnn_img,
		//	vertices,
		//	4,
		//	window_color);
	}
	// remove padding
	dnn_img = dnn_img(cv::Rect(padding_size, padding_size, width, height));

	cv::cvtColor(dnn_img, dnn_img, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);
	torch::Tensor out_tensor = module->forward(inputs).toTensor();
	std::cout << out_tensor.slice(1, 0, num_paras) << std::endl;
	std::vector<double> paras;
	for (int i = 0; i < num_paras; i++) {
		paras.push_back(out_tensor.slice(1, i, i + 1).item<float>());
	}
	// adjust paras
	for (int i = 0; i < num_paras; i++) {
		if (paras[i] > 1.0)
			paras[i] = 1.0;
		if (paras[i] < 0.0)
			paras[i] = 0.0;
	}
	// find the average color for window/non-window
	cv::Scalar bg_avg_color(0, 0, 0);
	cv::Scalar win_avg_color(0, 0, 0);
	{
		int bg_count = 0;
		int win_count = 0;
		for (int i = 0; i < dst_classify.size().height; i++) {
			for (int j = 0; j < dst_classify.size().width; j++) {
				if ((int)dst_classify.at<uchar>(i, j) == 0) {
					win_avg_color.val[0] += src.at<cv::Vec3b>(i, j)[0];
					win_avg_color.val[1] += src.at<cv::Vec3b>(i, j)[1];
					win_avg_color.val[2] += src.at<cv::Vec3b>(i, j)[2];
					win_count++;
				}
				else {
					bg_avg_color.val[0] += src.at<cv::Vec3b>(i, j)[0];
					bg_avg_color.val[1] += src.at<cv::Vec3b>(i, j)[1];
					bg_avg_color.val[2] += src.at<cv::Vec3b>(i, j)[2];
					bg_count++;
				}
			}
		}
		if (win_count > 0) {
			win_avg_color.val[0] = win_avg_color.val[0] / win_count;
			win_avg_color.val[1] = win_avg_color.val[1] / win_count;
			win_avg_color.val[2] = win_avg_color.val[2] / win_count;
		}
		if (bg_count > 0) {
			bg_avg_color.val[0] = bg_avg_color.val[0] / bg_count;
			bg_avg_color.val[1] = bg_avg_color.val[1] / bg_count;
			bg_avg_color.val[2] = bg_avg_color.val[2] / bg_count;
		}
	}
	// write back to json file
	fp = fopen(metajson.c_str(), "w"); // non-Windows use "w"
	rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
	doc.AddMember("valid", bvalid, alloc);

	rapidjson::Value bg_color_json(rapidjson::kArrayType);
	bg_color_json.PushBack(bg_avg_color.val[0], alloc);
	bg_color_json.PushBack(bg_avg_color.val[1], alloc);
	bg_color_json.PushBack(bg_avg_color.val[2], alloc);
	doc.AddMember("bg_color", bg_color_json, alloc);

	rapidjson::Value win_color_json(rapidjson::kArrayType);
	win_color_json.PushBack(win_avg_color.val[0], alloc);
	win_color_json.PushBack(win_avg_color.val[1], alloc);
	win_color_json.PushBack(win_avg_color.val[2], alloc);
	doc.AddMember("window_color", win_color_json, alloc);

	// predict img by DNN
	cv::Mat syn_img;
	if (!bground) {
		int img_rows = round(paras[0] * (imageRows.second - imageRows.first) + imageRows.first);
		int img_cols = round(paras[1] * (imageCols.second - imageCols.first) + imageCols.first);
		int img_groups = round(paras[2] * (imageGroups.second - imageGroups.first) + imageGroups.first);
		double relative_width = paras[3];
		double relative_height = paras[4];

		rapidjson::Value paras_json(rapidjson::kObjectType);
		paras_json.AddMember("rows", img_rows, alloc);
		paras_json.AddMember("cols", img_cols, alloc);
		paras_json.AddMember("grouping", img_groups, alloc);
		paras_json.AddMember("relativeWidth", relative_width, alloc);
		paras_json.AddMember("relativeHeight", relative_height, alloc);
		doc.AddMember("paras", paras_json, alloc);

		syn_img = generateFacadeSynImage(width, height, img_rows, img_cols, img_groups, relative_width, relative_height);
	}
	else {
		int img_rows = round(paras[0] * (imageRows.second - imageRows.first) + imageRows.first);
		int img_cols = round(paras[1] * (imageCols.second - imageCols.first) + imageCols.first);
		int img_groups = round(paras[2] * (imageGroups.second - imageGroups.first) + imageGroups.first);
		int img_doors = round(paras[3] * (imageDoors.second - imageDoors.first) + imageDoors.first);
		double relative_width = paras[4];
		double relative_height = paras[5];
		double relative_door_width = paras[6];
		double relative_door_height = paras[7];

		rapidjson::Value paras_json(rapidjson::kObjectType);
		paras_json.AddMember("rows", img_rows, alloc);
		paras_json.AddMember("cols", img_cols, alloc);
		paras_json.AddMember("grouping", img_groups, alloc);
		paras_json.AddMember("doors", img_doors, alloc);
		paras_json.AddMember("relativeWidth", relative_width, alloc);
		paras_json.AddMember("relativeHeight", relative_height, alloc);
		paras_json.AddMember("relativeDWidth", relative_door_width, alloc);
		paras_json.AddMember("relativeDHeight", relative_door_height, alloc);
		doc.AddMember("paras", paras_json, alloc);

		syn_img = generateFacadeSynImage(width, height, img_rows, img_cols, img_groups, img_doors, relative_width, relative_height, relative_door_width, relative_door_height);
	}
	char writeBuffer[10240];
	rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
	rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
	doc.Accept(writer);
	fclose(fp);

	// recover to the original image
	cv::resize(syn_img, syn_img, src.size());
	cv::resize(dnn_img, dnn_img, src.size());
	if (bDebug)
		cv::imwrite(segs_folder + "/" + img_name.substr(found), dst_classify);
	for (int i = 0; i < syn_img.size().height; i++) {
		for (int j = 0; j < syn_img.size().width; j++) {
			if (syn_img.at<cv::Vec3b>(i, j)[0] == 0) {
				syn_img.at<cv::Vec3b>(i, j)[0] = win_avg_color.val[0];
				syn_img.at<cv::Vec3b>(i, j)[1] = win_avg_color.val[1];
				syn_img.at<cv::Vec3b>(i, j)[2] = win_avg_color.val[2];
			}
			else {
				syn_img.at<cv::Vec3b>(i, j)[0] = bg_avg_color.val[0];
				syn_img.at<cv::Vec3b>(i, j)[1] = bg_avg_color.val[1];
				syn_img.at<cv::Vec3b>(i, j)[2] = bg_avg_color.val[2];
			}
		}
	}
	if (bDebug)
		cv::imwrite(dnns_folder + "/" + img_name.substr(found), syn_img);

	for (int i = 0; i < dnn_img.size().height; i++) {
		for (int j = 0; j < dnn_img.size().width; j++) {
			if (dnn_img.at<cv::Vec3b>(i, j)[0] == 0) {
				dnn_img.at<cv::Vec3b>(i, j)[0] = win_avg_color.val[0];
				dnn_img.at<cv::Vec3b>(i, j)[1] = win_avg_color.val[1];
				dnn_img.at<cv::Vec3b>(i, j)[2] = win_avg_color.val[2];
			}
			else {
				dnn_img.at<cv::Vec3b>(i, j)[0] = bg_avg_color.val[0];
				dnn_img.at<cv::Vec3b>(i, j)[1] = bg_avg_color.val[1];
				dnn_img.at<cv::Vec3b>(i, j)[2] = bg_avg_color.val[2];
			}
		}
	}
	if (bDebug)
		cv::imwrite(segs_color_folder + "/" + img_name.substr(found), dnn_img);
}

std::vector<double> eval_segmented_gt(std::string seg_img_file, std::string gt_img_file) {
	cv::Mat seg_img = cv::imread(seg_img_file);
	cv::Mat gt_img = cv::imread(gt_img_file);
	int gt_wall_num = 0;
	int seg_wall_tp = 0;
	int gt_non_wall_num = 0;
	int seg_non_wall_tp = 0;
	for (int i = 0; i < gt_img.size().height; i++) {
		for (int j = 0; j < gt_img.size().width; j++) {
			// wall
			if (gt_img.at<cv::Vec3b>(i, j)[0] == 0 && gt_img.at<cv::Vec3b>(i, j)[1] == 0 && gt_img.at<cv::Vec3b>(i, j)[2] == 255) {
				gt_wall_num++;
				if (seg_img.at<cv::Vec3b>(i, j)[0] == 255 && seg_img.at<cv::Vec3b>(i, j)[1] == 255 && seg_img.at<cv::Vec3b>(i, j)[2] == 255) {
					seg_wall_tp++;
				}
			}
			else {// non-wall
				gt_non_wall_num++;
				if (seg_img.at<cv::Vec3b>(i, j)[0] == 0 && seg_img.at<cv::Vec3b>(i, j)[1] == 0 && seg_img.at<cv::Vec3b>(i, j)[2] == 0) {
					seg_non_wall_tp++;
				}
			}
		}
	}
	// return pixel accuracy and class accuracy
	std::vector<double> eval_metrix;
	eval_metrix.push_back(1.0 * (seg_wall_tp + seg_non_wall_tp) / (gt_wall_num + gt_non_wall_num)); // pixel accuracy
	eval_metrix.push_back(1.0 * seg_wall_tp / gt_wall_num); // wall accuracy
	eval_metrix.push_back(1.0 * seg_non_wall_tp / gt_non_wall_num); // non-wall accuracy
	/*std::cout << "gt_wall_num is " << gt_wall_num << std::endl;
	std::cout << "gt_non_wall_num is " << gt_non_wall_num << std::endl;
	std::cout << "seg_wall_tp is " << seg_wall_tp << std::endl;
	std::cout << "seg_non_wall_tp is " << seg_non_wall_tp << std::endl;
	std::cout << "pixel accuracy is " << eval_metrix[0] << std::endl;
	std::cout << "wall accuracy is " << eval_metrix[1] << std::endl;
	std::cout << "non-wall accuracy is " << eval_metrix[2] << std::endl;*/
	return eval_metrix;
}

void eval_dataset_postprocessing(std::string label_img) {
	cv::Mat src_img = cv::imread(label_img);
	bool bContainDoor = false;
	for (int i = 0; i < src_img.size().height; i++) {
		for (int j = 0; j < src_img.size().width; j++) {
			if (src_img.at<cv::Vec3b>(i, j)[0] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 128 && src_img.at<cv::Vec3b>(i, j)[2] == 0)
				bContainDoor = true;
		}
	}
	if (!bContainDoor) {
		for (int i = 0; i < src_img.size().height; i++) {
			for (int j = 0; j < src_img.size().width; j++) {
				// window
				if (src_img.at<cv::Vec3b>(i, j)[0] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 0 && src_img.at<cv::Vec3b>(i, j)[2] == 128) {
					src_img.at<cv::Vec3b>(i, j)[0] = 255;
					src_img.at<cv::Vec3b>(i, j)[1] = 0;
					src_img.at<cv::Vec3b>(i, j)[2] = 0;
				}
				if (src_img.at<cv::Vec3b>(i, j)[0] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 0 && src_img.at<cv::Vec3b>(i, j)[2] == 0) {
					src_img.at<cv::Vec3b>(i, j)[0] = 0;
					src_img.at<cv::Vec3b>(i, j)[1] = 0;
					src_img.at<cv::Vec3b>(i, j)[2] = 255;
				}

			}
		}
	}
	if (bContainDoor) {
		for (int i = 0; i < src_img.size().height; i++) {
			for (int j = 0; j < src_img.size().width; j++) {
				// door
				if (src_img.at<cv::Vec3b>(i, j)[0] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 0 && src_img.at<cv::Vec3b>(i, j)[2] == 128) {
					src_img.at<cv::Vec3b>(i, j)[0] = 0;
					src_img.at<cv::Vec3b>(i, j)[1] = 255;
					src_img.at<cv::Vec3b>(i, j)[2] = 0;
				}
				// window
				if (src_img.at<cv::Vec3b>(i, j)[0] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 128 && src_img.at<cv::Vec3b>(i, j)[2] == 0) {
					src_img.at<cv::Vec3b>(i, j)[0] = 255;
					src_img.at<cv::Vec3b>(i, j)[1] = 0;
					src_img.at<cv::Vec3b>(i, j)[2] = 0;
				}
				if (src_img.at<cv::Vec3b>(i, j)[0] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 0 && src_img.at<cv::Vec3b>(i, j)[2] == 0) {
					src_img.at<cv::Vec3b>(i, j)[0] = 0;
					src_img.at<cv::Vec3b>(i, j)[1] = 0;
					src_img.at<cv::Vec3b>(i, j)[2] = 255;
				}

			}
		}
	}
	cv::imwrite(label_img, src_img);
}

void facade_clustering_kkmeans(std::string in_img_file, std::string seg_img_file, std::string color_img_file, int clusters) {
	// Here we declare that our samples will be 2 dimensional column vectors.  
	// (Note that if you don't know the dimensionality of your vectors at compile time
	// you can change the 2 to a 0 and then set the size at runtime)
	typedef matrix<double, 0, 1> sample_type;
	cv::Mat src_img = cv::imread(in_img_file, CV_LOAD_IMAGE_ANYCOLOR);
	std::cout << "src_img channels is " << src_img.channels() << std::endl;
	// Now we are making a typedef for the kind of kernel we want to use.  I picked the
	// radial basis kernel because it only has one parameter and generally gives good
	// results without much fiddling.
	typedef radial_basis_kernel<sample_type> kernel_type;


	// Here we declare an instance of the kcentroid object.  It is the object used to 
	// represent each of the centers used for clustering.  The kcentroid has 3 parameters 
	// you need to set.  The first argument to the constructor is the kernel we wish to 
	// use.  The second is a parameter that determines the numerical accuracy with which 
	// the object will perform part of the learning algorithm.  Generally, smaller values 
	// give better results but cause the algorithm to attempt to use more dictionary vectors 
	// (and thus run slower and use more memory).  The third argument, however, is the 
	// maximum number of dictionary vectors a kcentroid is allowed to use.  So you can use
	// it to control the runtime complexity.  
	kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 16);

	// Now we make an instance of the kkmeans object and tell it to use kcentroid objects
	// that are configured with the parameters from the kc object we defined above.
	kkmeans<kernel_type> test(kc);

	std::vector<sample_type> samples;
	std::vector<sample_type> initial_centers;

	sample_type m(src_img.channels());

	for (int i = 0; i < src_img.size().height; i++) {
		for (int j = 0; j < src_img.size().width; j++) {
			if (src_img.channels() == 3) {
				m(0) = src_img.at<cv::Vec3b>(i, j)[0] * 1.0 / 255;
				m(1) = src_img.at<cv::Vec3b>(i, j)[1] * 1.0 / 255;
				m(2) = src_img.at<cv::Vec3b>(i, j)[2] * 1.0 / 255;
			}
			else {
				m(0) = (int)src_img.at<uchar>(i, j) * 1.0 / 255;
			}
			// add this sample to our set of samples we will run k-means 
			samples.push_back(m);
		}
	}

	// tell the kkmeans object we made that we want to run k-means with k set to 3. 
	// (i.e. we want 3 clusters)
	test.set_number_of_centers(clusters);

	// You need to pick some initial centers for the k-means algorithm.  So here
	// we will use the dlib::pick_initial_centers() function which tries to find
	// n points that are far apart (basically).  
	pick_initial_centers(clusters, initial_centers, samples, test.get_kernel());

	// now run the k-means algorithm on our set of samples.  
	test.train(samples, initial_centers);

	cv::Mat out_img(src_img.size().height, src_img.size().width, CV_8UC3, cv::Scalar(255, 255, 255));
	std::vector<cv::Scalar> clusters_colors;
	std::vector<cv::Scalar> clusters_colors_seg;
	std::vector<int> clusters_points;
	clusters_colors.resize(clusters);
	clusters_colors_seg.resize(clusters);
	clusters_points.resize(clusters);
	for (int i = 0; i < clusters; i++) {
		clusters_colors_seg[i] = cv::Scalar(0, 0, 0);
		clusters_colors[i] = cv::Scalar(0, 0, 0);
		clusters_points[i] = 0;
	}
	int count = 0;
	// 
	if (src_img.channels() == 3) {
		count = 0;
		for (int i = 0; i < src_img.size().height; i++) {
			for (int j = 0; j < src_img.size().width; j++) {
				clusters_colors[test(samples[count])][0] += src_img.at<cv::Vec3b>(i, j)[0];
				clusters_colors[test(samples[count])][1] += src_img.at<cv::Vec3b>(i, j)[1];
				clusters_colors[test(samples[count])][2] += src_img.at<cv::Vec3b>(i, j)[2];
				clusters_points[test(samples[count])] ++;
				count++;
			}
		}
		for (int i = 0; i < clusters; i++) {
			clusters_colors[i][0] = clusters_colors[i][0] / clusters_points[i];
			clusters_colors[i][1] = clusters_colors[i][1] / clusters_points[i];
			clusters_colors[i][2] = clusters_colors[i][2] / clusters_points[i];
		}
	}
	else if (src_img.channels() == 1) { //gray image
		int count = 0;
		for (int i = 0; i < src_img.size().height; i++) {
			for (int j = 0; j < src_img.size().width; j++) {
				clusters_colors[test(samples[count])][0] += (int)src_img.at<uchar>(i, j);
				clusters_points[test(samples[count])] ++;
				count++;
			}
		}
		for (int i = 0; i < clusters; i++) {
			clusters_colors[i][0] = clusters_colors[i][0] / clusters_points[i];
		}
	}
	else {
		//do nothing
	}

	cv::Mat color_img;
	cv::resize(src_img, color_img, cv::Size(src_img.size().width, src_img.size().height));
	if (src_img.channels() == 3) {
		int count = 0;
		for (int i = 0; i < color_img.size().height; i++) {
			for (int j = 0; j < color_img.size().width; j++) {
				color_img.at<cv::Vec3b>(i, j)[0] = clusters_colors[test(samples[count])][0];
				color_img.at<cv::Vec3b>(i, j)[1] = clusters_colors[test(samples[count])][1];
				color_img.at<cv::Vec3b>(i, j)[2] = clusters_colors[test(samples[count])][2];
				count++;
			}
		}
	}
	else if (src_img.channels() == 1) { //gray image
		int count = 0;
		for (int i = 0; i < color_img.size().height; i++) {
			for (int j = 0; j < color_img.size().width; j++) {
				color_img.at<uchar>(i, j) = (uchar)clusters_colors[test(samples[count])][0];
				count++;
			}
		}
	}
	else {
		//do nothing
	}
	imwrite(color_img_file, color_img);

	// compute cluster colors
	int darkest_cluster = -1;
	cv::Scalar darkest_color(255, 255, 255);
	for (int i = 0; i < clusters; i++) {
		std::cout << "clusters_colors " << i << " is " << clusters_colors[i] << std::endl;
		if (src_img.channels() == 3) {
			if (clusters_colors[i][0] < darkest_color[0] && clusters_colors[i][1] < darkest_color[1] && clusters_colors[i][2] < darkest_color[2]) {
				darkest_color[0] = clusters_colors[i][0];
				darkest_color[1] = clusters_colors[i][1];
				darkest_color[2] = clusters_colors[i][2];
				darkest_cluster = i;
			}
		}
		else {
			if (clusters_colors[i][0] < darkest_color[0]) {
				darkest_color[0] = clusters_colors[i][0];
				darkest_cluster = i;
			}
		}
	}
	count = 0;
	for (int i = 0; i < out_img.size().height; i++) {
		for (int j = 0; j < out_img.size().width; j++) {
			if (test(samples[count]) == darkest_cluster) {
				out_img.at<cv::Vec3b>(i, j)[0] = 0;
				out_img.at<cv::Vec3b>(i, j)[1] = 0;
				out_img.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else {
				out_img.at<cv::Vec3b>(i, j)[0] = 255;
				out_img.at<cv::Vec3b>(i, j)[1] = 255;
				out_img.at<cv::Vec3b>(i, j)[2] = 255;

			}
			count++;
		}
	}
	imwrite(seg_img_file, out_img);
}

void facade_clustering_spectral(std::string in_img_file, std::string seg_img_file, std::string color_img_file, int clusters) {
	// Here we declare that our samples will be 2 dimensional column vectors.  
	// (Note that if you don't know the dimensionality of your vectors at compile time
	// you can change the 2 to a 0 and then set the size at runtime)
	typedef matrix<double, 0, 1> sample_type;
	cv::Mat src_img = cv::imread(in_img_file, CV_LOAD_IMAGE_ANYCOLOR);
	std::cout << "src_img channels is " << src_img.channels() << std::endl;
	// Now we are making a typedef for the kind of kernel we want to use.  I picked the
	// radial basis kernel because it only has one parameter and generally gives good
	// results without much fiddling.
	typedef radial_basis_kernel<sample_type> kernel_type;
	std::vector<sample_type> samples;

	sample_type m(src_img.channels());

	for (int i = 0; i < src_img.size().height; i++) {
		for (int j = 0; j < src_img.size().width; j++) {
			if (src_img.channels() == 3) {
				m(0) = src_img.at<cv::Vec3b>(i, j)[0] * 1.0 / 255;
				m(1) = src_img.at<cv::Vec3b>(i, j)[1] * 1.0 / 255;
				m(2) = src_img.at<cv::Vec3b>(i, j)[2] * 1.0 / 255;
			}
			else {
				m(0) = (int)src_img.at<uchar>(i, j) * 1.0 / 255;
			}
			// add this sample to our set of samples we will run k-means 
			samples.push_back(m);
		}
	}

	// Finally, we can also solve the same kind of non-linear clustering problem with
	// spectral_cluster().  The output is a vector that indicates which cluster each sample
	// belongs to.  Just like with kkmeans, it assigns each point to the correct cluster.
	std::vector<unsigned long> assignments = spectral_cluster(kernel_type(0.1), samples, clusters);

	cv::Mat out_img(src_img.size().height, src_img.size().width, CV_8UC3, cv::Scalar(255, 255, 255));
	std::vector<cv::Scalar> clusters_colors;
	std::vector<cv::Scalar> clusters_colors_seg;
	std::vector<int> clusters_points;
	clusters_colors.resize(clusters);
	clusters_colors_seg.resize(clusters);
	clusters_points.resize(clusters);
	for (int i = 0; i < clusters; i++) {
		clusters_colors_seg[i] = cv::Scalar(0, 0, 0);
		clusters_colors[i] = cv::Scalar(0, 0, 0);
		clusters_points[i] = 0;
	}
	int count = 0;

	// 
	if (src_img.channels() == 3) {
		count = 0;
		for (int i = 0; i < src_img.size().height; i++) {
			for (int j = 0; j < src_img.size().width; j++) {
				clusters_colors[assignments[count]][0] += src_img.at<cv::Vec3b>(i, j)[0];
				clusters_colors[assignments[count]][1] += src_img.at<cv::Vec3b>(i, j)[1];
				clusters_colors[assignments[count]][2] += src_img.at<cv::Vec3b>(i, j)[2];
				clusters_points[assignments[count]] ++;
				count++;
			}
		}
		for (int i = 0; i < clusters; i++) {
			clusters_colors[i][0] = clusters_colors[i][0] / clusters_points[i];
			clusters_colors[i][1] = clusters_colors[i][1] / clusters_points[i];
			clusters_colors[i][2] = clusters_colors[i][2] / clusters_points[i];
		}
	}
	else if (src_img.channels() == 1) { //gray image
		int count = 0;
		for (int i = 0; i < src_img.size().height; i++) {
			for (int j = 0; j < src_img.size().width; j++) {
				clusters_colors[assignments[count]][0] += (int)src_img.at<uchar>(i, j);
				clusters_points[assignments[count]] ++;
				count++;
			}
		}
		for (int i = 0; i < clusters; i++) {
			clusters_colors[i][0] = clusters_colors[i][0] / clusters_points[i];
		}
	}
	else {
		//do nothing
	}

	cv::Mat color_img;
	cv::resize(src_img, color_img, cv::Size(src_img.size().width, src_img.size().height));
	if (src_img.channels() == 3) {
		int count = 0;
		for (int i = 0; i < color_img.size().height; i++) {
			for (int j = 0; j < color_img.size().width; j++) {
				color_img.at<cv::Vec3b>(i, j)[0] = clusters_colors[assignments[count]][0];
				color_img.at<cv::Vec3b>(i, j)[1] = clusters_colors[assignments[count]][1];
				color_img.at<cv::Vec3b>(i, j)[2] = clusters_colors[assignments[count]][2];
				count++;
			}
		}
	}
	else if (src_img.channels() == 1) { //gray image
		int count = 0;
		for (int i = 0; i < color_img.size().height; i++) {
			for (int j = 0; j < color_img.size().width; j++) {
				color_img.at<uchar>(i, j) = (uchar)clusters_colors[assignments[count]][0];
				count++;
			}
		}
	}
	else {
		//do nothing
	}
	imwrite(color_img_file, color_img);

	// compute cluster colors
	int darkest_cluster = -1;
	cv::Scalar darkest_color(255, 255, 255);
	for (int i = 0; i < clusters; i++) {
		std::cout << "clusters_colors " << i << " is " << clusters_colors[i] << std::endl;
		if (src_img.channels() == 3) {
			if (clusters_colors[i][0] < darkest_color[0] && clusters_colors[i][1] < darkest_color[1] && clusters_colors[i][2] < darkest_color[2]) {
				darkest_color[0] = clusters_colors[i][0];
				darkest_color[1] = clusters_colors[i][1];
				darkest_color[2] = clusters_colors[i][2];
				darkest_cluster = i;
			}
		}
		else {
			if (clusters_colors[i][0] < darkest_color[0]) {
				darkest_color[0] = clusters_colors[i][0];
				darkest_cluster = i;
			}
		}
	}
	count = 0;
	for (int i = 0; i < out_img.size().height; i++) {
		for (int j = 0; j < out_img.size().width; j++) {
			if (assignments[count] == darkest_cluster) {
				out_img.at<cv::Vec3b>(i, j)[0] = 0;
				out_img.at<cv::Vec3b>(i, j)[1] = 0;
				out_img.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else {
				out_img.at<cv::Vec3b>(i, j)[0] = 255;
				out_img.at<cv::Vec3b>(i, j)[1] = 255;
				out_img.at<cv::Vec3b>(i, j)[2] = 255;

			}
			count++;
		}
	}
	imwrite(seg_img_file, out_img);
}

std::vector<string> get_all_files_names_within_folder(std::string folder)
{
	std::vector<string> names;
	string search_path = folder + "/*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			/*if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}*/
			if (string(fd.cFileName).compare(".") != 0 && string(fd.cFileName).compare("..") != 0)
				names.push_back(fd.cFileName);
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

int find_threshold(cv::Mat src, bool bground) {
	//Convert pixel values to other color spaces.
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	/// Load an image
	cv::Mat src_gray = bgr[2];
	for (int threshold = 40; threshold < 160; threshold += 5) {
		cv::Mat dst;
		cv::threshold(src_gray, dst, threshold, max_BINARY_value, cv::THRESH_BINARY);
		int count = 0;
		for (int i = 0; i < dst.size().height; i++) {
			for (int j = 0; j < dst.size().width; j++) {
				//noise
				if ((int)dst.at<uchar>(i, j) == 0) {
					count++;
				}
			}
		}
		float percentage = count * 1.0 / (dst.size().height * dst.size().width);
		std::cout << "percentage is " << percentage << std::endl;
		if (percentage > 0.25 && !bground)
			return threshold;
		if (percentage > 0.25 && bground)
			return threshold;
	}
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
	std::cout << "NR is " << NR << std::endl;
	std::cout << "NC is " << NC << std::endl;
	std::cout << "FH is " << FH << std::endl;
	std::cout << "FW is " << FW << std::endl;
	std::cout << "ratioWidth is " << ratioWidth << std::endl;
	std::cout << "ratioHeight is " << ratioHeight << std::endl;
	std::cout << "WH is " << WH << std::endl;
	std::cout << "WW is " << WW << std::endl;
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
	std::cout << "NR is " << NR << std::endl;
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
	std::cout << "DW is " << DW << std::endl;

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

void create_different_segs() {
	std::string path("../data/val");
	std::vector<std::string> facades = get_all_files_names_within_folder(path);
	for (int i = 0; i < facades.size(); i++) {
		// origin
		std::string facade = path + "/" + facades[i] + "/img.png";
		std::string facade_seg = path + "/" + facades[i] + "/seg.png";
		std::string facade_color = path + "/" + facades[i] + "/color.png";
		std::cout << facade << std::endl;
		std::cout << facade_seg << std::endl;
		std::cout << facade_color << std::endl;
		//facade_clustering_kkmeans(facade, facade_seg, facade_color, 3);

		// bgr histeq
		cv::Mat src = cv::imread(facade);
		std::vector<cv::Mat> bgr;   //destination array
		cv::split(src, bgr);//split source 
		for (int i = 0; i < 3; i++)
			cv::equalizeHist(bgr[i], bgr[i]);
		cv::Mat dst;
		cv::merge(bgr, dst);
		std::string facade_bgr_histeq = path + "/" + facades[i] + "/img_bgr_histeq.png";
		facade_seg = path + "/" + facades[i] + "/seg_bgr_histeq.png";
		facade_color = path + "/" + facades[i] + "/color_bgr_histeq.png";
		cv::imwrite(facade_bgr_histeq, dst);
		//facade_clustering_kkmeans(facade_bgr_histeq, facade_seg, facade_color, 3);

		// hsv histeq
		cv::Mat hsv;
		cvtColor(src, hsv, cv::COLOR_BGR2HSV);
		std::vector<cv::Mat> channels;   //destination array
		cv::split(hsv, channels);//split source 
		for (int i = 0; i < 3; i++)
			cv::equalizeHist(channels[i], channels[i]);
		//cv::merge(bgr, dst);	
		/// Load an image
		cv::imwrite("../data/test_hsv.png", channels[2]);
		std::string facade_hsv = path + "/" + facades[i] + "/img_hsv_histeq.png";
		facade_seg = path + "/" + facades[i] + "/seg_hsv_histeq.png";
		facade_color = path + "/" + facades[i] + "/color_hsv_histeq.png";
		cv::imwrite(facade_hsv, channels[2]);
		//facade_clustering_kkmeans(facade_hsv, facade_seg, facade_color, 3);

		// threshold seg
		int threshold = find_threshold(src, false);
		cv::Mat thre_seg;
		cv::threshold(channels[2], thre_seg, threshold, max_BINARY_value, cv::THRESH_BINARY);
		facade_seg = path + "/" + facades[i] + "/seg_thre.png";
		cv::imwrite(facade_seg, thre_seg);
	}
}

void eval_different_segs(std::string result_file) {

	std::string path("../data/val");
	std::ofstream out_param(result_file, std::ios::app);
	out_param << "facade_id";
	out_param << ",";
	out_param << "seg_pixel_accuracy";
	out_param << ",";
	out_param << "seg_wall_accuracy";
	out_param << ",";
	out_param << "seg_non_wall_accuracy";
	out_param << ",";
	out_param << "bgr_pixel_accuracy";
	out_param << ",";
	out_param << "bgr_wall_accuracy";
	out_param << ",";
	out_param << "bgr_non_wall_accuracy";
	out_param << ",";
	out_param << "hsv_pixel_accuracy";
	out_param << ",";
	out_param << "hsv_wall_accuracy";
	out_param << ",";
	out_param << "hsv_non_wall_accuracy";
	out_param << ",";
	out_param << "thre_pixel_accuracy";
	out_param << ",";
	out_param << "thre_wall_accuracy";
	out_param << ",";
	out_param << "thre_non_wall_accuracy";
	out_param << "\n";
	std::vector<std::string> facades = get_all_files_names_within_folder(path);
	for (int i = 0; i < facades.size(); i++) {
		// origin
		std::string facade = path + "/" + facades[i] + "/img.png";
		std::string facade_gt = path + "/" + facades[i] + "/label.png";
		std::string facade_seg = path + "/" + facades[i] + "/seg.png";
		std::string facade_seg_bgr = path + "/" + facades[i] + "/seg_bgr_histeq.png";
		std::string facade_seg_hsv = path + "/" + facades[i] + "/seg_hsv_histeq.png";
		std::string facade_seg_thre = path + "/" + facades[i] + "/seg_thre.png";
		std::cout << facade << std::endl;
		std::cout << facade_seg << std::endl;
		std::cout << facade_seg_bgr << std::endl;
		std::cout << facade_seg_hsv << std::endl;
		std::cout << facade_seg_thre << std::endl;

		std::vector<double> results = eval_segmented_gt(facade_seg, facade_gt);
		out_param << facades[i];
		out_param << ",";
		out_param << results[0];
		out_param << ",";
		out_param << results[1];
		out_param << ",";
		out_param << results[2];
		results.clear();
		results = eval_segmented_gt(facade_seg_bgr, facade_gt);
		out_param << ",";
		out_param << results[0];
		out_param << ",";
		out_param << results[1];
		out_param << ",";
		out_param << results[2];
		results.clear();
		results = eval_segmented_gt(facade_seg_hsv, facade_gt);
		out_param << ",";
		out_param << results[0];
		out_param << ",";
		out_param << results[1];
		out_param << ",";
		out_param << results[2];
		results.clear();
		results = eval_segmented_gt(facade_seg_thre, facade_gt);
		out_param << ",";
		out_param << results[0];
		out_param << ",";
		out_param << results[1];
		out_param << ",";
		out_param << results[2];
		out_param << "\n";
	}
	out_param.close();
}