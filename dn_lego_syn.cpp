#include <torch/script.h> // One-stop header.
#include "dn_lego_syn.h"

int main(int argc, const char* argv[]) {
	if (argc != 4) {
		std::cerr << "usage: app <path-to-metadata> <path-to-model-config-JSON-file>\n";
		return -1;
	}
	{
		cv::Mat croppedImage;
		bool bvalid = chipping("output/D4/cgv_r/0001/metadata/0001_0017.json", argv[3], croppedImage, true, "0001_0.9922_0017_15JAN21161308.png");
		if (bvalid) {
			cv::Mat dnn_img;
			segment_chip(croppedImage, dnn_img, "output/D4/cgv_r/0001/metadata/0001_0017.json", argv[3], true, "0001_0.9922_0017_15JAN21161308.png");
			feedDnn(dnn_img, "output/D4/cgv_r/0001/metadata/0001_0017.json", argv[3], true, "0001_0.9922_0017_15JAN21161308.png");
		}
	}
	return 0;
	std::string path(argv[1]);
	std::vector<int> clustersID = clustersList(argv[2], 4, "cgv_r");
	std::vector<std::string> clusters;
	clusters.resize(clustersID.size());
	for (int i = 0; i < clustersID.size(); i++) {
		if (clustersID[i] < 10)
			clusters[i] = "000" + std::to_string(clustersID[i]);
		else if(clustersID[i] < 100)
			clusters[i] = "00" + std::to_string(clustersID[i]);
	}
	for (int i = 0; i < clusters.size(); i++) {
		std::vector<std::string> metaFiles = get_all_files_names_within_folder(path + "/" + clusters[i] + "/metadata");
		for (int j = 0; j < metaFiles.size(); j++) {
			std::string metajason = path + "/" + clusters[i] + "/metadata/" + metaFiles[j];
			std::cout << metajason << std::endl;
		}
	}
	return 0;
}

std::vector<int> clustersList(std::string metajson, int regionId, std::string modelType) {
	// read image json file
	FILE* fp = fopen(metajson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document doc;
	doc.ParseStream(is);
	rapidjson::Value& regions = doc["regions"];
	std::string region = "D" + std::to_string(regionId);
	rapidjson::Value& regionContent = regions[region.c_str()];
	rapidjson::Value& regionModels = regionContent["models"];
	rapidjson::Value& regionModel = regionModels[modelType.c_str()];
	rapidjson::Value& clustersSelected = regionModel["selected"];
	std::vector<int> clustersList;
	clustersList.resize(clustersSelected.Size());
	for (int i = 0; i < clustersSelected.Size(); i++) {
		clustersList[i] = clustersSelected[i]["id"].GetInt();
		//std::cout << clustersList[i] << std::endl;
	}
	return clustersList;
}

bool chipping(std::string metajson, std::string modeljson, cv::Mat& croppedImage, bool bDebug, std::string img_filename) {
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
	if (facChip_size[0] < 30.0 && facChip_size[0] > 15.0 && facChip_size[1] < 30.0 && facChip_size[1] > 15.0 && score > 0.95) {
		type = 1;
		bvalid = true;
	}
	else if (facChip_size[0] > 30.0 && facChip_size[1] < 30.0 && facChip_size[1] > 12.9 && score > 0.95) {
		type = 2;
		bvalid = true;
	}
	else if (facChip_size[0] < 30.0 && facChip_size[0] > 12.9 && facChip_size[1] > 30.0 && score > 0.95) {
		type = 3;
		bvalid = true;
	}
	else if (facChip_size[0] > 30.0 && facChip_size[1] > 30.0 && score > 0.85) {
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
				for(int c = 0; c < 3; c++)
					avg_color.val[c] += src.at<cv::Vec3b>(i, j)[c];
			}
		}
		rapidjson::Value avg_color_json(rapidjson::kArrayType);
		for (int i = 0; i < 3; i++) {
			avg_color.val[i] = avg_color.val[i] / (src.size().height * src.size().width);
			avg_color_json.PushBack(avg_color.val[i], alloc);
		}
		doc.AddMember("bg_color", avg_color_json, alloc);

		char writeBuffer[10240];
		rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
		rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
		doc.Accept(writer);
		fclose(fp);
		return false;
	}

	// read model config json file
	fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	memset(readBuffer, 0, sizeof(readBuffer));
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);

	std::vector<double> tmp_array = read1DArray(docModel, "targetChipSize");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the targetChipSize member in the JSON file" << std::endl;
		return false;
	}
	double target_width = tmp_array[0];
	double target_height = tmp_array[1];

	cv::Mat src_chip;
	src_chip = cv::imread(img_name);
	// crop a chip
	croppedImage = crop_chip(src_chip, type, bground, facChip_size, target_width, target_height);
	// adjust the chip
	if (type == 2 || type == 3 || type == 4)
		croppedImage = adjust_chip(src_chip, croppedImage, type, bground, facChip_size, target_width, target_height);
	if (bDebug) {
		// get facade folder path
		std::string facades_folder = readStringValue(docModel, "facadesFolder");
		// get chips folder path
		std::string chips_folder = readStringValue(docModel, "chipsFolder");
		cv::imwrite(facades_folder + "/" + img_filename, src_chip);
		cv::imwrite(chips_folder + "/" + img_filename, croppedImage);
	}
	return true;
}

bool segment_chip(cv::Mat croppedImage, cv::Mat& dnn_img, std::string metajson, std::string modeljson, bool bDebug, std::string img_filename) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	// default size for NN
	int height = 224; // DNN image height
	int width = 224; // DNN image width
	std::vector<double> tmp_array = read1DArray(docModel, "defaultSize");
	width = tmp_array[0];
	height = tmp_array[1];
	// load image
	cv::Mat src, dst_ehist, dst_classify;
	src = croppedImage;
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	dst_ehist = bgr[2];
	int threshold = 0;
	// kkmeans classification
	dst_classify = facade_clustering_kkmeans(dst_ehist, cluster_number);
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(dst_classify, scale_img, cv::Size(width, height));
	// correct the color
	for (int i = 0; i < scale_img.size().height; i++) {
		for (int j = 0; j < scale_img.size().width; j++) {
			//noise
			if ((int)scale_img.at<uchar>(i, j) < 128) {
				scale_img.at<uchar>(i, j) = (uchar)0;
			}
			else
				scale_img.at<uchar>(i, j) = (uchar)255;
		}
	}

	// dilate to remove noises
	int dilation_type = cv::MORPH_RECT;
	cv::Mat dilation_dst;
	int kernel_size = 5;
	cv::Mat element = cv::getStructuringElement(dilation_type, cv::Size(kernel_size, kernel_size), cv::Point(kernel_size / 2, kernel_size / 2));
	/// Apply the dilation operation
	cv::dilate(scale_img, dilation_dst, element);

	// alignment
	float angle = findSkewAngle(dilation_dst);
	if (bDebug)
		std::cout << "angle is " << angle << std::endl;
	// rotate the image
	cv::Mat aligned_img;
	cv::Point2f offset(dilation_dst.cols / 2, dilation_dst.rows / 2);
	cv::Mat rot_mat = cv::getRotationMatrix2D(offset, angle, 1.0);
	cv::warpAffine(dilation_dst, aligned_img, rot_mat, dilation_dst.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
	// clean up the image
	aligned_img = cleanAlignedImage(aligned_img, 0.10);
	// add padding
	int padding_size = 5;
	int borderType = cv::BORDER_CONSTANT;
	cv::Scalar value(255, 255, 255);
	cv::Mat aligned_img_padding;
	cv::copyMakeBorder(aligned_img, aligned_img_padding, padding_size, padding_size, padding_size, padding_size, borderType, value);

	// find contours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(aligned_img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<cv::Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
	}
	dnn_img = cv::Mat(aligned_img_padding.size(), CV_8UC3, bg_color);
	for (int i = 1; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		// check the validity of the rect
		float area_contour = cv::contourArea(contours[i]);
		float area_rect = boundRect[i].width * boundRect[i].height;
		if (area_rect < 10) continue;
		if (area_rect < 100) {
			cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
			continue;
		}
		float ratio = area_contour / area_rect;
		int tmp = 0;
		while (ratio < 0.5 && (boundRect[i].height - tmp) > 0) {
			area_rect = boundRect[i].width * (boundRect[i].height - tmp);
			ratio = area_contour / area_rect;
			tmp += 2;
		}
		cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y + tmp / 2), cv::Point(boundRect[i].br().x, boundRect[i].br().y - tmp / 2), window_color, -1);
	}
	// remove padding
	dnn_img = dnn_img(cv::Rect(padding_size, padding_size, width, height));
	if (bDebug) {
		// get segs folder path
		std::string segs_folder = readStringValue(docModel, "segsFolder");
		// get segs folder path
		std::string resizes_folder = readStringValue(docModel, "resizesFolder");
		// get dilates folder path
		std::string dilates_folder = readStringValue(docModel, "dilatesFolder");
		// get aligns folder path
		std::string aligns_folder = readStringValue(docModel, "alignsFolder");
		// get dnn folder path
		std::string dnnsIn_folder = readStringValue(docModel, "dnnsInFolder");
		//
		cv::imwrite(segs_folder + "/" + img_filename, dst_classify);
		cv::imwrite(resizes_folder + "/" + img_filename, scale_img);
		cv::imwrite(dilates_folder + "/" + img_filename, dilation_dst);
		cv::imwrite(aligns_folder + "/" + img_filename, aligned_img);
		cv::imwrite(dnnsIn_folder + "/" + img_filename, dnn_img);
	}
	// write back to json file
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
	writeBackAvgColors(metajson, true, bg_avg_color, win_avg_color);

	return true;
}

std::vector<double> feedDnn(cv::Mat dnn_img, std::string metajson, std::string modeljson, bool bDebug, std::string img_filename) {
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	std::string classifier_name;

	rapidjson::Value& grammars = docModel["grammars"];
	// classifier
	rapidjson::Value& grammar_classifier = grammars["classifier"];
	// path of DN model
	classifier_name = readStringValue(grammar_classifier, "model");
	int num_classes = readNumber(grammar_classifier, "number_paras", 6);
	if (bDebug) {
		std::cout << "classifier_name is " << classifier_name << std::endl;
	}

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
	int best_class = 1;

	if(false)
	{
		// Deserialize the ScriptModule from a file using torch::jit::load().
		std::shared_ptr<torch::jit::script::Module> classifier_module = torch::jit::load(classifier_name);
		classifier_module->to(at::kCUDA);
		assert(classifier_module != nullptr);
		torch::Tensor out_tensor = classifier_module->forward(inputs).toTensor();
		std::cout << out_tensor.slice(1, 0, num_classes) << std::endl;
		double best_score = 0;
		for (int i = 0; i < num_classes; i++) {
			double tmp = out_tensor.slice(1, i, i + 1).item<float>();
			if (tmp > best_score) {
				best_score = tmp;
				best_class = i;
			}
		}
		best_class = best_class + 1;
		std::cout << "DNN class is " << best_class << std::endl;
	}

	// choose conresponding estimation DNN
	std::string model_name;
	std::string grammar_name = "grammar" + std::to_string(best_class);
	rapidjson::Value& grammar = grammars[grammar_name.c_str()];
	// path of DN model
	model_name = readStringValue(grammar, "model");
	if (bDebug)
		std::cout << "model_name is " << model_name << std::endl;
	// number of paras
	int num_paras = readNumber(grammar, "number_paras", 5);

	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_name);
	module->to(at::kCUDA);
	assert(module != nullptr);
	torch::Tensor out_tensor_grammar = module->forward(inputs).toTensor();
	std::cout << out_tensor_grammar.slice(1, 0, num_paras) << std::endl;
	std::vector<double> paras;
	for (int i = 0; i < num_paras; i++) {
		paras.push_back(out_tensor_grammar.slice(1, i, i + 1).item<float>());
	}
	fclose(fp);
	std::vector<double> predictions;
	return predictions;
	if (best_class == 1) {
		predictions = grammar1(modeljson, paras, bDebug);
	}
	else if (best_class == 2) {
		predictions = grammar2(modeljson, paras, bDebug);
	}
	else if (best_class == 3) {
		predictions = grammar3(modeljson, paras, bDebug);
	}
	else if (best_class == 4) {
		predictions = grammar4(modeljson, paras, bDebug);
	}
	else if (best_class == 5) {
		predictions = grammar5(modeljson, paras, bDebug);
	}
	else if (best_class == 6) {
		predictions = grammar6(modeljson, paras, bDebug);
	}
	else {
		//do nothing
		predictions = grammar1(modeljson, paras, bDebug);
	}
	// write back to json file
	fp = fopen(metajson.c_str(), "rb"); // non-Windows use "r"
	memset(readBuffer, 0, sizeof(readBuffer));
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document doc;
	doc.ParseStream(is);
	fclose(fp);
	fp = fopen(metajson.c_str(), "w"); // non-Windows use "w"
	rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
	if (predictions.size() == 5) {
		int img_rows = predictions[0];
		int img_cols = predictions[1];
		int img_groups = predictions[2];
		double relative_width = predictions[3];
		double relative_height = predictions[4];

		rapidjson::Value paras_json(rapidjson::kObjectType);
		paras_json.AddMember("rows", img_rows, alloc);
		paras_json.AddMember("cols", img_cols, alloc);
		paras_json.AddMember("grouping", img_groups, alloc);
		paras_json.AddMember("relativeWidth", relative_width, alloc);
		paras_json.AddMember("relativeHeight", relative_height, alloc);
		doc.AddMember("paras", paras_json, alloc);
	}
	if (predictions.size() == 8) {
		int img_rows = predictions[0];
		int img_cols = predictions[1];
		int img_groups = predictions[2];
		int img_doors = predictions[3];
		double relative_width = predictions[4];
		double relative_height = predictions[5];
		double relative_door_width = predictions[6];
		double relative_door_height = predictions[7];

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
	}
	char writeBuffer[10240];
	rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
	rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
	doc.Accept(writer);
	fclose(fp);
	return predictions;
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
	int img_rows = round(paras[0] * (imageRows.second - imageRows.first) + imageRows.first);
	int img_cols = round(paras[1] * (imageCols.second - imageCols.first) + imageCols.first);
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
	int img_rows = round(paras[0] * (imageRows.second - imageRows.first) + imageRows.first);
	int img_cols = round(paras[1] * (imageCols.second - imageCols.first) + imageCols.first);
	int img_groups = 1;
	int img_doors = round(paras[2] * (imageDoors.second - imageDoors.first) + imageDoors.first);
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
	int img_cols = round(paras[0] * (imageCols.second - imageCols.first) + imageCols.first);
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
	int img_cols = round(paras[0] * (imageCols.second - imageCols.first) + imageCols.first);
	int img_groups = 1;
	int img_doors = round(paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first);
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
	int img_rows = round(paras[0] * (imageRows.second - imageRows.first) + imageRows.first);
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
	int img_rows = round(paras[0] * (imageRows.second - imageRows.first) + imageRows.first);
	int img_cols = 1;
	int img_groups = 1;
	int img_doors = round(paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first);
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

void synthesis(std::vector<double> predictions, cv::Size src_size, std::string dnnsOut_folder, cv::Scalar win_avg_color, cv::Scalar bg_avg_color, std::string img_filename, bool bDebug){
	cv::Mat syn_img;
	if (predictions.size() == 5) {
		int img_rows = predictions[0];
		int img_cols = predictions[1];
		int img_groups = predictions[2];
		double relative_width = predictions[3];
		double relative_height = predictions[4];
		syn_img = generateFacadeSynImage(224, 224, img_rows, img_cols, img_groups, relative_width, relative_height);
	}
	if(predictions.size() == 8) {
		int img_rows = predictions[0];
		int img_cols = predictions[1];
		int img_groups = predictions[2];
		int img_doors = predictions[3];
		double relative_width = predictions[4];
		double relative_height = predictions[5];
		double relative_door_width = predictions[6];
		double relative_door_height = predictions[7];
		syn_img = generateFacadeSynImage(224, 224, img_rows, img_cols, img_groups, img_doors, relative_width, relative_height, relative_door_width, relative_door_height);
	}
	// recover to the original image
	cv::resize(syn_img, syn_img, src_size);
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
	if (bDebug) {
		cv::imwrite(dnnsOut_folder + "/" + img_filename, syn_img);
	}
}

cv::Mat crop_chip(cv::Mat src_chip, int type, bool bground, std::vector<double> facChip_size, double target_width, double target_height) {
	cv::Mat croppedImage;
	if (type == 1) {
		croppedImage = src_chip.clone();
	}
	else if (type == 2) {
		double target_ratio_width = target_width / facChip_size[0];
		double padding_width_ratio = (1 - target_ratio_width) * 0.5;
		// crop 30 * 30
		croppedImage = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, 0, src_chip.size().width * target_ratio_width, src_chip.size().height));
	}
	else if (type == 3) {
		double target_ratio_height = target_height / facChip_size[1];
		double padding_height_ratio = 0;
		if (!bground) {
			padding_height_ratio = (1 - target_ratio_height) * 0.5;
		}
		else {
			padding_height_ratio = (1 - target_ratio_height);
		}
		croppedImage = src_chip(cv::Rect(0, src_chip.size().height * padding_height_ratio, src_chip.size().width, src_chip.size().height * target_ratio_height));
	}
	else if (type == 4) {
		// crop 30 * 30
		double target_ratio_width = target_width / facChip_size[0];
		double target_ratio_height = target_height / facChip_size[1];
		double padding_width_ratio = (1 - target_ratio_width) * 0.5;
		double padding_height_ratio = 0;
		if (!bground) {
			padding_height_ratio = (1 - target_ratio_height) * 0.5;
		}
		else {
			padding_height_ratio = (1 - target_ratio_height);
		}
		croppedImage = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
	}
	else {
		// do nothing
	}
	return croppedImage;
}

cv::Mat adjust_chip(cv::Mat src_chip, cv::Mat chip, int type, bool bground, std::vector<double> facChip_size, double target_width, double target_height) {
	// load image
	cv::Mat dst_ehist, dst_classify;
	cv::Mat hsv;
	cvtColor(chip, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	dst_ehist = bgr[2];
	int threshold = 0;
	// threshold classification
	dst_classify = facade_clustering_kkmeans(dst_ehist, cluster_number);
	// find the boundary
	int scan_line = 0;
	// bottom 
	int pos_bot = 0;
	for (int i = dst_classify.size().height - 1; i >=0; i--) {
		scan_line = 0;
		for (int j = 0; j < dst_classify.size().width; j++) {
			//noise
			if ((int)dst_classify.at<uchar>(i, j) == 0) {
				scan_line++;
			}
		}
		if (scan_line * 1.0 / dst_classify.size().width < 0.90) { // threshold is 0.9
			pos_bot = i;
			break;
		}

	}
	pos_bot = dst_classify.size().height - 1 - pos_bot;
	
	// left
	int pos_left = 0;
	for (int i = 0; i < dst_classify.size().width; i++) {
		scan_line = 0;
		for (int j = 0; j < dst_classify.size().height; j++) {
			//noise
			if ((int)dst_classify.at<uchar>(j, i) == 0) {
				scan_line++;
			}
		}
		if (scan_line * 1.0 / dst_classify.size().width > 0.10) { // threshold is 0.1
			pos_left = i;
			break;
		}

	}
	// right
	int pos_right = 0;
	for (int i = dst_classify.size().width - 1; i >= 0;  i--) {
		scan_line = 0;
		for (int j = 0; j < dst_classify.size().height; j++) {
			//noise
			if ((int)dst_classify.at<uchar>(j, i) == 0) {
				scan_line++;
			}
		}
		if (scan_line * 1.0 / dst_classify.size().width > 0.10) { // threshold is 0.1
			pos_right = i;
			break;
		}

	}
	pos_right = dst_classify.size().width - 1 - pos_right;
	
	cv::Mat croppedImage;
	if (type == 2) {
		if (pos_right == 0 && pos_left == 0)
			return chip;
		double target_ratio_width = target_width / facChip_size[0];
		double padding_width_ratio = (1 - target_ratio_width) * 0.5;
		if (pos_right > pos_left) {
			if (src_chip.size().width * padding_width_ratio - pos_right > 0)
				croppedImage = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio - pos_right, 0, src_chip.size().width * target_ratio_width, src_chip.size().height));
			else
				croppedImage = src_chip(cv::Rect(0, 0, src_chip.size().width * target_ratio_width, src_chip.size().height));
		}
		else {
			if (src_chip.size().width * padding_width_ratio + pos_left + src_chip.size().width * target_ratio_width  < src_chip.size().width)
				croppedImage = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio + pos_left, 0, src_chip.size().width * target_ratio_width, src_chip.size().height));
			else
				croppedImage = src_chip(cv::Rect(src_chip.size().width - src_chip.size().width * target_ratio_width, 0, src_chip.size().width * target_ratio_width, src_chip.size().height));

		}
		return croppedImage;
	}

	if (!bground) {
		if (pos_right == 0 && pos_left == 0)
			return chip;
		if (type == 3) {
			double target_ratio_height = target_height / facChip_size[1];
			double padding_height_ratio = 0;
			padding_height_ratio = (1 - target_ratio_height) * 0.5;
			if (pos_right > pos_left) {
				croppedImage = src_chip(cv::Rect(0, src_chip.size().height * padding_height_ratio, src_chip.size().width - pos_right, src_chip.size().height * target_ratio_height));
			}
			else {
				croppedImage = src_chip(cv::Rect(pos_left, src_chip.size().height * padding_height_ratio, src_chip.size().width - pos_left, src_chip.size().height * target_ratio_height));
			}
		}

		if (type == 4) {
			// crop 30 * 30
			double target_ratio_width = target_width / facChip_size[0];
			double target_ratio_height = target_height / facChip_size[1];
			double padding_width_ratio = (1 - target_ratio_width) * 0.5;
			double padding_height_ratio = (1 - target_ratio_height) * 0.5;
			if (pos_right > pos_left) {
				croppedImage = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width - pos_right, src_chip.size().height * target_ratio_height));
			}
			else {
				croppedImage = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio + pos_left, src_chip.size().height * padding_height_ratio, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			}
		}
		return croppedImage;
	}
	else {
		if (pos_bot == 0)
			return chip;
		if (type == 3) {
			double target_ratio_height = target_height / facChip_size[1];
			double padding_height_ratio = 0;
			padding_height_ratio = (1 - target_ratio_height);
			if (src_chip.size().height * padding_height_ratio - pos_bot >= 0)
				croppedImage = src_chip(cv::Rect(0, src_chip.size().height * padding_height_ratio - pos_bot, src_chip.size().width, src_chip.size().height * target_ratio_height));
			else
				croppedImage = src_chip(cv::Rect(0, 0, src_chip.size().width, src_chip.size().height - pos_bot));
		}
		if (type == 4) {
			// crop 30 * 30
			double target_ratio_width = target_width / facChip_size[0];
			double target_ratio_height = target_height / facChip_size[1];
			double padding_width_ratio = (1 - target_ratio_width) * 0.5;
			double padding_height_ratio = 0;
			padding_height_ratio = (1 - target_ratio_height);
			if (src_chip.size().height * padding_height_ratio - pos_bot >= 0)
				croppedImage = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, src_chip.size().height * padding_height_ratio - pos_bot, src_chip.size().width * target_ratio_width, src_chip.size().height * target_ratio_height));
			else
				croppedImage = src_chip(cv::Rect(src_chip.size().width * padding_width_ratio, 0, src_chip.size().width * target_ratio_width, src_chip.size().height - pos_bot));
		}
		return croppedImage;
	}
}

cv::Mat facade_clustering_kkmeans(cv::Mat src_img,  int clusters) {
	// Here we declare that our samples will be 2 dimensional column vectors.  
	// (Note that if you don't know the dimensionality of your vectors at compile time
	// you can change the 2 to a 0 and then set the size at runtime)
	typedef matrix<double, 0, 1> sample_type;
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

	std::vector<cv::Scalar> clusters_colors;
	std::vector<int> clusters_points;
	clusters_colors.resize(clusters);
	clusters_points.resize(clusters);
	for (int i = 0; i < clusters; i++) {
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
	cv::Mat out_img;
	cv::resize(src_img, out_img, cv::Size(src_img.size().width, src_img.size().height));
	count = 0;
	if (src_img.channels() == 1) {
		for (int i = 0; i < out_img.size().height; i++) {
			for (int j = 0; j < out_img.size().width; j++) {
				if (test(samples[count]) == darkest_cluster) {
					out_img.at<uchar>(i, j) = (uchar)0;
				}
				else {
					out_img.at<uchar>(i, j) = (uchar)255;

				}
				count++;
			}
		}
	}
	else {
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
	}
	return out_img;
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

float findSkewAngle(cv::Mat src_img) {
	// add padding
	int padding_size = 5;
	int borderType = cv::BORDER_CONSTANT;
	cv::Scalar value(255, 255, 255);
	cv::Mat src_padding;
	cv::copyMakeBorder(src_img, src_padding, padding_size, padding_size, padding_size, padding_size, borderType, value);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(src_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	/// Find the rotated rectangles and ellipses for each contour
	std::vector<cv::RotatedRect> minRect(contours.size());
	std::vector<cv::RotatedRect> minEllipse(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		minRect[i] = cv::minAreaRect(cv::Mat(contours[i]));
	}
	std::vector<float> angles;
	for (int i = 0; i < contours.size(); i++) {
		if (hierarchy[i][3] != 0) continue;
		float tmp = minRect[i].angle;
		if (tmp < -45.)
			tmp += 90.;
		angles.push_back(tmp);
	}
	std::sort(angles.begin(), angles.end());
	float first_q = angles[angles.size() / 4];
	float median_q = angles[angles.size() / 2];
	float third_q = angles[3 * angles.size() / 4];
	/*std::cout << "1 qt Angle is " << first_q << std::endl;
	std::cout << "Median Angle is " << median_q << std::endl;
	std::cout << "3 qt Angle is " << third_q << std::endl;*/
	float threshold = 5;
	if (abs(first_q - median_q) < threshold && abs(third_q - median_q) < threshold)
		return median_q;
	else
		return 0;
}

cv::Mat cleanAlignedImage(cv::Mat src, float threshold) {
	// horz
	int count = 0;
	cv::Mat result = src.clone();
	for (int i = 0; i < src.size().height; i++) {
		count = 0;
		for (int j = 0; j < src.size().width; j++) {
			//noise
			if ((int)src.at<uchar>(i, j) == 0) {
				count++;
			}
		}
		if (count * 1.0 / src.size().width < threshold) {
			for (int j = 0; j < src.size().width; j++) {
				result.at<uchar>(i, j) = (uchar)255;
			}
		}

	}
	// vertical
	for (int i = 0; i < src.size().width; i++) {
		count = 0;
		for (int j = 0; j < src.size().height; j++) {
			//noise
			if ((int)src.at<uchar>(j, i) == 0) {
				count++;
			}
		}
		if (count * 1.0 / src.size().height < threshold) {
			for (int j = 0; j < src.size().height; j++) {
				result.at<uchar>(j, i) = (uchar)255;
			}
		}

	}
	return result;
}

void writeBackAvgColors(std::string metajson, bool bvalid, cv::Scalar bg_avg_color, cv::Scalar win_avg_color) {
	FILE* fp = fopen(metajson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document doc;
	doc.ParseStream(is);
	fclose(fp);

	// write back to json file
	fp = fopen(metajson.c_str(), "w"); // non-Windows use "w"
	rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
	if (doc.HasMember("valid"))
		doc["valid"].SetBool(bvalid);
	else
		doc.AddMember("valid", bvalid, alloc);

	rapidjson::Value bg_color_json(rapidjson::kArrayType);
	bg_color_json.PushBack(bg_avg_color.val[0], alloc);
	bg_color_json.PushBack(bg_avg_color.val[1], alloc);
	bg_color_json.PushBack(bg_avg_color.val[2], alloc);
	if (doc.HasMember("bg_color")) {
		doc["bg_color"].Clear();
		doc["bg_color"].PushBack(bg_avg_color.val[0], alloc);
		doc["bg_color"].PushBack(bg_avg_color.val[1], alloc);
		doc["bg_color"].PushBack(bg_avg_color.val[2], alloc);
	}
	else
		doc.AddMember("bg_color", bg_color_json, alloc);

	rapidjson::Value win_color_json(rapidjson::kArrayType);
	win_color_json.PushBack(win_avg_color.val[0], alloc);
	win_color_json.PushBack(win_avg_color.val[1], alloc);
	win_color_json.PushBack(win_avg_color.val[2], alloc);
	if (doc.HasMember("window_color")) {
		doc["window_color"].Clear();
		doc["window_color"].PushBack(win_avg_color.val[0], alloc);
		doc["window_color"].PushBack(win_avg_color.val[1], alloc);
		doc["window_color"].PushBack(win_avg_color.val[2], alloc);
	}
	else
		doc.AddMember("window_color", win_color_json, alloc);

	char writeBuffer[10240];
	rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
	rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
	doc.Accept(writer);
	fclose(fp);
}

cv::Scalar readColor(std::string metajson, std::string color_name) {
	FILE* fp = fopen(metajson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document doc;
	doc.ParseStream(is); 
	std::vector<double>tmp_array = read1DArray(doc, color_name.c_str());
	return cv::Scalar(tmp_array[0], tmp_array[1], tmp_array[2]);
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
