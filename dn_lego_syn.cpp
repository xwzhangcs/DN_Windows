#include "dn_lego_syn.h"
#include <stack>
#include "Utils.h"
#include "dn_lego_eval.h"
#include "optGrammarParas.h"

int main(int argc, const char* argv[]) {
	if (argc != 4) {
		std::cerr << "usage: app <path-to-metadata> <path-to-model-config-JSON-file>\n";
		return -1;
	}
	std::string path(argv[1]);
	std::vector<std::string> clusters = get_all_files_names_within_folder(argv[1]);
	ModelInfo mi;
	readModeljson(argv[3], mi);
	initial_models(mi);
	for (int i = 0; i < clusters.size(); i++) {
		std::vector<std::string> metaFiles = get_all_files_names_within_folder(path + "/" + clusters[i] + "/metadata");
		for (int j = 0; j < metaFiles.size(); j++) {
			std::string metajson = path + "/" + clusters[i] + "/metadata/" + metaFiles[j];
			std::string img_filename = clusters[i] + "_" + metaFiles[j].substr(0, metaFiles[j].find(".json")) + ".png";
			std::cout << metajson << ", " << img_filename << std::endl;
			// read metajson
			FacadeInfo fi;
			readMetajson(metajson, fi);
			cv::Mat croppedImage;
			bool bvalid = chipping(fi, mi, croppedImage, true, true, img_filename);	
		}
	}
	return 0;
}

void readMetajson(std::string metajson, FacadeInfo& fi) {
	// read image json file
	FILE* fp = fopen(metajson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document doc;
	doc.ParseStream(is);
	// size of chip
	fi.facadeSize = util::read1DArray(doc, "size");
	// roof 
	fi.roof = util::readBoolValue(doc, "roof", false);
	// ground
	fi.ground = util::readBoolValue(doc, "ground", false);
	// image file
	fi.imgName = util::readStringValue(doc, "imagename");
	// score
	fi.score = util::readNumber(doc, "score", 0.2);
	fclose(fp);
}

void writeMetajson(std::string metajson, FacadeInfo& fi) {
	FILE* fp = fopen(metajson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document doc;
	doc.ParseStream(is);
	fclose(fp);
	// write back to json file
	fp = fopen(metajson.c_str(), "wb"); // non-Windows use "w"
	rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
	if (doc.HasMember("valid"))
		doc["valid"].SetBool(fi.valid);
	else
		doc.AddMember("valid", fi.valid, alloc);
	// add real chip size
	if (doc.HasMember("chip_size")) {
		doc["chip_size"].Clear();
		doc["chip_size"].PushBack(fi.chip_size[0], alloc);
		doc["chip_size"].PushBack(fi.chip_size[1], alloc);

	}
	else {
		rapidjson::Value chip_json(rapidjson::kArrayType);
		chip_json.PushBack(fi.chip_size[0], alloc);
		chip_json.PushBack(fi.chip_size[1], alloc);
		doc.AddMember("chip_size", chip_json, alloc);
	}
	// writeback confidence values
	if (doc.HasMember("confidences")) {
		doc["confidences"].Clear();
		for (int i = 0; i < fi.conf.size(); i++)
			doc["confidences"].PushBack(fi.conf[i], alloc);
	}
	else {
		rapidjson::Value confidence_json(rapidjson::kArrayType);
		for (int i = 0; i < fi.conf.size(); i++)
			confidence_json.PushBack(fi.conf[i], alloc);
		doc.AddMember("confidences", confidence_json, alloc);
	}
	// initialize the grammar attribute and paras attribute
	if (doc.HasMember("grammar")) {
		doc["grammar"].SetInt(fi.grammar);
	}
	else
		doc.AddMember("grammar", fi.grammar, alloc);
	// grammar
	if (fi.grammar % 2 != 0) {
		int img_rows = fi.rows;
		int img_cols = fi.cols;
		int img_groups = fi.grouping;
		double relative_width = fi.relativeWidth;
		double relative_height = fi.relativeHeight;
		if (doc.HasMember("paras")) {
			doc["paras"]["rows"].SetInt(img_rows);
			doc["paras"]["cols"].SetInt(img_cols);
			doc["paras"]["grouping"].SetInt(img_groups);
			doc["paras"]["relativeWidth"].SetDouble(relative_width);
			doc["paras"]["relativeHeight"].SetDouble(relative_height);
		}
		else {
			rapidjson::Value paras_json(rapidjson::kObjectType);
			paras_json.AddMember("rows", img_rows, alloc);
			paras_json.AddMember("cols", img_cols, alloc);
			paras_json.AddMember("grouping", img_groups, alloc);
			paras_json.AddMember("relativeWidth", relative_width, alloc);
			paras_json.AddMember("relativeHeight", relative_height, alloc);
			doc.AddMember("paras", paras_json, alloc);
		}

	}
	else{
		int img_rows = fi.rows;
		int img_cols = fi.cols;
		int img_groups = fi.grouping;
		int img_doors = fi.doors;
		double relative_width = fi.relativeWidth;
		double relative_height = fi.relativeHeight;
		double relative_door_width = fi.relativeDWidth;
		double relative_door_height = fi.relativeDHeight;
		if (doc.HasMember("paras")) {
			doc["paras"]["rows"].SetInt(img_rows);
			doc["paras"]["cols"].SetInt(img_cols);
			doc["paras"]["grouping"].SetInt(img_groups);
			doc["paras"]["doors"].SetInt(img_doors);
			doc["paras"]["relativeWidth"].SetDouble(relative_width);
			doc["paras"]["relativeHeight"].SetDouble(relative_height);
			doc["paras"]["relativeDWidth"].SetDouble(relative_door_width);
			doc["paras"]["relativeDHeight"].SetDouble(relative_door_height);
		}
		else {
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

	}
	char writeBuffer[10240];
	rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
	rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
	doc.Accept(writer);
	fclose(fp);
}

void readModeljson(std::string modeljson, ModelInfo& mi) {
	// read model config json file
	FILE* fp = fopen(modeljson.c_str(), "rb"); // non-Windows use "r"
	char readBuffer[10240];
	memset(readBuffer, 0, sizeof(readBuffer));
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	fclose(fp);
	mi.targetChipSize = util::read1DArray(docModel, "targetChipSize");
	mi.defaultSize = util::read1DArray(docModel, "defaultSize");
	mi.reject_model = util::readStringValue(docModel, "reject_model");
	mi.debug = util::readBoolValue(docModel, "debug", false);
	rapidjson::Value& grammars = docModel["grammars"];
	// classifier
	rapidjson::Value& grammar_classifier = grammars["classifier"];
	// path of DN model
	mi.classifier_path = util::readStringValue(grammar_classifier, "model");
	mi.number_grammars = util::readNumber(grammar_classifier, "number_paras", 6);
	// get facade folder path
	mi.facadesFolder = util::readStringValue(docModel, "facadesFolder");
	mi.chipsFolder = util::readStringValue(docModel, "chipsFolder");
	mi.segsFolder = util::readStringValue(docModel, "segsFolder");
	mi.dnnsInFolder = util::readStringValue(docModel, "dnnsInFolder");
	mi.dnnsOutFolder = util::readStringValue(docModel, "dnnsOutFolder");
	// get grammars
	for (int i = 0; i < mi.number_grammars; i++) {
		std::string grammar_name = "grammar" + std::to_string(i + 1);
		rapidjson::Value& grammar = grammars[grammar_name.c_str()];
		// path of DN model
		mi.grammars[i].grammar_id = i + 1;
		mi.grammars[i].model_path = util::readStringValue(grammar, "model");
		// number of paras
		mi.grammars[i].number_paras = util::readNumber(grammar, "number_paras", 5);
		if (i == 0 || i == 1) {
			// range of Rows
			mi.grammars[i].rangeOfRows = util::read1DArray(grammar, "rangeOfRows");
			// range of Cols
			mi.grammars[i].rangeOfCols = util::read1DArray(grammar, "rangeOfCols");
			// range of relativeW
			mi.grammars[i].relativeWidth = util::read1DArray(grammar, "relativeWidth");
			// range of relativeH
			mi.grammars[i].relativeHeight = util::read1DArray(grammar, "relativeHeight");
			if (i == 1) {
				// range of Doors
				mi.grammars[i].rangeOfDoors = util::read1DArray(grammar, "rangeOfDoors");
				// relativeDWidth
				mi.grammars[i].relativeDWidth = util::read1DArray(grammar, "relativeDWidth");
				// relativeDHeight
				mi.grammars[i].relativeDHeight = util::read1DArray(grammar, "relativeDHeight");
			}
		}
		else if (i == 2 || i == 3) {
			// range of Cols
			mi.grammars[i].rangeOfCols = util::read1DArray(grammar, "rangeOfCols");
			// range of relativeW
			mi.grammars[i].relativeWidth = util::read1DArray(grammar, "relativeWidth");
			if (i == 3) {
				// range of Doors
				mi.grammars[i].rangeOfDoors = util::read1DArray(grammar, "rangeOfDoors");
				// relativeDWidth
				mi.grammars[i].relativeDWidth = util::read1DArray(grammar, "relativeDWidth");
				// relativeDHeight
				mi.grammars[i].relativeDHeight = util::read1DArray(grammar, "relativeDHeight");
			}
		}
		else {
			// range of Rows
			mi.grammars[i].rangeOfRows = util::read1DArray(grammar, "rangeOfRows");
			// range of relativeH
			mi.grammars[i].relativeHeight = util::read1DArray(grammar, "relativeHeight");
			if (i == 5) {
				// range of Doors
				mi.grammars[i].rangeOfDoors = util::read1DArray(grammar, "rangeOfDoors");
				// relativeDWidth
				mi.grammars[i].relativeDWidth = util::read1DArray(grammar, "relativeDWidth");
				// relativeDHeight
				mi.grammars[i].relativeDHeight = util::read1DArray(grammar, "relativeDHeight");
			}

		}
	}
}

void initial_models(ModelInfo& mi) {
	// load reject model
	reject_classifier_module = torch::jit::load(mi.reject_model);
	reject_classifier_module->to(at::kCUDA);
	assert(reject_classifier_module != nullptr);
}

int reject(std::string img_name, std::vector<double> facadeSize, std::vector<double> targetSize, double score, bool bDebug) {
	int type = 0;
	if (facadeSize[0] < targetSize[0]  && facadeSize[0] > 0.5 * targetSize[0] && facadeSize[1] < targetSize[1] && facadeSize[1] > 0.5 * targetSize[1] && score > 0.94) {
		type = 1;
	}
	else if (facadeSize[0] > targetSize[0] && facadeSize[1] < targetSize[1] && facadeSize[1] > 0.5 * targetSize[1] && score > 0.65) {
		type = 2;
	}
	else if (facadeSize[0] < targetSize[0] && facadeSize[0] > 0.5 * targetSize[0] && facadeSize[1] > targetSize[1] && score > 0.65) {
		type = 3;
	}
	else if (facadeSize[0] > targetSize[0] && facadeSize[1] > targetSize[1] && score > 0.68) {
		type = 4;
	}
	else {
		// do nothing
	}
	return type;
}

int reject(std::string img_name, std::string model_path, std::vector<double> facadeSize, std::vector<double> targetSize, std::vector<double> defaultImgSize, bool bDebug) {
	// prepare inputs
	cv::Mat src_img = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat scale_img;
	cv::resize(src_img, scale_img, cv::Size(defaultImgSize[0], defaultImgSize[1]));
	cv::Mat dnn_img_rgb;
	cv::cvtColor(scale_img, dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);
	// reject model classifier
	// Deserialize the ScriptModule from a file using torch::jit::load().
	//std::shared_ptr<torch::jit::script::Module> reject_classifier_module = torch::jit::load(model_path);
	//reject_classifier_module->to(at::kCUDA);
	//assert(reject_classifier_module != nullptr);
	torch::Tensor out_tensor = reject_classifier_module->forward(inputs).toTensor();

	torch::Tensor confidences_tensor = torch::softmax(out_tensor, 1);

	double best_score = 0;
	int best_class = -1;
	for (int i = 0; i < 2; i++) {
		double tmp = confidences_tensor.slice(1, i, i + 1).item<float>();
		if (tmp > best_score) {
			best_score = tmp;
			best_class = i;
		}
	}
	if (bDebug) {
		//std::cout << out_tensor.slice(1, 0, 2) << std::endl;
		std::cout << confidences_tensor.slice(1, 0, 2) << std::endl;
		std::cout << "DNN class is " << best_class << std::endl;
	}
	if (best_class == 1) // bad facades
		return 0;
	else {
		int type = 0;
		if (facadeSize[0] < targetSize[0] && facadeSize[1] < targetSize[1]) {
			type = 1;
		}
		else if (facadeSize[0] > targetSize[0] && facadeSize[1] < targetSize[1]) {
			type = 2;
		}
		else if (facadeSize[0] < targetSize[0] && facadeSize[1] > targetSize[1]) {
			type = 3;
		}
		else if (facadeSize[0] > targetSize[0] && facadeSize[1] > targetSize[1]) {
			type = 4;
		}
		else {
			// do nothing
		}
		return type;
	}
}

bool chipping(FacadeInfo& fi, ModelInfo& mi, cv::Mat& croppedImage, bool bMultipleChips, bool bDebug, std::string img_filename) {
	// size of chip
	std::vector<double> facadeSize = fi.facadeSize;
	// roof 
	bool broof = fi.roof;
	// ground
	bool bground = fi.ground;
	// image file
	std::string img_name = fi.imgName;
	// score
	double score = fi.score;
	// first decide whether it's a valid chip
	std::vector<double> targetSize = mi.targetChipSize;
	if (targetSize.size() != 2) {
		std::cout << "Please check the targetChipSize member in the JSON file" << std::endl;
		return false;
	}
	if (bDebug) {
		std::cout << "facadeSize is " << facadeSize << std::endl;
		std::cout << "broof is " << broof << std::endl;
		std::cout << "bground is " << bground << std::endl;
		std::cout << "img_name is " << img_name << std::endl;
		std::cout << "score is " << score << std::endl;
		std::cout << "targetSize is " << targetSize << std::endl;
	}
	int type = reject(img_name, mi.reject_model, facadeSize, targetSize, mi.defaultSize, mi.debug);
	return 0;
	if (type == 0) {
		fi.valid = false;
		return false;
	}
	cv::Mat src_chip;
	src_chip = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
	std::vector<cv::Mat> cropped_chips = crop_chip(src_chip, type, bground, facadeSize, targetSize, bMultipleChips);
	// choose the best chip
	croppedImage = cropped_chips[0];// use the best chip to pass through those testings
	return true;
}

std::vector<cv::Mat> crop_chip(cv::Mat src_chip, int type, bool bground, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips) {
	std::vector<cv::Mat> cropped_chips;
	if (type == 1) {

	}
	else if (type == 2) {

	}
	else if (type == 3) {

	}
	else if (type == 4) {

	}
	else {
		// do nothing
	}
	return cropped_chips;
}

bool segment_chip(cv::Mat croppedImage, cv::Mat& dnn_img, FacadeInfo& fi, ModelInfo& mi, bool bDebug, std::string img_filename) {
	// default size for NN
	int height = 224; // DNN image height
	int width = 224; // DNN image width
	std::vector<double> tmp_array = mi.defaultSize;
	width = tmp_array[0];
	height = tmp_array[1];
	// segmentation
	cv::Mat dst_seg = croppedImage.clone();
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(dst_seg, scale_img, cv::Size(width, height));
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
	int kernel_size = 3;
	cv::Mat element = cv::getStructuringElement(dilation_type, cv::Size(kernel_size, kernel_size), cv::Point(kernel_size / 2, kernel_size / 2));
	/// Apply the dilation operation
	cv::dilate(scale_img, dilation_dst, element);
	// alignment
	cv::Mat aligned_img = deSkewImg(dilation_dst);
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
	//
	dnn_img = cv::Mat(aligned_img_padding.size(), CV_8UC3, bg_color);
	for (int i = 1; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
	}
	// remove padding
	dnn_img = dnn_img(cv::Rect(padding_size, padding_size, width, height));

	// write back to json file
	cv::Scalar bg_avg_color(0, 0, 0);
	cv::Scalar win_avg_color(0, 0, 0);
	{
		int bg_count = 0;
		int win_count = 0;
		for (int i = 0; i < dst_seg.size().height; i++) {
			for (int j = 0; j < dst_seg.size().width; j++) {
				if ((int)dst_seg.at<uchar>(i, j) == 0) {
					if (croppedImage.channels() == 4) {
						win_avg_color.val[0] += croppedImage.at<cv::Vec4b>(i, j)[0];
						win_avg_color.val[1] += croppedImage.at<cv::Vec4b>(i, j)[1];
						win_avg_color.val[2] += croppedImage.at<cv::Vec4b>(i, j)[2];
					}
					if (croppedImage.channels() == 3) {
						win_avg_color.val[0] += croppedImage.at<cv::Vec3b>(i, j)[0];
						win_avg_color.val[1] += croppedImage.at<cv::Vec3b>(i, j)[1];
						win_avg_color.val[2] += croppedImage.at<cv::Vec3b>(i, j)[2];
					}
					win_count++;
				}
				else {
					if (croppedImage.channels() == 4) {
						bg_avg_color.val[0] += croppedImage.at<cv::Vec4b>(i, j)[0];
						bg_avg_color.val[1] += croppedImage.at<cv::Vec4b>(i, j)[1];
						bg_avg_color.val[2] += croppedImage.at<cv::Vec4b>(i, j)[2];
					}
					if (croppedImage.channels() == 3) {
						bg_avg_color.val[0] += croppedImage.at<cv::Vec3b>(i, j)[0];
						bg_avg_color.val[1] += croppedImage.at<cv::Vec3b>(i, j)[1];
						bg_avg_color.val[2] += croppedImage.at<cv::Vec3b>(i, j)[2];
					}
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
	fi.win_color.resize(3);
	fi.bg_color.resize(3);
	for (int i = 0; i < 3; i++) {
		fi.win_color[i] = win_avg_color.val[i];
		fi.bg_color[i] = bg_avg_color.val[i];
	}

	return true;
}

std::vector<double> feedDnn(cv::Mat dnn_img, FacadeInfo& fi, ModelInfo& mi, bool bDebug, std::string img_filename) {
	// path of DN model
	std::string classifier_name = mi.classifier_path;
	int num_classes = mi.number_grammars;
	cv::Mat dnn_img_rgb;
	cv::cvtColor(dnn_img, dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);

	int best_class = -1;
	std::vector<double> confidence_values;
	confidence_values.resize(num_classes);
	if(true)
	{
		// Deserialize the ScriptModule from a file using torch::jit::load().
		std::shared_ptr<torch::jit::script::Module> classifier_module = torch::jit::load(classifier_name);
		classifier_module->to(at::kCUDA);
		assert(classifier_module != nullptr);
		torch::Tensor out_tensor = classifier_module->forward(inputs).toTensor();
		//std::cout << out_tensor.slice(1, 0, num_classes) << std::endl;

		torch::Tensor confidences_tensor = torch::softmax(out_tensor, 1);
		std::cout << confidences_tensor.slice(1, 0, num_classes) << std::endl;

		double best_score = 0;
		for (int i = 0; i < num_classes; i++) {
			double tmp = confidences_tensor.slice(1, i, i + 1).item<float>();
			confidence_values[i] = tmp;
			if (tmp > best_score) {
				best_score = tmp;
				best_class = i;
			}
		}
		best_class = best_class + 1;
		std::cout << "DNN class is " << best_class << std::endl;
	}
	// adjust the best_class
	if (!fi.ground) {
		if (best_class % 2 == 0)
			best_class = best_class - 1;
	}
	// choose conresponding estimation DNN
	// path of DN model
	std::string model_name = mi.grammars[best_class - 1].model_path;
	// number of paras
	int num_paras = mi.grammars[best_class - 1].number_paras;

	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_name);
	module->to(at::kCUDA);
	assert(module != nullptr);
	torch::Tensor out_tensor_grammar = module->forward(inputs).toTensor();
	std::cout << out_tensor_grammar.slice(1, 0, num_paras) << std::endl;
	std::vector<double> paras;
	for (int i = 0; i < num_paras; i++) {
		paras.push_back(out_tensor_grammar.slice(1, i, i + 1).item<float>());
	}
	for (int i = 0; i < num_paras; i++) {
		if (paras[i] < 0)
			paras[i] = 0;
	}

	std::vector<double> predictions;
	if (best_class == 1) {
		predictions = grammar1(mi, paras, bDebug);
	}
	else if (best_class == 2) {
		predictions = grammar2(mi, paras, bDebug);
	}
	else if (best_class == 3) {
		predictions = grammar3(mi, paras, bDebug);
	}
	else if (best_class == 4) {
		predictions = grammar4(mi, paras, bDebug);
	}
	else if (best_class == 5) {
		predictions = grammar5(mi, paras, bDebug);
	}
	else if (best_class == 6) {
		predictions = grammar6(mi, paras, bDebug);
	}
	else {
		//do nothing
		predictions = grammar1(mi, paras, bDebug);
	}
	// write back to fi
	fi.conf.resize(num_classes);
	for (int i = 0; i < num_classes; i++)
		fi.conf[i] = confidence_values[i];

	fi.grammar = best_class;
	if (predictions.size() == 5) {
		fi.rows = predictions[0];
		fi.cols = predictions[1];
		fi.grouping = predictions[2];
		fi.relativeWidth = predictions[3];
		fi.relativeHeight = predictions[4];
			
	}
	if (predictions.size() == 8) {
		fi.rows = predictions[0];
		fi.cols = predictions[1];
		fi.grouping = predictions[2];
		fi.doors = predictions[3];
		fi.relativeWidth = predictions[4];
		fi.relativeHeight = predictions[5];
		fi.relativeDWidth = predictions[6];
		fi.relativeDHeight = predictions[7];	

	}
	return predictions;
}

void synthesis(std::vector<double> predictions, cv::Size src_size, std::string dnnsOut_folder, cv::Scalar win_avg_color, cv::Scalar bg_avg_color, bool bDebug, std::string img_filename){
	cv::Mat syn_img;
	if (predictions.size() == 5) {
		int img_rows = predictions[0];
		int img_cols = predictions[1];
		int img_groups = predictions[2];
		double relative_width = predictions[3];
		double relative_height = predictions[4];
		syn_img = util::generateFacadeSynImage(224, 224, img_rows, img_cols, img_groups, relative_width, relative_height);
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
		syn_img = util::generateFacadeSynImage(224, 224, img_rows, img_cols, img_groups, img_doors, relative_width, relative_height, relative_door_width, relative_door_height);
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
	cv::imwrite(dnnsOut_folder + "/" + img_filename, syn_img);
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

cv::Mat deSkewImg(cv::Mat src_img) {
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	// add padding
	int padding_size = 5;
	int borderType = cv::BORDER_CONSTANT;
	cv::Scalar value(255, 255, 255);
	cv::Mat img_padding;
	cv::copyMakeBorder(src_img, img_padding, padding_size, padding_size, padding_size, padding_size, borderType, value);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	if (contours.size() <= 3)
	{
		contours.clear();
		hierarchy.clear();
		int clear_border = 3;
		cv::Mat tmp = src_img(cv::Rect(clear_border, clear_border, src_img.size().width - 2 * clear_border, src_img.size().width - 2 * clear_border)).clone();
		cv::Mat tmp_img_padding;
		cv::copyMakeBorder(tmp, tmp_img_padding, padding_size + clear_border, padding_size + clear_border, padding_size + clear_border, padding_size + clear_border, borderType, bg_color);
		cv::findContours(tmp_img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	}
	std::vector<cv::RotatedRect> minRect(contours.size());
	std::vector<cv::Moments> mu(contours.size());
	std::vector<cv::Point2f> mc(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(cv::Mat(contours[i]));
		mu[i] = moments(contours[i], false);
		mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	cv::Mat drawing = img_padding.clone();
	//first step
	for (int i = 0; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		float angle = minRect[i].angle;
		if (angle < 0) {
			// draw RotatedRect
			cv::Point2f vertices2f[4];
			minRect[i].points(vertices2f);
			// Convert them so we can use them in a fillConvexPoly
			std::vector<cv::Point> points;
			for (int i = 0; i < 4; ++i) {
				points.push_back(vertices2f[i]);
			}
			cv::fillConvexPoly(drawing, points, bg_color, 8, 0);
		}
	}
	// second step
	for (int i = 0; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		float angle = minRect[i].angle;
		if (angle < 0) {
			if (angle < -45.)
				angle += 90.;
			cv::Mat tmp(img_padding.size(), CV_8UC3, bg_color);
			drawContours(tmp, contours, i, window_color, -1, 8, hierarchy, 0, cv::Point());
			// rotate the contour
			cv::Mat tmp_gray;
			cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
			cv::Mat aligned_tmp;
			cv::Mat rot_mat = cv::getRotationMatrix2D(mc[i], angle, 1.0);
			cv::warpAffine(tmp_gray, aligned_tmp, rot_mat, tmp_gray.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
			tmp_gray.empty();
			tmp_gray = aligned_tmp.clone();
			// copy to the output img
			for (int m = 0; m < drawing.size().height; m++) {
				for (int n = 0; n < drawing.size().width; n++) {
					if ((int)tmp_gray.at<uchar>(m, n) < 128 && (int)drawing.at<uchar>(m, n) == 255) {
						drawing.at<uchar>(m, n) = (uchar)0;
					}
					else if ((int)tmp_gray.at<uchar>(m, n) < 128 && (int)drawing.at<uchar>(m, n) == 0) {
						drawing.at<uchar>(m, n) = (uchar)255;
					}
					else {

					}
				}
			}
		}
	}
	drawing = drawing(cv::Rect(padding_size, padding_size, 224, 224));
	cv::Mat aligned_img = cleanAlignedImage(drawing, 0.10);
	return aligned_img;
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

std::vector<double> grammar1(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageRows(mi.grammars[0].rangeOfRows[0], mi.grammars[0].rangeOfRows[1]);
	// range of Cols
	std::pair<int, int> imageCols(mi.grammars[0].rangeOfCols[0], mi.grammars[0].rangeOfCols[1]);
	// relativeWidth
	std::pair<double, double> imageRelativeWidth(mi.grammars[0].relativeWidth[0], mi.grammars[0].relativeWidth[1]);
	// relativeHeight
	std::pair<double, double> imageRelativeHeight(mi.grammars[0].relativeHeight[0], mi.grammars[0].relativeHeight[1]);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.7)
		img_rows++;
	int img_cols = paras[1] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[1] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.7)
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

std::vector<double> grammar2(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageRows(mi.grammars[1].rangeOfRows[0], mi.grammars[1].rangeOfRows[1]);
	// range of Cols
	std::pair<int, int> imageCols(mi.grammars[1].rangeOfCols[0], mi.grammars[1].rangeOfCols[1]);
	// range of Doors
	std::pair<int, int> imageDoors(mi.grammars[1].rangeOfDoors[0], mi.grammars[1].rangeOfDoors[1]);
	// relativeWidth
	std::pair<double, double> imageRelativeWidth(mi.grammars[1].relativeWidth[0], mi.grammars[1].relativeWidth[1]);
	// relativeHeight
	std::pair<double, double> imageRelativeHeight(mi.grammars[1].relativeHeight[0], mi.grammars[1].relativeHeight[1]);
	// relativeDWidth
	std::pair<double, double> imageDRelativeWidth(mi.grammars[1].relativeDWidth[0], mi.grammars[1].relativeDWidth[1]);
	// relativeDHeight
	std::pair<double, double> imageDRelativeHeight(mi.grammars[1].relativeDHeight[0], mi.grammars[1].relativeDHeight[1]);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.7)
		img_rows++;
	int img_cols = paras[1] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[1] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.7)
		img_cols++;
	int img_groups = 1;
	int img_doors = paras[2] * (imageDoors.second - imageDoors.first) + imageDoors.first;
	if (paras[2] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.7)
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

std::vector<double> grammar3(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Cols
	std::pair<int, int> imageCols(mi.grammars[2].rangeOfCols[0], mi.grammars[2].rangeOfCols[1]);
	// relativeWidth
	std::pair<double, double> imageRelativeWidth(mi.grammars[2].relativeWidth[0], mi.grammars[2].relativeWidth[1]);
	int img_rows = 1;
	int img_cols = paras[0] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[0] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.7)
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

std::vector<double> grammar4(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageCols(mi.grammars[3].rangeOfCols[0], mi.grammars[3].rangeOfCols[1]);
	// range of Doors
	std::pair<int, int> imageDoors(mi.grammars[3].rangeOfDoors[0], mi.grammars[3].rangeOfDoors[1]);
	// relativeWidth
	std::pair<double, double> imageRelativeWidth(mi.grammars[3].relativeWidth[0], mi.grammars[3].relativeWidth[1]);
	// relativeDWidth
	std::pair<double, double> imageDRelativeWidth(mi.grammars[3].relativeDWidth[0], mi.grammars[3].relativeDWidth[1]);
	// relativeDHeight
	std::pair<double, double> imageDRelativeHeight(mi.grammars[3].relativeDHeight[0], mi.grammars[3].relativeDHeight[1]);
	int img_rows = 1;;
	int img_cols = paras[0] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[0] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.7)
		img_cols++;
	int img_groups = 1;
	int img_doors = paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first;
	if (paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.7)
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

std::vector<double> grammar5(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageRows(mi.grammars[4].rangeOfRows[0], mi.grammars[4].rangeOfRows[1]);
	// relativeHeight
	std::pair<double, double> imageRelativeHeight(mi.grammars[4].relativeHeight[0], mi.grammars[4].relativeHeight[1]);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.7)
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

std::vector<double> grammar6(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageRows(mi.grammars[5].rangeOfRows[0], mi.grammars[5].rangeOfRows[1]);
	// range of Doors
	std::pair<int, int> imageDoors(mi.grammars[5].rangeOfDoors[0], mi.grammars[5].rangeOfDoors[1]);
	// relativeHeight
	std::pair<double, double> imageRelativeHeight(mi.grammars[5].relativeHeight[0], mi.grammars[5].relativeHeight[1]);
	// relativeDWidth
	std::pair<double, double> imageDRelativeWidth(mi.grammars[5].relativeDWidth[0], mi.grammars[5].relativeDWidth[1]);
	// relativeDHeight
	std::pair<double, double> imageDRelativeHeight(mi.grammars[5].relativeDHeight[0], mi.grammars[5].relativeDHeight[1]);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.7)
		img_rows++;
	int img_cols = 1;
	int img_groups = 1;
	int img_doors = paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first;
	if (paras[1] * (imageDoors.second - imageDoors.first) + imageDoors.first - img_doors > 0.7)
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
