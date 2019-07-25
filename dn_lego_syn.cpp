#include "dn_lego_syn.h"
#include <stack>
#include "Utils.h"
#include "dn_lego_eval.h"
#include "optGrammarParas.h"

void test_rejection_model(std::string images_path, ModelInfo& mi);

int main(int argc, const char* argv[]) {
	if (argc != 4) {
		std::cerr << "usage: app <path-to-metadata> <path-to-model-config-JSON-file>\n";
		return -1;
	}
	std::string path(argv[1]);
	std::vector<std::string> clusters = get_all_files_names_within_folder(argv[1]);
	ModelInfo mi;
	readModeljson(argv[3], mi);
	//test_rejection_model("../data/D4/facades", mi);
	//return 0;
	/*cv::Mat src = cv::imread("../data/test/0010_0009.png", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat dst_seg;
	apply_segmentation_model(src, dst_seg, mi, true, "../data/test/0010_0009_fake.png");
	return 0;*/
	for (int i = 0; i < clusters.size(); i++) {
		std::vector<std::string> metaFiles = get_all_files_names_within_folder(path + "/" + clusters[i] + "/metadata");
		for (int j = 0; j < metaFiles.size(); j++) {
			std::string metajson = path + "/" + clusters[i] + "/metadata/" + metaFiles[j];
			std::string img_filename = clusters[i] + "_" + metaFiles[j].substr(0, metaFiles[j].find(".json")) + ".png";
			std::cout << metajson << ", " << img_filename << std::endl;
			// read metajson
			FacadeInfo fi;
			readMetajson(metajson, fi);
			cv::Mat chip_seg;
			bool bvalid = chipping(fi, mi, chip_seg, true, mi.debug, img_filename);
			if (bvalid) {
				cv::Mat dnn_img;
				process_chip(chip_seg, dnn_img, mi, mi.debug, img_filename);
				std::vector<double> predictions = feedDnn(dnn_img, fi, mi, mi.debug, img_filename);
				if (fi.win_color.size() > 0 && fi.bg_color.size() > 0) {
					cv::Scalar win_avg_color(fi.win_color[0], fi.win_color[1], fi.win_color[2]);
					cv::Scalar bg_avg_color(fi.bg_color[0], fi.bg_color[1], fi.bg_color[2]);
					synthesis(predictions, chip_seg.size(), mi.dnnsOutFolder, win_avg_color, bg_avg_color, mi.debug, img_filename);
				}
			}
			//writeMetajson(metajson, fi);
		}
	}
	return 0;
}

void test_rejection_model(std::string images_path, ModelInfo& mi) {
	std::vector<std::string> images = get_all_files_names_within_folder(images_path);
	for (int i = 0; i < images.size(); i++) {
		std::string img_name = images_path + '/' + images[i];
		cv::Mat src_img = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
		if (src_img.channels() == 4) // ensure there're 3 channels
			cv::cvtColor(src_img, src_img, CV_BGRA2BGR);
		// prepare inputs
		cv::Mat scale_img;
		cv::resize(src_img, scale_img, cv::Size(224, 224)); 
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
		torch::Tensor out_tensor = mi.reject_classifier_module->forward(inputs).toTensor();

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
		if (best_score > 0.96) {
			cv::imwrite("../data/D4/A/" + images[i], src_img);
		}
		else {
			cv::imwrite("../data/D4/B/" + images[i], src_img);
		}
		if (true) {
			//std::cout << out_tensor.slice(1, 0, 2) << std::endl;
			std::cout << img_name << ": "<<confidences_tensor.slice(1, 0, 2) << std::endl;
			std::cout << "Reject class is " << best_class << std::endl;
		}
	}
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
	// write back bg_color
	if (fi.bg_color.size() == 3) {
		if (doc.HasMember("bg_color")) {
			doc["bg_color"].Clear();
			doc["bg_color"].PushBack(fi.bg_color[0], alloc);
			doc["bg_color"].PushBack(fi.bg_color[1], alloc);
			doc["bg_color"].PushBack(fi.bg_color[2], alloc);
		}
		else {
			rapidjson::Value color_json(rapidjson::kArrayType);
			color_json.PushBack(fi.bg_color[0], alloc);
			color_json.PushBack(fi.bg_color[1], alloc);
			color_json.PushBack(fi.bg_color[2], alloc);
			rapidjson::Value n("bg_color", doc.GetAllocator());
			doc.AddMember(n, color_json, alloc);
		}
	}
	if (fi.valid) {
		// add real chip size
		if (fi.chip_size.size() == 2) {
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
		}
		// write back window_color
		if (fi.win_color.size() == 3) {
			if (doc.HasMember("window_color")) {
				doc["window_color"].Clear();
				doc["window_color"].PushBack(fi.win_color[0], alloc);
				doc["window_color"].PushBack(fi.win_color[1], alloc);
				doc["window_color"].PushBack(fi.win_color[2], alloc);
			}
			else {
				rapidjson::Value color_json(rapidjson::kArrayType);
				color_json.PushBack(fi.win_color[0], alloc);
				color_json.PushBack(fi.win_color[1], alloc);
				color_json.PushBack(fi.win_color[2], alloc);
				rapidjson::Value n("window_color", doc.GetAllocator());
				doc.AddMember(n, color_json, alloc);
			}
		}
		// writeback confidence values
		if (fi.conf.size() > 0) {
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
		}
		// initialize the grammar attribute and paras attribute
		if (fi.grammar > 0) {
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
			else {
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
	mi.segImageSize = util::read1DArray(docModel, "segImageSize");
	mi.defaultSize = util::read1DArray(docModel, "defaultSize");
	mi.paddingSize = util::read1DArray(docModel, "paddingSize");
	std::string reject_model = util::readStringValue(docModel, "reject_model");
	// load reject model
	mi.reject_classifier_module = torch::jit::load(reject_model);
	mi.reject_classifier_module->to(at::kCUDA);
	assert(mi.reject_classifier_module != nullptr);
	std::string seg_model = util::readStringValue(docModel, "seg_model");
	// load segmentation model
	mi.seg_module = torch::jit::load(seg_model);
	mi.seg_module->to(at::kCUDA);
	assert(mi.seg_module != nullptr);
	mi.debug = util::readBoolValue(docModel, "debug", false);
	rapidjson::Value& grammars = docModel["grammars"];
	// classifier
	rapidjson::Value& grammar_classifier = grammars["classifier"];
	// path of DN model
	std::string classifier_path = util::readStringValue(grammar_classifier, "model");
	// load grammar classifier model
	mi.classifier_module = torch::jit::load(classifier_path);
	mi.classifier_module->to(at::kCUDA);
	assert(mi.classifier_module != nullptr);
	mi.number_grammars = util::readNumber(grammar_classifier, "number_paras", 6);
	// get facade folder path
	mi.facadesFolder = util::readStringValue(docModel, "facadesFolder");
	mi.invalidfacadesFolder = util::readStringValue(docModel, "invalidfacadesFolder");
	mi.chipsFolder = util::readStringValue(docModel, "chipsFolder");
	mi.segsFolder = util::readStringValue(docModel, "segsFolder");
	mi.dnnsInFolder = util::readStringValue(docModel, "dnnsInFolder");
	mi.dnnsOutFolder = util::readStringValue(docModel, "dnnsOutFolder");
	mi.dilatesFolder = util::readStringValue(docModel, "dilatesFolder");
	mi.alignsFolder = util::readStringValue(docModel, "alignsFolder");
	// get grammars
	for (int i = 0; i < mi.number_grammars; i++) {
		std::string grammar_name = "grammar" + std::to_string(i + 1);
		rapidjson::Value& grammar = grammars[grammar_name.c_str()];
		// path of DN model
		mi.grammars[i].grammar_id = i + 1;
		std::string model_path = util::readStringValue(grammar, "model");
		mi.grammars[i].grammar_model = torch::jit::load(model_path);
		mi.grammars[i].grammar_model->to(at::kCUDA);
		assert(mi.grammars[i].grammar_model != nullptr);
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

int reject(cv::Mat src_img, std::vector<double> facadeSize, std::vector<double> targetSize, double score, bool bDebug) {
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

int reject(cv::Mat src_img, FacadeInfo& fi, ModelInfo& mi,  bool bDebug) {
	// size of chip
	std::vector<double> facadeSize = fi.facadeSize;
	// size of nn image size
	std::vector<double> defaultImgSize = mi.defaultSize;
	// size of ideal chip size
	std::vector<double> targetSize = mi.targetChipSize;
	// if facades are too small threshold is 3m
	if (facadeSize[0] < 6 || facadeSize[1] < 6)
		return 0;
	// if the images are too small threshold is 25 by 25
	if (src_img.size().height < 25 || src_img.size().width < 25)
		return 0;
	// prepare inputs
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
	torch::Tensor out_tensor = mi.reject_classifier_module->forward(inputs).toTensor();

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
		std::cout << "Reject class is " << best_class << std::endl;
	}
	if (best_class == 1 || best_score < 0.96) // bad facades
		return 0;
	else {
		fi.good_conf = best_score;
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

bool chipping(FacadeInfo& fi, ModelInfo& mi, cv::Mat& chip_seg, bool bMultipleChips, bool bDebug, std::string img_filename) {
	// size of chip
	std::vector<double> facadeSize = fi.facadeSize;
	// roof 
	bool broof = fi.roof;
	// ground
	bool bground = fi.ground;
	// image file
	std::string img_name = fi.imgName;
	cv::Mat src_facade = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
	if (src_facade.channels() == 4) // ensure there're 3 channels
		cv::cvtColor(src_facade, src_facade, CV_BGRA2BGR);
	// score
	double score = fi.score;
	// first decide whether it's a valid chip
	std::vector<double> targetSize = mi.targetChipSize;
	if (targetSize.size() != 2) {
		std::cout << "Please check the targetChipSize member in the JSON file" << std::endl;
		return false;
	}
	// if it's not a roof
	int type = 0;
	if(!broof)
		type = reject(src_facade, fi, mi, mi.debug);
	if (type == 0) {
		fi.valid = false;
		// compute avg color
		cv::Scalar avg_color(0, 0, 0);
		for (int i = 0; i < src_facade.size().height; i++) {
			for (int j = 0; j < src_facade.size().width; j++) {
				for (int c = 0; c < 3; c++) {
					if (src_facade.channels() == 4)
						avg_color.val[c] += src_facade.at<cv::Vec4b>(i, j)[c];
					if (src_facade.channels() == 3)
						avg_color.val[c] += src_facade.at<cv::Vec3b>(i, j)[c];
				}
			}
		}
		fi.bg_color.resize(3);
		for (int i = 0; i < 3; i++) {
			avg_color.val[i] = avg_color.val[i] / (src_facade.size().height * src_facade.size().width);
			fi.bg_color[i] = avg_color.val[i];
		}
		if (!broof && bDebug) {
			cv::imwrite(mi.invalidfacadesFolder + "/" + img_filename, src_facade);
		}
		return false;
	}
	if (bDebug) {
		std::cout << "facadeSize is " << facadeSize << std::endl;
		std::cout << "broof is " << broof << std::endl;
		std::cout << "bground is " << bground << std::endl;
		std::cout << "score is " << score << std::endl;
		std::cout << "targetSize is " << targetSize << std::endl;
	}
	fi.valid = true;
	// choose the best chip
	std::vector<cv::Mat> cropped_chips = crop_chip_no_ground(src_facade.clone(), type, facadeSize, targetSize, bMultipleChips);
	cv::Mat croppedImage = cropped_chips[choose_best_chip(cropped_chips, mi, bDebug, img_filename)];// use the best chip to pass through those testings
	// segmentation
	apply_segmentation_model(croppedImage, chip_seg, mi, bDebug, img_filename);
	// add real chip size
	int chip_width = croppedImage.size().width;
	int chip_height = croppedImage.size().height;
	int src_width = src_facade.size().width;
	int src_height = src_facade.size().height;
	fi.chip_size.push_back(chip_width * 1.0 / src_width * facadeSize[0]);
	fi.chip_size.push_back(chip_height * 1.0 / src_height * facadeSize[1]);
	// write back to json file
	cv::Scalar bg_avg_color(0, 0, 0);
	cv::Scalar win_avg_color(0, 0, 0);
	{
		int bg_count = 0;
		int win_count = 0;
		for (int i = 0; i < chip_seg.size().height; i++) {
			for (int j = 0; j < chip_seg.size().width; j++) {
				if ((int)chip_seg.at<uchar>(i, j) == 0) {
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

	if (bDebug) {
		/*for (int i = 0; i < cropped_chips.size(); i++) {
			cv::imwrite("../data/confidences/" + std::to_string(i) + '_' + img_filename, cropped_chips[i]);
		}*/
		std::cout << "Facade type is " << type << std::endl;
		cv::imwrite(mi.facadesFolder + "/" + img_filename, src_facade);
		cv::imwrite(mi.chipsFolder + "/" + img_filename, croppedImage);
		cv::imwrite(mi.segsFolder + "/" + img_filename, chip_seg);
	}
	return true;
}

std::vector<cv::Mat> crop_chip_no_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips) {
	std::vector<cv::Mat> cropped_chips;
	double target_width = targetSize[0];
	double target_height = targetSize[1];
	if (type == 1) {
		cropped_chips.push_back(src_facade);
	}
	else if (type == 2) {
		double target_ratio_width = target_width / facadeSize[0];
		double target_ratio_height = target_height / facadeSize[1];
		if (target_ratio_height > 1.0)
			target_ratio_height = 1.0;
		if (facadeSize[0] < 1.6 * target_width || !bMultipleChips) {
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			// crop target size
			cv::Mat tmp = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			cropped_chips.push_back(tmp);
		}
		else {
			// push back multiple chips
			int index = 0;
			double start_width_ratio = index * 0.1; // not too left
			std::vector<double> confidences;
			while (start_width_ratio + target_ratio_width < 0.9) { // not too right
				// get the cropped img
				cv::Mat tmp = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				cropped_chips.push_back(tmp);
				index++;
				start_width_ratio = index * 0.1;
			}
		}
	}
	else if (type == 3) {
		double target_ratio_height = target_height / facadeSize[1];
		double target_ratio_width = target_width / facadeSize[0];
		if (target_ratio_width >= 1.0)
			target_ratio_width = 1.0;
		if (facadeSize[1] < 1.6 * target_height || !bMultipleChips) {
			double start_height_ratio = (1 - target_ratio_height) * 0.5;
			cv::Mat tmp = src_facade(cv::Rect(0, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			cropped_chips.push_back(tmp);
		}
		else {
			// push back multiple chips
			int index = 0;
			double start_height_ratio = index * 0.1;
			while (start_height_ratio + target_ratio_height < 0.9) {
				// get the cropped img
				cv::Mat tmp = src_facade(cv::Rect(0, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				cropped_chips.push_back(tmp);
				index++;
				start_height_ratio = index * 0.1;
			}
		}
	}
	else if (type == 4) {
		double longer_dim = facadeSize[0] > facadeSize[1]? facadeSize[0]: facadeSize[1];
		double target_dim = facadeSize[0] > facadeSize[1] ? target_width : target_height;
		bool bLonger_width = facadeSize[0] > facadeSize[1] ? true : false;
		double target_ratio_width = target_width / facadeSize[0];
		double target_ratio_height = target_height / facadeSize[1];
		if (longer_dim < 1.6 * target_dim || !bMultipleChips) {
			// crop target size
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			double start_height_ratio = (1 - target_ratio_height) * 0.5;
			cv::Mat tmp = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			cropped_chips.push_back(tmp);
		}
		else if (bLonger_width) {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 0;
			double start_width_ratio = index * 0.1;
			double start_height_ratio = (1 - target_ratio_height) * 0.5;
			while (start_width_ratio + target_ratio_width < 0.9) {
				// get the cropped img
				cv::Mat tmp = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				cropped_chips.push_back(tmp);
				index++;
				start_width_ratio = index * 0.1;
			}
		}
		else {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 0;
			double start_height_ratio = index * 0.1;
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			while (start_height_ratio + target_ratio_height < 0.9) {
				// get the cropped img
				cv::Mat tmp = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				cropped_chips.push_back(tmp);
				index++;
				start_height_ratio = index * 0.1;
			}
		}
	}
	else {
		// do nothing
	}
	return cropped_chips;
}

std::vector<cv::Mat> crop_chip_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips) {
	std::vector<cv::Mat> cropped_chips;
	double target_width = targetSize[0];
	double target_height = targetSize[1];
	if (type == 1) {
		cropped_chips.push_back(src_facade);
	}
	else if (type == 2) {
		double target_ratio_width = target_width / facadeSize[0];
		double target_ratio_height = target_height / facadeSize[1];
		if (target_ratio_height > 1.0)
			target_ratio_height = 1.0;
		if (facadeSize[0] < 1.6 * target_width || !bMultipleChips) {
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			// crop target size
			cv::Mat tmp = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			cropped_chips.push_back(tmp);
		}
		else {
			// push back multiple chips
			int index = 0;
			double start_width_ratio = index * 0.1; // not too left
			std::vector<double> confidences;
			while (start_width_ratio + target_ratio_width < 0.9) { // not too right
																   // get the cropped img
				cv::Mat tmp = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				cropped_chips.push_back(tmp);
				index++;
				start_width_ratio = index * 0.1;
			}
		}
	}
	else if (type == 3) {
		double target_ratio_height = target_height / facadeSize[1];
		double target_ratio_width = target_width / facadeSize[0];
		if (target_ratio_width >= 1.0)
			target_ratio_width = 1.0;
		if (facadeSize[1] < 1.6 * target_height || !bMultipleChips) {
			double start_height_ratio = (1 - target_ratio_height);
			cv::Mat tmp = src_facade(cv::Rect(0, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			cropped_chips.push_back(tmp);
		}
	}
	else if (type == 4) {
		double longer_dim = facadeSize[0] > facadeSize[1] ? facadeSize[0] : facadeSize[1];
		double target_dim = facadeSize[0] > facadeSize[1] ? target_width : target_height;
		bool bLonger_width = facadeSize[0] > facadeSize[1] ? true : false;
		double target_ratio_width = target_width / facadeSize[0];
		double target_ratio_height = target_height / facadeSize[1];
		if (longer_dim < 1.6 * target_dim || !bMultipleChips) {
			// crop target size
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			double start_height_ratio = (1 - target_ratio_height);
			cv::Mat tmp = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			cropped_chips.push_back(tmp);
		}
		else if (bLonger_width) {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 0;
			double start_width_ratio = index * 0.1;
			double start_height_ratio = (1 - target_ratio_height);
			while (start_width_ratio + target_ratio_width < 0.9) {
				// get the cropped img
				cv::Mat tmp = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				cropped_chips.push_back(tmp);
				index++;
				start_width_ratio = index * 0.1;
			}
		}
		else {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 0;
			double start_height_ratio = (1 - target_ratio_height);
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			cv::Mat tmp = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			cropped_chips.push_back(tmp);
		}
	}
	else {
		// do nothing
	}
	return cropped_chips;
}

std::vector<double> compute_chip_info(cv::Mat chip, ModelInfo& mi, bool bDebug, std::string img_filename) {
	std::vector<double> chip_info;
	cv::Mat chip_src = chip.clone();
	cv::Mat chip_seg;
	apply_segmentation_model(chip_src, chip_seg, mi, false, img_filename);
	cv::Mat dnn_img;
	process_chip(chip_seg, dnn_img, mi, false, img_filename);
	// go to grammar classifier
	int num_classes = mi.number_grammars;
	cv::Mat dnn_img_rgb;
	cv::cvtColor(dnn_img.clone(), dnn_img_rgb, CV_BGR2RGB);
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
	// Deserialize the ScriptModule from a file using torch::jit::load().
	torch::Tensor out_tensor = mi.classifier_module->forward(inputs).toTensor();
	torch::Tensor confidences_tensor = torch::softmax(out_tensor, 1);
	//std::cout << confidences_tensor.slice(1, 0, num_classes) << std::endl;
	double best_score = 0;
	for (int i = 0; i < num_classes; i++) {
		double tmp = confidences_tensor.slice(1, i, i + 1).item<float>();
		if (tmp > best_score) {
			best_score = tmp;
			best_class = i;
		}
	}
	// currently only add confidence value to the info vector
	chip_info.push_back(best_score);
	if (bDebug)
	{
		cv::imwrite(img_filename, chip_src);
	}
	return chip_info;
}

int choose_best_chip(std::vector<cv::Mat> chips, ModelInfo& mi, bool bDebug, std::string img_filename) {
	int best_chip_id = 0;
	if (chips.size() == 1)
		best_chip_id = 0;
	else {
		std::vector<double> confidence_values;
		confidence_values.resize(chips.size());
		for (int i = 0; i < chips.size(); i++) {
			std::string filename = "../data/chips/" + img_filename.substr(0, img_filename.find(".png")) + "_" +to_string(i) +".png";
			confidence_values[i] = compute_chip_info(chips[i], mi, bDebug, filename)[0];
			if (bDebug) {
				std::cout << "chip " << i<< " score is " << confidence_values[i] << std::endl;
			}
		}
		// find the best chip id
		best_chip_id = std::max_element(confidence_values.begin(), confidence_values.end()) - confidence_values.begin();
		if (bDebug) {
			std::cout << "best_chip_id is " << best_chip_id << std::endl;
		}
	}
	return best_chip_id;
}

std::vector<int> adjust_chip(cv::Mat chip) {
	std::vector<int> boundaries;
	boundaries.resize(4); // top, bottom, left and right
	if (chip.channels() != 1) {
		boundaries[0] = 0;
		boundaries[1] = chip.size().height - 1;
		boundaries[2] = 0;
		boundaries[3] = chip.size().width - 1;
		return boundaries;
	}
	// find the boundary
	double threshold = 0.9;
	bool bStopscan = false;
	// top 
	int pos_top = 0;
	for (int i = 0; i < chip.size().height; i++) {
		if (bStopscan)
			break;
		for (int j = 0; j < chip.size().width; j++) {
			if ((int)chip.at<uchar>(i, j) == 0) {
				pos_top = i;
				bStopscan = true;
				break;
			}
		}
	}
	// bottom 
	bStopscan = false;
	int pos_bot = 0;
	for (int i = chip.size().height - 1; i >= 0; i--) {
		if (bStopscan)
			break;
		for (int j = 0; j < chip.size().width; j++) {
			//noise
			if ((int)chip.at<uchar>(i, j) == 0) {
				pos_bot = i;
				bStopscan = true;
				break;
			}
		}
	}

	// left
	bStopscan = false;
	int pos_left = 0;
	for (int i = 0; i < chip.size().width; i++) {
		if (bStopscan)
			break;
		for (int j = 0; j < chip.size().height; j++) {
			//noise
			if ((int)chip.at<uchar>(j, i) == 0) {
				pos_left = i;
				bStopscan = true;
				break;
			}
		}
	}
	// right
	bStopscan = false;
	int pos_right = 0;
	for (int i = chip.size().width - 1; i >= 0; i--) {
		if (bStopscan)
			break;
		for (int j = 0; j < chip.size().height; j++) {
			//noise
			if ((int)chip.at<uchar>(j, i) == 0) {
				pos_right = i;
				bStopscan = true;
				break;
			}
		}
	}
	boundaries[0] = pos_top;
	boundaries[1] = pos_bot;
	boundaries[2] = pos_left;
	boundaries[3] = pos_right;
	return boundaries;
}

void apply_segmentation_model(cv::Mat &croppedImage, cv::Mat &chip_seg, ModelInfo& mi, bool bDebug, std::string img_filename) {
	int run_times = 3;
	cv::Mat src_img = croppedImage.clone();
	// scale to seg size
	cv::resize(src_img, src_img, cv::Size(mi.segImageSize[0], mi.segImageSize[1]));
	cv::Mat dnn_img_rgb;
	cv::cvtColor(src_img, dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, (int)mi.segImageSize[0], (int)mi.segImageSize[1], 3 }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.5).div(0.5);
	img_tensor[0][1] = img_tensor[0][1].sub(0.5).div(0.5);
	img_tensor[0][2] = img_tensor[0][2].sub(0.5).div(0.5);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);
	std::vector<std::vector<int>> color_mark;
	color_mark.resize((int)mi.segImageSize[1]);
	for (int i = 0; i < color_mark.size(); i++) {
		color_mark[i].resize((int)mi.segImageSize[0]);
		for (int j = 0; j < color_mark[i].size(); j++) {
			color_mark[i][j] = 0;
		}
	}
	// run three times
	for (int i = 0; i < run_times; i++) {
		torch::Tensor out_tensor = mi.seg_module->forward(inputs).toTensor();
		out_tensor = out_tensor.squeeze().detach().permute({ 1,2,0 });
		out_tensor = out_tensor.add(1).mul(0.5 * 255).clamp(0, 255).to(torch::kU8);
		//out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
		out_tensor = out_tensor.to(torch::kCPU);
		cv::Mat resultImg((int)mi.segImageSize[0], (int)mi.segImageSize[1], CV_8UC3);
		std::memcpy((void*)resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8)*out_tensor.numel());
		// gray img
		// correct the color
		for (int i = 0; i < resultImg.size().height; i++) {
			for (int j = 0; j < resultImg.size().width; j++) {
				if (resultImg.at<cv::Vec3b>(i, j)[0] > 160)
					color_mark[i][j] += 0;
				else
					color_mark[i][j] += 1;
			}
		}
		/*if (bDebug) {
			cv::cvtColor(resultImg, resultImg, CV_RGB2BGR);
			cv::imwrite("../data/test/seg_" + to_string(i) + ".png", resultImg);
		}*/
	}
	cv::Mat gray_img((int)mi.segImageSize[0], (int)mi.segImageSize[1], CV_8UC1);
	int num_majority = ceil(0.5 * run_times);
	for (int i = 0; i < color_mark.size(); i++) {
		for (int j = 0; j < color_mark[i].size(); j++) {
			if (color_mark[i][j] < num_majority)
				gray_img.at<uchar>(i, j) = (uchar)0;
			else
				gray_img.at<uchar>(i, j) = (uchar)255;
		}
	}
	// scale to grammar size
	cv::resize(gray_img, chip_seg, croppedImage.size());
	// correct the color
	for (int i = 0; i < chip_seg.size().height; i++) {
		for (int j = 0; j < chip_seg.size().width; j++) {
			//noise
			if ((int)chip_seg.at<uchar>(i, j) < 128) {
				chip_seg.at<uchar>(i, j) = (uchar)0;
			}
			else
				chip_seg.at<uchar>(i, j) = (uchar)255;
		}
	}
	if (bDebug) {
		std::cout << "num_majority is " << num_majority << std::endl;
	}
	//adjust boundaries
	std::vector<int> boundaries = adjust_chip(chip_seg);
	chip_seg = chip_seg(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
	croppedImage = croppedImage(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
}

bool process_chip(cv::Mat chip_seg, cv::Mat& dnn_img, ModelInfo& mi, bool bDebug, std::string img_filename) {
	// default size for NN
	int width = mi.defaultSize[0] - 2 * mi.paddingSize[0];
	int height = mi.defaultSize[1] - 2 * mi.paddingSize[1];
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(chip_seg, scale_img, cv::Size(width, height));
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
	int padding_size = mi.paddingSize[0];
	int borderType = cv::BORDER_CONSTANT;
	cv::Mat aligned_img_padding;
	cv::copyMakeBorder(aligned_img, aligned_img_padding, padding_size, padding_size, padding_size, padding_size, borderType, bg_color);

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
	if (bDebug) {
		cv::imwrite(mi.dilatesFolder + "/" + img_filename, dilation_dst);
		cv::imwrite(mi.alignsFolder + "/" + img_filename, aligned_img);
		cv::imwrite(mi.dnnsInFolder + "/" + img_filename, dnn_img);
	}
	return true;
}

std::vector<double> feedDnn(cv::Mat dnn_img, FacadeInfo& fi, ModelInfo& mi, bool bDebug, std::string img_filename) {
	int num_classes = mi.number_grammars;
	cv::Mat dnn_img_rgb;
	cv::cvtColor(dnn_img.clone(), dnn_img_rgb, CV_BGR2RGB);
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
		torch::Tensor out_tensor = mi.classifier_module->forward(inputs).toTensor();
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
	// number of paras
	int num_paras = mi.grammars[best_class - 1].number_paras;

	torch::Tensor out_tensor_grammar = mi.grammars[best_class - 1].grammar_model->forward(inputs).toTensor();
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
	drawing = drawing(cv::Rect(padding_size, padding_size, src_img.size().width, src_img.size().height));
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
	if (bDebug) {
		std::cout << "paras[0] is " << paras[0] << std::endl;
		std::cout << "paras[1] is " << paras[1] << std::endl;
		std::cout << "img_rows is " << paras[0] * (imageRows.second - imageRows.first) + imageRows.first << std::endl;
		std::cout << "img_cols is " << paras[1] * (imageCols.second - imageCols.first) + imageCols.first<< std::endl;
	}
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
