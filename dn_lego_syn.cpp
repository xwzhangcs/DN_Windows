#include "dn_lego_syn.h"
#include <stack>
#include "Utils.h"
#include "dn_lego_eval.h"
#include "optGrammarParas.h"
#include <windows.h>

int main(int argc, const char* argv[]) {
	if (argc != 4) {
		std::cerr << "usage: app <path-to-metadata> <path-to-model-config-JSON-file>\n";
		return -1;
	}
	eval_seg_models("../data/test", "../data/test/pix2pix_256", "../seg_model.pt", 256, "../data/test/pix2pix_256.txt");
	//adjust_seg_colors("../data/test/B", "../data/test/B_adjust");
	//img_convert("D:/LEGO_meeting_summer_2019/1014/test/B");
	//test_overlay_images("D:/LEGO_meeting_summer_2019/1014/test/A", "D:/LEGO_meeting_summer_2019/1014/test/B", "D:/LEGO_meeting_summer_2019/1014/test/overlay");
	//
	return 0;
	/*std::string aoi = "../data/metrics/eval";
	FacadeSeg eval_obj;
	eval_obj.eval(aoi + "/pix2pix", aoi + "/gt", aoi + "/pix2pix_eval.txt");
	eval_obj.eval(aoi + "/deepFill", aoi + "/gt", aoi + "/deepFill_eval.txt");
	eval_obj.eval(aoi + "/our_before", aoi + "/gt", aoi + "/our_eval.txt");
	eval_obj.eval(aoi + "/our_after_gt", aoi + "/gt", aoi + "/our_opt_gt_eval.txt");
	eval_obj.eval(aoi + "/our_after_seg", aoi + "/gt", aoi + "/our_opt_v1_eval.txt");
	eval_obj.eval(aoi + "/our_after_seg_without_blob", aoi + "/gt", aoi + "/our_opt_v2_eval.txt");
	return 0;*/
	std::string path(argv[1]);
	std::vector<std::string> clusters = get_all_files_names_within_folder(argv[1]);
	ModelInfo mi;
	readModeljson(argv[3], mi);
	test_segmentation_model("D:/LEGO_meeting_summer_2019/1012/src_facades/backup_v3", mi);
	return 0;
	std::clock_t start;
	double duration;
	start = std::clock();
	test_seg2grammars(mi, "../data/test_opt", "../data/test_opt_out");
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "duration: " << duration << '\n';
	return 0;
	for (int i = 0; i < clusters.size(); i++) {
		std::vector<std::string> metaFiles = get_all_files_names_within_folder(path + "/" + clusters[i] + "/metadata");
		for (int j = 0; j < metaFiles.size(); j++) {
			std::string metajson = path + "/" + clusters[i] + "/metadata/" + metaFiles[j];
			std::string img_filename = clusters[i] + "_" + metaFiles[j].substr(0, metaFiles[j].find(".json")) + ".png";
			std::cout << metajson << ", " << img_filename << std::endl;
			/*if (img_filename != "0001_0048.png")
				continue;*/
			// read metajson
			FacadeInfo fi;
			readMetajson(metajson, fi);
			ChipInfo chip;
			bool bvalid = chipping(fi, mi, chip, true, mi.debug, img_filename);
			if (bvalid) {
				cv::Mat dnn_img;
				process_chip(chip, mi, mi.debug, img_filename);
				std::vector<double> predictions = feedDnn(chip, fi, mi, mi.debug, img_filename);
				std::cout << fi.win_color << ", " << fi.bg_color << std::endl;
				if (fi.win_color.size() > 0 && fi.bg_color.size() > 0) {
					cv::Scalar win_avg_color(fi.win_color[0], fi.win_color[1], fi.win_color[2], 0);
					cv::Scalar bg_avg_color(fi.bg_color[0], fi.bg_color[1], fi.bg_color[2], 0);
					std::string img_name = fi.imgName;
					cv::Mat src_facade = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
					synthesis(predictions, chip.seg_image.size(), mi.dnnsOutFolder, win_avg_color, bg_avg_color, mi.debug, img_filename);
				}
			}
			//writeMetajson(metajson, fi);
		}
	}
	return 0;
}

void eval_seg_models(std::string images_path, std::string output_path, std::string model_path, int segImageSize, std::string results_txt) {
	std::vector<std::string> images = get_all_files_names_within_folder(images_path + "/A");
	std::cout << "images size is " << images.size() << std::endl;
	// load model
	std::shared_ptr<torch::jit::script::Module> seg_module;
	seg_module = torch::jit::load(model_path);
	seg_module->to(at::kCUDA);
	assert(mi.seg_module != nullptr);
	std::ofstream out_param(results_txt, std::ios::app);
	out_param << "facade_id";
	out_param << ",";
	out_param << "pAccuracy";
	out_param << ",";
	out_param << "precision";
	out_param << ",";
	out_param << "recall";
	out_param << "\n";
	double avg_accuracy = 0;
	double avg_precision = 0;
	double avg_recall = 0;
	for (int index = 0; index < images.size(); index++) {
		std::string img_name = images_path + "/A/" + images[index];
		cv::Mat src_img = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
		if (src_img.channels() == 4) // ensure there're 3 channels
			cv::cvtColor(src_img, src_img, CV_BGRA2BGR);
		int run_times = 3;
		// scale to seg size
		cv::Mat scale_img;
		cv::resize(src_img, scale_img, cv::Size(segImageSize, segImageSize));
		cv::Mat dnn_img_rgb;
		cv::cvtColor(scale_img, dnn_img_rgb, CV_BGR2RGB);
		cv::Mat img_float;
		dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
		int channels = 3;
		auto img_tensor = torch::from_blob(img_float.data, { 1, (int)segImageSize, segImageSize, channels }).to(torch::kCUDA);
		img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
		img_tensor[0][0] = img_tensor[0][0].sub(0.5).div(0.5);
		img_tensor[0][1] = img_tensor[0][1].sub(0.5).div(0.5);
		img_tensor[0][2] = img_tensor[0][2].sub(0.5).div(0.5);

		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(img_tensor);
		std::vector<std::vector<int>> color_mark;
		color_mark.resize((int)segImageSize);
		for (int i = 0; i < color_mark.size(); i++) {
			color_mark[i].resize((int)segImageSize);
			for (int j = 0; j < color_mark[i].size(); j++) {
				color_mark[i][j] = 0;
			}
		}
		// run three times
		for (int i = 0; i < run_times; i++) {
			torch::Tensor out_tensor;
			// load segmentation model
			out_tensor = seg_module->forward(inputs).toTensor();
			out_tensor = out_tensor.squeeze().detach().permute({ 1,2,0 });
			out_tensor = out_tensor.add(1).mul(0.5 * 255).clamp(0, 255).to(torch::kU8);
			//out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
			out_tensor = out_tensor.to(torch::kCPU);
			cv::Mat resultImg((int)segImageSize, segImageSize, CV_8UC3);
			std::memcpy((void*)resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8)*out_tensor.numel());
			// gray img
			// correct the color
			for (int h = 0; h < resultImg.size().height; h++) {
				for (int w = 0; w < resultImg.size().width; w++) {
					if (resultImg.at<cv::Vec3b>(h, w)[0] > 160)
						color_mark[h][w] += 0;
					else
						color_mark[h][w] += 1;
				}
			}
		}
		cv::Mat gray_img((int)segImageSize, (int)segImageSize, CV_8UC1);
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
		cv::Mat seg_img(src_img.size(), CV_8UC3);
		cv::resize(gray_img, gray_img, src_img.size());
		// correct the color
		for (int i = 0; i < seg_img.size().height; i++) {
			for (int j = 0; j < seg_img.size().width; j++) {
				//noise
				if ((int)gray_img.at<uchar>(i, j) < 128) {
					seg_img.at<cv::Vec3b>(i, j)[0] = 0;
					seg_img.at<cv::Vec3b>(i, j)[1] = 0;
					seg_img.at<cv::Vec3b>(i, j)[2] = 255;
				}
				else {
					seg_img.at<cv::Vec3b>(i, j)[0] = 255;
					seg_img.at<cv::Vec3b>(i, j)[1] = 0;
					seg_img.at<cv::Vec3b>(i, j)[2] = 0;
				}
			}
		}
		std::string output_img_name = output_path + "/" + images[index];
		cv::imwrite(output_img_name, seg_img);

		//eval
		cv::Mat gt_img = cv::imread(images_path + "/B/" + images[index], CV_LOAD_IMAGE_UNCHANGED);
		std::vector<double> evaluations = eval_accuracy(seg_img, gt_img);
		out_param << images[index];
		out_param << ",";
		out_param << evaluations[0];
		out_param << ",";
		out_param << evaluations[1];
		out_param << ",";
		out_param << evaluations[2];
		out_param << "\n";
		avg_accuracy += evaluations[0];
		avg_precision += evaluations[1];
		avg_recall += evaluations[2];
	}
	out_param << "Average_Score";
	out_param << ",";
	out_param << avg_accuracy / images.size();
	out_param << ",";
	out_param << avg_precision / images.size();
	out_param << ",";
	out_param << avg_recall / images.size();
	out_param << "\n";
}

void test_affine_transformation(std::string image_path, std::string output_path) {
	std::vector<std::string> images = get_all_files_names_within_folder(image_path);
	std::cout << "images size is " << images.size() << std::endl;
	cv::Scalar wall_color(255, 0, 0); // white back ground
	cv::Scalar window_color(0, 0, 255); // black for windows
	for (int i = 0; i < images.size(); i++) {
		std::string img_name = image_path + '/' + images[i];
		cv::Mat src_img = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat seg_rgb = cv::imread("../data/gt/" + images[i], CV_LOAD_IMAGE_UNCHANGED);
		double score_opt = 0;
		std::vector<double> paras_opt;
		paras_opt.resize(5);
		for (int translateX = -20; translateX < 20; translateX += 5) {
			for (int translateY = -20; translateY < 20; translateY += 5) {
				for (int theta = -10; theta <= 10; theta += 2) {
					for (int scaleX = -5; scaleX <= 5; scaleX += 1) {
						for (int scaleY = -5; scaleY <= 5; scaleY += 1) {
							// Stores affine transformation matrix
							cv::Mat affineTransform(2, 3, CV_32FC1);
							// Initialise translational, rotational and scaling values for transformation
							affineTransform.at<float>(0, 0) = cos(theta * CV_PI / 180) * (1.0 + scaleX * 0.1);
							affineTransform.at<float>(0, 1) = -sin(theta * CV_PI / 180) * (1.0 + scaleY * 0.1);
							affineTransform.at<float>(0, 2) = translateX;

							affineTransform.at<float>(1, 0) = sin(theta * CV_PI / 180) * (1.0 + scaleX * 0.1);
							affineTransform.at<float>(1, 1) = cos(theta * CV_PI / 180) * (1.0 + scaleY * 0.1);
							affineTransform.at<float>(1, 2) = translateY;
							cv::Mat dst_tmp = cv::Mat(src_img.size(), CV_8UC3, wall_color);
							warpAffine(src_img, dst_tmp, affineTransform, src_img.size(), cv::INTER_LINEAR,
								cv::BORDER_CONSTANT,
								cv::Scalar(255, 0, 0));

							std::vector<double> evaluations = util::eval_accuracy(dst_tmp, seg_rgb);
							double score_tmp = 0.5 * evaluations[1] + 0.5 *evaluations[2];
							//std::cout << "score_tmp is " << score_tmp << std::endl;
							if (score_tmp > score_opt) {
								score_opt = score_tmp;
								paras_opt[0] = translateX;
								paras_opt[1] = translateY;
								paras_opt[2] = theta;
								paras_opt[3] = scaleX;
								paras_opt[4] = scaleY;
							}

						}
					}
				}
			}
		}
		cv::Mat affineTransform_opt(2, 3, CV_32FC1);
		// Initialise translational, rotational and scaling values for transformation
		affineTransform_opt.at<float>(0, 0) = cos(paras_opt[2] * CV_PI / 180) * (1.0 + paras_opt[3] * 0.1);
		affineTransform_opt.at<float>(0, 1) = -sin(paras_opt[2] * CV_PI / 180) * (1.0 + paras_opt[4] * 0.1);
		affineTransform_opt.at<float>(0, 2) = paras_opt[0];

		affineTransform_opt.at<float>(1, 0) = sin(paras_opt[2] * CV_PI / 180) * (1.0 + paras_opt[3] * 0.1);
		affineTransform_opt.at<float>(1, 1) = cos(paras_opt[2] * CV_PI / 180) * (1.0 + paras_opt[4] * 0.1);
		affineTransform_opt.at<float>(1, 2) = paras_opt[1];
		cv::Mat dst_img = cv::Mat(src_img.size(), CV_8UC3, wall_color);
		warpAffine(src_img, dst_img, affineTransform_opt, src_img.size(), cv::INTER_LINEAR,
			cv::BORDER_CONSTANT,
			cv::Scalar(255, 0, 0));
		cv::imwrite(output_path + '/' + images[i], dst_img);
		std::vector<double> evaluations = util::eval_accuracy(dst_img, seg_rgb);
		double score = /*0.5 * evaluations[0] +*/ 0.5 * evaluations[1] + 0.5 *evaluations[2];
		std::cout << "score after is " << score << std::endl;
		std::cout << "affineTransform_opt is " << affineTransform_opt << std::endl;
	}
}

void test_seg2grammars(ModelInfo& mi, std::string image_path, std::string output_path) {
	std::vector<std::string> images = get_all_files_names_within_folder(image_path);
	std::cout << "images size is " << images.size() << std::endl;
	for (int i = 0; i < images.size(); i++) {
		std::string img_name = image_path + '/' + images[i];
		cv::Mat src_img = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
		if(src_img.channels() == 4)
			cv::cvtColor(src_img.clone(), src_img, CV_BGRA2GRAY);
		// default size for NN
		int width = mi.defaultSize[0] - 2 * mi.paddingSize[0];
		int height = mi.defaultSize[1] - 2 * mi.paddingSize[1];
		cv::Scalar bg_color(255, 255, 255); // white back ground
		cv::Scalar window_color(0, 0, 0); // black for windows
		cv::Mat scale_img;
		cv::resize(src_img, scale_img, cv::Size(width, height));
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
		cv::Mat dnn_img = cv::Mat(aligned_img_padding.size(), CV_8UC3, bg_color);
		for (int i = 1; i< contours.size(); i++)
		{
			if (hierarchy[i][3] != 0) continue;
			cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
		}
		//
		int num_classes = mi.number_grammars;
		// 
		std::vector<int> separation_x;
		std::vector<int> separation_y;
		cv::Mat spacing_img = dnn_img.clone();
		find_spacing(spacing_img, separation_x, separation_y, true);
		int spacing_r = separation_y.size() / 2;
		int spacing_c = separation_x.size() / 2;

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
		if (true)
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
		// generate RGB seg img
		cv::Mat seg_rgb = cv::Mat(src_img.size(), CV_8UC3, bg_color);
		for (int i = 0; i < src_img.size().height; i++) {
			for (int j = 0; j < src_img.size().width; j++) {
				//noise
				if ((int)src_img.at<uchar>(i, j) == 0) {
					seg_rgb.at<cv::Vec3b>(i, j)[0] = 0;
					seg_rgb.at<cv::Vec3b>(i, j)[1] = 0;
					seg_rgb.at<cv::Vec3b>(i, j)[2] = 255;
				}
				else {
					seg_rgb.at<cv::Vec3b>(i, j)[0] = 255;
					seg_rgb.at<cv::Vec3b>(i, j)[1] = 0;
					seg_rgb.at<cv::Vec3b>(i, j)[2] = 0;
				}
			}
		}
		seg_rgb = cv::imread("../data/pix2pix/" + images[i], CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat gt_img = cv::imread("../data/gt/" + images[i], CV_LOAD_IMAGE_UNCHANGED);
		std::vector<double> predictions;
		if (best_class == 1) {
			predictions = grammar1(mi, paras, true);
		}
		else if (best_class == 2) {
			predictions = grammar2(mi, paras, true);
		}
		else if (best_class == 3) {
			predictions = grammar3(mi, paras, true);
		}
		else if (best_class == 4) {
			predictions = grammar4(mi, paras, true);
		}
		else if (best_class == 5) {
			predictions = grammar5(mi, paras, true);
		}
		else if (best_class == 6) {
			predictions = grammar6(mi, paras, true);
		}
		else {
			//do nothing
			predictions = grammar1(mi, paras, true);
		}
		if (best_class % 2 == 0) {
			if (abs(predictions[0] + 1 - spacing_r) <= 1 && predictions[0] > 1)
				predictions[0] = spacing_r - 1;
		}
		else {
			if (abs(predictions[0] - spacing_r) <= 1 && predictions[0] > 1)
				predictions[0] = spacing_r;
		}
		if (abs(predictions[1] - spacing_c) <= 1 && predictions[1] > 1)
			predictions[1] = spacing_c;
		// opt
		double score_opt = 0;
		bool bOpt = true;
		std::vector<double> predictions_opt;
		std::vector<double> trans_opt;
		if (predictions.size() == 5 && bOpt) {
			opt_without_doors(seg_rgb, predictions_opt, predictions, mi.opt_step, mi.opt_range);
			//opt_without_doors(seg_rgb, predictions_opt, trans_opt, predictions);
		}
		if (predictions.size() == 8 && bOpt) {
			opt_with_doors(seg_rgb, predictions_opt, predictions, mi.opt_step, mi.opt_range);
		}
		/*cv::Scalar win_avg_color(0, 0, 255, 0);
		cv::Scalar bg_avg_color(255, 0, 0, 0);
		cv::Mat syn_img = synthesis(predictions, src_img.size(), output_path, win_avg_color, bg_avg_color, true, images[i]);*/
		cv::Scalar win_avg_color(0, 0, 255, 0);
		cv::Scalar bg_avg_color(255, 0, 0, 0);
		int gt_blobs = blobs(gt_img);
		cv::Mat syn_img = synthesis(predictions, src_img.size(), "../data", win_avg_color, bg_avg_color, false, "syn.png");
		std::vector<double> evaluations = eval_accuracy(syn_img, gt_img);
		int tmp_blobs = 0;
		if(predictions.size() == 5)
			tmp_blobs = predictions[0] * predictions[1];
		else
			tmp_blobs = predictions[0] * predictions[1] + predictions[3];
		double blobs_score = 1 - 1.0 * abs(tmp_blobs - gt_blobs) / gt_blobs;
		//double score = 0.25 * evaluations[0] + 0.25 * evaluations[1] + 0.25 * evaluations[2] + 0.25 * blobs_score;
		double score = 0.34 * evaluations[0] + 0.33 * evaluations[1] + 0.33 *evaluations[2];
		//double score = 0.5 * evaluations[1] + 0.5 *evaluations[2];
		std::cout << "score before is " << score << std::endl;
		std::cout << "predictions[0] is " << predictions[0] << std::endl;
		std::cout << "predictions[1] is " << predictions[1] << std::endl;
		std::cout << "predictions[3] is " << predictions[3] << std::endl;
		std::cout << "predictions[4] is " << predictions[4] << std::endl;
		std::cout << "predictions_opt size is " << predictions_opt.size() << std::endl;
		cv::Mat syn_img_opt = synthesis_opt(predictions_opt, src_img.size(), win_avg_color, bg_avg_color, true, output_path + "/" + images[i]);
		cv::imwrite("../data/test.png", seg_rgb);

		std::vector<double> evaluations_opt = eval_accuracy(syn_img_opt, gt_img);
		std::cout << "predictions_opt[0] is " << predictions_opt[0] << std::endl;
		std::cout << "predictions_opt[1] is " << predictions_opt[1] << std::endl;
		std::cout << "predictions_opt[3] is " << predictions_opt[3] << std::endl;
		std::cout << "predictions_opt[4] is " << predictions_opt[4] << std::endl;
		std::cout << "predictions_opt[5] is " << predictions_opt[5] << std::endl;
		std::cout << "predictions_opt[6] is " << predictions_opt[6] << std::endl;
		std::cout << "predictions_opt[7] is " << predictions_opt[7] << std::endl;
		std::cout << "predictions_opt[8] is " << predictions_opt[8] << std::endl;
		if (predictions.size() == 5)
			tmp_blobs = predictions_opt[0] * predictions_opt[1];
		else
			tmp_blobs = predictions_opt[0] * predictions_opt[1] + predictions_opt[3];
		blobs_score = 1 - 1.0 * abs(tmp_blobs - gt_blobs) / gt_blobs;
		std::cout << "evaluations_opt[0] is " << evaluations_opt[0] << std::endl;
		std::cout << "evaluations_opt[1] is " << evaluations_opt[1] << std::endl;
		std::cout << "evaluations_opt[2] is " << evaluations_opt[2] << std::endl;
		//score = 0.25 * evaluations_opt[0] + 0.25 * evaluations_opt[1] + 0.25 *evaluations_opt[2] + 0.25 * blobs_score;
		score = 0.34 * evaluations_opt[0] + 0.33 * evaluations_opt[1] + 0.33 *evaluations_opt[2];
		//score = 0.5 * evaluations_opt[1] + 0.5 *evaluations_opt[2];
		std::cout << "score after is " << score << std::endl;
	}
}

int blobs(cv::Mat& src_img) {
	// find the number of windows & doors
	int padding_size = 5;
	int borderType = cv::BORDER_CONSTANT;
	cv::Mat padding_img, src_gray;
	cv::copyMakeBorder(src_img, padding_img, padding_size, padding_size, padding_size, padding_size, borderType, cv::Scalar(255, 0, 0));
	cv::cvtColor(padding_img, src_gray, CV_BGR2GRAY);
	for (int i = 0; i < padding_img.size().height; i++) {
		for (int j = 0; j < padding_img.size().width; j++) {
			// wall
			if (padding_img.at<cv::Vec3b>(i, j)[0] == 0 && padding_img.at<cv::Vec3b>(i, j)[1] == 0 && padding_img.at<cv::Vec3b>(i, j)[2] == 255) {
				src_gray.at<uchar>(i, j) = (uchar)0;
			}
			else {// non-wall
				src_gray.at<uchar>(i, j) = (uchar)255;
			}
		}
	}
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(src_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));
	return contours.size() - 1;
}

void opt_without_doors(cv::Mat& seg_rgb, std::vector<double>& predictions_opt, std::vector<double> predictions_init, int step, double range) {
	double score_opt = 0;
	predictions_opt.resize(9);
	int width = seg_rgb.size().width; 
	int height = seg_rgb.size().height;
	double FH = height * 1.0 / predictions_init[0];
	double FW = width * 1.0 / predictions_init[1];
	double step_size_w = step / FW;
	double step_size_h = step / FH;
	double range_adjust = range;
	int v3_min = - ceil(range_adjust / step_size_w);
	int v3_max = -v3_min;
	int v4_min = -ceil(range_adjust / step_size_h);
	int v4_max = -v4_min;
	double step_size_m_w = step * 1.0 / width;
	double step_size_m_h = step * 1.0 / height;
	int gt_blobs = blobs(seg_rgb);
	std::cout << "gt_blobs is " << gt_blobs << std::endl;
	std::cout << "step_size_w is " << step_size_w << std::endl;
	std::cout << "step_size_h is " << step_size_h << std::endl;
	std::cout << "v3_min is " << v3_min << std::endl;
	std::cout << "v3_max is " << v3_max << std::endl;
	std::cout << "v4_min is " << v4_min << std::endl;
	std::cout << "v4_max is " << v4_max << std::endl;
	std::cout << "step_size_m_w is " << step_size_m_w << std::endl;
	std::cout << "step_size_m_h is " << step_size_m_h << std::endl;
	for (int v1 = -2; v1 <= 2; v1++) {
		for (int v2 = -2; v2 <= 2; v2++) {
			for (int v3 = v3_min; v3 <= v3_max; v3++) {
				for (int v4 = v4_min; v4 <= v4_max; v4++) {
					for (int m_t = 0; m_t <= 5; m_t++) {
						for (int m_b = 0; m_b <= 5; m_b++) {
							for (int m_l = 0; m_l <= 5; m_l++) {
								for (int m_r = 0; m_r <= 5; m_r++) {
									std::vector<double> predictions_tmp;
									predictions_tmp.push_back(v1 + predictions_init[0]);
									predictions_tmp.push_back(v2 + predictions_init[1]);
									predictions_tmp.push_back(predictions_init[2]);
									if (v3 * step_size_w + predictions_init[3] < 0 || v3 * step_size_w + predictions_init[3] > 1 - step_size_w)
										continue;
									predictions_tmp.push_back(v3 * step_size_w + predictions_init[3]);
									if (v4 * step_size_h + predictions_init[4] < 0 || v4 * step_size_h + predictions_init[4] > 1 - step_size_h)
										continue;
									predictions_tmp.push_back(v4 * step_size_h + predictions_init[4]);
									predictions_tmp.push_back(m_t * step_size_m_h);
									predictions_tmp.push_back(m_b * step_size_m_h);
									predictions_tmp.push_back(m_l * step_size_m_w);
									predictions_tmp.push_back(m_r * step_size_m_w);
									cv::Scalar win_avg_color(0, 0, 255, 0);
									cv::Scalar bg_avg_color(255, 0, 0, 0);
									cv::Mat syn_img = synthesis_opt(predictions_tmp, seg_rgb.size(), win_avg_color, bg_avg_color, false, "../data/tmp.png");
									std::vector<double> evaluations = eval_accuracy(syn_img, seg_rgb);
									int tmp_blobs = predictions_tmp[0] * predictions_tmp[1];
									double blobs_score = 1 - 1.0 * abs(tmp_blobs - gt_blobs) / gt_blobs;
									//std::cout << "tmp_blobs is " << tmp_blobs << std::endl;
									//std::cout << "blobs_score is " << blobs_score << std::endl;
									//double score_tmp = 0.25 * evaluations[0] + 0.25 * evaluations[1] + 0.25 *evaluations[2] + 0.25 * blobs_score;
									double score_tmp = 0.34 * evaluations[0] + 0.33 * evaluations[1] + 0.33 *evaluations[2];
									//double score_tmp =  0.5 * evaluations[1] + 0.5 *evaluations[2];
									//std::cout << "score_tmp is " << score_tmp << std::endl;
									if (score_tmp > score_opt) {
										score_opt = score_tmp;
										for (int index = 0; index < predictions_tmp.size(); index++) {
											predictions_opt[index] = predictions_tmp[index];
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

void opt_without_doors(cv::Mat& seg_rgb, std::vector<double>& predictions_opt, std::vector<double>& trans_opt, std::vector<double> predictions_init) {
	double score_opt = 0;
	predictions_opt.resize(9);
	trans_opt.resize(3);
	double step_size = 0.02;
	double step_size_m = 0.03;
	int gt_blobs = blobs(seg_rgb);
	std::cout << "gt_blobs is " << gt_blobs << std::endl;
	for (int v1 = -1; v1 <= 1; v1++) {
		for (int v2 = -1; v2 <= 1; v2++) {
			for (int v3 = -8; v3 <= 8; v3++) {
				for (int v4 = -8; v4 <= 8; v4++) {
					for (int m_t = 0; m_t <= 2; m_t++) {
						for (int m_b = 0; m_b <= 2; m_b++) {
							for (int m_l = 0; m_l <= 2; m_l++) {
								for (int m_r = 0; m_r <= 2; m_r++) {
									std::vector<double> predictions_tmp;
									predictions_tmp.push_back(v1 + predictions_init[0]);
									predictions_tmp.push_back(v2 + predictions_init[1]);
									predictions_tmp.push_back(predictions_init[2]);
									if (v3 * step_size + predictions_init[3] < 0 || v3 * step_size + predictions_init[3] > 0.95)
										continue;
									predictions_tmp.push_back(v3 * step_size + predictions_init[3]);
									if (v4 * step_size + predictions_init[4] < 0 || v4 * step_size + predictions_init[4] > 0.95)
										continue;
									predictions_tmp.push_back(v4 * step_size + predictions_init[4]);
									predictions_tmp.push_back(m_t * step_size_m);
									predictions_tmp.push_back(m_b * step_size_m);
									predictions_tmp.push_back(m_l * step_size_m);
									predictions_tmp.push_back(m_r * step_size_m);
									cv::Scalar win_avg_color(0, 0, 255, 0);
									cv::Scalar bg_avg_color(255, 0, 0, 0);
									cv::Mat syn_img = synthesis_opt(predictions_tmp, seg_rgb.size(), win_avg_color, bg_avg_color, false, "../data/tmp.png");
									for (int theta = -4; theta <= 4; theta += 1) {
										for (int scaleX = 0; scaleX <= 0; scaleX += 1) {
											for (int scaleY = 0; scaleY <= 0; scaleY += 1) {
												// Stores affine transformation matrix
												cv::Mat affineTransform(2, 3, CV_32FC1);
												// Initialise translational, rotational and scaling values for transformation
												affineTransform.at<float>(0, 0) = cos(theta * CV_PI / 180) * (1.0 + scaleX * 0.1);
												affineTransform.at<float>(0, 1) = -sin(theta * CV_PI / 180) * (1.0 + scaleY * 0.1);
												affineTransform.at<float>(0, 2) = 0;

												affineTransform.at<float>(1, 0) = sin(theta * CV_PI / 180) * (1.0 + scaleX * 0.1);
												affineTransform.at<float>(1, 1) = cos(theta * CV_PI / 180) * (1.0 + scaleY * 0.1);
												affineTransform.at<float>(1, 2) = 0;
												cv::Mat dst_tmp = cv::Mat(syn_img.size(), CV_8UC3, bg_avg_color);
												warpAffine(syn_img, dst_tmp, affineTransform, syn_img.size(), cv::INTER_LINEAR,
													cv::BORDER_CONSTANT,
													cv::Scalar(255, 0, 0));

												std::vector<double> evaluations = eval_accuracy(dst_tmp, seg_rgb);
												int tmp_blobs = predictions_tmp[0] * predictions_tmp[1];
												double blobs_score = 1 - 1.0 * abs(tmp_blobs - gt_blobs) / gt_blobs;
												//std::cout << "tmp_blobs is " << tmp_blobs << std::endl;
												//std::cout << "blobs_score is " << blobs_score << std::endl;
												double score_tmp = 0.25 * evaluations[0] + 0.25 * evaluations[1] + 0.25 *evaluations[2] + 0.25 * blobs_score;
												//std::cout << "score_tmp is " << score_tmp << std::endl;
												if (score_tmp > score_opt) {
													score_opt = score_tmp;
													for (int index = 0; index < predictions_tmp.size(); index++) {
														predictions_opt[index] = predictions_tmp[index];
													}
													trans_opt[0] = theta;
													trans_opt[1] = scaleX;
													trans_opt[2] = scaleY;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

void opt_with_doors(cv::Mat& seg_rgb, std::vector<double>& predictions_opt, std::vector<double> predictions_init, int step, double range) {
	double score_opt = 0;
	predictions_opt.resize(13);
	int width = seg_rgb.size().width;
	int height = seg_rgb.size().height;
	double FH = height * 1.0 / predictions_init[0];
	double FW = width * 1.0 / predictions_init[1];
	double step_size_w = step / FW;
	double step_size_h = step / FH;
	double range_adjust = range;
	int v4_min = -ceil(range_adjust / step_size_w);
	int v4_max = -v4_min;
	int v5_min = -ceil(range_adjust / step_size_h);
	int v5_max = -v5_min;
	double DFW = width * 1.0 / predictions_init[3];
	double DFH = 1.0 * height ;
	double step_size_dw = step / DFW;
	double step_size_dh = step / DFH;
	int v6_min = -ceil(range_adjust / step_size_dw);
	int v6_max = -v6_min;

	double step_size_m_w = step * 1.0 / width;
	double step_size_m_h = step * 1.0 / height;
	double step_size_m_d = step * 1.0 / height;
	int gt_blobs = blobs(seg_rgb);
	for (int v1 = -1; v1 <= 1; v1++) {
		for (int v2 = -1; v2 <= 1; v2++) {
			for (int v3 = -1; v3 <= 1; v3++) {
				for (int v4 = v4_min; v4 <= v4_max; v4++) {
					for (int v5 = v5_min; v5 <= v5_max; v5++) {
						for (int v6 = v6_min; v6 <= v6_max; v6++) {
							for (int v7 = -2; v7 <= 2; v7++) {
								for (int m_t = 0; m_t <= 2; m_t++) {
									for (int m_b = 0; m_b <= 2; m_b++) {
										for (int m_l = 0; m_l <= 2; m_l++) {
											for (int m_r = 0; m_r <= 2; m_r++) {
												for (int m_d = 1; m_d <= 3; m_d++) {
													std::vector<double> predictions_tmp;
													predictions_tmp.push_back(v1 + predictions_init[0]);
													predictions_tmp.push_back(v2 + predictions_init[1]);
													predictions_tmp.push_back(predictions_init[2]);
													predictions_tmp.push_back(v3 + predictions_init[3]);
													if (v4 * step_size_w + predictions_init[4] < 0 || v4 * step_size_w + predictions_init[4] > 1 - step_size_w)
														continue;
													predictions_tmp.push_back(v4 * step_size_w + predictions_init[4]);
													if (v5 * step_size_h + predictions_init[5] < 0 || v5 * step_size_h + predictions_init[5] > 1 - step_size_h)
														continue;
													predictions_tmp.push_back(v5 * step_size_h + predictions_init[5]);
													if (v6 * step_size_dw + predictions_init[6] < 0 || v6 * step_size_dw + predictions_init[6] > 1 - step_size_dw)
														continue;
													predictions_tmp.push_back(v6 * step_size_dw + predictions_init[6]);
													predictions_tmp.push_back(v7 * step_size_dh + predictions_init[7]);

													predictions_tmp.push_back(m_t * step_size_m_h);
													predictions_tmp.push_back(m_b * step_size_m_h);
													predictions_tmp.push_back(m_l * step_size_m_w);
													predictions_tmp.push_back(m_r * step_size_m_w);
													predictions_tmp.push_back(m_d * step_size_m_d);
													//std::cout << "predictions_tmp size is " << predictions_tmp.size() << std::endl;
													cv::Scalar win_avg_color(0, 0, 255, 0);
													cv::Scalar bg_avg_color(255, 0, 0, 0);
													cv::Mat syn_img = synthesis_opt(predictions_tmp, seg_rgb.size(), win_avg_color, bg_avg_color, false, "../data/tmp.png");
													//cv::imwrite("../data/test.png", seg_rgb);
													int tmp_blobs = predictions_tmp[0] * predictions_tmp[1] + predictions_tmp[3];
													double blobs_score = 1 - 1.0 * abs(tmp_blobs - gt_blobs) / gt_blobs;
													std::vector<double> evaluations = util::eval_accuracy(syn_img, seg_rgb);
													//double score_tmp = 0.25 * evaluations[0] + 0.25 * evaluations[1] + 0.25 *evaluations[2] + 0.25 * blobs_score;
													double score_tmp = 0.34 * evaluations[0] + 0.33 * evaluations[1] + 0.33 *evaluations[2];
													//double score_tmp = 0.5 * evaluations[1] + 0.5 *evaluations[2];
													//std::cout << "score_tmp is " << score_tmp << std::endl;
													if (score_tmp > score_opt) {
														score_opt = score_tmp;
														for (int index = 0; index < predictions_tmp.size(); index++) {
															predictions_opt[index] = predictions_tmp[index];
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

void generate_synFacade(std::string src_image_name, std::vector<double> paras, std::string out_image_name){
	cv::Mat src_img = cv::imread(src_image_name, CV_LOAD_IMAGE_UNCHANGED);
	// default setting
	// range of Rows
	std::pair<int, int> imageRows(2, 10);
	// range of Cols
	std::pair<int, int> imageCols(2, 10);
	// relativeWidth
	std::pair<double, double> imageRelativeWidth(0.3, 0.8);
	// relativeHeight
	std::pair<double, double> imageRelativeHeight(0.3, 0.8);
	int img_rows = paras[0] * (imageRows.second - imageRows.first) + imageRows.first;
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.9)
		img_rows++;
	int img_cols = paras[1] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[1] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.9)
		img_cols++;
	int img_groups = 1;
	double relative_width = paras[2] * (imageRelativeWidth.second - imageRelativeWidth.first) + imageRelativeWidth.first;
	double relative_height = paras[3] * (imageRelativeHeight.second - imageRelativeHeight.first) + imageRelativeHeight.first;
	if (true) {
		std::cout << "img_rows is " << paras[0] * (imageRows.second - imageRows.first) + imageRows.first << std::endl;
		std::cout << "img_cols is " << paras[1] * (imageCols.second - imageCols.first) + imageCols.first << std::endl;
	}
	std::vector<double> results;
	results.push_back(img_rows);
	results.push_back(img_cols);
	results.push_back(img_groups);
	results.push_back(relative_width);
	results.push_back(relative_height);
	cv::Scalar win_avg_color(0, 0, 255, 0);
	cv::Scalar bg_avg_color(255, 0, 0, 0);
	synthesis(results, src_img.size(), "../data/test_sync", win_avg_color, bg_avg_color, true, out_image_name);

}

void findPatches(std::string image_name, std::string output_path, int step) {
	cv::Mat src_img = cv::imread(image_name, CV_LOAD_IMAGE_UNCHANGED);
	std::vector<cv::Rect> rectangles;
	int w_count = src_img.size().width / step;
	int h_count = src_img.size().height / step;
	int start_x = 0.5 * (src_img.size().width - w_count * step);
	int start_y = 0.5 * (src_img.size().height - h_count * step);
	std::cout << "w_count is " << w_count << std::endl;
	std::cout << "h_count is " << h_count << std::endl;
	std::cout << "start_x is " << start_x << std::endl;
	std::cout << "start_y is " << start_y << std::endl;
	for (int i = 0; i < w_count; i++) {
		for (int j = 0; j < h_count; j++) {
			cv::Point center(start_x + i * step + 0.5 * step, start_y + j * step + 0.5 * step);
			std::cout << "center is " << center << std::endl;
			for (int width = step; width <= 3 * step; width += 0.5 * step) {
				for (int height = step; height <= 3 * step; height += 0.5 * step) {
					cv::Point l1(center.x - 0.5 * width, center.y - 0.5 * height);
					cv::Point r1(center.x + 0.5 * width, center.y + 0.5 * height);
					if (l1.x < 0 || l1.y < 0)
						continue;
					if (r1.x > src_img.size().width || r1.y > src_img.size().height)
						continue;
					cv::Rect tmp = cv::Rect(l1, cv::Size2f(width, height));
					rectangles.push_back(tmp);
				}
			}
		}
	}
	// test
	cv::RNG rng(12345);
	cv::Mat drawing = src_img.clone();
	for (int index = 0; index < rectangles.size(); index++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		cv::rectangle(drawing, rectangles[index], color, 1, 8);
	}
	cv::imwrite("../data/patch.png", drawing);
}

void adjust_seg_colors(std::string image_path, std::string output_path) {
	std::vector<std::string> images = get_all_files_names_within_folder(image_path);
	std::cout << "images size is " << images.size() << std::endl;
	for (int i = 0; i < images.size(); i++) {
		std::string img_name = image_path + '/' + images[i];
		cv::Mat src_img = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
		if (src_img.channels() == 4) // ensure there're 3 channels
			cv::cvtColor(src_img, src_img, CV_BGRA2BGR);
		cv::Mat out = src_img.clone();
		for (int h = 0; h < src_img.size().height; h++) {
			for (int w = 0; w < src_img.size().width; w++) {
				if (src_img.at<cv::Vec3b>(h, w)[0] > 200 || src_img.at<cv::Vec3b>(h, w)[1] > 0) {
					out.at<cv::Vec3b>(h, w)[0] = 255;
					out.at<cv::Vec3b>(h, w)[1] = 0;
					out.at<cv::Vec3b>(h, w)[2] = 0;
				}
				else {
					out.at<cv::Vec3b>(h, w)[0] = 0;
					out.at<cv::Vec3b>(h, w)[1] = 0;
					out.at<cv::Vec3b>(h, w)[2] = 255;
				}
			}
		}
		cv::imwrite(output_path + '/' + images[i], out);
	}
}

void conver2seg(std::string image_path, std::string output_path) {
	std::vector<std::string> images = get_all_files_names_within_folder(image_path);
	std::cout << "images size is " << images.size() << std::endl;
	for (int i = 0; i < images.size(); i++) {
		std::string img_name = image_path + '/' + images[i];
		cv::Mat src_img = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
		if (src_img.channels() == 4) // ensure there're 3 channels
			cv::cvtColor(src_img, src_img, CV_BGRA2BGR);
		cv::Mat src_gray;
		cv::cvtColor(src_img, src_gray, CV_BGR2GRAY);
		// only 2 colors
		int color_1 = (int)src_gray.at<uchar>(0, 0);
		int color_2 = 0;
		int bFind = false;
		for (int h = 0; h < src_img.size().height; h++) {
			if (bFind)
				break;
			for (int w = 0; w < src_img.size().width; w++) {
				color_2 = (int)src_gray.at<uchar>(h, w);
				if (color_2 != color_1) {
					bFind = true;
					break;
				}
			}
		}
		bool bFirst = false;
		if (color_2 < color_1)
			bFirst = true;
		cv::Mat out = src_img.clone();
		for (int h = 0; h < src_img.size().height; h++) {
			for (int w = 0; w < src_img.size().width; w++) {
				if (bFirst) {
					if ((int)src_gray.at<uchar>(h, w) == color_1) {
						out.at<cv::Vec3b>(h, w)[0] = 255;
						out.at<cv::Vec3b>(h, w)[1] = 0;
						out.at<cv::Vec3b>(h, w)[2] = 0;
					}
					else {
						out.at<cv::Vec3b>(h, w)[0] = 0;
						out.at<cv::Vec3b>(h, w)[1] = 0;
						out.at<cv::Vec3b>(h, w)[2] = 255;
					}
				}
				else {
					if ((int)src_gray.at<uchar>(h, w) == color_1) {
						out.at<cv::Vec3b>(h, w)[0] = 0;
						out.at<cv::Vec3b>(h, w)[1] = 0;
						out.at<cv::Vec3b>(h, w)[2] = 255;
					}
					else {
						out.at<cv::Vec3b>(h, w)[0] = 255;
						out.at<cv::Vec3b>(h, w)[1] = 0;
						out.at<cv::Vec3b>(h, w)[2] = 0;
					}
				}
			}
		}
		cv::imwrite(output_path + '/' + images[i], out);
	}
}

void split_images(std::string image_path, std::string output_path) {
	cv::Mat src = cv::imread(image_path, CV_LOAD_IMAGE_UNCHANGED);
	std::vector<int> scan_lines;
	scan_lines.push_back(0);
	scan_lines.push_back(90);
	scan_lines.push_back(190);
	scan_lines.push_back(280);
	scan_lines.push_back(src.size().width);
	std::cout << src.size() << std::endl;
	for (int i = 0; i < scan_lines.size() - 1; i++) {
		cv::Mat split_img = src(cv::Rect(scan_lines[i], 0, scan_lines[i + 1] - scan_lines[i], src.size().height));
		cv::imwrite(output_path + "/" + to_string(i) + ".png", split_img);
	}
}

void merge_images(std::string images_path, std::string output_path, int width, int height) {
	std::vector<std::string> images = get_all_files_names_within_folder(images_path);
	cv::Mat matDst(cv::Size(width, height), CV_8UC1);
	int start_x = 0;
	int start_y = 0;
	for (int index = 0; index < images.size(); index++) {
		std::string img_name = images_path + "/" + images[index];
		std::cout << "img_name is " << img_name << std::endl;
		cv::Mat src_img = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
		src_img.copyTo(matDst(cv::Rect(start_x, start_y, src_img.size().width, src_img.size().height)));
		start_x += src_img.size().width;
	}
	cv::imwrite(output_path, matDst);
}

void test_color(std::string image_1_path, std::string image_2_path, std::string output_path) {
	std::vector<std::string> images = get_all_files_names_within_folder(image_1_path);
	std::cout << "images size is " << images.size() << std::endl;
	for (int i = 0; i < images.size(); i++) {
		std::string image_1 = image_1_path + '/' + images[i];
		cv::Mat src_1 = cv::imread(image_1, CV_LOAD_IMAGE_UNCHANGED);
		std::string image_2 = image_2_path + '/' + images[i];
		cv::Mat src_2 = cv::imread(image_2, CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat outpt_img = src_1.clone();
		// write back to json file
		cv::Scalar bg_avg_color(0, 0, 0);
		cv::Scalar win_avg_color(0, 0, 0);
		{
			int bg_count = 0;
			int win_count = 0;
			for (int i = 0; i < src_2.size().height; i++) {
				for (int j = 0; j < src_2.size().width; j++) {
					if ((int)src_2.at<uchar>(i, j) == 0) {
						if (src_1.channels() == 4) {
							win_avg_color.val[0] += src_1.at<cv::Vec4b>(i, j)[0];
							win_avg_color.val[1] += src_1.at<cv::Vec4b>(i, j)[1];
							win_avg_color.val[2] += src_1.at<cv::Vec4b>(i, j)[2];
						}
						if (src_1.channels() == 3) {
							win_avg_color.val[0] += src_1.at<cv::Vec3b>(i, j)[0];
							win_avg_color.val[1] += src_1.at<cv::Vec3b>(i, j)[1];
							win_avg_color.val[2] += src_1.at<cv::Vec3b>(i, j)[2];
						}
						win_count++;
					}
					else {
						if (src_1.channels() == 4) {
							bg_avg_color.val[0] += src_1.at<cv::Vec4b>(i, j)[0];
							bg_avg_color.val[1] += src_1.at<cv::Vec4b>(i, j)[1];
							bg_avg_color.val[2] += src_1.at<cv::Vec4b>(i, j)[2];
						}
						if (src_1.channels() == 3) {
							bg_avg_color.val[0] += src_1.at<cv::Vec3b>(i, j)[0];
							bg_avg_color.val[1] += src_1.at<cv::Vec3b>(i, j)[1];
							bg_avg_color.val[2] += src_1.at<cv::Vec3b>(i, j)[2];
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
		for (int i = 0; i < src_2.size().height; i++) {
			for (int j = 0; j < src_2.size().width; j++) {
				if ((int)src_2.at<uchar>(i, j) == 0) {
					outpt_img.at<cv::Vec3b>(i, j)[0] = win_avg_color.val[0];
					outpt_img.at<cv::Vec3b>(i, j)[1] = win_avg_color.val[1];
					outpt_img.at<cv::Vec3b>(i, j)[2] = win_avg_color.val[2];
				}
				else {
					outpt_img.at<cv::Vec3b>(i, j)[0] = bg_avg_color.val[0];
					outpt_img.at<cv::Vec3b>(i, j)[1] = bg_avg_color.val[1];
					outpt_img.at<cv::Vec3b>(i, j)[2] = bg_avg_color.val[2];
				}
			}
		}
		std::cout << "win_avg_color is " << win_avg_color << std::endl;
		std::cout << "bg_avg_color is " << bg_avg_color << std::endl;
		cv::imwrite(output_path + '/' + images[i], outpt_img);
	}
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
		/*if (best_score > 0.96) {
			cv::imwrite("../data/A/" + images[i], src_img);
		}
		else {
			cv::imwrite("../data/B/" + images[i], src_img);
		}*/
		if (true) {
			//std::cout << out_tensor.slice(1, 0, 2) << std::endl;
			std::cout << img_name << ": " << confidences_tensor.slice(1, 0, 2) << std::endl;
			//std::cout << img_name << ": "<< log(confidences_tensor.slice(1, 1, 2).item<float>()) << std::endl;
			std::cout << "Reject class is " << best_class << std::endl;
		}
	}
}

void test_chip_choose(std::string images_path, std::string output, ModelInfo& mi) {
	std::vector<std::string> facades = get_all_files_names_within_folder(images_path);
	for (int index = 0; index < facades.size(); index++) {
		std::vector<std::string> chip_images = get_all_files_names_within_folder(images_path + "/" + facades[index]);
		double best_score = 10000;
		int best_class = -1;
		for (int i = 0; i < chip_images.size(); i++) {
			std::string img_name = images_path + "/" + facades[index] + '/' + chip_images[i];
			std::cout << img_name << std::endl;
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
			double tmp = log(confidences_tensor.slice(1, 1, 2).item<float>());
			if (tmp < best_score) {
				best_score = tmp;
				best_class = i;
			}
		}
		cv::Mat src_img = cv::imread(images_path + "/" + facades[index] + '/' + chip_images[best_class], CV_LOAD_IMAGE_UNCHANGED);
		cv::imwrite(output + "/" + facades[index] + ".png", src_img);
	}
}

void test_segmentation_model(std::string images_path, ModelInfo& mi) {
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	std::vector<std::string> images = get_all_files_names_within_folder(images_path + "/src");
	for (int index = 0; index < images.size(); index++) {
		std::string img_name = images_path + "/src/" + images[index];
		std::cout << "img_name is " << img_name << std::endl;
		cv::Mat src_img = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
		if (src_img.channels() == 4) // ensure there're 3 channels
			cv::cvtColor(src_img, src_img, CV_BGRA2BGR);
		int run_times = 3;
		// scale to seg size
		cv::Mat scale_img;
		cv::resize(src_img, scale_img, cv::Size(mi.segImageSize[0], mi.segImageSize[1]));
		if (false) {
			cv::imwrite(images_path + "/A/" + images[index], scale_img);
		}
		cv::Mat dnn_img_rgb;
		if (mi.seg_module_type == 0) {
			cv::cvtColor(scale_img, dnn_img_rgb, CV_BGR2RGB);
		}
		else if (mi.seg_module_type == 1) {
			cv::Mat scale_histeq, hsv_src;
			cvtColor(scale_img, hsv_src, cv::COLOR_BGR2HSV);
			std::vector<cv::Mat> bgr;   //destination array
			cv::split(hsv_src, bgr);//split source 
			cv::equalizeHist(bgr[2], bgr[2]);
			cv::merge(bgr, scale_histeq);
			cvtColor(scale_histeq, scale_histeq, cv::COLOR_HSV2BGR);
			cv::cvtColor(scale_histeq, dnn_img_rgb, CV_BGR2RGB);
		}
		else {
			cv::Mat scale_pan, hsv_src;
			cvtColor(scale_img, hsv_src, cv::COLOR_BGR2HSV);
			std::vector<cv::Mat> bgr;   //destination array
			cv::split(hsv_src, bgr);//split source 
			cv::equalizeHist(bgr[2], bgr[2]);
			dnn_img_rgb = bgr[2];
		}
		cv::Mat img_float;
		dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
		int channels = 3;
		if (mi.seg_module_type == 2)
			channels = 1;
		auto img_tensor = torch::from_blob(img_float.data, { 1, (int)mi.segImageSize[0], (int)mi.segImageSize[1], channels }).to(torch::kCUDA);
		img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
		img_tensor[0][0] = img_tensor[0][0].sub(0.5).div(0.5);
		if (mi.seg_module_type != 2) {
			img_tensor[0][1] = img_tensor[0][1].sub(0.5).div(0.5);
			img_tensor[0][2] = img_tensor[0][2].sub(0.5).div(0.5);
		}

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
			torch::Tensor out_tensor;
			if (mi.seg_module_type == 0) {
				out_tensor = mi.seg_module->forward(inputs).toTensor();
			}
			else if (mi.seg_module_type == 1) {
				out_tensor = mi.seg_module_histeq->forward(inputs).toTensor();
			}
			else {
				out_tensor = mi.seg_module_pan->forward(inputs).toTensor();
			}
			out_tensor = out_tensor.squeeze().detach().permute({ 1,2,0 });
			out_tensor = out_tensor.add(1).mul(0.5 * 255).clamp(0, 255).to(torch::kU8);
			//out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
			out_tensor = out_tensor.to(torch::kCPU);
			cv::Mat resultImg((int)mi.segImageSize[0], (int)mi.segImageSize[1], CV_8UC3);
			std::memcpy((void*)resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8)*out_tensor.numel());
			// gray img
			// correct the color
			for (int h = 0; h < resultImg.size().height; h++) {
				for (int w = 0; w < resultImg.size().width; w++) {
					if (resultImg.at<cv::Vec3b>(h, w)[0] > 160)
						color_mark[h][w] += 0;
					else
						color_mark[h][w] += 1;
				}
			}
			if (false) {
				cv::cvtColor(resultImg, resultImg, CV_RGB2BGR);
				cv::imwrite(images_path + "/B/" + images[index], resultImg);
			}
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
		cv::Mat chip_seg;
		cv::resize(gray_img, chip_seg, src_img.size());
		// correct the color
		for (int i = 0; i < chip_seg.size().height; i++) {
			for (int j = 0; j < chip_seg.size().width; j++) {
				//noise
				if ((int)chip_seg.at<uchar>(i, j) < 100) {
					chip_seg.at<uchar>(i, j) = (uchar)0;
				}
				else
					chip_seg.at<uchar>(i, j) = (uchar)255;
			}
		}
		std::string output_img_name = "";
		if(mi.seg_module_type == 0)
			output_img_name = images_path + "/segs_normal/" + images[index];
		else if(mi.seg_module_type == 1)
			output_img_name = images_path + "/segs_histeq/" + images[index];
		else
			output_img_name = images_path + "/segs_pan/" + images[index];
		cv::imwrite(output_img_name, chip_seg);
		// compute color
		cv::Scalar bg_avg_color(0, 0, 0);
		cv::Scalar win_avg_color(0, 0, 0);
		{
			int bg_count = 0;
			int win_count = 0;
			for (int i = 0; i < chip_seg.size().height; i++) {
				for (int j = 0; j < chip_seg.size().width; j++) {
					if ((int)chip_seg.at<uchar>(i, j) == 0) {
						win_avg_color.val[0] += src_img.at<cv::Vec3b>(i, j)[0];
						win_avg_color.val[1] += src_img.at<cv::Vec3b>(i, j)[1];
						win_avg_color.val[2] += src_img.at<cv::Vec3b>(i, j)[2];
						win_count++;
					}
					else {
						bg_avg_color.val[0] += src_img.at<cv::Vec3b>(i, j)[0];
						bg_avg_color.val[1] += src_img.at<cv::Vec3b>(i, j)[1];
						bg_avg_color.val[2] += src_img.at<cv::Vec3b>(i, j)[2];
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


		// add padding
		int padding_size = mi.paddingSize[0];
		int borderType = cv::BORDER_CONSTANT;
		cv::Mat aligned_img_padding;
		cv::copyMakeBorder(chip_seg, aligned_img_padding, padding_size, padding_size, padding_size, padding_size, borderType, bg_color);
		// bbox and color
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(aligned_img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));

		std::vector<float> area_contours(contours.size());
		std::vector<cv::Rect> boundRect(contours.size());
		std::vector<cv::Scalar> colors(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
			area_contours[i] = cv::contourArea(contours[i]);
		}
		cv::Mat dnn_img = cv::Mat(aligned_img_padding.size(), CV_8UC3, bg_avg_color);
		cv::Mat dnn_img_binary = cv::Mat(aligned_img_padding.size(), CV_8UC3, cv::Scalar(255, 0, 0));
		for (int i = 0; i < contours.size(); i++)
		{
			if (hierarchy[i][3] != 0) continue;
			if (area_contours[i] < 15)
				continue;
			//cv::drawContours(dnn_img, contours, i, cv::Scalar(0, 0, 0), 1, 8, hierarchy, 0, cv::Point());
			cv::rectangle(dnn_img_binary, cv::Point(boundRect[i].tl().x + 1, boundRect[i].tl().y + 1), cv::Point(boundRect[i].br().x - 1, boundRect[i].br().y - 1), cv::Scalar(0, 0, 255), -1);
			cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x + 1, boundRect[i].tl().y + 1), cv::Point(boundRect[i].br().x - 1, boundRect[i].br().y - 1), win_avg_color, -1);
		}
		dnn_img = dnn_img(cv::Rect(padding_size, padding_size, src_img.size().width, src_img.size().height));
		dnn_img_binary = dnn_img_binary(cv::Rect(padding_size, padding_size, src_img.size().width, src_img.size().height));
		std::string output_name = images_path + "/segs_color/" + images[index];
		cv::imwrite(output_name, dnn_img);
		output_name = images_path + "/segs_binary/" + images[index];
		cv::imwrite(output_name, dnn_img_binary);
	}
}

cv::Mat pix2pix_seg(cv::Mat& src_img, ModelInfo& mi) {
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	if (src_img.channels() == 4) // ensure there're 3 channels
		cv::cvtColor(src_img, src_img, CV_BGRA2BGR);
	int run_times = 3;
	// scale to seg size
	cv::Mat scale_img;
	cv::resize(src_img, scale_img, cv::Size(mi.segImageSize[0], mi.segImageSize[1]));
	cv::Mat dnn_img_rgb;
	cv::cvtColor(scale_img, dnn_img_rgb, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	int channels = 3;
	if (mi.seg_module_type == 2)
		channels = 1;
	auto img_tensor = torch::from_blob(img_float.data, { 1, (int)mi.segImageSize[0], (int)mi.segImageSize[1], channels }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.5).div(0.5);
	if (mi.seg_module_type != 2) {
		img_tensor[0][1] = img_tensor[0][1].sub(0.5).div(0.5);
		img_tensor[0][2] = img_tensor[0][2].sub(0.5).div(0.5);
	}

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
		torch::Tensor out_tensor;
		out_tensor = mi.seg_module->forward(inputs).toTensor();
		out_tensor = out_tensor.squeeze().detach().permute({ 1,2,0 });
		out_tensor = out_tensor.add(1).mul(0.5 * 255).clamp(0, 255).to(torch::kU8);
		//out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
		out_tensor = out_tensor.to(torch::kCPU);
		cv::Mat resultImg((int)mi.segImageSize[0], (int)mi.segImageSize[1], CV_8UC3);
		std::memcpy((void*)resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8)*out_tensor.numel());
		// gray img
		// correct the color
		for (int h = 0; h < resultImg.size().height; h++) {
			for (int w = 0; w < resultImg.size().width; w++) {
				if (resultImg.at<cv::Vec3b>(h, w)[0] > 160)
					color_mark[h][w] += 0;
				else
					color_mark[h][w] += 1;
			}
		}
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
	cv::Mat chip_seg;
	cv::resize(gray_img, chip_seg, src_img.size());
	// correct the color
	for (int i = 0; i < chip_seg.size().height; i++) {
		for (int j = 0; j < chip_seg.size().width; j++) {
			//noise
			if ((int)chip_seg.at<uchar>(i, j) < 100) {
				chip_seg.at<uchar>(i, j) = (uchar)0;
			}
			else
				chip_seg.at<uchar>(i, j) = (uchar)255;
		}
	}
	// compute color
	cv::Scalar bg_avg_color(0, 0, 0);
	cv::Scalar win_avg_color(0, 0, 0);
	{
		int bg_count = 0;
		int win_count = 0;
		for (int i = 0; i < chip_seg.size().height; i++) {
			for (int j = 0; j < chip_seg.size().width; j++) {
				if ((int)chip_seg.at<uchar>(i, j) == 0) {
					win_avg_color.val[0] += src_img.at<cv::Vec3b>(i, j)[0];
					win_avg_color.val[1] += src_img.at<cv::Vec3b>(i, j)[1];
					win_avg_color.val[2] += src_img.at<cv::Vec3b>(i, j)[2];
					win_count++;
				}
				else {
					bg_avg_color.val[0] += src_img.at<cv::Vec3b>(i, j)[0];
					bg_avg_color.val[1] += src_img.at<cv::Vec3b>(i, j)[1];
					bg_avg_color.val[2] += src_img.at<cv::Vec3b>(i, j)[2];
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


	// add padding
	int padding_size = mi.paddingSize[0];
	int borderType = cv::BORDER_CONSTANT;
	cv::Mat aligned_img_padding;
	cv::copyMakeBorder(chip_seg, aligned_img_padding, padding_size, padding_size, padding_size, padding_size, borderType, bg_color);
	// bbox and color
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(aligned_img_padding, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));

	std::vector<float> area_contours(contours.size());
	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<cv::Scalar> colors(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		area_contours[i] = cv::contourArea(contours[i]);
	}
	cv::Mat dnn_img = cv::Mat(aligned_img_padding.size(), CV_8UC3, bg_avg_color);
	for (int i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		if (area_contours[i] < 15)
			continue;
		cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x + 1, boundRect[i].tl().y + 1), cv::Point(boundRect[i].br().x - 1, boundRect[i].br().y - 1), win_avg_color, -1);
	}
	dnn_img = dnn_img(cv::Rect(padding_size, padding_size, src_img.size().width, src_img.size().height));
	return dnn_img;
}

void test_classifier_model(std::string images_path, ModelInfo& mi, bool bDebug) {
	std::vector<std::string> images = get_all_files_names_within_folder(images_path);
	int num_classes = mi.number_grammars;
	for (int i = 0; i < images.size(); i++) {
		std::string img_name = images_path + '/' + images[i];
		std::cout << "img_name is " << img_name << std::endl;
		cv::Mat dnn_img_rgb = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
		if (dnn_img_rgb.channels() == 4) {// ensure there're 3 channels
			cv::cvtColor(dnn_img_rgb, dnn_img_rgb, CV_BGRA2BGR);
		}
		
		cv::cvtColor(dnn_img_rgb, dnn_img_rgb, CV_BGR2RGB);
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
		if (true)
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
		std::cout << "predictions is " << predictions << std::endl;
	}
}

void img_convert(std::string images_path) {
	std::vector<std::string> images = get_all_files_names_within_folder(images_path);
	std::cout << "images size is " << images.size() << std::endl;
	for (int i = 0; i < images.size(); i++) {
		std::string image_name = images_path + '/' + images[i];
		cv::Mat src_img = cv::imread(image_name, CV_LOAD_IMAGE_UNCHANGED);
		if (src_img.channels() == 4) {// ensure there're 3 channels
			cv::cvtColor(src_img, src_img, CV_BGRA2BGR);
			cv::imwrite(image_name, src_img);
		}
		//cv::Mat src_img = cv::imread("D:/LEGO_meeting_summer_2019/1012/src_facades/backup_v3/B/facade_00061.png", CV_LOAD_IMAGE_UNCHANGED);
		//if (src_img.channels() == 4)
		//	cv::cvtColor(src_img.clone(), src_img, CV_BGRA2BGR);
		//for (int i = 0; i < src_img.size().height; i++) {
		//	for (int j = 0; j < src_img.size().width; j++) {
		//		// wall
		//		if (src_img.at<cv::Vec3b>(i, j)[0] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 0 && src_img.at<cv::Vec3b>(i, j)[2] == 255) {
		//			
		//		}
		//		else
		//		{
		//			src_img.at<cv::Vec3b>(i, j)[0] = 255;
		//			src_img.at<cv::Vec3b>(i, j)[1] = 0;
		//			src_img.at<cv::Vec3b>(i, j)[2] = 0;
		//		}
		//	}
		//}
	}
}

void img_convert(std::string images_path, std::string segs_path) {
	std::vector<std::string> images = get_all_files_names_within_folder(images_path);
	std::cout << "images size is " << images.size() << std::endl;
	for (int i = 0; i < images.size(); i++) {
		std::string image_name = images_path + '/' + images[i];
		cv::Mat src_img = cv::imread(image_name, CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat seg_img = cv::imread(segs_path + '/' + images[i], CV_LOAD_IMAGE_UNCHANGED);
		cv::resize(seg_img, seg_img, src_img.size());
		if (seg_img.channels() == 4) {// ensure there're 3 channels
			cv::cvtColor(seg_img, seg_img, CV_BGRA2BGR);
		}
		cv::imwrite(segs_path + '/' + images[i], seg_img);
	}
}

void test_overlay_images(std::string image_1_path, std::string image_2_path, std::string output_path) {
	std::vector<std::string> images = get_all_files_names_within_folder(image_1_path);
	std::cout << "images size is " << images.size() << std::endl;
	for (int i = 0; i < images.size(); i++) {
		std::string image_1 = image_1_path + '/' + images[i];
		cv::Mat src_1 = cv::imread(image_1, CV_LOAD_IMAGE_UNCHANGED);
		if(src_1.channels() == 3)
			cv::cvtColor(src_1, src_1, CV_BGR2BGRA);
		if (src_1.channels() == 1)
			cv::cvtColor(src_1, src_1, CV_GRAY2BGRA);
		std::string image_2 = image_2_path + '/' + images[i];
		cv::Mat src_2 = cv::imread(image_2, CV_LOAD_IMAGE_UNCHANGED);
		if (src_2.channels() == 3)
			cv::cvtColor(src_2, src_2, CV_BGR2BGRA);
		if (src_2.channels() == 1)
			cv::cvtColor(src_2, src_2, CV_GRAY2BGRA);
		double alpha = 0.8; double beta;
		beta = (1.0 - alpha);
		cv::Mat dst;
		cv::addWeighted(src_1, alpha, src_2, beta, 0.0, dst);
		cv::imwrite(output_path + '/' + images[i], dst);
	}
}

void collect_roi_images(std::string images_path, std::string output_path) {
	std::vector<std::string> images = get_all_files_names_within_folder(images_path);
	std::cout << "images size is " << images.size() << std::endl;
	cv::Scalar bg_color(255, 0, 0); // white back ground
	cv::Scalar fg_color(0, 0, 255); // white back ground
	for (int i = 0; i < images.size(); i++) {
		std::string image_name = images_path + '/' + images[i];
		cv::Mat src_img = cv::imread(image_name, CV_LOAD_IMAGE_UNCHANGED);
		if (src_img.channels() == 4) {// ensure there're 3 channels
			cv::cvtColor(src_img, src_img, CV_BGRA2BGR);
		}
		cv::Mat src_gray(src_img.size(), CV_8UC1);
		for (int h = 0; h < src_img.size().height; h++) {
			for (int w = 0; w < src_img.size().width; w++) {
				if (src_img.at<cv::Vec3b>(h, w)[0] > 160) {
					src_gray.at<uchar>(h, w) = (uchar)0;
				}
				else {
					src_gray.at<uchar>(h, w) = (uchar)255;
				}
			}
		}
		// find contours
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(src_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		std::cout << "contours.size() is " << contours.size() << std::endl;
		std::vector<cv::Rect> boundRect(contours.size());
		std::vector<int> roi_x_tl;
		std::vector<int> roi_y_tl;
		std::vector<int> roi_x_br;
		std::vector<int> roi_y_br;
		for (int i = 0; i < contours.size(); i++)
		{
			boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
			roi_x_tl.push_back(boundRect[i].tl().x);
			roi_y_tl.push_back(boundRect[i].tl().y);
			roi_x_br.push_back(boundRect[i].br().x);
			roi_y_br.push_back(boundRect[i].br().y);
		}
		//
		cv::Mat dnn_img = cv::Mat(src_gray.size(), CV_8UC3, bg_color);
		int min_tl_x = *std::min_element(roi_x_tl.begin(), roi_x_tl.end());
		int min_tl_y = *std::min_element(roi_y_tl.begin(), roi_y_tl.end());
		int max_br_x = *std::max_element(roi_x_br.begin(), roi_x_br.end());
		int max_br_y = *std::max_element(roi_y_br.begin(), roi_y_br.end());
		cv::rectangle(dnn_img, cv::Point(min_tl_x, min_tl_y), cv::Point(max_br_x, max_br_y), fg_color, -1);
		cv::imwrite(output_path + "/" + images[i], dnn_img);
		
	}
}

void test_old_segmentation(std::string images_path, std::string output_path) {
	std::vector<std::string> images = get_all_files_names_within_folder(images_path);
	cv::Scalar bg_color(255, 0, 0); // white back ground
	cv::Scalar fg_color(0, 0, 255); // white back ground
	for (int i = 0; i < images.size(); i++) {
		std::string image_name = images_path + '/' + images[i];
		cv::Mat src_img = cv::imread(image_name, CV_LOAD_IMAGE_UNCHANGED);
		if (src_img.channels() == 4) {// ensure there're 3 channels
			cv::cvtColor(src_img, src_img, CV_BGRA2BGR);
		}
		// load image
		cv::Mat hsv, dst_ehist, dst_classify;
		cvtColor(src_img, hsv, cv::COLOR_BGR2HSV);
		std::vector<cv::Mat> bgr;   //destination array
		cv::split(hsv, bgr);//split source 
		for (int i = 0; i < 3; i++)
			cv::equalizeHist(bgr[i], bgr[i]);
		dst_ehist = bgr[2];
		dst_classify = facade_clustering_kkmeans(dst_ehist, 2);
		cv::imwrite(output_path + "/" + images[i], dst_classify);
	}
}

void test_spacing(std::string images_path, ModelInfo& mi, bool bDebug) {
	std::vector<std::string> images = get_all_files_names_within_folder(images_path);
	std::cout << "images size is " << images.size() << std::endl;
	for (int i = 0; i < images.size(); i++) {
		std::string img_name = images_path + '/' + images[i];
		std::cout << "img_name is " << img_name << std::endl;
		cv::Mat src_img = cv::imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
		if (src_img.size().width < 25)
			continue;
		std::vector<int> separation_x;
		std::vector<int> separation_y;
		find_spacing(src_img, separation_x, separation_y, bDebug);
		if (separation_x.size() > 0) {
			// compute spacing
			std::vector<int> space_x;
			for (int i = 0; i < separation_x.size(); i += 2) {
				space_x.push_back(separation_x[i + 1] - separation_x[i]);
			}
			int max_spacing_x_id =std::max_element(space_x.begin(), space_x.end()) - space_x.begin();
			double ratio_x = space_x[max_spacing_x_id] * 1.0 / src_img.size().width;
			if (ratio_x > 0.24)
			{
				if (space_x.size() >= 2) {
					float average_spacing = accumulate(space_x.begin(), space_x.end(), 0.0) / space_x.size();
					if (abs(space_x[max_spacing_x_id] - average_spacing) / src_img.size().width < 0.05)
						continue;
				}
				// split into two images
				cv::Mat img_1 = src_img(cv::Rect(0, 0, separation_x[max_spacing_x_id * 2], src_img.size().height));
				cv::Mat img_2 = src_img(cv::Rect(separation_x[max_spacing_x_id * 2 + 1], 0, src_img.size().width - separation_x[max_spacing_x_id * 2 + 1], src_img.size().height));
				if(img_1.size().width > img_2.size().width)
					cv::imwrite("../data/output_spacing/" + images[i], img_1);
				else
					cv::imwrite("../data/output_spacing/" + images[i], img_2);
			}
		}

	}
}

double get_image_quality_score(cv::Mat src_img, ModelInfo& mi) {
	// size of nn image size
	std::vector<double> defaultImgSize = mi.defaultSize;
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
	if (best_class == 1)
		return 0;
	else {
		return best_score;
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
	// segmentation model type
	mi.seg_module_type = util::readNumber(docModel, "seg_model_type", 0);
	if (mi.seg_module_type == 1) {
		// load segmentation_pan model
		std::string seg_model_pan = util::readStringValue(docModel, "seg_model_pan");
		// load segmentation model
		mi.seg_module_pan = torch::jit::load(seg_model_pan);
		mi.seg_module_pan->to(at::kCUDA);
		assert(mi.seg_module_pan != nullptr);
	}
	else if (mi.seg_module_type == 2) {
		// load segmentation_histeq model
		std::string seg_model_histeq = util::readStringValue(docModel, "seg_model_histeq");
		// load segmentation model
		mi.seg_module_histeq = torch::jit::load(seg_model_histeq);
		mi.seg_module_histeq->to(at::kCUDA);
		assert(mi.seg_module_histeq != nullptr);
	}
	else{
		std::string seg_model = util::readStringValue(docModel, "seg_model");
		// load segmentation model
		mi.seg_module = torch::jit::load(seg_model);
		mi.seg_module->to(at::kCUDA);
		assert(mi.seg_module != nullptr);
	}
	//
	mi.debug = util::readBoolValue(docModel, "debug", false);
	mi.bOpt = util::readBoolValue(docModel, "opt", false);
	mi.opt_step = util::readNumber(docModel, "opt_step", 2);
	mi.opt_range = util::readNumber(docModel, "opt_range", 0.2);
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
	mi.facadesSegFolder = util::readStringValue(docModel, "facadesSegFolder");
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
	if (best_class == 1 || best_score < 0.90) // bad facades
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

bool chipping(FacadeInfo& fi, ModelInfo& mi, ChipInfo &chip, bool bMultipleChips, bool bDebug, std::string img_filename) {
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
	std::vector<ChipInfo> cropped_chips;
	if(!bground)
		cropped_chips = crop_chip_no_ground(src_facade.clone(), type, facadeSize, targetSize, bMultipleChips);
	else
		cropped_chips = crop_chip_ground(src_facade.clone(), type, facadeSize, targetSize, bMultipleChips);

	int best_chip_id = choose_best_chip(cropped_chips, mi, bDebug, img_filename);
	// adjust the best chip
	cv::Mat croppedImage = cropped_chips[best_chip_id].src_image.clone(); 
	cv::Mat chip_seg;
	{
		apply_segmentation_model(croppedImage, chip_seg, mi, bDebug, img_filename);
		std::vector<int> boundaries = adjust_chip(chip_seg.clone());
		chip_seg = chip_seg(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
		croppedImage = croppedImage(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
	}
	//pre_process(chip_seg, croppedImage, mi, bDebug, img_filename);
	
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
	// copy info to the chip
	chip.src_image = croppedImage.clone();
	chip.seg_image = chip_seg.clone();
	if (bDebug) {
		/*for (int i = 0; i < cropped_chips.size(); i++) {
			cv::imwrite("../data/confidences/" + std::to_string(i) + '_' + img_filename, cropped_chips[i]);
		}*/
		std::cout << "Facade type is " << type << std::endl;
		cv::imwrite(mi.facadesFolder + "/" + img_filename, src_facade);
	}
	return true;
}

void pre_process(cv::Mat &chip_seg, cv::Mat& croppedImage, ModelInfo& mi, bool bDebug, std::string img_filename) {
	// --- step 1
	apply_segmentation_model(croppedImage, chip_seg, mi, bDebug, img_filename);
	cv::Mat scale_img;
	cv::resize(chip_seg, scale_img, cv::Size(mi.defaultSize[0], mi.defaultSize[1]));
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
	int dilation_type = cv::MORPH_RECT;
	cv::Mat img_dilation;
	int kernel_size = 3;
	cv::Mat element = cv::getStructuringElement(dilation_type, cv::Size(kernel_size, kernel_size), cv::Point(kernel_size / 2, kernel_size / 2));
	/// Apply the dilation operation
	cv::dilate(scale_img, img_dilation, element);
	cv::resize(img_dilation, img_dilation, chip_seg.size());
	// correct the color
	for (int i = 0; i < img_dilation.size().height; i++) {
		for (int j = 0; j < img_dilation.size().width; j++) {
			//noise
			if ((int)img_dilation.at<uchar>(i, j) < 128) {
				img_dilation.at<uchar>(i, j) = (uchar)0;
			}
			else
				img_dilation.at<uchar>(i, j) = (uchar)255;
		}
	}
	int padding_size = mi.paddingSize[0];
	int borderType = cv::BORDER_CONSTANT;
	cv::Mat padding_img;
	cv::copyMakeBorder(img_dilation, padding_img, padding_size, padding_size, padding_size, padding_size, borderType, cv::Scalar(255, 255, 255));
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(padding_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<float> area_contours(contours.size());
	std::vector<cv::Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		area_contours[i] = cv::contourArea(contours[i]);
	}
	std::sort(area_contours.begin(), area_contours.end());
	float target_contour_area = area_contours[area_contours.size() / 2];
	for (int i = 0; i < contours.size(); i++)
	{
		float area_contour = cv::contourArea(contours[i]);
		if (area_contour < 0.5 * target_contour_area) {
			cv::rectangle(padding_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), cv::Scalar(255, 255, 255), -1);
		}
	}
	chip_seg = padding_img(cv::Rect(padding_size, padding_size, chip_seg.size().width, chip_seg.size().height));
	// --- step 2
	std::vector<int> boundaries = adjust_chip(chip_seg.clone());
	chip_seg = chip_seg(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
	croppedImage = croppedImage(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
	// ---- step 3
	if (chip_seg.size().width < 25)
		return;
	std::vector<int> separation_x;
	std::vector<int> separation_y;
	cv::Mat spacing_img = chip_seg.clone();
	float threshold_spacing = 0.238;
	find_spacing(spacing_img, separation_x, separation_y, bDebug); 
	if(separation_x.size() % 2 != 0)
		return;
	if (separation_x.size() > 0) {
		// compute spacing
		std::vector<int> space_x;
		for (int i = 0; i < separation_x.size(); i += 2) {
			space_x.push_back(separation_x[i + 1] - separation_x[i]);
		}
		int max_spacing_x_id = std::max_element(space_x.begin(), space_x.end()) - space_x.begin();
		double ratio_x = space_x[max_spacing_x_id] * 1.0 / spacing_img.size().width;
		if (bDebug)
			std::cout << "ratio_x is " << ratio_x << std::endl;
		if (ratio_x > threshold_spacing)
		{
			if (space_x.size() >= 2) {
				float average_spacing = accumulate(space_x.begin(), space_x.end(), 0.0) / space_x.size();
				if (abs(space_x[max_spacing_x_id] - average_spacing) / spacing_img.size().width < 0.05)
					return;
			}
			// split into two images
			cv::Mat img_1_seg = spacing_img(cv::Rect(0, 0, separation_x[max_spacing_x_id * 2], spacing_img.size().height));
			cv::Mat img_2_seg = spacing_img(cv::Rect(separation_x[max_spacing_x_id * 2 + 1], 0, spacing_img.size().width - separation_x[max_spacing_x_id * 2 + 1], spacing_img.size().height));
			cv::Mat chip_spacing;
			if (img_1_seg.size().width > img_2_seg.size().width) {
				chip_spacing = croppedImage(cv::Rect(0, 0, separation_x[max_spacing_x_id * 2], spacing_img.size().height));
				// clean up
				std::vector<int> boundaries = adjust_chip(img_1_seg.clone());
				img_1_seg = img_1_seg(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
				chip_spacing = chip_spacing(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
				chip_seg = img_1_seg.clone();
				croppedImage = chip_spacing.clone();
			}				
			else {
				chip_spacing = croppedImage(cv::Rect(separation_x[max_spacing_x_id * 2 + 1], 0, spacing_img.size().width - separation_x[max_spacing_x_id * 2 + 1], spacing_img.size().height));
				// clearn up
				std::vector<int> boundaries = adjust_chip(img_2_seg.clone());
				img_2_seg = img_2_seg(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
				chip_spacing = chip_spacing(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
				chip_seg = img_2_seg.clone();
				croppedImage = chip_spacing.clone();
			}
		}
	}
}

std::vector<ChipInfo> crop_chip_no_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips) {
	std::vector<ChipInfo> cropped_chips;
	double ratio_upper = 0.95;
	double ratio_lower = 0.05;
	double ratio_step = 0.05;
	double target_width = targetSize[0];
	double target_height = targetSize[1];
	if (type == 1) {
		ChipInfo chip;
		chip.src_image = src_facade;
		chip.x = 0;
		chip.y = 0;
		chip.width = src_facade.size().width;
		chip.height = src_facade.size().height;
		cropped_chips.push_back(chip);
	}
	else if (type == 2) {
		double target_ratio_width = target_width / facadeSize[0];
		double target_ratio_height = target_height / facadeSize[1];
		if (target_ratio_height > 1.0)
			target_ratio_height = 1.0;
		if (facadeSize[0] < 1.6 * target_width || !bMultipleChips) {
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			// crop target size
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = src_facade.size().width * start_width_ratio;
			chip.y = 0;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
		else {
			// push back multiple chips
			int index = 1;
			double start_width_ratio = index * ratio_lower; // not too left
			std::vector<double> confidences;
			while (start_width_ratio + target_ratio_width < ratio_upper) { // not too right
				// get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));;
				chip.x = src_facade.size().width * start_width_ratio;
				chip.y = 0;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_width_ratio = index * ratio_step;
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
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(0, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = 0;
			chip.y = src_facade.size().height * start_height_ratio;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
		else {
			// push back multiple chips
			int index = 1;
			double start_height_ratio = index * ratio_lower;
			while (start_height_ratio + target_ratio_height < ratio_upper) {
				// get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(0, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				chip.x = 0;
				chip.y = src_facade.size().height * start_height_ratio;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_height_ratio = index * ratio_step;
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
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = src_facade.size().width * start_width_ratio;
			chip.y = src_facade.size().height * start_height_ratio;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
		else if (bLonger_width) {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 1;
			double start_width_ratio = index * ratio_lower;
			double start_height_ratio = (1 - target_ratio_height) * 0.5;
			while (start_width_ratio + target_ratio_width < ratio_upper) {
				// get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				chip.x = src_facade.size().width * start_width_ratio;
				chip.y = src_facade.size().height * start_height_ratio;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_width_ratio = index * ratio_step;
			}
		}
		else {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 1;
			double start_height_ratio = index * ratio_lower;
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			while (start_height_ratio + target_ratio_height < ratio_upper) {
				// get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				chip.x = src_facade.size().width * start_width_ratio;
				chip.y = src_facade.size().height * start_height_ratio;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_height_ratio = index * ratio_step;
			}
		}
	}
	else {
		// do nothing
	}
	return cropped_chips;
}

std::vector<ChipInfo> crop_chip_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips) {
	double ratio_upper = 0.95;
	double ratio_lower = 0.05;
	double ratio_step = 0.05;
	std::vector<ChipInfo> cropped_chips;
	double target_width = targetSize[0];
	double target_height = targetSize[1];
	if (type == 1) {
		ChipInfo chip;
		chip.src_image = src_facade;
		chip.x = 0;
		chip.y = 0;
		chip.width = src_facade.size().width;
		chip.height = src_facade.size().height;
		cropped_chips.push_back(chip);
	}
	else if (type == 2) {
		double target_ratio_width = target_width / facadeSize[0];
		double target_ratio_height = target_height / facadeSize[1];
		if (target_ratio_height > 1.0)
			target_ratio_height = 1.0;
		if (facadeSize[0] < 1.6 * target_width || !bMultipleChips) {
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			// crop target size
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = src_facade.size().width * start_width_ratio;
			chip.y = 0;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
		else {
			// push back multiple chips
			int index = 1;
			double start_width_ratio = index * ratio_lower; // not too left
			std::vector<double> confidences;
			while (start_width_ratio + target_ratio_width < ratio_upper) { // not too right
																   // get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, 0, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				chip.x = src_facade.size().width * start_width_ratio;
				chip.y = 0;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_width_ratio = index * ratio_step;
			}
		}
	}
	else if (type == 3) {
		double target_ratio_height = target_height / facadeSize[1];
		double target_ratio_width = target_width / facadeSize[0];
		if (target_ratio_width >= 1.0)
			target_ratio_width = 1.0;
		double start_height_ratio = (1 - target_ratio_height);
		ChipInfo chip;
		chip.src_image = src_facade(cv::Rect(0, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
		chip.x = 0;
		chip.y = src_facade.size().height * start_height_ratio;
		chip.width = src_facade.size().width * target_ratio_width;
		chip.height = src_facade.size().height * target_ratio_height;
		cropped_chips.push_back(chip);
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
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = src_facade.size().width * start_width_ratio;
			chip.y = src_facade.size().height * start_height_ratio;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
		else if (bLonger_width) {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 1;
			double start_width_ratio = index * ratio_lower;
			double start_height_ratio = (1 - target_ratio_height);
			while (start_width_ratio + target_ratio_width < ratio_upper) {
				// get the cropped img
				ChipInfo chip;
				chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
				chip.x = src_facade.size().width * start_width_ratio;
				chip.y = src_facade.size().height * start_height_ratio;
				chip.width = src_facade.size().width * target_ratio_width;
				chip.height = src_facade.size().height * target_ratio_height;
				cropped_chips.push_back(chip);
				index++;
				start_width_ratio = index * ratio_step;
			}
		}
		else {
			// check multiple chips and choose the one that has the highest confidence value
			int index = 1;
			double start_height_ratio = (1 - target_ratio_height);
			double start_width_ratio = (1 - target_ratio_width) * 0.5;
			ChipInfo chip;
			chip.src_image = src_facade(cv::Rect(src_facade.size().width * start_width_ratio, src_facade.size().height * start_height_ratio, src_facade.size().width * target_ratio_width, src_facade.size().height * target_ratio_height));
			chip.x = src_facade.size().width * start_width_ratio;
			chip.y = src_facade.size().height * start_height_ratio;
			chip.width = src_facade.size().width * target_ratio_width;
			chip.height = src_facade.size().height * target_ratio_height;
			cropped_chips.push_back(chip);
		}
	}
	else {
		// do nothing
	}
	return cropped_chips;
}

std::vector<double> compute_chip_info(ChipInfo chip, ModelInfo& mi, bool bDebug, std::string img_filename) {
	std::vector<double> chip_info;
	apply_segmentation_model(chip.src_image.clone(), chip.seg_image, mi, false, img_filename);
	//adjust boundaries
	std::vector<int> boundaries = adjust_chip(chip.seg_image.clone());
	chip.seg_image = chip.seg_image(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
	chip.src_image = chip.src_image(cv::Rect(boundaries[2], boundaries[0], boundaries[3] - boundaries[2] + 1, boundaries[1] - boundaries[0] + 1));
	process_chip(chip, mi, false, img_filename);
	// go to grammar classifier
	int num_classes = mi.number_grammars;
	cv::Mat dnn_img_rgb;
	cv::cvtColor(chip.dnnIn_image.clone(), dnn_img_rgb, CV_BGR2RGB);
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
		cv::imwrite(img_filename, chip.src_image);
	}
	return chip_info;
}

int choose_best_chip(std::vector<ChipInfo> chips, ModelInfo& mi, bool bDebug, std::string img_filename) {
	int best_chip_id = 0;
	if (chips.size() == 1)
		best_chip_id = 0;
	else {
		std::vector<double> confidence_values;
		std::string path = "../data/chips/" + img_filename.substr(0, img_filename.find(".png"));
		if (bDebug) {
			CreateDirectory(path.c_str(), NULL);
		}
		confidence_values.resize(chips.size());
		// method 1
		if (false) {
			for (int i = 0; i < chips.size(); i++) {
				std::string filename = path + "/chip_" + to_string(i) + ".png";
				confidence_values[i] = compute_chip_info(chips[i], mi, bDebug, filename)[0];
				// try to use reject model to get image quality score
				if (bDebug) {
					std::cout << "chip " << i << " score is " << confidence_values[i] << std::endl;
				}
			}
			// find the best chip id
			best_chip_id = std::max_element(confidence_values.begin(), confidence_values.end()) - confidence_values.begin();
		}
		// method 2
		if (true) {
			for (int i = 0; i < chips.size(); i++) {
				if (bDebug) {
					std::string filename = path + "/chip_" + to_string(i) + ".png";
					cv::imwrite(filename, chips[i].src_image);
				}
				cv::Mat src_img = chips[i].src_image.clone();
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
				confidence_values[i] = log(confidences_tensor.slice(1, 1, 2).item<float>());
			}
			best_chip_id = std::min_element(confidence_values.begin(), confidence_values.end()) - confidence_values.begin();
		}

		if (bDebug) {
			std::cout << "best_chip_id is " << best_chip_id << std::endl;
			std::string filename = "../data/best_chip/" + img_filename;
			cv::imwrite(filename, chips[best_chip_id].src_image);
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
	double thre_upper = 1.1; // don't apply here
	double thre_lower = 0.1;
	// top 
	int pos_top = 0;
	int black_pixels = 0;
	for (int i = 0; i < chip.size().height; i++) {
		black_pixels = 0;
		for (int j = 0; j < chip.size().width; j++) {
			if ((int)chip.at<uchar>(i, j) == 0) {
				black_pixels++;
			}
		}
		double ratio = black_pixels * 1.0 / chip.size().width;
		if (ratio < thre_upper && ratio > thre_lower) {
			pos_top = i;
			break;
		}
	}

	// bottom 
	black_pixels = 0;
	int pos_bot = 0;
	for (int i = chip.size().height - 1; i >= 0; i--) {
		black_pixels = 0;
		for (int j = 0; j < chip.size().width; j++) {
			//noise
			if ((int)chip.at<uchar>(i, j) == 0) {
				black_pixels++;
			}
		}
		double ratio = black_pixels * 1.0 / chip.size().width;
		if (ratio < thre_upper && ratio > thre_lower) {
			pos_bot = i;
			break;
		}
	}

	// left
	black_pixels = 0;
	int pos_left = 0;
	for (int i = 0; i < chip.size().width; i++) {
		black_pixels = 0;
		for (int j = 0; j < chip.size().height; j++) {
			//noise
			if ((int)chip.at<uchar>(j, i) == 0) {
				black_pixels++;
			}
		}
		double ratio = black_pixels * 1.0 / chip.size().height;
		if (ratio < thre_upper && ratio > thre_lower) {
			pos_left = i;
			break;
		}
	}
	// right
	black_pixels = 0;
	int pos_right = 0;
	for (int i = chip.size().width - 1; i >= 0; i--) {
		black_pixels = 0;
		for (int j = 0; j < chip.size().height; j++) {
			//noise
			if ((int)chip.at<uchar>(j, i) == 0) {
				black_pixels++;
			}
		}
		double ratio = black_pixels * 1.0 / chip.size().height;
		if (ratio < thre_upper && ratio > thre_lower) {
			pos_right = i;
			break;
		}
	}
	boundaries[0] = pos_top;
	boundaries[1] = pos_bot;
	boundaries[2] = pos_left;
	boundaries[3] = pos_right;
	return boundaries;
}

void find_spacing(cv::Mat src_img, std::vector<int> &separation_x, std::vector<int> &separation_y, bool bDebug) {
	if (src_img.channels() == 4 ) {
		separation_x.clear();
		separation_y.clear();
		return;
	}
	if (src_img.channels() == 3) {
		cv::cvtColor(src_img, src_img, CV_BGR2GRAY);
	}
	// horizontal 
	bool bSpacing_pre = false;
	bool bSpacing_curr = false;
	for (int i = 0; i < src_img.size().width; i++) {
		bSpacing_curr = true;
		for (int j = 0; j < src_img.size().height; j++) {
			//noise
			if (src_img.channels() == 1) {
				if ((int)src_img.at<uchar>(j, i) == 0) {
					bSpacing_curr = false;
					break;
				}
			}
			else {
				if (src_img.at<cv::Vec3b>(j, i)[0] == 0 && src_img.at<cv::Vec3b>(j, i)[1] == 0 && src_img.at<cv::Vec3b>(j, i)[1] == 0) {
					bSpacing_curr = false;
					break;
				}
			}
		}
		if (bSpacing_pre != bSpacing_curr) {
			separation_x.push_back(i);
		}
		bSpacing_pre = bSpacing_curr;
	}

	bSpacing_pre = false;
	bSpacing_curr = false;
	int spacing_y = -1;
	for (int i = 0; i < src_img.size().height; i++) {
		bSpacing_curr = true;
		for (int j = 0; j < src_img.size().width; j++) {
			if (src_img.channels() == 1) {
				if ((int)src_img.at<uchar>(i, j) == 0) {
					bSpacing_curr = false;
					break;
				}
			}
			else {
				if (src_img.at<cv::Vec3b>(i, j)[0] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 0) {
					bSpacing_curr = false;
					break;
				}
			}
		}
		if (bSpacing_pre != bSpacing_curr) {
			separation_y.push_back(i);
		}
		bSpacing_pre = bSpacing_curr;
	}
	if (bDebug) {
		std::cout << "separation_x is " << separation_x << std::endl;
		std::cout << "separation_y is " << separation_y << std::endl;
	}
	return;
}

void apply_segmentation_model(cv::Mat &croppedImage, cv::Mat &chip_seg, ModelInfo& mi, bool bDebug, std::string img_filename) {
	int run_times = 3;
	cv::Mat src_img = croppedImage.clone();
	// scale to seg size
	cv::resize(src_img, src_img, cv::Size(mi.segImageSize[0], mi.segImageSize[1]));
	cv::Mat dnn_img_rgb;
	if (mi.seg_module_type == 0) {
		cv::cvtColor(src_img, dnn_img_rgb, CV_BGR2RGB);
	}
	else if (mi.seg_module_type == 1) {
		cv::Mat scale_histeq, hsv_src;
		cvtColor(src_img, hsv_src, cv::COLOR_BGR2HSV);
		std::vector<cv::Mat> bgr;   //destination array
		cv::split(hsv_src, bgr);//split source 
		cv::equalizeHist(bgr[2], bgr[2]);
		cv::merge(bgr, scale_histeq);
		cvtColor(scale_histeq, scale_histeq, cv::COLOR_HSV2BGR);
		cv::cvtColor(scale_histeq, dnn_img_rgb, CV_BGR2RGB);
	}
	else {
		cv::Mat hsv_src;
		cvtColor(src_img, hsv_src, cv::COLOR_BGR2HSV);
		std::vector<cv::Mat> bgr;   //destination array
		cv::split(hsv_src, bgr);//split source 
		cv::equalizeHist(bgr[2], bgr[2]);
		dnn_img_rgb = bgr[2];
	}
	cv::Mat img_float;
	dnn_img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
	int channels = 3;
	if (mi.seg_module_type == 2)
		channels = 1;
	auto img_tensor = torch::from_blob(img_float.data, { 1, (int)mi.segImageSize[0], (int)mi.segImageSize[1], channels }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.5).div(0.5);
	if (channels == 3) {
		img_tensor[0][1] = img_tensor[0][1].sub(0.5).div(0.5);
		img_tensor[0][2] = img_tensor[0][2].sub(0.5).div(0.5);
	}

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
		torch::Tensor out_tensor;
		if (mi.seg_module_type == 0) {
			out_tensor = mi.seg_module->forward(inputs).toTensor();
		}
		else if (mi.seg_module_type == 1) {
			out_tensor = mi.seg_module_histeq->forward(inputs).toTensor();
		}
		else {
			out_tensor = mi.seg_module_pan->forward(inputs).toTensor();
		}
		out_tensor = out_tensor.squeeze().detach().permute({ 1,2,0 });
		out_tensor = out_tensor.add(1).mul(0.5 * 255).clamp(0, 255).to(torch::kU8);
		//out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
		out_tensor = out_tensor.to(torch::kCPU);
		cv::Mat resultImg((int)mi.segImageSize[0], (int)mi.segImageSize[1], CV_8UC3);
		std::memcpy((void*)resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8)*out_tensor.numel());
		// gray img
		// correct the color
		for (int h = 0; h < resultImg.size().height; h++) {
			for (int w = 0; w < resultImg.size().width; w++) {
				if (resultImg.at<cv::Vec3b>(h, w)[0] > 160)
					color_mark[h][w] += 0;
				else
					color_mark[h][w] += 1;
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
}

bool process_chip(ChipInfo &chip, ModelInfo& mi, bool bDebug, std::string img_filename) {
	// default size for NN
	int width = mi.defaultSize[0] - 2 * mi.paddingSize[0];
	int height = mi.defaultSize[1] - 2 * mi.paddingSize[1];
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(chip.seg_image, scale_img, cv::Size(width, height));
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
	std::vector<std::vector<cv::Rect>> largestRect(contours.size());
	std::vector<bool> bIntersectionbbox(contours.size());
	std::vector<float> area_contours(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		area_contours[i] = cv::contourArea(contours[i]);
		bIntersectionbbox[i] = false;
	}
	std::sort(area_contours.begin(), area_contours.end());
	float target_contour_area = area_contours[area_contours.size() / 2];
	// find the largest rectangles
	cv::Mat drawing(aligned_img_padding.size(), CV_8UC3, bg_color);
	for (int i = 0; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		cv::Mat tmp(aligned_img_padding.size(), CV_8UC3, window_color);
		drawContours(tmp, contours, i, bg_color, -1, 8, hierarchy, 0, cv::Point());
		cv::Mat tmp_gray;
		cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
		cv::Rect tmp_rect = findLargestRectangle(tmp_gray);
		largestRect[i].push_back(tmp_rect);
		float area_contour = cv::contourArea(contours[i]);
		float area_rect = 0;
		area_rect += tmp_rect.width * tmp_rect.height;
		float ratio = area_rect / area_contour;
		while (ratio < 0.60) { // find more largest rectangles in the rest area
							   // clear up the previous rectangles
			tmp_gray.empty();
			cv::rectangle(tmp, cv::Point(tmp_rect.tl().x, tmp_rect.tl().y), cv::Point(tmp_rect.br().x, tmp_rect.br().y), window_color, -1);
			cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
			tmp_rect = findLargestRectangle(tmp_gray);
			area_rect += tmp_rect.width * tmp_rect.height;
			if (tmp_rect.width * tmp_rect.height > 100)
				largestRect[i].push_back(tmp_rect);
			ratio = area_rect / area_contour;
		}
	}
	// check intersection
	for (int i = 0; i < contours.size(); i++) {
		if (hierarchy[i][3] != 0 || bIntersectionbbox[i]) {
			bIntersectionbbox[i] = true;
			continue;
		}
		for (int j = i + 1; j < contours.size(); j++) {
			if (findIntersection(boundRect[i], boundRect[j])) {
				bIntersectionbbox[i] = true;
				bIntersectionbbox[j] = true;
				break;
			}
		}
	}
	//
	cv::Mat dnn_img = cv::Mat(aligned_img_padding.size(), CV_8UC3, bg_color);
	for (int i = 1; i< contours.size(); i++)
	{
		if (hierarchy[i][3] != 0) continue;
		// check the validity of the rect
		//float area_contour = cv::contourArea(contours[i]);
		//float area_rect = boundRect[i].width * boundRect[i].height;
		//if (area_contour < 0.5 * target_contour_area) continue;
		//float ratio = area_contour / area_rect;
		//if (!bIntersectionbbox[i] /*&& (ratio > 0.60 || area_contour < 160)*/) {
		//	cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
		//}
		//else {
		//	for (int j = 0; j < 1; j++)
		//		cv::rectangle(dnn_img, cv::Point(largestRect[i][j].tl().x, largestRect[i][j].tl().y), cv::Point(largestRect[i][j].br().x, largestRect[i][j].br().y), window_color, -1);
		//}
		cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x, boundRect[i].tl().y), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
	}
	chip.dilation_dst = dilation_dst;
	chip.aligned_img = aligned_img;
	chip.dnnIn_image = dnn_img;
	if (bDebug) {
		cv::imwrite(mi.chipsFolder + "/" + img_filename, chip.src_image);
		cv::imwrite(mi.segsFolder + "/" + img_filename, chip.seg_image);
		cv::imwrite(mi.dilatesFolder + "/" + img_filename, chip.dilation_dst);
		cv::imwrite(mi.alignsFolder + "/" + img_filename, chip.aligned_img);
		cv::imwrite(mi.dnnsInFolder + "/" + img_filename, chip.dnnIn_image);
	}
	return true;
}

bool post_process_chip(ChipInfo &chip, ModelInfo& mi, bool bDebug, std::string img_filename) {
	return true;
}

std::vector<double> feedDnn(ChipInfo &chip, FacadeInfo& fi, ModelInfo& mi, bool bDebug, std::string img_filename) {
	int num_classes = mi.number_grammars;
	// 
	std::vector<int> separation_x;
	std::vector<int> separation_y;
	cv::Mat spacing_img = chip.dnnIn_image.clone();
	find_spacing(spacing_img, separation_x, separation_y, bDebug);
	int spacing_r = separation_y.size() / 2;
	int spacing_c = separation_x.size() / 2;

	cv::Mat dnn_img_rgb;
	cv::cvtColor(chip.dnnIn_image.clone(), dnn_img_rgb, CV_BGR2RGB);
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
	if (best_class % 2 == 0) {
		if (abs(predictions[0] + 1 - spacing_r) <= 1 && predictions[0] > 1)
			predictions[0] = spacing_r - 1;
	}
	else {
		if (abs(predictions[0] - spacing_r) <= 1 && predictions[0] > 1)
			predictions[0] = spacing_r;
	}
	if (abs(predictions[1] - spacing_c) <= 1 && predictions[1] > 1)
		predictions[1] = spacing_c;
	// opt
	double score_opt = 0;
	bool bOpt = mi.bOpt;
	// generate red + blue seg img
	cv::Mat seg_src = chip.seg_image.clone();
	cv::Mat seg_rgb = cv::Mat(seg_src.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	for (int i = 0; i < seg_src.size().height; i++) {
		for (int j = 0; j < seg_src.size().width; j++) {
			//noise
			if ((int)seg_src.at<uchar>(i, j) == 0) {
				seg_rgb.at<cv::Vec3b>(i, j)[0] = 0;
				seg_rgb.at<cv::Vec3b>(i, j)[1] = 0;
				seg_rgb.at<cv::Vec3b>(i, j)[2] = 255;
			}
			else {
				seg_rgb.at<cv::Vec3b>(i, j)[0] = 255;
				seg_rgb.at<cv::Vec3b>(i, j)[1] = 0;
				seg_rgb.at<cv::Vec3b>(i, j)[2] = 0;
			}
		}
	}
	std::vector<double> predictions_opt;
	if (predictions.size() == 5 && bOpt) {
		opt_without_doors(seg_rgb, predictions_opt, predictions, mi.opt_step, mi.opt_range);
	}
	if (predictions.size() == 8 && bOpt) {
		opt_with_doors(seg_rgb, predictions_opt, predictions, mi.opt_step, mi.opt_range);
	}
	if (bOpt) {
		for (int i = 0; i < predictions.size(); i++)
			predictions[i] = predictions_opt[i];
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

cv::Mat synthesis(std::vector<double> predictions, cv::Size src_size, std::string dnnsOut_folder, cv::Scalar win_avg_color, cv::Scalar bg_avg_color, bool bDebug, std::string img_filename){
	cv::Mat syn_img;
	if (predictions.size() == 5) {
		int img_rows = predictions[0];
		int img_cols = predictions[1];
		int img_groups = predictions[2];
		double relative_width = predictions[3];
		double relative_height = predictions[4];
		syn_img = util::generateFacadeSynImage(224, 224, img_rows, img_cols, img_groups, relative_width, relative_height);
		std::cout << "img_rows is " << img_rows << std::endl;
		std::cout << "img_cols is " << img_cols << std::endl;
		std::cout << "img_groups is " << img_groups << std::endl;
		std::cout << "relative_width is " << relative_width << std::endl;
		std::cout << "relative_height is " << relative_height << std::endl;
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
	if (predictions.size() == 9) {
		int img_rows = predictions[0];
		int img_cols = predictions[1];
		int img_groups = predictions[2];
		double relative_width = predictions[3];
		double relative_height = predictions[4];
		double margin_t = predictions[5];
		double margin_b = predictions[6];
		double margin_l = predictions[7];
		double margin_r = predictions[8];
		syn_img = util::generateFacadeSynImage_new(224, 224, img_rows, img_cols, img_groups, relative_width, relative_height, margin_t, margin_b, margin_l, margin_r);
		/*std::cout << "img_rows is " << img_rows << std::endl;
		std::cout << "img_cols is " << img_cols << std::endl;
		std::cout << "img_groups is " << img_groups << std::endl;
		std::cout << "relative_width is " << relative_width << std::endl;
		std::cout << "relative_height is " << relative_height << std::endl;*/
	}
	if (predictions.size() == 13) {
		int img_rows = predictions[0];
		int img_cols = predictions[1];
		int img_groups = predictions[2];
		int img_doors = predictions[3];
		double relative_width = predictions[4];
		double relative_height = predictions[5];
		double relative_door_width = predictions[6];
		double relative_door_height = predictions[7];
		double margin_t = predictions[8];
		double margin_b = predictions[9];
		double margin_l = predictions[10];
		double margin_r = predictions[11];
		double margin_d = predictions[12];
		syn_img = util::generateFacadeSynImage_new(224, 224, img_rows, img_cols, img_groups, img_doors, relative_width, relative_height, relative_door_width, relative_door_height, margin_t, margin_b, margin_l, margin_r, margin_d);
		/*std::cout << "img_rows is " << img_rows << std::endl;
		std::cout << "img_cols is " << img_cols << std::endl;
		std::cout << "img_groups is " << img_groups << std::endl;
		std::cout << "relative_width is " << relative_width << std::endl;
		std::cout << "relative_height is " << relative_height << std::endl;*/
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
	if(bDebug)
		cv::imwrite(dnnsOut_folder + "/" + img_filename, syn_img);
	return syn_img;
}

cv::Mat synthesis_opt(std::vector<double> predictions, cv::Size src_size, cv::Scalar win_color, cv::Scalar bg_color, bool bDebug, std::string img_filename) {
	int height = src_size.height;
	int width = src_size.width;
	int thickness = -1;
	cv::Mat syn_img(height, width, CV_8UC3, bg_color);
	if (predictions.size() == 9) {
		int NR = predictions[0];
		int NC = predictions[1];
		int NG = predictions[2];
		double ratioWidth = predictions[3];
		double ratioHeight = predictions[4];
		double margin_t = predictions[5];
		double margin_b = predictions[6];
		double margin_l = predictions[7];
		double margin_r = predictions[8];
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
		// draw facade image
		for (int i = 0; i < NR; ++i) {
			for (int j = 0; j < NC; ++j) {
				float x1 = FW * j + margin_l * width;
				float y1 = FH * i + margin_t * height;
				float x2 = x1 + WW;
				float y2 = y1 + WH;
				cv::rectangle(syn_img, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), win_color, thickness);
			}
		}
	}
	if (predictions.size() == 13) {
		int NR = predictions[0];
		int NC = predictions[1];
		int NG = predictions[2];
		int ND = predictions[3];
		double ratioWidth = predictions[4];
		double ratioHeight = predictions[5];
		double ratioDWidth = predictions[6];
		double ratioDHeight = predictions[7];
		double margin_t = predictions[8];
		double margin_b = predictions[9];
		double margin_l = predictions[10];
		double margin_r = predictions[11];
		double margin_d = predictions[12];

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
				cv::rectangle(syn_img, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), win_color, thickness);
			}
		}
		// doors
		for (int i = 0; i < ND; i++) {
			float x1 = DFW * i + margin_l * width;
			float y1 = height - DH - margin_b * height;
			float x2 = x1 + DW;
			float y2 = y1 + DH;
			cv::rectangle(syn_img, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), win_color, thickness);
		}

	}
	if (bDebug)
		cv::imwrite(img_filename, syn_img);
	return syn_img;
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
	cv::Mat aligned_img = cleanAlignedImage(drawing, 0.05);
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

// Returns the largest rectangle inscribed within regions of all non-zero pixels
cv::Rect findLargestRectangle(cv::Mat image) {
	assert(image.channels() == 1);
	cv::Mat mask = (image > 0) / 255;
	mask.convertTo(mask, CV_16S);

	// Get the largest area rectangle under a histogram
	auto maxHist = [](cv::Mat hist) -> cv::Rect {
		// Append -1 to both ends
		cv::copyMakeBorder(hist, hist, 0, 0, 1, 1, cv::BORDER_CONSTANT, cv::Scalar::all(-1));
		cv::Rect maxRect(-1, -1, 0, 0);

		// Initialize stack to first element
		std::stack<int> colStack;
		colStack.push(0);

		// Iterate over columns
		for (int c = 0; c < hist.cols; c++) {
			// Ensure stack is only increasing
			while (hist.at<int16_t>(c) < hist.at<int16_t>(colStack.top())) {
				// Pop larger element
				int h = hist.at<int16_t>(colStack.top()); colStack.pop();
				// Get largest rect at popped height using nearest smaller element on both sides
				cv::Rect rect(colStack.top(), 0, c - colStack.top() - 1, h);
				// Update best rect
				if (rect.area() > maxRect.area())
					maxRect = rect;
			}
			// Push this column
			colStack.push(c);
		}
		return maxRect;
	};

	cv::Rect maxRect(-1, -1, 0, 0);
	cv::Mat height = cv::Mat::zeros(1, mask.cols, CV_16SC1);
	for (int r = 0; r < mask.rows; r++) {
		// Extract a single row
		cv::Mat row = mask.row(r);
		// Get height of unbroken non-zero values per column
		height = (height + row);
		height.setTo(0, row == 0);

		// Get largest rectangle from this row up
		cv::Rect rect = maxHist(height);
		if (rect.area() > maxRect.area()) {
			maxRect = rect;
			maxRect.y = r - maxRect.height + 1;
		}
	}

	return maxRect;
}

bool insideRect(cv::Rect a1, cv::Point p) {
	bool bresult = false;
	if (p.x >= a1.tl().x && p.x <= a1.br().x && p.y >= a1.tl().y && p.y <= a1.br().y)
		bresult = true;
	return bresult;
}

bool findIntersection(cv::Rect a1, cv::Rect a2) {
	// a2 insection with a1 
	if (insideRect(a1, a2.tl()))
		return true;
	if (insideRect(a1, cv::Point(a2.tl().x, a2.br().y)))
		return true;
	if (insideRect(a1, a2.br()))
		return true;
	if (insideRect(a1, cv::Point(a2.br().x, a2.tl().y)))
		return true;
	// a1 insection with a2
	if (insideRect(a2, a1.tl()))
		return true;
	if (insideRect(a2, cv::Point(a1.tl().x, a1.br().y)))
		return true;
	if (insideRect(a2, a1.br()))
		return true;
	if (insideRect(a2, cv::Point(a1.br().x, a1.tl().y)))
		return true;
	return false;
}

cv::Mat facade_clustering_kkmeans(cv::Mat src_img, int clusters) {
	// Here we declare that our samples will be 2 dimensional column vectors.  
	// (Note that if you don't know the dimensionality of your vectors at compile time
	// you can change the 2 to a 0 and then set the size at runtime)
	typedef matrix<double, 0, 1> sample_type;
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
			if (src_img.channels() == 4) {
				m(0) = src_img.at<cv::Vec4b>(i, j)[0] * 1.0 / 255;
				m(1) = src_img.at<cv::Vec4b>(i, j)[1] * 1.0 / 255;
				m(2) = src_img.at<cv::Vec4b>(i, j)[2] * 1.0 / 255;
			}
			else if (src_img.channels() == 3) {
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
	if (src_img.channels() == 4) {
		count = 0;
		for (int i = 0; i < src_img.size().height; i++) {
			for (int j = 0; j < src_img.size().width; j++) {
				clusters_colors[test(samples[count])][0] += src_img.at<cv::Vec4b>(i, j)[0];
				clusters_colors[test(samples[count])][1] += src_img.at<cv::Vec4b>(i, j)[1];
				clusters_colors[test(samples[count])][2] += src_img.at<cv::Vec4b>(i, j)[2];
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
		//std::cout << "clusters_colors " << i << " is " << clusters_colors[i] << std::endl;
		if (src_img.channels() == 3 || src_img.channels() == 4) {
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
	else if (src_img.channels() == 3) {
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
	else {
		for (int i = 0; i < out_img.size().height; i++) {
			for (int j = 0; j < out_img.size().width; j++) {
				if (test(samples[count]) == darkest_cluster) {
					out_img.at<cv::Vec4b>(i, j)[0] = 0;
					out_img.at<cv::Vec4b>(i, j)[1] = 0;
					out_img.at<cv::Vec4b>(i, j)[2] = 0;
				}
				else {
					out_img.at<cv::Vec4b>(i, j)[0] = 255;
					out_img.at<cv::Vec4b>(i, j)[1] = 255;
					out_img.at<cv::Vec4b>(i, j)[2] = 255;

				}
				count++;
			}
		}
	}
	return out_img;
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
	if (paras[0] * (imageRows.second - imageRows.first) + imageRows.first - img_rows > 0.8)
		img_rows++;
	int img_cols = paras[1] * (imageCols.second - imageCols.first) + imageCols.first;
	if (paras[1] * (imageCols.second - imageCols.first) + imageCols.first - img_cols > 0.8)
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

std::vector<double> grammar3(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Cols
	std::pair<int, int> imageCols(mi.grammars[2].rangeOfCols[0], mi.grammars[2].rangeOfCols[1]);
	// relativeWidth
	std::pair<double, double> imageRelativeWidth(mi.grammars[2].relativeWidth[0], mi.grammars[2].relativeWidth[1]);
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

std::vector<double> grammar5(ModelInfo& mi, std::vector<double> paras, bool bDebug) {
	// range of Rows
	std::pair<int, int> imageRows(mi.grammars[4].rangeOfRows[0], mi.grammars[4].rangeOfRows[1]);
	// relativeHeight
	std::pair<double, double> imageRelativeHeight(mi.grammars[4].relativeHeight[0], mi.grammars[4].relativeHeight[1]);
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

std::vector<double> eval_accuracy_old(const cv::Mat& seg_img, const cv::Mat& gt_img) {
	int gt_p = 0;
	int seg_tp = 0;
	int seg_fn = 0;
	int gt_n = 0;
	int seg_tn = 0;
	int seg_fp = 0;

	assert(seg_img.channels() == 3 && gt_img.channels() == 3);
	// Convert seg_img and gt_img into masks
	cv::Mat seg_r(seg_img.size(), CV_8UC1);
	cv::Mat seg_b(seg_img.size(), CV_8UC1);
	cv::Mat gt_r(seg_img.size(), CV_8UC1);
	cv::Mat gt_b(seg_img.size(), CV_8UC1);
	cv::mixChannels(
		std::vector<cv::Mat>{ seg_img, gt_img },
		std::vector<cv::Mat>{ seg_b, seg_r, gt_b, gt_r },
		std::vector<int>{ 0, 0, 2, 1, 3, 2, 5, 3 });
	cv::Mat seg = (seg_b == 0) & (seg_r == 255);
	cv::Mat gt = (gt_b == 0) & (gt_r == 255);

	gt_p = cv::countNonZero(gt);
	//gt_n = cv::countNonZero(~gt);
	gt_n = gt.size().width * gt.size().height - gt_p;
	seg_tp = cv::countNonZero(gt & seg);
	//seg_fn = cv::countNonZero(gt & ~seg);
	seg_fn = gt_p - seg_tp;
	seg_tn = cv::countNonZero(~gt & ~seg);
	//seg_fp = cv::countNonZero(~gt & seg);
	seg_fp = gt_n - seg_tn;

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