#include "dn_lego_eval.h"
#include "Utils.h"

void FacadeSeg::eval(std::string seg_images_path, std::string gt_images_path, std::string results_txt) {
	std::vector<std::string> facades_folders = get_all_files_names_within_folder(gt_images_path);
	std::ofstream out_param(results_txt, std::ios::app);
	out_param << "facade_id";
	out_param << ",";
	out_param << "pAccuracy";
	out_param << ",";
	out_param << "mIou";
	out_param << "\n";
	for (int i = 0; i < 3/*facades_folders.size()*/; i++) {
		std::string facade_labeled_file = gt_images_path + "/" + facades_folders[i];
		std::cout << "labeled_img is " << facade_labeled_file << std::endl;
		std::string seg_file = seg_images_path + "/" + facades_folders[i];
		std::cout << "seg_img is " << seg_file << std::endl;
		std::vector<double> seg_results = eval_accuracy(seg_file, facade_labeled_file);
		//std::vector<int> seg_pattern_results = find_spacing(seg_file);
		std::vector<int> gt_pattern_results = find_spacing(facade_labeled_file);
		out_param << facades_folders[i];
		out_param << ",";
		out_param << seg_results[0];
		out_param << ",";
		out_param << seg_results[1];
		out_param << "\n";
	}
}

std::vector<double> FacadeSeg::eval_accuracy(std::string seg_img_file, std::string gt_img_file) {
	cv::Mat seg_img = cv::imread(seg_img_file, CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat gt_img = cv::imread(gt_img_file, CV_LOAD_IMAGE_UNCHANGED);
	int gt_wall_num = 0;
	int seg_wall_tp = 0;
	int seg_wall_fn = 0;
	int gt_non_wall_num = 0;
	int seg_non_wall_tp = 0;
	int seg_non_wall_fn = 0;
	for (int i = 0; i < gt_img.size().height; i++) {
		for (int j = 0; j < gt_img.size().width; j++) {
			// wall
			if (gt_img.at<cv::Vec3b>(i, j)[0] == 0 && gt_img.at<cv::Vec3b>(i, j)[1] == 0 && gt_img.at<cv::Vec3b>(i, j)[2] == 255) {
				gt_wall_num++;
				if (seg_img.at<cv::Vec3b>(i, j)[0] == 0 && seg_img.at<cv::Vec3b>(i, j)[1] == 0 && seg_img.at<cv::Vec3b>(i, j)[2] == 255) {
					seg_wall_tp++;
				}
				else
					seg_wall_fn++;
			}
			else {// non-wall
				gt_non_wall_num++;
				if (seg_img.at<cv::Vec3b>(i, j)[0] == 255 && seg_img.at<cv::Vec3b>(i, j)[1] == 0 && seg_img.at<cv::Vec3b>(i, j)[2] == 0) {
					seg_non_wall_tp++;
				}
				else
					seg_non_wall_fn++;
			}
		}
	}
	// return pixel accuracy and class accuracy
	std::vector<double> eval_metrix;
	// pixel accuracy 
	eval_metrix.push_back(1.0 * (seg_wall_tp + seg_non_wall_tp) / (gt_wall_num + gt_non_wall_num));
	// mIou
	double wall_iou = 1.0 * seg_wall_tp / (gt_wall_num + seg_non_wall_fn);
	double non_wall_iou = 1.0 * seg_non_wall_tp / (gt_non_wall_num + seg_wall_fn);
	eval_metrix.push_back(0.5 * (wall_iou + non_wall_iou));
	std::cout << "gt_wall_num is " << gt_wall_num << std::endl;
	std::cout << "gt_non_wall_num is " << gt_non_wall_num << std::endl;
	std::cout << "seg_wall_tp is " << seg_wall_tp << std::endl;
	std::cout << "seg_wall_fn is " << seg_wall_fn << std::endl;
	std::cout << "seg_non_wall_tp is " << seg_non_wall_tp << std::endl;
	std::cout << "seg_non_wall_fn is " << seg_non_wall_fn << std::endl;
	return eval_metrix;
}

std::vector<double> FacadeSeg::eval_pattern(std::string seg_img_file, std::string gt_img_file) {
	cv::Mat seg_img = cv::imread(seg_img_file, CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat gt_img = cv::imread(gt_img_file, CV_LOAD_IMAGE_UNCHANGED);
	int gt_row_num = 0;
	int seg_row_tp = 0;
	int gt_col_num = 0;
	int seg_col_num = 0;
	
	// return pixel accuracy and class accuracy
	std::vector<double> eval_metrix;
	return eval_metrix;
}

std::vector<int> FacadeSeg::find_spacing(std::string img_file) {
	cv::Mat src_img = cv::imread(img_file, CV_LOAD_IMAGE_UNCHANGED);
	if (src_img.channels() == 4) {
		cv::cvtColor(src_img, src_img, CV_BGRA2BGR);
	}
	int rows = 0;
	int cols = 0;
	// horizontal 
	bool bSpacing_pre = false;
	bool bSpacing_curr = false;
	for (int i = 0; i < src_img.size().width; i++) {
		bSpacing_curr = false;
		for (int j = 0; j < src_img.size().height; j++) {
			if (src_img.at<cv::Vec3b>(j, i)[0] == 0 && src_img.at<cv::Vec3b>(j, i)[1] == 0 && src_img.at<cv::Vec3b>(j, i)[2] == 255) {
				bSpacing_curr = true;
				break;
			}
		}
		if (bSpacing_pre != bSpacing_curr) {
			rows++;
		}
		bSpacing_pre = bSpacing_curr;
	}

	bSpacing_pre = false;
	bSpacing_curr = false;
	int spacing_y = -1;
	for (int i = 0; i < src_img.size().height; i++) {
		bSpacing_curr = false;
		for (int j = 0; j < src_img.size().width; j++) {
			if (src_img.at<cv::Vec3b>(i, j)[0] == 0 && src_img.at<cv::Vec3b>(i, j)[1] == 0 && src_img.at<cv::Vec3b>(i, j)[2] == 255) {
				bSpacing_curr = true;
				break;
			}
		}
		if (bSpacing_pre != bSpacing_curr) {
			cols++;
		}
		bSpacing_pre = bSpacing_curr;
	}
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
	/*cv::imwrite("../data/test.png", src_gray);
	/// Draw contours
	cv::RNG rng(12345);
	cv::Mat drawing = cv::Mat::zeros(padding_img.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point());
	}
	cv::imwrite("../data/contour.png", drawing);*/

	std::cout << "rows is " << rows << std::endl;
	std::cout << "cols is " << cols << std::endl;
	std::cout << "# winodws is " << contours.size() - 1 << std::endl;
	std::vector<int> results;
	results.push_back(rows / 2);
	results.push_back(cols / 2);
	results.push_back(contours.size() - 1);
	return results;
}

std::vector<std::string> FacadeSeg::get_all_files_names_within_folder(std::string folder)
{
	std::vector<std::string> names;
	std::string search_path = folder + "/*.*";
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