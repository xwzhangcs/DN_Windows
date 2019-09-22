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
	for (int i = 0; i < facades_folders.size(); i++) {
		std::string facade_labeled_file = gt_images_path + "/" + facades_folders[i] + "/label.png";
		std::cout << "labeled_img is " << facade_labeled_file << std::endl;
		std::string seg_file = seg_images_path + facades_folders[i] + ".png";
		std::cout << "seg_img is " << seg_file << std::endl;
		std::vector<double> seg_results = eval_segmented_gt(seg_file, facade_labeled_file);
		out_param << facades_folders[i];
		out_param << ",";
		out_param << seg_results[0];
		out_param << ",";
		out_param << seg_results[1];
		out_param << "\n";
	}
}

std::vector<double> FacadeSeg::eval_segmented_gt(std::string seg_img_file, std::string gt_img_file) {
	cv::Mat seg_img = cv::imread(seg_img_file, CV_LOAD_IMAGE_ANYCOLOR);
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
				if ((int)seg_img.at<uchar>(i, j) == 255) {
					seg_wall_tp++;
				}
			}
			else {// non-wall
				gt_non_wall_num++;
				if ((int)seg_img.at<uchar>(i, j) == 0 ) {
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