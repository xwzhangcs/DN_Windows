#include "dn_lego_eval.h"
#include "Utils.h"

void FacadeSeg::eval() {
	/*std::string chips_path("../data/chips");
	std::vector<std::string> chipFiles = get_all_files_names_within_folder(chips_path);
	for (int i = 0; i < chipFiles.size(); i++) {
		std::string img_file_name = chipFiles[i];
		std::string chip_img_file = chips_path + "/" + img_file_name;
		std::string segOut_file_name = "../data/eval_seg/" + img_file_name;
		std::string dnnOut_file_name = "../data/eval_dnn/" + img_file_name;
		generateSegOutAndDnnOut(chip_img_file, argv[3], segOut_file_name, dnnOut_file_name, true);
	}*/
	std::string path("../data/eval_dataset");
	std::vector<std::string> facades_folders = get_all_files_names_within_folder(path);
	std::ofstream out_param("../data/eval_results.txt", std::ios::app);
	out_param << "facade_id";
	out_param << ",";
	out_param << "seg_pix_accuracy";
	out_param << ",";
	out_param << "seg_wall_accuracy";
	out_param << ",";
	out_param << "seg_wins_accuracy";
	out_param << ",";
	out_param << "dnn_pix_accuracy";
	out_param << ",";
	out_param << "dnn_wall_accuracy";
	out_param << ",";
	out_param << "dnn_wins_accuracy";
	out_param << "\n";
	for (int i = 0; i < facades_folders.size(); i++) {
		std::string facade_labeled_file = path + "/" + facades_folders[i] + "/label.png";
		std::cout << "labeled_img is " << facade_labeled_file << std::endl;
		std::string seg_file = "../data/eval_seg/" + facades_folders[i] + ".png";
		std::string dnn_file = "../data/eval_dnn/" + facades_folders[i] + ".png";
		std::cout << "seg_img is " << seg_file << std::endl;
		std::cout << "dnn_img is " << dnn_file << std::endl;
		std::vector<double> seg_results = eval_segmented_gt(seg_file, facade_labeled_file);
		std::vector<double> dnn_results = eval_dnn_gt(dnn_file, facade_labeled_file);
		out_param << facades_folders[i];
		out_param << ",";
		out_param << seg_results[0];
		out_param << ",";
		out_param << seg_results[1];
		out_param << ",";
		out_param << seg_results[2];
		out_param << ",";
		out_param << dnn_results[0];
		out_param << ",";
		out_param << dnn_results[1];
		out_param << ",";
		out_param << dnn_results[2];
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

std::vector<double> FacadeSeg::eval_dnn_gt(std::string dnn_img_file, std::string gt_img_file) {
	cv::Mat dnn_img = cv::imread(dnn_img_file);
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
				if (dnn_img.at<cv::Vec3b>(i, j)[0] == 255 && dnn_img.at<cv::Vec3b>(i, j)[1] == 255 && dnn_img.at<cv::Vec3b>(i, j)[2] == 255) {
					seg_wall_tp++;
				}
			}
			else {// non-wall
				gt_non_wall_num++;
				if (dnn_img.at<cv::Vec3b>(i, j)[0] == 0 && dnn_img.at<cv::Vec3b>(i, j)[1] == 0 && dnn_img.at<cv::Vec3b>(i, j)[2] == 0) {
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
	return eval_metrix;
}


void FacadeSeg::eval_dataset_postprocessing(std::string label_img) {
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

void FacadeSeg::facade_clustering_kkmeans(std::string in_img_file, std::string seg_img_file, std::string color_img_file, int clusters) {
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

void FacadeSeg::facade_clustering_spectral(std::string in_img_file, std::string seg_img_file, std::string color_img_file, int clusters) {
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

void FacadeSeg::create_different_segs() {
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

int FacadeSeg::find_threshold(cv::Mat src, bool bground) {
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
		//std::cout << "percentage is " << percentage << std::endl;
		if (percentage > 0.25 && !bground)
			return threshold;
		if (percentage > 0.25 && bground)
			return threshold;
	}
}

void FacadeSeg::eval_different_segs(std::string result_file) {

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