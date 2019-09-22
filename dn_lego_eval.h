#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <Windows.h>

#include <dlib/clustering.h>
#include <dlib/rand.h>

using namespace dlib;
using namespace std;
class FacadeSeg
{
public:
	std::vector<double> eval_segmented_gt(std::string seg_img_file, std::string gt_img_file);
	std::vector<std::string> get_all_files_names_within_folder(std::string folder);
	void eval(std::string seg_images_path, std::string gt_images_path, std::string results_txt);
};