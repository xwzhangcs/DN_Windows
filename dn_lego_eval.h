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
    int const max_BINARY_value = 255;
	void facade_clustering_kkmeans(std::string in_img_file, std::string seg_img_file, std::string color_img_file, int clusters);
	void facade_clustering_spectral(std::string in_img_file, std::string seg_img_file, std::string color_img_file, int clusters);
	void eval_dataset_postprocessing(std::string label_img);
	std::vector<double> eval_segmented_gt(std::string seg_img_file, std::string gt_img_file);

	void eval_different_segs(std::string result_file);
	void create_different_segs();

	/// Function header
	std::vector<std::string> get_all_files_names_within_folder(std::string folder);
	int find_threshold(cv::Mat src, bool bground);
};