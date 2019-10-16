#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <Windows.h>

#include <dlib/clustering.h>
#include <dlib/rand.h>

using namespace dlib;

// Holds information about a facade 
struct FacadeInfo {
	std::string imgName;          // Facade texture (ROI of Building::atlasImg) 
	std::vector<double> facadeSize;// Width, height of inscribed rectangle (rUTM)
	bool ground;                // Whether facade touches ground 
	bool roof;                  // Whether facade is a roof 
	float score;                // Score of the facade image 
	bool valid;                 // Whether parameters are valid 
	float good_conf;			// confidence value for the good facade
	int grammar;                // Procedural grammar to use 
	std::vector<double> conf;  // Grammar confidence values
	std::vector<double> chip_size;// Width, height of selected chip (rUTM) 
	std::vector<int> bg_color;         // Color of facade background 
	std::vector<int> win_color;        // Color of windows and doors 
	int rows;                   // Number of rows of windows per chip 
	int cols;                   // Number of columns of windows per chip 
	int grouping;               // Number of windows per group 
	float relativeWidth;        // Window width wrt cell (0, 1) 
	float relativeHeight;       // Window height wrt cell (0, 1) 
	int doors;                  // Number of doors per chip 
	float relativeDWidth;       // Width of doors wrt cell (0, 1) 
	float relativeDHeight;      // Height of doors wrt facade chip (0, 1) 
};

// Hold information about Grammars
struct Grammar {
	std::shared_ptr<torch::jit::script::Module> grammar_model;
	int number_paras;
	int grammar_id;
	std::vector<double> rangeOfRows;
	std::vector<double> rangeOfCols;
	std::vector<double> rangeOfGrouping;
	std::vector<double> rangeOfDoors;
	std::vector<double> relativeWidth;
	std::vector<double> relativeHeight;
	std::vector<double> relativeDWidth;
	std::vector<double> relativeDHeight;
};

// Holds information about NNs
struct ModelInfo {
	std::string facadesFolder;
	std::string facadesSegFolder;
	std::string invalidfacadesFolder;
	std::string chipsFolder;
	std::string segsFolder;
	std::string dnnsInFolder;
	std::string dnnsOutFolder;
	std::string dilatesFolder;
	std::string alignsFolder;
	bool debug;
	std::vector<double> defaultSize;
	std::vector<double> paddingSize;
	std::vector<double> targetChipSize;
	std::vector<double> segImageSize;
	Grammar grammars[6];
	std::shared_ptr<torch::jit::script::Module> classifier_module;
	int number_grammars;
	std::shared_ptr<torch::jit::script::Module> reject_classifier_module;
	std::shared_ptr<torch::jit::script::Module> seg_module;
	std::shared_ptr<torch::jit::script::Module> seg_module_pan;
	std::shared_ptr<torch::jit::script::Module> seg_module_histeq;
	int seg_module_type;
	bool bOpt;
	int opt_step;
	double opt_range;
};

// Hold chip info
struct ChipInfo {
	cv::Mat src_image;
	cv::Mat seg_image;
	cv::Mat dnnIn_image;
	cv::Mat dilation_dst;
	cv::Mat aligned_img;
	int x; // Rect x
	int y; // Rect y
	int width; // Rect width
	int height; // Rect height
};

/**** test functions *****/
void test_rejection_model(std::string images_path, ModelInfo& mi);
void test_chip_choose(std::string images_path, std::string output, ModelInfo& mi);
void test_segmentation_model(std::string images_path, ModelInfo& mi);
void test_classifier_model(std::string images_path, ModelInfo& mi, bool bDebug);
void test_overlay_images(std::string image_1_path, std::string image_2_path, std::string output_path);
void collect_roi_images(std::string images_path, std::string output_path);
void test_spacing(std::string images_path, ModelInfo& mi, bool bDebug);
void test_color(std::string image_1_path, std::string image_2_path, std::string output_path);
cv::Mat facade_clustering_kkmeans(cv::Mat src_img, int clusters);
void test_old_segmentation(std::string images_path, std::string output_path);
void split_images(std::string image_path, std::string output_path);
void merge_images(std::string images_path, std::string output_path, int width, int height);
void adjust_seg_colors(std::string image_path, std::string output_path);
void conver2seg(std::string image_path, std::string output_path);
void findPatches(std::string image_name, std::string output_path, int step);
void test_seg2grammars(ModelInfo& mi, std::string image_name, std::string output_path);
void generate_synFacade(std::string src_image_name, std::vector<double> paras, std::string out_image_name);
void test_affine_transformation(std::string image_path, std::string output_path);
cv::Mat pix2pix_seg(cv::Mat& src_img, ModelInfo& mi);
int blobs(cv::Mat& src_img);
void img_convert(std::string images_path);
void img_convert(std::string images_path, std::string segs_path);
void eval_seg_models(std::string images_path, std::string output_path, std::string model_path, int segImageSize, std::string results_txt);
void test_ensemble_models(std::string images_path, std::string output_path, std::string model_imgs_path, std::string results_txt);

/**** helper functions *****/
std::vector<std::string> get_all_files_names_within_folder(std::string folder);
int reject(cv::Mat src_img, std::vector<double> facadeSize, std::vector<double> targetSize, double score, bool bDebug);
int reject(cv::Mat src_img, FacadeInfo& fi, ModelInfo& mi, bool bDebug);
double get_image_quality_score(cv::Mat src_img, ModelInfo& mi);
void readMetajson(std::string metajson, FacadeInfo& fi);
void readModeljson(std::string modeljson, ModelInfo& mi);
void writeMetajson(std::string metajson, FacadeInfo& fi);
cv::Mat cleanAlignedImage(cv::Mat src, float threshold);
cv::Mat deSkewImg(cv::Mat src_img);
void apply_segmentation_model(cv::Mat &croppedImage, cv::Mat &chip_seg, ModelInfo& mi, bool bDebug, std::string img_filename);
std::vector<int> adjust_chip(cv::Mat chip);
int choose_best_chip(std::vector<ChipInfo> chips, ModelInfo& mi, bool bDebug, std::string img_filename);
std::vector<double> compute_chip_info(ChipInfo chip, ModelInfo& mi, bool bDebug, std::string img_filename);
void find_spacing(cv::Mat src_img, std::vector<int> &separation_x, std::vector<int> &separation_y, bool bDebug);
cv::Rect findLargestRectangle(cv::Mat image);
bool findIntersection(cv::Rect a1, cv::Rect a2);
bool insideRect(cv::Rect a1, cv::Point p);
void pre_process(cv::Mat &chip_seg, cv::Mat& croppedImage, ModelInfo& mi, bool bDebug, std::string img_filename);

/**** steps *****/
bool chipping(FacadeInfo& fi, ModelInfo& mi, ChipInfo& chip, bool bMultipleChips, bool bDebug, std::string img_filename);
std::vector<ChipInfo> crop_chip_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
std::vector<ChipInfo> crop_chip_no_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
bool process_chip(ChipInfo &chip, ModelInfo& mi, bool bDebug, std::string img_filename);
std::vector<double> feedDnn(ChipInfo &chip, FacadeInfo& fi, ModelInfo& mi, bool bDebug, std::string img_filename);
cv::Mat synthesis(std::vector<double> predictions, cv::Size src_size, std::string dnnsOut_folder, cv::Scalar win_avg_color, cv::Scalar bg_avg_color, bool bDebug, std::string img_filename);

/**** grammar predictions ****/
std::vector<double> grammar1(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar2(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar3(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar4(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar5(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar6(ModelInfo& mi, std::vector<double> paras, bool bDebug);

/**** opt ****/
void opt_without_doors(cv::Mat& seg_rbg, std::vector<double>& predictions_opt, std::vector<double> predictions_init, int step, double range);
void opt_without_doors(cv::Mat& seg_rbg, std::vector<double>& predictions_opt, std::vector<double>& trans_opt, std::vector<double> predictions_init);
void opt_with_doors(cv::Mat& seg_rbg, std::vector<double>& predictions_opt, std::vector<double> predictions_init, int step, double range);
cv::Mat synthesis_opt(std::vector<double> predictions, cv::Size src_size, cv::Scalar win_color, cv::Scalar bg_color, bool bDebug, std::string img_filename);
std::vector<double> eval_accuracy_old(const cv::Mat& seg_img, const cv::Mat& gt_img);
std::vector<double> eval_accuracy(const cv::Mat& seg_img, const cv::Mat& gt_img);