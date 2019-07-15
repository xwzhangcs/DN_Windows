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
	std::string model_path;
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
	std::string invalidfacadesFolder;
	std::string chipsFolder;
	std::string segsFolder;
	std::string dnnsInFolder;
	std::string dnnsOutFolder;
	bool debug;
	std::vector<double> defaultSize;
	std::vector<double> targetChipSize;
	Grammar grammars[6];
	std::string classifier_path;
	int number_grammars;
	std::string reject_model;

};
/**** model variables *****/
std::shared_ptr<torch::jit::script::Module> reject_classifier_module;
void initial_models(ModelInfo& mi);

/**** helper functions *****/
std::vector<std::string> get_all_files_names_within_folder(std::string folder);
int reject(std::string img_name, std::vector<double> facadeSize, std::vector<double> targetSize, double score, bool bDebug);
int reject(std::string img_name, std::string model_path, std::vector<double> facadeSize, std::vector<double> targetSize, std::vector<double> defaultImgSize, bool bDebug);
void readMetajson(std::string metajson, FacadeInfo& fi);
void readModeljson(std::string modeljson, ModelInfo& mi);
void writeMetajson(std::string metajson, FacadeInfo& fi);
cv::Mat cleanAlignedImage(cv::Mat src, float threshold);
cv::Mat deSkewImg(cv::Mat src_img);

/**** steps *****/
bool chipping(FacadeInfo& fi, ModelInfo& mi, cv::Mat& croppedImage, bool bMultipleChips, bool bDebug, std::string img_filename);
std::vector<cv::Mat> crop_chip_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
std::vector<cv::Mat> crop_chip_no_ground(cv::Mat src_facade, int type, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
bool segment_chip(cv::Mat croppedImage, cv::Mat& dnn_img, FacadeInfo& fi, ModelInfo& mi, bool bDebug, std::string img_filename);
std::vector<double> feedDnn(cv::Mat dnn_img, FacadeInfo& fi, ModelInfo& mi, bool bDebug, std::string img_filename);
void synthesis(std::vector<double> predictions, cv::Size src_size, std::string dnnsOut_folder, cv::Scalar win_avg_color, cv::Scalar bg_avg_color, bool bDebug, std::string img_filename);

/**** grammar predictions ****/
std::vector<double> grammar1(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar2(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar3(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar4(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar5(ModelInfo& mi, std::vector<double> paras, bool bDebug);
std::vector<double> grammar6(ModelInfo& mi, std::vector<double> paras, bool bDebug);
