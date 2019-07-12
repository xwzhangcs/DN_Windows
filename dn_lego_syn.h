#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <Windows.h>

#include <dlib/clustering.h>
#include <dlib/rand.h>

using namespace dlib;

int const max_BINARY_value = 255;
int const cluster_number = 2;

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

};

/**** helper functions *****/
std::vector<std::string> get_all_files_names_within_folder(std::string folder);
cv::Mat facade_clustering_kkmeans(cv::Mat src_img, int clusters);
int reject(std::string img_name, std::vector<double> facadeSize, std::vector<double> targetSize, double score);
void readMetajson(std::string metajson, FacadeInfo& fi);
void readModeljson(std::string modeljson, ModelInfo& mi);

/**** steps *****/
bool chipping(FacadeInfo& fi, std::string modeljson, cv::Mat& croppedImage, bool bMultipleChips, bool bDebug, std::string img_filename);
std::vector<cv::Mat> crop_chip(cv::Mat src_chip, std::string modeljson, int type, bool bground, std::vector<double> facadeSize, std::vector<double> targetSize, bool bMultipleChips);
cv::Mat adjust_chip(cv::Mat chip);
bool checkFacade(std::string facade_name);
void saveInvalidFacade(std::string metajson, std::string img_name, bool bDebug, std::string img_filename);

bool segment_chip(cv::Mat croppedImage, cv::Mat& dnn_img, std::string metajson, std::string modeljson, bool bDebug, std::string img_filename);
cv::Mat cleanAlignedImage(cv::Mat src, float threshold);
cv::Mat deSkewImg(cv::Mat src_img);
void writebackColor(std::string metajson, std::string attr, cv::Scalar color);
cv::Rect findLargestRectangle(cv::Mat image);
bool findIntersection(cv::Rect a1, cv::Rect a2);
bool insideRect(cv::Rect a1, cv::Point p);

std::vector<double> feedDnn(cv::Mat dnn_img, std::string metajson, std::string modeljson, bool bDebug, std::string img_filename);

std::vector<double> compute_confidence(cv::Mat croppedImage, std::string modeljson, bool bDebug);
std::vector<double> compute_door_paras(cv::Mat croppedImage, std::string modeljson, bool bDebug);

void synthesis(std::vector<double> predictions, cv::Size src_size, std::string dnnsOut_folder, cv::Scalar win_avg_color, cv::Scalar bg_avg_color, cv::Scalar win_histeq_color, cv::Scalar bg_histeq_color, std::string img_filename, bool bDebug);
cv::Scalar readColor(std::string metajson, std::string color_name);
bool readGround(std::string metajson);
double readScore(std::string metajson);
// For evaluating
void generateSegOutAndDnnOut(std::string chip_img_file, std::string modeljson, std::string segOut_file_name, std::string dnnOut_file_name, bool bDebug);

