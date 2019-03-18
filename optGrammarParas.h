#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <dlib/optimization.h>
#include "Utils.h"

class optGrammarParas {

	typedef dlib::matrix<double, 0, 1> column_vector;

	class BFGSSolver {
	private:
		cv::Mat target_img;
		std::vector<double> init_paras;
		int grammar_id;
		int paras_num;
		std::string modeljson;

	public:
		BFGSSolver(const cv::Mat& target_img, const std::vector<double>& init_paras, const int grammar_id, const int paras_num, const std::string modeljson) {
			this->target_img = target_img;
			this->init_paras = init_paras;
			this->grammar_id = grammar_id;
			this->paras_num = paras_num;
			this->modeljson = modeljson;
		}

		double operator() (const column_vector& arg) const {
			std::vector<double> paras;
			for (int i = 0; i < paras_num; i++) {
				paras.push_back(arg(i));
			}
			/*for (int i = 0; i < paras_num; i++) {
				std::cout << paras[i] << ", ";
			}
			std::cout << std::endl;*/
			try {
				double score = 0.0f;
				// compute IOU
				// -- go to correct grammar
				std::vector<double> predictions;
				if (grammar_id == 1) {
					predictions = util::grammar1(modeljson, paras, false);
				}
				else if (grammar_id == 2) {
					predictions = util::grammar2(modeljson, paras, false);
				}
				else if (grammar_id == 3) {
					predictions = util::grammar3(modeljson, paras, false);
				}
				else if (grammar_id == 4) {
					predictions = util::grammar4(modeljson, paras, false);
				}
				else if (grammar_id == 5) {
					predictions = util::grammar5(modeljson, paras, false);
				}
				else if (grammar_id == 6) {
					predictions = util::grammar6(modeljson, paras, false);
				}
				else {
					//do nothing
					predictions = util::grammar1(modeljson, paras, false);
				}
				// -- get predicted img
				cv::Mat syn_img;
				if (predictions.size() == 5) {
					int img_rows = predictions[0];
					int img_cols = predictions[1];
					int img_groups = predictions[2];
					double relative_width = predictions[3];
					double relative_height = predictions[4];
					syn_img = util::generateFacadeSynImage(224, 224, img_rows, img_cols, img_groups, relative_width, relative_height);
				}
				if (predictions.size() == 8) {
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
				// -- compute score
				int inter_cnt = 0;
				for (int r = 0; r < target_img.rows; r++) {
					for (int c = 0; c < target_img.cols; c++) {
						int syn_b = syn_img.at<cv::Vec3b>(r, c)[0];
						int syn_g = syn_img.at<cv::Vec3b>(r, c)[0];
						int syn_r = syn_img.at<cv::Vec3b>(r, c)[0];
						int src_b = target_img.at<cv::Vec3b>(r, c)[0];
						int src_g = target_img.at<cv::Vec3b>(r, c)[0];
						int src_r = target_img.at<cv::Vec3b>(r, c)[0];
						if (syn_b == 0 && syn_g == 0 && syn_r == 0 && src_b == 0 && src_g == 0 && src_r == 0)
							inter_cnt++;
						if (syn_b == 255 && syn_g == 255 && syn_r == 255 && src_b == 255 && src_g == 255 && src_r == 255)
							inter_cnt++;
					}
				}
				score = inter_cnt * 1.0 / (target_img.rows * target_img.cols);
				std::cout << "score is " << score << std::endl;
				return score;
			}
			catch (...) {
				std::cout << "exception" << std::endl;
				return 0;
			}
		}
	};

protected:
	optGrammarParas();
	~optGrammarParas();

public:
	static std::vector<double> fit(const cv::Mat& src_img, const std::vector<double>& ini_paras, const int grammar_id, const int paras_num, const std::string modeljson);
};