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
			for (int i = 0; i < paras_num; i++) {
				std::cout << paras[i] << ", ";
			}
			std::cout << std::endl;
			try {
				double score = 0.0f;
				for (int i = 0; i < paras_num; i++) {
					if (paras[i] < 0 || paras[i] > 1)
						return 0.0;
				}
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
				predictions.push_back(paras[paras_num - 4]);
				predictions.push_back(paras[paras_num - 3]);
				predictions.push_back(paras[paras_num - 2]);
				predictions.push_back(paras[paras_num - 1]);
				for (int i = 0; i < predictions.size(); i++) {
					std::cout << predictions[i] << ", ";
				}
				std::cout << std::endl;
				// -- get predicted img
				cv::Mat syn_img;
				cv::Scalar win_avg_color(0, 0, 255, 0);
				cv::Scalar bg_avg_color(255, 0, 0, 0);
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
				}
				// recover to the original image
				cv::resize(syn_img, syn_img, target_img.size());
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
				// -- compute score
				std::vector<double> evaluations = util::eval_accuracy(syn_img, target_img);
				score = /*0.5 * evaluations[0] + */0.5 * evaluations[1] + 0.5 * evaluations[2];
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