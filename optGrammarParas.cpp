#include "optGrammarParas.h"


optGrammarParas::optGrammarParas() {
}

optGrammarParas::~optGrammarParas() {
}

std::vector<double> optGrammarParas::fit(const cv::Mat& src_img, const std::vector<double>& ini_paras, const int grammar_id, const int paras_num, const std::string modeljson) {
	try {
		//std::cout << "total points is " << total_points << std::endl;
		column_vector starting_point(paras_num);
		for (int i = 0; i < paras_num; i++) {
			starting_point(i) = ini_paras[i];
		}

		BFGSSolver solver(src_img, ini_paras, grammar_id, paras_num, modeljson);
		find_max_using_approximate_derivatives(dlib::bfgs_search_strategy(), dlib::objective_delta_stop_strategy(1e-4), solver, starting_point, 1, 0.005);
		std::vector<double> ans(paras_num);
		for (int i = 0; i < paras_num; i++) {
			ans[i] = starting_point(i);
		}
		return ans;
	}
	catch (char* ex) {
		std::cout << ex << std::endl;
	}
	catch (std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}
	catch (...) {
		std::cout << "BFGS optimization failure." << std::endl;
	}

	return{};
}