cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dn_lego_syn)

set(CMAKE_BUILD_TYPE Release)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("D:\\DN_Windows\\include\\")

add_executable(dn_lego_syn Utils.h Utils.cpp optGrammarParas.h optGrammarParas.cpp dn_lego_eval.h dn_lego_eval.cpp dn_lego_syn.h dn_lego_syn.cpp)
target_link_libraries(dn_lego_syn "${TORCH_LIBRARIES}")
target_link_libraries(dn_lego_syn "${OpenCV_LIBS}")
set_property(TARGET dn_lego_syn PROPERTY CXX_STANDARD 14)