cmake_minimum_required(VERSION 3.10)
project(yolo_model)

set(CMAKE_CXX_STANDARD 17)  


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(get_package_path src/get_package_path.cpp)
#add_executable(card_augmenter src/card_augmenter.cpp)
