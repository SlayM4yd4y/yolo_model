cmake_minimum_required(VERSION 3.10)
project(yolo_model)

set(CMAKE_CXX_STANDARD 17)  

find_package(OpenCV REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(SOURCES
    src/card_augmenter.cpp
)


include_directories(${OpenCV_INCLUDE_DIRS})

add_library(get_package_path
    src/get_package_path.cpp)

add_executable(card_augmenter src/card_augmenter.cpp)
target_link_libraries(card_augmenter get_package_path ${OpenCV_LIBS})
target_include_directories(card_augmenter PRIVATE ${OpenCV_INCLUDE_DIRS})

