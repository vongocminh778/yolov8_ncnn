cmake_minimum_required(VERSION 3.5)

project(detection LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

# NCNN
set(ncnn_DIR /home/pi/ncnn/build/install/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# add common libs
set(LIBS_COMMON_DIR $ENV{APPS_DIR}/libs/common/libs)
message("LIBS_COMMON_DIR: ${LIBS_COMMON_DIR}")


list(APPEND INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${LIBS_COMMON_DIR}/inc
        )
        
set(SOURCE
        src/yolo_interface.cpp
        src/yolov8.cpp)

add_library(${PROJECT_NAME} STATIC ${SOURCE})

target_include_directories(${PROJECT_NAME} PUBLIC inc ${INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC ncnn ${OpenCV_LIBS})


