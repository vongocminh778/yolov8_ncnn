##################### STANDARD HEADER #########################

cmake_minimum_required(VERSION 3.5)

##################### PROJECT DEF #############################
project(test_yolo)
set(CMAKE_BUILD_TYPE Release)

# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)


find_package(PkgConfig REQUIRED)
#Use Pkg-config to configure GStreamer
pkg_search_module(GLIB REQUIRED glib-2.0) 
pkg_check_modules(GST REQUIRED
                    gstreamer-1.0
                    gstreamer-video-1.0
                    gstreamer-app-1.0
                    )
                    
find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

####################### INCLUDE   #############################

include_directories(${INCLUDE_DIRS} ${GST_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
add_executable(${PROJECT_NAME} test_yolo.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE  detection ${OpenCV_LIBS} ${GST_LIBRARIES})