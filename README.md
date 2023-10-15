# Object Detection library v1.0 (YOLOV8)

## Introduction

This repository contains tracking library and code test.<br/>
To include this library into the main project, add it into the CMakeLists.txt in the root directory:
```
add_subdirectory(<path-to-tracking-directory>)
```

## Build test app
Step to build tracking_test app to test tracking:
```
$ git clone project
$ cd to Repo
$ Compiling OpenCV_4.4 with CUDA
$ mkdir build
$ cd build/
$ cmake ../
$ make -j6
$ mv tests/test_yolo ..
$ cd ../
$ ./test_yolo

sudo pip3.8 install virtualenv


```

Output binary (test_yolo) will be created in the build/tests directory.
