#ifndef DETECT_NORMAL_YOLOV8_HPP
#define DETECT_NORMAL_YOLOV8_HPP
#include "fstream"
// #include "common_tensorrt.hpp"
// #include "NvInferPlugin.h"

#include <iostream>
#include <fstream>
// #include <NvInfer.h>
#include <memory>
// #include <NvOnnxParser.h>
#include <vector>
// #include <cuda_runtime_api.h>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/core/cuda.hpp>
// #include <opencv2/cudawarping.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/cudaarithm.hpp>
// #include "opencv2/cudaobjdetect.hpp"
// #include "opencv2/cudaimgproc.hpp"
#include <algorithm>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <net.h>
#include "cpu.h"

#include <common.hpp>

using namespace det;

// struct Object
// {
//     cv::Rect_<float> rect;
//     int label;
//     float prob;
// };

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

class YOLOv8
{
public:
	explicit YOLOv8(const std::string& engine_file_path, int _target_size=320, bool use_gpu=true);
	~YOLOv8();

	// void make_pipe(bool warmup = true);
	// void copy_from_Mat(const cv::Mat& image);
	// void letterbox(
	// 	const cv::Mat& image,
	// 	cv::Mat& out,
	// 	cv::Size& size
	// );
	// void preprocessImage(const cv::Mat& frame, float* gpu_input);
	void infer(const cv::Mat& rgb, std::vector<Object>& objects, float score_thres, float iou_thres);
	// void postprocess(
	// 	std::vector<Object>& objs,
	// 	std::vector<std::string> CLASS_NAMES,
	// 	std::vector<std::string> class_filter,
	// 	float score_thres = 0.25f,
	// 	float iou_thres = 0.65f,
	// 	int topk = 100,
	// 	int num_labels = 23
	// );

	static void draw_objects(
		const cv::Mat& image,
		cv::Mat& res,
		const std::vector<Object>& objs,
		const std::vector<std::string>& CLASS_NAMES,
		const std::vector<std::vector<unsigned int>>& COLORS
	);
	// int num_bindings;
	// int num_inputs = 0;
	// int num_outputs = 0;
	// std::vector<Binding> input_bindings;
	// std::vector<Binding> output_bindings;
	// std::vector<void*> host_ptrs;
	// std::vector<void*> device_ptrs;

	// PreParam pparam;
private:
	// nvinfer1::ICudaEngine* engine = nullptr;
	// nvinfer1::IRuntime* runtime = nullptr;
	// nvinfer1::IExecutionContext* context = nullptr;
	// cudaStream_t stream = nullptr;
	// Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };

	ncnn::Net yolo_engine;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];

	int num_channels;
	int num_anchors;
	cv::Mat nchw;
	cv::Size size_input;
	int width_input;
	int height_input;
	int channels_input;

};

#endif //DETECT_NORMAL_YOLOV8_HPP