#include "yolov8.hpp"

YOLOv8::YOLOv8(const std::string& engine_file_path, int _target_size, bool use_gpu)
{

	yolo_engine.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo_engine.opt = ncnn::Option();

	#if NCNN_VULKAN
    	yolo_engine.opt.use_vulkan_compute = use_gpu;
    	std::cout<<"Use GPU: "<<use_gpu<<std::endl;
	#endif
    	std::cout<<"Num CPU: "<<ncnn::get_big_cpu_count()<<std::endl;
    	yolo_engine.opt.num_threads = ncnn::get_big_cpu_count();

    // yolo.opt.use_int8_inference = true;
    // yolo.opt.num_threads = 4;
	std::cout<<"param model: "<<engine_file_path.substr(0, engine_file_path.length()-4)+".param"<<std::endl;
    yolo_engine.load_param((engine_file_path.substr(0, engine_file_path.length()-4)+".param").c_str());
    yolo_engine.load_model(engine_file_path.c_str());
    

    target_size = _target_size;
    mean_vals[0] = 103.53f;
    mean_vals[1] = 116.28f;
    mean_vals[2] = 123.675f;
    norm_vals[0] = 1.0 / 255.0f;
    norm_vals[1] = 1.0 / 255.0f;
    norm_vals[2] = 1.0 / 255.0f;
}

YOLOv8::~YOLOv8()
{
	// this->context->destroy();
	// this->engine->destroy();
	// this->runtime->destroy();
	// cudaStreamDestroy(this->stream);
	// for (auto& ptr : this->device_ptrs)
	// {
	// 	CHECK(cudaFree(ptr));
	// }
	// for (auto& ptr : this->host_ptrs)
	// {
	// 	CHECK(cudaFreeHost(ptr));
	// }
}


static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}
static float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}
static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = grid_strides.size();
    const int num_class = 80;
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;

            objects.push_back(obj);
        }
    }
}

void YOLOv8::infer(const cv::Mat& rgb, std::vector<Object>& objects, float score_thres, float iou_thres)
{
	int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo_engine.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    ncnn::Mat out;
    ex.extract("output", out);

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    generate_proposals(grid_strides, out, score_thres, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, iou_thres);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort objects by area
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
}

// void YOLOv8::postprocess(
// 	std::vector<Object>& objs,
// 	std::vector<std::string> CLASS_NAMES,
// 	std::vector<std::string> class_filter,
// 	float score_thres,
// 	float iou_thres,
// 	int topk,
// 	int num_labels
// )
// {
// 	objs.clear();
// 	auto& dw = this->pparam.dw;
// 	auto& dh = this->pparam.dh;
// 	auto& width = this->pparam.width;
// 	auto& height = this->pparam.height;
// 	auto& ratio = this->pparam.ratio;
// 	std::vector<cv::Rect> bboxes;
// 	std::vector<float> scores;
// 	std::vector<int> labels;
// 	std::vector<int> indices;

//     float *output = static_cast<float*>(this->host_ptrs[0]);
// 	for (int i = 0; i < num_anchors; i++)
// 	{
//         // #ifdef YOLOV8
// 		if (num_channels-num_labels==4)
// 		{
// 			// printf("YOLOV888888888888888888888888\n");
//             auto bboxes_ptr = output;
//             auto scores_ptr = output + 4;
//             auto max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
//             float score = *max_s_ptr;
//             if (score > score_thres)
//             {
//                 float x = *bboxes_ptr++ - dw;
//                 float y = *bboxes_ptr++ - dh;
//                 float w = *bboxes_ptr++;
//                 float h = *bboxes_ptr;

//                 float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
//                 float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
//                 float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
//                 float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

//                 int label = max_s_ptr - scores_ptr;
//                 cv::Rect_<float> bbox;
//                 bbox.x = x0;
//                 bbox.y = y0;
//                 bbox.width = x1 - x0;
//                 bbox.height = y1 - y0;

//                 bboxes.push_back(bbox);
//                 labels.push_back(label);
//                 scores.push_back(score);
//             }
// 		}
//         // #else
// 		else
// 		{
// 			// printf("YOLOV666666666666666666\n");
// 			auto confidence = output[4];
//             if (confidence >= score_thres)
//             {
//                 auto bboxes_ptr = output;
//                 auto scores_ptr = output + 5;
//                 auto max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
//                 float score = confidence* *max_s_ptr;
//                 if (score > score_thres)
//                 {
//                     float x = *bboxes_ptr++ - dw;
//                     float y = *bboxes_ptr++ - dh;
//                     float w = *bboxes_ptr++;
//                     float h = *bboxes_ptr;

//                     float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
//                     float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
//                     float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
//                     float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

//                     int label = max_s_ptr - scores_ptr;
//                     cv::Rect_<float> bbox;
//                     bbox.x = x0;
//                     bbox.y = y0;
//                     bbox.width = x1 - x0;
//                     bbox.height = y1 - y0;

//                     bboxes.push_back(bbox);
//                     labels.push_back(label);
//                     scores.push_back(score);
//                 }
//             }
// 		}
//         // #endif
//         output += num_channels;
// 	}

// #ifdef BATCHED_NMS
// 	cv::dnn::NMSBoxesBatched(
// 		bboxes,
// 		scores,
// 		labels,
// 		score_thres,
// 		iou_thres,
// 		indices
// 	);
// #else
// 	cv::dnn::NMSBoxes(
// 		bboxes,
// 		scores,
// 		score_thres,
// 		iou_thres,
// 		indices
// 	);
// #endif
// 	int cnt = 0;
// 	for (auto& i : indices)
// 	{
// 		if (cnt >= topk)
// 		{
// 			break;
// 		}
// 		if (class_filter.size()>0)
// 		{
// 			if (std::find(class_filter.begin(), class_filter.end(), CLASS_NAMES[labels[i]]) == class_filter.end())
// 			{
// 				continue;
// 			}
// 		}
// 		Object obj;
// 		obj.rect = bboxes[i];
// 		obj.prob = scores[i];
// 		obj.label = labels[i];
// 		objs.push_back(obj);
// 		cnt += 1;
// 	}
// }

void YOLOv8::draw_objects(
	const cv::Mat& image,
	cv::Mat& res,
	const std::vector<Object>& objs,
	const std::vector<std::string>& CLASS_NAMES,
	const std::vector<std::vector<unsigned int>>& COLORS
)
{
	res = image.clone();
	int index_obj=0;
	for (auto& obj : objs)
	{
		cv::Scalar color = cv::Scalar(
			COLORS[obj.label][0],
			COLORS[obj.label][1],
			COLORS[obj.label][2]
		);
		cv::rectangle(
			res,
			obj.rect,
			color,
			2
		);

		char text[256];
		sprintf(
			text,
			"%s %.1f%%",
			CLASS_NAMES[obj.label].c_str(),
			obj.prob * 100
		);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(
			text,
			cv::FONT_HERSHEY_SIMPLEX,
			0.4,
			1,
			&baseLine
		);

		int x = (int)obj.rect.x;
		int y = (int)obj.rect.y + 1;

		if (y > res.rows)
			y = res.rows;

		cv::rectangle(
			res,
			cv::Rect(x, y, label_size.width, label_size.height + baseLine),
			{ 0, 0, 255 },
			-1
		);

		cv::putText(
			res,
			text,
			cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX,
			0.4,
			{ 255, 255, 255 },
			1
		);

		cv::putText(
			res,
			std::to_string(index_obj),
			cv::Point((int)obj.rect.x + (int)obj.rect.width, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX,
			1,
			{ 255, 255, 255 },
			2
		);
		index_obj++;
	}
}

