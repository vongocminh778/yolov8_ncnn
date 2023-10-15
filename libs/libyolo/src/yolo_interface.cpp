#include "yolo_interface.hpp"

YOLO_Interface::YOLO_Interface(std::string filenameconfig, std::vector<std::string> class_filter)
{
    class_filter_=class_filter;
    std::size_t botDirPos = filenameconfig.find_last_of("/");
    std::string dir = filenameconfig.substr(0, botDirPos);
    // std::cout<<"dir: "<<dir<<std::endl;
    getYOLOconfig(filenameconfig);
    detector = new YOLOv8(dir+"/"+Yoloconfig.detector_engine_file_path, 320, true);
    // detector->make_pipe(true);
}

YOLO_Interface::~YOLO_Interface()
{
    detector->~YOLOv8();
}

bool YOLO_Interface::getYOLOconfig(std::string fileName)
{
    std::ifstream in(fileName.c_str());
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        return false;
    }
    std::string name;
    std::string value;
    while (std::getline(in, name, '='))
    {
        getline(in, value);
        if (name == "detector_engine_file_path") {
            Yoloconfig.detector_engine_file_path = value;
        }
        else if (name == "score_thres_") {
            Yoloconfig.score_thres_ = std::stof(value);
        }
        else if (name == "iou_thres_") {
            Yoloconfig.iou_thres_ = std::stof(value);
        }
        else if (name == "topk_") {
            Yoloconfig.topk_ = std::stoi(value);
        }
        else if (name == "num_labels_") {
            Yoloconfig.num_labels_ = std::stoi(value);
        }
        
        // if(str.size() > 0)
        //     vecOfStrs.push_back(str);
    }
    in.close();
    return true;
}


void YOLO_Interface::Detect(cv:: Mat img, std::vector<Object>& objs)
{
    // detector->copy_from_Mat(img);
    std::vector<Object> objs_temp;
	detector->infer(img, objs, Yoloconfig.score_thres_, Yoloconfig.iou_thres_);
    for (auto& obj : objs)
	{
        if (class_filter_.size()>0)
		{
			if (std::find(class_filter_.begin(), class_filter_.end(), CLASS_NAMES[obj.label]) == class_filter_.end())
			{
				continue;
			}
            else
                objs_temp.push_back(obj);
		}
    }
    objs.clear();
    objs.assign(objs_temp.begin(), objs_temp.end()); 
	// detector->postprocess(objs, CLASS_NAMES, class_filter_, Yoloconfig.score_thres_, Yoloconfig.iou_thres_, Yoloconfig.topk_, Yoloconfig.num_labels_);
}

void YOLO_Interface::Draw_Objects(const cv::Mat& image,
	cv::Mat& res,
	const std::vector<Object>& objs
)
{
    detector->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
}