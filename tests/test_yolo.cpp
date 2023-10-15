#include "yolo_interface.hpp"

#include <unistd.h>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace chrono;

int main()
{
    float fps;
	YOLO_Interface * YOLO;
    cv::Mat res;
	std::vector<Object> objs;
	YOLO = new YOLO_Interface("/home/pi/yolo/libs/libyolo/config/YOLO_Config.txt", {"car"});

    // class represent for capturing video from camera and reading video file or image sequences
    VideoCapture videoCapture;
    Mat videoFrame;

    // open camera
    videoCapture.open(0);
    namedWindow("VideoCapture", WINDOW_AUTOSIZE);

    // check open camera open sucessed or failed
    if(!videoCapture.isOpened())
    {
        cout << "Can't open camera" << endl;
    }
    else
    {
        videoCapture.set(CAP_PROP_FRAME_WIDTH, 640);
        videoCapture.set(CAP_PROP_FRAME_HEIGHT, 480);

        while (true)
        {
            //read video frame from camera and show in windows
            videoCapture.read(videoFrame);
            objs.clear();
            double timer = (double)cv::getTickCount();
            YOLO->Detect(videoFrame, objs);
            fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);
			cv::putText(
						videoFrame,
						std::to_string(fps),
						cv::Point(50, 400),
						cv::FONT_HERSHEY_SIMPLEX,
						1,
						{ 255, 255, 255 },
						2
					);
			YOLO->Draw_Objects(videoFrame, res, objs);

            // imshow("VideoCapture", videoFrame);
            imshow("VideoCapture", res);
            if(waitKey(30) >= 0) break;
        }
    }
    // delete YOLO;
	return 0;
}
