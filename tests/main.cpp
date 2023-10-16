#include "yolo_interface.hpp"
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <signal.h> // Để sử dụng signal
#include <wiringPi.h>
#include <wiringSerial.h>

using namespace cv;
using namespace std;
using namespace chrono;

volatile sig_atomic_t flag = 0;
pthread_mutex_t mutex_result = PTHREAD_MUTEX_INITIALIZER;

int countObjectsInROI1 = 0;
int countObjectsInROI2 = 0;

void sigint_handler(int signum) {
    flag = 1; // Đặt biến flag để thoát khỏi vòng lặp
}

void* fn_serial(void *args){
    int fd;
    while (!flag) {
        if((fd = serialOpen ("/dev/ttyACM0", 9600)) < 0 ){
			fprintf (stderr, "Unable to open serial device: %s\n", strerror (errno));
		}
        else{
            pthread_mutex_lock(&mutex_result);
            int roi1 = countObjectsInROI1; // làn a
            int roi2 = countObjectsInROI2; // làn b
            pthread_mutex_unlock(&mutex_result);
            
            if (roi1 < 1 && roi2 < 1) {
                // Nếu cả a và b đều nhỏ hơn 5, gửi qua 1
                serialPuts(fd, "1");
                serialFlush(fd);
                printf("%s\n", "1");
                fflush(stdout);
            } else {
                if (roi1 > roi2) {
                    // Nếu tổng a > b, gửi qua ký tự 2
                    serialPuts(fd, "2");
                    serialFlush(fd);
                    printf("%s\n", "2");
                    fflush(stdout);
                } else if (roi1 < roi2) {
                    // Nếu tổng a < b, gửi qua ký tự 3
                    serialPuts(fd, "3");
                    serialFlush(fd);
                    printf("%s\n", "3");
                    fflush(stdout);
                }
            }
            printf("Count in ROI1: %d - Count in ROI2: %d \n", roi1, roi2);
            delay(1000);
        }
    }
    return NULL;
}

int main()
{
    float fps;
    pthread_t thread_serial = 0;
    cv::Point ROI_top_left_1 = cv::Point(10, 10);
    cv::Point ROI_right_bottom_1 = cv::Point(300, 230);
    cv::Point ROI_top_left_2 = cv::Point(310, 240);
    cv::Point ROI_right_bottom_2 = cv::Point(630, 470);
	YOLO_Interface * YOLO;
    cv::Mat res;
	std::vector<Object> objs;

	YOLO = new YOLO_Interface("/home/pi/yolo/libs/libyolo/config/YOLO_Config.txt", {"car"});
    pthread_create(&thread_serial, NULL, &fn_serial, NULL);

    // class represent for capturing video from camera and reading video file or image sequences
    VideoCapture videoCapture;
    Mat videoFrame;

    // open camera
    videoCapture.open(0);
    videoCapture.set(CAP_PROP_FRAME_WIDTH, 640);
    videoCapture.set(CAP_PROP_FRAME_HEIGHT, 480);

    namedWindow("VideoCapture", WINDOW_AUTOSIZE);

    // check open camera open sucessed or failed
    if(!videoCapture.isOpened())
    {
        std::cout<<"Failed to open camera."<<std::endl;
        return (-1);
    }
    else
    {
        std::cout << "Hit ESC to exit" << "\n" ;
        while (true)
        {
            //read video frame from camera and show in windows
            int tempCountROI1 = 0;
            int tempCountROI2 = 0;

            videoCapture.read(videoFrame);
            // Vẽ hình chữ nhật màu đỏ ở góc trên bên trái
            cv::rectangle(videoFrame, ROI_top_left_1, ROI_right_bottom_1, cv::Scalar(0, 0, 255), 2);

            // Vẽ hình chữ nhật màu xanh ở góc dưới bên phải
            cv::rectangle(videoFrame, ROI_top_left_2, ROI_right_bottom_2, cv::Scalar(0, 255, 0), 2);

            objs.clear();
            double timer = (double)cv::getTickCount();
            YOLO->Detect(videoFrame, objs);
            fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);
			for (const Object& obj : objs) {
                // Lấy tọa độ của đối tượng
                cv::Point objCenter = (obj.rect.br() + obj.rect.tl())*0.5;

                // Kiểm tra xem đối tượng có nằm trong ROI1 không
                if (objCenter.x >= ROI_top_left_1.x && objCenter.x <= ROI_right_bottom_1.x &&
                    objCenter.y >= ROI_top_left_1.y && objCenter.y <= ROI_right_bottom_1.y) {
                    tempCountROI1++;
                }

                // Kiểm tra xem đối tượng có nằm trong ROI2 không
                if (objCenter.x >= ROI_top_left_2.x && objCenter.x <= ROI_right_bottom_2.x &&
                    objCenter.y >= ROI_top_left_2.y && objCenter.y <= ROI_right_bottom_2.y) {
                    tempCountROI2++;
                }
            }

            // In số lượng đối tượng trong ROI1 và ROI2
            pthread_mutex_lock(&mutex_result);
            countObjectsInROI1 = tempCountROI1;
            countObjectsInROI2 = tempCountROI2;
            pthread_mutex_unlock(&mutex_result);

            cv::putText(
						videoFrame,
						std::to_string(fps),
						cv::Point(50, 400),
						cv::FONT_HERSHEY_SIMPLEX,
						1,
						{ 255, 255, 255 },
						2
					);
                    
			YOLO->Draw_Objects(videoFrame, videoFrame, objs);

            imshow("VideoCapture", videoFrame);
            int keycode = cv::waitKey(10) & 0xff ; 
            if (keycode == 27) break ;
        }
    }
    videoCapture.release();
    cv::destroyAllWindows() ;
    delete YOLO;
    pthread_join(thread_serial, NULL);
	return 0;
}
