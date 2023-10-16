# Object Detection NCNN (YOLOV8)

## Check if the camera is working?
```
$ gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! videoscale ! clockoverlay time-format="%D %H:%M:%S" ! autovideosink
```

## Build test app
```
$ ls /dev/tty* # check device arduino (ttyACM0, ttyAMA0)
$ cd yolo
$ mkdir build
$ cd build/
$ cmake ../
$ make -j4
$ sudo ./tests/yolov8
```

## Capture images to trainning
```
$ cd capture
$ python3 capture_imgs.py
```
Press 'S' to save images.

## Create service
```
$ cd ~
$ sudo nano run_yolo.sh
```
```
#!/bin/bash

echo "Run Object Detection Yolov8"
cd /home/pi/yolo/build
sudo ./tests/yolov8
```

```
$ sudo /home/pi/run_yolo.sh
```

## SSH
```
$ ping raspberrypi.local
$ ssh pi@raspberrypi.local # alternative by ip ssh pi@192.168.*.* pass: 1
$ ifconfig # check ip

```
