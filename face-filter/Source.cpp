#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, const char** argv) {

    VideoCapture cam(0);
    if (!cam.isOpened()) return -1;
    
    Mat frame;
    while (1) {
        cam.read(frame);
        imshow("camera", frame);

        waitKey(1);
    }
    cam.release();

    return 0;
}