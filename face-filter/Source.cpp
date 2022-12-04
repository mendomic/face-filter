#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Windows.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

// face_detection
// Basic face detection that will place rectangles over faces
void face_detection() {
    VideoCapture cap(0);
    Mat src;
    Mat dst;
    Mat gray;

    classifier.load("C:\\OpenCV-4.6.0\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");

    const int FPS = 30;

    if (!cap.isOpened()) return;
    while (1) {
        cap.read(src);
        src.copyTo(dst);
        cvtColor(dst, gray, COLOR_BGR2GRAY, 0);

        //detect faces
        classifier.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // draw faces
        for (auto&& face : faces) {
            rectangle(dst, face, Scalar(0, 255, 0), 2);
        }
    }
}

int main(int argc, const char** argv) {

    CascadeClassifier face_cascade;
    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
        "{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
        "{camera|0|Camera device number.}");
    parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
        "You can use Haar or LBP features.\n\n");
    parser.printMessage();

    FacemarkKazemi::Params params;
    params.configfile = configfile_name;
    Ptr<Facemark> facemark = FacemarkKazemi::create(params);
    facemark->setFaceDetector(myDetector);
    
    vector<Rect> faces;
    vector< vector<Point2f> > shapes;
    Mat img;

    while (1) {
        faces.clear();
        shapes.clear();
        cam >> img;
        resize(img, img, Size(600, 600), 0, 0, INTER_LINEAR_EXACT);
        facemark->getFaces(img, faces);
        if (faces.size() == 0) {
            cout << "No faces found in this frame" << endl;
        }
        else {
            for (size_t i = 0; i < faces.size(); i++)
            {
                cv::rectangle(img, faces[i], Scalar(255, 0, 0));
            }
            if (facemark->fit(img, faces, shapes))
            {
                for (unsigned long i = 0; i < faces.size(); i++) {
                    for (unsigned long k = 0; k < shapes[i].size(); k++)
                        cv::circle(img, shapes[i][k], 3, cv::Scalar(0, 0, 255), FILLED);
                }
            }
        }
        namedWindow("Detected_shape");
        imshow("Detected_shape", img);
        if (waitKey(1) >= 0) break;
    }
    */
    
    /*VideoCapture cam(0);
    if (!cam.isOpened()) return -1;
    
    Mat frame;
    while (1) {
        cam.read(frame);
        imshow("camera", frame);
        waitKey(1);
    }
    cam.release();*/
    
 

    return 0;
}

/*
* References
https://docs.opencv.org/3.4/d8/d3c/tutorial_face_landmark_detection_in_video.html


*/