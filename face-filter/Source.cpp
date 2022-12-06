#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/photo.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::face;

void divideIntoTriangles(Rect rect, vector<Point2f>& points, vector< vector<int> >& delaunayTri);
void warpTriangle(Mat& img_src, Mat& img_dst, vector<Point2f>& triangle_src, vector<Point2f>& triangle_dst);

//Divide the face into triangles for warping
void divideIntoTriangles(Rect rect, vector<Point2f>& points, vector< vector<int> >& Tri) {

    // Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);
    // Insert points into subdiv
    for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
        subdiv.insert(*it);
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point2f> pt(3);
    vector<int> ind(3);
    for (size_t i = 0; i < triangleList.size(); i++)
    {
        Vec6f triangle = triangleList[i];
        pt[0] = Point2f(triangle[0], triangle[1]);
        pt[1] = Point2f(triangle[2], triangle[3]);
        pt[2] = Point2f(triangle[4], triangle[5]);
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
            for (int j = 0; j < 3; j++)
                for (size_t k = 0; k < points.size(); k++)
                    if (abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
                        ind[j] = (int)k;
            Tri.push_back(ind);
        }
    }
}
void warpTriangle(Mat& img_src, Mat& img_dst, vector<Point2f>& triangle_src, vector<Point2f>& triangle_dst) {
    Rect rectangle1 = boundingRect(triangle_src);
    Rect rectangle2 = boundingRect(triangle_dst);

    // Offset points by left top corner of the respective rectangles
    vector<Point2f> triangle_srcRect, triangle_dstRect;
    vector<Point> triangle_dstRectInt;
    for (int i = 0; i < 3; i++) {
        triangle_srcRect.push_back(Point2f(triangle_src[i].x - rectangle1.x, triangle_src[i].y - rectangle1.y));
        triangle_dstRect.push_back(Point2f(triangle_dst[i].x - rectangle2.x, triangle_dst[i].y - rectangle2.y));
        triangle_dstRectInt.push_back(Point((int)(triangle_dst[i].x - rectangle2.x), (int)(triangle_dst[i].y - rectangle2.y))); // for fillConvexPoly
    }
    
    // Get mask by filling triangle
    Mat mask = Mat::zeros(rectangle2.height, rectangle2.width, CV_32FC3);
    fillConvexPoly(mask, triangle_dstRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
    
    // Apply warpImage to small rectangular patches
    Mat img_srcRect;
    img_src(rectangle1).copyTo(img_srcRect);
    Mat img_dstRect = Mat::zeros(rectangle2.height, rectangle2.width, img_srcRect.type());
    Mat warp_mat = getAffineTransform(triangle_srcRect, triangle_dstRect);
    warpAffine(img_srcRect, img_dstRect, warp_mat, img_dstRect.size(), INTER_LINEAR, BORDER_REFLECT_101);
    multiply(img_dstRect, mask, img_dstRect);
    multiply(img_dst(rectangle2), Scalar(1.0, 1.0, 1.0) - mask, img_dst(rectangle2));
    img_dst(rectangle2) = img_dst(rectangle2) + img_dstRect;
}

void faceSwap(Mat& img_src, VideoCapture& camera, CascadeClassifier& face_cascade, string model) {
    if (!camera.isOpened()) {
        cerr << "ERROR: Failed to open camera feed" << endl;
        return;
    }
    if (img_src.empty()) {
        cerr << "ERROR: Failed to load source image" << endl;
        return;
    }
    if (face_cascade.empty()) {
        cerr << "ERROR: Failed to load cascade classifier" << endl;
        return;
    }

    // Load pretrained model
    Ptr < Facemark > facemark = FacemarkLBF::create();
    try {
        facemark->loadModel(model);
    }
    catch (cv::Exception& e) {
        cerr << e.what() << endl;
        cerr << "ERROR: Failed to load pretrained model" << endl;
        return;
    }

    // Detect faces on source image
    vector<Rect> faces_src;
    vector< vector<Point2f> > shape_src;
    face_cascade.detectMultiScale(img_src, faces_src);
    facemark->fit(img_src, faces_src, shape_src);
    if (shape_src.empty()) {
        cerr << "ERROR: No face detected in source image" << endl;
        return;
    }

    // Process source image
    vector<Point2f> points_src;
    points_src = shape_src[0];
    img_src.convertTo(img_src, CV_32F);
        
    // Find convex hull
    vector<Point2f> boundary_image_src;
    vector<int> index;
    convexHull(Mat(points_src), index, false, false);
    for (size_t j = 0; j < index.size(); j++) {
        boundary_image_src.push_back(points_src[index[j]]);
    }

    // Triangulation for points on the convex hull
    vector< vector<int> > triangles;
    Rect rect(0, 0, img_src.cols, img_src.rows);
    divideIntoTriangles(rect, boundary_image_src, triangles);

    // While loop for camera feed
    while (true) {
        // Get a frame from camera
        Mat frame_dst;
        camera.read(frame_dst);

        // Detect faces in camera frame
        vector< vector<Point2f> > shape_dst;
        vector<Rect> faces_dst;
        face_cascade.detectMultiScale(frame_dst, faces_dst);
        facemark->fit(frame_dst, faces_dst, shape_dst);

        // If faces were detected
        if (!shape_dst.empty()) {
            Mat img_dstWarped = frame_dst.clone();
            img_dstWarped.convertTo(img_dstWarped, CV_32F);

            // Find convex hull
            vector<Point2f> points_dst = shape_dst[0];
            vector<Point2f> boundary_image_dst;
            for (size_t j = 0; j < index.size(); j++) {
                boundary_image_dst.push_back(points_dst[index[j]]);
            }
            
            // Apply affine transformation to Delaunay triangles
            for (size_t j = 0; j < triangles.size(); j++) {
                vector<Point2f> triangle_src;
                vector<Point2f> triangle_dst;
                // Get points for img_src, img_dst corresponding to the triangles
                for (int k = 0; k < 3; k++) {
                    triangle_src.push_back(boundary_image_src[triangles[j][k]]);
                    triangle_dst.push_back(boundary_image_dst[triangles[j][k]]);
                }
                warpTriangle(img_src, img_dstWarped, triangle_src, triangle_dst);
            }
            
            // Calculate mask
            vector<Point> hull;
            for (size_t j = 0; j < boundary_image_dst.size(); j++) {
                Point pt((int)boundary_image_dst[j].x, (int)boundary_image_dst[j].y);
                hull.push_back(pt);
            }
            Mat mask = Mat::zeros(frame_dst.rows, frame_dst.cols, frame_dst.depth());
            fillConvexPoly(mask, &hull[0], (int)hull.size(), Scalar(255, 255, 255));

            // Clone seamlessly and apply mask.
            Rect r = boundingRect(boundary_image_dst);
            Point center = (r.tl() + r.br()) / 2;
            Mat output;
            img_dstWarped.convertTo(img_dstWarped, CV_8UC3);
            seamlessClone(img_dstWarped, frame_dst, mask, center, output, NORMAL_CLONE);
            
            // Output face swapped frame
            imshow("Face Swapped", output);
        }
        else {
            // Output regular frame if no faces detected
            imshow("Face Swapped", frame_dst);
        }
        
        // Runs until any key is pressed
        if (waitKey(1) >= 0) {
            return;
        }
    }
}

int main(int argc, const char** argv) {  
    string src = argv[1];
    int cam = atoi(argv[2]);
    string model = argv[3];
    string cascade = argv[4];
    
    Mat img_src = imread(src);
    VideoCapture camera(cam);
    CascadeClassifier face_cascade;
    face_cascade.load(cascade); 

    faceSwap(img_src, camera, face_cascade, model);
    
    camera.release();
    return 1;
}

/*
References:
https://docs.opencv.org/3.4/d8/d3c/tutorial_face_landmark_detection_in_video.html
https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml
https://github.com/opencv/opencv_contrib/blob/4.x/modules/face/samples/sample_face_swapping.cpp
*/