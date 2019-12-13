/*
Author : Asher
Date   : 2019.12.09
Reference:
1. https://blog.csdn.net/zhaoyin214/article/details/83040650
2. https://blog.csdn.net/u013512448/article/details/77804161
3. https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
4. https://blog.csdn.net/u012525096/article/details/78348765
*/

#include <opencv2/opencv.hpp>
#include <vector>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    try
    {
        cv::VideoCapture cvCap(0);

        if (!cvCap.isOpened())
        {
            std::cout << "Can not open the camera." << std::endl;
            return 1;
        }

        // load face detection and pose estimation models.
        dlib::frontal_face_detector dlibSvmFaceDetector;
        dlib::shape_predictor dlibSpFaceLandmark;

        std::vector<dlib::rectangle> dlibRectsFaces;
        std::vector<dlib::full_object_detection> dlibDetsShapes;

        cv::Mat cvImgFrame;
        cv::Mat cvImgFrameGray;

        clock_t clkBegin;
        clock_t clkEnd;

        dlibSvmFaceDetector = dlib::get_frontal_face_detector();
        dlib::deserialize("./shape_predictor_68_face_landmarks.dat") >> dlibSpFaceLandmark;

        while (true)
        {
            if (!cvCap.read(cvImgFrame))
            {
                break;
            }

            clkBegin = clock();

            cv::flip(cvImgFrame, cvImgFrame, 1);
            cv::cvtColor(cvImgFrame, cvImgFrameGray, cv::COLOR_BGR2GRAY);
            dlib::cv_image<unsigned char> dlibImgFrameGray(cvImgFrameGray);

            dlibRectsFaces = dlibSvmFaceDetector(dlibImgFrameGray);
            dlibDetsShapes.clear();

            for (unsigned int idxFace = 0; idxFace < dlibRectsFaces.size(); idxFace++)
            {
                dlib::full_object_detection shape = dlibSpFaceLandmark(dlibImgFrameGray, dlibRectsFaces[idxFace]);
                dlibDetsShapes.push_back(shape);

                cv::rectangle(cvImgFrame, cvRect(
                    dlibRectsFaces[idxFace].left(),
                    dlibRectsFaces[idxFace].top(),
                    dlibRectsFaces[idxFace].width(),
                    dlibRectsFaces[idxFace].height()),
                    cv::Scalar(0, 255, 0), 1);

                for (int idxLandmark = 0; idxLandmark < dlibSpFaceLandmark.num_parts(); idxLandmark++)
                {
                    cv::circle(cvImgFrame, cvPoint(
                        dlibDetsShapes[idxFace].part(idxLandmark).x(),
                        dlibDetsShapes[idxFace].part(idxLandmark).y()),
                        1, cv::Scalar(0, 0, 255), -1);

                    int ratio = 1;
                    //Dlib's feature points are arranged in order,so we can get 6 landmarks below by id
                    std::vector<cv::Point2d> image_points;
                    image_points.push_back(cv::Point2d(int(dlibDetsShapes[idxFace].part(30).x() / ratio), int(shape.part(30).y()) / ratio));    // Nose tip
                    image_points.push_back(cv::Point2d(int(dlibDetsShapes[idxFace].part(8).x() / ratio), int(shape.part(8).y()) / ratio));    // Chin
                    image_points.push_back(cv::Point2d(int(dlibDetsShapes[idxFace].part(36).x() / ratio), int(shape.part(36).y()) / ratio));     // Left eye left corner
                    image_points.push_back(cv::Point2d(int(dlibDetsShapes[idxFace].part(45).x() / ratio), int(shape.part(45).y()) / ratio));    // Right eye right corner
                    image_points.push_back(cv::Point2d(int(dlibDetsShapes[idxFace].part(48).x() / ratio), int(shape.part(48).y()) / ratio));    // Left Mouth corner
                    image_points.push_back(cv::Point2d(int(dlibDetsShapes[idxFace].part(54).x() / ratio), int(shape.part(54).y()) / ratio));    // Right mouth corner

                    std::vector<cv::Point3d> model_points;
                    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
                    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
                    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
                    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
                    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
                    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner

                    // Camera internals
                    double focal_length = cvImgFrame.cols; // Approximate focal length.
                    Point2d center = cv::Point2d(cvImgFrame.cols / 2, cvImgFrame.rows / 2);
                    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
                    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

                    cout << "Camera Matrix " << endl << camera_matrix << endl;
                    // Output rotation and translation
                    cv::Mat rotation_vector; // Rotation in axis-angle form
                    cv::Mat translation_vector;

                    // Solve for pose
                    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

                    // Project a 3D point (0, 0, 1000.0) onto the image plane.
                    // We use this to draw a line sticking out of the nose

                    vector<Point3d> nose_end_point3D;
                    vector<Point2d> nose_end_point2D;
                    nose_end_point3D.push_back(Point3d(0, 0, 1000.0));

                    projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);

                    cv::line(cvImgFrame, image_points[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);

                    cout << "Rotation Vector " << endl << rotation_vector << endl;
                    cout << "Translation Vector" << endl << translation_vector << endl;

                    cout << nose_end_point2D << endl;
                }

            }
            clkEnd = clock();
            std::cout << "Running time: " << (double)(clkEnd - clkBegin) <<
                "ms" << std::endl;

            // Display it all on the screen
            cv::imshow("webcam", cvImgFrame);

            if ('q' == (char)(0xFF & cv::waitKey(10)))
            {
                cv::destroyAllWindows();
                cvCap.release();
                break;
            }
        }
    }
    catch (dlib::serialization_error& e)
    {
        std::cout << "You need dlib's default face landmarking model file " <<
            "to run this example." << std::endl;
        std::cout << std::endl << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << "\nexception thrown!" << std::endl;
        std::cout << e.what() << std::endl;
    }

}