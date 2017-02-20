#include <iostream>
#include <string>
#include <sstream>

using namespace std;

#include <opencv2/opencv.hpp>

using namespace cv;

// the majority of this code is from http://opencv-srf.blogspot.com/2010/09/object-detection-using-color-seperation.html
// although i made the following changes:
//
//  -  only video from the webcam is displayed along with a circle marker overlay, in contrast to the blog program
//  which drew lines along the object's path and displayed the tresholded image.
//
//  -  i track a blue expo marker instead of a red ball, so i changed the threshold values.
//
//  -  i make it so that the overlayed image is cleared every frame and a circle is only drawn when the program
//     thinks it sees a match.

int main( int argc, char** argv )
{
    VideoCapture cap( CV_CAP_ANY );

    if ( !cap.isOpened() )
    {
        cout << "Cannot open the web cam" << endl;
        return -1;
    }

    int iLowH = 89;
    int iHighH = 127;

    int iLowS = 91;
    int iHighS = 255;

    int iLowV = 122;
    int iHighV = 255;

    int iLastX = -1;
    int iLastY = -1;

    Mat imgTmp;
    cap.read(imgTmp);

    Mat overlay = Mat::zeros( imgTmp.size(), CV_8UC3 );

    while (true)
    {
        Mat imgOriginal;
        bool bSuccess = cap.read(imgOriginal);

        if (!bSuccess)
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }

        Mat imgHSV;
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);

        Mat imgThresholded;
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);

        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        Moments oMoments = moments(imgThresholded);
        overlay = Mat::zeros( imgTmp.size(), CV_8UC3 );

        double dM01 = oMoments.m01;
        double dM10 = oMoments.m10;
        double dArea = oMoments.m00;

        if (dArea > 250000 )
        {
            int posX = dM10 / dArea;
            int posY = dM01 / dArea;

            if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
                circle(overlay, Point(posX, posY), 30, Scalar(0,0,256), 10 );

            iLastX = posX;
            iLastY = posY;
        }

        imgOriginal = imgOriginal + overlay;
        imshow("Test", imgOriginal);

        if (waitKey(30) == 27)
            break;
    }

   return 0;
}
