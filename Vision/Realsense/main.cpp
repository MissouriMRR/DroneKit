#include "wchar.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "pxcsession.h"
#include "pxcsensemanager.h"
#include "pxccapture.h"

#include <iostream>

using namespace std;
using namespace cv;

const String FRONTALFACE_HAAR_CASCADE_PATH( "C:\\Users\\Christopher\\Documents\\opencv\\data\\haarcascades_cuda\\haarcascade_frontalface_default.xml" );
const String WINDOW_TITLE("Face Detection Test");
const Scalar RECT_COLOR(0, 255, 0);

const int ESC_KEY = 27;

const int RES_W = 640;
const int RES_H = 480;
const float FPS = 60;

void PXCImage2CVMat(Mat& img, PXCImage *pxcImg, PXCImage::PixelFormat format)
{
    PXCImage::ImageData data;
    PXCImage::ImageInfo info;
    pxcImg->AcquireAccess(PXCImage::ACCESS_READ, format, &data);
    info = pxcImg->QueryInfo();

    int width = info.width;
    int height = info.height;

    if (!format)
      format = info.format;

    int type;

    if (format == PXCImage::PIXEL_FORMAT_Y8)
      type = CV_8UC1;
    else if (format == PXCImage::PIXEL_FORMAT_RGB24)
      type = CV_8UC3;
    else if (format == PXCImage::PIXEL_FORMAT_DEPTH_F32)
      type = CV_32FC1;
    else if (format == PXCImage::PIXEL_FORMAT_DEPTH)
      type = CV_16UC1;

    img = cv::Mat(cv::Size(width, height), type, data.planes[0]);
    pxcImg->ReleaseAccess(&data);
}


int main()
{
  Ptr<cuda::CascadeClassifier> gpuCascade = cuda::CascadeClassifier::create(FRONTALFACE_HAAR_CASCADE_PATH);
  PXCSenseManager * sm = PXCSenseManager::CreateInstance();
  PXCCapture::Sample * sample = nullptr;
  vector<Rect> faces;
  Mat img;
  Mat gray;
  cuda::GpuMat gpuImg;
  cuda::GpuMat buffer;

  sm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, RES_W, RES_H, FPS);
  sm->Init();

  gpuCascade->setMinNeighbors(4);
  gpuCascade->setScaleFactor(1.01);

  for ( pxcStatus sts; cv::waitKey( 1 ) != ESC_KEY; )
  {
    sts = sm->AcquireFrame( false );
    
    if (sts < PXC_STATUS_NO_ERROR)
      break;

    sample = sm->QuerySample();
    PXCImage2CVMat(img, sample->color, PXCImage::PIXEL_FORMAT_RGB24);
    cvtColor(img, gray, CV_BGR2GRAY);
    gpuImg.upload(gray);

    gpuCascade->detectMultiScale(gpuImg, buffer);
    gpuCascade->convert(buffer, faces);

    for (Rect& face : faces)
      rectangle(img, face, RECT_COLOR);

    imshow( WINDOW_TITLE, img );
    sm->ReleaseFrame();
  }

  sm->Close();
}