#include "wchar.h"
#include "opencv2/opencv.hpp"
#include "pxcsession.h"
#include "pxcsensemanager.h"
#include "pxccapture.h"

cv::Mat PXCImage2CVMat(PXCImage *pxcImage, PXCImage::PixelFormat format)
{
    PXCImage::ImageData data;
    PXCImage::ImageInfo info;
    pxcImage->AcquireAccess(PXCImage::ACCESS_READ, format, &data);
    info = pxcImage->QueryInfo();

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

    cv::Mat ocvImage = cv::Mat(cv::Size(width, height), type, data.planes[0]);
    pxcImage->ReleaseAccess(&data);

    return ocvImage;
}


int main()
{
  PXCSenseManager * sm = PXCSenseManager::CreateInstance();
  PXCCapture::Sample * sample = nullptr;
  cv::Mat img;

  sm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, 640, 480, 60);
  sm->Init();

  for ( pxcStatus sts; cv::waitKey( 1 ) != 27; )
  {
    sts = sm->AcquireFrame(false);
    
    if (sts < PXC_STATUS_NO_ERROR)
      break;

    sample = sm->QuerySample();
    img = PXCImage2CVMat(sample->color, PXCImage::PIXEL_FORMAT_RGB24);
    cv::imshow( "RealSense Test", img );

    sm->ReleaseFrame();
  }

  sm->Close();
}