#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "gco-v3.0-master/GCoptimization.h"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    //Start clock
    long start = clock();
    srand(time(NULL)); //Seed to get randmon patches

    //Create Gaussian-Mixture-Models
    //EM model;
    Mat labels;

    //Open another image
    Mat image1, image2, patch, newimg, source ;
    source = imread("FA.jpg");
    //image2 = imread("circle.jpg");


     //ouput images
    cv::Mat meanImg(source.rows, source.cols, CV_32FC3);
    cv::Mat fgImg(source.rows, source.cols, CV_8UC3);
    cv::Mat bgImg(source.rows, source.cols, CV_8UC3);

    //convert the input image to float
    cv::Mat floatSource;
    source.convertTo(floatSource, CV_32F);

    //now convert the float image to column vector
    cv::Mat samples(source.rows * source.cols, 3, CV_32FC1);
    int idx = 0;
    for (int y = 0; y < source.rows; y++) {
        cv::Vec3f* row = floatSource.ptr<cv::Vec3f > (y);
        for (int x = 0; x < source.cols; x++) {
            samples.at<cv::Vec3f > (idx++, 0) = row[x];
        }
    }

    //we need just 2 clusters
    Ptr<ml::EM> em = ml::EM::create();
    em->setClustersNumber(2);
    em->trainEM( samples, noArray(), labels, noArray() );

    //the two dominating colors
    cv::Mat means = em->getMeans();
    //the weights of the two dominant colors
    cv::Mat weights = em->getWeights();

    //we define the foreground as the dominant color with the largest weight
    const int fgId = weights.at<float>(0) > weights.at<float>(1) ? 0 : 1;

    //now classify each of the source pixels
    idx = 0;
    for (int y = 0; y < source.rows; y++) {
        for (int x = 0; x < source.cols; x++) {

            //classify
            const int result = cvRound(em->predict2(samples.row(idx++), noArray() )[1]);
            //get the according mean (dominant color)
            const double* ps = means.ptr<double>(result, 0);

            //set the according mean value to the mean image
            float* pd = meanImg.ptr<float>(y, x);
            //float images need to be in [0..1] range
            pd[0] = ps[0] / 255.0;
            pd[1] = ps[1] / 255.0;
            pd[2] = ps[2] / 255.0;

            //set either foreground or background
            if (result == fgId) {
                fgImg.at<cv::Point3_<uchar> >(y, x, 0) = source.at<cv::Point3_<uchar> >(y, x, 0);
            } else {
                bgImg.at<cv::Point3_<uchar> >(y, x, 0) = source.at<cv::Point3_<uchar> >(y, x, 0);
            }
        }
    }

    cv::imshow("original Image", source);
    cv::imshow("Means", meanImg);
    cv::imshow("Foreground", fgImg);
    cv::imshow("Background", bgImg);



    waitKey();
    return 0;
        
}
