#include <string>
#include <stdio.h>
#include "opencv2\opencv.hpp"

#include "ASiftDetector.h"
#include "..\GIST\gist-classifier.hpp"

#define ExtractGistFeature false

using namespace cv;
using namespace std;

Mat lookUpTable(1, 256, CV_8U);
const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.75f;   // Nearest neighbor matching ratio

ASiftDetector feature;

void MultiScaleTempalteMatching(Mat frame, Mat gray, Mat brand, double thresh = 0.6) {
  cvtColor(brand, brand, CV_BGR2GRAY);
  Mat result;
  Mat subgray = gray.clone();
  vector<Rect> detect_result;
  while (subgray.cols > brand.cols) {
    double scale = (double)subgray.cols / (double)frame.cols;
    matchTemplate(subgray, brand, result, CV_TM_CCOEFF_NORMED);
    double maxVal, minVal;
    Point maxPoint;
    minMaxLoc(result, &minVal, &maxVal, NULL, &maxPoint);
    if (maxVal > thresh) {
      Rect detected_rect = Rect(maxPoint.x / scale, maxPoint.y / scale, brand.cols / scale, brand.rows / scale);
      detect_result.push_back(detected_rect);
    }
    resize(subgray, subgray, Size(0, 0), 0.707, 0.707);
  }
  for (auto& r : detect_result) {
    rectangle(frame, r, Scalar(0, 255, 255), 3);
  }
  imshow("detect brand", frame);
}

void FindObject(Mat& frame, Mat& brand_image, vector<KeyPoint>& frame_keypoints, Mat& frame_descriptors, vector<KeyPoint>& brand_keypoints, Mat& brand_descriptors) {
  //feature->detect(gray,frame_keypoints);
  //feature->compute(gray, frame_keypoints, frame_descriptors);
  /*Ptr<DescriptorMatcher> flann = DescriptorMatcher::create("FlannBased");
  vector<DMatch> matches;
  flann->match(frame_descriptor, brand_descriptors, matches);*/
  BFMatcher matcher(NORM_L1);
  vector< vector<DMatch> > nn_matches;
  matcher.knnMatch(frame_descriptors, brand_descriptors, nn_matches, 2);
  vector<Point2f> matchedPoint1, matchedPoint2;
  vector<KeyPoint> matched1, matched2, inliers1, inliers2;
  vector<DMatch> good_matches;
  for (size_t i = 0; i < nn_matches.size(); i++) {
    DMatch first = nn_matches[i][0];
    float dist1 = nn_matches[i][0].distance;
    float dist2 = nn_matches[i][1].distance;

    if (dist1 < nn_match_ratio * dist2) {
      matched1.push_back(frame_keypoints[first.queryIdx]);
      matched2.push_back(brand_keypoints[first.trainIdx]);
      matchedPoint1.push_back(frame_keypoints[first.queryIdx].pt);
      matchedPoint2.push_back(brand_keypoints[first.trainIdx].pt);
    }
  }

  if (matchedPoint1.size() < 8) return;

  Mat homographyMask;
  Mat H = findHomography(matchedPoint2, matchedPoint1, homographyMask, CV_RANSAC, 5.0);

  for (int i = 0; i < homographyMask.rows; i++) {
    if (homographyMask.at<uchar>(0, i)) {
      inliers1.push_back(matched1[i]);
      inliers2.push_back(matched2[i]);
      good_matches.push_back(DMatch(inliers1.size() - 1, inliers2.size() - 1, 0));
    }
  }
  //cout << "inlier:" << inliers1.size() << "/" << matched1.size() << endl;
  if (good_matches.size() < 5) return;

  Mat res;
  //drawKeypoints(frame, frame_keypoints, frame);
  
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(brand_image.cols, 0);
  obj_corners[2] = cvPoint(brand_image.cols, brand_image.rows); obj_corners[3] = cvPoint(0, brand_image.rows);
  std::vector<Point2f> scene_corners(4);

  H = estimateRigidTransform(matchedPoint2, matchedPoint1, true);
  if (!H.data) return;

  transform(obj_corners, scene_corners, H);

  float min_side = 999999, max_side = 0;
  for (int i = 0; i < 4; i++) {
    if (i % 2) {
      min_side = (norm(scene_corners[i] - scene_corners[(i + 1) % 4]) < min_side) ? norm(scene_corners[i] - scene_corners[(i + 1) % 4]) : min_side;
    }
    else {
      max_side = (norm(scene_corners[i] - scene_corners[(i + 1) % 4]) > max_side) ? norm(scene_corners[i] - scene_corners[(i + 1) % 4]) : max_side;
    }
    if (scene_corners[i].x < -50 || scene_corners[i].x > frame.cols + 50 || scene_corners[i].y < -50 || scene_corners[i].y > frame.rows + 50) {
      return;
    }
  }
  if (min_side*2.5 < max_side || min_side > max_side) return;

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line(frame, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2);
  line(frame, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 2);
  line(frame, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 2);
  line(frame, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 2);
  drawMatches(frame, inliers1, brand_image, inliers2, good_matches, res);
  imshow("detect brand", res);
  if (good_matches.size() > 0) {
    waitKey(1);
  }
}

void getColorHistogram(Mat& image, vector<float>& histogram) {
  Mat LUVImage;
  cvtColor(image, LUVImage, CV_BGR2Lab);
  GaussianBlur(LUVImage, LUVImage, Size(5, 5), 0, 0, BORDER_DEFAULT);
  vector<Mat> bgr_planes;
  split(LUVImage, bgr_planes);

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 };
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
  calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
  calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound((double)hist_w / histSize);

  Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(b_hist, b_hist, 1, 0, NORM_L1, -1, Mat());
  normalize(g_hist, g_hist, 1, 0, NORM_L1, -1, Mat());
  normalize(r_hist, r_hist, 1, 0, NORM_L1, -1, Mat());

  /// push the histograms
  for (int i = 1; i < histSize; i++) {
    histogram.push_back(b_hist.at<float>(i));
  }
  for (int i = 1; i < histSize; i++) {
    histogram.push_back(g_hist.at<float>(i));
  }
  for (int i = 1; i < histSize; i++) {
    histogram.push_back(r_hist.at<float>(i));
  }
  cout << sum(Mat(histogram))[0] << endl;
}

float histogram_intersection(vector<float>& hist1, vector<float>& hist2) {
  assert(hist1.size() == hist2.size());
  float sum = 0;
  for (int i = 1; i < hist1.size(); i++) {
    sum += min(hist1[i], hist2[i]);
  }
  return sum;
}

float sum_sqr(vector<float>& hist1, vector<float>& hist2) {
  assert(hist1.size() == hist2.size());
  float sum = 0;
  for (int i = 1; i < hist1.size(); i++) {
    //sum += (hist1[i] - hist2[i]) * (hist1[i] - hist2[i]);
    sum += abs(hist1[i] - hist2[i]);
  }
  return sum;
}

int main(int argc, char* argv[]) {
  string rgb_filename = "C:\\Users\\Am\\Downloads\\dataset\\dataset\\Videos\\data_test1.rgb";
  int width = 480;
  int height = 270;
  double image_sample_rate = 30;
  FILE *IN_FILE;
  IN_FILE = fopen(rgb_filename.c_str(), "rb");
  if (IN_FILE == NULL) {
    fprintf(stderr, "Error Opening File for Reading");
    return false;
  }

  fseek(IN_FILE, width*height * 3 * image_sample_rate * 165, SEEK_SET);

  uchar* p = lookUpTable.data;
  for (int i = 0; i < 256; ++i)
    p[i] = (i / 16) * 16;

  Mat subway_image = imread("C:\\Users\\Am\\Downloads\\dataset\\dataset\\Brand Images\\subway_logo.bmp");
  //resize(subway_image, subway_image, Size(0, 0), 0.5, 0.5);
  Mat subway_gray;
  cvtColor(subway_image, subway_gray, CV_BGR2GRAY);
  //GaussianBlur(subway_gray, subway_gray, Size(5, 5), 0, 0, BORDER_DEFAULT);
  vector<KeyPoint> subway_keypoints;
  Mat subway_descriptors;
  feature.detectAndCompute(subway_gray, subway_keypoints, subway_descriptors);
  sqrt(subway_descriptors, subway_descriptors);//https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

  //Mat subway_show;
  //drawKeypoints(subway_image, subway_keypoints, subway_show);
  //imshow("subway", subway_show);
  //waitKey(0);

  Mat starbucks_image = imread("C:\\Users\\Am\\Downloads\\dataset\\dataset\\Brand Images\\starbucks_logo.bmp");
  //resize(starbucks_image, starbucks_image, Size(0, 0), 0.5, 0.5);
  Mat starbucks_gray;
  cvtColor(starbucks_image, starbucks_gray, CV_BGR2GRAY);
  //GaussianBlur(starbucks_gray, starbucks_gray, Size(5, 5), 0, 0, BORDER_DEFAULT);
  vector<KeyPoint> starbucks_keypoints;
  Mat starbucks_descriptors;
  feature.detectAndCompute(starbucks_gray, starbucks_keypoints, starbucks_descriptors);
  sqrt(starbucks_descriptors, starbucks_descriptors);//https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

  //resize(subway_image, subway_image, Size(0, 0), 0.2, 0.2);
  //LUT(subway_image, lookUpTable, subway_image);

  Mat prevgray, gray, flow, cflow, frame;
  Mat motion2color;
  int frame_count = 0;
  int shot_count = 0;

  Mat motionHist = Mat::zeros(200, 1000, CV_8UC3);

  double averageMag = 0.0;

  vector<vector<float>> color_histograms;

  while (!feof(IN_FILE)) {
    Mat_<Vec3b> frame_(height, width);
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          frame_(h, w)[2-c] = fgetc(IN_FILE);
        }
      }
    }
    Mat frame = frame_;
    frame_count++;
    shot_count++;
    if (frame_count % 30 == 0) {
      cout << frame_count / 30 << "s" << endl;
    }
    if (frame_count == 1) {
      vector<float> color_hist = extractGistFeature(frame);
      color_histograms.push_back(color_hist);
    }
    imwrite("frame.bmp", frame);
    cvtColor(frame, gray, CV_BGR2GRAY);
    vector<KeyPoint> frame_keypoints;
    Mat frame_descriptors;
    feature.detectAndCompute(gray, frame_keypoints, frame_descriptors);
    sqrt(frame_descriptors, frame_descriptors);//https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
    FindObject(frame, subway_image, frame_keypoints, frame_descriptors, subway_keypoints, subway_descriptors);
    FindObject(frame, starbucks_image, frame_keypoints, frame_descriptors, starbucks_keypoints, starbucks_descriptors);
    resize(gray, gray, Size(0, 0), 0.25, 0.25);

    if (prevgray.data) {
      //double t = (double)cvGetTickCount();
      GaussianBlur(gray, gray, Size(5, 5), 0, 0, BORDER_DEFAULT);
      calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
      //t = (double)cvGetTickCount() - t;
      //cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
      vector<Mat> xyflow;
      split(flow, xyflow);
      Mat xflowdx, xflowdy, yflowdx, yflowdy;
      Sobel(xyflow[0], xflowdx, xyflow[0].depth(), 1, 0, 5);
      Sobel(xyflow[0], xflowdy, xyflow[0].depth(), 0, 1, 5);
      Sobel(xyflow[1], yflowdx, xyflow[1].depth(), 1, 0, 5);
      Sobel(xyflow[1], yflowdy, xyflow[1].depth(), 0, 1, 5);
      Mat flowMagnitude = abs(xflowdx) + abs(xflowdy) + abs(yflowdx) + abs(yflowdy);
      double meanMagnitude = mean(flowMagnitude)[0];
      //cout << meanMagnitude << endl;
      if (meanMagnitude > averageMag * 3 && shot_count > image_sample_rate) {
        cout << "new shot." << endl;
        shot_count = 0;
#if ExtractGistFeature
        vector<float> color_hist = extractGistFeature(frame);
        color_histograms.push_back(color_hist);
        cout << "hist distance:" << endl;
        for (int i = 0; i < color_histograms.size()-1; i++) {
          cout << sum_sqr(color_histograms[i], color_hist) << " ";
        }
        cout << endl;
#endif
      }
      if (averageMag == 0) averageMag = meanMagnitude;
      else averageMag = averageMag *0.9 + meanMagnitude*0.1;
      if (frame_count % 1000 == 0) motionHist = Mat::zeros(motionHist.size(), motionHist.type());
      motionHist(Range(max(200 - meanMagnitude, 0.0), 200), Range(frame_count % 1000, frame_count % 1000+1)) = Scalar(255,255,255);
      motionHist.at<cv::Vec3b>(Point(frame_count % 1000, max(200 - averageMag, 0.0))) = Vec3b(0,0,255);

      imshow("video", frame);
      imshow("motion hist", motionHist);
      waitKey(1);//int(1000 / image_sample_rate)
    }

    
    std::swap(prevgray, gray);
  }
}