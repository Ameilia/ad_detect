#include <string>
#include <stdio.h>
#include "opencv2\opencv.hpp"
#include "opencv2\shape.hpp"

#include "ASiftDetector.h"
#include "..\GIST\gist-classifier.hpp"

#define ExtractGistFeature false

using namespace cv;
using namespace std;

Mat lookUpTable(1, 256, CV_8U);
const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.7f;   // Nearest neighbor matching ratio

ASiftDetector feature;

map<string, double> thresholds;
map<string, vector<vector<Point>>> shape_context_descriptors;
map<string, vector<Point>> biggest_shape;
map<string, vector<float>> shape_histogram;
map<string, int> all_contour_area;
Ptr<ShapeDistanceExtractor> shape_context_extractor;

float histogram_intersection(vector<float>& hist1, vector<float>& hist2) {
  assert(hist1.size() == hist2.size());
  float sum = 0;
  for (int i = 1; i < hist1.size(); i++) {
    sum += min(hist1[i], hist2[i]);
  }
  return sum;
}

float histogram_sum_sqr(vector<float>& hist1, vector<float>& hist2) {
  assert(hist1.size() == hist2.size());
  float sum = 0;
  for (int i = 1; i < hist1.size(); i++) {
    //sum += (hist1[i] - hist2[i]) * (hist1[i] - hist2[i]);
    sum += abs(hist1[i] - hist2[i]);
  }
  return sum;
}

void getShapeHistogram(vector<Point>& shape, vector<float>& histogram, int dim = 100) {
  vector<vector<Point>> contours;
  contours.push_back(shape);
  Rect bb = boundingRect(shape);
  Mat contour_image = Mat::zeros(bb.size(), CV_8UC1);
  drawContours(contour_image, contours, 0, Scalar(255), -1, 8, noArray(), INT_MAX, Point(-bb.x, -bb.y));
  resize(contour_image, contour_image, Size(dim, dim));
  histogram.resize(dim * 2);
  for (int i = 0; i < dim * 2; i++) histogram[i] = 0;
  for (int h = 0; h < dim; h++) {
    for (int w = 0; w < dim; w++) {
      histogram[w] += (double)contour_image.at<uchar>(h, w) / 255.0;
      //histogram[dim + h] += (double)contour_image.at<uchar>(h, w) / 255.0;
    }
  }
  float sum = 0;
  for (int i = 0; i < dim * 2; i++) sum += histogram[i];
  for (int i = 0; i < dim * 2; i++) histogram[i] /= sum;
  //imshow("contour", contour_image);
  //waitKey(0);
}

void ShapeMatchingSingle(Mat frame, vector<vector<Point>>& shape1, string brand_name = "subway", double threshold = 0.8) {
  int shape_detected = 0;
  for (int i = 0; i < shape1.size();i++) {
    if (shape1[i].size() < 100) continue;
    //float distance = shape_context_extractor->computeDistance(shape1[i], biggest_shape[brand_name]);
    //float distance = matchShapes(shape1[i], biggest_shape[brand_name], CV_CONTOURS_MATCH_I2, 0.0);
    vector<float> shape_hist;
    getShapeHistogram(shape1[i], shape_hist);
    float distance = histogram_intersection(shape_hist, shape_histogram[brand_name]);
    if (distance > threshold) {
      shape_detected++;
      rectangle(frame, boundingRect(shape1[i]), Scalar(0, 0, 255), 3);
      //try {
      //  drawContours(frame, shape1, i, Scalar(0, 0, 255), 1, -1);
      //}
      //catch (...) {
      //}
    }
  }
  if (shape_detected > 0) {
    imshow("detected shapes", frame);
    waitKey(1);
  }
}

void ShapeMatching(Mat frame, vector<vector<Point>>& shape1, string brand_name = "subway", double threshold = 0.2) {
  if (shape1.size() > 20) return;
  auto shape2 = shape_context_descriptors[brand_name];
  vector<vector<Point>> matched_shapes1, matched_shapes2;
  for (auto& s1 : shape1) {
    for (auto& s2 : shape2) {
      if (s1.size() < 10 || s2.size() < 10 || s1.size() > 500) continue;
      //double t = (double)cvGetTickCount();
      float distance = shape_context_extractor->computeDistance(s1, s2);
      //t = (double)cvGetTickCount() - t;
      //cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
      if (distance < threshold) {
        matched_shapes1.push_back(s1);
        matched_shapes2.push_back(s2);
      }
    }
  }
  cout << brand_name << " matched " << matched_shapes1.size() << endl;
  int matched_area = 0;
  for (auto& ms : matched_shapes2) {
    matched_area += contourArea(ms);
  }
  if (matched_area > all_contour_area[brand_name] * 0.5) {
    for (int i = 0; i < matched_shapes1.size(); i++) {
      try {
        drawContours(frame, matched_shapes1, i, Scalar(0, 0, 255), 1, -1);
      }
      catch (...) {

      }
    }
    imshow("blob found", frame);
    waitKey(1);
  }
}

void getColorFilters(Mat image, vector<Vec2f>& colors, string brand_name="subway") {
  Mat lab_image;
  cvtColor(image, lab_image, CV_BGR2Lab);
  vector<Mat> channels;
  split(lab_image, channels);
  Mat ab_image;
  vector<Mat> ab_channels;
  ab_channels.push_back(channels[1]);
  ab_channels.push_back(channels[2]);
  merge(ab_channels, ab_image);
  Mat float_image;
  ab_image.convertTo(float_image, CV_32FC2);
  //imshow("a_image", channels[1]);
  //imshow("b_image", channels[2]);
  Mat image_for_flood = image.clone();
  Mat mask = Mat::zeros(image.rows+2, image.cols+2,CV_8UC1);
  floodFill(image_for_flood, mask, Point(10, 10), Scalar(0, 0, 0), NULL, Scalar(10,10,10), Scalar(20,20,20));
  floodFill(image_for_flood, mask, Point(image.cols - 10, 10), Scalar(0, 0, 0), NULL, Scalar(10, 10, 10), Scalar(100, 100, 100));
  mask = mask(Rect(1, 1, image.cols, image.rows));
  mask = 1- mask;
  Vec3b* image_data = (Vec3b*)image.data;
  for (int w = 0; w < image.cols; w++) {
    for (int h = 0; h < image.rows; h++) {
      Vec3b pixel = image.at<Vec3b>(h, w);
      if (pixel[0] > 220 && pixel[1] > 220 && pixel[2] > 220) {
        mask.at<uchar>(h, w) = 0;
      }
    }
  }
  int fg_count = sum(mask)[0];
  mask *= 255;
  
  Mat image_for_kmeans = Mat(fg_count, 1, CV_32FC2);
  int p = 0;
  for (int w = 0; w < image.cols; w++) {
    for (int h = 0; h < image.rows; h++) {
      uchar m = mask.at<uchar>(h, w);
      if (m == 255) {
        image_for_kmeans.at<Vec2f>(p++) = float_image.at<Vec2f>(h, w);
      }
    }
  }
  assert(p == fg_count);
  
  Mat kmeans_labels;
  TermCriteria criteria;
  criteria.epsilon = 1e-3;
  criteria.maxCount = 1000;
  Mat kmeans_centers;
  kmeans(image_for_kmeans, 3, kmeans_labels, criteria,10, KMEANS_PP_CENTERS, kmeans_centers);
  int center_count[5] = { 0 };
  for (int i = 0; i < kmeans_labels.rows; i++) {
    center_count[kmeans_labels.at<int>(i)]++;
  }
  for (int i = 0; i < 3; i++) {
    cout << center_count[i] << " ";
    if (center_count[i] > 1000) {
      colors.push_back(Vec2f(kmeans_centers.at<float>(i,0), kmeans_centers.at<float>(i, 1)));
    }
  }
  cout << endl;
  cout << kmeans_centers << endl;
  cout << Mat(colors) << endl;
  vector<vector<Point> > contour2D;
  findContours(mask, contour2D, RETR_LIST, CHAIN_APPROX_NONE);
  shape_context_descriptors[brand_name] = contour2D;
  int area = 0;
  int max_area = 0, max_area_index = 0;
  for (int i = 0; i < contour2D.size();i++) {
    int a = contourArea(contour2D[i]);
    area += a;
    if (a > max_area) max_area = a, max_area_index = i;
  }
  all_contour_area[brand_name] = area;
  biggest_shape[brand_name] = contour2D[max_area_index];
  vector<float> brand_contour_hist;
  getShapeHistogram(contour2D[max_area_index], brand_contour_hist);
  shape_histogram[brand_name] = brand_contour_hist;
  
  //for (int i = 0; i < contour2D.size();i++) {
  //  Mat clean_image = Mat::zeros(image.size(), image.type());
  //  drawContours(clean_image, contour2D, i, Scalar(0, 0, 255), 2, -1);
  //  imshow("contour", clean_image);
  //  waitKey(0);
  //}
  //imshow("image", image);
  //imshow("mask", mask);
  //for (auto c : colors) {
  //  Mat l = Mat(image.size(), CV_8UC1, Scalar(255));
  //  Mat a = Mat(image.size(), CV_8UC1, Scalar(c[0]));
  //  Mat b = Mat(image.size(), CV_8UC1, Scalar(c[1]));
  //  Mat lab,rgb;
  //  merge(vector<Mat>{ a,b,l }, lab);
  //  lab.convertTo(rgb, CV_Lab2BGR);
  //  imshow("color", rgb);
  //  waitKey(0);
  //}
}

void ColorFiltering(Mat frame, vector<Vec2f>& colors, string window_name = "subway") {
  //GaussianBlur(frame, frame, Size(5, 5), 0, 0, BORDER_DEFAULT);
  Mat lab_frame;
  cvtColor(frame, lab_frame, CV_BGR2Lab);
  vector<Mat> channels;
  split(lab_frame, channels);
  Mat ab_frame;
  vector<Mat> ab_channels;
  ab_channels.push_back(channels[1]);
  ab_channels.push_back(channels[2]);
  merge(ab_channels, ab_frame);
  Mat float_frame;
  ab_frame.convertTo(float_frame, CV_32FC2);
  Mat foreground = Mat::zeros(frame.size(), CV_8UC1);
  int color_count = colors.size();
  int fg_count = 0;
  for (int w = 0; w < frame.cols; w++) {
    for (int h = 0; h < frame.rows; h++) {
      for (int c = 0; c < color_count; c++) {
        //if (max(abs(float_frame.at<Vec2f>(h, w)[0] - colors[c][0]), abs(float_frame.at<Vec2f>(h, w)[1] - colors[c][1])) < threshold) {
        if (norm(float_frame.at<Vec2f>(h, w) - colors[c]) < thresholds[window_name]) {
          foreground.at<uchar>(h, w) = 255;
          fg_count++;
        }
      }
    }
  }
  
  cv::Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(7,13));
  morphologyEx(foreground, foreground, cv::MORPH_CLOSE, element);
  //cv::Mat element2 = cv::getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
  //morphologyEx(foreground, foreground, cv::MORPH_DILATE, element);
  cv::imshow(window_name, foreground);
  cv::waitKey(1);
  vector<vector<Point> > contour2D;
  findContours(foreground, contour2D, RETR_LIST, CHAIN_APPROX_NONE);
  if(window_name == "mcdonald") ShapeMatchingSingle(frame, contour2D, window_name);
  
  if (fg_count > frame.rows * frame.cols / 10) {
    thresholds[window_name] -= 1.0;
  }
  if (fg_count < frame.rows * frame.cols / 50) {
    thresholds[window_name] += 1.0;
  }
  //cout << window_name << ":" << thresholds[window_name] << endl;
}

void MultiScaleTempalteMatching(Mat frame, Mat gray, Mat brand, double thresh = 0.6) {
  Mat brand_x, brand_y, brand_grad;
  Sobel(brand, brand_x, brand.depth(), 1, 0, 3);
  Sobel(brand, brand_y, brand.depth(), 0, 1, 3);
  brand_grad = abs(brand_x) + abs(brand_y);
  //GaussianBlur(brand_grad, brand_grad, Size(3, 3), 0, 0, BORDER_DEFAULT);
  //cvtColor(brand, brand, CV_BGR2GRAY);
  Mat result;
  Mat subgray = gray.clone();
  vector<Rect> detect_result;
  while (subgray.cols >= brand.cols) {
    double scale = (double)subgray.cols / (double)frame.cols;
    Mat gray_x, gray_y, gray_grad;
    Sobel(subgray, gray_x, subgray.depth(), 1, 0, 5);
    Sobel(subgray, gray_y, subgray.depth(), 0, 1, 5);
    gray_grad = abs(gray_x) + abs(gray_y);
    //GaussianBlur(gray_grad, gray_grad, Size(3, 3), 0, 0, BORDER_DEFAULT);
    
    matchTemplate(gray_grad, brand_grad, result, CV_TM_CCOEFF_NORMED);
    double maxVal, minVal;
    Point maxPoint;
    minMaxLoc(result, &minVal, &maxVal, NULL, &maxPoint);
    if (maxVal > thresh) {
      Rect detected_rect = Rect(maxPoint.x / scale, maxPoint.y / scale, brand.cols / scale, brand.rows / scale);
      detect_result.push_back(detected_rect);
    }
    resize(subgray, subgray, Size(0, 0), sqrt(sqrt(0.5)), sqrt(sqrt(0.5)));
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
  BFMatcher matcher(NORM_L2);
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

  perspectiveTransform(obj_corners, scene_corners, H);
  if (H.at<double>(0, 0) < 0.1 || H.at<double>(1, 1) < 0.1)return;
  if (H.at<double>(0, 0) > H.at<double>(1, 1) * 4 || H.at<double>(0, 0) * 4 < H.at<double>(1, 1)) return;
  float min_side = 999999, max_side = 0;
  for (int i = 0; i < 4; i++) {
    if (scene_corners[i].x < -50 || scene_corners[i].x > frame.cols + 50 || scene_corners[i].y < -50 || scene_corners[i].y > frame.rows + 50) {
      return;
    }
  }
  line(frame, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2);
  line(frame, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 2);
  line(frame, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 2);
  line(frame, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 2);
  drawMatches(frame, inliers1, brand_image, inliers2, good_matches, res);
  //cout << H << endl;
  imshow("detect brand1", res);
  if (good_matches.size() > 0) {
    waitKey(1);
  }

  //H = estimateRigidTransform(matchedPoint2, matchedPoint1, true);
  //if (!H.data) return;

  //transform(obj_corners, scene_corners, H);
  //if (H.at<double>(0, 0) < 0.1 || H.at<double>(1, 1) < 0.1)return;
  //if (H.at<double>(0, 0) > H.at<double>(1, 1) * 4 || H.at<double>(0, 0) * 4 < H.at<double>(1, 1)) return;

  //float min_side = 999999;
  //float max_side = 0;
  //for (int i = 0; i < 4; i++) {
  //  if (scene_corners[i].x < -50 || scene_corners[i].x > frame.cols + 50 || scene_corners[i].y < -50 || scene_corners[i].y > frame.rows + 50) {
  //    return;
  //  }
  //}

  ////-- Draw lines between the corners (the mapped object in the scene - image_2 )
  //line(frame, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2);
  //line(frame, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 2);
  //line(frame, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 2);
  //line(frame, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 2);
  //drawMatches(frame, inliers1, brand_image, inliers2, good_matches, res);
  //imshow("detect brand", res);
  //if (good_matches.size() > 0) {
  //  waitKey(1);
  //}
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

int main(int argc, char* argv[]) {
  //string rgb_filename = "D:\\jy\\mm\\project\\dataset\\Videos\\data_test1.rgb";
  string rgb_filename = "D:\\jy\\mm\\project\\dataset2\\Videos\\data_test2.rgb";
  int width = 480;
  int height = 270;
  double image_sample_rate = 30;
  FILE *IN_FILE;
  IN_FILE = fopen(rgb_filename.c_str(), "rb");
  if (IN_FILE == NULL) {
    fprintf(stderr, "Error Opening File for Reading");
    return false;
  }

  fseek(IN_FILE, width*height * 3 * image_sample_rate * 142, SEEK_SET);//no bigger than 180, it will overflow. Try multiply fseek.

  uchar* p = lookUpTable.data;
  for (int i = 0; i < 256; ++i)
    p[i] = (i / 16) * 16;

  shape_context_extractor = cv::createHausdorffDistanceExtractor();//cv::createShapeContextDistanceExtractor();

  //Mat subway_image = imread("D:\\jy\\mm\\project\\dataset\\Brand Images\\subway_logo.bmp");
  Mat subway_image = imread("D:\\jy\\mm\\project\\dataset2\\Brand Images\\Mcdonalds_logo.jpg");
  vector<Vec2f> subway_colors;
  getColorFilters(subway_image, subway_colors,"mcdonald");
  resize(subway_image, subway_image, Size(0, 0), 0.25, 0.25);
  Mat subway_gray;
  //Mat lab_subway;
  //cvtColor(subway_image, lab_subway, CV_BGR2Lab);
  //vector<Mat> channels;
  //split(lab_subway, channels);
  //subway_gray = channels[0];
  cvtColor(subway_image, subway_gray, CV_BGR2GRAY);
  //GaussianBlur(subway_gray, subway_gray, Size(5, 5), 0, 0, BORDER_DEFAULT);
  vector<KeyPoint> subway_keypoints;
  Mat subway_descriptors;
  feature.detectAndCompute(subway_gray, subway_keypoints, subway_descriptors);
  //sqrt(subway_descriptors, subway_descriptors);//https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

  //Mat subway_show;
  //drawKeypoints(subway_image, subway_keypoints, subway_show);
  //imshow("subway", subway_show);
  //waitKey(0);

  //Mat starbucks_image = imread("D:\\jy\\mm\\project\\dataset\\Brand Images\\starbucks_logo.bmp");
  Mat starbucks_image = imread("D:\\jy\\mm\\project\\dataset2\\Brand Images\\nfl_logo.bmp");
  //resize(starbucks_image, starbucks_image, Size(0, 0), 0.0625, 0.0625);
  vector<Vec2f> starbucks_colors;
  getColorFilters(starbucks_image, starbucks_colors,"nfl");
  Mat starbucks_gray;
  cvtColor(starbucks_image, starbucks_gray, CV_BGR2GRAY);
  //GaussianBlur(starbucks_gray, starbucks_gray, Size(5, 5), 0, 0, BORDER_DEFAULT);
  vector<KeyPoint> starbucks_keypoints;
  Mat starbucks_descriptors;
  feature.detectAndCompute(starbucks_gray, starbucks_keypoints, starbucks_descriptors);
  //sqrt(starbucks_descriptors, starbucks_descriptors);//https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

  //resize(subway_image, subway_image, Size(0, 0), 0.2, 0.2);
  //LUT(subway_image, lookUpTable, subway_image);

  thresholds["mcdonald"] = 50.0;
  thresholds["nfl"] = 50.0;

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
    ColorFiltering(frame, subway_colors, "mcdonald");
    ColorFiltering(frame, starbucks_colors,"nfl");
    double t = (double)cvGetTickCount();
    //Mat lab_frame;
    //cvtColor(frame, lab_frame, CV_BGR2Lab);
    //vector<Mat> frame_channels;
    //split(lab_frame, frame_channels);
    //gray = frame_channels[0];
    cvtColor(frame, gray, CV_BGR2GRAY);
    vector<KeyPoint> frame_keypoints;
    Mat frame_descriptors;
    feature.detectAndCompute(gray, frame_keypoints, frame_descriptors);
    FindObject(frame, subway_image, frame_keypoints, frame_descriptors, subway_keypoints, subway_descriptors);
    FindObject(frame, starbucks_image, frame_keypoints, frame_descriptors, starbucks_keypoints, starbucks_descriptors);
    t = (double)cvGetTickCount() - t;
    cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
    //MultiScaleTempalteMatching(frame, gray, subway_gray);
    //MultiScaleTempalteMatching(frame, gray, starbucks_gray);

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