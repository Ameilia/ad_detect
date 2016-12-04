#pragma once
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <cstring>
#include <cmath>
#include "gist-classifier.hpp"
#include "gist.h"
#include "standalone_image.h"
using namespace std;

const static int GIST_SIZE = 128;
const static int feature_vector_length = 960;
const static int nblocks = 4;
const static int n_scale = 3;
const static int orientations_per_scale[50] = { 8,8,4 };
const static int pos_label = 1;
const static int neg_label = 2;
static image_list_t *Gabor;

/* Displays image in window, which closes upon user keypress. */
void flash_image(IplImage* img, const char* window_name) {
  cvNamedWindow(window_name, CV_WINDOW_AUTOSIZE);
  cvShowImage(window_name, img);
  char c = cvWaitKey(0);
  cvDestroyWindow(window_name);
}

/* Convert OpenCV IplImage into LEAR's desired color_image_t format. */
/* Direct access using a c++ wrapper using code found under topic
"Simple and efficient access" at link:*/
/* http://www.cs.iit.edu/~agam/cs512/lect-notes/opencv-intro/opencv-intro.html#SECTION00053000000000000000*/
void convert_IplImage_to_LEAR(IplImage* src, color_image_t* dst) {
  assert(src->width == GIST_SIZE && src->height == GIST_SIZE);
  assert(src->depth == IPL_DEPTH_8U);
  RgbImage imgA(src);
  int x, y, i = 0;
  for (y = 0; y < GIST_SIZE; y++) {
    for (x = 0; x < GIST_SIZE; x++) {
      dst->c1[i] = imgA[y][x].r;
      dst->c2[i] = imgA[y][x].g;
      dst->c3[i] = imgA[y][x].b;
      i++;
    }
  }
  assert(i == (GIST_SIZE * GIST_SIZE));
}

/*
* Computes the GIST descriptor for a copy of img resized to 128x128
* (GIST_SIZE x GIST_SIZE).
*/
float* my_compute_gist(IplImage* img, IplImage* rsz) {
  /* Resize image to 128x128 before calculating GIST descriptor. */
  assert(img);
  assert(rsz);
  cvResize(img, rsz, CV_INTER_LINEAR);
  /* Lear's compute_gist takes a ppm and computes gist descriptor for it. */
  color_image_t *lear = color_image_new(GIST_SIZE, GIST_SIZE);
  assert(lear);
  convert_IplImage_to_LEAR(rsz, lear);
  assert(Gabor);
  float *desc = color_gist_scaletab(lear, nblocks, n_scale, orientations_per_scale);
  /* Cleanup. */
  color_image_delete(lear);
  return desc;
}

/* Counts the number non-empty lines in a file. */
int number_of_lines(const char* filename) {
  ifstream file(filename);
  assert(file);
  int n = 0;
  string line;
  while (getline(file, line)) {
    /* Ignore empty lines. */
    if (strlen(line.c_str()) != 0) {
      n++;
    }
  }
  return n;
}

std::vector<float> extractGistFeature(cv::Mat image) {
  IplImage image_copy = image;
  IplImage* new_image = &image_copy;
  cv::Mat image_resize;
  if (image.cols > image.rows) {
    image = image(cv::Rect((image.cols - image.rows) / 2, 0, image.rows, image.rows));
  }
  else {
    image = image(cv::Rect(0, (image.rows - image.cols) / 2, image.cols, image.cols));
  }
  cv::resize(image, image_resize, cv::Size(128, 128));
  cv::GaussianBlur(image_resize, image_resize, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
  IplImage image_copy_resize = image_resize;
  IplImage* new_image_resize = &image_copy_resize;
  color_image_t *lear = color_image_new(GIST_SIZE, GIST_SIZE);
  assert(lear);
  convert_IplImage_to_LEAR(new_image_resize, lear);
  float *desc = color_gist_scaletab(lear, nblocks, n_scale, orientations_per_scale);
  color_image_delete(lear);
  std::vector<float> desc_vec;
  desc_vec.reserve(960);
  for (int i = 0; i < 960; i++) {
    desc_vec.push_back(desc[i]);
  }
  return desc_vec;
}

/*
* For each image file named in imagelist_filename, extract the GIST
* descriptor (960-element vector) and add the vector into training
* matrix.  If savexml is provided, store the training matrix in
* XML-form to file specified by savexml.  This function assumes that
* the images in imagelist_filename are all "positive" or all
* "negative" samples.  If you want to mix "positive" and "negative"
* samples, you'll have to store the image labels in your own matrix
* and xml-file; this function only calculates the features (GIST
* descriptors).
*
* imagelist_file : a list of image filenames. path/foo.jpg \n
* path/bar.jpg, etc.
*
* savexml : file to save the OpenCV training matrix.
*/
CvMat* feature_vectors_img128x128(const char* imagelist_filename,
                                  char* savexml = NULL) {
  int number_samples = number_of_lines(imagelist_filename);
  CvMat* training = cvCreateMat(number_samples, feature_vector_length, CV_32FC1);
  CvMat row;
  int i = 0, row_index = 0;
  ifstream imagelist_file(imagelist_filename);
  assert(imagelist_file);
  string filename;
  IplImage *img, *gist_img;
  float *desc;

  printf("Beginning to extract GIST descriptors from %s images\n", imagelist_filename);
  while (getline(imagelist_file, filename)) {
    /* Ignore empty lines. */
    if (strlen(filename.c_str()) == 0) {
      continue;
    }
    img = cvLoadImage(filename.c_str());
    if (!img) {
      cout << "Error opening image file named: " << filename << "\n";
      assert(img);
    }
    gist_img = cvCreateImage(cvSize(GIST_SIZE, GIST_SIZE), IPL_DEPTH_8U, 3);
    desc = my_compute_gist(img, gist_img);
    /* Save descriptor in training matrix. */
    assert(row_index < number_samples);
    cvGetRow(training, &row, row_index);
    for (i = 0; i < feature_vector_length; i++) {
      cvSetReal1D(&row, i, desc[i]);
    }
    row_index++;

    /* Clean up descriptor. */
    free(desc);
  }
  assert(row_index == number_samples);

  if (savexml != NULL) {
    cvSave(savexml, training);
  }
  /* Clean up. */
  imagelist_file.close();
  return training;
}