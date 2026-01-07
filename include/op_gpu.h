#pragma once
#include "common_include.h"
#include "utils.h"
#include <cmath>

#define CHECK(x) cuda_check(x, __FILE__, __LINE__)

void cuda_check(cudaError_t state, std::string file, int line);

void affine_bilinear(unsigned char* src, const int src_w, const int src_h, float* dst, const int dst_w, const int dst_h,
                    const AffineMat matrix);

void conf_filter_kernel(float* src, int src_box_width, int src_box_num, float* dst, int dst_box_width, int topK, int num_class,
                        float conf_thresh);

void nms_fast(float* src, int src_box_width, int topK, float iou_thresh);