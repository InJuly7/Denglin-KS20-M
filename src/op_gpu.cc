#include "op_gpu.h"

void cuda_check(cudaError_t state, std::string file, int line) {
    if (cudaSuccess != state) {
        std::cout << "CUDA Error code num is:" << state;
        std::cout << "CUDA Error:" << cudaGetErrorString(state);
        std::cout << "Error location:" << file << ": " << line;
        abort();
    }
}

inline __device__ __host__ int iDivUp(int a, int b) { return (a + b - 1) / b; }

__device__ float3 uchar3_to_float3(uchar3 v) { return make_float3(v.x, v.y, v.z); }

__device__ bool in_bounds(int x, int y, int cols, int rows) { return (x >= 0 && x < cols && y >= 0 && y < rows); }

__device__ float3 operator*(float value0, float3 value1) {
    float3 result;
    result.x = value0 * value1.x;
    result.y = value0 * value1.y;
    result.z = value0 * value1.z;
    return result;
}

__device__ float3 operator+(float value0, float3 value1) {
    float3 result;
    result.x = value0 + value1.x;
    result.y = value0 + value1.y;
    result.z = value0 + value1.z;
    return result;
}

__device__ float3 operator+=(float3& value0, const float3& value1) {
    value0.x += value1.x;
    value0.y += value1.y;
    value0.z += value1.z;
    return value0;
}

__device__ void affine_project_kernel(const AffineMat* matrix, int x, int y, float* proj_x, float* proj_y) {
    *proj_x = matrix->v0 * x + matrix->v1 * y + matrix->v2;
    *proj_y = matrix->v3 * x + matrix->v4 * y + matrix->v5;
}

__global__ void affine_bilinear_kernel(unsigned char* src, const int src_w, const int src_h, float* dst, const int dst_w,
                                            const int dst_h, const AffineMat matrix, const float3 paddingValue, const float3 alpha,
                                            const float3 beta) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) {
        return;
    }
    float2 src_xy = make_float2(0.0f, 0.0f);
    affine_project_kernel(&matrix, x, y, &src_xy.x, &src_xy.y);

    int src_x0 = __float2int_rd(src_xy.x);
    int src_y0 = __float2int_rd(src_xy.y);
    int src_x1 = src_x0 + 1;
    int src_y1 = src_y0 + 1;

    float wx0 = src_x1 - src_xy.x;  // hx
    float wx1 = src_xy.x - src_x0;  // lx
    float wy0 = src_y1 - src_xy.y;  // hy
    float wy1 = src_xy.y - src_y0;  // ly

    float3 src_value0, src_value1, value0;
    bool flag0 = in_bounds(src_x0, src_y0, src_w, src_h);
    bool flag1 = in_bounds(src_x1, src_y0, src_w, src_h);
    bool flag2 = in_bounds(src_x0, src_y1, src_w, src_h);
    bool flag3 = in_bounds(src_x1, src_y1, src_w, src_h);

    uchar3* input = (uchar3*)(src + src_y0 * src_w * 3);
    // paddingValue = 114
    src_value0 = flag0 ? uchar3_to_float3(input[src_x0]) : paddingValue;
    src_value1 = flag1 ? uchar3_to_float3(input[src_x1]) : paddingValue;
    value0 = wx0 * wy0 * src_value0;   // hx * hy = w1
    value0 += wx1 * wy0 * src_value1;  // lx * hy = w2

    input = (uchar3*)(src + src_y1 * src_w * 3);
    src_value0 = flag2 ? uchar3_to_float3(input[src_x0]) : paddingValue;
    src_value1 = flag3 ? uchar3_to_float3(input[src_x1]) : paddingValue;
    value0 += wx0 * wy1 * src_value0;  // hx * ly = w3
    value0 += wx1 * wy1 * src_value1;  // lx * ly = w4
    value0 = 0.5f + value0;

    // bgr to rgb
    float3 sum;
    sum.x = __float2int_rd(value0.z);
    sum.y = __float2int_rd(value0.y);
    sum.z = __float2int_rd(value0.x);

    // Normalize, hwc to chw
    float* output = (float*)dst + y * dst_w + x;
    output[0] = sum.x * alpha.x + beta.x;
    output[dst_w * dst_h] = sum.y * alpha.y + beta.y;
    output[2 * dst_w * dst_h] = sum.z * alpha.z + beta.z;
}

__global__ void conf_filter_kernel(float* src, int src_box_width, int src_box_num, float* dst, int dst_box_width, int topK,
                                          int num_class, float conf_threshold) {
    if (dst[0] >= topK) {
        return;
    }
    int box_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (box_idx >= src_box_num) {
        return;
    }
    float* pitem = src + box_idx * src_box_width;
    float* class_confidence = pitem + 4;
    // 第一个类别的概率 confidence, label = 0
    float confidence = *class_confidence++;
    int label = 0;
    // Get Max {80 class probability} and chass
    for (int i = 1; i < num_class; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }
    if (confidence < conf_thresh) {
        return;
    }
    // 原子操作记录有效框数量: dst的第一个元素用于存储有效框的数量
    int index = atomicAdd(dst, 1);
    if (index >= topK) {
        return;
    }

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;

    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    float* pout_item = dst + 1 + index * dst_box_width;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;
    // 116 = 84+32； 39 = 7+32
    memcpy(pout_item, pitem + num_class, 32 * sizeof(float));
}

__global__ void nms_fast_kernel(float* src, int src_box_width, int topK, float iou_threshold) {
    int box_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int count = min(int(src[0]), topK);
    if (box_idx >= count) {
        return;
    }
    float* pcurrent = src + 1 + box_idx * src_box_width;
    // 每个框 都需要与其余的框做IOU 操作
    // x1, y1, x2, y2, score, class, valid_flag
    for (int i = 0; i < count; ++i) {
        float* pitem = src + 1 + i * src_box_width;
        // 校验是否相同类别
        if (i == box_idx || pcurrent[5] != pitem[5]) continue;

        if (pitem[4] >= pcurrent[4]) {
            // if (pitem[4] == pcurrent[4] && i < box_idx) continue;
            float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);
            if (iou > iou_threshold) {
                // 小概率的框 被NMS 滤掉
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

// Resize_padding + Normalize + BGR->RGB + HWC->CHW
void affine_bilinear(unsigned char* src, const int src_w, const int src_h, float* dst, const int dst_w, const int dst_h,
                          const AffineMat matrix) {
    int block_size = 16;
    const dim3 block(block_size, block_size);
    const dim3 grid(iDivUp(dst_w, block_size), iDivUp(dst_h, block_size));
    const float3 paddingValue = make_float3(114.0f, 114.0f, 114.0f);
    const float3 alpha = make_float3(1.0 / 255.0f, 1.0 / 255.0f, 1.0 / 255.0f);
    const float3 beta = make_float3(0.0f, 0.0f, 0.0f);
    affine_bilinear_kernel<<<grid, block>>>(src, src_w, src_h, dst, dst_w, dst_h, matrix, paddingValue, alpha, beta);
}

// 置信度过滤
void conf_filter(float* src, int src_box_width, int src_box_num, float* dst, int dst_box_width, int topK, int num_class,
                        float conf_thresh) {
    std::cout << "Debug decodeDevice Start:" << std::endl;
    int block_size = 256;
    dim3 block(block_size);
    dim3 grid(iDivUp(src_box_num, block_size));
    // 第一个标志位 记录有效框数量：
    // int dstArea = 1 + dst_box_width * topK;
    std::cout << "	num_class: " << num_class << std::endl;
    std::cout << "	conf_thresh: " << conf_thresh << std::endl;
    std::cout << "	src_box_num: " << src_box_num << std::endl;
    std::cout << "	src_box_num: " << src_box_num << std::endl;
    std::cout << "	dst_box_width: " << dst_box_width << std::endl;
    std::cout << "	topK: " << topK << std::endl;
    conf_filter_device_kernel<<<grid, block>>>(src, src_box_width, src_box_num, dst, dst_box_width, topK, num_class, conf_thresh);
    std::cout << "Debug decodeDevice Ends:" << std::endl;
}

// 非极大值抑制
void nms_fast(float* src, int src_box_width, int topK, float iou_thresh) {
    std::cout << "Debug nmsDeviceV1 Start:" << std::endl;
    int block_size = 128;
    dim3 block(block_size);
    dim3 grid(iDivUp(topK, block_size));
    std::cout << "	param.topK: " << param.topK << std::endl;
    std::cout << "	param.batch_size: " << param.batch_size << std::endl;
    std::cout << "	param.iou_thresh: " << param.iou_thresh << std::endl;
    std::cout << "	srcWidth: " << srcWidth << std::endl;
    std::cout << "	srcHeight: " << srcHeight << std::endl;
    std::cout << "	srcArea: " << srcArea << std::endl;
    nms_fast_kernel<<<grid, block>>>(src, src_box_width, topK, iou_thresh);
    std::cout << "Debug nmsDeviceV1 Ends:" << std::endl;
}