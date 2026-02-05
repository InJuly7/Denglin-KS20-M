#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <dlnne/dlnne.h>
#include <cuda_runtime_api.h>

namespace utils {
namespace dataSets {
const std::vector<std::string> coco80 = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};
const std::vector<std::string> coco91 = {
    "person",     "bicycle",       "car",           "motorcycle",  "airplane",    "bus",           "train",        "truck",
    "boat",       "traffic light", "fire hydrant",  "street sign", "stop sign",   "parking meter", "bench",        "bird",
    "cat",        "dog",           "horse",         "sheep",       "cow",         "elephant",      "bear",         "zebra",
    "giraffe",    "hat",           "backpack",      "umbrella",    "shoe",        "eye glasses",   "handbag",      "tie",
    "suitcase",   "frisbee",       "skis",          "snowboard",   "sports ball", "kite",          "baseball bat", "baseball glove",
    "skateboard", "surfboard",     "tennis racket", "bottle",      "plate",       "wine glass",    "cup",          "fork",
    "knife",      "spoon",         "bowl",          "banana",      "apple",       "sandwich",      "orange",       "broccoli",
    "carrot",     "hot dog",       "pizza",         "donut",       "cake",        "chair",         "couch",        "potted plant",
    "bed",        "mirror",        "dining table",  "window",      "desk",        "toilet",        "door",         "tv",
    "laptop",     "mouse",         "remote",        "keyboard",    "cell phone",  "microwave",     "oven",         "toaster",
    "sink",       "refrigerator",  "blender",       "book",        "clock",       "vase",          "scissors",     "teddy bear",
    "hair drier", "toothbrush",    "hair brush"};
const std::vector<std::string> voc20 = {"aeroplane", "bicycle",     "bird",  "boat",        "bottle", "bus",      "car",
                                        "cat",       "chair",       "cow",   "diningtable", "dog",    "horse",    "motorbike",
                                        "person",    "pottedplant", "sheep", "sofa",        "train",  "tvmonitor"};
const std::vector<std::string> face2 = {"non-face", "face"};
const std::vector<std::string> obb15 = {
    "plane",  "ship",          "storage tank",  "baseball diamond", "tennis court", "basketball court",  "ground track field", "harbor",
    "bridge", "large vehicle", "small vehicle", "helicopter",       "roundabout",   "soccer ball field", "swimming pool"};
};  // namespace dataSets

struct Box {
    float cx, cy, w, h;
    float left, top, right, bottom;
    float confidence;
    int label;
    std::string class_name;
    cv::Scalar color = cv::Scalar(0, 255, 0);
    bool flag = true;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label, cv::Scalar color = cv::Scalar(0, 255, 0))
        : left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label), color(color) {}
    bool operator<(const Box& A) const { return confidence > A.confidence; }
};

struct label_info {
    int label;
    float conf;
    cv::Point tl;  // 左上角
    cv::String info = "";
    cv::Scalar color = cv::Scalar(0, 255, 0);
    int char_width = 11;
    int det_info_render_width = 15;
    double font_scale = 0.6;

    label_info() = default;
    label_info(int label, float conf, cv::Point tl, cv::String info, cv::Scalar color)
        : label(label), conf(conf), tl(tl), info(info), color(color) {}
};

struct mask_canvas {
    cv::Mat mask_instance_bgr;
    cv::Mat canvas;
    cv::Rect roisrc;
    float weight = 0.45;

    mask_canvas() = default;
    mask_canvas(cv::Mat mask_instance_bgr, cv::Mat canvas, cv::Rect roisrc, float weight = 0.45)
        : mask_instance_bgr(mask_instance_bgr), canvas(canvas), roisrc(roisrc), weight(weight) {}
};

struct Point {
    float x, y;
    bool flag;
    Point() = default;
    Point(float x, float y, bool flag) : x(x), y(y), flag(flag) {}
};

enum class InputStream { IMAGE, VIDEO, CAMERA };

enum class ColorMode { RGB, GRAY };

struct AffineMat {
    float v0, v1, v2;
    float v3, v4, v5;
};

class HostTimer {
   public:
    HostTimer() { t1 = std::chrono::high_resolution_clock::now(); }
    // while timing for cuda code, add "cudaDeviceSynchronize();" before this
    float getUsedTime() {
        t2 = std::chrono::high_resolution_clock::now();
        auto duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        return duration_2.count() / 1000;  // ms
    }

    ~HostTimer() {}

   private:
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
};

class DeviceTimer {
   public:
    DeviceTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
    }
    ~DeviceTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    float getUsedTime() {
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float total_time;
        cudaEventElapsedTime(&total_time, start, end);
        return total_time;
    }

    float getUsedTime(cudaStream_t stream) {
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        float total_time;
        cudaEventElapsedTime(&total_time, start, end);
        return total_time;
    }

   private:
    cudaEvent_t start, end;
};

static std::string trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static std::string toLower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::map<std::string, std::string> loadIni(const std::string& path) {
    std::ifstream in(path);
    std::map<std::string, std::string> kv;
    if (!in.is_open()) {
        return kv;
    }
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            continue;
        }
        auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        kv[key] = value;
    }
    return kv;
}

static std::string getStr(const std::map<std::string, std::string>& kv, const std::string& key, const std::string& def) {
    auto it = kv.find(key);
    return it == kv.end() ? def : it->second;
}

static int getInt(const std::map<std::string, std::string>& kv, const std::string& key, int def) {
    auto it = kv.find(key);
    return it == kv.end() ? def : std::stoi(it->second);
}

static float getFloat(const std::map<std::string, std::string>& kv, const std::string& key, float def) {
    auto it = kv.find(key);
    return it == kv.end() ? def : std::stof(it->second);
}

static bool getBool(const std::map<std::string, std::string>& kv, const std::string& key, bool def) {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    std::string v = toLower(it->second);
    return (v == "1" || v == "true" || v == "yes" || v == "on");
}

static std::string resolveModelPath(const std::string& value) {
    if (value.empty()) return value;
    if (!value.empty() && value.front() == '/') return value;
    return std::string("/data/model/") + value;
}

static utils::InputStream parseInputStream(const std::string& value) {
    std::string v = toLower(value);
    if (v == "video") return utils::InputStream::VIDEO;
    if (v == "camera") return utils::InputStream::CAMERA;
    return utils::InputStream::IMAGE;
}

// 计算仿射变换的逆变换
inline void invertAffineTransform(const float src2dst[2][3], float dst2src[2][3]) {
    // 提取 2x2 旋转/缩放矩阵
    float a = src2dst[0][0];
    float b = src2dst[0][1];
    float c = src2dst[1][0];
    float d = src2dst[1][1];

    // 提取平移向量
    float tx = src2dst[0][2];
    float ty = src2dst[1][2];

    // 计算行列式
    float det = a * d - b * c;

    // 计算逆变换矩阵
    dst2src[0][0] = d / det;
    dst2src[0][1] = -b / det;
    dst2src[0][2] = -(d * tx - b * ty) / det;
    dst2src[1][0] = -c / det;
    dst2src[1][1] = a / det;
    dst2src[1][2] = -(-c * tx + a * ty) / det;
}

inline void Affine_Matrix(int src_w, int src_h, int dst_w, int dst_h, utils::AffineMat& src2dst, utils::AffineMat& dst2src) {
    // 读入图像后，计算仿射矩阵
    float a = float(dst_h) / src_h;
    float b = float(dst_w) / src_w;
    float scale = a < b ? a : b;

    // cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5, 0.f, scale,
    //                    (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
    // cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    // cv::invertAffineTransform(src2dst, dst2src);

    // 创建 src2dst 变换矩阵
    float src2dst_matrix[2][3] = {{scale, 0.f, (-scale * src_w + dst_w + scale - 1) * 0.5f},
                                  {0.f, scale, (-scale * src_h + dst_h + scale - 1) * 0.5f}};

    src2dst.v0 = src2dst_matrix[0][0];
    src2dst.v1 = src2dst_matrix[0][1];
    src2dst.v2 = src2dst_matrix[0][2];
    src2dst.v3 = src2dst_matrix[1][0];
    src2dst.v4 = src2dst_matrix[1][1];
    src2dst.v5 = src2dst_matrix[1][2];

    // std::cout << "v0: " << src2dst.v0 << " v1: " << src2dst.v1 << " v2: " << src2dst.v2 << std::endl;
    // std::cout << "v3: " << src2dst.v3 << " v4: " << src2dst.v4 << " v5: " << src2dst.v5 << std::endl;

    // 计算逆变换
    float dst2src_matrix[2][3];
    invertAffineTransform(src2dst_matrix, dst2src_matrix);

    // 赋值给成员变量
    dst2src.v0 = dst2src_matrix[0][0];
    dst2src.v1 = dst2src_matrix[0][1];
    dst2src.v2 = dst2src_matrix[0][2];
    dst2src.v3 = dst2src_matrix[1][0];
    dst2src.v4 = dst2src_matrix[1][1];
    dst2src.v5 = dst2src_matrix[1][2];

    // std::cout << "v0: " << dst2src.v0 << " v1: " << dst2src.v1 << " v2: " << dst2src.v2 << std::endl;
    // std::cout << "v3: " << dst2src.v3 << " v4: " << dst2src.v4 << " v5: " << dst2src.v5 << std::endl;
}

// void saveBinaryFile(float* vec, size_t len, const std::string& file) {
//     std::ofstream out(file, std::ios::out | std::ios::binary);
//     if (!out.is_open()) return;
//     out.write((const char*)vec, sizeof(float) * len);
//     out.close();
// }

// std::vector<uint8_t> readBinaryFile(const std::string& file) {
//     std::ifstream in(file, std::ios::in | std::ios::binary);
//     if (!in.is_open()) return {};

//     in.seekg(0, std::ios::end);
//     size_t length = in.tellg();

//     std::vector<uint8_t> data;
//     if (length > 0) {
//         in.seekg(0, std::ios::beg);
//         data.resize(length);

//         in.read((char*)&data[0], length);
//     }
//     in.close();
//     return data;
// }

// void xywh2xyxyxyxy(const float cx, const float cy, const float w, const float h, const float angle, float (&point)[4][2],
//                    const AffineMat& m_dst2src) {
//     float cosA = cos(angle);
//     float sinA = sin(angle);
//     float vec1_x = (w / 2) * cosA;
//     float vec1_y = (w / 2) * sinA;
//     float vec2_x = (-1) * (h / 2) * sinA;
//     float vec2_y = (h / 2) * cosA;

//     point[0][0] = cx + vec1_x + vec2_x;
//     point[0][1] = cy + vec1_y + vec2_y;

//     point[1][0] = cx + vec1_x - vec2_x;
//     point[1][1] = cy + vec1_y - vec2_y;

//     point[2][0] = cx - vec1_x - vec2_x;
//     point[2][1] = cy - vec1_y - vec2_y;

//     point[3][0] = cx - vec1_x + vec2_x;
//     point[3][1] = cy - vec1_y + vec2_y;

//     for (int i = 0; i < 4; i++) {
//         point[i][0] = m_dst2src.v0 * point[i][0] + m_dst2src.v1 * point[i][1] + m_dst2src.v2;
//         point[i][1] = m_dst2src.v3 * point[i][0] + m_dst2src.v4 * point[i][1] + m_dst2src.v5;
//     }
// }

// void DumpGPUMemoryToFile(const void* gpu_ptr, size_t size_bytes, const char* filepath) {
//     try {
//         void* host_data = malloc(size_bytes);
//         if (host_data == nullptr) {
//             std::cerr << "Failed to allocate host memory of size " << size_bytes << " bytes" << std::endl;
//             exit(1);
//         }

//         cudaError_t cuda_status = cudaMemcpy(host_data, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
//         if (cuda_status != cudaSuccess) {
//             std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(cuda_status) << std::endl;
//             free(host_data);
//             exit(1);
//         }

//         std::ofstream outfile(filepath, std::ios::binary);
//         if (!outfile.is_open()) {
//             std::cerr << "Failed to open file: " << filepath << std::endl;
//             free(host_data);
//             exit(1);
//         }

//         outfile.write(static_cast<const char*>(host_data), size_bytes);
//         outfile.close();

//         free(host_data);
//         std::cout << "Successfully dumped " << size_bytes << " bytes to " << filepath << std::endl;

//         exit(0);
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         exit(1);
//     }
// }

// void DumpCPUMemoryToFile(const void* cpu_ptr, size_t size_bytes, const char* filepath) {
//     try {
//         std::ofstream outfile(filepath, std::ios::binary);
//         if (!outfile.is_open()) {
//             std::cerr << "Failed to open file: " << filepath << std::endl;
//             exit(1);
//         }

//         outfile.write(static_cast<const char*>(cpu_ptr), size_bytes);
//         outfile.close();

//         std::cout << "Successfully dumped " << size_bytes << " bytes to " << filepath << std::endl;

//         exit(0);
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         exit(1);
//     }
// }

// std::string get_fileformat(std::string filepath) { return filepath.substr(filepath.size() - 4, 4); }
}  // namespace utils