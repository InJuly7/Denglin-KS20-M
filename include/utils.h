#pragma once
#include "common_include.h"

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

const std::vector<cv::Scalar> color80{
    cv::Scalar(128, 77, 207),  cv::Scalar(65, 32, 208),   cv::Scalar(0, 224, 45),    cv::Scalar(3, 141, 219),   cv::Scalar(80, 239, 253),
    cv::Scalar(239, 184, 12),  cv::Scalar(7, 144, 145),   cv::Scalar(161, 88, 57),   cv::Scalar(0, 166, 46),    cv::Scalar(218, 113, 53),
    cv::Scalar(193, 33, 128),  cv::Scalar(190, 94, 113),  cv::Scalar(113, 123, 232), cv::Scalar(69, 205, 80),   cv::Scalar(18, 170, 49),
    cv::Scalar(89, 51, 241),   cv::Scalar(153, 191, 154), cv::Scalar(27, 26, 69),    cv::Scalar(20, 186, 194),  cv::Scalar(210, 202, 167),
    cv::Scalar(196, 113, 204), cv::Scalar(9, 81, 88),     cv::Scalar(191, 162, 67),  cv::Scalar(227, 73, 120),  cv::Scalar(177, 31, 19),
    cv::Scalar(133, 102, 137), cv::Scalar(146, 72, 97),   cv::Scalar(145, 243, 208), cv::Scalar(2, 184, 176),   cv::Scalar(219, 220, 93),
    cv::Scalar(238, 153, 134), cv::Scalar(197, 169, 160), cv::Scalar(204, 201, 106), cv::Scalar(13, 24, 129),   cv::Scalar(40, 38, 4),
    cv::Scalar(5, 41, 34),     cv::Scalar(46, 94, 129),   cv::Scalar(102, 65, 107),  cv::Scalar(27, 11, 208),   cv::Scalar(191, 240, 183),
    cv::Scalar(225, 76, 38),   cv::Scalar(193, 89, 124),  cv::Scalar(30, 14, 175),   cv::Scalar(144, 96, 90),   cv::Scalar(181, 186, 86),
    cv::Scalar(102, 136, 34),  cv::Scalar(158, 71, 15),   cv::Scalar(183, 81, 247),  cv::Scalar(73, 69, 89),    cv::Scalar(123, 73, 232),
    cv::Scalar(4, 175, 57),    cv::Scalar(87, 108, 23),   cv::Scalar(105, 204, 142), cv::Scalar(63, 115, 53),   cv::Scalar(105, 153, 126),
    cv::Scalar(247, 224, 137), cv::Scalar(136, 21, 188),  cv::Scalar(122, 129, 78),  cv::Scalar(145, 80, 81),   cv::Scalar(51, 167, 149),
    cv::Scalar(162, 173, 20),  cv::Scalar(252, 202, 17),  cv::Scalar(10, 40, 3),     cv::Scalar(150, 90, 254),  cv::Scalar(169, 21, 68),
    cv::Scalar(157, 148, 180), cv::Scalar(131, 254, 90),  cv::Scalar(7, 221, 102),   cv::Scalar(19, 191, 184),  cv::Scalar(98, 126, 199),
    cv::Scalar(210, 61, 56),   cv::Scalar(252, 86, 59),   cv::Scalar(102, 195, 55),  cv::Scalar(160, 26, 91),   cv::Scalar(60, 94, 66),
    cv::Scalar(204, 169, 193), cv::Scalar(126, 4, 181),   cv::Scalar(229, 209, 196), cv::Scalar(195, 170, 186), cv::Scalar(155, 207, 148)};
const std::vector<cv::Scalar> color91{
    cv::Scalar(148, 99, 164),  cv::Scalar(65, 172, 90),   cv::Scalar(18, 117, 190),  cv::Scalar(173, 208, 229), cv::Scalar(37, 162, 147),
    cv::Scalar(121, 99, 42),   cv::Scalar(218, 173, 104), cv::Scalar(193, 213, 138), cv::Scalar(142, 168, 45),  cv::Scalar(107, 143, 94),
    cv::Scalar(242, 89, 7),    cv::Scalar(87, 218, 248),  cv::Scalar(126, 168, 9),   cv::Scalar(86, 152, 105),  cv::Scalar(155, 135, 251),
    cv::Scalar(73, 234, 44),   cv::Scalar(177, 37, 42),   cv::Scalar(219, 215, 54),  cv::Scalar(124, 207, 143), cv::Scalar(7, 81, 209),
    cv::Scalar(254, 18, 130),  cv::Scalar(71, 54, 73),    cv::Scalar(172, 198, 63),  cv::Scalar(64, 217, 224),  cv::Scalar(105, 224, 25),
    cv::Scalar(41, 52, 130),   cv::Scalar(220, 27, 193),  cv::Scalar(65, 222, 86),   cv::Scalar(250, 150, 201), cv::Scalar(201, 150, 105),
    cv::Scalar(104, 96, 142),  cv::Scalar(111, 230, 54),  cv::Scalar(105, 24, 22),   cv::Scalar(42, 226, 101),  cv::Scalar(67, 26, 144),
    cv::Scalar(155, 113, 106), cv::Scalar(152, 196, 216), cv::Scalar(58, 68, 152),   cv::Scalar(68, 230, 213),  cv::Scalar(169, 143, 129),
    cv::Scalar(191, 102, 41),  cv::Scalar(5, 73, 170),    cv::Scalar(15, 73, 233),   cv::Scalar(95, 13, 71),    cv::Scalar(25, 92, 218),
    cv::Scalar(85, 173, 16),   cv::Scalar(247, 158, 17),  cv::Scalar(36, 28, 8),     cv::Scalar(31, 100, 134),  cv::Scalar(131, 71, 45),
    cv::Scalar(158, 190, 91),  cv::Scalar(90, 207, 220),  cv::Scalar(125, 77, 228),  cv::Scalar(40, 156, 67),   cv::Scalar(35, 250, 69),
    cv::Scalar(229, 61, 245),  cv::Scalar(210, 201, 106), cv::Scalar(184, 35, 131),  cv::Scalar(47, 124, 120),  cv::Scalar(1, 114, 23),
    cv::Scalar(99, 181, 17),   cv::Scalar(77, 141, 151),  cv::Scalar(79, 33, 95),    cv::Scalar(194, 111, 146), cv::Scalar(187, 199, 138),
    cv::Scalar(129, 215, 40),  cv::Scalar(160, 209, 144), cv::Scalar(139, 121, 58),  cv::Scalar(97, 208, 197),  cv::Scalar(185, 105, 171),
    cv::Scalar(160, 96, 136),  cv::Scalar(232, 26, 26),   cv::Scalar(34, 165, 109),  cv::Scalar(19, 86, 215),   cv::Scalar(205, 209, 199),
    cv::Scalar(131, 91, 25),   cv::Scalar(51, 201, 16),   cv::Scalar(64, 35, 128),   cv::Scalar(120, 161, 247), cv::Scalar(123, 164, 190),
    cv::Scalar(15, 191, 40),   cv::Scalar(11, 44, 117),   cv::Scalar(198, 136, 70),  cv::Scalar(14, 224, 240),  cv::Scalar(60, 186, 193),
    cv::Scalar(253, 190, 129), cv::Scalar(134, 228, 173), cv::Scalar(219, 156, 214), cv::Scalar(137, 67, 254),  cv::Scalar(178, 223, 250),
    cv::Scalar(219, 199, 139)};
const std::vector<cv::Scalar> color20{
    cv::Scalar(128, 77, 207),  cv::Scalar(65, 32, 208),   cv::Scalar(0, 224, 45),    cv::Scalar(3, 141, 219),  cv::Scalar(80, 239, 253),
    cv::Scalar(239, 184, 12),  cv::Scalar(7, 144, 145),   cv::Scalar(161, 88, 57),   cv::Scalar(0, 166, 46),   cv::Scalar(218, 113, 53),
    cv::Scalar(193, 33, 128),  cv::Scalar(190, 94, 113),  cv::Scalar(113, 123, 232), cv::Scalar(69, 205, 80),  cv::Scalar(18, 170, 49),
    cv::Scalar(89, 51, 241),   cv::Scalar(153, 191, 154), cv::Scalar(27, 26, 69),    cv::Scalar(20, 186, 194), cv::Scalar(210, 202, 167),
    cv::Scalar(196, 113, 204), cv::Scalar(9, 81, 88),     cv::Scalar(191, 162, 67),  cv::Scalar(227, 73, 120), cv::Scalar(177, 31, 19)};
const std::vector<cv::Scalar> color15{
    cv::Scalar(128, 77, 207), cv::Scalar(65, 32, 208),   cv::Scalar(0, 224, 45),    cv::Scalar(3, 141, 219),  cv::Scalar(80, 239, 253),
    cv::Scalar(239, 184, 12), cv::Scalar(7, 144, 145),   cv::Scalar(161, 88, 57),   cv::Scalar(0, 166, 46),   cv::Scalar(218, 113, 53),
    cv::Scalar(193, 33, 128), cv::Scalar(190, 94, 113),  cv::Scalar(113, 123, 232), cv::Scalar(69, 205, 80),  cv::Scalar(18, 170, 49),
    cv::Scalar(89, 51, 241),  cv::Scalar(153, 191, 154), cv::Scalar(27, 26, 69),    cv::Scalar(20, 186, 194), cv::Scalar(210, 202, 167)};

struct YoloParameter {
    std::string model_name = "";
    std::string onnx_model_path = "";
    std::string rlym_model_path = "";
    std::string quantize_model_path = "";
    std::string mlir_model_path = "";
    std::string quantize_mlir_model_path = "";
    std::string engine_path = "";
    std::string quantize_engine_path = "";
    std::string save_path = "";

    std::string video_path = "";
    std::string image_path = "";

    std::map<std::string, std::vector<int>> inputs;
    std::map<std::string, std::vector<int>> outputs;

    // Input Image size
    int src_h = 0, src_w = 0;

    // Model Input size
    int batch_size = 0;
    int channel = 0;
    int dst_h = 0;
    int dst_w = 0;

    // Model Output size
    int boxes_num = 0;
    int boxes_width = 0;
    int dst_boxes_width = 0;
    int mask_channel = 0;
    int mask_grid_h = 0;
    int mask_grid_w = 0;
    std::vector<std::string> class_names;
    std::vector<cv::Scalar> colors;
    int num_class = 0;

    float iou_thresh = 0.0f;
    float conf_thresh = 0.0f;
    int topK = 0;

    std::string winname = "Denglin-KS20-M";
    int char_width = 11;
    double font_scale = 0.6;
    int det_info_render_width = 15;

    bool is_show = false;
    bool is_save = false;
    bool is_debug = false;
    int delaytime = 15;  // 15ms
};

struct Box {
    float cx, cy, w, h;
    float left, top, right, bottom;
    float points[4][2];
    float confidence;
    float angle;
    int label;
    bool flag = true;
    std::array<float, 3> covariance_matrix;
    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int label)
        : left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) {}

    Box(float cx, float cy, float w, float h, float angle, float confidence, int label)
        : cx(cx), cy(cy), w(w), h(h), angle(angle), confidence(confidence), label(label) {
        // 计算协方差矩阵
        float a = (w * w) / 12.0f;
        float b = (h * h) / 12.0f;
        float cos_value = cos(angle);
        float sin_value = sin(angle);
        covariance_matrix[0] = a * cos_value * cos_value + b * sin_value * sin_value;
        covariance_matrix[1] = a * sin_value * sin_value + b * cos_value * cos_value;
        covariance_matrix[2] = (a - b) * sin_value * cos_value;
    }

    bool operator<(const Box& A) const { return confidence > A.confidence; }
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
    DeviceTimer::DeviceTimer() {
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

bool setInputStream(const InputStream& source, const std::string& imagePath, const std::string& videoPath, const int& cameraID,
                    cv::VideoCapture& capture, int& delayTime, int& src_h, int& src_w) {
    int total_frames = 0;
    switch (source) {
        case InputStream::IMAGE:
            capture.open(imagePath);
            total_frames = 1;
            std::cout << "total_frames = " << total_frames << std::endl;
            break;
        case InputStream::VIDEO:
            capture.open(videoPath);
            // 获取视频总帧数
            total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
            std::cout << "total_frames = " << total_frames << std::endl;
            break;
        case InputStream::CAMERA:
            capture.open(cameraID);
            // 配置成 无穷大
            total_frames = INT_MAX;
            break;
        default:
            break;
    }
    delayTime = 15;
    src_h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    src_w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    std::cout << "src_h: " << src_h << " src_w: " << src_w << std::endl;
    return capture.isOpened();
};

void saveBinaryFile(float* vec, size_t len, const std::string& file) {
    std::ofstream out(file, std::ios::out | std::ios::binary);
    if (!out.is_open()) return;
    out.write((const char*)vec, sizeof(float) * len);
    out.close();
}

std::vector<uint8_t> readBinaryFile(const std::string& file) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open()) return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

std::vector<unsigned char> loadModel(const std::string& mode_file, const std::string& engine_file, int& length) {
    std::ifstream slz(engine_file, std::ios::in | std::ios::binary);
    if (!slz.is_open()) {
        std::cout << "Build serialize engine " << std::endl;
        auto builder = dl::nne::CreateInferBuilder();
        auto network = builder->CreateNetwork();
        auto parser = dl::nne::CreateParser();
        parser->Parse(mode_file.c_str(), *network);
        dl::nne::Engine* engine = nullptr;
        engine = builder->BuildEngine(*network);
        parser->Destroy();
        network->Destroy();
        builder->Destroy();
        auto ser_res = engine->Serialize();
        std::ofstream new_slz(engine_file);
        new_slz.write(static_cast<char*>(ser_res->Data()), static_cast<int64_t>(ser_res->Size()));
        new_slz.close();
        ser_res->Destroy();
        slz.open(engine_file, std::ios::in | std::ios::binary);
    }

    slz.seekg(0, std::ios::end);
    length = static_cast<uint64_t>(slz.tellg());
    slz.seekg(0, std::ios::beg);
    std::vector<uint8_t> slz_data;
    slz_data.resize(length);
    slz.read((char*)&slz_data[0], static_cast<int64_t>(length));
    slz.close();
    return slz_data;
}

void setRenderWindow(InitParameter& param) {
    if (!param.is_show) return;
    int max_w = 960;
    int max_h = 540;
    float scale_h = (float)param.src_h / max_h;
    float scale_w = (float)param.src_w / max_w;
    if (scale_h > 1.f && scale_w > 1.f) {
        float scale = scale_h < scale_w ? scale_h : scale_w;
        cv::namedWindow(param.winname, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);  // for Linux
        cv::resizeWindow(param.winname, int(param.src_w / scale), int(param.src_h / scale));
        param.char_width = 16;
        param.det_info_render_width = 18;
        param.font_scale = 0.9;
    } else {
        cv::namedWindow(param.winname);
    }
}

void xywh2xyxyxyxy(const float cx, const float cy, const float w, const float h, const float angle, float (&point)[4][2],
                   const AffineMat& m_dst2src) {
    float cosA = cos(angle);
    float sinA = sin(angle);
    float vec1_x = (w / 2) * cosA;
    float vec1_y = (w / 2) * sinA;
    float vec2_x = (-1) * (h / 2) * sinA;
    float vec2_y = (h / 2) * cosA;

    point[0][0] = cx + vec1_x + vec2_x;
    point[0][1] = cy + vec1_y + vec2_y;

    point[1][0] = cx + vec1_x - vec2_x;
    point[1][1] = cy + vec1_y - vec2_y;

    point[2][0] = cx - vec1_x - vec2_x;
    point[2][1] = cy - vec1_y - vec2_y;

    point[3][0] = cx - vec1_x + vec2_x;
    point[3][1] = cy - vec1_y + vec2_y;

    for (int i = 0; i < 4; i++) {
        point[i][0] = m_dst2src.v0 * point[i][0] + m_dst2src.v1 * point[i][1] + m_dst2src.v2;
        point[i][1] = m_dst2src.v3 * point[i][0] + m_dst2src.v4 * point[i][1] + m_dst2src.v5;
    }
}

void draw_Text(cv::Mat& image, const std::string& text, const cv::Point& position, const cv::Scalar& color) {
    cv::putText(image,
                text,                      // 文字内容
                position,                  // 位置 (x, y)
                cv::FONT_HERSHEY_SIMPLEX,  // 字体
                0.6,                       // 字体大小
                color,                     // 颜色
                2,                         // 线条粗细
                cv::LINE_AA);              // 抗锯齿
}



// 创建自适应大小的OpenCV显示窗口(主要用于Linux)
void createAdaptiveWindow(const cv::Mat& img, const std::string& windows_title, int max_w = 960, int max_h = 540) {
    if (img.empty()) {
        return;
    }
    cv::namedWindow(windows_title, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    // 如果图像超过最大尺寸,按比例缩放窗口
    if (img.rows > max_h || img.cols > max_w) {
        double scale = std::min(static_cast<double>(max_w) / img.cols, static_cast<double>(max_h) / img.rows);
        cv::resizeWindow(windows_title, static_cast<int>(img.cols * scale), static_cast<int>(img.rows * scale));
    }
}


void DumpGPUMemoryToFile(const void* gpu_ptr, size_t size_bytes, const char* filepath) {
    try {
        void* host_data = malloc(size_bytes);
        if (host_data == nullptr) {
            std::cerr << "Failed to allocate host memory of size " << size_bytes << " bytes" << std::endl;
            exit(1);
        }

        cudaError_t cuda_status = cudaMemcpy(host_data, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(cuda_status) << std::endl;
            free(host_data);
            exit(1);
        }

        std::ofstream outfile(filepath, std::ios::binary);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            free(host_data);
            exit(1);
        }

        outfile.write(static_cast<const char*>(host_data), size_bytes);
        outfile.close();

        free(host_data);
        std::cout << "Successfully dumped " << size_bytes << " bytes to " << filepath << std::endl;

        exit(0);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        exit(1);
    }
}

void DumpCPUMemoryToFile(const void* cpu_ptr, size_t size_bytes, const char* filepath) {
    try {
        std::ofstream outfile(filepath, std::ios::binary);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            exit(1);
        }

        outfile.write(static_cast<const char*>(cpu_ptr), size_bytes);
        outfile.close();

        std::cout << "Successfully dumped " << size_bytes << " bytes to " << filepath << std::endl;

        exit(0);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        exit(1);
    }
}

get_fileformat(std::string filepath) { return filepath.substr(filepath.size() - 4, 4); }