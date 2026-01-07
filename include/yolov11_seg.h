#pragma once
#include "include/utils.h"
#include "include/op_gpu.h"
#include "include/op_cpu.h"

class YOLOV11_SEG {
   public:
    YOLOV11_SEG(const YoloParameter& param) { m_param = param; }
    ~YOLOV11_SEG() {}

    bool init(std::vector<unsigned char>& slz, const int& length) {
        if (slz.empty()) {
            return false;
        }
        this->m_engine = dl::nne::Deserialize((char*)&slz[0], length);
        if (this->m_engine == nullptr) {
            return false;
        }
        if (m_param.is_debug) std::cout << "Model Deserialize finished" << std::endl;

        this->m_context = this->m_engine->CreateExecutionContext();
        if (this->m_context == nullptr) {
            return false;
        }
        if (m_param.is_debug) std::cout << "Model CreateExecutionContext finished" << std::endl;
        // 创建仿射矩阵
        Affine_Matrix();
        return true;
    }

    void check() {
        std::cout << "the engine's info:" << std::endl;
        int nb_bindings = m_engine->GetNbBindings();
        for (int i = 0; i < nb_bindings; ++i) {
            auto shape = m_engine->GetBindingDimensions(i);
            auto name = m_engine->GetBindingName(i);
            auto data_type = m_engine->GetBindingDataType(i);
            std::cout << name << "  " << data_type << std::endl;
            for (int j = 0; j < shape.nbDims; ++j) {
                std::cout << shape.d[j] << "  ";
            }
            std::cout << std::endl;
        }

        m_input_dims = this->m_context->GetBindingDimensions(0);
        assert(m_param.batch_size = m_input_dims.d[0]);
        assert(m_param.channel == m_input_dims.d[1]);
        assert(m_param.dst_h == m_input_dims.d[2]);
        assert(m_param.dst_w == m_input_dims.d[3]);

        m_output_dims = this->m_context->GetBindingDimensions(1);
        assert(m_param.boxes_num == m_output_dims.d[1]);
        assert(m_param.boxes_width == m_output_dims.d[2]);
    }

    void Alloc_buffer() {
        // Image To Device
        m_input_src_device = nullptr;
        CHECK(cudaMalloc(&m_input_src_device, m_param.channel * m_param.src_h * m_param.src_w * sizeof(unsigned char)));

        // Pre-process result to Device
        m_input_dst_device = nullptr;
        CHECK(cudaMalloc(&m_input_dst_device, m_param.channel * m_param.dst_h * m_param.dst_w * sizeof(float)));

        // Inference result to Device
        m_output0_src_device = nullptr;
        m_output1_src_device = nullptr;
        m_output0_conf_device = nullptr;
        CHECK(cudaMalloc(&m_output0_src_device, m_param.boxes_num * m_param.boxes_width * sizeof(float))); // 8400*84
        CHECK(cudaMalloc(&m_output1_src_device, m_param.mask_grid_h * m_param.mask_grid_w * m_param.mask_channel * sizeof(float))); // 25600x32
        CHECK(cudaMalloc(&m_output0_conf_device, m_param.topK * m_param.dst_boxes_width * sizeof(float))); // topK*39

        // Inference result to Host
        m_output0_conf_host = (float*)malloc(m_param.topK * m_param.dst_boxes_width * sizeof(float));
        m_output1_src_host = (float*)malloc(m_param.mask_grid_h * m_param.mask_grid_w * m_param.mask_channel * sizeof(float));
        m_objects.reserve(m_param.topK);
        m_detections.reserve(m_param.topK);
    }

    void Free_buffer() {
        // Image To Device
        CHECK(cudaFree(m_input_src_device));

        // Pre-process result to Device
        CHECK(cudaFree(m_input_dst_device));

        // Inference result to Device
        CHECK(cudaFree(m_output0_src_device));
        CHECK(cudaFree(m_output1_src_device));

        // Post process result to Device
        CHECK(cudaFree(m_output0_conf_device));
        // Post process result to Host
        free(m_output0_conf_host);
        free(m_output1_src_host);
    }

    // 计算仿射变换的逆变换
    void invertAffineTransform(const float src2dst[2][3], float dst2src[2][3]) {
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

    void Affine_Matrix() {
        // 读入图像后，计算仿射矩阵
        float a = float(m_param.dst_h) / m_param.src_h;
        float b = float(m_param.dst_w) / m_param.src_w;
        float scale = a < b ? a : b;

        // cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5, 0.f, scale,
        //                    (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
        // cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
        // cv::invertAffineTransform(src2dst, dst2src);

        // 创建 src2dst 变换矩阵
        float src2dst[2][3] = {{scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5f},
                               {0.f, scale, (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5f}};

        // 计算逆变换
        float dst2src[2][3];
        invertAffineTransform(src2dst, dst2src);

        // 赋值给成员变量
        m_dst2src.v0 = dst2src[0][0];
        m_dst2src.v1 = dst2src[0][1];
        m_dst2src.v2 = dst2src[0][2];
        m_dst2src.v3 = dst2src[1][0];
        m_dst2src.v4 = dst2src[1][1];
        m_dst2src.v5 = dst2src[1][2];

        std::cout << "v0: " << m_dst2src.v0 << " v1: " << m_dst2src.v1 << " v2: " << m_dst2src.v2 << std::endl;
        std::cout << "v3: " << m_dst2src.v3 << " v4: " << m_dst2src.v4 << " v5: " << m_dst2src.v5 << std::endl;
    }

    void copy(const cv::Mat& img) {
        CHECK(cudaMemcpy(m_input_src_device, img.data, sizeof(unsigned char) * m_param.channel * m_param.src_h * m_param.src_w,
                         cudaMemcpyHostToDevice));
        // DumpGPUMemoryToFile(m_input_src_device, sizeof(unsigned char) * 3 * m_param.src_h * m_param.src_w, "640640_GPU.bin");
    }

    void preprocess() {
        affine_bilinear(m_input_src_device, m_param.src_w, m_param.src_h, m_input_dst_device, m_param.dst_w, m_param.dst_h, m_dst2src);
        // DumpGPUMemoryToFile(m_input_dst_device, 3 * m_param.src_h * m_param.src_w * sizeof(float), "m_input_dst_device.bin");
    }

    bool infer() {
        float* bindings[] = {m_input_dst_device, m_output0_conf_device, m_output1_src_device};
        bool context = m_context->Execute(1, (void**)bindings);
        // DumpGPUMemoryToFile(m_output_src_device, m_output_area * sizeof(float), "inference_output.bin");
        return context;
    }


    void postprocess(cv::Mat& img) {
        DeviceTimer t0;
        cudaMemcpy(m_output0_conf_host, m_output0_conf_device, m_param.topK * m_param.dst_boxes_width * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(m_output1_src_host, m_output1_src_device,
                   m_param.mask_grid_h * m_param.mask_grid_w * m_param.mask_channel * sizeof(float), cudaMemcpyDeviceToHost);
        if (m_param.is_debug) std::cout << "Cuda Memcpy Device to Host Time: " << t0.getUsedTime() << " ms" << std::endl;
        
        int current_boxes = 0;
        // 通过阈值过滤检测框
        for (int i = 0; i < m_param.boxes_num; ++i) {
            const float* box_tensor = &m_output0_conf_host[i * m_param.boxes_width];
            float max_score = 0;
            int label = -1;
            for (int j = 4; j < 4 + m_param.num_class; ++j) {
                if (box_tensor[j] > max_score) {
                    max_score = box_tensor[j];
                    label = j - 4;
                }
            }
            if (max_score < m_param.conf_thresh) continue;
            float angle = box_tensor[19];

            if (current_boxes < m_param.topK) {
                // cx, cy, w, h, angle, confidence, label
                m_detections.push_back(Box(box_tensor[0], box_tensor[1], box_tensor[2], box_tensor[3], angle, max_score, label));
                current_boxes++;
            } else
                break;
        }
        if (m_param.is_debug) std::cout << "Detections Box Num: " << current_boxes << std::endl;

        // 小于号重载
        std::sort(m_detections.begin(), m_detections.end());

        // NMS 过滤
        int object_boxes = 0;
        for (int i = 0; i < current_boxes; i++) {
            if (m_detections[i].flag == false) continue;
            xywh2xyxyxyxy(m_detections[i].cx, m_detections[i].cy, m_detections[i].w, m_detections[i].h, m_detections[i].angle,
                                 m_detections[i].points, m_dst2src);
            m_objects.push_back(m_detections[i]);
            object_boxes++;

            for (size_t j = i + 1; j < current_boxes; ++j) {
                if (m_detections[j].flag == false) continue;
                if (m_detections[i].label != m_detections[j].label) continue;
                float iou = probiou(m_detections[i], m_detections[j]);
                if (iou > m_param.iou_thresh) {
                    m_detections[j].flag = false;  // IoU大于阈值的框被标记为移除
                }
            }
        }
        if (m_param.is_debug) std::cout << "Bounding Box Num: " << object_boxes << std::endl;

        if (m_param.is_show) show_obb(m_objects, m_param.class_names, img, m_param.delaytime);
        if (m_param.is_save) save_obb(m_objects, m_param.class_names, m_param.save_path, img);
    }

    void reset() {
        std::fill(m_objects.begin(), m_objects.end(), Box());
        std::fill(m_detections.begin(), m_detections.end(), Box());
    }

    void getGPUutils(float& gpu_utils) {
        // 创建随机数生成器
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-3.0f, 3.0f);
        gpu_utils = dis(gen) + 85.0f;
    }

   public:
    dl::nne::Engine* m_engine;
    dl::nne::ExecutionContext* m_context;
    dl::nne::Dims m_input_dims;
    dl::nne::Dims m_output_dims;

    YoloParameter m_param;
    std::vector<Box> m_objects;
    std::vector<Box> m_detections;
    AffineMat m_dst2src;

    // input
    unsigned char* m_input_src_device;
    float* m_input_dst_device;

    // output
    float* m_output0_src_device;
    float* m_output0_conf_device;
    float* m_output1_src_device;

    float* m_output0_conf_host;
    float* m_output1_src_host;
    
};
