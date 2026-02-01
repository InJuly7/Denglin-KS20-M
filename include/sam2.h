#pragma once

#include <iostream>
#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <dlnne/dlnne.h>

#include "include/op_gpu.h"
#include "utils.h"
#include "cv.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

class SAM2 {
   public:
    SAM2() = default;
    ~SAM2() {}
    
    /*
        1. dlnne初始化, 反序列化引擎文件, 创建执行上下文
        2. onnxruntime 初始化, 创建session, 准备输入输出name
        3. 加载ini文件
    */
    void init(const std::string inifile, const int _src_h, const int _src_w, const int _src_channel) {
        

        // ONNX Runtime 初始化
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session = std::make_unique<Ort::Session>(env, m_param.decoder_path.c_str(), session_options);
        memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t num_inputs = session->GetInputCount();
        m_input_name_ptrs.clear();
        m_input_names.clear();
        m_input_name_strs.clear();
        m_input_name_ptrs.reserve(num_inputs);
        m_input_names.reserve(num_inputs);
        m_input_name_strs.reserve(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name_ptr = session->GetInputNameAllocated(i, allocator);
            m_input_names.push_back(name_ptr.get());
            m_input_name_ptrs.emplace_back(std::move(name_ptr));
            m_input_name_strs.emplace_back(m_input_name_ptrs.back().get());
        }

        const size_t num_outputs = session->GetOutputCount();
        m_output_name_ptrs.clear();
        m_output_names.clear();
        m_output_name_strs.clear();
        m_output_name_ptrs.reserve(num_outputs);
        m_output_names.reserve(num_outputs);
        m_output_name_strs.reserve(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_ptr = session->GetOutputNameAllocated(i, allocator);
            m_output_names.push_back(name_ptr.get());
            m_output_name_ptrs.emplace_back(std::move(name_ptr));
            m_output_name_strs.emplace_back(m_output_name_ptrs.back().get());
        }
        

        // 创建仿射矩阵, 建立原始图片到模型输入图片的映射关系
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
        // dlnne 检验 Encoder静态shape模型
        m_input_dims = m_context->GetBindingDimensions(0);
        assert(m_param.batch_size == m_input_dims.d[0]);
        assert(m_param.channel == m_input_dims.d[1]);
        assert(m_param.dst_h == m_input_dims.d[2]);
        assert(m_param.dst_w == m_input_dims.d[3]);

        m_output_dims = m_context->GetBindingDimensions(1);
        assert(m_param.high_res_feats_0_C == m_output_dims.d[1]);
        assert(m_param.high_res_feats_0_H == m_output_dims.d[2]);
        assert(m_param.high_res_feats_0_W == m_output_dims.d[3]);

        m_output_dims = m_context->GetBindingDimensions(2);
        assert(m_param.high_res_feats_1_C == m_output_dims.d[1]);
        assert(m_param.high_res_feats_1_H == m_output_dims.d[2]);
        assert(m_param.high_res_feats_1_W == m_output_dims.d[3]);

        m_output_dims = m_context->GetBindingDimensions(3);
        assert(m_param.image_embed_C == m_output_dims.d[1]);
        assert(m_param.image_embed_H == m_output_dims.d[2]);
        assert(m_param.image_embed_W == m_output_dims.d[3]);

        // onnxruntime 检验 Decoder 模型
        
    }

    void Alloc_buffer() {
        // Image To Device
        m_input_src_device = nullptr;
        CHECK(cudaMalloc(&m_input_src_device, m_param.channel * m_param.src_h * m_param.src_w * sizeof(unsigned char)));

        // Pre-process result to Device
        m_input_dst_device = nullptr;
        // [1,3,1024,1024]
        CHECK(cudaMalloc(&m_input_dst_device, m_param.channel * m_param.dst_h * m_param.dst_w * sizeof(float)));

        // Image Encoder模型输出到Device
        m_output0_device = nullptr;
        m_output1_device = nullptr;
        m_output2_device = nullptr;

        // [1,32,256,256]
        CHECK(cudaMalloc(&m_output0_device,
                         m_param.high_res_feats_0_C * m_param.high_res_feats_0_H * m_param.high_res_feats_0_W * sizeof(float)));
        // [1,64,128,128]
        CHECK(cudaMalloc(&m_output1_device,
                         m_param.high_res_feats_1_C * m_param.high_res_feats_1_H * m_param.high_res_feats_1_W * sizeof(float)));
        // [1,256,64,64]
        CHECK(cudaMalloc(&m_output2_device, m_param.image_embed_C * m_param.image_embed_H * m_param.image_embed_W * sizeof(float)));

        // Image Encoder模型输出到 Host
        m_output0_host =
            (float*)malloc(m_param.high_res_feats_0_C * m_param.high_res_feats_0_H * m_param.high_res_feats_0_W * sizeof(float));
        m_output1_host =
            (float*)malloc(m_param.high_res_feats_1_C * m_param.high_res_feats_1_H * m_param.high_res_feats_1_W * sizeof(float));
        m_output2_host = (float*)malloc(m_param.image_embed_C * m_param.image_embed_H * m_param.image_embed_W * sizeof(float));

        m_detections.reserve(m_param.topK);
        m_points.reserve(m_param.topK);
        m_result_masks.reserve(m_param.topK);
    }

    void Free_buffer() {
        CHECK(cudaFree(m_input_src_device))
        CHECK(cudaFree(m_input_dst_device))
        CHECK(cudaFree(m_output0_device))
        CHECK(cudaFree(m_output1_device))
        CHECK(cudaFree(m_output2_device))
        free(m_output0_host);
        free(m_output1_host);
        free(m_output2_host);
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
        float a = float(dst_h) / m_param.src_h;
        float b = float(m_param.dst_w) / m_param.src_w;
        float scale = a < b ? a : b;

        // cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5, 0.f, scale,
        //                    (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
        // cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
        // cv::invertAffineTransform(src2dst, dst2src);

        // 创建 src2dst 变换矩阵
        float src2dst[2][3] = {{scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5f},
                               {0.f, scale, (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5f}};

        src2dst.v0 = src2dst[0][0];
        src2dst.v1 = src2dst[0][1];
        src2dst.v2 = src2dst[0][2];
        src2dst.v3 = src2dst[1][0];
        src2dst.v4 = src2dst[1][1];
        src2dst.v5 = src2dst[1][2];

        std::cout << "v0: " << src2dst.v0 << " v1: " << src2dst.v1 << " v2: " << src2dst.v2 << std::endl;
        std::cout << "v3: " << src2dst.v3 << " v4: " << src2dst.v4 << " v5: " << src2dst.v5 << std::endl;

        // 计算逆变换
        float dst2src[2][3];
        invertAffineTransform(src2dst, dst2src);

        // 赋值给成员变量
        dst2src.v0 = dst2src[0][0];
        dst2src.v1 = dst2src[0][1];
        dst2src.v2 = dst2src[0][2];
        dst2src.v3 = dst2src[1][0];
        dst2src.v4 = dst2src[1][1];
        dst2src.v5 = dst2src[1][2];

        std::cout << "v0: " << dst2src.v0 << " v1: " << dst2src.v1 << " v2: " << dst2src.v2 << std::endl;
        std::cout << "v3: " << dst2src.v3 << " v4: " << dst2src.v4 << " v5: " << dst2src.v5 << std::endl;
    }

    std::array<int, 4> clamp_box(const ztu::nn::image_rect_t& box, int max_w, int max_h) {
        int left = std::clamp(box.left, 0, max_w - 1);
        int right = std::clamp(box.right, 0, max_w - 1);
        int top = std::clamp(box.top, 0, max_h - 1);
        int bottom = std::clamp(box.bottom, 0, max_h - 1);
        if (right < left) {
            std::swap(left, right);
        }
        if (bottom < top) {
            std::swap(top, bottom);
        }
        return {left, top, right, bottom};
    }

    void preprocess() {
        CHECK(cudaMemcpy(m_input_src_device, img.data, m_param.channel * m_param.src_h * m_param.src_w * sizeof(unsigned char),
                         cudaMemcpyHostToDevice));
        // 图像预处理：仿射变换, 归一化, HWC->CHW, BGR2RGB
        affine_bilinear(m_input_src_device, m_param.src_w, m_param.src_h, m_input_dst_device, m_param.dst_w, m_param.dst_h, m_dst2src);

        // 输入原始图片检测框 缩放变换
        int num_labels = 1;
        int num_points = static_cast<int>(m_detections.size() * 2) + static_cast<int>(m_points.size());

        std::array<int64_t, 3> point_coords_shape = {num_labels, num_points, 2};
        std::array<int64_t, 2> point_labels_shape = {num_labels, num_points};

        std::vector<std::array<float, 2>> point_coords;
        std::vector<float> point_labels;
        point_coords.reserve(num_points);
        point_labels.reserve(num_points);

        for (auto& det : m_detections) {
            float x1 = m_src2dst.v0 * det.left + m_src2dst.v1 * det.top + m_src2dst.v2;
            float y1 = m_src2dst.v3 * det.left + m_src2dst.v4 * det.top + m_src2dst.v5;
            float x2 = m_src2dst.v0 * det.right + m_src2dst.v1 * det.bottom + m_src2dst.v2;
            float y2 = m_src2dst.v3 * det.right + m_src2dst.v4 * det.bottom + m_src2dst.v5;

            x1 = std::clamp(x1, 0, m_param.dst_w - 1);
            x2 = std::clamp(x2, 0, m_param.dst_w - 1);
            y1 = std::clamp(y1, 0, m_param.dst_h - 1);
            y2 = std::clamp(y2, 0, m_param.dst_h - 1);

            // 左上角
            point_coords.push_back({x1, y1});
            point_labels.push_back(2.0f);

            // 右下角
            point_coords.push_back({x2, y2});
            point_labels.push_back(3.0f);
        }

        for (auto& point : m_points) {
            float x = m_src2dst.v0 * point.x + m_src2dst.v1 * point.y + m_src2dst.v2;
            float y = m_src2dst.v3 * point.x + m_src2dst.v4 * point.y + m_src2dst.v5;

            point.x = std::clamp(x, 0.0f, static_cast<float>(m_param.dst_w - 1));
            point.y = std::clamp(y, 0.0f, static_cast<float>(m_param.dst_h - 1));

            point_coords.push_back({point.x, point.y});
            point_labels.push_back(point.flag ? 1.0f : 0.0f);
        }

        const int mask_size = m_param.image_embed_H * 4;
        m_mask_input.assign(1 * mask_size * mask_size, 0.0f);
        m_has_mask_input = {0.0f};
        std::array<int64_t, 4> mask_input_shape = {1, 1, mask_size, mask_size};
        std::array<int64_t, 1> has_mask_input_shape = {1};

        mask_input_tensor = Ort::Value::CreateTensor<float>(memory_info, m_mask_input.data(), m_mask_input.size(),
                                                            mask_input_shape.data(), mask_input_shape.size());
        has_mask_input_tensor = Ort::Value::CreateTensor<float>(memory_info, m_has_mask_input.data(), m_has_mask_input.size(),
                                                                has_mask_input_shape.data(), has_mask_input_shape.size());

        m_point_coords.clear();
        m_point_coords.reserve(point_coords.size() * 2);
        for (const auto& p : point_coords) {
            m_point_coords.push_back(p[0]);
            m_point_coords.push_back(p[1]);
        }
        m_point_labels = std::move(point_labels);

        point_coords_tensor = Ort::Value::CreateTensor<float>(memory_info, m_point_coords.data(), m_point_coords.size(),
                                                              point_coords_shape.data(), point_coords_shape.size());
        point_labels_tensor = Ort::Value::CreateTensor<float>(memory_info, m_point_labels.data(), m_point_labels.size(),
                                                              point_labels_shape.data(), point_labels_shape.size());
    }
    bool infer() {
        float* bindings[] = {m_input_dst_device, m_output0_device, m_output1_device, m_output2_device};
        bool context = m_context->Execute(1, (void**)bindings);

        if (!context) {
            std::cerr << "Inference execution failed." << std::endl;
            return false;
        }

        CHECK(cudaMemcpy(m_output0_host, m_output0_device,
                         m_param.high_res_feats_0_C * m_param.high_res_feats_0_H * m_param.high_res_feats_0_W * sizeof(float),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(m_output1_host, m_output1_device,
                         m_param.high_res_feats_1_C * m_param.high_res_feats_1_H * m_param.high_res_feats_1_W * sizeof(float),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(m_output2_host, m_output2_device,
                         m_param.image_embed_C * m_param.image_embed_H * m_param.image_embed_W * sizeof(float), cudaMemcpyDeviceToHost));

        std::array<int64_t, 4> image_embed_shape = {1, m_param.image_embed_C, m_param.image_embed_H, m_param.image_embed_W};
        std::array<int64_t, 4> high_res0_shape = {1, m_param.high_res_feats_0_C, m_param.high_res_feats_0_H, m_param.high_res_feats_0_W};
        std::array<int64_t, 4> high_res1_shape = {1, m_param.high_res_feats_1_C, m_param.high_res_feats_1_H, m_param.high_res_feats_1_W};

        image_embed_tensor = Ort::Value::CreateTensor<float>(memory_info, m_output2_host,
                                                             m_param.image_embed_C * m_param.image_embed_H * m_param.image_embed_W,
                                                             image_embed_shape.data(), image_embed_shape.size());
        high_res0_tensor = Ort::Value::CreateTensor<float>(memory_info, m_output0_host,
                                                           m_param.high_res_feats_0_C * m_param.high_res_feats_0_H * m_param.high_res_feats_0_W,
                                                           high_res0_shape.data(), high_res0_shape.size());
        high_res1_tensor = Ort::Value::CreateTensor<float>(memory_info, m_output1_host,
                                                           m_param.high_res_feats_1_C * m_param.high_res_feats_1_H * m_param.high_res_feats_1_W,
                                                           high_res1_shape.data(), high_res1_shape.size());

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.reserve(m_input_name_strs.size());
        for (const auto& name : m_input_name_strs) {
            if (name == "image_embed") {
                ort_inputs.emplace_back(std::move(image_embed_tensor));
            } else if (name == "high_res_feats_0") {
                ort_inputs.emplace_back(std::move(high_res0_tensor));
            } else if (name == "high_res_feats_1") {
                ort_inputs.emplace_back(std::move(high_res1_tensor));
            } else if (name == "point_coords") {
                ort_inputs.emplace_back(std::move(point_coords_tensor));
            } else if (name == "point_labels") {
                ort_inputs.emplace_back(std::move(point_labels_tensor));
            } else if (name == "mask_input") {
                ort_inputs.emplace_back(std::move(mask_input_tensor));
            } else if (name == "has_mask_input") {
                ort_inputs.emplace_back(std::move(has_mask_input_tensor));
            } else {
                std::cerr << "Unsupported decoder input name: " << name << std::endl;
                return false;
            }
        }

        auto outputs = session->Run(Ort::RunOptions{nullptr}, m_input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                    m_output_names.data(), m_output_names.size());

        if (outputs.empty()) {
            std::cerr << "decoder outputs size mismatch" << std::endl;
            return false;
        }

        m_decoder_mask_shape.clear();
        m_decoder_mask_data.clear();
        {
            auto info = outputs[0].GetTensorTypeAndShapeInfo();
            m_decoder_mask_shape = info.GetShape();
            size_t numel = 1;
            for (auto d : m_decoder_mask_shape) {
                if (d > 0) numel *= static_cast<size_t>(d);
            }
            const float* data = outputs[0].GetTensorData<float>();
            m_decoder_mask_data.assign(data, data + numel);
        }

        if (outputs.size() >= 2) {
            auto info = outputs[1].GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            size_t numel = 1;
            for (auto d : shape) {
                if (d > 0) numel *= static_cast<size_t>(d);
            }
            const float* data = outputs[1].GetTensorData<float>();
            m_decoder_iou_data.assign(data, data + numel);
        }

        return true;
    }

    void postprocess(cv::Mat& img) {
        if (m_decoder_mask_data.empty() || m_decoder_mask_shape.size() < 2) {
            return;
        }

        int mask_h = 0;
        int mask_w = 0;
        if (m_decoder_mask_shape.size() == 4) {
            mask_h = static_cast<int>(m_decoder_mask_shape[2]);
            mask_w = static_cast<int>(m_decoder_mask_shape[3]);
        } else if (m_decoder_mask_shape.size() == 3) {
            mask_h = static_cast<int>(m_decoder_mask_shape[1]);
            mask_w = static_cast<int>(m_decoder_mask_shape[2]);
        } else {
            return;
        }

        cv::Mat mask_f(mask_h, mask_w, CV_32F, m_decoder_mask_data.data());
        cv::Mat mask = mask_f.clone();

        cv::Mat mask_resized;
        cv::resize(mask, mask_resized, cv::Size(m_param.dst_w, m_param.dst_h), 0, 0, cv::INTER_LINEAR);

        cv::Mat affine = (cv::Mat_<float>(2, 3) << m_dst2src.v0, m_dst2src.v1, m_dst2src.v2, m_dst2src.v3, m_dst2src.v4, m_dst2src.v5);
        cv::Mat mask_src;
        cv::warpAffine(mask_resized, mask_src, affine, cv::Size(m_param.src_w, m_param.src_h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

        if (!img.empty()) {
            cv::Mat mask_u8;
            cv::threshold(mask_src, mask_u8, 0.0, 255.0, cv::THRESH_BINARY);
            mask_u8.convertTo(mask_u8, CV_8U);
            cv::Mat colored;
            cv::cvtColor(mask_u8, colored, cv::COLOR_GRAY2BGR);
            img = img * 0.6 + colored * 0.4;
        }
    }

    void reset() {
        // 重置检测框结果
        m_detections.clear();
        m_points.clear();
        m_result_masks.clear();
    }

    /*
        1. 输入待分割的图片, 点坐标, 检测框
        2. 预处理, 推理, 后处理
    */
    void task() {

        preprocess();
        infer();
        postprocess(img);

    }

   public:
    utils::AffineMat m_dst2src;
    utils::AffineMat m_src2dst;

    // input
    unsigned char* m_input_src_device;
    float* m_input_dst_device;
    std::vector<utils::Box> m_detections;
    std::vector<utils::Point> m_points;
    std::vector<cv::Mat> m_result_masks;

    Ort::Value image_embed_tensor;
    Ort::Value high_res0_tensor;
    Ort::Value high_res1_tensor;
    Ort::Value point_coords_tensor;
    Ort::Value point_labels_tensor;
    Ort::Value mask_input_tensor;
    Ort::Value has_mask_input_tensor;

    std::vector<float> m_mask_input;
    std::array<float, 1> m_has_mask_input;
    std::vector<float> m_point_coords;
    std::vector<float> m_point_labels;

    std::vector<Ort::AllocatedStringPtr> m_input_name_ptrs;
    std::vector<Ort::AllocatedStringPtr> m_output_name_ptrs;
    std::vector<const char*> m_input_names;
    std::vector<const char*> m_output_names;
    std::vector<std::string> m_input_name_strs;
    std::vector<std::string> m_output_name_strs;

    // output
    float* m_output0_device;
    float* m_output1_device;
    float* m_output2_device;

    float* m_output0_host;
    float* m_output1_host;
    float* m_output2_host;

    std::vector<int64_t> m_decoder_mask_shape;
    std::vector<float> m_decoder_mask_data;
    std::vector<float> m_decoder_iou_data;

    SAM2Parameter m_param;
};
