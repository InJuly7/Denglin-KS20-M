#pragma once

#include <iostream>
#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <cstring>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <dlnne/dlnne.h>

#include "op_gpu.h"
#include "utils.h"
#include "cv.h"
#include "ort.h"

class SAM2 {
   public:
    SAM2() = default;
    ~SAM2() {}

    static float clampf(float v, float lo, float hi) {
        return std::max(lo, std::min(v, hi));
    }
    
    void init(const std::string inifile, const int _src_h, const int _src_w, const int _src_channel) {
        auto kv = utils::loadIni(inifile);
        encoder_model_name = utils::getStr(kv, "encoder_model_name", "image_encoder_s");
        decoder_model_name = utils::getStr(kv, "decoder_model_name", "image_decoder_s");
        std::string model_dir = utils::getStr(kv, "model_dir", "/data/model/sam2/");
        save_path = utils::getStr(kv, "save_path", "/home/linaro/program/yolov11_seg_deepsort/sam2_result.jpg");
        is_debug = utils::getBool(kv, "is_debug", true);
        is_save = utils::getBool(kv, "is_save", false);
        is_show = utils::getBool(kv, "is_show", false);
        encoder_is_dynamic = utils::getBool(kv, "encoder_is_dynamic", true);
        decoder_is_dynamic = utils::getBool(kv, "decoder_is_dynamic", false);
        delaytime = utils::getInt(kv, "delaytime", 15);

        // Encoder Proprecess:
        src_h = _src_h;
        src_w = _src_w;
        src_channel = _src_channel;
        batch_size = utils::getInt(kv, "batch_size", 1);
        dst_channel = utils::getInt(kv, "dst_channel", 3);
        dst_h = utils::getInt(kv, "dst_h", 640);
        dst_w = utils::getInt(kv, "dst_w", 640);
        encoder_input_num = utils::getInt(kv, "encoder_input_num", 1);
        encoder_output_num = utils::getInt(kv, "encoder_output_num", 2);

        // Encoder Postprocess:
        feat0_c = utils::getInt(kv, "feat0_c", 32);
        feat0_h = utils::getInt(kv, "feat0_h", 256);
        feat0_w = utils::getInt(kv, "feat0_w", 256);
        feat1_c = utils::getInt(kv, "feat1_c", 64);
        feat1_h = utils::getInt(kv, "feat1_h", 128);
        feat1_w = utils::getInt(kv, "feat1_w", 128);
        embed_c = utils::getInt(kv, "embed_c", 256);
        embed_h = utils::getInt(kv, "embed_h", 64);
        embed_w = utils::getInt(kv, "embed_w", 64);
        

        // Encoder inference
        dlrt::dlrtInit(encoder_model_name, dlrt_param, encoder_input_num, encoder_output_num, is_debug, model_dir);

        // Decoder Preprocess
        num_prompt = utils::getInt(kv, "num_prompt", 1);
        mask_channel = utils::getInt(kv, "mask_channel", 1);
        mask_h = utils::getInt(kv, "mask_h", 256);
        mask_w = utils::getInt(kv, "mask_w", 256);
        has_mask_input = utils::getBool(kv, "has_mask_input", false);
        decoder_input_num = utils::getInt(kv, "decoder_input_num", 7);
        decoder_output_num = utils::getInt(kv, "decoder_output_num", 2);

        // Decoder inference
        ort::OrtInit(decoder_model_name, ort_param, decoder_input_num, decoder_output_num, is_debug, decoder_is_dynamic, model_dir);
        
        // Decoder inference: init after num_points is determined in decoder_preprocess

        // 创建仿射矩阵, 建立原始图片到模型输入图片的映射关系
        utils::Affine_Matrix(src_w, src_h, dst_w, dst_h, src2dst, dst2src);
    }

    void encoder_preprocess(const cv::Mat& img) {
        // Image To Device
        input_src_device = nullptr;
        CHECK(cudaMalloc(&input_src_device, src_channel * src_h * src_w * sizeof(unsigned char)));
        CHECK(cudaMemcpy(input_src_device, img.data, src_channel * src_h * src_w * sizeof(unsigned char), cudaMemcpyHostToDevice));
        // Pro-precess
        // input_0
        // input_src_device: [1,src_h,src_w,3] -> dlrt_param.input_buffers[0]: [1,3,dst_h,dst_w]
        // 图像预处理：仿射变换, 归一化, HWC->CHW, BGR2RGB
        affine_bilinear_pad0(input_src_device, src_w, src_h, dlrt_param.input_buffers[0].device_ptr, dst_w, dst_h, dst2src);
        if (is_debug) std::cout << "Encoder Preprocess done." << std::endl;
        CHECK(cudaFree(input_src_device));
    }

    bool encoder_infer() {
        bool success = dlrt::infer(dlrt_param);
        assert(success == true);
        if (is_debug) std::cout << "Encoder Inference done." << std::endl;
        return success;
    }

    void encoder_postprocess() {}
    void decoder_preprocess(const std::vector<utils::Box>& boxes, const std::vector<utils::Point>& points = {}) {
        point_coords.clear();
        point_labels.clear();
        // 输入原始图片检测框 缩放变换
        num_points = static_cast<int>(boxes.size() * 2) + static_cast<int>(points.size());
        if (num_points <= 0) {
            return;
        }
        std::array<int64_t, 3> point_coords_shape = {1, num_points, 2};
        std::array<int64_t, 2> point_labels_shape = {1, num_points};

        for (auto& box : boxes) {
            float x1 = src2dst.v0 * box.left + src2dst.v1 * box.top + src2dst.v2;
            float y1 = src2dst.v3 * box.left + src2dst.v4 * box.top + src2dst.v5;
            float x2 = src2dst.v0 * box.right + src2dst.v1 * box.bottom + src2dst.v2;
            float y2 = src2dst.v3 * box.right + src2dst.v4 * box.bottom + src2dst.v5;

            x1 = clampf(x1, 0.0f, static_cast<float>(dst_w - 1));
            x2 = clampf(x2, 0.0f, static_cast<float>(dst_w - 1));
            y1 = clampf(y1, 0.0f, static_cast<float>(dst_h - 1));
            y2 = clampf(y2, 0.0f, static_cast<float>(dst_h - 1));

            // 左上角
            point_coords.push_back({x1, y1});
            point_labels.push_back(2.0f);  // box-left-top

            // 右下角
            point_coords.push_back({x2, y2});
            point_labels.push_back(3.0f);  // box-right-bottom
        }

        for (const auto& point : points) {
            float x = src2dst.v0 * point.x + src2dst.v1 * point.y + src2dst.v2;
            float y = src2dst.v3 * point.x + src2dst.v4 * point.y + src2dst.v5;

            float x_clamped = clampf(x, 0.0f, static_cast<float>(dst_w - 1));
            float y_clamped = clampf(y, 0.0f, static_cast<float>(dst_h - 1));

            point_coords.push_back({x_clamped, y_clamped});
            point_labels.push_back(point.flag ? 1.0f : 0.0f);
        }


        mask_input_host = (float*)malloc(mask_channel * mask_h * mask_w * sizeof(float));
        memset(mask_input_host, 0, mask_channel * mask_h * mask_w * sizeof(float));
        // SAM2 decoder 通常在没有 mask 输入时使用 0，若无历史 mask 可强制为 0
        has_mask_input_value = has_mask_input ? 1.0f : 0.0f;

        assert(mask_h == embed_h * 4 && mask_w == embed_w * 4);
        std::vector<std::vector<int64_t>> input_shapes = {
            {1, embed_c, embed_h, embed_w},     // image_embeddings
            {1, feat0_c, feat0_h, feat0_w},     // feat0
            {1, feat1_c, feat1_h, feat1_w},     // feat1
            {1, num_points, 2},                 // point_coords
            {1, num_points},                    // point_labels
            {1, mask_channel, mask_h, mask_w},  // mask_input
            {1}                                 // has_mask_input
        };
        std::vector<bool> is_dynamic = {false, false, false, true, true, false, false};
        std::vector<bool> is_from_gpu = {true, true, true, false, false, false, false};
        std::vector<float*> input_ptrs = {
            dlrt_param.output_buffers[2].device_ptr,  // image_embeddings
            dlrt_param.output_buffers[0].device_ptr,  // feat0
            dlrt_param.output_buffers[1].device_ptr,  // feat1
            point_coords.data()->data(),              // point_coords
            point_labels.data(),                      // point_labels
            mask_input_host,                          // mask_input
            &has_mask_input_value,                    // has_mask_input
        };

        ort::BufferInit(ort_param, input_shapes, is_dynamic, is_from_gpu, input_ptrs);
    }

    bool decoder_infer() {
        ort::infer(ort_param);
        if (ort_param.output_buffers.empty()) {
            return false;
        }

        // output_0: low_res_masks
        const auto& mask_buf = ort_param.output_buffers[0];
        decoder_mask_shape = mask_buf.shape;
        size_t mask_numel = 1;
        for (auto d : decoder_mask_shape) {
            mask_numel *= static_cast<size_t>(d);
        }
        decoder_mask_data.resize(mask_numel);
        if (mask_buf.device_ptr) {
            std::memcpy(decoder_mask_data.data(), mask_buf.device_ptr, mask_numel * sizeof(float));
        }

        // output_1: iou_predictions (optional)
        if (ort_param.output_buffers.size() > 1) {
            const auto& iou_buf = ort_param.output_buffers[1];
            decoder_iou_shape = iou_buf.shape;
            size_t iou_numel = 1;
            for (auto d : decoder_iou_shape) {
                iou_numel *= static_cast<size_t>(d);
            }
            decoder_iou_data.resize(iou_numel);
            if (iou_buf.device_ptr) {
                std::memcpy(decoder_iou_data.data(), iou_buf.device_ptr, iou_numel * sizeof(float));
            }
        }
        return true;
    }

    void decoder_postprocess(cv::Mat& img) {
        if (decoder_mask_data.empty() || decoder_mask_shape.size() < 2) {
            return;
        }

        int mask_h = 0;
        int mask_w = 0;
        int num_masks = 1;
        int batch = 1;
        if (decoder_mask_shape.size() == 4) {
            batch = static_cast<int>(decoder_mask_shape[0]);
            num_masks = static_cast<int>(decoder_mask_shape[1]);
            mask_h = static_cast<int>(decoder_mask_shape[2]);
            mask_w = static_cast<int>(decoder_mask_shape[3]);
        } else if (decoder_mask_shape.size() == 3) {
            num_masks = static_cast<int>(decoder_mask_shape[0]);
            mask_h = static_cast<int>(decoder_mask_shape[1]);
            mask_w = static_cast<int>(decoder_mask_shape[2]);
        } else {
            return;
        }

        int best_idx = 0;
        if (!decoder_iou_data.empty() && num_masks > 1) {
            float best = decoder_iou_data[0];
            for (int i = 1; i < num_masks; ++i) {
                if (decoder_iou_data[i] > best) {
                    best = decoder_iou_data[i];
                    best_idx = i;
                }
            }
        }

        size_t mask_area = static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w);
        size_t offset = static_cast<size_t>(best_idx) * mask_area;
        if (decoder_mask_shape.size() == 4) {
            offset = static_cast<size_t>(0) * static_cast<size_t>(num_masks) * mask_area + offset;
        }
        if (offset + mask_area > decoder_mask_data.size()) {
            return;
        }

        cv::Mat mask_f(mask_h, mask_w, CV_32F, decoder_mask_data.data() + offset);
        cv::Mat mask = mask_f.clone();

        cv::Mat mask_resized;
        cv::resize(mask, mask_resized, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

        float scale = std::min(static_cast<float>(dst_h) / src_h, static_cast<float>(dst_w) / src_w);
        int new_h = static_cast<int>(src_h * scale);
        int new_w = static_cast<int>(src_w * scale);
        int start_h = (dst_h - new_h) / 2;
        int start_w = (dst_w - new_w) / 2;
        start_h = std::max(0, start_h);
        start_w = std::max(0, start_w);
        new_h = std::min(new_h, dst_h - start_h);
        new_w = std::min(new_w, dst_w - start_w);
        cv::Mat mask_no_pad = mask_resized(cv::Rect(start_w, start_h, new_w, new_h));

        cv::Mat mask_src;
        cv::resize(mask_no_pad, mask_src, cv::Size(src_w, src_h), 0, 0, cv::INTER_LINEAR);

        if (!img.empty()) {
            if (is_debug) {
                double min_val = 0.0, max_val = 0.0;
                cv::minMaxLoc(mask_src, &min_val, &max_val);
                cv::Scalar mean_val = cv::mean(mask_src);
                std::cout << "mask_src stats: min=" << min_val << " max=" << max_val << " mean=" << mean_val[0] << std::endl;
                if (!decoder_iou_data.empty()) {
                    std::cout << "iou_predictions:";
                    for (size_t i = 0; i < decoder_iou_data.size(); ++i) {
                        std::cout << " " << decoder_iou_data[i];
                    }
                    std::cout << std::endl;
                }
            }
            cv::Mat mask_u8;
            cv::threshold(mask_src, mask_u8, 0.0, 255.0, cv::THRESH_BINARY);
            mask_u8.convertTo(mask_u8, CV_8U);
            cv::Mat colored;
            cv::cvtColor(mask_u8, colored, cv::COLOR_GRAY2BGR);
            colored.setTo(cv::Scalar(0, 255, 0), mask_u8);

            utils::mask_canvas mask_canvas;
            mask_canvas.mask_instance_bgr = colored;
            mask_canvas.canvas = cv::Mat::zeros(img.size(), img.type());
            mask_canvas.roisrc = cv::Rect(0, 0, img.cols, img.rows);
            mask_canvas.weight = 0.4f;

            std::vector<utils::mask_canvas> masks;
            masks.emplace_back(std::move(mask_canvas));
            cv_utils::draw(img, {}, masks, {}, false, true, false, is_show, is_save, "SAM2", save_path);
        }
    }
    
   public:
    std::string encoder_model_name = "image_encoder_s";
    std::string decoder_model_name = "image_decoder_s";
    std::string save_path = "/home/linaro/program/yolov11_seg_deepsort/sma2_result.jpg";
    bool is_debug = true;
    bool is_save = false;
    bool is_show = false;
    bool encoder_is_dynamic = false;
    bool decoder_is_dynamic = true;
    int delaytime = 15;
    // Encoder Proprecess:
    int src_h = 0;
    int src_w = 0;
    int src_channel = 0;
    int batch_size = 1;
    int dst_channel = 3;
    int dst_h = 0;
    int dst_w = 0;
    int encoder_input_num = 1;
    // Postprocess:
    int feat0_c = 0;
    int feat0_h = 0;
    int feat0_w = 0;
    int feat1_c = 0;
    int feat1_h = 0;
    int feat1_w = 0;
    int embed_c = 0;
    int embed_h = 0;
    int embed_w = 0;
    int encoder_output_num = 2;

    dlrt::dlrtParam dlrt_param;
    utils::AffineMat src2dst;
    utils::AffineMat dst2src;

    // encoder input
    unsigned char* input_src_device;

    // encoder output
    float* encoder_output0_host;
    float* encoder_output1_host;
    float* encoder_output2_host;

    // decoder input
    bool has_mask_input;
    float has_mask_input_value = 0.0f;
    float* mask_input_host;
    std::vector<std::array<float, 2>> point_coords;
    std::vector<float> point_labels;
    int num_prompt = 1;
    int num_points = 0;
    int mask_channel;
    int mask_h;
    int mask_w;
    std::vector<float> mask_input;

    int decoder_input_num = 0;
    int decoder_output_num = 0;
    ort::OrtParam ort_param;

    // decoder output
    std::vector<float> decoder_mask_data;
    std::vector<int64_t> decoder_mask_shape;
    std::vector<float> decoder_iou_data;
    std::vector<int64_t> decoder_iou_shape;
};
