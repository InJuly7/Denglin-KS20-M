#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <dlnne/dlnne.h>

#include "op_gpu.h"
#include "utils.h"
#include "cv.h"

class YOLOV11_SEG {
   public:
    YOLOV11_SEG() = default;
    ~YOLOV11_SEG() {}

    void init(const std::string inifile, const int _src_h, const int _src_w, const int _src_channel) {
        auto kv = utils::loadIni(inifile);
        model_name = utils::getStr(kv, "model_name", "yolov11s_seg");
        save_path = utils::getStr(kv, "save_path", "/home/linaro/program/yolov11_seg_deepsort/result.jpg");
        is_debug = utils::getBool(kv, "is_debug", true);
        is_save = utils::getBool(kv, "is_save", false);
        is_show = utils::getBool(kv, "is_show", false);
        delaytime = utils::getInt(kv, "delaytime", 15);
        num_class = utils::getInt(kv, "num_class", 80);
        class_names = utils::dataSets::coco80;
        colors = cv_utils::Colors::color80;

        // Proprecess:
        src_h = _src_h;
        src_w = _src_w;
        src_channel = _src_channel;
        batch_size = utils::getInt(kv, "batch_size", 1);
        dst_channel = utils::getInt(kv, "dst_channel", 3);
        dst_h = utils::getInt(kv, "dst_h", 640);
        dst_w = utils::getInt(kv, "dst_w", 640);
        input_num = utils::getInt(kv, "input_num", 1);
        // Postprocess:
        boxes_num = utils::getInt(kv, "boxes_num", 8400);
        boxes_width = utils::getInt(kv, "boxes_width", 116);
        dst_boxes_width = utils::getInt(kv, "dst_boxes_width", 39);
        mask_channel = utils::getInt(kv, "mask_channel", 32);
        mask_h = utils::getInt(kv, "mask_h", 160);
        mask_w = utils::getInt(kv, "mask_w", 160);
        output_num = utils::getInt(kv, "output_num", 2);
        conf_thresh = utils::getFloat(kv, "conf_thresh", 0.50f);
        iou_thresh = utils::getFloat(kv, "iou_thresh", 0.50f);
        topK = utils::getInt(kv, "topK", 300);
        // inference
        dlrt::dlrtInit(model_name, dlrt_param, input_num, output_num, is_debug);

        // 创建仿射矩阵
        utils::Affine_Matrix(src_w, src_h, dst_w, dst_h, src2dst, dst2src);
    }

    void preprocess(const cv::Mat& img) {
        // Image To Device
        input_src_device = nullptr;
        CHECK(cudaMalloc(&input_src_device, src_channel * src_h * src_w * sizeof(unsigned char)));
        CHECK(cudaMemcpy(input_src_device, img.data, sizeof(unsigned char) * src_channel * src_h * src_w, cudaMemcpyHostToDevice));
        // Pre-process
        // input_0
        // input_src_device: [1,src_h,src_w,3] -> dlrt_param.input_buffers[0]: [1,3,dst_h,dst_w]
        affine_bilinear(input_src_device, src_w, src_h, dlrt_param.input_buffers[0].device_ptr, dst_w, dst_h, dst2src);
        if (is_debug) std::cout << "Preprocess done." << std::endl;
        // Image To Device
        CHECK(cudaFree(input_src_device));
        // DumpGPUMemoryToFile(m_input_dst_device, 3 * m_param.src_h * m_param.src_w * sizeof(float), "m_input_dst_device.bin");
    }

    bool infer() {
        bool success = dlrt::infer(dlrt_param);
        assert(success == true);
        if (is_debug) std::cout << "Inference done." << std::endl;
        return success;
        // DumpGPUMemoryToFile(m_output_src_device, m_output_area * sizeof(float), "inference_output.bin");
    }

    void postprocess(cv::Mat& img) {
        // Post-process
        // output_0
        // dlrt_param.output_buffers[0]: [1,mask_channel,mask_h,mask_w]
        // output_1
        // dlrt_param.output_buffers[1]: [1,boxes_num,boxes_width] -> output_1_conf_device: [1,1+topK,dst_boxes_width]
        utils::DeviceTimer t0;
        output1_conf_device = nullptr;
        CHECK(cudaMalloc(&output1_conf_device, (1 + topK * dst_boxes_width) * sizeof(float)));

        // Inference result to Host
        output0_src_host = (float*)malloc(mask_channel * mask_h * mask_w * sizeof(float));
        output1_conf_host = (float*)malloc((1 + topK * dst_boxes_width) * sizeof(float));


        conf_filter(dlrt_param.output_buffers[1].device_ptr, boxes_width, boxes_num, output1_conf_device, dst_boxes_width, topK, num_class,
                    conf_thresh);
        nms_fast(output1_conf_device, dst_boxes_width, topK, iou_thresh);

        
        cudaMemcpy(output1_conf_host, output1_conf_device, (1 + topK * dst_boxes_width) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(output0_src_host, dlrt_param.output_buffers[0].device_ptr, mask_channel * mask_h * mask_w * sizeof(float),
                   cudaMemcpyDeviceToHost);
        if (is_debug) std::cout << "Post process GPU: " << t0.getUsedTime() << " ms" << std::endl;

        int num_boxes = std::min((int)(output1_conf_host)[0], topK);  // conf filter 后的目标数量
        int current_boxes = 0;
        if (is_debug) std::cout << "Bounding Box Num: " << num_boxes << std::endl;

        cv::Mat mask160 = cv::Mat::zeros(1, mask_h * mask_w, CV_32F);
        cv::Mat img_canvas = cv::Mat::zeros(cv::Size(src_w, src_h), CV_8UC3);
        img_canvas.setTo(cv::Scalar(0, 0, 0));
        cv::Rect thresh_roi_160 = cv::Rect(0, 0, mask_h, mask_w);
        cv::Rect thresh_roi_src = cv::Rect(0, 0, src_w, src_h);

        Eigen::MatrixXf mask_eigen160 = Eigen::MatrixXf(1, mask_h * mask_w);
        float downsample_scale = static_cast<float>(mask_h) / static_cast<float>(dst_h);

        // [160*160,32]
        Eigen::Map<Eigen::MatrixXf> img_seg_(output0_src_host, mask_h * mask_w, mask_channel);
        std::vector<utils::mask_canvas> masks;
        std::vector<utils::Box> boxes;
        std::vector<utils::label_info> labels;

        for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
            // left top right bottom ,conf,label,flag,segmentation[32]
            float* ptr = output1_conf_host + 1 + box_idx * dst_boxes_width;
            if (ptr[6]) {

                int label = ptr[5];
                float conf = ptr[4];
                cv::Scalar color = colors[label];
                std::string class_name = class_names[label];
                if(is_debug) std::cout << "	label: " << label << " conf " << conf << std::endl;

                // [32,1] 分割编码系数
                Eigen::Map<Eigen::MatrixXf> img_obj_seg_(ptr + 7, mask_channel, 1);
                // [160*160,1] 计算物体的分割掩码, 掩码概率矩阵
                mask_eigen160 = img_seg_ * img_obj_seg_;
                cv::eigen2cv(mask_eigen160, mask160);

                // sigmoid 函数: 1/(1+e^(-x)) 表示掩码概率
                cv::exp(-mask160, mask160);
                mask160 = 1.f / (1.f + mask160);
                // [25600,1] --> [160,160]
                // cv::Mat.reshape 将 m_mask160 重塑为 160 行, 自动计算列数，1：通道数
                mask160 = mask160.reshape(1, mask_h);
                if (is_debug) std::cout << "	m_mask160.shape: " << mask160.rows << " " << mask160.cols << std::endl;
                // std::cout << "	left: " << ptr[0] << "; top: " << ptr[1] << "; right: " << ptr[2] << "; bottom: " << ptr[3] <<
                // std::endl; 计算检测框在掩码尺度上的坐标 从[640,640]的坐标 映射到 [160,160]的坐标 ptr[0:3] left top right bottom
                int x_lt_160 = std::round(ptr[0] * downsample_scale);
                int y_lt_160 = std::round(ptr[1] * downsample_scale);
                int x_rb_160 = std::round(ptr[2] * downsample_scale);
                int y_rb_160 = std::round(ptr[3] * downsample_scale);
                // std::cout << "	x_lt_160: " << x_lt_160 << "; y_lt_160: " << y_lt_160 << "; x_rb_160: " << x_rb_160
                //           << "; y_rb_160: " << y_rb_160 << std::endl;

                // 创建掩码尺度上的感兴趣区域（ROI）
                // 使用 & 运算符计算掩码尺度矩形与 thresh_roi_160 的交集, 保证后续掩码操作不会越界
                cv::Rect roi160 = cv::Rect(x_lt_160, y_lt_160, x_rb_160 - x_lt_160, y_rb_160 - y_lt_160) & thresh_roi_160;
                // std::cout << "	roi160.shape: " << roi160.height << " " << roi160.width << std::endl;
                if (roi160.width == 0 || roi160.height == 0) continue;

                // 计算检测框在原始图像尺度上的坐标
                int x_lt_src = std::round(dst2src.v0 * ptr[0] + dst2src.v1 * ptr[1] + dst2src.v2);
                int y_lt_src = std::round(dst2src.v3 * ptr[0] + dst2src.v4 * ptr[1] + dst2src.v5);
                int x_rb_src = std::round(dst2src.v0 * ptr[2] + dst2src.v1 * ptr[3] + dst2src.v2);
                int y_rb_src = std::round(dst2src.v3 * ptr[2] + dst2src.v4 * ptr[3] + dst2src.v5);
                if (is_debug)
                    std::cout << "	x_lt_src: " << x_lt_src << "; y_lt_src: " << y_lt_src << "; x_rb_src: " << x_rb_src
                              << "; y_rb_src: " << y_rb_src << std::endl;

                // 保证后续操作不会超出原图像尺寸范围
                cv::Rect roisrc = cv::Rect(x_lt_src, y_lt_src, x_rb_src - x_lt_src, y_rb_src - y_lt_src) & thresh_roi_src;
                // std::cout << "	roisrc.shape: " << roisrc.height << " " << roisrc.width << std::endl;
                if (roisrc.width == 0 || roisrc.height == 0) continue;

                // for opencv >=4.7(faster)
                // cv::Mat mask_instance;
                // cv::resize(cv::Mat(m_mask160, roi160), mask_instance, cv::Size(roisrc.width, roisrc.height), cv::INTER_LINEAR);
                // mask_instance = mask_instance > 0.5f;
                // cv::cvtColor(mask_instance, mask_instance, cv::COLOR_GRAY2BGR);
                // mask_instance.setTo(color, mask_instance);
                // cv::addWeighted(mask_instance, 0.45, m_img_canvas(roisrc), 1.0, 0., m_img_canvas(roisrc));

                // for opencv >=3.2.0
                cv::Mat mask_instance;
                // 从 掩码概率矩阵 mask160 中提取 roi160 定义的区域
                // 使用插值方法 i调整大小到 (roisrc.width, roisrc.height) 尺寸
                cv::resize(cv::Mat(mask160, roi160), mask_instance, cv::Size(roisrc.width, roisrc.height), cv::INTER_LINEAR);
                // 将掩码二值化 - 大于 0.5 的值设为 1（物体），小于等于 0.5 的值设为 0（背景）
                mask_instance = mask_instance > 0.5f;

                cv::Mat mask_instance_bgr;
                // 将单通道掩码转换为 3 通道 BGR 格式，以便应用颜色
                cv::cvtColor(mask_instance, mask_instance_bgr, cv::COLOR_GRAY2BGR);
                // 将掩码中的物体区域（值为1的区域）设置为当前类别的颜色
                mask_instance_bgr.setTo(color, mask_instance);
                cv::String det_info = class_names[label] + " " + cv::format("%.4f", ptr[4]);

                utils::Box box = utils::Box(x_lt_src, y_lt_src, x_rb_src, y_rb_src, conf, label, color);
                utils::label_info label_info = utils::label_info(label, conf, cv::Point(x_lt_src, y_lt_src), det_info, color);
                utils::mask_canvas mask = utils::mask_canvas(mask_instance_bgr, img_canvas, roisrc, 0.45f);

                boxes.push_back(box);
                labels.push_back(label_info);
                masks.push_back(mask);
                current_boxes++;
            }
        }
        
        if (is_show) cv_utils::draw(img, boxes, masks, labels, true, true, true, is_show);
        if (is_save) cv_utils::draw(img, boxes, masks, labels, false, true, false, is_show, is_save, windows_title, save_path);  

        // Post process result to Device
        CHECK(cudaFree(output1_conf_device));

        // Post process result to Host
        free(output0_src_host);
        free(output1_conf_host);
        if (is_debug) std::cout << "Post process done. Final Box Num: " << current_boxes << std::endl;
    }

   public:
    std::string model_name = "";
    std::string save_path = "";
    bool is_show = false;
    bool is_save = false;
    bool is_debug = false;
    int delaytime = 15;  // 15ms
    int num_class = 0;
    std::vector<std::string> class_names;
    std::vector<cv::Scalar> colors;

    int src_h = 0;
    int src_w = 0;
    int src_channel = 3;

    int batch_size = 1;
    int dst_channel = 3;
    int dst_h = 0;
    int dst_w = 0;
    int input_num = 0;

    int boxes_num = 0;
    int boxes_width = 0;
    int dst_boxes_width = 0;
    int mask_channel = 0;
    int mask_h = 0;
    int mask_w = 0;
    int output_num = 0;
    float iou_thresh = 0.0f;
    float conf_thresh = 0.0f;
    int topK = 0;

    const std::string windows_title = "Denglin-KS20-M";
    int char_width = 11;
    double font_scale = 0.6;
    int det_info_render_width = 15;

    // inference engine
    dlrt::dlrtParam dlrt_param;

    utils::AffineMat dst2src;
    utils::AffineMat src2dst;

    // input
    unsigned char* input_src_device;

    // output
    float* output1_conf_device;

    float* output1_conf_host;
    float* output0_src_host;
};
