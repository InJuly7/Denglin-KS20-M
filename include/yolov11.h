#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <dlnne/dlnne.h>

#include "op_gpu.h"
#include "utils.h"
#include "cv.h"

class YOLOV11 {
   public:
    YOLOV11() = default;
    ~YOLOV11() {}

    void init(const std::string inifile, const int _src_h, const int _src_w, const int _src_channel) {
        auto kv = utils::loadIni(inifile);
        model_name = utils::getStr(kv, "model_name", "yolov11s");
        std::string model_dir = utils::getStr(kv, "model_dir", "");
        save_path = utils::getStr(kv, "save_path", "/home/linaro/program/yolov11_seg_deepsort/result.jpg");
        is_debug = utils::getBool(kv, "is_debug", true);
        is_save = utils::getBool(kv, "is_save", false);
        is_show = utils::getBool(kv, "is_show", false);
        delaytime = utils::getInt(kv, "delaytime", 15);
        num_class = utils::getInt(kv, "num_class", 80);
        class_names = utils::dataSets::coco80;
        colors = cv_utils::Colors::color80;

        // Preprocess
        src_h = _src_h;
        src_w = _src_w;
        src_channel = _src_channel;
        batch_size = utils::getInt(kv, "batch_size", 1);
        dst_channel = utils::getInt(kv, "dst_channel", 3);
        dst_h = utils::getInt(kv, "dst_h", 640);
        dst_w = utils::getInt(kv, "dst_w", 640);
        input_num = utils::getInt(kv, "input_num", 1);

        // Postprocess
        boxes_num = utils::getInt(kv, "boxes_num", 8400);
        boxes_width = utils::getInt(kv, "boxes_width", 84);
        dst_boxes_width = utils::getInt(kv, "dst_boxes_width", 7);
        output_num = utils::getInt(kv, "output_num", 1);
        conf_thresh = utils::getFloat(kv, "conf_thresh", 0.50f);
        iou_thresh = utils::getFloat(kv, "iou_thresh", 0.50f);
        topK = utils::getInt(kv, "topK", 300);

        // Inference
        dlrt::dlrtInit(model_name, dlrt_param, input_num, output_num, is_debug, model_dir);

        // Affine matrix
        utils::Affine_Matrix(src_w, src_h, dst_w, dst_h, src2dst, dst2src);
    }

    void preprocess(const cv::Mat& img) {
        input_src_device = nullptr;
        CHECK(cudaMalloc(&input_src_device, src_channel * src_h * src_w * sizeof(unsigned char)));
        CHECK(cudaMemcpy(input_src_device, img.data, sizeof(unsigned char) * src_channel * src_h * src_w, cudaMemcpyHostToDevice));
        affine_bilinear(input_src_device, src_w, src_h, dlrt_param.input_buffers[0].device_ptr, dst_w, dst_h, dst2src);
        if (is_debug) std::cout << "Preprocess done." << std::endl;
        CHECK(cudaFree(input_src_device));
    }

    bool infer() {
        bool success = dlrt::infer(dlrt_param);
        assert(success == true);
        CHECK(cudaDeviceSynchronize());
        if (is_debug) std::cout << "Inference done." << std::endl;
        return success;
    }

    void postprocess(cv::Mat& img, bool draw_result = true) {
        utils::DeviceTimer t0;
        output1_conf_device = nullptr;
        CHECK(cudaMalloc(&output1_conf_device, (1 + topK * dst_boxes_width) * sizeof(float)));
        CHECK(cudaMemset(output1_conf_device, 0, (1 + topK * dst_boxes_width) * sizeof(float)));

        output1_conf_host = (float*)malloc((1 + topK * dst_boxes_width) * sizeof(float));

        conf_filter_nomask(dlrt_param.output_buffers[0].device_ptr, boxes_width, boxes_num, output1_conf_device, dst_boxes_width, topK,
                           num_class, conf_thresh);
        nms_fast(output1_conf_device, dst_boxes_width, topK, iou_thresh);

        CHECK(cudaDeviceSynchronize());

        cudaMemcpy(output1_conf_host, output1_conf_device, (1 + topK * dst_boxes_width) * sizeof(float), cudaMemcpyDeviceToHost);
        if (is_debug) std::cout << "Post process GPU: " << t0.getUsedTime() << " ms" << std::endl;

        int num_boxes = std::min((int)(output1_conf_host)[0], topK);
        int current_boxes = 0;
        if (is_debug) std::cout << "Bounding Box Num: " << num_boxes << std::endl;

        last_boxes.clear();
        last_labels.clear();

        for (int box_idx = 0; box_idx < num_boxes; ++box_idx) {
            float* ptr = output1_conf_host + 1 + box_idx * dst_boxes_width;
            if (!ptr[6]) {
                continue;
            }
            int label = static_cast<int>(ptr[5]);
            float conf = ptr[4];
            if (label < 0 || label >= num_class) {
                continue;
            }
            cv::Scalar color = colors[label];
            std::string class_name = class_names[label];

            int x_lt_src = std::round(dst2src.v0 * ptr[0] + dst2src.v1 * ptr[1] + dst2src.v2);
            int y_lt_src = std::round(dst2src.v3 * ptr[0] + dst2src.v4 * ptr[1] + dst2src.v5);
            int x_rb_src = std::round(dst2src.v0 * ptr[2] + dst2src.v1 * ptr[3] + dst2src.v2);
            int y_rb_src = std::round(dst2src.v3 * ptr[2] + dst2src.v4 * ptr[3] + dst2src.v5);

            utils::Box box = utils::Box(x_lt_src, y_lt_src, x_rb_src, y_rb_src, conf, label, color);
            cv::String det_info = class_name + " " + cv::format("%.4f", conf);
            utils::label_info label_info = utils::label_info(label, conf, cv::Point(x_lt_src, y_lt_src), det_info, color);

            last_boxes.push_back(box);
            last_labels.push_back(label_info);
            current_boxes++;
        }

        if (draw_result) {
            cv_utils::draw(img, last_boxes, {}, last_labels, true, false, true, is_show, is_save, windows_title, save_path);
        }

        CHECK(cudaFree(output1_conf_device));
        free(output1_conf_host);
        if (is_debug) std::cout << "Post process done. Final Box Num: " << current_boxes << std::endl;
    }

   public:
    std::string model_name = "";
    std::string save_path = "";
    bool is_show = false;
    bool is_save = false;
    bool is_debug = false;
    int delaytime = 15;
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
    int output_num = 0;
    float iou_thresh = 0.0f;
    float conf_thresh = 0.0f;
    int topK = 0;

    const std::string windows_title = "Denglin-KS20-M";

    dlrt::dlrtParam dlrt_param;

    utils::AffineMat dst2src;
    utils::AffineMat src2dst;

    unsigned char* input_src_device;

    float* output1_conf_device;
    float* output1_conf_host;

    std::vector<utils::Box> last_boxes;
    std::vector<utils::label_info> last_labels;
};
