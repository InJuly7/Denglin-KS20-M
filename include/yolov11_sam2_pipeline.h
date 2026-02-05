#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "yolov11.h"
#include "sam2.h"

class YOLOV11_SAM2_PIPELINE {
   public:
    YOLOV11_SAM2_PIPELINE() = default;
    ~YOLOV11_SAM2_PIPELINE() {}

    void init(const std::string& yolo_inifile, const std::string& sam2_inifile,
              const int src_h, const int src_w, const int src_channel) {
        yolo.init(yolo_inifile, src_h, src_w, src_channel);
        sam2.init(sam2_inifile, src_h, src_w, src_channel);
    }

    void run(cv::Mat& img) {
        // YOLOV11: preprocess + infer + postprocess (不绘制，仅保留 boxes)
        yolo.preprocess(img);
        yolo.infer();
        yolo.postprocess(img, false);

        // SAM2: encoder + decoder (使用 YOLO 的 boxes)
        // 若没有检测到框，跳过 SAM2
        if (yolo.last_boxes.empty()) {
            return;
        }

        sam2.encoder_preprocess(img);
        sam2.encoder_infer();
        sam2.decoder_preprocess(yolo.last_boxes, {});
        sam2.decoder_infer();
        sam2.decoder_postprocess(img);
    }

   public:
    YOLOV11 yolo;
    SAM2 sam2;
};
