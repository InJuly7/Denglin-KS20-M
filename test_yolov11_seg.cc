#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv.hpp>
#include <videoio.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <dlnne/dlnne.h>

#include "include/utils.h"
#include "include/op_gpu.h"
#include "include/op_cpu.h"

#include "include/yolov11_seg.h"

void setParameters(YoloParameter& initParameters) {
    initParameters.model_name = "yolov11s_seg";
    initParameters.onnx_model_path = "/data/model/yolov11/yolov11s_seg_transpose.onnx";
    initParameters.rlym_model_path = "/data/model/yolov11/yolov11s_seg_transpose.rlym";
    initParameters.quantize_model_path = "/data/model/yolov11/yolov11s_seg_transpose.quantized.rlym" initParameters.mlir_model_path =
        "/data/model/yolov11/yolov11s_seg_transpose.mlir";
    initParameters.quantize_mlir_model_path = "/data/model/yolov11/yolov11s_seg_transpose.quantized.mlir";
    initParameters.engine_path = "/data/model/yolov11/yolov11s_seg_transpose.slz";
    initParameters.quantize_engine_path = "/data/model/yolov11/yolov11s_seg_transpose.quantized.slz";

    initParameters.image_path = "/home/linaro/program/yolov11_seg_deepsort/ssd_horse.jpg";
    initParameters.save_path = "/home/linaro/program/yolov11_seg_deepsort/result.jpg";
    initParameters.video_path = "";

    initParameters.batch_size = 1;
    initParameters.dst_h = 640;
    initParameters.dst_w = 640;
    initParameters.channel = 3;
    initParameters.mask_channel = 32;
    initParameters.mask_grid_h = 160;
    initParameters.mask_grid_w = 160;

    // 原始图像大小，在读取图像后赋值
    initParameters.src_h = 0;
    initParameters.src_w = 0;

    initParameters.boxes_num = 8400;
    // xywh+class80+mask32
    initParameters.boxes_width = 116;
    initParameters.dst_boxes_width = 39;
    initParameters.num_class = 80;
    initParameters.class_names = dataSets::coco80;
    initParameters.colors = Colors::color80;

    initParameters.inputs = {{"images", {1, 3, 640, 640}}};
    initParameters.outputs = {{"output0", {1, 8400, 116}}, {"output1", {1, 32, 160, 160}}};

    initParameters.conf_thresh = 0.50f;
    initParameters.iou_thresh = 0.50f;
    initParameters.topK = 300;

    initParameters.is_debug = false;
    initParameters.is_save = false;
    initParameters.is_show = false;
    initParameters.delaytime = 15;  // 15ms
}

void task(YOLOV11_SEG& yolo, const YoloParameter& param, cv::Mat& img, const int& delayTime) {
    yolo.copy(img);
    DeviceTimer d_t1;
    yolo.preprocess();
    float duration_1 = d_t1.getUsedTime();

    HostTimer h_t2;
    yolo.infer();
    float duration_2 = h_t2.getUsedTime();

    HostTimer h_t3;
    yolo.postprocess(img);
    float duration_3 = h_t3.getUsedTime();

    std::cout << "YOLOV11s_seg 前处理: " << duration_1 << " ms 推理: " << duration_2 << " ms 后处理: " << duration_3 << "ms" << std::endl;
    yolo.reset();
}

int main(int argc, char* argv[]) {
    YoloParameter param;
    setParameters(param);

    InputStream source;
    // source = InputStream::VIDEO;
    source = InputStream::IMAGE;

    int delay_time = 15;
    cv::VideoCapture capture;
    // 设置 src_h, src_w, batch_size
    if (!setInputStream(source, param.image_path, param.video_path, capture, delay_time, param.batch_size, param.src_h, param.src_w)) {
        std::cout << "read the input data errors!" << std::endl;
        return -1;
    }

    YOLOV11_SEG yolo(param);
    int length = 0;
    std::vector<unsigned char> slz_file = loadModel(param.model_path, param.engine_path, length);
    if (slz_file.empty()) {
        std::cout << "slz_file is empty!" << std::endl;
        return -1;
    }
    if (param.is_debug) std::cout << "load Model finished" << std::endl;

    if (!yolo.init(slz_file, length)) {
        std::cout << "yolo init errors!" << std::endl;
        return -1;
    }
    if (param.is_debug) std::cout << "YOLOV11s_SEG Init finished" << std::endl;
    yolo.Alloc_buffer();
    if (param.is_debug) yolo.check();

    cv::Mat frame;
    while (capture.isOpened()) {
        if (source == InputStream::VIDEO) {
            capture.read(frame);
            task(yolo, param, frame, delay_time);
        } else if (source == InputStream::IMAGE) {
            frame = cv::imread(param.image_path);
            task(yolo, param, frame, delay_time);
            capture.release();
        }
    }
    yolo.Free_buffer();
    return 0;
}