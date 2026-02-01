#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <cctype>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <dlnne/dlnne.h>

#include "include/utils.h"
#include "include/cv.h"
#include "include/dlrt.h"
#include "include/yolov11_seg.h"



void task(YOLOV11_SEG& yolo, cv::Mat& img) {
    utils::DeviceTimer d_t1;
    yolo.preprocess(img);
    float duration_1 = d_t1.getUsedTime();

    utils::HostTimer h_t2;
    yolo.infer();
    float duration_2 = h_t2.getUsedTime();

    utils::HostTimer h_t3;
    yolo.postprocess(img);
    float duration_3 = h_t3.getUsedTime();

    std::cout << "YOLOV11s_seg 前处理: " << duration_1 << " ms 推理: " << duration_2 << " ms 后处理: " << duration_3 << "ms" << std::endl;
}

int main(int argc, char* argv[]) {
    const std::string config_path = (argc > 1) ? argv[1] : "config/yolov11_seg.ini";
    
    auto kv = utils::loadIni(config_path);
    std::string image_path = utils::getStr(kv, "image_path", "/home/linaro/program/yolov11_seg_deepsort/ssd_horse.jpg");
    std::string video_path = utils::getStr(kv, "video_path", "");
    int camera_id = utils::getInt(kv, "camera_id", -1);
    int delaytime = utils::getInt(kv, "delaytime", 15);
    utils::InputStream source = utils::parseInputStream(utils::getStr(kv, "input_stream", "image"));
    bool is_debug = utils::getBool(kv, "is_debug", true);

    int src_h = 0;
    int src_w = 0;
    int src_channel = 0;
    cv::VideoCapture capture;
    if (!cv_utils::setInputStream(source, image_path, video_path, camera_id, capture, src_h, src_w, src_channel, is_debug)) {
        std::cout << "read the input data errors!" << std::endl;
        return -1;
    }
    YOLOV11_SEG yolo_seg;
    yolo_seg.init(config_path, src_h, src_w, src_channel);
    if (yolo_seg.is_debug) std::cout << "YOLOV11s_SEG Init finished" << std::endl;
    if (yolo_seg.is_debug) dlrt::check_engine_info(yolo_seg.dlrt_param.engine_context);

    cv::Mat frame;
    while (capture.isOpened()) {
        if (source == utils::InputStream::VIDEO) {
            capture.read(frame);
            task(yolo_seg, frame);
        } else if (source == utils::InputStream::IMAGE) {
            frame = cv::imread(image_path);
            task(yolo_seg, frame);
            capture.release();
        }
    }
    dlrt::BufferFree(yolo_seg.dlrt_param);
    return 0;
}