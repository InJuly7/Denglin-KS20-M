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
#include "include/yolov11.h"

void task(YOLOV11& yolo, cv::Mat& img) {
    utils::DeviceTimer d_t1;
    yolo.preprocess(img);
    float duration_1 = d_t1.getUsedTime();

    utils::HostTimer h_t2;
    yolo.infer();
    float duration_2 = h_t2.getUsedTime();

    utils::HostTimer h_t3;
    yolo.postprocess(img);
    float duration_3 = h_t3.getUsedTime();

    std::cout << "YOLOV11 前处理: " << duration_1 << " ms 推理: " << duration_2 << " ms 后处理: " << duration_3 << "ms" << std::endl;
}

int main(int argc, char* argv[]) {
    const std::string config_path = (argc > 1) ? argv[1] : "config/yolov11.ini";

    auto kv = utils::loadIni(config_path);
    std::string image_path = utils::getStr(kv, "image_path", "/home/linaro/program/yolov11_seg_deepsort/ssd_horse.jpg");
    std::string video_path = utils::getStr(kv, "video_path", "");
    int camera_id = utils::getInt(kv, "camera_id", -1);
    int delaytime = utils::getInt(kv, "delaytime", 15);
    std::string source_str = utils::getStr(kv, "input_stream", "");
    if (source_str.empty()) {
        source_str = utils::getStr(kv, "input_source", "image");
    }
    size_t comment_pos = source_str.find_first_of("#;");
    if (comment_pos != std::string::npos) {
        source_str = source_str.substr(0, comment_pos);
    }
    source_str = utils::trim(source_str);
    utils::InputStream source = utils::parseInputStream(source_str);
    bool is_debug = utils::getBool(kv, "is_debug", true);
    bool is_save = utils::getBool(kv, "is_save", false);
    std::string save_path = utils::getStr(kv, "save_path", "/home/linaro/program/yolov11_seg_deepsort/result.mp4");

    int src_h = 0;
    int src_w = 0;
    int src_channel = 0;
    cv::VideoCapture capture;
    if (!cv_utils::setInputStream(source, image_path, video_path, camera_id, capture, src_h, src_w, src_channel, is_debug)) {
        std::cout << "read the input data errors!" << std::endl;
        return -1;
    }

    YOLOV11 yolo;
    yolo.init(config_path, src_h, src_w, src_channel);
    if (yolo.is_debug) std::cout << "YOLOV11 Init finished" << std::endl;
    if (yolo.is_debug) dlrt::check_engine_info(yolo.dlrt_param.engine_context);

    cv::Mat frame;
    cv::VideoWriter writer;
    bool save_video = (source == utils::InputStream::VIDEO) && is_save;
    if (save_video) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = capture.get(cv::CAP_PROP_FPS);
        if (fps <= 0.0) fps = 25.0;
        int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (!writer.open(save_path, fourcc, fps, cv::Size(width, height), true)) {
            std::cout << "open video writer failed: " << save_path << std::endl;
            save_video = false;
        }
    }
    while (capture.isOpened()) {
        if (source == utils::InputStream::VIDEO) {
            if (!capture.read(frame) || frame.empty()) {
                break;
            }
            // 视频保存由 VideoWriter 处理，避免每帧覆盖图片
            bool orig_save = yolo.is_save;
            yolo.is_save = false;
            task(yolo, frame);
            yolo.is_save = orig_save;
            if (save_video && !frame.empty()) {
                writer.write(frame);
            }
            if (delaytime > 0) {
                cv::waitKey(delaytime);
            }
        } else if (source == utils::InputStream::IMAGE) {
            frame = cv::imread(image_path);
            if (frame.empty()) {
                std::cout << "read image failed: " << image_path << std::endl;
                capture.release();
                break;
            }
            task(yolo, frame);
            capture.release();
        }
    }

    if (writer.isOpened()) {
        writer.release();
    }
    dlrt::BufferFree(yolo.dlrt_param);
    return 0;
}
