#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "include/utils.h"
#include "include/cv.h"
#include "include/dlrt.h"
#include "include/yolov11_sam2_pipeline.h"

void task(YOLOV11_SAM2_PIPELINE& pipeline, cv::Mat& img) {
    utils::HostTimer t;
    pipeline.run(img);
    float duration = t.getUsedTime();
    std::cout << "YOLOV11+SAM2 pipeline total: " << duration << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    const std::string yolo_config_path = (argc > 1) ? argv[1] : "config/yolov11.ini";
    const std::string sam2_config_path = (argc > 2) ? argv[2] : "config/sam2.ini";

    auto kv = utils::loadIni(yolo_config_path);
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

    YOLOV11_SAM2_PIPELINE pipeline;
    pipeline.init(yolo_config_path, sam2_config_path, src_h, src_w, src_channel);
    if (pipeline.yolo.is_debug) std::cout << "YOLOV11 Init finished" << std::endl;
    if (pipeline.sam2.is_debug) std::cout << "SAM2 Init finished" << std::endl;

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
            bool orig_save = pipeline.sam2.is_save;
            pipeline.sam2.is_save = false;
            task(pipeline, frame);
            pipeline.sam2.is_save = orig_save;
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
            task(pipeline, frame);
            capture.release();
        }
    }

    if (writer.isOpened()) {
        writer.release();
    }
    dlrt::BufferFree(pipeline.yolo.dlrt_param);
    dlrt::BufferFree(pipeline.sam2.dlrt_param);
    ort::BufferFree(pipeline.sam2.ort_param);
    return 0;
}
