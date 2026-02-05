#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "include/utils.h"
#include "include/cv.h"
#include "include/dlrt.h"
#include "include/sam2.h"

void task(SAM2& sam2, cv::Mat& img, const std::vector<utils::Box>& boxes, const std::vector<utils::Point>& points) {
    utils::DeviceTimer d_t1;
    sam2.encoder_preprocess(img);
    float duration_1 = d_t1.getUsedTime();

    utils::HostTimer h_t2;
    sam2.encoder_infer();
    float duration_2 = h_t2.getUsedTime();

    utils::HostTimer h_t3;
    sam2.decoder_preprocess(boxes, points);
    sam2.decoder_infer();
    sam2.decoder_postprocess(img);
    float duration_3 = h_t3.getUsedTime();

    std::cout << "SAM2 encoder前处理: " << duration_1 << " ms 推理: " << duration_2 << " ms decoder: " << duration_3 << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    const std::string config_path = (argc > 1) ? argv[1] : "config/sam2.ini";

    auto kv = utils::loadIni(config_path);
    std::string image_path = utils::getStr(kv, "image_path", "/home/linaro/program/yolov11_seg_deepsort/ssd_horse.jpg");
    std::string video_path = utils::getStr(kv, "video_path", "");
    int camera_id = utils::getInt(kv, "camera_id", -1);
    int delaytime = utils::getInt(kv, "delaytime", 15);
    utils::InputStream source = utils::parseInputStream(utils::getStr(kv, "input_stream", "image"));
    bool is_debug = utils::getBool(kv, "is_debug", true);

    // 输入原始图片的检测框坐标（用户提供）
    std::vector<utils::Box> boxes;
    boxes.emplace_back(822, 370, 1523, 576, 1.0f, 0);
    std::vector<utils::Point> points;

    int src_h = 0;
    int src_w = 0;
    int src_channel = 0;
    cv::VideoCapture capture;
    if (!cv_utils::setInputStream(source, image_path, video_path, camera_id, capture, src_h, src_w, src_channel, is_debug)) {
        std::cout << "read the input data errors!" << std::endl;
        return -1;
    }

    // 如需点提示，可在此处添加 points

    SAM2 sam_2;
    sam_2.init(config_path, src_h, src_w, src_channel);
    if (sam_2.is_debug) std::cout << "SAM2 Init finished" << std::endl;

    cv::Mat frame;
    while (capture.isOpened()) {
        if (source == utils::InputStream::VIDEO) {
            capture.read(frame);
            task(sam_2, frame, boxes, points);
            if (delaytime > 0) {
                cv::waitKey(delaytime);
            }
        } else if (source == utils::InputStream::IMAGE) {
            frame = cv::imread(image_path);
            task(sam_2, frame, boxes, points);
            capture.release();
        }
    }

    dlrt::BufferFree(sam_2.dlrt_param);
    ort::BufferFree(sam_2.ort_param);
    return 0;
}