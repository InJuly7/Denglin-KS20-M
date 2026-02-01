#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <dlnne/dlnne.h>

#include "include/utils.h"
#include "include/cv.h"
#include "include/dlrt.h"
#include "include/sam2.h"

void setParameters(SAM2Parameter& initParameters) {
    initParameters.model_name = "sam2_image_encoder_s";
    initParameters.onnx_model_path = "/data/model/sam2/image_encoder_s.onnx";
    initParameters.rlym_model_path = "/data/model/sam2/image_encoder_s.rlym";
    initParameters.quantize_model_path = "/data/model/sam2/image_encoder_s.quantized.rlym";
    initParameters.mlir_model_path = "/data/model/sam2/image_encoder_s.mlir";
    initParameters.quantize_mlir_model_path = "/data/model/sam2/image_encoder_s.quantized.mlir";
    initParameters.engine_path = "/data/model/sam2/image_encoder_s.slz";
    initParameters.quantize_engine_path = "/data/model/sam2/image_encoder_s.quantized.slz";

    initParameters.decoder_path = "/data/model/sam2/image_decoder_s.onnx";

    initParameters.image_path = "/home/linaro/program/yolov11_seg_deepsort/ssd_horse.jpg";
    initParameters.save_path = "/home/linaro/program/yolov11_seg_deepsort/result.jpg";
    initParameters.video_path = "";

    initParameters.batch_size = 1;
    initParameters.channel = 3;
    initParameters.dst_h = 1024;
    initParameters.dst_w = 1024;

    initParameters.high_res_feats_0_C = 32;
    initParameters.high_res_feats_0_H = 256;
    initParameters.high_res_feats_0_W = 256;

    initParameters.high_res_feats_1_C = 128;
    initParameters.high_res_feats_1_H = 128;
    initParameters.high_res_feats_1_H = 128;

    initParameters.image_embed_C = 256;
    initParameters.image_embed_H = 64;
    initParameters.image_embed_W = 64;

    // 原始图像大小，在读取图像后赋值
    initParameters.src_h = 0;
    initParameters.src_w = 0;

    // Decode点的个数, 读取检测框, 点后赋值
    initParameters.num_points = 0;

    // Encoder模型 静态shape
    initParameters.encoder_inputs = {{"image", {1, 3, 1024, 1024}}};
    initParameters.encoder_outputs = {
        {"high_res_feats_0", {1, 32, 256, 256}}, {"high_res_feats_1", {1, 64, 128, 128}}, {"image_embed", {1, 256, 64, 64}}};

    initParameters.is_debug = true;
    initParameters.is_save = false;
    initParameters.is_show = false;
    initParameters.topK = 100;
    initParameters.delaytime = 15;  // 15ms
}

void task(SAM2& sam2, const SAM2Parameter& param, cv::Mat& img, const int& delayTime) {
    sam2.copy(img);
    utils::DeviceTimer d_t1;
    sam2.preprocess();
    float duration_1 = d_t1.getUsedTime();

    utils::HostTimer h_t2;
    sam2.infer();
    float duration_2 = h_t2.getUsedTime();

    utils::HostTimer h_t3;
    sam2.postprocess(img);
    float duration_3 = h_t3.getUsedTime();

    sam2.reset();
}

int main(int argc, char* argv[]) {
    // 输入原始图片 与 检测框坐标
    /*
        label: 0 conf 0.836942
        x_lt_src: 431; y_lt_src: 124; x_rb_src: 450; y_rb_src: 178
        
        label: 0 conf 0.887545
        x_lt_src: 272; y_lt_src: 13; x_rb_src: 348; y_rb_src: 236
        
        label: 7 conf 0.751577
        x_lt_src: 1; y_lt_src: 105; x_rb_src: 132; y_rb_src: 197
        
        label: 17 conf 0.946285
        x_lt_src: 217; y_lt_src: 72; x_rb_src: 421; y_rb_src: 374
        
        label: 16 conf 0.899929
        x_lt_src: 144; y_lt_src: 203; x_rb_src: 196; y_rb_src: 345
    */
    std::vector<utils::Box> boxes;
    boxes.push_back(utils::Box(431, 124, 450, 178, 0.836942, 0));
    boxes.push_back(utils::Box(272, 13, 348, 236, 0.887545, 0));
    boxes.push_back(utils::Box(1, 105, 132, 197, 0.751577, 7));
    boxes.push_back(utils::Box(217, 72, 421, 374, 0.946285, 17));
    boxes.push_back(utils::Box(144, 203, 196, 345, 0.899929, 16));

    std::vector<utils::Point> points;
    
    SAM2Parameter param;
    // 设置默认参数
    setParameters(param);

    utils::InputStream source;
    // source = utils::InputStream::VIDEO;
    source = utils::InputStream::IMAGE;

    int delay_time = 15;
    cv::VideoCapture capture;
    // 设置 src_h, src_w, batch_size
    if (!cv_utils::setInputStream(source, param.image_path, param.video_path, param.camera_id, capture, delay_time, param.src_h, param.src_w)) {
        std::cout << "read the input data errors!" << std::endl;
        return -1;
    }
    // 设置 
    SAM2 sam2(param);
    int length = 0;
    std::vector<unsigned char> slz_file = dlrt::loadModel(param.engine_path, param.engine_path, length);
    if (slz_file.empty()) {
        std::cout << "slz_file is empty!" << std::endl;
        return -1;
    }
    if (param.is_debug) std::cout << "load Model finished" << std::endl;

    if (!sam2.init()) {
        std::cout << "sam2 init errors!" << std::endl;
        return -1;
    }
    if (param.is_debug) std::cout << "YOLOV11s_SEG Init finished" << std::endl;
    sam2.Alloc_buffer();
    if (param.is_debug) sam2.check();

    cv::Mat frame;
    while (capture.isOpened()) {
        if (source == utils::InputStream::VIDEO) {
            capture.read(frame);
            sam2.task(sam2, param, frame, delay_time);
        } else if (source == utils::InputStream::IMAGE) {
            frame = cv::imread(param.image_path);
            sam2.task(sam2, param, frame, delay_time);
            capture.release();
        }
    }
    sam2.Free_buffer();
    return 0;
}