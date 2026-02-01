#pragma once

#include <climits>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "utils.h"

namespace cv_utils {

namespace Colors {
    const std::vector<cv::Scalar> color80{
        cv::Scalar(128, 77, 207),  cv::Scalar(65, 32, 208),   cv::Scalar(0, 224, 45),    cv::Scalar(3, 141, 219),
        cv::Scalar(80, 239, 253),  cv::Scalar(239, 184, 12),  cv::Scalar(7, 144, 145),   cv::Scalar(161, 88, 57),
        cv::Scalar(0, 166, 46),    cv::Scalar(218, 113, 53),  cv::Scalar(193, 33, 128),  cv::Scalar(190, 94, 113),
        cv::Scalar(113, 123, 232), cv::Scalar(69, 205, 80),   cv::Scalar(18, 170, 49),   cv::Scalar(89, 51, 241),
        cv::Scalar(153, 191, 154), cv::Scalar(27, 26, 69),    cv::Scalar(20, 186, 194),  cv::Scalar(210, 202, 167),
        cv::Scalar(196, 113, 204), cv::Scalar(9, 81, 88),     cv::Scalar(191, 162, 67),  cv::Scalar(227, 73, 120),
        cv::Scalar(177, 31, 19),   cv::Scalar(133, 102, 137), cv::Scalar(146, 72, 97),   cv::Scalar(145, 243, 208),
        cv::Scalar(2, 184, 176),   cv::Scalar(219, 220, 93),  cv::Scalar(238, 153, 134), cv::Scalar(197, 169, 160),
        cv::Scalar(204, 201, 106), cv::Scalar(13, 24, 129),   cv::Scalar(40, 38, 4),     cv::Scalar(5, 41, 34),
        cv::Scalar(46, 94, 129),   cv::Scalar(102, 65, 107),  cv::Scalar(27, 11, 208),   cv::Scalar(191, 240, 183),
        cv::Scalar(225, 76, 38),   cv::Scalar(193, 89, 124),  cv::Scalar(30, 14, 175),   cv::Scalar(144, 96, 90),
        cv::Scalar(181, 186, 86),  cv::Scalar(102, 136, 34),  cv::Scalar(158, 71, 15),   cv::Scalar(183, 81, 247),
        cv::Scalar(73, 69, 89),    cv::Scalar(123, 73, 232),  cv::Scalar(4, 175, 57),    cv::Scalar(87, 108, 23),
        cv::Scalar(105, 204, 142), cv::Scalar(63, 115, 53),   cv::Scalar(105, 153, 126), cv::Scalar(247, 224, 137),
        cv::Scalar(136, 21, 188),  cv::Scalar(122, 129, 78),  cv::Scalar(145, 80, 81),   cv::Scalar(51, 167, 149),
        cv::Scalar(162, 173, 20),  cv::Scalar(252, 202, 17),  cv::Scalar(10, 40, 3),     cv::Scalar(150, 90, 254),
        cv::Scalar(169, 21, 68),   cv::Scalar(157, 148, 180), cv::Scalar(131, 254, 90),  cv::Scalar(7, 221, 102),
        cv::Scalar(19, 191, 184),  cv::Scalar(98, 126, 199),  cv::Scalar(210, 61, 56),   cv::Scalar(252, 86, 59),
        cv::Scalar(102, 195, 55),  cv::Scalar(160, 26, 91),   cv::Scalar(60, 94, 66),    cv::Scalar(204, 169, 193),
        cv::Scalar(126, 4, 181),   cv::Scalar(229, 209, 196), cv::Scalar(195, 170, 186), cv::Scalar(155, 207, 148)};
    const std::vector<cv::Scalar> color91{
        cv::Scalar(148, 99, 164),  cv::Scalar(65, 172, 90),   cv::Scalar(18, 117, 190),  cv::Scalar(173, 208, 229),
        cv::Scalar(37, 162, 147),  cv::Scalar(121, 99, 42),   cv::Scalar(218, 173, 104), cv::Scalar(193, 213, 138),
        cv::Scalar(142, 168, 45),  cv::Scalar(107, 143, 94),  cv::Scalar(242, 89, 7),    cv::Scalar(87, 218, 248),
        cv::Scalar(126, 168, 9),   cv::Scalar(86, 152, 105),  cv::Scalar(155, 135, 251), cv::Scalar(73, 234, 44),
        cv::Scalar(177, 37, 42),   cv::Scalar(219, 215, 54),  cv::Scalar(124, 207, 143), cv::Scalar(7, 81, 209),
        cv::Scalar(254, 18, 130),  cv::Scalar(71, 54, 73),    cv::Scalar(172, 198, 63),  cv::Scalar(64, 217, 224),
        cv::Scalar(105, 224, 25),  cv::Scalar(41, 52, 130),   cv::Scalar(220, 27, 193),  cv::Scalar(65, 222, 86),
        cv::Scalar(250, 150, 201), cv::Scalar(201, 150, 105), cv::Scalar(104, 96, 142),  cv::Scalar(111, 230, 54),
        cv::Scalar(105, 24, 22),   cv::Scalar(42, 226, 101),  cv::Scalar(67, 26, 144),   cv::Scalar(155, 113, 106),
        cv::Scalar(152, 196, 216), cv::Scalar(58, 68, 152),   cv::Scalar(68, 230, 213),  cv::Scalar(169, 143, 129),
        cv::Scalar(191, 102, 41),  cv::Scalar(5, 73, 170),    cv::Scalar(15, 73, 233),   cv::Scalar(95, 13, 71),
        cv::Scalar(25, 92, 218),   cv::Scalar(85, 173, 16),   cv::Scalar(247, 158, 17),  cv::Scalar(36, 28, 8),
        cv::Scalar(31, 100, 134),  cv::Scalar(131, 71, 45),   cv::Scalar(158, 190, 91),  cv::Scalar(90, 207, 220),
        cv::Scalar(125, 77, 228),  cv::Scalar(40, 156, 67),   cv::Scalar(35, 250, 69),   cv::Scalar(229, 61, 245),
        cv::Scalar(210, 201, 106), cv::Scalar(184, 35, 131),  cv::Scalar(47, 124, 120),  cv::Scalar(1, 114, 23),
        cv::Scalar(99, 181, 17),   cv::Scalar(77, 141, 151),  cv::Scalar(79, 33, 95),    cv::Scalar(194, 111, 146),
        cv::Scalar(187, 199, 138), cv::Scalar(129, 215, 40),  cv::Scalar(160, 209, 144), cv::Scalar(139, 121, 58),
        cv::Scalar(97, 208, 197),  cv::Scalar(185, 105, 171), cv::Scalar(160, 96, 136),  cv::Scalar(232, 26, 26),
        cv::Scalar(34, 165, 109),  cv::Scalar(19, 86, 215),   cv::Scalar(205, 209, 199), cv::Scalar(131, 91, 25),
        cv::Scalar(51, 201, 16),   cv::Scalar(64, 35, 128),   cv::Scalar(120, 161, 247), cv::Scalar(123, 164, 190),
        cv::Scalar(15, 191, 40),   cv::Scalar(11, 44, 117),   cv::Scalar(198, 136, 70),  cv::Scalar(14, 224, 240),
        cv::Scalar(60, 186, 193),  cv::Scalar(253, 190, 129), cv::Scalar(134, 228, 173), cv::Scalar(219, 156, 214),
        cv::Scalar(137, 67, 254),  cv::Scalar(178, 223, 250), cv::Scalar(219, 199, 139)};
    const std::vector<cv::Scalar> color20{
        cv::Scalar(128, 77, 207),  cv::Scalar(65, 32, 208),  cv::Scalar(0, 224, 45),   cv::Scalar(3, 141, 219),
        cv::Scalar(80, 239, 253),  cv::Scalar(239, 184, 12), cv::Scalar(7, 144, 145),  cv::Scalar(161, 88, 57),
        cv::Scalar(0, 166, 46),    cv::Scalar(218, 113, 53), cv::Scalar(193, 33, 128), cv::Scalar(190, 94, 113),
        cv::Scalar(113, 123, 232), cv::Scalar(69, 205, 80),  cv::Scalar(18, 170, 49),  cv::Scalar(89, 51, 241),
        cv::Scalar(153, 191, 154), cv::Scalar(27, 26, 69),   cv::Scalar(20, 186, 194), cv::Scalar(210, 202, 167),
        cv::Scalar(196, 113, 204), cv::Scalar(9, 81, 88),    cv::Scalar(191, 162, 67), cv::Scalar(227, 73, 120),
        cv::Scalar(177, 31, 19)};
    const std::vector<cv::Scalar> color15{
        cv::Scalar(128, 77, 207),  cv::Scalar(65, 32, 208),  cv::Scalar(0, 224, 45),   cv::Scalar(3, 141, 219),
        cv::Scalar(80, 239, 253),  cv::Scalar(239, 184, 12), cv::Scalar(7, 144, 145),  cv::Scalar(161, 88, 57),
        cv::Scalar(0, 166, 46),    cv::Scalar(218, 113, 53), cv::Scalar(193, 33, 128), cv::Scalar(190, 94, 113),
        cv::Scalar(113, 123, 232), cv::Scalar(69, 205, 80),  cv::Scalar(18, 170, 49),  cv::Scalar(89, 51, 241),
        cv::Scalar(153, 191, 154), cv::Scalar(27, 26, 69),   cv::Scalar(20, 186, 194), cv::Scalar(210, 202, 167)};
};  // namespace Colors

static bool setInputStream(const utils::InputStream& source, const std::string& imagePath, const std::string& videoPath,
                           const int& cameraID, cv::VideoCapture& capture, int& src_h, int& src_w, int& src_channel, bool is_debug = false) {
    switch (source) {
        case utils::InputStream::IMAGE:
            capture.open(imagePath);
            src_h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
            src_w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
            src_channel = 3;
            if (is_debug) std::cout << "src_h: " << src_h << " src_w: " << src_w << " src_channel: " << src_channel << std::endl;
            break;
        case utils::InputStream::VIDEO:
            capture.open(videoPath);
            src_h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
            src_w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
            src_channel = 3;
            break;
        case utils::InputStream::CAMERA:
            capture.open(cameraID);
            src_h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
            src_w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
            src_channel = 3;
            break;
        default:
            break;
    }
    return capture.isOpened();
};

static bool readFrame(cv::VideoCapture& capture, cv::Mat& frame) {
    return capture.read(frame);
}

static bool draw_bb(cv::Mat& img, const std::vector<utils::Box>& objectss) {
    if (img.empty() || objectss.empty()) {
        return false;
    }
    for (auto& box : objectss) {
        cv::rectangle(img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), box.color, 2, cv::LINE_AA);
    }
    return true;
}

static bool draw_label_info(cv::Mat& img, const std::vector<utils::label_info>& label_infos) {
    if (img.empty() || label_infos.empty()) return false;
    cv::Point text_box[1][4];
    const cv::Point* text_box_point[1] = {text_box[0]};
    int num_points[] = {4};
    for (auto& label : label_infos) {
        text_box[0][0] = cv::Point(label.tl.x, label.tl.y);
        text_box[0][1] = cv::Point(label.tl.x + label.info.size() * label.char_width, label.tl.y);
        text_box[0][2] = cv::Point(label.tl.x + label.info.size() * label.char_width, label.tl.y - label.det_info_render_width);
        text_box[0][3] = cv::Point(label.tl.x, label.tl.y - label.det_info_render_width);
        cv::fillPoly(img, text_box_point, num_points, 1, label.color);
        cv::putText(img, label.info, text_box[0][0], cv::FONT_HERSHEY_DUPLEX, label.font_scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    return true;
}

static bool draw_mask(cv::Mat& img, const std::vector<utils::mask_canvas>& masks) {
    if (img.empty() || masks.empty()) {
        return false;
    }
    for (const auto& mask : masks) {
        if (!mask.mask_instance_bgr.empty()) {
            // 将彩色掩码与图像画布叠加 掩码区域权重为 0.45，检测框区域权重为 1.0，创建半透明效果
            cv::addWeighted(mask.mask_instance_bgr, mask.weight, mask.canvas(mask.roisrc), 1.0, 0., masks[0].canvas(mask.roisrc));
        }
    }
    img = img + masks[0].canvas;
    return true;
}

static bool draw(cv::Mat& img, const std::vector<utils::Box>& objects, const std::vector<utils::mask_canvas>& masks = {},
                 const std::vector<utils::label_info>& labelInfos = {}, bool enable_draw_bb = false, bool enable_draw_mask = false,
                 bool enable_draw_label_info = false, bool is_show = false, bool is_save = false,
                 const std::string windows_title = "Denglin-KS20-M", const std::string save_path = "output.jpg") {
    cv::Mat src_img = img.clone();
    if (img.empty()) {
        return false;
    }

    if (is_show) {
        cv::namedWindow(windows_title, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);  // allow window resize(Linux)
        int max_w = 960;
        int max_h = 540;
        if (img.rows > max_h || img.cols > max_w) {
            cv::resizeWindow(windows_title, max_w, img.rows * max_w / img.cols);
        }
    }

    if (enable_draw_mask) {
        if (!masks.empty()) {
            draw_mask(img, masks);
        }
    }

    if (enable_draw_bb) {
        if (!objects.empty()) {
            draw_bb(img, objects);
        }
    }

    if (enable_draw_label_info) {
        if (!labelInfos.empty()) {
            draw_label_info(img, labelInfos);
        }
    }

    if (is_show) cv::imshow(windows_title, img);
    if (is_save) cv::imwrite(save_path, img);
    return true;
}

// void setRenderWindow(YoloParameter& param) {
//     if (!param.is_show) return;
//     int max_w = 960;
//     int max_h = 540;
//     float scale_h = (float)param.src_h / max_h;
//     float scale_w = (float)param.src_w / max_w;
//     if (scale_h > 1.f && scale_w > 1.f) {
//         float scale = scale_h < scale_w ? scale_h : scale_w;
//         cv::namedWindow(param.winname, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);  // for Linux
//         cv::resizeWindow(param.winname, int(param.src_w / scale), int(param.src_h / scale));
//         param.char_width = 16;
//         param.det_info_render_width = 18;
//         param.font_scale = 0.9;
//     } else {
//         cv::namedWindow(param.winname);
//     }
// }

// void draw_Text(cv::Mat& image, const std::string& text, const cv::Point& position, const cv::Scalar& color) {
//     cv::putText(image,
//                 text,                      // 文字内容
//                 position,                  // 位置 (x, y)
//                 cv::FONT_HERSHEY_SIMPLEX,  // 字体
//                 0.6,                       // 字体大小
//                 color,                     // 颜色
//                 2,                         // 线条粗细
//                 cv::LINE_AA);              // 抗锯齿
// }

// // 创建自适应大小的OpenCV显示窗口(主要用于Linux)
// void createAdaptiveWindow(const cv::Mat& img, const std::string& windows_title, int max_w = 960, int max_h = 540) {
//     if (img.empty()) {
//         return;
//     }
//     cv::namedWindow(windows_title, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
//     // 如果图像超过最大尺寸,按比例缩放窗口
//     if (img.rows > max_h || img.cols > max_w) {
//         double scale = std::min(static_cast<double>(max_w) / img.cols, static_cast<double>(max_h) / img.rows);
//         cv::resizeWindow(windows_title, static_cast<int>(img.cols * scale), static_cast<int>(img.rows * scale));
//     }
// }

}  // namespace cv_utils
