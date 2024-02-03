# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\ocr_crnn_process.h`

```py
//
// Created by fujiayi on 2020/7/3.
//
#pragma once

#include "common.h"
#include <opencv2/opencv.hpp>
#include <vector>

// 定义全局常量，表示识别图像的形状
extern const std::vector<int> REC_IMAGE_SHAPE;

// 旋转并裁剪图像，返回处理后的图像
cv::Mat get_rotate_crop_image(const cv::Mat &srcimage,
                              const std::vector<std::vector<int>> &box);

// 调整图像大小，保持宽高比，返回处理后的图像
cv::Mat crnn_resize_img(const cv::Mat &img, float wh_ratio);

// 模板函数，返回迭代器范围内的最大元素的索引
template <class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}
```