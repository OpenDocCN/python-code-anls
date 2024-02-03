# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\ocr_db_post_process.h`

```py
// 源文件的创建日期和作者信息
// Created by fujiayi on 2020/7/2.
// 防止头文件被重复包含
#pragma once
// 包含 OpenCV 库
#include <opencv2/opencv.hpp>
// 包含 vector 头文件
#include <vector>

// 从预测结果中获取边界框的函数声明
std::vector<std::vector<std::vector<int>>>
boxes_from_bitmap(const cv::Mat &pred, const cv::Mat &bitmap);

// 过滤标签检测结果的函数声明
std::vector<std::vector<std::vector<int>>>
filter_tag_det_res(const std::vector<std::vector<std::vector<int>>> &o_boxes,
                   float ratio_h, float ratio_w, const cv::Mat &srcimg);
```