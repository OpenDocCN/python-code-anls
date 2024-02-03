# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\ocr_cls_process.h`

```py
// 版权声明，声明代码版权归 PaddlePaddle 作者所有
//
// 根据 Apache 许可证版本 2.0 进行许可;
// 除非符合许可证的规定，否则不得使用此文件
// 您可以在以下网址获取许可证的副本
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则软件
// 根据许可证“按原样”分发，
// 没有任何明示或暗示的保证或条件
// 请查看许可证以获取特定语言的权限和限制

// 防止头文件被重复包含
#pragma once

// 包含自定义的 common.h 头文件
#include "common.h"
// 包含 OpenCV 库
#include <opencv2/opencv.hpp>
// 包含 vector 容器
#include <vector>

// 声明一个全局常量，表示分类图像的形状
extern const std::vector<int> CLS_IMAGE_SHAPE;

// 定义一个函数，用于调整图像大小
cv::Mat cls_resize_img(const cv::Mat &img);
```