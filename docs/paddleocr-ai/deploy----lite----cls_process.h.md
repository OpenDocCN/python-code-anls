# `.\PaddleOCR\deploy\lite\cls_process.h`

```py
// 版权声明，声明代码版权归 PaddlePaddle 作者所有
//
// 根据 Apache 许可证 2.0 版本授权使用该文件；
// 除非符合许可证的规定，否则不得使用该文件
// 可以在以下网址获取许可证的副本
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则根据许可证分发的软件
// 基于“按原样”分发，没有任何明示或暗示的保证或条件
// 请查看许可证以获取有关权限和限制的详细信息
//
// 防止重复包含头文件
#pragma once

// 包含 C 标准库头文件
#include <cstring>
// 包含文件流操作头文件
#include <fstream>
// 包含输入输出流操作头文件
#include <iostream>
// 包含智能指针头文件
#include <memory>
// 包含字符串操作头文件
#include <string>
// 包含向量操作头文件
#include <vector>

// 禁止 Lint 工具对该行进行检查
#include "math.h" //NOLINT
// 包含 OpenCV 核心功能头文件
#include "opencv2/core.hpp"
// 包含 OpenCV 图像编解码头文件
#include "opencv2/imgcodecs.hpp"
// 包含 OpenCV 图像处理头文件
#include "opencv2/imgproc.hpp"

// 函数声明，用于调整分类任务的图像大小
cv::Mat ClsResizeImg(cv::Mat img);
```