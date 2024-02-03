# `.\PaddleOCR\deploy\lite\crnn_process.h`

```
// 版权声明，版权归 PaddlePaddle 作者所有
//
// 根据 Apache 许可证 2.0 版本授权
// 除非符合许可证规定，否则不得使用此文件
// 可以在以下网址获取许可证的副本
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则根据许可证分发的软件
// 基于“按原样”分发，没有任何明示或暗示的保证或条件
// 请查看许可证以获取特定语言的权限和限制

#pragma once

// 包含头文件
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// 包含数学库头文件
#include "math.h" //NOLINT
// 包含 OpenCV 核心头文件
#include "opencv2/core.hpp"
// 包含 OpenCV 图像编解码头文件
#include "opencv2/imgcodecs.hpp"
// 包含 OpenCV 图像处理头文件
#include "opencv2/imgproc.hpp"

// 定义函数 CrnnResizeImg，用于调整图像大小
cv::Mat CrnnResizeImg(cv::Mat img, float wh_ratio, int rec_image_height);

// 定义函数 ReadDict，用于读取字典文件
std::vector<std::string> ReadDict(std::string path);

// 定义函数 GetRotateCropImage，用于获取旋转裁剪后的图像
cv::Mat GetRotateCropImage(cv::Mat srcimage, std::vector<std::vector<int>> box);

// 定义模板函数 Argmax，用于计算最大值的索引
template <class ForwardIterator>
inline size_t Argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}
```