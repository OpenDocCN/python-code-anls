# `.\PaddleOCR\deploy\lite\db_post_process.h`

```py
// 版权声明，版权归 PaddlePaddle 作者所有
//
// 根据 Apache 许可证 2.0 版本授权
// 除非符合许可证的规定，否则不得使用此文件
// 可以在以下网址获取许可证的副本
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则根据许可证分发的软件
// 基于“按原样”分发，没有任何明示或暗示的保证或条件
// 请查看许可证以获取特定语言的权限和限制

#pragma once

#include <math.h>

#include <iostream>
#include <map>
#include <vector>

#include "clipper.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// 定义一个 clamp 函数，用于将值限制在指定范围内
template <class T> T clamp(T x, T min, T max) {
  if (x > max)
    return max;
  if (x < min)
    return min;
  return x;
}

// 将 OpenCV 的 Mat 对象转换为二维向量
std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

// 计算轮廓区域的距离
void GetContourArea(std::vector<std::vector<float>> box, float unclip_ratio,
                    float &distance);

// 对旋转矩形进行解封
cv::RotatedRect Unclip(std::vector<std::vector<float>> box, float unclip_ratio);

// 将 OpenCV 的 Mat 对象转换为二维向量
std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

// 按照浮点数进行排序
bool XsortFp32(std::vector<float> a, std::vector<float> b);

// 按照整数进行排序
bool XsortInt(std::vector<int> a, std::vector<int> b);

// 按顺时针顺序对点进行排序
std::vector<std::vector<int>>
OrderPointsClockwise(std::vector<std::vector<int>> pts);

// 获取最小框
std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box, float &ssid);

// 快速计算框的得分
float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);

// 从位图中获取框
std::vector<std::vector<std::vector<int>>>
BoxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap,
                std::map<std::string, double> Config);

// 过滤标签检测结果
std::vector<std::vector<std::vector<int>>>
FilterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes, float ratio_h,
                float ratio_w, cv::Mat srcimg);
```