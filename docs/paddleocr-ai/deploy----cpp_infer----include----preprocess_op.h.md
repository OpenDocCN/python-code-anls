# `.\PaddleOCR\deploy\cpp_infer\include\preprocess_op.h`

```py
// 版权声明，告知代码版权归属及使用许可
// 根据 Apache 许可证 2.0 版本使用此文件
// 只有在遵守许可证的情况下才能使用此文件
// 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
// 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"按原样"分发的
// 没有任何明示或暗示的担保或条件，包括但不限于特定用途的适用性
// 请查看许可证以获取有关权限和限制的详细信息

#pragma once

// 包含所需的头文件
#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// 命名空间 PaddleOCR
namespace PaddleOCR {

// 归一化类，用于对图像进行归一化处理
class Normalize {
public:
  // 运行归一化操作，包括均值、缩放等参数
  virtual void Run(cv::Mat *im, const std::vector<float> &mean,
                   const std::vector<float> &scale, const bool is_scale = true);
};

// RGB -> CHW 转换类
class Permute {
public:
  // 运行转换操作，将图像从 RGB 转换为 CHW 格式
  virtual void Run(const cv::Mat *im, float *data);
};

// 批量 RGB -> CHW 转换类
class PermuteBatch {
public:
  // 运行批量转换操作，将多个图像从 RGB 转换为 CHW 格式
  virtual void Run(const std::vector<cv::Mat> imgs, float *data);
};

// 图像缩放类，用于调整图像大小
class ResizeImgType0 {
public:
  // 运行图像缩放操作，根据限制类型和边长调整图像大小
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img,
                   std::string limit_type, int limit_side_len, float &ratio_h,
                   float &ratio_w, bool use_tensorrt);
};

// CRNN 图像缩放类
class CrnnResizeImg {
public:
  // 运行 CRNN 图像缩放操作，根据宽高比调整图像大小
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                   bool use_tensorrt = false,
                   const std::vector<int> &rec_image_shape = {3, 32, 320});
};

// 分类器图像缩放类
class ClsResizeImg {
public:
  // 运行分类器图像缩放操作，根据指定形状调整图像大小
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img,
                   bool use_tensorrt = false,
                   const std::vector<int> &rec_image_shape = {3, 48, 192});
};

// 表格图像缩放类
class TableResizeImg {
public:
  // 运行表格图像缩放操作，根据最大长度调整图像大小
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img,
                   const int max_len = 488);
};

// 表格图像填充类
class TablePadImg {
# 定义一个公共类，包含一个虚拟方法Run，用于处理图像的缩放
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img,
                   const int max_len = 488);
};

# 定义一个Resize类，包含一个虚拟方法Run，用于处理图像的缩放并指定高度和宽度
class Resize {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img, const int h,
                   const int w);
};

# 命名空间PaddleOCR，用于封装PaddleOCR相关的类和方法
} // namespace PaddleOCR
```