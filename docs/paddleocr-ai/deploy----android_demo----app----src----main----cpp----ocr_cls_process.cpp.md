# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\ocr_cls_process.cpp`

```
// 版权声明，版权归 PaddlePaddle 作者所有
//
// 根据 Apache 许可证 2.0 版本使用此文件；
// 除非符合许可证的规定，否则不得使用此文件。
// 您可以在以下网址获取许可证的副本：
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则根据许可证分发的软件
// 均基于“原样”分发，不附带任何明示或暗示的担保或条件。
// 请查看许可证以获取特定语言的权限和限制。

#include "ocr_cls_process.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iostream>
#include <vector>

// 定义分类器输入图像的形状
const std::vector<int> CLS_IMAGE_SHAPE = {3, 48, 192};

// 缩放图像到分类器输入图像的大小
cv::Mat cls_resize_img(const cv::Mat &img) {
  // 获取分类器输入图像的通道数、宽度和高度
  int imgC = CLS_IMAGE_SHAPE[0];
  int imgW = CLS_IMAGE_SHAPE[2];
  int imgH = CLS_IMAGE_SHAPE[1];

  // 计算图像的宽高比
  float ratio = float(img.cols) / float(img.rows);
  int resize_w = 0;
  // 根据宽高比调整图像的宽度
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  // 调整图像大小
  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_CUBIC);

  // 如果调整后的宽度小于分类器输入图像的宽度，则进行边界填充
  if (resize_w < imgW) {
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, int(imgW - resize_w),
                       cv::BORDER_CONSTANT, {0, 0, 0});
  }
  // 返回调整后的图像
  return resize_img;
}
```