# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\ocr_crnn_process.cpp`

```py
// 包含版权声明和许可证信息
// 根据 Apache 许可证 2.0 版本使用此文件
// 如果不符合许可证要求，不得使用此文件
// 许可证详情请查看 http://www.apache.org/licenses/LICENSE-2.0

#include "ocr_crnn_process.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iostream>
#include <vector>

// 定义字符类型为中文
const std::string CHARACTER_TYPE = "ch";
// 定义最大字典长度
const int MAX_DICT_LENGTH = 6624;
// 定义识别图像的形状
const std::vector<int> REC_IMAGE_SHAPE = {3, 32, 320};

// 缩放和归一化图像
static cv::Mat crnn_resize_norm_img(cv::Mat img, float wh_ratio) {
  // 获取图像通道数、宽度和高度
  int imgC = REC_IMAGE_SHAPE[0];
  int imgW = REC_IMAGE_SHAPE[2];
  int imgH = REC_IMAGE_SHAPE[1];

  // 如果字符类型为中文，则重新计算图像宽度
  if (CHARACTER_TYPE == "ch")
    imgW = int(32 * wh_ratio);

  // 计算图像宽高比
  float ratio = float(img.cols) / float(img.rows);
  int resize_w = 0;
  // 根据比例调整图像宽度
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));
  // 调整图像大小
  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_CUBIC);

  // 将图像转换为浮点型并归一化
  resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

  // 对图像进行归一化处理
  for (int h = 0; h < resize_img.rows; h++) {
    for (int w = 0; w < resize_img.cols; w++) {
      resize_img.at<cv::Vec3f>(h, w)[0] =
          (resize_img.at<cv::Vec3f>(h, w)[0] - 0.5) * 2;
      resize_img.at<cv::Vec3f>(h, w)[1] =
          (resize_img.at<cv::Vec3f>(h, w)[1] - 0.5) * 2;
      resize_img.at<cv::Vec3f>(h, w)[2] =
          (resize_img.at<cv::Vec3f>(h, w)[2] - 0.5) * 2;
    }
  }

  // 创建一个新的矩阵dist，通过在resize_img周围添加边框来调整大小，使其宽度与imgW相等
  cv::Mat dist;
  cv::copyMakeBorder(resize_img, dist, 0, 0, 0, int(imgW - resize_w),
                     cv::BORDER_CONSTANT, {0, 0, 0});

  // 返回调整大小后的矩阵dist
  return dist;
// 调整图像大小，保持宽高比为指定值
cv::Mat crnn_resize_img(const cv::Mat &img, float wh_ratio) {
  // 获取图像通道数、宽度和高度
  int imgC = REC_IMAGE_SHAPE[0];
  int imgW = REC_IMAGE_SHAPE[2];
  int imgH = REC_IMAGE_SHAPE[1];

  // 如果字符类型为中文，则重新设置图像宽度
  if (CHARACTER_TYPE == "ch") {
    imgW = int(32 * wh_ratio);
  }

  // 计算图像宽高比
  float ratio = float(img.cols) / float(img.rows);
  int resize_w = 0;
  // 根据比例调整图像宽度
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));
  // 调整图像大小
  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, imgH));
  return resize_img;
}

// 获取旋转裁剪后的图像
cv::Mat get_rotate_crop_image(const cv::Mat &srcimage,
                              const std::vector<std::vector<int>> &box) {

  // 复制边框坐标
  std::vector<std::vector<int>> points = box;

  // 提取边框 x、y 坐标
  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  // 计算裁剪区域的左、右、上、下边界
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));

  // 裁剪图像
  cv::Mat img_crop;
  srcimage(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

  // 调整边框坐标相对于裁剪区域的偏移
  for (int i = 0; i < points.size(); i++) {
    points[i][0] -= left;
  points[i][1] -= top;
  // 将每个点的 y 坐标减去顶部的偏移量

  int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                pow(points[0][1] - points[1][1], 2)));
  // 计算裁剪后图像的宽度

  int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                 pow(points[0][1] - points[3][1], 2)));
  // 计算裁剪后图像的高度

  cv::Point2f pts_std[4];
  // 创建标准四边形的四个点
  pts_std[0] = cv::Point2f(0., 0.);
  pts_std[1] = cv::Point2f(img_crop_width, 0.);
  pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
  pts_std[3] = cv::Point2f(0.f, img_crop_height);

  cv::Point2f pointsf[4];
  // 创建原始四边形的四个点
  pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
  pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
  pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
  pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

  cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);
  // 获取透视变换矩阵

  cv::Mat dst_img;
  // 创建目标图像
  cv::warpPerspective(img_crop, dst_img, M,
                      cv::Size(img_crop_width, img_crop_height),
                      cv::BORDER_REPLICATE);
  // 进行透视变换

  if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
    // 如果目标图像的高度大于等于宽度的1.5倍
    cv::transpose(dst_img, dst_img);
    // 转置目标图像
    cv::flip(dst_img, dst_img, 0);
    // 沿 y 轴翻转目标图像
    return dst_img;
    // 返回处理后的图像
  } else {
    return dst_img;
    // 返回原始处理后的图像
  }
# 闭合之前的代码块
```