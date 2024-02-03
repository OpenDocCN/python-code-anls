# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\preprocess.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include "common.h"
// 包含 common.h 头文件

#include <jni.h>
// 包含 JNI 头文件

#include <opencv2/opencv.hpp>
// 包含 OpenCV 头文件

cv::Mat bitmap_to_cv_mat(JNIEnv *env, jobject bitmap);
// 声明函数 bitmap_to_cv_mat，将 Android Bitmap 转换为 OpenCV Mat

cv::Mat resize_img(const cv::Mat &img, int height, int width);
// 声明函数 resize_img，调整图像大小

void neon_mean_scale(const float *din, float *dout, int size,
                     const std::vector<float> &mean,
                     const std::vector<float> &scale);
// 声明函数 neon_mean_scale，使用 NEON 指令进行均值和缩放处理
```