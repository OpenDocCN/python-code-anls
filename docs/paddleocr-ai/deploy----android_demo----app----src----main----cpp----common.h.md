# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\common.h`

```
// 定义头文件的预处理指令，防止头文件被重复包含
// Created by fu on 4/25/18.

// 引入数值计算库
#pragma once
#import <numeric>
// 引入向量容器库
#import <vector>

// 如果是在 Android 平台下
#ifdef __ANDROID__

// 引入 Android 日志库
#include <android/log.h>

// 定义日志标签
#define LOG_TAG "OCR_NDK"

// 定义输出信息的宏，使用 Android 日志库输出信息
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
// 如果不是在 Android 平台下
#else
// 引入标准输入输出库
#include <stdio.h>
// 定义输出信息的宏，使用标准输出流输出信息
#define LOGI(format, ...)                                                      \
  fprintf(stdout, "[" LOG_TAG "]" format "\n", ##__VA_ARGS__)
#define LOGW(format, ...)                                                      \
  fprintf(stdout, "[" LOG_TAG "]" format "\n", ##__VA_ARGS__)
#define LOGE(format, ...)                                                      \
  fprintf(stderr, "[" LOG_TAG "]Error: " format "\n", ##__VA_ARGS__)
#endif

// 定义返回码枚举，表示操作成功
enum RETURN_CODE { RETURN_OK = 0 };

// 定义网络类型枚举，表示 OCR 网络和内部 OCR 网络
enum NET_TYPE { NET_OCR = 900100, NET_OCR_INTERNAL = 991008 };

// 定义模板函数，计算向量中元素的乘积
template <typename T> inline T product(const std::vector<T> &vec) {
  // 如果向量为空，返回 0
  if (vec.empty()) {
    return 0;
  }
  // 使用 std::accumulate 函数计算向量中元素的乘积
  return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<T>());
}
```