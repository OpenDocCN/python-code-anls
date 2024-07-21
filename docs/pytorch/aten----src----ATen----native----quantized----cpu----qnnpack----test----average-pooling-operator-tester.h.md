# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\average-pooling-operator-tester.h`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <pytorch_qnnpack.h>

// AveragePoolingOperatorTester 类的定义
class AveragePoolingOperatorTester {
 public:
  // 设置填充值，同时在垂直和水平方向上应用相同的填充值
  inline AveragePoolingOperatorTester& padding(uint32_t padding) {
    this->paddingHeight_ = padding;
    this->paddingWidth_ = padding;
    return *this;
  }

  // 设置填充值，分别指定垂直和水平方向上的填充值
  inline AveragePoolingOperatorTester& padding(
      uint32_t paddingHeight,
      uint32_t paddingWidth) {
    this->paddingHeight_ = paddingHeight;
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  // 设置垂直方向上的填充值
  inline AveragePoolingOperatorTester& paddingHeight(uint32_t paddingHeight) {
    this->paddingHeight_ = paddingHeight;
    return *this;
  }

  // 设置水平方向上的填充值
  inline AveragePoolingOperatorTester& paddingWidth(uint32_t paddingWidth) {
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  // 获取当前设置的垂直填充值
  inline uint32_t paddingHeight() const {
    return this->paddingHeight_;
  }

  // 获取当前设置的水平填充值
  inline uint32_t paddingWidth() const {
    return this->paddingWidth_;
  }

  // 设置输入图像的尺寸（高度和宽度）
  inline AveragePoolingOperatorTester& inputSize(
      size_t inputHeight,
      size_t inputWidth) {
    // 断言输入的高度和宽度至少为1
    assert(inputHeight >= 1);
    assert(inputWidth >= 1);
    this->inputHeight_ = inputHeight;
    this->inputWidth_ = inputWidth;
    return *this;
  }

  // 设置输入图像的高度
  inline AveragePoolingOperatorTester& inputHeight(size_t inputHeight) {
    // 断言输入的高度至少为1
    assert(inputHeight >= 1);
    this->inputHeight_ = inputHeight;
    return *this;
  }

  // 获取当前设置的输入图像高度
  inline size_t inputHeight() const {
    return this->inputHeight_;
  }

  // 设置输入图像的宽度
  inline AveragePoolingOperatorTester& inputWidth(size_t inputWidth) {
    // 断言输入的宽度至少为1
    assert(inputWidth >= 1);
    this->inputWidth_ = inputWidth;
    return *this;
  }

  // 获取当前设置的输入图像宽度
  inline size_t inputWidth() const {
    return this->inputWidth_;
  }

  // 设置通道数
  inline AveragePoolingOperatorTester& channels(size_t channels) {
    // 断言通道数不为0
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  // 获取当前设置的通道数
  inline size_t channels() const {
    return this->channels_;
  }

  // 设置批处理大小
  inline AveragePoolingOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  // 获取当前设置的批处理大小
  inline size_t batchSize() const {
    return this->batchSize_;
  }

  // 设置池化窗口大小，同时在垂直和水平方向上应用相同的大小
  inline AveragePoolingOperatorTester& poolingSize(uint32_t poolingSize) {
    // 断言池化窗口大小至少为1
    assert(poolingSize >= 1);
    this->poolingHeight_ = poolingSize;
    this->poolingWidth_ = poolingSize;
    return *this;
  }

  // 设置池化窗口大小，分别指定垂直和水平方向上的大小
  inline AveragePoolingOperatorTester& poolingSize(
      uint32_t poolingHeight,
      uint32_t poolingWidth) {
    // 断言池化窗口高度和宽度至少为1
    assert(poolingHeight >= 1);
    assert(poolingWidth >= 1);
    this->poolingHeight_ = poolingHeight;
    this->poolingWidth_ = poolingWidth;
    return *this;
  }

  // 设置池化窗口的垂直大小
  inline AveragePoolingOperatorTester& poolingHeight(uint32_t poolingHeight) {
    this->poolingHeight_ = poolingHeight;
    return *this;
  }

  // 设置池化窗口的水平大小
  inline AveragePoolingOperatorTester& poolingWidth(uint32_t poolingWidth) {
    this->poolingWidth_ = poolingWidth;
    return *this;
  }

 private:
  // 私有成员变量，用于存储当前设置的各种参数
  uint32_t paddingHeight_{0};
  uint32_t paddingWidth_{0};
  size_t inputHeight_{0};
  size_t inputWidth_{0};
  size_t channels_{0};
  size_t batchSize_{1};
  uint32_t poolingHeight_{1};
  uint32_t poolingWidth_{1};
};
    // 断言池化高度至少为1，确保参数合法性
    assert(poolingHeight >= 1);
    // 设置对象的池化高度
    this->poolingHeight_ = poolingHeight;
    // 返回对象自身以支持链式调用
    return *this;
  }

  // 返回对象当前的池化高度
  inline uint32_t poolingHeight() const {
    return this->poolingHeight_;
  }

  // 设置对象的池化宽度，并确保参数合法性
  inline AveragePoolingOperatorTester& poolingWidth(uint32_t poolingWidth) {
    assert(poolingWidth >= 1);
    this->poolingWidth_ = poolingWidth;
    return *this;
  }

  // 返回对象当前的池化宽度
  inline uint32_t poolingWidth() const {
    return this->poolingWidth_;
  }

  // 设置对象的步长（高度和宽度相同），并确保参数合法性
  inline AveragePoolingOperatorTester& stride(uint32_t stride) {
    assert(stride >= 1);
    this->strideHeight_ = stride;
    this->strideWidth_ = stride;
    return *this;
  }

  // 设置对象的步长（分别指定高度和宽度），并确保参数合法性
  inline AveragePoolingOperatorTester& stride(
      uint32_t strideHeight,
      uint32_t strideWidth) {
    assert(strideHeight >= 1);
    assert(strideWidth >= 1);
    this->strideHeight_ = strideHeight;
    this->strideWidth_ = strideWidth;
    return *this;
  }

  // 设置对象的步长（仅设置高度），并确保参数合法性
  inline AveragePoolingOperatorTester& strideHeight(uint32_t strideHeight) {
    assert(strideHeight >= 1);
    this->strideHeight_ = strideHeight;
    return *this;
  }

  // 返回对象当前的步长高度
  inline uint32_t strideHeight() const {
    return this->strideHeight_;
  }

  // 设置对象的步长（仅设置宽度），并确保参数合法性
  inline AveragePoolingOperatorTester& strideWidth(uint32_t strideWidth) {
    assert(strideWidth >= 1);
    this->strideWidth_ = strideWidth;
    return *this;
  }

  // 返回对象当前的步长宽度
  inline uint32_t strideWidth() const {
    return this->strideWidth_;
  }

  // 返回对象的输出高度
  inline size_t outputHeight() const {
    // 计算输入的填充高度，如果小于等于池化高度则返回1，否则计算输出高度
    const size_t paddedInputHeight = inputHeight() + paddingHeight() * 2;
    if (paddedInputHeight <= poolingHeight()) {
      return 1;
    } else {
      return (paddedInputHeight - poolingHeight()) / strideHeight() + 1;
    }
  }

  // 返回对象的输出宽度
  inline size_t outputWidth() const {
    // 计算输入的填充宽度，如果小于等于池化宽度则返回1，否则计算输出宽度
    const size_t paddedInputWidth = inputWidth() + paddingWidth() * 2;
    if (paddedInputWidth <= poolingWidth()) {
      return 1;
    } else {
      return (paddedInputWidth - poolingWidth()) / strideWidth() + 1;
    }
  }

  // 设置输入像素步长，并确保非零
  inline AveragePoolingOperatorTester& inputPixelStride(
      size_t inputPixelStride) {
    assert(inputPixelStride != 0);
    this->inputPixelStride_ = inputPixelStride;
    return *this;
  }

  // 返回对象当前的输入像素步长
  inline size_t inputPixelStride() const {
    // 如果未设置输入像素步长，则返回通道数作为默认值
    if (this->inputPixelStride_ == 0) {
      return channels();
    } else {
      // 否则返回设置的输入像素步长，并确保不小于通道数
      assert(this->inputPixelStride_ >= channels());
      return this->inputPixelStride_;
    }
  }

  // 设置输出像素步长，并确保非零
  inline AveragePoolingOperatorTester& outputPixelStride(
      size_t outputPixelStride) {
    assert(outputPixelStride != 0);
    this->outputPixelStride_ = outputPixelStride;
    return *this;
  }

  // 返回对象当前的输出像素步长
  inline size_t outputPixelStride() const {
    // 如果未设置输出像素步长，则返回通道数作为默认值
    if (this->outputPixelStride_ == 0) {
      return channels();
    } else {
      // 否则返回设置的输出像素步长，并确保不小于通道数
      assert(this->outputPixelStride_ >= channels());
      return this->outputPixelStride_;
    }
  }

  // 设置下一个输入的尺寸（高度和宽度），并确保参数合法性
  inline AveragePoolingOperatorTester& nextInputSize(
      uint32_t nextInputHeight,
      uint32_t nextInputWidth) {
    assert(nextInputHeight >= 1);
    assert(nextInputWidth >= 1);
    this->nextInputHeight_ = nextInputHeight;
    // 返回对象自身以支持链式调用

    // 断言下一个输入的高度和宽度至少为1，确保参数合法性
    assert(nextInputHeight >= 1);
    assert(nextInputWidth >= 1);
    // 设置对象的下一个输入高度和宽度
    this->nextInputHeight_ = nextInputHeight;
    this->nextInputWidth_ = nextInputWidth;
    // 返回对象自身以支持链式调用
    return *this;
  }
  // 设置下一个输入宽度，并返回当前对象的引用
  inline AveragePoolingOperatorTester& nextInputWidth(uint32_t nextInputWidth) {
    // 断言下一个输入宽度必须大于等于1
    assert(nextInputWidth >= 1);
    this->nextInputWidth_ = nextInputWidth;
    return *this;
  }

  // 获取当前对象的下一个输入高度
  inline uint32_t nextInputHeight() const {
    // 如果下一个输入高度为0，则返回当前输入高度
    if (this->nextInputHeight_ == 0) {
      return inputHeight();
    } else {
      return this->nextInputHeight_;
    }
  }

  // 设置下一个输入高度，并返回当前对象的引用
  inline AveragePoolingOperatorTester& nextInputHeight(uint32_t nextInputHeight) {
    // 断言下一个输入高度必须大于等于1
    assert(nextInputHeight >= 1);
    this->nextInputHeight_ = nextInputHeight;
    return *this;
  }

  // 获取当前对象的下一个输入宽度
  inline uint32_t nextInputWidth() const {
    // 如果下一个输入宽度为0，则返回当前输入宽度
    if (this->nextInputWidth_ == 0) {
      return inputWidth();
    } else {
      return this->nextInputWidth_;
    }
  }

  // 计算下一个输出高度
  inline size_t nextOutputHeight() const {
    // 计算填充后的下一个输入高度
    const size_t paddedNextInputHeight =
        nextInputHeight() + paddingHeight() * 2;
    // 根据池化高度计算下一个输出高度
    if (paddedNextInputHeight <= poolingHeight()) {
      return 1;
    } else {
      return (paddedNextInputHeight - poolingHeight()) / strideHeight() + 1;
    }
  }

  // 计算下一个输出宽度
  inline size_t nextOutputWidth() const {
    // 计算填充后的下一个输入宽度
    const size_t paddedNextInputWidth = nextInputWidth() + paddingWidth() * 2;
    // 根据池化宽度计算下一个输出宽度
    if (paddedNextInputWidth <= poolingWidth()) {
      return 1;
    } else {
      return (paddedNextInputWidth - poolingWidth()) / strideWidth() + 1;
    }
  }

  // 设置下一个批次大小，并返回当前对象的引用
  inline AveragePoolingOperatorTester& nextBatchSize(size_t nextBatchSize) {
    // 断言下一个批次大小必须大于等于1
    assert(nextBatchSize >= 1);
    this->nextBatchSize_ = nextBatchSize;
    return *this;
  }

  // 获取当前对象的下一个批次大小
  inline size_t nextBatchSize() const {
    // 如果下一个批次大小为0，则返回当前批次大小
    if (this->nextBatchSize_ == 0) {
      return batchSize();
    } else {
      return this->nextBatchSize_;
    }
  }

  // 设置输入的比例因子，并返回当前对象的引用
  inline AveragePoolingOperatorTester& inputScale(float inputScale) {
    // 断言输入比例因子必须大于0且为正常数
    assert(inputScale > 0.0f);
    assert(std::isnormal(inputScale));
    this->inputScale_ = inputScale;
    return *this;
  }

  // 获取当前对象的输入比例因子
  inline float inputScale() const {
    return this->inputScale_;
  }

  // 设置输入的零点，并返回当前对象的引用
  inline AveragePoolingOperatorTester& inputZeroPoint(uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  // 获取当前对象的输入零点
  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  // 设置输出的比例因子，并返回当前对象的引用
  inline AveragePoolingOperatorTester& outputScale(float outputScale) {
    // 断言输出比例因子必须大于0且为正常数
    assert(outputScale > 0.0f);
    assert(std::isnormal(outputScale));
    this->outputScale_ = outputScale;
    return *this;
  }

  // 获取当前对象的输出比例因子
  inline float outputScale() const {
    return this->outputScale_;
  }

  // 设置输出的零点，并返回当前对象的引用
  inline AveragePoolingOperatorTester& outputZeroPoint(uint8_t outputZeroPoint) {
    this->outputZeroPoint_ = outputZeroPoint;
    return *this;
  }

  // 获取当前对象的输出零点
  inline uint8_t outputZeroPoint() const {
    return this->outputZeroPoint_;
  }

  // 设置量化的最小值，并返回当前对象的引用
  inline AveragePoolingOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 获取当前对象的量化的最小值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置量化的最大值，并返回当前对象的引用
  inline AveragePoolingOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline AveragePoolingOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void testQ8() const {
    // 创建随机设备和随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    // 创建生成无符号8位整数的随机数绑定器
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 初始化输入向量，大小为 batch size * 输入高度 * 输入宽度 * 输入像素步长 + 通道数 - 1
    std::vector<uint8_t> input(
        (batchSize() * inputHeight() * inputWidth() - 1) * inputPixelStride() +
        channels());
    // 初始化输出向量，大小为 batch size * 输出高度 * 输出宽度 * 输出像素步长 + 通道数 - 1
    std::vector<uint8_t> output(
        (batchSize() * outputHeight() * outputWidth() - 1) *
            outputPixelStride() +
        channels());
    // 初始化参考输出向量，大小为 batch size * 输出高度 * 输出宽度 * 通道数
    std::vector<float> outputRef(
        batchSize() * outputHeight() * outputWidth() * channels());
    }
  }

  void testSetupQ8() const {
    // 创建随机设备和随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    // 创建生成无符号8位整数的随机数绑定器
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 初始化输入向量，大小为 batch size * 输入高度 * 输入宽度 * 输入像素步长 + 通道数 与 next batch size * next 输入高度 * next 输入宽度 * 输入像素步长 + 通道数 中的较大者
    std::vector<uint8_t> input(std::max(
        (batchSize() * inputHeight() * inputWidth() - 1) * inputPixelStride() +
            channels(),
        (nextBatchSize() * nextInputHeight() * nextInputWidth() - 1) *
                inputPixelStride() +
            channels()));
    // 初始化输出向量，大小为 batch size * 输出高度 * 输出宽度 * 输出像素步长 + 通道数 与 next batch size * next 输出高度 * next 输出宽度 * 输出像素步长 + 通道数 中的较大者
    std::vector<uint8_t> output(std::max(
        (batchSize() * outputHeight() * outputWidth() - 1) *
                outputPixelStride() +
            channels(),
        (nextBatchSize() * nextOutputHeight() * nextOutputWidth() - 1) *
                outputPixelStride() +
            channels()));
    // 初始化参考输出向量，大小为 batch size * 输出高度 * 输出宽度 * 通道数
    std::vector<float> outputRef(
        batchSize() * outputHeight() * outputWidth() * channels());
    // 初始化下一个批次的参考输出向量，大小为 next batch size * next 输出高度 * next 输出宽度 * 通道数
    std::vector<float> nextOutputRef(
        nextBatchSize() * nextOutputHeight() * nextOutputWidth() * channels());
    }
  }

 private:
  uint32_t paddingHeight_{0};
  uint32_t paddingWidth_{0};
  size_t inputHeight_{1};
  size_t inputWidth_{1};
  size_t channels_{1};
  size_t batchSize_{1};
  size_t inputPixelStride_{0};
  size_t outputPixelStride_{0};
  uint32_t poolingHeight_{1};
  uint32_t poolingWidth_{1};
  uint32_t strideHeight_{1};
  uint32_t strideWidth_{1};
  size_t nextInputHeight_{0};
  size_t nextInputWidth_{0};
  size_t nextBatchSize_{0};
  float inputScale_{1.0f};
  float outputScale_{1.0f};
  uint8_t inputZeroPoint_{121};
  uint8_t outputZeroPoint_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};


注释：

# 这是一个语法错误示例，因为单独的 }; 是无效的代码块，缺少了开头的语法结构或者声明。
# 在实际的编程中，这样的错误通常表示需要查找并修复缺失的代码块或语句。
```