# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\sigmoid-operator-tester.h`

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

class SigmoidOperatorTester {
 public:
  // 设置测试通道数，并进行断言检查
  inline SigmoidOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  // 返回当前设置的通道数
  inline size_t channels() const {
    return this->channels_;
  }

  // 设置输入步长，并进行断言检查
  inline SigmoidOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  // 返回当前设置的输入步长，若未设置则返回通道数
  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->inputStride_ >= this->channels_);
      return this->inputStride_;
    }
  }

  // 设置输出步长，并进行断言检查
  inline SigmoidOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  // 返回当前设置的输出步长，若未设置则返回通道数
  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->outputStride_ >= this->channels_);
      return this->outputStride_;
    }
  }

  // 设置批次大小
  inline SigmoidOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  // 返回当前设置的批次大小
  inline size_t batchSize() const {
    return this->batchSize_;
  }

  // 设置输入缩放因子，并进行断言检查
  inline SigmoidOperatorTester& inputScale(float inputScale) {
    assert(inputScale > 0.0f);
    assert(std::isnormal(inputScale));
    this->inputScale_ = inputScale;
    return *this;
  }

  // 返回当前设置的输入缩放因子
  inline float inputScale() const {
    return this->inputScale_;
  }

  // 设置输入零点
  inline SigmoidOperatorTester& inputZeroPoint(uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  // 返回当前设置的输入零点
  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  // 返回固定的输出缩放因子
  inline float outputScale() const {
    return 1.0f / 256.0f;
  }

  // 返回固定的输出零点
  inline uint8_t outputZeroPoint() const {
    return 0;
  }

  // 设置量化的最小值
  inline SigmoidOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 返回当前设置的量化的最小值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置量化的最大值
  inline SigmoidOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回当前设置的量化的最大值
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置测试迭代次数
  inline SigmoidOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前设置的测试迭代次数
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 执行 Q8 版本的 Sigmoid 操作的测试
  void testQ8() const {
    // 初始化随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 初始化输入向量大小为 (batchSize() - 1) * inputStride() + channels()，并填充随机数
    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + channels());

// 根据 batchSize()、outputStride() 和 channels() 计算输出向量的大小，并初始化为零向量

    std::vector<float> outputRef(batchSize() * channels());

// 根据 batchSize() 和 channels() 计算参考输出向量的大小，并初始化为零向量

    for (size_t iteration = 0; iteration < iterations(); iteration++) {

// 迭代执行多次，次数由 iterations() 函数返回

      std::generate(input.begin(), input.end(), std::ref(u8rng));

// 生成随机输入数据，填充到 input 向量中，使用 u8rng 作为随机数生成器

      std::fill(output.begin(), output.end(), 0xA5);

// 将输出向量 output 填充为十六进制值 0xA5

      /* Compute reference results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = inputScale() *
              (int32_t(input[i * inputStride() + c]) -
               int32_t(inputZeroPoint()));
          const float sigmoidX = 1.0f / (1.0f + exp(-x));
          const float scaledSigmoidX = sigmoidX / outputScale();
          float y = scaledSigmoidX;
          y = std::min<float>(y, int32_t(qmax()) - int32_t(outputZeroPoint()));
          y = std::max<float>(y, int32_t(qmin()) - int32_t(outputZeroPoint()));
          outputRef[i * channels() + c] = y + int32_t(outputZeroPoint());
        }
      }

// 计算参考输出结果，对每个输入批次和通道进行操作，应用了量化和反量化的过程

      /* Create, setup, run, and destroy Sigmoid operator */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());

// 初始化 QNNPACK 库，确保成功

      pytorch_qnnp_operator_t sigmoidOp = nullptr;

// 声明一个 QNNPACK 操作符指针，初始化为 nullptr

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_sigmoid_nc_q8(
              channels(),
              inputZeroPoint(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              qmin(),
              qmax(),
              0,
              &sigmoidOp));

// 创建一个 QNNPACK Sigmoid 运算符，配置其输入输出量化参数

      ASSERT_NE(nullptr, sigmoidOp);

// 确保成功创建了 Sigmoid 运算符，验证指针非空

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_sigmoid_nc_q8(
              sigmoidOp,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

// 配置 Sigmoid 运算符的输入输出数据，设置完成

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(sigmoidOp, nullptr /* thread pool */));

// 运行 Sigmoid 运算符，执行量化神经网络推理

      ASSERT_EQ(
          pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(sigmoidOp));
      sigmoidOp = nullptr;

// 删除 Sigmoid 运算符，释放资源，确保指针置为空

      /* Verify results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              float(int32_t(output[i * outputStride() + c])),
              outputRef[i * channels() + c],
              0.6f);
        }
      }

// 验证推理结果的正确性，使用 ASSERT_NEAR 进行浮点数的近似比较
    }

// 结束迭代循环，函数执行完成

 private:
  size_t batchSize_{1};
  size_t channels_{1};
  size_t inputStride_{0};
  size_t outputStride_{0};
  float inputScale_{0.75f};
  uint8_t inputZeroPoint_{121};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};

// 私有成员变量的定义，包括批次大小、通道数、输入输出步长、输入缩放、输入零点、量化参数范围和迭代次数
};


注释：


# 这行代码是一个分号，用于结束代码语句或表达式。在这里，它是一个独立的代码行，可能是用于结束某个代码块或语句的标记。
```