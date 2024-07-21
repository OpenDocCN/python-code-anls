# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\fully-connected-operator-tester.h`

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
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>
#include <memory>

#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <qnnpack/AlignedAllocator.h>

// 定义 FullyConnectedOperatorTester 类，用于测试全连接操作符的功能
class FullyConnectedOperatorTester {
 public:
  // 设置输入通道数，并进行断言确保通道数不小于 1
  inline FullyConnectedOperatorTester& inputChannels(size_t inputChannels) {
    assert(inputChannels >= 1);
    this->inputChannels_ = inputChannels;
    return *this;
  }

  // 返回当前输入通道数
  inline size_t inputChannels() const {
    return this->inputChannels_;
  }

  // 设置输出通道数，并进行断言确保通道数不小于 1
  inline FullyConnectedOperatorTester& outputChannels(size_t outputChannels) {
    assert(outputChannels >= 1);
    this->outputChannels_ = outputChannels;
    return *this;
  }

  // 返回当前输出通道数
  inline size_t outputChannels() const {
    return this->outputChannels_;
  }

  // 设置批处理大小
  inline FullyConnectedOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  // 返回当前批处理大小
  inline size_t batchSize() const {
    return this->batchSize_;
  }

  // 设置输入步长，并进行断言确保步长不小于 1
  inline FullyConnectedOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride >= 1);
    this->inputStride_ = inputStride;
    return *this;
  }

  // 返回当前输入步长，如果未设置则默认为输入通道数
  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return inputChannels();
    } else {
      assert(this->inputStride_ >= inputChannels());
      return this->inputStride_;
    }
  }

  // 设置输出步长，并进行断言确保步长不小于 1
  inline FullyConnectedOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride >= 1);
    this->outputStride_ = outputStride;
    return *this;
  }

  // 返回当前输出步长，如果未设置则默认为输出通道数
  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return outputChannels();
    } else {
      assert(this->outputStride_ >= outputChannels());
      return this->outputStride_;
    }
  }

  // 设置是否为每通道量化的标志
  inline FullyConnectedOperatorTester& per_channel(bool per_channel) {
    this->per_channel_ = per_channel;
    return *this;
  }

  // 返回是否为每通道量化的标志
  inline bool per_channel() const {
    return this->per_channel_;
  }

  // 设置量化的最小值
  inline FullyConnectedOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 返回量化的最小值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置量化的最大值
  inline FullyConnectedOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回量化的最大值
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置测试迭代次数
  inline FullyConnectedOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回测试迭代次数
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 定义枚举类型 Mode，表示量化的模式
  enum class Mode {
    Static,   // 静态量化模式
    Dynamic,  // 动态量化模式
    Runtime,  // 运行时量化模式
  };

  // 测试 Q8 算法的函数，根据给定的量化模式进行测试
  void testQ8(const Mode mode) const {
    // 创建随机设备对象
    std::random_device randomDevice;
    // 使用随机设备生成随机数引擎
    auto rng = std::mt19937(randomDevice());
    // 创建一个绑定到 rng 的 s32rng 函数对象，用于生成范围在 -10000 到 10000 之间的随机整数
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    // 创建一个绑定到 rng 的 u8rng 函数对象，用于生成范围在 0 到 255 之间的随机无符号 8 位整数
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    // 创建一个绑定到 rng 的 f32rng 函数对象，用于生成范围在 1 到 5 之间的随机单精度浮点数
    auto f32rng =
        std::bind(std::uniform_real_distribution<float>(1, 5), rng);

    // 初始化 input 向量，其大小为 (batchSize() - 1) * inputStride() + inputChannels() + 8
    std::vector<uint8_t> input(
        (batchSize() - 1) * inputStride() + inputChannels() + 8);
    // 初始化 kernel 向量，其大小为 outputChannels() * inputChannels()
    std::vector<uint8_t> kernel(outputChannels() * inputChannels());
    // 初始化 bias 向量，其大小为 outputChannels()
    std::vector<int32_t> bias(outputChannels());
    // 初始化 output 向量，其大小为 (batchSize() - 1) * outputStride() + outputChannels()
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + outputChannels());
    // 初始化 output_dynamic 向量，其大小与 output 向量相同，用于存储浮点数类型的输出
    std::vector<float> output_dynamic(output.size());
    // 初始化 accumulators 向量，其大小为 batchSize() * outputChannels()
    std::vector<int32_t> accumulators(batchSize() * outputChannels());

    // 初始化 inputPtr 指向 input 向量中偏移为 8 的位置
    const uint8_t* const inputPtr = input.data() + 8;
    // 初始化 inputZeroPoint 为 127，用作输入的零点偏移量

    // 将输出通道数量调整为 8 的倍数，为 SSE/ARM 内核选择最小公倍数
    size_t num_zero_points_padded = outputChannels() + 8;
    // 初始化 kernelZeroPoints 向量，大小为 num_zero_points_padded，所有元素为 127
    std::vector<uint8_t> kernelZeroPoints(num_zero_points_padded, 127);
};


注释：


# 这行代码是一个空的代码块闭合标记
```