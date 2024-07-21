# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\hardswish-operator-tester.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>   // 引入标准库中的算法功能
#include <cassert>     // 引入标准库中的断言功能
#include <cmath>       // 引入数学函数库
#include <cstddef>     // 引入标准库中的 size_t 定义
#include <cstdlib>     // 引入标准库中的通用函数
#include <functional>  // 引入函数对象的相关定义
#include <random>      // 引入随机数生成器的相关功能
#include <vector>      // 引入动态数组的支持

#include <pytorch_qnnpack.h>  // 引入 PyTorch 的 QNNPACK 库

class HardswishOperatorTester {
 public:
  inline HardswishOperatorTester& channels(size_t channels) {  // 定义设置通道数的方法
    assert(channels != 0);  // 断言通道数不为零
    this->channels_ = channels;  // 设置对象的通道数属性
    return *this;  // 返回当前对象的引用
  }

  inline size_t channels() const {  // 定义获取通道数的方法
    return this->channels_;  // 返回对象的通道数属性
  }

  inline HardswishOperatorTester& inputStride(size_t inputStride) {  // 定义设置输入步长的方法
    assert(inputStride != 0);  // 断言输入步长不为零
    this->inputStride_ = inputStride;  // 设置对象的输入步长属性
    return *this;  // 返回当前对象的引用
  }

  inline size_t inputStride() const {  // 定义获取输入步长的方法
    if (this->inputStride_ == 0) {  // 如果输入步长为零
      return this->channels_;  // 返回对象的通道数作为默认步长
    } else {  // 否则
      assert(this->inputStride_ >= this->channels_);  // 断言输入步长不小于通道数
      return this->inputStride_;  // 返回对象的输入步长属性
    }
  }

  inline HardswishOperatorTester& outputStride(size_t outputStride) {  // 定义设置输出步长的方法
    assert(outputStride != 0);  // 断言输出步长不为零
    this->outputStride_ = outputStride;  // 设置对象的输出步长属性
    return *this;  // 返回当前对象的引用
  }

  inline size_t outputStride() const {  // 定义获取输出步长的方法
    if (this->outputStride_ == 0) {  // 如果输出步长为零
      return this->channels_;  // 返回对象的通道数作为默认步长
    } else {  // 否则
      assert(this->outputStride_ >= this->channels_);  // 断言输出步长不小于通道数
      return this->outputStride_;  // 返回对象的输出步长属性
    }
  }

  inline HardswishOperatorTester& batchSize(size_t batchSize) {  // 定义设置批次大小的方法
    this->batchSize_ = batchSize;  // 设置对象的批次大小属性
    return *this;  // 返回当前对象的引用
  }

  inline size_t batchSize() const {  // 定义获取批次大小的方法
    return this->batchSize_;  // 返回对象的批次大小属性
  }

  inline HardswishOperatorTester& inputScale(float inputScale) {  // 定义设置输入比例的方法
    assert(inputScale > 0.0f);  // 断言输入比例大于零
    assert(std::isnormal(inputScale));  // 断言输入比例为正常数值
    this->inputScale_ = inputScale;  // 设置对象的输入比例属性
    return *this;  // 返回当前对象的引用
  }

  inline float inputScale() const {  // 定义获取输入比例的方法
    return this->inputScale_;  // 返回对象的输入比例属性
  }

  inline HardswishOperatorTester& inputZeroPoint(uint8_t inputZeroPoint) {  // 定义设置输入零点的方法
    this->inputZeroPoint_ = inputZeroPoint;  // 设置对象的输入零点属性
    return *this;  // 返回当前对象的引用
  }

  inline uint8_t inputZeroPoint() const {  // 定义获取输入零点的方法
    return this->inputZeroPoint_;  // 返回对象的输入零点属性
  }

  inline HardswishOperatorTester& outputScale(float outputScale) {  // 定义设置输出比例的方法
    assert(outputScale > 0.0f);  // 断言输出比例大于零
    assert(std::isnormal(outputScale));  // 断言输出比例为正常数值
    this->outputScale_ = outputScale;  // 设置对象的输出比例属性
    return *this;  // 返回当前对象的引用
  }

  inline float outputScale() const {  // 定义获取输出比例的方法
    return this->outputScale_;  // 返回对象的输出比例属性
  }

  inline HardswishOperatorTester& outputZeroPoint(uint8_t outputZeroPoint) {  // 定义设置输出零点的方法
    this->outputZeroPoint_ = outputZeroPoint;  // 设置对象的输出零点属性
    return *this;  // 返回当前对象的引用
  }

  inline uint8_t outputZeroPoint() const {  // 定义获取输出零点的方法
    return this->outputZeroPoint_;  // 返回对象的输出零点属性
  }

  inline HardswishOperatorTester& qmin(uint8_t qmin) {  // 定义设置最小量化值的方法
    this->qmin_ = qmin;  // 设置对象的最小量化值属性
    return *this;  // 返回当前对象的引用
  }

  inline uint8_t qmin() const {  // 定义获取最小量化值的方法
    return this->qmin_;  // 返回对象的最小量化值属性
  }

  inline HardswishOperatorTester& qmax(uint8_t qmax) {  // 定义设置最大量化值的方法
    this->qmax_ = qmax;  // 设置对象的最大量化值属性
    return *this;  // 返回当前对象的引用
  }

  inline uint8_t qmax() const {  // 定义获取最大量化值的方法
    return this->qmax_;  // 返回对象的最大量化值属性
  }

  inline HardswishOperatorTester& iterations(size_t iterations) {
    // 设置迭代次数属性
    this->iterations_ = iterations;
    return *this;  // 返回当前对象的引用
  }

  inline size_t iterations() const {
    return this->iterations_;  // 获取迭代次数属性
  }

 private:
  size_t channels_ = 1;             // 默认通道数为1
  size_t inputStride_ = 0;          // 默认输入步长为0
  size_t outputStride_ = 0;         // 默认输出步长为0
  size_t batchSize_ = 1;            // 默认批次大小为1
  float inputScale_ = 1.0f;         // 默认输入比例为1.0
  uint8_t inputZeroPoint_ = 0;      // 默认输入零点为0
  float outputScale_ = 1.0f;        // 默认输出比例为1.0
  uint8_t outputZeroPoint_ = 0;     // 默认输出零点为0
  uint8_t qmin_ = 0;                // 默认最小量化值为0
  uint8_t qmax_ = 255;              // 默认最大量化值为255
  size_t iterations_ = 1;           // 默认迭代次数为1
};
    // 将参数 iterations 设置为成员变量 iterations_
    this->iterations_ = iterations;
    // 返回当前对象的引用，用于支持链式调用
    return *this;
  }

  // 返回成员变量 iterations_ 的值
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 执行 Q8 测试
  void testQ8() const {
    // 创建随机设备对象
    std::random_device randomDevice;
    // 创建 Mersenne Twister 伪随机数生成器对象
    auto rng = std::mt19937(randomDevice());
    // 创建返回类型为 uint8_t 的均匀分布绑定对象
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建输入、输出和参考输出向量
    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    std::vector<uint8_t> output((batchSize() - 1) * outputStride() + channels());
    std::vector<float> outputRef(batchSize() * channels());

    // 执行指定次数的迭代
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 生成随机输入数据
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      // 将输出向量填充为 0xA5
      std::fill(output.begin(), output.end(), 0xA5);

      /* 计算参考结果 */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          // 计算输入值的量化后值
          const float x = inputScale() *
              (int32_t(input[i * inputStride() + c]) -
               int32_t(inputZeroPoint()));
          // 计算 Hardswish 函数应用后的值
          const float hardswishX =
            x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
          // 缩放 Hardswish 函数输出
          const float scaledHardswishX = hardswishX / outputScale();
          // 量化输出值并截断到指定范围
          float y = scaledHardswishX;
          y = std::min<float>(y, int32_t(qmax()) - int32_t(outputZeroPoint()));
          y = std::max<float>(y, int32_t(qmin()) - int32_t(outputZeroPoint()));
          // 存储最终的量化输出结果到 outputRef
          outputRef[i * channels() + c] = y + int32_t(outputZeroPoint());
        }
      }

      /* 创建、设置、运行和销毁 Hardswish 运算符 */
      // 初始化 QNNPACK 库
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      // 声明 Hardswish 运算符指针
      pytorch_qnnp_operator_t hardswishOp = nullptr;

      // 创建 Q8 格式的 Hardswish 运算符
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_hardswish_nc_q8(
              channels(),
              inputZeroPoint(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              qmin(),
              qmax(),
              0,
              &hardswishOp));
      ASSERT_NE(nullptr, hardswishOp);

      // 设置 Hardswish 运算符的输入和输出
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_hardswish_nc_q8(
              hardswishOp,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      // 运行 Hardswish 运算符
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(hardswishOp, nullptr /* thread pool */));

      // 销毁 Hardswish 运算符
      ASSERT_EQ(
          pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(hardswishOp));
      hardswishOp = nullptr;

      /* 验证结果 */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          // 检查输出是否接近参考输出
          ASSERT_NEAR(
              float(int32_t(output[i * outputStride() + c])),
              outputRef[i * channels() + c],
              0.6f);
        }
      }
    }
  }

这部分代码定义了一个类的私有成员结束。


 private:
  size_t batchSize_{1};

声明了一个私有成员变量 `batchSize_`，默认初始化为 1，表示批处理大小。


  size_t channels_{1};

声明了一个私有成员变量 `channels_`，默认初始化为 1，表示数据通道数。


  size_t inputStride_{0};

声明了一个私有成员变量 `inputStride_`，默认初始化为 0，表示输入数据的步幅。


  size_t outputStride_{0};

声明了一个私有成员变量 `outputStride_`，默认初始化为 0，表示输出数据的步幅。


  float inputScale_{0.75f};

声明了一个私有成员变量 `inputScale_`，默认初始化为 0.75，表示输入数据的缩放比例。


  uint8_t inputZeroPoint_{121};

声明了一个私有成员变量 `inputZeroPoint_`，默认初始化为 121，表示输入数据的零点偏移。


  float outputScale_{0.75f};

声明了一个私有成员变量 `outputScale_`，默认初始化为 0.75，表示输出数据的缩放比例。


  uint8_t outputZeroPoint_{121};

声明了一个私有成员变量 `outputZeroPoint_`，默认初始化为 121，表示输出数据的零点偏移。


  uint8_t qmin_{0};

声明了一个私有成员变量 `qmin_`，默认初始化为 0，表示量化的最小值。


  uint8_t qmax_{255};

声明了一个私有成员变量 `qmax_`，默认初始化为 255，表示量化的最大值。


  size_t iterations_{15};

声明了一个私有成员变量 `iterations_`，默认初始化为 15，表示迭代次数。
};


注释：


# 这行代码似乎是一个语法错误或者代码截断。在正常的程序中，这种结构通常是一个对象、数组或函数的结束标志或分隔符。
```