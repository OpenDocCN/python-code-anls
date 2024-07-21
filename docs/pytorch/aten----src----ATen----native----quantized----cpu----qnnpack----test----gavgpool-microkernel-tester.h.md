# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\gavgpool-microkernel-tester.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>  // 包含标准库算法头文件
#include <cassert>    // 包含断言相关头文件
#include <cmath>      // 包含数学函数头文件
#include <cstddef>    // 包含标准库大小类型定义头文件
#include <cstdlib>    // 包含标准库通用工具头文件
#include <functional> // 包含函数对象头文件
#include <random>     // 包含随机数生成器头文件
#include <vector>     // 包含动态数组容器头文件

#include <qnnpack/AlignedAllocator.h> // 包含 QNNPACK 的内存对齐分配器头文件
#include <qnnpack/params.h>           // 包含 QNNPACK 的参数定义头文件
#include <qnnpack/requantization.h>   // 包含 QNNPACK 的重新量化函数头文件

class GAvgPoolMicrokernelTester {
 public:
  inline GAvgPoolMicrokernelTester& m(size_t m) {
    assert(m != 0);   // 断言 m 不为 0
    this->m_ = m;     // 设置私有成员 m_
    return *this;     // 返回当前对象引用
  }

  inline size_t m() const {
    return this->m_;  // 返回私有成员 m_
  }

  inline GAvgPoolMicrokernelTester& n(size_t n) {
    assert(n != 0);   // 断言 n 不为 0
    this->n_ = n;     // 设置私有成员 n_
    return *this;     // 返回当前对象引用
  }

  inline size_t n() const {
    return this->n_;  // 返回私有成员 n_
  }

  inline GAvgPoolMicrokernelTester& nr(size_t nr) {
    assert(nr != 0);  // 断言 nr 不为 0
    this->nr_ = nr;   // 设置私有成员 nr_
    return *this;     // 返回当前对象引用
  }

  inline size_t nr() const {
    return this->nr_; // 返回私有成员 nr_
  }

  inline size_t packedN() const {
    return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();  // 返回对齐到 nr 的 n 的大小
  }

  inline GAvgPoolMicrokernelTester& xStride(size_t xStride) {
    assert(xStride != 0);       // 断言 xStride 不为 0
    this->xStride_ = xStride;   // 设置私有成员 xStride_
    return *this;               // 返回当前对象引用
  }

  inline size_t xStride() const {
    if (this->xStride_ == 0) {
      return n();              // 如果 xStride_ 为 0，则返回 n()
    } else {
      assert(this->xStride_ >= n());
      return this->xStride_;   // 否则返回 xStride_
    }
  }

  inline GAvgPoolMicrokernelTester& xScale(float xScale) {
    assert(xScale > 0.0f);      // 断言 xScale 大于 0
    assert(std::isnormal(xScale));  // 断言 xScale 是正常数
    this->xScale_ = xScale;     // 设置私有成员 xScale_
    return *this;               // 返回当前对象引用
  }

  inline float xScale() const {
    return this->xScale_;       // 返回私有成员 xScale_
  }

  inline GAvgPoolMicrokernelTester& xZeroPoint(uint8_t xZeroPoint) {
    this->xZeroPoint_ = xZeroPoint;  // 设置私有成员 xZeroPoint_
    return *this;                    // 返回当前对象引用
  }

  inline uint8_t xZeroPoint() const {
    return this->xZeroPoint_;        // 返回私有成员 xZeroPoint_
  }

  inline GAvgPoolMicrokernelTester& yScale(float yScale) {
    assert(yScale > 0.0f);      // 断言 yScale 大于 0
    assert(std::isnormal(yScale));  // 断言 yScale 是正常数
    this->yScale_ = yScale;     // 设置私有成员 yScale_
    return *this;               // 返回当前对象引用
  }

  inline float yScale() const {
    return this->yScale_;       // 返回私有成员 yScale_
  }

  inline GAvgPoolMicrokernelTester& yZeroPoint(uint8_t yZeroPoint) {
    this->yZeroPoint_ = yZeroPoint;  // 设置私有成员 yZeroPoint_
    return *this;                    // 返回当前对象引用
  }

  inline uint8_t yZeroPoint() const {
    return this->yZeroPoint_;        // 返回私有成员 yZeroPoint_
  }

  inline GAvgPoolMicrokernelTester& yMin(uint8_t yMin) {
    this->yMin_ = yMin;         // 设置私有成员 yMin_
    return *this;               // 返回当前对象引用
  }

  inline uint8_t yMin() const {
    return this->yMin_;         // 返回私有成员 yMin_
  }

  inline GAvgPoolMicrokernelTester& yMax(uint8_t yMax) {
    this->yMax_ = yMax;         // 设置私有成员 yMax_
    return *this;               // 返回当前对象引用
  }

  inline uint8_t yMax() const {
    return this->yMax_;         // 返回私有成员 yMax_
  }

  inline GAvgPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;  // 设置私有成员 iterations_
    return *this;                    // 返回当前对象引用
  }

  inline size_t iterations() const {
    return this->iterations_;   // 返回私有成员 iterations_
  }

  void test(pytorch_q8gavgpool_up_ukernel_function q8gavgpool) const {
    std::random_device randomDevice;  // 创建随机设备对象
    // 使用随机设备生成随机数引擎 rng
    auto rng = std::mt19937(randomDevice());
    // 使用 std::bind 绑定 uniform_int_distribution 和 rng，生成 u8rng 函数对象
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建长度为 (m() - 1) * xStride() + n() 的 uint8_t 类型向量 x
    std::vector<uint8_t> x((m() - 1) * xStride() + n());
    // 创建长度为 n() 的 uint8_t 类型零向量 zero
    std::vector<uint8_t> zero(n());
    // 创建长度为 n() 的 uint8_t 类型向量 y
    std::vector<uint8_t> y(n());
    // 创建长度为 n() 的 uint8_t 类型向量 yRef，用于存储参考结果
    std::vector<uint8_t> yRef(n());
    // 创建长度为 n() 的 float 类型向量 yFP，用于存储浮点运算结果
    std::vector<float> yFP(n());
    // 创建长度为 n() 的 int32_t 类型向量 yAcc，用于存储累加结果
    std::vector<int32_t> yAcc(n());

    // 迭代执行计算，每次迭代都进行如下操作
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 生成随机数据填充向量 x
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      // 将向量 y 填充为 0xA5
      std::fill(y.begin(), y.end(), 0xA5);

      /* Prepare quantization parameters */
      // 计算平均池化量化参数
      const union pytorch_qnnp_avgpool_quantization_params quantizationParams =
          pytorch_qnnp_compute_avgpool_quantization_params(
              -int32_t(xZeroPoint()) * int32_t(m()),
              xScale() / (yScale() * float(m())),
              yZeroPoint(),
              yMin(),
              yMax());
      // 计算标量平均池化量化参数
      const union pytorch_qnnp_avgpool_quantization_params
          scalarQuantizationParams =
              pytorch_qnnp_compute_scalar_avgpool_quantization_params(
                  -int32_t(xZeroPoint()) * int32_t(m()),
                  xScale() / (yScale() * float(m())),
                  yZeroPoint(),
                  yMin(),
                  yMax());

      /* Compute reference results */
      // 计算参考结果
      for (size_t j = 0; j < n(); j++) {
        // 初始化累加器为标量量化参数的偏置值
        int32_t acc = scalarQuantizationParams.scalar.bias;
        // 对每一列数据进行累加
        for (size_t i = 0; i < m(); i++) {
          acc += x[i * xStride() + j];
        }
        // 将累加结果保存到 yAcc 中
        yAcc[j] = acc;
        // 对累加结果进行量化，并保存到 yRef 中
        yRef[j] = pytorch_qnnp_avgpool_quantize(acc, scalarQuantizationParams);
        // 使用浮点运算计算 yFP，并进行范围限制
        yFP[j] = float(acc) * (xScale() / (yScale() * float(m()))) +
            float(yZeroPoint());
        yFP[j] = std::min<float>(yFP[j], float(yMax()));
        yFP[j] = std::max<float>(yFP[j], float(yMin()));
      }

      /* Call optimized micro-kernel */
      // 调用优化的微内核函数 q8gavgpool 进行计算
      q8gavgpool(
          m(),
          n(),
          x.data(),
          xStride() * sizeof(uint8_t),
          zero.data(),
          y.data(),
          &quantizationParams);

      /* Verify results */
      // 验证计算结果
      for (size_t i = 0; i < n(); i++) {
        // 使用断言验证 y[i] 在指定范围内
        ASSERT_LE(uint32_t(y[i]), uint32_t(yMax()))
            << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_GE(uint32_t(y[i]), uint32_t(yMin()))
            << "at position " << i << ", m = " << m() << ", n = " << n();
        // 使用断言验证浮点运算结果与 yFP[i] 的接近程度
        ASSERT_NEAR(float(int32_t(y[i])), yFP[i], 0.5001f)
            << "at position " << i << ", m = " << m() << ", n = " << n()
            << ", acc = " << yAcc[i];
        // 使用断言验证量化结果与 yRef[i] 的相等性
        ASSERT_EQ(uint32_t(yRef[i]), uint32_t(y[i]))
            << "at position " << i << ", m = " << m() << ", n = " << n()
            << ", acc = " << yAcc[i];
      }
    }
  }
    // 创建一个大小为 packedN() 的对齐分配器为 16 的 int32_t 向量
    std::vector<int32_t, AlignedAllocator<int32_t, 16>> mpAcc(packedN());
    // 创建一个大小为 n() 的 uint8_t 向量 zero，用于存储零值
    std::vector<uint8_t> zero(n());
    // 创建一个大小为 n() 的 uint8_t 向量 y，用于存储计算后的结果
    std::vector<uint8_t> y(n());
    // 创建一个大小为 n() 的 uint8_t 向量 yRef，用于存储参考结果
    std::vector<uint8_t> yRef(n());
    // 创建一个大小为 n() 的 float 向量 yFP，用于存储浮点数结果
    std::vector<float> yFP(n());
    // 创建一个大小为 n() 的 int32_t 向量 yAcc，用于存储累加结果
    std::vector<int32_t> yAcc(n());

    // 迭代执行 iterations() 次
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 使用 u8rng 函数生成随机数填充 x 向量
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      // 将 y 向量填充为 0xA5
      std::fill(y.begin(), y.end(), 0xA5);

      /* 准备量化参数 */
      // 计算平均池化的量化参数并存储在 quantizationParams 中
      const union pytorch_qnnp_avgpool_quantization_params quantizationParams =
          pytorch_qnnp_compute_avgpool_quantization_params(
              -int32_t(xZeroPoint()) * int32_t(m()),
              xScale() / (yScale() * float(m())),
              yZeroPoint(),
              yMin(),
              yMax());
      // 计算标量版本的平均池化量化参数并存储在 scalarQuantizationParams 中
      const union pytorch_qnnp_avgpool_quantization_params
          scalarQuantizationParams =
              pytorch_qnnp_compute_scalar_avgpool_quantization_params(
                  -int32_t(xZeroPoint()) * int32_t(m()),
                  xScale() / (yScale() * float(m())),
                  yZeroPoint(),
                  yMin(),
                  yMax());

      /* 计算参考结果 */
      // 计算参考结果并填充到 yAcc、yRef 和 yFP 向量中
      for (size_t j = 0; j < n(); j++) {
        int32_t acc = scalarQuantizationParams.scalar.bias;
        for (size_t i = 0; i < m(); i++) {
          acc += x[i * xStride() + j];
        }

        yAcc[j] = acc;
        yRef[j] = pytorch_qnnp_avgpool_quantize(acc, scalarQuantizationParams);
        yFP[j] = float(acc) * (xScale() / (yScale() * float(m()))) +
            float(yZeroPoint());
        yFP[j] = std::min<float>(yFP[j], float(yMax()));
        yFP[j] = std::max<float>(yFP[j], float(yMin()));
      }

      /* 调用优化的微内核 */
      // 调用 q8gavgpool 函数执行量化平均池化操作
      q8gavgpool(
          m(),
          n(),
          x.data(),
          xStride() * sizeof(uint8_t),
          zero.data(),
          mpAcc.data(),
          y.data(),
          &quantizationParams);

      /* 验证结果 */
      // 验证 y 向量的结果是否满足预期条件，并在不满足时输出相关信息
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(uint32_t(y[i]), uint32_t(yMax()))
            << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_GE(uint32_t(y[i]), uint32_t(yMin()))
            << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_NEAR(float(int32_t(y[i])), yFP[i], 0.5001f)
            << "at position " << i << ", m = " << m() << ", n = " << n()
            << ", acc = " << yAcc[i];
        ASSERT_EQ(uint32_t(yRef[i]), uint32_t(y[i]))
            << "at position " << i << ", m = " << m() << ", n = " << n()
            << ", acc = " << yAcc[i];
      }
    }
  }

 private:
  size_t m_{1};  // m 维度的大小，默认为 1
  size_t n_{1};  // n 维度的大小，默认为 1
  size_t nr_{1};  // nr 维度的大小，默认为 1
  size_t xStride_{0};  // x 向量的跨度，默认为 0
  float xScale_{1.25f};  // x 的缩放因子，默认为 1.25
  float yScale_{0.75f};  // y 的缩放因子，默认为 0.75
  uint8_t xZeroPoint_{121};  // x 的零点，默认为 121
  uint8_t yZeroPoint_{133};  // y 的零点，默认为 133
  uint8_t yMin_{0};  // y 的最小值，默认为 0
  uint8_t yMax_{255};  // y 的最大值，默认为 255
  size_t iterations_{15};  // 迭代次数，默认为 15
};



# 这行代码是一个单独的分号，通常用于终止语句或语句块。在此处，它可能是代码的一部分，但没有上下文很难确定其确切作用。
# 可能情况是这行分号是某个语句的结束，但由于缺少周围代码，无法准确描述其具体含义。
```