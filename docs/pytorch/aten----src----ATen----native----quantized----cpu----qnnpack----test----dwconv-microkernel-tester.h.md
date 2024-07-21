# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\dwconv-microkernel-tester.h`

```py
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

#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

// DWConvMicrokernelTester 类的定义，用于测试深度可分离卷积的微内核
class DWConvMicrokernelTester {
 public:
  // 设置并返回测试的宽度
  inline DWConvMicrokernelTester& width(uint32_t width) {
    assert(width >= 1);  // 断言宽度必须大于等于1
    this->width_ = width;
    return *this;
  }

  // 返回当前设置的测试宽度
  inline uint32_t width() const {
    return this->width_;
  }

  // 设置并返回子采样率
  inline DWConvMicrokernelTester& subsampling(uint32_t subsampling) {
    assert(subsampling >= 1);  // 断言子采样率必须大于等于1
    this->subsampling_ = subsampling;
    return *this;
  }

  // 返回当前设置的子采样率
  inline uint32_t subsampling() const {
    return this->subsampling_;
  }

  // 设置并返回通道数
  inline DWConvMicrokernelTester& channels(uint32_t channels) {
    assert(channels >= 1);  // 断言通道数必须大于等于1
    this->channels_ = channels;
    return *this;
  }

  // 返回当前设置的通道数
  inline uint32_t channels() const {
    return this->channels_;
  }

  // 设置并返回压缩比率
  inline DWConvMicrokernelTester& cr(uint32_t cr) {
    assert(cr != 0);  // 断言压缩比率不能为0
    assert((cr & (cr - 1)) == 0);  // 断言压缩比率是2的幂
    this->cr_ = cr;
    return *this;
  }

  // 返回当前设置的压缩比率
  inline uint32_t cr() const {
    return this->cr_;
  }

  // 返回经过压缩比率调整后的通道数
  inline uint32_t packedChannels() const {
    return (channels() + (cr() - 1)) & -cr();
  }

  // 设置并返回卷积核的高度
  inline DWConvMicrokernelTester& kernelHeight(uint32_t kernelHeight) {
    assert(kernelHeight != 0);  // 断言卷积核高度不能为0
    this->kernelHeight_ = kernelHeight;
    return *this;
  }

  // 返回当前设置的卷积核高度
  inline uint32_t kernelHeight() const {
    return this->kernelHeight_;
  }

  // 设置并返回卷积核的宽度
  inline DWConvMicrokernelTester& kernelWidth(uint32_t kernelWidth) {
    assert(kernelWidth != 0);  // 断言卷积核宽度不能为0
    this->kernelWidth_ = kernelWidth;
    return *this;
  }

  // 返回当前设置的卷积核宽度
  inline uint32_t kernelWidth() const {
    return this->kernelWidth_;
  }

  // 返回卷积核的总大小
  inline uint32_t kernelSize() const {
    return kernelHeight() * kernelWidth();
  }

  // 设置并返回输入步长
  inline DWConvMicrokernelTester& inputStride(uint32_t inputStride) {
    assert(inputStride != 0);  // 断言输入步长不能为0
    this->inputStride_ = inputStride;
    return *this;
  }

  // 返回当前设置的输入步长
  inline uint32_t inputStride() const {
    if (this->inputStride_ == 0) {
      return channels();
    } else {
      assert(this->inputStride_ >= channels());
      return this->inputStride_;
    }
  }

  // 设置并返回输出步长
  inline DWConvMicrokernelTester& outputStride(uint32_t outputStride) {
    assert(outputStride != 0);  // 断言输出步长不能为0
    this->outputStride_ = outputStride;
    return *this;
  }

  // 返回当前设置的输出步长
  inline uint32_t outputStride() const {
    if (this->outputStride_ == 0) {
      return channels();
    } else {
      assert(this->outputStride_ >= channels());
      return this->outputStride_;
    }
  }

  // 设置输入零点值并返回当前对象
  inline DWConvMicrokernelTester& inputZeroPoint(uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  // 返回当前设置的输入零点值
  inline uint8_t inputZeroPoint() const {

    return this->inputZeroPoint_;
  }


这段代码定义了一个用于测试深度可分离卷积微内核的测试器类 `DWConvMicrokernelTester`，包含了各种设置和获取测试参数的方法，并使用断言确保参数的有效性和正确性。
    // 返回当前对象的 inputZeroPoint_ 成员变量
    return this->inputZeroPoint_;
  }

  // 设置 kernelZeroPoint_ 成员变量并返回当前对象的引用
  inline DWConvMicrokernelTester& kernelZeroPoint(uint8_t kernelZeroPoint) {
    this->kernelZeroPoint_ = kernelZeroPoint;
    return *this;
  }

  // 返回当前对象的 kernelZeroPoint_ 成员变量
  inline uint8_t kernelZeroPoint() const {
    return this->kernelZeroPoint_;
  }

  // 设置 qmin_ 成员变量并返回当前对象的引用
  inline DWConvMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 返回当前对象的 qmin_ 成员变量
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置 qmax_ 成员变量并返回当前对象的引用
  inline DWConvMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回当前对象的 qmax_ 成员变量
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置 iterations_ 成员变量并返回当前对象的引用
  inline DWConvMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前对象的 iterations_ 成员变量
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 执行测试函数，传入卷积函数和是否按通道计算的标志
  void test(
      pytorch_q8dwconv2d_up_ukernel_function q8dwconv,
      bool per_channel = false) const {
    // 生成随机设备和随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    // 创建生成均匀分布的随机整数的函数对象
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    // 创建生成均匀分布的随机无符号整数的函数对象
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 初始化输入向量，大小由卷积核大小、宽度、下采样率等参数决定
    std::vector<uint8_t> input(
        (kernelSize() + (width() * subsampling() - 1) * kernelHeight() - 1) *
            inputStride() +
        channels() + 8);
    // 初始化卷积核向量，大小由通道数和卷积核大小决定
    std::vector<uint8_t> kernel(channels() * kernelSize());
    // 初始化打包后的权重向量，大小由卷积核大小和打包后的通道数决定
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedWeights(
        (kernelSize() + sizeof(int32_t) / sizeof(uint8_t)) * packedChannels());
    // 初始化偏置向量，大小由打包后的通道数决定
    std::vector<int32_t> bias(packedChannels());
    // 初始化累加器向量，大小由宽度和通道数决定
    std::vector<int32_t> accumulators(width() * channels());
    // 初始化输出向量，大小由宽度和输出步长决定
    std::vector<uint8_t> output((width() - 1) * outputStride() + channels());
    // 初始化间接输入向量，大小由卷积核大小和宽度、下采样率决定
    std::vector<const uint8_t*> indirectInput(
        kernelSize() + (width() * subsampling() - 1) * kernelHeight());

    // 设置输入指针，指向输入数据的起始位置加偏移量
    const uint8_t* inputPtr = input.data() + 8;

    // 迭代执行测试函数，次数由 iterations_ 成员变量决定
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 生成随机输入数据
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      // 生成随机卷积核数据
      std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      // 生成随机偏置数据
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      // 将累加器数据初始化为零
      std::fill(accumulators.begin(), accumulators.end(), 0);

      // 断言：输入数据的最大值和最小值不相等
      ASSERT_NE(
          *std::max_element(input.cbegin(), input.cend()),
          *std::min_element(input.cbegin(), input.cend()));
      // 断言：卷积核数据的最大值和最小值不相等
      ASSERT_NE(
          *std::max_element(kernel.cbegin(), kernel.cend()),
          *std::min_element(kernel.cbegin(), kernel.cend()));

      // 将打包后的权重数据填充为预定义的值 0xA5
      std::fill(packedWeights.begin(), packedWeights.end(), 0xA5);

      // 初始化零点数据向量，大小为通道数加上预定义偏移量
      size_t num_zero_points_padded = channels() + 8;
      std::vector<uint8_t> kernel_zero_points(
          num_zero_points_padded, 0);
      // 如果按通道计算，则生成随机的通道零点数据
      if (per_channel) {
        std::generate(
            kernel_zero_points.begin(),
            kernel_zero_points.begin() + channels(),
            std::ref(u8rng));
      }

      // 调用 PyTorch 的打包函数，准备权重数据用于量化卷积
      pytorch_pack_q8dw_w(
          kernelHeight(),
          kernelWidth(),
          channels(),
          cr(),
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          // 如果不是运行时量化，则获取输入的零点
          inputZeroPoint(),
          // 获取卷积核的零点数据
          kernel_zero_points.data(),
#if defined(__arm__) || defined(_M_ARM)
          // 如果是 ARM 架构，使用魔数方法进行 FP32 重新量化
          const uint8_t referenceOutput = pytorch_qnnp_fp32_requantize_magic(
              accumulators[x * channels() + c], scalarRequantizationParams, c);
#else
          // 如果不是 ARM 架构，使用标准的 FP32 重新量化方法
          const uint8_t referenceOutput = pytorch_qnnp_fp32_requantize(
              accumulators[x * channels() + c], scalarRequantizationParams, c);
#endif
          // 计算累加器的缩放值
          const double scaledAccumulator =
              accumulators[x * channels() + c] * requantization_scales[c] +
              double(outputZeroPoint);
          // 将累加器的值限制在指定的量化范围内
          const double clampedAccumulator = std::max(
              std::min(scaledAccumulator, double(qmax())), double(qmin()));
          // 断言检查累加器值与输出值的接近程度，允许的误差为0.6
          ASSERT_NEAR(
              clampedAccumulator, double(output[x * outputStride() + c]), 0.6)
              << "x = " << x << ", channel = " << c;
          // 断言检查参考输出与实际输出的一致性
          ASSERT_EQ(
              uint32_t(referenceOutput),
              uint32_t(output[x * outputStride() + c]))
              << "x = " << x << ", channel = " << c;
        }
      }
    }
  }

  // 执行测试函数
  void test(
      pytorch_q8dwconv2d_mp_ukernel_function q8dwconv,
      bool per_channel = false) const {
    // 断言检查卷积核尺寸是否为 25
    ASSERT_EQ(25, kernelSize())
        << "only 5x5 microkernel is currently supported";

    // 初始化随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 初始化输入数据向量
    std::vector<uint8_t> input(
        (kernelSize() + (width() * subsampling() - 1) * kernelHeight() - 1) *
            inputStride() +
        channels() + 8);
    // 初始化卷积核数据向量
    std::vector<uint8_t> kernel(channels() * kernelSize());
    // 初始化按对齐要求分配的权重数据向量
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedWeights(
        (kernelSize() + sizeof(int32_t) / sizeof(uint8_t)) * packedChannels());
    // 初始化偏置数据向量
    std::vector<int32_t> bias(packedChannels());
    // 初始化累加器数据向量
    std::vector<int32_t> accumulators(width() * channels());
    // 初始化 MP 累加器数据向量
    std::vector<int32_t> mpAcc(width() * packedChannels());
    // 初始化输出数据向量
    std::vector<uint8_t> output((width() - 1) * outputStride() + channels());
    // 初始化间接输入数据向量
    std::vector<const uint8_t*> indirectInput(
        kernelSize() + (width() * subsampling() - 1) * kernelHeight());

    // 设置输入指针，跳过前8字节
    const uint8_t* inputPtr = input.data() + 8;

#if defined(__arm__) || defined(_M_ARM)
          // 如果是 ARM 架构，使用魔数方法进行 FP32 重新量化
          const uint8_t referenceOutput = pytorch_qnnp_fp32_requantize_magic(
              accumulators[x * channels() + c], scalarRequantizationParams, c);
#else
          // 如果不是 ARM 架构，使用标准的 FP32 重新量化方法
          const uint8_t referenceOutput = pytorch_qnnp_fp32_requantize(
              accumulators[x * channels() + c], scalarRequantizationParams, c);
#endif
          // 计算经过量化和输出零点偏移后的累加器值
          const double scaledAccumulator =
              accumulators[x * channels() + c] * requantization_scales[c] +
              double(outputZeroPoint);
          // 对累加器值进行上下限裁剪，确保在量化范围内
          const double clampedAccumulator = std::max(
              std::min(scaledAccumulator, double(qmax())), double(qmin()));
          // 断言：验证裁剪后的累加器值与输出数组中的值之间的近似性
          ASSERT_NEAR(
              clampedAccumulator, double(output[x * outputStride() + c]), 0.6)
              << "x = " << x << ", channel = " << c;
          // 断言：验证输出数组中的值与参考输出的值的一致性
          ASSERT_EQ(
              uint32_t(referenceOutput),
              uint32_t(output[x * outputStride() + c]))
              << "x = " << x << ", channel = " << c;
        }
      }
    }
  }

 private:
  // 神经网络层的参数
  uint32_t channels_{1};
  uint32_t cr_{1};
  uint32_t width_{1};
  uint32_t subsampling_{1};
  uint32_t kernelHeight_{1};
  uint32_t kernelWidth_{1};
  uint32_t inputStride_{0};
  uint32_t outputStride_{0};
  uint8_t inputZeroPoint_{127};
  uint8_t kernelZeroPoint_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{3};
};
```