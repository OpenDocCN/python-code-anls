# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\convolution-operator-tester.h`

```py
/*
 * 版权声明和许可信息，指出此代码的版权和许可协议
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// 引入必要的头文件
#include <algorithm>        // 标准算法库，包括算法操作函数
#include <cassert>          // 断言相关的库，用于运行时检查
#include <cmath>            // 数学函数库，包含各种数学函数和常量
#include <cstddef>          // 标准库定义的常量宏
#include <cstdlib>          // 标准库的一般工具函数
#include <functional>       // 函数对象库，包含函数对象类及其操作
#include <random>           // 随机数库，用于生成随机数和分布
#include <vector>           // 动态数组库，包含向量容器类
#include <memory>           // 动态内存管理，包含智能指针等
#include <pytorch_qnnpack.h>    // PyTorch QNNPACK 库，用于量化神经网络计算
#include <qnnpack_func.h>   // QNNPACK 库的函数接口

#include "test_utils.h"     // 测试工具函数，用于测试支持

using namespace qnnpack::testing;   // 使用 qnnpack::testing 命名空间

// 定义 ConvolutionOperatorTester 类
class ConvolutionOperatorTester {
 public:
  // 返回测试器的维度数量
  inline size_t dimensionality() const {
    return this->dimensionality_;
  }

  // 设置测试器的维度数量
  inline ConvolutionOperatorTester& dimensionality(size_t dimensionality) {
    // 断言维度必须是 2 或 3
    assert(dimensionality == 2 || dimensionality == 3);
    this->dimensionality_ = dimensionality;
    return *this;
  }

  // 设置填充参数，根据不同维度数量设置不同的填充深度
  inline ConvolutionOperatorTester& padding(uint32_t padding) {
    if (this->dimensionality_ == 3) {
      this->paddingDepth_ = padding;
    }
    this->paddingHeight_ = padding;
    this->paddingWidth_ = padding;
    return *this;
  }

  // 设置二维填充参数
  inline ConvolutionOperatorTester& padding(
      uint32_t paddingHeight,
      uint32_t paddingWidth) {
    this->paddingHeight_ = paddingHeight;
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  // 设置三维填充参数
  inline ConvolutionOperatorTester& padding(
      uint32_t paddingDepth,
      uint32_t paddingHeight,
      uint32_t paddingWidth) {
    this->paddingDepth_ = paddingDepth;
    return this->padding(paddingHeight, paddingWidth);
  }

  // 设置深度填充参数
  inline ConvolutionOperatorTester& paddingDepth(uint32_t paddingDepth) {
    this->paddingDepth_ = paddingDepth;
    return *this;
  }

  // 设置高度填充参数
  inline ConvolutionOperatorTester& paddingHeight(uint32_t paddingHeight) {
    this->paddingHeight_ = paddingHeight;
    return *this;
  }

  // 设置宽度填充参数
  inline ConvolutionOperatorTester& paddingWidth(uint32_t paddingWidth) {
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  // 返回当前深度填充参数
  inline uint32_t paddingDepth() const {
    return this->paddingDepth_;
  }

  // 返回当前高度填充参数
  inline uint32_t paddingHeight() const {
    return this->paddingHeight_;
  }

  // 返回当前宽度填充参数
  inline uint32_t paddingWidth() const {
    return this->paddingWidth_;
  }

  // 设置输入的高度和宽度
  inline ConvolutionOperatorTester& inputSize(
      uint32_t inputHeight,
      uint32_t inputWidth) {
    assert(inputHeight >= 1);
    assert(inputWidth >= 1);
    this->inputHeight_ = inputHeight;
    this->inputWidth_ = inputWidth;
    return *this;
  }

  // 设置输入的深度、高度和宽度
  inline ConvolutionOperatorTester& inputSize(
      uint32_t inputDepth,
      uint32_t inputHeight,
      uint32_t inputWidth) {
    assert(inputDepth >= 1);
    this->inputDepth_ = inputDepth;
    return this->inputSize(inputHeight, inputWidth);
  }

  // 设置输入的深度
  inline ConvolutionOperatorTester& inputDepth(uint32_t inputDepth) {
    assert(inputDepth >= 1);
    this->inputDepth_ = inputDepth;
    return *this;
  }

  // 返回当前输入的深度
  inline uint32_t inputDepth() const {
    return this->inputDepth_;
  }

  // 设置输入的高度
  inline ConvolutionOperatorTester& inputHeight(uint32_t inputHeight) {
    assert(inputHeight >= 1);
    this->inputHeight_ = inputHeight;
    return *this;
  }

  // 设置输入的宽度
  inline ConvolutionOperatorTester& inputWidth(uint32_t inputWidth) {
    assert(inputWidth >= 1);
    this->inputWidth_ = inputWidth;
    return *this;
  }

 private:
  // 私有成员变量，存储测试器的维度数量、填充参数和输入尺寸
  size_t dimensionality_{2};
  uint32_t paddingDepth_{0};
  uint32_t paddingHeight_{0};
  uint32_t paddingWidth_{0};
  uint32_t inputDepth_{1};
  uint32_t inputHeight_{1};
  uint32_t inputWidth_{1};
};
    // 断言输入高度大于等于1，确保输入合法性
    assert(inputHeight >= 1);
    // 设置对象的输入高度
    this->inputHeight_ = inputHeight;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline uint32_t inputHeight() const {
    // 返回对象的输入高度
    return this->inputHeight_;
  }

  inline ConvolutionOperatorTester& inputWidth(uint32_t inputWidth) {
    // 断言输入宽度大于等于1，确保输入合法性
    assert(inputWidth >= 1);
    // 设置对象的输入宽度
    this->inputWidth_ = inputWidth;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline uint32_t inputWidth() const {
    // 返回对象的输入宽度
    return this->inputWidth_;
  }

  inline ConvolutionOperatorTester& groups(uint32_t groups) {
    // 断言分组数量大于等于1，确保输入合法性
    assert(groups >= 1);
    // 设置对象的分组数量
    this->groups_ = groups;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline uint32_t groups() const {
    // 返回对象的分组数量
    return this->groups_;
  }

  inline ConvolutionOperatorTester& groupInputChannels(
      size_t groupInputChannels) {
    // 断言每组输入通道数大于等于1，确保输入合法性
    assert(groupInputChannels >= 1);
    // 设置对象的每组输入通道数
    this->groupInputChannels_ = groupInputChannels;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline size_t groupInputChannels() const {
    // 返回对象的每组输入通道数
    return this->groupInputChannels_;
  }

  inline ConvolutionOperatorTester& per_channel(bool per_channel) {
    // 设置是否为每通道参数（per channel）模式
    this->per_channel_ = per_channel;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline bool per_channel() const {
    // 返回是否为每通道参数（per channel）模式
    return this->per_channel_;
  }

  inline ConvolutionOperatorTester& groupOutputChannels(
      size_t groupOutputChannels) {
    // 断言每组输出通道数大于等于1，确保输入合法性
    assert(groupOutputChannels >= 1);
    // 设置对象的每组输出通道数
    this->groupOutputChannels_ = groupOutputChannels;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline size_t groupOutputChannels() const {
    // 返回对象的每组输出通道数
    return this->groupOutputChannels_;
  }

  inline ConvolutionOperatorTester& batchSize(size_t batchSize) {
    // 设置对象的批处理大小
    this->batchSize_ = batchSize;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline size_t batchSize() const {
    // 返回对象的批处理大小
    return this->batchSize_;
  }

  inline ConvolutionOperatorTester& kernelSize(uint32_t kernelSize) {
    // 断言卷积核大小大于等于1，确保输入合法性
    assert(kernelSize >= 1);
    // 如果是3D卷积，则设置卷积核深度
    if (this->dimensionality_ == 3) {
      this->kernelDepth_ = kernelSize;
    }
    // 设置卷积核高度和宽度
    this->kernelHeight_ = kernelSize;
    this->kernelWidth_ = kernelSize;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline ConvolutionOperatorTester& kernelSize(
      uint32_t kernelHeight,
      uint32_t kernelWidth) {
    // 断言卷积核高度和宽度都大于等于1，确保输入合法性
    assert(kernelHeight >= 1);
    assert(kernelWidth >= 1);
    // 设置卷积核高度和宽度
    this->kernelHeight_ = kernelHeight;
    this->kernelWidth_ = kernelWidth;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline ConvolutionOperatorTester& kernelSize(
      uint32_t kernelDepth,
      uint32_t kernelHeight,
      uint32_t kernelWidth) {
    // 断言3D卷积核深度大于等于1，确保输入合法性
    assert(kernelDepth >= 1);
    // 设置卷积核深度
    this->kernelDepth_ = kernelDepth;
    // 调用重载的 kernelSize 方法设置卷积核高度和宽度
    return this->kernelSize(kernelHeight, kernelWidth);
  }

  inline ConvolutionOperatorTester& kernelDepth(uint32_t kernelDepth) {
    // 断言卷积核深度大于等于1，确保输入合法性
    assert(kernelDepth >= 1);
    // 设置卷积核深度
    this->kernelDepth_ = kernelDepth;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline uint32_t kernelDepth() const {
    // 返回对象的卷积核深度
    return this->kernelDepth_;
  }

  inline ConvolutionOperatorTester& kernelHeight(uint32_t kernelHeight) {
    // 断言卷积核高度大于等于1，确保输入合法性
    assert(kernelHeight >= 1);
    // 设置卷积核高度
    this->kernelHeight_ = kernelHeight;
    // 返回对象自身，支持链式调用
    return *this;
  }

  inline uint32_t kernelHeight() const {
    // 返回对象的卷积核高度
    return this->kernelHeight_;
  }

  inline ConvolutionOperatorTester& kernelWidth(uint32_t kernelWidth) {
    // 断言卷积核宽度大于等于1，确保输入合法性
    assert(kernelWidth >= 1);
    // 设置卷积核宽度
    this->kernelWidth_ = kernelWidth;
    // 返回对象自身，支持链式调用
    return *this;
  }
  // 设置当前对象的 kernelWidth_ 属性为指定值，并返回当前对象的引用
  this->kernelWidth_ = kernelWidth;
  return *this;
}

// 返回当前对象的 kernelWidth_ 属性值
inline uint32_t kernelWidth() const {
  return this->kernelWidth_;
}

// 设置当前对象的 dilation 相关属性为指定值，并返回当前对象的引用
inline ConvolutionOperatorTester& dilation(uint32_t dilation) {
  // 断言 dilation 至少为 1
  assert(dilation >= 1);
  // 若对象的 dimensionality_ 为 3，则设置 dilationDepth_ 属性
  if (this->dimensionality_ == 3) {
    this->dilationDepth_ = dilation;
  }
  // 设置 dilationHeight_ 和 dilationWidth_ 属性
  this->dilationHeight_ = dilation;
  this->dilationWidth_ = dilation;
  return *this;
}

// 设置当前对象的 dilationHeight_ 和 dilationWidth_ 属性为指定值，并返回当前对象的引用
inline ConvolutionOperatorTester& dilation(
    uint32_t dilationHeight,
    uint32_t dilationWidth) {
  // 断言 dilationHeight 和 dilationWidth 至少为 1
  assert(dilationHeight >= 1);
  assert(dilationWidth >= 1);
  // 设置 dilationHeight_ 和 dilationWidth_ 属性
  this->dilationHeight_ = dilationHeight;
  this->dilationWidth_ = dilationWidth;
  return *this;
}

// 设置当前对象的 dilationDepth_, dilationHeight_ 和 dilationWidth_ 属性为指定值，并返回当前对象的引用
inline ConvolutionOperatorTester& dilation(
    uint32_t dilationDepth,
    uint32_t dilationHeight,
    uint32_t dilationWidth) {
  // 断言 dilationDepth 至少为 1
  assert(dilationDepth >= 1);
  // 设置 dilationDepth_ 属性，并调用前一个 dilation 方法设置 dilationHeight_ 和 dilationWidth_ 属性
  this->dilationDepth_ = dilationDepth;
  return this->dilation(dilationHeight, dilationWidth);
}

// 设置当前对象的 dilationDepth_ 属性为指定值，并返回当前对象的引用
inline ConvolutionOperatorTester& dilationDepth(uint32_t dilationDepth) {
  // 断言 dilationDepth 至少为 1
  assert(dilationDepth >= 1);
  // 设置 dilationDepth_ 属性
  this->dilationDepth_ = dilationDepth;
  return *this;
}

// 返回当前对象的 dilationDepth_ 属性值
inline uint32_t dilationDepth() const {
  return this->dilationDepth_;
}

// 设置当前对象的 dilationHeight_ 属性为指定值，并返回当前对象的引用
inline ConvolutionOperatorTester& dilationHeight(uint32_t dilationHeight) {
  // 断言 dilationHeight 至少为 1
  assert(dilationHeight >= 1);
  // 设置 dilationHeight_ 属性
  this->dilationHeight_ = dilationHeight;
  return *this;
}

// 返回当前对象的 dilationHeight_ 属性值
inline uint32_t dilationHeight() const {
  return this->dilationHeight_;
}

// 设置当前对象的 dilationWidth_ 属性为指定值，并返回当前对象的引用
inline ConvolutionOperatorTester& dilationWidth(uint32_t dilationWidth) {
  // 断言 dilationWidth 至少为 1
  assert(dilationWidth >= 1);
  // 设置 dilationWidth_ 属性
  this->dilationWidth_ = dilationWidth;
  return *this;
}

// 返回当前对象的 dilationWidth_ 属性值
inline uint32_t dilationWidth() const {
  return this->dilationWidth_;
}

// 设置当前对象的 subsampling 相关属性为指定值，并返回当前对象的引用
inline ConvolutionOperatorTester& subsampling(uint32_t subsampling) {
  // 断言 subsampling 至少为 1
  assert(subsampling >= 1);
  // 若对象的 dimensionality_ 为 3，则设置 subsamplingDepth_ 属性
  if (this->dimensionality_ == 3) {
    this->subsamplingDepth_ = subsampling;
  }
  // 设置 subsamplingHeight_ 和 subsamplingWidth_ 属性
  this->subsamplingHeight_ = subsampling;
  this->subsamplingWidth_ = subsampling;
  return *this;
}

// 设置当前对象的 subsamplingHeight_ 和 subsamplingWidth_ 属性为指定值，并返回当前对象的引用
inline ConvolutionOperatorTester& subsampling(
    uint32_t subsamplingHeight,
    uint32_t subsamplingWidth) {
  // 断言 subsamplingHeight 和 subsamplingWidth 至少为 1
  assert(subsamplingHeight >= 1);
  assert(subsamplingWidth >= 1);
  // 设置 subsamplingHeight_ 和 subsamplingWidth_ 属性
  this->subsamplingHeight_ = subsamplingHeight;
  this->subsamplingWidth_ = subsamplingWidth;
  return *this;
}

// 设置当前对象的 subsamplingDepth_, subsamplingHeight_ 和 subsamplingWidth_ 属性为指定值，并返回当前对象的引用
inline ConvolutionOperatorTester& subsampling(
    uint32_t subsamplingDepth,
    uint32_t subsamplingHeight,
    uint32_t subsamplingWidth) {
  // 断言 subsamplingDepth 至少为 1
  assert(subsamplingDepth >= 1);
  // 设置 subsamplingDepth_ 属性，并调用前一个 subsampling 方法设置 subsamplingHeight_ 和 subsamplingWidth_ 属性
  this->subsamplingDepth_ = subsamplingDepth;
  return this->subsampling(subsamplingHeight, subsamplingWidth);
}

// 设置当前对象的 subsamplingDepth_ 属性为指定值，并返回当前对象的引用
inline ConvolutionOperatorTester& subsamplingDepth(
    uint32_t subsamplingDepth) {
  // 断言 subsamplingDepth 至少为 1
  assert(subsamplingDepth >= 1);
  // 设置 subsamplingDepth_ 属性
  this->subsamplingDepth_ = subsamplingDepth;
  return *this;
}

// 返回当前对象的 subsamplingDepth_ 属性值
inline uint32_t subsamplingDepth() const {
  // 返回当前对象的 subsamplingDepth_ 成员变量的值
  return this->subsamplingDepth_;
}

// 设置 subsamplingHeight_ 成员变量，并返回当前对象的引用
inline ConvolutionOperatorTester& subsamplingHeight(
    uint32_t subsamplingHeight) {
  // 断言 subsamplingHeight 大于等于 1
  assert(subsamplingHeight >= 1);
  this->subsamplingHeight_ = subsamplingHeight;
  return *this;
}

// 返回当前对象的 subsamplingHeight_ 成员变量的值
inline uint32_t subsamplingHeight() const {
  return this->subsamplingHeight_;
}

// 设置 subsamplingWidth_ 成员变量，并返回当前对象的引用
inline ConvolutionOperatorTester& subsamplingWidth(
    uint32_t subsamplingWidth) {
  // 断言 subsamplingWidth 大于等于 1
  assert(subsamplingWidth >= 1);
  this->subsamplingWidth_ = subsamplingWidth;
  return *this;
}

// 返回当前对象的 subsamplingWidth_ 成员变量的值
inline uint32_t subsamplingWidth() const {
  return this->subsamplingWidth_;
}

// 设置 inputPixelStride_ 成员变量，并返回当前对象的引用
inline ConvolutionOperatorTester& inputPixelStride(size_t inputPixelStride) {
  // 断言 inputPixelStride 大于等于 1
  assert(inputPixelStride >= 1);
  this->inputPixelStride_ = inputPixelStride;
  return *this;
}

// 返回当前对象的 inputPixelStride_ 成员变量的值
inline size_t inputPixelStride() const {
  // 如果 inputPixelStride_ 为 0，则返回计算值 groupInputChannels() * groups()
  if (this->inputPixelStride_ == 0) {
    return groupInputChannels() * groups();
  } else {
    // 否则，断言 inputPixelStride_ 大于等于 groupInputChannels() * groups()
    assert(this->inputPixelStride_ >= groupInputChannels() * groups());
    return this->inputPixelStride_;
  }
}

// 设置 outputPixelStride_ 成员变量，并返回当前对象的引用
inline ConvolutionOperatorTester& outputPixelStride(
    size_t outputPixelStride) {
  // 断言 outputPixelStride 大于等于 1
  assert(outputPixelStride >= 1);
  this->outputPixelStride_ = outputPixelStride;
  return *this;
}

// 返回当前对象的 outputPixelStride_ 成员变量的值
inline size_t outputPixelStride() const {
  // 如果 outputPixelStride_ 为 0，则返回计算值 groupOutputChannels() * groups()
  if (this->outputPixelStride_ == 0) {
    return groupOutputChannels() * groups();
  } else {
    // 否则，断言 outputPixelStride_ 大于等于 groupOutputChannels() * groups()
    assert(this->outputPixelStride_ >= groupOutputChannels() * groups());
    return this->outputPixelStride_;
  }
}

// 返回经过 dilationDepth() 计算的 dilatedKernelDepth_
inline uint32_t dilatedKernelDepth() const {
  return (kernelDepth() - 1) * dilationDepth() + 1;
}

// 返回经过 dilationHeight() 计算的 dilatedKernelHeight_
inline uint32_t dilatedKernelHeight() const {
  return (kernelHeight() - 1) * dilationHeight() + 1;
}

// 返回经过 dilationWidth() 计算的 dilatedKernelWidth_
inline uint32_t dilatedKernelWidth() const {
  return (kernelWidth() - 1) * dilationWidth() + 1;
}

// 返回输出深度 outputDepth_ 的计算值
inline size_t outputDepth() const {
  const size_t paddedInputDepth = inputDepth() + paddingDepth() * 2;
  if (paddedInputDepth <= dilatedKernelDepth()) {
    return 1;
  } else {
    return (paddedInputDepth - dilatedKernelDepth()) / subsamplingDepth() + 1;
  }
}

// 返回输出高度 outputHeight_ 的计算值
inline size_t outputHeight() const {
  const size_t paddedInputHeight = inputHeight() + paddingHeight() * 2;
  if (paddedInputHeight <= dilatedKernelHeight()) {
    return 1;
  } else {
    return (paddedInputHeight - dilatedKernelHeight()) / subsamplingHeight() +
        1;
  }
}

// 返回输出宽度 outputWidth_ 的计算值
inline size_t outputWidth() const {
  const size_t paddedInputWidth = inputWidth() + paddingWidth() * 2;
  if (paddedInputWidth <= dilatedKernelWidth()) {
    return 1;
  } else {
    return (paddedInputWidth - dilatedKernelWidth()) / subsamplingWidth() + 1;
  }
}

// 设置 qmin_ 成员变量，并返回当前对象的引用
inline ConvolutionOperatorTester& qmin(uint8_t qmin) {
  this->qmin_ = qmin;
  return *this;
}

// 返回当前对象的 qmin_ 成员变量的值
inline uint8_t qmin() const {
  return this->qmin_;
}

// 设置 qmax_ 成员变量，并返回当前对象的引用
inline ConvolutionOperatorTester& qmax(uint8_t qmax) {
  this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline ConvolutionOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void testQ8(const Mode mode = Mode::Static) const {
    // 生成随机设备对象
    std::random_device randomDevice;
    // 使用随机设备对象创建 Mersenne Twister 伪随机数生成器
    auto rng = std::mt19937(randomDevice());
    // 创建绑定到均匀分布的函数对象，生成范围为 [-10000, 10000] 的 int32_t 数字
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    // 创建绑定到均匀分布的函数对象，生成 uint8_t 数字
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    // 创建绑定到均匀分布的函数对象，生成范围为 [1, 5) 的 float 数字
    auto f32rng =
        std::bind(std::uniform_real_distribution<float>(1, 5), rng);

    // 创建输入向量，其大小与输入数据的维度相关
    std::vector<uint8_t> input(
        batchSize() *
            ((inputDepth() * inputHeight() * inputWidth() - 1) *
                 inputPixelStride() +
             groups() * groupInputChannels()) +
        8);
    // 创建内核向量，其大小与内核数据的维度相关
    std::vector<uint8_t> kernel(
        groups() * groupOutputChannels() * kernelHeight() * kernelDepth() *
        kernelWidth() * groupInputChannels());
    // 创建偏置向量，其大小与输出数据的维度相关
    std::vector<int32_t> bias(groups() * groupOutputChannels());
    // 创建输出向量，其大小与输出数据的维度相关
    std::vector<uint8_t> output(
        batchSize() *
        ((outputDepth() * outputHeight() * outputWidth() - 1) *
             outputPixelStride() +
         groups() * groupOutputChannels()));
    // 创建累加器向量，其大小与输出数据的维度相关
    std::vector<int32_t> accumulators(
        batchSize() * outputDepth() * outputHeight() * outputWidth() *
        groups() * groupOutputChannels());

    // 指向输入数据的指针，偏移 8 个字节
    const uint8_t* inputPtr = input.data() + 8;
    // 输入数据的零点偏移量为 127
    const uint8_t inputZeroPoint = 127;
    // 计算填充后的零点个数，确保为 SSE/ARM 内核所需的最小公倍数
    size_t num_zero_points_padded =
      (groups() * groupOutputChannels() + 8);
    // 创建填充后的内核零点向量，其大小为 num_zero_points_padded，所有元素为 127
    std::vector<uint8_t> kernelZeroPoints(num_zero_points_padded, 127);

    }
  }

 private:
  uint32_t paddingDepth_{0};
  uint32_t paddingHeight_{0};
  uint32_t paddingWidth_{0};
  size_t inputDepth_{1};
  size_t inputHeight_{1};
  size_t inputWidth_{1};
  uint32_t groups_{1};
  size_t groupInputChannels_{1};
  size_t inputPixelStride_{0};
  size_t groupOutputChannels_{1};
  size_t outputPixelStride_{0};
  size_t batchSize_{1};
  uint32_t kernelDepth_{1};
  uint32_t kernelHeight_{1};
  uint32_t kernelWidth_{1};
  uint32_t dilationDepth_{1};
  uint32_t dilationHeight_{1};
  uint32_t dilationWidth_{1};
  uint32_t subsamplingDepth_{1};
  uint32_t subsamplingHeight_{1};
  uint32_t subsamplingWidth_{1};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
  bool per_channel_{false};
  size_t dimensionality_{2}; // 2 or 3
};



# 这行代码是一个单独的分号，通常用于语句结束，但在这里单独使用可能是一个错误或者是未完成的代码段。
# 在程序中，单独的分号可能被误用，应当注意是否需要删除或者补充其他代码来正确编写。
```