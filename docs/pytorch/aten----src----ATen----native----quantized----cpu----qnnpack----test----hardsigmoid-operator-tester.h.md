# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\hardsigmoid-operator-tester.h`

```
/*
 * 版权所有 (c) Facebook, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的 LICENSE 文件中所述的 BSD 样式许可证进行许可。
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

// 定义 HardsigmoidOperatorTester 类，用于测试硬阈值化操作符
class HardsigmoidOperatorTester {
 public:
  // 设置通道数，并进行断言检查确保通道数不为零
  inline HardsigmoidOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  // 返回当前设置的通道数
  inline size_t channels() const {
    return this->channels_;
  }

  // 设置输入步长，并进行断言检查确保步长不为零
  inline HardsigmoidOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  // 返回当前设置的输入步长，如果步长为零则返回通道数
  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->inputStride_ >= this->channels_);
      return this->inputStride_;
    }
  }

  // 设置输出步长，并进行断言检查确保步长不为零
  inline HardsigmoidOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  // 返回当前设置的输出步长，如果步长为零则返回通道数
  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->outputStride_ >= this->channels_);
      return this->outputStride_;
    }
  }

  // 设置批次大小
  inline HardsigmoidOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  // 返回当前设置的批次大小
  inline size_t batchSize() const {
    return this->batchSize_;
  }

  // 设置输入缩放比例，并进行断言检查确保大于零且为正常数值
  inline HardsigmoidOperatorTester& inputScale(float inputScale) {
    assert(inputScale > 0.0f);
    assert(std::isnormal(inputScale));
    this->inputScale_ = inputScale;
    return *this;
  }

  // 返回当前设置的输入缩放比例
  inline float inputScale() const {
    return this->inputScale_;
  }

  // 设置输入零点
  inline HardsigmoidOperatorTester& inputZeroPoint(uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  // 返回当前设置的输入零点
  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  // 返回输出缩放比例
  inline float outputScale() const {
    return this->outputScale_;
  }

  // 返回输出零点
  inline uint8_t outputZeroPoint() const {
    return this->outputZeroPoint_;
  }

  // 设置量化的最小值
  inline HardsigmoidOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 返回当前设置的量化最小值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置量化的最大值
  inline HardsigmoidOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回当前设置的量化最大值
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置迭代次数
  inline HardsigmoidOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前设置的迭代次数
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 执行 Q8 版本的测试
  void testQ8() const {
    // 使用随机设备创建随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    // 创建一个生成均匀分布的随机数的函数对象
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    // 创建输入向量，大小为 (batchSize() - 1) * inputStride() + channels()
    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    
    // 创建输出向量，大小为 (batchSize() - 1) * outputStride() + channels()
    std::vector<uint8_t> output((batchSize() - 1) * outputStride() + channels());
    
    // 创建参考输出向量，大小为 batchSize() * channels()
    std::vector<float> outputRef(batchSize() * channels());

    // 迭代执行指定次数的操作
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      
      // 使用指定的随机数生成器填充输入向量
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      
      // 将输出向量填充为固定值 0xA5
      std::fill(output.begin(), output.end(), 0xA5);

      /* 计算参考结果 */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          // 计算输入的浮点数表示值
          const float x = inputScale() *
              (int32_t(input[i * inputStride() + c]) -
               int32_t(inputZeroPoint()));
          
          // 计算硬sigmoid函数的输出值
          const float hardsigmoidX =
            std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
          
          // 根据输出量化参数进行缩放
          const float scaledHardsigmoidX = hardsigmoidX / outputScale();
          
          // 将缩放后的值约束在输出的最大最小量化值范围内
          float y = scaledHardsigmoidX;
          y = std::min<float>(y, int32_t(qmax()) - int32_t(outputZeroPoint()));
          y = std::max<float>(y, int32_t(qmin()) - int32_t(outputZeroPoint()));
          
          // 将最终结果存储到参考输出向量中
          outputRef[i * channels() + c] = y + int32_t(outputZeroPoint());
        }
      }

      /* 创建、设置、执行并销毁硬sigmoid算子 */
      // 初始化 QNNPACK 库
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t hardsigmoidOp = nullptr;
      
      // 创建 QNNPACK 中的硬sigmoid算子
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_hardsigmoid_nc_q8(
              channels(),
              inputZeroPoint(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              qmin(),
              qmax(),
              0,
              &hardsigmoidOp));
      ASSERT_NE(nullptr, hardsigmoidOp);

      // 设置硬sigmoid算子的输入、输出及其它参数
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_hardsigmoid_nc_q8(
              hardsigmoidOp,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      // 执行硬sigmoid算子
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(hardsigmoidOp, nullptr /* thread pool */));

      // 删除硬sigmoid算子
      ASSERT_EQ(
          pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(hardsigmoidOp));
      hardsigmoidOp = nullptr;

      /* 验证结果 */
      // 检查计算结果是否接近预期值，容差为 0.6f
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              float(int32_t(output[i * outputStride() + c])),
              outputRef[i * channels() + c],
              0.6f);
        }
      }
    }
};


注释：


# 这行代码似乎是一个语法错误或不完整的语句，在正常情况下应该是一段 JavaScript 或类似的代码块的结尾。
# 缺少前面的代码上下文，无法确定其具体作用或用途。
```