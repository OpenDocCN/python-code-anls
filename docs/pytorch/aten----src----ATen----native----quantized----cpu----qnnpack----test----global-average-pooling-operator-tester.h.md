# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\global-average-pooling-operator-tester.h`

```
/*
 * 版权所有（c）Facebook，Inc.及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的LICENSE文件中找到的BSD风格许可证进行许可。
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <pytorch_qnnpack.h>

// 全局平均池化运算符测试器类
class GlobalAveragePoolingOperatorTester {
 public:
  // 设置通道数，并返回当前对象的引用
  inline GlobalAveragePoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  // 返回当前设置的通道数
  inline size_t channels() const {
    return this->channels_;
  }

  // 设置宽度，并返回当前对象的引用
  inline GlobalAveragePoolingOperatorTester& width(size_t width) {
    assert(width != 0);
    this->width_ = width;
    return *this;
  }

  // 返回当前设置的宽度
  inline size_t width() const {
    return this->width_;
  }

  // 设置输入步长，并返回当前对象的引用
  inline GlobalAveragePoolingOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  // 返回当前设置的输入步长，若未设置则默认为通道数
  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return channels();
    } else {
      assert(this->inputStride_ >= channels());
      return this->inputStride_;
    }
  }

  // 设置输出步长，并返回当前对象的引用
  inline GlobalAveragePoolingOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  // 返回当前设置的输出步长，若未设置则默认为通道数
  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return channels();
    } else {
      assert(this->outputStride_ >= channels());
      return this->outputStride_;
    }
  }

  // 设置批处理大小，并返回当前对象的引用
  inline GlobalAveragePoolingOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  // 返回当前设置的批处理大小
  inline size_t batchSize() const {
    return this->batchSize_;
  }

  // 设置输入比例，并返回当前对象的引用
  inline GlobalAveragePoolingOperatorTester& inputScale(float inputScale) {
    assert(inputScale > 0.0f);
    assert(std::isnormal(inputScale));
    this->inputScale_ = inputScale;
    return *this;
  }

  // 返回当前设置的输入比例
  inline float inputScale() const {
    return this->inputScale_;
  }

  // 设置输入零点，并返回当前对象的引用
  inline GlobalAveragePoolingOperatorTester& inputZeroPoint(
      uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  // 返回当前设置的输入零点
  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  // 设置输出比例，并返回当前对象的引用
  inline GlobalAveragePoolingOperatorTester& outputScale(float outputScale) {
    assert(outputScale > 0.0f);
    assert(std::isnormal(outputScale));
    this->outputScale_ = outputScale;
    return *this;
  }

  // 返回当前设置的输出比例
  inline float outputScale() const {
    return this->outputScale_;
  }

  // 设置输出零点，并返回当前对象的引用
  inline GlobalAveragePoolingOperatorTester& outputZeroPoint(
      uint8_t outputZeroPoint) {
    this->outputZeroPoint_ = outputZeroPoint;
    return *this;
  }

  // 返回当前设置的输出零点
  inline uint8_t outputZeroPoint() const {
    return this->outputZeroPoint_;
  }

  // 设置输出最小值，并返回当前对象的引用
  inline GlobalAveragePoolingOperatorTester& outputMin(uint8_t outputMin) {
    this->outputMin_ = outputMin;
    return *this;
  }
    // 返回当前对象的引用，用于链式调用
    return *this;
  }

  // 返回成员变量 outputMin_ 的值
  inline uint8_t outputMin() const {
    return this->outputMin_;
  }

  // 设置成员变量 outputMax_ 的值，并返回当前对象的引用，支持链式调用
  inline GlobalAveragePoolingOperatorTester& outputMax(uint8_t outputMax) {
    this->outputMax_ = outputMax;
    return *this;
  }

  // 返回成员变量 outputMax_ 的值
  inline uint8_t outputMax() const {
    return this->outputMax_;
  }

  // 设置成员变量 iterations_ 的值，并返回当前对象的引用，支持链式调用
  inline GlobalAveragePoolingOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回成员变量 iterations_ 的值
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 对 Q8 进行测试
  void testQ8() const {
    // 随机数设备
    std::random_device randomDevice;
    // 随机数生成器
    auto rng = std::mt19937(randomDevice());
    // 生成 uint8_t 类型的均匀分布随机数的函数对象
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建输入向量，计算所需大小
    std::vector<uint8_t> input(
        (batchSize() * width() - 1) * inputStride() + channels());
    // 创建输出向量，计算所需大小
    std::vector<uint8_t> output(batchSize() * outputStride());
    // 创建参考输出向量，计算所需大小
    std::vector<float> outputRef(batchSize() * channels());


这些注释描述了每个函数的目的和作用，以及在 `testQ8()` 函数中使用的随机数生成和向量创建过程。
    // 迭代执行指定次数的操作
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 使用随机数生成器填充输入数据
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      // 将输出数据全部设置为固定值 0xA5
      std::fill(output.begin(), output.end(), 0xA5);

      /* 计算参考结果 */
      // 计算缩放比例，用于从输入数据到输出数据的转换
      const double scale =
          double(inputScale()) / (double(width()) * double(outputScale()));
      // 遍历每个批次中的每个通道
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          // 初始化累加器
          double acc = 0.0f;
          // 计算累加器的值，考虑输入数据的偏移量
          for (size_t k = 0; k < width(); k++) {
            acc += double(
                int32_t(input[(i * width() + k) * inputStride() + j]) -
                int32_t(inputZeroPoint()));
          }
          // 计算输出的参考值
          outputRef[i * channels() + j] =
              float(acc * scale + double(outputZeroPoint()));
          // 确保输出值不超过指定的最大值和最小值
          outputRef[i * channels() + j] = std::min<float>(
              outputRef[i * channels() + j], float(outputMax()));
          outputRef[i * channels() + j] = std::max<float>(
              outputRef[i * channels() + j], float(outputMin()));
        }
      }

      /* 创建、设置、运行和销毁 Add 运算符 */
      // 初始化 QNNPACK 库
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      // 创建全局平均池化运算符
      pytorch_qnnp_operator_t globalAveragePoolingOp = nullptr;

      // 创建全局平均池化运算符，并设置参数
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_global_average_pooling_nwc_q8(
              channels(),
              inputZeroPoint(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              outputMin(),
              outputMax(),
              0,
              &globalAveragePoolingOp));
      ASSERT_NE(nullptr, globalAveragePoolingOp);

      // 设置全局平均池化运算符的输入和输出数据
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_global_average_pooling_nwc_q8(
              globalAveragePoolingOp,
              batchSize(),
              width(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      // 执行全局平均池化运算符
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(
              globalAveragePoolingOp, nullptr /* thread pool */));

      // 销毁全局平均池化运算符
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(globalAveragePoolingOp));
      globalAveragePoolingOp = nullptr;

      /* 验证结果 */
      // 验证输出数据是否在指定范围内
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(
              uint32_t(output[i * outputStride() + c]), uint32_t(outputMax()));
          ASSERT_GE(
              uint32_t(output[i * outputStride() + c]), uint32_t(outputMin()));
          // 检查输出数据与参考结果的近似程度
          ASSERT_NEAR(
              float(int32_t(output[i * outputStride() + c])),
              outputRef[i * channels() + c],
              0.80f)
              << "in batch index " << i << ", channel " << c;
        }
      }
  }

这是一个类的私有部分结束标志，表示以下是私有成员变量和方法的定义。


  }

类的私有部分的最后一个成员变量定义结束。


 private:

声明以下的成员变量和方法为私有，外部无法直接访问。


  size_t batchSize_{1};

私有成员变量 `batchSize_`，表示批处理大小，初始值为 `1`。


  size_t width_{1};

私有成员变量 `width_`，表示数据的宽度，初始值为 `1`。


  size_t channels_{1};

私有成员变量 `channels_`，表示数据的通道数，初始值为 `1`。


  size_t inputStride_{0};

私有成员变量 `inputStride_`，表示输入数据的跨度，初始值为 `0`。


  size_t outputStride_{0};

私有成员变量 `outputStride_`，表示输出数据的跨度，初始值为 `0`。


  float inputScale_{1.0f};

私有成员变量 `inputScale_`，表示输入数据的缩放比例，初始值为 `1.0`。


  float outputScale_{1.0f};

私有成员变量 `outputScale_`，表示输出数据的缩放比例，初始值为 `1.0`。


  uint8_t inputZeroPoint_{121};

私有成员变量 `inputZeroPoint_`，表示输入数据的零点偏移，初始值为 `121`。


  uint8_t outputZeroPoint_{133};

私有成员变量 `outputZeroPoint_`，表示输出数据的零点偏移，初始值为 `133`。


  uint8_t outputMin_{0};

私有成员变量 `outputMin_`，表示输出数据的最小值，初始值为 `0`。


  uint8_t outputMax_{255};

私有成员变量 `outputMax_`，表示输出数据的最大值，初始值为 `255`。


  size_t iterations_{1};

私有成员变量 `iterations_`，表示迭代次数，初始值为 `1`。
};


注释：


# 这行代码是一个语法错误，"}"是无法单独出现的，它通常需要与 "{" 成对出现，
# 在这个代码片段中，"}"可能是代码块的结尾标志，但缺少了对应的 "{"，
# 这会导致程序在执行时出现语法错误并停止运行。
```