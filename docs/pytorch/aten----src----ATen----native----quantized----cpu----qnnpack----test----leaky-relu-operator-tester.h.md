# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\leaky-relu-operator-tester.h`

```
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据位于源树根目录中的LICENSE文件中的BSD风格许可证进行许可。
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

class LeakyReLUOperatorTester {
 public:
  // 设置测试的通道数，并断言通道数不为0
  inline LeakyReLUOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  // 返回当前设置的通道数
  inline size_t channels() const {
    return this->channels_;
  }

  // 设置输入步长，并断言步长不为0
  inline LeakyReLUOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  // 返回当前设置的输入步长，若为0则返回通道数
  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->inputStride_ >= this->channels_);
      return this->inputStride_;
    }
  }

  // 设置输出步长，并断言步长不为0
  inline LeakyReLUOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  // 返回当前设置的输出步长，若为0则返回通道数
  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->outputStride_ >= this->channels_);
      return this->outputStride_;
    }
  }

  // 设置批次大小
  inline LeakyReLUOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  // 返回当前设置的批次大小
  inline size_t batchSize() const {
    return this->batchSize_;
  }

  // 设置负斜率，并断言在区间 (0, 1) 内
  inline LeakyReLUOperatorTester& negativeSlope(float negativeSlope) {
    assert(negativeSlope > 0.0f);
    assert(negativeSlope < 1.0f);
    this->negativeSlope_ = negativeSlope;
    return *this;
  }

  // 返回当前设置的负斜率
  inline float negativeSlope() const {
    return this->negativeSlope_;
  }

  // 设置输入数据的缩放因子，并断言大于0且正常数值
  inline LeakyReLUOperatorTester& inputScale(float inputScale) {
    assert(inputScale > 0.0f);
    assert(std::isnormal(inputScale));
    this->inputScale_ = inputScale;
    return *this;
  }

  // 返回当前设置的输入数据缩放因子
  inline float inputScale() const {
    return this->inputScale_;
  }

  // 设置输入数据的零点偏移
  inline LeakyReLUOperatorTester& inputZeroPoint(uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  // 返回当前设置的输入数据零点偏移
  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  // 设置输出数据的缩放因子，并断言大于0且正常数值
  inline LeakyReLUOperatorTester& outputScale(float outputScale) {
    assert(outputScale > 0.0f);
    assert(std::isnormal(outputScale));
    this->outputScale_ = outputScale;
    return *this;
  }

  // 返回当前设置的输出数据缩放因子
  inline float outputScale() const {
    return this->outputScale_;
  }

  // 设置输出数据的零点偏移
  inline LeakyReLUOperatorTester& outputZeroPoint(uint8_t outputZeroPoint) {
    this->outputZeroPoint_ = outputZeroPoint;
    return *this;
  }

  // 返回当前设置的输出数据零点偏移
  inline uint8_t outputZeroPoint() const {
    return this->outputZeroPoint_;
  }

  // 设置量化的最小值
  inline LeakyReLUOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    // 返回当前设置的量化最小值
    return *this;
  }


这段代码定义了一个 `LeakyReLUOperatorTester` 类，用于测试 Leaky ReLU 操作符的各种参数配置。
    // 返回当前对象的引用，用于链式调用
    return *this;
  }

  // 返回当前对象的 qmin_ 成员变量值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置 qmax_ 成员变量的值，并返回当前对象的引用，用于链式调用
  inline LeakyReLUOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回当前对象的 qmax_ 成员变量值
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置 iterations_ 成员变量的值，并返回当前对象的引用，用于链式调用
  inline LeakyReLUOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前对象的 iterations_ 成员变量值
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 执行 Q8 测试函数，生成随机数并初始化输入、输出向量
  void testQ8() const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 初始化输入向量，计算大小基于 batchSize 和 inputStride
    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    // 初始化输出向量，计算大小基于 batchSize 和 outputStride
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + channels());
    // 初始化参考输出向量，计算大小基于 batchSize 和 channels
    std::vector<float> outputRef(batchSize() * channels());
    // 对每个迭代执行以下操作，迭代次数由 iterations() 函数返回
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 使用随机数生成器 u8rng 生成随机输入数据
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      // 将输出数据初始化为 0xA5
      std::fill(output.begin(), output.end(), 0xA5);

      /* 计算参考结果 */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          // 计算输入值 x，应用输入量化参数
          const float x = inputScale() *
              (int32_t(input[i * inputStride() + c]) -
               int32_t(inputZeroPoint()));
          // 应用 LeakyReLU 激活函数
          float y = (x < 0.0f ? x * negativeSlope() : x) / outputScale();
          // 限制输出值的范围在 [qmin(), qmax()] 内
          y = std::min<float>(y, int32_t(qmax()) - int32_t(outputZeroPoint()));
          y = std::max<float>(y, int32_t(qmin()) - int32_t(outputZeroPoint()));
          // 计算最终输出值并保存在 outputRef 中
          outputRef[i * channels() + c] = y + float(int32_t(outputZeroPoint()));
        }
      }

      /* 创建、设置、运行和销毁 LeakyReLU 运算符 */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t leakyReLUOp = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_leaky_relu_nc_q8(
              channels(),
              negativeSlope(),
              inputZeroPoint(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              qmin(),
              qmax(),
              0,
              &leakyReLUOp));
      ASSERT_NE(nullptr, leakyReLUOp);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_leaky_relu_nc_q8(
              leakyReLUOp,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(leakyReLUOp, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(leakyReLUOp));
      leakyReLUOp = nullptr;

      /* 验证结果的正确性 */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          // 使用容许误差 0.6f 验证输出是否与参考输出接近
          ASSERT_NEAR(
              float(int32_t(output[i * outputStride() + c])),
              outputRef[i * channels() + c],
              0.6f);
        }
      }
    }
  }

 private:
  size_t batchSize_{1};             // 批处理大小，默认为 1
  size_t channels_{1};              // 通道数，默认为 1
  size_t inputStride_{0};           // 输入数据步长，默认为 0
  size_t outputStride_{0};          // 输出数据步长，默认为 0
  float negativeSlope_{0.5f};       // LeakyReLU 激活函数的负斜率，默认为 0.5
  float outputScale_{0.75f};        // 输出量化参数的缩放比例，默认为 0.75
  uint8_t outputZeroPoint_{133};    // 输出量化参数的零点，默认为 133
  float inputScale_{1.25f};         // 输入量化参数的缩放比例，默认为 1.25
  uint8_t inputZeroPoint_{121};     // 输入量化参数的零点，默认为 121
  uint8_t qmin_{0};                 // 量化的最小值，默认为 0
  uint8_t qmax_{255};               // 量化的最大值，默认为 255
  size_t iterations_{15};           // 迭代次数，默认为 15
};
```