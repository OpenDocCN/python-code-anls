# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\clamp-operator-tester.h`

```py
/*
 * 版权所有（C）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的LICENSE文件中所述的BSD风格许可证进行许可。
 */

#pragma once

#include <algorithm>     // 包含标准算法库，用于算法操作
#include <cassert>       // 包含断言库，用于运行时检查
#include <cstddef>       // 包含stddef库，定义了多种标准库类型和宏
#include <cstdlib>       // 包含标准库的通用工具函数
#include <functional>    // 包含功能库，用于定义函数对象
#include <random>        // 包含随机数生成库
#include <vector>        // 包含向量库，用于定义动态数组

#include <pytorch_qnnpack.h>  // 包含PyTorch QNNPACK头文件，可能为量化神经网络库

class ClampOperatorTester {
 public:
  // 设置通道数，并进行非零断言检查
  inline ClampOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  // 返回当前设置的通道数
  inline size_t channels() const {
    return this->channels_;
  }

  // 设置输入步幅，并进行非零断言检查
  inline ClampOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  // 返回当前设置的输入步幅，如果未设置，则默认为通道数
  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->inputStride_ >= this->channels_);
      return this->inputStride_;
    }
  }

  // 设置输出步幅，并进行非零断言检查
  inline ClampOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  // 返回当前设置的输出步幅，如果未设置，则默认为通道数
  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->outputStride_ >= this->channels_);
      return this->outputStride_;
    }
  }

  // 设置批处理大小
  inline ClampOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  // 返回当前设置的批处理大小
  inline size_t batchSize() const {
    return this->batchSize_;
  }

  // 设置量化的最小值
  inline ClampOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 返回当前设置的量化最小值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置量化的最大值
  inline ClampOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回当前设置的量化最大值
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置测试迭代次数
  inline ClampOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前设置的测试迭代次数
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 执行U8类型的测试
  void testU8() const {
    // 初始化随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 初始化输入、输出和参考输出向量
    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + channels());
    std::vector<uint8_t> outputRef(batchSize() * channels());
    // 对每次迭代执行以下操作，迭代次数由 iterations() 返回
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 使用 u8rng 生成随机数据填充 input 容器
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      // 用 0xA5 填充 output 容器
      std::fill(output.begin(), output.end(), 0xA5);

      /* 计算参考结果 */
      // 遍历每个 batch 中的每个通道，对 input 数据进行量化至 [qmin(), qmax()] 区间
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          // 获取 input 中的数据 x
          const uint8_t x = input[i * inputStride() + c];
          // 对 x 进行量化至 [qmin(), qmax()] 区间，并存入 outputRef
          const uint8_t y = std::min(std::max(x, qmin()), qmax());
          outputRef[i * channels() + c] = y;
        }
      }

      /* 创建、设置、运行并销毁 Sigmoid 运算符 */
      // 初始化 QNNPACK 库
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t clampOp = nullptr;

      // 创建一个 clamp 运算符，用于将数据量化至 [qmin(), qmax()] 区间
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_clamp_nc_u8(
              channels(), qmin(), qmax(), 0, &clampOp));
      ASSERT_NE(nullptr, clampOp);

      // 设置 clamp 运算符的输入和输出数据以及其他参数
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_clamp_nc_u8(
              clampOp,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      // 执行 clamp 运算符
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(clampOp, nullptr /* thread pool */));

      // 销毁 clamp 运算符
      ASSERT_EQ(
          pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(clampOp));
      clampOp = nullptr;

      /* 验证结果 */
      // 验证 output 是否在 [qmin(), qmax()] 区间内
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(uint32_t(output[i * channels() + c]), uint32_t(qmax()))
              << "at position " << i << ", batch size = " << batchSize()
              << ", channels = " << channels();
          ASSERT_GE(uint32_t(output[i * channels() + c]), uint32_t(qmin()))
              << "at position " << i << ", batch size = " << batchSize()
              << ", channels = " << channels();
          // 检查 output 是否与参考输出 outputRef 相符
          ASSERT_EQ(
              uint32_t(outputRef[i * channels() + c]),
              uint32_t(output[i * outputStride() + c]))
              << "at position " << i << ", batch size = " << batchSize()
              << ", channels = " << channels() << ", qmin = " << qmin()
              << ", qmax = " << qmax();
        }
      }
    }
  }

 private:
  size_t batchSize_{1};      // 批次大小，默认为 1
  size_t channels_{1};        // 通道数，默认为 1
  size_t inputStride_{0};     // 输入数据步长，默认为 0
  size_t outputStride_{0};    // 输出数据步长，默认为 0
  uint8_t qmin_{0};           // 数据量化下限，默认为 0
  uint8_t qmax_{255};         // 数据量化上限，默认为 255
  size_t iterations_{15};     // 迭代次数，默认为 15
};


注释：


# 这行代码似乎是一个单独的代码块结束的标志，但缺少了开始的语句，因此在这里没有实际的作用。
```