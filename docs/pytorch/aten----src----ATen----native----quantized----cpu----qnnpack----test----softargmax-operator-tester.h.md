# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\softargmax-operator-tester.h`

```py
/*
 * 版权声明和许可信息
 * 本源代码使用 BSD 风格许可证授权，许可细节详见源代码根目录下的 LICENSE 文件
 */

#pragma once

#include <algorithm>  // 标准算法库，包含各种算法函数
#include <cassert>    // 提供 assert 断言功能
#include <cmath>      // 数学函数库
#include <cstddef>    // 提供 size_t 和 nullptr_t 定义
#include <cstdlib>    // 标准库的常用函数
#include <functional> // 函数对象相关
#include <random>     // 提供随机数生成器
#include <vector>     // 标准向量容器

#include <pytorch_qnnpack.h> // 引入 PyTorch QNNPACK 头文件

// SoftArgMaxOperatorTester 类定义
class SoftArgMaxOperatorTester {
 public:
  // 设置通道数，返回对象本身的引用
  inline SoftArgMaxOperatorTester& channels(size_t channels) {
    assert(channels != 0);  // 断言通道数不为零
    this->channels_ = channels;
    return *this;
  }

  // 获取通道数
  inline size_t channels() const {
    return this->channels_;
  }

  // 设置输入步长，返回对象本身的引用
  inline SoftArgMaxOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);  // 断言输入步长不为零
    this->inputStride_ = inputStride;
    return *this;
  }

  // 获取输入步长
  inline size_t inputStride() const {
    // 如果输入步长为零，则返回通道数
    if (this->inputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->inputStride_ >= this->channels_);  // 断言输入步长大于等于通道数
      return this->inputStride_;
    }
  }

  // 设置输出步长，返回对象本身的引用
  inline SoftArgMaxOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);  // 断言输出步长不为零
    this->outputStride_ = outputStride;
    return *this;
  }

  // 获取输出步长
  inline size_t outputStride() const {
    // 如果输出步长为零，则返回通道数
    if (this->outputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->outputStride_ >= this->channels_);  // 断言输出步长大于等于通道数
      return this->outputStride_;
    }
  }

  // 设置批处理大小，返回对象本身的引用
  inline SoftArgMaxOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  // 获取批处理大小
  inline size_t batchSize() const {
    return this->batchSize_;
  }

  // 设置输入比例因子，返回对象本身的引用
  inline SoftArgMaxOperatorTester& inputScale(float inputScale) {
    assert(inputScale > 0.0f);      // 断言输入比例因子大于零
    assert(std::isnormal(inputScale)); // 断言输入比例因子是正常数
    this->inputScale_ = inputScale;
    return *this;
  }

  // 获取输入比例因子
  inline float inputScale() const {
    return this->inputScale_;
  }

  // 设置输入零点，返回对象本身的引用
  inline SoftArgMaxOperatorTester& inputZeroPoint(uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  // 获取输入零点
  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  // 获取输出比例因子
  inline float outputScale() const {
    return 1.0f / 256.0f;
  }

  // 获取输出零点
  inline uint8_t outputZeroPoint() const {
    return 0;
  }

  // 设置迭代次数，返回对象本身的引用
  inline SoftArgMaxOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 获取迭代次数
  inline size_t iterations() const {
    return this->iterations_;
  }

  // Q8 测试函数
  void testQ8() const {
    // 创建随机设备和随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建输入、输出和参考输出向量
    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    std::vector<uint8_t> output((batchSize() - 1) * outputStride() + channels());
    std::vector<float> outputRef(batchSize() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 对于每次迭代，执行以下操作
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      // 使用 u8rng 函数对象生成随机输入数据

      std::fill(output.begin(), output.end(), 0xA5);
      // 将输出数据填充为固定值 0xA5

      /* Compute reference results */
      // 计算参考结果
      for (size_t i = 0; i < batchSize(); i++) {
        // 对于每个批次中的数据
        const int32_t maxInput = *std::max_element(
            input.data() + i * inputStride(),
            input.data() + i * inputStride() + channels());
        // 计算当前批次中输入数据的最大值

        float sumExp = 0.0f;
        // 初始化指数和为0
        for (size_t c = 0; c < channels(); c++) {
          // 对于每个通道
          sumExp +=
              exp((int32_t(input[i * inputStride() + c]) - maxInput) *
                  inputScale());
          // 计算指数和，用于归一化计算
        }

        for (size_t c = 0; c < channels(); c++) {
          // 对于每个通道
          outputRef[i * channels() + c] =
              exp((int32_t(input[i * inputStride() + c]) - maxInput) *
                  inputScale()) /
              (sumExp * outputScale());
          // 计算 SoftArgMax 运算的输出值
          
          outputRef[i * channels() + c] =
              std::min(outputRef[i * channels() + c], 255.0f);
          // 将输出值限制在 0 到 255 之间
        }
      }

      /* Create, setup, run, and destroy SoftArgMax operator */
      // 创建、设置、运行和销毁 SoftArgMax 运算符
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      // 初始化 PyTorch QNNPACK 库

      pytorch_qnnp_operator_t softArgMaxOp = nullptr;
      // 声明 SoftArgMax 运算符指针为空

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_softargmax_nc_q8(
              channels(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              0,
              &softArgMaxOp));
      // 创建 SoftArgMax 运算符

      ASSERT_NE(nullptr, softArgMaxOp);
      // 检查 SoftArgMax 运算符是否成功创建

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_softargmax_nc_q8(
              softArgMaxOp,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));
      // 设置 SoftArgMax 运算符的输入和输出数据

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(softArgMaxOp, nullptr /* thread pool */));
      // 运行 SoftArgMax 运算符

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(softArgMaxOp));
      // 删除 SoftArgMax 运算符

      softArgMaxOp = nullptr;
      // 将 SoftArgMax 运算符指针重置为空

      /* Verify results */
      // 验证结果
      for (size_t i = 0; i < batchSize(); i++) {
        // 对于每个批次
        for (size_t c = 0; c < channels(); c++) {
          // 对于每个通道
          ASSERT_NEAR(
              float(int32_t(output[i * outputStride() + c])),
              outputRef[i * channels() + c],
              0.6f);
          // 检查 SoftArgMax 运算后的输出值是否接近参考结果
        }
      }
    }
  }

 private:
  size_t batchSize_{1};
  size_t channels_{1};
  size_t inputStride_{0};
  size_t outputStride_{0};
  float inputScale_{0.176080093};
  uint8_t inputZeroPoint_{121};
  size_t iterations_{15};
};


注释：

# 这行代码是一个语法错误，缺少了完整的代码逻辑和上下文，无法执行或理解其作用。
```