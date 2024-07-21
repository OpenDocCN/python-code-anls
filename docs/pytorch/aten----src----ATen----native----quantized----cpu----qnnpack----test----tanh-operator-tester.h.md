# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\tanh-operator-tester.h`

```
/*
 * 版权所有（C）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 本源代码根目录中的LICENSE文件中包含的BSD风格许可证适用于此源代码。
 */

#pragma once

#include <algorithm>   // 引入算法标准库，用于一些算法操作
#include <cassert>     // 引入断言标准库，用于条件检查
#include <cmath>       // 引入数学标准库，用于数学计算
#include <cstddef>     // 引入cstddef标准库，用于定义size_t和nullptr
#include <cstdlib>     // 引入cstdlib标准库，用于通用函数
#include <functional>  // 引入函数对象标准库，用于函数封装
#include <random>      // 引入随机数生成标准库，用于随机数生成
#include <vector>      // 引入向量容器标准库，用于存储动态数组

#include <pytorch_qnnpack.h>  // 引入pytorch的QNNPACK库

class TanHOperatorTester {
 public:
  // 设置通道数，并返回当前对象的引用
  inline TanHOperatorTester& channels(size_t channels) {
    assert(channels != 0);   // 断言通道数不为0
    this->channels_ = channels;
    return *this;
  }

  // 返回当前对象的通道数
  inline size_t channels() const {
    return this->channels_;
  }

  // 设置输入步长，并返回当前对象的引用
  inline TanHOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);   // 断言输入步长不为0
    this->inputStride_ = inputStride;
    return *this;
  }

  // 返回当前对象的输入步长
  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->inputStride_ >= this->channels_);   // 断言输入步长大于等于通道数
      return this->inputStride_;
    }
  }

  // 设置输出步长，并返回当前对象的引用
  inline TanHOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);   // 断言输出步长不为0
    this->outputStride_ = outputStride;
    return *this;
  }

  // 返回当前对象的输出步长
  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->outputStride_ >= this->channels_);   // 断言输出步长大于等于通道数
      return this->outputStride_;
    }
  }

  // 设置批次大小，并返回当前对象的引用
  inline TanHOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  // 返回当前对象的批次大小
  inline size_t batchSize() const {
    return this->batchSize_;
  }

  // 设置输入缩放因子，并返回当前对象的引用
  inline TanHOperatorTester& inputScale(float inputScale) {
    assert(inputScale > 0.0f);   // 断言输入缩放因子大于0
    assert(std::isnormal(inputScale));   // 断言输入缩放因子为正常数
    this->inputScale_ = inputScale;
    return *this;
  }

  // 返回当前对象的输入缩放因子
  inline float inputScale() const {
    return this->inputScale_;
  }

  // 设置输入零点，并返回当前对象的引用
  inline TanHOperatorTester& inputZeroPoint(uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  // 返回当前对象的输入零点
  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  // 返回输出缩放因子（固定为1/128）
  inline float outputScale() const {
    return 1.0f / 128.0f;
  }

  // 返回输出零点（固定为128）
  inline uint8_t outputZeroPoint() const {
    return 128;
  }

  // 设置Q值的最小值，并返回当前对象的引用
  inline TanHOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 返回当前对象的Q值最小值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置Q值的最大值，并返回当前对象的引用
  inline TanHOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回当前对象的Q值最大值
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置迭代次数，并返回当前对象的引用
  inline TanHOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前对象的迭代次数
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 进行Q8类型的测试
  void testQ8() const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 计算输入向量的大小并创建对应大小的向量
    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    // 创建一个具有所需大小的输出向量，输出的大小由批次大小、输出步长和通道数决定
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + channels());
    // 创建一个参考输出向量，其大小由批次大小和通道数决定
    std::vector<float> outputRef(batchSize() * channels());

    // 循环执行指定次数的迭代过程
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 使用随机数生成器 u8rng 生成输入向量 input
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      // 将输出向量 output 填充为固定值 0xA5

      /* 计算参考结果 */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          // 计算输入值 x，进行量化
          const float x = inputScale() *
              (int32_t(input[i * inputStride() + c]) -
               int32_t(inputZeroPoint()));
          // 计算 tanh 函数值 tanhX
          const float tanhX = tanh(x);
          // 将 tanh 函数值进行缩放得到 scaledTanHX
          const float scaledTanHX = tanhX / outputScale();
          // 对输出进行量化调整
          float y = scaledTanHX;
          y = std::min<float>(y, int32_t(qmax()) - int32_t(outputZeroPoint()));
          y = std::max<float>(y, int32_t(qmin()) - int32_t(outputZeroPoint()));
          // 将调整后的值保存到 outputRef 中
          outputRef[i * channels() + c] = y + int32_t(outputZeroPoint());
        }
      }

      /* 创建、设置、运行和销毁 TanH 运算符 */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      // 初始化 TanH 运算符指针
      pytorch_qnnp_operator_t tanhOp = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          // 创建 Q8 格式的 TanH 运算符
          pytorch_qnnp_create_tanh_nc_q8(
              channels(),
              inputZeroPoint(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              qmin(),
              qmax(),
              0,
              &tanhOp));
      ASSERT_NE(nullptr, tanhOp);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          // 设置 TanH 运算符的参数和输入输出数据
          pytorch_qnnp_setup_tanh_nc_q8(
              tanhOp,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          // 执行 TanH 运算符
          pytorch_qnnp_run_operator(tanhOp, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(tanhOp));
      tanhOp = nullptr;

      /* 验证结果的准确性 */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          // 使用容忍度 0.6f 检验输出结果是否接近于参考结果
          ASSERT_NEAR(
              float(int32_t(output[i * outputStride() + c])),
              outputRef[i * channels() + c],
              0.6f);
        }
      }
    }
};
```