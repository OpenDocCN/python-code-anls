# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\add-operator-tester.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// 包含必要的标准库头文件
#include <algorithm>    // 标准算法库
#include <cmath>        // 数学函数库
#include <cstddef>      // C标准库定义的基本数据类型
#include <cstdlib>      // C标准库的常用函数
#include <functional>   // 函数对象库
#include <random>       // 随机数生成库
#include <vector>       // 动态数组库

// 包含 PyTorch QNNPACK 头文件
#include <pytorch_qnnpack.h>

// AddOperatorTester 类定义
class AddOperatorTester {
 public:
  // 设置通道数，返回当前对象的引用
  inline AddOperatorTester& channels(size_t channels) {
    // 断言通道数不为0
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  // 返回当前通道数
  inline size_t channels() const {
    return this->channels_;
  }

  // 设置输入张量 a 的步长，返回当前对象的引用
  inline AddOperatorTester& aStride(size_t aStride) {
    // 断言步长不为0
    assert(aStride != 0);
    this->aStride_ = aStride;
    return *this;
  }

  // 返回输入张量 a 的步长
  inline size_t aStride() const {
    // 如果步长为0，则返回通道数
    if (this->aStride_ == 0) {
      return this->channels_;
    } else {
      // 否则，断言步长不小于通道数，并返回步长
      assert(this->aStride_ >= this->channels_);
      return this->aStride_;
    }
  }

  // 设置输入张量 b 的步长，返回当前对象的引用
  inline AddOperatorTester& bStride(size_t bStride) {
    // 断言步长不为0
    assert(bStride != 0);
    this->bStride_ = bStride;
    return *this;
  }

  // 返回输入张量 b 的步长
  inline size_t bStride() const {
    // 如果步长为0，则返回通道数
    if (this->bStride_ == 0) {
      return this->channels_;
    } else {
      // 否则，断言步长不小于通道数，并返回步长
      assert(this->bStride_ >= this->channels_);
      return this->bStride_;
    }
  }

  // 设置输出张量 y 的步长，返回当前对象的引用
  inline AddOperatorTester& yStride(size_t yStride) {
    // 断言步长不为0
    assert(yStride != 0);
    this->yStride_ = yStride;
    return *this;
  }

  // 返回输出张量 y 的步长
  inline size_t yStride() const {
    // 如果步长为0，则返回通道数
    if (this->yStride_ == 0) {
      return this->channels_;
    } else {
      // 否则，断言步长不小于通道数，并返回步长
      assert(this->yStride_ >= this->channels_);
      return this->yStride_;
    }
  }

  // 设置批量大小，返回当前对象的引用
  inline AddOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  // 返回批量大小
  inline size_t batchSize() const {
    return this->batchSize_;
  }

  // 设置输入张量 a 的缩放因子，返回当前对象的引用
  inline AddOperatorTester& aScale(float aScale) {
    // 断言缩放因子大于0且是正常数
    assert(aScale > 0.0f);
    assert(std::isnormal(aScale));
    this->aScale_ = aScale;
    return *this;
  }

  // 返回输入张量 a 的缩放因子
  inline float aScale() const {
    return this->aScale_;
  }

  // 设置输入张量 a 的零点偏移量，返回当前对象的引用
  inline AddOperatorTester& aZeroPoint(uint8_t aZeroPoint) {
    this->aZeroPoint_ = aZeroPoint;
    return *this;
  }

  // 返回输入张量 a 的零点偏移量
  inline uint8_t aZeroPoint() const {
    return this->aZeroPoint_;
  }

  // 设置输入张量 b 的缩放因子，返回当前对象的引用
  inline AddOperatorTester& bScale(float bScale) {
    // 断言缩放因子大于0且是正常数
    assert(bScale > 0.0f);
    assert(std::isnormal(bScale));
    this->bScale_ = bScale;
    return *this;
  }

  // 返回输入张量 b 的缩放因子
  inline float bScale() const {
    return this->bScale_;
  }

  // 设置输入张量 b 的零点偏移量，返回当前对象的引用
  inline AddOperatorTester& bZeroPoint(uint8_t bZeroPoint) {
    this->bZeroPoint_ = bZeroPoint;
    return *this;
  }

  // 返回输入张量 b 的零点偏移量
  inline uint8_t bZeroPoint() const {
    return this->bZeroPoint_;
  }

  // 设置输出张量 y 的缩放因子，返回当前对象的引用
  inline AddOperatorTester& yScale(float yScale) {
    // 断言缩放因子大于0且是正常数
    assert(yScale > 0.0f);
    assert(std::isnormal(yScale));
    this->yScale_ = yScale;
    return *this;
  }

  // 返回输出张量 y 的缩放因子
  inline float yScale() const {
    return this->yScale_;
  }

  // 设置输出张量 y 的零点偏移量，返回当前对象的引用
  inline AddOperatorTester& yZeroPoint(uint8_t yZeroPoint) {
    this->yZeroPoint_ = yZeroPoint;
    return *this;
  }

  // 返回输出张量 y 的零点偏移量
  inline uint8_t yZeroPoint() const {
    return this->yZeroPoint_;
  }

private:
  // 私有成员变量，用于存储各种参数
  size_t channels_;
  size_t aStride_;
  size_t bStride_;
  size_t yStride_;
  size_t batchSize_;
  float aScale_;
  uint8_t aZeroPoint_;
  float bScale_;
  uint8_t bZeroPoint_;
  float yScale_;
  uint8_t yZeroPoint_;
};
  // 返回当前对象的引用，用于链式调用
  return *this;
}

  // 返回成员变量 yZeroPoint_ 的值
  inline uint8_t yZeroPoint() const {
    return this->yZeroPoint_;
  }

  // 设置成员变量 qmin_ 的值，并返回当前对象的引用，用于链式调用
  inline AddOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 返回成员变量 qmin_ 的值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置成员变量 qmax_ 的值，并返回当前对象的引用，用于链式调用
  inline AddOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回成员变量 qmax_ 的值
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置成员变量 iterations_ 的值，并返回当前对象的引用，用于链式调用
  inline AddOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回成员变量 iterations_ 的值
  inline size_t iterations() const {
    return this->iterations_;
  }

void testQ8() const {
  // 创建随机设备对象
  std::random_device randomDevice;
  // 创建 Mersenne Twister 伪随机数生成器，并用随机设备种子初始化
  auto rng = std::mt19937(randomDevice());
  // 创建生成均匀分布的无符号 8 位整数的函数对象
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  // 创建大小为 (batchSize() - 1) * aStride() + channels() 的无符号 8 位整数向量 a
  std::vector<uint8_t> a((batchSize() - 1) * aStride() + channels());
  // 创建大小为 (batchSize() - 1) * bStride() + channels() 的无符号 8 位整数向量 b
  std::vector<uint8_t> b((batchSize() - 1) * bStride() + channels());
  // 创建大小为 (batchSize() - 1) * yStride() + channels() 的无符号 8 位整数向量 y
  std::vector<uint8_t> y((batchSize() - 1) * yStride() + channels());
  // 创建大小为 batchSize() * channels() 的单精度浮点数向量 yRef
  std::vector<float> yRef(batchSize() * channels());
    // 对每次迭代执行以下操作，迭代次数由 iterations() 返回确定
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 使用 u8rng 函数对象生成随机数填充容器 a
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      // 使用 u8rng 函数对象生成随机数填充容器 b
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      // 将容器 y 全部填充为 0xA5
      std::fill(y.begin(), y.end(), 0xA5);

      // 如果 batchSize() 乘以 channels() 大于 3，则执行以下断言
      if (batchSize() * channels() > 3) {
        // 断言 a 中的最大元素不等于最小元素
        ASSERT_NE(
            *std::max_element(a.cbegin(), a.cend()),
            *std::min_element(a.cbegin(), a.cend()));
        // 断言 b 中的最大元素不等于最小元素
        ASSERT_NE(
            *std::max_element(b.cbegin(), b.cend()),
            *std::min_element(b.cbegin(), b.cend()));
      }

      /* 计算参考结果 */
      // 遍历每个 batch 的数据
      for (size_t i = 0; i < batchSize(); i++) {
        // 遍历每个通道的数据
        for (size_t c = 0; c < channels(); c++) {
          // 使用公式计算 yRef 中的每个元素值
          yRef[i * channels() + c] = float(yZeroPoint()) +
              float(int32_t(a[i * aStride() + c]) - int32_t(aZeroPoint())) *
                  (aScale() / yScale()) +
              float(int32_t(b[i * bStride() + c]) - int32_t(bZeroPoint())) *
                  (bScale() / yScale());
          // 将计算结果限制在 qmax() 和 qmin() 范围内
          yRef[i * channels() + c] =
              std::min<float>(yRef[i * channels() + c], float(qmax()));
          yRef[i * channels() + c] =
              std::max<float>(yRef[i * channels() + c], float(qmin()));
        }
      }

      /* 创建、设置、运行并销毁 Add 运算符 */
      // 初始化 PyTorch QNNPACK
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      // 创建 Add 运算符
      pytorch_qnnp_operator_t add_op = nullptr;
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_add_nc_q8(
              channels(),
              aZeroPoint(),
              aScale(),
              bZeroPoint(),
              bScale(),
              yZeroPoint(),
              yScale(),
              qmin(),
              qmax(),
              0,
              &add_op));
      ASSERT_NE(nullptr, add_op);

      // 设置 Add 运算符的输入和输出数据
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_add_nc_q8(
              add_op,
              batchSize(),
              a.data(),
              aStride(),
              b.data(),
              bStride(),
              y.data(),
              yStride()));

      // 运行 Add 运算符
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(add_op, nullptr /* thread pool */));

      // 删除 Add 运算符
      ASSERT_EQ(
          pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(add_op));
      add_op = nullptr;

      /* 验证结果 */
      // 遍历每个 batch 的数据
      for (size_t i = 0; i < batchSize(); i++) {
        // 遍历每个通道的数据
        for (size_t c = 0; c < channels(); c++) {
          // 断言 y 中的每个元素在 qmin() 和 qmax() 范围内
          ASSERT_LE(uint32_t(y[i * yStride() + c]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(y[i * yStride() + c]), uint32_t(qmin()));
          // 断言计算出的 y 值与参考值的误差在 0.6f 以内
          ASSERT_NEAR(
              float(int32_t(y[i * yStride() + c])),
              yRef[i * channels() + c],
              0.6f);
        }
      }
    }
  }



// 结束了一个类的私有部分和类定义的结尾
private:
  // 批处理大小，默认为1
  size_t batchSize_{1};
  // 通道数，默认为1
  size_t channels_{1};
  // 输入张量a的步长，默认为0
  size_t aStride_{0};
  // 输入张量b的步长，默认为0
  size_t bStride_{0};
  // 输出张量y的步长，默认为0
  size_t yStride_{0};
  // 输入张量a的缩放因子，默认为0.75
  float aScale_{0.75f};
  // 输入张量b的缩放因子，默认为1.25
  float bScale_{1.25f};
  // 输出张量y的缩放因子，默认为0.96875
  float yScale_{0.96875f};
  // 输入张量a的零点，默认为121
  uint8_t aZeroPoint_{121};
  // 输入张量b的零点，默认为127
  uint8_t bZeroPoint_{127};
  // 输出张量y的零点，默认为133
  uint8_t yZeroPoint_{133};
  // 量化后的最小值，默认为0
  uint8_t qmin_{0};
  // 量化后的最大值，默认为255
  uint8_t qmax_{255};
  // 迭代次数，默认为15
  size_t iterations_{15};



// 完成了类成员变量的声明和初始化，私有部分结束


这段代码是 C++ 中类的私有成员变量的声明和初始化部分，每个变量都有默认值或者初始化值。
};


注释：


// 这行代码表示一个空代码块结束的标志
// 在 C++ 中，分号 ";" 表示一个空语句，通常用于指示一个空的语句块的结束
// 该分号后面没有代码，表示这是一个空的语句块
// 可能在代码中用于占位或者后续填充代码
// 不包含任何实际逻辑或功能
```