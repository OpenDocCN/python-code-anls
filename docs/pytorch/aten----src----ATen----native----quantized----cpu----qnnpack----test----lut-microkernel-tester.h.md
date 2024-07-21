# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\lut-microkernel-tester.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm> // 包含标准库中的算法头文件
#include <cassert>   // 包含断言相关的头文件
#include <cstddef>   // 包含标准库中的stddef头文件，定义了各种标准类型和宏
#include <cstdlib>   // 包含标准库中的cstdlib头文件，定义了各种通用工具函数
#include <functional> // 包含标准库中的functional头文件，提供了各种函数对象的类模板
#include <random>     // 包含标准库中的random头文件，提供了随机数生成器相关功能
#include <vector>     // 包含标准库中的vector头文件，提供了动态数组功能

#include <qnnpack/params.h> // 包含qnnpack库中的params.h头文件

class LUTMicrokernelTester {
 public:
  // 设置测试的大小n，并进行断言确保n不为0
  inline LUTMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  // 返回当前设置的测试大小n
  inline size_t n() const {
    return this->n_;
  }

  // 设置是否原地操作的标志，并返回当前对象的引用
  inline LUTMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  // 返回当前是否设置为原地操作
  inline bool inplace() const {
    return this->inplace_;
  }

  // 设置测试迭代次数iterations，并返回当前对象的引用
  inline LUTMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前设置的测试迭代次数iterations
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 执行测试，传入一个函数指针x8lut作为参数
  void test(pytorch_x8lut_ukernel_function x8lut) const {
    // 创建随机设备
    std::random_device randomDevice;
    // 创建 Mersenne Twister 伪随机数生成器，并初始化
    auto rng = std::mt19937(randomDevice());
    // 创建生成均匀分布的函数对象，生成范围为uint8_t类型的随机数
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建长度为n的uint8_t类型向量x，用生成的随机数填充
    std::vector<uint8_t> x(n());
    // 创建长度为256的uint8_t类型向量t，用生成的随机数填充
    std::vector<uint8_t> t(256);
    // 创建长度为n的uint8_t类型向量y和yRef，用于保存结果
    std::vector<uint8_t> y(n());
    std::vector<uint8_t> yRef(n());

    // 迭代测试iterations次
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 生成随机输入向量x
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      // 生成随机查找表t
      std::generate(t.begin(), t.end(), std::ref(u8rng));
      
      // 如果设置为原地操作，则生成随机输出向量y
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng));
      } else {
        // 否则将输出向量y的所有元素置为0xA5
        std::fill(y.begin(), y.end(), 0xA5);
      }

      // 根据是否原地操作选择输入数据指针
      const uint8_t* xData = inplace() ? y.data() : x.data();

      /* 计算参考结果 */
      for (size_t i = 0; i < n(); i++) {
        // 使用查找表t计算结果保存到yRef
        yRef[i] = t[xData[i]];
      }

      /* 调用优化的微内核函数 */
      x8lut(n(), xData, t.data(), y.data());

      /* 验证结果 */
      for (size_t i = 0; i < n(); i++) {
        // 断言每个输出y的值与参考结果yRef的值相等
        ASSERT_EQ(uint32_t(yRef[i]), uint32_t(y[i]))
            << "at position " << i << ", n = " << n();
      }
    }
  }

 private:
  size_t n_{1};        // 默认测试大小为1
  bool inplace_{false}; // 默认不进行原地操作
  size_t iterations_{15}; // 默认测试迭代次数为15
};
```