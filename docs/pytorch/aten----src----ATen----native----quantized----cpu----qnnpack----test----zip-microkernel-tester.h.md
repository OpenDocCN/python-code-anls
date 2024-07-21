# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\zip-microkernel-tester.h`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdlib>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack/params.h>

// 定义 ZipMicrokernelTester 类，用于测试 Zip 微内核功能
class ZipMicrokernelTester {
 public:
  // 设置 n 参数，并进行断言确保 n 不为 0
  inline ZipMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  // 返回当前的 n 参数值
  inline size_t n() const {
    return this->n_;
  }

  // 设置 g 参数，并进行断言确保 g 不为 0
  inline ZipMicrokernelTester& g(size_t g) {
    assert(g != 0);
    this->g_ = g;
    return *this;
  }

  // 返回当前的 g 参数值
  inline size_t g() const {
    return this->g_;
  }

  // 设置 iterations 参数
  inline ZipMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前的 iterations 参数值
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 测试函数，接受一个 pytorch_xzipc_ukernel_function 类型的参数 xzip
  void test(pytorch_xzipc_ukernel_function xzip) const {
    // 随机数生成设备和引擎
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    // 生成 8 位无符号整数的均匀分布
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建大小为 n * g 的输入向量 x 和大小为 g * n 的输出向量 y
    std::vector<uint8_t> x(n() * g());
    std::vector<uint8_t> y(g() * n());

    // 进行若干次迭代测试
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 使用 u8rng 生成随机输入向量 x
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      // 将输出向量 y 填充为 0xA5
      std::fill(y.begin(), y.end(), 0xA5);

      /* 调用优化的微内核函数 */
      xzip(n(), x.data(), y.data());

      /* 验证结果 */
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < g(); j++) {
          // 使用 ASSERT_EQ 断言检查 y[i * g() + j] 和 x[j * n() + i] 是否相等，若不相等则输出错误信息
          ASSERT_EQ(uint32_t(y[i * g() + j]), uint32_t(x[j * n() + i]))
              << "at element " << i << ", group " << j;
        }
      }
    }
  }

  // 测试函数，接受一个 pytorch_xzipv_ukernel_function 类型的参数 xzip
  void test(pytorch_xzipv_ukernel_function xzip) const {
    // 随机数生成设备和引擎
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    // 生成 8 位无符号整数的均匀分布
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建大小为 n * g 的输入向量 x 和大小为 g * n 的输出向量 y
    std::vector<uint8_t> x(n() * g());
    std::vector<uint8_t> y(g() * n());

    // 进行若干次迭代测试
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 使用 u8rng 生成随机输入向量 x
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      // 将输出向量 y 填充为 0xA5
      std::fill(y.begin(), y.end(), 0xA5);

      /* 调用优化的微内核函数 */
      xzip(n(), g(), x.data(), y.data());

      /* 验证结果 */
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < g(); j++) {
          // 使用 ASSERT_EQ 断言检查 y[i * g() + j] 和 x[j * n() + i] 是否相等，若不相等则输出错误信息
          ASSERT_EQ(uint32_t(y[i * g() + j]), uint32_t(x[j * n() + i]))
              << "at element " << i << ", group " << j;
        }
      }
    }
  }

 private:
  // 成员变量 n_ 代表向量中的元素数目
  size_t n_{1};
  // 成员变量 g_ 代表向量的组数
  size_t g_{1};
  // 成员变量 iterations_ 代表测试迭代次数
  size_t iterations_{3};
};
```