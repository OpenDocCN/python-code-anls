# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\avgpool-microkernel-tester.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>    // 包含算法库，例如 std::min, std::max 等
#include <cassert>      // 包含断言库，用于运行时检查条件
#include <cmath>        // 包含数学函数库，例如 std::sqrt, std::pow 等
#include <cstddef>      // 包含标准库定义的大小类型和空指针宏
#include <cstdlib>      // 包含通用工具函数库，例如 std::malloc, std::free 等
#include <functional>   // 包含函数对象库，用于创建和操作函数对象
#include <random>       // 包含随机数生成库，例如 std::uniform_int_distribution 等
#include <vector>       // 包含向量容器库，用于存储动态大小的数组

#include <qnnpack/AlignedAllocator.h>   // 包含 QNNPACK 中的内存对齐分配器头文件
#include <qnnpack/params.h>             // 包含 QNNPACK 中的参数定义头文件
#include <qnnpack/requantization.h>     // 包含 QNNPACK 中的重新量化函数头文件

class AvgPoolMicrokernelTester {
 public:
  // 设置并返回 n 的值，用于指定测试中处理的样本数
  inline AvgPoolMicrokernelTester& n(size_t n) {
    assert(n != 0);         // 断言 n 不为零
    this->n_ = n;           // 设置私有成员变量 n_
    return *this;
  }

  // 返回当前设置的 n 值，表示测试中处理的样本数
  inline size_t n() const {
    return this->n_;        // 返回私有成员变量 n_
  }

  // 设置并返回 s 的值，用于指定测试中的步长大小
  inline AvgPoolMicrokernelTester& s(size_t s) {
    assert(s != 0);         // 断言 s 不为零
    this->s_ = s;           // 设置私有成员变量 s_
    return *this;
  }

  // 返回当前设置的 s 值，表示测试中的步长大小
  inline size_t s() const {
    return this->s_;        // 返回私有成员变量 s_
  }

  // 设置并返回 kh 的值，用于指定测试中的核高度
  inline AvgPoolMicrokernelTester& kh(size_t kh) {
    assert(kh != 0);        // 断言 kh 不为零
    this->kh_ = kh;         // 设置私有成员变量 kh_
    return *this;
  }

  // 返回当前设置的 kh 值，表示测试中的核高度
  inline size_t kh() const {
    return this->kh_;       // 返回私有成员变量 kh_
  }

  // 设置并返回 kw 的值，用于指定测试中的核宽度
  inline AvgPoolMicrokernelTester& kw(size_t kw) {
    assert(kw != 0);        // 断言 kw 不为零
    this->kw_ = kw;         // 设置私有成员变量 kw_
    return *this;
  }

  // 返回当前设置的 kw 值，表示测试中的核宽度
  inline size_t kw() const {
    return this->kw_;       // 返回私有成员变量 kw_
  }

  // 返回当前设置的 ks 值，表示测试中的核大小
  inline size_t ks() const {
    return kh() * kw();     // 计算核大小并返回
  }

  // 返回当前设置的 packedKs 值，表示经过打包后的核大小
  inline size_t packedKs() const {
    if (kc() < kr()) {
      return ks();         // 如果 kc 小于 kr，则直接返回 ks
    } else if (ks() <= mr()) {
      return mr();         // 如果 ks 小于等于 mr，则返回 mr
    } else {
      return (ks() - mr()) % qr() == 0
          ? ks()
          : ((ks() - mr()) / qr() + 1) * qr() + mr();
                           // 否则计算并返回经过打包后的核大小
    }
  }

  // 设置并返回 mr 的值，用于指定测试中的 mr 大小
  inline AvgPoolMicrokernelTester& mr(size_t mr) {
    assert(mr != 0);        // 断言 mr 不为零
    this->mr_ = mr;         // 设置私有成员变量 mr_
    return *this;
  }

  // 返回当前设置的 mr 值，表示测试中的 mr 大小
  inline size_t mr() const {
    return this->mr_;       // 返回私有成员变量 mr_
  }

  // 设置并返回 qr 的值，用于指定测试中的 qr 大小
  inline AvgPoolMicrokernelTester& qr(size_t qr) {
    assert(qr != 0);        // 断言 qr 不为零
    this->qr_ = qr;         // 设置私有成员变量 qr_
    return *this;
  }

  // 返回当前设置的 qr 值，表示测试中的 qr 大小
  inline size_t qr() const {
    return this->qr_;       // 返回私有成员变量 qr_
  }

  // 设置并返回 kc 的值，用于指定测试中的 kc 大小
  inline AvgPoolMicrokernelTester& kc(size_t kc) {
    assert(kc != 0);        // 断言 kc 不为零
    this->kc_ = kc;         // 设置私有成员变量 kc_
    return *this;
  }

  // 返回当前设置的 kc 值，表示测试中的 kc 大小
  inline size_t kc() const {
    return this->kc_;       // 返回私有成员变量 kc_
  }

  // 设置并返回 kr 的值，用于指定测试中的 kr 大小
  inline AvgPoolMicrokernelTester& kr(size_t kr) {
    assert(kr != 0);        // 断言 kr 不为零
    this->kr_ = kr;         // 设置私有成员变量 kr_
    return *this;
  }

  // 返回当前设置的 kr 值，表示测试中的 kr 大小
  inline size_t kr() const {
    return this->kr_;       // 返回私有成员变量 kr_
  }

  // 返回当前设置的 packedN 值，表示经过打包后的样本数大小
  inline size_t packedN() const {
    return kc() % kr() == 0 ? kc() : (kc() / kr() + 1) * kr();
                           // 计算并返回经过打包后的样本数大小
  }

  // 设置并返回 xStride 的值，用于指定测试中的 x 方向步长
  inline AvgPoolMicrokernelTester& xStride(size_t xStride) {
    assert(xStride != 0);   // 断言 xStride 不为零
    this->xStride_ = xStride;   // 设置私有成员变量 xStride_
    return *this;
  }

  // 返回当前设置的 xStride 值，表示测试中的 x 方向步长
  inline size_t xStride() const {
    if (this->xStride_ == 0) {
      return kc();         // 如果 xStride 为零，则返回 kc
    } else {
      assert(this->xStride_ >= kc());
      return this->xStride_;   // 否则返回私有成员变量 xStride_
    }
  }

  // 设置并返回 yStride 的值，用于指定测试中的 y 方向步长
  inline AvgPoolMicrokernelTester& yStride(size_t yStride) {
    assert(yStride != 0);   // 断言 yStride 不为零
    this->yStride_ = yStride;   // 设置私有成员变量 yStride_
    return *this;
  }

  // 返回当前设置的 yStride 值，表示测试中的 y 方向步长
  inline size_t yStride() const {
    if (this->yStride_ == 0) {
      return kc();         // 如果 yStride 为零，则返回 kc
    } else {
      assert(this->yStride_ >= kc());
      return this->yStride_;   // 否则返回私有成员变量 yStride_
    }
  }

  // 设置并返回 xScale 的值，用于指定测试中的 x 缩放比例
  inline AvgPoolMicrokernelTester& xScale
    // 断言 xScale 必须大于 0
    assert(xScale > 0.0f);
    // 断言 xScale 是一个正常数（即不是 NaN、无穷大或零）
    assert(std::isnormal(xScale));
    // 将传入的 xScale 值赋给对象的 xScale 成员变量
    this->xScale_ = xScale;
    // 返回当前对象的引用
    return *this;
  }

  // 返回对象的 xScale 成员变量的值
  inline float xScale() const {
    return this->xScale_;
  }

  // 设置对象的 xZeroPoint 成员变量的值，并返回当前对象的引用
  inline AvgPoolMicrokernelTester& xZeroPoint(uint8_t xZeroPoint) {
    this->xZeroPoint_ = xZeroPoint;
    return *this;
  }

  // 返回对象的 xZeroPoint 成员变量的值
  inline uint8_t xZeroPoint() const {
    return this->xZeroPoint_;
  }

  // 设置对象的 yScale 成员变量的值，并返回当前对象的引用
  inline AvgPoolMicrokernelTester& yScale(float yScale) {
    // 断言 yScale 必须大于 0
    assert(yScale > 0.0f);
    // 断言 yScale 是一个正常数（即不是 NaN、无穷大或零）
    assert(std::isnormal(yScale));
    // 将传入的 yScale 值赋给对象的 yScale 成员变量
    this->yScale_ = yScale;
    // 返回当前对象的引用
    return *this;
  }

  // 返回对象的 yScale 成员变量的值
  inline float yScale() const {
    return this->yScale_;
  }

  // 设置对象的 yZeroPoint 成员变量的值，并返回当前对象的引用
  inline AvgPoolMicrokernelTester& yZeroPoint(uint8_t yZeroPoint) {
    this->yZeroPoint_ = yZeroPoint;
    return *this;
  }

  // 返回对象的 yZeroPoint 成员变量的值
  inline uint8_t yZeroPoint() const {
    return this->yZeroPoint_;
  }

  // 设置对象的 yMin 成员变量的值，并返回当前对象的引用
  inline AvgPoolMicrokernelTester& yMin(uint8_t yMin) {
    this->yMin_ = yMin;
    return *this;
  }

  // 返回对象的 yMin 成员变量的值
  inline uint8_t yMin() const {
    return this->yMin_;
  }

  // 设置对象的 yMax 成员变量的值，并返回当前对象的引用
  inline AvgPoolMicrokernelTester& yMax(uint8_t yMax) {
    this->yMax_ = yMax;
    return *this;
  }

  // 返回对象的 yMax 成员变量的值
  inline uint8_t yMax() const {
    return this->yMax_;
  }

  // 设置对象的 iterations 成员变量的值，并返回当前对象的引用
  inline AvgPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回对象的 iterations 成员变量的值
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 测试函数，接受一个 pytorch_q8avgpool_up_ukernel_function 函数对象
  void test(pytorch_q8avgpool_up_ukernel_function q8avgpool) const {
    // 生成随机设备对象
    std::random_device randomDevice;
    // 使用随机设备生成 Mersenne Twister 伪随机数生成器
    auto rng = std::mt19937(randomDevice());
    // 使用 bind 绑定到 uint8_t 类型的均匀分布，形成函数对象 u8rng
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建间接指针数组 indirectX，大小为 packedKs() + (n() * s() - 1) * kh()
    std::vector<const uint8_t*> indirectX(packedKs() + (n() * s() - 1) * kh());
    // 创建 x 向量，大小为 (indirectX.size() - 1) * xStride() + kc()
    std::vector<uint8_t> x((indirectX.size() - 1) * xStride() + kc());

    // 创建大小为 kc() 的零向量 zero
    std::vector<uint8_t> zero(kc());
    // 创建大小为 (n() - 1) * yStride() + kc() 的 y 向量
    std::vector<uint8_t> y((n() - 1) * yStride() + kc());
    // 创建大小为 n() * kc() 的 yRef 向量
    std::vector<uint8_t> yRef(n() * kc());
    // 创建大小为 n() * kc() 的 yFP 向量
    std::vector<float> yFP(n() * kc());
    // 创建大小为 n() * kc() 的 yAcc 向量
    std::vector<int32_t> yAcc(n() * kc());
    }
  }

  // 测试函数，接受一个 pytorch_q8avgpool_mp_ukernel_function 函数对象
  void test(pytorch_q8avgpool_mp_ukernel_function q8avgpool) const {
    // 生成随机设备对象
    std::random_device randomDevice;
    // 使用随机设备生成 Mersenne Twister 伪随机数生成器
    auto rng = std::mt19937(randomDevice());
    // 使用 bind 绑定到 uint8_t 类型的均匀分布，形成函数对象 u8rng
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建间接指针数组 indirectX，大小为 packedKs() + (n() * s() - 1) * kh()
    std::vector<const uint8_t*> indirectX(packedKs() + (n() * s() - 1) * kh());
    // 创建 x 向量，大小为 (indirectX.size() - 1) * xStride() + kc()
    std::vector<uint8_t> x((indirectX.size() - 1) * xStride() + kc());
    // 使用 AlignedAllocator 创建大小为 packedN() 的 mpAcc 向量
    std::vector<int32_t, AlignedAllocator<int32_t, 16>> mpAcc(packedN());

    // 创建大小为 kc() 的零向量 zero
    std::vector<uint8_t> zero(kc());
    // 创建大小为 (n() - 1) * yStride() + kc() 的 y 向量
    std::vector<uint8_t> y((n() - 1) * yStride() + kc());
    // 创建大小为 n() * kc() 的 yRef 向量
    std::vector<uint8_t> yRef(n() * kc());
    // 创建大小为 n() * kc() 的 yFP 向量
    std::vector<float> yFP(n() * kc());
    // 创建大小为 n() * kc() 的 yAcc 向量
    std::vector<int32_t> yAcc(n() * kc());
    }
  }
};


注释：


# 这是一个代码块的结束，一般用于结束一个函数、循环、条件语句等。在这里，表示结束了一个语句块或函数定义。
```