# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\maxpool-microkernel-tester.h`

```
/*
 * 版权声明，此代码版权归 Facebook 及其关联公司所有。
 * 保留所有权利。
 *
 * 此源代码基于 BSD 风格许可证发布，许可证详见源代码根目录下的 LICENSE 文件。
 */

#pragma once

// 引入标准库头文件
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

// 引入 QNNPACK 库的头文件
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

// MaxPoolMicrokernelTester 类定义
class MaxPoolMicrokernelTester {
 public:
  // 设置并返回参数 n
  inline MaxPoolMicrokernelTester& n(size_t n) {
    // 确保 n 不为零
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  // 返回参数 n 的值
  inline size_t n() const {
    return this->n_;
  }

  // 设置并返回参数 s
  inline MaxPoolMicrokernelTester& s(size_t s) {
    // 确保 s 不为零
    assert(s != 0);
    this->s_ = s;
    return *this;
  }

  // 返回参数 s 的值
  inline size_t s() const {
    return this->s_;
  }

  // 设置并返回参数 kh
  inline MaxPoolMicrokernelTester& kh(size_t kh) {
    // 确保 kh 不为零
    assert(kh != 0);
    this->kh_ = kh;
    return *this;
  }

  // 返回参数 kh 的值
  inline size_t kh() const {
    return this->kh_;
  }

  // 设置并返回参数 kw
  inline MaxPoolMicrokernelTester& kw(size_t kw) {
    // 确保 kw 不为零
    assert(kw != 0);
    this->kw_ = kw;
    return *this;
  }

  // 返回参数 kw 的值
  inline size_t kw() const {
    return this->kw_;
  }

  // 返回 ks 的计算结果
  inline size_t ks() const {
    return kh() * kw();
  }

  // 返回 packedKs 的计算结果
  inline size_t packedKs() const {
    if (kc() < kr()) {
      return ks();
    } else if (ks() <= mr()) {
      return mr();
    } else {
      return (ks() - mr()) % qr() == 0
          ? ks()
          : ((ks() - mr()) / qr() + 1) * qr() + mr();
    }
  }

  // 设置并返回参数 mr
  inline MaxPoolMicrokernelTester& mr(size_t mr) {
    // 确保 mr 不为零
    assert(mr != 0);
    this->mr_ = mr;
    return *this;
  }

  // 返回参数 mr 的值
  inline size_t mr() const {
    return this->mr_;
  }

  // 设置并返回参数 qr
  inline MaxPoolMicrokernelTester& qr(size_t qr) {
    // 确保 qr 不为零
    assert(qr != 0);
    this->qr_ = qr;
    return *this;
  }

  // 返回参数 qr 的值
  inline size_t qr() const {
    return this->qr_;
  }

  // 设置并返回参数 kc
  inline MaxPoolMicrokernelTester& kc(size_t kc) {
    // 确保 kc 不为零
    assert(kc != 0);
    this->kc_ = kc;
    return *this;
  }

  // 返回参数 kc 的值
  inline size_t kc() const {
    return this->kc_;
  }

  // 设置并返回参数 kr
  inline MaxPoolMicrokernelTester& kr(size_t kr) {
    // 确保 kr 不为零
    assert(kr != 0);
    this->kr_ = kr;
    return *this;
  }

  // 返回参数 kr 的值
  inline size_t kr() const {
    return this->kr_;
  }

  // 返回 packedN 的计算结果
  inline size_t packedN() const {
    return kc() % kr() == 0 ? kc() : (kc() / kr() + 1) * kr();
  }

  // 设置并返回参数 xStride
  inline MaxPoolMicrokernelTester& xStride(size_t xStride) {
    // 确保 xStride 不为零
    assert(xStride != 0);
    this->xStride_ = xStride;
    return *this;
  }

  // 返回参数 xStride 的值
  inline size_t xStride() const {
    if (this->xStride_ == 0) {
      return kc();
    } else {
      assert(this->xStride_ >= kc());
      return this->xStride_;
    }
  }

  // 设置并返回参数 yStride
  inline MaxPoolMicrokernelTester& yStride(size_t yStride) {
    // 确保 yStride 不为零
    assert(yStride != 0);
    this->yStride_ = yStride;
    return *this;
  }

  // 返回参数 yStride 的值
  inline size_t yStride() const {
    if (this->yStride_ == 0) {
      return kc();
    } else {
      assert(this->yStride_ >= kc());
      return this->yStride_;
    }
  }

  // 设置参数 qmin
  inline MaxPoolMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }
    // 返回当前对象的引用，用于支持连续调用
    return *this;
  }

  // 返回最小量化参数 qmin_
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置最大量化参数 qmax_，并返回当前对象的引用，支持连续调用
  inline MaxPoolMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回最大量化参数 qmax_
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置迭代次数 iterations_，并返回当前对象的引用，支持连续调用
  inline MaxPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回迭代次数 iterations_
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 执行测试，使用给定的 u8maxpool 函数
  void test(pytorch_u8maxpool_ukernel_function u8maxpool) const {
    // 生成随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建间接索引向量 indirectX 和输入向量 x
    std::vector<const uint8_t*> indirectX(packedKs() + (n() * s() - 1) * kh());
    std::vector<uint8_t> x((indirectX.size() - 1) * xStride() + kc());

    // 创建零向量 zero、输出向量 y 和参考输出向量 yRef
    std::vector<uint8_t> zero(kc());
    std::vector<uint8_t> y((n() - 1) * yStride() + kc());
    std::vector<uint8_t> yRef(n() * kc());

    // 执行多次迭代
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 生成随机输入向量 x
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      // 将输出向量 y 填充为 0xA5
      std::fill(y.begin(), y.end(), 0xA5);

      // 更新间接索引向量 indirectX
      for (size_t i = 0; i < indirectX.size(); i++) {
        indirectX[i] = x.data() + i * xStride();
      }
      // 对 indirectX 进行随机排列
      std::shuffle(indirectX.begin(), indirectX.end(), rng);

      /* 准备量化参数 */
      const union pytorch_qnnp_u8_clamping_params clampingParams =
          pytorch_qnnp_compute_u8_clamping_params(qmin(), qmax());

      /* 计算参考结果 */
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          uint8_t maxValue = 0;
          for (size_t j = 0; j < ks(); j++) {
            maxValue = std::max(maxValue, indirectX[i * s() * kh() + j][k]);
          }
          maxValue = std::min(maxValue, qmax());
          maxValue = std::max(maxValue, qmin());
          yRef[i * kc() + k] = maxValue;
        }
      }

      /* 调用优化的微内核函数 */
      u8maxpool(
          n(),
          ks(),
          kc(),
          indirectX.data(),
          y.data(),
          (kh() * s() - packedKs()) * sizeof(void*),
          (yStride() - kc()) * sizeof(uint8_t),
          &clampingParams);

      /* 验证结果 */
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_EQ(
              uint32_t(yRef[i * kc() + k]), uint32_t(y[i * yStride() + k]))
              << "at pixel " << i << ", channel " << k << ", n = " << n()
              << ", ks = " << kh() << "x" << kw() << " (" << ks()
              << "), kc = " << kc();
        }
      }
    }
  }
};



# 这行代码似乎是一个语法错误或意外的分号，可能是代码中的一个错误，应该检查并修复它。
```