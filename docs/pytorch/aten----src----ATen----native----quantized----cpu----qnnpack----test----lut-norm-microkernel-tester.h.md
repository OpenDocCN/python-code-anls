# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\lut-norm-microkernel-tester.h`

```
/*
 * 版权所有（c）Facebook，Inc.及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的LICENSE文件中所述的BSD风格许可证进行许可。
 */

#pragma once

#include <algorithm>            // 包含标准算法库，用于算法操作
#include <cassert>              // 包含断言库，用于运行时检查
#include <cstddef>              // 包含stddef库，定义了多个标准库的重要类型和宏
#include <cstdlib>              // 包含cstdlib库，定义了常用的杂项函数及类型转换函数
#include <functional>           // 包含函数对象库，提供了各种函数对象的支持
#include <random>               // 包含随机数生成器库，用于生成随机数
#include <vector>               // 包含向量容器库，用于动态数组管理

#include <qnnpack/params.h>     // 引用qnnpack/params.h头文件，可能是特定的库或API的参数定义

class LUTNormMicrokernelTester {
 public:
  inline LUTNormMicrokernelTester& n(size_t n) {
    assert(n != 0);             // 断言：n不能为0，用于运行时检查
    this->n_ = n;               // 设置成员变量n_
    return *this;
  }

  inline size_t n() const {
    return this->n_;            // 返回成员变量n_
  }

  inline LUTNormMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;   // 设置成员变量inplace_
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;      // 返回成员变量inplace_
  }

  inline LUTNormMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;   // 设置成员变量iterations_
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;   // 返回成员变量iterations_
  }

  void test(pytorch_u8lut32norm_ukernel_function u8lut32norm) const {
    std::random_device randomDevice;    // 创建随机设备对象
    auto rng = std::mt19937(randomDevice());  // 创建Mersenne Twister随机数引擎
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);  // 创建uint8_t类型的随机数生成器
    auto u32rng = std::bind(
        std::uniform_int_distribution<uint32_t>(
            1, std::numeric_limits<uint32_t>::max() / (257 * n())),
        rng);   // 创建uint32_t类型的随机数生成器，限定范围为1到uint32_t类型最大值的一部分

    std::vector<uint8_t> x(n());    // 创建长度为n()的uint8_t类型向量x
    std::vector<uint32_t> t(256);   // 创建长度为256的uint32_t类型向量t
    std::vector<uint8_t> y(n());    // 创建长度为n()的uint8_t类型向量y
    std::vector<float> yRef(n());   // 创建长度为n()的float类型向量yRef

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));  // 使用u8rng生成x向量的随机数据
      std::generate(t.begin(), t.end(), std::ref(u32rng)); // 使用u32rng生成t向量的随机数据
      
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng)); // 如果inplace为true，使用u8rng生成y向量的随机数据
      } else {
        std::fill(y.begin(), y.end(), 0xA5);    // 如果inplace为false，将y向量填充为0xA5
      }

      const uint8_t* xData = inplace() ? y.data() : x.data();  // 根据inplace()选择性使用y.data()或x.data()作为xData

      /* 计算参考结果 */
      uint32_t sum = 0;
      for (size_t i = 0; i < n(); i++) {
        sum += t[xData[i]];     // 计算sum，用于后续的比例计算
      }
      for (size_t i = 0; i < n(); i++) {
        yRef[i] = 256.0f * float(t[xData[i]]) / float(sum);   // 计算每个元素的参考结果
        yRef[i] = std::min(yRef[i], 255.0f);   // 对yRef[i]进行最小值限制
      }

      /* 调用优化的微内核 */
      u8lut32norm(n(), xData, t.data(), y.data());   // 调用优化的微内核函数u8lut32norm

      /* 验证结果 */
      for (size_t i = 0; i < n(); i++) {
        ASSERT_NEAR(yRef[i], float(y[i]), 0.5f)    // 使用断言验证yRef[i]与y[i]的近似程度，允许误差为0.5f
            << "at position " << i << ", n = " << n() << ", sum = " << sum;   // 输出错误信息的详细位置信息
      }
    }
  }

 private:
  size_t n_{1};         // 默认n的初始值为1
  bool inplace_{false}; // 默认inplace的初始值为false
  size_t iterations_{15}; // 默认iterations的初始值为15
};
```