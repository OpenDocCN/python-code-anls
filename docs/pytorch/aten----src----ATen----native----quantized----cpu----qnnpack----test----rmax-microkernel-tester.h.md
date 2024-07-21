# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\rmax-microkernel-tester.h`

```py
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在根目录下的LICENSE文件中以BSD风格许可证授权。
 */

#pragma once

#include <algorithm>       // 引入算法库，包含常见算法如std::max
#include <cassert>         // 引入断言库，用于运行时检查条件
#include <cstddef>         // 引入stddef.h标准头文件，定义了各种数据类型和宏
#include <cstdlib>         // 引入cstdlib标准头文件，包含常用函数和宏
#include <functional>      // 引入函数对象库，用于创建和操作函数对象
#include <random>          // 引入随机数库，提供随机数生成功能
#include <vector>          // 引入向量容器库，提供动态数组功能

#include <qnnpack/params.h>  // 引入QNNPACK参数头文件

class RMaxMicrokernelTester {
 public:
  // 设置测试的向量大小n
  inline RMaxMicrokernelTester& n(size_t n) {
    assert(n != 0);       // 断言n不为0
    this->n_ = n;
    return *this;
  }

  // 返回当前测试的向量大小n
  inline size_t n() const {
    return this->n_;
  }

  // 设置测试的迭代次数
  inline RMaxMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前测试的迭代次数
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 执行测试函数，传入优化后的微内核函数u8rmax
  void test(pytorch_u8rmax_ukernel_function u8rmax) const {
    std::random_device randomDevice;   // 创建随机设备对象
    auto rng = std::mt19937(randomDevice());  // 创建Mersenne Twister随机数生成器
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);  // 创建uint8_t类型的随机数生成器

    std::vector<uint8_t> x(n());    // 创建长度为n的uint8_t类型向量x
    for (size_t iteration = 0; iteration < iterations(); iteration++) {  // 迭代测试次数
      std::generate(x.begin(), x.end(), std::ref(u8rng));  // 使用u8rng生成随机数填充向量x

      /* 计算参考结果 */
      uint8_t yRef = 0;   // 初始化参考结果为0
      for (size_t i = 0; i < n(); i++) {
        yRef = std::max(yRef, x[i]);  // 计算向量x中的最大值，更新yRef
      }

      /* 调用优化后的微内核函数 */
      const uint8_t y = u8rmax(n(), x.data());  // 调用u8rmax计算向量x的最大值，保存结果到y

      /* 验证结果 */
      ASSERT_EQ(yRef, y) << "n = " << n();   // 断言yRef与y相等，如果不相等则输出测试失败信息
    }
  }

 private:
  size_t n_{1};           // 默认向量大小为1
  size_t iterations_{15}; // 默认迭代次数为15
};
```