# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\clamp-microkernel-tester.h`

```
/*
 * 版权所有（C）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在根目录下的LICENSE文件中找到的BSD样式许可下许可。
 */

#pragma once

#include <algorithm>   // 包含STL的算法头文件
#include <cassert>     // 包含断言相关的头文件
#include <cstddef>     // 包含size_t的头文件
#include <cstdlib>     // 包含C标准库的头文件
#include <functional>  // 包含函数对象相关的头文件
#include <random>      // 包含随机数生成器相关的头文件
#include <vector>      // 包含向量容器相关的头文件

#include <qnnpack/params.h>           // 包含QNNPACK库的参数头文件
#include <qnnpack/requantization.h>   // 包含QNNPACK库的重新量化头文件

class ClampMicrokernelTester {
 public:
  inline ClampMicrokernelTester& n(size_t n) {   // 设置测试向量长度，并返回测试对象的引用
    assert(n != 0);                            // 断言：长度不为0
    this->n_ = n;                              // 将长度设置为指定值
    return *this;                              // 返回当前对象的引用
  }

  inline size_t n() const {                    // 获取当前测试向量长度
    return this->n_;                           // 返回保存的长度值
  }

  inline ClampMicrokernelTester& inplace(bool inplace) {   // 设置是否原地操作，并返回测试对象的引用
    this->inplace_ = inplace;                              // 设置原地操作标志
    return *this;                                          // 返回当前对象的引用
  }

  inline bool inplace() const {                 // 获取当前是否原地操作
    return this->inplace_;                      // 返回原地操作标志
  }

  inline ClampMicrokernelTester& qmin(uint8_t qmin) {   // 设置最小量化值，并返回测试对象的引用
    this->qmin_ = qmin;                                // 设置最小量化值
    return *this;                                      // 返回当前对象的引用
  }

  inline uint8_t qmin() const {                  // 获取当前最小量化值
    return this->qmin_;                          // 返回保存的最小量化值
  }

  inline ClampMicrokernelTester& qmax(uint8_t qmax) {   // 设置最大量化值，并返回测试对象的引用
    this->qmax_ = qmax;                                // 设置最大量化值
    return *this;                                      // 返回当前对象的引用
  }

  inline uint8_t qmax() const {                  // 获取当前最大量化值
    return this->qmax_;                          // 返回保存的最大量化值
  }

  inline ClampMicrokernelTester& iterations(size_t iterations) {   // 设置测试迭代次数，并返回测试对象的引用
    this->iterations_ = iterations;                                // 设置迭代次数
    return *this;                                                  // 返回当前对象的引用
  }

  inline size_t iterations() const {            // 获取当前测试迭代次数
    return this->iterations_;                   // 返回保存的迭代次数
  }

  void test(pytorch_u8clamp_ukernel_function u8clamp) const {   // 执行测试函数，传入优化的微内核函数指针
    std::random_device randomDevice;                           // 创建随机设备对象
    auto rng = std::mt19937(randomDevice());                   // 创建Mersenne Twister随机数生成器
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);   // 创建生成器绑定到uint8_t范围的随机数分布

    std::vector<uint8_t> x(n());               // 创建长度为n的uint8_t向量x
    std::vector<uint8_t> y(n());               // 创建长度为n的uint8_t向量y
    std::vector<uint8_t> yRef(n());            // 创建长度为n的uint8_t向量yRef
    for (size_t iteration = 0; iteration < iterations(); iteration++) {   // 迭代测试次数
      std::generate(x.begin(), x.end(), std::ref(u8rng));   // 生成随机数填充向量x
      if (inplace()) {                                      // 如果选择原地操作
        std::generate(y.begin(), y.end(), std::ref(u8rng)); // 则生成随机数填充向量y
      } else {
        std::fill(y.begin(), y.end(), 0xA5);                // 否则填充向量y为0xA5
      }
      const uint8_t* xData = inplace() ? y.data() : x.data();   // 设置输入数据指针，根据原地操作决定使用y或者x

      /* 准备夹紧参数 */
      const union pytorch_qnnp_u8_clamping_params clampingParams =
          pytorch_qnnp_compute_u8_clamping_params(qmin(), qmax());   // 计算夹紧参数

      /* 计算参考结果 */
      for (size_t i = 0; i < n(); i++) {   // 遍历长度为n的数据
        yRef[i] = std::max(std::min(xData[i], qmax()), qmin());   // 计算夹紧后的参考结果
      }

      /* 调用优化的微内核 */
      u8clamp(n(), xData, y.data(), &clampingParams);   // 调用传入的优化微内核函数

      /* 验证结果 */
      for (size_t i = 0; i < n(); i++) {   // 遍历长度为n的数据
        ASSERT_LE(uint32_t(y[i]), uint32_t(qmax()))   // 使用断言验证结果在夹紧范围内
            << "at position " << i << ", n = " << n();   // 输出错误位置和长度信息
        ASSERT_GE(uint32_t(y[i]), uint32_t(qmin()))   // 使用断言验证结果在夹紧范围内
            << "at position " << i << ", n = " << n();   // 输出错误位置和长度信息
        ASSERT_EQ(uint32_t(yRef[i]), uint32_t(y[i]))   // 使用断言验证结果与参考结果一致
            << "at position " << i << ", n = " << n() << ", qmin = " << qmin()   // 输出错误位置、长度和夹紧参数信息
            << ", qmax = " << qmax();   // 输出夹紧参数信息
      }
    }
  }



// 这是一个类的私有部分的结尾，关闭了类的定义和实现部分的声明

 private:
  size_t n_{1};
  // 类的私有成员变量 n_，默认初始化为 1

  bool inplace_{false};
  // 类的私有成员变量 inplace_，默认初始化为 false

  uint8_t qmin_{0};
  // 类的私有成员变量 qmin_，默认初始化为 0

  uint8_t qmax_{255};
  // 类的私有成员变量 qmax_，默认初始化为 255

  size_t iterations_{15};
  // 类的私有成员变量 iterations_，默认初始化为 15



// 这段代码定义了一个类的私有成员变量，包括整数和布尔类型，每个变量都有默认值


这样，注释就解释了每一行代码的作用和默认的初始化值。
};
```