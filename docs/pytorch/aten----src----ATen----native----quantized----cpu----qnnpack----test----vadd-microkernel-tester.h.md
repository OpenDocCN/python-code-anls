# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\vadd-microkernel-tester.h`

```
/*
 * 版权声明
 * 版权所有（c）Facebook，Inc.及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的LICENSE文件中的BSD样式许可证进行许可。
 */

#pragma once

#include <algorithm>   // 引入标准库中的算法头文件
#include <cassert>     // 引入断言头文件，用于运行时条件检查
#include <cstddef>     // 引入标准库中的stddef头文件
#include <cstdlib>     // 引入标准库中的cstdlib头文件
#include <functional>  // 引入函数对象相关的头文件
#include <random>      // 引入随机数生成器相关的头文件
#include <vector>      // 引入向量容器相关的头文件

#include <qnnpack/params.h>         // 引入QNNPACK中的参数头文件
#include <qnnpack/requantization.h> // 引入QNNPACK中的重新量化头文件

class VAddMicrokernelTester {
 public:
  inline VAddMicrokernelTester& n(size_t n) {  // 设置n的值
    assert(n != 0);                          // 断言n不为0
    this->n_ = n;                            // 将n值赋给类成员变量n_
    return *this;
  }

  inline size_t n() const {  // 获取n的值
    return this->n_;         // 返回类成员变量n_
  }

  inline VAddMicrokernelTester& inplaceA(bool inplaceA) {  // 设置是否在A上原地操作的标志
    this->inplaceA_ = inplaceA;                            // 将inplaceA值赋给类成员变量inplaceA_
    return *this;
  }

  inline bool inplaceA() const {  // 获取是否在A上原地操作的标志
    return this->inplaceA_;       // 返回类成员变量inplaceA_
  }

  inline VAddMicrokernelTester& inplaceB(bool inplaceB) {  // 设置是否在B上原地操作的标志
    this->inplaceB_ = inplaceB;                            // 将inplaceB值赋给类成员变量inplaceB_
    return *this;
  }

  inline bool inplaceB() const {  // 获取是否在B上原地操作的标志
    return this->inplaceB_;       // 返回类成员变量inplaceB_
  }

  inline VAddMicrokernelTester& aScale(float aScale) {  // 设置A的缩放因子
    assert(aScale > 0.0f);                              // 断言aScale大于0
    assert(std::isnormal(aScale));                      // 断言aScale为正规数
    this->aScale_ = aScale;                             // 将aScale值赋给类成员变量aScale_
    return *this;
  }

  inline float aScale() const {  // 获取A的缩放因子
    return this->aScale_;        // 返回类成员变量aScale_
  }

  inline VAddMicrokernelTester& aZeroPoint(uint8_t aZeroPoint) {  // 设置A的零点
    this->aZeroPoint_ = aZeroPoint;                              // 将aZeroPoint值赋给类成员变量aZeroPoint_
    return *this;
  }

  inline uint8_t aZeroPoint() const {  // 获取A的零点
    return this->aZeroPoint_;          // 返回类成员变量aZeroPoint_
  }

  inline VAddMicrokernelTester& bScale(float bScale) {  // 设置B的缩放因子
    assert(bScale > 0.0f);                              // 断言bScale大于0
    assert(std::isnormal(bScale));                      // 断言bScale为正规数
    this->bScale_ = bScale;                             // 将bScale值赋给类成员变量bScale_
    return *this;
  }

  inline float bScale() const {  // 获取B的缩放因子
    return this->bScale_;        // 返回类成员变量bScale_
  }

  inline VAddMicrokernelTester& bZeroPoint(uint8_t bZeroPoint) {  // 设置B的零点
    this->bZeroPoint_ = bZeroPoint;                              // 将bZeroPoint值赋给类成员变量bZeroPoint_
    return *this;
  }

  inline uint8_t bZeroPoint() const {  // 获取B的零点
    return this->bZeroPoint_;          // 返回类成员变量bZeroPoint_
  }

  inline VAddMicrokernelTester& yScale(float yScale) {  // 设置输出Y的缩放因子
    assert(yScale > 0.0f);                              // 断言yScale大于0
    assert(std::isnormal(yScale));                      // 断言yScale为正规数
    this->yScale_ = yScale;                             // 将yScale值赋给类成员变量yScale_
    return *this;
  }

  inline float yScale() const {  // 获取输出Y的缩放因子
    return this->yScale_;        // 返回类成员变量yScale_
  }

  inline VAddMicrokernelTester& yZeroPoint(uint8_t yZeroPoint) {  // 设置输出Y的零点
    this->yZeroPoint_ = yZeroPoint;                              // 将yZeroPoint值赋给类成员变量yZeroPoint_
    return *this;
  }

  inline uint8_t yZeroPoint() const {  // 获取输出Y的零点
    return this->yZeroPoint_;          // 返回类成员变量yZeroPoint_
  }

  inline VAddMicrokernelTester& qmin(uint8_t qmin) {  // 设置量化的最小值
    this->qmin_ = qmin;                              // 将qmin值赋给类成员变量qmin_
    return *this;
  }

  inline uint8_t qmin() const {  // 获取量化的最小值
    return this->qmin_;        // 返回类成员变量qmin_
  }

  inline VAddMicrokernelTester& qmax(uint8_t qmax) {  // 设置量化的最大值
    this->qmax_ = qmax;                              // 将qmax值赋给类成员变量qmax_
    return *this;
  }

  inline uint8_t qmax() const {  // 获取量化的最大值
    return this->qmax_;        // 返回类成员变量qmax_
  }

  inline VAddMicrokernelTester& iterations(size_t iterations) {  // 设置测试的迭代次数
    this->iterations_ = iterations;                              // 将iterations值赋给类成员变量iterations_
    return *this;
  }

  inline size_t iterations() const {  // 获取测试的迭代次数
    return this->iterations_;        // 返回类成员变量iterations_
  }

  void test(pytorch_q8vadd_ukernel_function q8vadd) const {  // 执行测试函数，传入q8vadd函数指针
    std::random_device randomDevice;  // 创建随机设备对象
    auto rng = std::mt19937(randomDevice());  // 创建以随机设备为种子的Mersenne Twister随机数生成器
    // 创建一个函数对象 u8rng，用于生成 uint8_t 类型的随机数
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建三个长度为 n() 的 uint8_t 类型的向量 a、b、y
    std::vector<uint8_t> a(n());
    std::vector<uint8_t> b(n());
    std::vector<uint8_t> y(n());

    // 创建一个长度为 n() 的 float 类型的向量 yFP
    std::vector<float> yFP(n());

    // 创建一个长度为 n() 的 uint8_t 类型的向量 yRef
    std::vector<uint8_t> yRef(n());

    // 迭代执行操作，执行 iterations() 次循环
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 生成随机数填充向量 a 和 b
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));

      // 根据 inplaceA() 和 inplaceB() 的值决定填充向量 y 的方式
      if (inplaceA() || inplaceB()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng));
      } else {
        std::fill(y.begin(), y.end(), 0xA5);  // 填充向量 y 的所有元素为 0xA5
      }

      // 根据 inplaceA() 的值选择 aData 的来源
      const uint8_t* aData = inplaceA() ? y.data() : a.data();

      // 根据 inplaceB() 的值选择 bData 的来源
      const uint8_t* bData = inplaceB() ? y.data() : b.data();

      /* 准备量化参数 */
      const union pytorch_qnnp_add_quantization_params quantizationParams =
          pytorch_qnnp_compute_add_quantization_params(
              aZeroPoint(),
              bZeroPoint(),
              yZeroPoint(),
              aScale() / yScale(),
              bScale() / yScale(),
              qmin(),
              qmax());

      // 计算标量量化参数
      const union pytorch_qnnp_add_quantization_params
          scalarQuantizationParams =
              pytorch_qnnp_compute_scalar_add_quantization_params(
                  aZeroPoint(),
                  bZeroPoint(),
                  yZeroPoint(),
                  aScale() / yScale(),
                  bScale() / yScale(),
                  qmin(),
                  qmax());

      /* 计算参考结果 */
      for (size_t i = 0; i < n(); i++) {
        // 计算浮点数结果 yFP[i]
        yFP[i] = float(yZeroPoint()) +
            float(int32_t(aData[i]) - int32_t(aZeroPoint())) *
                (aScale() / yScale()) +
            float(int32_t(bData[i]) - int32_t(bZeroPoint())) *
                (bScale() / yScale());

        // 确保 yFP[i] 在 qmin() 和 qmax() 范围内
        yFP[i] = std::min<float>(yFP[i], float(qmax()));
        yFP[i] = std::max<float>(yFP[i], float(qmin()));

        // 使用 pytorch_qnnp_add_quantize 函数计算 yRef[i]
        yRef[i] = pytorch_qnnp_add_quantize(
            aData[i], bData[i], scalarQuantizationParams);
      }

      /* 调用优化的微内核 */
      q8vadd(n(), aData, bData, y.data(), &quantizationParams);

      /* 验证结果 */
      for (size_t i = 0; i < n(); i++) {
        // 使用 ASSERT_LE 验证 y[i] <= qmax()
        ASSERT_LE(uint32_t(y[i]), uint32_t(qmax()))
            << "at " << i << ", n = " << n();

        // 使用 ASSERT_GE 验证 y[i] >= qmin()
        ASSERT_GE(uint32_t(y[i]), uint32_t(qmin()))
            << "at " << i << ", n = " << n();

        // 使用 ASSERT_NEAR 验证浮点数 y[i] 与 yFP[i] 的接近程度
        ASSERT_NEAR(float(int32_t(y[i])), yFP[i], 0.6f)
            << "at " << i << ", n = " << n();

        // 使用 ASSERT_EQ 验证 y[i] 与 yRef[i] 的相等性
        ASSERT_EQ(uint32_t(yRef[i]), uint32_t(y[i]))
            << "at " << i << ", n = " << n();
      }
    }
};


注释：


# 这是一个单独的分号，通常用于语句结束
```