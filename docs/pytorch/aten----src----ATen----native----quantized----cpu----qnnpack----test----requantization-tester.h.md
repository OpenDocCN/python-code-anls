# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\requantization-tester.h`

```py
/*
 * 版权所有（c）Facebook公司及其附属公司。
 * 保留所有权利。
 *
 * 此源代码在根目录中的LICENSE文件中发现的BSD样式许可下许可。
 */

#pragma once

#include <cstddef>
#include <cstdlib>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack/params.h>
#include <qnnpack/scalar-utils.h>

class RequantizationTester {
 public:
  // 设置并返回s值
  inline RequantizationTester& s(uint32_t s) {
    this->s_ = s;
    return *this;
  }

  // 返回当前s值
  inline uint32_t s() const {
    return this->s_;
  }

  // 根据s计算并返回scale值
  inline float scale() const {
    return ldexpf(1.0f, -s());
  }

  // 设置并返回zeroPoint值
  inline RequantizationTester& zeroPoint(int32_t zeroPoint) {
    this->zeroPoint_ = zeroPoint;
    return *this;
  }

  // 返回当前zeroPoint值
  inline int32_t zeroPoint() const {
    return this->zeroPoint_;
  }

  // 设置并返回qmin值
  inline RequantizationTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 返回当前qmin值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置并返回qmax值
  inline RequantizationTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回当前qmax值
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置并返回iterations值
  inline RequantizationTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前iterations值
  inline size_t iterations() const {
    return this->iterations_;
  }

  /*
   * 测试是否通过PO2对数进行准确的重新量化：
   * ((i - zero point) * 2**s)
   * - scale = exp2(-s)
   * - zero point在[0, 255]之间
   * - 没有输出截断
   * 只要((i - zero point) * 2**s)不溢出，则应确保生成精确的i。
   */
  void testExactDivideByPO2(pytorch_requantization_function requantize) const {
    ASSERT_GE(zeroPoint(), 0);  // 确保zeroPoint不小于0
    ASSERT_LE(zeroPoint(), 255);  // 确保zeroPoint不大于255

    /* 注意：需要s >= 1以确保scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);  // 确保s不小于1
    ASSERT_LT(s(), 32);  // 确保s小于32

    std::vector<int32_t> inputs(256);  // 创建256个元素的整数向量inputs
    std::vector<uint8_t> outputs(inputs.size());  // 创建与inputs相同大小的无符号字节向量outputs
    const int32_t maxI = (uint32_t(std::numeric_limits<int32_t>::max()) >> s()) + zeroPoint();  // 计算maxI的值
    const int32_t minI = -(-uint32_t(std::numeric_limits<int32_t>::min()) >> s()) + zeroPoint();  // 计算minI的值
    for (int32_t i = 0; i < 256; i++) {
      const int32_t clampedI = std::max(minI, std::min(maxI, i));  // 确保i在[minI, maxI]范围内
      inputs[i] = int32_t(uint32_t(clampedI - zeroPoint()) << s());  // 计算并存储inputs[i]的值
    }
    // 调用外部函数requantize进行重新量化操作
    requantize(
        inputs.size(),
        inputs.data(),
        scale(),
        zeroPoint(),
        qmin(),
        qmax(),
        outputs.data());
    for (int32_t i = 0; i < 256; i++) {
      const int32_t clampedI = std::max(minI, std::min(maxI, i));  // 确保i在[minI, maxI]范围内
      // 检查输出是否与预期相符
      ASSERT_EQ(clampedI, outputs[i])
          << "i = " << i << ", clamped i = " << clampedI << ", min i = " << minI
          << ", max i = " << maxI << ", s = " << s()
          << ", zero point = " << zeroPoint();
  /*
   * Test that requantization of numbers (i * 2**s + sign(i - zero point) *
   * 2**(s-1)) with
   * - scale = exp2(-s)
   * - zero point in [1, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not
   * overflow.
   */
  void testDivideByPO2WithRoundingUp(pytorch_requantization_function requantize) {
    ASSERT_GE(zeroPoint(), 0);  // 断言：确保 zeroPoint() 返回值大于等于 0
    ASSERT_LE(zeroPoint(), 255);  // 断言：确保 zeroPoint() 返回值小于等于 255

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);  // 断言：确保 s() 返回值大于等于 1
    ASSERT_LT(s(), 32);  // 断言：确保 s() 返回值小于 32

    std::vector<int32_t> inputs(256);  // 创建一个大小为 256 的 int32_t 类型向量 inputs
    std::vector<uint8_t> outputs(inputs.size());  // 创建一个与 inputs 大小相同的 uint8_t 类型向量 outputs
    for (int32_t i = 0; i < 256; i++) {  // 循环遍历 i 从 0 到 255
      const int64_t input =
          RequantizationTester::shiftLeft(i - zeroPoint(), s()) -
          (INT64_C(1) << (s() - 1)) + (int64_t)(i <= zeroPoint());  // 计算 input 值
      inputs[i] = int32_t(input);  // 将计算得到的 input 转换为 int32_t 并赋值给 inputs[i]
    }
    requantize(
        inputs.size(),  // 调用 requantize 函数，传递 inputs 的大小作为参数
        inputs.data(),  // 传递 inputs 数据的指针作为参数
        scale(),  // 传递 scale() 返回值作为参数
        zeroPoint(),  // 传递 zeroPoint() 返回值作为参数
        qmin(),  // 传递 qmin() 返回值作为参数
        qmax(),  // 传递 qmax() 返回值作为参数
        outputs.data());  // 传递 outputs 数据的指针作为参数
    for (int32_t i = 0; i < 256; i++) {  // 循环遍历 i 从 0 到 255
      const int64_t input =
          RequantizationTester::shiftLeft(i - zeroPoint(), s()) -
          (INT64_C(1) << (s() - 1)) + (int64_t)(i <= zeroPoint());  // 重新计算 input 值
      if (int32_t(input) == input) {  // 如果 input 转换为 int32_t 与原始 input 相等
        ASSERT_EQ(i, uint32_t(outputs[i]))  // 断言：确保 i 等于 outputs[i] 的无符号 32 位整数形式
            << "i = " << i << ", input = " << input << ", s = " << s()  // 输出调试信息
            << ", zero point = " << zeroPoint();
      }
    }
  }

  /*
   * Test that requantization of numbers (i * 2**s + sign(i - zero point) *
   * 2**(s-1)) with
   * - scale = exp2(-s)
   * - zero point in [1, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not
   * overflow.
   */
  void testDivideByPO2WithRoundingDown(pytorch_requantization_function requantize) {
    ASSERT_GE(zeroPoint(), 0);  // 断言：确保 zeroPoint() 返回值大于等于 0
    ASSERT_LE(zeroPoint(), 255);  // 断言：确保 zeroPoint() 返回值小于等于 255

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);  // 断言：确保 s() 返回值大于等于 1
    ASSERT_LT(s(), 32);  // 断言：确保 s() 返回值小于 32

    std::vector<int32_t> inputs(256);  // 创建一个大小为 256 的 int32_t 类型向量 inputs
    std::vector<uint8_t> outputs(inputs.size());  // 创建一个与 inputs 大小相同的 uint8_t 类型向量 outputs
    for (int32_t i = 0; i < 256; i++) {  // 循环遍历 i 从 0 到 255
      const int64_t input =
          RequantizationTester::shiftLeft(i - zeroPoint(), s()) +
          (INT64_C(1) << (s() - 1)) - (int64_t)(i >= zeroPoint());  // 计算 input 值
      inputs[i] = int32_t(input);  // 将计算得到的 input 转换为 int32_t 并赋值给 inputs[i]
    }
    requantize(
        inputs.size(),  // 调用 requantize 函数，传递 inputs 的大小作为参数
        inputs.data(),  // 传递 inputs 数据的指针作为参数
        scale(),  // 传递 scale() 返回值作为参数
        zeroPoint(),  // 传递 zeroPoint() 返回值作为参数
        qmin(),  // 传递 qmin() 返回值作为参数
        qmax(),  // 传递 qmax() 返回值作为参数
        outputs.data());  // 传递 outputs 数据的指针作为参数
    for (int32_t i = 0; i < 256; i++) {  // 循环遍历 i 从 0 到 255
      const int64_t input =
          RequantizationTester::shiftLeft(i - zeroPoint(), s()) +
          (INT64_C(1) << (s() - 1)) - (int64_t)(i >= zeroPoint());  // 重新计算 input 值
      if (int32_t(input) == input) {  // 如果 input 转换为 int32_t 与原始 input 相等
        ASSERT_EQ(i, uint32_t(outputs[i]))  // 断言：确保 i 等于 outputs[i] 的无符号 32 位整数形式
            << "i = " << i << ", input = " << input << ", s = " << s()  // 输出调试信息
            << ", zero point = " << zeroPoint();
      }
    }
  }

  void testDivideByPO2WithRoundingAway(pytorch_requantization_function requantize) {
    // 确保 zeroPoint() 返回的值大于等于 0
    ASSERT_GE(zeroPoint(), 0);
    // 确保 zeroPoint() 返回的值小于等于 255
    ASSERT_LE(zeroPoint(), 255);

    /* 注意：需要 s() >= 1 确保 scale = exp2(-s) < 1.0 */
    // 确保 s() 返回的值大于等于 1
    ASSERT_GE(s(), 1);
    // 确保 s() 返回的值小于 32
    ASSERT_LT(s(), 32);

    // 创建一个包含 256 个元素的 int32_t 类型的向量
    std::vector<int32_t> inputs(256);
    // 创建一个包含 inputs 大小的 uint8_t 类型的向量
    std::vector<uint8_t> outputs(inputs.size());
    // 遍历 0 到 255 的整数
    for (int32_t i = 0; i < 256; i++) {
      // 计算输入值，使用 RequantizationTester 的 shiftLeft 函数
      int64_t input = RequantizationTester::shiftLeft(i - zeroPoint(), s());
      // 调整输入值，使其在范围内
      if (input > 0) {
        input -= INT64_C(1) << (s() - 1);
      } else if (input < 0) {
        input += INT64_C(1) << (s() - 1);
      }
      // 将计算后的输入值存入 inputs 向量中
      inputs[i] = int32_t(input);
    }
    // 调用 requantize 函数进行重新量化处理
    requantize(
        inputs.size(),
        inputs.data(),
        scale(),
        zeroPoint(),
        qmin(),
        qmax(),
        outputs.data());
    // 再次遍历 0 到 255 的整数
    for (uint32_t i = 0; i < 256; i++) {
      // 计算输入值，使用 RequantizationTester 的 shiftLeft 函数
      int64_t input = RequantizationTester::shiftLeft(i - zeroPoint(), s());
      // 调整输入值，使其在范围内
      if (input > 0) {
        input -= INT64_C(1) << (s() - 1);
      } else if (input < 0) {
        input += INT64_C(1) << (s() - 1);
      }
      // 如果输入值能转换为 int32_t，则进行断言检查
      if (int32_t(input) == input) {
        ASSERT_EQ(i, uint32_t(outputs[i]))
            << "i = " << i << ", input = " << input << ", s = " << s()
            << ", zero point = " << zeroPoint();
      }
    }
  }

  // 测试特殊情况的函数
  void testSpecialCases(pytorch_requantization_function requantize) {
    // 创建一个包含 256 个元素的 int32_t 类型的向量
    std::vector<int32_t> inputs(256);
    // 创建一个包含 inputs 大小的 uint8_t 类型的向量
    std::vector<uint8_t> outputs(inputs.size());

    // 将 inputs 向量填充为 int32_t 类型的最小值
    std::fill(
        inputs.begin(), inputs.end(), std::numeric_limits<int32_t>::min());
    // 遍历 zeroPoint 变量从 0 到 255
    for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
      // 调用 requantize 函数进行重新量化处理
      requantize(
          inputs.size(),
          inputs.data(),
          ldexpf(1.0f, -32) /* scale */,
          zeroPoint /* zero point */,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          outputs.data());
      // 断言最小输出值为零点减一的最大值
      ASSERT_EQ(
          std::max(int32_t(0), zeroPoint - 1),
          *std::min_element(outputs.cbegin(), outputs.cend()));
    }

    // 将 inputs 向量填充为 int32_t 类型的最大值
    std::fill(
        inputs.begin(), inputs.end(), std::numeric_limits<int32_t>::max());
    // 调用 requantize 函数进行重新量化处理
    requantize(
        inputs.size(),
        inputs.data(),
        0x1.FFFFFEp-1f /* scale */,
        std::numeric_limits<uint8_t>::max() /* zero point */,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        outputs.data());
    // 遍历输出向量，断言所有元素等于 uint8_t 类型的最大值
    for (size_t i = 0; i < inputs.size(); i++) {
      ASSERT_EQ(std::numeric_limits<uint8_t>::max(), outputs[i]);
    }
  }

  // 测试精确随机情况的函数
  void testRandomCasesPrecise(pytorch_requantization_function requantize) {
    // 创建一个随机设备对象
    std::random_device randomDevice;
    // 创建一个 Mersenne Twister 引擎对象
    std::mt19937 mtRng(randomDevice());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 迭代多次执行以下代码块，次数由 iterations() 函数确定

      // 使用 mt19937 引擎和 uniform_int_distribution 创建 rng 函数对象
      auto rng = std::bind(std::uniform_int_distribution<uint8_t>(), mtRng);

      // 创建长度为 4096 的 int32_t 类型向量 inputs 和 uint8_t 类型向量 outputs
      std::vector<int32_t> inputs(4096);
      std::vector<uint8_t> outputs(inputs.size());

      // 定义量化零点为 128
      const uint8_t zeroPoint = UINT8_C(128);

      // 创建范围在 [0x1.000000p-23f, 0x1.FFFFFEp-1f] 内的均匀分布
      std::uniform_real_distribution<float> scaleDistribution(
          0x1.000000p-23f, 0x1.FFFFFEp-1f);

      // 从分布中随机选择一个 scale 值
      const float scale = scaleDistribution(mtRng);

      // 填充 inputs 向量，将 rng() 的结果除以 scale，并转换为 int32_t 存储
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t approximateOutput = rng();
        const int32_t input =
            int32_t(double(approximateOutput) / double(scale));
        inputs[i] = input;
      }

      // 调用 requantize 函数进行重新量化操作
      requantize(
          inputs.size(),
          inputs.data(),
          scale,
          zeroPoint,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          outputs.data());

      /* 确保输出不全相同，因为这种情况下测试不会验证太多 */
      ASSERT_NE(
          *std::max_element(outputs.cbegin(), outputs.cend()),
          *std::min_element(outputs.cbegin(), outputs.cend()));

      // 验证每个输入值的精确重新量化结果是否与参考输出相等
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t referenceOutput = pytorch_scalar_requantize_precise(
            inputs[i],
            scale,
            zeroPoint,
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max());
        ASSERT_EQ(uint32_t(referenceOutput), uint32_t(outputs[i]));
      }
    }
  }

  // 使用 pytorch_requantization_function 类型作为参数的随机测试函数
  void testRandomCasesApproximate(pytorch_requantization_function requantize) {
    // 创建随机设备对象和 mt19937 引擎对象 mtRng
    std::random_device randomDevice;
    std::mt19937 mtRng(randomDevice());
    // 迭代次数由 iterations() 函数确定，执行指定次数的循环
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 创建随机数生成器 rng，生成范围为 uint8_t 类型的随机数
      auto rng = std::bind(std::uniform_int_distribution<uint8_t>(), mtRng);

      // 创建长度为 4096 的整数向量 inputs 和相同长度的字节向量 outputs
      std::vector<int32_t> inputs(4096);
      std::vector<uint8_t> outputs(inputs.size());

      // 设置 zeroPoint 为 128
      const uint8_t zeroPoint = UINT8_C(128);
      // 创建一个范围在 [2^-23, 2^-1] 的均匀分布的浮点数分布器
      std::uniform_real_distribution<float> scaleDistribution(
          0x1.000000p-23f, 0x1.FFFFFEp-1f);
      // 生成一个随机的 scale 值
      const float scale = scaleDistribution(mtRng);
      // 填充 inputs 向量，通过 rng 生成的随机数除以 scale 得到近似的输入值
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t approximateOutput = rng();
        const int32_t input =
            int32_t(double(approximateOutput) / double(scale));
        inputs[i] = input;
      }

      // 调用 requantize 函数，对 inputs 中的数据进行重新量化处理，将结果存入 outputs 中
      requantize(
          inputs.size(),
          inputs.data(),
          scale,
          zeroPoint,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          outputs.data());

      /* 确保输出不全相同，因为这种情况下测试不会有效验证 */
      ASSERT_NE(
          *std::max_element(outputs.cbegin(), outputs.cend()),
          *std::min_element(outputs.cbegin(), outputs.cend()));

      // 检查每个输出值与参考输出值之间的差异是否在允许范围内
      for (size_t i = 0; i < inputs.size(); i++) {
        const double referenceOutput =
            RequantizationTester::requantizeApproximate(
                inputs[i],
                scale,
                zeroPoint,
                std::numeric_limits<uint8_t>::min(),
                std::numeric_limits<uint8_t>::max());
        ASSERT_LE(fabs(referenceOutput - double(outputs[i])), 0.55)
            << "input = " << inputs[i] << ", output = " << uint32_t(outputs[i])
            << ", reference output = " << referenceOutput;
      }
    }
  }

  // 对随机情况进行测试，比较 requantize 和 requantizeReference 的结果
  void testRandomCasesAgainstReference(
      pytorch_requantization_function requantize,
      pytorch_requantization_function requantizeReference) {
    // 创建随机设备和随机数生成器 mtRng
    std::random_device randomDevice;
    std::mt19937 mtRng(randomDevice());
    // 迭代执行若干次，由 iterations() 返回的次数决定
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 创建一个 RNG，生成 uint8_t 类型的均匀分布随机数
      auto rng = std::bind(std::uniform_int_distribution<uint8_t>(), mtRng);

      // 创建一个大小为 4096 的 int32_t 向量作为输入
      std::vector<int32_t> inputs(4096);
      // 创建一个与 inputs 大小相同的 uint8_t 向量作为输出
      std::vector<uint8_t> outputs(inputs.size());
      // 创建一个与 inputs 大小相同的 uint8_t 向量作为参考输出
      std::vector<uint8_t> referenceOutputs(inputs.size());

      // 设置 zeroPoint 为 128
      const uint8_t zeroPoint = UINT8_C(128);
      // 创建一个 float 型的均匀实数分布，范围为 [0x1.000000p-23f, 0x1.FFFFFEp-1f]
      std::uniform_real_distribution<float> scaleDistribution(
          0x1.000000p-23f, 0x1.FFFFFEp-1f);
      // 从 scaleDistribution 中生成一个 scale
      const float scale = scaleDistribution(mtRng);
      // 遍历 inputs，根据 scale 计算每个输入的 int32_t 值并存入 inputs 向量
      for (size_t i = 0; i < inputs.size(); i++) {
        // 生成一个近似输出 approximateOutput
        const uint8_t approximateOutput = rng();
        // 根据 scale 计算输入 input
        const int32_t input =
            int32_t(double(approximateOutput) / double(scale));
        inputs[i] = input;
      }

      // 调用 requantize 函数对输入进行重新量化，输出存入 outputs 向量
      requantize(
          inputs.size(),
          inputs.data(),
          scale,
          zeroPoint,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          outputs.data());

      // 调用 requantizeReference 函数对输入进行重新量化，输出存入 referenceOutputs 向量
      requantizeReference(
          inputs.size(),
          inputs.data(),
          scale,
          zeroPoint,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          referenceOutputs.data());

      /* 确保输出不完全相同，因为这种情况下测试结果不具备验证性 */
      ASSERT_NE(
          *std::max_element(outputs.cbegin(), outputs.cend()),
          *std::min_element(outputs.cbegin(), outputs.cend()));

      // 检查每个输出元素是否与参考输出元素相等
      for (size_t i = 0; i < inputs.size(); i++) {
        ASSERT_EQ(uint32_t(referenceOutputs[i]), uint32_t(outputs[i]));
      }
    }
};



# 这行代码是一个空的代码块结束符号，用于结束一个代码块或语句块。
# 在这个特定的示例中，这行代码本身没有实际的功能，只是一个结束符号。
```