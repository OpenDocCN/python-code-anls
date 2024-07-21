# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\gemm-microkernel-tester.h`

```
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码采用BSD风格许可证授权，许可证文件可在此源代码根目录中的LICENSE文件中找到。
 */

#pragma once

#include <algorithm>        // 包含标准库算法头文件
#include <cassert>          // 包含标准库断言头文件
#include <cmath>            // 包含标准库数学函数头文件
#include <cstddef>          // 包含标准库stddef头文件
#include <cstdlib>          // 包含标准库通用工具函数头文件
#include <functional>       // 包含标准库函数对象头文件
#include <random>           // 包含标准库随机数生成头文件
#include <vector>           // 包含标准库向量头文件

#include <fp16.h>           // 包含fp16头文件

#include <qnnpack/AlignedAllocator.h>    // 包含QNNPACK库中的内存对齐分配器头文件
#include <qnnpack/pack.h>                // 包含QNNPACK库中的打包函数头文件
#include <qnnpack/params.h>              // 包含QNNPACK库中的参数头文件
#include <qnnpack/requantization.h>      // 包含QNNPACK库中的重新量化函数头文件

class GemmMicrokernelTester {
 public:
  inline GemmMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline GemmMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline GemmMicrokernelTester& np(size_t np) {
    this->np_ = np;
    return *this;
  }

  inline size_t np() const {
    return this->np_;
  }

  inline GemmMicrokernelTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  inline size_t kr() const {
    return this->kr_;
  }

  inline GemmMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline GemmMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline GemmMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline GemmMicrokernelTester& ks(size_t ks) {
    this->ks_ = ks;
    return *this;
  }

  inline size_t ks() const {
    return this->ks_;
  }

  inline size_t packedK() const {
    return k() % kr() == 0 ? k() : (k() / kr() + 1) * kr();
  }

  inline size_t packedN() const {
    return n() % np() == 0 ? n() : (n() / np() + 1) * np();
  }

  inline size_t biasN() const {
    return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
  }

  inline GemmMicrokernelTester& aStride(size_t aStride) {
    this->aStride_ = aStride;
    return *this;
  }

  inline size_t aStride() const {
    return this->aStride_ == 0 ? k() : this->aStride_;
  }

  inline GemmMicrokernelTester& cStride(size_t cStride) {
    this->cStride_ = cStride;
    return *this;
  }

  inline size_t cStride() const {
    return this->cStride_ == 0 ? n() : this->cStride_;
  }

  inline GemmMicrokernelTester& aZeroPoint(uint8_t aZeroPoint) {
    this->aZeroPoint_ = aZeroPoint;
    return *this;
  }

  inline uint8_t aZeroPoint() const {
    return this->aZeroPoint_;
  }

  inline GemmMicrokernelTester& bZeroPoint(uint8_t bZeroPoint) {
    this->bZeroPoint_ = bZeroPoint;
    return *this;
  }

  inline uint8_t bZeroPoint() const {
    return this->bZeroPoint_;
  }

  inline GemmMicrokernelTester& multiplier(const float multiplier) {
    this->multiplier_ = multiplier;
    return *this;
  }

  inline float multiplier() const {
    // 返回当前对象的乘数值
    return this->multiplier_;
  }

  // 设置并返回当前对象的qmin值
  inline GemmMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 返回当前对象的qmin值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置并返回当前对象的qmax值
  inline GemmMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回当前对象的qmax值
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置并返回当前对象的迭代次数
  inline GemmMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前对象的迭代次数
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 执行测试函数，验证条件和随机数据生成
  void test(pytorch_q8gemm_ukernel_function qgemm) const {
    // 断言验证矩阵维度条件
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());

    // 随机数生成器初始化
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    auto f32rng =
        std::bind(std::uniform_real_distribution<float>(1, 5), rng);

    // 分配和初始化输入矩阵和向量
    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * k());
    std::vector<int32_t> bias(n());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedW(
        packedN() * packedK() + biasN() * sizeof(uint32_t) / sizeof(uint8_t));
    std::vector<uint8_t> c((m() - 1) * cStride() + n());
    std::vector<int32_t> acc(m() * n());
    std::vector<uint8_t> cRef(m() * n());

    // 设置aPtr指向矩阵a的偏移位置
    const uint8_t* aPtr = a.data() + 8;

    // 迭代执行多次测试
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 生成随机数据填充矩阵a, b, bias和初始值为0xA5的矩阵c
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0xA5);

      // 初始化packedW并填充零点
      std::fill(packedW.begin(), packedW.end(), bZeroPoint());

      // 填充kernel_zero_points并生成随机数据
      size_t num_zero_points_padded = n() + 8;
      std::vector<uint8_t> kernel_zero_points
        (num_zero_points_padded, bZeroPoint());
      std::generate(kernel_zero_points.begin(), kernel_zero_points.end(), std::ref(u8rng));
      pytorch_pack_q8gemm_w(
          n(),
          k(),
          nr(),
          np(),
          kr(),
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
    // 如果不是在运行时使用 QNNPACK 运行时量化，则执行以下内容
    aZeroPoint(),  // 获取 a 的零点量化参数
    bZeroPoint(),  // 获取 b 的零点量化参数
#endif
    b.data(),       // 获取 b 的数据
    bias.data(),    // 获取偏置项的数据
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
    kernel_zero_points.data(),
    // 如果是在运行时使用 QNNPACK 运行时量化，则获取 kernel_zero_points 的数据
#endif
#endif
          packedW.data());

      // 断言：确保数组 a 中的最大元素不等于最小元素
      ASSERT_NE(
          *std::max_element(a.cbegin(), a.cend()),
          *std::min_element(a.cbegin(), a.cend()));
      // 断言：确保数组 b 中的最大元素不等于最小元素
      ASSERT_NE(
          *std::max_element(b.cbegin(), b.cend()),
          *std::min_element(b.cbegin(), b.cend()));

      /* Compute 32-bit results and output quantization arguments */
      // 将数组 acc 所有元素置零
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kIndex = 0; kIndex < k(); kIndex++) {
            // 断言：确保 n() 不超过 packedN()
            ASSERT_LE(n(), packedN());
            // 断言：确保 acc 数组索引有效
            ASSERT_LT(mIndex * n() + nIndex, acc.size());
            // 断言：确保 a 数组索引有效
            ASSERT_LT(mIndex * k() + kIndex, a.size());
            // 计算量化后的乘积并累加到 acc 中
            acc[mIndex * n() + nIndex] +=
                (int32_t(aPtr[mIndex * aStride() + kIndex]) -
                 int32_t(aZeroPoint())) *
                (int32_t(b[nIndex * k() + kIndex]) - int32_t(kernel_zero_points[nIndex]));
          }
          // 添加偏置值到 acc 中
          acc[mIndex * n() + nIndex] += bias[nIndex];
        }
      }

      // 计算 acc 数组中的最小值和最大值
      const int32_t accMin = *std::min_element(acc.cbegin(), acc.cend());
      const int32_t accMax = *std::max_element(acc.cbegin(), acc.cend());
      // 如果 M x N >= 3，则断言：确保 acc 的最大值不等于最小值
      if (m() * n() >= 3) {
        ASSERT_NE(accMax, accMin)
            << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }

      // 计算缩放因子 cScale 和零点值 cZeroPoint
      const double cScale = uint32_t(accMax - accMin) >= 256
          ? double(uint32_t(accMax - accMin)) / 255.0
          : 1.00001;
      const uint8_t cZeroPoint = uint8_t(std::max(
          std::min(
              lrint(127.5 - 0.5 * double(accMin + accMax) / cScale),
              long(std::numeric_limits<uint8_t>::max())),
          long(std::numeric_limits<uint8_t>::min())));

      // 生成重新量化的比例尺
      std::vector<float> requantization_scales(num_zero_points_padded);
      auto scale_generator = [&]() -> float {return (f32rng()/cScale);};
      std::generate(
          requantization_scales.begin(),
          requantization_scales.end(),
          std::ref(scale_generator));
      // 计算卷积量化参数
      const union pytorch_qnnp_conv_quantization_params quantizationParams =
          pytorch_qnnp_compute_conv_quantization_params(
              aZeroPoint(),
              kernel_zero_points.data(),
              requantization_scales.data(),
              cZeroPoint,
              qmin(),
              qmax());
      // 计算标量重新量化参数
      const union pytorch_qnnp_fp32_requantization_params
          scalarRequantizationParams =
              pytorch_qnnp_compute_scalar_fp32_requantization_params(
                  requantization_scales.data(), cZeroPoint, qmin(), qmax());

      // 调用 QGEMM 函数执行量化矩阵乘法
      qgemm(
          m(),
          n(),
          k(),
          aPtr,
          aStride() * sizeof(uint8_t),
          packedW.data(),
          c.data(),
          cStride() * sizeof(uint8_t),
          0,
          &quantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
      // 如果目标架构为 ARM，使用量化神经网络包中的特定函数进行浮点数重量化
      // 否则，使用通用的浮点数重量化函数
      #if defined(__arm__) || defined(_M_ARM)
          cRef[mIndex * n() + nIndex] = pytorch_qnnp_fp32_requantize_magic(
              acc[mIndex * n() + nIndex], scalarRequantizationParams, nIndex);
      #else
          cRef[mIndex * n() + nIndex] = pytorch_qnnp_fp32_requantize(
              acc[mIndex * n() + nIndex], scalarRequantizationParams, nIndex);
      #endif

        }
      }

      // 对计算结果进行断言，确保在量化范围内
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_LE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmin()));
          ASSERT_EQ(
              uint32_t(c[mIndex * cStride() + nIndex]),
              uint32_t(cRef[mIndex * n() + nIndex]))
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << (uint32_t)cRef[mIndex * n() + nIndex]
              << " (accumulator = " << acc[mIndex * n() + nIndex]
              << "), optimized = " << (uint32_t)c[mIndex * cStride() + nIndex]
              << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k()
              << ", requantization scale = " << requantization_scales[nIndex]
              << ", output zero point = " << int32_t(cZeroPoint);
        }
      }
    }
  }

  // 执行测试函数，验证实现的正确性
  void test(pytorch_q8gemm_dq_ukernel_function qgemm) const {
    // 断言输入矩阵维度在有效范围内
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());

    // 随机数生成器初始化
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 分配并初始化输入矩阵 a, b, bias, packedW, c, acc
    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * k());
    std::vector<float, AlignedAllocator<float, 32>> bias(std::max<size_t>(8, n()));
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedW(
        packedN() * packedK() + biasN() * sizeof(uint32_t) / sizeof(uint8_t));
    std::vector<float> c((m() - 1) * cStride() + n());
    std::vector<float> acc(m() * n());

    // 设置指向 a 的指针，略过填充字节
    const uint8_t* aPtr = a.data() + 8;

    // 执行多次迭代进行测试
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 生成随机数据填充矩阵 a, b, bias
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      // 将矩阵 c 初始化为零
      std::fill(c.begin(), c.end(), 0.0f);

      // 将 packedW 初始化为 bZeroPoint()
      std::fill(packedW.begin(), packedW.end(), bZeroPoint());

      // 生成随机数据填充 kernel_zero_points
      size_t num_zero_points_padded = n() + 8;
      std::vector<uint8_t> kernel_zero_points
        (num_zero_points_padded, bZeroPoint());
      std::generate(kernel_zero_points.begin(), kernel_zero_points.end(), std::ref(u8rng));
      
      // 调用矩阵打包函数，准备进行量化卷积计算
      pytorch_pack_q8gemm_w(
          n(),
          k(),
          nr(),
          np(),
          kr(),
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          aZeroPoint(),                          // 如果未启用运行时量化，则使用a的零点
          bZeroPoint(),                          // 如果未启用运行时量化，则使用b的零点
#endif
          b.data(),                              // 获取b的数据指针
          nullptr,
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          kernel_zero_points.data(),              // 如果启用了运行时量化，则使用kernel的零点数组
#endif
          packedW.data());                       // 获取packedW的数据指针

      ASSERT_NE(
          *std::max_element(a.cbegin(), a.cend()), // 断言：a中的最大值不等于最小值
          *std::min_element(a.cbegin(), a.cend())); // 断言：a中的最小值不等于最大值
      ASSERT_NE(
          *std::max_element(b.cbegin(), b.cend()), // 断言：b中的最大值不等于最小值
          *std::min_element(b.cbegin(), b.cend())); // 断言：b中的最小值不等于最大值

      auto f32rng =
          std::bind(std::uniform_real_distribution<float>(1, 5), rng); // 创建一个均匀分布在[1, 5]之间的随机数生成器
      std::vector<float> dequantization_scales(num_zero_points_padded); // 创建存储反量化比例的向量
      std::generate(
          dequantization_scales.begin(),
          dequantization_scales.end(),
          std::ref(f32rng)); // 使用f32rng生成反量化比例向量中的值
      /* Compute 32-bit results and output quantization arguments */
      std::fill(acc.begin(), acc.end(), 0); // 将acc向量的所有元素填充为0
      for (size_t mIndex = 0; mIndex < m(); mIndex++) { // 循环遍历m维度
        for (size_t nIndex = 0; nIndex < n(); nIndex++) { // 循环遍历n维度
          for (size_t kIndex = 0; kIndex < k(); kIndex++) { // 循环遍历k维度
            ASSERT_LE(n(), packedN()); // 断言：n维度小于等于packedN()
            ASSERT_LT(mIndex * n() + nIndex, acc.size()); // 断言：mIndex*n() + nIndex小于acc向量的大小
            ASSERT_LT(mIndex * k() + kIndex, a.size()); // 断言：mIndex*k() + kIndex小于a向量的大小
            acc[mIndex * n() + nIndex] +=
                (int32_t(aPtr[mIndex * aStride() + kIndex]) - // 将aPtr中的数据减去a的零点
                 int32_t(aZeroPoint())) *
                (int32_t(b[nIndex * k() + kIndex]) - int32_t(kernel_zero_points[nIndex])); // 将b中的数据减去kernel的零点，并累加到acc中
          }
          acc[mIndex * n() + nIndex] =
            acc[mIndex * n() + nIndex] *
            dequantization_scales[nIndex] +
            bias[nIndex]; // 将acc中的值乘以反量化比例，并加上偏置值
        }
      }

      const struct pytorch_qnnp_conv_dynamic_quantization_params quantizationParams{
        aZeroPoint(),                             // 使用a的零点
        kernel_zero_points.data(),                 // 使用kernel的零点数组
        dequantization_scales.data(),             // 使用反量化比例数组
      };

      qgemm(
          m(),
          n(),
          k(),
          aPtr,
          aStride() * sizeof(uint8_t),
          packedW.data(),
          bias.data(),
          c.data(),
          cStride(),
          0,
          &quantizationParams);                    // 执行量化矩阵乘法，并传递量化参数

      for (size_t mIndex = 0; mIndex < m(); mIndex++) { // 循环遍历m维度
        for (size_t nIndex = 0; nIndex < n(); nIndex++) { // 循环遍历n维度
          ASSERT_NEAR(
              c[mIndex * cStride() + nIndex],       // 断言：c中的值接近于acc中的值
              acc[mIndex * n() + nIndex],
              std::abs(acc[mIndex * n() + nIndex]) * 1.0e-4f) // 允许的误差边界为acc中值的绝对值乘以1.0e-4
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << acc[mIndex * n() + nIndex]
              << ", optimized = " << c[mIndex * cStride() + nIndex]
              << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  void test(pytorch_q8conv_ukernel_function qconv) const {
    ASSERT_LE(m(), mr()); // 断言：m维度小于等于mr维度
    ASSERT_LE(n(), nr()); // 断言：n维度小于等于nr维度
    ASSERT_GE(k(), kr()); // 断言：k维度大于等于kr维度

    std::random_device randomDevice; // 创建随机设备
    auto rng = std::mt19937(randomDevice()); // 使用随机设备创建随机数生成器
    // 创建一个返回范围为[-10000, 10000]的随机整数的函数对象，并绑定到rng上
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    
    // 创建一个返回范围为[0, 255]的随机无符号8位整数的函数对象，并绑定到rng上
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    
    // 创建一个返回范围为[1, 5]的随机单精度浮点数的函数对象，并绑定到rng上
    auto f32rng =
        std::bind(std::uniform_real_distribution<float>(1, 5), rng);

    // 创建一个大小为((mr() - 1) * aStride() + k() + 8)的无符号8位整数向量a
    std::vector<uint8_t> a((mr() - 1) * aStride() + k() + 8);
    
    // 创建一个大小为(n() * ks() * k())的无符号8位整数向量b
    std::vector<uint8_t> b(n() * ks() * k());
    
    // 创建一个带有32字节对齐分配器的无符号8位整数向量packedW，
    // 大小为(ks() * packedN() * packedK() + biasN() * sizeof(uint32_t) / sizeof(uint8_t))
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedW(
        ks() * packedN() * packedK() +
        biasN() * sizeof(uint32_t) / sizeof(uint8_t));
    
    // 创建一个大小为n()的32位整数向量bias
    std::vector<int32_t> bias(n());
    
    // 创建一个大小为((m() - 1) * cStride() + n())的无符号8位整数向量c
    std::vector<uint8_t> c((m() - 1) * cStride() + n());
    
    // 创建一个大小为(m() * n())的32位整数向量acc
    std::vector<int32_t> acc(m() * n());
    
    // 创建一个大小为(m() * n())的无符号8位整数向量cRef
    std::vector<uint8_t> cRef(m() * n());
    
    // 创建一个大小为(mr() * ks())的指向无符号8位整数指针数组im2col
    std::vector<const uint8_t*> im2col(mr() * ks());
    
    // 设置aPtr指向a.data() + 8处的常量无符号8位整数指针
    const uint8_t* aPtr = a.data() + 8;

    // 对每个迭代执行以下操作
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
        // 使用u8rng生成随机值填充向量a
        std::generate(a.begin(), a.end(), std::ref(u8rng));
        
        // 使用u8rng生成随机值填充向量b
        std::generate(b.begin(), b.end(), std::ref(u8rng));
        
        // 使用s32rng生成随机值填充向量bias
        std::generate(bias.begin(), bias.end(), std::ref(s32rng));
        
        // 使用0xA5填充向量c
        std::fill(c.begin(), c.end(), 0xA5);

        // 使用bZeroPoint()填充向量packedW
        std::fill(packedW.begin(), packedW.end(), bZeroPoint());

        // 创建大小为n() + 8的零点填充向量kernel_zero_points，并使用u8rng生成随机值
        size_t num_zero_points_padded = n() + 8;
        std::vector<uint8_t> kernel_zero_points
            (num_zero_points_padded, bZeroPoint());
        std::generate(kernel_zero_points.begin(), kernel_zero_points.end(), std::ref(u8rng));

        // 调用pytorch_pack_q8conv_w函数，传递参数n(), ks(), k(), np(), kr()
        // （此处省略后续参数）
        pytorch_pack_q8conv_w(
            n(),
            ks(),
            k(),
            np(),
            kr(),
            // 后续参数由于截断，未包含在此处注释中
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          aZeroPoint(),  // 调用函数 aZeroPoint()
          bZeroPoint(),  // 调用函数 bZeroPoint()
#endif
          b.data(),       // 获取 b 的数据指针
          bias.data(),    // 获取 bias 的数据指针
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          kernel_zero_points.data(),  // 获取 kernel_zero_points 的数据指针
#if defined(__arm__) || defined(_M_ARM)
          cRef[mIndex * n() + nIndex] = pytorch_qnnp_fp32_requantize_magic(
              acc[mIndex * n() + nIndex], scalarRequantizationParams, nIndex);  // 对 acc 中的数据进行特殊重新量化计算，适用于 ARM 平台
#else
          cRef[mIndex * n() + nIndex] = pytorch_qnnp_fp32_requantize(
              acc[mIndex * n() + nIndex], scalarRequantizationParams, nIndex);  // 对 acc 中的数据进行标准重新量化计算
#endif
        }
      }

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_LE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmax()));  // 断言确保 c 中的值不超过 qmax()
          ASSERT_GE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmin()));  // 断言确保 c 中的值不低于 qmin()
          ASSERT_EQ(
              uint32_t(c[mIndex * cStride() + nIndex]),
              uint32_t(cRef[mIndex * n() + nIndex]))
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << uint32_t(cRef[mIndex * n() + nIndex])
              << " (accumulator = " << acc[mIndex * n() + nIndex]
              << "), optimized = " << uint32_t(c[mIndex * cStride() + nIndex])
              << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k()
              << ", requantization scale = " << requantization_scales[nIndex]
              << ", output zero point = " << int32_t(cZeroPoint);  // 断言确保 c 的值与 cRef 中的值相等，并输出详细信息
        }
      }
    }
  }

  static void q8gemm_compute_row_sum(
      const uint8_t* a,
      size_t m,
      size_t k,
      size_t stride,
      const int32_t multiplier,
      int32_t* row_sum,
      pytorch_q8sum_rows_ukernel_function q8sum_rows) {
    const size_t block_size = 4;
    for (size_t block_start = 0; block_start < m; block_start += block_size) {
      q8sum_rows(
          a + block_start * stride,  // 计算块的起始位置
          std::min(block_size, m - block_start),  // 确保块的大小不超过剩余元素数量
          k,  // 矩阵的列数
          stride,  // 矩阵的跨度
          multiplier,  // 乘法器
          row_sum + block_start);  // 存储每行总和的数组位置
    }
  }

  void test(pytorch_q8gemm_xzp_ukernel_function qgemm) const {
    ASSERT_LE(m(), mr());  // 断言确保 m() <= mr()
    ASSERT_LE(n(), nr());  // 断言确保 n() <= nr()
    ASSERT_GE(k(), kr());  // 断言确保 k() >= kr()

    std::random_device randomDevice;  // 随机数设备
    auto rng = std::mt19937(randomDevice());  // 使用随机设备生成随机数引擎
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);  // 绑定生成指定范围内随机整数的函数
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);  // 绑定生成随机无符号整数的函数

    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);  // 长度为 (m()-1)*aStride()+k()+8 的向量 a
    std::vector<uint8_t> b(n() * k());  // 长度为 n()*k() 的向量 b
    std::vector<int32_t> bias(n());  // 长度为 n() 的偏置向量 bias
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedW(
        packedN() * packedK() + biasN() * sizeof(uint32_t) / sizeof(uint8_t));  // 使用特定对齐方式和分配器的 packedW 向量
    std::vector<int32_t> aRowSums(m());  // 长度为 m() 的行总和向量 aRowSums
    std::vector<uint8_t> c((m() - 1) * cStride() + n());  // 长度为 (m()-1)*cStride()+n() 的向量 c
    std::vector<int32_t> acc(m() * n());  // 长度为 m()*n() 的累加器向量 acc
    std::vector<uint8_t> cRef(m() * n());  // 长度为 m()*n() 的参考向量 cRef
    // 创建指向 a 的指针，偏移 8 个元素位置，指向 a.data() 的第 9 个元素
    const uint8_t* aPtr = a.data() + 8;

    // 循环迭代，执行 iterations() 次数
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // 使用 u8rng 函数对象生成随机数填充容器 a
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      // 使用 u8rng 函数对象生成随机数填充容器 b
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      // 使用 s32rng 函数对象生成随机数填充容器 bias
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));

      // 将 packedW 容器全部填充为 0
      std::fill(packedW.begin(), packedW.end(), 0);
      
      // 调用 pytorch_pack_swizzle_q8gemm_b 函数，对参数进行打包和调整
      pytorch_pack_swizzle_q8gemm_b(
          n(),    // n 的值作为参数传递
          k(),    // k 的值作为参数传递
          np(),   // np 的值作为参数传递
          kr(),   // kr 的值作为参数传递
          8,      // 固定值 8 作为参数传递
// 如果未定义宏PYTORCH_QNNPACK_RUNTIME_QUANTIZATION，则执行以下操作
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
      // 调用aZeroPoint()和bZeroPoint()函数，获取零点值
      aZeroPoint(),
      bZeroPoint(),
}
}

// 测试函数，接受pytorch_hgemm_ukernel_function类型的参数hgemm
void test(pytorch_hgemm_ukernel_function hgemm) const {
  // 断言：m()小于等于mr()，确保m维度不超过mr维度
  ASSERT_LE(m(), mr());
  // 断言：n()小于等于nr()，确保n维度不超过nr维度
  ASSERT_LE(n(), nr());
  // 断言：k()大于等于kr()，确保k维度不小于kr维度
  ASSERT_GE(k(), kr());
  // 断言：aStride()大于等于k()，确保a的跨度不小于k
  ASSERT_GE(aStride(), k());
  // 断言：cStride()大于等于n()，确保c的跨度不小于n

  ASSERT_GE(cStride(), n());

  // 随机数生成设备
  std::random_device randomDevice;
  // 创建一个绑定了fp16_ieee_from_fp32_value函数的rng对象，用于将float转换为fp16格式
  auto rng = std::bind(
      fp16_ieee_from_fp32_value,
      std::bind(
          std::uniform_real_distribution<float>(),
          std::mt19937(randomDevice())));

  // 创建长度为(m() - 1) * aStride() + k() + 4的uint16_t向量a
  std::vector<uint16_t> a((m() - 1) * aStride() + k() + 4);
  // 创建长度为n() * k()的uint16_t向量b
  std::vector<uint16_t> b(n() * k());
  // 使用AlignedAllocator分配对齐内存，创建长度为packedN() * packedK() + biasN()的uint16_t向量packedW
  std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> packedW(
      packedN() * packedK() + biasN());
  // 创建长度为n()的uint16_t向量bias
  std::vector<uint16_t> bias(n());
  // 创建长度为(mr() - 1) * cStride() + nr()的uint16_t向量c
  std::vector<uint16_t> c((mr() - 1) * cStride() + nr());
  // 创建长度为m() * n()的float向量cRef
  std::vector<float> cRef(m() * n());

  // 设置aPtr指向a.data() + 4
  const uint16_t* aPtr = a.data() + 4;

  // 定义pytorch_qnnp_fp16_clamping_params结构体clampingParams，并设置scale为0x3C00（1.0）
  struct pytorch_qnnp_fp16_clamping_params clampingParams;
  clampingParams.scale = UINT16_C(0x3C00) /* 1.0 */;
}

// 测试函数，接受pytorch_sgemm_ukernel_function类型的参数sgemm
void test(pytorch_sgemm_ukernel_function sgemm) const {
  // 断言同上，确保维度符合要求
  ASSERT_LE(m(), mr());
  ASSERT_LE(n(), nr());
  ASSERT_GE(k(), kr());
  ASSERT_GE(aStride(), k());
  ASSERT_GE(cStride(), n());

  // 随机数生成设备
  std::random_device randomDevice;
  // 创建一个绑定了std::uniform_real_distribution<float>的rng对象，用于生成随机数
  auto rng = std::bind(
      std::uniform_real_distribution<float>(), std::mt19937(randomDevice()));

  // 创建长度为(m() - 1) * aStride() + k()的float向量a
  std::vector<float> a((m() - 1) * aStride() + k());
  // 创建长度为n() * k()的float向量b
  std::vector<float> b(n() * k());
  // 使用AlignedAllocator分配对齐内存，创建长度为packedN() * packedK() + biasN()的float向量packedW
  std::vector<float, AlignedAllocator<float, 32>> packedW(
      packedN() * packedK() + biasN());
  // 创建长度为(mr() - 1) * cStride() + nr()的float向量c
  std::vector<float> c((mr() - 1) * cStride() + nr());
  // 创建长度为m() * n()的float向量cRef
  std::vector<float> cRef(m() * n());

}

// 测试函数，接受pytorch_sconv_ukernel_function类型的参数sconv
void test(pytorch_sconv_ukernel_function sconv) const {
  // 断言同上，确保维度符合要求
  ASSERT_LE(m(), mr());
  ASSERT_LE(n(), nr());
  ASSERT_GE(k(), kr());

  // 随机数生成设备
  std::random_device randomDevice;
  // 创建一个mt19937类型的rng对象，用于生成随机数
  auto rng = std::mt19937(randomDevice());
  // 创建一个绑定了std::uniform_real_distribution<float>的f32rng对象，用于生成随机数
  auto f32rng = std::bind(
      std::uniform_real_distribution<float>(), std::mt19937(randomDevice()));

  // 创建长度为(mr() - 1) * aStride() + k() + 8的float向量a
  std::vector<float> a((mr() - 1) * aStride() + k() + 8);
  // 创建长度为n() * ks() * k()的float向量b
  std::vector<float> b(n() * ks() * k());
  // 使用AlignedAllocator分配对齐内存，创建长度为ks() * packedK() * packedN() + biasN()的float向量packedW
  std::vector<float, AlignedAllocator<float, 32>> packedW(
      ks() * packedK() * packedN() + biasN());
  // 创建长度为n()的float向量bias
  std::vector<float> bias(n());
  // 创建长度为(m() - 1) * cStride() + n()的float向量c
  std::vector<float> c((m() - 1) * cStride() + n());
  // 创建长度为m() * n()的float向量cRef
  std::vector<float> cRef(m() * n());
  // 创建长度为mr() * ks()的const float指针向量im2col
}

private:
// 私有成员变量，设置默认值
size_t mr_{1};
size_t nr_{1};
size_t np_{1};
size_t kr_{1};
size_t m_{1};
size_t n_{1};
size_t k_{1};
size_t ks_{1};
size_t aStride_{0};
size_t cStride_{0};
uint8_t aZeroPoint_{127};
uint8_t bZeroPoint_{127};
uint8_t qmin_{0};
uint8_t qmax_{255};
size_t iterations_{15};
float multiplier_{2.0f};
};
```