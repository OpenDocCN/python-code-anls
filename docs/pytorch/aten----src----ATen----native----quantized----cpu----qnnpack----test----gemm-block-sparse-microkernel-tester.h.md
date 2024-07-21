# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\gemm-block-sparse-microkernel-tester.h`

```
/*
 * 版权声明: Facebook, Inc. 及其关联公司版权所有。
 * 保留所有权利。
 *
 * 此源代码根据根目录下的 LICENSE 文件中的 BSD 风格许可证进行许可。
 */

#pragma once

#include <algorithm>  // 包含算法库，用于使用 STL 算法
#include <cassert>    // 包含断言库，用于编写断言
#include <cmath>      // 包含数学库，用于数学运算
#include <cstddef>    // 包含 cstddef 库，定义了 size_t 等
#include <cstdlib>    // 包含 cstdlib 库，定义了标准库函数
#include <functional> // 包含函数对象库，用于函数对象操作
#include <random>     // 包含随机数库，用于生成随机数
#include <vector>     // 包含向量库，用于使用 STL 向量容器

#include <fp16.h>                 // 包含 fp16 库，可能用于半精度浮点数支持
#include <pack_block_sparse.h>    // 包含块稀疏打包库头文件
#include <qnnpack/AlignedAllocator.h>  // 包含内存对齐分配器头文件
#include <qnnpack/params.h>       // 包含 QNNPACK 参数头文件
#include <qnnpack/requantization.h>   // 包含 QNNPACK 重新量化头文件

#define MAYBE_UNUSED __attribute__((unused))  // 定义一个可能未使用的宏

namespace {
  // 填充块稀疏权重函数
  void fillBlockSparseWeights(
      uint8_t* b,            // 目标数据指针
      size_t N,              // 输出尺寸 N
      size_t K,              // 输出尺寸 K
      size_t row_block_size, // 行块大小
      size_t col_block_size, // 列块大小
      float sparsity,        // 稀疏度
      const uint8_t* zero_points) {  // 零点数组指针
    std::random_device randomDevice;  // 随机设备
    auto rng = std::mt19937(randomDevice());  // Mersenne Twister 随机数引擎
    std::bernoulli_distribution dist{sparsity};  // 伯努利分布，用于生成稀疏的随机数
    for (uint32_t n = 0; n < N ; n += row_block_size) {  // 遍历输出尺寸 N
      for (uint32_t k = 0; k < K; k += col_block_size) {  // 遍历输出尺寸 K
        if (dist(rng)) {  // 根据稀疏度随机决定是否填充数据
          for (uint32_t nb = 0; (nb < row_block_size) && (n + nb < N); ++nb) {  // 遍历行块大小
            for (uint32_t kb = 0; (kb < col_block_size) && (k + kb < K); ++kb) {  // 遍历列块大小
              *(b + (n + nb) * K + k + kb) = zero_points[n + nb];  // 填充数据
            }
          }
        }
      }
    }
  }

  // 临时调试工具，稍后将删除
  MAYBE_UNUSED void printMatrix(const char* name, const uint8_t* a, const size_t M, const size_t N) {
    std::cout << "Matrix START:" << name << "...\n";  // 打印矩阵开始信息
    for (uint32_t m = 0; m < M ; ++m) {  // 遍历矩阵行数 M
      for (uint32_t n = 0; n < N; n++) {  // 遍历矩阵列数 N
        std::cout << (const uint32_t)(*(a + m * N + n)) << ", ";  // 打印矩阵元素
      }
      std::cout << std::endl;  // 换行
    }
    std::cout << "Matrix END...\n\n";  // 打印矩阵结束信息
  }

  MAYBE_UNUSED void printMatrix(const char* name, const float* a, const size_t M, const size_t N) {
    std::cout << "Matrix START:" << name << "...\n";  // 打印矩阵开始信息
    for (uint32_t m = 0; m < M ; ++m) {  // 遍历矩阵行数 M
      for (uint32_t n = 0; n < N; n++) {  // 遍历矩阵列数 N
        std::cout << (*(a + m * N + n)) << ", ";  // 打印矩阵元素
      }
      std::cout << std::endl;  // 换行
    }
    std::cout << "Matrix END...\n\n";  // 打印矩阵结束信息
  }

}

// GemmBlockSparseMicrokernelTester 类定义
class GemmBlockSparseMicrokernelTester {
 public:
  // 设置 mr 大小，并返回当前对象引用
  inline GemmBlockSparseMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  // 获取当前 mr 大小
  inline size_t mr() const {
    return this->mr_;
  }

  // 设置 nr 大小，并返回当前对象引用
  inline GemmBlockSparseMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  // 获取当前 nr 大小
  inline size_t nr() const {
    return this->nr_;
  }

  // 设置 m 大小，并返回当前对象引用
  inline GemmBlockSparseMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  // 获取当前 m 大小
  inline size_t m() const {
    return this->m_;
  }

  // 设置 n 大小，并返回当前对象引用
  inline GemmBlockSparseMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  // 获取当前 n 大小
  inline size_t n() const {
    return this->n_;
  }

  // 设置 k 大小，并返回当前对象引用
  inline GemmBlockSparseMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  // 获取当前 k 大小
  inline size_t k() const {

    return this->k_;
  }

  // 待实现的测试函数，用于测试块稀疏的 GEMM 微内核
  void testGemmBlockSparseMicrokernel() {
    // 未实现测试函数内容
  }
};
  // 返回当前对象的成员变量 k_
  return this->k_;
}

// 设置并返回 ks_ 成员变量
inline GemmBlockSparseMicrokernelTester& ks(size_t ks) {
  this->ks_ = ks;
  return *this;
}

// 设置并返回 rowBlockSize_ 成员变量
inline GemmBlockSparseMicrokernelTester& rowBlockSize(size_t block_size) {
  this->rowBlockSize_ = block_size;
  return *this;
}

// 设置并返回 colBlockSize_ 成员变量
inline GemmBlockSparseMicrokernelTester& colBlockSize(size_t block_size) {
  this->colBlockSize_ = block_size;
  return *this;
}

// 设置并返回 sparsity_ 成员变量
inline GemmBlockSparseMicrokernelTester& sparsity(float s) {
  this->sparsity_ = s;
  return *this;
}

// 返回当前对象的成员变量 ks_
inline size_t ks() const {
  return this->ks_;
}

// 返回当前对象的成员变量 rowBlockSize_
inline size_t rowBlockSize() const {
  return this->rowBlockSize_;
}

// 返回当前对象的成员变量 colBlockSize_
inline size_t colBlockSize() const {
  return this->colBlockSize_;
}

// 返回当前对象的成员变量 sparsity_
inline float sparsity() const {
  return this->sparsity_;
}

// 计算并返回 biasN_ 成员变量的值
inline size_t biasN() const {
  return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
}

// 设置并返回 aStride_ 成员变量
inline GemmBlockSparseMicrokernelTester& aStride(size_t aStride) {
  this->aStride_ = aStride;
  return *this;
}

// 返回当前对象的成员变量 aStride_；如果 aStride_ 为 0，则返回 k() 的值
inline size_t aStride() const {
  return this->aStride_ == 0 ? k() : this->aStride_;
}

// 设置并返回 cStride_ 成员变量
inline GemmBlockSparseMicrokernelTester& cStride(size_t cStride) {
  this->cStride_ = cStride;
  return *this;
}

// 返回当前对象的成员变量 cStride_；如果 cStride_ 为 0，则返回 n() 的值
inline size_t cStride() const {
  return this->cStride_ == 0 ? n() : this->cStride_;
}

// 设置并返回 aZeroPoint_ 成员变量
inline GemmBlockSparseMicrokernelTester& aZeroPoint(uint8_t aZeroPoint) {
  this->aZeroPoint_ = aZeroPoint;
  return *this;
}

// 返回当前对象的成员变量 aZeroPoint_
inline uint8_t aZeroPoint() const {
  return this->aZeroPoint_;
}

// 设置并返回 bZeroPoint_ 成员变量
inline GemmBlockSparseMicrokernelTester& bZeroPoint(uint8_t bZeroPoint) {
  this->bZeroPoint_ = bZeroPoint;
  return *this;
}

// 返回当前对象的成员变量 bZeroPoint_
inline uint8_t bZeroPoint() const {
  return this->bZeroPoint_;
}

// 设置并返回 multiplier_ 成员变量
inline GemmBlockSparseMicrokernelTester& multiplier(const float multiplier) {
  this->multiplier_ = multiplier;
  return *this;
}

// 返回当前对象的成员变量 multiplier_
inline float multiplier() const {
  return this->multiplier_;
}

// 设置并返回 qmin_ 成员变量
inline GemmBlockSparseMicrokernelTester& qmin(uint8_t qmin) {
  this->qmin_ = qmin;
  return *this;
}

// 返回当前对象的成员变量 qmin_
inline uint8_t qmin() const {
  return this->qmin_;
}

// 设置并返回 qmax_ 成员变量
inline GemmBlockSparseMicrokernelTester& qmax(uint8_t qmax) {
  this->qmax_ = qmax;
  return *this;
}

// 返回当前对象的成员变量 qmax_
inline uint8_t qmax() const {
  return this->qmax_;
}

// 设置并返回 iterations_ 成员变量
inline GemmBlockSparseMicrokernelTester& iterations(size_t iterations) {
  this->iterations_ = iterations;
  return *this;
}

// 返回当前对象的成员变量 iterations_
inline size_t iterations() const {
  return this->iterations_;
}

// 测试函数，接受一个函数指针参数 qgemm
void test(pytorch_q8gemm_dq_sparse_ukernel_function qgemm) const {
  // 断言 m() 不大于 mr()
  ASSERT_LE(m(), mr());
  // 断言 n() 不大于 nr()

  ASSERT_LE(n(), nr());

  // 随机数生成器初始化
  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  // 生成范围为 -10000 到 10000 的均匀分布的整数的随机数生成器
  auto s32rng =
      std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
  // 生成范围为 0 到 255 的均匀分布的无符号整数的随机数生成器
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  // 分配存储空间给向量 a，长度为 (m()-1)*aStride()+k()+8
  std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
  // 分配存储空间给向量 b，长度为 n()*k()
  std::vector<uint8_t> b(n() * k());
    std::vector<float, AlignedAllocator<float, 32>> bias(std::max<size_t>(8, n()));
    // 创建一个长度为 std::max<size_t>(8, n()) 的 float 类型向量，使用自定义的内存分配器 AlignedAllocator<float, 32>
    
    std::vector<float> c((m() - 1) * cStride() + n());
    // 创建一个长度为 (m() - 1) * cStride() + n() 的 float 类型向量
    
    std::vector<float> acc(m() * n());
    // 创建一个长度为 m() * n() 的 float 类型向量
    
    const uint8_t* aPtr = a.data();
    // 将 vector a 的数据作为指向常量 uint8_t 类型的指针 aPtr
    
    }
  }

  template <typename SPARSE_INDICES_DTYPE, typename GEMM_UKERNEL_DTYPE>
  void test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_function packa,
      GEMM_UKERNEL_DTYPE qgemm) const {
    // 断言 m() 小于或等于 mr()，以确保条件满足
    ASSERT_LE(m(), mr());
    // 断言 n() 小于或等于 nr()，以确保条件满足
    ASSERT_LE(n(), nr());

    // 创建一个随机设备对象
    std::random_device randomDevice;
    // 创建一个基于随机设备的 Mersenne Twister 伪随机数生成器
    auto rng = std::mt19937(randomDevice());
    // 创建一个返回随机 int32_t 范围在 -10000 到 10000 之间的函数对象
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    // 创建一个返回随机 uint8_t 范围的函数对象
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 创建一个长度为 (m() - 1) * aStride() + k() + 8 的 uint8_t 类型向量 a
    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    // 创建一个长度为 n() * k() 的 uint8_t 类型向量 b
    std::vector<uint8_t> b(n() * k());
    // 创建一个长度为 std::max<size_t>(8, n()) 的 float 类型向量 bias，使用自定义的内存分配器 AlignedAllocator<float, 32>
    std::vector<float, AlignedAllocator<float, 32>> bias(std::max<size_t>(8, n()));
    // 创建一个长度为 (m() - 1) * cStride() + n() 的 float 类型向量 c
    std::vector<float> c((m() - 1) * cStride() + n());
    // 创建一个长度为 m() * n() 的 float 类型向量 acc
    std::vector<float> acc(m() * n());
    // 计算 m_blocks，表示在 m 维度上块的数量
    auto m_blocks = (m() + mr()  - 1) / mr();
    
    // While colBlockSize() is what kr is, we reuse 8x4/4x4 packing kernels
    // and thus a_packed has to be allocated accordingly.
    // 设置 kr_value 为 4
    const uint32_t kr_value = 4;
    // 计算 k_blocks，表示在 k 维度上块的数量
    auto k_blocks = (k() + kr_value  - 1) / kr_value;
    // 创建一个长度为 m_blocks * k_blocks * mr() * kr_value + 8 的 uint8_t 类型向量 a_packed，并初始化为 0
    std::vector<uint8_t> a_packed((m_blocks * k_blocks * mr() * kr_value) + 8, 0);

    // 将 vector a 的数据作为指向常量 uint8_t 类型的指针 aPtr
    
    }
  }

 private:
  // 私有成员变量定义
  size_t mr_{1};
  size_t nr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  size_t ks_{1};
  size_t aStride_{0};
  size_t cStride_{0};
  size_t rowBlockSize_{1};
  size_t colBlockSize_{4};
  uint8_t aZeroPoint_{0};
  uint8_t bZeroPoint_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{10};
  float multiplier_{2.0f};
  float sparsity_{0.7f};
};


注释：


# 这是一个空的代码块，只包含一个右花括号 '}'。
```