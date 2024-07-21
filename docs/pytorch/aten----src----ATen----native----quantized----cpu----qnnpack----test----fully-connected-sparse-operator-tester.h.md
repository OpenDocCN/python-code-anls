# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\fully-connected-sparse-operator-tester.h`

```
/*
 * 版权所有（c）Facebook，Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的 LICENSE 文件中找到的 BSD 风格许可证进行许可。
 */

#pragma once

#include <algorithm>                // 标准库：包含常用算法
#include <cmath>                    // 标准库：包含数学函数
#include <cstddef>                  // 标准库：包含 size_t 定义
#include <cstdlib>                  // 标准库：包含常用函数
#include <functional>               // 标准库：包含函数对象
#include <random>                   // 标准库：包含随机数生成器
#include <vector>                   // 标准库：包含向量容器
#include <memory>                   // 标准库：包含智能指针

#include <pack_block_sparse.h>      // 外部库：包含块稀疏打包
#include <pytorch_qnnpack.h>        // 外部库：包含 PyTorch QNNPack
#include <qnnpack_func.h>           // 外部库：包含 QNNPack 函数
#include <qnnpack/AlignedAllocator.h> // 外部库：包含对齐内存分配器

#define MAYBE_UNUSED __attribute__((unused)) // 宏定义：表示未使用的变量

namespace {
  void fillBlockSparseWeights(
      uint8_t* b,
      size_t N,
      size_t K,
      size_t row_block_size,
      size_t col_block_size,
      float sparsity,
      const uint8_t* zero_points) {
    std::random_device randomDevice;   // 创建随机数设备对象
    auto rng = std::mt19937(randomDevice()); // 使用随机数设备创建梅森旋转算法引擎
    std::bernoulli_distribution dist{sparsity}; // 创建伯努利分布对象，用于生成稀疏性随机数
    // 遍历行块
    for (uint32_t n = 0; n < N ; n += row_block_size) {
      // 遍历列块
      for (uint32_t k = 0; k < K; k += col_block_size) {
        // 根据稀疏性随机数，设置块稀疏权重
        if (dist(rng)) {
          // 遍历行内的每个元素
          for (uint32_t nb = 0; (nb < row_block_size) && (n + nb < N); ++nb) {
            // 遍历列内的每个元素
            for (uint32_t kb = 0; (kb < col_block_size) && (k + kb < K); ++kb) {
              *(b + (n + nb) * K + k + kb) = zero_points[n + nb]; // 设置权重值为零点值
            }
          }
        }
      }
    }
  }

  // 临时调试工具，稍后将删除
  MAYBE_UNUSED void printMatrix(const char* name, const uint8_t* a, const size_t M, const size_t N) {
    std::cout << "Matrix START:" << name << "...\n"; // 输出矩阵开始信息
    // 遍历矩阵行
    for (uint32_t m = 0; m < M ; ++m) {
      // 遍历矩阵列
      for (uint32_t n = 0; n < N; n++) {
        std::cout << (const uint32_t)(*(a + m * N + n)) << ", "; // 输出矩阵元素值
      }
      std::cout << std::endl; // 换行
    }
    std::cout << "Matrix END...\n\n"; // 输出矩阵结束信息
  }

  MAYBE_UNUSED void printMatrix(const char* name, const float* a, const size_t M, const size_t N) {
    std::cout << "Matrix START:" << name << "...\n"; // 输出矩阵开始信息
    // 遍历矩阵行
    for (uint32_t m = 0; m < M ; ++m) {
      // 遍历矩阵列
      for (uint32_t n = 0; n < N; n++) {
        std::cout << (*(a + m * N + n)) << ", "; // 输出矩阵元素值
      }
      std::cout << std::endl; // 换行
    }
    std::cout << "Matrix END...\n\n"; // 输出矩阵结束信息
  }

}

class FullyConnectedSparseOperatorTester {
 public:
  inline FullyConnectedSparseOperatorTester& inputChannels(size_t inputChannels) {
    assert(inputChannels >= 1); // 断言：输入通道数大于等于1
    this->inputChannels_ = inputChannels; // 设置输入通道数
    return *this;
  }

  inline size_t inputChannels() const {
    return this->inputChannels_; // 返回输入通道数
  }

  inline FullyConnectedSparseOperatorTester& outputChannels(size_t outputChannels) {
    assert(outputChannels >= 1); // 断言：输出通道数大于等于1
    this->outputChannels_ = outputChannels; // 设置输出通道数
    return *this;
  }

  inline size_t outputChannels() const {
    return this->outputChannels_; // 返回输出通道数
  }

  inline FullyConnectedSparseOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize; // 设置批处理大小
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_; // 返回批处理大小
  }

  inline FullyConnectedSparseOperatorTester& inputStride(size_t inputStride) {

    this->inputStride_ = inputStride; // 设置输入步幅
    return *this;
  }

  inline size_t inputStride() const {
    return this->inputStride_; // 返回输入步幅
  }

  inline FullyConnectedSparseOperatorTester& outputStride(size_t outputStride) {
    this->outputStride_ = outputStride; // 设置输出步幅
    return *this;
  }

  inline size_t outputStride() const {
    return this->outputStride_; // 返回输出步幅
  }

 private:
  size_t inputChannels_ = 0; // 输入通道数，默认为0
  size_t outputChannels_ = 0; // 输出通道数，默认为0
  size_t batchSize_ = 1; // 批处理大小，默认为1
  size_t inputStride_ = 0; // 输入步幅，默认为0
  size_t outputStride_ = 0; // 输出步幅，默认为0
};
    // 确保输入步长大于等于1，断言检查
    assert(inputStride >= 1);
    // 设置对象的输入步长为给定值，并返回对象自身的引用
    this->inputStride_ = inputStride;
    return *this;
  }

  // 返回对象当前的输入步长
  inline size_t inputStride() const {
    // 如果输入步长为0，则返回输入通道数
    if (this->inputStride_ == 0) {
      return inputChannels();
    } else {
      // 否则，确保输入步长不小于输入通道数，并返回当前输入步长
      assert(this->inputStride_ >= inputChannels());
      return this->inputStride_;
    }
  }

  // 设置对象的输出步长为给定值，并返回对象自身的引用
  inline FullyConnectedSparseOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride >= 1);
    this->outputStride_ = outputStride;
    return *this;
  }

  // 返回对象当前的输出步长
  inline size_t outputStride() const {
    // 如果输出步长为0，则返回输出通道数
    if (this->outputStride_ == 0) {
      return outputChannels();
    } else {
      // 否则，确保输出步长不小于输出通道数，并返回当前输出步长
      assert(this->outputStride_ >= outputChannels());
      return this->outputStride_;
    }
  }

  // 设置对象的量化最小值，并返回对象自身的引用
  inline FullyConnectedSparseOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  // 返回对象当前的量化最小值
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置对象的量化最大值，并返回对象自身的引用
  inline FullyConnectedSparseOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回对象当前的量化最大值
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置对象的迭代次数，并返回对象自身的引用
  inline FullyConnectedSparseOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回对象当前的迭代次数
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 设置对象的行块大小，并返回对象自身的引用
  inline FullyConnectedSparseOperatorTester& rowBlockSize(size_t block_size) {
    this->rowBlockSize_ = block_size;
    return *this;
  }

  // 设置对象的列块大小，并返回对象自身的引用
  inline FullyConnectedSparseOperatorTester& colBlockSize(size_t block_size) {
    this->colBlockSize_ = block_size;
    return *this;
  }

  // 设置对象的稀疏度，并返回对象自身的引用
  inline FullyConnectedSparseOperatorTester& sparsity(float s) {
    this->sparsity_ = s;
    return *this;
  }

  // 返回对象当前的行块大小
  inline size_t rowBlockSize() const {
    return this->rowBlockSize_;
  }

  // 返回对象当前的列块大小
  inline size_t colBlockSize() const {
    return this->colBlockSize_;
  }

  // 返回对象当前的稀疏度
  inline float sparsity() const {
    return this->sparsity_;
  }

  // 定义枚举类型 Mode，表示测试模式为动态或运行时
  enum class Mode {
    Dynamic,
    Runtime,
  };

  // 对 Q8 进行测试，根据指定的模式进行测试
  void testQ8(const Mode mode) const {
    // 初始化随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    // 定义生成随机32位有符号整数的函数对象
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    // 定义生成随机8位无符号整数的函数对象
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    // 定义生成随机单精度浮点数的函数对象
    auto f32rng = std::bind(std::uniform_real_distribution<float>(1, 5), rng);

    // 初始化输入向量，确保其长度足够容纳所有输入
    std::vector<uint8_t> input(
        (batchSize() - 1) * inputStride() + inputChannels() + 8);
    // 初始化内核向量，确保其长度为输出通道数乘以输入通道数
    std::vector<uint8_t> kernel(outputChannels() * inputChannels());
    // 初始化偏置向量，确保其长度为输出通道数
    std::vector<int32_t> bias(outputChannels());
    // 初始化输出向量，确保其长度足够容纳所有输出
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + outputChannels());
    // 初始化动态输出向量，确保其长度与输出向量相同
    std::vector<float> output_dynamic(output.size());
    // 初始化累加器向量，确保其长度为批次大小乘以输出通道数
    std::vector<int32_t> accumulators(batchSize() * outputChannels());
    // 初始化浮点型累加器向量，确保其长度与累加器向量相同
    std::vector<float> accumulators_float(batchSize() * outputChannels());

    // 定义输入指针，指向输入向量的数据
    const uint8_t* const inputPtr = input.data();
    // 定义输入零点值为127
    const uint8_t inputZeroPoint = 127;
    // 确保输出通道数是8的倍数
    // Make number of output channels multiple of 8.
    // 这是我们所有 SSE/ARM 内核的最小公分母。
    size_t num_zero_points_padded = outputChannels() + 8;
    // 创建一个包含 num_zero_points_padded 个元素的向量，每个元素都是 127
    std::vector<uint8_t> kernelZeroPoints(num_zero_points_padded, 127);
    
    }
    
    void testQ8_prepacked(const Mode mode) const {
      // 创建一个随机数生成器
      std::random_device randomDevice;
      auto rng = std::mt19937(randomDevice());
      // 创建一个返回随机整数的函数对象
      auto s32rng =
          std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
      // 创建一个返回随机无符号整数的函数对象
      auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
      // 创建一个返回随机浮点数的函数对象
      auto f32rng =
          std::bind(std::uniform_real_distribution<float>(1, 5), rng);
    
      // 创建输入向量，其大小为 ((batchSize() - 1) * inputStride() + inputChannels() + 8)
      std::vector<uint8_t> input(
          (batchSize() - 1) * inputStride() + inputChannels() + 8);
      // 创建内核向量，其大小为 outputChannels() * inputChannels()
      std::vector<uint8_t> kernel(outputChannels() * inputChannels());
      // 创建偏置向量，其大小为 outputChannels()
      std::vector<int32_t> bias(outputChannels());
      // 创建输出向量，其大小为 ((batchSize() - 1) * outputStride() + outputChannels())
      std::vector<uint8_t> output(
          (batchSize() - 1) * outputStride() + outputChannels());
      // 创建动态输出向量，其大小与输出向量相同
      std::vector<float> output_dynamic(output.size());
      // 创建累加器向量，其大小为 batchSize() * outputChannels()
      std::vector<int32_t> accumulators(batchSize() * outputChannels());
      // 创建浮点数累加器向量，其大小为 batchSize() * outputChannels()
      std::vector<float> accumulators_float(batchSize() * outputChannels());
    
      // 声明指向 input 的常量指针
      const uint8_t* const inputPtr = input.data();
      // 设置输入零点值为 127
      const uint8_t inputZeroPoint = 127;
      // 使输出通道数成为 8 的倍数。
      // 这是我们所有 SSE/ARM 内核的最小公分母。
      size_t num_zero_points_padded = outputChannels() + 8;
      // 创建一个包含 num_zero_points_padded 个元素的向量，每个元素都是 127
      std::vector<uint8_t> kernelZeroPoints(num_zero_points_padded, 127);
    
      }
    }
    
    private:
    size_t inputChannels_{1};
    size_t inputStride_{0};
    size_t outputChannels_{1};
    size_t outputStride_{0};
    size_t batchSize_{1};
    uint8_t qmin_{0};
    uint8_t qmax_{255};
    size_t iterations_{1};
    float sparsity_{0.7f};
    size_t rowBlockSize_{1};
    size_t colBlockSize_{4};
};
```