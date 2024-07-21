# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\max-pooling-operator-tester.h`

```py
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 本源代码根目录下的LICENSE文件中包含的BSD风格许可证授权此源代码的使用。
 */

#pragma once

#include <algorithm>         // C++标准库算法头文件
#include <cassert>           // C++标准库断言头文件
#include <cstddef>           // C++标准库大小类型头文件
#include <cstdlib>           // C标准库头文件
#include <functional>        // C++标准库函数对象头文件
#include <random>            // C++标准库随机数生成器头文件
#include <vector>            // C++标准库向量头文件

#include <pytorch_qnnpack.h> // PyTorch QNNPACK头文件

class MaxPoolingOperatorTester {
 public:
  inline MaxPoolingOperatorTester& padding(uint32_t padding) {
    this->paddingHeight_ = padding;
    this->paddingWidth_ = padding;
    return *this;
  }

  inline MaxPoolingOperatorTester& padding(
      uint32_t paddingHeight,
      uint32_t paddingWidth) {
    this->paddingHeight_ = paddingHeight;
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  inline MaxPoolingOperatorTester& paddingHeight(uint32_t paddingHeight) {
    this->paddingHeight_ = paddingHeight;
    return *this;
  }

  inline MaxPoolingOperatorTester& paddingWidth(uint32_t paddingWidth) {
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  inline uint32_t paddingHeight() const {
    return this->paddingHeight_;
  }

  inline uint32_t paddingWidth() const {
    return this->paddingWidth_;
  }

  inline MaxPoolingOperatorTester& inputSize(
      size_t inputHeight,
      size_t inputWidth) {
    assert(inputHeight >= 1);          // 断言输入高度至少为1
    assert(inputWidth >= 1);           // 断言输入宽度至少为1
    this->inputHeight_ = inputHeight;
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline MaxPoolingOperatorTester& inputHeight(size_t inputHeight) {
    assert(inputHeight >= 1);          // 断言输入高度至少为1
    this->inputHeight_ = inputHeight;
    return *this;
  }

  inline size_t inputHeight() const {
    return this->inputHeight_;
  }

  inline MaxPoolingOperatorTester& inputWidth(size_t inputWidth) {
    assert(inputWidth >= 1);           // 断言输入宽度至少为1
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline size_t inputWidth() const {
    return this->inputWidth_;
  }

  inline MaxPoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);             // 断言通道数不为0
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline MaxPoolingOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline MaxPoolingOperatorTester& poolingSize(uint32_t poolingSize) {
    assert(poolingSize >= 1);          // 断言池化尺寸至少为1
    this->poolingHeight_ = poolingSize;
    this->poolingWidth_ = poolingSize;
    return *this;
  }

  inline MaxPoolingOperatorTester& poolingSize(
      uint32_t poolingHeight,
      uint32_t poolingWidth) {
    assert(poolingHeight >= 1);        // 断言池化高度至少为1
    assert(poolingWidth >= 1);         // 断言池化宽度至少为1
    this->poolingHeight_ = poolingHeight;
    this->poolingWidth_ = poolingWidth;
    return *this;
  }

  inline MaxPoolingOperatorTester& poolingHeight(uint32_t poolingHeight) {
    assert(poolingHeight >= 1);        // 断言池化高度至少为1
    this->poolingHeight_ = poolingHeight;
    return *this;
  }
  // 返回当前对象的引用
  return *this;
}

// 返回池化操作的高度
inline uint32_t poolingHeight() const {
  return this->poolingHeight_;
}

// 设置并返回池化操作的宽度
inline MaxPoolingOperatorTester& poolingWidth(uint32_t poolingWidth) {
  assert(poolingWidth >= 1);
  this->poolingWidth_ = poolingWidth;
  return *this;
}

// 返回池化操作的宽度
inline uint32_t poolingWidth() const {
  return this->poolingWidth_;
}

// 设置并返回步幅（高度和宽度相同）
inline MaxPoolingOperatorTester& stride(uint32_t stride) {
  assert(stride >= 1);
  this->strideHeight_ = stride;
  this->strideWidth_ = stride;
  return *this;
}

// 设置并返回步幅（分别设置高度和宽度）
inline MaxPoolingOperatorTester& stride(
    uint32_t strideHeight,
    uint32_t strideWidth) {
  assert(strideHeight >= 1);
  assert(strideWidth >= 1);
  this->strideHeight_ = strideHeight;
  this->strideWidth_ = strideWidth;
  return *this;
}

// 设置并返回步幅的高度
inline MaxPoolingOperatorTester& strideHeight(uint32_t strideHeight) {
  assert(strideHeight >= 1);
  this->strideHeight_ = strideHeight;
  return *this;
}

// 返回步幅的高度
inline uint32_t strideHeight() const {
  return this->strideHeight_;
}

// 设置并返回步幅的宽度
inline MaxPoolingOperatorTester& strideWidth(uint32_t strideWidth) {
  assert(strideWidth >= 1);
  this->strideWidth_ = strideWidth;
  return *this;
}

// 返回步幅的宽度
inline uint32_t strideWidth() const {
  return this->strideWidth_;
}

// 设置并返回膨胀系数（高度和宽度相同）
inline MaxPoolingOperatorTester& dilation(uint32_t dilation) {
  assert(dilation >= 1);
  this->dilationHeight_ = dilation;
  this->dilationWidth_ = dilation;
  return *this;
}

// 设置并返回膨胀系数（分别设置高度和宽度）
inline MaxPoolingOperatorTester& dilation(
    uint32_t dilationHeight,
    uint32_t dilationWidth) {
  assert(dilationHeight >= 1);
  assert(dilationWidth >= 1);
  this->dilationHeight_ = dilationHeight;
  this->dilationWidth_ = dilationWidth;
  return *this;
}

// 设置并返回膨胀系数的高度
inline MaxPoolingOperatorTester& dilationHeight(uint32_t dilationHeight) {
  assert(dilationHeight >= 1);
  this->dilationHeight_ = dilationHeight;
  return *this;
}

// 返回膨胀系数的高度
inline uint32_t dilationHeight() const {
  return this->dilationHeight_;
}

// 设置并返回膨胀系数的宽度
inline MaxPoolingOperatorTester& dilationWidth(uint32_t dilationWidth) {
  assert(dilationWidth >= 1);
  this->dilationWidth_ = dilationWidth;
  return *this;
}

// 返回膨胀系数的宽度
inline uint32_t dilationWidth() const {
  return this->dilationWidth_;
}

// 返回经过膨胀后的池化操作的高度
inline uint32_t dilatedPoolingHeight() const {
  return (poolingHeight() - 1) * dilationHeight() + 1;
}

// 返回经过膨胀后的池化操作的宽度
inline uint32_t dilatedPoolingWidth() const {
  return (poolingWidth() - 1) * dilationWidth() + 1;
}

// 返回输出的高度
inline size_t outputHeight() const {
  const size_t paddedInputHeight = inputHeight() + paddingHeight() * 2;
  if (paddedInputHeight <= dilatedPoolingHeight()) {
    return 1;
  } else {
    return (paddedInputHeight - dilatedPoolingHeight()) / strideHeight() + 1;
  }
}

// 返回输出的宽度
inline size_t outputWidth() const {
  const size_t paddedInputWidth = inputWidth() + paddingWidth() * 2;
  if (paddedInputWidth <= dilatedPoolingWidth()) {
    return 1;
  } else {
    return (paddedInputWidth - dilatedPoolingWidth()) / strideWidth() + 1;
  }
}
  // 如果输入像素步长为零，则返回通道数作为默认值；否则返回设置的像素步长
  inline size_t inputPixelStride() const {
    if (this->inputPixelStride_ == 0) {
      return channels();
    } else {
      assert(this->inputPixelStride_ >= channels());
      return this->inputPixelStride_;
    }
  }

  // 设置输入像素步长，并返回当前对象的引用
  inline MaxPoolingOperatorTester& inputPixelStride(size_t inputPixelStride) {
    assert(inputPixelStride != 0);
    this->inputPixelStride_ = inputPixelStride;
    return *this;
  }

  // 如果输出像素步长为零，则返回通道数作为默认值；否则返回设置的像素步长
  inline size_t outputPixelStride() const {
    if (this->outputPixelStride_ == 0) {
      return channels();
    } else {
      assert(this->outputPixelStride_ >= channels());
      return this->outputPixelStride_;
    }
  }

  // 设置输出像素步长，并返回当前对象的引用
  inline MaxPoolingOperatorTester& outputPixelStride(size_t outputPixelStride) {
    assert(outputPixelStride != 0);
    this->outputPixelStride_ = outputPixelStride;
    return *this;
  }

  // 返回设置的下一个输入图像的高度，如果未设置则返回当前输入图像的高度
  inline uint32_t nextInputHeight() const {
    if (this->nextInputHeight_ == 0) {
      return inputHeight();
    } else {
      return this->nextInputHeight_;
    }
  }

  // 设置下一个输入图像的高度，并返回当前对象的引用
  inline MaxPoolingOperatorTester& nextInputHeight(uint32_t nextInputHeight) {
    assert(nextInputHeight >= 1);
    this->nextInputHeight_ = nextInputHeight;
    return *this;
  }

  // 返回设置的下一个输入图像的宽度，如果未设置则返回当前输入图像的宽度
  inline uint32_t nextInputWidth() const {
    if (this->nextInputWidth_ == 0) {
      return inputWidth();
    } else {
      return this->nextInputWidth_;
    }
  }

  // 设置下一个输入图像的宽度，并返回当前对象的引用
  inline MaxPoolingOperatorTester& nextInputWidth(uint32_t nextInputWidth) {
    assert(nextInputWidth >= 1);
    this->nextInputWidth_ = nextInputWidth;
    return *this;
  }

  // 返回计算后的下一个输出图像的高度
  inline size_t nextOutputHeight() const {
    const size_t paddedNextInputHeight =
        nextInputHeight() + paddingHeight() * 2;
    if (paddedNextInputHeight <= dilatedPoolingHeight()) {
      return 1;
    } else {
      return (paddedNextInputHeight - dilatedPoolingHeight()) / strideHeight() + 1;
    }
  }

  // 返回计算后的下一个输出图像的宽度
  inline size_t nextOutputWidth() const {
    const size_t paddedNextInputWidth = nextInputWidth() + paddingWidth() * 2;
    if (paddedNextInputWidth <= dilatedPoolingWidth()) {
      return 1;
    } else {
      return (paddedNextInputWidth - dilatedPoolingWidth()) / strideWidth() + 1;
    }
  }

  // 返回设置的下一个批处理大小，如果未设置则返回当前批处理大小
  inline size_t nextBatchSize() const {
    if (this->nextBatchSize_ == 0) {
      return batchSize();
    } else {
      return this->nextBatchSize_;
    }
  }

  // 设置下一个批处理大小，并返回当前对象的引用
  inline MaxPoolingOperatorTester& nextBatchSize(size_t nextBatchSize) {
    assert(nextBatchSize >= 1);
    this->nextBatchSize_ = nextBatchSize;
    return *this;
  }
  // 返回当前对象的引用
  return *this;
}

  // 返回最小量化值 qmin_
  inline uint8_t qmin() const {
    return this->qmin_;
  }

  // 设置最大量化值 qmax_ 并返回当前对象的引用
  inline MaxPoolingOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  // 返回当前对象的最大量化值 qmax_
  inline uint8_t qmax() const {
    return this->qmax_;
  }

  // 设置迭代次数 iterations_ 并返回当前对象的引用
  inline MaxPoolingOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  // 返回当前对象的迭代次数 iterations_
  inline size_t iterations() const {
    return this->iterations_;
  }

  // 对 uint8 类型的输入进行测试
  void testU8() const {
    // 创建随机设备和随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 初始化输入、输出和参考输出向量
    std::vector<uint8_t> input(
        (batchSize() * inputHeight() * inputWidth() - 1) * inputPixelStride() +
        channels());
    std::vector<uint8_t> output(
        (batchSize() * outputHeight() * outputWidth() - 1) *
            outputPixelStride() +
        channels());
    std::vector<uint8_t> outputRef(
        batchSize() * outputHeight() * outputWidth() * channels());
  }

  // 对 uint8 类型的设置进行测试
  void testSetupU8() const {
    // 创建随机设备和随机数生成器
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    // 初始化输入、输出、参考输出和下一个参考输出向量，根据最大尺寸
    std::vector<uint8_t> input(std::max(
        (batchSize() * inputHeight() * inputWidth() - 1) * inputPixelStride() +
            channels(),
        (nextBatchSize() * nextInputHeight() * nextInputWidth() - 1) *
                inputPixelStride() +
            channels()));
    std::vector<uint8_t> output(std::max(
        (batchSize() * outputHeight() * outputWidth() - 1) *
                outputPixelStride() +
            channels(),
        (nextBatchSize() * nextOutputHeight() * nextOutputWidth() - 1) *
                outputPixelStride() +
            channels()));
    std::vector<float> outputRef(
        batchSize() * outputHeight() * outputWidth() * channels());
    std::vector<float> nextOutputRef(
        nextBatchSize() * nextOutputHeight() * nextOutputWidth() * channels());
  }
};



# 这行代码是一个单独的分号，通常在编程语言中用于结束语句或表达式。
# 在这个上下文中，分号可能表示一个语句的结束，但它是孤立的，因此可能是一个错误或不完整的代码片段。
# 它没有任何实际的功能，可能是由于复制粘贴或其他错误导致的。
```