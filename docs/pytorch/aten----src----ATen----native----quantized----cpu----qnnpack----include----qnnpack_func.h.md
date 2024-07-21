# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\include\qnnpack_func.h`

```
#pragma once
// 一次性包含标准库cstdlib和QNNPACK运算符头文件
#include <cstdlib>
#include <qnnpack/operator.h>

// 声明QNNPACK命名空间
namespace qnnpack {

// 定义PrePackConvWeights类，用于预包装卷积权重
class PrePackConvWeights final {
 public:
  // 构造函数，初始化预包装卷积权重对象
  PrePackConvWeights(
      const pytorch_qnnp_operator_t convolution, // QNNPACK卷积操作符
      const uint8_t* kernel_zero_points,         // 卷积核零点
      const uint8_t* kernel,                     // 卷积核
      const int32_t* bias);                      // 偏置

  // 返回预包装权重数据的指针
  void* getPackedWeights() const
  {
    return packed_weights_;
  }

  // 返回输出通道数
  int64_t getOutputChannels() const
  {
    return output_channels_;
  }

  // 析构函数，释放预包装权重数据的内存
  ~PrePackConvWeights()
  {
    if (packed_weights_ != nullptr) {
      free(packed_weights_);
    }
  }

  // 删除默认构造函数、拷贝构造函数和赋值运算符重载
  PrePackConvWeights() = delete;
  PrePackConvWeights(const PrePackConvWeights&) = delete;
  PrePackConvWeights& operator=(const PrePackConvWeights&) = delete;

 private:
  void* packed_weights_ = nullptr; // 预包装权重数据指针
  int64_t output_channels_;        // 输出通道数
};

// 定义PackBMatrix类，用于打包B矩阵数据
class PackBMatrix final {
 public:
  // 构造函数，用于通道间量化的B矩阵打包
  PackBMatrix(
      size_t input_channels,                    // 输入通道数
      size_t output_channels,                   // 输出通道数
      const uint8_t* kernel_zero_points,        // 卷积核零点
      const float* requantization_scale,        // 重量化比例
      const uint8_t* kernel,                    // 卷积核
      const int32_t* bias);                     // 偏置

  // 用于动态模式量化的构造函数
  // 在动态模式中，我们不支持通道级量化，
  // 因此避免了为每个通道的零点和重量化比例分配内存的开销
  PackBMatrix(
      size_t input_channels,                    // 输入通道数
      size_t output_channels,                   // 输出通道数
      const uint8_t kernel_zero_point,          // 卷积核零点
      const float requantization_scale,         // 重量化比例
      const uint8_t* kernel,                    // 卷积核
      const int32_t* bias);                     // 偏置

  // 返回打包后的权重数据指针
  void* getPackedWeights() const
  {
    return packed_weights_;
  }

  // 解包权重数据，将卷积核零点应用到卷积核中
  void unpackWeights(
      const uint8_t* kernel_zero_points,        // 卷积核零点
      int8_t* kernel) const;                    // 卷积核

  // 返回输入通道数
  size_t getInputChannels() const
  {
    return input_channels_;
  }

  // 返回输出通道数
  size_t getOutputChannels() const
  {
    return output_channels_;
  }

  // 析构函数，释放打包权重数据的内存
  ~PackBMatrix()
  {
    if (packed_weights_ != nullptr) {
      free(packed_weights_);
    }
  }

  // 删除默认构造函数、拷贝构造函数和赋值运算符重载
  PackBMatrix() = delete;
  PackBMatrix(const PackBMatrix&) = delete;
  PackBMatrix& operator=(const PackBMatrix&) = delete;

 private:
  void* packed_weights_ = nullptr;   // 打包后的权重数据指针
  size_t input_channels_;            // 输入通道数
  size_t output_channels_;           // 输出通道数
};

// 定义枚举类型pytorch_qnnp_status，用于表示QNNPACK线性操作的状态
enum pytorch_qnnp_status qnnpackLinear(
    const size_t batch_size,                // 批量大小
    const size_t input_channels,            // 输入通道数
    const size_t output_channels,           // 输出通道数
    const uint8_t input_zero_point,         // 输入零点
    const uint8_t* kernel_zero_points,      // 卷积核零点
    const float* requantization_scales,     // 重量化比例
    const uint8_t output_zero_point,        // 输出零点
    const uint8_t output_min,               // 输出最小值
    const uint8_t output_max,               // 输出最大值
    const uint8_t* input,                   // 输入数据
    const size_t input_stride,              // 输入步长
    void* packed_weights,                   // 打包后的权重数据
    uint8_t* output,                        // 输出数据
    const size_t output_stride,             // 输出步长
    pthreadpool_t threadpool);              // 线程池

// 定义QNNPACK卷积操作函数qnnpackConv
enum pytorch_qnnp_status qnnpackConv(
    const pytorch_qnnp_operator_t convolution, // QNNPACK卷积操作符
    void* packed_weights,                     // 打包后的权重数据
    const size_t batch_size,                  // 批量大小
    const size_t input_depth,                 // 输入深度
    const size_t input_height,                // 输入高度
    const size_t input_width,                 // 输入宽度
    const uint8_t input_zero_point,           // 输入零点
    const uint8_t* input,                     // 输入数据
    const uint8_t* kernel_zero_points,        // 卷积核零点



    // 输出数据
    uint8_t* output,
    const size_t output_stride,               // 输出步长
    pthreadpool_t threadpool);                // 线程池

} // namespace qnnpack
    const float* requantization_scales,

# 定义一个指向常量浮点数数组的指针 `requantization_scales`，用于存储重新量化的比例因子


    const uint8_t output_zero_point,

# 定义一个常量无符号8位整数 `output_zero_point`，表示输出的零点偏移量


    const uint8_t output_min,

# 定义一个常量无符号8位整数 `output_min`，表示输出的最小值


    const uint8_t output_max,

# 定义一个常量无符号8位整数 `output_max`，表示输出的最大值


    uint8_t* output,

# 定义一个指向无符号8位整数的指针 `output`，用于存储结果数据的输出


    pthreadpool_t threadpool);

# 定义一个指向 `pthreadpool_t` 类型的线程池指针 `threadpool`，用于并行执行任务的线程池对象
// 使用 QNNPACK 库实现的反卷积操作函数，接受 QNNPACK 运算符和相关参数
enum pytorch_qnnp_status qnnpackDeConv(
    const pytorch_qnnp_operator_t deconvolution,  // QNNPACK 反卷积运算符
    void* packed_weights,                         // 打包的权重数据
    const size_t batch_size,                      // 批处理大小
    const size_t input_height,                    // 输入图像高度
    const size_t input_width,                     // 输入图像宽度
    const uint8_t input_zero_point,               // 输入零点
    const uint8_t* input,                         // 输入数据指针
    const uint8_t* kernel_zero_points,            // 内核零点数组
    const float* requantization_scales,           // 重新量化比例数组
    const uint8_t output_zero_point,              // 输出零点
    const uint8_t output_min,                     // 输出最小值
    const uint8_t output_max,                     // 输出最大值
    uint8_t* output,                              // 输出数据指针
    pthreadpool_t threadpool);                    // 线程池

// 使用 QNNPACK 库实现的动态线性运算函数，接受相关参数
enum pytorch_qnnp_status qnnpackLinearDynamic(
    const size_t batch_size,                      // 批处理大小
    const size_t input_channels,                  // 输入通道数
    const size_t output_channels,                 // 输出通道数
    const uint8_t input_zero_point,               // 输入零点
    const uint8_t* kernel_zero_points,            // 内核零点数组
    const float* dequantization_scales,           // 反量化比例数组
    const uint8_t* input,                         // 输入数据指针
    const size_t input_stride,                    // 输入步幅
    void* packed_weights,                         // 打包的权重数据
    const float* bias,                            // 偏置数组
    float* output,                                // 输出数据指针
    const size_t output_stride,                   // 输出步幅
    pthreadpool_t threadpool);                    // 线程池

} // namespace qnnpack


这段代码定义了两个函数 `qnnpackDeConv` 和 `qnnpackLinearDynamic`，分别实现了基于 QNNPACK 库的反卷积和动态线性运算。每个函数接受特定的参数，用于执行相应的操作，并返回 `pytorch_qnnp_status` 枚举类型表示操作的执行状态。
```