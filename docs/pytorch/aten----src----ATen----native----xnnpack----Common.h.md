# `.\pytorch\aten\src\ATen\native\xnnpack\Common.h`

```
#pragma once
// 如果宏 USE_XNNPACK 被定义，则包含以下头文件

#ifdef USE_XNNPACK

#include <xnnpack.h>  // 包含 XNNPACK 头文件
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>  // 包含 pthreadpool-cpp 头文件
#include <c10/util/ArrayRef.h>  // 包含 ArrayRef 头文件
#include <limits>  // 包含数值极限的头文件
#include <memory>  // 包含内存管理的头文件

namespace at::native::xnnpack {

// 定义结构体 Deleter 用于释放 xnn_operator_t 类型的指针
struct Deleter final {
  void operator()(const xnn_operator_t op) const {
    xnn_delete_operator(op);
  }
};

// 使用智能指针管理 xnn_operator_t 类型的指针，定义为 Operator 类型
using Operator = std::unique_ptr<xnn_operator, Deleter>;

// 表示线性层运算的上下文信息
struct ContextLinear final {
  Operator op;          // 指向线性层操作符的智能指针
  int64_t output_channels;  // 输出通道数

  // 禁止默认构造函数
  ContextLinear() = delete;

  // 构造函数，接受操作符和输出通道数作为参数
  ContextLinear(Operator&& o, int64_t o_channels) : op(std::move(o)), output_channels(o_channels) {}

  // 定义静态常量表示最小和最大浮点数值
  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

// 表示二维卷积运算的上下文信息，包含所有相关的参数
struct ContextConv2D final {
  Operator op;                          // 指向卷积操作符的智能指针
  std::array<int64_t, 4> weight_size_;  // 权重尺寸数组
  std::array<int64_t, 2> padding_;      // 填充数组
  std::array<int64_t, 2> output_padding_;  // 输出填充数组
  std::array<int64_t, 2> stride_;       // 步幅数组
  std::array<int64_t, 2> dilation_;     // 膨胀数组
  bool transposed_;                     // 是否转置
  int64_t groups_;                      // 组数

  // 禁止默认构造函数
  ContextConv2D() = delete;

  // 构造函数，接受操作符和各种参数作为参数
  ContextConv2D(
      Operator&& o,
      std::array<int64_t, 4> weight_size,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> output_padding,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> dilation,
      bool transposed,
      int64_t groups)
      : op(std::move(o)),
        weight_size_(weight_size),
        padding_(padding),
        output_padding_(output_padding),
        stride_(stride),
        dilation_(dilation),
        transposed_(transposed),
        groups_(groups) {}

  // 定义静态常量表示最小和最大浮点数值
  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

namespace internal {

// 定义布局结构，用于描述张量的排列方式

struct Layout final {
  // 表示4维激活张量的布局信息
  struct Activation4D final {
    static constexpr size_t batch = 0u;    // 批次维度索引
    static constexpr size_t channels = 1u; // 通道维度索引
    static constexpr size_t height = 2u;   // 高度维度索引
    static constexpr size_t width = 3u;    // 宽度维度索引
  };

  // 表示N维激活张量的布局信息
  struct ActivationND final {
    // 一些运算符可能不仅限于4维张量。在这种情况下，XNNPACK 使用 _nc 后缀来表示该运算符，并期望所有维度（除了通道）被展平为一个参数：batch_size。
    static int64_t batch(const IntArrayRef tensor) {
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      // 处理批次大小为零的情况
      int64_t batch = tensor[0];

      for (size_t index = 1u; index < (tensor.size() - 1u); ++index) {
        batch *= tensor[index];
      }

      return batch;
    };

    // 获取通道维度大小的函数
    static int64_t channel(const IntArrayRef tensor) {
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      return tensor.back();
    };
  };

  // 表示卷积滤波器的布局信息
  struct Filter final {
    static constexpr size_t output = 0u;  // 输出维度索引
    static constexpr size_t input = 1u;   // 输入维度索引
    // 定义一个结构体的静态常量，表示高度为2
    static constexpr size_t height = 2u;
    
    // 定义一个结构体的静态常量，表示宽度为3
    static constexpr size_t width = 3u;
    
    
    
    // 定义另一个结构体的静态常量，表示高度为0
    static constexpr size_t height = 0u;
    
    // 定义另一个结构体的静态常量，表示宽度为1
    static constexpr size_t width = 1u;
    
    
    这些代码片段定义了两个不同的结构体（类似于类），每个结构体中包含两个静态常量：`height` 和 `width`。第一个结构体中的常量表示一个特定的高度和宽度，而第二个结构体中的常量则表示另一组不同的高度和宽度。
} // 结束 namespace at::native::xnnpack::internal
} // 结束 namespace at::native::xnnpack

#endif /* USE_XNNPACK */

// 开始声明命名空间 at::native::xnnpack
namespace at::native::xnnpack {
    // 声明函数 bool available()，用于检查 xnnpack 是否可用
    bool available();
} // 结束 namespace at::native::xnnpack
```