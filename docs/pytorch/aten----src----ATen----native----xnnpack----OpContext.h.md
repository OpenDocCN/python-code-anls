# `.\pytorch\aten\src\ATen\native\xnnpack\OpContext.h`

```
// 只有在首次包含此头文件时，才会编译此代码
#pragma once

// 如果定义了 USE_XNNPACK 宏，则继续编译
#ifdef USE_XNNPACK

// 包含以下头文件，用于后续声明和定义
#include <ATen/core/ivalue.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/Tensor.h>

// 命名空间 at::native::xnnpack 包含了以下声明和定义
namespace at::native::xnnpack {

// 声明一些使用别名的数据结构，用于序列化和反序列化
using SerializationTypeLinearPrePack = std::tuple<
    Tensor,                                     // 原始权重张量
    std::optional<Tensor>,                      // 原始偏置张量（可选）
    std::optional<Scalar>,                      // 输出最小值（可选）
    std::optional<Scalar>>;                     // 输出最大值（可选）

using SerializationTypeConv2dPrePack = std::tuple<
    Tensor,                                     // 原始权重张量
    std::optional<Tensor>,                      // 原始偏置张量（可选）
    std::vector<int64_t>,                       // 步长
    std::vector<int64_t>,                       // 填充
    std::vector<int64_t>,                       // 膨胀
    int64_t,                                    // 分组数
    std::optional<Scalar>,                      // 输出最小值（可选）
    std::optional<Scalar>>;                     // 输出最大值（可选）

using SerializationTypeTransposeConv2dPrePack = std::tuple<
    Tensor,                                     // 原始权重张量
    std::optional<Tensor>,                      // 原始偏置张量（可选）
    std::vector<int64_t>,                       // 步长
    std::vector<int64_t>,                       // 填充
    std::vector<int64_t>,                       // 膨胀
    std::vector<int64_t>,                       // 输出填充
    int64_t,                                    // 分组数
    std::optional<Scalar>,                      // 输出最小值（可选）
    std::optional<Scalar>>;                     // 输出最大值（可选）

// LinearOpContext 类，继承自 torch::jit::CustomClassHolder 类
class LinearOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;                          // 原始权重张量
  std::optional<Tensor> orig_bias_;              // 原始偏置张量（可选）
  std::optional<Scalar> output_min_;             // 输出最小值（可选）
  std::optional<Scalar> output_max_;             // 输出最大值（可选）
  bool orig_weight_and_bias_freed_;              // 标记原始权重和偏置是否已释放

 public:
  // 解包函数，返回序列化后的数据结构
  SerializationTypeLinearPrePack unpack() {
    // 检查原始权重和偏置是否已释放，如果已释放则抛出错误
    TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
    // 返回包含原始权重、原始偏置、输出最小值和输出最大值的元组
    return std::make_tuple(orig_weight_, orig_bias_, output_min_, output_max_);
  }

  // 虚拟函数，子类需实现的运行函数，传入输入张量并返回结果张量
  virtual Tensor run(const Tensor& input) = 0;

  // 虚拟函数，子类需实现的释放原始权重和偏置的函数
  virtual void free_orig_weight_and_bias() = 0;
};

// XNNPackLinearOpContext 类，继承自 LinearOpContext 类
class XNNPackLinearOpContext final : public LinearOpContext {
 private:
  ContextLinear op_context_;                    // 线性操作的上下文对象

 public:
  // 构造函数，初始化对象时传入权重、偏置、输出最小值、输出最大值和上下文
  XNNPackLinearOpContext(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      const std::optional<Scalar>& min,
      const std::optional<Scalar>& max,
      ContextLinear&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);           // 移动权重张量
    orig_bias_ = std::move(bias);               // 移动偏置张量
    output_min_ = min;                          // 设置输出最小值
    output_max_ = max;                          // 设置输出最大值
    orig_weight_and_bias_freed_ = false;        // 初始状态下，原始权重和偏置未释放
  }

  // 实现父类的纯虚函数，运行线性操作并返回结果张量
  Tensor run(const Tensor& input) override;

  // 实现父类的纯虚函数，释放原始权重和偏置的函数
  void free_orig_weight_and_bias() override;

  // 静态函数，创建 XNNPackLinearOpContext 的实例，传入权重、偏置、输出最小值和输出最大值
  static c10::intrusive_ptr<LinearOpContext> create_context(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      const std::optional<Scalar>& output_min,
      const std::optional<Scalar>& output_max);
};

// Conv2dOpContext 类，继承自 torch::jit::CustomClassHolder 类
class Conv2dOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;                          // 原始权重张量
  std::optional<Tensor> orig_bias_;              // 原始偏置张量（可选）
  std::vector<int64_t> stride_;                 // 步长
  std::vector<int64_t> padding_;                // 填充
  std::vector<int64_t> dilation_;               // 膨胀
  int64_t groups_;                              // 分组数
  std::optional<Scalar> output_min_;             // 输出最小值（可选）
  std::optional<Scalar> output_max_;             // 输出最大值（可选）
  bool orig_weight_and_bias_freed_;              // 标记原始权重和偏置是否已释放

 public:
  // 解包函数，返回序列化后的数据结构
  SerializationTypeConv2dPrePack unpack() {
    // 检查原始权重和偏置是否已释放，如果已释放则抛出错误
    TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
    // 返回包含原始权重、原始偏置、步长、填充、膨胀、分组数、输出最小值和输出最大值的元组
    return std::make_tuple(orig_weight_, orig_bias_, stride_, padding_, dilation_, groups_, output_min_, output_max_);
  }

  // 虚拟函数，子类需实现的运行函数，传入输入张量并返回结果张量
  virtual Tensor run(const Tensor& input) = 0;

  // 虚拟函数，子类需实现的释放原始权重和偏置的函数
  virtual void free_orig_weight_and_bias() = 0;
};

// 结束 at::native::xnnpack 命名空间
} // namespace at::native::xnnpack

// 结束 USE_XNNPACK 宏的条件编译
#endif // USE_XNNPACK
    # 返回一个 std::tuple 对象，包含以下内容：
    # - orig_weight_: 原始权重
    # - orig_bias_: 原始偏置
    # - stride_: 步长
    # - padding_: 填充
    # - dilation_: 膨胀系数
    # - groups_: 分组数
    # - output_min_: 输出最小值
    # - output_max_: 输出最大值
      }
      
      # 一个纯虚函数，需要在派生类中实现，用于执行具体的运算操作，输入参数是一个 Tensor 对象，返回值也是一个 Tensor 对象
      virtual Tensor run(const Tensor& input) = 0;
      
      # 一个纯虚函数，需要在派生类中实现，用于释放原始权重和偏置
      virtual void free_orig_weight_and_bias() = 0;
};

// TransposeConv2dOpContext 类的定义，继承自 torch::jit::CustomClassHolder
class TransposeConv2dOpContext : public torch::jit::CustomClassHolder {
 protected:
  // 存储原始权重
  Tensor orig_weight_;
  // 可选的原始偏置
  std::optional<Tensor> orig_bias_;
  // 步幅（stride）
  std::vector<int64_t> stride_;
  // 填充（padding）
  std::vector<int64_t> padding_;
  // 输出填充（output_padding）
  std::vector<int64_t> output_padding_;
  // 膨胀（dilation）
  std::vector<int64_t> dilation_;
  // 分组数
  int64_t groups_;
  // 可选的输出最小值
  std::optional<Scalar> output_min_;
  // 可选的输出最大值
  std::optional<Scalar> output_max_;
  // 标记原始权重和偏置是否已释放
  bool orig_weight_and_bias_freed_;

 public:
  // 反序列化为 SerializationTypeTransposeConv2dPrePack 对象的方法
  SerializationTypeTransposeConv2dPrePack unpack() {
    // 检查原始权重和偏置是否已被释放
    TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
    // 返回元组，包含所有必要的参数
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        output_padding_,
        dilation_,
        groups_,
        output_min_,
        output_max_);
  }

  // 纯虚函数，子类必须实现，用于执行操作
  virtual Tensor run(const Tensor& input) = 0;
  // 纯虚函数，子类必须实现，用于释放原始权重和偏置
  virtual void free_orig_weight_and_bias() = 0;
};

// XNNPackConv2dOpContext 类，继承自 Conv2dOpContext 类
class XNNPackConv2dOpContext final : public Conv2dOpContext {
 private:
  // XNNPack 特定的上下文信息
  ContextConv2D op_context_;
  // 用于同步的互斥锁，防止多线程访问同一资源时的竞争
  std::mutex xnnp_mutex_;

 public:
  // 构造函数，初始化 XNNPackConv2dOpContext 对象
  XNNPackConv2dOpContext(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      uint64_t groups,
      const std::optional<Scalar>& min,
      const std::optional<Scalar>& max,
      ContextConv2D&& op_context)
      : op_context_(std::move(op_context)) {
    // 初始化权重
    orig_weight_ = std::move(weight);
    // 初始化偏置
    orig_bias_ = std::move(bias);
    // 初始化填充
    padding_ = std::move(padding);
    // 初始化步幅
    stride_ = std::move(stride);
    // 初始化膨胀
    dilation_ = std::move(dilation);
    // 初始化分组数
    groups_ = groups;
    // 初始化输出最小值
    output_min_ = min;
    // 初始化输出最大值
    output_max_ = max;
    // 标记原始权重和偏置未被释放
    orig_weight_and_bias_freed_ = false;
  }

  // 重写的 run 方法，执行卷积操作
  Tensor run(const Tensor& input) override;
  // 重写的 free_orig_weight_and_bias 方法，释放原始权重和偏置
  void free_orig_weight_and_bias() override;

  // 静态方法，创建 XNNPackConv2dOpContext 对象的上下文
  static c10::intrusive_ptr<Conv2dOpContext> create_context(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      const std::optional<Scalar>& output_min,
      const std::optional<Scalar>& output_max);
};
class XNNPackTransposeConv2dOpContext final : public TransposeConv2dOpContext {
 private:
  ContextConv2D op_context_;
  // xnnpack convs use indirection buffer.
  // These buffers need setup at runtime and/or when input
  // dims change. If we are running the same model on multiple
  // threads, this can lead to contention where indirection buffer
  // is being accessed and updated at the same time from two different
  // threads.
  // 定义私有成员变量 op_context_，用于存储上下文信息
  std::mutex xnnp_mutex_; // 定义互斥量 xnnp_mutex_，用于线程同步

 public:
  // XNNPackTransposeConv2dOpContext 的构造函数，初始化对象
  XNNPackTransposeConv2dOpContext(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& output_padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      uint64_t groups,
      const std::optional<Scalar>& min,
      const std::optional<Scalar>& max,
      ContextConv2D&& op_context)
      : op_context_(std::move(op_context)) {
    // 初始化原始权重和偏置
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    // 初始化填充、输出填充、步长、膨胀等参数
    padding_ = std::move(padding);
    output_padding_ = std::move(output_padding);
    stride_ = std::move(stride);
    dilation_ = std::move(dilation);
    // 初始化组数、输出最小值和最大值
    groups_ = groups;
    output_min_ = min;
    output_max_ = max;
    // 标记原始权重和偏置未释放
    orig_weight_and_bias_freed_ = false;
  }

  // 重写父类方法，执行转置卷积操作
  Tensor run(const Tensor& input) override;
  // 重写父类方法，释放原始权重和偏置
  void free_orig_weight_and_bias() override;

  // 静态方法，创建 XNNPackTransposeConv2dOpContext 对象
  static c10::intrusive_ptr<TransposeConv2dOpContext> create_context(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& output_padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      const std::optional<Scalar>& output_min,
      const std::optional<Scalar>& output_max);
};

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
```