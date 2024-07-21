# `.\pytorch\aten\src\ATen\native\mkldnn\OpContext.h`

```py
#pragma once
// 预处理指令：只包含一次头文件

#include <ATen/Tensor.h>
// 引入 ATen 库中的 Tensor 类

#include <ATen/core/ivalue.h>
// 引入 ATen 库中的 IValue 类

#include <ATen/native/mkldnn/Common.h>
// 引入 ATen 库中 MKLDNN 相关的通用功能头文件

#if AT_MKLDNN_ENABLED()
// 如果 MKLDNN 功能被启用

namespace at {
namespace native {
namespace mkldnn {

const static std::map<std::string, ideep::attr_t> fusion_attr_map = {
    // 静态变量，将字符串与 MKLDNN 属性映射关联起来
    {"none", ideep::attr_t()},
    {"relu", ideep::attr_t::fuse_relu()},
};

using SerializationTypeConvPrePack = std::tuple<
    Tensor,
    std::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    std::vector<int64_t>,
    std::string>;
// 使用别名定义序列化类型 ConvPrePack，包含多种数据类型的元组

class ConvOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  std::optional<Tensor> orig_bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  std::vector<int64_t> input_size_;
  std::string attr_;
  // 定义卷积操作上下文的基类，包含卷积操作所需的参数和属性

 public:
  SerializationTypeConvPrePack unpack() {
    // 解包函数，返回序列化类型的元组
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        dilation_,
        groups_,
        input_size_,
        attr_);
  }

  virtual Tensor run(const Tensor& input) = 0;
  // 纯虚函数，子类需要实现的卷积运行函数，接收输入张量并返回输出张量

  virtual void run(const Tensor& input, void* output) = 0;
  // 纯虚函数，子类需要实现的卷积运行函数，接收输入张量并输出到给定地址
};

class MkldnnConvOpContext final : public ConvOpContext {
 private:
  ContextConv op_context_;
  // MKLDNN 卷积操作上下文对象

 public:
  MkldnnConvOpContext(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      uint64_t groups,
      std::vector<int64_t>&& input_size,
      ContextConv&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    padding_ = std::move(padding);
    stride_ = std::move(stride);
    dilation_ = std::move(dilation);
    groups_ = groups;
    input_size_ = std::move(input_size);
    // MKLDNN 卷积操作上下文的构造函数，接收卷积参数和上下文信息
  }

  Tensor run(const Tensor& input) override;
  // 实现基类纯虚函数，运行 MKLDNN 卷积操作并返回输出张量

  void run(const Tensor& input, void* output) override;
  // 实现基类纯虚函数，运行 MKLDNN 卷积操作并输出到给定地址

  static c10::intrusive_ptr<ConvOpContext> create_context(
      Tensor&& weight,
      std::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      std::vector<int64_t>&& input_size,
      const ideep::attr_t& attr);
  // 静态函数，创建 MKLDNN 卷积操作上下文对象的工厂函数，返回上下文的指针
};

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
// 结束条件编译指令，判断 MKLDNN 功能是否启用
```