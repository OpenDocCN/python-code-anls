# `.\pytorch\torch\csrc\api\src\nn\modules\upsampling.cpp`

```
# 包含 Torch 库中的上采样模块定义头文件
#include <torch/nn/modules/upsampling.h>

# 包含标准字符串库
#include <string>

# 定义了 torch::nn 命名空间的别名 F
namespace F = torch::nn::functional;

# 定义了 torch::nn 命名空间
namespace torch {
namespace nn {

# 定义了 UpsampleImpl 类的构造函数，接受 UpsampleOptions 参数对象
UpsampleImpl::UpsampleImpl(
    const UpsampleOptions& options_) // NOLINT(modernize-pass-by-value)
    : options(options_) {}

# 重置 UpsampleImpl 类的成员函数
void UpsampleImpl::reset() {}

# 打印 UpsampleImpl 类的信息到输出流
void UpsampleImpl::pretty_print(std::ostream& stream) const {
  # 输出 UpsampleImpl 类的信息，包括 scale_factor 或 size 和 mode 的枚举值
  stream << "torch::nn::Upsample(";
  if (options.scale_factor() != c10::nullopt) {
    stream << "scale_factor=" << at::ArrayRef<double>(*options.scale_factor());
  } else {
    stream << "size=" << at::ArrayRef<int64_t>(*options.size());
  }
  stream << ", mode=" << enumtype::get_enum_name(options.mode()) << ")";
}

# 实现了 UpsampleImpl 类的前向传播函数，根据输入进行上采样操作
Tensor UpsampleImpl::forward(const Tensor& input) {
  # 定义上采样的模式变量 mode
  F::InterpolateFuncOptions::mode_t mode;
  # 根据 options.mode() 的枚举类型选择相应的上采样模式
  if (std::holds_alternative<enumtype::kNearest>(options.mode())) {
    mode = torch::kNearest;
  } else if (std::holds_alternative<enumtype::kLinear>(options.mode())) {
    mode = torch::kLinear;
  } else if (std::holds_alternative<enumtype::kBilinear>(options.mode())) {
    mode = torch::kBilinear;
  } else if (std::holds_alternative<enumtype::kBicubic>(options.mode())) {
    mode = torch::kBicubic;
  } else if (std::holds_alternative<enumtype::kTrilinear>(options.mode())) {
    mode = torch::kTrilinear;
  }

  # 调用 Torch 的插值函数进行实际的上采样操作
  return F::detail::interpolate(
      input,
      options.size(),
      options.scale_factor(),
      mode,
      options.align_corners(),
      c10::nullopt,
      false);
}

} // namespace nn
} // namespace torch
```