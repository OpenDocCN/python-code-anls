# `.\pytorch\torch\csrc\api\src\nn\modules\linear.cpp`

```py
// 包含 Torch 库中的线性函数和初始化模块
#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>
#include <torch/nn/modules/linear.h>

// 包含 Torch 的数据类型定义和实用工具
#include <torch/types.h>
#include <torch/utils.h>

// 包含标准数学和整数库
#include <cmath>
#include <cstdint>

// 使用命名空间别名 F 指向 torch::nn::functional
namespace F = torch::nn::functional;

// 定义 torch::nn 命名空间
namespace torch {
namespace nn {

// IdentityImpl 类的成员函数实现

void IdentityImpl::reset() {}

// 打印 IdentityImpl 对象的信息到输出流
void IdentityImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Identity()";
}

// 前向传播函数，直接返回输入的张量
Tensor IdentityImpl::forward(const Tensor& input) {
  return input;
}

// ============================================================================

// LinearImpl 类的构造函数实现
LinearImpl::LinearImpl(const LinearOptions& options_) : options(options_) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

// 重置参数的函数
void LinearImpl::reset() {
  // 根据选项创建权重参数
  weight = register_parameter(
      "weight", torch::empty({options.out_features(), options.in_features()}));
  // 如果选项包含偏置，则创建偏置参数；否则创建不需要梯度的偏置参数
  if (options.bias()) {
    bias = register_parameter("bias", torch::empty(options.out_features()));
  } else {
    bias = register_parameter("bias", {}, /*requires_grad=*/false);
  }

  // 重置参数的具体数值
  reset_parameters();
}

// 重置参数数值的函数
void LinearImpl::reset_parameters() {
  // 使用 Kaiming 均匀分布初始化权重参数
  torch::nn::init::kaiming_uniform_(
      weight, std::sqrt(5)); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  
  // 如果偏置参数已定义，则使用均匀分布初始化偏置参数
  if (bias.defined()) {
    auto [fan_in, fan_out] =
        torch::nn::init::_calculate_fan_in_and_fan_out(weight);
    const auto bound = 1 / std::sqrt(fan_in);
    torch::nn::init::uniform_(bias, -bound, bound);
  }
}

// 打印 LinearImpl 对象的信息到输出流
void LinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Linear(in_features=" << options.in_features()
         << ", out_features=" << options.out_features()
         << ", bias=" << options.bias() << ")";
}

// 前向传播函数，使用线性函数对输入进行处理
Tensor LinearImpl::forward(const Tensor& input) {
  return F::linear(input, weight, bias);
}

// ============================================================================

// FlattenImpl 类的构造函数实现
FlattenImpl::FlattenImpl(const FlattenOptions& options_) : options(options_) {}

// 重置函数，暂无具体实现
void FlattenImpl::reset() {}

// 打印 FlattenImpl 对象的信息到输出流
void FlattenImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Flatten(start_dim=" << options.start_dim()
         << ", end_dim=" << options.end_dim() << ")";
}

// 前向传播函数，对输入进行扁平化处理
Tensor FlattenImpl::forward(const Tensor& input) {
  return input.flatten(options.start_dim(), options.end_dim());
}

// ============================================================================

// UnflattenImpl 类的构造函数实现
UnflattenImpl::UnflattenImpl(UnflattenOptions options_)
    : options(std::move(options_)) {}

// 重置函数，暂无具体实现
void UnflattenImpl::reset() {}

// 打印 UnflattenImpl 对象的信息到输出流
void UnflattenImpl::pretty_print(std::ostream& stream) const {
  auto namedshape = options.namedshape();
  if (!namedshape.empty()) {
    stream << "torch::nn::Unflatten(dim=\"" << options.dimname()
           << "\", unflattened_size={";
    size_t i = 0;
    for (; i < namedshape.size() - 1; ++i) {
      stream << "{\"" << std::get<0>(namedshape[i]) << "\", "
             << std::get<1>(namedshape[i]) << "}, ";
    }
    // 如果 namedshape[i] 的第一个元素是字符串，表示是命名形状，需要进行特殊处理
    stream << "{\"" << std::get<0>(namedshape[i]) << "\", "
           << std::get<1>(namedshape[i]) << "}})";
  } else {
    // 如果 namedshape[i] 的第一个元素不是字符串，表示是普通形状，构建对应的 Unflatten 操作
    stream << "torch::nn::Unflatten(dim=" << options.dim()
           << ", unflattened_size={";
    auto sizes = options.sizes();
    size_t i = 0;
    // 遍历 sizes 数组，生成 Unflatten 操作所需的尺寸信息
    for (; i < sizes.size() - 1; ++i) {
      stream << sizes[i] << ", ";
    }
    // 添加最后一个尺寸信息，结束 Unflatten 操作的构建
    stream << sizes[i] << "})";
  }
}

Tensor UnflattenImpl::forward(const Tensor& input) {
  auto namedshape = options.namedshape();  // 获取选项中的命名形状
  if (!namedshape.empty()) {  // 如果命名形状不为空
    auto dimname =
        torch::Dimname::fromSymbol(torch::Symbol::dimname(options.dimname()));  // 从选项中获取维度名称的符号，创建维度名称对象
    std::vector<int64_t> sizes;  // 创建存储尺寸的向量
    std::vector<torch::Dimname> names;  // 创建存储维度名称的向量
    for (auto i : namedshape) {  // 遍历命名形状
      names.push_back(
          torch::Dimname::fromSymbol(torch::Symbol::dimname(std::get<0>(i))));  // 获取每个命名形状的维度名称，并添加到名称向量中
      sizes.push_back(std::get<1>(i));  // 获取每个命名形状的尺寸，并添加到尺寸向量中
    }
    return input.unflatten(dimname, sizes, names);  // 调用输入张量的 unflatten 方法，根据维度名称、尺寸和名称重新组织张量
  }
  return input.unflatten(options.dim(), options.sizes());  // 如果没有命名形状，则根据选项中的维度和尺寸重新组织输入张量
}

// ============================================================================

BilinearImpl::BilinearImpl(const BilinearOptions& options_)
    : options(options_) {  // BilinearImpl 类的构造函数，初始化选项
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();  // 调用 reset 方法进行初始化
}

void BilinearImpl::reset() {
  weight = register_parameter(
      "weight",
      torch::empty(
          {options.out_features(),
           options.in1_features(),
           options.in2_features()}));  // 注册参数 weight，并用给定维度的空张量进行初始化
  if (options.bias()) {  // 如果选项中指定存在偏置
    bias = register_parameter("bias", torch::empty(options.out_features()));  // 注册参数 bias，并用给定维度的空张量进行初始化
  } else {
    bias = register_parameter("bias", torch::Tensor(), /*requires_grad=*/false);  // 否则，注册不需要梯度的空张量作为 bias
  }

  reset_parameters();  // 调用 reset_parameters 方法进行参数的具体初始化
}

void BilinearImpl::reset_parameters() {
  const auto bound = 1.0 / std::sqrt(weight.size(1));  // 计算初始化参数的上下界
  init::uniform_(weight, -bound, bound);  // 使用均匀分布在指定范围内初始化 weight 参数
  if (bias.defined()) {  // 如果 bias 参数已定义
    init::uniform_(bias, -bound, bound);  // 使用均匀分布在指定范围内初始化 bias 参数
  }
}

void BilinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Bilinear(in1_features=" << options.in1_features()
         << ", in2_features=" << options.in2_features()
         << ", out_features=" << options.out_features()
         << ", bias=" << options.bias() << ")";  // 打印 BilinearImpl 对象的信息，包括输入特征数、输出特征数及是否有偏置
}

Tensor BilinearImpl::forward(const Tensor& input1, const Tensor& input2) {
  return F::bilinear(input1, input2, weight, bias);  // 调用 F::bilinear 函数进行双线性操作的前向传播
}

} // namespace nn
} // namespace torch
```