# `.\pytorch\torch\csrc\api\src\nn\modules\padding.cpp`

```py
// 包含 torch 库中的 padding 头文件
#include <torch/nn/modules/padding.h>

// 包含 torch 库中的 expanding_array 头文件
#include <torch/expanding_array.h>

// 定义命名空间 F 为 torch::nn::functional
namespace F = torch::nn::functional;

// 定义 torch::nn 命名空间
namespace torch {
namespace nn {

// ReflectionPadImpl 类的模板实现，构造函数
template <size_t D, typename Derived>
ReflectionPadImpl<D, Derived>::ReflectionPadImpl(
    const ReflectionPadOptions<D>& options_)
    : options(options_) {}

// 重置函数的实现，无操作
template <size_t D, typename Derived>
void ReflectionPadImpl<D, Derived>::reset() {}

// 前向传播函数的实现，调用 F::detail::pad 对输入进行反射填充
template <size_t D, typename Derived>
Tensor ReflectionPadImpl<D, Derived>::forward(const Tensor& input) {
  return F::detail::pad(input, options.padding(), torch::kReflect, 0);
}

// 打印信息函数的实现，输出类的信息和填充参数
template <size_t D, typename Derived>
void ReflectionPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReflectionPad" << D << "d"
         << "(padding=" << options.padding() << ")";
}

// 实例化 ReflectionPadImpl 类模板，分别用于 1D、2D、3D
template class ReflectionPadImpl<1, ReflectionPad1dImpl>;
template class ReflectionPadImpl<2, ReflectionPad2dImpl>;
template class ReflectionPadImpl<3, ReflectionPad3dImpl>;

// ============================================================================

// ReplicationPadImpl 类的模板实现，构造函数
template <size_t D, typename Derived>
ReplicationPadImpl<D, Derived>::ReplicationPadImpl(
    const ReplicationPadOptions<D>& options_)
    : options(options_) {}

// 重置函数的实现，无操作
template <size_t D, typename Derived>
void ReplicationPadImpl<D, Derived>::reset() {}

// 前向传播函数的实现，调用 F::detail::pad 对输入进行复制填充
template <size_t D, typename Derived>
Tensor ReplicationPadImpl<D, Derived>::forward(const Tensor& input) {
  return F::detail::pad(input, options.padding(), torch::kReplicate, 0);
}

// 打印信息函数的实现，输出类的信息和填充参数
template <size_t D, typename Derived>
void ReplicationPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReplicationPad" << D << "d"
         << "(padding=" << options.padding() << ")";
}

// 实例化 ReplicationPadImpl 类模板，分别用于 1D、2D、3D
template class ReplicationPadImpl<1, ReplicationPad1dImpl>;
template class ReplicationPadImpl<2, ReplicationPad2dImpl>;
template class ReplicationPadImpl<3, ReplicationPad3dImpl>;

// ============================================================================

// ZeroPadImpl 类的模板实现，构造函数
template <size_t D, typename Derived>
ZeroPadImpl<D, Derived>::ZeroPadImpl(const ZeroPadOptions<D>& options_)
    : options(options_) {}

// 重置函数的实现，无操作
template <size_t D, typename Derived>
void ZeroPadImpl<D, Derived>::reset() {}

// 前向传播函数的实现，调用 F::detail::pad 对输入进行常量填充
template <size_t D, typename Derived>
Tensor ZeroPadImpl<D, Derived>::forward(const Tensor& input) {
  return F::detail::pad(input, options.padding(), torch::kConstant, 0);
}

// 打印信息函数的实现，输出类的信息和填充参数
template <size_t D, typename Derived>
void ZeroPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ZeroPad" << D << "d"
         << "(padding=" << options.padding() << ")";
}

// 实例化 ZeroPadImpl 类模板，分别用于 1D、2D、3D
template class ZeroPadImpl<1, ZeroPad1dImpl>;
template class ZeroPadImpl<2, ZeroPad2dImpl>;
template class ZeroPadImpl<3, ZeroPad3dImpl>;

// ============================================================================

// ConstantPadImpl 类的模板实现，构造函数
template <size_t D, typename Derived>
ConstantPadImpl<D, Derived>::ConstantPadImpl(
    const ConstantPadOptions<D>& options_)
    : options(options_) {}

// 重置函数的实现，无操作
template <size_t D, typename Derived>
// 重置 ConstantPadImpl 类的成员函数 reset，但是这里是空实现，没有具体的操作

// 实现 ConstantPadImpl 类模板的 forward 函数，用于对输入张量进行常数填充操作
template <size_t D, typename Derived>
Tensor ConstantPadImpl<D, Derived>::forward(const Tensor& input) {
  // 调用 F::detail::pad 函数对输入张量进行填充操作
  return F::detail::pad(
      input, options.padding(), torch::kConstant, options.value());
}

// 实现 ConstantPadImpl 类模板的 pretty_print 函数，用于美化打印输出信息到流中
template <size_t D, typename Derived>
void ConstantPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  // 打印类的信息，包括维度 D，填充选项和填充值
  stream << "torch::nn::ConstantPad" << D << "d"
         << "(padding=" << options.padding() << ", value=" << options.value()
         << ")";
}

// 实例化 ConstantPadImpl 模板类分别为 ConstantPad1dImpl, ConstantPad2dImpl, ConstantPad3dImpl
template class ConstantPadImpl<1, ConstantPad1dImpl>;
template class ConstantPadImpl<2, ConstantPad2dImpl>;
template class ConstantPadImpl<3, ConstantPad3dImpl>;

// 结束 torch::nn 命名空间
} // namespace nn

// 结束 torch 命名空间
} // namespace torch
```