# `.\pytorch\torch\csrc\api\src\nn\modules\pooling.cpp`

```
#include <torch/nn/modules/pooling.h>  // 引入 Torch 中的池化模块头文件

#include <torch/expanding_array.h>  // 引入 Torch 中的扩展数组头文件

namespace F = torch::nn::functional;  // 定义别名 F，指向 torch::nn::functional 命名空间

namespace torch {
namespace nn {

template <size_t D, typename Derived>
AvgPoolImpl<D, Derived>::AvgPoolImpl(const AvgPoolOptions<D>& options_)
    : options(options_) {}  // AvgPoolImpl 类的构造函数，初始化选项

template <size_t D, typename Derived>
void AvgPoolImpl<D, Derived>::reset() {}  // AvgPoolImpl 类的重置函数，空实现

template <size_t D, typename Derived>
void AvgPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::AvgPool" << D << "d"
         << "(kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride() << ", padding=" << options.padding()
         << ")";
}  // AvgPoolImpl 类的 pretty_print 函数，打印池化层的参数信息到流中

Tensor AvgPool1dImpl::forward(const Tensor& input) {
  return F::detail::avg_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad());
}  // AvgPool1dImpl 类的前向传播函数，执行一维平均池化操作

Tensor AvgPool2dImpl::forward(const Tensor& input) {
  return F::detail::avg_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad(),
      options.divisor_override());
}  // AvgPool2dImpl 类的前向传播函数，执行二维平均池化操作

Tensor AvgPool3dImpl::forward(const Tensor& input) {
  return F::detail::avg_pool3d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad(),
      options.divisor_override());
}  // AvgPool3dImpl 类的前向传播函数，执行三维平均池化操作

template class AvgPoolImpl<1, AvgPool1dImpl>;  // 实例化 AvgPoolImpl 类模板，1 维平均池化
template class AvgPoolImpl<2, AvgPool2dImpl>;  // 实例化 AvgPoolImpl 类模板，2 维平均池化
template class AvgPoolImpl<3, AvgPool3dImpl>;  // 实例化 AvgPoolImpl 类模板，3 维平均池化

// ============================================================================

template <size_t D, typename Derived>
MaxPoolImpl<D, Derived>::MaxPoolImpl(const MaxPoolOptions<D>& options_)
    : options(options_) {}  // MaxPoolImpl 类的构造函数，初始化选项

template <size_t D, typename Derived>
void MaxPoolImpl<D, Derived>::reset() {}  // MaxPoolImpl 类的重置函数，空实现

template <size_t D, typename Derived>
void MaxPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::MaxPool" << D << "d"
         << "(kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride() << ", padding=" << options.padding()
         << ", dilation=" << options.dilation()
         << ", ceil_mode=" << options.ceil_mode() << ")";
}  // MaxPoolImpl 类的 pretty_print 函数，打印池化层的参数信息到流中

Tensor MaxPool1dImpl::forward(const Tensor& input) {
  return F::detail::max_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}  // MaxPool1dImpl 类的前向传播函数，执行一维最大池化操作

std::tuple<Tensor, Tensor> MaxPool1dImpl::forward_with_indices(
    const Tensor& input) {
  return F::detail::max_pool1d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}  // MaxPool1dImpl 类的前向传播函数，执行一维最大池化操作并返回索引
// 返回二维最大池化操作的结果张量，调用了F命名空间中的detail::max_pool2d函数
Tensor MaxPool2dImpl::forward(const Tensor& input) {
  return F::detail::max_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

// 返回带索引的二维最大池化操作的结果张量和索引张量，调用了F命名空间中的detail::max_pool2d_with_indices函数
std::tuple<Tensor, Tensor> MaxPool2dImpl::forward_with_indices(
    const Tensor& input) {
  return F::detail::max_pool2d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

// 返回三维最大池化操作的结果张量，调用了F命名空间中的detail::max_pool3d函数
Tensor MaxPool3dImpl::forward(const Tensor& input) {
  return F::detail::max_pool3d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

// 返回带索引的三维最大池化操作的结果张量和索引张量，调用了F命名空间中的detail::max_pool3d_with_indices函数
std::tuple<Tensor, Tensor> MaxPool3dImpl::forward_with_indices(
    const Tensor& input) {
  return F::detail::max_pool3d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

// 实例化一维最大池化操作的模板类MaxPoolImpl，其中包含一维最大池化的具体实现MaxPool1dImpl
template class MaxPoolImpl<1, MaxPool1dImpl>;
// 实例化二维最大池化操作的模板类MaxPoolImpl，其中包含二维最大池化的具体实现MaxPool2dImpl
template class MaxPoolImpl<2, MaxPool2dImpl>;
// 实例化三维最大池化操作的模板类MaxPoolImpl，其中包含三维最大池化的具体实现MaxPool3dImpl;

// ============================================================================

// 返回一维自适应最大池化操作的结果张量，调用了F命名空间中的detail::adaptive_max_pool1d函数
Tensor AdaptiveMaxPool1dImpl::forward(const Tensor& input) {
  return F::detail::adaptive_max_pool1d(input, options.output_size());
}

// 返回带索引的一维自适应最大池化操作的结果张量和索引张量，调用了F命名空间中的detail::adaptive_max_pool1d_with_indices函数
std::tuple<Tensor, Tensor> AdaptiveMaxPool1dImpl::forward_with_indices(
    const Tensor& input) {
  return F::detail::adaptive_max_pool1d_with_indices(
      input, options.output_size());
}

// 返回二维自适应最大池化操作的结果张量，调用了F命名空间中的detail::adaptive_max_pool2d函数
Tensor AdaptiveMaxPool2dImpl::forward(const Tensor& input) {
  return F::detail::adaptive_max_pool2d(input, options.output_size());
}

// 返回带索引的二维自适应最大池化操作的结果张量和索引张量，调用了F命名空间中的detail::adaptive_max_pool2d_with_indices函数
std::tuple<Tensor, Tensor> AdaptiveMaxPool2dImpl::forward_with_indices(
    const Tensor& input) {
  return F::detail::adaptive_max_pool2d_with_indices(
      input, options.output_size());
}

// 返回三维自适应最大池化操作的结果张量，调用了F命名空间中的detail::adaptive_max_pool3d函数
Tensor AdaptiveMaxPool3dImpl::forward(const Tensor& input) {
  return F::detail::adaptive_max_pool3d(input, options.output_size());
}

// 返回带索引的三维自适应最大池化操作的结果张量和索引张量，调用了F命名空间中的detail::adaptive_max_pool3d_with_indices函数
std::tuple<Tensor, Tensor> AdaptiveMaxPool3dImpl::forward_with_indices(
    const Tensor& input) {
  return F::detail::adaptive_max_pool3d_with_indices(
      input, options.output_size());
}

// 实例化一维自适应最大池化操作的模板类AdaptiveMaxPoolImpl，其中包含一维自适应最大池化的具体实现AdaptiveMaxPool1dImpl
template class AdaptiveMaxPoolImpl<1, ExpandingArray<1>, AdaptiveMaxPool1dImpl>;
// 实例化二维自适应最大池化操作的模板类AdaptiveMaxPoolImpl，其中包含二维自适应最大池化的具体实现AdaptiveMaxPool2dImpl
template class AdaptiveMaxPoolImpl<
    2,
    ExpandingArrayWithOptionalElem<2>,
    AdaptiveMaxPool2dImpl>;
// 实例化三维自适应最大池化操作的模板类AdaptiveMaxPoolImpl，其中包含三维自适应最大池化的具体实现AdaptiveMaxPool3dImpl;

// ============================================================================

// 返回一维自适应平均池化操作的结果张量，调用了F命名空间中的detail::adaptive_avg_pool1d函数
Tensor AdaptiveAvgPool1dImpl::forward(const Tensor& input) {
  return F::detail::adaptive_avg_pool1d(input, options.output_size());
}

// 返回二维自适应平均池化操作的结果张量，调用了F命名空间中的detail::adaptive_avg_pool2d函数
Tensor AdaptiveAvgPool2dImpl::forward(const Tensor& input) {
  return F::detail::adaptive_avg_pool2d(input, options.output_size());
}
// 返回一个适应性平均池化操作的结果张量
Tensor AdaptiveAvgPool3dImpl::forward(const Tensor& input) {
  return F::detail::adaptive_avg_pool3d(input, options.output_size());
}

// 实例化自适应平均池化操作模板类，处理1维情况
template class AdaptiveAvgPoolImpl<1, ExpandingArray<1>, AdaptiveAvgPool1dImpl>;
// 实例化自适应平均池化操作模板类，处理2维情况
template class AdaptiveAvgPoolImpl<
    2,
    ExpandingArrayWithOptionalElem<2>,
    AdaptiveAvgPool2dImpl>;
// 实例化自适应平均池化操作模板类，处理3维情况
template class AdaptiveAvgPoolImpl<
    3,
    ExpandingArrayWithOptionalElem<3>,
    AdaptiveAvgPool3dImpl>;

// ============================================================================

// 构造函数，初始化最大反池化操作的选项
template <size_t D, typename Derived>
MaxUnpoolImpl<D, Derived>::MaxUnpoolImpl(const MaxUnpoolOptions<D>& options_)
    : options(options_) {}

// 重置最大反池化操作
template <size_t D, typename Derived>
void MaxUnpoolImpl<D, Derived>::reset() {}

// 打印最大反池化操作的信息到输出流
template <size_t D, typename Derived>
void MaxUnpoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::MaxUnpool" << D << "d"
         << "(kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride() << ", padding=" << options.padding()
         << ")";
}

// 执行1维最大反池化操作
Tensor MaxUnpool1dImpl::forward(
    const Tensor& input,
    const Tensor& indices,
    const std::optional<std::vector<int64_t>>& output_size) {
  return F::detail::max_unpool1d(
      input,
      indices,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      output_size);
}

// 执行2维最大反池化操作
Tensor MaxUnpool2dImpl::forward(
    const Tensor& input,
    const Tensor& indices,
    const std::optional<std::vector<int64_t>>& output_size) {
  return F::detail::max_unpool2d(
      input,
      indices,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      output_size);
}

// 执行3维最大反池化操作
Tensor MaxUnpool3dImpl::forward(
    const Tensor& input,
    const Tensor& indices,
    const std::optional<std::vector<int64_t>>& output_size) {
  return F::detail::max_unpool3d(
      input,
      indices,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      output_size);
}

// 实例化最大反池化操作模板类，处理1维情况
template class MaxUnpoolImpl<1, MaxUnpool1dImpl>;
// 实例化最大反池化操作模板类，处理2维情况
template class MaxUnpoolImpl<2, MaxUnpool2dImpl>;
// 实例化最大反池化操作模板类，处理3维情况
template class MaxUnpoolImpl<3, MaxUnpool3dImpl>;

// ============================================================================

// 构造函数，初始化分数最大池化操作的选项
FractionalMaxPool2dImpl::FractionalMaxPool2dImpl(
    FractionalMaxPool2dOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

// 重置分数最大池化操作
void FractionalMaxPool2dImpl::reset() {
  // 注册缓冲区，存储随机采样数据
  _random_samples =
      register_buffer("_random_samples", options._random_samples());
  // 检查是否同时指定了输出大小和池化比例，否则报错
  if (options.output_size() == c10::nullopt &&
      options.output_ratio() == c10::nullopt) {
    TORCH_CHECK(
        false,
        "FractionalMaxPool2d requires specifying either ",
        "an output size, or a pooling ratio");
  }
  // 检查是否同时指定了输出大小和池化比例，否则报错
  if (options.output_size() != c10::nullopt &&
      options.output_ratio() != c10::nullopt) {
    // 检查条件，如果 output_size 和 output_ratio 同时被指定，则抛出错误信息
    TORCH_CHECK(
        false, "only one of output_size and output_ratio may be specified");
    // 如果 options 中指定了 output_ratio 而非 output_size
    if (options.output_ratio() != c10::nullopt) {
        // 获取 options 中的 output_ratio 并创建一个双精度浮点数数组的引用
        at::ArrayRef<double> output_ratio =
            at::ArrayRef<double>(options.output_ratio().value());
        // 检查 output_ratio[0] 和 output_ratio[1] 是否在 (0, 1) 之间
        if (!(0 < output_ratio[0] && output_ratio[0] < 1 && 0 < output_ratio[1] &&
              output_ratio[1] < 1)) {
            // 如果不在指定范围内，则抛出错误信息，显示当前的 output_ratio 值
            TORCH_CHECK(
                false,
                "output_ratio must be between 0 and 1 (got ",
                output_ratio,
                ")");
        }
    }
}

// 以下代码定义了 FractionalMaxPool2dImpl 类的 forward 方法，用于执行 2D 分数最大池化操作
Tensor FractionalMaxPool2dImpl::forward(const Tensor& input) {
  return F::detail::fractional_max_pool2d(
      input,
      options.kernel_size(),   // 使用选项中的 kernel_size 执行池化操作
      options.output_size(),   // 使用选项中的 output_size 指定输出大小
      options.output_ratio(),  // 使用选项中的 output_ratio 指定输出比例
      _random_samples);        // 使用成员变量 _random_samples 作为随机采样
}

// 以下代码定义了 FractionalMaxPool2dImpl 类的 forward_with_indices 方法，返回 2D 分数最大池化结果及其索引
std::tuple<Tensor, Tensor> FractionalMaxPool2dImpl::forward_with_indices(
    const Tensor& input) {
  return F::detail::fractional_max_pool2d_with_indices(
      input,
      options.kernel_size(),   // 使用选项中的 kernel_size 执行池化操作
      options.output_size(),   // 使用选项中的 output_size 指定输出大小
      options.output_ratio(),  // 使用选项中的 output_ratio 指定输出比例
      _random_samples);        // 使用成员变量 _random_samples 作为随机采样
}

// 以下代码定义了 FractionalMaxPool2dImpl 类的 pretty_print 方法，用于将对象信息打印到输出流
void FractionalMaxPool2dImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::FractionalMaxPool2d()";  // 打印对象信息
}

// 以下代码定义了 FractionalMaxPool3dImpl 类的构造函数，接受 FractionalMaxPool3dOptions 类型的参数
FractionalMaxPool3dImpl::FractionalMaxPool3dImpl(
    FractionalMaxPool3dOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();  // 调用 reset 方法进行初始化
}

// 以下代码定义了 FractionalMaxPool3dImpl 类的 reset 方法，用于重置对象状态
void FractionalMaxPool3dImpl::reset() {
  _random_samples =
      register_buffer("_random_samples", options._random_samples());  // 注册 _random_samples 缓冲区
  if (options.output_size() == c10::nullopt &&
      options.output_ratio() == c10::nullopt) {
    TORCH_CHECK(
        false,
        "FractionalMaxPool3d requires specifying either ",
        "an output size, or a pooling ratio");  // 检查必须指定 output_size 或 output_ratio
  }
  if (options.output_size() != c10::nullopt &&
      options.output_ratio() != c10::nullopt) {
    TORCH_CHECK(
        false, "only one of output_size and output_ratio may be specified");  // 检查只能指定一个 output_size 或 output_ratio
  }
  if (options.output_ratio() != c10::nullopt) {
    at::ArrayRef<double> output_ratio =
        at::ArrayRef<double>(options.output_ratio().value());
    if (!(0 < output_ratio[0] && output_ratio[0] < 1 && 0 < output_ratio[1] &&
          output_ratio[1] < 1 && 0 < output_ratio[2] && output_ratio[2] < 1)) {
      TORCH_CHECK(
          false,
          "output_ratio must be between 0 and 1 (got ",
          output_ratio,
          ")");  // 检查 output_ratio 必须在 0 到 1 之间
    }
  }
}

// 以下代码定义了 FractionalMaxPool3dImpl 类的 forward 方法，用于执行 3D 分数最大池化操作
Tensor FractionalMaxPool3dImpl::forward(const Tensor& input) {
  return F::detail::fractional_max_pool3d(
      input,
      options.kernel_size(),   // 使用选项中的 kernel_size 执行池化操作
      options.output_size(),   // 使用选项中的 output_size 指定输出大小
      options.output_ratio(),  // 使用选项中的 output_ratio 指定输出比例
      _random_samples);        // 使用成员变量 _random_samples 作为随机采样
}

// 以下代码定义了 FractionalMaxPool3dImpl 类的 forward_with_indices 方法，返回 3D 分数最大池化结果及其索引
std::tuple<Tensor, Tensor> FractionalMaxPool3dImpl::forward_with_indices(
    const Tensor& input) {
  return F::detail::fractional_max_pool3d_with_indices(
      input,
      options.kernel_size(),   // 使用选项中的 kernel_size 执行池化操作
      options.output_size(),   // 使用选项中的 output_size 指定输出大小
      options.output_ratio(),  // 使用选项中的 output_ratio 指定输出比例
      _random_samples);        // 使用成员变量 _random_samples 作为随机采样
}

// 以下代码定义了 FractionalMaxPool3dImpl 类的 pretty_print 方法，用于将对象信息打印到输出流
void FractionalMaxPool3dImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::FractionalMaxPool3d()";  // 打印对象信息
}

// ============================================================================

// 以下代码定义了 LPPoolImpl 类的构造函数，接受 LPPoolOptions<D> 类型的参数
template <size_t D, typename Derived>
LPPoolImpl<D, Derived>::LPPoolImpl(const LPPoolOptions<D>& options_)
    : options(options_) {}

// 以下代码定义了 LPPoolImpl 类的 reset 方法的模板实现，用于重置对象状态
template <size_t D, typename Derived>
void LPPoolImpl<D, Derived>::reset() {}

// 以下代码定义了 LPPoolImpl 类的成员函数的模板实现，但未完整列出
// 将 LPPoolImpl 模板类的 pretty_print 方法实现为输出 LPPool 的详细信息
void LPPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  // 将流设置为输出布尔值时显示 true/false
  stream << std::boolalpha << "torch::nn::LPPool" << D << "d("
         // 输出 LPPool 的参数信息：规范类型、核大小、步长、取整模式
         << "norm_type=" << options.norm_type() << ", "
         << "kernel_size=" << options.kernel_size() << ", "
         << "stride=" << options.stride() << ", "
         << "ceil_mode=" << options.ceil_mode() << ")";
}

// LPPool1dImpl 类的 forward 方法，调用 F::detail::lp_pool1d 实现 1 维 LPPool 操作
Tensor LPPool1dImpl::forward(const Tensor& input) {
  return F::detail::lp_pool1d(
      input,
      options.norm_type(),     // 使用的规范类型
      options.kernel_size(),   // 使用的核大小
      options.stride(),        // 使用的步长
      options.ceil_mode());    // 使用的取整模式
}

// 实例化 LPPoolImpl 模板类为 LPPool1dImpl，模板参数为 1
template class LPPoolImpl<1, LPPool1dImpl>;

// LPPool2dImpl 类的 forward 方法，调用 F::detail::lp_pool2d 实现 2 维 LPPool 操作
Tensor LPPool2dImpl::forward(const Tensor& input) {
  return F::detail::lp_pool2d(
      input,
      options.norm_type(),     // 使用的规范类型
      options.kernel_size(),   // 使用的核大小
      options.stride(),        // 使用的步长
      options.ceil_mode());    // 使用的取整模式
}

// 实例化 LPPoolImpl 模板类为 LPPool2dImpl，模板参数为 2
template class LPPoolImpl<2, LPPool2dImpl>;

// LPPool3dImpl 类的 forward 方法，调用 F::detail::lp_pool3d 实现 3 维 LPPool 操作
Tensor LPPool3dImpl::forward(const Tensor& input) {
  return F::detail::lp_pool3d(
      input,
      options.norm_type(),     // 使用的规范类型
      options.kernel_size(),   // 使用的核大小
      options.stride(),        // 使用的步长
      options.ceil_mode());    // 使用的取整模式
}

// 实例化 LPPoolImpl 模板类为 LPPool3dImpl，模板参数为 3
template class LPPoolImpl<3, LPPool3dImpl>;

// 结束 torch::nn 命名空间
} // namespace nn

// 结束 torch 命名空间
} // namespace torch
```