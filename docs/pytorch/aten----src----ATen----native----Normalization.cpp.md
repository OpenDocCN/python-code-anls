# `.\pytorch\aten\src\ATen\native\Normalization.cpp`

```py
// 定义宏，用于标记仅支持方法操作符的 Torch 版本
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 Tensor 类的头文件以及相关依赖库
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>

// 包含 CUDA 钩子接口的头文件
#include <ATen/detail/CUDAHooksInterface.h>

// 包含 CPU 环路运算的头文件
#include <ATen/native/cpu/Loops.h>

// 包含批量归一化的头文件
#include <ATen/native/batch_norm.h>

// 包含标准化操作的头文件
#include <ATen/native/Normalization.h>

// 包含 Resize 相关的头文件
#include <ATen/native/Resize.h>

// 包含 CPU 混合数据类型的头文件
#include <ATen/native/cpu/mixed_data_type.h>

// 包含 c10 实用工具中的 irange 函数的头文件
#include <c10/util/irange.h>

// 包含 OpMathType 的头文件
#include <ATen/OpMathType.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含下列头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含以下批量归一化实现的特定头文件
#else
#include <ATen/ops/_batch_norm_impl_index.h>
#include <ATen/ops/_batch_norm_impl_index_backward_native.h>
#include <ATen/ops/_batch_norm_impl_index_native.h>
#include <ATen/ops/_native_batch_norm_legit_native.h>
#include <ATen/ops/_native_batch_norm_legit_no_training.h>
#include <ATen/ops/_native_batch_norm_legit_no_training_native.h>
#include <ATen/ops/_batch_norm_with_update.h>
#include <ATen/ops/_batch_norm_with_update_native.h>
#include <ATen/ops/_batch_norm_no_update.h>
#include <ATen/ops/_batch_norm_no_update_native.h>
#include <ATen/ops/batch_norm_backward_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/batch_norm.h>
#include <ATen/ops/batch_norm_native.h>
#include <ATen/ops/batch_norm_update_stats_native.h>
#include <ATen/ops/cudnn_batch_norm.h>
#include <ATen/ops/cudnn_batch_norm_backward.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/instance_norm_native.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/miopen_batch_norm.h>
#include <ATen/ops/miopen_batch_norm_backward.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_batch_norm_backward.h>
#include <ATen/ops/native_batch_norm_backward_native.h>
#include <ATen/ops/native_batch_norm_native.h>
#include <ATen/ops/_native_batch_norm_legit.h>
#include <ATen/ops/renorm_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sqrt.h>
#endif

// 包含 SymIntArrayRef 类的头文件
#include <c10/core/SymIntArrayRef.h>

// 包含一些常用的标准库头文件
#include <utility>
#include <vector>

// 定义 miopen 维度的最大值
static const int MIOPEN_DIM_MAX = 5;

// 进入 at 命名空间中的 meta 命名空间
namespace at::meta {

// 定义 TORCH_META_FUNC 宏，用于重命名 renorm 函数
TORCH_META_FUNC(renorm)(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm) {
  // 检查 p 必须是实数值
  TORCH_CHECK(!p.isComplex(), "renorm: p must be real-valued");
  // 检查 p 必须大于 0
  TORCH_CHECK(p.toDouble() > 0.0, "renorm: non-positive-norm not supported");
  // 检查 maxnorm 必须是实数值
  TORCH_CHECK(!maxnorm.isComplex(), "renorm: maxnorm must be real-valued");
  // 检查 maxnorm 必须大于等于 0
  TORCH_CHECK(maxnorm.toDouble() >= 0.0,
              "renorm: expected maxnorm to be >= 0 but got ", maxnorm.toDouble());
  // 获取输入张量的维度
  const auto ndim = self.dim();
  // 检查输入张量的维度必须大于 1
  TORCH_CHECK(ndim > 1, "renorm: input needs at least 2 dimensions, got ", ndim, " dimensions");
  // 设置输出的原始步长
  set_output_raw_strided(0, self.sizes(), {}, self.options());
}

}  // namespace at::meta

// 进入 at 命名空间中的 native 命名空间
namespace at::native {

// 这里是文件的末尾，无需添加更多注释
DEFINE_DISPATCH(batch_norm_cpu_stub);
DEFINE_DISPATCH(batch_norm_cpu_collect_stats_stub);
DEFINE_DISPATCH(batch_norm_cpu_backward_stub);
DEFINE_DISPATCH(renorm_scale_factor_stub);

namespace {
  // 检查维度是否匹配输入特征数目，抛出错误信息
  void check_dims_match_num_input_features(const char* arg_name, SymInt expected, SymInt actual){
    TORCH_CHECK(actual == expected,
             arg_name, " should contain ", expected, " elements not ", actual);
  }

  // 如果定义了张量 t，则重复该张量 repeat 次；否则返回未定义的张量
  static inline Tensor repeat_if_defined(const Tensor& t, SymInt repeat) {
    if (t.defined()) {
      return t.repeat_symint(repeat);
    }
    return t;
  }
}

template<typename T>
struct InvStd {
  // 计算标准差的倒数
  T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / std::sqrt(var + epsilon);
    }
    return invstd;
  }
};

template<typename T>
struct Var {
  // 直接返回方差
  T operator()(T var, double epsilon) const {
    return var;
  }
};

static inline bool is_contiguous(const Tensor& t) {
  // 判断张量是否是连续的，或者在指定的内存格式下是连续的
  return t.is_contiguous() || t.is_contiguous(at::MemoryFormat::ChannelsLast) || t.is_contiguous(at::MemoryFormat::ChannelsLast3d);
}

// 对于某些不明确的情况，可能会出现通道优先的连续张量，其建议的内存格式为连续的。
// 参见 https://github.com/pytorch/pytorch/issues/63224 获取详细信息。
static inline MemoryFormat suggest_memory_format_contig(const Tensor& t) {
  // 建议适合的内存格式，如果张量是连续的，则返回 Contiguous；
  // 否则根据特定情况返回 ChannelsLast3d 或 ChannelsLast。
  return t.is_contiguous() ?
    at::MemoryFormat::Contiguous : (t.is_contiguous(at::MemoryFormat::ChannelsLast3d) ?
    at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast);
}

template<typename scalar_t, typename param_t>
std::tuple<Tensor,Tensor,Tensor> batch_norm_cpu_transform_input_template(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& save_mean /* optional */, const Tensor& save_invstd /* optional */,
    const Tensor& running_mean /* optional */, const Tensor& running_var /* optional */,
    bool train, double eps, Tensor& output) {

  bool all_contiguous = is_contiguous(input)
    && is_contiguous(output)
    && (!weight.defined() || weight.is_contiguous())
    && (!bias.defined() || bias.is_contiguous())
    && running_mean.is_contiguous()
    && running_var.is_contiguous();

  // 推断连续路径
  if (all_contiguous) {
    if (input.numel() != 0) {
      // 调用批归一化 CPU 核心函数
      batch_norm_cpu_stub(kCPU, output, input, weight, bias,
          save_mean, save_invstd, running_mean, running_var, train, eps);
    }
    // 返回结果元组
    return std::make_tuple(output, save_mean, save_invstd);
  }

  const int64_t ndim = input.dim();
  // 辅助函数，将 1 维张量转换为与输入广播兼容的 nd 张量
  // 所有元素都放入通道维度
  DimVector sizes(ndim, 1), strides(ndim, 0);
  auto as_nd = [&](const Tensor& t) {
    TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
    sizes[1] = t.sizes()[0];
    strides[1] = t.strides()[0];
  // 返回使用自定义步幅（strides）和大小（sizes）的张量视图
  return t.as_strided(sizes, strides);
};

auto mean = as_nd(train ? save_mean : running_mean); // 根据训练状态选择均值张量
auto invstd = as_nd([&]{ // 使用Lambda函数获取标准差的倒数张量
  if (train) {
    return save_invstd; // 如果在训练阶段，返回保存的标准差的倒数张量
  } else {
    return 1 / at::sqrt(running_var + eps); // 如果不在训练阶段，计算标准差的倒数张量
  }
}());
constexpr bool mixed_type = !std::is_same<scalar_t, param_t>::value; // 检查标量类型是否混合
const auto dtype = mixed_type ? kFloat : input.scalar_type(); // 根据混合类型选择数据类型
auto w = weight.defined() ? as_nd(weight) : // 获取权重张量，如果未定义则使用标量1创建
    at::detail::scalar_tensor_static(1, dtype, kCPU);
auto b = bias.defined() ? as_nd(bias) : // 获取偏置张量，如果未定义则使用标量0创建
    at::detail::scalar_tensor_static(0, dtype, kCPU);

auto iter = TensorIteratorConfig() // 创建张量迭代器配置
  .add_output(output) // 添加输出张量
  .add_input(input) // 添加输入张量
  .add_input(mean) // 添加均值张量
  .add_input(invstd) // 添加标准差的倒数张量
  .add_input(w) // 添加权重张量
  .add_input(b) // 添加偏置张量
  .check_all_same_dtype(false) // 禁止检查所有张量是否具有相同的数据类型
  .promote_inputs_to_common_dtype(false) // 禁止将输入张量提升为共同的数据类型
  .build(); // 构建迭代器

cpu_kernel(iter, [=](scalar_t input, param_t mean, param_t invstd, param_t weight, param_t bias) -> scalar_t {
  // CPU内核函数，计算标准化和线性变换后的输出
  return ((input - mean) * invstd) * weight + bias;
});
return std::make_tuple(output, save_mean, save_invstd); // 返回输出张量以及保存的均值和标准差的倒数
// 结束前一段代码块的大括号，标记其作用域结束
}

// 定义一个模板函数，用于 CPU 上的批归一化参数更新，返回更新后的统计量和变换后的方差
template<typename scalar_t, typename param_t, template<typename T> class VarTransform>
std::tuple<Tensor,Tensor> batch_norm_cpu_update_stats_template(
    // 输入参数：输入张量、运行时均值、运行时方差、动量、eps、保存均值、保存变换后的方差
    const Tensor& input, const Tensor& running_mean, const Tensor& running_var,
    double momentum, double eps, Tensor& save_mean, Tensor& save_var_transform) {

  // 使用 acc_type 定义 accscalar_t 为 scalar_t 的累加类型
  using accscalar_t = at::acc_type<scalar_t, false>;

  // 计算输入张量的通道数
  int64_t n_input = input.size(1);
  // 检查输入张量是否为空，并报错提示
  TORCH_CHECK(input.numel() != 0, "input tensor must have at least one element, but got input_sizes = ", input.sizes());
  // 计算 batch 的大小
  int64_t n = input.numel() / n_input;

  // 检查输入张量是否全连续
  bool all_contiguous = is_contiguous(input);
  // 检查是否为混合类型
  constexpr bool mixed_type = !std::is_same<scalar_t, param_t>::value;
  // 确定张量的数据类型
  const auto dtype = mixed_type ? kFloat : input.scalar_type();

  // 获取保存均值和保存变换后方差的访问器
  auto save_mean_a = save_mean.accessor<param_t, 1>();
  auto save_var_transform_a = save_var_transform.accessor<param_t, 1>();

  // 获取运行时均值和方差的访问器
  auto running_mean_a = conditional_accessor_1d<param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<param_t>(running_var);

  // 如果输入张量全连续
  if (all_contiguous) {
    // 创建用于存储均值和方差累加和的张量
    auto _mean = at::empty({n_input}, input.options().dtype(dtype));
    auto _var_sum = at::empty({n_input}, input.options().dtype(dtype));
    auto _mean_a = _mean.accessor<param_t, 1>();
    auto _var_sum_a = _var_sum.accessor<param_t, 1>();
    auto momentum_ = static_cast<param_t>(momentum);

    // 调用批归一化 CPU 收集统计信息的函数
    batch_norm_cpu_collect_stats_stub(kCPU, _mean, _var_sum, input);

    // 并行处理每个通道
    parallel_for(0, n_input, 1, [&](int64_t b_begin, int64_t b_end) {
      for (const auto f : c10::irange(b_begin, b_end)) {
        // 将计算得到的均值保存到 save_mean_a 中
        save_mean_a[f] = _mean_a[f];
        // 计算变换后的方差并保存到 save_var_transform_a 中
        save_var_transform_a[f] = VarTransform<accscalar_t>{}(_var_sum_a[f] / n, eps);

        // 如果存在运行时均值，则根据动量更新运行时均值
        if (running_mean.defined()) {
          running_mean_a[f] = momentum_ * _mean_a[f] + (1 - momentum_) * running_mean_a[f];
        }
        // 如果存在运行时方差，则根据动量更新运行时方差
        if (running_var.defined()) {
          accscalar_t unbiased_var = _var_sum_a[f] / (n - 1);
          running_var_a[f] = momentum_ * unbiased_var + (1 - momentum_) * running_var_a[f];
        }
      }
    });

    // 返回更新后的保存均值和保存变换后的方差
    return std::make_tuple(save_mean, save_var_transform);
  }

  // 如果输入张量非全连续的情况下执行以下代码
  // 获取通道步长和输入数据指针
  auto channel_stride = input.strides()[1];
  auto in_data = input.data_ptr<scalar_t>();
  // 设置缩减迭代器的配置
  auto reduce_iter = TensorIteratorConfig()
      .add_input(input)
      .resize_outputs(false)
      .declare_static_shape(input.sizes(), /*squash_dims=*/1)
      .check_all_same_dtype(false)
      .promote_inputs_to_common_dtype(false)
      .build();

  // 并行处理每个通道
  parallel_for(0, n_input, 1, [&](int64_t b_begin, int64_t b_end) {
    TensorIterator iter(reduce_iter);
    for (const auto f : c10::irange(b_begin, b_end)) {
      // 对每个输入计算方差
      iter.unsafe_replace_operand(0, in_data + channel_stride * f);
      // 初始化方差总和
      accscalar_t var_sum = 0;
      // 从保存的均值中获取当前均值
      auto mean = static_cast<accscalar_t>(save_mean_a[f]);
      // 并行执行内核函数，计算方差总和
      cpu_serial_kernel(iter, [&](const scalar_t i) -> void {
        // 计算每个元素与均值之差的平方，并累加到方差总和中
        var_sum += (i - mean) * (i - mean);
      });
      // 将方差总和归一化并应用方差变换函数，保存结果
      save_var_transform_a[f] = VarTransform<accscalar_t>{}(var_sum / n, eps);

      // 更新运行时均值和方差的移动平均值
      if (running_mean.defined()) {
        // 使用指数加权平均更新运行时均值
        running_mean_a[f] = momentum * mean + (1 - momentum) * running_mean_a[f];
      }
      if (running_var.defined()) {
        // 计算无偏方差并使用指数加权平均更新运行时方差
        accscalar_t unbiased_var = var_sum / (n - 1);
        running_var_a[f] = momentum * unbiased_var + (1 - momentum) * running_var_a[f];
      }
    }
  });
  // 返回保存的均值和变换后的方差
  return std::make_tuple(save_mean, save_var_transform);
// 如果 grad_input_mask[0] 为 true，则创建一个与 input 相同形状和内存格式的空张量作为梯度输入
Tensor grad_input;
// 如果 grad_input_mask[1] 为 true，则创建一个形状为 {input.size(1)} 的空张量作为梯度权重
Tensor grad_weight;
// 如果 grad_input_mask[2] 为 true，则创建一个形状为 {input.size(1)} 的空张量作为梯度偏置
Tensor grad_bias;

// 如果 grad_input_mask[0] 为 true，表示需要计算梯度输入
if (grad_input_mask[0]) {
    grad_input = at::empty_like(input, input.suggest_memory_format());
}
// 如果 grad_input_mask[1] 为 true，表示需要计算梯度权重
if (grad_input_mask[1]) {
    grad_weight = at::empty({input.size(1)}, input.options().dtype(dtype));
}
// 如果 grad_input_mask[2] 为 true，表示需要计算梯度偏置
if (grad_input_mask[2]) {
    grad_bias = at::empty({input.size(1)}, input.options().dtype(dtype));
}

// 检查是否所有的输入和梯度输出张量都是连续的，并且具有相同的内存格式
bool all_contiguous = is_contiguous(input)
    && is_contiguous(grad_out_)
    && input.suggest_memory_format() == grad_out_.suggest_memory_format();

// 如果所有的输入和梯度输出张量都是连续的
if (all_contiguous) {
    // 如果 grad_input_mask[0] 为 true，为了直接操作指针，需要确保 input 和 grad_out_ 具有相同的内存格式
    if (grad_input_mask[0]) {
        // 创建一个与 input 相同形状和建议的内存格式的空张量作为梯度输入
        grad_input = at::empty_like(input, suggest_memory_format_contig(input));
    }
    // 调用 CPU 版本的批归一化反向传播函数，计算梯度输入、梯度权重和梯度偏置
    batch_norm_cpu_backward_stub(kCPU, grad_input, grad_weight, grad_bias,
        grad_out_, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
    // 返回梯度输入、梯度权重和梯度偏置的元组
    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }

  // 创建weight_a、grad_weight_a和grad_bias_a访问器，用于操作权重和梯度
  auto weight_a = conditional_accessor_1d<const param_t>(weight);
  auto grad_weight_a = conditional_accessor_1d<param_t>(grad_weight);
  auto grad_bias_a = conditional_accessor_1d<param_t>(grad_bias);

  // 获取输入的第二维大小和总元素数目
  int64_t n_input = input.size(1);
  int64_t n = input.numel() / n_input;

  // 创建save_mean_a和save_invstd_a访问器，用于操作保存的均值和逆标准差
  auto save_mean_a = conditional_accessor_1d<const param_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<const param_t>(save_invstd);

  // 创建running_mean_a和running_var_a访问器，用于操作运行时的均值和方差
  auto running_mean_a = conditional_accessor_1d<const param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<const param_t>(running_var);

  // 获取输入的维度数量
  const int64_t ndim = input.dim();

  // 减少除了维度1以外的所有维度
  DimVector reduce_dims(ndim - 1);
  reduce_dims[0] = 0;
  for (const auto i : c10::irange(2, ndim)) {
    reduce_dims[i - 1] = i;
  }

  // 计算grad_out_在指定维度上的求和
  auto sum = at::sum(grad_out_, /*dims=*/reduce_dims);
  auto sum_a = sum.accessor<scalar_t, 1>();

  // 设置TensorIterator以在输入上执行迭代，除了压缩维度1以外的所有维度
  auto reduce_iter = TensorIteratorConfig()
      .add_const_input(input)
      .add_const_input(grad_out_)
      .resize_outputs(false)
      .declare_static_shape(input.sizes(), /*squash_dims=*/1)
      .build();

  // 创建unary_iter和binary_iter TensorIterator，用于执行特定的张量操作
  TensorIterator unary_iter;
  TensorIterator binary_iter;
  if (grad_input_mask[0]) {
    unary_iter.build(
        TensorIteratorConfig()
        .add_output(grad_input)
        .add_const_input(train ? input : grad_out_)
        .resize_outputs(false)
        .declare_static_shape(input.sizes(), /*squash_dims=*/1));

    if (train) {
      binary_iter.build(
          TensorIteratorConfig()
          .add_output(grad_input)
          .add_input(grad_input)
          .add_const_input(grad_out_)
          .resize_outputs(false)
          .declare_static_shape(input.sizes(), /*squash_dims=*/1));
    }
  }
  // 返回梯度输入、梯度权重和梯度偏置的元组
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// _select_batch_norm_backend 函数用于选择适合的批归一化后端实现，根据输入张量及相关参数做出选择
BatchNormBackend _select_batch_norm_backend(
    const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& running_mean,
    const Tensor& running_var, bool training, double eps) {

  // 获取全局上下文
  auto& ctx = at::globalContext();
  // 检查是否启用了 CuDNN
  bool cudnn_enabled = ctx.userEnabledCuDNN();

  // 根据一系列条件判断是否选择 CuDNN 批归一化后端
  if (
      input.is_cuda() // 输入张量在 GPU 上
      && input.scalar_type() != at::kBFloat16 && weight.scalar_type() != at::kBFloat16 // 不使用 BFLOAT16
      && (input.scalar_type() != at::kHalf || weight.scalar_type() == at::kFloat) // 不同时使用半精度和浮点数
      && weight.defined() && bias.defined() // 权重和偏置已定义
      && ((running_mean.defined() && running_var.defined()) // 运行均值和方差已定义
        || (!running_mean.defined() && !running_var.defined() && training)) // 或者在训练时未定义均值和方差
      && (input.dim() >= 3) // 输入张量维度至少为 3
      && ((input.sym_size(0) <= 880801 && training) // 空间维度小于等于 880801 且在训练时
          || (input.sym_size(0) <= 65535 && !training)) // 或者空间维度小于等于 65535 且不在训练时
      && detail::getCUDAHooks().compiledWithCuDNN() // 编译时支持 CuDNN
      && eps >= detail::getCUDAHooks().batchnormMinEpsilonCuDNN() // epsilon 大于等于 CuDNN 最小值
      && cudnn_enabled && detail::getCUDAHooks().versionCuDNN() >= 5110L // CuDNN 版本大于等于 5110
      && input.sym_numel() < std::numeric_limits<std::int32_t>::max() // 某些 CuDNN 内核具有 32 位索引限制
  ) {
    return BatchNormBackend::Cudnn; // 返回 CuDNN 批归一化后端
  }

  // 根据一系列条件判断是否选择 MIOpen 批归一化后端
  if (
      input.is_cuda() // 输入张量在 GPU 上
      && input.dim() <= MIOPEN_DIM_MAX // 输入张量维度小于等于 MIOPEN_DIM_MAX
      && input.scalar_type() != at::kDouble // 不使用双精度
      && input.scalar_type() != at::kBFloat16 // 不使用 BFLOAT16
      && (weight.scalar_type() != at::kHalf) // 权重不使用半精度
      && weight.defined() && bias.defined() // 权重和偏置已定义
      && ((running_mean.defined() && running_var.defined()) // 运行均值和方差已定义
        || (!running_mean.defined() && !running_var.defined() && training)) // 或者在训练时未定义均值和方差
      && (input.dim() >= 3) // 输入张量维度至少为 3
      && detail::getCUDAHooks().compiledWithMIOpen() // 编译时支持 MIOpen
      && cudnn_enabled // 启用了 CuDNN
      && input.suggest_memory_format() != MemoryFormat::ChannelsLast // 推荐的内存格式不是 ChannelsLast
      && input.suggest_memory_format() != MemoryFormat::ChannelsLast3d // 也不是 ChannelsLast3d
  ) {
    return BatchNormBackend::Miopen; // 返回 MIOpen 批归一化后端
  }

  return BatchNormBackend::Native; // 默认返回本地批归一化后端
}


// _batch_norm_impl_index(_backward) 在 JIT 中用于保持运行时选择的后端，同时保持有关使用后端的信息，以便能够使用相应的反向实现。
// XXX: 后端的索引需要在此函数及其 _backward 函数之间保持同步。
// TODO: 移除 cudnn_enabled 参数
std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t> _batch_norm_impl_index(
    const Tensor& input, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& bias_opt /* optional */, const std::optional<Tensor>& running_mean_opt /* optional */, const std::optional<Tensor>& running_var_opt /* optional */,
    bool training, double momentum, double eps, bool cudnn_enabled) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的权重张量中借用数据，转换为可能拥有的张量对象
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  // 获取权重张量的引用
  const Tensor& weight = *weight_maybe_owned;
  // 获取偏置张量的引用，若未提供则返回空张量
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  // 获取运行均值张量的引用，若未提供则返回空张量
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  // 获取运行方差张量的引用，若未提供则返回空张量
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  // 获取输入张量的特征数
  auto num_features = input.sym_sizes()[1];

  // 如果输入张量的元素数为0
  if (input.sym_numel() == 0) {
    // 创建一个空的张量作为保留值
    Tensor reserve = at::empty({0}, input.options().dtype(kByte));
    // 根据输入张量的数据类型创建保存均值的张量
    auto options = input.options().dtype(
        at::toAccumulateType(input.scalar_type(), /*is_cuda=*/input.is_cuda()));
    auto save_mean = at::empty_symint(c10::SymIntArrayRef({num_features}), options);
    // 根据输入张量的数据类型创建保存标准差的张量
    auto save_invstd = at::empty_symint(c10::SymIntArrayRef({std::move(num_features)}), options);

    // 不返回输入张量的视图，不返回空张量以防止破坏梯度链
    auto out = input.clone();
    // 如果定义了权重张量，则将输出乘以权重的第一个元素
    if (weight.defined()) out = out * weight[0];
    // 如果定义了偏置张量，则将输出加上偏置的第一个元素
    if (bias.defined()) out = out + bias[0];
    // 返回输出张量、保存均值张量、保存标准差张量、保留张量和整数0的元组
    return std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t>(
        out, save_mean, save_invstd, reserve, 0);
  }

  // 如果定义了运行均值张量
  if (running_mean.defined()) {
    // 检查运行均值张量的维度是否与输入特征数相匹配
    check_dims_match_num_input_features("running_mean", num_features, running_mean.sym_numel());
  } else if (!training) {
    // 若不是训练模式且未定义运行均值张量，则抛出错误
    AT_ERROR("running_mean must be defined in evaluation mode");
  }
  // 如果定义了运行方差张量
  if (running_var.defined()) {
    // 检查运行方差张量的维度是否与输入特征数相匹配
    check_dims_match_num_input_features("running_var", num_features, running_var.sym_numel());
  } else if (!training) {
    // 若不是训练模式且未定义运行方差张量，则抛出错误
    AT_ERROR("running_var must be defined in evaluation mode");
  }
  // 如果定义了权重张量
  if (weight.defined()) {
    // 检查权重张量的维度是否与输入特征数相匹配
    check_dims_match_num_input_features("weight", num_features, weight.sym_numel());
  }
  // 如果定义了偏置张量
  if (bias.defined()) {
    // 检查偏置张量的维度是否与输入特征数相匹配
    check_dims_match_num_input_features("bias", std::move(num_features), bias.sym_numel());
  }

  // 根据输入、权重、偏置、运行均值、运行方差、训练标志和eps值选择批归一化的后端
  BatchNormBackend backend = _select_batch_norm_backend(input, weight, bias, running_mean, running_var, training, eps);

  // 如果选择的是 Cudnn 后端
  if (backend == BatchNormBackend::Cudnn) {
    // 对输入张量和相关张量进行内存格式连续化
    auto input_c = input.contiguous(input.suggest_memory_format());
    auto weight_c = weight.contiguous();
    auto bias_c = bias.contiguous();
    auto rmean_c = running_mean.defined() ? running_mean.contiguous() : running_mean;
    auto rvar_c = running_var.defined() ? running_var.contiguous() : running_var;

    // 调用 Cudnn 批归一化函数，返回输出、保存均值、保存方差、保留值
    auto [output, save_mean, save_var, reserve] =
        at::cudnn_batch_norm(input_c, weight_c, bias_c, rmean_c, rvar_c,
                             training, momentum, eps);

    // 返回输出张量、保存均值张量、保存方差张量、保留张量和整数1的元组
    return std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t>(
        output, save_mean, save_var, reserve, 1);
  }

  // 创建一个空的张量作为保留值
  Tensor reserve = at::empty({0}, input.options().dtype(kByte));

  // 如果选择的是 Miopen 后端
  if (backend == BatchNormBackend::Miopen) {
    // 返回标准化后的张量，使用 miopen_batch_norm 函数进行计算，处理输入、权重、偏置、
    // 以及可能的运行时均值和方差，如果已定义则使用连续的版本，否则使用原始版本。
    // training 表示是否处于训练模式，momentum 是动量参数，eps 是防止除以零的小值。
    return std::tuple_cat(
             at::miopen_batch_norm(
               input.contiguous(), weight.contiguous(), bias.contiguous(),
               running_mean.defined() ? running_mean.contiguous() : running_mean,
               running_var.defined() ? running_var.contiguous() : running_var,
               training, momentum, eps),
             std::tuple<Tensor>(reserve),  // 将 reserve 转换为单元素元组添加到返回值中
             std::make_tuple(2));          // 创建一个包含整数 2 的元组并添加到返回值中
  }

  // 返回标准化后的张量，使用 native_batch_norm 函数进行计算，处理输入、权重、偏置、
  // 以及运行时均值和方差，training 表示是否处于训练模式，momentum 是动量参数，eps 是防止
  // 除以零的小值。
  return std::tuple_cat(
           at::native_batch_norm(
             input, weight, bias, running_mean, running_var, training, momentum, eps),
           std::tuple<Tensor>(reserve),  // 将 reserve 转换为单元素元组添加到返回值中
           std::make_tuple(0));          // 创建一个包含整数 0 的元组并添加到返回值中
}

std::tuple<Tensor, Tensor, Tensor> _batch_norm_impl_index_backward(
    int64_t impl_index,
    const Tensor& input, const Tensor& grad_output, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& running_mean_opt /* optional */, const std::optional<Tensor>& running_var_opt /* optional */, const std::optional<Tensor>& save_mean_opt /* optional */, const std::optional<Tensor>& save_var_transform_opt /* optional */,
    bool train, double epsilon, std::array<bool, 3> output_mask, const Tensor &reservedSpace) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的权重张量中借用权重张量，以确保在权重未定义时得到一个有效的引用
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;  // 获取权重张量的引用
  // 获取运行均值、方差、保存的均值和变换后的方差张量的引用，若它们未定义则使用默认的空张量
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  const Tensor& save_mean = c10::value_or_else(save_mean_opt, [] {return Tensor();});
  const Tensor& save_var_transform = c10::value_or_else(save_var_transform_opt, [] {return Tensor();});

  // 如果输入张量元素数量为零，则进行梯度计算和返回处理
  if (input.numel() == 0) {
    std::vector<int64_t> dims(input.dim() - 1);
    dims[0] = 0;
    std::iota(dims.begin() + 1, dims.end(), 2);

    // 避免返回空张量破坏梯度链，计算梯度输入、权重和偏置
    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;
    if (output_mask[2]) {
      grad_bias = grad_output.sum(dims);  // 计算偏置梯度
    }
    if (output_mask[1]) {
      grad_weight = (grad_output * input).sum(dims);  // 计算权重梯度
    }
    if (output_mask[0] && weight.defined()) {
      grad_input = grad_output * weight[0];  // 计算输入梯度
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);  // 返回计算得到的梯度
  }

  // 如果实现索引为0或训练模式为假，则调用原生的批量归一化反向传播函数
  if (impl_index == 0 || (!train)) {
    return at::native_batch_norm_backward(grad_output, input, weight, running_mean, running_var, save_mean, save_var_transform, train, epsilon, output_mask);
  } else if (impl_index == 1) {
    // TODO: _batch_norm_impl_index_backward 仅在 JIT 中使用。在 cudnn_batch_norm_backward 内部进行 cudnn NHWC 格式转换
    return at::cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, epsilon, reservedSpace);
  } else if (impl_index == 2) {
    // 调用 miopen_batch_norm_backward 进行批量归一化反向传播
    return at::miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, epsilon);
  }
  // 若实现索引不在已知的范围内，抛出错误信息
  TORCH_INTERNAL_ASSERT(false, "Unsupported impl_index in _batch_norm_impl_index_backward: ", impl_index);
}

// TODO: remove cudnn_enabled arg
// 执行批量归一化操作，包括输入、可选的权重和偏置、可选的运行均值和方差等参数
Tensor batch_norm(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    // 定义 batch_norm 执行函数，输入参数包括权重、偏置、运行时均值、运行时方差、训练标志、动量、ε值、CUDNN 是否启用
    bool batch_norm(const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt,
                    const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt,
                    bool training, double momentum, double eps, bool cudnn_enabled) {
        // 获取权重，如果未提供则使用空 Tensor
        const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});
        // 获取偏置，如果未提供则使用空 Tensor
        const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
        // 获取运行时均值，如果未提供则使用空 Tensor
        const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
        // 获取运行时方差，如果未提供则使用空 Tensor
        const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
    
        // 调用底层 batch_norm 实现函数，返回第一个值
        return std::get<0>(at::_batch_norm_impl_index(input, weight, bias, running_mean, running_var,
                                                      training, momentum, eps, cudnn_enabled));
    
        // TODO: 在两周的 FC 窗口后切换到新的堆栈
    
        // 如果正在训练
        // if (training) {
        //   // 选择 batch_norm 的后端
        //   BatchNormBackend backend = _select_batch_norm_backend(input, weight, bias, running_mean, running_var, training, eps);
        //   // 如果后端是 Cudnn 或者 Miopen
        //   if (backend == BatchNormBackend::Cudnn || backend == BatchNormBackend::Miopen) {
        //     // 对输入进行连续化处理
        //     auto input_c = input;
        //     if (backend == BatchNormBackend::Cudnn) {
        //         input_c = input.contiguous(input.suggest_memory_format());
        //     } else {
        //         input_c = input.contiguous();
        //     }
        //     // 对权重、偏置、运行时均值、运行时方差进行连续化处理
        //     auto weight_c = weight.contiguous();
        //     auto bias_c = bias.contiguous();
        //     auto rmean_c = running_mean.defined() ? running_mean.contiguous() : running_mean;
        //     auto rvar_c = running_var.defined() ? running_var.contiguous() : running_var;
        //     // 调用带更新的 batch_norm 函数
        //     return std::get<0>(at::_batch_norm_with_update(input_c, weight_c, bias_c, const_cast<Tensor&>(rmean_c),
        //                                                   const_cast<Tensor&>(rvar_c), momentum, eps));
        //   } else {
        //     // 调用带更新的 batch_norm 函数
        //     return std::get<0>(at::_batch_norm_with_update(input, weight, bias, const_cast<Tensor&>(running_mean),
        //                                                   const_cast<Tensor&>(running_var), momentum, eps));
        //   }
        // } else {
        //   // 调用无更新的 batch_norm 函数
        //   return std::get<0>(at::_batch_norm_no_update(input, weight, bias, running_mean, running_var,
        //                                               momentum, eps));
        // }
    }
}

Tensor instance_norm(
    const Tensor& input, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& bias_opt /* optional */, const std::optional<Tensor>& running_mean_opt /* optional */, const std::optional<Tensor>& running_var_opt /* optional */,
    bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的权重张量中获取权重，如果未提供则使用默认张量
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  // 从可选的偏置张量中获取偏置，如果未提供则创建一个空张量
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  // 从可选的运行均值张量中获取运行均值，如果未提供则创建一个空张量
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  // 从可选的运行方差张量中获取运行方差，如果未提供则创建一个空张量
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  // 检查是否使用输入统计信息或者是否已定义运行均值和运行方差张量
  TORCH_CHECK(use_input_stats || (running_mean.defined() && running_var.defined()),
           "Expected running_mean and running_var to be defined when use_input_stats is false");

  // 获取输入张量的符号尺寸，并对其进行调整以适应批次归一化的形状
  std::vector<SymInt> shape = input.sym_sizes().vec();
  SymInt b = input.sym_size(0);
  SymInt c = input.sym_size(1);
  shape[1] = b * c;
  shape[0] = SymInt(1);

  // 如果权重张量已定义，则根据批次大小重复权重张量
  Tensor weight_ = repeat_if_defined(weight, b);
  // 如果偏置张量已定义，则根据批次大小重复偏置张量
  Tensor bias_ = repeat_if_defined(bias, b);
  // 如果运行均值张量已定义，则根据批次大小重复运行均值张量
  Tensor running_mean_ = repeat_if_defined(running_mean, b);
  // 如果运行方差张量已定义，则根据批次大小重复运行方差张量
  Tensor running_var_ = repeat_if_defined(running_var, b);

  // 将输入张量重塑为连续的张量，并根据新的形状进行视图调整
  auto input_reshaped = input.contiguous().view_symint(shape);
  // 应用批次归一化操作，使用给定的权重、偏置、运行均值和运行方差
  auto out = at::batch_norm(input_reshaped, weight_, bias_, running_mean_, running_var_,
                            use_input_stats, momentum, eps, cudnn_enabled);

  // 由于运行均值和运行方差是常量，但我们想要修改它们的数据，因此进行别名处理
  // 更新运行均值
  if (running_mean.defined()) {
    at::alias(running_mean).copy_(running_mean_.view_symint({ b, c }).mean(0, false));
  }
  // 更新运行方差
  if (running_var.defined()) {
    at::alias(running_var).copy_(running_var_.view_symint({ std::move(b), std::move(c) }).mean(0, false));
  }

  // 将输出张量的形状重新调整回与输入相同的符号尺寸
  return out.view_symint(input.sym_sizes());
}

std::tuple<Tensor, Tensor> batch_norm_update_stats_cpu(
        const Tensor& self, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, double momentum) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的运行均值张量中获取运行均值，如果未提供则创建一个空张量
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  // 从可选的运行方差张量中获取运行方差，如果未提供则创建一个空张量
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  // 检查输入张量、运行均值和运行方差是否为混合数据类型
  const bool mixed_type = is_mixed_type(self, running_mean, running_var);
  return AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "batch_norm_update_stats_cpu", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    if (mixed_type) {
      // 检查混合数据类型的数据一致性
      check_mixed_data_type(self, running_mean, running_var);
      // 应用模板函数更新批次归一化的统计信息
      return batch_norm_cpu_update_stats_template<scalar_t, opmath_t, Var>(self, running_mean, running_var, momentum, 0);
    } else {
      # 如果不满足前面的条件，执行这个分支
      # 调用模板函数 batch_norm_cpu_update_stats_template，使用 scalar_t 类型作为输入和输出类型，以及 Var 作为模板参数
      # 参数依次为 self（当前对象）、running_mean（运行时均值）、running_var（运行时方差）、momentum（动量参数）、0（额外参数）
      return batch_norm_cpu_update_stats_template<scalar_t, scalar_t, Var>(self, running_mean, running_var, momentum, 0);
    }
std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_cpu_out(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
                                                  bool train, double momentum, double eps, Tensor& out, Tensor& save_mean, Tensor& save_var) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的张量中借用权重，如果不存在则创建一个空张量
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  // 获取偏置张量，如果不存在则创建一个空张量
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  // 获取运行时均值张量，如果不存在则创建一个空张量
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  // 获取运行时方差张量，如果不存在则创建一个空张量
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  // 检查张量的后端，确保所有张量都在 CPU 后端
  checkBackend("batch_norm_cpu_out", {self, weight, bias, running_mean, running_var}, Backend::CPU);
  // 调整输出张量的大小
  at::native::resize_output(out, self.sizes());

  // 检查是否存在混合类型的数据
  const bool mixed_type = is_mixed_type(self, weight, bias, running_mean, running_var);
  // 根据张量的浮点类型和是否存在混合类型，选择不同的模板函数进行批归一化
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "batch_norm", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    if (mixed_type) {
      // 如果存在混合类型，则检查并执行相应的批归一化计算
      check_mixed_data_type(self, weight, bias, running_mean, running_var);
      if (!train) {
        // 如果不是训练模式，直接对输入进行变换
        return batch_norm_cpu_transform_input_template<scalar_t, opmath_t>(self, weight, bias, save_mean, save_var, running_mean, running_var, train, eps, out);
      } else {
        // 如果是训练模式，调整 save_mean 和 save_var 的大小，并更新统计信息后再进行变换
        at::native::resize_output(save_mean, {self.size(1)});
        at::native::resize_output(save_var, {self.size(1)});
        auto save_stats = batch_norm_cpu_update_stats_template<scalar_t, opmath_t, InvStd>(self, running_mean, running_var, momentum, eps, save_mean, save_var);
        return batch_norm_cpu_transform_input_template<scalar_t, opmath_t>(self, weight, bias, std::get<0>(save_stats), std::get<1>(save_stats), running_mean, running_var, train, eps, out);
      }
    } else {
      // 如果不存在混合类型，则根据张量的浮点类型执行批归一化计算
      if (!train) {
        // 如果不是训练模式，直接对输入进行变换
        return batch_norm_cpu_transform_input_template<scalar_t, scalar_t>(self, weight, bias, save_mean, save_var, running_mean, running_var, train, eps, out);
      } else {
        // 如果是训练模式，调整 save_mean 和 save_var 的大小，并更新统计信息后再进行变换
        at::native::resize_output(save_mean, {self.size(1)});
        at::native::resize_output(save_var, {self.size(1)});
        auto save_stats = batch_norm_cpu_update_stats_template<scalar_t, scalar_t, InvStd>(self, running_mean, running_var, momentum, eps, save_mean, save_var);
        return batch_norm_cpu_transform_input_template<scalar_t, scalar_t>(self, weight, bias, std::get<0>(save_stats), std::get<1>(save_stats), running_mean, running_var, train, eps, out);
      }
    }
  });

  // 返回更新后的输出张量及相关统计信息
  return std::tuple<Tensor& ,Tensor&, Tensor&>(out, save_mean, save_var);
}
std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_with_update_cpu(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, double momentum, double eps) {
  // 输出变量定义
  Tensor output, save_mean, save_var;
  // 调用 batch_norm_cpu 函数获取 output, save_mean, save_var
  std::tie(output, save_mean, save_var) =
      // 调用 batch_norm_cpu_out 函数，计算 batch normalization 结果
      batch_norm_cpu(input, weight_opt, bias_opt, running_mean, running_var,
                     /*train=*/true, momentum, eps);

  // 返回包含 output, save_mean, save_var 的元组
  return std::make_tuple(output, save_mean, save_var, running_mean, running_var);
}
    // 调用 batch_norm_cpu 函数对输入进行批归一化处理，更新权重、偏置、均值、方差等状态
    batch_norm_cpu(input, weight_opt, bias_opt, running_mean, running_var, /*update*/true, momentum, eps);
    // 创建一个空的 Tensor 对象 reserve，用于存储额外的中间计算结果，数据类型为 kByte
    Tensor reserve = at::empty({0}, input.options().dtype(kByte));
    // 返回一个包含四个 Tensor 对象的元组，分别为输出结果、保存的均值、保存的方差和 reserve
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, save_mean, save_var, reserve);
}

// 执行批量归一化（带更新），使用CPU计算并将结果输出到指定张量中
std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> _batch_norm_with_update_cpu_out(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, double momentum, double eps,
    Tensor& out, Tensor& save_mean, Tensor& save_var, Tensor& reserve) {
  
  // 调用 batch_norm_cpu_out 函数执行批量归一化，并更新输出、保存均值和保存方差
  std::tie(out, save_mean, save_var) =
    batch_norm_cpu_out(input, weight_opt, bias_opt, running_mean, running_var, /*update*/true, momentum, eps, out, save_mean, save_var);
  
  // 返回更新后的输出张量、保存的均值、保存的方差和预留张量的元组引用
  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(out, save_mean, save_var, reserve);
}

// 执行批量归一化（不带更新），使用CPU计算并返回结果张量的元组
std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_no_update(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    double momentum, double eps) {
  
  // 获取或创建运行时均值和方差张量
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  
  // 执行批量归一化（不更新），返回输出张量、保存的均值和保存的方差
  Tensor output, save_mean, save_var;
  std::tie(output, save_mean, save_var) =
    batch_norm_cpu(input, weight_opt, bias_opt, const_cast<Tensor&>(running_mean), const_cast<Tensor&>(running_var), /*update*/false, momentum, eps);
  
  // 创建并返回结果张量的元组，包括输出、保存的均值、保存的方差和预留张量
  Tensor reserve = at::empty({0}, input.options().dtype(kByte));
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, save_mean, save_var, reserve);
}

// 执行合法的 CPU 批量归一化，返回输出张量、保存的均值和保存的方差的元组
std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_cpu(
    const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, bool train, double momentum, double eps) {
  
  // 调用 batch_norm_cpu 函数执行批量归一化，返回输出张量、保存的均值和保存的方差
  return batch_norm_cpu(self, weight_opt, bias_opt, running_mean, running_var, train, momentum, eps);
}

// 执行合法的 CPU 批量归一化（不使用统计信息），返回输出张量、保存的均值和保存的方差的元组
std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_no_stats_cpu(
    const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    bool train, double momentum, double eps) {
  
  // 调用 batch_norm_cpu 函数执行批量归一化，传递空的运行时均值和方差，返回输出张量、保存的均值和保存的方差
  return batch_norm_cpu(self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, eps);
}

// 执行合法的 CPU 批量归一化（不训练模式），返回输出张量、保存的均值和保存的方差的元组
std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_no_training(
    const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    const Tensor& running_mean, const Tensor& running_var, double momentum, double eps) {
  
  // 调用 at::_native_batch_norm_legit 函数执行批量归一化，传递不训练模式的参数，返回输出张量、保存的均值和保存的方差
  return at::_native_batch_norm_legit(self, weight_opt, bias_opt, const_cast<Tensor&>(running_mean), const_cast<Tensor&>(running_var), /*train=*/false, momentum, eps);
}
// 使用给定的输入和参数计算批归一化的反向传播结果，并返回梯度和中间计算结果的元组
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cpu(
    const Tensor& grad_out,                   // 输入梯度
    const Tensor& self,                       // 输入张量
    const std::optional<Tensor>& weight_opt,  // 可选的权重张量
    const std::optional<Tensor>& running_mean_opt,  // 可选的运行均值张量
    const std::optional<Tensor>& running_var_opt,   // 可选的运行方差张量
    const std::optional<Tensor>& save_mean_opt,     // 可选的保存均值张量
    const std::optional<Tensor>& save_invstd_opt,  // 可选的保存标准差的倒数张量
    bool train,                               // 是否处于训练模式
    double eps,                               // epsilon 值，用于数值稳定性
    std::array<bool,3> grad_input_mask) {      // 梯度输入的掩码数组

  // 从可选的权重张量中获取权重，并确保类型匹配
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 从可选的运行均值张量中获取运行均值，如果未提供则返回一个空张量
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});

  // 从可选的运行方差张量中获取运行方差，如果未提供则返回一个空张量
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  // 从可选的保存均值张量中获取保存均值，如果未提供则返回一个空张量
  const Tensor& save_mean = c10::value_or_else(save_mean_opt, [] {return Tensor();});

  // 从可选的保存标准差的倒数张量中获取保存标准差的倒数，如果未提供则返回一个空张量
  const Tensor& save_invstd = c10::value_or_else(save_invstd_opt, [] {return Tensor();});

  // 检查输入张量和各种参数是否包含混合数据类型，并返回布尔值
  const bool mixed_type = is_mixed_type(self, weight, running_mean, running_var, save_mean, save_invstd);

  // 根据输入张量的数据类型进行分发，调用对应的批归一化反向传播模板函数
  return AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "batch_norm_backward_cpu", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    
    // 如果输入数据类型混合，则进行类型检查
    if (mixed_type) {
      check_mixed_data_type(self, weight, running_mean, running_var, save_mean, save_invstd);
    }
    
    // 调用具体的批归一化反向传播模板函数，返回计算结果的元组
    return batch_norm_backward_cpu_template<scalar_t, opmath_t>(
        grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, eps, grad_input_mask);
  });
}


这段代码是用于批归一化的反向传播计算。它接受输入梯度、输入张量及其它参数，并根据这些参数计算反向传播结果。
    // 如果不是训练阶段，则执行批量归一化反向传播的 CPU 模板函数，返回计算得到的梯度输入
    } else {
      return batch_norm_backward_cpu_template<scalar_t, scalar_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, eps, grad_input_mask);
    }
} // 结束 TORCH_IMPL_FUNC(renorm_out) 函数定义

TORCH_IMPL_FUNC(renorm_out)(const Tensor& self, const Scalar& p, int64_t dim,
                            const Scalar& maxnorm, const Tensor& out) {
  auto self_sizes = self.sizes();
  // 确保维度 dim 在合法范围内
  dim = c10::maybe_wrap_dim(dim, self_sizes.size());

  // 创建用于指示要减少的维度的向量
  DimVector reduce_dims(self_sizes.size());
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  reduce_dims.erase(reduce_dims.begin() + dim);

  // 对于 CUDA 半精度数据，以浮点精度计算范数，然后转换为半精度
  auto dtype = self.scalar_type();
  auto acc_type = at::toAccumulateType(dtype, /*is_cuda=*/true);
  Tensor norm;
  if (acc_type != dtype) {
    // 计算向量的范数，使用指定的维度和精度，保持维度为真，使用指定的累积类型
    norm = at::linalg_vector_norm(self, p.toDouble(), reduce_dims,
                                  /*keepdim=*/true, /*dtype=*/acc_type);
  } else {
    norm = at::linalg_vector_norm(self, p.toDouble(), reduce_dims,
                                  /*keepdim=*/true);
  }

  // 创建一个与 norm 具有相同大小的空张量作为缩放因子
  auto factor = (acc_type == c10::toRealValueType(dtype)) ?
      norm : at::empty(norm.sizes(), self.options());

  // 配置张量迭代器，用于操作 factor 和 norm 张量
  auto iter = TensorIteratorConfig()
      .add_output(factor)
      .add_input(norm)
      .set_check_mem_overlap(false)
      .cast_common_dtype_to_outputs(true)
      .build();

  // 调用具体的 renorm_scale_factor_stub 函数处理迭代器和最大范数值
  renorm_scale_factor_stub(iter.device_type(), iter, maxnorm.toDouble());

  // 将缩放因子应用到 self 和 out 张量的乘法操作
  at::mul_outf(self, factor, const_cast<Tensor&>(out));
}

} // 结束 at::native 命名空间
```