# `.\pytorch\aten\src\ATen\native\group_norm.cpp`

```
// 定义宏以仅包含方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含相关头文件
#include <ATen/native/group_norm.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <c10/util/accumulate.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含以下头文件；否则包含另一组头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/group_norm_native.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_group_norm.h>
#include <ATen/ops/native_group_norm_backward_native.h>
#include <ATen/ops/native_group_norm_native.h>
#endif

#include <array>
#include <functional>
#include <tuple>
#include <vector>

namespace at::native {

// 检查 group norm 的输入参数有效性
template <typename T>
void check_group_norm_inputs(
    const Tensor& input,        // 输入张量
    const Tensor& weight,       // 权重张量
    const Tensor& bias,         // 偏置张量
    T C,                       // 通道数
    int64_t num_groups) {       // 组数
  TORCH_CHECK(
      num_groups > 0,
      "Expected num groups to be greater than 0, got ", num_groups);  // 检查组数是否大于0
  TORCH_CHECK(
      C % num_groups == 0,
      "Expected number of channels in input to be divisible by ",
      "num_groups, but got input of shape ",
      input.sizes(),
      " and "
      "num_groups=",
      num_groups);  // 检查输入通道数是否能被组数整除
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && at::symint::numel<T>(weight) == C),
      "Expected weight to be a vector of size equal to the number of ",
      "channels in input, but got weight of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());  // 检查权重张量是否定义并且符合输入通道数要求
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && at::symint::numel<T>(bias) == C),
      "Expected bias to be a vector of size equal to the number of ",
      "channels in input, but got bias of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());  // 检查偏置张量是否定义并且符合输入通道数要求
}

// 执行 group norm 算法，返回归一化后的张量及中间变量
std::tuple<Tensor, Tensor, Tensor> native_group_norm(
    const Tensor& X,                   // 输入张量
    const std::optional<Tensor>& gamma_opt /* optional */,  // 可选参数 gamma
    const std::optional<Tensor>& beta_opt /* optional */,   // 可选参数 beta
    int64_t N,                          // 样本数
    int64_t C,                          // 通道数
    int64_t HxW,                        // 高度乘宽度
    int64_t group,                      // 组数
    double eps) {                       // epsilon 参数
  // 查看注释以了解可选张量的处理方式
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma = *gamma_maybe_owned;
  const Tensor& beta = c10::value_or_else(beta_opt, [] { return Tensor(); });

  // 再次检查以支持扩展权重调用 native_group_norm，同时保存前向传播的均值和方差
  check_group_norm_inputs(X, gamma, beta, C, group);

  // 根据设备类型推荐内存格式
  auto memory_format = X.device().is_cpu() ?
      X.suggest_memory_format() : at::MemoryFormat::Contiguous;

  // 检查输入张量是否是连续的
  TORCH_CHECK(X.is_contiguous(memory_format));

  // 检查是否存在混合数据类型
  bool mixed_type = is_mixed_type(X, gamma, beta);
  if (mixed_type) {
    // 调用函数检查输入数据的混合数据类型是否符合要求
    check_mixed_data_type(X, gamma, beta);
  }

  // 创建一个与输入张量 X 具有相同形状和选项的空张量 Y
  Tensor Y = at::native::empty_like(
      X,
      c10::nullopt /* dtype */,    // 指定数据类型为空，表示保持与 X 相同的数据类型
      c10::nullopt /* layout */,   // 指定布局为空，表示保持与 X 相同的布局
      c10::nullopt /* device */,   // 指定设备为空，表示保持与 X 相同的设备
      c10::nullopt /* pin_memory */,  // 指定 pin_memory 为空，表示不指定内存固定
      memory_format);              // 指定内存格式

  // 根据混合类型计算参数的数据类型
  const auto dtype = param_scalar_type(X, mixed_type);

  // 创建一个与输入张量 X 形状相同，数据类型为 dtype 的空张量 mean
  Tensor mean = at::empty({N, group}, X.options().dtype(dtype));

  // 创建一个与输入张量 X 形状相同，数据类型为 dtype 的空张量 rstd
  Tensor rstd = at::empty({N, group}, X.options().dtype(dtype));

  // 调用 GroupNormKernel 进行分组归一化计算
  GroupNormKernel(
      X.device().type(),    // 输入张量 X 的设备类型
      X,                    // 输入张量 X
      gamma,                // 缩放参数 gamma
      beta,                 // 偏置参数 beta
      N,                    // 样本数
      C,                    // 通道数
      HxW,                  // 特征图大小
      group,                // 分组数
      eps,                  // 归一化小常数
      Y,                    // 输出张量 Y
      mean,                 // 输出张量 mean
      rstd);                // 输出张量 rstd

  // 返回包含 Y, mean, rstd 的元组作为结果
  return std::make_tuple(Y, mean, rstd);
// 返回类型为包含三个张量的元组：梯度 dY、输入张量 X、均值 mean、倒数标准差 rstd、可选的 gamma 张量、N、C、HxW、group 和梯度输入掩码的参数
std::tuple<Tensor, Tensor, Tensor> native_group_norm_backward(
    const Tensor& dY,                           // 输入张量的梯度
    const Tensor& X,                            // 输入张量
    const Tensor& mean,                         // 均值张量
    const Tensor& rstd,                         // 倒数标准差张量
    const std::optional<Tensor>& gamma_opt,     // 可选的 gamma 张量
    int64_t N,                                  // 批次大小
    int64_t C,                                  // 通道数
    int64_t HxW,                                // 高度乘以宽度
    int64_t group,                              // 分组数
    std::array<bool, 3> grad_input_mask) {      // 梯度输入掩码

  // 如果 gamma_opt 存在，则将其解引用为 gamma 张量，否则抛出错误
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma = *gamma_maybe_owned;

  // 检查输入张量 X 和梯度张量 dY 的标量类型是否相同，否则抛出错误
  TORCH_CHECK(
      X.scalar_type() == dY.scalar_type(),
      "Expected scalar types of X and dY are same.");

  // 检查是否存在混合类型数据，如果是则检查数据类型
  bool mixed_type = is_mixed_type(X, mean, rstd);
  if (mixed_type) {
    check_mixed_data_type(X, mean, rstd);
  }

  // 根据设备类型选择内存格式，如果是 CPU 设备，则推荐内存格式，否则选择连续内存格式
  auto memory_format = X.device().is_cpu() ?
      X.suggest_memory_format() : at::MemoryFormat::Contiguous;

  // 初始化输出张量 dX、dgamma 和 dbeta
  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;

  // 根据梯度输入掩码的第一个位，如果为 true，则创建与 X 相同属性的空张量 dX
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(
        X,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        memory_format);
  }

  // 根据梯度输入掩码的第二个位，如果为 true，则创建与 gamma 相同属性的空张量 dgamma
  if (grad_input_mask[1]) {
    dgamma = at::native::empty_like(
        gamma,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }

  // 根据梯度输入掩码的第三个位，如果为 true，则创建与 gamma 相同属性的空张量 dbeta
  if (grad_input_mask[2]) {
    dbeta = at::native::empty_like(
        gamma,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }

  // 调用 GroupNormBackwardKernel 函数执行 Group Norm 的反向传播操作
  GroupNormBackwardKernel(
      X.device().type(),    // 设备类型
      dY,                   // 梯度 dY
      X,                    // 输入张量 X
      mean,                 // 均值张量
      rstd,                 // 倒数标准差张量
      gamma,                // gamma 张量
      N,                    // 批次大小
      C,                    // 通道数
      HxW,                  // 高度乘以宽度
      group,                // 分组数
      dX,                   // 输出梯度 dX
      dgamma,               // 输出梯度 dgamma
      dbeta);               // 输出梯度 dbeta

  // 返回包含 dX、dgamma 和 dbeta 的元组
  return std::make_tuple(dX, dgamma, dbeta);
}
    // 使用 weight_opt 来创建一个 MaybeOwned<Tensor> 对象，此处涉及到对可选张量的处理
    c10::MaybeOwned<Tensor> weight_maybe_owned =
        at::borrow_from_optional_tensor(weight_opt);
    // 将 MaybeOwned<Tensor> 解引用为常量 Tensor 引用，确保 weight 变量不为空
    const Tensor& weight = *weight_maybe_owned;
    // 使用 c10::value_or_else 函数获取 bias_opt 的值，如果为空则返回一个空的 Tensor 对象
    const Tensor& bias = c10::value_or_else(bias_opt, [] { return Tensor(); });
    
    // 获取输入张量的尺寸信息，N 为 batch 大小，C 为通道数
    const auto N = input.sym_size(0);
    const auto C = input.sym_size(1);
    // 检查 GroupNorm 的输入，包括输入张量、权重、偏置、通道数和分组数
    check_group_norm_inputs(input, weight, bias, C, num_groups);
    
    // 获取输入张量的形状信息，input_shape 是一个向量，包含了张量的各维度大小
    const auto input_shape = input.sym_sizes();
    // 计算张量在除去 batch 和 channel 后维度的乘积，即图像的高度乘以宽度
    const auto HxW =
        c10::multiply_integers(input_shape.slice(2));
    
    // 创建一个空的 Tensor 对象 kEmpty
    const Tensor kEmpty;
    // 根据输入张量的建议内存格式获取内存格式信息
    auto memory_format = input.suggest_memory_format();
    // 根据设备类型判断是否需要对输入张量进行内存连续性处理，得到处理后的张量 X
    const auto& X = input.device().is_cpu() || input.device().is_xpu() ?
        input.contiguous(memory_format) : input.contiguous();
    // 如果权重定义了，则获取其内存连续的副本；否则使用 kEmpty
    const auto& gamma = weight.defined() ? weight.contiguous() : kEmpty;
    // 如果偏置定义了，则获取其内存连续的副本；否则使用 kEmpty
    const auto& beta = bias.defined() ? bias.contiguous() : kEmpty;
    // 检查 gamma 张量是否未定义或其元素数量是否等于 C
    TORCH_CHECK(!gamma.defined() || gamma.sym_numel() == C);
    // 检查 beta 张量是否未定义或其元素数量是否等于 C
    TORCH_CHECK(!beta.defined() || beta.sym_numel() == C);
    
    // 调用 native_group_norm_symint 函数进行 Group Normalization 操作，并返回其第一个元素
    return std::get<0>(
        at::native_group_norm_symint(X, gamma, beta, N, C, HxW, num_groups, eps));
}

DEFINE_DISPATCH(GroupNormKernel);
DEFINE_DISPATCH(GroupNormBackwardKernel);

// 从 pytorch/xla 仓库移植过来的函数，实现组归一化操作
std::tuple<at::Tensor, at::Tensor, at::Tensor> math_group_norm(
    const Tensor& input,                                 // 输入张量
    const std::optional<Tensor>& weight_opt,              // 可选的权重张量
    const std::optional<Tensor>& bias_opt,                // 可选的偏置张量
    int64_t N,                                            // 批次大小
    int64_t C,                                            // 通道数
    int64_t HxW,                                          // 高度乘宽度
    int64_t group,                                        // 组数
    double eps) {                                         // epsilon 参数，用于数值稳定性

  // 从可选权重张量中获取权重张量
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 从可选偏置张量中获取偏置张量，如果未提供则创建一个空张量
  const Tensor& bias = c10::value_or_else(bias_opt, [] { return Tensor(); });

  // 获取输入张量的形状
  auto input_shape = input.sizes();

  // 将输入张量重塑为指定形状
  at::Tensor input_reshaped = input.view({1, N * group, N ? -1 : 1});

  // 执行批归一化操作，返回输出张量、均值和标准差
  auto outputs = at::native_batch_norm(
      input_reshaped,
      /*weight=*/{},         // 权重不在这里使用
      /*bias=*/{},           // 偏置不在这里使用
      /*running_mean=*/{},   // 运行时均值不在这里使用
      /*running_var=*/{},    // 运行时方差不在这里使用
      /*training=*/true,     // 训练模式
      /*momentum=*/0,        // 动量设为 0
      eps);                  // epsilon 参数

  // 获取输出张量
  at::Tensor out = std::get<0>(outputs);

  // 将输出张量重塑为原始形状
  out = out.view(input_shape);

  // 创建一个形状匹配输入张量的仿射参数形状向量
  std::vector<int64_t> affine_param_shape(input.dim(), 1);
  affine_param_shape[1] = C;

  // 根据权重和偏置进行仿射变换
  if (weight.defined() && bias.defined()) {
    out = bias.view(affine_param_shape)
              .addcmul(out, weight.view(affine_param_shape), 1);
  } else if (weight.defined()) {
    out = out.mul(weight.view(affine_param_shape));
  } else if (bias.defined()) {
    out = out.add(bias.view(affine_param_shape));
  }

  // 将均值和标准差转换为与输入张量相同的数据类型
  at::Tensor mean = std::get<1>(outputs).to(c10::TensorOptions().dtype(input.scalar_type())).view({N, group});
  at::Tensor rstd = std::get<2>(outputs).to(c10::TensorOptions().dtype(input.scalar_type())).view({N, group});

  // 返回输出张量、均值和标准差的元组
  return std::make_tuple(out, mean, rstd);
}
} // namespace at::native
```