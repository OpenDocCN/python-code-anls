# `.\pytorch\aten\src\ATen\native\mkldnn\Linear.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/mkldnn_linear_backward_input.h>
#include <ATen/ops/mkldnn_linear_backward_input_native.h>
#include <ATen/ops/mkldnn_linear_backward_native.h>
#include <ATen/ops/mkldnn_linear_backward_weights.h>
#include <ATen/ops/mkldnn_linear_backward_weights_native.h>
#include <ATen/ops/mkldnn_linear_native.h>
#endif

#if !AT_MKLDNN_ENABLED()

// 如果未启用 MKLDNN 支持，则定义相关函数，抛出错误信息
namespace at {
namespace native {

// 定义一个不支持的版本的 mkldnn_linear 函数
Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight, const std::optional<Tensor>& bias_opt) {
  TORCH_CHECK(false, "mkldnn_linear: ATen not compiled with MKLDNN support");
}

// 定义一个不支持的版本的 mkldnn_linear_backward_input 函数
Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_linear_backward_input: ATen not compiled with MKLDNN support");
}

// 定义一个不支持的版本的 mkldnn_linear_backward_weights 函数
std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_linear_backward_weights: ATen not compiled with MKLDNN support");
}

// 定义一个不支持的版本的 mkldnn_linear_backward 函数
std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output_t,
    const Tensor& weight, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "mkldnn_linear_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

// 如果启用了 MKLDNN 支持，则包含相关头文件
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

// 在 at::native 命名空间下定义 MKLDNN 的线性函数

namespace at {
namespace native {

// 定义 MKLDNN 支持的 mkldnn_linear 函数
Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight_t, const std::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的 bias_opt 引用中获取 Tensor 对象
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 检查输入张量的维度和布局是否符合要求
  const int64_t dim = self.dim();
  TORCH_CHECK(
      self.dim() != 0,
      "mkldnn_linear: input needs to has dim at least 1, input dim ",
      self.dim());
  TORCH_CHECK(self.is_mkldnn(),
      "mkldnn_linear: input needs to be mkldnn layout");

  // 如果输入张量的标量类型是 BFloat16，需要检查 CPU 是否支持相应的指令集
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_linear: bf16 path needs the cpu support avx_ne_convert or avx512bw, avx512vl and avx512dq");
  } else if (self.scalar_type() == ScalarType::Half) {
    // 如果输入张量的标量类型是 Half，需要检查 CPU 是否支持相应的指令集
    // （此处可能还有未完整的代码，但由于注释要求不改变缩进和代码结构，保持原样）


注释：以上为 C++ 代码的注释，按照要求为每行代码添加了相应的解释和说明。
  // 检查是否支持 MKL-DNN 的 FP16 设备特性，否则抛出错误信息
  TORCH_CHECK(mkldnn_fp16_device_check(),
      "mkldnn_linear: fp16 path needs the cpu support avx_ne_convert or avx512_fp16");
}

// 如果输入张量的维度不是2，将其重塑，如果重塑操作会导致内存复制则执行
auto self_reshaped =
    dim == 2 ? self : self.reshape({-1, self.size(self.dim() - 1)});

// 将重塑后的张量转换为 MKL-DNN 张量
const ideep::tensor x = itensor_from_mkldnn(self_reshaped);

// 判断权重张量是否为 MKL-DNN 张量或者连续内存的普通张量，选择其一作为 Tensor 对象
const Tensor weight = (weight_t.is_mkldnn() || weight_t.is_contiguous()) ? weight_t : weight_t.contiguous();
// 将选择的权重张量转换为 MKL-DNN 张量
const ideep::tensor w = itensor_from_tensor(weight);

// 定义输出张量 y
ideep::tensor y;
// 如果有偏置，则将偏置张量转换为 MKL-DNN 张量并执行内积计算
if (bias.defined()) {
  const ideep::tensor b = itensor_from_tensor(bias);
  ideep::inner_product_forward::compute(x, w, b, y);
} else {
  // 否则仅执行内积计算
  ideep::inner_product_forward::compute(x, w, y);
}

// 获取输入张量的尺寸
auto input_size = self.sizes();
// 创建输出尺寸向量，从输入尺寸中复制除最后一维的所有维度，并添加权重张量的第一维
std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
output_size.push_back(weight.size(0));

// 如果输入张量的维度不是2，则返回重塑后的 MKL-DNN 张量，否则直接返回 MKL-DNN 张量
if (self.dim() != 2) {
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt()).reshape(output_size);
}
return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                               self.options().device_opt());
}

// 计算 MKL-DNN 线性层反向传播的输入
Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight_t){
  // 检查 grad_output 是否为 MKL-DNN 布局
  TORCH_CHECK(grad_output.is_mkldnn(),
      "mkldnn_linear_backward: grad_output needs to be mkldnn layout");
  // 检查 weight_t 是否为 CPU 上的浮点密集张量
  TORCH_CHECK(weight_t.device().is_cpu() && weight_t.scalar_type() == kFloat,
      "mkldnn_linear_backward: weight_t needs to be a dense tensor");
  // 重新整形 grad_output 为二维张量
  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;

  // 从 MKL-DNN 张量获取 grad_output 的 ideep::tensor 引用
  ideep::tensor& grady = itensor_from_mkldnn(grad_output_reshaped);
  // 如果 weight_t 连续，则使用它；否则创建连续版本
  const Tensor weight = weight_t.is_contiguous() ? weight_t : weight_t.contiguous();
  // 将 weight 转换为 MKL-DNN 视图的 ideep::tensor
  const ideep::tensor w = itensor_view_from_dense(weight);

  // 创建输入重整形后的尺寸向量
  std::vector<int64_t> input_reshaped_size;
  input_reshaped_size.push_back(grad_output_reshaped.size(0));
  input_reshaped_size.push_back(weight.size(1));

  // 创建 gradx 以用于内积反向数据计算
  ideep::tensor gradx;
  ideep::inner_product_backward_data::compute(
    grady, w, {input_reshaped_size.begin(), input_reshaped_size.end()}, gradx);

  // 如果输入尺寸超过二维，则返回新的 MKL-DNN 张量，并重整形为 input_size
  if (input_size.size() > 2) {
    return new_with_itensor_mkldnn(std::move(gradx), optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                   grad_output.options().device_opt()).reshape(input_size);
  }
  // 否则直接返回新的 MKL-DNN 张量
  return new_with_itensor_mkldnn(std::move(gradx), optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

// 计算 MKL-DNN 线性层反向传播的权重和偏置
std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  // 检查 grad_output 和 input 是否为 MKL-DNN 布局
  TORCH_CHECK(grad_output.is_mkldnn() && input.is_mkldnn(),
      "mkldnn_linear_backward: grad_output and input needs to be mkldnn layout");
  // 检查 weight 是否为 CPU 上的浮点密集张量
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == kFloat,
      "mkldnn_linear_backward: weight needs to be a dense tensor");

  // 重新整形 grad_output 和 input 为二维张量
  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() > 2 ? input.reshape({-1, input.size(input.dim() - 1)}) : input;

  // 获取 grad_output 和 input 的 ideep::tensor 引用
  ideep::tensor& grady = itensor_from_mkldnn(grad_output_reshaped);
  ideep::tensor& x = itensor_from_mkldnn(input_reshaped);
  ideep::tensor gradw, gradb;
  
  // 根据是否定义偏置，计算内积反向权重和偏置
  if (bias_defined) {
    ideep::inner_product_backward_weights::compute(x, grady, gradw, gradb);
  } else {
    ideep::inner_product_backward_weights::compute(x, grady, gradw);
  }

  // 返回转换为密集张量后的 gradw 和 gradb 的 tuple
  return std::tuple<Tensor, Tensor>{
    mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw),
                    optTypeMetaToScalarType(weight.options().dtype_opt()),
                    weight.options().device_opt())),
    mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradb),
                    optTypeMetaToScalarType(weight.options().dtype_opt()),
                    weight.options().device_opt()))};
}
// 定义一个函数 mkldnn_linear_backward，用于计算 MKLDNN 线性层的反向传播
std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    // 输入张量
    const Tensor& input, 
    // 梯度输出张量
    const Tensor& grad_output,
    // 权重张量
    const Tensor& weight,
    // 输出掩码数组，确定输出的哪些部分需要计算梯度
    std::array<bool,3> output_mask) {
  
  // 定义梯度输入、梯度权重、梯度偏置张量
  Tensor grad_input, grad_weight, grad_bias;
  
  // 如果输出掩码中第一个位置为真，则计算梯度输入
  if (output_mask[0]) {
    grad_input = at::mkldnn_linear_backward_input(input.sizes(), grad_output, weight);
  }
  
  // 如果输出掩码中第二个或第三个位置为真，则计算梯度权重和梯度偏置
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  
  // 返回梯度输入、梯度权重、梯度偏置的元组
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

// 定义静态函数 mkldnn_linear_pointwise，用于执行 MKLDNN 线性层的逐点运算
static Tensor mkldnn_linear_pointwise(
    // 输入张量
    const Tensor& input_t,
    // 权重张量
    const Tensor& weight_t,
    // 可选的偏置张量
    const std::optional<Tensor>& bias_opt,
    // 属性名称
    c10::string_view attr,
    // 标量列表
    torch::List<std::optional<at::Scalar>> scalars,
    // 可选的算法名称
    std::optional<c10::string_view> algorithm) {
  
  // 对输入张量进行连续化
  auto input = input_t.contiguous();
  // 获取输入张量的大小
  auto input_size = input.sizes();

  // 如果输入是连续张量，则确保它具有默认的连续步幅以获得更好的性能
  input = may_convert_to_default_contiguous_strides(input);

  // 获取输入张量的维度
  const int64_t dim = input.dim();
  // 根据输入张量的维度重新形状化输入张量
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});

  // 创建输出张量的大小，去除最后一个维度，然后添加权重张量的第一个维度
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight_t.size(0));
  // 根据给定选项创建空的输出张量
  auto output = at::empty(output_size, input.options());
  
  // 如果输出张量的符号元素数量为 0，则直接返回空输出张量
  if (output.sym_numel() == 0) {
    return output;
  }
  
  // 如果输入张量不是二维的，则重新形状化输出张量
  if (dim != 2) {
    std::vector<int64_t> output_size_reshaped = {input_reshaped.size(0),
                                                 weight_t.size(0)};
    output = output.reshape(output_size_reshaped);
  }

  // 设置自动求导的调度键集排除
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  // 从普通张量创建 MKLDNN 张量
  ideep::tensor mkldnn_output = itensor_from_tensor(output);

  // 从可选的偏置张量中借用 Tensor
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  // 获取偏置张量的引用
  const Tensor& bias = *bias_maybe_owned;

  // 将稠密输入张量视图转换为 MKLDNN 张量视图
  const ideep::tensor mkldnn_input = itensor_view_from_dense(input_reshaped);

  // 定义一个可选的 MKLDNN 偏置张量
  std::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  // 如果偏置张量已定义，则从普通张量创建 MKLDNN 偏置张量
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }
  // 从普通张量创建 MKLDNN 权重张量
  const ideep::tensor w = itensor_from_tensor(weight_t);

  // 创建一个空的属性对象
  ideep::attr_t op_attr = ideep::attr_t();
  // 如果属性不为 "none"，则设置操作的属性
  if (attr != "none") {
    auto it = fusion_unary_attr_map().find(attr);
    TORCH_CHECK(
        it != fusion_unary_attr_map().end(), "Fusion behavior undefined.");
    op_attr = it->second(scalars, algorithm);
  }

  // 如果 MKLDNN 偏置张量有值，则执行内积前向计算，否则执行无偏置的内积前向计算
  if (mkldnn_bias.has_value()) {
    ideep::inner_product_forward::compute</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input,
        w,
        mkldnn_bias.value(),
        mkldnn_output,
        op_attr);
  } else {
    ideep::inner_product_forward::compute</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input,
        w,
        mkldnn_output,
        op_attr);
  }

  // 如果输入张量不是二维的，则重新形状化输出张量为最初的输出大小
  if (dim != 2) {
    output = output.reshape(output_size);
  }

  // 返回计算结果输出张量
  return output;
}
// 创建一个静态函数，用于执行基于MKL-DNN的线性点对点二进制运算
static Tensor mkldnn_linear_pointwise_binary(
    const Tensor& input_t,               // 输入张量
    const Tensor& other_t,               // 第二个输入张量
    const Tensor& weight_t,              // 权重张量
    const std::optional<Tensor>& bias_opt, // 可选的偏置张量
    c10::string_view attr) {             // 算法属性字符串视图

  // 从可选的偏置张量中获取有效的引用
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 检查输入张量的类型（设备、布局、数据类型）是否一致，设备为CPU，数据类型为float或bfloat16
  check_mkldnn_binary_fusion_inputs(input_t, other_t, weight_t, bias);

  // 对输入张量进行连续化处理
  auto input = input_t.contiguous();

  // 查找给定属性对应的二进制融合算法映射
  auto it_binary = fusion_binary_alg_map().find(attr);
  TORCH_CHECK(
      it_binary != fusion_binary_alg_map().end(), "Fusion behavior undefined.");

  // 获取输入张量的尺寸信息
  auto input_size = input.sizes();

  const int64_t dim = input.dim();
  // 如果维度为2，则不做改变，否则将输入张量重塑为二维
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});

  // 构建输出张量的尺寸，移除最后一个维度，并添加权重张量的维度
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight_t.size(0));
  // 创建与输入张量相同类型的空输出张量
  auto output = at::empty(output_size, input.options());
  // 如果输出张量的元素数量为0，则直接返回空输出张量
  if (output.sym_numel() == 0) {
    return output;
  }

  // 对第二个输入张量进行连续化处理
  auto other_reshaped = other_t.contiguous();

  // 如果维度不为2，则进一步重塑输出张量和第二个输入张量
  if (dim != 2) {
    std::vector<int64_t> output_size_reshaped = {
        input_reshaped.size(0), weight_t.size(0)};
    output = output.reshape(output_size_reshaped);
    other_reshaped = other_reshaped.reshape(output_size_reshaped);
  }

  // 检查重塑后的输出张量和第二个输入张量的尺寸是否相同
  TORCH_CHECK(
      output.sizes() == other_reshaped.sizes(),
      "linear_binary_run expects the size of output and other tensor to be the same");

  // 进入自动梯度调度键集的排除调度键保护块
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);

  // 将PyTorch张量转换为MKL-DNN张量
  ideep::tensor mkldnn_output = itensor_from_tensor(output);
  const ideep::tensor mkldnn_other = itensor_from_tensor(other_reshaped);
  const ideep::tensor mkldnn_input = itensor_view_from_dense(input_reshaped);

  std::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  // 如果存在偏置张量，则将其转换为MKL-DNN张量
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }
  const ideep::tensor w = itensor_from_tensor(weight_t);

  // 获取第二个输入张量的描述符和操作属性
  auto other_desc = mkldnn_other.get_desc();
  auto op_attr = ideep::attr_t::fuse_binary(it_binary->second, other_desc);

  // 如果存在偏置张量，则执行带偏置的二进制内积前向计算；否则执行不带偏置的计算
  if (mkldnn_bias.has_value()) {
    ideep::inner_product_forward::compute_binary</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input,
        mkldnn_other,
        w,
        mkldnn_bias.value(),
        mkldnn_output,
        op_attr);
  } else {
    ideep::inner_product_forward::compute_binary</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input, mkldnn_other, w, mkldnn_output, op_attr);
  }

  // 如果维度不为2，则恢复输出张量的形状
  if (dim != 2) {
    output = output.reshape(output_size);
  }

  // 返回计算结果的输出张量
  return output;
}

#if AT_MKL_ENABLED()
#include <mkl.h>

static Tensor mkl_linear(
    const Tensor& self,
    const Tensor& mkl_weight_t,
    const Tensor& origin_weight_t,
    const std::optional<Tensor>& bias_opt,
    // 定义一个函数，实现一个线性层的计算，返回输出张量
    c10::MaybeOwned<Tensor> mkl_linear(
        // 输入张量自动管理，可能包含可选的偏置
        const Tensor& self,
        // 原始权重张量
        const Tensor& origin_weight_t,
        // 可选的偏置张量
        c10::optional<Tensor> bias_opt,
        // MKL-DNN支持的预打包批次大小
        const int64_t prepack_batch_size) {
      // 如果有偏置，则将其作为常规引用进行操作
      c10::MaybeOwned<Tensor> bias_maybe_owned =
          at::borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;
      // 检查输入张量与原始权重张量的数据类型是否相同
      TORCH_CHECK(
          self.options().type_equal(origin_weight_t.options()),
          "Input type (",
          self.toString(),
          ") and weight type (",
          origin_weight_t.toString(),
          ") should be the same");
      // 如果有偏置，检查输入张量与偏置张量的数据类型是否相同
      TORCH_CHECK(
          !bias.defined() || (self.options().type_equal(bias.options())),
          "Input type (",
          self.toString(),
          ") and bias type (",
          bias.toString(),
          ") should be the same");
      // 检查MKL线性层的权重数据类型是否为float
      TORCH_CHECK(
          mkl_weight_t.scalar_type() == origin_weight_t.scalar_type() &&
              origin_weight_t.scalar_type() == kFloat,
          "mkl_linear: weight dtype should be float");
    
      // 排除自动求导分发键的上下文
      c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
      // 获取输入张量的尺寸
      auto input_size = self.sizes();
      // 计算输出张量的尺寸，移除最后一个维度并附加原始权重张量的第一维度
      std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
      output_size.push_back(origin_weight_t.size(0));
      // 创建一个与输出尺寸相同的空张量
      auto output = at::empty(output_size, self.options());
    
      // 如果输入张量的对称元素数量为0，填充输出张量为0并返回
      if (self.sym_numel() == 0) {
        // 避免在self.size(self.dim() - 1)==0时调用self.numel() / 0
        return output.fill_(0);
      }
      // 如果输出张量的对称元素数量为0，直接返回空的输出张量
      if (output.sym_numel() == 0) {
        return output;
      }
    
      // 计算M值，为输入张量的总元素数除以最后一个维度的大小
      int64_t M = self.numel() / self.size(self.dim() - 1);
      // 如果M等于预打包批次大小并且原始权重张量是MKL-DNN类型
      if (M == prepack_batch_size && mkl_weight_t.is_mkldnn()) {
        // 如果输入张量是连续的，则使用它本身；否则进行连续化
        auto self_ = self.is_contiguous() ? self : self.contiguous();
        auto K = origin_weight_t.size(1);
        auto N = origin_weight_t.size(0);
        // 从MKL-DNN张量获取权重数据
        const ideep::tensor& w = itensor_from_mkldnn(mkl_weight_t);
        auto in_ptr = self_.data_ptr<float>();
        auto weight_ptr = (float*)(w.get_data_handle());
        auto out_ptr = output.data_ptr<float>();
        // 如果存在偏置，则进行并行计算
        if (bias.defined()) {
          auto bias_ = bias.is_contiguous() ? bias : bias.contiguous();
          auto bias_ptr = bias_.data_ptr<float>();
          at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
            // 复制偏置数据到输出张量的每一行
            for (const auto d : c10::irange(begin, end)) {
              memcpy(out_ptr + d * N, bias_ptr, sizeof(float) * N);
            }
          });
        }
        // 使用cblas库计算SGEMM矩阵乘法操作
        cblas_sgemm_compute(
            CblasRowMajor,
            CblasNoTrans,
            CblasPacked,
            M,
            N,
            K,
            in_ptr,
            K,
            weight_ptr,
            K,
            bias.defined() ? 1.f : 0.f,
            out_ptr,
            N);
      } else {
        // 否则调用ATen库的linear_out函数进行线性层计算
        output = at::linear_out(output, self, origin_weight_t, bias_opt);
      }
      // 返回最终的输出张量
      return output;
    }
} // 结束 namespace native

TORCH_LIBRARY_IMPL(mkl, CPU, m) {
  // 实现 mkl 库在 CPU 上的函数注册，注册线性操作函数 mkl_linear
  m.impl(TORCH_SELECTIVE_NAME("mkl::_mkl_linear"), TORCH_FN(mkl_linear));
}

TORCH_LIBRARY_IMPL(mkl, MkldnnCPU, m) {
  // 实现 mkl 库在 MkldnnCPU 上的函数注册，注册线性操作函数 mkl_linear
  m.impl(TORCH_SELECTIVE_NAME("mkl::_mkl_linear"), TORCH_FN(mkl_linear));
}

#endif// AT_MKL_ENABLED

TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  // 实现 mkldnn 库在 CPU 上的函数注册，注册线性点操作函数 mkldnn_linear_pointwise
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise"),
      TORCH_FN(mkldnn_linear_pointwise));
  // 注册 mkldnn 库在 CPU 上的二进制线性点操作函数 mkldnn_linear_pointwise_binary
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise.binary"),
      TORCH_FN(mkldnn_linear_pointwise_binary));
}

TORCH_LIBRARY_IMPL(mkldnn, MkldnnCPU, m) {
  // 实现 mkldnn 库在 MkldnnCPU 上的函数注册，注册线性点操作函数 mkldnn_linear_pointwise
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise"),
      TORCH_FN(mkldnn_linear_pointwise));
  // 注册 mkldnn 库在 MkldnnCPU 上的二进制线性点操作函数 mkldnn_linear_pointwise_binary
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise.binary"),
      TORCH_FN(mkldnn_linear_pointwise_binary));
}

} // 结束 namespace at

#endif // AT_MKLDNN_ENABLED
```