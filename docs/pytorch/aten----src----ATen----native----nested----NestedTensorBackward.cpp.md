# `.\pytorch\aten\src\ATen\native\nested\NestedTensorBackward.cpp`

```py
// 引入 ATen 库中的各种头文件，用于张量操作和计算
#include <ATen/native/nested/NestedTensorMath.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/layer_norm.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/nested/NestedTensorMath.h>
#include <ATen/native/layer_norm.h>
#include <c10/core/DeviceType.h>

// 定义 ATen 命名空间下的 native 命名空间，用于嵌套张量相关的操作
namespace at {
namespace native {

// 在 NestedTensorMath.cpp 文件中查看 [nested tensor matmul] 注释
std::tuple<Tensor, Tensor> matmul_backward_nested(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other,
    std::array<bool, 2> grad_input_mask) {
  // 如果梯度张量未定义，返回空张量元组
  if (!grad.defined()) {
    return std::make_tuple(Tensor(), Tensor());
  }
  Tensor grad_self, grad_other;
  // 如果需要计算 self 的梯度
  if (grad_input_mask[0]) {
    // 计算 grad 对 other 的转置矩阵的乘积，并赋给 grad_self
    grad_self = at::matmul(grad, other.transpose(-1, -2));
  }
  // 如果需要计算 other 的梯度
  if (grad_input_mask[1]) {
    // 计算 self 的转置矩阵与 grad 的乘积，并赋给 grad_other
    grad_other = at::matmul(self.transpose(-1, -2), grad);
  }
  // 返回计算得到的梯度张量 grad_self 和 grad_other 的元组
  return std::make_tuple(grad_self, grad_other);
}

// 反向传播过程中的嵌套线性层梯度计算
std::tuple<Tensor, Tensor, Tensor> nested_linear_backward(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    std::array<bool, 3> output_mask) {
  // 如果梯度输出张量未定义，返回空张量元组
  if (!grad_output.defined()) {
    return std::tuple<Tensor, Tensor, Tensor>{Tensor(), Tensor(), Tensor()};
  }
  Tensor grad_input, grad_weight, grad_bias;
  // 确保梯度输出张量是连续的
  auto grad_output_contiguous = grad_output.contiguous();
  // 获取输入和梯度输出张量的嵌套张量实现指针
  auto* nt_grad_output = get_nested_tensor_impl(grad_output_contiguous);
  auto* nt_input = get_nested_tensor_impl(input);
  // 内部断言，确保嵌套张量实现非空并且连续
  TORCH_INTERNAL_ASSERT(nt_grad_output != nullptr);
  TORCH_INTERNAL_ASSERT(nt_input != nullptr);
  TORCH_INTERNAL_ASSERT(nested_tensor_impl_is_contiguous(nt_grad_output));
  // 获取梯度输出和输入的缓冲区
  auto grad_output_buffer = nt_grad_output->get_buffer();
  auto input_buffer = nt_input->get_buffer();

  // 重塑梯度输出缓冲区形状以便矩阵乘法
  auto reshaped_grad = grad_output_buffer.reshape({-1, weight.size(0)});

  // 如果需要计算输入的梯度
  if (output_mask[0]) {
    // 计算重塑后的梯度乘以权重的矩阵乘积，并视图重塑结果
    auto grad_input_buffer = at::mm(reshaped_grad, weight).view({-1});
    auto grad_input_nt_size = nt_input->get_nested_sizes().clone();
    grad_input = wrap_buffer(grad_input_buffer, grad_input_nt_size);
  }
  // 如果需要计算权重的梯度
  if (output_mask[1]) {
    // 计算重塑后的梯度的转置乘以输入缓冲区的矩阵乘积
    grad_weight =
        at::mm(reshaped_grad.t(), input_buffer.reshape({-1, weight.size(1)}));
  }
  // 如果需要计算偏置的梯度
  if (output_mask[2]) {
    // 按第一个维度对重塑后的梯度进行求和，得到偏置的梯度
    grad_bias = reshaped_grad.sum(0);
  }
  // 返回计算得到的梯度张量 grad_input, grad_weight 和 grad_bias 的元组
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

// 嵌套 softmax 反向传播过程
Tensor nested_softmax_backward(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype) {
  // 内部断言，确保 grad 和 output 张量都是嵌套张量
  TORCH_INTERNAL_ASSERT(grad.is_nested(), "Should be nested grad")
  TORCH_INTERNAL_ASSERT(output.is_nested(), "Should be nested output")

  auto output_ptr = get_nested_tensor_impl(output);
  auto grad_ptr = get_nested_tensor_impl(grad);
  int64_t ntensors = output_ptr->size(0);
  // 如果输出张量中张量数量为 0，直接返回空张量
  if (ntensors == 0) {
    // 克隆梯度张量并返回副本
    return grad.clone();
  }
  // 将维度转换为正数维度，确保在有效范围内
  int64_t positive_dim = at::maybe_wrap_dim(dim, output_ptr->dim());

  // 获取输出张量的信息
  const Tensor &output_buffer = output_ptr->get_buffer(),
               &output_sizemat = output_ptr->get_nested_sizes();

  // 获取梯度张量的信息
  const Tensor &grad_sizemat = grad_ptr->get_nested_sizes();

  // 断言输出张量和梯度张量的尺寸信息相等
  TORCH_INTERNAL_ASSERT(output_sizemat.equal(grad_sizemat));

  // 创建一个梯度输出张量，形状与输出缓冲区相同
  Tensor grad_output =
      wrap_buffer(at::empty_like(output_buffer), output_sizemat.clone());

  // 将梯度输出张量解绑定为单独的张量片段，用于计算导数
  std::vector<Tensor> grad_output_unbind{grad_output.unbind()},
      grad_unbind{grad.unbind()}, output_unbind{output.unbind()};

  // 对每个张量执行 softmax 反向传播数据计算
  for(const auto i: c10::irange(ntensors)) {
    at::_softmax_backward_data_out(
        grad_output_unbind[i],
        grad_unbind[i],
        output_unbind[i],
        positive_dim - 1,
        input_dtype);
  }
  // 返回计算得到的梯度输出张量
  return grad_output;
}

// 从梯度计算反向传播的基本求和，假设条件为 #82387
Tensor _nested_sum_backward_cpu(
  const Tensor& grad,  // 输入参数：梯度张量
  const Tensor& nested_self,  // 输入参数：嵌套张量自身
  OptionalIntArrayRef opt_dims,  // 可选参数：维度数组引用
  bool keepdim) {  // 输入参数：是否保持维度

  auto nt_self = get_nested_tensor_impl(nested_self);  // 获取嵌套张量的实现
  auto nt_grad = get_nested_tensor_impl(grad);  // 获取梯度张量的实现
  const Tensor& grad_buffer = nt_grad->get_buffer();  // 获取梯度张量的缓冲区
  const Tensor& self_buffer = nt_self->get_buffer();  // 获取嵌套张量的缓冲区
  auto grad_sizes = nt_grad->get_nested_sizes();  // 获取梯度张量的嵌套尺寸
  auto self_sizes = nt_self->get_nested_sizes();  // 获取嵌套张量的嵌套尺寸
  int64_t ntensors = nt_self->size(0);  // 获取嵌套张量的数量
  const Tensor& self_grad_buffer = self_buffer.new_empty(self_buffer.sizes());  // 创建与自身缓冲区尺寸相同的空张量

  auto num_segments = at::prod(grad_sizes, -1);  // 计算梯度尺寸的累积乘积
  auto segment_lengths = self_sizes.select(1, -1);  // 获取嵌套尺寸的最后一个维度

  // 这段逻辑暂时假设
  // (1) 所有梯度嵌套张量都是连续的
  // (2) 梯度嵌套张量在缓冲区中是连续存储的
  AT_DISPATCH_ALL_TYPES_AND2(
    ScalarType::Half, ScalarType::BFloat16, self_grad_buffer.scalar_type(), "nested_sum_dim_cpu", [&]() {
    auto* self_grad_data = self_grad_buffer.data_ptr<scalar_t>();  // 获取自身梯度数据指针
    const auto* output_grad_data = grad_buffer.const_data_ptr<scalar_t>();  // 获取输出梯度数据指针
    int64_t out_idx = 0, in_idx = 0;
    for (const auto i : c10::irange(ntensors)) {  // 遍历嵌套张量数量
      int64_t segments = num_segments[i].item<int64_t>();  // 获取段数
      int64_t segment_length = segment_lengths[i].item<int64_t>();  // 获取段长度
      for (auto j = 0; j < segments; j++) {  // 遍历每个段
        scalar_t output_grad = output_grad_data[out_idx];  // 获取输出梯度
        for (auto k = 0; k < segment_length; k++) {  // 遍历每个段的长度
          self_grad_data[in_idx] = output_grad;  // 将输出梯度赋值给自身梯度数据
          in_idx += 1;
        }
        out_idx += 1;
      }
    }
  });

  return wrap_buffer(self_grad_buffer, self_sizes);  // 返回包装后的自身梯度缓冲区和尺寸

}


Tensor _nested_select_backward_symint(
  const Tensor& grad,  // 输入参数：梯度张量
  const Tensor& nested_self,  // 输入参数：嵌套张量自身
  int64_t dim,  // 输入参数：维度
  c10::SymInt index) {  // 输入参数：符号整数索引

  auto nt_self = get_nested_tensor_impl(nested_self);  // 获取嵌套张量的实现
  const Tensor& self_buffer = nt_self->get_buffer();  // 获取嵌套张量的缓冲区
  const auto self_sizes = nt_self->get_nested_sizes();  // 获取嵌套张量的嵌套尺寸
  const Tensor& self_grad_buffer = self_buffer.new_zeros(self_buffer.sizes());  // 创建与自身缓冲区尺寸相同的零张量

  auto nt_grad = wrap_buffer(self_grad_buffer, self_sizes);  // 封装自身梯度缓冲区和尺寸
  nt_grad.select_symint(dim, index).copy_(grad);  // 根据维度和符号整数索引选择并复制梯度

  return nt_grad;  // 返回封装后的自身梯度

}

Tensor gelu_backwards_nested(const Tensor& grad, const Tensor& self, c10::string_view approximate){
    auto partial_gelu_backward = [approximate](auto && PH1, auto && PH2) { return at::gelu_backward(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), approximate); };
    return map_nt_binary(grad, self, partial_gelu_backward);  // 使用 map_nt_binary 对 gelu 反向传播进行部分求解
}

// 对 relu 命名约定
Tensor threshold_backwards_nested(const Tensor& grad_output, const Tensor& input, const Scalar& threshold){
    auto partial_relu_backward = [threshold](auto && PH1, auto && PH2) { return at::threshold_backward(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), threshold); };
    return map_nt_binary(grad_output, input, partial_relu_backward);  // 使用 map_nt_binary 对 relu 反向传播进行部分求解
}
// 定义函数 silu_backward_nested，计算 SiLU 激活函数的反向传播
Tensor silu_backward_nested(const Tensor& grad_output, const Tensor& self){
    // 定义局部函数 partial_silu_backward，调用 at::silu_backward 函数进行计算
    auto partial_silu_backward = [](auto && PH1, auto && PH2) { return at::silu_backward(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2)); };
    // 对 grad_output 和 self 进行 map_nt_binary 操作，使用 partial_silu_backward 函数
    return map_nt_binary(grad_output, self, partial_silu_backward);
}

// 定义函数 layer_norm_backward_nested，用于计算层归一化操作的反向传播
std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_nested(
    const Tensor& grad,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    std::array<bool, 3> grad_input_mask) {
  // 获取 grad 和 input 的 NestedTensor 实现
  auto* nt_impl_grad = get_nested_tensor_impl(grad);
  auto* nt_impl_input = get_nested_tensor_impl(input);
  // 获取 weight 和 bias（如果存在）
  const auto& weight = *weight_opt;
  const auto& bias = *bias_opt;
  // 获取输入 NestedTensor 的大小信息
  const auto& sizes = nt_impl_input->get_nested_sizes();
  // 检查 NestedTensor 输入、归一化参数、权重和偏置的有效性
  auto M_N = _check_nested_layer_norm_inputs(
      *nt_impl_input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;

  // 获取 gamma 和 beta 张量，并确保其是连续的
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  // 定义用于存储梯度的张量
  Tensor dInput;
  Tensor dgamma;
  Tensor dbeta;
  auto input_buffer = nt_impl_input->get_buffer();
  auto grad_buffer = nt_impl_grad->get_buffer();

  // 根据 grad_input_mask 的不同位，分配或初始化 dInput 张量
  if (grad_input_mask[0]) {
    dInput = at::native::empty_like(
        input_buffer,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  } else {
    dInput = at::native::zeros_like(
        input_buffer,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }

  // 根据 grad_input_mask 的不同位，分配或初始化 dgamma 张量
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous)
                   : at::native::zeros_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous);
  }

  // 根据 grad_input_mask 的不同位，分配或初始化 dbeta 张量
  if (grad_input_mask[2]) {
    // 根据 M 的值分配或初始化 dbeta 张量
    dbeta = M > 0 ? at::native::empty_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous)
                  : at::native::zeros_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous);
  }
    # 如果 M 大于 0，则创建一个与 beta 相同形状的空张量 dbeta，
    # 用于存储 LayerNorm 反向传播过程中的 beta 梯度；否则创建与 beta 相同形状的零张量 dbeta。
    dbeta = M > 0 ? at::native::empty_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous)
                  : at::native::zeros_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous);
  }
  # 如果 M 大于 0，则执行 LayerNormBackwardKernel 函数，计算 LayerNorm 的反向传播。
  # 传递相应的缓冲区、均值、倒数标准差、gamma 权重、M 和 N 的值，并计算 dInput、dgamma 和 dbeta。
  if (M > 0) {
    LayerNormBackwardKernel(
        input_buffer.is_cuda() ? kCUDA : kCPU,  # 根据输入缓冲区是否在 CUDA 上执行，确定使用的计算设备
        grad_buffer,                            # 梯度缓冲区
        input_buffer,                           # 输入缓冲区
        mean,                                   # 均值
        rstd,                                   # 倒数标准差
        *gamma,                                 # gamma 权重
        M,                                      # 批量大小 M
        N,                                      # 特征大小 N
        &dInput,                                # 输出的输入梯度
        &dgamma,                                # 输出的 gamma 梯度
        &dbeta);                                # 输出的 beta 梯度
  }
  # 返回一个包含 dInput、dgamma 和 dbeta 的元组，其中 dInput 被包装在指定大小的缓冲区中
  return std::make_tuple(
      wrap_buffer(dInput, sizes),               # 将 dInput 包装成特定大小的缓冲区
      std::move(dgamma),                        # 移动 dgamma 到返回的元组中
      std::move(dbeta));                        # 移动 dbeta 到返回的元组中
}

} // namespace native
} // namespace at
```