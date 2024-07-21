# `.\pytorch\aten\src\ATen\native\SoftMax.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，用于仅在方法运算符中断言

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/cpu/SoftmaxKernel.h>
#include <ATen/NamedTensorUtils.h>
// 包含 ATen 库的头文件，用于张量操作和计算

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_log_softmax.h>
#include <ATen/ops/_log_softmax_backward_data_native.h>
#include <ATen/ops/_log_softmax_native.h>
#include <ATen/ops/_masked_softmax_backward_native.h>
#include <ATen/ops/_masked_softmax_native.h>
#include <ATen/ops/_softmax.h>
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/log_softmax.h>
#include <ATen/ops/log_softmax_native.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/softmax_native.h>
#include <ATen/ops/special_log_softmax_native.h>
#include <ATen/ops/special_softmax_native.h>
#endif
// 根据宏的定义，选择性地包含不同的 ATen 操作头文件

#include <c10/core/TensorOptions.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
// 包含 c10 库的头文件，提供张量选项和宏定义

namespace at::meta {
// 进入 ATen 命名空间 at::meta

TORCH_META_FUNC(_softmax)
(const Tensor& input, const int64_t dim, const bool half_to_float) {
  // 定义 _softmax 的元函数，接受输入张量、维度和是否转换为浮点数的标志作为参数

  int64_t dim_ = maybe_wrap_dim(dim, input.dim());
  // 根据输入的维度和张量的维度，确定有效的维度值

  auto output_options =
      input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 获取输入张量的选项，并设置内存格式为 LEGACY_CONTIGUOUS_MEMORY_FORMAT

  if (half_to_float) {
    output_options = output_options.dtype(ScalarType::Float);
    // 如果需要将半精度转换为单精度，则设置输出选项的数据类型为 Float
  }

  int64_t input_dim = input.dim() > 0 ? input.dim() : 1;
  // 确定输入张量的维度数量，至少为 1

  TORCH_CHECK(
      dim_ >= 0 && dim_ < input_dim,
      "dim must be non-negative and less than input dimensions");
  // 使用断言检查维度 dim 是否合法

  set_output_raw_strided(0, input.sizes(), {}, output_options);
  // 设置输出张量的原始步幅，并使用指定的选项
}

TORCH_META_FUNC(_log_softmax) (
  const Tensor& input,
  const int64_t dim,
  const bool half_to_float) {
  // 定义 _log_softmax 的元函数，接受输入张量、维度和是否转换为浮点数的标志作为参数

  int64_t dim_ = maybe_wrap_dim(dim, input.dim());
  // 根据输入的维度和张量的维度，确定有效的维度值

  auto output_options =
      input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 获取输入张量的选项，并设置内存格式为 LEGACY_CONTIGUOUS_MEMORY_FORMAT

  if (half_to_float) {
    output_options = output_options.dtype(ScalarType::Float);
    // 如果需要将半精度转换为单精度，则设置输出选项的数据类型为 Float
  }

  int64_t input_dim = input.dim() > 0 ? input.dim() : 1;
  // 确定输入张量的维度数量，至少为 1

  TORCH_CHECK(
      dim_ >= 0 && dim_ < input_dim,
      "dim must be non-negative and less than input dimensions");
  // 使用断言检查维度 dim 是否合法

  set_output_raw_strided(0, input.sizes(), {}, output_options);
  // 设置输出张量的原始步幅，并使用指定的选项
}

TORCH_META_FUNC(_softmax_backward_data)
(const Tensor& grad,
 const Tensor& output,
 int64_t dim,
 ScalarType input_dtype) {
  // 定义 _softmax_backward_data 的元函数，接受梯度张量、输出张量、维度和输入数据类型作为参数

  TensorArg grad_arg{grad, "grad", 1}, output_arg{output, "output", 2};
  // 创建张量参数对象，用于后续的检查

  checkSameSize("softmax_backward", grad_arg, output_arg);
  // 检查梯度张量和输出张量的尺寸是否相同

  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  // 根据输入的维度和梯度张量的维度，确定有效的维度值

  auto grad_input_options =
      grad.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 获取梯度张量的选项，并设置内存格式为 LEGACY_CONTIGUOUS_MEMORY_FORMAT

  bool half_to_float = grad.scalar_type() != input_dtype;
  // 检查是否需要将半精度转换为单精度

  if (half_to_float) {
    // 如果需要转换为单精度
    // 下面的代码仅适用于 CUDA 实现，这里的 "okay" 是允许的
    // 检查梯度的标量类型是否为 float，并且输入的数据类型为 half
    // 这里放置注释是因为 CPU 实现的 _softmax 不支持 half 到 float 的转换。
    // 在 CUDA 实现中有一个 TORCH_CHECK 语句，理想情况下也应该放在这里，
    // 但是至少有一个测试中，这个内核的 CPU 实现中梯度和输入的数据类型不匹配，
    // 并不是所有情况下梯度类型是 float，输入数据类型是 half（参见 #63057）。

    if (grad.scalar_type() == ScalarType::Float &&
        input_dtype == ScalarType::Half) {
      // 如果满足条件，则将 grad_input_options 的数据类型设置为 ScalarType::Half
      grad_input_options = grad_input_options.dtype(ScalarType::Half);
    }
  }

  // 计算梯度的维度，如果梯度的维度大于 0，则使用 grad.dim()，否则使用默认值 1
  int64_t grad_dim = grad.dim() > 0 ? grad.dim() : 1;
  // 检查 dim_ 是否在有效范围内，必须非负且小于输入的维度
  TORCH_CHECK(
      dim_ >= 0 && dim_ < grad_dim,
      "dim must be non-negative and less than input dimensions");

  // 设置输出张量的原始步幅，输出张量为第 0 个索引，大小为 grad.sizes()，没有额外的步幅信息，
  // 使用之前可能根据条件更新过的 grad_input_options。
  set_output_raw_strided(0, grad.sizes(), {}, grad_input_options);
} // 结束 _log_softmax_backward_data 函数定义

namespace at::meta {

// 定义 _log_softmax_backward_data 函数，用于反向传播 LogSoftMax 操作的梯度计算
TORCH_META_FUNC(_log_softmax_backward_data)
(const Tensor& grad,
 const Tensor& output,
 int64_t dim,
 ScalarType input_dtype){
  
  // 计算在操作维度上的有效维度值
  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());

  // 根据输入梯度的内存格式创建梯度输入选项
  TensorOptions grad_input_options(
      grad.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));

  // 检查是否需要将半精度转换为单精度
  bool half_to_float = grad.scalar_type() != input_dtype;
  if (half_to_float) {
    // 如果输入梯度类型是 float，而输入数据类型是 half，则使用 half 类型的选项
    if (grad.scalar_type() == ScalarType::Float &&
        input_dtype == ScalarType::Half) {
      grad_input_options = grad_input_options.dtype(ScalarType::Half);
    }
  }

  // 检查操作维度的有效性
  int64_t grad_dim = grad.dim() > 0 ? grad.dim() : 1;
  TORCH_CHECK(
      dim_ >= 0 && dim_ < grad_dim,
      "dim must be non-negative and less than input dimensions");

  // 设置输出张量的原始步幅
  set_output_raw_strided(0, grad.sizes(), {}, grad_input_options);
}

} // 结束 at::meta 命名空间

namespace at::native {

// 匿名命名空间内定义的模板函数，用于执行 Host 端的 SoftMax 操作
template <typename scalar_t, bool LogSoftMax, bool MaskedSoftMax = false>
void host_softmax(
    Tensor output,
    const Tensor& input,
    const int64_t dim,
    bool* mask = nullptr,
    const std::optional<int64_t> mask_type_ = {}) {

  // 如果使用 MaskedSoftMax，则需检查 mask_type_ 是否已定义
  if (MaskedSoftMax) {
    TORCH_CHECK(mask_type_.has_value(), "Mask Type should be defined");
    int64_t mask_type = mask_type_.value();
    // 检查 mask_type 是否合法
    TORCH_CHECK((mask_type == 0) || (mask_type == 1) || (mask_type == 2), "Mask Type should be 0 (src_mask) or 1 (src_key_padding_mask), or 2 (default_mask)");
  }

  // 计算外部尺寸、操作维度的尺寸和内部尺寸
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);
  int64_t inner_size = 1;
  for (const auto i : c10::irange(dim)) {
    outer_size *= input.size(i);
  }
  for (int64_t i = dim + 1; i < input.dim(); ++i) {
    // 累积计算内部尺寸
    inner_size *= input.size(i);
  }
}

// 匿名命名空间内定义的模板函数，用于执行 Host 端的 SoftMax 反向传播操作
template <typename scalar_t, bool LogSoftMax, bool MaskedSoftMax = false>
void host_softmax_backward(
    const Tensor& gI,
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    bool* mask = nullptr) {

  // 计算外部尺寸、梯度张量在操作维度上的尺寸和内部尺寸
  int64_t outer_size = 1;
  int64_t dim_size = grad.size(dim);
  int64_t inner_size = 1;
  for (const auto i : c10::irange(dim)) {
    outer_size *= grad.size(i);
  }
  for (int64_t i = dim + 1; i < grad.dim(); ++i) {
    // 累积计算内部尺寸
    inner_size *= grad.size(i);
  }
}
    // 计算每个内部维度的总大小
    inner_size *= grad.size(i);
  }
  // 计算维度步长
  int64_t dim_stride = inner_size;
  // 计算外部步长
  int64_t outer_stride = dim_size * dim_stride;
  // 获取梯度输入数据的基地址
  scalar_t* gradInput_data_base = gI.data_ptr<scalar_t>();
  // 获取输出数据的基地址
  scalar_t* output_data_base = output.data_ptr<scalar_t>();
  // 获取梯度输出数据的基地址
  scalar_t* gradOutput_data_base = grad.data_ptr<scalar_t>();
  // 获取掩码数据的基地址
  bool* mask_data_base = mask;
  // 计算并行任务的颗粒大小，限制在内部维度和GRAIN_SIZE/dim_size之间的较小值
  int64_t grain_size = std::min(internal::GRAIN_SIZE / dim_size, (int64_t)1);
  // 并行执行for循环，遍历所有外部和内部索引范围
  parallel_for(
      0, outer_size * inner_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
          // 计算外部索引和内部索引
          int64_t outer_idx = i / inner_size;
          int64_t inner_idx = i % inner_size;
          // 计算梯度输入数据的地址
          scalar_t* gradInput_data =
              gradInput_data_base + outer_idx * outer_stride + inner_idx;
          // 计算输出数据的地址
          scalar_t* output_data =
              output_data_base + outer_idx * outer_stride + inner_idx;
          // 计算梯度输出数据的地址
          const scalar_t* gradOutput_data =
              gradOutput_data_base + outer_idx * outer_stride + inner_idx;
          // 初始化掩码数据为空指针
          bool* mask_data = nullptr;
          // 如果使用掩码Softmax，设置掩码数据的地址
          if (MaskedSoftMax) {
            mask_data = mask_data_base + outer_idx * outer_stride + inner_idx;
          }

          // 初始化求和变量为0
          acc_type<scalar_t, false> sum = 0;
          // 遍历所有维度
          for (const auto d : c10::irange(dim_size)) {
            // 如果不是MaskedSoftmax或者掩码不为真，则执行下列语句
            if (!MaskedSoftMax || !mask_data[d * dim_stride]) {
              // 如果使用LogSoftmax，则将梯度输出数据加到求和变量上
              if (LogSoftMax) {
                sum += gradOutput_data[d * dim_stride];
              } else {
                // 否则，将梯度输出数据乘以输出数据后加到求和变量上
                sum +=
                    gradOutput_data[d * dim_stride] * output_data[d * dim_stride];
              }
            }
          }

          // 再次遍历所有维度
          for (const auto d : c10::irange(dim_size)) {
            // 如果使用MaskedSoftmax并且掩码为真，则将梯度输入数据设为0
            if (MaskedSoftMax && mask_data[d * dim_stride]) {
              gradInput_data[d * dim_stride] = 0;
            }
            // 否则，根据是否使用LogSoftmax，计算梯度输入数据的值
            else if (LogSoftMax) {
              gradInput_data[d * dim_stride] = gradOutput_data[d * dim_stride] -
                  std::exp(output_data[d * dim_stride]) * sum;
            } else {
              gradInput_data[d * dim_stride] = output_data[d * dim_stride] *
                  (gradOutput_data[d * dim_stride] - sum);
            }
          }
        }
      });
} // namespace

// 定义 TORCH_IMPL_FUNC 函数 softmax_cpu_out，计算输入张量的 softmax 在 CPU 上，并输出到指定张量
TORCH_IMPL_FUNC(softmax_cpu_out)
(const Tensor& input,
 const int64_t dim,
 const bool half_to_float,
 const Tensor& output) {
  
  // 检查是否有 half_to_float 转换，如果有则报错，CPU 上不支持这种转换
  TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on CPU");

  // 如果输入张量为空，则直接返回
  if (input.numel() == 0) {
    return;
  }

  // 将输入张量变成连续的
  auto input_ = input.contiguous();
  // 对 dim 进行边界处理
  int64_t dim_ = maybe_wrap_dim(dim, input_.dim());

  // 如果输入张量的维度为 0，则将其视作一个元素的张量
  if (input_.dim() == 0) {
    input_ = input_.view(1);
  }

  // 检查 dim 是否在有效范围内
  TORCH_CHECK(
      dim_ >= 0 && dim_ < input_.dim(),
      "dim must be non-negative and less than input dimensions");

  // 如果输入张量的维度大于 0 并且 dim_ 等于最后一个维度，调用 softmax_lastdim_kernel 计算
  if (input_.ndimension() > 0 && dim_ == input_.ndimension() - 1) {
    softmax_lastdim_kernel(kCPU, output, input_);
  } else {
    // 否则，调用 softmax_kernel 计算
    softmax_kernel(kCPU, output, input_, dim_);
  }
}

// 定义 TORCH_IMPL_FUNC 函数 log_softmax_cpu_out，计算输入张量的 log softmax 在 CPU 上，并输出到指定张量
TORCH_IMPL_FUNC(log_softmax_cpu_out)
(const Tensor& input,
 const int64_t dim,
 const bool half_to_float,
 const Tensor& output) {
  
  // 检查是否有 half_to_float 转换，如果有则报错，CPU 上不支持这种转换
  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on CPU");

  // 如果输入张量为空，则直接返回
  if (input.numel() == 0) {
    return;
  }

  // 将输入张量变成连续的
  auto input_ = input.contiguous();
  // 对 dim 进行边界处理
  int64_t dim_ = maybe_wrap_dim(dim, input_.dim());

  // 如果输入张量的维度为 0，则将其视作一个元素的张量
  if (input_.dim() == 0) {
    input_ = input_.view(1);
  }

  // 如果输入张量的维度大于 0 并且 dim_ 等于最后一个维度，调用 log_softmax_lastdim_kernel 计算
  if (input_.ndimension() > 0 && dim_ == input_.ndimension() - 1) {
    log_softmax_lastdim_kernel(kCPU, output, input_);
  } else {
    // 否则，调用 log_softmax_kernel 计算
    log_softmax_kernel(kCPU, output, input_, dim_);
  }
}

// 定义 TORCH_IMPL_FUNC 函数 softmax_backward_cpu_out，计算 softmax 反向传播梯度在 CPU 上，并输出到指定张量
TORCH_IMPL_FUNC(softmax_backward_cpu_out)
(const Tensor& grad,
 const Tensor& output,
 int64_t dim,
 ScalarType input_dtype,
 const Tensor& grad_input) {
  
  // 对 dim 进行边界处理
  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  // 将梯度张量和输出张量变成连续的
  auto grad_ = grad.contiguous();
  auto output_ = output.contiguous();

  // 如果输出张量为空，则直接返回
  if (output.numel() == 0) {
    return;
  }

  // 如果梯度张量的维度为 0，则将其视作一个元素的张量
  if (grad_.dim() == 0) {
    grad_ = grad_.view(1);
  }

  // 如果输出张量的维度为 0，则将其视作一个元素的张量
  if (output_.dim() == 0) {
    output_ = output_.view(1);
  }

  // 如果梯度张量的维度大于 0 并且 dim_ 等于最后一个维度，调用 softmax_backward_lastdim_kernel 计算
  if (grad_.ndimension() > 0 && dim_ == grad_.ndimension() - 1) {
    softmax_backward_lastdim_kernel(kCPU, grad_input, grad_, output);
  } else {
    // 否则，调用 softmax_backward_kernel 计算
    softmax_backward_kernel(kCPU, grad_input, grad_, output, dim_);
  }
}

// 定义 TORCH_IMPL_FUNC 函数 log_softmax_backward_cpu_out，计算 log softmax 反向传播梯度在 CPU 上，并输出到指定张量
TORCH_IMPL_FUNC(log_softmax_backward_cpu_out) (
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype,
    const Tensor& grad_input) {
  
  // 对 dim 进行边界处理
  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  // 将梯度张量和输出张量变成连续的
  auto grad_ = grad.contiguous();
  auto output_ = output.contiguous();

  // 如果输出张量不为空，则进行下一步操作
  if (output.numel() != 0) {
    // 如果梯度张量的维度为 0，则将其视作一个元素的张量
    if (grad_.dim() == 0)
      grad_ = grad_.view(1);
    // 如果输出张量的维度为 0，则将其视作一个元素的张量
    if (output_.dim() == 0) {
      output_ = output_.view(1);
    }
    // 如果梯度张量的维度大于 0 并且 dim_ 等于最后一个维度，调用 log_softmax_backward_lastdim_kernel 计算
    if (grad_.ndimension() > 0 && dim_ == grad_.ndimension() - 1) {
      log_softmax_backward_lastdim_kernel(kCPU, grad_input, grad_, output_);
    } else {
      // 否则，调用 log_softmax_backward_kernel 计算
      log_softmax_backward_kernel(kCPU, grad_input, grad_, output_, dim_);
    }
  }
}

// 定义 softmax 函数，计算输入张量的 softmax，并返回结果
Tensor softmax(const Tensor& input_, const int64_t dim_, std::optional<ScalarType> dtype) {
  // 使用 lambda 表达式包裹计算过程
  auto result = [&]() {
    // 临时取消命名保护
    NoNamesGuard guard;
    // 函数体略

    NoNamesGuard guard;
    // 函数体略
  };
  // 返回计算结果
  return result();
}
    // 检查输入是否在 CUDA 上运行，并且是半精度（half）类型，并且要求输出是单精度（float）类型
    if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
        // 对输入进行 softmax 操作，并返回结果
        return at::_softmax(input_, dim_, true);
    } else {
        // 如果不满足条件，则根据指定的 dtype 转换输入的数据类型，如果未指定则保持不变
        Tensor converted = dtype.has_value() ? input_.toType(dtype.value()) : input_;
        // 对转换后的数据进行 softmax 操作，并返回结果
        return at::_softmax(converted, dim_, false);
    }
  }();
  // 将输入张量的命名信息传播到输出张量中
  namedinference::propagate_names(result, input_);
  // 返回进行 softmax 后的结果张量
  return result;
}

Tensor& softmax_out(
    const Tensor& input_,
    const int64_t dim_,
    std::optional<ScalarType> dtype,
    Tensor& output_) {
  Tensor output_temp;
  // 检查输入张量是否在 CUDA 上并且数据类型为半精度（Half），目标数据类型为单精度（Float）
  if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half &&
      dtype == ScalarType::Float) {
    // 如果输出张量不是连续的，创建一个与其相同选项的空张量
    if (!output_.is_contiguous()) {
      auto options =
          TensorOptions().dtype(output_.dtype()).device(output_.device());
      output_temp = at::empty(output_.sizes(), options);
      // 执行 softmax 操作，将结果写入临时张量
      at::_softmax_out(output_temp, input_, dim_, true);
    } else {
      // 直接执行 softmax 操作，将结果写入输出张量
      at::_softmax_out(output_, input_, dim_, true);
    }
  } else {
    // 转换输入张量的数据类型，若未指定目标数据类型则保持原数据类型
    Tensor converted =
        dtype.has_value() ? input_.toType(dtype.value()) : input_;
    // 如果输出张量不是连续的，创建一个与其相同选项的空张量
    if (!output_.is_contiguous()) {
      auto options =
          TensorOptions().dtype(output_.dtype()).device(output_.device());
      output_temp = at::empty(output_.sizes(), options);
      // 执行 softmax 操作，将结果写入临时张量
      at::_softmax_out(output_temp, converted, dim_, false);
    } else {
      // 直接执行 softmax 操作，将结果写入输出张量
      at::_softmax_out(output_, converted, dim_, false);
    }
  }

  // 如果输出张量不是连续的，调整输出张量的大小，并将临时张量的内容复制到输出张量
  if (!output_.is_contiguous()) {
    output_.resize_(output_temp.sizes());
    output_.copy_(output_temp);
  }

  // 返回输出张量的引用
  return output_;
}

// special_softmax, alias for softmax
// 对 softmax 的简单包装，直接调用 ATen 提供的 softmax 函数
Tensor special_softmax(const Tensor& input_, const int64_t dim_, std::optional<ScalarType> dtype) {
  return at::softmax(input_, dim_, dtype);
}

// 计算输入张量在指定维度上的 log_softmax
Tensor log_softmax(const Tensor& input_, const int64_t dim_, std::optional<ScalarType> dtype) {
  // 使用 lambda 函数包裹以避免命名冲突，并根据条件选择执行不同的 log_softmax 操作
  auto result = [&]() {
    NoNamesGuard guard;
    // 检查输入张量是否在 CUDA 上并且数据类型为半精度（Half），目标数据类型为单精度（Float）
    if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
        return at::_log_softmax(input_, dim_, true);
    } else {
        // 转换输入张量的数据类型，若未指定目标数据类型则保持原数据类型
        Tensor converted = dtype.has_value()? input_.toType(dtype.value()) : input_;
        // 执行 log_softmax 操作
        return at::_log_softmax(converted, dim_, false);
    }
  }();
  // 将结果张量的命名信息从输入张量传播到输出
  namedinference::propagate_names(result, input_);
  // 返回执行 log_softmax 后的结果张量
  return result;
}

Tensor& log_softmax_out(
    const Tensor& input_,
    const int64_t dim_,
    std::optional<ScalarType> dtype,
    Tensor& output_) {
  Tensor output_temp;
  // 检查输入张量是否在 CUDA 上并且数据类型为半精度（Half），目标数据类型为单精度（Float）
  if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half &&
      dtype == ScalarType::Float) {
    // 如果输出张量不是连续的，创建一个与其相同选项的空张量
    if (!output_.is_contiguous()) {
      auto options =
          TensorOptions().dtype(output_.dtype()).device(output_.device());
      output_temp = at::empty(output_.sizes(), options);
      // 执行 log_softmax 操作，将结果写入临时张量
      at::_log_softmax_out(output_temp, input_, dim_, true);
    } else {
      // 直接执行 log_softmax 操作，将结果写入输出张量
      at::_log_softmax_out(output_, input_, dim_, true);
    }
  } else {
    // 转换输入张量的数据类型，若未指定目标数据类型则保持原数据类型
    Tensor converted =
        dtype.has_value() ? input_.toType(dtype.value()) : input_;
    // 如果输出张量不是连续的，创建一个与其相同选项的空张量
    if (!output_.is_contiguous()) {
      auto options =
          TensorOptions().dtype(output_.dtype()).device(output_.device());
      output_temp = at::empty(output_.sizes(), options);
      // 执行 log_softmax 操作，将结果写入临时张量
      at::_log_softmax_out(output_temp, converted, dim_, false);
    } else {
      // 直接执行 log_softmax 操作，将结果写入输出张量
      at::_log_softmax_out(output_, converted, dim_, false);
    }
  }

  // 如果输出张量不是连续的，调整输出张量的大小，并将临时张量的内容复制到输出张量
  if (!output_.is_contiguous()) {
    output_.resize_(output_temp.sizes());
    output_.copy_(output_temp);
  }

  // 返回输出张量的引用
  return output_;
}
    output_.copy_(output_temp);
  }

  return output_;


注释：


    // 将 output_temp 的值复制到 output_ 中
    output_.copy_(output_temp);
  }

  // 返回复制后的 output_ 对象
  return output_;


这段代码看起来是C++或类似语言的代码片段。注释解释了每行代码的作用，尤其是 `output_.copy_(output_temp);` 这一行是在将 `output_temp` 的内容复制到 `output_` 中。最后一行 `return output_;` 则返回了复制后的 `output_` 对象。
}

// 定义特殊版本的 log_softmax 函数，对输入张量在指定维度进行 log_softmax 操作
Tensor special_log_softmax(const Tensor& input, const int64_t dim, std::optional<ScalarType> dtype) {
  return at::log_softmax(input, dim, dtype);
}

// 定义几个分发器，用于分发不同版本的 softmax 和 log_softmax 的内核实现
DEFINE_DISPATCH(softmax_lastdim_kernel);
DEFINE_DISPATCH(log_softmax_lastdim_kernel);
DEFINE_DISPATCH(softmax_backward_lastdim_kernel);
DEFINE_DISPATCH(log_softmax_backward_lastdim_kernel);

DEFINE_DISPATCH(softmax_kernel);
DEFINE_DISPATCH(log_softmax_kernel);
DEFINE_DISPATCH(softmax_backward_kernel);
DEFINE_DISPATCH(log_softmax_backward_kernel);

// 定义 softmax 函数，对输入张量在指定命名维度上进行 softmax 操作
Tensor softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  return at::softmax(self, dimname_to_position(self, dim), dtype);
}

// 定义 log_softmax 函数，对输入张量在指定命名维度上进行 log_softmax 操作
Tensor log_softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  return at::log_softmax(self, dimname_to_position(self, dim), dtype);
}

// 在 CPU 上实现带掩码的 softmax 操作
Tensor masked_softmax_cpu(const Tensor& input_, const Tensor& mask_, const std::optional<int64_t> dim_, const std::optional<int64_t> mask_type_) {

  // 使掩码张量连续存储，以便后续操作
  auto mask = mask_.contiguous();
  auto mask_type = mask_type_; // 可能会在下面进行转换

  // 检查掩码张量类型是否为布尔类型
  TORCH_CHECK(
      mask_.scalar_type() == ScalarType::Bool,
      "Mask should be a boolean tensor");

  // 检查掩码张量维度是否正确
  if ((mask.dim() != 2) || (input_.dim() != 4)) {
    // 当掩码类型为 0 或 1 时，仅允许用于 2D 掩码和 4D 输入
    mask_type = 2;
  }

  if (mask_type == 2) {
      // 当掩码类型为 2 时，检查掩码形状是否与输入形状匹配
      TORCH_CHECK(input_.sizes() == mask.sizes(),
                  "For mask_type == 2 mask shape should match input shape")
  } else if (mask_type == 1) {
      // 当掩码类型为 1 时，检查掩码形状是否为 (B, L)
      TORCH_CHECK((input_.sizes()[0] == mask.sizes()[0]) && (input_.sizes()[2] == mask.sizes()[1]),
                  "For mask_type == 1 mask shape should be (B, L)");
      if (dim_ != input_.dim() - 1) {
            // 如果 softmax 应用于最后一个维度，则以优化方式处理填充掩码，否则扩展为通用的 4D 掩码
            mask = mask_.view({input_.sizes()[0], 1, 1, input_.sizes()[2]});
            mask = mask.expand(input_.sizes()).contiguous();
            mask_type = 2;
      }
  } else if (mask_type == 0) {
      // 当掩码类型为 0 时，检查掩码形状是否为 (L, L)
      TORCH_CHECK((mask.dim() == 2) && (input_.sizes()[2] == mask.sizes()[0]) && (input_.sizes()[2] == mask.sizes()[1]),
                  "For mask_type == 0 mask shape should be (L, L)");
      if (dim_ != input_.dim() - 1) {
            // 如果 softmax 应用于最后一个维度，则以优化方式处理注意力掩码，否则扩展为通用的 4D 掩码
            mask = mask.view({1, 1, input_.sizes()[2], input_.sizes()[2]});
            mask = mask.expand(input_.sizes()).contiguous();
            mask_type = 2;
      }
  }

  // 创建与输入张量相同形状和类型的输出张量
  Tensor output = at::empty_like(input_, input_.options());
  // 使输入张量连续存储，以便后续操作
  auto input = input_.contiguous();
  // 获取进行 softmax 操作的维度，如果未指定，则使用输入张量的最后一个维度
  int64_t dim = dim_.has_value() ? dim_.value() : input.dim() - 1;
  dim = maybe_wrap_dim(dim, input_.dim());

  // 如果输入张量是标量，直接返回空张量
  if (input.dim() == 0) {
    // 将输入张量视图调整为1维张量
    input = input.view(1);
  }

  // 使用AT_DISPATCH_FLOATING_TYPES_AND2宏，根据输入张量的数据类型调度不同的操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "masked_softmax", [&] {
        // 调用host_softmax函数进行softmax计算，其中包括以下参数：
        // output: 输出张量
        // input: 输入张量
        // dim: 指定进行softmax的维度
        // mask.data_ptr<bool>(): 表示掩码张量的数据指针，用于在softmax中进行掩码处理
        // mask_type: 控制掩码的类型
        host_softmax<
            scalar_t,               // 模板参数：数据类型
            false /* LogSoftMax */, // 是否为对数softmax
            true /* MaskedSoftMax */>( // 是否为掩码softmax
            output, input, dim, mask.data_ptr<bool>(), mask_type);
      });
  // 返回计算后的输出张量
  return output;
}
```