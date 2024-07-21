# `.\pytorch\aten\src\ATen\native\FunctionOfAMatrixUtils.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/FunctionOfAMatrixUtils.h>

#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_compute_linear_combination_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// 定义派发函数的分发器
DEFINE_DISPATCH(_compute_linear_combination_stub);

// 如果 `coefficients` 是一个 [m, n] 张量，
// `input` 是一个 [n, ...] 张量，那么输出 `output` 将是一个 [m, ...] 张量，
// 其中对于每个 i 在范围内：
//    对于每个 j 在范围内：
//        output[i, ...] += coefficients[i, j] * input[j, ...]
//
// 注意：如果 input.dtype == scalar_t<T>，则 coefficients.dtype == T。
// 这在 scalar_t<T> == complex<T> 时是相关的。
Tensor _compute_linear_combination(const Tensor& input, const Tensor& coefficients) {
  // 检查输入张量 input 不是空的
  TORCH_CHECK(input.ndimension() > 0 && input.numel() > 0, "Empty tensor not supported");
  // 获取 coefficients 的第一个维度大小
  auto output_first_dim_size = coefficients.size(0);

  // 设置输出张量的尺寸，保持与输入张量的内存格式一致
  auto output_sizes = input.sizes().vec();
  output_sizes[0] = output_first_dim_size;
  auto output = at::zeros(
    output_sizes,
    input.options().memory_format(at::MemoryFormat::Contiguous)
  );

  // 调用 native::_compute_linear_combination_out 函数来执行线性组合操作
  native::_compute_linear_combination_out(input, coefficients, output);

  // 返回结果张量
  return output;
}

// 注意：该函数使用了 __restrict__ 内存修饰符，
// 这意味着如果 `output` 实际上由 `input` 别名引用，那么产生的结果是未定义的。
Tensor& _compute_linear_combination_out(const Tensor& input, const Tensor& coefficients, Tensor& output) {
  // 获取 coefficients 和 input 的第一个维度大小
  auto output_first_dim_size = coefficients.size(0);
  auto input_first_dim_size = coefficients.size(1);

  // 回顾一下 `coefficients` 是一个 [m, n] 张量，
  // `input` 是一个 [n, ...] 张量，`output` 是一个 [m, ...] 张量。
  // 我们重新整理张量以共同的维度 == input.dim() + 1，以便
  // coefficients.sizes() = [m, 1 (代替 n), 1 重复 (input.dim() - 1) 次],
  // input.sizes() = [1, 1 (代替 n), ...],
  // output.sizes() = [m, 1 (代替 n), ...]。
  // 在内核中遍历新重新整理的张量的第二个维度。
  // 这样做是为了避免内核中的同步/原子操作，
  // 同时也保证了自动求导所需的确定性。

  // 重新整理 output 张量
  auto output_to_broadcasted_dim = output.unsqueeze(1);
  auto output_restrided_sizes = output_to_broadcasted_dim.sizes().vec();
  auto output_restrided_strides = output_to_broadcasted_dim.strides().vec();
  output_restrided_sizes[1] = 1;
  output_restrided_strides[1] = 0;
  auto output_restrided = output.as_strided(
    output_restrided_sizes,
    output_restrided_strides
  );
  // 输出的尺寸和步幅
  output_restrided_strides
  );

  // 对输入进行重新排列
  auto input_to_broadcasted_dim = input.unsqueeze(0);
  auto input_restrided_sizes = input_to_broadcasted_dim.sizes().vec();
  auto input_restrided_strides = input_to_broadcasted_dim.strides().vec();
  input_restrided_sizes[1] = 1;
  input_restrided_strides[1] = 0;
  auto input_restrided = input.as_strided(
    input_restrided_sizes,
    input_restrided_strides
  );

  // 对系数进行重新排列
  auto coefficients_restrided_sizes = std::vector<int64_t>(input.dim() + 1, 1);
  coefficients_restrided_sizes[0] = output_first_dim_size;
  coefficients_restrided_sizes[1] = 1;
  auto coefficients_restrided_strides = std::vector<int64_t>(input.dim() + 1, 0);
  coefficients_restrided_strides[0] = coefficients.stride(0);
  coefficients_restrided_strides[1] = 0;
  auto coefficients_restrided = coefficients.as_strided(
    coefficients_restrided_sizes,
    coefficients_restrided_strides
  );

  // 配置张量迭代器
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // 输出故意设置为 0 步幅
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(output_restrided)
    .add_input(input_restrided)
    .add_input(coefficients_restrided)
    .build();

  // 在内核中遍历尺寸为 n 的维度，
  // 这是 `input` 的第一个维度和 `coefficients` 的第二个维度
  auto input_stride = input.stride(0);
  auto coeff_stride = coefficients.stride(1);
  _compute_linear_combination_stub(
    iter.device_type(),
    iter,
    input_stride,
    coeff_stride,
    input_first_dim_size
  );
  return output;
}

} // namespace at::native
```