# `.\pytorch\aten\src\ATen\native\cpu\UnfoldBackwardKernel.cpp`

```
// 定义宏，根据操作系统选择正确的 __restrict 或 __restrict__ 关键字
#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

// 注意命名：这里的 grad_in 并不意味着它是关于输入的梯度，
// grad_in/grad_out 只是 unfold_backward 内核的输入和输出。
//
// unfold_backward，算法说明。
//
// 如果有 out = in.unfold(dim, size, step)，则
// out.shape[dim] == (in.shape[dim] - size) / step + 1，
// out.shape[-1] == size。
// out.dims() == in.dims() + 1。
//
// unfold_backward 接收 grad_in 并返回 grad_out，使得
// grad_in.shape == out.shape，
// grad_out.shape = in.shape。
//
// unfold_backward 考虑以下两种情况：
// case1. step >= size。
// case2. step < size。
//
// case1. step >= size。
// 在此情况下，迭代处理 grad_in 并执行以下复制：
// grad_out[..., i_out_dim,...] = grad_in[..., i_in_dim,..., i_in_last_dim]，
// 其中 i_out_dim = i_in_dim * step + i_in_last_dim。
//
// case2. step < size。
// 在此情况下，迭代处理 grad_out，
// 其中 grad_out[...,i_out_dim,...] 累积所有值
// grad_in[...,i_in_dim,...,i_in_last_dim]，其中
// i_in_dim 在 [left_idx_fold, right_idx_fold] 范围内，
// i_in_last_dim = i_out_dim - i_in_dim * step，
// left_idx_fold = (i_out_dim - size) / step，
// 如果 i_out_dim 在 [left_idx_fold * step, left_idx_fold * step + size) 内，
// 否则 left_idx_fold / step + 1，
// right_idx_fold = i_out_dim / step。
//
// 简单来说，给定 i_out_dim，我们找到与之相交的 grad_in 的折叠，
// 这些恰好是 [left_idx_fold, right_idx_fold]，
// 然后将 grad_in[...,i_in_dim,...,i_in_last_dim] 的相应值添加到 grad_out[...,i_out_dim,...] 中。

namespace at::native {

namespace {

// 内部模板函数 _unfold_backward_internal_kernel，处理特定类型的张量操作
template <typename scalar_t>
void _unfold_backward_internal_kernel(
  TensorIterator& iter,             // 张量迭代器，用于迭代操作
  int64_t size,                     // unfold 操作的窗口大小
  int64_t step,                     // unfold 操作的步长
  int64_t grad_in_dim_stride,       // grad_in 的维度步长
  int64_t grad_in_last_dim_stride,  // grad_in 的最后一维步长
  int64_t grad_in_dim_size,         // grad_in 的维度大小
  int64_t grad_out_dim_stride       // grad_out 的维度步长
) {
  // 如果迭代器的元素数量为 0，则直接返回
  if (iter.numel() == 0) {
    return;
  }

  // 使用 lambda 函数进行循环操作
  auto loop = [&](char** data, const int64_t* strides, int64_t nelems) {
    auto* RESTRICT grad_out_ptr = data[0];       // grad_out 指针，使用 RESTRICT 关键字限定
    auto* RESTRICT grad_in_ptr = data[1];        // grad_in 指针，使用 RESTRICT 关键字限定
    auto* RESTRICT idx_dim_ptr = data[2];        // 索引维度指针，使用 RESTRICT 关键字限定


这是注释了部分代码块，符合你的要求吗？
    // 对于每个元素执行循环，elem是迭代器的占位符，不使用
    for (const auto elem C10_UNUSED : c10::irange(nelems)) {
      // 将grad_out_ptr和grad_in_ptr转换为scalar_t类型的指针
      auto* RESTRICT grad_out_data = reinterpret_cast<scalar_t*>(grad_out_ptr);
      auto* RESTRICT grad_in_data = reinterpret_cast<scalar_t*>(grad_in_ptr);

      // 解析idx_dim_ptr指向的整数作为idx_dim
      auto idx_dim = *reinterpret_cast<int64_t*>(idx_dim_ptr);

      // 计算left_fold_idx，根据idx_dim、size和step确定左折叠的起始索引
      // 如果idx_dim > size，则(left_fold_idx = (idx_dim - size) / step)，否则为0
      int64_t left_fold_idx = (idx_dim > size) ? (idx_dim - size) / step : 0;
      // 如果idx_dim不在左折叠的有效范围内，则向上取整left_fold_idx
      if (!(left_fold_idx * step <= idx_dim && idx_dim < left_fold_idx * step + size)) {
        ++left_fold_idx;
      }

      // 计算right_fold_idx，根据idx_dim和step确定右折叠的结束索引
      auto right_fold_idx = idx_dim / step;
      // 如果right_fold_idx超出grad_in_dim_size的范围，则设置为grad_in_dim_size - 1
      right_fold_idx = (right_fold_idx >= grad_in_dim_size)
        ? (grad_in_dim_size - 1) : right_fold_idx;

      // 对于left_fold_idx到right_fold_idx之间的每个fold_idx进行循环
      for (auto fold_idx = left_fold_idx; fold_idx <= right_fold_idx; ++fold_idx) {
        // 计算idx_last_dim，作为当前处理维度的索引偏移
        auto idx_last_dim = idx_dim - fold_idx * step;
        // 更新grad_out_data，累加grad_in_data的对应元素到grad_out_data
        *grad_out_data += grad_in_data[fold_idx * grad_in_dim_stride
                                    + idx_last_dim * grad_in_last_dim_stride];
      }

      // 更新指针位置，以便处理下一个元素
      grad_out_ptr += strides[0];
      grad_in_ptr += strides[1];
      idx_dim_ptr += strides[2];
    }
  };

  // 使用iter对象调用for_each方法执行循环操作
  iter.for_each(loop);
}

// 定义一个函数，用于计算反折叠操作的 CPU 内核
void unfold_backward_cpu_kernel(
  Tensor& grad_out, // 输出梯度张量
  const Tensor& grad_in, // 输入梯度张量
  int64_t dim, // 维度参数
  int64_t size, // 尺寸参数
  int64_t step // 步长参数
) {
  dim = maybe_wrap_dim(dim, grad_out.dim()); // 确定有效的维度值，以处理负数
  // 最后一个维度存储了折叠后的数据
  auto last_dim = maybe_wrap_dim(-1, grad_in.dim());

  // 确保输入梯度张量在指定维度上有非空步长
  auto grad_in_dim_stride = ensure_nonempty_stride(grad_in, dim);
  // 确保输入梯度张量在最后一个维度上有非空步长
  auto grad_in_last_dim_stride = ensure_nonempty_stride(grad_in, last_dim);
  // 确保输入梯度张量在指定维度上有非空大小
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);

  // 确保输出梯度张量在指定维度上有非空步长
  auto grad_out_dim_stride = ensure_nonempty_stride(grad_out, dim);

  // 创建一个张量迭代器，用于迭代处理反折叠操作的梯度输出张量
  TensorIterator iter = _make_unfold_backward_iter_over_grad_out(
      grad_out, grad_in, dim, size, step);

  // 根据数据类型分发处理函数，执行反折叠内核操作
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(),
    "unfold_backward_cpu", [&] {
      _unfold_backward_internal_kernel<scalar_t>(
        iter,
        size,
        step,
        grad_in_dim_stride,
        grad_in_last_dim_stride,
        grad_in_dim_size,
        grad_out_dim_stride
      );
    }
  );
}

// 注册 CPU 版本的反折叠操作内核函数
REGISTER_DISPATCH(unfold_backward_stub, &unfold_backward_cpu_kernel);

// 结束定义命名空间 at::native
} // namespace at::native
```