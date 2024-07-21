# `.\pytorch\aten\src\ATen\native\UnfoldBackward.h`

```
// 预处理指令，指示编译器只包含当前头文件一次
#pragma once

// 包含 ATen 核心张量和迭代器相关的头文件
#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/NonEmptyUtils.h>

// 根据条件选择性地包含 ATen 函数或操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#endif

// 定义 ATen 的 native 命名空间
namespace at::native {

// 定义指针类型 unfold_backward_fn，表示一个函数指针，用于反折操作
using unfold_backward_fn = void (*)(
  Tensor& grad_in,
  const Tensor& grad,
  int64_t dim,
  int64_t size,
  int64_t step
);

// 声明 unfold_backward_stub，用于后向反折操作的分发
DECLARE_DISPATCH(unfold_backward_fn, unfold_backward_stub);

// 匿名命名空间，局部化定义以下的函数和变量

// 名称命名注意事项：命名方式不同寻常
// grad_in 并不表示相对于输入的梯度，
// grad_in/grad_out 只是 unfold_backward 内核的输入/输出。

static C10_UNUSED TensorIterator _make_unfold_backward_iter_over_grad_out(
  Tensor& grad_out,
  const Tensor& grad_in,
  int64_t dim,
  int64_t size,
  int64_t step
) {
  // 修正维度 dim，确保在 grad_out 的维度范围内
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // 最后一个维度存储折叠信息

  // 确保 grad_out 在维度 dim 上的尺寸不为空
  auto grad_out_dim_size = ensure_nonempty_size(grad_out, dim);
  // 确保 grad_in 在维度 dim 上的尺寸不为空
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);
  // 决定在维度 `dim` 上迭代的元素数
  auto iter_dim_size = std::min(
    grad_out_dim_size,
    (grad_in_dim_size - 1) * step + size
  );

  /* 准备 grad_out 用于 TensorIterator { */
  auto grad_out_strides = ensure_nonempty_vec(grad_out.strides().vec());
  auto grad_out_sizes = ensure_nonempty_vec(grad_out.sizes().vec());
  grad_out_sizes[dim] = iter_dim_size;
  auto grad_out_restrided = grad_out.as_strided(
    grad_out_sizes, grad_out_strides
  );
  /* } */

  /* 准备 grad_in 用于 TensorIterator { */
  auto grad_in_strides = ensure_nonempty_vec(grad_in.strides().vec());
  auto grad_in_sizes = ensure_nonempty_vec(grad_in.sizes().vec());

  // 设置维度 dim 的步长为 0
  // 尺寸为 1，因为该维度在内核内部索引
  grad_in_strides[dim] = 0;
  grad_in_sizes[dim] = 1;

  grad_in_strides.pop_back();
  grad_in_sizes.pop_back();

  auto grad_in_restrided = grad_in.squeeze(-1).as_strided(
    grad_in_sizes, grad_in_strides
  );
  /* } */

  // 在 TensorIterator 迭代期间，我们需要知道 grad_out[i_1,...,i_dim,...i_n] 中的 i_dim
  // idx_dim 存储这些信息
  /* 准备 idx_dim 用于 TensorIterator { */
  auto idx_dim = at::arange(
    0, iter_dim_size, grad_in.options().dtype(at::kLong)
  );

  auto grad_out_dim = ensure_nonempty_dim(grad_out.dim());

  auto idx_dim_strides = std::vector<int64_t>(grad_out_dim, 0);
  auto idx_dim_sizes = std::vector<int64_t>(grad_out_dim, 1);

  idx_dim_strides[dim] = 1;
  idx_dim_sizes[dim] = iter_dim_size;

  // idx_dim 尺寸将会广播，由 TensorIterator 中 grad_out 尺寸确定
  auto idx_dim_restrided = idx_dim.as_strided(idx_dim_sizes, idx_dim_strides);
  /* } */

  // 配置 TensorIteratorConfig 对象 iter
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // 设置不检查内存重叠
    .check_all_same_dtype(false)   // 设置不检查所有张量是否具有相同的数据类型
    .resize_outputs(false)         // 设置不调整输出尺寸
    .add_owned_output(grad_out_restrided)   // 添加 grad_out 作为输出张量
    .add_owned_const_input(grad_in_restrided)  // 添加 grad_in 作为常量输入张量
    .add_owned_input(idx_dim_restrided);       // 添加 idx_dim 作为输入张量
    .add_owned_const_input(idx_dim_restrided)
    .build();


# 添加一个拥有常量输入的操作，使用 idx_dim_restrided 作为输入
# 构建这个操作


  return iter;


# 返回构建完成的迭代器对象
}

}

} // namespace at::native


// 这些语句用于结束 at::native 命名空间的定义
```