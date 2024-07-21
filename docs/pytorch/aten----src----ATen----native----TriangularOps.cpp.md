# `.\pytorch\aten\src\ATen\native\TriangularOps.cpp`

```
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS，用于限制仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的核心 Tensor 类和相关头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/TriangularOpsUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <c10/util/irange.h>

// 根据编译选项决定是否包含整体操作函数头文件或者逐个操作函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/trace_backward_native.h>
#include <ATen/ops/tril_native.h>
#include <ATen/ops/triu_native.h>
#include <ATen/ops/zeros.h>
#endif

// ATen 命名空间开始
namespace at::meta {

// 定义 TORCH_META_FUNC 宏实现 tril 函数的元信息处理
TORCH_META_FUNC(tril)(const Tensor& self, int64_t k) {
  // 检查输入张量 self 至少有两个维度
  TORCH_CHECK(self.dim() >= 2, "tril: input tensor must have at least 2 dimensions")
  // 设置输出张量的原始步长，未指定额外的步长
  set_output_raw_strided(0, self.sizes(), {}, self.options());
}

// 定义 TORCH_META_FUNC 宏实现 triu 函数的元信息处理
TORCH_META_FUNC(triu)(const Tensor& self, int64_t k) {
  // 检查输入张量 self 至少有两个维度
  TORCH_CHECK(self.dim() >= 2, "triu: input tensor must have at least 2 dimensions")
  // 设置输出张量的原始步长，未指定额外的步长
  set_output_raw_strided(0, self.sizes(), {}, self.options());
}

}  // namespace at::meta

// ATen 命名空间内的 native 命名空间开始
namespace at::native {
namespace {

// 实现 triu/tril 操作的模板函数 apply_triu_tril_single
template <typename scalar_t>
void apply_triu_tril_single(
    scalar_t* result,                 // 结果数组的指针
    const scalar_t* self,             // 输入数组的指针
    bool inplace,                     // 是否原地操作的标志
    int64_t k,                        // 对角线偏移量
    int64_t n,                        // 行数
    int64_t m,                        // 列数
    int64_t res_row_stride,           // 结果数组行步长
    int64_t res_col_stride,           // 结果数组列步长
    int64_t self_row_stride,          // 输入数组行步长
    int64_t self_col_stride,          // 输入数组列步长
    bool upper) {                     // 是否是上三角操作的标志

  // 常量定义，零的整数偏移量
  constexpr int64_t zero = 0;

  // 根据是否是上三角操作选择并行处理方式
  if (upper) {
    // 并行处理每一行 i
    parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
      for (int64_t i : c10::irange(start, end)) {
        // 对每行 i 进行操作，将对角线以下的元素置为零
        for (int64_t j = 0; j < std::min(m, i + k); j++) {
          result[i * res_row_stride + j * res_col_stride] = static_cast<scalar_t>(0);
        }
        // 如果不是原地操作，则复制剩余部分的输入数组到结果数组中
        if (!inplace) {
          for (int64_t j = std::max(zero, i + k); j < m; j++) {
            result[i * res_row_stride + j * res_col_stride] = self[i * self_row_stride + j * self_col_stride];
          }
        }
      }
    });
  } else {
    // 并行处理每一行 i
    parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
      for (int64_t i : c10::irange(start, end)) {
        // 对每行 i 进行操作，将对角线以上的元素置为零
        for (int64_t j = std::max(zero, i + k + 1); j < m; j++) {
          result[i * res_row_stride + j * res_col_stride] = static_cast<scalar_t>(0);
        }
        // 如果不是原地操作，则复制剩余部分的输入数组到结果数组中
        if (!inplace) {
          for (int64_t j = zero; j < std::min(m, i + k + 1); j++) {
            result[i * res_row_stride + j * res_col_stride] = self[i * self_row_stride + j * self_col_stride];
          }
        }
      }
    });
  }
}

// ATen 内部命名空间结束
} // namespace
} // namespace at::native


这段代码是一些关于 ATen 库的 C++ 实现，主要涉及三角矩阵操作 triu 和 tril 的并行实现，以及一些相关的头文件和宏定义。
void apply_triu_tril(const Tensor& result, const Tensor& self, bool inplace, int64_t k, bool upper) {
  // 获取输入张量 self 的最后两个维度的大小
  auto n = self.size(-2);
  auto m = self.size(-1);
  // 获取 self 的数据指针和相关步长
  auto self_data = self.const_data_ptr<scalar_t>();
  auto self_stride = (self.dim() > 2 && self.stride(-3) > 0) ? self.stride(-3) : 1;
  // 计算处理的批次数目
  auto batchsize = batchCountTrilTriu(result);
  // 获取 self 的行步长和列步长
  auto self_row_stride = self.stride(-2);
  auto self_col_stride = self.stride(-1);

  // 获取结果张量 result 的数据指针
  auto result_data = result.data_ptr<scalar_t>();
  // 检查是否需要分配结果张量的步长
  int64_t result_stride, result_row_stride, result_col_stride;
  if (result_data != self_data) {
    // 如果 result 和 self 不是同一个张量，则获取 result 的步长
    result_stride = (result.dim() > 2 && result.stride(-3) > 0) ? result.stride(-3) : 1;
    result_row_stride = result.stride(-2);
    result_col_stride = result.stride(-1);
  } else {
    // 如果 result 和 self 是同一个张量，则使用 self 的步长
    result_stride = self_stride;
    result_row_stride = self_row_stride;
    result_col_stride = self_col_stride;
  }

  // 并行处理每个批次的数据
  parallel_for(0, batchsize, 0, [&](int64_t start, int64_t end) {
    for (const auto b : c10::irange(start, end)) {
      // 指向当前批次在 self 和 result 中的数据指针
      const scalar_t* self_batch = &self_data[b * self_stride];
      scalar_t* result_batch = &result_data[b * result_stride];
      // 调用单个批次的三角矩阵操作函数
      apply_triu_tril_single<scalar_t>(
          result_batch,
          self_batch,
          inplace,
          k,
          n,
          m,
          result_row_stride,
          result_col_stride,
          self_row_stride,
          self_col_stride,
          upper);
    }
  });
}

struct UpperTriangle {
  static constexpr const char* op_name = "triu";
  static constexpr bool upper = true;
};

struct LowerTriangle {
  static constexpr const char *op_name = "tril";
  static constexpr bool upper = false;
};

template <typename Triangle>
void compute_triu_tril(const Tensor& self, int64_t k, const Tensor &result) {
  // 如果 self 张量为空，则直接返回
  if (self.numel() == 0) {
    return;
  }

  // 检查是否是原地操作
  bool inplace_op = self.is_same(result);

  bool inplace_update = false;
  Tensor self_c;
  // 检查是否需要在批处理上进行原地操作
  std::tie(inplace_update, self_c) = checkTrilTriuBatchContiguous(self, inplace_op);

  Tensor result_c;
  // 如果是原地操作且不是原地更新，则创建一个与 result 类型和形状相同的新张量
  if (inplace_op && !inplace_update) {
    result_c = at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
    result_c = result;
  }

  // 根据三角形类型的不同调用不同的三角形计算函数
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      ScalarType::ComplexHalf,
      ScalarType::BFloat16,
      ScalarType::Half,
      ScalarType::Bool,
      self.scalar_type(),
      Triangle::op_name,
      [&]{
        apply_triu_tril<scalar_t>(result_c, self_c, inplace_op && inplace_update, k, Triangle::upper);
      });

  // 如果是原地操作且不是原地更新，则将计算结果拷贝回 result 中
  if (inplace_op && !inplace_update) {
    result.copy_(result_c);
  }
}

}  // namespace

// 实现 tril_cpu 函数，调用 compute_triu_tril 用于计算下三角矩阵
TORCH_IMPL_FUNC(tril_cpu)(const Tensor& self, int64_t k, const Tensor &result) {
  compute_triu_tril<LowerTriangle>(self, k, result);
}

// 实现 triu_cpu 函数，调用 compute_triu_tril 用于计算上三角矩阵
TORCH_IMPL_FUNC(triu_cpu)(const Tensor& self, int64_t k, const Tensor &result) {
  compute_triu_tril<UpperTriangle>(self, k, result);
}

// 计算对称整数数组的迹的反向传播
Tensor trace_backward_symint(const Tensor& grad, c10::SymIntArrayRef sizes) {
  // 如果 sizes 的维度不为 2，则直接返回
  if (sizes.size() != 2) {
    throw std::runtime_error("expected matrix input");


// 抛出运行时错误，指示期望的矩阵输入未提供
throw std::runtime_error("expected matrix input");



  auto grad_input = at::zeros_symint(sizes[0] * sizes[1], grad.options());


// 创建一个大小为 sizes[0] * sizes[1] 的零张量 grad_input，使用与 grad 张量相同的选项
auto grad_input = at::zeros_symint(sizes[0] * sizes[1], grad.options());



  auto indices = at::arange(0, grad_input.numel(), sizes[1] + 1, grad.options().dtype(at::kLong));


// 生成一个从 0 开始、步长为 sizes[1] + 1、元素数量为 grad_input.numel() 的索引张量 indices，
// 使用与 grad 张量相同的长整型数据类型选项
auto indices = at::arange(0, grad_input.numel(), sizes[1] + 1, grad.options().dtype(at::kLong));



  // for composite compliance, use out-of-place variant of
  // `index_fill` if grad tensor is a Tensor Subclass.


// 如果 grad 张量是 Tensor Subclass，则为了复合一致性，使用 index_fill 的无地方变体。
// Tensor Subclass 是指 grad 张量的子类。



  if (isTensorSubclassLike(grad)) {
    grad_input = grad_input.index_fill(0, indices, grad);
  } else {
    grad_input.index_fill_(0, indices, grad);
  }


// 如果 grad 是 Tensor Subclass，则使用无地方变体 index_fill 对 grad_input 进行填充；
// 否则，直接在 grad_input 上进行 in-place 填充。
if (isTensorSubclassLike(grad)) {
  grad_input = grad_input.index_fill(0, indices, grad);
} else {
  grad_input.index_fill_(0, indices, grad);
}



  return grad_input.view_symint(sizes);


// 返回重塑后的 grad_input，使其形状符合 sizes 的对称整数视图
return grad_input.view_symint(sizes);
}

}  // namespace at::native
```