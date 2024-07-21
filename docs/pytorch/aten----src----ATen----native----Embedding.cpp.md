# `.\pytorch\aten\src\ATen\native\Embedding.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/BinaryOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/embedding_backward_native.h>
#include <ATen/ops/embedding_dense_backward.h>
#include <ATen/ops/embedding_dense_backward_native.h>
#include <ATen/ops/embedding_native.h>
#include <ATen/ops/embedding_renorm_native.h>
#include <ATen/ops/embedding_sparse_backward.h>
#include <ATen/ops/embedding_sparse_backward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#include <c10/util/irange.h>

#include <cstring>
#include <memory>
#include <utility>
#include <vector>


namespace at::native {

// 计算符号整数的嵌入操作，返回嵌入结果张量
Tensor embedding_symint(const Tensor & weight, const Tensor & indices,
                        c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
  TORCH_CHECK(weight.dim() == 2,  "'weight' must be 2-D");
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding", indices_arg, {kLong, kInt});

  // TODO: use tensor.index() after improving perf
  // 如果 indices 是 1 维张量，直接使用 index_select 进行选择
  if (indices.dim() == 1) {
    return weight.index_select(0, indices);
  }

  // 构造输出张量的大小
  auto size = indices.sym_sizes().vec();
  for (const auto& d : weight.sym_sizes().slice(1)) {
    size.push_back(d);
  }

  // 将权重张量按照 indices 重塑后进行选择，并根据 size 返回符号整数张量
  return weight.index_select(0, indices.reshape(-1)).view_symint(size);
}

// 计算符号整数嵌入的反向传播
Tensor embedding_backward_symint(
    const Tensor & grad, const Tensor & indices, c10::SymInt num_weights,
    c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
  if (sparse) {
    // 如果稀疏模式下，调用 embedding_sparse_backward 进行反向传播
    // TODO: if we teach sparse tensor how to propagate symints, the guard
    // here is not strictly necessary.  However, we think it is fine as is
    // because num weights is derived from a parameter and therefore
    // typically not varying.
    return at::embedding_sparse_backward(
      grad, indices,
      num_weights.guard_int(__FILE__, __LINE__),
      padding_idx.guard_int(__FILE__, __LINE__),
      scale_grad_by_freq);
  } else {
    // 否则调用 embedding_dense_backward_symint 进行反向传播
    return at::embedding_dense_backward_symint(
      grad, indices, std::move(num_weights), padding_idx, scale_grad_by_freq);
  }
}

// 计算稀疏嵌入的反向传播
Tensor embedding_sparse_backward(
    const Tensor & grad_, const Tensor & indices_, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq) {

  auto indices_arg = TensorArg(indices_, "indices", 2);
  checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});

  // TODO: implement scale_grad_by_freq
  // 暂时不支持 scale_grad_by_freq 参数
  if (scale_grad_by_freq) {
    AT_ERROR(
        "embedding_backward: scale_grad_by_freq not supported with sparse gradients");
  }

  // 复制张量 grad 和 indices
  Tensor indices = indices_;
  Tensor grad = grad_;
  // 如果 padding_idx 不为 -1，则进行额外处理
    // 创建一个包含可选张量的列表 c，这些张量的索引不等于 padding_idx
    c10::List<std::optional<Tensor>> c({indices != padding_idx});
    // 使用列表 c 对索引进行索引操作，获取新的索引张量
    indices = indices.index(c);
    // 使用列表 c 对梯度 grad 进行索引操作，获取新的梯度张量
    grad = grad.index(c);
  }

  // 计算梯度张量的最后一个维度的大小
  auto num_features = grad_.sym_size(-1);
  // 创建一个包含权重大小的数组
  auto weight_size = std::array<c10::SymInt, 2>{{ num_weights, num_features }};
  // 获取梯度张量的选项（设备、数据类型等）
  auto dense_options = grad.options();

  // 检查是否所有的梯度都来自 padding_idx
  if (grad.sym_numel() == 0) {
    // 如果梯度张量为空，则返回一个空的稀疏 COO 张量
    return at::_sparse_coo_tensor_unsafe_symint(at::empty({1, 0}, indices_.options().dtype(kLong)),
                                         at::empty_symint({c10::SymInt(0), std::move(num_features)}, dense_options),
                                         weight_size);
  }

  // 重塑索引张量为形状为 {1, -1} 的张量
  auto index = indices.reshape({1, -1});
  // 重塑梯度张量为形状为 {c10::SymInt(-1), num_features} 的符号整数张量
  auto values = grad.reshape_symint({c10::SymInt(-1), std::move(num_features)});
  // 返回一个不安全的符号整数稀疏 COO 张量
  return at::_sparse_coo_tensor_unsafe_symint(index.to(kLong), values, weight_size);
}

Tensor embedding_dense_backward_cpu(
    const Tensor & grad_, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq) {

  // 将 indices 参数包装成 TensorArg 对象，用于后续类型检查
  auto indices_arg = TensorArg(indices, "indices", 2);
  // 检查 indices 参数的标量类型，应为 kLong 或 kInt 类型
  checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});

  // 创建一个全零的梯度权重张量，形状为 (num_weights, grad_.size(-1))，使用和 grad_ 相同的选项
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());
  // 创建一个连续的 indices 张量副本
  auto indices_contig = indices.contiguous();
  // 计算 indices 张量中元素的总数
  int64_t numel = indices.numel();
  // 创建一个连续的梯度张量，形状为 (numel, grad_.size(-1))
  auto grad = grad_.contiguous().view({numel, grad_.size(-1)});

  // 配置 TensorIterator，用于在梯度权重上执行元素添加操作
  auto add_iter = TensorIteratorConfig()
    .add_output(grad_weight)           // 输出：grad_weight
    .add_input(grad_weight)            // 输入：grad_weight
    .add_const_input(grad)             // 输入：grad（常量）
    .resize_outputs(false)             // 不调整输出大小
    .declare_static_shape(grad.sizes(), /*squash_dims=*/0)  // 使用静态形状声明
    .build();

  // 获取梯度权重和梯度张量数据的指针及步长信息
  const auto gW_data = reinterpret_cast<char*>(grad_weight.data_ptr());
  const auto gO_data = reinterpret_cast<const char*>(grad.const_data_ptr());
  const auto gW_stride = grad_weight.strides()[0] * grad_weight.element_size();
  const auto gO_stride = grad.strides()[0] * grad.element_size();

  // 根据 indices 的标量类型分发执行 embedding_dense_backward_cpu 函数
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_dense_backward_cpu", [&] () {
    auto indices_data = indices_contig.const_data_ptr<index_t>();

    // 如果 scale_grad_by_freq 为真，则创建一个 counts 数组，用于统计每个索引出现的次数
    std::unique_ptr<index_t[]> counts;
    if (scale_grad_by_freq) {
      counts.reset(new index_t[num_weights]);
      for (const auto i : c10::irange(numel)) {
        counts[indices_data[i]] = 0;
      }
      for (const auto i : c10::irange(numel)) {
        counts[indices_data[i]]++;
      }
    }

    // 并行处理每个索引区间的梯度计算
    auto parallel_section = [&](index_t start, index_t end) {
      TensorIterator iter(add_iter);
      for (const auto i : c10::irange(numel)) {
        if (indices_data[i] != padding_idx) {
          index_t k = indices_data[i];
          if (k >= start && k < end) {
            double scale = 1.0;
            if (scale_grad_by_freq) {
              // 如果 scale_grad_by_freq 为真，则按索引 k 的出现次数缩放梯度
              scale /= counts[k];
            }

            // 用 scale 加权添加 grad[i] 到 grad_weight[k]
            iter.unsafe_replace_operand(0, gW_data + k * gW_stride);
            iter.unsafe_replace_operand(1, gW_data + k * gW_stride);
            iter.unsafe_replace_operand(2, const_cast<char*>(gO_data + i * gO_stride));
            // 调用 add_stub 执行加法操作
            add_stub(kCPU, iter, scale);
          }
        }
      }
    };

    // 使用 at::parallel_for 并行处理 num_weights 个区间，每个区间大小为 1000
    at::parallel_for(0, num_weights, 1000, parallel_section);

  });

  // 返回梯度权重张量
  return grad_weight;
}

Tensor & embedding_renorm_cpu_(
    // 获取传入的 self 引用，并创建自动类型检查对象 self_arg
    Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
  // 创建 indices 的自动类型检查对象 indices_arg
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 2);
  // 检查 self 的维度是否为 2
  checkDim("embedding_renorm_", self_arg, 2);
  // 检查 indices 的标量类型是否为 kLong 或 kInt
  checkScalarTypes("embedding_renorm_", indices_arg, {kLong, kInt});

  // 将 indices 转换为连续存储的 Tensor
  auto indices_contig = indices.contiguous();
  // 获取 indices 中元素的数量
  auto num_indices = indices.numel();

  // 根据 indices 的标量类型进行分发处理
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_renorm_cpu_", [&]() {
    // 获取 indices_contig 的常量数据指针
    auto data_ptr = indices_contig.const_data_ptr<index_t>();
    // 根据 data_ptr 创建排序后的 indices 向量
    auto sorted_indices = std::vector<index_t>(data_ptr, data_ptr + num_indices);
    // 对 sorted_indices 进行排序
    std::sort(sorted_indices.begin(), sorted_indices.end());

    // 由于在循环内部操作了 Tensor，因此不能使用 at::parallel_for。
    // 更多细节请参见 github.com/pytorch/pytorch/issues/28370。
    // 遍历排序后的 indices
    for (const auto i : c10::irange(num_indices)) {
      // 如果当前索引大于 0 并且与前一个索引相同，则跳过当前循环
      if (i > 0 && sorted_indices[i] == sorted_indices[i - 1]) {
        continue;
      }
      // 获取 self 中 sorted_indices[i] 对应的行数据
      auto row = self[sorted_indices[i]];
      // 计算该行数据的 norm，并转换为 double 类型
      auto norm = row.norm(norm_type).item<double>();
      // 如果 norm 超过 max_norm，则进行缩放操作
      if (norm > max_norm) {
        auto scale = max_norm / (norm + 1e-7);
        row *= scale;
      }
    }
  });

  // 返回经处理后的 self 引用
  return self;
}

}  // namespace at::native
```