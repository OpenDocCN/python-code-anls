# `.\pytorch\aten\src\ATen\FunctionalInverses.cpp`

```py
// 引入 ATen 库中的相关头文件和命名空间
#include <ATen/FunctionalInverses.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtilsMulti.h>

#include <utility>

// 定义了 at::functionalization 命名空间
namespace at::functionalization {

// 这段逻辑类似于 autograd 代码中用于视图反向调用的逻辑。
// 不能轻易地共享它，因为（最终）这些函数将全部调用 `permute/unsqueeze_copy()` 而不是 `permute/unsqueeze`。
// 实现了对张量进行排列的逆操作
static Tensor permute_inverse(const Tensor& self, IntArrayRef dims, InverseReturnMode inverse_return_mode) {
  // 反转排列顺序
  auto ndims = dims.size();
  std::vector<int64_t> dims_(ndims);
  for(const auto i : c10::irange(ndims)) {
    dims_[at::maybe_wrap_dim(dims[i], ndims)] = i;
  }
  // 根据需要选择是否返回视图
  if (inverse_return_mode != InverseReturnMode::NeverView) {
    return at::permute(self, dims_);
  } else {
    return at::permute_copy(self, dims_);
  }
}

// 实现了对张量进行插入新维度的复制操作
static Tensor unsqueeze_copy_to(const Tensor & self, c10::SymIntArrayRef sizes, InverseReturnMode inverse_return_mode) {
  auto result = self;
  bool need_alias = (inverse_return_mode == InverseReturnMode::AlwaysView);
  int64_t nDims = sizes.size();
  for(const auto dim : c10::irange(nDims)) {
    if (sizes[dim] == 1) {
      need_alias = false;
      // 根据需要选择是否返回视图
      if (inverse_return_mode != InverseReturnMode::NeverView) {
        result = at::unsqueeze(result, dim);
      } else {
        result = at::unsqueeze_copy(result, dim);
      }
    }
  }

  // 返回一个别名以确保输出在必要时是一个视图
  return need_alias ? at::alias(result) : result;
}

// 重载版本，允许指定维度列表进行插入新维度的复制操作
static Tensor unsqueeze_copy_to(const Tensor & self, IntArrayRef dim, c10::SymIntArrayRef sizes, InverseReturnMode inverse_return_mode) {
  const auto ndim = sizes.size();
  const auto mask = at::dim_list_to_bitset(dim, ndim);
  Tensor result = self;
  bool need_alias = (inverse_return_mode == InverseReturnMode::AlwaysView);
  // 如果张量维度为零，NumPy中不会报错但我们仍需避免反向操作中的插入新维度。
  if (ndim == 0) {
    // 返回一个别名以确保输出在必要时是一个视图
    return need_alias ? at::alias(result) : result;
  }

  // 遍历维度列表，根据条件进行插入新维度的复制操作
  for (const auto d : c10::irange(ndim)) {
    if (mask.test(d) && sizes[d] == 1) {
      need_alias = false;
      // 根据需要选择是否返回视图
      if (inverse_return_mode != InverseReturnMode::NeverView) {
        result = at::unsqueeze(result, d);
      } else {
        result = at::unsqueeze_copy(result, d);
      }
    }
  }

  // 返回一个别名以确保输出在必要时是一个视图
  return need_alias ? at::alias(result) : result;
}

// 注释 [Functionalization Pass: View Inverses]：
// 该文件包含每个“视图逆”的实现。
// 从数学意义上来说，这些并不是真正的逆操作：每个视图逆描述了如何撤销原始视图（尽管使用了不同的参数）。
//
// 例如，下面是一个删除了别名操作的程序，并展示了视图逆的作用：
// view1 = input1.view_op(args...)
// 创建一个新的视图 view1，通过对 input1 执行 view_op 操作并传递参数 args...

// view1.add_(1) (perform a mutation on the view, which should also modify input)
// 在 view1 上执行 add_ 操作，对视图进行突变，这也应该修改 input1

// version of the program with no aliasing, that instead uses view_inverse functions:
// 无别名的程序版本，使用 view_inverse 函数替代：
// view_copy1 = input1.view_copy_op(args...)
// 创建一个没有别名的视图 view_copy1，通过对 input1 执行 view_copy_op 操作并传递参数 args...
// view_copy1.add_(1) (perform a mutation on view_copy1. At this point, input1 is NOT modified)
// 在 view_copy1 上执行 add_ 操作，对视图进行突变。此时，input1 没有被修改。

// x = view_op_inverse(input1, view_copy1, args...)
// x = view_op_inverse(input1, view_copy1, args...)，使用 view_copy1 的逆操作来计算 x。

// at this point, input1 and x should be equal
// 此时，input1 和 x 应该是相等的。

// Note that input1 is also passed as an argument to view_op_inverse in the above example.
// 注意，在上述示例中，input1 也作为参数传递给 view_op_inverse。
// This isn't actually required for most view operators: it's only required for view ops
// 这实际上对于大多数视图操作并不需要：它仅在无法通过视图张量和参数确定基础张量的大小时才需要。
// Examples are slice/select/scatter/squeeze/as_strided.
// 例如 slice/select/scatter/squeeze/as_strided。
// We happen to be passing in the base tensor in all cases, mostly to make the codegen simpler.
// 我们偶尔会在所有情况下传递基础张量，主要是为了使代码生成更简单。
// But you'll see below that the "base" argument is ignored by most view_inverse implementations.
// 但是，下面你会看到，大多数 view_inverse 实现都会忽略 "base" 参数。

// ----------------------------------------------------------
// Implementations of each view_inverse() function are below.
// 下面是每个 view_inverse() 函数的实现。
// One of these needs to be implemented for every existing non-composite view operator.
// 每个现有的非复合视图操作符都需要实现其中之一。
// The codegen automatically generates the corresponding function declaration.
// 代码生成器会自动生成相应的函数声明。
// ----------------------------------------------------------

Tensor FunctionalInverses::_fw_primal_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t level) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _fw_primal() during the functionalization pass. For now, this is not supported.");
    return Tensor();
}

Tensor FunctionalInverses::_make_dual_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode, const at::Tensor& tangent, int64_t level) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _make_dual() during the functionalization pass. For now, this is not supported.");
    return Tensor();
}

Tensor FunctionalInverses::view_as_real_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::view_as_complex(mutated_view);
    } else {
      return at::view_as_complex_copy(mutated_view);
    }
}

Tensor FunctionalInverses::view_as_complex_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::view_as_real(mutated_view.resolve_conj());
    } else {
      return at::view_as_real_copy(mutated_view.resolve_conj());
    }
}

Tensor FunctionalInverses::_conj_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::_conj(mutated_view);
    } else {
      return at::_conj_copy(mutated_view);
    }
}
Tensor FunctionalInverses::_neg_view_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // 如果逆返回模式不是 NeverView
    if (inverse_return_mode != InverseReturnMode::NeverView) {
        // 返回 mutated_view 的负视图
        return at::_neg_view(mutated_view);
    } else {
        // 否则返回 mutated_view 的负视图的副本
        return at::_neg_view_copy(mutated_view);
    }
}

Tensor FunctionalInverses::as_strided_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::SymIntArrayRef size, at::SymIntArrayRef stride, std::optional<c10::SymInt> storage_offset) {
    // 如果逆返回模式是 AlwaysView
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
        // 注意：假设 mutated_view 是 base 的窄化视图。
        // 对于功能化，我们不应该这样做
        return mutated_view.as_strided_symint(
            base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
        // 否则调用 base 的 as_strided_scatter_symint 方法
        return base.as_strided_scatter_symint(mutated_view, size, stride, std::move(storage_offset));
    }
}

Tensor FunctionalInverses::diagonal_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t offset, int64_t dim1, int64_t dim2) {
    // 如果逆返回模式是 AlwaysView
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
        // 注意：假设 mutated_view 是 base 的窄化视图。
        // 对于功能化，我们不应该这样做
        return mutated_view.as_strided_symint(
            base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
        // 否则调用 base 的 diagonal_scatter 方法
        return base.diagonal_scatter(mutated_view, offset, dim1, dim2);
    }
}

Tensor FunctionalInverses::expand_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::SymIntArrayRef size, bool implicit) {
    // 如果逆返回模式是 AlwaysView
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
        // 注意：假设 mutated_view 是 base 的扩展视图。
        // 对于功能化，我们不应该这样做
        return mutated_view.as_strided_symint(
            base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
        // 否则执行 base + at::sum_to(...) 操作
        return base + at::sum_to(
            mutated_view - base,
            base.sym_sizes(),
            /*always_return_non_view=*/inverse_return_mode == InverseReturnMode::NeverView
        );
    }
}

Tensor FunctionalInverses::permute_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::IntArrayRef dims) {
    // 调用 functionalization 命名空间中的 permute_inverse 方法
    return at::functionalization::permute_inverse(mutated_view, dims, inverse_return_mode);
}

Tensor FunctionalInverses::_reshape_alias_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::SymIntArrayRef size, at::SymIntArrayRef stride) {
    // 注意：我直接调用 reshape()，忽略了步长（strides）。
    // _reshape_alias() 不是用户代码可用，而是 reshape() 的实现细节。
    // 直接传递步长可能在某些情况下会导致问题，例如：
    // b = a[0]; c = b.reshape(...); c.add_(1); print(a)
    // 如果我们在这里最终运行 _reshape_alias_inverse() 调用，如果同时传入大小和步长，
    // 由于 `mutated_view` 的存储空间不足，调用会失败。
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      // 如果不要求永远返回视图，则调用 _reshape_alias_symint()，传入 mutated_view、base 的符号化大小和步长
      return at::_reshape_alias_symint(mutated_view, base.sym_sizes(), base.sym_strides());
    } else {
      // 否则，调用 _reshape_alias_copy_symint()，传入 mutated_view、base 的符号化大小和步长
      return at::_reshape_alias_copy_symint(mutated_view, base.sym_sizes(), base.sym_strides());
    }
}

Tensor FunctionalInverses::select_int_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dim, c10::SymInt index) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      // 返回一个具有对称整数索引的窥视视图
      return mutated_view.as_strided_symint(
          base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
      // 对基本张量进行选择散布对称整数索引的操作
      return base.select_scatter_symint(mutated_view, dim, std::move(index));
    }
}

Tensor FunctionalInverses::detach_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // the functionalization pass doesn't care about autograd metadata - as a view, I think detach() is just an identity function
    // 分离视图张量，用于功能化过程
    return mutated_view;
}

Tensor FunctionalInverses::lift_fresh_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // 返回变异视图张量，用于功能化过程
    return mutated_view;
}

Tensor FunctionalInverses::slice_Tensor_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dim, std::optional<c10::SymInt> start, std::optional<c10::SymInt> end, c10::SymInt step) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      // 使用对称整数切片反转变异视图张量
      return mutated_view.slice_inverse_symint(
          base, dim, std::move(start), std::move(end), std::move(step));
    } else {
      // 使用对称整数切片散布基本张量
      return base.slice_scatter_symint(mutated_view, dim, std::move(start), std::move(end), std::move(step));
    }
}

Tensor FunctionalInverses::split_Tensor_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, c10::SymInt split_size, int64_t dim) {
    // It would be nice if this logic could be re-used from autograd's split_backward(), but I don't think it can.
    // For functionalization, we have only have one of the tensors from the TensorList outputed by split(), and we want to layer i
    // on top of the base tensor.
    // For autograd, we have all of the tensors outputted by split() and we just want to stack them.
    // 计算维度大小，并根据需求调整维度
    dim = at::maybe_wrap_dim(dim, base.dim());
    auto dim_size = base.sym_size(dim);
    auto start = split_size * mutated_view_idx;
    auto end = split_size + start;
    if (end > dim_size) end = dim_size;

    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      // 使用对称整数切片反转变异视图张量
      return mutated_view.slice_inverse_symint(base, dim, start, end, 1);
    } else {
      // 使用对称整数切片散布基本张量
      return base.slice_scatter_symint(mutated_view, dim, start, end, 1);
    }
}
Tensor FunctionalInverses::split_with_sizes_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, c10::SymIntArrayRef split_sizes, int64_t dim) {
    // 根据需要修正维度，确保在合理范围内
    dim = at::maybe_wrap_dim(dim, base.dim());
    // 获取指定维度的符号大小
    auto dim_size = base.sym_size(dim);
    // 计算起始索引，通过累加之前所有的分割大小
    c10::SymInt start = 0;
    for (auto i = 0; i < mutated_view_idx; ++i) {
        start += split_sizes[i];
    }
    // 计算结束索引，确保不超过维度大小
    auto end = start + split_sizes[mutated_view_idx];
    if (end > dim_size) end = dim_size;

    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
        // 注意：假定 mutated_view 是 base 的一个窄视图。
        // 在功能化过程中，我们不应该执行这个操作。
        return mutated_view.slice_inverse_symint(base, dim, start, end, 1);
    } else {
        // 返回通过 scatter 操作得到的张量
        return base.slice_scatter_symint(mutated_view, dim, start, end, 1);
    }
}

Tensor FunctionalInverses::squeeze_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // 将 mutated_view 按照 base 的符号大小扩展一个维度，并复制到新张量中返回
    return unsqueeze_copy_to(mutated_view, base.sym_sizes(), inverse_return_mode);
}

Tensor FunctionalInverses::squeeze_dim_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dim) {
    // 将 mutated_view 按照指定的维度 dim 扩展一个维度，并复制到新张量中返回
    return unsqueeze_copy_to(mutated_view, dim, base.sym_sizes(), inverse_return_mode);
}

Tensor FunctionalInverses::squeeze_dims_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, IntArrayRef dim) {
    // 将 mutated_view 按照指定的维度 dim 扩展多个维度，并复制到新张量中返回
    return unsqueeze_copy_to(mutated_view, dim, base.sym_sizes(), inverse_return_mode);
}

Tensor FunctionalInverses::t_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
        // 根据 inverse_return_mode 返回 mutated_view 的转置视图或副本
        return at::t(mutated_view);
    } else {
        return at::t_copy(mutated_view);
    }
}

Tensor FunctionalInverses::transpose_int_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dim0, int64_t dim1) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
        // 根据 inverse_return_mode 返回 mutated_view 的转置张量或其副本
        return transpose(mutated_view, dim0, dim1);
    } else {
        return transpose_copy(mutated_view, dim0, dim1);
    }
}

Tensor FunctionalInverses::_nested_view_from_buffer_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, const Tensor& nested_sizes, const Tensor& nested_strides, const Tensor& storage_offsets) {
    // 在功能化过程中，不支持嵌套张量，因此会抛出错误信息并返回空张量
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _nested_view_from_buffer() during the functionalization pass. For now, nested tensors aren't supported during functionalization");
    return Tensor();
}
// 从嵌套视图的逆操作中获取嵌套值的函数
Tensor FunctionalInverses::_nested_view_from_jagged_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, const Tensor& offsets, const Tensor& dummy, const std::optional<Tensor>& lengths, int64_t ragged_idx, const c10::optional<Tensor>& min_seqlen, const c10::optional<Tensor>& max_seqlen) {
  // 调用内部函数 _nested_get_values 从 mutated_view 中获取值
  auto values = at::_nested_get_values(mutated_view);
  // 如果不要求永远返回视图，则直接返回获取的值
  if (inverse_return_mode != InverseReturnMode::NeverView) {
    return values;
  } else {
    // 否则克隆获取的值，指定内存格式为连续存储
    return values.clone(/*memory_format=*/at::MemoryFormat::Contiguous);
  }
}

// 从嵌套视图的逆操作中获取值的逆操作
Tensor FunctionalInverses::_nested_get_values_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
  // 获取 base 的偏移量、长度、ragged_idx 和 dummy
  auto offsets = at::_nested_get_offsets(base);
  auto lengths = at::_nested_get_lengths(base);
  auto ragged_idx = at::_nested_get_ragged_idx(base);
  auto dummy = at::_nested_get_jagged_dummy(base);
  auto min_seqlen = at::_nested_get_min_seqlen(base);
  auto max_seqlen = at::_nested_get_max_seqlen(base);
  // 调用内部函数 _nested_view_from_jagged 获取变异视图的嵌套视图
  auto nt = at::_nested_view_from_jagged(
      mutated_view, offsets, dummy, lengths, ragged_idx,
      // 如果 min_seqlen 定义了，将其作为可选参数传递；否则传递空值
      (min_seqlen.defined() ? c10::optional<Tensor>(min_seqlen) : c10::nullopt),
      // 如果 max_seqlen 定义了，将其作为可选参数传递；否则传递空值
      (max_seqlen.defined() ? c10::optional<Tensor>(max_seqlen) : c10::nullopt));

  // 如果不要求永远返回视图，则直接返回获取的嵌套视图
  if (inverse_return_mode != InverseReturnMode::NeverView) {
    return nt;
  } else {
    // 否则克隆获取的嵌套视图，指定内存格式为连续存储
    return nt.clone(/*memory_format=*/at::MemoryFormat::Contiguous);
  }
}

// 对变异视图执行 unsqueeze 操作的逆操作
Tensor FunctionalInverses::unsqueeze_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dim) {
    // 如果不要求永远返回视图，则使用 squeeze 操作并指定维度 dim
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::squeeze(mutated_view, dim);
    } else {
      // 否则使用 squeeze_copy 操作并指定维度 dim
      return at::squeeze_copy(mutated_view, dim);
    }
}

// 对变异视图执行 _indices 操作的逆操作
Tensor FunctionalInverses::_indices_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // 抛出内部断言错误，表示在函数化过程中调用了 _indices()，不支持稀疏张量的函数化
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    // 返回空张量
    return Tensor();
}

// 对变异视图执行 _values 操作的逆操作
Tensor FunctionalInverses::_values_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // 抛出内部断言错误，表示在函数化过程中调用了 _values()，不支持稀疏张量的函数化
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _values() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    // 返回空张量
    return Tensor();
}

// 对变异视图执行 indices 操作的逆操作
Tensor FunctionalInverses::indices_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // 抛出内部断言错误，表示在函数化过程中调用了 indices()，不支持稀疏张量的函数化
    TORCH_INTERNAL_ASSERT(false, "Attempted to call indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    // 返回空张量
    return Tensor();
}

// 对变异视图执行 values 操作的逆操作
Tensor FunctionalInverses::values_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    # 调用 TORCH_INTERNAL_ASSERT 断言函数，传入 false 和错误消息字符串
    TORCH_INTERNAL_ASSERT(false, "Attempted to call values() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    # 返回一个空的 Tensor 对象
    return Tensor();
}

Tensor FunctionalInverses::_sparse_broadcast_to_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::IntArrayRef size) {
    // 在功能化过程中调用了 _sparse_broadcast_to()，抛出断言错误
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _sparse_broadcast_to() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    // 返回空的 Tensor 对象
    return Tensor();
}

Tensor FunctionalInverses::crow_indices_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // 在功能化过程中调用了 crow_indices()，抛出断言错误
    TORCH_INTERNAL_ASSERT(false, "Attempted to call crow_indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    // 返回空的 Tensor 对象
    return Tensor();
}

Tensor FunctionalInverses::col_indices_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // 在功能化过程中调用了 col_indices()，抛出断言错误
    TORCH_INTERNAL_ASSERT(false, "Attempted to call col_indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    // 返回空的 Tensor 对象
    return Tensor();
}

Tensor FunctionalInverses::ccol_indices_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // 在功能化过程中调用了 ccol_indices()，抛出断言错误
    TORCH_INTERNAL_ASSERT(false, "Attempted to call ccol_indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    // 返回空的 Tensor 对象
    return Tensor();
}

Tensor FunctionalInverses::row_indices_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // 在功能化过程中调用了 row_indices()，抛出断言错误
    TORCH_INTERNAL_ASSERT(false, "Attempted to call row_indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    // 返回空的 Tensor 对象
    return Tensor();
}

Tensor FunctionalInverses::unbind_int_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, int64_t dim) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // 注意：假设 mutated_view 是 base 的窄视图。
      // 在功能化过程中我们不应该这样做
      // 返回 mutated_view 的一个新的视图，使用 base 的符号尺寸和步幅
      return mutated_view.as_strided_symint(
          base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
      // 根据 dim 选择和 scatter 更新 base 的值
      dim = at::maybe_wrap_dim(dim, base.sizes().size());
      return base.select_scatter(mutated_view, dim, mutated_view_idx);
    }
}

Tensor FunctionalInverses::view_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::SymIntArrayRef size) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      // 返回 mutated_view 的符号整数视图，使用 base 的符号尺寸
      return mutated_view.view_symint(base.sym_sizes());
    } else {
      // 返回 mutated_view 的符号整数视图的副本，使用 base 的符号尺寸
      return at::view_copy_symint(mutated_view, base.sym_sizes());
    }
}


Tensor FunctionalInverses::view_dtype_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::ScalarType dtype) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      // 返回 mutated_view 的视图，使用 base 的标量类型
      return mutated_view.view(base.scalar_type());
    } else {
      # 如果条件不满足，则执行这个分支
      # 返回一个新的张量，复制自变异视图的数据，并保持与基本张量相同的数据类型
      return at::view_copy(mutated_view, base.scalar_type());
    }
}

Tensor FunctionalInverses::unfold_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dimension, int64_t size, int64_t step) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // 对于视图模式，假设mutated_view是base的窄视图。
      // 在这里不应该用于功能化。
      return mutated_view.as_strided_symint(
          base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
      // I think autograd and the functionalization pass want the exact same thing here, but need to test to confirm.
      // unfold_backward() is safe to use here because it is NOT a view op.
      // (note: technically, we'll have an extra memory copy.
      // We'd need to add an aliasing version of unfold_backward to fix that though).
      // 这里安全使用unfold_backward()，因为它不是视图操作。
      // （注意：在技术上，我们会有一个额外的内存复制。
      // 我们需要添加一个别名版本的unfold_backward来解决这个问题）。
      TORCH_CHECK(
        !(inverse_return_mode == InverseReturnMode::ViewOrScatterInverse && size > step),
        "While executing unfold, functionalization encountered a tensor being mutated that has internal overlap. \
When using torch.compile (or running functionalization directly), this is banned \
as the behavior is not well defined. Consider cloning the tensor before mutating it, \
or removing the mutation from your model."
          );
      // 当执行展开时，功能化遇到被改变的张量具有内部重叠的情况。
      // 在使用torch.compile（或直接运行功能化）时，这是被禁止的，因为行为定义不清楚。
      // 考虑在改变张量之前克隆它，或从模型中移除突变。
      return unfold_backward(mutated_view, base.sizes(), dimension, size, step);
    }
}

Tensor FunctionalInverses::alias_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      // 返回mutated_view的别名。
      return at::alias(mutated_view);
    } else {
      // 返回mutated_view的别名副本。
      return at::alias_copy(mutated_view);
    }
}

Tensor FunctionalInverses::chunk_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, int chunks, int dim) {
    // TODO: Can the logic from TensorShape.cpp be reused here somehow?
    // TODO: 是否可以在这里以某种方式重用TensorShape.cpp中的逻辑？
    const auto dim_size = base.sym_size(dim);
    auto split_size = (dim_size + chunks - 1) / chunks;
    std::vector<c10::SymInt> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
    // 使用给定的分块大小分割基本张量和变异视图的反函数。
    return split_with_sizes_inverse(base, mutated_view, inverse_return_mode, mutated_view_idx, split_sizes, dim);
}

Tensor FunctionalInverses::narrow_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, int dim, c10::SymInt start, c10::SymInt length) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // 对于视图模式，假设mutated_view是base的窄视图。
      // 在这里不应该用于功能化。
      return mutated_view.slice_inverse_symint(base, dim, std::move(start), start + length, 1);
    } else {
      // 使用scatter方法在基本张量和变异视图之间进行切片反函数。
      return base.slice_scatter_symint(
          mutated_view, dim, std::move(start), start + length, 1);
    }
}
// 定义一个名为 FunctionalInverses 的类的方法 slice_inverse_inverse，返回类型为 Tensor
// 方法接受参数：一个常量引用 base Tensor，一个常量引用 mutated_view Tensor，一个枚举类型参数 inverse_return_mode，一个常量引用 src Tensor，一个整型参数 dim，两个可选的 SymInt 类型参数 start 和 end，以及一个 SymInt 类型参数 step
Tensor FunctionalInverses::slice_inverse_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, const at::Tensor & src, int64_t dim, std::optional<c10::SymInt> start, std::optional<c10::SymInt> end, c10::SymInt step) {
    // 如果 inverse_return_mode 等于 InverseReturnMode::NeverView，则执行以下逻辑
    if (inverse_return_mode == InverseReturnMode::NeverView) {
      // 调用 at::slice_copy_symint 方法，将 mutated_view 在指定维度 dim 上的切片拷贝到新的 Tensor 中，并返回结果
      return at::slice_copy_symint(
          mutated_view, dim, std::move(start), std::move(end), std::move(step));
    } else {
      // 否则，调用 mutated_view 自身的 slice_symint 方法，返回在指定维度 dim 上的切片
      return mutated_view.slice_symint(
          dim, std::move(start), std::move(end), std::move(step));
    }
}

// 结束 at::functionalization 命名空间
} // namespace at::functionalization
```