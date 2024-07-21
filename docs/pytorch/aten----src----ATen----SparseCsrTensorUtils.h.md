# `.\pytorch\aten\src\ATen\SparseCsrTensorUtils.h`

```
#pragma once

#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#else
#include <ATen/ops/_sparse_compressed_tensor_unsafe.h>
#include <ATen/ops/resize_as_sparse_native.h>
#endif

// 定义一个宏，根据给定的布局类型调度执行对应的操作
#define AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(LAYOUT, NAME, ...) \
  [&] {                                                              \
    const auto& the_layout = LAYOUT;                                 \
    switch (the_layout) {                                            \
      case kSparseCsr:                                               \
      case kSparseCsc:                                               \
      case kSparseBsr:                                               \
      case kSparseBsc:                                               \
        return __VA_ARGS__();                                        \
      default:                                                       \
        // 如果布局类型不是稀疏压缩格式，则抛出错误信息
        AT_ERROR(                                                    \
            NAME,                                                    \
            " expected sparse compressed tensor layout but got ",    \
            the_layout);                                             \
    }                                                                \
  }()

// 定义一个宏，根据给定的布局类型调度执行对应的行稀疏压缩操作
#define AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(                \
    LAYOUT, NAME, ROW_DIM_ACTION, COLUMN_DIM_ACTION)              \
  [&]() {                                                         \
    const auto& the_layout = LAYOUT;                              \
    switch (the_layout) {                                         \
      case kSparseCsr:                                            \
      case kSparseBsr:                                            \
        return (ROW_DIM_ACTION)();                                \
      case kSparseCsc:                                            \
      case kSparseBsc:                                            \
        return (COLUMN_DIM_ACTION)();                             \
      default:                                                    \
        // 如果布局类型不是行稀疏压缩格式，则抛出错误信息
        AT_ERROR(                                                 \
            NAME,                                                 \
            " expected sparse compressed tensor layout but got ", \
            the_layout);                                          \
    }                                                             \
  }()

// 定义一个宏，根据给定的布局类型调度执行对应的普通稀疏压缩格式操作
#define AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(              \
    LAYOUT, NAME, NO_BLOCK_ACTION, BLOCK_ACTION)                  \
  [&]() {                                                         \
    const auto& the_layout = LAYOUT;                              \
    switch (the_layout) {                                         \
      // 根据给定的稀疏张量布局类型进行不同的处理
      case kSparseCsr:                                            \
      case kSparseCsc:                                            \
        // 对于 CSR 和 CSC 布局，返回空块操作函数 NO_BLOCK_ACTION
        return (NO_BLOCK_ACTION)();                               \
      case kSparseBsr:                                            \
      case kSparseBsc:                                            \
        // 对于 BSR 和 BSC 布局，返回块操作函数 BLOCK_ACTION
        return (BLOCK_ACTION)();                                  \
      default:                                                    \
        // 对于其他未知的布局类型，抛出错误并指明期望的布局类型和实际得到的布局类型
        AT_ERROR(                                                 \
            NAME,                                                 \
            " expected sparse compressed tensor layout but got ", \
            the_layout);                                          \
    }                                                             \
  }()
#define AT_DISPATCH_SPARSE_ROW_COMPRESSED_LAYOUTS(                    \
    LAYOUT, NAME, ROW_DIM_ACTION)                                     \
  [&]() {                                                             \  // 定义一个宏，用于根据稀疏行压缩布局调度操作
    const auto& the_layout = LAYOUT;                                  \  // 获取传入的布局参数
    switch (the_layout) {                                             \  // 开始根据布局参数进行选择
      case kSparseCsr:                                                \  // 如果布局是稀疏CSR格式
      case kSparseBsr:                                                \  // 或者稀疏BSR格式
        return (ROW_DIM_ACTION)();                                    \  // 执行行维度操作并返回结果
      default:                                                        \  // 如果布局不符合预期
        AT_ERROR(                                                     \  // 抛出错误，指示期望的稀疏行压缩张量布局
            NAME,                                                     \  // 使用传入的名称
            " expected sparse row compressed tensor layout but got ", \  // 提示得到的实际布局
            the_layout);                                              \  // 输出实际的布局
    }                                                                 \
  }()

#define AT_DISPATCH_SPARSE_COL_COMPRESSED_LAYOUTS(                       \
    LAYOUT, NAME, COL_DIM_ACTION)                                        \
  [&]() {                                                                \  // 定义一个宏，用于根据稀疏列压缩布局调度操作
    const auto& the_layout = LAYOUT;                                     \  // 获取传入的布局参数
    switch (the_layout) {                                                \  // 开始根据布局参数进行选择
      case kSparseCsc:                                                   \  // 如果布局是稀疏CSC格式
      case kSparseBsc:                                                   \  // 或者稀疏BSC格式
        return (COL_DIM_ACTION)();                                       \  // 执行列维度操作并返回结果
      default:                                                           \  // 如果布局不符合预期
        AT_ERROR(                                                        \  // 抛出错误，指示期望的稀疏列压缩张量布局
            NAME,                                                        \  // 使用传入的名称
            " expected sparse column compressed tensor layout but got ", \  // 提示得到的实际布局
            the_layout);                                                 \  // 输出实际的布局
    }                                                                    \
  }()

#define AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(LAYOUT, NAME, ACTION)  \
  [&]() {                                                                     \  // 定义一个宏，用于根据非块压缩布局调度操作
    const auto& the_layout = LAYOUT;                                          \  // 获取传入的布局参数
    switch (the_layout) {                                                     \
      // 开始对布局类型进行判断
      case kSparseCsr:                                                        \
      case kSparseCsc:                                                        \
        // 如果是稀疏矩阵的CSR或CSC布局，则执行空操作（返回一个空的操作）
        return (ACTION)();                                                    \
      default:                                                                \
        // 如果不是上述两种稀疏布局类型，则抛出错误，指明期望的布局类型
        AT_ERROR(                                                             \
            NAME,                                                             \
            " expected sparse compressed (non-block) tensor layout but got ", \
            the_layout);                                                      \
    }                                                                         \
  }()
#define AT_DISPATCH_SPARSE_COMPRESSED_BLOCK_LAYOUTS(LAYOUT, NAME, ACTION) \
  [&]() {                                                                 \
    // 获取传入的布局参数 LAYOUT 的引用
    const auto& the_layout = LAYOUT;                                      \
    // 根据布局类型进行分支判断
    switch (the_layout) {                                                 \
      // 如果是稀疏块压缩格式 kSparseBsr 或 kSparseBsc，则执行 ACTION 函数对象
      case kSparseBsr:                                                    \
      case kSparseBsc:                                                    \
        return (ACTION)();                                                \
      // 对于其他未知的布局类型，抛出错误
      default:                                                            \
        AT_ERROR(                                                         \
            NAME,                                                         \
            " expected sparse compressed block tensor layout but got ",   \
            the_layout);                                                  \
    }                                                                     \
  }()

#define AT_DISPATCH_SPARSE_VALUE_TYPES(TYPE, NAME, ...) \
  // 调用 AT_DISPATCH_SWITCH 宏，处理类型 TYPE 的分发
  AT_DISPATCH_SWITCH(                                   \
      TYPE,                                             \
      NAME,                                             \
      // 使用 AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4 处理各种类型和复杂类型
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4(      \
          kComplexHalf, kHalf, kBool, kBFloat16, __VA_ARGS__))

namespace at::sparse_csr {

using SparseCsrTensor = Tensor;

// 检查给定的布局是否属于稀疏压缩格式之一
inline bool is_sparse_compressed(const Layout& layout) {
  switch (layout) {
    // 如果是 kSparseCsr、kSparseCsc、kSparseBsr 或 kSparseBsc 中的一种，返回 true
    case kSparseCsr:
    case kSparseCsc:
    case kSparseBsr:
    case kSparseBsc:
      return true;
    // 对于其他布局类型，返回 false
    default:;
  }
  return false;
}

// 检查给定的 Tensor 是否使用稀疏压缩格式
inline bool is_sparse_compressed(const Tensor& self) {
  return is_sparse_compressed(self.layout());
}

// 获取稀疏 CSR Tensor 的实现指针
inline SparseCsrTensorImpl* get_sparse_csr_impl(const SparseCsrTensor& self) {
  // 使用 AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS 宏处理 self 的布局
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(), "get_sparse_csr_impl", [&] {});
  // 返回类型转换后的 CSR Tensor 实现指针
  return static_cast<SparseCsrTensorImpl*>(self.unsafeGetTensorImpl());
}

// 将布局枚举转换为字符串表示
inline std::string layoutToString(
    Layout layout,
    bool upper = false,
    bool lower = false) {
  switch (layout) {
    // 根据布局类型返回相应的字符串表示
    case kSparseCsr:
      return (upper ? "CSR" : (lower ? "csr" : "Csr"));
    case kSparseCsc:
      return (upper ? "CSC" : (lower ? "csc" : "Csc"));
    case kSparseBsr:
      return (upper ? "BSR" : (lower ? "bsr" : "Bsr"));
    case kSparseBsc:
      return (upper ? "BSC" : (lower ? "bsc" : "Bsc"));
    // 对于未知布局类型，抛出错误
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return "";
  }
}

// 检查布局是否为压缩行格式
inline bool isCompressedRow(Layout layout) {
  // 使用 AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS 宏处理行压缩布局
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout, "isCompressedRow", [&] { return true; }, [&] { return false; });
}

// 检查布局是否为压缩列格式
inline bool isCompressedColumn(Layout layout) {
  // 使用 AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS 宏处理列压缩布局
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout,
      "isCompressedColumn",
      [&] { return false; },
      [&] { return true; });
}
// 根据布局类型生成对应的压缩指标名称
inline std::string compressedIndicesName(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout,
      "compressedIndicesName",
      [&] { return "crow_indices"; },  // 如果布局是行压缩，则返回 "crow_indices"
      [&] { return "ccol_indices"; });  // 如果布局是列压缩，则返回 "ccol_indices"
}

// 根据布局类型生成对应的非压缩指标名称
inline std::string plainIndicesName(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout,
      "plainIndicesName",
      [&] { return "col_indices"; },   // 如果布局是行压缩，则返回 "col_indices"
      [&] { return "row_indices"; });  // 如果布局是列压缩，则返回 "row_indices"
}

// 根据布局类型返回对应的压缩维度名称
inline std::string compressedDimName(Layout layout) {
  switch (layout) {
    case kSparseCsr:
      return "row";           // 对于行压缩，返回 "row"
    case kSparseCsc:
      return "column";        // 对于列压缩，返回 "column"
    case kSparseBsr:
      return "row block";     // 对于行块压缩，返回 "row block"
    case kSparseBsc:
      return "column block";  // 对于列块压缩，返回 "column block"
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return "";
  }
}

// 根据布局类型返回对应的非压缩维度名称
inline std::string plainDimName(Layout layout) {
  switch (layout) {
    case kSparseCsr:
      return "column";        // 对于行压缩，返回 "column"
    case kSparseCsc:
      return "row";           // 对于列压缩，返回 "row"
    case kSparseBsr:
      return "column block";  // 对于行块压缩，返回 "column block"
    case kSparseBsc:
      return "row block";     // 对于列块压缩，返回 "row block"
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return "";
  }
}

// 根据布局类型和张量尺寸返回行维度的大小
inline size_t rowDimension(Layout layout, IntArrayRef size) {
  return size.size() - (isCompressedRow(layout) ? 2 : 1);  // 如果是行压缩，则减去2，否则减去1
}

// 根据布局类型和张量尺寸返回列维度的大小
inline size_t columnDimension(Layout layout, IntArrayRef size) {
  return size.size() - (isCompressedColumn(layout) ? 2 : 1);  // 如果是列压缩，则减去2，否则减去1
}

// 根据布局类型、张量尺寸和密集维度数返回压缩维度的大小
inline size_t compressedDimension(
    Layout layout,
    IntArrayRef size,
    size_t dense_ndim = 0) {
  return size.size() - dense_ndim - (isCompressedRow(layout) ? 2 : 1);  // 如果是行压缩，则减去2，否则减去1
}

// 根据布局类型、张量尺寸和密集维度数返回非压缩维度的大小
inline size_t plainDimension(
    Layout layout,
    IntArrayRef size,
    size_t dense_ndim = 0) {
  return size.size() - dense_ndim - (isCompressedRow(layout) ? 1 : 2);  // 如果是行压缩，则减去1，否则减去2
}

// 返回张量的批量维度数目
inline int64_t numBatchDimensions(Tensor const& self) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(),
      "numBatchDimensions",
      [&self] { return self.crow_indices().dim() - 1; },   // 如果是行压缩，则返回 crow_indices 的维度减1
      [&self] { return self.ccol_indices().dim() - 1; });  // 如果是列压缩，则返回 ccol_indices 的维度减1
}

// 返回压缩和非压缩的索引对
inline std::pair<Tensor, Tensor> getCompressedPlainIndices(Tensor const& self) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(),
      "getCompressedPlainIndices",
      [&self] {
        return std::make_pair(self.crow_indices(), self.col_indices());  // 如果是行压缩，则返回 crow_indices 和 col_indices
      },
      [&self] {
        return std::make_pair(self.ccol_indices(), self.row_indices());  // 如果是列压缩，则返回 ccol_indices 和 row_indices
      });
}

// 返回索引的数据类型
inline ScalarType getIndexDtype(Tensor const& self) {
  switch (self.layout()) {
    case kSparseCsr:
    case kSparseBsr:
      return self.crow_indices().scalar_type();  // 如果是行压缩，则返回 crow_indices 的数据类型
    case kSparseCsc:
    case kSparseBsc:
      return self.ccol_indices().scalar_type();  // 如果是列压缩，则返回 ccol_indices 的数据类型
    case kSparse:
      return self._indices().scalar_type();      // 如果是一般稀疏张量，则返回 _indices 的数据类型
    default:
      return ScalarType::Long;                   // 默认返回 Long 类型
  }
}

// 翻转压缩布局类型
inline Layout flip_compressed_layout(Layout layout) {
  switch (layout) {
    case kSparseCsr:
      return kSparseCsc;    // 如果是行压缩，则返回列压缩
    case kSparseCsc:
      return kSparseCsr;    // 如果是列压缩，则返回行压缩
    # 如果输入参数 layout 为 kSparseBsr，则返回 kSparseBsc
    case kSparseBsr:
      return kSparseBsc;
    # 如果输入参数 layout 为 kSparseBsc，则返回 kSparseBsr
    case kSparseBsc:
      return kSparseBsr;
    # 如果输入参数 layout 不是 kSparseBsr 或 kSparseBsc，则抛出错误信息并返回 kSparseCsr
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return kSparseCsr;
    }
}

// 获取张量块大小的内联函数
inline DimVector getBlockSize(Tensor const& self) {
  // 计算批次维度数量
  int64_t n_batch = numBatchDimensions(self);
  // 返回从索引 n_batch + 1 到 2 的张量大小作为 DimVector
  return at::DimVector(self.values().sizes().slice(n_batch + 1, 2));
}

// 获取对称整数块大小的内联函数
inline at::OptionalArray<at::SymInt> getSymIntBlockSize(Tensor const& self) {
  // 如果布局是稀疏的 Bsr 或 Bsc
  if (self.layout() == at::kSparseBsr || self.layout() == at::kSparseBsc) {
    // 计算批次维度数量
    int64_t n_batch = numBatchDimensions(self);
    // 返回从索引 n_batch + 1 到 2 的对称大小作为 OptionalArray
    return self.values().sym_sizes().slice(n_batch + 1, 2).vec();
  } else {
    // 否则返回空的 OptionalArray
    return {};
  }
}

// 仅处理稀疏压缩二元操作的简单情况的模板函数
template <typename binary_op_t, typename binary_op_out_t>
inline bool only_sparse_compressed_binary_op_trivial_cases(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out,
    const binary_op_t& binary_op,
    const binary_op_out_t& binary_op_out) {
  // 断言 self、other 和 out 都是稀疏压缩的
  TORCH_INTERNAL_ASSERT(at::sparse_csr::is_sparse_compressed(self));
  TORCH_INTERNAL_ASSERT(at::sparse_csr::is_sparse_compressed(other));
  TORCH_INTERNAL_ASSERT(at::sparse_csr::is_sparse_compressed(out));

  // 如果 self 和 other 相同，则直接进行二元操作并返回 true
  if (self.is_same(out) && self.is_same(other)) {
    binary_op_out(self.values(), other.values(), alpha);
    return true;
  }
  // 如果 self 和 other 不同但 self 与 out 相同
  if (self.is_same(other)) {
    // 获取压缩和普通索引，设置 out 的成员张量并返回 true
    auto [compressed_indices, plain_indices] =
        at::sparse_csr::getCompressedPlainIndices(self);
    static_cast<SparseCsrTensorImpl*>(out.unsafeGetTensorImpl())
        ->set_member_tensors(
            compressed_indices,
            plain_indices,
            binary_op(self.values(), other.values(), alpha),
            self.sizes());
    return true;
  }
  // 其他情况返回 false
  return false;
}

// 仅处理稀疏压缩加法操作的简单情况的内联函数
inline bool only_sparse_compressed_add_trivial_cases(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  // 调用通用的稀疏压缩二元操作处理函数
  return only_sparse_compressed_binary_op_trivial_cases(
      self,
      other,
      alpha,
      out,
      // 定义加法的二元操作
      [](const Tensor& v1, const Tensor& v2, const Scalar& alpha) {
        return v1.add(v2, alpha);
      },
      // 定义加法的就地操作
      [](const Tensor& v1, const Tensor& v2, const Scalar& alpha) {
        return v1.add_(v2, alpha);
      });
}

// 转换张量类型的函数
inline Tensor to_type(const Tensor& input, ScalarType dtype) {
  // 获取压缩和普通索引
  auto [compressed_indices, plain_indices] =
      at::sparse_csr::getCompressedPlainIndices(input);
  // 返回具有新数据类型的稀疏压缩张量
  return at::_sparse_compressed_tensor_unsafe(
      compressed_indices,
      plain_indices,
      std::move(input.values()).to(dtype),
      input.sizes(),
      dtype,
      input.layout(),
      input.device(),
      input.options().pinned_memory_opt());
}

// 创建累加缓冲区的函数模板
template <typename acc_t, typename scalar_t>
inline std::tuple<Tensor, Tensor> create_acc_buffer(
    TensorOptions option,
    ScalarType type,
    int64_t nnz = -1) {
  // 新值张量和新值累加张量
  Tensor new_values, new_values_acc;
  constexpr bool need_acc = !std::is_same_v<scalar_t, acc_t>;
  // 是否是整数类型
  bool is_integral = at::isIntegralType(type, /*includeBool=*/true);
  // 如果需要累加张量
  if constexpr (need_acc) {
    // 获取对应的累加数据类型
    auto acc_dtype = CppTypeToScalarType<acc_t>::value;
    # 初始化一个空的张量 `new_values_acc`，数据类型为 `acc_dtype`
    new_values_acc = at::empty({}, option.dtype(acc_dtype));
    
    # 如果 `is_integral` 为真，则 `new_values` 为 `new_values_acc` 的引用；否则初始化一个空的张量 `new_values`，数据类型由 `option` 决定
    new_values = is_integral ? new_values_acc : at::empty({}, option);
    
    # 如果前面的条件不满足，则 `new_values` 和 `new_values_acc` 均初始化为空的张量，数据类型由 `option` 决定
    } else {
        new_values = new_values_acc = at::empty({}, option);
    }
    
    # 如果 `nnz` 不等于 -1，则调整 `new_values` 和 `new_values_acc` 的大小为 `nnz`，然后返回作为元组
    if (nnz != -1) {
        return std::make_tuple(
            new_values.resize_(nnz), new_values_acc.resize_(nnz));
    # 否则直接返回 `new_values` 和 `new_values_acc` 作为元组
    } else {
        return std::make_tuple(new_values, new_values_acc);
    }
} // 结束当前函数的实现

// 定义一个内联函数 `copy_from_acc_buffer`，接受两个参数 `new_values` 和 `new_values_acc`
inline void copy_from_acc_buffer(Tensor& new_values, Tensor& new_values_acc) {
    // 检查 `new_values_acc` 是否和 `new_values` 指向相同的对象
    if (!new_values_acc.is_same(new_values)) {
        // 如果不相同，则将 `new_values_acc` 的内容复制给 `new_values`
        new_values.copy_(new_values_acc);
    }
}

} // 结束命名空间 `at::sparse_csr`
```