# `.\pytorch\aten\src\ATen\native\LinearAlgebraUtils.h`

```py
#pragma once
// 防止头文件被多次包含

#include <c10/core/ScalarType.h>
// 引入标量类型相关的头文件
#include <c10/util/irange.h>
// 引入范围迭代器的头文件
#include <c10/util/Exception.h>
// 引入异常处理的头文件
#include <c10/util/strides.h>
// 引入步长计算相关的头文件
#include <ATen/core/Tensor.h>
// 引入张量核心功能的头文件
#include <ATen/ExpandUtils.h>
// 引入张量扩展工具的头文件
#include <ATen/TensorUtils.h>
// 引入张量工具的头文件
#include <ATen/native/TensorIterator.h>
// 引入张量迭代器的头文件
#include <ATen/native/TransposeType.h>
// 引入转置类型的头文件
#include <limits>
// 引入数值极限的头文件
#include <type_traits>
// 引入类型特性的头文件
#include <sstream>
// 引入字符串流的头文件
#include <cstring>
// 引入字符串处理的头文件
#include <cctype>
// 引入字符处理的头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 如果未定义 AT_PER_OPERATOR_HEADERS，则引入通用函数的头文件
#else
#include <ATen/ops/arange.h>
// 否则引入范围生成函数的头文件
#include <ATen/ops/empty.h>
// 引入创建空张量的头文件
#include <ATen/ops/empty_like.h>
// 引入创建类似空张量的头文件
#include <ATen/ops/empty_strided.h>
// 引入创建带步长的空张量的头文件
#include <ATen/ops/zeros.h>
// 引入创建全零张量的头文件
#endif

namespace at::native {

inline c10::MaybeOwned<Tensor> expect_resolved_conj(const Tensor& tensor) {
  // 如果张量需要共轭解析，则返回已解析的张量，否则返回借用的张量
  if (tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

inline DimVector batched_matrix_contiguous_strides(
    const IntArrayRef sizes,
    const bool f_contig = false) {
  // 根据尺寸数组计算连续的批量矩阵步长向量
  auto strides = c10::contiguous_strides(sizes);
  auto dim = strides.size();

  if (f_contig && dim >= 2) {
    // 如果需要 F-contiguous 的矩阵，则调整最后两个维度的步长
    strides[dim - 1] = std::max(sizes[dim - 2], static_cast<int64_t>(1));
    strides[dim - 2] = 1;
  }
  return strides;
}

/*
 * 克隆张量以满足以下条件：
 * 如果我们将张量视为大小为 (B, M, N)，其中 B 是任意数量的批量维度，则：
 * - 每个 (M, N) 矩阵以列优先形式排列
 * - 让张量 P 大小为 (B, M, N)，Q 大小为 (B, M', N')。
 *   那么在内存中，从 P.data_ptr()[B * M * N] 开始的 MxN 矩阵与从 Q.data_ptr()[B * M' * N'] 开始的对应批次的 M'xN' 矩阵相同。
 */
inline Tensor cloneBatchedColumnMajor(const Tensor& src) {
  // 如果 src 已经是批量列主格式，则效率高（不会重新排序数据）
  // 因为第一次转置将使张量连续，而克隆连续的张量速度很快。
  auto result = src.mT().clone(at::MemoryFormat::Contiguous);
  result.transpose_(-2, -1);
  return result;
}

/*
 * contig 参数选择 C-contig（true）或 F-contig（false）
 */
inline c10::MaybeOwned<Tensor> borrow_else_clone(const bool cond, const Tensor& borrow, const Tensor& clone, const bool contig) {
  return cond ? c10::MaybeOwned<Tensor>::borrowed(borrow)
              : c10::MaybeOwned<Tensor>::owned(contig ? clone.clone(MemoryFormat::Contiguous)
                                                      : cloneBatchedColumnMajor(clone));
}


这样，我们为给定的 C++ 代码添加了详细的注释，解释了每个函数和相关变量的作用和功能。
/*
 * This method is designed to be a faster alternative to
 * `cloneBatchedColumnMajor` with some additional features,
 * namely:
 * 1. It uses `copy` instead of `clone` which could be much faster.
 * 2. `nrows` parameter used to create inputs with the number of rows larger
 * than the original input, which is required for some LAPACK/MAGMA methods.
 * 3. `desired_batch_size` is used to create copies with the batch size
 * which is either the original batch size of the input, or its larger
 * broadcasted shape.
 */
inline Tensor copyBatchedColumnMajor(const Tensor& src, int64_t nrows = -1,
    at::OptionalIntArrayRef desired_batch_sizes = c10::nullopt) {
  // Determine the number of rows to copy; default to src's second-last dimension if nrows is -1
  nrows = (nrows == -1) ? src.size(-2) : nrows;
  // Determine the sizes of the copy operation, either from desired_batch_sizes or inferred from src
  auto copy_sizes = desired_batch_sizes.has_value()
    ? desired_batch_sizes.value().vec()
    : IntArrayRef(src.sizes().data(), src.dim() - 2).vec();
  // Append nrows and the last dimension of src to copy_sizes
  copy_sizes.insert(copy_sizes.end(), {nrows, src.size(-1)});
  // Calculate the strides for the new tensor using column-major order
  const auto copy_strides = batched_matrix_contiguous_strides(copy_sizes, /*f-contig*/true);
  // Create a new tensor with the calculated sizes and strides, with options similar to src
  auto copy = at::empty_strided(copy_sizes, copy_strides, src.options());
  // Copy the relevant part of src (batched matrices) into the new tensor
  copy.narrow(-2, 0, src.size(-2)).copy_(src);
  // Return the newly created tensor copy
  return copy;
}

/*
 * Given batches of matrices with arbitrary batch dim,
 * computes the number of batches.
 */
inline int64_t batchCount(const Tensor& batched_matrices) {
  int64_t result = 1;
  // Multiply all dimensions except the last two to compute batch count
  for (int64_t i = 0; i < batched_matrices.ndimension() - 2; i++) {
    result *= batched_matrices.size(i);
  }
  return result;
}

// Computes the number of elements of a matrix in a batched matrix tensor
inline int64_t matrixStride(const Tensor& batched_matrices) {
  // Return the stride of a single matrix in a batched matrix tensor
  return batched_matrices.size(-1) * batched_matrices.size(-2);
}

// Validates input shapes for operations on batches of square matrices (inverse, cholesky, symeig, eig)
inline void checkIsMatrix(const Tensor& A, const char* const f_name, const char* const arg_name = "A") {
  // Ensure tensor A has at least 2 dimensions
  TORCH_CHECK(A.dim() >= 2, f_name, ": The input tensor ", arg_name, " must have at least 2 dimensions.");
}

inline void squareCheckInputs(const Tensor& self, const char* const f_name, const char* const arg_name = "A") {
  // Check if tensor self is a batch of square matrices
  checkIsMatrix(self, f_name, arg_name);
  // Ensure all matrices in the batch are square (symmetric)
  TORCH_CHECK(self.sym_size(-1) == self.sym_size(-2),
              f_name,
              ": ", arg_name, " must be batches of square matrices, "
              "but they are ", self.sym_size(-2), " by ", self.sym_size(-1), " matrices");
}

inline void checkInputsSolver(const Tensor& A,
                                     const Tensor& B,
                                     const bool left,
                                     const char* const f_name) {
  // Check input shapes for solving equations involving matrices A and B
  squareCheckInputs(A, f_name, "A");
  checkIsMatrix(B, f_name, "B");
  // Check compatibility of shapes based on whether A is on the left or right side of the equation
  TORCH_CHECK(left ? A.size(-2) == B.size(-2) : A.size(-1) == B.size(-1),
              f_name, ": Incompatible shapes of A and B for the equation ",
              left ? "AX = B" : "XA = B",
              " (", A.size(-2), "x", A.size(-1), " and ", B.size(-2), "x", B.size(-1), ")");
}
// 检查张量是否行或列连续。这可以用于线性代数方法，例如解线性方程组，其中要求张量是连续的列主要矩阵。
inline bool is_row_or_column_contiguous(const Tensor& t) {
  // 可以将此函数更加通用化，类似于在矩阵乘法中的检查方式，可以通过步长（例如 (6, 12, 1, 3) 或 (3, 1, 9)）消除拷贝，但这很复杂。
  // 我们选择为简单起见保守处理。
  return t.is_contiguous() || t.transpose(-2, -1).is_contiguous();
}

// 根据连续性和共轭性返回转置类型。
inline TransposeType to_transpose_type(const bool contig, const bool conj) {
  if (conj) {
    if (contig) { TORCH_INTERNAL_ASSERT(false, "Invalid transpose type"); }
    else {        return TransposeType::ConjTranspose; }
  } else {
    if (contig) { return TransposeType::NoTranspose; }
    else {        return TransposeType::Transpose; }
  }
}

// 此函数设计用于与最小化 L(ax - b) = 0 的线性代数方法配合使用，其中 L 通常是单位映射（如 `solve`）或 L2 范数（如 `lstsq`）。
// 预期 `a` 和 `b` 是连续的列主要张量，具有以下附加属性：
// 1. a.dim() == b.dim()
// 2. a.shape[:-2] 广播到 b.shape[:-2]
// 3. 对于 i=0,..., a.dim() - 3，a.size(i) <= b.size(i)（仅限批处理维度）
//
// MAGMA/LAPACK 在原地修改张量 `a`，此方法的主要目标是内存效率，即如果存在索引 i 满足 a.shape[i] < b.shape[i]，
// 0 <= i <= a.dim() - 3，则在广播形状的情况下，保留 `a` 的缓冲副本及用于检查是否已访问 `a` 的特定批处理维度的标志。
// 如果已访问，则从缓冲区复制数据到 `a`。复制次数不超过 prod(max(a.shape[:-2], b.shape[:-2]) - a.shape[:-2] + 1)，
// 此值由具有非空批处理维度的张量达到。
//
// func_t `f` 是一个可调用对象，接受 scalar_t* a_working_ptr, scalar_t* b_working_ptr, int64_t a_linear_batch_idx 作为参数。
// a_working_ptr 和 b_working_ptr 可直接传递给 LAPACK/MAGMA 程序，
// 而 a_linear_batch_idx 是 3D 表示中的索引，对应于 a_working_ptr 指向的内存，
// 换句话说：
// a_working_ptr == a.view({-1, a.size(-2), a.size(-1)}.select(0, a_linear_batch_idx).data_ptr<scalar_t>();
// a_linear_batch_idx 对于存储与 `a` 相关的元数据非常有用，例如其秩或奇异值（参见 linalg_lstsq）。
template<typename scalar_t, typename func_t>
void batch_iterator_with_broadcasting(const Tensor& a, const Tensor& b, const func_t& f) {
  // 获取 `a` 和 `b` 的批处理大小，排除最后两个维度
  IntArrayRef a_batch_sizes(a.sizes().data(), a.dim() - 2);
  IntArrayRef b_batch_sizes(b.sizes().data(), b.dim() - 2);

  // 创建包含 a 批次索引的张量
  auto a_linear_batch_idx = at::arange(batchCount(a)).view(a_batch_sizes);
  // 创建包含 b 批次索引的张量
  auto b_linear_batch_idx = at::arange(batchCount(b)).view(b_batch_sizes);

  // 设置张量迭代器
  TensorIterator iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    // 设置是否检查内存重叠，这里设为 false，表示不检查
    .check_all_same_dtype(false)
    // 设置是否检查所有张量的数据类型是否相同，这里设为 false，表示不检查
    .resize_outputs(false)
    // 设置是否调整输出大小，这里设为 false，表示不调整
    .add_output(b_linear_batch_idx)
    // 向迭代器添加输出张量 b_linear_batch_idx
    .add_input(a_linear_batch_idx)
    // 向迭代器添加输入张量 a_linear_batch_idx
    .build();
    // 构建迭代器

  auto m = a.size(-2);
  // 获取张量 a 在倒数第二维的大小
  auto n = a.size(-1);
  // 获取张量 a 在倒数第一维的大小
  auto a_3d = a.view({batchCount(a), m, n});
  // 将张量 a 重塑为三维张量，维度为 batchCount(a) × m × n
  auto b_3d = b.view({batchCount(b), b.size(-2), b.size(-1)});
  // 将张量 b 重塑为三维张量，维度为 batchCount(b) × b.size(-2) × b.size(-1)

  auto a_broadcasts_over_b = (a_batch_sizes != b_batch_sizes);
  // 检查张量 a 是否在张量 b 上广播
  Tensor a_buffer, a_was_accessed, a_buffer_3d;
  // 定义用于存储临时数据的张量和标志位

  std::function<void(int64_t)> check_if_copy_needed_for_a
    = [](int64_t /*a_curr_linear_batch_idx*/){};
    // 定义用于检查是否需要复制张量 a 的函数，默认为空函数

  if (a_broadcasts_over_b) {
    // 如果张量 a 在张量 b 上广播
    a_buffer = at::empty_strided(a.sizes(), a.strides(), a.options())
      .copy_(a);
    // 创建张量 a 的副本 a_buffer
    a_was_accessed = at::zeros(batchCount(a), at::kBool);
    // 创建一个布尔类型的张量 a_was_accessed，所有元素初始化为 false
    a_buffer_3d = a_buffer.view({batchCount(a), m, n});
    // 将张量 a_buffer 重塑为三维张量，维度为 batchCount(a) × m × n

    check_if_copy_needed_for_a = [&](int64_t a_curr_linear_batch_idx) {
      auto* a_was_accessed_flag = a_was_accessed
        .select(0, a_curr_linear_batch_idx)
        .data_ptr<bool>();
      // 获取张量 a_was_accessed 的指定索引处的标志位指针
      if (!(*a_was_accessed_flag)) {
        *a_was_accessed_flag = true;
      }
      else {
        a_3d.select(0, a_curr_linear_batch_idx)
          .copy_(a_buffer_3d.select(0, a_curr_linear_batch_idx));
        // 如果已访问过，则复制 a_buffer_3d 中对应的数据到 a_3d 中
      }
    };
  }

  auto loop = [&](char** data, const int64_t* strides, int64_t nelems) {
    // 定义循环操作函数，接受数据指针、步幅数组和元素个数作为参数
    auto* b_batch_idx_ptr = data[0];
    // 获取 batch 索引指针
    auto* a_batch_idx_ptr = data[1];
    // 获取 batch 索引指针

    for (const auto elem C10_UNUSED : c10::irange(nelems)) {
      // 对于每个元素，遍历指定范围
      auto b_curr_linear_batch_idx = *reinterpret_cast<int64_t*>(b_batch_idx_ptr);
      // 获取当前 b 张量的线性批次索引
      auto a_curr_linear_batch_idx = *reinterpret_cast<int64_t*>(a_batch_idx_ptr);
      // 获取当前 a 张量的线性批次索引

      check_if_copy_needed_for_a(a_curr_linear_batch_idx);
      // 检查是否需要复制张量 a

      auto* a_working_ptr = a_3d.select(0, a_curr_linear_batch_idx)
        .data_ptr<scalar_t>();
      // 获取当前工作的张量 a 的数据指针
      auto* b_working_ptr = b_3d.select(0, b_curr_linear_batch_idx)
        .data_ptr<scalar_t>();
      // 获取当前工作的张量 b 的数据指针
      f(a_working_ptr, b_working_ptr, a_curr_linear_batch_idx);
      // 调用函数 f 处理当前工作的张量 a 和 b 的数据，以及当前的批次索引

      b_batch_idx_ptr += strides[0];
      // 更新 b_batch_idx_ptr 指向下一个元素
      a_batch_idx_ptr += strides[1];
      // 更新 a_batch_idx_ptr 指向下一个元素
    }
  };
  iter.serial_for_each(loop, {0, batchCount(b)});
  // 使用迭代器 iter 对循环函数 loop 进行串行执行，处理 batchCount(b) 个元素
// 结束函数 _get_epsilon 的定义

// 根据 ScalarType 获取浮点类型（除了 half）的 epsilon 值
inline double _get_epsilon(const ScalarType& sc_type) {
  switch (sc_type) {
    case at::ScalarType::Float:
      return static_cast<double>(std::numeric_limits<float>::epsilon());
    case at::ScalarType::Double:
      return std::numeric_limits<double>::epsilon();
    default:
      // 报错，此函数不处理除了 float 和 double 之外的类型
      AT_ERROR("This function doesn't handle types other than float and double");
  }
}

// 验证输入形状和设备是否符合要求
// 用于线性求解方法（solve, cholesky_solve, lu_solve, triangular_solve）
inline void linearSolveCheckInputs(const Tensor& self, const Tensor& A, const char* name) {
  // 检查 b 和 A 是否在同一设备上
  TORCH_CHECK(self.device() == A.device(),
              "Expected b and A to be on the same device, but found b on ",
              self.device(), " and A on ", A.device(), " instead.");

  // 检查 b 和 A 是否具有相同的数据类型
  TORCH_CHECK(self.scalar_type() == A.scalar_type(),
              "Expected b and A to have the same dtype, but found b of type ",
              self.scalar_type(), " and A of type ", A.scalar_type(), " instead.");

  // 检查 A 是否是批量的方阵
  TORCH_CHECK(A.size(-1) == A.size(-2),
              "A must be batches of square matrices, "
              "but they are ", A.size(-2), " by ", A.size(-1), " matrices");

  // 检查 A 的最后两维和 b 的维度是否匹配
  TORCH_CHECK(A.size(-1) == self.size(-2),
              "Incompatible matrix sizes for ", name, ": each A "
              "matrix is ", A.size(-1), " by ", A.size(-1),
              " but each b matrix is ", self.size(-2), " by ", self.size(-1));
}

// 检查是否是浮点类型或复数类型的 Tensor
inline void checkFloatingOrComplex(const Tensor& t, const char* const f_name, const bool allow_low_precision_dtypes=true) {
  auto dtype = t.scalar_type();
  // 检查是否是浮点类型或复数类型
  TORCH_CHECK((at::isFloatingType(dtype) || at::isComplexType(dtype)),
              f_name, ": Expected a floating point or complex tensor as input. Got ", dtype);
  if (!allow_low_precision_dtypes) {
    // 如果不允许低精度的数据类型，则进一步检查是否是支持的类型
    TORCH_CHECK(dtype == kFloat || dtype == kDouble || dtype == kComplexFloat || dtype == kComplexDouble,
                f_name, ": Low precision dtypes not supported. Got ", dtype);
  }
}

// 检查 TensorList 中的所有 Tensor 是否具有相同的维度
inline void checkAllSameDim(TensorList tensors, int64_t dim) {
  for (auto &t : tensors) {
    // 检查每个 Tensor 的维度是否符合预期
    TORCH_CHECK(t.dim() == dim, "Tensor dimension is ", t.dim(), ", expected ", dim, " instead.");
  }
}
// 在线性代数运算中，根据输入张量 arg1 和 arg2，广播它们的批处理维度
inline std::tuple<std::vector<int64_t>, std::vector<int64_t>> _linalg_broadcast_batch_dims(const Tensor& arg1, const Tensor& arg2) {
  // 从 arg1 和 arg2 的尺寸中获取批处理维度大小
  IntArrayRef arg1_batch_sizes(arg1.sizes().data(), arg1.ndimension() - 2);
  IntArrayRef arg2_batch_sizes(arg2.sizes().data(), arg2.ndimension() - 2);
  // 推断扩展后的批处理部分的尺寸
  std::vector<int64_t> expand_batch_portion = infer_size(arg1_batch_sizes, arg2_batch_sizes);

  // 构造扩展后的 arg1 尺寸
  std::vector<int64_t> arg1_expand_size({expand_batch_portion});
  arg1_expand_size.insert(arg1_expand_size.end(), { arg1.size(-2), arg1.size(-1) });

  // 构造扩展后的 arg2 尺寸
  std::vector<int64_t> arg2_expand_size({expand_batch_portion});
  arg2_expand_size.insert(arg2_expand_size.end(), { arg2.size(-2), arg2.size(-1) });

  // 返回包含扩展后尺寸的元组
  return std::make_tuple(std::move(arg1_expand_size), std::move(arg2_expand_size));
}

// 在线性代数运算中，根据输入张量 arg1 和 arg2，广播它们的批处理维度，同时处理线性求解的输入检查
inline std::tuple<Tensor,Tensor> _linalg_broadcast_batch_dims(const Tensor& arg1, const Tensor& arg2, const char* name) {
  // 如果传入的 name 不为空指针，则进行线性求解的输入检查
  if (name != nullptr) {
    linearSolveCheckInputs(arg1, arg2, name);
  }

  // 获取扩展后的 arg1 和 arg2 尺寸
  auto [arg1_expand_size, arg2_expand_size] = at::native::_linalg_broadcast_batch_dims(arg1, arg2);

  // 如果尺寸已经是扩展后的，则直接使用，否则进行扩展
  auto arg1_broadcasted  = arg1_expand_size == arg1.sizes() ? arg1 : arg1.expand(arg1_expand_size);
  auto arg2_broadcasted  = arg2_expand_size == arg2.sizes() ? arg2 : arg2.expand(arg2_expand_size);

  // 返回广播后的张量元组
  return std::make_tuple(arg1_broadcasted, arg2_broadcasted);
}

// 根据输入张量 t1 和 t2 的大小及指定的批处理维度数，返回广播后的批处理尺寸
inline std::vector<int64_t> broadcast_batch_size(const Tensor& t1, const Tensor& t2, int64_t n_batch_dims) {
  // 获取 t1 和 t2 的批处理维度尺寸
  IntArrayRef t1_batch_sizes(t1.sizes().data(), n_batch_dims);
  IntArrayRef t2_batch_sizes(t2.sizes().data(), n_batch_dims);
  // 推断扩展后的批处理尺寸
  auto broadcasted_batch_sizes = infer_size(t1_batch_sizes, t2_batch_sizes);
  // 返回广播后的批处理尺寸
  return broadcasted_batch_sizes;
}

// 返回将给定轴移动到张量 self 的最后位置的排列
inline Tensor _move_to_end(const Tensor& self, IntArrayRef axes) {
  const std::vector<int64_t> a = axes.vec();
  const int64_t ndim = self.ndimension();
  std::vector<int64_t> perm;

  // 将非移动轴添加到排列中
  for (const auto i : c10::irange(ndim)) {
    auto it = std::find(a.begin(), a.end(), i);
    if (it == a.end()) {
       perm.push_back(i);
    }
  }
  // 将移动的轴添加到排列中
  for (auto i : a) {
    perm.push_back(i);
  }

  // 检查排列长度与张量维度是否匹配
  TORCH_CHECK((int64_t)perm.size() == ndim,
    "duplicate or invalid axis in 'dim' argument for tensor with ndim==", ndim);

  // 返回按排列重排后的张量
  return self.permute(perm);
}

// 解析 linalg_qr 中的 "mode" 参数，返回一个布尔值元组 (compute_q, reduced)
inline std::tuple<bool, bool> _parse_qr_mode(c10::string_view mode) {
  bool compute_q;
  bool reduced;
  // 根据 mode 参数设置 compute_q 和 reduced 的值
  if (mode == "reduced") {
    compute_q = true;
    reduced = true;
  } else if (mode == "complete") {
    compute_q = true;
    reduced = false;
  } else if (mode == "r") {
    compute_q = false;
    reduced = true;
  } else {
    // 如果 mode 参数不匹配任何已知模式，则抛出错误
    TORCH_CHECK(false, "Invalid mode argument '", mode, "' for linalg_qr");
  }
  // 返回 compute_q 和 reduced 的布尔值元组
  return std::make_tuple(compute_q, reduced);
}
    reduced = true; // 在这种模式下实际上是无关紧要的
  } else {
      // 使用 TORCH_CHECK 来确保模式 mode 的有效性，如果不是预期的模式，则抛出错误信息
      TORCH_CHECK(false, "qr received unrecognized mode '", mode,
                  "' but expected one of 'reduced' (default), 'r', or 'complete'");
  }
  // 返回一个包含计算结果和 reduced 值的元组
  return std::make_tuple(compute_q, reduced);
}

// 结束一个代码块的闭合括号

// Function to compute sizes, strides and the extra columns for the Q matrix in the QR Decomposition
inline std::tuple<DimVector, DimVector, int64_t> _compute_geometry_for_Q(
    const Tensor& input,    // 输入张量
    bool reduced) {         // 是否为减少模式的标志位
  int64_t m = input.size(-2), n = input.size(-1);    // 获取输入张量的倒数第二和倒数第一维度的大小
  int64_t n_columns_q;    // Q 矩阵的列数

  // 根据 `reduced` 参数计算 Q 矩阵的大小需求
  DimVector q_sizes(input.sizes());    // 复制输入张量的尺寸
  if (!reduced && m > n) {    // 如果非减少模式且 m > n
    q_sizes[input.dim() - 1] = m;    // 设置 Q 矩阵的最后一维大小为 m
    n_columns_q = m;    // 设置 Q 矩阵的列数为 m
  } else {
    q_sizes[input.dim() - 1] = n;    // 否则设置 Q 矩阵的最后一维大小为 n
    n_columns_q = std::min(m, n);    // 设置 Q 矩阵的列数为 m 和 n 中的较小值
  }
  auto q_strides = batched_matrix_contiguous_strides(q_sizes, /*f-contig*/true);    // 计算 Q 矩阵的步长
  return std::make_tuple(q_sizes, q_strides, n_columns_q);    // 返回 Q 矩阵的尺寸、步长和列数
}

inline bool svd_uses_cusolver(const Tensor& A) {
  // 如果 CUDA 可用，并且有 CuSOLVER 库，且不优先选择 Magma 后端，则使用 CuSOLVER
  return A.is_cuda()
         && at::globalContext().hasCuSOLVER()
         && at::globalContext().linalgPreferredBackend() != at::LinalgBackend::Magma;
}


// Function used instead of .to so that the original strides are retained
// .to doesn't retain strides and make the output tensor contiguous
inline Tensor same_stride_to(const Tensor& original_tensor, const at::TensorOptions& options) {
  auto strided_to = at::empty_strided(original_tensor.sizes(),    // 使用原始张量的尺寸创建一个新张量
                                      original_tensor.strides(),    // 使用原始张量的步长
                                      options);    // 使用给定选项创建新张量
  strided_to.copy_(original_tensor);    // 复制原始张量的数据到新张量
  return strided_to;    // 返回新张量
}

// Creates a dimension permutation array that can be given to `at::permute()`, which will shift
// the two specified dimensions to the end of a tensor, without changing the order of
// the other dimensions. `dim1` will be placed at the very end, and `dim0` will be
// placed just to the left of it.
//
// For instance, given a 4-D tensor, dimensions 1 and 3 can be shifted to the end by
// calling `create_dim_backshift_permutation(1, 3, 4)`. The resulting vector will
// be `vec(0, 2, 1, 3)`.
inline std::vector<int64_t> create_dim_backshift_permutation(int64_t dim0, int64_t dim1, int64_t ndim) {
  TORCH_CHECK(
    (dim0 != dim1) && (dim0 < ndim) && (dim0 >= 0) && (dim1 < ndim) && (dim1 >= 0),
    "duplicate or invalid dimensions");    // 检查维度参数的有效性
  std::vector<int64_t> permutation(ndim);    // 创建一个长度为 ndim 的排列数组
  int64_t cur_permuted_dim = 0;
  for (const auto dim_ind : c10::irange(ndim)) {    // 遍历维度范围
    if ((dim_ind != dim0) && (dim_ind != dim1)) {
      permutation[cur_permuted_dim++] = dim_ind;    // 将非指定维度的索引存入排列数组
    }
  }
  permutation[cur_permuted_dim++] = dim0;    // 存入 dim0 维度的索引
  permutation[cur_permuted_dim] = dim1;    // 存入 dim1 维度的索引
  return permutation;    // 返回创建的维度排列数组
}

// Creates a dimension permutation array that can be given to `at::permute()`, which
// will reverse a given permutation.
// The reverse permutation array is created by swapping the indices and their
// associated values from the given permutation array.
// 创建逆置排列的向量，输入为一个排列向量，输出为其逆置排列向量
inline std::vector<int64_t> create_reverse_permutation(std::vector<int64_t> permutation) {
  // 获取排列向量的长度
  int64_t ndim = permutation.size();
  // 创建一个与排列向量同样长度的逆置排列向量
  std::vector<int64_t> reverse_permutation(ndim);
  // 遍历排列向量的每个元素，构建逆置排列向量
  for (const auto dim_ind : c10::irange(ndim)) {
    reverse_permutation[permutation[dim_ind]] = dim_ind;
  }
  // 返回逆置排列向量
  return reverse_permutation;
}

// 计算 MAGMA/LAPACK 中 cgesdd/zgesdd 函数的 R-work 数组大小
// 参考 https://github.com/Reference-LAPACK/lapack/blob/122506cd8b6ce050a200920c3d4c0b153b150fd8/SRC/cgesdd.f#L186
inline int64_t computeLRWorkDim(const char jobz, int64_t m, int64_t n) {
  // 计算 m 和 n 的最小值
  auto mn = std::min(m, n);
  // 计算 m 和 n 的最大值
  auto mx = std::max(m, n);
  // 根据 jobz 参数不同返回不同的 R-work 数组大小
  if (jobz == 'N') {
#ifdef __APPLE__
    // macOS 平台下的设置，基于 LAPACK 3.2.1
    return 7 * mn;
#else
    // LAPACK 3.6+ 版本的设置
    return 5 * mn;
#endif
  }
  // 当 mx 超过 10 * mn 时，返回的 R-work 数组大小
  if (mx > 10 * mn) {
    return 5 * mn * mn + 5 * mn;
  }
  // 默认情况下返回的 R-work 数组大小
  return std::max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn);
}

// 检查 uplo 参数是否有效，允许的字符串为 "u", "U", "l", "L"
inline void checkUplo(const c10::string_view uplo) {
  // 将 uplo 参数转换为大写
  char uplo_uppercase = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo[0])));
  // 使用 TORCH_CHECK 验证 uplo 的有效性
  TORCH_CHECK(uplo.size() == 1 && (uplo_uppercase == 'U' || uplo_uppercase == 'L'),
    "Expected UPLO argument to be 'L' or 'U', but got ", uplo);
}

// 检查 result 和 input 张量是否在相同的设备上
inline void checkSameDevice(const std::string& fn_name, Tensor result, Tensor input, const std::string& result_name = "result") {
  // 使用 TORCH_CHECK 验证 result 和 input 张量是否在同一设备上
  TORCH_CHECK(
      result.device() == input.device(),
      fn_name,
      ": Expected ", result_name, " and input tensors to be on the same device, but got ",
      result_name, " on ", result.device(), " and input on ", input.device());
}

// 检查 result 和 input 张量的数据类型是否兼容（用于 _out 变体函数）
// 大多数线性代数函数要求输入和输出张量具有相同的数据类型（浮点或复数类型）
// 根据 https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
// 使用 c10::canCast 检查是否可以安全复制数据类型
inline void checkLinalgCompatibleDtype(const std::string& fn_name, Tensor result, Tensor input, const std::string& result_name = "result") {
  // 检查是否可以将 input 张量的数据类型安全地转换为 result 张量的数据类型
  bool can_cast = c10::canCast(input.scalar_type(), result.scalar_type());
  // 使用 TORCH_CHECK 验证数据类型是否兼容
  TORCH_CHECK(
      can_cast,
      fn_name,
      ": Expected ", result_name, " to be safely castable from ", input.scalar_type(), " dtype, but got ",
      result_name, " with dtype ", result.scalar_type());
}
/*
  检查输出类型是否与期望的结果类型兼容，如果不兼容则抛出错误。
*/
inline void checkLinalgCompatibleDtype(const std::string& fn_name, ScalarType out_type, ScalarType result_type, const std::string& out_name = "result") {
  bool can_cast = c10::canCast(result_type, out_type);
  TORCH_CHECK(
      can_cast,
      fn_name,
      ": Expected ", out_name, " to be safely castable from ", result_type, " dtype, but got ",
      out_name, " with dtype ", out_type);
}

/*
  检查容忍度张量是否为复数类型，如果是则抛出错误。
*/
inline void checkNotComplexTolerance(const Tensor& tol, const c10::string_view f_name, const c10::string_view tol_name) {
  TORCH_CHECK(!at::isComplexType(tol.scalar_type()),
              f_name, ": ", tol_name, " tensor of complex type is not supported. Got ", tol.scalar_type());
}

/*
  判断解线性方程组 matmul(input, x) = other 时 'other' 张量的类型：
  * 若 'other' 是一维张量或一批一维张量（向量情况）
  * 或 'other' 是二维张量或一批二维张量（矩阵情况）
  对于批量输入，我们需要能够区分这两种情况。
*/
inline bool linalg_solve_is_vector_rhs(const Tensor& input, const Tensor& other) {
  auto expected_batched_rhs_shape = SymIntArrayRef(input.sym_sizes().data(), input.dim() - 1); // input.shape[:-1]
  bool vector_case = other.dim() == 1 || (input.dim() - 1 == other.dim() && other.sym_sizes().equals(expected_batched_rhs_shape));
  return vector_case;
}

/*
  计算原始形状为 original_shape 的张量的线性索引，以便像使用广播张量那样访问其元素。
*/
inline Tensor get_linear_indices(int64_t numel, IntArrayRef original_shape, IntArrayRef broadcast_shape) {
  TensorOptions options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  return at::arange(numel, options).view(original_shape).broadcast_to(broadcast_shape).contiguous();
}

class BroadcastLinearIndices {
 private:
  Tensor linear_indices_;
  bool is_broadcasting_;

 public:
  BroadcastLinearIndices(
      int64_t numel,
      IntArrayRef original_shape,
      IntArrayRef broadcast_shape) : is_broadcasting_(!original_shape.equals(broadcast_shape)) {
    // 假设 broadcast_shape 是原始形状 original_shape 的广播形状，
    // 我们需要计算与 original_shape 兼容的线性索引，以访问原始张量中与广播张量对应的元素。
    if (is_broadcasting_) {
      linear_indices_ =
          get_linear_indices(numel, original_shape, broadcast_shape);
    }
  }
  int64_t operator()(int64_t broadcast_linear_index) {
    // 如果正在进行广播操作，返回线性索引数组中对应的数据指针所指向的整型值
    return is_broadcasting_
        ? linear_indices_.data_ptr<int64_t>()[broadcast_linear_index]
        // 否则，返回广播线性索引本身
        : broadcast_linear_index;
  }
}  // 结束 at::native 命名空间

// 检查输入张量是否支持 BLAS 兼容的列优先顺序
inline bool is_blas_compatible_column_major_order(const Tensor& input) {
    // 获取输入张量的步幅数组
    IntArrayRef input_strides = input.strides();
    // 获取输入张量的大小数组
    IntArrayRef input_sizes = input.sizes();
    // 获取输入张量的维度数
    auto ndim = input.dim();
    // 断言：仅在调试模式下检查张量维度至少为 2
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim >= 2);
    // 如果维度数大于 3，则返回经过 -2 和 -1 转置后是否连续
    if (ndim > 3) {
        return input.transpose(-2, -1).is_contiguous();
    }
    // 获取前导维度的步幅
    auto leading_dimension = input_strides[ndim - 1];
    // 获取行数
    auto rows = input_sizes[ndim - 2];
    bool batch_stride_compatible = true;
    // 如果维度数为 3
    if (ndim == 3) {
        // 获取列数
        auto cols = input_sizes[ndim - 1];
        // 检查批次步幅是否兼容
        batch_stride_compatible =
            input_strides[ndim - 3] >= leading_dimension * cols;
    }
    // 检查是否满足列优先顺序的条件
    return (input_strides[ndim - 2] == 1) &&
        (leading_dimension >= std::max<int64_t>(1, rows)) &&
        batch_stride_compatible;
}

// 检查输入张量是否支持 BLAS 兼容的行优先顺序
inline bool is_blas_compatible_row_major_order(const Tensor& input) {
    // 获取输入张量的步幅数组
    IntArrayRef input_strides = input.strides();
    // 获取输入张量的大小数组
    IntArrayRef input_sizes = input.sizes();
    // 获取输入张量的维度数
    auto ndim = input.dim();
    // 断言：仅在调试模式下检查张量维度至少为 2
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim >= 2);
    // 如果维度数大于 3，则直接返回是否连续
    if (ndim > 3) {
        return input.is_contiguous();
    }
    // 获取前导维度的步幅
    auto leading_dimension = input_strides[ndim - 2];
    // 获取列数
    auto cols = input_sizes[ndim - 1];
    bool batch_stride_compatible = true;
    // 如果维度数为 3
    if (ndim == 3) {
        // 获取行数
        auto rows = input_sizes[ndim - 2];
        // 检查批次步幅是否兼容
        batch_stride_compatible =
            input_strides[ndim - 3] >= leading_dimension * rows;
    }
    // 检查是否满足行优先顺序的条件
    return (input_strides[ndim - 1] == 1) &&
        (leading_dimension >= std::max<int64_t>(1, cols)) &&
        batch_stride_compatible;
}
```