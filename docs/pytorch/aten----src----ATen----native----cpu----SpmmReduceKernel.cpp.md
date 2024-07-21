# `.\pytorch\aten\src\ATen\native\cpu\SpmmReduceKernel.cpp`

```
// 定义宏，仅包含方法运算符的断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入张量操作的基本头文件
#include <ATen/core/Tensor.h>
// 引入张量扩展的实用工具
#include <ATen/ExpandUtils.h>
// 引入分发机制
#include <ATen/Dispatch.h>
// 引入并行计算支持
#include <ATen/Parallel.h>
// 引入向量功能的头文件
#include <ATen/cpu/vec/functional.h>
// 引入向量操作的头文件
#include <ATen/cpu/vec/vec.h>
// 引入稀疏矩阵-矩阵乘法的头文件
#include <ATen/native/cpu/SpmmReduceKernel.h>
// 引入减少操作的实用工具
#include <ATen/native/cpu/ReduceUtils.h>
// 引入CPU工具的实用功能
#include <ATen/native/cpu/utils.h>
// 引入范围计算工具
#include <c10/util/irange.h>
// 引入操作数的数学类型
#include <ATen/OpMathType.h>

// 根据宏定义是否包含每个操作符的单独头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros.h>
#endif

// 创建名为at的命名空间，嵌套命名空间native
namespace at { namespace native {

// 创建匿名命名空间，用于定义实现细节不对外部公开
namespace {

// 定义模板函数，实现稀疏矩阵-矩阵乘法的内核操作
template <typename scalar_t, typename index_t, ReductionType reduce>
// 内联函数，更新输出指针指向的数据
inline void _update(at::opmath_type<scalar_t>* out_ptr, int64_t e, int64_t c, const scalar_t val, const scalar_t* other_data, int64_t K) {
  // 使用opmath_type类型定义scalar_t的类型别名
  using opmath_t = at::opmath_type<scalar_t>;
  // 使用Vectorized类型定义scalar_t的向量化操作
  using Vec = vec::Vectorized<scalar_t>;
  // 使用VecType类型定义scalar_t的向量类型
  using aVec = VecType<scalar_t>;
  // 定义向量大小为向量化操作大小的四倍
  constexpr int64_t kVecSize = Vec::size();
  // 定义向量长度为向量大小乘以4
  constexpr int64_t kVLEN = kVecSize * 4;

  // 初始化k为0
  int64_t k = 0;
  // 创建val_vec向量，并初始化为val的值
  aVec val_vec = aVec((opmath_t)val);
  // 计算other_data的偏移量，指向当前列的起始位置
  const scalar_t* other_ptr = other_data + c * K;

  // 对于每个k，执行向量化操作直到K的前kVLEN的倍数
  for (; k < K - (K % kVLEN); k += kVLEN) {
    // 加载输出向量数据
    aVec out_vec0 = aVec::loadu(out_ptr + k);
    aVec out_vec1 = aVec::loadu(out_ptr + k + kVecSize);
    aVec out_vec2 = aVec::loadu(out_ptr + k + kVecSize * 2);
    aVec out_vec3 = aVec::loadu(out_ptr + k + kVecSize * 3);

    // 更新输出向量数据
    out_vec0 = update<aVec, reduce>(out_vec0, aVec::loadu(other_ptr + k) * val_vec);
    out_vec1 = update<aVec, reduce>(out_vec1, aVec::loadu(other_ptr + k + kVecSize) * val_vec);
    out_vec2 = update<aVec, reduce>(out_vec2, aVec::loadu(other_ptr + k + kVecSize * 2) * val_vec);
    out_vec3 = update<aVec, reduce>(out_vec3, aVec::loadu(other_ptr + k + kVecSize * 3) * val_vec);

    // 存储更新后的输出向量数据
    out_vec0.store(out_ptr + k);
    out_vec1.store(out_ptr + k + kVecSize);
    out_vec2.store(out_ptr + k + kVecSize * 2);
    out_vec3.store(out_ptr + k + kVecSize * 3);
  }
  // 对于每个k，执行非向量化操作直到K的前kVecSize的倍数
  for (; k < K - (K % kVecSize); k += kVecSize) {
    // 加载输出向量数据
    aVec out_vec = aVec::loadu(out_ptr + k);
    // 更新输出向量数据
    out_vec = update<aVec, reduce>(out_vec, aVec::loadu(other_ptr + k) * val_vec);
    // 存储更新后的输出向量数据
    out_vec.store(out_ptr + k);
  }
  // 对于剩余的k，执行逐元素更新操作
  for (; k < K; k++) {
    // 加载输出数据值
    opmath_t out_val = opmath_t(out_ptr[k]);
    // 更新输出数据值
    out_val = update<opmath_t, reduce>(out_val, opmath_t(other_ptr[k]) * opmath_t(val));
    // 存储更新后的输出数据值
    out_ptr[k] = out_val;
  }
}

// 定义稀疏矩阵-矩阵乘法的内核实现函数
template <typename scalar_t, typename index_t, ReductionType reduce>
void spmm_reduce_kernel_impl(
    const Tensor& out,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& other_) {

  // 获取稀疏矩阵的非零元素个数
  int64_t nnz = values.numel();
  // 如果非零元素个数为0，直接返回
  if (nnz == 0) {
    return;
  }

  auto other = other_.contiguous();

  // 通过 TensorAccessor 访问 `crow_indices`、`col_indices` 和 `values`
  scalar_t* out_data = out.data_ptr<scalar_t>();
  auto csr_data = crow_indices.accessor<const index_t, 1>();  // 获取 csr 矩阵的行指针数组
  auto col_data = col_indices.accessor<const index_t, 1>();  // 获取列索引数组
  auto val_data = values.accessor<const scalar_t, 1>();      // 获取数值数组
  const scalar_t* other_data = other.const_data_ptr<scalar_t>();  // 获取 `other_` 张量的数据指针

  int64_t M = crow_indices.numel() - 1;  // M 是非零元素所在行数
  int64_t K = other.size(-1);            // K 是 `other_` 张量的最后一个维度大小

  int num_threads = at::get_num_threads();  // 获取当前线程数
  using opmath_t = at::opmath_type<scalar_t>;
  Tensor buffer;
  opmath_t* buffer_data = nullptr;
  static constexpr bool need_acc = is_reduced_floating_point_v<scalar_t>;

  // 如果需要累加器，则创建并初始化缓冲区
  if constexpr (need_acc) {
    auto acc_type = at::toAccumulateType(out.scalar_type(), /*is_cuda=*/true);
    buffer = at::zeros({num_threads, K}, out.options().dtype(acc_type));
    buffer_data = buffer.data_ptr<opmath_t>();
  }

  // 并行处理稀疏 CSR 格式的矩阵
  utils::parallel_sparse_csr(csr_data, M, nnz, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads,
                "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    opmath_t* buffer_ptr = nullptr;

    int64_t row_start, row_end;
    for (const auto m : c10::irange(begin, end)) {
      row_start = csr_data[m];
      row_end = csr_data[m + 1];

      scalar_t* out_ptr = out_data + m * K;

      // 根据是否需要累加器选择合适的缓冲区指针
      if constexpr (need_acc) {
        buffer_ptr = buffer_data + tid * K;
      } else {
        buffer_ptr = reinterpret_cast<opmath_t*>(out_ptr);
      }

      // 步骤 1: 根据 reduce 类型重新初始化输出行
      int64_t count = row_end - row_start;
      if (count != 0) {
        _init<scalar_t, reduce>(out_ptr, buffer_ptr, K, /*include_self*/false);
      }

      // 步骤 2: 进行归约操作，按行块减少写入内存带宽
      constexpr int64_t CHUNK_SIZE = 16;
      for (int64_t e0 = row_start; e0 < row_end; e0 += CHUNK_SIZE) {
        int64_t e1 = std::min(e0 + CHUNK_SIZE, row_end);
        for (const auto e : c10::irange(e0, e1)) {
          int64_t c = col_data[e];
          scalar_t val = val_data[e];
          _update<scalar_t, index_t, reduce>(buffer_ptr, e, c, val, other_data, K);
        }
      }

      // 如果需要累加器，则在步骤 2 后将缓冲区内容转换到输出行
      if constexpr (need_acc) {
        if (count != 0) {
          vec::convert(buffer_ptr, out_ptr, K);
        }
      }

      // 步骤 3: 完成最终写入操作
      write<scalar_t, reduce>(out_ptr, count, K);
    }
  });
}

// update both val and arg, used for `amin` and `amax`
// it is a little troublesome to vectorize it since `scalar_t` and `index_t`
// might have different vector length, for example, each vector holds 8 floats
// and 4 int64_t.
// 定义一个函数模板，用于更新值和索引，用于最小值和最大值的计算
template <typename scalar_t, typename index_t, ReductionType reduce>
inline void update_with_index(scalar_t *val, scalar_t new_val, index_t *arg, index_t new_arg) {
  // 根据指定的 reduce 类型，更新值和索引
  if ((reduce == ReductionType::MIN && new_val < *val) ||
      (reduce == ReductionType::MAX && new_val > *val) ||
      at::_isnan<scalar_t>(new_val)) {
    *val = new_val;
    *arg = new_arg;
  }
}

// 定义一个函数模板，用于稀疏矩阵乘法的值和索引的归约操作
template <typename scalar_t, typename index_t, ReductionType reduce>
void spmm_reduce_arg_kernel_impl(
    const Tensor& out,
    const Tensor& arg_out,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& other_) {

  TORCH_CHECK(reduce == ReductionType::MAX || reduce == ReductionType::MIN);
  // 获取稀疏矩阵的非零元素个数
  int64_t nnz = values.numel();
  // 如果非零元素个数为 0，则直接返回
  if (nnz == 0) {
    return;
  }

  // 将 other_ 张量转换为连续存储
  auto other = other_.contiguous();

  // 获取输出张量的数据指针和索引输出张量的数据指针
  scalar_t* out_data = out.data_ptr<scalar_t>();
  index_t* arg_out_data = arg_out.data_ptr<index_t>();
  // 获取稀疏矩阵的行索引、列索引和数值数据访问器
  auto csr_data = crow_indices.accessor<const index_t, 1>();
  auto col_data = col_indices.accessor<const index_t, 1>();
  auto val_data = values.accessor<const scalar_t, 1>();
  // 获取 other 张量的常量数据指针
  const scalar_t* other_data = other.const_data_ptr<scalar_t>();

  // 获取稀疏矩阵的行数 M 和 other 张量的列数 K
  int64_t M = crow_indices.numel() - 1;
  int64_t K = other.size(-1);

  // 获取线程数
  int num_threads = at::get_num_threads();
  // 定义操作类型 opmath_t，并初始化缓冲区
  using opmath_t = at::opmath_type<scalar_t>;
  Tensor buffer;
  opmath_t* buffer_data = nullptr;
  // 判断是否需要累加器
  static constexpr bool need_acc = is_reduced_floating_point_v<scalar_t>;
  if constexpr (need_acc) {
    auto acc_type = at::toAccumulateType(out.scalar_type(), /*is_cuda=*/true);
    // 使用累加类型创建全零的缓冲区张量
    buffer = at::zeros({num_threads, K}, out.options().dtype(acc_type));
    buffer_data = buffer.data_ptr<opmath_t>();
  }

  // 并行处理每一行稀疏矩阵的数据
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    // 检查线程 ID 是否在有效范围内
    TORCH_CHECK(tid < num_threads,
                "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    // 定义缓冲区指针
    opmath_t* buffer_ptr = nullptr;

    // 定义行的起始位置、结束位置和 c 变量
    int64_t row_start, row_end, c;
    // 对于给定范围 [begin, end) 中的每个元素 m 进行迭代
    for (const auto m : c10::irange(begin, end)) {
      // 获取当前行的起始和结束索引
      row_start = csr_data[m];
      row_end = csr_data[m + 1];

      // 计算当前行对应的输出数据和输出索引的起始位置
      scalar_t* out_ptr = out_data + m * K;
      index_t* arg_out_ptr = arg_out_data + m * K;

      // 根据 need_acc 条件选择合适的缓冲区指针
      if constexpr (need_acc) {
        buffer_ptr = buffer_data + tid * K;
      } else {
        buffer_ptr = reinterpret_cast<opmath_t*>(out_ptr);
      }

      // 如果当前行不为空
      if (row_end != row_start) {
        // 初始化输出数据和缓冲区，根据 reduce 类型进行操作
        _init<scalar_t, reduce>(out_ptr, buffer_ptr, K, /*include_self*/false);

        // 遍历当前行中的每个元素 e
        for (const auto e : c10::irange(row_start, row_end)) {
          // 获取列索引和对应的值
          c = col_data[e];
          opmath_t val = opmath_t(val_data[e]);

          // 获取其他行对应列的数据指针
          const scalar_t* other_ptr = other_data + c * K;

          // 遍历当前行的每个输出维度 k
          for (const auto k : c10::irange(K)) {
            // 使用索引更新缓冲区中的值，并更新输出索引和行索引
            update_with_index<opmath_t, index_t, reduce>(
                &buffer_ptr[k], opmath_t(val * other_ptr[k]), &arg_out_ptr[k], index_t(e));
          };
        }
      }

      // 如果需要累加操作，则在当前行不为空时执行向量转换
      if constexpr (need_acc) {
        if (row_end != row_start) {
          vec::convert(buffer_ptr, out_ptr, K);
        }
      }
    }
// 函数定义，用于反向传播计算稀疏矩阵乘法（SPMM）输入的梯度，针对不同的归约类型
template <typename scalar_t, typename index_t, ReductionType reduce>
void spmm_reduce_backward_input_kernel_impl(
    // 梯度自身张量，用于计算梯度
    const Tensor& grad_self,
    // 输出梯度张量，作为计算的输入之一
    const Tensor& grad_out_,
    // 行指示器的张量，描述每行的起始位置
    const Tensor& crow_indices,
    // 列指示器的张量，描述每个非零元素的列索引
    const Tensor& col_indices,
    // 其他张量，包含用于计算的额外数据
    const Tensor& other_,
    // 行索引的张量，描述每个非零元素的行索引
    const Tensor& row_indices) {

  // 计算非零元素的数量
  int64_t nnz = grad_self._nnz();
  // 如果非零元素数量为0，则直接返回
  if (nnz == 0) {
    return;
  }

  // 对输出梯度和其他张量进行连续化处理，确保数据连续性
  auto grad_out = grad_out_.contiguous();
  auto other = other_.contiguous();

  // 获取梯度自身张量的值，并按照标量类型存取
  auto values = grad_self.values();
  auto grad_values_data = values.accessor<scalar_t, 1>();
  // 获取输出梯度数据的常量指针
  const scalar_t* grad_out_data = grad_out.const_data_ptr<scalar_t>();
  // 获取行指示器数据的访问器，用于存取常量索引类型数据
  auto crow_data = crow_indices.accessor<const index_t, 1>();
  // 获取列指示器数据的访问器，用于存取常量索引类型数据
  auto col_data = col_indices.accessor<const index_t, 1>();
  // 获取其他张量数据的常量指针
  const scalar_t* other_data = other.const_data_ptr<scalar_t>();
  // 获取行索引数据的访问器，用于存取常量索引类型数据
  auto row_data = row_indices.accessor<const index_t, 1>();

  // 获取输出梯度的列数 K
  int64_t K = grad_out.size(1);

  // 使用向量化操作类型 Vec
  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
  // 并行处理非零元素的计算
  at::parallel_for(0, nnz, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      // 获取当前非零元素的行和列索引
      index_t row = row_data[i], col = col_data[i];

      // 使用向量化操作计算当前元素的值
      scalar_t val = vec::map2_reduce_all<scalar_t>(
          // 乘法操作
          [](Vec x, Vec y) { return x * y; },
          // 加法操作
          [](Vec x, Vec y) { return x + y; },
          // other_data 和 grad_out_data 中的数据
          other_data + col * K,
          grad_out_data + row * K,
          K);

      // 如果归约类型为 MEAN，则需要计算均值
      if (reduce == ReductionType::MEAN) {
        // 获取当前行的起始和结束位置
        index_t row_start = crow_data[row], row_end = crow_data[row + 1];
        // 对当前值进行均值处理
        val /= (row_end - row_start);
      }

      // 将计算得到的值存入梯度值数据中
      grad_values_data[i] = val;
    }
  });
}

// 对于归约类型为 'amax' 或 'amin' 的反向传播
template <typename scalar_t, typename index_t>
void spmm_reduce_backward_input_arg_kernel_impl(
    // 梯度自身张量，用于计算梯度
    const Tensor& grad_self,
    // 输出梯度张量，作为计算的输入之一
    const Tensor& grad_out_,
    // 列指示器的张量，描述每个非零元素的列索引
    const Tensor& col_indices,
    // 其他张量，包含用于计算的额外数据
    const Tensor& other_,
    // 输出参数张量，用于存储最大或最小值的索引
    const Tensor& arg_out_) {

  // 计算非零元素的数量
  int64_t nnz = grad_self._nnz();
  // 如果非零元素数量为0，则直接返回
  if (nnz == 0) {
    return;
  }

  // 对输出梯度、其他张量和参数输出张量进行连续化处理，确保数据连续性
  auto grad_out = grad_out_.contiguous();
  auto other = other_.contiguous();
  auto arg_out = arg_out_.contiguous();

  // 获取梯度自身张量的值，并按照标量类型存取
  auto grad_values = grad_self.values();
  auto grad_values_data = grad_values.accessor<scalar_t, 1>();
  // 获取输出梯度数据的常量指针
  const scalar_t* grad_out_data = grad_out.const_data_ptr<scalar_t>();
  // 获取列指示器数据的访问器，用于存取常量索引类型数据
  auto col_data = col_indices.accessor<const index_t, 1>();
  // 获取其他张量数据的常量指针
  const scalar_t* other_data = other.const_data_ptr<scalar_t>();
  // 获取参数输出数据的指针，用于存取索引类型数据
  index_t* arg_out_data = arg_out.data_ptr<index_t>();

  // 获取输出梯度的行数 M 和列数 K
  int64_t M = grad_out.size(0);
  int64_t K = grad_out.size(1);
  // 创建一个与输出梯度相同大小的张量 grad，用于存储梯度数据
  auto grad = at::empty({M, K}, grad_out.options());
  scalar_t* grad_data = grad.mutable_data_ptr<scalar_t>();

  // 并行处理每一行的计算
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; row++) {
      // 对于每一行，遍历其所有列
      for (int64_t j = 0; j < K; j++) {
        // 初始化当前最大或最小值的索引为0
        index_t argmax = 0;
        // 获取当前元素在张量中的索引位置
        index_t idx = row * K + j;
        // 遍历所有列的数据，寻找最大或最小值的索引
        for (int64_t col = 1; col < nnz; col++) {
          // 获取下一个元素的索引位置
          index_t next_idx = row * K + col;
          // 如果下一个元素大于当前元素，则更新最大或最小值的索引
          if (grad_out_data[next_idx] > grad_out_data[idx]) {
            idx = next_idx;
            argmax = col;
          }
        }
        // 将最大或最小值的索引存入参数输出张量中
        arg_out_data[idx] = argmax;
        // 将计算得到的梯度值存入 grad_data 中
        grad_data[idx] = grad_out_data[idx];
      }
    }
  });
}
    // 遍历范围从 begin 到 end 的索引 m
    for (const auto m : c10::irange(begin, end)) {
      // 计算 grad_out_ptr 指向的地址，每次递增 K 个元素
      const scalar_t* grad_out_ptr = grad_out_data + m * K;
      // 计算 grad_ptr 指向的地址，每次递增 K 个元素
      scalar_t* grad_ptr = grad_data + m * K;
      // 计算 arg_out_ptr 指向的地址，每次递增 K 个元素
      index_t* arg_out_ptr = arg_out_data + m * K;

      // 遍历 K 个元素
      for (const auto k : c10::irange(K)) {
        // 检查 arg_out_ptr[k] 是否等于 nnz
        if (arg_out_ptr[k] == index_t(nnz)) {
          // 如果是，则将 grad_ptr[k] 设为零
          grad_ptr[k] = scalar_t(0);
        } else {
          // 否则，根据 arg_out_data[m * K + k] 找到对应的列 col
          index_t col = col_data[arg_out_data[m * K + k]];
          // 根据列 col 和当前位置 k 计算权重并更新 grad_ptr[k]
          grad_ptr[k] = other_data[col * K + k] * grad_out_ptr[k];
        }
      }
    }
  });

  // 执行 scatter_add 操作，考虑使用原子操作来并行化
  for (const auto i : c10::irange(M * K)) {
    // 获取 arg_out_data[i] 的值作为索引 ind
    index_t ind = arg_out_data[i];
    // 如果 ind 不等于 nnz，则将 grad_data[i] 累加到 grad_values_data[ind]
    if (ind != index_t(nnz)) {
      grad_values_data[ind] += grad_data[i];
    }
  }
// 定义稀疏矩阵乘法的核心函数，用于将归约操作应用于输入矩阵的特定行
void spmm_reduce_kernel(
    const Tensor& out,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    const Tensor& other,
    ReductionType reduce_op) {
    // 使用AT_DISPATCH_FLOATING_TYPES_AND宏展开所有浮点类型和BFloat16类型的计算，
    // 在"spmm_reduce_kernel"上下文中执行以下lambda表达式
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_kernel", [&]() {
      // 使用AT_DISPATCH_INDEX_TYPES宏展开所有索引类型的计算，
      // 在"spmm_reduce_indices"上下文中执行以下lambda表达式
      AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_indices", [&]() {
        // 使用AT_DISPATCH_REDUCTION_TYPES宏展开所有reduce操作的计算，
        // 在当前lambda表达式中执行以下lambda表达式
        AT_DISPATCH_REDUCTION_TYPES(reduce_op, [&]() {
          // 调用spmm_reduce_kernel_impl模板函数，使用当前lambda表达式中捕获的参数
          // 参数分别为输出(out)、行索引(crow_indices)、列索引(col_indices)、数值(values)和其他参数(other)
          spmm_reduce_kernel_impl<scalar_t, index_t, reduce>(
              out, crow_indices, col_indices, values, other);
        });
      });
    });
    
    
    这段代码展示了一个嵌套的宏调用和lambda表达式的使用方式，用于根据不同的数据类型和操作类型分发并执行特定的函数模板。
void spmm_reduce_arg_kernel(
    const Tensor& out,                       // 输出张量 out
    const Tensor& arg_out,                   // 输出张量 arg_out
    const Tensor& crow_indices,              // 稀疏矩阵的行索引张量
    const Tensor& col_indices,               // 稀疏矩阵的列索引张量
    const Tensor& values,                    // 稀疏矩阵的值张量
    const Tensor& other,                     // 其他输入张量
    ReductionType reduce_op) {               // 减少操作的类型
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_indices", [&]() {
      AT_DISPATCH_REDUCTION_TYPES(reduce_op, [&]() {
        spmm_reduce_arg_kernel_impl<scalar_t, index_t, reduce>(  // 调用具体的内核实现函数
            out, arg_out, crow_indices, col_indices, values, other);
      });
    });
  });
}

void spmm_reduce_backward_input_kernel(
    const Tensor& grad_self,                 // 梯度自身张量
    const Tensor& grad_out,                  // 输出梯度张量
    const Tensor& crow_indices,              // 稀疏矩阵的行索引张量
    const Tensor& col_indices,               // 稀疏矩阵的列索引张量
    const Tensor& other,                     // 其他输入张量
    const Tensor& row_indices,               // 稀疏矩阵的行索引张量
    ReductionType reduce_op) {               // 减少操作的类型
  TORCH_CHECK(reduce_op == ReductionType::SUM || reduce_op == ReductionType::MEAN);  // 检查减少操作类型的有效性
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, other.scalar_type(), "spmm_reduce_backward_input_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_backward_input_indices", [&]() {
      AT_DISPATCH_REDUCTION_TYPES(reduce_op, [&]() {
        spmm_reduce_backward_input_kernel_impl<scalar_t, index_t, reduce>(  // 调用具体的内核实现函数
            grad_self, grad_out, crow_indices, col_indices, other, row_indices);
      });
    });
  });
}

void spmm_reduce_backward_input_arg_kernel(
    const Tensor& grad_self,                 // 梯度自身张量
    const Tensor& grad_out,                  // 输出梯度张量
    const Tensor& col_indices,               // 稀疏矩阵的列索引张量
    const Tensor& other,                     // 其他输入张量
    const Tensor& arg_out,                   // 输出 arg_out 张量
    ReductionType reduce_op) {               // 减少操作的类型
  TORCH_CHECK(reduce_op == ReductionType::MAX || reduce_op == ReductionType::MIN);  // 检查减少操作类型的有效性
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, other.scalar_type(), "spmm_reduce_backward_input_arg_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_backward_input_arg_indices", [&]() {
      spmm_reduce_backward_input_arg_kernel_impl<scalar_t, index_t>(  // 调用具体的内核实现函数
          grad_self, grad_out, col_indices, other, arg_out);
    });
  });
}

void spmm_reduce_normalize_values_kernel(
    const Tensor& normalized_values,         // 规范化后的值张量
    const Tensor& values,                    // 原始值张量
    const Tensor& crow_indices,              // 稀疏矩阵的行索引张量
    const Tensor& row_indices) {             // 稀疏矩阵的行索引张量
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_normalize_values_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "spmm_reduce_normalize_values_indices", [&]() {
      spmm_reduce_normalize_values_kernel_impl<scalar_t, index_t>(  // 调用具体的内核实现函数
          normalized_values, values, crow_indices, row_indices);
    });
  });
}

void spmm_reduce_backward_other_kernel(
    const Tensor& grad_other,                // 其他梯度张量
    const Tensor& grad_out,                  // 输出梯度张量
    const Tensor& crow_indices,              // 稀疏矩阵的行索引张量
    const Tensor& values,                    // 稀疏矩阵的值张量
    const Tensor& row_indices,               // 稀疏矩阵的行索引张量
    const Tensor& ccol_indices,              // 稀疏矩阵的列索引张量
    const Tensor& csr2csc,                   // 稀疏矩阵的转置指示张量
    // 检查 reduce_op 是否为 SUM 或 MEAN，必须是这两种类型之一
    TORCH_CHECK(reduce_op == ReductionType::SUM || reduce_op == ReductionType::MEAN);
    // 需要将 row_indices 按照 CSR 到 CSC 的顺序重新排列
    auto row = row_indices.index_select(0, csr2csc);

    Tensor val;
    // 如果 reduce_op 是 MEAN 类型
    if (reduce_op == ReductionType::MEAN) {
        // 对于 "mean" 类型的规约，需要对值进行归一化
        // 使用与 values 相同大小和选项的空张量来存储归一化后的值
        Tensor normalized_values = at::empty(values.sizes(), values.options());
        // 调用归一化值的内核函数，根据行索引和非零元素计数来归一化值
        spmm_reduce_normalize_values_kernel(normalized_values, values, crow_indices, row_indices);
        // 将归一化后的值按照 CSC 的顺序重新排列
        val = normalized_values.index_select(0, csr2csc);
    } else {
        // 如果 reduce_op 是 SUM 类型，则直接按照 CSC 的顺序选择原始的值
        val = values.index_select(0, csr2csc);
    }

    // 调用规约内核函数，对给定的输入进行稀疏矩阵乘积（SPMM）规约操作
    spmm_reduce_kernel(grad_other, ccol_indices, row, val, grad_out, ReductionType::SUM);
} // 匿名命名空间的结束

void spmm_reduce_backward_other_arg_kernel(
    const Tensor& grad_other,  // 接收梯度关于其它参数的张量
    const Tensor& grad_out,    // 接收梯度输出的张量
    const Tensor& col_indices, // 列索引张量
    const Tensor& values,      // 值张量
    const Tensor& arg_out,     // 输出参数张量
    ReductionType reduce_op)   // 减少操作类型
{
  // 确保减少操作是 MAX 或 MIN 类型
  TORCH_CHECK(reduce_op == ReductionType::MAX || reduce_op == ReductionType::MIN);

  // 根据值的数据类型（浮点型或者 BF16）调度相应的操作
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, values.scalar_type(), "spmm_reduce_backward_other_arg_kernel", [&]() {
    // 根据列索引的数据类型调度相应的操作
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "spmm_reduce_backward_other_arg_indices", [&]() {
      // 调用实际的内核函数实现，传递对应的参数
      spmm_reduce_backward_other_arg_kernel_impl<scalar_t, index_t>(
          grad_other, grad_out, col_indices, values, arg_out);
    });
  });
}

// 匿名命名空间的结束

// 注册分发函数以便后续调用
REGISTER_DISPATCH(spmm_reduce_stub, &spmm_reduce_kernel);
REGISTER_DISPATCH(spmm_reduce_arg_stub, &spmm_reduce_arg_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_input_stub, &spmm_reduce_backward_input_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_input_arg_stub, &spmm_reduce_backward_input_arg_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_other_stub, &spmm_reduce_backward_other_kernel);
REGISTER_DISPATCH(spmm_reduce_backward_other_arg_stub, &spmm_reduce_backward_other_arg_kernel);

}} // at::native 命名空间的结束
```