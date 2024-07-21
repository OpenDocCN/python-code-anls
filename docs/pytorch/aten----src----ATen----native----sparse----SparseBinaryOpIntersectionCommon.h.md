# `.\pytorch\aten\src\ATen\native\sparse\SparseBinaryOpIntersectionCommon.h`

```
#pragma once
// 包含头文件 <ATen/Tensor.h>，定义了张量操作相关的函数和结构
#include <ATen/Tensor.h>
// 包含头文件 <ATen/native/TensorIterator.h>，定义了张量迭代器相关的功能
#include <ATen/native/TensorIterator.h>
// 包含头文件 <ATen/Dispatch.h>，定义了分发机制相关的功能
#include <ATen/Dispatch.h>
// 包含头文件 <ATen/native/sparse/Macros.h>，定义了稀疏张量操作的宏
#include <ATen/native/sparse/Macros.h>
// 包含头文件 <ATen/ExpandUtils.h>，定义了张量扩展相关的功能
#include <ATen/ExpandUtils.h>
// 包含头文件 <ATen/native/SparseTensorUtils.h>，定义了稀疏张量工具函数
#include <ATen/native/SparseTensorUtils.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含以下头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含以下头文件
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/result_type.h>
#endif

// 如果定义了 GPUCC 宏，则定义 NAME 为 "sparse_binary_op_intersection_cuda"
// 否则定义 NAME 为 "sparse_binary_op_intersection_cpu"
#ifdef GPUCC
#define NAME "sparse_binary_op_intersection_cuda"
#else
#define NAME "sparse_binary_op_intersection_cpu"
#endif

// 命名空间 at::native 中的匿名命名空间
namespace at::native {

// 使用 at::sparse::get_sparse_impl 函数
namespace {

using at::sparse::get_sparse_impl;

// ForwardIt: 只支持传统的随机访问迭代器
// 查找边界的函数模板，根据 is_lower 参数确定是找到下界还是上界
template<class ForwardIt, class T, bool is_lower = true>
static FUNCAPI INLINE
ForwardIt find_bound(ForwardIt first, ForwardIt last, const T& value) {
    ForwardIt RESTRICT it;
    typename std::iterator_traits<ForwardIt>::difference_type count, step;
    // 注意：std::distance(first, last) 在 CUDA 上编译虽然有效，但结果不正确，
    // 所以只有传统的随机访问迭代器在此代码中是安全的。
    count = last - first;

    while (count > 0) {
      it = first;
      step = count / 2;
      // 避免使用 std::advance(it, step)，虽然它在 CUDA 上可以工作，但与 std::distance 不同。
      it += step;
      // 区分找到下界和上界的决策。
      // 注意，下界是 *it 大于等于 value 的最小索引处的值，或者最后一个位置。
      // 类似地，上界是 *it 大于 value 的最小索引处的值，或者最后一个位置。
      // 当 is_lower = true 且 *it < value 时，我们知道 *it 及其之前的值不能包含下界，
      // 因此将初始迭代器范围从 [first, first + count] 调整为 [first + step + 1, first + count - (step + 1)]，
      // 其中 +1 跳过了刚刚评估的 *it < value 的元素。
      // 当 is_lower = false 时，逻辑类似。
      if (is_lower ? *it < value : value >= *it) {
        first = ++it;
        count -= step + 1;
      }
      else {
        count = step;
      }
    }
    return first;
}

// kernel_t 模板结构体，用于启动特定类型的内核函数
template <template <typename func_t> class kernel_t>
struct KernelLauncher {
  // 启动函数模板，接受 TensorIteratorBase 类型的迭代器和函数对象 f
  template <typename func_t>
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    kernel_t<func_t>::launch(iter, f);
  }
};

// 创建用于值选择交集的张量迭代器
TensorIterator make_value_selection_intersection_iter(
    const Tensor& lhs_values,
    const Tensor& lhs_select_idx,
    const Tensor& rhs_values,
    const Tensor& rhs_select_idx,
    const Tensor& intersection_counts) {
  // lambda 函数 res_values_sizes，返回推断出的结果值尺寸
  const auto res_values_sizes = [&]() -> std::vector<int64_t> {
    auto sizes = infer_size(
        // 保留 nnz 维度
        lhs_values.sizes(),
        // 移除 nnz 维度以实现平滑广播
        rhs_values.sizes().slice(1));
    // 定义一个 lambda 函数，用于计算重塑后的索引张量
    const auto restride_idx = [&res_values](const Tensor& idx) -> Tensor {
        // 创建一个与结果张量相同维度的大小向量，初始化为1
        auto idx_sizes = std::vector<int64_t>(res_values.dim(), 1);
        // 创建一个与结果张量相同维度的步幅向量，初始化为0
        auto idx_strides = std::vector<int64_t>(res_values.dim(), 0);
        // 将第一个维度的大小设置为输入索引张量的元素数目
        idx_sizes[0] = idx.numel();
        // 将第一个维度的步幅设置为1
        idx_strides[0] = 1;
        // 使用输入索引张量的大小和步幅重塑为新的张量
        return idx.as_strided(idx_sizes, idx_strides);
    };
    
    // 定义一个 lambda 函数，用于计算重塑后的数值张量
    const auto restride_values = [&lhs_select_idx](const Tensor& values) -> Tensor {
        // 创建一个与输入数值张量相同维度的大小向量
        auto values_sizes = at::DimVector(values.sizes());
        // 创建一个与输入数值张量相同维度的步幅向量
        auto values_strides = at::DimVector(values.strides());
        // 将第一个维度的大小设置为 lhs_select_idx 的元素数目
        values_sizes[0] = lhs_select_idx.numel();
        // 将第一个维度的步幅设置为0
        values_strides[0] = 0;
        // 使用 lhs_select_idx 的大小和步幅重塑为新的张量
        return values.as_strided(values_sizes, values_strides);
    };
    
    // 创建一个 TensorIteratorConfig 对象并配置
    auto iter = TensorIteratorConfig()
      // 设置不检查内存重叠
      .set_check_mem_overlap(false)
      // 设置不检查所有输入张量的数据类型是否相同
      .check_all_same_dtype(false)
      // 设置不自动调整输出张量的大小
      .resize_outputs(false)
      // 添加结果数值张量作为输出
      .add_owned_output(res_values)
      // 添加重塑后的 lhs_values 作为输入
      .add_owned_input(restride_values(lhs_values))
      // 添加重塑后的 lhs_select_idx 作为输入
      .add_owned_input(restride_idx(lhs_select_idx))
      // 添加重塑后的 rhs_values 作为输入
      .add_owned_input(restride_values(rhs_values))
      // 添加重塑后的 rhs_select_idx 作为输入
      .add_owned_input(restride_idx(rhs_select_idx))
      // 添加重塑后的 intersection_counts 作为输入
      .add_owned_input(restride_idx(intersection_counts))
      // 构建 Tensor 迭代器
      .build();
    
    // 返回配置好的 Tensor 迭代器
    return iter;
// 结束函数模板参数列表定义

template <
  template <typename func_t> class kernel_t,  // 模板模板参数，接受一个模板模板参数 func_t
  typename value_selection_intersection_kernel_t,  // 类型参数，用于交集操作的值选择内核类型
  typename index_t = int64_t,  // 类型参数，默认为 int64_t，用于索引
  int64_t max_static_len = 0>  // 非类型参数，默认为 0，用于最大静态长度
void _sparse_binary_op_intersection_kernel_impl(
    Tensor& res,  // 引用类型参数，结果张量
    const Tensor& x_,  // 引用类型参数，输入张量 x
    const Tensor& y_,  // 引用类型参数，输入张量 y
    const std::vector<int64_t>& broadcasted_shape,  // 引用类型参数，广播形状的向量
    const std::optional<Tensor>& x_hash_opt_ = c10::nullopt,  // 可选类型参数，输入张量 x 的哈希值的可选项，默认为空
    const std::optional<Tensor>& y_hash_opt_ = c10::nullopt,  // 可选类型参数，输入张量 y 的哈希值的可选项，默认为空
    const bool accumulate_matches = true,  // 布尔类型参数，默认为 true，指示是否累积匹配
    const bool distributive_with_sum = true  // 布尔类型参数，默认为 true，指示操作是否与求和分布
) {
  // 公共数据类型检查适用于原地操作时。
  // 这是因为 binary_of_t 生成新值，可能新的值的数据类型不等于 res 的数据类型。
  // 在这种情况下，我们应尽早报错，以避免多余的内核运行。
  const auto common_dtype = at::result_type(x_, y_);
  TORCH_CHECK(canCast(common_dtype, res.scalar_type()),
      "Can't convert result type ", common_dtype,
      " to output ", res.scalar_type());

  using KernelLauncher = KernelLauncher<kernel_t>;  // 使用类型别名 KernelLauncher

  using OptTensor = std::optional<Tensor>;  // 使用类型别名 OptTensor，表示可选的张量

  // 如果操作和求和不是分布的，则需要 coalesce。
  const auto coalesce_if_not_distributive = [distributive_with_sum](const Tensor& t, const OptTensor& t_hash_opt) -> auto {
    // 在这种情况下不需要 coalesce。
    if (distributive_with_sum) {
      return std::make_tuple(t, t_hash_opt);
    } else {
      // 否则进行 coalesce 并强制重新计算哈希。
      return std::make_tuple(t.coalesce(), static_cast<OptTensor>(c10::nullopt));
    }
  };

  Tensor x, y;
  OptTensor x_hash_opt, y_hash_opt;
  std::tie(x, x_hash_opt) = coalesce_if_not_distributive(x_, x_hash_opt_);
  std::tie(y, y_hash_opt) = coalesce_if_not_distributive(y_, y_hash_opt_);

  // 给定稀疏张量 x 和 y，决定哪一个是源张量，哪一个可能是已 coalesce 的张量。
  // 源张量和可能 coalesce 张量的索引被哈希，然后源张量的索引的哈希值在可能 coalesce 张量的索引的哈希值中进行二分查找。
  // 如果可能 coalesce 已经 coalesce，根据哈希方法的性质（见下文），哈希值已经排序，我们可以避免任何显式的排序例程。
  Tensor probably_coalesced, source;
  OptTensor probably_coalesced_indices_hash_opt, source_indices_hash_opt;
  std::tie(probably_coalesced, probably_coalesced_indices_hash_opt, source, source_indices_hash_opt) = [&]() -> auto {
    // 情况 1：x 或 y 中有一个已经 coalesce。
    if ((x.is_coalesced() ^ y.is_coalesced())) {
      return x.is_coalesced()
        ? std::make_tuple(x, x_hash_opt, y, y_hash_opt)
        : std::make_tuple(y, y_hash_opt, x, x_hash_opt);
    }
    // 情况 2：x 和 y 都是已经 coalesce 或者都是未 coalesce。
    // 如果两者都已 coalesce，则在更大的张量中搜索更快。
    // 当两者都未 coalesce 时也是如此。
    // 在这种情况下，选择作为 source 的张量和作为 probably_coalesced 的张量。
    return x.size(0) > y.size(0)
        ? std::make_tuple(x, x_hash_opt, y, y_hash_opt)
        : std::make_tuple(y, y_hash_opt, x, x_hash_opt);
  }();
}
    else {
      // 声明两个张量变量：larger 和 smaller
      Tensor larger, smaller;
      // 声明两个可选的张量变量：larger_hash_opt 和 smaller_hash_opt
      OptTensor larger_hash_opt, smaller_hash_opt;
      // 使用 lambda 表达式将较大和较小的张量及其哈希值分配给对应的变量
      std::tie(larger, larger_hash_opt, smaller, smaller_hash_opt) = [&]() -> auto {
        // 如果 x 的非零元素数量大于等于 y 的非零元素数量，则返回 x 和 y
        // 否则返回 y 和 x
        return x._nnz() >= y._nnz()
          ? std::make_tuple(x, x_hash_opt, y, y_hash_opt)
          : std::make_tuple(y, y_hash_opt, x, x_hash_opt);
      }();

      // 如果在均匀分布下，较大张量中的元素很可能会被访问多次，
      // 最好对其进行合并以提升性能。
      // 获取较大张量的尺寸信息
      const auto larger_sizes = larger.sizes();
      // 计算稀疏维度的元素个数
      const auto sparse_dim_numel = std::accumulate(
          larger_sizes.begin(),
          larger_sizes.begin() + larger.sparse_dim(),
          1,
          std::multiplies<int64_t>());
      
      // 如果非零元素数量大于较大张量形状的稀疏维度之前所有维度形状的乘积，
      // 根据鸽巢原理，至少有一个桶包含 nnz / prod(larger.shape[:sparse_dim]) 个元素。
      // 这提供了交集中最大计数的下界估计。
      // 此条件非常保守，因为我们并未实际检查这种事件是否发生，
      // 尽管在均匀分布下这种情况很可能发生，因为均匀分布具有最高的不确定性（最大化熵）。
      const auto max_count_lower_bound = larger._nnz() / sparse_dim_numel;
      constexpr int64_t MAX_COPIES_PER_THREAD = 50;
      
      // 如果最大计数的下界估计大于每个线程的最大复制数限制，
      // 则强制对较大张量进行合并，并强制重新计算哈希值。
      // 否则保持原样返回张量及其哈希值。
      return max_count_lower_bound > MAX_COPIES_PER_THREAD
        ? std::make_tuple(larger.coalesce(), static_cast<OptTensor>(c10::nullopt), smaller, smaller_hash_opt)
        : std::make_tuple(larger, larger_hash_opt, smaller, smaller_hash_opt);
    }
  }();

  // 使用的哈希函数将一个 d 维索引映射到一个线性偏移量，
  // 该线性偏移量足以容纳形状为 broadcasted_shape(x.shape, y.shape) 的密集张量。
  // 即 idx -> \sum_{i = 0}^d idx[i] * hash_coeffs[i]，
  // 其中 hash_coeffs 是形状为 broadcasted_shape(x.shape, y.shape) 的连续张量的步长。
  // 假设以下维度排序：最右边的维度是最快变化的维度，最左边的是最慢变化的维度，
  // 这在 hash_coeffs 的定义中是隐含的，
  // 可以证明该哈希函数实际上是双射的，因此是一个完美哈希函数（永不冲突）。

  // 针对 Tensor 类型需要拥有的存储空间。
  const auto hash_coeffs_storage = [&]() -> auto {
    // 创建广播形状中稀疏维度的形状向量
    const auto broadcasted_sparse_dim_shape = std::vector<int64_t>(
      broadcasted_shape.begin(),
      broadcasted_shape.begin() + probably_coalesced.sparse_dim()
    );
    // 计算连续张量的步长
    auto strides = c10::contiguous_strides(broadcasted_sparse_dim_shape);
  return at::sparse::TensorGeometryHolder<max_static_len>(strides, strides, probably_coalesced.options());
}();

const auto hash_coeffs = std::get<0>(*hash_coeffs_storage);

const auto nnz_arange = at::arange(
    std::max(probably_coalesced._nnz(), source._nnz()),
    source._indices().options());
const auto probably_coalesced_nnz_arange = nnz_arange.narrow(-1, 0, probably_coalesced._nnz());

// non-const because of gcc-5/clang-5 issues
auto sparse_dim = probably_coalesced.sparse_dim();

// Apply the hash function to probably_coalesced.indices
const auto probably_coalesced_indices_hash = [&]() -> Tensor {
  // probably_coalesced is coalesced and hash provided? Reuse it!
  if (probably_coalesced_indices_hash_opt.has_value()) {
    return (*probably_coalesced_indices_hash_opt).contiguous();
  }

  const auto indices = probably_coalesced._indices();
  // non-const because of gcc-5/clang-5 issues
  auto indices_dim_stride = indices.stride(0);
  auto indices_nnz_stride = indices.stride(1);

  auto hash = at::empty({probably_coalesced._nnz()}, indices.options().dtype(kLong));

  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(hash)
    .add_input(probably_coalesced_nnz_arange)
    .build();

  {
    const auto* RESTRICT ptr_indices = indices.const_data_ptr<index_t>();

    // Launching a kernel to compute hashes for sparse indices
    KernelLauncher::launch(iter,
        // NOTE: capture by value required by CUDA
        [=] FUNCAPI (index_t nnz_idx) -> int64_t {
        int64_t hash = 0;
        if (!ptr_indices) {
          return hash;
        }
        const auto* RESTRICT ptr_indices_dim = ptr_indices + nnz_idx * indices_nnz_stride;
        for (int64_t dim = 0; dim < sparse_dim; ++dim) {
          const auto dim_hash_coeff = hash_coeffs[dim];
          const auto dim_index = ptr_indices_dim[dim * indices_dim_stride];
          hash += dim_index * dim_hash_coeff;
        }
        return hash;
    });
  }

  return hash;
}();

// Now that we have hash values of probably_coalesced.indices,
// we need to decide whether they need to get sorted.
Tensor sorted_hash, argsort_hash;
std::tie(sorted_hash, argsort_hash) = [&]() -> std::tuple<Tensor, Tensor> {
  if (probably_coalesced.is_coalesced()) {
    // NOTE: argsort.dtype == nnz_arange.dtype
    // Directly use probably_coalesced_indices_hash if already coalesced
    const auto argsort = nnz_arange.narrow(-1, 0, probably_coalesced._nnz());
    return std::make_tuple(probably_coalesced_indices_hash, argsort);
  } else {
    // NOTE: we want argsort.dtype == nnz_arange.dtype,
    // but sort() produces indices of type int64_t,
    // so we convert to nnz_arange.dtype to avoid issues
    // with pointer types in the kernels below.
    // Sort probably_coalesced_indices_hash and convert argsort to nnz_arange.dtype
    Tensor sorted, argsort;
    std::tie(sorted, argsort) = probably_coalesced_indices_hash.sort();
    return std::make_tuple(sorted, argsort.to(nnz_arange.scalar_type()));
  }
}();
    }
  }();

  // 执行哈希交集计算。
  // 定义 s_hash = hash(source.indices),
  //     pc_hash = hash(probably_coalesced.indices)，然后
  // 对于 i = 0, ..., len(s_hash) - 1:
  //     lb = <argsort_hash 中为 s_hash[i] 的值的下界的索引>,
  //     up = <argsort_hash 中为 s_hash[i] 的值的上界的索引>,
  //     intersection_count[i] = up - lb
  //     intersection_first_idx[i] = lb.
  //
  // intersection_count 和 intersection_first_idx 用于形成选择交集值的索引。
  Tensor intersection_count, intersection_first_idx;
  // 使用 lambda 函数返回两个 Tensor 类型的对象
  std::tie(intersection_count, intersection_first_idx) = [&]() -> std::tuple<Tensor, Tensor> {
    // 获取 source 的非零元素数量
    const auto source_nnz = source._nnz();
    // 创建一个二维 Tensor 用于存储交集计算的中间结果
    auto intersection_buffer = at::empty({2, source_nnz}, sorted_hash.options());
    // 分别选择二维 Tensor 的第一行和第二行作为 intersection_count 和 intersection_first_idx
    auto intersection_count = intersection_buffer.select(0, 0);
    auto intersection_first_idx = intersection_buffer.select(0, 1);

    // 获取 source 的索引和 nnz_arange
    const auto source_indices = source._indices();
    const auto source_arange = nnz_arange.narrow(-1, 0, source_nnz);
    // 获取 source_indices 的步长
    auto indices_dim_stride = source_indices.stride(0);
    auto indices_nnz_stride = source_indices.stride(1);
    // 创建一个临时 Tensor，以解决 gcc-5/clang-5 的问题
    auto dummy = at::empty({1}, source_arange.options());

    // 如果 source_indices_hash_opt 有值，则使用它；否则创建一个空的 Tensor
    auto hash = source_indices_hash_opt.has_value()
      ? (*source_indices_hash_opt).contiguous()
      : at::empty({0}, probably_coalesced._indices().options().dtype(kLong));
    // 获取 hash 数据指针
    const auto* RESTRICT hash_ptr = source_indices_hash_opt.has_value()
      ? hash.data_ptr<int64_t>()
      : nullptr;

    // 配置 Tensor 迭代器，设置检查内存重叠为 false，并添加相应的输入输出
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .add_owned_output(dummy.expand_as(source_arange))
      .add_input(source_arange)
      .build();
    {
      // 获取源稀疏张量的索引数据指针，限定为常量指针
      const auto* RESTRICT ptr_indices = source_indices.const_data_ptr<index_t>();
      // 获取排序后哈希表的数据指针，限定为常量指针，类型为 int64_t
      const auto* RESTRICT ptr_sorted_hash = sorted_hash.const_data_ptr<int64_t>();
      // 获取排序后哈希表的长度
      const auto sorted_hash_len = sorted_hash.numel();
      // 获取交集计数的数据指针，限定为 int64_t 类型
      auto* RESTRICT ptr_intersection_count = intersection_count.data_ptr<int64_t>();
      // 获取交集首索引的数据指针，限定为 int64_t 类型
      auto* RESTRICT ptr_intersection_first_idx = intersection_first_idx.data_ptr<int64_t>();
    
      // 将哈希计算与哈希交集融合
      KernelLauncher::launch(iter,
          // 注意：CUDA 需要按值捕获
          [=] FUNCAPI (index_t nnz_idx) -> index_t {
          int64_t hash = 0;
          // 如果有哈希指针，则直接取出哈希值
          if (hash_ptr) {
            hash = hash_ptr[nnz_idx];
          } else if (sparse_dim) {
            // 计算哈希值
            const auto* RESTRICT ptr_indices_dim = ptr_indices + nnz_idx * indices_nnz_stride;
            for (int64_t dim = 0; dim < sparse_dim; ++dim) {
              const auto dim_hash_coeff = hash_coeffs[dim];
              const auto dim_index = ptr_indices_dim[dim * indices_dim_stride];
              hash += dim_index * dim_hash_coeff;
            }
          }
    
          // 执行哈希值的交集计算
          // 查找哈希值的下界
          const auto* RESTRICT lb = find_bound<const int64_t*, int64_t, /*is_lower=*/true>(
              ptr_sorted_hash,
              ptr_sorted_hash + sorted_hash_len,
              hash
          );
    
          // 查找哈希值的上界
          const auto* RESTRICT ub = find_bound<const int64_t*, int64_t, /*is_lower=*/false>(
              ptr_sorted_hash,
              ptr_sorted_hash + sorted_hash_len,
              hash
          );
    
          // 计算交集数量
          ptr_intersection_count[nnz_idx] = ub - lb;
          // 计算交集的首索引位置
          ptr_intersection_first_idx[nnz_idx] = lb - ptr_sorted_hash;
    
          return 0;
      });
    }
    
    // 返回交集计数和交集首索引的元组
    return std::make_tuple(intersection_count, intersection_first_idx);
    }();
    
    // 克隆源张量的索引
    const auto res_indices = source._indices().clone();
    // 获取二元操作结果的数据类型
    const auto binary_op_res_dtype = at::result_type(source._values(), probably_coalesced._values());
    // 应用值选择交集核函数
    const auto res_values = value_selection_intersection_kernel_t::apply(
        source._values().to(binary_op_res_dtype),
        nnz_arange.narrow(-1, 0, source._nnz()),
        probably_coalesced._values().to(binary_op_res_dtype),
        intersection_first_idx.to(nnz_arange.scalar_type()),
        intersection_count,
        argsort_hash,
        accumulate_matches).to(res.scalar_type());
    // 获取结果的稀疏维度
    const auto res_sparse_dim = source.sparse_dim();
    // 获取结果的密集维度
    const auto res_dense_dim = source.dense_dim();
    // 获取广播后的形状
    const auto& res_shape = broadcasted_shape;
    // 获取结果的非零元素数量
    const auto res_nnz = source._nnz();
    
    // 获取结果的稀疏实现指针
    auto* res_sparse_impl = get_sparse_impl(res);
    // 调整稀疏实现的大小并设置索引和值
    res_sparse_impl->raw_resize_(res_sparse_dim, res_dense_dim, res_shape);
    res_sparse_impl->set_indices_and_values_unsafe(res_indices, res_values);
    res_sparse_impl->set_nnz_and_narrow(res_nnz);
    // 设置结果为合并状态
    res._coalesced_(source.is_coalesced());
// 匿名命名空间，用于限定不同编译单元中具有相同名称的符号的作用域
namespace {

// 实现稀疏张量的二元操作的核心函数，将结果写入 res
template <
  // 模板参数 kernel_t 是一个模板模板参数，表示核函数
  template <typename func_t> class kernel_t,
  // value_selection_intersection_kernel_t 是一个类型参数，表示值选择交集核函数
  typename value_selection_intersection_kernel_t>
void _sparse_binary_op_intersection_kernel_out(
    // 输出张量，用于存放操作的结果
    Tensor& res,
    // 第一个输入稀疏张量 x
    const Tensor& x,
    // 第二个输入稀疏张量 y
    const Tensor& y,
    // 可选参数，第一个输入张量的哈希值张量
    const std::optional<Tensor>& x_hash_opt = c10::nullopt,
    // 可选参数，第二个输入张量的哈希值张量
    const std::optional<Tensor>& y_hash_opt = c10::nullopt,
    // 是否与求和分配相关，默认为 true
    const bool distributive_with_sum = true
) {
  // 检查输入张量 x 和 y 是否为稀疏张量，并且维度、稀疏维度及大小匹配
  TORCH_CHECK(
      (x.is_sparse() && y.is_sparse())
      && (x.dim() == y.dim()) && (x.sparse_dim() == y.sparse_dim())
      && (x.sizes().slice(0, x.sparse_dim()) == y.sizes().slice(0, y.sparse_dim())),
      NAME, "(): expects sparse inputs with equal dimensionality, ",
      "number of sparse dimensions, and shape of sparse dimensions");
  // 检查输入张量 x 和 y 的索引类型是否相同（即 long 或 int）
  TORCH_CHECK(
      x._indices().scalar_type() == y._indices().scalar_type(),
      NAME, "(): expects inputs' indices to be of the same dtype (i.e. long or int)");

  // 函数用于检查哈希值张量的有效性
  const auto check_hash_validity = [](const Tensor& t, const std::optional<Tensor>& t_hash_opt) {
    if (!t_hash_opt.has_value()) {
      return;
    }
    const auto &t_hash = *t_hash_opt;
    // 检查哈希值张量 t_hash 的维度、数据类型和大小是否匹配对应的稀疏张量 t 的索引
    TORCH_INTERNAL_ASSERT(
        t_hash.dim() == 1 && t_hash.scalar_type() == kLong && t_hash.size(-1) == t._indices().size(-1),
        NAME, "(): explicit hash values need to be a 1-dim Long tensor with the ",
        "NSE matching that of the corresponding sparse tensor.");
  };

  // 分别检查输入张量 x 和 y 的哈希值的有效性
  check_hash_validity(x, x_hash_opt);
  check_hash_validity(y, y_hash_opt);

  // 推断广播后的形状
  const auto broadcasted_shape = infer_size(x.sizes(), y.sizes());

  // 最大稀疏维度的限制
  constexpr int64_t max_sparse_dims = 8;

  // COO 索引目前仅支持 64 位整数
  using index_t = int64_t;

  // 根据最大稀疏维度选择具体的核函数实现
  if (max_sparse_dims > x.sparse_dim()) {
    // 调用具体的稀疏二元操作核心实现，传递给模板参数 index_t 为 8
    _sparse_binary_op_intersection_kernel_impl<
      kernel_t, value_selection_intersection_kernel_t, index_t, 8>(
        res, x, y, broadcasted_shape, x_hash_opt, y_hash_opt, distributive_with_sum);
  } else {
    // 调用具体的稀疏二元操作核心实现，传递默认的 index_t
    _sparse_binary_op_intersection_kernel_impl<
      kernel_t, value_selection_intersection_kernel_t, index_t>(
        res, x, y, broadcasted_shape, x_hash_opt, y_hash_opt, distributive_with_sum);
  }
}

} // 匿名命名空间结束

} // namespace at::native 结束
```