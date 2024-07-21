# `.\pytorch\aten\src\ATen\native\EmbeddingBag.cpp`

```py
// 定义宏，仅允许使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的分发、并行、张量操作、子类相关工具、CPU 向量化等头文件
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>

// 包含 ATen 库的 CPU BLAS、非符号广播相关头文件
#include <ATen/native/CPUBlas.h>
#include <ATen/native/NonSymbolicBC.h>

// 包含 C10 库的循环范围、半精度数据类型定义
#include <c10/util/irange.h>
#include <c10/util/Half.h>

// 根据是否使用 FBGEMM 决定引入的头文件
#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmConvert.h>
#else
#include <caffe2/perfkernels/embedding_lookup_idx.h>
#endif

// 包含标准库头文件
#include <algorithm>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

// 根据条件引入不同的 ATen 操作头文件集合
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_embedding_bag.h>
#include <ATen/ops/_embedding_bag_backward_native.h>
#include <ATen/ops/_embedding_bag_dense_backward.h>
#include <ATen/ops/_embedding_bag_dense_backward_native.h>
#include <ATen/ops/_embedding_bag_forward_only.h>
#include <ATen/ops/_embedding_bag_forward_only_native.h>
#include <ATen/ops/_embedding_bag_native.h>
#include <ATen/ops/_embedding_bag_per_sample_weights_backward_native.h>
#include <ATen/ops/_embedding_bag_sparse_backward.h>
#include <ATen/ops/_embedding_bag_sparse_backward_native.h>
#include <ATen/ops/embedding_backward_native.h>
#include <ATen/ops/embedding_bag_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/max.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros.h>
#endif

// 命名空间定义
namespace {
  const int MODE_SUM = 0;  // 求和模式常量
  const int MODE_MEAN = 1; // 均值模式常量
  const int MODE_MAX = 2;  // 最大值模式常量
}

namespace at::native {

// scalar_t 类型的 dot_impl 模板函数声明
template<typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);

// 静态函数，根据 offsets 张量生成 offset2bag 张量
static void make_offset2bag(const Tensor &offsets, Tensor& offset2bag) {
  offset2bag.index_add_(
      0, offsets, at::ones_like(offsets, LEGACY_CONTIGUOUS_MEMORY_FORMAT)); // 使用 offsets 张量更新 offset2bag 张量，添加 [1 0 1 0 1]
  offset2bag[0] -= 1;                     // 将 offset2bag 第一个元素减 1，变为 [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0, offset2bag.scalar_type());     // 对 offset2bag 张量进行累积和运算，类型与 offset2bag 一致，得到 [0 0 1 1 2]
}

// 匿名命名空间，用于实现一些局部函数

// 根据 indices 和 offsets 张量类型升级，并返回升级后的 indices 和 offsets
std::pair<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>> promoteIndicesAndOffsets(
    const Tensor& indices,
    const Tensor& offsets) {
  const auto commonType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {
      indices.scalar_type() == commonType ? c10::MaybeOwned<Tensor>::borrowed(indices)
                                          : c10::MaybeOwned<Tensor>::owned(indices.toType(commonType)),
      offsets.scalar_type() == commonType ? c10::MaybeOwned<Tensor>::borrowed(offsets)
                                          : c10::MaybeOwned<Tensor>::owned(offsets.toType(commonType))};
}

// 判断是否可以使用快速实现的 index_select_add，只有在特定条件下才适用
template<typename index_t>
// 检查是否可以使用快速路径索引选择和累加的实现，仅在满足特定条件时适用
template <typename data_t, typename index_t>
static typename std::enable_if<std::is_same<data_t, double>::value, void>::type
index_select_add(
    const Tensor& select_indices,          // 用作索引的选择张量
    const Tensor& add_indices,             // 用作索引的累加张量
    const Tensor& src,                     // 源张量，包含嵌入向量
    Tensor& output,                        // 输出张量，将累加结果存储在这里
    const Tensor& /*offsets*/,             // 偏移量（未使用）
    bool /*include_last_offset*/,          // 是否包括最后一个偏移量（未使用）
    Tensor& bag_size,                      // 每个输出袋的大小
    index_t padding_idx,                   // 填充索引，用于跳过填充位置的计算
    _EmbeddingBagKernelCache* /* fbgemm_kernel_cache */  // 缓存用于加速的内核信息（未使用）
) {
  // 检查选择张量和累加张量的元素数量是否相等
  TORCH_CHECK(select_indices.numel() == add_indices.numel());
  // 获取累加张量和选择张量的索引数据指针
  auto* add_indices_data = add_indices.const_data_ptr<index_t>();
  auto* select_indices_data = select_indices.const_data_ptr<index_t>();
  // 获取源张量和输出张量的数据指针
  auto* src_data = src.const_data_ptr<data_t>();
  auto* output_data = output.data_ptr<data_t>();
  // 初始化袋大小数据指针为空，如果定义了袋大小张量则获取其数据指针
  index_t* bag_size_data = nullptr;
  if (bag_size.defined()) {
    bag_size_data = bag_size.data_ptr<index_t>();
  }
  // 获取累加张量/选择张量的元素数量
  auto numel = add_indices.numel();
  // 获取源张量的第二个维度大小（假设是嵌入向量的维度）
  int64_t ddim = src.size(1);
  // 获取源张量的第一维度大小（假设是词汇表的大小）
  auto vocab_size = src.size(0);
  // 获取源张量和输出张量的步长
  auto src_stride0 = src.strides()[0];
  auto src_stride1 = src.strides()[1];
  auto output_stride0 = output.strides()[0];
  auto output_stride1 = output.strides()[1];

  // 遍历累加张量/选择张量中的索引，执行索引选择和累加操作
  for (const auto i : c10::irange(numel)) {
    // 如果选择的索引等于填充索引，则跳过当前迭代，不参与累加
    auto idx = select_indices_data[i];
    TORCH_CHECK(
        idx >= 0 && idx < vocab_size,
        "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
        idx);
    // 如果索引不等于填充索引
    if (idx != padding_idx) {
      // 调用 CPU 端 BLAS 库的 axpy 函数，将 src_data 中的数据加到 output_data 中
      at::native::cpublas::axpy<data_t>(ddim, 1,
              // 计算源数据的地址偏移，并使用对应的步长进行访问
              src_data + src_stride0 * idx, src_stride1,
              // 将结果数据的地址偏移，并使用对应的步长进行访问
              output_data + output_stride0 * add_indices_data[i], output_stride1);
    } else if (bag_size.defined()) {
      // 如果定义了 bag_size 变量
      // 减少 bag_size_data 中指定索引的值，以反映该索引是填充的
      // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
      bag_size_data[add_indices_data[i]]--;
    }
  }
} // 结束函数 fbgemm_spmdm_report_error_

namespace {
// 匿名命名空间，用于定义局部的函数或变量，限制其作用域
template <typename index_t>
void fbgemm_spmdm_report_error_(
    int64_t output_size,
    int index_size,
    int64_t N,
    const index_t* offsets,
    const index_t* indices) {
  // 遍历输出的每一行
  for (const auto m : c10::irange(output_size)) {
    // 遍历当前行 m 对应的索引
    for (index_t i = offsets[m]; i < offsets[m + 1]; ++i) {
      // 检查索引 i 是否在有效范围内
      TORCH_CHECK(i < index_size);
      index_t idx = indices[i];
      // 检查索引值 idx 是否在有效范围 [0, N)
      TORCH_CHECK(
          0 <= idx && idx < N,
          "Index ",
          i,
          " of input takes value ",
          idx,
          " which is not in the valid range [0, ",
          N,
          ")");
    }
  }
  // 最后一个偏移值应等于索引数组的大小，否则抛出异常
  TORCH_CHECK(
      offsets[output_size] == index_size,
      "Your input appears to be incorrect: the last offset value should be "
       "the size of the indices tensor, but it seems not to be the case.");
}
} // 结束匿名命名空间

template <typename data_t, typename index_t>
typename std::enable_if<
    std::is_same<data_t, at::Half>::value ||
        std::is_same<data_t, at::BFloat16>::value,
    void>::type
index_select_add(
    const Tensor& select_indices,
    const Tensor& add_indices,
    const Tensor& src,
    Tensor& output,
    const Tensor& offsets,
    bool include_last_offset,
    Tensor& bag_size,
    index_t padding_idx,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  // 获取 src 的第二维大小
  int64_t ddim = src.size(1);
  // 获取 select_indices 的数据指针
  auto* select_indices_data = select_indices.const_data_ptr<index_t>();
  // 获取 output 的数据指针
  auto* output_data = output.data_ptr<data_t>();

  // 如果满足快速路径的条件，执行以下操作
  if (is_fast_path_index_select(src, output, padding_idx)) {
    // 将 src 强制转换为连续的张量
    auto src_contig = src.contiguous();
    // 获取 src 的数据指针
    auto* src_data = src_contig.const_data_ptr<data_t>();
    // 初始化输出大小为 offsets 的元素数量减一
    int64_t output_size = offsets.numel() - 1;
    // 获取 offsets 的数据指针
    auto* offsets_data = offsets.const_data_ptr<index_t>();
    // 如果包括最后一个偏移量，保持不变
    // 否则调整输出大小，并创建包含最后一个偏移量的 offsets_include_last
    std::vector<index_t> offsets_include_last;
    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      if (offsets.numel() > 0) {
        std::memcpy(
            offsets_include_last.data(),
            offsets.const_data_ptr<index_t>(),
            sizeof(index_t) * offsets.numel());
      }
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }
#if defined(USE_FBGEMM)
    // 如果使用 FBGEMM，确定是否是 BF16 类型的张量
    constexpr bool isbf16 = std::is_same_v<data_t, at::Half> ? false : true;
    // 获取 FBGEMM 的缓存，或者生成对应的 SpMDM 核函数
    auto kernel_16bit_index_t = fbgemm_kernel_cache
        ? fbgemm_kernel_cache
              ->getCallback</* has_weight */ false, index_t, uint16_t>(ddim)
        : fbgemm::GenerateEmbeddingSpMDM<uint16_t, index_t, index_t, uint16_t>(
              /* block_size */ ddim,
              /* has_weight */ false,
              /* normalize_by_lengths */ false,
              /* prefetch */ 16,
              /* is_weight_positional */ false,
              /* use_offsets */ true,
              /* is_bf16_out */ isbf16,
              /* is_bf16_in */ isbf16);
    # 使用 ATen 的并行函数进行并行操作，处理从 0 到 output_size 的索引范围
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
          # 调用 kernel_16bit_index_t 函数处理指定范围的数据
          bool success = kernel_16bit_index_t(
              /* output_size */ end_idx - start_idx,  # 输出数据的大小
              /* index_size */ offsets_data[end_idx] - offsets_data[start_idx],  # 索引数据的大小
              /* data_size */ src.size(0),  # 输入数据的大小
              /* input */ reinterpret_cast<const uint16_t*>(src_data),  # 输入数据的指针
              /* indices */ select_indices_data + offsets_data[start_idx],  # 索引数据的起始指针
              /* offsets_or_lengths */ offsets_data + start_idx,  # 偏移数据的起始指针
              /* weights */ nullptr,  # 权重数据，此处为空指针
              /* output */ reinterpret_cast<uint16_t*>(output_data + start_idx * ddim));  # 输出数据的起始指针
          # 如果执行失败，则报告错误并打印相关信息
          if (!success) {
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,  # 输出数据的大小
                offsets_data[end_idx] - offsets_data[start_idx],  # 索引数据的大小
                src.size(0),  # 输入数据的大小
                offsets_data + start_idx,  # 偏移数据的起始指针
                select_indices_data + offsets_data[start_idx]);  # 索引数据的起始指针
          }
        });
  } else {
    // Ensure the number of select_indices matches add_indices
    TORCH_CHECK(select_indices.numel() == add_indices.numel());
    // Obtain constant pointers to source data and additional indices data
    auto* src_data = src.const_data_ptr<data_t>();
    auto* add_indices_data = add_indices.const_data_ptr<index_t>();
    // Initialize pointer to bag size data; set to nullptr initially
    index_t* bag_size_data = nullptr;
    // Assign bag_size_data pointer if bag_size tensor is defined
    if (bag_size.defined()) {
      bag_size_data = bag_size.data_ptr<index_t>();
    }
    // Determine vocabulary size from source tensor
    auto vocab_size = src.size(0);
    // Obtain strides of source and output tensors
    auto src_stride0 = src.strides()[0];
    auto src_stride1 = src.strides()[1];
    auto output_stride0 = output.strides()[0];
    auto output_stride1 = output.strides()[1];
    // Determine number of elements in add_indices tensor
    auto numel = add_indices.numel();

    // Create an empty tensor for temporary storage of source data in FP32 format
    Tensor src_fp32 = at::empty({ddim}, src.options().dtype(at::kFloat));
    auto* src_data_fp32 = src_fp32.mutable_data_ptr<float>();

    // Initialize the intermediate output buffer to be 0.
    Tensor output_fp32 = at::zeros({output_size, ddim}, output.options().dtype(at::kFloat));
    auto* output_data_fp32 = output_fp32.data_ptr<float>();
    // Define vectorized types for BFloat16 and float operations
    using bVec = vec::Vectorized<BFloat16>;
    using fVec = vec::Vectorized<float>;
    // Perform parallel processing over the range [0, output_size)
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
          // Invoke embedding lookup function with specified parameters
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/nullptr,
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data_fp32 + start_idx * ddim);
          // Iterate over indices from start_idx to end_idx
          for (int64_t i = start_idx; i < end_idx; i++) {
            // Convert FP32 intermediate buffer result back to 16 bit for output dtype
            if constexpr (std::is_same<data_t, at::Half>::value) {
              // FP16 conversion
              for (const auto d : c10::irange(ddim)) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            } else {
              // BF16 conversion
              int64_t d = 0;
              // Process data in vectorized batches of bVec::size()
              for (; d < ddim - (ddim % bVec::size()); d += bVec::size()) {
                fVec temp_fp32_0 = fVec::loadu(output_data_fp32 + ddim * i + d);
                fVec temp_fp32_1 =
                    fVec::loadu(output_data_fp32 + ddim * i + d + fVec::size());
                convert_float_bfloat16(temp_fp32_0, temp_fp32_1)
                    .store(output_data + i * ddim + d);
              }
              // Process remaining elements not fitting into vectorized batches
              for (; d < ddim; d++) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            }
          }
        });
#endif
  }
    // 创建一个与 output 相同大小的全零张量，数据类型为 float
    Tensor output_fp32 =
        at::zeros({output.size(0), ddim}, output.options().dtype(at::kFloat));
    // 获取 output_fp32 的指针，以便后续操作
    auto* output_data_fp32 = output_fp32.data_ptr<float>();

    // 遍历每个索引 i，其中 numel 是索引的总数目
    for (const auto i : c10::irange(numel)) {
      // 如果索引等于 padding_idx，则跳过该索引，不参与计算
      auto idx = select_indices_data[i];
      // 检查 idx 是否在有效范围内，如果不在则抛出错误信息
      TORCH_CHECK(
          idx >= 0 && idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          idx);
      if (idx != padding_idx) {
        // 将 src_data + src_stride0 * idx 复制到 src_data_fp32 中
        for (const auto d : c10::irange(ddim)) {
          src_data_fp32[d] = static_cast<float>(
              (src_data + src_stride0 * idx)[d * src_stride1]);
        }
        // 使用 cpublas 库的 axpy 函数将 src_data_fp32 的值加到 output_data_fp32 的指定位置上
        at::native::cpublas::axpy<float>(
            ddim,
            1,
            src_data_fp32,
            1,
            output_data_fp32 + ddim * add_indices_data[i],
            1);

      } else if (bag_size.defined()) {
        // 如果定义了 bag_size，将对应位置的 bag_size_data 减一，表示该索引是填充的
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        bag_size_data[add_indices_data[i]]--;
      }
    }

    // 遍历每个输出的元素 i
    for (const auto i : c10::irange(output.size(0))) {
      // 将 FP32 类型的中间缓冲区结果转换回输出的 16 位数据类型
      for (const auto d : c10::irange(ddim)) {
        (output_data + output_stride0 * i)[d * output_stride1] =
            static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
      }
    }
  // 如果数据类型是 float，使用 enable_if 条件语句确保函数仅支持该数据类型
  template<typename data_t, typename index_t>
  typename std::enable_if<std::is_same<data_t, float>::value, void>::type
  // 索引选择与加法操作，接受选择索引、添加索引、源张量、输出张量、偏移量、是否包括最后一个偏移量、包大小张量、填充索引、缓存对象作为参数
  index_select_add(const Tensor &select_indices,
                                 const Tensor &add_indices,
                                 const Tensor &src,
                                 Tensor &output,
                                 const Tensor& offsets,
                                 bool include_last_offset,
                                 Tensor &bag_size,
                                 index_t padding_idx,
                                 _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
    // 获取源张量的第二维大小
    int64_t ddim = src.size(1);
    // 获取选择索引数据的指针
    auto* select_indices_data = select_indices.const_data_ptr<index_t>();
    // 获取输出张量数据的指针（假设为 float 类型）
    auto* output_data = output.data_ptr<float>();

    // 如果满足快速路径索引选择的条件
    if (is_fast_path_index_select(src, output, padding_idx)) {
      // 创建源张量的连续副本
      auto src_contig = src.contiguous();
      // 获取连续副本的数据指针
      auto* src_data = src_contig.const_data_ptr<float>();
      // 输出大小初始化为偏移量张量元素数减一
      int64_t output_size = offsets.numel() - 1;
      // 获取偏移量张量数据指针
      auto* offsets_data = offsets.const_data_ptr<index_t>();
      // 创建包含最后一个偏移量的偏移量向量（如果需要）
      std::vector<index_t> offsets_include_last;

      // 如果包含最后一个偏移量
      if (include_last_offset) {
        output_size = offsets.numel() - 1;
      } else {
        output_size = offsets.numel();
        // 调整 offsets_include_last 大小以容纳额外的偏移量
        offsets_include_last.resize(offsets.numel() + 1);
        // 如果偏移量张量元素数大于零，复制数据到 offsets_include_last
        if (offsets.numel() > 0) {
          std::memcpy(
              offsets_include_last.data(),
              offsets.const_data_ptr<index_t>(),
              sizeof(index_t) * offsets.numel());
        }
        // 将最后一个偏移量设置为选择索引的元素数
        offsets_include_last[offsets.numel()] = select_indices.numel();
        // 更新偏移量数据指针为 offsets_include_last 的数据指针
        offsets_data = offsets_include_last.data();
      }

#ifdef USE_FBGEMM
      // 如果定义了 USE_FBGEMM 宏
      auto kernel_fp32_index_t =
        fbgemm_kernel_cache ?
        // 如果缓存不为空，获取缓存的回调函数指针
        fbgemm_kernel_cache->getCallback</* has_weight */ false, index_t, float>(ddim) :
        // 否则生成新的嵌入式稀疏矩阵乘法的 FP32 版本的函数指针
        fbgemm::GenerateEmbeddingSpMDM<float, index_t, index_t>(
          /* block_size */ddim,
          /* has_weight */false,
          /* normalize_by_lengths */false,
          /* prefetch */16,
          /* is_weight_positional */false,
          /* use_offsets */true
        );
#endif
      // 并行处理函数，从 start_idx 到 end_idx
      at::parallel_for(
          0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
#ifdef USE_FBGEMM
            // 调用 FP32 版本的嵌入式稀疏矩阵乘法函数
            bool success = kernel_fp32_index_t(
              /* output_size */end_idx - start_idx,
              /* index_size */offsets_data[end_idx] - offsets_data[start_idx],
              /* data_size */src.size(0),
              /* input */src_data,
              /* indices */select_indices_data + offsets_data[start_idx],
              /* offsets_or_lengths */offsets_data + start_idx,
              /* weights */nullptr,
              /* output */output_data + start_idx * ddim);
            // 如果执行失败，报告错误
            if (!success) {
              fbgemm_spmdm_report_error_(
                  end_idx - start_idx,
                  offsets_data[end_idx] - offsets_data[start_idx],
                  src.size(0),
                  offsets_data + start_idx,
                  select_indices_data + offsets_data[start_idx]);
            }
#else
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/nullptr,
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data + start_idx * ddim);
#endif
        });
  } else {
    // 断言确保 select_indices 和 add_indices 的元素数量相同
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    // 获取源数据的指针
    auto* src_data = src.const_data_ptr<float>();
    // 获取 add_indices 的索引数据指针
    auto* add_indices_data = add_indices.const_data_ptr<index_t>();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 初始化 bag_size_data 为 nullptr
    index_t* bag_size_data = nullptr;
    // 如果定义了 bag_size，则获取其数据指针
    if (bag_size.defined()) {
      bag_size_data = bag_size.data_ptr<index_t>();
    }
    // 获取词汇表大小
    auto vocab_size = src.size(0);
    // 获取源张量的步长
    auto src_stride0 = src.strides()[0];
    auto src_stride1 = src.strides()[1];
    // 获取输出张量的步长
    auto output_stride0 = output.strides()[0];
    auto output_stride1 = output.strides()[1];
    // 获取 add_indices 的元素数量
    auto numel = add_indices.numel();
    // 遍历 add_indices 的每个索引
    for (const auto i : c10::irange(numel)) {
      // 跳过等于 padding_idx 的索引，以便不包含在减少操作中
      auto idx = select_indices_data[i];
      // 检查 idx 是否在有效范围内
      TORCH_CHECK(
          idx >= 0 && idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          idx);
      // 如果 idx 不是 padding_idx
      if (idx != padding_idx) {
        // 使用 BLAS 函数 axpy 将 src_data 中的数据按比例加到 output_data 中
        at::native::cpublas::axpy<float>(
            ddim,
            1,
            src_data + src_stride0 * idx,
            src_stride1,
            output_data + output_stride0 * add_indices_data[i],
            output_stride1);
      } else if (bag_size.defined()) {
        // 如果定义了 bag_size，则减少对应索引的 bag_size
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        bag_size_data[add_indices_data[i]]--;
      }
    }
  }
}

// This function fuses the following three fns:
// index_select (using select_indices as the index)
// mul (scaling by per_sample_weights)
// index_add (using add_indices as the index)
template <typename data_t, typename index_t>
// 仅当 data_t 是 double 类型时，启用此函数
static typename std::enable_if<std::is_same<data_t, double>::value, void>::type
index_select_scale_add(
    const Tensor& select_indices,
    const Tensor& add_indices,
    const Tensor& scale,
    const Tensor& src,
    Tensor& output,
    const Tensor& /*offsets*/,
    bool /*include_last_offset*/,
    Tensor& bag_size,
    index_t padding_idx,
    // 用于更新嵌入袋（embedding bag）操作中的输出张量，考虑了选择和添加索引
    _EmbeddingBagKernelCache* /* fbgemm_kernel_cache */) {
      // 断言：选择索引和添加索引的元素数量应该相等
      AT_ASSERT(select_indices.numel() == add_indices.numel());
      // 获取添加索引的数据指针
      auto* add_indices_data = add_indices.const_data_ptr<index_t>();
      // 获取选择索引的数据指针
      auto* select_indices_data = select_indices.const_data_ptr<index_t>();
      // 获取源数据的指针
      auto* src_data = src.const_data_ptr<data_t>();
      // 获取输出数据的指针
      auto* output_data = output.data_ptr<data_t>();
      // 初始化袋大小数据指针为空
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      index_t* bag_size_data = nullptr;
      // 如果定义了袋大小张量，则获取其数据指针
      if (bag_size.defined()) {
        bag_size_data = bag_size.data_ptr<index_t>();
      }
      // 获取添加索引的元素数量
      auto numel = add_indices.numel();
      // 获取源张量的第一个维度大小
      int64_t ddim = src.size(1);
      // 获取源张量的第零维度大小（即词汇表大小）
      auto vocab_size = src.size(0);
      // 获取源张量的第一个维度步幅
      auto src_stride0 = src.strides()[0];
      // 获取源张量的第二个维度步幅
      auto src_stride1 = src.strides()[1];
      // 获取输出张量的第一个维度步幅
      auto output_stride0 = output.strides()[0];
      // 获取输出张量的第二个维度步幅
      auto output_stride1 = output.strides()[1];
    
      // 获取缩放因子数据的指针
      auto* scale_data = scale.const_data_ptr<data_t>();
      // 获取缩放因子张量的步幅
      auto scale_stride = scale.strides()[0];
    
      // 遍历添加索引中的元素
      for (const auto i : c10::irange(numel)) {
        // 如果选择索引为填充索引，则跳过这些索引，不参与计算
        auto idx = select_indices_data[i];
        TORCH_CHECK(
            idx >= 0 && idx < vocab_size,
            "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
            idx);
        if (idx != padding_idx) {
          // 获取源张量中与选择索引对应的基地址
          auto* src_base = src_data + src_stride0 * idx;
          // 获取输出张量中与添加索引对应的基地址
          auto* output_base = output_data + output_stride0 * add_indices_data[i];
          // 获取当前缩放因子
          auto scale = scale_data[i * scale_stride];
          // 遍历维度进行计算和累加
          for (const auto j : c10::irange(ddim)) {
            output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
          }
        } else if (bag_size.defined()) {
          // 如果选择索引是填充索引并且定义了袋大小张量，则减少袋大小以反映填充索引的存在
          // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
          bag_size_data[add_indices_data[i]]--;
        }
      }
// 如果 data_t 是 at::Half 或者 at::BFloat16 类型之一，启用该函数模板，返回类型为 void
template <typename data_t, typename index_t>
typename std::enable_if<
    std::is_same<data_t, at::Half>::value ||
        std::is_same<data_t, at::BFloat16>::value,
    void>::type
index_select_scale_add(
    const Tensor& select_indices,    // 选择的索引张量
    const Tensor& add_indices,       // 添加的索引张量（未使用）
    const Tensor& scale,             // 缩放因子张量
    const Tensor& src,               // 源张量
    Tensor& output,                  // 输出张量
    const Tensor& offsets,           // 偏移量张量
    bool include_last_offset,        // 是否包含最后一个偏移量标志
    Tensor& bag_size,                // 袋子大小张量（未使用）
    index_t padding_idx,             // 填充索引
    _EmbeddingBagKernelCache* fbgemm_kernel_cache) {  // 指向 _EmbeddingBagKernelCache 类的指针

  int64_t ddim = src.size(1);                           // 获取源张量的第二维大小
  auto* scale_data = scale.const_data_ptr<data_t>();     // 获取缩放因子张量的常量数据指针
  auto* select_indices_data = select_indices.const_data_ptr<index_t>();  // 获取选择索引张量的常量数据指针
  auto* output_data = output.data_ptr<data_t>();         // 获取输出张量的数据指针

  // 如果可以使用快速路径优化索引选择和缩放操作
  if (is_fast_path_index_select_scale(src, scale, output, padding_idx)) {
    auto src_contig = src.contiguous();                  // 获取源张量的连续版本
    auto* src_data = src_contig.const_data_ptr<data_t>();  // 获取源张量连续版本的常量数据指针
    int64_t output_size = offsets.numel() - 1;           // 初始化输出大小为偏移量张量的元素数减一
    auto* offsets_data = offsets.const_data_ptr<index_t>();  // 获取偏移量张量的常量数据指针
    std::vector<index_t> offsets_include_last;

    // 根据标志选择是否包含最后一个偏移量
    if (include_last_offset) {
      output_size = offsets.numel() - 1;                 // 输出大小为偏移量张量的元素数减一
    } else {
      output_size = offsets.numel();                     // 输出大小为偏移量张量的元素数
      offsets_include_last.resize(offsets.numel() + 1);  // 调整包含最后一个偏移量的向量大小
      std::memcpy(
          offsets_include_last.data(),                   // 复制偏移量数据到包含最后一个偏移量的向量
          offsets.const_data_ptr<index_t>(),
          sizeof(index_t) * offsets.numel());
      offsets_include_last[offsets.numel()] = select_indices.numel();  // 设置最后一个偏移量的值
      offsets_data = offsets_include_last.data();        // 更新偏移量数据指针
    }

    // 创建一个与缩放因子张量相同大小和数据类型的新张量 scale_fp32
    Tensor scale_fp32 = at::empty(scale.sizes(), scale.options().dtype(at::kFloat));
    auto* scale_data_fp32 = scale_fp32.mutable_data_ptr<float>();  // 获取 scale_fp32 张量的可变数据指针

#if defined(USE_FBGEMM)
    constexpr bool isbf16 = std::is_same_v<data_t, at::Half> ? false : true;  // 检查 data_t 是否为 at::Half 类型
    // 如果是 BFloat16 类型，则将 BFloat16 转换为 Float32
    if constexpr (isbf16) {
      fbgemm::Bfloat16ToFloat_simd(
          reinterpret_cast<const fbgemm::bfloat16*>(scale_data),  // 将缩放因子数据转换为 BFloat16 类型的指针
          scale_data_fp32,              // Float32 缩放因子数据指针
          scale_fp32.numel());          // 缩放因子张量的元素数量
    } else {  // 如果是 Half 类型，则将 Half 转换为 Float32
      fbgemm::Float16ToFloat_simd(
          reinterpret_cast<const fbgemm::float16*>(scale_data),   // 将缩放因子数据转换为 Half 类型的指针
          scale_data_fp32,              // Float32 缩放因子数据指针
          scale_fp32.numel());          // 缩放因子张量的元素数量
    }
    // 获取或生成指定类型的嵌入层稀疏矩阵乘法核函数
    auto kernel_16bit_index_t = fbgemm_kernel_cache
        ? fbgemm_kernel_cache
              ->getCallback</* has_weight */ true, index_t, uint16_t>(ddim)
        : fbgemm::GenerateEmbeddingSpMDM<uint16_t, index_t, index_t, uint16_t>(
              /* block_size */ ddim,                       // 块大小为源张量的第二维大小
              /* has_weight */ true,                       // 设置是否包含权重
              /* normalize_by_lengths */ false,            // 是否通过长度进行归一化
              /* prefetch */ 16,                           // 预取值大小
              /* is_weight_positional */ false,            // 是否是权重位置化
              /* use_offsets */ true,                      // 是否使用偏移量
              /* is_bf16_out */ isbf16,                    // 输出是否为 BFloat16 类型
              /* is_bf16_in */ isbf16);                    // 输入是否为 BFloat16 类型
    # 使用 ATen 库的并行函数 `parallel_for`，并指定迭代范围从 0 到 output_size，步长为 1
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
          # 在每个并行任务中，调用 kernel_16bit_index_t 函数处理数据
          bool success = kernel_16bit_index_t(
              /* output_size */ end_idx - start_idx,  # 计算输出大小
              /* index_size */ offsets_data[end_idx] - offsets_data[start_idx],  # 计算索引大小
              /* data_size */ src.size(0),  # 获取输入数据的大小
              /* input */ reinterpret_cast<const uint16_t*>(src_data),  # 将输入数据转换为 uint16_t 类型
              /* indices */ select_indices_data + offsets_data[start_idx],  # 索引数据的起始位置
              /* offsets_or_lengths */ offsets_data + start_idx,  # 偏移或长度数据的起始位置
              /* weights */ scale_data_fp32 + offsets_data[start_idx],  # 权重数据的起始位置
              /* output */ reinterpret_cast<uint16_t*>(output_data + start_idx * ddim));  # 输出数据的起始位置
          # 如果处理不成功，则报告错误，包括处理的范围和相关数据
          if (!success) {
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,  # 处理的范围大小
                offsets_data[end_idx] - offsets_data[start_idx],  # 处理的索引范围大小
                src.size(0),  # 输入数据的大小
                offsets_data + start_idx,  # 偏移数据的起始位置
                select_indices_data + offsets_data[start_idx]);  # 选择的索引数据的起始位置
          }
        });
#else
    // 如果不满足条件，则执行以下代码块

    // 初始化中间输出缓冲区为全零
    Tensor output_fp32 =
        at::zeros({output_size, ddim}, output.options().dtype(at::kFloat));
    auto* output_data_fp32 = output_fp32.data_ptr<float>();

    // 将 scale 数组的数据转换为 float 类型，并存储在 scale_data_fp32 数组中
    for (const auto i : c10::irange(scale.numel())) {
      scale_data_fp32[i] = static_cast<float>(scale_data[i]);
    }

    // 并行循环操作，对 output 数据进行填充
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
          // 调用 EmbeddingLookupIdx 函数进行索引查找和加权操作
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/scale_data_fp32 + offsets_data[start_idx],
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data_fp32 + start_idx * ddim);

          // 遍历填充 output_data 数据
          for (int64_t i = start_idx; i < end_idx; i++) {
            // 根据数据类型 data_t 进行条件编译，将 FP32 类型的中间缓冲区转换为 data_t 类型
            if constexpr (std::is_same<data_t, at::Half>::value) {
              // 如果是 FP16 类型
              for (const auto d : c10::irange(ddim)) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            } else {
              // 如果是 BF16 类型
              int64_t d = 0;
              // 使用向量化指令处理大部分数据
              for (; d < ddim - (ddim % bVec::size()); d += bVec::size()) {
                fVec temp_fp32_0 = fVec::loadu(output_data_fp32 + ddim * i + d);
                fVec temp_fp32_1 =
                    fVec::loadu(output_data_fp32 + ddim * i + d + fVec::size());
                convert_float_bfloat16(temp_fp32_0, temp_fp32_1)
                    .store(output_data + i * ddim + d);
              }
              // 处理剩余不足一个向量的数据
              for (; d < ddim; d++) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            }
          }
        });
#endif
  } else {
    // 如果条件不满足，则执行以下代码块

    // 断言 select_indices 和 add_indices 的元素数相等
    AT_ASSERT(select_indices.numel() == add_indices.numel());

    // 获取 src 和 add_indices 的数据指针
    auto* src_data = src.const_data_ptr<data_t>();
    auto* add_indices_data = add_indices.const_data_ptr<index_t>();

    // 初始化 bag_size_data 为 nullptr，如果定义了 bag_size，则分配内存并获取数据指针
    index_t* bag_size_data = nullptr;
    if (bag_size.defined()) {
      bag_size_data = bag_size.data_ptr<index_t>();
    }

    // 获取 src 和 output 的相关信息
    auto vocab_size = src.size(0);
    auto src_stride0 = src.strides()[0];
    auto src_stride1 = src.strides()[1];
    auto output_stride0 = output.strides()[0];
    auto output_stride1 = output.strides()[1];
    auto scale_stride = scale.strides()[0];
    auto numel = add_indices.numel();
    // 初始化中间输出缓冲区为全零张量，数据类型为 float
    Tensor output_fp32 =
        at::zeros({output.size(0), ddim}, output.options().dtype(at::kFloat));
    // 获取指向输出缓冲区数据的指针
    auto* output_data_fp32 = output_fp32.data_ptr<float>();
    
    // 遍历每个索引 i，其中 i 范围在 [0, numel) 内
    for (const auto i : c10::irange(numel)) {
      // 如果索引值等于 padding_idx，则跳过，不参与计算
      auto idx = select_indices_data[i];
      TORCH_CHECK(
          idx >= 0 && idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          idx);
      if (idx != padding_idx) {
        // 计算 src_base 的起始地址，用于从输入数据中提取对应的向量
        auto* src_base = src_data + src_stride0 * idx;
        // 计算 output_base_fp32 的起始地址，用于在 FP32 中累加结果
        auto* output_base_fp32 = output_data_fp32 + ddim * add_indices_data[i];
        // 获取当前 idx 对应的 scale 值
        auto scale = scale_data[i * scale_stride];
        // 遍历维度 j，对 output_base_fp32 中的每个元素进行累加
        for (const auto j : c10::irange(ddim)) {
          output_base_fp32[j] += static_cast<float>(src_base[j * src_stride1]) *
              static_cast<float>(scale);
        }
      } else if (bag_size.defined()) {
        // 如果定义了 bag_size，则减少相应索引处的 bag_size 值，表示该索引被填充
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        bag_size_data[add_indices_data[i]]--;
      }
    }
    
    // 将 FP32 中间缓冲区的结果转换回输出数据的 16 位整数类型
    for (const auto i : c10::irange(output.size(0))) {
      // 遍历每个维度 d，将 output_data_fp32 中的结果转换为 data_t 类型后存入 output_data
      for (const auto d : c10::irange(ddim)) {
        (output_data + output_stride0 * i)[d * output_stride1] =
            static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
      }
    }
// 结束 C++ 的命名空间
}
// 定义一个模板函数，限定当 data_t 为 float 时返回 void
template<typename data_t, typename index_t>
typename std::enable_if<std::is_same<data_t, float>::value, void>::type
// 函数名 index_select_scale_add，接受多个 Tensor 类型参数和其他变量
index_select_scale_add(const Tensor &select_indices,
                                          const Tensor &add_indices,
                                          const Tensor &scale,
                                          const Tensor &src,
                                          Tensor &output,
                                          const Tensor& offsets,
                                          bool include_last_offset,
                                          Tensor &bag_size,
                                          index_t padding_idx,
                                          _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  // 获取 src 的第二维度大小
  int64_t ddim = src.size(1);
  // 获取 scale 的 float 数据指针
  auto* scale_data = scale.const_data_ptr<float>();
  // 获取 select_indices 的 index_t 数据指针
  auto* select_indices_data = select_indices.const_data_ptr<index_t>();
  // 获取 output 的 float 数据指针
  auto* output_data = output.data_ptr<float>();

  // 如果满足快速路径条件，则调用函数处理快速路径
  if (is_fast_path_index_select_scale(src, scale, output, padding_idx)) {
    // 创建 src 的连续内存副本
    auto src_contig = src.contiguous();
    // 获取 src 的 float 数据指针
    auto* src_data = src_contig.const_data_ptr<float>();
    // 计算 output_size 为 offsets 的元素数减一
    int64_t output_size = offsets.numel() - 1;
    // 获取 offsets 的 index_t 数据指针
    auto* offsets_data = offsets.const_data_ptr<index_t>();
    // 定义包含最后一个偏移量的偏移量向量
    std::vector<index_t> offsets_include_last;

    // 如果 include_last_offset 为 true，则使用 offsets 的元素数减一
    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      // 否则，设置 output_size 为 offsets 的元素数
      output_size = offsets.numel();
      // 调整 offsets_include_last 的大小以容纳额外的元素
      offsets_include_last.resize(offsets.numel() + 1);
      // 复制 offsets 数据到 offsets_include_last
      std::memcpy(
          offsets_include_last.data(),
          offsets.const_data_ptr<index_t>(),
          sizeof(index_t) * offsets.numel());
      // 设置 offsets_include_last 的最后一个元素为 select_indices 的元素数
      offsets_include_last[offsets.numel()] = select_indices.numel();
      // 将 offsets_data 指向 offsets_include_last 的数据
      offsets_data = offsets_include_last.data();
    }

    // 如果定义了 USE_FBGEMM
#ifdef USE_FBGEMM
    // 定义 kernel_fp32_index_t 作为 fbgemm_kernel_cache 的回调函数
    auto kernel_fp32_index_t =
      fbgemm_kernel_cache ?
      fbgemm_kernel_cache->getCallback</* has_weight */ true, index_t, float>(ddim) :
      fbgemm::GenerateEmbeddingSpMDM<float, index_t, index_t>(
        /* block_size */ddim,
        /* has_weight */true,
        /* normalize_by_lengths */false,
        /* prefetch */16,
        /* is_weight_positional */false,
        /* use_offsets */true
      );
#endif

    // 使用并行方式处理从 start_idx 到 end_idx 的索引范围
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
#ifdef USE_FBGEMM
          // 如果定义了 USE_FBGEMM，则使用 fbgemm 库进行稀疏矩阵乘法运算
          bool success = kernel_fp32_index_t(
            /* output_size */end_idx - start_idx,
            /* index_size */offsets_data[end_idx] - offsets_data[start_idx],
            /* data_size */src.size(0),
            /* input */src_data,
            /* indices */select_indices_data + offsets_data[start_idx],
            /* offsets_or_lengths */offsets_data + start_idx,
            /* weights */scale_data + offsets_data[start_idx],
            /* output */output_data + start_idx * ddim);
          if (!success) {
            // 如果运算失败，报告错误
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,
                offsets_data[end_idx] - offsets_data[start_idx],
                src.size(0),
                offsets_data + start_idx,
                select_indices_data + offsets_data[start_idx]);
          }
#else
          // 如果未定义 USE_FBGEMM，则使用 Caffe2 的 EmbeddingLookupIdx 函数进行稀疏矩阵乘法运算
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/scale_data + offsets_data[start_idx],
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data + start_idx * ddim);
#endif
        });
  } else {
    // 如果 select_indices 和 add_indices 的元素数量相等，则继续执行以下代码
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto* src_data = src.const_data_ptr<float>();
    auto* add_indices_data = add_indices.const_data_ptr<index_t>();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    index_t* bag_size_data = nullptr;
    if (bag_size.defined()) {
      // 如果定义了 bag_size，则获取其数据指针
      bag_size_data = bag_size.data_ptr<index_t>();
    }
    auto vocab_size = src.size(0);
    auto src_stride0 = src.strides()[0];
    auto src_stride1 = src.strides()[1];
    auto output_stride0 = output.strides()[0];
    auto output_stride1 = output.strides()[1];
    auto scale_stride = scale.strides()[0];
    auto numel = add_indices.numel();
    // 对于每个索引 i 在范围内，执行以下操作
    for (const auto i : c10::irange(numel)) {
      // 我们可以跳过等于 padding_idx 的索引，以确保它们不包含在计算中
      auto idx = select_indices_data[i];
      // 检查索引的有效性，确保 idx 大于等于 0 且小于词汇表大小 vocab_size
      TORCH_CHECK(
          idx >= 0 && idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          idx);
      // 如果 idx 不等于 padding_idx
      if (idx != padding_idx) {
        // 计算源数据的基地址和输出数据的基地址
        auto* src_base = src_data + src_stride0 * idx;
        auto* output_base = output_data + output_stride0 * add_indices_data[i];
        // 获取当前 scale 数据
        auto scale = scale_data[i * scale_stride];
        // 针对每个维度 j 进行操作
        for (const auto j : c10::irange(ddim)) {
          // 将 src_base 中的数据乘以 scale，并加到 output_base 中对应位置上
          output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
        }
      } else if (bag_size.defined()) {
        // 如果定义了 bag_size，则减少对应位置 add_indices_data[i] 处的 bag_size
        // 以反映该索引是填充的情况
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        bag_size_data[add_indices_data[i]]--;
      }
    }
  }


这段代码的作用是根据给定的条件执行嵌入袋（embedding bag）的操作，计算加权和或平均值，并处理填充索引（padding index）的情况。
}  // namespace



void check_arguments(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  // 检查张量类型是否为长整型或整型
  checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  // 检查张量类型是否为长整型或整型
  checkScalarTypes("embedding_bag", offsets_arg, {kLong, kInt});
  // 检查indices和offsets张量是否具有相同的数据类型
  checkSameType("embedding_bag", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  // 检查张量类型是否为半精度浮点型、bfloat16、单精度浮点型或双精度浮点型
  checkScalarTypes(
      "embedding_bag", weight_arg, {kHalf, kBFloat16, kFloat, kDouble});

  AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_embedding_bag_cpu_impl", [&]() {
    if (offsets.size(0) > 0) {
      index_t offset_0 = offsets.const_data_ptr<index_t>()[0];
      index_t offset_n = offsets.const_data_ptr<index_t>()[offsets.size(0)-1];
      // 检查第一个偏移是否为0，即小批量的第一个序列是否从位置0开始
      TORCH_CHECK(offset_0 == 0, "offsets[0] has to be 0, i.e., the first sequence "
                                "in the mini-batch has to start from position 0. "
                                "However, got ", offsets[0]);
      // 检查最后一个偏移是否不大于indices张量的长度
      TORCH_CHECK(offset_n <= indices.size(0), "offsets[-1] can not "
                  "be greater than input's length ", indices.size(0), " but got offsets[-1] of ",
                  offset_n);
    }
  });

  if (per_sample_weights.has_value() && per_sample_weights.value().defined()) {
    // 当使用per_sample_weights时，检查mode是否为MODE_SUM
    TORCH_CHECK(mode == MODE_SUM,
        "embedding_bag: per_sample_weights only supported with mode='sum'");
    auto per_input_weights_arg = TensorArg(
        per_sample_weights.value(),"per_sample_weights", 1);
    // 检查权重张量和per_sample_weights张量是否具有相同的数据类型
    checkSameType("embedding_bag", weight_arg, per_input_weights_arg);
    // 检查per_sample_weights张量是否为一维张量
    TORCH_CHECK(per_sample_weights.value().dim() == 1);
    // 检查per_sample_weights张量的元素数量是否与indices张量的元素数量相同
    TORCH_CHECK(per_sample_weights.value().numel() == indices.numel());
  }

  if (include_last_offset) {
    // 当include_last_offset为true时，检查offsets张量的长度是否至少为1
    TORCH_CHECK(
        offsets.size(0) >= 1,
        "include_last_offset: number of offset should be at least 1");
  }
}

void make_bag_size_out(
    Tensor& bag_size_out,
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool include_last_offset,
    const bool requires_grad) {
  if (requires_grad || mode == MODE_MEAN || mode == MODE_MAX) {
    // 计算小袋子的数量，根据需要包括最后一个偏移量
    auto num_bags = offsets.size(0) - (include_last_offset ? 1 : 0);
    // 调整bag_size_out张量的大小，以包含小袋子的数量
    at::native::resize_(bag_size_out, {num_bags}, c10::nullopt);
    // 对于MODE_MEAN和MODE_MAX计算以下内容（后者用于反向传播）
    if (num_bags != 1) {
      // 计算小袋子的大小，排除最后一个小袋子
      bag_size_out.slice(0, 0, bag_size_out.size(0) - 1, 1) =
          offsets.slice(0, 1, num_bags, 1) -
          offsets.slice(0, 0, num_bags - 1, 1);
    }
    if (num_bags > 0) {
      // 计算最后一个小袋子的大小
      bag_size_out[-1] = indices.size(0) - offsets[num_bags - 1];
    }
  } else {
    // 如果不需要梯度或者mode不是MODE_MEAN或MODE_MAX，则直接调整大小
    at::native::resize_(bag_size_out, offsets.sizes(), c10::nullopt);
  }
}

void make_max_indices_out(
    Tensor& max_indices_out,
    const Tensor& weight,
    const Tensor& indices,
    // 计算偏移量的张量大小
    const Tensor& offsets,
    // 包的大小张量
    const Tensor& bag_size,
    // 模式，指定操作的类型
    const int64_t mode,
    // 是否包括最后一个偏移量
    bool include_last_offset) {
      // 获取包的数量
      int64_t numBags = offsets.size(0);
      // 如果模式为最大模式
      if (mode == MODE_MAX) {
        // 如果包括最后一个偏移量
        if (include_last_offset) {
          // 检查包的数量至少为1
          TORCH_CHECK(
            numBags >= 1, "include_last_offset: numBags should be at least 1");
          // 减去最后一个包的偏移量
          numBags -= 1;
        }
        // 调整输出张量的大小为[numBags, weight.sizes()[1]]
        at::native::resize_(max_indices_out, {numBags, weight.sizes()[1]}, c10::nullopt);
        // 将输出张量置零
        at::native::zero_(max_indices_out);
      } else {
          // 否则，根据包的大小调整输出张量的大小
          at::native::resize_(max_indices_out, bag_size.sizes(), c10::nullopt);
      }
}

void make_offset2bag_out(
    Tensor& offset2bag,                     // 输出参数，用于存储计算得到的 offset2bag
    Tensor& output,                         // 输出张量，用于存储最终的输出结果
    const Tensor& weight,                   // 权重张量
    const Tensor& indices,                  // 索引张量
    const Tensor& offsets,                  // 偏移张量
    const int64_t mode,                     // 模式，决定如何计算结果
    const std::optional<Tensor>& per_sample_weights,  // 可选的每个样本权重张量
    const int64_t padding_idx) {            // 填充索引

  // 为了节省计算量，在 'sum' 模式下使用快速路径时，跳过计算 offset2bag，因为它不会被使用。
  bool fast_path_sum = is_fast_path(weight, per_sample_weights, output, padding_idx);

  if (mode == MODE_MEAN || mode == MODE_MAX || !fast_path_sum) {
    // 调整 offset2bag 的大小为 indices.size(0) + 1，使用空值填充
    at::native::resize_(offset2bag, {indices.size(0) + 1}, c10::nullopt);
    // 将 offset2bag 初始化为零
    at::native::zero_(offset2bag);

    int64_t offsets_size = offsets.size(0);
    bool include_last_offset = (output.size(0) == offsets_size - 1);
    // 当 include_last_offset 为 true 时，忽略偏移中的最后一个索引
    // 在 include_last_offset 为 true 且 offsets[-1] != indices.size(0) 时修复段错误
    // 参见 https://github.com/pytorch/pytorch/issues/89677 获取更多细节。
    Tensor _offsets = offsets;
    if (include_last_offset) {
      _offsets = offsets.narrow(0, 0, offsets_size - 1);
    }
    // 计算 offset2bag
    make_offset2bag(_offsets, offset2bag);
    // 将 offset2bag 的大小调整为 indices.size(0)，使用空值填充
    at::native::resize_(offset2bag, {indices.size(0)}, c10::nullopt);
    // 只有在慢路径中才初始化 output
    at::native::zero_(output);
  }
}

static Tensor make_bag_size(
    const Tensor& offsets,                  // 偏移张量
    const Tensor& indices,                  // 索引张量
    const int64_t mode,                     // 模式，决定如何计算结果
    const bool include_last_offset,         // 是否包含最后一个偏移
    const bool requires_grad) {             // 是否需要梯度计算

  // 创建一个与 offsets 相同大小的空张量 bag_size
  Tensor bag_size = at::empty(offsets.sizes(), offsets.options());
  // 计算并返回 bag_size
  make_bag_size_out(bag_size, offsets, indices, mode, include_last_offset, requires_grad);
  return bag_size;
}

static Tensor make_max_indices(
    const Tensor& weight,                   // 权重张量
    const Tensor& indices,                  // 索引张量
    const Tensor& offsets,                  // 偏移张量
    const Tensor& bag_size,                 // bag 大小张量
    const int64_t mode,                     // 模式，决定如何计算结果
    bool include_last_offset) {             // 是否包含最后一个偏移

  // 创建一个与 bag_size 相同大小的空张量 max_indices
  Tensor max_indices = at::empty(bag_size.sizes(), offsets.options());
  // 计算并返回 max_indices
  make_max_indices_out(max_indices, weight, indices, offsets, bag_size, mode, include_last_offset);
  return max_indices;
}

static Tensor make_offset2bag(
    Tensor& output,                         // 输出参数，用于存储最终的输出结果
    const Tensor& weight,                   // 权重张量
    const Tensor& indices,                  // 索引张量
    const Tensor& offsets,                  // 偏移张量
    const int64_t mode,                     // 模式，决定如何计算结果
    const std::optional<Tensor>& per_sample_weights,  // 可选的每个样本权重张量
    const int64_t padding_idx) {            // 填充索引

  // 创建一个空的 offset2bag 张量
  Tensor offset2bag = at::empty({0}, offsets.options());
  // 调用 make_offset2bag_out 计算 offset2bag，并返回结果
  make_offset2bag_out(offset2bag, output, weight, indices, offsets, mode, per_sample_weights, padding_idx);
  return offset2bag;
}

static Tensor apply_bag_size(
    const int64_t mode,                     // 模式，决定如何计算结果
    Tensor &output,                         // 输出张量，用于存储最终的输出结果
    const Tensor &bag_size) {               // bag 大小张量

  if (mode == MODE_MEAN) {
    // 计算 bag_size_，这里使用了 max 函数，然后转换为与 output 相同的选项类型
    auto bag_size_ = at::max(bag_size, at::ones_like(bag_size, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
                         .to(output.options())
                         .unsqueeze(1)
                         .expand_as(output);
    // 对 output 应用 bag_size_ 进行归一化
    output /= bag_size_;
  }
  // 返回最终的输出结果 output
  return output;
}

static Tensor apply_bag_size_backward(
    // 如果 mode 等于 MODE_MEAN，则执行以下逻辑
    if (mode == MODE_MEAN) {
        // 计算每个子袋的逆大小，即每个子袋的平均元素数的倒数
        auto inv_bag_size_ = (1 / bag_size.to(output.options()))
                               .unsqueeze(1)
                               .index_select(0, offset2bag);
        // 将输出张量乘以每个子袋的逆大小，实现均值池化操作
        output *= inv_bag_size_;
    }
    // 返回处理后的输出张量
    return output;
}

`
# 定义函数 `embedding_bag_cpu_max_out`，处理 CPU 上的嵌入袋操作，返回最大值的索引
template <typename scalar_t>
void embedding_bag_cpu_max_out(
    // 输出参数，保存每个袋子中最大值的索引
    Tensor* max_indices,
    // 嵌入权重张量
    const Tensor& weight,
    // 索引张量，指定每个元素对应的嵌入
    const Tensor& indices,
    // 偏移到袋子的张量，标志每个袋子在 `output` 中的起始位置
    const Tensor& offset2bag,
    // 输出张量，存储计算结果
    const Tensor& output,
    // 是否包含最后一个偏移量
    bool include_last_offset,
    // 每个袋子的大小
    Tensor& bag_size,
    // 填充索引，指示填充值
    int64_t padding_idx) {
  
  // 获取索引张量中的元素数量
  int64_t numIndices = indices.numel();
  // 获取权重张量的特征大小
  int64_t featureSize = weight.size(1);
  // 获取词汇表大小
  int64_t vocab_size = weight.size(0);
  
  // 根据索引类型分派操作，这里实现嵌入袋的最大化操作
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_cpu_max_out", [&] {
    // 获取索引数据指针
    auto* indices_data = indices.const_data_ptr<index_t>();
    // 获取偏移到袋子数据指针
    auto* offset2bag_data = offset2bag.data_ptr<index_t>();

    // 初始化最大索引数据指针和步长
    index_t* max_indices_data = nullptr;
    int64_t max_indices_stride = 0;
    if (max_indices) {
      max_indices_data = max_indices->data_ptr<index_t>();
      max_indices_stride = max_indices->strides()[0];
    }

    // 获取权重数据指针和输出数据指针
    auto* weight_data = weight.const_data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    // 获取袋子大小数据指针
    auto* bag_size_data = bag_size.data_ptr<index_t>();
    // 获取权重张量的步长
    auto weight_stride0 = weight.strides()[0];
    auto weight_stride1 = weight.strides()[1];
    // 获取输出张量的步长
    auto output_stride = output.strides()[0];
    // 获取袋子数量
    int64_t numBags = bag_size.size(0);
    // 初始化袋子是否为空的标志
    std::vector<bool> bag_empty(numBags, true);

    // 遍历索引张量中的每个元素
    for (const auto i : c10::irange(numIndices)) {
      // 获取当前元素所属的袋子
      auto bag = offset2bag_data[i];
      // 获取词汇索引
      auto word_idx = indices_data[i];
      
      // 检查词汇索引是否有效
      TORCH_CHECK(
          word_idx >= 0 && word_idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          word_idx);
      
      // 如果词汇索引不是填充索引
      if (word_idx != static_cast<index_t>(padding_idx)) {
        // 检查当前袋子是否为空
        bool is_first_for_bag = bag_empty[bag];
        
        // 遍历特征维度
        for (const auto dim : c10::irange(featureSize)) {
          // 获取当前袋子中当前维度的值
          auto& current_item = output_data[output_stride * bag + dim];
          // 获取权重值
          auto weight_item =
              weight_data[weight_stride0 * word_idx + dim * weight_stride1];

          // 如果是袋子的第一个元素或者权重值大于当前值
          if (is_first_for_bag || (weight_item > current_item)) {
            // 更新当前值为权重值
            current_item = weight_item;
            // 如果存在最大索引，则更新最大索引
            if (max_indices_data) {
              max_indices_data[max_indices_stride * bag + dim] = word_idx;
            }
          }
        }
        // 如果是袋子的第一个元素，则将袋子标记为非空
        if (is_first_for_bag) {
          bag_empty[bag] = false;
        }
      } else {
        // 如果词汇索引是填充索引，则减小袋子大小
        bag_size_data[bag]--;
      }
    }
  });
}

// 定义 CPU 上的嵌入袋操作，处理输出参数
void _embedding_bag_cpu_impl_out(Tensor& output, Tensor& offset2bag,
                            Tensor& bag_size, Tensor* max_indices,
                            const Tensor &weight, const Tensor &indices,
                            const Tensor &offsets, const int64_t mode,
                            const std::optional<Tensor>& per_sample_weights,
                            bool include_last_offset, int64_t padding_idx, _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  // 如果操作模式是 MEAN 或 SUM
  if (mode == MODE_MEAN || mode == MODE_SUM) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, weight.scalar_type(), "embedding_bag_no_grad_cpu_out",
      [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx, &fbgemm_kernel_cache]() {
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_no_grad_cpu_out",
        [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx, &fbgemm_kernel_cache]() {
        // 检查是否有每个样本权重，并且权重已定义
        if (per_sample_weights.has_value() && per_sample_weights.value().defined()) {
          // 断言模式为 MODE_SUM
          TORCH_INTERNAL_ASSERT(mode == MODE_SUM);
          // 使用权重进行索引选择和比例加法
          index_select_scale_add<scalar_t, index_t>(
            indices, offset2bag, per_sample_weights.value(), weight, output, offsets, include_last_offset, bag_size, padding_idx, fbgemm_kernel_cache);
        } else {
          // 使用权重进行索引选择和加法（无权重）
          index_select_add<scalar_t, index_t>(indices, offset2bag, weight, output, offsets, include_last_offset, bag_size, padding_idx, fbgemm_kernel_cache);
        }
      });
    });
    // 应用 bag_size 的模式
    apply_bag_size(mode, output, bag_size);
    // 如果模式为 MODE_SUM，则将 bag_size 清零，确保输出一致性
    if (mode == MODE_SUM) {
      // make bag_size output deterministic
      at::native::zero_(bag_size);
    }
    // 如果存在 max_indices，则复制 bag_size 到 max_indices
    if (max_indices) {
      max_indices->copy_(bag_size);
    }
  } else { // MODE_MAX
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        weight.scalar_type(),
        "embedding_bag_cpu_max_out",
        [&]() {
          // 调用 CPU 版本的 embedding_bag 最大输出函数
          embedding_bag_cpu_max_out<scalar_t>(
              max_indices,
              weight,
              indices,
              offset2bag,
              output,
              include_last_offset,
              bag_size,
              padding_idx);
        });
  }


注释结束
// embedding_bag 函数的包装器，用于确保除了 `weight` 外的所有输入张量都是连续的。
// 这样做是为了在反向传播时避免额外的 `.contiguous()` 调用。
// 详细信息请参阅 native_functions.yaml 中的 "NOTE [ embedding_bag Native Functions ]"。

std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag(const Tensor &weight, const Tensor &indices,
              const Tensor &offsets, const bool scale_grad_by_freq,
              const int64_t mode, bool sparse, const std::optional<Tensor>& per_sample_weights_opt,
              bool include_last_offset, std::optional<int64_t> padding_idx_opt) {
  // 查看 [Note: hacky wrapper removal for optional tensor]
  // 从可选张量中获取 per_sample_weights，确保其是一个已拥有的张量
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  
  // 初始化 padding_idx，默认为 -1
  int64_t padding_idx = -1;

  // 如果存在 padding_idx_opt 的值，则更新 padding_idx
  if (padding_idx_opt.has_value()) {
    auto num_embeddings = weight.size(0);
    padding_idx = padding_idx_opt.value();
    // 检查 padding_idx 是否在合法范围内，即必须满足 -num_embeddings <= padding_idx < num_embeddings
    TORCH_CHECK(
      (padding_idx >= -num_embeddings) && (padding_idx < num_embeddings),
      "padding_idx must be within the number of embeddings, -", num_embeddings,
      " through ", num_embeddings - 1, ", but got ", padding_idx);
    
    // 调用 maybe_wrap_dim 函数，确保 padding_idx 在 weight.size(0) 的有效维度内
    padding_idx = maybe_wrap_dim(padding_idx, weight.size(0));
  }
  
  // 初始化一个存储返回值的元组 out
  std::tuple<Tensor, Tensor, Tensor, Tensor> out;
  
  // 如果 weight 不需要梯度并且梯度相关的信息未定义，则调用 _embedding_bag_forward_only
  if (!weight.requires_grad() && !weight._fw_grad(/*level=*/0).defined()) {
    out = at::_embedding_bag_forward_only(
      weight, indices.contiguous(), offsets.contiguous(), scale_grad_by_freq,
      mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  } else {
    // 否则调用 _embedding_bag 函数，计算嵌入向量的加权和
    out = at::_embedding_bag(
      weight, indices.contiguous(), offsets.contiguous(), scale_grad_by_freq,
      mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  }
  
  // 返回计算结果的元组 out
  return out;
};

// 定义了一个函数，用于计算嵌入袋（embedding bag）的前向传播
std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag(const Tensor &weight, const Tensor &indices,
              const Tensor &offsets, const bool scale_grad_by_freq,
              const int64_t mode, bool sparse, const std::optional<Tensor>& per_sample_weights_opt,
              bool include_last_offset) {
  // 调用ATen的native函数embedding_bag来执行实际的计算
  return at::native::embedding_bag(weight, indices, offsets, scale_grad_by_freq,
      mode, sparse, per_sample_weights_opt, include_last_offset, c10::nullopt);
}

// 假设除了`weight`以外的所有输入张量都是连续的。
// 查看native_functions.yaml中的NOTE [ embedding_bag Native Functions ]获取详细信息
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_forward_only_cpu(const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse, const std::optional<Tensor>& per_sample_weights_opt, bool include_last_offset,
                  int64_t padding_idx) {
  // 查看[Note: hacky wrapper removal for optional tensor]，将optional的张量转换为MaybeOwned对象
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  // 忽略scale_grad_by_freq和sparse参数
  std::ignore = scale_grad_by_freq;
  std::ignore = sparse;
  // 调用实际的CPU嵌入袋实现函数
  return _embedding_bag_cpu_impl(
      weight,
      indices,
      offsets,
      mode,
      per_sample_weights,
      include_last_offset,
      padding_idx,
      /*requires_grad=*/false);
}

// 假设除了`weight`以外的所有输入张量都是连续的。
// 查看native_functions.yaml中的NOTE [ embedding_bag Native Functions ]获取详细信息
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_cpu(const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse, const std::optional<Tensor>& per_sample_weights_opt, bool include_last_offset,
                  int64_t padding_idx) {
  // 查看[Note: hacky wrapper removal for optional tensor]，将optional的张量转换为MaybeOwned对象
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  // 忽略scale_grad_by_freq和sparse参数
  std::ignore = scale_grad_by_freq;
  std::ignore = sparse;

  // 调用实际的CPU嵌入袋实现函数
  return _embedding_bag_cpu_impl(
      weight,
      indices,
      offsets,
      mode,
      per_sample_weights,
      include_last_offset,
      padding_idx,
      /*requires_grad=*/true);
}

void _embedding_bag_cpu_out(
    at::Tensor& output,
    at::Tensor& offset2bag,
    at::Tensor& bag_size,
    at::Tensor* p_max_indices,
    const at::Tensor& weight,
    const at::Tensor& indices_,
    const at::Tensor& offsets_,
    const bool /* scale_grad_by_freq */,
    const int64_t mode,
    const bool /* sparse */,
    const std::optional<at::Tensor>& per_sample_weights,
    const bool include_last_offset,
    const std::optional<int64_t>& padding_idx,
    // 通过 promoteIndicesAndOffsets 函数升级 indices_ 和 offsets_，返回升级后的引用
    auto [indicesMaybeOwned, offsetsMaybeOwned] = promoteIndicesAndOffsets(indices_, offsets_);
    // 解构升级后的 indicesMaybeOwned 和 offsetsMaybeOwned，获取其引用
    const auto& indices = *indicesMaybeOwned;
    const auto& offsets = *offsetsMaybeOwned;
    // 检查参数的有效性，确保 weight、indices、offsets、mode 等参数合法性
    at::native::check_arguments(
        weight, indices, offsets, mode, per_sample_weights, include_last_offset);
    
    // 创建 offset2bag 和 output 的输出张量，用于存储计算结果
    at::native::make_offset2bag_out(
        offset2bag,
        output,
        weight,
        indices,
        offsets,
        mode,
        per_sample_weights,
        padding_idx.value_or(-1));
    
    // 计算 bag_size 的输出张量，用于存储计算结果
    at::native::make_bag_size_out(
        bag_size, offsets, indices, mode, include_last_offset, false);
    
    // 如果 p_max_indices 不为空指针，则计算 max_indices 的输出张量
    if (p_max_indices) {
      at::native::make_max_indices_out(
          *p_max_indices,
          weight,
          indices,
          offsets,
          bag_size,
          mode,
          include_last_offset);
    }
    
    // 调用 _embedding_bag_cpu_impl_out 函数执行嵌入操作，存储结果在 output 中
    at::native::_embedding_bag_cpu_impl_out(
        output,
        offset2bag,
        bag_size,
        p_max_indices,
        weight,
        indices,
        offsets,
        mode,
        per_sample_weights,
        include_last_offset,
        padding_idx.value_or(-1),
        fbgemm_kernel_cache);
}

// 定义函数 `_embedding_bag_backward`，用于计算 Embedding Bag 操作的梯度反向传播
Tensor _embedding_bag_backward(const Tensor &grad, const Tensor &indices_,
                              const Tensor &offsets_,
                              const Tensor &offset2bag,
                              const Tensor &bag_size_,
                              const Tensor &max_indices_,
                              int64_t num_weights,
                              bool scale_grad_by_freq, int64_t mode,
                              bool sparse, const std::optional<Tensor>& per_sample_weights_opt,
                              int64_t padding_idx) {
    // 调用 ATen 库中的 `_embedding_bag_backward_symint` 函数进行梯度计算
    return at::native::_embedding_bag_backward_symint(
        grad, indices_, offsets_, offset2bag, bag_size_, max_indices_, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights_opt, padding_idx);
}

// 假定所有输入张量都是连续的。
// 参见 native_functions.yaml 中的 [Note: embedding_bag Native Functions] 获取详细信息
Tensor _embedding_bag_backward_symint(const Tensor &grad, const Tensor &indices_,
                              const Tensor &offsets_,
                              const Tensor &offset2bag,
                              const Tensor &bag_size_,
                              const Tensor &max_indices_,
                              c10::SymInt num_weights,
                              bool scale_grad_by_freq, int64_t mode,
                              bool sparse, const std::optional<Tensor>& per_sample_weights_opt,
                              int64_t padding_idx) {
    // 查看 [Note: hacky wrapper removal for optional tensor] 的内容
    c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
    const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

    // 提升 indices 和 offsets 的可变性，并获取它们的常量引用
    auto [indicesMaybeOwned, offsetsMaybeOwned] = promoteIndicesAndOffsets(indices_, offsets_);
    const auto& indices = *indicesMaybeOwned;
    const auto& offsets = *offsetsMaybeOwned;

    // 创建 TensorArg 对象，检查 indices 和 offsets 的标量类型和连续性
    auto indices_arg = TensorArg(indices, "indices", 1);
    checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
    checkContiguous("embedding_bag", indices_arg);
    auto offsets_arg = TensorArg(offsets, "offsets", 1);
    checkScalarTypes("embedding_bag", offsets_arg, {kLong, kInt});
    checkSameType("embedding_bag", indices_arg, offsets_arg);
    checkContiguous("embedding_bag", offsets_arg);

    Tensor offset2bag_;

    // 如果 indices 的符号元素数量不为 0，并且 offset2bag 的符号元素数量为 0
    if (indices.sym_numel() != 0 && offset2bag.sym_numel() == 0) {
        // 创建全零的 offset2bag 张量，形状为 {indices.size(0) + 1}
        offset2bag_ = offsets.new_zeros(
          {indices.size(0) + 1}, offsets.options()); // offset2bag = [0 0 0 0 0]

        // 使用 offsets 创建 offset2bag
        make_offset2bag(offsets, offset2bag_);

        // 对于符合复合兼容性的情况，如果 `offset2bag_` 是 CCT
        // 则不能调用 `resize_`，而是调用 `narrow` 切片张量
        if (isTensorSubclassLike(offset2bag_)) {
          offset2bag_ = offset2bag_.narrow(0, 0, indices.size(0));
        } else {
          offset2bag_.resize_({indices.size(0)});
        }
    } else {
        auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    # 检查输入参数的数据类型，确保它们符合预期的标量类型
    checkScalarTypes("embedding_bag", offset2bag_arg, {kLong, kInt});
    # 检查张量是否连续存储，以确保后续操作的有效性
    checkContiguous("embedding_bag", offset2bag_arg);
    # 将传入的偏移数组赋值给类成员变量offset2bag_
    offset2bag_ = offset2bag;
  }

  # 如果稀疏标志为真，则调用稀疏版本的嵌入袋梯度计算函数
  if (sparse) {
    return at::_embedding_bag_sparse_backward_symint(
        grad, indices, offsets, offset2bag_, bag_size_, std::move(num_weights),
        scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  } else {
    # 否则，调用密集版本的嵌入袋梯度计算函数
    return at::_embedding_bag_dense_backward_symint(
        grad, indices, offset2bag_, bag_size_, max_indices_, std::move(num_weights),
        scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  }


这段代码的功能是根据稀疏（sparse）标志调用不同的嵌入袋（embedding bag）的梯度计算函数。
}

// 定义静态函数 _embedding_bag_dense_backward_cpu_max，用于计算稠密反向传播的梯度
// 根据梯度 grad、袋子大小 bag_size 和最大索引 max_indices，以及权重数量 num_weights 进行计算
static Tensor _embedding_bag_dense_backward_cpu_max(
    const Tensor& grad,
    const Tensor& bag_size,
    const Tensor& max_indices,
    int64_t num_weights) {
  // 断言最大索引 max_indices 已经定义
  AT_ASSERT(max_indices.defined());
  // 创建一个与 grad 相同大小的零张量 index_grad_weight
  auto index_grad_weight =
      at::zeros({num_weights, grad.sizes()[1]}, grad.options());
  // 从 max_indices 中选择非空的最大索引
  auto nonempty_max_indices = max_indices.index_select(0, bag_size.nonzero().view(-1));
  // 从 grad 中选择非空的梯度
  auto nonempty_grad = grad.index_select(0, bag_size.nonzero().view(-1));

  // 遍历每一个维度 dim
  for (const auto dim : c10::irange(grad.sizes()[1])) {
    // 在 index_grad_weight 的第 dim 列上，根据非空的最大索引和梯度添加索引
    index_grad_weight.select(1, dim).index_add_(
      0, nonempty_max_indices.select(1, dim), nonempty_grad.select(1, dim));
  }
  // 返回计算得到的 index_grad_weight
  return index_grad_weight;
}

// 定义模板函数 compute_counts，计算给定 indices 数据的计数
template<typename index_t>
static std::vector<index_t> compute_counts(
    int64_t num_weights,
    const index_t* indices_data,
    int64_t indices_length) {
  // 创建一个大小为 num_weights 的计数向量 counts，初始值为 0
  std::vector<index_t> counts(num_weights, 0);
  // 遍历 indices 数据，增加对应位置的计数
  for (const auto i : c10::irange(indices_length)) {
    counts[indices_data[i]]++;
  }
  // 返回计数结果 counts
  return counts;
}

// 定义模板函数 compute_counts_uniq，计算唯一索引的计数向量 counts_uniq
// counts_uniq 存储下一个唯一元素在 (排序后的) indices 向量中的索引
//
// 例如：
// indices: [0, 0, 0, 1, 3, 3, 4]
// counts: [3, 1, 0, 2, 1, 0]
// counts_uniq: [3, 4, 6, 7]
//
// 唯一索引可以在索引 0, 3, 4, 6 处找到
template<typename index_t>
static std::vector<index_t> compute_counts_uniq(
    int64_t num_weights,
    const index_t* indices_data,
    int64_t indices_length,
    const std::vector<index_t>& counts) {
  // 创建一个空的 counts_uniq 向量，预留 num_weights 大小
  std::vector<index_t> counts_uniq;
  counts_uniq.reserve(num_weights);
  int64_t o = 0;
  // 遍历 indices_length，按照 counts 的步长增加 counts_uniq 中的值
  for (int64_t i = 0; i < indices_length; i += counts[indices_data[i]]) {
    counts_uniq.push_back(counts[indices_data[i]]);
    if (o > 0) {
      counts_uniq[o] += counts_uniq[o - 1];
    }
    o++;
  }
  // 返回计算得到的 counts_uniq
  return counts_uniq;
}

// 定义模板函数 _embedding_bag_dense_backward_cpu_sum_mean，计算稠密反向传播的求和或均值
// 根据梯度 grad、索引 indices、offset2bag、bag_size、权重数量 num_weights、是否按频率缩放 scale_grad_by_freq、模式 mode、每样本权重 per_sample_weights、索引梯度权重 index_grad_weight 和填充索引 padding_idx 进行计算
template <typename scalar_t>
void _embedding_bag_dense_backward_cpu_sum_mean(
    const Tensor& grad,
    const Tensor& indices_,
    const Tensor& offset2bag__,
    const Tensor& bag_size_,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights_,
    Tensor& index_grad_weight,
    int64_t padding_idx) {

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  // 强制转换 offset2bag__ 为可修改的张量 offset2bag_
  Tensor &offset2bag_ = const_cast<Tensor &>(offset2bag__);

  // 对 indices_ 进行排序，并保存排序结果在 ind_sort_ 中
  auto ind_sort_ = indices_.sort();
  // 获取排序后的索引 indices 和排序索引 ind_sort
  auto indices = std::get<0>(ind_sort_);
  auto ind_sort = std::get<1>(ind_sort_);
  // 根据排序后的 ind_sort 从 offset2bag_ 中选择偏移量
  auto offset2bag = offset2bag_.index_select(0, ind_sort);

  optional<Tensor> per_sample_weights;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 如果给定了 per_sample_weights_，则根据排序后的 indices 选择对应的权重
  const scalar_t* per_sample_weights_data;
  optional<int64_t> per_sample_weights_stride;
  if (per_sample_weights_.defined()) {
    per_sample_weights = per_sample_weights_.index_select(0, ind_sort);
    per_sample_weights_data = per_sample_weights->const_data_ptr<scalar_t>();
  }
  per_sample_weights_stride = per_sample_weights->strides()[0];



  // 获取 per_sample_weights 的步幅（stride），即其在内存中存储时每个维度的间隔
  per_sample_weights_stride = per_sample_weights->strides()[0];



  int64_t numel = indices.numel();



  // 计算 indices 张量的元素总数
  int64_t numel = indices.numel();



  // 显式捕获所有必要的变量以解决 Windows 构建问题
  // TODO: 当 Windows 正确捕获嵌套 lambda 中的变量时，修复此问题
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_embedding_bag_dense_backward_cpu_sum_mean",
    [&indices, &offset2bag, &bag_size_, &num_weights, &numel, &per_sample_weights,
      &per_sample_weights_data, &per_sample_weights_stride, &mode, &scale_grad_by_freq,
      &grad, &index_grad_weight, &padding_idx] {



    auto* indices_data = indices.const_data_ptr<index_t>();
    auto* offset2bag_data = offset2bag.const_data_ptr<index_t>();
    auto* bag_size_data = bag_size_.const_data_ptr<index_t>();



    // 获取 indices、offset2bag 和 bag_size_ 张量的常量数据指针
    auto* indices_data = indices.const_data_ptr<index_t>();
    auto* offset2bag_data = offset2bag.const_data_ptr<index_t>();
    auto* bag_size_data = bag_size_.const_data_ptr<index_t>();



    auto counts = compute_counts(num_weights, indices_data, numel);
    auto next_unique_index_idx =
        compute_counts_uniq(num_weights, indices_data, numel, counts);



    // 计算权重数、计算唯一索引的下标
    auto counts = compute_counts(num_weights, indices_data, numel);
    auto next_unique_index_idx =
        compute_counts_uniq(num_weights, indices_data, numel, counts);



    auto loop =
      [&next_unique_index_idx, &indices_data, &offset2bag_data, &bag_size_data, &per_sample_weights,
        &mode, &per_sample_weights_data, &per_sample_weights_stride, &scale_grad_by_freq,
        &counts, &grad, &index_grad_weight, &padding_idx
      ](index_t start, index_t end) {



      // 定义一个 lambda 函数，用于循环处理每个索引范围内的操作
      auto loop =
        [&next_unique_index_idx, &indices_data, &offset2bag_data, &bag_size_data, &per_sample_weights,
          &mode, &per_sample_weights_data, &per_sample_weights_stride, &scale_grad_by_freq,
          &counts, &grad, &index_grad_weight, &padding_idx
        ](index_t start, index_t end) {



      for (index_t i = start; i < end; i++) {
        index_t start = i == 0 ? 0 : next_unique_index_idx[i - 1];
        index_t index = indices_data[start];



        // 遍历处理每个索引范围内的元素，获取起始索引和当前索引值
        for (index_t i = start; i < end; i++) {
          index_t start = i == 0 ? 0 : next_unique_index_idx[i - 1];
          index_t index = indices_data[start];



        if (index != static_cast<index_t>(padding_idx)) {



          // 如果当前索引不是填充索引
          if (index != static_cast<index_t>(padding_idx)) {



          for (index_t j = start; j < next_unique_index_idx[i]; j++) {
            index_t source = offset2bag_data[j];
            double scale = 1.0;
            if (per_sample_weights) {
              AT_ASSERT(mode == MODE_SUM);
              scale = per_sample_weights_data[*per_sample_weights_stride * j];
            }
            if (scale_grad_by_freq) {
              scale /= counts[indices_data[i]];
            }
            if (mode == MODE_MEAN) {
              auto bag_size = bag_size_data[source];
              if (bag_size != 0) {
                scale /= bag_size;
              }
            }
            int64_t ddim = grad.size(1);
            auto igwd = index_grad_weight.data_ptr<scalar_t>();
            auto gd = grad.const_data_ptr<scalar_t>();
            at::native::cpublas::axpy<scalar_t>(ddim, (scalar_t)scale, gd + ddim * source, 1,
                        igwd + ddim * index, 1);
          }



          // 遍历处理当前索引对应的源索引范围，应用比例并更新梯度
          for (index_t j = start; j < next_unique_index_idx[i]; j++) {
            index_t source = offset2bag_data[j];
            double scale = 1.0;
            if (per_sample_weights) {
              AT_ASSERT(mode == MODE_SUM);
              scale = per_sample_weights_data[*per_sample_weights_stride * j];
            }
            if (scale_grad_by_freq) {
              scale /= counts[indices_data[i]];
            }
            if (mode == MODE_MEAN) {
              auto bag_size = bag_size_data[source];
              if (bag_size != 0) {
                scale /= bag_size;
              }
            }
            int64_t ddim = grad.size(1);
            auto igwd = index_grad_weight.data_ptr<scalar_t>();
            auto gd = grad.const_data_ptr<scalar_t>();
            at::native::cpublas::axpy<scalar_t>(ddim, (scalar_t)scale, gd + ddim * source, 1,
                        igwd + ddim * index, 1);
          }



      }



      // 如果元素总数大于 1000，则并行执行循环
      if (numel > 1000) {
        at::parallel_for(0, (int64_t)next_unique_index_idx.size(), 0, loop);
      } else {
        loop(0, (int64_t)next_unique_index_idx.size());
      }
    }
  });



      // 如果元素总数大于 1000，则并行执行循环
      if (numel > 1000) {
        at::parallel_for(0, (int64_t)next_unique_index_idx.size(), 0, loop);
      } else {
        loop(0, (int64_t)next_unique_index_idx.size());
      }
    }
  });
// 结束当前函数的定义，这里的右大括号对应函数 _embedding_bag_dense_backward_cpu 的开始
}

// 定义函数 _embedding_bag_dense_backward_cpu，接受多个参数并返回一个 Tensor 类型对象
Tensor _embedding_bag_dense_backward_cpu(const Tensor &grad_, const Tensor &indices_,
                                  const Tensor &offset2bag__,
                                  const Tensor &bag_size_,
                                  const Tensor& max_indices_, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode, const std::optional<Tensor>& per_sample_weights__opt,
                                  int64_t padding_idx) {
  // 创建一个 MaybeOwned<Tensor> 类型的对象 per_sample_weights__maybe_owned，并通过 at::borrow_from_optional_tensor 方法从可选的张量 per_sample_weights__opt 中获取数据
  c10::MaybeOwned<Tensor> per_sample_weights__maybe_owned = at::borrow_from_optional_tensor(per_sample_weights__opt);
  // 从 MaybeOwned<Tensor> 中获取实际的张量 per_sample_weights_
  const Tensor& per_sample_weights_ = *per_sample_weights__maybe_owned;

  // indices_, offsets_ 和 offset2bag__ 在这里假设已经有了正确的数据类型和连续性，这是因为在 _embedding_bag_backward 中进行了检查。
  // 更多细节请参见 native_functions.yaml 中的 NOTE [ embedding_bag Native Functions ]。
  auto grad = grad_.contiguous();  // 对梯度 grad_ 进行连续性处理，保存为 grad
  auto grad_arg = TensorArg(grad, "grad_", 1);  // 创建 TensorArg 类型对象 grad_arg，用于描述 grad_
  checkScalarTypes(
      "embedding_bag", grad_arg, {kHalf, kBFloat16, kFloat, kDouble});  // 检查 grad 的标量类型是否在 {kHalf, kBFloat16, kFloat, kDouble} 中

  // 如果 mode 等于 MODE_MAX，则调用 _embedding_bag_dense_backward_cpu_max 函数，并返回其结果
  if (mode == MODE_MAX) {
    return _embedding_bag_dense_backward_cpu_max(
        grad_, bag_size_, max_indices_, num_weights);
  }
  // 使用 AT_ASSERT 确保 mode 的值为 MODE_MEAN 或 MODE_SUM
  AT_ASSERT(mode == MODE_MEAN || mode == MODE_SUM);

  // 创建一个和 grad 具有相同大小和选项的全零张量 index_grad_weight
  auto index_grad_weight =
      at::zeros({num_weights, grad.sizes()[1]}, grad.options());

  // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏来处理所有浮点类型（包括 Half 和 BFloat16）
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "embedding_bag_backward",  // 提供一个描述字符串
      [&] {  // lambda 函数开始
        // 调用 _embedding_bag_dense_backward_cpu_sum_mean 函数，传入相应参数
        _embedding_bag_dense_backward_cpu_sum_mean<scalar_t>(
            grad,
            indices_,
            offset2bag__,
            bag_size_,
            num_weights,
            scale_grad_by_freq,
            mode,
            per_sample_weights_,
            index_grad_weight,
            padding_idx);
      });  // lambda 函数结束
  // 返回 index_grad_weight
  return index_grad_weight;
}

// 开始定义 _embedding_bag_per_sample_weights_backward_cpu_template 函数模板
template<typename scalar_t>
Tensor _embedding_bag_per_sample_weights_backward_cpu_template(
    const Tensor& grad,
    const Tensor& weight,  // 注意：这里是嵌入表，不是每个样本的权重
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx) {
```  
# 定义函数开始，声明一个整型变量 `padding_idx` 作为参数。

  TORCH_CHECK(
      mode == MODE_SUM,
      "embedding_bag_backward: per_sample_weights only supported for mode='sum'");
```py  
# 检查条件，如果 `mode` 不等于 `MODE_SUM`，则抛出错误信息，指出当前只支持 `mode='sum'` 模式。

  AT_ASSERT(grad.dim() == 2);
  auto embedding_features = grad.sizes()[1];
```  
# 断言 `grad` 张量的维度为2。获取 `grad` 张量的第二维大小，并将其赋值给 `embedding_features`。

  auto [indicesMaybeOwned, offsetsMaybeOwned] = promoteIndicesAndOffsets(indices_, offsets_);
  const auto& indices = *indicesMaybeOwned;
  const auto& offsets = *offsetsMaybeOwned;
```py  
# 调用 `promoteIndicesAndOffsets` 函数，将 `indices_` 和 `offsets_` 作为参数传入，获取返回的元组，并将元组解包为 `indices` 和 `offsets` 常量引用。

  AT_ASSERT(indices.dim() == 1);
  auto num_samples = indices.size(0);
```  
# 断言 `indices` 张量的维度为1。获取 `indices` 张量的大小作为 `num_samples`。

  AT_ASSERT(weight.dim() == 2);
  AT_ASSERT(weight.sizes()[1] == embedding_features);
```py  
# 断言 `weight` 张量的维度为2，并且第二维的大小与 `embedding_features` 相同。

  auto output = at::zeros({num_samples}, grad.options());
```  
# 创建一个与 `num_samples` 大小相同的零张量 `output`，其选项与 `grad` 张量相同。

  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
  checkContiguous("embedding_bag", indices_arg);
```py  
# 创建一个张量参数对象 `indices_arg`，并对其进行类型检查，要求其标量类型为长整型或整型，同时检查其是否是连续的。

  Tensor offset2bag_;
  if (indices.numel() != 0 && offset2bag.numel() == 0) {
    offset2bag_ = at::zeros(
       {indices.size(0) + 1}, offset2bag.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, offset2bag_);

    at::native::resize_(offset2bag_, {indices.size(0)}, c10::nullopt);
  } else {
    auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    checkScalarTypes("embedding_bag", offset2bag_arg, {kLong, kInt});
    checkContiguous("embedding_bag", offset2bag_arg);
    offset2bag_ = offset2bag;
  }
```  
# 如果 `indices` 张量的元素个数不为0且 `offset2bag` 张量的元素个数为0，则创建一个大小为 `indices.size(0) + 1` 的零张量 `offset2bag_`，并调用 `make_offset2bag` 函数填充其值。然后调整 `offset2bag_` 的大小为 `{indices.size(0)}`。否则，对 `offset2bag` 张量进行类型检查，要求其标量类型为长整型或整型，同时检查其是否是连续的，并将其赋值给 `offset2bag_`。

  auto* grad_data = grad.const_data_ptr<scalar_t>();
  auto grad_stride0 = grad.strides()[0];
  auto grad_stride1 = grad.strides()[1];

  auto* weight_data = weight.const_data_ptr<scalar_t>();
  auto weight_stride0 = weight.strides()[0];
  auto weight_stride1 = weight.strides()[1];
```py  
# 获取 `grad` 和 `weight` 张量的数据指针，并分别获取其步幅。

  // explicitly capture all required variables to work around windows build
  // TODO: fix this when windows can correctly capture variables in nested lambda
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_embedding_bag_per_sample_weights_backward_cpu_template",
    [&indices, &output, &offset2bag_, &num_samples, &embedding_features,
      &grad_data, &grad_stride0, &grad_stride1, &weight_data, &weight_stride0, &weight_stride1,
      &padding_idx] () {
    auto* indices_data = indices.const_data_ptr<index_t>();

    // The following are contiguous
    auto* output_data = output.data_ptr<scalar_t>();
    auto* offset2bag_data = offset2bag_.const_data_ptr<index_t>();

    // XXX: 64 was arbitrarily chosen. There is probably a sweet spot for this number.
```  
# 使用 `AT_DISPATCH_INDEX_TYPES` 宏，根据 `indices` 张量的标量类型分发处理模板。在 lambda 函数中捕获所有必要的变量，以解决 Windows 构建的问题。定义 `indices_data`、`output_data` 和 `offset2bag_data` 分别指向 `indices`、`output` 和 `offset2bag_` 张量的数据指针，以及一些其他变量。
    // 并行循环，处理从 0 到 num_samples 的数据，步长为 64
    parallel_for(0, num_samples, 64,
      // lambda 函数捕获了多个引用，用于并行处理
      [&embedding_features, &grad_data, &grad_stride0, &grad_stride1, &weight_data, &weight_stride0,
        &weight_stride1, &offset2bag_data, &indices_data, &output_data, &padding_idx](index_t begin, index_t end) {
      // 对于每个样本索引从 begin 到 end
      for (index_t sample_idx = begin; sample_idx < end; sample_idx++) {
        // 获取当前样本的偏移索引和嵌入索引
        auto bag_idx = offset2bag_data[sample_idx];
        auto embedding_idx = indices_data[sample_idx];
    
        // 如果嵌入索引不等于填充索引
        if (embedding_idx != static_cast<index_t>(padding_idx)) {
          // 计算输出数据中的值，通过点积实现
          output_data[sample_idx] = dot_impl<scalar_t>(
              embedding_features,
              // 对梯度数据和权重数据进行动态转换并计算
              const_cast<scalar_t*>(grad_data + grad_stride0 * bag_idx), grad_stride1,
              const_cast<scalar_t*>(weight_data + weight_stride0 * embedding_idx), weight_stride1);
        }
      }
    });
    // 返回处理后的输出数据
    return output;
}

// 反向传播函数，计算嵌入操作的梯度，针对 CPU 实现
Tensor _embedding_bag_per_sample_weights_backward_cpu(
    const Tensor& grad,                   // 输入的梯度张量
    const Tensor& weight,                 // 嵌入表，注意不是每个样本的权重
    const Tensor& indices,                // 索引张量，指示每个样本所用的嵌入向量
    const Tensor& offsets,                // 偏移张量，指示每个样本的起始位置
    const Tensor& offset2bag,             // 偏移到包的映射张量
    int64_t mode,                         // 嵌入操作模式
    int64_t padding_idx) {                // 填充索引

  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "_embedding_bag_per_sample_weights_backward_cpu",
      [&]() {
        // 调用模板函数，计算嵌入操作的梯度
        return _embedding_bag_per_sample_weights_backward_cpu_template<
            scalar_t>(
            grad, weight, indices, offsets, offset2bag, mode, padding_idx);
      });
}

// 计算稀疏梯度的反向传播，用于符号整数嵌入
Tensor _embedding_bag_sparse_backward_symint(
    const Tensor &grad_,                  // 输入的梯度张量
    const Tensor &indices,                // 索引张量，指示每个样本所用的嵌入向量的位置
    const Tensor &offsets,                // 偏移张量，指示每个样本的起始位置
    const Tensor &offset2bag,             // 偏移到包的映射张量
    const Tensor &bag_size_,              // 每个包的大小
    SymInt num_weights,                   // 权重数量
    bool scale_grad_by_freq,              // 是否按频率缩放梯度
    int64_t mode,                         // 嵌入操作模式
    const std::optional<Tensor>& per_sample_weights_opt,  // 每个样本的权重（可选）
    int64_t padding_idx) {                // 填充索引

  // 处理可选的每个样本权重张量
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  // 确保 indices、offsets 和 offset2bag 张量的数据类型和内存布局正确，
  // 这是因为 _embedding_bag_backward 函数中已经进行了检查。
  // 另请参阅 native_functions.yaml 中的注释以获取更多细节。
  
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  Tensor grad = grad_;

  // 根据 offset2bag 选择索引梯度
  Tensor index_grad = grad_.index_select(0, offset2bag);

  // 根据包的大小应用反向传播操作
  index_grad = apply_bag_size_backward(mode, index_grad, offset2bag, bag_size_);

  // 如果定义了每个样本的权重，则乘以权重
  if (per_sample_weights.defined()) {
    AT_ASSERT(mode == MODE_SUM);
    index_grad.mul_(per_sample_weights.unsqueeze(1));
  }

  // 调用 native::embedding_backward_symint 函数进行符号整数嵌入的反向传播
  return native::embedding_backward_symint(index_grad, indices, std::move(num_weights), padding_idx,
                                    scale_grad_by_freq, true);
}
} // namespace at::native
```