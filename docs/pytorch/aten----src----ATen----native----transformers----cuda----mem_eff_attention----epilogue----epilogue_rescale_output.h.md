# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\epilogue\epilogue_rescale_output.h`

```py
/*! \file
  \brief Threadblock scoped GEMM epilogue for Tensor Ops.

  The epilogue rearranges matrix product results in shared memory to match
  tensor layouts in global memory, supporting conversion and reduction ops.

  This file is derived from cutlass/epilogue/threadblock/epilogue.h and
  extends functionality by using "row_id" to determine corresponding `m_prime`
  / `s_prime` for output rescaling.
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <cassert>
#endif

#include <cutlass/aligned_buffer.h>  // Provides aligned memory buffer functionality
#include <cutlass/array.h>           // Array data structure support
#include <cutlass/cutlass.h>         // Top-level Cutlass include
#include <cutlass/functional.h>      // Functional utilities
#include <cutlass/layout/tensor.h>   // Tensor layout definitions
#include <cutlass/layout/vector.h>   // Vector layout definitions
#include <cutlass/numeric_types.h>   // Numeric type definitions
#include <cutlass/tensor_coord.h>    // Tensor coordinate manipulation

#include <cutlass/gemm/gemm.h>  // General matrix multiplication definitions

#include <cutlass/transform/pitch_linear_thread_map.h>  // Thread map utilities
#include <cutlass/transform/threadblock/regular_tile_iterator.h>  // Tile iterator utilities

#include <cutlass/epilogue/threadblock/epilogue_base.h>  // Base epilogue functionality
#include <cutlass/epilogue/threadblock/predicated_tile_iterator.h>  // Predicated tile iterator
#include <cutlass/numeric_types.h>  // Numeric type definitions

#include <cutlass/array.h>            // Array data structure support (again for redundancy)
#include <cutlass/cutlass.h>          // Top-level Cutlass include (again for redundancy)
#include <cutlass/epilogue/thread/scale_type.h>  // Scaling types for epilogue operations
#include <cutlass/functional.h>       // Functional utilities (again for redundancy)
#include <cutlass/numeric_conversion.h>  // Numeric type conversion utilities
#include <cutlass/numeric_types.h>    // Numeric type definitions (again for redundancy)

#include <ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_pipelined.h>  // ATen-specific epilogue support

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
// output <- alpha * accumulator + beta * source
//   with:
//     alpha = 1 / s_prime (normalize when isLast=True, 1 otherwise)
//     beta = alpha / m_prime (renormalize output when max changes)
//     source is the current output
template <
    typename ElementOutput_,       ///< Data type for output tensors
    typename ElementSource_,       ///< Data type for source (typically matches `ElementOutput_`)
    int Count,                     ///< Number of elements per operation
    typename ElementAccumulator_,  ///< Accumulator data type
    typename ElementCompute_,      ///< Data type for computing linear combination
    bool isFirst,                  ///< Whether it is the first operation
    bool isLast,                   ///< Whether it is the last operation
    typename FragmentAlphaBeta_,   ///< Fragment storing alpha and beta coefficients
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest  ///< Rounding style for floating point operations
>
// 定义一个模板类 MemoryEfficientAttentionNormalize，用于实现内存高效的注意力归一化操作
class MemoryEfficientAttentionNormalize {
 public:
  // 定义模板类型别名
  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  // 设置常量 kCount，并使用它定义数组片段类型别名
  static int const kCount = Count;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;
  using FragmentAlphaBeta = FragmentAlphaBeta_;

  // 设置常量 kRound 用于舍入样式
  static FloatRoundStyle const kRound = Round;

 private:
  //
  // 数据成员
  //

  // 常量引用成员变量 s_prime_ 和 m_prime_
  FragmentAlphaBeta const& s_prime_;
  FragmentAlphaBeta const& m_prime_;

 public:
  /// 构造函数，用于初始化对象，从主机内存中加载可能的指针
  CUTLASS_HOST_DEVICE
  MemoryEfficientAttentionNormalize(
      FragmentAlphaBeta const& s_prime,
      FragmentAlphaBeta const& m_prime)
      : s_prime_(s_prime), m_prime_(m_prime) {}

  /// 检查是否需要源数据
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return !isFirst;
  }

  /// 在后处理中进行序列化归约时所需的函数
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  /// 计算线性缩放操作: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      int row,
      FragmentAccumulator const& accumulator,
      FragmentSource const& source) const {
    assert(!isFirst);

    // 将源数据转换为内部计算数值类型
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round>
        source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    // 将转换后的数据转换为目标数据类型
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // 执行二进制运算
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;

    // 根据 isFirst 和 isLast 的值确定 alpha 和 beta 的计算逻辑
    ElementCompute alpha = isLast ? (1 / s_prime_[row]) : 1;
    ElementCompute beta = alpha * m_prime_[row];

    intermediate = mul_add_source(beta, converted_source); // X = beta * C

    intermediate = mul_add_accumulator(
        alpha, converted_accumulator, intermediate); // D = alpha * Accum + X

    return destination_converter(intermediate);
  }

  /// 计算线性缩放操作: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(int row, FragmentAccumulator const& accumulator)
      const {
    assert(isFirst);

    // 将累加器数据转换为内部计算数值类型
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    // 将转换后的数据转换为目标数据类型
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;
    // 创建一个 NumericArrayConverter 对象，用于将数组从 ElementCompute 类型转换到 ElementOutput 类型，数组长度为 kCount，转换时采用 Round 策略
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    // 使用 accumulator_converter 将累加器 accumulator 转换为 ComputeFragment 类型
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // 定义一个 ComputeFragment 类型的变量 intermediate，以及一个乘法函数对象 mul_accumulator
    ComputeFragment intermediate;
    multiplies<ComputeFragment> mul_accumulator;

    // 根据条件判断是否为最后一次迭代，确定 alpha 的值：若是最后一次迭代，则为 1/s_prime_[row]，否则为 1
    ElementCompute alpha = isLast ? (1 / s_prime_[row]) : 1;

    // 计算 intermediate，即 alpha * converted_accumulator，并赋值给 intermediate
    intermediate = mul_accumulator(
        alpha, converted_accumulator); // X =  alpha * C + uniform

    // 将 intermediate 通过 destination_converter 转换为 ElementOutput 类型并返回结果
    return destination_converter(intermediate);
}
};

} // namespace thread

namespace threadblock {
template <
    typename EO,
    typename ES,
    int Count,
    typename EA,
    typename EC,
    bool F,
    bool L,
    typename FAB,
    FloatRoundStyle R>
// 定义结构体 ApplyEpilogueOp，用于处理 MemoryEfficientAttentionNormalize 模板的特化
struct ApplyEpilogueOp<thread::MemoryEfficientAttentionNormalize<
    EO,
    ES,
    Count,
    EA,
    EC,
    F,
    L,
    FAB,
    R>> {
  // 使用 Op 别名引用 MemoryEfficientAttentionNormalize 类型
  using Op = thread::
      MemoryEfficientAttentionNormalize<EO, ES, Count, EA, EC, F, L, FAB, R>;
  
  // 定义静态函数 apply，接受 output_op、row_id、accum、source 参数，返回 Op::FragmentOutput 类型
  static CUTLASS_DEVICE typename Op::FragmentOutput apply(
      Op const& output_op,
      int row_id,
      typename Op::FragmentAccumulator const& accum,
      typename Op::FragmentSource const& source) {
    return output_op(row_id, accum, source);
  }
  
  // 定义静态函数 apply，接受 output_op、row_id、accum 参数，返回 Op::FragmentOutput 类型
  static CUTLASS_DEVICE typename Op::FragmentOutput apply(
      Op const& output_op,
      int row_id,
      typename Op::FragmentAccumulator const& accum) {
    return output_op(row_id, accum);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
```