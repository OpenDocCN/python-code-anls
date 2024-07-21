# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\kernel_backward.h`

```
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cmath>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <ATen/cuda/PhiloxUtils.cuh>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/fast_math.h>
#include <cutlass/functional.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/vector.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>

#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/threadblock/epilogue_smem_accumulator.h>
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/epilogue/warp/tile_iterator_tensor_op.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/default_mma_core_simt.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm70.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>
#include <cutlass/integer_subbyte.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/platform/platform.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/vector_iterator.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_accum_lambda_iterator.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_from_smem.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_pipelined.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/epilogue_predicated_tile_iterator.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/transform/tile_smem_loader.h>

#include <cinttypes>
#include <c10/util/Exception.h>

using namespace gemm_kernel_utils;

namespace PyTorchMemEffAttention {
namespace {

template <typename FragmentType, int32_t kNumThreads>
struct GmemTile {
  /*
    Helper functions to efficient store/load RF to gmem

    GEMM accumulators have a particular format on A100, and
    it takes some compute/shared-memory to rearrange them to
    a RowMajor or ColumnMajor format in global memory through
    an Epilogue. The same complexity goes for loading into RF.
  */
};
    /*
        This class loads/stores data from/to global memory in a structured manner.
        It supports operations like matrix multiplication accumulation (GEMM) efficiently.
    
        Here's the detailed breakdown:
    
        GmemTile tile;  // Define an object 'tile' of type GmemTile
    
        // Iterate through a loop N times
        for (int i = 0; i < N; ++i) {
          // ...
    
          Fragment accum;  // Define an object 'accum' of type Fragment
    
          // Initialize or load 'accum' based on iteration index 'i'
          if (i == 0) {
            accum.clear();  // Clear 'accum' if it's the first iteration
          } else {
            tile.load(accum);  // Load 'accum' from 'tile' if not the first iteration
          }
    
          mma(accum, ...);  // Perform matrix multiply-accumulate operation on 'accum'
    
          if (i < N-1) {
            // Store 'accum' in 'tile' for the next GEMM iteration
            tile.store(accum);
          } else {
            // Perform epilogue operation (e.g., store in tensor in RowMajor format)
            epilogue(accum);
          }
    
          // ...
        }
    */
    
    // Define the data type 'AccessType' for global memory access, representing 128 bits per thread
    using AccessType = cutlass::Array<float, 4>;
    
    // Constants for memory access calculations
    static constexpr int32_t kBytes = sizeof(AccessType);  // Size of 'AccessType' in bytes
    static constexpr int32_t kStride = kNumThreads * AccessType::kElements;  // Memory stride
    static constexpr int32_t kNumIters = FragmentType::kElements / AccessType::kElements;  // Number of iterations
    static constexpr int32_t kElementsStored = kNumThreads * FragmentType::kElements;  // Total elements stored
    
    // Ensure alignment of 'FragmentType' on 128-bit boundary
    static_assert(
        FragmentType::kElements % AccessType::kElements == 0,
        "fragment not aligned on 128 bits");
    
    float* ptr;  // Pointer to data in global memory
    
    // Device function to load data from global memory into 'FragmentType'
    CUTLASS_DEVICE void load(FragmentType& fragment, int thread_id) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNumIters; ++i) {
        // Calculate global memory access pointer
        AccessType* __restrict__ gmem_ptr = reinterpret_cast<AccessType*>(
            ptr + thread_id * AccessType::kElements + i * kStride);
    
        // Temporary storage for sub-fragment of data
        AccessType sub_fragment;
    
        // Perform global memory load operation
        cutlass::arch::global_load<AccessType, kBytes>(
            sub_fragment, gmem_ptr, true);
    
        // Unroll and copy elements from 'sub_fragment' to 'fragment'
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < AccessType::kElements; ++j) {
          fragment[i * AccessType::kElements + j] = sub_fragment[j];
        }
      }
    }
    
    // Device function to store data from 'FragmentType' into global memory
    CUTLASS_DEVICE void store(FragmentType const& fragment, int thread_id) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNumIters; ++i) {
        // Calculate global memory access pointer
        AccessType* __restrict__ gmem_ptr = reinterpret_cast<AccessType*>(
            ptr + thread_id * AccessType::kElements + i * kStride);
    
        // Temporary storage for sub-fragment of data
        AccessType sub_fragment;
    
        // Unroll and copy elements from 'fragment' to 'sub_fragment'
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < AccessType::kElements; ++j) {
          sub_fragment[j] = fragment[i * AccessType::kElements + j];
        }
    
        // Perform global memory store operation
        cutlass::arch::global_store<AccessType, kBytes>(
            sub_fragment, gmem_ptr, true);
      }
    }
    
    // Device function to perform atomic addition of data from 'FragmentType' into global memory
    CUTLASS_DEVICE void storeAtomicAdd(
        FragmentType const& fragment,
        int thread_id) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNumIters; ++i) {
        // Calculate global memory access pointer
        float* gmem_ptr = ptr + thread_id * AccessType::kElements + i * kStride;
    
        // Unroll and perform atomic addition of elements from 'fragment' into global memory
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < AccessType::kElements; ++j) {
          float val = fragment[i * AccessType::kElements + j];
          float* ptr = gmem_ptr + j;
          atomicAdd(ptr, val);  // Atomic addition operation
        }
      }
    }
};

// 定义一个结构体 AtomicLock，用于实现原子锁的功能
struct AtomicLock {
  // 静态方法 acquire，用于获取锁
  CUTLASS_DEVICE static void acquire(
      int32_t* lock,      // 指向要操作的锁变量的指针
      int set_val,        // 设置锁的目标值
      int thread_id) {    // 线程 ID，用于决定哪个线程执行获取锁操作
    if (thread_id == 0) {
      // 如果是第一个线程，循环尝试原子比较并交换操作，直到成功设置锁
      while (atomicCAS(lock, 0 /*cmp*/, set_val /*setval*/) != set_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        // 在 CUDA 架构版本大于等于 7.0 时，暂停当前线程 40 纳秒
        __nanosleep(40);
#endif
      }
    }
    // 同步所有线程
    __syncthreads();
  }

  // 静态方法 release，用于释放锁
  CUTLASS_DEVICE static void release(int32_t* lock, int thread_id) {
    if (thread_id == 0) {
      int status = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      // 在 CUDA 架构版本大于等于 7.0 时，使用全局存储器释放锁
      asm volatile("st.global.release.gpu.b32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(status));
#else
      // 否则使用常规的全局存储器指令释放锁
      asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#endif
    }
  }
};

// 模板函数 getWarpsPerSmBw，根据标量类型和架构返回每个 SM 的线程束数量
template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSmBw() {
  bool is_half = !cutlass::platform::is_same<scalar_t, float>::value;
  if (Arch::kMinComputeCapability >= 80) {
    // 如果计算能力大于等于 80，根据是否半精度返回不同的线程束数
    return is_half ? 12 : 8;
  }
  // 默认返回 8 个线程束
  return 8;
}
} // namespace

// AttentionBackwardKernel 结构体模板，用于注意力机制的反向传播
template <
    typename ArchTag_,                          // 目标架构标签
    typename scalar_t_,                         // 输入/输出类型
    bool kIsAligned_,                           // 是否对齐内存访问
    bool kApplyDropout_,                        // 是否应用 dropout
    bool kPreload_,                             // 是否预加载下一个 GEMM
    int kBlockSizeI_,                           // 块的维度 I
    int kBlockSizeJ_,                           // 块的维度 J
    int kMaxK_ = (int)cutlass::platform::numeric_limits<uint32_t>::max(), // 最大 K 维度
    bool kKeysQueriesAlignedToBlockSize_ = false // 键和查询是否与块大小对齐
>
struct AttentionBackwardKernel {
  enum CustomMaskType {
    NoCustomMask = 0,        // 无自定义掩码
    CausalFromTopLeft = 1,   // 从左上角因果掩码
    CausalFromBottomRight = 2 // 从右下角因果掩码
  // 定义枚举类型 NumCustomMaskTypes，表示自定义掩码类型的数量
  NumCustomMaskTypes,
};

// 定义类型别名 scalar_t，使用 scalar_t_
using scalar_t = scalar_t_;

// 定义类型别名 output_t，使用 scalar_t 作为输出类型
using output_t = scalar_t;

// 定义类型别名 output_accum_t，使用 float 作为输出累加器类型
using output_accum_t = float;

// 定义类型别名 lse_scalar_t，使用 float 作为 LSE 标量类型
using lse_scalar_t = float;

// 定义类型别名 accum_t，使用 float 作为累加器类型
using accum_t = float;

// 定义类型别名 ArchTag，使用 ArchTag_ 作为架构标签类型
using ArchTag = ArchTag_;

// 定义静态常量 bool 类型，指示数据是否对齐到块大小
static constexpr bool kIsAligned = kIsAligned_;

// 定义静态常量 bool 类型，指示是否应用 dropout
static constexpr bool kApplyDropout = kApplyDropout_;

// 定义静态常量 bool 类型，指示是否预加载数据
static constexpr bool kPreload = kPreload_;

// 定义静态常量 int 类型，表示块的大小（维度 I）
static constexpr int kBlockSizeI = kBlockSizeI_;

// 定义静态常量 int 类型，表示块的大小（维度 J）
static constexpr int kBlockSizeJ = kBlockSizeJ_;

// 定义静态常量 int 类型，表示最大的 K 大小
static constexpr int kMaxK = kMaxK_;

// 定义静态常量 bool 类型，指示 keys 和 queries 是否对齐到块大小
static constexpr bool kKeysQueriesAlignedToBlockSize =
    kKeysQueriesAlignedToBlockSize_;

// 定义静态常量 int64_t 类型，表示线程束大小（warp size）
static constexpr int64_t kWarpSize = 32;

// 如果 scalar_t 的位数小于等于 16，则设置 kIsHalf 为 true
static constexpr bool kIsHalf = cutlass::sizeof_bits<scalar_t>::value <= 16;

// 如果 kIsHalf 为 true 并且 kMaxK 小于等于 kBlockSizeI，则设置 kOutputInRF 为 true
static constexpr bool kOutputInRF = kIsHalf && kMaxK <= kBlockSizeI;

// 静态断言，如果预加载开启，则要求 kIsHalf 为 true、架构支持的最小计算能力大于等于 80、并且 kOutputInRF 为 true
static_assert(
    !kPreload ||
        (kIsHalf && ArchTag::kMinComputeCapability >= 80 && kOutputInRF),
    "preload MMA not supported");

// 定义一系列静态常量 bool 类型，指示是否在前处理阶段预加载不同的数据
static constexpr bool kPrologueQK = kPreload;
static constexpr bool kPrologueGV = kPreload;
static constexpr bool kPrologueDOV = kPreload;
static constexpr bool kPrologueGQ = kPreload;
static constexpr bool kPrologueGK = kPreload;

// 定义静态常量 int64_t 类型，表示每个块的线程束数量
static constexpr int64_t kNumWarpsPerBlock =
    (kBlockSizeI * kBlockSizeJ) / (32 * 32);

// 定义静态常量 bool 类型，指示是否在 f16 内核中计算 delta
static constexpr bool kKernelComputesDelta =
    kIsHalf && (kOutputInRF || ArchTag::kMinComputeCapability != 70);

// 定义静态常量 int64_t 类型，表示每个线程束的线程数
static constexpr int64_t kNumThreads = kWarpSize * kNumWarpsPerBlock;

// 定义静态常量 int64_t 类型，表示每个 SM 上的最小块数
static constexpr int64_t kMinBlocksPerSm =
    getWarpsPerSmBw<scalar_t, ArchTag>() / kNumWarpsPerBlock;

// 定义 GemmType 别名，使用 ArchTag 和 scalar_t 作为默认的 Gemm 类型
using GemmType = DefaultGemmType<ArchTag, scalar_t>;

// 定义 DefaultConfig 类型别名，使用 GemmType 的默认 Gemm 配置
using DefaultConfig =
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        typename GemmType::OpClass,
        ArchTag,
        scalar_t,
        scalar_t,
        scalar_t, // ElementC
        accum_t // ElementAccumulator
        >;

// 计算最佳对齐值，选择 DefaultConfig 的 A 和 B 的最大对齐值
static constexpr auto kOptimalAlignement = cutlass::platform::max(
    DefaultConfig::kAlignmentA,
    DefaultConfig::kAlignmentB);

// 定义 GemmType 的最小对齐值
static constexpr auto kMinimumAlignment = GemmType::kMinimumAlignment;

// 定义 MatmulQK 结构体
struct MatmulQK {
  /*
  attn_T = k_j @ q_i.transpose(-2, -1) # 矩阵乘法
  attn_T = (attn_T - logsumexp[i_start:i_end].unsqueeze(1).transpose(-2,
  -1)).exp() # 结尾处理

  其中 attn_T.shape = (kBlockSizeJ, kBlockSizeI)
  */
  // 定义 ThreadblockShape 别名，使用 kBlockSizeJ、kBlockSizeI 和 GemmType 的 ThreadK 作为 GEMM 的形状
  using ThreadblockShape =
      cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    // 定义 Warp 的形状，用于矩阵乘法的线程块

    using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
        scalar_t, // ElementA，A矩阵元素类型
        cutlass::layout::RowMajor, // LayoutA，A矩阵布局
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment, // A矩阵对齐方式
        scalar_t, // ElementB，B矩阵元素类型
        cutlass::layout::ColumnMajor, // LayoutB，B矩阵布局
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment, // B矩阵对齐方式
        accum_t, // ElementC，C矩阵元素类型
        cutlass::layout::RowMajor, // LayoutC，C矩阵布局
        typename GemmType::OpClass, // 运算类型
        ArchTag, // 架构标签
        ThreadblockShape, // 线程块的形状
        WarpShape, // Warp的形状
        typename GemmType::InstructionShape, // 指令形状
        DefaultConfig::kStages, // 操作阶段数
        typename GemmType::Operator,
        false, // AccumulatorsInRowMajor，累加器是否行主要（false为列主要）
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    // 定义默认的 MMA（矩阵乘法加法）操作类

    using MmaCore = typename DefaultMma::MmaCore;
    // 定义 MMA 核心类型

    using Mma =
        typename MakeCustomMma<typename DefaultMma::ThreadblockMma, kMaxK>::Mma;
    // 定义自定义的 MMA（矩阵乘法加法）操作类

    // 用于从全局内存高效加载偏置矩阵块（Bij）到共享内存
    using BiasLoader = TileSmemLoader<
        scalar_t,
        // Bij 应用于转置后的注意力矩阵块 (Pij.T)。Bij 是按行主序加载的，但需要具有转置形状，以便获得相同的元素。
        cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kM>,
        MmaCore::kThreads,
        // 输入限制: kv_len 必须是该值的倍数
        128 / cutlass::sizeof_bits<scalar_t>::value>;
    // 定义从全局内存加载偏置块的模板类

    // 存储至共享内存的后处理，以便后续第二次矩阵乘法使用
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    // 定义用于第二次矩阵乘法的后处理操作类

    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Iterator;
    // 定义默认 MMA 累加器的迭代器

    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
    // 定义 B2bGemm 的累加器共享存储类型
  };

  struct MatmulGradV {
    /*
    grad_v[j_start:j_end] += attn_T @ do_i # matmul

    Dimensions: (kBlockSizeJ * kNumWarpsPerBlock, kBlockSizeI, K)
    (we might need to iterate multiple times on K)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    // 定义用于矩阵乘法线程块的形状

    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    // 定义 Warp 的形状，用于矩阵乘法的线程块

    using InstructionShape = typename GemmType::InstructionShape;
    // 定义指令形状类型
    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA, 数据类型 ElementA，表示矩阵元素类型
        cutlass::layout::RowMajor, // LayoutA, 矩阵 A 的布局，行主序
        DefaultConfig::kAlignmentA,  // 对齐方式
        scalar_t, // ElementB, 数据类型 ElementB，表示矩阵元素类型
        cutlass::layout::RowMajor, // LayoutB, 矩阵 B 的布局，行主序
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment, // 对齐方式
        output_t,  // 输出元素类型
        cutlass::layout::RowMajor, // LayoutC, 输出矩阵的布局，行主序
        accum_t,  // 累加器类型
        typename GemmType::OpClass,  // 操作类型
        ArchTag,  // 架构标签
        ThreadblockShape,  // Threadblock 的形状
        WarpShape,  // Warp 的形状
        typename GemmType::InstructionShape,  // 指令形状
        typename DefaultConfig::EpilogueOutputOp,  // 输出 epilogue 操作类型
        void, // ThreadblockSwizzle - not used, 线程块重排，未使用
        DefaultConfig::kStages,  // 阶段数
        false, // SplitKSerial, 是否串行分割 K 维度
        typename GemmType::Operator>;  // Gemm 运算器类型

    // if dropout:
    //   for computing dVj += (Pij.T * Zij) @ dOi
    //   Pij_dropped.T = Pij.T * Zij is computed on the fly as fragments of
    //   Pij.T are loaded in. The reason we do it this way is because Pij.T and
    //   Zij are reused in later steps, while Pij_dropped.T is only needed in
    //   this step. computing Pij_dropped.T on the fly allows us to avoid
    //   keeping all 3 of Pij_dropped.T, Pij.T, and Zij in shared memory at the
    //   same time.
    // if no dropout:
    //   for computing dVj += Pij.T @ dOi
    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape, // WarpShape, Warp 的形状
            typename DefaultGemm::Mma::Operator::
                InstructionShape, // InstructionShape, 指令形状
            typename DefaultGemm::Mma::Operator::
                IteratorA, // RegularWarpIterator, 常规 Warp 迭代器
            typename DefaultGemm::Mma::Policy // Policy, 策略
            >::WarpIterator;

    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,  // Mma 类型
            MatmulQK::AccumulatorSharedStorage::Shape::kN,  // 累加器共享存储的形状
            WarpIteratorA,  // Warp 迭代器 A 类型
            kApplyDropout>; // kScaleOperandA, 是否应用 dropout

    using Mma = typename DefaultMmaFromSmem::Mma;  // Mma 运算器类型
    using IteratorB = typename Mma::IteratorB;  // 迭代器 B 类型
    using WarpCount = typename Mma::WarpCount;  // Warp 计数类型

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;  // 默认的输出操作类型
    using DefaultEpilogue = typename DefaultGemm::Epilogue;  // 默认的 Epilogue 类型
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;  // 输出瓦片迭代器类型
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;  // 累加瓦片在全局内存中的类型
  };

  struct MatmulDOIVJ {
    /*
    doi_t_vj = do_i @ v_j.transpose(-2, -1) # matmul
    tmp = (doi_t_vj - Di.unsqueeze(1)) * attn # inplace / epilogue?
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;  // Threadblock 的形状
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;  // Warp 的形状

    using ElementC = output_t;  // 元素类型 C
    using ElementAccum = accum_t;  // 累加器元素类型
    // 定义一个无操作的输出操作，将结果存储到全局内存
    using BiasGradEpilogueOutputOp =
        typename cutlass::epilogue::thread::LinearCombination<
            ElementC,
            DefaultConfig::EpilogueOutputOp::kCount,
            typename DefaultConfig::EpilogueOutputOp::ElementAccumulator,
            typename DefaultConfig::EpilogueOutputOp::ElementCompute,
            cutlass::epilogue::thread::ScaleType::Nothing>;

    // 定义默认的 GEMM（General Matrix Multiply）配置
    using DefaultGemm = typename cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA，矩阵A的元素类型
        cutlass::layout::RowMajor, // LayoutA，矩阵A的布局方式
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment, // 对齐方式
        scalar_t, // ElementB，矩阵B的元素类型
        cutlass::layout::ColumnMajor, // LayoutB，矩阵B的布局方式
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment, // 对齐方式
        ElementC, // ElementC，矩阵C的元素类型
        cutlass::layout::RowMajor, // LayoutC，矩阵C的布局方式
        ElementAccum, // ElementAccumulator，累加器的元素类型
        typename GemmType::OpClass, // 操作类别
        ArchTag, // 架构标签
        ThreadblockShape, // 线程块形状
        WarpShape, // Warp形状
        typename GemmType::InstructionShape, // 指令形状
        BiasGradEpilogueOutputOp, // EpilogueOutputOp，后处理输出操作
        void, // ThreadblockSwizzle（未使用）
        // 当使用预加载、应用dropout且Zij瓦片超过64x64时，多个预加载和3个阶段会超出A100的共享内存容量。为了节省共享内存，设定阶段数上限。
        kPreload && kApplyDropout && (kBlockSizeI * kBlockSizeJ > 64 * 64)
            ? cutlass::const_min(2, DefaultConfig::kStages)
            : DefaultConfig::kStages, // Stages，阶段数
        false, // SplitKSerial，是否串行分割K
        typename GemmType::Operator, // 操作器类型
        cutlass::gemm::SharedMemoryClearOption::kNone>; // 共享内存清除选项

    // 定义MMA（Mixed Matrix-Vector Multiply）操作
    using Mma = typename MakeCustomMma<typename DefaultGemm::Mma, kMaxK>::Mma;

    // 定义累加器Lambda迭代器类型
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        ElementAccum,
        kWarpSize>::Iterator;

    // 定义用于写入偏置梯度的后处理操作
    using BiasGradEpilogue = typename DefaultGemm::Epilogue;

    // 定义用于将数据存储到共享内存中，以便稍后用于第二次矩阵乘法的B2bGemm（Block-to-Block Gemm）操作
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename DefaultGemm::Mma::Operator::IteratorC,
        typename DefaultGemm::Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;

    // 定义用于存储累加器的共享存储空间类型
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
};
    // 使用默认的 GEMM（General Matrix Multiply）实现定义
    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA, 第一个矩阵元素类型
        cutlass::layout::RowMajor, // LayoutA, 第一个矩阵布局（行主序）
        DefaultConfig::kAlignmentA, 第一个矩阵对齐方式
        scalar_t, // ElementB, 第二个矩阵元素类型
        cutlass::layout::RowMajor, // LayoutB, 第二个矩阵布局（行主序）
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment, // 第二个矩阵对齐方式，根据条件选择
        output_t, // 输出元素类型
        cutlass::layout::RowMajor, // LayoutC, 输出矩阵布局（行主序）
        accum_t, // 累加器类型
        typename GemmType::OpClass, // 操作类别
        ArchTag, // 架构标签
        ThreadblockShape, // Threadblock 形状
        WarpShape, // Warp 形状
        typename GemmType::InstructionShape, // 指令形状
        typename DefaultConfig::EpilogueOutputOp, // 终结操作输出操作类型
        void, // ThreadblockSwizzle - 不使用的线程块重排方式
        DefaultConfig::kStages, // GEMM 计算阶段数
        false, // SplitKSerial - 是否串行分割 K 维度
        typename GemmType::Operator>; // GEMM 操作符类型

    // 使用默认的 Warp 迭代器 A，从共享内存中获取
    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape, // Mma 操作符形状
            typename DefaultGemm::Mma::Operator::InstructionShape, // Mma 操作符指令形状
            typename DefaultGemm::Mma::Operator::IteratorA, // Mma 操作符 A 迭代器类型
            typename DefaultGemm::Mma::Policy>::WarpIterator;

    // 使用默认的从共享内存中的 Mma 操作
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma, // 默认的 Mma 操作类型
            MatmulDOIVJ::AccumulatorSharedStorage::Shape::kN, // 累加器共享存储的形状
            WarpIteratorA, // Warp 迭代器 A
            false>; // kScaleOperandA - A 操作数是否需要缩放

    // Mma 类型定义
    using Mma = typename DefaultMmaFromSmem::Mma;

    // Mma 操作符 B 的迭代器类型
    using IteratorB = typename Mma::IteratorB;

    // Mma 的 Warp 数量
    using WarpCount = typename Mma::WarpCount;

    // Epilogue 部分定义

    // 默认的输出操作类型
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;

    // 默认的 Epilogue 类型
    using DefaultEpilogue = typename DefaultGemm::Epilogue;

    // 输出瓦片迭代器类型
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;

    // 累加器全局内存瓦片类型
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;

  };

  // MatmulGradK 结构体定义
  struct MatmulGradK {

    // grad_k <- tmp.transpose(-2, -1) @ q_i

    // Threadblock 形状定义
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;

    // Warp 形状定义
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;

    // 指令形状定义
    using InstructionShape = typename GemmType::InstructionShape;
    // 使用默认的 GEMM 内核类型进行矩阵乘法运算
    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA, 矩阵元素类型 A
        cutlass::layout::RowMajor, // LayoutA, A 矩阵布局为行主序
        DefaultConfig::kAlignmentA, // A 矩阵的对齐方式
        scalar_t, // ElementB, 矩阵元素类型 B
        cutlass::layout::RowMajor, // LayoutB, B 矩阵布局为行主序
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment, // B 矩阵的对齐方式
        output_t, // 输出元素类型
        cutlass::layout::RowMajor, // LayoutC, 输出矩阵布局为行主序
        accum_t, // 累加器类型
        typename GemmType::OpClass, // GEMM 运算类别
        ArchTag, // 架构标签
        ThreadblockShape, // 线程块形状
        WarpShape, // 线程束形状
        typename GemmType::InstructionShape, // 指令形状
        typename DefaultConfig::EpilogueOutputOp, // 输出后处理操作
        void, // ThreadblockSwizzle - 不使用线程块重排
        DefaultConfig::kStages, // GEMM 运算阶段数
        false, // SplitKSerial - 不进行 K 维度的串行分割
        typename GemmType::Operator>; // GEMM 操作类型

    // 使用从共享内存中创建默认的 Warp 迭代器类型 A
    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape, // 操作形状
            typename DefaultGemm::Mma::Operator::InstructionShape, // 指令形状
            typename DefaultGemm::Mma::Operator::IteratorA, // 迭代器 A 类型
            typename DefaultGemm::Mma::Policy>::WarpIterator;

    // 使用从共享内存中创建的默认 Mma 类型，针对 N 维度的操作
    using DefaultMmaFromSmemN =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma, // GEMM Mma 类型
            MatmulQK::AccumulatorSharedStorage::Shape::kN, // 最大 K 维度
            WarpIteratorA, // Warp 迭代器类型 A
            false>; // 不进行操作数 A 的缩放

    // 使用从共享内存中创建的默认 Mma 类型，针对 T 维度的操作
    using DefaultMmaFromSmemT =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma, // GEMM Mma 类型
            MatmulDOIVJ::AccumulatorSharedStorage::Shape::kM, // 最大 K 维度
            WarpIteratorA, // Warp 迭代器类型 A
            false, // 不进行操作数 A 的缩放
            kPreload>; // 对 A 矩阵进行预加载

    // 根据条件选择 T 或 N 维度的 Mma 类型
    using DefaultMmaFromSmem = typename cutlass::platform::conditional<
        DefaultMmaFromSmemT::kIsTransposedA, // 根据 A 是否转置来选择
        DefaultMmaFromSmemT,
        DefaultMmaFromSmemN>::type;

    // 提取出 Mma 类型
    using Mma = typename DefaultMmaFromSmem::Mma;

    // 提取出 B 矩阵的迭代器类型
    using IteratorB = typename Mma::IteratorB;

    // 提取出 Warp 计数类型
    using WarpCount = typename Mma::WarpCount;

    // Epilogue 阶段的类型
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;

    // 默认的 Epilogue 类型
    using DefaultEpilogue = typename DefaultGemm::Epilogue;

    // 创建可预取迭代器类型，用于输出矩阵块
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;
  };

  // NOTE: nvcc 12.4 has correctness errors with this on M60 (sm52)
  // when there is an attention bias. Let's just disable it for now.
  static constexpr auto kMinSm = ArchTag::kMinComputeCapability;
  static constexpr bool kEnableSplitKeys = kMinSm >= 70;

  // 根据硬件架构的最小计算能力确定是否启用分割键功能
  static constexpr bool kNeedsAccumGradQ = kEnableSplitKeys ||
      !cutlass::platform::is_same<output_accum_t, output_t>::value;
  // 根据条件判断是否需要累积 gradQ
  static constexpr bool kNeedsAccumGradK = !kOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;
  // 根据条件判断是否需要累积 gradK
  static constexpr bool kNeedsAccumGradV = !kOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  struct GradQTempStorage {
    int32_t lock;
    int32_t counter;
    int32_t pad[2]; // 用于填充到128位
    output_accum_t buffer[MatmulGradQ::AccumTileGmem::kElementsStored];
    // 存储用于 gradQ 累积的临时缓冲区
  };

  struct Params {
    // 输入张量
    const scalar_t* query_ptr = nullptr; // [Mq, nH, K]，查询张量指针
    const scalar_t* key_ptr = nullptr; // [Mk, nH, K]，键张量指针
    const scalar_t* value_ptr = nullptr; // [Mk, nH, Kv]，值张量指针
    const scalar_t* bias_ptr = nullptr; // 偏置张量指针
    const lse_scalar_t* logsumexp_ptr = nullptr; // [nH, Mq]，对数和指数张量指针
    const scalar_t* output_ptr = nullptr; // [Mq, nH, Kv]，输出张量指针
    const scalar_t* grad_output_ptr = nullptr; // [Mq, nH, Kv]，梯度输出张量指针
    accum_t* delta_ptr = nullptr; // [nH, Mq]，增量张量指针
    const int32_t* cu_seqlens_q_ptr = nullptr; // 查询序列长度指针
    const int32_t* cu_seqlens_k_ptr = nullptr; // 键序列长度指针

    // 输出张量
    output_t* grad_query_ptr = nullptr; //  [Mq, nH, K]，查询梯度张量指针
    output_t* grad_key_ptr = nullptr; //    [Mk, nH, K]，键梯度张量指针
    output_t* grad_value_ptr = nullptr; //  [Mk, nH, Kv]，值梯度张量指针
    output_t* grad_bias_ptr = nullptr; // 偏置梯度张量指针

    // 累加器
    output_accum_t* workspace = nullptr; // [Mq, Kq] + [Mkv, Kq] + [Mkv, Kv]，工作空间张量指针
    output_accum_t* workspace_gv =
        nullptr; // (将由核函数计算)，值梯度工作空间张量指针
    GradQTempStorage* workspace_gq =
        nullptr; // (将由核函数计算)，查询梯度工作空间张量指针

    // 滑动窗口。如果为0则忽略
    int32_t window_size = 0;

    // 缩放因子
    accum_t scale = 1.0f;

    // 维度/步长
    int32_t head_dim = -1; // 头维度
    int32_t head_dim_value = -1; // 值头维度
    int32_t num_queries = -1; // 查询数量
    int32_t num_keys = -1; // 键数量
    int32_t num_heads = -1; // 头数量
    uint8_t custom_mask_type = NoCustomMask; // 自定义掩码类型

    int32_t q_strideM = -1; // 查询步长
    int32_t k_strideM = -1; // 键步长
    int32_t v_strideM = -1; // 值步长
    int32_t bias_strideM = 0; // 偏置步长
    int32_t gO_strideM = -1; // 梯度输出步长
    int32_t gB_strideM = -1; // 偏置梯度步长
    int8_t gQKV_strideM_multiplier = 1; // 3表示打包，否则为1

    at::PhiloxCudaState rng_engine_inputs = {0, 0}; // 随机数生成引擎输入状态

    // 基于 batch_id 和 head_id 的随机数生成序列偏移量
    unsigned long long dropout_batch_head_rng_offset = 0;
    float dropout_prob = 0.0f; // 丢弃概率

    CUTLASS_HOST_DEVICE int32_t o_strideM() const {
      return head_dim_value * num_heads; // 计算输出步长
    }
    CUTLASS_HOST_DEVICE int32_t gQ_strideM() const {
      return gQKV_strideM_multiplier * num_heads * head_dim; // 计算 gradQ 步长
    }
  };
    // 返回 gK_strideM 的值，计算为 gQKV_strideM_multiplier 乘以 num_heads 乘以 head_dim
    CUTLASS_HOST_DEVICE int32_t gK_strideM() const {
      return gQKV_strideM_multiplier * num_heads * head_dim;
    }
    
    // 返回 gV_strideM 的值，计算为 gQKV_strideM_multiplier 乘以 num_heads 乘以 head_dim_value
    CUTLASS_HOST_DEVICE int32_t gV_strideM() const {
      return gQKV_strideM_multiplier * num_heads * head_dim_value;
    }

    // 以下所有变量仅在 `advance_to_block` 中使用，并且不应使用寄存器

    // 输出流的步长
    int64_t o_strideH = -1;
    // 查询流的步长
    int32_t q_strideH = -1;
    // 键流的步长
    int32_t k_strideH = -1;
    // 值流的步长
    int32_t v_strideH = -1;
    // 偏置流的步长
    int64_t bias_strideH = 0;
    // 输出流的批次步长
    int64_t o_strideB = -1;
    // 查询流的批次步长
    int64_t q_strideB = -1;
    // 键流的批次步长
    int64_t k_strideB = -1;
    // 值流的批次步长
    int64_t v_strideB = -1;
    // 偏置流的批次步长
    int64_t bias_strideB = 0;
    // LSE 流的批次步长
    int64_t lse_strideB = -1;
    // LSE 流的步长
    int64_t lse_strideH = -1;
    // Delta 流的批次步长
    int64_t delta_strideB = -1;
    // Delta 流的步长
    int64_t delta_strideH = -1;
    // 批次数目
    int32_t num_batches = -1;
    // 键的分割数
    int16_t num_splits_key = 1; // 我们在内核中使用 `gridDim.x`

    // 全局输出流的批次步长
    int64_t gO_strideB = 0;
    // 全局查询流的批次步长
    int64_t gQ_strideB = 0;
    // 全局键流的批次步长
    int64_t gK_strideB = 0;
    // 全局值流的批次步长
    int64_t gV_strideB = 0;
    // 全局偏置流的批次步长
    int64_t gB_strideB = 0;
    // 全局输出流的步长
    int64_t gO_strideH = 0;
    // 全局查询流的步长
    int64_t gQ_strideH = 0;
    // 全局键流的步长
    int64_t gK_strideH = 0;
    // 全局值流的步长
    int64_t gV_strideH = 0;
    // 全局偏置流的步长
    int64_t gB_strideH = 0;

    // 返回设备上的键分割数目，内核中使用 `gridDim.x`
    CUTLASS_HOST_DEVICE int16_t num_splits_key_device() const {
    // 如果在 CUDA 设备上编译，根据条件返回 gridDim.x 或 1
    #ifdef __CUDA_ARCH__
      return kEnableSplitKeys ? gridDim.x : 1;
    // 如果在主机端测试，直接返回 num_splits_key
    #else
      return num_splits_key; // for host-side tests
    #endif
    }

    // 返回用于设备端的分割键值，根据条件返回 blockIdx.x 或 0
    CUTLASS_HOST_DEVICE int16_t split_key_device() const {
    // 如果在 CUDA 设备上编译，根据条件返回 blockIdx.x 或 0
    #ifdef __CUDA_ARCH__
      return kEnableSplitKeys ? blockIdx.x : 0;
    // 如果在主机端测试，直接返回 0
    #else
      return 0; // for host-side tests
    #endif
    }

    // 打印调试信息，并返回 true
    #if 0
      PRINT_T0("[b:%d h:%d] dp[0]:%f Q:%f K:%f V:%f LSE:%f",
        int(blockIdx.z), int(blockIdx.y),
        float(delta_ptr[0]),
        float(query_ptr[0]), float(key_ptr[0]), float(value_ptr[0]),
        float(logsumexp_ptr[0])
      )
    #endif
      return true;
    }

    // 返回用于块和网格的维度尺寸
    __host__ dim3 getBlocksGrid() const {
      return dim3(num_splits_key, num_heads, num_batches);
    }

    // 返回用于线程网格的维度尺寸
    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize * kNumWarpsPerBlock, 1, 1);
    }

    // 返回用于计算 gk（key 梯度）的工作空间元素数量
    CUTLASS_HOST_DEVICE int64_t workspace_elements_gk() const {
      // 如果不需要计算 key 梯度，返回 0
      if (!kNeedsAccumGradK) {
        return 0;
      }
      // 计算并返回工作空间元素数量
      return num_splits_key * kBlockSizeJ *
          align_up(head_dim, (int32_t)kBlockSizeI);
    }

    // 返回用于计算 gv（value 梯度）的工作空间元素数量
    CUTLASS_HOST_DEVICE int64_t workspace_elements_gv() const {
      // 如果不需要计算 value 梯度，返回 0
      if (!kNeedsAccumGradV) {
        return 0;
      }
      // 计算并返回工作空间元素数量
      return num_splits_key * kBlockSizeJ *
          align_up(head_dim_value, (int32_t)kBlockSizeI);
    }

    // 返回用于计算 gq（query 梯度）的工作空间元素数量
    CUTLASS_HOST_DEVICE int64_t workspace_elements_gq() const {
      // 如果不需要计算 query 梯度，返回 0
      if (!kNeedsAccumGradQ) {
        return 0;
      }
      // 计算 num_blocks 和 num_cols，返回工作空间元素数量
      int num_blocks = ceil_div(num_queries, kBlockSizeI);
      int num_cols = ceil_div(head_dim, MatmulGradQ::ThreadblockShape::kN);
      return num_blocks * num_cols * sizeof(GradQTempStorage) /
          sizeof(output_accum_t);
    }

    // 返回工作空间的步长 BH
    CUTLASS_HOST_DEVICE int64_t workspace_strideBH() const {
      // 返回工作空间元素数量的 128 位对齐值
      return align_up(
          workspace_elements_gk() + workspace_elements_gv() +
              workspace_elements_gq(),
          int64_t(4));
    }

    // 返回需要运行该核函数所需的缓冲区大小
    CUTLASS_HOST_DEVICE int64_t workspace_size() const {
      // 返回每个批次、头部和 BH 步长的浮点数大小的缓冲区大小
      return num_batches * num_heads * workspace_strideBH() * sizeof(float);
    }

    // 返回是否需要将工作空间清零的布尔值
    CUTLASS_HOST_DEVICE bool should_zero_workspace() const {
      // 如果 split 数量大于 1 或窗口大小大于 0，返回 true
      return num_splits_key > 1 || window_size > 0;
    }
  };

  // 用于保存 Zij 矩阵的共享存储。如果不使用 dropout，则使用空数组节省共享内存空间
  using ZijSharedStorage = typename cutlass::platform::conditional<
      kApplyDropout,
      typename MatmulQK::AccumulatorSharedStorage,
      // 在不占用空间的情况下使用的虚拟共享存储对象
      typename cutlass::gemm::threadblock::AccumulatorSharedStorage<
#ifdef _WIN32
          // 在 Windows 下编译时，由于未知大小数组不被允许，将 Zij 共享存储的大小设为 1
          typename cutlass::gemm::GemmShape<1, 1, 0>,
#else
          // 在其他平台上，将 Zij 共享存储大小设为 0
          typename cutlass::gemm::GemmShape<0, 0, 0>,
#endif
#endif
          typename MatmulQK::AccumulatorSharedStorage::Element,
          typename MatmulQK::AccumulatorSharedStorage::Layout,
          typename cutlass::MatrixShape<0, 0>>>::type;


// 结束条件预处理指令，该行结束了一个条件编译块

  struct SharedStoragePrologue {
    struct {
      // 存储器中的数组，用于存储 do_i * o_i 的和（在某个维度上求和）
      cutlass::Array<accum_t, kBlockSizeI> di;

      // MatmulQK 类中 Mma 结构的共享存储 A，用于矩阵乘法加速
      typename MatmulQK::Mma::SharedStorageA mm_qk_k;
    } persistent;
    union {
      struct {
        // part1 - after Q.K / dV / dO.V
        union {
          // 1. efficient load of bias tile Bij, which is then applied to Pij
          typename MatmulQK::BiasLoader::SmemTile bias;
          // 4. store Pij. it is needed:
          // - in dVj += (Pij.T * Zij) @ dOi
          // - in dSij = Pij * (dPij - Di)
          // 6. dVj += (Pij.T * Zij) @ dOi
          // 10. write to fragment
          typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
        };
        // 5. store Zij. it is needed in dVj += (Pij.T * Zij) @ dOi
        ZijSharedStorage zij;

        union {
          // 2. prologue for dVj
          // 6. workspace for dVj += (Pij.T * Zij) @ dOi
          typename MatmulGradV::Mma::SharedStorage mm_gradV;
          // 7. dVj epilogue
          typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
        };

        // 3. prologue for dPij_dropped
        // 8. used in dPij_dropped = dOi @ Vj.T
        typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
      } part1;

      struct {
        // part2 - dQ
        union {
          typename MatmulQK::AccumulatorSharedStorage
              tmpT_shared_storage; // (from part1)
          typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        };
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::Mma::SharedStorage mm_gradQ; // (preload)
        union {
          // store dB = dSij to global memory
          typename MatmulDOIVJ::BiasGradEpilogue::SharedStorage gradB_epilogue;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;
        };

      } part2;

      struct {
        // part3 - after last iteration on dQ's epilogue / dK
        union {
          typename MatmulQK::AccumulatorSharedStorage
              tmpT_shared_storage; // (from part1)
          typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        };
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::DefaultEpilogue::SharedStorage
            gradQ_epilogue_lastIter;

        typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
      } part3;

      struct {
        // part4 - after last iteration on dK's epilogue / preload next K.Q_t
        typename MatmulQK::Mma::SharedStorageB mm_qk_q;

        // If we reach end of current key, dump RF->gmem with "final" epilogues
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } part4;
    };
    static void print_size() {
      // Field size
      // 这是一个静态方法，用于打印结构体的大小信息，但在代码中没有具体的实现

    union {
      struct {
        // part1 - after Q.K / dV / dO.V
        union {
          // 1. efficient load of bias tile Bij, which is then applied to Pij
          typename MatmulQK::BiasLoader::SmemTile bias;
          // 4. store Pij. it is needed:
          // - in dVj += (Pij.T * Zij) @ dOi
          // - in dSij = Pij * (dPij - Di)
          // 6. dVj += (Pij.T * Zij) @ dOi
          // 10. write to fragment
          typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
        };
        // 5. store Zij. it is needed in dVj += (Pij.T * Zij) @ dOi
        ZijSharedStorage zij;

        union {
          // 2. prologue for dVj
          // 6. workspace for dVj += (Pij.T * Zij) @ dOi
          typename MatmulGradV::Mma::SharedStorage mm_gradV;
          // 7. dVj epilogue
          typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
        };

        // 3. prologue for dPij_dropped
        // 8. used in dPij_dropped = dOi @ Vj.T
        typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
      } part1;

      struct {
        // part2 - dQ
        union {
          typename MatmulQK::AccumulatorSharedStorage
              tmpT_shared_storage; // (from part1)
          typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        };
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::Mma::SharedStorage mm_gradQ; // (preload)
        union {
          // store dB = dSij to global memory
          typename MatmulDOIVJ::BiasGradEpilogue::SharedStorage gradB_epilogue;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;
        };

      } part2;

      struct {
        // part3 - after last iteration on dQ's epilogue / dK
        union {
          typename MatmulQK::AccumulatorSharedStorage
              tmpT_shared_storage; // (from part1)
          typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        };
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::DefaultEpilogue::SharedStorage
            gradQ_epilogue_lastIter;

        typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
      } part3;

      struct {
        // part4 - after last iteration on dK's epilogue / preload next K.Q_t
        typename MatmulQK::Mma::SharedStorageB mm_qk_q;

        // If we reach end of current key, dump RF->gmem with "final" epilogues
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } part4;
    };
    static void print_size() {
      // Field size
      // 这是一个静态方法，用于打印结构体的大小信息，但在代码中没有具体的实现
    }
// 输出整个结构体中字段 persistent 的大小（以字节为单位）
printf("Total smem: %d bytes\n", int(sizeof(SharedStoragePrologue)));

// 输出 persistent 结构体中字段 mm_qk_k 的大小（以字节为单位）
printf("  persistent: %db\n", FSZ(persistent));
// 输出 persistent 结构体中字段 mm_qk_k 中的子字段 mm_qk_k 的大小（以字节为单位）
printf("    mm_qk_k: %db\n", FSZ(persistent.mm_qk_k));

// 输出结构体中字段 part1 的大小（以字节为单位）
printf("  part1: %db\n", FSZ(part1));
// 输出结构体中字段 part1 中的子字段 bias 的大小（以字节为单位）
printf("    bias: %db\n", FSZ(part1.bias));
// 输出结构体中字段 part1 中的子字段 attn_shared_storage 的大小（以字节为单位）
printf("    attn_shared_storage: %db\n", FSZ(part1.attn_shared_storage));
// 输出结构体中字段 part1 中的子字段 zij 的大小（以字节为单位）
printf("    zij: %db\n", FSZ(part1.zij));
// 输出结构体中字段 part1 中的子字段 mm_gradV 的大小（以字节为单位）
printf("    mm_gradV: %db\n", FSZ(part1.mm_gradV));
// 输出结构体中字段 part1 中的子字段 gradV_epilogue 的大小（以字节为单位）
printf("    gradV_epilogue: %db\n", FSZ(part1.gradV_epilogue));
// 输出结构体中字段 part1 中的子字段 mm_doivj 的大小（以字节为单位）
printf("    mm_doivj: %db\n", FSZ(part1.mm_doivj));

// 输出结构体中字段 part2 的大小（以字节为单位）
printf("  part2: %db\n", FSZ(part2));
// 输出结构体中字段 part2 中的子字段 tmpT_shared_storage 的大小（以字节为单位）
printf("    tmpT_shared_storage: %db\n", FSZ(part2.tmpT_shared_storage));
// 输出结构体中字段 part2 中的子字段 tmp_shared_storage 的大小（以字节为单位）
printf("    tmp_shared_storage: %db\n", FSZ(part2.tmp_shared_storage));
// 输出结构体中字段 part2 中的子字段 mm_gradK 的大小（以字节为单位）
printf("    mm_gradK: %db\n", FSZ(part2.mm_gradK));
// 输出结构体中字段 part2 中的子字段 mm_gradQ 的大小（以字节为单位）
printf("    mm_gradQ: %db\n", FSZ(part2.mm_gradQ));
// 输出结构体中字段 part2 中的子字段 gradB_epilogue 的大小（以字节为单位）
printf("    gradB_epilogue: %db\n", FSZ(part2.gradB_epilogue));
// 输出结构体中字段 part2 中的子字段 gradQ_epilogue 的大小（以字节为单位）
printf("    gradQ_epilogue: %db\n", FSZ(part2.gradQ_epilogue));

// 输出结构体中字段 part3 的大小（以字节为单位）
printf("  part3: %db\n", FSZ(part3));
// 输出结构体中字段 part3 中的子字段 tmpT_shared_storage 的大小（以字节为单位）
printf("    tmpT_shared_storage: %db\n", FSZ(part3.tmpT_shared_storage));

// 输出结构体中字段 part4 的大小（以字节为单位）
printf("  part4: %db\n", FSZ(part4));
// 输出结构体中字段 part4 中的子字段 mm_qk_q 的大小（以字节为单位）
printf("    mm_qk_q: %db\n", FSZ(part4.mm_qk_q));
// 输出结构体中字段 part4 中的子字段 gradK_epilogue_final 的大小（以字节为单位）
printf("    gradK_epilogue_final: %db\n", FSZ(part4.gradK_epilogue_final));
// 输出结构体中字段 part4 中的子字段 gradV_epilogue_final 的大小（以字节为单位）
printf("    gradV_epilogue_final: %db\n", FSZ(part4.gradV_epilogue_final));
}

// 定义宏 FIELD，用于在类或结构体的成员函数中获取内部结构体或类的字段的引用
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE auto& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

// 以下为多个 FIELD 宏的使用示例，返回对应字段的引用
FIELD(persistent, di)
FIELD(persistent, mm_qk_k)
FIELD(part1, bias)
FIELD(part1, attn_shared_storage)
FIELD(part1, zij)
FIELD(part1, mm_gradV)
FIELD(part1, gradV_epilogue)
FIELD(part1, mm_doivj)
FIELD(part2, mm_gradK)
FIELD(part2, mm_gradQ)
FIELD(part2, gradB_epilogue)
FIELD(part2, gradQ_epilogue)
FIELD(part2, tmp_shared_storage)
FIELD(part3, tmpT_shared_storage)
FIELD(part3, gradQ_epilogue_lastIter)
FIELD(part3, gradK_epilogue)
FIELD(part4, mm_qk_q)
FIELD(part4, gradK_epilogue_final)
FIELD(part4, gradV_epilogue_final)
};

// 定义结构体 SharedStorageNoPrologue，包含一个 persistent 结构体，用于存储相关数据
struct SharedStorageNoPrologue {
  struct {
    cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
  } persistent;
    // 定义一个联合体，用于存储多个不同类型的数据结构
    union {
      // part1 - Q.K matmul
      // 定义一个结构体 part1，包含 MatmulQK::Mma::SharedStorageA 类型的变量 mm_qk_k，
      // 和 MatmulQK::Mma::SharedStorageB 类型的变量 mm_qk_q
      typename MatmulQK::Mma::SharedStorageA mm_qk_k;
      typename MatmulQK::Mma::SharedStorageB mm_qk_q;
    } part1;

    // part2 - compute gradV
    // 定义一个结构体 part2
    struct {
      // 1. efficient load of bias tile Bij, which is then applied to Pij
      // 定义一个联合体，用于存储不同类型的数据结构，其中 bias 是 MatmulQK::BiasLoader::SmemTile 类型的变量，
      // 用于高效加载偏置矩阵 Bij，并应用于 Pij
      // 2. store Pij to shared memory. it is needed:
      //    - in this step, where it is used in dVj += (Pij.T * Zij) @ dOi
      //    - in next step where it is used in dSij = Pij * (dPij - Di)
      // 定义一个联合体，用于存储不同类型的数据结构，attn_shared_storage 是 MatmulQK::AccumulatorSharedStorage 类型的变量，
      // 用于存储 Pij 到共享内存中，这在计算 dVj 时需要 (Pij.T * Zij) @ dOi，以及在下一步计算 dSij = Pij * (dPij - Di) 时使用
      union {
        typename MatmulQK::BiasLoader::SmemTile bias;
        typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
      };
      // 3. store Zij. it is needed in this step, where it is used
      // to compute Pij_dropped = Pij * Zij on the fly as fragments of Pij are
      // loaded for the computation of dVj.
      // 定义一个变量 zij，类型为 ZijSharedStorage，用于存储 Zij 数据，
      // 在计算 dVj 时用于实时计算 Pij_dropped = Pij * Zij，因为 Pij 的片段被加载用于 dVj 的计算
      ZijSharedStorage zij;

      // 定义一个联合体，用于存储不同类型的数据结构，mm_gradV 是 MatmulGradV::Mma::SharedStorage 类型的变量，
      // 用于存储 gradV 的中间计算结果
      union {
        typename MatmulGradV::Mma::SharedStorage mm_gradV;
        typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
      };
    } part2;

    // part3 - DO.V matmul
    // 定义一个结构体 part3
    struct {
      // first compute dPij = (dOi @ Vj.T) * Zij
      // and dSij = Pij * (dPij - Di)
      // 定义一个嵌套结构体，用于计算 dPij 和 dSij：
      // attn_shared_storage 来自 part2，用于计算 dSij = Pij * (dPij - Di)
      // mm_doivj 是 MatmulDOIVJ::Mma::SharedStorage 类型的变量，用于计算 dOiVj
      struct {
        typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
        typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
      };
      // then store dB = dSij to global memory
      // 定义一个变量 gradB_epilogue，类型为 MatmulDOIVJ::BiasGradEpilogue::SharedStorage，
      // 用于存储 dB = dSij 到全局内存中
      typename MatmulDOIVJ::BiasGradEpilogue::SharedStorage gradB_epilogue;
    } part3;

    // part4 - compute gradQ
    // 定义一个结构体 part4，用于计算 gradQ
    struct {
      // tmpT_shared_storage 来自 part2，用于中间计算
      typename MatmulQK::AccumulatorSharedStorage tmpT_shared_storage;
      typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
      // 定义一个联合体，用于存储不同类型的数据结构
      union {
        typename MatmulGradQ::Mma::SharedStorage mm_gradQ;
        typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;
        typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue_lastIter;
      };
    } part4;

    // part5 - compute gradK
    // 定义一个结构体 part5，用于计算 gradK
    struct {
      // tmpT_shared_storage 来自 part2，用于中间计算
      typename MatmulQK::AccumulatorSharedStorage tmpT_shared_storage;
      typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
      // 定义一个联合体，用于存储不同类型的数据结构
      union {
        typename MatmulGradK::Mma::SharedStorage mm_gradK;
        typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
      };
    } part5;

    // part6 - store RF accumulated into gmem
    // 定义一个结构体 part6，用于将 RF 累积存储到全局内存中
    struct {
      typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue_final;
      typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue_final;
    } part6;
// 输出 SharedStorageNoPrologue 结构体中字段的大小（以字节为单位）
#define FIELD_SIZEOF(f) int((sizeof(((SharedStorageNoPrologue*)0)->f)))

// 打印 SharedStorageNoPrologue 结构体的总大小
printf("Total smem: %d bytes\n", int(sizeof(SharedStorageNoPrologue)));
// 打印各个字段在结构体中的大小
printf("  persistent: %db\n", FIELD_SIZEOF(persistent));
printf("  part1: %db\n", FIELD_SIZEOF(part1));
printf("  part2: %db\n", FIELD_SIZEOF(part2));
printf("  part3: %db\n", FIELD_SIZEOF(part3));
printf("  part4: %db\n", FIELD_SIZEOF(part4));
printf("  part5: %db\n", FIELD_SIZEOF(part5));
printf("  part6: %db\n", FIELD_SIZEOF(part6));
}

// 定义用于返回结构体中特定字段引用的宏
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE auto& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

// 使用 FIELD 宏定义各个字段的访问方法
FIELD(persistent, di)
FIELD(part1, mm_qk_k)
FIELD(part1, mm_qk_q)
FIELD(part2, bias)
FIELD(part2, attn_shared_storage)
FIELD(part2, zij)
FIELD(part2, mm_gradV)
FIELD(part2, gradV_epilogue)
FIELD(part3, mm_doivj)
FIELD(part3, gradB_epilogue)
FIELD(part4, tmpT_shared_storage)
FIELD(part4, tmp_shared_storage)
FIELD(part4, mm_gradQ)
FIELD(part4, gradQ_epilogue)
FIELD(part4, gradQ_epilogue_lastIter)
FIELD(part5, mm_gradK)
FIELD(part5, gradK_epilogue)
FIELD(part6, gradK_epilogue_final)
FIELD(part6, gradV_epilogue_final)
};

// 使用条件类型选择模板，根据 kPreload 的值选择不同的共享存储结构体
using SharedStorage = typename cutlass::platform::conditional<
    kPreload,
    SharedStoragePrologue,
    SharedStorageNoPrologue>::type;

// 定义输出结构体 OutputFragments
struct OutputFragments {
  // 使用 MatmulGradV 和 MatmulGradK 中的 FragmentC 类型定义 gradV 和 gradK 字段
  typename MatmulGradV::Mma::FragmentC gradV;
  typename MatmulGradK::Mma::FragmentC gradK;

  // 清空 gradV 和 gradK 字段的方法
  CUTLASS_DEVICE void clear() {
    gradV.clear();
    gradK.clear();
  }
};

// 静态函数，用于检查参数 p 中的内存对齐情况
static bool __host__ check_supported(Params const& p) {
  // 检查各个指针是否按照 kMinimumAlignment 对齐
  CHECK_ALIGNED_PTR(p.query_ptr, kMinimumAlignment);
  CHECK_ALIGNED_PTR(p.key_ptr, kMinimumAlignment);
  CHECK_ALIGNED_PTR(p.value_ptr, kMinimumAlignment);
  CHECK_ALIGNED_PTR(p.output_ptr, kMinimumAlignment);
  CHECK_ALIGNED_PTR(p.grad_output_ptr, kMinimumAlignment);
  CHECK_ALIGNED_PTR(p.bias_ptr, kMinimumAlignment);
  
  // 检查条件并抛出相应的错误信息
  TORCH_CHECK(
      p.num_heads <= 1 || p.lse_strideH % 8 == 0,
      "LSE is not correctly aligned (strideH)");
  TORCH_CHECK(
      p.num_batches <= 1 || p.lse_strideB % 8 == 0,
      "LSE is not correctly aligned (strideB)");
  TORCH_CHECK(
      p.num_heads <= 1 || p.q_strideH % kMinimumAlignment == 0,
      "query is not correctly aligned (strideH)");
  TORCH_CHECK(
      p.num_heads <= 1 || p.k_strideH % kMinimumAlignment == 0,
      "key is not correctly aligned (strideH)");
  TORCH_CHECK(
      p.num_heads <= 1 || p.v_strideH % kMinimumAlignment == 0,
      "value is not correctly aligned (strideH)");
  TORCH_CHECK(
      p.num_batches <= 1 || p.q_strideB % kMinimumAlignment == 0,
      "query is not correctly aligned (strideB)");
  TORCH_CHECK(
      p.num_batches <= 1 || p.k_strideB % kMinimumAlignment == 0,
      "key is not correctly aligned (strideB)");
    # 检查是否满足值的对齐要求（strideB），如果 num_batches <= 1 则忽略对齐检查
    TORCH_CHECK(
        p.num_batches <= 1 || p.v_strideB % kMinimumAlignment == 0,
        "value is not correctly aligned (strideB)");

    # 检查查询的对齐要求（strideM）
    TORCH_CHECK(
        p.q_strideM % kMinimumAlignment == 0,
        "query is not correctly aligned (strideM)");

    # 检查键的对齐要求（strideM）
    TORCH_CHECK(
        p.k_strideM % kMinimumAlignment == 0,
        "key is not correctly aligned (strideM)");

    # 检查值的对齐要求（strideM）
    TORCH_CHECK(
        p.v_strideM % kMinimumAlignment == 0,
        "value is not correctly aligned (strideM)");

    # 如果存在偏置指针，检查偏置在批次维度上的对齐要求（strideB）
    if (p.bias_ptr) {
        TORCH_CHECK(
            p.num_batches <= 1 || p.bias_strideB % kMinimumAlignment == 0,
            "attn_bias is not correctly aligned (strideB). ",
            "attn_bias.stride(0) = ", p.bias_strideB, ", and should be a "
            "multiple of ", kMinimumAlignment, ".");
        
        # 检查偏置在头数维度上的对齐要求（strideH）
        TORCH_CHECK(
            p.num_heads <= 1 || p.bias_strideH % kMinimumAlignment == 0,
            "attn_bias is not correctly aligned (strideH) ."
            "attn_bias.stride(1) = ", p.bias_strideH, ", and should be a "
            "multiple of ", kMinimumAlignment, ".");
        
        # 检查偏置在查询维度上的对齐要求（strideM）
        TORCH_CHECK(
            p.num_queries <= 1 || p.bias_strideM % kMinimumAlignment == 0,
            "attn_bias is not correctly aligned (strideM). "
            "attn_bias.stride(2) = ", p.bias_strideM, ", and should be a ",
            "multiple of ", kMinimumAlignment, ".");
    }

    # 如果存在梯度偏置指针，检查梯度偏置在批次维度上的对齐要求（strideB）
    if (p.grad_bias_ptr) {
        TORCH_CHECK(
            p.num_batches <= 1 || p.gB_strideB % kMinimumAlignment == 0,
            "attn_bias.grad is not correctly aligned (strideB)");
        
        # 检查梯度偏置在头数维度上的对齐要求（strideH）
        TORCH_CHECK(
            p.num_heads <= 1 || p.gB_strideH % kMinimumAlignment == 0,
            "attn_bias.grad is not correctly aligned (strideH)");
        
        # 检查梯度偏置在查询维度上的对齐要求（strideM）
        TORCH_CHECK(
            p.gB_strideM % kMinimumAlignment == 0,
            "attn_bias.grad is not correctly aligned (strideM)");
    }

    # 检查是否同时使用 CuSeqlen 和偏置，如果是则报错
    TORCH_CHECK(
        !(p.cu_seqlens_q_ptr && p.bias_ptr),
        "CuSeqlen + bias not implemented yet");

    # 检查自定义掩码类型的合法性
    TORCH_CHECK(
        p.custom_mask_type < NumCustomMaskTypes,
        "Invalid value for `custom_mask_type`");

    # 检查 dropout 概率的合法性
    TORCH_CHECK(
        p.dropout_prob <= 1.0f && p.dropout_prob >= 0.0f,
        "Invalid value for `dropout_prob`");

    # 检查是否设置了正确的 dropout 支持标志
    TORCH_CHECK(
        kApplyDropout || p.dropout_prob == 0.0f,
        "Set `kApplyDropout`=True to support `dropout_prob > 0`");

    # 检查头维度的有效性
    TORCH_CHECK(p.head_dim > 0, "Invalid value for `head_dim`");

    # 检查头维度值的有效性
    TORCH_CHECK(p.head_dim_value > 0, "Invalid value for `head_dim_value`");

    # 检查查询数的有效性
    TORCH_CHECK(p.num_queries > 0, "Invalid value for `num_queries`");

    # 检查键数的有效性
    TORCH_CHECK(p.num_keys > 0, "Invalid value for `num_keys`");

    # 检查头数的有效性
    TORCH_CHECK(p.num_heads > 0, "Invalid value for `num_heads`");

    # 检查批次数的有效性
    TORCH_CHECK(p.num_batches > 0, "Invalid value for `num_batches`");

    # 检查头维度是否不超过最大限制
    TORCH_CHECK(p.head_dim <= kMaxK, "kMaxK: Expected `head_dim < kMaxK`");

    # 检查头维度值是否不超过最大限制
    TORCH_CHECK(
        p.head_dim_value <= kMaxK, "kMaxK: Expected `head_dim_value < kMaxK`");
    // 如果 kKeysQueriesAlignedToBlockSize 为真，则执行以下检查
    if (kKeysQueriesAlignedToBlockSize) {
      // 检查是否不支持 cu_seqlen
      TORCH_CHECK(
          p.cu_seqlens_k_ptr == nullptr,
          "This kernel does not support cu_seqlen");
      TORCH_CHECK(
          p.cu_seqlens_q_ptr == nullptr,
          "This kernel does not support cu_seqlen");
      // 检查 num_queries 是否能被 kBlockSizeI 整除
      TORCH_CHECK(
          p.num_queries % kBlockSizeI == 0,
          "kKeysQueriesAlignedToBlockSize condition not respected");
      // 检查 num_keys 是否能被 kBlockSizeJ 整除
      TORCH_CHECK(
          p.num_keys % kBlockSizeJ == 0,
          "kKeysQueriesAlignedToBlockSize condition not respected");
    }
    // 检查是否启用 SplitKeys，如果不启用则报错
    TORCH_CHECK(
        kEnableSplitKeys || p.num_splits_key == 1, "SplitKeys is disabled");
    // 检查 num_splits_key 是否大于 0
    TORCH_CHECK(
        p.num_splits_key > 0, "Invalid `num_splits_key` (expected >0)");
    // 检查 num_splits_key 是否小于或等于 ceil_div(p.num_keys, kBlockSizeJ)
    TORCH_CHECK(
        p.num_splits_key <= cutlass::ceil_div(p.num_keys, kBlockSizeJ),
        "Invalid `num_splits_key` (",
        p.num_splits_key,
        ") - too large for `num_keys` = ",
        p.num_keys);
    // 如果 window_size 不为 0，则进行以下检查
    if (p.window_size != 0) {
      // 如果 custom_mask_type 不是 NoCustomMask，则报错
      TORCH_CHECK(
          p.custom_mask_type != NoCustomMask,
          "LocalAttention only supported in causal mode");
    }
    // 返回 true，表示检查通过
    return true;
  }

  // 定义静态的 attention_kernel 函数，接收参数 p
  static CUTLASS_DEVICE void attention_kernel(Params p) {
    // 外部共享内存声明和初始化
    extern __shared__ char smem_buffer[];
    // 将共享内存转换为 SharedStorage 类型
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);

    // 线程 ID 和 warp ID 计算
    uint16_t thread_id = threadIdx.x;
    uint8_t warp_id = warp_uniform(thread_id / 32);
    uint8_t lane_id = thread_id % 32;

    // 计算 key 的起始位置
    int32_t key_start = p.split_key_device() * kBlockSizeJ;
    // 如果 key_start 大于等于 num_keys，则直接返回
    if (key_start >= p.num_keys) {
      return;
    }
    // 如果 kPrologueQK 为真，则执行以下操作
    if (kPrologueQK) {
      // 获取 query 的起始位置
      int32_t query_start = getQueryStart(p, key_start);
      // 执行 prologueQkNextIteration 函数
      prologueQkNextIteration<true>(
          shared_storage, p, query_start, key_start, warp_id, lane_id);
    }

    // 如果 kKernelComputesDelta 为真，则计算 delta 值
    if (kKernelComputesDelta) {
      // 计算最佳元素数量
      constexpr int kOptimalElements =
          128 / cutlass::sizeof_bits<scalar_t>::value;
      // 根据 head_dim_value 是否能整除 kOptimalElements，选择不同模板参数
      if (p.head_dim_value % kOptimalElements == 0) {
        // 循环处理每个 query 的 delta 计算
        for (int query_start = 0; query_start < p.num_queries;
             query_start += kBlockSizeI) {
          computeDelta<kOptimalElements>(p, query_start, warp_id, lane_id);
        }
      } else {
        // head_dim_value 不能整除 kOptimalElements 的情况下，使用模板参数为 1
        for (int query_start = 0; query_start < p.num_queries;
             query_start += kBlockSizeI) {
          computeDelta<1>(p, query_start, warp_id, lane_id);
        }
      }
      // 同步线程，等待所有线程完成计算
      __syncthreads();
    }

    // 定义输出片段对象
    OutputFragments output_frags;

    // 定义随机数状态结构体
    curandStatePhilox4_32_10_t rng_state_init;
    if (kApplyDropout) {
      // 如果需要应用 dropout
      // 查看注释 [Seed and Offset Device]
      // 从 RNG 引擎输入中解包种子
      auto seeds = at::cuda::philox::unpack(p.rng_engine_inputs);
      // 注意：注意力矩阵 P 的每个元素（形状为 batch_sz, n_heads, n_queries, n_keys）
      // 都与 RNG 序列中的单个偏移量相关联。我们使用从该块的 batch_id 和 head_id
      // 对应的 (n_queries, n_keys) 矩阵开始的偏移量初始化 RNG 状态。
      // 初始化 RNG 状态非常昂贵，因此我们在每个 kernel 中运行一次，而不是每次迭代都运行。
      // 每次迭代都会复制初始化的 RNG 状态，并根据需要进行偏移。
      curand_init(
          std::get<0>(seeds),
          0,
          std::get<1>(seeds) + p.dropout_batch_head_rng_offset,
          &rng_state_init);
    }

    CUTLASS_PRAGMA_UNROLL
    // 使用 CUTLASS_PRAGMA_UNROLL 指令进行循环展开优化
    for (; key_start < p.num_keys;
         key_start += p.num_splits_key_device() * kBlockSizeJ) {
      output_frags.clear();

      int32_t next_key = key_start;
      // 获取查询起始位置
      int32_t query_start = getQueryStart(p, key_start);
      while (next_key == key_start && query_start < p.num_queries) {
        // 此处
        // vvvvvvvvvvvvvv
        // 使用 warp_uniform 函数生成 warp_id
        warp_id = warp_uniform(warp_id);
        // ^^^^^^^^^^^^^^
        // 使所有内容使用较少的寄存器文件（RF），并且速度提高了 10%。为什么？
        // 我不知道。我的理论是，它迫使 `nvcc` 重新计算索引、偏移量等...
        // 而不是从上一次迭代中保留它们，这可以防止大量的寄存器溢出。

        // 处理块 IJ 的数据
        processBlockIJ<kKeysQueriesAlignedToBlockSize>(
            shared_storage,
            output_frags,
            p,
            query_start,
            key_start,
            rng_state_init,
            warp_id,
            lane_id);

        int32_t next_query;
        // 增加迭代次数
        incrIteration(p, query_start, key_start, next_query, next_key);
        query_start = next_query;
      }
      if (kOutputInRF) {
        // 将输出碎片写入全局内存
        writeFragsToGmem<kKeysQueriesAlignedToBlockSize>(
            shared_storage, output_frags, p, key_start, warp_id, lane_id);
      } else if (getQueryStart(p, key_start) >= p.num_queries) {
        // 如果查询起始位置大于或等于总查询数，则填充 GradKV
        zfillGradKV<kKeysQueriesAlignedToBlockSize>(
            p, key_start, warp_id, lane_id);
      }
      // 等待所有线程块执行完毕
      __syncthreads();
    }
  }

  // 如果跳过边界检查，则静态方法 zfillGradKV
  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void zfillGradKV(
      Params const& p,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    constexpr int kThreadsPerKey = 8;
    constexpr int kParallelKeys = kNumThreads / kThreadsPerKey;
    // 确保 kParallelKeys 是 kBlockSizeJ 的倍数
    static_assert(kBlockSizeJ % kParallelKeys == 0, "");
    // 此函数并非真正优化，但应该很少使用
    // 仅在某些键是“无用”的情况下使用，因为它们不会参与由因果掩码引起的任何查询

    // 线程 ID
    int thread_id = 32 * warp_id + lane_id;
    // 线程在 kThreadsPerKey 中的偏移量
    int k_shift = lane_id % kThreadsPerKey;

    CUTLASS_PRAGMA_UNROLL
    // 循环遍历 kBlockSizeJ 范围内的索引 j，以 kParallelKeys 为步长增加 j
    for (int j = 0; j < kBlockSizeJ; j += kParallelKeys) {
      // 计算当前线程的 key，根据线程 ID 和 kThreadsPerKey 计算
      int key = key_start + j + (thread_id / kThreadsPerKey);
      // 如果未开启跳过边界检查且 key 超出 num_keys 的范围，则跳过当前循环
      if (!skipBoundsChecks && key >= p.num_keys) {
        continue;
      }
      // 获取当前 key 对应的梯度值指针和梯度键指针
      auto gv_ptr = p.grad_value_ptr + key * p.gV_strideM();
      auto gk_ptr = p.grad_key_ptr + key * p.gK_strideM();

      // 初始化 gv_ptr 指向的内存块中的值为 0
      for (int k = k_shift; k < p.head_dim_value; k += kThreadsPerKey) {
        gv_ptr[k] = scalar_t(0);
      }
      // 初始化 gk_ptr 指向的内存块中的值为 0
      for (int k = k_shift; k < p.head_dim; k += kThreadsPerKey) {
        gk_ptr[k] = scalar_t(0);
      }
    }
  }

  // 定义模板函数，根据 skipBoundsChecks 参数处理输入输出数据块
  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void processBlockIJ(
      SharedStorage& shared_storage,
      OutputFragments& output_frags,
      Params& p,
      int32_t query_start,
      int32_t key_start,
      const curandStatePhilox4_32_10_t& curand_state_init,
      uint8_t warp_id,
      uint8_t lane_id) {
    // 初始化 dropout_keep_mask_doivj，用于控制是否应用 dropout
    cutlass::Array<cutlass::uint1b_t, MatmulDOIVJ::Mma::FragmentC::kElements>
        dropout_keep_mask_doivj;
    dropout_keep_mask_doivj.fill(cutlass::uint1b_t{1});
    // 计算 dropout 缩放因子，若应用 dropout 则计算相应比例
    const float dropout_scale =
        kApplyDropout ? 1.0 / (1.0 - p.dropout_prob) : 1.0f;

    // 初始化无偏移的矩阵坐标
    cutlass::MatrixCoord no_offset{0, 0};
    // 初始化缩放因子
    accum_t scale = p.scale;
    // 计算线程 ID
    int16_t thread_id = 32 * warp_id + lane_id;

    // 定义 lambda 函数，重新计算线程 ID 和 warp ID，以降低寄存器压力
    auto rematerializeThreadIds = [&]() {
      // 通过 warp_uniform 函数重新计算 warp_id，以及通过取模计算 lane_id
      warp_id = warp_uniform(thread_id / 32);
      lane_id = thread_id % 32;
      thread_id = 32 * warp_id + lane_id;
    };

    // 判断是否为第一个查询
    bool isFirstQuery = (query_start == getQueryStart(p, key_start));
    int32_t next_query, next_key;
    // 根据当前查询和键的起始位置计算下一个查询和键的位置
    incrIteration(p, query_start, key_start, next_query, next_key);
    // 判断是否为最后一个查询
    bool isLastQuery = next_key != key_start;

    // 初始化 di_rf 变量为 0
    accum_t di_rf = accum_t(0);
    // 如果线程 ID 小于 kBlockSizeI，则处理对应的 delta 值
    if (thread_id < kBlockSizeI) {
      if (query_start + thread_id < p.num_queries) {
        di_rf = p.delta_ptr[query_start + thread_id];
      }
      // 将计算得到的 di_rf 存储到 shared_storage 中
      shared_storage.di()[thread_id] = di_rf;
    }

    // 计算当前数据块中的查询数量，可以根据 skipBoundsChecks 参数决定是否进行边界检查
    int32_t num_queries_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kN
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kN, p.num_queries - query_start));
    // 计算当前数据块中的键数量，同样可以根据 skipBoundsChecks 参数决定是否进行边界检查
    int32_t num_keys_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kM
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start));

    // 定义处理 GradV 梯度的 lambda 函数
    auto prologueGradV = [&](int col) {
      // 初始化迭代器 iterator_dO，用于处理 p.grad_output_ptr 中的数据
      typename MatmulGradV::Mma::IteratorB iterator_dO(
          {int32_t(p.gO_strideM)},
          const_cast<scalar_t*>(p.grad_output_ptr + query_start * p.gO_strideM + col),
          {num_queries_in_block, p.head_dim_value - col},
          thread_id,
          no_offset);
      // 调用 MatmulGradV::Mma::prologue 函数进行处理
      MatmulGradV::Mma::prologue(
          shared_storage.mm_gradV(),
          iterator_dO,
          thread_id,
          num_queries_in_block);
    };
    // 定义lambda函数prologueGradQ，用于初始化MatmulGradQ的迭代器并调用前导操作
    auto prologueGradQ = [&](int col) {
      // 创建MatmulGradQ的迭代器，访问p.key_ptr中指定列的数据
      typename MatmulGradQ::Mma::IteratorB iterator_K(
          {int32_t(p.k_strideM)},
          const_cast<scalar_t*>(p.key_ptr + key_start * p.k_strideM + col),
          {num_keys_in_block, p.head_dim - col},
          thread_id,
          no_offset);
      // 执行MatmulGradQ的前导操作，初始化共享存储的mm_gradQ
      MatmulGradQ::Mma::prologue(
          shared_storage.mm_gradQ(), iterator_K, thread_id, num_keys_in_block);
    };

    // 定义lambda函数prologueGradK，用于初始化MatmulGradK的迭代器并调用前导操作
    auto prologueGradK = [&](int col) {
      // 创建MatmulGradK的迭代器，访问p.query_ptr中指定列的数据
      typename MatmulGradK::Mma::IteratorB iterator_Q(
          {int32_t(p.q_strideM)},
          const_cast<scalar_t*>(p.query_ptr + query_start * p.q_strideM + col),
          {num_queries_in_block, p.head_dim - col},
          thread_id,
          no_offset);
      // 执行MatmulGradK的前导操作，初始化共享存储的mm_gradK
      MatmulGradK::Mma::prologue(
          shared_storage.mm_gradK(),
          iterator_Q,
          thread_id,
          num_queries_in_block);
    };

    // 定义lambda函数prologueDOV，用于初始化MatmulDOIVJ的迭代器并调用前导操作
    auto prologueDOV = [&]() {
      // 创建MatmulDOIVJ的两个迭代器，分别访问p.grad_output_ptr和p.value_ptr的数据
      typename MatmulDOIVJ::Mma::IteratorA iterator_A(
          {int32_t(p.gO_strideM)},
          const_cast<scalar_t*>(p.grad_output_ptr + query_start * p.gO_strideM),
          {num_queries_in_block, p.head_dim_value},
          thread_id,
          no_offset);
      typename MatmulDOIVJ::Mma::IteratorB iterator_B(
          {int32_t(p.v_strideM)},
          const_cast<scalar_t*>(p.value_ptr + key_start * p.v_strideM),
          {p.head_dim_value, num_keys_in_block},
          thread_id,
          no_offset);
      // 执行MatmulDOIVJ的前导操作，初始化共享存储的mm_doivj
      MatmulDOIVJ::Mma::prologue(
          shared_storage.mm_doivj(),
          iterator_A,
          iterator_B,
          thread_id,
          p.head_dim_value);
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulQK
    /////////////////////////////////////////////////////////////////////////////////////////////////
#if 0
      auto accum_ref_attnT = shared_storage.attn_shared_storage().accum_ref();
      PRINT_TENSOR4x4_T0_L0("attn_T", accum_ref_attnT);
#endif

      // 如果使用了 dropout，计算 Zij 并将其写入共享内存。
      // Zij 中的每个元素：
      // - dropout_p 的概率为 0
      // - 1 / (1 - dropout_p) 的概率为 1 - dropout_p
      if (kApplyDropout) {
        auto zij = shared_storage.zij().accum_ref();
        // 每个线程生成 Zij 中一行连续的元素序列。
        // 它们必须来自同一行，因为从连续的随机数序列中采样比跳跃采样效率更高，
        // 并且 Z 的每个元素的线性偏移映射到随机数序列的偏移。
        // 对于 Z，一行的末尾和下一行的开头有相邻的偏移量，
        // 但对于 Zij（全局矩阵的块），这种情况未必如此。
        // 我们必须用值填充整个 `zij` 共享内存（即使超出 K 维度的边界），
        // 否则在 GEMM 运算期间可能会得到 NaN 值。
        const int kQueriesPerBlock = kBlockSizeI;
        const int threads_per_row = cutlass::fast_min(
            int32_t(kNumThreads / kQueriesPerBlock), num_keys_in_block);
        const int elts_per_thread = cutlass::round_nearest(
            cutlass::ceil_div(num_keys_in_block, threads_per_row), 4);

        const int thread_i = thread_id / threads_per_row;
        const int thread_start_j =
            (thread_id % threads_per_row) * elts_per_thread;

        if (thread_i < kQueriesPerBlock && thread_start_j < num_keys_in_block) {
          curandStatePhilox4_32_10_t curand_state = curand_state_init;
          skipahead(
              (query_start + thread_i) * p.num_keys +
                  (key_start + thread_start_j),
              &curand_state);

          // 一次生成 Zij 的 4 个元素
          for (int zij_start_col_idx = thread_start_j; zij_start_col_idx <
               cutlass::fast_min<int32_t>(thread_start_j + elts_per_thread,
                                          num_keys_in_block);
               zij_start_col_idx += 4) {
            const float4 rand_uniform_quad = curand_uniform4(&curand_state);

            CUTLASS_PRAGMA_UNROLL
            for (int quad_idx = 0; quad_idx < 4; ++quad_idx) {
              // 我们将 Zij 转置写入，因为在计算 dV 时注意力也是转置的。
              zij.at({zij_start_col_idx + quad_idx /*k*/, thread_i /*q*/}) =
                  (&rand_uniform_quad.x)[quad_idx] > p.dropout_prob
                  ? scalar_t(dropout_scale)
                  : scalar_t(0);
            }
          }
        }
        // 等待所有线程完成 Zij 的填充
        __syncthreads();
#if 0
        PRINT_TENSOR4x4_T0_L0("zij", zij);
        PRINT_TENSOR4x4_T0_L0_START("zij", zij, kBlockSizeJ - 4, kBlockSizeI - 4);
#endif
      }
#endif

        // 保存掩码以供后续的 DOIVJ 矩阵乘法

        // 计算当前线程块内的 warp 索引，用于确定输出矩阵坐标
        int warp_idx_mn_0 = warp_id %
            (MatmulDOIVJ::Mma::Base::WarpCount::kM *
             MatmulDOIVJ::Mma::Base::WarpCount::kN);
        
        // 计算输出矩阵坐标
        auto output_tile_coords_doivj = cutlass::MatrixCoord{
            warp_idx_mn_0 % MatmulDOIVJ::Mma::Base::WarpCount::kM,
            warp_idx_mn_0 / MatmulDOIVJ::Mma::Base::WarpCount::kM};
        
        // 获取当前线程的 lane 偏移量
        auto lane_offset = MatmulDOIVJ::AccumLambdaIterator::get_lane_offset(
            lane_id, warp_id, output_tile_coords_doivj);
        
        // 迭代处理每一行
        MatmulDOIVJ::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {},  // 空的行迭代器
            [&](int accum_m /*q*/, int accum_n /*k*/, int idx) {
              // 检查 zij 矩阵中的值是否为 0
              if (zij.at({accum_n, accum_m}) == scalar_t(0)) {
                // 如果 zij 的值为 0，则将 dropout_keep_mask_doivj 对应位置置为 0
                dropout_keep_mask_doivj[idx] = cutlass::uint1b_t{0};
              }
            },
            [&](int accum_m) {});  // 空的行迭代器结束处理
      
      }
      // 等待所有线程块的执行完成
      __syncthreads();
    }
    // 重新生成线程 ID

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradV 矩阵乘法
    //
    // grad_v[j_start:j_end] += attn_T @ do_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    
    // 检查是否只需单次迭代的 GradV 矩阵乘法
    constexpr bool kSingleIterationGradV =
        kMaxK <= MatmulGradV::ThreadblockShape::kN;
    // 对于每一列，从0开始迭代，直到达到单次迭代梯度或头维度值的上限
    // col 每次增加 MatmulGradV::ThreadblockShape::kN
    for (int col = 0; col < (kSingleIterationGradV ? 1 : p.head_dim_value);
         col += MatmulGradV::ThreadblockShape::kN) {
      
      // 使用 MatmulGradV 定义的类型别名 Mma 和 AccumTileGmem
      using Mma = typename MatmulGradV::Mma;
      using AccumTileGmem = typename MatmulGradQ::AccumTileGmem;

      // 定义 gemm 的问题大小
      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block, p.head_dim_value - col, num_queries_in_block);
      
      // 创建用于后续计算的输出迭代器
      auto createEpilogueIter = [&]() {
        return typename MatmulGradV::OutputTileIterator(
            typename MatmulGradV::OutputTileIterator::Params{p.gV_strideM()},
            p.grad_value_ptr + key_start * p.gV_strideM() + col,
            {num_keys_in_block, p.head_dim_value - col},
            thread_id);
      };

      // 初始化矩阵乘法运算中 B 操作数的迭代器
      typename Mma::IteratorB iterator_B(
          {int32_t(p.gO_strideM)},
          const_cast<scalar_t*>(p.grad_output_ptr + query_start * p.gO_strideM + col),
          {num_queries_in_block, p.head_dim_value - col},
          thread_id,
          no_offset);

      // 初始化 Mma 对象进行矩阵乘法操作
      Mma mma(
          // 操作数 A: Pij.T
          shared_storage.attn_shared_storage().accum_ref(),
          // 操作数 A_scale Zij.T:
          // 如果使用 dropout，则操作数 A 是 Pij_dropped.T = Pij.T * Zij.T
          // 这会随着加载 Pij.T 片段时即时计算
          shared_storage.zij().accum_ref(),
          // 操作数 B: dOi - 在之前计算 dVj 时加载到共享内存中的数据
          shared_storage.mm_gradV().operand_B_ref(),
          thread_id,
          warp_id,
          lane_id);

      // 计算存储索引，用于存储累积数据到全局内存中
      int storage_id = col / MatmulGradV::ThreadblockShape::kN;
      AccumTileGmem gmem_tile{
          p.workspace_gv + storage_id * AccumTileGmem::kElementsStored};

      // 如果不在寄存器文件中输出结果
      if (!kOutputInRF) {
        if (isFirstQuery || !kNeedsAccumGradV) {
          // 清空输出片段的数据
          output_frags.gradV.clear();
        } else {
          // 从全局内存中加载输出片段的数据
          gmem_tile.load(output_frags.gradV, thread_id);
        }
      }

      // 设置 Mma 对象的前处理完成标志
      mma.set_prologue_done(kPrologueGV);

      // 计算 gemm_k_iterations，即 K 维度上的迭代次数
      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // 同步所有线程块，确保之前的计算完成
      __syncthreads();

      // 执行线程块范围内的矩阵乘加运算
      mma(gemm_k_iterations,
          output_frags.gradV,
          iterator_B,
          output_frags.gradV);

      // 再次同步所有线程块，确保所有计算完成
      __syncthreads();

      // 如果启用了前处理 GV，并且不是单次迭代 GradV，且未达到头维度的上限
      if (kPrologueGV && !kSingleIterationGradV &&
          col + MatmulGradV::ThreadblockShape::kN < p.head_dim_value) {
        // 执行 GradV 的前处理
        prologueGradV(col + MatmulGradV::ThreadblockShape::kN);
      }

      // 如果不在寄存器文件中输出结果
      if (!kOutputInRF) {
        if (kNeedsAccumGradV && !isLastQuery) {
          // 将输出片段的数据存储回全局内存
          gmem_tile.store(output_frags.gradV, thread_id);
        } else {
          // 在全局内存中累积 GradV 的后处理
          accumulateInGmem<MatmulGradV>(
              shared_storage.gradV_epilogue(),
              output_frags.gradV,
              createEpilogueIter(),
              isFirstQuery || kNeedsAccumGradV,
              warp_id,
              lane_id);
        }
      }
    }
    // 同步所有线程块，确保整个 CUDA 核函数执行完毕
    __syncthreads();
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulDOIVJ
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      // 使用MatmulDOIVJ定义的Mma类型别名Mma
      using Mma = typename MatmulDOIVJ::Mma;
      
      // 创建类型为IteratorA的迭代器，用于访问梯度输出数据
      typename Mma::IteratorA iterator_A(
          {int32_t(p.gO_strideM)},  // 使用给定的步长创建封装梯度输出的迭代器
          const_cast<scalar_t*>(p.grad_output_ptr + query_start * p.gO_strideM),  // 设置起始位置和数据指针
          {num_queries_in_block, p.head_dim_value},  // 指定迭代器的维度
          thread_id,  // 当前线程ID
          no_offset);  // 不使用偏移量

      // 创建类型为IteratorB的迭代器，用于访问值数据的转置
      typename Mma::IteratorB iterator_B(
          {int32_t(p.v_strideM)},  // 使用给定的步长创建封装值的迭代器
          const_cast<scalar_t*>(p.value_ptr + key_start * p.v_strideM),  // 设置起始位置和数据指针
          {p.head_dim_value, num_keys_in_block},  // 指定迭代器的维度
          thread_id,  // 当前线程ID
          no_offset);  // 不使用偏移量

      // 创建Matmul对象mma，用于执行矩阵乘法累加操作
      Mma mma(shared_storage.mm_doivj(), thread_id, warp_id, lane_id);
      mma.set_prologue_done(kPrologueDOV);  // 设置前导处理标志为kPrologueDOV
      mma.set_zero_outside_bounds(!skipBoundsChecks);  // 设置是否在边界之外清零的标志

      // 创建用于累加结果的FragmentC类型的变量accum
      typename Mma::FragmentC accum;
      accum.clear();  // 清空累加器

      // 计算GEMM的k迭代次数
      auto gemm_k_iterations =
          (p.head_dim_value + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // 执行线程块范围内的矩阵乘加操作
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      __syncthreads();  // 同步线程块内的所有线程

      // 如果需要前导处理GQ，则执行prologueGradQ函数
      if (kPrologueGQ) {
        prologueGradQ(0);
      }
      // 如果需要前导处理GK，则执行prologueGradK函数
      if (kPrologueGK) {
        prologueGradK(0);
      }

      // 计算warp_idx_mn_0，用于确定输出矩阵块的坐标
      int warp_idx_mn_0 =
          warp_id % (Mma::Base::WarpCount::kM * Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / Mma::Base::WarpCount::kM};

      // TODO: This must be terribly inefficient. There must be a better way
      // tmp [RF] <- (accum [RF] - Di [smem] ) * attn_T.T [smem]
      // attn_shared_storage  [smem] <- tmp.T
      // tmp_shared_storage [smem] <- tmp

      {
        // 使用MatmulDOIVJ定义的AccumLambdaIterator类型别名LambdaIterator
        using LambdaIterator = typename MatmulDOIVJ::AccumLambdaIterator;
        auto lane_offset = LambdaIterator::get_lane_offset(
            lane_id, warp_id, output_tile_coords);  // 获取lane的偏移量

        // 如果应用了dropout，计算dPij = dPij_dropped * Zij
        if (kApplyDropout) {
          LambdaIterator::iterateRows(
              lane_offset,
              [&](int accum_m) {},  // lambda函数，用于处理每行累加器
              [&](int accum_m, int accum_n, int idx) {  // lambda函数，用于处理每个元素
                if (dropout_keep_mask_doivj[idx].get()) {  // 如果dropout保留掩码为真
                  accum[idx] *= dropout_scale;  // 应用dropout缩放因子
                } else {
                  accum[idx] = 0;  // 否则清零
                }
              },
              [&](int accum_m) {});  // lambda函数，用于处理每行结束后的清理工作
        }

        // 获取共享内存中的attn_T，并赋值给attn_T
        auto attn_T = shared_storage.attn_shared_storage().accum_ref();
        accum_t current_di;
        // 定义变量 current_di 用于存储 shared_storage.di() 中的当前值

        // 使用 LambdaIterator 遍历行
        LambdaIterator::iterateRows(
            lane_offset,  // 起始行偏移量
            [&](int accum_m) { current_di = shared_storage.di()[accum_m]; },  // 获取 current_di 的值
            [&](int accum_m, int accum_n, int idx) {
              // 如果 skipBoundsChecks 为真或者 (accum_m, accum_n) 在有效范围内
              if (skipBoundsChecks ||
                  (accum_m < num_queries_in_block &&
                   accum_n < num_keys_in_block)) {
                // 获取 attn_T 中的注意力值 attn
                accum_t attn = attn_T.at({accum_n, accum_m});
                // 计算累积值 accum[idx]，更新为 (accum[idx] - current_di) * attn
                accum[idx] = (accum[idx] - current_di) * attn;
              } else {
                // 超出有效范围时，将 accum[idx] 置为0
                accum[idx] = 0;
              }
            },
            [&](int accum_m) {

            });

        // 将偏置梯度 dBij 存储到全局内存中
        // dBij = dSij = Pij * (dPij - Di)
        if (p.grad_bias_ptr != nullptr) {
          // 初始化输出迭代器 output_iter
          typename MatmulDOIVJ::BiasGradEpilogue::OutputTileIterator
              output_iter(
                  typename MatmulDOIVJ::BiasGradEpilogue::OutputTileIterator::
                      Params{p.gB_strideM},
                  // grad_bias_ptr 是偏移量，指向一个形状为 (queries, keys) 的矩阵
                  // 这里的指针算术操作产生了指向该矩阵中当前块起始位置的指针
                  p.grad_bias_ptr + query_start * p.gB_strideM + key_start,
                  {num_queries_in_block, num_keys_in_block},
                  thread_id);

          // 无操作的后处理运算符 - 仅进行类型转换并将 accum 的内容存储到全局内存中
          typename MatmulDOIVJ::BiasGradEpilogue::OutputOp output_op({1, 1});
          // 初始化偏置梯度 epilogue 对象
          typename MatmulDOIVJ::BiasGradEpilogue epilogue(
              shared_storage.gradB_epilogue(), thread_id, warp_id, lane_id);
          // 将累积值 accum 通过 epilogue 存储到 output_iter
          epilogue(output_op, output_iter, accum, output_iter);
        }

        // 将累积值 accum 缩放并更新
        accum = accum * scale;
#endif

__syncthreads();
// 如果矩阵未转置，则执行以下代码块
if (!MatmulGradK::DefaultMmaFromSmem::kIsTransposedA) {
  // 获取共享存储中的临时变量引用
  auto tmpT = shared_storage.tmpT_shared_storage().accum_ref();
  // 使用LambdaIterator迭代行
  LambdaIterator::iterateRows(
      lane_offset,
      [&](int accum_m) {},
      // 对每个迭代行进行操作，将accum数组的值写入tmpT中
      [&](int accum_m, int accum_n, int idx) {
        tmpT.at({accum_n, accum_m}) = scalar_t(accum[idx]);
      },
      [&](int accum_m) {});
}

// 将累积结果写入共享内存
MatmulDOIVJ::B2bGemm::accumToSmem(
    shared_storage.tmp_shared_storage(),
    accum,
    lane_id,
    output_tile_coords);
__syncthreads();
}

// 强制`nvcc`重新计算依赖于下面变量的值，以减少寄存器文件使用并防止一些溢出
p.head_dim = warp_uniform(p.head_dim);
p.k_strideM = warp_uniform(p.k_strideM);
rematerializeThreadIds();

///////////////////////////////////////////////////////////////////////////////////////////////
// GradQ matmul
//
// grad_q[i_start:i_end] += tmp @ k_j
///////////////////////////////////////////////////////////////////////////////////////////////

// 如果知道在编译时迭代的次数，则跳过循环及其相关分支
constexpr bool kSingleIterationGradQ =
    kMaxK <= MatmulGradQ::ThreadblockShape::kN;
}

///////////////////////////////////////////////////////////////////////////////////////////////
// GradK matmul
//
// grad_k[i_start:i_end] += tmp.transpose(-2, -1) @ q_i
///////////////////////////////////////////////////////////////////////////////////////////////

rematerializeThreadIds();

// 如果知道在编译时迭代的次数，则跳过循环及其相关分支
constexpr bool kSingleIterationGradK =
    kMaxK <= MatmulGradK::ThreadblockShape::kN;
}

}

static CUTLASS_HOST_DEVICE int32_t getQueryStartShift(Params const& p) {
if (p.custom_mask_type == NoCustomMask && p.num_splits_key_device() > 1) {
  // 计算查询开始的偏移量，根据分片设备数和块大小I计算
  return (p.split_key_device() * kBlockSizeI) % getQueryEnd(p);
}
// 默认返回0
return 0;
}

// 迭代顺序逻辑
static CUTLASS_HOST_DEVICE int32_t
getQueryStart(Params const& p, int32_t key_start) {
// 返回最小查询键的偏移加上查询开始的偏移量
return getSmallestQueryForKey(p, key_start) + getQueryStartShift(p);
};
static CUTLASS_HOST_DEVICE int32_t getQueryEnd(Params const& p) {
// 返回对齐到块大小I的查询结束位置
return align_up(p.num_queries, kBlockSizeI);
};

static CUTLASS_HOST_DEVICE int32_t
getSmallestQueryForKey(Params const& p, int32_t key_start) {
if (p.custom_mask_type == NoCustomMask) {
  // 如果没有自定义掩码类型，返回0
  return 0;
}
int32_t shift = p.custom_mask_type == CausalFromBottomRight
    ? p.num_keys - p.num_queries
    : 0;
int32_t window_size =
    p.window_size == 0 ? p.num_queries + p.num_keys : p.window_size;

auto last_key_for_block =
    cutlass::fast_min(key_start + kBlockSizeJ, p.num_keys) - 1;
int first_query = key_start - shift;
// 计算第一个查询和最后一个查询的位置
int last_query = last_key_for_block - shift + window_size - 1;
    // 检查是否没有需要计算的查询范围，若是则返回查询的结束位置
    if (last_query < 0 || first_query >= p.num_queries) {
      return getQueryEnd(p); // nothing to compute in this column
    }
    // 确保 `first_query` 不小于 0
    first_query = cutlass::fast_max(0, first_query);
    // 返回 `first_query` 所在的块的起始位置
    return (first_query / kBlockSizeI) * kBlockSizeI;
  };

  // 返回将写入给定块的核块数量
  // 通常等于键的分割数，但在某些情况下可能不同，比如因果情况或不同的序列长度
  static CUTLASS_HOST_DEVICE int32_t
  getNumParallelBlocksForQuery(Params const& p, int32_t query_start) {
    // 计算键的块数，使用 `ceil_div` 函数向上取整
    int16_t num_key_blocks = ceil_div(p.num_keys, kBlockSizeJ);
    // 如果存在自定义掩码类型
    if (p.custom_mask_type != NoCustomMask) {
      int32_t shift = p.custom_mask_type == CausalFromBottomRight
          ? p.num_keys - p.num_queries
          : 0;
      // 计算当前块的最后查询和键的最后索引
      int32_t last_query_for_block =
          cutlass::fast_min(query_start + kBlockSizeI, p.num_queries) - 1;
      int32_t last_key_for_block =
          cutlass::fast_min(last_query_for_block + shift, p.num_keys - 1);
      // 计算当前块的第一个键的索引
      int32_t first_key_for_block = p.window_size == 0
          ? 0
          : cutlass::fast_max(query_start - p.window_size + 1 + shift, 0);

      // 根据窗口大小调整键块的数量
      if (p.window_size == 0) {
        num_key_blocks = last_key_for_block / kBlockSizeJ + 1;
      } else {
        num_key_blocks = (last_key_for_block / kBlockSizeJ) -
            (first_key_for_block / kBlockSizeJ) + 1;
      }

      // 如果没有有效的键块，则将 `num_key_blocks` 设置为 0
      if (last_key_for_block < 0 || first_key_for_block >= p.num_keys) {
        num_key_blocks = 0;
      }
    }
    // 返回最小值，限制并行块数与设备上键分割数之间的关系
    return cutlass::fast_min(p.num_splits_key_device(), num_key_blocks);
  };

  // 增加迭代，计算下一个要处理的块
  static CUTLASS_HOST_DEVICE void incrIteration(
      Params const& p,
      int32_t query_start,
      int32_t key_start,
      int32_t& next_query,
      int32_t& next_key) {
    // 计算下一个查询块的起始位置
    next_query = query_start + kBlockSizeI;
    next_key = key_start;
    auto query_shift = getQueryStartShift(p);
    // 如果存在查询偏移
    if (query_shift) {
      // 如果下一个查询超出了查询数目，则跳转到下一个键的最小查询位置
      if (next_query >= p.num_queries) {
        next_query = getSmallestQueryForKey(p, key_start);
        return;
      } else if (query_start < query_shift && query_shift <= next_query) {
        // 在此情况下跳转到下一个键
      } else {
        return;
      }
    } else {
      // 如果窗口大小大于 0
      if (p.window_size > 0) {
        int32_t shift = p.custom_mask_type == CausalFromBottomRight
            ? p.num_keys - p.num_queries
            : 0;
        // 最后一个未掩码的键
        int last_key_for_block =
            cutlass::fast_min(key_start + kBlockSizeJ, p.num_keys) - 1;
        int last_query = last_key_for_block - shift + p.window_size - 1;
        // 如果下一个查询小于等于最后查询并且小于查询数目，则返回
        if (next_query <= last_query && next_query < p.num_queries) {
          return;
        }
      } else if (next_query < p.num_queries) {
        return;
      }
      // 跳转到下一个键
    }
    // 下一个键
    next_key = key_start + p.num_splits_key_device() * kBlockSizeJ;
  // 获取下一个查询的起始位置，根据给定参数 p 和 next_key
  next_query = getQueryStart(p, next_key);
}

template <bool kForceReloadK>
static CUTLASS_DEVICE void prologueQkNextIteration(
    SharedStorage& shared_storage,
    Params const& p,
    int32_t query_start,
    int32_t key_start,
    uint8_t warp_id,
    uint8_t lane_id) {
  // 如果查询起始位置超过了总查询数或者键起始位置超过了总键数，则直接返回
  if (query_start >= p.num_queries || key_start >= p.num_keys) {
    return;
  }

  // 确定是否强制重新加载 k，或者共享内存不包含整个矩阵
  static constexpr bool kReloadK =
      kForceReloadK || !MatmulQK::Mma::kSmemContainsEntireMat;
  // 计算当前线程在 warp 中的 ID
  int thread_id = 32 * warp_id + lane_id;
  // 创建键矩阵的迭代器
  typename MatmulQK::Mma::IteratorA iterator_A(
      {int32_t(p.k_strideM)},
      const_cast<scalar_t*>(p.key_ptr + key_start * p.k_strideM),
      {p.num_keys - key_start, p.head_dim},
      thread_id,
      cutlass::MatrixCoord{0, 0});

  // 创建查询矩阵的迭代器
  typename MatmulQK::Mma::IteratorB iterator_B(
      {int32_t(p.q_strideM)},
      const_cast<scalar_t*>(p.query_ptr + query_start * p.q_strideM),
      {p.head_dim, p.num_queries - query_start},
      thread_id,
      cutlass::MatrixCoord{0, 0});

  // 执行矩阵乘积的前处理
  MatmulQK::Mma::prologue<kReloadK, true>(
      shared_storage.mm_qk_k(),
      shared_storage.mm_qk_q(),
      iterator_A,
      iterator_B,
      thread_id,
      p.head_dim);
}

template <bool skipBoundsChecks>
static CUTLASS_DEVICE void writeFragsToGmem(
    SharedStorage& shared_storage,
    OutputFragments& output_frags,
    Params const& p,
    int32_t key_start,
    uint8_t warp_id,
    uint8_t lane_id) {
  // 计算当前线程在 warp 中的 ID
  uint16_t thread_id = 32 * warp_id + lane_id;
  // 计算当前块中的键数，如果跳过边界检查，则使用 Mma 的形状参数 kM
  int32_t num_keys_in_block = skipBoundsChecks
      ? MatmulQK::Mma::Shape::kM
      : cutlass::fast_min(
            (int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start);
  
  // 创建梯度值输出的迭代器
  typename MatmulGradV::OutputTileIterator outputV_it(
      typename MatmulGradV::OutputTileIterator::Params{p.gV_strideM()},
      p.grad_value_ptr + key_start * p.gV_strideM(),
      {num_keys_in_block, p.head_dim_value},
      thread_id);
  
  // 在全局内存中累加梯度值
  accumulateInGmem<MatmulGradV>(
      shared_storage.gradV_epilogue_final(),
      output_frags.gradV,
      outputV_it,
      true,
      warp_id,
      lane_id);

  // 创建梯度键输出的迭代器
  typename MatmulGradK::OutputTileIterator outputK_it(
      typename MatmulGradK::OutputTileIterator::Params{p.gK_strideM()},
      p.grad_key_ptr + key_start * p.gK_strideM(),
      {num_keys_in_block,
       false ? MatmulGradK::ThreadblockShape::kN : p.head_dim},
      thread_id);
  
  // 在全局内存中累加梯度键
  accumulateInGmem<MatmulGradK>(
      shared_storage.gradK_epilogue_final(),
      output_frags.gradK,
      outputK_it,
      true,
      warp_id,
      lane_id);
}

template <typename MatmulT>
static CUTLASS_DEVICE void accumulateInGmem(
    typename MatmulT::DefaultEpilogue::SharedStorage& epilogue_smem,
    typename MatmulT::Mma::FragmentC const& accum,
    typename MatmulT::OutputTileIterator output_it,
    bool first,
    uint8_t warp_id,
    uint8_t lane_id) {
    // 使用别名指定默认的后处理（Epilogue）、输出操作（OutputOp）和矩阵乘法累加器（Mma）
    using DefaultEpilogue = typename MatmulT::DefaultEpilogue;
    using DefaultOutputOp = typename MatmulT::DefaultOutputOp;
    using Mma = typename MatmulT::Mma;
    // 计算当前线程的唯一标识符
    int thread_id = 32 * warp_id + lane_id;
    // 根据条件分发执行不同的操作，这里根据 kIsFirst 判断选择不同的 ScaleType
    DISPATCH_BOOL(
        first, kIsFirst, ([&]() {
          // 根据 kIsFirst 的值选择 ScaleType，决定是否对输出进行 beta 缩放
          static constexpr auto ScaleType = kIsFirst
              ? cutlass::epilogue::thread::ScaleType::Nothing
              : cutlass::epilogue::thread::ScaleType::NoBetaScaling;
          // 使用线性组合类型的后处理输出操作类型
          using EpilogueOutputOp =
              typename cutlass::epilogue::thread::LinearCombination<
                  typename DefaultOutputOp::ElementOutput,
                  DefaultOutputOp::kCount,
                  typename DefaultOutputOp::ElementAccumulator,
                  typename DefaultOutputOp::ElementCompute,
                  ScaleType>;
          // 定义用于线程块后处理的 Epilogue 类型
          using Epilogue =
              typename cutlass::epilogue::threadblock::EpiloguePipelined<
                  typename DefaultEpilogue::Shape,
                  typename Mma::Operator,
                  DefaultEpilogue::kPartitionsK,
                  typename MatmulT::OutputTileIterator,
                  typename DefaultEpilogue::AccumulatorFragmentIterator,
                  typename DefaultEpilogue::WarpTileIterator,
                  typename DefaultEpilogue::SharedLoadIterator,
                  EpilogueOutputOp,
                  typename DefaultEpilogue::Padding,
                  DefaultEpilogue::kFragmentsPerIteration,
                  true // IterationsUnroll
                  >;
          // 创建 EpilogueOutputOp 对象进行重新缩放
          EpilogueOutputOp rescale({1, 1});
          // 使用 Epilogue 对象处理后处理操作
          Epilogue epilogue(epilogue_smem, thread_id, warp_id, lane_id);
          epilogue(rescale, output_it, accum, output_it);
        }));
  }

  template <int kElementsPerAccess>
  // 计算 Delta 的函数模板
  static CUTLASS_DEVICE void computeDelta(
      // 参数列表，包括参数结构体 p，查询起始位置 query_start，warp ID 和 lane ID
      Params const& p,
      int32_t query_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    // 每个线程计算一个 Delta 值
    // 根据 warp 配置，可能有同一 warp 中的多个线程处理同一行数据
    using AccessType = cutlass::Array<scalar_t, kElementsPerAccess>;
    // 静态断言确保 kNumThreads 大于等于 kBlockSizeI
    static_assert(kNumThreads >= kBlockSizeI, "");
    // 计算每行中的线程数
    static constexpr int kNumThreadsPerLine = kNumThreads / kBlockSizeI;
    // 计算当前线程的唯一标识符
    int16_t thread_id = 32 * warp_id + lane_id;

    // 计算当前线程处理的首列位置
    int16_t laneFirstCol = kElementsPerAccess * (lane_id % kNumThreadsPerLine);
    // 计算当前线程处理的行索引
    int16_t laneRow = thread_id / kNumThreadsPerLine;
    // 检查是否满足行索引的条件
    bool rowPred = (query_start + laneRow) < p.num_queries;
    // 综合行索引条件
    bool pred = rowPred;

    // 在 Windows 上，使用 __restrict__ AccessType* 的语法将指针类型转换为 const AccessType*
    const AccessType* __restrict__ grad_output_ptr =
        reinterpret_cast<const AccessType*>(
            p.grad_output_ptr + (query_start + laneRow) * p.gO_strideM +
            laneFirstCol);
    const AccessType* __restrict__ output_ptr =
        reinterpret_cast<const AccessType*>(
            p.output_ptr + (query_start + laneRow) * p.o_strideM() +
            laneFirstCol);
    // 定义输出指针，用于访问输出数据的特定位置

    static constexpr int64_t kMaxIters =
        kMaxK / (kElementsPerAccess * kNumThreadsPerLine);
    // 计算迭代的最大次数，基于数据访问模式和线程数

    constexpr int kPipelineStages = 2;
    // 定义管道阶段数量

    accum_t delta_value = accum_t(0);
    // 初始化累加变量 delta_value

    using GlobalLoad =
        cutlass::arch::global_load<AccessType, sizeof(AccessType)>;
    // 使用全局加载器 GlobalLoad 加载数据

    AccessType frag_grad_output[kPipelineStages];
    AccessType frag_output[kPipelineStages];
    // 定义片段数组，用于存储梯度输出和输出数据的片段

    auto loadAndIncrement = [&](int ld_pos, bool is_valid) {
      // 加载数据并增加指针位置
      frag_grad_output[ld_pos].clear();
      // 清空梯度输出片段
      frag_output[ld_pos].clear();
      // 清空输出数据片段
      GlobalLoad(frag_grad_output[ld_pos], grad_output_ptr, is_valid);
      // 使用全局加载器加载梯度输出数据到片段数组
      GlobalLoad(frag_output[ld_pos], output_ptr, is_valid);
      // 使用全局加载器加载输出数据到片段数组
      grad_output_ptr += kNumThreadsPerLine;
      // 增加梯度输出数据指针位置
      output_ptr += kNumThreadsPerLine;
      // 增加输出数据指针位置
    };

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < kPipelineStages - 1; ++iter) {
      // 使用指令调用优化循环展开
      int ld_pos = iter % kPipelineStages;
      // 计算当前加载位置
      pred = pred &&
          (laneFirstCol + iter * kElementsPerAccess * kNumThreadsPerLine) <
              p.head_dim_value;
      // 计算预测值，检查是否超出输出数据维度
      loadAndIncrement(ld_pos, pred);
      // 调用加载和增加函数
    }

    auto columnIteration = [&](int iter) {
      // 列迭代函数
      int ld_pos = (iter + kPipelineStages - 1) % kPipelineStages;
      // 计算加载位置
      pred = pred &&
          (laneFirstCol +
           (iter + kPipelineStages - 1) * kElementsPerAccess *
               kNumThreadsPerLine) < p.head_dim_value;
      // 计算预测值，检查是否超出输出数据维度
      loadAndIncrement(ld_pos, pred);
      // 调用加载和增加函数
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < AccessType::kElements; ++i) {
        // 使用指令调用优化循环展开
        delta_value += accum_t(frag_output[iter % kPipelineStages][i]) *
            accum_t(frag_grad_output[iter % kPipelineStages][i]);
        // 计算 delta_value，执行输出数据片段和梯度输出片段的累加乘积
      }
    };

    // 如果 K 的下限较小，可以展开循环
    if (kMaxK <= 256) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < kMaxIters; ++iter) {
        // 使用指令调用优化循环展开
        columnIteration(iter);
        // 调用列迭代函数
      }
    } else {
      int num_iters =
          ceil_div(p.head_dim_value, kElementsPerAccess * kNumThreadsPerLine) *
          (kElementsPerAccess * kNumThreadsPerLine);
      // 计算迭代次数，确保能够覆盖整个数据维度
      for (int iter = 0; iter < num_iters; ++iter) {
        // 循环执行列迭代函数
        columnIteration(iter);
      }
    }

    // 在工作线程之间进行归约
    static_assert(
        kNumThreadsPerLine == 1 || kNumThreadsPerLine == 2 ||
            kNumThreadsPerLine == 4,
        "");
    // 静态断言，确保线程数符合要求
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < kNumThreadsPerLine; i *= 2) {
      // 使用指令调用优化循环展开
      delta_value = delta_value + __shfl_xor_sync(0xffffffff, delta_value, i);
      // 使用 Warp Shuffle 操作进行归约
    }

    // 存储结果到全局内存
    if (rowPred) {
      p.delta_ptr[query_start + laneRow] = delta_value;
      // 如果行预测为真，将 delta_value 存储到指定位置
    }
  }
};

// 结束了一个未命名的命名空间

template <typename AK>
// 定义了一个模板函数 attention_kernel_backward_batched_impl，使用 AK 类型作为模板参数
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
// CUDA __global__ 函数声明，使用 __launch_bounds__ 限制线程块启动参数
attention_kernel_backward_batched_impl(typename AK::Params p) {
  // 检查是否可以进入下一个线程块，否则返回
  if (!p.advance_to_block()) {
    return;
  }
  // 调用 AK 类型的 attention_kernel 函数，传入参数 p
  AK::attention_kernel(p);
}

template <typename AK>
// 定义了一个模板函数 attention_kernel_backward_batched，使用 AK 类型作为模板参数
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
// CUDA __global__ 函数声明，使用 __launch_bounds__ 限制线程块启动参数
attention_kernel_backward_batched(typename AK::Params params);

} // namespace PyTorchMemEffAttention
// 结束了命名空间 PyTorchMemEffAttention
```