# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\kernel_forward.h`

```
/*
 * 版权所有 Meta Platforms, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在根目录下的 LICENSE 文件中使用 BSD 风格许可证授权。
 */
#pragma once

#include <ATen/cuda/PhiloxUtils.cuh> // 导入 CUDA 的 PhiloxUtils.cuh 头文件
#include <c10/util/Exception.h>     // 导入 C10 异常处理工具

#include <curand_kernel.h>           // 导入 CUDA 的 curand_kernel.h 头文件
#include <cmath>                     // 导入数学函数库
#include <vector>                    // 导入向量操作库

#include <cutlass/bfloat16.h>                            // 导入 cutlass 的 bfloat16 头文件
#include <cutlass/fast_math.h>                           // 导入 cutlass 的 fast_math 头文件
#include <cutlass/gemm/gemm.h>                           // 导入 cutlass 的矩阵乘法头文件
#include <cutlass/layout/matrix.h>                       // 导入 cutlass 的矩阵布局头文件
#include <cutlass/layout/vector.h>                       // 导入 cutlass 的向量布局头文件
#include <cutlass/matrix.h>                             // 导入 cutlass 的矩阵头文件
#include <cutlass/numeric_types.h>                       // 导入 cutlass 的数值类型头文件
#include <cutlass/tensor_ref.h>                          // 导入 cutlass 的张量引用头文件

#include <cutlass/epilogue/threadblock/default_epilogue_simt.h>    // 导入 cutlass 的默认线程块后处理 SIMT 头文件
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h> // 导入 cutlass 的默认线程块后处理张量操作头文件
#include <cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h> // 导入 cutlass 的默认线程块后处理 Volta 张量操作头文件

#include <cutlass/gemm/device/default_gemm_configuration.h>   // 导入 cutlass 的默认 GEMM 配置头文件
#include <cutlass/gemm/kernel/default_gemm.h>                 // 导入 cutlass 的默认 GEMM 核心头文件
#include <cutlass/gemm/threadblock/default_mma.h>             // 导入 cutlass 的默认 MMA（Mixed Matrix Accumulation）线程块头文件
#include <cutlass/gemm/threadblock/default_mma_core_simt.h>    // 导入 cutlass 的默认 MMA 核心 SIMT 头文件
#include <cutlass/gemm/threadblock/default_mma_core_sm70.h>    // 导入 cutlass 的默认 MMA 核心 SM70 头文件
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>    // 导入 cutlass 的默认 MMA 核心 SM75 头文件
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>    // 导入 cutlass 的默认 MMA 核心 SM80 头文件
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>      // 导入 cutlass 的线程块调度头文件
#include <cutlass/matrix_shape.h>                              // 导入 cutlass 的矩阵形状头文件
#include <cutlass/platform/platform.h>                         // 导入 cutlass 的平台头文件
#include <cutlass/transform/threadblock/predicated_tile_iterator.h> // 导入 cutlass 的带有断言的瓦片迭代器头文件

#include <ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h>      // 导入 ATen 的 CUDA 内存高效注意力机制调试工具头文件
#include <ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_pipelined.h> // 导入 ATen 的 CUDA 内存高效注意力机制管道化后处理头文件
#include <ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_rescale_output.h> // 导入 ATen 的 CUDA 内存高效注意力机制输出重缩头文件

#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h>    // 导入 ATen 的 CUDA 内存高效注意力机制自定义 MMA 头文件
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h> // 导入 ATen 的 CUDA 内存高效注意力机制查找默认 MMA 头文件
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_from_smem.h> // 导入 ATen 的 CUDA 内存高效注意力机制从 SMEM 获取 MMA 头文件
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h>  // 导入 ATen 的 CUDA 内存高效注意力机制 GEMM 核心工具头文件
#include <ATen/native/transformers/cuda/mem_eff_attention/transform/tile_smem_loader.h> // 导入 ATen 的 CUDA 内存高效注意力机制 SMEM 瓦片加载器头文件

#include <cinttypes>  // 导入 C 语言整数类型头文件

using namespace gemm_kernel_utils;  // 使用 gemm_kernel_utils 命名空间

namespace PyTorchMemEffAttention {
namespace {

// 获取每个 SM 的 warp 数量，根据架构不同返回不同的值
template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSmFw() {
  return (
      Arch::kMinComputeCapability >= 80 &&
              !cutlass::platform::is_same<scalar_t, float>::value
          ? 16
          : 12);
}

// 原子级别的浮点数最大值更新函数，参考自 https://stackoverflow.com/a/51549250
static CUTLASS_DEVICE float atomicMaxFloat(float* addr, float value) {
  return !signbit(value)
             ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
             : __uint_as_float(
                   atomicMin((unsigned int *)addr, __float_as_uint(value)));
}

} // namespace
    // 如果 Q/K/V 在内存中正确对齐，并且可以运行快速的内核
    bool isAligned_,
    // 每个块中查询的数量
    int kQueriesPerBlock_,
    // 每个块中键的数量
    int kKeysPerBlock_,
    // value.shape[-1] 和 query.shape[-1] 的最大值上限
    int kMaxK_ = (int)cutlass::platform::numeric_limits<uint32_t>::max(),
    // 对于某些原因，在 V100 上速度较慢
    // 如果在编译时确定永远不需要 dropout，则设置为 false
    bool kSupportsDropout_ = true,
    // 是否支持偏置
    bool kSupportsBias_ = true>
struct AttentionKernel {
  // 自定义遮罩类型枚举
  enum CustomMaskType {
    NoCustomMask = 0,
    CausalFromTopLeft = 1,
    CausalFromBottomRight = 2,
    NumCustomMaskTypes,
  };

  // 标量类型定义
  using scalar_t = scalar_t_;
  // 累加器类型定义
  using accum_t = float;
  // LSE（LogSumExp）标量类型定义
  using lse_scalar_t = float;
  // 输出类型定义
  using output_t = scalar_t;
  // 两次迭代之间的累加器类型，使用 `accum_t` 在 f16 上提高性能但会导致数值误差
  using output_accum_t = accum_t;
  // 是否支持 dropout
  static constexpr bool kSupportsDropout = kSupportsDropout_;
  // 是否支持偏置
  static constexpr bool kSupportsBias = kSupportsBias_;
  // 每个块的键数
  static constexpr int kKeysPerBlock = kKeysPerBlock_;
  // 每个块的查询数
  static constexpr int kQueriesPerBlock = kQueriesPerBlock_;
  // 最大 K 值
  static constexpr int kMaxK = kMaxK_;
  // 是否对齐
  static constexpr bool kIsAligned = isAligned_;
  // 是否单值迭代
  static constexpr bool kSingleValueIteration = kMaxK <= kKeysPerBlock;
  // AlignLSE 值，用于后向计算的块大小
  static constexpr int32_t kAlignLSE = 32;
  // 是否半精度
  static constexpr bool kIsHalf = cutlass::sizeof_bits<scalar_t>::value == 16;
  // 是否预加载 V
  static constexpr bool kPreloadV =
      ArchTag::kMinComputeCapability >= 80 && kIsHalf;
  // 是否在寄存器文件中保留输出
  static constexpr bool kKeepOutputInRF = kSingleValueIteration;
  // 是否需要输出累加器缓冲区
  static constexpr bool kNeedsOutputAccumulatorBuffer = !kKeepOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  // 确保 kQueriesPerBlock 是 32 的倍数
  static_assert(kQueriesPerBlock % 32 == 0, "");
  // 确保 kKeysPerBlock 是 32 的倍数
  static_assert(kKeysPerBlock % 32 == 0, "");
  // 每个块的 warps 数量
  static constexpr int kNumWarpsPerBlock =
      kQueriesPerBlock * kKeysPerBlock / (32 * 32);
  // 每个 warps 的大小
  static constexpr int kWarpSize = 32;

  // 启动限制
  static constexpr int kNumThreads = kWarpSize * kNumWarpsPerBlock;
  // 每个 SM 的最小块数
  static constexpr int kMinBlocksPerSm =
      getWarpsPerSmFw<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  struct Params {
    // 输入张量
    const scalar_t* query_ptr = nullptr; // [num_queries, num_heads, head_dim]
    const scalar_t* key_ptr = nullptr; // [num_keys, num_heads, head_dim]
    const scalar_t* value_ptr = nullptr; // [num_keys, num_heads, head_dim_value]
    const scalar_t* attn_bias_ptr = nullptr; // [num_heads, num_queries, num_keys]
    const int32_t* seqstart_q_ptr = nullptr;
    const int32_t* seqstart_k_ptr = nullptr;

    const int32_t* seqlen_k_ptr = nullptr;
    uint32_t causal_diagonal_offset = 0;

    // 输出张量
    output_t* output_ptr = nullptr; // [num_queries, num_heads, head_dim_value]
    output_accum_t* output_accum_ptr = nullptr; // [num_queries, num_heads, head_dim_value]
    lse_scalar_t* logsumexp_ptr = nullptr; // [num_heads, num_queries] - 可以为 null

    // 滑动窗口，如果为 0 则忽略
    int32_t window_size = 0;

    // 缩放比例
    accum_t scale = 0.0;

    // 维度/步幅
    int32_t head_dim = 0;
    int32_t head_dim_value = 0;
    int32_t num_queries = 0;
    int32_t num_keys = 0;
    int32_t num_keys_absolute = 0;

    // 自定义遮罩类型
    uint8_t custom_mask_type = NoCustomMask;

    int32_t q_strideM = 0;
    int32_t k_strideM = 0;
    int32_t v_strideM = 0;
    int32_t bias_strideM = 0;

    int32_t o_strideM = 0;
    // 下面的所有内容仅在 `advance_to_block` 中使用，并且不应使用寄存器

    // 定义各种数据流的步长
    int32_t q_strideH = 0;
    int32_t k_strideH = 0;
    int32_t v_strideH = 0;
    int64_t bias_strideH = 0;

    int64_t q_strideB = 0;
    int64_t k_strideB = 0;
    int64_t v_strideB = 0;
    int64_t bias_strideB = 0;

    // 定义批次数量和头数
    int32_t num_batches = 0;
    int32_t num_heads = 0;

    // dropout 相关变量
    bool use_dropout = false; // 是否使用 dropout
    unsigned long long dropout_batch_head_rng_offset = 0; // 随机数生成器偏移量
    float dropout_prob = 0.0f; // dropout 概率
    at::PhiloxCudaState rng_engine_inputs = at::PhiloxCudaState(0, 0); // 随机数引擎状态
    int64_t* extragraph_offset = nullptr; // 额外图偏移量指针
    int64_t* seed = nullptr; // 随机种子指针

    // 移动指针以处理需要处理的内容
    // 如果没有要处理的工作，则返回 "false"
    }

    // 获取块的网格维度
    __host__ dim3 getBlocksGrid() const {
      return dim3(
          // 计算块的数量，确保至少包含一个查询
          ceil_div(num_queries, (int32_t)kQueriesPerBlock),
          // 使用的头数
          num_heads,
          // 使用的批次数
          num_batches);
    }

    // 获取线程的网格维度
    __host__ dim3 getThreadsGrid() const {
      return dim3(
          // 每个 warp 中的线程数
          kWarpSize,
          // 每个块中的 warp 数量
          kNumWarpsPerBlock,
          // 每个块中的线程块数
          1);
    }
  };

  struct MM0 {
    /*
      在这个第一个矩阵乘法中，我们计算 `Q @ K.T` 的一个块。
      当计算结果仍然存储在寄存器中时，我们在共享内存中更新 `mi`, `m_prime`, `s_prime`，
      然后将这个值存储到一个后续用作第二个矩阵乘法（见 MM1）操作数 A 的共享内存中。
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            scalar_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA =
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment;
    static constexpr int kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::ColumnMajor, // LayoutB,
        kAlignmentB,
        accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        OpClass,
        ArchTag, // ArchTag
        ThreadblockShape, // ThreadblockShape
        WarpShape, // WarpShape
        typename GemmType::InstructionShape, // InstructionShape
        ArchTag::kMinComputeCapability >= 80 && kIsHalf
            ? 4
            : DefaultConfig::kStages,
        typename GemmType::Operator // Operator
        >::DefaultMma;
    using MmaCore = typename DefaultMma::MmaCore;
    // 定义迭代器类型 IteratorA，指向默认的矩阵乘法算子 A 的迭代器
    using IteratorA = typename DefaultMma::IteratorA;
    // 定义迭代器类型 IteratorB，指向默认的矩阵乘法算子 B 的迭代器
    using IteratorB = typename DefaultMma::IteratorB;
    // 定义默认的线程块矩阵乘法类型 DefaultThreadblockMma
    using DefaultThreadblockMma = typename DefaultMma::ThreadblockMma;
    // 根据条件选择特定的矩阵乘法算子类型 Mma
    using Mma = typename cutlass::platform::conditional<
        kSingleValueIteration,
        typename MakeCustomMma<DefaultThreadblockMma, kMaxK>::Mma,
        DefaultThreadblockMma>::type;
    // 定义累加器的 Lambda 迭代器类型 AccumLambdaIterator
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Iterator;
    // 静态断言，确保线程块的乘法核数量满足预期
    static_assert(
        MmaCore::WarpCount::kM * MmaCore::WarpCount::kN *
                MmaCore::WarpCount::kK ==
            kNumWarpsPerBlock,
        "");

    // 用于从全局内存高效加载偏置矩阵 Bij 到共享内存的载入器类型 BiasLoader
    using BiasLoader = TileSmemLoader<
        scalar_t,
        cutlass::MatrixShape<kQueriesPerBlock, kKeysPerBlock>,
        MmaCore::kThreads,
        // 输入限制：kv_len 必须是该值的倍数
        128 / cutlass::sizeof_bits<scalar_t>::value>;

    // 用于存储 Epilogue 的结构，以便后续的第二次矩阵乘法使用
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    // 累加器的共享内存存储类型 AccumulatorSharedStorage
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
};
  
struct MM1 {
    /**
      Second matmul: perform `attn @ V` where `attn` is the attention (not
      normalized) and stored in shared memory
    */
    // 定义 GemmType 类型，作为默认 Gemm 类型的别名
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    // 定义操作类 OpClass，来自 GemmType
    using OpClass = typename GemmType::OpClass;
    // 定义默认的 Gemm 配置类型 DefaultConfig
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            output_accum_t, // ElementC
            accum_t // ElementAccumulator
            >;
    // 从共享内存加载的对齐要求 kAlignmentA
    static constexpr int kAlignmentA = DefaultConfig::kAlignmentA;
    // 是否对齐的条件下的 kAlignmentB
    static constexpr int kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    // 定义线程块的形状 ThreadblockShape
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    // 定义线程束的形状 WarpShape
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    // 定义指令的形状 InstructionShape，来自 GemmType
    using InstructionShape = typename GemmType::InstructionShape;

    // 定义矩阵 B 的布局类型 LayoutB
    using LayoutB = cutlass::layout::RowMajor;
    // 定义默认的 GEMM 类型，用于矩阵乘法计算
    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA, 第一个矩阵元素类型
        cutlass::layout::RowMajor, // LayoutA, 第一个矩阵布局（行主序）
        kAlignmentA, // ElementA 的对齐方式
        scalar_t, // ElementB, 第二个矩阵元素类型
        LayoutB, // LayoutB, 第二个矩阵布局
        kAlignmentB, // ElementB 的对齐方式
        output_accum_t, // 输出累加器类型
        cutlass::layout::RowMajor, // LayoutC, 输出矩阵布局
        accum_t, // 累加器类型
        OpClass, // 操作类别
        ArchTag, // 架构标签
        ThreadblockShape, // 线程块形状
        WarpShape, // Warp 形状
        typename GemmType::InstructionShape, // 指令形状
        typename DefaultConfig::EpilogueOutputOp, // 回传输出操作
        void, // ThreadblockSwizzle - 未使用
        ArchTag::kMinComputeCapability >= 80 && kIsHalf
            ? 4
            : DefaultConfig::kStages, // 阶段数，根据条件选择
        false, // SplitKSerial
        typename GemmType::Operator>; // GEMM 运算符类型

    // 定义 Warp A 迭代器类型，从共享内存中获取
    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Policy::Operator::Shape, // WarpShape
            typename DefaultGemm::Mma::Policy::Operator::InstructionShape,
            typename DefaultGemm::Mma::Policy::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;

    // 定义默认从共享内存中获取的 Mma 类型
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma, // Mma 类型
            MM0::AccumulatorSharedStorage::Shape::kN, // kMaxK，最大 K 值
            WarpIteratorA, // Warp A 迭代器
            false>; // kScaleOperandA，是否缩放操作数 A

    // 定义 Mma 类型
    using Mma = typename DefaultMmaFromSmem::Mma;

    // 定义 IteratorB 类型
    using IteratorB = typename Mma::IteratorB;

    // 定义 WarpCount 类型
    using WarpCount = typename Mma::WarpCount;

    // 静态断言，检查线程块的 Warp 数目是否与给定值相符
    static_assert(
        WarpCount::kM * WarpCount::kN * WarpCount::kK == kNumWarpsPerBlock,
        "");

    // 定义默认的 Epilogue 类型
    using DefaultEpilogue = typename DefaultGemm::Epilogue;

    // 定义输出 Tile 迭代器类型，用于后处理
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_t>;

    // 定义累加输出 Tile 迭代器类型，用于后处理
    using OutputTileIteratorAccum =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_accum_t>;

    // MM0 的 A 对齐值
    static constexpr int64_t kAlignmentQ = MM0::kAlignmentA;

    // MM0 的 B 对齐值
    static constexpr int64_t kAlignmentK = MM0::kAlignmentB;

    // MM0 的 V 对齐值
    static constexpr int64_t kAlignmentV = 1;

    // 共享存储 - 取决于内核参数
    struct ScalingCoefs {
        cutlass::Array<accum_t, kQueriesPerBlock> m_prime; // 主要值 m
        cutlass::Array<accum_t, kQueriesPerBlock> s_prime; // 主要值 s
        cutlass::Array<accum_t, kQueriesPerBlock> mi; // 中间值 mi
        cutlass::Array<accum_t, kQueriesPerBlock> out_rescale; // 输出重新缩放值
        cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>
            addition_storage; // 附加存储空间
    };

    // Epilogue 结束后的共享存储结构
    struct SharedStorageEpilogueAtEnd : ScalingCoefs {
        struct SharedStorageAfterMM0 {
            // MM0 中可能被覆盖的所有内容
            union {
                typename MM0::BiasLoader::SmemTile bias; // 偏置加载器的共享内存 Tile
                typename MM0::AccumulatorSharedStorage si; // MM0 累加器的共享存储
            };
            typename MM1::Mma::SharedStorage mm1; // MM1 中的共享存储
        };
    };
    // 定义一个联合体，包含三种不同的共享存储类型
    union {
      typename MM0::Mma::SharedStorage mm0;  // 在 MM0 后的共享存储
      SharedStorageAfterMM0 after_mm0;        // MM0 后的共享存储之后
      typename MM1::DefaultEpilogue::SharedStorage epilogue;  // 默认尾声的共享存储
    };

    // 返回默认尾声的共享存储的引用
    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return epilogue;
    }
  };

  // 定义一个结构体，包含缩放系数和在循环中的共享存储
  struct SharedStorageEpilogueInLoop : ScalingCoefs {
    // 定义一个在 MM0 后的共享存储的联合体
    struct SharedStorageAfterMM0 {
      // 这里的内容可能在 MM0 运行期间被覆盖
      union {
        typename MM0::BiasLoader::SmemTile bias;  // 偏置加载器的共享内存瓦片
        typename MM0::AccumulatorSharedStorage si;  // 累加器的共享存储
      };
      typename MM1::Mma::SharedStorage mm1;  // MM1 的共享存储
      typename MM1::DefaultEpilogue::SharedStorage epilogue;  // 默认尾声的共享存储
    };

    // 定义一个联合体，包含 MM0 的共享存储和 MM0 后的共享存储
    union {
      typename MM0::Mma::SharedStorage mm0;  // MM0 的共享存储
      SharedStorageAfterMM0 after_mm0;        // MM0 后的共享存储
    };

    // 返回默认尾声的共享存储的引用
    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return after_mm0.epilogue;
    }
  };

  // 使用条件模板定义共享存储的类型，根据 kSingleValueIteration 或 kKeepOutputInRF 来选择不同的结构体
  using SharedStorage = typename cutlass::platform::conditional<
      kSingleValueIteration || kKeepOutputInRF,
      SharedStorageEpilogueAtEnd,  // 使用结尾的共享存储
      SharedStorageEpilogueInLoop  // 使用循环中的共享存储
    >::type;

  // 静态函数，检查给定的参数是否被支持
  static bool __host__ check_supported(Params const& p) {
    // 检查指针是否按指定的对齐方式对齐
    CHECK_ALIGNED_PTR(p.query_ptr, kAlignmentQ);
    CHECK_ALIGNED_PTR(p.key_ptr, kAlignmentK);
    CHECK_ALIGNED_PTR(p.value_ptr, kAlignmentV);

    // 如果支持偏置，则继续检查偏置参数的对齐情况
    if (kSupportsBias) {
      CHECK_ALIGNED_PTR(p.attn_bias_ptr, kAlignmentQ);
      TORCH_CHECK(
          p.num_batches <= 1 || p.bias_strideB % kAlignmentQ == 0,
          "attn_bias is not correctly aligned (strideB). ",
          "attn_bias.stride( 0) = ", p.bias_strideB, ", and should be a "
          "multiple of ", kAlignmentQ, ".");
      TORCH_CHECK(
          p.num_heads <= 1 || p.bias_strideH % kAlignmentQ == 0,
          "attn_bias is not correctly aligned (strideH). "
          "attn_bias.stride(1) = ", p.bias_strideH, ", and should be a "
          "multiple of ", kAlignmentQ, ".");
      TORCH_CHECK(
          p.num_queries <= 1 || p.bias_strideM % kAlignmentQ == 0,
          "attn_bias is not correctly aligned (strideM). "
          "attn_bias.stride(2) = ", p.bias_strideM, ", and should be a "
          "multiple of ", kAlignmentQ, ".");
    }

    // 检查查询、键、值的步长是否按指定的对齐方式对齐
    TORCH_CHECK(
        p.q_strideM % kAlignmentQ == 0,
        "query is not correctly aligned (strideM)");
    TORCH_CHECK(
        p.k_strideM % kAlignmentK == 0,
        "key is not correctly aligned (strideM)");
    TORCH_CHECK(
        p.v_strideM % kAlignmentV == 0,
        "value is not correctly aligned (strideM)");
    TORCH_CHECK(
        p.num_heads <= 1 || p.q_strideH % kAlignmentQ == 0,
        "query is not correctly aligned (strideH)");
    TORCH_CHECK(
        p.num_heads <= 1 || p.k_strideH % kAlignmentK == 0,
        "key is not correctly aligned (strideH)");
    TORCH_CHECK(
        p.num_heads <= 1 || p.v_strideH % kAlignmentV == 0,
        "value is not correctly aligned (strideH)");
    // 检查自定义掩码类型是否有效，确保其小于预定义的掩码类型数量
    TORCH_CHECK(
        p.custom_mask_type < NumCustomMaskTypes,
        "invalid value for `custom_mask_type`");

    // 如果窗口大小大于0，则进一步检查自定义掩码类型是否支持
    if (p.window_size > 0) {
      TORCH_CHECK(
          p.custom_mask_type == CausalFromTopLeft ||
              p.custom_mask_type == CausalFromBottomRight,
          "custom_mask_type not supported");
    }
    // 函数执行成功，返回 true
    return true;
  }

  // 定义一个静态函数，实现注意力机制的核心计算
  static void CUTLASS_DEVICE attention_kernel(Params& p) {
    // 在这个代码块中，我们仅仅会：
    // - 读取 query[query_start:query_end, :]
    // - 写入 output[query_start:query_end, :]

    // 外部共享内存，用于存储共享数据结构
    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime = shared_storage.s_prime;
    auto& mi = shared_storage.mi;
    auto& out_rescale = shared_storage.out_rescale;
    // 计算查询起始位置
    const uint32_t query_start = blockIdx.x * kQueriesPerBlock;

    // 静态断言，确保每个块中的查询数量小于每个块中的线程数乘以每个线程的最大数量
    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");

    // 如果线程 ID 小于每个块中的查询数量，则初始化相关的共享存储数据
    if (thread_id() < kQueriesPerBlock) {
      s_prime[thread_id()] = accum_t(0);  // 初始化 s_prime
      out_rescale[thread_id()] = accum_t(1.0);  // 初始化 out_rescale
      m_prime[thread_id()] =
          -cutlass::platform::numeric_limits<accum_t>::infinity();  // 初始化 m_prime
      mi[thread_id()] = -cutlass::platform::numeric_limits<accum_t>::infinity();  // 初始化 mi
    }

    typename MM1::Mma::FragmentC accum_o;
    accum_o.clear();  // 清空累加器

    // 定义一个函数，用于创建输出迭代器，根据列数进行初始化
    auto createOutputIter = [&](int col) -> typename MM1::OutputTileIterator {
      using OutputTileIterator = typename MM1::OutputTileIterator;
      return OutputTileIterator(
          typename OutputTileIterator::Params{(int32_t)p.o_strideM},
          p.output_ptr,
          typename OutputTileIterator::TensorCoord{
              p.num_queries, p.head_dim_value},
          thread_id(),
          {0, col});
    };

    // 定义一个函数，用于创建输出累加迭代器，根据列数进行初始化
    auto createOutputAccumIter = [&](int col) ->
        typename MM1::OutputTileIteratorAccum {
          using OutputTileIteratorAccum = typename MM1::OutputTileIteratorAccum;
          return OutputTileIteratorAccum(
              typename OutputTileIteratorAccum::Params{
                  (int32_t)(p.head_dim_value * p.num_heads)},
              p.output_accum_ptr,
              typename OutputTileIteratorAccum::TensorCoord{
                  p.num_queries, p.head_dim_value},
              thread_id(),
              {0, col});
        };

    curandStatePhilox4_32_10_t curand_state_init;
    // 如果支持 dropout 并且正在使用 dropout
    if (kSupportsDropout && p.use_dropout) {
      // 从 RNG 引擎输入中解包种子
      const auto seeds = at::cuda::philox::unpack(p.rng_engine_inputs);
      
      // 如果 RNG 引擎输入已被捕获
      if (p.rng_engine_inputs.captured_) {
        // 查看注释 [Seed and Offset Device]
        // 当我们处于 CUDA 图捕获模式时，种子和偏移量存储在设备上
        // 我们传入 int64_t* seed 和 int64_t* offset 作为临时空间，
        // 用于在前向传播期间存储 RNG 状态并在反向传播时保存。
        auto [seed, offset] = seeds;
        *p.seed = seed;               // 将种子存储到 p.seed 中
        *p.extragraph_offset = offset; // 将偏移量存储到 p.extragraph_offset 中
      }

      // 每个注意力矩阵 P 的元素，形状为 (batch_sz, n_heads, n_queries, n_keys)，
      // 关联着 RNG 序列中的单个偏移量。我们用 (n_queries, n_keys) 矩阵的开始偏移
      // 来初始化 RNG 状态，针对当前块的 batch_id 和 head_id。
      // 初始化 RNG 状态非常昂贵，因此我们每个核函数运行一次，而不是每个迭代运行一次。
      // 每次迭代都会复制初始化的 RNG 状态，并根据需要进行偏移。
      curand_init(
          std::get<0>(seeds), // 种子
          0,
          std::get<1>(seeds) + p.dropout_batch_head_rng_offset, // 偏移量
          &curand_state_init); // 初始化的 CURAND 状态
    }

    // 遍历 keys
    // 如果保持输出在 RF 中
    if (kKeepOutputInRF) {
      // 声明并初始化常量 kIsFirst 和 kIsLast
      constexpr bool kIsFirst = true;
      constexpr bool kIsLast = true;
      // 使用 MM1 的默认后处理器和默认配置的输出操作类型
      using DefaultEpilogue = typename MM1::DefaultEpilogue;
      using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
      // 定义 ElementCompute 类型
      using ElementCompute = typename DefaultOp::ElementCompute;
      // 定义 EpilogueOutputOp 类型，用于规范化注意力计算的输出
      using EpilogueOutputOp =
          typename cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
              output_t, // 输出类型
              output_accum_t, // 源类型
              DefaultOp::kCount,
              typename DefaultOp::ElementAccumulator, // 累加器类型
              output_accum_t, // 计算类型
              kIsFirst,
              kIsLast,
              cutlass::Array<ElementCompute, kQueriesPerBlock>>;
      // 定义 Epilogue 类型，用于线程块后处理
      using Epilogue =
          typename cutlass::epilogue::threadblock::EpiloguePipelined<
              typename DefaultEpilogue::Shape,
              typename MM1::Mma::Operator,
              DefaultEpilogue::kPartitionsK,
              typename MM1::OutputTileIterator, // 目标迭代器类型
              typename DefaultEpilogue::AccumulatorFragmentIterator,
              typename DefaultEpilogue::WarpTileIterator,
              typename DefaultEpilogue::SharedLoadIterator,
              EpilogueOutputOp,
              typename DefaultEpilogue::Padding,
              DefaultEpilogue::kFragmentsPerIteration,
              true, // IterationsUnroll
              typename MM1::OutputTileIteratorAccum // 源迭代器类型
              >;
      // 创建目标迭代器对象 dest_iter
      auto dest_iter = createOutputIter(0);
      // 初始化 rescale 对象，用于后处理规范化
      EpilogueOutputOp rescale(s_prime, out_rescale);
      // 初始化 epilogue 对象，传入共享存储、线程 ID、warp ID 和 lane ID
      Epilogue epilogue(
          shared_storage.epilogue_shared_storage(),
          thread_id(),
          warp_id(),
          lane_id());
      // 执行后处理操作，传入 rescale 对象、目标迭代器 dest_iter 和累加器 accum_o
      epilogue(rescale, dest_iter, accum_o);
    }

    // 7. 计算 logsumexp
    // 为了简化反向传播，我们使用 `inf` 填充 logsumexp
    // 这样可以避免一些边界检查，在前向传播期间成本不更高
    // 确保 kQueriesPerBlock 小于 kNumWarpsPerBlock * kWarpSize
    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    // 如果 logsumexp 指针非空，并且线程 ID 小于 kQueriesPerBlock
    if (p.logsumexp_ptr && thread_id() < kQueriesPerBlock) {
      // 计算 lse_dim 的值，确保其是 p.num_queries 向上取整到 kAlignLSE 的倍数
      auto lse_dim = ceil_div((int32_t)p.num_queries, kAlignLSE) * kAlignLSE;
      constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
      // 如果线程 ID 小于 p.num_queries
      if (thread_id() < p.num_queries) {
        // 计算 logsumexp 的值，将结果存入 p.logsumexp_ptr
        p.logsumexp_ptr[thread_id()] = accum_t(mi[thread_id()] / kLog2e) +
            cutlass::fast_log(accum_t(s_prime[thread_id()]));
      } else if (thread_id() < lse_dim) {
        // 否则，将 inf 存入 p.logsumexp_ptr
        p.logsumexp_ptr[thread_id()] =
            cutlass::platform::numeric_limits<accum_t>::infinity();
      }
    }
    /* Iterates on the accumulator and corresponding position on result matrix
    
    (1) Update `mi[r]` to the max value of the row `r`
    (2) In a second iteration do the following:
        (a) accum   <- exp(accum - mi)
        (b) m_prime <- exp(m_prime - mi)
        (c) s_prime <- s_prime * m_prime + sum(accum)
    
    All of this is done on registers, before we store all of this
    on shared memory for the next matmul with Value.
    */
    using Fragment = typename WarpIteratorC::Fragment;
    using LambdaIterator = typename DefaultMmaAccumLambdaIterator<
        WarpIteratorC,
        accum_t,
        kWarpSize>::Iterator;
    // Convert to `accum_t` (rather than double)
    constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E

    static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0, "");
    static constexpr int kLinesPerWarp = kQueriesPerBlock / kNumWarpsPerBlock;

    // Scale the fragment by the logarithm base change constant times the scaling factor
    frag = cutlass::multiplies<Fragment>()(scaling * kLog2e, frag);

    // Calculate the lane offset for the LambdaIterator based on lane_id, warp_id, and tile_offset
    auto lane_offset =
        LambdaIterator::get_lane_offset(lane_id, warp_id, tile_offset);

    // First update `mi` to the max per-row
    {
      accum_t max;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {
            // Initialize max to negative infinity
            max = -cutlass::platform::numeric_limits<accum_t>::infinity();
          },
          [&](int accum_m, int accum_n, int idx) {
            // Update max with the maximum value from frag within the valid column range
            if (accum_n < max_col) {
              max = cutlass::fast_max(max, frag[idx]);
            }
          },
          [&](int accum_m) {
            // Atomic operation to update mi[accum_m] with the max value found in the row
            // Using atomicMaxFloat due to optimization considerations
            atomicMaxFloat(&mi[accum_m], max);
          });
    }

    // Ensure all threads have synchronized to share the updated values of `mi`
    __syncthreads();

    // Prepare for expensive operation of exponentiation, potentially restoring `mi` to negative infinity
    bool restore_mi_to_minus_inf = false;
    if (lane_id < kLinesPerWarp) {
      // 如果当前线程在每个warp中的lane_id小于kLinesPerWarp
      int id = warp_id * kLinesPerWarp + lane_id;
      // 计算当前线程的全局id
      auto m_prime_id = m_prime[id];
      // 获取m_prime数组中id位置的值
      auto mi_id = mi[id];
      // 获取mi数组中id位置的值
      bool changed = m_prime_id < mi_id; // `false` if both are -inf
      // 检查是否m_prime_id小于mi_id，如果是则为true，否则为false（如果都是-inf则为false）
      if (changed) {
        // 如果changed为true
        auto m_prime_exp = exp2f(m_prime_id - mi_id);
        // 计算2的(m_prime_id - mi_id)次幂
        out_rescale[id] = m_prime_exp;
        // 将计算结果存入out_rescale数组中的id位置
        s_prime[id] *= m_prime_exp;
        // 更新s_prime数组中id位置的值
      } else {
        // 如果changed为false
        // 只有在启用偏置时，注意力的第一个值可能全部被掩盖为`-inf`。
        // 在这种情况下，我们要避免`nan = exp2f(-inf - (-inf))`，因此我们临时将`mi`设置为0
        if (kSupportsBias &&
            mi_id == -cutlass::platform::numeric_limits<accum_t>::infinity()) {
          // 恢复mi为-inf的标志设置为true
          restore_mi_to_minus_inf = true;
          // 将mi数组中id位置的值设置为0.0f
          mi[id] = 0.0f;
        }
        // 将1.0f存入out_rescale数组中的id位置
        out_rescale[id] = 1.0f;
      }
    }
    __syncthreads(); // 更新输出片段
    if (kKeepOutputInRF && !is_first) {
      // 如果需要保留输出在寄存器文件中且不是第一个
      accum_t line_rescale;
      // 声明线性缩放因子line_rescale
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { line_rescale = out_rescale[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag_o[idx] = frag_o[idx] * line_rescale;
          },
          [&](int accum_m) {});
      // 迭代每一行数据，应用线性缩放因子到frag_o数组中的每个元素
    }
    // 更新accum_m, accum_n, ...
    {
      accum_t mi_row, total_row;
      // 声明mi_row和total_row
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { mi_row = mi[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag[idx] =
                (accum_n < max_col) ? exp2f(frag[idx] - mi_row) : accum_t(0.0);
          },
          [&](int accum_m) {});
      // 迭代每一行数据，更新frag数组中的元素
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { total_row = 0.0; },
          [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
          [&](int accum_m) {
            if (LambdaIterator::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              // 注意：我们可以原子地将`total_row`添加到`s_prime`中，但在这里避免原子操作更快（而且确定性更好）
              // 将total_row添加到addition_storage中的指定位置
              addition_storage
                  [accum_m + kQueriesPerBlock * tile_offset.column()] =
                      total_row;
            }
          });
      // 迭代每一行数据，累加frag中的元素到total_row，并将结果存入addition_storage数组中
    }
    __syncthreads();
    if (lane_id < kLinesPerWarp) {
      // 如果当前线程在每个warp中的lane_id小于kLinesPerWarp
      int id = warp_id * kLinesPerWarp + lane_id;
      // 计算当前线程的全局id
      accum_t total_row = s_prime[id];
      // 获取s_prime数组中id位置的值，存入total_row
      if (restore_mi_to_minus_inf) {
        // 如果需要恢复mi到-inf
        // 恢复`mi`，见上文我们设置了`restore_mi_to_minus_inf=true`
        mi[id] = -cutlass::platform::numeric_limits<accum_t>::infinity();
      } else {
        // 否则将mi数组中id位置的值赋给m_prime数组中id位置的值
        m_prime[id] = mi[id];
      }
      // 循环执行以下操作WarpCount::kN次
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < MM0::MmaCore::WarpCount::kN; ++i) {
        // 将addition_storage中指定位置的值添加到total_row中
        total_row += addition_storage[id + kQueriesPerBlock * i];
      }
      // 将total_row存入s_prime数组中id位置
      s_prime[id] = total_row;
    }
  }

  static CUTLASS_DEVICE int8_t lane_id() {
    return threadIdx.x;
  }


    # 返回当前线程在其块中的 X 索引
    static CUTLASS_DEVICE int8_t warp_id() {


    # 返回当前线程在其块中的 Y 索引
    return threadIdx.y;
  }


    # 返回当前线程的全局唯一标识符，由 X 和 Y 索引组合而成
    static CUTLASS_DEVICE int16_t thread_id() {
};

// 结束了前面未完结的匿名命名空间

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    // 定义了一个 GPU 核函数，使用 AK 类型的参数 p
    attention_kernel_batched_impl(typename AK::Params p) {
  // 如果无法推进到下一个块，则返回
  if (!p.advance_to_block()) {
    return;
  }
  // 调用 AK 类的 attention_kernel 方法，处理参数 p
  AK::attention_kernel(p);
}

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    // 声明了一个 GPU 核函数，使用 AK 类型的参数 params
    attention_kernel_batched(typename AK::Params params);

// 结束了命名空间 PyTorchMemEffAttention
} // namespace PyTorchMemEffAttention
```