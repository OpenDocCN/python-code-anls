# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\threadblock\default_dq_mma_multistage.h`

```py
#pragma once

#include <cutlass/gemm/threadblock/default_mma.h>
#include <ATen/native/cuda/cutlass_extensions/arch/mma.h>

#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/dq_mma_multistage.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/warp/default_mma_tensor_op.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h>
#include <ATen/native/cuda/cutlass_extensions/tile_interleaved_layout.h>

#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template<
    /// Type for elementA
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Stages in GEMM
    int kStages,
    ///
    typename Operator,
    ///
    SharedMemoryClearOption SharedMemoryClear>
struct DqMma<ElementA,
             LayoutA,
             kAlignmentA,
             ElementB,
             LayoutB,
             kAlignmentB,
             ElementScale,
             LayoutScale,
             kAlignmentScale,
             ElementAccumulator,
             layout::RowMajor,
             OperatorClass,
             ArchTag,
             ThreadblockShape,
             WarpShape,
             InstructionShape,
             kStages,
             Operator,
             SharedMemoryClear,
             typename platform::enable_if<(ArchTag::kMinComputeCapability >= 80)>::type> {

    // 验证 ElementA 必须是 fp16 或 bf16 类型
    static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                  "Element A must be fp16 or bf16");

    // 验证 Operator 必须是 OpMultiplyAddDequantizeInterleavedBToA 类型
    static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
                  "Mma multistage must dequantize after ldsm");

    // 验证 ElementB 必须是 uint8 或 uint4 类型
    static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                  "Element B must be uint8 or uint4");
    // 根据 ElementA 的大小和对齐方式判断缓存操作类型，如果大小乘以对齐方式等于 128，则为 Global，否则为 Always
    static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    // 根据 ElementB 的大小和对齐方式判断缓存操作类型，如果大小乘以对齐方式等于 128，则为 Global，否则为 Always
    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    // 定义 MmaCore 组件
    // MmaCore 不依赖于阶段，因此至少传入 3，以便创建多阶段的 Mma 核心部件
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        ElementA,
                                                                        LayoutA,
                                                                        ElementB,
                                                                        LayoutB,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        OperatorClass,
                                                                        std::max(kStages, 3),
                                                                        Operator,
                                                                        false,
                                                                        CacheOpA,
                                                                        CacheOpB>;

    // 定义 A 操作数瓦片的迭代器
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        1,
        ThreadMapA,
        AccessTypeA>;

    // 定义 B 操作数瓦片的迭代器
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        0,
        ThreadMapB,
        AccessTypeB>;

    // 用于比例迭代器的 ThreadMap
    // 静态断言，确保MmaCore::Shape::kN能够被kAlignmentScale整除，否则会出现编译错误
    static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");

    // 定义一个类型别名，表示用于线程映射的迭代器类型
    using IteratorScaleThreadMap =
        transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                                                  MmaCore::Shape::kN / kAlignmentScale,
                                                  kAlignmentScale>;

    // 定义从缩放操作数中迭代出瓦片的迭代器类型
    using IteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                ElementScale,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;

    // 使用IteratorScale作为SmemIteratorScale的类型别名
    using SmemIteratorScale = IteratorScale;

    // 定义快速交织和偏置数值数组转换器的类型别名
    using Converter = FastInterleavedAndBiasedNumericArrayConverter<ElementA,
                                                                    ElementB,
                                                                    MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

    // 定义线程块范围内的流水线矩阵乘法类型
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape,
                                                                       IteratorA,
                                                                       typename MmaCore::SmemIteratorA,
                                                                       MmaCore::kCacheOpA,
                                                                       IteratorB,
                                                                       typename MmaCore::SmemIteratorB,
                                                                       MmaCore::kCacheOpB,
                                                                       IteratorScale,
                                                                       SmemIteratorScale,
                                                                       ElementAccumulator,
                                                                       layout::RowMajor,
                                                                       typename MmaCore::MmaPolicy,
                                                                       kStages,
                                                                       Converter,
                                                                       SharedMemoryClear>;
    };

    template<
        /// Type for element A
        typename ElementA,
        /// Layout type for A matrix operand
        typename LayoutA,
        /// Access granularity of A matrix in units of elements
        int kAlignmentA,
        /// Type for element B
        typename ElementB,
        /// Access granularity of B matrix in units of elements
        int kAlignmentB,
        /// Element type for the input scale
        typename ElementScale,
        /// Layout for the scale operand
        typename LayoutScale,
        /// Access granularity of Scales in unit of elements
        int kAlignmentScale,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Operator class tag
        typename OperatorClass,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: GemmShape)
        typename InstructionShape,
        /// Stages in GEMM
        int kStages,
        ///
        typename Operator,
        ///
        SharedMemoryClearOption SharedMemoryClear,
        ///
        int RowsPerTile,
        ///
        int ColumnsInterleaved>
    struct DqMma<ElementA,
                 LayoutA,
                 kAlignmentA,
                 ElementB,
                 layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>,
                 kAlignmentB,
                 ElementScale,
                 LayoutScale,
                 kAlignmentScale,
                 ElementAccumulator,
                 layout::RowMajor,
                 OperatorClass,
                 ArchTag,
                 ThreadblockShape,
                 WarpShape,
                 InstructionShape,
                 kStages,
                 Operator,
                 SharedMemoryClear,
                 typename platform::enable_if<(ArchTag::kMinComputeCapability >= 80)>::type> {

        static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                      "Element A must be fp16 or bf16");

        static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
                      "Mma multistage must dequantize after ldsm");

        static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                      "Element B must be uint8 or uint4");

        // Determine caching operation based on ElementA's size and alignment
        static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128) ?
                                                                        cutlass::arch::CacheOperation::Global :
                                                                        cutlass::arch::CacheOperation::Always;



        // Static assertion to ensure ElementA is either half_t or bfloat16_t
        static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                      "Element A must be fp16 or bf16");

        // Static assertion to ensure Operator is arch::OpMultiplyAddDequantizeInterleavedBToA
        static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
                      "Mma multistage must dequantize after ldsm");

        // Static assertion to ensure ElementB is either uint8_t or uint4b_t
        static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                      "Element B must be uint8 or uint4");

        // Determine caching operation kind for ElementA based on its size and alignment
        static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128) ?
                                                                        cutlass::arch::CacheOperation::Global :
                                                                        cutlass::arch::CacheOperation::Always;
    // 定义静态常量 CacheOpB，根据 ElementB 的大小和对齐方式判断缓存操作类型
    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128) ?
                                                                    cutlass::arch::CacheOperation::Global :
                                                                    cutlass::arch::CacheOperation::Always;

    // 定义 MmaCore 组件
    // MmaCore 不依赖于阶段，因此至少传入 3 以确保创建多阶段的 mma 核心组件
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        ElementA,
                                                                        LayoutA,
                                                                        ElementB,
                                                                        layout::ColumnMajor,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        OperatorClass,
                                                                        std::max(kStages, 3),
                                                                        Operator,
                                                                        false,
                                                                        CacheOpA,
                                                                        CacheOpB>;

    // 定义 A 操作数的迭代器
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;  // 使用 MmaCore 的 A 操作数迭代器线程映射
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>; // 使用 ElementA 和 kAlignmentA 定义访问类型
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,  // 定义矩阵形状
        ElementA,  // A 操作数元素类型
        LayoutA,   // A 操作数布局
        1,         // 访问迭代器步长
        ThreadMapA,  // A 操作数迭代器线程映射类型
        AccessTypeA>;  // 访问类型为预测瓦片访问迭代器
private:
    // 确保 MmaCore::Shape::kN 能够整除 ColumnsInterleaved，用于后续计算和布局的正确性
    static_assert(!(MmaCore::Shape::kN % ColumnsInterleaved), "");
    // 确保 RowsPerTile 等于 MmaCore::Shape::kK，这是矩阵乘法中的必要条件
    static_assert(RowsPerTile == MmaCore::Shape::kK, "");

    // 定义原始的线程映射和瓦片排列
    using OriginalThreadMap       = typename MmaCore::IteratorThreadMapB;
    using OriginalWarpArrangement = typename OriginalThreadMap::Detail::WarpThreadArrangement;
    // 确保原始瓦片排列的 Strided 维度能够整除 ColumnsInterleaved，以确保数据访问的正确性
    static_assert(!(OriginalWarpArrangement::kStrided % ColumnsInterleaved), "");

    // 定义全局内存（Global Memory）迭代器形状，用于访问 B 操作数的数据
    using GmemIteratorShape =
        MatrixShape<MmaCore::Shape::kK * ColumnsInterleaved, MmaCore::Shape::kN / ColumnsInterleaved>;
    // 使用线程映射和原始瓦片排列的数据结构，定义全局内存迭代器的线程映射
    using GmemThreadMapB = transform::PitchLinearWarpRakedThreadMap<
        layout::PitchLinearShape<GmemIteratorShape::kRow, GmemIteratorShape::kColumn>,
        OriginalThreadMap::kThreads,
        layout::PitchLinearShape<OriginalWarpArrangement::kContiguous * ColumnsInterleaved,
                                 OriginalWarpArrangement::kStrided / ColumnsInterleaved>,
        MmaCore::kAccessSizeInBits / sizeof_bits<ElementB>::value>;

public:
    // 定义 B 操作数的迭代器线程映射，用于线程块操作
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    // 定义 B 操作数的访问类型，使用预测的瓦片访问迭代器
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::
        PredicatedTileAccessIterator<GmemIteratorShape, ElementB, layout::ColumnMajor, 0, GmemThreadMapB, AccessTypeB>;

    // 定义用于规模迭代器的线程映射
    static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
    // 使用条带分割的线程映射定义规模迭代器，用于规模操作数的访问
    using IteratorScaleThreadMap =
        transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                                                  MmaCore::Shape::kN / kAlignmentScale,
                                                  kAlignmentScale>;

    // 定义用于规模操作数的迭代器
    using IteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                ElementScale,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;

    // 使用规模迭代器定义共享内存规模迭代器
    using SmemIteratorScale = IteratorScale;

    // 定义快速交织和偏置的数值数组转换器，用于 A 和 B 操作数的转换
    using Converter = FastInterleavedAndBiasedNumericArrayConverter<ElementA,
                                                                    ElementB,
                                                                    MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

    // 定义线程块范围的流水线矩阵乘法
    // 使用 ThreadblockMma 类型别名定义一个多阶段整数量化乘法矩阵运算的模板实例化
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<
        typename MmaCore::Shape,                         // 矩阵乘法核心的形状
        IteratorA,                                       // A 矩阵的迭代器类型
        typename MmaCore::SmemIteratorA,                 // A 矩阵共享内存迭代器类型
        MmaCore::kCacheOpA,                              // A 矩阵的缓存操作类型
        IteratorB,                                       // B 矩阵的迭代器类型
        typename MmaCore::SmemIteratorB,                 // B 矩阵共享内存迭代器类型
        MmaCore::kCacheOpB,                              // B 矩阵的缓存操作类型
        IteratorScale,                                   // 尺度因子的迭代器类型
        SmemIteratorScale,                               // 尺度因子共享内存迭代器类型
        ElementAccumulator,                              // 元素累加器类型
        layout::RowMajor,                                // 矩阵布局类型（行主序）
        typename MmaCore::MmaPolicy,                     // 矩阵乘法核心的策略类型
        kStages,                                         // 阶段数
        Converter,                                       // 数据类型转换器
        SharedMemoryClear                                // 共享内存清除策略
    >;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
```