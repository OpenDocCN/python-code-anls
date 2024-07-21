# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\threadblock\default_dq_mma_pipelined.h`

```py
#pragma once

#include <cutlass/gemm/threadblock/default_mma.h>
#include <ATen/native/cuda/cutlass_extensions/arch/mma.h>

#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/dq_mma_pipelined.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/warp/default_mma_tensor_op.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h>
#include <ATen/native/cuda/cutlass_extensions/tile_interleaved_layout.h>

#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

// 定义一个模板结构体 DqMma，用于实现 DQ（Dynamic Quantization）操作的 MMA（Mixed Matrix Multiplication）。
template<
    /// Type for element A
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
    /// Operation performed by GEMM
    typename Operator>
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
             2,  // Use 2 pipelines for DQ MMA
             Operator,
             SharedMemoryClearOption::kNone,
             typename platform::enable_if<(ArchTag::kMinComputeCapability < 80)>::type> {

    // 静态断言：要求 ElementA 必须是 fp16 或 bf16 类型
    static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                  "Element A must be fp16 or bf16");

    // 静态断言：要求 ElementB 必须是 uint8 或 uint4 类型
    static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                  "Element B must be uint8 or uint4");

    // 是否在 LDG 操作后进行 DQ 操作，根据 Operator 类型决定
    static constexpr bool DqAfterLDG        = platform::is_same<arch::OpMultiplyAdd, Operator>::value;
    
    // 是否支持 BF16 MMA 操作，根据架构标签 ArchTag 决定
    static constexpr bool arch_has_bf16_mma = ArchTag::kMinComputeCapability >= 80;


这段代码定义了一个模板结构体 `DqMma`，用于实现动态量化（DQ）的 MMA 操作。它包括多个模板参数，静态断言来确保输入类型的正确性，以及布尔值常量来确定支持的特定功能。
    // 定义 MmaCoreElementA 类型为 ElementA 或 half_t，取决于 arch_has_bf16_mma 的条件
    using MmaCoreElementA = typename platform::conditional<arch_has_bf16_mma, ElementA, half_t>::type;
    
    // 定义 MmaCoreElementB 类型为 MmaCoreElementA 或 ElementB，取决于 DqAfterLDG 的条件
    using MmaCoreElementB = typename platform::conditional<DqAfterLDG, MmaCoreElementA, ElementB>::type;
    
    // 定义 MmaCore 组件，用于执行 GEMM 的核心运算
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        MmaCoreElementA,
                                                                        LayoutA,
                                                                        MmaCoreElementB,
                                                                        LayoutB,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        OperatorClass,
                                                                        2,
                                                                        Operator>;
    
    // 定义 A 操作数的迭代器，遍历以矩阵形式排列的数据块
    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
        ElementA,
        LayoutA,
        1,
        typename MmaCore::IteratorThreadMapA,
        kAlignmentA>;
    
    // 定义 B 操作数的迭代器，遍历以矩阵形式排列的数据块
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
        ElementB,
        LayoutB,
        0,
        typename MmaCore::IteratorThreadMapB,
        kAlignmentB>;
    
    // 用于缩放迭代器的 ThreadMap，以确保对齐和分块访问
    static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
    using IteratorScaleThreadMap =
        transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                                                  MmaCore::Shape::kN / kAlignmentScale,
                                                  kAlignmentScale>;
    
    // 定义缩放操作数的迭代器，以列主序方式排列
    using IteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                ElementScale,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;
    // 使用模板元编程根据条件选择数据类型，若硬件支持 bf16_mma 则选择 ElementScale，否则选择 half_t
    using SmemScaleType = typename platform::conditional<arch_has_bf16_mma, ElementScale, half_t>::type;

    // 定义共享内存迭代器类型，用于访问 Threadblock 中的矩阵块，支持数据类型为 SmemScaleType
    using SmemIteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                SmemScaleType,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;

    // 定义转换器集合类型，用于数据类型转换，包括 IteratorB 到 MmaCore::MmaPolicy::Operator 和 Operator
    using Converters = SetConverters<IteratorB, typename MmaCore::MmaPolicy::Operator, Operator>;

    // 定义基于线程块的流水线矩阵乘法，使用指定的模板参数配置
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaPipelined<typename MmaCore::Shape,  // 矩阵形状
                                                                      IteratorA,  // 迭代器 A
                                                                      typename MmaCore::SmemIteratorA,  // 共享内存迭代器 A
                                                                      IteratorB,  // 迭代器 B
                                                                      typename MmaCore::SmemIteratorB,  // 共享内存迭代器 B
                                                                      IteratorScale,  // 迭代器 Scale
                                                                      SmemIteratorScale,  // 共享内存迭代器 Scale
                                                                      ElementAccumulator,  // 元素累加器
                                                                      layout::RowMajor,  // 行主序布局
                                                                      typename MmaCore::MmaPolicy,  // Mma 策略
                                                                      typename Converters::TransformAfterLDG,  // LDG 后的转换器
                                                                      typename Converters::TransformAfterLDS>;  // LDS 后的转换器
};

// Specialization to handle column major interleave B
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
    /// Layout for matrix B (specialized for column-major tile interleave)
    typename layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>,
    /// Alignment for matrix B
    kAlignmentB,
    /// Element type for scale
    typename ElementScale,
    /// Layout for scale
    typename LayoutScale,
    /// Alignment for scale
    kAlignmentScale,
    /// Element type for accumulation
    typename ElementAccumulator,
    /// Row-major layout for matrix C
    layout::RowMajor,
    /// Operator class
    typename OperatorClass,
    /// Architecture tag for tuning
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Number of rows per tile
    int RowsPerTile,
    /// Number of columns interleaved per tile
    int ColumnsInterleaved>
struct DqMma<
    ElementA,
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
    2,
    Operator,
    SharedMemoryClearOption::kNone,
    typename platform::enable_if<(ArchTag::kMinComputeCapability < 80)>::type
> {

    // Assertion for ElementA type, must be half_t or bfloat16_t
    static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                  "Element A must be fp16 or bf16");

    // Assertion for ElementB type, must be uint8_t or uint4b_t
    static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                  "Element B must be uint8 or uint4");

    // Check if DqAfterLDG is true when Operator is arch::OpMultiplyAdd
    static constexpr bool DqAfterLDG = platform::is_same<arch::OpMultiplyAdd, Operator>::value;

    // Check if the architecture supports bf16 MMA operations
    static constexpr bool arch_has_bf16_mma = ArchTag::kMinComputeCapability >= 80;

    // Define MmaCoreElementA based on architecture support for bf16 MMA
    using MmaCoreElementA = typename platform::conditional<arch_has_bf16_mma, ElementA, half_t>::type;

    // Define MmaCoreElementB based on DqAfterLDG condition
    using MmaCoreElementB = typename platform::conditional<DqAfterLDG, MmaCoreElementA, ElementB>::type;

    // Define the MmaCore components
    # 使用别名 MmaCore 定义一个模板类型，代表了一个特定的 GEMM 核心实现
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        MmaCoreElementA,
                                                                        LayoutA,
                                                                        MmaCoreElementB,
                                                                        layout::ColumnMajor,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        OperatorClass,
                                                                        2,
                                                                        Operator>;

    # 定义 A 操作数的瓦片迭代器类型，用于迭代 A 矩阵的瓦片
    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,  # 瓦片的形状（M, K）
        ElementA,  # A 元素的类型
        LayoutA,   # A 矩阵的布局
        1,         # 步长为1，即每次迭代移动一个元素
        typename MmaCore::IteratorThreadMapA,  # 迭代器线程映射类型，根据 MmaCore 的 A 矩阵迭代器线程映射类型进行定义
        kAlignmentA>  # A 矩阵元素的对齐要求
private:
    // 确保 MmaCore::Shape::kN 能够被 ColumnsInterleaved 整除
    static_assert(!(MmaCore::Shape::kN % ColumnsInterleaved), "");
    // 确保 RowsPerTile 等于 MmaCore::Shape::kK
    static_assert(RowsPerTile == MmaCore::Shape::kK, "");

    // 使用 MmaCore::IteratorThreadMapB 作为 OriginalThreadMap
    using OriginalThreadMap = typename MmaCore::IteratorThreadMapB;
    // 获取 OriginalThreadMap 的 WarpThreadArrangement 类型
    using OriginalWarpArrangement = typename OriginalThreadMap::Detail::WarpThreadArrangement;
    // 确保 OriginalWarpArrangement::kStrided 能够被 ColumnsInterleaved 整除
    static_assert(!(OriginalWarpArrangement::kStrided % ColumnsInterleaved), "");

    // 定义 GmemIteratorShape，表示全局内存迭代器的形状
    using GmemIteratorShape =
        MatrixShape<MmaCore::Shape::kK * ColumnsInterleaved, MmaCore::Shape::kN / ColumnsInterleaved>;
    // 使用 PitchLinearWarpRakedThreadMap 定义 GmemThreadMapB
    using GmemThreadMapB = transform::PitchLinearWarpRakedThreadMap<
        layout::PitchLinearShape<GmemIteratorShape::kRow, GmemIteratorShape::kColumn>,
        OriginalThreadMap::kThreads,
        layout::PitchLinearShape<OriginalWarpArrangement::kContiguous * ColumnsInterleaved,
                                 OriginalWarpArrangement::kStrided / ColumnsInterleaved>,
        MmaCore::kAccessSizeInBits / sizeof_bits<ElementB>::value>;

public:
    // 定义 B 操作数的迭代器
    using IteratorB = cutlass::transform::threadblock::
        PredicatedTileIterator<GmemIteratorShape, ElementB, layout::ColumnMajor, 0, GmemThreadMapB, kAlignmentB>;

    // ThreadMap 用于 Scale 迭代器
    // 确保 MmaCore::Shape::kN 能够被 kAlignmentScale 整除
    static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
    // 定义 IteratorScaleThreadMap 作为 Scale 迭代器的 ThreadMap
    using IteratorScaleThreadMap =
        transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                                                  MmaCore::Shape::kN / kAlignmentScale,
                                                  kAlignmentScale>;

    // 定义 Scale 操作数的迭代器
    using IteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                ElementScale,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;

    // 定义 SmemScaleType 作为 SmemIteratorScale 的元素类型，根据 arch_has_bf16_mma 条件选择 ElementScale 或 half_t 类型
    using SmemScaleType = typename platform::conditional<arch_has_bf16_mma, ElementScale, half_t>::type;
    // 定义 SmemIteratorScale，表示共享内存中 Scale 的迭代器
    using SmemIteratorScale =
        cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                                                                SmemScaleType,
                                                                LayoutScale,
                                                                0,
                                                                IteratorScaleThreadMap,
                                                                kAlignmentScale>;

    // 定义 Converters，用于设置 B 迭代器和 MmaCore::MmaPolicy::Operator 之间的转换器
    using Converters = SetConverters<IteratorB, typename MmaCore::MmaPolicy::Operator, Operator>;

    // 定义线程块范围内的流水线矩阵乘法
    # 使用ThreadblockMma类型别名来定义一个Cutlass库中的特定模板类实例化
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaPipelined<
        typename MmaCore::Shape,                    // 使用MmaCore中的形状定义
        IteratorA,                                  // IteratorA类型参数
        typename MmaCore::SmemIteratorA,            // 使用MmaCore中的SmemIteratorA定义
        IteratorB,                                  // IteratorB类型参数
        typename MmaCore::SmemIteratorB,            // 使用MmaCore中的SmemIteratorB定义
        IteratorScale,                              // IteratorScale类型参数
        SmemIteratorScale,                          // SmemIteratorScale类型参数
        ElementAccumulator,                         // ElementAccumulator类型参数
        layout::RowMajor,                           // 使用行主要布局
        typename MmaCore::MmaPolicy,                // 使用MmaCore中的MmaPolicy定义
        typename Converters::TransformAfterLDG,     // 使用Converters中的TransformAfterLDG定义
        typename Converters::TransformAfterLDS      // 使用Converters中的TransformAfterLDS定义
    >;
};

// 结束 namespace threadblock
}  // namespace threadblock

// 结束 namespace gemm
}  // namespace gemm

// 结束 namespace cutlass
}  // namespace cutlass
```