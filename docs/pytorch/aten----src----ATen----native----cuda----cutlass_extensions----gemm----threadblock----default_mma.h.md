# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\threadblock\default_mma.h`

```
#pragma once

#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma_multistage.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma_pipelined.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_mma_bf16.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp), fp16 activation & int8 weight
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
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
struct DefaultMma<cutlass::half_t,
                  LayoutA,
                  kAlignmentA,
                  uint8_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator> {

private:
    // Compute the alignment scale factor based on the size of half_t
    static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;

    // Define the specific MMA (Mixed Matrix Accumulation) operation type
    using Mma = DqMma<half_t,
                      LayoutA,
                      kAlignmentA,
                      uint8_t,
                      LayoutB,
                      kAlignmentB,
                      half_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      2,
                      Operator>;

public:
    // Define the core MMA (Mixed Matrix Accumulation) components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply operation
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output (OperatorClass TensorOp), fp16 activation & int4 weight
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
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
struct DefaultMma<cutlass::half_t,
                  LayoutA,
                  kAlignmentA,
                  uint4b_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator> {

private:
    /// Compute alignment scale based on half_t size to 128 bits
    static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;

    /// Define specialized DQ (Dot Product of Quaternion) MMA (Matrix Multiply-Add)
    using Mma = DqMma<half_t,
                      LayoutA,
                      kAlignmentA,
                      uint4b_t,
                      LayoutB,
                      kAlignmentB,
                      half_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      2,
                      Operator>;

public:
    /// Define the core MMA operation type used in this specialization
    using MmaCore = typename Mma::MmaCore;

    /// Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    /// Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    /// Define the threadblock-scoped pipelined matrix multiply operation
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Number of stages in the computation loop
    int kStages,
    /// Shared memory clear option (enum type)
    SharedMemoryClearOption SharedMemoryClear>
// 定义一个结构体模板，用于描述默认的矩阵乘法运算器，适用于半精度浮点数和8位无符号整数的组合
struct DefaultMma<cutlass::half_t,                  // A矩阵元素类型为半精度浮点数
                  LayoutA,                           // A矩阵布局类型
                  kAlignmentA,                       // A矩阵访存对齐方式
                  uint8_t,                           // B矩阵元素类型为8位无符号整数
                  LayoutB,                           // B矩阵布局类型
                  kAlignmentB,                       // B矩阵访存对齐方式
                  ElementAccumulator,                // 累加器元素类型
                  layout::RowMajor,                  // 输出矩阵的布局类型为行主序
                  arch::OpClassTensorOp,             // 操作类别为Tensor操作
                  ArchTag,                           // 架构标签，用于优化
                  ThreadblockShape,                  // 线程块大小的类型
                  WarpShape,                         // warp大小的类型
                  InstructionShape,                  // 指令级别大小的类型
                  kStages,                           // GEMM操作的阶数
                  Operator,                          // 指定的GEMM操作类型
                  false,                             // 是否清空共享内存选项为false
                  SharedMemoryClear> {               // 共享内存清理选项

private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;  // 计算半精度浮点数对齐的比例

    // 使用DqMma模板定义Mma类型，指定各种模板参数
    using Mma = DqMma<half_t,
                      LayoutA,
                      kAlignmentA,
                      uint8_t,
                      LayoutB,
                      kAlignmentB,
                      half_t,
                      layout::RowMajor,
                      kAlignmentScale,
                      ElementAccumulator,
                      layout::RowMajor,
                      arch::OpClassTensorOp,
                      ArchTag,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      kStages,
                      Operator,
                      SharedMemoryClear>;

public:
    // 定义MmaCore组件类型，从Mma类型中提取
    using MmaCore = typename Mma::MmaCore;

    // 定义A操作数的迭代器类型，从Mma类型中提取
    using IteratorA = typename Mma::IteratorA;

    // 定义B操作数的迭代器类型，从Mma类型中提取
    using IteratorB = typename Mma::IteratorB;

    // 定义线程块级别的流水线矩阵乘法运算类型，从Mma类型中提取
    using ThreadblockMma = typename Mma::ThreadblockMma;
};
// 定义一个模板结构体 DefaultMma，用于多种类型和常量参数化的矩阵乘法运算，默认特化
struct DefaultMma<cutlass::half_t,  // 使用 half_t 类型作为矩阵元素类型
                  LayoutA,           // A 矩阵的布局类型
                  kAlignmentA,       // A 矩阵的对齐粒度
                  uint4b_t,          // B 矩阵元素类型（通常是压缩表示）
                  LayoutB,           // B 矩阵的布局类型
                  kAlignmentB,       // B 矩阵的对齐粒度
                  ElementAccumulator,// 累加器元素类型
                  layout::RowMajor,  // A 矩阵的行主序列
                  arch::OpClassTensorOp, // 使用张量核心操作类别
                  ArchTag,           // 架构标签（例如 Ampere）
                  ThreadblockShape,  // 线程块级别的矩阵乘法形状
                  WarpShape,         // 线程束级别的矩阵乘法形状
                  InstructionShape,  // 指令级别的矩阵乘法形状
                  kStages,           // 多级乘法器阶段数
                  Operator,          // GEMM 运算操作符
                  false,             // 不执行共享内存清除
                  SharedMemoryClear> // 共享内存清除选项
{
private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;  // 计算对齐缩放因子

    using Mma = DqMma<half_t,           // 使用 half_t 作为操作数类型
                      LayoutA,          // A 矩阵布局类型
                      kAlignmentA,      // A 矩阵对齐粒度
                      uint4b_t,         // B 矩阵元素类型（通常是压缩表示）
                      LayoutB,          // B 矩阵布局类型
                      kAlignmentB,      // B 矩阵对齐粒度
                      half_t,           // 累加器元素类型为 half_t
                      layout::RowMajor, // A 矩阵行主序列
                      kAlignmentScale,  // 计算后的对齐缩放值
                      ElementAccumulator,  // 累加器元素类型
                      layout::RowMajor, // 累加器行主序列
                      arch::OpClassTensorOp,  // 使用张量核心操作类别
                      ArchTag,          // 架构标签
                      ThreadblockShape, // 线程块级别的矩阵乘法形状
                      WarpShape,        // 线程束级别的矩阵乘法形状
                      InstructionShape, // 指令级别的矩阵乘法形状
                      kStages,          // 多级乘法器阶段数
                      Operator,         // GEMM 运算操作符
                      SharedMemoryClear>;  // 共享内存清除选项

public:
    // 定义 MmaCore 组件类型
    using MmaCore = typename Mma::MmaCore;

    // 定义 A 操作数的迭代器类型
    using IteratorA = typename Mma::IteratorA;

    // 定义 B 操作数的迭代器类型
    using IteratorB = typename Mma::IteratorB;

    // 定义线程块范围的流水线矩阵乘法
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

// 在 Ampere 架构上，针对 fp16 x fp16 进行特化以使用 2 级多级乘法器。有助于在共享内存不足以进行 3 级或更多级别时避免寄存器溢出。
template<
    /// A 矩阵操作数的布局类型
    typename LayoutA,
    /// A 矩阵的元素访问粒度（以元素为单位）
    int kAlignmentA,
    /// B 矩阵操作数的布局类型
    typename LayoutB,
    /// B 矩阵的元素访问粒度（以元素为单位）
    int kAlignmentB,
    /// 内部累加器的元素类型
    typename ElementAccumulator,
    /// 线程块级别的矩阵乘法形状（概念：GemmShape）
    typename ThreadblockShape,
    /// 线程束级别的矩阵乘法形状（概念：GemmShape）
    typename WarpShape,
    /// 指令级别的矩阵乘法形状（概念：GemmShape）
    typename InstructionShape,
    /// GEMM 运算执行的操作
    typename Operator,
    /// 使用 zfill 或谓词处理超出边界的 cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// 通过使用索引数组收集 A 操作数
    bool GatherA,
    /// 通过使用索引数组收集 B 操作数
    bool GatherB>
// 定义模板结构体 DefaultMma，使用半精度浮点类型 half_t 进行矩阵乘法计算
// 其中模板参数包括两个矩阵的布局 LayoutA 和 LayoutB，以及对齐方式 kAlignmentA 和 kAlignmentB
// 还有计算过程中需要的累加器类型 ElementAccumulator，矩阵布局类型 layout::RowMajor
// 使用的张量操作类 OpClassTensorOp 和架构类型 arch::Sm80
// 以及线程块形状 ThreadblockShape、warp 形状 WarpShape 和指令形状 InstructionShape
// Operator 为乘法操作的阶数，SharedMemoryClear 表示是否清空共享内存，GatherA 和 GatherB 为数据收集标志
struct DefaultMma<half_t,
                  LayoutA,
                  kAlignmentA,
                  half_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  arch::Sm80,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator,
                  false,
                  SharedMemoryClear,
                  GatherA,
                  GatherB> {

    // 定义 MmaCore 组件，通过 DefaultMmaCore 类来生成多级 MMA 核心计算组件
    // 3 是故意选定的值，用于触发多级 MMA 计算组件
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        half_t,
                                                                        LayoutA,
                                                                        half_t,
                                                                        LayoutB,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        arch::OpClassTensorOp,
                                                                        3,
                                                                        Operator>;

    // 定义 A 操作数的迭代器，访问 A 矩阵的小块（tile）
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<half_t, kAlignmentA>;
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        half_t,
        LayoutA,
        1,
        ThreadMapA,
        AccessTypeA,
        GatherA>;

    // 定义 B 操作数的迭代器，访问 B 矩阵的小块（tile）
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<half_t, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        half_t,
        LayoutB,
        0,
        ThreadMapB,
        AccessTypeB,
        GatherB>;

    // 定义线程块级别的多级矩阵乘法计算
    # 使用 ThreadblockMma 别名来定义一个具有多阶段操作的矩阵乘法运算单元
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
        typename MmaCore::Shape,                           // 使用 MmaCore 定义的形状类型
        IteratorA,                                         // 迭代器类型 A
        typename MmaCore::SmemIteratorA,                   // 使用 MmaCore 定义的共享内存迭代器类型 A
        MmaCore::kCacheOpA,                                // MmaCore 定义的缓存操作类型 A
        IteratorB,                                         // 迭代器类型 B
        typename MmaCore::SmemIteratorB,                   // 使用 MmaCore 定义的共享内存迭代器类型 B
        MmaCore::kCacheOpB,                                // MmaCore 定义的缓存操作类型 B
        ElementAccumulator,                                // 元素累加器类型
        layout::RowMajor,                                  // 行优先存储布局
        typename MmaCore::MmaPolicy,                       // 使用 MmaCore 定义的乘法累加策略类型
        2>;                                                // 多阶段操作的阶段数为 2
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass


注释：


// 结束 threadblock 命名空间
};

// 结束 gemm 命名空间
}  // namespace threadblock

// 结束 cutlass 命名空间
}  // namespace gemm
}  // namespace cutlass


这段代码是 C++ 中的命名空间闭合语句。在 C++ 中，命名空间用于避免全局命名冲突，将一组相关的声明（变量、函数、类等）封装在一个作用域中。在这里，`threadblock`、`gemm` 和 `cutlass` 都是命名空间。代码的作用是结束这些命名空间的定义，确保命名空间的作用范围正确闭合。
```