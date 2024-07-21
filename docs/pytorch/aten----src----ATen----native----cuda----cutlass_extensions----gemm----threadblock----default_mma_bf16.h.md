# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\threadblock\default_mma_bf16.h`

```
#pragma once

#include <cutlass/gemm/threadblock/default_mma.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma_multistage.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_dq_mma_pipelined.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp), bf16 activation & bf16 weight
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
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB>
struct DefaultMma<bfloat16_t,
                  LayoutA,
                  kAlignmentA,
                  bfloat16_t,
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
                  Operator,
                  false,
                  SharedMemoryClear,
                  GatherA,
                  GatherB> {

private:
    // Conversions only needed pre-ampere. This will trigger mma pipeline, so we convert before STS.
    static constexpr bool arch_has_bf16_mma = ArchTag::kMinComputeCapability >= 80;
    // Define MmaElementA based on architecture support for bf16 or fallback to half_t
    using MmaElementA = typename platform::conditional<arch_has_bf16_mma, bfloat16_t, half_t>::type;
    // Define MmaElementB similarly based on architecture support
    using MmaElementB = typename platform::conditional<arch_has_bf16_mma, bfloat16_t, half_t>::type;

public:
    // Define the MmaCore components
    // 使用 cutlass 库中的 DefaultMmaCore 类型定义 MmaCore，表示默认的矩阵乘法核心
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        MmaElementA,
                                                                        LayoutA,
                                                                        MmaElementB,
                                                                        LayoutB,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        arch::OpClassTensorOp,
                                                                        2,
                                                                        Operator>;

    // 使用 cutlass 库中的 PredicatedTileIterator 类型定义 IteratorA，表示A操作数的迭代器
    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,  // 迭代器操作的矩阵形状
        bfloat16_t,  // 元素类型为 bfloat16
        LayoutA,  // A 矩阵的布局
        1,  // 迭代器访问策略（增量）
        typename MmaCore::IteratorThreadMapA,  // A 矩阵的迭代线程映射策略
        kAlignmentA,  // 对齐方式
        GatherA>;  // Gather 类型（用于数据获取）

    // 使用 cutlass 库中的 PredicatedTileIterator 类型定义 IteratorB，表示B操作数的迭代器
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,  // 迭代器操作的矩阵形状
        bfloat16_t,  // 元素类型为 bfloat16
        LayoutB,  // B 矩阵的布局
        0,  // 迭代器访问策略（增量）
        typename MmaCore::IteratorThreadMapB,  // B 矩阵的迭代线程映射策略
        kAlignmentB,  // 对齐方式
        GatherB>;  // Gather 类型（用于数据获取）

    // 使用 cutlass 库中的 MmaPipelined 类型定义 ThreadblockMma，表示线程块范围的流水线矩阵乘法
    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape,
                                                                    IteratorA,
                                                                    typename MmaCore::SmemIteratorA,
                                                                    IteratorB,
                                                                    typename MmaCore::SmemIteratorB,
                                                                    ElementAccumulator,
                                                                    layout::RowMajor,
                                                                    typename MmaCore::MmaPolicy>;
// 结构模板 DefaultMma 的部分特化，用于在Ampere架构上针对bf16 x bf16的情况进行优化，
// 使用多级别的mma操作以避免在没有足够共享内存的情况下进行寄存器溢出的大瓦片时使用3级以上阶段
template<
    /// A矩阵操作数的布局类型
    typename LayoutA,
    /// A矩阵的访问粒度，单位为元素数
    int kAlignmentA,
    /// B矩阵操作数的布局类型
    typename LayoutB,
    /// B矩阵的访问粒度，单位为元素数
    int kAlignmentB,
    /// 内部累加器的元素类型
    typename ElementAccumulator,
    /// 线程块级别的瓦片大小（概念：GemmShape）
    typename ThreadblockShape,
    /// 单元级别的瓦片大小（概念：GemmShape）
    typename WarpShape,
    /// 指令级别的瓦片大小（概念：GemmShape）
    typename InstructionShape,
    /// GEMM操作的类型
    typename Operator,
    /// 使用zfill或谓词进行越界cp.async的共享内存清除选项
    SharedMemoryClearOption SharedMemoryClear,
    /// 通过使用索引数组来进行A操作数的聚合
    bool GatherA,
    /// 通过使用索引数组来进行B操作数的聚合
    bool GatherB>
struct DefaultMma<bfloat16_t,
                  LayoutA,
                  kAlignmentA,
                  bfloat16_t,
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

    // 定义MmaCore组件
    // 这里故意使用3来触发适用于mma多级别的组件
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        bfloat16_t,
                                                                        LayoutA,
                                                                        bfloat16_t,
                                                                        LayoutB,
                                                                        ElementAccumulator,
                                                                        layout::RowMajor,
                                                                        arch::OpClassTensorOp,
                                                                        3,
                                                                        Operator>;

    // 定义A操作数的迭代器，遍历瓦片
    using ThreadMapA  = typename MmaCore::IteratorThreadMapA;
    // 定义模板别名，表示A操作数的内存访问类型和迭代器类型
    using AccessTypeA = cutlass::Array<bfloat16_t, kAlignmentA>;
    using IteratorA   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,  // 定义A操作数迭代器的矩阵形状
        bfloat16_t,                                                       // 操作数类型为bfloat16_t
        LayoutA,                                                          // A操作数的布局类型
        1,                                                                // 迭代器步长为1
        ThreadMapA,                                                       // 线程映射策略
        AccessTypeA,                                                      // 内存访问类型
        GatherA>;                                                         // Gather策略
    
    // 定义模板别名，表示B操作数的内存访问类型和迭代器类型
    using ThreadMapB  = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<bfloat16_t, kAlignmentB>;
    using IteratorB   = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,  // 定义B操作数迭代器的矩阵形状
        bfloat16_t,                                                       // 操作数类型为bfloat16_t
        LayoutB,                                                          // B操作数的布局类型
        0,                                                                // 迭代器步长为0
        ThreadMapB,                                                       // 线程映射策略
        AccessTypeB,                                                      // 内存访问类型
        GatherB>;                                                         // Gather策略
    
    // 定义线程块范围的多阶段矩阵乘法
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<typename MmaCore::Shape,        // 矩阵乘法核心的形状
                                                                     IteratorA,                       // A操作数的迭代器类型
                                                                     typename MmaCore::SmemIteratorA, // A操作数的共享内存迭代器类型
                                                                     MmaCore::kCacheOpA,              // A操作数的缓存操作类型
                                                                     IteratorB,                       // B操作数的迭代器类型
                                                                     typename MmaCore::SmemIteratorB, // B操作数的共享内存迭代器类型
                                                                     MmaCore::kCacheOpB,              // B操作数的缓存操作类型
                                                                     ElementAccumulator,              // 元素累加器类型
                                                                     layout::RowMajor,                // 输出矩阵的布局类型
                                                                     typename MmaCore::MmaPolicy,     // 矩阵乘法策略
                                                                     2>;                              // 多阶段矩阵乘法的阶段数
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp), bf16 activation & int8 weight
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
struct DefaultMma<cutlass::bfloat16_t,
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
    // Compute the alignment scale factor based on the size of bfloat16_t
    static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

    // Define the MMA operation specialization for bf16/bf16, int8 layout with row-major output
    using Mma = DqMma<bfloat16_t,
                      LayoutA,
                      kAlignmentA,
                      uint8_t,
                      LayoutB,
                      kAlignmentB,
                      bfloat16_t,
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
    // Define the type of the MMA core used in the specialization
    using MmaCore = typename Mma::MmaCore;

    // Define the iterator type for matrix A in the MMA operation
    using IteratorA = typename Mma::IteratorA;

    // Define the iterator type for matrix B in the MMA operation
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply operation
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output (OperatorClass TensorOp), bf16 activation & int4 weight
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,


注释：
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
struct DefaultMma<cutlass::bfloat16_t,
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
    // 计算对齐尺度
    static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

    // 使用 DqMma 模板定义 Mma 类型别名，配置模板参数
    using Mma = DqMma<bfloat16_t,
                      LayoutA,
                      kAlignmentA,
                      uint4b_t,
                      LayoutB,
                      kAlignmentB,
                      bfloat16_t,
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
    // 定义 MmaCore 组件类型别名，基于 Mma::MmaCore
    using MmaCore = typename Mma::MmaCore;

    // 定义 A 矩阵操作数的迭代器类型别名，基于 Mma::IteratorA
    using IteratorA = typename Mma::IteratorA;

    // 定义 B 矩阵操作数的迭代器类型别名，基于 Mma::IteratorB
    using IteratorB = typename Mma::IteratorB;

    // 定义线程块级别的流水线矩阵乘法类型别名，基于 Mma::ThreadblockMma
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

template<
    /// A 矩阵操作数的布局类型
    typename LayoutA,
    /// A 矩阵中元素的访问粒度（单位：元素）
    int kAlignmentA,
    /// B 矩阵操作数的布局类型
    typename LayoutB,
    /// B 矩阵中元素的访问粒度（单位：元素）
    int kAlignmentB,
    /// 内部累加器的元素类型
    typename ElementAccumulator,
    /// 指示用于调优的架构标签
    typename ArchTag,
    /// 线程块级别的矩阵乘法形状（概念：GemmShape）
    typename ThreadblockShape,
    /// 单元级别的矩阵乘法形状（概念：GemmShape）
    typename WarpShape,
    /// 指令级别的矩阵乘法形状（概念：GemmShape）
    typename InstructionShape,
    /// GEMM 执行的操作类型
    typename Operator,
    ///
    int kStages,
    /// 共享内存清除选项
    SharedMemoryClearOption SharedMemoryClear>
struct DefaultMma<cutlass::bfloat16_t,                           // 使用cutlass库中的bfloat16_t类型作为模板参数，定义了DefaultMma结构体
                  LayoutA,                                        // A矩阵操作数的布局类型模板参数
                  kAlignmentA,                                    // A矩阵的访问粒度，以元素为单位的对齐模板参数
                  uint8_t,                                        // B矩阵元素类型模板参数为uint8_t
                  LayoutB,                                        // B矩阵操作数的布局类型模板参数
                  kAlignmentB,                                    // B矩阵的访问粒度，以元素为单位的对齐模板参数
                  ElementAccumulator,                             // 内部累加器的元素类型模板参数
                  layout::RowMajor,                               // 输出结果的行主序布局
                  arch::OpClassTensorOp,                          // 表示操作类为Tensor操作的架构标签模板参数
                  ArchTag,                                        // 架构标签模板参数，用于优化调整
                  ThreadblockShape,                               // 线程块级矩阵乘法的形状（GemmShape的概念）
                  WarpShape,                                      // 线程束级矩阵乘法的形状（GemmShape的概念）
                  InstructionShape,                               // 指令级矩阵乘法的形状（GemmShape的概念）
                  kStages,                                        // GEMM操作的阶段数模板参数
                  Operator,                                       // GEMM操作的具体实现模板参数
                  false,                                          // 是否清空共享内存的选项，此处为false
                  SharedMemoryClear> {                            // 共享内存清空选项模板参数

private:
    static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;  // 计算对齐比例，以128字节为基准除以bfloat16_t类型的位数大小

    using Mma = DqMma<bfloat16_t,                                   // 使用DqMma类模板，指定bfloat16_t类型作为操作数和累加器类型
                      LayoutA,                                      // A矩阵布局类型模板参数
                      kAlignmentA,                                  // A矩阵的访问粒度，以元素为单位的对齐模板参数
                      uint8_t,                                      // B矩阵元素类型为uint8_t
                      LayoutB,                                      // B矩阵布局类型模板参数
                      kAlignmentB,                                  // B矩阵的访问粒度，以元素为单位的对齐模板参数
                      bfloat16_t,                                   // 累加器元素类型为bfloat16_t
                      layout::RowMajor,                             // 输出结果的行主序布局
                      kAlignmentScale,                              // 对齐比例模板参数
                      ElementAccumulator,                           // 内部累加器元素类型模板参数
                      layout::RowMajor,                             // 累加器布局类型模板参数
                      arch::OpClassTensorOp,                        // 表示操作类为Tensor操作的架构标签模板参数
                      ArchTag,                                      // 架构标签模板参数，用于优化调整
                      ThreadblockShape,                             // 线程块级矩阵乘法的形状（GemmShape的概念）
                      WarpShape,                                    // 线程束级矩阵乘法的形状（GemmShape的概念）
                      InstructionShape,                             // 指令级矩阵乘法的形状（GemmShape的概念）
                      kStages,                                      // GEMM操作的阶段数模板参数
                      Operator,                                     // GEMM操作的具体实现模板参数
                      SharedMemoryClear>;                           // 共享内存清空选项模板参数

public:
    // 定义MmaCore组件类型
    using MmaCore = typename Mma::MmaCore;

    // 定义对A操作数矩阵块的迭代器
    using IteratorA = typename Mma::IteratorA;

    // 定义对B操作数矩阵块的迭代器
    using IteratorB = typename Mma::IteratorB;

    // 定义线程块范围内的流水线矩阵乘法
    using ThreadblockMma = typename Mma::ThreadblockMma;
};
// 定义一个模板结构体 DefaultMma，用于描述深度神经网络中的默认矩阵乘法运算
struct DefaultMma<cutlass::bfloat16_t,                       // 数据类型为 bfloat16_t
                  LayoutA,                                    // A 操作数的布局
                  kAlignmentA,                                // A 操作数的对齐方式
                  uint4b_t,                                   // B 操作数的数据类型
                  LayoutB,                                    // B 操作数的布局
                  kAlignmentB,                                // B 操作数的对齐方式
                  ElementAccumulator,                         // 元素累加器的数据类型
                  layout::RowMajor,                           // 结果矩阵的布局
                  arch::OpClassTensorOp,                      // 硬件操作类别
                  ArchTag,                                    // 硬件架构标签
                  ThreadblockShape,                           // 线程块形状
                  WarpShape,                                  // 线程束形状
                  InstructionShape,                           // 指令形状
                  kStages,                                    // 管道阶数
                  Operator,                                   // 操作类型
                  false,                                      // 是否使用屏蔽内存清零
                  SharedMemoryClear> {                        // 共享内存是否清零

private:
    // 计算对齐比例，128 字节除以 bfloat16_t 类型的位数
    static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

    // 使用 Double-Q 脉动 MMA（Mixed Precision Matrix Multiply Accumulate）进行定义
    using Mma = DqMma<bfloat16_t,                             // A 操作数类型
                      LayoutA,                                // A 操作数的布局
                      kAlignmentA,                            // A 操作数的对齐方式
                      uint4b_t,                               // B 操作数的数据类型
                      LayoutB,                                // B 操作数的布局
                      kAlignmentB,                            // B 操作数的对齐方式
                      bfloat16_t,                             // 元素累加器的数据类型
                      layout::RowMajor,                       // 结果矩阵的布局
                      kAlignmentScale,                        // 对齐比例
                      ElementAccumulator,                     // 元素累加器的数据类型
                      layout::RowMajor,                       // 中间结果的布局
                      arch::OpClassTensorOp,                  // 硬件操作类别
                      ArchTag,                                // 硬件架构标签
                      ThreadblockShape,                       // 线程块形状
                      WarpShape,                              // 线程束形状
                      InstructionShape,                       // 指令形状
                      kStages,                                // 管道阶数
                      Operator,                               // 操作类型
                      SharedMemoryClear>;                     // 共享内存是否清零

public:
    // 定义 MmaCore 组件，即 DqMma 结构体中的 MmaCore 类型
    using MmaCore = typename Mma::MmaCore;

    // 定义 A 操作数的迭代器类型
    using IteratorA = typename Mma::IteratorA;

    // 定义 B 操作数的迭代器类型
    using IteratorB = typename Mma::IteratorB;

    // 定义线程块级别的流水线矩阵乘法运算类型 ThreadblockMma
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
```