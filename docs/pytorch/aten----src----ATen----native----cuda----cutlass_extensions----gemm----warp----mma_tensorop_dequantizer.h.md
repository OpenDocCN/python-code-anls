# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\warp\mma_tensorop_dequantizer.h`

```py
`
/*
 * 文件描述：定义用于针对张量核心进行矩阵乘法操作的warp级迭代器。
 */

#pragma once

#include <cutlass/cutlass.h>

#include <cutlass/array.h>  // 包含了处理数组的相关功能
#include <cutlass/matrix_shape.h>  // 包含了处理矩阵形状的相关功能
#include <cutlass/numeric_types.h>  // 包含了处理数值类型的相关功能
#include <cutlass/tensor_ref.h>  // 包含了处理张量引用的相关功能

#include <cutlass/arch/arch.h>  // 包含了处理架构的相关功能
#include <cutlass/arch/memory_sm75.h>  // 包含了处理内存与SM75架构的相关功能
#include <cutlass/gemm/gemm.h>  // 包含了处理GEMM操作的相关功能

#include <cutlass/layout/matrix.h>  // 包含了处理矩阵布局的相关功能
#include <cutlass/layout/pitch_linear.h>  // 包含了处理基于线性间距布局的相关功能
#include <cutlass/layout/tensor.h>  // 包含了处理张量布局的相关功能

#include <cutlass/functional.h>  // 包含了处理功能性操作的相关功能
#include <cutlass/platform/platform.h>  // 包含了处理平台相关操作的相关功能

//#include <src/fastertransformer/utils/cuda_bf16_wrapper.h>
//#ifdef ENABLE_BF16
#include <cuda_bf16.h>  // 引入CUDA BF16库，处理BF16相关操作
//#endif

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template<
    typename MmaOperator_,  // 矩阵乘法操作器类型
    typename Shape_,  // 矩阵形状类型
    Operand Operand,  // 操作数类型标识
    typename Element_,  // 元素数据类型
    /// 操作数的布局类型
    typename Layout_,
    /// 参与单个矩阵操作的线程数
    int Threads,
    /// 启用类型，通常用于模板元编程
    typename Enable = void>
// 定义 MmaTensorOpDequantizer 类，用于 bfloat16 数据类型在 Ampere 架构下的特化处理
class MmaTensorOpDequantizer;

////////////////////////////////////////////////////////////////////////////////
// Ampere 架构下的 bfloat16 特化处理模板
template<
    // 基础矩阵乘法操作器类型 (概念: MmaTensorOp)
    typename MmaOperator_,
    // Warp 级别矩阵乘法的形状 (概念: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    bfloat16_t,
    layout::RowMajor,
    32,
    typename platform::enable_if<
        MmaOperator_::ArchTag::kMinComputeCapability >= 80
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type> {

public:
    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // 正在使用的架构特定的 mma 操作器
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma 指令形状
    using InstructionShape = typename ArchMmaOperator::Shape;

    // 这是加载指令与计算指令之间的比率。
    static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    /// Scales 的类型
    using ElementScale = bfloat16_t;

    /// 在执行 Mma 前保存 B 数据的片段
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

    // 保存应用于 B 的比例数据片段
    // 每个矩阵迭代在 N 维度需要 1 个 fp16
    static constexpr int kColsPerMmaPerThread = 1;
    using FragmentScale = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

    /// Warp mma 形状
    using Shape = Shape_;

    /// Scales 在共享内存中的布局
    using Layout = layout::RowMajor;

    /// 从张量中加载元素的 TensorRef 类型
    using TensorRef = TensorRef<ElementScale, Layout>;

    // 构造函数，初始化 MmaTensorOpDequantizer 实例
    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
    {
        // 计算 warp 偏移量
        const int warp_offset   = warp_idx_n * Shape::kN;
        // 计算 quad
        const int quad          = lane_idx / 4;
        // 计算线程偏移量
        const int thread_offset = warp_offset + quad;
        // 设置指针指向共享内存中的比例数据起始位置
        pointer_                = smem_scales.data() + thread_offset;
    }

    // 加载函数，从共享内存加载比例数据到 scale_frag
    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {
        // 循环展开，加载每个 Mma 列迭代对应的比例数据
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
            scale_frag[mma_n_iter] = pointer_[mma_n_iter * InstructionShape::kN];
        }
    }

    // 解量化函数，应用比例因子到操作数片段
    CUTLASS_DEVICE
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
//#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        // 定义 _MmaOperandB 为 ArchMmaOperator 的 FragmentB 类型
        using _MmaOperandB        = typename ArchMmaOperator::FragmentB;
        // 定义 ExpandedMmaOperandB 为 _MmaOperandB 元素类型的数组，长度为 kExpansionFactor * _MmaOperandB 元素个数
        using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor * _MmaOperandB::kElements>;
        // 断言，确保扩展后的 OperandB 的元素个数乘以 MmaOperator 的迭代次数等于 FragmentDequantizedOperand 的元素个数
        static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                          == FragmentDequantizedOperand::kElements,
                      "");

        // 将 scale_frag 强制转换为 __nv_bfloat16 指针
        const __nv_bfloat16* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(&scale_frag);

        // 将 operand_frag 强制转换为 ExpandedMmaOperandB 指针
        ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
        // 循环展开，对每个 Mma 迭代执行以下操作
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
            // 断言，确保 ExpandedMmaOperandB 的元素个数是偶数
            static_assert(ExpandedMmaOperandB::kElements % 2 == 0, "");

            // 读取并扩展 scale_ptr[mma_n_iter]，存入 scalex2
            __nv_bfloat162  scalex2            = __bfloat162bfloat162(scale_ptr[mma_n_iter]);
            // 将 operand_frag_ptr[mma_n_iter] 强制转换为 __nv_bfloat162* 指针
            __nv_bfloat162* operand_bf16x2_ptr = reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[mma_n_iter]);
            // 循环展开，对 ExpandedMmaOperandB 的每对元素执行 scalex2 的二元半精度浮点数乘法
            CUTLASS_PRAGMA_UNROLL
            for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii) {
                operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii], scalex2);
            }
        }
#else
        // 对于老架构的慢路径，这里没有实现。如果需要在较旧的架构上执行 HMMA，应该在将 scales 存储到共享内存之前进行缩放转换，并使用 fp16 的反量化器。这将避免 GEMM 主循环中的大量转换指令。
        // 在此目的下故意未实现慢路径。
        arch::device_breakpoint();
#endif
    }

private:
    // 指向 ElementScale 类型的常量指针
    ElementScale const* pointer_;
};

////////////////////////////////////////////////////////////////////////////////

// 适用于图灵架构和安培架构的特化版本
template<
    /// 底层矩阵乘操作符（概念：MmaTensorOp）
    typename MmaOperator_,
    /// 瓦片级矩阵乘形状（概念：GemmShape）
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    half_t,
    layout::RowMajor,
    32,
    typename platform::enable_if<
        MmaOperator_::ArchTag::kMinComputeCapability >= 75
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type> {

public:
    /// Mma 操作符
    using MmaOperator = MmaOperator_;

    // 正在使用的特定于架构的 mma 操作符
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma 指令形状
    using InstructionShape = typename ArchMmaOperator::Shape;

    // 这是载入指令与计算指令的比例
    static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    /// 比例类型
    using ElementScale = half_t;

    /// 在执行 Mma 之前保持 B 数据的片段
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;
    // 定义一个常量表达式，表示每个线程在 N 维度上进行 Mma 前的缩放数据
    // 每个矩阵迭代需要使用 1 个 fp16 数据
    static constexpr int kColsPerMmaPerThread = 1;
    // 定义一个数组类型 FragmentScale，用于存储每个线程的缩放数据，每个数组元素占据 kColumn 个数据
    using FragmentScale = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

    /// Warp mma shape
    // 使用模板参数 Shape_ 来定义 Warp mma 的形状
    using Shape = Shape_;

    /// Layout of the scales in shared memory
    // 使用 RowMajor 布局来定义共享内存中的缩放数据布局
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    // 定义 TensorRef 类型，用于从张量中加载元素，元素类型为 ElementScale，使用 RowMajor 布局
    using TensorRef = TensorRef<ElementScale, Layout>;

    // 构造函数 MmaTensorOpDequantizer，用于初始化对象
    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
    {
        // 计算 warp 的偏移量，基于 warp_idx_n 和 Shape::kN
        const int warp_offset   = warp_idx_n * Shape::kN;
        // 计算四个线程组（quad）的索引
        const int quad          = lane_idx / 4;
        // 计算线程偏移量，基于 warp_offset 和 quad
        const int thread_offset = warp_offset + quad;
        // 计算指针的位置，指向 smem_scales 中的特定位置数据
        pointer_                = smem_scales.data() + thread_offset;
    }

    // 成员函数 load，用于加载缩放数据到 scale_frag 中
    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {
        // 使用指令展开（UNROLL）来遍历 MmaOperator 的迭代次数 kColumn
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
            // 从指针 pointer_ 处加载数据到 scale_frag[mma_n_iter] 中
            scale_frag[mma_n_iter] = pointer_[mma_n_iter * InstructionShape::kN];
        }
    }

    // 成员函数 dequantize，用于对 operand_frag 进行反量化操作，使用 scale_frag 进行缩放
    CUTLASS_DEVICE
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
        // 定义 _MmaOperandB 类型，使用 ArchMmaOperator::FragmentB
        using _MmaOperandB        = typename ArchMmaOperator::FragmentB;
        // 定义 ExpandedMmaOperandB 类型，是 _MmaOperandB 的扩展数组，元素个数为 kExpansionFactor * _MmaOperandB::kElements
        using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor * _MmaOperandB::kElements>;
        // 静态断言，确保 ExpandedMmaOperandB 的元素个数乘以 MmaOperator::MmaIterations::kColumn 等于 FragmentDequantizedOperand::kElements
        static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                          == FragmentDequantizedOperand::kElements,
                      "");

        // 定义乘法操作符 mul_op，用于进行乘法操作
        multiplies<ExpandedMmaOperandB> mul_op;

        // 将 operand_frag 强制转换为 ExpandedMmaOperandB 指针 operand_frag_ptr
        ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
        // 使用指令展开（UNROLL）来遍历 MmaOperator 的迭代次数 kColumn
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
            // 对 operand_frag_ptr[mma_n_iter] 进行乘法操作，乘数为 scale_frag[mma_n_iter]
            operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
        }
    }
private:
    ElementScale const* pointer_;
};

////////////////////////////////////////////////////////////////////////////////

// Specialization for Volta A x RowMajor B tensorOp, for 32x32x4 interleaved gemm
template<
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    half_t,
    layout::RowMajor,
    32,
    typename platform::enable_if<
        platform::is_same<typename MmaOperator_::ArchTag, arch::Sm70>::value
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::RowMajor>::value>::type> {

public:
    static_assert(platform::is_same<typename MmaOperator_::InterleavedTileShape, GemmShape<32, 32, 4>>::value, "");

    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma operator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    /// Type of the scales
    using ElementScale = half_t;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

    /// Warp mma shape
    using Shape = Shape_;

    // Fragment to hold scale data to apply to B before mma
    // Each 32x32x4 matmul uses 8 elements from B.
    static constexpr int ColsPerMmaTile  = 32;
    static constexpr int TileNIterations = Shape::kN / ColsPerMmaTile;
    using FragmentScale                  = Array<ElementScale, TileNIterations * 8>;
    using AccessType                     = Array<ElementScale, 8>;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = TensorRef<ElementScale, Layout>;

    // Constructor for initializing MmaTensorOpDequantizer object
    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
    {
        // Calculate the offset within shared memory for scales based on warp index and lane index
        const int warp_offset   = warp_idx_n * Shape::kN;
        const int base_col      = lane_idx & 0xF8; // Ensure alignment to 8 elements (0xF8 = 248 in decimal)
        const int thread_offset = warp_offset + base_col;
        // Set pointer to the starting address of scales in shared memory
        pointer_                = smem_scales.data() + thread_offset;
    }

    // Method to load scale data from shared memory into FragmentScale
    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {
        AccessType* scale_frag_ptr = reinterpret_cast<AccessType*>(&scale_frag);

        CUTLASS_PRAGMA_UNROLL
        for (int tile_iter = 0; tile_iter < TileNIterations; ++tile_iter) {
            // Load scale data from shared memory into scale_frag using AccessType
            // Each iteration jumps by ColsPerMmaTile (32) elements in pointer_
            scale_frag_ptr[tile_iter] = *reinterpret_cast<AccessType const*>(pointer_ + ColsPerMmaTile * tile_iter);
        }
    }

    // Method to dequantize operand_frag using scale_frag
    CUTLASS_DEVICE
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
        // 断言：确保FragmentScale和FragmentDequantizedOperand的元素数目相等，否则输出空字符串
        static_assert(FragmentScale::kElements == FragmentDequantizedOperand::kElements, "");
    
        // 创建一个乘法操作对象，类型为FragmentDequantizedOperand
        multiplies<FragmentDequantizedOperand> mul_op;
        
        // 使用乘法操作对象mul_op对操作数片段operand_frag和标度片段scale_frag进行乘法运算
        operand_frag = mul_op(operand_frag, scale_frag);
    }
private:
    ElementScale const* pointer_;
};

////////////////////////////////////////////////////////////////////////////////

// Specialization for Volta A x ColumnMajor B tensorOp, for 32x32x4 interleaved gemm
template<
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    half_t,
    layout::RowMajor,
    32,
    typename platform::enable_if<
        platform::is_same<typename MmaOperator_::ArchTag, arch::Sm70>::value
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type> {

public:
    static_assert(platform::is_same<typename MmaOperator_::InterleavedTileShape, GemmShape<32, 32, 4>>::value, "");

    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    /// Type of the scales
    using ElementScale = half_t;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

    /// Warp mma shape
    using Shape = Shape_;

    // Fragment to hold scale data to apply to B before mma
    // Each 32x32x4 matmul uses 8 elements from B.
    static constexpr int ColsPerMmaTile  = 32;
    static constexpr int TileNIterations = Shape::kN / ColsPerMmaTile;
    using FragmentScale                  = Array<ElementScale, TileNIterations * 2>;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = TensorRef<ElementScale, Layout>;

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
    {
        // Calculate the starting offset within shared memory based on warp and lane indices
        const int warp_offset   = warp_idx_n * Shape::kN;
        // Calculate the base column index for this thread within the warp
        const int base_col      = lane_idx & 0xF8 + lane_idx % 4;
        // Calculate the absolute offset of the current thread within the entire matrix
        const int thread_offset = warp_offset + base_col;
        // Set pointer_ to point to the starting scale data for this thread
        pointer_                = smem_scales.data() + thread_offset;
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {
        CUTLASS_PRAGMA_UNROLL
        for (int tile_iter = 0; tile_iter < TileNIterations; ++tile_iter) {
            // Unroll loop to load scale values for each tile iteration
            CUTLASS_PRAGMA_UNROLL
            for (int mma_iter = 0; mma_iter < 2; ++mma_iter) {
                // Load scale values from shared memory using pointer_ offset
                scale_frag[tile_iter * 2 + mma_iter] = pointer_[ColsPerMmaTile * tile_iter + 4 * mma_iter];
            }
        }
    }

    CUTLASS_DEVICE
    // 对给定的操作数片段进行反量化操作，使用指定的缩放因子片段
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
        // 定义片段乘法操作的操作数类型为 ArchMmaOperator 的 FragmentB
        using MmaOperandB = typename ArchMmaOperator::FragmentB;
        // 计算总共的 MMA 操作次数，每个 Tile 包含 2 * TileNIterations 个 MMA 操作
        static constexpr int total_n_mmas = 2 * TileNIterations;
        // 静态断言，确保操作数片段的元素个数等于 MmaOperandB 的元素个数乘以总 MMA 操作次数
        static_assert(MmaOperandB::kElements * total_n_mmas == FragmentDequantizedOperand::kElements, "");
    
        // 定义乘法操作类模板实例
        multiplies<MmaOperandB> mul_op;
    
        // 将操作数片段的指针重新解释为 MmaOperandB 类型的指针
        MmaOperandB* operand_frag_ptr = reinterpret_cast<MmaOperandB*>(&operand_frag);
        
        // 循环展开，对每个 MMA 操作进行反量化操作
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < total_n_mmas; ++mma_n_iter) {
            // 使用乘法操作符对操作数片段中的每个元素进行反量化
            operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
        }
    }
# 定义一个私有成员变量指针，指向 ElementScale 类型的常量对象
private:
    ElementScale const* pointer_;
};

////////////////////////////////////////////////////////////////////////////////

# 结束 warp 命名空间的声明
}  // namespace warp
# 结束 gemm 命名空间的声明
}  // namespace gemm
# 结束 cutlass 命名空间的声明
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
```