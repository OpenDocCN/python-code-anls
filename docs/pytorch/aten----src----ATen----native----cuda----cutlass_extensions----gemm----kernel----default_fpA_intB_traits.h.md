# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\kernel\default_fpA_intB_traits.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <cutlass/arch/arch.h>
// 包含 Cutlass 库的架构相关头文件

#include <cutlass/arch/mma.h>
// 包含 Cutlass 库的矩阵乘法加速器相关头文件

#include <cutlass/bfloat16.h>
// 包含 Cutlass 库的 bfloat16 类型相关头文件

#include <cutlass/cutlass.h>
// 包含 Cutlass 库的核心头文件

#include <cutlass/gemm/gemm.h>
// 包含 Cutlass 库的矩阵乘法头文件

#include <cutlass/layout/matrix.h>
// 包含 Cutlass 库的矩阵布局相关头文件

#include <ATen/native/cuda/cutlass_extensions/arch/mma.h>
// 包含 Cutlass 库的 MMA（Mixed Matrix-Vector Multiply Accumulate）扩展架构头文件

#include <ATen/native/cuda/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h>
// 包含 Cutlass 库的混合矩阵乘法 B 布局内核头文件

namespace cutlass {
namespace gemm {
namespace kernel {

template<typename TypeA, typename TypeB, typename arch, typename Enable = void>
struct MixedGemmArchTraits {
};
// 通用模板，用于定义混合矩阵乘法的架构特性，未特化时为空

template<typename arch>
struct MixedGemmArchTraits<float, float, arch> {
    // float 类型的特化模板，针对特定架构 arch

    static constexpr int Stages = 2;
    // 定义计算阶段数为 2

    using OperatorClass = cutlass::arch::OpClassSimt;
    // 使用 SIMT 操作类

    using AccType = float;
    // 定义累加器类型为 float

    using LayoutB = cutlass::layout::RowMajor;
    // 使用行主布局类型 LayoutB

    static constexpr int ElementsPerAccessA = 1;
    // A 矩阵每次访问元素个数为 1

    static constexpr int ElementsPerAccessB = 1;
    // B 矩阵每次访问元素个数为 1

    static constexpr int ElementsPerAccessC = 1;
    // C 矩阵每次访问元素个数为 1

    static constexpr int ThreadblockK = 8;
    // 线程块大小 K 维度为 8

    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    // 指令形状为 1x1x1 的矩阵乘法形状

    using Operator = cutlass::arch::OpMultiplyAdd;
    // 使用乘加操作符
};

// ========================= Volta Traits ===========================
// Volta 架构特性模板定义

// Volta 架构在全局内存加载后始终进行去量化。这将实例化任何 Volta 的 HMMA 张量核心。
// 注意，Volta 不支持原生的 bfloat 类型，因此权重和激活值将被转换为 fp16，
// 在 fp16 中进行计算，然后转换为 bf16 输出。

template<typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    cutlass::arch::Sm70,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
                                          || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {
    // 特化模板，用于 Volta 架构的混合矩阵乘法特性

private:
    using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm70>;
    // 使用 TypeB 和 Sm70 架构定义的布局细节类型

public:
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;
    // 线程块大小 K 维度由布局细节确定

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    // 使用张量核操作类

    using AccType = float;
    // 定义累加器类型为 float

    using LayoutB = typename LayoutDetails::Layout;
    // 使用布局细节中定义的布局类型 LayoutB

    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
    // A 矩阵每次访问元素个数为 128 / TypeA 位数

    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    // B 矩阵每次访问元素个数由布局细节确定

    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;
    // C 矩阵每次访问元素个数为 128 / TypeA 位数

    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
    // 指令形状为 8x8x4 的矩阵乘法形状

    using Operator = typename LayoutDetails::Operator;
    // 使用布局细节中定义的操作符类型
};

// ======================= Turing Traits ==============================
// Turing 架构特性模板定义

// 注意，Turing 架构不支持原生的 bfloat 类型，因此权重和激活值将被转换为 fp16，
// 在 fp16 中进行计算，然后转换为 bf16 输出。

template<typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    cutlass::arch::Sm75,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
                                          || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {
    // 特化模板，用于 Turing 架构的混合矩阵乘法特性
    # 当类型 TypeA 是 cutlass::half_t 或 cutlass::bfloat16_t 时，启用特定的模板参数
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
                                          || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {
// 定义私有别名 LayoutDetails 作为 LayoutDetailsB<TypeB, cutlass::arch::Sm75> 的别名
private:
    using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm75>;

public:
    // 定义静态常量 ThreadblockK，其值为 LayoutDetails 的 ThreadblockK 成员
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    // 定义 OperatorClass 别名为 cutlass::arch::OpClassTensorOp
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    // 定义 AccType 别名为 float
    using AccType = float;
    // 定义 LayoutB 别名为 LayoutDetails 的 Layout 成员类型
    using LayoutB = typename LayoutDetails::Layout;

    // 计算 ElementsPerAccessA 的静态常量，128 除以 TypeA 的比特数
    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
    // 定义 ElementsPerAccessB 的静态常量，其值为 LayoutDetails 的 ElementsPerAccess 成员
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    // 计算 ElementsPerAccessC 的静态常量，128 除以 TypeA 的比特数
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;
    // 定义 InstructionShape 别名为 cutlass::gemm::GemmShape<16, 8, 8>
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

    // 定义 Operator 别名为 LayoutDetails 的 Operator 成员类型
    using Operator = typename LayoutDetails::Operator;
};

// ======================= Ampere Traits ==============================
// MixedGemmArchTraits 结构模板，用于 TypeA 和 TypeB，针对 cutlass::arch::Sm80
template<typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA,
    TypeB,
    cutlass::arch::Sm80,
    // 当 TypeA 是 cutlass::half_t 或 cutlass::bfloat16_t 时，启用此结构
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
                                          || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type> {
private:
    // 定义私有别名 LayoutDetails 作为 LayoutDetailsB<TypeB, cutlass::arch::Sm80> 的别名
    using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm80>;

public:
    // 定义静态常量 ThreadblockK，其值为 LayoutDetails 的 ThreadblockK 成员
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    // 定义 OperatorClass 别名为 cutlass::arch::OpClassTensorOp
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    // 定义 AccType 别名为 float
    using AccType = float;
    // 定义 LayoutB 别名为 LayoutDetails 的 Layout 成员类型
    using LayoutB = typename LayoutDetails::Layout;

    // 计算 ElementsPerAccessA 的静态常量，128 除以 TypeA 的比特数
    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
    // 定义 ElementsPerAccessB 的静态常量，其值为 LayoutDetails 的 ElementsPerAccess 成员
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    // 计算 ElementsPerAccessC 的静态常量，128 除以 TypeA 的比特数
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;
    // 定义 InstructionShape 别名为 cutlass::gemm::GemmShape<16, 8, 16>
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    // 定义 Operator 别名为 LayoutDetails 的 Operator 成员类型
    using Operator = typename LayoutDetails::Operator;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
```