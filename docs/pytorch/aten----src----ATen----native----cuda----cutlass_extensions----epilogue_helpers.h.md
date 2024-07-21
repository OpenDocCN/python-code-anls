# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\epilogue_helpers.h`

```
/**
 * @file epilogue_helpers.h
 *
 * This file includes types for the epilogues. The empty structs exist so we can signal to template
 * code the type of epilogue we want to run, and let the underlying code specify the details such as
 * element types, accumulator type and elements per vector access.
 *
 */

#pragma once

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <ATen/native/cuda/cutlass_extensions/epilogue/thread/ft_fused_activations.h>

namespace fastertransformer {

// Struct defining an epilogue operation with SILU activation and bias
struct EpilogueOpBiasSilu {};

// Struct defining an epilogue operation with ReLU activation and bias
struct EpilogueOpBiasReLU {};

// Struct defining an epilogue operation with FT GELU activation and bias
struct EpilogueOpBiasFtGelu {};

// Struct defining an epilogue operation with bias
struct EpilogueOpBias {};

// Struct defining an epilogue operation without bias
struct EpilogueOpNoBias {};

// Template struct defining an epilogue operation with specific parameters
template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator, typename Op>
struct Epilogue {
};

// Specialization for Epilogue with SILU activation and bias
template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasSilu> {
    // Define 'Op' using LinearCombinationSilu for SILU activation
    using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType,
                                                                ElementsPerVectorAccess,
                                                                ElementAccumulator,
                                                                ElementAccumulator,
                                                                cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

// Specialization for Epilogue with ReLU activation and bias
template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasReLU> {
    // Define 'Op' using LinearCombinationRelu for ReLU activation
    using Op = cutlass::epilogue::thread::LinearCombinationRelu<ElementType,
                                                                ElementsPerVectorAccess,
                                                                ElementAccumulator,
                                                                ElementAccumulator,
                                                                cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

// Partial specialization for Epilogue with FT GELU activation and bias (continuation in actual code)
    # 定义一个类型别名 Op，用于线性组合的通用尾声操作
    using Op = cutlass::epilogue::thread::LinearCombinationGeneric<
        cutlass::epilogue::thread::GELU_taylor,  # 使用 GELU Taylor 版本作为线性组合的尾声操作
        ElementType,  # 元素类型，通常是操作的数据类型
        ElementsPerVectorAccess,  # 每次向量访问的元素数目
        ElementAccumulator,  # 元素累加器类型
        ElementAccumulator,  # 另一个元素累加器类型
        cutlass::epilogue::thread::ScaleType::NoBetaScaling,  # 没有 beta 缩放类型
        cutlass::FloatRoundStyle::round_to_nearest,  # 浮点数舍入风格，四舍五入到最近的整数
        true  # 布尔值参数，此处意义为启用某些特定功能或配置
    >;
};

// 结构模板 Epilogue 的特化，处理带偏置的情况
template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBias> {
    // 使用 LinearCombination 类处理线程内部的线性组合运算，无需 beta 缩放
    using Op = cutlass::epilogue::thread::LinearCombination<ElementType,
                                                            ElementsPerVectorAccess,
                                                            ElementAccumulator,
                                                            ElementAccumulator,
                                                            cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

// 结构模板 Epilogue 的特化，处理无偏置的情况
template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpNoBias> {
    // 使用 LinearCombination 类处理线程内部的线性组合运算，使用默认的 beta 缩放
    using Op = cutlass::epilogue::thread::LinearCombination<ElementType,
                                                            ElementsPerVectorAccess,
                                                            ElementAccumulator,
                                                            ElementAccumulator,
                                                            cutlass::epilogue::thread::ScaleType::Default>;
};

// fastertransformer 命名空间的结尾
}  // namespace fastertransformer
```