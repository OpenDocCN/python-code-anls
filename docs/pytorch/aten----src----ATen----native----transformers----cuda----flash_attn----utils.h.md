# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\utils.h`

```py
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pytorch_flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

// 定义模板函数 relu2，用于对输入 x 进行 ReLU 操作
template<typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

// 特化模板函数 relu2 为 half_t 类型，利用 GPU 指令执行 max 操作
template<>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n"
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

// 条件编译下的特化模板函数 relu2 为 bfloat16_t 类型，利用 GPU 指令执行 max 操作
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
    asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// 条件编译下的模板函数 convert_relu2，将 float2 类型 x 转换并执行 ReLU 操作
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

// 特化模板函数 convert_relu2 为 half_t 类型，利用 GPU 指令执行浮点转换和 ReLU 操作
template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

// 特化模板函数 convert_relu2 为 bfloat16_t 类型，利用 GPU 指令执行浮点转换和 ReLU 操作
template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// 结构模板 MaxOp，定义模板化的最大值操作符
template<typename T>
struct MaxOp {
    // 在设备上执行的最大值操作
    __device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

// 特化 MaxOp 结构体为 float 类型，优化以提高速度
template <>
struct MaxOp<float> {
    // 这样略快
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    // 定义静态成员函数 run，用于在所有线程中执行归约操作
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        // 使用 __shfl_xor_sync 实现异步线程间数据交换，并执行操作 op
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        // 递归调用 run 函数，继续进行归约操作，直到 THREADS 为 2
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
    // 特化模板，处理 THREADS 为 2 的情况
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        // 使用 __shfl_xor_sync 实现异步线程间数据交换，并执行操作 op
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        // 返回运算结果
        return x;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    // 静态断言，验证 tCrA 的第一维度大小等于 acc 的第一维度大小（MMA_M）
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    // 静态断言，验证 tCrB 的第一维度大小等于 acc 的第二维度大小（MMA_N）
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    // 静态断言，验证 tCrA 和 tCrB 的第二维度大小相等（MMA_K）
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    
    // 使用 ThrCopyA 的 retile_D 方法对 tCrA 进行重塑
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    // 静态断言，验证 tCsA 的第一维度大小等于 tCrA_copy_view 的第一维度大小（M）
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    
    // 使用 ThrCopyB 的 retile_D 方法对 tCrB 进行重塑
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    // 静态断言，验证 tCsB 的第一维度大小等于 tCrB_copy_view 的第一维度大小（N）
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    
    // 如果 A_in_regs 为 false，则从 tCsA 复制数据到 tCrA_copy_view
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    // 如果 B_in_regs 为 false，则从 tCsB 复制数据到 tCrB_copy_view
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    
    // 循环执行 gemm 操作，每次处理 tCrA 和 tCrB 的第三维度数据，并将结果累加到 acc 中
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            // 如果 i 不是最后一个维度，且 A_in_regs 为 false，则继续从 tCsA 复制数据到 tCrA_copy_view
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            // 如果 i 不是最后一个维度，且 B_in_regs 为 false，则继续从 tCsB 复制数据到 tCrB_copy_view
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        // 执行 tiled_mma 进行矩阵乘法累加操作，结果累加到 acc 中
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    // 断言：确保 tCrA 的第一维度大小等于 acc 的第一维度大小 (MMA_M)
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     
    // 断言：确保 tCrB 的第一维度大小等于 acc 的第二维度大小 (MMA_N)
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     
    // 断言：确保 tCrA 和 tCrB 的第二维度大小相同 (MMA_K)
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     
    // 将 tCrB 视图复制到共享内存中以便使用
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    // 断言：确保 tCsB 的第一维度大小等于 tCrB_copy_view 的第一维度大小 (N)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            
    // 将 tCsB 的数据复制到 tCrB_copy_view 中的相应位置
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    // 循环遍历 tCrA 的第二维度
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        // 如果 i 不是 tCrA 的最后一个维度
        if (i < size<2>(tCrA) - 1) {
            // 将 tCsB 的下一个维度数据复制到 tCrB_copy_view 中的相应位置
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        // 使用 tiled_mma 计算 tCrA、tCrB 中的数据，并累加到 acc 中
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// 将 acc_layout 从 (MMA=4, MMA_M, MMA_N) 转换为 (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    // 确保 acc_layout 的第一维度大小为 4
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    // 确保 acc_layout 的秩为 3
    static_assert(decltype(rank(acc_layout))::value == 3);
    // 对 acc_layout 进行逻辑分割，生成转换后的布局
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 将 acc_layout 从 (MMA=4, MMA_M, MMA_N) 转换为 ((4, 2), MMA_M, MMA_N / 2)
// 如果使用 m16n8k16，则转换为 (4, MMA_M, MMA_N)，如果使用 m16n8k8 则不做转换
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    // 确保 acc_layout 的第一维度大小为 4
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    // 确保 acc_layout 的秩为 3
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    // 确保 mma_shape_K 是 8 或 16
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    // 根据 mma_shape_K 的不同进行不同的布局转换
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 将 acc_layout 从 (MMA=4, MMA_M, MMA_N) 转换为 ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    // 使用 Underscore 作为 X 的别名
    using X = Underscore;
    // 确保 acc_layout 的第一维度大小为 4
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    // 确保 acc_layout 的秩为 3
    static_assert(decltype(rank(acc_layout))::value == 3);
    // 对 acc_layout 进行逻辑划分，以 Shape<X, X, _2> 作为划分依据
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    // 创建新的布局，按顺序排列划分后的结果
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    // 定义源类型为 Engine 的值类型
    using From_type = typename Engine::value_type;
    // 获取张量的元素数量
    constexpr int numel = decltype(size(tensor))::value;
    // 创建数值数组转换器，将 From_type 类型转换为 To_type 类型，numel 为元素数量
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: 这要求张量必须是“连续”的
    // 将张量数据重新解释为 From_type 类型的数组，并使用转换器进行转换操作
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    // 创建新的张量，使用转换后的数据和原始张量的布局
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    // 获取张量的元素数量
    constexpr int numel = decltype(size(tensor))::value;
    // 确保元素数量为偶数
    static_assert(numel % 2 == 0);
    // 定义 value_t 为 Engine 的值类型
    using value_t = typename Engine::value_type;
    // HACK: 这要求张量必须是“连续”的
    // 将张量重新解释为 uint32_t 类型的张量
    Tensor tensor_uint32 = recast<uint32_t>(tensor);
    #pragma unroll
    // 对张量中的每个元素应用 relu2 函数
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// 在 SM80 及以上，我们可以将 fp32 -> fp16/bf16 转换和 relu 合并为 1 条指令
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    // 定义源类型为 Engine 的值类型
    using From_type = typename Engine::value_type;
    // 确保 To_type 类型为 cutlass::half_t 或 cutlass::bfloat16_t
    static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
    // 确保 From_type 类型为 float
    static_assert(std::is_same_v<float, From_type>);
    // 获取张量的元素数量
    constexpr int numel = decltype(size(tensor))::value;
    // 确保元素数量为偶数
    static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // HACK: 这要求张量必须是“连续”的
    // 将张量重新解释为 float2 类型的张量
    Tensor tensor_float2 = recast<float2>(tensor);
    // 创建新的 uint32_t 类型的张量
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
    #pragma unroll
    // 对新张量中的每个元素应用 convert_relu2 函数
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    // 使用新的数据创建张量，并使用原始张量的布局
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
    // 创建新的张量，进行类型转换为 To_type
    Tensor out = pytorch_flash::convert_type<To_type>(tensor);
    // 对新张量应用 relu_ 函数
    pytorch_flash::relu_(out);
#endif
    // 返回结果张量
    return out;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

// 阻塞直到所有但前 N 个 cp.async.commit_group 操作完成。
// 这与 cute::cp_async_wait 不同之处在于当 N = 0 时，我们不调用 cp.async.wait_all
// （这等效于 commit_group 然后 wait_group 0）。
// 而是直接调用 cp.async.wait_group 0，这样稍微更快一些。
// 参考链接：https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});                          // 断言 S 的秩为 3
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});                          // 断言 D 的秩为 3
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // 断言 S 和 D 的第一个维度大小相等（MMA）
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // 断言 S 和 D 的第二个维度大小相等（MMA_M）
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // 断言 S 和 D 的第三个维度大小相等（MMA_K）
    // 当 Clear_OOB_MN && !Clear_OOB_K 时，静态断言失败
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {                               // 对 S 的第二个维度（M）进行迭代
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {       // 根据条件判断是否进行下一步操作
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {                       // 对 S 的第三个维度（K）进行迭代
                if (Is_even_K || predicate_K(k)) {                      // 根据条件判断是否进行拷贝操作或清除操作
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));     // 调用 cute 库中的拷贝函数
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));                            // 调用 cute 库中的清除函数
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));                                    // 调用 cute 库中的清除函数
        }
    }
    // TD [2023-04-13]: 下面的代码段很奇怪，可能会导致竞态条件。
    // 我认为是因为这些拷贝操作都在 if 语句块内。
    // if (Is_even_K) {
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(S); ++m) {
    //         if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //             copy(tiled_copy, S(_, m, _), D(_, m, _));
    //         } else if (Clear_OOB_MN) {
    //             clear(D(_, m, _));
    //         }
    //     }
    // } else {  // 在这种情况下，如果先迭代 K，速度稍快
    //     #pragma unroll
    //     for (int k = 0; k < size<2>(S); ++k) {
    // 如果 predicate_K(k) 成立时执行以下代码块
    if (predicate_K(k)) {
        // 对于 S 的第一维大小的每个索引 m，进行循环遍历
        #pragma unroll
        for (int m = 0; m < size<1>(S); ++m) {
            // 如果 Is_even_MN 为真或者 identity_MN(0, m, 0) 的第一个元素小于 max_MN
            if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
                // 将 S(_, m, k) 的内容复制到 D(_, m, k)
                copy(tiled_copy, S(_, m, k), D(_, m, k));
            } else if (Clear_OOB_MN) {
                // 否则，如果 Clear_OOB_MN 为真，清空 D(_, m, k)
                clear(D(_, m, k));
            }
        }
    } else if (Clear_OOB_K) {
        // 否则，如果 Clear_OOB_K 为真
        // 在这种情况下，不存在 !Clear_OOB_K && Clear_OOB_MN 的情况
        if (Clear_OOB_MN || Is_even_MN) {
            // 如果 Clear_OOB_MN 或者 Is_even_MN 为真，清空 D(_, _, k)
            clear(D(_, _, k));
        } else {
            // 否则，对于 S 的第一维大小的每个索引 m，进行循环遍历
            #pragma unroll
            for (int m = 0; m < size<1>(S); ++m) {
                // 如果 !(Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)
                if (!(Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)) {
                    // 清空 D(_, m, k)
                    clear(D(_, m, k));
                }
            }
        }
    }
    // 结束 if (predicate_K(k)) 的条件判断
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      Tensor<Engine3, Layout3> const &predicate_K,
                                      const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});                               // 确保张量 S 的秩为 3
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});                               // 确保张量 D 的秩为 3
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                          // 确保张量 S 和 D 在第 0 维度上的大小相同
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                          // 确保张量 S 和 D 在第 1 维度上的大小相同
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                          // 确保张量 S 和 D 在第 2 维度上的大小相同
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
    #pragma unroll                                                          // 循环展开指令
    for (int m = 0; m < size<1>(S); ++m) {                                   // 遍历第 1 维度上的索引 m
        // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // 如果 identity_MN(0, m, 0) 大于等于 min_MN 且小于 max_MN，则进入条件判断
            // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
            #pragma unroll                                                  // 循环展开指令
            for (int k = 0; k < size<2>(S); ++k) {                           // 遍历第 2 维度上的索引 k
                if (Is_even_K || predicate_K(k)) {                          // 判断是否为偶数或者满足 predicate_K(k) 的条件
                    cute::copy(S(_, m, k), D(_, m, k));                     // 调用 cute 命名空间下的 copy 函数进行数据复制
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace pytorch_flash
```