# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\rotary.h`

```py
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>  // 引入复制算法

#include <ATen/native/transformers/cuda/flash_attn/utils.h>  // 引入 CUDA 相关实用工具

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pytorch_flash {

using namespace cute;  // 使用 cute 命名空间

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true, bool Clear_OOB_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_interleaved(Tensor<Engine0, Layout0> const &S,  // 源张量 S
                                               Tensor<Engine1, Layout1> &D,  // 目标张量 D
                                               Tensor<Engine2, Layout2> const &Cos,  // 余弦张量 Cos
                                               Tensor<Engine2, Layout2> const &Sin,  // 正弦张量 Sin
                                               Tensor<Engine3, Layout3> const &identity_MN,  // 身份张量 identity_MN
                                               const int max_MN, const int min_MN,  // 最大和最小维度 MN
                                               const int dim, const int rotary_dim) {  // 维度和旋转维度
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});  // 断言 S 的秩为 3
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});  // 断言 D 的秩为 3
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // 断言 S 和 D 的第一维度大小相同
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // 断言 S 和 D 的第二维度大小相同
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // 断言 S 和 D 的第三维度大小相同
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));  // 断言 S 和 Cos 的第二维度大小相同
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));  // 断言 S 和 Cos 的第三维度大小相同
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));  // 断言 S 和 Sin 的第二维度大小相同
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));  // 断言 S 和 Sin 的第三维度大小相同
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));  // 断言 Cos 和 Sin 的第一维度大小相同
    static_assert(decltype(size<0>(S))::value == decltype(size<0>(Cos))::value * 2);  // 断言 S 的第一维度大小是 Cos 的两倍
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // 断言 Cos 的第一维度大小是偶数，用于快速将 fp16/bf16 转换为 fp32
    Tensor rCos = make_fragment_like(Cos);  // 创建与 Cos 相同类型的 rCos 张量
    Tensor rSin = make_fragment_like(Sin);  // 创建与 Sin 相同类型的 rSin 张量
    Tensor rS = make_fragment_like(S);  // 创建与 S 相同类型的 rS 张量
    #pragma unroll  // 循环展开指令，用于优化循环性能
    // 遍历索引 m，范围在 0 到 S 的第一个维度大小之间（size<1>(S)）
    for (int m = 0; m < size<1>(S); ++m) {
        // 检查 identity_MN 函数返回的第一个元素是否在 min_MN 和 max_MN 之间
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // 使用 #pragma unroll 指令，可能是用于循环展开优化
            // 遍历索引 k，范围在 0 到 S 的第二个维度大小之间（size<2>(S)）
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                // 如果 Is_even_K 为真，或者 identity_MN 函数返回的第二个元素小于 dim
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
                    // 复制 S(_, m, k) 到 rS(_, m, k)
                    cute::copy(S(_, m, k), rS(_, m, k));
                    // 如果 identity_MN 函数返回的第二个元素小于 rotary_dim
                    if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
                        // 复制 Cos(_, m, k) 到 rCos(_, m, k) 和 Sin(_, m, k) 到 rSin(_, m, k)
                        cute::copy(Cos(_, m, k), rCos(_, m, k));
                        cute::copy(Sin(_, m, k), rSin(_, m, k));
                        // 将 rS(_, m, k) 转换为 float 类型的 Tensor，命名为 S_fp32
                        Tensor S_fp32 = convert_type<float>(rS(_, m, k));
                        Tensor cos_fp32 = convert_type<float>(rCos(_, m, k));
                        Tensor sin_fp32 = convert_type<float>(rSin(_, m, k));
                        // 使用 #pragma unroll 指令，可能是用于循环展开优化
                        #pragma unroll
                        // 遍历索引 i，范围在 rS 的第一个维度大小的一半之间
                        for (int i = 0; i < size<0>(rS) / 2; ++i) {
                            // 计算复数乘法
                            float real = S_fp32(2 * i) * cos_fp32(i) - S_fp32(2 * i + 1) * sin_fp32(i);
                            float imag = S_fp32(2 * i) * sin_fp32(i) + S_fp32(2 * i + 1) * cos_fp32(i);
                            // 更新 S_fp32 中的实部和虚部
                            S_fp32(2 * i) = real;
                            S_fp32(2 * i + 1) = imag;
                        }
                        // 创建 S_fp32 的副本，使 convert_type 正常工作
                        Tensor S_fp32_copy = make_fragment_like(S_fp32);
                        cute::copy(S_fp32, S_fp32_copy);
                        // 使用 Engine0 的 value_type 类型 T，将 S_fp32_copy 转换为 S_og_type
                        using T = typename Engine0::value_type;
                        Tensor S_og_type = convert_type<T>(S_fp32_copy);
                        // 将 S_og_type 复制回 rS(_, m, k)
                        cute::copy(S_og_type, rS(_, m, k));
                    }
                    // 将 rS(_, m, k) 复制到 D(_, m, k)
                    cute::copy(rS(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    // 如果 Clear_OOB_K 为真，清空 D(_, m, k)
                    cute::clear(D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true, bool Clear_OOB_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_contiguous(Tensor<Engine0, Layout0> const &S,
                                              Tensor<Engine1, Layout1> &D,
                                              Tensor<Engine2, Layout2> const &Cos,
                                              Tensor<Engine2, Layout2> const &Sin,
                                              Tensor<Engine3, Layout3> const &identity_MN,
                                              const int max_MN, const int min_MN,
                                              const int dim, const int rotary_dim) {
    // 静态断言：确保张量 S 和 D 的秩为 3
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    // 静态断言：确保张量 S 和 D 在不同维度上的尺寸相同
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // 静态断言：确保张量 S 和 Cos 在第一维度上的尺寸相同
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    // 静态断言：确保张量 S 和 Cos 在第二维度上的尺寸相同
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    // 静态断言：确保张量 S 和 Sin 在第一维度上的尺寸相同
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    // 静态断言：确保张量 S 和 Sin 在第二维度上的尺寸相同
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    // 静态断言：确保张量 S 和 Cos 在第零维度上的尺寸相同
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(Cos));                     // MMA
    // 静态断言：确保张量 Cos 和 Sin 在第零维度上的尺寸相同
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));
    // 静态断言：确保张量 Cos 的第零维度的尺寸是偶数
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32

    // 根据 Cos 张量创建与之同样尺寸的临时片段张量 rCos
    Tensor rCos = make_fragment_like(Cos);
    // 根据 Sin 张量创建与之同样尺寸的临时片段张量 rSin
    Tensor rSin = make_fragment_like(Sin);
    // 根据 S 张量创建与之同样尺寸的临时片段张量 rS
    Tensor rS = make_fragment_like(S);
    // 根据 rS 的部分切片创建 rS_other 临时片段张量
    Tensor rS_other = make_fragment_like(rS(_, 0, 0));

    // #pragma unroll 指令：用于循环展开，根据编译器的优化策略进行展开
    #pragma unroll
    // 遍历第一维度为 1 的张量 S 的所有 m 索引
    for (int m = 0; m < size<1>(S); ++m) {
        // 检查 identity_MN 函数返回的第一个元素是否在指定范围内
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // 使用 #pragma unroll 指令，展开内层循环以优化性能
            // 遍历第二维度为 2 的张量 S 的所有 k 索引
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                // 如果 Is_even_K 为真或者 identity_MN 函数返回的第二个元素小于 dim
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
                    // 复制 S 的子张量 S(_, m, k) 到 rS(_, m, k)
                    cute::copy(S(_, m, k), rS(_, m, k));
                    // 如果 identity_MN 函数返回的第二个元素小于 rotary_dim
                    if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
                        // 确定是否是左侧部分
                        const bool is_left = get<1>(identity_MN(0, 0, k)) < rotary_dim / 2;
                        // 创建 gS_other 张量作为 S(_, m, k) 的另一部分
                        Tensor gS_other = make_tensor(S(_, m, k).data() + (is_left ? rotary_dim / 2 : -rotary_dim / 2), S(_, m, k).layout());
                        // 复制 gS_other 到 rS_other
                        cute::copy(gS_other, rS_other);
                        // 如果是第一个线程，则打印 rS(_, m, k) 和 rS_other 张量
                        // if (cute::thread0()) { print_tensor(rS(_, m, k)); print_tensor(rS_other); }
                        // 创建 gCos 和 gSin 张量，分别作为 Cos(_, m, k) 和 Sin(_, m, k) 的一部分
                        Tensor gCos = make_tensor(Cos(_, m, k).data() + (is_left ? 0 : -rotary_dim / 2), Cos(_, m, k).layout());
                        Tensor gSin = make_tensor(Sin(_, m, k).data() + (is_left ? 0 : -rotary_dim / 2), Sin(_, m, k).layout());
                        // 复制 gCos 到 rCos(_, m, k) 和 gSin 到 rSin(_, m, k)
                        cute::copy(gCos, rCos(_, m, k));
                        cute::copy(gSin, rSin(_, m, k));
                        // 如果是第一个线程，则打印 rCos(_, m, k) 和 rSin(_, m, k) 张量
                        // if (cute::thread0()) { print_tensor(rCos(_, m, k)); print_tensor(rSin(_, m, k)); }
                        // 将 rS(_, m, k)、rS_other、rCos(_, m, k) 和 rSin(_, m, k) 张量转换为 float 类型
                        Tensor S_fp32 = convert_type<float>(rS(_, m, k));
                        Tensor S_other_fp32 = convert_type<float>(rS_other);
                        Tensor cos_fp32 = convert_type<float>(rCos(_, m, k));
                        Tensor sin_fp32 = convert_type<float>(rSin(_, m, k));
                        // 遍历 rS 的第零维度，对每个元素执行复杂运算
                        #pragma unroll
                        for (int i = 0; i < size<0>(rS); ++i) {
                            S_fp32(i) = S_fp32(i) * cos_fp32(i) + S_other_fp32(i) * (is_left ? -sin_fp32(i) : sin_fp32(i));
                        }
                        // 复制 S_fp32 到 S_fp32_copy 以便 convert_type 正常工作
                        Tensor S_fp32_copy = make_fragment_like(S_fp32);
                        cute::copy(S_fp32, S_fp32_copy);
                        // 使用 Engine0 的 value_type 类型 T，将 S_fp32_copy 转换回原始类型 S_og_type
                        using T = typename Engine0::value_type;
                        Tensor S_og_type = convert_type<T>(S_fp32_copy);
                        // 复制 S_og_type 到 rS(_, m, k)
                        cute::copy(S_og_type, rS(_, m, k));
                        // 如果是第一个线程，则打印 rS(_, m, k) 张量
                        // if (cute::thread0()) { print_tensor(rS(_, m, k)); }
                    }
                    // 将 rS(_, m, k) 复制到 D(_, m, k)
                    cute::copy(rS(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    // 如果 Clear_OOB_K 为真，则清空 D(_, m, k)
                    cute::clear(D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace pytorch_flash
```