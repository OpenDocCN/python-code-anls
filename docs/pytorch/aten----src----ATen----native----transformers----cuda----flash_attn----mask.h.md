# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\mask.h`

```
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>  // 引入 cute 库中的 tensor 头文件

namespace pytorch_flash {

using namespace cute;  // 使用 cute 命名空间中的定义

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;  // 计算当前线程在 warp 内的 ID
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;  // 根据线程 ID 计算列索引偏移量
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;  // 根据列索引偏移量计算基础列索引
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;  // 计算最终列索引
            if (col_idx >= max_seqlen_k) {  // 如果列索引超出最大序列长度
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;  // 将对应位置的 tensor 元素置为负无穷
                }
            }
        }
    }
}

template <bool HasWSLeft=true, typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_local(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset,
                                        const int max_seqlen_q, const int warp_row_stride,
                                        const int window_size_left, const int window_size_right) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;  // 计算当前线程在 warp 内的 ID
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;  // 根据线程 ID 计算列索引偏移量
    #pragma unroll
    // 对 tensor 的第一个维度进行迭代，每次迭代增加 warp_row_stride 的步长
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        // 计算当前行的起始索引
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        // 使用 #pragma unroll 指令，展开循环，优化性能
        #pragma unroll
        // 对 tensor 的第二个维度进行迭代
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            // 计算当前行的索引
            const int row_idx = row_idx_base + i * 8;
            // 计算当前列的左边界索引，确保不超出边界
            const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
            // 计算当前列的右边界索引，确保不超出边界
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
            // 使用 #pragma unroll 指令，展开循环，优化性能
            #pragma unroll
            // 对 tensor 的第三个维度进行迭代
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                // 计算当前列的起始索引
                const int col_idx_base = col_idx_offset + nj * 8;
                // 使用 #pragma unroll 指令，展开循环，优化性能
                #pragma unroll
                // 对 tensor 的第四个维度进行迭代
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    // 计算当前列的索引
                    const int col_idx = col_idx_base + j;
                    // 检查当前索引是否在有效范围内，若不在则将对应元素置为负无穷
                    if (col_idx >= col_idx_limit_right || (HasWSLeft && col_idx < col_idx_limit_left)) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
            // 下面的代码段是注释掉的调试信息输出，用于显示特定变量的值和 tensor 的部分内容
            // if (cute::thread0()) {
            //     printf("mi = %d, i = %d, row_idx = %d, max_seqlen_k = %d\n", mi, i, row_idx, max_seqlen_k);
            //     print(tensor(make_coord(i, mi), _));
            //     // print(tensor(_, j + nj * size<1, 0>(tensor)));
            // }
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_causal(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                         const int max_seqlen_k, const int row_idx_offset,
                                         const int max_seqlen_q, const int warp_row_stride) {
    // Causal masking is equivalent to local masking with window_size_left = infinity and window_size_right = 0
    // 调用 apply_mask_local 函数进行因果遮蔽，左窗口大小为无穷大，右窗口大小为0
    apply_mask_local</*HasWSLeft=*/false>(tensor, col_idx_offset_, max_seqlen_k, row_idx_offset,
                                          max_seqlen_q, warp_row_stride, -1, 0);
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void apply_mask_causal_w_idx(
    Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &idx_rowcol,
    const int col_idx_offset_, const int max_seqlen_k, const int row_idx_offset)
{
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    // 断言 tensor 和 idx_rowcol 是二维张量
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 2, "Only support 2D Tensor");
    // 静态断言，确保 tensor 和 idx_rowcol 的尺寸匹配
    CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(idx_rowcol));
    CUTE_STATIC_ASSERT_V(size<1>(tensor) == size<1>(idx_rowcol));
    // 对 tensor 进行遍历，根据 idx_rowcol 的值进行因果遮蔽
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // 计算列索引限制，确保不超过 max_seqlen_k，与 row_idx_offset 和 idx_rowcol 的值有关
        const int col_idx_limit = std::min(max_seqlen_k, 1 + row_idx_offset + get<0>(idx_rowcol(mi, 0)));
        // 对每个 ni 进行遍历
        #pragma unroll
        for (int ni = 0; ni < size<1, 1>(tensor); ++ni) {
            // 如果列索引偏移量加上 idx_rowcol 的值超过 col_idx_limit，则设定 tensor 对应位置的值为负无穷
            if (col_idx_offset_ + get<1>(idx_rowcol(0, ni)) >= col_idx_limit) {
                tensor(mi, ni) = -INFINITY;
            }
        }
        // 如果是线程0，则打印相关信息，用于调试
        // if (cute::thread0()) {
        //     printf("ni = %d, j = %d, col_idx = %d, max_seqlen_k = %d\n", ni, j, col_idx, max_seqlen_k);
        //     print(tensor(_, make_coord(j, ni)));
        //     // print(tensor(_, j + ni * size<1, 0>(tensor)));
        // }
    }
}

template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q,
                                    const int window_size_left, const int window_size_right,
                                    const float alibi_slope=0.f)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    };

};

}  // namespace pytorch_flash


这段代码主要涉及了在 CUDA 设备上实现的因果遮蔽（causal masking）功能，对张量进行局部遮蔽处理。
```