# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\alibi.h`

```py
// 包含cmath标准库，提供数学函数支持
#include <cmath>

// 引入cute库中的tensor模块
#include <cute/tensor.hpp>

// 引入cutlass库的头文件和数组定义
#include <cutlass/cutlass.h>
#include <cutlass/array.h>

// 引入ATen库中用于CUDA的Transformer模块下的flash_attn工具函数
#include <ATen/native/transformers/cuda/flash_attn/utils.h>

// 声明pytorch_flash命名空间，用于封装当前代码
namespace pytorch_flash {

// 使用cute命名空间下的tensor类
using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Alibi模板结构体，根据Is_causal模板参数选择性应用偏置
template <bool Is_causal>
struct Alibi {

    // 声明常量成员alibi_slope、max_seqlen_k、max_seqlen_q
    const float alibi_slope;
    const int max_seqlen_k, max_seqlen_q;

    // 构造函数，初始化Alibi对象的常量成员
    __forceinline__ __device__ Alibi(const float alibi_slope, const int max_seqlen_k, const int max_seqlen_q)
        : alibi_slope(alibi_slope)
        , max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };

    // 应用Alibi偏置的方法，根据Is_causal模板参数选择不同的操作路径
    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_alibi(Tensor<Engine, Layout> &tensor,
                                      const int col_idx_offset_,
                                      const int row_idx_offset,
                                      const int warp_row_stride) {
        // tensor具有形状 (ncol=(2, MMA_M), nrow=(2, MMA_N))
        static_assert(Layout::rank == 2, "Only support 2D Tensor");
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        
        // 如果Is_causal为真，则执行以下操作
        if constexpr (Is_causal) {  // Simpler, we add the same bias vector to all rows
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(tensor); ++mi) {
                        tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                    }
                }
            }
        } else {  // Bias depends on both row_idx and col_idx
            // 否则，偏置取决于row_idx和col_idx的组合
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor); ++i) {
                    const int row_idx = row_idx_base + i * 8;
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;
                            // 应用Alibi偏置计算公式
                            tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                        }
                    }
                }
            }
        }
    }

};

}  // namespace pytorch_flash
```