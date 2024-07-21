# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\softmax.h`

```py
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include <ATen/native/transformers/cuda/flash_attn/philox.cuh>
#include <ATen/native/transformers/cuda/flash_attn/utils.h>

namespace pytorch_flash {

using namespace cute;

#define UNFUSE_FMA
////////////////////////////////////////////////////////////////////////////////////////////////////

// 函数模板：thread_reduce_
// 实现线程内的张量归约操作，将二维张量tensor的每一行归约为一维张量summary
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        // 初始化或更新summary(mi)为第一个元素的值
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            // 继续使用op对summary(mi)与tensor(mi, ni)进行归约操作
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

// 函数模板：quad_allreduce_
// 实现四路全局归约操作，将src中的数据进行四路归约，并存储到dst中
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        // 使用Allreduce对src(i)进行四路归约操作，结果存入dst(i)
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

// 函数模板：reduce_
// 实现张量tensor的归约操作，包括线程内归约和四路全局归约
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    // 调用thread_reduce_进行线程内归约
    thread_reduce_<zero_init>(tensor, summary, op);
    // 调用quad_allreduce_进行四路全局归约
    quad_allreduce_(summary, summary, op);
}

// 函数模板：reduce_max
// 实现对张量tensor的最大值归约操作
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    // 使用MaxOp对tensor进行归约操作
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

// 函数模板：reduce_sum
// 实现对张量tensor的求和归约操作
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    // 使用SumOp对tensor进行归约操作
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

// 函数模板：scale_apply_exp2
// 对张量tensor的所有元素应用指数函数exp，并可选地按照max缩放
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    // 断言确保 Layout0 是二维张量，否则抛出错误信息
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    // 断言确保 Layout1 是一维张量，否则抛出错误信息
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    // 静态断言检查 max 张量的第一维度大小与 tensor 张量的第一维度大小相同
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    // 使用 #pragma unroll 告诉编译器对接下来的循环进行展开优化
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // 如果 max(mi) 是 -inf，则 max_scaled 设为 0，否则根据 Scale_max 条件选择乘以 scale 或者乘以 M_LOG2E 的结果
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        // 使用 #pragma unroll 告诉编译器对接下来的循环进行展开优化
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // 如果定义了 UNFUSE_FMA，使用 __fmul_rn 函数计算 tensor(mi, ni) 的值
            // 否则直接计算 tensor(mi, ni) 的值
            tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
        }
    }
}

// 结束函数模板定义的右花括号

// Apply the exp to all the elements.
template <bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
    // 静态断言，确保 Layout0 是二维张量
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    // 静态断言，确保 Layout1 是一维张量
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    // 断言 max 张量的第一维大小与 tensor 张量的第一维大小相同
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    // 循环遍历 tensor 张量的第一维
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // 创建 MaxOp 对象，用于计算最大值
        MaxOp<float> max_op;
        // 初始化 max(mi) 为 tensor(mi, 0) 或者使用 max_op 操作后的结果
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        // 循环遍历 tensor 张量的第二维，计算每行的最大值
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        // 对计算得到的最大值应用 Allreduce 操作
        max(mi) = Allreduce<4>::run(max(mi), max_op);

        // 如果 max(mi) 为 -INFINITY，则将 max_scaled 设为 0；否则乘以 scale 得到 max_scaled
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
        // 初始化 sum(mi) 为 0
        sum(mi) = 0;
        // 循环遍历 tensor 张量的第二维，计算 exp2f(tensor(mi, ni) * scale - max_scaled) 并求和
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // 替代计算 exp(x - max)，使用 exp2(x * log_2(e) - max * log_2(e))，利用 ffma 指令优化性能
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            sum(mi) += tensor(mi, ni);
        }
        // 创建 SumOp 对象，用于求和操作
        SumOp<float> sum_op;
        // 对计算得到的 sum(mi) 应用 Allreduce 操作
        sum(mi) = Allreduce<4>::run(sum(mi), sum_op);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {

    // 定义 TensorT 类型为 float 类型的张量
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    // 定义 row_max 和 row_sum 为 TensorT 类型的张量
    TensorT row_max, row_sum;

    // 默认构造函数
    __forceinline__ __device__ Softmax() {};

    // 模板函数，用于执行 Softmax 操作
    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    // 强制内联和设备函数修饰符，用于声明在 GPU 设备上执行且强制内联的函数
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        // 重新构造 acc_s 张量，从 (MMA=4, MMA_M, MMA_N) 改为 (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), pytorch_flash::convert_layout_acc_rowcol(acc_s.layout()));
        // 确保 scores 张量的第一个维度大小与 kNRows 相等
        static_assert(decltype(size<0>(scores))::value == kNRows);
        // 如果是第一次执行
        if (Is_first) {
            // 对 scores 执行最大值约减操作，结果存储在 row_max 中
            pytorch_flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            // 对 scores 执行指数函数的缩放操作，基数为 2 的 softmax_scale_log2 次幂
            pytorch_flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // 对 scores 执行求和操作，结果存储在 row_sum 中
            pytorch_flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            // 创建一个与 row_max 相同大小的张量 scores_max_prev，并将 row_max 的内容复制到 scores_max_prev 中
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            // 对 scores 执行最大值约减操作，结果存储在 row_max 中，不初始化为零
            pytorch_flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            // 重新构造 acc_o 张量，从 (MMA=4, MMA_M, MMA_K) 改为 (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), pytorch_flash::convert_layout_acc_rowcol(acc_o.layout()));
            // 确保 acc_o_rowcol 张量的第一个维度大小与 kNRows 相等
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            // 对每个 mi 进行循环
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                // 计算当前 mi 下的 scores 最大值，如果 Check_inf 为真，则处理特殊情况
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                // 计算 scores 的缩放因子，基数为 2 的 softmax_scale_log2 次幂
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                // 更新 row_sum(mi) 的值
                row_sum(mi) *= scores_scale;
                // 对 acc_o_rowcol 的每个 ni 进行循环，更新其值
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            // 对 scores 执行指数函数的缩放操作，基数为 2 的 softmax_scale_log2 次幂
            pytorch_flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // 在这里不跨线程执行归一化 softmax 所需的 reduce 操作，因为不需要使用 row_sum
            // 我们在需要归一化 softmax 时才执行该 reduce 操作
            pytorch_flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };
    // 使用 __forceinline__ 和 __device__ 声明强制内联和设备函数，这些通常用于 CUDA 编程
    TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        // 创建 SumOp<float> 对象，用于执行求和操作
        SumOp<float> sum_op;
        // 调用 quad_allreduce_ 函数，将 row_sum 数据进行全局四路归约，使用 sum_op 操作符
        quad_allreduce_(row_sum, row_sum, sum_op);
        // 创建 lse 张量，与 row_sum 具有相同的片段形状
        TensorT lse = make_fragment_like(row_sum);
        // 创建 acc_o_rowcol 张量，使用 acc_o 的数据和按照 pytorch_flash::convert_layout_acc_rowcol 转换后的布局
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), pytorch_flash::convert_layout_acc_rowcol(acc_o.layout()));
        // 静态断言，确保 acc_o_rowcol 的第一个维度大小等于 kNRows
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        // 循环遍历 acc_o_rowcol 的第一个维度 mi
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            // 获取 row_sum 的第 mi 个元素值，用于后续计算
            float sum = row_sum(mi);
            // 计算 sum 的倒数 inv_sum，处理 sum 为 0 或 NaN 的情况
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            // 计算 lse 的第 mi 个元素值，处理 sum 为 0 或 NaN 的情况，使用 softmax_scale 和 __logf(sum)
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            // 根据 Is_dropout 判断是否应用 dropout，计算 scale 值
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            // 循环遍历 acc_o_rowcol 的第二个维度 ni
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                // 将 acc_o_rowcol 的第 mi 行第 ni 列元素乘以 scale
                acc_o_rowcol(mi, ni) *= scale;
            }
        }
        // 返回 lse 张量
        return lse;
    };
};

}  // namespace pytorch_flash
```