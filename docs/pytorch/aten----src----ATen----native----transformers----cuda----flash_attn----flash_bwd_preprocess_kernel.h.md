# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\flash_bwd_preprocess_kernel.h`

```
/***********************************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>  // 引入 cute 库中的 copy 算法

#include <cutlass/cutlass.h>  // 引入 cutlass 库的主头文件
#include <cutlass/array.h>    // 引入 cutlass 库中的数组定义
#include <cutlass/numeric_types.h>  // 引入 cutlass 库中的数值类型定义

#include <ATen/native/transformers/cuda/flash_attn/block_info.h>  // 引入 ATen 库中的块信息定义
#include <ATen/native/transformers/cuda/flash_attn/kernel_traits.h>  // 引入 ATen 库中的核特性定义
#include <ATen/native/transformers/cuda/flash_attn/utils.h>  // 引入 ATen 库中的实用工具函数

namespace pytorch_flash {

using namespace cute;  // 使用 cute 命名空间

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS_PER_ROW, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void dot_do_o(Tensor<Engine0, Layout0> const &do_, Tensor<Engine0, Layout0> const &o,
                                Tensor<Engine1, Layout1> &dP_sum, const int gdP_col_stride, const float scale) {
    static_assert(Layout0::rank == 3, "Only support 3D Tensor");  // 静态断言，确保 Layout0 是 3 维张量
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");  // 静态断言，确保 Layout1 是 1 维张量
    CUTE_STATIC_ASSERT_V(do_.layout() == o.layout());  // 断言确保 do_ 和 o 的布局相同
    
    // 重新整形 do_ 和 o，从 (8, kBlockM / 32, kHeadDim / 64) 到 (kBlockM / 32, 8 * kHeadDim / 64)
    // 最后一个坐标是 "page"
    Tensor do_reshaped = make_tensor(do_.data(), make_layout(get<1>(do_.layout()),
                                                             make_layout(get<0>(do_.layout()),
                                                                         get<2>(do_.layout()))));
    Tensor o_reshaped = make_tensor(o.data(), do_reshaped.layout());
    
    // 将 do_ 和 o 转换为 float 类型的张量
    Tensor do_fp32 = pytorch_flash::convert_type<float>(do_reshaped);
    Tensor o_fp32 = pytorch_flash::convert_type<float>(o_reshaped);
    
    #pragma unroll
    // 循环遍历 mi，计算 dP_sum_cur
    for (int mi = 0; mi < size<0>(do_reshaped); ++mi) {
        float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
        #pragma unroll
        // 循环遍历 ni，继续计算 dP_sum_cur
        for (int ni = 1; ni < size<1>(do_reshaped); ni++) {
            dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
        }
        pytorch_flash::SumOp<float> sum_op;
        // 运行 Allreduce 操作，并乘以 scale
        dP_sum_cur = pytorch_flash::Allreduce<THREADS_PER_ROW>::run(dP_sum_cur, sum_op) * scale;
        
        // 如果 threadIdx.x 是 THREADS_PER_ROW 的倍数，则写入 dP_sum
        if (threadIdx.x % THREADS_PER_ROW == 0) {
            dP_sum(mi * gdP_col_stride + threadIdx.x / THREADS_PER_ROW) = dP_sum_cur;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// 计算 dot(do, o) 并将结果 (softmax_d) 作为单独的核函数写入全局内存。
// 这在我们希望在 seqlen_k 跨向后向并行化的情况下使用。
template<bool Clear_dQaccum=true, typename Kernel_traits, typename Params>
inline __device__ void compute_dot_do_o(const Params &params) {
    using Element = typename Kernel_traits::Element;         // 使用 Kernel_traits 的 Element 类型
    using ElementAccum = typename Kernel_traits::ElementAccum;  // 使用 Kernel_traits 的 ElementAccum 类型
    // 使用别名定义 Kernel_traits 中的 index_t 类型
    using index_t = typename Kernel_traits::index_t;

    // 获取当前线程块在 x 轴上的索引，用于批处理的块索引
    const int m_block = blockIdx.x;
    // 获取当前线程块在 y 轴上的索引，用于头部的块索引
    const int bidb = blockIdx.y;
    // 获取当前线程块在 z 轴上的索引，用于线程索引
    const int bidh = blockIdx.z;
    // 获取当前线程在 x 轴上的索引，用于处理线程内部的并行
    const int tidx = threadIdx.x;

    // 获取常量 kBlockM，表示线程块中的 M 维度大小
    constexpr int kBlockM = Kernel_traits::kBlockM;
    // 获取常量 kHeadDim，表示头部维度大小
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    // 根据参数和块索引 bidb 创建 BlockInfo 对象 binfo
    const BlockInfo binfo(params, bidb);

    // 如果当前线程块对应的数据偏移超过了实际的序列长度 q，则直接返回
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // 计算 do（Q 维度输出）的行偏移量
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    // 计算 o（O 维度输出）的行偏移量
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    // 计算 dq_accum（Q 累积梯度）的行偏移量
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + (m_block * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded;
    // 计算 dP_sum（softmax 梯度和）的行偏移量
    const index_t row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM;

    // 创建 gdO 张量，表示 Q 维度输出的梯度
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    // 创建 gO 张量，表示 O 维度输出
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    // 创建 gdQaccum 张量，表示 Q 累积梯度
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.h * params.d_rounded, _1{}));
    // 创建 dP_sum 张量，表示 softmax 梯度和
    Tensor dP_sum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    // 创建 Kernel_traits 中的 GmemTiledCopydO 类型对象 gmem_tiled_copy_dO
    typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
    // 获取当前线程 tidx 的 gmem_tiled_copy_dO 切片
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);

    // 创建 Kernel_traits 中的 GmemTiledCopydQaccum 类型对象 gmem_tiled_copy_dQaccum
    typename Kernel_traits::GmemTiledCopydQaccum gmem_tiled_copy_dQaccum;
    // 获取当前线程 tidx 的 gmem_tiled_copy_dQaccum 切片
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    // 使用 gmem_thr_copy_dO 分区化 gdO 为 tdOgdO
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    // 使用 gmem_thr_copy_dO 分区化 gO 为 tdOgO
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    // 使用 gmem_thr_copy_dQaccum 分区化 gdQaccum 为 tdQgdQaccum
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    // 创建 cdO 张量，表示标识张量 (BLK_M, BLK_K) -> (blk_m, blk_k)
    Tensor cdO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // 对 gmem_thr_copy_dO 中的数据进行分区操作，返回分区后的 Tensor 对象 tdOcdO
    Tensor tdOcdO = gmem_thr_copy_dO.partition_S(cdO);

    // 为 k 分配谓词张量
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdOgdO)));
    
    // 设置 k 的边界谓词
    #pragma unroll
    for (int k = 0; k < size(tdOpdO); ++k) {
        // 根据 tdOcdO 中的数据设置 tdOpdO(k) 的谓词，判断条件为 get<1>(tdOcdO(0, 0, k)) < params.d
        tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d;
    }

    // 根据 tdOgdO 的形状创建与其类似的片段张量 tdOrdO 和 tdOrO
    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);

    // 使用 pytorch_flash::copy 函数复制数据
    pytorch_flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM
    );

    // 再次使用 pytorch_flash::copy 函数复制数据
    pytorch_flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgO, tdOrO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM
    );

    // 使用 dot_do_o 函数进行矩阵乘法操作，将结果放入 dP_sum 中
    dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, dP_sum,
                                                Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
    
    // 如果 Clear_dQaccum 为真，则执行以下操作
    if (Clear_dQaccum) {
        // 创建与 tdQgdQaccum 类似的零张量 zero，并将其清空
        Tensor zero = make_fragment_like(tdQgdQaccum);
        clear(zero);
        // 使用 cute::copy 函数将零张量复制到 tdQgdQaccum 中
        cute::copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Params>
inline __device__ void clear_dKVaccum(const Params &params) {
    // 定义类型别名，ElementAccum 是 Kernel_traits 的 ElementAccum 类型
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // 定义索引类型，index_t 是 Kernel_traits 的 index_t 类型
    using index_t = typename Kernel_traits::index_t;

    // 获取当前线程块的索引
    const int n_block = blockIdx.x;
    // 获取当前线程块在批次中的索引
    const int bidb = blockIdx.y;
    // 获取当前线程块在头部维度中的索引
    const int bidh = blockIdx.z;
    // 获取当前线程在线程块中的索引
    const int tidx = threadIdx.x;

    // 从 Kernel_traits 中获取常量 kBlockN 和 kHeadDim
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    // 根据 Params 和 bidb 创建 BlockInfo 对象 binfo
    const BlockInfo binfo(params, bidb);
    // 如果当前线程块处理的位置超过实际的序列长度，则返回
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    // 计算行偏移量，用于定位 dK/V 累积值在内存中的位置
    const index_t row_offset_dkv_accum = ((bidb * params.h_k + bidh) * params.seqlen_k_rounded + n_block * kBlockN) * params.d_rounded;

    // 创建 Tensor gdKaccum，指向 dK 累积值在全局内存中的位置
    Tensor gdKaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dk_accum_ptr) + row_offset_dkv_accum),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
    // 创建 Tensor gdVaccum，指向 dV 累积值在全局内存中的位置
    Tensor gdVaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dv_accum_ptr) + row_offset_dkv_accum),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});

    // 创建 Kernel_traits::GmemTiledCopydQaccum 对象 gmem_tiled_copy_dKVaccum
    typename Kernel_traits::GmemTiledCopydQaccum gmem_tiled_copy_dKVaccum;
    // 获取当前线程的切片 gmem_thr_copy_dKVaccum
    auto gmem_thr_copy_dKVaccum = gmem_tiled_copy_dKVaccum.get_thread_slice(tidx);
    // 分区化 dKgdKaccum 和 dVgdVaccum
    Tensor tdKgdKaccum = gmem_thr_copy_dKVaccum.partition_D(gdKaccum);
    Tensor tdVgdVaccum = gmem_thr_copy_dKVaccum.partition_D(gdVaccum);
    // 创建与 tdKgdKaccum 相似的 zero Tensor
    Tensor zero = make_fragment_like(tdKgdKaccum);
    // 清空 zero Tensor
    clear(zero);
    // 将 zero Tensor 复制到 tdKgdKaccum 和 tdVgdVaccum
    cute::copy(gmem_tiled_copy_dKVaccum, zero, tdKgdKaccum);
    cute::copy(gmem_tiled_copy_dKVaccum, zero, tdVgdVaccum);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// 将 dQ 从 dQaccum（float 类型）转换为 fp16/bf16 类型
// 当我们想要跨 seqlen_k 并行化向后传播时使用
template<typename Kernel_traits, typename Params>
inline __device__ void convert_dQ(const Params &params, const int nsplits) {
    // 定义类型别名，Element 是 Kernel_traits 的 Element 类型，ElementAccum 是 Kernel_traits 的 ElementAccum 类型
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // 定义索引类型，index_t 是 Kernel_traits 的 index_t 类型
    using index_t = typename Kernel_traits::index_t;

    // 定义共享内存
    extern __shared__ char smem_[];

    // 获取当前线程块的索引
    const int m_block = blockIdx.x;
    // 获取当前线程块在批次中的索引
    const int bidb = blockIdx.y;
    // 获取当前线程块在头部维度中的索引
    const int bidh = blockIdx.z;
    // 获取当前线程在线程块中的索引
    const int tidx = threadIdx.x;

    // 从 Kernel_traits 中获取常量 kBlockM 和 kHeadDim
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    // 根据 Params 和 bidb 创建 BlockInfo 对象 binfo
    const BlockInfo binfo(params, bidb);
    // 如果当前线程块处理的位置超过实际的序列长度，则返回
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    // 计算行偏移量，用于访问dq张量中的数据
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    // 计算累积行偏移量，用于访问dq_accum张量中的数据
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + (m_block * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded;

    // 创建gdQ张量，表示dq张量的全局内存视图
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));
    // 创建gdQaccum张量，表示dq_accum张量的全局内存视图
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.h * params.d_rounded, _1{}));

    // 创建sdQ张量，表示dq张量的共享内存视图
    Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                             typename Kernel_traits::SmemLayoutdQ{});

    // 定义全局内存tiled拷贝操作对象，并获取当前线程的片段
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    // 定义全局内存tiled累加拷贝操作对象，并获取当前线程的片段
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    // 定义tiled mma操作对象，并创建共享内存tiled拷贝dQ操作
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    // 将sdQ张量划分为小片段，用于后续操作
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // 将sdQ张量划分为更小片段，用于全局内存拷贝dQ操作
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    // 将gdQ张量划分为小片段，用于全局内存拷贝dQ操作
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    // 将gdQaccum张量划分为更小片段，用于全局内存累加拷贝dQ_accum操作
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum);

    // 根据tiled mma操作对象的形状，将acc_dq张量划分为小片段
    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));

    // 创建一个与tdQgdQaccum形状相似的tdQrdQaccum张量
    Tensor tdQrdQaccum = make_fragment_like(tdQgdQaccum);
    // 清空acc_dq张量
    clear(acc_dq);
    // 遍历每个分片，从tdQgdQaccum拷贝数据到tdQrdQaccum，并累加到acc_dq
    for (int s = 0; s < nsplits; ++s) {
        cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, tdQrdQaccum);
        #pragma unroll
        for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) += tdQrdQaccum(i); }
        tdQgdQaccum.data() = tdQgdQaccum.data() + params.dq_accum_split_stride;
    }
    // 对acc_dq张量的每个元素乘以params.scale_softmax_rp_dropout
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // 将acc_dq张量从fp32转换为fp16
    Tensor rdQ = pytorch_flash::convert_type<Element>(acc_dq);
    // 将rdQ张量划分为更小片段，用于共享内存tiled拷贝dQ操作
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);  // ((Atom,AtomNum), MMA_N, MMA_N)
    // 使用共享内存tiled拷贝dQ操作，从taccdQrdQ拷贝数据到taccdQsdQ
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    // 同步所有线程
    __syncthreads();
    // 创建tdQrdQ张量，形状与tdQgdQ相同的元素张量
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    # 使用 cute 命令进行复制操作，将 gmem_tiled_copy_dQ 中的数据从 tdQsdQ 复制到 tdQrdQ。
    
    Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // 创建一个形状为 (BLK_M, BLK_K) 的单位张量 cdQ
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);  // 将 cdQ 张量按维度 D 进行划分，得到 tdQcdQ 张量
    
    // 创建一个形状为 (2,) 的布尔张量 tdQpdQ
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    
    #pragma unroll
    // 对 tdQpdQ 中的每个元素进行迭代
    for (int k = 0; k < size(tdQpdQ); ++k) {
        // 将 tdQcdQ(0, 0, k) 的第一个分量与 params.d 比较，将比较结果存储到 tdQpdQ(k) 中
        tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d;
    }
    
    // 调用 pytorch_flash::copy 函数进行复制操作，不清空超出界限的部分，并传入多个参数
    pytorch_flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
// Convert dK and dV from dKaccum and dVaccum (in float) to fp16/bf16.
// This is used in the case where we want to parallelize the backward across seqlen_q.
template<typename Kernel_traits, typename Params>
inline __device__ void convert_dKV(const Params &params) {
    // Define the types used in the kernel from Kernel_traits
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory allocation for temporary storage
    extern __shared__ char smem_[];

    // Block indices
    const int n_block = blockIdx.x;  // Index of the current block within the grid
    const int bidb = blockIdx.y;     // Index of the batch block within the grid
    const int bidh = blockIdx.z;     // Index of the head block within the grid
    const int tidx = threadIdx.x;    // Thread index within the block

    // Constants defined by Kernel_traits
    constexpr int kBlockN = Kernel_traits::kBlockN;   // Number of elements per block
    constexpr int kHeadDim = Kernel_traits::kHeadDim; // Dimensionality of the head

    // Compute the BlockInfo for the current batch block
    const BlockInfo binfo(params, bidb);

    // Check if the current block is out of bounds with respect to actual_seqlen_k
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    // Compute offsets into the global memory for dK, dV, and dKaccum/dVaccum
    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    const index_t row_offset_dkv_accum = ((bidb * params.h_k + bidh) * params.seqlen_k_rounded
                                          + n_block * kBlockN) * params.d_rounded;

    // Create Tensor objects for dK, dV, dKaccum, dVaccum in global memory
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dv_row_stride, _1{}));
    Tensor gdKaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dk_accum_ptr) + row_offset_dkv_accum),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  Stride<Int<kHeadDim>, _1>{});
    Tensor gdVaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dv_accum_ptr) + row_offset_dkv_accum),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  Stride<Int<kHeadDim>, _1>{});

    // Create Tensor objects for dK and dV in shared memory
    Tensor sdK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                             typename Kernel_traits::SmemLayoutdKV{});
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Initialize the gmem_tiled_copy_dKV object from Kernel_traits
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dKV;
}
    // 获取当前线程的 gmem_tiled_copy_dKV 对象的线程切片
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);

    // 定义 gmem_tiled_copy_dKVaccum 对象，用于原子累加操作
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dKVaccum;
    
    // 获取当前线程的 gmem_tiled_copy_dKVaccum 对象的线程切片
    auto gmem_thr_copy_dKVaccum = gmem_tiled_copy_dKVaccum.get_thread_slice(tidx);

    // 定义 tiled_mma_dkv 对象，用于矩阵乘加操作
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    
    // 创建 smem_tiled_copy_dKV 对象，使用 tiled_mma_dkv 对象和 SmemCopyAtomdKV 类型
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    
    // 获取当前线程的 smem_tiled_copy_dKV 对象的线程切片
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    
    // 对 sdK 和 sdV 张量进行分区，得到 taccdKsdK 和 taccdVsdV
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);  // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // 对 sdK 和 sdV 张量进行分区，得到 tdKsdK 和 tdVsdV
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
    Tensor tdKgdKaccum = gmem_thr_copy_dKVaccum.partition_S(gdKaccum);
    Tensor tdVgdVaccum = gmem_thr_copy_dKVaccum.partition_S(gdVaccum);

    // 根据 tiled_mma_dkv 对象对 acc_dk 和 acc_dv 进行分区
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    
    // 静态断言，验证 acc_dk 和 tdKgdKaccum 的大小是否相等
    CUTE_STATIC_ASSERT_V(size(acc_dk) == size(tdKgdKaccum));
    CUTE_STATIC_ASSERT_V(size(acc_dv) == size(tdVgdVaccum));

    // 根据 tdKgdKaccum 和 tdVgdVaccum 创建与之形状相同的张量
    Tensor tdKrdKaccum = make_fragment_like(tdKgdKaccum);
    Tensor tdVrdVaccum = make_fragment_like(tdVgdVaccum);
    
    // 使用 cute::copy 函数将 tdKgdKaccum 和 tdVgdVaccum 复制到 tdKrdKaccum 和 tdVrdVaccum 中
    cute::copy(gmem_tiled_copy_dKVaccum, tdKgdKaccum, tdKrdKaccum);
    cute::copy(gmem_tiled_copy_dKVaccum, tdVgdVaccum, tdVrdVaccum);
    
    // 对 acc_dk 和 acc_dv 中的元素进行缩放操作
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) {
        acc_dk(i) = tdKrdKaccum(i) * params.scale_softmax_rp_dropout;
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dv); ++i) {
        acc_dv(i) = tdVrdVaccum(i) * params.rp_dropout;
    }

    // 将 acc_dk 和 acc_dv 转换为 Element 类型的张量 rdK 和 rdV
    Tensor rdK = pytorch_flash::convert_type<Element>(acc_dk);
    Tensor rdV = pytorch_flash::convert_type<Element>(acc_dv);
    
    // 根据 rdK 和 rdV 对 smem_thr_copy_dKV 对象中的 taccdKsdK 和 taccdVsdV 进行重排列
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);  // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);  // ((Atom,AtomNum), MMA_N, MMA_N)
    
    // 使用 cute::copy 函数将 taccdKrdK 和 taccdVrdV 复制到 taccdKsdK 和 taccdVsdV 中
    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    
    // 同步所有线程
    __syncthreads();
    
    // 创建与 tdKgdK 形状相同的 tdKrdK 和与 tdVgdV 形状相同的 tdVrdV
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    
    // 使用 cute::copy 函数将 tdKsdK 和 tdVsdV 复制到 tdKrdK 和 tdVrdV 中
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);

    // 创建与 Shape<Int<kBlockN>, Int<kHeadDim>>{} 相同形状的 cdKV 张量
    Tensor cdKV = make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    
    // 对 gmem_thr_copy_dKV 中的 cdKV 进行分区
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    
    // 创建与 tdKgdKaccum 形状相同的布尔型张量 tdKVpdKV
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));

    // 对 tdKgdKaccum 和 tdVgdVaccum 进行 unroll 处理
    #pragma unroll
    // 遍历 tdKVpdKV 数组，根据条件设定每个元素的值
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // 设置 Clear_OOB_K 为 false，因为不希望将零值写入 gmem
    pytorch_flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        // 调用 pytorch_flash::copy 函数，从 tdKrdK, tdKgdK 读取数据写入 gmem_tiled_copy_dKV
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    pytorch_flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        // 再次调用 pytorch_flash::copy 函数，从 tdVrdV, tdVgdV 读取数据写入 gmem_tiled_copy_dKV
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
}

// 结束命名空间 "flash"
} // namespace flash
```