# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\flash_fwd_kernel.h`

```
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>  // 包含 copy 算法的头文件

#include <cutlass/cutlass.h>  // 包含 Cutlass 库的头文件
#include <cutlass/array.h>    // 包含 Cutlass 数组定义的头文件
#include <cutlass/numeric_types.h>  // 包含 Cutlass 数值类型定义的头文件


#include <ATen/native/transformers/cuda/flash_attn/block_info.h>  // 包含 block_info.h 头文件
#include <ATen/native/transformers/cuda/flash_attn/kernel_traits.h>  // 包含 kernel_traits.h 头文件
#include <ATen/native/transformers/cuda/flash_attn/utils.h>  // 包含 utils.h 头文件
#include <ATen/native/transformers/cuda/flash_attn/softmax.h>  // 包含 softmax.h 头文件
#include <ATen/native/transformers/cuda/flash_attn/mask.h>  // 包含 mask.h 头文件
#include <ATen/native/transformers/cuda/flash_attn/dropout.h>  // 包含 dropout.h 头文件
#include <ATen/native/transformers/cuda/flash_attn/rotary.h>  // 包含 rotary.h 头文件

namespace pytorch_flash {

using namespace cute;  // 使用 cute 命名空间

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;  // 定义元素类型 Element
    using ElementAccum = typename Kernel_traits::ElementAccum;  // 定义累加元素类型 ElementAccum
    using index_t = typename Kernel_traits::index_t;  // 定义索引类型 index_t

    // Shared memory.
    extern __shared__ char smem_[];  // 声明共享内存 smem_

    // The thread index.
    const int tidx = threadIdx.x;  // 获取线程在块内的索引

    constexpr int kBlockM = Kernel_traits::kBlockM;  // 定义常量 kBlockM
    constexpr int kBlockN = Kernel_traits::kBlockN;  // 定义常量 kBlockN
    constexpr int kHeadDim = Kernel_traits::kHeadDim;  // 定义常量 kHeadDim
    constexpr int kNWarps = Kernel_traits::kNWarps;  // 定义常量 kNWarps

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);  // 解包 philox_args 获得种子偏移量
    pytorch_flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);  // 初始化 Dropout 类对象 dropout

    // Save seed and offset for backward. If we don't have this here, the 0-th thread block might
    // exit early and no one saves the rng state.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        if (params.philox_args.captured_) {
            *params.seed = std::get<0>(seed_offset);  // 保存种子到 params.seed
            *params.extragraph_offset = std::get<1>(seed_offset);  // 保存偏移量到 params.extragraph_offset
        }
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);  // 根据参数和 bidb 创建 BlockInfo 对象 binfo
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;  // 如果 m_block * kBlockM 大于等于 binfo.actual_seqlen_q，则返回

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);  // 计算 n_block_max
    // 如果 Is_causal 或 Is_local 为真，则执行以下代码块
    if (Is_causal || Is_local) {
        // 计算 n_block_max 的值，限制在 n_block_max 和 ((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right) / kBlockN 的较小值
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // 如果以下条件成立，则输出调试信息
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    
    // 提前退出并将 gO 和 gLSE 写为 0。这也包括 actual_seqlen_k == 0 的情况。
    // 否则可能会读取超出范围的 gK 和 gV 元素。
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        // 计算 gO 的行偏移量
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        // 计算 gLSE 的行偏移量
        const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        
        // 创建 gO 张量，表示输出的梯度
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.o_row_stride, _1{}));
        
        // 创建 gLSE 张量，表示 softmax 操作的 log sum exp 输出
        Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                                  Shape<Int<kBlockM>>{}, Stride<_1>{});

        // 定义一个适用于当前设备的 gmem_tiled_copy_O 对象
        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        // 获取当前线程的 gmem_tiled_copy_O 的切片
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        // 对 gO 进行分区，得到 tOgO 张量
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        // 创建一个形状与 tOgO 相同的 tOrO 张量，并将其清零
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        
        // 构建一个与 gO 形状相同的 identity layout 张量 cO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // 使用 identity layout 对 gmem_tiled_copy_O 进行再次分区
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        // 创建一个布尔类型的 tOpO 张量，形状与 tOgO 的第二维度大小相同
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        
        // 如果 Is_even_K 为假，则执行以下循环
        if (!Is_even_K) {
            #pragma unroll
            // 遍历 tOpO 张量的每个元素
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        
        // 执行从 tOgO 到 tOrO 的拷贝操作，条件是 Clear_OOB_MN 为假，Clear_OOB_K 也为假
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        
        // 遍历 tOgO 的第一维度
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            // 获取 tOcO 中的第一维度索引
            const int row = get<0>(tOcO(0, m, 0));
            // 如果 row 小于 binfo.actual_seqlen_q - m_block * kBlockM，并且 tOcO 的第二维度索引为 0，则将 gLSE(row) 设置为无穷大
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        // 函数返回，结束当前执行路径
        return;
    }
    
    // 如果 tidx 等于 0，则输出调试信息
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // 我们以逆序迭代块。这是因为最后一个块是唯一一个
    // 根据 binfo 和 params 的信息计算 Q 张量在全局内存中的偏移量
    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    // 将 K 张量移动到最后一个块的位置
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    // 将 V 张量移动到最后一个块的位置
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    // 计算 P 张量在全局内存中的偏移量
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    // 创建全局内存中 Q 张量的 Tensor 对象
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    // 创建全局内存中 K 张量的 Tensor 对象
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // 创建全局内存中 V 张量的 Tensor 对象
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    // 创建全局内存中 P 张量的 Tensor 对象
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    // 创建共享内存中 Q 张量的 Tensor 对象
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // 如果 Share_Q_K_smem 为真，sK 和 sV 与 sQ 共享同一块共享内存；否则，它们紧接着 sQ 后面
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    // 创建用于转置后的 V 张量的 Tensor 对象
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // 创建未混洗的 V 张量的 Tensor 对象
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // 创建用于全局内存到共享内存的 QKV 拷贝的对象
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    // 获取当前线程的 Q 张量的分片
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    // 将全局内存中的 Q 张量划分为适合拷贝的片段
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    // 将共享内存中的 Q 张量划分为适合拷贝的片段
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    // 将全局内存中的 K 张量划分为适合拷贝的片段
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    // 使用 gmem_thr_copy_QKV 对象的 partition_D 方法，将 sK 进行分区
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    // 使用 gmem_thr_copy_QKV 对象的 partition_S 方法，将 gV 进行分区，标注为 (VCPY, VCPY_N, VCPY_K)
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    // 使用 gmem_thr_copy_QKV 对象的 partition_D 方法，将 sV 进行分区
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // 根据 Kernel_traits 的定义选择 tiled_mma 的类型，并获取线程 tidx 对应的切片
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    // 使用 thr_mma 对象的 partition_fragment_A 方法，将 sQ 进行分区，标注为 (MMA, MMA_M, MMA_K)
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    // 使用 thr_mma 对象的 partition_fragment_B 方法，将 sK 进行分区，标注为 (MMA, MMA_N, MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);
    // 使用 thr_mma 对象的 partition_fragment_B 方法，将 sVtNoSwizzle 进行分区，标注为 (MMA, MMA_K, MMA_N)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);

    // 使用 thr_mma 对象的 partition_C 方法，将 gP 进行分区
    Tensor tSgS  = thr_mma.partition_C(gP);

    // 调用 tiled_mma 对象的 partition_fragment_C 方法，使用 Shape<Int<kBlockM>, Int<kHeadDim>> 进行分区，标注为 (MMA, MMA_M, MMA_K)
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    //
    // Copy Atom retiling
    //

    // 使用 make_tiled_copy_A 方法创建 tiled_mma 的类型为 SmemCopyAtom 的 smem_tiled_copy_Q 对象
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    // 获取 smem_tiled_copy_Q 对象的线程 tidx 对应的切片
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // 使用 smem_thr_copy_Q 对象的 partition_S 方法，将 sQ 进行分区
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    // 使用 make_tiled_copy_B 方法创建 tiled_mma 的类型为 SmemCopyAtom 的 smem_tiled_copy_K 对象
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    // 获取 smem_tiled_copy_K 对象的线程 tidx 对应的切片
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    // 使用 smem_thr_copy_K 对象的 partition_S 方法，将 sK 进行分区
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    // 使用 make_tiled_copy_B 方法创建 tiled_mma 的类型为 SmemCopyAtomTransposed 的 smem_tiled_copy_V 对象
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    // 获取 smem_tiled_copy_V 对象的线程 tidx 对应的切片
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    // 使用 smem_thr_copy_V 对象的 partition_S 方法，将 sVt 进行分区
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // 构造 tQpQ 和 tKVpKV 作为布尔类型的张量，用于大小为 tQsQ 和 tKsK 的索引
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // 使用 make_identity_tensor 创建 cQ 和 cKV 张量，分别标注为 (BLK_M, BLK_K) 和 (BLK_N, BLK_K)
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // 使用 gmem_thr_copy_QKV 对象的 partition_S 方法，将 cQ 进行分区，标注为 (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    // 使用 gmem_thr_copy_QKV 对象的 partition_S 方法，将 cKV 进行分区，标注为 (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    // 分配大小为 tQsQ 的布尔类型张量，用于 k 的预测
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    // 创建一个名为 tKVpKV 的布尔型张量，形状与 tKsK 相同
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // 根据是否是偶数 K 来设置 k 的边界条件
    if (!Is_even_K) {
        // 如果 K 不是偶数，则设置 tQpQ 中的谓词
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        // 如果 K 不是偶数，则设置 tKVpKV 中的谓词
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue（序言部分）

    // 不需要清除 sQ smem 块，因为只会写入有效输出
    pytorch_flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    // 如果 Kernel_traits::Is_Q_in_regs 为真，则执行内存复制操作的异步屏障
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    // 如果 Kernel_traits::Share_Q_K_smem 为真，则进行以下操作
    if (Kernel_traits::Share_Q_K_smem) {
        // 执行异步等待操作，等待之前的内存复制完成
        pytorch_flash::cp_async_wait<0>();
        // 同步所有线程，等待之前的操作完成
        __syncthreads();
        // 创建 tSrQ_copy_view，表示 smem_thr_copy_Q 重塑后的视图
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        // 断言 tSsQ 的大小等于 tSrQ_copy_view 的大小
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        // 将数据从 tSsQ 复制到 smem_tiled_copy_Q
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        // 同步所有线程，确保复制操作完成
        __syncthreads();
    }

    // 设置 n_block 的初始值为 n_block_max - 1
    int n_block = n_block_max - 1;
    // 不需要清除 sK smem 块，因为会屏蔽掉分数
    pytorch_flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    // 执行内存复制操作的异步屏障
    cute::cp_async_fence();

    // 如果 Kernel_traits::Is_Q_in_regs 为真且 Kernel_traits::Share_Q_K_smem 为假，则执行以下操作
    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        // 执行异步等待操作，等待之前的内存复制完成
        pytorch_flash::cp_async_wait<1>();
        // 同步所有线程，等待之前的操作完成
        __syncthreads();
        // 创建 tSrQ_copy_view，表示 smem_thr_copy_Q 重塑后的视图
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        // 断言 tSsQ 的大小等于 tSrQ_copy_view 的大小
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        // 将数据从 tSsQ 复制到 smem_tiled_copy_Q
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    // 清空 acc_o 张量
    clear(acc_o);

    // 创建一个大小为 2*size<1>(acc_o) 的 Softmax 对象
    pytorch_flash::Softmax<2 * size<1>(acc_o)> softmax;

    // 计算 alibi_slope 的值，如果不需要 alibi 或者 alibi_slopes_ptr 为 nullptr，则设为 0.0f
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    // 创建一个 Mask 对象，用于掩码操作
    pytorch_flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // 出于性能原因，分开两种迭代方式：
    // 需要在 S 上进行掩码的迭代和不需要的迭代。
    // 对于最后一个块，当 K 和 V 的长度不是 kBlockN 的倍数时，需要在 S 上进行掩码。
    // 如果是因果的，对于 ceil_div(kBlockM, kBlockN) 最后几个块也需要在 S 上进行掩码。
    // 至少会有 1 次 "masking" 迭代。
    // 如果 seqlen_k 不是偶数，那么可能会在块的中间结束。在这种情况下，我们需要屏蔽两个块（例如，当 kBlockM == kBlockN 时），而不仅仅是一个。
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    // 根据条件确定需要进行的屏蔽步骤数，用于控制迭代次数或屏蔽块数

    #pragma unroll
    }
    // 使用 #pragma 指令进行循环展开优化，确保代码段在编译时进行循环展开

    // 这些是在 S 上不需要屏蔽的迭代步骤
    for (; n_block >= n_block_min; --n_block) {
        // 在每个迭代中执行以下操作：

        // 调用partition_fragment_C函数，对tiled_mma进行分区操作，生成acc_s张量
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        
        // 清空acc_s张量的内容
        clear(acc_s);

        // 等待异步复制操作完成
        pytorch_flash::cp_async_wait<0>();

        // 同步所有线程
        __syncthreads();

        // 调整tVgV.data()的位置，以便下一次计算
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

        // 使用copy函数复制数据到tVsV, tKVcKV, tKVpKV中
        pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);

        // 等待所有的异步复制操作完成
        cute::cp_async_fence();

        // 执行矩阵乘法运算，结果存储在acc_s中
        pytorch_flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        // 等待所有的异步复制操作完成
        pytorch_flash::cp_async_wait<0>();

        // 同步所有线程
        __syncthreads();

        // 如果n_block大于n_block_min，调整tKgK.data()的位置，以便下一次计算
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));

            // 使用copy函数复制数据到tKsK, tKVcKV, tKVpKV中
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);

            // 这里的cp_async_fence必须放在if块内部，否则同步将不正确，可能导致竞争条件
            cute::cp_async_fence();
        }

        // 应用掩码操作到acc_s张量上
        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        // 对acc_s进行softmax操作
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // 将acc_s转换为元素类型为Element的张量rP
        Tensor rP = pytorch_flash::convert_type<Element>(acc_s);

        // 计算块的行索引和列索引
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);

        // 如果启用了Return_softmax，执行以下操作：
        if (Return_softmax) {
            // 创建一个与rP相似的张量rP_drop
            Tensor rP_drop = make_fragment_like(rP);

            // 复制rP到rP_drop中
            cute::copy(rP, rP_drop);

            // 应用dropout操作到rP_drop中
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kNWarps
            );

            // 复制rP_drop到tSgS中
            cute::copy(rP_drop, tSgS);

            // 调整tSgS.data()的位置，以便下一次计算
            tSgS.data() = tSgS.data() + (-kBlockN);
        }

        // 如果启用了Is_dropout，应用dropout操作到rP中
        if (Is_dropout) {
            dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        // 将rP从(MMA=4, MMA_M, MMA_N)重塑为((4, 2), MMA_M, MMA_N / 2)
        // 如果使用m16n8k16或(4, MMA_M, MMA_N)如果使用m16n8k8
        Tensor tOrP = make_tensor(rP.data(), pytorch_flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        pytorch_flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    // 对softmax的输出进行归一化处理，并返回lse张量
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // 将acc_o从fp32转换为fp16/bf16，并返回rO张量
    Tensor rO = pytorch_flash::convert_type<Element>(acc_o);

    // 创建一个与sQ相似的张量sO
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // 将 sO 分区以匹配累加器的分区方式
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    // 获取线程切片以供 smem_tiled_copy_O 使用
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    // 将 smem_thr_copy_O 根据 rO 重塑为新的 Tensor ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    // 根据 sO 将 smem_thr_copy_O 划分为 Tensor ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    // sO 和 sQ 大小相同，因此此处无需同步
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    // 将 smem_tiled_copy_O 中的数据复制到 taccOrO 和 taccOsO 中
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    // 计算 gO 的行偏移量
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    // 计算 gLSE 的行偏移量
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    // 创建 gO 的 Tensor
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    // 创建 gLSE 的 Tensor
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    // 创建 gmem_tiled_copy_O 对象并获取线程切片
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    // 根据 sO 将 gmem_thr_copy_O 划分为 Tensor ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    // 根据 gO 将 gmem_thr_copy_O 划分为 Tensor ((Atom,AtomNum),MMA_M,MMA_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    // 同步所有线程
    __syncthreads();

    // 创建 tOrO 与 tOsO 相同形状的 Tensor
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    // 将 gmem_tiled_copy_O 中的数据复制到 tOsO 到 tOrO
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // 创建用于累加器 caccO 的单位张量
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // 将 thr_mma 中的 caccO 分区为 Tensor (MMA,MMA_M,MMA_K)
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // 逻辑上将 taccOcO 转换为 ((2, 2), MMA_M, MMA_K)，然后仅取行索引
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    // 确保 taccOcO_row 的大小与 lse 的大小相同
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    // 遍历 taccOcO_row，根据条件将 lse 的值写入到 gLSE 中
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // 创建 sO 的单位布局 Tensor
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    // 使用 identity 布局重复分区
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    // 创建 tOpO，形状与 tOgO 的最后一个维度相同的布尔型 Tensor
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    // 如果 Is_even_K 不是偶数，则执行以下循环，将 tOpO 中的元素设置为 tOcO(0, 0, k) 的第二个元素是否小于 params.d
    #pragma unroll
    for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }

    // Clear_OOB_K 必须为 false，因为我们不希望将零写入到 gmem
    // 调用 pytorch_flash::copy 函数，参数 Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false
    // 将 gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM 传递给该函数
    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

    using Element = typename Kernel_traits::Element;  // 定义数据元素类型
    using ElementAccum = typename Kernel_traits::ElementAccum;  // 定义累加元素类型
    using index_t = typename Kernel_traits::index_t;  // 定义索引类型

    // Shared memory.
    extern __shared__ char smem_[];  // 声明共享内存

    // The thread index.
    const int tidx = threadIdx.x;  // 获取线程索引

    constexpr int kBlockM = Kernel_traits::kBlockM;  // 定义块大小 M
    constexpr int kBlockN = Kernel_traits::kBlockN;  // 定义块大小 N
    constexpr int kHeadDim = Kernel_traits::kHeadDim;  // 定义头部维度
    constexpr int kNWarps = Kernel_traits::kNWarps;  // 定义线程束数量

    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyOaccum,
        typename Kernel_traits::GmemTiledCopyO
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;  // 定义输出元素类型

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);  // 创建块信息对象，根据 Is_even_MN 决定是否可变长度

    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }

    // 检查是否需要处理当前块，如果不需要则返回
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // 计算每个分割区块的数量
    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;

    // 计算当前分割区块的最小块索引
    const int n_block_min = !Is_local
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);

    // 计算当前分割区块的最大块索引
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);

    // 如果是因果或本地操作，进一步限制最大块索引
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // 如果 n_block_min 大于等于 n_block_max，说明没有有效的块可以处理
        // 我们提前退出，并将 gOaccum 写入 0，gLSEaccum 写入 -inf
        // 否则可能会读取 gK 和 gV 中的越界元素，
        // 或者在组合来自不同块的 gOaccum 时得到错误的结果。

        // 计算输出张量的偏移量
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        // 计算 gOaccum 的偏移量
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_rounded;
        // 计算 gLSEaccum 的偏移量
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

        // 创建 gOaccum 张量
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                      make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
        // 创建 gLSEaccum 张量
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                       Shape<Int<kBlockM>>{}, Stride<_1>{});

        // 创建 gmem_tiled_copy_Oaccum 对象
        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        // 获取线程切片
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        // 对 gOaccum 进行划分
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        // 创建临时张量 tOrOaccum，并清空其内容
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);

        // 创建身份布局张量 cO，用于 sO 的构建
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // 使用身份布局对划分进行重复分区
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        // 创建 tOpO 张量，用于标记是否为偶数 K
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            // 如果 K 不是偶数，设置 tOpO 的值
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // 由于不想将零写入 gmem，Clear_OOB_K 必须为 false
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        // 遍历 tOgOaccum 的第一维度，处理 gLSEaccum
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
        }
        // 提前返回，处理完成
        return;
    }

    // 我们以相反的顺序迭代块。这是因为最后一个块是唯一一个
    // 计算 Q 矩阵在全局内存中的偏移量，考虑批次和行步长以及多头注意力的相关参数
    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    
    // 将 K 和 V 移动到最后一个块
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
    
    // 计算 K 矩阵在全局内存中的偏移量，考虑块表格或者默认计算方式
    const index_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : block_table[block_table_idx] * params.k_batch_stride + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    
    // 计算 V 矩阵在全局内存中的偏移量，考虑块表格或者默认计算方式
    const index_t row_offset_v = block_table == nullptr
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : block_table[block_table_idx] * params.v_batch_stride + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    
    // 创建全局内存中的 gQ 张量，用于存储 Q 矩阵数据
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    
    // 创建全局内存中的 gK 张量，用于存储 K 矩阵数据
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    
    // 创建全局内存中的 gV 张量，用于存储 V 矩阵数据
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    
    // 创建共享内存中的 sQ 张量，使用特定的共享内存布局
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    
    // 创建共享内存中的 sK 张量，使用特定的共享内存布局
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    
    // 创建共享内存中的 sV 张量，使用特定的共享内存布局
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    // 使用 sV 数据创建转置后的 Tensor，采用 Kernel_traits::SmemLayoutVtransposed 布局
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    
    // 使用 sV 数据创建不进行 swizzle 的 Tensor，采用 Kernel_traits::SmemLayoutVtransposedNoSwizzle 布局
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // 声明 GmemTiledCopyQKV 类型的对象 gmem_tiled_copy_QKV
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    
    // 获取当前线程索引 tidx 对应的线程切片 gmem_thr_copy_QKV
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    // 使用 gmem_thr_copy_QKV 对象划分并复制 gQ 数据的 Tensor tQgQ
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    // 使用 gmem_thr_copy_QKV 对象划分并复制 sQ 数据的 Tensor tQsQ
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    // 使用 gmem_thr_copy_QKV 对象划分并复制 gK 数据的 Tensor tKgK
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    // 使用 gmem_thr_copy_QKV 对象划分并复制 sK 数据的 Tensor tKsK
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    // 使用 gmem_thr_copy_QKV 对象划分并复制 gV 数据的 Tensor tVgV
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    // 使用 gmem_thr_copy_QKV 对象划分并复制 sV 数据的 Tensor tVsV
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // 声明 TiledMma 类型的对象 tiled_mma
    typename Kernel_traits::TiledMma tiled_mma;
    // 获取当前线程索引 tidx 对应的线程切片 thr_mma
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    
    // 使用 thr_mma 对象划分 sQ 数据的 Tensor tSrQ （MMA, MMA_M, MMA_K）
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    // 使用 thr_mma 对象划分 sK 数据的 Tensor tSrK （MMA, MMA_N, MMA_K）
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);
    // 使用 thr_mma 对象划分 sVtNoSwizzle 数据的 Tensor tOrVt （MMA, MMA_K, MMA_N）
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);

    // 使用 tiled_mma 对象和 Shape<Int<kBlockM>, Int<kHeadDim>> 参数划分结果 Tensor acc_o （MMA, MMA_M, MMA_K）
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    //
    // Copy Atom retiling
    //

    // 创建 SmemCopyAtom 类型的 tiled_copy_A 对象 smem_tiled_copy_Q，并获取当前线程索引 tidx 对应的线程切片 smem_thr_copy_Q
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // 使用 smem_thr_copy_Q 划分并复制 sQ 数据的 Tensor tSsQ
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    // 创建 SmemCopyAtom 类型的 tiled_copy_B 对象 smem_tiled_copy_K，并获取当前线程索引 tidx 对应的线程切片 smem_thr_copy_K
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    // 使用 smem_thr_copy_K 划分并复制 sK 数据的 Tensor tSsK
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    // 创建 SmemCopyAtomTransposed 类型的 tiled_copy_B 对象 smem_tiled_copy_V，并获取当前线程索引 tidx 对应的线程切片 smem_thr_copy_V
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    // 使用 smem_thr_copy_V 划分并复制 sVt 数据的 Tensor tOsVt
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // PREDICATES
    //

    // 分配用于 m 和 n 的谓词张量
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // 构建 sQ 和 sK 的身份布局 Tensor
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // 使用 gmem_thr_copy_QKV 对象划分并复制 cQ 数据的 Tensor tQcQ （ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    // 使用 gmem_thr_copy_QKV 对象划分并复制 cKV 数据的 Tensor tKVcKV （BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    // 分配用于 k 的谓词张量
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // 设置用于 k 边界的谓词
    // 如果 K 不是偶数，则执行以下操作
    if (!Is_even_K) {
        // 对 tQpQ 进行循环展开，检查每个元素是否小于 params.d
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        // 对 tKVpKV 进行循环展开，检查每个元素是否小于 params.d
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // 从 Knew 复制到 K，可选择应用旋转嵌入
    typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
    auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
    // 从 Knew 复制到 K，连续版本，可选择应用旋转嵌入
    typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
    auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);

    // Read Q from gmem to smem, optionally apply rotary embedding.
    // 如果不追加 KV 或者旋转维度为 0，则执行以下操作
    if (!Append_KV || params.rotary_dim == 0) {
        // 我们不需要清除 sQ smem 块，因为我们只会写入有效的输出
        pytorch_flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
    } else {
        // 计算旋转矩阵在数组中的偏移量，基于序列长度和是否是因果或局部操作
        const index_t row_offset_cossin = (binfo.seqlen_k_cache + (Is_causal || Is_local ? m_block * kBlockM : 0)) * (params.rotary_dim / 2);
        
        // 创建 gCos 张量，指向旋转余弦数组中的特定位置，以及对应的形状和步长
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        
        // 创建 gSin 张量，指向旋转正弦数组中的特定位置，以及对应的形状和步长
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        
        // 创建 gCosCont 张量，用于连续旋转余弦，指向特定位置，以及对应的形状和步长
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                      make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        
        // 创建 gSinCont 张量，用于连续旋转正弦，指向特定位置，以及对应的形状和步长
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                      make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        
        // 将 gCos 分区到 gmem_thr_copy_rotary 中，返回结果作为 tRgCos
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        
        // 将 gSin 分区到 gmem_thr_copy_rotary 中，返回结果作为 tRgSin
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        
        // 将 gCosCont 分区到 gmem_thr_copy_rotary_cont 中，返回结果作为 tRgCosCont
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        
        // 将 gSinCont 分区到 gmem_thr_copy_rotary_cont 中，返回结果作为 tRgSinCont
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        
        // 根据是否旋转交错，选择不同的旋转复制函数，复制 tQgQ, tQsQ, tRgCos, tRgSin 到 tQcQ
        if (params.is_rotary_interleaved) {
            pytorch_flash::copy_rotary_interleaved<Is_even_K>(
                tQgQ, tQsQ, tRgCos, tRgSin, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
                0, params.d, params.rotary_dim
            );
        } else {
            pytorch_flash::copy_rotary_contiguous<Is_even_K>(
                tQgQ, tQsQ, tRgCosCont, tRgSinCont, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
                0, params.d, params.rotary_dim
            );
        }
    }

    // 计算 n_block 的值
    int n_block = n_block_max - 1;
    
    // 不需要清除 sK smem 块，因为之后会对得分进行掩码操作
    pytorch_flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    
    // 异步复制操作的同步栅栏
    cute::cp_async_fence();

    // 清空 acc_o 张量
    clear(acc_o);

    // 创建 softmax 函数对象，针对 acc_o 张量的最后两个维度
    pytorch_flash::Softmax<2 * size<1>(acc_o)> softmax;
    // 计算 alibi_slope，根据是否有 alibi 进行条件判断并计算斜率
    const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    
    // 创建 Mask 对象用于序列处理，根据参数设置是否需要掩码操作
    pytorch_flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // 出于性能考虑，分离出两种迭代方式：
    // 需要在 S 上进行掩码操作的迭代和不需要的迭代。
    // 在以下情况下需要在 S 上进行掩码操作：
    // - 当最后一个块的 K 和 V 的长度不是 kBlockN 的倍数时
    // - 如果是因果的，最后 ceil_div(kBlockM, kBlockN) 个块也需要掩码操作
    // 至少会有 1 次“掩码”迭代。

    // 如果 seqlen_k 不是偶数，则可能在块的中间结束。在这种情况下，需要掩码 2 个块（例如，当 kBlockM == kBlockN 时），而不仅仅是 1 个块。
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    }

    // 这些是不需要在 S 上进行掩码操作的迭代

    // Epilogue

    // 计算 softmax，标准化并获取 log sum exp
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);
    // if (cute::thread0()) { print(lse); }

    // 创建 sOaccum 张量，用于累加器分区
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // 将 sO 分区以匹配累加器分区
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = pytorch_flash::convert_type<ElementO>(acc_o);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sOaccum 比 sQ 大，因此需要在此处同步线程
    // TODO: 为 sOaccum 分配足够的 smem
    if constexpr (Split) { __syncthreads(); }

    // 复制操作，将 taccOrOaccum 和 taccOsOaccum 复制到 smem_tiled_copy_Oaccum
    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    // 计算行偏移量，用于累加器和输出张量的索引计算
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                         + m_block * kBlockM) * params.d_rounded;
    const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    // 创建名为 gOaccum 的 Tensor 对象，用于表示累积的输出张量
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_p)),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    
    // 创建名为 gLSEaccum 的 Tensor 对象，用于表示累积的softmax输出张量
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
    
    // 如果线程索引为 0，则打印输出一些变量的值（调试用）
    // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

    // 创建名为 gmem_tiled_copy_Oaccum 的 GmemTiledCopyO 对象
    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    // 获取当前线程的 gmem_tiled_copy_Oaccum 对象的切片
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    // 使用 gmem_thr_copy_Oaccum 对象对 sOaccum 进行分区，并得到 Tensor 对象 tOsOaccum
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    // 使用 gmem_thr_copy_Oaccum 对象对 gOaccum 进行分区，并得到 Tensor 对象 tOgOaccum
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    // 同步线程，等待所有线程完成之前的操作
    __syncthreads();

    // 创建名为 tOrOaccum 的 Tensor 对象，其形状与 tOgOaccum 相同
    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    // 使用 cute::copy 函数将 tOsOaccum 复制到 tOrOaccum 中
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    // 创建名为 caccO 的单位张量 Tensor 对象，用于 MMA 计算
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // 使用 thr_mma 对 caccO 进行分区，并得到 Tensor 对象 taccOcO
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    
    // 将 taccOcO 根据逻辑条件进行划分，得到 Tensor 对象 taccOcO_row
    // 并且仅取行索引
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    // 检查 taccOcO_row 的大小是否与 lse 的大小相同
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    
    // 如果 taccOcO_row 的第一个元素的第二个分量为 0，则执行下面的代码块
    if (get<1>(taccOcO_row(0)) == 0) {
        // 遍历 lse 的所有元素
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            // 获取当前元素的行索引
            const int row = get<0>(taccOcO_row(mi));
            // 如果行索引小于 binfo.actual_seqlen_q - m_block * kBlockM，则将 lse(mi) 写入 gLSEaccum 中对应位置
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // 创建名为 cO 的单位张量 Tensor 对象，用于表示输出张量的布局
    // 其形状与 sOaccum 相同
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    
    // 使用 gmem_thr_copy_Oaccum 对象对 cO 进行分区，并得到 Tensor 对象 tOcO
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    
    // 创建名为 tOpO 的布尔类型 Tensor 对象，用于表示输出的布局
    // 其形状与 tOgOaccum 的第二个维度大小相同
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    
    // 如果不是偶数 K，则执行下面的代码块
    if (!Is_even_K) {
        // 遍历 tOpO 的所有元素
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    
    // 调用 pytorch_flash::copy 函数，根据给定的条件复制数据到输出张量中
    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    // __syncthreads();
    // if (cute::thread0()) { print(tOgOaccum); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params &params) {
    // 当前线程块在M维度上的索引
    const int m_block = blockIdx.x;
    // 当前线程块在batch维度上的索引
    const int bidb = blockIdx.y;
    // 当前线程块在head维度上的索引
    const int bidh = blockIdx.z;

    // 我们希望前向传播和反向传播生成相同的dropout模式（随机数生成器），而不限制它们具有相同数量的线程或者以相同的顺序遍历注意力矩阵。
    // 在Philox随机数生成器中，我们使用偏移量来存储batch、head和lane id（在warp内）。我们使用子序列来存储注意力矩阵中16x32块的位置。
    // 这样，只要我们有batch、head和在注意力矩阵中16x32块的位置，我们就可以生成完全相同的dropout模式。

    // 调用具体的注意力计算函数，处理一个M维度的块
    pytorch_flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, m_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    const int m_block = blockIdx.x;
    // 当Split为真时，batch维度上的块索引
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // 当Split为真时，head维度上的块索引
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    // 当Split为真时，分割的索引
    const int n_split_idx = Split ? blockIdx.y : 0;
    // 当Split为真时，总的分割数
    const int num_n_splits = Split ? gridDim.y : 1;
    // 调用具体的注意力计算函数，处理一个M维度的块，支持分割key/value的情况
    pytorch_flash::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
inline __device__ void combine_attn_seqk_parallel(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    // 确保最大分割数不超过128
    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    // 确保块大小在指定的范围内
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    // 确保线程数目符合预期，如果不符合则触发静态断言错误
    static_assert(kNThreads == 128, "We assume that each block has 128 threads");

    // 定义共享内存数组，每个块内的共享内存大小为 kBlockM + 1，用于减少银行冲突
    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    // 获取当前线程和块的索引
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    // 计算当前块在全局数组中的起始偏移量
    const index_t row_offset_lse = bidx * kBlockM;

    // 创建用于存储累积 LSE 值的张量 gLSEaccum，从全局内存中读取数据
    Tensor gLSEaccum = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
        Shape<Int<kMaxSplits>, Int<kBlockM>>{},
        make_stride(params.b * params.h * params.seqlen_q, _1{}));

    // 创建用于存储 LSE 值的张量 gLSE，从全局内存中读取数据
    Tensor gLSE = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
        Shape<Int<kBlockM>>{}, Stride<_1>{});

    // 计算每个线程需要处理的 LSE 值的数量
    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // 从全局内存中读取 LSE 值并存储到共享内存中，然后进行转置操作
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        // 如果索引有效，则从 gLSEaccum 中读取对应位置的值，否则设置为 -INFINITY
        ElementAccum lse = (row < params.num_splits && col < params.b * params.h * params.seqlen_q - bidx * kBlockM) ?
                            gLSEaccum(row, col) : -INFINITY;
        // 将有效的 LSE 值存储到共享内存 sLSE 中
        if (row < kMaxSplits) { sLSE[row][col] = lse; }
        // 如果需要调试输出，可以打开下面的 printf 语句
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse); }
    }

    // 等待所有线程执行完毕，确保共享内存数据加载完毕
    __syncthreads();

    // 创建用于存储累积 LSE 值的张量 lse_accum
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});

    // 计算每个线程需要处理的转置操作的行数
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);

    // 从共享内存 sLSE 中读取数据并进行转置操作，存储到 lse_accum 中
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        // 如果索引有效，则从共享内存 sLSE 中读取对应位置的值，否则设置为 -INFINITY
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
        // 如果需要调试输出，可以打开下面的 printf 语句
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
    }

    // 计算在分裂维度上的 LSE 的 logsumexp 值
    ElementAccum lse_max = lse_accum(0);
    #pragma unroll
    // 计算每个线程中的本地 LSE 最大值
    for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }

    // 创建一个 MaxOp<float> 的对象
    MaxOp<float> max_op;

    // 执行 Allreduce 操作以获取全局的 LSE 最大值
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);

    // 处理特殊情况：当所有本地 LSE 值都为 -INFINITY 时，将 lse_max 设置为 0.0f
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;

    // 计算指数和，用于归一化
    float lse_sum = expf(lse_accum(0) - lse_max);

    // 使用 #pragma unroll 指令展开循环，计算其他本地 LSE 的指数和
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }

    // 创建一个 SumOp<float> 的对象
    SumOp<float> sum_op;

    // 执行 Allreduce 操作以获取全局的 LSE 指数和
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);

    // 计算 log-sum-exp，用于归一化
    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;

    // 如果当前线程在块内的索引符合条件，则更新全局的 LSE 值
    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) {
        gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
    }

    // 将 scales exp(lse - lse_logsum) 存储在共享内存中
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        // 确保存储的位置在有效范围内
        if (row < params.num_splits && col < kBlockM) {
            sLSE[row][col] = expf(lse_accum(l) - lse_logsum);
        }
    }
    __syncthreads();

    // 计算偏移量并创建输出张量 gOaccum
    const index_t row_offset_oaccum = bidx * kBlockM * params.d_rounded;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 Stride<Int<kHeadDim>, _1>{});

    // 定义常量 kBlockN，并声明相关的布局和复制类型
    constexpr int kBlockN = kNThreads / kBlockM;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;

    // 获取线程片段并创建与 gOaccum 相对应的张量
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);

    // 创建与 tOgOaccum 相同形状的输出张量 tOrO 和 tOrOaccum，并将 tOrO 清零
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    // 创建布尔类型的标识张量
    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});

    // 使用相同的线程片段创建 tOcOaccum 张量
    Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);

    // 创建布尔类型的输出张量 tOpOaccum
    Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // 如果 K 不是偶数，则执行以下操作
    if (!Is_even_K) {
        // 对于 tOpOaccum 中的每个元素，检查是否小于 params.d，将结果保存到 tOpOaccum 中
        #pragma unroll
        for (int k = 0; k < size(tOpOaccum); ++k) { tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d; }
    }
    // 加载 Oaccum 然后进行缩放并累加到 O 中
    for (int split = 0; split < params.num_splits; ++split) {
        // 调用 pytorch_flash::copy 函数，根据 Is_even_MN 和 Is_even_K 的值复制数据
        pytorch_flash::copy</*Is_even_MN=*/false, Is_even_K>(
            gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * params.seqlen_q - bidx * kBlockM
        );
        // 对于 tOrOaccum 的每个元素进行遍历
        #pragma unroll
        for (int m = 0; m < size<1>(tOrOaccum); ++m) {
            // 获取 tOcOaccum 中指定位置的行索引
            int row = get<0>(tOcOaccum(0, m, 0));
            // 获取对应的缩放因子
            ElementAccum lse_scale = sLSE[split][row];
            // 对 tOrOaccum 的每个元素进行累加操作
            #pragma unroll
            for (int k = 0; k < size<2>(tOrOaccum); ++k) {
                #pragma unroll
                for (int i = 0; i < size<0>(tOrOaccum); ++i) {
                    tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
                }
            }
        // 如果线程是第一个线程，打印缩放因子和部分数据内容
        // if (cute::thread0()) { printf("lse_scale = %f, %f\n", sLSE[split][0], sLSE[split][1]); print(tOrOaccum); }
        }
        // 更新 tOgOaccum 的数据，增加偏移量 params.b * params.h * params.seqlen_q * params.d_rounded
        tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d_rounded;
    }
    // 如果线程是第一个线程，打印整个 tensor tOrO 的内容
    // if (cute::thread0()) { print_tensor(tOrO); }

    // 将 tOrO 转换为类型为 Element 的 Tensor rO
    Tensor rO = pytorch_flash::convert_type<Element>(tOrO);
    // 写入到 gO 中
    #pragma unroll
    for (int m = 0; m < size<1>(rO); ++m) {
        // 计算索引 idx
        const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
        // 如果 idx 小于 params.b * params.h * params.seqlen_q，则执行以下操作
        if (idx < params.b * params.h * params.seqlen_q) {
            // 计算 batch_idx 和 head_idx
            const int batch_idx = idx / (params.h * params.seqlen_q);
            const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
            // 计算 Q 行索引
            const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
            // 获取目标存储位置 o_ptr
            auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride
                + head_idx * params.o_head_stride + row * params.o_row_stride;
            // 对 rO 的每个元素进行遍历
            #pragma unroll
            for (int k = 0; k < size<2>(rO); ++k) {
                // 获取列索引 col
                const int col = get<1>(tOcOaccum(0, m, k));
                // 创建 Tensor gO，指向 o_ptr + col 处的内存，用于存储数据
                Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                                        Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
                // 复制 rO(_, m, k) 的数据到 gO
                copy(rO(_, m, k), gO);
                // 如果 Is_even_K 为真或者 tOpOaccum(k) 为真，则执行以下操作
                // if (bidx == 0 && tidx == 0) { printf("tidx = %d, idx = %d, batch_idx = %d, head_idx = %d, row = %d, col = %d\n", tidx, idx, batch_idx, head_idx, row, col); print(rO(_, m, k)); print(gO); }
                // reinterpret_cast<uint64_t *>(o_ptr)[col / 4] = recast<uint64_t>(rO)(0, m, k);
            }
        }
    }
}

} // namespace pytorch_flash
```