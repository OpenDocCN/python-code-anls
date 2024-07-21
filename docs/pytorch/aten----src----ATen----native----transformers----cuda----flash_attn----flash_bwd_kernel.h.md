# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\flash_bwd_kernel.h`

```
/***************************************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/PhiloxUtils.cuh>
#include <cute/algorithm/copy.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include <ATen/native/transformers/cuda/flash_attn/block_info.h>
#include <ATen/native/transformers/cuda/flash_attn/kernel_traits.h>
#include <ATen/native/transformers/cuda/flash_attn/utils.h>
#include <ATen/native/transformers/cuda/flash_attn/softmax.h>
#include <ATen/native/transformers/cuda/flash_attn/mask.h>
#include <ATen/native/transformers/cuda/flash_attn/dropout.h>
#include <ATen/native/transformers/cuda/flash_attn/alibi.h>

// 声明命名空间 pytorch_flash
namespace pytorch_flash {

// 使用 cute 命名空间
using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// 模板函数定义，返回一个用于 tiled mma 操作的布局
template <int MMA_N,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_B_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA           const& tiled_mma) {
    // 定义常量 TileShape_N 和 TileShape_K，分别为 tiled_mma 的第 1 和第 2 维大小
    constexpr int TileShape_N = decltype(tiled_mma.template tile_size_mnk<1>())::value;
    constexpr int TileShape_K = decltype(tiled_mma.template tile_size_mnk<2>())::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    // 获取 AtomShape_N 的值
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // 计算每个 warp 的数量，其中 kNWarpsN 为 TileShape_N 除以 AtomShape_N 的两倍
    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    // 计算 MMA 操作中的 N 维度步长
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    // 创建一个布局 t，用于 tiled mma 操作
    // 布局包含 Shape 和 Stride 信息，Shape 表示维度大小，Stride 表示步长
    auto t = make_tile(Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,   // (8, 2, 2) or (8, 4, 2)
                              Stride<_1, Int<MMAStride_N>, _8> >{},       // (1, 64, 8) or (1, 32, 8)
                       make_layout(Int<TileShape_K>{}));
    // 如果是第 0 线程，则输出调试信息
    // if (cute::thread0()) {printf("make_tiled_copy_B_warpcontiguousN "); print(t); printf("\n");  }
    // 返回 tiled mma 操作的结果
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutB_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// 模板函数定义，返回一个用于 tiled mma 操作的布局
template <int MMA_N,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA           const& tiled_mma) {
    // 定义常量 TileShape_M 和 TileShape_N，分别为 tiled_mma 的第 0 和第 1 维大小
    constexpr int TileShape_M = decltype(tiled_mma.template tile_size_mnk<0>())::value;
    constexpr int TileShape_N = decltype(tiled_mma.template tile_size_mnk<1>())::value;


这段代码主要包含了两个模板函数的定义，这些函数用于生成用于 tiled mma 操作的布局信息，包括 Shape 和 Stride 的计算和定义。
    // 定义类型别名 AtomShape_MNK 为 TiledMMA::AtomShape_MNK
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    // 从 AtomShape_MNK 中获取第一个维度的大小作为常量 AtomShape_N
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // 计算每个 Warp 内的线程数 kNWarpsN，假设 ValLayout 固定为 2，因此除以 2
    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    // 计算 MMA 操作的步长 MMAStride_N，为 MMA_N 乘以 AtomShape_N 的两倍
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    // 创建一个 Tile 布局 t，结合 TileShape_M 和 MMA 的布局要求
    auto t = make_tile(make_layout(Int<TileShape_M>{}),
                       Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,   // (8, 2, 2) 或 (8, 4, 2)
                              Stride<_1, Int<MMAStride_N>, _8> >{});       // (1, 64, 8) 或 (1, 32, 8)
    // 如果当前线程是第一个线程，打印相关信息
    // if (cute::thread0()) {printf("make_tiled_copy_C_warpcontiguousN "); print(t); printf("\n");  }
    // 返回基于输入参数和布局 t 的 tiled_copy 实现结果
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
// 结束之前的代码块，这里应该是一个函数或代码段的末尾
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// 模板函数，计算某些参数下的 dq, dk, dv 的值，处理一列数据块
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory allocation for this block's threads
    extern __shared__ char smem_[];

    // Get the thread index within the block
    const int tidx = threadIdx.x;

    // Constants derived from Kernel_traits
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    // Obtain block information for computation
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    // Early return if the current block's offset exceeds the sequence length
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    // Calculate maximum number of m blocks based on sequence length of q
    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    // Adjust m_block_max if Is_local is true, limiting based on window size
    if (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }

    // Calculate row offsets for various tensors based on block and head indices
    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    // 计算 dQ_accum 的行偏移量，根据当前线程块的索引和参数计算得到
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + ((m_block_max - 1) * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded
        // 如果是确定性操作，每个线程块会向不同的 dQ_accum 缓冲区执行 atomicAdd 操作
        + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);

    // 计算 LSE 的行偏移量，基于当前线程块的 bidb 和 bidh，以及 seqlen_q 和 m_block_max 参数
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q
        + (m_block_max - 1) * kBlockM;

    // 计算 dpsum 的行偏移量，基于当前线程块的 bidb 和 bidh，以及 seqlen_q_rounded 和 m_block_max 参数
    const index_t row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded
        + (m_block_max - 1) * kBlockM;

    // 创建 gQ 张量，用于访问 q_ptr 中的数据，基于 row_offset_q 和相关的参数
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));

    // 创建 gK 张量，用于访问 k_ptr 中的数据，基于 row_offset_k 和相关的参数
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    // 创建 gV 张量，用于访问 v_ptr 中的数据，基于 row_offset_v 和相关的参数
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

    // 创建 gdO 张量，用于访问 do_ptr 中的数据，基于 row_offset_do 和相关的参数
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));

    // 创建 gO 张量，用于访问 o_ptr 中的数据，基于 row_offset_o 和相关的参数
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));

    // 创建 gdQ 张量，用于访问 dq_ptr 中的数据，基于 row_offset_dq 和相关的参数
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));

    // 创建 gdQaccum 张量，用于访问 dq_accum_ptr 中的数据，基于 row_offset_dq_accum 和相关的参数
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.h * params.d_rounded, _1{}));

    // 创建 gLSE 张量，用于访问 softmax_lse_ptr 中的数据，基于 row_offset_lse 和相关的参数
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    // 创建 gdPsum 张量，用于访问 dsoftmax_sum 中的数据，基于 row_offset_dpsum 和相关的参数
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    // 创建 sQ 张量，用于访问 smem_ 中的数据，根据 Kernel_traits::SmemLayoutQdO 进行布局
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    // 创建名为 sQt 的 Tensor 对象，使用 sQ 数据并采用指定的内存布局（SmemLayoutQdOtransposed）
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});

    // 创建名为 sQtNoSwizzle 的 Tensor 对象，使用 sQ 数据并采用指定的内存布局（SmemLayoutQdOtransposedNoSwizzle）
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});

    // 根据 Double_buffer 变量的值，创建名为 sdO 的 Tensor 对象，使用 sQ 数据之后的内存位置，并采用指定的内存布局（SmemLayoutQdO）
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});

    // 创建名为 sdOt 的 Tensor 对象，使用 sdO 数据并采用指定的内存布局（SmemLayoutQdOtransposed）
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});

    // 创建名为 sdOtransposedNoSwizzle 的 Tensor 对象，使用 sdO 数据并采用指定的内存布局（SmemLayoutQdOtransposedNoSwizzle）
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});

    // 创建名为 sK 的 Tensor 对象，使用 sdO 数据之后的内存位置，并采用指定的内存布局（SmemLayoutKV）
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});

    // 创建名为 sV 的 Tensor 对象，使用 sK 数据之后的内存位置，并采用指定的内存布局（SmemLayoutKV）
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});

    // 创建名为 sKt 的 Tensor 对象，使用 sK 数据并采用指定的内存布局（SmemLayoutKtransposed）
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});

    // 创建名为 sKtNoSwizzle 的 Tensor 对象，使用 sK 数据并采用指定的内存布局（SmemLayoutKtransposedNoSwizzle）
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});

    // 根据 Kernel_traits::Is_V_in_regs 条件，创建名为 sdS 的 Tensor 对象，使用 sV 或者 sK 数据之后的内存位置，并采用指定的内存布局（SmemLayoutPdS）
    Tensor sdS = make_tensor(!Kernel_traits::Is_V_in_regs ? sV.data() + size(sV) : sK.data() + size(sK),
                             typename Kernel_traits::SmemLayoutPdS{});

    // 创建名为 sdSt 的 Tensor 对象，使用 sdS 数据并采用指定的内存布局（SmemLayoutPdStransposed）
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});

    // 创建名为 sdStNoSwizzle 的 Tensor 对象，使用 sdS 数据并采用指定的内存布局（SmemLayoutPdStransposedNoSwizzle）
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});

    // 创建名为 sP 的 Tensor 对象，使用 sdS 数据之后的内存位置，并采用指定的内存布局（SmemLayoutPdS）
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});

    // 创建名为 sPt 的 Tensor 对象，使用 sP 数据并采用指定的内存布局（SmemLayoutPdStransposed）
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});

    // 创建名为 sPtNoSwizzle 的 Tensor 对象，使用 sP 数据并采用指定的内存布局（SmemLayoutPdStransposedNoSwizzle）
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});

    // 创建名为 sdQ 的 Tensor 对象，与 sP 共享同一内存空间，使用指定的内存布局（SmemLayoutdQ）
    Tensor sdQ = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutdQ{});

    // 定义名为 gmem_tiled_copy_QKV 的类型为 Kernel_traits::GmemTiledCopyQKV 的对象
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    // 获取线程 tidx 对应的 gmem_tiled_copy_QKV 对象的线程切片
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    // 根据 Is_first 条件选择不同的类型作为 GmemTiledCopydO 的模板参数，创建名为 gmem_tiled_copy_dO 的对象
    using GmemTiledCopydO = std::conditional_t<
        Is_first,
        typename Kernel_traits::GmemTiledCopydO,
        typename Kernel_traits::GmemTiledCopyQKV
    >;
    GmemTiledCopydO gmem_tiled_copy_dO;
    // 获取线程 tidx 对应的 gmem_tiled_copy_dO 对象的线程切片
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);

    // 创建名为 gmem_tiled_copy_dQ 的对象，类型为 Kernel_traits::GmemTiledCopydQ
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    // 获取线程 tidx 对应的 gmem_tiled_copy_dQ 对象的线程切片
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    // 根据 Seq_parallel 条件选择不同类型作为 GmemLayoutAtomdQaccum 的模板参数，创建名为 gmem_tiled_copy_dQaccum 的对象
    using GmemLayoutAtomdQaccum = std::conditional_t<
        !Seq_parallel,
        typename Kernel_traits::GmemTiledCopydQaccum,
        typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd
    >;
    GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
    // 获取线程 tidx 对应的 gmem_tiled_copy_dQaccum 对象的线程切片
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    // 将 gQ 的数据按照 gmem_thr_copy_QKV 对象的方式划分，并存储到名为 tQgQ 的 Tensor 对象中
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    // 将 sQ 的数据按照 gmem_thr_copy_QKV 对象的方式划分，并存储到名为 tQsQ 的 Tensor 对象中
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    // 将 gdO 的数据按照 gmem_thr_copy_dO 对象的方式划分，并存储到名为 tdOgdO 的 Tensor 对象中
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    // 将 sdO 的数据按照 gmem_thr_copy_dO 对象的方式划分，并存储到名为 tdOsdO 的 Tensor 对象中
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    // 将 gO 的数据按照 gmem_thr_copy_dO 对象的方式划分，并存储到名为 tdOgO 的 Tensor 对象中
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    // 将 gK 分区复制到线程内存并命名为 tKgK，用于后续计算 (KCPY, KCPY_N, KCPY_K)
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    // 将 sK 分区复制到线程内存并命名为 tKsK，用于后续计算
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    // 将 gV 分区复制到线程内存并命名为 tVgV，用于后续计算 (VCPY, VCPY_N, VCPY_K)
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    // 将 sV 分区复制到线程内存并命名为 tVsV，用于后续计算
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    // 将 sdQ 分区复制到线程内存并命名为 tdQsdQ，用于后续计算 ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);
    // 将 gdQ 分区复制到线程内存并命名为 tdQgdQ，用于后续计算
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    // 将 gdQaccum 分区复制到线程内存并命名为 tdQgdQaccum，用于后续计算
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    // 如果是第一个线程，打印 tdQgdQaccum 的布局信息
    // if (cute::thread0()) { print(tdQgdQaccum.layout()); printf("\n"); }
    // 同步所有线程
    // __syncthreads();
    // 如果 blockIdx 全为 0 且 tidx 小于 64，则打印 tidx 和 tdQgdQaccum 数据的地址
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx < 64) {
    //     printf("tidx = %d, tdQgdQaccum = 0x%p\n", tidx, tdQgdQaccum.data());
    // }

    // 使用 Kernel_traits 获取 TiledMmaSdP 的类型，并初始化 tiled_mma_sdp
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    // 根据线程索引 tidx 切片获取 tiled_mma_sdp 中的数据，命名为 thr_mma_sdp
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    // 将 sQ 分区复制到 thr_mma_sdp 并命名为 tSrQ，用于后续计算 (MMA,MMA_N,MMA_K)
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);
    // 将 sK 分区复制到 thr_mma_sdp 并命名为 tSrK，用于后续计算 (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    // 将 sdO 分区复制到 thr_mma_sdp 并命名为 tdPrdO，用于后续计算 (MMA,MMA_N,MMA_K)
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);
    // 将 sV 分区复制到 thr_mma_sdp 并命名为 tdPrV，用于后续计算 (MMA,MMA_N,MMA_K)
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    // 使用 Kernel_traits 获取 TiledMmadKV 的类型，并初始化 tiled_mma_dkv
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    // 根据线程索引 tidx 切片获取 tiled_mma_dkv 中的数据，命名为 thr_mma_dkv
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    // 将 sdStNoSwizzle 分区复制到 thr_mma_dkv 并命名为 tdKrdSt，用于后续计算 (MMA, MMA_N, MMA_N)
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle);
    // 将 sQtNoSwizzle 分区复制到 thr_mma_dkv 并命名为 tdKrQt，用于后续计算 (MMA, MMA_K, MMA_N)
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);
    // 将 sPtNoSwizzle 分区复制到 thr_mma_dkv 并命名为 tdVrPt，用于后续计算 (MMA, MMA_N, MMA_N)
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);
    // 将 sdOtransposedNoSwizzle 分区复制到 thr_mma_dkv 并命名为 tdVrdO，用于后续计算 (MMA, MMA_K, MMA_N)
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle);

    // 使用 Kernel_traits 获取 TiledMmadQ 的类型，并初始化 tiled_mma_dq
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    // 根据线程索引 tidx 切片获取 tiled_mma_dq 中的数据，命名为 thr_mma_dq
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    // 将 sdS 分区复制到 thr_mma_dq 并命名为 tdQrdS，用于后续计算 (MMA, MMA_N, MMA_N)
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);
    // 将 sKtNoSwizzle 分区复制到 thr_mma_dq 并命名为 tdQrKt，用于后续计算 (MMA, MMA_K, MMA_N)
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    // 根据 tiled_mma_dkv 和 Shape 初始化 acc_dk，用于累加结果 (MMA, MMA_N, MMA_K)
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    // 根据 tiled_mma_dkv 和 Shape 初始化 acc_dv，用于累加结果 (MMA, MMA_N, MMA_K)
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    //
    // Copy Atom retiling
    //

    // 使用 SmemCopyAtom 初始化 smem_tiled_copy_QdO，并从 tiled_mma_sdp 中获取切片
    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    // 根据线程索引 tidx 切片获取 smem_tiled_copy_QdO 中的数据，命名为 smem_thr_copy_QdO
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    // 将 sQ 分区复制到 smem_thr_copy_QdO 并命名为 tSsQ，用于后续计算
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    // 将 sdO 分区复制到 smem_thr_copy_QdO 并命名为 tdPsdO，用于后续计算
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    // 使用 SmemCopyAtom 初始化 smem_tiled_copy_KV，并从 tiled_mma_sdp 中获取切片
    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    // 根据线程索引 tidx 切片获取 smem_tiled_copy_KV 中的数据，命名为 smem_thr_copy_KV
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    // 将 sK 分区复制到 smem_thr_copy_KV 并命名为 tSsK，用于后续计算
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    // 复制输入张量 sV 到线程局部内存，并根据 KV 矩阵的分区对其进行分割
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // 创建一个 tiled_mma_sdp 类型的线程局部内存复制对象，用于 sP 和 sdS 的分割以匹配累加器的分区
    auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    // 根据分割后的 sP 创建张量 tPsP
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // 复制输入张量 sdS 到线程局部内存，并根据 PdS 的分区对其进行分割
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // 创建一个 tiled_mma_dkv 类型的线程局部内存复制对象，用于 sPt 和 sdSt 的分割
    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    // 根据分割后的 sPt 创建张量 tdVsPt 和 tdKsdSt
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    // 创建一个 tiled_mma_dkv 类型的线程局部内存复制对象，用于 sdOt 和 sQt 的分割
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    // 根据分割后的 sdOt 创建张量 tdVsdOt 和 tdKsQt
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    // 创建一个 tiled_mma_dq 类型的线程局部内存复制对象，用于 sdS 的分割
    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    // 根据分割后的 sdS 创建张量 tdQsdS
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    // 创建一个 tiled_mma_dq 类型的线程局部内存复制对象，用于 sKt 的分割
    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    // 根据分割后的 sKt 创建张量 tdQsKt
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    // 创建一个 tiled_mma_dq 类型的线程局部内存复制对象，用于 sdQ 的分割
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    // 根据分割后的 sdQ 创建张量 taccdQsdQ
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    //
    // PREDICATES
    //

    // 创建一个与输入张量 sQ 形状相同的单位张量 cQ
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // 根据分割后的 cQ 创建张量 tQcQ
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);

    // 创建一个与输入张量 sK 形状相同的单位张量 cKV
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // 根据分割后的 cKV 创建张量 tKVcKV
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);
    // 为 k 分配谓词张量
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    // 为 k 分配谓词张量
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // 设置 k 边界的谓词
    if (!Is_even_K) {
        // 如果 K 不是偶数，则执行以下操作
        #pragma unroll
        // 遍历 tQpQ 张量的所有元素
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        // 遍历 tKVpKV 张量的所有元素
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // 开始处理

    // 在第一次读写之前，我们将先更新 gdQ 和 gdQaccum。
    tdQgdQ.data() = tdQgdQ.data() + kBlockM * params.dq_row_stride;
    tdQgdQaccum.data() = tdQgdQaccum.data() + kBlockM * params.h * params.d_rounded;

    // 初始化 m_block
    int m_block = m_block_max - 1;
    // 计算 m_block_min 的值，根据条件不同有不同的设置
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);
    // 如果不是本地操作，则确保 m_block_min <= m_block:
    // 我们之前检查过 n_block * kBlockN < actual_seqlen_k，因此在因果情况下，
    // n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k < actual_seqlen_q。
    // 所以 m_block_min <= (actual_seqlen_q - 1) / kBlockM。
    // 回想一下 m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) = (actual_seqlen_q + kBlockM - 1) / kBlockM。
    // 因此 m_block_m - 1 = (actual_seqlen_q - 1) / kBlockM。
    // 我们得出 m_block_min <= m_block，因此至少会有一次 for 循环迭代。
    // 然而，如果是本地操作，则可能会有一些 K & V 块没有处理任何查询。
    // 我们可能需要提前退出并为这些块写入 0 到 dK 和 dV。
    // 否则，对于不进入 for 循环的情况，我们将得到错误的结果。
    // 并且我们可能会从 gQ 和 gdO 中读取超出边界的元素。
    // 这也包括 actual_seqlen_q == 0 的情况。
    // 检查是否为本地操作或者不是偶数的MN，并且m_block小于m_block_min时执行以下代码块
    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        // 计算dk的行偏移量
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
          + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        // 计算dv的行偏移量
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
          + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        // 创建gdK张量，表示梯度dk
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                 make_stride(params.dk_row_stride, _1{}));
        // 创建gdV张量，表示梯度dv
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                 make_stride(params.dv_row_stride, _1{}));
        // 定义gmem_tiled_copy_dKV，用于处理D和KV的拷贝
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        // 获取线程切片的gmem_thr_copy_dKV
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        // 分区处理gdK
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        // 分区处理gdV
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        // 创建tdKrdK，用于保存kd的拷贝
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        // 创建tdVrdV，用于保存vd的拷贝
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        // 清空tdKrdK
        clear(tdKrdK);
        // 清空tdVrdV
        clear(tdVrdV);
        // 创建cdKV，表示从(BLK_N,BLK_K)到(blk_n,blk_k)的单位张量
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));
        // 分区处理cdKV
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        // 创建tdKVpdKV，用于保存tdKVcdKV的布尔掩码
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        // 使用#pragma unroll对tdKVpdKV进行循环赋值，根据tdKVcdKV的数据是否小于params.d进行判断
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
        // 使用pytorch_flash::copy对tdKrdK和tdKgdK进行异步拷贝
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        // 使用pytorch_flash::copy对tdVrdV和tdVgdV进行异步拷贝
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        // 返回，结束函数执行
        return;
    }

    // 如果Double_buffer为真且m_block为奇数，则执行以下代码块，用于sQ的双缓冲
    if (Double_buffer && m_block % 2 == 1) {  // Double buffer for sQ
        // 对tQsQ、tSsQ和tdKsQt进行数据更新
        tQsQ.data() = tQsQ.data() + size(sQ);
        tSsQ.data() = tSsQ.data() + size(sQ);
        tdKsQt.data() = tdKsQt.data() + size(sQ);
    }

    // 如果不是第一次操作且不是顺序并行，或者params.deterministic为真，则执行同步线程
    if ((!Is_first && !Seq_parallel) || params.deterministic) { __syncthreads(); }

    // 如果Kernel_traits::Is_V_in_regs为真，则执行以下代码块，用于清除smem瓦片以处理预测的离线加载
    if (Kernel_traits::Is_V_in_regs) {
        // 使用pytorch_flash::copy对tVgV和tVsV进行清除操作，并设置Clear_OOB_MN为true
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        // 执行异步内存拷贝的屏障
        pytorch_flash::cp_async_fence();
    }
    // 创建一个与 tdOgdO 具有相同形状的张量 tdOrdO
    Tensor tdOrdO = make_fragment_like(tdOgdO);
    // 创建一个与 tdOgO 具有相同形状的张量 tdOrO
    Tensor tdOrO = make_fragment_like(tdOgO);
    // 如果不是第一次执行，则清空共享内存的瓦片以处理断言外的加载
    if (!Is_first) {
        // 使用指定的条件复制函数，将 gmem_tiled_copy_dO 中的数据复制到 tdOgdO 和 tdOsdO 中
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        // 使用指定的条件复制函数，将 gmem_tiled_copy_dO 中的数据复制到 tdOgdO 和 tdOrdO 中
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        // 使用指定的条件复制函数，将 gmem_tiled_copy_dO 中的数据复制到 tdOgO 和 tdOrO 中
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
    // 使用指定的条件复制函数，将 gmem_tiled_copy_QKV 中的数据复制到 tQgQ 和 tQsQ 中
    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );

    // 创建一个形状为 (BLK_M,BLK_N) 的单位张量 caccS
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    // 对 caccS 进行分区，返回形状为 (MMA,MMA_N,MMA_N) 的张量 taccScS
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // 将 taccScS 转换为 ((2, 2), MMA_N, MMA_N) 的形状，并仅获取行索引
    Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
    // 创建一个元素类型为 ElementAccum 的张量 lse，其形状由 taccScS_row 的大小决定
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    // 对 lse 中的每个元素进行迭代
    for (int mi = 0; mi < size(lse); ++mi) {
        // 获取 taccScS_row 中索引为 mi 的元素的第一个维度值作为行索引
        const int row = get<0>(taccScS_row(mi));
        // 如果 Is_even_MN 为真或者行索引小于有效序列长度减去 m_block 乘以 kBlockM，则将 gLSE(row) 赋给 lse(mi)，否则赋给 INFINITY
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    // 当行索引越界时，我们希望 LSE = inf。在这种情况下，Q 将为零，K 也将为零，得分也将为零。
    // 若 LSE = 0，则概率将全为 1，与 V 相乘（V 为零）也没问题。但使用 ALiBi 时，可能会修改这些得分，概率可能变为 NaN。
    // 因此，对于越界行，我们将 LSE 设为 inf，这样概率始终为 0。

    // 使用指定的条件复制函数，将 gmem_tiled_copy_QKV 中的数据复制到 tKgK 和 tKsK 中
    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    // 如果 Kernel_traits::Is_V_in_regs 为假，则使用指定的条件复制函数，将 gmem_tiled_copy_QKV 中的数据复制到 tVgV 和 tVsV 中
    if (!Kernel_traits::Is_V_in_regs) {
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
    }
    // 执行异步复制操作的同步屏障
    pytorch_flash::cp_async_fence();

    // 如果当前线程为 0，则打印 tdOgdO 的布局信息
    // if (cute::thread0()) { print(tdOgdO.layout()); printf("\n"); print(tdOrdO); print(tdOrO); }
    // 如果是第一次迭代
    if (Is_first) {
        // 将 tdOrdO 复制到 tdOsdO
        cute::copy(tdOrdO, tdOsdO);
        // 执行 dot_do_o 操作，计算内核函数中的一部分
        dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, gdPsum,
                                                    Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
    }

    // 如果内核函数要求将 V 存储在寄存器中
    if (Kernel_traits::Is_V_in_regs) {
        // 等待异步复制操作完成
        cute::cp_async_wait<1>();
        // 同步线程，确保前面的所有操作完成
        __syncthreads();
        // 将 tdPrV 在共享内存中进行视图重组
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        // 静态断言，验证 tdPsV 和 tdPrV_copy_view 的大小相同
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
        // 将 smem_tiled_copy_KV 中的数据复制到 tdPsV 和 tdPrV_copy_view
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    // 从 CUDA 的 philox 参数中解包种子和偏移量
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    unsigned long long seed = std::get<0>(seeds);
    unsigned long long offset = std::get<1>(seeds);
    // 创建 Dropout 对象，用于执行随机丢弃操作
    pytorch_flash::Dropout dropout(seed, offset, params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // 清空 acc_dv 和 acc_dk 中的数据
    clear(acc_dv);
    clear(acc_dk);

    // 计算 alibi_slope 值，如果没有 alibi 或 alibi_slopes_ptr 为 nullptr，则设为 0.0f
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    // 创建 Alibi 对象，用于处理 Alibi 相关逻辑
    pytorch_flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    // 如果启用了 Dropout
    if (Is_dropout) {
        // 对 acc_dv 中的每个元素乘以 params.rp_dropout
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    // 对 acc_dk 中的每个元素乘以 params.scale_softmax_rp_dropout
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // 将 acc_dk 和 acc_dv 中的数据从 fp32 转换为 fp16
    Tensor rdK = pytorch_flash::convert_type<Element>(acc_dk);
    Tensor rdV = pytorch_flash::convert_type<Element>(acc_dv);

    // 创建 sdK 和 sdV 张量，用于表示 SMEM 中的数据布局
    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // 创建 smem_tiled_copy_dKV，用于处理 SMEM 中数据的复制和分割
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    // 将 rdK 和 sdK 中的数据重组和分割，以匹配累加器的分区
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // 在写入 sK 和 sV 的相同位置时，需要使用 syncthreads 进行同步
    // 否则，某些线程可能在 dQ gemm 中读取 sK 时，同时修改其位置，导致竞争条件
    // 如果是最后一次迭代，循环的结尾已经包含了 __syncthreads()
    if (!Is_last) { __syncthreads(); }

    // 将 taccdKrdK 复制到 taccdKsdK 中，使用 smem_tiled_copy_dKV
    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    // 使用 cute 库的 copy 函数，将 taccdVrdV 的数据复制到 taccdVsdV 中

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
       + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    // 计算 dk 数据的行偏移量，用于访问 params.dk_ptr 中的数据

    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
       + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    // 计算 dv 数据的行偏移量，用于访问 params.dv_ptr 中的数据

    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dk_row_stride, _1{}));
    // 创建 gdK 张量，用于从全局内存中读取 dk 数据

    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dv_row_stride, _1{}));
    // 创建 gdV 张量，用于从全局内存中读取 dv 数据

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    // 根据 Kernel_traits 定义的类型，创建 gmem_tiled_copy_dKV 对象

    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    // 获取 gmem_tiled_copy_dKV 对象在当前线程 tidx 上的切片

    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    // 使用 gmem_tiled_copy_dKV 对象对 sdK 进行切片，得到 tdKsdK 张量

    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    // 使用 gmem_tiled_copy_dKV 对象对 gdK 进行切片，得到 tdKgdK 张量

    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    // 使用 gmem_tiled_copy_dKV 对象对 sdV 进行切片，得到 tdVsdV 张量

    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
    // 使用 gmem_tiled_copy_dKV 对象对 gdV 进行切片，得到 tdVgdV 张量

    __syncthreads();
    // 同步所有线程，确保之前的所有内存操作都已完成

    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    // 创建一个形状与 tdKgdK 相同的 tdKrdK 张量

    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    // 使用 cute 库的 copy 函数，将 tdKsdK 的数据复制到 tdKrdK 中

    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    // 创建一个形状与 tdVgdV 相同的 tdVrdV 张量

    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    // 使用 cute 库的 copy 函数，将 tdVsdV 的数据复制到 tdVrdV 中

    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // 创建一个形状为 sdK 的尺寸的单位张量 cdKV

    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    // 使用 gmem_tiled_copy_dKV 对象对 cdKV 进行切片，得到 tdKVcdKV 张量

    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    // 创建一个形状与 tdKgdK 的最后维度大小相同的布尔张量 tdKVpdKV

    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // 遍历 tdKVpdKV 张量，根据 tdKVcdKV 张量中的数据设置其值，判断是否小于 params.d

    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    // 使用 pytorch_flash 库的 copy 函数，将 tdKrdK 的数据按指定条件复制到 tdKgdK 中

    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    // 使用 pytorch_flash 库的 copy 函数，将 tdVrdV 的数据按指定条件复制到 tdVgdV 中
// 结束 flash 命名空间

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Is_even_M, bool Is_even_K, typename Params>
inline __device__ void compute_dq_dk_dv(const Params &params) {

    // 为批次的块索引
    const int bidb = blockIdx.x;
    // 为头部的块索引
    const int bidh = blockIdx.y;
    // 线程索引
    const int tidx = threadIdx.x;

    // 计算 kBlockN 对 seqlen_k 取整后的最大块数
    const int n_block_max = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    if (n_block_max == 1) {
        // 如果只有一个块，调用特定模板函数并处理第一个块
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, true, true>(params, bidb, bidh, 0);
    } else {
        // 使用倒序迭代从 n_block_max - 1 到 0 可能会节省一个寄存器
        // 处理最后一个块
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, true, false>(params, bidb, bidh, n_block_max - 1);
        // 处理中间的块
        for (int n_block = n_block_max - 2; n_block > 0; n_block--) {
            compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, false, false>(params, bidb, bidh, n_block);
        }
        // 处理第一个块
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, false, true>(params, bidb, bidh, 0);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ void compute_dq_dk_dv_seqk_parallel(const Params &params) {

    // 为批次的块索引
    const int bidb = blockIdx.y;
    // 为头部的块索引
    const int bidh = blockIdx.z;

    // 如果是确定性的，每个线程块将对不同的 dQ_accum 缓冲区进行 atomicAdd 操作
    for (int n_block = blockIdx.x; n_block < (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN; n_block += gridDim.x) {
        // 调用特定模板函数，处理每个块的计算，Seq_parallel 设置为 true 表示序列化并行处理
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, false, false, /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace flash
```