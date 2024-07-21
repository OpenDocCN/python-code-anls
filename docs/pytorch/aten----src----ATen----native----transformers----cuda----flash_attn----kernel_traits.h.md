# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\kernel_traits.h`

```
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>  // 引入CUTE库中的copy算法

#include <cutlass/cutlass.h>  // 引入Cutlass库的主头文件
#include <cutlass/layout/layout.h>  // 引入Cutlass库中布局相关的头文件
#include <cutlass/numeric_types.h>  // 引入Cutlass库中的数值类型定义

namespace pytorch_flash{

using namespace cute;  // 使用cute命名空间

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;  // 使用给定的元素类型作为Element类型
    static constexpr bool Has_cp_async = true;  // 设置异步拷贝标志为true
#else
    using Element = cutlass::half_t;  // 否则，默认使用cutlass库中的half_t作为Element类型
    static constexpr bool Has_cp_async = false;  // 设置异步拷贝标志为false
#endif

    using ElementAccum = float;  // 定义累加器元素类型为float
    using index_t = int64_t;  // 定义索引类型为int64_t

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    // 根据elem_type类型选择不同的MMA原子操作结构体
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    // 在不支持SM80架构时，使用默认的MMA原子操作结构体
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    // 根据条件选择不同的共享内存复制原子操作结构体
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    // 在不支持SM75架构时，默认使用的共享内存复制原子操作结构体
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

// 如果Share_Q_K_smem为true，则强制Is_Q_in_regs为true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;  // 继承基类的Element类型
    using ElementAccum = typename Base::ElementAccum;  // 继承基类的ElementAccum类型
    using index_t = typename Base::index_t;  // 继承基类的index_t类型
    static constexpr bool Has_cp_async = Base::Has_cp_async;  // 继承基类的Has_cp_async标志
    using SmemCopyAtom = typename Base::SmemCopyAtom;  // 继承基类的SmemCopyAtom类型
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;  // 继承基类的SmemCopyAtomTransposed类型

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;  // 定义是否共享Q_K的标志
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;  // 如果Share_Q_K_smem为true，则强制Is_Q_in_regs为true

    // 定义线程数
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;  // 定义块的M维度大小
    static constexpr int kBlockN = kBlockN_;  // 定义块的N维度大小
    static constexpr int kHeadDim = kHeadDim_;  // 定义头部维度大小
    static_assert(kHeadDim % 32 == 0);  // 确保头部维度可以被32整除
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;  // 根据头部维度选择适当的共享内存块大小
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);  // 根据头部维度选择适当的全局内存块大小
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;  // 根据共享内存块大小选择适当的交织参数
};
    // 定义一个别名 TiledMma，表示使用 TiledMMA 的模板
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,  // 使用 Base 类的 MMA_Atom_Arch 类型作为参数
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 定义一个布局，Shape 是一个 3D 形状，这里是 4x1x1 或者 8x1x1 的线程组
        Tile<Int<16 * kNWarps>, _16, _16>>;  // 定义一个 Tile，指定大小为 16 * kNWarps x 16 x 16

    // 定义一个别名 SmemLayoutAtomQ，使用 decltype 推导类型
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},  // 使用 Swizzle 模板进行重排列，参数为 kSwizzle 和 3, 3
                    // 这里使用 kBlockKSmem，因为对于 d=128，使用 kHeadDim 会得到错误的结果
                    Layout<Shape<_8, Int<kBlockKSmem>>,  // 定义一个布局，Shape 是一个 2D 形状，这里是 8 x kBlockKSmem
                           Stride<Int<kBlockKSmem>, _1>>{}));  // 定义布局的步长

    // 定义一个别名 SmemLayoutQ，使用 decltype 推导类型，将 SmemLayoutAtomQ 转换为指定形状
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},  // 使用 SmemLayoutAtomQ 作为输入
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));  // 定义一个形状，Shape 是一个 2D 形状，这里是 kBlockM x kHeadDim

    // 定义一个别名 SmemLayoutKV，使用 decltype 推导类型，将 SmemLayoutAtomQ 转换为指定形状
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},  // 使用 SmemLayoutAtomQ 作为输入
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));  // 定义一个形状，Shape 是一个 2D 形状，这里是 kBlockN x kHeadDim

    // 定义一个别名 SmemLayoutVtransposed，使用 decltype 推导类型，组合 SmemLayoutKV 和 按行主序的形状
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));

    // 定义一个别名 SmemLayoutVtransposedNoSwizzle，使用 decltype 推导类型，获取 SmemLayoutVtransposed 的非重排列部分
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    // 定义一个别名 SmemLayoutAtomO，使用 decltype 推导类型
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},  // 使用 Swizzle 模板进行重排列，参数为 kSwizzle 和 3, 3
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,  // 定义一个布局，Shape 是一个 2D 形状，这里是 8 x kBlockKSmem
                           Stride<Int<kBlockKSmem>, _1>>{}));  // 定义布局的步长

    // 定义一个别名 SmemLayoutO，使用 decltype 推导类型，将 SmemLayoutAtomO 转换为指定形状
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},  // 使用 SmemLayoutAtomO 作为输入
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));  // 定义一个形状，Shape 是一个 2D 形状，这里是 kBlockM x kHeadDim

    // 定义一个别名 SmemCopyAtomO，表示一个元素的默认复制操作
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    // 定义一个别名 SmemCopyAtomOaccum，表示一个元素累加的默认复制操作
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    // 定义一个静态常量 kSmemQSize，表示 SmemLayoutQ 大小乘以每个元素的字节大小
    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    // 定义一个静态常量 kSmemKVSize，表示 SmemLayoutKV 大小乘以两倍的每个元素的字节大小
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    // 定义一个静态常量 kSmemSize，根据 Share_Q_K_smem 决定使用 SmemQSize 和 SmemKVSize 的最大值或它们的和
    static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;

    // 定义一个静态常量 kGmemElemsPerLoad，表示每次加载的 gmem 元素数量
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    // 静态断言，确保 kHeadDim 是 kGmemElemsPerLoad 的倍数
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    
    // kBlockKSmem 在 d=128 时比 kBlockKGmem 快 6-10%，因为避免了银行冲突
    // 定义一个静态常量 kGmemThreadsPerRow，表示每行 gmem 的线程数
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    // 静态断言，确保 kNThreads 是 kGmemThreadsPerRow 的倍数
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");

    // 使用 CACHEGLOBAL 替代 CACHEALWAYS，因为 Q 和 K/V 不会从同一地址读取，这样略微更快
    # 定义类型别名 Gmem_copy_struct，根据是否具有异步拷贝的能力选择不同的类型
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,  # 如果支持异步拷贝，则选择 SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t> 类型
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy  # 否则选择 DefaultCopy 类型
    >;
    
    # 定义类型别名 GmemTiledCopyQKV，用于执行按瓦片方式复制操作，用于键值查询操作
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},  # 使用 Gmem_copy_struct 进行复制操作
                        GmemLayoutAtom{},  # 使用默认的布局原子 GmemLayoutAtom
                        Layout<Shape<_1, _8>>{}));  # 使用形状为 <1, 8> 的布局，每次读取8个值

    # 定义类型别名 GmemTiledCopyO，用于执行按瓦片方式复制操作，用于输出操作
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},  # 使用 DefaultCopy 进行复制操作
                        GmemLayoutAtom{},  # 使用默认的布局原子 GmemLayoutAtom
                        Layout<Shape<_1, _8>>{}));  # 使用形状为 <1, 8> 的布局，每次存储8个值

    # 定义类型别名 GmemLayoutAtomOaccum，根据条件选择不同的线程布局
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,  # 如果 kBlockKSmem 等于 32，则选择 <16, 8> 线程布局，每行8个线程
        Layout<Shape<_16, _8>, Stride<_8, _1>>,  # 线程布局为 <16, 8>，步长为 <8, 1>
        Layout<Shape<_8, _16>, Stride<_16, _1>>  # 否则选择 <8, 16> 线程布局，每行16个线程，步长为 <16, 1>
    >;
    
    # 定义类型别名 GmemTiledCopyOaccum，用于执行按瓦片方式复制操作，用于输出积累操作
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},  # 使用 DefaultCopy 进行复制操作，用于累积元素
                        GmemLayoutAtomOaccum{},  # 使用选择的布局原子 GmemLayoutAtomOaccum
                        Layout<Shape<_1, _4>>{}));  # 使用形状为 <1, 4> 的布局，每次存储4个值

    # 定义类型别名 GmemLayoutAtomRotcossin，使用相同的布局原子 GmemLayoutAtom
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    
    # 定义类型别名 GmemTiledCopyRotcossin，用于执行按瓦片方式复制操作，用于正弦余弦旋转操作
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},  # 使用 UniversalCopy<uint64_t> 进行复制操作
                        GmemLayoutAtomRotcossin{},  # 使用相同的布局原子 GmemLayoutAtomRotcossin
                        Layout<Shape<_1, _4>>{}));  # 使用形状为 <1, 4> 的布局，每次加载4个值

    # 定义类型别名 GmemTiledCopyRotcossinCont，用于执行按瓦片方式复制操作，用于连续的正弦余弦旋转操作
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},  # 使用 DefaultCopy 进行复制操作
                        GmemLayoutAtomRotcossin{},  # 使用相同的布局原子 GmemLayoutAtomRotcossin
                        Layout<Shape<_1, _8>>{}));  # 使用形状为 <1, 8> 的布局，每次加载8个值
};

// Is_V_in_regs is an option to reduce smem usage, but will increase register pressure.
// No_double_buffer is another option to reduce smem usage, but will slow things down.
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
         bool Is_V_in_regs_=false, bool No_double_buffer_=false, typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Is_V_in_regs = Is_V_in_regs_;
    static constexpr bool No_double_buffer = No_double_buffer_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutNdKV == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);

    // Define tiled matrix-multiply architecture for various layouts
    using TiledMmaSdP = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
        Tile<Int<16 * AtomLayoutMSdP>, Int<16 * kNWarps / AtomLayoutMSdP>, _16>>;
    using TiledMmadKV = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<kNWarps / AtomLayoutNdKV>, _1>>,
        Tile<Int<16 * AtomLayoutNdKV>, Int<16 * kNWarps / AtomLayoutNdKV>, _16>>;
    using TiledMmadQ = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        Tile<Int<16 * AtomLayoutMdQ>, Int<16 * kNWarps / AtomLayoutMdQ>, _16>>;

    // Define shared memory layout for data organization
    using SmemLayoutAtomQdO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO = decltype(tile_to_shape(
        SmemLayoutAtomQdO{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    // 定义一个类型别名 SmemLayoutAtomKV，其类型为 composition 的结果，接受 Swizzle、Layout 作为参数
    // kSwizzle 为常数 3，Shape 包含 Int<kBlockM / kNWarps> 和 Int<kBlockKSmem>，Stride 包含 Int<kBlockKSmem> 和 _1
    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    // 定义一个类型别名 SmemLayoutKV，其类型为 tile_to_shape 的结果，接受 SmemLayoutAtomKV 和 make_shape 作为参数
    // make_shape 包含 Int<kBlockN> 和 Int<kHeadDim>
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    // 定义一个类型别名 SmemLayoutKtransposed，其类型为 composition 的结果，接受 SmemLayoutKV 和 make_layout 作为参数
    // make_layout 包含 Shape<Int<kHeadDim>, Int<kBlockN>> 和 GenRowMajor
    using SmemLayoutKtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));

    // 定义一个类型别名 SmemLayoutKtransposedNoSwizzle，其类型为 get_nonswizzle_portion 的结果，接受 SmemLayoutKtransposed 作为参数
    using SmemLayoutKtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutKtransposed{}));

    // 静态断言，确保 kBlockN 至少为 32
    static_assert(kBlockN >= 32);

    // 定义常量 kPBlockN，如果 kBlockN 大于等于 64，则为 64，否则为 32
    static constexpr int kPBlockN = kBlockN >= 64 ? 64 : 32;

    // 静态断言，确保 kPBlockN 只能是 16、32 或 64
    static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);

    // 定义常量 kSwizzlePdS，始终为 3
    static constexpr int kSwizzlePdS = 3;

    // 定义一个类型别名 SmemLayoutAtomPdS，其类型为 composition 的结果，接受 Swizzle、Layout 作为参数
    // kSwizzlePdS 为常数 3，Shape 包含 Int<kBlockM> 和 Int<kPBlockN>，Stride 包含 Int<kPBlockN> 和 _1
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, 3, 3>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                           Stride<Int<kPBlockN>, _1>>{}));

    // 定义一个类型别名 SmemLayoutPdS，其类型为 tile_to_shape 的结果，接受 SmemLayoutAtomPdS 和 make_shape 作为参数
    // make_shape 包含 Int<kBlockM> 和 Int<kBlockN>
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));

    // 定义一个类型别名 SmemLayoutPdStransposed，其类型为 composition 的结果，接受 SmemLayoutPdS 和 make_layout 作为参数
    // make_layout 包含 Shape<Int<kBlockN>, Int<kBlockM>> 和 GenRowMajor
    using SmemLayoutPdStransposed = decltype(
        composition(SmemLayoutPdS{}, make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, GenRowMajor{})));

    // 定义一个类型别名 SmemLayoutPdStransposedNoSwizzle，其类型为 get_nonswizzle_portion 的结果，接受 SmemLayoutPdStransposed 作为参数
    using SmemLayoutPdStransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutPdStransposed{}));

    // 定义一个类型别名 SmemCopyAtomPdS，其类型为 Copy_Atom 的结果，接受 DefaultCopy 和 elem_type 作为参数
    using SmemCopyAtomPdS = Copy_Atom<DefaultCopy, elem_type>;

    // 定义一个类型别名 SmemLayoutQdOtransposed，其类型为 composition 的结果，接受 SmemLayoutQdO 和 make_layout 作为参数
    // make_layout 包含 Shape<Int<kHeadDim>, Int<kBlockM>> 和 GenRowMajor
    using SmemLayoutQdOtransposed = decltype(
        composition(SmemLayoutQdO{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));

    // 定义一个类型别名 SmemLayoutQdOtransposedNoSwizzle，其类型为 get_nonswizzle_portion 的结果，接受 SmemLayoutQdOtransposed 作为参数
    using SmemLayoutQdOtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQdOtransposed{}));

    // 定义一个类型别名 SmemLayoutAtomdKV，其类型为 composition 的结果，接受 Swizzle、Layout 作为参数
    // kSwizzle 为常数 3，Shape 包含 _8 和 Int<kBlockKSmem>，Stride 包含 Int<kBlockKSmem> 和 _1
    using SmemLayoutAtomdKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    // 定义一个类型别名 SmemLayoutdKV，其类型为 tile_to_shape 的结果，接受 SmemLayoutAtomdKV 和 make_shape 作为参数
    // make_shape 包含 Int<kBlockN> 和 Int<kHeadDim>
    using SmemLayoutdKV = decltype(tile_to_shape(
        SmemLayoutAtomdKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    // 定义一个类型别名 SmemCopyAtomdKV，其类型为 Copy_Atom 的结果，接受 DefaultCopy 和 elem_type 作为参数
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;

    // 定义一个类型别名 SmemLayoutAtomdQ，其类型为 composition 的结果，接受 Swizzle、Layout 作为参数
    // kSwizzle 为常数 3，Shape 包含 _8 和 Int<kBlockKSmem>，Stride 包含 Int<kBlockKSmem> 和 _1
    using SmemLayoutAtomdQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    // 定义一个类型别名 SmemLayoutdQ，其类型为 tile_to_shape 的结果，接受 SmemLayoutAtomdQ 和 make_shape 作为参数
    // make_shape 包含 Int<kBlockM> 和 Int<kHeadDim>
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    // 定义一个类型别名 SmemCopyAtomdQ，使用默认拷贝策略和元素类型 elem_type
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;

    // 计算双缓冲情况下的 sQ 的静态内存大小
    static constexpr int kSmemQdOSize = size(SmemLayoutQdO{}) * (No_double_buffer ? 2 : 3) * sizeof(Element);
    // 计算 KV 数据的静态内存大小
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    // 计算 dS 数据的静态内存大小
    static constexpr int kSmemdSSize = size(SmemLayoutPdS{}) * sizeof(Element);
    // 计算 P 数据的静态内存大小
    static constexpr int kSmemPSize = size(SmemLayoutPdS{}) * sizeof(Element);
    // 计算 dQ 数据的静态内存大小
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);

    // 计算总的静态内存大小，考虑不同的条件
    static constexpr int kSmemSize = kSmemQdOSize
        + (!Is_V_in_regs
           ? kSmemKVSize + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)
           : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)));

    // 计算在单列块模式下的静态内存大小
    static constexpr int kSmemSize1colblock = kSmemQdOSize
        + (!Is_V_in_regs
           ? kSmemKVSize + kSmemdSSize + kSmemPSize
           : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + kSmemPSize));

    // 定义每次加载的全局内存元素数量，使用 cute::uint128_t 类型
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    // 确保 kHeadDim 是 kGmemElemsPerLoad 的倍数，以避免错误
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");

    // 计算每行的全局内存线程数，以避免线程数与全局内存元素数量的冲突
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    // 确保 kNThreads 是 kGmemThreadsPerRow 的倍数，以保证全局内存线程分配的正确性
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");

    // 定义全局内存的布局结构 GmemLayoutAtom
    using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    // 根据是否支持异步拷贝，选择全局内存的拷贝结构
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;

    // 定义处理 QKV 数据的分块拷贝类型 GmemTiledCopyQKV
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    // 定义处理 dO 数据的分块拷贝类型 GmemTiledCopydO
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    // 定义处理 dKV 数据的分块拷贝类型 GmemTiledCopydKV
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    // 定义处理 dQ 数据的分块拷贝类型 GmemTiledCopydQ
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    // 定义模板别名 GmemLayoutAtomdQaccum，条件化于 kBlockKSmem 是否等于 32
    // 如果是，使用 Shape<_32, _8> 和 Stride<_8, _1> 的布局，表示每行8个线程
    // 否则，使用 Shape<_16, _16> 和 Stride<_16, _1> 的布局，表示每行16个线程
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape<_32, _8>,  // Thread layout, 8 threads per row
               Stride<_8, _1>>,
        Layout<Shape<_16, _16>,  // Thread layout, 16 threads per row
               Stride<_16, _1>>
    >;

    // 定义模板别名 GmemTiledCopydQaccum，使用 make_tiled_copy 创建基于 GmemLayoutAtomdQaccum 的复制操作
    // 使用 Layout<Shape<_1, _4>>，表示每个存储单元包含4个值
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per store

    // 定义模板别名 GmemTiledCopydQaccumAtomicAdd，使用 make_tiled_copy 创建原子加法操作
    // 使用 Layout<Shape<_8, _32>, Stride<_32, _1>>，表示每行8个线程，每行内部的步长为32
    // 使用 Layout<Shape<_1, _1>>，表示每个存储单元包含1个值
    using GmemTiledCopydQaccumAtomicAdd = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        Layout<Shape<_8, _32>,  // Thread layout, 8 threads per row
                               Stride<_32, _1>>{},
                        Layout<Shape<_1, _1>>{}));  // Val layout, 1 val per store
};

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace pytorch_flash
```