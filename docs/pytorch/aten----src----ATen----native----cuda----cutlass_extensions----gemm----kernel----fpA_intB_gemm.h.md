# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\kernel\fpA_intB_gemm.h`

```
/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include <cutlass/cutlass.h>

#include <cutlass/arch/arch.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/semaphore.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Mma_,                 ///! Threadblock-scoped matrix multiply-accumulate
         typename Epilogue_,            ///! Epilogue
         typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
         typename KernelArch,           ///! The Architecture this kernel is compiled for. Used since SIMT kernels lose top-level arch.
         bool SplitKSerial              ///! If true, code supporting split-K via serial reduction is enabled.
         >
struct GemmFpAIntB {

    using Mma                       = Mma_;               // 定义 Mma 为模板参数 Mma_
    using Epilogue                  = Epilogue_;          // 定义 Epilogue 为模板参数 Epilogue_
    using ThreadblockSwizzle        = ThreadblockSwizzle_; // 定义 ThreadblockSwizzle 为模板参数 ThreadblockSwizzle_
    using ArchTag                   = KernelArch;         // 定义 ArchTag 为模板参数 KernelArch
    static bool const kSplitKSerial  = SplitKSerial;       // 定义并初始化 kSplitKSerial 为模板参数 SplitKSerial 的值
    // 定义别名，表示 Epilogue 的输出操作类型
    using EpilogueOutputOp = typename Epilogue::OutputOp;
    // 定义别名，表示 ThreadblockSwizzle 类型
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    // 定义常量布尔值，表示是否分割 K 维度进行串行化
    static bool const kSplitKSerial = SplitKSerial;

    // 定义别名，表示 Mma 的 A 迭代器元素类型
    using ElementA = typename Mma::IteratorA::Element;
    // 定义别名，表示 Mma 的 A 迭代器布局类型
    using LayoutA = typename Mma::IteratorA::Layout;
    // 定义别名，表示 Mma 的 B 迭代器元素类型
    using ElementB = typename Mma::IteratorB::Element;
    // 定义别名，表示 Mma 的 B 迭代器布局类型
    using LayoutB = typename Mma::IteratorB::Element;
    // 定义别名，表示 Epilogue 的输出瓦片迭代器元素类型
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    // 定义别名，表示 Mma 的 C 矩阵布局类型
    using LayoutC = typename Mma::LayoutC;
    // 定义别名，表示元素缩放类型，与 ElementC 相同
    using ElementScale = ElementC;

    // 定义静态常量，表示是否对 A 矩阵进行复数变换
    static ComplexTransform const kTransformA = Mma::kTransformA;
    // 定义静态常量，表示是否对 B 矩阵进行复数变换
    static ComplexTransform const kTransformB = Mma::kTransformA;

    // Type definitions about the mainloop.
    // 定义别名，表示 Mma 的操作器类型
    using Operator = typename Mma::Operator;
    // 定义别名，表示 Mma 的操作器类别类型
    using OperatorClass = typename Mma::Operator::OperatorClass;
    // 定义别名，表示 Mma 的 ThreadblockShape 类型
    using ThreadblockShape = typename Mma::Shape;
    // 定义别名，表示 Mma 的 WarpShape 类型
    using WarpShape = typename Mma::Operator::Shape;
    // 定义别名，表示 Mma 的指令形状类型
    using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
    // 定义别名，表示 Mma 的架构标签类型
    using ArchTag = typename Mma::ArchTag;

    // 定义静态常量，表示 Mma 的阶段数
    static int const kStages = Mma::kStages;
    // 定义静态常量，表示 Mma 的 A 迭代器访问类型的对齐
    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    // 定义静态常量，表示 Mma 的 B 迭代器访问类型的对齐
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    // 定义静态常量，表示 Epilogue 的输出瓦片迭代器每次访问的元素数
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    /// Warp count (concept: GemmShape)
    // 定义别名，表示 Mma 的 WarpCount 类型
    using WarpCount = typename Mma::WarpCount;
    // 计算总的线程数，每个 Warp 包含 32 个线程
    static int const kThreadCount = 32 * WarpCount::kCount;

    // 计算 B 矩阵的行交错因子
    static constexpr int kInterleave = Mma::IteratorB::Shape::kRow / Mma::Shape::kK;

    /// Parameters structure
    // 参数结构的开始
    // 定义一个结构体 Arguments，用于存储 GEMM 矩阵乘法的参数和配置信息
    struct Arguments {
        GemmUniversalMode mode = GemmUniversalMode::kGemm;  // 指定默认的 GEMM 模式为普通矩阵乘法

        cutlass::gemm::GemmCoord                         problem_size;       // 存储矩阵乘法的问题尺寸信息
        typename Mma::IteratorA::TensorRef               ref_A;              // 存储矩阵 A 的迭代器引用
        typename Mma::IteratorB::TensorRef               ref_B;              // 存储矩阵 B 的迭代器引用
        typename Mma::IteratorScale::TensorRef           ref_scale;          // 存储乘法的标量因子的迭代器引用
        typename Epilogue::OutputTileIterator::TensorRef ref_C;              // 存储矩阵 C 的迭代器引用
        typename Epilogue::OutputTileIterator::TensorRef ref_D;              // 存储矩阵 D 的迭代器引用

        // 控制串行化的 split-k
        int batch_count;                                // 存储批处理的计数器

        typename EpilogueOutputOp::Params output_op;    // 存储输出操作的参数

        // 用于 gather+scatter 操作
        int const* gather_A_indices;                    // 存储矩阵 A 的索引数组
        int const* gather_B_indices;                    // 存储矩阵 B 的索引数组
        int const* scatter_D_indices;                   // 存储矩阵 D 的索引数组

        // 用于支持 Gemm Universal
        int batch_stride_D = 0;                         // 存储批处理中矩阵 D 的跨度

        //
        // 方法
        //

        CUTLASS_HOST_DEVICE
        Arguments() {}                                  // 默认构造函数，无参数

        CUTLASS_HOST_DEVICE
        // 构造函数，初始化所有成员变量
        Arguments(cutlass::gemm::GemmCoord const&                  problem_size,
                  typename Mma::IteratorA::TensorRef               ref_A,
                  typename Mma::IteratorB::TensorRef               ref_B,
                  typename Mma::IteratorScale::TensorRef           ref_scale,
                  typename Epilogue::OutputTileIterator::TensorRef ref_C,
                  typename Epilogue::OutputTileIterator::TensorRef ref_D,
                  int                                              serial_split_k_factor,
                  typename EpilogueOutputOp::Params                output_op = typename EpilogueOutputOp::Params(),
                  int const*                                       gather_A_indices  = nullptr,
                  int const*                                       gather_B_indices  = nullptr,
                  int const*                                       scatter_D_indices = nullptr):
            // 使用参数初始化各个成员变量
            problem_size(problem_size),
            ref_A(ref_A),
            ref_B(ref_B),
            ref_scale(ref_scale),
            ref_C(ref_C),
            ref_D(ref_D),
            batch_count(serial_split_k_factor),
            output_op(output_op),
            gather_A_indices(gather_A_indices),
            gather_B_indices(gather_B_indices),
            scatter_D_indices(scatter_D_indices)
        {
        }
    };

    /// Parameters structure
    // 参数结构体
    struct Params
    {
        // 定义用于矩阵乘法的问题尺寸坐标
        cutlass::gemm::GemmCoord                         problem_size;
        // 定义用于网格划分的形状坐标
        cutlass::gemm::GemmCoord                         grid_tiled_shape;
        // 控制瓦片化形状的对数值
        int                                              swizzle_log_tile;
        // 迭代器A的参数类型和引用
        typename Mma::IteratorA::Params                  params_A;
        typename Mma::IteratorA::TensorRef               ref_A;
        // 迭代器B的参数类型和引用
        typename Mma::IteratorB::Params                  params_B;
        typename Mma::IteratorB::TensorRef               ref_B;
        // 迭代器Scale的参数类型和引用
        typename Mma::IteratorScale::Params              params_scale;
        typename Mma::IteratorScale::TensorRef           ref_scale;
        // Epilogue输出矩阵迭代器C的参数类型和引用
        typename Epilogue::OutputTileIterator::Params    params_C;
        typename Epilogue::OutputTileIterator::TensorRef ref_C;
        // Epilogue输出矩阵迭代器D的参数类型和引用
        typename Epilogue::OutputTileIterator::Params    params_D;
        typename Epilogue::OutputTileIterator::TensorRef ref_D;
        // Epilogue输出操作的参数类型
        typename EpilogueOutputOp::Params                output_op;
        // 信号量指针
        int*                                             semaphore;
        // GEMM操作中K维度的大小
        int                                              gemm_k_size;
        // 用于gather+scatter操作的常量指针数组
        // A的gather索引
        int const* gather_A_indices;
        // B的gather索引
        int const* gather_B_indices;
        // D的scatter索引
        int const* scatter_D_indices;

        //
        // Methods
        //

        // 默认构造函数，初始化参数
        Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0) {}

        // 带参数的构造函数，根据给定参数初始化对象
        Params(Arguments const&                args,
               int                             device_sms,
               int                             sm_occupancy):
            // 初始化问题尺寸
            problem_size(args.problem_size),
            // 根据网格划分和瓦片化策略获取对数化的瓦片大小
            swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
            // 初始化迭代器A的参数和引用
            params_A(args.ref_A.layout()),
            ref_A(args.ref_A),
            // 初始化迭代器B的参数和引用
            params_B(args.ref_B.layout()),
            ref_B(args.ref_B),
            // 初始化迭代器Scale的参数和引用
            params_scale(args.ref_scale.layout()),
            ref_scale(args.ref_scale),
            // 初始化迭代器C的参数和引用
            params_C(args.ref_C.layout()),
            ref_C(args.ref_C),
            // 初始化迭代器D的参数和引用
            params_D(args.ref_D.layout()),
            ref_D(args.ref_D),
            // 初始化输出操作的参数
            output_op(args.output_op),
            // 初始化gather操作的A索引
            gather_A_indices(args.gather_A_indices),
            // 初始化gather操作的B索引
            gather_B_indices(args.gather_B_indices),
            // 初始化scatter操作的D索引
            scatter_D_indices(args.scatter_D_indices)
        {
            ThreadblockSwizzle swizzle;
            // 根据问题尺寸、线程块形状和批次数量获取网格划分形状
            grid_tiled_shape = swizzle.get_tiled_shape(
                args.problem_size,
                {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
                args.batch_count);

            // 获取GEMM操作中K维度的大小
            gemm_k_size = args.problem_size.k();
        }

        // 获取工作空间大小，此处未实际分配工作空间
        size_t get_workspace_size() const
        {
            return 0;
        }

        // 初始化工作空间，此处无需初始化
        Status init_workspace(void *workspace,cudaStream_t stream = nullptr)
        {
            return Status::kSuccess;
        }

        // 获取网格维度大小
        dim3 get_grid_dims() const
        {
            return ThreadblockSwizzle().get_grid_shape(grid_tiled_shape);
        }
    };

    /// Shared memory storage structure
    // 共享存储联合体，用于在两种不同的计算内核间切换
    union SharedStorage {
        // 主循环计算内核的共享存储
        typename Mma::SharedStorage      main_loop;
        // 结束计算内核的共享存储
        typename Epilogue::SharedStorage epilogue;
    };

    //
    // Methods
    //

    // 默认构造函数，初始化一个 GEMM 计算的 FP16 A 型整数 B 型矩阵乘法
    CUTLASS_HOST_DEVICE
    GemmFpAIntB() {}

    /// 判断当前的计算内核是否满足对齐要求
    CUTLASS_HOST_DEVICE
    static Status can_implement(Arguments const& args)
    {

        // 确定输入矩阵 A 的对齐要求
        static int const kAlignmentA =
            (platform::is_same<typename Mma::IteratorA::Layout, layout::ColumnMajorInterleaved<32>>::value) ?
                32 :
            (platform::is_same<typename Mma::IteratorA::Layout, layout::ColumnMajorInterleaved<64>>::value) ?
                64 :
                Mma::IteratorA::AccessType::kElements;

        // 确定输入矩阵 B 的对齐要求
        static int const kAlignmentB =
            (platform::is_same<typename Mma::IteratorB::Layout, layout::RowMajorInterleaved<32>>::value) ?
                32 :
            (platform::is_same<typename Mma::IteratorB::Layout, layout::RowMajorInterleaved<64>>::value) ?
                64 :
                Mma::IteratorB::AccessType::kElements;

        // 确定输入矩阵乘积的缩放因子的对齐要求
        static int const kAlignmentScale = Mma::IteratorScale::AccessType::kElements;

        // 确定输出矩阵 C 的对齐要求
        static int const kAlignmentC = (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                          layout::ColumnMajorInterleaved<32>>::value) ?
                                           32 :
                                       (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                          layout::ColumnMajorInterleaved<64>>::value) ?
                                           64 :
                                           Epilogue::OutputTileIterator::kElementsPerAccess;

        // 检查输入张量 args.ref_A 是否满足对齐要求 kAlignmentA
        if (!TensorRef_aligned(args.ref_A, kAlignmentA)) {
            return Status::kErrorMisalignedOperand;
        }

        // 检查输入张量 args.ref_B 是否满足对齐要求 kAlignmentB
        if (!TensorRef_aligned(args.ref_B, kAlignmentB)) {
            return Status::kErrorMisalignedOperand;
        }

        // 检查缩放因子张量 args.ref_scale 是否满足对齐要求 kAlignmentScale
        if (!TensorRef_aligned(args.ref_scale, kAlignmentScale)) {
            return Status::kErrorMisalignedOperand;
        }

        // 检查输出张量 args.ref_C 是否满足对齐要求 kAlignmentC
        if (!TensorRef_aligned(args.ref_C, kAlignmentC)) {
            return Status::kErrorMisalignedOperand;
        }

        // 检查输出张量 args.ref_D 是否满足对齐要求 kAlignmentC
        if (!TensorRef_aligned(args.ref_D, kAlignmentC)) {
            return Status::kErrorMisalignedOperand;
        }

        // 所有对齐要求均满足，返回成功状态
        return Status::kSuccess;
    }

    // 模板结构体，用于在标准早于 C++17 的编译器下编译该代码
    // 即使未使用，dummy 模板参数仍然存在于命名空间中以满足全特化模板的要求
    template<bool B, typename dummy = void>
    struct KernelRunner {
        // 如果未实现，运行内核时会调用未实现函数的错误处理
        CUTLASS_DEVICE
        static void run_kernel(Params const& params, SharedStorage& shared_storage)
        {
            CUTLASS_NOT_IMPLEMENTED();
        }
    };

    // 模板结构体的尾部，缺少实现导致编译错误
    template<typename dummy>
    };

    // CUTLASS_DEVICE 宏定义的设备函数，目前无具体实现
    CUTLASS_DEVICE
    // 定义静态方法 `invoke`，接受 `Params` 和 `SharedStorage` 作为参数
    static void invoke(Params const &params, SharedStorage &shared_storage)
    {
        // 创建一个 `GemmFpAIntB` 的对象 `op`
        GemmFpAIntB op;
        // 调用 `op` 对象的调用运算符重载，传入 `params` 和 `shared_storage`
        op(params, shared_storage);
    }

    /*
        为了提高编译速度，如果 `CUDA_ARCH` 与 `cutlass` 内核操作符的 `ArchTag` 不对应，则不编译设备操作符。
    */
    /// 执行一个 GEMM 操作
    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
        // 如果 CUDA 架构版本在 7.0 到 7.5 之间
        static constexpr bool compile_needed = platform::is_same<KernelArch, arch::Sm70>::value;
        // 设置是否需要编译的标志，根据 KernelArch 是否与 arch::Sm70 相同来决定
        KernelRunner<compile_needed>::run_kernel(params, shared_storage);
        // 运行相应的核函数，根据 compile_needed 决定是否进行编译
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && (__CUDA_ARCH__ < 800)
        // 如果 CUDA 架构版本在 7.5 到 8.0 之间
        static constexpr bool compile_needed = platform::is_same<KernelArch, arch::Sm75>::value;
        // 设置是否需要编译的标志，根据 KernelArch 是否与 arch::Sm75 相同来决定
        KernelRunner<compile_needed>::run_kernel(params, shared_storage);
        // 运行相应的核函数，根据 compile_needed 决定是否进行编译
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 900)
        // 如果 CUDA 架构版本在 8.0 到 9.0 之间
        static constexpr bool compile_needed = platform::is_same<KernelArch, arch::Sm80>::value;
        // 设置是否需要编译的标志，根据 KernelArch 是否与 arch::Sm80 相同来决定
        KernelRunner<compile_needed>::run_kernel(params, shared_storage);
        // 运行相应的核函数，根据 compile_needed 决定是否进行编译
#else
        // 如果没有匹配到支持的 CUDA 架构版本
        CUTLASS_NOT_IMPLEMENTED();
        // 抛出未实现的错误，表示当前 CUDA 架构版本不支持
#endif
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
```