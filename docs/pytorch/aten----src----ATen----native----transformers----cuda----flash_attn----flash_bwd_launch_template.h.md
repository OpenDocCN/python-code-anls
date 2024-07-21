# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\flash_bwd_launch_template.h`

```py
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/cuda/CUDAException.h>

#include <ATen/native/transformers/cuda/flash_attn/static_switch.h>
#include <ATen/native/transformers/cuda/flash_attn/flash.h>
#include <ATen/native/transformers/cuda/flash_attn/flash_bwd_preprocess_kernel.h>
#include <ATen/native/transformers/cuda/flash_attn/flash_bwd_kernel.h>

namespace pytorch_flash {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#endif

#if defined(ARCH_SUPPORTS_FLASH) && defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 11 && \
    defined(__CUDACC_VER_MINOR__) && __CUDACC_VER_MINOR__ >= 8
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_BACKWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_bwd_params params)

// Define kernel for computing dq, dk, dv with optional dropout and causal settings
DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_dk_dv_loop_kernel, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Is_even_M, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
       // Call function to compute dq, dk, dv based on provided parameters
       pytorch_flash::compute_dq_dk_dv<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K>(params);
    #else
        // Output error message if FLASH is not supported on this architecture
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// Define kernel for computing dq, dk, dv in a sequence-parallel manner
DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
        // Assert condition to ensure Is_causal and Is_local are mutually exclusive
        static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
        // Call function to compute dq, dk, dv with sequence-parallel strategy
        pytorch_flash::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K>(params);
    #else
        // Output error message if FLASH is not supported on this architecture
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// Define kernel for computing dot product and output gradients
template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
    // Call function to compute dot product and gradients with specified traits
    pytorch_flash::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
}

// Define kernel for clearing accumulated gradients of key and value
template<typename Kernel_traits>
__global__ void flash_bwd_clear_dkvaccum_kernel(const Flash_bwd_params params) {
    // Call function to clear accumulated gradients of key and value
    pytorch_flash::clear_dKVaccum<Kernel_traits>(params);
}

// Define kernel for converting dq gradients
template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(const Flash_bwd_params params, const int nsplits) {
    // Call function to convert dq gradients based on kernel traits and split count
    pytorch_flash::convert_dQ<Kernel_traits>(params, nsplits);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_kernel(const Flash_bwd_params params) {
    // 调用 pytorch_flash 命名空间中的 convert_dKV 函数，传入参数 params
    pytorch_flash::convert_dKV<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd_seqk_parallel(Flash_bwd_params &params, cudaStream_t stream) {
    // 计算每个线程块在 M 方向上的数量
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);
    
    // 计算每个线程块在 N 方向上的数量
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = num_n_block;
    
    // 如果需要确定性行为，则动态调整 gridDimx 的值
    if (params.deterministic) {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        gridDimx = (dprops->multiProcessorCount + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_n(gridDimx, params.b, params.h);

    // 根据是否确定性选择调用不同版本的 flash_bwd_dot_do_o_kernel 函数
    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 判断是否满足 is_even_MN 的条件，用于在循环中应用掩码
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    
    // 判断是否满足 is_even_K 的条件
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    
    // 设置共享内存的大小为 smem_size_dq_dk_dv
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    // printf("smem_size_dq_dk_dv = %d\n", smem_size_dq_dk_dv);
}


这段代码主要涉及 CUDA 的核函数调用和一些参数计算。每个函数和变量的作用已经在注释中详细说明。
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
                LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        // 如果不是 IsEvenKConst，我们还将 IsEvenMNConst 设置为 false 以减少模板的数量。
                        // 如果 head dim > 128，将 IsEvenMNConst 设置为 false 以减少模板的数量。
                        // 如果 Is_local 为 true，则将 Is_causal 设置为 false。
                        auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst>;
                        // auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, false, Is_causal, false, false, true, true>;
                        if (smem_size_dq_dk_dv >= 48 * 1024)  {
                            // 设置 kernel 函数的动态共享内存大小
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                        }
                        // 调用 kernel 函数进行 GPU 加速计算
                        kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                        // 检查 CUDA 核函数调用是否成功
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });

    // 设置 kernel_dq 指针指向 flash_bwd_convert_dq_kernel<Kernel_traits> 函数模板
    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    // 如果 Kernel_traits::kSmemdQSize 大于等于 48 * 1024
    if (Kernel_traits::kSmemdQSize >= 48 * 1024)  {
        // 设置 kernel_dq 函数的动态共享内存大小
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    // 调用 kernel_dq 函数进行 GPU 加速计算
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    // 检查 CUDA 核函数调用是否成功
    C10_CUDA_KERNEL_LAUNCH_CHECK();
// 模板函数，用于执行带有闪存注意力机制的反向传播。根据Is_dropout参数决定是否执行反向传播。
template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    // 如果未定义FLASHATTENTION_DISABLE_BACKWARD宏，则执行序列化并行的闪存注意力机制反向传播
    #ifndef FLASHATTENTION_DISABLE_BACKWARD
    run_flash_bwd_seqk_parallel<Kernel_traits, Is_dropout>(params, stream);
    #endif
}

// 模板函数，执行带有32维头部维度的多头注意力机制的反向传播。
template<typename T>
void run_mha_bwd_hdim32(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;  // 头部维度为32
    int device;
    cudaGetDevice(&device);  // 获取当前设备编号
    int max_smem_per_block;
    // 查询当前设备支持的每个线程块的最大共享内存量（优化版）
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
        C10_CUDA_CHECK(status_);  // 检查CUDA函数执行状态
    }
    // 如果参数p_dropout小于1，使用DROPOUT_SWITCH宏开关，决定是否执行以下代码块
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 如果最大共享内存量足够支持特定计算量，执行以下代码块
        if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) { // 104 KB
            // 如果Is_dropout为false，使用更多寄存器保持V（值）数据，执行带有指定参数的闪存注意力机制反向传播
            if constexpr(!Is_dropout) {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream);
            } else {  // 否则执行带有不同参数的闪存注意力机制反向传播
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream);
            }
        } else {  // 如果最大共享内存量不足，执行默认参数的闪存注意力机制反向传播
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream);
        }
    });
}

// 模板函数，执行带有64维头部维度的多头注意力机制的反向传播。
template<typename T>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;  // 头部维度为64
    int device;
    cudaGetDevice(&device);  // 获取当前设备编号
    int max_smem_per_block;
    // 查询当前设备支持的每个线程块的最大共享内存量（优化版）
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
        C10_CUDA_CHECK(status_);  // 检查CUDA函数执行状态
    }
    // 如果参数p_dropout小于1，使用DROPOUT_SWITCH宏开关，决定是否执行以下代码块
    // printf("max_smem_per_block = %d\n", max_smem_per_block);  // 打印当前最大共享内存量（调试用）
    // 如果参数 p_dropout 小于 1，则启用 dropout
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 改变 AtomLayoutMdQ 从 2 到 4 并不增加时间消耗
        // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=64, N=128, K=8, M_split=2, N_split=4, K_split=2, 不使用 dropout
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>>(params, stream);
        // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=64, N=128, K=8, M_split=2, N_split=4, K_split=2, 使用 dropout
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, true, false, T>>(params, stream);
        // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=128, N=128, K=8, M_split=2, N_split=4, K_split=4, 不使用 dropout
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 2, 4, 4, false, false, T>>(params, stream);
        // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=128, N=64, K=8, M_split=4, N_split=2, K_split=4, 不使用 dropout
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream);
        
        // 如果每个块的最大共享内存大于等于 144 KB
        if (max_smem_per_block >= 144 * 1024) {
            // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=128, N=128, K=8, M_split=4, N_split=4, K_split=4, 不使用 dropout
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream);
            // 这种设置会有大量寄存器溢出
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream);
        } else {
            // 如果 params.h == params.h_k
            // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=64, N=128, K=8, M_split=2, N_split=4, K_split=4, 使用 dropout
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream);
            // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=128, N=64, K=8, M_split=4, N_split=2, K_split=4, 不使用 dropout
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream);
            // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=128, N=64, K=8, M_split=4, N_split=2, K_split=4, 使用 dropout
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, true, false, T>, Is_dropout>(params, stream);
        }
    });
    // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=128, N=64, K=8, M_split=4, N_split=2, K_split=4, 使用 dropout
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, true, false, T>>(params, stream);
    // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=64, N=64, K=4, M_split=2, N_split=2, K_split=2, 使用 dropout
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 4, 2, 2, 2, true, false, T>>(params, stream);
    // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=32, N=128, K=4, M_split=1, N_split=4, K_split=1, 不使用 dropout
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 128, 4, 1, 4, 1, false, false, T>>(params, stream);
    // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=16, N=128, K=4, M_split=1, N_split=4, K_split=1, 不使用 dropout
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 16, 128, 4, 1, 4, 1, false, false, T>>(params, stream);
    // M=128, N=64 的计算速度较慢，可能是因为需要多次读写 dQaccum
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 2, 2, 2, false, T>>(params, stream);
    // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=128, N=64, K=8, 不使用 dropout
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, false, T>>(params, stream);
    // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=64, N=64, K=4, 不使用 dropout
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 4, false, T>>(params, stream);

    // 使用 Flash_bwd_kernel_traits 参数运行反向传播，M=128, N=64, K=4, M_split=4, N_split=2, K_split=4, 不使用 dropout
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 4, 4, 2, 4, false, false, T>>(params, stream);
}

template<typename T>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    // 定义头部维度为96
    constexpr static int Headdim = 96;
    // 获取当前设备编号
    int device;
    cudaGetDevice(&device);
    // 定义每个线程块的最大共享内存大小
    int max_smem_per_block;
    // 查询设备属性以获取最大可选共享内存大小
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // 检查CUDA函数调用是否成功
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // 如果需要，打印最大可选共享内存大小
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    // 使用宏定义处理可能的dropout情况
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 如果最大共享内存大于等于116KB，选择不同的内核调用配置
        if (max_smem_per_block >= 116 * 1024) {
            // 如果不是dropout，选择较小的内存配置
            if constexpr(!Is_dropout) {  // 92KB
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream);
            } else {  // 116 KB
                // dropout情况下选择更快的配置，因为寄存器不足
                // 运行反向传播的FLASH算法
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_dropout>(params, stream);
            }
        } else {
            // 共享内存不足的情况下，选择默认配置
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream);
        }
    });
}

template<typename T>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    // 定义头部维度为128
    constexpr static int Headdim = 128;
    // 获取当前设备编号
    int device;
    cudaGetDevice(&device);
    // 定义每个线程块的最大共享内存大小
    int max_smem_per_block;
    // 查询设备属性以获取最大可选共享内存大小
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // 检查CUDA函数调用是否成功
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // 如果需要，打印最大可选共享内存大小
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 如果 dropout 概率小于 1，则执行以下代码块
        // 使用并行序列反向传播的 Flash_bwd 内核，使用以下参数：Headdim, 64, 128, 8, 2, 4, 2, false, false, T
        if (max_smem_per_block >= 144 * 1024) {
            // 如果每个块的共享内存大于等于 144 KB，则选择此路径
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>, Is_dropout>(params, stream);
        } else {
            // 如果每个块的共享内存小于 144 KB，则选择此路径
            // 使用 Flash_bwd 内核，参数为：Headdim, 64, 64, 8, 4, 2, 2, true, false, T
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, false, T>, Is_dropout>(params, stream);
        }
    });
// 定义一个模板函数，用于执行头尺寸为160的多头自注意力的反向传播
template<typename T>
void run_mha_bwd_hdim160(Flash_bwd_params &params, cudaStream_t stream) {
    // 定义头尺寸为160的常量
    constexpr static int Headdim = 160;
    // 获取当前设备的编号
    int device;
    cudaGetDevice(&device);
    // 定义每个线程块允许的最大共享内存大小
    int max_smem_per_block;
    // 查询当前设备的最大可选共享内存大小属性
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // 检查CUDA函数调用是否成功
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // 如果未启用丢弃功能，则执行以下代码块
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 如果每个线程块允许的最大共享内存大小大于或等于116KB
        if (max_smem_per_block >= 116 * 1024) {
            // 运行Flash反向传播，使用特定的内核特征和丢弃模式
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream);
        } else {
            // 否则，运行Flash反向传播，使用备用的内核特征和丢弃模式
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, true, T>, Is_dropout>(params, stream);
        }
    });
}

// 定义一个模板函数，用于执行头尺寸为192的多头自注意力的反向传播
template<typename T>
void run_mha_bwd_hdim192(Flash_bwd_params &params, cudaStream_t stream) {
    // 定义头尺寸为192的常量
    constexpr static int Headdim = 192;
    // 获取当前设备的编号
    int device;
    cudaGetDevice(&device);
    // 定义每个线程块允许的最大共享内存大小
    int max_smem_per_block;
    // 查询当前设备的最大可选共享内存大小属性
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // 检查CUDA函数调用是否成功
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // 如果未启用丢弃功能，则执行以下代码块
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 如果每个线程块允许的最大共享内存大小大于或等于136KB
        if (max_smem_per_block >= 136 * 1024) {
            // 运行Flash反向传播，使用特定的内核特征和丢弃模式
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream);
        } else {
            // 否则，运行Flash反向传播，使用备用的内核特征和丢弃模式
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, true, T>, Is_dropout>(params, stream);
        }
    });
}

// 定义一个模板函数，用于执行头尺寸为224的多头自注意力的反向传播
template<typename T>
void run_mha_bwd_hdim224(Flash_bwd_params &params, cudaStream_t stream) {
    // 定义头尺寸为224的常量
    constexpr static int Headdim = 224;
    // 如果未启用丢弃功能，则执行以下代码块
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 运行Flash反向传播，使用特定的内核特征和丢弃模式
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream);
    });
}

// 定义一个模板函数，用于执行头尺寸为256的多头自注意力的反向传播
template<typename T>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    // 定义头尺寸为256的常量
    constexpr static int Headdim = 256;
    // 获取当前设备的编号
    int device;
    cudaGetDevice(&device);
    // 定义每个线程块允许的最大共享内存大小
    int max_smem_per_block;
    // 查询当前设备的最大可选共享内存大小属性
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // 检查CUDA函数调用是否成功
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // 如果未启用丢弃功能，则执行以下代码块
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 如果每个线程块允许的最大共享内存大小大于或等于特定值
        // 根据不同的条件选择合适的Flash反向传播的内核特征和丢弃模式
    });
}
    // 如果参数中的 dropout 概率小于 1，启用 dropout 功能
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 检查最大每块共享内存是否大于等于 176 KB，适用于 H100 GPU
        if (max_smem_per_block >= 176 * 1024) {  // H100
            // 运行使用 Flash_bwd_kernel_traits 参数的后向传播闪存操作，不启用 dropout
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream);
        } else if (max_smem_per_block >= 144 * 1024) {  // A100，为了节省共享内存不使用双缓冲
            // 运行使用 Flash_bwd_kernel_traits 参数的后向传播闪存操作，启用 dropout
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>, Is_dropout>(params, stream);
        } else { // sm86 和 sm89，最大共享内存为 99 KB，仅支持不使用 dropout 的情况，使用寄存器 V 并且不使用双缓冲
            if constexpr (!Is_dropout) {
                // 运行使用 Flash_bwd_kernel_traits 参数的后向传播闪存操作，不启用 dropout
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 32, 8, 4, 1, 2, true, true, T>, false>(params, stream);
            }
        }
    });
}

}; // 结束命名空间 pytorch_flash
```