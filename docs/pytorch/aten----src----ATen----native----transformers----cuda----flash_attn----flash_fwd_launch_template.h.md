# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\flash_fwd_launch_template.h`

```py
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContextLight.h>

#include <ATen/native/transformers/cuda/flash_attn/flash.h>
#include <ATen/native/transformers/cuda/flash_attn/static_switch.h>
#include <ATen/native/transformers/cuda/flash_attn/flash_fwd_kernel.h>

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
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

// Definition for the flash_fwd_kernel function template
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        // Enforce constraints for Is_causal and Is_local
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        // Call compute_attn function with specified template parameters
        pytorch_flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Return_softmax>(params);
    #else
        // Output error message if FLASH is not supported on the current architecture
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// Definition for the flash_fwd_splitkv_kernel function template
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Split, bool Append_KV) {
    #if defined(ARCH_SUPPORTS_FLASH)
        // Call compute_attn_splitkv function with specified template parameters
        pytorch_flash::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Split, Append_KV>(params);
    #else
        // Output error message if FLASH is not supported on the current architecture
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// Definition for the flash_fwd_splitkv_combine_kernel function template
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits, bool Is_even_K) {
    static_assert(Log_max_splits >= 1); // Ensure Log_max_splits is at least 1
    // Call combine_attn_seqk_parallel function with specified template parameters
    pytorch_flash::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

// Function to execute flash_fwd with specified parameters and CUDA stream
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize; // Define smem_size constant from Kernel_traits
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21
    // 计算需要的块数，保证覆盖所有序列长度
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    // 定义 CUDA 网格维度，用于并行处理
    dim3 grid(num_m_block, params.b, params.h);
    // 检查是否满足一些特定条件，决定是否为偶数的标志
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    // 检查是否满足特定的头维度条件
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    // 检查是否需要返回 softmax 结果
    const bool return_softmax = params.p_ptr != nullptr;
    // 根据 is_even_MN 的值选择不同的代码路径
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        // 根据 is_even_K 的值选择不同的代码路径
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            // 根据局部性和因果性选择不同的代码路径
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                // 根据返回 softmax 的需求选择不同的代码路径
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    // 根据是否有 alibi 斜率选择不同的代码路径
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        // 如果满足特定条件，选择优化模板
                        // 设置 kernel 指针以选择特定的模板化函数
                        auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, ReturnSoftmaxConst && Is_dropout>;
                        // 检查共享内存大小是否足够大，以设置 CUDA 函数属性
                        if (smem_size >= 48 * 1024) {
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                        }
                        // 调用 CUDA kernel 函数进行计算，使用指定的网格和线程块尺寸
                        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                        // 检查 CUDA kernel 启动是否成功
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });
    }
    if (params.num_splits > 1) {
        // 如果要进行数据分割，则选择合适的 kBlockM 值以增加并行性。
        // 当使用 128 个线程时，我们可以一次加载 512 个元素，因此如果 headdim 能被 128 整除，则 kBlockM = 4。
        // 如果 headdim 能被 64 整除，则设置 kBlockM = 8，依此类推。
        // 根据 headdim 的不同情况选择不同的 kBlockM 值，以优化并行处理。
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        
        // 计算并设置 CUDA 线程块的维度，确保正确分配并行任务。
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        
        // 根据 num_splits 的大小选择合适的分割策略和相关的 CUDA 核函数进行调用。
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            }
            // 检查 CUDA 核函数启动是否成功
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
}

template<typename T, int Headdim>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    // 定义块大小常量 kBlockM 为 64，适用于所有头维度
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    // TD [2023-08-28]: 对于头维度为 96 和块大小为 64 x 256，nvcc 发生段错误
    // 对于头维度为 192 和块大小为 64 x 128，也会发生段错误
    // 还有对于头维度为 160，经旋转添加后，块大小为 64 x 128 也会发生段错误
    constexpr static int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64);
    // 运行 Flash 前向分离 KV 的前向传播
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>>(params, stream);
}

template<typename T>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // 运行 Flash 前向传播，头维度为 32，块大小为 128 x 128
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if constexpr(!Is_dropout) {
                // 使用 8 个线程束，在 seqlen=2k 时慢 18%
                // 使用块大小 (64 x 256)，在 seqlen=2k 时慢 27%
                // 使用块大小 (256 x 64)，因寄存器溢出慢 85%，适合于 seqlen=2k
                // 运行 Flash 前向传播，头维度为 64，块大小为 128 x 128
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                // 运行 Flash 前向传播，头维度为 64，块大小为 128 x 64
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        });
    });
}

template<typename T>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    // 在头维度为 96 时，获取当前 CUDA 设备的属性
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 如果 dropout 概率小于 1，启用 dropout
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // 根据是否是因果模式选择不同的操作
            // 对于 sm86 或 sm89，因果模式下 64 x 64 是最快的（因为它是正方形）
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    // 如果不是因果模式，使用 Flash_fwd_kernel_traits 配置为 Headdim, 128, 64, 4 的前向运行
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    // 如果是因果模式，使用 Flash_fwd_kernel_traits 配置为 Headdim, 64, 64, 4 的前向运行
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                // 对于非 sm86 或 sm89 的情况，使用 Flash_fwd_kernel_traits 配置为 Headdim, 128, 64, 4 的前向运行
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // 下面两行代码总是较慢，因此被注释掉
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // 这两行代码总是较慢，因此被注释掉
            // run_flash_fwd<Flash_fwd_kernel_traits<96, 128, 128, 4, true, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<96, 64, 128, 4, true, T>>(params, stream);
        });
    });
}

// 定义模板函数，用于执行头维度为128的多头注意力前向传播
template<typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    // 定义常量头维度为128
    constexpr static int Headdim = 128;
    // 获取当前 CUDA 设备的属性
    auto dprops = at::cuda::getCurrentDeviceProperties();
    // 检查当前设备是否为sm8x架构
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    // 开启丢弃率的条件分支，若params.p_dropout小于1，则进入
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 开启是否因果的条件分支，若params.is_causal为真，则进入
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // 若不使用丢弃率
            if constexpr(!Is_dropout) {
                // 对于sm86或sm89架构，对于非因果情况使用128 x 32最快（因为每个SM可以有2个CTA）
                if (is_sm8x) {
                    if constexpr(!Is_causal) {
                        // 运行多头注意力前向传播，使用128头，每头64个查询、64个键、32个值，4个线程块，不使用丢弃和因果
                        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                    } else {
                        // 运行多头注意力前向传播，使用128头，每头64个查询、64个键、64个值，4个线程块，不使用丢弃但使用因果
                        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                    }
                } else {
                    // 运行多头注意力前向传播，使用128头，每头64个查询、64个键、64个值，4个线程块，不使用丢弃和因果
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
                // 其他潜在的多头注意力前向传播选项，这些选项并未在当前代码中使用
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // 使用8个线程束（128 x 128和256 x 64）对于长度为2k的序列比较慢，因此未被选择
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // 对于H100和A100，第一组选项较好；对于A6000，第二组选项的占用率稍微好一些
            } else {
                // 使用丢弃率运行多头注意力前向传播，使用128头，每头64个查询、64个键、32个值，4个线程块，不使用因果
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            }
        });
    });
}

// 定义模板函数，用于执行头维度为160的多头注意力前向传播
template<typename T>
void run_mha_fwd_hdim160(Flash_fwd_params &params, cudaStream_t stream) {
    // 定义常量头维度为160
    constexpr static int Headdim = 160;
    // 获取当前 CUDA 设备的属性
    auto dprops = at::cuda::getCurrentDeviceProperties();
    // 检查当前设备是否为sm8x架构
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // 如果是 A100 或者 H100，128 x 32 是最快的配置。
            // 对于 sm86 或者 sm89，因为它是正方形，64 x 64 对因果模型是最快的配置，
            // 而对于非因果模型，128 x 64 和 8 个 warps 是最快的配置。
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    // 运行带有指定参数的前向闪存（Flash_fwd）内核
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    // 运行带有指定参数的前向闪存（Flash_fwd）内核
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                // 运行带有指定参数的前向闪存（Flash_fwd）内核
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // 下面是一些备选配置的运行示例，根据需要取消注释来使用不同的内核配置。
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    // 定义头维度为192
    constexpr static int Headdim = 192;
    // 如果概率小于1，启用dropout
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 如果是因果的
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // 如果不使用dropout
            if constexpr(!Is_dropout) {
                // 运行指定参数的前向传播Flash操作
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                // 使用dropout，运行指定参数的前向传播Flash操作
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // 注释掉的代码，这些是备选的前向传播Flash操作，根据需要选择不同的参数进行运行
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim224(Flash_fwd_params &params, cudaStream_t stream) {
    // 定义头维度为224
    constexpr static int Headdim = 224;
    int device;
    // 获取当前设备
    cudaGetDevice(&device);
    int max_smem_per_block;
    // 获取当前设备的最大共享内存限制
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // 如果获取共享内存失败，则抛出CUDA错误
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    // 如果概率小于1，启用dropout
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // 如果是因果的
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // 如果当前设备支持的共享内存大于等于计算需要的共享内存大小（112 KB）
            if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64)) {  // 112 KB
                // 运行指定参数的前向传播Flash操作
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                // 否则，使用较小的配置运行前向传播Flash操作
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // 注释掉的代码，这些是备选的前向传播Flash操作，根据需要选择不同的参数进行运行
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // 无法使用128 x 32配置和8个warp，因为对于headdim 224，kBlockKSmem = 32。
            // 如果N = 32，每次加载的元素仅为1024个，每次加载8个元素。
            // 这意味着我们只能使用128个线程而不是256个线程。
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    // 定义头维度为256
    constexpr static int Headdim = 256;
    // 定义变量用于存储 CUDA 设备 ID
    int device;
    // 获取当前 CUDA 设备 ID
    cudaGetDevice(&device);
    // 定义变量用于存储每个多处理器的最大共享内存和每个块的最大共享内存
    int max_smem_per_sm, max_smem_per_block;
    // 获取当前设备的属性：每个多处理器的最大共享内存
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    // 获取当前设备的属性：每个块的最大共享内存（支持选择性最大共享内存）
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // 检查 CUDA 函数调用是否成功
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // 根据条件打印每个多处理器和每个块的最大共享内存（注释掉的代码）
    // printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm, max_smem_per_block);
    // 根据参数条件进行丢弃操作和因果操作的选择
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // 根据硬件类型和内存条件选择不同的内核启动配置
            // 对于 A100，选择 128 x 64 (128KB 共享内存)
            // 对于 H100，选择 64 x 64 (96KB 共享内存)，以便每个多处理器可以获得2个线程块
            if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // 注释掉的代码：选择 64KB 共享内存的内核启动配置
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // 注释掉的代码：选择 96KB 共享内存的内核启动配置
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

}; // namespace pytorch_flash
```