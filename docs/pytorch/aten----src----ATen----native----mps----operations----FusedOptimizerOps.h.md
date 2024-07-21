# `.\pytorch\aten\src\ATen\native\mps\operations\FusedOptimizerOps.h`

```py
#pragma once
// 预处理指令：确保本文件仅被编译一次

#include <ATen/native/mps/OperationUtils.h>
// 包含头文件：导入 ATen 库的操作工具

namespace at::native {
namespace mps {

static const char* FUSED_ADAM_OPS = R"METAL(
// 定义 Metal 着色器程序字符串常量 FUSED_ADAM_OPS

#include <metal_stdlib>
// 包含 Metal 标准库头文件

#define kmaxThreadGroups 32
// 定义整型常量 kmaxThreadGroups，表示最大线程组数
#define kmaxTensors 32
// 定义整型常量 kmaxTensors，表示最大张量数
#define chunk_size 65536
// 定义整型常量 chunk_size，表示分块大小

constexpr constant uint kParamIdx = 0;
// 常量表达式：参数索引，初始化为 0
constexpr constant uint kGradIdx = kParamIdx + kmaxTensors;
// 常量表达式：梯度索引，初始化为 参数索引 + 最大张量数
constexpr constant uint kExpAvgIdx = kGradIdx + kmaxTensors;
// 常量表达式：指数平均索引，初始化为 梯度索引 + 最大张量数
constexpr constant uint kMomentumBufferListIdx = kGradIdx + kmaxTensors;
// 常量表达式：动量缓冲列表索引，初始化为 梯度索引 + 最大张量数
constexpr constant uint kExpAvgSqIdx = kExpAvgIdx + kmaxTensors;
// 常量表达式：平方指数平均索引，初始化为 指数平均索引 + 最大张量数
constexpr constant uint kMaxExpAvgSqIdx = kExpAvgSqIdx + kmaxTensors;
// 常量表达式：最大平方指数平均索引，初始化为 平方指数平均索引 + 最大张量数
constexpr constant uint kStateStepsIdx = kExpAvgSqIdx + kmaxTensors;
// 常量表达式：状态步骤索引，初始化为 平方指数平均索引 + 最大张量数
constexpr constant uint kStateStepsIdxForAmsgrad = kMaxExpAvgSqIdx + kmaxTensors;
// 常量表达式：用于 Amsgrad 的状态步骤索引，初始化为 最大平方指数平均索引 + 最大张量数

template<typename T, typename state_steps_t>
struct AdamArguments {
    metal::array<device T *,  kmaxTensors>   params        [[ id(kParamIdx) ]];
    // Metal 结构体：Adam 算法参数，包含设备上 T 类型的参数指针数组
    metal::array<device T *,  kmaxTensors>   grads         [[ id(kGradIdx) ]];
    // Metal 结构体：Adam 算法参数，包含设备上 T 类型的梯度指针数组
    metal::array<device T *,  kmaxTensors>   exp_avgs      [[ id(kExpAvgIdx) ]];
    // Metal 结构体：Adam 算法参数，包含设备上 T 类型的指数平均指针数组
    metal::array<device T *,  kmaxTensors>   exp_avg_sqs   [[ id(kExpAvgSqIdx) ]];
    // Metal 结构体：Adam 算法参数，包含设备上 T 类型的平方指数平均指针数组
    metal::array<device state_steps_t *,  kmaxTensors>   state_steps   [[ id(kStateStepsIdx) ]];
    // Metal 结构体：Adam 算法参数，包含设备上 state_steps_t 类型的状态步骤指针数组
};

template<typename T, typename state_steps_t>
struct AdamAmsgradArguments {
    metal::array<device T *,  kmaxTensors>   params        [[ id(kParamIdx) ]];
    // Metal 结构体：Adam Amsgrad 算法参数，包含设备上 T 类型的参数指针数组
    metal::array<device T *,  kmaxTensors>   grads         [[ id(kGradIdx) ]];
    // Metal 结构体：Adam Amsgrad 算法参数，包含设备上 T 类型的梯度指针数组
    metal::array<device T *,  kmaxTensors>   exp_avgs      [[ id(kExpAvgIdx) ]];
    // Metal 结构体：Adam Amsgrad 算法参数，包含设备上 T 类型的指数平均指针数组
    metal::array<device T *,  kmaxTensors>   exp_avg_sqs   [[ id(kExpAvgSqIdx) ]];
    // Metal 结构体：Adam Amsgrad 算法参数，包含设备上 T 类型的平方指数平均指针数组
    metal::array<device T *,  kmaxTensors>   max_exp_avg_sqs   [[ id(kMaxExpAvgSqIdx) ]];
    // Metal 结构体：Adam Amsgrad 算法参数，包含设备上 T 类型的最大平方指数平均指针数组
    metal::array<device state_steps_t *,  kmaxTensors>   state_steps   [[ id(kStateStepsIdxForAmsgrad) ]];
    // Metal 结构体：Adam Amsgrad 算法参数，包含设备上 state_steps_t 类型的状态步骤指针数组
};

template<typename T>
struct SgdArguments {
    metal::array<device T *,  kmaxTensors>   params        [[ id(kParamIdx) ]];
    // Metal 结构体：SGD 算法参数，包含设备上 T 类型的参数指针数组
    metal::array<device T *,  kmaxTensors>   grads         [[ id(kGradIdx) ]];
    // Metal 结构体：SGD 算法参数，包含设备上 T 类型的梯度指针数组
};

template<typename T>
struct SgdMomentumArguments {
    metal::array<device T *,  kmaxTensors>   params        [[ id(kParamIdx) ]];
    // Metal 结构体：SGD Momentum 算法参数，包含设备上 T 类型的参数指针数组
    metal::array<device T *,  kmaxTensors>   grads         [[ id(kGradIdx) ]];
    // Metal 结构体：SGD Momentum 算法参数，包含设备上 T 类型的梯度指针数组
    metal::array<device T *,  kmaxTensors>   momentum_buffer_list       [[ id(kMomentumBufferListIdx) ]];
    // Metal 结构体：SGD Momentum 算法参数，包含设备上 T 类型的动量缓冲列表指针数组
};

struct MetadataArguments {
    uint32_t numels[kmaxTensors];
    // 元数据结构体：包含最大张量数目
    uint32_t threadgroup_to_tensor[kmaxThreadGroups];
    // 元数据结构体：包含最大线程组数目
    uint32_t threadgroup_to_chunk[kmaxThreadGroups];
    // 元数据结构体：包含最大线程组数目
};

enum ADAM_MODE : uint8_t {
  ORIGINAL = 0,
  ADAMW = 1
};
// 枚举类型 ADAM_MODE：Adam 优化器模式，包含原始和 AdamW

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
inline void adam_math_amsgrad(
    device T & param,
    // 函数：Adam Amsgrad 数学计算
    device T & grad,
    device T & exp_avg,
    device T & exp_avg_sq,
    device T & max_exp_avg_sq,
    device state_steps_t & state_steps,
    // 参数：参数、梯度、指数平均、平方指数平均、最大平方指数平均、状态步骤
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    // 参数：学习率、beta1、beta2、权重衰减、eps
    # 定义一个名为 maximize 的常量，其类型为 uint8_t
    const uint8_t maximize
template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
kernel void fused_adam_amsgrad(
    // 定义函数参数列表
    device AdamAmsgradArguments<T, state_steps_t> & args    [[buffer(0)]],
    // 元数据参数
    constant MetadataArguments & metadata_args [[buffer(1)]],
    // 学习率
    constant float & lr             [[buffer(2)]],
    // Adam优化器的beta1参数
    constant float & beta1          [[buffer(3)]],
    // Adam优化器的beta2参数
    constant float & beta2          [[buffer(4)]],
    // 权重衰减
    constant float & weight_decay   [[buffer(5)]],
    // epsilon值，用于数值稳定性
    constant float & eps            [[buffer(6)]],
    // 是否最大化目标函数的标志
    constant uint8_t   & maximize       [[buffer(7)]],
    // 线程在线程组中的位置
    uint tid [[thread_position_in_threadgroup]],
    // 线程组在网格中的位置
    uint tgid [[threadgroup_position_in_grid]],
) {
  // 保存原始梯度
  T grad_ = grad;

  // 如果需要最大化目标函数，将梯度取反
  if (maximize) {
    grad = -grad;
  }

  // 更新参数、梯度、一阶和二阶动量
  if (weight_decay != 0) {
    switch (adam_mode) {
      case ADAM_MODE::ORIGINAL:
        // 根据不同的adam_mode更新梯度，加入权重衰减项
        grad += param * weight_decay;
        break;
      case ADAM_MODE::ADAMW:
        // 根据不同的adam_mode更新参数，加入权重衰减项
        param -= lr * weight_decay * param;
        break;
    }
  }

  // 更新一阶动量
  exp_avg = beta1 * exp_avg + (1 - beta1) * grad;
  // 更新二阶动量
  exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad;

  // 计算一阶动量的偏差修正项
  const float casted_state_steps = static_cast<float>(state_steps);
  const T bias_correction1 = 1 - metal::precise::pow(beta1, casted_state_steps);
  const T step_size = lr / bias_correction1;

  // 计算二阶动量的偏差修正项
  const T bias_correction2 = 1 - metal::precise::pow(beta2, casted_state_steps);
  const T bias_correction2_sqrt = metal::precise::sqrt(bias_correction2);

  // 更新最大的二阶动量估计
  max_exp_avg_sq = metal::max(max_exp_avg_sq, exp_avg_sq);

  // 计算分母项
  const T denom = (metal::precise::sqrt(max_exp_avg_sq) / bias_correction2_sqrt) + eps;

  // 更新参数
  param -= step_size * exp_avg / denom;

  // 恢复原始梯度
  grad = grad_;
}
    // 定义每个线程组内的线程数目
    uint tptg [[threads_per_threadgroup]]) {

    // 获取当前线程组在张量中的位置索引
    const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
    // 获取当前线程组所属的数据块索引
    const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
    // 计算当前数据块在整个张量中的偏移量
    const uint32_t chunk_offset = chunk_idx * chunk_size;
    // 计算当前数据块内的元素个数
    const uint32_t numel = metadata_args.numels[tensor_loc] - chunk_offset;

    // 获取当前张量在优化器状态中的步数
    const auto step_count = args.state_steps[tensor_loc];

    // 指向当前数据块在各种参数数组中的起始位置
    auto param = args.params[tensor_loc] + chunk_offset;
    auto grad = args.grads[tensor_loc] + chunk_offset;
    auto exp_avg = args.exp_avgs[tensor_loc] + chunk_offset;
    auto exp_avg_sq = args.exp_avg_sqs[tensor_loc] + chunk_offset;
    auto max_exp_avg_sq = args.max_exp_avg_sqs[tensor_loc] + chunk_offset;

    // 循环处理当前数据块内的元素
    for (uint32_t i_start = tid; i_start < numel && i_start < chunk_size; i_start += tptg) {
      // 调用 Adam 优化算法中的更新函数
      adam_math_amsgrad<T, state_steps_t, adam_mode>(
        *(param + i_start),       // 当前参数
        *(grad + i_start),        // 当前梯度
        *(exp_avg + i_start),     // 当前一阶矩估计
        *(exp_avg_sq + i_start),  // 当前二阶矩估计
        *(max_exp_avg_sq + i_start),  // 当前二阶矩的最大估计
        *step_count,              // 当前步数
        lr,                       // 学习率
        beta1,                    // 一阶矩的指数衰减率
        beta2,                    // 二阶矩的指数衰减率
        weight_decay,             // 权重衰减（L2 正则化因子）
        eps,                      // 防止除零的小常数
        maximize                  // 是否最大化目标函数（布尔值）
      );
    }
// 定义了一个 CUDA Kernel 函数 `fused_adam`，用于执行融合的 Adam 优化算法
kernel void fused_adam(
    // GPU 设备内存中的参数结构体，包含模板类型 T 的参数
    device AdamArguments<T, state_steps_t> & args    [[buffer(0)]],
    // 不可变元数据参数结构体，包含了线程组相关信息
    constant MetadataArguments & metadata_args [[buffer(1)]],
    // 学习率，以常量方式传递给 Kernel
    constant float & lr             [[buffer(2)]],
    // Adam 优化算法的超参数 beta1，以常量方式传递给 Kernel
    constant float & beta1          [[buffer(3)]],
    // Adam 优化算法的超参数 beta2，以常量方式传递给 Kernel
    constant float & beta2          [[buffer(4)]],
    // 权重衰减（weight decay）的参数，以常量方式传递给 Kernel
    constant float & weight_decay   [[buffer(5)]],
    // Adam 优化算法的 epsilon 参数，以常量方式传递给 Kernel
    constant float & eps            [[buffer(6)]],
    // 是否最大化目标值的标志，以常量方式传递给 Kernel
    constant uint8_t   & maximize       [[buffer(7)]],
    // 线程在线程组内的位置索引
    uint tid [[thread_position_in_threadgroup]],
    // 线程组在整个网格中的位置索引
    uint tgid [[threadgroup_position_in_grid]],
    // 每个线程组内线程的总数
    uint tptg [[threads_per_threadgroup]])
{
    // 获取当前线程组对应的张量位置
    const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
    // 获取当前线程组对应的块索引
    const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
    // 计算当前线程组对应的块偏移量
    const uint32_t chunk_offset = chunk_idx * chunk_size;
    // 获取当前张量的元素数量，减去块偏移量
    const uint32_t numel = metadata_args.numels[tensor_loc] - chunk_offset;

    // 获取当前线程组处理的参数、梯度、指数加权平均值和平方梯度平均值
    auto param = args.params[tensor_loc] + chunk_offset;
    auto grad = args.grads[tensor_loc] + chunk_offset;
    auto exp_avg = args.exp_avgs[tensor_loc] + chunk_offset;
    auto exp_avg_sq = args.exp_avg_sqs[tensor_loc] + chunk_offset;

    // 遍历当前线程组处理的元素范围
    for (uint32_t i_start = tid; i_start < numel && i_start < chunk_size; i_start += tptg) {
        // 调用特定模式下的 Adam 数学函数，执行参数更新
        adam_math<T, state_steps_t, adam_mode>(
            *(param + i_start),
            *(grad + i_start),
            *(exp_avg + i_start),
            *(exp_avg_sq + i_start),
            *step_count,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize
        );
    }
}

// 定义注册宏 `REGISTER_FUSED_ADAM_OP`，用于注册不同数据类型和模式的融合 Adam 优化算法的 CUDA Kernel
#define REGISTER_FUSED_ADAM_OP(DTYPE, STATE_STEPS_DTYPE, ADAM_MODE_DTYPE, HOST_NAME, KERNEL_NAME, ARGUMENTS_STRUCT)       \
template                                                                                                                \
[[host_name(#HOST_NAME "_" #DTYPE "_" #STATE_STEPS_DTYPE)]]                                                              \
kernel void KERNEL_NAME<DTYPE, STATE_STEPS_DTYPE, ADAM_MODE_DTYPE>(                                                      \
    // GPU 设备内存中的参数结构体，包含模板类型 DTYPE 和 STATE_STEPS_DTYPE 的参数
    device ARGUMENTS_STRUCT<DTYPE, STATE_STEPS_DTYPE> & args    [[buffer(0)]],                                            \
    // 不可变元数据参数结构体，包含了线程组相关信息
    constant MetadataArguments & metadata_args [[buffer(1)]],                                                             \
    // 学习率，以常量方式传递给 Kernel
    constant float & lr             [[buffer(2)]],                                                                         \
    // Adam 优化算法的超参数 beta1，以常量方式传递给 Kernel
    constant float & beta1          [[buffer(3)]],                                                                         \
    // Adam 优化算法的超参数 beta2，以常量方式传递给 Kernel
    constant float & beta2          [[buffer(4)]],                                                                         \
    // 权重衰减（weight decay）的参数，以常量方式传递给 Kernel
    constant float & weight_decay   [[buffer(5)]],                                                                         \
    // Adam 优化算法的 epsilon 参数，以常量方式传递给 Kernel
    constant float & eps            [[buffer(6)]],                                                                         \
    // 是否最大化目标值的标志，以常量方式传递给 Kernel
    constant uint8_t   & maximize       [[buffer(7)]],                                                                    \
    // 线程在线程组内的位置索引
    uint tid [[thread_position_in_threadgroup]],                                                                           \
    // 线程组在整个网格中的位置索引
    uint tgid [[threadgroup_position_in_grid]],                                                                            \
    // 每个线程组内线程的总数
    uint tptg [[threads_per_threadgroup]])                                                                                 \
// 注册基于不同数据类型和优化模式的融合SGD（随机梯度下降）操作
REGISTER_FUSED_ADAM_OP(float, float, ADAM_MODE::ADAMW, fused_adamw, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(float, half, ADAM_MODE::ADAMW, fused_adamw, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(half, float, ADAM_MODE::ADAMW, fused_adamw, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(half, half, ADAM_MODE::ADAMW, fused_adamw, fused_adam, AdamArguments);
REGISTER_FUSED_ADAM_OP(float, float, ADAM_MODE::ORIGINAL, fused_adam_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(float, half, ADAM_MODE::ORIGINAL, fused_adam_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(half, float, ADAM_MODE::ORIGINAL, fused_adam_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(half, half, ADAM_MODE::ORIGINAL, fused_adam_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(float, float, ADAM_MODE::ADAMW, fused_adamw_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(float, half, ADAM_MODE::ADAMW, fused_adamw_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(half, float, ADAM_MODE::ADAMW, fused_adamw_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);
REGISTER_FUSED_ADAM_OP(half, half, ADAM_MODE::ADAMW, fused_adamw_amsgrad, fused_adam_amsgrad, AdamAmsgradArguments);

// 定义SGD动量法数学操作，其中T为参数和梯度的数据类型
template <typename T>
inline void sgd_momentum_math(
    device T & param,                    // 参数
    device T & grad,                     // 梯度
    device T & momentum_buffer,          // 动量缓存
    const float weight_decay,            // 权重衰减
    const float momentum,                // 动量参数
    const float lr,                      // 学习率
    const float dampening,               // 阻尼
    const uint8_t nesterov,              // 是否使用Nesterov动量
    const uint8_t maximize,              // 是否最大化优化
    const uint8_t is_first_step          // 是否第一步
) {
  auto grad_ = grad;
  if (maximize) {
      grad_ *= -1.0;                    // 如果最大化优化，将梯度取反
  }
  if (weight_decay != 0) {
      grad_ += weight_decay * param;    // 加上权重衰减对参数的影响
  }

  momentum_buffer = is_first_step ? grad_ : (momentum * momentum_buffer + (1 - dampening) * grad_);
  // 更新动量缓存，考虑是否是第一步

  if (nesterov) {
      grad_ += momentum * momentum_buffer;  // 如果使用Nesterov动量，调整梯度
  } else {
      grad_ = momentum_buffer;              // 否则直接使用动量缓存
  }

  param -= lr * grad_;  // 更新参数
}

// 定义普通SGD优化数学操作，其中T为参数和梯度的数据类型
template <typename T>
inline void sgd_math(
    device T & param,                    // 参数
    device T & grad,                     // 梯度
    const float weight_decay,            // 权重衰减
    const float momentum,                // 动量参数（无效，未使用）
    const float lr,                      // 学习率
    const float dampening,               // 阻尼（无效，未使用）
    const uint8_t nesterov,              // 是否使用Nesterov动量（无效，未使用）
    const uint8_t maximize,              // 是否最大化优化
    const uint8_t is_first_step          // 是否第一步
) {
  auto grad_ = grad;
  if (maximize) {
      grad_ *= -1.0;                    // 如果最大化优化，将梯度取反
  }
  if (weight_decay != 0) {
      grad_ += weight_decay * param;    // 加上权重衰减对参数的影响
  }

  param -= lr * grad_;  // 更新参数
}

// 定义融合SGD的核函数，其中T为参数和梯度的数据类型
template <typename T>
kernel void fused_sgd(
    device   SgdArguments<T> & args    [[buffer(0)]],     // SGD参数
    constant MetadataArguments & metadata_args [[buffer(1)]],   // 元数据参数
    constant float & weight_decay   [[buffer(2)]],     // 权重衰减
    constant float & momentum       [[buffer(3)]],     // 动量参数
    constant float & lr             [[buffer(4)]],     // 学习率
    constant float & dampening      [[buffer(5)]],     // 阻尼
    constant uint8_t & nesterov     [[buffer(6)]],     // 是否使用Nesterov动量
    constant uint8_t   & maximize   [[buffer(7)]]      // 是否最大化优化
) {
    // 声明一个常量引用，指向每个线程组中第一个步骤的标志，存储在缓冲区中的位置
    constant uint8_t & is_first_step [[buffer(8)]],
    // 线程在线程组中的索引
    uint tid [[thread_position_in_threadgroup]],
    // 线程组在整个网格中的位置
    uint tgid [[threadgroup_position_in_grid]],
    // 每个线程组中的线程总数
    uint tptg [[threads_per_threadgroup]]) {

    // 获取当前线程组在张量中的位置索引
    const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
    // 获取当前线程组在块中的索引
    const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
    // 计算当前线程组在块中的偏移量
    const uint32_t chunk_offset = chunk_idx * chunk_size;
    // 计算当前张量的元素数量，减去块偏移量得到当前线程组的有效元素数量
    const uint32_t numel = metadata_args.numels[tensor_loc] - chunk_offset;

    // 参数和梯度指针分别指向当前线程组处理的张量参数和梯度的起始位置
    auto param = args.params[tensor_loc] + chunk_offset;
    auto grad = args.grads[tensor_loc] + chunk_offset;

    // 循环处理当前线程组中的每个线程所负责的元素范围
    for (uint32_t i_start = tid; i_start < numel && i_start < chunk_size; i_start += tptg) {
      // 调用 SGD 数学函数，对当前元素的参数和梯度进行更新
      sgd_math<T>(
        *(param + i_start),    // 当前参数的地址
        *(grad + i_start),     // 当前梯度的地址
        weight_decay,          // 权重衰减
        momentum,              // 动量
        lr,                    // 学习率
        dampening,             // 阻尼
        nesterov,              // Nesterov 动量
        maximize,              // 最大化标志
        is_first_step          // 是否第一步
      );
    }
}


这段代码是一个使用Metal语言编写的GPU并行计算内核函数。它处理张量参数和梯度的更新，利用线程组和线程索引来并行处理数据。
// 定义模板函数，用于执行融合的 SGD（随机梯度下降）动量优化算法
template <typename T>
kernel void fused_sgd_momentum(
    device   SgdMomentumArguments<T> & args    [[buffer(0)]], // 从缓冲区读取 SGD 动量优化参数
    constant MetadataArguments & metadata_args [[buffer(1)]], // 从缓冲区读取元数据参数
    constant float & weight_decay   [[buffer(2)]], // 从缓冲区读取权重衰减参数
    constant float & momentum       [[buffer(3)]], // 从缓冲区读取动量参数
    constant float & lr             [[buffer(4)]], // 从缓冲区读取学习率参数
    constant float & dampening          [[buffer(5)]], // 从缓冲区读取阻尼参数
    constant uint8_t & nesterov          [[buffer(6)]], // 从缓冲区读取 Nesterov 加速参数
    constant uint8_t   & maximize       [[buffer(7)]], // 从缓冲区读取最大化标志位参数
    constant uint8_t   & is_first_step       [[buffer(8)]], // 从缓冲区读取是否第一步参数
    uint tid [[thread_position_in_threadgroup]], // 线程在线程组中的位置索引
    uint tgid [[threadgroup_position_in_grid]], // 线程组在网格中的位置索引
    uint tptg [[threads_per_threadgroup]]) // 每个线程组中的线程数

{
    // 从元数据参数中获取张量在线程组中的位置
    const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
    // 从元数据参数中获取块的索引
    const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
    // 计算块的偏移量
    const uint32_t chunk_offset = chunk_idx * chunk_size;
    // 计算剩余元素数量
    const uint32_t numel = metadata_args.numels[tensor_loc] - chunk_offset;

    // 从参数结构中获取当前处理的参数、梯度和动量缓冲列表
    auto param = args.params[tensor_loc] + chunk_offset;
    auto grad = args.grads[tensor_loc] + chunk_offset;
    auto momentum_buffer_list = args.momentum_buffer_list[tensor_loc] + chunk_offset;

    // 遍历当前线程处理的所有元素
    for (uint32_t i_start = tid; i_start < numel && i_start < chunk_size; i_start += tptg) {
        // 执行 SGD 动量优化算法的数学运算
        sgd_momentum_math<T>(
            *(param + i_start),
            *(grad + i_start),
            *(momentum_buffer_list + i_start),
            weight_decay,
            momentum,
            lr,
            dampening,
            nesterov,
            maximize,
            is_first_step
        );
    }
}

// 定义宏，用于注册不同数据类型的融合 SGD 操作
#define REGISTER_FUSED_SGD_OP(DTYPE, HOST_NAME, KERNEL_NAME, ARGUMENTS_STRUCT)       \
template                                    \
[[host_name(#HOST_NAME "_" #DTYPE)]]        \
kernel void KERNEL_NAME<DTYPE>(             \
    device   ARGUMENTS_STRUCT<DTYPE> & args    [[buffer(0)]],\
    constant MetadataArguments & metadata_args [[buffer(1)]],\
    constant float & weight_decay   [[buffer(2)]],\
    constant float & momentum       [[buffer(3)]],\
    constant float & lr             [[buffer(4)]],\
    constant float & dampening          [[buffer(5)]],\
    constant uint8_t & nesterov          [[buffer(6)]],\
    constant uint8_t   & maximize       [[buffer(7)]],\
    constant uint8_t   & is_first_step       [[buffer(8)]],\
    uint tid [[thread_position_in_threadgroup]],\
    uint tgid [[threadgroup_position_in_grid]],\
    uint tptg [[threads_per_threadgroup]])

// 注册不同数据类型的融合 SGD 操作模板实例化
REGISTER_FUSED_SGD_OP(float, fused_sgd, fused_sgd, SgdArguments);
REGISTER_FUSED_SGD_OP(half, fused_sgd, fused_sgd, SgdArguments);
REGISTER_FUSED_SGD_OP(float, fused_sgd_momentum, fused_sgd_momentum, SgdMomentumArguments);
REGISTER_FUSED_SGD_OP(half, fused_sgd_momentum, fused_sgd_momentum, SgdMomentumArguments);
// 定义静态函数 getCPLState，返回一个包含 Metal 计算管线状态和 Metal 函数的 pair
static std::pair<id<MTLComputePipelineState>, id<MTLFunction>> getCPLState(const std::string& fname) {
    // 定义静态变量 MetalShaderLibrary，初始化为 FUSED_ADAM_OPS 类型的库
    static MetalShaderLibrary lib(FUSED_ADAM_OPS, 0);
    // 调用 MetalShaderLibrary 对象的 getPipelineStateForFunc 方法，获取指定函数名对应的计算管线状态
    // 调用 MetalShaderLibrary 对象的 getMTLFunction 方法，获取指定函数名对应的 Metal 函数
    return std::make_pair(lib.getPipelineStateForFunc(fname), lib.getMTLFunction(fname));
}

// 结束 mps 命名空间
} // namespace mps

// 结束 at::native 命名空间
} // namespace at::native
```