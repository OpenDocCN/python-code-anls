# `.\pytorch\torch\csrc\jit\passes\mobile_optimizer_type.h`

```py
#pragma once


// 使用 #pragma once 指令确保头文件只被编译一次，防止多重包含问题


#include <cstdint>


// 包含 <cstdint> 头文件，提供了固定大小的整数类型，如 int8_t


enum class MobileOptimizerType : int8_t {
  CONV_BN_FUSION,
  INSERT_FOLD_PREPACK_OPS,
  REMOVE_DROPOUT,
  FUSE_ADD_RELU,
  HOIST_CONV_PACKED_PARAMS,
  CONV_1D_TO_2D,
  VULKAN_AUTOMATIC_GPU_TRANSFER,
};


// 定义一个枚举类型 MobileOptimizerType，基础类型为 int8_t（8 位有符号整数）
// 枚举类型包含以下成员：
//   - CONV_BN_FUSION：卷积批归一化融合优化
//   - INSERT_FOLD_PREPACK_OPS：插入折叠预打包操作优化
//   - REMOVE_DROPOUT：移除 dropout 优化
//   - FUSE_ADD_RELU：融合加法和ReLU操作优化
//   - HOIST_CONV_PACKED_PARAMS：提升卷积打包参数优化
//   - CONV_1D_TO_2D：一维卷积转换为二维卷积优化
//   - VULKAN_AUTOMATIC_GPU_TRANSFER：Vulkan自动GPU传输优化
// 每个成员都对应一个整数值，用于标识不同的优化类型
```