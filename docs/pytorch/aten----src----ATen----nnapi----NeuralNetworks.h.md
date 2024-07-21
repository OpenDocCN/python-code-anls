# `.\pytorch\aten\src\ATen\nnapi\NeuralNetworks.h`

```
/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*

Most of NeuralNetworks.h has been stripped for simplicity.
We don't need any of the function declarations since
we call them all through dlopen/dlsym.
Operation codes are pulled directly from serialized models.

*/

// 定义最小化神经网络头文件的宏，避免重复包含
#ifndef MINIMAL_NEURAL_NETWORKS_H
#define MINIMAL_NEURAL_NETWORKS_H

// 包含必要的标准整数类型头文件
#include <stdint.h>

// 定义操作结果代码的枚举
typedef enum {
    ANEURALNETWORKS_NO_ERROR = 0,                      // 操作成功
    ANEURALNETWORKS_OUT_OF_MEMORY = 1,                 // 内存不足
    ANEURALNETWORKS_INCOMPLETE = 2,                    // 操作不完整
    ANEURALNETWORKS_UNEXPECTED_NULL = 3,               // 遇到意外的空指针
    ANEURALNETWORKS_BAD_DATA = 4,                      // 数据错误
    ANEURALNETWORKS_OP_FAILED = 5,                     // 操作失败
    ANEURALNETWORKS_BAD_STATE = 6,                     // 状态异常
    ANEURALNETWORKS_UNMAPPABLE = 7,                    // 无法映射
    ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE = 8,      // 输出大小不足
    ANEURALNETWORKS_UNAVAILABLE_DEVICE = 9,            // 设备不可用
} ResultCode;

// 定义操作数类型代码的枚举
typedef enum {
    ANEURALNETWORKS_FLOAT32 = 0,                       // 32位浮点数
    ANEURALNETWORKS_INT32 = 1,                         // 32位整数
    ANEURALNETWORKS_UINT32 = 2,                        // 32位无符号整数
    ANEURALNETWORKS_TENSOR_FLOAT32 = 3,                // 32位浮点数张量
    ANEURALNETWORKS_TENSOR_INT32 = 4,                  // 32位整数张量
    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM = 5,           // 8位非对称量化张量
    ANEURALNETWORKS_BOOL = 6,                          // 布尔值
    ANEURALNETWORKS_TENSOR_QUANT16_SYMM = 7,           // 16位对称量化张量
    ANEURALNETWORKS_TENSOR_FLOAT16 = 8,                // 16位浮点数张量
    ANEURALNETWORKS_TENSOR_BOOL8 = 9,                  // 8位布尔值张量
    ANEURALNETWORKS_FLOAT16 = 10,                      // 16位浮点数
    ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL = 11,// 8位每通道对称量化张量
    ANEURALNETWORKS_TENSOR_QUANT16_ASYMM = 12,         // 16位非对称量化张量
    ANEURALNETWORKS_TENSOR_QUANT8_SYMM = 13,           // 8位对称量化张量
} OperandCode;

// 定义优先级代码的枚举
typedef enum {
    ANEURALNETWORKS_PREFER_LOW_POWER = 0,              // 低功耗优先
    ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,     // 快速单一答案优先
    ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2,        // 持续速度优先
} PreferenceCode;

// 定义神经网络内存、模型、设备、编译、执行和事件的结构体
typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;
typedef struct ANeuralNetworksModel ANeuralNetworksModel;
typedef struct ANeuralNetworksDevice ANeuralNetworksDevice;
typedef struct ANeuralNetworksCompilation ANeuralNetworksCompilation;
typedef struct ANeuralNetworksExecution ANeuralNetworksExecution;
typedef struct ANeuralNetworksEvent ANeuralNetworksEvent;

// 定义神经网络操作类型的整数类型
typedef int32_t ANeuralNetworksOperationType;

// 定义神经网络操作数类型的结构体
typedef struct ANeuralNetworksOperandType {
    int32_t type;               // 操作数类型
    uint32_t dimensionCount;    // 维度数量
    const uint32_t* dimensions; // 维度数组
    float scale;                // 缩放因子
    int32_t zeroPoint;          // 零点
} ANeuralNetworksOperandType;

#endif  // MINIMAL_NEURAL_NETWORKS_H
```