# `.\pytorch\aten\src\ATen\cudnn\cudnn-wrapper.h`

```
#pragma once
#pragma once 指令确保头文件只被包含一次，防止重复定义

#include <cudnn.h>
#include <cudnn.h> 包含 CUDA 深度神经网络库（CuDNN）的头文件

#define STRINGIFY(x) #x
#define STRING(x) STRINGIFY(x)
宏定义 STRINGIFY 将参数 x 转换成字符串字面量
宏定义 STRING 则调用 STRINGIFY 将 x 转换为字符串

#if CUDNN_MAJOR < 6
#if 指令，用于条件编译，判断 CuDNN 主版本号是否小于 6

#pragma message ("CuDNN v" STRING(CUDNN_MAJOR) " found, but need at least CuDNN v6. You can get the latest version of CuDNN from https://developer.nvidia.com/cudnn or disable CuDNN with USE_CUDNN=0")
#pragma message 指令，输出一条编译时的信息消息
输出一条提示消息，说明当前找到的 CuDNN 版本号，但需要至少 CuDNN v6。提供了获取最新版本 CuDNN 的链接，或者建议通过 USE_CUDNN=0 禁用 CuDNN

#pragma message "We strongly encourage you to move to 6.0 and above."
输出一条提示消息，鼓励用户升级到 CuDNN v6.0 及以上

#pragma message "This message is intended to annoy you enough to update."
输出一条提示消息，表达出为促使用户更新而特意提醒的意图

#endif
结束条件编译块

#undef STRINGIFY
#undef STRING
取消前面定义的宏 STRINGIFY 和 STRING，清理预处理定义
```