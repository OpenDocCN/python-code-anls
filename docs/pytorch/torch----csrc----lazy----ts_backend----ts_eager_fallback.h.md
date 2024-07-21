# `.\pytorch\torch\csrc\lazy\ts_backend\ts_eager_fallback.h`

```py
#pragma once
// 使用预处理指令#pragma once，确保本头文件只被编译一次

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <functional>

namespace torch {
namespace lazy {

// 声明一个函数force_eager_fallback，用于判断是否需要强制使用eager模式回退
bool force_eager_fallback(c10::Symbol op);

// 声明一个函数ltc_eager_fallback，处理ltc操作符的eager模式回退
void ltc_eager_fallback(
    const c10::OperatorHandle& op,  // 接收一个操作符句柄
    torch::jit::Stack* stack);      // 接收一个指向torch::jit::Stack的指针

// 声明一个函数ts_eager_fallback，处理ts操作符的eager模式回退
void ts_eager_fallback(
    const c10::OperatorHandle& op,   // 接收一个操作符句柄
    torch::jit::Stack* stack,        // 接收一个指向torch::jit::Stack的指针
    c10::DeviceType device_type);    // 接收一个DeviceType参数，表示设备类型

// 声明一个函数register_ts_ltc_eager_fallback，用于注册ts和ltc的eager模式回退
// 这个函数应该只被主Torchscript后端初始化函数显式调用
void register_ts_ltc_eager_fallback();

} // namespace lazy
} // namespace torch
```