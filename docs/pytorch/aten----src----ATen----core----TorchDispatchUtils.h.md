# `.\pytorch\aten\src\ATen\core\TorchDispatchUtils.h`

```
// 预处理指令，表示此文件在编译时只包含一次
#pragma once

// 包含 Torch 库的头文件
#include <torch/library.h>

// 包含 ATen 库的调度器头文件
#include <ATen/core/dispatch/Dispatcher.h>

// 包含 C10 库的 ArrayRef 工具类头文件
#include <c10/util/ArrayRef.h>

// 包含 C10 库的 Optional 类头文件
#include <c10/util/Optional.h>

// 包含 C10 库的 TorchDispatchModeTLS 实现头文件
#include <c10/core/impl/TorchDispatchModeTLS.h>

// 命名空间定义：at::impl
namespace at::impl {

    // 定义了一个 Torch API 函数，用于检查张量是否具有调度功能
    TORCH_API bool tensor_has_dispatch(const at::Tensor& t);

    // 定义了一个 Torch API 函数，用于检查张量列表是否具有调度功能
    TORCH_API bool tensorlist_has_dispatch(at::ITensorListRef li);

    // 定义了一个 Torch API 函数，用于检查张量列表是否具有调度功能（处理了 std::optional 包装）
    TORCH_API bool tensorlist_has_dispatch(const c10::List<std::optional<at::Tensor>>& li);

    // 使用命名空间 c10::impl 的 dispatch_mode_enabled 函数
    using c10::impl::dispatch_mode_enabled;

} // namespace at::impl
```