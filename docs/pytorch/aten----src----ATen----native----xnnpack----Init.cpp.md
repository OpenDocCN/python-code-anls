# `.\pytorch\aten\src\ATen\native\xnnpack\Init.cpp`

```
#ifdef USE_XNNPACK
// 如果定义了 USE_XNNPACK 宏，则编译以下代码块

#include <ATen/native/xnnpack/Common.h>
#include <c10/util/Exception.h>

namespace at::native::xnnpack {
namespace internal {
namespace {

// XNNPACK 是否已经初始化的标志
bool is_initialized_ = false;

// 初始化 XNNPACK
bool initialize() {
  using namespace internal;

  // 只有在未初始化时才执行初始化操作
  if (!is_initialized_) {
    // 调用 XNNPACK 的初始化函数
    const xnn_status status = xnn_initialize(nullptr);
    // 根据返回状态判断初始化是否成功
    is_initialized_ = (xnn_status_success == status);

    // 如果初始化失败，则根据不同的失败原因输出警告信息
    if (!is_initialized_) {
      if (xnn_status_out_of_memory == status) {
        TORCH_WARN_ONCE("Failed to initialize XNNPACK! Reason: Out of memory.");
      } else if (xnn_status_unsupported_hardware == status) {
        TORCH_WARN_ONCE("Failed to initialize XNNPACK! Reason: Unsupported hardware.");
      } else {
        TORCH_WARN_ONCE("Failed to initialize XNNPACK! Reason: Unknown error!");
      }
    }
  }

  // 返回初始化状态
  return is_initialized_;
}

// 反初始化 XNNPACK
bool C10_UNUSED deinitialize() {
  using namespace internal;

  // 只有在已初始化时才执行反初始化操作
  if (is_initialized_) {
    // 调用 XNNPACK 的反初始化函数
    const xnn_status status = xnn_deinitialize();
    // 根据返回状态判断反初始化是否成功
    is_initialized_ = !(xnn_status_success == status);

    // 如果反初始化失败，则输出警告信息
    if (is_initialized_) {
      TORCH_WARN_ONCE("Failed to uninitialize XNNPACK! Reason: Unknown error!");
    }
  }

  // 返回反初始化后的状态
  return !is_initialized_;
}

} // namespace
} // namespace internal

// 检查 XNNPACK 是否可用的函数
bool available() {
  // 调用内部的初始化函数检查 XNNPACK 是否可用
  return internal::initialize();
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
```