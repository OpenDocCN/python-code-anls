# `.\pytorch\torch\csrc\utils\cuda_enabled.h`

```py
#pragma once

namespace torch::utils {

// 定义一个内联函数，用于检查是否启用了 CUDA
inline constexpr bool cuda_enabled() {
#ifdef USE_CUDA
  // 如果定义了 USE_CUDA 宏，则返回 true
  return true;
#else
  // 如果未定义 USE_CUDA 宏，则返回 false
  return false;
#endif
}

} // namespace torch::utils
```