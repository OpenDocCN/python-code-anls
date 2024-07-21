# `.\pytorch\aten\src\ATen\cuda\CUDAUtils.h`

```
#pragma once

#include <ATen/cuda/CUDAContext.h>  // 引入 CUDA 相关的头文件

namespace at::cuda {

// 检查给定张量列表中的每个张量是否与当前设备匹配
inline bool check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {  // 如果张量列表为空，则直接返回true
    return true;
  }
  // 获取当前 CUDA 设备
  Device curDevice = Device(kCUDA, current_device());
  // 遍历张量列表
  for (const Tensor& t : ts) {
    // 如果张量的设备与当前设备不匹配，则返回false
    if (t.device() != curDevice) return false;
  }
  // 所有张量都匹配当前设备，返回true
  return true;
}

} // namespace at::cuda
```