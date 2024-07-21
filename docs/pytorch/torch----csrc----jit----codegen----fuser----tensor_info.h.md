# `.\pytorch\torch\csrc\jit\codegen\fuser\tensor_info.h`

```py
#pragma once
#include <torch/csrc/Export.h>

#include <cstdint>

namespace torch {
namespace jit {
namespace fuser {

// Host-side view of TensorInfo
// TensorInfo 的宿主端视图

// Note dims[0] - we need to dynamically allocate the dims.
// 注意 dims[0] - 我们需要动态分配 dims。

struct TORCH_API TensorInfo {
  // Return a pointer to sizes array
  // 返回 sizes 数组的指针
  uint32_t* sizes(size_t nDim) {
    return &sizes_strides[0];
  }
  
  // Return a pointer to strides array for the given dimension
  // 返回给定维度的 strides 数组的指针
  uint32_t* strides(size_t nDim) {
    return &sizes_strides[nDim];
  }

  // Pointer to data associated with this TensorInfo
  // 与此 TensorInfo 相关联的数据指针
  void* data;

#pragma GCC diagnostic ignored "-Wpedantic"
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  // Flexible array member holding sizes and strides of the tensor
  // 柔性数组成员，保存张量的大小和步幅
  uint32_t sizes_strides[0];
#pragma GCC diagnostic pop
};

} // namespace fuser
} // namespace jit
} // namespace torch
```