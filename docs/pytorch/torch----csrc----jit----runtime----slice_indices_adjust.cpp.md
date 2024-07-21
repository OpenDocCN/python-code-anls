# `.\pytorch\torch\csrc\jit\runtime\slice_indices_adjust.cpp`

```
// 引入 Torch 库中的头文件，包含了运行时切片索引调整的函数声明
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>

// 引入 C10 库中的异常处理头文件和 int64_t 类型定义
#include <c10/util/Exception.h>
#include <cstdint>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 定义函数 slice_indices_adjust，用于调整列表切片的索引
int64_t slice_indices_adjust(
    int64_t length,     // 列表的长度
    int64_t* start,     // 切片的起始索引
    int64_t* stop,      // 切片的结束索引
    int64_t step) {     // 切片的步长

  // 检查步长不能为零
  TORCH_CHECK(step != 0, "List slice should have non-zero step")

  // 检查步长不能超出 INT64 的范围
  TORCH_CHECK(step >= -INT64_MAX, "List slice step is out of bounds")

  // 根据 PySlice_Unpack 的规则调整起始索引
  if (*start == INT64_MAX) {
    *start = (step < 0) ? INT64_MAX : 0;
  }
  // 根据 PySlice_Unpack 的规则调整结束索引
  if (*stop == INT64_MAX) {
    *stop = (step < 0) ? INT64_MIN : INT64_MAX;
  }

  // 根据 PySlice_AdjustIndices 的规则调整起始索引
  if (*start < 0) {
    *start += length;
    if (*start < 0) {
      *start = (step < 0) ? -1 : 0;
    }
  } else if (*start >= length) {
    *start = (step < 0) ? length - 1 : length;
  }

  // 根据 PySlice_AdjustIndices 的规则调整结束索引
  if (*stop < 0) {
    *stop += length;
    if (*stop < 0) {
      *stop = (step < 0) ? -1 : 0;
    }
  } else if (*stop >= length) {
    *stop = (step < 0) ? length - 1 : length;
  }

  // 根据步长正负情况计算切片的长度并返回
  if (step < 0) {
    if (*stop < *start) {
      return (*start - *stop - 1) / (-step) + 1;
    }
  } else {
    if (*start < *stop) {
      return (*stop - *start - 1) / step + 1;
    }
  }
  // 默认返回值，表示切片长度为 0
  return 0;
}

} // namespace torch::jit
```