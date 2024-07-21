# `.\pytorch\aten\src\ATen\native\Sorting.h`

```py
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub 头文件

#include <cstdint>
// 包含标准整数类型的头文件

namespace at {
class TensorBase;
}
// 在 at 命名空间中声明 TensorBase 类

namespace at::native {

enum class QUANTILE_INTERPOLATION_MODE : uint8_t {
  LINEAR,       // 线性插值模式
  LOWER,        // 较低插值模式
  HIGHER,       // 较高插值模式
  MIDPOINT,     // 中点插值模式
  NEAREST       // 最近邻插值模式
};
// 定义 QUANTILE_INTERPOLATION_MODE 枚举，表示不同的分位数插值方式

using sort_fn = void(*)(const TensorBase&, const TensorBase&, const TensorBase&, int64_t, bool, bool);
// 定义 sort_fn 类型别名，表示排序函数指针类型

using topk_fn = void(*)(const TensorBase&, const TensorBase&, const TensorBase&, int64_t, int64_t, bool, bool);
// 定义 topk_fn 类型别名，表示 topk 函数指针类型

DECLARE_DISPATCH(sort_fn, sort_stub);
// 声明 sort_stub 函数的调度分发接口

DECLARE_DISPATCH(topk_fn, topk_stub);
// 声明 topk_stub 函数的调度分发接口

void _fill_indices(const TensorBase &indices, int64_t dim);
// 声明 _fill_indices 函数，用于填充索引数据到给定维度

} // namespace at::native
// 结束 at::native 命名空间
```