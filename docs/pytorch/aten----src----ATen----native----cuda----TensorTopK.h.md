# `.\pytorch\aten\src\ATen\native\cuda\TensorTopK.h`

```
#pragma once
// 使用 pragma once 指令确保头文件只被包含一次，避免重复定义

#include <cstdint>
// 包含标准整数类型头文件，以使用 int64_t 类型

namespace at {
// 开始命名空间 at

class TensorBase;
// 声明 TensorBase 类，但不定义其具体实现

}

namespace at {
namespace native {
// 嵌套在 at 命名空间下的 native 命名空间

void launch_gather_topk_kernel(
    const TensorBase& self,
    // 以 TensorBase 引用作为第一个参数，表示被操作的张量对象自身
    int64_t k, int64_t dim, bool largest,
    // 整数 k、dim 和布尔值 largest，用于指定操作的参数
    const TensorBase& values, const TensorBase& indices);
    // 两个 TensorBase 类型的引用作为输入参数，表示要填充的值和索引

}}
// 结束 at 命名空间下的 native 命名空间
```