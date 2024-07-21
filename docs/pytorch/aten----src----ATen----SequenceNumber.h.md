# `.\pytorch\aten\src\ATen\SequenceNumber.h`

```
#pragma once

# 使用 `#pragma once` 来确保头文件只被编译一次，防止多重包含。


#include <c10/macros/Export.h>
#include <cstdint>

# 引入头文件 `<c10/macros/Export.h>` 和 `<cstdint>`，分别用于导出符号和定义整数类型。


// A simple thread local enumeration, used to link forward and backward pass
// ops and is used by autograd and observers framework

# 定义一个简单的线程本地枚举，用于链接前向和后向传递操作，被自动求导和观察器框架使用。


namespace at::sequence_number {

# 进入命名空间 `at::sequence_number`。


TORCH_API uint64_t peek();
TORCH_API uint64_t get_and_increment();

# 声明两个函数 `peek()` 和 `get_and_increment()`，它们分别返回 `uint64_t` 类型的值，使用 `TORCH_API` 指定它们的导出属性。


} // namespace at::sequence_number

# 结束命名空间 `at::sequence_number`。
```