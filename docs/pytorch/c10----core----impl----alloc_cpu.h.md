# `.\pytorch\c10\core\impl\alloc_cpu.h`

```py
#pragma once

# 使用 `#pragma once` 指令，确保当前头文件只被编译一次，避免重复包含


#include <c10/macros/Export.h>

# 包含 `c10/macros/Export.h` 头文件，该头文件可能定义了导出符号相关的宏和设置


#include <cstddef>

# 包含 `<cstddef>` 头文件，提供了 `std::size_t` 类型以及与大小相关的操作


namespace c10 {

# 进入命名空间 `c10`


C10_API void* alloc_cpu(size_t nbytes);

# 声明函数 `alloc_cpu`，返回 `void*` 指针，接受一个 `size_t` 类型的参数 `nbytes`，用于在 CPU 上分配内存


C10_API void free_cpu(void* data);

# 声明函数 `free_cpu`，不返回值，接受一个 `void*` 类型的参数 `data`，用于释放在 CPU 上分配的内存


} // namespace c10

# 结束命名空间 `c10`
```