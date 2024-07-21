# `.\pytorch\torch\csrc\Types.h`

```
#ifndef THP_TYPES_INC
#define THP_TYPES_INC

# 如果 THP_TYPES_INC 宏未定义，则开始条件编译，防止多次包含


#include <cstddef>

# 包含标准库头文件 <cstddef>，提供 std::size_t 和 nullptr_t 的定义


#ifndef INT64_MAX
#include <cstdint>
#endif

# 如果 INT64_MAX 宏未定义，则条件包含 <cstdint> 头文件，提供整数类型的固定宽度定义


template <typename T>
struct THPTypeInfo {};

# 定义一个空的模板结构 THPTypeInfo，用于提供类型 T 的信息


#endif

# 结束条件编译指令
```