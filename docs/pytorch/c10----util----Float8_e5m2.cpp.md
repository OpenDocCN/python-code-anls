# `.\pytorch\c10\util\Float8_e5m2.cpp`

```py
#include <c10/util/Float8_e5m2.h>  // 包含 Float8_e5m2 类的头文件

namespace c10 {

static_assert(
    std::is_standard_layout_v<Float8_e5m2>,  // 使用 static_assert 检查 Float8_e5m2 是否是标准布局
    "c10::Float8_e5m2 must be standard layout.");  // 如果不是标准布局，输出错误信息

} // namespace c10  // 命名空间 c10 的结束标记
```