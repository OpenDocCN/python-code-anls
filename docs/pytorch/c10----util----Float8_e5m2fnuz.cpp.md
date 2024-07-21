# `.\pytorch\c10\util\Float8_e5m2fnuz.cpp`

```
#include <c10/macros/Macros.h>
#include <c10/util/Float8_e5m2fnuz.h>

namespace c10 {

// 使用 static_assert 来检查类型 Float8_e5m2fnuz 是否是标准布局
static_assert(
    std::is_standard_layout_v<Float8_e5m2fnuz>,
    "c10::Float8_e5m2 must be standard layout.");

} // namespace c10
```