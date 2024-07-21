# `.\pytorch\aten\src\ATen\PadNd.h`

```
#pragma once
#include <c10/util/Exception.h>
#include <c10/util/string_view.h>

// 声明命名空间 at，用于存放所有的符号和类型
namespace at {

// 定义枚举类型 padding_mode，表示填充模式
enum class padding_mode {
  reflect,    // 反射填充模式
  replicate,  // 复制填充模式
  circular,   // 循环填充模式
  constant,   // 常数填充模式
};

// 定义一个静态内联函数 padding_mode_string，接受 padding_mode 枚举作为参数，返回对应填充模式的字符串表示
static inline c10::string_view padding_mode_string(padding_mode m) {
  // 使用 switch 语句根据填充模式枚举值 m 返回对应的字符串表示
  switch (m) {
    case padding_mode::reflect:
      return "reflect";   // 返回 "reflect" 表示反射填充模式
    case padding_mode::replicate:
      return "replicate"; // 返回 "replicate" 表示复制填充模式
    case padding_mode::circular:
      return "circular";  // 返回 "circular" 表示循环填充模式
    case padding_mode::constant:
      return "constant";  // 返回 "constant" 表示常数填充模式
  }
  // 如果枚举值 m 不在上述定义的范围内，则报错并显示无效的填充模式和具体枚举值
  TORCH_CHECK(false, "Invalid padding mode (", static_cast<int64_t>(m), ")");
}

} // namespace at
```