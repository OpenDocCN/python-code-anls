# `.\pytorch\aten\src\ATen\core\Range.cpp`

```
#include <ATen/core/Range.h>  // 包含 ATen 库中的 Range.h 文件，用于范围处理

#include <ostream>  // 包含标准输出流操作相关的头文件

namespace at {

std::ostream& operator<<(std::ostream& out, const Range& range) {
  out << "Range[" << range.begin << ", " << range.end << "]";  // 输出 Range 对象的起始和结束值
  return out;  // 返回输出流对象
}

}  // namespace at  // 结束 ATen 命名空间
```