# `.\pytorch\c10\core\TensorOptions.cpp`

```
#include <c10/core/TensorOptions.h>

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/util/Optional.h>

#include <iostream>

namespace c10 {

// Note: TensorOptions properties are all optional, but (almost) all have
// getters that supply a default when the corresponding property is missing.
// Here we print the values returned by the default-supplying getters for
// properties that have them, along with an annotation if the value is
// returned by default. This gives the full picture of both the object's
// internal state and what its getters will return.

// 重载输出流操作符，用于打印 TensorOptions 对象的属性
std::ostream& operator<<(std::ostream& stream, const TensorOptions& options) {
  // Lambda 函数 print，用于打印属性和是否使用了默认值
  auto print = [&](const char* label, auto prop, bool has_prop) {
    stream << label << std::boolalpha << prop << (has_prop ? "" : " (default)");
  };

  // 打印 dtype 属性及是否使用了默认值
  print("TensorOptions(dtype=", options.dtype(), options.has_dtype());
  // 打印 device 属性及是否使用了默认值
  print(", device=", options.device(), options.has_device());
  // 打印 layout 属性及是否使用了默认值
  print(", layout=", options.layout(), options.has_layout());
  // 打印 requires_grad 属性及是否使用了默认值
  print(", requires_grad=", options.requires_grad(), options.has_requires_grad());
  // 打印 pinned_memory 属性及是否使用了默认值
  print(", pinned_memory=", options.pinned_memory(), options.has_pinned_memory());

  // 注意：memory_format() getter 不提供默认值，故此处直接输出其状态
  stream << ", memory_format=";
  if (options.has_memory_format()) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    stream << *options.memory_format_opt(); // 如果有值，则打印其值
  } else {
    stream << "(nullopt)"; // 如果没有值，则输出 (nullopt)
  }
  stream << ")";

  return stream; // 返回输出流
}

} // namespace c10


这段代码定义了一个输出流操作符的重载，用于打印 `TensorOptions` 对象的各种属性。
```