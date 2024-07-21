# `.\pytorch\c10\util\Type_no_demangle.cpp`

```
#include <c10/macros/Macros.h>
// 包含 c10 库中的宏定义文件 Macros.h

#if HAS_DEMANGLE == 0
// 如果 HAS_DEMANGLE 宏定义为 0，则执行以下代码块

#include <c10/util/Type.h>
// 包含 c10 库中的 Type.h 头文件

namespace c10 {
// 声明 c10 命名空间

std::string demangle(const char* name) {
  // 定义 demangle 函数，接受一个指向字符的指针 name

  return std::string(name);
  // 返回一个包含 name 字符串内容的 std::string 对象
}

} // namespace c10
// 结束 c10 命名空间的定义
#endif // !HAS_DEMANGLE
// 结束条件编译指令，表示如果 HAS_DEMANGLE 不为 0，则结束条件编译块
```