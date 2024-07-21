# `.\pytorch\c10\util\TypeCast.cpp`

```
// 包含 C10 库的 TypeCast.h 头文件

#include <c10/util/TypeCast.h>

// 定义 c10 命名空间
namespace c10 {

// 指示该函数不会返回，即使有 return 语句也不会执行到
[[noreturn]] void report_overflow(const char* name) {
  // 创建字符串流对象 oss
  std::ostringstream oss;
  // 格式化输出错误信息，指出值无法转换为指定类型 name 而不发生溢出
  oss << "value cannot be converted to type " << name << " without overflow";
  // 抛出运行时异常，包含 oss 中的错误信息
  throw std::runtime_error(oss.str()); // rather than domain_error (issue 33562)
}

} // namespace c10
```