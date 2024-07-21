# `.\pytorch\aten\src\ATen\core\operator_name.cpp`

```
// 引入 ATen 库中的 operator_name.h 文件

#include <ATen/core/operator_name.h>

// 定义 c10 命名空间，包含 OperatorName 相关函数和类
namespace c10 {

// 将 OperatorName 对象转换为字符串表示形式
std::string toString(const OperatorName& opName) {
  // 创建一个输出字符串流对象
  std::ostringstream oss;
  // 将 opName 写入到输出流中
  oss << opName;
  // 返回输出流中的字符串表示
  return oss.str();
}

// 定义 OperatorName 对象的流输出操作符重载
std::ostream& operator<<(std::ostream& os, const OperatorName& opName) {
  // 将 opName 的名称部分写入输出流
  os << opName.name;
  // 如果重载名称非空，添加重载名称到输出流中
  if (!opName.overload_name.empty()) {
    os << "." << opName.overload_name;
  }
  // 返回输出流
  return os;
}

} // namespace c10
```