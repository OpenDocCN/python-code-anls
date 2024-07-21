# `.\pytorch\aten\src\ATen\templates\ATenOpList.cpp`

```py
// 包含 ATen 操作列表的头文件
#include <ATen/core/ATenOpList.h>

// 包含必要的标准库头文件
#include <string>
#include <cstring>
#include <utility>
#include <unordered_set>
#include <ATen/core/operator_name.h>

// ${generated_comment} 占位符，可能会被代码生成器替换为生成的注释

// ATen 命名空间
namespace at {

// 匿名命名空间，用于定义局部结构和函数

// 比较操作符，用于比较一对操作符名称是否相等
struct OpNameEquals final {
  bool operator()(const std::pair<const char*, const char*>& lhs, const std::pair<const char*, const char*>& rhs) const {
      return 0 == strcmp(lhs.first, rhs.first) && 0 == strcmp(lhs.second, rhs.second);
  }
};

// 哈希函数对象，用于计算操作符名称对的哈希值
struct OpNameHash final {
  size_t operator()(const std::pair<const char*, const char*>& p) const {
      // 使用 std::hash<std::string>，因为 std::hash<const char*> 会哈希指针而不是指向的字符串
      return std::hash<std::string>()(p.first) ^ (~ std::hash<std::string>()(p.second));
  }
};

// 判断给定的操作符名称是否是自定义操作符
bool is_custom_op(const c10::OperatorName& opName) {
  // 静态无序集合，存储一对操作符名称的指针，使用自定义的哈希和相等函数对象
  static std::unordered_set<std::pair<const char*, const char*>, OpNameHash, OpNameEquals> ops {
    ${aten_ops} // 这里可能会被代码生成器替换为实际的操作符名称对列表
    {"", ""}    // 初始为空字符串对，占位符
  };
  // 返回集合中是否包含给定的操作符名称对
  return ops.count(std::make_pair(
             opName.name.c_str(), opName.overload_name.c_str())) == 0;
}

} // namespace at
```