# `.\pytorch\aten\src\ATen\core\Range.h`

```
#pragma once
// 预处理指令，确保本文件仅被编译一次

#include <cstdint>
// 包含标准整数类型的头文件，如 int64_t

#include <iosfwd>
// 前置声明标准输入输出流的头文件

namespace at {
// 命名空间定义开始

struct Range {
  // 定义名为 Range 的结构体
  Range(int64_t begin, int64_t end)
    : begin(begin)
    , end(end) {}
    // 结构体的构造函数，初始化 begin 和 end 成员变量

  int64_t size() const { return end - begin; }
  // 返回 Range 对象的大小，即 end 和 begin 之差

  Range operator/(int64_t divisor) {
    // 重载除法运算符，使 Range 对象可以与整数相除
    return Range(begin / divisor, end / divisor);
    // 返回一个新的 Range 对象，其 begin 和 end 分别除以给定的除数
  }

  int64_t begin;
  // 整型变量，表示 Range 的起始值

  int64_t end;
  // 整型变量，表示 Range 的结束值
};

std::ostream& operator<<(std::ostream& out, const Range& range);
// 重载输出流操作符，使得 Range 对象可以直接输出到流中

}  // namespace at
// 命名空间定义结束
```