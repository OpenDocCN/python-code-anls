# `.\pytorch\c10\util\ParallelGuard.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/macros/Macros.h>
// 包含 c10 库中的宏定义文件

namespace c10 {

// 声明命名空间 c10

// RAII thread local guard that tracks whether code is being executed in
// `at::parallel_for` or `at::parallel_reduce` loop function.
// RAII 线程局部保护器，用于跟踪代码是否在 `at::parallel_for` 或 `at::parallel_reduce` 循环函数中执行。

class C10_API ParallelGuard {
 public:
  // 公共成员函数声明部分

  // 静态函数，用于检查并返回是否启用了并行性
  static bool is_enabled();

  // 构造函数，用于初始化并设置并行性状态
  ParallelGuard(bool state);

  // 析构函数，用于清理并恢复先前的并行性状态
  ~ParallelGuard();

 private:
  // 私有成员变量部分

  // 保存先前的并行性状态
  bool previous_state_;
};

} // namespace c10
// 命名空间 c10 的结束
```