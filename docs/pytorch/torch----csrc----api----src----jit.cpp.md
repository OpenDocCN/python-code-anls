# `.\pytorch\torch\csrc\api\src\jit.cpp`

```py
#include <torch/jit.h>
// 引入 Torch JIT 模块

#include <ATen/core/stack.h>
// 引入 ATen 核心模块中的 stack.h

#include <torch/csrc/jit/frontend/ir_emitter.h>
// 引入 Torch JIT 前端的 IR emitter

#include <memory>
// 引入内存管理相关的标准库头文件

#include <string>
// 引入字符串操作相关的标准库头文件

namespace torch {
namespace jit {

std::shared_ptr<CompilationUnit> compile(const std::string& source) {
  // 创建一个共享指针指向 CompilationUnit 类对象，作为编译结果的容器
  auto module = std::make_shared<CompilationUnit>();

  // 在 CompilationUnit 中定义新模块，使用传入的源代码字符串 source 进行定义，
  // nativeResolver() 提供符号解析器，nullptr 表示没有额外的依赖项
  module->define(c10::nullopt, source, nativeResolver(), nullptr);

  // 返回定义好的模块对象
  return module;
}

} // namespace jit
} // namespace torch
```