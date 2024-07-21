# `.\pytorch\torch\csrc\jit\mobile\code.h`

```
// 防止头文件被多次包含的预处理指令
#pragma once

// 引入标准库中的向量容器
#include <vector>

// 引入 ATen 库中的相关头文件
#include <ATen/core/ivalue.h>
#include <ATen/core/operator_name.h>

// 引入 Torch 库中的 JIT 运行时指令头文件
#include <torch/csrc/jit/runtime/instruction.h>

// 声明命名空间 torch::jit::mobile
namespace torch {
namespace jit {
namespace mobile {

// 使用 typedef 定义 Stack 为 std::vector<c10::IValue> 类型
using Stack = std::vector<c10::IValue>;

// 使用 typedef 定义 DebugHandle 为 int64_t 类型
using DebugHandle = int64_t;

// 前置声明 Function 类
class Function;

// 定义结构体 Code，用于存储模型推理代码相关信息
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct Code {
  // 存储指令序列的向量
  std::vector<Instruction> instructions_;
  
  // 存储调试信息句柄的向量
  std::vector<DebugHandle> debug_handles_;
  
  // 存储运算符名称的向量
  std::vector<c10::OperatorName> op_names_;
  
  // 存储运算符输入大小的向量
  std::vector<int> operator_input_sizes_;
  
  // 存储操作函数指针的向量
  std::vector<std::function<void(Stack&)>> operators_;
  
  // 存储常量值的向量
  std::vector<c10::IValue> constants_;
  
  // 存储类型信息的向量
  std::vector<c10::TypePtr> types_;
  
  // 存储函数指针的向量，表示代码中引用的函数对象
  // TODO: 实际导出 CALL 指令后可以移除此注释
  // 在实现 parseMethods() 时，可能需要两阶段导入方案，
  // 先构造所有函数对象，然后追加引用的函数指针。
  std::vector<mobile::Function*> functions_;
  
  // 聚合输出大小的变量
  size_t register_size_ = 0;
  
  // 标志位，指示操作函数指针数组是否已填充
  bool initialized = false;
};

} // namespace mobile
} // namespace jit
} // namespace torch
```