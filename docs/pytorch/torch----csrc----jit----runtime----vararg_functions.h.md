# `.\pytorch\torch\csrc\jit\runtime\vararg_functions.h`

```py
// 预处理指令，确保头文件只被编译一次
#pragma once

// 包含 ATen 库的 List.h 头文件，用于处理列表
#include <ATen/core/List.h>

// 包含 ATen 库的 functional.h 头文件，用于函数式编程支持
#include <ATen/core/functional.h>

// 包含 ATen 库的 ivalue.h 头文件，用于处理 IValue 类型
#include <ATen/core/ivalue.h>

// 包含 ATen 库的 jit_type.h 头文件，用于 JIT 类型支持
#include <ATen/core/jit_type.h>

// 包含 ATen 库的 stack.h 头文件，用于堆栈操作
#include <ATen/core/stack.h>

// torch::jit 命名空间，包含了 TorchScript JIT 编译器的相关函数和类
namespace torch::jit {

// 定义函数 tupleUnpack，用于从堆栈中解包元组
void tupleUnpack(Stack& stack);

// 定义函数 format，格式化堆栈中的输入数据
void format(Stack& stack, size_t num_inputs);

// 定义函数 einsum，执行爱因斯坦求和符号计算
void einsum(Stack& stack, size_t num_inputs);

// 定义函数 percentFormat，百分比格式化堆栈中的输入数据
void percentFormat(Stack& stack, size_t num_inputs);

// 定义函数 listUnpack，从堆栈中解包列表
void listUnpack(Stack& stack, size_t num_outputs);

// 定义函数 tupleConstruct，构造元组并压入堆栈
void tupleConstruct(Stack& stack, size_t num_inputs);

// 定义函数 namedTupleConstruct，构造具名元组并压入堆栈
void namedTupleConstruct(Stack& stack, c10::TypePtr type, size_t num_inputs);

// 定义函数 listConstruct，构造列表并压入堆栈
void listConstruct(Stack& stack, const c10::Type& list_type, size_t num_inputs);

// 定义函数 dictConstruct，构造字典并压入堆栈
void dictConstruct(Stack& stack, const c10::Type& type, size_t num_inputs);

// 定义函数 createObject，创建一个 Object 对象用于在图中作为常量使用，避免引用循环
// as_weak_ref 参数用于创建一个非拥有 CompilationUnit 引用的 Object
void createObject(
    Stack& stack,
    const at::ClassTypePtr& type,
    bool as_weak_ref = false);

// 定义函数 isinstance，检查对象是否属于指定的类型
void isinstance(Stack& stack, at::ArrayRef<at::TypePtr> types);

// 定义函数 tupleSlice，对元组进行切片操作
void tupleSlice(Stack& stack, size_t begin, size_t end);

// 定义函数 dequantize，对量化张量进行去量化操作
void dequantize(Stack& stack);

} // namespace torch::jit
```