# `.\pytorch\torch\csrc\jit\mobile\method.h`

```
#pragma once
// 预处理指令：确保本头文件只被编译一次

#include <ATen/core/ivalue.h>
// 引入 ATen 库中的 IValue 类

#include <torch/csrc/jit/mobile/function.h>
// 引入 Torch 库中的移动端函数相关定义

namespace torch {
namespace jit {
namespace mobile {

class Module;
// 声明 Module 类，用于后续的引用

struct TORCH_API Method {
  // 方法结构体，表示模块中的一个方法

  Method(const Module* owner, Function* function);
  // 构造函数：使用给定的模块和函数指针初始化 Method 对象

  void run(Stack& stack) const;
  // 执行方法：接收一个栈引用参数，执行方法的运算过程

  void run(Stack&& stack) const {
    run(stack);
  }
  // 右值引用版本的执行方法，调用左值引用版本来执行实际逻辑

  c10::IValue operator()(std::vector<c10::IValue> stack) const;
  // 函数调用运算符重载：接收一个 IValue 类型的向量作为参数，执行方法调用并返回结果

  const std::string& name() const {
    return function_->name();
  }
  // 获取方法名：返回该方法对应函数的名称

  int64_t get_debug_handle(size_t pc) const {
    return function_->get_debug_handle(pc);
  }
  // 获取调试句柄：返回方法中给定程序计数器（PC）位置处的调试句柄

  Function& function() const {
    return *function_;
  }
  // 获取函数引用：返回该方法对应的函数对象的引用

 private:
  // 方法对象是由单个模块所有的
  // owner_ 指向模块的常量指针，用于引用该方法所属的模块
  const Module* owner_;

  // 底层未绑定函数
  // function_ 指向函数的指针，表示该方法所调用的函数对象
  Function* function_;
};

} // namespace mobile
} // namespace jit
} // namespace torch
```