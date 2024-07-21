# `.\pytorch\torch\csrc\jit\frontend\error_report.h`

```
#pragma once


// 声明指令：指定头文件只编译一次



#include <c10/util/Optional.h>
#include <torch/csrc/jit/frontend/tree.h>


// 包含必要的头文件：包括可选类型和树结构，用于 JIT 前端

namespace torch {
namespace jit {


// 定义命名空间 torch::jit



struct Call {
  std::string fn_name;
  SourceRange caller_range;
};


// 定义结构体 Call，表示函数调用的名称和源码范围



struct TORCH_API ErrorReport : public std::exception {
  ErrorReport(const ErrorReport& e);


// 定义错误报告结构体 ErrorReport，继承自 std::exception
// 构造函数：复制构造函数声明



  explicit ErrorReport(SourceRange r);
  explicit ErrorReport(const TreeRef& tree) : ErrorReport(tree->range()) {}
  explicit ErrorReport(const Token& tok) : ErrorReport(tok.range) {}


// 构造函数：根据源码范围、树引用或标记生成错误报告



  const char* what() const noexcept override;


// 重写 std::exception 的虚函数 what()，返回错误信息的 C 风格字符串



  struct TORCH_API CallStack {
    // These functions are used to report why a function was being compiled
    // (i.e. what was the call stack of user functions at compilation time that
    // led to this error)
    CallStack(const std::string& name, const SourceRange& range);
    ~CallStack();

    // Change the range that is relevant for the current function (i.e. after
    // each successful expression compilation, change it to the next expression)
    static void update_pending_range(const SourceRange& range);
  };


// 定义嵌套结构体 CallStack，用于描述函数编译时的调用堆栈
// 构造函数：初始化调用堆栈记录特定函数名称和源码范围
// 析构函数：清理调用堆栈
// 静态成员函数：更新当前函数相关的源码范围

  static std::string current_call_stack();


// 声明静态成员函数：获取当前调用堆栈的字符串表示



 private:
  template <typename T>
  friend const ErrorReport& operator<<(const ErrorReport& e, const T& t);


// 声明友元模板函数：允许错误报告对象与流插入运算符 << 交互



  mutable std::stringstream ss;
  OwnedSourceRange context;
  mutable std::string the_message;
  std::vector<Call> error_stack;
};


// 成员变量声明：
// - ss: 可变的字符串流，用于动态生成错误消息
// - context: 拥有的源码范围对象
// - the_message: 可变的错误消息字符串
// - error_stack: 函数调用堆栈的向量容器



template <typename T>
const ErrorReport& operator<<(const ErrorReport& e, const T& t) {
  e.ss << t;
  return e;
}


// 定义模板函数：重载 << 运算符，将参数 t 插入错误报告 e 的字符串流 ss 中
// 返回错误报告对象的常引用



} // namespace jit
} // namespace torch


// 命名空间尾部标记
```