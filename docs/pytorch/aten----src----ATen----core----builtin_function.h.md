# `.\pytorch\aten\src\ATen\core\builtin_function.h`

```py
#pragma once

#include <ATen/core/function.h>
#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <functional>
#include <utility>

namespace torch::jit {

// 自定义结构体 BuiltinOpFunction，继承自 Function 类
struct BuiltinOpFunction : public Function {
  // 构造函数，初始化成员变量
  BuiltinOpFunction(
      c10::QualifiedName qualname,          // 函数限定名称
      c10::FunctionSchema schema,           // 函数的模式
      std::function<void(Stack&)> callable, // 可调用函数对象，接受 Stack 引用参数
      std::string doc_string = "")          // 函数的文档字符串，默认为空字符串
      : name_(std::move(qualname)),
        callable_(std::move(callable)),
        schema_(std::move(schema)),
        doc_string_(std::move(doc_string)) {
    TORCH_INTERNAL_ASSERT(schema_.returns().size() == 1);  // 确保返回值的大小为1
  }

  // 返回函数的文档字符串
  c10::string_view doc_string() const override {
    return doc_string_;
  }

  // 执行函数，调用内部的可调用对象 callable_
  void run(Stack& stack) override {
    callable_(stack);
  }

  // 异步执行函数，暂时不使用 TaskLauncher 参数
  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& stack,
      TaskLauncher /* not used */) override {
    run(stack);  // 调用同步的 run 函数
    // 创建一个 Future 对象，标记为已完成，并传递栈顶的值
    auto res = c10::make_intrusive<c10::ivalue::Future>(stack.front().type());
    res->markCompleted(std::move(stack.front()));
    return res;
  }

  // 返回函数的限定名称
  const c10::QualifiedName& qualname() const override {
    return name_;
  }

  // 确保函数已定义，这里是空操作
  void ensure_defined() override {
    // nop
  }

  // 返回函数的模式
  const c10::FunctionSchema& getSchema() const override {
    return schema_;
  }

  // 返回函数输入参数的数量
  size_t num_inputs() const override {
    return schema_.arguments().size();
  }

  // 设置函数的模式
  Function& setSchema(c10::FunctionSchema schema) override {
    schema_ = std::move(schema);
    return *this;
  }

  // 调用函数，第二个参数未定义大小，接受 Code 类的函数对象
  bool call(
      Stack& stack,
      std::optional<size_t>,
      c10::function_ref<void(const Code&)>) override {
    run(stack);  // 调用同步的 run 函数
    return false;
  }

  // 调用函数，接受 mobile::Code 类的函数对象
  bool call(Stack& stack, c10::function_ref<void(const mobile::Code&)>) override {
    run(stack);  // 调用同步的 run 函数
    return false;
  }

  // 默认析构函数
  ~BuiltinOpFunction() override = default;

 private:
  c10::QualifiedName name_;                    // 函数的限定名称
  std::function<void(Stack&)> callable_;       // 可调用函数对象
  c10::FunctionSchema schema_;                 // 函数的模式
  std::string doc_string_;                     // 函数的文档字符串
};

} // namespace torch::jit


这段代码定义了一个 `BuiltinOpFunction` 结构体，用于表示内置操作函数对象，继承自 `Function` 类，实现了各种函数操作的接口，包括同步执行和异步执行，函数模式的获取和设置，以及函数的调用和文档字符串的处理等功能。
```