# `.\pytorch\test\cpp\jit\test_exception.cpp`

```py
/*
 * 我们有一个用于异常的 Python 单元测试在 test/jit/test_exception.py。
 * 在这里添加一个 CPP 版本来验证从 C++ 抛出的期望异常类型。
 * 这在 Python 代码中很难测试，因为 C++ 异常会被转换为 Python 异常。
 */
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/jit.h>
#include <iostream>
#include <stdexcept>

namespace torch {
namespace jit {

namespace py = pybind11;

// 定义测试用例 TestException，测试断言异常
TEST(TestException, TestAssertion) {
  // 定义包含 Python 代码的字符串 pythonCode
  std::string pythonCode = R"PY(
  def foo():
    raise AssertionError("An assertion failed")
  )PY";
  
  // 编译 Python 代码并获取函数对象 cu_ptr
  auto cu_ptr = torch::jit::compile(pythonCode);
  
  // 将 cu_ptr 中名为 "foo" 的函数转换为 GraphFunction 对象 gf
  torch::jit::GraphFunction* gf =
      (torch::jit::GraphFunction*)&cu_ptr->get_function("foo");
  
  // 输出 gf 所包含的图形表示
  std::cerr << "Graph is\n" << *gf->graph() << std::endl;

  // 初始化用于捕获异常信息的变量
  bool is_jit_exception = false;
  std::string message;
  std::optional<std::string> exception_class;

  // 尝试运行 cu_ptr 中名为 "foo" 的函数，并捕获 JITException 异常
  try {
    cu_ptr->run_method("foo");
  } catch (JITException& e) {
    // 如果捕获到 JITException 异常，则设置相应标志和记录异常信息
    is_jit_exception = true;
    message = e.what();
    exception_class = e.getPythonClassName();
  }
  
  // 断言捕获到 JITException 异常
  EXPECT_TRUE(is_jit_exception);
  // 断言异常类为空（因为这里是 AssertionError，不是自定义的异常类）
  EXPECT_FALSE(exception_class);
  // 断言异常信息中包含特定的错误消息
  EXPECT_TRUE(
      message.find("RuntimeError: AssertionError: An assertion failed") !=
      std::string::npos);
}

// 自定义异常值类 MyPythonExceptionValue，继承自 SugaredValue
struct MyPythonExceptionValue : public torch::jit::SugaredValue {
  explicit MyPythonExceptionValue(const py::object& exception_class) {
    // 构造函数，获取并设置异常类的完全限定名
    qualified_name_ =
        (py::str(py::getattr(exception_class, "__module__", py::str(""))) +
         py::str(".") +
         py::str(py::getattr(exception_class, "__name__", py::str(""))))
            .cast<std::string>();
  }

  std::string kind() const override {
    // 返回异常值类型描述
    return "My Python exception";
  }

  // 从 PythonExceptionValue::call 简化而来的 call 方法
  std::shared_ptr<torch::jit::SugaredValue> call(
      const torch::jit::SourceRange& loc,
      torch::jit::GraphFunction& caller,
      at::ArrayRef<torch::jit::NamedValue> args,
      at::ArrayRef<torch::jit::NamedValue> kwargs,
      size_t n_binders) override {
    // 检查参数数量，这里期望参数为一个
    TORCH_CHECK(args.size() == 1);
    // 获取错误消息的值
    Value* error_message = args.at(0).value(*caller.graph());
    // 插入异常类名常量
    Value* qualified_class_name =
        insertConstant(*caller.graph(), qualified_name_, loc);
    // 返回 ExceptionMessageValue 的共享指针
    return std::make_shared<ExceptionMessageValue>(
        error_message, qualified_class_name);
  }

 private:
  std::string qualified_name_; // 异常类的完全限定名
};

// 简单解析器类 SimpleResolver，继承自 Resolver
class SimpleResolver : public torch::jit::Resolver {
 public:
  explicit SimpleResolver() {}

  std::shared_ptr<torch::jit::SugaredValue> resolveValue(
      const std::string& name,
      torch::jit::GraphFunction& m,
      const torch::jit::SourceRange& loc) override {
    // 实现 resolveValue 方法，根据名称解析 SugaredValue 值
    // 这里是简单的实现示例，具体操作可以参考 toSugaredValue 方法
    // toSugaredValue 方法定义在 caffe2:_C 中，是一个 Python 扩展，我们无法将其添加为 cpp_binary 的依赖项
    // 此处未完整实现解析逻辑，需要根据具体需求进行完善

      // 这里是简单的实现示例，具体操作可以参考 toSugaredValue 方法
      // toSugaredValue 方法定义在 caffe2:_C 中，是一个 Python 扩展，我们无法将其添加为 cpp_binary 的依赖项
      // 此处未完整实现解析逻辑，需要根据具体需求进行完善
      return nullptr; // 返回空指针作为默认实现
  }
};

} // namespace jit
} // namespace torch
    // 如果 name 等于 "SimpleValueError"，则从 Python 全局变量中获取该对象
    if (name == "SimpleValueError") {
      py::object obj = py::globals()["SimpleValueError"];
      // 返回一个自定义的异常对象，包含获取到的 Python 对象
      return std::make_shared<MyPythonExceptionValue>(obj);
    }
    // 如果上面的条件不满足，则触发一个 Torch 错误检查，报告无法解析 name 的错误信息
    TORCH_CHECK(false, "resolveValue: can not resolve '", name, "{}'");
  }

  // 解析类型的函数，根据给定的名称和源范围返回空指针
  torch::jit::TypePtr resolveType(
      const std::string& name,
      const torch::jit::SourceRange& loc) override {
    return nullptr;
  }
/*
 * - 定义一个单元测试 TestCustomException，用于测试自定义异常的情况
 * - 使用 py::scoped_interpreter 创建 Python 解释器的作用域
 */
TEST(TestException, TestCustomException) {
  py::scoped_interpreter guard{};
  /*
   * - 在 Python 中执行字符串形式的代码，定义一个 SimpleValueError 异常类
   * - 该异常类继承自 ValueError，并在初始化方法中调用父类的构造函数
   */
  py::exec(R"PY(
  class SimpleValueError(ValueError):
    def __init__(self, message):
      super().__init__(message)
  )PY");

  // 定义一个包含 Python 代码字符串的 std::string 对象
  std::string pythonCode = R"PY(
  def foo():
    raise SimpleValueError("An assertion failed")
  )PY";

  // 创建 TorchScript 的 Parser 对象，解析传入的 Python 代码
  torch::jit::Parser p(
      std::make_shared<torch::jit::Source>(pythonCode, "<string>", 1));
  // 解析 Python 函数定义，构建为 torch::jit::Def 对象
  auto def = torch::jit::Def(p.parseFunction(/*is_method=*/false));
  // 输出解析得到的函数定义
  std::cerr << "Def is:\n" << def << std::endl;
  // 创建 TorchScript 的 CompilationUnit 对象
  auto cu = std::make_shared<torch::jit::CompilationUnit>();
  // 定义 CompilationUnit，包括一个函数定义和一个简单的解析器
  (void)cu->define(
      c10::nullopt,
      {},
      {},
      {def},
      // 使用 SimpleResolver 替代 PythonResolver，因为 PythonResolver
      // 在 torch/csrc/jit/python/script_init.cpp 中定义，不在头文件中
      {std::make_shared<SimpleResolver>()},
      nullptr);
  // 获取名为 "foo" 的函数对象，转换为 GraphFunction 类型
  torch::jit::GraphFunction* gf =
      (torch::jit::GraphFunction*)&cu->get_function("foo");
  // 输出函数的计算图
  std::cerr << "Graph is\n" << *gf->graph() << std::endl;
  // 初始化异常处理相关变量
  bool is_jit_exception = false;
  std::optional<std::string> exception_class;
  std::string message;
  // 尝试执行名为 "foo" 的方法，捕获可能抛出的 JITException 异常
  try {
    cu->run_method("foo");
  } catch (JITException& e) {
    // 如果捕获到 JITException 异常，设置相关标志和异常信息
    is_jit_exception = true;
    exception_class = e.getPythonClassName();
    message = e.what();
  }
  // 断言捕获到 JITException 异常
  EXPECT_TRUE(is_jit_exception);
  // 断言异常类名为 "__main__.SimpleValueError"
  EXPECT_EQ("__main__.SimpleValueError", *exception_class);
  // 断言异常消息包含 "__main__.SimpleValueError: An assertion failed"
  EXPECT_TRUE(
      message.find("__main__.SimpleValueError: An assertion failed") !=
      std::string::npos);
}
```