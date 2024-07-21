# `.\pytorch\aten\src\ATen\core\function.h`

```
#pragma once
// 防止头文件被多次包含

#include <ATen/core/function_schema.h>
// 包含 ATen 库的函数模式定义

#include <ATen/core/ivalue.h>
// 包含 ATen 库的 IValue 类定义

#include <ATen/core/qualified_name.h>
// 包含 ATen 库的限定名称定义

#include <c10/util/Exception.h>
// 包含 c10 库的异常处理定义

#include <c10/util/FunctionRef.h>
// 包含 c10 库的函数引用定义

namespace c10 {
struct FunctionSchema;
};
// 定义 c10 命名空间，包含 FunctionSchema 结构体声明

namespace at {
TORCH_API void launch(std::function<void()> func);
}
// 定义 at 命名空间，包含 launch 函数声明

namespace torch::jit {

struct Graph;
// 声明 Graph 结构体

struct Code;
// 声明 Code 结构体

namespace mobile {
struct Code;
}
// 声明 mobile 命名空间下的 Code 结构体

using Stack = std::vector<at::IValue>;
// 使用 ATen 库中的 IValue 类型定义 Stack 别名

using Kwargs = std::unordered_map<std::string, at::IValue>;
// 使用 ATen 库中的 IValue 类型定义 Kwargs 别名

struct RecursiveMethodCallError : public std::exception {};
// 声明 RecursiveMethodCallError 结构体，继承自 std::exception

using TaskLauncher = std::function<void(std::function<void()>)>;
// 使用 std::function 定义 TaskLauncher 类型别名，接受一个函数参数

TORCH_API void preoptimizeGraph(
    std::shared_ptr<Graph>& graph,
    bool disable_autocast = false);
// 声明 preoptimizeGraph 函数，接受一个图的共享指针和一个布尔值参数

// A Function is a pure Graph with no implicit `self` object bound.
// It contains schema information and the executor that manages the
// execution of the function. Method is a wrapper around an
// underlying Function that also provides a `self` object.
struct TORCH_API Function {
  Function() = default;
  // 默认构造函数

  Function(const Function&) = default;
  // 复制构造函数

  Function& operator=(const Function&) = default;
  // 复制赋值运算符重载

  Function(Function&&) noexcept = default;
  // 移动构造函数

  Function& operator=(Function&&) noexcept = default;
  // 移动赋值运算符重载

  virtual c10::string_view doc_string() const {
    // 虚函数，返回文档字符串视图，默认为空字符串视图
    static constexpr c10::string_view no_doc_string = "";
    return no_doc_string;
  }

  virtual bool isGraphFunction() const {
    // 虚函数，返回 false，表示不是图函数
    return false;
  }

  virtual void run(Stack& stack) = 0;
  // 纯虚函数，要求子类实现，在栈上运行函数

  virtual c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& /*stack*/,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      C10_UNUSED TaskLauncher taskLauncher = at::launch) {
    // 虚函数，暂未实现，用于异步运行函数
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
    return {};
  }

  at::IValue operator()(Stack stack, const Kwargs& kwargs = Kwargs()) {
    // 重载运算符，执行函数调用
    getSchema().checkAndNormalizeInputs(stack, kwargs);
    run(stack);
    return stack.front();
  }

  virtual const c10::QualifiedName& qualname() const = 0;
  // 纯虚函数，要求子类实现，返回限定名称的引用

  const std::string& name() const {


这段代码定义了一些命名空间、结构体和函数，用于支持模型执行和管理。
    // 返回当前对象的限定名称的名称部分
    return qualname().name();
    
    // 如果此方法还未定义，则运行其method_creator函数
    virtual void ensure_defined() = 0;
    
    // 获取当前函数对象的函数模式（schema）
    virtual const c10::FunctionSchema& getSchema() const = 0;
    
    // 返回当前函数对象期望的输入数量
    virtual size_t num_inputs() const = 0;
    
    // 设置当前函数对象的函数模式（schema）
    virtual Function& setSchema(c10::FunctionSchema schema) = 0;
    
    // call() 定义不同解释器实现与函数对象的交互方式。基本上，解释器需要提供一个回调函数
    // 以便在提供 Code 对象时告知函数对象应该执行什么操作。与其设计返回可选的 Code 对象的签名，
    // 需要解释器特殊处理空值情况，而回调方法更合理，让函数对象自己定义回退行为。
    // 如果 call() 返回 true，则回调成功完成，否则返回 false。
    // 
    // 用于服务器解释器的重载版本，需要一个 bailout size 用于图执行器。
    virtual bool call(
        Stack&,
        std::optional<size_t>,
        c10::function_ref<void(const Code&)>) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
      return false;
    }
    
    // 用于移动设备解释器的重载版本。
    virtual bool call(Stack&, c10::function_ref<void(const mobile::Code&)>) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
      return false;
    }
    
    // 默认析构函数，使用默认行为
    virtual ~Function() = default;
};
} // namespace torch::jit
```