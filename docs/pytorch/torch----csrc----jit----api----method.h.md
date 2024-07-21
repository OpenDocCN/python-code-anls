# `.\pytorch\torch\csrc\jit\api\method.h`

```
#pragma once
// 指令：防止头文件重复包含，仅编译一次

#include <ATen/core/function.h>
// 包含：导入 ATen 核心功能函数头文件

#include <ATen/core/ivalue.h>
// 包含：导入 ATen 中 IValue 类的头文件

#include <ATen/core/stack.h>
// 包含：导入 ATen 中栈数据结构的头文件

#include <torch/csrc/api/include/torch/imethod.h>
// 包含：导入 Torch 中 IMethood 接口的头文件

#include <torch/csrc/jit/api/function_impl.h>
// 包含：导入 Torch JIT 中函数实现的头文件

namespace torch::jit {
// 命名空间：定义 Torch JIT 模块的范围

using ObjectPtr = c10::intrusive_ptr<c10::ivalue::Object>;
// 别名：定义 ObjectPtr 为 c10::ivalue::Object 的内部指针类型

// 表示一个模块中的方法，例如类 M 中的 f 方法
struct TORCH_API Method : public torch::IMethod {
  // 结构体：继承自 torch::IMethod 接口的 Method

  Method(ObjectPtr owner, Function* function);
  // 方法：构造函数，初始化方法的所有者和函数

  // 返回包含该方法的模块
  Module owner() const;
  // 方法：返回当前方法所属的模块

  // 返回原始的对象指针，用于当方法被 torchbind 对象所拥有时
  ObjectPtr raw_owner() const;
  // 方法：返回原始对象指针，当方法被 torchbind 对象拥有时使用

  void run(Stack& stack);
  // 方法：运行方法，传入栈引用作为参数

  void run(Stack&& stack) {
    run(stack);
  }
  // 方法：运行方法，传入栈的右值引用作为参数，并调用运行方法的栈引用版本

  c10::IValue operator()(
      std::vector<c10::IValue> stack,
      const Kwargs& kwargs = Kwargs()) const override;
  // 方法：重载操作符()，允许方法像函数一样调用，传入堆栈和关键字参数，并返回 IValue 类型的值

  // 异步运行方法。在该函数上的调用会启动一个 JIT 解释器，逐个在调用线程上执行操作。
  // 模型可以利用异步操作，例如 `fork`，来启动一个异步任务，该任务将在提供的 `taskLauncher` 上启动。
  c10::intrusive_ptr<c10::ivalue::Future> run_async(
      std::vector<c10::IValue> stack,
      const Kwargs& kwargs = Kwargs(),
      TaskLauncher taskLauncher = at::launch);
  // 方法：异步运行方法，传入堆栈、关键字参数和任务启动器，并返回 Future 类型的指针

  // 返回方法关联的图形
  std::shared_ptr<Graph> graph() const {
    return toGraphFunction(*function_).graph();
  }
  // 方法：返回与该方法关联的图形的共享指针

  // 返回方法的名称
  const std::string& name() const override {
    return function_->name();
  }
  // 方法：返回方法的名称

  // 返回方法的输入参数数量
  size_t num_inputs() const {
    return function_->num_inputs();
  }
  // 方法：返回方法的输入参数数量

  // 获取执行器
  GraphExecutor& get_executor() {
    return toGraphFunction(*function_).get_executor();
  }
  // 方法：获取方法的图形执行器

  // 返回方法的底层未绑定函数
  Function& function() const {
    return *function_;
  }
  // 方法：返回方法的底层未绑定函数

 private:
  void setArgumentNames(std::vector<std::string>&) const override;
  // 私有方法：设置参数名称，重写自基类的虚函数

  // 方法被单个模块唯一拥有。这个原始指针允许查找模块。
  ObjectPtr owner_;
  // 成员变量：方法的所有者对象指针

  // 底层未绑定函数
  Function* function_;
  // 成员变量：底层未绑定函数指针
};

namespace script {
// 命名空间：脚本命名空间，用于提供公共 API 的向后兼容性

// 曾经存在过一个 `script::` 命名空间，现在已删除。这是为了公共 API 的向后兼容性；新代码不应使用此类型别名。
using Method = ::torch::jit::Method;
// 别名：定义 Method 为 torch::jit::Method 类的别名
} // namespace script

} // namespace torch::jit
// 命名空间：结束 Torch JIT 模块的范围
```