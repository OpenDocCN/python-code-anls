# `.\pytorch\torch\csrc\jit\runtime\static\static_method.h`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <torch/csrc/api/include/torch/imethod.h>
// 引入 Torch 中的 IMethod 接口

#include <torch/csrc/jit/runtime/static/impl.h>
// 引入 Torch 中静态运行时的实现

namespace torch::jit {

class StaticMethod : public torch::IMethod {
// 定义 StaticMethod 类，继承自 torch::IMethod 接口
 public:
  StaticMethod(
      std::shared_ptr<StaticModule> static_module,
      std::string method_name)
      : static_module_(std::move(static_module)),
        method_name_(std::move(method_name)) {
    // 构造函数：接受静态模块和方法名称作为参数
    TORCH_CHECK(static_module_);
    // 使用 TORCH_CHECK 确保 static_module_ 不为空
  }

  c10::IValue operator()(
      std::vector<IValue> args,
      const IValueMap& kwargs = IValueMap()) const override {
    // 重载运算符：调用 StaticModule 的运算符重载，执行静态方法
    return (*static_module_)(std::move(args), kwargs);
    // 返回调用结果
  }

  const std::string& name() const override {
    // 获取方法名称的常量引用
    return method_name_;
  }

 protected:
  void setArgumentNames(
      std::vector<std::string>& argument_names_out) const override {
    // 设置参数名称列表
    const auto& schema = static_module_->schema();
    // 获取静态模块的模式
    CAFFE_ENFORCE(schema.has_value());
    // 使用 CAFFE_ENFORCE 确保模式值存在
    const auto& arguments = schema->arguments();
    // 获取模式的参数列表
    argument_names_out.clear();
    // 清空输出的参数名称列表
    argument_names_out.reserve(arguments.size());
    // 预留足够的空间以容纳所有参数
    std::transform(
        arguments.begin(),
        arguments.end(),
        std::back_inserter(argument_names_out),
        [](const c10::Argument& arg) -> std::string { return arg.name(); });
    // 使用 lambda 表达式将参数名称转换为字符串并插入到参数名称列表中
  }

 private:
  std::shared_ptr<StaticModule> static_module_;
  // 指向静态模块的共享指针
  std::string method_name_;
  // 方法名称
};

} // namespace torch::jit
// 结束 torch::jit 命名空间
```