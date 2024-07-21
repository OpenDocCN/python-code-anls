# `.\pytorch\torch\csrc\api\include\torch\imethod.h`

```
#pragma once
#include <ATen/core/ivalue.h>
#include <vector>

namespace torch {

class TORCH_API IMethod {
  /*
  IMethod provides a portable interface for torch methods, whether
  they are backed by torchscript or python/deploy.

  This is helpful since torchscript methods provide additional information
  (e.g. FunctionSchema, Graph) which aren't available in pure python methods.

  Higher level APIs should prefer depending on this interface rather
  than a specific implementation of it, to promote portability and reuse, and
  avoid unintentional dependencies on e.g. script methods.

  Note: This API is experimental, and may evolve.
  */
 public:
  using IValueList = std::vector<c10::IValue>;  // 定义了一个 IValueList 类型，是存储 c10::IValue 的 std::vector
  using IValueMap = std::unordered_map<std::string, at::IValue>;  // 定义了一个 IValueMap 类型，是存储 std::string 到 at::IValue 的映射

  IMethod() = default;  // 默认构造函数
  IMethod(const IMethod&) = default;  // 拷贝构造函数
  IMethod& operator=(const IMethod&) = default;  // 拷贝赋值运算符重载
  IMethod(IMethod&&) noexcept = default;  // 移动构造函数
  IMethod& operator=(IMethod&&) noexcept = default;  // 移动赋值运算符重载
  virtual ~IMethod() = default;  // 虚析构函数，用于多态销毁对象

  // 纯虚函数，定义了方法调用运算符，返回 c10::IValue
  virtual c10::IValue operator()(
      std::vector<c10::IValue> args,
      const IValueMap& kwargs = IValueMap()) const = 0;

  // 纯虚函数，返回方法的名称，必须由派生类实现
  virtual const std::string& name() const = 0;

  // 返回参数名称的有序列表，适用于脚本和 Python 方法的通用性需求
  // 比起 ScriptMethod 的 FunctionSchema 更加可移植
  const std::vector<std::string>& getArgumentNames() const;

 protected:
  // 设置参数名称，由派生类实现
  virtual void setArgumentNames(
      std::vector<std::string>& argumentNames) const = 0;

 private:
  mutable bool isArgumentNamesInitialized_{false};  // 可变的成员变量，用于标记参数名称是否已初始化
  mutable std::vector<std::string> argumentNames_;  // 可变的成员变量，存储参数名称的列表
};

} // namespace torch
```