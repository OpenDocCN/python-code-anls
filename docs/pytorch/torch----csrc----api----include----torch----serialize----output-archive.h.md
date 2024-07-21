# `.\pytorch\torch\csrc\api\include\torch\serialize\output-archive.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <torch/csrc/Export.h>
// 导入 Torch 库中的导出宏定义

#include <torch/csrc/jit/api/module.h>
// 导入 Torch 的 JIT 模块 API 中的模块定义

#include <iosfwd>
// 前置声明标准输入输出流相关类

#include <memory>
// 导入内存管理相关工具类

#include <string>
// 导入处理字符串的工具类

#include <utility>
// 导入实用工具类

namespace at {
class Tensor;
} // namespace at
// 声明 at 命名空间，并提供 Tensor 类的前置声明

namespace torch {
using at::Tensor;
namespace jit {
struct Module;
} // namespace jit
} // namespace torch
// 使用 at 命名空间中的 Tensor 类，并声明 torch 命名空间中的 jit 模块和 Module 结构体

namespace torch {
namespace serialize {
class TORCH_API OutputArchive final {
 public:
  explicit OutputArchive(std::shared_ptr<jit::CompilationUnit> cu);
  // 显式构造函数，接受一个 JIT 编译单元的共享指针作为参数

  explicit OutputArchive()
      : cu_(std::make_shared<jit::CompilationUnit>()),
        module_("__torch__.Module", cu_) {}
  // 默认构造函数，创建一个 JIT 编译单元的共享指针和一个名称为 "__torch__.Module" 的 Module 对象

  // 移动构造函数是允许的
  OutputArchive(OutputArchive&&) = default;
  // 移动构造函数，默认实现

  OutputArchive& operator=(OutputArchive&&) = default;
  // 移动赋值运算符，默认实现

  // 复制构造函数是禁用的
  OutputArchive(OutputArchive&) = delete;
  // 禁用复制构造函数

  OutputArchive& operator=(OutputArchive&) = delete;
  // 禁用复制赋值运算符

  std::shared_ptr<jit::CompilationUnit> compilation_unit() const {
    return cu_;
  }
  // 返回当前的 JIT 编译单元的共享指针

  /// Writes an `IValue` to the `OutputArchive`.
  void write(const std::string& key, const c10::IValue& ivalue);
  // 向 OutputArchive 写入一个 IValue 类型的数据，使用给定的键作为标识

  /// Writes a `(key, tensor)` pair to the `OutputArchive`, and marks it as
  /// being or not being a buffer (non-differentiable tensor).
  void write(
      const std::string& key,
      const Tensor& tensor,
      bool is_buffer = false);
  // 向 OutputArchive 写入一个键值对，其中值是一个 Tensor 对象，并可以标记其为缓冲区（不可微分的张量）

  /// Writes a nested `OutputArchive` under the given `key` to this
  /// `OutputArchive`.
  void write(const std::string& key, OutputArchive& nested_archive);
  // 向 OutputArchive 中写入一个嵌套的 OutputArchive 对象，使用给定的键标识该嵌套对象

  /// Saves the `OutputArchive` into a serialized representation in a file at
  /// `filename`.
  void save_to(const std::string& filename);
  // 将 OutputArchive 对象保存为序列化形式的文件，文件名由 filename 参数指定

  /// Saves the `OutputArchive` into a serialized representation into the given
  /// `stream`.
  void save_to(std::ostream& stream);
  // 将 OutputArchive 对象保存为序列化形式，并写入到给定的输出流 stream 中

  /// Saves the `OutputArchive` into a serialized representation using the
  /// given writer function.
  void save_to(const std::function<size_t(const void*, size_t)>& func);
  // 使用给定的写入函数将 OutputArchive 对象保存为序列化形式

  /// Forwards all arguments to `write()`.
  /// Useful for generic code that can be re-used for both `OutputArchive` and
  /// `InputArchive` (where `operator()` forwards to `read()`).
  template <typename... Ts>
  void operator()(Ts&&... ts) {
    write(std::forward<Ts>(ts)...);
  }
  // 运算符重载，将所有参数转发给 write() 函数，用于处理通用代码，可以在 OutputArchive 和 InputArchive 中重复使用

 private:
  std::shared_ptr<jit::CompilationUnit> cu_;
  // JIT 编译单元的共享指针

  jit::Module module_;
  // JIT 模块对象
};
} // namespace serialize
} // namespace torch
// 结束命名空间 torch 和 serialize
```