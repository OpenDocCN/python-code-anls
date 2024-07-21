# `.\pytorch\torch\csrc\api\src\serialize\output-archive.cpp`

```
// 包含 Torch 序列化所需的头文件

#include <torch/serialize/output-archive.h>

// 包含 Torch 的类型定义和实用函数

#include <torch/types.h>
#include <torch/utils.h>

// 包含 Torch JIT 模块的相关头文件

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/export.h>

// 包含 C10 库的异常处理头文件

#include <c10/util/Exception.h>

// 包含 C++ 标准库的头文件

#include <memory>
#include <ostream>
#include <string>

// Torch 命名空间开始

namespace torch {
// Torch 序列化命名空间开始
namespace serialize {

// OutputArchive 类的构造函数，接收一个 JIT 编译单元的共享指针作为参数
OutputArchive::OutputArchive(std::shared_ptr<jit::CompilationUnit> cu)
    : cu_(std::move(cu)),
      // 使用给定的编译单元和模块名创建模块对象，启用名称重整
      module_("__torch__.Module", cu_, /*shouldMangle=*/true) {}

// 写入方法，将给定的键和 IValue 对象注册为模块的属性
void OutputArchive::write(const std::string& key, const c10::IValue& ivalue) {
  module_.register_attribute(key, ivalue.type(), ivalue);
}

// 写入方法，将给定的键和张量注册为模块的参数或缓冲区
void OutputArchive::write(
    const std::string& key,
    const Tensor& tensor,
    bool is_buffer) {
  module_.register_parameter(key, tensor, is_buffer);
}

// 写入方法，将给定的键和嵌套的 OutputArchive 对象注册为模块的子模块
void OutputArchive::write(
    const std::string& key,
    OutputArchive& nested_archive) {
  module_.register_module(key, nested_archive.module_);
}

// 将模块保存到指定的文件中
void OutputArchive::save_to(const std::string& filename) {
  jit::ExportModule(module_, filename);
}

// 将模块保存到指定的输出流中
void OutputArchive::save_to(std::ostream& stream) {
  jit::ExportModule(module_, stream);
}

// 将模块保存到指定的函数对象中
void OutputArchive::save_to(
    const std::function<size_t(const void*, size_t)>& func) {
  jit::ExportModule(module_, func);
}

} // namespace serialize
} // namespace torch
// Torch 命名空间结束
```