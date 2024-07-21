# `.\pytorch\torch\csrc\jit\api\module_save.cpp`

```py
// 包含 Torch 的模块导出相关头文件
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/export.h>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 将模型保存到给定的输出流中，不使用字节码格式
void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) const {
  ExportModule(*this, out, extra_files, false /* bytecode_format */);
}

// 将模型保存到指定的文件中，不使用字节码格式
void Module::save(const std::string& filename, const ExtraFilesMap& extra_files) const {
  ExportModule(*this, filename, extra_files, false /* bytecode_format */);
}

// 为移动端保存模型到给定的输出流中，可以选择是否保存移动端调试信息和使用 FlatBuffer 格式
void Module::_save_for_mobile(
    std::ostream& out,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info,
    bool use_flatbuffer) const {
  ExportModule(
      *this,
      out,
      extra_files,
      true /* bytecode_format */,
      save_mobile_debug_info,
      use_flatbuffer);
}

// 为移动端保存模型到指定的文件中，可以选择是否保存移动端调试信息和使用 FlatBuffer 格式
void Module::_save_for_mobile(
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info,
    bool use_flatbuffer) const {
  ExportModule(
      *this,
      filename,
      extra_files,
      true /* bytecode_format */,
      save_mobile_debug_info,
      use_flatbuffer);
}

} // namespace torch::jit
```