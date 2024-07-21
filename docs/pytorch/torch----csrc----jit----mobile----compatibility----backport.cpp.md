# `.\pytorch\torch\csrc\jit\mobile\compatibility\backport.cpp`

```
namespace torch {
namespace jit {

#include <ATen/core/ivalue.h>  // 引入 ATen 库中的 IValue 类
#include <caffe2/serialize/file_adapter.h>  // 引入 caffe2 序列化库中的文件适配器
#include <caffe2/serialize/inline_container.h>  // 引入 caffe2 序列化库中的内联容器
#include <torch/csrc/jit/mobile/compatibility/backport.h>  // 引入 Torch 移动端兼容性模块的后向兼容头文件
#include <torch/csrc/jit/mobile/compatibility/backport_manager.h>  // 引入 Torch 移动端兼容性模块的后向兼容管理器头文件
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>  // 引入 Torch 移动端兼容性模块的模型兼容性头文件

#include <string>  // 引入字符串标准库

using caffe2::serialize::IStreamAdapter;  // 使用 caffe2 序列化库中的 IStreamAdapter 类
using caffe2::serialize::PyTorchStreamWriter;  // 使用 caffe2 序列化库中的 PyTorchStreamWriter 类

const static BackportManager backportManager;  // 定义静态的后向兼容管理器对象 backportManager

// 声明 _backport_for_mobile_impl 方法，以便 _backport_for_mobile() 函数重载可以直接调用该方法
bool _backport_for_mobile_impl(
    std::istream& oss,
    PyTorchStreamWriter& writer,
    const int64_t to_version);

// 实现 _backport_for_mobile 函数，接受输入流和输出流，并进行模型版本回溯
bool _backport_for_mobile(
    std::istream& in,
    std::ostream& out,
    const int64_t to_version) {
  // 定义写入函数对象，将数据写入输出流
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };
  // 创建 PyTorchStreamWriter 对象，并调用 _backport_for_mobile_impl 进行实际的回溯操作
  PyTorchStreamWriter writer(writer_func);
  return _backport_for_mobile_impl(in, writer, to_version);
}

// 实现 _backport_for_mobile 函数，接受输入流和输出文件名，并进行模型版本回溯
bool _backport_for_mobile(
    std::istream& in,
    const std::string& output_filename,
    const int64_t to_version) {
  // 创建 PyTorchStreamWriter 对象，并调用 _backport_for_mobile_impl 进行实际的回溯操作
  PyTorchStreamWriter writer(output_filename);
  return _backport_for_mobile_impl(in, writer, to_version);
}

// 实现 _backport_for_mobile 函数，接受输入文件名和输出流，并进行模型版本回溯
bool _backport_for_mobile(
    const std::string& input_filename,
    std::ostream& out,
    const int64_t to_version) {
  // 打开输入文件流，如果打开失败则报错
  std::ifstream file_stream;
  std::unique_ptr<IStreamAdapter> istream_adapter;
  file_stream.open(input_filename, std::ifstream::in | std::ifstream::binary);
  if (!file_stream) {
    AT_ERROR("open file failed, file path: ", input_filename);
  }
  // 定义写入函数对象，将数据写入输出流
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };
  // 创建 PyTorchStreamWriter 对象，并调用 _backport_for_mobile_impl 进行实际的回溯操作
  PyTorchStreamWriter writer(writer_func);
  return _backport_for_mobile_impl(file_stream, writer, to_version);
}

// 实现 _backport_for_mobile 函数，接受输入文件名和输出文件名，并进行模型版本回溯
bool _backport_for_mobile(
    const std::string& input_filename,
    const std::string& output_filename,
    const int64_t to_version) {
  // 打开输入文件流，如果打开失败则报错
  std::ifstream file_stream;
  file_stream.open(input_filename, std::ifstream::in | std::ifstream::binary);
  if (!file_stream) {
    AT_ERROR("open file failed, file path: ", input_filename);
  }
  // 创建 PyTorchStreamWriter 对象，并调用 _backport_for_mobile_impl 进行实际的回溯操作
  PyTorchStreamWriter writer(output_filename);
  return _backport_for_mobile_impl(file_stream, writer, to_version);
}

// 实现 _backport_for_mobile_impl 方法，接受输入流、写入器和目标版本号，进行实际的模型回溯操作
bool _backport_for_mobile_impl(
    std::istream& oss,
    PyTorchStreamWriter& writer,
    const int64_t to_version) {
  // 如果 backportManager 中不存在目标版本号 + 1 的回溯函数，则返回 false
  if (!backportManager.hasBytecodeBackportFunction(to_version + 1)) {
    return false;
  }
  // 将输入流的位置移动到开头，获取模型的当前字节码版本号
  oss.seekg(0, oss.beg);
  auto from_version = _get_model_bytecode_version(oss);  // 获取模型的当前字节码版本号
  // 调用 backportManager 的回溯方法，进行模型回溯操作
  return backportManager.backport(oss, writer, from_version, to_version);
}

} // namespace jit
} // namespace torch
```