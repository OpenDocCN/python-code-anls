# `.\pytorch\torch\csrc\jit\serialization\import_export_helpers.h`

```py
#pragma once
// 声明这个头文件只会被编译一次

#include <memory>
// 包含标准库中的内存管理相关头文件
#include <string>
// 包含标准库中的字符串处理相关头文件

namespace caffe2::serialize {
// 命名空间 caffe2::serialize，用于序列化相关功能

class PyTorchStreamReader;
// 声明 PyTorchStreamReader 类

}

namespace torch::jit {
// 命名空间 torch::jit，用于 JIT 编译相关功能

struct Source;
// 声明结构体 Source

// 将类类型的限定符名转换为对应的源文件路径，用于存档
//
// Qualifier 是如 foo.bar.baz 的限定符名
// 返回: libs/foo/bar/baz.py
std::string qualifierToArchivePath(
    const std::string& qualifier,
    const std::string& export_prefix);
// 声明函数 qualifierToArchivePath，用于将限定符名转换为存档路径

// 在存档中根据限定符名从读取器中查找源文件
std::shared_ptr<Source> findSourceInArchiveFromQualifier(
    caffe2::serialize::PyTorchStreamReader& reader,
    const std::string& export_prefix,
    const std::string& qualifier);
// 声明函数 findSourceInArchiveFromQualifier，用于根据限定符名在存档中查找源文件

} // namespace torch::jit
// 结束命名空间 torch::jit
```