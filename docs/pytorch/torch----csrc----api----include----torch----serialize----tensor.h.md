# `.\pytorch\torch\csrc\api\include\torch\serialize\tensor.h`

```py
#pragma once

该指令告诉编译器只包含当前头文件一次，避免重复包含。


#include <torch/serialize/archive.h>
#include <torch/types.h>

引入了两个头文件，分别是用于序列化存档的 `archive.h` 和 `types.h`，这些头文件包含了与张量序列化和反序列化相关的类型和函数声明。


namespace torch {

定义了命名空间 `torch`，所有的函数和类都在这个命名空间下。


inline serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Tensor& tensor) {

定义了一个内联函数 `operator<<`，该函数将 `const Tensor&` 类型的 `tensor` 写入到 `serialize::OutputArchive&` 类型的 `archive` 中。


  archive.write("0", tensor);

使用 `archive` 的 `write` 方法将名为 `"0"` 的键和 `tensor` 写入到存档中。


  return archive;
}

返回 `archive` 本身，以便支持链式操作。


inline serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Tensor& tensor) {

定义了一个内联函数 `operator>>`，该函数从 `serialize::InputArchive&` 类型的 `archive` 中读取数据到 `Tensor&` 类型的 `tensor` 中。


  archive.read("0", tensor);

使用 `archive` 的 `read` 方法从存档中读取键为 `"0"` 的数据，并将其写入到 `tensor` 中。


  return archive;
}

返回 `archive` 本身，以便支持链式操作。


} // namespace torch

命名空间 `torch` 的结束。
```