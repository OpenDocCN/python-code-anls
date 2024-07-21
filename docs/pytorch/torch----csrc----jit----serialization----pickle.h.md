# `.\pytorch\torch\csrc\jit\serialization\pickle.h`

```
#pragma once
// 预处理指令：确保此头文件只包含一次

#include <ATen/core/ivalue.h>
// 包含 ATen 库中的 IValue 类定义

#include <c10/util/ArrayRef.h>
// 包含 c10 库中的 ArrayRef 类定义

#include <caffe2/serialize/inline_container.h>
// 包含 caffe2 序列化库中的 inline_container 头文件

#include <torch/csrc/Export.h>
// 包含 Torch 库中的导出定义

#include <torch/csrc/jit/serialization/pickler.h>
// 包含 Torch JIT 序列化模块中的 pickler 头文件

#include <torch/csrc/jit/serialization/unpickler.h>
// 包含 Torch JIT 序列化模块中的 unpickler 头文件

namespace torch::jit {

/// Pickle an IValue by calling a function to handle writing the data.
///
/// `writer` is a function that takes in a pointer to a chunk of memory and its
/// size and consumes it.
///
/// See `jit::pickle` for more details.
TORCH_API void pickle(
    std::function<void(const char* data_start, size_t data_len)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table = nullptr);
// 定义函数声明：使用指定的写入函数将给定的 IValue 对象进行序列化（pickle）处理

/// Save a `torch::IValue` in a format compatible with Python's `pickle` module
///
/// If present, `tensor_table` is a pointer to a table in which tensors that
/// are contained within `ivalue` are stored, and the bytes returned by the
/// pickler will only include references to these tensors in the table. This can
/// be used to keep the binary blob size small.
/// If not provided, tensors are stored in the same byte stream as the pickle
/// data, similar to `torch.save()` in eager Python.
///
/// Pickled values can be loaded in Python and C++:
/// \rst
/// .. code-block:: cpp
///
///  torch::IValue float_value(2.3);
///
///  // TODO: when tensors are stored in the pickle, delete this
///  std::vector<at::Tensor> tensor_table;
///  auto data = torch::jit::pickle(float_value, &tensor_table);
///
///  std::vector<torch::IValue> ivalues =
///      torch::jit::unpickle(data.data(), data.size());
///
/// .. code-block:: python
///
///   values = torch.load('data.pkl')
///   print(values)
///
/// \endrst
TORCH_API std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table = nullptr);
// 定义函数声明：将给定的 IValue 对象以兼容 Python pickle 模块格式保存为字节流

/// Save a `torch::IValue` in a format that can be loaded by both
/// `torch::pickle_load` in C++ and `torch.load` in Python.
TORCH_API std::vector<char> pickle_save(const IValue& ivalue);
// 定义函数声明：将给定的 IValue 对象保存为既能在 C++ 中加载的格式，也能在 Python 中通过 torch.load 加载的格式

/// Deserialize a `torch::IValue` from bytes produced by either
/// `torch::pickle_save` in C++ or `torch.save` in Python
TORCH_API IValue pickle_load(const std::vector<char>& data);
// 定义函数声明：从由 torch::pickle_save（在 C++ 中）或 torch.save（在 Python 中）生成的字节流中反序列化为 IValue 对象

/// `reader` is a function that takes in a size to read from some pickled
/// binary. `reader` should remember where it last read, and return
/// the number of bytes read.
/// See `torch::pickle` for details.
/// type_resolver is used to resolve any JIT type based on type str
TORCH_API IValue unpickle(
    std::function<size_t(char*, size_t)> reader,
    TypeResolver type_resolver,
    c10::ArrayRef<at::Tensor> tensor_table,
    c10::TypePtr (*type_parser)(const std::string&) =
        Unpickler::defaultTypeParser,
    ObjLoader obj_loader = nullptr);
// 定义函数声明：使用指定的读取函数将给定的 pickled 二进制数据解码为 torch::IValue 对象

/// Decode a chunk of memory containing pickled data into its `torch::IValue`s.
///
/// If any `torch::IValue`s in the pickled data are `Object`s, then a
/// `class_resolver` function must be provided.
///
/// See `torch::pickle` for details.
/// Function declaration for unpickling data into torch::IValue.
///
/// This function decodes a chunk of memory (`data`) of specified `size` into
/// a torch::IValue object. It supports various configurations:
/// - `type_resolver`: A function resolving custom types in the unpickled data.
/// - `tensor_table`: An optional reference to an array of tensors used during unpickling.
/// - `type_parser`: A function parsing type information from strings.
///
/// It requires an `ObjLoader` when dealing with `Object` types in the pickled data.
TORCH_API IValue unpickle(
    const char* data,
    size_t size,
    TypeResolver type_resolver = nullptr,
    c10::ArrayRef<at::Tensor> tensor_table = {},
    c10::TypePtr (*type_parser)(const std::string&) =
        Unpickler::defaultTypeParser);

/// Function declaration for unpickling data into torch::IValue with object loading.
///
/// This function is an overloaded version of `unpickle` that additionally requires
/// an `ObjLoader` (`obj_loader`) to handle `Object` types in the pickled data.
///
/// It supports the same parameters as the base `unpickle` function:
/// - `type_resolver`: A function resolving custom types.
/// - `tensor_table`: An optional array reference of tensors.
/// - `type_parser`: A function parsing type information from strings.
TORCH_API IValue unpickle(
    const char* data,
    size_t size,
    ObjLoader obj_loader,
    TypeResolver type_resolver = nullptr,
    c10::ArrayRef<at::Tensor> tensor_table = {},
    c10::TypePtr (*type_parser)(const std::string&) =
        Unpickler::defaultTypeParser);

#ifndef C10_MOBILE
/// A custom reader class implementing caffe2::serialize::ReadAdapterInterface.
///
/// This class `VectorReader` reads data from a vector of characters (`data_`).
/// It provides functionality to get the size of the data and to read data
/// from specified positions (`pos`) into a buffer (`buf`).
class VectorReader : public caffe2::serialize::ReadAdapterInterface {
 public:
  /// Constructor initializing the `VectorReader` with a given vector of data.
  VectorReader(std::vector<char> data) : data_(std::move(data)) {}

  /// Returns the size of the stored data.
  size_t size() const override {
    return data_.size();
  }

  /// Reads `n` bytes of data from `pos` in the stored data into `buf`.
  /// `what` parameter describes the operation for logging or debugging.
  size_t read(uint64_t pos, void* buf, size_t n, const char* what)
      const override;

 private:
  std::vector<char> data_;  ///< Vector containing the data to be read.
};
#endif
```