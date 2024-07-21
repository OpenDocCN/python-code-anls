# `.\pytorch\torch\csrc\jit\serialization\pickle.cpp`

```
// 包含 Torch 序列化和反序列化相关的头文件

#include <torch/csrc/jit/serialization/pickle.h>

// 包含 Torch 的 IValue 类型定义
#include <ATen/core/ivalue.h>

// 包含用于序列化的辅助函数
#include <caffe2/serialize/inline_container.h>

// 包含 Torch 的导出相关定义
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/serialization/export.h>

// 包含 Torch 的导入相关定义
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/import_read.h>

// Torch 的命名空间
namespace torch::jit {

// 序列化函数，将 IValue 对象序列化成字节流
void pickle(
    std::function<void(const char* data_start, size_t data_len)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  // 创建 Pickler 对象，用于序列化
  Pickler pickler(std::move(writer), tensor_table, nullptr, nullptr);
  pickler.protocol();
  // 将 IValue 对象压入 Pickler 中进行序列化
  pickler.pushIValue(ivalue);
  pickler.stop();
}

// 序列化函数，将 IValue 对象序列化成字节数组
std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  // 创建一个空的字符向量用于存储序列化后的数据
  std::vector<char> data;

  // 调用上面的 pickle 函数，将数据追加到 data 向量中
  pickle(
      [&](const char* bytes, size_t len) {
        data.insert(data.end(), bytes, bytes + len);
      },
      ivalue,
      tensor_table);

  return data;
}

// 保存序列化数据到容器中，适配移动设备的限制
std::vector<char> pickle_save(const at::IValue& ivalue) {
#ifndef C10_MOBILE
  // Pickle the IValue into an array of bytes
  std::vector<char> pickle_data;
  Pickler pickler([&](const char* buf, size_t size) {
    pickle_data.insert(pickle_data.end(), buf, buf + size);
  });
  pickler.protocol();
  pickler.pushIValue(ivalue);
  pickler.stop();

  // 创建一个空的容器数据向量
  std::vector<char> container_data;
  container_data.reserve(pickle_data.size());

  // 使用 PyTorchStreamWriter 将数据写入容器中
  caffe2::serialize::PyTorchStreamWriter writer(
      [&](const void* void_bytes, size_t len) {
        const char* bytes = reinterpret_cast<const char*>(void_bytes);
        container_data.insert(container_data.end(), bytes, bytes + len);
        return len;
      });

  // 写入生成的字节数据和相关张量数据到文件
  writeArchiveAndTensors(
      "data",
      pickle_data.data(),
      pickle_data.size(),
      pickler.tensorData(),
      writer);
  return container_data;
#else
  // 移动设备上不支持此操作，抛出错误
  AT_ERROR(
      "pickle_save not supported on mobile "
      "(see https://github.com/pytorch/pytorch/pull/30108)");
#endif
}

#ifndef C10_MOBILE
// 从 VectorReader 中读取数据到缓冲区中
size_t VectorReader::read(uint64_t pos, void* buf, size_t n, const char* what)
    const {
  std::copy(
      data_.data() + pos, data_.data() + pos + n, reinterpret_cast<char*>(buf));
  return n;
}
#endif

// 反序列化函数，从字节数组加载 IValue 对象
IValue pickle_load(const std::vector<char>& data) {
  // 读取 pickle 数据
#ifndef C10_MOBILE
  caffe2::serialize::PyTorchStreamReader reader(
      std::make_unique<VectorReader>(data));

  // 从读取的数据中恢复对象和张量
  return readArchiveAndTensors(
      "data",
      /*pickle_prefix=*/"",
      /*tensor_prefix=*/"",
      /*type_resolver=*/c10::nullopt,
      /*obj_loader=*/c10::nullopt,
      /*device=*/c10::nullopt,
      reader);
#else
  // 移动设备上不支持此操作，抛出错误
  AT_ERROR(
      "pickle_load not supported on mobile "
      "(see https://github.com/pytorch/pytorch/pull/30108)");
#endif
};

// 反序列化函数的声明，用于加载序列化后的数据
IValue unpickle(
    // 创建一个 Unpickler 对象，使用给定的 reader 函数、type_resolver 函数对象、tensor_table 引用、
    // obj_loader 对象和 type_parser 函数指针作为参数
    Unpickler unpickler(
        std::move(reader),
        std::move(type_resolver),
        tensor_table,
        std::move(obj_loader),
        type_parser);
    // 调用 Unpickler 对象的 parse_ivalue() 方法解析数据并返回结果
    return unpickler.parse_ivalue();
} // 结束命名空间 torch::jit

// 从二进制数据中反序列化对象
IValue unpickle(
    const char* data,
    size_t size,
    TypeResolver type_resolver,
    c10::ArrayRef<at::Tensor> tensor_table,
    c10::TypePtr (*type_parser)(const std::string&)) {
  // 调用另一个版本的 unpickle 函数，传递给它参数 data, size, nullptr, std::move(type_resolver), tensor_table, type_parser
  return unpickle(
      data, size, nullptr, std::move(type_resolver), tensor_table, type_parser);
}

// 从二进制数据中反序列化对象
IValue unpickle(
    const char* data,
    size_t size,
    ObjLoader obj_loader,
    TypeResolver type_resolver,
    c10::ArrayRef<at::Tensor> tensor_table,
    c10::TypePtr (*type_parser)(const std::string&)) {
  // 初始化已读取字节数
  size_t bytes_read = 0;
  // 调用另一个版本的 unpickle 函数，使用 lambda 函数作为读取数据的方式
  return unpickle(
      [&](char* buffer, size_t len) -> size_t {
        // 如果已读取的字节数大于等于总大小，返回 0 表示无数据可读
        if (bytes_read >= size) {
          return 0;
        }
        // 计算实际要读取的长度，不超过剩余未读取的数据长度
        len = std::min(size - bytes_read, len);
        // 将数据从源 data 复制到目标 buffer
        const char* start = data + bytes_read;
        std::memcpy(buffer, start, len);
        bytes_read += len;
        return len;
      },
      std::move(type_resolver),
      tensor_table,
      type_parser,
      std::move(obj_loader));
}
```