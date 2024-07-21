# `.\pytorch\torch\csrc\jit\serialization\import_read.h`

```py
#pragma once


// 使用预处理指令#pragma once，确保头文件只被编译一次



#include <torch/csrc/jit/serialization/unpickler.h>
#include <memory>


// 引入 Torch 库中的反序列化模块头文件和标准 C++ 库中的内存管理模块



namespace caffe2::serialize {
class PyTorchStreamReader;
} // namespace caffe2::serialize


// 定义命名空间 caffe2::serialize，包含 PyTorchStreamReader 类的前向声明



namespace torch::jit {


// 定义命名空间 torch::jit，包含接下来的函数和类型定义



TORCH_API IValue readArchiveAndTensors(
    const std::string& archive_name,
    const std::string& pickle_prefix,
    const std::string& tensor_prefix,
    std::optional<TypeResolver> type_resolver,
    std::optional<ObjLoader> obj_loader,
    std::optional<at::Device> device,
    caffe2::serialize::PyTorchStreamReader& stream_reader,
    c10::TypePtr (*type_parser)(const std::string&) =
        Unpickler::defaultTypeParser,
    std::shared_ptr<DeserializationStorageContext> storage_context = nullptr);


// 函数声明 readArchiveAndTensors，用于从存档中读取数据和张量
// 参数说明：
// - archive_name: 存档名称
// - pickle_prefix: pickle 文件前缀
// - tensor_prefix: 张量文件前缀
// - type_resolver: 可选的类型解析器
// - obj_loader: 可选的对象加载器
// - device: 可选的设备类型
// - stream_reader: PyTorchStreamReader 对象的引用
// - type_parser: 类型解析器函数指针，默认为 Unpickler::defaultTypeParser
// - storage_context: 可选的反序列化存储上下文的共享指针，默认为空指针



bool check_zip_file(
    const std::shared_ptr<caffe2::serialize::ReadAdapterInterface>& rai);


// 函数声明 check_zip_file，用于检查是否为 ZIP 文件
// 参数说明：
// - rai: 共享指针，指向 caffe2::serialize::ReadAdapterInterface 接口的实例



} // namespace torch::jit


// 结束命名空间 torch::jit
```