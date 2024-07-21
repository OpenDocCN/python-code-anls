# `.\pytorch\torch\csrc\jit\serialization\import.h`

```py
#pragma once

#include <ATen/core/ivalue.h>  // 引入 ATen 库中的 IValue 类
#include <caffe2/serialize/inline_container.h>  // 引入 caffe2 序列化库中的 inline_container.h
#include <torch/csrc/jit/api/module.h>  // 引入 Torch JIT 模块的 API 头文件
#include <torch/csrc/jit/ir/ir.h>  // 引入 Torch JIT 中间表示的 IR 头文件

#include <istream>  // 引入标准输入流库

namespace caffe2::serialize {
class ReadAdapterInterface;  // 定义 caffe2 序列化库中的 ReadAdapterInterface 类
} // namespace caffe2::serialize

namespace torch::jit {

class DeserializationStorageContext;  // 定义 Torch JIT 中的 DeserializationStorageContext 类

/// 从文件中导入序列化的 IR 模块
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,  // 编译单元的共享指针
    const std::string& filename,  // 文件名
    std::optional<c10::Device> device = c10::nullopt,  // 可选的设备参数
    bool load_debug_files = true);  // 是否加载调试文件

/// 从输入流中导入序列化的 IR 模块
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,  // 编译单元的共享指针
    std::istream& in,  // 输入流
    std::optional<c10::Device> device = c10::nullopt,  // 可选的设备参数
    bool load_debug_files = true);  // 是否加载调试文件

/// 从 ReadAdapterInterface 对象中导入序列化的 IR 模块
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,  // 编译单元的共享指针
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,  // caffe2 序列化库中的 ReadAdapterInterface 对象的唯一指针
    std::optional<c10::Device> device = c10::nullopt,  // 可选的设备参数
    bool load_debug_files = true);  // 是否加载调试文件

/// 从文件中导入序列化的 IR 模块，同时包含额外的文件映射
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,  // 编译单元的共享指针
    const std::string& filename,  // 文件名
    std::optional<c10::Device> device,  // 可选的设备参数
    ExtraFilesMap& extra_files,  // 额外文件映射
    bool load_debug_files = true,  // 是否加载调试文件
    bool restore_shapes = false);  // 是否恢复形状

// 从 torch.Package 的统一序列化格式中读取
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,  // 编译单元的共享指针
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,  // caffe2 序列化库中的 PyTorchStreamReader 对象的共享指针
    std::shared_ptr<torch::jit::DeserializationStorageContext> storage_context,  // Torch JIT 中的 DeserializationStorageContext 对象的共享指针
    std::optional<at::Device> device,  // 可选的设备参数
    const std::string& ts_id /* torchscript identifier inside package */);  // torchscript 标识符

/// 从输入流中导入序列化的 IR 模块，同时包含额外的文件映射
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,  // 编译单元的共享指针
    std::istream& in,  // 输入流
    std::optional<c10::Device> device,  // 可选的设备参数
    ExtraFilesMap& extra_files,  // 额外文件映射
    bool load_debug_files = true,  // 是否加载调试文件
    bool restore_shapes = false);  // 是否恢复形状

/// 从 ReadAdapterInterface 对象中导入序列化的 IR 模块，同时包含额外的文件映射
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,  // 编译单元的共享指针
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,  // caffe2 序列化库中的 ReadAdapterInterface 对象的唯一指针
    std::optional<c10::Device> device,  // 可选的设备参数
    ExtraFilesMap& extra_files,  // 额外文件映射
    bool load_debug_files = true);  // 是否加载调试文件

/// 从 ReadAdapterInterface 对象中导入序列化的 IR 模块，同时包含额外的文件映射
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,  // 编译单元的共享指针
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,  // caffe2 序列化库中的 ReadAdapterInterface 对象的共享指针
    std::optional<c10::Device> device,  // 可选的设备参数
    ExtraFilesMap& extra_files,  // 额外文件映射
    bool load_debug_files = true);  // 是否加载调试文件

/// 从给定的输入流中加载序列化的 `Module`
///
/// 输入流中必须包含通过 C++ 中的 `torch::jit::ExportModule` 导出的序列化 `Module`
TORCH_API Module load(
    std::istream& in,  // 输入流
    std::optional<c10::Device> device = c10::nullopt,  // 可选的设备参数
    bool load_debug_files = true);  // 是否加载调试文件

/// 从给定的输入流中加载序列化的 `Module`，同时包含额外的文件映射
TORCH_API Module load(
    std::istream& in,  // 输入流
    std::optional<c10::Device> device,  // 可选的设备参数
    ExtraFilesMap& extra_files,  // 额外文件映射
    bool load_debug_files = true);  // 是否加载调试文件

/// 从给定的文件名中加载序列化的 `Module`
///
/// 文件名指定的文件必须包含
/// 加载一个序列化的 `Module`，可以通过 Python 的 `ScriptModule.save()` 或 C++ 的 `torch::jit::ExportModule` 导出。
/// 使用给定的文件名 `filename` 加载模块，可选的指定设备 `device`，默认加载调试文件。
TORCH_API Module load(
    const std::string& filename,
    std::optional<c10::Device> device = c10::nullopt,
    bool load_debug_files = true);

/// 加载一个序列化的 `Module`，可以通过 Python 的 `ScriptModule.save()` 或 C++ 的 `torch::jit::ExportModule` 导出。
/// 使用给定的文件名 `filename` 加载模块，可选的指定设备 `device`，额外的文件映射 `extra_files`，默认加载调试文件。
TORCH_API Module load(
    const std::string& filename,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// 从给定的共享指针 `rai` 中加载序列化的 `Module`。
/// 读取适配器 `rai` 必须包含一个序列化的 `Module`，可以通过 Python 的 `ScriptModule.save()` 或 C++ 的 `torch::jit::ExportModule` 导出。
TORCH_API Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device = c10::nullopt,
    bool load_debug_files = true);

/// 从给定的共享指针 `rai` 中加载序列化的 `Module`。
/// 读取适配器 `rai` 必须包含一个序列化的 `Module`，可以通过 Python 的 `ScriptModule.save()` 或 C++ 的 `torch::jit::ExportModule` 导出。
/// 可选的指定设备 `device`，额外的文件映射 `extra_files`，默认加载调试文件。
TORCH_API Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// 从给定的 `ivalue` 和 `source` 加载 JIT 模块。
/// `ivalue` 包含模块的状态信息，`source` 包含额外的文件映射，`constants` 是模块的常量。
TORCH_API Module jitModuleFromSourceAndConstants(
    const IValue& ivalue,
    const ExtraFilesMap& source,
    const std::vector<IValue>& constants,
    int32_t version);

/// 解析并初始化 JIT 模块，从给定的 `data` 和 `size`。
/// `data` 是模块的数据，`size` 是数据的大小，`extra_files` 是额外的文件映射，可选的指定设备 `device`。
TORCH_API Module parse_and_initialize_jit_module(
    const std::shared_ptr<char>& data,
    size_t size,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = c10::nullopt);

/// 从给定的文件名 `filename` 加载 JIT 模块。
/// `extra_files` 是额外的文件映射，可选的指定设备 `device`。
TORCH_API Module load_jit_module_from_file(
    const std::string& filename,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = c10::nullopt);

/// 从给定的输入流 `in` 中加载 JIT 模块。
/// `extra_files` 是额外的文件映射，可选的指定设备 `device`。
TORCH_API Module load_jit_module_from_stream(
    std::istream& in,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = c10::nullopt);

/// 解析并初始化 JIT 模块，从给定的 `data` 和 `size`。
/// `data` 是模块的数据，`size` 是数据的大小，`extra_files` 是额外的文件映射，指定设备 `device`。
TORCH_API Module parse_and_initialize_jit_module(
    const std::shared_ptr<char>& data,
    size_t size,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device);

/// 对象加载函数，用于从 `input` 创建 `type` 类型的对象。
TORCH_API c10::intrusive_ptr<c10::ivalue::Object> ObjLoaderFunc(
    const at::StrongTypePtr& type,
    IValue input);

} // namespace torch::jit
```