# `.\pytorch\torch\csrc\jit\mobile\import.h`

```py
#pragma once
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/parse_operators.h>

#include <istream>
#include <memory>

#include <caffe2/serialize/file_adapter.h>

// 命名空间 torch::jit 中的相关实用工具和数据结构的使用声明
namespace torch {
namespace jit {

// 使用 caffe2 序列化库中的文件适配器和流适配器
using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::ReadAdapterInterface;

// 用于存储额外文件路径与内容的映射关系
using ExtraFilesMap = std::unordered_map<std::string, std::string>;

// 定义常量，用于指定归档文件中特定内容的键名
constexpr const char* kArchiveNameBytecode = "bytecode";
constexpr const char* kArchiveNameConstants = "constants";
constexpr const char* kArchiveNameVersion = "version";

// 以下是一系列函数，用于将序列化的移动模块加载到 mobile::Module 对象中

// 从输入流中加载移动模块
TORCH_API mobile::Module _load_for_mobile(
    std::istream& in,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_file,
    uint64_t module_load_options = kDefaultMobileLoadOptions);

// 从文件名加载移动模块
TORCH_API mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files);

// 从自定义的读取适配器接口加载移动模块
TORCH_API mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options = kDefaultMobileLoadOptions);

// 从文件名加载移动模块，可选加载选项
TORCH_API mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options);

// 从输入流中加载移动模块，可选设备
TORCH_API mobile::Module _load_for_mobile(
    std::istream& in,
    std::optional<at::Device> device = c10::nullopt);

// 从文件名加载移动模块，可选设备
TORCH_API mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device = c10::nullopt);

// 从自定义的读取适配器接口加载移动模块，可选设备
TORCH_API mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device = c10::nullopt);

/**
 * 仅加载“extra/”文件夹中在映射（extra_files）中指定的文件内容。
 * 将相应的值填充到映射中，并在提取完额外文件后停止加载。
 *
 * 这个 API 的目的是能够在 Linux CPU 设备上加载 GPU 模型，
 * 只提取额外文件以便可以检查生成 .ptl 归档时添加的元数据。
 */
void _load_extra_only_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files);

// 当前在 mobile/import.cpp 和 model_compatibility.cpp 中使用。
// 在 model_compatibility.cpp 开始使用简化版本的 type_resolver 和 obj_loader 后，应该移除。
// 解析移动模块中类型名称的辅助函数
at::TypePtr resolveTypeNameMobile(
    const c10::QualifiedName& qn,
    std::shared_ptr<CompilationUnit> compilation_unit);

// 移动模块类型解析器
c10::StrongTypePtr typeResolverMobile(
    const c10::QualifiedName& qn,
    const std::shared_ptr<CompilationUnit>& compilation_unit);

// 移动模块对象加载器
c10::intrusive_ptr<c10::ivalue::Object> objLoaderMobile(
    const std::string& filename,
    const c10::QualifiedName& qn,
    const std::shared_ptr<CompilationUnit>& compilation_unit);

} // namespace jit
} // namespace torch
    // 声明一个常量引用，类型为 at::StrongTypePtr，表示一个强类型指针
    const at::StrongTypePtr& type,
    // 声明一个常量引用，类型为 at::IValue，表示一个输入参数的常量引用
    const at::IValue& input,
    // 声明一个非常量引用，类型为 mobile::CompilationUnit，表示一个移动编译单元的引用
    mobile::CompilationUnit& mobile_compilation_unit);
    // 函数声明结束
// 给定一个流读取器 `stream_reader`，检查其中是否存在 `bytecode` 归档中的张量
bool isTensorInBytecodeArchive(
    caffe2::serialize::PyTorchStreamReader& stream_reader);

namespace mobile {

/**
 * 给定一个 torch::jit::mobile::Module，返回该移动模块中使用的操作符名称集合
 * (包括重载名称)。此方法遍历指定模型(module)中所有方法的字节码，
 * 提取所有的根操作符名称。根操作符是模型直接调用的操作符
 * (与非根操作符相对，后者可能被根操作符间接调用)。
 *
 */
TORCH_API std::set<std::string> _export_operator_list(
    torch::jit::mobile::Module& module);

} // namespace mobile

} // namespace jit
} // namespace torch
```