# `.\pytorch\torch\csrc\jit\serialization\export.h`

```
#pragma once

// 包含头文件：caffe2 序列化中的内联容器
#include <caffe2/serialize/inline_container.h>
// 包含头文件：PyTorch JIT 模块接口
#include <torch/csrc/jit/api/module.h>
// 包含头文件：PyTorch JIT IR（Intermediate Representation，中间表示）定义
#include <torch/csrc/jit/ir/ir.h>
// 包含头文件：PyTorch JIT 序列化导出字节码
#include <torch/csrc/jit/serialization/export_bytecode.h>
// 包含头文件：PyTorch JIT 序列化 FlatBuffer 序列化器
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
// 包含头文件：PyTorch JIT 序列化 Pickler（泡菜器）
#include <torch/csrc/jit/serialization/pickler.h>
// 包含头文件：PyTorch JIT 序列化 Python 打印
#include <torch/csrc/jit/serialization/python_print.h>
// 包含头文件：PyTorch JIT 序列化存储上下文
#include <torch/csrc/jit/serialization/storage_context.h>
// 包含头文件：PyTorch JIT 序列化类型名称唯一化器
#include <torch/csrc/jit/serialization/type_name_uniquer.h>
// 包含头文件：ONNX 定义
#include <torch/csrc/onnx/onnx.h>
// 包含头文件：输出流
#include <ostream>

// 命名空间声明：ONNX_NAMESPACE，包含 ONNX 模型协议
namespace ONNX_NAMESPACE {
    class ModelProto;
}

// 命名空间声明：torch::jit，包含 PyTorch JIT 模块
namespace torch::jit {

// 使用 typedef 定义 RawDataExportMap，用于记录需要导出的参数
// 当 defer_weight_export 为 true 时，返回的 map 包含 {外部引用名称} -> {待导出的 at::Tensor}
// 调用者有责任适当地导出这些内容。
using RawDataExportMap = std::unordered_map<std::string, at::Tensor>;

// 使用 typedef 定义 SymbolDimMap，用于映射 ShapeSymbol 到字符串的映射
using SymbolDimMap = std::map<c10::ShapeSymbol, std::string>;

// 使用 typedef 定义 DimSymbolMap，用于映射字符串到 ShapeSymbol 的映射
using DimSymbolMap = std::map<std::string, c10::ShapeSymbol>;

// 使用 typedef 定义 NodeNameMap，用于映射节点到名称的映射
using NodeNameMap = std::unordered_map<const Node*, std::string>;

// 使用 typedef 定义 NodeAttrNameMap，用于映射节点到属性名称的映射
// 用于模块化导出安置函数和节点属性。
using NodeAttrNameMap = std::
    unordered_map<const Node*, std::unordered_map<std::string, std::string>>;

// 函数声明：导出 PyTorch 图为 ONNX 模型
TORCH_API std::tuple<
    std::shared_ptr<::ONNX_NAMESPACE::ModelProto>,   // 返回 ONNX 模型协议的共享指针
    RawDataExportMap,   // 返回原始数据导出映射
    SymbolDimMap,   // 返回符号维度映射
    bool,   // 返回是否延迟权重导出的标志
    NodeNameMap   // 返回节点名称映射
>
export_onnx(
    const std::shared_ptr<Graph>& graph,   // 输入参数：PyTorch 图的共享指针
    const std::map<std::string, at::Tensor>& initializers,   // 输入参数：初始化器映射
    int64_t onnx_opset_version,   // 输入参数：ONNX 操作集版本
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,   // 输入参数：动态轴映射
    bool defer_weight_export = false,   // 输入参数：是否延迟权重导出，默认为 false
    ::torch::onnx::OperatorExportTypes operator_export_type =
        ::torch::onnx::OperatorExportTypes::ONNX,   // 输入参数：操作符导出类型，默认为 ONNX
    bool strip_doc_string = true,   // 输入参数：是否去除文档字符串，默认为 true
    bool keep_initializers_as_inputs = true,   // 输入参数：是否保留初始化器作为输入，默认为 true
    const std::map<std::string, int>& custom_opsets = {},   // 输入参数：自定义操作集
    bool add_node_names = true,   // 输入参数：是否添加节点名称，默认为 true
    bool use_external_data_format = false,   // 输入参数：是否使用外部数据格式，默认为 false
    const std::string& onnx_file_path = std::string(),   // 输入参数：ONNX 文件路径，默认为空字符串
    const NodeAttrNameMap& node_attr_to_name = {});   // 输入参数：节点属性到名称的映射

// 函数声明：将 ONNX 模型协议序列化为字符串
TORCH_API std::string serialize_model_proto_to_string(
    const std::shared_ptr<::ONNX_NAMESPACE::ModelProto>& model_proto);   // 输入参数：ONNX 模型协议的共享指针

// 函数声明：检查 ONNX 协议的有效性
TORCH_API void check_onnx_proto(const std::string& proto_string);   // 输入参数：ONNX 协议字符串

// 空的命名空间声明，用于统一旧版和统一格式 TorchScript 序列化
};

// 用于测试目的的函数声明：将 PyTorch 图和初始化器映射漂亮地打印为 ONNX
TORCH_API std::string pretty_print_onnx(
    const std::shared_ptr<Graph>& graph,   // 输入参数：PyTorch 图的共享指针
    const std::map<std::string, at::Tensor>& initializers,   // 输入参数：初始化器映射
    int64_t onnx_opset_version,   // 输入参数：ONNX 操作集版本
    bool defer_weight_export,   // 输入参数：是否延迟权重导出
    // 定义枚举类型变量 operator_export_type，并初始化为 ONNX，指定导出类型为 ONNX
    ::torch::onnx::OperatorExportTypes operator_export_type =
        ::torch::onnx::OperatorExportTypes::ONNX,
    // 布尔变量 google_printer，控制是否启用 Google 打印器，默认为 false
    bool google_printer = false,
    // 布尔变量 keep_initializers_as_inputs，控制是否将初始值保留为输入，默认为 true
    bool keep_initializers_as_inputs = true,
    // 常量引用，使用 std::map<std::string, int> 类型的 custom_opsets 参数，用于自定义操作集合
    const std::map<std::string, int>& custom_opsets = {},
    // 布尔变量 add_node_names，控制是否添加节点名称，默认为 true
    bool add_node_names = true);
// 导出模块到输出流，支持指定附加文件和多种格式选项
TORCH_API void ExportModule(
    const Module& module,  // 要导出的模块对象
    std::ostream& out,     // 输出流，将导出的内容写入到这个流中
    const ExtraFilesMap& metadata = ExtraFilesMap(),  // 附加的元数据文件映射，默认为空
    bool bytecode_format = false,  // 是否使用字节码格式，默认为false
    bool save_mobile_debug_info = false,  // 是否保存移动端调试信息，默认为false
    bool use_flatbuffer = false);  // 是否使用FlatBuffer格式，默认为false

// 导出模块到指定文件名，支持指定附加文件和多种格式选项
TORCH_API void ExportModule(
    const Module& module,  // 要导出的模块对象
    const std::string& filename,  // 输出的文件名，将导出的内容写入到这个文件中
    const ExtraFilesMap& metadata = ExtraFilesMap(),  // 附加的元数据文件映射，默认为空
    bool bytecode_format = false,  // 是否使用字节码格式，默认为false
    bool save_mobile_debug_info = false,  // 是否保存移动端调试信息，默认为false
    bool use_flatbuffer = false);  // 是否使用FlatBuffer格式，默认为false

// 导出模块到用户指定的写入函数中，支持指定附加文件和多种格式选项
TORCH_API void ExportModule(
    const Module& module,  // 要导出的模块对象
    const std::function<size_t(const void*, size_t)>& writer_func,  // 用户定义的写入函数
    const ExtraFilesMap& metadata = ExtraFilesMap(),  // 附加的元数据文件映射，默认为空
    bool bytecode_format = false,  // 是否使用字节码格式，默认为false
    bool save_mobile_debug_info = false,  // 是否保存移动端调试信息，默认为false
    bool use_flatbuffer = false);  // 是否使用FlatBuffer格式，默认为false

// 写入一个pickle归档及其中引用的张量数据到输出流中
TORCH_API void writeArchiveAndTensors(
    const std::string& archive_name,  // 归档文件名
    const char* pickle_bytes,  // pickle归档的字节流数据
    size_t size,  // pickle归档数据的大小
    const std::vector<at::Tensor>& tensors,  // 要写入的张量向量
    caffe2::serialize::PyTorchStreamWriter& out);  // 输出流对象

// 设置导出模块时的额外文件钩子函数，用于根据环境生成元数据文件
using ExportModuleExtraFilesHook = std::function<ExtraFilesMap(const Module&)>;
TORCH_API void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook);

/**
 * 导出一个脚本模块的新字节码，并返回基于当前代码库的操作列表，用于LiteScriptModule。
 * 如果已经有LiteScriptModule并想获取当前的操作列表，请调用_export_operator_list。
 */
TORCH_API std::vector<std::string> export_opnames(const Module& m);

// 控制JIT如何生成字节码的配置类
struct TORCH_API BytecodeEmitMode {
  static bool is_default_value_for_unspecified_arg_enabled();  // 是否启用未指定参数默认值的生成
  static void set_default_value_for_unspecified_arg_enabled(bool enabled);  // 设置是否启用未指定参数默认值的生成

  static bool is_default_args_before_out_args_enabled();  // 是否启用默认参数在输出参数之前的生成
  static void set_default_args_before_out_args_enabled(bool enabled);  // 设置是否启用默认参数在输出参数之前的生成

  static bool is_emit_promoted_ops_enabled();  // 是否启用推广操作的生成
  static void set_default_emit_promoted_ops_enabled(bool enabled);  // 设置是否启用推广操作的生成
};

// 控制JIT字节码生成的RAII守卫，用于控制输入参数的生成方式
// default_value_for_unspecified_arg:
// true：生成默认参数值的指令（如LOADC）。
// false：不生成默认参数值的指令，而是从操作符模式中获取。
// default_args_before_out_args：
// true：反向兼容支持允许输出参数和默认参数的操作符，指定的参数数量将反序列化为（#所有参数 - #默认参数）。
// false：指定的参数数量将反序列化为（#所有参数）。
// 定义一个结构体 BytecodeEmitModeGuard，用于管理字节码生成模式的设置
struct TORCH_API BytecodeEmitModeGuard {
  // 构造函数，初始化字节码生成模式的各个选项，并保存当前的设置
  BytecodeEmitModeGuard(
      bool enable_default_value_for_unspecified_arg,
      bool enable_default_args_before_out_args,
      bool enable_emit_promoted_ops)
      : prev_default_value_for_unspecified_arg_mode(
            BytecodeEmitMode::is_default_value_for_unspecified_arg_enabled()),  // 保存当前默认值模式
        prev_default_args_before_out_args(
            BytecodeEmitMode::is_default_args_before_out_args_enabled()),  // 保存当前默认参数模式
        prev_default_emit_promoted_ops(
            BytecodeEmitMode::is_emit_promoted_ops_enabled()) {  // 保存当前推广操作模式
    // 设置新的字节码生成模式选项
    BytecodeEmitMode::set_default_value_for_unspecified_arg_enabled(
        enable_default_value_for_unspecified_arg);
    BytecodeEmitMode::set_default_args_before_out_args_enabled(
        enable_default_args_before_out_args);
    BytecodeEmitMode::set_default_emit_promoted_ops_enabled(
        enable_emit_promoted_ops);
  }

  // 析构函数，在对象销毁时恢复之前保存的字节码生成模式设置
  ~BytecodeEmitModeGuard() {
    BytecodeEmitMode::set_default_value_for_unspecified_arg_enabled(
        prev_default_value_for_unspecified_arg_mode);  // 恢复之前的默认值模式设置
    BytecodeEmitMode::set_default_args_before_out_args_enabled(
        prev_default_args_before_out_args);  // 恢复之前的默认参数设置
    BytecodeEmitMode::set_default_emit_promoted_ops_enabled(
        prev_default_emit_promoted_ops);  // 恢复之前的推广操作设置
  }

  // 保存之前的默认值模式
  bool prev_default_value_for_unspecified_arg_mode;
  // 保存之前的默认参数模式
  bool prev_default_args_before_out_args;
  // 保存之前的推广操作模式
  bool prev_default_emit_promoted_ops;
};

// 将给定的 IValue 向量转换为元组 IValue
TORCH_API IValue to_tuple(std::vector<IValue> ivalues);

// 创建一个名为 Table 的函数，接受一个字符串-IValue对的向量，并返回一个 IValue
TORCH_API IValue Table(const std::vector<std::pair<std::string, IValue>>& entries);

// 在接口调用推出后移除这些开关
TORCH_API void enableMobileInterfaceCallExport();  // 启用移动接口调用导出
bool getMobileInterfaceCallExport();  // 获取移动接口调用导出状态

// 从全局获取编译选项
TORCH_API CompilationOptions getOptionsFromGlobal();

// 将 JIT 模块保存到文件
TORCH_API void save_jit_module(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files = ExtraFilesMap());

// 将 JIT 模块保存为字节流
TORCH_API DetachedBuffer::UniqueDetachedBuffer save_jit_module_to_bytes(
    const Module& module,
    const ExtraFilesMap& extra_files = ExtraFilesMap());

// 将 JIT 模块保存到写函数中
TORCH_API void save_jit_module_to_write_func(
    const Module& module,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info,
    const std::function<size_t(const void*, size_t)>& writer_func);

// torch::jit 命名空间结束
} // namespace torch::jit
```