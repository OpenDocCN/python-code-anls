# `.\pytorch\torch\csrc\jit\serialization\export.cpp`

```
// 包含 Torch 序列化导出的头文件
#include <torch/csrc/jit/serialization/export.h>

// 包含 ATen 库的头文件
#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/core/functional.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

// 包含 Torch 自动微分的符号计算相关头文件
#include <torch/csrc/autograd/symbolic.h>

// 包含 Torch JIT 编译器日志相关头文件
#include <torch/csrc/jit/jit_log.h>

// 包含 Torch JIT 编译器中的优化和消除死代码相关头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 包含 Torch JIT 编译器中的内联优化相关头文件
#include <torch/csrc/jit/passes/inliner.h>

// 包含 Torch JIT 运行时指令相关头文件
#include <torch/csrc/jit/runtime/instruction.h>

// 包含 Torch JIT 序列化的常量定义头文件
#include <torch/csrc/jit/serialization/import_export_constants.h>

// 包含 Torch JIT 序列化导入导出函数头文件
#include <torch/csrc/jit/serialization/import_export_functions.h>

// 包含 Torch JIT 序列化导入导出助手函数头文件
#include <torch/csrc/jit/serialization/import_export_helpers.h>

// 包含 Torch JIT 对 ONNX 格式的支持头文件
#include <torch/csrc/jit/serialization/onnx.h>

// 包含 Torch ONNX 向后兼容性支持头文件
#include <torch/csrc/onnx/back_compat.h>

// 包含 Torch ONNX 的主要功能头文件
#include <torch/csrc/onnx/onnx.h>

// 包含 Torch 版本信息的头文件
#include <torch/version.h>

// 忽略以下代码段的警告："-Wnewline-eof"
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wnewline-eof")
#include <onnx/checker.h>
C10_DIAGNOSTIC_POP()

// 包含 ONNX Protobuf 格式定义的头文件
#include <onnx/onnx_pb.h>

// 包含 ONNX Protobuf 格式的工具函数头文件
#include <onnx/proto_utils.h>

// 忽略以下代码段的警告："-Wsuggest-override"
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
#include <onnx/shape_inference/implementation.h>
C10_DIAGNOSTIC_POP()

// 包含标准库头文件
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Torch JIT 命名空间
namespace torch::jit {

// 获取小端数据的函数，接收 ATen 张量作为参数
static std::string get_little_endian_data(const at::Tensor& t) {
  // 根据平台字节顺序选择不同的处理方式
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  // 如果是小端字节顺序，则直接将张量数据转换成字符串返回
  return std::string(
      static_cast<char*>(t.data_ptr()), t.element_size() * t.numel());
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // 如果是大端字节顺序，则需要逐个字节交换处理
  const size_t element_size = t.element_size();
  const size_t num_elements = t.numel();

  std::vector<char> data_copy{
      static_cast<char*>(t.data_ptr()),
      static_cast<char*>(t.data_ptr()) + element_size * num_elements};

  for (size_t i = 0; i < num_elements; ++i) {
    char* start_byte = data_copy.data() + i * element_size;
    char* end_byte = start_byte + element_size - 1;
    /* keep swapping */
    // 逐个字节进行交换，以处理大端字节顺序下的数据
    for (size_t count = 0; count < element_size / 2; ++count) {
      std::swap(*start_byte, *end_byte);
      ++start_byte;
      --end_byte;
    }
  }

  // 将交换后的数据转换成字符串返回
  return std::string(data_copy.data(), element_size * num_elements);
#else
// 如果不是预期的小端或大端字节顺序，则抛出错误
#error Unexpected or undefined __BYTE_ORDER__
#endif
}

// 将归档数据和张量写入输出流的函数
void writeArchiveAndTensors(
    const std::string& archive_name,
    const char* data,
    size_t size,
    const std::vector<at::Tensor>& tensors,
    caffe2::serialize::PyTorchStreamWriter& out) {
  // 归档内文件名的前缀
  std::string prefix = archive_name + "/";
  size_t i = 0;
  // 遍历张量列表
  for (const auto& td : tensors) {
    // 获取可写的张量数据
    WriteableTensorData writable_td = getWriteableTensorData(td);
    // 构造文件名，格式为 prefix + 序号
    std::string fname = prefix + std::to_string(i++);
    // 将数据写入输出流
    out.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
  }
  // 写入归档文件名为 archive_name + ".pkl" 的数据
  std::string fname = archive_name + ".pkl";
  out.writeRecord(fname, data, size);
}

// 匿名命名空间，定义了一些与 ONNX 相关的静态常量和别名
namespace {
namespace onnx_torch = ::torch::onnx;
namespace onnx = ::ONNX_NAMESPACE;

// 无效的操作集版本号
const static int kInvalidOpsetVersion = -1;
const static int kMainOpsetVersion = 20;
// 定义主要操作集版本号为 20

// 根据 https://github.com/onnx/onnx/blob/master/onnx/helper.py 中的 OP_SET_ID_VERSION_MAP
constexpr static std::array<int64_t, kMainOpsetVersion + 1>
    kOpsetVersionToIRVersion = {
        kInvalidOpsetVersion,   // 无效操作集版本号
        3, // opset 1            // 操作集 1 对应的 IR 版本号为 3
        kInvalidOpsetVersion,   // 无效操作集版本号
        kInvalidOpsetVersion,   // 无效操作集版本号
        kInvalidOpsetVersion,   // 无效操作集版本号
        3, // opset 5            // 操作集 5 对应的 IR 版本号为 3
        3, // opset 6            // 操作集 6 对应的 IR 版本号为 3
        3, // opset 7            // 操作集 7 对应的 IR 版本号为 3
        3, // opset 8            // 操作集 8 对应的 IR 版本号为 3
        4, // opset 9            // 操作集 9 对应的 IR 版本号为 4
        5, // opset 10           // 操作集 10 对应的 IR 版本号为 5
        6, // opset 11           // 操作集 11 对应的 IR 版本号为 6
        7, // opset 12           // 操作集 12 对应的 IR 版本号为 7
        7, // opset 13           // 操作集 13 对应的 IR 版本号为 7
        7, // opset 14           // 操作集 14 对应的 IR 版本号为 7
        8, // opset 15           // 操作集 15 对应的 IR 版本号为 8
        8, // opset 16           // 操作集 16 对应的 IR 版本号为 8
        8, // opset 17           // 操作集 17 对应的 IR 版本号为 8
        8, // opset 18           // 操作集 18 对应的 IR 版本号为 8
        9, // opset 19           // 操作集 19 对应的 IR 版本号为 9
        9, // opset 20           // 操作集 20 对应的 IR 版本号为 9
};

std::string getNodeStackTraceString(const Node* n) {
  return n->sourceRange().str();
}

void validateBlock(
    Block* b,
    onnx_torch::OperatorExportTypes operator_export_type) {
  for (auto node : b->nodes()) {
    for (Block* sub_block : node->blocks()) {
      validateBlock(sub_block, operator_export_type);
    }
    // 定义一个宏，用于在导出失败时抛出运行时异常，提供更好的错误信息和源码行数
#define FAIL_EXPORT(name)                          \
  throw std::runtime_error(                        \
      std::string("ONNX export failed: ") + name + \
      "\n\nGraph we tried to export:\n" + b->owningGraph()->toString());
    // 对于某些特定类型的操作符，提供特殊的错误信息
    if (node->kind() == prim::PythonOp) {
      if (operator_export_type !=
          onnx_torch::OperatorExportTypes::ONNX_FALLTHROUGH) {
        auto py_node = static_cast<PythonOp*>(node);
        FAIL_EXPORT(
            "Couldn't export Python operator " + py_node->name() +
            "\n\nDefined at:\n" + getNodeStackTraceString(node))
      }
    } else {
      if (node->kind() == prim::PackPadded || node->kind() == prim::PadPacked) {
        if (operator_export_type !=
            onnx_torch::OperatorExportTypes::ONNX_FALLTHROUGH) {
          FAIL_EXPORT(
              "Cannot export individual pack_padded_sequence or pad_packed_sequence; these operations must occur in pairs.\n\nUsage of this operation occurred at:\n" +
              getNodeStackTraceString(node));
        }
      }
      // 检查是否支持 ATen 操作，如果不支持且不是必须为 None，则导出失败
      bool is_aten_enabled = operator_export_type ==
              onnx_torch::OperatorExportTypes::ONNX_ATEN_FALLBACK ||
          operator_export_type == onnx_torch::OperatorExportTypes::ONNX_ATEN ||
          operator_export_type ==
              onnx_torch::OperatorExportTypes::ONNX_FALLTHROUGH;
      if (node->kind().is_aten() && !is_aten_enabled && !node->mustBeNone()) {
        FAIL_EXPORT(
            "Couldn't export operator " + node->kind().toDisplayString() +
            "\n\nDefined at:\n" + getNodeStackTraceString(node));
      }
    }
#undef FAIL_EXPORT
  }
}

void validateGraph(
    const std::shared_ptr<Graph>& graph,
    onnx_torch::OperatorExportTypes operator_export_type) {
  validateBlock(graph->block(), operator_export_type);
}
// 根据给定的根路径，返回文件根目录路径
std::string GetFileRootPath(const std::string& rootPath) {
  std::string rootPath_ = rootPath;
  // 首先，将斜杠字符统一为正斜杠
  std::replace(rootPath_.begin(), rootPath_.end(), '\\', '/');
  // 其次，去除末尾的斜杠（如果有的话）
  std::regex trailer("/+$");
  std::string root = std::regex_replace(rootPath_, trailer, std::string());
  // 提取最后一个斜杠前的部分作为文件夹路径
  std::string folder = root.substr(0, root.find_last_of('/'));
  if (folder == rootPath_) { // 如果没有指定根文件夹，则选择当前工作目录
    return std::string(".");
  }
  return folder;
}

// 根据外部引用名称生成有效的文件名
std::string GetExternalFileName(
    const std::optional<std::string>& external_ref) {
  auto tensorName = external_ref.value();
  const std::string illegalChars = "\\/:?\"<>|";
  // 替换非法字符为下划线
  for (char& i : tensorName) {
    if (illegalChars.find(i) != std::string::npos) {
      i = '_';
    }
  }
  return tensorName;
}

// 关闭文件指针所指向的文件
void CloseFile(FILE* fp) {
  fclose(fp);
}

// 创建外部文件，并将张量数据写入其中
void CreateExternalFile(
    const at::Tensor& tensor,
    const std::string& tensorName,
    const std::string& onnx_file_path) {
  auto folder = GetFileRootPath(onnx_file_path);
  std::string fullFilePath = folder + "/" + tensorName;
  // 使用智能指针管理文件指针，并自动在退出时调用CloseFile()关闭文件
  std::unique_ptr<FILE, decltype(&CloseFile)> fp(
      fopen(fullFilePath.c_str(), "wb"), &CloseFile);
  if (fp == nullptr) {
    throw std::runtime_error(
        std::string("ONNX export failed. Could not open file or directory: ") +
        fullFilePath);
  }
  // 获取张量数据并以小端序写入文件
  std::string s = get_little_endian_data(tensor);
  fwrite(s.c_str(), tensor.element_size(), tensor.numel(), fp.get());
} // fclose() 在此通过 CloseFile() 被调用，如果 FILE* 不是空指针的话

// 图编码器类，用于导出图形数据到ONNX格式
class GraphEncoder {
 public:
  GraphEncoder(
      const std::shared_ptr<Graph>& graph,
      int64_t onnx_opset_version,
      onnx_torch::OperatorExportTypes operator_export_type,
      const std::map<std::string, at::Tensor>& initializers,
      const std::unordered_map<
          std::string,
          std::unordered_map<int64_t, std::string>>& dynamic_axes,
      bool defer_weight_export,
      bool strip_doc,
      bool keep_initializers_as_inputs,
      const std::map<std::string, int>& custom_opsets,
      bool add_node_names,
      bool use_external_data_format,
      const std::string& onnx_file_path,
      NodeAttrNameMap node_attr_to_name = {});

  // 返回模型的 ONNX 格式的 Protobuf 对象
  std::shared_ptr<onnx::ModelProto> get_model_proto() {
    return model_proto_;
  }

  // 返回符号维度参数映射
  SymbolDimMap get_symbol_dim_param_map() {
    return symbol_dim_map_;
  }

  // 返回原始数据导出映射
  RawDataExportMap get_raw_data_export_map() {
    return raw_data_export_map_;
  }

  // 返回是否使用外部数据格式的标志
  bool get_use_external_data_format() {
    return use_external_data_format_;
  }

  // 返回 ONNX 节点名称映射
  NodeNameMap get_onnx_node_names() {
};

// 将 ATen 类型转换为相应的 ONNX 类型
onnx::TensorProto_DataType ATenTypeToOnnxType(at::ScalarType at_type) {
  switch (at_type) {
    case at::kDouble:
      return onnx::TensorProto_DataType_DOUBLE;
    case at::kFloat:
      return onnx::TensorProto_DataType_FLOAT;
    case at::kHalf:
      return onnx::TensorProto_DataType_FLOAT16;
    case at::kByte:
      return onnx::TensorProto_DataType_UINT8;
    // 如果张量类型是8位有符号整数（char），返回对应的ONNX数据类型INT8
    case at::kChar:
      return onnx::TensorProto_DataType_INT8;
    // 如果张量类型是16位有符号整数（short），返回对应的ONNX数据类型INT16
    case at::kShort:
      return onnx::TensorProto_DataType_INT16;
    // 如果张量类型是32位有符号整数（int），返回对应的ONNX数据类型INT32
    case at::kInt:
      return onnx::TensorProto_DataType_INT32;
    // 如果张量类型是64位有符号整数（long），返回对应的ONNX数据类型INT64
    case at::kLong:
      return onnx::TensorProto_DataType_INT64;
    // 如果张量类型是布尔型（bool），返回对应的ONNX数据类型BOOL
    case at::kBool:
      return onnx::TensorProto_DataType_BOOL;
    // 如果张量类型是8位量化有符号整数（QInt8），返回对应的ONNX数据类型INT8
    case at::kQInt8:
      return onnx::TensorProto_DataType_INT8;
    // 如果张量类型是8位量化无符号整数（QUInt8），返回对应的ONNX数据类型UINT8
    case at::kQUInt8:
      return onnx::TensorProto_DataType_UINT8;
    // 如果张量类型是32位量化有符号整数（QInt32），返回对应的ONNX数据类型INT32
    case at::kQInt32:
      return onnx::TensorProto_DataType_INT32;
    // 如果张量类型是16位浮点数（BFloat16），返回对应的ONNX数据类型BFLOAT16
    case at::kBFloat16:
      return onnx::TensorProto_DataType_BFLOAT16;
    // 如果张量类型是8位浮点数（FLOAT8E4M3FN），返回对应的ONNX数据类型FLOAT8E4M3FN
    case at::kFloat8_e4m3fn:
      return onnx_torch::TensorProto_DataType_FLOAT8E4M3FN;
    // 如果张量类型是8位浮点数（FLOAT8E5M2），返回对应的ONNX数据类型FLOAT8E5M2
    case at::kFloat8_e5m2:
      return onnx_torch::TensorProto_DataType_FLOAT8E5M2;
    // 如果张量类型是8位浮点数（FLOAT8E4M3FNUZ），返回对应的ONNX数据类型FLOAT8E4M3FNUZ
    case at::kFloat8_e4m3fnuz:
      return onnx_torch::TensorProto_DataType_FLOAT8E4M3FNUZ;
    // 如果张量类型是8位浮点数（FLOAT8E5M2FNUZ），返回对应的ONNX数据类型FLOAT8E5M2FNUZ
    case at::kFloat8_e5m2fnuz:
      return onnx_torch::TensorProto_DataType_FLOAT8E5M2FNUZ;
    // 如果张量类型不在已知类型中，抛出异常，显示未知的张量标量类型
    default:
      TORCH_CHECK(
          false,
          "ScalarType ",
          toString(at_type),
          " is an unexpected tensor scalar type");
  }
// 将 ATen 的属性类型映射到对应的 ONNX 属性类型
onnx::AttributeProto_AttributeType ATenAttributeKindToOnnxAttributeType(
    AttributeKind at_kind,  // 输入的 ATen 属性类型
    const jit::Symbol name) {  // 属性的名称
  switch (at_kind) {
    case AttributeKind::f:  // 如果属性类型是 float
      return onnx::AttributeProto_AttributeType_FLOAT;  // 返回 ONNX 的 float 类型
    case AttributeKind::fs:  // 如果属性类型是 float 数组
      return onnx::AttributeProto_AttributeType_FLOATS;  // 返回 ONNX 的 float 数组类型
    case AttributeKind::i:  // 如果属性类型是 int
      return onnx::AttributeProto_AttributeType_INT;  // 返回 ONNX 的 int 类型
    case AttributeKind::is:  // 如果属性类型是 int 数组
      return onnx::AttributeProto_AttributeType_INTS;  // 返回 ONNX 的 int 数组类型
    case AttributeKind::s:  // 如果属性类型是 string
      return onnx::AttributeProto_AttributeType_STRING;  // 返回 ONNX 的 string 类型
    case AttributeKind::ss:  // 如果属性类型是 string 数组
      return onnx::AttributeProto_AttributeType_STRINGS;  // 返回 ONNX 的 string 数组类型
    case AttributeKind::t:  // 如果属性类型是 tensor
      return onnx::AttributeProto_AttributeType_TENSOR;  // 返回 ONNX 的 tensor 类型
    case AttributeKind::ts:  // 如果属性类型是 tensor 数组
      return onnx::AttributeProto_AttributeType_TENSORS;  // 返回 ONNX 的 tensor 数组类型
    case AttributeKind::ty:  // 如果属性类型是 type proto
      return onnx::AttributeProto_AttributeType_TYPE_PROTO;  // 返回 ONNX 的 type proto 类型
    case AttributeKind::tys:  // 如果属性类型是 type proto 数组
      return onnx::AttributeProto_AttributeType_TYPE_PROTOS;  // 返回 ONNX 的 type proto 数组类型
    case AttributeKind::g:  // 如果属性类型是 graph
      return onnx::AttributeProto_AttributeType_GRAPH;  // 返回 ONNX 的 graph 类型
    case AttributeKind::gs:  // 如果属性类型是 graph 数组
      return onnx::AttributeProto_AttributeType_GRAPHS;  // 返回 ONNX 的 graph 数组类型
    default:  // 如果属性类型未知
      std::ostringstream err_msg;  // 创建错误消息流
      err_msg << "attribute \"" << name.toDisplayString()
              << "\" has unexpected kind: " << toString(at_kind);  // 构建错误消息内容
      throw std::runtime_error(err_msg.str());  // 抛出运行时异常，显示错误消息
  }
}

// 构造函数：GraphEncoder 类的构造函数，用于将 PyTorch 图编码为 ONNX 格式
GraphEncoder::GraphEncoder(
    const std::shared_ptr<Graph>& graph,  // PyTorch 图的共享指针
    int64_t onnx_opset_version,  // ONNX 的操作集版本
    onnx_torch::OperatorExportTypes operator_export_type,  // 操作符导出类型
    const std::map<std::string, at::Tensor>& initializers,  // 初始值的映射表
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,  // 动态轴的映射表
    bool defer_weight_export,  // 是否延迟权重导出
    bool strip_doc,  // 是否剥离文档
    bool keep_initializers_as_inputs,  // 是否保持初始值作为输入
    const std::map<std::string, int>& custom_opsets,  // 自定义操作集映射表
    bool add_node_names,  // 是否添加节点名称
    bool use_external_data_format,  // 是否使用外部数据格式
    const std::string& onnx_file_path,  // ONNX 文件路径
    NodeAttrNameMap node_attr_to_name)  // 节点属性名称映射表
    // 初始化 model_proto_，使用 std::make_shared 创建一个新的 onnx::ModelProto 对象
    model_proto_(std::make_shared<onnx::ModelProto>()),
    // 设置导出操作类型
    operator_export_type_(operator_export_type),
    // 是否剥离文档字符串
    strip_doc_(strip_doc),
    // 是否延迟权重导出
    defer_weight_export_(defer_weight_export),
    // 是否使用外部数据格式
    use_external_data_format_(use_external_data_format),
    // 设置目标 ONNX 操作集的版本号
    onnx_opset_version_(onnx_opset_version),
    // 自定义操作集的映射
    custom_opsets_(custom_opsets),
    // 图的引用
    graph_(graph),
    // 移动构造节点属性到名称映射
    node_attr_to_name_(std::move(node_attr_to_name)) {
    
    // 设置模型生产者名称为 "pytorch"
    model_proto_->set_producer_name("pytorch");
    // 检查是否支持给定的 onnx_opset_version
    TORCH_CHECK(
        onnx_opset_version > 0 &&
            static_cast<size_t>(onnx_opset_version) <
                kOpsetVersionToIRVersion.size() &&
            kOpsetVersionToIRVersion[onnx_opset_version] != kInvalidOpsetVersion,
        "Unsupported onnx_opset_version: ",
        onnx_opset_version);
    
    // 设置模型的 IR 版本
    model_proto_->set_ir_version(kOpsetVersionToIRVersion[onnx_opset_version]);
    // 设置模型的生产者版本为当前的 PyTorch 版本
    model_proto_->set_producer_version(TORCH_VERSION);
    // 验证图的有效性
    validateGraph(graph, operator_export_type);
    
    // 如果图的 Proto 大小超过 2GB，设置 use_external_data_format 为 true
    if (!use_external_data_format &&
        GetGraphProtoSize(
            model_proto_->mutable_graph(),
            graph,
            add_node_names,
            use_external_data_format,
            onnx_file_path,
            initializers) > INT_MAX) {
      GRAPH_DEBUG(
          "Exporting model exceed maximum protobuf size of 2GB. Storing model parameters in external data files");
      use_external_data_format = true;
      // 将 use_external_data_format_ 设置为 true，用于返回值
      use_external_data_format_ = use_external_data_format;
    }
    
    // 如果使用外部数据格式，检查输出文件路径是否为空
    if (use_external_data_format) {
      TORCH_CHECK(
          !onnx_file_path.empty(),
          "The serialized model is larger than the 2GiB limit imposed by the protobuf library. ",
          "Therefore the output file must be a file path, so that the ONNX external data can ",
          "be written to the same directory. Please specify the output file name.");
    }
    
    // 添加操作集导入项到模型中，设置版本为 onnx_opset_version
    auto* imp = model_proto_->add_opset_import();
    // 这是我们目标的 ONNX 操作集的版本
    imp->set_version(onnx_opset_version);
    
    // 编码图结构到 model_proto_ 的图中
    EncodeGraph(
        model_proto_->mutable_graph(),
        graph,
        initializers,
        dynamic_axes,
        keep_initializers_as_inputs,
        add_node_names,
        use_external_data_format,
        onnx_file_path);
    
    // 遍历自定义操作集，为每个 domain 添加相应的操作集版本号
    for (const std::string& domain : domains_) {
      auto* opset = model_proto_->add_opset_import();
      opset->set_domain(domain);
      // 检查是否已注册该 domain 的版本号，若未注册则设置为版本 1
      auto it = custom_opsets.find(domain);
      if (it == custom_opsets.end())
        opset->set_version(1);
      else {
        opset->set_version(it->second);
      }
    }
    
    // 遍历所有自定义操作集，为每个自定义操作集设置版本号
    for (auto const& custom_opset : custom_opsets) {
    # 检查 domains_ 列表中是否不包含 custom_opset.first 元素
    if (!std::count(domains_.begin(), domains_.end(), custom_opset.first)) {
      # 如果 custom_opset.first 不在 domains_ 中，发出警告消息
      TORCH_WARN(
          "Custom opset domain: '",
          custom_opset.first,
          "' provided is not used in the model. ",
          "Please verify custom opset domain names.");
    }
  }
void GraphEncoder::TensorTypeToONNXType(
    const TensorTypePtr& tensor_type,                         // 接收一个智能指针指向TensorType对象，表示要转换的张量类型
    const std::string& dim_name_prefix,                       // 维度名称的前缀，用于命名维度参数
    const std::string& name,                                  // 张量的名称
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,  // 包含动态轴映射的哈希表，用于处理动态形状
    onnx::TypeProto_Tensor* onnx_tensor_type,                 // 要编码的ONNX张量类型对象的指针
    bool assign_dim_param) {                                  // 是否分配维度参数的标志
  if (tensor_type->dim()) {                                   // 如果张量类型具有维度信息
    onnx::TensorShapeProto* shape = onnx_tensor_type->mutable_shape();  // 获取ONNX张量类型的形状信息
    auto sizes = tensor_type->symbolic_sizes().sizes().value();  // 获取张量的尺寸信息
    for (const auto i : c10::irange(sizes.size())) {           // 迭代所有尺寸
      shape->add_dim();                                       // 添加一个新的维度描述
      if ((dynamic_axes.find(name) != dynamic_axes.end()) &&   // 如果动态轴中存在当前名称
          (dynamic_axes.at(name).find(i) != dynamic_axes.at(name).end())) {
        shape->mutable_dim(i)->set_dim_param(dynamic_axes.at(name).at(i));  // 设置维度参数为动态轴中的对应值
        if (!sizes[i].is_static()) {                           // 如果尺寸是动态的
          symbol_dim_map_[sizes[i]] = dynamic_axes.at(name).at(i);  // 更新符号维度映射表
        }
      } else if (sizes[i].is_static()) {                       // 如果尺寸是静态的
        shape->mutable_dim(i)->set_dim_value(sizes[i].static_size());  // 设置维度值为静态尺寸大小
      } else if (assign_dim_param) {                           // 如果需要分配维度参数
        if (symbol_dim_map_.find(sizes[i]) == symbol_dim_map_.end()) {
          symbol_dim_map_[sizes[i]] =
              dim_name_prefix + name + "_dim_" + std::to_string(i);  // 使用命名规则为动态尺寸分配维度参数
        }
        shape->mutable_dim(i)->set_dim_param(symbol_dim_map_[sizes[i]]);  // 设置维度参数为已分配的值
      }
    }
  }
  if (tensor_type->scalarType()) {                            // 如果张量类型具有标量类型
    onnx_tensor_type->set_elem_type(                          // 设置ONNX张量类型的元素类型
        ATenTypeToOnnxType(tensor_type->scalarType().value()));  // 调用函数将ATen标量类型转换为ONNX类型
  }
}
    # 如果节点类型为 TensorType
    if (TensorTypePtr tensor_type = node_type->cast<TensorType>()) {
        # 获取节点的 ONNX 类型，并设置其为 Tensor 类型
        onnx::TypeProto_Tensor* onnx_tensor_type = onnx_type->mutable_tensor_type();
        # 将 ATen 中的 Long 类型转换为对应的 ONNX 类型
        onnx_tensor_type->set_elem_type(ATenTypeToOnnxType(at::kLong));
    
    # 如果节点类型为 FloatType
    } else if (FloatTypePtr float_type = node_type->cast<FloatType>()) {
        # 获取节点的 ONNX 类型，并设置其为 Tensor 类型
        onnx::TypeProto_Tensor* onnx_tensor_type = onnx_type->mutable_tensor_type();
        # 将 ATen 中的 Float 类型转换为对应的 ONNX 类型
        onnx_tensor_type->set_elem_type(ATenTypeToOnnxType(at::kFloat));
    
    # 如果节点类型为 ListType
    } else if (ListTypePtr list_type = node_type->cast<ListType>()) {
        # 获取列表元素的类型
        auto list_elem_type = list_type->getElementType();
        # 获取节点的 ONNX 类型，并设置其为 Sequence 类型
        onnx::TypeProto_Sequence* sequence_type = onnx_type->mutable_sequence_type();
        # 获取 Sequence 类型的元素类型，并设置为对应的 ONNX 类型
        onnx::TypeProto* onnx_tensor_type = sequence_type->mutable_elem_type();
        EncodeValueInfoType(onnx_tensor_type, list_elem_type, n, dynamic_axes);
    
    # 如果节点类型为 OptionalType
    } else if (OptionalTypePtr optional_type = node_type->cast<OptionalType>()) {
        # 获取 Optional 类型的元素类型
        auto elem_type = optional_type->getElementType();
        # 如果元素类型为 TensorType
        if (TensorTypePtr tensor_type = elem_type->cast<TensorType>()) {
            # 获取节点的 ONNX 类型，并设置其为 Optional 类型
            onnx::TypeProto_Optional* onnx_optional_type = onnx_type->mutable_optional_type();
            # 获取 Optional 类型的元素类型，并设置为 Tensor 类型的 ONNX 表示
            onnx::TypeProto_Tensor* onnx_tensor_type = onnx_optional_type->mutable_elem_type()->mutable_tensor_type();
            # 将 ATen 中的 Tensor 类型转换为对应的 ONNX 类型
            TensorTypeToONNXType(
                tensor_type,
                dim_name_prefix,
                n->debugName(),
                dynamic_axes,
                onnx_tensor_type);
        
        # 如果元素类型为 ListType 中的 TensorType
        } else if (ListTypePtr inner_node_type = elem_type->cast<ListType>()) {
            auto list_elem_type = inner_node_type->getElementType();
            if (TensorTypePtr tensor_type = list_elem_type->cast<TensorType>()) {
                # 获取节点的 ONNX 类型，并设置其为 Optional 类型
                onnx::TypeProto_Optional* onnx_optional_type = onnx_type->mutable_optional_type();
                # 获取 Optional 类型的元素类型，并设置为 Sequence 类型的 ONNX 表示
                onnx::TypeProto_Sequence* onnx_optional_sequence_type = onnx_optional_type->mutable_elem_type()->mutable_sequence_type();
                # 获取 Sequence 类型中元素的类型，并设置为 Tensor 类型的 ONNX 表示
                onnx::TypeProto_Tensor* onnx_tensor_type = onnx_optional_sequence_type->mutable_elem_type()->mutable_tensor_type();
                # 将 ATen 中的 Tensor 类型转换为对应的 ONNX 类型
                TensorTypeToONNXType(
                    tensor_type,
                    dim_name_prefix,
                    n->debugName(),
                    dynamic_axes,
                    onnx_tensor_type);
            }
        }
    }
}

void GraphEncoder::EncodeValueInfo(
    onnx::GraphProto* graph_proto,
    onnx::ValueInfoProto* v,
    const Value* n,
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes) {
  // 从输入值（Value对象）获取调试名称
  std::string name = n->debugName();
  // 将调试名称设置为输出值信息的名称
  v->set_name(name);
  // 编码值信息的类型信息，包括数据类型和动态轴信息
  EncodeValueInfoType(v->mutable_type(), n->type(), n, dynamic_axes);
}

void GraphEncoder::EncodeGraph(
    onnx::GraphProto* graph_proto,
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,
    bool keep_initializers_as_inputs,
    bool add_node_names,
    bool use_external_data_format,
    const std::string& onnx_file_path) {
  // 编码整个图结构，调用EncodeBlock来处理每个基本块
  EncodeBlock(
      graph_proto,
      graph->block(),
      initializers,
      dynamic_axes,
      keep_initializers_as_inputs,
      add_node_names,
      use_external_data_format,
      onnx_file_path);
}

void GraphEncoder::EncodeBlock(
    onnx::GraphProto* graph_proto,
    const Block* block,
    const std::map<std::string, at::Tensor>& initializers,
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,
    bool keep_initializers_as_inputs,
    bool add_node_names,
    bool use_external_data_format,
    const std::string& onnx_file_path) {
  // 确保图协议不为空
  TORCH_INTERNAL_ASSERT(graph_proto != nullptr);
  // 检查当前块是否属于主图还是子图，并设置相应的名称
  if (nullptr == block->owningNode()) {
    // 主图的情况
    graph_proto->set_name("main_graph");
  } else {
    // 子图的情况，设置一个更有意义的名称
    // 每个子图附带一个唯一编号
    std::string block_name = "sub_graph";
    if (num_blocks_) {
      block_name += std::to_string(num_blocks_);
    }
    num_blocks_++;
    graph_proto->set_name(block_name);
  }

  // 根据参数keep_initializers_as_inputs的值，决定是否将初始化器添加为输入
  // 如果keep_initializers_as_inputs=true，则将初始化器作为输入添加到ONNX图中
  // 否则，只添加非参数输入作为图的输入，而不添加初始化器（参数）
  if (keep_initializers_as_inputs) {
    for (auto input : block->inputs()) {
      // 添加图输入
      onnx::ValueInfoProto* v = graph_proto->add_input();
      // 编码输入值的信息
      EncodeValueInfo(graph_proto, v, input, dynamic_axes);
    }
  } else {
    // 不将初始化器作为输入的情况
    for (auto input : block->inputs()) {
      auto it = initializers.find(input->debugName());
      if (it == initializers.end()) {
        // 输入不是初始化器时，将其添加为图输入
        onnx::ValueInfoProto* v = graph_proto->add_input();
        // 编码输入值的信息
        EncodeValueInfo(graph_proto, v, input, dynamic_axes);
      }
    }
  }
  // 添加当前块的输出作为图的输出
  for (auto output : block->outputs()) {
    onnx::ValueInfoProto* v = graph_proto->add_output();
  // 遍历块中的每个节点，对每个节点进行处理
  for (auto node : block->nodes()) {
    // 如果节点必须为None，则跳过处理，这种节点通常用于实现可选输入
    if (node->mustBeNone()) {
      // None 节点用于表示不提供可选输入的情况。可以创建一个 Undefined 节点，
      // 并将其输出作为该输入的占位符。
      continue;
    }
    // 如果节点类型为 LocalFunctionDef，将其编码为本地函数定义并添加到模型协议中
    if (node->kind() == ::c10::Symbol::onnx("LocalFunctionDef")) {
      auto* func_proto = model_proto_->add_functions();
      EncodeLocalFunction(
          graph_proto,
          func_proto,
          node,
          add_node_names,
          use_external_data_format,
          onnx_file_path);
      continue;
    }
    // 否则，将节点编码为图协议中的节点，并添加到图中
    auto* n_proto = graph_proto->add_node();
    EncodeNode(
        graph_proto,
        n_proto,
        node,
        add_node_names,
        use_external_data_format,
        onnx_file_path);
  }
  // 将初始化器添加到图协议中，用于描述图的初始状态
  AddInitializersIntoGraphProto(
      graph_proto,
      block,
      initializers,
      use_external_data_format,
      onnx_file_path);
// 将初始化器添加到图协议中
void GraphEncoder::AddInitializersIntoGraphProto(
    onnx::GraphProto* graph_proto,  // ONNX图协议指针，用于存储初始化器信息
    const Block* block,  // 表示图的块结构
    const std::map<std::string, at::Tensor>& initializers,  // 包含初始化器张量的映射
    bool use_external_data_format,  // 是否使用外部数据格式
    const std::string& onnx_file_path) {  // ONNX文件路径
  TORCH_INTERNAL_ASSERT(block->inputs().size() >= initializers.size());  // 内部断言，验证输入数量不少于初始化器数量
  for (auto input : block->inputs()) {  // 遍历块的输入
    auto name_tensor_pair = initializers.find(input->debugName());  // 查找与输入名称对应的初始化器
    if (name_tensor_pair == initializers.end()) {  // 如果找不到对应的初始化器，则跳过
      continue;
    }
    auto p = graph_proto->add_initializer();  // 添加一个初始化器到图协议中
    p->set_name(name_tensor_pair->first);  // 设置初始化器的名称
    EncodeTensor(  // 调用EncodeTensor函数，对张量进行编码
        p,
        name_tensor_pair->second,
        name_tensor_pair->first,
        use_external_data_format,
        onnx_file_path);
  }
}

// 计算图协议的大小
unsigned long long int GraphEncoder::GetGraphProtoSize(
    onnx::GraphProto* graph_proto,  // ONNX图协议指针，用于计算大小
    const std::shared_ptr<Graph>& graph,  // 图的共享指针
    bool add_node_names,  // 是否添加节点名称
    bool use_external_data_format,  // 是否使用外部数据格式
    const std::string& onnx_file_path,  // ONNX文件路径
    const std::map<std::string, at::Tensor>& initializers) {  // 包含初始化器张量的映射
  // 模型大小 = 所有初始化器大小之和 + 所有ONNX常量节点大小之和

  // 添加所有初始化器的大小
  onnx::GraphProto graph_proto_copy = onnx::GraphProto(*graph_proto);  // 创建图协议的副本
  unsigned long long int size = graph_proto_copy.ByteSizeLong();  // 获取图协议副本的字节大小
  for (auto input : graph->inputs()) {  // 遍历图的输入
    auto name_tensor_pair = initializers.find(input->debugName());  // 查找与输入名称对应的初始化器
    if (name_tensor_pair == initializers.end()) {  // 如果找不到对应的初始化器，则跳过
      continue;
    }
    auto tensor_proto = graph_proto_copy.add_initializer();  // 添加一个初始化器到图协议的副本中
    const at::Tensor& tensor = name_tensor_pair->second;  // 获取初始化器张量
    for (auto d : tensor.sizes()) {  // 遍历张量的维度
      tensor_proto->add_dims(d);  // 添加维度信息到初始化器中
    }
    tensor_proto->set_data_type(ATenTypeToOnnxType(tensor.scalar_type()));  // 设置数据类型

    // 实际上不将缓冲区复制到tensor_proto中，因为这很昂贵。我们只需要其大小。
    size += tensor_proto->ByteSizeLong();  // 增加tensor_proto的字节大小
    size += tensor.element_size() * tensor.numel();  // 计算张量的总大小（元素大小乘以元素数量）
  }

  // 添加所有为Tensor的onnx::Constant节点的大小
  for (const auto& node : graph->nodes()) {  // 遍历图的节点
    if (node->kind() == ::c10::onnx::Constant &&  // 如果节点是ONNX常量
        node->hasAttribute(attr::value) &&  // 并且具有value属性
        node->kindOf(attr::value) == AttributeKind::t) {  // 并且value属性的类型是Tensor
      at::Tensor tensor = node->t(attr::value);  // 获取节点的Tensor

      // 实际上不将缓冲区复制到n_proto中，因为这很昂贵。我们只需要其大小。
      auto* n_proto = graph_proto_copy.add_node();  // 添加一个节点到图协议的副本中
      EncodeNode(  // 调用EncodeNode函数，对节点进行编码
          &graph_proto_copy,
          n_proto,
          node,
          add_node_names,
          use_external_data_format,
          onnx_file_path);

      // 计算张量的大小（字节）
      size += n_proto->ByteSizeLong();  // 增加n_proto的字节大小
      size += tensor.element_size() * tensor.numel();  // 计算张量的总大小（元素大小乘以元素数量）
    }
  }
  return size;  // 返回计算出的总大小
}

// 对节点进行编码
void GraphEncoder::EncodeNode(
    onnx::GraphProto* graph_proto,  // ONNX图协议指针，用于存储节点信息
    onnx::NodeProto* node_proto,  // 节点协议指针，用于表示节点信息
    const Node* node,  // 节点对象
    bool add_node_names,  // 是否添加节点名称
    bool use_external_data_format,  // 是否使用外部数据格式
    const std::string& onnx_file_path) {  // ONNX文件路径
  if (!strip_doc_) {  // 如果不剥离文档信息
    // 将节点的文档字符串设置为节点源范围的字符串表示
    node_proto->set_doc_string(node->sourceRange().str());
  }
  // 遍历节点的输入
  for (auto input : node->inputs()) {
    // 如果输入节点必须为None，则添加空字符串作为输入
    if (input->node()->mustBeNone()) {
      node_proto->add_input("");
    } else {
      // 否则添加输入节点的调试名称作为输入
      node_proto->add_input(input->debugName());
    }
  }
  // 遍历节点的输出
  for (auto output : node->outputs()) {
    // 添加输出节点的调试名称作为节点的输出
    node_proto->add_output(output->debugName());
    // 对输出节点进行中间值信息的编码
    EncodeIntermediateValueInfo(graph_proto, output);
  }
  // 如果节点的类型不是 ONNX 类型
  if (!node->kind().is_onnx()) {
    std::string domain;
    // 如果节点类型是 ATen 或者 Caffe2 类型
    if (node->kind().is_aten() || node->kind().is_caffe2()) {
      domain = node->kind().domainString();
    } else { // 自定义命名空间和域
      domain = node->kind().ns().toUnqualString();
    }
    // 将域名插入到域名集合中
    domains_.insert(domain);
    // 设置节点的域
    node_proto->set_domain(domain);
  }
  // 如果操作导出类型为 ONNX
  if (operator_export_type_ == onnx_torch::OperatorExportTypes::ONNX) {
    // 断言节点类型不是 ATen、Prim 或者属性节点
    TORCH_INTERNAL_ASSERT(
        !node->kind().is_aten() && !node->kind().is_prim() &&
        !node->kind().is_attr());
  }
  // 设置节点的操作类型
  node_proto->set_op_type(node->kind().toUnqualString());
  // 获取节点名称属性的符号
  const auto node_name_attribute_symbol =
      Symbol::attr(::torch::onnx::kOnnxNodeNameAttribute);
  // 如果需要添加节点名称
  if (add_node_names) {
    // 构造节点名称，格式为操作类型_序号
    std::string node_name =
        node_proto->op_type() + "_" + std::to_string(num_op_nodes_);
    // 如果节点具有节点名称属性，则使用该属性作为节点名称
    if (node->hasAttribute(node_name_attribute_symbol)) {
      node_name = node->s(node_name_attribute_symbol);
    }
    // 设置节点的名称
    node_proto->set_name(node_name);
    // 在节点映射中记录节点名称
    onnx_node_name_map_[node] = node_name;
    // 增加操作节点计数
    num_op_nodes_++;
  }
  // 查找节点属性到名称的映射
  auto attrs_it = node_attr_to_name_.find(node);
  // 遍历节点的属性名称
  for (auto attr_name : node->attributeNames()) {
    // 如果属性名称为节点名称属性，则跳过
    if (attr_name == node_name_attribute_symbol) {
      continue; // 跳过节点名称属性
    }
    // 如果在节点属性到名称的映射中找到节点的属性
    if (attrs_it != node_attr_to_name_.end()) {
      auto attr_it = attrs_it->second.find(attr_name.toUnqualString());
      // 如果找到对应的属性值，则添加属性到节点
      if (attr_it != attrs_it->second.end()) {
        AddAttribute(
            node_proto, attr_name, attr_it->second, node->kindOf(attr_name));
        continue;
      }
    }
    // 否则添加属性到节点
    AddAttribute(
        node_proto, node, attr_name, use_external_data_format, onnx_file_path);
  }
  // 如果节点的类型是 ::c10::onnx::Loop
  if (node->kind() == ::c10::onnx::Loop) {
    TORCH_INTERNAL_ASSERT(node->blocks().size() == 1);

    // 添加名为 "body" 的属性到节点，类型为 GRAPH
    auto body = node_proto->add_attribute();
    body->set_name("body");
    body->set_type(onnx::AttributeProto_AttributeType_GRAPH);
    // 创建并编码节点的块体
    auto g = body->mutable_g();
    EncodeBlock(
        g,
        node->blocks()[0],
        {},
        {},
        true,
        true,
        use_external_data_format,
        onnx_file_path);
  }
  // 如果节点的类型是 ::c10::onnx::If
  if (node->kind() == ::c10::onnx::If) {
    TORCH_INTERNAL_ASSERT(node->blocks().size() == 2);

    // 添加名为 "then_branch" 的属性到节点，类型为 GRAPH
    auto then_branch = node_proto->add_attribute();
    then_branch->set_name("then_branch");
    then_branch->set_type(onnx::AttributeProto_AttributeType_GRAPH);
    // 创建并编码节点的 then 分支
    auto true_g = then_branch->mutable_g();
    EncodeBlock(
        true_g,
        node->blocks()[0],
        {},
        {},
        true,
        true,
        use_external_data_format,
        onnx_file_path);
  }
    // 调用 EncodeBlock 函数，用于编码 true 分支的内容
    EncodeBlock(
        true_g,                      // true 分支的图形对象
        node->blocks()[0],           // 获取节点的第一个块
        {},                          // 空的输入参数
        {},                          // 空的输出参数
        true,                        // 标志：表示这是 true 分支
        true,                        // 标志：表示这是实际块而不是占位块
        use_external_data_format,    // 是否使用外部数据格式的标志
        onnx_file_path               // ONNX 文件路径
    );

    // 在节点属性中添加 else_branch 属性
    auto else_branch = node_proto->add_attribute();
    else_branch->set_name("else_branch");  // 设置属性名称为 else_branch
    else_branch->set_type(onnx::AttributeProto_AttributeType_GRAPH);  // 设置属性类型为图形对象

    auto false_g = else_branch->mutable_g();  // 获取 else_branch 属性的可变图形对象
    // 调用 EncodeBlock 函数，用于编码 false 分支的内容
    EncodeBlock(
        false_g,                     // false 分支的图形对象
        node->blocks()[1],           // 获取节点的第二个块
        {},                          // 空的输入参数
        {},                          // 空的输出参数
        true,                        // 标志：表示这是 true 分支
        true,                        // 标志：表示这是实际块而不是占位块
        use_external_data_format,    // 是否使用外部数据格式的标志
        onnx_file_path               // ONNX 文件路径
    );
}
}

void GraphEncoder::AddAttribute(
    onnx::NodeProto* node_proto,
    const jit::Symbol name,
    const std::string& ref_attr_name,
    const AttributeKind attr_kind) {
  // 向节点属性列表中添加一个新属性
  auto attr = node_proto->add_attribute();
  // 断言属性名为有效符号
  TORCH_INTERNAL_ASSERT(name.is_attr());
  // 设置属性的名称为未限定字符串形式的属性名
  attr->set_name(name.toUnqualString());
  // 设置属性的引用属性名称
  attr->set_ref_attr_name(ref_attr_name);
  // 根据属性类型转换函数将 ATen 属性类型映射为 ONNX 属性类型，并设置属性的类型
  attr->set_type(ATenAttributeKindToOnnxAttributeType(attr_kind, name));
}

void GraphEncoder::AddAttribute(
    onnx::NodeProto* node_proto,
    const jit::Node* node,
    const jit::Symbol name,
    const bool use_external_data_format,
    const std::string& onnx_file_path) {
  // 匿名函数，用于创建属性张量的名称
  auto createAttributeTensorName =
      [](const onnx::NodeProto* node_proto,
         onnx::TensorProto* tensor_proto,
         const jit::Symbol attr_name,
         size_t& num_external_data) -> std::string {
    // 如果张量已有名称，则返回其名称
    if (tensor_proto->has_name()) {
      return tensor_proto->name();
    }
    // 如果节点没有名称，则创建一个新名称
    if (!node_proto->has_name()) {
      auto name = node_proto->op_type() + "_" + attr_name.toDisplayString() +
          "_" + std::to_string(num_external_data);
      num_external_data++;
      return name;
    } else {
      // 使用节点名称和属性名创建名称
      return node_proto->name() + "_" + attr_name.toDisplayString();
    }
  };

  // 向节点属性列表中添加一个新属性
  auto attr = node_proto->add_attribute();
  // 断言属性名为有效符号
  TORCH_INTERNAL_ASSERT(name.is_attr());
  // 设置属性的名称为未限定字符串形式的属性名
  attr->set_name(name.toUnqualString());
  // 根据属性类型转换函数将 ATen 属性类型映射为 ONNX 属性类型，并设置属性的类型
  attr->set_type(ATenAttributeKindToOnnxAttributeType(node->kindOf(name), name));
  
  // 根据属性类型执行相应的操作
  switch (node->kindOf(name)) {
    case AttributeKind::f:
      // 设置属性的浮点值
      attr->set_f(node->f(name));
      break;
    case AttributeKind::fs:
      // 将属性的浮点列表逐个添加到属性中
      for (auto& v : node->fs(name))
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        attr->add_floats(v);
      break;
    case AttributeKind::i:
      // 设置属性的整数值
      attr->set_i(node->i(name));
      break;
    case AttributeKind::is:
      // 将属性的整数列表逐个添加到属性中
      for (auto& v : node->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      // 设置属性的字符串值
      attr->set_s(node->s(name));
      break;
    case AttributeKind::ss:
      // 将属性的字符串列表逐个添加到属性中
      for (auto& v : node->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      // 获取属性的张量，并为其设置名称和编码
      auto t = attr->mutable_t();
      if (use_external_data_format && !t->has_name()) {
        // 如果使用外部数据格式且张量没有名称，则创建张量名称
        t->set_name(createAttributeTensorName(node_proto, t, name, num_external_data_));
      }
      // 编码张量数据
      EncodeTensor(
          t, node->t(name), {}, use_external_data_format, onnx_file_path);
    } break;
    case AttributeKind::ts:
      // 将属性的张量列表逐个编码并添加到属性中
      for (auto& v : node->ts(name)) {
        auto t = attr->add_tensors();
        if (use_external_data_format && !t->has_name()) {
          // 如果使用外部数据格式且张量没有名称，则创建张量名称
          t->set_name(createAttributeTensorName(
              node_proto, t, name, num_external_data_));
        }
        // 编码张量数据
        EncodeTensor(t, v, {}, use_external_data_format, onnx_file_path);
      }
      break;
    // 当属性类型为 ty 时执行以下逻辑
    case AttributeKind::ty: {
      // 设置属性类型为 TYPE_PROTO
      attr->set_type(onnx::AttributeProto_AttributeType_TYPE_PROTO);
      // 获取属性的类型对象
      auto tp = attr->mutable_tp();
      // 获取节点中名称为 name 的类型指针
      const TypePtr& node_type = node->ty(name);
      // 将节点类型编码为 TypeProto，并设置相关的名称
      EncodeTypeProto(
          tp, node_type, node_proto->op_type() + "_" + name.toDisplayString());
    } break;

    // 当属性类型为 tys 时执行以下逻辑
    case AttributeKind::tys: {
      // 设置属性类型为 TYPE_PROTOS
      attr->set_type(onnx::AttributeProto_AttributeType_TYPE_PROTOS);
      // 初始化索引值
      size_t index = 0;
      // 遍历节点中名称为 name 的类型列表
      for (auto& v : node->tys(name)) {
        // 添加一个 TypeProto 对象到属性的 type_protos 列表中
        auto tp = attr->add_type_protos();
        // 将类型 v 编码为 TypeProto，并设置相关的名称和索引
        EncodeTypeProto(
            tp,
            v,
            node_proto->op_type() + "_" + name.toDisplayString() + "_" +
                std::to_string(index));
        index++;
      }
    } break;

    // 当属性类型为 g 时执行以下逻辑
    case AttributeKind::g: {
      // 获取属性的 GraphProto 对象
      auto g = attr->mutable_g();
      // 将节点中名称为 name 的图编码为 GraphProto
      EncodeGraph(
          g,
          node->g(name),
          {},
          {},
          true,
          true,
          use_external_data_format,
          onnx_file_path);
    } break;

    // 当属性类型为 gs 时执行以下逻辑
    case AttributeKind::gs:
      // 遍历节点中名称为 name 的图列表
      for (auto& v : node->gs(name)) {
        // 添加一个 GraphProto 对象到属性的 graphs 列表中
        auto g = attr->add_graphs();
        // 将图 v 编码为 GraphProto
        EncodeGraph(
            g, v, {}, {}, true, true, use_external_data_format, onnx_file_path);
      }
      break;

    // 默认情况下抛出运行时错误，显示属性类型不支持的消息
    default:
      std::ostringstream err_msg;
      err_msg << "attribute \"" << name.toDisplayString()
              << "\" has unexpected kind: " << toString(node->kindOf(name));
      throw std::runtime_error(err_msg.str());
  }
void GraphEncoder::AddAttribute(
    onnx::FunctionProto* func_proto,
    const std::string& name) {
  // 断言函数 proto 非空，确保函数对象有效
  TORCH_INTERNAL_ASSERT(nullptr != func_proto);
  // 向函数 proto 添加一个属性
  func_proto->add_attribute(name);
}

void GraphEncoder::EncodeLocalFunctionOpsetImport(
    onnx::FunctionProto* func_proto,
    const Node* n,
    std::unordered_set<std::string>& custom_domains) {
  // 如果节点不是 ONNX 类型
  if (!n->kind().is_onnx()) {
    std::string domain;
    // 如果节点类型是 aten 或 caffe2
    if (n->kind().is_aten() || n->kind().is_caffe2()) {
      domain = n->kind().domainString(); // 获取域字符串
    } else { // 自定义命名空间和域
      domain = n->kind().ns().toUnqualString();
    }
    domains_.insert(domain); // 将域添加到已知域集合中

    // 如果自定义域集合中不存在当前域
    if (custom_domains.find(domain) == custom_domains.end()) {
      custom_domains.insert(domain); // 将当前域添加到自定义域集合中

      auto* custom_imp = func_proto->add_opset_import(); // 添加操作集导入对象
      custom_imp->set_domain(domain); // 设置自定义域

      // 检查是否已注册域版本，若未注册则设置为版本 1
      auto it = custom_opsets_.find(domain);
      if (it == custom_opsets_.end())
        custom_imp->set_version(1);
      else {
        custom_imp->set_version(it->second); // 设置已注册的域版本
      }
    }
  }

  // 递归处理当前节点的所有子节点
  for (auto* b : n->blocks()) {
    for (auto* sub_n : b->nodes()) {
      EncodeLocalFunctionOpsetImport(func_proto, sub_n, custom_domains);
    }
  }
}

void GraphEncoder::EncodeLocalFunction(
    onnx::GraphProto* graph_proto,
    onnx::FunctionProto* func_proto,
    const Node* n,
    bool add_node_names,
    bool use_external_data_format,
    const std::string& onnx_file_path) {
  const auto fsub_g = n->g(Symbol::attr("graph")); // 获取节点的子图
  func_proto->set_name(n->s(::c10::attr::name)); // 设置函数 proto 的名称

  // 添加子图的输入和输出到函数 proto
  for (auto input : fsub_g->inputs()) {
    func_proto->add_input(input->debugName());
  }
  for (auto output : fsub_g->outputs()) {
    func_proto->add_output(output->debugName());
  }

  // 编码属性名称到函数 proto 中
  if (n->hasAttribute(Symbol::attr("attributes"))) {
    for (const auto& attr_name : n->ss(Symbol::attr("attributes"))) {
      AddAttribute(func_proto, attr_name);
    }
  }

  auto* imp = func_proto->add_opset_import();
  imp->set_version(onnx_opset_version_); // 设置目标 ONNX 操作集的版本

  const auto& domain = n->s(Symbol::attr("domain")); // 获取节点的域
  func_proto->set_domain(domain); // 设置函数 proto 的域
  domains_.insert(domain); // 将域添加到已知域集合中
  std::unordered_set<std::string> custom_domains;

  // 遍历处理子图的所有节点
  for (auto* fsub_n : fsub_g->nodes()) {
    if (fsub_n->mustBeNone()) {
      continue; // 跳过 None 节点
    }
    auto* n_proto = func_proto->add_node(); // 添加节点到函数 proto
    EncodeNode(
        graph_proto,
        n_proto,
        fsub_n,
        add_node_names,
        use_external_data_format,
        onnx_file_path); // 编码节点信息
    EncodeLocalFunctionOpsetImport(func_proto, fsub_n, custom_domains); // 处理节点的操作集导入
  }
}

void GraphEncoder::EncodeTypeProto(
    onnx::TypeProto* type_proto,
    const TypePtr& node_type,
    // 如果节点类型可以转换为 TensorTypePtr 类型
    if (TensorTypePtr tensor_type = node_type->cast<TensorType>()) {
        // 获取节点的类型信息，并将其转换为 onnx::TypeProto_Tensor
        onnx::TypeProto_Tensor* onnx_tensor_type = type_proto->mutable_tensor_type();
        // 调用函数 TensorTypeToONNXType 将 TensorType 转换为对应的 ONNX 类型
        TensorTypeToONNXType(tensor_type, "", name, {}, onnx_tensor_type);
    } else if (ListTypePtr list_type = node_type->cast<ListType>()) {
        // 如果节点类型可以转换为 ListTypePtr 类型
        // 获取节点的类型信息，并将其转换为 onnx::TypeProto_Sequence
        onnx::TypeProto_Sequence* seq_type = type_proto->mutable_sequence_type();
        // 获取列表元素的类型
        auto elem_type = list_type->getElementType();
        // 调用 EncodeTypeProto 函数编码元素类型到 ONNX 类型描述中
        EncodeTypeProto(seq_type->mutable_elem_type(), elem_type, name);
    }
// 编码张量数据到 ONNX 的 TensorProto 对象中
void GraphEncoder::EncodeTensor(
    onnx::TensorProto* tensor_proto,                  // TensorProto 对象，用于存储编码后的张量数据
    const at::Tensor& tensor,                         // 输入的 PyTorch 张量对象
    const std::optional<std::string>& external_ref,   // 可选的外部引用名称，用于延迟导出数据
    const bool use_external_data_format,              // 是否使用外部数据格式
    const std::string& onnx_file_path                 // ONNX 文件路径，用于创建外部文件
) {
  // 将张量的维度信息添加到 TensorProto 对象中
  for (auto d : tensor.sizes()) {
    tensor_proto->add_dims(d);
  }
  // 设置 TensorProto 的数据类型，将 PyTorch 的数据类型转换为 ONNX 的数据类型
  tensor_proto->set_data_type(ATenTypeToOnnxType(tensor.scalar_type()));

  at::Tensor t;
  // 对于量化张量，需要调用 contiguous()，因为 CPU 上的 HalfTensor 没有 contiguous()
  // TODO: 对于量化张量，不在其上调用 .cpu()，因为在某些情况下会失败
  if (tensor.is_quantized()) {
    t = tensor.contiguous();
  } else {
    t = tensor.contiguous().cpu();
  }

  // 检查延迟权重导出设置，确保外部引用和使用外部数据格式不同时开启
  TORCH_INTERNAL_ASSERT(
      !((defer_weight_export_ && external_ref) && use_external_data_format));

  // 如果开启了延迟权重导出且有外部引用，则将数据存储到 raw_data_export_map 中
  // 否则根据 use_external_data_format 的设置将数据存储到 TensorProto 中或外部文件中
  if (defer_weight_export_ && external_ref) {
    // 目前使用张量的名称作为外部查找名称，以避免修改 ONNX protobuf 结构
    TORCH_INTERNAL_ASSERT(external_ref.value() == tensor_proto->name());
    TORCH_INTERNAL_ASSERT(
        raw_data_export_map_.count(external_ref.value()) == 0);
    raw_data_export_map_[external_ref.value()] = t;
    tensor_proto->set_raw_data("__EXTERNAL");
  } else {
    TORCH_INTERNAL_ASSERT(t.is_contiguous());
    size_t tensorSize = static_cast<size_t>(c10::multiply_integers(
        std::begin(tensor.sizes()), std::end(tensor.sizes())));
    // 如果使用外部数据格式并且张量大小超过阈值，则将数据写入外部文件
    if (use_external_data_format &&
        tensorSize > ParamSizeThresholdForExternalStorage) {
      TORCH_INTERNAL_ASSERT(!onnx_file_path.empty());
      TORCH_INTERNAL_ASSERT(tensor_proto->has_name());
      auto tensorName = GetExternalFileName(tensor_proto->name());
      CreateExternalFile(t, tensorName, onnx_file_path);
      // 设置 TensorProto 的数据位置为外部
      onnx::StringStringEntryProto* location =
          tensor_proto->mutable_external_data()->Add();
      location->set_key("location");
      location->set_value(tensorName);
      tensor_proto->set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
    } else {
      // 根据 ONNX ParseData 函数的注释，张量数据始终是小端序的
      tensor_proto->set_raw_data(get_little_endian_data(t));
    }
  }
}

// 编码中间值信息到 ONNX 的 GraphProto 对象中
void GraphEncoder::EncodeIntermediateValueInfo(
    onnx::GraphProto* graph_proto,   // GraphProto 对象，用于存储编码后的图信息
    const Value* v                  // 要编码的中间值
) {
  // 用于 ONNX 本地函数节点的值信息编码动机
  auto n = v->node();
  // 仅对非 ONNX 或非 ATen 节点编码值信息
  if (n->kind().is_onnx() || n->kind().is_aten()) {
    // 略过对 ONNX 或 ATen 节点的值信息编码

auto n = v->node();
// 如果节点是 ONNX 或 ATen 类型的，则跳过对值信息的编码
if (n->kind().is_onnx() || n->kind().is_aten()) {
    return;
  }
  // 如果节点所属图不是当前处理的主图，则不进行值信息的编码
  if (n->owningGraph() != graph_.get()) {
    return;
  }
  // 遍历主图的输出节点，如果当前节点是输出节点，则不编码值信息
  for (const auto* o : graph_->outputs()) {
    if (o == v) {
      return;
    }
  }
  // 向图协议消息中添加值信息
  auto v_info_p = graph_proto->add_value_info();
  // 调用 EncodeValueInfo 函数，将节点 v 的值信息编码并添加到 v_info_p 中
  EncodeValueInfo(graph_proto, v_info_p, v);
} // namespace



std::string pretty_print_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    bool google_printer,
    bool keep_initializers_as_inputs,
    const std::map<std::string, int>& custom_opsets,
    bool add_node_names) {
  auto graph_encoder = GraphEncoder(
      graph,
      onnx_opset_version,
      operator_export_type,
      initializers,
      std::unordered_map<
          std::string,
          std::unordered_map<int64_t, std::string>>{},
      defer_weight_export,
      true,
      keep_initializers_as_inputs,
      custom_opsets,
      add_node_names,
      false,
      std::string());
  if (google_printer) {
    return graph_encoder.get_model_proto()->DebugString();
  }
  return prettyPrint(*graph_encoder.get_model_proto());
}



std::tuple<
    std::shared_ptr<::ONNX_NAMESPACE::ModelProto>,
    RawDataExportMap,
    SymbolDimMap,
    bool,
    NodeNameMap>
export_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    const std::unordered_map<
        std::string,
        std::unordered_map<std::int64_t, std::string>>& dynamic_axes,
    bool defer_weight_export,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    bool strip_doc_string,
    bool keep_initializers_as_inputs,
    const std::map<std::string, int>& custom_opsets,
    bool add_node_names,
    bool use_external_data_format,
    const std::string& onnx_file_path,
    const NodeAttrNameMap& node_attr_to_name) {
  auto graph_encoder = GraphEncoder(
      graph,
      onnx_opset_version,
      operator_export_type,
      initializers,
      dynamic_axes,
      defer_weight_export,
      strip_doc_string,
      keep_initializers_as_inputs,
      custom_opsets,
      add_node_names,
      use_external_data_format,
      onnx_file_path,
      node_attr_to_name);
  GRAPH_DEBUG("onnx proto:", prettyPrint(*graph_encoder.get_model_proto()));
  return std::make_tuple(
      graph_encoder.get_model_proto(),
      graph_encoder.get_raw_data_export_map(),
      graph_encoder.get_symbol_dim_param_map(),
      graph_encoder.get_use_external_data_format(),
      graph_encoder.get_onnx_node_names());
}



std::string serialize_model_proto_to_string(
    const std::shared_ptr<::ONNX_NAMESPACE::ModelProto>& model_proto) {
  return model_proto->SerializeAsString();
}



void check_onnx_proto(const std::string& proto_string) {
  onnx::ModelProto model;
  if (!ParseProtoFromBytes(&model, proto_string.c_str(), proto_string.size())) {
    throw std::runtime_error("Invalid ONNX proto string.");
  }
}
    return;
  }
  // 1. 基线检查
  // 这两个检查防止生成损坏的图，并在发生这种情况时导出错误。
  onnx::checker::check_model(model);
  onnx::shape_inference::InferShapes(model);
  // 2. 完整检查
  // 应用严格模式的形状类型推断检查，检查是否为有效的 ONNX 图。
  // 由于一些用户不需要完全有效的 ONNX 图来运行模型，如果失败，我们只是将此信息作为警告消息添加。
  try {
    auto* schema_registry = onnx::OpSchemaRegistry::Instance();
    // 创建形状推断选项对象，启用类型检查和严格模式
    onnx::ShapeInferenceOptions options{
        /*check_type_val=*/true,
        /*strict_mode_val=*/true};
    // 使用给定的选项进行形状推断
    onnx::shape_inference::InferShapes(model, schema_registry, options);
  } catch (const onnx::InferenceError& ex) {
    // 如果严格的 ONNX 形状推断失败，记录警告信息
    TORCH_WARN(
        "导出的 ONNX 模型未通过 ONNX 形状推断。"
        "该模型将无法在 ONNX Runtime 中执行。"
        "如果这不是您期望的结果，且您认为这是一个 bug，"
        "请在 https://github.com/pytorch/pytorch/issues 报告问题。"
        "严格的 ONNX 形状推断错误报告：",
        ex.what());
  }
}

} // namespace torch::jit
```