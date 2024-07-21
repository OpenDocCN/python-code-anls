# `.\pytorch\torch\csrc\jit\mobile\compatibility\model_compatibility.cpp`

```
// 包含头文件：ATen 库中的 IValue 类定义
#include <ATen/core/ivalue.h>
// 包含头文件：Caffe2 序列化库中的文件适配器定义
#include <caffe2/serialize/file_adapter.h>
// 包含头文件：Caffe2 序列化库中的内联容器定义
#include <caffe2/serialize/inline_container.h>
// 包含头文件：Torch JIT 编译单元的 API 定义
// 在简化 type_resolver 和 obj_loader 后移除
#include <torch/csrc/jit/api/compilation_unit.h>
// 包含头文件：Torch 移动端模型兼容性检查
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
// 包含头文件：Torch 移动端文件格式定义
#include <torch/csrc/jit/mobile/file_format.h>
// 包含头文件：Torch 移动端 flatbuffer 加载器定义
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
// 包含头文件：Torch 移动端导入功能定义
// 在简化 type_resolver 和 obj_loader 后移除
#include <torch/csrc/jit/mobile/import.h>
// 包含头文件：Torch 移动端类型解析器定义
#include <torch/csrc/jit/mobile/type_parser.h>
// 包含头文件：Torch JIT 序列化导入导出常量定义
#include <torch/csrc/jit/serialization/import_export_constants.h>
// 包含头文件：Torch JIT 序列化读取导入定义
#include <torch/csrc/jit/serialization/import_read.h>

// 包含头文件：Caffe2 序列化库中的内存适配器定义
#include <caffe2/serialize/in_memory_adapter.h>
// 包含头文件：标准字符串流定义
#include <sstream>
// 包含头文件：标准字符串定义
#include <string>
// 包含头文件：无序集合定义
#include <unordered_set>
// 包含头文件：向量容器定义
#include <vector>

// 命名空间：c10 命名空间，包含类型解析器 parseType 函数声明
namespace c10 {
    TypePtr parseType(const std::string& pythonStr);
} // namespace c10

// 命名空间：torch 命名空间
namespace torch {
namespace jit {

// 使用语句：使用 Caffe2 序列化库中的文件适配器
using caffe2::serialize::FileAdapter;
// 使用语句：使用 Caffe2 序列化库中的流适配器
using caffe2::serialize::IStreamAdapter;
// 使用语句：使用 Caffe2 序列化库中的 PyTorchStreamReader 类
using caffe2::serialize::PyTorchStreamReader;
// 使用语句：使用 Caffe2 序列化库中的读适配器接口
using caffe2::serialize::ReadAdapterInterface;

// 函数定义：从指定存档名称和流阅读器中读取存档内容并返回 IValue
c10::IValue readArchive(
    const std::string& archive_name,
    PyTorchStreamReader& stream_reader) {
  // 可选设备变量初始化
  std::optional<at::Device> device;
  // 创建共享指针：JIT 编译单元
  std::shared_ptr<CompilationUnit> compilation_unit =
      std::make_shared<CompilationUnit>();

  // TODO (T90180710): 简化 type_resolver 和 obj_loader，从模型获取字节码版本时
  // 定义类型解析器 lambda 函数
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    return typeResolverMobile(qn, compilation_unit);
  };

  // 创建共享指针：移动端 JIT 编译单元
  std::shared_ptr<mobile::CompilationUnit> mobile_compilation_unit =
      std::make_shared<mobile::CompilationUnit>();
  // 定义对象加载器 lambda 函数
  auto obj_loader = [&](const at::StrongTypePtr& type, IValue input) {
    return objLoaderMobile(type, input, *mobile_compilation_unit);
  };

  // 检查是否是字节码张量在常量存档中
  bool bytecode_tensor_in_constants_archive =
      (archive_name == "bytecode" && !isTensorInBytecodeArchive(stream_reader));

  // 调用 readArchiveAndTensors 函数读取存档和张量，并返回结果
  auto ivalues = torch::jit::readArchiveAndTensors(
      archive_name,
      /*pickle_prefix=*/"",
      /*tensor_prefix=*/bytecode_tensor_in_constants_archive ? "constants/" : "",
      type_resolver,
      obj_loader,
      device,
      stream_reader,
      nullptr);
  return ivalues;
}

// 函数定义：从 PyTorchStreamReader 中获取字节码 IValue 数组
std::vector<IValue> get_bytecode_ivalues(PyTorchStreamReader& reader) {
  return std::move(*readArchive("bytecode", reader).toTuple()).elements().vec();
}

/********************** Bytecode **********************/

// 函数声明：获取模型字节码版本号
uint64_t _get_model_bytecode_version(
    const std::vector<IValue>& bytecode_ivalues);

// 函数定义：从输入流中获取模型字节码版本号
static uint64_t _get_model_bytecode_version_from_bytes(char* data, size_t size) {
  // 保存当前流位置
  auto orig_pos = in.tellg();
  // 将流指针移动到流的起始位置
  in.seekg(0, in.beg);
  // 调用 get_stream_content 函数获取流内容数据和大小
  auto [data, size] = get_stream_content(in);
  // 将流指针移动回原来的位置
  in.seekg(orig_pos, in.beg);
  // 返回从字节数据中获取的模型字节码版本号
  return _get_model_bytecode_version_from_bytes(data.get(), size);
}
// 从文件名获取模型字节码版本号
uint64_t _get_model_bytecode_version(const std::string& filename) {
    // 打开给定文件名的输入文件流
    std::ifstream ifile(filename);
    // 调用重载函数，返回输入文件流的字节码版本号
    return _get_model_bytecode_version(ifile);
}

// 从 ReadAdapterInterface 共享指针获取模型字节码版本号
uint64_t _get_model_bytecode_version(
    std::shared_ptr<ReadAdapterInterface> rai) {
    // 获取适配器中的数据内容和大小
    auto [data, size] = get_rai_content(rai.get());
    // 调用函数，返回数据指针和大小的字节码版本号
    return _get_model_bytecode_version_from_bytes(data.get(), size);
}

// 从 Zip 形式的模型文件中获取字节码版本号
static uint64_t _get_model_bytecode_version_zip(
    std::shared_ptr<ReadAdapterInterface> rai) {
    // 检查是否为 Zip 文件
    if (!check_zip_file(rai)) {
        // 如果不是 Zip 文件，抛出错误信息
        TORCH_CHECK(
            false,
            "Failed to open .ptl file please ensure the model was exported for mobile");
    }
    // 创建 PyTorchStreamReader 对象
    PyTorchStreamReader reader(std::move(rai));
    // 调用函数，从读取器获取字节码 IValues，并返回版本号
    auto bytecode_values = get_bytecode_ivalues(reader);
    return _get_model_bytecode_version(bytecode_values);
}

// 从字节数组和大小获取模型字节码版本号
uint64_t _get_model_bytecode_version_from_bytes(char* data, size_t size) {
    // 检查数据指针不为空
    TORCH_CHECK(data != nullptr, "Pointer to bytes is null.");
    // 检查数据大小大于等于文件格式头大小
    TORCH_CHECK(size >= kFileFormatHeaderSize, "Unrecognized data format");
    // 获取文件格式
    auto format = getFileFormat(data);
    // 根据文件格式进行处理
    switch (format) {
        case FileFormat::FlatbufferFileFormat: {
            // 如果是 FlatBuffer 文件格式，返回其字节码版本号
            return get_bytecode_version_from_bytes(data);
        }
        case FileFormat::ZipFileFormat: {
            // 如果是 Zip 文件格式，创建内存读取适配器
            auto rai =
                std::make_unique<caffe2::serialize::MemoryReadAdapter>(data, size);
            // 调用 Zip 版本号获取函数，并返回结果
            auto version = _get_model_bytecode_version_zip(std::move(rai));
            return version;
        }
        default:
            // 未识别的文件格式，抛出错误信息
            TORCH_CHECK(false, "Unrecognized data format");
    }
}

// 从字节码 IValues 获取模型字节码版本号
uint64_t _get_model_bytecode_version(
    const std::vector<IValue>& bytecode_ivalues) {
    // 检查 IValues 非空且第一个元素为整数
    if (!bytecode_ivalues.empty() && bytecode_ivalues[0].isInt()) {
        // 获取整数型模型版本号
        int64_t model_version = bytecode_ivalues[0].toInt();
        // 检查版本号大于 0
        TORCH_CHECK(
            model_version > 0,
            "Expected model bytecode version > 0 got ",
            model_version);
        // 返回版本号的无符号整数形式
        return static_cast<uint64_t>(model_version);
    }
    // 未能获取字节码版本号，抛出错误信息
    TORCH_CHECK(false, "Failed to get bytecode version.");
}

/********************** Operator Version **********************/

// 前向声明获取模型操作符版本号
uint64_t _get_model_operator_version(
    PyTorchStreamReader& reader); // Forward Declare

// 从输入流获取模型操作符版本号
uint64_t _get_model_operator_version(std::istream& in) {
    // 创建输入流适配器
    std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
    // 调用函数，返回操作符版本号
    return _get_model_operator_version(std::move(rai));
}

// 从文件名获取模型操作符版本号
uint64_t _get_model_operator_version(const std::string& filename) {
    // 创建文件适配器
    std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
    // 调用函数，返回操作符版本号
    return _get_model_operator_version(std::move(rai));
}

// 从读取适配器接口获取模型操作符版本号
uint64_t _get_model_operator_version(
    std::shared_ptr<ReadAdapterInterface> rai) {
    // 检查是否为 Zip 文件
    if (!check_zip_file(rai)) {
        // 如果不是 Zip 文件，抛出错误信息
        TORCH_CHECK(
            false,
            "Failed to open .ptl file please ensure the model was exported for mobile");
    }
    // 创建 PyTorchStreamReader 对象
    PyTorchStreamReader reader(std::move(rai));
    // 调用函数，返回操作符版本号
    return _get_model_operator_version(reader);
}

// 从 PyTorchStreamReader 对象获取模型操作符版本号
uint64_t _get_model_operator_version(PyTorchStreamReader& reader) {
    // 调用读取器对象的版本号获取函数，并返回结果
    return reader.version();
}

/********************** Operators and Info **********************/

// 前向声明
// 定义一个函数，用于从字节码的IValue向量中获取模型操作符及其信息的无序映射
std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::vector<IValue> bytecode_ivalues);

// 从输入流中获取模型操作符及其信息的无序映射
std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::istream& in) {
  // 创建一个IStreamAdapter对象，用于适配输入流
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  // 调用上面定义的函数，并返回结果
  return _get_model_ops_and_info(std::move(rai));
}

// 从文件名中获取模型操作符及其信息的无序映射
std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    const std::string& filename) {
  // 创建一个FileAdapter对象，用于适配文件名
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  // 调用上面定义的函数，并返回结果
  return _get_model_ops_and_info(std::move(rai));
}

// 从适配器接口中获取模型操作符及其信息的无序映射
std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::shared_ptr<ReadAdapterInterface> rai) {
  // 检查适配器是否为ZIP文件，若不是则返回空映射并警告
  if (!check_zip_file(rai)) {
    TORCH_WARN("Failed to open zip file for model ops.");
    return std::unordered_map<std::string, OperatorInfo>{};
  }
  // 使用适配器创建PyTorchStreamReader对象
  PyTorchStreamReader reader(std::move(rai));
  // 获取字节码的IValue向量
  auto bytecode_values = get_bytecode_ivalues(reader);
  // 调用上面定义的函数，并返回结果
  return _get_model_ops_and_info(bytecode_values);
}

/* 一个函数，用于获取模型的根（顶级）操作符及其兼容性信息。
 * 这些根操作符可以调用其中的其他操作符（追踪的操作符），
 * 一个根操作符可以根据内部代码路径调用多个不同的追踪操作符。
 * 这些追踪操作符不会被此函数返回。这些操作符被抽象为运行时的实现细节，
 * 追踪操作符本身也可以调用其他操作符，因此在此API中检索它们是困难的，
 * 而且它们在模型运行时版本之间的值是不确定的。
 * 因此，这个API在兼容性用例中可能会产生误报。
 * 模型的所有根操作符都存在于目标运行时中，但不是所有的追踪操作符都存在，
 * 这会阻止模型的运行。
 **/
std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::vector<IValue> bytecode_ivalues) {
  // 定义最小版本号，包含操作符架构信息的版本号应大于等于6
  constexpr uint64_t min_version_with_schema = 6;
  // 如果字节码的版本号小于最小版本号，则发出警告
  if (_get_model_bytecode_version(bytecode_ivalues) < min_version_with_schema) {
    TORCH_WARN(
        "Only models with bytecode version 6 and above contain operator schema information. Please re-export your model to generate it");
  }
  // 初始化结果映射
  std::unordered_map<std::string, OperatorInfo> result;
  // 若字节码为空，则发出警告并返回空映射
  if (bytecode_ivalues.empty()) {
    TORCH_WARN("Failed to get model ops and info.");
    return result;
  }
  // 循环遍历字节码中的所有函数
  for (const auto i : c10::irange(1, bytecode_ivalues.size())) {
    // 获取方法元组
    const auto& method_tuple = bytecode_ivalues.at(i).toTupleRef().elements();
    // 获取操作符元组
    auto operators_tuple = method_tuple.at(1).toTupleRef().elements()[1];
    // 获取操作符
    auto operators = operators_tuple.toTupleRef().elements()[1];
    // 遍历 operators 对象的每一个元素（op_tuple）
    for (auto& op_tuple : operators.toTupleRef().elements()) {
      // 获取当前元素 op_tuple 的元组引用
      const auto& op = op_tuple.toTupleRef().elements();

      // 提取操作符的名称
      std::string op_name = op.at(0).toStringRef();
      // 提取操作符的重载名称
      std::string op_overload_name = op.at(1).toStringRef();
      // 如果存在重载名称，则将其追加到操作符名称中
      if (!op_overload_name.empty()) {
        op_name.append(".");
        op_name.append(op_overload_name);
      }

      // 提取操作符的参数个数（schema size）
      if (op.size() > 2) {
        // 如果元组 op 的长度大于 2，则使用第三个元素作为参数个数，存入 result 中
        result.emplace(op_name, OperatorInfo{(int)op.at(2).toInt()});
      } else { // 如果没有参数信息，则使用默认值存入 result 中
        result.emplace(op_name, OperatorInfo{});
      }
    }
  }
  // 返回最终结果字典 result
  return result;
/********************** Get Type Table **********************/

// 前向声明函数 _get_mobile_model_contained_types，用于获取移动模型中包含的类型集合
std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::vector<IValue>& bytecode_ivalues);

// 从输入流中获取移动模型中包含的类型集合
std::unordered_set<std::string> _get_mobile_model_contained_types(
    std::istream& in) {
  // 创建输入流适配器，用于处理输入流
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  // 调用实际处理函数来获取类型集合
  return _get_mobile_model_contained_types(std::move(rai));
}

// 从文件名中获取移动模型中包含的类型集合
std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::string& filename) {
  // 创建文件适配器，用于处理文件
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  // 调用实际处理函数来获取类型集合
  return _get_mobile_model_contained_types(std::move(rai));
}

// 从读取适配器接口中获取移动模型中包含的类型集合
std::unordered_set<std::string> _get_mobile_model_contained_types(
    std::shared_ptr<ReadAdapterInterface> rai) {
  // 检查是否是有效的 ZIP 文件
  if (!check_zip_file(rai)) {
    // 如果检查失败，抛出异常并显示错误信息
    TORCH_CHECK(
        false,
        "Failed to open .ptl file please ensure the model was exported for mobile");
  }
  // 使用 PyTorch 的流读取器来读取数据
  PyTorchStreamReader reader(std::move(rai));
  // 获取字节码的 IValues
  auto bytecode_values = get_bytecode_ivalues(reader);
  // 调用实际处理函数来获取类型集合
  return _get_mobile_model_contained_types(bytecode_values);
}

// 获取字节码中包含的类型集合，并去重。每个字符串代表一个原子类型，例如 "str", "Tensor" 等。
// 输入格式类似 "Dict[int, Tuple[Tensor, Tensor, Tensor]]"，输出为去重后的类型集合，例如 {Dict, int, Tuple, Tensor}
std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::vector<IValue>& bytecode_ivalues) {
  // 初始化包含的类型集合
  std::unordered_set<std::string> contained_types;
  // 记录已解析的类型名称，用于避免重复解析
  std::unordered_set<std::string> parsed_type_names_records;
  // 遍历字节码 IValues
  for (const auto i : c10::irange(1, bytecode_ivalues.size())) {
    // 获取元组中的方法
    const auto& method_tuple = bytecode_ivalues.at(i).toTupleRef().elements();
    // 获取类型表元组
    auto type_table_tuple =
        method_tuple.at(1).toTupleRef().elements()[BYTECODE_INDEX_TYPE];
    // 获取类型表
    const auto& type_table =
        type_table_tuple.toTupleRef().elements()[1].toTupleRef().elements();

    // 遍历类型表，获取类型名称列表
    std::vector<std::string> type_name_list;
    for (const auto& type_definition : type_table) {
      std::string type_name = type_definition.toStringRef();
      type_name_list.emplace_back(type_name);
    }
    // 使用类型名称列表进行解析
    at::TypeParser parser(type_name_list);
    parser.parseList();
    // 获取解析后的类型集合
    contained_types = parser.getContainedTypes();
  }

  return contained_types;
}

/********************** Compatibility Checker **********************/

// 从输入流中获取模型兼容性信息
ModelCompatibilityInfo ModelCompatibilityInfo::get(std::istream& in) {
  // 创建输入流适配器，用于处理输入流
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  // 调用实际的静态方法来获取模型兼容性信息
  return get(std::move(rai));
}

// 静态方法：从读取适配器接口中获取模型兼容性信息
ModelCompatibilityInfo ModelCompatibilityInfo::get(
    // 创建一个指向 FileAdapter 对象的智能指针，使用给定的文件名初始化它
    std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
    // 调用 get 函数，并将 rai 指针移动给它，函数返回的是 get 函数的结果
    return get(std::move(rai));
}

ModelCompatibilityInfo ModelCompatibilityInfo::get(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai) {
  // 检查给定的读取适配器是否指向一个有效的 ZIP 文件
  if (!check_zip_file(rai)) {
    // 如果检查失败，抛出一个错误，指示无法打开 ZIP 文件用于模型兼容性信息
    TORCH_CHECK(
        false, "Failed to open zip file for model compatibility information");
  }
  // 使用给定的读取适配器创建 PyTorchStreamReader 对象
  PyTorchStreamReader reader(std::move(rai));
  // 从读取器中获取模型的字节码值
  std::vector<IValue> bytecode_values = get_bytecode_ivalues(reader);
  // 获取模型的字节码版本号
  uint64_t model_bytecode_version =
      _get_model_bytecode_version(bytecode_values);
  // 获取模型的操作和信息
  auto model_info = _get_model_ops_and_info(bytecode_values);
  // 获取模型包含的类型集合
  std::unordered_set<std::string> type_table =
      _get_mobile_model_contained_types(bytecode_values);
  // 获取模型的操作符版本号
  uint64_t operator_version = _get_model_operator_version(reader);
  // 返回模型兼容性信息对象，包括字节码版本号、模型信息、类型表和操作符版本号
  return ModelCompatibilityInfo{
      model_bytecode_version, model_info, type_table, operator_version};
}

ModelCompatCheckResult is_compatible(
    RuntimeCompatibilityInfo runtime_info,
    ModelCompatibilityInfo model_info) {
  // 初始化模型兼容性检查结果，状态为 OK，没有错误
  ModelCompatCheckResult result = {ModelCompatibilityStatus::OK, {}};
  // 检查模型的字节码版本是否大于运行时支持的最大字节码版本
  if (model_info.bytecode_version >
      runtime_info.min_max_supported_bytecode_version.second) {
    // 如果大于最大支持版本，将状态设置为 ERROR，记录错误信息
    result.status = ModelCompatibilityStatus::ERROR;
    std::ostringstream s;
    s << "model bytecode version " << model_info.bytecode_version
      << " is greater than the max supported bytecode version in runtime "
      << runtime_info.min_max_supported_bytecode_version.second;
    result.errors.emplace_back(s.str());
  } else if (
      model_info.bytecode_version <
      runtime_info.min_max_supported_bytecode_version.first) {
    // 检查模型的字节码版本是否小于运行时支持的最小字节码版本
    // 如果小于最小支持版本，将状态设置为 ERROR，记录错误信息
    result.status = ModelCompatibilityStatus::ERROR;
    std::ostringstream s;
    s << "model bytecode version " << model_info.bytecode_version
      << " is less than the minimum supported bytecode version in runtime "
      << runtime_info.min_max_supported_bytecode_version.first;
    result.errors.emplace_back(s.str());
  }

  // 获取运行时支持的类型集合
  std::unordered_set<std::string> supported_type = runtime_info.supported_types;

  // 检查模型的类型表中是否包含不受当前运行时支持的类型
  for (const auto& type_name : model_info.type_table) {
    if (supported_type.find(type_name) == supported_type.end()) {
      // 如果模型包含不受支持的类型，将状态设置为 ERROR，记录错误信息
      result.status = ModelCompatibilityStatus::ERROR;
      std::ostringstream s;
      s << "Primitive type: '" << type_name
        << "' is not supported in current runtime";
      result.errors.push_back(s.str());
    }
  }

  // 检查模型的操作符信息是否与当前运行时兼容
  std::unordered_map<std::string, OperatorInfo> operator_info =
      model_info.operator_info;
  for (auto const& op : operator_info) {
    std::string op_name = op.first;
    OperatorInfo model_op_info = op.second;

    // 检查操作符是否在当前运行时中存在
    // 如果不存在，将状态设置为 ERROR，记录错误信息
    // 由于代码截断，无法查看完整的检查操作符的实现
    // 检查运行时信息中是否存在指定操作符的信息
    if (runtime_info.operator_info.find(op_name) ==
        runtime_info.operator_info.end()) {
      // 如果不存在，设置结果状态为错误
      result.status = ModelCompatibilityStatus::ERROR;
      // 创建字符串流对象 s，记录错误信息并加入到结果的错误列表中
      std::ostringstream s;
      s << "Operator '" << op_name << "' missing from runtime (not found)";
      result.errors.push_back(s.str());
    } else {
      // 获取运行时操作符的信息
      OperatorInfo runtime_op_info = runtime_info.operator_info.at(op_name);

      // 如果运行时操作符没有模式信息，说明出现了错误情况，不可用
      if (!runtime_op_info.num_schema_args.has_value()) {
        // 设置结果状态为错误
        result.status = ModelCompatibilityStatus::ERROR;
        // 创建字符串流对象 s，记录错误信息并加入到结果的错误列表中
        std::ostringstream s;
        s << "Operator '" << op_name
          << "' missing from runtime (missing schema)";
        result.errors.push_back(s.str());
      } else {
        // 检查模型操作符是否有模式信息
        if (model_op_info.num_schema_args.has_value() &&
            (model_op_info.num_schema_args.value() >
             runtime_op_info.num_schema_args.value())) {
          // 如果模型中的参数比运行时多，则设置结果状态为错误
          result.status = ModelCompatibilityStatus::ERROR;
          // 创建字符串流对象 s，记录错误信息并加入到结果的错误列表中
          std::ostringstream s;
          s << "Operator schema for '" << op_name << "' has "
            << model_op_info.num_schema_args.value()
            << " args in model but only "
            << runtime_op_info.num_schema_args.value() << " in the runtime";
          result.errors.push_back(s.str());
        }
      }
    }
  }

  // 检查操作符版本是否兼容
  if (model_info.operator_version <
          runtime_info.min_max_supported_opperator_versions.first ||
      model_info.operator_version >
          runtime_info.min_max_supported_opperator_versions.second) {
    // 如果模型操作符版本不在运行时支持的版本范围内，设置结果状态为错误
    result.status = ModelCompatibilityStatus::ERROR;
    // 创建字符串流对象 s，记录错误信息并加入到结果的错误列表中
    std::ostringstream s;
    s << "Model Operator Version " << model_info.operator_version
      << " is not within supported version range of the runtime "
      << runtime_info.min_max_supported_opperator_versions.first << " to "
      << runtime_info.min_max_supported_opperator_versions.second;
    result.errors.push_back(s.str());
  }

  // 返回最终的结果
  return result;
}

} // namespace jit
} // namespace torch
```