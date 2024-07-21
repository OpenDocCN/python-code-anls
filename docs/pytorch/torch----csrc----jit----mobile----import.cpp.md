# `.\pytorch\torch\csrc\jit\mobile\import.cpp`

```
// 包含 Torch 移动端 JIT 的相关头文件
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/parse_operators.h>

// 包含 ATen 库的相关头文件
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>

// 包含 C10 库的相关头文件
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/irange.h>

// 包含 Caffe2 序列化的相关头文件
#include <caffe2/serialize/in_memory_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <caffe2/serialize/read_adapter_interface.h>
#include <caffe2/serialize/versions.h>

// 包含 Torch 移动端 JIT API 的相关头文件
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/mobile/upgrader_mobile.h>

// 包含 Torch JIT 运行时的指令相关头文件
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <torch/csrc/jit/serialization/import_read.h>

// 包含 Torch 自定义类相关头文件
#include <torch/custom_class.h>

// 包含标准库的头文件
#include <string>
#include <vector>

// 下面是一些注释，描述了导入过程中使用的字节码包的结构和示例
// 这些字节码通常存储在 bytecode.pkl 文件中，用于描述移动端模型

// 示例字节码结构:
// (4,  # 模型版本号 (caffe2::serialize::kProducedBytecodeVersion)
//  # 第一个方法
//  (
//   # 函数名
//   '__torch__.m.forward',
//   # 代码
//   (('instructions',
//     (('STOREN', 1, 2),
//      ('DROPR', 1, 0),
//      ('MOVE', 2, 0),
//      ('OP', 0, 0),
//      ('RET', 0, 0))),
//    ('operators', (('aten::Int', 'Tensor'),)),
//    ('constants', ()),
//    ('types', ()),
//    ('register_size', 2)),
//   # schema -- optional (forward-compatible addition to version 4)
//   (('arguments',
//     ((('name', 'x'), ('type', 'Tensor'), ('default_value', 13)),
//      ...)),  # 更多参数在这里
//    ('returns',
//     ((('name', ''), ('type', 'Tensor'), ('default_value', None)),
//      ...)),  # 更多返回值在这里
//   )),
//  # 更多方法在这里
//  ...)

// 另外，模块的调试信息可以保存在 mobile_debug_handles.pkl 中，示例如下:
// (4,
//  ('__torch__.m.forward',
//   (('module_debug_handles', 10))))
//   这里的 10 是调试句柄

// 另外，还可以单独存储和可选地调用 callstack_debug_map
// 这个调试信息序列化了内联的调用堆栈 (InlinedCallStack 数据结构)
// callstack_debug_map 序列化了元组
// (int64_t(debug_handle), int64_t(source_range_tag), InlinedCallStack)
// source_range_tag 映射到 .debug_pkl 文件，其中该标记将其映射到源范围

// InlinedCallStack 序列化为:
// IValue(InlinedCallStack) = {IValue(ModuleInstanceInfo),
// int64_t(source_range_tag), IValue(InlinedCallStack)}

// ModuleInstanceInfo 被序列化为 (class_type_name, instance_name) 的元组
// 注意，当前字节码版本不支持向后兼容性
// This format and process need to be revisited and redesigned if we want to
// support backward compatibility in future.

// Note that the following function-schema fields are not supported:
//  - Argument::{known_length_,kwarg_only_}
//  - FunctionSchema::{overload_name_, is_vararg_, is_varret_}

// 命名空间定义开始，包含了用于解析模型和类的相关功能
namespace torch {
namespace jit {
// 引用命名空间，用于序列化和读取内存适配器
using caffe2::serialize::MemoryReadAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

// 解析操作码的函数声明
OpCode parseOpCode(const char* str);

// 解析移动端类型名，根据特定的前缀判断是否为PyTorch类类型
TypePtr resolveTypeNameMobile(
    const c10::QualifiedName& qn,
    std::shared_ptr<CompilationUnit> compilation_unit) {
  // HACK: first we check whether the name starts with special prefix to
  // tell if it's a supported pytorch class type. There are two special
  // prefixes. "__torch__" for nn module, and "torch.jit" from to_backend.
  // This is a reliable
  // check today, but there is no guarantee that this is the case. The
  // real solution is to merge type parsers so we can share class
  // resolution logic.
  static const c10::QualifiedName torchPrefix = "__torch__";
  static const c10::QualifiedName jitPrefix = "torch.jit";
  if (torchPrefix.isPrefixOf(qn) || jitPrefix.isPrefixOf(qn)) {
    // 如果当前类类型在编译单元中不存在，则创建并注册类类型
    if (compilation_unit->get_class(qn) == nullptr) {
      auto typeptr = ClassType::create(qn, compilation_unit, true);
      compilation_unit->register_type(typeptr);
    }
    // 返回解析后的类类型
    return compilation_unit->get_class(qn);
  } else {
    // 否则，解析并返回通用的C10类型
    return c10::parseType(qn.qualifiedName());
  }
}

// 解析移动端强类型指针
c10::StrongTypePtr typeResolverMobile(
    const c10::QualifiedName& qn,
    const std::shared_ptr<CompilationUnit>& compilation_unit) {
  return c10::StrongTypePtr(
      compilation_unit, resolveTypeNameMobile(qn, compilation_unit));
}

// 加载移动端对象
c10::intrusive_ptr<c10::ivalue::Object> objLoaderMobile(
    const at::StrongTypePtr& type,
    const IValue& input,
    mobile::CompilationUnit& mobile_compilation_unit) {
  auto cls = type.type_->expect<at::ClassType>();
  auto qn = cls->name();
  // 构造方法名
  c10::QualifiedName method_name(qn.value(), "__setstate__");
  // 查找是否存在指定的设置状态函数
  auto setstate = mobile_compilation_unit.find_function(method_name);
  // 查找自定义类并尝试获取其设置状态方法
  auto find_custom_class_with_setstate = [&qn]() -> c10::ClassTypePtr {
    auto custom_class_type = torch::jit::getCustomClass(qn->qualifiedName());
    if (custom_class_type && custom_class_type->findMethod("__setstate__")) {
      return custom_class_type;
    }
    return nullptr;
  };
  if (setstate) {
    // 如果存在设置状态方法，则创建对象并运行设置状态
    auto obj = c10::ivalue::Object::create(type, 0);
    Stack stack({obj, input});
    setstate->run(stack);
    return obj;
  } else if (auto custom_class_type = find_custom_class_with_setstate()) {
    // 如果存在自定义类，创建对象并运行其设置状态方法
    auto obj = c10::ivalue::Object::create(
        c10::StrongTypePtr(nullptr, custom_class_type), 1);
    Stack stack({obj, input});
    custom_class_type->getMethod("__setstate__").run(stack);
    return obj;
  } else {
    // 否则，将输入转换为通用字典，并根据其大小创建对象
    auto dict = std::move(input).toGenericDict();
    size_t ndict = dict.size();
    auto obj = c10::ivalue::Object::create(type, ndict);
    # 获取字典的迭代器，指向字典的起始位置
    auto it = dict.begin();
    # 使用范围循环遍历从0到ndict-1的整数序列
    for (const auto i : c10::irange(ndict)) {
      # 在类（cls）中添加或检查属性，属性名从迭代器指向的键转换而来，类型为键的类型
      cls->addOrCheckAttribute(it->key().toStringRef(), it->key().type());
      # 在对象（obj）的第i个槽位设置值，该值从迭代器指向的值获取
      obj->setSlot(i, it->value());
      # 将迭代器向前移动到下一个键值对
      ++it;
    }
    # 返回设置好属性和槽位的对象
    return obj;
  }
// 尝试在类方法注册时注册方法，需要满足参数非空且第一个参数名为 "self"
void tryRegisterMethod(const std::vector<c10::Argument>& args, Function& func) {
    if (args.empty() || args[0].name() != "self") {
        return;  // 如果参数为空或第一个参数不是 "self"，则直接返回
    }

    // 如果第一个参数是一个类类型，且该类未注册过同名方法，则将方法添加到类中
    if (auto cls = args[0].type()->castRaw<ClassType>()) {
        if (C10_UNLIKELY(cls->findMethod(func.name()))) {
            return;  // 如果类中已存在同名方法，则直接返回
        }
        cls->addMethod(&func);  // 否则将方法添加到类中
    }
}

// 用于加载字节码包的反序列化类
class BytecodeDeserializer final {
 public:
  // 构造函数，接受一个 PyTorchStreamReader 的独占指针和模块加载选项
  explicit BytecodeDeserializer(
      std::unique_ptr<PyTorchStreamReader> reader,
      uint64_t module_load_options = 0);
  // 反序列化函数，返回一个 mobile::Module 对象，可指定设备
  mobile::Module deserialize(std::optional<at::Device> device);
  // 另一个反序列化函数，返回一个 mobile::Module 对象，同时接受额外文件映射
  mobile::Module deserialize(
      std::optional<at::Device> device,
      ExtraFilesMap& extra_files);
  // 仅反序列化额外文件，不加载主模块，可指定设备和额外文件映射
  void deserialize_only_extra(
      std::optional<at::Device> device,
      ExtraFilesMap& extra_files);

 private:
  // 解析类型名称，根据限定名返回类型指针
  TypePtr resolveTypeName(const c10::QualifiedName& qn);
  // 初始化升级器，针对 mobile::Function 对象
  void init_upgrader(mobile::Function* function);
  // 解析方法，接受元组元素和调试句柄，将方法添加到 mobile::CompilationUnit
  void parseMethods(
      c10::ivalue::TupleElements&& vals,
      std::optional<c10::ivalue::TupleElements>&& debug_handles,
      mobile::CompilationUnit& mcu);
  // 读取归档数据，根据归档名称返回 c10::IValue
  c10::IValue readArchive(
      const std::string& archive_name,
      std::shared_ptr<mobile::CompilationUnit> mcu);
  // 解析函数模式，根据函数名称、模式表和模型版本，初始化 mobile::Function
  void parseFunctionSchema(
      const std::string& function_name,
      IValue* schemaTable,
      const int64_t& model_version,
      mobile::Function* function);
  std::shared_ptr<CompilationUnit> compilation_unit_;  // 编译单元指针
  std::unordered_set<std::string> imported_libs_;  // 导入的库集合
  std::unique_ptr<PyTorchStreamReader> reader_{};  // PyTorchStreamReader 的独占指针
  std::optional<at::Device> device_;  // 可选的设备描述符
  uint64_t module_load_options_;  // 模块加载选项
  // 从 model.ptl 中的版本或 `.data/version` 计算的运算器版本，用于确定所需运行时的最小版本
  // 如果小于当前运行时版本，则在加载阶段应用升级器
  uint64_t operator_version_;  // 操作符版本
  uint64_t bytecode_version_;  // 字节码版本
};

// BytecodeDeserializer 类的构造函数实现
BytecodeDeserializer::BytecodeDeserializer(
    std::unique_ptr<PyTorchStreamReader> reader,
    uint64_t module_load_options)
    : compilation_unit_(std::make_shared<CompilationUnit>()),
      reader_(std::move(reader)),
      module_load_options_(module_load_options) {}

// 解析类型名称函数的实现，根据限定名返回类型指针
TypePtr BytecodeDeserializer::resolveTypeName(const c10::QualifiedName& qn) {
  return resolveTypeNameMobile(qn, compilation_unit_);
}

// 解析函数模式函数的实现，用于解析函数模式和方法
// 保持在 BytecodeDeserializer 中需要 compilation_unit_，用于函数模式解析
// 以后可能重构，使其不依赖于特定的 BytecodeDeserializer，例如解析其他表
void BytecodeDeserializer::parseFunctionSchema(
    const std::string& function_name,
    IValue* schemaTable,
    const int64_t& model_version,
    mobile::Function* function) {
    const int64_t& model_version,
    mobile::Function* function) {
  // function schema
  // 如果提供了 schemaTable，表示存在函数的参数和返回值定义
  if (schemaTable) { // (schema is optional for back compat)
    // 定义一个 lambda 函数 parseArgList，用于解析参数列表
    auto parseArgList = [this,
                         function](c10::ivalue::TupleElements&& argTables) {
      // 创建一个空的参数列表
      std::vector<c10::Argument> args;
      // 遍历参数表中的每个参数
      for (auto& argTable : argTables) {
        // 获取参数表中的元素列表
        auto argTableElements = std::move(argTable.toTupleRef()).elements();
        // 获取参数名
        auto name =
            expect_field(argTableElements, "name", BYTECODE_INDEX_ARGUMENT_NAME)
                .toStringRef();
        // 解析参数类型，返回类型指针
        c10::TypePtr type = resolveTypeName(
            (expect_field(
                 argTableElements, "type", BYTECODE_INDEX_ARGUMENT_TYPE))
                .toStringRef());
        // 获取默认值，作为 IValue 类型
        IValue default_value = expect_field(
            argTableElements,
            "default_value",
            BYTECODE_INDEX_ARGUMENT_DEFAULT_VALUE);
        // 将解析得到的参数信息加入 args 列表中
        args.emplace_back(
            name,
            std::move(type),
            c10::nullopt /*N*/,
            std::move(default_value));
      }
      // 尝试注册方法（这里假设是将参数信息注册到 function 对象中）
      tryRegisterMethod(args, *function);
      // 返回解析后的参数列表
      return args;
    };
    // 获取 schemaTable 的元素列表
    auto schemaTableElements = std::move(schemaTable->toTupleRef()).elements();
    // 获取参数列表
    auto arg_list = std::move(expect_field(
                                  schemaTableElements,
                                  "arguments",
                                  BYTECODE_INDEX_SCHEMA_ARGUMENTS)
                                  .toTupleRef())
                        .elements();
    // 获取返回值列表
    auto ret_list =
        std::move(
            expect_field(
                schemaTableElements, "returns", BYTECODE_INDEX_SCHEMA_RETURNS)
                .toTupleRef())
            .elements();
    // 创建函数的 FunctionSchema 对象
    c10::FunctionSchema schema(
        function_name,
        "" /*overload_name*/,
        parseArgList(std::move(arg_list)),  // 解析参数列表
        parseArgList(std::move(ret_list)),  // 解析返回值列表
        false /*is_varargs*/,
        false /*is_varret*/);
    // 将 schema 设置到 function 对象中
    function->setSchema(std::move(schema));
  }
}

// 初始化升级程序，将升级的字节码函数追加到给定的函数对象中
void BytecodeDeserializer::init_upgrader(mobile::Function* function) {
  // 遍历升级器字节码列表，将每个字节码函数追加到给定函数对象中
  for (auto& byteCodeFunctionWithOperator : getUpgraderBytecodeList()) {
    function->append_function(byteCodeFunctionWithOperator.function);
  }
}

// 解析方法，从给定的元组元素中提取字节码信息，并存储到移动编译单元中
void BytecodeDeserializer::parseMethods(
    c10::ivalue::TupleElements&& vals,  // 字节码元组元素
    std::optional<c10::ivalue::TupleElements>&& debug_handles,  // 可选的调试句柄元组元素
    mobile::CompilationUnit& mcu  // 移动编译单元引用
) {
  TORCH_CHECK(!vals.empty(), "Bytecode has no elements. ");

  // 将字节码版本初始化为 kProducedBytecodeVersion 引入时的默认版本号
  constexpr uint64_t default_version = 0x3L;
  bytecode_version_ = default_version;
  size_t method_i_start = 0;

  if (vals[0].isInt()) {
    bytecode_version_ = vals[0].toInt();
    method_i_start = 1;
  }

  // 检查字节码版本是否在支持范围内
  TORCH_CHECK(
      caffe2::serialize::kMinSupportedBytecodeVersion <= bytecode_version_ &&
      bytecode_version_ <= caffe2::serialize::kMaxSupportedBytecodeVersion,
      "Lite Interpreter version number does not match. ",
      "The model version must be between ",
      caffe2::serialize::kMinSupportedBytecodeVersion,
      " and ",
      caffe2::serialize::kMaxSupportedBytecodeVersion,
      " but the model version is ",
      bytecode_version_);

  if (debug_handles) {
    // 检查调试信息与字节码值的数量是否匹配
    TORCH_CHECK(
        debug_handles->size() == vals.size(),
        "The numbers of bytecode values and debug info values do not match.");
  }

  // 处理移动模块中的所有方法
  for (const auto i : c10::irange(method_i_start, vals.size())) {
    auto element = std::move(vals[i]);
    auto m_tuple = std::move(element.toTupleRef()).elements();
    const std::string& function_name = m_tuple[0].toStringRef();

    // 确定是否存储函数模式，适用于较旧的文件
    IValue* schemaTable =
        (bytecode_version_ > 0x4L ||
         (bytecode_version_ == 0x4L && m_tuple.size() >= 3))
        ? &m_tuple[2]
        : nullptr;

    // 创建移动函数对象
    auto function =
        std::make_unique<mobile::Function>(c10::QualifiedName(function_name));

    // 提取指令、操作符和常量列表
    auto ins_list =
        std::move(
            expect_field(
                std::move(m_tuple[1]).toTupleRef().elements(),
                "instructions", BYTECODE_INDEX_INSTRUCTION)
                .toTupleRef())
            .elements();

    auto ops_list =
        std::move(
            expect_field(
                std::move(m_tuple[1]).toTupleRef().elements(),
                "operators", BYTECODE_INDEX_OPERATOR)
                .toTupleRef())
            .elements();

    auto consts_list =
        std::move(
            expect_field(
                std::move(m_tuple[1]).toTupleRef().elements(),
                "constants", BYTECODE_INDEX_CONSTANT)
                .toTupleRef())
            .elements();
    // 从 codeTableElements 中获取 "types" 字段对应的值，并转换为元组引用
    auto types_list =
        std::move(expect_field(codeTableElements, "types", BYTECODE_INDEX_TYPE)
                      .toTupleRef())
            .elements();
    
    // 从 codeTableElements 中获取 "register_size" 字段对应的值，并转换为 int64_t
    int64_t register_size =
        expect_field(
            codeTableElements, "register_size", BYTECODE_INDEX_REGISTER_SIZE)
            .toInt();

    // 如果 debug_handles 存在，则将第 i 个元素转换为元组引用的元素列表
    c10::ivalue::TupleElements debug_handles_m_tuple;
    if (debug_handles) {
      debug_handles_m_tuple =
          std::move(std::move((*debug_handles)[i]).toTupleRef()).elements();
    }
    
    // 初始化函数的升级器
    init_upgrader(function.get());

    // 1. 首先处理从模型中获取的所有操作符
    parseOperators(std::move(ops_list), module_load_options_, function.get());

    // 2. 判断是否需要进行升级
    bool use_upgrader =
        (operator_version_ < caffe2::serialize::kProducedFileFormatVersion);

    // 解析指令集
    parseInstructions(
        function_name,
        std::move(ins_list),
        debug_handles_m_tuple,
        function.get());

    // 3. 如果需要升级，将操作符指令从 OP 修改为 CALL 指令（在下一次 PR 中，use_upgrader 将传递给 parseInstruction 函数，执行实际的修改）
    if (use_upgrader) {
      applyUpgrader(function.get(), operator_version_);
    }

    // 解析常量列表
    parseConstants(consts_list, function.get());

    // 解析类型列表
    parseTypes(types_list, function.get());

    // 设置函数的寄存器大小
    function->set_register_size(register_size);

    // 解析函数的模式和架构
    parseFunctionSchema(
        function_name, schemaTable, bytecode_version_, function.get());

    // 注册函数到 mcu
    mcu.register_function(std::move(function));
}
}

// 反序列化函数，仅
    std::unique_ptr<ReadAdapterInterface> rai,
    // 可选参数，用于指定设备类型
    std::optional<c10::Device> device,
    // 额外文件的映射，用于存储额外的文件信息
    ExtraFilesMap& extra_files,
    // 模块加载选项的位掩码
    uint64_t module_load_options) {
  // 获取模块观察器的实例
  auto observer = torch::observerConfig().getModuleObserver();
  // 使用不安全的 std::rand() 函数生成实例键
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  auto instance_key = std::rand();

  // 存储元数据的无序映射
  std::unordered_map<std::string, std::string> metadata_map;
  // 如果存在模块观察器，通知其开始加载模型
  if (observer) {
    observer->onEnterLoadModel(instance_key);
    // 获取默认的额外文件列表，并将其添加到 extra_files 中
    auto defaultExtraFileList = observer->getDefaultExtraFiles();
    for (const auto& fileName : defaultExtraFileList) {
      extra_files.insert(std::make_pair(fileName, ""));
    }
  }

  // 计算模型的大小
  const size_t model_size = rai != nullptr ? rai->size() : 0;
  // 创建 PyTorchStreamReader 对象，并使用 rai 进行初始化
  auto reader = std::make_unique<PyTorchStreamReader>(std::move(rai));
  // 如果设置了解析所有额外文件映射的选项
  if (module_load_options &
      MobileModuleLoadOptions::PARSE_ALL_EXTRA_FILE_MAPS) {
    // 获取所有记录的文件列表
    std::vector<std::string> all_files = reader->getAllRecords();
    // 遍历文件列表，将以 "extra/" 开头的文件添加到 extra_files 中
    for (auto& file_name : all_files) {
      if (file_name.find("extra/") == 0) {
        extra_files[file_name.substr(6)] = "";
      }
    }
  }
  // 使用 BytecodeDeserializer 对象进行反序列化
  BytecodeDeserializer deserializer(std::move(reader), module_load_options);

  // 错误消息字符串
  std::string error_message;
  // 使用 c10::make_scope_exit 创建的 guard 对象，在作用域结束时执行以下操作
  auto guard = c10::make_scope_exit([&]() {
    // 如果没有模块观察器，则直接返回
    if (!observer) {
      return;
    }
    // 仅反序列化额外文件的内容，并从中提取元数据
    deserializer.deserialize_only_extra(device, extra_files);

    // 从额外文件中处理并获取元数据
    metadata_map = observer->processMetadataFromExtra(extra_files);

    // 在模型加载失败时，通知模块观察器
    observer->onFailLoadModel(
        instance_key,
        error_message.empty() ? "Unknown exception" : error_message.c_str(),
        metadata_map);
  });

  try {
    // 调用 deserializer 对象的 deserialize 方法加载模型
    mobile::Module result = deserializer.deserialize(device, extra_files);
    // 如果存在模块观察器
    if (observer) {
      // 将模型名称和模型大小添加到 metadata_map 中
      extra_files.insert(std::make_pair("model_name", result.name()));
      extra_files.insert(
          std::make_pair("model_size", std::to_string(model_size)));
      // 处理额外文件中的元数据并获取
      metadata_map = observer->processMetadataFromExtra(extra_files);
      // 通知模块观察器加载模型完成
      observer->onExitLoadModel(instance_key, metadata_map);
    }
    // 设置模型的元数据
    result.setMetadata(metadata_map);
    // 释放 guard 对象，避免在作用域结束时再次执行操作
    guard.release();
    // 返回加载的模型结果
    return result;
  } catch (c10::Error& error) {
    // 捕获并处理 c10::Error 异常
    error_message = error.what();
    // 抛出捕获的异常以继续传播
    TORCH_RETHROW(error);
  }
} // namespace

// 从内存加载移动端模块，返回一个移动端模块对象
mobile::Module _load_mobile_from_bytes(
    const std::shared_ptr<char>& data,
    size_t size,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  // 检查数据大小是否符合文件格式头部大小
  TORCH_CHECK(size >= kFileFormatHeaderSize, "Format error");
  // 获取数据的文件格式
  auto format = getFileFormat(data.get());
  // 根据文件格式进行处理
  switch (format) {
    case FileFormat::ZipFileFormat: {
      // 使用内存读取适配器创建独特指针
      std::unique_ptr<ReadAdapterInterface> rai =
          std::make_unique<MemoryReadAdapter>(data.get(), size);
      // 调用内部加载移动端实现函数并返回结果
      return _load_for_mobile_impl(
          std::move(rai), device, extra_files, module_load_options);
    }
    case FileFormat::FlatbufferFileFormat: {
      // 解析并初始化移动端模块，返回结果
      return parse_and_initialize_mobile_module(
          data, size, device, &extra_files);
    }
    default: {
      // 如果文件格式不支持，抛出格式错误异常
      TORCH_CHECK(false, "Format error");
    }
  }
}

// namespace结束
} // namespace

// 从输入流加载移动端模块，返回一个移动端模块对象
mobile::Module _load_for_mobile(
    std::istream& in,
    std::optional<at::Device> device) {
  // 初始化额外文件映射
  ExtraFilesMap extra_files;
  // 调用具体的加载移动端函数并返回结果
  return _load_for_mobile(in, device, extra_files);
}

// 从文件名加载移动端模块，返回一个移动端模块对象
mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device) {
  // 初始化额外文件映射
  ExtraFilesMap extra_files;
  // 调用具体的加载移动端函数并返回结果
  return _load_for_mobile(filename, device, extra_files);
}

// 从自定义读取适配器加载移动端模块，返回一个移动端模块对象
mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device) {
  // 初始化额外文件映射
  ExtraFilesMap extra_files;
  // 调用具体的加载移动端函数并返回结果
  return _load_for_mobile(std::move(rai), device, extra_files);
}

// 从输入流加载移动端模块，返回一个移动端模块对象
mobile::Module _load_for_mobile(
    std::istream& in,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  // 如果输入流的文件格式是FlatBuffer格式
  if (getFileFormat(in) == FileFormat::FlatbufferFileFormat) {
    // 获取输入流的内容和大小
    auto [data, size] = get_stream_content(in);
    // 调用从字节加载移动端函数并返回结果
    return _load_mobile_from_bytes(
        data, size, device, extra_files, module_load_options);
  }
  // 使用输入流适配器创建独特指针
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  // 调用内部加载移动端实现函数并返回结果
  auto module = _load_for_mobile_impl(
      std::move(rai), device, extra_files, module_load_options);
  return module;
}

// 从文件名加载移动端模块，返回一个移动端模块对象
mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  // 调用加载移动端函数，使用默认加载选项
  return _load_for_mobile(
      filename, device, extra_files, kDefaultMobileLoadOptions);
}

// 从文件名加载移动端模块，返回一个移动端模块对象
mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  // 获取观察器配置的模块观察器
  auto observer = torch::observerConfig().getModuleObserver();
  // 如果观察器存在，插入模型路径到额外文件映射中
  if (observer) {
    extra_files.insert(std::make_pair("model_path", filename));
  }
  // 获取文件格式
  auto format = getFileFormat(filename);
  
  // 如果文件格式是FlatBuffer格式
  if (format == FileFormat::FlatbufferFileFormat) {
    // 获取文件内容和大小
    auto [data, size] = get_file_content(filename.c_str());
    // 调用从字节加载移动端函数并返回结果
    return _load_mobile_from_bytes(
        data, size, device, extra_files, module_load_options);
  }

  // 使用文件适配器创建独特指针
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  // 调用内部加载移动端实现函数并返回结果
  return _load_for_mobile_impl(
      std::move(rai), device, extra_files, module_load_options);
}
// 加载移动端模型，并根据需要优化非平坦缓冲模型的文件读取
TORCH_API mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  // 调用接口获取读取适配器的内容和大小
  auto [data, size] = get_rai_content(rai.get());
  // 调用函数加载从字节数据中创建的移动端模型
  return _load_mobile_from_bytes(
      data, size, device, extra_files, module_load_options);
}

// 仅加载移动端模型的额外文件
void _load_extra_only_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  // 获取模块观察器实例
  auto observer = torch::observerConfig().getModuleObserver();
  // 生成随机实例键
  auto instance_key = std::rand();
  // 如果观察器存在，则在加载模型时调用观察器的进入加载模型方法
  if (observer) {
    observer->onEnterLoadModel(instance_key);
  }

  // 获取文件格式
  auto format = getFileFormat(filename);
  switch (format) {
    case FileFormat::ZipFileFormat: {
      // 使用文件适配器创建唯一指针，读取文件
      std::unique_ptr<FileAdapter> rai =
          std::make_unique<FileAdapter>(filename);
      // 使用PyTorch的流读取器创建读取器，传递给字节码反序列化器
      auto reader = std::make_unique<PyTorchStreamReader>(std::move(rai));
      BytecodeDeserializer deserializer(std::move(reader));
      // 仅反序列化额外文件
      deserializer.deserialize_only_extra(device, extra_files);
      break;
    }
    case FileFormat::FlatbufferFileFormat: {
      // TODO: 当前flatbuffers实现将总是加载整个模块，包括额外文件。理想情况下，应该只获取额外文件
      load_mobile_module_from_file(filename, c10::nullopt, &extra_files);
      break;
    }
    default: {
      // 抛出格式错误异常
      TORCH_CHECK(false, "Format error");
    }
  }
}

namespace mobile {

// 导出运算符列表
std::set<std::string> _export_operator_list(
    torch::jit::mobile::Module& module) {
  // 创建存储运算符名字的集合
  std::set<std::string> operator_list;
  // 遍历模块的所有方法
  for (Method func : module.get_methods()) {
    // 获取方法的函数对象
    const Function& function = func.function();
    // 获取函数对象的代码
    const auto& code = function.get_code();
    // 获取操作符名字的向量
    std::vector<c10::OperatorName> const& op_names = code.op_names_;
    // 将操作符名字添加到集合中，以避免重复
    for (auto& op_name : op_names) {
      operator_list.insert(toString(op_name));
    }
  }
  // 返回存储唯一运算符名字的集合
  return operator_list;
}

} // namespace mobile
```