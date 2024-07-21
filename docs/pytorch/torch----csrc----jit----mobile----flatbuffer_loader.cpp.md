# `.\pytorch\torch\csrc\jit\mobile\flatbuffer_loader.cpp`

```py
// 如果定义了 FLATBUFFERS_VERSION_MAJOR 宏，则产生编译错误，要求 flatbuffer_loader.h 文件不得包含任何 flatbuffers 头文件
#ifdef FLATBUFFERS_VERSION_MAJOR
#error "flatbuffer_loader.h must not include any flatbuffers headers"
#endif // FLATBUFFERS_VERSION_MAJOR

// 包含标准库头文件
#include <array>          // 提供固定大小数组的支持
#include <istream>        // 提供输入流的支持
#include <memory>         // 提供智能指针等内存管理工具
#include <string>         // 提供字符串相关操作
#include <tuple>          // 提供元组的支持
#include <unordered_map>  // 提供无序映射容器支持
#include <unordered_set>  // 提供无序集合容器支持
#include <utility>        // 提供一般性工具组件
#include <vector>         // 提供动态数组的支持

// 包含 ATen 库的相关头文件
#include <ATen/ATen.h>                     // 提供张量操作支持
#include <ATen/core/dynamic_type.h>        // 提供动态类型支持
#include <ATen/core/ivalue.h>              // 提供 IValue 类型支持
#include <ATen/core/qualified_name.h>      // 提供限定名称支持
#include <c10/core/CPUAllocator.h>         // 提供 CPU 内存分配器支持
#include <c10/core/impl/alloc_cpu.h>       // 提供 CPU 内存分配支持
#include <c10/util/Exception.h>            // 提供异常处理支持
#include <c10/util/Optional.h>             // 提供可选值支持
#include <c10/util/ScopeExit.h>            // 提供作用域退出支持
#include <caffe2/serialize/inline_container.h>  // 提供序列化支持
#include <torch/csrc/jit/mobile/file_format.h>  // 提供移动端文件格式支持
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>  // 提供 FlatBuffer 加载支持
#include <torch/csrc/jit/mobile/function.h>  // 提供函数定义支持
#include <torch/csrc/jit/mobile/import.h>    // 提供导入支持
#include <torch/csrc/jit/mobile/interpreter.h>  // 提供解释器支持
#include <torch/csrc/jit/mobile/module.h>    // 提供模块支持
#include <torch/csrc/jit/mobile/observer.h>  // 提供观察器支持
#include <torch/csrc/jit/mobile/type_parser.h>  // 提供类型解析支持
#include <torch/csrc/jit/runtime/instruction.h>  // 提供指令支持
#include <torch/csrc/jit/serialization/export_bytecode.h>  // 提供字节码导出支持
#include <torch/csrc/jit/serialization/import_export_constants.h>  // 提供导入导出常量支持
#include <torch/csrc/jit/serialization/import_read.h>  // 提供导入读取支持
#include <torch/custom_class.h>             // 提供自定义类支持

// 根据操作系统选择合适的内存分配头文件
#ifdef _WIN32
#include <malloc.h>        // Windows 下的内存分配支持
#else
#include <cstdlib>         // POSIX 系统下的内存分配支持
#endif

// 根据预定义宏选择合适的 FlatBuffer 头文件和命名空间
#if defined(FB_XPLAT_BUILD) || defined(FBCODE_CAFFE2)
#include <torch/csrc/jit/serialization/mobile_bytecode_generated_fbsource.h> // NOLINT
namespace flatbuffers = flatbuffers_fbsource;
#define FLATBUFFERS_MAX_ALIGNMENT FLATBUFFERS_FBSOURCE_MAX_ALIGNMENT
#else
#include <torch/csrc/jit/serialization/mobile_bytecode_generated.h> // NOLINT
#endif

// 声明 torch 和 jit 命名空间
namespace torch {
namespace jit {

// 自定义类的前缀和 torch 相关前缀的字符串视图
static constexpr c10::string_view kCustomClassPrefix =
    "__torch__.torch.classes";
static constexpr c10::string_view kTorchPrefix = "__torch__";
static constexpr c10::string_view kJitPrefix = "torch.jit";
# 定义 FlatbufferLoader 类，用于加载和解析 Flatbuffer 数据
class FlatbufferLoader final {
 public:
  # 构造函数
  FlatbufferLoader();

  # 定义 IValueParser 函数指针类型，用于解析 mobile 序列化的 IValue 数据
  typedef IValue (*IValueParser)(FlatbufferLoader&, const mobile::serialization::IValue&);
  
  # 注册 IValueParser 函数，将解析器与特定的 IValueUnion 类型关联起来
  void registerIValueParser(
      mobile::serialization::IValueUnion ivalue_type,
      IValueParser parser);
  
  # 解析 mobile 模块，返回 mobile::Module 对象
  mobile::Module parseModule(mobile::serialization::Module* module, char* end);

  # 提取 JIT 源码和常量，并存储到 jit_sources 和 constants 中
  void extractJitSourceAndConstants(
      ExtraFilesMap* jit_sources,
      std::vector<IValue>* constants);

  # 定义 TypeResolver 函数指针类型，用于解析类型字符串并返回 TypePtr
  typedef TypePtr (*TypeResolver)(
      const std::string& type_str,
      std::shared_ptr<CompilationUnit> cu);

  # 注册 TypeResolver 函数，用于解析和注册类型解析器
  void internal_registerTypeResolver(TypeResolver type_resolver);

  # 获取指定位置的 IValue 对象的引用
  IValue& getIValue(uint32_t pos) {
    TORCH_CHECK(pos < all_ivalues_.size());
    return all_ivalues_[pos];
  }

  # 获取指定位置的 mobile::Function 指针
  mobile::Function* getFunction(uint32_t pos) {
    return all_functions_[pos];
  }

  # 获取指定位置的 ClassTypePtr 类型
  ClassTypePtr getType(uint32_t pos) {
    TORCH_CHECK(pos < all_types_.size());
    return all_types_[pos];
  }

  # 获取指定索引处的 c10::Storage 对象
  c10::Storage getStorage(uint32_t index);

  # 获取或创建平面缓冲区中类型注解的 TypePtr 对象
  TypePtr getOrCreateTypeAnnotations(const flatbuffers::String* offset);

  # 获取或创建用于 mobile::serialization::Object 的 ClassTypePtr 对象
  ClassTypePtr getOrCreateClassTypeForObject(
      const mobile::serialization::Object* object);

  # 返回当前解析的 Flatbuffer 输入模块的指针
  const mobile::serialization::Module* getCurrentFlatbufferInput() {
    return module_;
  }

  # 设置是否应复制张量内存的标志位
  void setShouldCopyTensorMemory(bool should_copy_tensor_memory) {
    should_copy_tensor_memory_ = should_copy_tensor_memory;
  }

  # mobile 模块的编译单元指针
  std::shared_ptr<mobile::CompilationUnit> mcu_;

  # 编译单元指针
  std::shared_ptr<CompilationUnit> cu_;

 private:
  # 解析 mobile::serialization::IValue 对象，返回 IValue
  IValue parseIValue(const mobile::serialization::IValue* ivalue);

  # 解析 mobile::serialization::Function 对象，返回 unique_ptr<mobile::Function>
  std::unique_ptr<mobile::Function> parseFunction(
      const mobile::serialization::Function* method);

  # 解析并填充指定索引处的 mobile::serialization::IValue 对象
  void parseAndPopulate(
      uint32_t i,
      const mobile::serialization::IValue* ivalue);

  # 存储所有 mobile::Function 对象的映射表
  std::unordered_map<uint32_t, mobile::Function*> all_functions_;

  # 存储所有 ClassTypePtr 类型的向量
  std::vector<ClassTypePtr> all_types_;

  # 存储已初始化类型的索引集合
  std::unordered_set<uint32_t> initialized_types_;

  # 存储类型注解的映射表
  std::unordered_map<const flatbuffers::String*, TypePtr> type_annotations_;

  # 存储是否已加载存储的标志向量
  std::vector<bool> storage_loaded_;

  # 存储 c10::Storage 对象的向量
  std::vector<c10::Storage> storages_;

  # 存储所有 IValue 对象的向量
  std::vector<IValue> all_ivalues_;

  # 存储 mobile::serialization::IValueUnion 类型到 IValueParser 函数指针的映射表
  std::array<
      IValueParser,
      static_cast<uint8_t>(mobile::serialization::IValueUnion::MAX) + 1>
      ivalue_parsers_;

  # 类型解析器函数指针
  TypeResolver type_resolver_ = nullptr;

  # 当前处理的 mobile 模块指针
  mobile::serialization::Module* module_ = nullptr;

  # 指示是否已解析模块的标志位
  bool module_parsed_ = false;

  # 指示是否应复制张量内存的标志位
  bool should_copy_tensor_memory_ = false;

  # mobile 模块中 mobile_ivalue_size_ 之前的元素数量
  uint32_t mobile_ivalue_size_ = 0;
};

# 解析列表类型的函数声明
IValue parseList(
    FlatbufferLoader&,
    const mobile::serialization::IValue& ivalue);

# 解析张量类型的函数声明
IValue parseTensor(
    FlatbufferLoader&,
    const mobile::serialization::IValue& ivalue);

# 解析元组类型的函数声明
IValue parseTuple(
    FlatbufferLoader&,
    const mobile::serialization::IValue& ivalue);

# 解析字典类型的函数声明
IValue parseDict(
    FlatbufferLoader&,
    const mobile::serialization::IValue& ivalue);

# 解析对象类型的函数声明
IValue parseObject(
    FlatbufferLoader&,
    const mobile::serialization::IValue& ivalue);

# 解析整数列表类型的函数声明
IValue parseIntList(
    FlatbufferLoader&,
    const mobile::serialization::IValue& ivalue);
    // 声明一个函数参数，左值引用类型，用于加载 Flatbuffer
    FlatbufferLoader&,
    // 声明一个函数参数，常量引用类型，表示是移动端序列化的值
    const mobile::serialization::IValue& ivalue);
void FlatbufferLoader::registerIValueParser(
    mobile::serialization::IValueUnion ivalue_type,
    IValueParser parser) {
  // 注册指定的 IValue 类型和相应的解析器函数
  ivalue_parsers_[static_cast<uint8_t>(ivalue_type)] = parser;
}

void FlatbufferLoader::internal_registerTypeResolver(
    TypeResolver resolver) {
  // 内部注册类型解析器函数
  type_resolver_ = std::move(resolver);
}

TypePtr resolveType(
    const std::string& type_string,
    std::shared_ptr<CompilationUnit> cu) {
  // 解析给定的类型字符串，返回对应的 TypePtr
  TypePtr type;
  c10::string_view type_str(type_string);
  if (type_str.starts_with(kCustomClassPrefix)) {
    // 如果类型字符串以自定义类前缀开头，获取自定义类的 TypePtr
    type = getCustomClass(type_string);
    TORCH_CHECK(
        type, "The implementation of class ", type_string, " cannot be found.");
  } else if (
      type_str.starts_with(kTorchPrefix) || type_str.starts_with(kJitPrefix)) {
    // 如果类型字符串以 Torch 或 JIT 前缀开头，创建对应的 ClassType 或获取已注册的 ClassType
    c10::QualifiedName qn(type_string);
    if (cu->get_class(qn) == nullptr) {
      auto classtype = ClassType::create(qn, cu, true);
      cu->register_type(classtype);
      type = classtype;
    } else {
      type = cu->get_class(qn);
    }
  } else {
    // 否则，解析标准类型字符串并返回对应的 TypePtr
    type = c10::parseType(type_string);
  }
  return type;
}

FlatbufferLoader::FlatbufferLoader()
    : mcu_(std::make_shared<mobile::CompilationUnit>()),
      cu_(std::make_shared<CompilationUnit>()),
      ivalue_parsers_{nullptr} {
  // 初始化 FlatbufferLoader 对象，设置默认的 CompilationUnit 和注册的 IValue 解析器函数
  registerIValueParser(mobile::serialization::IValueUnion::NONE, &parseBasic);
  registerIValueParser(mobile::serialization::IValueUnion::Int, &parseBasic);
  registerIValueParser(mobile::serialization::IValueUnion::Bool, &parseBasic);
  registerIValueParser(mobile::serialization::IValueUnion::Double, &parseBasic);
  registerIValueParser(
      mobile::serialization::IValueUnion::ComplexDouble, &parseBasic);
  registerIValueParser(
      mobile::serialization::IValueUnion::TensorMetadata, &parseTensor);
  registerIValueParser(mobile::serialization::IValueUnion::String, &parseBasic);
  registerIValueParser(mobile::serialization::IValueUnion::List, &parseList);
  registerIValueParser(
      mobile::serialization::IValueUnion::IntList, &parseIntList);
  registerIValueParser(
      mobile::serialization::IValueUnion::DoubleList, &parseDoubleList);
  registerIValueParser(
      mobile::serialization::IValueUnion::BoolList, &parseBoolList);
  registerIValueParser(mobile::serialization::IValueUnion::Tuple, &parseTuple);
  registerIValueParser(mobile::serialization::IValueUnion::Dict, &parseDict);
  registerIValueParser(
      mobile::serialization::IValueUnion::Object, &parseObject);
  registerIValueParser(mobile::serialization::IValueUnion::Device, &parseBasic);
  registerIValueParser(
      mobile::serialization::IValueUnion::EnumValue, &parseEnum);
  internal_registerTypeResolver(&resolveType);
}
    TypeResolver type_resolver) {
  type_resolver_ = type_resolver;



// 将传入的 type_resolver 参数赋值给成员变量 type_resolver_
type_resolver_ = type_resolver;
}

void parseExtraFilesFromVector(
    const flatbuffers::Vector<flatbuffers::Offset<
        torch::jit::mobile::serialization::ExtraFile>>* files,
    ExtraFilesMap* extra_files) {
  // 遍历传入的文件向量
  for (uint32_t i = 0; i < files->size(); ++i) {
    // 获取当前索引处的额外文件对象指针
    const auto* extra_file = files->Get(i);
    // 将文件名和内容字符串存入额外文件映射中
    (*extra_files)[extra_file->name()->str()] = extra_file->content()->str();
  }
}

void parseExtraFiles(
    mobile::serialization::Module* module,
    ExtraFilesMap& extra_files) {
  // 获取模块中的额外文件偏移量
  auto extra_files_offsets = module->extra_files();
  // 调用解析额外文件向量的函数
  parseExtraFilesFromVector(extra_files_offsets, &extra_files);
}

void FlatbufferLoader::parseAndPopulate(
    uint32_t i,
    const mobile::serialization::IValue* ivalue) {
  // 如果当前 IValue 是函数类型
  if (const auto* func = ivalue->val_as_Function()) {
    // 解析函数并注册到所有函数列表中
    auto func_ptr = parseFunction(func);
    all_functions_[i] = func_ptr.get();
    mcu_->register_function(std::move(func_ptr));
  } else {
    // 解析 IValue 对象
    all_ivalues_[i] = parseIValue(ivalue);
  }
}

mobile::Module FlatbufferLoader::parseModule(
    mobile::serialization::Module* module,
    char* end) {
  // 设置当前模块对象
  module_ = module;
  // 清空数据结构
  all_ivalues_.clear();
  all_types_.clear();
  storages_.clear();
  storage_loaded_.clear();
  module_parsed_ = false;

  // 获取模块中的 IValues 和对象类型
  const auto* ivalues = module->ivalues();
  // 检查 IValues 和对象类型字段是否有效
  TORCH_CHECK(
      ivalues && module->object_types(),
      "Parsing flatbuffer module: Corrupted ivalues/object_types field");
  // 检查 IValues 字段的有效性
  TORCH_CHECK(
      reinterpret_cast<const char*>(ivalues) < end, "Corrupted ivalues field");
  // 检查存储数据大小是否合法
  TORCH_CHECK(
      module->storage_data_size() >= 0,
      "Parsing flatbuffer module: illegal storage_data_size: ",
      module->storage_data_size(),
      ", expected to be non negative");
  // 调整容器大小
  all_ivalues_.resize(ivalues->size());
  all_types_.resize(module->object_types()->size());
  storages_.resize(module->storage_data_size());
  storage_loaded_.resize(module->storage_data_size(), false);

  // 设置移动 IValue 大小
  mobile_ivalue_size_ = module_->mobile_ivalue_size();
  if (mobile_ivalue_size_ == 0 || mobile_ivalue_size_ > ivalues->size()) {
    mobile_ivalue_size_ = ivalues->size();
  }

  // 遍历所有移动 IValue
  for (uint32_t i = 0; i < mobile_ivalue_size_; i++) {
    const auto* ival = ivalues->Get(i);
    // 检查移动 IValue 对象的有效性
    TORCH_CHECK(
        reinterpret_cast<const char*>(ival) < end, "Corrupted ivalue item")
    // 解析并填充移动 IValue
    parseAndPopulate(i, ival);
  }
  
  // 获取模块状态对象的 IValue 引用
  IValue& module_ivalue = getIValue(module->state_obj());

  // 注册函数
  for (const auto& f : all_functions_) {
    // 获取函数的类索引
    uint32_t class_index =
        ivalues->Get(f.first)->val_as_Function()->class_type();
    // 获取类类型并添加方法
    ClassTypePtr class_type = all_types_[class_index];
    class_type->addMethod(f.second);
  }

  // 标记模块已解析
  module_parsed_ = true;
  // 创建移动模块对象
  auto m = mobile::Module(module_ivalue.toObject(), mcu_);
  m.set_min_operator_version(module->operator_version());
  m.set_bytecode_version(module->bytecode_version());
  return m;
}

void appendUpgraderFunctions(mobile::Function* function) {
  // 如果未禁用升级器功能
#ifndef DISABLE_UPGRADER
  // 遍历获取升级器字节码列表中的函数
  for (auto& byteCodeFunctionWithOperator : getUpgraderBytecodeList()) {
    function->append_function(byteCodeFunctionWithOperator.function);


注释：


    // 将 byteCodeFunctionWithOperator 对象中的 function 成员添加到 function 对象的末尾
    function->append_function(byteCodeFunctionWithOperator.function);


这行代码的作用是将 `byteCodeFunctionWithOperator` 对象中的 `function` 成员添加到另一个名为 `function` 的对象的末尾。
#endif
}

// 解析给定的方法（Function）的序列化数据，并返回一个包含该方法信息的 unique_ptr 智能指针
std::unique_ptr<mobile::Function> FlatbufferLoader::parseFunction(
    const mobile::serialization::Function* method) {
  // 使用方法的限定名创建一个 mobile::Function 对象
  auto function = std::make_unique<mobile::Function>(
      c10::QualifiedName(method->qn()->str()));

  // 遍历方法的指令列表，并将每条指令添加到 function 对象中
  for (const auto* inst : *method->instructions()) {
    function->append_instruction(
        static_cast<OpCode>(inst->op()), inst->x(), inst->n());
  }

  // 遍历方法的常量列表，并将每个常量添加到 function 对象中
  for (uint32_t i : *method->constants()) {
    function->append_constant(getIValue(i));
  }

  // 添加升级函数到 function 对象中
  appendUpgraderFunctions(function.get());

  // 2. 判断是否需要升级器
  const uint32_t operator_version = module_->operator_version();
  bool use_upgrader =
      (operator_version < caffe2::serialize::kProducedFileFormatVersion);

  // 遍历方法的操作符列表，并将每个操作符添加到 function 对象中
  for (const auto* op : *method->operators()) {
    std::optional<int> num_args = c10::nullopt;
    if (op->num_args_serialized() > -1) {
      num_args = op->num_args_serialized();
    }
    function->append_operator(
        op->name()->str(), op->overload_name()->str(), num_args);
  }

  // 初始化 function 对象的操作符
  function->initialize_operators(true);

  // 遍历方法的类型注解列表，并将每个类型注解添加到 function 对象中
  for (const auto i : *method->type_annotations()) {
    function->append_type(getOrCreateTypeAnnotations(i));
  }

  // 3. 如果需要升级器，将调用 applyUpgrader 函数来应用升级器功能
  if (use_upgrader) {
#ifndef DISABLE_UPGRADER
    applyUpgrader(function.get(), operator_version);
#endif
  }

  // 设置方法的寄存器大小到 function 对象中
  function->set_register_size(method->register_size());

  // 如果方法有模式定义，尝试解析模式并设置到 function 对象的 schema 中
  if (method->schema()) {
    try {
      auto parseArgList = [this](const auto* args_fb) {
        std::vector<c10::Argument> args;
        for (const auto* arg_tb : *args_fb) {
          IValue default_value = getIValue(arg_tb->default_value());
          TypePtr type_ptr = getOrCreateTypeAnnotations(arg_tb->type());
          auto arg = c10::Argument(
              arg_tb->name()->str(),
              std::move(type_ptr),
              c10::nullopt /*N*/,
              std::move(default_value));
          args.emplace_back(std::move(arg));
        }
        return args;
      };

      // 创建方法的函数模式（FunctionSchema）并设置到 function 对象的 schema 中
      c10::FunctionSchema schema(
          method->qn()->str(),
          "" /*overload_name*/,
          parseArgList(method->schema()->arguments()),
          parseArgList(method->schema()->returns()),
          false /*is_varargs*/,
          false /*is_varret*/);

      function->setSchema(std::move(schema));
    } catch (const c10::Error& e) {
      // 捕获并忽略任何 c10 错误
    }
  }

  // 返回包含解析方法信息的 function 智能指针
  return function;
}

// 解析枚举类型数据的函数
IValue parseEnum(
    FlatbufferLoader& loader,
    // 获取传入的 ivalue 中的枚举值
    const auto* enum_val = ivalue.val_as_EnumValue();
    // 根据枚举值的类型名从 loader 获取或创建类型注解，并转换为 c10::EnumType 类型
    auto enum_type = loader.getOrCreateTypeAnnotations(enum_val->type_name())
                       ->cast<c10::EnumType>();
    // 断言确保 enum_type 不为空，如果为空则输出错误信息
    AT_ASSERT(
        enum_type,
        "Enum with type: " + enum_val->type_name()->str() + " not found.");
    // 获取枚举值的具体值（IValue 对象）
    IValue val = loader.getIValue(enum_val->value());
    // 遍历枚举类型的所有名称和对应的值
    for (const auto& p : enum_type->enumNamesValues()) {
        // 如果找到与传入的枚举值匹配的值，则创建一个 EnumHolder 对象并返回对应的 IValue
        if (p.second == val) {
            auto enum_holder = c10::make_intrusive<at::ivalue::EnumHolder>(
                enum_type, p.first, p.second);
            return IValue(std::move(enum_holder));
        }
    }
    // 如果未找到匹配的枚举值，则输出错误信息
    AT_ASSERT(
        false, "Enum with type: " + enum_val->type_name()->str() + " not found.");
}
// 函数 parseBasic：解析基本数据类型的序列化对象
IValue parseBasic(
    FlatbufferLoader&,
    const mobile::serialization::IValue& ivalue) {
  // 根据序列化对象的值类型进行不同的处理
  switch (ivalue.val_type()) {
    case mobile::serialization::IValueUnion::NONE:
      return {};  // 如果值类型为 NONE，返回空对象
    case mobile::serialization::IValueUnion::Int:
      return ivalue.val_as_Int()->int_val();  // 返回整数值
    case mobile::serialization::IValueUnion::Bool:
      return ivalue.val_as_Bool()->bool_val();  // 返回布尔值
    case mobile::serialization::IValueUnion::Double:
      return ivalue.val_as_Double()->double_val();  // 返回双精度浮点数
    case mobile::serialization::IValueUnion::ComplexDouble: {
      const auto* comp = ivalue.val_as_ComplexDouble();
      return c10::complex<double>(comp->real(), comp->imag());  // 返回复数
    }
    case mobile::serialization::IValueUnion::String:
      return ivalue.val_as_String()->data()->str();  // 返回字符串
    case mobile::serialization::IValueUnion::Device: {
      return c10::Device(ivalue.val_as_Device()->str()->str());  // 返回设备类型
    }
    default:
      return {};  // 默认情况返回空对象
  }
}

// 函数 parseTensorFromMetadata：从元数据解析张量
at::Tensor parseTensorFromMetadata(
    FlatbufferLoader* loader,
    const mobile::serialization::TensorMetadata* tensor_md) {
  // 获取张量的标量类型
  at::ScalarType type = static_cast<at::ScalarType>(tensor_md->scalar_type());
  // 根据标量类型创建张量的选项
  auto options = at::CPU(type).options();
  at::Tensor tensor;
  if (tensor_md->quantized_schema() != nullptr) {
    // 如果张量有量化模式
    const auto* schema = tensor_md->quantized_schema();
    auto qscheme_type = static_cast<at::QScheme>(schema->qscheme());
    switch (qscheme_type) {
      case at::kPerTensorAffine: {
        // 按张量仿射量化模式创建张量
        tensor = at::_empty_affine_quantized(
            {0}, options, schema->scale(), schema->zero_point());
      } break;
      case at::kPerChannelAffineFloatQParams:
      case at::kPerChannelAffine: {
        // 按通道仿射量化模式创建张量
        at::Tensor scales = parseTensorFromMetadata(loader, schema->scales());
        at::Tensor zero_points =
            parseTensorFromMetadata(loader, schema->zero_points());
        tensor = at::_empty_per_channel_affine_quantized(
            {0}, scales, zero_points, schema->axis(), options);
      } break;
      default:
        // 不支持的量化类型抛出错误
        TORCH_CHECK(
            false,
            "Unsupported tensor quantization type in serialization ",
            toString(qscheme_type));
        break;
    }
  } else {
    // 如果没有量化模式，创建一个未初始化的张量
    tensor = at::empty({0}, options);
  }
  at::TensorImpl* impl = tensor.unsafeGetTensorImpl();

  // 设置张量的存储
  c10::Storage storage;
  storage = loader->getStorage(tensor_md->storage_location_index());
  impl->set_storage_keep_dtype(storage);
  impl->set_storage_offset(tensor_md->storage_offset());

  // 设置张量的尺寸和步长
  std::vector<int64_t> size{
      tensor_md->sizes()->begin(), tensor_md->sizes()->end()};
  std::vector<int64_t> stride{
      tensor_md->strides()->begin(), tensor_md->strides()->end()};
  impl->set_sizes_and_strides(size, stride);

  // 在非最小边缘运行时，将张量包装为变量（支持自动求导）
#ifndef MIN_EDGE_RUNTIME
  tensor = autograd::make_variable(tensor, tensor_md->requires_grad());
#endif
  return tensor;
}

// 函数 parseTensor：解析张量
IValue parseTensor(
    FlatbufferLoader& loader,
    // 声明一个名为 parseTensorFromMetadata 的函数，接收一个指向 loader 对象的指针和一个指向 const 类型的 TensorMetadata 对象的指针作为参数，
    // 返回一个 mobile::serialization::IValue 对象的引用
    const mobile::serialization::IValue& ivalue) {
      // 声明并初始化一个指向 mobile::serialization::TensorMetadata 类型的指针 tensor_md，
      // 将 ivalue 中的数据解析为 TensorMetadata，并赋值给 tensor_md
      const mobile::serialization::TensorMetadata* tensor_md =
          ivalue.val_as_TensorMetadata();
      // 调用 parseTensorFromMetadata 函数，传递 loader 对象的地址和 tensor_md 指针作为参数，
      // 返回函数的结果
      return parseTensorFromMetadata(&loader, tensor_md);
    }
}

IValue parseList(
    FlatbufferLoader& loader,
    const mobile::serialization::IValue& ivalue) {
  // 将 ivalue 转换为 List 类型指针
  const mobile::serialization::List* list = ivalue.val_as_List();
  // 创建一个通用的列表 res，元素类型为 AnyType
  auto res = c10::impl::GenericList(AnyType::get());
  // 遍历 list 中的每个元素的索引 i
  for (int i : *list->items()) {
    // 使用 loader 获取索引 i 对应的 IValue，加入 res 列表中
    res.emplace_back(loader.getIValue(i));
  }
  // 获取或创建 list 的类型注解，并设置到 res 的元素类型中
  auto type = loader.getOrCreateTypeAnnotations(list->annotation_str());
  res.unsafeSetElementType(type->containedType(0));
  // 返回结果列表 res
  return res;
}

template <typename T, typename U>
std::vector<T> parseListNative(const U* list) {
  // 断言 list 不为 nullptr
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(list != nullptr);
  // 使用 list 的 items() 构造 T 类型的 vector，并返回
  return {list->items()->begin(), list->items()->end()};
}

IValue parseIntList(
    FlatbufferLoader&,
    const mobile::serialization::IValue& ivalue) {
  // 将 ivalue 转换为 IntList 类型引用
  const auto& list = ivalue.val_as_IntList();
  // 调用 parseListNative<int64_t>() 处理 IntList，并返回结果
  return parseListNative<int64_t>(list);
}

IValue parseDoubleList(
    FlatbufferLoader&,
    const mobile::serialization::IValue& ivalue) {
  // 将 ivalue 转换为 DoubleList 类型引用
  const auto& list = ivalue.val_as_DoubleList();
  // 调用 parseListNative<double>() 处理 DoubleList，并返回结果
  return parseListNative<double>(list);
}

IValue parseBoolList(
    FlatbufferLoader&,
    const mobile::serialization::IValue& ivalue) {
  // 将 ivalue 转换为 BoolList 类型引用
  const auto& list = ivalue.val_as_BoolList();
  // 调用 parseListNative<uint8_t>() 处理 BoolList，并得到结果 res
  std::vector<uint8_t> res = parseListNative<uint8_t>(list);
  // 创建 c10::List<bool> 类型的 boollist
  c10::List<bool> boollist;
  // 将 res 中的每个元素添加到 boollist 中
  for (auto x : res) {
    boollist.push_back(x);
  }
  // 返回 boollist
  return boollist;
}

IValue parseTuple(
    FlatbufferLoader& loader,
    const mobile::serialization::IValue& ivalue) {
  // 将 ivalue 转换为 Tuple 类型引用
  const auto& tuple = ivalue.val_as_Tuple();
  // 创建空的 IValue 向量 res
  std::vector<IValue> res;
  // 遍历 tuple 中的每个元素的索引 i
  for (int i : *tuple->items()) {
    // 使用 loader 获取索引 i 对应的 IValue，加入 res 向量中
    res.emplace_back(loader.getIValue(i));
  }
  // 使用 c10::ivalue::Tuple 创建并返回包含 res 的元组
  return c10::ivalue::Tuple::create(res);
}

IValue parseDict(
    FlatbufferLoader& loader,
    const mobile::serialization::IValue& ivalue) {
  // 将 ivalue 转换为 Dict 类型指针
  const auto* dict = ivalue.val_as_Dict();
  // 创建一个通用的字典 result，键和值的类型都为 AnyType
  auto result = c10::impl::GenericDict(AnyType::get(), AnyType::get());
  // 获取 dict 的键和值
  const auto* keys = dict->keys();
  const auto* values = dict->values();
  // 遍历 dict 中的每个键值对
  for (size_t i = 0; i < keys->size(); ++i) {
    // 获取第 i 个键和值
    uint32_t key = keys->Get(i);
    uint32_t val = values->Get(i);
    // 使用 loader 获取键和值对应的 IValue，并插入或更新到 result 中
    result.insert_or_assign(loader.getIValue(key), loader.getIValue(val));
  }
  // 获取或创建 dict 的类型注解，并设置到 result 的键和值类型中
  auto type = loader.getOrCreateTypeAnnotations(dict->annotation_str());
  result.unsafeSetKeyType(type->containedType(0));
  result.unsafeSetValueType(type->containedType(1));
  // 返回结果字典 result
  return result;
}

ClassTypePtr FlatbufferLoader::getOrCreateClassTypeForObject(
    const mobile::serialization::Object* object) {
  // 获取对象类型的类指针 cls
  auto cls = getType(object->type_index());
  // 获取对象类型的详细信息
  const mobile::serialization::ObjectType* obj_type =
      module_->object_types()->Get(object->type_index());
  // 如果 cls 为空
  if (cls == nullptr) {
    // 获取对象类型名称的字符串视图 qn_str
    c10::string_view qn_str(
        obj_type->type_name()->c_str(), obj_type->type_name()->size());
    // 如果对象类型名称以 kTorchPrefix 或 kJitPrefix 开头
    if (qn_str.starts_with(kTorchPrefix) || qn_str.starts_with(kJitPrefix)) {
      // 创建 QualifiedName 对象 qn
      c10::QualifiedName qn(obj_type->type_name()->str());
      // 从 cu_ 中获取类指针 cls
      cls = cu_->get_class(qn);
      // 如果 cls 为空
      if (cls == nullptr) {
        // 创建新的 ClassType 类对象 cls，并注册到 cu_ 中
        cls = ClassType::create(qn, cu_, true);
        cu_->register_type(cls);
      }
    }
  } else {
    // 如果对象类型为普通类，根据其完整限定名解析并转换为 ClassType
    cls = c10::parseType(std::string(qn_str))->cast<ClassType>();
  }
  // 检查对象的类型索引是否有效
  TORCH_CHECK(object->type_index() < all_ivalues_.size());
  // 将对象的类型与对应的 ClassType 关联存储到 all_types_ 中
  all_types_[object->type_index()] = cls;

  // 如果对象类型为具有字段的类
  if (obj_type->type() == mobile::serialization::TypeType::CLASS_WITH_FIELD) {
    // 遍历对象的属性列表
    for (uint32_t i = 0; i < object->attrs()->size(); i++) {
      // 获取属性的 IValue 表示
      IValue val = getIValue(object->attrs()->Get(i));
      // 使用具体对象字段的类型设置字段的类型
      cls->addAttribute(
          obj_type->attr_names()->Get(i)->str(),
          val.type<c10::DynamicType>());
    }
  }
  // 将对象的类型索引插入到已初始化类型的集合中
  initialized_types_.insert(object->type_index());
}
// 返回解析得到的 ClassType
return cls;
}

IValue parseObject(
    FlatbufferLoader& loader,
    const mobile::serialization::IValue& ivalue) {
  // 从传入的序列化 IValue 中获取对象信息
  const mobile::serialization::Object* object = ivalue.val_as_Object();
  // 断言对象指针不为空，用于调试目的
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(object != nullptr);
  // 获取当前的 Flatbuffer 输入
  const auto* cur_input = loader.getCurrentFlatbufferInput();
  // 根据对象的类型索引获取对象类型信息
  const mobile::serialization::ObjectType* obj_type =
      cur_input->object_types()->Get(object->type_index());
  // 获取或创建对象的类类型
  auto cls = loader.getOrCreateClassTypeForObject(object);
  // 创建一个堆栈对象用于处理对象属性
  Stack stack;
  // 根据对象类型执行不同的操作
  switch (obj_type->type()) {
    case mobile::serialization::TypeType::CLASS_WITH_FIELD: {
      // 对于带字段的类，创建一个包含指定属性数量的 IValue 对象
      auto obj = c10::ivalue::Object::create(
          at::StrongTypePtr(loader.cu_, cls), object->attrs()->size());
      // 遍历对象的属性列表，为对象设置属性值
      for (uint32_t i = 0; i < object->attrs()->size(); i++) {
        // 获取属性对应的 IValue 值
        IValue val = loader.getIValue(object->attrs()->Get(i));
        // 设置对象的属性值
        obj->setSlot(i, std::move(val));
      }
      // 返回创建的对象
      return obj;
    }
    case mobile::serialization::TypeType::CLASS_WITH_SETSTATE: {
      // 对于带有 __setstate__ 方法的类，获取对象状态并执行设置状态函数
      IValue input = loader.getIValue(object->state());
      mobile::Function* setstate = loader.getFunction(object->setstate_func());
      // 创建一个空的对象
      auto obj =
          c10::ivalue::Object::create(at::StrongTypePtr(loader.cu_, cls), 0);
      // 将对象和输入状态压入堆栈
      stack.emplace_back(obj);
      stack.emplace_back(std::move(input));
      // 执行设置状态函数
      setstate->run(stack);
      // 返回创建的对象
      return obj;
    }
    case mobile::serialization::TypeType::CUSTOM_CLASS: {
      // 对于自定义类，获取类的类型信息
      auto custom_class_type =
          torch::jit::getCustomClass(cls->name()->qualifiedName());
      // 获取对象的状态信息
      IValue input = loader.getIValue(object->state());
      // 创建一个包含单个输入参数的对象
      auto obj = c10::ivalue::Object::create(
          c10::StrongTypePtr(nullptr, custom_class_type), 1);
      // 将对象和输入状态压入堆栈
      stack.emplace_back(obj);
      stack.emplace_back(std::move(input));
      // 执行对象的 __setstate__ 方法
      custom_class_type->getMethod("__setstate__").run(stack);
      // 返回创建的对象
      return obj;
    }
    default:
      // 默认情况下，如果未匹配到任何类型，触发断言错误
      AT_ASSERT(false, "need to be object");
  }
}

// 解析给定的序列化 IValue，并返回解析结果
IValue FlatbufferLoader::parseIValue(
    const mobile::serialization::IValue* ivalue) {
  return ivalue_parsers_[static_cast<uint32_t>(ivalue->val_type())](
      *this, *ivalue);
}

// 空函数定义，用于删除操作，不执行任何操作
void deleteNothing2(void*);
void deleteNothing2(void*) {}

// 获取指定索引处的存储对象
c10::Storage FlatbufferLoader::getStorage(uint32_t index) {
  // 检查索引是否有效
  TORCH_CHECK(index < storage_loaded_.size());
  TORCH_CHECK(index < storages_.size());
  // 如果存储对象尚未加载，则进行加载
  if (!storage_loaded_[index]) {
    // 获取模块的存储数据对象
    auto* storage = module_->storage_data()->GetMutableObject(index);
    // 获取存储数据的大小
    size_t size = storage->data()->size();

    at::DataPtr data;
    // 如果需要复制张量内存，则分配新的内存并复制数据
    if (should_copy_tensor_memory_) {
      auto* allocator = at::GetCPUAllocator();
      data = allocator->allocate(size);
      memcpy(data.get(), storage->data()->data(), size);
    } else {
      // 否则，直接使用存储数据的指针，并指定删除函数为空
      void* ptr = static_cast<void*>(storage->mutable_data()->data());
      data = at::DataPtr(ptr, ptr, deleteNothing2, DeviceType::CPU);
    }
    // 创建存储对象并标记为已加载
    storages_[index] =
        c10::Storage(c10::Storage::use_byte_size_t(), size, std::move(data));
    storage_loaded_[index] = true;
  }
  // 返回指定索引处的存储对象
  return storages_[index];
}

// 获取或创建类型注解信息
TypePtr FlatbufferLoader::getOrCreateTypeAnnotations(
    // 在 type_annotations_ 中查找指定的 offset 对应的类型注解
    auto iter = type_annotations_.find(offset);
    // 如果找到了对应的 offset，则返回其关联的类型注解
    if (iter != type_annotations_.end()) {
        return iter->second;
    }
    // 如果未找到对应的 offset，则通过 type_resolver_ 解析 offset 对应的字符串，并生成对应的 TypePtr 类型
    TypePtr type = type_resolver_(offset->str(), cu_);
    // 将新解析得到的类型注解 type 与 offset 关联存储到 type_annotations_ 中
    type_annotations_[offset] = type;
    // 返回新解析得到的类型注解 type
    return type;
} // 关闭命名空间

void FlatbufferLoader::extractJitSourceAndConstants(
    ExtraFilesMap* jit_sources,
    std::vector<IValue>* constants) {
  AT_ASSERT(
      module_parsed_,
      "Need to first parse a flatbuffer file before extracting jit_sources");

  const auto* ivalues = module_->ivalues();
  for (uint32_t i = mobile_ivalue_size_; i < ivalues->size(); i++) {
    const auto* ival = ivalues->Get(i);
    parseAndPopulate(i, ival);
  }
  // 注册函数
  for (const auto& f : all_functions_) {
    if (f.first >= mobile_ivalue_size_) {
      uint32_t class_index =
          ivalues->Get(f.first)->val_as_Function()->class_type();
      ClassTypePtr class_type = all_types_[class_index];
      class_type->addMethod(f.second);
    }
  }
  const auto* jit_constants = module_->jit_constants();
  for (const auto i : c10::irange(jit_constants->size())) {
    constants->emplace_back(getIValue(jit_constants->Get(i)));
  }
  parseExtraFilesFromVector(module_->jit_sources(), jit_sources);
}

namespace { // 匿名命名空间

mobile::Module parse_and_initialize_mobile_module(
    void* data,
    size_t size,
    std::optional<at::Device>,
    ExtraFilesMap* extra_files,
    bool should_copy_tensor_memory) {
  // TODO(T128189662): If not copying, enforce that data is aligned to
  // kFlatbufferDataAlignmentBytes, and add unit tests.

  // 在解析前验证 Flatbuffer 模块的有效性
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t*>(data), size);
  TORCH_CHECK(
      mobile::serialization::VerifyModuleBuffer(verifier),
      "Malformed Flatbuffer module");

  FlatbufferLoader loader;
  loader.setShouldCopyTensorMemory(should_copy_tensor_memory);

  // Flatbuffer 没有提供与缓冲区交互时提供缓冲区大小的方法
  auto* flatbuffer_module = mobile::serialization::GetMutableModule(data);
  auto* end = static_cast<char*>(data) + size;
  mobile::Module m = loader.parseModule(flatbuffer_module, end);
  if (extra_files != nullptr) {
    parseExtraFiles(flatbuffer_module, *extra_files);
  }
  return m;
}

mobile::Module parse_and_initialize_mobile_module(
    std::shared_ptr<char> data,
    size_t size,
    std::optional<at::Device> device,
    ExtraFilesMap* extra_files) {
  mobile::Module m = parse_and_initialize_mobile_module(
      data.get(),
      size,
      device,
      extra_files,
      /*should_copy_tensor_memory=*/false);
  m.set_delete_memory(std::move(data));
  return m;
}

mobile::Module parse_and_initialize_mobile_module_for_jit(
    void* data,
    size_t size,
    ExtraFilesMap& jit_sources,
    std::vector<IValue>& jit_constants,
    std::optional<at::Device> /* 未使用的参数 */,
    bool should_copy_tensor_memory) {
    ExtraFilesMap* extra_files) {
  // 检查模块数据是否包含标识符，用于格式验证
  TORCH_CHECK(
      mobile::serialization::ModuleBufferHasIdentifier(data), "Format error");

  // TODO(T128189662): 强制要求数据对齐到 kFlatbufferDataAlignmentBytes，并添加单元测试。

  // 使用验证器验证 Flatbuffer 模块数据的完整性
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t*>(data), size);
  TORCH_CHECK(
      mobile::serialization::VerifyModuleBuffer(verifier),
      "Malformed Flatbuffer module");

  // 创建 FlatbufferLoader 实例
  FlatbufferLoader loader;
  // 获取可变的 Flatbuffer 模块指针
  auto* flatbuffer_module = mobile::serialization::GetMutableModule(data);
  // 计算数据结尾指针
  auto* end = static_cast<char*>(data) + size;
  // 解析 Flatbuffer 模块数据并返回移动端模块对象
  mobile::Module m = loader.parseModule(flatbuffer_module, end);
  
  // 如果有额外文件映射，则解析额外文件
  if (extra_files != nullptr) {
    parseExtraFiles(flatbuffer_module, *extra_files);
  }

  // 提取 JIT 源码和常量到 jit_sources 和 jit_constants
  loader.extractJitSourceAndConstants(&jit_sources, &jit_constants);
  
  // 返回解析后的移动端模块对象
  return m;
}
}

// 从文件加载移动端模块
mobile::Module load_mobile_module_from_file(
    const std::string& filename,        // 文件名
    std::optional<c10::Device> device,  // 设备可选参数
    ExtraFilesMap* extra_files) {       // 额外文件映射指针
  auto [data, size] = get_file_content(filename.c_str());  // 获取文件内容
  return parse_and_initialize_mobile_module(
      std::move(data), size, device, extra_files);  // 解析和初始化移动端模块
}

// 从流获取字节码版本号
uint64_t get_bytecode_version(std::istream& in) {
  auto [data, size] = get_stream_content(in);  // 获取流内容
  return get_bytecode_version_from_bytes(data.get());  // 获取字节码版本号
}

// 从文件获取字节码版本号
uint64_t get_bytecode_version(const std::string& filename) {
  auto [data, size] = get_file_content(filename.c_str());  // 获取文件内容
  return get_bytecode_version_from_bytes(data.get());  // 获取字节码版本号
}

// 从字节缓冲获取字节码版本号
uint64_t get_bytecode_version_from_bytes(char* flatbuffer_content) {
  TORCH_CHECK(
      mobile::serialization::ModuleBufferHasIdentifier(flatbuffer_content),  // 检查缓冲内容是否包含标识符
      "Format error");  // 如果格式错误则抛出异常
  auto* flatbuffer_module =
      mobile::serialization::GetMutableModule(flatbuffer_content);  // 获取可变模块指针
  return flatbuffer_module->bytecode_version();  // 返回字节码版本号
}

// 从字节缓冲获取模块信息
mobile::ModuleInfo get_module_info_from_flatbuffer(char* flatbuffer_content) {
  auto* ff_module = mobile::serialization::GetMutableModule(flatbuffer_content);  // 获取可变模块指针
  mobile::ModuleInfo minfo;  // 模块信息对象
  minfo.operator_version = ff_module->operator_version();  // 运算符版本号
  minfo.bytecode_version = ff_module->bytecode_version();  // 字节码版本号

  uint32_t mobile_ivalue_size = ff_module->mobile_ivalue_size();  // 移动 IValue 的数量
  if (mobile_ivalue_size == 0) {
    mobile_ivalue_size = ff_module->ivalues()->size();  // 如果数量为零，则获取 IValue 的大小
  }

  std::vector<std::string> type_name_list;  // 类型名称列表
  for (uint32_t i = 0; i < mobile_ivalue_size; i++) {
    const auto* ival = ff_module->ivalues()->Get(i);  // 获取第 i 个 IValue
    if (const auto* func = ival->val_as_Function()) {  // 如果是函数类型
      minfo.function_names.insert(func->qn()->str());  // 插入函数名称
      for (const auto* op : *func->operators()) {  // 遍历函数的操作符
        at::OperatorName opname(op->name()->str(), op->overload_name()->str());  // 操作符名称
        minfo.opname_to_num_args[mobile::operator_str(opname)] =  // 操作符名称到参数个数的映射
            op->num_args_serialized();  // 序列化的参数个数
      }
      for (const auto* type_ann : *func->type_annotations()) {
        type_name_list.push_back(type_ann->str());  // 添加类型注解到类型名称列表
      }
    }
  }
  c10::TypeParser parser(type_name_list);  // 类型解析器
  parser.parseList();  // 解析类型列表
  minfo.type_names = parser.getContainedTypes();  // 获取包含的类型名称
  return minfo;  // 返回模块信息对象
}

// 从流复制并加载移动端模块
mobile::Module load_mobile_module_from_stream_with_copy(
    std::istream& in,                      // 输入流
    std::optional<at::Device> device,       // 设备可选参数
    ExtraFilesMap* extra_files) {           // 额外文件映射指针
  auto [data, size] = get_stream_content(in);  // 获取流内容
  return parse_and_initialize_mobile_module(
      std::move(data), size, device, extra_files);  // 解析和初始化移动端模块
}

// 解析 FlatBuffer 而不创建对象
mobile::Module parse_flatbuffer_no_object(
    std::shared_ptr<char> data,  // 共享指针指向的字节缓冲数据
    size_t size,                  // 缓冲数据大小
  std::optional<at::Device> device) {
  (void)device;
  (void)size;

  // 用于验证 Flatbuffer 模块在解析之前的有效性。
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t*>(data.get()), size);
  // 使用 TORCH_CHECK 确保 Flatbuffer 模块的有效性，否则抛出异常。
  TORCH_CHECK(
      mobile::serialization::VerifyModuleBuffer(verifier),
      "Malformed Flatbuffer module");

  // 获取可变的 Flatbuffer 模块指针。
  auto* flatbuffer_module = mobile::serialization::GetMutableModule(data.get());
  // 创建 FlatbufferLoader 实例用于加载模块。
  FlatbufferLoader loader;
  // 注册自定义的 IValue 解析器，处理 Object 类型的数据。
  loader.registerIValueParser(
      mobile::serialization::IValueUnion::Object,
      +[](FlatbufferLoader& loader,
          const mobile::serialization::IValue& ivalue) {
        // 获取 Object 类型的数据。
        const mobile::serialization::Object* object = ivalue.val_as_Object();
        // 获取或创建用于该 Object 的类类型。
        auto cls = loader.getOrCreateClassTypeForObject(object);
        // 创建一个 c10::IValue::Object 实例。
        auto obj = c10::ivalue::Object::create(
            at::StrongTypePtr(loader.cu_, cls), object->attrs()->size());
        // 遍历 Object 的属性并设置到新创建的对象中。
        for (uint32_t i = 0; i < object->attrs()->size(); i++) {
          IValue val = loader.getIValue(object->attrs()->Get(i));
          obj->setSlot(i, std::move(val));
        }
        return static_cast<c10::IValue>(obj);
      });

  // 计算数据结束的指针。
  auto* end = data.get() + size;
  // 使用 FlatbufferLoader 解析 Flatbuffer 模块，返回 mobile::Module 对象。
  mobile::Module m = loader.parseModule(flatbuffer_module, end);
  // 设置模块在销毁时释放数据内存。
  m.set_delete_memory(std::move(data));
  // 返回解析得到的 mobile::Module 对象。
  return m;
} // 关闭 namespace jit

bool register_flatbuffer_loader() {
    // 注册 flatbuffer 加载器的函数，目前只是简单返回 true
    return true;
}

} // 关闭 namespace torch
```