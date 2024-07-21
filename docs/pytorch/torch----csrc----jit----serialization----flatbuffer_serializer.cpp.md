# `.\pytorch\torch\csrc\jit\serialization\flatbuffer_serializer.cpp`

```
// 包含 Torch 序列化的 FlatBuffer 序列化器头文件
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>

// 如果 FLATBUFFERS_VERSION_MAJOR 已定义，编译错误提示
#ifdef FLATBUFFERS_VERSION_MAJOR
#error "flatbuffer_serializer.h must not include any flatbuffers headers"
#endif // FLATBUFFERS_VERSION_MAJOR

// 包含标准库头文件
#include <fstream>              // 文件流操作
#include <functional>           // 函数对象
#include <stdexcept>            // 标准异常类
#include <string>               // 字符串
#include <unordered_map>        // 无序映射容器
#include <utility>              // 实用工具组件
#include <vector>               // 向量容器

// 包含 Torch 相关头文件
#include <ATen/ATen.h>                          // ATen 张量库
#include <c10/core/CPUAllocator.h>              // CPU 分配器
#include <c10/util/Exception.h>                 // 异常处理
#include <caffe2/serialize/versions.h>          // Caffe2 序列化版本
#include <torch/csrc/jit/mobile/code.h>         // 移动端代码
#include <torch/csrc/jit/mobile/train/export_data.h> // 移动端训练数据导出
#include <torch/csrc/jit/passes/inliner.h>      // 内联器
#include <torch/csrc/jit/runtime/instruction.h> // 运行时指令

// 根据条件包含 FlatBuffer 相关头文件
#if defined(FB_XPLAT_BUILD) || defined(FBCODE_CAFFE2)
#include <torch/csrc/jit/serialization/mobile_bytecode_generated_fbsource.h> // NOLINT
namespace flatbuffers = flatbuffers_fbsource;
#define FLATBUFFERS_MAX_ALIGNMENT FLATBUFFERS_FBSOURCE_MAX_ALIGNMENT
#else
#include <torch/csrc/jit/serialization/mobile_bytecode_generated.h> // NOLINT
#endif

// Torch JIT 命名空间
namespace torch::jit {

// 使用 FlatBufferBuilder
using flatbuffers::FlatBufferBuilder;

// 使用移动端序列化命名空间
using mobile::serialization::CreateArg;
using mobile::serialization::CreateDebugInfo;
using mobile::serialization::CreateDict;
using mobile::serialization::CreateFunctionDirect;
using mobile::serialization::CreateIValue;
using mobile::serialization::CreateList;
using mobile::serialization::CreateModule;
using mobile::serialization::CreateObject;
using mobile::serialization::CreateOperator;
using mobile::serialization::CreateTensorMetadataDirect;
using mobile::serialization::CreateTupleDirect;

// 私有实现细节命名空间
namespace {

// 待移除的 TODO 标记，等待 Caffe2 生产的字节码版本 >= 9 且 FlatBuffer 发布
constexpr uint32_t kMinVersion = 9;

// 在 FlatBuffer 中索引 0 存储 IValue 的 NONE
constexpr int kNoneIndex = 0;

// 获取实际类型的辅助函数
static TypePtr realType(TypePtr type) {
  if (auto dyn = type->castRaw<c10::DynamicType>()) {
    return dyn->fallback();
  } else {
    return type;
  }
}

// 打印类型信息的辅助函数
auto print_type(const c10::Type& t) -> std::optional<std::string> {
  auto namedType = t.cast<c10::NamedType>();
  if (namedType && namedType->name()) {
    return namedType->name().value().qualifiedName();
  }
  if (auto dyn = t.castRaw<c10::DynamicType>()) {
    return dyn->fallback()->annotation_str();
  }
  return c10::nullopt;
}

// FlatBuffer 序列化器类定义
class FlatbufferSerializer {
 public:
  // 默认构造函数
  FlatbufferSerializer() = default;

  // 序列化模块为 FlatBuffer 的方法
  flatbuffers::DetachedBuffer serializeModule(
      const mobile::Module& module,                           // 要序列化的移动端模块
      bool include_tensor_data_in_flatbuffer,                  // 是否包含张量数据在 FlatBuffer 中
      const ExtraFilesMap& extra_files = ExtraFilesMap(),      // 额外文件映射
      const ExtraFilesMap& jit_sources = ExtraFilesMap(),      // JIT 源文件映射
      const std::vector<IValue>& jit_constants = {});         // JIT 常量数据向量

 private:
  // 存储 IValue 并获取索引的模板函数
  template <typename It>
  std::vector<uint32_t> storeIValuesAndGetIndexes(
      flatbuffers::FlatBufferBuilder& fbb,      // FlatBuffer 构建器
      It begin,                                 // 起始迭代器
      It end);                                  // 结束迭代器
};

} // namespace
  return indexes;
}

// 将元组转换为FlatBuffer格式
flatbuffers::Offset<mobile::serialization::Tuple> tupleToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const IValue& tuple);

// 将列表转换为FlatBuffer格式
flatbuffers::Offset<mobile::serialization::List> listToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const IValue& list);

// 将字典转换为FlatBuffer格式
flatbuffers::Offset<mobile::serialization::Dict> dictToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const IValue& list);

// 将对象转换为FlatBuffer格式
flatbuffers::Offset<mobile::serialization::Object> objectToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const IValue& ivalue);

// 将张量元数据转换为FlatBuffer格式
flatbuffers::Offset<mobile::serialization::TensorMetadata> tensorToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const IValue& ivalue);

// 将函数转换为FlatBuffer格式
flatbuffers::Offset<mobile::serialization::Function> functionToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const std::string& qn,
    const mobile::Function& func);

// 将IValue转换为FlatBuffer格式
flatbuffers::Offset<mobile::serialization::IValue> iValueToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const IValue& ivalue);

// 创建FlatBuffer格式的模式对象
flatbuffers::Offset<jit::mobile::serialization::Schema> CreateFBSchema(
    flatbuffers::FlatBufferBuilder& fbb,
    const std::vector<Argument>& args,
    const std::vector<Argument>& returns,
    const c10::TypePrinter& type_printer);

// 将类类型转换为FlatBuffer格式
flatbuffers::Offset<mobile::serialization::ObjectType> classTypeToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const ClassTypePtr& class_ptr);

// 存储IValue并返回其索引
uint32_t storeIValueAndGetIndex(
    flatbuffers::FlatBufferBuilder& fbb,
    const IValue& ivalue);

// 存储函数并返回其索引
uint32_t storeFunctionAndGetIndex(
    flatbuffers::FlatBufferBuilder& fbb,
    const std::string& qn,
    const mobile::Function& function);

// 存储类类型并返回其索引
uint32_t storeClassTypeAndGetIndex(
    flatbuffers::FlatBufferBuilder& fbb,
    const ClassTypePtr& class_type);

// 存储额外文件并返回其偏移量
flatbuffers::Offset<flatbuffers::Vector<
    flatbuffers::Offset<mobile::serialization::ExtraFile>>>
storeExtraFilesAndGetOffset(
    FlatBufferBuilder& fbb,
    const ExtraFilesMap& extra_files);

// 在ivalue_offsets_向量中插入IValue并返回其索引
uint32_t insertIValue(
    flatbuffers::Offset<mobile::serialization::IValue> ivalue) {
  uint32_t size = ivalue_offsets_.size();
  ivalue_offsets_.push_back(ivalue);
  return size;
}

// 存储张量数据的向量
std::vector<at::Tensor> tensor_data_;

// 用于存储memoized存储映射的无序映射
std::unordered_map<const void*, uint32_t> memoized_storage_map_;

// 存储IValue偏移量的向量
std::vector<flatbuffers::Offset<mobile::serialization::IValue>>
    ivalue_offsets_;

// 存储对象类型偏移量的向量
std::vector<flatbuffers::Offset<mobile::serialization::ObjectType>>
    obj_types_offset_;

// 存储从限定名称到序列化类、类型或函数的映射的无序映射
// qualified name to serialized class, type or function
std::unordered_map<std::string, uint32_t> qn_to_serialized_values_;

// 一些IValue对象的缓存
// cache of some ivalues
struct IValueHash {
  size_t operator()(const IValue& val) const {
    return IValue::hash(val);
  }
};

// 用于比较IValue对象是否相等的结构
struct IValueEqual {
  // Copy of this
  // https://www.internalfb.com/code/aros/[3b875bce7ffa2adacdcea9b3e0cb6d304737a193]/xros/third-party/caffe2/caffe2/aten/src/ATen/core/ivalue.cpp?lines=266
    // 自定义的函数对象，用于比较两个 IValue 对象的相等性
    struct IValueEqual {
      // 重载函数调用操作符，接受两个 IValue 对象作为参数
      bool operator()(const IValue& lhs, const IValue& rhs) const {
        // 如果其中一个是 Tensor，则进行指针比较
        if (lhs.isTensor() || rhs.isTensor()) {
          // 如果两个都是 Tensor，则比较它们的地址是否相同
          if (lhs.isTensor() && rhs.isTensor()) {
            return (&lhs.toTensor()) == (&rhs.toTensor());
          }
          return false;  // 如果只有一个是 Tensor，则不相等
        }
        // 否则调用 IValue 的 equals 方法进行比较
        IValue eq = lhs.equals(rhs);
        // 如果 equals 返回的是布尔类型，则直接返回其值
        if (eq.isBool()) {
          return eq.toBool();
        }
        return false;  // 其他情况视为不相等
      }
    };
    
    // 使用自定义的 IValueEqual 函数对象作为键的无序映射表，值类型为 uint32_t
    std::unordered_map<IValue, uint32_t, IValueHash, IValueEqual> cached_ivalues_;
    
    // 指向 mobile::CompilationUnit 类型的常量指针，初始化为 nullptr
    const mobile::CompilationUnit* mcu_ = nullptr;
  创建空向量，用于存储指令向量
  std::vector<mobile::serialization::Instruction> instruction_vector;
  使用代码对象中的指令数量预留空间
  instruction_vector.reserve(code.instructions_.size());
  遍历每个指令，将其操作码、N 值和 X 值添加到指令向量中
  for (const auto& inst : code.instructions_) {
    instruction_vector.emplace_back(inst.op, inst.N, inst.X);
  }

  创建空向量，用于存储运算符向量
  std::vector<flatbuffers::Offset<mobile::serialization::Operator>>
      operator_vector;
  使用代码对象中的运算符名称数量预留空间
  operator_vector.reserve(code.op_names_.size());
  遍历运算符名称的范围
  for (const auto i : c10::irange(code.op_names_.size())) {
    获取运算符名称和相应的输入大小
    const auto& opname = code.op_names_[i];
    const int op_size = code.operator_input_sizes_[i];
    将运算符的名称、重载名称和输入大小添加到运算符向量中
    operator_vector.push_back(CreateOperator(
        fbb,
        fbb.CreateSharedString(opname.name),
        fbb.CreateSharedString(opname.overload_name),
        op_size));
  }

  获取代码对象中的常量向量
  const auto& constants = code.constants_;
  
  创建空向量，用于存储常量索引
  std::vector<uint32_t> constant_indexes;
  预留空间以容纳所有常量的索引
  constant_indexes.reserve(constants.size());
  遍历常量向量，为每个常量存储其值并获取索引，然后添加到常量索引向量中
  for (const auto& constant : constants) {
    constant_indexes.push_back(storeIValueAndGetIndex(fbb, constant));
  }

  定义 Torch 类型的字符串前缀
  static const std::string torch_prefix("__torch__");
  static const std::string class_prefix("__torch__.torch.classes");
  
  创建空向量，用于存储类型偏移量
  std::vector<flatbuffers::Offset<flatbuffers::String>> type_offsets;

  遍历代码对象中的类型指针
  for (const TypePtr& t : code.types_) {
    获取实际类型的注释字符串并添加到类型偏移量向量中
    auto type_str = realType(t)->annotation_str();
    // 如果类型字符串以 torch_prefix 开头
    if (type_str.find(torch_prefix) == 0) {
      // 使用 TORCH_CHECK 进行断言检查，确保 type_str 以 class_prefix 开头
      TORCH_CHECK(
          type_str.find(class_prefix) == 0,
          "__torch__ types other than custom c++ classes (__torch__.torch.classes)"
          "are not supported in lite interpreter. ",
          "Workaround: instead of using arbitrary class type (class Foo()), ",
          "define a pytorch class (class Foo(torch.nn.Module)).");
    }

    // 将 type_str 转换为共享字符串并添加到 type_offsets 向量中
    type_offsets.push_back(fbb.CreateSharedString(type_str));
  }

  // 由于寄存器位置嵌入到字节码中，传递寄存器大小
  auto register_size = static_cast<int>(code.register_size_);

  // 定义 type_printer lambda 函数，用于打印类型信息
  auto type_printer = [&](const c10::Type& t) -> std::optional<std::string> {
    auto namedType = t.cast<c10::NamedType>();
    if (namedType && namedType->name()) {
      return namedType->name().value().qualifiedName();
    }
    if (auto dyn = t.castRaw<c10::DynamicType>()) {
      return dyn->fallback()->annotation_str();
    }
    return c10::nullopt;
  };

  // 初始化 schema_offset 和 class_index
  flatbuffers::Offset<mobile::serialization::Schema> schema_offset = 0;
  uint32_t class_index = 0;
  
  // 如果函数对象 func 有模式信息
  if (func.hasSchema()) {
    const auto& schema = func.getSchema();
    
    // 检查是否存在重载名，移动模块不支持重载
    TORCH_CHECK(
        schema.overload_name().empty(), // @TODO: is this check correct?
        "Overloads are not supported in mobile modules.");
    
    // 检查是否存在 *args，移动模块不支持 Python *args
    TORCH_CHECK(
        !schema.is_vararg(),
        "Python *args are not supported in mobile modules.");
    
    // 检查是否存在 varret，移动模块不支持变量返回值
    TORCH_CHECK(
        !schema.is_varret(),
        "A variable number of return values is not supported in mobile modules.");
    
    // 创建模式的 flatbuffers 偏移量，并使用 type_printer 打印类型信息
    schema_offset =
        CreateFBSchema(fbb, schema.arguments(), schema.returns(), type_printer);
    
    // 获取模式中第一个参数的类类型，并将其存储并获取索引
    auto classtype = schema.arguments()[0].type()->cast<ClassType>();
    class_index = storeClassTypeAndGetIndex(fbb, classtype);
  }

  // 创建调试信息的 flatbuffers 偏移量
  auto debug_info_offset =
      CreateDebugInfo(fbb, fbb.CreateVector(code.debug_handles_));

  // 创建函数对象的 flatbuffers 偏移量，并返回
  auto function_offset = CreateFunctionDirect(
      fbb,
      qn.c_str(),
      &instruction_vector,
      &operator_vector,
      &constant_indexes,
      &type_offsets,
      register_size,
      schema_offset,
      debug_info_offset,
      class_index);
  
  // 返回函数对象的 flatbuffers 偏移量
  return function_offset;
// 创建一个空的 FlatBufferBuilder 对象来构建 Flatbuffer 数据
FlatBufferBuilder fbb;

// 将 module 的编译单元设置为当前处理的模块的编译单元
mcu_ = &module.compilation_unit();

// 插入一个空的 IValue，标记为 NONE 类型，索引为 0
insertIValue(CreateIValue(fbb, mobile::serialization::IValueUnion::NONE, 0));

// 获取 module 中的所有方法
auto methods = module.get_methods();
// 准备存储函数索引的向量
std::vector<uint32_t> functions_index;
functions_index.reserve(methods.size());
// 遍历每个方法，存储函数偏移并记录索引
for (const auto& method : methods) {
  auto func_offset = storeFunctionAndGetIndex(
      fbb, method.function().qualname().qualifiedName(), method.function());
  functions_index.push_back(func_offset);
}

// 创建存储函数索引的 Flatbuffers 向量
auto functions_offset = fbb.CreateVector(functions_index);

// 存储 module 的 _ivalue() 并获取其索引
uint32_t ivalue_index = storeIValueAndGetIndex(fbb, module._ivalue());

// 初始化存储数据偏移的变量
flatbuffers::Offset<flatbuffers::Vector<
    flatbuffers::Offset<mobile::serialization::StorageData>>>
    storage_data_offset = 0;

// 存储额外文件（extra_files）并获取其偏移量
auto extra_files_offset = storeExtraFilesAndGetOffset(fbb, extra_files);

// 存储 JIT 源代码文件（jit_sources）并获取其偏移量
auto jit_source_offset = storeExtraFilesAndGetOffset(fbb, jit_sources);

// 准备存储 JIT 常量的索引
std::vector<uint32_t> jit_constants_indexes;
jit_constants_indexes.reserve(jit_constants.size());
// 遍历 JIT 常量列表，存储每个常量的索引
for (const auto& ival : jit_constants) {
  jit_constants_indexes.emplace_back(storeIValueAndGetIndex(fbb, ival));
}

// 获取 mobile_ivalue_size 的值作为常量
const uint32_t mobile_ivalue_size = ivalue_offsets_.size();

// 获取模块的最小操作符版本号，并转换为 uint32_t 类型
const uint32_t operator_version =
    static_cast<uint32_t>(module.min_operator_version());

// 获取模块的字节码版本号，并确保不低于最小版本号 kMinVersion
uint32_t bytecode_version = static_cast<uint32_t>(module.bytecode_version());
if (bytecode_version < kMinVersion) {
  bytecode_version = kMinVersion;
}

// 注意：存储数据的保存必须是最后执行的操作。
// 如果 include_tensor_data_in_flatbuffer 为 true，则准备存储张量数据的向量
if (include_tensor_data_in_flatbuffer) {
  std::vector<flatbuffers::Offset<mobile::serialization::StorageData>>
      storage_data;
    // 遍历每个张量数据对象
    for (auto td : tensor_data_) {
      // 检查张量数据存储的设备类型是否为 CPU
      if (td.storage().device_type() != DeviceType::CPU) {
        // 如果不是 CPU 设备，则创建一个空张量对象并将其设置为与原始张量相同的存储
        td = at::empty({0}, td.options())
                 .set_(
                     td.storage(),
                     /* storage_offset = */ 0,
                     /* size = */
                     {static_cast<int64_t>(
                         td.storage().nbytes() / td.element_size())},
                     /* stride = */ {1})
                 .cpu();
      }
      // 强制向量对齐
      fbb.ForceVectorAlignment(
          td.storage().nbytes(), sizeof(uint8_t), FLATBUFFERS_MAX_ALIGNMENT);
      // 创建存储数据的 FlatBuffer 偏移量
      auto storage_offset = mobile::serialization::CreateStorageData(
          fbb,
          fbb.CreateVector(
              reinterpret_cast<const uint8_t*>(td.storage().data()),
              td.storage().nbytes()));
      // 将存储数据偏移量添加到存储数据向量中
      storage_data.push_back(storage_offset);
    }
    // 创建存储数据偏移量的 FlatBuffer 向量
    storage_data_offset = fbb.CreateVector(storage_data);
  }

  // 创建模块对象
  auto mod = CreateModule(
      fbb,
      /*bytecode_version=*/bytecode_version,
      extra_files_offset, /* extra_files */
      functions_offset,
      ivalue_index,
      fbb.CreateVector(ivalue_offsets_),
      tensor_data_.size(),
      storage_data_offset,
      fbb.CreateVector(obj_types_offset_),
      jit_source_offset,
      fbb.CreateVector(jit_constants_indexes),
      operator_version,
      mobile_ivalue_size);
  // 完成 FlatBuffer 构建，返回最终的 FlatBuffer 对象
  FinishModuleBuffer(fbb, mod);
  // 释放 FlatBufferBuilder 对象的内存，并返回其指针
  return fbb.Release();
}

flatbuffers::Offset<mobile::serialization::Tuple> FlatbufferSerializer::
    tupleToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& tuple) {
  // 获取元组的元素列表
  const auto& elements = tuple.toTuple()->elements();
  // 将元组的元素转换为索引列表
  std::vector<uint32_t> items =
      storeIValuesAndGetIndexes(fbb, elements.begin(), elements.end());
  // 创建并返回 FlatBuffers 中的 Tuple 对象
  return CreateTupleDirect(fbb, &items);
}

flatbuffers::Offset<mobile::serialization::List> FlatbufferSerializer::listToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const IValue& list) {
  // 获取列表的元素
  const auto& elements = list.toList();
  // 将列表的元素转换为索引列表
  std::vector<uint32_t> items =
      storeIValuesAndGetIndexes(fbb, elements.begin(), elements.end());
  // 创建并返回 FlatBuffers 中的 List 对象
  return CreateList(
      fbb,
      fbb.CreateVector(items),
      fbb.CreateSharedString(
          realType(list.type<c10::Type>())->annotation_str(print_type)));
}

flatbuffers::Offset<mobile::serialization::Dict> FlatbufferSerializer::dictToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const IValue& ivalue) {
  // 获取字典的键值对
  const auto& dict = ivalue.toGenericDict();
  std::vector<uint32_t> keys;
  std::vector<uint32_t> values;
  keys.reserve(dict.size());
  values.reserve(dict.size());
  // 遍历字典，将键和值转换为索引列表
  for (const auto& entry : dict) {
    int key_index = storeIValueAndGetIndex(fbb, entry.key());
    keys.push_back(key_index);
    int value_index = storeIValueAndGetIndex(fbb, entry.value());
    values.push_back(value_index);
  }

  // 创建并返回 FlatBuffers 中的 Dict 对象
  return CreateDict(
      fbb,
      fbb.CreateVector(keys),
      fbb.CreateVector(values),
      fbb.CreateSharedString(
          realType(ivalue.type<c10::Type>())->annotation_str(print_type)));
}

flatbuffers::Offset<mobile::serialization::ObjectType> FlatbufferSerializer::
    classTypeToFB(flatbuffers::FlatBufferBuilder& fbb, const ClassTypePtr& class_ptr) {
  // 初始化类型为 UNSET
  mobile::serialization::TypeType typetype =
      mobile::serialization::TypeType::UNSET;

  // 初始化类的方法名称向量为 0
  flatbuffers::Offset<
      flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>>
      names_offset = 0;

  // 获取类的 __setstate__ 和 __getstate__ 方法
  c10::QualifiedName setstate_name(*class_ptr->name(), "__setstate__");
  c10::QualifiedName getstate_name(*class_ptr->name(), "__getstate__");
  const mobile::Function* setstate = mcu_->find_function(setstate_name);
  const mobile::Function* getstate = mcu_->find_function(getstate_name);

  // 根据方法是否存在设置类型
  if (setstate != nullptr && getstate != nullptr) {
    typetype = mobile::serialization::TypeType::CLASS_WITH_SETSTATE;
  } else if (
      class_ptr->findMethod("__setstate__") &&
      class_ptr->findMethod("__getstate__")) {
    typetype = mobile::serialization::TypeType::CUSTOM_CLASS;
  } else {
    // 如果没有特定方法，则获取类的属性名称列表
    size_t num_attr = class_ptr->numAttributes();
    std::vector<flatbuffers::Offset<flatbuffers::String>> names;
    std::vector<uint32_t> type_index;
    for (size_t i = 0; i < num_attr; ++i) {
      names.push_back(fbb.CreateSharedString(class_ptr->getAttributeName(i)));
    }
    names_offset = fbb.CreateVector(names);
    // 设置类型类型为 CLASS_WITH_FIELD，这是一个枚举值，用于序列化
    typetype = mobile::serialization::TypeType::CLASS_WITH_FIELD;
  }

  // 创建一个字符串偏移量，用于类的完全限定名
  auto name_offset = fbb.CreateString(class_ptr->name()->qualifiedName());
  
  // 调用函数 CreateObjectType 创建一个对象类型，传入名称偏移量、类型类型、和可能的其他参数
  return CreateObjectType(fbb, name_offset, typetype, names_offset);
// 存储函数并获取其索引，如果函数已经存储过，则直接返回存储的索引
uint32_t FlatbufferSerializer::storeFunctionAndGetIndex(
    flatbuffers::FlatBufferBuilder& fbb,
    const std::string& qn,
    const mobile::Function& function) {
  // 查找函数名在映射表中的位置
  auto iter = qn_to_serialized_values_.find(qn);
  // 如果找到了，返回已存储的索引
  if (iter != qn_to_serialized_values_.end()) {
    return iter->second;
  }

  // 创建表示函数的FlatBuffer偏移量
  auto offset = CreateIValue(
      fbb,
      mobile::serialization::IValueUnion::Function,
      functionToFB(fbb, qn, function).Union());

  // 插入偏移量并获取新的索引
  uint32_t index = insertIValue(offset);
  // 将函数名和索引存入映射表
  qn_to_serialized_values_[qn] = index;
  // 返回新插入的索引
  return index;
}

// 存储类类型并获取其索引，如果类类型已经存储过，则直接返回存储的索引
uint32_t FlatbufferSerializer::storeClassTypeAndGetIndex(
    FlatBufferBuilder& fbb,
    const ClassTypePtr& class_ptr) {
  // 获取类类型的完全限定名
  const auto& type_str = class_ptr->name()->qualifiedName();
  // 查找类类型在映射表中的位置
  auto iter = qn_to_serialized_values_.find(type_str);
  // 如果找到了，返回已存储的索引
  if (iter != qn_to_serialized_values_.end()) {
    return iter->second;
  }

  // 将类类型转换为FlatBuffer偏移量
  auto offset = classTypeToFB(fbb, class_ptr);
  // 获取新的类类型偏移量的索引
  uint32_t res = obj_types_offset_.size();
  // 将新的偏移量插入偏移量列表中
  obj_types_offset_.push_back(offset);
  // 将类类型名和索引存入映射表
  qn_to_serialized_values_[type_str] = res;
  // 返回新插入的索引
  return res;
}

// 将对象转换为FlatBuffer表示
flatbuffers::Offset<mobile::serialization::Object> FlatbufferSerializer::
    objectToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue) {
  // 获取对象和对象类型
  auto obj = ivalue.toObject();
  auto type = obj->type();

  // 初始化状态变量和方法索引
  flatbuffers::Offset<flatbuffers::Vector<uint32_t>> attrs = 0;
  uint32_t state_index = 0;
  uint32_t setstate_func_index = 0;
  // 构建__setstate__方法的完全限定名
  const auto qn = type->name()->qualifiedName() + ".__setstate__";
  // 查找对象类型中的__getstate__和__setstate__方法
  auto getstate = type->findMethod("__getstate__");
  auto setstate = type->findMethod("__setstate__");
  if (getstate && setstate) {
    // 调用__getstate__方法获取对象的状态
    auto state = (*getstate)({obj});
    // 存储状态并获取其索引
    state_index = storeIValueAndGetIndex(fbb, state);
    // 查找__setstate__方法在映射表中的索引
    auto func_index = qn_to_serialized_values_.find(qn);
    if (func_index != qn_to_serialized_values_.end()) {
      // 如果找到了，设置__setstate__方法的索引
      setstate_func_index = func_index->second;
    }
  } else {
    // 如果没有找到__getstate__和__setstate__方法，则存储对象的属性
    size_t num_attr = type->numAttributes();
    std::vector<uint32_t> tuple_index;
    for (size_t i = 0; i < num_attr; ++i) {
      tuple_index.push_back(storeIValueAndGetIndex(fbb, obj->getSlot(i)));
    }
    // 创建表示属性的FlatBuffer向量
    attrs = fbb.CreateVector(tuple_index);
  }

  // 存储对象类型并获取其索引
  uint32_t type_index = storeClassTypeAndGetIndex(fbb, type);
  // 创建并返回表示对象的FlatBuffer对象
  return CreateObject(fbb, type_index, state_index, attrs, setstate_func_index);
}
    // 根据张量的量化方案选择不同的处理方式
    switch (tensor.qscheme()) {
      case at::kPerTensorAffine:
        // 如果是每张量仿射量化方案，获取标量和零点
        scale = tensor.q_scale();
        zero_point = tensor.q_zero_point();
        break;
      case at::kPerChannelAffineFloatQParams:
      case at::kPerChannelAffine: {
        // 如果是每通道仿射量化方案，获取每通道的标量和零点，以及通道轴
        scales = tensorToFB(fbb, tensor.q_per_channel_scales());
        zero_points = tensorToFB(fbb, tensor.q_per_channel_zero_points());
        axis = tensor.q_per_channel_axis();
      } break;
      default:
        // 如果量化方案不支持，则抛出错误信息
        TORCH_CHECK(
            false,
            "Unsupported tensor quantization type in serialization ",
            toString(tensor.qscheme()));
        break;
    }

    // 创建量化模式的序列化结构
    qschema_offset = mobile::serialization::CreateQuantizedSchema(
        fbb,
        static_cast<int8_t>(tensor.qscheme()),
        scale,
        zero_point,
        scales,
        zero_points,
        axis);

  }

  // 获取张量的存储地址
  void* addr = storage.unsafeGetStorageImpl();
  uint32_t storage_index = 0;
  // 查找存储地址在memoized_storage_map_中的索引
  auto it = memoized_storage_map_.find(addr);
  if (it != memoized_storage_map_.end()) {
    // 如果地址已经在map中存在，直接使用其索引
    storage_index = it->second;
  } else {
    // 如果地址不存在，将张量存储到tensor_data_中，并记录索引
    storage_index = tensor_data_.size();
    memoized_storage_map_[addr] = storage_index;
    tensor_data_.push_back(tensor);
  }

  // 构造张量的大小和步长信息
  std::vector<int> sizes{tensor.sizes().begin(), tensor.sizes().end()};
  std::vector<int> strides{tensor.strides().begin(), tensor.strides().end()};

  // 创建张量的元数据并直接返回
  return CreateTensorMetadataDirect(
      fbb,
      /* storage_location_index */ storage_index,
      /* scalar_type */ static_cast<int8_t>(tensor.scalar_type()),
      /* int32_t storage_offset */ tensor.storage_offset(),
      /* sizes */ &sizes,
      /* strides */ &strides,
      /* bool requires_grad */ tensor.requires_grad(),
      /* qschema */ qschema_offset);
  // 如果传入的 ivalue 是 None 类型，则返回预定义的 kNoneIndex
  if (ivalue.isNone()) {
    return kNoneIndex;
  }

  try {
    // 尝试在 cached_ivalues_ 中查找当前 ivalue
    auto iter = cached_ivalues_.find(ivalue);
    // 如果找到了，直接返回对应的索引
    if (iter != cached_ivalues_.end()) {
      return iter->second;
    }
  } catch (...) {
    // 如果出现异常，通常是因为 ivalue 不可哈希化或没有适当的 operator==，我们不处理这种情况，跳过哈希处理
  }

  // 将 ivalue 转换成 FlatBuffer，并获取其在 FlatBuffer 中的偏移量
  auto offset = iValueToFB(fbb, ivalue);
  // 将转换后的数据插入到数据结构中，并获取其索引
  uint32_t index = insertIValue(offset);

  try {
    // 将 ivalue 与其索引存入 cached_ivalues_，以便后续快速查找
    cached_ivalues_[ivalue] = index;
  } catch (...) {
    // 如果出现异常，通常是因为 ivalue 不可哈希化或没有适当的 operator==，我们不处理这种情况，跳过哈希处理
  }

  // 返回生成的索引值
  return index;
}

// 将 IValue 转换为 FlatBuffer 中的数据结构
flatbuffers::Offset<mobile::serialization::IValue> FlatbufferSerializer::
    iValueToFB(flatbuffers::FlatBufferBuilder& fbb, const IValue& ivalue) {
  // 引入 mobile::serialization::IValueUnion 命名空间
  using mobile::serialization::IValueUnion;

  // 初始化 ivalue_type 为 NONE 类型，offset 为 0
  IValueUnion ivalue_type = IValueUnion::NONE;
  flatbuffers::Offset<void> offset = 0;

  // 根据 ivalue 的类型进行不同的处理
  if (ivalue.isTensor()) {
    // 如果是 Tensor 类型，将 ivalue_type 设置为 TensorMetadata，并转换为 FlatBuffer 结构
    ivalue_type = IValueUnion::TensorMetadata;
    offset = tensorToFB(fbb, ivalue).Union();
  } else if (ivalue.isTuple()) {
    // 如果是 Tuple 类型，将 ivalue_type 设置为 Tuple，并转换为 FlatBuffer 结构
    ivalue_type = IValueUnion::Tuple;
    offset = tupleToFB(fbb, ivalue).Union();
  } else if (ivalue.isDouble()) {
    // 如果是 Double 类型，将 ivalue_type 设置为 Double，并转换为 FlatBuffer 结构
    ivalue_type = IValueUnion::Double;
    offset = fbb.CreateStruct(mobile::serialization::Double(ivalue.toDouble()))
                 .Union();
  } else if (ivalue.isComplexDouble()) {
    // 如果是 ComplexDouble 类型，将 ivalue_type 设置为 ComplexDouble，并转换为 FlatBuffer 结构
    auto comp = ivalue.toComplexDouble();
    ivalue_type = IValueUnion::ComplexDouble;
    offset = fbb.CreateStruct(mobile::serialization::ComplexDouble(
                                  comp.real(), comp.imag()))
                 .Union();
  } else if (ivalue.isInt()) {
    // 如果是 Int 类型，将 ivalue_type 设置为 Int，并转换为 FlatBuffer 结构
    ivalue_type = IValueUnion::Int;
    offset =
        fbb.CreateStruct(mobile::serialization::Int(ivalue.toInt())).Union();
  } else if (ivalue.isBool()) {
    // 如果是 Bool 类型，将 ivalue_type 设置为 Bool，并转换为 FlatBuffer 结构
    ivalue_type = IValueUnion::Bool;
    offset =
        fbb.CreateStruct(mobile::serialization::Bool(ivalue.toBool())).Union();
  } else if (ivalue.isString()) {
    // 如果是 String 类型，将 ivalue_type 设置为 String，并转换为 FlatBuffer 结构
    ivalue_type = IValueUnion::String;
    offset = mobile::serialization::CreateString(
                 fbb, fbb.CreateSharedString(ivalue.toStringRef()))
                 .Union();
  } else if (ivalue.isGenericDict()) {
    // 如果是 GenericDict 类型，将 ivalue_type 设置为 Dict，并转换为 FlatBuffer 结构
    ivalue_type = IValueUnion::Dict;
    offset = dictToFB(fbb, ivalue).Union();
  } else if (ivalue.isNone()) {
    // 如果是 None 类型，将 ivalue_type 设置为 NONE，并设置 offset 为 0
    ivalue_type = IValueUnion::NONE;
    offset = 0;
  } else if (ivalue.isIntList()) {
    // 如果是 IntList 类型，将 ivalue_type 设置为 IntList，并转换为 FlatBuffer 结构
    ivalue_type = IValueUnion::IntList;
    offset = mobile::serialization::CreateIntList(
                 fbb, fbb.CreateVector(ivalue.toIntVector()))
                 .Union();
  } else if (ivalue.isDoubleList()) {
    // 如果是 DoubleList 类型，将 ivalue_type 设置为 DoubleList，并转换为 FlatBuffer 结构
    offset = mobile::serialization::CreateDoubleList(
                 fbb, fbb.CreateVector(ivalue.toDoubleVector()))
                 .Union();
  } else {
    // 如果是其他未知类型，抛出异常
    throw std::runtime_error("Unsupported IValue type");
  }

  // 创建 mobile::serialization::IValue 的 FlatBuffer 结构，并返回其 offset
  return mobile::serialization::CreateIValue(fbb, ivalue_type, offset);
}
    // 如果 ivalue 是双精度浮点数列表类型
    if (ivalue.isDoubleList()) {
        // 设置 ivalue_type 为双精度浮点数列表类型
        ivalue_type = IValueUnion::DoubleList;
        // 将 ivalue 转换为双精度浮点数列表
        auto doublelist = ivalue.toDoubleList();
        // 创建存储 doublelist 的字节向量
        std::vector<uint8_t> double_vec(doublelist.begin(), doublelist.end());
        // 使用字节向量创建并序列化双精度浮点数列表，获取偏移量
        offset = mobile::serialization::CreateDoubleList(
                     fbb, fbb.CreateVector(double_vec))
                     .Union();
    // 如果 ivalue 是布尔值列表类型
    } else if (ivalue.isBoolList()) {
        // 设置 ivalue_type 为布尔值列表类型
        ivalue_type = IValueUnion::BoolList;
        // 将 ivalue 转换为布尔值列表
        auto boollist = ivalue.toBoolList();
        // 创建存储 boollist 的字节向量
        std::vector<uint8_t> bool_vec(boollist.begin(), boollist.end());
        // 使用字节向量直接创建并序列化布尔值列表，获取偏移量
        offset =
            mobile::serialization::CreateBoolListDirect(fbb, &bool_vec).Union();
    // 如果 ivalue 是通用列表类型
    } else if (ivalue.isList()) {
        // 设置 ivalue_type 为通用列表类型
        ivalue_type = IValueUnion::List;
        // 将通用列表 ivalue 转换为 flatbuffers 的表达形式，并获取偏移量
        offset = listToFB(fbb, ivalue).Union();
    // 如果 ivalue 是对象类型
    } else if (ivalue.isObject()) {
        // 设置 ivalue_type 为对象类型
        ivalue_type = IValueUnion::Object;
        // 将对象 ivalue 转换为 flatbuffers 的表达形式，并获取偏移量
        offset = objectToFB(fbb, ivalue).Union();
    // 如果 ivalue 是设备类型
    } else if (ivalue.isDevice()) {
        // 设置 ivalue_type 为设备类型
        ivalue_type = IValueUnion::Device;
        // 获取设备 ivalue 的字符串表示，创建共享字符串，并使用它创建设备对象的 flatbuffers 表达形式，获取偏移量
        offset = mobile::serialization::CreateDevice(
                     fbb, fbb.CreateSharedString(ivalue.toDevice().str()))
                     .Union();
    // 如果 ivalue 是枚举类型
    } else if (ivalue.isEnum()) {
        // 获取枚举持有器
        const auto& enum_holder = ivalue.toEnumHolder();
        // 获取枚举类型的完全限定类名
        const auto& qualified_class_name =
            enum_holder->type()->qualifiedClassName();
        // 存储枚举值并获取其索引位置
        uint32_t ival_pos = storeIValueAndGetIndex(fbb, enum_holder->value());
        // 设置 ivalue_type 为枚举值类型
        ivalue_type = IValueUnion::EnumValue;
        // 使用完全限定类名和枚举值索引创建枚举值对象的 flatbuffers 表达形式，获取偏移量
        offset = mobile::serialization::CreateEnumValue(
                     fbb,
                     fbb.CreateSharedString(qualified_class_name.qualifiedName()),
                     ival_pos)
                     .Union();
    // 如果 ivalue 的类型无效
    } else {
        // 抛出错误，显示无效的 ivalue 类型及其标签
        AT_ERROR("Invalid IValue type for serialization: ", ivalue.tagKind());
    }
    // 使用给定的 flatbuffers 构建器创建 IValue 对象，并返回其 flatbuffers 表达形式
    return CreateIValue(fbb, ivalue_type, offset);
/// 结束 torch::jit 命名空间的定义

} // namespace torch::jit

/// 保存移动端模块到文件
void save_mobile_module(
    const mobile::Module& module,  // 移动端模块的引用
    const std::string& filename,   // 要保存的文件名
    const ExtraFilesMap& extra_files,   // 额外的文件映射
    const ExtraFilesMap& jit_sources,   // JIT 源码映射
    const std::vector<IValue>& jit_constants) {  // JIT 常量向量

  // 将移动端模块保存为字节流
  auto buffer = save_mobile_module_to_bytes(
      module, extra_files, jit_sources, jit_constants);

  // 以二进制写入方式打开文件流
  std::fstream ofile(filename, std::ios::binary | std::ios::out);

  // 将字节流数据写入文件
  ofile.write(
      reinterpret_cast<char*>(buffer->data()),   // 字节流数据的指针转换为字符指针
      static_cast<std::streamsize>(buffer->size()));  // 写入数据的大小

  // 关闭文件流
  ofile.close();
}

/// 删除 DetachedBuffer 的实例及其内部 flatbuffers::DetachedBuffer（如果存在）
/// 用作 std::unique_ptr 的自定义删除器；参见 UniqueDetachedBuffer 和 make_unique_detached_buffer
void DetachedBuffer::destroy(DetachedBuffer* buf) {
  // 可能为 null
  delete static_cast<flatbuffers::DetachedBuffer*>(buf->data_owner_);
  delete buf;
}

/// 提供对 DetachedBuffer::destroy() 的访问
struct DetachedBufferFriend {
  /// 返回一个 UniqueDetachedBuffer，包装提供的 DetachedBuffer
  static DetachedBuffer::UniqueDetachedBuffer make_unique_detached_buffer(
      DetachedBuffer* buf) {
    return DetachedBuffer::UniqueDetachedBuffer(buf, DetachedBuffer::destroy);
  }
};

/// 将移动端模块保存为字节流
DetachedBuffer::UniqueDetachedBuffer save_mobile_module_to_bytes(
    const mobile::Module& module,   // 移动端模块的引用
    const ExtraFilesMap& extra_files,   // 额外的文件映射
    const ExtraFilesMap& jit_sources,   // JIT 源码映射
    const std::vector<IValue>& jit_constants) {  // JIT 常量向量

  // 创建 FlatbufferSerializer 实例
  FlatbufferSerializer fb_serializer;

  // 序列化模块为 flatbuffers::DetachedBuffer
  flatbuffers::DetachedBuffer buf = fb_serializer.serializeModule(
      module,
      /*include_tensor_data_in_flatbuffer=*/true,
      extra_files,
      jit_sources,
      jit_constants);

  // 创建 flatbuffers::DetachedBuffer* 的指针
  flatbuffers::DetachedBuffer* buf_ptr =
      new flatbuffers::DetachedBuffer(std::move(buf));

  // 创建 DetachedBuffer 实例
  DetachedBuffer* ret =
      new DetachedBuffer(buf_ptr->data(), buf_ptr->size(), buf_ptr);

  // 返回使用自定义删除器的 UniqueDetachedBuffer
  return DetachedBufferFriend::make_unique_detached_buffer(ret);
}

/// 将移动端模块保存到函数对象中
void save_mobile_module_to_func(
    const mobile::Module& module,   // 移动端模块的引用
    const std::function<size_t(const void*, size_t)>& writer_func) {  // 写入函数对象

  // 将移动端模块保存为字节流
  auto buffer = save_mobile_module_to_bytes(module);

  // 调用写入函数对象写入字节流数据
  writer_func(buffer->data(), buffer->size());
}

/// 注册 flatbuffer 序列化器
bool register_flatbuffer_serializer() {
  return true;
}

/// 结束命名空间定义 torch::jit

} // namespace torch::jit
```