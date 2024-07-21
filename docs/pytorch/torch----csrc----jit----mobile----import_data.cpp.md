# `.\pytorch\torch\csrc\jit\mobile\import_data.cpp`

```py
/**
 * Including necessary header files for TorchScript mobile module import.
 */
#include <torch/csrc/jit/mobile/import_data.h>

#include <ATen/Functions.h>  // Tensor-related functions from ATen
#include <ATen/core/ivalue.h>  // IValue data structure from ATen
#include <c10/util/irange.h>  // Utility for iterating ranges
#include <caffe2/serialize/file_adapter.h>  // FileAdapter for serialization
#include <caffe2/serialize/inline_container.h>  // InlineContainer for serialization
#include <torch/csrc/jit/api/compilation_unit.h>  // CompilationUnit for TorchScript
#include <torch/csrc/jit/mobile/file_format.h>  // File format definitions for mobile module
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>  // Flatbuffer loader for mobile module
#include <torch/csrc/jit/mobile/import.h>  // Import functions for mobile module
#include <torch/csrc/jit/mobile/import_export_common.h>  // Common import/export utilities
#include <torch/csrc/jit/mobile/module.h>  // Module definition for mobile module
#include <torch/csrc/jit/mobile/observer.h>  // Observer support for mobile module
#include <torch/csrc/jit/mobile/type_parser.h>  // Type parser for mobile module
#include <torch/csrc/jit/runtime/instruction.h>  // Instructions for TorchScript runtime
#include <torch/csrc/jit/serialization/unpickler.h>  // Unpickler for TorchScript serialization
#include <torch/custom_class.h>  // Custom class support in TorchScript

#include <caffe2/serialize/in_memory_adapter.h>  // In-memory adapter for serialization
#include <exception>  // Standard exception handling
#include <fstream>  // File stream operations
#include <string>  // String handling
#include <vector>  // Vector operations

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;  // Alias for FileAdapter namespace
using caffe2::serialize::IStreamAdapter;  // Alias for IStreamAdapter namespace
using caffe2::serialize::MemoryReadAdapter;  // Alias for MemoryReadAdapter namespace
using caffe2::serialize::PyTorchStreamReader;  // Alias for PyTorchStreamReader namespace
using caffe2::serialize::ReadAdapterInterface;  // Alias for ReadAdapterInterface namespace

namespace {

/**
 * Class responsible for deserializing IValue from a ZIP file containing "data.pkl".
 */
class IValueUnpickler final {
 public:
  explicit IValueUnpickler(std::unique_ptr<PyTorchStreamReader> reader);

  /**
   * Deserialize the content of "data.pkl" and return the corresponding IValue.
   * @param device Optional device to deserialize data onto.
   * @return The deserialized IValue object.
   */
  c10::IValue deserialize(std::optional<at::Device> device);

 private:
  /**
   * Read and deserialize the specified archive from the reader.
   * @param archive_name Name of the archive file to read.
   * @param mcu CompilationUnit to associate with the deserialized content.
   * @param device Optional device to deserialize data onto.
   * @return The deserialized IValue object.
   */
  c10::IValue readArchive(
      const std::string& archive_name,
      std::shared_ptr<mobile::CompilationUnit> mcu,
      std::optional<at::Device> device);

  std::shared_ptr<CompilationUnit> compilation_unit_;  // Shared pointer to CompilationUnit
  std::unique_ptr<PyTorchStreamReader> reader_;  // Unique pointer to PyTorchStreamReader
};

/**
 * Constructor for IValueUnpickler.
 * @param reader Unique pointer to PyTorchStreamReader to initialize with.
 */
IValueUnpickler::IValueUnpickler(std::unique_ptr<PyTorchStreamReader> reader)
    : compilation_unit_(std::make_shared<CompilationUnit>()),
      reader_(std::move(reader)) {}

/**
 * Deserialize "data.pkl" from the ZIP file and return the corresponding IValue.
 * @param device Optional device to deserialize data onto.
 * @return The deserialized IValue object.
 */
c10::IValue IValueUnpickler::deserialize(std::optional<at::Device> device) {
  auto mcu = std::make_shared<mobile::CompilationUnit>();

  // NOLINTNEXTLINE(performance-move-const-arg)
  return readArchive("data", mcu, std::move(device));
}

/**
 * Read and deserialize the specified archive from the reader.
 * @param archive_name Name of the archive file to read.
 * @param mcu CompilationUnit to associate with the deserialized content.
 * @param device Optional device to deserialize data onto.
 * @return The deserialized IValue object.
 */
c10::IValue IValueUnpickler::readArchive(
    const std::string& archive_name,
    std::shared_ptr<mobile::CompilationUnit> mcu,
    std::optional<at::Device> device) {
  std::stringstream picklename;
  picklename << archive_name << ".pkl";
  at::DataPtr pickle_ptr;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t pickle_size;
  std::tie(pickle_ptr, pickle_size) = reader_->getRecord(picklename.str());

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  auto reader = [&](char* buffer, size_t len) -> size_t {
    if (bytes_read >= pickle_size) {
      return 0;
    }
    len = std::min(pickle_size - bytes_read, len);
    // Copy len bytes into buffer
    const char* start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;

    return len;
  };
    return len;
  };

  static const c10::QualifiedName torchPrefix = "__torch__";
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    TypePtr type;
    // HACK: first we check whether the name starts with `__torch__` to tell if
    // it's "supposed" to be a class type. This is a reliable check today, but
    // there is no guarantee that this is the case. The real solution is to
    // merge type parsers so we can share class resolution logic.
    if (torchPrefix.isPrefixOf(qn)) {
      // Check if the class is already registered; if not, create and register it
      if (compilation_unit_->get_class(qn) == nullptr) {
        auto typeptr = ClassType::create(qn, compilation_unit_, true);
        compilation_unit_->register_type(typeptr);
      }
      // Retrieve the class type from the compilation unit
      type = compilation_unit_->get_class(qn);
    } else {
      // Parse the type from the qualified name
      type = c10::parseType(qn.qualifiedName());
    }
    // Return a StrongTypePtr using the compilation unit and resolved type
    return c10::StrongTypePtr(compilation_unit_, type);
  };

  auto obj_loader = [&](const at::StrongTypePtr& type, IValue input) {
    auto cls = type.type_->expect<at::ClassType>();
    auto qn = cls->name();
    c10::QualifiedName method_name(qn.value(), "__setstate__");
    // Find the '__setstate__' method in the module compilation unit
    auto setstate = mcu->find_function(method_name);
    // Lambda to find custom class type with '__setstate__' method
    auto find_custom_class_with_setstate = [&qn]() -> c10::ClassTypePtr {
      auto custom_class_type = torch::jit::getCustomClass(qn->qualifiedName());
      if (custom_class_type && custom_class_type->findMethod("__setstate__")) {
        return custom_class_type;
      }
      return nullptr;
    };
    if (setstate) {
      // Create an object and invoke '__setstate__' method if available
      auto obj = c10::ivalue::Object::create(type, 0);
      Stack stack({obj, input});
      setstate->run(stack);
      return obj;
    } else if (auto custom_class_type = find_custom_class_with_setstate()) {
      // Create an object for custom class type and invoke '__setstate__' method
      auto obj = c10::ivalue::Object::create(
          c10::StrongTypePtr(nullptr, custom_class_type), 1);
      Stack stack({obj, input});
      custom_class_type->getMethod("__setstate__").run(stack);
      return obj;
    } else {
      // Handle as a generic dictionary input and create object with attributes
      auto dict = std::move(input).toGenericDict();
      size_t ndict = dict.size();
      auto obj = c10::ivalue::Object::create(type, ndict);
      auto it = dict.begin();
      for (const auto i : c10::irange(ndict)) {
        std::stringstream name;
        name << it->key();
        cls->addOrCheckAttribute(name.str(), it->key().type());
        obj->setSlot(i, it->value());
        ++it;
      }
      return obj;
    }
  };

  auto read_record = [&](const std::string& name) {
    // Construct the full record name within the archive and read it
    std::stringstream ss;
    ss << archive_name << "/" << name;
    return std::get<0>(reader_->getRecord(ss.str()));
  };

  // Create an Unpickler object with various callbacks and settings, then parse and return the result
  Unpickler unpickler(
      reader,
      std::move(type_resolver),
      std::move(obj_loader),
      std::move(read_record),
      // NOLINTNEXTLINE(performance-move-const-arg)
      std::move(device),
      false,
      nullptr);
  return unpickler.parse_ivalue();
}

/**
 * 从给定的读取适配器中反序列化 ZIP + Pickle 格式的参数映射并返回。
 */
std::map<std::string, at::Tensor> load_parameters_from_zip(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device) {
  // 使用 PyTorchStreamReader 从读取适配器中创建一个读取器
  auto reader = std::make_unique<PyTorchStreamReader>(std::move(rai));
  // 使用 IValueUnpickler 对象解序列化数据
  IValueUnpickler unpickler(std::move(reader));
  // 将反序列化后的结果转换为通用字典类型
  auto result = unpickler.deserialize(device).toGenericDict();
  // 创建一个标准库映射，用于保存字符串键和张量值
  std::map<std::string, at::Tensor> map;
  // 遍历反序列化后的结果，将每个键值对添加到映射中
  for (const auto& e : result) {
    // 提取键的字符串表示
    auto key = e.key().toStringRef();
    // 提取值并将其转换为张量，然后获取张量数据
    auto value = e.value().toTensor().tensor_data();
    // 将键值对添加到映射中
    map[key] = value;
  }
  // 返回包含参数映射的标准库映射
  return map;
}

} // namespace

/**
 * 从移动模块中提取存储的参数映射。
 * 预期的布局与 #_save_parameters() 创建的布局兼容。
 */
std::map<std::string, at::Tensor> mobile_module_to_parameter_map(
    const mobile::Module& module) {
  // 安全地查找具有预期名称的槽位。
  // 注意，如果属性不存在，c10::ivalue::Object::getAttr() 不安全。
  auto obj = module._ivalue();
  const std::vector<IValue>& slots = obj->slots();
  for (const auto i : c10::irange(slots.size())) {
    if (obj->type()->getAttributeName(i) ==
        mobile::internal::kSavedParametersAttributeName) {
      // 找到了具有正确名称的槽位；确保它是一个 Dict<string, Tensor>。
      c10::IValue data = slots[i];
      if (data.isGenericDict()) {
        auto data_dict = data.toGenericDict();

        // 键和值应该是 DynamicTypes，分别包装 String 和 Tensor。
        c10::DynamicType* keyType =
            data_dict.keyType()->castRaw<c10::DynamicType>();
        c10::DynamicType* valueType =
            data_dict.valueType()->castRaw<c10::DynamicType>();
        if (keyType != nullptr &&
            keyType->fallback()->kind() == TypeKind::StringType &&
            valueType != nullptr &&
            valueType->fallback()->kind() == TypeKind::TensorType) {
          // 名称和类型匹配；将内容复制到输出映射中。
          std::map<std::string, at::Tensor> params;
          for (const auto& e : data_dict) {
            // 源张量指向与模块关联的 flatbuffer 数据。
            // 但是，这个张量需要比模块生存时间更长，因为调用者不会持有模块的指针。
            // 因此，返回一个深拷贝。
            const auto& source = e.value().toTensor();
            at::Tensor copy = at::empty_like(source); // 必须具有相同的形状。
            copy.copy_(source);

            params[e.key().toStringRef()] = copy;
          }
          return params;
        }
      }
    }
  }

  // 如果在反序列化后的移动模块中找不到名称为 'mobile::internal::kSavedParametersAttributeName' 的 Dict<string, Tensor>，则抛出异常。
  TORCH_CHECK(
      false,
      "Could not find Dict<string, Tensor> named '",
      mobile::internal::kSavedParametersAttributeName,
      "' in deserialized mobile::Module");
}

static std::map<std::string, at::Tensor> _load_parameters_bytes(
    std::shared_ptr<char> data,
    size_t size,
    // 检查数据大小是否至少为文件格式头部大小，否则抛出异常信息"Unrecognized data format"
    TORCH_CHECK(size >= kFileFormatHeaderSize, "Unrecognized data format");
    // 获取数据的文件格式
    FileFormat format = getFileFormat(data.get());
    // 根据文件格式调用相应的解析器
    std::map<std::string, at::Tensor> map;
    switch (format) {
      // 对于 Flatbuffer 文件格式
      case FileFormat::FlatbufferFileFormat: {
        // 解析 Flatbuffer 数据，不包含对象，返回模型参数映射
        auto m = parse_flatbuffer_no_object(data, size, device);
        // 将解析后的模型参数映射转换为标准的参数映射
        map = mobile_module_to_parameter_map(m);
        break;
      }
    
      // 对于 Zip 文件格式
      case FileFormat::ZipFileFormat: {
        // 使用内存读取适配器创建 RAII 对象，封装数据和大小
        auto rai = std::make_unique<caffe2::serialize::MemoryReadAdapter>(
            data.get(), size);
        // 从 Zip 文件中加载参数到参数映射中，指定设备信息
        map = load_parameters_from_zip(std::move(rai), device);
        break;
      }
    
      // 默认情况，如果文件格式未识别，抛出异常信息"Unrecognized data format"
      default:
        TORCH_CHECK(false, "Unrecognized data format");
    }
    // 返回解析后的参数映射
    return map;
}

// 命名空间 jit 下的代码块结束

// 命名空间 torch 下的代码块结束

std::map<std::string, at::Tensor> _load_parameters(
    std::istream& in,
    std::optional<at::Device> device) {
  // 调用 get_stream_content 函数从输入流中获取数据和大小
  auto [data, size] = get_stream_content(in);
  // 调用 _load_parameters_bytes 函数加载参数，并返回结果
  return _load_parameters_bytes(std::move(data), size, device);
}

std::map<std::string, at::Tensor> _load_parameters(
    const std::string& filename,
    std::optional<at::Device> device) {
  // 调用 get_file_content 函数从文件名获取数据和大小
  auto [data, size] = get_file_content(filename.c_str());
  // 调用 _load_parameters_bytes 函数加载参数，并返回结果
  return _load_parameters_bytes(std::move(data), size, device);
}

} // namespace jit
} // namespace torch
```