# `.\pytorch\torch\csrc\jit\serialization\import.cpp`

```
// 包含 ATen 和 Caffe2 库中的头文件，用于操作张量和序列化
#include <ATen/core/interned_strings.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/in_memory_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <caffe2/serialize/istream_adapter.h>
#include <caffe2/serialize/read_adapter_interface.h>

// 包含 Torch 库中的 JIT 相关头文件，用于图的操作和模型的序列化
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/ir/graph_utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/operator_upgraders/upgraders_entry.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/unpickler.h>

// 包含 ATen 和 C10 库中的其它头文件，用于异常处理和辅助功能
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <fmt/format.h>

// 包含标准库头文件
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Torch JIT 命名空间
namespace torch::jit {

// 使用 Caffe2 序列化相关类的别名
using caffe2::serialize::MemoryReadAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

// 静态函数：验证对象在 '__setstate__' 之后的状态
static void postSetStateValidate(const IValue& v) {
  auto obj = v.toObject();
  const auto& objType = obj->type();
  // 遍历对象的属性，验证非可选属性是否已初始化
  for (const auto i : c10::irange(objType->numAttributes())) {
    const auto& attrType = objType->getAttribute(i);
    const auto& attrName = objType->getAttributeName(i);
    const auto& slot = obj->getSlot(i);
    // TODO: Issue #20497
    // 检查非 UnionType、OptionalType 和 NoneType 的属性是否未初始化
    if (attrType->kind() != TypeKind::UnionType &&
        attrType->kind() != TypeKind::OptionalType &&
        attrType->kind() != TypeKind::NoneType) {
      TORCH_CHECK(
          !slot.isNone(),
          fmt::format(
              "The field '{}' was left uninitialized after '__setstate__', "
              "but expected a value of type '{}'",
              attrName,
              attrType->repr_str()));
    }
  }
}

// 函数：根据类型和输入创建对象
c10::intrusive_ptr<c10::ivalue::Object> ObjLoaderFunc(
    const at::StrongTypePtr& type,
    IValue input) {
  auto cls = type.type_->expect<at::ClassType>();
  auto qn = cls->name();
  size_t n = cls->numAttributes();
  // 检查类是否具有有效的 '__setstate__' 和 '__getstate__' 方法
  if (checkHasValidSetGetState(cls)) {
    // 创建具有指定类型和属性数的对象
    auto obj = c10::ivalue::Object::create(type, n);
    // XXX: 不要优化 __setstate__，以防在初始化之前尝试特化该类。
    GraphOptimizerEnabledGuard guard(false);
    // 获取类的 __setstate__ 方法
    Function& set_state = cls->getMethod("__setstate__");
    // 因为我们在反序列化的过程中，可能还有列表和字典的标签不准确（例如，它们可能报告自己是 List[Any]）。
    // 但是我们需要运行 __setstate__ 方法，它会检查输入类型，并可能访问标签。
    // 由于 setstate 方法有已知的输入类型，我们可以通过将 set_state 的输入类型应用于传递的状态对象来正确恢复标签。
    // TODO: 一旦[serialization type tags]完成，删除此处代码
    restoreAccurateTypeTags(
        input, set_state.getSchema().arguments().at(1).type());
    // 调用 set_state 方法，传递 obj 和 input 作为参数
    set_state({obj, input});
    // 对 obj 进行设置状态后的验证
    postSetStateValidate(obj);
    // 返回 obj 对象
    return obj;
    } else {
    // 将 input 转换为通用字典类型
    auto dict = std::move(input).toGenericDict();
    // 创建一个类型为 type，大小为 n 的 Object 对象
    auto obj = c10::ivalue::Object::create(type, n);
    // 遍历索引范围为 0 到 n-1
    for (const auto i : c10::irange(n)) {
        // 使用类的属性名称作为键，从 dict 中获取值，并将其设置为 obj 的第 i 个槽位
        obj->setSlot(i, dict.at(cls->getAttributeName(i)));
    }
    // 返回 obj 对象
    return obj;
    }
}

namespace {

// 这是一个反序列化器类，用于从 pt 文件加载脚本模块。
// 文件内容是使用 PyTorchStreamWriter 写入的，具体详情请参考 caffe2/serialize/inline_container.h。
// 模块被保存在 pickle 中。readArchive() 被调用以解析和构建常量表和脚本模块。
class ScriptModuleDeserializer final {
 public:
  // 构造函数，接受编译单元和读取器的共享指针
  ScriptModuleDeserializer(
      std::shared_ptr<CompilationUnit> cu,
      std::shared_ptr<PyTorchStreamReader> reader)
      : compilation_unit_(std::move(cu)),
        reader_(std::move(reader)),
        code_prefix_("code/"),  // 设置代码前缀为 "code/"
        pickle_dir_prefix_(""),  // 设置 pickle 目录前缀为空字符串
        tensor_dir_prefix_(""),  // 设置张量目录前缀为空字符串
        // 初始化源导入器，用于查找归档中的源码文件
        source_importer_(
            compilation_unit_,
            &constants_table_,
            [this](const std::string& qualifier) {
              return findSourceInArchiveFromQualifier(
                  *reader_, code_prefix_, qualifier);
            },
            reader_->version()) {}

  // 构造函数，接受编译单元、读取器、pickle 目录前缀、张量目录前缀和存储上下文的共享指针
  ScriptModuleDeserializer(
      std::shared_ptr<CompilationUnit> cu,
      std::shared_ptr<PyTorchStreamReader> reader,
      std::string pickle_dir_prefix,
      std::string tensor_dir_prefix,
      std::shared_ptr<DeserializationStorageContext> storage_context)
      : compilation_unit_(std::move(cu)),
        reader_(std::move(reader)),
        storage_context_(std::move(storage_context)),
        code_prefix_(".data/ts_code/code/"),  // 设置代码前缀为 ".data/ts_code/code/"
        pickle_dir_prefix_(std::move(pickle_dir_prefix)),  // 设置 pickle 目录前缀
        tensor_dir_prefix_(std::move(tensor_dir_prefix)),  // 设置张量目录前缀
        // 初始化源导入器，用于查找归档中的源码文件
        source_importer_(
            compilation_unit_,
            &constants_table_,
            [this](const std::string& qualifier) {
              return findSourceInArchiveFromQualifier(
                  *reader_, code_prefix_, qualifier);
            },
            reader_->version()) {}

  // 反序列化函数，接受设备、额外文件映射和是否恢复形状的参数
  Module deserialize(
      std::optional<at::Device> device,
      ExtraFilesMap& extra_files,
      bool restore_shapes = false);

 private:
  // 读取归档的私有方法，接受归档名称作为参数
  IValue readArchive(const std::string& archive_name);

  std::shared_ptr<CompilationUnit> compilation_unit_;  // 编译单元的共享指针
  std::shared_ptr<PyTorchStreamReader> reader_;  // PyTorchStreamReader 的共享指针
  std::shared_ptr<DeserializationStorageContext> storage_context_;  // 反序列化存储上下文的共享指针
  std::optional<at::Device> device_;  // 可选的设备类型
  std::vector<at::IValue> constants_table_;  // 常量表，存储 IValue 类型的向量
  std::string code_prefix_;  // 代码前缀字符串
  std::string pickle_dir_prefix_;  // pickle 目录前缀字符串
  std::string tensor_dir_prefix_;  // 张量目录前缀字符串
  SourceImporter source_importer_;  // 源导入器对象
};

// readArchive 方法的实现，读取指定归档名称的内容
IValue ScriptModuleDeserializer::readArchive(const std::string& archive_name) {
  // 类型解析器函数，加载给定限定名的类型
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    auto cls = source_importer_.loadType(qn);
    return c10::StrongTypePtr(compilation_unit_, std::move(cls));
  };

  // 调用 readArchiveAndTensors 函数读取归档和张量，返回结果
  return readArchiveAndTensors(
      /*archive_name=*/archive_name,
      /*pickle_prefix=*/pickle_dir_prefix_,
      /*tensor_prefix=*/tensor_dir_prefix_,
      type_resolver,
      ObjLoaderFunc,
      device_,
      *reader_,
      nullptr,
      storage_context_);
}
// 重写量化卷积操作，为了向后兼容性而进行修改
void rewriteQuantizedConvForBC(const Module& module) {
  // 原始的量化二维卷积操作的图形字符串表示
  const std::string& old_quantized_conv2d = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv2d(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
         return (%r) )";

  // 原始的带ReLU的量化二维卷积操作的图形字符串表示
  const std::string& old_quantized_conv2d_relu = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv2d_relu(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
         return (%r) )";

  // 原始的量化三维卷积操作的图形字符串表示
  const std::string& old_quantized_conv3d = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv3d(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
         return (%r) )";

  // 原始的带ReLU的量化三维卷积操作的图形字符串表示
  const std::string& old_quantized_conv3d_relu = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv3d_relu(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
         return (%r) )";

  // 更新后的量化二维卷积操作的图形字符串表示
  const std::string& new_quantized_conv2d = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv2d(%x, %packed_params, %r_scale, %r_zero_point)
         return (%r) )";

  // 更新后的带ReLU的量化二维卷积操作的图形字符串表示
  const std::string& new_quantized_conv2d_relu = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv2d_relu(%x, %packed_params, %r_scale, %r_zero_point)
         return (%r) )";

  // 更新后的量化三维卷积操作的图形字符串表示
  const std::string& new_quantized_conv3d = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv3d(%x, %packed_params, %r_scale, %r_zero_point)
         return (%r) )";

  // 更新后的带ReLU的量化三维卷积操作的图形字符串表示
  const std::string& new_quantized_conv3d_relu = R"(
graph(%x, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point):
         %r = quantized::conv3d_relu(%x, %packed_params, %r_scale, %r_zero_point)
         return (%r) )";

  // 创建子图重写器对象
  SubgraphRewriter rewriter;
  // 包含原始模式字符串和更新后模式字符串的键值对列表
  static const std::vector<std::pair<std::string, std::string>>
      patterns_and_replacements = {
          {old_quantized_conv2d, new_quantized_conv2d},
          {old_quantized_conv2d_relu, new_quantized_conv2d_relu},
          {old_quantized_conv3d, new_quantized_conv3d},
          {old_quantized_conv3d_relu, new_quantized_conv3d_relu},
      };
  // 注册每个替换模式到重写器中
  for (const auto& item : patterns_and_replacements) {
    rewriter.RegisterRewritePattern(item.first, item.second);
  }
  // 在模块上运行重写器，进行模块内部的替换操作
  rewriter.runOnModule(module);

  // 递归地对模块的每个子模块调用此函数，以确保整个模块树都被更新
  for (const Module& child : module.children()) {
    rewriteQuantizedConvForBC(child);
  }
}

// 脚本模块反序列化器的反序列化函数，将二进制数据反序列化为模块对象
Module ScriptModuleDeserializer::deserialize(
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool restore_shapes) {
  // 在加载开始之前，我们先填充升级程序图映射表
  populate_upgraders_graph_map();

  // 记录 API 使用情况，只记录一次
  C10_LOG_API_USAGE_ONCE("torch.jit.load");

  // 设定设备
  device_ = device;

  // 加载额外文件
  for (const auto& kv : extra_files) {
    // 构建额外文件在 reader 中的键名
    const std::string& key = "extra/" + kv.first;
    // 如果 reader 中存在该记录
    if (reader_->hasRecord(key)) {
      // 读取元数据
      auto [meta_ptr, meta_size] = reader_->getRecord(key);
      // 将元数据转换为字符串存储在 extra_files 中
      extra_files[kv.first] =
          std::string(static_cast<char*>(meta_ptr.get()), meta_size);
    }
  }

  // 检查是否存在旧版模型的 JSON 文件，并且 code_prefix_ 为 "code/"
  if (reader_->hasRecord("model.json") && code_prefix_ == "code/") {
    // 在移动设备上不支持旧版模型格式，抛出错误
    AT_ERROR("Legacy model format is not supported on mobile.");
  }

  // 从归档文件中读取名为 "constants" 的数据，转换为元组
  auto tuple = readArchive("constants").toTuple();
  // 将元组中的每个元素转换为 IValue，并添加到 constants_table_ 中
  for (auto constant : tuple->elements()) {
    constants_table_.push_back(constant.toIValue());
  }

  // 从归档文件中读取名为 "data" 的数据，转换为 Module 类型
  auto m_ivalue = readArchive("data");
  auto m = Module(m_ivalue.toObject());

  // 重写量化卷积以保证向后兼容性
  rewriteQuantizedConvForBC(m);

  // 检查并加载保存的追踪输入
  if (restore_shapes && reader_->hasRecord("traced_inputs.pkl")) {
    // 从归档文件中读取名为 "traced_inputs" 的数据，转换为 GenericDict 类型
    auto dict = readArchive("traced_inputs").toGenericDict();
    // 遍历追踪输入字典
    for (const auto& entry : dict) {
      // 获取追踪输入对应的方法，转换为 GraphFunction
      auto inputs = entry.value().toList().vec();
      auto g =
          toGraphFunction(m.get_method(entry.key().toStringRef()).function())
              .graph();
      Stack stack(inputs.begin(), inputs.end());
      // 如果图的输入数与堆栈大小相差1，将模型的 IValue 作为第一个输入
      if (g->inputs().size() == stack.size() + 1) {
        stack.insert(stack.begin(), m_ivalue);
      }
      // 设置输入张量的类型
      setInputTensorTypes(*g, stack, /*complete=*/true);
      // 传播输入形状
      PropagateInputShapes(g);
    }
  } else {
    // 如果没有保存的追踪输入且需要恢复形状，则警告
    if (restore_shapes) {
      TORCH_WARN("Cannot restore shapes as no traced inputs were stored");
    }
  }

  // 记录加载 API 使用的元数据
  c10::LogAPIUsageMetadata(
      "torch.script.load.metadata",
      {{"serialization_id", reader_->serializationId()}});
  
  // 返回加载的模块
  return m;
}
} // namespace



Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    std::optional<at::Device> device,
    bool load_debug_files) {
  // 创建一个额外文件映射
  ExtraFilesMap extra_files;
  // 调用另一个重载的 import_ir_module 函数，并返回其结果
  return import_ir_module(
      std::move(cu), in, device, extra_files, load_debug_files);
}



static Module _load_jit_module_from_bytes(
    const std::shared_ptr<char>& data,
    size_t size,
    std::shared_ptr<CompilationUnit> cu,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool restore_shapes);



Module parse_and_initialize_jit_module(
    const std::shared_ptr<char>& data,
    size_t size,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device) {
  // 调用函数以填充升级器图映射
  populate_upgraders_graph_map();
  // 创建额外的 JIT 文件映射
  ExtraFilesMap jit_files;
  // 创建 JIT 常量的向量
  std::vector<IValue> jit_constants;
  // 解析和初始化 JIT 模块，并为 JIT 函数准备 Mobile 模块
  mobile::Module mobilem = parse_and_initialize_mobile_module_for_jit(
      data.get(), size, jit_files, jit_constants, device, &extra_files);

  // 从 Mobile 模块和常量创建 JIT 模块
  Module m = jitModuleFromSourceAndConstants(
      mobilem._ivalue(),
      jit_files,
      jit_constants,
      static_cast<int32_t>(mobilem.bytecode_version()));
  // 设置删除内存的函数
  m.set_delete_memory(data);
  // 返回 JIT 模块
  return m;
}



Module load_jit_module_from_file(
    const std::string& filename,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device) {
  // 获取文件内容
  auto data = get_file_content(filename.c_str());
  // 解析和初始化 JIT 模块并返回
  return parse_and_initialize_jit_module(
      std::get<0>(data), std::get<1>(data), extra_files, device);
}



Module load_jit_module_from_stream(
    std::istream& in,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device) {
  // 获取流的内容
  auto data = get_stream_content(in);
  // 解析和初始化 JIT 模块并返回
  return parse_and_initialize_jit_module(
      std::get<0>(data), std::get<1>(data), extra_files, device);
}



Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files,
    bool restore_shapes) {
  // 将流的位置设置为开头
  in.seekg(0, in.beg);
  // 注意: Zipformat 可能是大文件。因此直接使用流版本，而不是一次性读取文件。
  // 检查流的文件格式是否为 Flatbuffer 文件格式
  if (getFileFormat(in) != FileFormat::FlatbufferFileFormat) {
    // 创建 PyTorch 的流阅读器对象
    auto reader = std::make_unique<PyTorchStreamReader>(&in);
    reader->setShouldLoadDebugSymbol(load_debug_files);
    // 创建脚本模块反序列化器并返回反序列化结果
    ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
    return deserializer.deserialize(device, extra_files, restore_shapes);
  }
  // 获取流的内容并调用 _load_jit_module_from_bytes 函数加载 JIT 模块
  auto [data, size] = get_stream_content(in);
  return _load_jit_module_from_bytes(
      data, size, cu, device, extra_files, restore_shapes);
}



Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<PyTorchStreamReader> reader,
    std::shared_ptr<DeserializationStorageContext> storage_context,
    std::optional<at::Device> device,
    // 使用 ScriptModuleDeserializer 类来反序列化脚本模块
    ScriptModuleDeserializer deserializer(
        // 移动 cu 到 deserializer 中，cu 是一个 ComputeUnit 对象
        std::move(cu),
        // 移动 reader 到 deserializer 中，reader 是一个 Reader 对象
        std::move(reader),
        // 设置 pickle_dir_prefix 参数，指定 pickle 文件的存储路径前缀为 ".data/ts_code/" + ts_id + "/"
        /* pickle_dir_prefix = */ ".data/ts_code/" + ts_id + "/",
        // 设置 tensor_dir_prefix 参数，指定张量数据的存储路径前缀为 ".data/"
        /* tensor_dir_prefix = */ ".data/",
        // 移动 storage_context 到 deserializer 中，storage_context 是一个 StorageContext 对象
        std::move(storage_context));
    
    // 创建 ExtraFilesMap 对象用于额外文件的映射
    ExtraFilesMap extra_files;
    
    // 调用 deserializer 的 deserialize 方法进行反序列化操作，返回反序列化后的结果
    return deserializer.deserialize(device, extra_files);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    std::optional<at::Device> device,
    bool load_debug_files) {
  ExtraFilesMap extra_files;
  // 调用import_ir_module的重载版本，传入空的extra_files映射
  return import_ir_module(
      std::move(cu), filename, device, extra_files, load_debug_files);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files,
    bool restore_shapes) {
  // 如果文件格式不是Flatbuffer格式，则使用PyTorchStreamReader读取文件内容
  if (getFileFormat(filename) != FileFormat::FlatbufferFileFormat) {
    // 创建一个PyTorchStreamReader对象，设置是否加载调试符号
    auto reader = std::make_unique<PyTorchStreamReader>(filename);
    reader->setShouldLoadDebugSymbol(load_debug_files);
    // 创建ScriptModuleDeserializer对象，传入CompilationUnit和reader，然后反序列化
    ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
    // 返回反序列化后的模块
    return deserializer.deserialize(device, extra_files, restore_shapes);
  }
  // 否则，使用get_file_content函数读取文件内容，并调用_load_jit_module_from_bytes函数加载模块
  auto [data, size] = get_file_content(filename.c_str());
  return _load_jit_module_from_bytes(
      data, size, cu, device, extra_files, restore_shapes);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<at::Device> device,
    bool load_debug_files) {
  ExtraFilesMap extra_files;
  // 调用import_ir_module的重载版本，传入空的extra_files映射
  return import_ir_module(
      std::move(cu), std::move(rai), device, extra_files, load_debug_files);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files) {
  // 将std::unique_ptr<ReadAdapterInterface>转换为std::shared_ptr<ReadAdapterInterface>
  std::shared_ptr<ReadAdapterInterface> rai_shared = std::move(rai);
  // 调用import_ir_module的重载版本，传入std::shared_ptr<ReadAdapterInterface>和extra_files映射
  return import_ir_module(
      std::move(cu), rai_shared, device, extra_files, load_debug_files);
}

Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<ReadAdapterInterface> rai,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files) {
  // 创建一个PyTorchStreamReader对象，使用std::shared_ptr<ReadAdapterInterface>作为参数
  auto reader = std::make_shared<PyTorchStreamReader>(std::move(rai));
  reader->setShouldLoadDebugSymbol(load_debug_files);
  // 创建ScriptModuleDeserializer对象，传入CompilationUnit和reader，然后反序列化
  ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
  // 返回反序列化后的模块
  return deserializer.deserialize(device, extra_files);
}

Module load(
    std::istream& in,
    std::optional<at::Device> device,
    bool load_debug_files) {
  // 创建一个CompilationUnit对象
  auto cu = std::make_shared<CompilationUnit>();
  // 调用import_ir_module函数，传入CompilationUnit对象、输入流in和其他参数
  return import_ir_module(std::move(cu), in, device, load_debug_files);
}

Module load(
    std::istream& in,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files) {
  // 创建一个CompilationUnit对象
  auto cu = std::make_shared<CompilationUnit>();
  // 调用import_ir_module函数，传入CompilationUnit对象、输入流in、extra_files映射和其他参数
  return import_ir_module(
      std::move(cu), in, device, extra_files, load_debug_files);
}

Module load(
    const std::string& filename,
    std::optional<at::Device> device,


这些注释详细解释了每个函数的作用及其参数的含义，帮助理解代码的执行流程和功能。
    // 创建一个名为cu的智能指针，指向一个新的CompilationUnit对象
    auto cu = std::make_shared<CompilationUnit>();
    // 调用import_ir_module函数，将cu作为参数传入，filename作为文件名参数传入，
    // device作为设备参数传入，load_debug_files作为是否加载调试文件的标志传入，
    // 并返回其结果
    return import_ir_module(std::move(cu), filename, device, load_debug_files);
// 从文件加载模块，返回Module对象
Module load(
    const std::string& filename,  // 文件名
    std::optional<at::Device> device,  // 设备选项（可选）
    ExtraFilesMap& extra_files,  // 额外文件映射
    bool load_debug_files) {  // 是否加载调试文件

  auto cu = std::make_shared<CompilationUnit>();  // 创建共享的编译单元对象
  return import_ir_module(
      std::move(cu), filename, device, extra_files, load_debug_files);  // 调用import_ir_module导入IR模块并返回
}

// 从自定义读取适配器加载模块，返回Module对象
Module load(
    std::shared_ptr<ReadAdapterInterface> rai,  // 自定义读取适配器接口指针
    std::optional<c10::Device> device,  // 设备选项（可选）
    bool load_debug_files) {  // 是否加载调试文件

  auto cu = std::make_shared<CompilationUnit>();  // 创建共享的编译单元对象
  ExtraFilesMap extra_files;  // 创建额外文件映射
  return import_ir_module(
      std::move(cu), std::move(rai), device, extra_files, load_debug_files);  // 调用import_ir_module导入IR模块并返回
}

// 从自定义读取适配器加载模块，返回Module对象
Module load(
    std::shared_ptr<ReadAdapterInterface> rai,  // 自定义读取适配器接口指针
    std::optional<c10::Device> device,  // 设备选项（可选）
    ExtraFilesMap& extra_files,  // 额外文件映射
    bool load_debug_files) {  // 是否加载调试文件

  auto cu = std::make_shared<CompilationUnit>();  // 创建共享的编译单元对象
  return import_ir_module(
      std::move(cu), std::move(rai), device, extra_files, load_debug_files);  // 调用import_ir_module导入IR模块并返回
}

// 从字节数据加载JIT模块，返回Module对象
Module _load_jit_module_from_bytes(
    const std::shared_ptr<char>& data,  // 字节数据指针
    size_t size,  // 数据大小
    std::shared_ptr<CompilationUnit> cu,  // 共享的编译单元指针
    std::optional<c10::Device> device,  // 设备选项（可选）
    ExtraFilesMap& extra_files,  // 额外文件映射
    bool restore_shapes) {  // 是否恢复形状信息

  TORCH_CHECK(size >= kFileFormatHeaderSize, "Unrecognized data format");  // 检查数据大小是否符合文件格式头大小
  auto format = getFileFormat(data.get());  // 获取数据格式
  switch (format) {
    case FileFormat::FlatbufferFileFormat: {  // 如果是FlatBuffer格式文件
      return parse_and_initialize_jit_module(data, size, extra_files, device);  // 解析并初始化JIT模块并返回
    }
    case FileFormat::ZipFileFormat: {  // 如果是ZIP文件格式
      auto rai = std::make_unique<MemoryReadAdapter>(data.get(), size);  // 创建内存读取适配器
      auto reader = std::make_unique<PyTorchStreamReader>(std::move(rai));  // 创建PyTorch流读取器
      ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));  // 创建脚本模块反序列化器
      return deserializer.deserialize(device, extra_files, restore_shapes);  // 反序列化模块并返回
    }
    default:
      TORCH_CHECK(false, "Unrecognized data format");  // 如果未识别的数据格式，抛出异常
  }
}

// 重新创建对象的方法和类型
static IValue recreateObject(IValue ivalue, const TypeResolver& resolver) {
  if (ivalue.isObject()) {  // 如果值是对象
    auto obj = ivalue.toObject();  // 获取对象
    auto classtype_old = obj->type();  // 获取对象的旧类型
    auto newtype = resolver(*classtype_old->name());  // 解析并获取新类型
    size_t n = classtype_old->numAttributes();  // 获取对象的属性数量
    auto newobj = c10::ivalue::Object::create(newtype, n);  // 创建新对象
    for (const auto i : c10::irange(n)) {
      newobj->setSlot(i, recreateObject(obj->getSlot(i), resolver));  // 递归重新创建对象的属性值
    }
    return newobj;  // 返回新对象
  } else if (ivalue.isList()) {  // 如果值是列表
    auto res = c10::impl::GenericList(ivalue.type()->containedType(0));  // 创建新的泛型列表
    for (const auto& ival : ivalue.toList()) {
      res.emplace_back(recreateObject(ival, resolver));  // 递归重新创建列表的元素
    }
    return res;  // 返回新列表
  } else if (ivalue.isGenericDict()) {  // 如果值是通用字典
    // 如果输入值是 GenericDict 类型，则进行以下操作
    auto result = c10::impl::GenericDict(
        ivalue.type()->containedType(0), ivalue.type()->containedType(1));
    // 遍历输入值的 GenericDict，重新创建 key 和 value 的对象，并插入到 result 中
    for (const auto& kv : ivalue.toGenericDict()) {
      result.insert_or_assign(
          recreateObject(kv.key(), resolver),
          recreateObject(kv.value(), resolver));
    }
    // 返回结果 GenericDict
    return result;
  } else if (ivalue.isTuple()) {
    // 如果输入值是 Tuple 类型，则进行以下操作
    std::vector<IValue> res;
    // 遍历 Tuple 中的每个元素，使用 resolver 重新创建对象，并添加到 res 中
    for (const auto& ival : ivalue.toTuple()->elements()) {
      res.push_back(recreateObject(ival, resolver));
    }
    // 使用 res 创建新的 Tuple，并返回
    return c10::ivalue::Tuple::create(res);
  }
  // 如果输入值是叶子类型，则直接返回该值
  // Leaf types are returned verbatim.
  return ivalue;
// 定义一个名为 jitModuleFromSourceAndConstants 的函数，用于从给定的 IValue、源文件映射、常量向量和版本号创建模块对象
Module jitModuleFromSourceAndConstants(
    const IValue& ivalue,  // 输入参数：表示要重建的对象的 IValue
    const ExtraFilesMap& source,  // 输入参数：包含额外源文件的映射
    const std::vector<IValue>& constants,  // 输入参数：常量向量
    int32_t version) {  // 输入参数：版本号

  // 创建一个共享的编译单元
  auto compilation_unit = std::make_shared<CompilationUnit>();

  // 创建一个 SourceImporter 对象，用于导入源文件
  SourceImporter importer(
      compilation_unit,
      &constants,
      [&source](const std::string& qualifier) -> std::shared_ptr<Source> {
        // 从源文件映射中查找指定限定符的源文件
        auto source_iter = source.find(qualifier);
        if (source_iter == source.end()) {
          return nullptr;  // 如果找不到，返回空指针
        }
        // 创建并返回一个新的 Source 对象，包含指定的源码内容、限定符等信息
        return std::make_shared<Source>(
            source_iter->second, qualifier, 1, nullptr, Source::COPIES_STRING);
      },
      version);  // 版本号作为参数传递给 SourceImporter

  // 类型解析器函数，用于从限定名称加载类型
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    auto cls = importer.loadType(qn);  // 使用 importer 加载指定限定名称的类型
    return c10::StrongTypePtr(compilation_unit, std::move(cls));  // 返回加载的类型指针
  };

  // 使用 type_resolver 重新创建 ivalue 对象，并转换为 Object
  auto newIvalue = recreateObject(ivalue, type_resolver).toObject();

  // 使用新的 IValue 对象创建一个 Module 对象
  Module m(newIvalue);

  // 对 Module 进行量化卷积重写
  rewriteQuantizedConvForBC(m);

  // 返回创建的 Module 对象
  return m;
}

// 结束 torch::jit 命名空间
} // namespace torch::jit
```