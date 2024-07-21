# `.\pytorch\torch\csrc\jit\serialization\export_module.cpp`

```
// 包含 Torch 序列化相关的头文件

#include <torch/csrc/jit/serialization/export.h>

// 包含 C++ 标准库和 Torch 相关头文件

#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/backends/backend_debug_info.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/method.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

// 包含 Caffe2 序列化相关的头文件

#include <caffe2/serialize/inline_container.h>

// 包含 ATen 张量操作的头文件

#include <ATen/ATen.h>

// 包含 ATen 类型和限定名称相关的头文件

#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>

// 包含 C++ 标准库的头文件

#include <cerrno>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Torch JIT 命名空间

namespace torch::jit {

// 从全局获取编译选项并返回 CompilationOptions 对象

CompilationOptions getOptionsFromGlobal() {
  CompilationOptions compilation_options;

  // 设置 CompilationOptions 的各项选项，使用 BytecodeEmitMode 类来查询是否启用特定选项

  compilation_options.enable_default_args_before_out_args =
      BytecodeEmitMode::is_default_args_before_out_args_enabled();
  compilation_options.enable_default_value_for_unspecified_arg =
      BytecodeEmitMode::is_default_value_for_unspecified_arg_enabled();
  compilation_options.enable_emit_promoted_ops =
      BytecodeEmitMode::is_emit_promoted_ops_enabled();
  compilation_options.incl_interface_call = getMobileInterfaceCallExport();

  // 设置模型版本为 Caffe2 序列化定义的版本

  compilation_options.model_version =
      caffe2::serialize::kProducedBytecodeVersion;

  return compilation_options;
}

// 将 initializer_list 转换为 IValue 类型的元组对象并返回

static IValue to_tuple(std::initializer_list<IValue> ivalues) {
  return c10::ivalue::Tuple::create(ivalues);
}

// 将 vector<IValue> 转换为 IValue 类型的元组对象并返回

IValue to_tuple(std::vector<IValue> ivalues) {
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

// 创建并返回包含字符串和 IValue 对组的表对象的 IValue 类型元组对象

IValue Table(const std::vector<std::pair<std::string, IValue>>& entries) {
  std::vector<IValue> ivalue_entries;
  ivalue_entries.reserve(entries.size());

  // 将每个字符串和 IValue 对组转换为元组对象，并添加到 ivalue_entries 中

  for (const auto& e : entries) {
    ivalue_entries.push_back(to_tuple({e.first, e.second}));
  }

  // 调用 to_tuple 函数将 ivalue_entries 转换为最终的元组对象并返回

  return to_tuple(std::move(ivalue_entries));
}

// 匿名命名空间，用于定义静态函数 GetExtraFilesHook

namespace {

// 返回 ExportModuleExtraFilesHook 类型的静态函数引用，初始化为 nullptr

ExportModuleExtraFilesHook& GetExtraFilesHook() {
  static ExportModuleExtraFilesHook func = nullptr;
  return func;
}

}  // namespace

}  // namespace torch::jit
/**
 * 如果类型不是 NamedTuple，则返回 default_type_str。如果类型是 NamedTuple，
 * 则返回描述 NamedTuple 内容的字符串，其结构如下：
 * "qualified_named[ NamedTuple, [
 *     [field_name_1, field_type_1],
 *     [field_name_2, field_type_2],
 *     ...
 *   ]
 * ]"
 *  例如 NamedTuple 类型：
 *  "__torch__.base_models.sparse_nn.pytorch_preproc_types.PreprocOutputType[
 *     NamedTuple, [
 *         [float_features, Tensor],
 *         [id_list_features, List[Tensor]],
 *         [label,  Tensor],
 *         [weight, Tensor],
 *         ...
 *     ]"
 *
 * @param compilation_unit Jit 编译单元，用于查找函数模式。
 * @param type_ptr 类型指针，可能是任何类型。
 * @param default_type_str 默认的字符串表示。该字符串可以是 type_ptr->str()、type_ptr->annotation_str()
 *                        或 type_ptr->repr_str() 中的一个。在某些情况下，它们在不同的场景中可能不同。
 *                        例如，Tensor 类型可能是 "Tensor"、"Tensor (inferred)" 或 "Tensor[]"，我们只想要 "Tensor"。
 *                        当 type_ptr 不是 NamedTuple 时，将其作为默认返回值的一部分。
 * @return 字符串表示。
 */
std::string get_named_tuple_str_or_default(
    const CompilationUnit& compilation_unit,
    const TypePtr& type_ptr,
    std::string default_type_str) {
  // 如果类型是 TupleType
  if (type_ptr->kind() == TypeKind::TupleType) {
    // 对于简单类型（如 Tensor, Tensor），移动类型解析可以解析它，并且编译单元不会有其定义。
    // 此时返回默认的类型字符串。
    // 检查是否存在给定名称的命名元组类型定义
    if (compilation_unit.get_named_tuple(type_ptr->str())) {
      // 获取命名元组类型的指针
      auto named_tuple_ptr = compilation_unit.get_named_tuple(type_ptr->str());
      // 如果命名元组指针不为空
      if (named_tuple_ptr != nullptr) {
        // 获取命名元组类型的字符串表示
        std::string named_tuple_str = type_ptr->str();
        named_tuple_str.append("[NamedTuple, [");
        std::vector<IValue> name_type_pairs;

        // 遍历命名元组的字段名和字段类型
        for (auto it = named_tuple_ptr->schema()->arguments().begin();
             it != named_tuple_ptr->schema()->arguments().end();
             it++) {
          // 获取字段名
          const std::string named_tuple_name = it->name();
          // 获取字段类型
          const c10::TypePtr& named_tuple_type = it->type();
          // 如果字段类型是推断类型，Python 中 str() 返回 "Tensor"，否则返回 "Tensor[]"
          // 在 C++ 中，repr_str() 始终返回 "Tensor"
          std::string named_tuple_type_str = it->is_inferred_type()
              ? named_tuple_type->str()
              : named_tuple_type->repr_str();
          // 如果字段类型是 NamedTuple，则递归解析并获取其字符串表示
          named_tuple_type_str = get_named_tuple_str_or_default(
              compilation_unit, named_tuple_type, named_tuple_type_str);
          // 将字段名和类型组成的元组添加到 name_type_pairs 中
          name_type_pairs.emplace_back(
              c10::ivalue::Tuple::create({it->name(), named_tuple_type_str}));

          // 构建命名元组类型的字符串表示
          named_tuple_str.append("[")
              .append(named_tuple_name)
              .append(", ")
              .append(named_tuple_type_str)
              .append("]");
          // 如果不是最后一个字段，添加逗号分隔符
          if (it != named_tuple_ptr->schema()->arguments().end() - 1) {
            named_tuple_str.append(",");
          }
        }
        named_tuple_str.append("]]");
        // 返回构建的命名元组类型字符串
        return named_tuple_str;
      }
    }
  }
  // 如果没有找到匹配的命名元组类型定义，则返回默认类型字符串
  return default_type_str;
}

// 获取函数元组，返回一个由两个 IValue 对象组成的 std::pair
std::pair<IValue, IValue> getFunctionTuple(
    const CompilationUnit& compilation_unit,       // 编译单元的引用，用于获取类型信息
    const mobile::Function& func,                  // 移动函数对象的引用，表示当前处理的函数
    BackendDebugInfoRecorder& debug_info_recorder,  // 后端调试信息记录器的引用
    TypeNameUniquer& type_name_uniquer_) {          // 类型名称唯一化器的引用，用于处理类型名称的唯一性

  const auto& mobile_code = func.get_code();       // 获取移动函数对象的代码

  // instructions
  std::vector<IValue> instructions;                // 存储指令的 IValue 对象向量
  instructions.reserve(mobile_code.instructions_.size());  // 预留足够的空间以容纳所有指令
  for (Instruction ins : mobile_code.instructions_) {
    instructions.emplace_back(to_tuple({toString(ins.op), ins.X, ins.N}));  // 将指令转换为元组并存储到向量中
  }

  // operators
  std::vector<IValue> operators;                   // 存储操作符的 IValue 对象向量
  operators.reserve(mobile_code.op_names_.size()); // 预留足够的空间以容纳所有操作符
  for (const auto i : c10::irange(mobile_code.op_names_.size())) {
    const auto& opname = mobile_code.op_names_[i];  // 获取操作符的名称
    const int size = mobile_code.operator_input_sizes_[i];  // 获取操作符输入大小
    if (BytecodeEmitMode::is_default_value_for_unspecified_arg_enabled()) {
      operators.emplace_back(to_tuple({opname.name, opname.overload_name}));  // 如果未指定默认参数值，则存储操作符名和重载名
    } else {
      operators.emplace_back(
          to_tuple({opname.name, opname.overload_name, size}));  // 否则，存储操作符名、重载名和输入大小
    }
  }

  // types
  std::vector<IValue> types;                       // 存储类型信息的 IValue 对象向量
  types.reserve(mobile_code.types_.size());        // 预留足够的空间以容纳所有类型
  static const std::string torch_prefix("__torch__");  // Torch 类型前缀字符串
  static const std::string class_prefix("__torch__.torch.classes");  // Torch 类的类前缀字符串

  for (const TypePtr& ty : mobile_code.types_) {
    auto t = ty;                                  // 获取当前类型指针
    if (auto dyn = t->castRaw<c10::DynamicType>()) {
      t = dyn->fallback();                        // 如果是动态类型，则获取其回退类型
    }
    std::string type_str = t->annotation_str();   // 获取类型的注释字符串
    if (t->kind() == TypeKind::DictType) {
      // 对于字典类型，其包含两个子类型，分别是键和值，可能是命名元组类型
      const TypePtr& key_type = t->containedTypes()[0];     // 获取字典的键类型
      const TypePtr& value_type = t->containedTypes()[1];   // 获取字典的值类型
      std::string key_type_str = get_named_tuple_str_or_default(
          compilation_unit, key_type, key_type->annotation_str());  // 获取键类型的字符串表示
      std::string value_type_str = get_named_tuple_str_or_default(
          compilation_unit, value_type, value_type->annotation_str());  // 获取值类型的字符串表示

      // 构造字典的字符串表示，如 "Dict[str, ...]"
      std::string dict_str;
      dict_str.append("Dict[")
          .append(key_type_str)
          .append(",")
          .append(value_type_str)
          .append("]");
      types.emplace_back(dict_str);               // 将构造好的字典类型字符串存储到向量中
      continue;
    } else if (t->kind() == TypeKind::TupleType) {
      std::string named_tuple_str =
          get_named_tuple_str_or_default(compilation_unit, t, type_str);  // 获取命名元组类型的字符串表示
      types.emplace_back(named_tuple_str);       // 将命名元组类型字符串存储到向量中
      continue;
    }
  } else if (type_str.find(torch_prefix) == 0) {
    // 如果 type_str 是以 torch_prefix 开头的字符串，则执行以下逻辑
    TORCH_CHECK(
        type_str.find(class_prefix) == 0,
        "__torch__ types other than custom c++ classes (__torch__.torch.classes)"
        "are not supported in lite interpreter. ",
        "Workaround: instead of using arbitrary class type (class Foo()), ",
        "define a pytorch class (class Foo(torch.nn.Module)). The problematic type is: ",
        type_str);
    // 检查 type_str 是否以 class_prefix 开头，如果不是则抛出异常并显示相关信息
  }

  // 将 type_str 添加到 types 向量中
  types.emplace_back(type_str);

  // 将 register_size 转换为 int 类型
  auto register_size = static_cast<int>(mobile_code.register_size_);

  // 创建 codeTable 对象，存储 instructions、operators、constants、types 和 register_size
  auto codeTable = Table(
      {{"instructions", to_tuple(instructions)},
       {"operators", to_tuple(operators)},
       {"constants", to_tuple(mobile_code.constants_)},
       {"types", to_tuple(types)},
       {"register_size", register_size}});

  // 获取函数的 schema
  const auto& schema = func.getSchema();

  // 定义 type_printer 函数对象，用于获取类型的唯一名称
  auto type_printer = [&](const c10::Type& t) -> std::optional<std::string> {
    auto namedType = t.cast<c10::NamedType>();
    if (namedType && namedType->name()) {
      return type_name_uniquer_.getUniqueName(namedType).qualifiedName();
    }
    return c10::nullopt;
  };

  // 定义 makeArgTuple 函数对象，用于创建参数的元组
  auto makeArgTuple = [&](const std::vector<Argument>& args) {
    std::vector<IValue> argTables;
    for (auto&& arg : args) {
      TORCH_CHECK(
          !arg.N(),
          "Arguments with known list lengths are not supported in mobile modules.");
      TORCH_CHECK(
          !arg.kwarg_only(),
          "Keyword-only arguments are not supported in mobile modules.");
      /*
        This part adds the argument's name, type and default_value in
        `bytecode.pkl` This has to be consistent with the `code/` directory
        which has annotated py code of the entire module. `type_printer` uses
        `TypeNameUniquer` to get the managled name of the argument. This helps
        in having the right object reference when a class method is called using
        the `self` argument.

        arg.type()->annotation_str(type_printer) => mangled unique name of the
        module/submodule
      */
      auto arg_type = arg.type();
      if (auto dyn = arg_type->castRaw<c10::DynamicType>()) {
        arg_type = dyn->fallback();
      }
      // 将参数的名称、类型和默认值添加到 argTables 中
      argTables.emplace_back(Table({
          {"name", arg.name()},
          {"type", arg_type->annotation_str(type_printer)},
          {"default_value", arg.default_value()},
      }));
    }
    return to_tuple(argTables);
  };

  // 创建 schemaTable 对象，存储 arguments 和 returns
  auto schemaTable = Table({
      {"arguments", makeArgTuple(schema.arguments())},
      {"returns", makeArgTuple(schema.returns())},
  });

  // 定义字符串 qn，并根据函数名称确定其值
  std::string qn;
  if (func.name() == "__setstate__" || func.name() == "__getstate__") {
    auto classtype = func.getSchema().arguments()[0].type()->cast<ClassType>();
    TORCH_INTERNAL_ASSERT(
        classtype, "class is null ", func.qualname().qualifiedName());
    qn = c10::QualifiedName(
             type_name_uniquer_.getUniqueName(classtype), func.name())
             .qualifiedName();
  } else {
    qn = func.qualname().qualifiedName();
  }
  auto bytecode_vals = to_tuple({qn, codeTable, schemaTable});

  std::optional<IValue> debug_info_vals;
  // 获取函数的限定名
  // 根据不同情况选择合适的限定名：如果是类成员函数，使用类名加函数名；否则直接使用函数的限定名
  IValue module_debug_tuple =
      c10::ivalue::Tuple::create(mobile_code.debug_handles_);
  // 函数的调试信息
  // 这里存储了一组调试句柄。
  // 我们总是保存调试句柄。
  // 调试句柄由 debug_handle_manager 生成，
  // 对应于 {source_range, inlinedCallStackPtr}，我们将单独序列化它们。
  auto function_debug_info =
      Table({{"function_debug_handles", module_debug_tuple}});
  debug_info_vals = to_tuple({qn, function_debug_info});
  // 返回字节码和调试信息的元组
  return std::make_pair(bytecode_vals, debug_info_vals);
}

// 将移动模块的函数添加到元素列表和调试信息元素列表中
void pushMobileFunctionsToIValues(
    const CompilationUnit& compilation_unit,
    const mobile::Module& module,
    std::vector<c10::IValue>& elements,
    std::vector<c10::IValue>& debugInfoElements,
    BackendDebugInfoRecorder& recorder,
    TypeNameUniquer& uniquer) {
  // 遍历移动模块中的每个方法
  for (const auto& method : module.get_methods()) {
    // 获取函数元组
    auto tuple = getFunctionTuple(
        compilation_unit, method.function(), recorder, uniquer);
    // 将函数元组的第一个元素添加到元素列表
    elements.push_back(std::move(tuple.first));
    // 将函数元组的第二个元素添加到调试信息元素列表
    debugInfoElements.push_back(std::move(tuple.second));
  }
}

// 表示一个模块方法的结构体
struct ModuleMethod {
  // 构造函数，初始化模块、函数和导出名
  ModuleMethod(Module m, const GraphFunction& f, c10::QualifiedName n)
      : module(std::move(m)), function(f), exportName(std::move(n)) {}
  Module module;
  const GraphFunction& function;
  c10::QualifiedName exportName;
};

// 检查给定模块是否为降低的模块
bool isLoweredModule(const Module& m) {
  // 获取模块的类型名称
  c10::QualifiedName type_name;
  if (m.type()->name()) {
    type_name = m.type()->name().value();
  }
  bool isLoweredModule = false;
  // 检查类型名称是否包含 "LoweredModule" 字符串
  for (const auto& atom : type_name.atoms()) {
    if (atom == "LoweredModule") {
      isLoweredModule = true;
      break;
    }
  }
  return isLoweredModule;
}

// 检查全局静态映射的后端调试信息中是否包含该模块及其子模块的调试信息，如果有，则将所有映射组合并返回一个映射
void getBackendDebugInfoMap(
    const Module& m,
    BackendDebugInfoMapType& debug_map) {
  // 如果模块是降低的模块
  if (isLoweredModule(m)) {
    // 获取后端调试信息
    auto backend_debug_info =
        m.attr("__backend_debug_info").toCustomClass<PyTorchBackendDebugInfo>();
    const auto& map = backend_debug_info->getDebugInfoMap();
    if (map) {
      // 将调试信息映射插入到全局调试信息映射中
      debug_map.insert(map.value().begin(), map.value().end());
    }
  }
  // 递归处理子模块
  for (const auto& c : m.children()) {
    getBackendDebugInfoMap(c, debug_map);
  }
}

// 获取模块的后端源代码范围记录
SourceRangeRecords getBackendSourceRanges(const Module& m) {
  SourceRangeRecords sr_records;
  // 如果模块是降低的模块
  if (isLoweredModule(m)) {
    constexpr size_t kSourceRange = 1;
    // 获取后端调试信息
    auto backend_debug_info =
        m.attr("__backend_debug_info").toCustomClass<PyTorchBackendDebugInfo>();
    const auto& map = backend_debug_info->getDebugInfoMap();
    if (map) {
      const auto& map_val = map.value();
      // 遍历调试信息映射
      for (const auto& it : map_val) {
        // 获取源代码范围
        auto& source_range =
            std::get<kDebugInfoTupleSourceRangeIndex>(it.second);
        // 将源代码范围记录添加到记录列表中
        sr_records.emplace_back(
            std::numeric_limits<size_t>::max(), source_range);
        // 获取内联调用栈指针
        auto cs_ptr = std::get<kDebugInfoTupleInlinedCSIndex>(it.second);
        if (cs_ptr) {
          // 遍历内联调用栈指针中的每个元素
          for (const auto& e : cs_ptr->vec()) {
            // 获取源代码范围
            const auto sr = std::get<kSourceRange>(e);
            // 将源代码范围记录添加到记录列表中
            sr_records.emplace_back(std::numeric_limits<size_t>::max(), sr);
          }
        }
      }
    }
  }
    }
  }
  // 遍历给定节点的子节点
  for (const auto& c : m.children()) {
    // 获取子节点的后端源范围记录
    const auto& child_sr_records = getBackendSourceRanges(c);
    // 扩展sr_records以容纳子节点的源范围记录
    sr_records.reserve(sr_records.size() + child_sr_records.size());
    // 将子节点的源范围记录移动到sr_records中
    std::move(
        child_sr_records.begin(),
        child_sr_records.end(),
        std::back_inserter(sr_records));
  }
  // 返回完整的sr_records，包含了所有子节点的源范围记录
  return sr_records;
// 结束匿名命名空间

// 返回一个静态的原子布尔值标志，用于控制移动接口调用的导出
// 该函数用于保护 `InterfaceCall` 的使用，目前对 `InterfaceCall` 的支持应该已经成熟
auto& mobileInterfaceCallExport() {
  static std::atomic<bool> flag{true};
  return flag;
}

} // namespace

// 启用移动接口调用的导出
TORCH_API void enableMobileInterfaceCallExport() {
  // 设置 `mobileInterfaceCallExport` 标志为 true，使用 relaxed 内存顺序
  mobileInterfaceCallExport().store(true, std::memory_order_relaxed);
}

// 获取移动接口调用的导出状态
bool getMobileInterfaceCallExport() {
  return mobileInterfaceCallExport().load(std::memory_order_relaxed);
}

// 设置导出模块额外文件的钩子函数
void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook) {
  // 获取并移动额外文件钩子函数
  GetExtraFilesHook() = std::move(hook);
}

// 序列化脚本模块
void ScriptModuleSerializer::serialize(
    const Module& module,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  // 记录 Torch API 使用情况，仅记录一次
  C10_LOG_API_USAGE_ONCE("torch.jit.save");

  // 写入额外文件
  writeExtraFiles(module, extra_files);

  // 序列化模型对象
  writeArchive(
      module._ivalue(),
      /*archive_name=*/"data",
      /*archive_dir=*/"",
      /*tensor_dir=*/"data/");

  // 转换模块类型信息
  convertTypes(module.type());

  // 写入代码文件
  writeFiles("code/");

  // 将代码中的张量常量写入单独的存档，以确保加载代码不依赖于加载数据
  std::vector<IValue> ivalue_constants(
      constant_table_.begin(), constant_table_.end());
  if (bytecode_format) {
    writeArchive(
        c10::ivalue::Tuple::create(ivalue_constants),
        /*archive_name=*/"constants",
        /*archive_dir=*/"",
        /*tensor_dir=*/"constants/",
        /*use_storage_context=*/true);

    // 写入字节码
    writeByteCode(module, save_mobile_debug_info);
  } else {
    writeArchive(
        c10::ivalue::Tuple::create(ivalue_constants),
        /*archive_name=*/"constants",
        /*archive_dir=*/"",
        /*tensor_dir=*/"constants/");
  }

  // 如果存在跟踪输入，则写入存档
  if (!module.retrieve_traced_inputs().empty()) {
    writeArchive(
        module.retrieve_traced_inputs(),
        /*archive_name=*/"traced_inputs",
        /*archive_dir=*/"",
        /*tensor_dir=*/"traced_inputs/",
        /*use_storage_context=*/false,
        /*skip_tensor_data=*/true);
  }

  // 获取文件流并设置最小（动态）版本
  for (auto& item : file_streams_) {
    writer_.setMinVersion(item.value().minVersion());
  }
}

// 写入存档
void ScriptModuleSerializer::writeArchive(
    const IValue& value,
    const std::string& archive_name,
    const std::string& archive_dir,
    const std::string& tensor_dir,
    bool use_storage_context,
    bool skip_tensor_data) {
    // 省略的部分不需要注释
}
  std::vector<char> data;
  // 用于存储序列化数据的字符向量

  // Vector to capture the run-time class types during pickling the IValues
  // 用于在序列化 IValues 过程中捕获运行时类类型的向量
  std::vector<c10::ClassTypePtr> memoizedClassTypes;

  std::vector<std::string> tensor_names;
  // 存储张量名称的字符串向量

  // tensors that are already serialized in use_storage_context
  // 在 use_storage_context 中已经序列化的张量
  std::unordered_set<std::string> serialized_tensors;

  Pickler data_pickle(
      [&](const char* buf, size_t size) {
        data.insert(data.end(), buf, buf + size);
      },
      nullptr,
      [&](const c10::ClassTypePtr& t) {
        return type_name_uniquer_.getUniqueName(t);
      },
      &memoizedClassTypes,
      [&](const at::Tensor& tensor) {
        // returns a string to use in picker.cpp as storage obj key
        // 返回一个字符串，用作 picker.cpp 中存储对象键的标识
        if (use_storage_context) {
          bool already_serialized =
              storage_context_.hasStorage(tensor.storage());
          std::string tensor_name =
              std::to_string(
                  storage_context_.getOrAddStorage(tensor.storage())) +
              ".storage";
          if (already_serialized) {
            // this case is hit when storage has been serialized already
            // from a torch.package context
            // 当存储已经从 torch.package 上下文中序列化时，会命中此处情况
            serialized_tensors.insert(tensor_name);
          }
          tensor_names.push_back(tensor_name);
        } else {
          tensor_names.push_back(std::to_string(tensor_names.size()));
        }
        return tensor_names.back();
      });

  data_pickle.protocol();
  // 设置 Pickler 的协议版本

  data_pickle.pushIValue(value);
  // 将给定的 IValue 值推入 Pickler 中

  data_pickle.stop();
  // 停止 Pickler 操作

  // write out tensor data
  // 写出张量数据
  size_t i = 0;

  TORCH_INTERNAL_ASSERT(tensor_names.size() == data_pickle.tensorData().size());
  // 断言：张量名称的数量应与 Pickler 中的张量数据数量相同

  for (const auto& td : data_pickle.tensorData()) {
    std::string tensor_name = tensor_names[i++];
    if (td.is_meta() || skip_tensor_data) {
      // If metadata or skipping tensor data, write null record
      // 如果是元数据或者跳过张量数据，则写入空记录
      writer_.writeRecord(tensor_dir + tensor_name, nullptr, 0);
      continue;
    }
    WriteableTensorData writable_td = getWriteableTensorData(td);
    if (use_storage_context && serialized_tensors.count(tensor_name)) {
      // If storage has been serialized already, skip
      // 如果存储已经被序列化，则跳过
      continue;
    }
    writer_.writeRecord(
        tensor_dir + tensor_name,
        writable_td.data(),
        writable_td.sizeInBytes());
    // 写入张量数据记录
  }

  std::string fname = archive_dir + archive_name + ".pkl";
  writer_.writeRecord(fname, data.data(), data.size());
  // 写入最终的数据记录到文件

  // serialize all the captured run-time class types
  // 序列化捕获的所有运行时类类型
  for (const c10::ClassTypePtr& wroteType : memoizedClassTypes) {
    convertNamedType(wroteType);
    // 转换命名类型
  }
void ScriptModuleSerializer::writeExtraFiles(
    const Module& module,
    const ExtraFilesMap& extra_files) {
  // Write out extra files.
  // 遍历额外文件映射，将每个文件写入记录
  for (const auto& kv : extra_files) {
    const std::string key = "extra/" + kv.first;
    // 使用写入器写入记录，记录键为 "extra/" + 文件名，数据为文件内容，大小为文件大小
    writer_.writeRecord(key, kv.second.data(), kv.second.size());
  }
  auto hook = GetExtraFilesHook();
  // 获取额外文件钩子函数，并检查是否存在
  if (hook) {
    // 调用钩子函数获取额外文件映射
    ExtraFilesMap hook_files = hook(module);
    // 遍历钩子函数返回的文件映射
    for (const auto& kv : hook_files) {
      // 检查钩子返回的文件是否已经存在于额外文件中，如果是则跳过并发出警告
      if (extra_files.find(kv.first) != extra_files.end()) {
        TORCH_WARN_ONCE(
            "An extra files hook attempted to write ",
            kv.first,
            " but ",
            "this is already written in extra files and so will be skipped. ",
            "This warning will only appear once per process.");
        continue;
      }
      const std::string key = "extra/" + kv.first;
      // 使用写入器写入记录，记录键为 "extra/" + 文件名，数据为文件内容，大小为文件大小
      writer_.writeRecord(key, kv.second.data(), kv.second.size());
    }
  }
}

void ScriptModuleSerializer::updateSourceRangeTags(
    const SourceRangeRecords& ranges) {
  // 更新源范围标签
  for (const auto& range : ranges) {
    // 如果当前范围不在源范围标签中，则将其添加，并分配当前源范围标签
    if (source_range_tags_.find(range.range) == source_range_tags_.end()) {
      source_range_tags_[range.range] = current_source_range_tag_;
      current_source_range_tag_++;
    }
  }
}

void ScriptModuleSerializer::convertTypes(const at::NamedTypePtr& root_type) {
  // 添加类依赖项并遍历处理每个类的类型转换
  class_deps_.add(root_type);
  for (size_t i = 0; i < class_deps_.size(); ++i) {
    // 注意：convertNameType 可能会扩展 class_deps_，因此需要重新检查大小
    convertNamedType(class_deps_[i]);
  }
}

void ScriptModuleSerializer::writeFiles(const std::string& code_dir) {
  // 初始化当前源范围标签为零
  current_source_range_tag_ = 0;
  // 遍历文件流列表
  for (auto& item : file_streams_) {
    // 将限定符转换为存档路径，并获取文件名
    const std::string filename = qualifierToArchivePath(item.key(), code_dir);

    std::string src = item.value().str();

    // 只有在记录大小超过阈值 kMinToCompress 时才压缩这些记录
    static constexpr size_t kMinToCompress = 200;
    // 使用写入器写入记录，记录键为文件名，数据为文件内容，大小为文件大小，并根据是否压缩做相应处理
    writer_.writeRecord(
        filename,
        src.c_str(),
        src.size(),
        src.size() > kMinToCompress /*compress*/);

    // 写入调试信息
    std::string debugFilename = filename + ".debug_pkl";
    SourceRangePickler source_range_pickler;
    // 更新源范围标签
    updateSourceRangeTags(item.value().ranges());
    // 将范围数据序列化，并写入记录
    auto range_data =
        source_range_pickler.pickle(item.value().ranges(), source_range_tags_);
    writer_.writeRecord(
        debugFilename,
        range_data.data(),
        range_data.size(),
        range_data.size() > kMinToCompress /*compress*/);
  }
}

void ScriptModuleSerializer::writeByteCode(
    const Module& module,
    //
    const bool save_mobile_debug_info) {
  // 创建一个空的 IValue 元素列表
  std::vector<c10::IValue> elements;
  // 创建一个用于记录调试信息的对象
  BackendDebugInfoRecorder debug_info_recorder;
  // 设置要写入的字节码版本号
  int64_t version_to_write = caffe2::serialize::kProducedBytecodeVersion;

  // 将字节码版本号转换为 IValue，并添加到元素列表中
  elements.emplace_back(static_cast<int64_t>(version_to_write));
  // 创建一个用于存储调试信息元素的列表
  std::vector<c10::IValue> debug_info_elements;
  // 始终保存调试句柄信息，将版本号添加到调试信息元素列表中
  debug_info_elements.emplace_back(static_cast<int64_t>(version_to_write));

  // 将 JIT 模块转换为移动端模块
  mobile::Module mobile_module =
      jitModuleToMobile(module, getOptionsFromGlobal());

  // 将移动端模块的函数添加到元素列表中，同时记录调试信息
  pushMobileFunctionsToIValues(
      *module._ivalue()->compilation_unit(),
      mobile_module,
      elements,
      debug_info_elements,
      debug_info_recorder,
      type_name_uniquer_);

  // 将元素列表转换为元组
  auto telements = to_tuple(std::move(elements));
  // 将元组写入归档文件，保存为字节码
  writeArchive(
      telements,
      /*archive_name=*/"bytecode",
      /*archive_dir=*/"",
      /*tensor_dir=*/"constants/",
      /*use_storage_context=*/true);

  // 将调试信息元素列表转换为元组
  auto debug_info_telements = to_tuple(std::move(debug_info_elements));

  // 如果保存移动端调试信息开关开启
  if (save_mobile_debug_info) {
    // 写入调试信息元组到归档文件，保存为移动端调试句柄
    writeArchive(
        debug_info_telements,
        /*archive_name=*/"mobile_debug_handles",
        /*archive_dir=*/"",
        /*tensor_dir=*/"mobile_debug_handles/");
    static constexpr size_t kMinToCompress = 200;
    // 对于委托后端，获取包含在调试信息中的源范围
    // 由于委托后端替换原始模块为降低后的模块，无法序列化原始模块的代码
    // 因此从委托模块中提取源范围，并存储在单独的归档文件中
    auto backend_source_range_records = getBackendSourceRanges(module);
    // 创建源范围 pickler 对象
    SourceRangePickler source_range_pickler;
    // 更新源范围标签
    updateSourceRangeTags(backend_source_range_records);
    // 对源范围进行 pickling 处理，得到范围数据
    auto range_data = source_range_pickler.pickle(
        backend_source_range_records, source_range_tags_);
    // 设置调试文件名
    std::string debugFilename = "delegated_backends.debug_pkl";
    // 调用写入记录的方法，将调试文件名、数据、数据大小和是否需要压缩的标志写入
    writer_.writeRecord(
        debugFilename,
        range_data.data(),
        range_data.size(),
        range_data.size() > kMinToCompress /*compress*/);

    // 对于委托的后端，获取调试信息映射
    // 这将与其他未委托的模块的调试信息映射合并
    BackendDebugInfoMapType backend_debug_info_map;
    getBackendDebugInfoMap(module, backend_debug_info_map);

    // 现在获取调试句柄到内联调用堆栈指针映射
    // 并将其序列化到单独的存档中
    const auto& debug_info = mobile_module.getDebugTable().getCallStackPtrMap();
    BackendDebugInfoMapType debug_handle_cs_ptr_map(
        debug_info.begin(), debug_info.end());
    CallStackDebugInfoPickler cs_debug_info_pickler;
    // 使用调用堆栈调试信息拾取器对映射进行序列化
    auto cs_data = cs_debug_info_pickler.pickle(
        debug_handle_cs_ptr_map, source_range_tags_);

    // 写出映射数据：[调试句柄, {源范围, 内联调用堆栈}]
    std::string filename = "callstack_debug_map.pkl";
    writer_.writeRecord(
        filename,
        cs_data.data(),
        cs_data.size(),
        cs_data.size() > kMinToCompress /*compress*/);
}

namespace {

// 打印类型的可选字符串，如果类型是动态类型，则使用其回退类型的注释字符串
std::optional<std::string> type_printer(
    const c10::Type& type,
    torch::jit::TypeNameUniquer& type_name_uniquer) {
  if (auto dyn = type.castRaw<c10::DynamicType>()) {
    return dyn->fallback()->annotation_str(
        [&](auto&& t) { return type_printer(t, type_name_uniquer); });
  }
  auto namedType = type.cast<c10::NamedType>();
  // 如果是命名类型且有名称，则返回其唯一限定名称
  if (namedType && namedType->name()) {
    return type_name_uniquer.getUniqueName(namedType).qualifiedName();
  }
  // 否则返回空值
  return c10::nullopt;
}

} // namespace

// 转换命名类型为脚本模块序列化器
void ScriptModuleSerializer::convertNamedType(
    const c10::NamedTypePtr& class_type) {
  // 如果类型已经转换过，则直接返回
  if (converted_types_.count(class_type)) {
    return;
  }
  // 将类型标记为已转换
  converted_types_.insert(class_type);
  // 获取类型的唯一限定名称
  auto qualname = type_name_uniquer_.getUniqueName(class_type);
  std::string qualifier = qualname.prefix();
  // 查找类型对应的 PythonPrint 对象，若不存在则创建
  PythonPrint* pp = file_streams_.find(qualifier);

  if (!pp) {
    // 若不存在则创建 PythonPrint 对象并插入到文件流中
    pp = &file_streams_.insert(
        std::move(qualifier),
        PythonPrint(
            constant_table_,
            class_deps_,
            [&](const c10::Type& t) {
              return type_printer(t, type_name_uniquer_);
            },
            /*enforce_importable=*/true));
  }
  // 打印命名类型信息到 PythonPrint 对象
  pp->printNamedType(class_type);
}

// 序列化脚本模块的统一格式
void ScriptModuleSerializer::serialize_unified_format(
    Module& module,
    uint64_t script_module_id) {
  // 定义存档目录
  const std::string archive_dir =
      ".data/ts_code/" + std::to_string(script_module_id) + "/";

  // 序列化模型对象
  writeArchive(
      module._ivalue(),
      "data",
      archive_dir,
      /*tensor_dir=*/".data/",
      /*use_storage_context=*/true);
  // 转换模块类型信息
  convertTypes(module.type());
  // 将代码中的张量常量写入单独的存档文件
  std::vector<IValue> ivalue_constants(
      constant_table_.begin(), constant_table_.end());
  writeArchive(
      c10::ivalue::Tuple::create(ivalue_constants),
      "constants",
      archive_dir,
      /*tensor_dir=*/".data/",
      /*use_storage_context=*/true);

  // 注意：除了调用这个函数外，还需要调用 writeFiles() 来实际保存代码（张量会被保存）
}

// 返回序列化存储上下文的引用
SerializationStorageContext& ScriptModuleSerializer::storage_context() {
  return storage_context_;
}

// 导出模块到输出流
void ExportModule(
    const Module& module,
    std::ostream& out,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info,
    bool use_flatbuffer) {
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    // 使用输出流写入数据
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };
  // 调用另一个版本的 ExportModule 函数
  ExportModule(
      module,
      writer_func,
      extra_files,
      bytecode_format,
      save_mobile_debug_info,
      use_flatbuffer);
}

// 导出模块到文件名指定的位置
void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info,
    bool use_flatbuffer) {
    bool use_flatbuffer) {
  // 如果不使用 flatbuffer，则使用 PyTorchStreamWriter 创建文件流并序列化模型
  if (!use_flatbuffer) {
    // 创建一个 PyTorchStreamWriter 对象，指定要写入的文件名
    caffe2::serialize::PyTorchStreamWriter writer(filename);
    // 使用 ScriptModuleSerializer 对象对模型进行序列化
    ScriptModuleSerializer serializer(writer);
    serializer.serialize(
        module, extra_files, bytecode_format, save_mobile_debug_info);
    // 直接返回，不执行下面的代码
    return;
  }
  // 使用 flatbuffer
  // 打开一个二进制写入文件流，用于存储序列化后的模型
  std::ofstream ofile;
  ofile.open(filename, std::ios::binary | std::ios::out);
  // 检查文件打开是否失败
  if (ofile.fail()) {
    // 如果失败，根据错误码生成错误信息
    std::stringstream message;
    if (errno == ENOENT) {
      message << "Parent directory of " << filename << " does not exist.\n";
    } else {
      message << "Error while opening file: " << errno << '\n';
    }
    // 抛出异常并附上错误信息
    TORCH_CHECK(false, message.str());
  }
  // 调用 ExportModule 函数，将模型及相关参数序列化到文件中
  ExportModule(
      module,
      ofile,
      extra_files,
      bytecode_format,
      save_mobile_debug_info,
      use_flatbuffer);
}

// 将 JIT 模块保存到文件中
void save_jit_module(
    const Module& module,                        // 输入参数：JIT 模块的引用
    const std::string& filename,                 // 输入参数：保存文件的文件名
    const ExtraFilesMap& extra_files) {          // 输入参数：额外文件映射表
  auto buffer = save_jit_module_to_bytes(module, extra_files);  // 调用函数将 JIT 模块保存为字节流
  std::fstream ofile(filename, std::ios::binary | std::ios::out);  // 打开二进制文件流
  ofile.write(
      reinterpret_cast<char*>(buffer->data()), buffer->size()); // 写入字节流数据到文件 // NOLINT
  ofile.close();                                // 关闭文件流
}

// 将 JIT 模块保存为字节流
DetachedBuffer::UniqueDetachedBuffer save_jit_module_to_bytes(
    const Module& module,                       // 输入参数：JIT 模块的引用
    const ExtraFilesMap& extra_files) {         // 输入参数：额外文件映射表
  ExtraFilesMap jitfiles;                       // 存储 JIT 文件的映射表
  std::vector<IValue> constants;                // 存储常量的向量
  jitModuleToPythonCodeAndConstants(module, &jitfiles, &constants);  // 转换 JIT 模块为 Python 代码和常量
  CompilationOptions options = getOptionsFromGlobal();  // 获取编译选项
  mobile::Module mobilem = jitModuleToMobile(module, options);  // 将 JIT 模块转换为移动端模块
  return save_mobile_module_to_bytes(mobilem, extra_files, jitfiles, constants);  // 调用函数将移动端模块保存为字节流
}

// 将 JIT 模块保存为字节流，并通过提供的写入函数写入
void save_jit_module_to_write_func(
    const Module& module,                       // 输入参数：JIT 模块的引用
    const ExtraFilesMap& extra_files,           // 输入参数：额外文件映射表
    bool save_mobile_debug_info,                // 输入参数：是否保存移动端调试信息
    const std::function<size_t(const void*, size_t)>& writer_func) {  // 输入参数：写入函数
  (void)save_mobile_debug_info;                 // 忽略参数
  auto buffer = save_jit_module_to_bytes(module, extra_files);  // 调用函数将 JIT 模块保存为字节流
  writer_func(reinterpret_cast<void*>(buffer->data()), buffer->size());  // 调用提供的写入函数写入字节流数据
}

// 导出 JIT 模块
void ExportModule(
    const Module& module,                       // 输入参数：JIT 模块的引用
    const std::function<size_t(const void*, size_t)>& writer_func,  // 输入参数：写入函数
    const ExtraFilesMap& extra_files,           // 输入参数：额外文件映射表
    bool bytecode_format,                       // 输入参数：是否使用字节码格式
    bool save_mobile_debug_info,                // 输入参数：是否保存移动端调试信息
    bool use_flatbuffer) {                      // 输入参数：是否使用 FlatBuffer
  if (use_flatbuffer) {
    save_jit_module_to_write_func(
        module, extra_files, save_mobile_debug_info, writer_func);  // 如果使用 FlatBuffer，则调用函数保存 JIT 模块到提供的写入函数
  } else {
    caffe2::serialize::PyTorchStreamWriter writer(writer_func);  // 创建 PyTorchStreamWriter 对象
    ScriptModuleSerializer serializer(writer);   // 创建序列化器对象
    serializer.serialize(
        module, extra_files, bytecode_format, save_mobile_debug_info);  // 序列化 JIT 模块
  }
}

namespace {
// 导出操作名字集合
void export_opnames(const script::Module& m,    // 输入参数：脚本模块的引用
                    std::set<std::string>& opnames) {  // 输入输出参数：操作名字的集合
  mobile::Module mobile_m = jitModuleToMobile(m, getOptionsFromGlobal());  // 将脚本模块转换为移动端模块
  for (const auto& method : mobile_m.get_methods()) {  // 遍历移动端模块的方法
    for (const auto& op : method.function().get_code().op_names_) {
      // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
      opnames.emplace(
          op.overload_name.empty() ? op.name
                                   : op.name + "." + op.overload_name);  // 将操作名字加入集合中，处理重载名字
    }
  }
}
} // namespace

// 导出操作名字的函数
std::vector<std::string> export_opnames(const script::Module& m) {  // 输入参数：脚本模块的引用
  std::set<std::string> names;                  // 存储操作名字的集合
  export_opnames(m, names);                     // 调用函数导出操作名字
  return std::vector<std::string>(names.begin(), names.end());  // 将集合转换为向量并返回
}

// 线程本地标志，控制是否生成字节码的默认输入指令
thread_local bool emitBytecodeDefaultInputs =
    caffe2::serialize::kProducedBytecodeVersion <= 5 ? true : false;  // 判断生成字节码版本并赋初值

// 检查是否启用生成字节码的默认输入值
bool BytecodeEmitMode::is_default_value_for_unspecified_arg_enabled() {
  return emitBytecodeDefaultInputs;             // 返回是否启用生成字节码的默认输入值
}

// 设置是否启用生成字节码的默认输入值
void BytecodeEmitMode::set_default_value_for_unspecified_arg_enabled(
    # 设置 emitBytecodeDefaultInputs 变量为传入的 enabled 值
    emitBytecodeDefaultInputs = enabled;
}



// 结束 torch::jit 命名空间
} // namespace torch::jit

thread_local bool emitDefautlArgsWithOutArgs =
    caffe2::serialize::kProducedBytecodeVersion <= 6 ? false : true;



bool BytecodeEmitMode::is_default_args_before_out_args_enabled() {
  // 返回当前线程局部变量 emitDefautlArgsWithOutArgs 的值
  return emitDefautlArgsWithOutArgs;
}



void BytecodeEmitMode::set_default_args_before_out_args_enabled(bool enabled) {
  // 设置当前线程局部变量 emitDefautlArgsWithOutArgs 的值为 enabled
  emitDefautlArgsWithOutArgs = enabled;
}



thread_local bool emitDefaultEmitPromotedOps =
    caffe2::serialize::kProducedBytecodeVersion <= 7 ? false : true;



bool BytecodeEmitMode::is_emit_promoted_ops_enabled() {
  // 返回当前线程局部变量 emitDefaultEmitPromotedOps 的值
  return emitDefaultEmitPromotedOps;
}



void BytecodeEmitMode::set_default_emit_promoted_ops_enabled(bool enabled) {
  // 设置当前线程局部变量 emitDefaultEmitPromotedOps 的值为 enabled
  emitDefaultEmitPromotedOps = enabled;
}



// 结束文件或代码段
} // namespace torch::jit
```