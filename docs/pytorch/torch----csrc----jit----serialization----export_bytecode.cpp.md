# `.\pytorch\torch\csrc\jit\serialization\export_bytecode.cpp`

```py
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <utility>

#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/export.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/method.h>
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
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

#include <caffe2/serialize/inline_container.h>

namespace torch::jit {

// 静态函数，用于收集对象的所有 __setstate__ 和 __getstate__ 方法
static std::vector<Method> gatherGetSetStates(const ObjectPtr& obj) {
  std::vector<Method> methods;
  // 使用深度优先搜索遍历对象的依赖关系，将所有 setstate/getstates 添加到初始堆栈中
  std::vector<ObjectPtr> ivalue_stack;
  ivalue_stack.emplace_back(obj);
  while (!ivalue_stack.empty()) {
    ObjectPtr cur = ivalue_stack.back();
    ivalue_stack.pop_back();
    auto type = cur->type();
    Function* setstate = type->findMethod("__setstate__");
    Function* getstate = type->findMethod("__getstate__");
    if (getstate && setstate) {
      if (setstate->isGraphFunction()) {
        methods.emplace_back(cur, setstate);
      }
      if (getstate->isGraphFunction()) {
        methods.emplace_back(cur, getstate);
      }
    } else {
      for (size_t i = 0, n = type->numAttributes(); i < n; ++i) {
        IValue field = cur->getSlot(i);
        if (field.isObject()) {
          ivalue_stack.emplace_back(field.toObject());
        }
      }
    }
  }
  return methods;
}

// 静态函数，用于查找模块中所有依赖的函数
static std::vector<Method> findAllDependentFunctions(
    const Module& module,
    Graph& graph) {
  std::vector<Method> methods;
  std::unordered_set<c10::string_view> called_method_names;
  auto nodes = findAllNodes(graph, c10::prim::CallMethod, true);
  for (Node* node : nodes) {
    if (auto iface = node->input(0)->type()->castRaw<InterfaceType>()) {
      const FunctionSchema* schema = iface->getMethod(node->s(attr::name));
      called_method_names.insert(schema->name());
    }
  }



// 循环结束，退出了所有的嵌套循环，函数的执行即将结束
for (const auto& submodule : module.modules()) {
    // 遍历 module 下的每个 submodule
    for (const auto& m : submodule.get_methods()) {
        // 遍历每个 submodule 的方法列表
        // 检查当前方法的名称是否在 called_method_names 集合中
        if (called_method_names.find(m.function().qualname().name()) !=
            called_method_names.end()) {
            // 如果找到当前方法的名称，则将该方法加入到 methods 列表中
            methods.emplace_back(m);
        }
    }
}
// 返回收集到的方法列表
return methods;



// 函数结束，返回 methods 列表作为结果
// NOTE: order of functions returned will be:
// 1. functions originated from the methods passed in will be first
// 2. All the dependent functions will come afterwards.
// This order is meaningful because currently mobile Module looks up
// methods with linear search.
static std::vector<std::unique_ptr<GraphFunction>> inlineFunctions(
    const std::vector<Method>& initial_methods,
    bool incl_dependent_functions) {
  // 用于记录已访问的方法及其函数指针的集合
  std::set<std::pair<std::string, Function*>> visited;
  // 用于存放待处理的方法队列
  std::deque<Method> stack;
  // 将初始方法复制到堆栈中
  std::copy(
      initial_methods.begin(),
      initial_methods.end(),
      std::back_inserter(stack));
  // 存放内联函数的列表
  std::vector<std::unique_ptr<GraphFunction>> inlined_functions;
  // 处理堆栈中的方法，直到堆栈为空
  while (!stack.empty()) {
    // 从堆栈中取出当前方法
    Method cur = stack.front();
    stack.pop_front();
    // 创建一个表示方法所有者和函数指针的元组
    auto tup = std::make_pair(
        cur.owner()._ivalue()->type()->name()->qualifiedName(),
        &cur.function());
    // 如果已经访问过这个方法，则跳过
    if (visited.find(tup) != visited.end()) {
      continue;
    }
    // 将该方法标记为已访问
    visited.insert(tup);
    // 获取方法对应的图函数对象
    const auto& f = toGraphFunction(cur.function());
    // 复制图对象以便内联化
    auto graph = f.graph()->copyUnique();
    // 执行内联化操作
    Inline(*graph);
    // 创建方法的限定名
    c10::QualifiedName qn(*cur.owner()._ivalue()->type()->name(), f.name());

    // 如果需要包括依赖函数，则查找所有依赖的方法并加入堆栈
    if (incl_dependent_functions) {
      std::vector<Method> dependent_methods =
          findAllDependentFunctions(cur.owner(), *graph);
      std::copy(
          dependent_methods.begin(),
          dependent_methods.end(),
          std::back_inserter(stack));
    }
    // 创建一个新的内联函数对象并添加到列表中
    auto inlined_func = std::make_unique<GraphFunction>(
        qn, std::move(graph), f.function_creator());
    inlined_func->setSchema(f.getSchema());
    inlined_functions.emplace_back(std::move(inlined_func));
  }
  // 返回所有内联函数的列表
  return inlined_functions;
}

mobile::Code compileGraphToMobileCode(
    const std::string& name,
    const std::shared_ptr<Graph>& graph,
    const CompilationOptions& compilation_options,
    BackendDebugInfoRecorder& debug_info_recorder) {
  // 创建一个移动端代码对象
  MobileCode code(
      graph,
      name,
      compilation_options.enable_default_value_for_unspecified_arg,
      compilation_options.enable_default_args_before_out_args,
      compilation_options.enable_emit_promoted_ops);

  // 创建一个移动端代码对象的实例
  mobile::Code mobile_code;

  // 运算符名称列表
  std::vector<std::string> method_names;
  // 运算符调试句柄列表
  std::vector<int64_t> op_debug_handles;
  // 下一个新的运算符索引
  int next_new_op_index = 0;

  // 获取操作符到指定参数的映射关系
  auto op_to_specified_args = code.op_to_num_specified_args();

  // 遍历代码指令列表
  for (size_t i = 0; i < code.instructions().size(); ++i) {
    // 获取当前指令
    Instruction ins = code.instructions()[i];
    // 检查指令类型是否为 OP 或者 OPN，并`
    // 检查指令的操作码是否为 OP 或者 OPN，并且 X 字段是否等于 next_new_op_index
    if ((ins.op == OP || ins.op == OPN) && ins.X == next_new_op_index) {
      // 发现一个新操作符（假设新操作符按照 ins.X 升序排列）
      auto node = code.instructions_source()[i];
      // 获取操作符的名称
      const c10::OperatorName& opname = node->schema().operator_name();
      // 将操作符名称转换为字符串
      auto unique_name = c10::toString(opname);
      // 对于具有可变参数的操作符，添加默认参数可能会导致混淆，因此不允许。
      // 对于 num_args = -1 的操作符，表示此操作符的参数数量在运行时不可用，
      // 我们不会在运行时进行任何向后兼容的适配。
      std::optional<int> num_args = c10::nullopt;
      // 查找操作符名称在 op_to_specified_args 中的映射
      auto it = op_to_specified_args.find(unique_name);
      if (it != op_to_specified_args.end()) {
        num_args = it->second;
      }
      // 将操作符的输入大小信息加入 mobile_code.operator_input_sizes_
      mobile_code.operator_input_sizes_.emplace_back(num_args.value_or(-1));
      // 将操作符名称加入 mobile_code.op_names_
      mobile_code.op_names_.emplace_back(opname);
      // 创建操作符函数对象
      auto func = mobile::makeOperatorFunction(opname, num_args);
      // 断言操作符函数是否成功创建
      TORCH_INTERNAL_ASSERT(
          func.has_value(),
          "Operator with name: ",
          toString(opname),
          " not found");
      // 将操作符函数加入 mobile_code.operators_
      mobile_code.operators_.emplace_back(*func);
      // 更新下一个新操作符的索引
      next_new_op_index++;
    }
    // 此处的 CALL 指令表示未内联的内置（非图形）函数调用。
    // 在这里，我们将这些函数的 CALL 指令转换为 INTERFACE_CALL 指令，
    // 运行时将根据栈中第 0 个参数的类型查找并直接调用该函数。
    if (ins.op == CALL) {
      auto node = code.instructions_source()[i];
      if (node->kind() == prim::CallMethod) {
        // 注意：替换指令
        // 计算方法名在常量表和方法名列表中的索引
        auto method_name_idx =
            code.constant_table().size() + method_names.size();
        // 将方法名加入方法名列表
        method_names.emplace_back(node->s(attr::name));
        // 替换指令为 INTERFACE_CALL
        ins = Instruction{
            INTERFACE_CALL,
            static_cast<int32_t>(method_name_idxc_cast<int32_t>(method_name_idx),
            static_cast<uint16_t>(node->inputs().size())};
      } else {
        // 如果节点类型不支持，则抛出错误信息
        TORCH_INTERNAL_ASSERT(
            false, "Unsupported node kind on CALL opcode for mobile");
      }
    } else if (ins.op == RET) {
      // 如果指令是返回指令（RET），处理返回指令的相关逻辑
      auto node = code.instructions_source()[i];
      // 获取当前指令的源节点信息
      for (const auto& input : node->inputs()) {
        // 遍历当前节点的输入
        const auto& input_type = input->type();
        // 获取输入的类型信息
        if (input_type->kind() == TypeKind::ListType ||
            input_type->kind() == TypeKind::DictType) {
          // 如果输入类型是列表类型或字典类型
          for (const TypePtr& element_type : input_type->containedTypes()) {
            // 遍历包含在列表或字典中的元素类型
            TORCH_CHECK(
                element_type->kind() != TypeKind::ClassType,
                "Returning a list or dictionary with pytorch class type ",
                "is not supported in mobile module "
                "(List[Foo] or Dict[int, Foo] for class Foo(torch.nn.Module)). "
                "Workaround: instead of using pytorch class as their element type, ",
                "use a combination of list, dictionary, and single types.");
          }
        }
      }
    } else {
      // 对于不是返回指令的其他指令，检查是否支持在移动模块中使用
      TORCH_CHECK(
          isOpSupportedInMobile(ins.op),
          toString(ins.op),
          " is not supported in mobile module.");
    }
    auto node = code.instructions_source()[i];
    // 获取当前指令的源节点信息
    int64_t debug_handle = debug_info_recorder.getNextDebugHandle(node);
    // 获取当前指令的调试信息句柄
    // 注意：指令与调试句柄之间是一对一的对应关系
    mobile_code.instructions_.emplace_back(ins);
    // 将指令添加到移动代码的指令列表中
    mobile_code.debug_handles_.emplace_back(debug_handle);
    // 将调试句柄添加到移动代码的调试句柄列表中
  }

  // 复制常量
  mobile_code.constants_ = code.constant_table();

  // 复制常量并添加上面转换的 INTERFACE_CALL 节点的方法名称
  for (auto& method_name : method_names) {
    mobile_code.constants_.emplace_back(method_name);
    // 将方法名称添加到移动代码的常量列表中
  }

  mobile_code.types_ = code.type_table();
  // 设置移动代码的类型表
  mobile_code.register_size_ = code.register_size();
  // 设置移动代码的寄存器大小
  return mobile_code;
  // 返回构建好的移动代码对象
// 将 JIT 函数转换为移动端函数对象的唯一指针
std::unique_ptr<mobile::Function> convertJitFunctionToMobileFunction(
    const GraphFunction& function,                  // 输入的 JIT 图函数对象
    const CompilationOptions& options) {            // 编译选项
  BackendDebugInfoRecorder debug_handle;            // 后端调试信息记录器
  auto mobileCode = compileGraphToMobileCode(       // 编译 JIT 图为移动端代码
      function.name(), function.graph(), options, debug_handle);  // 使用函数名、图、选项和调试信息进行编译
  const auto& schema = function.getSchema();        // 获取函数的架构信息
  return std::make_unique<mobile::Function>(       // 返回移动端函数对象的唯一指针
      function.qualname(), std::move(mobileCode), schema);  // 使用函数限定名、移动端代码和架构来构造函数对象
}

// 将移动端函数对象转换为代码表
IValue convertMobileFunctionToCodeTable(
    const mobile::Function& func,                   // 输入的移动端函数对象
    const CompilationOptions& compilation_options) {// 编译选项
  auto code = func.get_code();                      // 获取函数对象的代码对象
  std::vector<IValue> instructions;                 // 存储指令的 IValue 向量
  instructions.reserve(code.instructions_.size());  // 预留指令数量的空间
  for (Instruction ins : code.instructions_) {      // 遍历函数对象的指令
    instructions.emplace_back(to_tuple({toString(ins.op), ins.X, ins.N}));  // 将指令操作码、X 和 N 转换为元组存入向量
  }

  std::vector<IValue> operators;                    // 存储操作符的 IValue 向量
  operators.reserve(code.op_names_.size());         // 预留操作符数量的空间
  for (unsigned i = 0; i < code.op_names_.size(); ++i) {  // 遍历函数对象的操作符
    const auto& opname = code.op_names_[i];         // 获取操作符名称
    const int size = code.operator_input_sizes_[i]; // 获取操作符输入大小
    if (compilation_options.enable_default_value_for_unspecified_arg) {
      operators.emplace_back(to_tuple({opname.name, opname.overload_name}));  // 若启用未指定参数的默认值，则存入名称和重载名称
    } else {
      operators.emplace_back(
          to_tuple({opname.name, opname.overload_name, size}));  // 否则存入名称、重载名称和输入大小
    }
  }

  std::vector<IValue> types;                        // 存储类型的 IValue 向量
  for (const TypePtr& t : code.types_) {            // 遍历函数对象的类型
    std::string type_str = t->annotation_str();     // 获取类型的注释字符串
    types.emplace_back(type_str);                   // 将类型注释字符串存入向量
  }

  auto register_size = static_cast<int>(code.register_size_);  // 获取寄存器大小并转换为整数
  auto codeTable = Table(                           // 创建代码表对象
      {{"instructions", to_tuple(instructions)},    // 包含指令
       {"operators", to_tuple(operators)},          // 包含操作符
       {"constants", to_tuple(code.constants_)},    // 包含常量
       {"types", to_tuple(types)},                  // 包含类型
       {"register_size", register_size}});          // 包含寄存器大小

  return codeTable;                                 // 返回代码表对象
}

// 检查函数架构是否符合要求
static void checkSchema(const c10::FunctionSchema& schema) {
  TORCH_CHECK(
      schema.overload_name().empty(),               // 检查重载名称是否为空，不支持重载
      "Overloads are not supported in mobile modules.");  // 若不为空则抛出异常，不支持移动模块中的重载
  TORCH_CHECK(
      !schema.is_vararg(),                         // 检查是否使用 Python *args，不支持
      "Python *args are not supported in mobile modules.");  // 若使用则抛出异常，不支持在移动模块中使用 Python *args
  TORCH_CHECK(
      !schema.is_varret(),                         // 检查是否存在可变数量的返回值，不支持
      "A variable number of return values is not supported in mobile modules.");  // 若存在则抛出异常，不支持在移动模块中返回可变数量的值
}

// 检查模块是否为降低的模块
static bool isLoweredModule(const Module& m) {
  c10::QualifiedName type_name;                    // 定义类型的限定名
  if (m.type()->name()) {                          // 若模块有名称
    type_name = m.type()->name().value();          // 获取模块的限定名
  }
  bool isLoweredModule = false;                    // 初始化是否为降低模块为假
  for (const auto& atom : type_name.atoms()) {     // 遍历限定名中的原子
    if (atom == "LoweredModule") {                 // 若找到 "LoweredModule" 原子
      isLoweredModule = true;                      // 标记为降低模块
      break;
    }
  }
  return isLoweredModule;                          // 返回是否为降低模块的标志
}

// 检查全局静态后端调试信息映射中是否包含该模块及其任何子模块的调试信息，并合并返回
static void getBackendDebugInfoMap(
    const Module& m,                               // 输入的模块
    BackendDebugInfoMapType& debug_map) {           // 后端调试信息映射
  if (isLoweredModule(m)) {                        // 检查是否为降低模块
    auto backend_debug_info =                      // 获取模块的后端调试信息
        m.attr("__backend_debug_info").toCustomClass<PyTorchBackendDebugInfo>();
    const auto& map = backend_debug_info->getDebugInfoMap();
    // 获取后端调试信息的映射，存储在常量引用 map 中
    if (map) {
      // 如果映射有效，则将其内容插入到 debug_map 中
      debug_map.insert(map.value().begin(), map.value().end());
    }
  }
  // 递归处理每个子节点 m，获取其后端调试信息映射并存入 debug_map
  for (const auto& c : m.children()) {
    getBackendDebugInfoMap(c, debug_map);
  }
// 获取模块中最小的操作符版本号，从版本映射中查找并返回
static uint64_t get_min_operator_version_from_version_map(
    const mobile::Module& module) {
  // 初始设定最小版本为 Caffe2 序列化支持的最小文件格式版本号
  uint64_t min_version = caffe2::serialize::kMinSupportedFileFormatVersion;
  // 遍历模块中所有的方法
  for (const auto& func : module.compilation_unit().methods()) {
    // 遍历每个方法中的操作符名称
    for (const auto& op_name : func->get_code().op_names_) {
      // 构建操作符的完整名称（如果有重载的话）
      auto schema_name = op_name.overload_name.empty()
          ? op_name.name
          : op_name.name + "." + op_name.overload_name;
      // 在操作符版本映射中查找当前操作符的版本信息
      auto version_entry = get_operator_version_map().find(schema_name);
      // 如果找到了对应的版本信息
      if (version_entry != get_operator_version_map().end()) {
        const auto& entry = version_entry->second;
        // 更新最小版本号为当前版本号和已记录版本号的最大值
        min_version = std::max(
            min_version, uint64_t(entry[entry.size() - 1].bumped_at_version));
      }
    }
  }
  // 返回计算出的最小操作符版本号
  return min_version;
}

// 将 JIT 模块转换为移动端模块
mobile::Module jitModuleToMobile(
    const Module& module,
    const CompilationOptions& options) {
  // 创建一个共享指针指向移动端编译单元
  std::shared_ptr<mobile::CompilationUnit> mcu =
      std::make_shared<mobile::CompilationUnit>();
  // 创建调试信息记录器
  BackendDebugInfoRecorder debug_info_recorder;

  // 获取要导出的方法列表，包括获取和设置状态的方法
  std::vector<Method> methods_to_export = module.get_methods();
  std::vector<Method> getsetstates = gatherGetSetStates(module._ivalue());
  std::copy(
      getsetstates.begin(),
      getsetstates.end(),
      std::back_inserter(methods_to_export));

  // 对每个内联函数进行处理，生成移动端代码并注册到编译单元中
  for (const auto& func :
       inlineFunctions(methods_to_export, options.incl_interface_call)) {
    // 编译图形式函数为移动端代码
    auto mobile_code = compileGraphToMobileCode(
        func->name(), func->graph(), options, debug_info_recorder);
    // 获取函数的模式(schema)
    const auto& schema = func->getSchema();
    // 检查模式(schema)的有效性
    checkSchema(schema);
    // 创建移动端函数对象并注册到编译单元中
    auto mobile_func = std::make_unique<mobile::Function>(
        func->qualname(), std::move(mobile_code), schema);
    mcu->register_function(std::move(mobile_func));
  }

  // 使用模块的 IValue 创建移动端模块
  mobile::Module m(module._ivalue(), mcu);
  // 设置调试句柄标志为真
  m.setHasDebugHandles(true);
  // 创建后端调试信息映射
  BackendDebugInfoMapType backend_debug_info_map;
  // 获取模块的后端调试信息映射
  getBackendDebugInfoMap(module, backend_debug_info_map);
  // 停止记录调试句柄和代码片段指针映射
  auto debug_handle_cs_ptr_map = debug_info_recorder.stopRecording();
  // 将调试句柄和代码片段指针映射插入到调试信息表中
  debug_handle_cs_ptr_map.insert(
      backend_debug_info_map.begin(), backend_debug_info_map.end());
  // 设置调试表
  m.setDebugTable(MobileDebugTable(
      debug_handle_cs_ptr_map.begin(), debug_handle_cs_ptr_map.end()));
  // 设置模块的最小操作符版本号
  m.set_min_operator_version(get_min_operator_version_from_version_map(m));
  // 设置字节码版本号
  m.set_bytecode_version(options.model_version);
  // 返回移动端模块
  return m;
}

// 命名空间结束：torch::jit
} // namespace torch::jit
```