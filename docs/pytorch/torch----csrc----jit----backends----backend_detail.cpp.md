# `.\pytorch\torch\csrc\jit\backends\backend_detail.cpp`

```py
// 返回一个对应于节点图的下一个调试句柄的API，用于预处理函数从模块的降级方法中获取节点的调试句柄。
// 实现：给定图形
// 对于图的每个节点，通过debug_info_recorder请求调试句柄。
// debug_info_recorder返回下一个调试句柄，并记录具有相应调试信息的节点，如源范围和内联调用堆栈。
NodeToDebugHandle generate_debug_handles(
    BackendDebugInfoRecorder& debug_info_recorder,
    const std::shared_ptr<Graph>& graph) {
  // 创建一个映射，将图中的节点映射到调试句柄
  NodeToDebugHandle node_to_debug_handles;

  // 创建一个堆栈以访问的块
  std::stack<Block*> blocks_to_visit;
  // TODO: 查看是否可以使用DepthFirstGraphNodeIterator
  // 目前它需要非const图，但也许我们可以通用化以使其可以适用于两者。
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    // 遍历块中的每个节点
    for (Node* n : b->nodes()) {
      // 获取节点的调试句柄
      DebugHandleType debug_handle = debug_info_recorder.getNextDebugHandle(n);
      // 将节点与调试句柄映射存储到映射表中
      node_to_debug_handles.emplace(n, debug_handle);
      // 将子块添加到堆栈中以进行后续处理
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  // 返回映射表，将图中的每个节点映射到其调试句柄
  return node_to_debug_handles;
}

// 返回后端预处理函数的映射，该映射存储了各个后端名称和对应的预处理函数
std::unordered_map<std::string, BackendPreprocessFunction>&
backendPreprocessFunctions() {
  // 静态局部变量，存储后端名称和预处理函数的映射
  static std::unordered_map<std::string, BackendPreprocessFunction>
      preprocess_functions;
  return preprocess_functions;
}

// 检查是否存在与给定名称对应的后端预处理函数
bool hasBackendPreprocessFunction(const std::string& name) {
  // 返回是否存在给定名称的后端预处理函数的布尔值
  return backendPreprocessFunctions().count(name);
}

// 注册后端预处理函数，将给定名称和对应的预处理函数添加到预处理函数映射中
void registerBackendPreprocessFunction(
    const std::string& name,
    BackendPreprocessFunction preprocess_function) {
  // 向后端预处理函数映射中注册新的预处理函数
  backendPreprocessFunctions()[name] = preprocess_function;
}
    const BackendPreprocessFunction& preprocess) {

# 定义函数 `registerBackendPreprocessFunction`，接收参数 `name`（后端名称）和 `preprocess`（预处理函数）
  TORCH_CHECK(

      !detail::hasBackendPreprocessFunction(name),

# 检查是否已经注册了指定名称的后端预处理函数，如果已注册则抛出错误
      "Preprocessing function for backend ",
      name,
      " is already registered. Ensure that registration is only called once.");
  detail::backendPreprocessFunctions()[name] = preprocess;

# 将指定名称的后端预处理函数与其预处理函数对象关联，存储到后端预处理函数映射表中
}

BackendPreprocessFunction getBackendPreprocessFunction(
    const std::string& name) {
  // 检查是否已注册给定名称的后端预处理函数
  TORCH_CHECK(
      hasBackendPreprocessFunction(name),
      "Preprocessing function for backend ",
      name,
      " is not registered.");
  // 返回给定名称对应的后端预处理函数
  return backendPreprocessFunctions()[name];
}

Module codegen_backend_module(
    const std::string& backend_name,
    const Module& orig_module,
    const c10::Dict<IValue, IValue>& method_compile_spec,
    // 从 method_compile_spec 中获取方法名称并转换为字符串
    std::string method_name = e.key().toStringRef();
    // 定义方法的代码模板
    static const auto method_ct = at::jit::CodeTemplate(R"(
            def $method(self${,def_inputs}):
                typed_inputs: List[Any] = [${fwd_inputs,}]
                // 如果后端可用，则执行方法
                if self.__backend.is_available() :
                  $unpack, = self.__backend.execute(self.__handles["$method"], typed_inputs)
                  ${refine,}
                  return $ret
                else:
                  // 抛出异常，指示后端不可用
                  raise Exception("Backend is not available.")
            )");
    // 定义包装方法的代码模板
    static const auto wrapper_method_ct = at::jit::CodeTemplate(R"(
            def $method(self${,def_inputs}):
                // 调用降级模块的对应方法
                return self.__loweredModule__.$method(${fwd_inputs})
            )");

    // 创建方法模板环境和包装方法模板环境
    at::jit::TemplateEnv method_te, wrapper_method_te;
    method_te.s("method", method_name);
    wrapper_method_te.s("method", method_name);
    // 获取原始模块中指定方法的引用
    auto method = orig_module.get_method(method_name);
    auto& function = method.function();
    auto& schema = function.getSchema();

    // 为函数签名（def_inputs）和传递给 backend.execute 的参数（fwd_inputs）生成输入列表
    std::vector<std::string> def_inputs, fwd_inputs;
    for (const auto& arg : schema.arguments()) {
      auto name = arg.name();

      // 跳过 self 参数，因为它只在签名中存在且始终存在
      if (name == "self") {
        continue;
      }

      auto default_value = arg.default_value();

      if (arg.kwarg_only()) {
        // 如果这是一个关键字参数，则需要在定义中和调用 backend_execute 时作为 keyword=value 发出
        TORCH_INTERNAL_ASSERT(default_value.has_value());
        std::stringstream def_ss, fwd_ss;
        // 注释参数的类型
        def_ss << name << ": " << arg.type()->annotation_str(nullptr) << "=";
        fwd_ss << name << "=" << name;
        default_value->repr(
            def_ss, [](std::ostream&, const IValue&) -> bool { return false; });
        def_inputs.emplace_back(def_ss.str());
        fwd_inputs.emplace_back(fwd_ss.str());
      } else {
        // 如果不是关键字参数，则在签名和调用 backend_execute 时按原样发出
        std::stringstream def_ss;
        // 注释参数的类型
        def_ss << name << ": " << arg.type()->annotation_str(nullptr);
        def_inputs.emplace_back(def_ss.str());
        fwd_inputs.emplace_back(name);
      }
    }

    // 生成逗号分隔的标识符列表以解包
    // 创建用于输出的字符串流和类型检查的字符串流
    std::stringstream out_ss, type_check_ss;
    // 存储类型检查语句的字符串向量
    std::vector<std::string> type_checks;
    // 断言返回的模式(schema)中只有一个返回值
    TORCH_INTERNAL_ASSERT(schema.returns().size() == 1);
    // 获取第一个返回类型
    auto out_ty = schema.returns().at(0).type();

    // 将第一个输出变量命名为"_0"
    out_ss << "_0";
    // 构建类型检查的基础字符串
    type_check_ss << "assert isinstance(_0, ";

    // 尝试将返回类型解析为元组类型
    auto out_tuple_ty = out_ty->cast<TupleType>();

    if (out_tuple_ty) {
      // 如果是元组类型，获取其元素列表
      auto tuple_elements = out_tuple_ty->elements();
      // 添加第一个元素的类型检查语句
      type_check_ss << tuple_elements[0]->annotation_str() << ")";
      type_checks.emplace_back(type_check_ss.str());
      // 遍历元组的其他元素，构建类型检查语句
      for (unsigned i = 1, e = tuple_elements.size(); i < e; ++i) {
        type_check_ss.str(std::string());
        type_check_ss.clear();
        out_ss << ", _" << i;
        type_check_ss << "assert isinstance(_" << i << ", "
                      << tuple_elements[i]->annotation_str() << ")";
        type_checks.emplace_back(type_check_ss.str());
      }
    } else {
      // 如果不是元组类型，直接添加返回类型的类型检查语句
      type_check_ss << out_ty->annotation_str() << ")";
      type_checks.emplace_back(type_check_ss.str());
    }

    // 设置模板引擎中的变量值
    method_te.v("def_inputs", def_inputs);
    method_te.v("fwd_inputs", fwd_inputs);
    method_te.v("refine", type_checks);
    method_te.s("unpack", out_ss.str());

    // 设置包装方法模板引擎中的变量值
    wrapper_method_te.v("def_inputs", def_inputs);
    wrapper_method_te.v("fwd_inputs", fwd_inputs);
    // 格式化并添加包装方法到包装方法列表
    wrapper_methods.push_back(wrapper_method_ct.format(wrapper_method_te));

    // 如果返回的类型是单元素元组，则在最终输出中添加一个额外的逗号
    if (out_tuple_ty && out_tuple_ty->elements().size() == 1) {
      out_ss << ",";
    }

    // 将最终输出字符串设置到模板引擎的"ret"变量中
    method_te.s("ret", out_ss.str());

    // 使用降低后的模块定义方法模板引擎格式化生成的方法，并定义到降低后的模块中
    loweredModule.define(method_ct.format(method_te), loweredModuleResolver());
  }

  // 如果后端可用，则调用__setstate__以确保返回的模块准备就绪
  // 否则，抛出警告表明返回的模块在加载到具有后端的设备之前不可执行
  loweredModule.run_method("__create_backend");
  if (loweredModule.run_method("__is_available").toBool()) {
    // 创建一个包含方法编译规范、处理后的模块属性和是否创建后端的状态的元组
    auto state = at::ivalue::Tuple::create(
        method_compile_spec,
        loweredModule.attr("__processed_module"),
        /*create_backend*/ false);
    // 调用__setstate__方法来设置模块的状态
    loweredModule.run_method("__setstate__", state);
  } else {
    // 发出警告消息，指示某个后端不可用，但仍然允许在具有该后端可用设备上保存和加载模块
    TORCH_WARN(
        "Backend [",
        backend_name,
        "] is not available. Execution of this Module is still possible by "
        "saving and loading on a device where the backend is available.");
    }
    
    // 停止记录调试信息并获取调试信息映射
    auto debug_info_map = debug_info_recorder.stopRecording();
    
    // 运行 loweredModule 对象的 __create_backend_debug_info 方法
    loweredModule.run_method("__create_backend_debug_info");
    
    // 从 loweredModule 对象中获取 __backend_debug_info 属性，并转换为 PyTorchBackendDebugInfo 自定义类
    auto backend_debug_info = loweredModule.attr("__backend_debug_info")
                                    .toCustomClass<PyTorchBackendDebugInfo>();
    
    // 将调试信息映射设置到 backend_debug_info 对象中
    backend_debug_info->setDebugInfoMap(std::move(debug_info_map));
    
    // 将 loweredModule 对象注册到 wrapper 中，以混淆自定义序列化逻辑
    wrapper.register_module("__loweredModule__", loweredModule);
    
    // 遍历 wrapper_methods 容器中的方法，并将其定义到 wrapper 中
    for (auto& method : wrapper_methods) {
      wrapper.define(method);
    }
    
    // 返回包含注册模块和定义方法的 wrapper 对象
    return wrapper;
}
} // namespace detail
} // namespace jit
} // namespace torch
```