# `.\pytorch\test\cpp\jit\test_backend_compiler_preprocess.cpp`

```
namespace torch {
namespace jit {
namespace {
// 对于这个后端，实际的编译过程发生在 preprocess 函数中。
// 放在这里用于展示整个后端的示例。在运行时后端库需要编译时会使用它。
// 如果在运行时没有编译需求，可以传入一个虚拟的函数。
c10::IValue preprocess(
    const Module& mod,
    const c10::Dict<IValue, IValue>& method_compile_spec,
    const BackendDebugHandleGenerator& generate_debug_handles) {
  // 创建一个空的字典用于存储编译后的结果
  c10::Dict<IValue, IValue> compiled(StringType::get(), StringType::get());

  // 遍历模块中的每个方法
  for (const auto& method : mod.get_methods()) {
    // 获取方法对应的图形函数，并复制其图形
    auto graph = toGraphFunction(method.function()).graph()->copy();

    // 必须内联图形以便调试信息映射
    Inline(*graph);

    // 由于要测试模块层次结构，可能会有 getattr 节点，在内联后没有实际作用。
    // 没有移除它们会导致编译错误，因此需要消除死代码以移除这些 getattr 节点。
    EliminateDeadCode(graph);

    // 获取方法名称作为键
    auto key = method.name();

    // 生成节点的调试句柄
    auto node_debug_handles = generate_debug_handles(graph);

    // 创建一个字符串流用于构建编译后的 blob
    std::stringstream ss;
    for (const auto& node : graph->nodes()) {
      // 根据节点类型进行处理
      switch (node->kind()) {
        case prim::Constant:
          ss << node->kind().toDisplayString() << "#"
             << toIValue(node->output()).value();
          ss << "<debug_handle>" << node_debug_handles[node];
          break;
        case aten::add:
          ss << node->kind().toQualString();
          ss << "<debug_handle>" << node_debug_handles[node];
          break;
        case aten::sub:
          ss << node->kind().toQualString();
          ss << "<debug_handle>" << node_debug_handles[node];
          break;
        default:
          // 如果节点类型不支持，抛出错误
          TORCH_CHECK(
              false,
              "The node of ",
              node->kind().toQualString(),
              " is not supported in this compiler. Source code: ",
              node->sourceRange().str());
          break;
      }
      ss << ",";
    }

    // 将流转换为字符串，并去除末尾的逗号
    std::string blob = ss.str();
    if (!blob.empty()) {
      blob.pop_back();
    }

    // 将编译后的 blob 插入到字典中，以方法名为键
    compiled.insert(method.name(), blob);
  }

  // 返回包含所有方法编译后结果的字典
  return compiled;
}

// 定义后端名称的常量表达式
constexpr auto backend_name = "backend_with_compiler_demo";

// 注册预处理函数到后端
static auto pre_reg = backend_preprocess_register(backend_name, preprocess);
} // namespace
} // namespace jit
} // namespace torch
```