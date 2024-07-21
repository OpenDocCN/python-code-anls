# `.\pytorch\torch\csrc\jit\passes\freeze_module.cpp`

```
#include <torch/csrc/jit/passes/freeze_module.h>  // 包含冻结模块所需的头文件

#include <torch/csrc/jit/jit_log.h>  // 包含 JIT 日志相关的头文件

#include <c10/util/irange.h>  // 包含 C10 库中的整数范围实用函数的头文件
#include <torch/csrc/jit/api/function_impl.h>  // 包含 JIT 函数实现的头文件
#include <torch/csrc/jit/ir/alias_analysis.h>  // 包含 JIT 别名分析相关的头文件
#include <torch/csrc/jit/passes/autocast.h>  // 包含 JIT 自动类型转换相关的头文件
#include <torch/csrc/jit/passes/clear_profiling.h>  // 包含 JIT 清除分析数据相关的头文件
#include <torch/csrc/jit/passes/eliminate_no_ops.h>  // 包含 JIT 消除无操作节点相关的头文件
#include <torch/csrc/jit/passes/inliner.h>  // 包含 JIT 内联函数相关的头文件
#include <torch/csrc/jit/passes/lower_tuples.h>  // 包含 JIT 降低元组操作相关的头文件
#include <torch/csrc/jit/runtime/graph_executor_impl.h>  // 包含 JIT 图执行器实现的头文件

#include <stack>  // 包含 C++ 标准库中的栈容器的头文件
#include <utility>  // 包含 C++ 标准库中的实用工具的头文件

namespace torch {
namespace jit {

namespace {

std::vector<std::string> splitName(const std::string& name) {
  std::vector<std::string> result;  // 创建存储分割后子字符串的结果向量
  std::string sub_name;  // 声明存储每个子字符串的变量
  std::istringstream name_stream(name);  // 创建基于字符串流的输入流
  while (std::getline(name_stream, sub_name, '.')) {  // 使用'.'分割字符串并存储到 result 中
    result.push_back(std::move(sub_name));
  }
  return result;  // 返回分割后的子字符串向量
}

template <typename Iter>
std::string concatName(const Iter& begin, const Iter& end) {
  std::string combined_name = "";  // 创建空字符串以存储组合后的名称
  for (Iter it = begin; it != end; ++it) {
    const std::string& sub_name = *it;  // 获取当前迭代器指向的子字符串
    if (!combined_name.empty()) {
      combined_name += ".";  // 如果不是第一个子字符串，则在前面加上'.'
    }
    combined_name += sub_name;  // 将当前子字符串添加到组合后的名称中
  }
  return combined_name;  // 返回组合后的名称字符串
}

class AttributePropagator {
 public:
  AttributePropagator(
      Module& module,
      std::vector<std::string>& preservedAttrs,
      bool freezeInterfaces,
      bool preserveParameters)
      : module_(module),
        freezeInterfaces_(freezeInterfaces),
        preserveParameters_(preserveParameters) {
    auto checkName = [this](std::string& name) {  // 定义 lambda 函数检查名称是否存在并处理
      const auto resolved_name = resolveName(name);  // 解析名称以获取实际模块和属性名称

      if (resolved_name) {
        const auto& parent_module = resolved_name->first;
        const auto& attr_name = resolved_name->second;
        if (parent_module.hasattr(attr_name)) {  // 如果父模块具有指定属性
          auto value = parent_module.attr(attr_name);  // 获取属性的值
          // 如果值是模块，且需要保留接口，则将其插入保留的子模块集合中
          if (value.isModule()) {
            preservedSubModule_.insert(value.toModule()._ivalue());
          }
          insertMutableAttr(attr_name, value, parent_module._ivalue());  // 插入可变属性
        } else {
          auto fn = parent_module.get_method(attr_name);  // 获取方法对象
          preservedMethods_.insert(&fn.function());  // 将方法对象添加到保留方法集合中
        }
        return true;  // 返回名称存在的标志
      }

      return false;  // 返回名称不存在的标志
    };

    // 默认情况下保留 forward 方法，但并非所有模块都定义了 forward 方法
    if (module_.find_method("forward")) {
      auto method = module_.get_method("forward");  // 获取 forward 方法
      preservedMethods_.insert(&method.function());  // 将 forward 方法添加到保留方法集合中
    }

    for (auto name : preservedAttrs) {
      TORCH_CHECK(checkName(name), "Unknown name: " + name);  // 检查每个保留的属性名称是否存在
    }
  }

  void optimizeSubGraphs(
      std::shared_ptr<Graph>& graph,
      const std::function<void(std::shared_ptr<Graph>&)>& func) {
    func(graph);  // 执行传入的优化函数
    std::stack<Block*> blocks({graph->block()});  // 创建堆栈以处理子图的块
    // 当堆栈不为空时执行循环，处理每个块及其子块中的节点
    while (!blocks.empty()) {
      // 从堆栈中获取顶部块的指针
      Block* block = blocks.top();
      // 弹出堆栈顶部块
      blocks.pop();
      // 遍历当前块中的每个节点
      for (auto n : block->nodes()) {
        // 遍历当前节点的子块列表
        for (Block* sub_block : n->blocks()) {
          // 将子块压入堆栈以便后续处理
          blocks.push(sub_block);
        }
        // 如果当前节点的类型为 prim::fork
        if (n->kind() == prim::fork) {
          // 获取节点的子图，并对其进行优化处理
          auto subgraph = n->g(attr::Subgraph);
          optimizeSubGraphs(subgraph, func);
        }
      }
    }
  }

  // 定义一个函数 run，用于执行内联和优化操作
  void run() {
    // 定义一个 lambda 函数 applyInline，对子图进行内联优化
    auto applyInline = [](std::shared_ptr<Graph>& subgraph) {
      Inline(*subgraph);
      ClearProfilingInformation(subgraph);
    };
    // 定义一个 lambda 函数 applyOptimizations，对子图应用优化操作
    auto applyOptimizations = [](std::shared_ptr<Graph>& subgraph) {
#ifndef C10_MOBILE
      // 如果不是移动设备平台，则对子图进行自动类型转换
      Autocast(subgraph);
#endif
      // 运行优化操作，包括不展开非常量循环和不常量传播用户类
      runOptimization(
          subgraph,
          /* unroll_non_constant_loops? */ false,
          /* const_prop_user_classes? */ false);
      // 消除无操作节点
      EliminateNoOps(subgraph);
      // 将简单元组降级
      LowerSimpleTuples(subgraph);
    };

    // 存储接口需要重新分配类型的映射关系
    std::unordered_map<std::string, std::unordered_set<std::string>>
        interfacesToReassignType;

    // 遍历保留的方法集合，分析每个函数的图形表示
    for (auto function : preservedMethods_) {
      GRAPH_DEBUG("Analyzing function: " + function->name());
      auto graph = toGraphFunction(*function).graph();
      // 优化图形的子图
      optimizeSubGraphs(graph, applyInline);
      // 如果需要冻结接口，则内联接口调用并更新类型映射
      if (freezeInterfaces_) {
        inlineInterfaceCalls(graph, interfacesToReassignType);
      }
    }

    // 重新分配接口类型
    reassignInterfaceTypes(interfacesToReassignType);

    // 遍历保留的方法集合，记录每个函数图形的可变属性
    for (auto function : preservedMethods_) {
      GRAPH_DEBUG("Recording mutable attrs for function: " + function->name());
      auto graph = toGraphFunction(*function).graph();
      // 记录模块中显式设置的可变属性，这些属性无法被折叠
      recordMutableAttrs(graph);
    }

    // 遍历保留的方法集合，传播每个函数图形的属性
    for (auto function : preservedMethods_) {
      GRAPH_DEBUG("Propagating function: " + function->name());
      auto graph = toGraphFunction(*function).graph();
      // 传播属性到图形中
      propagateAttributes(graph);
      // 对图形中的子图进行优化操作
      optimizeSubGraphs(graph, applyOptimizations);
    }
    // 清理冻结的模块状态
    GRAPH_DEBUG("Cleaning up module");
    cleanupFrozenModule();
  }

 private:
  using ResolvedName = std::pair<Module, std::string>;

  // 尝试解析限定名称（例如 submodule1.submodule2.foo）。
  // 如果限定名称在根模块中存在，则返回未限定的属性/函数名称和父模块；否则返回 nullopt。
  // 例如：
  // submodule1.submodule2.foo -> {submodule2, "foo"}
  // submodule1.non_existent_module.foo -> nullopt
  std::optional<ResolvedName> resolveName(const std::string& name) {
    auto sub_names = splitName(name);
    if (sub_names.empty()) {
      return c10::nullopt;
    }
    auto& attr_name = sub_names.back();
    auto cur_module = module_;
    std::vector<ResolvedName> attr_infos;
    attr_infos.reserve(sub_names.size() - 1);

    // 遍历限定名称的每个部分，尝试从当前模块中查找子模块
    for (size_t i = 0; i < sub_names.size() - 1; ++i) {
      bool found = false;
      const auto& sub_name = sub_names[i];
      // 遍历当前模块的命名子模块
      for (const auto& child_module : cur_module.named_children()) {
        if (child_module.name == sub_name) {
          attr_infos.emplace_back(cur_module._ivalue(), child_module.name);
          cur_module = child_module.value;
          found = true;
          break;
        }
      }
      // 如果未找到匹配的子模块，则返回空
      if (!found) {
        return c10::nullopt;
      }
    }
    // 返回解析后的模块和属性名称
    // 检查当前模块是否具有指定属性或是否能找到指定方法
    if (cur_module.hasattr(attr_name) || cur_module.find_method(attr_name)) {
      // 我们暂时不想将这些模块标记为可变的；这可能会干扰内联过程。
      // 相反，我们会记录用户希望保留它们的事实。
      // 这些模块将在清理准备阶段（recordReferenceAttrs）进行处理。
      for (auto& attr_info : attr_infos) {
        // 获取属性信息中的父模块和子属性名
        const auto& parent_module = attr_info.first;
        auto& sub_name = attr_info.second;
        // 将子属性名添加到对应父模块的用户保留属性集合中
        userPreservedAttrs_[parent_module._ivalue()].insert(
            std::move(sub_name));
      }
      // 返回当前模块和属性名的移动对
      return std::make_pair(std::move(cur_module), std::move(attr_name));
    }

    // 如果条件不满足，则返回空的 optional 对象
    return c10::nullopt;
  }

  // 加载模块路径，并将路径中的属性名按顺序存储在 names_ 中
  bool _loadModulePath(Value* input, std::shared_ptr<Graph>& graph) {
    // 获取输入值对应的节点
    Node* node = input->node();
    // 清空 names_ 容器
    names_.clear();
    // 循环直到找到输入节点的类型与图的第一个输入节点类型相同的输出节点
    while (!(node->outputs()[0]->type() == graph->inputs()[0]->type())) {
      // 如果当前节点是 prim::GetAttr 类型
      if (node->kind() == prim::GetAttr) {
        // 将属性名添加到 names_ 容器的前端
        names_.push_front(node->s(attr::name));
        // 获取当前节点输入的节点，并继续循环
        node = node->inputs()[0]->node();
      } else {
        // 如果当前节点不是 prim::GetAttr 类型，则返回失败
        return false;
      }
    }

    // 成功加载路径，返回 true
    return true;
  }

  // 获取模块路径并返回其属性名序列，如果加载路径失败则返回空的 optional 对象
  std::optional<std::deque<std::string>> getModulePath(
      Value* input,
      std::shared_ptr<Graph>& graph) {
    // 调用 _loadModulePath 方法加载路径，并获取成功与否的状态
    bool success = _loadModulePath(input, graph);
    // 如果加载失败，则返回空的 optional 对象
    if (!success) {
      return c10::nullopt;
    }
    // 返回路径中的属性名序列
    return names_;
  }

  // 根据模块路径从 attrModule 中获取模块，并检查是否需要保留
  template <typename Iter>
  bool getModuleFromPath(
      Module& attrModule,
      const Iter& begin,
      const Iter& end) {
    // 遍历路径中的每一个模块名
    for (Iter it = begin; it != end; ++it) {
      // 获取当前迭代器指向的模块名
      const std::string& moduleName = *it;
      // 如果需要保留的属性集合中包含当前模块名对应的属性
      if (preservedAttrs_.count(attrModule.attr(moduleName))) {
        // 返回失败
        return false;
      }
      // 将当前模块名对应的属性更新为 attrModule，并继续下一个迭代
      attrModule = attrModule.attr(moduleName).toModule();
    }
    // 成功获取模块路径中的所有模块，返回 true
    return true;
  }
  // 返回 true 表示找到常量属性，否则返回 false
  bool findConstantAttr(
      Value* input,
      std::string& name,
      Module& attrModule,
      std::shared_ptr<Graph>& graph) {
    // 检查输入值的类型是否是接口类型或者期望的模块类型，如果不是则返回 false
    if (!input->type()->cast<InterfaceType>() &&
        !input->type()->expectRef<ClassType>().is_module()) {
      return false;
    }

    // 将输入值的模块路径加载到 this->names_
    if (!_loadModulePath(input, graph)) {
      return false;
    }

    // 根据 names_ 中的路径重新确定 attrModule
    if (!getModuleFromPath(attrModule, names_.begin(), names_.end())) {
      return false;
    }

    // 获取指定名称的属性
    auto attr = attrModule.attr(name);

    // 检查属性是否属于不可变类型，若是则查找是否在保留的标量属性集合中
    if (!AliasDb::isMutableType(attr.type())) {
      auto it = preservedScalarAttrs_.find(attrModule._ivalue());
      return it == preservedScalarAttrs_.end() || !it->second.count(name);
    }

    // 如果属性属于可变类型，则检查是否在保留的属性集合中
    if (preservedAttrs_.count(attr)) {
      return false;
    }

    // 如果属性类型不是类类型，则检查与保留属性的重叠情况
    if (!attr.type()->cast<ClassType>()) {
      for (auto& ivalue : preservedAttrs_) {
        if (!ivalue.isObject() && ivalue.overlaps(attr)) {
          return false;
        }
      }
    }

    // 如果以上条件都通过，则返回 true
    return true;
  }

  // 插入可变属性的名称和属性值到相应的保留集合中
  void insertMutableAttr(
      const std::string& name,
      const IValue& attr,
      const ModulePtr& attrModule) {
    // 如果属性是可变类型，则插入到保留属性集合中
    if (AliasDb::isMutableType(attr.type())) {
      preservedAttrs_.insert(attr);
    } else {
      // 否则，将属性名称插入到保留的标量属性集合中
      preservedScalarAttrs_[attrModule].insert(name);
    }
  }

  // 记录图中所有块的可变属性
  void recordMutableAttrs(std::shared_ptr<Graph>& graph) {
    // 使用图中的根块初始化堆栈
    std::stack<Block*> blocks({graph->block()});

    // 创建图的别名分析数据库，并设置为冻结状态
    std::unique_ptr<AliasDb> aliasDb =
        std::make_unique<AliasDb>(graph, /* isFrozen */ true);
    // 只要 blocks 不为空就循环执行
    while (!blocks.empty()) {
      // 从栈顶取出一个 Block 对象
      Block* block = blocks.top();
      // 弹出栈顶元素
      blocks.pop();
      // 遍历当前 Block 中的所有节点
      for (auto n : block->nodes()) {
        // 遍历当前节点 n 的所有子 Block，将子 Block 压入栈中
        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }

        // 检查节点 n 的类型是否为 prim::ModuleContainerIndex，如果是，则抛出错误信息
        TORCH_CHECK(
            n->kind() != prim::ModuleContainerIndex,
            "Freezing modules containing prim::ModuleContainerIndex is not supported");

        // 如果节点 n 的类型是 prim::SetAttr 或 prim::GetAttr
        if (n->kind() == prim::SetAttr || n->kind() == prim::GetAttr) {
          // 默认情况下，如果存在接口类型的属性，则冻结失败
          // 如果 freezeInterfaces_ 开启，则接口会被类似于其他属性一样折叠
          TORCH_CHECK(
              freezeInterfaces_ ||
                  !(n->kind() == prim::GetAttr &&
                    n->output()->type()->cast<InterfaceType>()),
              "attempted to freeze a module that uses interface attributes");
          // 获取属性的名称
          auto name = n->s(attr::name);
          // 获取当前模块的引用
          auto attrModule = module_;
          // 如果找不到常量属性，则继续下一个节点的处理
          if (!findConstantAttr(n->inputs()[0], name, attrModule, graph)) {
            continue;
          }

          // 获取属性值
          auto attr = attrModule.attr(name);
          // 如果节点类型为 prim::GetAttr
          if (n->kind() == prim::GetAttr) {
            auto type = n->output()->type();
            // 如果属性是对象或者不可变类型，则跳过记录子模块
            if (attr.isObject() || !AliasDb::isMutableType(attr.type())) {
              continue;
            }
            // 将使用过的属性插入 usedAttrs_ 中
            usedAttrs_.insert(attr);
          }

          // 如果节点类型为 prim::SetAttr 或具有输出写入器
          if (n->kind() == prim::SetAttr || aliasDb->hasOutputWriters(n)) {
            // 如果是 prim::GetAttr 类型，则在 DEBUG 模式下记录相关信息
            GRAPH_DEBUG(
                n->kind() == prim::GetAttr ? "attribute: " + name + " in %" +
                        n->output()->debugName() + " has inplace writer"
                                           : "attribute: " + name + " is set");
            // 获取属性模块的指针
            auto mptr = attrModule._ivalue();
            // 插入可变属性
            insertMutableAttr(name, attr, mptr);
          }
        } else if (n->kind() == prim::fork) {
          // 如果节点类型为 prim::fork，则对分支子图应用记录可变属性的函数
          applyToForkSubgraph(
              n,
              graph,
              // NOLINTNEXTLINE(modernize-avoid-bind)
              std::bind(
                  &AttributePropagator::recordMutableAttrs,
                  *this,
                  std::placeholders::_1));
        }
      }
    }
    // FIXME: 当前别名分析无法跟踪子值。
    // 这不是常见情况，对于冻结，检测并报错。
    // 初始化已看到的 IValue::HashAliasedIValues 集合
    IValue::HashAliasedIValues seen;
    // 遍历已使用的属性集合
    for (auto& val : usedAttrs_) {
      // 获取属性值的子值集合
      IValue::HashAliasedIValues subValues;
      val.getSubValues(subValues);
      // 检查所有子值是否与已看到的值不重叠，否则抛出错误
      TORCH_CHECK(
          std::all_of(
              subValues.begin(),
              subValues.end(),
              [&seen](const IValue& v) { return seen.count(v) == 0; }),
          "module contains attributes values that overlaps ",
          val);
      // 将当前属性的所有子值插入 seen 集合中
      seen.insert(subValues.begin(), subValues.end());
  }
}

// 重写梯度的方法，根据输入类型不同进行不同的处理
IValue overrideGradient(IValue attr) {
  // 如果是张量类型
  if (attr.isTensor()) {
    auto& t = attr.toTensor();
    // 如果张量需要梯度
    if (t.requires_grad()) {
      // 分离张量，并关闭其梯度计算
      auto detached = t.detach();
      detached.set_requires_grad(false);
      // 更新输入参数为不需要梯度的张量
      attr = IValue(std::move(detached));
    }
  } else if (attr.isTuple()) {
    // 如果是元组类型
    auto tuple = std::move(attr).toTuple();
    const auto& elems = tuple->elements();
    // 遍历元组的元素，递归调用重写梯度方法
    for (const auto idx : c10::irange(elems.size())) {
      tuple->unsafeSetElement(idx, overrideGradient(elems[idx]));
    }
    attr = std::move(tuple);
  } else if (attr.isList()) {
    // 如果是列表类型
    c10::List<IValue> elems = std::move(attr).toList();
    // 遍历列表的元素，递归调用重写梯度方法
    for (const auto i : c10::irange(elems.size())) {
      elems.set(i, overrideGradient(elems.extract(i)));
    }
    attr = elems;
  } else if (attr.isGenericDict()) {
    // 如果是字典类型
    auto dict = std::move(attr).toGenericDict();
    // 遍历字典的值，递归调用重写梯度方法
    for (const auto& pair : dict) {
      auto val = pair.value();
      val = overrideGradient(std::move(val));
    }
    attr = dict;
  } else if (attr.isObject() && !attr.toObjectRef().type()->is_module()) {
    // 如果是对象类型并且不是模块类型
    auto obj_type = attr.type()->expect<ClassType>();
    auto obj_value = std::move(attr).toObject();
    auto sub_attributes = obj_type->getAttributes();
    // 遍历对象的属性，递归调用重写梯度方法
    for (const auto& sub_attr : sub_attributes) {
      auto sub_attr_val = obj_value->getAttr(sub_attr.getName());
      sub_attr_val = overrideGradient(std::move(sub_attr_val));
    }
    return obj_value;
  }

  // 返回处理后的属性值
  return attr;
}

// 仅当 'freezeInterfaces' 参数为打开状态时调用该方法。
// 检索与接口相关联的模块，并内联调用的方法。
bool inlineInterfaceCall(Node* n, const IValue& attr) {
  auto class_type = attr.type()->expect<ClassType>();
  bool inlined = false;
  // 遍历节点的输出用途
  for (auto use : n->output()->uses()) {
    auto user_node = use.user;
    if (user_node->kind() == prim::CallMethod) {
      // 获取调用方法的名称
      const std::string& methodName = user_node->s(attr::name);
      // 获取方法的函数对象
      Function& function = class_type->getMethod(methodName);
      // 尝试将方法转换为图形函数
      if (auto graphFunction = tryToGraphFunction(function)) {
        // 执行内联调用
        GRAPH_UPDATE(
            "Inlining interface method '",
            function.name(),
            "' to ",
            *user_node);

        GRAPH_UPDATE("Function body: ", graphFunction->optimized_graph());
        inlineCallTo(user_node, graphFunction);
        inlined = true;
      }
    }
  }
  // 返回内联是否成功的标志
  return inlined;
}
    // 返回变量 inlined
    return inlined;
    }
    
    //   [Note: Inlining interfaces strategy]
    // There's two structures that are relevant to freezing:
    // - the graph describing the computation in a method
    // - the module describing the data structure of the module instance.
    //
    // 首先，在 inlineInterfaceCalls 中，我们进行接口的内联。这是与普通内联不同的单独步骤，因为在接口类型的 CallMethod 中需要比普通 CallMethod 多一些步骤。
    //
    // 接下来，我们需要简化模块数据结构的结构，这在 cleanupFrozenModule 中的大部分步骤中完成。
    //
    // 但是，由于在方法内部，可以将接口的值更改为实现该接口的另一个模块，这会带来复杂性。
    //
    // 例如：
    //
    // impl: MyInterface
    // ...
    // def forward(self, x):
    //     if x > 0:
    //         self.impl = my_interface_impl
    //
    // 这在冻结中是不允许的，因为在这种情况下，我们无法展平模块结构，因为 self.impl 的类型将会改变。
    //
    // 为了处理这种情况，我们执行以下操作：
    //   1. inlineInterfaceCalls:
    //     a. 内联图，并在此过程中记录所有接口
    //     b. 同时，检查（抛出错误）不允许的 SetAttr 调用。
    //   2. 调用 reassignInterfaceTypes，将接口类型重新分配给它们的具体类型。这是一个单独的步骤，以避免干扰 inlineInterfaceCalls（注意：可能不需要作为单独的步骤完成）
    //   3. 最终的 cleanupFrozenModule 将重新排序模块数据结构，并期望所有接口类型都已被移除。
    void inlineInterfaceCalls(
        std::shared_ptr<Graph>& graph,
        std::unordered_map<std::string, std::unordered_set<std::string>>&
            interfacesToRetype) {
      auto block = graph->block();
      std::stack<Block*> blocks({block});
    // 当块栈不为空时，循环处理块栈中的块对象
    while (!blocks.empty()) {
      // 从块栈顶取出一个块对象
      Block* block = blocks.top();
      blocks.pop();
      // 遍历当前块中的所有节点
      for (auto n : block->nodes()) {
        // 遍历当前节点的所有子块，并将其推入块栈
        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }
        // 如果当前节点是 prim::GetAttr 类型
        if (n->kind() == prim::GetAttr) {
          // 如果当前节点的输出类型不是 InterfaceType，则跳过处理
          if (!n->output()->type()->cast<InterfaceType>()) {
            continue;
          }
          // 获取属性名
          auto name = n->s(attr::name);
          // 复制模块对象
          auto attrModule = module_;
          // 获取输入节点
          auto input = n->inputs()[0];
          // 检查是否找到常量属性
          TORCH_CHECK(
              findConstantAttr(input, name, attrModule, graph),
              "failed to freeze interface attribute '" + name + "'");
          // 内部断言，确保 attrModule 中存在指定的属性名
          TORCH_INTERNAL_ASSERT(attrModule.hasattr(name));
          // 获取属性值
          auto attr = attrModule.attr(name);
          // 内联接口调用
          inlineInterfaceCall(n, attr);
          // 重置 GetAttr 节点的输出类型为具体的模块类型
          n->output()->setType(attr.type());

          // 记录此操作，以便稍后在 reassignInterfaceTypes() 中重新分配类型
          // 参见 [Note: Inlining interfaces strategy]
          auto path = getModulePath(input, graph);
          TORCH_INTERNAL_ASSERT(path.has_value());
          auto path_str = concatName(path->begin(), path->end());
          // 将接口类型标记为需要重新分配类型
          interfacesToRetype[path_str].insert(name);
        } else if (n->kind() == prim::SetAttr) {
          // 检查是否正在对接口类型的参数进行赋值
          // 参见 [Note: Inlining interfaces strategy]
          auto name = n->s(attr::name);
          auto attrModule = module_;
          auto input = n->inputs()[0];

          // 如果输入节点的类型不是接口类型并且不是模块类型的引用，跳过处理
          if (!input->type()->cast<InterfaceType>() &&
              !input->type()->expectRef<ClassType>().is_module()) {
            continue;
          }

          // 获取输入节点所在的模块路径
          auto path = getModulePath(input, graph);
          TORCH_INTERNAL_ASSERT(path.has_value());
          // 更新 attrModule，使其成为与输入节点匹配的模块
          getModuleFromPath(attrModule, path->begin(), path->end());

          // 获取属性类型，并断言不是接口类型，因为冻结操作不支持在接口类型上进行 SetAttr
          const auto& attrType = attrModule.type()->getAttribute(name);
          TORCH_INTERNAL_ASSERT(
              !attrType->cast<InterfaceType>(),
              "Freezing does not support SetAttr on an interface type. ",
              "SetAttr is attempted on '",
              name,
              "'");
        } else if (n->kind() == prim::fork) {
          // 应用于分支子图，内联接口调用
          applyToForkSubgraph(
              n,
              graph,
              // NOLINTNEXTLINE(modernize-avoid-bind)
              std::bind(
                  &AttributePropagator::inlineInterfaceCalls,
                  *this,
                  std::placeholders::_1,
                  interfacesToRetype));
        }
      }
    }
  }
}

// See [Note: Inlining interfaces strategy]
// This modifies the internal structure of module types to reassign the
// type from an interface type to its concrete type.
// 重新分配接口类型的方法，用于修改模块类型的内部结构，将接口类型重新赋值为其具体类型。
void reassignInterfaceTypes(
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        interfacesToRetype) {
  // 遍历接口类型重新赋值的映射
  for (const auto& it : interfacesToRetype) {
    const std::string& modulePath = it.first;
    // 拆分模块路径
    const std::vector<std::string>& splitPath = splitName(modulePath);
    Module attrModule = module_;
    // 根据拆分的路径获取模块
    getModuleFromPath(attrModule, splitPath.begin(), splitPath.end());

    // 遍历需要重新赋值的接口类型集合
    for (const std::string& name : it.second) {
      auto subvalue = attrModule.attr(name);
      auto subvalueType = subvalue.type();
      // 不安全地修改属性的类型
      attrModule.type()->unsafeChangeAttributeType(name, subvalueType);
    }
  }
}

void propagateAttributes(std::shared_ptr<Graph>& graph) {
  std::unordered_map<ModulePtr, std::unordered_map<std::string, Value*>>
      attrValues;
  auto isEval = !module_.hasattr("training") || !module_.is_training();
  GRAPH_DEBUG("Freezing Module: ", module_.type()->name()->name());
  auto block = graph->block();
  std::stack<Block*> blocks({block});

  // 设置插入点为节点的开始位置
  Node* m = *block->nodes().begin();
  WithInsertPoint guard(m);
}

void applyToForkSubgraph(
    Node* n,
    std::shared_ptr<Graph>& graph,
    const std::function<void(std::shared_ptr<Graph>&)>& func) {
  TORCH_CHECK(n->kind() == prim::fork);
  auto attrModule = module_;
  auto node = n->inputs()[0]->node();

  // 检查fork的第一个参数是否为模块。这个模块用作基础模块（类似于前向传播中的'self'）来解析GetAttrs。
  // 否则，使用module_应用冻结操作。
  if (node->kind() == prim::GetAttr &&
      node->output()->type()->cast<ClassType>()) {
    auto name = node->s(attr::name);
    auto input = node->inputs()[0];
    if (!findConstantAttr(input, name, attrModule, graph)) {
      // 需要保留模块。
      return;
    }
    attrModule = attrModule.attr(name).toModule();
    std::swap(module_, attrModule);
  }

  auto subgraph = n->g(attr::Subgraph);
  func(subgraph);
  module_ = attrModule;
}

bool moduleEscapes(Module& subModule, std::shared_ptr<Graph>& graph) {
  // 检查子模块是否逃逸
  for (auto& output : graph->outputs()) {
    if (subModule.type()->isSubtypeOf(*output->type())) {
      return true;
    }
  }
  return preservedSubModule_.count(subModule._ivalue());
}

void removeExtraWaitCalls(Block* b) {
  auto nodes = b->nodes();
    // 对 nodes 中的每个节点进行迭代处理
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      // 获取当前节点指针
      auto node = *it;
      // 如果当前节点的类型不是 aten::wait，则跳过处理
      if (node->kind() != aten::wait) {
        continue;
      }
      // 断言当前节点的输入数量为 1
      TORCH_INTERNAL_ASSERT(node->inputs().size() == 1);
      // 断言当前节点的输出数量为 1
      TORCH_INTERNAL_ASSERT(node->outputs().size() == 1);
      // 如果当前节点的输入类型不是 FutureType，则可以删除 aten::wait 操作符
      if (node->input()->type()->kind() != TypeKind::FutureType) {
        // 将当前节点的输出替换为输入
        node->output()->replaceAllUsesWith(node->input());
        // 销毁当前迭代器指向的节点
        it.destroyCurrent();
      }
    }
    // 对剩余的节点进行递归处理
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      // 获取当前节点指针
      auto node = *it;
      // 对当前节点的每个子块进行递归删除额外的等待调用
      for (auto sub_b : node->blocks()) {
        removeExtraWaitCalls(sub_b);
      }
    }
  }

  // cleanupFrozenModule 函数用于清理冻结模块。执行以下操作：
  // 1) 删除未使用的属性。
  // 2) 删除未引用的子模块。
  // 3) 删除非公共未引用方法。
  void cleanupFrozenModule() {
    // 遍历 preservedMethods_ 中的每个函数指针
    for (auto function : preservedMethods_) {
      // 获取函数对应的图形
      auto graph = toGraphFunction(*function).graph();
      // 记录图中引用的属性
      recordReferencedAttrs(graph);
      // 处理共享的类类型
      handleSharedClassType(module_, graph);
      // 删除图中多余的等待调用
      removeExtraWaitCalls(graph->block());
      // 清除优化后的图形数据
      toGraphFunction(*function).clear_optimized_graphs();
    }
    // 删除未使用的属性
    removeUnusedAttrs();
  }

  // 准备清理阶段。在此阶段记录所有包含可变属性的子模块。
  void recordReferencedAttrs(std::shared_ptr<Graph>& graph) {
    // 使用堆栈记录图中的块
    std::stack<Block*> blocks({graph->block()});
    // 使用集合记录模块的实例
    std::set<ModulePtr> modules({module_._ivalue()});
    while (!blocks.empty()) {
      // 从堆栈中取出顶部的块，并移除
      Block* block = blocks.top();
      blocks.pop();
      // 遍历当前块中的每个节点
      for (auto n : block->nodes()) {
        // 对于节点 n 中的每个子块，将其推入堆栈中以便后续处理
        for (Block* subBlock : n->blocks()) {
          blocks.push(subBlock);
        }
        // 如果节点 n 是 prim::GetAttr 类型
        if (n->kind() == prim::GetAttr) {
          // 获取属性名
          auto& name = n->s(attr::name);
          // 遍历所有模块，寻找类型与 n 的输入节点类型相同的模块
          for (auto& mptr : modules) {
            auto module = Module(mptr);
            // 如果模块的类型与 n 的输入节点类型相同
            if (module.type() == n->inputs()[0]->type()) {
              // 断言模块包含指定的属性名
              TORCH_INTERNAL_ASSERT(module.hasattr(name));
              auto module = Module(mptr);  // 重新获取模块
              auto attr = module.attr(name);  // 获取属性值
              // 插入可变属性，将属性值与模块指针关联起来
              insertMutableAttr(name, attr, mptr);
              // 如果属性值是模块类型，将其插入模块集合中
              if (attr.isModule()) {
                modules.insert(attr.toModule()._ivalue());
              }
            }
          }
        } else if (n->kind() == prim::fork) {
          // 对于 prim::fork 类型节点，将其应用于分支图中
          applyToForkSubgraph(
              n,
              graph,
              // NOLINTNEXTLINE(modernize-avoid-bind)
              std::bind(
                  &AttributePropagator::recordReferencedAttrs,
                  *this,
                  std::placeholders::_1));
        }
      }
    }
    // 需要单独处理用户想要保留的属性，
    // 因为可能用户保留的模块在图中从未被引用
    for (const auto& attr_info : userPreservedAttrs_) {
      const auto& parent_module = attr_info.first;
      for (const auto& attr_name : attr_info.second) {
        // 获取属性值
        const auto value = parent_module->getAttr(attr_name);
        // 插入可变属性，将属性值与父模块关联起来
        insertMutableAttr(attr_name, value, parent_module);
      }
    }
  }

  // 该函数递归地遍历子模块，识别每个类类型需要保留的属性槽位
  //
  // 注意 'attrsToKeep[type].insert(type->numAttributes())' 意味着需要保留
  // 类型 'type' 及其方法的所有属性槽位。当子模块逃逸时（即被返回时），模块被保留。
  void handleSharedClassType(Module& module, std::shared_ptr<Graph>& graph) {
    auto type = module.type();  // 获取模块的类型
    size_t N = type->numAttributes();  // 获取模块类型的属性数目
    // 如果模块逃逸，则保留其所有属性和方法
    if (moduleEscapes(module, graph)) {
      attrsToKeep_[type].insert(N);
      return;
    }
    auto it2 = preservedScalarAttrs_.find(module._ivalue());
    // 将模块的值插入到类型对应的共享类型子模块中
    SharedTypeSubModules_[type].insert(module._ivalue());
    // 初始化要保留的属性集合为空集合
    attrsToKeep_[type].insert({});
    // 遍历范围为0到N的索引
    for (const auto i : c10::irange(N)) {
      // 获取属性名
      auto name = type->getAttributeName(i);
      // 获取模块中的属性值
      auto attr = module.attr(name);
      // 获取属性的类型
      auto attrTy = attr.type();

      // 忽略变量未初始化的警告
      // 判断属性是否可变
      bool isMutable;
      if (AliasDb::isMutableType(attrTy)) {
        isMutable = preservedAttrs_.count(attr);
      } else {
        // 检查是否在保留的标量属性中
        isMutable =
            it2 != preservedScalarAttrs_.end() && it2->second.count(name);
      }
      // 如果属性可变
      if (isMutable) {
        // 将属性索引插入要保留的属性集合
        attrsToKeep_[type].insert(i);
        // 如果属性是模块类型
        if (attr.isModule()) {
          // 查看 [注释: 内联接口策略]
          TORCH_CHECK(
              !type->getAttribute(i)->cast<InterfaceType>(),
              "Unexpected interface attribute '" + name + "' during freezing");

          // 将属性转换为模块，并处理共享类类型
          auto attrModule = attr.toModule();
          handleSharedClassType(attrModule, graph);
        }
      }
    }
  }

  // 移除冻结模块的每个子模块的未使用属性和方法
  // 此函数遍历其子模块属性的ClassType，包括其自身类型
  void removeUnusedAttrs() {
    // 初始化要移除的属性和方法的列表
    std::vector<std::string> attrsToRemove;
    std::vector<Function*> funcsToRemove;
    // 遍历要保留的属性集合
    for (auto& it : attrsToKeep_) {
      // 获取类型和其属性数量
      auto& type = it.first;
      size_t N = type->numAttributes();
      // 如果要保留的属性集合包含N，则跳过
      if (it.second.count(N)) {
        continue;
      }
      // 遍历类型的所有属性
      for (const auto i : c10::irange(N)) {
        // 如果要保留的属性集合不包含索引i
        if (it.second.count(i) == 0) {
          // 将属性名加入要移除的列表
          attrsToRemove.push_back(type->getAttributeName(i));
        }
      }
      // 遍历类型的所有方法
      for (auto& fn : type->methods()) {
        // 如果方法不在保留的方法集合中，则加入要移除的方法列表
        if (preservedMethods_.count(fn)) {
          continue;
        }
        funcsToRemove.push_back(fn);
      }

      // 移除要移除的属性
      for (auto& name : attrsToRemove) {
        for (auto& val : SharedTypeSubModules_[type]) {
          auto mod = val.toModule();
          mod._ivalue()->unsafeRemoveAttr(name);
        }
        type->unsafeRemoveAttribute(name);
      }
      // 移除要移除的方法
      for (auto fn : funcsToRemove) {
        type->unsafeRemoveMethod(fn->name());
        auto mod = SharedTypeSubModules_[type].begin()->toModule();
        mod._ivalue()->compilation_unit()->unsafeRemoveMethod(fn->qualname());
      }

      // 清空要移除的属性和方法列表
      attrsToRemove.clear();
      funcsToRemove.clear();
  }

  // 包含无法折叠或用户指示保留的属性。
  IValue::HashAliasedIValues preservedAttrs_;

  // 被跟踪的不可变类型（标量）按其属性名称而不是 IValues。
  std::unordered_map<ModulePtr, std::unordered_set<std::string>>
      preservedScalarAttrs_;

  // 包含用户指定要在冻结模块中保留的方法。
  std::unordered_set<Function*> preservedMethods_;

  // 包含用户指定要保留在冻结模块中的子模块。
  std::unordered_set<ModulePtr> preservedSubModule_;

  // 跟踪所有使用过的属性 IValues，可以别名。
  IValue::HashAliasedIValues usedAttrs_;

  // 包含每个 ClassType 需要保留的属性槽位。
  std::unordered_map<ClassTypePtr, std::unordered_set<size_t>> attrsToKeep_;

  // 包含共享同一 ClassType 的子模块。
  std::unordered_map<ClassTypePtr, IValue::HashAliasedIValues>
      SharedTypeSubModules_;

  // 模块的引用。
  Module& module_;

  // 允许冻结包含接口的模块。
  bool freezeInterfaces_;

  // 保留模块参数。
  bool preserveParameters_;

  // 包含属性名称的队列（例如 {"self", "subModule", "a"}）。
  std::deque<std::string> names_;

  // 参见 [Constant Object Weak CompilationUnit Reference]。
  std::unordered_map<
      c10::intrusive_ptr<at::ivalue::Object>,
      c10::intrusive_ptr<at::ivalue::Object>>
      object_memo_;

  // 包含用户希望与其拥有的模块一起保留的属性名称。
  std::unordered_map<ModulePtr, std::unordered_set<std::string>>
      userPreservedAttrs_;
void checkModuleDoesNotReturnSelf(const Module& module) {
  // 检查模块是否包含 "forward" 方法
  if (module.find_method("forward")) {
    // 获取模块的 "forward" 方法
    Method method = module.get_method("forward");
    // 遍历方法的计算图的输出
    for (auto& output : method.graph()->outputs()) {
      // 检查输出类型是否与模块类型相同，防止模块返回自身
      TORCH_CHECK(
          output->type() != module.type(),
          "attempted to freeze a module that return itself");
    }
  }
}

Module freeze_module(
    const Module& module,
    std::vector<std::string> preservedAttrs,
    bool freezeInterfaces,
    bool preserveParameters) {
  // 检查模块不会返回自身
  checkModuleDoesNotReturnSelf(module);

  // 克隆输入模块，并设置克隆的模块为可修改
  auto moduleClone = module.clone(true);
  // 创建属性传播器对象，传入模块克隆、保留的属性列表、冻结接口标志和保留参数标志
  AttributePropagator attrPropagator(
      moduleClone, preservedAttrs, freezeInterfaces, preserveParameters);
  // 运行属性传播器
  attrPropagator.run();
  // 返回克隆的模块
  return moduleClone;
}

void freeze_module_inplace(
    Module* module,
    std::vector<std::string> preservedAttrs,
    bool freezeInterfaces,
    bool preserveParameters) {
  // 检查模块指针不为空
  TORCH_CHECK(module != nullptr, "module cannot be nullptr");
  // 检查模块不会返回自身
  checkModuleDoesNotReturnSelf(*module);
  // 创建属性传播器对象，传入模块指针、保留的属性列表、冻结接口标志和保留参数标志
  AttributePropagator attrPropagator(
      *module, preservedAttrs, freezeInterfaces, preserveParameters);
  // 运行属性传播器
  attrPropagator.run();
}

} // namespace jit
} // namespace torch
```