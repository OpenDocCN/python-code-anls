# `.\pytorch\torch\csrc\jit\passes\fixup_trace_scope_blocks.cpp`

```
#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>

#include <algorithm>

namespace torch {
namespace jit {

namespace {

// 判断节点是否符合条件，条件是节点的类型是 prim::TracedModuleForward 或 prim::TracedFork
bool isEligibleNode(Node* n) {
  return n->kind() == prim::TracedModuleForward ||
      n->kind() == prim::TracedFork;
}

// 这个 pass 执行了几个操作：
// 1) 它检查 TracedModuleForward 节点，并解析该方法调用的 `self` 类型。
//    它将该类型作为输入添加到块中，并将对应的 TracedAttr 值作为节点输入添加。
//    这确保了 `self` 是节点上的显式使用，这是我们在下游处理中利用的属性。例如：
// 2) 将所有对 prim::TracedAttr 值的引用转换为 prim::GetAttr 调用，尽可能在最紧凑的范围内进行。
//    具体来说，对于每个 prim::TracedAttr 值的使用，我们比较该属性的作用域与使用的作用域。
//    对于两者之间不共享的所有原子，我们会发出 GetAttr 节点。例如，如果属性 `f.param` 在作用域 `f` 中被引用，
//    我们在 `f` 块中发出 GetAttr[name="param"](%self) 节点，其中 `self` 是先前添加到块中的 `self` 参数。
// 3) 摧毁所有的 prim::TracedAttr 节点，因为它们不应再有使用。

// 快速示例：
//
// 输入图：
//
//     graph(%self : ClassType<Module>,
//           %x : Float(3, 4)):
//       %1 : bool = prim::TracedAttr[scope="__module.training"]()
//       %2 : ClassType<Module> = prim::TracedAttr[scope="__module.f"]()
//       %3 : Float(4, 4) = prim::TracedAttr[scope="__module.f.param"]()
//       %4 : bool = prim::TracedAttr[scope="__module.f.training"]()
//       = prim::TracedModuleForward[scope="__module.f"](),
//         block0():
//           %6 : Float(3, 4) = aten::mm(%x, %3),
//           -> ()
//       return (%6)
//
// 第一步后的差异：
//
//     -   = prim::TracedModuleForward[scope="__module.f"](),
//     -    block0():
//     +   = prim::TracedModuleForward[scope="__module.f"](%2),
//     +    block0(%self : ClassType<Module>):
//
// 第二步后的差异：
//
//       graph(%self.1 : ClassType<Module>,
//             %x : Float(3, 4)):
//       +  %9 : ClassType<Module> = prim::GetAttr[name="f"](%self.1)
//         %1 : bool = prim::TracedAttr[scope="__module.training"]()
//           <....>
//         %4 : bool = prim::TracedAttr[scope="__module.f.training"]()
//       -   = prim::TracedModuleForward[scope="__module.f"](%2),
//       +   = prim::TracedModuleForward[scope="__module.f"](%9),
//           block0(%self : ClassType<Module>):
//       -      %6 : Float(3, 4) = aten::mm(%x, %3),
//       +      %8 : Tensor = prim::GetAttr[name="param"](%self)
//
//       +      %6 : Float(3, 4) = aten::mm(%x, %8),
//             -> ()
//         return (%6)
//
// The diff after step (3)
//
//       -  %1 : bool = prim::TracedAttr[scope="__module.training"]()
//       -  %2 : ClassType<Module> = prim::TracedAttr[scope="__module.f"]()
//       -  %3 : Float(4, 4) = prim::TracedAttr[scope="__module.f.param"]()
//       -  %4 : bool = prim::TracedAttr[scope="__module.f.training"]()
struct ConvertTracedAttrReferences {
  void run(const std::shared_ptr<Graph>& graph) {
    // 构建一个表格，将每个 TracedAttr 节点的属性全名映射到节点的输出值
    buildAttrMap(graph);
    // 步骤 1：向所有 TracedForward 节点添加 self 参数
    addSelfArgToTracedForwardNodes(graph->block());
    // 步骤 2：将属性引用转换为本地的 GetAttr 节点
    convertAttrReferencesToLocalGetAttrs(
        graph->block(), "__module", graph->inputs()[0]);
    // 步骤 3：销毁所有 TracedAttr 节点
    destroyTracedAttrNodes(graph);
  }

 private:
  // 构建属性映射表，用于追踪每个 TracedAttr 节点的全名和输出值
  void buildAttrMap(const std::shared_ptr<Graph>& graph) {
    for (Node* n : graph->nodes()) {
      if (n->kind() == prim::TracedAttr) {
        attr_qualname_to_value[n->s(attr::scope)] = n->output();
      }
    }
  }

  // 向所有 TracedForward 节点添加 self 参数
  void addSelfArgToTracedForwardNodes(Block* b) {
    for (Node* n : b->nodes()) {
      if (n->kind() == prim::TracedModuleForward) {
        // 向节点添加 TracedAttr 对应的值作为输入参数
        n->addInput(attr_qualname_to_value.at(n->s(attr::scope)));
        // 设置节点的 self 参数类型
        n->blocks()[0]->addInput("self")->setType(
            attr_qualname_to_value.at(n->s(attr::scope))->type());
        // 递归处理子块中的节点
        addSelfArgToTracedForwardNodes(n->blocks()[0]);
      }
      if (n->kind() == prim::TracedFork) {
        // 递归处理分支的子块中的节点
        addSelfArgToTracedForwardNodes(n->blocks()[0]);
      }
    }
  }

  // 递归函数，遍历图中的所有块，将属性引用转换为本地的 GetAttr 节点
  std::vector<Value*> convertAttrReferencesToLocalGetAttrs(
      Block* b,
      const c10::QualifiedName& prefix,
      Value* self) {
    // 存储未解决的 TracedAttr 引用
    std::vector<Value*> unresolved_tracedattrs;
    // 使用映射表避免在给定范围内重复发出 GetAttr 节点
    // 这里不依赖于公共子表达式消除 (CSE)，因为我们无法保证在 GetAttr 节点上的正确性
    // （我认为）

    // To ensure we don't emit redundant GetAttr Nodes in a given scope,
    // we maintain this map of original TracedAttr Value* to the Value*
    // corresponding to the GetAttr for that attribute.
    // We don't rely on CSE here because we currently can't reason about
    // the correctness of CSE over GetAttr Nodes (i think)
    std::unordered_map<Value*, Value*> attr_to_local_getattr;
    // TODO: Implement the rest of the function logic
``` 

这段代码片段说明了如何在代码中添加注释，以解释每一行代码的作用。
    // 用于存储本地重映射关系的哈希表，将 Value 指针映射到另一个 Value 指针
    std::unordered_map<Value*, Value*> local_remaps;

    // 遍历基本块 b 中的每个节点 n
    for (Node* n : b->nodes()) {
      // 如果节点的类型是 prim::TracedModuleForward
      if (n->kind() == prim::TracedModuleForward) {
        // 将节点 n 的第一个子块进行属性引用到本地获取属性的转换，获取未解析的值列表
        auto sub_unresolved = convertAttrReferencesToLocalGetAttrs(
            n->blocks()[0], n->s(attr::scope), n->blocks()[0]->inputs()[0]);
        // 将转换得到的未解析值添加到节点 n 的输入中
        for (Value* v : sub_unresolved) {
          n->addInput(v);
        }
      } else if (!n->blocks().empty()) {  // 如果节点有子块但不是 TracedModuleForward 类型
        // 遍历节点 n 的所有子块
        for (Block* sub_block : n->blocks()) {
          // 将子块中的属性引用转换为本地获取属性，并获取未解析的值列表
          auto sub_unresolved =
              convertAttrReferencesToLocalGetAttrs(sub_block, prefix, self);
          // 将转换得到的未解析值添加到节点 n 的输入中
          for (Value* v : sub_unresolved) {
            n->addInput(v);
          }
        }
      }

      // 遍历节点 n 的所有输入值
      for (size_t inp_idx = 0; inp_idx < n->inputs().size(); ++inp_idx) {
        Value* inp = n->input(inp_idx);

        // 短路处理：如果我们已经为这个属性生成了一个新的 Value，直接使用它
        if (local_remaps.count(inp)) {
          // 替换节点 n 的第 inp_idx 个输入为本地重映射表中对应的值
          n->replaceInput(inp_idx, local_remaps[inp]);
          continue;
        }

        // 设置插入点保护，在 b->param_node()->next() 后面插入新节点
        WithInsertPoint guard(b->param_node()->next());
        // 在节点 n 上替换跟踪属性的输入
        replaceTracedAttrInputOnNode(
            n, inp_idx, prefix, self, local_remaps, unresolved_tracedattrs);
      } // for (Value *inp : n->inputs())
    } // for (Node *n : b->nodes())
    // 返回未解析的跟踪属性值的列表
    return unresolved_tracedattrs;
  }

  // 在节点 n 上替换跟踪属性的输入
  void replaceTracedAttrInputOnNode(
      Node* n,
      size_t inp_idx,
      const c10::QualifiedName& prefix,
      Value* self,
      std::unordered_map<Value*, Value*>& local_remaps,
      std::vector<Value*>& unresolved_tracedattrs) {
    auto inp = n->inputs()[inp_idx];
    auto inp_node = inp->node();
    auto prefix_atoms = prefix.atoms();
    // 检查输入节点是否为 prim::TracedAttr 类型
    if (inp_node->kind() == prim::TracedAttr) {
      // 获取属性的限定名称
      auto attr_qualname = c10::QualifiedName(inp_node->s(attr::scope));
      // 如果限定名称以给定前缀开头
      if (prefix.isPrefixOf(attr_qualname)) {
        // 前缀匹配情况：属性位于当前作用域或子作用域内。持续发出 GetAttr 节点，直到找到正确的属性。
        auto attr_atoms = attr_qualname.atoms();
        // 替换值的起始为当前节点
        Value* replaced_value = self;
        // 遍历限定名称中的每个属性原子
        for (const auto i : c10::irange(attr_atoms.size())) {
          // 如果当前索引小于前缀原子的数量
          if (i < prefix_atoms.size()) {
            // 断言当前属性原子与前缀原子相同
            TORCH_INTERNAL_ASSERT(attr_atoms[i] == prefix_atoms[i]);
          } else {
            // 否则，在图中插入 GetAttr 节点，更新替换值
            replaced_value = n->owningBlock()->owningGraph()->insertGetAttr(
                replaced_value, attr_atoms[i]);
          } // if (i < prefix_atoms.size())
        } // for(const auto i : c10::irange(attr_atoms.size()))
        // 替换输入节点的索引为更新后的值
        n->replaceInput(inp_idx, replaced_value);
        // 将本地重映射更新为替换值
        local_remaps[inp] = replaced_value;
      } else {
        // 非前缀匹配情况：属性位于模块层次结构中较高的位置。为此属性在块中添加一个捕获输入，并添加到调用者处理的值向量中。
        // 创建一个具有相同元数据的新值作为替换
        Value* remapped = n->owningBlock()->addInput()->copyMetadata(inp);
        // 替换输入节点的索引为新值
        n->replaceInput(inp_idx, remapped);
        // 将未解决的 tracedattrs 添加到列表中
        unresolved_tracedattrs.push_back(inp);
        // 更新本地重映射为新值
        local_remaps[inp] = remapped;
      } // if (prefix.isPrefixOf(attr_qualname))
    } // if (inp_node->kind() == prim::TracedAttr)
  }

  // 前一遍应该已经删除了所有 TracedAttr 节点的使用。在这里显式删除它们。
  void destroyTracedAttrNodes(const std::shared_ptr<Graph>& graph) {
    // 遍历属性限定名称到值的映射
    for (auto& kv : attr_qualname_to_value) {
      // 销毁每个映射中的节点
      kv.second->node()->destroy();
    }
  }

  // 对于每个 prim::TracedAttr，记录 `scope` 值映射到图中该属性的值。
  std::unordered_map<std::string, Value*> attr_qualname_to_value;
};

// 遍历程序中的所有节点，并对每个使用的值进行处理：
// 如果被引用的值不在支配该节点的作用域内，
// 则将块和节点输出添加以提升它到支配该使用的作用域中。
struct MakeDefsDominateUses {
  MakeDefsDominateUses() = default;

  // 运行处理过程，从给定的块开始
  void run(Block* b) {
    // 处理块的参数节点
    processNode(b->param_node(), b);
    // 处理块中的每个节点
    for (Node* n : b->nodes()) {
      processNode(n, b);
    }
    // 处理块的返回节点
    processNode(b->return_node(), b);
  }

 private:
  // 处理节点中的输入值
  void processNode(Node* n, Block* b) {
    // 对节点的每个输入进行处理
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      Value* inp = n->inputs()[i];

      // 如果已经通过先前处理的使用提升到这个级别，则切换到重新映射的值
      Value* inp_remapped = inp;
      if (remap.count(inp_remapped)) {
        n->replaceInput(i, remap[inp_remapped]);
        inp_remapped = remap[inp_remapped];
      }

      // 虽然这个条件并非必需，但在通常情况下（使用局部值）可以节省大量计算
      if (inp_remapped->node()->owningBlock() != b) {
        // 查找这个节点和生成此输入的节点之间的公共祖先块。
        // 要使此输入使用有效，值的定义必须在这个公共祖先块中。
        Block* common_ancestor =
            n->findCommonAncestorBlockWith(inp_remapped->node());

        Value* v_itr = inp_remapped;
        Block* b_itr = inp_remapped->node()->owningBlock();

        // 从此输入的初始定义开始，迭代到更广泛的块，沿途添加块输出和节点输出。
        // 然后，在重新映射表中记录提升的值，以便后续的使用引用提升后的值，如果满足支配条件。
        while (b_itr != common_ancestor) {
          b_itr->registerOutput(v_itr);
          Value* remapped =
              b_itr->owningNode()->addOutput()->setType(v_itr->type());
          v_itr = remapped;
          b_itr = b_itr->owningNode()->owningBlock();
        }
        // 从现在开始，对 `inp` 的引用将被替换为对提升后的值 `v_itr` 的引用
        remap[inp] = v_itr;
        n->replaceInput(i, remap[inp]);
      }
    }

    // 如果节点符合条件，递归运行处理过程以处理其首个块
    if (isEligibleNode(n)) {
      run(n->blocks()[0]);
    }
  }

  // 该表维护了我们在使用中看到的值和提升值之间的映射关系。
  // 我们使用它来确保使用引用的是在支配作用域内的值。
  using RemappingTable = std::unordered_map<Value*, Value*>;
  RemappingTable remap;
};

// 对于除了图的块之外的所有块，将多个块返回转换为 TupleConstruct。
// 这是将块转换为方法所必需的。（并且在 self 为 nullptr 的情况下，
// 这也是正确内联块所必需的）。
void convertReturnsToTuples(Block* b) {
  for (Node* n : b->nodes()) {
    if (n->kind() == prim::TracedFork) {
      // 如果节点类型是 prim::TracedFork，则递归处理其第一个子块
      convertReturnsToTuples(n->blocks()[0]);
    } else if (n->kind() == prim::TracedModuleForward) {
      // 如果节点类型是 prim::TracedModuleForward
      TORCH_INTERNAL_ASSERT(n->blocks().size() == 1);
      // 确保节点只有一个子块
      convertReturnsToTuples(n->blocks()[0]);

      // 获取当前图和节点的子块
      Graph* g = b->owningGraph();
      Block* sub_block = n->blocks()[0];
      
      // 如果子块有多个输出
      if (sub_block->outputs().size() > 1) {
        {
          // 在子块的返回节点之前插入一个 Tuple 节点
          WithInsertPoint guard(sub_block->return_node());
          Node* return_tup =
              g->insertNode(g->createTuple(sub_block->outputs()));
          // 清空子块的输出
          while (!sub_block->outputs().empty()) {
            sub_block->eraseOutput(0);
          }
          // 将 Tuple 节点的输出注册为子块的新输出
          sub_block->registerOutput(return_tup->output());
        }

        // 将节点的输出合并为一个 Tuple
        std::vector<TypePtr> types;
        for (size_t i = 0; i < n->outputs().size(); ++i) {
          types.push_back(n->output(i)->type());
        }
        // 添加一个新的输出，类型为 TupleType
        Value* tup_output = n->addOutput()->setType(TupleType::create(types));
        // 在节点后面插入 Tuple 解包节点
        Node* tup_unpack = g->createTupleUnpack(tup_output)->insertAfter(n);
        // 替换节点的输出为 Tuple 解包节点的输出
        for (size_t i = 0; i < tup_unpack->outputs().size(); ++i) {
          auto rev_idx = tup_unpack->outputs().size() - i - 1;
          n->output(rev_idx)->replaceAllUsesWith(tup_unpack->output(rev_idx));
          n->eraseOutput(rev_idx);
        }
      } else if (sub_block->outputs().empty()) {
        // 如果子块没有输出，在其返回节点之前插入一个 None 节点
        WithInsertPoint guard(sub_block->return_node());
        sub_block->registerOutput(g->insertNode(g->createNone())->output());
        // 添加一个新的输出，类型为 NoneType
        n->addOutput()->setType(NoneType::get());
      }
    }
  }
// Lambda lift Values (i.e. add Graph inputs for the purpose of
// referencing values that dominate the block) and convert
// the block to a Graph. blocks()[0] on each TracedModuleForward then
// appears as a Graph attribute attr::Subgraph
void lambdaLiftBlocksAndConvertToGraph(Block* b) {
  // Iterate over all nodes in the block
  for (Node* n : b->nodes()) {
    // Check if the node is eligible for lambda lifting
    if (isEligibleNode(n)) {
      // Recursively perform lambda lifting and graph conversion on the first block
      // of the current node
      lambdaLiftBlocksAndConvertToGraph(n->blocks()[0]);

      // Create a new Graph object
      auto graph = std::make_shared<Graph>();
      // Map to hold remapped Values
      std::unordered_map<Value*, Value*> remaps;
      // Clone the contents of the first block of the current node into the new Graph
      graph->block()->cloneFrom(n->blocks()[0], [&](Value* v) {
        if (!remaps.count(v)) {
          // Create a new input for the graph and copy metadata from the original value
          remaps[v] = graph->addInput()->copyMetadata(v);
          // Add the original value as an input to the current node
          n->addInput(v);
        }
        return remaps[v];
      });
      // Perform linting on the created Graph
      LintGraph(graph);
      // Set the created Graph as an attribute attr::Subgraph of the current node
      n->g_(attr::Subgraph, graph);
      // Erase the first block of the current node
      n->eraseBlock(0);
    }
  }
}

// Find a unique name to add this method as
// We try {method_name}, {method_name}1, {method_name}2, ...
std::string mangleMethodName(
    const std::string& method_name,
    const ClassTypePtr& mod_type) {
  // Iterate indefinitely until a unique method name is found
  for (size_t method_idx = 0;; method_idx++) {
    auto mangled = method_name;
    if (method_idx != 0) {
      // Append a numeric suffix if method_idx is not zero
      mangled += std::to_string(method_idx);
    }
    bool found = false;
    // Iterate over existing methods in mod_type to check for name collisions
    for (Function* fn : mod_type->methods()) {
      if (fn->name() == mangled) {
        found = true;
        break;
      }
    }
    // If no collision is found, return the unique mangled method name
    if (!found) {
      return mangled;
    }
  }
  // If execution reaches here, an assertion error is raised as a failsafe
  TORCH_INTERNAL_ASSERT(false);
}

// Register the attr::Subgraph Graph values as Functions in the
// class compilation unit and register that Function as a method
// on the corresponding Module in the Module hierarchy. Note that we
// unique the methods by naming them forward, forward1, forward2...
void createMethodCalls(const std::shared_ptr<Graph>& g) {
  // Iterate over all nodes in the graph g
  for (auto node_itr = g->nodes().begin(); node_itr != g->nodes().end();) {
    Node* n = *node_itr++;
    // If the node represents a TracedFork, recursively process its subgraph
    if (n->kind() == prim::TracedFork) {
      createMethodCalls(n->g(attr::Subgraph));
    }
    // If the node represents a TracedModuleForward, process it to create a method call
    else if (n->kind() == prim::TracedModuleForward) {
      // Set the insertion point for node n
      WithInsertPoint ip(n);

      // Obtain the type of the callee module from the first input of node n
      ClassTypePtr callee_mod_type = n->input(0)->type()->expect<ClassType>();

      // Recursively create method calls for the subgraph of node n
      createMethodCalls(n->g(attr::Subgraph));

      // Generate a mangled method name based on "forward" for the callee module type
      auto mangled_method_name = mangleMethodName("forward", callee_mod_type);
      // Construct a qualified name for the function using callee module's name and mangled name
      auto qualname = c10::QualifiedName(
          callee_mod_type->name().value(), mangled_method_name);
      // Create a new function in the compilation unit of callee_mod_type
      Function* f = callee_mod_type->compilation_unit()->create_function(
          qualname, n->g(attr::Subgraph));
      // Add the created function as a method to the callee module type
      callee_mod_type->addMethod(f);

      // Prepare NamedValues for inputs of node n
      std::vector<NamedValue> nvs;
      for (Value* i : n->inputs()) {
        nvs.emplace_back(i->node()->sourceRange(), i);
      }
      // Match the schema of the function f with inputs of node n and create a method call
      auto schema = matchSchema(f->getSchema(), n->sourceRange(), *g, nvs, {});
      // Insert a method call in graph g and obtain the returned value
      Value* retval = g->insertMethodCall(f->qualname().name(), schema);
      // Replace all uses of the output of node n with the returned value
      n->output()->replaceAllUsesWith(retval);
      // Destroy node n
      n->destroy();
    }
  }
}

void inlineScopeBlocks(Block* b) {
  // Iterate over all nodes in the block b
  for (auto n_itr = b->nodes().begin(); n_itr != b->nodes().end();) {
    // 从迭代器中获取指向 Node 的指针并移动到下一个位置
    Node* n = *n_itr++;
    // 遍历 Node n 中的所有子 Block，对每个子 Block 调用 inlineScopeBlocks 函数
    for (Block* sub_b : n->blocks()) {
      inlineScopeBlocks(sub_b);
    }
    // 如果 Node n 的类型是 prim::TracedModuleForward
    if (n->kind() == prim::TracedModuleForward) {
      // 将 Block 转换为图形表示以便内联处理
      auto graph = std::make_shared<Graph>();
      // 使用 lambda 函数将当前 Block 克隆到新创建的图形中，并更新 Value 指针的映射关系
      std::unordered_map<Value*, Value*> remaps;
      graph->block()->cloneFrom(n->blocks()[0], [&](Value* v) {
        remaps[v] = graph->block()->addInput()->copyMetadata(v);
        // 将新的输入 Value 添加到 Node n 的输入中
        n->addInput(v);
        return remaps[v];
      });

      // 设置插入点为当前 Node n
      WithInsertPoint insert_point(n);
      // 断言 Node n 的输入数量与新图的输入数量相同
      AT_ASSERT(n->inputs().size() == graph->inputs().size());
      // 将新图插入到当前图的指定位置，并返回新图中的输出
      auto new_outputs = insertGraph(*n->owningGraph(), *graph, n->inputs());
      // 获取当前 Node n 的旧输出
      const auto& old_outputs = n->outputs();

      // 断言新旧输出的数量相同
      AT_ASSERT(new_outputs.size() == old_outputs.size());
      // 替换 Node n 的旧输出为新输出
      for (const auto i : c10::irange(old_outputs.size())) {
        old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
      }
      // 销毁当前 Node n
      n->destroy();
    }
  }
} // namespace

// 将标记为 prim::TracedFork 的节点转换为实际的 fork 节点
void convertTracedForksToRealForks(const std::shared_ptr<Graph>& g) {
  // 迭代图中的每个节点
  for (auto itr = g->nodes().begin(); itr != g->nodes().end();) {
    Node* n = *itr++;
    // 如果节点的类型是 prim::TracedFork
    if (n->kind() == prim::TracedFork) {
      // 设置插入点为当前节点 n
      WithInsertPoint guard(n);
      // 在图中插入一个新的 fork 节点，复制原节点的属性
      Node* new_fork_node =
          g->insertNode(g->create(prim::fork, n->outputs().size()))
              ->copyAttributes(*n);
      // 将原节点 n 的输入添加到新的 fork 节点中
      for (Value* i : n->inputs()) {
        new_fork_node->addInput(i);
      }
      // 复制输出的元数据，并替换所有使用原节点输出的地方为新 fork 节点的输出
      for (size_t i = 0; i < new_fork_node->outputs().size(); ++i) {
        new_fork_node->outputs()[i]->copyMetadata(n->outputs()[i]);
        n->outputs()[i]->replaceAllUsesWith(new_fork_node->outputs()[i]);
      }
      // 销毁原节点 n
      n->destroy();
    }
  }
}

// 运行一些清理 passes 以使图更加清晰
void runCleanupPasses(const std::shared_ptr<Graph>& g) {
  // 遍历图中的每个节点
  for (Node* n : g->nodes()) {
    // 如果节点的类型是 prim::TracedFork
    if (n->kind() == prim::TracedFork) {
      // 获取节点 n 的子图
      auto subgraph = n->g(attr::Subgraph);
      // 如果处于内联模式，则内联子图
      if (getInlineEverythingMode()) {
        Inline(*subgraph);
      }
      // 转换标记为 prim::TracedFork 的节点为实际的 fork 节点
      convertTracedForksToRealForks(subgraph);
      // 降低简单元组
      LowerSimpleTuples(subgraph);
      // 消除死代码
      EliminateDeadCode(subgraph);
      // 对子图进行 lint 检查
      LintGraph(subgraph);
    }
  }
  // 如果处于内联模式，则内联整个图
  if (getInlineEverythingMode()) {
    Inline(*g);
  }
  // 转换标记为 prim::TracedFork 的节点为实际的 fork 节点
  convertTracedForksToRealForks(g);
  // 降低简单元组
  LowerSimpleTuples(g);
  // 消除死代码
  EliminateDeadCode(g);
  // 对图进行 lint 检查
  LintGraph(g);
}

// 对模块运行清理 passes
void runCleanupPasses(Module* m) {
  // 获取模块中的方法
  auto methods = m->get_methods();
  // 对每个子模块运行清理 passes
  for (auto module : m->children()) {
    runCleanupPasses(&module);
  }
  // 对每个方法的图运行清理 passes
  for (auto& method : methods) {
    runCleanupPasses(method.graph());
  }
}

} // namespace

// 修复跟踪范围块中的问题
void FixupTraceScopeBlocks(std::shared_ptr<Graph>& graph, Module* self) {
  // 如果 self 存在
  if (self) {
    // 运行 ConvertTracedAttrReferences pass
    ConvertTracedAttrReferences().run(graph);
  } else {
    // 确保图中不存在 prim::TracedAttr 类型的节点
    for (Node* n : graph->nodes()) {
      TORCH_INTERNAL_ASSERT(n->kind() != prim::TracedAttr);
    }
  }
  // 确保定义先于使用
  MakeDefsDominateUses().run(graph->block());
  // 转换返回值为元组
  convertReturnsToTuples(graph->block());
  // 如果没有 self，内联所有作用域块
  if (!self) {
    inlineScopeBlocks(graph->block());
    // 对 TracedFork 节点进行 lambda 提升并转换为图
    lambdaLiftBlocksAndConvertToGraph(graph->block());
    // 运行清理 passes
    runCleanupPasses(graph);
  } else {
    // 对 TracedFork 节点进行 lambda 提升并转换为图
    lambdaLiftBlocksAndConvertToGraph(graph->block());
    // 创建方法调用
    createMethodCalls(graph);
    // 对 self 运行清理 passes
    runCleanupPasses(self);
    // 对 graph 运行清理 passes
    runCleanupPasses(graph);
  }
}

} // namespace jit
} // namespace torch
```