# `.\pytorch\torch\csrc\jit\passes\onnx\list_model_parameters.cpp`

```
// 包含 Torch JIT 前端错误报告、日志、死代码消除等头文件
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/list_model_parameters.h>

// Torch JIT 命名空间
namespace torch {
namespace jit {

// Torch JIT 中的 ONNX 命名空间，引入 C10 和 ONNX 命名空间
namespace onnx {
using namespace ::c10::onnx;
}

// findSubModuleAttr 函数追踪 getAttr 链路，反向定位子模块。
// 例如：对于模块 M {
//   attributes {
//     A = <SubModule at ...>
//   }
//   ...
//   %A = prim::GetAttr[name="A"](%self)
//   ...
//   %B = prim::GetAttr[name="B"](%A)
//   ...
//   %weight = prim::GetAttr[name="scale"](%B)
//   ...
std::deque<std::string> findSubModuleAttr(
    Value* input,
    std::string& name,
    Module& attrModule,
    std::shared_ptr<Graph>& graph) {
  Node* node = input->node();
  std::deque<std::string> moduleNames;

  // 从内部子模块开始循环，沿着链路向上直到达到顶层模块。
  while (node->outputs().at(0)->type() != graph->inputs().at(0)->type()) {
    if (node->kind() == prim::GetAttr) {
      moduleNames.push_front(node->s(attr::name)); // 将获取的属性名添加到队列前端
      node = node->inputs()[0]->node(); // 获取当前节点的输入节点，并继续循环
    } else {
      return moduleNames; // 若节点类型不是 prim::GetAttr，直接返回模块名队列
    }
  }
  // 将内部模块赋值给 attrModule。
  for (auto& moduleName : moduleNames) {
    attrModule = attrModule.attr(moduleName).toModule(); // 获取属性模块并转换为模块类型
  }
  return moduleNames; // 返回模块名队列
}

// 将参数作为函数参数添加到函数的架构中
Value* addParamAsArgument(Function* function, std::string& name, IValue& attr) {
  auto schema = function->getSchema(); // 获取函数的架构
  auto args = schema.arguments(); // 获取函数的参数列表
  args.emplace_back(name, nullptr, c10::nullopt, attr); // 向参数列表中添加新的参数
  auto new_schema = FunctionSchema(
      schema.name(),
      schema.overload_name(),
      args,
      schema.returns(),
      schema.is_vararg(),
      schema.is_varret());
  function->setSchema(new_schema); // 设置更新后的函数架构
  return toGraphFunction(*function).graph()->addInput(name)->setType(
      attr.type()); // 将参数添加到函数的图中，并设置类型
}

// 获取块中的参数属性，并根据模块的训练状态确定是否处于评估模式
std::vector<IValue> getParamAttributes(
    Block* block,
    std::shared_ptr<Graph>& graph,
    const Module& module_,
    Function* function_,
    std::unordered_map<std::string, Value*>& attrValues) {
  auto isEval = !module_.hasattr("training") || !module_.is_training(); // 判断模块是否处于训练状态

  Node* m = *block->nodes().begin(); // 获取块中的第一个节点
  WithInsertPoint guard(m); // 设置插入点为当前节点

  std::vector<IValue> parameterIValues = {}; // 初始化参数值的向量
  std::unordered_set<Node*> nodesToDestroy; // 创建待销毁节点的集合
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // 可以销毁节点 n
    // 在块的开头插入新节点
    if (isEval && n->kind() == aten::batch_norm) {
      auto inputs = n->inputs();
      auto weight_v = module_.attr("weight");
      attrValues["weight"] = graph->insertConstant(weight_v.toTensor(), nullptr, n);
      auto bias_v = module_.attr("bias");
      attrValues["bias"] = graph->insertConstant(bias_v.toTensor(), nullptr, n);
      auto running_mean_v = module_.attr("running_mean");
      attrValues["running_mean"] = graph->insertConstant(running_mean_v.toTensor(), nullptr, n);
      auto running_var_v = module_.attr("running_var");
      attrValues["running_var"] = graph->insertConstant(running_var_v.toTensor(), nullptr, n);
      nodesToDestroy.emplace(n);
    }
  }

  // 从图中移除所有待销毁节点
  for (Node* n : nodesToDestroy) {
    n->removeAllInputs();
  }
  
  return parameterIValues; // 返回参数值的向量
}
    // 检查节点类型是否为获取属性（prim::GetAttr）或设置属性（prim::SetAttr）
    if (n->kind() == prim::GetAttr || n->kind() == prim::SetAttr) {
      // 如果节点类型为获取属性（prim::GetAttr）
      if (n->kind() == prim::GetAttr) {
        // 遍历节点输出的所有使用
        for (auto use : n->output()->uses()) {
          // 如果使用节点的类型为 prim::PythonOp，则抛出错误报告
          if (use.user->kind() == prim::PythonOp)
            throw ErrorReport(n->sourceRange())
                << "Couldn't export Python method.";
        }
      }

      // 获取属性名称
      auto name = n->s(attr::name);
      // 设置属性模块为当前模块
      auto attrModule = module_;
      // 获取输入值
      auto input = n->inputs()[0];

      // 查找子模块属性的名称列表
      auto moduleNames = findSubModuleAttr(input, name, attrModule, graph);
      // 如果属性模块中没有该属性名称，则继续下一个节点处理
      if (!attrModule.hasattr(name))
        continue;
      // 获取属性值
      auto attr = attrModule.attr(name);
      // 参数常量初始化为空指针
      Value* paramConst = nullptr;

      // 构建完整的属性名称路径
      std::string fullName("");
      for (auto& name : moduleNames) {
        fullName += name + '.';
      }
      fullName += name;

      // 获取属性模块的类型
      auto type = attrModule.type();
      // 查找属性名称在类型中的槽位
      auto slot = *type->findAttributeSlot(name);

      // 将 model_parameters 和 model_buffers 添加为模型输入，按图中出现顺序保持顺序
      if (type->is_parameter(slot) || type->is_buffer(slot) ||
          (attr.isObject() && !attr.toObjectRef().type()->is_module()) ||
          attr.isBool()) {
        // 如果属性值是张量且不存在于 attrValues 中
        if (attrValues.find(fullName) == attrValues.end() &&
            attr.isTensor()) { // TODO: Handle float/int
          // 断言属性为张量类型
          TORCH_INTERNAL_ASSERT(attr.isTensor());
          // 转换属性为张量
          auto tensor_ = attr.toTensor();
          // 如果是评估模式且张量需要梯度，则将其分离并设置梯度不可求导
          if (isEval && tensor_.requires_grad()) {
            tensor_ = tensor_.detach();
            tensor_.set_requires_grad(false);
            attr = IValue(tensor_);
          }
          // 将张量添加到参数值列表中
          parameterIValues.emplace_back(attr.toTensor());
          // 将参数常量作为函数参数添加
          paramConst = addParamAsArgument(function_, fullName, attr);
          // 将属性值插入到属性值映射中
          attrValues.insert({fullName, paramConst});
        } else if (attr.isObject() && !attr.toObjectRef().type()->is_module()) {
          // 仅支持下面注册的 torch 类的对象
          try {
            // 运行对象的 "__getstate__" 方法并将结果添加到参数值列表中
            parameterIValues.emplace_back(
                script::Object(attr.toObject()).run_method("__getstate__"));
            // 将参数常量作为函数参数添加
            paramConst = addParamAsArgument(function_, fullName, attr);
            // 将属性值插入到属性值映射中
            attrValues.insert({fullName, paramConst});
          } catch (const std::exception&) {
            // 抛出错误报告，指示处理模型参数时遇到未知类型
            throw ErrorReport(n->sourceRange())
                << "Unknown type " << attr.type()->repr_str()
                << " encountered in handling model params."
                << " This class type does not extend __getstate__ method.";
          }
        } else if (attr.isNone() || (attr.isBool() && name == "training")) {
          // 对于 ONNX，此属性是常量
          // 尝试在图中插入常量值，并替换节点输出的所有使用
          auto attrVal = tryInsertConstant(*graph, attr);
          n->output()->replaceAllUsesWith(*attrVal);
          // 将节点添加到待销毁节点列表中
          nodesToDestroy.emplace(n);
        }
      }
    }
    // 对于节点 n 的每一个子块，遍历其块列表
    for (Block* sub_block : n->blocks()) {
      // 调用函数 getParamAttributes 获取子块的参数属性值列表
      auto nextParameterIValues =
          getParamAttributes(sub_block, graph, module_, function_, attrValues);
      // 将获取到的参数属性值列表追加到 parameterIValues 的末尾
      parameterIValues.insert(
          std::end(parameterIValues),
          std::begin(nextParameterIValues),
          std::end(nextParameterIValues));
    }
  }
  // 遍历需要销毁的节点列表 nodesToDestroy
  for (auto n : nodesToDestroy) {
    // 销毁节点 n
    n->destroy();
  }
  // 返回汇总后的 parameterIValues，即所有节点的参数属性值列表
  return parameterIValues;
// 将主模块作为常量插入到图中
void insertMainModuleAsConstant(const std::shared_ptr<Graph>& graph) {
    // 创建一个常量节点
    auto* constNode = graph->create(prim::CreateObject);
    // 设置常量节点的输出类型为图的第一个输入类型
    constNode->output()->setType(graph->inputs().at(0)->type());
    // 在图的第一个节点之前插入常量节点
    auto it = graph->nodes().begin();
    constNode->insertBefore(*it);
    // 用常量节点的输出替换图的第一个输入的所有用法
    graph->inputs().at(0)->replaceAllUsesWith(constNode->output());
    // 删除图的第一个输入
    graph->eraseInput(0);
}

// 列出模块的参数及其初始值
std::pair<Module, std::vector<IValue>> list_module_parameters(
    const Module& module) {
  // 克隆模块，包括其所有子模块
  Module moduleClone = module.clone(true);
  // 获取模块的名为 "forward" 的方法
  Method method = moduleClone.get_method("forward");
  // 获取方法的函数对象
  auto function = &method.function();
  // 将函数对象转换为图对象
  auto graph = toGraphFunction(*function).graph();
  // 一个映射，用于存储属性名称和值的引用，以避免重复
  std::unordered_map<std::string, Value*> attrValues = {};

  // 调试信息，显示正在获取函数的属性
  GRAPH_DEBUG("Fetch attributes for function: " + function->name());
  // 获取参数的初始值作为 IValue 的向量
  std::vector<IValue> parameterIValues = getParamAttributes(
      graph->block(), graph, moduleClone, function, attrValues);
  // 将主模块作为常量插入到图中
  insertMainModuleAsConstant(graph);
  // 调试信息，显示列出的参数作为输入
  GRAPH_DEBUG("Listed parameters as inputs: ", *graph);

  // 返回模块的克隆和参数的初始值向量
  return std::make_pair(moduleClone, parameterIValues);
}

// 命名空间 jit 结束
} // namespace jit

// 命名空间 torch 结束
} // namespace torch
```