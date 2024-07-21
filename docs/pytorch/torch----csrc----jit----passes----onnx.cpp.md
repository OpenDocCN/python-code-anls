# `.\pytorch\torch\csrc\jit\passes\onnx.cpp`

```
// 包含头文件：torch/csrc/jit/passes/onnx.h，提供了关于ONNX的一些功能
#include <torch/csrc/jit/passes/onnx.h>

// 包含头文件：ATen/core/functional.h，提供了一些核心的功能函数
#include <ATen/core/functional.h>

// 包含头文件：c10/util/Exception.h，提供了异常处理相关的功能
#include <c10/util/Exception.h>

// 包含头文件：c10/util/irange.h，提供了一些与范围相关的功能
#include <c10/util/irange.h>

// 包含头文件：torch/csrc/autograd/function.h，提供了自动求导相关的功能
#include <torch/csrc/autograd/function.h>

// 包含头文件：torch/csrc/autograd/symbolic.h，提供了符号求导相关的功能
#include <torch/csrc/autograd/symbolic.h>

// 包含头文件：torch/csrc/jit/ir/constants.h，提供了IR常量相关的功能
#include <torch/csrc/jit/ir/constants.h>

// 包含头文件：torch/csrc/jit/jit_log.h，提供了JIT日志相关的功能
#include <torch/csrc/jit/jit_log.h>

// 包含头文件：torch/csrc/jit/passes/dead_code_elimination.h，提供了死代码消除相关的功能
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 包含头文件：torch/csrc/jit/passes/onnx/constant_map.h，提供了ONNX常量映射相关的功能
#include <torch/csrc/jit/passes/onnx/constant_map.h>

// 包含头文件：torch/csrc/jit/passes/onnx/helper.h，提供了ONNX辅助函数相关的功能
#include <torch/csrc/jit/passes/onnx/helper.h>

// 包含头文件：torch/csrc/jit/passes/onnx/onnx_log.h，提供了ONNX日志相关的功能
#include <torch/csrc/jit/passes/onnx/onnx_log.h>

// 包含头文件：torch/csrc/jit/passes/onnx/shape_type_inference.h，提供了ONNX形状和类型推断相关的功能
#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>

// 包含头文件：torch/csrc/jit/python/python_ir.h，提供了Python IR相关的功能
#include <torch/csrc/jit/python/python_ir.h>

// 包含头文件：torch/csrc/utils/pybind.h，提供了Python绑定相关的功能
#include <torch/csrc/utils/pybind.h>

// 包含头文件：sstream，提供了字符串流处理相关的功能
#include <sstream>

// 包含头文件：unordered_set，提供了无序集合相关的功能
#include <unordered_set>

// 引入torch命名空间
namespace torch {

// 引入jit命名空间
namespace jit {

// 移除图中所有的打印操作
void removePrintOps(Block* block) {
  // 遍历图中的每个节点
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    // 递归处理每个节点包含的子块
    for (auto b : it->blocks()) {
      removePrintOps(b);
    }
    // 如果当前节点是打印操作或者警告操作
    if (it->kind() == prim::Print || it->kind() == aten::warn) {
      // 遍历处理当前节点的所有输入
      for (size_t i = 0; i < it->inputs().size();) {
        auto input = it->inputs().at(i);
        // 仅处理常量输入，因为可能会有副作用
        if (input->uses().size() == 1 &&
            input->node()->kind() == prim::Constant) {
          // 移除当前输入并销毁对应的常量节点
          it->removeInput(i);
          input->node()->destroy();
        } else {
          ++i;
        }
      }
      // 移除当前节点
      it.destroyCurrent();
    }
  }
}

// 移除图中的所有打印操作
void RemovePrintOps(std::shared_ptr<Graph>& graph) {
  // 调用移除打印操作的函数，从图的根块开始
  removePrintOps(graph->block());
  // 打印移除打印操作后的图信息
  GRAPH_DUMP("After RemovePrintOps: ", graph);
}

// 检查ONNX兼容性，确保函数schema中的输入是符合ONNX规范的
void checkONNXCompatibility(const c10::FunctionSchema& schema) {
  // 在ONNX中，所有的输入都是张量，不支持张量列表
  // 所以最多支持一个输入张量列表
  bool has_tensor_list = false;
  const auto& args = schema.arguments();
  for (const auto& arg : args) {
    if (arg.name() == "_caffe2_preallocated_outputs") {
      continue;
    }
    auto type = arg.type();
    // 如果类型是OptionalType，则获取其元素类型
    if (type->kind() == TypeKind::OptionalType) {
      type = reinterpret_cast<OptionalType*>(type.get())->getElementType();
      // 递归的OptionalType不被支持
      TORCH_INTERNAL_ASSERT(type->kind() != TypeKind::OptionalType);
    }
    // 如果类型是ListType，则检查其元素类型是否为TensorType
    if (type->kind() == TypeKind::ListType) {
      const auto& elem_type =
          reinterpret_cast<ListType*>(type.get())->getElementType();
      if (elem_type->isSubtypeOf(*TensorType::get())) {
        TORCH_INTERNAL_ASSERT(
            !has_tensor_list,
            "ONNX export supports at most one TensorList as input.");
        has_tensor_list = true;
      }
    }
  }
}

// 预处理Caffe2操作，处理图中的所有节点，目前未具体实现该函数
void preprocessCaffe2Ops(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto b : it->blocks()) {
      preprocessCaffe2Ops(b);
    }
    // 此处存在语法错误：额外的右大括号
    }
  }
  // 执行死代码消除，允许删除具有副作用节点
  EliminateDeadCode(
      block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

// 结束jit命名空间
} // namespace jit

// 结束torch命名空间
} // namespace torch
// 调用 preprocessCaffe2Ops 函数，预处理传入图形的块
void PreprocessCaffe2Ops(std::shared_ptr<Graph>& graph) {
  preprocessCaffe2Ops(graph->block());
  // 输出调用预处理后的图形状态
  GRAPH_DUMP("After PreprocessCaffe2Ops: ", graph);
}

// 将 PythonOps 转换为符合 ONNX 语义的节点
std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& graph,
    ::torch::onnx::OperatorExportTypes operator_export_type) {
  // 获取常量值映射实例，并清空映射
  auto constant_value_map = ConstantValueMap::getInstance();
  ConstantValueMap::ClearMaps();
  // 创建新的图形对象
  auto new_graph = std::make_shared<Graph>(graph->current_scope());
  py::dict env;
  // 环境中的值，用于常量时间存在性检查，与 env 中的值保持一致
  py::set values_in_env;
  try {
    // 将图形的块转换为 ONNX 图形
    BlockToONNX(
        graph->block(),
        new_graph->block(),
        operator_export_type,
        env,
        values_in_env);
  } catch (std::runtime_error& ex) {
    // 输出异常时正在构建的 ONNX 图形状态
    ONNX_LOG(
        "ONNX graph being constructed during exception:\n",
        new_graph->toString());
    throw;
  }
  // 输出转换为 ONNX 后的新图形状态
  GRAPH_DUMP("after ToONNX: ", new_graph);
  // 清空常量值映射
  ConstantValueMap::ClearMaps();
  return new_graph;
}

// 将 Block 转换为 ONNX
// is_sub_block = true 表示旧块（aten 图）位于子块中（例如，如果子块），并且我们希望将其转换为 ONNX 图中的父块。
// 在这种情况下，我们不注册输入/输出或消除死代码。
py::dict BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    py::dict& env,
    py::set& values_in_env,
    bool is_sub_block) {
  // 符号上下文对象
  torch::autograd::SymbolicContext ctx{};
  ctx.block = new_block;

  // 输出旧块图形的信息
  GRAPH_DEBUG(
      "BlockToONNX: graph of old block: ",
      old_block->owningGraph()->toString());

  // 初始化上下文和环境
  if (!is_sub_block) {
    // 如果不是子块，则为旧块的每个输入添加元数据复制后的输入节点
    for (auto input : old_block->inputs()) {
      auto n = ctx.block->addInput()->copyMetadata(input);
      auto py_n = py::cast(n);
      env[py::cast(input)] = py_n;
      values_in_env.add(py_n);
    }
  }

  // 确定所有输入是否为静态。这用于每个节点确定是否传播形状。
  if (!is_sub_block) {
    bool static_input_shape = AllGraphInputsStatic(ctx.block->owningGraph());
    ConstantValueMap::SetAllGraphInputsStatic(static_input_shape);
  }

  // 最后，访问图中的所有节点
  for (auto node : old_block->nodes()) {
    NodeToONNX(node, ctx.block, operator_export_type, env, values_in_env);
  }

  // 如果是子块，则返回环境
  if (is_sub_block) {
    return env;
  }

  // 否则，注册旧块的每个输出
  for (auto output : old_block->outputs()) {
    auto py_value = env[py::cast(output)];
    Value* value = py_value.cast<Value*>();
    ctx.block->registerOutput(value);
  }
  // 运行 DCE 清理未使用的函数和原位操作
  EliminateDeadCode(
      ctx.block,
      true,
      DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);

  // 返回空字典
  return py::dict();
}
// 判断是否可以进行常量折叠的条件：节点的类型不是常量并且在常量数值映射中存在该节点的数值
bool ConstantFoldCondition(torch::jit::Value* output) {
  auto fold_condition = output->node()->kind() != c10::onnx::Constant &&
      ConstantValueMap::HasValue(output->debugName());
  // 检查节点的类型是否可靠，即在类型可靠性映射中存在该节点的类型信息，否则返回false
  auto reliable_value =
      ConstantValueMap::GetTypeReliable(output->debugName()).value_or(false);
  // 返回是否满足常量折叠条件的结果
  return fold_condition && reliable_value;
}

// 将节点转换为 ONNX 格式的函数
void NodeToONNX(
    Node* old_node,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    py::dict& env,
    py::set& values_in_env) {
  // 导入必要的 Python 模块
  py::object onnx = py::module::import("torch.onnx");
  py::object onnx_globals = py::module::import("torch.onnx._globals");
  py::object onnx_registration =
      py::module::import("torch.onnx._internal.registration");

  // 设置所有的 Lambda 辅助函数。

  // 返回环境映射中节点 n 对应的值
  auto envFn = [&env](Value* n) -> Value* {
    auto py_n = py::cast(n);
    // 检查环境映射中是否包含节点 n
    TORCH_CHECK(env.contains(py_n), "Dangling node reference");
    // 获取节点对应的 Python 值
    auto py_value = env[py_n];
    // 检查获取的值不是空值
    TORCH_CHECK(!py_value.is_none(), "Unused node was subsequently used");
    // 将 Python 值转换为节点的值，并返回
    Value* value = py_value.cast<Value*>();
    return value;
  };

  // 将新的输出放入环境映射中，并且如果符号调用未设置输出类型，则从输入图中复制类型。
  auto setOutputs = [&](const std::string& op_name,
                        Node* node,
                        const value_list& outputs) {
    auto old_outputs = node->outputs();
    // 统计旧的输出数量，不包括 Handles
    auto num_old_outputs = old_outputs.size();
    // 检查符号操作生成的输出数量是否正确
    if (outputs.size() != num_old_outputs) {
      std::ostringstream ss;
      ss << "symbolic for " << op_name
         << " produced an incorrect number of outputs (expected ";
      ss << num_old_outputs << ", but got " << outputs.size() << ")";
      throw std::runtime_error(ss.str());
    }
    // 对于常量节点，不需要 params_dict 信息，因此将其设置为空字典。
    const ParamMap empty_params_dict = {};
    // 获取当前导出的 ONNX 版本
    auto opset_version = py::cast<int>(
        onnx_globals.attr("GLOBALS").attr("export_onnx_opset_version"));
    // 进行其他处理（此处应包含设置输出的逻辑，代码被截断，未完）
  };

  // 克隆节点并将其添加到新图中
  auto cloneNode = [&](Node* node) {
    auto n_ = new_block->appendNode(
        new_block->owningGraph()->createClone(node, envFn));
    // 遍历节点的所有输出
    for (const auto i : c10::irange(node->outputs().size())) {
      // 将克隆节点的输出类型设置为原节点的输出类型
      // n_->outputs()[i]->setType(node->outputs()[i]->type());
      // 将 Python 输出值添加到环境映射中
      auto py_output = py::cast(n_->output(i));
      env[py::cast(node->output(i))] = py_output;
      // 将 Python 输出值添加到环境中的值集合中
      values_in_env.add(py_output);
    }
  };

  // 内联 prim::PythonOp 子块节点并将其附加到 ONNX 图中
  auto inlineAutograd = [&](Node* PythonOpNode) {
    for (auto subblock : PythonOpNode->blocks()) {
      // 遍历 PythonOpNode 的子块
      for (const auto i : c10::irange(PythonOpNode->inputs().size())) {
        // 获取 PythonOpNode 的第 i 个输入，并从环境中获取对应的 Python 值
        auto py_value = env[py::cast(PythonOpNode->inputs()[i])];
        // 将获取到的 Python 值存入环境中，使用子块的第 i 个输入作为键
        env[py::cast(subblock->inputs()[i])] = py_value;
        // 将 py_value 添加到 values_in_env 集合中
        values_in_env.add(py_value);
      }
      // 遍历子块中的节点，对每个节点调用 NodeToONNX 函数进行转换
      for (auto* node : subblock->nodes()) {
        NodeToONNX(node, new_block, operator_export_type, env, values_in_env);
      }
      // 处理子块的输出
      for (const auto i : c10::irange(PythonOpNode->outputs().size())) {
        // 获取子块的第 i 个输出，并从环境中获取对应的 Python 值
        auto py_value = env[py::cast(subblock->outputs()[i])];
        // 将获取到的 Python 值存入环境中，使用 PythonOpNode 的第 i 个输出作为键
        env[py::cast(PythonOpNode->outputs()[i])] = py_value;
        // 将 py_value 添加到 values_in_env 集合中
        values_in_env.add(py_value);
      }
    }
  };

  // Cast output of symbolic() python implementation
  auto processSymbolicOutput = [&](const std::string& op_name,
                                   Node* n,
                                   const py::object& raw_output) {
    // 如果 raw_output 是 None，则克隆当前节点
    if (raw_output.ptr() == Py_None) {
      cloneNode(n);
      return;
    }
    // 将输出转换回 C++ 类型，并放入新图中
    std::vector<Value*> outputs;
    try {
      if (py::isinstance<Value>(raw_output)) {
        outputs = value_list{py::cast<Value*>(raw_output)};
      } else {
        outputs = py::cast<std::vector<Value*>>(raw_output);
      }
    } catch (const std::exception& ex) {
      // 抛出异常，指示类型转换错误
      std::ostringstream ss;
      ss << "Error casting results of symbolic for " << op_name
         << ": expected to return list of op nodes, instead received type '"
         << py::str(raw_output.get_type()) << "': " << py::str(raw_output);
      throw std::runtime_error(ss.str());
    }

    // 设置节点的输出
    setOutputs(op_name, n, outputs);
  };

  auto callPySymbolicFunction = [&](Node* n) {
    // 将大部分参数处理工作委托给 Python
    py::tuple py_inputs(n->inputs().size());
    Py_ssize_t input_nr = 0;
    for (auto* input : n->inputs()) {
      // 将节点的输入转换为 Python 对象，并放入 py_inputs 中
      py_inputs[input_nr++] = py::cast(envFn(input));
    }

    // 获取当前图和作用域
    Graph* g = new_block->owningGraph();
    WithInsertPoint insert_point_guard(new_block);
    WithCurrentScope scope_guard(*g, n->scope());

    // IMPORTANT: NEVER pass raw pointer of smart pointer managed objects to
    // Python. Check #87343 for details.
    // 调用 Python 的 _run_symbolic_function 函数，生成新节点列表和原始输出
    py::list new_nodes = py::list();
    py::object raw_output = onnx.attr("_run_symbolic_function")(
        g->shared_from_this(),
        new_block,
        n,
        py_inputs,
        env,
        values_in_env,
        new_nodes,
        operator_export_type);

    // 遍历由 _run_symbolic_function 创建的新节点，并传播元数据
    for (py::handle py_node : new_nodes) {
      Node* node = py_node.cast<Node*>();
      node->copyMetadata(n);
    }

    // 处理符号函数的输出
    processSymbolicOutput(n->kind().toUnqualString(), n, raw_output);
    // 在处理后输出当前图的状态
    GRAPH_DUMP("after processSymbolicOutput: ", g);
  };

  auto callPySymbolicMethod = [&](ConcretePythonOp* op) {
    // 测试是否存在符号函数；如果不存在则退出
    auto pyobj = py::handle(op->pyobj.get());
    auto func = op->autogradFunction();
    if (func) {
      // 如果存在自动求导函数，则使用其对应的对象
      pyobj = func->get();
    }
    
    // 获取导出 ONNX 操作集版本号
    py::object opset_version =
        onnx_globals.attr("GLOBALS").attr("export_onnx_opset_version");
    
    // 调用内部注册表注册模块中定义的符号方法
    bool is_registered_op =
        onnx_registration.attr("registry")
            .attr("is_registered_op")("prim::PythonOp", opset_version)
            .cast<bool>();
    
    // 获取全局变量中的自动求导内联设置
    py::bool_ is_autograd_inlining_enabled =
        py::cast<bool>(onnx_globals.attr("GLOBALS").attr("autograd_inlining"));
    
    if (!py::hasattr(pyobj, "symbolic") && !is_registered_op) {
      // 在 prim::PythonOp 中内联子图，除非以下条件之一成立：
      // 1. 此节点对象的 torch.autograd.Function 类具有 `symbolic` 方法。
      // 2. prim::PythonOp 已注册自定义导出符号方法。
      if ((operator_export_type == ::torch::onnx::OperatorExportTypes::ONNX ||
           operator_export_type ==
               ::torch::onnx::OperatorExportTypes::ONNX_ATEN_FALLBACK) &&
          (py::cast<bool>(is_autograd_inlining_enabled))) {
        try {
          // 尝试内联自动求导
          inlineAutograd(op);
        } catch (const std::exception& ex) {
          // 捕获异常并警告无法内联 PythonOp 的具体原因
          TORCH_WARN(
              "Unable to inline PythonOp: ",
              op->name(),
              " due to the following exception\n",
              ex.what(),
              "prim::PythonOp will be exported as is and without being inlined\n",
              "Try exporting with the following alternatives: \n",
              "1) Set operator_export_type to ONNX_FALLTHROUGH mode\n",
              "2) Register a symbolic method for the prim::PythonOp ",
              op->name());
          cloneNode(op); // 复制节点以避免内联失败
        }
      } else {
        cloneNode(op); // 复制节点以避免内联
      }
      return;
    }
    
    // 为 Python 准备参数。第一个参数是图形，后跟常规参数，其中变量替换为相应的节点。
    Py_ssize_t input_nr = 0;
    py::tuple py_symbolic_args(op->cconv.size());
    auto inputs = op->inputs();
    auto node_it = inputs.begin();
    auto scalar_it = op->scalar_args.begin();
    for (auto arg_type : op->cconv) {
      py::object obj;
      if (arg_type == 'c') {
        TORCH_CHECK(
            scalar_it != op->scalar_args.end(),
            "expected too many scalar args");
        obj = py::reinterpret_borrow<py::object>(
            py::handle((scalar_it++)->get()));
      } else if (arg_type == 'd') {
        TORCH_CHECK(node_it != inputs.end(), "expected too many inputs");
        obj = py::cast(envFn(*node_it++));
      } else {
        throw std::runtime_error("unexpected calling convention");
      }
      py_symbolic_args[input_nr++] = obj;
    }
    
    // 设置插入点到新块
    WithInsertPoint insert_point_guard(new_block);
    // 使用 WithCurrentScope 对象管理新块的作用域，确保操作在正确的图形上下文中执行
    WithCurrentScope scope_guard(*new_block->owningGraph(), op->scope());

    // 检查 Python 对象 pyobj 是否具有 "symbolic" 属性
    if (py::hasattr(pyobj, "symbolic")) {
      // 如果有 "symbolic" 属性，则调用其 symbolic 函数
      // 使用一个小型的桥接函数以便在参数不匹配时提供良好的错误消息
      // 将操作注册为自定义运算符
      // TODO: 找到一种更优雅的方法来实现此操作，避免直接操作内部 Python 模块。
      // TODO(justinchuby): 为这些 Python 操作定义一个命名空间。
      onnx_registration.attr("registry")
          .attr("register")(
              "::" + op->name(),
              opset_version,
              pyobj.attr("symbolic"),
              /* custom */ true);

      // 在 onnx 模块上调用 "_run_symbolic_method" 方法，传递图形上下文、操作名称、symbolic 函数和符号参数
      py::object raw_output = onnx.attr("_run_symbolic_method")(
          new_block->owningGraph()->shared_from_this(),
          op->name(),
          pyobj.attr("symbolic"),
          py_symbolic_args);

      // 处理 symbolic 函数的输出结果
      processSymbolicOutput(op->name(), op, raw_output);
    } else {
      // 如果没有 "symbolic" 属性，则应该已经注册了操作符
      TORCH_INTERNAL_ASSERT(is_registered_op);
      Node* n = static_cast<Node*>(op);
      // 设置节点的名称属性为操作名称
      n->s_(attr::name, op->name());

      // 调用 symbolic 函数
      // 在 onnx 模块上调用 "_run_symbolic_function" 方法，传递图形上下文、新块、节点、符号参数、环境、环境中的值、新节点列表和导出类型
      py::list new_nodes = py::list();
      py::object raw_output = onnx.attr("_run_symbolic_function")(
          new_block->owningGraph()->shared_from_this(),
          new_block,
          n,
          py_symbolic_args,
          env,
          values_in_env,
          new_nodes,
          operator_export_type);

      // 处理 symbolic 函数的输出结果
      processSymbolicOutput(op->kind().toUnqualString(), n, raw_output);
    }
  };

  // 获取旧节点的操作类型
  auto k = old_node->kind();
  if (k.is_caffe2()) {
    // 如果操作类型是 Caffe2，则直接克隆节点
    cloneNode(old_node);
  } else if (k == prim::PythonOp) {
    // 如果操作类型是 PythonOp，则调用 Python 符号方法处理
    callPySymbolicMethod(static_cast<ConcretePythonOp*>(old_node));
  } else {
    // 对于其他操作类型，调用 Python 符号函数处理
    callPySymbolicFunction(old_node);
  }
}

} // namespace jit
} // namespace torch
```