# `.\pytorch\torch\csrc\jit\passes\quantization\finalize.cpp`

```
// 包含 Torch 库中的量化优化相关头文件
#include <torch/csrc/jit/passes/quantization/finalize.h>

// 包含 Torch JIT 日志相关头文件
#include <torch/csrc/jit/jit_log.h>

// 包含 Torch JIT 清除分析数据相关头文件
#include <torch/csrc/jit/passes/clear_profiling.h>

// 包含 Torch JIT 公共子表达式消除相关头文件
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

// 包含 Torch JIT 常量池化相关头文件
#include <torch/csrc/jit/passes/constant_pooling.h>

// 包含 Torch JIT 常量传播相关头文件
#include <torch/csrc/jit/passes/constant_propagation.h>

// 包含 Torch JIT 死代码消除相关头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 包含 Torch JIT 模块冻结相关头文件
#include <torch/csrc/jit/passes/freeze_module.h>

// 包含 Torch JIT 循环展开相关头文件
#include <torch/csrc/jit/passes/loop_unrolling.h>

// 包含 Torch JIT Peephole 优化相关头文件
#include <torch/csrc/jit/passes/peephole.h>

// 包含 Torch JIT 预打包折叠相关头文件
#include <torch/csrc/jit/passes/prepack_folding.h>

// 包含 Torch JIT 量化模式相关头文件
#include <torch/csrc/jit/passes/quantization/quantization_patterns.h>

// 包含 Torch JIT 注册打包参数相关头文件
#include <torch/csrc/jit/passes/quantization/register_packed_params.h>

// 包含 Torch JIT 图遍历相关头文件
#include <torch/csrc/jit/runtime/graph_iterator.h>

// 包含标准库中的实用工具
#include <utility>

namespace torch {
namespace jit {

// 匿名命名空间下定义函数，用于向图中插入针对线性层的预打包和解包操作
void insertPrepackUnpackForLinear(std::shared_ptr<Graph>& graph) {
  // 调用线性层预打包和解包模式获取函数，返回模式和替换列表
  std::vector<QuantFusionInfo> patterns_and_replacements =
      linear_prepack_unpack_patterns();

  // 遍历模式和替换列表中的每一项
  for (const auto& entry : patterns_and_replacements) {
    // 创建子图重写器对象
    SubgraphRewriter rewriter;
    // 注册重写模式和替换方法
    rewriter.RegisterRewritePattern(entry.pattern, entry.replacement);
    // 在图上运行子图重写器，应用过滤器
    rewriter.runOnGraph(graph, entry.filters);
  }
}

// 匿名命名空间下定义函数，用于向图中插入针对卷积层的预打包和解包操作
void insertPrepackUnpackForConv(std::shared_ptr<Graph>& graph) {
  // 调用卷积层预打包和解包模式获取函数，返回模式和替换列表
  std::vector<QuantFusionInfo> patterns_and_replacements =
      conv_prepack_unpack_patterns();

  // 遍历模式和替换列表中的每一项
  for (const auto& entry : patterns_and_replacements) {
    // 创建子图重写器对象
    SubgraphRewriter rewriter;
    // 注册重写模式和替换方法
    rewriter.RegisterRewritePattern(entry.pattern, entry.replacement);
    // 在图上运行子图重写器，应用过滤器
    rewriter.runOnGraph(graph, entry.filters);
  }
}

// 匿名命名空间下定义函数，用于从图中移除打包参数插入和 FP 权重设置属性
void removePackedParamInsertionAndFPWeightsSetAttr(
    std::shared_ptr<Graph>& g,
    const std::unordered_set<std::string>& packed_param_attr_names) {
  // 深度优先图节点迭代器
  DepthFirstGraphNodeIterator it(g);
  Node* n = nullptr;
  std::vector<Node*> nodes_to_delete;
  // 遍历图中的每个节点
  while ((n = it.next()) != nullptr) {
    // 如果节点是 prim::SetAttr 类型
    if (n->kind() == prim::SetAttr) {
      // 获取属性名
      const std::string& attr_name = n->s(attr::name);
      // 如果属性名存在于打包参数属性名集合中
      if (packed_param_attr_names.count(attr_name)) {
        // 将节点添加到待删除节点列表中
        nodes_to_delete.push_back(n);
      } else {
        // 获取节点的输入值和模块的输入值
        Value* v = n->input(0);
        Value* self = g->inputs()[0];
        // 获取模块访问路径
        std::vector<std::string> paths = getModuleAccessPath(v, self);
        std::string path = joinPaths(paths);
        // 如果路径存在于打包参数属性名集合中
        if (packed_param_attr_names.count(path)) {
          // 将节点添加到待删除节点列表中
          nodes_to_delete.push_back(n);
        }
      }
    }
  }
  // 删除待删除节点的所有输入
  for (auto node : nodes_to_delete) {
    node->removeAllInputs();
  }
  // 销毁待删除节点
  for (auto node : nodes_to_delete) {
    node->destroy();
  }
  // 执行常量池化优化
  ConstantPooling(g);
  // 执行死代码消除优化
  EliminateDeadCode(g);
}

// 匿名命名空间下定义函数，用于从图中移除观察者调用方法
void removeObserverCallMethods(std::shared_ptr<Graph>& g) {
  // 深度优先图节点迭代器
  DepthFirstGraphNodeIterator it(g);
  Node* n = nullptr;
  std::vector<Node*> nodes_to_delete;
  // 遍历图中的每个节点
  while ((n = it.next()) != nullptr) {
      if (n->kind() == prim::CallMethod || n->kind() == prim::CallFunction) {
          const auto& methodName = n->s(attr::name);
          if (methodName == "forward" || methodName == "calculate") {
              nodes_to_delete.push_back(n);
          }
      }
  }
  // 删除待删除节点的所有输入
  for (auto node : nodes_to_delete) {
    node->removeAllInputs();
  }
  // 销毁待删除节点
  for (auto node : nodes_to_delete) {
    node->destroy();
  }
}
    # 检查节点类型是否为方法调用
    if (n->kind() == prim::CallMethod) {
      # 获取调用方法的属性名
      const std::string& attr_name = n->s(attr::name);
      # 检查属性名是否为"calculate_qparams"
      if (attr_name == "calculate_qparams") {
        # 获取调用节点的第一个输入节点
        auto observer_node = n->input(0)->node();
        # 检查输入节点是否为属性获取，并且属性名包含"_observer_"
        if (observer_node->kind() == prim::GetAttr &&
            observer_node->s(attr::name).find("_observer_") !=
                std::string::npos) {
          # 将符合条件的节点加入待删除列表
          nodes_to_delete.push_back(n);
        }
      }
    }
  }
  # 移除所有待删除节点的输入连接
  for (auto node : nodes_to_delete) {
    node->removeAllInputs();
  }
  # 销毁所有待删除节点
  for (auto node : nodes_to_delete) {
    node->destroy();
  }
  # 执行死代码消除优化
  EliminateDeadCode(g);
}

// 保留模块中特定方法的参数生成，只保留返回 None 的功能
void keepOnlyPackedParamsGeneration(Module& m, const std::string& method_name) {
  // 获取指定方法的计算图
  auto g = m.get_method(method_name).graph();
  // 获取方法的函数对象
  Function& function = m.get_method(method_name).function();
  // 获取方法的调用规范（schema）
  const auto& schema = function.getSchema();
  // 克隆方法的调用规范，但只包含返回类型为空的返回值
  auto new_schema = schema.cloneWithReturns({Argument("", NoneType::get())});
  // 删除计算图的所有输出
  for (size_t i = 0, output_size = g->outputs().size(); i < output_size; i++) {
    g->eraseOutput(i);
  }
  // 创建一个返回 None 的节点
  Node* none_node = g->createNone();
  // 注册这个节点作为计算图的输出
  g->registerOutput(none_node->output());
  // 将这个节点插入到返回节点之前
  none_node->insertBefore(g->return_node());
  // 设置方法的新调用规范
  function.setSchema(std::move(new_schema));
  // 删除死代码
  EliminateDeadCode(g);
}

} // namespace

// 对图中的量化模式进行融合
void QuantFusion(std::shared_ptr<Graph>& graph, QuantType quant_type) {
  // 初始化量化模式信息的向量
  std::vector<QuantFusionInfo> patterns;
  // 根据量化类型选择合适的量化融合模式
  if (quant_type == QuantType::DYNAMIC) {
    patterns = dynamic_quant_fusion_pattern_and_replacements();
    // 获取不包含动态激活量化的线性量化融合模式
    std::vector<QuantFusionInfo> patterns_wo_dynamic_activation_quant =
        dynamic_quantized_linear_pattern_and_replacements();
    // 将这些模式插入到总模式列表中
    patterns.insert(
        patterns.end(),
        patterns_wo_dynamic_activation_quant.begin(),
        patterns_wo_dynamic_activation_quant.end());
  } else {
    // 获取普通量化融合模式
    patterns = quant_fusion_pattern_and_replacements();
  }
  // 对每个融合模式进行重写注册，并在图上运行重写
  for (const auto& info : patterns) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(info.pattern, info.replacement);
    rewriter.runOnGraph(graph, info.filters);
  }
}

// 向图中插入预打包和解包操作
void InsertPrepackUnpack(std::shared_ptr<Graph>& graph) {
  // 向线性层插入预打包和解包操作
  insertPrepackUnpackForLinear(graph);
  // 向卷积层插入预打包和解包操作
  insertPrepackUnpackForConv(graph);
}

// 对模块中的每个方法插入预打包和解包操作
void InsertPrepackUnpack(Module& module) {
  // 遍历模块中的每个方法
  for (auto& method : module.get_methods()) {
    // 获取方法的计算图
    auto graph = method.graph();
    // 向计算图中插入预打包和解包操作
    InsertPrepackUnpack(graph);
  }
  // 遍历模块的每个子模块
  for (Module m : module.children()) {
    // 递归地向子模块中插入预打包和解包操作
    InsertPrepackUnpack(m);
  }
}

// 折叠量化预打包操作
void FoldQuantizedPrepackingOps(Module& module) {
  // 定义用于过滤节点的函数
  auto filter_fn = [](const Node* n) -> bool {
    // 判断节点是否为量化预打包操作
    return (
        n->kind() == Symbol::fromQualString("quantized::linear_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv1d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv2d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv3d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv_transpose1d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv_transpose2d_prepack"));
  };
  // 折叠模块中的量化预打包操作
  PrePackingOpsFolder(module, filter_fn, "quantized");
}

// 注册预打包参数
static std::unordered_set<std::string> RegisterPrePackingParams(
    Module& module,
    const std::string& method_name) {
  // 定义用于过滤节点的函数
  auto filter_fn = [](const Node* n) -> bool {
    // 根据节点是否为预打包操作来判断是否过滤
    // (此处缺失部分代码，应继续编写)
    # 检查节点 n 的类型是否为 quantized::linear_prepack、quantized::conv1d_prepack、quantized::conv2d_prepack、quantized::conv3d_prepack、
    # quantized::conv_transpose1d_prepack 或 quantized::conv_transpose2d_prepack 中的一种
    return (
        n->kind() == Symbol::fromQualString("quantized::linear_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv1d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv2d_prepack") ||
        n->kind() == Symbol::fromQualString("quantized::conv3d_prepack") ||
        n->kind() ==
            Symbol::fromQualString("quantized::conv_transpose1d_prepack") ||
        n->kind() ==
            Symbol::fromQualString("quantized::conv_transpose2d_prepack"));
  };

  # 调用 RegisterPrePackParams 函数注册预打包参数
  return RegisterPrePackParams(module, method_name, filter_fn, "");
}

Module Finalize(
    Module& module,
    QuantType quant_type,
    const std::vector<std::string>& preserved_attrs) {
  // 遍历模块的方法，清除各个方法对应图的性能分析信息
  for (auto func : module.type()->methods()) {
    ClearProfilingInformation(toGraphFunction(*func).graph());
  }

  // 获取模块中名为 "forward" 的方法对应的图
  auto graph = module.get_method("forward").graph();
  // 向图中插入预打包和解包操作
  InsertPrepackUnpack(graph);
  // 打印当前图的状态，用于调试，显示在量化融合操作前的状态
  GRAPH_DUMP("Before QuantFusion:", graph);
  // 对图执行量化融合操作，根据指定的量化类型
  QuantFusion(graph, quant_type);
  // 冻结模块中的操作，保留指定的属性
  auto frozen = freeze_module(module, preserved_attrs);
  // 折叠量化预打包操作
  FoldQuantizedPrepackingOps(frozen);
  // 返回冻结后的模块
  return frozen;
}

Module FinalizeOnDevicePTQ(
    Module& module,
    QuantType quant_type,
    const std::string& method_name) {
  // 遍历模块的方法，清除各个方法对应图的性能分析信息
  for (auto func : module.type()->methods()) {
    // 清除性能分析信息
    ClearProfilingInformation(toGraphFunction(*func).graph());
  }
  ClearProfilingInformation(toGraphFunction(*func).graph());
  // 清除与性能分析相关的信息，针对函数对象的图形表示

  const std::string kQuantizeString = "quantize_";
  // 定义用于匹配的量化操作的字符串前缀
  const auto matched_pos = method_name.find(kQuantizeString);
  // 在方法名中查找量化操作字符串的位置
  const auto end_pos = matched_pos + kQuantizeString.length();
  // 计算量化操作字符串的结束位置
  const std::string orig_method_name = method_name.substr(end_pos);
  // 提取原始方法名，去除量化操作字符串前缀
  TORCH_CHECK(
      matched_pos == 0,
      // 断言：量化操作只能添加到以"quantize_"开头的方法名中
      "Quantized ops can only be added to quantize_",
      orig_method_name,
      ". Please make sure to run quant/dequant nodes insertion step for on-device PTQ.");

  const std::string quantized_method_name = "quantized_" + orig_method_name;
  // 构造量化后的方法名
  auto graph = module.get_method(method_name).graph();
  // 获取原始方法对应的图形表示

  // 进行一些AOT（Ahead of Time）优化
  // CSE（公共子表达式消除）似乎是必需的，否则在某些实验中
  // 序列化模型将不正确，即无法反序列化
  EliminateCommonSubexpression(graph);
  // 消除公共子表达式
  EliminateDeadCode(graph);
  // 消除死代码
  PeepholeOptimize(graph);
  // 窥孔优化
  ConstantPropagation(graph);
  // 常量传播
  UnrollConstantLoops(graph);
  // 展开常量循环
  ConstantPooling(graph);
  // 常量池化

  InsertPrepackUnpack(graph);
  // 插入预打包和解包操作到图中
  GRAPH_DUMP("Before QuantFusion:", graph);
  // 输出图形表示以便进行量化融合前的检查
  QuantFusion(graph, quant_type);
  // 执行量化融合操作
  auto packed_param_attr_names = RegisterPrePackingParams(module, method_name);
  // 注册预打包参数

  GRAPH_DUMP("After QuantFusion + packed param registration:", graph);
  // 输出图形表示以便进行量化融合后和预打包参数注册后的检查

  // 现在我们做到了：
  // 1. 插入了量化后的权重打包参数
  // 2. 将打包参数插入到模块中
  // 3. 插入了量化操作
  // 接下来需要做的事情是：
  // 1. 在量化前向传播中复制此方法
  // 2. 删除由量化前向传播重置的fp权重的SetAttr
  // 3. 移除SetAttr节点，后续优化将去除产生打包参数的节点
  cloneMethod(module, method_name, quantized_method_name);
  // 克隆方法，生成量化后的方法

  // 移除量化后方法中的打包参数插入和fp权重的SetAttr
  removePackedParamInsertionAndFPWeightsSetAttr(
      quantized_graph, packed_param_attr_names);

  // 移除打包参数并不足够，因为这不会对观察器节点的getatts和callmethods做DCE
  // 因为callmethods具有副作用
  removeObserverCallMethods(quantized_graph);

  // 这一步骤移除了图中的返回输出和随后的DCE将移除所有操作
  // 在此之后，唯一剩下的应该是打包参数
  keepOnlyPackedParamsGeneration(module, method_name);

  // 返回经过量化处理后的模块
  return module;
}

} // namespace jit
} // namespace torch
```