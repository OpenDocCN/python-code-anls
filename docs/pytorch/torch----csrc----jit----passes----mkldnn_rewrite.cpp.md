# `.\pytorch\torch\csrc\jit\passes\mkldnn_rewrite.cpp`

```py
// 包含 ATen 库的配置和头文件
#include <ATen/Config.h>
#include <ATen/code_template.h>
// 包含 Torch JIT 的 IR 相关头文件
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
// 包含 Torch JIT 的优化 passes 头文件
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
// 包含 Torch JIT 的 MKLDNN 重写头文件
#include <torch/csrc/jit/passes/mkldnn_rewrite.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

// Torch 命名空间开始
namespace torch {
namespace jit {

// 如果 MKLDNN 可用
#if AT_MKLDNN_ENABLED()

// 获取节点 n 的第 idx 个输入张量的大小信息
static c10::VaryingShape<int64_t> getSizesOf(Node* n, size_t idx) {
  // 获取张量类型信息
  auto tt = n->input(idx)->type()->cast<TensorType>();
  // 返回张量的大小信息
  return tt->sizes();
}

// 为节点 n 插入预打包的卷积操作
static void insertPrePackedConvOpForNode(Node* n) {
  // 定义输入和权重的位置
  constexpr int POS_INPUT = 0;
  constexpr int POS_WEIGHT = 1;

  // 检查输入张量是否是 ChannelsLast 连续的
  if (!tensorexpr::isContiguous(
          n->input(POS_INPUT), at::MemoryFormat::ChannelsLast)) {
    // 输出调试信息并返回
    GRAPH_DEBUG(
        "insertPrePackedConvOpForNode: input is not ChannelsLast contiguous");
    return;
  }

  // 检查权重张量是否是 ChannelsLast 连续的
  if (!tensorexpr::isContiguous(
          n->input(POS_WEIGHT), at::MemoryFormat::ChannelsLast)) {
    // 输出调试信息并返回
    GRAPH_DEBUG(
        "insertPrePackedConvOpForNode: weight is not ChannelsLast contiguous");
    return;
  }

  // 对深度卷积留给 NNC 处理
  if (tensorexpr::conv2dIsSupportedJit(n)) {
    // 输出调试信息并返回
    GRAPH_DEBUG("insertPrePackedConvOpForNode: leave depthwise conv2d to NNC");
    return;
  }

  // 设置插入点为当前节点 n
  WithInsertPoint guard(n);
  // 获取当前节点所属图
  auto graph = n->owningGraph();

  // 获取输入张量的大小信息
  auto input_sizes = getSizesOf(n, POS_INPUT);
  // 创建表示输入大小的常量值
  IValue input_size_value(*input_sizes.concrete_sizes());
  auto input_size = graph->insertConstant(input_size_value);

  // 创建 MKLDNN 预打包的卷积节点
  auto prepack_node = graph->create(
      Symbol::fromQualString("mkldnn_prepacked::conv2d_prepack"), 1);

  // 跳过输入值，将其它输入添加到预打包节点
  for (const auto i : c10::irange(1, n->inputs().size())) {
    Value* v = n->input(i);
    prepack_node->addInput(v);
  }
  // 添加输入大小和属性值到预打包节点
  prepack_node->addInput(input_size);
  auto attr = graph->insertConstant(IValue("none"));
  prepack_node->addInput(attr);
  // 设置预打包节点的输出类型
  prepack_node->output()->setType(
      getCustomClass("__torch__.torch.classes.mkldnn.ConvOpContext"));
  // 在图中插入预打包节点
  graph->insertNode(prepack_node);

  // 创建 MKLDNN 预打包卷积运行节点
  auto prepack_conv = graph->insertNode(
      graph->create(Symbol::fromQualString("mkldnn_prepacked::conv2d_run"), 1));
  // 添加原始输入和预打包节点的输出到运行节点
  prepack_conv->addInput(n->input(0));
  prepack_conv->addInput(prepack_node->output());
  // 设置运行节点的输出类型
  prepack_conv->output()->setType(n->output()->type()->cast<TensorType>());

  // 替换原始节点的输出使用为运行节点的输出
  n->output()->replaceAllUsesWith(prepack_conv->output());
}

// 检查节点是否为 CPU 上的张量类型
static bool isTensorTypeCPU(Node* node) {
  // 遍历节点的所有输入
  for (const auto& input : node->inputs()) {
    // 获取输入的张量类型
    auto type = input->type()->cast<TensorType>();
    if (!type) {
      continue;
    }
    // 获取设备信息并检查是否为 CPU
    auto device = type->device();
    if (!device) {
      return false;
    }
    if (!device->is_cpu()) {
      return false;
    }
  }
  // 若所有输入都是 CPU 张量，则返回 true
  return true;
}

// 递归地为块 b 中的每个节点插入预打包的卷积操作
static void insertPrePackedConvOp(Block* b) {
  // 遍历块 b 中的所有节点
  for (Node* n : b->nodes()) {
    // 若节点包含子块，则递归地插入预打包的卷积操作
    for (Block* b : n->blocks()) {
      insertPrePackedConvOp(b);
    }
    // 对当前节点插入预打包的卷积操作
    insertPrePackedConvOpForNode(n);
  }
}
    if (n->kind() == aten::conv2d) {
      // 检查节点 n 的操作类型是否为 conv2d
      if (isTensorTypeCPU(n)) {
        // 检查节点 n 是否在 CPU 上执行
        insertPrePackedConvOpForNode(n);
        // 为节点 n 插入预打包的卷积操作
      }
    }
  }
  // 对基本块 b 执行死代码消除优化
  EliminateDeadCode(b);
}

static void insertMkldnnPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  // 在给定图形块中插入预打包的二维卷积操作
  insertPrePackedConvOp(graph->block());
}

static void insertMkldnnPrePackedOps(std::shared_ptr<Graph>& graph) {
  // 在给定图形中插入MKLDNN预打包的操作
  insertMkldnnPrePackedConv2dOp(graph);
}

static void FuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  // 定义卷积操作的模板字符串
  auto conv_op_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %input_size:int[], %dummy_attr:str):
        %packed_weight_bias = mkldnn_prepacked::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %input_size, %dummy_attr)
        %conv2d_res = mkldnn_prepacked::conv2d_run(%input, %packed_weight_bias)
        %res = aten::${op}(%conv2d_res)
        return (%res))");

  // 定义融合ReLU与预打包操作的模板字符串
  auto conv_op_fused_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %input_size:int[], %dummy_attr:str):
        %attr: str = prim::Constant[value="${op_attr}"]()
        %packed_weight_bias : __torch__.torch.classes.mkldnn.ConvOpContext = mkldnn_prepacked::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %input_size, %attr)
        %res = mkldnn_prepacked::conv2d_run(%input, %packed_weight_bias)
        return (%res))");

  // 遍历MKLDNN融合重写映射表
  for (auto const& it : mkldnn::fusion_rewrite_map) {
    std::string op = it.first;
    if (op == std::string("none")) {
      continue;
    }

    // 设置模板环境变量
    at::jit::TemplateEnv env;
    env.s("op", op);

    at::jit::TemplateEnv env_fused;
    env_fused.s("op_attr", op);

    // 注册重写模式
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(
        conv_op_rstring.format(env), conv_op_fused_rstring.format(env_fused));

    // 获取过滤器并在图形上运行重写器
    auto filters = it.second;
    rewriter.runOnGraph(graph, filters);
  }
}

static void PrePackingOpsFolder(Block* b) {
  // 定义可折叠操作的判断条件函数
  auto is_foldable_op = [](const Node* n) -> bool {
    return (
        n->kind() ==
        Symbol::fromQualString("mkldnn_prepacked::conv2d_prepack"));
  };

  // 存储待删除的节点集合
  std::unordered_set<Node*> nodes_to_delete;

  // 遍历块中的节点
  for (Node* n : b->nodes()) {
    // 递归调用以折叠操作
    for (Block* block : n->blocks()) {
      PrePackingOpsFolder(block);
    }
    // 如果节点是可折叠操作
    if (is_foldable_op(n)) {
      // 如果节点的输入是常量，则运行节点并获取输出
      auto optional_outputs = torch::jit::runNodeIfInputsAreConstant(n);
      if (optional_outputs) {
        auto outputs = optional_outputs.value();
        // 检查输出的数量是否为1
        TORCH_CHECK(outputs.size() == 1, "Prepack ops have single output");
        // 替换预打包操作的值
        Value* prepack_op_value = n->output(0);
        auto graph = n->owningGraph();
        WithInsertPoint ins(prepack_op_value->node());
        auto weak_class_obj =
            outputs[0].toObject()->copy_to_weak_compilation_ref();
        Value* packed_weight = graph->insertConstant(weak_class_obj)
                                   ->setType(n->output(0)->type());
        prepack_op_value->replaceAllUsesWith(packed_weight);
        nodes_to_delete.insert(n);
      }
    }
  }
  // 遍历要删除的节点列表，依次处理每个节点
  for (auto n : nodes_to_delete) {
    // 移除当前节点的所有输入连接
    n->removeAllInputs();
  }
  // 再次遍历要删除的节点列表，依次处理每个节点
  for (auto n : nodes_to_delete) {
    // 销毁当前节点对象
    n->destroy();
  }
// 折叠前置打包操作，传入图的块
static void FoldPrePackingOps(std::shared_ptr<Graph>& graph) {
  // 调用函数 PrePackingOpsFolder，对图的块进行操作
  PrePackingOpsFolder(graph->block());
}

// 融合卷积和逐元素操作，传入图
void FuseConvWithEltwise(std::shared_ptr<Graph>& graph) {
  // 输出调试信息，标记融合卷积和逐元素操作开始前的图状态
  GRAPH_DEBUG(
      "Before insertMkldnnPrePackedOps. Beginning of FuseConvWithEltwise\n",
      *graph);
  // 插入 MKLDNN 预打包操作
  insertMkldnnPrePackedOps(graph);
  // 输出调试信息，标记插入 MKLDNN 预打包操作后，融合 ReLU 和打包操作前的图状态
  GRAPH_DEBUG(
      "After insertMkldnnPrePackedOps, before FuseReluWithPackedOps\n", *graph);
  // 融合 ReLU 和打包操作
  FuseReluWithPackedOps(graph);
  // 输出调试信息，标记融合 ReLU 和打包操作后，折叠前置打包操作前的图状态
  GRAPH_DEBUG(
      "After FuseReluWithPackedOps, before FoldPrePackingOps\n", *graph);
  // 折叠前置打包操作
  FoldPrePackingOps(graph);
  // 输出调试信息，标记折叠前置打包操作后的图状态
  GRAPH_DEBUG("After FoldPrePackingOps. End of FuseConvWithEltwise\n", *graph);
}

#else

// 如果 MKLDNN 未启用，则输出相应信息
void FuseConvWithEltwise(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG("MKLDNN Not enabled");
}

#endif // AT_MKLDNN_ENABLED()

} // namespace jit
} // namespace torch
```