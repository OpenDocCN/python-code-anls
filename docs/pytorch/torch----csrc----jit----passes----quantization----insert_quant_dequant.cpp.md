# `.\pytorch\torch\csrc\jit\passes\quantization\insert_quant_dequant.cpp`

```
// 引入Torch的量化插入相关头文件
#include <torch/csrc/jit/passes/quantization/insert_quant_dequant.h>

// 引入C10库中的量化方案和工具类
#include <c10/core/QScheme.h>
#include <c10/util/irange.h>

// 引入Torch的模型前端相关头文件
#include <torch/csrc/jit/frontend/schema_matching.h>

// 引入Torch的IR子图匹配器
#include <torch/csrc/jit/ir/subgraph_matcher.h>

// 引入Torch的JIT日志记录器
#include <torch/csrc/jit/jit_log.h>

// 引入Torch的常量传播相关头文件
#include <torch/csrc/jit/passes/constant_propagation.h>

// 引入Torch的线性融合相关头文件
#include <torch/csrc/jit/passes/fuse_linear.h>

// 引入Torch的图重写辅助函数
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>

// 引入Torch的内联函数相关头文件
#include <torch/csrc/jit/passes/inliner.h>

// 引入Torch的量化辅助函数
#include <torch/csrc/jit/passes/quantization/helper.h>

// 引入Torch的子图重写相关头文件
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

// 引入标准库中的堆栈和实用工具
#include <stack>
#include <utility>

// 定义torch命名空间和jit子命名空间
namespace torch {
namespace jit {

// 匿名命名空间用于局部函数和类型别名
namespace {
// 使用图重写辅助函数命名空间中的模式信息类型
using graph_rewrite_helper::PatternInfo;

// 定义动态量化操作的元组类型，用于激活：选择量化参数、量化、反量化
using DynamicQuantOps = std::tuple<Node*, Node*, Node*>;

// 定义标量类型字符串常量
std::string kScalarType = "_scalar_type";

// 定义量化操作参数结构体
struct QuantOpParams {
  c10::QScheme qscheme{c10::kPerTensorAffine};
  std::vector<Value*> qparams;
  
  // 仅用于模板化insertQuantizationOps函数和部分代码的重用
  std::string back() const {
    return "AttributeDoesNotExist";
  }
};

// 转换为仿射量化方案
c10::QScheme toAffine(c10::QScheme qscheme) {
  switch (qscheme) {
    case c10::kPerTensorAffine:
    case c10::kPerTensorSymmetric:
      return c10::kPerTensorAffine;
    case c10::kPerChannelAffine:
    case c10::kPerChannelSymmetric:
      return c10::kPerChannelAffine;
    default:
      return qscheme;
  }
}

// 判断是否为通道量化
bool isPerChannel(at::QScheme qscheme) {
  return qscheme == c10::kPerChannelAffine ||
      qscheme == c10::kPerChannelSymmetric;
}

// 通过调用方法图检查值是否为权重
bool isWeight(Module& module, Value* v) {
  if (isWeight(v)) {
    return true;
  }
  std::optional<bool> result;
  auto* self = v->owningGraph()->inputs()[0];
  for (const Use& u : v->uses()) {
    Node* n = u.user;
    if (n->kind() == prim::CallMethod) {
      auto m_opt = getInvokedModuleOpt(module, n, self);
      if (!m_opt.has_value()) {
        return false;
      }
      auto m = *m_opt;
      auto g = m.get_method(n->s(attr::name)).graph();
      auto call_method_result = isWeight(m, g->inputs()[u.offset]);
      if (result.has_value()) {
        // 检查确保图中所有的调用方法产生相同的输出
        TORCH_CHECK(
            call_method_result == result.value(),
            "Expected all CallMethods to use either weight "
            "or non-weight value.",
            v->debugName());
      } else {
        result = call_method_result;
      }
    }
  }
  return result.has_value() ? result.value() : false;
}
// 在图中插入选择量化参数节点，返回该节点指针
Node* insertChooseQParams(Graph* graph, Value* original_val) {
  // 定义选择量化参数函数名
  std::string choose_qparams_func = "_choose_qparams_per_tensor";
  // 设置减少范围参数默认为 true，因为 qnnpack 后端忽略此参数
  bool reduce_range_param = true;
  // 插入常量节点，表示是否减少范围
  auto reduce_range = graph->insertConstant(reduce_range_param);
  // 创建 choose_qparams_per_tensor 节点，有两个输出 (scale, zero_point)
  Node* choose_qparams = graph->create(
      at::Symbol::aten(choose_qparams_func),
      {original_val, reduce_range},
      /* num_outputs = */ 2);
  // 设置第一个输出的调试名称为原始值名称 + ".scale"
  choose_qparams->output(0)->setDebugName(original_val->debugName() + ".scale");
  // 设置第一个输出的类型为浮点类型
  choose_qparams->output(0)->setType(FloatType::get());
  // 设置第二个输出的调试名称为原始值名称 + ".zero_point"
  choose_qparams->output(1)->setDebugName(
      original_val->debugName() + ".zero_point");
  // 设置第二个输出的类型为整数类型
  choose_qparams->output(1)->setType(IntType::get());
  // 在图中插入 choose_qparams 节点
  graph->insertNode(choose_qparams);
  // 返回 choose_qparams 节点指针
  return choose_qparams;
}

// 在图中插入量化节点，返回该节点指针
Node* insertQuant(
    Graph* graph,
    const std::vector<Value*>& inputs,
    NodeKind quant_kind,
    const std::string& debugName) {
  // 创建指定类型的节点，使用给定的输入值
  Node* quant = graph->create(quant_kind, inputs);
  // 设置输出值的调试名称
  quant->output()->setDebugName(debugName);
  // 在图中插入 quant 节点
  graph->insertNode(quant);
  // 返回 quant 节点指针
  return quant;
}

// 在图中插入反量化节点，返回该节点指针
Node* insertDeQuant(
    Graph* graph,
    Value* quantized_val,
    Value* original_val,
    size_t id = 0) {
  // 创建 dequantize 节点，使用量化值作为输入
  Node* dequant = graph->create(Symbol::aten("dequantize"), {quantized_val});
  // 设置输出值的调试名称为原始值名称 + ".dequant." + id
  dequant->output()
      ->setDebugName(
          original_val->debugName() + ".dequant." + std::to_string(id))
      // 设置输出值的类型为原始值的类型
      ->setType(original_val->type());
  // 在图中插入 dequant 节点
  graph->insertNode(dequant);
  // 返回 dequant 节点指针
  return dequant;
}

// 为原始值的所有使用插入反量化节点，并返回输出值的向量
std::vector<Value*> insertDeQuantForAllUse(
    Graph* graph,
    Value* quantized_val,
    Value* original_val) {
  // 复制使用到向量中，因为 value->uses() 返回引用，改变图也会改变使用列表
  const std::vector<Use> uses = original_val->uses();
  std::vector<Value*> outputs;
  // 遍历所有使用
  for (const auto i : c10::irange(uses.size())) {
    auto* user = uses[i].user;
    // 设置插入点为使用节点之前，确保使用节点和反量化节点位于同一块中，以便量化融合发生
    WithInsertPoint ins(user);
    // 插入反量化节点，传入量化值、原始值和序号
    Node* dequant = insertDeQuant(graph, quantized_val, original_val, i);
    // 替换使用节点的输入为反量化节点的输出
    user->replaceInput(uses[i].offset, dequant->output());
    // 将反量化节点的输出值添加到输出向量中
    outputs.push_back(dequant->output());
  }
  // 返回所有输出值的向量
  return outputs;
}

// 在图中插入量化参数节点，返回该节点指针
Node* insertQParam(
    Graph* graph,
    Value* quantized_input,
    NodeKind node_kind,
    const TypePtr& output_type,
    const std::string& param_name) {
  // 创建指定类型的节点，使用量化输入作为输入
  Node* qparam = graph->create(node_kind, {quantized_input});
  // 设置输出值的调试名称为量化输入名称 + "." + 参数名称
  qparam->output()
      ->setDebugName(quantized_input->debugName() + "." + param_name)
      // 设置输出值的类型为指定的输出类型
      ->setType(output_type);
  // 在图中插入 qparam 节点
  graph->insertNode(qparam);
  // 返回 qparam 节点指针
  return qparam;
}
// 将标量值插入到张量中
Node* insertScalarToTensor(Graph* graph, Value* scalar_value) {
  // 获取标量值所属的节点
  Node* n = scalar_value->node();
  // 在标量值节点的下一个插入点上操作
  WithInsertPoint ins(n->next());
  // 插入一个常量节点，表示浮点数类型
  Value* float_scalar_type = graph->insertConstant(IValue(c10::kFloat));
  // 插入一个空值常量节点
  Value* none = graph->insertConstant(IValue());
  // 创建一个 ATen 操作节点，将标量值转换为张量
  Node* tensor_node = graph->create(
      Symbol::aten("scalar_tensor"),
      {scalar_value, float_scalar_type, none, none, none});
  // 获取张量节点的输出值
  Value* tensor_output = tensor_node->output();
  // 设置张量输出值的调试名称
  tensor_output->setDebugName(scalar_value->debugName() + ".tensor");
  // 在计算图中插入张量节点
  graph->insertNode(tensor_node);
  // 将所有使用标量值的地方替换为张量输出值
  scalar_value->replaceAllUsesAfterNodeWith(tensor_node, tensor_output);
  // 返回创建的张量节点
  return tensor_node;
}

// 将张量插入到项目中
Node* insertItem(Graph* graph, Value* tensor, const TypePtr& output_type) {
  // 在张量节点的下一个插入点上操作
  WithInsertPoint ins(tensor->node()->next());
  // 创建一个 ATen 操作节点，将张量转换为项目（标量）
  Node* n = graph->create(Symbol::aten("item"), {tensor});
  // 获取项目节点的输出值（标量）
  Value* scalar = n->output();
  // 设置标量输出值的调试名称和类型
  scalar->setDebugName(tensor->debugName() + ".scalar")->setType(output_type);
  // 在计算图中插入项目节点
  graph->insertNode(n);
  // 返回创建的项目节点
  return n;
}

// 插入选择量化参数和去量化操作节点
DynamicQuantOps insertChooseQParamQuantDequant(
    Graph* graph,
    Value* original_val,
    Value* dtype,
    NodeKind quant_kind) {
  // 插入选择量化参数操作节点
  Node* choose_qparams = insertChooseQParams(graph, original_val);
  // 创建一个包含原始值及选择量化参数输出的输入值向量
  std::vector<Value*> quant_inputs = {original_val};
  for (auto& out : choose_qparams->outputs()) {
    quant_inputs.push_back(out);
  }
  // 添加数据类型作为输入
  quant_inputs.push_back(dtype);
  // 插入量化操作节点
  Node* quant = insertQuant(
      graph, quant_inputs, quant_kind, original_val->debugName() + ".quant");
  // 插入去量化操作节点
  Node* dequant = insertDeQuant(graph, quant->output(), original_val);
  // 返回选择量化参数、量化、去量化操作节点的元组
  return std::make_tuple(choose_qparams, quant, dequant);
}

// 插入 FP16 类型转换操作节点
Node* insertFP16CastOps(Graph* graph, Value* observer_out) {
  // 如果权重值超出 FP16 范围 [5.96e-8, 65504]，则将其饱和到此范围的最小值和最大值
  Node* saturated_weight =
      graph->create(Symbol::aten("_saturate_weight_to_fp16"), {observer_out});
  // 在计算图中插入饱和权重操作节点
  graph->insertNode(saturated_weight);
  // 执行计算图的静态分析
  graph->lint();
  // 返回创建的饱和权重节点
  return saturated_weight;
}

// 查找值 `v` 的观察器并返回观察器的名称
std::optional<std::string> findObserverName(Value* v) {
  // 获取值 `v` 所属的节点
  Node* n = v->node();
  // 如果节点类型为 prim::CallMethod 并且方法名为 "forward"
  if (n->kind() == prim::CallMethod && n->s(attr::name) == "forward") {
    // 获取模块实例作为输入的第一个节点
    auto module_instance = n->inputs().at(0);
    // 如果模块实例节点的类型为 prim::GetAttr 并且名称包含 "_observer_"
    if (module_instance->node()->kind() == prim::GetAttr &&
        module_instance->node()->s(attr::name).find("_observer_") !=
            std::string::npos) {
      // 返回观察器的名称
      return module_instance->node()->s(attr::name);
    }
  }
  // 如果未找到符合条件的观察器，则返回空值
  return c10::nullopt;
}

// 判断是否为占位符观察器
bool isPlaceholderObserver(Value* observer) {
  // 获取观察器的模块名称（如果存在）
  if (getModuleName(observer).has_value()) {
    auto name = getModuleName(observer).value();
    // 如果模块名称中包含 "PlaceholderObserver" 字符串
    // （不考虑位置，只要包含即可）
    # 检查字符串 `name` 中是否包含子字符串 "PlaceholderObserver"
    if (name.find("PlaceholderObserver") != std::string::npos) {
      # 如果包含，返回 true
      return true;
    }
  }
  # 如果未在循环中返回 true，则返回 false
  return false;
}

// 获取观察器的数据类型
at::ScalarType getObserverDtype(Module& module, Value* v) {
  // 查找观察器的名称
  auto observer_name = findObserverName(v);
  if (observer_name.has_value()) {
    // 获取观察器模块
    auto observer_module = module.attr(observer_name.value()).toModule();
    // 获取观察器模块中的数据类型
    at::ScalarType scalar_type = observer_module.attr("dtype").toScalarType();
    return scalar_type;
  }
  // 如果没有找到观察器名称，返回未定义的数据类型
  return at::ScalarType::Undefined;
}

// 获取嵌入包操作的观察器名称
std::optional<std::string> getEmbeddingBagObsName(
    script::Module& module,
    Node* n) {
  // 获取节点的输出值
  Value* v = n->output();
  // 获取观察器节点的输入值
  auto observer = n->input(0);
  // 根据输出值查找观察器名称，并获取观察器模块
  auto observer_module = module.attr(findObserverName(v).value()).toModule();
  // 如果观察器模块具有 "custom_op" 属性
  if (observer_module.hasattr("custom_op")) {
    // 获取自定义操作名称
    auto op_name = observer_module.attr("custom_op").toStringRef();
    // 如果观察器是占位符观察器，则返回操作名称；否则返回空字符串
    return isPlaceholderObserver(observer) ? std::move(op_name) : "";
  }
  // 如果没有 "custom_op" 属性，返回空值
  return c10::nullopt;
}

// 判断是否为嵌入包操作
bool isEmbeddingBagOp(
    Node* observer,
    std::optional<std::string> embedding_bag_name) {
  // 如果嵌入包名称存在，并且观察器名称包含 "embedding_bag_"，返回 true
  return embedding_bag_name &&
      embedding_bag_name.value().find("embedding_bag_") != std::string::npos;
}

// 插入量化和去量化节点到图中的模板函数
template <typename T>
Node* insertQuantDequantNodes(
    Value* self,
    Node* observer,
    T& qparams,
    const std::string& quantize_func);

// 插入量化和去量化节点到图中的特化函数，处理字符串向量的量化参数
template <>
Node* insertQuantDequantNodes<std::vector<std::string>>(
    Value* self,
    Node* observer,
    std::vector<std::string>& qparam_names,
    const std::string& quantize_func) {
  // 获取观察器节点所属的图
  Graph* g = observer->owningGraph();
  // 获取观察器节点的输出值
  Value* observer_out = observer->output();
  // 获取观察器节点的原始输入值
  Value* original_val = observer->input(1);
  std::vector<Value*> inputs = {observer_out};
  // 插入用于量化参数的 GetAttr 节点
  for (const auto& qparam_name : qparam_names) {
    inputs.push_back(g->insertGetAttr(self, qparam_name));
  }
  // 插入量化节点
  Node* quant = insertQuant(
      g,
      inputs,
      at::Symbol::aten(quantize_func),
      original_val->debugName() + ".quant");
  // 插入去量化节点
  Node* dequant = insertDeQuant(g, quant->output(), original_val);
  return dequant;
}

// 插入嵌入包操作节点到图中
Node* insertEmbeddingBagOps(Node* observer, const std::string& op_name) {
  // 获取观察器节点所属的图
  Graph* g = observer->owningGraph();
  // 获取观察器节点的输出值
  auto observer_out = observer->output();

  std::string prepack_fn, quant_fn;
  std::vector<Value*> prepack_inputs = {observer_out};
  // 根据操作名称选择预打包和量化函数
  if (op_name == "embedding_bag_4bit") {
    bool optimized_qparams = false;
    constexpr int NBINS = 200;
    constexpr float RATIO = 0.16;
    // 插入常量节点用于优化的量化参数
    Value* optimized_qparams_false = g->insertConstant(optimized_qparams);
    Value* nbins_200 = g->insertConstant(NBINS);
    Value* ratio_0_16 = g->insertConstant(RATIO);
    // 设置预打包和量化函数名称
    prepack_fn = "quantized::embedding_bag_4bit_prepack";
    quant_fn = "quantized::embedding_bag_4bit_rowwise_offsets";
    prepack_inputs.push_back(optimized_qparams_false);
    prepack_inputs.push_back(nbins_200);
    prepack_inputs.push_back(ratio_0_16);
  } else if (op_name == "embedding_bag_byte") {
    // 设置预打包函数名称
    prepack_fn = "quantized::embedding_bag_byte_prepack";
    // ...
    // 这里省略了其他条件分支的处理，具体内容应根据代码的完整性添加
  }
    // 设置量化函数名称为 "quantized::embedding_bag_byte_rowwise_offsets"
    quant_fn = "quantized::embedding_bag_byte_rowwise_offsets";
    } else {
      // 如果不是预期的量化类型，抛出断言错误信息
      TORCH_INTERNAL_ASSERT(
          false,
          "Graph Mode Quantization currently supports 4-bit and 8-bit embedding bag quantization.");
    }
    
    // 获取权重观察器的使用情况
    std::vector<Use> uses = observer_out->uses();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Node* embedding_bag_float_op;
    // 预期权重观察器的输出将被 embedding_bag 操作符使用
    for (const Use& use : uses) {
      if (matchCallFuncToUse(use, "embedding_bag", 2) ||
          matchAtenFuncToUse(use, "embedding_bag", 0)) {
        // 找到使用 embedding_bag 函数的节点
        embedding_bag_float_op = use.user;
      }
    }
    
    // 插入预打包操作节点
    Node* prepack = g->create(Symbol::fromQualString(prepack_fn), prepack_inputs);
    g->insertNode(prepack);
    
    // 获取 embedding_bag 操作符的输入
    std::vector<Value*> embedding_bag_inputs =
        // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
        embedding_bag_float_op->inputs().vec();
    // 创建量化 embedding 操作的输入
    std::vector<Value*> qembedding_bag_inputs = {prepack->output()};
    const auto inputs_size = embedding_bag_float_op->inputs().size();
    const bool is_aten_op =
        embedding_bag_float_op->kind() == Symbol::aten("embedding_bag");
    // 创建并插入量化 embedding 操作节点
    Value* none = g->insertConstant(IValue());
    Value* zero = g->insertConstant(IValue(0));
    bool pruned_wt = false;
    auto pruned_const = g->insertConstant(pruned_wt);
    
    if (is_aten_op) {
      // 检查 FP aten::embedding_bag 操作符的输入数量是否为 9
      TORCH_CHECK(
          inputs_size == 9,
          "Expecting FP aten::embedding_bag operator to have 9 inputs");
      // 将 embedding_bag 操作的部分输入复制到量化 embedding 操作的输入中
      // 输入 0 是预打包操作的输出
      // 最后一个输入在考虑到 4-bit 情况下的额外输入后添加
      for (unsigned long i = 1; i < inputs_size - 2; ++i) {
        qembedding_bag_inputs.push_back(embedding_bag_inputs[i]);
      }
      // 在量化 embedding 操作中，第 5 个输入表示稀疏梯度，用于推断时表示修剪的权重，目前不支持修剪，因此设置为 0
      qembedding_bag_inputs[5] = pruned_const;
    } else {
      // 检查 F.embedding_bag 操作符的输入数量是否为 12
      TORCH_CHECK(
          inputs_size == 12,
          "Expecting F.embedding_bag operator to have 12 inputs");
      // 将 embedding_bag 操作的指定输入复制到量化 embedding 操作的输入中
      qembedding_bag_inputs.push_back(embedding_bag_inputs[1]); // indices
      qembedding_bag_inputs.push_back(embedding_bag_inputs[3]); // offsets
      qembedding_bag_inputs.push_back(
          embedding_bag_inputs[6]); // scale_grad_by_freq
      qembedding_bag_inputs.push_back(zero); // mode
      qembedding_bag_inputs.push_back(pruned_const); // pruned_weights
    qembedding_bag_inputs.push_back(
        embedding_bag_inputs[9]); // 将 embedding_bag_inputs 的第 10 个元素添加到 qembedding_bag_inputs 中作为 per_sample_weights
  }

  qembedding_bag_inputs.push_back(none); // 将 none 添加到 qembedding_bag_inputs 中作为 compressed_indices_mapping
  qembedding_bag_inputs.push_back(embedding_bag_inputs[inputs_size - 2]); // 将 embedding_bag_inputs 的倒数第 2 个元素添加到 qembedding_bag_inputs 中

  TORCH_CHECK(
      embedding_bag_inputs[inputs_size - 1]->mustBeNone(),
      "Expected aten::embedding_bag padding_idx input to be None"); // 检查 embedding_bag_inputs 的最后一个元素是否为 None，用于验证 padding_idx 输入应为 None

  Node* qembedding_bag =
      g->create(Symbol::fromQualString(quant_fn), qembedding_bag_inputs); // 在图 g 中创建一个新节点 qembedding_bag，节点类型由 quant_fn 字符串指定，输入为 qembedding_bag_inputs
  if (is_aten_op) {
    WithInsertPoint ins(embedding_bag_float_op); // 在 embedding_bag_float_op 的插入点上进行操作
    g->insertNode(qembedding_bag); // 将 qembedding_bag 节点插入到图 g 中
    // 验证除了索引 0 之外的输出在图中没有被使用
    for (const auto i :
         c10::irange(1, embedding_bag_float_op->outputs().size())) {
      TORCH_CHECK(
          !embedding_bag_float_op->output(i)->hasUses(),
          "Expected aten::embedding_bag to only have use for its first output."); // 检查 embedding_bag_float_op 的除第一个输出之外的输出是否没有被使用
    }
  } else {
    g->insertNode(qembedding_bag); // 将 qembedding_bag 节点插入到图 g 中
  }
  embedding_bag_float_op->output(0)->replaceAllUsesWith(
      qembedding_bag->output()); // 将 embedding_bag_float_op 的第一个输出替换为 qembedding_bag 的输出
  embedding_bag_float_op->removeAllInputs(); // 移除 embedding_bag_float_op 的所有输入
  embedding_bag_float_op->destroy(); // 销毁 embedding_bag_float_op 节点
  g->lint(); // 对图 g 进行静态分析
  return qembedding_bag; // 返回创建的 qembedding_bag 节点
// 结束函数模板的右大括号

template <typename T>
void insertQuantizationOps(
    Module& module,
    Value* self,
    Node* observer,
    bool is_per_channel,
    T& qparams,
    QuantType quant_type = QuantType::STATIC) {
  // 获取观察节点所属的计算图
  Graph* g = observer->owningGraph();
  // 获取观察节点的输出值
  Value* observer_out = observer->output();
  // 在观察节点的下一个节点之前插入新节点
  WithInsertPoint ins(observer_out->node()->next());

  // 用于存储量化函数名称的字符串变量
  std::string quantize_func;
  // 根据是否按通道量化选择量化函数名称
  if (is_per_channel) {
    quantize_func = "quantize_per_channel";
  } else {
    quantize_func = "quantize_per_tensor";
  }
  // 获取观察节点的输入值中的原始值
  Value* original_val = observer->input(1);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 用于量化、选择量化参数、反量化节点的临时指针
  Node *quant, *choose_qparams, *dequant;
  
  // 获取嵌入包操作的名称，用于特定情况下的量化处理
  auto embedding_bag_name = getEmbeddingBagObsName(module, observer);
  // 如果观察节点是嵌入包操作，并且是权重类型
  if (isEmbeddingBagOp(observer, embedding_bag_name)) {
    if (isWeight(module, observer_out)) {
      // 插入嵌入包操作的量化处理，并替换输出为原始值
      auto op_name = embedding_bag_name.value();
      Node* dequant = insertEmbeddingBagOps(observer, op_name);
      observer_out->replaceAllUsesWith(original_val);
      original_val->replaceAllUsesAfterNodeWith(dequant, dequant->output());
    } else {
      // 对于嵌入包操作的索引输入，不进行量化处理，但需要插入观察节点
      observer_out->replaceAllUsesWith(original_val);
    }
    return;
  }
  
  // 如果量化类型为动态量化
  if (quant_type == QuantType::DYNAMIC) {
    // 如果观察节点的数据类型为半精度浮点数
    if (getObserverDtype(module, observer_out) == at::ScalarType::Half) {
      dequant = insertFP16CastOps(g, observer_out);
    } else if (!isWeight(module, observer_out)) {
      // 获取观察节点输出的数据类型
      auto observer_dtype = getObserverDtype(module, observer_out);
      // 如果数据类型需要量化，插入选择量化参数、量化和反量化节点
      if (observer_dtype == at::ScalarType::QUInt8 ||
          observer_dtype == at::ScalarType::QInt8) {
        Value* dtype = g->insertGetAttr(self, qparams.back());
        std::tie(choose_qparams, quant, dequant) =
            insertChooseQParamQuantDequant(
                g, observer_out, dtype, at::Symbol::aten(quantize_func));
      } else {
        // 不需要量化的数据类型，例如 float32，移除观察节点的调用
        observer_out->replaceAllUsesWith(original_val);
        return;
      }
    } else {
      // 对于权重张量，插入量化和反量化节点
      dequant = insertQuantDequantNodes(self, observer, qparams, quantize_func);
    }
  } else { // 静态量化
    // 插入量化和反量化节点
    dequant = insertQuantDequantNodes(self, observer, qparams, quantize_func);
  }
  // 替换观察节点的输出为原始值
  observer_out->replaceAllUsesWith(original_val);
  // 将原始值的使用替换为反量化节点的输出
  original_val->replaceAllUsesAfterNodeWith(dequant, dequant->output());
  // 打印图中的插入节点信息
  GRAPH_DUMP("insert nodes:", original_val->owningGraph());
}
// 复制并重写选择量化参数、量化和去量化的操作，通过图形匹配找到模式并执行替换
void ReplicateChooseQParamsQuantDequant(std::shared_ptr<Graph>& graph) {
  // 解析动态量化模式信息并创建模式图
  const PatternInfo& dynamic_quant_pattern = PatternInfo::parse_from_str(R"(
    graph(%a, %reduce_range, %a_dtype):
        %a_scale : float, %a_zero_point : int = aten::_choose_qparams_per_tensor(%a, %reduce_range)
        %a_quant = aten::quantize_per_tensor(%a, %a_scale, %a_zero_point, %a_dtype)
        %a_dequant = aten::dequantize(%a_quant)
        return (%a_dequant) )");
  const Graph& dynamic_quant_graph = *dynamic_quant_pattern.pattern_graph;

  // 查找并获取模式匹配的结果
  const auto& matches = findPatternMatches(dynamic_quant_graph, *graph);
  if (matches.empty()) {
    return;
  }

  // 获取模式变量映射
  const auto& vmap = dynamic_quant_pattern.vmap;
  Value* dequant_val = vmap.at("a_dequant");
  Node* pattern_dequant = dequant_val->node();
  Value* quant_val = vmap.at("a_quant");
  Node* pattern_quant = quant_val->node();
  Value* choose_qparam_val = vmap.at("a_scale");
  Node* pattern_choose_qparam = choose_qparam_val->node();

  // 存储需要重写的动态量化操作节点
  std::vector<DynamicQuantOps> nodes_to_rewrite;
  std::vector<Node*> choose_qparam_nodes_to_rewrite;
  for (const Match& match : matches) {
    // 获取匹配到的节点
    Node* matched_dequantize = match.nodes_map.at(pattern_dequant);
    Node* matched_quantize = match.nodes_map.at(pattern_quant);
    Node* matched_choose_qparam = match.nodes_map.at(pattern_choose_qparam);
    // 如果去量化操作被多次使用，则需要重写
    if (matched_dequantize->output()->uses().size() > 1) {
      nodes_to_rewrite.emplace_back(
          matched_choose_qparam, matched_quantize, matched_dequantize);
    }
  }

  // 重写节点中的量化操作
  for (const auto& nodes : nodes_to_rewrite) {
    auto quant_node = std::get<1>(nodes);
    auto dequant_node = std::get<2>(nodes);
    // 获取量化调用的输入
    Value* original_val = quant_node->inputs()[0];
    Value* dequant_out = dequant_node->output();
    Value* dtype = quant_node->inputs()[3];
    // 替换去量化的输出为新量化操作的输出
    std::vector<Use> uses = dequant_out->uses();
    for (const Use& use : uses) {
      auto* user = use.user;
      WithInsertPoint ins(user);
      auto quant_ops = insertChooseQParamQuantDequant(
          graph.get(), original_val, dtype, quant_node->kind());
      user->replaceInputWith(dequant_out, std::get<2>(quant_ops)->output());
    }
  }

  // 移除重写节点的输入
  Node *choose_qparams, *quant, *dequant;
  for (const auto& n : nodes_to_rewrite) {
    std::tie(choose_qparams, quant, dequant) = n;
    dequant->removeAllInputs();
    quant->removeAllInputs();
    choose_qparams->removeAllInputs();
  }

  // 销毁重写节点
  for (const auto& n : nodes_to_rewrite) {
    std::tie(choose_qparams, quant, dequant) = n;
    dequant->destroy();
    quant->destroy();
    choose_qparams->destroy();
  }
}

// 移除冗余的去量化操作
void RemoveRedundantDequantize(std::shared_ptr<Graph>& graph) {
  // 定义去量化操作的模式字符串
  const std::string dequantize = R"(
    graph(%a_quant):
        %a_dequant = aten::dequantize(%a_quant)
        return (%a_dequant) )";
  const std::string dequantize_replacement = R"(
    graph(%a):
        return (%a) )";


# 定义一个函数图，参数为%a，返回值为(%a)



  auto filter = [&](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto dequant_node = match_vmap.at(vmap.at("a_dequant"))->node();
    Value* dequant_out = dequant_node->output();
    // Values can be used multiple times in a single node
    if (dequant_out->uses().size() != 1) {
      return false;
    }
    Node* user = dequant_out->uses()[0].user;
    return isTensorInfoNode(user);
  };


# 定义一个lambda函数filter，接受Match对象和值映射vmap作为参数
# 从匹配结果中获取值映射match_vmap
# 获取名为"a_dequant"的值，并从中获取节点node()
# 获取dequant_node的输出值dequant_out
# 检查dequant_out是否在单个节点中被多次使用
# 如果是，返回false；否则，获取使用dequant_out的第一个用户节点user
# 返回user节点是否为TensorInfoNode类型的结果



  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(dequantize, dequantize_replacement);
  rewriter.runOnGraph(graph, filter);


# 创建SubgraphRewriter对象rewriter
# 向rewriter注册重写模式，使用dequantize和dequantize_replacement作为模式和替换
# 在图graph上运行rewriter，并使用filter函数作为过滤条件
}

// 从图中移除冗余的量化操作
void RemoveRedundantQuantizationOps(std::shared_ptr<Graph>& graph) {
  // 定义动态量化操作的图模式
  const std::string dynamic_quant_ops = R"(
    graph(%a, %reduce_range, %a_dtype):
        %a_scale : float, %a_zero_point : int = aten::_choose_qparams_per_tensor(%a, %reduce_range)
        %a_quant = aten::quantize_per_tensor(%a, %a_scale, %a_zero_point, %a_dtype)
        %a_dequant = aten::dequantize(%a_quant)
        return (%a_dequant) )";
  // 定义动态量化操作的替换图模式
  const std::string dynamic_quant_replacement = R"(
    graph(%a, %reduce_range, %a_dtype):
        return (%a) )";
  
  // 过滤器函数，用于确定哪些节点需要被重写
  auto filter = [&](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto dequant_node = match_vmap.at(vmap.at("a_dequant"))->node();
    Value* dequant_out = dequant_node->output();
    // 如果输出被多个节点使用，则不重写
    if (dequant_out->uses().size() != 1) {
      return false;
    }
    Node* user = dequant_out->uses()[0].user;
    // 判断用户节点是否支持动态量化
    return !nodeQuantizable(user, QuantType::DYNAMIC);
  };

  // 子图重写器对象
  SubgraphRewriter rewriter;
  // 注册重写模式
  rewriter.RegisterRewritePattern(dynamic_quant_ops, dynamic_quant_replacement);
  // 在图上应用重写器和过滤器
  rewriter.runOnGraph(graph, filter);
}

// 复制标量参数的限制函数
void ReplicateClampScalarArgs(std::shared_ptr<Graph>& graph) {
  // 待访问的块栈
  std::stack<Block*> blocks_to_visit;
  // 待重写的标量节点集合
  std::unordered_set<Node*> scalar_nodes_to_rewrite;

  // 初始化，将图的根块压入栈中
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    // 遍历块中的每个节点
    for (Node* n : b->nodes()) {
      // 检查节点的输出
      for (Value* output : n->outputs()) {
        // 如果输出是限制标量输入的使用且使用次数大于1，则加入待重写集合
        if (getClampScalarInputUse(output) && output->uses().size() > 1) {
          scalar_nodes_to_rewrite.insert(n);
        }
      }
      // 将节点的子块压入访问栈中
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  // 对于每个待重写的标量节点
  for (Node* n : scalar_nodes_to_rewrite) {
    const std::vector<Use> uses = n->output()->uses();
    // 遍历节点的使用
    for (const auto& use : uses) {
      Node* user = use.user;
      // 在用户节点插入点处插入克隆节点
      WithInsertPoint ins(user);
      Node* cloned_node = graph->createClone(n, [](Value* v) { return v; });
      graph->insertNode(cloned_node);
      // 替换用户节点的输入为克隆节点的输出
      user->replaceInput(use.offset, cloned_node->output());
    }
  }

  // 清空每个标量节点的所有输入
  for (Node* n : scalar_nodes_to_rewrite) {
    n->removeAllInputs();
  }

  // 销毁每个标量节点
  for (Node* n : scalar_nodes_to_rewrite) {
    n->destroy();
  }
}

// 检查计算量化参数结果的有效性
void checkCalculateQParamsResult(const IValue& qparams) {
  TORCH_CHECK(
      qparams.isTuple(),
      "`calculate_qparams` 函数应返回一个元组，但得到的是:",
      qparams.tagKind());
  auto tp = qparams.toTuple();
  TORCH_CHECK(
      tp->elements().size() == 2,
      "`calculate_qparams` 函数应返回大小为 2 的元组，但得到的是大小为 ",
      tp->elements().size());
  // 元组的前两个元素应该是 Tensor 类型
  for (const auto i : c10::irange(2)) {
    // 检查元组中第 i 个元素是否为张量类型，如果不是，则输出错误信息
    TORCH_CHECK(
        tp->elements()[i].isTensor(),
        "Element of Tuple is expected to be Tensor, but element ",
        i,
        " has type: ",
        tp->elements()[i].tagKind());
  }
}

class SubGraphCloneHelper {
 public:
  // 给定一组节点，构建对应的图形。
  // 用户应确保使用预期的输入运行此图形。
  std::unique_ptr<GraphFunction> buildGraphFromNodes(
      const std::vector<Node*>& nodes,
      const std::string& name);

  // 给定源节点列表，生成包含这些节点的图形。
  void buildObserverSubgraph(
      const std::vector<Node*>& src,
      std::shared_ptr<Graph> dest);

 private:
  // 在目标图形 g 中克隆节点。
  void cloneNodeInGraph(
      Node* node,
      std::shared_ptr<Graph>& g,
      std::unordered_map<Value*, Value*>& remap_values);
};

class InsertQuantDeQuantHelper {
 public:
  InsertQuantDeQuantHelper(QuantType quant_type, bool debug)
      : quant_type_(quant_type), debug_(debug) {}

  void run(Module& module, const std::string& method_name);

  void runForOnDevicePTQ(Module& module, const std::string& method_name);

  // 清理图中的观察节点和模块对象中的观察模块及 ClassType。
  void cleanup(Module& module);

  // 仅清理图中的观察节点，而不清理模块。
  // 这是针对 ondevice PTQ 的情况。
  void removeObserverNodes(Module& m);

  // 为了通过不需要观察的操作传播量化操作，
  // 我们首先内联图形，然后调用 PropagateQuantizationOps pass。
  void propagateQuantizationOps(Module& module);

  // 用于动态量化，选择性地运行权重观察器。
  // 提取对应于权重的子图，并与模块实例一起运行。
  void runWeightObserver(Module& module, const std::string& method_name);

 private:
  ModuleMethodVector getInvokedMethods(
      Module& module,
      const std::string& method_name);

  // 获取图形中给定值的量化参数映射，
  // 通过查找值的观察器模块并从观察器模块中提取量化参数来实现。
  std::tuple<c10::QScheme, QParamVector> getQSchemeAndQParamVector(
      script::Module& module,
      Node* n);
  QuantOpParams insertCalculateQParams(
      script::Module& module,
      Graph* g,
      Node* n);

  void checkQScheme(Graph* g, c10::QScheme qscheme) {
    if (qscheme_for_graph_.count(g)) {
      // FIXME[T110786721]: This check was broken before nevery failing.
      // Once fixed, this check triggers and fails tests.
      // Fix the tests that enabling this check produce!
      /*
      TORCH_CHECK(
          qscheme_for_graph_.at(g) == qscheme,
          "Quantizing same graph with different types of "
          "QSchemes is not supported.\n",
          " Expecting:",
          c10::toString(qscheme_for_graph_.at(g)),
          " Got:",
          c10::toString(qscheme));
      */
    } else {
      qscheme_for_graph_[g] = toAffine(qscheme);
  }

  // 定义函数，用于收集观察节点和要量化的值到量化模块
  void collectObserverNodesAndValueToQuantize(Module& module, Value*);

  // 清理函数，清理模块中的图形
  void cleanup(Module& module, Graph* g);

  // 移除观察节点的函数
  void removeObserverNodes(Graph* g);

  // 对张量进行量化的函数，操作模块，图形和自变量
  void quantizeTensors(Module& module, Graph* g, Value* self);

  // 插入计算量化参数和量化操作的函数
  void insertCalculateQParamsAndQuantizationOps(
      Module& module,
      Graph* g,
      Value* self);

  // 从模块中提取并运行权重观察器的函数
  void extractAndRunWeightObserver(
      Module& module,
      Value* self,
      Value* weight_value);

  // 递归查找生成值并添加到子图中的函数
  void findSubgraph(Value* self, Value* v, std::vector<Node*>& weight_subgraph);

  // 在此传递中量化两种类型的一般操作（即对浮点数和量化张量都有效的操作）
  // 对于仅操作形状的操作，例如flatten，通过与前一个反量化操作交换来进行量化
  // 对于操作张量值的操作，例如average pool，通过在操作后插入量化/反量化操作来进行量化
  // 还有一种特殊处理clamp/hardtanh
  void propagateQuantizationOps(Block* block);

  // 从其他量化张量传播量化参数的函数
  void propagateQParams(
      Value* original_output,
      const std::vector<Value*>& inputs,
      bool is_scalar = false,
      const std::optional<std::tuple<c10::QScheme, QParamVector>>& qparams_opt =
          c10::nullopt);

  // 检查值是否已量化的函数
  bool isQuantized(Value* v) {
    // 返回 quantized_values_ 中是否存在值 v 的计数是否不为零
    return quantized_values_.count(v) != 0;
    }
    
    std::unordered_map<Graph*, std::vector<std::string>>
        observer_modules_to_remove_;
    // 第一次遇到图时，仅从类型中移除观察模块属性；之后，由于属性已从 ClassType 中移除，
    // 我们将使用槽索引列表来重放此移除操作
    std::unordered_map<Graph*, std::vector<int>> removed_observer_slots_;
    
    std::unordered_map<Graph*, std::vector<Node*>> nodes_to_destroy_;
    // 图到观察节点的映射，可以使用观察节点获取已观察的原始值信息和量化参数
    std::unordered_map<Graph*, std::vector<Node*>> observer_nodes_for_graph_;
    
    // 从 qparam 名称（例如 _scale）到模块中的属性名称（例如 weight_scale_0）的映射
    std::unordered_map<Node*, std::unordered_map<std::string, std::string>>
        qparam_name_map_for_node_;
    
    // 记录每个图的量化方案，用于检查每个图仅以一种 QScheme 类型进行量化
    std::unordered_map<Graph*, c10::QScheme> qscheme_for_graph_;
    
    // 用于记录已量化值的集合，以便每个值仅被量化一次
    std::unordered_set<Value*> quantized_values_;
    
    // 将原始权重值映射到包含权重观察器和依赖节点的子图对应的 GraphFunction 的映射
    std::unordered_map<Value*, std::unique_ptr<GraphFunction>>
        weight_to_graph_fn_;
    
    QuantType quant_type_ = QuantType::STATIC;
    bool debug_ = false;
};

void InsertQuantDeQuantHelper::collectObserverNodesAndValueToQuantize(
    Module& module,
    Value* v) {
  // 获取值 v 所属的图形对象
  auto* g = v->owningGraph();
  // 查找值 v 对应的观察者名称
  auto observer_name = findObserverName(v);
  // 如果找不到观察者名称，则返回
  if (!observer_name) {
    return;
  }
  // 将图形 g 中的观察者模块名添加到待移除列表中
  observer_modules_to_remove_[g].push_back(observer_name.value());

  // 获取值 v 对应的观察者节点
  Node* observer = v->node();
  // 断言观察者节点是 prim::CallMethod 类型，并且调用的方法名是 "forward"，
  // 以及观察者节点的第一个输入节点是 prim::GetAttr 类型，并且属性名与 observer_name 匹配
  TORCH_INTERNAL_ASSERT(
      observer->kind() == prim::CallMethod &&
      observer->s(attr::name) == "forward" &&
      observer->inputs()[0]->node()->kind() == prim::GetAttr &&
      observer->inputs()[0]->node()->s(attr::name) == observer_name);

  // 将观察者节点添加到待销毁节点列表中
  nodes_to_destroy_[g].push_back(observer);
  // 将观察者模块的 GetAttr 节点添加到待销毁节点列表中
  nodes_to_destroy_[g].push_back(observer->inputs()[0]->node());
  // 将观察者节点添加到图形 g 的观察者节点列表中
  observer_nodes_for_graph_[g].push_back(observer);
}

void InsertQuantDeQuantHelper::removeObserverNodes(Module& module) {
  // 遍历模块中的所有方法，移除观察者节点
  for (auto& method : module.get_methods()) {
    removeObserverNodes(method.graph().get());
  }
  // 遍历模块中的所有子模块，递归移除观察者节点
  for (Module m : module.children()) {
    removeObserverNodes(m);
  }
}

void InsertQuantDeQuantHelper::removeObserverNodes(Graph* g) {
  // 如果待销毁节点列表中包含图形 g
  if (nodes_to_destroy_.count(g)) {
    // 移除列表中所有节点的输入
    for (auto& n : nodes_to_destroy_.at(g)) {
      n->removeAllInputs();
    }
    // 销毁列表中所有节点
    for (auto& n : nodes_to_destroy_.at(g)) {
      n->destroy();
    }
    // 清空待销毁节点列表
    nodes_to_destroy_.at(g).clear();
  }
}

void InsertQuantDeQuantHelper::cleanup(Module& module) {
  // 遍历模块中的所有方法，清理观察者节点
  for (auto& method : module.get_methods()) {
    cleanup(module, method.graph().get());
  }
  // 遍历模块中的所有子模块，递归清理观察者节点
  for (Module m : module.children()) {
    cleanup(m);
  }
}

void InsertQuantDeQuantHelper::cleanup(Module& module, Graph* g) {
  // 输出清理前的图形结构
  GRAPH_DUMP("Before Remove Observers:", g);
  // 移除图形 g 中的观察者节点
  removeObserverNodes(g);

  // 1. 如果此图形已经处理过，意味着观察者属性已经从类型中移除（参见步骤2），
  //    但这些属性的槽位索引仍然保留在列表中，我们将使用这些槽位索引重新执行观察者槽位的移除
  if (removed_observer_slots_.count(g)) {
    // 遍历需要移除的槽位索引，从模块中不安全地移除对应的槽位
    for (auto slot : removed_observer_slots_.at(g)) {
      module._ivalue()->unsafeRemoveSlot(slot);
    }
  }

  // 2. 从最后一个观察者模块开始向第一个模块顺序移除观察者模块，
  //    以减少时间复杂度。假设所有观察者模块都是在现有模块之后添加的，
  //    使用此优化后的时间复杂度为 O(N)，其中 N 是观察者模块的数量
  if (observer_modules_to_remove_.count(g)) {
    auto& observers = observer_modules_to_remove_.at(g);
    // 遍历待移除的观察者模块列表，并从模块中移除这些模块
    // 注意：这里假设了观察者模块是按顺序添加的，且后添加的在列表的末尾
    for (auto& observer_name : observers) {
      // Remove observer module logic goes here
    }
  }
}
    // 从最后一个观察者开始向前遍历观察者列表
    for (int64_t i = observers.size() - 1; i >= 0; --i) {
      // 获取当前观察者的名称
      auto observer_name = observers[i];
      // 打印调试信息，显示正在尝试移除的观察者名称
      GRAPH_DEBUG("Trying to remove: ", observer_name);
      // 检查模块类型是否包含该观察者属性
      if (module.type()->hasAttribute(observer_name)) {
        // 在这里记录槽位索引，以便在共享ClassType的其他对象中重放槽位移除操作
        removed_observer_slots_[g].push_back(
            module.type()->getAttributeSlot(observer_name));
        // 使用unsafeRemoveAttr方法不安全地移除模块的属性
        module._ivalue()->unsafeRemoveAttr(observer_name);
        // 使用unsafeRemoveAttribute方法不安全地移除模块类型的属性
        module.type()->unsafeRemoveAttribute(observer_name);
      }
    }
    // 清空观察者列表
    observers.clear();
  }
  // 打印图形状态的调试信息，显示移除观察者后的状态
  GRAPH_DUMP("After remove observers :", g);
}

void SubGraphCloneHelper::cloneNodeInGraph(
    Node* node,
    std::shared_ptr<Graph>& g,
    std::unordered_map<Value*, Value*>& remap_old_to_new) {
  auto* block = g->block();
  auto value_fn = [&](Value* v) {
    if (remap_old_to_new.count(v) == 0) {
      auto new_value = g->block()->addInput();
      remap_old_to_new[v] = new_value;
      new_value->copyMetadata(v);
      return new_value;
    } else {
      return remap_old_to_new[v];
    }
  };

  // 创建一个新节点，将克隆后的节点添加到图中
  auto new_node = block->appendNode(g->createClone(node, value_fn));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    remap_old_to_new[oo] = no;
  }
}

void SubGraphCloneHelper::buildObserverSubgraph(
    const std::vector<Node*>& weight_subgraph,
    std::shared_ptr<Graph> dest_graph) {
  std::unordered_map<Value*, Value*> remap_old_to_new;
  // 构建权重子图
  for (auto n : weight_subgraph) {
    cloneNodeInGraph(n, dest_graph, remap_old_to_new);
  }
  // 对目标图进行静态分析
  LintGraph(dest_graph);

  // 将最后一个节点的输出值作为子图的输出
  for (auto out : weight_subgraph.back()->outputs()) {
    dest_graph->registerOutput(remap_old_to_new[out]);
  }
  // 输出新的权重观察子图
  GRAPH_DUMP("New weight observer subgraph: ", dest_graph);
}

std::unique_ptr<GraphFunction> SubGraphCloneHelper::buildGraphFromNodes(
    const std::vector<Node*>& nodes,
    const std::string& name) {
  auto observer_subgraph = std::make_shared<Graph>();
  auto build_observer_graph = [&](GraphFunction& func) {
    buildObserverSubgraph(nodes, func.graph());
  };
  return std::make_unique<GraphFunction>(
      name, observer_subgraph, build_observer_graph);
}

void InsertQuantDeQuantHelper::findSubgraph(
    Value* self,
    Value* input_val,
    std::vector<Node*>& weight_subgraph) {
  Node* node = input_val->node();
  weight_subgraph.push_back(node);
  const auto& inputs = node->inputs().vec();
  for (auto v : inputs) {
    if (!hitGraphInput(v)) {
      // 递归查找权重子图
      findSubgraph(self, v, weight_subgraph);
    } else {
      TORCH_CHECK(
          v == self,
          "Unexpected value found when handling weight value "
          " in findSubgraph, traced back to:",
          v->debugName(),
          " which is not self:",
          self->debugName());
    }
  }
}

void InsertQuantDeQuantHelper::extractAndRunWeightObserver(
    Module& module,
    Value* self,
    Value* weight_value) {
  std::vector<Node*> weight_subgraph;
  // 如果权重值还未被处理过，则执行以下代码块
  if (weight_to_graph_fn_.count(weight_value) == 0) {
    // 提取权重观察子图节点
    findSubgraph(self, weight_value, weight_subgraph);

    // 反转子图以正确方向遍历
    std::reverse(weight_subgraph.begin(), weight_subgraph.end());

    // 使用权重观察节点构建图形
    SubGraphCloneHelper o;
    // 使用给定的节点构建图函数，并将其封装在 unique_ptr 中
    std::unique_ptr<GraphFunction> func =
        o.buildGraphFromNodes(weight_subgraph, "observer_subgraph");

    // 将构建好的图函数移动到权重值对应的映射中
    weight_to_graph_fn_[weight_value] = std::move(func);
  }

  // 创建一个包含模块输入的栈
  Stack module_inp = {module._ivalue()};

  // 使用权重值对应的图函数来执行计算，输入为模块输入栈
  weight_to_graph_fn_[weight_value]->run(module_inp);
}

void InsertQuantDeQuantHelper::quantizeTensors(
    Module& module,
    Graph* g,
    Value* self) {
  // 检查当前图是否有量化观察节点，如果没有则返回
  if (!observer_nodes_for_graph_.count(g)) {
    return;
  }
  // 对于每个观察节点执行以下操作
  for (auto* n : observer_nodes_for_graph_.at(g)) {
    // 获取观察节点的第二个输入作为原始值
    auto* original_value = n->input(1);
    // 获取量化方案和量化参数映射
    auto tp = getQSchemeAndQParamVector(module, n);
    auto qscheme = std::get<0>(tp);
    auto qparam_map = std::get<1>(tp);
    // 检查量化方案的有效性
    checkQScheme(g, qscheme);
    // 存储量化参数的名称列表
    std::vector<std::string> qparam_names;
    // 遍历量化参数映射
    for (auto& pr : qparam_map) {
      const auto& name = pr.first;
      const auto& qparam = pr.second;
      size_t uid = 0;
      // 根据原始值的调试名称创建唯一的量化参数名称
      auto qparam_name =
          original_value->debugName() + name + "_" + std::to_string(uid++);
      // 确保名称唯一性，直到找到未被使用的名称
      while (module.hasattr(qparam_name)) {
        qparam_name =
            original_value->debugName() + name + "_" + std::to_string(uid++);
      }
      // 将量化参数名称映射到节点的量化参数名称映射中
      qparam_name_map_for_node_[n][name] = qparam_name;
      // 在模块中注册量化参数作为属性
      module.register_attribute(qparam_name, qparam.type(), qparam);
      // 将量化参数名称添加到列表中
      qparam_names.push_back(qparam_name);
    }
    // 插入量化操作到模块中
    insertQuantizationOps(
        module, self, n, isPerChannel(qscheme), qparam_names, quant_type_);
  }
}

std::tuple<c10::QScheme, QParamVector> InsertQuantDeQuantHelper::
    getQSchemeAndQParamVector(script::Module& module, Node* n) {
  // TODO: refactor findObserverName to take Node* as input
  // 获取节点的输出值作为量化观察名称
  Value* v = n->output();
  // 断言输出值类型是 Tensor 类型
  TORCH_INTERNAL_ASSERT(
      v->type()->isSubtypeOf(*TensorType::get()),
      "Expected output of observer node to be Tensor");
  // 查找与输出值相关的量化观察名称
  auto observer_name = findObserverName(v);
  // 断言量化观察名称存在
  TORCH_INTERNAL_ASSERT(
      observer_name,
      "getQSchemeAndParamMap expects the corresponding observer for ",
      v->debugName(),
      " exists.");
  // 初始化量化参数向量和量化方案
  QParamVector qparams;
  c10::QScheme qscheme = c10::kPerTensorAffine;

  // 获取量化观察模块并查询其数据类型
  auto observer_module = module.attr(observer_name.value()).toModule();
  auto scalar_type = observer_module.attr("dtype");
  // 如果输入是占位符观察节点，为动态量化获取计算数据类型
  if (isPlaceholderObserver(n->input(0))) {
    if (observer_module.hasattr("is_dynamic") &&
        observer_module.attr("is_dynamic").toBool()) {
      // 如果是动态量化，将标量类型和数据类型添加到量化参数中
      qparams.emplace_back(kScalarType, observer_module.attr("dtype"));
    }
    return std::make_tuple(qscheme, std::move(qparams));
  } else if (scalar_type == at::ScalarType::Half) {
  // 返回一个包含量化方案和量化参数的元组
  return std::make_tuple(qscheme, std::move(qparams));
}
// 获取 observer_module 中的 calculate_qparams 方法
auto calculate_qparams = observer_module.get_method("calculate_qparams");
// 调用 calculate_qparams 方法，传入一个空的 IValue 向量，获取返回值
IValue result = calculate_qparams(std::vector<IValue>());
// 检查 calculate_qparams 返回的结果
checkCalculateQParamsResult(result);
// 检查 scalar_type 是否不为 Undefined
TORCH_CHECK(
    scalar_type.toScalarType() != at::ScalarType::Undefined,
    "dtype of observer can't be undefined");
// 将 result 转换为元组
auto tp = result.toTuple();
// 提取元组中的第一个元素（scale），并将其转换为 float 类型的 Tensor
at::Tensor scale = tp->elements()[0].toTensor().to(at::kFloat);
// 提取元组中的第二个元素（zero_point），并将其转换为 int 类型的 Tensor
at::Tensor zero_point = tp->elements()[1].toTensor().to(at::kInt);
// 注释：量化参数应按照 quantize_per_tensor/quantize_per_channel 函数的参数顺序出现

// 获取 observer_module 中的 qscheme 属性，并转换为 QScheme 类型
qscheme = observer_module.attr("qscheme").toQScheme();
// 如果量化方案是按通道的，则获取 ch_axis 属性并添加到 qparams 中
if (isPerChannel(qscheme)) {
  auto axis = observer_module.attr("ch_axis");
  qparams.emplace_back("_scale", scale);
  qparams.emplace_back("_zero_point", zero_point);
  qparams.emplace_back("_axis", axis.toInt());
} else {
  // 如果量化方案是按张量的，则将 scale 和 zero_point 作为 double 和 int64_t 类型添加到 qparams 中
  qparams.emplace_back("_scale", scale.item<double>());
  qparams.emplace_back("_zero_point", zero_point.item<int64_t>());
}
// 将 scalar_type 添加为 qparams 的最后一个元素
qparams.emplace_back(kScalarType, scalar_type);
// 返回一个包含量化方案和量化参数的元组
return std::make_tuple(qscheme, std::move(qparams));
ModuleMethodVector InsertQuantDeQuantHelper::getInvokedMethods(
    Module& module,
    const std::string& method_name) {
  // 获取指定方法的计算图
  auto graph = module.get_method(method_name).graph();

  // 存储调用的方法的列表
  ModuleMethodVector invoked_methods;
  
  // 待访问的代码块栈，从计算图的入口块开始
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());

  // 遍历计算图中的每个代码块和节点
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();

    // 遍历当前代码块中的每个节点
    for (Node* n : b->nodes()) {
      // 如果节点是方法调用
      if (n->kind() == prim::CallMethod) {
        auto module_instance = n->inputs()[0];
        auto module_method_name = n->s(attr::name);
        std::optional<Module> m;

        // 如果调用自身模块的方法
        if (module_instance == graph->inputs()[0]) {
          m = module;
        } else if (
            module_instance->node()->kind() == prim::GetAttr &&
            module_instance->node()->s(attr::name).find("_observer_") ==
                std::string::npos) {
          // 否则，根据节点的属性查找对应的模块
          m = getInvokedModuleOpt(module, n, graph->inputs()[0]);
        }

        // 如果找到模块，则加入到调用方法列表中
        if (m) {
          invoked_methods.emplace_back(*m, module_method_name);
        }
      }

      // 将当前节点的子代码块加入待访问栈中
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  
  // 返回调用的方法列表
  return invoked_methods;
}

void InsertQuantDeQuantHelper::propagateQParams(
    Value* original_output,
    const std::vector<Value*>& inputs,
    bool is_scalar,
    const std::optional<std::tuple<c10::QScheme, QParamVector>>& qparams_opt) {
  // 获取原始输出节点
  Node* n = original_output->node();
  Graph* graph = n->owningGraph();

  // 如果是标量，插入标量转换为张量的操作
  if (is_scalar) {
    n = insertScalarToTensor(graph, original_output);
    original_output = n->output();
  }

  // 断言输入值的数量为1，用于ATen函数
  TORCH_INTERNAL_ASSERT(
      inputs.size() == 1, "Expecting single input for the aten function");

  // 获取量化输入节点
  Value* quantized_input = inputs[0]->node()->input(0);

  // 插入在量化输入节点后面
  Node* quantized_input_node = quantized_input->node();
  WithInsertPoint ins(
      quantized_input_node->isAfter(n) ? quantized_input_node->next()
                                       : n->next());

  // 存储量化操作的输入值
  std::vector<Value*> quant_inputs;
  auto quant_kind = Symbol::aten("quantize_per_tensor");

  // 如果提供了量化参数选项
  if (qparams_opt.has_value()) {
    quant_inputs = {original_output};
    auto qscheme = std::get<0>(*qparams_opt);
    auto qparams = std::get<1>(*qparams_opt);

    // 如果是按通道量化，则使用对应的量化操作
    if (isPerChannel(qscheme)) {
      quant_kind = Symbol::aten("quantize_per_channel");
    }

    // 遍历量化参数，插入常量节点到图中
    for (const auto& qparam : qparams) {
      Value* qparam_val = graph->insertConstant(qparam.second);
      qparam_val->setDebugName(quantized_input->debugName() + qparam.first);
      quant_inputs.push_back(qparam_val);
    }
  } else {
  // 只支持每个张量的仿射量化张量的情况
  // 从前一个量化操作中获取量化参数
  Node* scale = insertQParam(
      graph,
      quantized_input,
      at::Symbol::aten("q_scale"),
      FloatType::get(),
      "q_scale");
  Node* zero_point = insertQParam(
      graph,
      quantized_input,
      at::Symbol::aten("q_zero_point"),
      IntType::get(),
      "q_zero_point");
  Node* dtype = insertQParam(
      graph, quantized_input, prim::dtype, IntType::get(), "dtype");
  quant_inputs = {
      original_output,
      scale->output(),
      zero_point->output(),
      dtype->output()};
}
Node* quant = insertQuant(
    graph, quant_inputs, quant_kind, original_output->debugName() + ".quant");
Value* quantized_output = quant->output();
// 用量化输出替换一般操作的原始输出的使用
original_output->replaceAllUsesAfterNodeWith(quant, quantized_output);
const auto& outputs =
    insertDeQuantForAllUse(graph, quantized_output, quantized_output);
for (auto* output : outputs) {
  if (is_scalar) {
    // 将反量化的张量转换回标量
    Node* item = insertItem(graph, output, FloatType::get());
    Value* scalar = item->output();
    output->replaceAllUsesAfterNodeWith(item, scalar);
    output = scalar;
  }
  quantized_values_.insert(output);
}
}

// 从输入集合中删除所有的去量化节点
void removeDequantizeFromInputs(const std::unordered_set<Value*>& inputs) {
  // 遍历输入集合中的每一个去量化值
  for (auto* dequantized_val : inputs) {
    // 获取与去量化值关联的节点
    auto* dequantize_node = dequantized_val->node();
    // 断言每个去量化值只有一个去量化节点与之关联
    TORCH_INTERNAL_ASSERT(
        dequantized_val->uses().size() == 1,
        "Expect to have one dequantize node for each use");
    // 用去量化节点的输入替换所有使用去量化值的地方
    dequantized_val->replaceAllUsesWith(dequantize_node->inputs()[0]);
    // 移除去量化节点的所有输入
    dequantize_node->removeAllInputs();
    // 销毁去量化节点
    dequantize_node->destroy();
  }
}

// 检查是否需要从输出传播量化操作
std::optional<std::vector<Value*>> getDequantizedInputs(Value* output) {
  // 获取传递给输出的所有输入
  auto inputs = getPassThroughInputs(output);
  if (!inputs.empty()) {
    // 不需要递归检查 prim::If 节点，因为如果 prim::If 的所有输入都是去量化的，
    // 在到达这一点之前去量化操作将被移除
    bool is_dequantized = true;
    for (auto* input : inputs) {
      // 调试信息：检查输入节点是否被量化
      GRAPH_DEBUG(
          "checking if input:",
          input->debugName(),
          " in node:",
          *input->node(),
          "is quantized");
      // 检查输入节点是否是 dequantize 操作
      is_dequantized &= input->node()->kind() == Symbol::aten("dequantize");
    }
    // 如果所有输入节点都是去量化操作，则返回这些输入节点
    if (is_dequantized) {
      return inputs;
    }
  }
  return c10::nullopt;
}

// 在块内传播量化操作的帮助器函数
void InsertQuantDeQuantHelper::propagateQuantizationOps(Block* block) {
  // 遍历块内的每一个节点
  for (Node* n : block->nodes()) {
    // 如果节点是 prim::If 类型
    if (n->kind() == prim::If) {
      // 递归地在每个分支块中传播量化操作
      for (Block* subblock : n->blocks()) {
        propagateQuantizationOps(subblock);
      }
      // 如果 if 节点没有输出，则继续下一个节点
      if (n->outputs().empty()) {
        continue;
      }
      // 如果 if 节点有多个输出，则不支持将去量化操作移除
      if (n->outputs().size() > 1) {
        // 不支持目前无法对有多个输出的 if 块进行去量化操作移除
        continue;
      }
    }
    // 如果节点是单输入的通用值 ATen 函数
    if (isSingleInputGeneralValueAtenFunction(n)) {
      // 遍历节点的每一个输出
      for (auto* output : n->outputs()) {
        // 如果输出已经被量化，则继续下一个输出
        if (isQuantized(output)) {
          continue;
        }
        // 获取输出的去量化输入
        if (auto inputs = getDequantizedInputs(output)) {
          // 传播量化参数到输出和输入
          propagateQParams(output, *inputs);
          // 如果节点是 clamp 操作
          if (isClamp(n)) {
            // 对 clamp/hardtanh 的 min 和 max 标量参数传播量化参数
            for (size_t i = 1; i <= 2; ++i) {
              propagateQParams(n->input(i), *inputs, /* is_scalar */ true);
            }
          }
        }
      }
    } else if (auto qparams_opt = getFixedQParams(n)) {
      // 如果节点有固定的量化参数
      for (auto* output : n->outputs()) {
        // 如果输出已经被量化，则继续下一个输出
        if (isQuantized(output)) {
          continue;
        }
        // 获取输出的去量化输入
        if (auto inputs = getDequantizedInputs(output)) {
          // 传播固定的量化参数到输出和输入
          propagateQParams(output, *inputs, /* is_scalar */ false, qparams_opt);
        }
      }
    } else {
      // 处理被 dequantize 操作量化的操作
      // 例如 flatten 操作需要执行以下步骤：
      // 1. 检查是否需要传播 dequantize 操作
      // 2. 从输入中移除 dequantize 操作
      // 3. 为所有输出插入 dequantize 操作
      // 以确保对具有多个输出的操作有效
      // 由于从输入中移除 dequantize 操作会修改图结构，
      // 这将影响未来检查所有输入是否已经量化的过程
      // （因为当前我们仅仅通过检查数值是否由 dequantize 操作产生来决定其是否被量化）
      
      // 存储被 dequantize 操作解量化的输入值
      std::unordered_set<Value*> dequantized_inputs;
      // 存储需要插入 dequantize 操作的输出值
      std::vector<Value*> outputs_to_dequantize;
      
      // 1. 收集需要解量化的输入和需要插入 dequantize 操作的输出
      for (auto* output : n->outputs()) {
        // 如果输出已经被量化，则跳过
        if (isQuantized(output)) {
          continue;
        }
        // 获取与该输出相关的 dequantize 操作的输入
        if (auto inputs = getDequantizedInputs(output)) {
          std::copy(
              inputs->begin(),
              inputs->end(),
              std::inserter(dequantized_inputs, dequantized_inputs.end()));
          outputs_to_dequantize.push_back(output);
        }
      }
      
      // 2. 从输入中移除 dequantize 操作
      removeDequantizeFromInputs(dequantized_inputs);
      
      // 3. 为输出插入 dequantize 操作
      for (auto* output : outputs_to_dequantize) {
        insertDeQuantForAllUse(output->owningGraph(), output, output);
      }
    }

    if (isBinaryOpWithScalarInput(n)) {
      // 当调试选项启用时，对于 add_scalar/mul_scalar 输出警告信息
      // 因为这些操作的量化参数取决于输入，编码这些方程在 IR 中过于复杂：
      // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/BinaryOps.cpp#L64-L74
      if (debug_) {
        TORCH_WARN_ONCE(
            "不支持对 add_scalar 和 mul_scalar 使用调试选项，请不要为使用这些操作的模型启用调试选项。");
      }
    }
  }
}

// 运行权重观察器的辅助函数，递归遍历模块中调用的方法，观察权重
void InsertQuantDeQuantHelper::runWeightObserver(
    Module& module,
    const std::string& method_name) {
  // 如果量化类型不是动态量化，直接返回
  if (quant_type_ != QuantType::DYNAMIC) {
    return;
  }

  // 获取调用方法的元组，并逐一处理
  for (auto& invoked_methods : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_methods);
    const auto& invoked_method_name = std::get<1>(invoked_methods);
    // 递归调用自身，处理被调用方法中的权重观察
    runWeightObserver(invoked_module, invoked_method_name);
  }
  
  // 获取指定方法的 Method 对象
  Method method = module.get_method(method_name);
  auto graph = method.graph();
  // 获取图的输入值
  Value* self = graph->inputs()[0];

  std::vector<Value*> weight_values;
  // 访问当前图中的所有块，查找权重值
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (auto n : b->nodes()) {
      for (Value* v : n->outputs()) {
        // 如果值不是 Tensor 类型，继续下一个值
        if (!v->type()->isSubtypeOf(*TensorType::get())) {
          continue;
        }
        // 查找观察器名称
        auto observer_name = findObserverName(v);
        // 如果找到观察器名称并且值是权重，则添加到权重值向量中
        if (observer_name && isWeight(module, v)) {
          weight_values.push_back(v);
        }
      }
      // 将节点的子块添加到访问栈中
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  
  // 对所有观察到的权重值，找到贡献于权重张量的子图，并运行该子图以观察权重
  for (const auto& v : weight_values) {
    extractAndRunWeightObserver(module, self, v);
  }
}

// 运行插入量化和去量化的辅助函数，递归处理调用方法
void InsertQuantDeQuantHelper::run(
    Module& module,
    const std::string& method_name) {
  // 获取调用方法的元组，并逐一处理
  for (auto& invoked_methods : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_methods);
    const auto& invoked_method_name = std::get<1>(invoked_methods);
    // 递归调用自身，处理被调用方法
    run(invoked_module, invoked_method_name);
  }

  // 获取指定方法的 Method 对象
  Method method = module.get_method(method_name);
  auto graph = method.graph();
  // 如果图中已经有观察节点，注册新参数
  if (observer_nodes_for_graph_.count(graph.get())) {
    for (auto* n : observer_nodes_for_graph_.at(graph.get())) {
      // 获取量化方案和量化参数向量
      auto tp = getQSchemeAndQParamVector(module, n);
      checkQScheme(graph.get(), std::get<0>(tp));
      auto qparam_map = std::get<1>(tp);
      // 检查参数映射是否为空
      if (!qparam_map.empty()) {
        TORCH_INTERNAL_ASSERT(
            qparam_name_map_for_node_.count(n),
            "Expected to have a qparam_name_map for node:",
            *n);
        auto qparam_name_map = qparam_name_map_for_node_.at(n);
        // 遍历量化参数映射，设置到模块属性中
        for (auto& pr : qparam_map) {
          const auto& name = pr.first;
          const auto& qparam = pr.second;
          module._ivalue()->setAttr(qparam_name_map.at(name), qparam);
        }
      }
    }
    return;
  }

  // prim::Param nodes do not belong to the graph. Hence the Insert
  // point is the beginning of graph node. This also safe guards against
  // observing a potentially mutated value due to some in-place operation
  // 确保 prim::Param 节点不属于图的一部分。因此插入点是图节点的起始位置。
  // 这也防止由于某些原地操作观察到可能发生变异的值。
  std::vector<Value*> input_values;
  for (const auto idx : c10::irange(1, method.num_inputs())) {
    // 遍历从第一个输入开始的所有图输入节点，索引从1开始
    auto& v = graph->inputs()[idx];
    // 如果节点的类型是 TensorType 的子类型，则将其添加到输入值向量中
    if (v->type()->isSubtypeOf(*TensorType::get())) {
      input_values.push_back(v);
    }
  }

  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  // 使用深度优先搜索遍历所有的基本块和节点
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end;) {
      // 遍历当前基本块中的所有节点
      Node* n = *it++;
      for (Value* v : n->outputs()) {
        // 如果节点输出的值不是 TensorType 的子类型，则继续下一个输出值
        if (!v->type()->isSubtypeOf(*TensorType::get())) {
          continue;
        }
        // 收集需要量化的观察节点和值
        collectObserverNodesAndValueToQuantize(module, v);
      }

      for (Block* subblock : n->blocks()) {
        // 将节点的子块加入到待访问的块栈中，以便后续遍历
        blocks_to_visit.push(subblock);
      }
    }
  }

  // 对输入值向量中的每个值，收集需要量化的观察节点和值
  for (Value* v : input_values) {
    collectObserverNodesAndValueToQuantize(module, v);
  }
  // 在量化张量之前，输出图的快照
  GRAPH_DUMP("Before Quantize Tensors:", graph);
  // 获取图的第一个输入作为 self 值
  Value* self = graph->inputs()[0];
  // 对图中的张量进行量化
  quantizeTensors(module, graph.get(), self);
  // 在量化张量之后，输出图的快照
  GRAPH_DUMP("After Quantize Tensors:", graph);
}

// 用于在模块中传播量化操作的辅助函数。该函数执行一系列操作来修改模块中的图。
void InsertQuantDeQuantHelper::propagateQuantizationOps(Module& module) {
  // 交换功能线性层，这可能涉及将线性层转换为量化操作
  SwapFunctionalLinear(module);
  // 获取模块的前向方法对应的图
  auto graph = module.get_method("forward").graph();
  // 内联所有函数调用，将函数调用展开到调用点
  Inline(*graph);
  // 对图进行常量传播，尝试用常量替换变量
  ConstantPropagation(graph);
  // 复制选择的量化参数和量化解量化操作
  ReplicateChooseQParamsQuantDequant(graph);
  // 移除冗余的量化操作
  RemoveRedundantQuantizationOps(graph);
  // 复制量化操作
  ReplicateQuant(graph);
  // 复制解量化操作
  ReplicateDeQuant(graph);
  // TODO: 添加过滤器以处理 clamp 模式，并移除此 pass
  // 复制 clamp 操作中的标量参数
  ReplicateClampScalarArgs(graph);
  // 递归地在图块中传播量化操作
  propagateQuantizationOps(graph->block());
  // 移除冗余的解量化操作
  RemoveRedundantDequantize(graph);
}

// 插入静态和动态量化节点到图中的辅助模板特化函数
template <>
Node* insertQuantDequantNodes<QuantOpParams>(
    Value* self,
    Node* observer,
    QuantOpParams& qparams,
    const std::string& quantize_func) {
  (void)self;
  Graph* g = observer->owningGraph();
  Value* observer_out = observer->output();
  Value* original_val = observer->input(1);
  std::vector<Value*> inputs;
  // + 1 for tensor to be quantized
  // 为要量化的张量预留输入空间
  inputs.reserve(qparams.qparams.size() + 1);
  inputs.push_back({observer_out});
  // 将量化参数添加到输入向量中
  for (const auto& qparam_values : qparams.qparams) {
    inputs.push_back(qparam_values);
  }
  // 插入量化节点到图中
  Node* quant = insertQuant(
      g,
      inputs,
      at::Symbol::aten(quantize_func),
      original_val->debugName() + ".quant");
  // 确保量化节点出现在其依赖值之后
  for (Value* v : inputs) {
    quant->moveAfter(v->node());
  }
  // 插入解量化节点到图中
  Node* dequant = insertDeQuant(g, quant->output(), original_val);
  dequant->moveAfter(quant);
  return dequant;
}

// 检查 calculate_qparams 的结果类型
void checkCalculateQParamsResultTypes(const Node* out) {
  TORCH_CHECK(
      out->outputs().size() == 2,
      "calculate_qparams should produce output of size 2 (scale, zero_point).");
  Value* scale = out->output(0);
  Value* zp = out->output(1);
  TORCH_CHECK(
      scale->type()->expect<TensorType>(),
      "Scale value should be of Tensor type.");
  TORCH_CHECK(
      zp->type()->expect<TensorType>(), "Zero-point value should be of Tensor type.");
}

// 插入 calculate_qparams 到图中的辅助函数
QuantOpParams InsertQuantDeQuantHelper::insertCalculateQParams(
    script::Module& module,
    Graph* g,
    // TODO: refactor findObserverName to take Node* as input
    // 获取自身值，这里假设g是一个Graph对象，inputs()返回图的输入节点数组，[0]表示第一个输入节点，即self
    Value* self = g->inputs()[0];
    // 获取节点n的输出值
    Value* v = n->output();
    // 断言节点n的输出值类型是TensorType的子类型
    TORCH_INTERNAL_ASSERT(
        v->type()->isSubtypeOf(*TensorType::get()),
        "Expected output of observer node to be Tensor");
    // 查找观察器名称对应的observer_name
    auto observer_name = findObserverName(v);
    // 断言observer_name不为空，确保观察器对应于节点v的存在
    TORCH_INTERNAL_ASSERT(
        observer_name,
        "getQSchemeAndParamMap expects the corresponding observer for ",
        v->debugName(),
        " exists.");
    // 创建空的值数组qparams_graph_values和QuantOpParams对象quant_op_params
    std::vector<Value*> qparams_graph_values;
    QuantOpParams quant_op_params;
    
    // 检查节点n的第一个输入节点是否为占位符观察器，不支持在设备端进行PTQ
    TORCH_CHECK(
        !isPlaceholderObserver(n->input(0)),
        "Placeholder observers are not supported in ondevice PTQ.");
    // 获取observer_module对象，假设module是一个模块，通过observer_name获取观察器模块
    auto observer_module = module.attr(observer_name.value()).toModule();
    // 插入节点操作，将self和observer_name.value()作为参数，返回observer_module_value
    Value* observer_module_value = g->insertGetAttr(self, observer_name.value());
    // 获取观察器模块的标量类型
    auto scalar_type = observer_module.attr("dtype");
    // 断言标量类型不是未定义的ScalarType
    TORCH_CHECK(
        scalar_type.toScalarType() != at::ScalarType::Undefined,
        "dtype of observer can't be undefined");
    // 如果标量类型是at::ScalarType::Half，则直接返回空的quant_op_params
    if (scalar_type == at::ScalarType::Half) {
      return quant_op_params;
    }
    // 获取观察器模块的calculate_qparams方法
    auto calculate_qparams = observer_module.get_method("calculate_qparams");
    // 获取calculate_qparams方法的schema
    auto calculate_qparams_schema = calculate_qparams.function().getSchema();
    // 匹配schema，返回MatchedSchema对象，传入observer_module_value作为参数
    MatchedSchema matched_schema = matchSchema(
        calculate_qparams_schema,
        v->node()->sourceRange(),
        *g,
        {observer_module_value},
        {});
    // 插入方法调用节点，调用calculate_qparams方法，并返回其节点
    Node* call = g->insertMethodCall("calculate_qparams", matched_schema)->node();
    // 插入节点，创建TupleUnpack节点，用于解包调用结果的输出
    Node* scale_zp_node = g->insertNode(g->createTupleUnpack(call->output(0)));
    // 检查calculate_qparams的结果类型是否符合预期
    checkCalculateQParamsResultTypes(scale_zp_node);
    // 获取观察器模块的qscheme属性，赋值给quant_op_params的qscheme字段
    auto qscheme = observer_module.attr("qscheme").toQScheme();
    quant_op_params.qscheme = qscheme;
    // 将scale_zp_node的第一个输出（scale值）添加到quant_op_params的qparams数组
    quant_op_params.qparams.push_back(scale_zp_node->output(0)); // scale Value*
    // 将scale_zp_node的第二个输出（zero_point值）添加到quant_op_params的qparams数组
    quant_op_params.qparams.push_back(
        scale_zp_node->output(1)); // zero_point Value*
    // 如果是按通道量化，则获取ch_axis属性值，添加到quant_op_params的qparams数组
    if (isPerChannel(qscheme)) {
      Value* ch_axis_value = g->insertGetAttr(observer_module_value, "ch_axis");
      quant_op_params.qparams.push_back(ch_axis_value);
    }
    // 获取观察器模块的dtype属性值，添加到quant_op_params的qparams数组
    Value* scalar_type_value = g->insertGetAttr(observer_module_value, "dtype");
    quant_op_params.qparams.push_back(scalar_type_value);
    // 返回quant_op_params对象
    return quant_op_params;
  }



// 结束 InsertQuantDeQuantHelper 类中的一个函数

void InsertQuantDeQuantHelper::insertCalculateQParamsAndQuantizationOps(
    Module& module,
    Graph* graph,
    Value* self) {
  // 检查当前图是否有关联的观察节点，若没有则返回
  if (!observer_nodes_for_graph_.count(graph)) {
    return;
  }
  // 遍历当前图中的观察节点
  for (auto* n : observer_nodes_for_graph_.at(graph)) {
    // 获取节点所属的图
    Graph* g = n->owningGraph();
    // 获取观察节点的输出值
    Value* observer_out = n->output();
    // 在观察节点的下一个节点之前插入操作点
    WithInsertPoint insert_qparams_calc(observer_out->node()->next());
    // 调用 insertCalculateQParams 方法计算量化参数
    auto quant_op_params = insertCalculateQParams(module, g, n);
    // 插入量化操作
    insertQuantizationOps(
        module,
        self,
        n,
        isPerChannel(quant_op_params.qscheme),
        quant_op_params,
        quant_type_);
  }
}

void InsertQuantDeQuantHelper::runForOnDevicePTQ(
    Module& module,
    const std::string& method_name) {
  // 大多数情况下，这里不会执行任何操作，因为我们预期量化准备步骤中的输入方法将被内联。
  // 因此，只有调用观察器前向调用的方法才会被执行。
  for (auto& invoked_methods : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_methods);
    const auto& invoked_method_name = std::get<1>(invoked_methods);
    runForOnDevicePTQ(invoked_module, invoked_method_name);
  }

  // 获取指定方法名的 Method 对象
  Method method = module.get_method(method_name);
  // 获取方法对应的计算图
  auto graph = method.graph();
  // 与 run 方法不同，我们不需要为不同调用位置中相同的图提取新的量化参数值。
  // 原因在于，对于设备上的 PTQ，我们不会：
  // 1. 运行 calculate_qparams
  // 2. 获取尺度和零点
  // 3. 获取轴和数据类型
  // 4. 将 2 和 3 中的注册值作为父模块的属性。
  // 相反，我们通过 insertCalculateQParams 在图中插入调用 calculate_qparams（1）。
  // 然后，我们不使用 2 和 3，而是获取输出值*，对于 3，我们插入 GetAttr 获取轴和数据类型，并使用这些值* 进行 insterQuantizationOps。

  // prim::Param 节点不属于图。因此，插入点是图节点的开头。这也防止因某些原位操作而观察到潜在变异的值。
  std::vector<Value*> input_values;
  // 遍历图的输入值
  for (const auto idx : c10::irange(1, method.num_inputs())) {
    auto& v = graph->inputs()[idx];
    // 如果值的类型是 TensorType 的子类型，则将其添加到输入值列表中
    if (v->type()->isSubtypeOf(*TensorType::get())) {
      input_values.push_back(v);
    }
  }

  // 遍历需要访问的块栈
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    // 遍历块中的节点
    for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end;) {
      Node* n = *it++;
      // 对节点的输出值进行收集观察节点和需要量化的值
      for (Value* v : n->outputs()) {
        if (!v->type()->isSubtypeOf(*TensorType::get())) {
          continue;
        }
        collectObserverNodesAndValueToQuantize(module, v);
      }

      // 将子块添加到需要访问的块栈中
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
  }
}

// 对输入值列表中的每个值进行遍历
for (Value* v : input_values) {
  // 调用函数，收集观察节点和需要量化的值
  collectObserverNodesAndValueToQuantize(module, v);
}

// 输出当前图形的调试信息，展示在插入计算量化参数和量化操作之前的状态
GRAPH_DUMP("Before insertCalculateQparamsAndQuantizationOps:", graph);

// 获取当前图形的第一个输入作为self指针
Value* self = graph->inputs()[0];

// 插入计算量化参数和量化操作到图形中
insertCalculateQParamsAndQuantizationOps(module, graph.get(), self);

// 输出当前图形的调试信息，展示在插入计算量化参数和量化操作之后的状态
GRAPH_DUMP("After insertCalculateQparamsAndQuantizationOps:", graph);
}

} // namespace

// 复制量化节点
void ReplicateQuant(std::shared_ptr<Graph>& graph) {
  // 待访问的块栈，初始为整个图的根块
  std::stack<Block*> blocks_to_visit;
  // 待重写的量化节点列表
  std::vector<Node*> quant_nodes_to_rewrite;
  blocks_to_visit.push(graph->block());

  // 遍历图中的所有块和节点
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // 查找量化节点，其输入为 prim::If 的输出
      if ((n->kind() == Symbol::aten("quantize_per_tensor") ||
           n->kind() == Symbol::aten("quantize_per_channel")) &&
          n->input(0)->node()->kind() == prim::If) {
        quant_nodes_to_rewrite.push_back(n);
      }
      // 将节点的子块加入待访问栈
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  // 对每个待重写的量化节点进行处理
  for (Node* n : quant_nodes_to_rewrite) {
    Node* if_node = n->input(0)->node();
    // 将产生量化参数的节点移到 prim::If 节点之前
    for (const auto i : c10::irange(1, n->inputs().size())) {
      n->input(i)->node()->moveBefore(if_node);
    }
    // 将量化节点的输出替换为 prim::If 节点的输出
    n->output()->replaceAllUsesWith(if_node->output());
    // 将量化节点添加到所有块的末尾
    for (Block* if_block : if_node->blocks()) {
      TORCH_CHECK(
          if_block->outputs().size() == 1,
          "replicate quantize only works for `if` node with one output right now");
      Value* ret_val = if_block->outputs()[0]; // 块的原始返回值
      std::vector<Value*> quantize_inputs = n->inputs().vec();
      quantize_inputs[0] = ret_val;
      // 在块的返回节点之前插入量化节点
      WithInsertPoint ins(if_block->return_node());
      Node* quant = graph->create(n->kind(), quantize_inputs);
      if_block->replaceOutput(0, quant->output());
      quant->output()->copyMetadata(ret_val);
      graph->insertNode(quant);
    }
  }

  // 清理量化节点的输入和销毁节点
  for (Node* n : quant_nodes_to_rewrite) {
    n->removeAllInputs();
  }
  for (Node* n : quant_nodes_to_rewrite) {
    n->destroy();
  }
}

// 复制去量化节点
void ReplicateDeQuant(std::shared_ptr<Graph>& graph) {
  std::stack<Block*> blocks_to_visit;
  std::vector<Node*> dequant_nodes_to_rewrite;
  blocks_to_visit.push(graph->block());

  // 遍历图中的所有块和节点
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // 找到去量化节点，并且其输出被多次使用
      if (n->kind() == Symbol::aten("dequantize") &&
          n->output()->uses().size() > 1) {
        dequant_nodes_to_rewrite.push_back(n);
      }
      // 将节点的子块加入待访问栈
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  // 对每个待重写的去量化节点进行处理
  for (Node* n : dequant_nodes_to_rewrite) {
    auto* quantized_val = n->input(0);
    auto* dequantized_val = n->output();
    // 在所有使用点插入去量化操作
    insertDeQuantForAllUse(graph.get(), quantized_val, dequantized_val);
  }

  // 清理去量化节点的输入和销毁节点
  for (Node* n : dequant_nodes_to_rewrite) {
    n->removeAllInputs();
  }

  for (Node* n : dequant_nodes_to_rewrite) {
    n->destroy();
  }
}

// 插入量化和去量化操作到模块
Module InsertQuantDeQuant(
    Module& input_module,
    const std::string& method_name,
    // 使用给定参数创建新的模块副本，视情况是否在原地进行操作
    Module module = input_module.clone(inplace);

    // 创建量化和反量化操作的辅助类实例，根据给定的量化类型和调试标志初始化
    InsertQuantDeQuantHelper h(quant_type, debug);

    // 运行权重观察器，将量化相关的操作应用到模块的权重参数上
    h.runWeightObserver(module, method_name);

    // 对模块应用量化和反量化操作，可能会修改模块的结构
    h.run(module, method_name);

    // 清理辅助类的状态或临时数据，确保不再需要的资源得到释放
    h.cleanup(module);

    // 在模块中传播量化操作，确保整个模块的一致性
    h.propagateQuantizationOps(module);

    // 返回经过量化操作后的模块副本
    return module;
// 定义名为 InsertQuantDeQuantOnDevicePTQ 的函数模块，用于在给定模块中的特定方法上插入量化和反量化节点
Module InsertQuantDeQuantOnDevicePTQ(
    Module& input_module,                            // 输入模块的引用
    const std::string& method_name,                  // 方法名
    bool inplace,                                    // 是否原地修改
    bool debug,                                      // 是否调试模式
    QuantType quant_type) {                           // 量化类型参数
  Module module = input_module.clone(inplace);       // 克隆输入模块
  const std::string kObserveString = "observe_";
  const auto matched_pos = method_name.find(kObserveString);  // 查找方法名中 "observe_" 的位置
  const auto end_pos = matched_pos + kObserveString.length();
  const std::string orig_method_name = method_name.substr(end_pos);  // 提取原始方法名
  TORCH_CHECK(
      matched_pos == 0,
      "Quant dequant nodes can only be added to observe_",      // 检查方法名是否以 "observe_" 开头
      orig_method_name,
      ". Please make sure to run prepare step for on-device PTQ.");

  std::string quantize_method_name = "quantize_" + orig_method_name;  // 构建量化方法名
  cloneMethod(module, method_name, quantize_method_name);             // 克隆方法以添加量化步骤
  InsertQuantDeQuantHelper h(quant_type, debug);                      // 创建帮助类对象
  h.runForOnDevicePTQ(module, quantize_method_name);                  // 执行量化节点插入操作
  h.removeObserverNodes(module);                                     // 移除观察节点
  // 下面的注释描述了不需要执行的函数，因为这些功能对于当前情况不适用
  // 不需要执行 ReplicateChooseQParamsQuantDequant: 这会传播动态量化的量化反量化操作
  // RemoveRedundantQuantizationOps: 这会移除与动态量化不相关的操作的激活观察器
  // 在我们的情况下，这些都不会发生，因为动态量化时不会观察到激活值
  // 尽管如此，仍然可以使用这个函数，因为上述两个方法实际上应该是空操作
  h.propagateQuantizationOps(module);                                 // 传播量化操作
  return module;                                                      // 返回修改后的模块
}
```