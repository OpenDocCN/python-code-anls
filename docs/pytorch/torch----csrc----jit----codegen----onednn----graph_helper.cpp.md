# `.\pytorch\torch\csrc\jit\codegen\onednn\graph_helper.cpp`

```py
// 引入 Torch 库中的相关头文件
#include <torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h>
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>

// 引入 ATen 库中的相关头文件
#include <ATen/core/functional.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// 定义命名空间，用于避免命名冲突
namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 定义一个别名 opkind 为 dnnl::graph::op::kind
using opkind = dnnl::graph::op::kind;

// 修复卷积操作中的可选偏置项，如果偏置不存在，则替换为常量 None
static void fixConvOptionalBias(Node* node) {
  if (node->namedInput("bias")->mustNotBeNone() == false) {
    auto g = node->owningGraph();
    auto n = g->createNone(); // 创建一个空值节点
    auto v = n->insertBefore(node)->output(); // 在当前节点之前插入空值节点
    node->replaceInput(2, v); // 替换第三个输入为新创建的空值节点的输出
  }
}

// 获取张量的维度信息，如果不是张量类型则返回空值
static std::optional<size_t> getDimensions(Value* v) {
  if (v->type()->isSubtypeOf(TensorType::get())) {
    return v->type()->cast<TensorType>()->sizes().size(); // 返回张量的维度数量
  } else {
    return c10::nullopt; // 返回空值表示无法获取维度信息
  }
}

// 创建一个 Wildcard 操作符，用于不能直接映射到 oneDNN Graph 的 PyTorch 操作
// 这些操作会直接传递给 oneDNN Graph 库的 add_op 调用
static Operator makeWildcardOp(Node* node) {
  auto o = Operator(node, opkind::Wildcard); // 创建一个 Wildcard 类型的操作符
  for (size_t i = 0; i < node->inputs().size(); i++) {
    o.setInput(0, i); // 设置输入操作数，使用默认值 0
  }
  for (size_t i = 0; i < node->outputs().size(); i++) {
    o.setOutput(i); // 设置输出操作数
  }
  return o; // 返回创建的 Wildcard 操作符
}

// 如果条件不满足，则创建一个 Wildcard 操作符，否则映射为指定的一DNN Graph 操作符
#define REQUIRE(cond)                                 \
  if (!(cond)) {                                      \
    GRAPH_DEBUG("Unsupported condition " #cond "\n"); \
    return makeWildcardOp(node);                      \
  }

// 创建一个元素级操作符，将 PyTorch 操作映射为对应的 oneDNN Graph 操作
Operator LlgaGraphHelper::makeEltwiseOp(Node* node, opkind kind) {
  return Operator(node, kind).setInput(0).setOutput(dnnl_graph_, 0); // 设置输入和输出操作数
}

// 创建一个二元操作符，将 PyTorch 操作映射为对应的 oneDNN Graph 操作
Operator LlgaGraphHelper::makeBinaryOp(Node* node, opkind kind) {
  REQUIRE(
      node->input(0)->type()->isSubtypeOf(TensorType::get()) &&
      node->input(1)->type()->isSubtypeOf(TensorType::get())) // 检查输入是否为张量类型
  return Operator(node, kind).setInput(0, 1).setOutput(dnnl_graph_, 0); // 设置输入和输出操作数
}

// 将 PyTorch 操作映射为对应的 oneDNN Graph 操作，如果无法映射则创建一个 Wildcard 操作
// 映射规则参照 third_party/ideep/mkl-dnn/src/interface/op_def.hpp 中定义的 oneDNN Graph 操作模式
Operator LlgaGraphHelper::createOperator(Node* node) {
  auto nodeKind = node->kind();
  // 使用 if-else 语句而不是 switch 语句，因为未来可能会添加具有函数模式的自定义操作。
  // 在那时，我们无论如何都需要使用 Symbol::fromQualString，
  // 但由于这段代码不在热路径中，我们对这个选择感到满意。
  // 检查节点类型是否为 "aten::conv2d"
  if (nodeKind == Symbol::fromQualString("aten::conv2d")) {
    // 修复卷积操作中的可选偏置参数
    fixConvOptionalBias(node);
    // 创建卷积操作符并设置属性
    return Operator(node, opkind::Convolution)
        .setInput(0, 1, 2)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::strides, Operator::Ints, 3)
        .setAttr(dnnl::graph::op::attr::pads_begin, Operator::Ints, 4)
        .setAttr(dnnl::graph::op::attr::pads_end, Operator::Ints, 4)
        .setAttr(dnnl::graph::op::attr::dilations, Operator::Ints, 5)
        .setAttr(dnnl::graph::op::attr::groups, Operator::Int, 6)
        .setAttr(dnnl::graph::op::attr::weights_format, std::string("OIX"))
        .setAttr(dnnl::graph::op::attr::data_format, std::string("NCX"));
  } else if (
      // 检查节点类型是否为 "aten::_convolution" 或 "aten::convolution"
      (nodeKind == Symbol::fromQualString("aten::_convolution")) ||
      (nodeKind == Symbol::fromQualString("aten::convolution"))) {
    // 检查是否存在转置标志，要求转置操作不存在
    bool transposed = toIValue(node->namedInput("transposed"))->toBool();
    REQUIRE(!transposed);
    // 创建卷积操作符并设置属性
    return Operator(node, opkind::Convolution)
        .setInput(0, 1, 2)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::strides, Operator::Ints, 3)
        .setAttr(dnnl::graph::op::attr::pads_begin, Operator::Ints, 4)
        .setAttr(dnnl::graph::op::attr::pads_end, Operator::Ints, 4)
        .setAttr(dnnl::graph::op::attr::dilations, Operator::Ints, 5)
        .setAttr(dnnl::graph::op::attr::groups, Operator::Int, 8)
        .setAttr(dnnl::graph::op::attr::weights_format, std::string("OIX"))
        .setAttr(dnnl::graph::op::attr::data_format, std::string("NCX"));
  } else if (nodeKind == Symbol::fromQualString("aten::batch_norm")) {
    // 获取是否处于训练模式
    auto training = toIValue(node->namedInput("training"));
    REQUIRE(training.has_value()); // 在脚本模式中无法获取训练状态
    // 如果不处于训练模式，则创建批归一化推断操作符并设置属性
    if (!training->toBool()) {
      return Operator(node, opkind::BatchNormInference)
          .setInput(0, 1, 2, 3, 4)
          .setOutput(dnnl_graph_, 0)
          .setAttr(dnnl::graph::op::attr::epsilon, Operator::Float, 7)
          .setAttr(dnnl::graph::op::attr::data_format, std::string("NCX"));
    }
  } else if (nodeKind == Symbol::fromQualString("aten::layer_norm")) {
    // 获取层归一化的标准化形状
    auto normalized_shape = toIValue(node->namedInput("normalized_shape"));
    REQUIRE(normalized_shape->toIntList().size() == 1); // 确保标准化形状为单一维度
    // 创建层归一化操作符并设置属性
    return Operator(node, opkind::LayerNorm)
        .setInput(0, 2, 3)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::epsilon, Operator::Float, 4)
        .setAttr(dnnl::graph::op::attr::keep_stats, false);
  } else if (nodeKind == Symbol::fromQualString("aten::addmm")) {
    // 获取 alpha 参数
    auto alpha = toIValue(node->namedInput("alpha"));
    // 从节点中获取名为 "beta" 的命名输入，并转换为对应的 IValue 类型
    auto beta = toIValue(node->namedInput("beta"));

    // 检查 alpha 和 beta 是否都有值
    if (alpha.has_value() && beta.has_value()) {
      // 如果 alpha 和 beta 都等于 1.0
      if ((alpha->toDouble() == 1.0) && (beta->toDouble() == 1.0)) {
        // 创建一个 MatMul 运算符，设置输入和输出
        return Operator(node, opkind::MatMul)
            .setInput(1, 2, 0)
            .setOutput(dnnl_graph_, 0);
      } else if ((alpha->toDouble() == 1.0) && (beta->toDouble() == 0.0)) {
        // 如果 alpha 等于 1.0 而 beta 等于 0.0
        // 创建一个 MatMul 运算符，设置输入和输出
        return Operator(node, opkind::MatMul)
            .setInput(1, 2)
            .setOutput(dnnl_graph_, 0);
      }
    }
  } else if (nodeKind == Symbol::fromQualString("aten::add"))
    // 如果节点类型是 "aten::add"，返回一个加法运算的二元操作符
    return makeBinaryOp(node, opkind::Add);
  else if (nodeKind == Symbol::fromQualString("aten::mul"))
    // 如果节点类型是 "aten::mul"，返回一个乘法运算的二元操作符
    return makeBinaryOp(node, opkind::Multiply);
  else if (nodeKind == Symbol::fromQualString("aten::div"))
    // 如果节点类型是 "aten::div"，返回一个除法运算的二元操作符
    return makeBinaryOp(node, opkind::Divide);
  else if (nodeKind == Symbol::fromQualString("aten::tanh"))
    // 如果节点类型是 "aten::tanh"，返回一个双曲正切运算的元素级操作符
    return makeEltwiseOp(node, opkind::Tanh);
  else if (nodeKind == Symbol::fromQualString("aten::relu"))
    // 如果节点类型是 "aten::relu"，返回一个ReLU激活函数的元素级操作符
    return makeEltwiseOp(node, opkind::ReLU);
  else if (nodeKind == Symbol::fromQualString("aten::elu"))
    // 如果节点类型是 "aten::elu"，返回一个ELU激活函数的元素级操作符，并设置 alpha 属性为 1.0
    return makeEltwiseOp(node, opkind::Elu)
        .setAttr(dnnl::graph::op::attr::alpha, Operator::Float, 1);
  else if (nodeKind == Symbol::fromQualString("aten::sigmoid"))
    // 如果节点类型是 "aten::sigmoid"，返回一个sigmoid激活函数的元素级操作符
    return makeEltwiseOp(node, opkind::Sigmoid);
  else if (nodeKind == Symbol::fromQualString("aten::gelu"))
    // 如果节点类型是 "aten::gelu"，返回一个GELU激活函数的元素级操作符
    return makeEltwiseOp(node, opkind::GELU);
  else if (nodeKind == Symbol::fromQualString("aten::round"))
    // 如果节点类型是 "aten::round"，返回一个取整操作的元素级操作符
    return makeEltwiseOp(node, opkind::Round);
  else if (nodeKind == Symbol::fromQualString("aten::exp"))
    // 如果节点类型是 "aten::exp"，返回一个指数函数的元素级操作符
    return makeEltwiseOp(node, opkind::Exp);
  else if (nodeKind == Symbol::fromQualString("aten::sqrt"))
    // 如果节点类型是 "aten::sqrt"，返回一个平方根函数的元素级操作符
    return makeEltwiseOp(node, opkind::Sqrt);
  else if (nodeKind == Symbol::fromQualString("aten::abs"))
    // 如果节点类型是 "aten::abs"，返回一个绝对值函数的元素级操作符
    return makeEltwiseOp(node, opkind::Abs);
  else if (nodeKind == Symbol::fromQualString("aten::square"))
    // 如果节点类型是 "aten::square"，返回一个平方函数的元素级操作符
    return makeEltwiseOp(node, opkind::Square);
  else if (nodeKind == Symbol::fromQualString("aten::clamp")) {
    // 如果节点类型是 "aten::clamp"
    // PyTorch API 已经检查了 min 和 max 不为 None，但我们也进行了检查
    auto clamp_min = toIValue(node->input(1));
    auto clamp_max = toIValue(node->input(2));
    // 确保 clamp_min 和 clamp_max 至少有一个不是 None
    REQUIRE(!(clamp_max->isNone() && clamp_min->isNone()));
    // 如果 clamp_min 是 None，则设置为负无穷大；否则，将其转换为浮点数
    auto clamp_min_value = (clamp_min->isNone())
        ? -std::numeric_limits<float>::infinity()
        : Operator::ScalarToFloat(node, 1);
    // 如果 clamp_max 是 None，则设置为正无穷大；否则，将其转换为浮点数
    auto clamp_max_value = (clamp_max->isNone())
        ? std::numeric_limits<float>::infinity()
        : Operator::ScalarToFloat(node, 2);
    // 创建一个 clamp 操作符，设置最小值和最大值属性
    return makeEltwiseOp(node, opkind::Clamp)
        .setAttr(dnnl::graph::op::attr::min, clamp_min_value)
        .setAttr(dnnl::graph::op::attr::max, clamp_max_value);
  } else if (nodeKind == Symbol::fromQualString("aten::hardtanh")) {
    // 如果节点类型是 "aten::clamp"，创建一个按元素操作的 Clamp 运算符
    return makeEltwiseOp(node, opkind::Clamp)
        // 设置运算符的属性 min，将 Operator::ScalarToFloat 转换为浮点数 1
        .setAttr(dnnl::graph::op::attr::min, Operator::ScalarToFloat, 1)
        // 设置运算符的属性 max，将 Operator::ScalarToFloat 转换为浮点数 2
        .setAttr(dnnl::graph::op::attr::max, Operator::ScalarToFloat, 2);
  } else if (nodeKind == Symbol::fromQualString("aten::hardswish")) {
    // 如果节点类型是 "aten::hardswish"，创建一个按元素操作的 HardSwish 运算符
    return makeEltwiseOp(node, opkind::HardSwish);
  } else if (nodeKind == Symbol::fromQualString("aten::log")) {
    // 如果节点类型是 "aten::log"，创建一个按元素操作的 Log 运算符
    return makeEltwiseOp(node, opkind::Log);
  } else if (nodeKind == Symbol::fromQualString("aten::leaky_relu")) {
    // 如果节点类型是 "aten::leaky_relu"，创建一个按元素操作的 LeakyReLU 运算符
    // 设置运算符的属性 alpha 为 Operator::Float 类型的 1
    return makeEltwiseOp(node, opkind::LeakyReLU)
        .setAttr(dnnl::graph::op::attr::alpha, Operator::Float, 1);
  } else if (nodeKind == Symbol::fromQualString("aten::relu6")) {
    // 如果节点类型是 "aten::relu6"，创建一个按元素操作的 Clamp 运算符，限制在 [0, 6] 范围内
    return makeEltwiseOp(node, opkind::Clamp)
        .setAttr(dnnl::graph::op::attr::min, 0.f)  // 设置最小值为 0
        .setAttr(dnnl::graph::op::attr::max, 6.f); // 设置最大值为 6
  } else if (
      (nodeKind == Symbol::fromQualString("aten::softmax")) ||
      (nodeKind == Symbol::fromQualString("aten::_softmax"))) {
    // 如果节点类型是 "aten::softmax" 或 "aten::_softmax"
    // 从节点的 "dim" 输入中获取轴的值，并创建 SoftMax 运算符
    auto axis = toIValue(node->namedInput("dim"))->toInt();
    return Operator(node, opkind::SoftMax)
        .setInput(0)                             // 设置输入端口为 0
        .setOutput(dnnl_graph_, 0)                // 设置输出端口为 dnnl_graph_ 的 0 号端口
        .setAttr(dnnl::graph::op::attr::axis, axis); // 设置轴属性为获取的轴值
  } else if (nodeKind == Symbol::fromQualString("aten::_log_softmax")) {
    // 如果节点类型是 "aten::_log_softmax"
    // 从节点的 "dim" 输入中获取轴的值，并创建 LogSoftmax 运算符
    auto axis = toIValue(node->namedInput("dim"))->toInt();
    return Operator(node, opkind::LogSoftmax)
        .setInput(0)                             // 设置输入端口为 0
        .setOutput(dnnl_graph_, 0)                // 设置输出端口为 dnnl_graph_ 的 0 号端口
        .setAttr(dnnl::graph::op::attr::axis, axis); // 设置轴属性为获取的轴值
  } else if (nodeKind == Symbol::fromQualString("aten::cat")) {
    // 如果节点类型是 "aten::cat"
    auto o = Operator(node, opkind::Concat);     // 创建一个 Concat 运算符
    // 要求："tensors" 输入的节点类型必须是 prim::ListConstruct
    REQUIRE(node->namedInput("tensors")->node()->kind() == prim::ListConstruct);
    // 要求："tensors" 输入的使用次数必须为 1
    REQUIRE(node->namedInput("tensors")->uses().size() == 1);
    // 要求："dim" 输入的节点类型必须是 prim::Constant
    REQUIRE(node->namedInput("dim")->node()->kind() == prim::Constant);
    // aten::cat 需要特殊处理，因为它接受一个 Tensor[] 作为输入。
    // 将 ListConstruct 的输入设置为 cat 的输入。
    auto listConstruct = node->input(0)->node();
    for (auto input : listConstruct->inputs())
      o.setInputValue(input);                    // 设置输入值
    return o.setOutput(dnnl_graph_, 0)           // 设置输出端口为 dnnl_graph_ 的 0 号端口
        .setAttr(dnnl::graph::op::attr::axis, Operator::Int, 1); // 设置轴属性为整数 1
  } else if (
      (nodeKind == Symbol::fromQualString("aten::max_pool2d")) ||
      (nodeKind == Symbol::fromQualString("aten::max_pool2d_with_indices"))) {
    // 如果节点类型是 "aten::max_pool2d" 或 "aten::max_pool2d_with_indices"
    // 当前，LLGA 不支持创建索引掩码。
    // 一旦支持，max_pool2d_with_indices 应该有不同的映射方式。
    // 要求：kernel_size 输入的节点类型必须是 prim::Constant
    REQUIRE(node->namedInput("kernel_size")->node()->kind() == prim::Constant);
    // 获取 ceil_mode 输入值，如果为 true，则 rounding_type 设为 "ceil"，否则为 "floor"
    auto rounding_type =
        toIValue(node->namedInput("ceil_mode"))->toBool() ? "ceil" : "floor";
    // 如果节点的操作类型是 "aten::max_pool2d"
    if (nodeKind == Symbol::fromQualString("aten::max_pool2d")) {
        // 创建一个 MaxPool 操作符，并设置输入输出
        return Operator(node, opkind::MaxPool)
            .setInput(0)
            .setOutput(dnnl_graph_, 0)
            // 设置内核大小为 1
            .setAttr(dnnl::graph::op::attr::kernel, Operator::Ints, 1)
            // 设置步长为 2
            .setAttr(dnnl::graph::op::attr::strides, Operator::Ints, 2)
            // 设置填充开始和结束为 3
            .setAttr(dnnl::graph::op::attr::pads_begin, Operator::Ints, 3)
            .setAttr(dnnl::graph::op::attr::pads_end, Operator::Ints, 3)
            // 设置膨胀率为 4
            .setAttr(dnnl::graph::op::attr::dilations, Operator::Ints, 4)
            // 设置取整方式为 ceil 或 floor
            .setAttr(
                dnnl::graph::op::attr::rounding_type, std::string(rounding_type))
            // 设置数据格式为 "NCX"
            .setAttr(dnnl::graph::op::attr::data_format, std::string("NCX"));
    } else if (nodeKind == Symbol::fromQualString("aten::avg_pool2d")) {
        // 对于操作类型 "aten::avg_pool2d"
        // 检查是否需要对所有常量进行检查
        REQUIRE(node->namedInput("kernel_size")->node()->kind() == prim::Constant);
        // 根据 ceil_mode 参数选择取整方式为 "ceil" 或 "floor"
        auto rounding_type =
            toIValue(node->namedInput("ceil_mode"))->toBool() ? "ceil" : "floor";
        auto divisor_override = toIValue(node->namedInput("divisor_override"));
        // 要求 divisor_override 必须为空
        REQUIRE(divisor_override->isNone());
        // 创建一个 AvgPool 操作符，并设置输入输出
        return Operator(node, opkind::AvgPool)
            .setInput(0)
            .setOutput(dnnl_graph_, 0)
            // 设置内核大小为 1
            .setAttr(dnnl::graph::op::attr::kernel, Operator::Ints, 1)
            // 设置步长为 2
            .setAttr(dnnl::graph::op::attr::strides, Operator::Ints, 2)
            // 设置填充开始和结束为 3
            .setAttr(dnnl::graph::op::attr::pads_begin, Operator::Ints, 3)
            .setAttr(dnnl::graph::op::attr::pads_end, Operator::Ints, 3)
            // 设置排除填充的方式为对应的布尔值
            .setAttr(dnnl::graph::op::attr::exclude_pad, !Operator::Bool(node, 5))
            // 设置取整方式为 ceil 或 floor
            .setAttr(
                dnnl::graph::op::attr::rounding_type, std::string(rounding_type))
            // 设置数据格式为 "NCX"
            .setAttr(dnnl::graph::op::attr::data_format, std::string("NCX"));
    } else if (nodeKind == Symbol::fromQualString("aten::matmul")) {
        // 对于操作类型 "aten::matmul"
        auto dim0 = getDimensions(node->namedInput("self")).value_or(-1);
        auto dim1 = getDimensions(node->namedInput("other")).value_or(-1);
        // TODO: 支持所有形状的组合
        // 要求维度为 (2,2) 或 (4,4) 或 (3,2)
        REQUIRE(
            (dim0 == 2 && dim1 == 2) || (dim0 == 4 && dim1 == 4) ||
            (dim0 == 3 && dim1 == 2));
        // 创建一个 MatMul 操作符，并设置输入输出
        return Operator(node, opkind::MatMul)
            .setInput(0, 1)
            .setOutput(dnnl_graph_, 0);
    } // 继续执行下面的判断条件
    else if (nodeKind == Symbol::fromQualString("aten::mm")) {
        // 对于操作类型 "aten::mm"
        // 创建一个 MatMul 操作符，并设置输入输出
        return Operator(node, opkind::MatMul)
            .setInput(0, 1)
            .setOutput(dnnl_graph_, 0);
    } else if (nodeKind == Symbol::fromQualString("aten::bmm")) {
        // 对于操作类型 "aten::bmm"
        // 创建一个 MatMul 操作符，并设置输入输出
        return Operator(node, opkind::MatMul)
            .setInput(0, 1)
            .setOutput(dnnl_graph_, 0);
    } else if (nodeKind == Symbol::fromQualString("aten::linear")) {
        // 对于操作类型 "aten::linear"
        // 创建一个 MatMul 操作符，并设置输入输出及属性 transpose_b 为 true
        return Operator(node, opkind::MatMul)
            .setInput(0, 1, 2)
            .setOutput(dnnl_graph_, 0)
            .setAttr(dnnl::graph::op::attr::transpose_b, true);
    } else if (nodeKind == Symbol::fromQualString("aten::permute")) {
        // 对于操作类型 "aten::permute"
        // 要求 aliasDb_ 不允许有输入写操作
        REQUIRE(aliasDb_->hasInputWriters(node) == false);
    // 如果节点类型是静态转置操作
    if (nodeKind == Symbol::fromQualString("aten::static_transpose")) {
        // 创建一个 Operator 对象，表示静态转置操作，并设置操作类型为 StaticTranspose
        return Operator(node, opkind::StaticTranspose)
            // 设置输入节点为当前节点的第一个输入
            .setInput(0)
            // 设置输出节点为 dnnl_graph_ 的第一个输出
            .setOutput(dnnl_graph_, 0)
            // 设置属性 order，将节点的名为 "dims" 的命名输入转换为整数向量
            .setAttr(
                dnnl::graph::op::attr::order,
                toIValue(node->namedInput("dims"))->toIntVector());
    } else if (nodeKind == Symbol::fromQualString("aten::contiguous")) {
        // 如果节点类型是 contiguous 操作
        // contiguous 操作只有在目标内存布局与源内存格式不同时才会映射到 oneDNN 图中
        // 此时步幅不同，但形状相同
        auto typeOfInput = node->input(0)->type()->expect<TensorType>();
        auto typeOfOutput = node->output(0)->type()->expect<TensorType>();
        auto inputStrides = typeOfInput->strides().concrete_sizes();
        auto outputStrides = typeOfOutput->strides().concrete_sizes();
        // 断言输入的步幅与输出的步幅不相等
        REQUIRE(inputStrides != outputStrides);
        // 创建一个 Operator 对象，表示重新排序操作，并设置操作类型为 Reorder
        return Operator(node, opkind::Reorder)
            // 设置输入节点为当前节点的第一个输入
            .setInput(0)
            // 设置输出节点为 dnnl_graph_ 的第一个输出
            .setOutput(dnnl_graph_, 0);
    }
    // 如果以上条件都不满足，记录调试信息，标记该节点为通配符操作
    GRAPH_DEBUG("Making ", nodeKind.toQualString(), " a wildcard");
    // 返回一个通配符操作对象，标记当前节点为通配符操作
    return makeWildcardOp(node);
}

// 从给定的值推断设备类型
static DeviceType inferDeviceFromValue(Value* v) {
  // 尝试将值转换为张量类型
  auto tt = v->type()->cast<TensorType>();
  if (!tt) {
    // 如果值不是张量类型，则默认返回 CPU 设备类型
    return at::kCPU;
  }
  // 获取张量的设备信息
  auto device = tt->device();
  if (!device) {
    // 如果设备信息不存在，则默认返回 CPU 设备类型
    return at::kCPU;
  }
  // 返回张量的设备类型
  return device->type();
}

// 推断整个图的设备类型
static DeviceType inferDevice(const std::shared_ptr<Graph>& graph) {
  // 从图的第一个输入推断设备类型
  auto dt = inferDeviceFromValue(graph->inputs()[0]);
  // 检查所有输入是否具有相同的设备类型
  TORCH_CHECK(
      std::all_of(
          graph->inputs().begin(),
          graph->inputs().end(),
          [dt](Value* v) { return inferDeviceFromValue(v) == dt; }),
      "All inputs must have the same deive type");
  // 返回推断出的设备类型
  return dt;
}

// 根据设备类型获取相应的 oneDNN 引擎类型
static dnnl::engine::kind getLlgaEngineKind(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      // 对于 CPU 设备类型，返回 oneDNN 的 CPU 引擎类型
      return dnnl::engine::kind::cpu;
    default:
      // 对于不支持的设备类型，抛出错误
      TORCH_CHECK(false, "Not support device type ", type);
  }
}

// 如果节点是 aten::cat 并且已分配到 opToOwningPartition，则可能将 prim::ListConstruct 添加到相同分区
static void mayAddListConstructIntoConcatPartition(
    Node* n,
    OpPartitionMap& opToOwningPartition) {
  // 由于 prim::ListConstruct 对 oneDNN 不可见，
  // 它不会出现在 partfuseritioning 结果中的任何分区中。
  // 我们需要重写 opToOwningPartition，使 prim::ListConstruct 在 '虚拟' 中与 aten::cat 处于相同的分区中，
  // 以便 prim::ListConstruct 可以通过图融合器融合到融合组中。
  // 我们强调 '虚拟'，因为 cat 的分区的 get_num_ops() 仍将返回 1。
  if (n->kind() == aten::cat && opToOwningPartition.has(n)) {
    // 获取 aten::cat 的输入 'tensors' 的节点，即 prim::ListConstruct 节点
    auto listConstrcut = n->namedInput("tensors")->node();
    // 获取 aten::cat 的分区 ID
    auto partitionId = opToOwningPartition.get(n);
    // 将 prim::ListConstruct 添加到相同的分区中
    opToOwningPartition.add(listConstrcut, partitionId);
  }
}

// 验证输入张量是否与 oneDNN 图兼容
// 标量将稍后转换为 1-D 张量，但不应是复数双精度
// 如果检查失败，则将操作转换为通配符
static bool checkInputCompatibility(Node* node) {
  // 获取所有输入
  auto allInputs = node->inputs();
  for (auto input : allInputs) {
    // 将输入转换为 IValue
    c10::IValue inputIValue = toIValue(input);
    if (inputIValue.isTensor()) {
      // 如果输入是张量
      const at::Tensor& tensor = inputIValue.toTensor();
      // 检查张量的设备类型是否为 CPU
      if (tensor.device() != at::kCPU) {
        return false;
      }
      auto dtype = tensor.scalar_type();
      // 检查张量的数据类型是否在支持的范围内
      if ((dtype != at::ScalarType::BFloat16) &&
          (dtype != at::ScalarType::Float) && (dtype != at::ScalarType::Long)) {
        // 允许 Long 数据类型，尽管 oneDNN 图不支持 Long 数据类型，
        // 因为 oneDNN 图最终不处理具有 Long 数据类型输入的操作，而是由 PyTorch 处理。
        return false;
      }
    } else if (inputIValue.isScalar()) {
      // 如果输入是标量
      if (inputIValue.isComplexDouble()) {
        return false;
      }
      // 这里并未完整展示，因为代码被截断
    }
  }
  // 输入兼容性检查通过
  return true;
}
    } else if (input->type()->isSubtypeOf(TensorType::get())) {
      // 检查输入是否是 TensorType 的子类型
      auto input_typeptr = input->type()->cast<TensorType>();
      // 尝试将输入类型转换为 TensorType，并获取其指针
      if (input_typeptr->scalarType().has_value()) {
        // 检查输入的 TensorType 是否包含标量类型信息
        at::ScalarType dtype = input_typeptr->scalarType().value();
        // 获取输入的标量类型
        if ((dtype != at::ScalarType::Float) &&
            (dtype != at::ScalarType::BFloat16)) {
          // 如果标量类型不是 Float 或 BFloat16，则返回 false
          return false;
        }
      }
    }
  }
  // 如果所有输入都满足条件，则返回 true
  return true;
}

// 构造函数：初始化 LLGA 图辅助对象
LlgaGraphHelper::LlgaGraphHelper(
    const std::shared_ptr<Graph>& graph,
    dnnl::graph::partition::policy policy) {
  // 推断设备类型并获取 LLGA 引擎类型
  auto deviceType = inferDevice(graph);
  auto engineKind = getLlgaEngineKind(deviceType);
  // 创建 LLGA 图对象
  dnnl_graph_ = std::make_unique<dnnl::graph::graph>(engineKind);
  // 创建 Torch 的别名数据库对象
  aliasDb_ = std::make_unique<torch::jit::AliasDb>(graph);
  // 调试信息：正在构建 LLGA 图
  GRAPH_DEBUG("Constructing LLGA graph");
  // TODO: 目前只选择顶层块中的节点
  for (auto* node : graph->block()->nodes()) {
    auto kindOfNode = node->kind();
    // 调试信息：尝试添加节点，显示节点的类型
    GRAPH_DEBUG("Trying to add ", kindOfNode.toQualString());
    // 检查节点的输入兼容性
    if (checkInputCompatibility(node)) {
      // 创建节点的操作符并添加到 LLGA 图中
      auto op = createOperator(node);
      dnnl_graph_->add_op(op.llgaOp());
      // 调试信息：已添加节点，显示节点的类型
      GRAPH_DEBUG("  Added node ", kindOfNode.toQualString());
    } else {
      // 调试信息：节点的输入不兼容
      GRAPH_DEBUG("Incompatible inputs for ", kindOfNode.toQualString());
      // 添加通配符操作到 LLGA 图中
      dnnl_graph_->add_op(makeWildcardOp(node).llgaOp());
    }

    // 将节点的输入值映射到其唯一 ID
    for (Value* input : node->inputs()) {
      tensorIdToValue_.emplace(input->unique(), input);
    }
  }

  // 完成 LLGA 图的构建
  dnnl_graph_->finalize();

  // 调试信息：获取分区信息
  GRAPH_DEBUG("Get Partitions");
  // 获取 LLGA 图的分区，根据指定的策略
  std::vector<dnnl::graph::partition> partitions =
      dnnl_graph_->get_partitions(policy);
  // 排除不支持的通配符分区
  for (auto& partition : partitions) {
    if (partition.is_supported()) {
      partitions_.push_back(partition);
    }
  }

  // 调试信息：已获取分区数量
  GRAPH_DEBUG("  Got #partitions: ", partitions_.size());
  // 将操作与其所属分区的映射关系添加到表中
  for (size_t partId = 0; partId < partitions_.size(); partId++) {
    for (auto opId : partitions_[partId].get_ops()) {
      opToOwningPartition_.add(opId, partId);
    }
  }

  // 再次扫描图进行后处理
  for (auto* node : graph->block()->nodes()) {
    // 尝试将 ListConstruct 节点合并到 Concat 分区中
    mayAddListConstructIntoConcatPartition(node, opToOwningPartition_);
  }
}

// 判断节点是否为 LLGA 子图
bool LlgaGraphHelper::isLlgaSubgraph(const Node* node) {
  return node->hasAttribute(attr::Subgraph) &&
      node->kind() == prim::oneDNNFusionGroup;
}

// 判断是否应将节点合并到子图中
bool LlgaGraphHelper::shouldMerge(Node* toMerge, Node* subgraph) {
  // 检查消费节点是否包含子图属性
  TORCH_CHECK(
      isLlgaSubgraph(subgraph),
      "The consumer node does not contain a subgraph");
  // 如果节点不应考虑合并，则返回 false
  if (!shouldConsiderForMerge(toMerge)) {
    return false;
  }
  // 检查节点是否属于相同的 LLGA 分区
  return opToOwningPartition_.get(toMerge) ==
      opToOwningPartition_.get(subgraph);
}

// 判断是否适合 LLGA 的节点类型（除了 conv 和 GEMM 外）
// 对于不支持 NNC 或者 oneDNN 执行速度更快的单操作分区，prim::ListConstruct 是个例外，因为我们希望将其与 cat 融合
static bool isBetterSuitedForLLGA(NodeKind kindOfOp) {
  return (
      (kindOfOp == aten::layer_norm) || (kindOfOp == aten::avg_pool2d) ||
      (kindOfOp == aten::matmul) || (kindOfOp == aten::max_pool2d) ||
      (kindOfOp == aten::conv2d) || (kindOfOp == aten::_convolution) ||
      (kindOfOp == aten::mm) || (kindOfOp == aten::linear) ||
      (kindOfOp == aten::cat) || (kindOfOp == prim::ListConstruct));
}
bool LlgaGraphHelper::checkForSingleOpPartition(Node* node) {
    // 检查节点是否有对应的操作分区
    if (opToOwningPartition_.has(node)) {
        // 获取节点所属的分区ID
        auto partitionId = opToOwningPartition_.get(node);
        // 检查该分区中操作的数量是否为1
        if (partitions_[partitionId].get_ops_num() == 1) {
            // 获取节点的类型
            auto kindOfNode = node->kind();
            // 判断节点类型是否更适合于LLGA优化
            return isBetterSuitedForLLGA(kindOfNode);
        } else {
            // 多操作分区
            return true;
        }
    } else {
        // 此操作未出现在任何分区中
        return false;
    }
}

bool LlgaGraphHelper::shouldConsiderForMerge(Node* node) {
    // 如果节点已经在合并过程中
    if (isLlgaSubgraph(node)) {
        return true;
    }
    // 否则，检查节点是否适合单操作分区合并
    return checkForSingleOpPartition(node);
}

Node* LlgaGraphHelper::createSingletonSubgraph(Node* n, AliasDb& aliasDb) {
    // 获取节点所属的分区ID
    auto partitionId = opToOwningPartition_.get(n);
    // 调试信息：创建名称为FusionGroup_partitionId的单例子图，用于节点类型n的优化
    GRAPH_DEBUG(
        "Creating FusionGroup_", partitionId, " for ", n->kind().toQualString());
    // 创建单例子图并更新别名数据库
    auto group = SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
        n, prim::oneDNNFusionGroup, aliasDb);
    // 将子图与分区ID关联
    opToOwningPartition_.add(group, partitionId);
    return group;
}

void LlgaGraphHelper::mergeNodeIntoSubgraph(
    Node* toMerge,
    Node* subgraphNode,
    AliasDb& aliasDb) {
    // 如果要合并的节点已经是LLGA子图
    if (isLlgaSubgraph(toMerge)) {
        // 调试信息：将toMerge的类型和分区ID合并到subgraphNode的类型和分区ID中
        GRAPH_DEBUG(
            "Merging ",
            toMerge->kind().toQualString(),
            "_",
            opToOwningPartition_.get(toMerge),
            " into ",
            subgraphNode->kind().toQualString(),
            "_",
            opToOwningPartition_.get(subgraphNode));
    } else {
        // 调试信息：将toMerge的类型合并到subgraphNode的类型和分区ID中
        GRAPH_DEBUG(
            "Merging ",
            toMerge->kind().toQualString(),
            " into ",
            subgraphNode->kind().toQualString(),
            "_",
            opToOwningPartition_.get(subgraphNode));
    }
    // 合并节点到子图并更新别名数据库
    SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
        toMerge, subgraphNode, aliasDb);
}

void LlgaGraphHelper::unmergeIfAnyNodeIsMissing(Node* subgraphNode) {
    // 检查节点是否是LLGA子图
    TORCH_CHECK(isLlgaSubgraph(subgraphNode), "Cannot unmerge a non-LLGA node");

    // 获取子图的分区ID和期望的操作数量
    auto partitionId = opToOwningPartition_.get(subgraphNode);
    auto expectOpNum = partitions_[partitionId].get_ops_num();
    // 计算子图实际支持的操作数量
    auto actualOpNum = countSupportedOps(subgraphNode->g(attr::Subgraph));

    // 如果期望操作数量与实际操作数量不符
    if (expectOpNum != actualOpNum) {
        // 调试信息：解除FusionGroup_partitionId的子图合并，期望expectOpNum个操作，实际得到actualOpNum个操作
        GRAPH_DEBUG(
            "Unmerging FusionGroup_",
            partitionId,
            ". Expected ",
            expectOpNum,
            " ops, but got ",
            actualOpNum,
            " ops.");
        // 解除子图的合并操作
        SubgraphUtils::unmergeSubgraph(subgraphNode);
    }
}

size_t LlgaGraphHelper::countSupportedOps(
    const std::shared_ptr<Graph>& graph) const {
    // TODO: 目前仅统计顶级块中的节点数
    size_t cnt = 0;
    for (auto* node : graph->block()->nodes()) {
        auto nodeKind = node->kind();
        // 排除常量和列表构造节点
        if ((nodeKind != prim::Constant) && (nodeKind != prim::ListConstruct)) {
            cnt++;
        }
    }
    return cnt;
}

std::vector<dnnl::graph::partition> LlgaGraphHelper::getPartitions() const {
    // 返回分区列表
    return partitions_;
}

std::map<size_t, Value*> LlgaGraphHelper::getTensorIdToValue() const {
    // 返回张量ID到值的映射
    return tensorIdToValue_;
}
// 构造函数，用于初始化一个 LlgaNodeWrapper 对象，接受一个指向常量 Node 的指针作为参数
LlgaNodeWrapper::LlgaNodeWrapper(const Node* node)
    : n(const_cast<Node*>(node)) { // NOLINT
  // 使用 TORCH_CHECK 断言确保节点 n 是一个 LLGA 子图节点，否则抛出异常
  TORCH_CHECK(
      LlgaGraphHelper::isLlgaSubgraph(n), "Cannot wrap a non-LLGA fusion node");
}

// 设置不透明布局的方法，接受一个偏移量作为参数
void LlgaNodeWrapper::setOpaqueLayout(size_t offset) {
  // 获取节点 n 的输出布局数量
  const auto num_output = n->is(attr::output_layouts).size();
  // 使用 TORCH_CHECK 断言确保偏移量在有效范围内，否则抛出异常
  TORCH_CHECK(
      offset < num_output,
      "Out of range. (Invalid index ",
      offset,
      " for attr::output_layouts with size ",
      num_output,
      ")");
  // 获取节点 n 的输出布局引用，并强制去除其常量性，以便修改
  auto& layouts =
      const_cast<std::vector<int64_t>&>(n->is(attr::output_layouts)); // NOLINT
  // 将指定偏移量的输出布局设置为 OPAQUE_LAYOUT
  layouts.at(offset) = OPAQUE_LAYOUT;
}

// 检查是否使用不透明布局的方法，接受一个偏移量作为参数
bool LlgaNodeWrapper::useOpaqueLayout(size_t offset) const {
  // 获取节点 n 的输出布局数量
  const auto num_output = n->is(attr::output_layouts).size();
  // 使用 TORCH_CHECK 断言确保偏移量在有效范围内，否则抛出异常
  TORCH_CHECK(
      offset < num_output,
      "Out of range. (Invalid index ",
      offset,
      " for attr::output_layouts with size ",
      num_output,
      ")");
  // 返回指定偏移量的输出布局是否为 OPAQUE_LAYOUT
  return n->is(attr::output_layouts)[offset] == OPAQUE_LAYOUT;
}

// 命名空间结束标记
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```