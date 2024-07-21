# `.\pytorch\torch\csrc\jit\runtime\autodiff.cpp`

```py
// 引入 Torch JIT 运行时自动微分的头文件
#include <torch/csrc/jit/runtime/autodiff.h>

// 引入必要的 ATen 和 C10 库头文件
#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 引入 Torch JIT 的日志记录功能
#include <torch/csrc/jit/jit_log.h>

// 引入 Torch JIT 中的优化和变换 Pass
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/update_differentiable_graph_requires_grad.h>

// 引入 Torch JIT 运行时操作符和符号脚本
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/symbolic_script.h>

// 引入 C++ 标准库头文件
#include <algorithm>
#include <memory>

// 使用 Torch JIT 的命名空间
namespace torch::jit {

// 定义两个类型别名，用于值映射和值集合
using value_map = std::unordered_map<Value*, Value*>;
using value_set = std::unordered_set<Value*>;

// 判断是否需要修剪梯度的函数，用于处理某些操作的多输出情况
static bool needTrimGrad(Node* n) {
  // 预定义需要修剪梯度的操作符集合
  static OperatorSet need_trim_grad_ops = {
      "aten::kthvalue(Tensor self, int k, int dim, bool keepdim) -> (Tensor, Tensor)",
      "aten::topk(Tensor self, int k, int dim, bool largest, bool sorted) -> (Tensor, Tensor)",
      "aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
      "aten::max_pool2d_with_indices(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> (Tensor, Tensor)"
  };
  
  // 检查节点是否属于预定义的需要修剪梯度的操作集合
  if (n->isMemberOf(need_trim_grad_ops)) {
    return true;
  }
  return false;
}

// 判断节点是否可微分的函数
bool isDifferentiable(const Node* n) {
  // TODO: 标量-张量操作应该被规范化

  // 预定义可微分操作符集合
  static OperatorSet differentiable_ops = {
      "aten::_slow_conv2d_forward(Tensor self, Tensor weight, int[] kernel_size, Tensor? bias, int[] stride, int[] padding) -> Tensor",
      "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"
  };

  // 检查节点是否为常量、自动求导零、自动求导加法或常量分块
  if (n->kind() == prim::Constant || n->kind() == prim::AutogradZero ||
      n->kind() == prim::AutogradAdd || n->kind() == prim::ConstantChunk ||
      n->kind() == prim::profile || n->kind() == prim::profile_ivalue)
    return true;

  // 检查节点是否属于预定义的可微分操作符集合
  if (n->isMemberOf(differentiable_ops))
    return true;

  // 处理特定的条件下的节点是否可微分的情况
  if (n->matches(
          "aten::dropout(Tensor input, float p, bool train) -> Tensor",
          attr::train)) {
    return n->get<bool>(attr::train).value();
  }

  // 处理特定的条件下的节点是否可微分的情况
  if (n->matches(
          "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor")) {
    // 检查节点是否具有名为 "size" 的属性，且其值为 int64_t 类型的列表，并且节点是隐式常量
    return n->get<c10::List<int64_t>>(attr::size) &&
        n->is_constant(attr::implicit);
  }

  // 获取节点的可能存在的模式（schema）
  auto schema = n->maybeSchema();
  // 如果存在模式且该模式对应的梯度信息存在，则返回 true
  if (schema && hasGradientInfoForSchema(*schema)) {
    return true;
  }

  // 线性块可能作为图执行器的输入，但在进行微分之前会被移除
  if (n->kind() == prim::GradOf) {
    // 获取 GradOf 节点的主体块
    auto body = n->blocks().at(0);
    // 检查主体块中的所有节点是否都可微分
    return std::all_of(
        body->nodes().begin(),
        body->nodes().end(),
        static_cast<bool (*)(const Node*)>(isDifferentiable));
  }

  // 公式仅定义在浮点标量上，因此对于其他情况，我们回退到自动求导
  for (const Value* input : n->inputs()) {
    // 如果输入值的类型是 NumberType::get()，则返回 false
    if (input->type() == NumberType::get()) {
      return false;
    }
  }

  // 默认返回 false
  return false;
}

// 检查图中所有节点是否都是可微的
bool isDifferentiable(Graph& g) {
    return std::all_of(
        g.nodes().begin(),
        g.nodes().end(),
        static_cast<bool (*)(const Node*)>(isDifferentiable));
}

// NB: 使用 torchscript 编写梯度
// 例如，aten::mul() 节点的定义如下
// def forward(x, y):
//     return x*y, (x, y)
// def backward(ctx, grad_output):
//     x, y = ctx
//     return (y * grad_output).sum_to_size(x), (x * grad_output).sum_to_size(y)
//
// 这段 Python 代码被编译成 GradientPair，其中包括前向图和反向图。
// 前向图将用于替换 grad_desc.f 中的节点，反向图将用于在反向传播过程中构造 GradOf(node)。
// Grad_values（也称为 gradOutputs）按 node->owningGraph() 中的**反向**顺序传播，
// 因此 GradientPair.forward 应在替换节点之后插入，以避免多次遍历图。
//
// 编译后的前向图的输出为 [real_outputs, ctx]
// 编译后的反向图的输入为 [ctx, grad_values]
// 我们随后运行 LowerSimpleTuples 来消除在此过程中生成的所有元组。
// 前向图中的原始节点和 TupleConstruct 节点将在稍后使用 EliminateDeadCode(block) 清除。
// 反向图中的 TupleUnPack 节点将在此文件中定义的 eliminateDeadcode(ReverseDetails) 中删除。
static std::optional<std::vector<Value*>> build_script_grad(
    Node* node,
    const ArrayRef<Value*>& grads) {
    auto graph = node->owningGraph();
    auto maybe_schema = node->maybeSchema();
    if (!maybe_schema) {
        return c10::nullopt;
    }
    auto compiled_graphs = gradientInfoForSchema(*maybe_schema);
    if (!compiled_graphs) {
        return c10::nullopt;
    }
    // 使用前向图替换 grad_desc.f 中的节点
    value_list new_outputs;
    {
        WithInsertPoint guard(node->next());
        auto fw_graph = compiled_graphs->forward;
        new_outputs = insertGraph(*graph, *fw_graph, node->inputs());
        new_outputs = unpackOutputs(new_outputs);
        auto outputs = node->outputs();
        AT_ASSERT(new_outputs.size() == outputs.size() + 1);
        for (const auto i : c10::irange(outputs.size())) {
            new_outputs.at(i)->setType(outputs[i]->type());
            outputs[i]->replaceAllUsesWith(new_outputs.at(i));
        }
    }

    // 使用反向图构造 reverse_block
    auto bw_graph = compiled_graphs->backward;
    auto grad_vec = grads.vec();
    if (needTrimGrad(node)) {
        grad_vec.erase(grad_vec.begin() + 1, grad_vec.end());
    }
    auto it = grad_vec.begin();
    grad_vec.insert(it, new_outputs.back());
    ArrayRef<Value*> grad(grad_vec);
    auto grad_inputs = insertGraph(*graph, *bw_graph, grad);
    grad_inputs = unpackOutputs(grad_inputs);
    return grad_inputs;
};

namespace {
// 定义 GradientHelper 类，用于帮助计算梯度
class GradientHelper {
 public:
  // 构造函数，初始化节点指针
  GradientHelper(Node* n) : node(n) {}

  // 计算梯度的主要方法，接受梯度值的数组并返回值的指针数组
  std::vector<Value*> gradient(ArrayRef<Value*> grad_values) {
    // 检查节点是否可微分，如果不可微分则抛出异常
    if (!isDifferentiable(node)) {
      throw std::runtime_error(
          std::string("differentiation of ") + node->kind().toDisplayString() +
          " is not supported, or it is missing necessary type information");
    }

    // 如果通过 torchscript 定义了自动微分，则使用该定义而不是符号化的方法
    auto script_grads = build_script_grad(node, grad_values);
    if (script_grads)
      return *script_grads;

    // 在 torchscript 中未找到定义，尝试使用符号化的梯度计算
    // TODO: 将所有内容迁移到使用 torchscript
    return buildSymbolicGradient(grad_values);
  }

 private:
  // 私有成员变量，表示当前操作的节点
  Node* node;

  // 符号化梯度计算方法
  std::vector<Value*> buildSymbolicGradient(
      const ArrayRef<Value*>& grad_values) {
    auto inputs = node->inputs();
    auto outputs = node->outputs();

    // 根据节点的种类进行不同的符号化梯度计算
    if (node->kind() == prim::AutogradAdd) {
      // 注意：AutogradAdd 操作不进行广播
      return {grad_values.at(0), grad_values.at(0)};
    } else if (node->kind() == prim::profile) {
      return {grad_values.at(0)};
    } else if (node->kind() == prim::ConstantChunk) {
      auto* g = node->owningGraph();

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      Value* input_list;
      // 如果 grad_values 中只有一个元素且类型是 ListType::ofTensors() 的子类型，则直接使用该元素
      if (grad_values.size() == 1 &&
          grad_values[0]->type()->isSubtypeOf(*ListType::ofTensors())) {
        input_list = grad_values[0];
      } else {
        // 否则，创建一个张量列表并将 grad_values 中的值插入其中
        input_list =
            g->insertNode(g->createList(TensorType::get(), grad_values))
                ->output();
      }

      auto* cDim = g->insertConstant(node->i(attr::dim));
      auto* cat_node = g->insertNode(g->create(aten::cat, 1));
      cat_node->addInput(input_list);
      cat_node->addInput(cDim);
      return {cat_node->output()};
    } else if (
        node->kind() == prim::Constant || node->kind() == prim::AutogradZero) {
      // 对于常量节点或自动微分零节点，直接返回空数组
      return {};
    }

    // 对于其他未明确处理的节点种类，可能需要根据具体情况进行扩展
    // （这里可能需要根据代码库的实际情况进行补充）
    // 如果执行到这里，说明可能存在未实现的节点种类或是特殊情况需要处理
    // （这里可能需要进一步的代码实现或错误处理）
    return {};  // 默认返回空数组
  }
};
    // 如果节点匹配到 _slow_conv2d_forward 函数
    } else if (
        node->matches(
            "aten::_slow_conv2d_forward(Tensor self, Tensor weight, int[] kernel_size, Tensor? bias, int[] stride, int[] padding) -> Tensor")) {
      auto graph = node->owningGraph();  // 获取当前节点所在的计算图
      auto backward_value = graph->insert(
          aten::_slow_conv2d_backward,  // 插入 _slow_conv2d_backward 操作节点
          {grad_values.at(0),  // 梯度值对应的输入
           inputs.at(0),  // 输入张量
           inputs.at(1),  // 权重张量
           node->namedInput(attr::kernel_size),  // 核大小作为命名输入
           node->namedInput(attr::stride),  // 步长作为命名输入
           node->namedInput(attr::padding),  // 填充作为命名输入
           graph->insertConstant(c10::List<bool>({true, true, true}))});  // 插入常量张量
      // 插入操作返回一个元组，如果有多个输出，则自动进行解包
      Node* tuple_unpack_node =
          graph->insertNode(graph->createTupleUnpack(backward_value));
      auto tuple_outputs = tuple_unpack_node->outputs();  // 获取元组解包后的输出
      AT_ASSERT(tuple_outputs.size() == size_t(3));  // 断言解包后的输出数量为3
      // 返回解包后的张量
      return {
          tuple_outputs[0],
          tuple_outputs[1],
          nullptr,
          tuple_outputs[2],
          nullptr,
          nullptr};

    // 如果节点匹配到 native_batch_norm 函数
    } else if (
        node->matches(
            "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)")) {
      auto graph = node->owningGraph();  // 获取当前节点所在的计算图
      auto backward_value = graph->insert(
          aten::native_batch_norm_backward,  // 插入 native_batch_norm_backward 操作节点
          {grad_values.at(0),  // 梯度值对应的输入
           inputs.at(0),  // 输入张量
           inputs.at(1),  // 权重张量
           inputs.at(3),  // running_mean
           inputs.at(4),  // running_var
           outputs.at(1),  // 输出的保存变量 weight
           outputs.at(2),  // 输出的保存变量 bias
           inputs.at(5),  // training 参数
           inputs.at(7),  // eps 参数
           graph->insertConstant(c10::List<bool>({true, true, true}))});  // 插入常量张量
      // 插入操作返回一个元组，如果有多个输出，则自动进行解包
      Node* tuple_unpack_node =
          graph->insertNode(graph->createTupleUnpack(backward_value));
      auto tuple_outputs = tuple_unpack_node->outputs();  // 获取元组解包后的输出
      AT_ASSERT(tuple_outputs.size() == size_t(3));  // 断言解包后的输出数量为3
      // 返回解包后的张量
      return {
          tuple_outputs[0],
          tuple_outputs[1],
          tuple_outputs[2],
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr};
    }

    // 如果以上条件都不满足，抛出运行时错误，指明节点无法求导
    throw std::runtime_error(
        std::string("failed to differentiate `") +
        node->kind().toDisplayString() + "`");
  }
};
} // namespace

// 如果我们有一个函数 y = f(x)，其雅可比矩阵为 J，则 f 的反向传播为 dx = J^t * dy。
// 注意，由于反向传播总是执行这个矩阵乘法，我们知道它将一个输入全为零的向量映射到一个输出全为零的向量，
// 不管它内部选择什么操作来实际执行矩阵乘法（大多数使用一些优化形式，从不生成 J^t）。
// 更一般地，我们知道所有的反向计算都是线性的，可以利用这一特性进行更积极的优化。
// 可以用生成已知为零输入的反向函数来替换任何反向函数，产生已知为零的输出。
// 此函数将每个已知线性反向函数封装在一个 'GradOf' 子块中，以便我们可以使用这些信息进行优化。
// 特别地，specializeAutogradZero 将观察线性块的所有输入是否都是 AutogradZeroTensor（autograd 用于表示零的张量），
// 然后将这些零值传播到块的输出中。
static std::vector<Value*> linearGradientForNode(
    Node* node,
    ArrayRef<Value*> grad_values) {
  auto& graph = *node->owningGraph();

  // FIXME: 如果 forward 有多个输出，我们只支持一个需要梯度的情况
  if (needTrimGrad(node)) {
    grad_values = grad_values.at(0);
  }
  // 在计算图中插入一个新节点，类型为 prim::GradOf，输入为 grad_values，输出个数为 0
  auto linear = graph.insertNode(graph.create(prim::GradOf, {grad_values}, 0));
  // 为了更方便阅读梯度图，记住前向操作的名称
  linear->s_(attr::name, node->kind().toDisplayString());
  // 给新节点添加一个块
  auto block = linear->addBlock();
  // 将插入点设置在新块内
  WithInsertPoint guard(block);
  // 使用 GradientHelper 类来计算梯度，并得到结果
  auto results = GradientHelper(node).gradient(grad_values);
  // 将每个梯度结果注册为块的输出，并将其复制到 linear 节点的输出中，保留元数据
  return fmap(results, [block, linear](Value* grad) -> Value* {
    if (!grad || grad->mustBeNone())
      return nullptr;
    block->registerOutput(grad);
    return linear->addOutput()->copyMetadata(grad);
  });
}

// ReverseDetails 结构体，包含梯度映射和反向块
struct ReverseDetails {
  ReverseDetails(value_map&& grad_map, Block* reverse_block)
      : grad_map(std::move(grad_map)), reverse_block(reverse_block) {}

  value_map grad_map;
  Block* reverse_block;
};

// AutogradAdd 是一个特殊的加法函数，处理 Undef 的情况
// AutogradAdd(a, b) == a + b 如果 defined(a) 和 defined(b)
// AutogradAdd(Undef, b) == b
// AutogradAdd(a, Undef) == a
// AutogradAdd(Undef, Undef) == Undef
static Value* createAutogradAdd(Value* a, Value* b) {
  auto graph = a->owningGraph();
  // 在计算图中插入一个新节点，类型为 prim::AutogradAdd，输入为 a 和 b
  return graph->insertNode(graph->create(prim::AutogradAdd, {a, b}))->output();
}

// 命名空间开始，用于封装私有函数和数据
namespace {

// 检查输出是否需要梯度
bool outputRequiresGrad(Value* output) {
  // 如果输出不是 TensorType，直接返回其 requires_grad 属性
  if (output->type()->castRaw<TensorType>() == nullptr) {
    return output->requires_grad();
  }
  // 否则，检查输出的 TensorType 是否包含 requiresGrad 属性，如果有则返回其值
  std::optional<bool> requiresGrad =
      output->type()->expectRef<TensorType>().requiresGrad();
  if (requiresGrad.has_value()) {
    return *requiresGrad;
  }

  // 如果输出节点不是 profile 节点，则默认需要梯度
  Node* n = output->node();
  if (n->kind() != prim::profile) {
    return true;
  }
  // 如果 profile 节点没有属性 attr::profiled_type，则默认需要梯度
  if (!n->hasAttribute(attr::profiled_type)) {
    // 返回 true
    return true;
  }
  // 返回给定节点 n 的 profiled_type 是否需要梯度
  return n->ty(attr::profiled_type)->requires_grad();
// 在命名空间结束处

// Before:
//   - grad_desc has field f initialized to the original 0-stage graph
// After:
//   - grad_desc 中的 f 字段已初始化为原始的 0 阶段图形
//   - f 的最后一个节点 (f->nodes().reverse()[0]) 是一个梯度节点，
//     其块具有对所有需要梯度的输出的 vjp 输入，
//     并对所有需要梯度的原始输入具有 vjp 输出
//   - grad_desc 设置了 df_input_vjps 和 df_output_vjps
//     (但 df_input_vjps 之后还会被修改)
static ReverseDetails addReverseInline(Gradient& grad_desc) {
  auto& graph = *grad_desc.f;
  // 注意：不插入 reverse_node 是有意的，以避免意外操作它
  // (例如在消除死代码时)，可以使用 std::cout << *reverse_node << 查看其状态。
  auto reverse_node = graph.create(prim::Reverse, 0);
  auto reverse_block = reverse_node->addBlock();
  WithInsertPoint guard(reverse_block);

  value_map grad_map; // x -> dx 映射
  const auto get_grad = [&](Value* v) -> Value* {
    auto it = grad_map.find(v);
    if (it == grad_map.end()) {
      auto autograd_zero = graph.insertNode(graph.createAutogradZero());
      it = grad_map.emplace(v, autograd_zero->output()).first;
    }
    return it->second;
  };
  const auto set_grad = [&](Value* x, Value* dx) {
    if (Value* prev_grad = grad_map[x]) {
      GRAPH_DEBUG("grad_map[", x->debugName(), "] = ", *grad_map[x]->node());
      grad_map[x] = createAutogradAdd(prev_grad, dx);
    } else {
      GRAPH_DEBUG("grad_map[", x->debugName(), "] = ", dx->debugName());
      grad_map[x] = dx;
    }
  };

  auto outputs = graph.outputs();
  for (size_t i = 0, num_outputs = outputs.size(); i < num_outputs; ++i) {
    Value* output = outputs[i];
    if (!outputRequiresGrad(output))
      continue;
    Value* output_grad = reverse_block->addInput()->setType(output->type());
    GRAPH_DEBUG(
        "Adding output_grad ",
        output_grad->debugName(),
        " for ",
        output->debugName());
    set_grad(output, output_grad);
    grad_desc.df_input_vjps.push_back(i);
  }

  for (auto it = graph.nodes().rbegin(), end = graph.nodes().rend(); it != end;
       ++it) {
    Node* node = *it;
    auto inputs = node->inputs();
    auto outputs = node->outputs();
    if (std::all_of(outputs.begin(), outputs.end(), [](Value* v) {
          return !v->requires_grad();
        })) {
      continue;
    }

    value_list grad_inputs =
        linearGradientForNode(node, fmap(node->outputs(), get_grad));
    LowerSimpleTuples(reverse_block);

    AT_ASSERT(grad_inputs.size() == node->inputs().size());
    for (size_t i = 0, num_inputs = grad_inputs.size(); i < num_inputs; ++i) {
      if (!inputs[i]->requires_grad())
        continue;
      // 注意：对于不可区分的输入，不返回关于需要梯度的值的梯度是正常的。
      // 这在例如 aten::type_as 案例中发生。
      if (!grad_inputs[i])
        continue;
      set_grad(inputs[i], grad_inputs[i]);
  }
}

auto inputs = graph.inputs();  // 获取计算图的输入节点列表
for (size_t i = 0, num_inputs = inputs.size(); i < num_inputs; ++i) {
  Value* input = inputs[i];  // 获取当前输入节点
  if (!input->requires_grad())  // 如果当前输入节点不需要计算梯度，则跳过
    continue;
  // 注意：对于需要计算梯度的计算图输入节点，可能出现没有定义梯度的情况，并非错误。
  // 这可能发生在非可微分的上下文中（例如作为 aten::type_as 的第二个输入）。
  // 在这种情况下，我们简单地忽略它作为输出，因为它不会产生任何有意义的值。
  if (grad_map.count(input) == 0)  // 如果梯度映射中不包含当前输入节点，则跳过
    continue;
  reverse_block->registerOutput(get_grad(input));  // 在反向传播块中注册当前输入节点的梯度
  grad_desc.df_output_vjps.push_back(i);  // 将当前输入节点的索引添加到梯度描述的输出向量中
}

Inline(graph);  // 内联计算图
return ReverseDetails(std::move(grad_map), reverse_block);  // 返回反向传播的详细信息
// 返回函数 f 中生成并在其反向程序中使用的值的拓扑排序列表。
static value_list getReverseCaptures(Gradient& grad_desc) {
  // 获取计算图
  auto& graph = *grad_desc.f;
  // 获取原始块
  auto primal_block = graph.block();

  // 创建用于存储反向捕获值的集合和列表，列表保持拓扑排序不变
  value_set reverse_captures_set;
  value_list reverse_captures; // Invariant: topo sorted

  // Lambda 函数，用于检查值 v 的使用情况
  auto check_uses = [&](Value* v) {
    // 遍历值 v 的使用
    for (auto use : v->uses()) {
      // 如果使用的节点所在的块是原始块，则跳过
      if (use.user->owningBlock() == primal_block)
        continue;
      // 如果值 v 是第一次出现（未见过），则将其添加到捕获列表和集合中
      if (reverse_captures_set.emplace(v).second) {
        reverse_captures.push_back(v);
      }
    }
  };

  // 遍历计算图的输入节点，并检查其使用情况
  for (Value* input : graph.inputs()) {
    check_uses(input);
  }

  // 遍历计算图的节点，并检查每个节点输出的值的使用情况
  for (Node* node : graph.nodes()) {
    for (Value* output : node->outputs())
      check_uses(output);
  }

  // 返回拓扑排序后的反向捕获值列表
  return reverse_captures;
}

// 将原始图中的临时值捕获以备后用，避免昂贵的重复计算。
// 然而，图中的很多节点都是常量，执行和复制它们成本很低，
// 所以最好只是将它们复制到反向图中，而不是不必要地污染输出列表。
static void liftConstants(Block* block, Block* move_to_this_block);

// 检查节点是否定义在指定的块内
static bool inBlock(Node* node, Block* container) {
  Block* b = node->owningBlock();
  while (b) {
    if (b == container) {
      return true;
    }
    b = b->owningNode() ? b->owningNode()->owningBlock() : nullptr;
  }
  return false;
}

// 将节点中的常量移动到指定的块中
static void liftConstants(Node* node, Block* move_to_this_block) {
  // Lambda 函数，用于处理异常情况
  static const auto err = [](Value*) -> Value* {
    throw std::runtime_error("unexpected input");
  };
  auto& graph = *node->owningGraph();
  
  // 遍历节点的输入值
  for (Value* input : node->inputs()) {
    // 如果输入节点不是常量，则继续下一个输入值
    if (input->node()->kind() != prim::Constant)
      continue;
    // 如果该常量已经定义在反向传播块中，则不需要复制和移动它
    if (inBlock(input->node(), move_to_this_block))
      continue;
    // 创建常量的克隆节点，并将其添加到指定的块中
    Node* lifted_constant = graph.createClone(input->node(), err);
    move_to_this_block->prependNode(lifted_constant);
    GRAPH_DEBUG(
        "Lifting constant ",
        input->debugName(),
        " from GradOf's block and adding ",
        lifted_constant->output()->debugName(),
        " to the backprop block");
    // 用克隆节点的输出替换节点的输入
    node->replaceInputWith(input, lifted_constant->output());
  }

  // 递归处理节点的子块
  for (Block* sub : node->blocks()) {
    liftConstants(sub, move_to_this_block);
  }
}

// 将块中的所有节点的常量移动到指定的块中
static void liftConstants(Block* block, Block* move_to_this_block) {
  // 遍历块中的所有节点，并将其中的常量移动到指定的块中
  for (Node* node : block->nodes()) {
    liftConstants(node, move_to_this_block);
  }
  // 将块的返回节点中的常量移动到指定的块中
  liftConstants(block->return_node(), move_to_this_block);
}
// 如果节点不是类型为 aten::_size_if_not_equal 的节点，则跳过处理
static void foldSizeIfNotEqual(Block* node);

// 对给定节点进行尺寸折叠处理
static void foldSizeIfNotEqual(Node* node) {
  // 遍历节点的输入值
  for (Value* input : node->inputs()) {
    // 如果输入值所属节点的类型不是 aten::_size_if_not_equal，则继续下一个输入值
    if (input->node()->kind() != aten::_size_if_not_equal) {
      continue;
    }

    // 获取输入节点的第一个和第二个输入值的类型，并期望它们是 TensorType 类型
    auto ptt_input =
        input->node()->input(0)->node()->input()->type()->expect<TensorType>();
    auto ptt_output =
        input->node()->input(1)->node()->input()->type()->expect<TensorType>();

    // 获取输入和输出的具体尺寸
    auto input_size = ptt_input->sizes().concrete_sizes();
    auto output_size = ptt_output->sizes().concrete_sizes();

    // 如果输入或输出尺寸不可用，则跳过处理
    if (!input_size || !output_size) {
      continue;
    }
    
    // 在 _grad_sum_to_size 的前面插入节点
    WithInsertPoint guard(node);
    IValue ival{};
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Value* size;
    // 如果输入和输出的尺寸不相等，则创建常量节点并替换输入值
    if (input_size != output_size) {
      size = node->owningGraph()->insertConstant(*input_size);
    } else {
      size = node->owningGraph()->insertConstant(IValue());
    }
    node->replaceInputWith(input, size);
  }

  // 递归处理节点的子块
  for (auto ib : node->blocks()) {
    foldSizeIfNotEqual(ib);
  }
}

// 在反向图形成时，折叠 aten::_size_if_not_equal 节点
// 在了解到 aten::_size_if_not_equal 参数的形状时，进行折叠
// 否则，它们将成为反向图的输入，并丢失这些信息，我们尚未对 Scalars 或 Lists 进行分析
static void foldSizeIfNotEqual(Block* reverse_block) {
  // 遍历反向块中的节点，对每个节点应用尺寸折叠处理
  for (auto n : reverse_block->nodes()) {
    foldSizeIfNotEqual(n);
  }
  // 对反向块的返回节点进行尺寸折叠处理
  foldSizeIfNotEqual(reverse_block->return_node());
}

// 删除尺寸捕获的重复内容
static void deduplicateSizeCaptures(
    Gradient& grad_desc,
    ReverseDetails& rev_info) {
  // 获取主块
  Block* primal_block = grad_desc.f->block();
  // 用于检查值是否仅在反向使用中
  const auto usedOnlyInReverse = [primal_block](Value* v) {
    const auto& uses = v->uses();
    // 检查值的每个用法是否在主块之外
    return std::all_of(uses.begin(), uses.end(), [primal_block](const Use& u) {
      return u.user->owningBlock() != primal_block;
    });
  };
  // 获取反向捕获的值
  auto captures = getReverseCaptures(grad_desc);
  // 将捕获转换为集合以便去重
  value_set capture_set(captures.begin(), captures.end());
  // 遍历捕获的每个值
  for (Value* capture : captures) {
    Node* node = capture->node();
    // 如果节点不匹配 "aten::size(Tensor self) -> int[]"，则继续下一个捕获
    if (!node->matches("aten::size(Tensor self) -> int[]")) {
      continue;
    }
    // 如果捕获的值仅在反向使用中，并且捕获集合中包含输入节点
    if (usedOnlyInReverse(capture) && capture_set.count(node->input())) {
      // 在反向块的节点开头插入节点
      WithInsertPoint insert_guard{*rev_info.reverse_block->nodes().begin()};
      // 创建 size 节点并替换捕获节点的所有用法
      auto* size =
          node->input()->owningGraph()->insert(aten::size, {node->input()});
      GRAPH_DEBUG(
          "deduplicateSizeCaptures: Replacing ",
          capture->debugName(),
          " with ",
          size->debugName());
      capture->replaceAllUsesWith(size);
      // 销毁捕获节点
      node->destroy();
    }
  }
}
// 对于给定的 ReverseDetails 结构，执行死代码消除。
static void eliminateDeadCode(ReverseDetails& rev_info) {
  // addReverseInline 必须调用 gradientForNode 如果任何输入需要梯度，
  // 但它会为所有输入生成 vjp。使用 DCE 移除不必要的节点。
  // 此外，对于中间变量的 requires_grad() 是对真实状态的近似，
  // 因此我们可能已经生成了一些梯度，但一旦到达不需要梯度的点，就会意识到它们是不必要的。
  // 当然，我们需要过滤 grad_map 的相应条目，因为我们不希望以后意外访问已释放的指针。
  std::function<void(const std::unordered_set<const Value*>&)> cb =
      [&](const std::unordered_set<const Value*>& live_values) {
        // 准备要删除的值的列表
        std::vector<Value*> to_erase;
        // 遍历 grad_map 中的条目
        for (auto& entry : rev_info.grad_map) {
          // 如果 live_values 中不包含 entry.second，说明该值不再使用
          if (!live_values.count(entry.second)) {
            // 将该值添加到待删除列表中
            to_erase.push_back(entry.first);
          }
        }
        // 遍历待删除列表，并从 grad_map 中擦除对应的条目
        for (Value* v : to_erase) {
          GRAPH_DEBUG(
              "Erasing unused value ", v->debugName(), " from grad_map");
          rev_info.grad_map.erase(v);
        }
      };
  // 执行死代码消除操作，传入回调函数 cb
  EliminateDeadCode(rev_info.reverse_block, std::move(cb));
}

// 对梯度和反向传播细节进行优化
static void Optimize(Gradient& grad_desc, ReverseDetails& rev_info) {
  // TODO: 有时会生成像 _grad_sum_to_size(_grad_sum_so_size(x, s1), s2) 这样的表达式，
  // 它们等价于 _grad_sum_to_size(x, s2)，可以节省一些捕获，但我不确定如何在这个阶段优化它们，
  // 因为我们不知道哪些 GradOf 块将被组合成导数。我想一个智能分析可以实现这一点，
  // 但在 1.0 发布之前我没有时间，所以我只将其作为一个针孔优化。
  // 提升常量
  liftConstants(rev_info.reverse_block, rev_info.reverse_block);
  // TODO: 看看是否可以用针孔优化替换这个 pass
  foldSizeIfNotEqual(rev_info.reverse_block);
  // 我们通常会添加很多 aten::size 调用（用于广播运算符的导数），它们通常会重复出现，
  // 并且可能会被多次捕获。在提升之前确保对它们进行去重。
  EliminateCommonSubexpression(grad_desc.f);
  // 去重大小捕获
  deduplicateSizeCaptures(grad_desc, rev_info);
  // 执行死代码消除
  eliminateDeadCode(rev_info);
}

// 从 `addReverseInline` 返回的 grad_desc.f 中获取 reverse_block 并拆分为其自己的图形，
// 存储在 df 中。所有第二阶段需要的中间变量被添加到 f 的输出，并作为 df 的输入。
// 更详细的描述请参见 autodiff.h 中的 "Gradient graphs" 笔记。
// 这个函数还初始化了在 `addReverseInline` 后未定义的 grad_desc 字段
// （并使用捕获临时变量扩展了 `df_input_vjps`）。
// 对 lambdaLiftReverse 函数进行 lambda 提升操作，将图的相关信息和反向信息作为参数传入
static void lambdaLiftReverse(Gradient& grad_desc, ReverseDetails& rev_info) {
  // 获取图对象的引用
  auto& graph = *grad_desc.f;
  // 获取反向信息中的反向块
  auto reverse_block = rev_info.reverse_block;

  // --------------------------------------------------------------------------
  // 1. Find values of f that need to be captured.
  // --------------------------------------------------------------------------
  // 首先，需要找出在 f 中产生但在 df 中被使用的所有值。
  // 这些值需要作为 df 的输入添加，并且如果它们不是 f 的输入或输出，则可能还需要将它们附加为 f 的输出。
  // 不变量：拓扑排序
  // 获得需要捕获的反向值列表
  value_list reverse_captures = getReverseCaptures(grad_desc);

  // --------------------------------------------------------------------------
  // 2. Prepare input/outputs lists for f and df
  // --------------------------------------------------------------------------
  // 构造 f 和 df 的输入/输出列表
  // primal_inputs/reverse_outputs 简单构造，
  // 但 primal_outputs/reverse_inputs 更加微妙。
  // 下面是它们预期的外观摘要：
  //
  // Primal outputs:
  //   [原始输出], [临时值]
  //
  // Reverse inputs:
  //   [输出 vjps (也称为梯度输出)], [临时 vjps]
  //   [捕获的原始值，按拓扑顺序]

  // -- 构造 primal_outputs, df_input_captures, f_real_outputs ----
  grad_desc.f_real_outputs = graph.outputs().size();

  // 使用无序映射来存储原始 primal_outputs 和 primal_inputs 的索引
  std::unordered_map<Value*, size_t> orig_primal_outputs_idx;
  std::unordered_map<Value*, size_t> orig_primal_inputs_idx;

  // 使用 emplace 避免替换已存在的索引，如果输出被重复
  for (size_t i = 0, num_outputs = graph.outputs().size(); i < num_outputs; ++i)
    orig_primal_outputs_idx.emplace(graph.outputs()[i], i);
  for (size_t i = 0, num_inputs = graph.inputs().size(); i < num_inputs; ++i)
    orig_primal_inputs_idx[graph.inputs()[i]] = i;

  // 注意：reverse_captures 已经是去重且按拓扑顺序的
  for (Value* capture_val : reverse_captures) {
    // 如果它已经是输出，我们不需要添加任何内容，
    // 但需要注册它需要被捕获的事实。
    if (orig_primal_outputs_idx.count(capture_val) > 0) {
      grad_desc.df_input_captured_outputs.push_back(
          orig_primal_outputs_idx[capture_val]);
    // 如果它是输入，我们可以将其添加为输出，但实际上更有效的是使用特殊类型的捕获。
    } else if (orig_primal_inputs_idx.count(capture_val) > 0) {
      grad_desc.df_input_captured_inputs.push_back(
          orig_primal_inputs_idx.at(capture_val));
    // 否则，它只是一个常规的中间值，我们需要将其添加为输出
    } else {
  } else {
    // 如果捕获的值之前不可用，需要为其创建一个新的临时输出
    // 这里创建一个临时输出是因为该值之前不在图中可用。

    auto out_index = graph.registerOutput(capture_val);
    // 调试信息：捕获一个临时值作为正向图的输出
    GRAPH_DEBUG(
        "Capturing a temporary ",
        capture_val->debugName(),
        " as ",
        graph.outputs()[out_index]->debugName(),
        " for forward graph");
    // 将新注册的输出索引添加到捕获描述的输入捕获输出列表中
    grad_desc.df_input_captured_outputs.emplace_back(
        graph.outputs().size() - 1);
  }

  // -- Add VJPs for temporaries, adjust df_input_vjps -------------------------
  // 注意 [可能的优化]: 当为该值生成第一个 VJP 后，尽快使用新添加的 VJP 输入，
  // 以减少该输入的生命周期（当前我们在添加到最终的 VJP 后才使用它）。
  for (size_t i = grad_desc.f_real_outputs; i < graph.outputs().size(); ++i) {
    Value* tmp = graph.outputs().at(i);
    // 只为确实需要梯度的中间值添加 VJP 输入。
    // 注意，我们检查 grad_map 的内容而不是 tmp->requires_grad()，
    // 因为这更加准确。tmp->requires_grad() 是一种过度估计（可能有误报），
    // 而我们为该值生成的梯度可能在优化过程中被删除（因为它对我们要微分的实际 f 的输出没有影响）。
    if (rev_info.grad_map.count(tmp) == 0)
      continue;

    Value* tmp_vjp_in = reverse_block->addInput()->setType(tmp->type());
    Value* tmp_vjp_prev = rev_info.grad_map.at(tmp);
    // 这里的操作有些奇怪，因为我们不能先创建一个和然后替换所有使用 tmp_vjp_prev 的节点
    // （这样会替换掉在和中的使用！），所以我们创建一个不正确的和来替换使用，并修复和。
    Value* new_vjp = createAutogradAdd(tmp_vjp_in, tmp_vjp_in);
    if (tmp_vjp_prev->node()->kind() == prim::Param) {
      // 无法将节点移动到块参数节点之后
      new_vjp->node()->moveBefore(
          *tmp_vjp_prev->node()->owningBlock()->nodes().begin());
    } else {
      new_vjp->node()->moveAfter(tmp_vjp_prev->node());
    }

    tmp_vjp_prev->replaceAllUsesWith(new_vjp);
    new_vjp->node()->replaceInput(1, tmp_vjp_prev);
    // 调试信息：grad_map[tmp 的调试名] = 新的 VJP 节点
    GRAPH_DEBUG("grad_map[", tmp->debugName(), "] = ", *new_vjp->node());
    // 将新添加的 VJP 输入索引添加到捕获描述的输入 VJP 列表中
    grad_desc.df_input_vjps.emplace_back(i);
  }

  // 将捕获的值作为反向块的形式参数添加
  // 然后输入为：[输出 VJP][临时 VJP][捕获值]
  // 构建从捕获的 'value' 到输入列表中的索引的映射，
  // 以便将该块提取到其自己的函数中
  std::unordered_map<Value*, size_t> capture_to_formal_index;
  const auto& add_capture = [&](Value* captured) {
    capture_to_formal_index[captured] = reverse_block->inputs().size();
    // 复制捕获的值的元数据，并将其作为新的输入添加到反向块中
    auto new_input = reverse_block->addInput()->copyMetadata(captured);
    GRAPH_DEBUG(
        "Capturing ",
        captured->debugName(),
        " as ",
        new_input->debugName(),
        " for an embedded backward block");

记录调试信息：捕获 captured 对象的调试名称作为新输入 new_input 的调试名称，用于嵌入式反向块。


  };

匿名函数定义结束。


  for (auto& offset : grad_desc.df_input_captured_inputs)
    add_capture(graph.inputs()[offset]);

遍历 grad_desc.df_input_captured_inputs 中的偏移量，将对应的 graph.inputs() 中的项添加到捕获列表中。


  for (auto& offset : grad_desc.df_input_captured_outputs)
    add_capture(graph.outputs()[offset]);

遍历 grad_desc.df_input_captured_outputs 中的偏移量，将对应的 graph.outputs() 中的项添加到捕获列表中。


  grad_desc.df = std::make_shared<Graph>();

创建一个名为 grad_desc.df 的新 Graph 对象，并使用 std::make_shared 进行管理。


  grad_desc.df->block()->cloneFrom(reverse_block, [&](Value* v) {
    return grad_desc.df->inputs()[capture_to_formal_index.at(v)];
  });

从 reverse_block 克隆块到 grad_desc.df 中，使用 lambda 函数返回 grad_desc.df 中输入的对应项作为形式索引。


  GRAPH_DUMP(" forward graph: ", &graph);

输出调试信息：前向图，打印 graph 的内容。


  GRAPH_DEBUG(" backward graph: ", *(reverse_block->owningNode()));

记录调试信息：反向图，使用 reverse_block 的 owningNode() 打印其内容。


  // reverse_node was just to hold onto reverse_block in a debuggable way
  // we can remove it now.
  reverse_block->owningNode()->destroy();

注释：reverse_node 仅为了以可调试的方式保存 reverse_block，现在可以安全地移除它。销毁 reverse_block 的 owningNode()。
}

// 将返回值打包成元组
static void packReturnValuesIntoTuple(const std::shared_ptr<Graph>& graph) {
  // 获取图中的返回节点
  auto returnNode = graph->block()->return_node();
  // 将插入点设置在返回节点之前
  WithInsertPoint wip(returnNode);
  // 创建一个元组节点，包含所有返回节点的输入
  auto tuple = graph->insertNode(graph->createTuple(returnNode->inputs()));
  // 移除返回节点的所有输入
  returnNode->removeAllInputs();
  // 将元组节点的输出作为返回节点的唯一输入
  returnNode->addInput(tuple->output());
}

// 执行自动求导，返回求导结果
Gradient differentiate(std::shared_ptr<Graph>& graph) {
  // 创建梯度描述对象
  Gradient grad_desc;
  // 检查图的所有权是否为1，因为 differentiate 会修改和销毁图
  TORCH_CHECK(
      graph.use_count() == 1,
      "differentiate will mutate and destroy the graph, so it requires "
      "graph.use_count() == 1, but found %d",
      graph.use_count());
  // 交换图对象，将其转移至 grad_desc.f
  std::swap(graph, grad_desc.f);
  // 注意处理输出时可能的重复性！

  // 输出 grad_desc.f 中的图形结构
  GRAPH_DUMP("grad_desc.f: ", grad_desc.f);
  // 在 grad_desc.f 的块中设置插入点
  WithInsertPoint guard(grad_desc.f->block());
  // 填充 df_input_vjps 和 df_output_vjps
  auto rev_info = addReverseInline(grad_desc);
  // 对 grad_desc 进行优化
  Optimize(grad_desc, rev_info);
  // 清理被 torchscript 中的前向图替换的旧节点
  EliminateDeadCode(grad_desc.f->block());

  // 填充 f、df、f_real_outputs、df_input_captures，
  // 修改 df_input_vjps（为临时变量添加新的 vjps）
  lambdaLiftReverse(grad_desc, rev_info);
  // 将 df 打包成元组
  packReturnValuesIntoTuple(grad_desc.df);

  // 我们创建了一个可微的前向图，
  // 将使用梯度分离后的张量运行它们，
  // 因此概要类型可能具有过时的 requires_grad=True，更新 requires_grad 属性
  UpdateDifferentiableGraphRequiresGrad(grad_desc.f, false);
  // 返回梯度描述对象
  return grad_desc;
}
} // namespace torch::jit
```