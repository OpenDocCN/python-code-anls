# `.\pytorch\torch\csrc\jit\passes\batch_mm.cpp`

```
#include <torch/csrc/jit/passes/batch_mm.h>

#include <ATen/core/functional.h>
#include <ATen/core/symbol.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

#include <ATen/ATen.h>
#include <algorithm>
#include <unordered_map>
#include <utility>

namespace torch {
namespace jit {

namespace {
// 定义了一个局部函数 aliasAnalysisIsSpecialCase，返回 AliasAnalysisKind 枚举类型的特定值
c10::AliasAnalysisKind aliasAnalysisIsSpecialCase() {
  return AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
}
} // namespace

// 这个 pass 主要用于在图中查找特定的树结构，其中叶子节点是 mm 操作，内部节点是 add 节点。
// 一旦找到这样的树，可以将其优化为两个 concat 操作和一个单独的 mm 操作。
// 这种模式主要出现在 RNN 的反向传播中，因为许多使用相同权重的矩阵乘法的导数形成了这样的树。
// 这种树通常也非常不平衡，即深度为 O(n)。

// 这种结构（或者任何 add 的 MM 树）可以优化为一个单独的 MM 操作，通过将所有左操作数拼接，
// 将所有右操作数拼接，然后进行矩阵乘法运算。

// 注意 [Further optimizations]
// 进一步的优化可以扩展 TreeToken 类，以检测所有 MM 操作是否具有相同的左右操作数。
// 在这种情况下，通过扩展左操作数并使用 bmm + sum 更有效，而不是通过 concat 在内存中重复。

// 注意 [Overlapping trees]
// 此外，还可以添加对部分重叠树的支持。目前算法中禁止了部分重叠的树，
// 这可能会导致错过一些优化选项，特别是拒绝的树可能会更大。
// 我没有实现这一点，因为在我看到的简单 RNN 案例中并不需要，所以决定保持简单。
// 如果我们以后实现这个功能，正确的解决方案可能是融合共同部分的 MM 操作，并假设它是外部两部分的输入叶子。

// 这个类实现了一个特定的图优化 pass，查找并优化特定的树结构。
struct BatchMM : public torch::jit::GraphOptimizerPass {
  void run(std::shared_ptr<torch::jit::Graph>& graph) override {
    // 实现在图中查找并优化特定树结构的逻辑
    optimizeTreeStructure(graph);
  }

private:
  // 这个函数实现了查找和优化特定树结构的逻辑
  void optimizeTreeStructure(std::shared_ptr<torch::jit::Graph>& graph) {
    // 实现在图中查找特定树结构并进行优化的具体细节
    // ...
  }
};

} // namespace jit
} // namespace torch
// recompute, unless the subtree is super small, but let's not get into such
// details).
// 重新计算，除非子树非常小，但是我们不深入讨论这些细节。

// The algorithm we're using is simple. We're iterating through the graph in the
// topological order and labeling nodes with TreeTokens. Then, we look for roots
// of the trees we formed and fuse them.
// 我们使用的算法很简单。我们按照拓扑顺序遍历图，并使用TreeTokens标记节点。然后，我们查找形成的树的根，并将它们融合。

// Tunable parameter. Set to something larger if it turns out to be better.
static constexpr size_t min_fusion_size = 4;
// 可调参数。如果效果更好，可以设置得更大。

static bool have_same_shape(at::TensorList inputs) {
  auto expected_sizes = inputs[0].sizes();
  return (std::all_of(
      inputs.begin(), inputs.end(), [expected_sizes](const at::Tensor& t) {
        return t.sizes() == expected_sizes;
      }));
}

static bool should_be_transposed(at::TensorList inputs) {
  return (std::all_of(inputs.begin(), inputs.end(), [](const at::Tensor& t) {
    return t.stride(0) == 1 && t.stride(1) == t.size(0);
  }));
}

static std::vector<at::Tensor> transpose_inputs(at::TensorList inputs) {
  return fmap(inputs, [](const at::Tensor& i) { return i.t(); });
}

static bool shape_is_fast_for_reduce(
    const at::Tensor& lhs,
    const at::Tensor& rhs) {
  size_t l = lhs.size(0);
  size_t m = lhs.size(1);
  size_t r = rhs.size(1);
  // Numbers obtained by some simple benchmarks of fp32 gemms on a TITAN V
  return m < 512 || ((l < 256 && r < 256) || (l > 256 && r > 256));
}

RegisterOperators mm_tree_reduction_reg({Operator(
    "prim::MMTreeReduce(...) -> Tensor",
    [](Stack& stack) {
      // 从堆栈中弹出输入数量
      auto num_inputs = pop(stack).toInt();
      // 创建一个存放张量的向量，并预留空间
      std::vector<at::Tensor> inputs;
      inputs.reserve(num_inputs);
      // 将堆栈中的张量移动到输入向量中
      for (auto it = stack.end() - num_inputs; it != stack.end(); ++it) {
        inputs.push_back(std::move(*it).toTensor());
      }
      // 从堆栈中移除已处理的输入张量
      drop(stack, num_inputs);
    
      // 断言输入向量不能为空
      AT_ASSERT(!inputs.empty());
      // 断言输入张量数量为偶数
      AT_ASSERT(inputs.size() % 2 == 0);
      // 计算每一侧张量的数量
      size_t side_num_elems = inputs.size() / 2;
      // 将输入向量分为左右两侧
      auto lhs_inputs = at::TensorList(inputs).slice(0, side_num_elems);
      auto rhs_inputs = at::TensorList(inputs).slice(side_num_elems);
    
      // TODO: 检查是否具有相同的形状和适合快速减少操作的形状
      if (have_same_shape(lhs_inputs) && have_same_shape(rhs_inputs) &&
          shape_is_fast_for_reduce(lhs_inputs[0], rhs_inputs[0])) {
        // 如果 lhs_inputs 或 rhs_inputs 不是连续的，at::cat 将通过慢速路径处理
        // 如果可能，视为连续的进行转置
        bool lhs_input_transposed = should_be_transposed(lhs_inputs);
        bool rhs_input_transposed = should_be_transposed(rhs_inputs);
        at::Tensor lhs, rhs;
        // 如果 lhs_input_transposed 为真，则转置输入并连接
        if (lhs_input_transposed) {
          std::vector<at::Tensor> lhs_contig_inputs =
              transpose_inputs(lhs_inputs);
          lhs = at::cat(lhs_contig_inputs, /*dim*/ 0);
          lhs = lhs.t();  // 对结果进行转置
        } else {
          lhs = at::cat(lhs_inputs, /*dim=*/1);
        }
        // 如果 rhs_input_transposed 为真，则转置输入并连接
        if (rhs_input_transposed) {
          std::vector<at::Tensor> rhs_contig_inputs =
              transpose_inputs(rhs_inputs);
          rhs = at::cat(rhs_contig_inputs, /*dim*/ 1);
          rhs = rhs.t();  // 对结果进行转置
        } else {
          rhs = at::cat(rhs_inputs, /*dim=*/0);
        }
        // 将 lhs 和 rhs 进行矩阵乘法，并将结果推入堆栈
        push(stack, at::mm(lhs, rhs));
      } else {
        // 如果形状不适合快速减少操作，则使用普通的矩阵乘法累加计算结果
        auto acc = at::mm(inputs[0], inputs[side_num_elems]);
        for (const auto i : c10::irange(1, side_num_elems)) {
          acc.add_(at::mm(inputs[i], inputs[side_num_elems + i]));
        }
        // 将累加结果推入堆栈
        push(stack, std::move(acc));
      }
    },
    aliasAnalysisIsSpecialCase())});
// TreeTokens将用于标记图中的节点，如果节点符合我们的mm/add树模式。基本上，我们在DAG上进行动态规划，
// 当我们到达具有输入A和B的节点N时，A和B已经被处理过，我们可以尝试统一它们的TreeTokens（如果它们有的话）
// 并构建一个更大的树。
struct TreeToken {
  uint64_t tree_size = 0; // 树的大小，以叶子节点数量（即mm操作数）衡量
  Node* node = nullptr; // 节点指针，指向图中的节点
  bool is_root = false; // 标志指示是否为根节点

  // 创建一个代表mm操作的TreeToken
  static TreeToken mm(Node* mm) {
    TreeToken token;
    token.tree_size = 1;
    token.node = mm;
    token.is_root = true;
    return token;
  }

  // 创建一个代表转置操作的TreeToken，返回的token可能无效，需要检查其布尔值！
  static TreeToken transpose(Node* t, TreeToken& inp_token) {
    TreeToken token;
    // 如果输入token的节点不匹配"aten::mm(Tensor self, Tensor mat2) -> Tensor"，则返回空token
    if (!inp_token.node->matches(
            "aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
      return token;
    }
    token.tree_size = 1;
    token.node = t;
    token.is_root = true;
    inp_token.is_root = false; // 将输入token标记为非根节点
    return token;
  }

  // 创建一个代表加法操作的TreeToken，返回的token可能无效，需要检查其布尔值！
  static TreeToken add(Node* add, TreeToken& l, TreeToken& r) {
    TreeToken token;
    // 检查是否是重叠树，或者左右节点不是根节点，如果是则返回空token
    if (&l == &r || !l.is_root || !r.is_root)
      return token;
    token.tree_size = l.tree_size + r.tree_size; // 计算合并后树的大小
    token.node = add; // 设置节点为加法节点
    token.is_root = true;
    l.is_root = r.is_root =
        false; // 保留子树，以防它们被再次使用
    return token;
  }

  explicit operator bool() {
    return is_root; // 返回当前token是否为根节点的布尔值
  }

  // 移除转置并收集矩阵乘法操作的节点
  std::vector<Node*> removeTransposesAndGatherMatmuls() {
    std::vector<Node*> matmuls; // 用于存储矩阵乘法节点的向量
    std::vector<Node*> queue{node}; // 使用队列存储待处理节点，初始为当前节点
    Graph* graph = node->owningGraph(); // 获取当前节点所属的图
    while (!queue.empty()) {
      auto n = queue.back(); // 取队列末尾节点
      queue.pop_back(); // 弹出队列末尾节点
      // 如果节点匹配"aten::mm(Tensor self, Tensor mat2) -> Tensor"，则加入matmuls向量
      if (n->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
        matmuls.push_back(n);
      } else if (n->matches("aten::t(Tensor self) -> Tensor")) {
        Node* input_node = n->input()->node(); // 获取当前节点输入的节点
        AT_ASSERT(input_node->matches(
            "aten::mm(Tensor self, Tensor mat2) -> Tensor")); // 断言输入节点是mm操作
        // (AB)^T == B^TA^T，进行矩阵转置操作
        WithInsertPoint insert_guard{input_node};
        Value* A = input_node->inputs()[0]; // 获取输入A
        Value* B = input_node->inputs()[1]; // 获取输入B
        Value* AT = graph->insert(aten::t, {A}); // 插入转置操作
        Value* BT = graph->insert(aten::t, {B}); // 插入转置操作
        Value* BTAT = graph->insert(aten::mm, {BT, AT}); // 插入矩阵乘法操作
        n->output()->replaceAllUsesWith(BTAT); // 替换节点的输出使用
        matmuls.push_back(BTAT->node()); // 将新节点加入matmuls向量
      } else if (
          n->matches(
              "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor")) {
        // 如果节点匹配加法操作，则将其输入节点加入队列
        queue.push_back(n->inputs()[0]->node());
        queue.push_back(n->inputs()[1]->node());
      } else {
        AT_ASSERTM(false, "Unsupported node found in a BatchMM tree!"); // 如果找到不支持的节点类型，则抛出异常
      }
    }
    return matmuls; // 返回收集到的矩阵乘法节点向量
  }
};

// 枚举类型，表示操作在左侧还是右侧
enum class Side { LHS, RHS };
static void BatchMMTreeReduce(Block* block, AliasDb& alias_db) {
    auto graph = block->owningGraph();  // 获取当前块所属的图对象

    // Look for trees in the block
    std::unordered_map<Node*, TreeToken> tokens;  // 创建一个映射节点到树令牌的无序映射表
    for (auto node : block->nodes()) {  // 遍历当前块中的所有节点
        if (node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor") &&  // 检查节点是否匹配矩阵乘法操作
            !alias_db.hasWriters(node)) {  // 检查节点是否没有写入操作
            tokens[node] = TreeToken::mm(node);  // 记录矩阵乘法节点的树令牌
        } else if (
            node->matches("aten::t(Tensor self) -> Tensor") &&  // 检查节点是否匹配转置操作
            !alias_db.hasWriters(node)) {  // 检查节点是否没有写入操作
            auto input_it = tokens.find(node->input()->node());  // 查找输入节点的树令牌
            if (input_it != tokens.end()) {
                tokens[node] = TreeToken::transpose(node, input_it->second);  // 记录转置操作节点的树令牌
            }
        } else if (
            node->matches(
                "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") &&  // 检查节点是否匹配加法操作
            !alias_db.hasWriters(node)) {  // 检查节点是否没有写入操作
            Node* lhs = node->inputs()[0]->node();  // 获取加法操作的左操作数节点
            Node* rhs = node->inputs()[1]->node();  // 获取加法操作的右操作数节点
            auto lhs_it = tokens.find(lhs);
            auto rhs_it = tokens.find(rhs);
            // See Note [Overlapping trees] (regarding the uses().size() == 1 check)
            // We could treat a subtree with multiple uses as if it was overlapping.
            // XXX: uses().size() == 1 is also something that guarantees that this
            // transform is valid, because we know for sure that the none of these
            // operands depend on the result of the other. If we were to remove this,
            // we need to compute a transitive closure and actually check the
            // dependencies.
            if (lhs_it != tokens.end() && rhs_it != tokens.end() &&
                lhs->output()->uses().size() == 1 &&
                rhs->output()->uses().size() == 1) {
                if (auto token = TreeToken::add(node, lhs_it->second, rhs_it->second)) {
                    tokens[node] = token;  // 记录加法操作节点的树令牌
                }
            }
        } else {
            for (auto block : node->blocks()) {
                BatchMMTreeReduce(block, alias_db);  // 递归处理节点的子块
            }
        }
    }

    // Merge trees we've found
    for (auto& item : tokens) {
        auto& root = item.second;
        if (!root || root.tree_size < min_fusion_size)
            continue;
        auto matmuls = root.removeTransposesAndGatherMatmuls();  // 移除转置并收集矩阵乘法节点
        WithInsertPoint insert_guard{root.node};
        Node* tree_reduce =
            graph->insertNode(graph->create(Symbol::prim("MMTreeReduce")));  // 创建 MMTreeReduce 节点
        for (Node* matmul : matmuls) {
            tree_reduce->addInput(matmul->inputs().at(0));  // 添加矩阵乘法的左操作数作为输入
        }
        for (Node* matmul : matmuls) {
            tree_reduce->addInput(matmul->inputs().at(1));  // 添加矩阵乘法的右操作数作为输入
        }
        root.node->output()->replaceAllUsesWith(tree_reduce->output());  // 替换树的根节点的输出
        // NB: don't bother with cleaning up after yourself. We'll use DCE for that.
        // 注意：不要担心清理自己。我们将使用 DCE 进行清理。
    }
}

static bool shape_is_fast_for_side(const at::Tensor& other_side_input) {
    // Cutoff chosed by benchmarking on a TITAN V
    return other_side_input.numel() <= 1024 * 2048;  // 检查张量的元素数量是否小于等于阈值
}

RegisterOperators mm_batch_side_reg({Operator(
    prim::MMBatchSide,
    [](const Node* node) -> Operation {
      // 计算除了第一个输入外的其他输入数量
      size_t num_other_side_inputs = node->inputs().size() - 1;
      // 获取节点指定的单侧（Side）属性，并转换为枚举类型
      Side single_side = static_cast<Side>(node->i(Symbol::attr("side")));
      // 返回一个 lambda 表达式，接收一个栈（Stack）作为参数
      return [num_other_side_inputs, single_side](Stack& stack) {
        // 声明侧输入（side_input）和其他侧输入（other_side_inputs）
        at::Tensor side_input;
        std::vector<at::Tensor> other_side_inputs;
        other_side_inputs.reserve(num_other_side_inputs);
        // 遍历栈中除了最后一个输入之外的其他输入，将它们转换为 Tensor 并存储在 other_side_inputs 中
        for (auto it = stack.end() - num_other_side_inputs; it != stack.end();
             ++it) {
          other_side_inputs.push_back(std::move(*it).toTensor());
        }
        // 从栈中移除 num_other_side_inputs 个元素
        drop(stack, num_other_side_inputs);
        // 弹出栈顶的 side_input
        pop(stack, side_input);

        // 获取其他输入中的任意一个
        auto any_other_input = other_side_inputs[0];
        // 如果所有其他输入具有相同的形状并且形状适合于侧（side）操作
        if (have_same_shape(other_side_inputs) &&
            shape_is_fast_for_side(other_side_inputs[0])) {
          // 根据单侧（side）是左侧还是右侧，拼接其他侧输入
          auto other_side_input =
              at::cat(other_side_inputs, single_side == Side::LHS ? 1 : 0);
          // 执行矩阵乘法操作
          auto mm_out = single_side == Side::LHS
              ? side_input.mm(other_side_input)  // 左侧矩阵乘法
              : other_side_input.mm(side_input);  // 右侧矩阵乘法
          // 将 mm_out 按指定维度分块，并将结果压入栈中
          auto outputs = at::chunk(
              mm_out,
              num_other_side_inputs,
              /*dim=*/single_side == Side::LHS ? 1 : 0);
          stack.insert(
              stack.end(),
              std::make_move_iterator(outputs.begin()),
              std::make_move_iterator(outputs.end()));
        } else {
          // 如果不能进行快速操作，根据单侧是左侧还是右侧，逐个计算结果并压入栈中
          if (single_side == Side::LHS) {
            for (at::Tensor& other : other_side_inputs) {
              stack.emplace_back(side_input.mm(other));
            }
          } else {
            for (at::Tensor& other : other_side_inputs) {
              stack.emplace_back(other.mm(side_input));
            }
          }
        }
      };
    },
    aliasAnalysisIsSpecialCase())});
// 定义静态函数 gatherIndependentMMUses，用于收集与给定 value 独立的矩阵乘法操作节点
static std::pair<std::vector<Node*>, std::vector<Node*>> gatherIndependentMMUses(
    Value* value,                  // 输入参数 value，表示要分析的值
    AliasDb& alias_db) {           // 别名数据库，用于别名分析

  // 定义内部 lambda 函数 postprocess，用于后处理矩阵乘法操作节点集合
  const auto postprocess = [&](std::vector<Node*> mms) {
    if (mms.empty()) {            // 如果集合为空，直接返回空集合
      return mms;
    }
    // 按照节点的拓扑顺序对 mms 进行排序
    std::sort(mms.begin(), mms.end(), [](Node* n, Node* m) {
      return n->isBefore(m);
    });
    // 过滤掉依赖的矩阵乘法操作节点
    for (const auto i : c10::irange(mms.size())) {
      if (mms[i] == nullptr)
        continue;
      for (size_t j = i + 1; j < mms.size(); ++j) {
        if (mms[j] == nullptr)
          continue;
        // 如果节点 j 不能在节点 i 前移动，则将节点 j 置为 nullptr
        if (!alias_db.couldMoveBeforeTopologically(mms[j], mms[i])) {
          mms[j] = nullptr;
        }
      }
    }
    // 返回过滤后的非空节点集合
    return c10::filter(mms, [](Node* n) { return n != nullptr; });
  };

  Block* block = value->node()->owningBlock();  // 获取 value 所在的基本块
  std::vector<Node*> lhses;   // 存储将 value 作为左操作数的节点集合
  std::vector<Node*> rhses;   // 存储将 value 作为右操作数的节点集合

  // 遍历 value 的所有使用点
  for (Use u : value->uses()) {
    // 检查使用点所在的基本块是否与 value 所在基本块相同，并且是矩阵乘法操作
    if (u.user->owningBlock() == block &&
        u.user->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor") &&
        !alias_db.hasWriters(u.user)) {
      // 根据使用点的偏移确定是左操作数还是右操作数
      if (u.offset == 0 && u.user->inputs()[1] != value) {
        lhses.push_back(u.user);  // 将该节点添加到左操作数集合
      } else if (u.offset == 1 && u.user->inputs()[0] != value) {
        rhses.push_back(u.user);  // 将该节点添加到右操作数集合
      }
    }
  }

  // 返回处理后的左右操作数节点集合
  return std::make_pair(
      postprocess(std::move(lhses)), postprocess(std::move(rhses)));
}

// 定义静态函数 BatchMMSide，用于处理批量矩阵乘法的操作
static void BatchMMSide(Block* block, AliasDb& alias_db) {
  // NB: 8 是当前的循环展开因子
  static constexpr size_t how_many_is_many = 8;

  // 定义内部 lambda 函数 batch_side，用于处理矩阵乘法操作的批量化
  const auto batch_side = [&](std::vector<Node*>& mms, Side side) {
    AT_ASSERT(!mms.empty());    // 断言 mms 非空
    // 逆序遍历 mms，保证节点间的拓扑顺序关系
    for (int64_t i = static_cast<int64_t>(mms.size()) - 2; i >= 0; --i) {
      // 检查是否可以在拓扑上将 mms[i] 移动到 mms[i + 1] 之前
      bool move_ok = alias_db.moveBeforeTopologicallyValid(mms[i], mms[i + 1]);
      AT_ASSERT(move_ok);       // 断言移动操作有效
    }
    // 设置插入点为 mms[0] 处
    WithInsertPoint insert_guard{mms[0]};
    Graph* graph = mms[0]->owningGraph();  // 获取所在图
    // 创建批量矩阵乘法节点 batch_mm
    Node* batch_mm = graph->create(
        prim::MMBatchSide,        // 操作类型为 MM 批量化
        /*inputs=*/{},            // 输入为空
        /*num_outputs=*/mms.size());  // 输出数量与 mms 大小相同
    graph->insertNode(batch_mm); // 将节点插入图中
    batch_mm->i_(Symbol::attr("side"), static_cast<int>(side));  // 设置属性 side
    Value* const_side = mms[0]->inputs().at(side == Side::LHS ? 0 : 1); // 获取常量输入值
    batch_mm->addInput(const_side);  // 添加常量输入
    // 遍历 mms 中的节点，将其作为输入添加到 batch_mm 中，并替换其输出使用点
    for (const auto i : c10::irange(mms.size())) {
      batch_mm->addInput(mms[i]->inputs().at(side == Side::LHS ? 1 : 0));
      mms[i]->output()->replaceAllUsesWith(batch_mm->outputs().at(i));
    }
  };

  std::unordered_set<Value*> considered_values;  // 存储已考虑过的值的集合
  // 遍历基本块中的所有节点
  for (Node* node : block->nodes()) {
    // 如果节点匹配 "aten::mm(Tensor self, Tensor mat2) -> Tensor"，并且在别名数据库中没有写操作
    if (node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor") &&
        !alias_db.hasWriters(node)) {
      
      // 遍历节点的所有输入值
      for (Value* input : node->inputs()) {
        
        // 如果该输入值已经被考虑过，跳过处理
        if (!considered_values.emplace(input).second) {
          continue;
        }
        
        // 收集独立的 mm 使用情况
        auto uses_with_many = gatherIndependentMMUses(input, alias_db);
        
        // 如果左侧的使用次数达到阈值 how_many_is_many，则进行批处理
        if (uses_with_many.first.size() >= how_many_is_many) {
          batch_side(uses_with_many.first, Side::LHS);
        }
        
        // 如果右侧的使用次数达到阈值 how_many_is_many，则进行批处理
        if (uses_with_many.second.size() >= how_many_is_many) {
          batch_side(uses_with_many.second, Side::RHS);
        }
      }
    } else {
      // 如果节点不匹配指定条件，则对其子块进行递归批处理
      for (Block* subblock : node->blocks()) {
        BatchMMSide(subblock, alias_db);
      }
    }
  }
}

// 检查图中是否存在矩阵乘积操作符
static bool hasMMOperators(std::shared_ptr<Graph>& graph) {
  // 创建深度优先图节点迭代器
  DepthFirstGraphNodeIterator it(graph);
  // 初始化节点指针
  Node* n = nullptr;
  // 迭代图中的每个节点
  while ((n = it.next()) != nullptr) {
    // 检查节点是否匹配矩阵乘积操作符
    if (n->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
      return true; // 如果找到匹配的操作符，返回true
    }
  }
  return false; // 如果未找到匹配的操作符，返回false
}

// 执行批量矩阵乘积优化
void BatchMM(std::shared_ptr<Graph>& graph) {
  // 如果图中不存在矩阵乘积操作符，则直接返回
  if (!hasMMOperators(graph)) {
    return;
  }
  // 创建图的别名数据库
  AliasDb alias_db(graph);
  // 在图的块上执行批量矩阵乘积树减少操作
  BatchMMTreeReduce(graph->block(), alias_db);
  // 在图的块上执行批量矩阵乘积侧面操作
  BatchMMSide(graph->block(), alias_db);
  // 消除死代码
  EliminateDeadCode(graph);

  // 可能由于转置重排而创建了之前不存在的连续转置序列。

  // 张量类型属性不能保证正确性
  PeepholeOptimize(graph, /*disable_shape_peepholes*/ true);
}

// 命名空间jit下的torch命名空间结束
} // namespace jit
// torch命名空间结束
} // namespace torch
```