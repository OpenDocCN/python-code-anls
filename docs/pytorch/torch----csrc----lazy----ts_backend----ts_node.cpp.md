# `.\pytorch\torch\csrc\lazy\ts_backend\ts_node.cpp`

```py
// 包含 Torch 框架的头文件，用于调试工具和节点定义
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

// 定义一个匿名命名空间，限定以下函数和变量的作用域
namespace {
// 如果 LTC_ENABLE_SOURCE_INFO 环境变量存在，则返回 Python 中的第一个用户帧
std::string GetFirstUserFrameInPythonIfEnabled() {
  // 静态变量，存储 LTC_ENABLE_SOURCE_INFO 环境变量的值
  static const auto LTC_ENABLE_SOURCE_INFO =
      std::getenv("LTC_ENABLE_SOURCE_INFO");
  // 如果环境变量不存在，返回空字符串
  if (!LTC_ENABLE_SOURCE_INFO) {
    return {};
  }
  // 调用 Torch 框架的函数，获取 Python 中的第一个用户帧
  return torch::lazy::GetFirstUserFrameInPython();
}
} // namespace

// 定义 Torch 框架的 lazy 命名空间
namespace torch {
namespace lazy {

// 计算操作数列表、形状和种子的哈希值
static hash_t OperandHashes(
    const OpList& operands,
    const c10::ArrayRef<Shape>& shapes,
    const hash_t& seed,
    bool bakeInSizes) {
  // 初始哈希值为种子
  hash_t hash = seed;
  // 遍历操作数列表
  for (auto& operand : operands) {
    // 如果操作数为空，则将特定值的哈希与当前哈希组合
    if (!operand) {
      hash = HashCombine(hash, static_cast<uint64_t>(kNullOpt));
      continue;
    }
    // 根据 bakeInSizes 标志选择计算操作数的哈希值或形状哈希值
    auto operand_hash = bakeInSizes ? operand.shapeHash() : operand.hash();
    // 将计算得到的哈希值与当前哈希组合
    hash = HashCombine(hash, operand_hash);
  }
  // 遍历形状列表，计算每个形状的哈希值并与当前哈希组合
  for (auto& shape : shapes) {
    hash = HashCombine(hash, shape.hash(bakeInSizes));
  }
  // 返回最终的哈希值
  return hash;
}

// 构造函数：使用操作种类、操作数、形状向量、输出数量和哈希种子初始化 TsNode 对象
TsNode::TsNode(
    OpKind op,
    OpList operands,
    std::vector<Shape>&& shapes,
    size_t num_outputs,
    hash_t hash_seed)
    : Node(op, operands, std::move(shapes), num_outputs) {
  // 组合操作种类的哈希和给定的哈希种子
  hash_seed = HashCombine(op.hash(), hash_seed);
  // 计算操作数、形状和哈希种子的哈希值，存储到 shape_hash_ 成员变量
  shape_hash_ = OperandHashes(operands, this->shapes(), hash_seed, true);
  // 如果启用动态形状，则使用相同的操作数和形状计算另一个哈希值，否则使用先前计算的 shape_hash_
  dag_hash_ =
      (enableDynamicShape()
           ? OperandHashes(operands, this->shapes(), hash_seed, false)
           : shape_hash_);
}

// 构造函数：使用操作种类、操作数、形状计算函数、输出数量和哈希种子初始化 TsNode 对象
TsNode::TsNode(
    OpKind op,
    OpList operands,
    const std::function<Shape()>& shape_fn,
    size_t num_outputs,
    hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {
  // 添加计算得到的形状到形状向量中
  addComputedShape(shape_fn);
}

// 构造函数：使用操作种类、操作数、输出数量和哈希种子初始化 TsNode 对象
TsNode::TsNode(OpKind op, OpList operands, size_t num_outputs, hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {}

// 构造函数：使用操作种类、形状、输出数量和哈希种子初始化 TsNode 对象
TsNode::TsNode(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed)
    : TsNode(op, {}, {std::move(shape)}, num_outputs, hash_seed) {}

// 返回节点的 dag_hash_ 成员变量，即动态形状哈希值
hash_t TsNode::hash() const {
  return dag_hash_;
}

// 返回节点的 shape_hash_ 成员变量，即形状哈希值
hash_t TsNode::shapeHash() const {
  return shape_hash_;
}

// 获取 Python 堆栈跟踪信息的函数
const std::string TsNode::getPythonStacktrace() const {
  return GetFirstUserFrameInPythonIfEnabled();
}

// 构造函数：使用操作数列表初始化 TensorList 对象
TensorList::TensorList(OpList values)
    : TsNode(
          /*op=*/ClassOpKind(),
          /*operands=*/values,
          /*shapes=*/std::vector<Shape>(),
          /*num_outputs=*/1,
          /*hash_seed=*/kHashSeed) {}

// 将 TensorList 对象降低为 torch::jit::GraphFunction 的操作向量
TSOpVector TensorList::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const {
  // 创建一个 torch::jit::Value* 的向量，存储操作数对应的输出操作
  std::vector<torch::jit::Value*> tensor_list;
  // 检查操作数列表不能为空
  TORCH_CHECK(!operands().empty());
  // 遍历操作数列表，将每个操作数的输出操作添加到 tensor_list 中
  for (const torch::lazy::Output& operand : operands()) {
    tensor_list.emplace_back(loctx->GetOutputOp(operand));
  }
  // 获取 GraphFunction 的图形，并插入一个新节点，创建一个操作数类型和 tensor_list 的列表节点
  auto graph = function->graph();
  auto listnode =
      graph->insertNode(graph->createList(tensor_list[0]->type(), tensor_list));
  // 返回包含列表节点输出的 TSOpVector
  return {listnode->output()};
}

} // namespace lazy
} // namespace torch
```