# `.\pytorch\torch\csrc\lazy\core\ir.cpp`

```py
// 引入 Torch Lazy 模块的头文件，用于后续操作
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>

// 开启动态形状缓存的选项（禁用形状的哈希）
C10_DEFINE_bool(
    ltc_enable_dynamic_shapes,
    false,
    "Whether dynamic shape is enabled");

// Torch Lazy 命名空间开始
namespace torch {
namespace lazy {

// 定义空输出的常量 kNullOutput
static const torch::lazy::Output kNullOutput = torch::lazy::Output();

// Output 类的哈希计算器重载，用于计算输出的哈希值
size_t Output::Hasher::operator()(const Output& output) const {
  return StdHashCombine(
      reinterpret_cast<std::ptrdiff_t>(output.node), output.index);
}

// 计算输出的哈希值
hash_t Output::hash() const {
  return HashCombine(node->hash(), Hash(index));
}

// 计算输出形状的哈希值
hash_t Output::shapeHash() const {
  return HashCombine(node->shapeHash(), Hash(index));
}

// 将输出转换为字符串表示
std::string Output::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ", index=" << index;
  return ss.str();
}

// 比较操作，判断两个值是否相等
bool Output::operator==(const Value& rhs) const {
  // 可能存在一个或两个 kNullValue，其 node 为 nullptr
  return (!node == !rhs.node) &&
      (!node || (node->hash() == rhs.node->hash() && index == rhs.index));
}

// 计算值的哈希值
hash_t Value::hash() const {
  return HashCombine(node->hash(), Hash(index));
}

// 计算值的形状哈希值
hash_t Value::shapeHash() const {
  return HashCombine(node->shapeHash(), Hash(index));
}

// 根据操作名获取操作类型
OpKind OpKind::Get(const std::string& name) {
  return OpKind(c10::Symbol::fromQualString(name));
}

// 计算操作类型的哈希值
hash_t OpKind::hash() const {
  return StringHash(op.toQualString());
}

// 是否启用动态形状
bool Node::enableDynamicShape() {
  // 静态变量，根据环境变量 "LTC_ENABLE_DYNAMIC_SHAPES" 判断是否启用
  static bool enabled = std::getenv("LTC_ENABLE_DYNAMIC_SHAPES") != nullptr;
  return enabled || FLAGS_ltc_enable_dynamic_shapes;
}

// Node 类的构造函数，初始化操作类型、输出数量和调试元数据
Node::Node(OpKind op, size_t num_outputs)
    : op_(op), num_outputs_(num_outputs), metadata_(GetMetaDataIfDebugging()) {}

// Node 类的构造函数，初始化操作类型、操作列表、形状和输出数量，并移动形状到节点内部
Node::Node(
    OpKind op,
    OpList operands,
    std::vector<Shape>&& shapes,
    size_t num_outputs)
    : Node(op, num_outputs) {
  // 将形状移动到节点内部
  shapes_.insert(
      shapes_.end(),
      std::make_move_iterator(shapes.begin()),
      std::make_move_iterator(shapes.end()));

  // 遍历操作列表，添加操作数到节点
  for (auto& operand : operands) {
    // 理想情况下，应由叶子节点类过滤可选操作数，但此处处理更简单
    if (!operand) {
      continue;
    }
    AddOperand(operand.node, operand.index);
  }
}

// Node 类的构造函数，初始化操作类型、操作列表、形状函数和输出数量
Node::Node(
    OpKind op,
    OpList operands,
    const std::function<Shape()>& shape_fn,
    size_t num_outputs)
    : Node(op, operands, std::vector<Shape>{}, num_outputs) {
  // 添加计算得到的形状
  addComputedShape(shape_fn);
}

// Node 类的构造函数，初始化操作类型、操作列表和输出数量
Node::Node(OpKind op, OpList operands, size_t num_outputs)
    : Node(op, operands, std::vector<Shape>{}, num_outputs) {}

// Node 类的构造函数，初始化操作类型、形状和输出数量
Node::Node(OpKind op, Shape shape, size_t num_outputs) : Node(op, num_outputs) {
  shapes_.push_back(std::move(shape));
}

// Node 类的析构函数，默认析构
Node::~Node() = default;

// 获取 IR 节点的完整形状
// （以下部分未提供代码，不在注释范围内）

// Torch Lazy 命名空间结束
} // namespace lazy
} // namespace torch
// 返回节点的形状数组的引用
c10::ArrayRef<Shape> Node::shapes() const {
  return shapes_;
}

// 根据输出索引获取输出的形状
const Shape& Node::shape(size_t output_index) const {
  return shapes_.at(output_index);
}

// 添加通过 shape_fn 计算得到的形状
void Node::addComputedShape(const std::function<Shape()>& shape_fn) {
  shapes_.push_back(computeShape(shape_fn));
}

// 定义 ShapeCache 类型为具有特定哈希和形状的缓存
using ShapeCache = Cache<hash_t, Shape, HashReducer>;

// 使用提供的 shape_fn 计算形状
Shape Node::computeShape(const std::function<Shape()>& shape_fn) {
  // 创建静态的 ShapeCache 对象
  static ShapeCache* cache = new ShapeCache(FLAGS_torch_lazy_shape_cache_size);

  // 计算形状的哈希值
  auto hash = shapeHash();
  // 尝试从缓存中获取形状
  auto shape = cache->Get(hash);
  // 如果缓存中不存在该形状，则计算并添加到缓存中
  if (shape == nullptr) {
    shape = cache->Add(hash, std::make_shared<Shape>(shape_fn()));
  }
  return *shape;
}

// 返回节点的操作数数组的引用
const std::vector<Output>& Node::operands() const {
  return operands_as_outputs_;
}

// 根据索引返回操作数的引用
const Output& Node::operand(size_t i) const {
  return operands_as_outputs_.at(i);
}

// 根据索引返回可空操作数的引用，如果索引超出数组大小则返回 kNullOutput
const Output& Node::nullable_operand(size_t i) const {
  // 在这里使用 kNullOutput 而不是 kNullValue，以避免隐式转换，从而能返回引用
  return i < operands_as_outputs_.size() ? operand(i) : kNullOutput;
}

// 返回节点的字符串表示形式
std::string Node::ToString() const {
  std::stringstream ss;
  // 输出节点的形状和操作
  ss << shapes() << " " << op();
  // 如果节点有多个输出，输出输出数量
  if (num_outputs() > 1) {
    ss << ", num_outputs=" << num_outputs();
  }
  // 如果节点的元数据作用域不为空，输出作用域
  if (!metadata().scope.empty()) {
    ss << ", scope=" << metadata().scope;
  }
  // 输出节点的简短帧信息
  EmitShortFrameInfo(ss, metadata().frame_info);
  return ss.str();
}

// 向节点添加操作数
void Node::AddOperand(NodePtr node, size_t index) {
  // 检查索引是否小于节点的输出数量
  TORCH_CHECK_LT(index, node->num_outputs());
  // 将节点添加到操作数数组中
  operands_.push_back(node);
  // 向操作数输出数组中添加对应的输出
  operands_as_outputs_.emplace_back(operands_.back().get(), index);
}
```