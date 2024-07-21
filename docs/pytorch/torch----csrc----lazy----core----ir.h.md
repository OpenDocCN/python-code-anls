# `.\pytorch\torch\csrc\lazy\core\ir.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/symbol.h>
// 包含 ATen 库的符号定义

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
// 包含 C++ 标准库头文件

#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Flags.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/core/shape.h>
// 包含 Torch 和 C10 库的特定头文件

C10_DECLARE_bool(ltc_enable_dynamic_shapes);
// 声明一个 C10 库中的布尔型全局变量 ltc_enable_dynamic_shapes

namespace torch {
namespace lazy {

static const hash_t kHashSeed(static_cast<uint32_t>(0x5a2d296e9));
// 定义并初始化一个静态常量 hash_t 类型的 kHashSeed

class Node;
struct Output;
struct Value;
// 声明几个结构体和类，用于表示计算图中的节点、输出和值

using NodePtr = std::shared_ptr<Node>;
// 使用 std::shared_ptr<Node> 别名 NodePtr

// 表示节点可以关联的操作种类。
struct TORCH_API OpKind {
  OpKind() = default;
  explicit OpKind(c10::Symbol op) : op(op) {}

  bool operator==(const OpKind& rhs) const {
    return op == rhs.op;
  }
  bool operator!=(const OpKind& rhs) const {
    return !operator==(rhs);
  }
  bool operator<(const OpKind& rhs) const {
    return c10::unique_t(op) < c10::unique_t(rhs.op);
  }
  // 比较运算符重载，用于操作符种类的比较

  hash_t hash() const;
  // 返回操作符种类的哈希值

  std::string ToString() const {
    return op.toQualString();
  }
  // 返回操作符种类的字符串表示

  // 获取现有操作对象，或创建一个新的。特定于惰性张量的操作应位于 'lazy_tensors::' 命名空间中。
  static OpKind Get(const std::string& name);
  // 静态方法，根据名称获取或创建操作种类对象

  c10::Symbol op;
  // 操作符的符号对象
};

inline std::ostream& operator<<(std::ostream& stream, const OpKind& op) {
  stream << op.ToString();
  return stream;
}
// 重载流输出运算符，用于将操作符种类对象输出到流中

using OpList = c10::ArrayRef<Value>;
// 使用 c10::ArrayRef<Value> 别名 OpList

hash_t OperandHashes(
    const OpList& operands,
    const hash_t& seed,
    bool bakeInSizes);
// 计算操作数列表的哈希值

// 图中的一个节点。需要降低存储额外数据的操作节点应继承此类，并在其中添加一个特定的成员。
// 例如，常量可能会创建一个新的 NodeConstant 类（继承自 Node），其中包含一个额外的 lazy_tensors::Literal 字段，
// 或者张量值可能会创建一个带有计算客户端数据句柄的 NodeTensor。
// 此处是类的说明注释，描述了节点类的基本结构和用法。
// 定义 TORCH_API 类 Node，表示计算图中的节点
class TORCH_API Node {
 public:
  // 静态方法，启用动态形状
  static bool enableDynamicShape();

  // 构造函数，使用操作名称和输出数量创建一个新的节点
  Node(OpKind op, size_t num_outputs);

  // 构造函数，使用操作名称、操作数列表和形状向量创建节点
  Node(
      OpKind op,
      OpList operands,
      std::vector<Shape>&& shapes,
      size_t num_outputs = 1);

  // 构造函数，使用操作名称、操作数列表和生成形状的函数创建节点
  Node(
      OpKind op,
      OpList operands,
      const std::function<Shape()>& shape_fn,
      size_t num_outputs = 1);

  // 构造函数，使用操作名称和操作数列表创建没有形状信息的节点
  Node(OpKind op, OpList operands, size_t num_outputs = 1);

  // 构造函数，使用操作名称和形状创建节点，没有操作数列表
  Node(OpKind op, Shape shape, size_t num_outputs = 1);

  // 虚析构函数
  virtual ~Node();

  // 获取节点的操作类型
  const OpKind& op() const {
    return op_;
  }

  // 获取节点的输出数量
  size_t num_outputs() const {
    return num_outputs_;
  }

  // 获取节点的全部形状
  virtual c10::ArrayRef<Shape> shapes() const;

  // 获取指定输出索引处的形状
  virtual const Shape& shape(size_t output_index = 0) const;

  // 添加由 shape_fn 计算的形状
  void addComputedShape(const std::function<Shape()>& shape_fn);

  // 如果未缓存，则使用提供的 shape_fn 计算形状
  Shape computeShape(const std::function<Shape()>& shape_fn);

  // 获取节点的全部操作数
  virtual const std::vector<Output>& operands() const;

  // 获取指定索引处的操作数
  virtual const Output& operand(size_t i) const;

  // 获取索引处的操作数（如果索引有效），否则返回 kNullOutput
  virtual const Output& nullable_operand(size_t i) const;

  // 返回用于查找编译图的 DAG 的哈希值
  virtual hash_t hash() const = 0;

  // 返回用于形状缓存的 DAG 的哈希值
  virtual hash_t shapeHash() const = 0;

  // 获取节点的元数据
  const MetaData& metadata() const {
    return metadata_;
  }

  // 获取用户元数据的指针
  UserMetaData* user_metadata() const {
    return user_metadata_.get();
  }

  // 设置用户元数据，返回之前的用户元数据指针
  std::shared_ptr<UserMetaData> SetUserMetadata(
      std::shared_ptr<UserMetaData> user_meta) {
    std::swap(user_metadata_, user_meta);
    // 返回节点的用户定义元数据
    return user_meta;
  }

  // 返回节点对象的字符串表示形式
  virtual std::string ToString() const;

 private:
  // 操作节点的唯一标识符
  OpKind op_;
  // 节点的输出数量，默认为1
  size_t num_outputs_ = 1;

  // IR节点附加的元数据
  MetaData metadata_;
  // 用户可以附加的用户定义元数据对象，派生自UserMetaData类
  std::shared_ptr<UserMetaData> user_metadata_;

 protected:
  // 添加节点的操作数，可以指定索引，默认为0
  void AddOperand(NodePtr node, size_t index = 0);

  // 节点的形状向量
  std::vector<Shape> shapes_;
  // 节点持有其操作数的实际引用
  std::vector<NodePtr> operands_;
  // 输出不持有节点的引用，也不持有使用它们的对象的引用，以避免循环引用计数问题
  std::vector<Output> operands_as_outputs_;
};

// 结束了一个命名空间 'lazy' 和 'torch'

// 输出流重载函数，将 Node 对象转换为流输出
inline std::ostream& operator<<(std::ostream& stream, const Node& node) {
  // 调用 Node 的 ToString 方法，并将结果输出到流中
  stream << node.ToString();
  return stream;
}

// Note: Keep this version of NodeCast for smooth PyTorch/XLA migration, and
// clean up once the migration is done.
// NodeCast 模板函数，根据节点的操作类型安全地将节点指针转换为指定类型 T 的指针
template <typename T>
const T* NodeCast(const Node* node, OpKind op) {
  // 如果节点的操作类型不匹配，则返回空指针
  if (op != node->op()) {
    return nullptr;
  }
#ifdef NDEBUG
  // 在 Release 模式下，使用静态转换 static_cast
  return static_cast<const T*>(node);
#else
  // 在 Debug 模式下，使用动态转换 dynamic_cast，以确保类型安全
  return &dynamic_cast<const T&>(*node);
#endif
}

// NodeCast 模板函数的另一版本，不需要指定 OpKind，但需要通过类型 T 的 ClassOpKind 匹配操作类型
template <typename T>
const T* NodeCast(const Node* node) {
  // 如果类型 T 的 ClassOpKind 不等于节点的操作类型，则返回空指针
  if (T::ClassOpKind() != node->op()) {
    return nullptr;
  }
  // 使用动态转换 dynamic_cast，因为有些 IR 类型共享相同的操作类型，需要确保安全转换
  return dynamic_cast<const T*>(node);
}

// 表示节点输出的结构体 Output
// 每个输出由节点和索引组成，完全确定每个单独输出
struct TORCH_API Output {
  // 哈希函数对象，用于输出的哈希计算
  struct Hasher {
    size_t operator()(const Output& output) const;
  };

  // 默认构造函数
  Output() = default;
  // 显式构造函数，使用节点指针和索引初始化输出
  explicit Output(const Node* node, size_t index = 0)
      : node(node), index(index) {}

  // 获取输出的哈希值
  hash_t hash() const;
  // 获取输出形状的哈希值
  hash_t shapeHash() const;

  // 判断两个输出是否相等
  bool operator==(const Output& rhs) const {
    return node == rhs.node && index == rhs.index;
  }

  // 用于比较要构造的节点的操作数和要重用的节点
  bool operator==(const Value& rhs) const;

  // 判断两个输出是否不相等
  bool operator!=(const Output& rhs) const {
    return !operator==(rhs);
  }

  // 获取输出的形状
  const Shape& shape() const {
    return node->shape(index);
  }

  // 将输出转换为字符串表示
  std::string ToString() const;

  // 提供输出的节点
  const Node* node{nullptr};
  // 输出引用节点的索引
  size_t index{0};
};

// 输出流重载函数，将 Output 对象转换为流输出
inline std::ostream& operator<<(std::ostream& stream, const Output& output) {
  // 调用 Output 的 ToString 方法，并将结果输出到流中
  stream << output.ToString();
  return stream;
}

// 使用 Output 作为键的无序映射的别名
template <typename T>
using OutputMap = std::unordered_map<Output, T, Output::Hasher>;

// 表示节点输入/操作数的结构体 Value
struct TORCH_API Value {
  // 默认构造函数
  Value() = default;
  /* implicit */ Value(NodePtr&& node, size_t index = 0)
      : node(std::move(node)), index(index) {}
  /* implicit */ Value(const NodePtr& node, size_t index = 0)
      : node(node), index(index) {}

  // 获取哈希值
  hash_t hash() const;
  // 获取形状的哈希值
  hash_t shapeHash() const;

  // 转换为 bool 类型，判断节点是否非空
  operator bool() const {
    return node != nullptr;
  }

  // 转换为 Output 类型，获取输出对象
  operator Output() const {
    return Output(node.get(), index);
  }

  // 获取形状
  const Shape& shape() const {
    return node->shape(index);
  }

  // 重载箭头操作符，获取节点指针
  Node* operator->() const {
    return node.get();
  }

  // 节点指针
  NodePtr node;
  // 输入索引，默认为 0
  size_t index = 0;
};

// 结束了命名空间 'lazy' 和 'torch'

} // namespace lazy
} // namespace torch

// 显式实例化模板，使得 ArrayRef<Value> 可以正常工作
namespace c10 {
template class at::ArrayRef<torch::lazy::Value>;
} // namespace c10
```