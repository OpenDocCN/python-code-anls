# `.\pytorch\torch\csrc\jit\passes\peephole_dict_idioms.cpp`

```py
// 包含 Torch 库中的别名分析和优化的头文件
#include <torch/csrc/jit/ir/alias_analysis.h>
// 包含 Torch 库中的字典常见优化模式的头文件
#include <torch/csrc/jit/passes/peephole_dict_idioms.h>

// Torch 命名空间开始
namespace torch {
namespace jit {

// 匿名命名空间中定义了 DictNodeImplBase 类
namespace {

// DictNodeImplBase 类，虚基类定义
class DictNodeImplBase {
 public:
  virtual ~DictNodeImplBase() = default;

  // 纯虚函数，检查是否包含给定的 IValue
  virtual bool contains(const IValue&) const = 0;
  // 纯虚函数，返回字典大小
  virtual size_t size() const = 0;
  // 纯虚函数，根据给定的 IValue 返回对应的 Value 指针
  virtual Value* get(const IValue&) const = 0;

  // 判断是否可以进行优化的函数
  bool canOptimize() {
    // 当没有重叠键和非常量键时，可以进行优化
    return !has_overlap_ && !has_non_const_key_;
  }

 protected:
  bool has_overlap_ = false;      // 是否有重叠的键
  bool has_non_const_key_ = false;  // 是否有非常量的键
};

// 模板类 DictNodeImpl，继承自 DictNodeImplBase
template <class KeyType>
class DictNodeImpl : public DictNodeImplBase {
 public:
  // 构造函数，接受一个转换函数和字典创建节点作为参数
  DictNodeImpl(
      std::function<KeyType(const IValue&)> ivalue_converter,
      Node* dict_creation_node)
      : ivalue_converter_(std::move(ivalue_converter)) {
    // 遍历字典创建节点的输入
    for (size_t i = 0; i < dict_creation_node->inputs().size(); i += 2) {
      auto key_opt = toIValue(dict_creation_node->input(i));

      // 如果无法转换为 IValue，则键非常量
      if (key_opt == c10::nullopt) {
        has_non_const_key_ = true;
        continue;
      }

      // 使用转换函数将键转换为 KeyType 类型
      KeyType key = ivalue_converter_(*key_opt);

      // 如果字典中不存在该键，则插入键值对；否则标记存在重叠键
      if (dict_.find(key) == dict_.end()) {
        dict_.emplace(key, dict_creation_node->input(i + 1));
      } else {
        has_overlap_ = true;
      }
    }
  }

  // 实现基类的虚函数，检查字典是否包含给定的 IValue
  bool contains(const IValue& ivalue) const override {
    auto key = ivalue_converter_(ivalue);
    return dict_.find(key) != dict_.end();
  }

  // 实现基类的虚函数，返回字典的大小
  size_t size() const override {
    return dict_.size();
  }

  // 实现基类的虚函数，根据给定的 IValue 返回对应的 Value 指针
  Value* get(const IValue& ivalue) const override {
    auto val = ivalue_converter_(ivalue);
    auto loc = dict_.find(val);
    if (loc != dict_.end()) {
      return loc->second;
    }
    // 如果找不到对应键的值，抛出错误
    TORCH_CHECK(false, "Cannot get non-existent key");
  }

 private:
  std::unordered_map<KeyType, Value*> dict_;  // 内部存储的字典
  std::function<KeyType(const IValue&)> ivalue_converter_;  // 转换函数
};

// DictNode 类定义开始
class DictNode {
 public:
  // 构造函数，接受一个字典创建节点作为参数
  explicit DictNode(Node* dict_creation_node) {
    auto dict_type = dict_creation_node->output()->type();
    auto key_value_types = dict_type->containedTypes();

    // 检查字典类型是否包含两种类型（键和值）
    TORCH_CHECK(
        key_value_types.size() == 2, "Dict must have 2 contained types");
    const auto& key_type = key_value_types[0];


这部分代码定义了 Torch 中用于字典优化的一些数据结构和算法，主要是为了在 JIT 编译器中对字典进行高效的操作和分析。
    switch (key_type->kind()) {
      case TypeKind::IntType: {
        // 定义一个函数对象 ivalue_converter，用于将 IValue 转换为 int64_t
        auto ivalue_converter = [](const IValue& ival) { return ival.toInt(); };
        // 使用 int64_t 类型的 ivalue_converter 创建 DictNodeImpl 对象
        impl_ = std::make_unique<DictNodeImpl<int64_t>>(
            std::move(ivalue_converter), dict_creation_node);
        // 跳出 case 分支
        break;
      }

      case TypeKind::FloatType: {
        // 定义一个函数对象 ivalue_converter，用于将 IValue 转换为 double
        auto ivalue_converter = [](const IValue& ival) {
          return ival.toDouble();
        };
        // 使用 double 类型的 ivalue_converter 创建 DictNodeImpl 对象
        impl_ = std::make_unique<DictNodeImpl<double>>(
            std::move(ivalue_converter), dict_creation_node);
        // 跳出 case 分支
        break;
      }

      case TypeKind::StringType: {
        // 定义一个函数对象 ivalue_converter，用于将 IValue 转换为 std::string
        auto ivalue_converter = [](const IValue& ival) {
          return *ival.toString();
        };
        // 使用 std::string 类型的 ivalue_converter 创建 DictNodeImpl 对象
        impl_ = std::make_unique<DictNodeImpl<std::string>>(
            std::move(ivalue_converter), dict_creation_node);
        // 跳出 case 分支
        break;
      }

      default:
        // 默认情况下，设置 impl_ 为 nullptr
        impl_ = nullptr;
    }
  }

  // 返回 impl_ 指向的对象是否可以优化的布尔值
  bool canOptimize() const {
    if (impl_) {
      return impl_->canOptimize();
    }
    return false;
  }

  // 返回 impl_ 指向的对象中的元素个数
  size_t size() const {
    if (impl_) {
      return impl_->size();
    }
    return 0;
  }

  // 根据 key 查找并返回对应的值，如果不存在则返回 std::nullopt
  std::optional<Value*> getOrNullopt(const IValue& key) const {
    if (impl_ && impl_->contains(key)) {
      return impl_->get(key);
    }
    return c10::nullopt;
  }

 private:
  // 持有 DictNodeImplBase 的独占指针
  std::unique_ptr<DictNodeImplBase> impl_;
};

// 检查值是否为字典类型
bool isDict(Value* v) {
  return v->type()->castRaw<DictType>() != nullptr;
}

// PeepholeOptimizeDictIdiomsImpl 类的实现
class PeepholeOptimizeDictIdiomsImpl {
 public:
  // 构造函数，接受一个图形对象作为参数
  explicit PeepholeOptimizeDictIdiomsImpl(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), aliasDb_(std::make_unique<AliasDb>(graph_)) {}

  // 运行优化算法的入口函数
  bool run() {
    // 收集所有被修改过的字典
    collectMutatedDicts(graph_->block());
    // 对图中的每个块运行优化算法
    return runBlock(graph_->block());
  }

 private:
  // 检查节点中是否有被修改过的字典
  void checkForMutatedDicts(Value* v) {
    if (isDict(v) && aliasDb_->hasWriters(v)) {
      mutated_dicts_.insert(v);
    }
  }

  // 收集所有被修改过的字典
  void collectMutatedDicts(Block* b) {
    // 遍历块的输入值
    for (Value* v : b->inputs()) {
      checkForMutatedDicts(v);
    }
    // 遍历块中的每个节点
    for (Node* n : b->nodes()) {
      // 检查节点的输出值
      for (Value* v : n->outputs()) {
        checkForMutatedDicts(v);
      }
      // 递归收集块中嵌套块的被修改过的字典
      for (Block* block : n->blocks()) {
        collectMutatedDicts(block);
      }
    }
  }

  // 获取字典节点的引用
  const DictNode& getDictNode(Node* creation_node) {
    auto cached = dict_cache_.find(creation_node);
    // 如果缓存中没有该节点的字典节点，则创建并缓存
    if (cached == dict_cache_.end()) {
      cached =
          dict_cache_.emplace(creation_node, DictNode(creation_node)).first;
    }

    return cached->second;
  }

  // 从字典中获取给定键对应的值
  std::optional<Value*> getValueFromDict(Node* dict_creation_node, Value* key) {
    const DictNode& dict_node = getDictNode(dict_creation_node);
    auto key_opt = toIValue(key);
    // 如果键不是常量，则返回空
    if (key_opt == c10::nullopt) {
      return c10::nullopt;
    }
    IValue key_ival = *key_opt;
    // 如果字典节点能够进行优化，则获取对应的值，否则返回空
    if (dict_node.canOptimize()) {
      return dict_node.getOrNullopt(key_ival);
    }
    return c10::nullopt;
  }

  // 计算字典的长度
  std::optional<int64_t> computeLen(Node* dict_creation_node) {
    const DictNode& dict_node = getDictNode(dict_creation_node);
    // 如果字典节点能够进行优化，则返回其大小，否则返回空
    if (dict_node.canOptimize()) {
      return static_cast<int64_t>(dict_node.size());
    }
    return c10::nullopt;
  }

  // 优化长度节点
  bool optimizeLen(Node* len_node, Node* creation_node) {
    if (creation_node->kind() == prim::DictConstruct) {
      auto len = computeLen(creation_node);
      // 如果成功计算长度，则用常量替换长度节点的输出
      if (len != c10::nullopt) {
        WithInsertPoint guard(len_node);
        len_node->output()->replaceAllUsesWith(graph_->insertConstant(len));
        return true;
      }
    }
    return false;
  }

  // 优化获取元素节点
  bool optimizeGetItem(Node* getitem_node, Node* creation_node) {
    if (creation_node->kind() == prim::DictConstruct) {
      auto key = getitem_node->input(1);
      auto value = getValueFromDict(creation_node, key);
      // 如果成功获取值，则替换获取元素节点的输出
      if (value != c10::nullopt) {
        getitem_node->output()->replaceAllUsesWith(*value);
        return true;
      }
    }
    return false;
  }

  // 在块上运行优化算法
  bool runBlock(Block* block) {
    bool changed = false;
    // 遍历给定代码块（block）中的每一个节点（Node* node）
    for (Node* node : block->nodes()) {
      // 对每个节点中的每个子代码块（Block* b）执行以下操作
      for (Block* b : node->blocks()) {
        // 运行并记录代码块的执行结果是否改变（changed）
        changed |= runBlock(b);
      }

      // 仅优化字典操作
      // 如果节点的输入为空，或者输入的第一个节点不是字典类型，则跳过当前节点的优化
      if (node->inputs().empty() || !isDict(node->input(0))) {
        continue;
      }

      // 获取当前节点的第一个输入
      auto first_input = node->input(0);

      // 仅优化未变异的输入操作
      // 如果当前输入在变异字典集合（mutated_dicts_）中，则跳过当前节点的优化
      if (mutated_dicts_.count(first_input)) {
        continue;
      }

      // 根据节点类型进行优化处理
      if (node->kind() == aten::len) {
        // 如果节点是长度操作（aten::len），则优化该操作并更新 changed
        changed |= optimizeLen(node, first_input->node());
      } else if (node->kind() == aten::__getitem__) {
        // 如果节点是获取元素操作（aten::__getitem__），则优化该操作并更新 changed
        changed |= optimizeGetItem(node, first_input->node());
      }
    }
    // 返回最终是否有改变的标志（changed）
    return changed;
  }

  // 图形表示的共享指针
  std::shared_ptr<Graph> graph_;
  // 变异字典的无序集合
  std::unordered_set<Value*> mutated_dicts_;
  // 别名数据库的唯一指针
  std::unique_ptr<AliasDb> aliasDb_;
  // 节点与字典节点映射的无序映射
  std::unordered_map<Node*, DictNode> dict_cache_;
};

} // namespace



// 结束了一个命名空间的定义

bool PeepholeOptimizeDictIdioms(const std::shared_ptr<Graph>& graph) {
  // 创建 PeepholeOptimizeDictIdiomsImpl 的实例，并传入图对象作为参数
  PeepholeOptimizeDictIdiomsImpl opt(graph);
  // 调用实例的 run 方法，执行字典习语的Peephole优化，并返回结果
  return opt.run();
}

} // namespace jit
} // namespace torch
```