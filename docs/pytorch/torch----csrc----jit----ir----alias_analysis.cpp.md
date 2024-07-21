# `.\pytorch\torch\csrc\jit\ir\alias_analysis.cpp`

```py
#include <torch/csrc/jit/ir/alias_analysis.h>  // 包含了别名分析的头文件

#include <ATen/core/interned_strings.h>  // 包含了ATen库中的interned字符串定义
#include <c10/util/flat_hash_map.h>  // 包含了flat_hash_map的定义，用于高效的哈希映射
#include <c10/util/irange.h>  // 包含了irange的定义，用于生成整数范围
#include <torch/csrc/jit/api/function_impl.h>  // 包含了函数实现的API头文件
#include <torch/csrc/jit/jit_log.h>  // 包含了JIT日志的定义
#include <torch/csrc/jit/passes/inliner.h>  // 包含了内联传递的定义
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>  // 包含了子图工具函数的定义
#include <torch/csrc/jit/runtime/operator.h>  // 包含了运算符的运行时定义
#include <fstream>  // 文件流相关的头文件
#include <iostream>  // 输入输出流相关的头文件

namespace torch::jit {

namespace {

c10::MaybeOwned<TypePtr> toSingleType(const AliasTypeSet& mut_types) {
  return mut_types.size() == 1
      ? c10::MaybeOwned<TypePtr>::borrowed(mut_types[0])  // 如果类型集合只有一个元素，则返回borrowed指针
      : c10::MaybeOwned<TypePtr>::owned(c10::UnionType::create(mut_types));  // 否则创建并返回一个包含所有类型的UnionType
}

// This class determines whether a type is mutable, and, if so, it maps
// the type to its "mutable equivalent" (see definition in
// `mapTypeToAliasTypeSet`). It uses a cache of TypePtrs to speed up these
// type lookups
class MutableTypePtrHelper {
 public:
  explicit MutableTypePtrHelper(
      ska::flat_hash_map<TypePtr, AliasTypeSet>* mutable_type_cache)
      : mutable_type_cache_(mutable_type_cache) {}  // 构造函数，初始化可变类型缓存

  // Map any mutable type to a type such that all other types which the
  // mutable type can alias will be mapped to the same type. For
  // example, calling this method on `Optional[List[int]]` should be
  // the same as calling this method on `List[int]`.
  //
  // Rules:
  //   - If the type is not mutable, return `nullopt`
  //   - If the type is a `Tuple`, that means that it's an immutable
  //     object that can itself contain mutable objects. We want to make
  //     sure that the mutable objects are correctly aliased, so we
  //     remove the immutable objects. (For example,
  //     `Tuple[int, Tensor]` would become `Tuple[Tensor]`, while
  //     `Tuple[int, str]` would be returned as `nullopt`.) This is a
  //     convenience that makes it easy to check if the `Tuple`
  //     contains only immutable objects, though it's not technically
  //     necessary
  //   - For any Tensor type (including Tensor types that are part of
  //     a larger container, e.g. `List[Tensor]`), return the
  //     "unshaped" version of that Tensor. An "unshaped" Tensor is a
  //     Tensor with shape information removed. For example, a Tensor
  //     of dimension 4 would map to the same type as a Tensor of
  //     dimension 1. This allows us to treat all subclasses of Tensor
  //     as a single, homogenous "Tensor" type.
  std::optional<AliasTypeSet> mapTypeToAliasTypeSet(const TypePtr& type) {
    if (mutable_type_cache_) {  // 如果有可变类型缓存
      const AliasTypeSet* result = mapTypeToBorrowedAliasTypeSet(type);  // 调用借用类型映射函数
      if (result) {
        return *result;  // 返回映射结果
      }
    }
    return mapTypeToAliasTypeSetImpl(type);  // 否则调用实际的类型映射函数
  }

  const AliasTypeSet* mapTypeToBorrowedAliasTypeSet(const TypePtr& type) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mutable_type_cache_ != nullptr);  // 内部断言，确保可变类型缓存不为空
    auto maybe_type_mapping = mutable_type_cache_->find(type);  // 在可变类型缓存中查找给定类型的映射
    // 检查 maybe_type_mapping 是否存在于 mutable_type_cache_ 中
    if (maybe_type_mapping != mutable_type_cache_->end()) {
      // 如果存在，则返回其对应的值的指针
      return &maybe_type_mapping->second;
    }

    // 调用 mapTypeToAliasTypeSetImpl 函数，获取类型 type 对应的 AliasTypeSet
    auto mutable_types = mapTypeToAliasTypeSetImpl(type);
    if (mutable_types) {
      // 将获取到的 AliasTypeSet 插入 mutable_type_cache_，并返回插入后的值的指针
      auto it =
          mutable_type_cache_->emplace(type, std::move(*mutable_types)).first;
      return &it->second;
    } else {
      // 如果无法获取 AliasTypeSet，则返回空指针
      return nullptr;
    }
  }

 private:
  // 根据类型 type 映射到对应的 AliasTypeSet，如果无法映射则返回空值
  std::optional<AliasTypeSet> mapTypeToAliasTypeSetImpl(const TypePtr& type) {
    switch (type->kind()) {
      case TypeKind::ListType:
      case TypeKind::DictType:
      case TypeKind::ClassType:
      case TypeKind::TensorType:
        // TODO: 查找缓存的包含类型。这里比较棘手，
        // 因为 `List[Optional[T]]` 应该仍然是 `List[Optional[Unshaped(T)]]`，
        // 但是 `mapTypeToAliasTypeSet(Optional[T])` 应该返回 `T`
        return AliasTypeSet{unshapedType(type)};
      case TypeKind::UnionType: {
        AliasTypeSet mutable_types;
        // 遍历联合类型的每个内部类型
        for (const TypePtr& inner :
             type->expectRef<UnionType>().containedTypes()) {
          // 调用自身递归获取每个内部类型的 AliasTypeSet
          if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
            // 将内部类型的 AliasTypeSet 合并到 mutable_types 中
            mutable_types.insert(
                mutable_types.end(),
                (*maybe_inner_types).begin(),
                (*maybe_inner_types).end());
          }
        }
        // 如果 mutable_types 为空，则返回空值
        if (mutable_types.empty()) {
          return c10::nullopt;
        }
        return mutable_types;
      }
      case TypeKind::OptionalType: {
        // 获取 Optional 类型的元素类型
        auto inner = type->castRaw<OptionalType>()->getElementType();
        // 递归调用 mapTypeToAliasTypeSet 获取元素类型的 AliasTypeSet
        return mapTypeToAliasTypeSet(inner);
      }
      case TypeKind::AnyType:
        // 对于 AnyType，直接返回包含该类型的 AliasTypeSet
        return {AliasTypeSet{type}};
      case TypeKind::FutureType: {
        // 获取 Future 类型的元素类型
        if (auto maybe_mut_types = mapTypeToAliasTypeSet(
                type->castRaw<FutureType>()->getElementType())) {
          // 如果可以获取到元素类型的 AliasTypeSet，则创建 FutureType 的 AliasTypeSet 返回
          return {AliasTypeSet{
              FutureType::create(*toSingleType(*maybe_mut_types))}};
        }
        // 否则返回空值
        return c10::nullopt;
      }
      case TypeKind::AwaitType: {
        // 获取 Await 类型的元素类型
        if (auto maybe_mut_types = mapTypeToAliasTypeSet(
                type->castRaw<AwaitType>()->getElementType())) {
          // 如果可以获取到元素类型的 AliasTypeSet，则创建 AwaitType 的 AliasTypeSet 返回
          return {
              AliasTypeSet{AwaitType::create(*toSingleType(*maybe_mut_types))}};
        }
        // 否则返回空值
        return c10::nullopt;
      }
      case TypeKind::TupleType: {
        std::vector<TypePtr> mutable_types;
        // 遍历元组类型的每个元素类型
        for (const TypePtr& inner : type->expectRef<TupleType>().elements()) {
          // 调用自身递归获取每个元素类型的 AliasTypeSet
          if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
            // 将元素类型的 AliasTypeSet 合并到 mutable_types 中
            mutable_types.insert(
                mutable_types.end(),
                (*maybe_inner_types).begin(),
                (*maybe_inner_types).end());
          }
        }
        // 如果 mutable_types 为空，则返回空值
        if (mutable_types.empty()) {
          return c10::nullopt;
        }
        return {AliasTypeSet{TupleType::create(mutable_types)}};
      }
      default:
        // 对于不支持的类型，返回空值
        return c10::nullopt;
    }
  }
  // 指向 ska::flat_hash_map 的指针，用于缓存类型映射到 AliasTypeSet 的结果
  ska::flat_hash_map<TypePtr, AliasTypeSet>* mutable_type_cache_;
};

// 检查类型是否为可变类型的实现函数
bool isMutableTypeImpl(
    const TypePtr& type,  // 输入参数：要检查的类型指针
    ska::flat_hash_map<TypePtr, AliasTypeSet>* mutable_type_cache) {
  // 检查常见情况，避免在 `mapTypeToAliasTypeSetPtrImpl` 中递归构造类型
  auto kind = type->kind();
  if (kind == TypeKind::TensorType || kind == TypeKind::ListType ||
      kind == TypeKind::ClassType || kind == TypeKind::DictType) {
    return true;  // 如果类型是张量、列表、类或字典类型，则判定为可变类型
  }
  MutableTypePtrHelper helper(mutable_type_cache);  // 创建辅助类实例
  if (mutable_type_cache) {
    return helper.mapTypeToBorrowedAliasTypeSet(type) != nullptr;  // 使用缓存的类型到可变类型集合的映射检查类型是否可变
  } else {
    return helper.mapTypeToAliasTypeSet(type).has_value();  // 检查类型是否在非缓存模式下为可变类型
  }
}

} // namespace

// 静态函数 `isMutableType` 不使用类型到可变类型等效的缓存
bool AliasDb::isMutableType(const TypePtr& type) {
  return isMutableTypeImpl(type, nullptr);  // 调用实现函数检查类型是否可变，不使用缓存
}

// 检查给定值是否为可变类型
bool AliasDb::isMutableType(const Value* v) {
  return isMutableType(v->type());  // 调用类型版本的 `isMutableType` 检查值的类型是否可变
}

// 使用类型到可变类型缓存
bool AliasDb::isMutableTypeInternal(const TypePtr& type) const {
  return isMutableTypeImpl(type, &mapped_mutable_types_);  // 调用实现函数，使用类型到可变类型缓存检查类型是否可变
}

// 检查给定值是否为可变类型，使用内部的类型版本
bool AliasDb::isMutableTypeInternal(const Value* v) const {
  return isMutableTypeInternal(v->type());  // 调用类型版本的 `isMutableTypeInternal` 检查值的类型是否可变
}

// 将类型映射到可变类型集合指针
const AliasTypeSet* AliasDb::mapTypeToAliasTypeSetPtr(
    const TypePtr& type) const {
  MutableTypePtrHelper helper(&mapped_mutable_types_);  // 创建辅助类实例
  return helper.mapTypeToBorrowedAliasTypeSet(type);  // 返回类型到可变类型集合的映射指针
}

AliasDb::~AliasDb() = default;  // 析构函数，默认实现

// 在分析期间用于跟踪所有写操作的结构体。分析完成后，将用于构建更高效的写入索引
struct AliasDb::WriteRegistry {
  // 注册写操作，关联值和节点
  void registerWrite(const Value* v, Node* n) {
    writes_[n].emplace_back(v);
  }
  // 注册写入所有包含的值，关联值和节点
  void registerWriteToAllContained(const Value* v, Node* n) {
    containedWrites_[n].emplace_back(v);
  }
  // 注册写入所有通配符
  void registerWriteToAllWildcards(Node* n) {
    writesToAllWildcards_.insert(n);
  }
  // 写操作映射：节点到值的映射
  std::unordered_map<Node*, std::vector<const Value*>> writes_;
  // 包含的写操作映射：节点到值的映射
  std::unordered_map<Node*, std::vector<const Value*>> containedWrites_;
  // 所有通配符的写操作集合
  std::unordered_set<Node*> writesToAllWildcards_;
};

// AliasDb 构造函数，初始化时传入图、是否冻结、是否下降函数调用
AliasDb::AliasDb(
    std::shared_ptr<Graph> graph,
    bool isFrozen,
    bool descendFunctionCalls)
    // 使用成员初始化列表初始化成员变量，通过 std::move 移动构造 graph_
    // isFrozen_ 表示对象是否被冻结
    // descend_function_calls_ 控制是否下降函数调用
    // memoryDAGBuilder_ 使用 std::make_unique 创建 MemoryDAGBuilder 对象的唯一指针
    // writeRegistry_ 使用 std::make_unique 创建 AliasDb::WriteRegistry 对象的唯一指针
    AliasDb::AliasDb(
        std::shared_ptr<Graph> graph_,
        bool isFrozen_,
        bool descendFunctionCalls_,
        std::unique_ptr<MemoryDAGBuilder> memoryDAGBuilder_,
        std::unique_ptr<AliasDb::WriteRegistry> writeRegistry_)
        : graph_(std::move(graph_)),
          isFrozen_(isFrozen_),
          descend_function_calls_(descendFunctionCalls_),
          memoryDAGBuilder_(std::move(memoryDAGBuilder_)),
          writeRegistry_(std::make_unique<AliasDb::WriteRegistry>()) {
      
      // 对图进行分析
      analyze(graph_);
    
      // 移动构造并创建内存 DAG
      memoryDAG_ = std::move(*memoryDAGBuilder_).createMemoryDAG();
      memoryDAGBuilder_ = nullptr; // 使进一步访问成为硬错误
    
      // 设置通配符对应的内存位置
      memoryDAG_->setWildcards(
          wildcards_, elementMap_, [&](const Value* v) -> Element* {
            return getWildcard(v->type());
          });
    
      // 基于写入注册表中的信息，建立各种写入索引
    
      // 初始化写入索引
      writeIndex_ = TWriteIndex();
      auto& writeIndex = *writeIndex_; // 使 operator[] 的使用更加清晰
    
      // 构建写入索引
      for (const auto& write : writeRegistry_->writes_) {
        Node* node = write.first;
        const std::vector<const Value*> writtenValues = write.second;
        for (const Value* writtenValue : writtenValues) {
          auto it = elementMap_.find(writtenValue);
          TORCH_INTERNAL_ASSERT(
              it != elementMap_.end(), "Tried to write to value not in MemoryDAG");
          const auto& writtenMemoryLocations =
              memoryDAG_->getMemoryLocations(it->second);
          writeIndex[node] |= writtenMemoryLocations;
        }
      }
    
      // 处理写入注册表中的包含写入
      for (const auto& write : writeRegistry_->containedWrites_) {
        Node* node = write.first;
        const std::vector<const Value*>& writtenValues = write.second;
        for (const Value* writtenValue : writtenValues) {
          auto elem = elementMap_.at(writtenValue);
          MemoryLocations writtenMemoryLocations;
          memoryDAG_->collectAllContainedMemoryLocations(
              elem, writtenMemoryLocations);
          writeIndex[node] |= writtenMemoryLocations;
        }
      }
    
      // 处理写入注册表中对所有通配符的写入
      for (const auto& write : writeRegistry_->writesToAllWildcards_) {
        for (const auto& pr : wildcardIndex_) {
          writeIndex[write].set(pr.second->index);
        }
      }
    
      // 现在构建了写入索引，将 WriteRegistry 置空，以防止未来访问
      // 这样可以防止索引与实际不一致（因为无法注册新的写入）
      writeRegistry_ = nullptr;
    
      // 初始化写入缓存
      buildWrittenToLocationsIndex();
      GRAPH_DEBUG(toString());
    }
}

// 检查节点 `n` 是否可变，即其输入节点是否有写操作的别名
bool AliasDb::isMutable(Node* n) const {
  // 创建一个空的值集合 `vs`
  ValueSet vs;
  // 遍历节点 `n` 的每一个输入节点，并将其添加到值集合 `vs` 中
  for (const auto input : n->inputs()) {
    vs.insert(input);
  }
  // 调用 `writesToAlias` 函数，判断节点 `n` 是否写入了 `vs` 中任意值的别名
  return writesToAlias(n, vs);
}

// 检查节点 `n` 的输入节点中是否有写操作的节点
bool AliasDb::hasInputWriters(const Node* n) const {
  // 遍历节点 `n` 的每一个输入节点
  for (const auto input : n->inputs()) {
    // 如果输入节点 `input` 有写操作，则返回 true
    if (hasWriters(input)) {
      return true;
    }
  }
  // 如果所有输入节点都没有写操作，则返回 false
  return false;
}

// 检查节点 `n` 的输出节点中是否有写操作的节点
bool AliasDb::hasOutputWriters(const Node* n) const {
  // 遍历节点 `n` 的每一个输出节点
  for (const auto output : n->outputs()) {
    // 如果输出节点 `output` 有写操作，则返回 true
    if (hasWriters(output)) {
      return true;
    }
  }
  // 如果所有输出节点都没有写操作，则返回 false
  return false;
}

// 检查节点 `n` 是否有写操作的节点（输入或输出节点）
bool AliasDb::hasWriters(const Node* n) const {
  // 节点 `n` 的输入节点或输出节点中有写操作的节点，则返回 true
  return hasInputWriters(n) || hasOutputWriters(n);
}

// 检查值 `v` 是否有写操作的节点
bool AliasDb::hasWriters(const Value* v) const {
  // 如果值 `v` 必须为空，则返回 false
  if (v->mustBeNone()) {
    return false;
  }

  // 在 `elementMap_` 中查找值 `v`
  auto it = elementMap_.find(v);
  // 如果找不到，则返回 false
  if (it == elementMap_.end()) {
    return false;
  }

  // 获取值 `v` 对应的元素 `el`
  const auto& el = it->second;
  // 检查 `el` 的内存位置是否与写入位置的索引相交
  return writtenToLocationsIndex_->intersects(
      memoryDAG_->getMemoryLocations(el));
}

// 递归获取节点 `n` 的写入位置集合
void AliasDb::getWritesImpl(Node* n, MemoryLocations& ret) const {
  // 如果节点 `n` 在写入索引中存在
  if (writeIndex_->count(n)) {
    // 获取节点 `n` 的写入位置集合并合并到 `ret` 中
    const auto& writes = writeIndex_->at(n);
    ret |= writes;
  }

  // 递归遍历节点 `n` 的每一个块和每一个节点，并获取写入位置集合
  for (auto block : n->blocks()) {
    for (auto node : block->nodes()) {
      getWritesImpl(node, ret);
    }
  }
}

// 判断节点 `n` 是否写入了 `vs` 中任意值的别名
bool AliasDb::writesToAlias(Node* n, const ValueSet& vs) const {
  // 获取节点 `n` 的写入位置集合
  const auto writtenTo = getWrites(n);
  // 如果写入位置集合为空，则返回 false
  if (writtenTo.empty()) {
    return false;
  }

  // 遍历值集合 `vs`
  MemoryLocations locs;
  for (const auto v : vs) {
    // 在 `elementMap_` 中查找值 `v`
    auto it = elementMap_.find(v);
    // 如果找到了，则获取其内存位置集合 `vlocs`
    if (it != elementMap_.end()) {
      const auto& vlocs = memoryDAG_->getMemoryLocations(it->second);
      // 如果节点 `n` 的写入位置集合与 `vlocs` 相交，则返回 true
      if (writtenTo.intersects(vlocs)) {
        return true;
      }
    }
  }

  // 如果没有找到任何相交的内存位置，则返回 false
  return false;
}

// 获取节点 `n` 的写入位置集合
MemoryLocations AliasDb::getWrites(Node* n) const {
  MemoryLocations writes;
  // 调用 `getWritesImpl` 函数，获取节点 `n` 的写入位置集合
  getWritesImpl(n, writes);
  return writes;
}

// 递归获取节点 `n` 的读取位置集合
void AliasDb::getReadsImpl(Node* n, MemoryLocations& ret) const {
  // 遍历节点 `n` 的每一个输入节点
  for (const auto input : n->inputs()) {
    // 在 `elementMap_` 中查找输入节点 `input`
    auto it = elementMap_.find(input);
    // 如果找到了，则获取其对应的元素 `el`
    if (it != elementMap_.end()) {
      auto el = it->second;
      // 收集元素 `el` 可能的所有包含的内存位置和它们的子元素
      memoryDAG_->collectAllContainedMemoryLocations(el, ret);
    }
  }

  // 递归遍历节点 `n` 的每一个块和每一个节点，并获取读取位置集合
  for (auto block : n->blocks()) {
    for (auto node : block->nodes()) {
      getReadsImpl(node, ret);
    }
  }
}

// 获取节点 `n` 的读取位置集合
MemoryLocations AliasDb::getReads(Node* n) const {
  MemoryLocations reads;
  // 调用 `getReadsImpl` 函数，获取节点 `n` 的读取位置集合
  getReadsImpl(n, reads);
  return reads;
}

// 获取元素 `e` 的名称
std::string AliasDb::getElementName(const Element* e) const {
  // 如果元素 `e` 的值集合为空
  if (e->values.empty()) {
    // 遍历 `wildcardIndex_`，查找与元素 `e` 匹配的通配符
    for (const auto& ent : wildcardIndex_) {
      if (ent.second == e) {
        return std::string("WILDCARD for type ") + ent.first->str();
      }
    }
    // 如果没有找到匹配的通配符，则返回默认的 "WILDCARD"
    return "WILDCARD";
  } else {
    std::ostringstream ss;
    # 检查表达式节点 e 中的值列表是否只包含一个元素
    if (e->values.size() == 1) {
      # 如果是单一元素，将其调试名称添加到输出流 ss 中，并返回字符串形式的结果
      ss << "%" << (*e->values.begin())->debugName();
      return ss.str();
    }
    # 如果值列表中有多个元素，则输出开头的括号
    ss << "(";
    # 遍历表达式节点 e 中的值列表，将每个值的调试名称添加到输出流 ss 中，并添加逗号分隔符
    for (const Value* v : e->values) {
      ss << "%" << v->debugName() << ", ";
    }
    # 添加结束的括号到输出流 ss 中
    ss << ")";
    # 返回拼接好的字符串形式结果
    return ss.str();
  }
}

void AliasDb::dump() const {
  // 打印当前别名数据库的字符串表示形式
  std::cout << toString();
}

std::string AliasDb::toString() const {
  // 创建一个字符串流对象
  std::stringstream ss{};

  // 添加图形部分的字符串表示
  ss << "\n===1. GRAPH===\n";
  ss << graph_->toString();  // 获取图形对象的字符串表示并添加到流中

  // 添加别名数据库部分的字符串表示
  ss << "\n===2. ALIAS DB===\n";
  // 遍历元素映射表
  for (const auto& ptrPair : elementMap_) {
    const auto element = ptrPair.second;
    int ct = 0;
    // 如果元素有指向其他元素的指针
    if (!element->pointsTo.empty()) {
      ss << getElementName(element) << " points to: ";
      // 遍历该元素指向的每个其他元素
      for (const auto pointedTo : element->pointsTo) {
        if (ct > 0) {
          ss << ", ";
        }
        ++ct;
        // 添加指向的元素的名称到字符串流中
        ss << getElementName(memoryDAG_->fromIndex(pointedTo));
      }
      ss << "\n";
    }
    ct = 0;
    // 如果元素包含其他元素
    if (!element->containedElements.empty()) {
      ss << getElementName(element) << " contains: ";
      // 遍历该元素包含的每个其他元素
      for (const auto contained : element->containedElements) {
        ss << getElementName(memoryDAG_->fromIndex(contained));
        if (ct > 0) {
          ss << ", ";
        }
        ++ct;
      }
      ss << "\n";
    }
  }

  // 添加写操作部分的字符串表示
  ss << "\n===3. Writes===\n";
  // 遍历写入索引表中的每个条目
  for (const auto& pr : *writeIndex_) {
    const auto node = pr.first;
    const auto& values = pr.second;
    // 将节点名称和相关值添加到字符串流中
    ss << *node;
    ss << "  ";
    for (const auto value : values) {
      ss << getElementName(memoryDAG_->fromIndex(value)) << ", ";
    }
    ss << "\n";
  }
  ss << "\n";
  // 返回字符串流中的所有内容作为字符串
  return ss.str();
}

bool AliasDb::dumpToGraphvizFile(const char* filename) const {
  // 打开指定文件用于写入
  std::ofstream dot_file(filename);
  // 检查文件是否成功打开
  if (!dot_file.good()) {
    std::cout << "Failed to create Graphviz file: '" << filename << "'\n";
    return false;
  }
  // 将别名数据库的图形表示写入到文件中
  dot_file << toGraphviz();
  return true;
}

std::string AliasDb::toGraphviz() const {
  // 创建一个字符串流对象用于生成Graphviz格式的输出
  std::stringstream dot;

  // 添加别名数据库的字符串表示作为注释
  dot << "/*\n";
  dot << toString();
  dot << "*/\n";

  // 开始生成Graphviz文件的主体部分
  dot << "digraph alias_db {\n"
      << "  rankdir=LR\n"
      << "  node [shape=rect, color=gray];\n"
      << "  edge [color=black];\n";

  // 遍历元素映射表，为每个元素及其指向的元素生成对应的Graphviz节点和边
  for (const auto& ptrPair : elementMap_) {
    const auto element = ptrPair.second;
    if (!element->pointsTo.empty()) {
      for (const auto pointedTo : element->pointsTo) {
        // 添加Graphviz格式的边
        dot << "  " << name(element) << " -> "
            << name(memoryDAG_->fromIndex(pointedTo)) << "\n";
      }
    }

        }
    }

    // 结束Graphviz文件内容
    dot << "}\n";

    // 返回生成的Graphviz格式字符串
    return dot.str();
}
    # 检查元素是否包含子元素，如果有则进入循环处理
    if (!element->containedElements.empty()) {
      # 遍历元素的每一个子元素
      for (const auto contained : element->containedElements) {
        # 构建 DOT 格式的图形描述，描述当前元素指向子元素的关系，使用虚线和蓝色表示
        dot << "  " << name(element) << " -> "
            << name(memoryDAG_->fromIndex(contained))
            << " [style=dashed, color=blue]\n";
      }
    }
  }
  
  # 添加 DOT 描述的结束标记
  dot << "}\n";
  # 返回生成的 DOT 描述的字符串
  return dot.str();
}

// 分析给定图中的每个输入节点，设置通配符信息
void AliasDb::analyze(const std::shared_ptr<Graph>& graph) {
  for (auto input : graph->inputs()) {
    setWildcard(input);  // 设置输入节点的通配符信息
  }
  analyze(graph->block());  // 分析整个图的主要块
}

// 分析给定的块中的每个节点
void AliasDb::analyze(Block* block) {
  for (auto node : block->nodes()) {
    analyze(node);  // 调用分析节点的具体方法
  }
}

// 分析单个节点
void AliasDb::analyze(Node* node) {
  analyzeImpl(node);  // 调用节点分析的实现方法
}

// 尝试使用注册的分析器分析节点，如果成功返回true
bool AliasDb::tryRegisteredAnalysis(Node* node) {
  const Operator& op = node->getOperator();  // 获取节点的运算符
  auto analysis = op.aliasAnalysisKind();  // 获取运算符的别名分析类型
  if (AliasAnalysisKind::PURE_FUNCTION == analysis) {
    analyzeCreator(node);  // 分析节点的创建者信息
    return true;
  }
  return false;
}

// 分析节点的具体实现
void AliasDb::analyzeImpl(Node* node) {
  auto op = node->maybeOperator();  // 获取节点的操作符
  const bool hasSpecialCase = aliasAnalysisHasSpecialCaseFor(node->kind());  // 检查节点是否有特殊情况的别名分析
  if (op) {
    const auto analysis = op->aliasAnalysisKind();  // 获取操作符的别名分析类型

    const bool registeredAsSpecialCase =
        analysis == AliasAnalysisKind::INTERNAL_SPECIAL_CASE;  // 检查是否注册为特殊情况

    // 如果注册为特殊情况但实际不包含特殊情况，则报错
    if (C10_UNLIKELY(registeredAsSpecialCase && !hasSpecialCase)) {
      TORCH_INTERNAL_ASSERT(
          false,
          "Op ",
          node->kind().toDisplayString(),
          " is registered with AliasAnalysisKind::INTERNAL_SPECIAL_CASE but doesn't have a special case.");
    } else if (C10_UNLIKELY(!registeredAsSpecialCase && hasSpecialCase)) {
      // 如果没有注册为特殊情况但实际包含特殊情况，则报错
      TORCH_INTERNAL_ASSERT(
          false,
          "Op ",
          node->kind().toDisplayString(),
          " has a special case and should be registered with AliasAnalysisKind::INTERNAL_SPECIAL_CASE but is registered with ",
          c10::toString(analysis));
    }
  } else {
    // 如果没有找到操作符并且没有特殊情况，报错
    if (!hasSpecialCase) {
      std::ostringstream oss;
      for (const auto input : node->inputs()) {
        oss << input->type()->str() << ", ";
      }
      oss << "\n\nCandidates:";
      const auto& candidates = getAllOperatorsFor(node->kind());
      for (const auto& candidate : candidates) {
        oss << "\n\t" << candidate->schema();
      }
      TORCH_INTERNAL_ASSERT(
          0,
          "We don't have an op for ",
          node->kind().toDisplayString(),
          " but it isn't a special case.  ",
          "Argument types: ",
          oss.str());
    }
  }

  // 对于未被规范化的节点，需要特殊处理
  switch (node->kind()) {
    case prim::If:
      return analyzeIf(node);  // 分析条件节点
    case prim::Loop:
      return analyzeLoop(node);  // 分析循环节点
    case prim::FusionGroup:
    case prim::CudaFusionGroup:
    case prim::oneDNNFusionGroup:
    case prim::FunctionalGraph:
    case prim::DifferentiableGraph:
    case prim::FallbackGraph:
      return analyzeSubgraph(node);  // 分析子图节点
    case prim::fork:
      return analyzeFork(node);  // 分析分支节点
    // 处理 aten::wait 操作，分析等待节点
    case aten::wait:
      return analyzeWait(node);
    
    // 处理 prim::awaitable 和 prim::awaitable_nowait 操作，分析可等待节点
    case prim::awaitable:
    case prim::awaitable_nowait:
      return analyzeAwaitable(node);
    
    // 处理 prim::awaitable_wait 操作，分析等待可等待节点
    case prim::awaitable_wait:
      return analyzeAwaitableWait(node);
    
    // 处理 prim::rpc_async, prim::rpc_sync, prim::rpc_remote 操作，分析 RPC 异步调用节点
    case prim::rpc_async:
    case prim::rpc_sync:
    case prim::rpc_remote:
      return analyzeRpcAsync(node);
    
    // 处理 aten::batch_norm 操作，分析批标准化节点
    case aten::batch_norm:
      return analyzeBatchNorm(node);
    
    // 处理 aten::instance_norm 操作，分析实例标准化节点
    case aten::instance_norm:
      return analyzeInstanceNorm(node);
    
    // 处理 prim::GradOf 操作，分析梯度节点
    case prim::GradOf:
      return analyzeGradOf(node);
    
    // 处理 prim::BroadcastMKLDNNTensors 操作，设置指向输入的指针
    case prim::BroadcastMKLDNNTensors: {
      makePointerTo(node->outputs().at(0), node->inputs().at(0));
      makePointerTo(node->outputs().at(1), node->inputs().at(1));
      return;
    }
    
    // 处理一系列用于创建对象和数据结构的操作，统一分析器处理
    // TODO: 对于 TensorExpr 别名正确性需要进一步考虑
    case prim::TensorExprGroup:
    case prim::TensorExprDynamicGroup:
    case prim::MKLDNNGroup:
    case prim::ConstantMKLDNNTensor:
    case prim::StaticSubgraph:
    case prim::Constant:
    case prim::AutogradZero:
    case prim::AutogradAdd:
    case prim::FusedConcat:
    case prim::MMTreeReduce:
    case prim::MMBatchSide:
    case prim::BroadcastSizes:
    case prim::ChunkSizes:
    // 这些操作在初始编译过程中应该不会出现，但因为某些依赖关系涉及闭包调用别名数据库，需要在此处理
    case prim::EmptyListLiteral:
    case prim::Closure:
    case prim::CreateObject:
    case prim::tolist:
    case prim::Uninitialized:
      return analyzeCreator(node);
    
    // 处理 prim::TupleConstruct, prim::DictConstruct, prim::ListConstruct 操作，分析容器构造节点
    case prim::TupleConstruct:
    case prim::DictConstruct:
    case prim::ListConstruct:
      return analyzeContainerConstruct(node);
    
    // 处理 prim::TupleUnpack, prim::TupleIndex, prim::TupleSlice, prim::ListUnpack, prim::PythonOp, prim::GetAttr 操作
    // 如果处于冻结状态并且是 prim::GetAttr 操作，检查其类型是否为模块，如果是则使用创建器分析器，否则使用提取器分析器
    case prim::TupleUnpack:
    case prim::TupleIndex:
    case prim::TupleSlice:
    case prim::ListUnpack:
    case prim::PythonOp:
    case prim::GetAttr:
      if (isFrozen_ && node->kind() == prim::GetAttr) {
        auto& ty = node->input()->type();
        if (ty->expectRef<ClassType>().is_module()) {
          return analyzeCreator(node);
        }
      }
      return analyzeExtractor(node);
    
    // 处理 prim::unchecked_cast 操作，创建指向输入的指针
    case prim::unchecked_cast:
      return makePointerTo(node->output(), node->input());
    
    // 处理 prim::ConstantChunk 操作，分析常量块节点
    case prim::ConstantChunk:
      return analyzeChunk(node);
    
    // 处理 prim::BroadcastingChunk 操作，分析广播块节点
    case prim::BroadcastingChunk:
      return analyzeBroadcastingChunk(node);
    
    // 处理 prim::SetAttr 操作，分析设置属性节点
    case prim::SetAttr:
      return analyzeSetAttr(node);
    
    // 处理 prim::profile_ivalue, prim::profile 操作，设置指向输入的指针
    case prim::profile_ivalue:
    case prim::profile:
      makePointerTo(node->output(), node->inputs().at(0));
      return;
    
    // 处理 prim::TypeCheck, prim::RequiresGradCheck 操作，设置指向输入的指针
    case prim::TypeCheck:
    case prim::RequiresGradCheck: {
      auto num_inputs = node->inputs().size();
      for (const auto i : c10::irange(num_inputs)) {
        makePointerTo(node->outputs().at(i), node->inputs().at(i));
      }
      return;
    }
    
    // 处理 prim::BailOut 操作，确保输入的第一个节点是 prim::BailoutTemplate，创建指向输出的指针
    case prim::BailOut:
      TORCH_INTERNAL_ASSERT(
          node->inputs().at(0)->node()->kind() == prim::BailoutTemplate);
      makePointerTo(node->output(), node->inputs().at(1));
      return;
    case prim::Guard:
      // 如果节点是 Guard 类型，将其输出指针指向输入节点的指针
      makePointerTo(node->output(), node->inputs().at(0));
      return;
    case prim::CallFunction:
    case prim::CallMethod: {
      // 如果节点是 CallFunction 或者 CallMethod 类型
      // TODO: 可以通过函数的总结来改进这部分，目前假设最坏情况
      // 如果不允许下降函数调用，则采用保守分析
      if (!descend_function_calls_) {
        return analyzeConservative(node);
      }
      // 尝试将节点转换为函数图
      auto g = tryToGraphFunction(node);
      if (!g) {
        return analyzeConservative(node);
      }
      // 这是一个未优化的路径 - 我们为每个函数调用复制子图，
      // 所以通常不启用递归分析。很少见到不内联的情况下需要精细的别名分析用例
      auto graph = g->optimized_graph();
      // 别名分析将使用 Value* 作为信息的映射，
      // 因此对于每个特定函数调用的分析，我们需要一个新的图形
      // 对于所有复制的图形，存储它们以避免在分析期间出现生命周期问题
      std::vector<std::shared_ptr<Graph>>& graphs =
          function_call_copies_[graph.get()];
      if (graphs.empty()) {
        // 如果图形集合为空，将优化后的图形添加到集合中，并分析子图
        graphs.push_back(graph);
        analyzeSubgraph(node, graph);
      } else {
        // 否则，复制图形，并将复制的图形添加到集合中，然后分析子图
        auto copied_graph = graph->copy();
        graphs.push_back(copied_graph);
        analyzeSubgraph(node, copied_graph);
      }
      return;
    }
    case prim::Enter:
    case prim::Exit:
      // 如果节点是 Enter 或者 Exit 类型
      // TODO: 可以通过函数的总结来改进这部分，目前假设最坏情况
      // 注意：如果更改了安全更改别名关系，则更新 safeToChangeAliasingRelationship
      return analyzeConservative(node);
    case prim::Print:
    case prim::isinstance:
      // 如果节点是 Print 或者 isinstance，这些操作不做任何事情
      return;
    default:
      // 对于其他所有操作
      // 如果尝试注册分析成功，则返回
      if (tryRegisteredAnalysis(node)) {
        return;
      }
  }

  // 如果代码执行到这里，表示应该有一个操作模式（op schema）
  TORCH_INTERNAL_ASSERT(op, "We should have an op schema if we get to here");
  // 获取操作的别名分析类型
  const AliasAnalysisKind analysis = op->aliasAnalysisKind();
  // 断言操作的别名分析类型不是 INTERNAL_SPECIAL_CASE，
  // 并且节点的类型没有特殊情况
  TORCH_INTERNAL_ASSERT(
      analysis != AliasAnalysisKind::INTERNAL_SPECIAL_CASE &&
          !aliasAnalysisHasSpecialCaseFor(node->kind()),
      "Special cases should be handled already if we're here.");

  // 如果节点的类型是 aten::、prim:: 或者 cuda::，则执行以下操作
  // TODO: 系统中没有任何依赖于 aten:: 和 prim:: 操作使用
  // AliasAnalysisKind::FROM_SCHEMA 或 AliasAnalysisKind::INTERNAL_SPECIAL_CASE，
  // 但这是所有当前操作的预期行为和一个良好的错误检查。
  // 如果以后有用例需要，我们可以考虑放宽这个约束。
    // 断言，确保分析类型为 AliasAnalysisKind::FROM_SCHEMA 或 AliasAnalysisKind::CONSERVATIVE
    TORCH_INTERNAL_ASSERT(
        analysis == AliasAnalysisKind::FROM_SCHEMA ||
            analysis == AliasAnalysisKind::CONSERVATIVE,
        "aten:: 和 prim:: 操作符应使用 AliasAnalysisKind::FROM_SCHEMA 或 "
        "AliasAnalysisKind::CONSERVATIVE（如果确实需要），但 ",
        node->kind().toDisplayString(),
        " 并未如此。注意：理想情况下，prim:: 操作符实际上不应具有模式(schema)，"
        "而应该使用 AliasAnalysisKind::INTERNAL_SPECIAL_CASE。"
    )
    
    // 如果分析类型为 AliasAnalysisKind::CONSERVATIVE
    if (analysis == AliasAnalysisKind::CONSERVATIVE) {
        // TODO: 先前的别名分析实现总是访问 node->schema ，导致 Node 类中的模式缓存填满整个图。
        // 不幸的是，我们的 JIT 传递开始依赖于这一点。详情请见 caffe2/torch/onnx/utils.py，
        // 一个无效的 JIT 图因为我们之前调用了 _jit_pass_erase_number_types 而调用了
        // _jit_pass_onnx，因此整数现在是张量而不是数字类型。因此，如果 _jit_pass_onnx 尝试查找
        // 操作符模式，它将崩溃。然而，在此之前调用的 _jit_pass_constant_propagation 运行别名分析，
        // 并预先填充了所有 Node 实例中的模式缓存，以便 _jit_pass_onnx 不再查找操作符以获取模式。
        // 我们应该修复这个问题。
        node->schema(); // 填充 Node 类中的模式缓存
    
        return analyzeConservative(node);
    }
    
    // 如果分析类型为 AliasAnalysisKind::FROM_SCHEMA
    TORCH_INTERNAL_ASSERT(
        analysis == AliasAnalysisKind::FROM_SCHEMA,
        "AliasAnalysisKind::CONSERVATIVE/PURE_FUNCTION/INTERNAL_SPECIAL_CASE 应该已在上面处理过"
    );
    
    // 获取节点的模式对象引用
    const auto& schema = node->schema();
    
    // 创建用于将模式的"形式"别名注解绑定到实际值的无序映射
    std::unordered_map<Symbol, Value*> formalToActual;
    
    // 遍历模式的参数列表
    for (const auto i : c10::irange(schema.arguments().size())) {
        // 获取参数的别名信息
        const at::AliasInfo* formal = schema.arguments()[i].alias_info();
        // 获取节点输入的实际值
        const auto& actualValue = node->inputs().at(i);
    
        // 如果没有别名注解，则跳过
        if (!formal) {
            continue;
        }
    
        // 如果实际值类型不可变，则跳过
        if (!isMutableTypeInternal(actualValue)) {
            continue;
        }
    
        // 对别名注解进行健全性检查
        TORCH_INTERNAL_ASSERT(
            formal->containedTypes().size() <= 1,
            "尚不支持别名分析的复合类型"
        );
        TORCH_INTERNAL_ASSERT(
            !formal->isWildcardBefore(),
            "一个输入值在开始时不可能是通配符"
        );
    
        // 特殊情况：如果别名信息包含类型但不包含之后的[]，例如 `Tensor(a!)[]`
        if (formal->containedTypes().size() == 1 && formal->beforeSets().empty()) {
            // 使用别名信息中的第一个包含的类型
            formal = &(formal->containedTypes()[0]);
        }
    
        // 获取形式别名注解的 beforeSet
        const auto& formalAlias = formal->beforeSet();
    // 如果已经绑定了这个别名，跳过处理
    if (formalToActual.count(formalAlias) != 0) {
      continue;
    }

    // 将形式参数绑定到实际值
    formalToActual[formalAlias] = actualValue;

    // 记录写操作
    if (formal->isWrite()) {
      registerWrite(actualValue, node);
    }

    // 处理 '->' 后面的设置
    if (formal->isWildcardAfter()) {
      TORCH_INTERNAL_ASSERT(
          formal->afterSets().size() == 1,
          "If the after set contains a wildcard, "
          "there should be no other alias sets specified.");
      // 如果 '->' 后包含通配符，设置通配符别名
      setWildcard(actualValue);
    } else {
      // 我们暂时不理解 '->' 后的其他情况，因此断言没有变化
      TORCH_INTERNAL_ASSERT(formal->beforeSets() == formal->afterSets());
    }
  }

  // 使用形式-实际映射为输出参数提供别名
  for (const auto i : c10::irange(schema.returns().size())) {
    const auto actual = node->outputs().at(i);
    const at::AliasInfo* formal = schema.returns()[i].alias_info();
    if (!formal) {
      // 这是一个新的张量，为其分配新的别名
      giveFreshAlias(actual);
      continue;
    }

    // 如果这种类型不能别名化，则跳过。这可能发生在 VarType 模式下
    if (!isMutableType(actual)) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        formal->containedTypes().size() <= 1,
        "Composite types for alias analysis not yet supported");
    TORCH_INTERNAL_ASSERT(formal->beforeSets() == formal->afterSets());
    if (formal->containedTypes().size() == 1 && formal->beforeSets().empty()) {
      // 使用别名信息中的第一个 containedType
      formal = &(formal->containedTypes()[0]);
    }
    if (formal->isWildcardBefore()) {
      TORCH_INTERNAL_ASSERT(
          formal->beforeSets().size() == 1,
          "If an output is a wildcard, "
          "there should be no other alias sets specified.");
      // 如果输出是通配符，则设置通配符别名
      setWildcard(actual);
      continue;
    }

    bool inputs_has_alias = false;
    for (const auto& formalAlias : formal->beforeSets()) {
      if (formalToActual.count(formalAlias)) {
        inputs_has_alias = true;
        auto toAlias = formalToActual.at(formalAlias);
        // 将实际值指向形式别名
        makePointerTo(actual, toAlias);
      }
    }
    // 如果所有遇到的别名注解都不在输入中:
    //   例如 foo(Tensor(a) self) -> Tensor(b)
    //   或 foo(Tensor(a) self) -> Tensor(b|c)
    // 否则，它是形式为 a|fresh，我们可以忽略，采取保守假设输出必须别名为 `a`，例如
    //   aten::cuda(Tensor(a) self) -> Tensor(a|fresh)
    if (!inputs_has_alias && !formal->beforeSets().empty()) {
      giveFreshAlias(actual);
    }

    // 记录写操作
    if (formal->isWrite()) {
      registerWrite(actual, node);
    }
  }
}

// 注册一个写入操作，记录值 `v` 被节点 `n` 写入
void AliasDb::registerWrite(const Value* v, Node* n, bool writeToContained) {
  // 如果值 `v` 不是可变类型，则无需注册写入操作
  if (!isMutableTypeInternal(v)) {
    // 如果值不可变，不需要注册写入操作
    return;
  }
  // 如果 `writeToContained` 为真，则注册值 `v` 被节点 `n` 写入到其所有包含值中
  if (writeToContained) {
    writeRegistry_->registerWriteToAllContained(v, n);
  } else {
    // 否则，直接注册值 `v` 被节点 `n` 写入
    writeRegistry_->registerWrite(v, n);
  }
}

void AliasDb::analyzeIf(Node* node) {
  // 对于 if 语句，输出的别名集合是 if 和 else 块生成的别名集合的并集
  const auto trueBlock = node->blocks().at(0);
  const auto falseBlock = node->blocks().at(1);
  analyze(trueBlock);
  analyze(falseBlock);

  // 遍历节点的每个输出
  for (const auto i : c10::irange(node->outputs().size())) {
    const auto nodeOutput = node->outputs()[i];

    const auto trueOutput = trueBlock->outputs().at(i);
    const auto falseOutput = falseBlock->outputs().at(i);

    // 创建指向节点输出 `nodeOutput` 的指针，指向 true 和 false 块的输出
    makePointerTo(nodeOutput, trueOutput);
    makePointerTo(nodeOutput, falseOutput);
  }
}

void AliasDb::analyzeLoop(Node* node) {
  const auto bodyBlock = node->blocks().at(0);
  const auto loopCarriedInputs = node->inputs().slice(2); // 跳过 max, cond
  const auto blockInputs = bodyBlock->inputs().slice(1); // 跳过 trip
  const auto blockOutputs = bodyBlock->outputs().slice(1); // 跳过 trip
  TORCH_INTERNAL_ASSERT(loopCarriedInputs.size() == blockInputs.size());
  TORCH_INTERNAL_ASSERT(blockOutputs.size() == node->outputs().size());

  // 运行循环体上的别名分析，直到块输出的别名信息收敛。将节点输入的别名复制到块输入
  mapAliases(blockInputs, loopCarriedInputs);

  // 分析循环体
  analyze(bodyBlock);

  // 将块输出的别名信息复制到节点输出
  mapAliases(node->outputs(), blockOutputs);
}

void AliasDb::analyzeGradOf(Node* node) {
  const auto grad_of_block = node->blocks().at(0);
  // 分析 grad_of 块
  analyze(grad_of_block);
  // 将 grad_of 块的输出的别名信息映射到节点的输出
  mapAliases(node->outputs(), grad_of_block->outputs());
}

void AliasDb::analyzeSubgraph(Node* node, std::shared_ptr<Graph> subgraph) {
  const auto subgraphBlock = subgraph->block();
  // CallFunction 节点有一个额外的第一个参数
  if (node->kind() == prim::CallFunction) {
    // 将子图块的输入的别名映射到节点的输入（跳过第一个参数）
    mapAliases(subgraphBlock->inputs(), node->inputs().slice(1));
  } else {
    // 否则，将子图块的输入的别名映射到节点的输入
    mapAliases(subgraphBlock->inputs(), node->inputs());
  }

  // 分析子图块
  analyze(subgraphBlock);

  // 注意：子图块的输出和节点的输出长度不一定相同。自动微分可能会在子图块中捕获额外的输出。
  TORCH_INTERNAL_ASSERT(
      subgraphBlock->outputs().size() >= node->outputs().size());
  for (size_t i = 0; i < node->outputs().size(); i++) {
    // 创建指向节点输出的指针，指向子图块的输出
    makePointerTo(node->outputs()[i], subgraphBlock->outputs()[i]);
  }
}

void AliasDb::analyzeSubgraph(Node* node) {
  const auto subgraph = node->g(attr::Subgraph);
  // 分析节点的子图
  return analyzeSubgraph(node, subgraph);
}
// 用于生成新值的节点
void AliasDb::analyzeCreator(Node* node) {
  // 遍历节点的所有输出值
  for (Value* output : node->outputs()) {
    // 给每个输出值分配一个新的别名
    giveFreshAlias(output);
  }
}

// 对从复合类型中提取值的节点进行分析。当前实现会为所有情况都创建通配符。
void AliasDb::analyzeExtractor(Node* node) {
  // 遍历节点的所有输出值
  for (const auto output : node->outputs()) {
    // 设置输出值为通配符
    setWildcard(output);
  }
}

// 对于 torch.chunk()，所有返回的张量可能与输入张量别名相同
void AliasDb::analyzeChunk(Node* node) {
  // 遍历节点的所有输出值
  for (auto output : node->outputs()) {
    // 将输出值指向输入值
    makePointerTo(output, node->input());
  }
}

void AliasDb::analyzeFork(Node* node) {
  // 遍历节点的所有输入值
  for (const auto input : node->inputs()) {
    // 设置输入值为通配符
    setWildcard(input);
  }

  // 给 fork 节点发出的每个输出值分配一个新的别名
  for (const auto output : node->outputs()) {
    giveFreshAlias(output);
  }
}

void AliasDb::analyzeWait(Node* node) {
  // 断言节点类型为 aten::wait
  TORCH_INTERNAL_ASSERT(node->kind() == aten::wait);
  // 遍历节点的所有输出值
  for (const auto output : node->outputs()) {
    // 设置输出值为通配符
    setWildcard(output);
  }
  // wait 所等待的 fork 子图可能会写入其任何输入值。由于无法可靠地恢复 fork 输入值，为安全起见，注册对所有通配符的写入。
  writeRegistry_->registerWriteToAllWildcards(node);
}

void AliasDb::analyzeAwaitable(Node* node) {
  // 遍历节点的所有输入值
  for (const auto input : node->inputs()) {
    // 设置输入值为通配符
    setWildcard(input);
  }

  // 给 awaitable 节点的每个输出值分配一个新的别名
  for (const auto output : node->outputs()) {
    giveFreshAlias(output);
  }
}

void AliasDb::analyzeAwaitableWait(Node* node) {
  // 断言节点类型为 prim::awaitable_wait
  TORCH_INTERNAL_ASSERT(node->kind() == prim::awaitable_wait);
  // 遍历节点的所有输出值
  for (const auto output : node->outputs()) {
    // 设置输出值为通配符
    setWildcard(output);
  }
  // 等待的 awaitable 子图可能会写入其任何输入值。由于无法可靠地恢复 awaitable 输入值，为安全起见，注册对所有通配符的写入。
  writeRegistry_->registerWriteToAllWildcards(node);
}

void AliasDb::analyzeRpcAsync(Node* node) {
  // 遍历节点的所有输入值
  for (const auto input : node->inputs()) {
    // 设置输入值为通配符
    setWildcard(input);
  }

  // 给 rpc_async 节点发出的每个输出值分配一个新的别名
  for (const auto output : node->outputs()) {
    giveFreshAlias(output);
  }
}

namespace {
std::optional<bool> getConstantBooleanInput(
    Node* node,
    const std::string& inputName) {
  // 断言节点包含指定名称的输入
  TORCH_INTERNAL_ASSERT(
      node->hasNamedInput(inputName), inputName + " input is expected");
  // 获取指定名称的输入值
  auto value = node->namedInput(inputName);
  // 断言输入值的类型为布尔类型
  TORCH_INTERNAL_ASSERT(
      value->type() == BoolType::get(),
      inputName + "training input is expected to be a bool");
  // 将输入值解析为布尔值并返回
  return constant_as<bool>(value);
}
} // namespace

// 对于 batch_norm，因为当前不支持 (a!)? 注释，并且行为取决于 training 的值，进行自定义处理
// 分析批量归一化节点的别名信息
void AliasDb::analyzeBatchNorm(Node* node) {
  // 对于每个输出值，分配一个新的别名
  for (Value* output : node->outputs()) {
    giveFreshAlias(output);
  }

  // 如果处于冻结状态，则直接返回，不进行进一步分析
  if (isFrozen_) {
    return;
  }

  // 获取节点的输入 "training" 的常量布尔值
  auto isTraining = getConstantBooleanInput(node, "training");

  // 如果 "training" 不是常量或者其值为 true
  if (!isTraining.has_value() || *isTraining) {
    // 确保节点具有名为 "running_mean" 的输入
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_mean"), "running_mean input is expected");
    auto runningMean = node->namedInput("running_mean");

    // 确保节点具有名为 "running_var" 的输入
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_var"), "running_var input is expected");
    auto runningVar = node->namedInput("running_var");

    // 注册对 "running_mean" 和 "running_var" 的写操作
    registerWrite(runningMean, node);
    registerWrite(runningVar, node);
  }
}

// 自定义行为用于实例归一化，因为当前不支持 (a!)? 注释，并且行为取决于 "use_input_stats" 的值
void AliasDb::analyzeInstanceNorm(Node* node) {
  // 对于每个输出值，分配一个新的别名
  for (Value* output : node->outputs()) {
    giveFreshAlias(output);
  }

  // 获取节点的输入 "use_input_stats" 的常量布尔值
  auto useInputStats = getConstantBooleanInput(node, "use_input_stats");

  // 如果 "use_input_stats" 不是常量或者其值为 true
  if (!useInputStats.has_value() || *useInputStats) {
    // 确保节点具有名为 "running_mean" 的输入
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_mean"), "running_mean input is expected");
    auto runningMean = node->namedInput("running_mean");

    // 确保节点具有名为 "running_var" 的输入
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_var"), "running_var input is expected");
    auto runningVar = node->namedInput("running_var");

    // 注册对 "running_mean" 和 "running_var" 的写操作
    registerWrite(runningMean, node);
    registerWrite(runningVar, node);
  }
}

// SetAttr: 写入到 `self` 字段
void AliasDb::analyzeSetAttr(Node* node) {
  // 获取节点的第一个输入作为 `self`
  const auto self = node->inputs().at(0);
  // 确保 `self` 的类型是 ClassType
  TORCH_INTERNAL_ASSERT(self->type()->kind() == TypeKind::ClassType);
  // 注册对 `self` 的写操作
  registerWrite(self, node);
  // 还要将被设置的值变为通配符
  const auto newValue = node->inputs().at(1);
  setWildcard(newValue);
}

// 用于任何没有精确别名摘要的情况，可能会写入任何输入并产生通配符
void AliasDb::analyzeConservative(Node* node) {
  // 遍历节点的每个输入
  for (const auto input : node->inputs()) {
    // 如果输入不是可变类型，则跳过
    if (!isMutableTypeInternal(input)) {
      continue;
    }
    // 注册对输入的写操作，并指定写入到包含的对象中
    registerWrite(input, node, /*writeToContained=*/true);
    // 将输入设为通配符
    setWildcard(input);
  }

  // 对节点的每个输出，都设置为通配符
  for (const auto output : node->outputs()) {
    setWildcard(output);
  }
}

// 列出使用列表的功能性非逃逸使用
bool AliasDb::functionalNonEscapingListUse(const Use& use) const {
  // 获取使用列表的节点和偏移量
  Node* n = use.user;
  size_t offset = use.offset;
  // 获取节点的输入作为容器
  Value* container = n->inputs().at(offset);

  // 只考虑列表类型的 aten 操作使用
  if (!container->type()->cast<ListType>()) {
    // 如果没有使用者或者使用者的类型不是 prim::Return，则返回 false
    return false;
  }

  /*
  一般情况下，我们认为任何进入另一个容器的 Value 都会进入堆，并且会与同类型的所有其他堆值发生别名关系。
  这种方法的优势包括：
  - 如果采用更复杂的方法，会使得许多复合列表/容器操作变得棘手。
  - 限制 AliasDb 的大小，因为大小为 10 的容器只包含 1 个内存 DAG 元素，而不是 10 个。
  - 当容器逃逸图表时，我们无需担心将包含的元素添加到通配符集合中。
  该方法的缺点是无法处理常见情况，即构造列表并将其传递给 aten 操作的情况。在这里，优化了一组常见操作，其中输出不与列表或列表元素发生别名关系。
  */

  // 只在图的输出中使用 - 没有进一步的使用，
  // 因此不会出现包含元素泄漏的情况
  if (use.user->kind() == prim::Return) {
    // 如果使用者是 prim::Return，并且属于当前图块，则返回 true
    return use.user->owningBlock() == graph_->block();
  }

  switch (use.user->kind()) {
    // 对于以下 aten 操作，返回 true
    case aten::cat:
    case aten::broadcast_tensors:
    case aten::stack:
    case aten::vstack:
    case aten::hstack:
    case aten::dstack:
      return true;
  }
  auto op = use.user->maybeOperator();
  if (op && op->aliasAnalysisKind() == AliasAnalysisKind::PURE_FUNCTION) {
    // 如果操作存在且其别名分析类型为 PURE_FUNCTION，则返回 true
    return true;
  }
  // 其他情况返回 false
  return false;
// 判断是否在函数式非逃逸元组使用情况下使用
bool AliasDb::functionalNonEscapingTupleUse(const Use& use) const {
  // 获取使用该 Use 对象的节点
  Node* n = use.user;
  // 获取使用位置的偏移量
  size_t offset = use.offset;
  // 获取容器对象，该对象是节点 n 在 offset 处的输入
  Value* container = n->inputs().at(offset);
  // 如果容器对象不是 TupleType 类型，则返回 false
  if (!container->type()->cast<TupleType>()) {
    return false;
  }
  // TODO(T97387453): Cover more ops that do not let escape tuples' elements.
  // 判断节点 n 是否是 prim::Return 类型，用于判断是否在返回语句中使用
  bool in_return_outputs = use.user->kind() == prim::Return;
  // 判断节点 n 是否在当前图的块中，用于判断是否不在嵌套子图中使用
  bool not_in_nested_subgraph = use.user->owningBlock() == graph_->block();
  // 返回是否在返回语句中使用且不在嵌套子图中使用的结果
  return in_return_outputs && not_in_nested_subgraph;
}

// 对列表、字典或元组构造进行分析：为实际容器创建一个别名元素，标记所有输入为通配符，
// 因为它们已经进入容器。然后，将适当类型的通配符集合添加到容器中的包含元素。
void AliasDb::analyzeContainerConstruct(Node* node) {
  // 断言节点的类型是列表构造、字典构造或元组构造
  TORCH_INTERNAL_ASSERT(
      node->kind() == prim::ListConstruct ||
      node->kind() == prim::DictConstruct ||
      node->kind() == prim::TupleConstruct);

  // 对于不可变类型的元组，不做进一步分析
  if (!isMutableTypeInternal(node->output())) {
    return;
  }

  // 断言节点只有一个输出
  TORCH_INTERNAL_ASSERT(node->outputs().size() == 1);
  auto container = node->output();

  // 优化处理：
  // 如果列表仅在 aten 操作中使用一次，并且操作的输出与输入没有别名关系，
  // 则将所有输入添加到列表的包含元素中，而不是通配符集合。
  if (container->uses().size() == 1 &&
      (functionalNonEscapingListUse(container->uses().at(0)) ||
       functionalNonEscapingTupleUse(container->uses().at(0)))) {
    // 为容器对象创建一个新的别名
    giveFreshAlias(container, false);
    // 将节点的所有输入添加到容器的包含元素中
    for (Value* v : node->inputs()) {
      addToContainedElements(v, container);
    }
    return;
  }

  // 为容器对象创建一个新的别名
  giveFreshAlias(container);
  // 获取容器元素的映射
  auto container_elem = elementMap_.at(container);
  // 遍历节点的所有输入
  for (auto input : node->inputs()) {
    // 设置输入为通配符，并获取可能的通配符元素
    auto maybe_wildcard_elem = setWildcard(input);
    // 如果存在通配符元素，则将其添加到容器元素的包含元素中
    if (maybe_wildcard_elem) {
      memoryDAGBuilder_->addToContainedElements(
          *maybe_wildcard_elem, container_elem);
    }
  }
}

// BroadcastingChunk: 所有输入都被广播，然后单独进行分块。
// 这是仅在图融合器中使用的中间节点。
void AliasDb::analyzeBroadcastingChunk(Node* node) {
  // 获取节点的输入和输出
  auto inputs = node->inputs();
  auto outputs = node->outputs();
  // 获取节点的块数
  auto nchunks = node->i(attr::chunks);
  // 遍历所有输入
  for (const auto index : c10::irange(inputs.size())) {
    // 每个 inputs[i] 都由恰好 `nchunks` 个不同的输出张量别名：
    // inputs[i] 产生 outputs[i * nchunks + k]，其中 k 在 [0..nchunks) 范围内
    auto output_begin = outputs.begin() + index * nchunks;
    // 遍历当前输入的所有输出
    for (auto it = output_begin; it != output_begin + nchunks; ++it) {
      // 创建指向 outputs[it] 的指针，并使其别名为 inputs[index]
      makePointerTo(*it, inputs.at(index));
    }
  }
}
// 检查给定元素是否是非别名值，即可能指向图中的别名类型，但本身不会是别名
bool AliasDb::nonAliasingValue(const Value* elem) const {
  // 这些值可能指向图中的别名类型，例如 None 值可能指向可选 if 节点的输出，
  // 但它们本身不会是别名
  return elem->mustBeNone() || elem->node()->kind() == prim::Uninitialized;
}

// 注册 `from` 是指向 `to` 的指针的事实
void AliasDb::makePointerTo(const Value* from, const Value* to) {
  if (nonAliasingValue(from) || nonAliasingValue(to)) {
    // 如果任一值保证是非别名，则不需要连接这两个元素。然而，非通配符的别名类型具有内存 DAG 元素，因此如果需要，我们会创建一个
    giveFreshAlias(from);
    giveFreshAlias(to);
    return;
  }

  // 不可变类型容器（如 `Optional`、`Tuple`、`Future` 和 `Union`）的包含类型是统一的，
  // 因此这些类型可以是可变的或不可变的，并且可以指向可变或不可变的类型。
  // `Any` 是可变的，但可以通过细化指向不可变类型
  if (isMutableTypeInternal(from) != isMutableTypeInternal(to)) {
    return;
  }
  // 都是不可变的
  if (!isMutableTypeInternal(from)) {
    return;
  }
  if (from == to) {
    return;
  }

  // 到了这一步，我们正在处理两个可变类型
  auto from_el = getOrCreateElement(from);
  auto to_el = getOrCreateElement(to);

  memoryDAGBuilder_->makePointerTo(from_el, to_el);
}

// 将 `inner` 添加到容器 `container` 的包含元素中
void AliasDb::addToContainedElements(const Value* inner, const Value* container) {
  if (!isMutableTypeInternal(inner)) {
    return;
  }

  auto inner_el = getOrCreateElement(inner);
  auto cont_el = getOrCreateElement(container);

  memoryDAGBuilder_->addToContainedElements(inner_el, cont_el);
}

// 检查两个值 `a` 和 `b` 是否可能是别名
bool AliasDb::mayAlias(const Value* a, const Value* b) const {
  if (!isMutableTypeInternal(a) || !isMutableTypeInternal(b)) {
    return false;
  }

  return memoryDAG_->mayAlias(elementMap_.at(a), elementMap_.at(b));
}

// 检查两个值集合 `a` 和 `b` 是否可能存在别名
bool AliasDb::mayAlias(const ValueSet& a, const ValueSet& b) const {
  if (a.empty() || b.empty()) {
    return false;
  }

  // 记录组 `a` 中所有内存位置
  MemoryLocations aMemLocs;
  for (const auto value : a) {
    auto it = elementMap_.find(value);
    if (it != elementMap_.end()) {
      aMemLocs |= memoryDAG_->getMemoryLocations(it->second);
    }
  }

  // 如果组 `b` 的任何内存位置与之重叠，则返回 true
  for (const auto value : b) {
    auto it = elementMap_.find(value);
    if (it != elementMap_.end()) {
      if (aMemLocs.intersects(memoryDAG_->getMemoryLocations(it->second))) {
        return true;
      }
    }
  }
  // 没有重叠，因此组 `a` 和 `b` 不共享内存位置
  return false;
}

// 检查值 `a` 和 `b` 是否可能包含别名
bool AliasDb::mayContainAlias(Value* a, Value* b) const {
  if (!isMutableTypeInternal(a) || !isMutableTypeInternal(b)) {
    return false;
  }
  return memoryDAG_->mayContainAlias(elementMap_.at(a), elementMap_.at(b));
}
// 返回与给定值数组 `vs` 中元素对应的 Element 指针向量
std::vector<Element*> AliasDb::getElements(at::ArrayRef<Value*> vs) const {
  // 创建空的 Element 指针向量
  std::vector<Element*> elements;
  // 遍历输入数组 `vs` 中的每个值
  for (const auto& val : vs) {
    // 如果该值属于可变类型，则将其对应的 Element 指针添加到向量中
    if (isMutableTypeInternal(val)) {
      elements.push_back(elementMap_.at(val));
    }
  }
  // 返回包含元素的指针向量
  return elements;
}

// 检查 `a` 和 `b` 数组中的值是否可能存在别名关系
bool AliasDb::mayContainAlias(
    const at::ArrayRef<Value*> a,
    const at::ArrayRef<Value*> b) const {
  // 获取 `a` 数组中的所有 Element 指针
  auto a_elems = getElements(a);
  // 如果 `a_elems` 为空，则返回 false；否则检查 `b` 数组中的值是否与 `a_elems` 中的任意元素可能存在别名
  return a_elems.empty() ? false
                         : memoryDAG_->mayContainAlias(a_elems, getElements(b));
}

// 检查单个值 `a` 与数组 `b` 中的值是否可能存在别名关系
bool AliasDb::mayContainAlias(Value* a, const at::ArrayRef<Value*> b) const {
  // 如果 `a` 不属于可变类型，则返回 false
  if (!isMutableTypeInternal(a)) {
    return false;
  }
  // 获取数组 `b` 中的所有 Element 指针
  auto b_elems = getElements(b);
  // 如果 `b_elems` 为空，则返回 false；否则检查 `a` 与 `b_elems` 中的任意元素是否可能存在别名
  return b_elems.empty()
      ? false
      : memoryDAG_->mayContainAlias(elementMap_.at(a), b_elems);
}

// 将 `from` 列表中的每个值指向 `to` 列表中对应的值
void AliasDb::mapAliases(at::ArrayRef<Value*> from, at::ArrayRef<Value*> to) {
  // 断言 `to` 和 `from` 列表长度相同
  TORCH_INTERNAL_ASSERT(to.size() == from.size());
  // 遍历 `to` 列表中的每个索引 `i`，将 `from[i]` 指向 `to[i]`
  for (const auto i : c10::irange(to.size())) {
    makePointerTo(from[i], to[i]);
  }
}

// 只能从 `create_functional_graphs` 中调用，断言用于防止意外使用
// FIXME：重构 AliasDb 构造以使其更能抵御变化，这样就不需要这种 hack 了
void AliasDb::createValue(const Value* value) {
  // 断言 `value` 的类型是可变类型
  TORCH_INTERNAL_ASSERT(isMutableTypeInternal(value->type()));
  // 在内存 DAG 中创建一个新值，并将其与 `value` 关联
  auto new_elem = memoryDAG_->unsafeMakeFreshValue(value);
  elementMap_[value] = new_elem;
}

// 给定一个值，为其创建一个新的别名，并根据需要添加通配符到包含的类型元素
void AliasDb::giveFreshAlias(
    const Value* value,
    bool add_wildcard_to_contained_elems) {
  // 将值的类型映射为其别名类型集合指针
  auto maybe_mut_types = mapTypeToAliasTypeSetPtr(value->type());
  // 如果映射不存在，则返回
  if (!maybe_mut_types) {
    return;
  }

  // 如果已经为该值创建了别名，则跳过
  if (elementMap_.count(value)) {
    return;
  }

  // 创建一个新的值，并将其作为 `value` 的新别名
  auto new_elem = memoryDAGBuilder_->makeFreshValue(value);
  elementMap_[value] = new_elem;
  
  // 如果需要向包含的元素添加通配符
  if (add_wildcard_to_contained_elems) {
    // 如果映射的类型集合大小大于 1，则将通配符类型元素指向所有包含的类型
    if (maybe_mut_types->size() > 1) {
      pointUnionTypeElementToAllContainedTypes(new_elem, *maybe_mut_types);
    } else {
      // 否则，将包含的类型添加到新元素中
      addContainedTypesToFreshElement(new_elem, *maybe_mut_types);
    }
  }
}

// 获取或创建与给定值关联的 Element 指针
Element* AliasDb::getOrCreateElement(const Value* value) {
  // 如果值不存在于元素映射中，则为其创建一个新的别名
  if (!elementMap_.count(value)) {
    giveFreshAlias(value);
  }
  // 返回与值关联的 Element 指针
  return elementMap_.at(value);
}

// 替换现有值 `existing` 的别名信息为新值 `new_value`
void AliasDb::replaceWithNewValue(Value* existing, Value* new_value) {
  // 断言两个值的非形状类型严格相等，用于替换别名信息
  TORCH_INTERNAL_ASSERT(
      *unshapedType(existing->type()) == *unshapedType(new_value->type()),
      "如果要替换别名信息，类型必须严格相等。已有值：'",
      existing->type()->repr_str(),
      "', 新值：'",
      new_value->type()->repr_str(),
      "'");
  // 如果 `existing` 不属于可变类型，则返回
  if (!isMutableTypeInternal(existing)) {
    return;
  }
  // 获取 `existing` 对应的 Element 指针，并将其关联到 `new_value` 上，然后移除 `existing` 的映射
  auto existing_elem = elementMap_.at(existing);
  elementMap_[new_value] = existing_elem;
  elementMap_.erase(existing);
  // 更新 Element 指针的值为 `new_value`
  existing_elem->values = {new_value};
}
// 将值从一个位置复制到另一个位置，并确保它们的类型严格相等
void AliasDb::copyValue(Value* from, Value* to) {
  // 断言源值和目标值的非形状类型必须严格相等，否则会抛出错误信息
  TORCH_INTERNAL_ASSERT(
      *unshapedType(from->type()) == *unshapedType(to->type()),
      "Types must be strictly equal if you are copying aliasing information. ",
      "Got from: '",
      from->type()->repr_str(),
      "', to: '",
      to->type()->repr_str(),
      "'");
  // 如果目标值的类型是不可变的，直接返回，不进行复制操作
  if (!isMutableTypeInternal(to)) {
    return;
  }
  // 获取源值在元素映射中的原始元素，并将目标值映射到该原始元素
  auto origElem = elementMap_.at(from);
  elementMap_[to] = origElem;
  // 将目标值添加到原始元素的值集合中
  origElem->values.insert(to);
}

// 尝试将节点 `n` 移动到节点 `movePoint` 后面，确保拓扑排序的有效性
bool AliasDb::moveAfterTopologicallyValid(Node* n, Node* movePoint) {
  return tryMove(n, movePoint, MoveSide::AFTER, /*dryRun=*/false);
}

// 判断是否可以将节点 `n` 移动到节点 `movePoint` 后面，仅进行拓扑排序的检查
bool AliasDb::couldMoveAfterTopologically(Node* n, Node* movePoint) {
  return tryMove(n, movePoint, MoveSide::AFTER, /*dryRun=*/true);
}

// 尝试将节点 `n` 移动到节点 `movePoint` 前面，考虑拓扑排序和移动方向的影响
bool AliasDb::moveBeforeTopologicallyValid(Node* n, Node* movePoint) {
  // 我们需要区分移动方向，而不仅仅是移动到 n->prev() 的位置。考虑以下情况的示例：
  // 如果依赖图形状如下
  //   n -> movePoint -> o
  // 那么 moveBefore(o) 的结果将是
  //   n, o, movePoint
  // 但 moveAfter(n) 会返回 false。
  return tryMove(n, movePoint, MoveSide::BEFORE, /*dryRun=*/false);
}

// 判断是否可以将节点 `n` 移动到节点 `movePoint` 前面，仅进行拓扑排序的检查
bool AliasDb::couldMoveBeforeTopologically(Node* n, Node* movePoint) {
  return tryMove(n, movePoint, MoveSide::BEFORE, /*dryRun=*/true);
}

// 检查一组值中是否有任何一个存在写操作
bool AliasDb::hasWriters(const at::ArrayRef<Value*>& values) const {
  return std::any_of(values.begin(), values.end(), [&](Value* value) {
    return hasWriters(value);
  });
}

// 判断一组值是否会在当前图形作用域之外逃逸
bool AliasDb::escapesScope(const at::ArrayRef<Value*>& vs) const {
  return mayContainAlias(graph_->inputs(), vs) ||
      mayContainAlias(graph_->outputs(), vs) || mayAliasWildcard(vs);
}

// 判断改变两组值之间的别名关系是否安全，遵循特定的正确性条件
bool AliasDb::safeToChangeAliasingRelationship(
    const at::ArrayRef<Value*>& a,
    const at::ArrayRef<Value*>& b) const {
  // 如果任一组值存在写操作，则返回不安全
  if (hasWriters(a) || hasWriters(b)) {
    return false;
  }
  // 如果两组值都可能逃逸当前作用域，则返回不安全
  return !(escapesScope(a) && escapesScope(b));
}

// 辅助类，用于支持拓扑安全的节点移动。详见 `tryMove()` 函数的详细说明。
class AliasDb::WorkingSet {
 public:
  explicit WorkingSet(Node* mover, const AliasDb& aliasDb) : aliasDb_(aliasDb) {
    mover_ = mover;
    // 将与 `mover_` 相同块中的用户节点添加到移动集合中
    for (const auto user : getUsersSameBlock(mover_)) {
      moverUsers_.insert(user);
    }
    // 获取 `mover_` 的写集合和读集合，并添加到相应的集合中
    moverWrites_ |= aliasDb_.getWrites(mover_);
    moverReads_ |= aliasDb_.getReads(mover_);
  }

  // 将节点 `n` 添加到工作集合中
  void add(Node* n) {
    nodes_.push_back(n);
    node_to_index_[n] = nodes_.size() - 1;
    // 将与 `n` 相同块中的用户节点添加到用户集合中
    for (const auto user : getUsersSameBlock(n)) {
      users_.insert(user);
    }
    // 获取 `n` 的写集合和读集合，并添加到相应的集合中
    writes_ |= aliasDb_.getWrites(n);
    reads_ |= aliasDb_.getReads(n);
  }

  // 清除移动节点 `mover_` 的信息
  void eraseMover() {
    mover_ = nullptr;
    moverWrites_.clear();
    moverReads_.clear();
    // 清空 moverUsers_ 容器
    moverUsers_.clear();
  }

  // 返回依赖节点列表
  const std::vector<Node*>& dependentNodes() {
    return nodes_;
  }

  // 判断工作集是否依赖于节点 `n`
  // 如果没有 mover_ 且 nodes_ 为空，则不依赖
  bool dependsOn(Node* n) const {
    if (!mover_ && nodes_.empty()) {
      return false;
    }

    // 如果存在数据依赖或者可变性依赖，则返回 true
    return hasDataDependency(n) || hasMutabilityDependency(n);
  }

 private:
  // 判断是否存在数据依赖
  bool hasDataDependency(Node* n) const {
    if (!mover_ && nodes_.empty()) {
      return false;
    }
    // 确定数据依赖的基准节点
    const Node* pivot = mover_ ? mover_ : nodes_.front();
    // 如果节点 `n` 在基准节点之后，则检查是否为其产出
    if (n->isAfter(pivot)) {
      return producesFor(n);
    } else {
      // 否则，检查是否为其消费
      return consumesFrom(n);
    }
  }

  // 判断是否存在可变性依赖
  bool hasMutabilityDependency(Node* n) const {
    // 检查节点 `n` 的写操作是否与当前工作集的读操作有交集
    const auto& nWrites = aliasDb_.getWrites(n);
    if (reads_.intersects(nWrites)) {
      return true;
    }
    // 如果存在 mover_ 并且节点 `n` 的写操作与 mover_ 的读操作有交集，则返回 true
    if (mover_ && moverReads_.intersects(nWrites)) {
      return true;
    }

    // 检查当前工作集的写操作是否与节点 `n` 的读操作有交集
    const auto& nReads = aliasDb_.getReads(n);
    if (writes_.intersects(nReads)) {
      return true;
    }
    // 如果存在 mover_ 并且当前工作集的写操作与节点 `n` 的读操作有交集，则返回 true
    if (mover_ && moverWrites_.intersects(nReads)) {
      return true;
    }
    // 否则，返回 false
    return false;
  }

  // 判断工作集是否为节点 `n` 产生任何被消费的值
  bool producesFor(Node* n) const {
    // 相当于询问：所有工作集中节点的使用集合是否包含节点 `n`
    if (mover_ && moverUsers_.count(n)) {
      return true;
    }
    // 检查工作集中是否存在节点 `n` 的使用者
    return users_.count(n) != 0;
  }

  // 判断工作集是否消费节点 `n` 产生的任何值
  bool consumesFrom(Node* n) const {
    // 获取与节点 `n` 在同一块中的输出的所有使用者
    const auto users = getUsersSameBlock(n);

    // 如果存在 mover_ 并且同一块中存在 mover_ 的使用者，则返回 true
    if (mover_ && users.count(mover_)) {
      return true;
    }
    // 否则，检查是否有任何节点是节点 `n` 的使用者
    return std::any_of(users.begin(), users.end(), [&](Node* user) {
      return node_to_index_.find(user) != node_to_index_.end();
    });
  }

  // 获取与节点 `n` 的输出相关的所有使用者，位于同一块中
  std::unordered_set<Node*> getUsersSameBlock(Node* n) const {
    std::unordered_set<Node*> users;
    for (const auto output : n->outputs()) {
      for (const auto& use : output->uses()) {
        // 查找与 use.user 和节点 `n` 相同块的节点
        if (auto sameBlock = findSameBlock(use.user, n)) {
          users.insert(sameBlock);
        }
      }
    }
    return users;
  }

  // 向上遍历 `target` 的代码块，直到找到与 `n` 共享块的节点
  // 如果找不到（例如，因为 `n` 是内部块而目标在外部），则返回 nullptr
  static Node* findSameBlock(Node* target, Node* n) {
    // 断言 target 和 n 属于同一图
    TORCH_INTERNAL_ASSERT(target->owningGraph() == n->owningGraph());
    // 如果 target 和 n 属于同一块，则返回 target
    if (target->owningBlock() == n->owningBlock()) {
      return target;
  } else {
    // 如果用户在一个子块中，向上遍历区块链，直到找到与当前节点共享区块的节点
    auto curNode = target;
    while (curNode->owningBlock() != n->owningBlock()) {
      curNode = curNode->owningBlock()->owningNode();
      // 如果遍历到空节点，则返回空指针
      if (curNode == nullptr) {
        return curNode;
      }
    }
    // 返回找到的共享区块的节点
    return curNode;
  }

  const AliasDb& aliasDb_;
  // 存储所有节点的向量
  std::vector<Node*> nodes_;
  // 用于节点的额外数据结构，以实现更快的查找
  // 因为 tryMove 方法使用频繁，我们希望它尽可能快速
  std::unordered_map<Node*, int64_t> node_to_index_;

  // Mover 的依赖项。我们单独跟踪这些，因为可能会从工作集中删除移动者
  Node* mover_;
  // 移动者写入的内存位置集合
  MemoryLocations moverWrites_;
  // 移动者读取的内存位置集合
  MemoryLocations moverReads_;
  // 使用移动者的节点集合
  std::unordered_set<Node*> moverUsers_;

  // 使用工作集的节点集合
  std::unordered_set<Node*> users_;
  // 被工作集写入的值 => 写入该值的节点数
  MemoryLocations writes_;
  // 工作集读取的值 => 读取该值的节点数
  MemoryLocations reads_;
};

// 尝试在保留值依赖关系的情况下将 `toMove` 移动到 `movePoint` 之前或之后。
// 如果 `dryRun` 被设置为 true，则仅检查移动是否可能，而不实际执行移动。
//
// 基本方法是：从 `toMove` 开始，向 `movePoint` 前进，逐个节点建立一个“工作集”。
// 当无法移动过节点（因为它依赖于工作集中的某些节点）时，将其添加到工作集中，并继续移动，直到达到 `moveAfter`。
//
// 返回 false 表示无法进行此移动。
bool AliasDb::tryMove(
    Node* toMove,        // 要移动的节点
    Node* movePoint,     // 移动目标节点
    MoveSide moveSide,   // 移动的方向（在之前还是之后）
    bool dryRun) {       // 是否为试运行

  if (toMove->owningBlock() != movePoint->owningBlock()) {
    return false;  // 如果 `toMove` 和 `movePoint` 不在同一个块中，无法移动
  }
  if (toMove == movePoint) {
    return true;   // 如果 `toMove` 就是 `movePoint`，无需移动
  }

  // 1. 从 `this` 向 `movePoint` 移动，建立依赖的工作集
  WorkingSet workingSet(toMove, *this);

  // 根据 `toMove` 和 `movePoint` 的相对位置确定移动方向
  int direction;
  if (toMove->isAfter(movePoint)) {
    direction = kPrevDirection;  // 向前移动
  } else {
    direction = kNextDirection;  // 向后移动
  }

  auto curNode = toMove->next_in_graph[direction];

  // 判断 `toMove` 是否在移动方向上，并且当前节点就是 `movePoint`
  bool toMoveIsOnMoveSide =
      (moveSide == MoveSide::BEFORE && toMove->isBefore(movePoint)) ||
      (moveSide == MoveSide::AFTER && toMove->isAfter(movePoint));

  if (toMoveIsOnMoveSide && curNode == movePoint) {
    return true;  // 如果 `toMove` 已经在正确的位置，无需移动
  }

  // 如果 `toMove` 或者 `movePoint` 有副作用，或者移动方向不正确，不能移动
  if (toMove->hasSideEffects() ||
      (!toMoveIsOnMoveSide && movePoint->hasSideEffects())) {
    return false;
  }

  // 逐个节点向前移动
  while (curNode != movePoint) {
    // 如果当前节点有副作用，则无法重新排序
    if (curNode->hasSideEffects()) {
      return false;
    }

    if (workingSet.dependsOn(curNode)) {
      // 如果无法移动过当前节点，将其添加到工作集中
      workingSet.add(curNode);
    }
    curNode = curNode->next_in_graph[direction];
  }

  // 2. 判断是否可以完全移动到 `movePoint`

  // 如果 `toMove` 在 `movePoint` 之前，并且需要将 `toMove` 和其依赖项分开
  const bool splitToMoveAndDeps =
      (moveSide == MoveSide::BEFORE && toMove->isBefore(movePoint)) ||
      (moveSide == MoveSide::AFTER && toMove->isAfter(movePoint));

  if (splitToMoveAndDeps) {
    // 将 `this` 从依赖关系中移除，以便可以将依赖项移动到 `movePoint` 之下，同时保持 `toMove` 在上方
    workingSet.eraseMover();
  }

  // 检查是否可以将工作集合移动到移动点之后
  if (workingSet.dependsOn(movePoint)) {
    // 如果不能，说明当前节点与移动点之间存在中间依赖关系，无法执行移动操作
    return false;
  }

  if (dryRun) {
    // 如果是 dryRun 模式，则直接返回 true，表示可以执行移动操作
    return true;
  }

  // 3. 执行移动操作
  TORCH_INTERNAL_ASSERT(curNode == movePoint);
  if (splitToMoveAndDeps) {
    // 移动节点 `toMove` 到 `movePoint` 的指定侧
    move(toMove, movePoint, moveSide);

    // 然后移动所有依赖于 `toMove` 的节点到 `movePoint` 的另一侧
    const auto reversed =
        moveSide == MoveSide::BEFORE ? MoveSide::AFTER : MoveSide::BEFORE;
    for (auto n : workingSet.dependentNodes()) {
      move(n, curNode, reversed);
      curNode = n;
    }
  } else {
    // 直接将 `toMove` 节点追加或者插入到 `movePoint` 的指定侧
    move(toMove, curNode, moveSide);
    curNode = toMove;
    for (auto n : workingSet.dependentNodes()) {
      move(n, curNode, moveSide);
      curNode = n;
    }
  }
  // 所有操作成功完成，返回 true
  return true;
}

// 辅助函数，用于通用化 `tryMove`
void AliasDb::move(Node* toMove, Node* movePoint, MoveSide moveSide) {
  switch (moveSide) {
    case MoveSide::BEFORE:
      // 将 `toMove` 节点移动到 `movePoint` 节点之前
      toMove->moveBefore(movePoint);
      break;
    case MoveSide::AFTER:
      // 将 `toMove` 节点移动到 `movePoint` 节点之后
      toMove->moveAfter(movePoint);
      break;
  }
}

bool AliasDb::writesToWildcard(Node* n) const {
  if (!writeIndex_->count(n)) {
    // 如果 `n` 不在写入索引中，返回 false
    return false;
  }
  const auto& writes = writeIndex_->at(n);

  // 检查写入的内存位置是否包含通配符元素
  for (const auto& pr : wildcardIndex_) {
    const auto wildcardElement = pr.second;
    if (writes.test(wildcardElement->index)) {
      // 如果写入的内存位置包含通配符元素的索引，返回 true
      return true;
    }
  }
  return false;
}

bool AliasDb::mayAliasWildcard(const Value* v) const {
  if (auto e = getWildcard(v->type())) {
    // 如果存在与给定值类型相对应的通配符元素，则返回该元素与 `v` 的内存别名关系
    return memoryDAG_->mayAlias(elementMap_.at(v), e);
  }
  // 如果不存在与给定值类型相对应的通配符元素，返回 false
  return false;
}

bool AliasDb::mayAliasWildcard(const at::ArrayRef<Value*> vs) const {
  // 检查数组中是否有任何值与通配符元素存在内存别名关系
  return std::any_of(
      vs.begin(), vs.end(), [&](Value* v) { return mayAliasWildcard(v); });
}

std::optional<Element*> AliasDb::tryGetOrCreateWildcard(const TypePtr& type) {
  auto maybe_mut_types = mapTypeToAliasTypeSetPtr(type);
  if (!maybe_mut_types) {
    // 如果无法映射给定类型到别名类型集合指针，返回空的 optional
    return c10::nullopt;
  }
  auto mut_type = toSingleType(*maybe_mut_types);
  auto existing_wildcard = wildcardIndex_.find(*mut_type);
  if (existing_wildcard != wildcardIndex_.end()) {
    // 如果存在相同别名类型的通配符元素，返回该元素的 optional
    return existing_wildcard->second;
  }

  // 创建新的通配符元素，并加入到 wildcardIndex_ 中
  auto wildcard_elem = memoryDAGBuilder_->makeFreshValue(nullptr);
  wildcardIndex_.emplace(*std::move(mut_type), wildcard_elem);
  if (maybe_mut_types->size() > 1) {
    // 如果别名类型集合有多个元素，将新创建的通配符元素关联到所有包含的类型中
    pointUnionTypeElementToAllContainedTypes(wildcard_elem, *maybe_mut_types);
  } else {
    // 如果别名类型集合只有一个元素，将该元素添加到新创建的通配符元素中
    addContainedTypesToFreshElement(wildcard_elem, *maybe_mut_types);
  }
  return wildcard_elem;
}

void AliasDb::pointUnionTypeElementToAllContainedTypes(
    Element* container_elem,
    const AliasTypeSet& mut_types) {
  // 遍历别名类型集合，为每个类型尝试获取或创建对应的通配符元素，并将指针关联到 `container_elem`
  for (const auto& mut_type : mut_types) {
    auto maybe_elem = tryGetOrCreateWildcard(mut_type);
    if (maybe_elem) {
      // 断言确保获取的通配符元素不同于 `container_elem`，并建立指针关系
      TORCH_INTERNAL_ASSERT(*maybe_elem != container_elem);
      memoryDAGBuilder_->makePointerTo(container_elem, *maybe_elem);
    }
  }
}

void AliasDb::addContainedTypesToFreshElement(
    Element* container_elem,
    const AliasTypeSet& mut_types) {
  // 遍历别名类型集合，为每个类型的包含类型尝试获取或创建对应的通配符元素，并将它们添加到 `container_elem` 的包含元素中
  for (const auto& mut_type : mut_types) {
    for (const auto& contained : mut_type->containedTypes()) {
      auto maybe_elem = tryGetOrCreateWildcard(contained);
      if (maybe_elem) {
        memoryDAGBuilder_->addToContainedElements(*maybe_elem, container_elem);
      }
    }
  }
}

// 在通配符索引中搜索与给定类型相对应的元素
// 如果找不到则返回 nullptr
Element* AliasDb::getWildcard(const TypePtr& type) const {
  auto maybe_mut_types = mapTypeToAliasTypeSetPtr(type);
  if (!maybe_mut_types) {
    // 如果无法映射给定类型到别名类型集合指针，返回空
    return {};
  }
  if (maybe_mut_types->size() > 1) {
    // 如果别名类型集合包含多个类型，创建一个联合类型并返回与之对应的通配符元素
    auto union_type = UnionType::create(*maybe_mut_types);
    auto it = wildcardIndex_.find(*union_type);
    if (it != wildcardIndex_.end()) {
      return it->second;
    }
  }
  // 如果别名类型集合只包含一个类型，直接返回与之对应的通配符元素
  auto it = wildcardIndex_.find(*maybe_mut_types->front());
  if (it != wildcardIndex_.end()) {
    return it->second;
  }
  return nullptr;
}
    // 在 wildcardIndex_ 中查找 union_type 对应的 <TypePtr, Element*> 键值对
    auto maybe_union_pair = wildcardIndex_.find(union_type);
    // 如果找到了对应的键值对
    if (maybe_union_pair != wildcardIndex_.end()) {
      // 返回该键值对的第二个元素，即 Element*，作为结果
      return (*maybe_union_pair).second;
    }
  } else {
    // 在 wildcardIndex_ 中查找 maybe_mut_types 的第一个元素对应的 <TypePtr, Element*> 键值对
    auto type_pair = wildcardIndex_.find((*maybe_mut_types)[0]);
    // 如果找到了对应的键值对
    if (type_pair != wildcardIndex_.end()) {
      // 返回该键值对的第二个元素，即 Element*，作为结果
      return type_pair->second;
    }
  }
  // 如果以上条件都未满足，则返回一个空值，表示未找到对应的结果
  return {};
}

// 结束命名空间 `torch::jit`

// 注册 `v` 作为通配符值。
std::optional<Element*> AliasDb::setWildcard(const Value* v) {
  // 尝试获取或创建与值类型相关的通配符元素
  std::optional<Element*> maybe_wildcardElement =
      tryGetOrCreateWildcard(v->type());
  if (!maybe_wildcardElement) {
    return c10::nullopt;
  }
  // 确保仍然为 `v` 创建相应的 Element，因为所有可变值都应有一个 Element
  getOrCreateElement(v);
  // 将 `v` 添加到通配符集合中
  wildcards_.insert(v);
  return *maybe_wildcardElement;
}

// 构建写入位置索引
void AliasDb::buildWrittenToLocationsIndex() {
  // 初始化返回的内存位置集合
  MemoryLocations ret;
  // 遍历写入索引中的每对键值对
  for (const auto& pr : *writeIndex_) {
    // 获取当前键对应的写入位置集合
    const auto& writtenLocs = pr.second;
    // 将当前写入位置集合合并到返回的集合中
    ret |= writtenLocs;
  }
  // 将合并后的结果赋值给写入位置索引
  writtenToLocationsIndex_ = ret;
}

// lint 检查函数
void Lint(const AliasDb* db) {
  // 初始化失败标志
  bool failed = false;

  // 初始化字符串流对象
  std::stringstream ss;
  // 对系统中的每个可变值执行以下操作
  for (const auto& v : db->graph_->all_values) {
    // 如果不是内部可变类型，则继续下一个循环
    if (!db->isMutableTypeInternal(v)) {
      continue;
    }
    // 在元素映射中查找当前值 `v`
    auto it = db->elementMap_.find(v);
    // 如果未找到，则设置失败标志并记录相关信息到字符串流中
    if (it == db->elementMap_.end()) {
      failed = true;
      ss << "Value %" << v->debugName() << " of type " << v->type()->repr_str()
         << " wasn't found in the element map.\n"
         << "It was defined in " << *v->node();
    }
  }
  // 使用内部断言确保没有失败情况，否则输出字符串流内容作为错误信息
  TORCH_INTERNAL_ASSERT(!failed, ss.str());

  // 以下是我们想要添加的两个检查，但在突变 API 更完整之前不能添加。
  // - aliasdb 中的每个可变值都属于图形中
  // - 所有容器值都有包含的元素
}

} // 命名空间结束：torch::jit
```