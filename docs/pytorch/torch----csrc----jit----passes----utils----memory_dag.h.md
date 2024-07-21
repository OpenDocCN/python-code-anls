# `.\pytorch\torch\csrc\jit\passes\utils\memory_dag.h`

```py
#pragma once

#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/sparse_bitset.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/csrc/Export.h>

// 使用压缩索引表示以加快比较速度
typedef c10::SparseBitVector<256> MemoryLocations;

// 命名空间 torch::jit 下的结构体 Value 的前置声明
struct Value;

// 别名类型定义，用于表示一组类型指针
using AliasTypeSet = std::vector<TypePtr>;

// `Element` 表示指向图中的一个顶点。它表示可能具有别名关系的任何内容，主要是 IR 中的 `Value`，也可能是通配符或容器内的类型（例如 `List[T]` 中的 `T`）
struct Element {
  // 构造函数，根据值和索引创建 Element 对象
  Element(const Value* value_, unsigned index_);
  // 通配符构造函数，根据索引创建 Element 对象
  explicit Element(unsigned index_);

  // 指向拥有的 DAG 中表示此元素的位向量的索引
  unsigned index;

  // 此元素 *可能* 指向的所有元素。由于控制流/复杂操作，可能会有多个可能的指向
  MemoryLocations pointsTo;

  // pointsTo 的反向引用
  MemoryLocations pointedFrom;

  // 元素可以包含其他元素（例如 List[Tensor]）
  MemoryLocations containedElements;

  // 此元素对应的值。如果此元素不表示一级值，则可能为空。这仅用于调试信息。
  std::unordered_set<const Value*> values;

 private:
  // 使 `from` 指向 `to`
  void makePointerTo(Element* from, Element* to);

  friend class MemoryDAG;
  
  // 我们通过缓存 `getMemoryLocations` 的结果来加速查询。
  // 如果缓存尚未填充，`nullopt` 表示为空。由于 `MemoryDAG` 是不可变的，这个缓存不需要失效。
  mutable std::optional<MemoryLocations> cachedMemoryLocations_;

  mutable std::optional<MemoryLocations> cachedAllContainedMemoryLocations_;
};

// 类 MemoryDAG
//
// 此类跟踪所有值的 "A points to B" 图。它被 AliasDb 使用以提供更高级别的 API。
//
// 我们维护一个 DAG，其中：
//   - 顶点（称为 "Element"）表示值和其他可能具有别名的实体（例如列表内的内容）
//   - 边表示 "points-to" 关系。
//
// DAG 中的叶子是不指向任何内容的实体，因此对应于唯一的 "memory location"。
//
/**
 * So, by traversing the "points-to" graph to the leaves, you can determine
 * which memory locations an element may point to.
 */
class TORCH_API MemoryDAG {
 public:
  /**
   * Constructs a MemoryDAG object with an index-to-element map.
   * This map represents elements in the "points-to" graph.
   * Deleting copy constructor to avoid confusion in Windows build.
   * Reference: https://stackoverflow.com/a/51033485/105137
   */
  explicit MemoryDAG(std::vector<std::unique_ptr<Element>> indexToElementMap)
      : indexToElementMap_(std::move(indexToElementMap)) {}

  // Explicitly deleted copy constructor and assignment operator

  /**
   * Returns the set of memory locations that a given Element might represent.
   */
  const MemoryLocations& getMemoryLocations(const Element* e) const;

  /**
   * Checks if two Elements `a` and `b` might potentially share a memory location.
   */
  bool mayAlias(const Element* a, const Element* b) const;

  /**
   * Checks if Element `a` holds reference to any memory that is stored in `b`, or vice versa.
   */
  bool mayContainAlias(const Element* a, const Element* b) const;

  /**
   * Checks if any Element in array `a` holds reference to any Element in array `b`.
   */
  bool mayContainAlias(const at::ArrayRef<Element*> a,
                      const at::ArrayRef<Element*> b) const;

  /**
   * Converts from the compressed index representation `x` to the corresponding Element pointer.
   */
  const Element* fromIndex(unsigned x) const;
  Element* fromIndex(unsigned x);

  /**
   * Collects all memory locations contained within the Element `elem`.
   */
  void collectAllContainedMemoryLocations(const Element* elem,
                                          MemoryLocations& cont) const;

  /**
   * Methods for mutating the MemoryDAG internals for efficiency reasons.
   * Use with caution as they affect cache consistency for memory locations.
   */

  /**
   * Efficiently adds wildcards to the MemoryDAG cache-awarely.
   */
  void setWildcards(const std::unordered_set<const Value*>& wildcards,
                    const ska::flat_hash_map<const Value*, Element*>& elementMap,
                    const std::function<Element*(const Value*)>& getWildcardElement);

  /**
   * Creates a fresh Element corresponding to the given Value `v`, bypassing safety checks.
   */
  Element* unsafeMakeFreshValue(const Value* v);

 private:
  /**
   * Retrieves all memory locations contained within the Element `elem`.
   */
  const MemoryLocations& getAllContainedMemoryLocations(const Element* elem) const;

  /**
   * Recursively collects all memory locations contained within the Element `elem`.
   */
  void collectAllContainedMemoryLocationsImpl(const Element* elem,
                                             MemoryLocations& cont) const;

  std::vector<std::unique_ptr<Element>> indexToElementMap_;
};

/**
 * Helper class for building up the points-to graph incrementally.
 *
 * Separates the building process to manage internal caching within MemoryDAG.
 */
// 定义一个名为 MemoryDAGBuilder 的类，用于构建内存的有向无环图（DAG）
class TORCH_API MemoryDAGBuilder {
 public:
  // 默认构造函数，使用默认参数初始化对象
  MemoryDAGBuilder() = default;

  // 删除复制构造函数，禁止对象的复制
  MemoryDAGBuilder(const MemoryDAGBuilder&) = delete;

  // 删除赋值运算符重载，禁止对象的赋值
  MemoryDAGBuilder& operator=(const MemoryDAGBuilder&) = delete;

  // 让指针 `from` 指向 `to` 所指向的元素
  void makePointerTo(Element* from, Element* to);

  // 将 `contained` 元素添加到 `container` 元素的包含元素列表中
  void addToContainedElements(Element* contained, Element* container);

  // 移动构造函数，创建并返回一个内存图 `MemoryDAG` 的唯一指针
  std::unique_ptr<MemoryDAG> createMemoryDAG() && {
    return std::make_unique<MemoryDAG>(std::move(indexToElementMap_));
  }

  // 创建一个新的元素（即不指向任何内容的元素），并返回其指针
  Element* makeFreshValue(const Value* v);

  // 声明 MemoryDAG 为友元类，以便访问私有成员
  friend MemoryDAG;

 private:
  // `MemoryDAGBuilder` 构建 `indexToElementMap_`，然后使用该映射构造 `MemoryDAG`
  std::vector<std::unique_ptr<Element>> indexToElementMap_;
};
} // namespace jit
} // namespace torch
```