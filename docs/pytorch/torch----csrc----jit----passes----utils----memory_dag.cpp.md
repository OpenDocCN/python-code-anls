# `.\pytorch\torch\csrc\jit\passes\utils\memory_dag.cpp`

```py
// 引入 Torch 库中的头文件，用于内存 DAG（Directed Acyclic Graph，有向无环图）的实现
#include <torch/csrc/jit/passes/utils/memory_dag.h>

// 引入 C10 库中的头文件，包括 flat_hash_map、算法和队列操作
#include <c10/util/flat_hash_map.h>
#include <algorithm>
#include <queue>

// Torch 命名空间开始
namespace torch {
namespace jit {
namespace {

// 创建一个指向关系，将 from 指向 to
void makePointerToImpl(Element* from, Element* to) {
  from->pointsTo.set(to->index);    // 设置 from 指向 to 的索引
  to->pointedFrom.set(from->index); // 设置 to 被 from 指向的索引
}

// 创建一个新的 Element，关联到给定的 Value（如果提供），并将其添加到 indexToElementMap_
Element* makeFreshValueImpl(
    const Value* v,
    std::vector<std::unique_ptr<Element>>& indexToElementMap_) {
  if (v == nullptr) {
    // 如果值为 nullptr，创建一个通配符元素，没有对应的值
    indexToElementMap_.emplace_back(
        std::make_unique<Element>(indexToElementMap_.size()));
    return indexToElementMap_.back().get();
  }
  indexToElementMap_.emplace_back(
      std::make_unique<Element>(v, indexToElementMap_.size()));
  return indexToElementMap_.back().get();
}
} // namespace

// Element 类的构造函数实现，关联到给定的 Value 和索引
Element::Element(const Value* value_, unsigned index_)
    : index(index_), values({value_}) {}

// Element 类的构造函数实现，仅关联到索引，没有关联到具体的 Value
Element::Element(unsigned index_) : index(index_), values({}) {}

// 返回给定索引处的 Element 对象的常量指针
const Element* MemoryDAG::fromIndex(unsigned x) const {
  TORCH_INTERNAL_ASSERT(x < indexToElementMap_.size());
  return indexToElementMap_[x].get();
}

// 返回给定索引处的 Element 对象的指针
Element* MemoryDAG::fromIndex(unsigned x) {
  TORCH_INTERNAL_ASSERT(x < indexToElementMap_.size());
  return indexToElementMap_[x].get();
}

// 检查两个 Element 是否可能有别名关系，根据它们的内存位置信息判断
bool MemoryDAG::mayAlias(const Element* a, const Element* b) const {
  const auto& aMemLoc = getMemoryLocations(a);
  const auto& bMemLoc = getMemoryLocations(b);

  return aMemLoc.intersects(bMemLoc); // 判断两者内存位置是否相交
}

// 检查两个 Element 是否可能包含别名关系，考虑它们的所有包含内存位置信息
bool MemoryDAG::mayContainAlias(const Element* a, const Element* b) const {
  return getAllContainedMemoryLocations(a).intersects(
      getAllContainedMemoryLocations(b)); // 判断两者所有包含的内存位置是否相交
}

// 获取给定 Element 的所有包含的内存位置信息
const MemoryLocations& MemoryDAG::getAllContainedMemoryLocations(
    const Element* elem) const {
  if (C10_UNLIKELY(!elem->cachedAllContainedMemoryLocations_.has_value())) {
    MemoryLocations cache;
    elem->cachedAllContainedMemoryLocations_ = MemoryLocations();
    collectAllContainedMemoryLocationsImpl(
        elem, *elem->cachedAllContainedMemoryLocations_);
  }
  return *elem->cachedAllContainedMemoryLocations_;
}

// 收集给定 Element 的所有包含的内存位置信息，并存储在 cont 中
void MemoryDAG::collectAllContainedMemoryLocations(
    const Element* elem,
    MemoryLocations& cont) const {
  // 如果已经对这个元素进行了递归操作，直接返回
  unsigned compIdx = elem->index;
  if (cont.test(compIdx)) {
    return;
  }

  if (C10_UNLIKELY(!elem->cachedAllContainedMemoryLocations_.has_value())) {
    MemoryLocations cache;
    collectAllContainedMemoryLocationsImpl(elem, cache);
    elem->cachedAllContainedMemoryLocations_ = std::move(cache);
  }
  cont |= *elem->cachedAllContainedMemoryLocations_;
}

// 实际执行收集给定 Element 的所有包含的内存位置信息的方法
void MemoryDAG::collectAllContainedMemoryLocationsImpl(
    const Element* elem,
    MemoryLocations& cont) const {
  unsigned compIdx = elem->index;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!cont.test(compIdx));
  cont.set(compIdx);

  // 遍历获取给定 Element 的所有内存位置信息
  for (const auto& mem_loc : getMemoryLocations(elem)) {


这段代码是一个内存 DAG 的实现，用于管理和分析程序中不同内存元素（Element）之间的指针关系和别名情况。
  // 从指定内存位置获取起始索引，然后收集所有包含的内存位置
  collectAllContainedMemoryLocations(fromIndex(mem_loc), cont);
}

// 遍历当前元素的所有包含元素
for (const auto& contained : elem->containedElements) {
  // 对每个包含的元素，从其索引获取起始位置，然后继续收集其包含的所有内存位置
  collectAllContainedMemoryLocations(fromIndex(contained), cont);
}
}

bool MemoryDAG::mayContainAlias(
    const Element* a,
    const at::ArrayRef<Element*> b) const {
  // 如果 b 是空的数组，直接返回 false
  if (b.empty()) {
    return false;
  }

  // 获取元素 a 的所有包含内存位置
  const auto& a_contained = getAllContainedMemoryLocations(a);

  // 检查数组 b 中是否有任何元素与 a 的内存位置有交集
  return std::any_of(b.begin(), b.end(), [this, &a_contained](Element* b_elem) {
    return a_contained.intersects(this->getAllContainedMemoryLocations(b_elem));
  });
}

bool MemoryDAG::mayContainAlias(
    const at::ArrayRef<Element*> a,
    const at::ArrayRef<Element*> b) const {
  // 如果 a 或者 b 是空的数组，直接返回 false
  if (a.empty() || b.empty()) {
    return false;
  }

  // 收集所有 a 中元素的包含内存位置
  MemoryLocations all_a_mlocs;
  for (const auto& elem : a) {
    collectAllContainedMemoryLocations(elem, all_a_mlocs);
  }

  // 收集所有 b 中元素的包含内存位置
  MemoryLocations all_b_mlocs;
  for (const auto& elem : b) {
    collectAllContainedMemoryLocations(elem, all_b_mlocs);
  }

  // 检查所有 a 的内存位置是否与所有 b 的内存位置有交集
  return all_a_mlocs.intersects(all_b_mlocs);
}

void MemoryDAGBuilder::makePointerTo(Element* from, Element* to) {
  // 调用底层实现方法建立从 from 到 to 的指针关系
  makePointerToImpl(from, to);
}

void MemoryDAGBuilder::addToContainedElements(
    Element* elem,
    Element* container) {
  // 断言 elem 不能直接包含自己，否则抛出异常
  TORCH_INTERNAL_ASSERT(
      elem != container, "Elements cannot contain themselves");
  // 将 elem 添加到 container 的包含元素集合中
  container->containedElements.set(elem->index);
}

// 为值 v 创建一个新的别名（即不指向任何值）
Element* MemoryDAGBuilder::makeFreshValue(const Value* v) {
  // 调用底层实现方法创建一个新的值，并加入到索引映射中
  return makeFreshValueImpl(v, indexToElementMap_);
}

// 此函数用于构建表示元素 e 的“别名集合”的位集（MemoryLocations 是一个类型别名为 c10::SparseBitVector）
const MemoryLocations& MemoryDAG::getMemoryLocations(const Element* e) const {
  // 关于缓存失效的注意事项：所有的变更应该通过 MemoryDAGBuilder 进行。因此，一旦我们通过构建器创建了不可变的 MemoryDAG，我们可以在这里缓存，而不必担心可能会失效。
  if (e->cachedMemoryLocations_) {
    return *e->cachedMemoryLocations_;
  }

  MemoryLocations ret;
  if (e->pointsTo.empty()) {
    // 基本情况：如果我们不指向任何东西，这个元素就是一个内存位置。返回其自身。
    ret.set(e->index);
  } else {
    // 否则，遍历所有指向当前元素的元素，并将它们的内存位置集合并到 ret 中
    for (auto el : e->pointsTo) {
      ret |= getMemoryLocations(fromIndex(el));
    }
  }

  // 缓存当前元素的内存位置集合并返回
  e->cachedMemoryLocations_ = std::move(ret);
  return *e->cachedMemoryLocations_;
}

void MemoryDAG::setWildcards(
    const std::unordered_set<const Value*>& wildcards,
    const ska::flat_hash_map<const Value*, Element*>& elementMap,
    const std::function<Element*(const Value*)>& getWildcardElement) {
  // 存储需要更新缓存的元素和其对应的内存位置
  std::unordered_map<Element*, MemoryLocations> cacheUpdates;
  // 如果一个元素被设置为通配符，那么它的所有内存位置必须指向通配符元素。
  for (const Value* v : wildcards) {
    auto wildcardElement = getWildcardElement(v);
    TORCH_INTERNAL_ASSERT(wildcardElement);

    // 获取元素 v 对应的内存位置集合，并更新缓存
    const MemoryLocations& pointeeSet = getMemoryLocations(elementMap.at(v));
    // 对于pointeeSet中的每个指向对象，进行处理
    for (const auto& pointee : pointeeSet) {
      // 获取pointee对象在索引中的位置
      auto from = this->fromIndex(pointee);
      // 避免出现通配符指向自身的循环
      if (from != wildcardElement) {
        // 将from指向wildcardElement的指针添加到数据结构中
        makePointerToImpl(from, wildcardElement);
      }
    }
    // 使用一个新的指针跟踪我们编辑过的内存位置
    // 将wildcardElement作为指针添加到cacheUpdates中的内存位置集合中
    cacheUpdates[wildcardElement] |= pointeeSet;
  }

  // 在原地更新缓存
  // 我们利用了只编辑内存位置这一事实
  //
  // 假设我们添加了一个指针从`MemoryLocationFoo -> WildcardBar`。
  // 对于每个元素，如果缓存包含`MemoryLocationFoo`，则必须将`WildcardBar`也添加到其中。
  for (const std::unique_ptr<Element>& e : this->indexToElementMap_) {
    // 重置e的缓存所有包含的内存位置集合
    e->cachedAllContainedMemoryLocations_.reset();
    if (e->values.empty()) {
      // 这个元素是一个通配符元素，可以跳过
      continue;
    }

    // 获取e的第一个值对应的通配符元素
    auto wildcardElement = getWildcardElement(*(e->values.begin()));
    if (!wildcardElement) {
      // 这个值不是通配符，跳过
      continue;
    }
    // 查找cacheUpdates中是否存在wildcardElement
    auto it = cacheUpdates.find(wildcardElement);
    if (it == cacheUpdates.end()) {
      // 没有重写任何MemoryLocation指向这个元素，跳过
      continue;
    }
    // 如果这个元素包含一个编辑过的内存位置，则更新缓存以包含指向的通配符元素
    if (getMemoryLocations(e.get()).intersects(it->second)) {
      e->cachedMemoryLocations_->set(wildcardElement->index);
    }
  }
}

Element* MemoryDAG::unsafeMakeFreshValue(const Value* v) {
  // 调用 MemoryDAG 类中的 makeFreshValueImpl 方法，使用当前对象的 indexToElementMap_
  return makeFreshValueImpl(v, indexToElementMap_);
}
} // namespace jit
} // namespace torch


这段代码的注释如上所示，解释了每行代码的作用和功能。
```