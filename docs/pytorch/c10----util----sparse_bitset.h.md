# `.\pytorch\c10\util\sparse_bitset.h`

```
  // 元素的索引，表示第一个位的起始位置
  unsigned ElementIndex;
  // 位向量元素的位字，每个字有 sizeof(BitWord) * CHAR_BIT 位
  std::array<BitWord, BITWORDS_PER_ELEMENT> Bits{};

  // 默认构造函数，将 ElementIndex 初始化为 ~0U 表示未定义状态
  SparseBitVectorElement() : ElementIndex(~0U) {}

public:
  // 显式构造函数，指定元素索引 Idx
  explicit SparseBitVectorElement(unsigned Idx) : ElementIndex(Idx) {}

  // 比较操作符，比较两个 SparseBitVectorElement 是否相等
  bool operator==(const SparseBitVectorElement& RHS) const {
    // 如果索引不同，直接返回 false
    if (ElementIndex != RHS.ElementIndex)
      return false;
    // 逐字比较 Bits 数组的内容
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      if (Bits[i] != RHS.Bits[i])
        return false;
    // 全部相等则返回 true
    return true;
  }

  // 不相等比较操作符，利用相等操作符的结果取反实现
  bool operator!=(const SparseBitVectorElement& RHS) const {
    return !(*this == RHS);
  }

  // 返回位于索引 Idx 处的位字内容
  BitWord word(unsigned Idx) const {
    // 断言确保 Idx 在有效范围内
    assert(Idx < BITWORDS_PER_ELEMENT);
    return Bits[Idx];
  }

  // 返回元素的索引值
  unsigned index() const {
    return ElementIndex;
  }

  // 判断当前元素是否为空，即 ElementIndex 是否为未定义状态
  bool empty() const {
    // 遍历每个位图元素中的位字，检查是否有非零的位，如果有则返回false
    // 如果所有位都是零，则返回true
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      if (Bits[i])
        return false;
    // 所有位都是零，返回true
    return true;
  }

  // 设置指定索引处的位为1
  void set(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] |= 1L << (Idx % BITWORD_SIZE);
  }

  // 检查并设置指定索引处的位，如果原来为0则设置为1，并返回true；如果原来已经是1则返回false
  bool test_and_set(unsigned Idx) {
    // 先获取原来的值
    bool old = test(Idx);
    // 如果原来是0，则设置为1
    if (!old) {
      set(Idx);
      return true;
    }
    // 原来已经是1，返回false
    return false;
  }

  // 将指定索引处的位重置为0
  void reset(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] &= ~(1L << (Idx % BITWORD_SIZE));
  }

  // 检查指定索引处的位，返回其值（0或1）
  bool test(unsigned Idx) const {
    return Bits[Idx / BITWORD_SIZE] & (1L << (Idx % BITWORD_SIZE));
  }

  // 计算位向量中设置为1的位数目
  size_type count() const {
    unsigned NumBits = 0;
    // 遍历每个位图元素，调用llvm库函数统计每个元素中设置为1的位数
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      NumBits += llvm::countPopulation(Bits[i]);
    return NumBits;
  }

  /// find_first - 返回第一个设置为1的位的索引。
  int find_first() const {
    // 遍历每个位图元素，找到第一个非零元素，返回其第一个设置为1的位的索引
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      if (Bits[i] != 0)
        return i * BITWORD_SIZE + llvm::countTrailingZeros(Bits[i]);
    // 如果所有元素都是空的，则抛出异常
    throw std::runtime_error("Illegal empty element");
  }

  /// find_last - 返回最后一个设置为1的位的索引。
  int find_last() const {
    // 从最后一个位图元素开始向前遍历，找到第一个非零元素，返回其最后一个设置为1的位的索引
    for (unsigned I = 0; I < BITWORDS_PER_ELEMENT; ++I) {
      unsigned Idx = BITWORDS_PER_ELEMENT - I - 1;
      if (Bits[Idx] != 0)
        return Idx * BITWORD_SIZE + BITWORD_SIZE -
            llvm::countLeadingZeros(Bits[Idx]);
    }
    // 如果所有元素都是空的，则抛出异常
    throw std::runtime_error("Illegal empty element");
  }

  /// find_next - 返回从“Curr”位开始的下一个设置为1的位的索引。
  /// 如果未找到下一个设置为1的位，则返回-1。
  int find_next(unsigned Curr) const {
    // 如果当前索引已经超过位向量的长度，直接返回-1
    if (Curr >= BITS_PER_ELEMENT)
      return -1;

    unsigned WordPos = Curr / BITWORD_SIZE;
    unsigned BitPos = Curr % BITWORD_SIZE;
    BitWord Copy = Bits[WordPos];
    assert(
        WordPos <= BITWORDS_PER_ELEMENT && "Word Position outside of element");

    // 屏蔽掉前面的位
    Copy &= ~0UL << BitPos;

    // 如果屏蔽后的值不为0，返回其第一个设置为1的位的索引
    if (Copy != 0)
      return WordPos * BITWORD_SIZE + llvm::countTrailingZeros(Copy);

    // 否则检查后续的元素，找到第一个非零元素，返回其第一个设置为1的位的索引
    for (unsigned i = WordPos + 1; i < BITWORDS_PER_ELEMENT; ++i)
      if (Bits[i] != 0)
        return i * BITWORD_SIZE + llvm::countTrailingZeros(Bits[i]);
    // 如果未找到任何设置为1的位，则返回-1
    return -1;
  }

  // 将该元素与RHS进行并操作，如果这个操作改变了该元素，则返回true。
  bool unionWith(const SparseBitVectorElement& RHS) {
    bool changed = false;
    // 遍历每个位图元素，将对应位置的位进行或操作
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
      // 如果已经有变化，则将old设为0，表示不再需要比较
      BitWord old = changed ? 0 : Bits[i];

      // 执行并操作
      Bits[i] |= RHS.Bits[i];
      // 如果在第一次发生变化，则记录变化
      if (!changed && old != Bits[i])
        changed = true;
    }
    // 返回是否发生了变化
    return changed;
  }

  // 如果该元素与RHS有任何公共的位（即同时为1），则返回true
  bool intersects(const SparseBitVectorElement& RHS) const {
    // 遍历每个位图元素，检查是否有任何位同时为1
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
      if (RHS.Bits[i] & Bits[i])
        return true;
    }
    // 如果没有任何位同时为1，则返回false
    return false;
  }
  // 返回布尔值 false
  return false;
}

// 与另一个稀疏位向量元素（RHS）求交集，并返回是否有变化。
// 如果该元素变成全零位，则将 BecameZero 设为 true。
bool intersectWith(const SparseBitVectorElement& RHS, bool& BecameZero) {
  bool changed = false;  // 记录是否有变化
  bool allzero = true;   // 记录是否全部为零位

  for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
    BitWord old = changed ? 0 : Bits[i];  // 如果已经变化，则 old 设为 0

    Bits[i] &= RHS.Bits[i];  // 将当前位向量与 RHS 的对应位向量取交集
    if (Bits[i] != 0)
      allzero = false;  // 如果当前位向量不为零，则 allzero 设为 false

    if (!changed && old != Bits[i])
      changed = true;  // 如果未变化且 old 与 Bits[i] 不同，则设为变化
  }
  BecameZero = allzero;  // 设置是否变成全零位的状态
  return changed;  // 返回是否有变化
}

// 与 RHS 的补集求交集，并返回是否有变化。
// 如果该元素变成全零位，则将 BecameZero 设为 true。
bool intersectWithComplement(
    const SparseBitVectorElement& RHS,
    bool& BecameZero) {
  bool changed = false;  // 记录是否有变化
  bool allzero = true;   // 记录是否全部为零位

  for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
    BitWord old = changed ? 0 : Bits[i];  // 如果已经变化，则 old 设为 0

    Bits[i] &= ~RHS.Bits[i];  // 将当前位向量与 RHS 的对应位向量的补集取交集
    if (Bits[i] != 0)
      allzero = false;  // 如果当前位向量不为零，则 allzero 设为 false

    if (!changed && old != Bits[i])
      changed = true;  // 如果未变化且 old 与 Bits[i] 不同，则设为变化
  }
  BecameZero = allzero;  // 设置是否变成全零位的状态
  return changed;  // 返回是否有变化
}

// 三参数版本的 intersectWithComplement 方法，计算 RHS1 & ~RHS2 的交集，并存入当前元素中。
// 如果该元素变成全零位，则将 BecameZero 设为 true。
void intersectWithComplement(
    const SparseBitVectorElement& RHS1,
    const SparseBitVectorElement& RHS2,
    bool& BecameZero) {
  bool allzero = true;  // 记录是否全部为零位

  for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
    Bits[i] = RHS1.Bits[i] & ~RHS2.Bits[i];  // 计算 RHS1 & ~RHS2 的交集并存入 Bits 中
    if (Bits[i] != 0)
      allzero = false;  // 如果当前位向量不为零，则 allzero 设为 false
  }
  BecameZero = allzero;  // 设置是否变成全零位的状态
}
};

// SparseBitVector 类模板，用于表示稀疏位向量
template <unsigned ElementSize = 128>
class SparseBitVector {
  // 使用 std::list 存储 SparseBitVectorElement 元素
  using ElementList = std::list<SparseBitVectorElement<ElementSize>>;
  // ElementList 迭代器类型定义
  using ElementListIter = typename ElementList::iterator;
  using ElementListConstIter = typename ElementList::const_iterator;
  // 定义位元素的大小
  enum { BITWORD_SIZE = SparseBitVectorElement<ElementSize>::BITWORD_SIZE };

  // 存储 SparseBitVectorElement 的列表
  ElementList Elements;
  // 可变的迭代器，指向当前元素。这对外部状态没有可见的影响，仅用于提高在相似索引下测试/修改位的性能。
  mutable ElementListIter CurrElementIter;

  // 类似 std::lower_bound 的函数，但我们从当前位置进行线性搜索。
  ElementListIter FindLowerBoundImpl(unsigned ElementIndex) const {
    // 缓存一个非 const 迭代器，因此需要使用 const_cast 来获取 begin/end，在 this 是 const 的情况下。为避免代码重复，
    // 该函数中 this 总是 const 的，我们在 FindLowerBound 和 FindLowerBoundConst 中处理 const 和非 const 的差异。
    ElementListIter Begin =
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<SparseBitVector<ElementSize>*>(this)->Elements.begin();
    ElementListIter End =
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<SparseBitVector<ElementSize>*>(this)->Elements.end();

    // 如果 Elements 为空，返回 Begin，并将 CurrElementIter 设置为 Begin。
    if (Elements.empty()) {
      CurrElementIter = Begin;
      return CurrElementIter;
    }

    // 确保当前迭代器有效。
    if (CurrElementIter == End)
      --CurrElementIter;

    // 从当前迭代器开始搜索，根据元素索引是向前还是向后进行搜索。
    ElementListIter ElementIter = CurrElementIter;
    if (CurrElementIter->index() == ElementIndex) {
      return ElementIter;
    } else if (CurrElementIter->index() > ElementIndex) {
      while (ElementIter != Begin && ElementIter->index() > ElementIndex)
        --ElementIter;
    } else {
      while (ElementIter != End && ElementIter->index() < ElementIndex)
        ++ElementIter;
    }
    // 更新 CurrElementIter 并返回搜索到的元素迭代器。
    CurrElementIter = ElementIter;
    return ElementIter;
  }

  // 返回 ElementIndex 对应的常量迭代器
  ElementListConstIter FindLowerBoundConst(unsigned ElementIndex) const {
    return FindLowerBoundImpl(ElementIndex);
  }

  // 返回 ElementIndex 对应的迭代器
  ElementListIter FindLowerBound(unsigned ElementIndex) {
    return FindLowerBoundImpl(ElementIndex);
  }

  // SparseBitVectorIterator 类，用于遍历位图中设置为 1 的位。为了效率，这个迭代器看起来比实际上更加复杂。
  class SparseBitVectorIterator {
   private:
    // 是否已经到达末尾
    bool AtEnd{false};

    // 指向 SparseBitVector 的指针
    const SparseBitVector<ElementSize>* BitVector = nullptr;

    // 当前位图中的元素迭代器
    ElementListConstIter Iter;

    // 当前位号
    unsigned BitNumber{0};

    // 当前元素中的当前字数
    unsigned WordNumber{0};

    // ...
    // 其余成员变量及函数实现省略，因为不影响对当前代码理解。
    # 当前位数对应的字数
    unsigned WordNumber{0};

    # 元素中当前位的位掩码
    typename SparseBitVectorElement<ElementSize>::BitWord Bits{0};

    # 将迭代器移动到位图中第一个非零位
    void AdvanceToFirstNonZero() {
      # 如果已经到达末尾，则直接返回
      if (AtEnd)
        return;
      # 如果位向量为空，则标记为结束
      if (BitVector->Elements.empty()) {
        AtEnd = true;
        return;
      }
      # 设置迭代器指向第一个元素
      Iter = BitVector->Elements.begin();
      # 计算位号
      BitNumber = Iter->index() * ElementSize;
      # 查找第一个非零位的位置
      unsigned BitPos = Iter->find_first();
      BitNumber += BitPos;
      # 计算字号
      WordNumber = (BitNumber % ElementSize) / BITWORD_SIZE;
      # 获取当前字中的位数据
      Bits = Iter->word(WordNumber);
      Bits >>= BitPos % BITWORD_SIZE;
    }

    # 将迭代器移动到下一个非零位
    void AdvanceToNextNonZero() {
      # 如果已经到达末尾，则直接返回
      if (AtEnd)
        return;

      # 循环直到找到下一个非零位
      while (Bits && !(Bits & 1)) {
        Bits >>= 1;
        BitNumber += 1;
      }

      # 如果当前字中的位已经用完
      if (!Bits) {
        # 查找下一个设置位的位置
        int NextSetBitNumber = Iter->find_next(BitNumber % ElementSize);
        # 如果在当前元素中找不到设置位，或者已经到达元素末尾，移动到下一个元素
        if (NextSetBitNumber == -1 || (BitNumber % ElementSize == 0)) {
          ++Iter;
          WordNumber = 0;

          # 如果没有下一个元素了，标记为结束
          if (Iter == BitVector->Elements.end()) {
            AtEnd = true;
            return;
          }
          # 设置下一个非零位的准备
          BitNumber = Iter->index() * ElementSize;
          NextSetBitNumber = Iter->find_first();
          BitNumber += NextSetBitNumber;
          WordNumber = (BitNumber % ElementSize) / BITWORD_SIZE;
          Bits = Iter->word(WordNumber);
          Bits >>= NextSetBitNumber % BITWORD_SIZE;
        } else {
          # 找到下一个非零位所在的字号
          WordNumber = (NextSetBitNumber % ElementSize) / BITWORD_SIZE;
          Bits = Iter->word(WordNumber);
          Bits >>= NextSetBitNumber % BITWORD_SIZE;
          BitNumber = Iter->index() * ElementSize;
          BitNumber += NextSetBitNumber;
        }
      }
    }

   public:
    # 默认构造函数
    SparseBitVectorIterator() = default;

    # 构造函数，设置迭代器为位图的起始或结束位置
    SparseBitVectorIterator(
        const SparseBitVector<ElementSize>* RHS,
        bool end = false)
        : AtEnd(end),
          BitVector(RHS),
          Iter(BitVector->Elements.begin()),
          WordNumber(~0) {
      # 移动迭代器到第一个非零位
      AdvanceToFirstNonZero();
    }

    # 前置递增运算符重载
    inline SparseBitVectorIterator& operator++() {
      ++BitNumber;
      Bits >>= 1;
      # 移动到下一个非零位
      AdvanceToNextNonZero();
      return *this;
    }

    # 后置递增运算符重载
    inline SparseBitVectorIterator operator++(int) {
      SparseBitVectorIterator tmp = *this;
      ++*this;
      return tmp;
    }

    # 返回当前设置位的位号
    unsigned operator*() const {
      return BitNumber;
    }
    // 定义 SparseBitVectorIterator 结构体，实现比较操作符 ==
    bool operator==(const SparseBitVectorIterator& RHS) const {
      // 如果两者都已经到达末尾，忽略其它字段的比较
      if (AtEnd && RHS.AtEnd)
        return true;
      // 否则，它们相等如果它们具有相同的位数和位图
      return AtEnd == RHS.AtEnd && RHS.BitNumber == BitNumber;
    }

    // 实现比较操作符 !=
    bool operator!=(const SparseBitVectorIterator& RHS) const {
      return !(*this == RHS);
    }
  };

 public:
  using iterator = SparseBitVectorIterator;

  // SparseBitVector 类的默认构造函数
  SparseBitVector() : Elements(), CurrElementIter(Elements.begin()) {}

  // SparseBitVector 类的拷贝构造函数
  SparseBitVector(const SparseBitVector& RHS)
      : Elements(RHS.Elements), CurrElementIter(Elements.begin()) {}

  // SparseBitVector 类的移动构造函数
  SparseBitVector(SparseBitVector&& RHS) noexcept
      : Elements(std::move(RHS.Elements)), CurrElementIter(Elements.begin()) {}

  // 清空 SparseBitVector 对象
  void clear() {
    Elements.clear();
  }

  // SparseBitVector 类的赋值操作符重载
  SparseBitVector& operator=(const SparseBitVector& RHS) {
    if (this == &RHS)
      return *this;

    Elements = RHS.Elements;
    CurrElementIter = Elements.begin();
    return *this;
  }

  // SparseBitVector 类的移动赋值操作符重载
  SparseBitVector& operator=(SparseBitVector&& RHS) noexcept {
    Elements = std::move(RHS.Elements);
    CurrElementIter = Elements.begin();
    return *this;
  }

  // 在位图中测试、重置和设置位
  bool test(unsigned Idx) const {
    if (Elements.empty())
      return false;

    unsigned ElementIndex = Idx / ElementSize;
    ElementListConstIter ElementIter = FindLowerBoundConst(ElementIndex);

    // 如果找不到应包含此位的元素，则无需进一步操作
    if (ElementIter == Elements.end() || ElementIter->index() != ElementIndex)
      return false;
    return ElementIter->test(Idx % ElementSize);
  }

  // 重置位图中的位
  void reset(unsigned Idx) {
    if (Elements.empty())
      return;

    unsigned ElementIndex = Idx / ElementSize;
    ElementListIter ElementIter = FindLowerBound(ElementIndex);

    // 如果找不到应包含此位的元素，则无需进一步操作
    if (ElementIter == Elements.end() || ElementIter->index() != ElementIndex)
      return;
    ElementIter->reset(Idx % ElementSize);

    // 当元素被清空时，删除它
    if (ElementIter->empty()) {
      ++CurrElementIter;
      Elements.erase(ElementIter);
    }
  }

  // 设置位图中的位
  void set(unsigned Idx) {
    unsigned ElementIndex = Idx / ElementSize;
    ElementListIter ElementIter;
    if (Elements.empty()) {
      // 如果元素列表为空，则在末尾插入新元素
      ElementIter = Elements.emplace(Elements.end(), ElementIndex);
  } else {
    // 找到 ElementIndex 的下界迭代器
    ElementIter = FindLowerBound(ElementIndex);

    // 如果 ElementIter 到达 Elements 的末尾或者当前迭代器指向的元素索引不等于 ElementIndex
    if (ElementIter == Elements.end() ||
        ElementIter->index() != ElementIndex) {
      // 可能已经到达 SparseBitVector 的开头，需要在当前元素之后插入新元素
      // 这要求将当前迭代器向前移动一个位置，因为 insert 是在当前迭代器之前插入的
      if (ElementIter != Elements.end() &&
          ElementIter->index() < ElementIndex)
        ++ElementIter;
      
      // 在 ElementIter 处插入新的元素 ElementIndex
      ElementIter = Elements.emplace(ElementIter, ElementIndex);
    }
  }
  // 更新当前元素迭代器为 ElementIter
  CurrElementIter = ElementIter;

  // 将 ElementIter 所指元素的第 Idx % ElementSize 位设置为 1
  ElementIter->set(Idx % ElementSize);
}

// 检测并设置指定位 Idx，返回旧值
bool test_and_set(unsigned Idx) {
  bool old = test(Idx);
  // 如果旧值为 false，则设置指定位 Idx
  if (!old) {
    set(Idx);
    return true;
  }
  return false;
}

// 判断两个 SparseBitVector 是否不相等
bool operator!=(const SparseBitVector& RHS) const {
  return !(*this == RHS);
}

// 判断两个 SparseBitVector 是否相等
bool operator==(const SparseBitVector& RHS) const {
  // 分别使用迭代器 Iter1 和 Iter2 遍历两个 SparseBitVector 的元素列表，比较是否完全相同
  ElementListConstIter Iter1 = Elements.begin();
  ElementListConstIter Iter2 = RHS.Elements.begin();

  for (; Iter1 != Elements.end() && Iter2 != RHS.Elements.end();
       ++Iter1, ++Iter2) {
    // 如果发现元素不相等，返回 false
    if (*Iter1 != *Iter2)
      return false;
  }
  // 如果两个列表都遍历完毕，则返回 true，表示两个 SparseBitVector 相等
  return Iter1 == Elements.end() && Iter2 == RHS.Elements.end();
}

// 将当前 SparseBitVector 与 RHS 合并，并返回是否有修改
bool operator|=(const SparseBitVector& RHS) {
  // 如果两个对象相同，直接返回 false
  if (this == &RHS)
    return false;

  // 如果当前对象为空，直接赋值为 RHS，并返回 true
  if (empty()) {
    *this = RHS;
    return true;
  }

  bool changed = false;
  ElementListIter Iter1 = Elements.begin();
  ElementListConstIter Iter2 = RHS.Elements.begin();

  // 如果 RHS 为空，返回 false
  if (RHS.Elements.empty())
    return false;

  // 合并两个 SparseBitVector 的元素列表
  while (Iter2 != RHS.Elements.end()) {
    if (Iter1 == Elements.end() || Iter1->index() > Iter2->index()) {
      // 在 Iter1 处插入 Iter2 指向的元素
      Elements.insert(Iter1, *Iter2);
      ++Iter2;
      changed = true;
    } else if (Iter1->index() == Iter2->index()) {
      // 对 Iter1 指向的元素与 Iter2 指向的元素进行并集操作
      changed |= Iter1->unionWith(*Iter2);
      ++Iter1;
      ++Iter2;
    } else {
      ++Iter1;
    }
  }
  // 更新当前元素迭代器为 Elements 的起始位置
  CurrElementIter = Elements.begin();
  return changed;
}

// 将当前 SparseBitVector 与 RHS 进行交集操作，并返回是否有修改
bool operator-=(const SparseBitVector& RHS) {
  return intersectWithComplement(RHS);
}

// 将当前 SparseBitVector 与 RHS 进行交集操作，并返回是否有修改
bool operator&=(const SparseBitVector& RHS) {
  // 如果两个对象相同，直接返回 false
  if (this == &RHS)
    return false;

  bool changed = false;
  ElementListIter Iter1 = Elements.begin();
  ElementListConstIter Iter2 = RHS.Elements.begin();

  // 检查两个位图是否都为空
  if (Elements.empty() && RHS.Elements.empty())
    return false;

  // 遍历两个 SparseBitVector，进行交集操作并删除不需要的元素
    while (Iter2 != RHS.Elements.end()) {
      // 如果 RHS 元素迭代器 Iter2 没有到达末尾
      if (Iter1 == Elements.end()) {
        // 如果当前对象的元素迭代器 Iter1 已经到达末尾
        CurrElementIter = Elements.begin();
        return changed;
      }

      // 如果当前元素的索引大于 RHS 元素的索引
      if (Iter1->index() > Iter2->index()) {
        ++Iter2;
      } else if (Iter1->index() == Iter2->index()) {
        // 如果当前元素的索引等于 RHS 元素的索引
        bool BecameZero = false;
        // 调用当前元素的 intersectWith 方法与 RHS 元素进行交集操作
        changed |= Iter1->intersectWith(*Iter2, BecameZero);
        // 如果当前元素变为零
        if (BecameZero) {
          ElementListIter IterTmp = Iter1;
          ++Iter1;
          // 从当前对象的元素列表中删除该元素
          Elements.erase(IterTmp);
        } else {
          ++Iter1;
        }
        ++Iter2;
      } else {
        // 如果当前元素的索引小于 RHS 元素的索引
        ElementListIter IterTmp = Iter1;
        ++Iter1;
        // 从当前对象的元素列表中删除该元素
        Elements.erase(IterTmp);
        changed = true;
      }
    }
    // 如果当前对象还有剩余元素，将它们从列表中删除
    if (Iter1 != Elements.end()) {
      Elements.erase(Iter1, Elements.end());
      changed = true;
    }
    // 设置当前元素迭代器为当前对象元素列表的起始位置
    CurrElementIter = Elements.begin();
    // 返回表示是否发生改变的布尔值
    return changed;
  }

  // 与 RHS 的补集相交，并返回是否发生了改变
  bool intersectWithComplement(const SparseBitVector& RHS) {
    // 如果当前对象与 RHS 是同一个对象
    if (this == &RHS) {
      // 如果当前对象不为空，清空当前对象并返回 true
      if (!empty()) {
        clear();
        return true;
      }
      // 否则返回 false
      return false;
    }

    bool changed = false;
    // 获取当前对象的元素迭代器
    ElementListIter Iter1 = Elements.begin();
    // 获取 RHS 的常量元素迭代器
    ElementListConstIter Iter2 = RHS.Elements.begin();

    // 如果当前对象或 RHS 的元素列表为空，则无需进一步操作
    if (Elements.empty() || RHS.Elements.empty())
      return false;

    // 循环遍历并求交集，必要时删除元素
    while (Iter2 != RHS.Elements.end()) {
      // 如果 RHS 元素迭代器 Iter2 没有到达末尾
      if (Iter1 == Elements.end()) {
        // 如果当前对象的元素迭代器 Iter1 已经到达末尾
        CurrElementIter = Elements.begin();
        return changed;
      }

      // 如果当前元素的索引大于 RHS 元素的索引
      if (Iter1->index() > Iter2->index()) {
        ++Iter2;
      } else if (Iter1->index() == Iter2->index()) {
        // 如果当前元素的索引等于 RHS 元素的索引
        bool BecameZero = false;
        // 调用当前元素的 intersectWithComplement 方法与 RHS 元素的补集进行交集操作
        changed |= Iter1->intersectWithComplement(*Iter2, BecameZero);
        // 如果当前元素变为零
        if (BecameZero) {
          ElementListIter IterTmp = Iter1;
          ++Iter1;
          // 从当前对象的元素列表中删除该元素
          Elements.erase(IterTmp);
        } else {
          ++Iter1;
        }
        ++Iter2;
      } else {
        // 如果当前元素的索引小于 RHS 元素的索引
        ++Iter1;
      }
    }
    // 设置当前元素迭代器为当前对象元素列表的起始位置
    CurrElementIter = Elements.begin();
    // 返回表示是否发生改变的布尔值
    return changed;
  }

  // 与 RHS 的补集相交，并返回是否发生了改变
  bool intersectWithComplement(const SparseBitVector<ElementSize>* RHS) const {
    // 调用重载函数，传入解引用后的 RHS 指针
    return intersectWithComplement(*RHS);
  }

  // 三参数版本的 intersectWithComplement
  // 将 RHS1 & ~RHS2 的结果存储到当前位图中
  void intersectWithComplement(
      const SparseBitVector<ElementSize>& RHS1,
      const SparseBitVector<ElementSize>& RHS2) {
    // 如果当前对象与 RHS1 是同一个对象
    if (this == &RHS1) {
      // 调用二参数版本的 intersectWithComplement 方法，传入 RHS2
      intersectWithComplement(RHS2);
      return;
    } else if (this == &RHS2) {
      // 如果当前对象与 RHS2 是同一个对象
      // 复制 RHS2 并调用二参数版本的 intersectWithComplement 方法，传入 RHS1 和复制后的 RHS2
      SparseBitVector RHS2Copy(RHS2);
      intersectWithComplement(RHS1, RHS2Copy);
      return;
    }

    // 清空当前对象的元素列表
    Elements.clear();
    // 设置当前元素迭代器为当前对象元素列表的起始位置
    CurrElementIter = Elements.begin();
    // 获取 RHS1 和 RHS2 的常量元素迭代器
    ElementListConstIter Iter1 = RHS1.Elements.begin();
    ElementListConstIter Iter2 = RHS2.Elements.begin();

    // 如果 RHS1 的元素列表为空，则无需进一步操作
    // 如果 RHS2 的元素列表为空，仍需复制 RHS1 的元素到当前对象
    // 如果 RHS1 的元素为空，则直接返回，无需进行任何操作。
    if (RHS1.Elements.empty())
      return;

    // 循环遍历 RHS2 的元素，逐步进行交集操作，并在必要时删除元素。
    while (Iter2 != RHS2.Elements.end()) {
      // 如果 Iter1 已经遍历完了 RHS1 的所有元素，则退出循环。
      if (Iter1 == RHS1.Elements.end())
        return;

      // 如果 Iter1 的索引大于 Iter2 的索引，则移动 Iter2 到下一个元素。
      if (Iter1->index() > Iter2->index()) {
        ++Iter2;
      } else if (Iter1->index() == Iter2->index()) {
        // 如果 Iter1 和 Iter2 的索引相同，则进行交集运算，并根据需要删除元素。
        bool BecameZero = false;
        Elements.emplace_back(Iter1->index());
        Elements.back().intersectWithComplement(*Iter1, *Iter2, BecameZero);
        if (BecameZero)
          Elements.pop_back();
        ++Iter1;
        ++Iter2;
      } else {
        // 否则，将 Iter1 的当前元素添加到结果集中，并移动 Iter1 到下一个元素。
        Elements.push_back(*Iter1++);
      }
    }

    // 将剩余的 Iter1 元素复制到结果集中。
    std::copy(Iter1, RHS1.Elements.end(), std::back_inserter(Elements));
  }

  // 使用指针版本的 intersectWithComplement 函数进行交集运算
  void intersectWithComplement(
      const SparseBitVector<ElementSize>* RHS1,
      const SparseBitVector<ElementSize>* RHS2) {
    intersectWithComplement(*RHS1, *RHS2);
  }

  // 使用引用版本的 intersects 函数检查是否存在交集
  bool intersects(const SparseBitVector<ElementSize>* RHS) const {
    return intersects(*RHS);
  }

  // 返回 true 如果当前 SparseBitVector 与 RHS 存在交集
  bool intersects(const SparseBitVector<ElementSize>& RHS) const {
    ElementListConstIter Iter1 = Elements.begin();
    ElementListConstIter Iter2 = RHS.Elements.begin();

    // 如果两个位图都为空，则返回 false
    if (Elements.empty() && RHS.Elements.empty())
      return false;

    // 循环遍历，检查是否存在公共位
    while (Iter2 != RHS.Elements.end()) {
      // 如果 Iter1 已经遍历完当前 SparseBitVector 的所有元素，则返回 false
      if (Iter1 == Elements.end())
        return false;

      // 如果 Iter1 的索引大于 Iter2 的索引，则移动 Iter2 到下一个元素。
      if (Iter1->index() > Iter2->index()) {
        ++Iter2;
      } else if (Iter1->index() == Iter2->index()) {
        // 如果 Iter1 和 Iter2 的索引相同，则检查它们是否有交集，有则返回 true。
        if (Iter1->intersects(*Iter2))
          return true;
        ++Iter1;
        ++Iter2;
      } else {
        // 否则，移动 Iter1 到下一个元素。
        ++Iter1;
      }
    }
    return false;
  }

  // 返回 true 如果当前 SparseBitVector 包含 RHS 中的所有位
  bool contains(const SparseBitVector<ElementSize>& RHS) const {
    // 创建当前 SparseBitVector 的副本，并求交集
    SparseBitVector<ElementSize> Result(*this);
    Result &= RHS;
    // 比较求得的交集与 RHS 是否相等，相等则返回 true。
    return (Result == RHS);
  }

  // 返回位图中第一个设置的位。如果没有设置的位，则返回 -1。
  int find_first() const {
    if (Elements.empty())
      return -1;
    // 获取第一个设置的位的索引，并返回全局索引。
    const SparseBitVectorElement<ElementSize>& First = *(Elements.begin());
    return (First.index() * ElementSize) + First.find_first();
  }

  // 返回位图中最后一个设置的位。如果没有设置的位，则返回 -1。
  int find_last() const {
    if (Elements.empty())
      return -1;
    // 获取最后一个设置的位的索引，并返回全局索引。
    const SparseBitVectorElement<ElementSize>& Last = *(Elements.rbegin());
    return (Last.index() * ElementSize) + Last.find_last();
  }

  // 返回 true 如果 SparseBitVector 为空
  bool empty() const {
    return Elements.empty();
  }

  // 返回位图中设置的位的总数
  unsigned count() const {
    unsigned BitCount = 0;
    // 计算位图中所有设置的位的总数
    for (ElementListConstIter Iter = Elements.begin(); Iter != Elements.end();
         ++Iter)
      BitCount += Iter->count();
    return BitCount;
  }



# 返回成员变量 BitCount 的值
    return BitCount;



  iterator begin() const {
    return iterator(this);
  }



# 返回一个迭代器，指向容器的起始位置
  iterator begin() const {
    return iterator(this);
  }



  iterator end() const {
    return iterator(this, true);
  }



# 返回一个迭代器，指向容器的结束位置
  iterator end() const {
    return iterator(this, true);
  }
};

// 方便的函数，允许在用户代码中进行 Or 和 And 操作，而无需对右值进行解引用。

template <unsigned ElementSize>
inline bool operator|=(
    SparseBitVector<ElementSize>& LHS,
    const SparseBitVector<ElementSize>* RHS) {
  // 使用 SparseBitVector 的成员函数 |= 进行位或操作
  return LHS |= *RHS;
}

template <unsigned ElementSize>
inline bool operator|=(
    SparseBitVector<ElementSize>* LHS,
    const SparseBitVector<ElementSize>& RHS) {
  // 使用 SparseBitVector 的成员函数 |= 进行位或操作
  return LHS->operator|=(RHS);
}

template <unsigned ElementSize>
inline bool operator&=(
    SparseBitVector<ElementSize>* LHS,
    const SparseBitVector<ElementSize>& RHS) {
  // 使用 SparseBitVector 的成员函数 &= 进行位与操作
  return LHS->operator&=(RHS);
}

template <unsigned ElementSize>
inline bool operator&=(
    SparseBitVector<ElementSize>& LHS,
    const SparseBitVector<ElementSize>* RHS) {
  // 使用 SparseBitVector 的成员函数 &= 进行位与操作
  return LHS &= *RHS;
}

// 中缀形式的并集、交集和差集操作的便捷函数。

template <unsigned ElementSize>
inline SparseBitVector<ElementSize> operator|(
    const SparseBitVector<ElementSize>& LHS,
    const SparseBitVector<ElementSize>& RHS) {
  // 创建一个新的 SparseBitVector 对象，并对其进行位或操作
  SparseBitVector<ElementSize> Result(LHS);
  Result |= RHS;
  return Result;
}

template <unsigned ElementSize>
inline SparseBitVector<ElementSize> operator&(
    const SparseBitVector<ElementSize>& LHS,
    const SparseBitVector<ElementSize>& RHS) {
  // 创建一个新的 SparseBitVector 对象，并对其进行位与操作
  SparseBitVector<ElementSize> Result(LHS);
  Result &= RHS;
  return Result;
}

template <unsigned ElementSize>
inline SparseBitVector<ElementSize> operator-(
    const SparseBitVector<ElementSize>& LHS,
    const SparseBitVector<ElementSize>& RHS) {
  // 创建一个新的 SparseBitVector 对象，并对其进行交集的补操作
  SparseBitVector<ElementSize> Result;
  Result.intersectWithComplement(LHS, RHS);
  return Result;
}

template <unsigned ElementSize>
std::ostream& operator<<(
    std::ostream& stream,
    const SparseBitVector<ElementSize>& vec) {
  bool first = true;
  stream << "{";
  // 将 SparseBitVector 对象中的元素输出到流中
  for (auto el : vec) {
    if (first) {
      first = false;
    } else {
      stream << ", ";
    }
    stream << el;
  }
  stream << "}";
  return stream;
}

} // end namespace c10
```