# `.\pytorch\c10\util\ArrayRef.h`

```
    // ArrayRef - 代表对数组的常量引用（内存中连续的0个或多个元素），即起始指针和长度。
    // 这允许各种 API 轻松方便地获取连续的元素。

    // 此类不拥有底层数据，它预期在数据驻留在某些其他缓冲区中，并且该缓冲区的生命周期超过 ArrayRef 的情况下使用。
    // 因此，通常情况下不安全将 ArrayRef 存储起来。

    // 这里意图是可以简单复制，因此应该通过值传递。

template <typename T>
class ArrayRef final {
 public:
  using iterator = const T*;  // 迭代器类型为常量指针
  using const_iterator = const T*;  // 常量迭代器类型为常量指针
  using size_type = size_t;  // 大小类型为 size_t
  using value_type = T;  // 值类型为 T

  using reverse_iterator = std::reverse_iterator<iterator>;  // 反向迭代器类型为标准库的反向迭代器

 private:
  const T* Data;  // 数组的起始指针，位于外部缓冲区中
  size_type Length;  // 元素的数量

  // 调试检查空指针不变性
  void debugCheckNullptrInvariant() {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        Data != nullptr || Length == 0,
        "created ArrayRef with nullptr and non-zero length! std::optional relies on this being illegal");
  }

 public:
  // 构造函数
  // 构造一个空的 ArrayRef。
  /* implicit */ constexpr ArrayRef() : Data(nullptr), Length(0) {}

  // 从单个元素构造 ArrayRef。
  // TODO：使这个构造函数显式
  constexpr ArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}

  // 从指针和长度构造 ArrayRef。
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA ArrayRef(const T* data, size_t length)
      : Data(data), Length(length) {
    debugCheckNullptrInvariant();  // 调用调试检查空指针不变性的方法
  }

  // 从范围构造 ArrayRef。
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA ArrayRef(const T* begin, const T* end)
      : Data(begin), Length(end - begin) {
    /// 调试函数，检查空指针不变性。
    debugCheckNullptrInvariant();
    }
    
    /// 从 SmallVector 构造一个 ArrayRef。这里使用模板化的方式，以避免在每次复制构造 ArrayRef 时实例化 SmallVectorTemplateCommon<T, U>。
    template <typename U>
    /* implicit */ ArrayRef(const SmallVectorTemplateCommon<T, U>& Vec)
        : Data(Vec.data()), Length(Vec.size()) {
      // 调用调试函数，检查空指针不变性。
      debugCheckNullptrInvariant();
    }
    
    template <
        typename Container,
        typename = std::enable_if_t<std::is_same_v<
            std::remove_const_t<decltype(std::declval<Container>().data())>,
            T*>>>
    /* implicit */ ArrayRef(const Container& container)
        : Data(container.data()), Length(container.size()) {
      // 调用调试函数，检查空指针不变性。
      debugCheckNullptrInvariant();
    }
    
    /// 从 std::vector 构造一个 ArrayRef。
    // 这里的 enable_if 确保不用于 std::vector<bool>，因为 ArrayRef 不能用于 std::vector<bool> 的位域。
    template <typename A>
    /* implicit */ ArrayRef(const std::vector<T, A>& Vec)
        : Data(Vec.data()), Length(Vec.size()) {
      static_assert(
          !std::is_same<T, bool>::value,
          "ArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
    }
    
    /// 从 std::array 构造一个 ArrayRef。
    template <size_t N>
    /* implicit */ constexpr ArrayRef(const std::array<T, N>& Arr)
        : Data(Arr.data()), Length(N) {}
    
    /// 从 C 数组构造一个 ArrayRef。
    template <size_t N>
    // NOLINTNEXTLINE(*c-arrays*)
    /* implicit */ constexpr ArrayRef(const T (&Arr)[N]) : Data(Arr), Length(N) {}
    
    /// 从 std::initializer_list 构造一个 ArrayRef。
    /* implicit */ constexpr ArrayRef(const std::initializer_list<T>& Vec)
        : Data(
              std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                               : std::begin(Vec)),
          Length(Vec.size()) {}
    
    /// @}
    /// @name Simple Operations
    /// @{
    
    /// 返回数组的起始迭代器。
    constexpr iterator begin() const {
      return Data;
    }
    
    /// 返回数组的结束迭代器。
    constexpr iterator end() const {
      return Data + Length;
    }
    
    // 由于 ArrayRef 只提供 const 迭代器，因此这些函数实际上与 iterator 相同。
    constexpr const_iterator cbegin() const {
      return Data;
    }
    constexpr const_iterator cend() const {
      return Data + Length;
    }
    
    /// 返回反向迭代器，指向数组的最后一个元素的下一个位置。
    constexpr reverse_iterator rbegin() const {
      return reverse_iterator(end());
    }
    constexpr reverse_iterator rend() const {
      return reverse_iterator(begin());
    }
    
    /// 检查数组是否为空。
    constexpr bool empty() const {
      return Length == 0;
    }
    
    /// 返回数组的数据指针。
    constexpr const T* data() const {
      return Data;
    }
    
    /// 返回数组的大小。
    constexpr size_t size() const {
      return Length;
    }
    
    /// 返回数组的第一个元素的引用。
    C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA const T& front() const {
      TORCH_CHECK(
          !empty(), "ArrayRef: attempted to access front() of empty list");
  // 返回数组中第一个元素的引用。
  return Data[0];
}

/// back - Get the last element.
C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA const T& back() const {
  // 检查数组是否为空，若为空则抛出异常。
  TORCH_CHECK(!empty(), "ArrayRef: attempted to access back() of empty list");
  // 返回数组中最后一个元素的引用。
  return Data[Length - 1];
}

/// equals - Check for element-wise equality.
constexpr bool equals(ArrayRef RHS) const {
  // 检查数组长度和 RHS 数组长度是否相等，以及数组内容是否逐个相等。
  return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
}

/// slice(n, m) - Take M elements of the array starting at element N
C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA ArrayRef<T> slice(size_t N, size_t M)
    const {
  // 检查切片的起始索引和长度是否有效，若无效则抛出异常。
  TORCH_CHECK(
      N + M <= size(),
      "ArrayRef: invalid slice, N = ",
      N,
      "; M = ",
      M,
      "; size = ",
      size());
  // 返回从索引 N 开始的长度为 M 的子数组的新的 ArrayRef 对象。
  return ArrayRef<T>(data() + N, M);
}

/// slice(n) - Chop off the first N elements of the array.
C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA ArrayRef<T> slice(size_t N) const {
  // 检查切片的长度是否有效，若无效则抛出异常。
  TORCH_CHECK(
      N <= size(), "ArrayRef: invalid slice, N = ", N, "; size = ", size());
  // 返回去除数组前 N 个元素后的新的 ArrayRef 对象。
  return slice(N, size() - N);
}

/// @}
/// @name Operator Overloads
/// @{
constexpr const T& operator[](size_t Index) const {
  // 返回数组中指定索引处的元素的引用。
  return Data[Index];
}

/// Vector compatibility
C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA const T& at(size_t Index) const {
  // 检查索引是否在有效范围内，若不在则抛出异常。
  TORCH_CHECK(
      Index < Length,
      "ArrayRef: invalid index Index = ",
      Index,
      "; Length = ",
      Length);
  // 返回数组中指定索引处的元素的引用。
  return Data[Index];
}

/// Disallow accidental assignment from a temporary.
///
/// The declaration here is extra complicated so that "arrayRef = {}"
/// continues to select the move assignment operator.
template <typename U>
std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    U&& Temporary) = delete;

/// Disallow accidental assignment from a temporary.
///
/// The declaration here is extra complicated so that "arrayRef = {}"
/// continues to select the move assignment operator.
template <typename U>
std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(
    std::initializer_list<U>) = delete;

/// @}
/// @name Expensive Operations
/// @{
std::vector<T> vec() const {
  // 返回包含数组所有元素的标准库 vector 对象的副本。
  return std::vector<T>(Data, Data + Length);
}

/// @}
};

// 重载流输出操作符，用于打印 ArrayRef 对象内容
template <typename T>
std::ostream& operator<<(std::ostream& out, ArrayRef<T> list) {
  int i = 0;
  out << "[";
  // 遍历 ArrayRef 中的元素，逐个输出到流中
  for (const auto& e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "]";
  return out;
}

/// @name ArrayRef Convenience constructors
/// @{

/// 从单个元素构造一个 ArrayRef
template <typename T>
ArrayRef<T> makeArrayRef(const T& OneElt) {
  return OneElt;
}

/// 从指针和长度构造一个 ArrayRef
template <typename T>
ArrayRef<T> makeArrayRef(const T* data, size_t length) {
  return ArrayRef<T>(data, length);
}

/// 从范围构造一个 ArrayRef
template <typename T>
ArrayRef<T> makeArrayRef(const T* begin, const T* end) {
  return ArrayRef<T>(begin, end);
}

/// 从 SmallVector 构造一个 ArrayRef
template <typename T>
ArrayRef<T> makeArrayRef(const SmallVectorImpl<T>& Vec) {
  return Vec;
}

/// 从 SmallVector 构造一个 ArrayRef
template <typename T, unsigned N>
ArrayRef<T> makeArrayRef(const SmallVector<T, N>& Vec) {
  return Vec;
}

/// 从 std::vector 构造一个 ArrayRef
template <typename T>
ArrayRef<T> makeArrayRef(const std::vector<T>& Vec) {
  return Vec;
}

/// 从 std::array 构造一个 ArrayRef
template <typename T, std::size_t N>
ArrayRef<T> makeArrayRef(const std::array<T, N>& Arr) {
  return Arr;
}

/// 从 ArrayRef 构造一个 ArrayRef (无操作)
template <typename T>
ArrayRef<T> makeArrayRef(const ArrayRef<T>& Vec) {
  return Vec;
}

/// 从 ArrayRef 构造一个 ArrayRef (无操作)
template <typename T>
ArrayRef<T>& makeArrayRef(ArrayRef<T>& Vec) {
  return Vec;
}

/// 从 C 数组构造一个 ArrayRef
template <typename T, size_t N>
// NOLINTNEXTLINE(*c-arrays*)
ArrayRef<T> makeArrayRef(const T (&Arr)[N]) {
  return ArrayRef<T>(Arr);
}

// 警告: 模板实例化不会愿意执行隐式转换以获取 c10::ArrayRef，
// 这就是为什么我们需要如此多的重载函数。

// 比较操作符：检查两个 c10::ArrayRef 对象是否相等
template <typename T>
bool operator==(c10::ArrayRef<T> a1, c10::ArrayRef<T> a2) {
  return a1.equals(a2);
}

// 比较操作符：检查两个 c10::ArrayRef 对象是否不相等
template <typename T>
bool operator!=(c10::ArrayRef<T> a1, c10::ArrayRef<T> a2) {
  return !a1.equals(a2);
}

// 比较操作符：检查 std::vector 和 c10::ArrayRef 是否相等
template <typename T>
bool operator==(const std::vector<T>& a1, c10::ArrayRef<T> a2) {
  return c10::ArrayRef<T>(a1).equals(a2);
}

// 比较操作符：检查 std::vector 和 c10::ArrayRef 是否不相等
template <typename T>
bool operator!=(const std::vector<T>& a1, c10::ArrayRef<T> a2) {
  return !c10::ArrayRef<T>(a1).equals(a2);
}

// 比较操作符：检查 c10::ArrayRef 和 std::vector 是否相等
template <typename T>
bool operator==(c10::ArrayRef<T> a1, const std::vector<T>& a2) {
  return a1.equals(c10::ArrayRef<T>(a2));
}

// 比较操作符：检查 c10::ArrayRef 和 std::vector 是否不相等
template <typename T>
bool operator!=(c10::ArrayRef<T> a1, const std::vector<T>& a2) {
  return !a1.equals(c10::ArrayRef<T>(a2));
}

// 使用 IntArrayRef 作为 ArrayRef<int64_t> 的别名
using IntArrayRef = ArrayRef<int64_t>;

// 此别名已弃用，因为不明确所有权语义。应使用 IntArrayRef 替代!
C10_DEFINE_DEPRECATED_USING(IntList, ArrayRef<int64_t>)

// 结束 c10 命名空间
} // namespace c10
```