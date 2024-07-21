# `.\pytorch\c10\util\SmallVector.h`

```py
// LLVM 项目中用于处理 'Normally small' 向量的类 SmallVector 的定义
// 此处的 SmallVectorBase 类包含了所有 SmallVector 共有的基本功能

#pragma once

#include <c10/macros/Macros.h>   // 引入 C10_API 宏，用于导出 SmallVectorBase
#include <c10/util/AlignOf.h>    // 引入 AlignOf 工具，可能用于内存对齐计算

#include <algorithm>            // 引入算法标准库，例如 std::min, std::max 等
#include <cassert>              // 引入断言库，用于调试时的断言检查
#include <cstddef>              // 引入标准库定义的常用类型与宏
#include <cstdlib>              // 引入标准库的一般工具函数，例如 malloc, free 等
#include <cstring>              // 引入 C 字符串操作函数库，例如 memcpy, memset 等
#include <functional>           // 引入函数对象和函数适配器的库
#include <initializer_list>     // 引入初始化列表库，支持使用初始化列表初始化对象
#include <iterator>             // 引入迭代器相关的库，例如 std::iterator_traits 等
#include <limits>               // 引入数值极限和类型特性的库
#include <memory>               // 引入智能指针和动态内存分配的库
#include <ostream>              // 引入输出流操作的库
#include <type_traits>          // 引入类型特性的库，例如 std::is_trivially_copyable 等
#include <utility>              // 引入实用工具库，例如 std::move, std::forward 等

namespace c10 {

/// SmallVectorBase 类是所有 SmallVector 的基类，提供了共有的容量管理和基本操作
///
/// Size_T 模板参数指定用于存储 Size 和 Capacity 的类型，以便调整大小
/// 使用 32 位大小有助于减小 SmallVector 的大小
/// 使用 64 位大小适合像 SmallVector<char> 这样的情况，32 位大小会限制向量大小在 ~4GB 内
template <class Size_T>
class C10_API SmallVectorBase {
 protected:
  void* BeginX;     // 指向向量起始元素的指针或指针代理
  Size_T Size = 0, Capacity;  // Size 表示当前向量的大小，Capacity 表示当前向量的容量

  /// 返回 Size_T 类型的最大值
  static constexpr size_t SizeTypeMax() {
    return std::numeric_limits<Size_T>::max();
  }

  /// 构造函数，初始化 BeginX 指针和 Capacity 容量
  SmallVectorBase(void* FirstEl, size_t TotalCapacity)
      : BeginX(FirstEl), Capacity(TotalCapacity) {}

  /// 这是 grow() 方法的辅助函数，为了减少代码重复而提出来的
  /// 如果不能至少扩展到 MinSize，将报告致命错误
  void* mallocForGrow(size_t MinSize, size_t TSize, size_t& NewCapacity);

  /// 这是仅适用于 POD 类型数据的 grow() 方法的实现，为了减少代码重复而提出来的
  /// 如果无法增加容量，将报告致命错误
  void grow_pod(const void* FirstEl, size_t MinSize, size_t TSize);

 public:
  SmallVectorBase() = delete;   // 删除默认构造函数
  size_t size() const {         // 返回当前向量的大小
    return Size;
  }
  size_t capacity() const {     // 返回当前向量的容量
    return Capacity;
  }

  /// 返回向量是否为空
  C10_NODISCARD bool empty() const {
    // 返回当前数组是否为空，通过检查 Size 是否为零来判断
    return !Size;
  }

  /// Set the array size to \p N, which the current array must have enough
  /// capacity for.
  ///
  /// This does not construct or destroy any elements in the vector.
  ///
  /// Clients can use this in conjunction with capacity() to write past the end
  /// of the buffer when they know that more elements are available, and only
  /// update the size later. This avoids the cost of value initializing elements
  /// which will only be overwritten.
  void set_size(size_t N) {
    // 断言当前设置的大小 N 不超过数组的容量，即 N <= capacity()
    assert(N <= capacity());
    // 将数组的大小 Size 设置为 N
    Size = N;
  }
};

/// 模板别名，根据 T 的大小选择 uint64_t 或 uint32_t 作为 SmallVector 的大小类型
template <class T>
using SmallVectorSizeType =
    std::conditional_t<sizeof(T) < 4 && sizeof(void*) >= 8, uint64_t, uint32_t>;

/// 计算第一个元素的偏移量
template <class T, typename = void>
struct SmallVectorAlignmentAndSize {
  // NOLINTNEXTLINE(*c-arrays*)
  alignas(SmallVectorBase<SmallVectorSizeType<T>>) char Base[sizeof(
      SmallVectorBase<SmallVectorSizeType<T>>)];
  // NOLINTNEXTLINE(*c-arrays*)
  alignas(T) char FirstEl[sizeof(T)];
};

/// 这部分不依赖于 T 是否为 POD 的 SmallVectorTemplateBase 的公共部分。
/// 针对 ArrayRef 使用额外的虚拟模板参数来避免要求 T 是完整的。
template <typename T, typename = void>
class SmallVectorTemplateCommon
    : public SmallVectorBase<SmallVectorSizeType<T>> {
  using Base = SmallVectorBase<SmallVectorSizeType<T>>;

  /// 获取第一个元素的地址。为了保证这个指针数学在 T 的小尺寸为 0 时也有效，
  /// 对于 SmallVectorStorage，即使 T 的小尺寸为 0，也必须有正确的对齐。
  void* getFirstEl() const {
    return const_cast<void*>(reinterpret_cast<const void*>(
        reinterpret_cast<const char*>(this) +
        offsetof(SmallVectorAlignmentAndSize<T>, FirstEl)));
  }
  // 'FirstEl' 后面的空间会被覆盖，不要在它之后添加任何实例变量。

 protected:
  SmallVectorTemplateCommon(size_t Size) : Base(getFirstEl(), Size) {}

  /// 增长 POD 类型的 SmallVector
  void grow_pod(size_t MinSize, size_t TSize) {
    Base::grow_pod(getFirstEl(), MinSize, TSize);
  }

  /// 如果这是一个未分配动态内存的小型向量，则返回 true
  bool isSmall() const {
    return this->BeginX == getFirstEl();
  }

  /// 将向量重置为小型状态
  void resetToSmall() {
    this->BeginX = getFirstEl();
    this->Size = this->Capacity = 0; // FIXME: 设置 Capacity 为 0 是可疑的。
  }

  /// 如果 V 是对给定范围的内部引用，则返回 true
  bool isReferenceToRange(const void* V, const void* First, const void* Last)
      const {
    // 使用 std::less 来避免 UB
    std::less<> LessThan;
    return !LessThan(V, First) && LessThan(V, Last);
  }

  /// 如果 V 是对此向量的内部引用，则返回 true
  bool isReferenceToStorage(const void* V) const {
    return isReferenceToRange(V, this->begin(), this->end());
  }

  /// 如果 First 和 Last 在此向量的存储中形成有效（可能为空）的范围，则返回 true
  bool isRangeInStorage(const void* First, const void* Last) const {
    // 使用 std::less 来避免 UB
    std::less<> LessThan;
    return !LessThan(First, this->begin()) && !LessThan(Last, First) &&
        !LessThan(this->end(), Last);
  }

  /// 除非 Elt 将因调整向量大小到 NewSize 而无效，否则返回 true
  bool isSafeToReferenceAfterResize(const void* Elt, size_t NewSize) {
    // 超出尾部
    if (C10_LIKELY(!isReferenceToStorage(Elt)))
      return true;
    // 如果 NewSize 小于等于当前 vector 的大小（size），则检查 Elt 是否在缩小后的有效范围内，返回 false 表示会被销毁。
    if (NewSize <= this->size())
      return Elt < this->begin() + NewSize;

    // 如果需要扩展，则返回 false。
    return NewSize <= this->capacity();
  }

  /// 检查在将向量大小调整为 NewSize 后，Elt 是否会失效。
  void assertSafeToReferenceAfterResize(const void* Elt, size_t NewSize) {
    (void)Elt; // 抑制未使用变量警告
    (void)NewSize; // 抑制未使用变量警告
    assert(
        isSafeToReferenceAfterResize(Elt, NewSize) &&
        "Attempting to reference an element of the vector in an operation "
        "that invalidates it");
  }

  /// 检查在向量增加 N 大小后，Elt 是否会失效。
  void assertSafeToAdd(const void* Elt, size_t N = 1) {
    this->assertSafeToReferenceAfterResize(Elt, this->size() + N);
  }

  /// 检查在清除范围 From 到 To 之间的任何部分后，Elt 是否会失效。
  void assertSafeToReferenceAfterClear(const T* From, const T* To) {
    if (From == To)
      return;
    this->assertSafeToReferenceAfterResize(From, 0);
    this->assertSafeToReferenceAfterResize(To - 1, 0);
  }
  template <
      class ItTy,
      std::enable_if_t<!std::is_same_v<std::remove_const_t<ItTy>, T*>, bool> =
          false>
  void assertSafeToReferenceAfterClear(ItTy, ItTy) {}

  /// 检查在向量增长时，范围 From 到 To 之间的任何部分是否会失效。
  void assertSafeToAddRange(const T* From, const T* To) {
    if (From == To)
      return;
    this->assertSafeToAdd(From, To - From);
    this->assertSafeToAdd(To - 1, To - From);
  }
  template <
      class ItTy,
      std::enable_if_t<!std::is_same_v<std::remove_const_t<ItTy>, T*>, bool> =
          false>
  void assertSafeToAddRange(ItTy, ItTy) {}

  /// 预留足够空间以添加一个元素，并返回更新后的元素指针（如果原先是对存储的引用）。
  template <class U>
  static const T* reserveForParamAndGetAddressImpl(
      U* This,
      const T& Elt,
      size_t N) {
    size_t NewSize = This->size() + N;
    // 如果当前容量足够容纳 NewSize，则直接返回指向 Elt 的指针。
    if (C10_LIKELY(NewSize <= This->capacity()))
      return &Elt;

    bool ReferencesStorage = false;
    int64_t Index = -1;
    // 如果不是按值传递参数，则检查是否 Elt 是对存储的引用，若是，则标记为引用存储并记录其索引。
    if constexpr (!U::TakesParamByValue) {
      if (C10_UNLIKELY(This->isReferenceToStorage(&Elt))) {
        ReferencesStorage = true;
        Index = &Elt - This->begin();
      }
    }
    // 扩展向量容量以容纳 NewSize。
    This->grow(NewSize);
  return ReferencesStorage ? This->begin() + Index : &Elt;



  // 如果 ReferencesStorage 为真，则返回迭代器指向的位置加上 Index；否则返回指向 Elt 的指针。
  return ReferencesStorage ? This->begin() + Index : &Elt;
}



 public:
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = std::reverse_iterator<iterator>;

  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;

  using Base::capacity;
  using Base::empty;
  using Base::size;



  // 使用基类中的 capacity、empty 和 size 成员
  using Base::capacity;
  using Base::empty;
  using Base::size;



  // forward iterator creation methods.
  iterator begin() {
    // 返回指向 BeginX 的迭代器
    return (iterator)this->BeginX;
  }



  const_iterator begin() const {
    // 返回指向 BeginX 的常量迭代器
    return (const_iterator)this->BeginX;
  }



  iterator end() {
    // 返回指向末尾的迭代器
    return begin() + size();
  }



  const_iterator end() const {
    // 返回指向末尾的常量迭代器
    return begin() + size();
  }



  // reverse iterator creation methods.
  reverse_iterator rbegin() {
    // 返回逆向迭代器，指向末尾
    return reverse_iterator(end());
  }



  const_reverse_iterator rbegin() const {
    // 返回逆向常量迭代器，指向末尾
    return const_reverse_iterator(end());
  }



  reverse_iterator rend() {
    // 返回逆向迭代器，指向起始
    return reverse_iterator(begin());
  }



  const_reverse_iterator rend() const {
    // 返回逆向常量迭代器，指向起始
    return const_reverse_iterator(begin());
  }



  size_type size_in_bytes() const {
    // 返回容器中元素占用的字节数
    return size() * sizeof(T);
  }



  constexpr size_type max_size() const {
    // 返回容器的最大可能尺寸
    return std::min(this->SizeTypeMax(), size_type(-1) / sizeof(T));
  }



  size_t capacity_in_bytes() const {
    // 返回容器的容量占用的字节数
    return capacity() * sizeof(T);
  }



  /// Return a pointer to the vector's buffer, even if empty().
  pointer data() {
    // 返回指向容器数据的指针，即使容器为空也有效
    return pointer(begin());
  }



  /// Return a pointer to the vector's buffer, even if empty().
  const_pointer data() const {
    // 返回指向容器数据的常量指针，即使容器为空也有效
    return const_pointer(begin());
  }



  // SmallVector::at is NOT from LLVM.
  reference at(size_type idx) {
    // 检索并返回索引为 idx 的元素的引用，如果索引超出有效范围，会触发断言错误
    assert(idx < size());
    return begin()[idx];
  }



  const_reference at(size_type idx) const {
    // 检索并返回索引为 idx 的元素的常量引用，如果索引超出有效范围，会触发断言错误
    assert(idx < size());
    return begin()[idx];
  }



  reference operator[](size_type idx) {
    // 返回索引为 idx 的元素的引用，如果索引超出有效范围，会触发断言错误
    assert(idx < size());
    return begin()[idx];
  }



  const_reference operator[](size_type idx) const {
    // 返回索引为 idx 的元素的常量引用，如果索引超出有效范围，会触发断言错误
    assert(idx < size());
    return begin()[idx];
  }



  reference front() {
    // 返回容器第一个元素的引用，如果容器为空，会触发断言错误
    assert(!empty());
    return begin()[0];
  }



  const_reference front() const {
    // 返回容器第一个元素的常量引用，如果容器为空，会触发断言错误
    assert(!empty());
    return begin()[0];
  }



  reference back() {
    // 返回容器最后一个元素的引用，如果容器为空，会触发断言错误
    assert(!empty());
    return end()[-1];
  }



  const_reference back() const {
    // 返回容器最后一个元素的常量引用，如果容器为空，会触发断言错误
    assert(!empty());
    return end()[-1];
  }
/// SmallVectorTemplateBase<TriviallyCopyable = false> - 这是用于处理非平凡类型 T 的方法实现的地方。
///
/// 我们使用平凡的移动/复制构造和平凡的析构来近似 is_trivially_copyable。
/// 虽然标准没有明确允许使用 memcpy 复制这些类型，但类型无法观察到这一点。
/// 这捕捉了 std::pair<POD, POD> 等重要情况，这些情况不是平凡可分配的。
///
/// 如果在这里构建失败，请回退到 C10_IS_TRIVIALLY_COPYABLE 并做好备注。
template <
    typename T,
    bool = (std::is_trivially_copy_constructible_v<T>) &&
           (std::is_trivially_move_constructible_v<T>) &&
           std::is_trivially_destructible_v<T>>
class SmallVectorTemplateBase : public SmallVectorTemplateCommon<T> {
  friend class SmallVectorTemplateCommon<T>;

 protected:
  static constexpr bool TakesParamByValue = false;
  using ValueParamT = const T&;

  /// 构造函数，用给定大小初始化 SmallVectorTemplateCommon<T>
  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  /// 销毁范围 [S, E) 内的元素
  static void destroy_range(T* S, T* E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }

  /// 将范围 [I, E) 的元素移动到以 "Dest" 开始的未初始化内存中，必要时构造元素
  template <typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(
        std::make_move_iterator(I), std::make_move_iterator(E), Dest);
  }

  /// 将范围 [I, E) 的元素复制到以 "Dest" 开始的未初始化内存中，必要时构造元素
  template <typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(I, E, Dest);
  }

  /// 扩展已分配的内存（不初始化新元素），将分配的内存大小加倍。保证至少为一个元素留出空间，如果指定了 MinSize，则保证空间足够容纳 MinSize 个元素
  void grow(size_t MinSize = 0);

  /// 创建一个足够大的新分配，将其大小通过 NewCapacity 返回。这是 grow() 的第一部分。
  T* mallocForGrow(size_t MinSize, size_t& NewCapacity) {
    return static_cast<T*>(
        SmallVectorBase<SmallVectorSizeType<T>>::mallocForGrow(
            MinSize, sizeof(T), NewCapacity));
  }

  /// 将现有元素移动到新分配的内存 NewElts 中，这是 grow() 的中间部分。
  void moveElementsForGrow(T* NewElts);

  /// 转移分配的所有权，完成 grow() 的最后部分。
  void takeAllocationForGrow(T* NewElts, size_t NewCapacity);

  /// 预留足够的空间以添加一个元素，并返回更新后的元素指针，以防它是对存储的引用。
  const T* reserveForParamAndGetAddress(const T& Elt, size_t N = 1) {
  /// 返回通过调用 reserveForParamAndGetAddressImpl() 方法预留空间后的元素地址。
  /// 如果该元素是存储区域的引用，则返回更新后的元素指针。
  T* reserveForParamAndGetAddress(T& Elt, size_t N = 1) {
    return const_cast<T*>(this->reserveForParamAndGetAddressImpl(this, Elt, N));
  }

  /// 转发右值参数 V，并返回其右值引用。
  static T&& forward_value_param(T&& V) {
    return std::move(V);
  }

  /// 转发常量左值参数 V，并返回其常量引用。
  static const T& forward_value_param(const T& V) {
    return V;
  }

  /// 在指定数量 NumElts 的情况下增长并分配内存，使用 Elt 进行初始化。
  void growAndAssign(size_t NumElts, const T& Elt) {
    // 手动增长以防 Elt 是内部引用。
    size_t NewCapacity = 0;
    T* NewElts = mallocForGrow(NumElts, NewCapacity); // 分配新内存空间
    std::uninitialized_fill_n(NewElts, NumElts, Elt); // 使用 Elt 初始化新分配的内存
    this->destroy_range(this->begin(), this->end()); // 销毁当前范围内的元素
    takeAllocationForGrow(NewElts, NewCapacity); // 将新分配的内存空间接管
    this->set_size(NumElts); // 更新容器的大小
  }

  /// 在末尾增加一个元素，并通过调用构造函数初始化它。
  void push_back(const T& Elt) {
    const T* EltPtr = reserveForParamAndGetAddress(Elt); // 预留空间并获取元素地址
    ::new ((void*)this->end()) T(*EltPtr); // 在末尾构造一个新元素
    this->set_size(this->size() + 1); // 增加容器的大小
  }

  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  /// 在末尾增加一个右值引用元素，并通过调用移动构造函数初始化它。
  void push_back(T&& Elt) {
    T* EltPtr = reserveForParamAndGetAddress(Elt); // 预留空间并获取元素地址
    ::new ((void*)this->end()) T(::std::move(*EltPtr)); // 在末尾构造一个新元素（移动构造）
    this->set_size(this->size() + 1); // 增加容器的大小
  }

  /// 删除末尾的一个元素。
  void pop_back() {
    this->set_size(this->size() - 1); // 减小容器的大小
    this->end()->~T(); // 销毁末尾元素
  }

  /// 在增长并在末尾构造新元素时使用，支持任意数量的构造参数。
  template <typename... ArgTypes>
  T& growAndEmplaceBack(ArgTypes&&... Args) {
    // 手动增长以防 Args 中有内部引用。
    size_t NewCapacity = 0;
    T* NewElts = mallocForGrow(0, NewCapacity); // 分配新内存空间
    ::new ((void*)(NewElts + this->size())) T(std::forward<ArgTypes>(Args)...); // 在末尾构造新元素
    moveElementsForGrow(NewElts); // 移动现有元素到新内存空间
    takeAllocationForGrow(NewElts, NewCapacity); // 将新分配的内存空间接管
    this->set_size(this->size() + 1); // 增加容器的大小
    return this->back(); // 返回新增加的元素的引用
  }
// 定义此函数为外部函数以防止 C++ 编译器内联它。
template <typename T, bool TriviallyCopyable>
void SmallVectorTemplateBase<T, TriviallyCopyable>::grow(size_t MinSize) {
  // 新容量初始化为0
  size_t NewCapacity = 0;
  // 为增长分配内存，并返回指向新元素的指针
  T* NewElts = mallocForGrow(MinSize, NewCapacity);
  // 将现有元素移动到新分配的内存中
  moveElementsForGrow(NewElts);
  // 将新的内存分配和容量更新到当前对象中
  takeAllocationForGrow(NewElts, NewCapacity);
}

// 定义此函数为外部函数以防止 C++ 编译器内联它。
template <typename T, bool TriviallyCopyable>
void SmallVectorTemplateBase<T, TriviallyCopyable>::moveElementsForGrow(
    T* NewElts) {
  // 将元素移动到新内存位置
  this->uninitialized_move(this->begin(), this->end(), NewElts);

  // 销毁原始元素
  destroy_range(this->begin(), this->end());
}

// 定义此函数为外部函数以防止 C++ 编译器内联它。
template <typename T, bool TriviallyCopyable>
void SmallVectorTemplateBase<T, TriviallyCopyable>::takeAllocationForGrow(
    T* NewElts,
    size_t NewCapacity) {
  // 如果不是从内联副本增长的，则释放旧空间
  if (!this->isSmall())
    free(this->begin());

  // 更新开始指针和容量
  this->BeginX = NewElts;
  this->Capacity = NewCapacity;
}

/// SmallVectorTemplateBase<TriviallyCopyable = true> - This is where we put
/// method implementations that are designed to work with trivially copyable
/// T's. This allows using memcpy in place of copy/move construction and
/// skipping destruction.
template <typename T>
class SmallVectorTemplateBase<T, true> : public SmallVectorTemplateCommon<T> {
  friend class SmallVectorTemplateCommon<T>;

 protected:
  /// True if it's cheap enough to take parameters by value. Doing so avoids
  /// overhead related to mitigations for reference invalidation.
  static constexpr bool TakesParamByValue = sizeof(T) <= 2 * sizeof(void*);

  /// Either const T& or T, depending on whether it's cheap enough to take
  /// parameters by value.
  using ValueParamT = std::conditional_t<TakesParamByValue, T, const T&>;

  // 构造函数，初始化基类 SmallVectorTemplateCommon<T>
  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  // 对于 POD 类型，不需要执行销毁循环。
  static void destroy_range(T*, T*) {}

  /// Move the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    // 直接进行复制
    uninitialized_copy(I, E, Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    // 任意迭代器类型；使用基本实现进行复制
    // 这里可能包含更详细的复制实现
    std::uninitialized_copy(I, E, Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename T1, typename T2>
  static void uninitialized_copy(
      T1* I,
      T1* E,
      T2* Dest,
      std::enable_if_t<std::is_same_v<std::remove_const_t<T1>, T2>>* =
          nullptr) {
    // 对于通过指针迭代的 POD 类型（包括 SmallVector 迭代器），使用 memcpy：
    // std::uninitialized_copy 会优化为 memmove，但这里我们可以使用 memcpy。
    // 注意，I 和 E 是迭代器，如果它们相等，使用 memcpy 可能会无效。
    if (I != E)
      memcpy(reinterpret_cast<void*>(Dest), I, (E - I) * sizeof(T));
  }

  /// Double the size of the allocated memory, guaranteeing space for at
  /// least one more element or MinSize if specified.
  void grow(size_t MinSize = 0) {
    this->grow_pod(MinSize, sizeof(T));
  }

  /// Reserve enough space to add one element, and return the updated element
  /// pointer in case it was a reference to the storage.
  const T* reserveForParamAndGetAddress(const T& Elt, size_t N = 1) {
    return this->reserveForParamAndGetAddressImpl(this, Elt, N);
  }

  /// Reserve enough space to add one element, and return the updated element
  /// pointer in case it was a reference to the storage.
  T* reserveForParamAndGetAddress(T& Elt, size_t N = 1) {
    return const_cast<T*>(this->reserveForParamAndGetAddressImpl(this, Elt, N));
  }

  /// Copy \p V or return a reference, depending on \a ValueParamT.
  static ValueParamT forward_value_param(ValueParamT V) {
    return V;
  }

  /// Increase the size of the container to \p NumElts and assign \p Elt to each element.
  void growAndAssign(size_t NumElts, T Elt) {
    // 如果 Elt 是内部引用，通过复制避免引用失效问题，同时不失去重新分配的优化。
    this->set_size(0);
    this->grow(NumElts);
    std::uninitialized_fill_n(this->begin(), NumElts, Elt);
    this->set_size(NumElts);
  }

  /// Extend the container by constructing a new element at the end with \p Args.
  template <typename... ArgTypes>
  T& growAndEmplaceBack(ArgTypes&&... Args) {
    // 如果 Args 包含内部引用，通过复制避免引用失效问题，同时不失去重新分配的优化。
    push_back(T(std::forward<ArgTypes>(Args)...));
    return this->back();
  }

 public:
  /// Add an element \p Elt to the end of the container.
  void push_back(ValueParamT Elt) {
    // 获取 \p Elt 的地址并预留空间，然后使用 memcpy 将其复制到容器末尾。
    const T* EltPtr = reserveForParamAndGetAddress(Elt);
    memcpy(reinterpret_cast<void*>(this->end()), EltPtr, sizeof(T));
    this->set_size(this->size() + 1);
  }

  /// Remove the last element from the container.
  void pop_back() {
    this->set_size(this->size() - 1);
  }
/// This class consists of common code factored out of the SmallVector class to
/// reduce code duplication based on the SmallVector 'N' template parameter.
template <typename T>
class SmallVectorImpl : public SmallVectorTemplateBase<T> {
  using SuperClass = SmallVectorTemplateBase<T>;

 public:
  using iterator = typename SuperClass::iterator;  ///< Iterator type for this class.
  using const_iterator = typename SuperClass::const_iterator;  ///< Const iterator type.
  using reference = typename SuperClass::reference;  ///< Reference type.
  using size_type = typename SuperClass::size_type;  ///< Type for size.

 protected:
  using SmallVectorTemplateBase<T>::TakesParamByValue;  ///< Flag for parameter passing type.
  using ValueParamT = typename SuperClass::ValueParamT;  ///< Type of value parameter.

  // Default ctor - Initialize to empty.
  explicit SmallVectorImpl(unsigned N) : SmallVectorTemplateBase<T>(N) {}

 public:
  SmallVectorImpl(const SmallVectorImpl&) = delete;  ///< Deleted copy constructor.

  ~SmallVectorImpl() {
    // Subclass has already destructed this vector's elements.
    // If this wasn't grown from the inline copy, deallocate the old space.
    if (!this->isSmall())
      free(this->begin());  ///< Free memory if not using small storage optimization.
  }

  /// Clears all elements in the vector.
  void clear() {
    this->destroy_range(this->begin(), this->end());  ///< Destroy elements in the range.
    this->Size = 0;  ///< Reset size to zero.
  }

 private:
  /// Internal method to resize the vector.
  template <bool ForOverwrite>
  void resizeImpl(size_type N) {
    if (N < this->size()) {
      this->pop_back_n(this->size() - N);  ///< Pop elements from the back.
    } else if (N > this->size()) {
      this->reserve(N);  ///< Ensure capacity is sufficient.
      for (auto I = this->end(), E = this->begin() + N; I != E; ++I)
        if (ForOverwrite)
          new (&*I) T;  ///< Placement new if overwriting existing elements.
        else
          new (&*I) T();  ///< Placement new with default constructor.
      this->set_size(N);  ///< Update size of the vector.
    }
  }

 public:
  /// Resize the vector to the specified size.
  void resize(size_type N) {
    resizeImpl<false>(N);  ///< Call internal resize method without overwriting.
  }

  /// Resize the vector to the specified size without initializing new values.
  void resize_for_overwrite(size_type N) {
    resizeImpl<true>(N);  ///< Call internal resize method for overwriting.
  }

  /// Resize the vector to the specified size and initialize new elements with NV.
  void resize(size_type N, ValueParamT NV) {
    if (N == this->size())
      return;  ///< If size matches, return early.

    if (N < this->size()) {
      this->pop_back_n(this->size() - N);  ///< Pop elements if size decreases.
      return;
    }

    // N > this->size(). Defer to append.
    this->append(N - this->size(), NV);  ///< Append NV to fill up to size N.
  }

  /// Ensure the vector can hold at least N elements.
  void reserve(size_type N) {
    if (this->capacity() < N)
      this->grow(N);  ///< Increase capacity if necessary.
  }

  /// Pop back the last NumItems elements from the vector.
  void pop_back_n(size_type NumItems) {
    assert(this->size() >= NumItems);  ///< Assert that enough elements exist.
    this->destroy_range(this->end() - NumItems, this->end());  ///< Destroy elements.
    this->set_size(this->size() - NumItems);  ///< Update size of the vector.
  }

  /// Pop back and return the last element of the vector.
  C10_NODISCARD T pop_back_val() {
    T Result = ::std::move(this->back());  ///< Move the last element.
    this->pop_back();  ///< Pop the last element from the vector.
    return Result;  ///< Return the moved element.
  }

  /// Swap the contents of this vector with another SmallVectorImpl instance.
  void swap(SmallVectorImpl& RHS) noexcept;

  /// Add the specified range to the end of the SmallVector.
  template <
      typename in_iter,
      typename = std::enable_if_t<std::is_convertible_v<
          typename std::iterator_traits<in_iter>::iterator_category,
          std::input_iterator_tag>>>
  void append(in_iter in_start, in_iter in_end) {
    this->assertSafeToAddRange(in_start, in_end);  ///< Assert range safety.
    size_type NumInputs = std::distance(in_start, in_end);  ///< Calculate number of inputs.
    this->reserve(this->size() + NumInputs);  ///< Reserve enough space.
    this->uninitialized_copy(in_start, in_end, this->end());  ///< Copy elements to end.
  // 增加当前向量的大小，使其包含新增的 NumInputs 个元素
  this->set_size(this->size() + NumInputs);
}

/// 向向量末尾添加 NumInputs 个 Elt 元素的副本。
void append(size_type NumInputs, ValueParamT Elt) {
  // 为参数 Elt 预留空间并获取其地址
  const T* EltPtr = this->reserveForParamAndGetAddress(Elt, NumInputs);
  // 在末尾未初始化地填充 NumInputs 个 Elt 元素
  std::uninitialized_fill_n(this->end(), NumInputs, *EltPtr);
  // 增加当前向量的大小，使其包含新增的 NumInputs 个元素
  this->set_size(this->size() + NumInputs);
}

/// 向向量末尾添加初始化列表 IL 中的元素。
void append(std::initializer_list<T> IL) {
  append(IL.begin(), IL.end());
}

/// 向向量末尾添加另一个 SmallVectorImpl 实例 RHS 中的元素。
void append(const SmallVectorImpl& RHS) {
  append(RHS.begin(), RHS.end());
}

/// 将向量的元素数量设置为 NumElts，并用值 Elt 赋值。
void assign(size_type NumElts, ValueParamT Elt) {
  // 注意：Elt 可能是内部引用。
  if (NumElts > this->capacity()) {
    // 如果需要的元素数量大于当前容量，则进行扩展并赋值。
    this->growAndAssign(NumElts, Elt);
    return;
  }

  // 覆盖现有元素。
  std::fill_n(this->begin(), std::min(NumElts, this->size()), Elt);
  if (NumElts > this->size())
    // 在末尾未初始化地填充 NumElts - size() 个 Elt 元素
    std::uninitialized_fill_n(this->end(), NumElts - this->size(), Elt);
  else if (NumElts < this->size())
    // 销毁多余的元素
    this->destroy_range(this->begin() + NumElts, this->end());
  // 设置向量的新大小
  this->set_size(NumElts);
}

// FIXME: 考虑覆盖现有元素，而不是清除并重新初始化它们 - 对于所有 assign(...) 变体。

template <
    typename in_iter,
    typename = std::enable_if_t<std::is_convertible_v<
        typename std::iterator_traits<in_iter>::iterator_category,
        std::input_iterator_tag>>>
/// 用区间 [in_start, in_end) 内的元素替换向量的内容。
void assign(in_iter in_start, in_iter in_end) {
  // 确保在清除后引用仍然安全
  this->assertSafeToReferenceAfterClear(in_start, in_end);
  // 清空当前向量
  clear();
  // 向向量末尾添加区间 [in_start, in_end) 内的元素
  append(in_start, in_end);
}

/// 用初始化列表 IL 中的元素替换向量的内容。
void assign(std::initializer_list<T> IL) {
  // 清空当前向量
  clear();
  // 向向量末尾添加初始化列表 IL 中的元素
  append(IL);
}

/// 用另一个 SmallVectorImpl 实例 RHS 的元素替换向量的内容。
void assign(const SmallVectorImpl& RHS) {
  // 用 RHS 的元素替换向量的内容
  assign(RHS.begin(), RHS.end());
}

/// 从向量中删除迭代器 I 指向的元素。
iterator erase(iterator I) {
  // 确保迭代器 I 指向有效的存储空间
  assert(
      this->isReferenceToStorage(I) && "Iterator to erase is out of bounds.");

  iterator N = I;
  // 将后续元素向前移动一位
  std::move(I + 1, this->end(), I);
  // 弹出最后一个元素
  this->pop_back();
  return (N);
}

/// 从向量中删除区间 [S, E) 内的元素。
iterator erase(iterator S, iterator E) {
  // 确保区间 [S, E) 在有效的存储空间内
  assert(this->isRangeInStorage(S, E) && "Range to erase is out of bounds.");

  iterator N = S;
  // 将区间 [E, end()) 内的元素向前移动到起始位置 S 处
  iterator I = std::move(E, this->end(), S);
  // 销毁多余的元素
  this->destroy_range(I, this->end());
  // 设置向量的新大小
  this->set_size(I - this->begin());
  return (N);
}

private:
/// 在迭代器 I 处插入一个元素 Elt。
template <class ArgType>
iterator insert_one_impl(iterator I, ArgType&& Elt) {
  // 调用者确保 ArgType 是 T 的派生类
  static_assert(
      std::is_same<std::remove_const_t<std::remove_reference_t<ArgType>>, T>::
          value,
      "ArgType must be derived from T!");

  if (I == this->end()) { // 对于空向量的重要特例。
    // 在向量末尾添加元素 Elt
    this->push_back(::std::forward<ArgType>(Elt));
    return this->end() - 1;
  }

  assert(
      this->isReferenceToStorage(I) &&
      "Insertion iterator is out of bounds.");

  // 如果迭代器 I 不是指向有效的存储空间，则抛出断言错误
    // 计算要插入位置的索引，相对于容器起始位置的偏移量
    size_t Index = I - this->begin();
    // 调用成员函数 reserveForParamAndGetAddress，为参数 Elt 预留空间并获取其地址
    std::remove_reference_t<ArgType>* EltPtr =
        this->reserveForParamAndGetAddress(Elt);
    // 将迭代器 I 更新为插入位置的迭代器
    I = this->begin() + Index;

    // 将最后一个元素移动到容器末尾，空出插入位置
    ::new ((void*)this->end()) T(::std::move(this->back()));
    // 向后移动容器中插入位置之后的所有元素，为新元素腾出空间
    std::move_backward(I, this->end() - 1, this->end());
    // 增加容器的大小
    this->set_size(this->size() + 1);

    // 如果刚刚移动的元素是要插入的元素，需要更新引用（如果不是按值传递的情况下才会发生）
    // 使用 static_assert 确保在按值传递的情况下，ArgType 必须是 'T' 类型
    static_assert(
        !TakesParamByValue || std::is_same<ArgType, T>::value,
        "ArgType must be 'T' when taking by value!");
    // 如果 EltPtr 指向的范围在插入位置 I 和容器末尾之间，则更新 EltPtr
    if (!TakesParamByValue && this->isReferenceToRange(EltPtr, I, this->end()))
      ++EltPtr;

    // 将参数 Elt 转发给插入位置 I 处的元素，并返回更新后的迭代器 I
    *I = ::std::forward<ArgType>(*EltPtr);
    return I;
  }

 public:
  // 在迭代器 I 处插入右值引用类型的元素 Elt
  iterator insert(iterator I, T&& Elt) {
    return insert_one_impl(I, this->forward_value_param(std::move(Elt)));
  }

  // 在迭代器 I 处插入常量引用类型的元素 Elt
  iterator insert(iterator I, const T& Elt) {
    return insert_one_impl(I, this->forward_value_param(Elt));
  }

  // 在迭代器 I 处插入 NumToInsert 个值为 Elt 的元素
  iterator insert(iterator I, size_type NumToInsert, ValueParamT Elt) {
    // 将迭代器 I 转换为插入元素位置的索引，避免在调用 reserve() 时使迭代器无效
    size_t InsertElt = I - this->begin();

    // 对于空向量的特殊情况，直接追加元素并返回插入位置的迭代器
    if (I == this->end()) {
      append(NumToInsert, Elt);
      return this->begin() + InsertElt;
    }

    // 断言：确保插入位置的迭代器处于有效范围内
    assert(
        this->isReferenceToStorage(I) &&
        "Insertion iterator is out of bounds.");

    // 确保有足够的空间，并获取参数 Elt 的地址（可能已更新）
    const T* EltPtr = this->reserveForParamAndGetAddress(Elt, NumToInsert);

    // 将迭代器 I 更新为插入元素位置的迭代器
    I = this->begin() + InsertElt;

    // 如果插入点之后的元素数量大于等于要插入的元素数量，使用简单的插入方法
    if (size_t(this->end() - I) >= NumToInsert) {
      // 记录旧的结束迭代器
      T* OldEnd = this->end();
      // 使用移动迭代器追加要插入的元素
      append(
          std::move_iterator<iterator>(this->end() - NumToInsert),
          std::move_iterator<iterator>(this->end()));

      // 将被替换的现有元素复制到新位置
      std::move_backward(I, OldEnd - NumToInsert, OldEnd);

      // 如果移动的是要插入的元素，则更新引用
      if (!TakesParamByValue && I <= EltPtr && EltPtr < this->end())
        EltPtr += NumToInsert;

      // 使用参数 EltPtr 填充插入位置开始的 NumToInsert 个元素
      std::fill_n(I, NumToInsert, *EltPtr);
      return I;
    }

    // 否则，要插入的元素数量超过了已有元素的数量，并且不是插入到末尾

    // 移动要被覆盖的元素
    T* OldEnd = this->end();
    // 增加容器的大小
    this->set_size(this->size() + NumToInsert);
    // 计算将要被覆盖的元素数量
    size_t NumOverwritten = OldEnd - I;
    // 使用未初始化的移动操作将现有元素移动到新的位置
    this->uninitialized_move(I, OldEnd, this->end() - NumOverwritten);
    // 如果我们刚刚移动了要插入的元素，则确保更新引用（如果 TakesParamByValue，则不会发生）
    if (!TakesParamByValue && I <= EltPtr && EltPtr < this->end())
      EltPtr += NumToInsert;

    // 替换被覆盖的部分
    std::fill_n(I, NumOverwritten, *EltPtr);

    // 插入未被覆盖的中间部分
    std::uninitialized_fill_n(OldEnd, NumToInsert - NumOverwritten, *EltPtr);
    return I;
  }

  template <
      typename ItTy,
      typename = std::enable_if_t<std::is_convertible_v<
          typename std::iterator_traits<ItTy>::iterator_category,
          std::input_iterator_tag>>>
  // 插入范围为 [From, To) 的元素
  iterator insert(iterator I, ItTy From, ItTy To) {
    // 将迭代器转换为元素索引，以避免在 reserve() 时使迭代器失效
    size_t InsertElt = I - this->begin();

    if (I == this->end()) { // 对空向量的重要特殊情况处理
      // 在尾部追加元素
      append(From, To);
      return this->begin() + InsertElt;
    }

    // 断言插入迭代器位于有效范围内
    assert(
        this->isReferenceToStorage(I) &&
        "Insertion iterator is out of bounds.");

    // 检查 reserve 操作是否会使迭代器失效
    this->assertSafeToAddRange(From, To);

    size_t NumToInsert = std::distance(From, To);

    // 确保有足够的空间
    reserve(this->size() + NumToInsert);

    // 恢复迭代器的有效性
    I = this->begin() + InsertElt;

    // 如果插入点到末尾的元素个数大于等于要插入的元素个数，可以简单插入
    if (size_t(this->end() - I) >= NumToInsert) {
      T* OldEnd = this->end();
      // 将现有元素向后移动，为新元素腾出空间
      append(
          std::move_iterator<iterator>(this->end() - NumToInsert),
          std::move_iterator<iterator>(this->end()));

      // 复制被替换的现有元素
      std::move_backward(I, OldEnd - NumToInsert, OldEnd);

      // 复制插入的新元素
      std::copy(From, To, I);
      return I;
    }

    // 否则，要插入的元素多于已有元素，并且不是在末尾插入

    // 移动将要被覆盖的元素
    T* OldEnd = this->end();
    this->set_size(this->size() + NumToInsert);
    size_t NumOverwritten = OldEnd - I;
    this->uninitialized_move(I, OldEnd, this->end() - NumOverwritten);

    // 替换被覆盖的部分
    for (T* J = I; NumOverwritten > 0; --NumOverwritten) {
      *J = *From;
      ++J;
      ++From;
    }

    // 插入未被覆盖的中间部分
    this->uninitialized_copy(From, To, OldEnd);
    return I;
  }

  // 插入 initializer_list 中的元素
  void insert(iterator I, std::initializer_list<T> IL) {
    insert(I, IL.begin(), IL.end());
  }

  // 在 vector 尾部就地构造元素
  template <typename... ArgTypes>
  reference emplace_back(ArgTypes&&... Args) {
    // 如果大小超出容量，则增长容量并构造元素
    if (C10_UNLIKELY(this->size() >= this->capacity()))
      return this->growAndEmplaceBack(std::forward<ArgTypes>(Args)...);
    ::new ((void*)this->end()) T(std::forward<ArgTypes>(Args)...);
    // 在 SmallVector 的末尾构造一个新元素，使用完美转发参数 Args 构造类型 T 的对象
    this->set_size(this->size() + 1);
    // 增加 SmallVector 的大小，因为添加了一个新元素
    return this->back();
    // 返回刚刚添加的元素的引用作为结果

  }

  SmallVectorImpl& operator=(const SmallVectorImpl& RHS);

  SmallVectorImpl& operator=(SmallVectorImpl&& RHS) noexcept(
      std::is_nothrow_move_constructible_v<T> &&
      std::is_nothrow_destructible_v<T>);
  // 移动赋值运算符，从 RHS 移动构造 SmallVectorImpl 的内容到当前对象

  bool operator==(const SmallVectorImpl& RHS) const {
    // 比较运算符，检查当前对象与 RHS 是否相等
    if (this->size() != RHS.size())
      return false;
    return std::equal(this->begin(), this->end(), RHS.begin());
    // 使用 std::equal 比较两个 SmallVectorImpl 对象的元素是否相等
  }
  bool operator!=(const SmallVectorImpl& RHS) const {
    // 不等于运算符，利用相等运算符的结果取反
    return !(*this == RHS);
  }

  bool operator<(const SmallVectorImpl& RHS) const {
    // 小于运算符，使用 std::lexicographical_compare 比较当前对象与 RHS 的元素大小
    return std::lexicographical_compare(
        this->begin(), this->end(), RHS.begin(), RHS.end());
  }
};

template <typename T>
void SmallVectorImpl<T>::swap(SmallVectorImpl<T>& RHS) noexcept {
  // 如果两个对象是同一个，则直接返回，无需交换
  if (this == &RHS)
    return;

  // 只有当两个向量都不是小向量时才能避免复制元素
  if (!this->isSmall() && !RHS.isSmall()) {
    // 交换非小向量的元素
    std::swap(this->BeginX, RHS.BeginX);
    std::swap(this->Size, RHS.Size);
    std::swap(this->Capacity, RHS.Capacity);
    return;
  }
  // 保证两个向量都有足够的空间以进行交换
  this->reserve(RHS.size());
  RHS.reserve(this->size());

  // 交换共享的元素
  size_t NumShared = this->size();
  if (NumShared > RHS.size())
    NumShared = RHS.size();
  for (size_type i = 0; i != NumShared; ++i)
    std::swap((*this)[i], RHS[i]);

  // 复制额外的元素
  if (this->size() > RHS.size()) {
    size_t EltDiff = this->size() - RHS.size();
    // 在 RHS 的末尾构造未初始化的剩余元素
    this->uninitialized_copy(this->begin() + NumShared, this->end(), RHS.end());
    // 调整 RHS 的大小以反映添加的元素
    RHS.set_size(RHS.size() + EltDiff);
    // 销毁未初始化的额外元素
    this->destroy_range(this->begin() + NumShared, this->end());
    // 调整 this 的大小以反映剩余的共享元素
    this->set_size(NumShared);
  } else if (RHS.size() > this->size()) {
    size_t EltDiff = RHS.size() - this->size();
    // 在 this 的末尾构造未初始化的剩余元素
    this->uninitialized_copy(RHS.begin() + NumShared, RHS.end(), this->end());
    // 调整 this 的大小以反映添加的元素
    this->set_size(this->size() + EltDiff);
    // 销毁未初始化的额外元素
    this->destroy_range(RHS.begin() + NumShared, RHS.end());
    // 调整 RHS 的大小以反映剩余的共享元素
    RHS.set_size(NumShared);
  }
}

template <typename T>
SmallVectorImpl<T>& SmallVectorImpl<T>::operator=(
    const SmallVectorImpl<T>& RHS) {
  // 避免自我赋值
  if (this == &RHS)
    return *this;

  // 如果当前向量有足够的空间，先赋值共同的元素，然后销毁多余的元素
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // 赋值共同的元素
    iterator NewEnd;
    if (RHSSize)
      NewEnd = std::copy(RHS.begin(), RHS.begin() + RHSSize, this->begin());
    else
      NewEnd = this->begin();

    // 销毁多余的元素
    this->destroy_range(NewEnd, this->end());

    // 调整大小
    this->set_size(RHSSize);
    return *this;
  }

  // 如果需要增加空间以容纳足够的元素，先销毁当前元素
  if (this->capacity() < RHSSize) {
    // 销毁当前元素
    this->clear();
    CurSize = 0;
    // 增加足够的空间
    this->grow(RHSSize);
  } else if (CurSize) {
    // 否则，使用赋值运算符来赋值已构造的元素
    std::copy(RHS.begin(), RHS.begin() + CurSize, this->begin());
  }

  // 在现有位置构造新元素
  this->uninitialized_copy(
      RHS.begin() + CurSize, RHS.end(), this->begin() + CurSize);

  // 设置尾部位置
  this->set_size(RHSSize);
  return *this;
}

template <typename T>
SmallVectorImpl<T>& SmallVectorImpl<T>::
operator=(SmallVectorImpl<T>&& RHS) noexcept(
    std::is_nothrow_move_constructible_v<T> &&
    std::is_nothrow_destructible_v<T>) {
  // 避免自我赋值
  if (this == &RHS)
    return *this;


这段代码是一个 C++ 的模板类 `SmallVectorImpl` 的成员函数实现，用于实现小型向量的交换和赋值操作。
    return *this;

  // 如果右操作数（RHS）不是小对象，清空当前对象并窃取其缓冲区。
  if (!RHS.isSmall()) {
    this->destroy_range(this->begin(), this->end());
    if (!this->isSmall())
      free(this->begin());
    this->BeginX = RHS.BeginX;
    this->Size = RHS.Size;
    this->Capacity = RHS.Capacity;
    RHS.resetToSmall();
    return *this;
  }

  // 如果当前对象已有足够空间，赋值共同的元素，然后销毁多余的元素。
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // 赋值共同的元素。
    iterator NewEnd = this->begin();
    if (RHSSize)
      NewEnd = std::move(RHS.begin(), RHS.end(), NewEnd);

    // 销毁多余的元素并调整边界。
    this->destroy_range(NewEnd, this->end());
    this->set_size(RHSSize);

    // 清空 RHS。
    RHS.clear();

    return *this;
  }

  // 如果需要增长以满足足够的元素数量，销毁当前元素。
  // 这样可以在增长过程中避免复制它们。
  // FIXME: 如果我们可以高效移动元素，这种做法可能并不合理。
  if (this->capacity() < RHSSize) {
    // 销毁当前元素。
    this->clear();
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // 否则，对已构造的元素使用赋值。
    std::move(RHS.begin(), RHS.begin() + CurSize, this->begin());
  }

  // 在原地移动构造新的元素。
  this->uninitialized_move(
      RHS.begin() + CurSize, RHS.end(), this->begin() + CurSize);

  // 设置结束位置。
  this->set_size(RHSSize);

  // 清空 RHS。
  RHS.clear();
  return *this;
/// Storage for the SmallVector elements.  This is specialized for the N=0 case
/// to avoid allocating unnecessary storage.
template <typename T, unsigned N>
struct SmallVectorStorage {
  alignas(T) char InlineElts[N * sizeof(T)];  // Inline storage for elements of type T
};

/// We need the storage to be properly aligned even for small-size of 0 so that
/// the pointer math in \a SmallVectorTemplateCommon::getFirstEl() is
/// well-defined.
template <typename T>
struct alignas(T) SmallVectorStorage<T, 0> {};  // Specialization for zero-size case

/// Forward declaration of SmallVector so that
/// calculateSmallVectorDefaultInlinedElements can reference
/// `sizeof(SmallVector<T, 0>)`.
template <typename T, unsigned N>
class /* LLVM_GSL_OWNER */ SmallVector;

/// Helper class for calculating the default number of inline elements for
/// `SmallVector<T>`.
///
/// This should be migrated to a constexpr function when our minimum
/// compiler support is enough for multi-statement constexpr functions.
template <typename T>
struct CalculateSmallVectorDefaultInlinedElements {
  // Parameter controlling the default number of inlined elements
  // for `SmallVector<T>`.
  //
  // The default number of inlined elements ensures that
  // 1. There is at least one inlined element.
  // 2. `sizeof(SmallVector<T>) <= kPreferredSmallVectorSizeof` unless
  // it contradicts 1.
  static constexpr size_t kPreferredSmallVectorSizeof = 64;

  // static_assert that sizeof(T) is not "too big".
  //
  // Because our policy guarantees at least one inlined element, it is possible
  // for an arbitrarily large inlined element to allocate an arbitrarily large
  // amount of inline storage. We generally consider it an antipattern for a
  // SmallVector to allocate an excessive amount of inline storage, so we want
  // to call attention to these cases and make sure that users are making an
  // intentional decision if they request a lot of inline storage.
  //
  // We want this assertion to trigger in pathological cases, but otherwise
  // not be too easy to hit. To accomplish that, the cutoff is actually somewhat
  // larger than kPreferredSmallVectorSizeof (otherwise,
  // `SmallVector<SmallVector<T>>` would be one easy way to trip it, and that
  // pattern seems useful in practice).
  //
  // One wrinkle is that this assertion is in theory non-portable, since
  // sizeof(T) is in general platform-dependent. However, we don't expect this
  // to be much of an issue, because most LLVM development happens on 64-bit
  // hosts, and therefore sizeof(T) is expected to *decrease* when compiled for
  // 32-bit hosts, dodging the issue. The reverse situation, where development
  // happens on a 32-bit host and then fails due to sizeof(T) *increasing* on a
  // 64-bit host, is expected to be very rare.
  static_assert(
      sizeof(T) <= 256,
      "You are trying to use a default number of inlined elements for "
      "`SmallVector<T>` but `sizeof(T)` is really big! Please use an "
      "explicit number of inlined elements with `SmallVector<T, N>` to make "
      "sure you really want that much inline storage.");

  // Discount the size of the header itself when calculating the maximum inline
  // bytes.
  static constexpr size_t PreferredInlineBytes =
      kPreferredSmallVectorSizeof - sizeof(SmallVector<T, 0>);
  // Calculate the number of elements of type T that can fit within the preferred
  // inline storage size.
  static constexpr size_t NumElementsThatFit = PreferredInlineBytes / sizeof(T);
  // The final value represents the default number of inlined elements for the
  // `SmallVector<T>`.
  static constexpr size_t value =
      NumElementsThatFit == 0 ? 1 : NumElementsThatFit;
};

/// This is a 'vector' (really, a variable-sized array), optimized
/// for the case when the array is small.  It contains some number of elements
/// in-place, which allows it to avoid heap allocation when the actual number of
/// elements is below that threshold.  This allows normal "small" cases to be
/// fast without losing generality for large inputs.
///
/// \note
/// In the absence of a well-motivated choice for the number of inlined
/// elements \p N, it is recommended to use \c SmallVector<T> (that is,
/// ```
/// SmallVector 类模板用于实现一个可变大小的数组，它在内存小于等于 N 时使用栈上空间，而不进行堆分配。
/// 这个类继承自 SmallVectorImpl<T> 和 SmallVectorStorage<T, N>。
template <
    typename T,
    unsigned N = CalculateSmallVectorDefaultInlinedElements<T>::value>
class /* LLVM_GSL_OWNER */ SmallVector : public SmallVectorImpl<T>,
                                         SmallVectorStorage<T, N> {
 public:
  /// 默认构造函数，创建一个空的 SmallVector 对象，使用 N 作为预分配空间大小。
  SmallVector() : SmallVectorImpl<T>(N) {}

  /// 析构函数，销毁 SmallVector 中的所有元素。
  ~SmallVector() {
    // Destroy the constructed elements in the vector.
    this->destroy_range(this->begin(), this->end());
  }

  /// 根据指定大小和初始值构造 SmallVector 对象。
  explicit SmallVector(size_t Size, const T& Value = T())
      : SmallVectorImpl<T>(N) {
    this->assign(Size, Value);
  }

  /// 根据迭代器范围 [S, E) 构造 SmallVector 对象。
  template <
      typename ItTy,
      typename = std::enable_if_t<std::is_convertible_v<
          typename std::iterator_traits<ItTy>::iterator_category,
          std::input_iterator_tag>>>
  SmallVector(ItTy S, ItTy E) : SmallVectorImpl<T>(N) {
    this->append(S, E);
  }

  /// 根据容器类型 Container 构造 SmallVector 对象，要求 Container 有有效的 .begin() 和 .end() 方法。
  /// 使用 std::enable_if 确保 Container 是输入迭代器类型。
  template <
      typename Container,
      std::enable_if_t<
          std::is_convertible_v<
              typename std::iterator_traits<
                  decltype(std::declval<Container>()
                               .begin())>::iterator_category,
              std::input_iterator_tag> &&
              std::is_convertible_v<
                  typename std::iterator_traits<
                      decltype(std::declval<Container>()
                                   .end())>::iterator_category,
                  std::input_iterator_tag>,
          int> = 0>
  explicit SmallVector(Container&& c) : SmallVectorImpl<T>(N) {
    this->append(c.begin(), c.end());
  }

  /// 使用初始化列表 IL 构造 SmallVector 对象。
  SmallVector(std::initializer_list<T> IL) : SmallVectorImpl<T>(N) {
    this->assign(IL);
  }

  /// 拷贝构造函数，从另一个 SmallVector 对象 RHS 拷贝构造。
  SmallVector(const SmallVector& RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(RHS);
  }

  /// 拷贝赋值运算符，从另一个 SmallVector 对象 RHS 拷贝赋值。
  SmallVector& operator=(const SmallVector& RHS) {
    SmallVectorImpl<T>::operator=(RHS);
    return *this;
  }

  /// 移动构造函数，从另一个 SmallVector 对象 RHS 移动构造。
  /// 使用 noexcept 保证在移动过程中不会抛出异常。
  SmallVector(SmallVector&& RHS) noexcept(
      std::is_nothrow_move_assignable_v<SmallVectorImpl<T>>)
      : SmallVectorImpl<T>(N) {
  // 如果 RHS 非空，则调用基类的 move 赋值操作符
  if (!RHS.empty())
    SmallVectorImpl<T>::operator=(::std::move(RHS));
}

// 注意：enable_if 限制了 Container 的类型，要求其具有返回有效输入迭代器的 .begin() 和 .end() 函数
template <
    typename Container,
    std::enable_if_t<
        std::is_convertible_v<
            typename std::iterator_traits<
                decltype(std::declval<Container>()
                             .begin())>::iterator_category,
            std::input_iterator_tag> &&
            std::is_convertible_v<
                typename std::iterator_traits<
                    decltype(std::declval<Container>()
                                 .end())>::iterator_category,
                std::input_iterator_tag>,
        int> = 0>
// 将 Container 的内容赋值给当前 SmallVector 对象
SmallVector& operator=(const Container& RHS) {
  this->assign(RHS.begin(), RHS.end());
  return *this;
}

// 移动构造函数，从 RHS 移动数据到当前对象，如果 RHS 非空
SmallVector(SmallVectorImpl<T>&& RHS) noexcept(
    std::is_nothrow_move_assignable_v<SmallVectorImpl<T>>)
    : SmallVectorImpl<T>(N) {
  if (!RHS.empty())
    SmallVectorImpl<T>::operator=(::std::move(RHS));
}

// 移动赋值操作符，从 RHS 移动数据到当前对象
SmallVector& operator=(SmallVector&& RHS) noexcept(
    std::is_nothrow_move_assignable_v<SmallVectorImpl<T>>) {
  SmallVectorImpl<T>::operator=(::std::move(RHS));
  return *this;
}

// 移动赋值操作符，从 RHS 移动数据到当前对象
SmallVector& operator=(SmallVectorImpl<T>&& RHS) noexcept(
    std::is_nothrow_move_constructible_v<SmallVectorImpl<T>>) {
  SmallVectorImpl<T>::operator=(::std::move(RHS));
  return *this;
}

// 注意：enable_if 限制了 Container 的类型，要求其具有返回有效输入迭代器的 .begin() 和 .end() 函数
template <
    typename Container,
    std::enable_if_t<
        std::is_convertible_v<
            typename std::iterator_traits<
                decltype(std::declval<Container>()
                             .begin())>::iterator_category,
            std::input_iterator_tag> &&
            std::is_convertible_v<
                typename std::iterator_traits<
                    decltype(std::declval<Container>()
                                 .end())>::iterator_category,
                std::input_iterator_tag>,
        int> = 0>
// 将 Container 的内容移动赋值给当前 SmallVector 对象
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
SmallVector& operator=(Container&& C) {
  this->assign(C.begin(), C.end());
  return *this;
}

// 使用初始化列表 IL 赋值给当前 SmallVector 对象
SmallVector& operator=(std::initializer_list<T> IL) {
  this->assign(IL);
  return *this;
}
};

/// 模板函数：计算 SmallVector 的容量大小（以字节为单位）
template <typename T, unsigned N>
inline size_t capacity_in_bytes(const SmallVector<T, N>& X) {
  return X.capacity_in_bytes();
}

/// 重载操作符<<：将 SmallVector 内容输出到输出流
template <typename T, unsigned N>
std::ostream& operator<<(std::ostream& out, const SmallVector<T, N>& list) {
  int i = 0;
  out << "[";
  // 遍历 SmallVector 中的元素，并输出到流中
  for (auto e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "]";
  return out;
}

/// 模板别名：从 RangeType 中获取值类型
template <typename RangeType>
using ValueTypeFromRangeType = std::remove_const_t<
    std::remove_reference_t<decltype(*std::begin(std::declval<RangeType&>()))>>;

/// 函数模板：将范围 Range 转换为 SmallVector
/// 用法示例：当你需要遍历一个范围并对结果进行排序时很有用。
template <unsigned Size, typename R>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
SmallVector<ValueTypeFromRangeType<R>, Size> to_vector(R&& Range) {
  return {std::begin(Range), std::end(Range)};
}

/// 函数模板重载：将范围 Range 转换为 SmallVector，使用默认大小
template <typename R>
SmallVector<
    ValueTypeFromRangeType<R>,
    CalculateSmallVectorDefaultInlinedElements<
        ValueTypeFromRangeType<R>>::value>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
to_vector(R&& Range) {
  return {std::begin(Range), std::end(Range)};
}

} // end namespace c10

namespace std {

/// 实现 std::swap，利用 SmallVector 的 swap 函数
template <typename T>
inline void swap(
    c10::SmallVectorImpl<T>& LHS,
    c10::SmallVectorImpl<T>& RHS) noexcept {
  LHS.swap(RHS);
}

/// 实现 std::swap，利用 SmallVector 的 swap 函数
template <typename T, unsigned N>
inline void swap(
    c10::SmallVector<T, N>& LHS,
    c10::SmallVector<T, N>& RHS) noexcept {
  LHS.swap(RHS);
}

} // end namespace std
```