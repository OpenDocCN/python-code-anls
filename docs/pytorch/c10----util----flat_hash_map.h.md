# `.\pytorch\c10\util\flat_hash_map.h`

```py
// Taken from
// https://github.com/skarupke/flat_hash_map/blob/2c4687431f978f02a3780e24b8b701d22aa32d9c/flat_hash_map.hpp
// with fixes applied:
// - https://github.com/skarupke/flat_hash_map/pull/25
// - https://github.com/skarupke/flat_hash_map/pull/26
// - replace size_t with uint64_t to fix it for 32bit
// - add "GCC diagnostic" pragma to ignore -Wshadow
// - make sherwood_v3_table::convertible_to_iterator public because GCC5 seems
// to have issues with it otherwise
// - fix compiler warnings in operator templated_iterator<const value_type>
// - make use of 'if constexpr' and eliminate AssignIfTrue template

// Copyright Malte Skarupke 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See http://www.boost.org/LICENSE_1_0.txt)

// Include necessary headers for the implementation
#pragma once
#include <c10/macros/Macros.h>  // Include macros from c10 library
#include <algorithm>             // Standard library algorithm functions
#include <cmath>                 // Mathematical functions
#include <cstddef>               // Standard definitions like size_t
#include <cstdint>               // Integer types like uint64_t
#include <functional>            // Function objects
#include <iterator>              // Iterator tags and operations
#include <stdexcept>             // Standard exceptions
#include <type_traits>           // Type traits
#include <utility>               // Utility components

// Suppress certain warnings specific to Clang
C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

// Suppress warning 4624 for MSVC, related to implicitly defined destructors
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4624)  // destructor was implicitly defined as deleted
#endif

// Define SKA_NOINLINE macro for noinline attribute based on compiler type
#ifdef _MSC_VER
#define SKA_NOINLINE(...) __declspec(noinline) __VA_ARGS__
#else
#define SKA_NOINLINE(...) __VA_ARGS__ __attribute__((noinline))
#endif

// Namespace ska and its hash policy structs
namespace ska {

// Forward declarations of hash policy structs
struct prime_number_hash_policy;
struct power_of_two_hash_policy;
struct fibonacci_hash_policy;

// Detail namespace containing internal implementation details
namespace detailv3 {

// Template for storing functors with result type Result
template <typename Result, typename Functor>
struct functor_storage : Functor {
  functor_storage() = default;
  functor_storage(const Functor& functor) : Functor(functor) {}

  // Operator overloads for calling the functor
  template <typename... Args>
  Result operator()(Args&&... args) {
    return static_cast<Functor&>(*this)(std::forward<Args>(args)...);
  }

  template <typename... Args>
  Result operator()(Args&&... args) const {
    return static_cast<const Functor&>(*this)(std::forward<Args>(args)...);
  }
};

// Specialization of functor_storage for function pointers
template <typename Result, typename... Args>
struct functor_storage<Result, Result (*)(Args...)> {
  typedef Result (*function_ptr)(Args...);
  function_ptr function;

  // Constructor initializing with a function pointer
  functor_storage(function_ptr function) : function(function) {}

  // Operator overload for calling the stored function pointer
  Result operator()(Args... args) const {
    return function(std::forward<Args>(args)...);
  }

  // Conversion operators to function_ptr for mutable and const access
  operator function_ptr&() {
    return function;
  }

  operator const function_ptr&() {
    return function;
  }
};

// Template for hashing keys or values using a hasher functor
template <typename key_type, typename value_type, typename hasher>
struct KeyOrValueHasher : functor_storage<uint64_t, hasher> {
  typedef functor_storage<uint64_t, hasher> hasher_storage;

  KeyOrValueHasher() = default;
  KeyOrValueHasher(const hasher& hash) : hasher_storage(hash) {}

  // Operator overload for hashing a key
  uint64_t operator()(const key_type& key) {
    return static_cast<hasher_storage&>(*this)(key);
  }

  // Const version of the operator overload for hashing a key
  uint64_t operator()(const key_type& key) const {
    return static_cast<const hasher_storage&>(*this)(key);
  }
};
    // 返回给定键的哈希值
    return static_cast<const hasher_storage&>(*this)(key);
  }
  // 返回给定值的哈希值（用于 std::pair）
  uint64_t operator()(const value_type& value) {
    return static_cast<hasher_storage&>(*this)(value.first);
  }
  // 返回给定值的哈希值的常量版本（用于 std::pair）
  uint64_t operator()(const value_type& value) const {
    return static_cast<const hasher_storage&>(*this)(value.first);
  }
  // 返回给定 std::pair 的第一个元素的哈希值
  template <typename F, typename S>
  uint64_t operator()(const std::pair<F, S>& value) {
    return static_cast<hasher_storage&>(*this)(value.first);
  }
  // 返回给定 std::pair 的第一个元素的哈希值的常量版本
  template <typename F, typename S>
  uint64_t operator()(const std::pair<F, S>& value) const {
    return static_cast<const hasher_storage&>(*this)(value.first);
  }
};
// 结束结构体定义

template <typename key_type, typename value_type, typename key_equal>
struct KeyOrValueEquality : functor_storage<bool, key_equal> {
  // 使用 functor_storage 来定义 KeyOrValueEquality 结构体，其中 key_equal 是比较器类型

  typedef functor_storage<bool, key_equal> equality_storage;
  // 定义一个名为 equality_storage 的类型别名，它继承自 functor_storage<bool, key_equal>

  KeyOrValueEquality() = default;
  // 默认构造函数

  KeyOrValueEquality(const key_equal& equality) : equality_storage(equality) {}
  // 构造函数，接受一个 key_equal 类型的参数，并初始化 equality_storage

  bool operator()(const key_type& lhs, const key_type& rhs) {
    return static_cast<equality_storage&>(*this)(lhs, rhs);
  }
  // 重载函数调用运算符，用于比较两个 key_type 类型的对象

  bool operator()(const key_type& lhs, const value_type& rhs) {
    return static_cast<equality_storage&>(*this)(lhs, rhs.first);
  }
  // 重载函数调用运算符，用于比较 key_type 类型的对象和 value_type 类型的对象

  bool operator()(const value_type& lhs, const key_type& rhs) {
    return static_cast<equality_storage&>(*this)(lhs.first, rhs);
  }
  // 重载函数调用运算符，用于比较 value_type 类型的对象和 key_type 类型的对象

  bool operator()(const value_type& lhs, const value_type& rhs) {
    return static_cast<equality_storage&>(*this)(lhs.first, rhs.first);
  }
  // 重载函数调用运算符，用于比较两个 value_type 类型的对象

  template <typename F, typename S>
  bool operator()(const key_type& lhs, const std::pair<F, S>& rhs) {
    return static_cast<equality_storage&>(*this)(lhs, rhs.first);
  }
  // 重载函数调用运算符，用于比较 key_type 类型的对象和 std::pair 类型的对象

  template <typename F, typename S>
  bool operator()(const std::pair<F, S>& lhs, const key_type& rhs) {
    return static_cast<equality_storage&>(*this)(lhs.first, rhs);
  }
  // 重载函数调用运算符，用于比较 std::pair 类型的对象和 key_type 类型的对象

  template <typename F, typename S>
  bool operator()(const value_type& lhs, const std::pair<F, S>& rhs) {
    return static_cast<equality_storage&>(*this)(lhs.first, rhs.first);
  }
  // 重载函数调用运算符，用于比较 value_type 类型的对象和 std::pair 类型的对象

  template <typename F, typename S>
  bool operator()(const std::pair<F, S>& lhs, const value_type& rhs) {
    return static_cast<equality_storage&>(*this)(lhs.first, rhs.first);
  }
  // 重载函数调用运算符，用于比较 std::pair 类型的对象和 value_type 类型的对象

  template <typename FL, typename SL, typename FR, typename SR>
  bool operator()(const std::pair<FL, SL>& lhs, const std::pair<FR, SR>& rhs) {
    return static_cast<equality_storage&>(*this)(lhs.first, rhs.first);
  }
  // 重载函数调用运算符，用于比较两个 std::pair 类型的对象

};
// 结构体 KeyOrValueEquality 的定义结束

static constexpr int8_t min_lookups = 4;
// 定义一个静态常量 min_lookups，类型为 int8_t，值为 4

template <typename T>
struct sherwood_v3_entry {
  // 定义一个模板结构体 sherwood_v3_entry，模板参数为类型 T

  sherwood_v3_entry() = default;
  // 默认构造函数

  sherwood_v3_entry(int8_t distance_from_desired)
      : distance_from_desired(distance_from_desired) {}
  // 构造函数，接受一个 int8_t 类型的参数 distance_from_desired，并初始化成员变量

  ~sherwood_v3_entry() = default;
  // 默认析构函数

  bool has_value() const {
    return distance_from_desired >= 0;
  }
  // 判断是否有值的成员函数，返回 distance_from_desired 是否大于等于 0

  bool is_empty() const {
    return distance_from_desired < 0;
  }
  // 判断是否为空的成员函数，返回 distance_from_desired 是否小于 0

  bool is_at_desired_position() const {
    return distance_from_desired <= 0;
  }
  // 判断是否在期望位置的成员函数，返回 distance_from_desired 是否小于等于 0

  template <typename... Args>
  void emplace(int8_t distance, Args&&... args) {
    new (std::addressof(value)) T(std::forward<Args>(args)...);
    distance_from_desired = distance;
  }
  // 模板成员函数 emplace，用于在指定位置构造对象，接受参数 distance 和 Args&&...

  void destroy_value() {
    value.~T();
    distance_from_desired = -1;
  }
  // 成员函数 destroy_value，用于销毁对象并重置 distance_from_desired

  int8_t distance_from_desired = -1;
  // 成员变量 distance_from_desired，默认初始化为 -1

  static constexpr int8_t special_end_value = 0;
  // 静态常量 special_end_value，类型为 int8_t，值为 0

  union {
    T value;
  };
  // 匿名联合体，成员变量 value 的类型为 T
};
inline int8_t log2(uint64_t value) {
  // NOLINTNEXTLINE(*c-arrays*)
  // 预先计算的log2表，以查找最高位的1对应的索引
  static constexpr int8_t table[64] = {
      63, 0,  58, 1,  59, 47, 53, 2,  60, 39, 48, 27, 54, 33, 42, 3,
      61, 51, 37, 40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4,
      62, 57, 46, 52, 38, 26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21,
      56, 45, 25, 31, 35, 16, 9,  12, 44, 24, 15, 8,  23, 7,  6,  5};
  // 将value的所有低位设置为1，从而找到value中最高位的1的索引
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  value |= value >> 32;
  // 根据预先计算的表，返回最高位的1对应的索引
  return table[((value - (value >> 1)) * 0x07EDD5E59A4E28C2) >> 58];
}

inline uint64_t next_power_of_two(uint64_t i) {
  --i;
  // 将i的所有低位设置为1，得到比i大且是2的幂次方的数
  i |= i >> 1;
  i |= i >> 2;
  i |= i >> 4;
  i |= i >> 8;
  i |= i >> 16;
  i |= i >> 32;
  ++i;
  return i;
}

// 从 http://en.cppreference.com/w/cpp/types/void_t 获取的实现
// 适用于考虑CWG1558并且兼容旧编译器
template <typename... Ts>
struct make_void {
  typedef void type;
};
template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

// 根据T是否有hash_policy来选择哈希策略
template <typename T, typename = void>
struct HashPolicySelector {
  typedef fibonacci_hash_policy type;
};
template <typename T>
struct HashPolicySelector<T, void_t<typename T::hash_policy>> {
  typedef typename T::hash_policy type;
};

// 哈希表的实现
template <
    typename T,
    typename FindKey,
    typename ArgumentHash,
    typename DetailHasher,
    typename ArgumentEqual,
    typename Equal,
    typename ArgumentAlloc,
    typename EntryAlloc>
class sherwood_v3_table : private EntryAlloc,
                          private DetailHasher,
                          private Equal {
  using Entry = detailv3::sherwood_v3_entry<T>;
  using AllocatorTraits = std::allocator_traits<EntryAlloc>;
  using EntryPointer = typename AllocatorTraits::pointer;

 public:
  struct convertible_to_iterator;

  using value_type = T;
  using size_type = uint64_t;
  using difference_type = std::ptrdiff_t;
  using hasher = ArgumentHash;
  using key_equal = ArgumentEqual;
  using allocator_type = EntryAlloc;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;

  sherwood_v3_table() = default;
  // 构造函数，初始化哈希表
  explicit sherwood_v3_table(
      size_type bucket_count,
      const ArgumentHash& hash = ArgumentHash(),
      const ArgumentEqual& equal = ArgumentEqual(),
      const ArgumentAlloc& alloc = ArgumentAlloc())
      : EntryAlloc(alloc), DetailHasher(hash), Equal(equal) {
  // 调整哈希表的桶数以进行重新哈希操作
  rehash(bucket_count);
}

// 使用默认哈希和相等函数，以及指定的分配器构造哈希表
sherwood_v3_table(size_type bucket_count, const ArgumentAlloc& alloc)
    : sherwood_v3_table(
          bucket_count,
          ArgumentHash(),
          ArgumentEqual(),
          alloc) {}

// 使用指定的哈希函数、相等函数和分配器构造哈希表
sherwood_v3_table(
    size_type bucket_count,
    const ArgumentHash& hash,
    const ArgumentAlloc& alloc)
    : sherwood_v3_table(bucket_count, hash, ArgumentEqual(), alloc) {}

// 使用指定的分配器构造哈希表
explicit sherwood_v3_table(const ArgumentAlloc& alloc) : EntryAlloc(alloc) {}

// 使用迭代器范围内的元素构造哈希表，并可选择指定桶数、哈希函数、相等函数和分配器
template <typename It>
sherwood_v3_table(
    It first,
    It last,
    size_type bucket_count = 0,
    const ArgumentHash& hash = ArgumentHash(),
    const ArgumentEqual& equal = ArgumentEqual(),
    const ArgumentAlloc& alloc = ArgumentAlloc())
    : sherwood_v3_table(bucket_count, hash, equal, alloc) {
  insert(first, last);  // 将迭代器范围内的元素插入到哈希表中
}

// 使用迭代器范围内的元素和指定的桶数、分配器构造哈希表
template <typename It>
sherwood_v3_table(
    It first,
    It last,
    size_type bucket_count,
    const ArgumentAlloc& alloc)
    : sherwood_v3_table(
          first,
          last,
          bucket_count,
          ArgumentHash(),
          ArgumentEqual(),
          alloc) {}

// 使用迭代器范围内的元素和指定的桶数、哈希函数、分配器构造哈希表
template <typename It>
sherwood_v3_table(
    It first,
    It last,
    size_type bucket_count,
    const ArgumentHash& hash,
    const ArgumentAlloc& alloc)
    : sherwood_v3_table(
          first,
          last,
          bucket_count,
          hash,
          ArgumentEqual(),
          alloc) {}

// 使用初始化列表构造哈希表，并可选择指定桶数、哈希函数、相等函数和分配器
sherwood_v3_table(
    std::initializer_list<T> il,
    size_type bucket_count = 0,
    const ArgumentHash& hash = ArgumentHash(),
    const ArgumentEqual& equal = ArgumentEqual(),
    const ArgumentAlloc& alloc = ArgumentAlloc())
    : sherwood_v3_table(bucket_count, hash, equal, alloc) {
  if (bucket_count == 0)
    rehash(il.size());  // 如果桶数为0，则根据初始化列表大小重新调整哈希表桶数
  insert(il.begin(), il.end());  // 将初始化列表中的元素插入到哈希表中
}

// 使用初始化列表构造哈希表，并可选择指定桶数、分配器
sherwood_v3_table(
    std::initializer_list<T> il,
    size_type bucket_count,
    const ArgumentAlloc& alloc)
    : sherwood_v3_table(
          il,
          bucket_count,
          ArgumentHash(),
          ArgumentEqual(),
          alloc) {}

// 使用初始化列表构造哈希表，并可选择指定桶数、哈希函数、分配器
sherwood_v3_table(
    std::initializer_list<T> il,
    size_type bucket_count,
    const ArgumentHash& hash,
    const ArgumentAlloc& alloc)
    : sherwood_v3_table(il, bucket_count, hash, ArgumentEqual(), alloc) {}

// 拷贝构造函数，使用其他哈希表的副本构造当前哈希表
sherwood_v3_table(const sherwood_v3_table& other)
    : sherwood_v3_table(
          other,
          AllocatorTraits::select_on_container_copy_construction(
              other.get_allocator())) {}

// 拷贝构造函数，使用其他哈希表的副本和指定的分配器构造当前哈希表
sherwood_v3_table(const sherwood_v3_table& other, const ArgumentAlloc& alloc)
    : EntryAlloc(alloc),
      DetailHasher(other),
      Equal(other),
      _max_load_factor(other._max_load_factor) {
  rehash_for_other_container(other);  // 为其他容器重新调整哈希表
  try {
    insert(other.begin(), other.end());  // 将其他哈希表的元素插入到当前哈希表中
  } catch (...) {
    // 捕获任何异常后的处理逻辑：
    // 清空当前对象
    clear();
    // 释放数据资源
    deallocate_data(entries, num_slots_minus_one, max_lookups);
    // 将异常继续抛出
    throw;
  }
  // 移动构造函数：从另一个对象移动资源构造当前对象
  sherwood_v3_table(sherwood_v3_table&& other) noexcept
      : EntryAlloc(std::move(other)),
        DetailHasher(std::move(other)),
        Equal(std::move(other)) {
    // 交换指针和状态信息
    swap_pointers(other);
  }
  // 移动构造函数：从另一个对象移动资源构造当前对象，并使用指定的分配器
  sherwood_v3_table(
      sherwood_v3_table&& other,
      const ArgumentAlloc& alloc) noexcept
      : EntryAlloc(alloc),
        DetailHasher(std::move(other)),
        Equal(std::move(other)) {
    // 交换指针和状态信息
    swap_pointers(other);
  }
  // 拷贝赋值运算符重载：从另一个对象拷贝赋值给当前对象
  sherwood_v3_table& operator=(const sherwood_v3_table& other) {
    // 自我赋值检测
    if (this == std::addressof(other))
      return *this;

    // 清空当前对象
    clear();
    // 根据分配器属性重置状态
    if constexpr (AllocatorTraits::propagate_on_container_copy_assignment::
                      value) {
      if (static_cast<EntryAlloc&>(*this) !=
          static_cast<const EntryAlloc&>(other)) {
        reset_to_empty_state();
      }
      // 将分配器赋值为其他对象的分配器
      static_cast<EntryAlloc&>(*this) = other;
    }
    // 拷贝最大负载因子
    _max_load_factor = other._max_load_factor;
    // 拷贝哈希函数对象
    static_cast<DetailHasher&>(*this) = other;
    // 拷贝相等比较函数对象
    static_cast<Equal&>(*this) = other;
    // 根据其他容器重新哈希
    rehash_for_other_container(other);
    // 插入其他对象的元素到当前对象中
    insert(other.begin(), other.end());
    return *this;
  }
  // 移动赋值运算符重载：从另一个对象移动赋值给当前对象
  sherwood_v3_table& operator=(sherwood_v3_table&& other) noexcept {
    // 自我赋值检测
    if (this == std::addressof(other))
      return *this;
    else if constexpr (AllocatorTraits::propagate_on_container_move_assignment::
                           value) {
      // 清空当前对象
      clear();
      // 重置为空状态
      reset_to_empty_state();
      // 移动赋值分配器
      static_cast<EntryAlloc&>(*this) = std::move(other);
      // 交换指针和状态信息
      swap_pointers(other);
    } else if (
        static_cast<EntryAlloc&>(*this) == static_cast<EntryAlloc&>(other)) {
      // 仅交换指针和状态信息
      swap_pointers(other);
    } else {
      // 清空当前对象
      clear();
      // 拷贝最大负载因子
      _max_load_factor = other._max_load_factor;
      // 根据其他容器重新哈希
      rehash_for_other_container(other);
      // 将其他对象的元素移动到当前对象中
      for (T& elem : other)
        emplace(std::move(elem));
      // 清空其他对象
      other.clear();
    }
    // 移动赋值哈希函数对象
    static_cast<DetailHasher&>(*this) = std::move(other);
    // 移动赋值相等比较函数对象
    static_cast<Equal&>(*this) = std::move(other);
    return *this;
  }
  // 析构函数：清空当前对象并释放数据资源
  ~sherwood_v3_table() {
    clear();
    deallocate_data(entries, num_slots_minus_one, max_lookups);
  }

  // 获取分配器的常量引用
  const allocator_type& get_allocator() const {
    return static_cast<const allocator_type&>(*this);
  }
  // 获取键相等比较函数的常量引用
  const ArgumentEqual& key_eq() const {
    return static_cast<const ArgumentEqual&>(*this);
  }
  // 获取哈希函数的常量引用
  const ArgumentHash& hash_function() const {
    return static_cast<const ArgumentHash&>(*this);
  }

  // 模板迭代器结构体
  template <typename ValueType>
  struct templated_iterator {
    // 默认构造函数
    templated_iterator() = default;
    // 带参数的构造函数：使用给定的当前指针
    templated_iterator(EntryPointer current) : current(current) {}
    // 当前指针
    EntryPointer current = EntryPointer();

    // 迭代器类别：前向迭代器
    using iterator_category = std::forward_iterator_tag;
    // 值类型
    using value_type = ValueType;
    // 差值类型
    using difference_type = ptrdiff_t;
    // 指针类型
    using pointer = ValueType*;
    // 引用类型
    using reference = ValueType&;
    friend bool operator==(
        const templated_iterator& lhs,
        const templated_iterator& rhs) {
      // 比较两个迭代器是否指向相同的元素
      return lhs.current == rhs.current;
    }
    friend bool operator!=(
        const templated_iterator& lhs,
        const templated_iterator& rhs) {
      // 判断两个迭代器是否不相等
      return !(lhs == rhs);
    }

    templated_iterator& operator++() {
      // 前缀递增运算符重载，跳过空元素直到找到非空元素
      do {
        ++current;
      } while (current->is_empty());
      return *this;
    }
    templated_iterator operator++(int) {
      // 后缀递增运算符重载，返回递增前的迭代器的副本
      templated_iterator copy(*this);
      ++*this;
      return copy;
    }

    ValueType& operator*() const {
      // 解引用运算符重载，返回当前迭代器指向元素的引用
      return current->value;
    }
    ValueType* operator->() const {
      // 成员访问运算符重载，返回指向当前迭代器指向元素的指针
      return std::addressof(current->value);
    }

    // 模板转换运算符重载，只在 value_type 不是 const 时生效
    // 避免 const 类型的编译器警告
    template <
        class target_type = const value_type,
        class = std::enable_if_t<
            std::is_same_v<target_type, const value_type> &&
            !std::is_same_v<target_type, value_type>>>
    operator templated_iterator<target_type>() const {
      return {current};
    }
  };
  using iterator = templated_iterator<value_type>;
  using const_iterator = templated_iterator<const value_type>;

  iterator begin() {
    // 返回指向第一个非空元素的迭代器
    for (EntryPointer it = entries;; ++it) {
      if (it->has_value())
        return {it};
    }
  }
  const_iterator begin() const {
    // 返回指向第一个非空元素的常量迭代器
    for (EntryPointer it = entries;; ++it) {
      if (it->has_value())
        return {it};
    }
  }
  const_iterator cbegin() const {
    // 返回指向第一个非空元素的常量迭代器，与 begin() 功能相同
    return begin();
  }
  iterator end() {
    // 返回指向尾后元素的迭代器
    return {
        entries + static_cast<ptrdiff_t>(num_slots_minus_one + max_lookups)};
  }
  const_iterator end() const {
    // 返回指向尾后元素的常量迭代器
    return {
        entries + static_cast<ptrdiff_t>(num_slots_minus_one + max_lookups)};
  }
  const_iterator cend() const {
    // 返回指向尾后元素的常量迭代器，与 end() 功能相同
    return end();
  }

  iterator find(const FindKey& key) {
    // 查找具有指定键的元素，并返回指向该元素的迭代器
    uint64_t index =
        hash_policy.index_for_hash(hash_object(key), num_slots_minus_one);
    EntryPointer it = entries + ptrdiff_t(index);
    for (int8_t distance = 0; it->distance_from_desired >= distance;
         ++distance, ++it) {
      if (compares_equal(key, it->value))
        return {it};
    }
    return end();
  }
  const_iterator find(const FindKey& key) const {
    // 查找具有指定键的元素，并返回指向该元素的常量迭代器
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<sherwood_v3_table*>(this)->find(key);
  }
  uint64_t count(const FindKey& key) const {
    // 返回具有指定键的元素的数量，返回值为0或1
    return find(key) == end() ? 0 : 1;
  }
  std::pair<iterator, iterator> equal_range(const FindKey& key) {
    // 返回指定键的元素的迭代器范围
    iterator found = find(key);
    if (found == end())
      return {found, found};
    else
      return {found, std::next(found)};
  }
  std::pair<const_iterator, const_iterator> equal_range(
      const FindKey& key) const {
    // 返回指定键的元素的常量迭代器范围
    const_iterator found = find(key);
    if (found == end())
      return {found, found};
    else
      return {found, std::next(found)};
  }
  else
    return {found, std::next(found)};
}

template <typename Key, typename... Args>
std::pair<iterator, bool> emplace(Key&& key, Args&&... args) {
  // 计算哈希值后确定插入位置的索引
  uint64_t index =
      hash_policy.index_for_hash(hash_object(key), num_slots_minus_one);
  // 获取当前索引位置的条目指针
  EntryPointer current_entry = entries + ptrdiff_t(index);
  // 初始化距离值
  int8_t distance_from_desired = 0;
  // 循环查找插入位置，直到找到合适的位置或者超过距离限制
  for (; current_entry->distance_from_desired >= distance_from_desired;
       ++current_entry, ++distance_from_desired) {
    // 如果找到了相等的键，则返回找到的位置和 false
    if (compares_equal(key, current_entry->value))
      return {{current_entry}, false};
  }
  // 否则，插入新的键值对
  return emplace_new_key(
      distance_from_desired,
      current_entry,
      std::forward<Key>(key),
      std::forward<Args>(args)...);
}

std::pair<iterator, bool> insert(const value_type& value) {
  return emplace(value);
}
std::pair<iterator, bool> insert(value_type&& value) {
  return emplace(std::move(value));
}
template <typename... Args>
iterator emplace_hint(const_iterator, Args&&... args) {
  // 使用 emplace 函数插入新的键值对，并返回迭代器指向插入的位置
  return emplace(std::forward<Args>(args)...).first;
}
iterator insert(const_iterator, const value_type& value) {
  // 插入给定的 const value_type 值
  return emplace(value).first;
}
iterator insert(const_iterator, value_type&& value) {
  // 插入给定的 value_type 值
  return emplace(std::move(value)).first;
}

template <typename It>
void insert(It begin, It end) {
  // 从迭代器范围 [begin, end) 插入元素
  for (; begin != end; ++begin) {
    emplace(*begin);
  }
}
void insert(std::initializer_list<value_type> il) {
  // 从 initializer_list 插入元素
  insert(il.begin(), il.end());
}

void rehash(uint64_t num_buckets) {
  // 确定新的桶的数量，以确保负载因子在合理范围内
  num_buckets = std::max(
      num_buckets,
      static_cast<uint64_t>(
          std::ceil(num_elements / static_cast<double>(_max_load_factor))));
  // 如果新桶的数量为 0，则重置哈希表状态
  if (num_buckets == 0) {
    reset_to_empty_state();
    return;
  }
  // 计算下一个最接近的素数桶的大小
  auto new_prime_index = hash_policy.next_size_over(num_buckets);
  // 如果新桶数量等于当前桶数量，则不进行重新哈希操作
  if (num_buckets == bucket_count())
    return;
  // 计算新的最大查找次数
  int8_t new_max_lookups = compute_max_lookups(num_buckets);
  // 分配新的桶空间
  EntryPointer new_buckets(
      AllocatorTraits::allocate(*this, num_buckets + new_max_lookups));
  // 初始化新桶的特殊结束项
  EntryPointer special_end_item =
      new_buckets + static_cast<ptrdiff_t>(num_buckets + new_max_lookups - 1);
  for (EntryPointer it = new_buckets; it != special_end_item; ++it)
    it->distance_from_desired = -1;
  special_end_item->distance_from_desired = Entry::special_end_value;
  // 交换旧桶和新桶的内容
  std::swap(entries, new_buckets);
  std::swap(num_slots_minus_one, num_buckets);
  --num_slots_minus_one;
  // 提交新的素数桶大小
  hash_policy.commit(new_prime_index);
  int8_t old_max_lookups = max_lookups;
  max_lookups = new_max_lookups;
  num_elements = 0;
  // 将旧桶中的元素重新插入到新桶中
  for (EntryPointer
           it = new_buckets,
           end = it + static_cast<ptrdiff_t>(num_buckets + old_max_lookups);
       it != end;
       ++it) {
    if (it->has_value()) {
      emplace(std::move(it->value));
      it->destroy_value();
    }
  }
}
  // 释放指定数量的新桶和旧最大查找次数的数据
  deallocate_data(new_buckets, num_buckets, old_max_lookups);
}

void reserve(uint64_t num_elements_) {
  // 计算所需的桶数以容纳指定数量的元素
  uint64_t required_buckets = num_buckets_for_reserve(num_elements_);
  // 如果所需桶数大于当前桶数，则重新哈希表
  if (required_buckets > bucket_count())
    rehash(required_buckets);
}

// 删除给定迭代器指向的元素，并返回一个可转换为迭代器类型的值
// 返回可转换为迭代器类型的值是因为查找指向下一个元素的迭代器不是免费的。
// 如果关心下一个迭代器，请将返回值转换为迭代器类型。
convertible_to_iterator erase(const_iterator to_erase) {
  EntryPointer current = to_erase.current;
  // 销毁当前元素的值
  current->destroy_value();
  --num_elements;
  // 将后续元素逐个向前移动，直到所有元素都到达期望的位置
  for (EntryPointer next = current + ptrdiff_t(1);
       !next->is_at_desired_position();
       ++current, ++next) {
    current->emplace(next->distance_from_desired - 1, std::move(next->value));
    next->destroy_value();
  }
  return {to_erase.current};
}

// 删除指定范围内的元素，并返回一个迭代器指向删除操作后的位置
iterator erase(const_iterator begin_it, const_iterator end_it) {
  // 如果起始迭代器等于结束迭代器，则直接返回起始迭代器
  if (begin_it == end_it)
    return {begin_it.current};
  // 逐个遍历要删除的范围内的元素
  for (EntryPointer it = begin_it.current, end = end_it.current; it != end;
       ++it) {
    if (it->has_value()) {
      it->destroy_value();
      --num_elements;
    }
  }
  // 如果结束迭代器指向容器的末尾，则返回容器的末尾迭代器
  if (end_it == this->end())
    return this->end();
  // 计算需要向前移动的元素数量，确保后续元素的正确性
  ptrdiff_t num_to_move = std::min(
      static_cast<ptrdiff_t>(end_it.current->distance_from_desired),
      end_it.current - begin_it.current);
  EntryPointer to_return = end_it.current - num_to_move;
  // 逐个将后续元素向前移动，直到所有元素都到达期望的位置
  for (EntryPointer it = end_it.current; !it->is_at_desired_position();) {
    EntryPointer target = it - num_to_move;
    target->emplace(
        it->distance_from_desired - num_to_move, std::move(it->value));
    it->destroy_value();
    ++it;
    num_to_move = std::min(
        static_cast<ptrdiff_t>(it->distance_from_desired), num_to_move);
  }
  return {to_return};
}

// 删除具有指定关键字的元素，并返回删除的元素数量（0或1）
uint64_t erase(const FindKey& key) {
  auto found = find(key);
  if (found == end())
    return 0;
  else {
    erase(found);
    return 1;
  }
}

// 清空所有元素
void clear() {
  // 遍历所有表条目，销毁每个有值的条目，并将元素计数归零
  for (EntryPointer it = entries,
                    end = it +
           static_cast<ptrdiff_t>(num_slots_minus_one + max_lookups);
       it != end;
       ++it) {
    if (it->has_value())
      it->destroy_value();
  }
  num_elements = 0;
}

// 将表的容量调整为最小可能值
void shrink_to_fit() {
  // 根据另一个容器的大小重新哈希表，以便于内存使用的最小化
  rehash_for_other_container(*this);
}

// 交换当前表与另一个表的内容
void swap(sherwood_v3_table& other) noexcept {
  using std::swap;
  // 交换指针以及哈希和相等性比较器
  swap_pointers(other);
  swap(static_cast<ArgumentHash&>(*this), static_cast<ArgumentHash&>(other));
  swap(
      static_cast<ArgumentEqual&>(*this), static_cast<ArgumentEqual&>(other));
  // 如果分配器支持容器交换，则交换分配器
  if (AllocatorTraits::propagate_on_container_swap::value)
    swap(static_cast<EntryAlloc&>(*this), static_cast<EntryAlloc&>(other));
}

// 返回当前表中元素的数量
uint64_t size() const {
  return num_elements;
}
// 返回当前表可能包含的最大元素数量
uint64_t max_size() const {
    return (AllocatorTraits::max_size(*this)) / sizeof(Entry);
  }
  // 返回当前分配器最大可分配空间除以每个条目的大小，得到容器能容纳的最大条目数
  uint64_t bucket_count() const {
    // 如果有槽位，则返回槽位数；否则返回 0
    return num_slots_minus_one ? num_slots_minus_one + 1 : 0;
  }
  // 返回当前分配器最大可分配空间减去最小查询次数，除以每个条目的大小，得到最大桶数
  size_type max_bucket_count() const {
    return (AllocatorTraits::max_size(*this) - min_lookups) / sizeof(Entry);
  }
  // 返回给定键的哈希桶索引
  uint64_t bucket(const FindKey& key) const {
    return hash_policy.index_for_hash(hash_object(key), num_slots_minus_one);
  }
  // 返回当前加载因子，即当前元素数除以桶数
  float load_factor() const {
    uint64_t buckets = bucket_count();
    if (buckets)
      return static_cast<float>(num_elements) / bucket_count();
    else
      return 0;
  }
  // 设置最大加载因子的值
  void max_load_factor(float value) {
    _max_load_factor = value;
  }
  // 返回当前设置的最大加载因子值
  float max_load_factor() const {
    return _max_load_factor;
  }

  // 返回容器是否为空，即元素数是否为零
  bool empty() const {
    return num_elements == 0;
  }

 private:
  // 指向条目的指针，初始化为默认空表
  EntryPointer entries = empty_default_table();
  // 桶数减一，初始化为零
  uint64_t num_slots_minus_one = 0;
  // 哈希策略选择器类型，根据 ArgumentHash 初始化
  typename HashPolicySelector<ArgumentHash>::type hash_policy;
  // 最大查询次数，默认为 detailv3::min_lookups 减一
  int8_t max_lookups = detailv3::min_lookups - 1;
  // 最大加载因子，默认为 0.5
  float _max_load_factor = 0.5f;
  // 元素数量，默认为零
  uint64_t num_elements = 0;

  // 返回初始化为空的默认表
  EntryPointer empty_default_table() {
    EntryPointer result =
        AllocatorTraits::allocate(*this, detailv3::min_lookups);
    EntryPointer special_end_item =
        result + static_cast<ptrdiff_t>(detailv3::min_lookups - 1);
    for (EntryPointer it = result; it != special_end_item; ++it)
      it->distance_from_desired = -1;
    special_end_item->distance_from_desired = Entry::special_end_value;
    return result;
  }

  // 计算适合的最大查询次数，根据桶的数量
  static int8_t compute_max_lookups(uint64_t num_buckets) {
    int8_t desired = detailv3::log2(num_buckets);
    return std::max(detailv3::min_lookups, desired);
  }

  // 根据元素数量计算保留的桶数
  uint64_t num_buckets_for_reserve(uint64_t num_elements_) const {
    return static_cast<uint64_t>(std::ceil(
        static_cast<double>(num_elements_) /
        std::min(0.5, static_cast<double>(_max_load_factor))));
  }
  // 重新哈希以适应其他容器的大小
  void rehash_for_other_container(const sherwood_v3_table& other) {
    rehash(
        std::min(num_buckets_for_reserve(other.size()), other.bucket_count()));
  }

  // 交换指针，用于容器之间的交换操作
  void swap_pointers(sherwood_v3_table& other) {
    using std::swap;
    swap(hash_policy, other.hash_policy);
    swap(entries, other.entries);
    swap(num_slots_minus_one, other.num_slots_minus_one);
    swap(num_elements, other.num_elements);
    swap(max_lookups, other.max_lookups);
    swap(_max_load_factor, other._max_load_factor);
  }

  // 插入新键值对，返回迭代器和是否插入成功的 pair
  template <typename Key, typename... Args>
  SKA_NOINLINE(std::pair<iterator, bool>)
  emplace_new_key(
      int8_t distance_from_desired,
      EntryPointer current_entry,
      Key&& key,
      Args&&... args) {
    using std::swap;
    // 如果桶数为零或距离期望值达到最大查询次数或元素数超出加载因子限制，则扩容
    if (num_slots_minus_one == 0 || distance_from_desired == max_lookups ||
        num_elements + 1 >
            (num_slots_minus_one + 1) * static_cast<double>(_max_load_factor)) {
      grow();
      // 返回插入结果
      return emplace(std::forward<Key>(key), std::forward<Args>(args)...);
    } else if (current_entry->is_empty()) {
      // 如果当前条目为空，则在当前位置插入新条目
      current_entry->emplace(
          distance_from_desired,
          std::forward<Key>(key),
          std::forward<Args>(args)...);
      // 增加元素数量计数器
      ++num_elements;
      // 返回包含新条目的迭代器和成功插入标志
      return {{current_entry}, true};
    }
    // 准备要插入的新值
    value_type to_insert(std::forward<Key>(key), std::forward<Args>(args)...);
    // 交换当前条目的距离和值与要插入的新值
    swap(distance_from_desired, current_entry->distance_from_desired);
    swap(to_insert, current_entry->value);
    // 创建指向当前条目的迭代器
    iterator result = {current_entry};
    // 开始循环插入
    for (++distance_from_desired, ++current_entry;; ++current_entry) {
      // 如果当前条目为空，则在当前位置插入新条目
      if (current_entry->is_empty()) {
        current_entry->emplace(distance_from_desired, std::move(to_insert));
        // 增加元素数量计数器
        ++num_elements;
        // 返回包含新条目的迭代器和成功插入标志
        return {result, true};
      } else if (current_entry->distance_from_desired < distance_from_desired) {
        // 如果当前条目的距离小于要插入的距离，交换当前条目的距离和值与要插入的新值
        swap(distance_from_desired, current_entry->distance_from_desired);
        swap(to_insert, current_entry->value);
        // 增加距离
        ++distance_from_desired;
      } else {
        // 增加距离
        ++distance_from_desired;
        // 如果距离达到最大查找次数，交换要插入的新值与结果当前位置的值，然后扩展哈希表并重新插入
        if (distance_from_desired == max_lookups) {
          swap(to_insert, result.current->value);
          grow();
          return emplace(std::move(to_insert));
        }
      }
    }
  }

  void grow() {
    // 根据当前桶数，扩展哈希表
    rehash(std::max(uint64_t(4), 2 * bucket_count()));
  }

  void deallocate_data(
      EntryPointer begin,
      uint64_t num_slots_minus_one_,
      int8_t max_lookups_) {
    // 使用分配器释放数据
    AllocatorTraits::deallocate(
        *this, begin, num_slots_minus_one_ + max_lookups_ + 1);
  }

  void reset_to_empty_state() {
    // 释放当前数据，重置表为空状态，重置最大查找次数
    deallocate_data(entries, num_slots_minus_one, max_lookups);
    entries = empty_default_table();
    num_slots_minus_one = 0;
    hash_policy.reset();
    max_lookups = detailv3::min_lookups - 1;
  }

  template <typename U>
  uint64_t hash_object(const U& key) {
    // 使用哈希器计算对象的哈希值
    return static_cast<DetailHasher&>(*this)(key);
  }
  template <typename U>
  uint64_t hash_object(const U& key) const {
    // 使用哈希器计算对象的哈希值（常量版本）
    return static_cast<const DetailHasher&>(*this)(key);
  }
  template <typename L, typename R>
  bool compares_equal(const L& lhs, const R& rhs) {
    // 使用相等性比较器比较两个对象是否相等
    return static_cast<Equal&>(*this)(lhs, rhs);
  }

 public:
  // 可转换为迭代器结构体
  struct convertible_to_iterator {
    EntryPointer it;

    // 转换为普通迭代器
    operator iterator() {
      if (it->has_value())
        return {it};
      else
        return ++iterator{it};
    }
    // 转换为常量迭代器
    operator const_iterator() {
      if (it->has_value())
        return {it};
      else
        return ++const_iterator{it};
    }
  };
};
} // namespace detailv3

// 定义一个结构体，实现不同的哈希策略
struct prime_number_hash_policy {
  // 返回 hash 取模 0 的结果
  static uint64_t mod0(uint64_t) {
    return 0llu;
  }
  // 返回 hash 取模 2 的结果
  static uint64_t mod2(uint64_t hash) {
    return hash % 2llu;
  }
  // 返回 hash 取模 3 的结果
  static uint64_t mod3(uint64_t hash) {
    return hash % 3llu;
  }
  // 返回 hash 取模 5 的结果
  static uint64_t mod5(uint64_t hash) {
    return hash % 5llu;
  }
  // 返回 hash 取模 7 的结果
  static uint64_t mod7(uint64_t hash) {
    return hash % 7llu;
  }
  // 返回 hash 取模 11 的结果
  static uint64_t mod11(uint64_t hash) {
    return hash % 11llu;
  }
  // 返回 hash 取模 13 的结果
  static uint64_t mod13(uint64_t hash) {
    return hash % 13llu;
  }
  // 返回 hash 取模 17 的结果
  static uint64_t mod17(uint64_t hash) {
    return hash % 17llu;
  }
  // 返回 hash 取模 23 的结果
  static uint64_t mod23(uint64_t hash) {
    return hash % 23llu;
  }
  // 返回 hash 取模 29 的结果
  static uint64_t mod29(uint64_t hash) {
    return hash % 29llu;
  }
  // 返回 hash 取模 37 的结果
  static uint64_t mod37(uint64_t hash) {
    return hash % 37llu;
  }
  // 返回 hash 取模 47 的结果
  static uint64_t mod47(uint64_t hash) {
    return hash % 47llu;
  }
  // 返回 hash 取模 59 的结果
  static uint64_t mod59(uint64_t hash) {
    return hash % 59llu;
  }
  // 返回 hash 取模 73 的结果
  static uint64_t mod73(uint64_t hash) {
    return hash % 73llu;
  }
  // 返回 hash 取模 97 的结果
  static uint64_t mod97(uint64_t hash) {
    return hash % 97llu;
  }
  // 返回 hash 取模 127 的结果
  static uint64_t mod127(uint64_t hash) {
    return hash % 127llu;
  }
  // 返回 hash 取模 151 的结果
  static uint64_t mod151(uint64_t hash) {
    return hash % 151llu;
  }
  // 返回 hash 取模 197 的结果
  static uint64_t mod197(uint64_t hash) {
    return hash % 197llu;
  }
  // 返回 hash 取模 251 的结果
  static uint64_t mod251(uint64_t hash) {
    return hash % 251llu;
  }
  // 返回 hash 取模 313 的结果
  static uint64_t mod313(uint64_t hash) {
    return hash % 313llu;
  }
  // 返回 hash 取模 397 的结果
  static uint64_t mod397(uint64_t hash) {
    return hash % 397llu;
  }
  // 返回 hash 取模 499 的结果
  static uint64_t mod499(uint64_t hash) {
    return hash % 499llu;
  }
  // 返回 hash 取模 631 的结果
  static uint64_t mod631(uint64_t hash) {
    return hash % 631llu;
  }
  // 返回 hash 取模 797 的结果
  static uint64_t mod797(uint64_t hash) {
    return hash % 797llu;
  }
  // 返回 hash 取模 1009 的结果
  static uint64_t mod1009(uint64_t hash) {
    return hash % 1009llu;
  }
  // 返回 hash 取模 1259 的结果
  static uint64_t mod1259(uint64_t hash) {
    return hash % 1259llu;
  }
  // 返回 hash 取模 1597 的结果
  static uint64_t mod1597(uint64_t hash) {
    return hash % 1597llu;
  }
  // 返回 hash 取模 2011 的结果
  static uint64_t mod2011(uint64_t hash) {
    return hash % 2011llu;
  }
  // 返回 hash 取模 2539 的结果
  static uint64_t mod2539(uint64_t hash) {
    return hash % 2539llu;
  }
  // 返回 hash 取模 3203 的结果
  static uint64_t mod3203(uint64_t hash) {
    return hash % 3203llu;
  }
  // 返回 hash 取模 4027 的结果
  static uint64_t mod4027(uint64_t hash) {
    return hash % 4027llu;
  }
  // 返回 hash 取模 5087 的结果
  static uint64_t mod5087(uint64_t hash) {
    return hash % 5087llu;
  }
  // 返回 hash 取模 6421 的结果
  static uint64_t mod6421(uint64_t hash) {
    return hash % 6421llu;
  }
  // 返回 hash 取模 8089 的结果
  static uint64_t mod8089(uint64_t hash) {
    return hash % 8089llu;
  }
  // 返回 hash 取模 10193 的结果
  static uint64_t mod10193(uint64_t hash) {
    return hash % 10193llu;
  }
  // 返回 hash 取模 12853 的结果
  static uint64_t mod12853(uint64_t hash) {
    return hash % 12853llu;
  }
  // 返回 hash 取模 16193 的结果
  static uint64_t mod16193(uint64_t hash) {
    return hash % 16193llu;
  }
  // 返回 hash 取模 20399 的结果
  static uint64_t mod20399(uint64_t hash) {
    return hash % 20399llu;
  }
  // 返回 hash 取模 25717 的结果
  static uint64_t mod25717(uint64_t hash) {
    return hash % 25717llu;
  }
  // 返回 hash 取模 32401 的结果
  static uint64_t mod32401(uint64_t hash) {
    return hash % 32401llu;
  }
  // 返回 hash 取模 40823 的结果
  static uint64_t mod40823(uint64_t hash) {
    // 返回 hash 取模 40823 的结果
  return hash % 210719881llu;
}
  return hash % 543724411781llu;
  }



// 计算给定哈希值对 543724411781llu 取模后的余数
static uint64_t mod543724411781(uint64_t hash) {
    return hash % 543724411781llu;
}
  static uint64_t mod883823312134381(uint64_t hash) {
    // 计算哈希值对883823312134381llu取模，返回结果
    return hash % 883823312134381llu;
  }
    // 对给定的哈希值进行模运算，使用固定的大素数 883823312134381 进行取模操作
    static uint64_t mod883823312134381(uint64_t hash) {
        return hash % 883823312134381llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 1113547595345903 进行取模操作
    static uint64_t mod1113547595345903(uint64_t hash) {
        return hash % 1113547595345903llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 1402982055436147 进行取模操作
    static uint64_t mod1402982055436147(uint64_t hash) {
        return hash % 1402982055436147llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 1767646624268779 进行取模操作
    static uint64_t mod1767646624268779(uint64_t hash) {
        return hash % 1767646624268779llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 2227095190691797 进行取模操作
    static uint64_t mod2227095190691797(uint64_t hash) {
        return hash % 2227095190691797llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 2805964110872297 进行取模操作
    static uint64_t mod2805964110872297(uint64_t hash) {
        return hash % 2805964110872297llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 3535293248537579 进行取模操作
    static uint64_t mod3535293248537579(uint64_t hash) {
        return hash % 3535293248537579llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 4454190381383713 进行取模操作
    static uint64_t mod4454190381383713(uint64_t hash) {
        return hash % 4454190381383713llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 5611928221744609 进行取模操作
    static uint64_t mod5611928221744609(uint64_t hash) {
        return hash % 5611928221744609llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 7070586497075177 进行取模操作
    static uint64_t mod7070586497075177(uint64_t hash) {
        return hash % 7070586497075177llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 8908380762767489 进行取模操作
    static uint64_t mod8908380762767489(uint64_t hash) {
        return hash % 8908380762767489llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 11223856443489329 进行取模操作
    static uint64_t mod11223856443489329(uint64_t hash) {
        return hash % 11223856443489329llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 14141172994150357 进行取模操作
    static uint64_t mod14141172994150357(uint64_t hash) {
        return hash % 14141172994150357llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 17816761525534927 进行取模操作
    static uint64_t mod17816761525534927(uint64_t hash) {
        return hash % 17816761525534927llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 22447712886978529 进行取模操作
    static uint64_t mod22447712886978529(uint64_t hash) {
        return hash % 22447712886978529llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 28282345988300791 进行取模操作
    static uint64_t mod28282345988300791(uint64_t hash) {
        return hash % 28282345988300791llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 35633523051069991 进行取模操作
    static uint64_t mod35633523051069991(uint64_t hash) {
        return hash % 35633523051069991llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 44895425773957261 进行取模操作
    static uint64_t mod44895425773957261(uint64_t hash) {
        return hash % 44895425773957261llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 56564691976601587 进行取模操作
    static uint64_t mod56564691976601587(uint64_t hash) {
        return hash % 56564691976601587llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 71267046102139967 进行取模操作
    static uint64_t mod71267046102139967(uint64_t hash) {
        return hash % 71267046102139967llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 89790851547914507 进行取模操作
    static uint64_t mod89790851547914507(uint64_t hash) {
        return hash % 89790851547914507llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 113129383953203213 进行取模操作
    static uint64_t mod113129383953203213(uint64_t hash) {
        return hash % 113129383953203213llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 142534092204280003 进行取模操作
    static uint64_t mod142534092204280003(uint64_t hash) {
        return hash % 142534092204280003llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 179581703095829107 进行取模操作
    static uint64_t mod179581703095829107(uint64_t hash) {
        return hash % 179581703095829107llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 226258767906406483 进行取模操作
    static uint64_t mod226258767906406483(uint64_t hash) {
        return hash % 226258767906406483llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 285068184408560057 进行取模操作
    static uint64_t mod285068184408560057(uint64_t hash) {
        return hash % 285068184408560057llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 359163406191658253 进行取模操作
    static uint64_t mod359163406191658253(uint64_t hash) {
        return hash % 359163406191658253llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 452517535812813007 进行取模操作
    static uint64_t mod452517535812813007(uint64_t hash) {
        return hash % 452517535812813007llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 570136368817120201 进行取模操作
    static uint64_t mod570136368817120201(uint64_t hash) {
        return hash % 570136368817120201llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 718326812383316683 进行取模操作
    static uint64_t mod718326812383316683(uint64_t hash) {
        return hash % 718326812383316683llu;
    }
    
    // 对给定的哈希值进行模运算，使用固定的大素数 905035071625626043 进行取模操作
    static
    return hash % 905035071625626043llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod1140272737634240411(uint64_t hash) {
    // 返回哈希值对 1140272737634240411llu 取模后的结果
    return hash % 1140272737634240411llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod1436653624766633509(uint64_t hash) {
    // 返回哈希值对 1436653624766633509llu 取模后的结果
    return hash % 1436653624766633509llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod1810070143251252131(uint64_t hash) {
    // 返回哈希值对 1810070143251252131llu 取模后的结果
    return hash % 1810070143251252131llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod2280545475268481167(uint64_t hash) {
    // 返回哈希值对 2280545475268481167llu 取模后的结果
    return hash % 2280545475268481167llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod2873307249533267101(uint64_t hash) {
    // 返回哈希值对 2873307249533267101llu 取模后的结果
    return hash % 2873307249533267101llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod3620140286502504283(uint64_t hash) {
    // 返回哈希值对 3620140286502504283llu 取模后的结果
    return hash % 3620140286502504283llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod4561090950536962147(uint64_t hash) {
    // 返回哈希值对 4561090950536962147llu 取模后的结果
    return hash % 4561090950536962147llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod5746614499066534157(uint64_t hash) {
    // 返回哈希值对 5746614499066534157llu 取模后的结果
    return hash % 5746614499066534157llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod7240280573005008577(uint64_t hash) {
    // 返回哈希值对 7240280573005008577llu 取模后的结果
    return hash % 7240280573005008577llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod9122181901073924329(uint64_t hash) {
    // 返回哈希值对 9122181901073924329llu 取模后的结果
    return hash % 9122181901073924329llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod11493228998133068689(uint64_t hash) {
    // 返回哈希值对 11493228998133068689llu 取模后的结果
    return hash % 11493228998133068689llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod14480561146010017169(uint64_t hash) {
    // 返回哈希值对 14480561146010017169llu 取模后的结果
    return hash % 14480561146010017169llu;
  }
  // 使用给定的大素数对哈希值进行取模运算，保证结果在合理范围内
  static uint64_t mod18446744073709551557(uint64_t hash) {
    // 返回哈希值对 18446744073709551557llu 取模后的结果
    return hash % 18446744073709551557llu;
  }

  using mod_function = uint64_t (*)(uint64_t);

  mod_function next_size_over(uint64_t& size) const {
    // prime_list 数组中存储了一系列素数，用于动态调整哈希表大小
    // 在 prime_list 中找到大于等于 size 的第一个素数
    const uint64_t* found = std::lower_bound(
        std::begin(prime_list), std::end(prime_list) - 1, size);
    // 更新 size 为找到的素数值
    size = *found;
    // 返回对应的哈希函数指针，该指针指向大于等于 size 的素数对应的哈希函数
    return mod_functions[1 + found - prime_list];
  }
  // 将当前的哈希函数指针更新为给定的新哈希函数
  void commit(mod_function new_mod_function) {
    current_mod_function = new_mod_function;
  }
  // 将当前的哈希函数指针重置为默认的 mod0 函数
  void reset() {
    current_mod_function = &mod0;
  }

  // 根据当前的哈希函数计算给定哈希值在哈希表中的索引位置
  uint64_t index_for_hash(uint64_t hash, uint64_t /*num_slots_minus_one*/)
      const {
    return current_mod_function(hash);
  }
  // 确保给定索引值在合法的范围内，即小于等于 num_slots_minus_one
  // 如果索引值超出范围，则使用当前的哈希函数重新计算索引
  uint64_t keep_in_range(uint64_t index, uint64_t num_slots_minus_one) const {
    return index > num_slots_minus_one ? current_mod_function(index) : index;
  }

 private:
  mod_function current_mod_function = &mod0;
};

// 定义一个名为 power_of_two_hash_policy 的结构体，实现哈希表的二次幂掩码策略
struct power_of_two_hash_policy {
  // 根据哈希值和槽位数减一计算索引
  uint64_t index_for_hash(uint64_t hash, uint64_t num_slots_minus_one) const {
    return hash & num_slots_minus_one;
  }
  // 保持索引在合法范围内
  uint64_t keep_in_range(uint64_t index, uint64_t num_slots_minus_one) const {
    return index_for_hash(index, num_slots_minus_one);
  }
  // 计算大于当前尺寸的下一个二次幂大小，并返回偏移量
  int8_t next_size_over(uint64_t& size) const {
    size = detailv3::next_power_of_two(size);
    return 0;
  }
  // 提交偏移量的变更
  void commit(int8_t) {}
  // 重置策略
  void reset() {}
};

// 定义一个名为 fibonacci_hash_policy 的结构体，实现哈希表的斐波那契哈希策略
struct fibonacci_hash_policy {
  // 根据哈希值计算索引，使用斐波那契乘数和位移操作
  uint64_t index_for_hash(uint64_t hash, uint64_t /*num_slots_minus_one*/) const {
    return (11400714819323198485ull * hash) >> shift;
  }
  // 保持索引在合法范围内
  uint64_t keep_in_range(uint64_t index, uint64_t num_slots_minus_one) const {
    return index & num_slots_minus_one;
  }

  // 计算大于当前尺寸的下一个二次幂大小，并返回偏移量
  int8_t next_size_over(uint64_t& size) const {
    size = std::max(uint64_t(2), detailv3::next_power_of_two(size));
    return static_cast<int8_t>(64 - detailv3::log2(size));
  }
  // 提交位移的变更
  void commit(int8_t shift_) {
    shift = shift_;
  }
  // 重置位移值
  void reset() {
    shift = 63;
  }

 private:
  int8_t shift = 63; // 默认位移值为63
};

// 定义一个模板类 flat_hash_map，继承自 detailv3::sherwood_v3_table
template <
    typename K,
    typename V,
    typename H = std::hash<K>,
    typename E = std::equal_to<K>,
    typename A = std::allocator<std::pair<K, V>>>
class flat_hash_map
    : public detailv3::sherwood_v3_table<
          std::pair<K, V>,
          K,
          H,
          detailv3::KeyOrValueHasher<K, std::pair<K, V>, H>,
          E,
          detailv3::KeyOrValueEquality<K, std::pair<K, V>, E>,
          A,
          typename std::allocator_traits<A>::template rebind_alloc<
              detailv3::sherwood_v3_entry<std::pair<K, V>>>> {
  using Table = detailv3::sherwood_v3_table<
      std::pair<K, V>,
      K,
      H,
      detailv3::KeyOrValueHasher<K, std::pair<K, V>, H>,
      E,
      detailv3::KeyOrValueEquality<K, std::pair<K, V>, E>,
      A,
      typename std::allocator_traits<A>::template rebind_alloc<
          detailv3::sherwood_v3_entry<std::pair<K, V>>>>;

 public:
  using key_type = K;
  using mapped_type = V;

  using Table::Table; // 继承构造函数

  flat_hash_map() = default; // 默认构造函数

  // 重载操作符[]，用于插入或访问元素
  inline V& operator[](const K& key) {
    return emplace(key, convertible_to_value()).first->second;
  }
  inline V& operator[](K&& key) {
    return emplace(std::move(key), convertible_to_value()).first->second;
  }

  // 访问元素，如果元素不存在则抛出异常
  V& at(const K& key) {
    auto found = this->find(key);
    if (found == this->end())
      throw std::out_of_range("Argument passed to at() was not in the map.");
    return found->second;
  }

  const V& at(const K& key) const {
    auto found = this->find(key);
    if (found == this->end())
      throw std::out_of_range("Argument passed to at() was not in the map.");
    return found->second;
  }

  using Table::emplace; // 继承 emplace 函数

  // 插入或更新元素
  template <typename M>
  std::pair<typename Table::iterator, bool> insert_or_assign(
      const key_type& key,
      M&& m) {
    // 尝试将键值对插入表中，如果键已存在，则更新其对应的值
    auto emplace_result = emplace(key, std::forward<M>(m));
    // 如果插入失败（键已存在），则更新已存在键对应的值
    if (!emplace_result.second)
      emplace_result.first->second = std::forward<M>(m);
    // 返回插入或更新的结果
    return emplace_result;
  }
  
  // 向表中插入或更新键值对，返回迭代器和是否成功的布尔值
  template <typename M>
  std::pair<typename Table::iterator, bool> insert_or_assign(
      key_type&& key,
      M&& m) {
    // 调用emplace函数尝试插入或更新键值对
    auto emplace_result = emplace(std::move(key), std::forward<M>(m));
    // 如果插入失败（键已存在），则更新已存在键对应的值
    if (!emplace_result.second)
      emplace_result.first->second = std::forward<M>(m);
    // 返回插入或更新的结果
    return emplace_result;
  }
  
  // 向表中插入或更新键值对，并返回插入或更新的迭代器
  template <typename M>
  typename Table::iterator insert_or_assign(
      typename Table::const_iterator,
      const key_type& key,
      M&& m) {
    // 调用上面定义的insert_or_assign函数，并传递给定的键和值
    return insert_or_assign(key, std::forward<M>(m)).first;
  }
  
  // 向表中插入或更新键值对，并返回插入或更新的迭代器
  template <typename M>
  typename Table::iterator insert_or_assign(
      typename Table::const_iterator,
      key_type&& key,
      M&& m) {
    // 调用上面定义的insert_or_assign函数，并传递给定的键和值
    return insert_or_assign(std::move(key), std::forward<M>(m)).first;
  }
  
  // 定义友元函数，用于判断两个flat_hash_map对象是否相等
  friend bool operator==(const flat_hash_map& lhs, const flat_hash_map& rhs) {
    // 首先比较两个表的大小
    if (lhs.size() != rhs.size())
      return false;
    // 遍历左侧表中的每个键值对
    for (const typename Table::value_type& value : lhs) {
      // 在右侧表中查找当前键对应的值
      auto found = rhs.find(value.first);
      // 如果未找到对应的键或者找到的值不相等，则返回false
      if (found == rhs.end() || value.second != found->second)
        return false;
    }
    // 若所有键值对均相等，则返回true
    return true;
  }
  
  // 定义友元函数，用于判断两个flat_hash_map对象是否不相等
  friend bool operator!=(const flat_hash_map& lhs, const flat_hash_map& rhs) {
    // 利用上面定义的==操作符来实现!=操作
    return !(lhs == rhs);
  }
  
 private:
  // 内部结构体，用于将convertible_to_value隐式转换为V类型
  struct convertible_to_value {
    operator V() const {
      return V();
    }
  };
};

// 定义模板类 flat_hash_set，继承自 detailv3::sherwood_v3_table，用于实现基于平摊分析的哈希集合
template <
    typename T,
    typename H = std::hash<T>,
    typename E = std::equal_to<T>,
    typename A = std::allocator<T>>
class flat_hash_set
    : public detailv3::sherwood_v3_table<
          T,
          T,
          H,
          detailv3::functor_storage<uint64_t, H>,
          E,
          detailv3::functor_storage<bool, E>,
          A,
          typename std::allocator_traits<A>::template rebind_alloc<
              detailv3::sherwood_v3_entry<T>>> {
  // 使用别名简化表达
  using Table = detailv3::sherwood_v3_table<
      T,
      T,
      H,
      detailv3::functor_storage<uint64_t, H>,
      E,
      detailv3::functor_storage<bool, E>,
      A,
      typename std::allocator_traits<A>::template rebind_alloc<
          detailv3::sherwood_v3_entry<T>>>;

 public:
  // 公共成员，表明 key_type 为 T 类型
  using key_type = T;

  // 继承构造函数
  using Table::Table;
  // 默认构造函数
  flat_hash_set() = default;

  // 插入元素的模板函数，支持多种参数类型
  template <typename... Args>
  std::pair<typename Table::iterator, bool> emplace(Args&&... args) {
    return Table::emplace(T(std::forward<Args>(args)...));
  }
  std::pair<typename Table::iterator, bool> emplace(const key_type& arg) {
    return Table::emplace(arg);
  }
  std::pair<typename Table::iterator, bool> emplace(key_type& arg) {
    return Table::emplace(arg);
  }
  std::pair<typename Table::iterator, bool> emplace(const key_type&& arg) {
    return Table::emplace(std::move(arg));
  }
  std::pair<typename Table::iterator, bool> emplace(key_type&& arg) {
    return Table::emplace(std::move(arg));
  }

  // 友元函数，比较两个 flat_hash_set 是否相等
  friend bool operator==(const flat_hash_set& lhs, const flat_hash_set& rhs) {
    // 如果大小不同，返回 false
    if (lhs.size() != rhs.size())
      return false;
    // 遍历 lhs 中的元素，检查是否都存在于 rhs 中
    for (const T& value : lhs) {
      if (rhs.find(value) == rhs.end())
        return false;
    }
    return true;
  }
  // 友元函数，比较两个 flat_hash_set 是否不等
  friend bool operator!=(const flat_hash_set& lhs, const flat_hash_set& rhs) {
    // 使用 operator== 的结果取反
    return !(lhs == rhs);
  }
};

// 定义结构体 power_of_two_std_hash，继承自 std::hash<T>，使用 ska::power_of_two_hash_policy 策略
template <typename T>
struct power_of_two_std_hash : std::hash<T> {
  typedef ska::power_of_two_hash_policy hash_policy;
};

} // end namespace ska

// 弹出先前压入的 Clang 编译诊断设置
C10_CLANG_DIAGNOSTIC_POP()

// 如果编译器为 MSVC 并且不是 Clang，则弹出警告设置
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
```