# `.\pytorch\c10\util\order_preserving_flat_hash_map.h`

```
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
// (See http://www.boost.org/LICENSE_1_0.txt)

// Modified to maintain insertion and deletion order through a doubly-linked
// list

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef _MSC_VER
// Define SKA_NOINLINE for Microsoft Visual C++ to mark functions as non-inline
#define SKA_NOINLINE(...) __declspec(noinline) __VA_ARGS__
#else
// Define SKA_NOINLINE for other compilers (e.g., GCC, Clang) to mark functions as non-inline
#define SKA_NOINLINE(...) __VA_ARGS__ __attribute__((noinline))
#endif

// Begin namespace ska_ordered for ordered hash policies
namespace ska_ordered {

// Forward declarations of hash policies
struct prime_number_hash_policy;
struct power_of_two_hash_policy;
struct fibonacci_hash_policy;

// Begin namespace detailv3 for internal implementation details
namespace detailv3 {

// Functor storage template for storing and invoking functors
template <typename Result, typename Functor>
struct functor_storage : Functor {
  functor_storage() = default;
  functor_storage(const Functor& functor) : Functor(functor) {}
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
  functor_storage(function_ptr function) : function(function) {}
  Result operator()(Args... args) const {
    return function(std::forward<Args>(args)...);
  }
  operator function_ptr&() {
    return function;
  }
  operator const function_ptr&() {
    return function;
  }
};

// Hasher template for keys or values, using a provided hasher functor
template <typename key_type, typename value_type, typename hasher>
struct KeyOrValueHasher : functor_storage<uint64_t, hasher> {
  typedef functor_storage<uint64_t, hasher> hasher_storage;
  KeyOrValueHasher() = default;
  KeyOrValueHasher(const hasher& hash) : hasher_storage(hash) {}

  // Hashing operator for keys
  uint64_t operator()(const key_type& key) {
    return static_cast<hasher_storage&>(*this)(key);
  }

  // Const hashing operator for keys
  uint64_t operator()(const key_type& key) const {
    return static_cast<const hasher_storage&>(*this)(key);
  }

  // Hashing operator for values
  uint64_t operator()(const value_type& value) {
  // 转换并调用此哈希器对象对value.first的哈希函数，返回哈希值
  return static_cast<hasher_storage&>(*this)(value.first);
}

uint64_t operator()(const value_type& value) const {
  // 转换并调用此常量哈希器对象对value.first的哈希函数，返回哈希值
  return static_cast<const hasher_storage&>(*this)(value.first);
}

template <typename F, typename S>
uint64_t operator()(const std::pair<F, S>& value) {
  // 转换并调用此哈希器对象对value.first的哈希函数，返回哈希值
  return static_cast<hasher_storage&>(*this)(value.first);
}

template <typename F, typename S>
uint64_t operator()(const std::pair<F, S>& value) const {
  // 转换并调用此常量哈希器对象对value.first的哈希函数，返回哈希值
  return static_cast<const hasher_storage&>(*this)(value.first);
}
};

// 定义模板结构体 KeyOrValueEquality，用于比较键或值的相等性
template <typename key_type, typename value_type, typename key_equal>
struct KeyOrValueEquality : functor_storage<bool, key_equal> {
    typedef functor_storage<bool, key_equal> equality_storage;
    
    // 默认构造函数
    KeyOrValueEquality() = default;
    
    // 带有相等性判断器的构造函数
    KeyOrValueEquality(const key_equal& equality) : equality_storage(equality) {}
    
    // 比较两个键的相等性
    bool operator()(const key_type& lhs, const key_type& rhs) {
        return static_cast<equality_storage&>(*this)(lhs, rhs);
    }
    
    // 比较键和值的相等性
    bool operator()(const key_type& lhs, const value_type& rhs) {
        return static_cast<equality_storage&>(*this)(lhs, rhs.first);
    }
    
    // 比较值和键的相等性
    bool operator()(const value_type& lhs, const key_type& rhs) {
        return static_cast<equality_storage&>(*this)(lhs.first, rhs);
    }
    
    // 比较两个值的相等性
    bool operator()(const value_type& lhs, const value_type& rhs) {
        return static_cast<equality_storage&>(*this)(lhs.first, rhs.first);
    }
    
    // 模板函数，比较键和 std::pair 的第一个元素的相等性
    template <typename F, typename S>
    bool operator()(const key_type& lhs, const std::pair<F, S>& rhs) {
        return static_cast<equality_storage&>(*this)(lhs, rhs.first);
    }
    
    // 模板函数，比较 std::pair 和键的第一个元素的相等性
    template <typename F, typename S>
    bool operator()(const std::pair<F, S>& lhs, const key_type& rhs) {
        return static_cast<equality_storage&>(*this)(lhs.first, rhs);
    }
    
    // 模板函数，比较值和 std::pair 的第一个元素的相等性
    template <typename F, typename S>
    bool operator()(const value_type& lhs, const std::pair<F, S>& rhs) {
        return static_cast<equality_storage&>(*this)(lhs.first, rhs.first);
    }
    
    // 模板函数，比较 std::pair 和值的第一个元素的相等性
    template <typename F, typename S>
    bool operator()(const std::pair<F, S>& lhs, const value_type& rhs) {
        return static_cast<equality_storage&>(*this)(lhs.first, rhs.first);
    }
    
    // 模板函数，比较两个 std::pair 的第一个元素的相等性
    template <typename FL, typename SL, typename FR, typename SR>
    bool operator()(const std::pair<FL, SL>& lhs, const std::pair<FR, SR>& rhs) {
        return static_cast<equality_storage&>(*this)(lhs.first, rhs.first);
    }
};

// 静态常量，最小查找次数为4
static constexpr int8_t min_lookups = 4;

// 定义模板结构体 sherwood_v3_entry
template <typename T>
struct sherwood_v3_entry {
    // NOLINTNEXTLINE(modernize-use-equals-default)
    sherwood_v3_entry() {}

    // 构造函数，初始化距离
    sherwood_v3_entry(int8_t distance_from_desired)
        : distance_from_desired(distance_from_desired) {}

    // NOLINTNEXTLINE(modernize-use-equals-default)
    ~sherwood_v3_entry() {}

    // 返回是否有值
    bool has_value() const {
        return distance_from_desired >= 0;
    }

    // 返回是否为空
    bool is_empty() const {
        return distance_from_desired < 0;
    }

    // 返回是否在期望位置
    bool is_at_desired_position() const {
        return distance_from_desired <= 0;
    }

    // emplace 函数，创建对象并设置距离
    template <typename... Args>
    void emplace(int8_t distance, Args&&... args) {
        new (std::addressof(value)) T(std::forward<Args>(args)...);
        distance_from_desired = distance;
    }

    // 销毁值
    void destroy_value() {
        value.~T();
        distance_from_desired = -1;
    }

    // 指向前一个节点的指针
    sherwood_v3_entry<T>* prev = nullptr;

    // 指向后一个节点的指针
    sherwood_v3_entry<T>* next = nullptr;

    // 距离目标位置的距离
    int8_t distance_from_desired = -1;

    // 静态常量，特殊结束值为0
    static constexpr int8_t special_end_value = 0;

    // 联合体，存储值
    union {
        T value;
    };
};
// 定义一个内联函数，计算一个64位整数的对数(log2)，返回值为8位有符号整数(int8_t)
inline int8_t log2(uint64_t value) {
  // 预定义的对数表，每个数字表示对应2的幂的对数值的索引
  static constexpr std::array<int8_t, 64> table = {
      63, 0,  58, 1,  59, 47, 53, 2,  60, 39, 48, 27, 54, 33, 42, 3,
      61, 51, 37, 40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4,
      62, 57, 46, 52, 38, 26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21,
      56, 45, 25, 31, 35, 16, 9,  12, 44, 24, 15, 8,  23, 7,  6,  5};
  // 将输入值转换为不小于该值的2的幂，然后查表返回对应的对数值
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  value |= value >> 32;
  return table[((value - (value >> 1)) * 0x07EDD5E59A4E28C2) >> 58];
}

// 定义一个内联函数，计算比输入值大的下一个2的幂，返回值为64位无符号整数(uint64_t)
inline uint64_t next_power_of_two(uint64_t i) {
  // 减1后，将最高位以下的所有位设置为1，然后加1，得到下一个2的幂
  --i;
  i |= i >> 1;
  i |= i >> 2;
  i |= i >> 4;
  i |= i >> 8;
  i |= i >> 16;
  i |= i >> 32;
  ++i;
  return i;
}

// 以下结构和模板定义用于选择哈希策略

// 用于根据类型T是否定义了hash_policy类型选择哈希策略
template <typename... Ts>
struct make_void {
  typedef void type;
};

// 用于定义void_t，当作为模板的一部分使用
template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

// 根据类型T是否定义了hash_policy类型选择哈希策略，默认使用fibonacci_hash_policy
template <typename T, typename = void>
struct HashPolicySelector {
  typedef fibonacci_hash_policy type;
};

// 如果类型T定义了hash_policy类型，则选择其定义的哈希策略
template <typename T>
struct HashPolicySelector<T, void_t<typename T::hash_policy>> {
  typedef typename T::hash_policy type;
};

// 以下是sherwood_v3_table类的定义，实现了一个哈希表

template <
    typename T,
    typename FindKey,
    typename ArgumentHash,
    typename Hasher,
    typename ArgumentEqual,
    typename Equal,
    typename ArgumentAlloc,
    typename EntryAlloc>
class sherwood_v3_table : private EntryAlloc, private Hasher, private Equal {
  using Entry = detailv3::sherwood_v3_entry<T>;  // 使用sherwood_v3_entry定义条目类型
  using AllocatorTraits = std::allocator_traits<EntryAlloc>;  // 使用allocator_traits获取分配器特性
  using EntryPointer = typename AllocatorTraits::pointer;  // 使用分配器特性定义指向条目的指针类型

 public:
  struct convertible_to_iterator;  // 定义convertible_to_iterator结构体

  using value_type = T;  // 定义值类型为T
  using size_type = uint64_t;  // 定义大小类型为64位无符号整数
  using difference_type = std::ptrdiff_t;  // 定义差异类型为ptrdiff_t
  using hasher = ArgumentHash;  // 定义哈希器类型为ArgumentHash
  using key_equal = ArgumentEqual;  // 定义键相等比较器类型为ArgumentEqual
  using allocator_type = EntryAlloc;  // 定义分配器类型为EntryAlloc
  using reference = value_type&;  // 定义引用类型为值类型的引用
  using const_reference = const value_type&;  // 定义常量引用类型为值类型的常量引用
  using pointer = value_type*;  // 定义指针类型为值类型的指针
  using const_pointer = const value_type*;  // 定义常量指针类型为值类型的常量指针

  // 默认构造函数
  sherwood_v3_table() = default;
  
  // 带参数的构造函数，设置桶数量、哈希器、键相等比较器和分配器
  explicit sherwood_v3_table(
      size_type bucket_count,
      const ArgumentHash& hash = ArgumentHash(),
      const ArgumentEqual& equal = ArgumentEqual(),
      const ArgumentAlloc& alloc = ArgumentAlloc())
      : EntryAlloc(alloc), Hasher(hash), Equal(equal) {
    rehash(bucket_count);
  }
  # 构造函数：使用给定的桶数和默认哈希、相等性比较和分配器来初始化表格
  sherwood_v3_table(size_type bucket_count, const ArgumentAlloc& alloc)
      : sherwood_v3_table(
            bucket_count,
            ArgumentHash(),
            ArgumentEqual(),
            alloc) {}
  # 构造函数：使用给定的桶数、哈希、相等性比较和分配器来初始化表格
  sherwood_v3_table(
      size_type bucket_count,
      const ArgumentHash& hash,
      const ArgumentAlloc& alloc)
      : sherwood_v3_table(bucket_count, hash, ArgumentEqual(), alloc) {}
  # 构造函数：使用给定的分配器来初始化表格
  explicit sherwood_v3_table(const ArgumentAlloc& alloc) : EntryAlloc(alloc) {}
  # 构造函数：使用迭代器范围内的元素、给定的桶数、哈希、相等性比较和分配器来初始化表格
  template <typename It>
  sherwood_v3_table(
      It first,
      It last,
      size_type bucket_count = 0,
      const ArgumentHash& hash = ArgumentHash(),
      const ArgumentEqual& equal = ArgumentEqual(),
      const ArgumentAlloc& alloc = ArgumentAlloc())
      : sherwood_v3_table(bucket_count, hash, equal, alloc) {
    insert(first, last);
  }
  # 构造函数：使用迭代器范围内的元素、给定的桶数和分配器来初始化表格
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
  # 构造函数：使用迭代器范围内的元素、给定的桶数、哈希和分配器来初始化表格
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
  # 构造函数：使用初始化列表 il 中的元素、给定的桶数、哈希、相等性比较和分配器来初始化表格
  sherwood_v3_table(
      std::initializer_list<T> il,
      size_type bucket_count = 0,
      const ArgumentHash& hash = ArgumentHash(),
      const ArgumentEqual& equal = ArgumentEqual(),
      const ArgumentAlloc& alloc = ArgumentAlloc())
      : sherwood_v3_table(bucket_count, hash, equal, alloc) {
    if (bucket_count == 0)
      rehash(il.size());
    insert(il.begin(), il.end());
  }
  # 构造函数：使用初始化列表 il 中的元素、给定的桶数和分配器来初始化表格
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
  # 构造函数：使用初始化列表 il 中的元素、给定的桶数、哈希和分配器来初始化表格
  sherwood_v3_table(
      std::initializer_list<T> il,
      size_type bucket_count,
      const ArgumentHash& hash,
      const ArgumentAlloc& alloc)
      : sherwood_v3_table(il, bucket_count, hash, ArgumentEqual(), alloc) {}
  # 拷贝构造函数：使用另一个表格 other 和其分配器来初始化表格
  sherwood_v3_table(const sherwood_v3_table& other)
      : sherwood_v3_table(
            other,
            AllocatorTraits::select_on_container_copy_construction(
                other.get_allocator())) {}
  # 拷贝构造函数：使用另一个表格 other 和指定的分配器来初始化表格
  sherwood_v3_table(const sherwood_v3_table& other, const ArgumentAlloc& alloc)
      : EntryAlloc(alloc),
        Hasher(other),
        Equal(other),
        _max_load_factor(other._max_load_factor) {
    rehash_for_other_container(other);
    try {
      insert(other.begin(), other.end());
  } catch (...) {
    // 捕获所有异常，并执行清理和释放操作，然后重新抛出异常
    clear();
    deallocate_data(entries, num_slots_minus_one, max_lookups);
    throw;
  }
}
// 移动构造函数，从另一个对象接管资源，使用移动语义初始化成员变量
sherwood_v3_table(sherwood_v3_table&& other) noexcept
    : EntryAlloc(std::move(other)),
      Hasher(std::move(other)),
      Equal(std::move(other)) {
  swap_pointers(other);
}
// 带有自定义分配器参数的移动构造函数，从另一个对象接管资源，使用移动语义初始化成员变量
sherwood_v3_table(
    sherwood_v3_table&& other,
    const ArgumentAlloc& alloc) noexcept
    : EntryAlloc(alloc), Hasher(std::move(other)), Equal(std::move(other)) {
  swap_pointers(other);
}
// 拷贝赋值运算符重载，避免自我赋值，清空当前对象，然后根据条件重新分配资源和插入元素
sherwood_v3_table& operator=(const sherwood_v3_table& other) {
  if (this == std::addressof(other))
    return *this;

  clear();
  // 根据条件重置到空状态，以保证容器与被赋值对象使用相同的分配器
  if constexpr (AllocatorTraits::propagate_on_container_copy_assignment::
                    value) {
    if (static_cast<EntryAlloc&>(*this) !=
        static_cast<const EntryAlloc&>(other)) {
      reset_to_empty_state();
    }
    static_cast<EntryAlloc&>(*this) = other;
  }
  // 拷贝最大装载因子和哈希函数、相等函数
  _max_load_factor = other._max_load_factor;
  static_cast<Hasher&>(*this) = other;
  static_cast<Equal&>(*this) = other;
  // 重新哈希以适应被赋值对象的容器大小，并插入所有元素
  rehash_for_other_container(other);
  insert(other.begin(), other.end());
  return *this;
}
// 移动赋值运算符重载，避免自我赋值，根据条件进行资源的移交和重新分配
sherwood_v3_table& operator=(sherwood_v3_table&& other) noexcept {
  if (this == std::addressof(other))
    return *this;
  else if constexpr (AllocatorTraits::propagate_on_container_move_assignment::
                         value) {
    // 清空当前对象，重置为空状态，并使用移动语义设置分配器
    clear();
    reset_to_empty_state();
    static_cast<EntryAlloc&>(*this) = std::move(other);
    swap_pointers(other);
  } else if (
      static_cast<EntryAlloc&>(*this) == static_cast<EntryAlloc&>(other)) {
    // 如果分配器相同，直接交换指针
    swap_pointers(other);
  } else {
    // 清空当前对象，拷贝最大装载因子，并重新哈希以适应被移动对象的容器大小
    clear();
    _max_load_factor = other._max_load_factor;
    rehash_for_other_container(other);
    // 使用移动语义插入所有元素，并清空被移动对象
    for (T& elem : other)
      emplace(std::move(elem));
    other.clear();
  }
  // 使用移动语义设置哈希函数和相等函数
  static_cast<Hasher&>(*this) = std::move(other);
  static_cast<Equal&>(*this) = std::move(other);
  return *this;
}
// 析构函数，清空当前对象，并释放所有分配的数据
~sherwood_v3_table() {
  clear();
  deallocate_data(entries, num_slots_minus_one, max_lookups);
}

// 获取当前对象使用的分配器的常量引用
const allocator_type& get_allocator() const {
  return static_cast<const allocator_type&>(*this);
}
// 获取键值比较器的常量引用
const ArgumentEqual& key_eq() const {
  return static_cast<const ArgumentEqual&>(*this);
}
// 获取哈希函数的常量引用
const ArgumentHash& hash_function() const {
  return static_cast<const ArgumentHash&>(*this);
}

// 模板化迭代器结构，用于迭代容器中的元素
template <typename ValueType>
struct templated_iterator {
  templated_iterator() = default;
  // 使用给定指针初始化迭代器
  templated_iterator(EntryPointer current) : current(current) {}
  EntryPointer current = EntryPointer(); // 当前迭代器的指针

  // 迭代器类型标签为前向迭代器
  using iterator_category = std::forward_iterator_tag;
  using value_type = ValueType; // 值类型为模板参数指定的类型
  using difference_type = ptrdiff_t; // 差值类型为指针的差值类型
  using pointer = ValueType*; // 指针类型为指向值类型的指针
  using reference = ValueType&; // 引用类型为值类型的引用

  // 比较操作符，判断两个迭代器是否相等
  friend bool operator==(
      const templated_iterator& lhs,
      const templated_iterator& rhs) {
    return lhs.current == rhs.current;
  }
    // 定义不等于操作符重载函数，用于比较两个 templated_iterator 是否不相等
    friend bool operator!=(
        const templated_iterator& lhs,
        const templated_iterator& rhs) {
      return !(lhs == rhs);
    }

    // 前置递增操作符重载函数，使迭代器指向下一个节点并返回自身引用
    templated_iterator& operator++() {
      current = current->next;
      return *this;
    }
    
    // 后置递增操作符重载函数，先创建当前迭代器的副本，再将当前迭代器指向下一个节点，最后返回副本
    templated_iterator operator++(int) {
      templated_iterator copy(*this);
      ++*this;
      return copy;
    }

    // 解引用操作符重载函数，返回当前节点的值的引用
    ValueType& operator*() const {
      return current->value;
    }
    
    // 成员访问操作符重载函数，返回指向当前节点值的指针
    ValueType* operator->() const {
      return std::addressof(current->value);
    }

    // 模板类型转换操作符重载函数，当 value_type 是 const 类型时自动禁用，避免编译器警告
    template <
        class target_type = const value_type,
        class = std::enable_if_t<
            std::is_same_v<target_type, const value_type> &&
            !std::is_same_v<target_type, value_type>>>
    operator templated_iterator<target_type>() const {
      return {current};
    }
  };

  // 使用 value_type 实例化的迭代器类型
  using iterator = templated_iterator<value_type>;
  
  // 使用 const value_type 实例化的迭代器类型
  using const_iterator = templated_iterator<const value_type>;

  // 返回容器的起始迭代器，即指向第一个元素的迭代器
  iterator begin() {
    return sentinel->next;
  }
  
  // 返回容器的起始迭代器（常量版本），即指向第一个元素的常量迭代器
  const_iterator begin() const {
    return sentinel->next;
  }
  
  // 返回容器的起始迭代器（常量版本），即指向第一个元素的常量迭代器
  const_iterator cbegin() const {
    return begin();
  }

  // 返回容器的结束迭代器，即指向末尾后一个位置的迭代器
  iterator end() {
    return sentinel;
  }
  
  // 返回容器的结束迭代器（常量版本），即指向末尾后一个位置的常量迭代器
  const_iterator end() const {
    return sentinel;
  }
  
  // 返回容器的结束迭代器（常量版本），即指向末尾后一个位置的常量迭代器
  const_iterator cend() const {
    return end();
  }

  // 根据给定的 key 查找元素的迭代器，返回找到的第一个元素的迭代器，找不到则返回 end()
  iterator find(const FindKey& key) {
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

  // 根据给定的 key 查找元素的迭代器（常量版本），通过去除 const 属性调用非常量版本的 find 函数
  const_iterator find(const FindKey& key) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<sherwood_v3_table*>(this)->find(key);
  }

  // 返回与给定 key 相等的元素个数，存在返回 1，不存在返回 0
  uint64_t count(const FindKey& key) const {
    return find(key) == end() ? 0 : 1;
  }

  // 返回一个迭代器对，表示与给定 key 相等的元素范围
  std::pair<iterator, iterator> equal_range(const FindKey& key) {
    iterator found = find(key);
    if (found == end())
      return {found, found};
    else
      return {found, std::next(found)};
  }

  // 返回一个常量迭代器对，表示与给定 key 相等的元素范围
  std::pair<const_iterator, const_iterator> equal_range(
      const FindKey& key) const {
    const_iterator found = find(key);
    if (found == end())
      return {found, found};
    else
      return {found, std::next(found)};
  }

  // 将具有给定 key 和参数 args 的新元素插入到容器中，返回一个迭代器和插入是否成功的布尔值对
  template <typename Key, typename... Args>
  std::pair<iterator, bool> emplace(Key&& key, Args&&... args) {
    uint64_t index =
        hash_policy.index_for_hash(hash_object(key), num_slots_minus_one);
    EntryPointer current_entry = entries + ptrdiff_t(index);
    int8_t distance_from_desired = 0;
    // 省略部分代码...
    // 在当前位置向后查找，直到找到合适的插入位置或者达到相同距离
    for (; current_entry->distance_from_desired >= distance_from_desired;
         ++current_entry, ++distance_from_desired) {
      // 如果找到已存在的键，不改变顺序，直接返回包含当前条目的迭代器和 false
      if (compares_equal(key, current_entry->value))
        return {{current_entry}, false};
    }
    // 如果未找到已存在的键，调用 emplace_new_key 插入新键值对
    return emplace_new_key(
        distance_from_desired,
        current_entry,
        std::forward<Key>(key),
        std::forward<Args>(args)...);
  }

  // 插入给定值 value 的副本
  std::pair<iterator, bool> insert(const value_type& value) {
    return emplace(value);
  }
  // 插入给定值 value 的移动语义版本
  std::pair<iterator, bool> insert(value_type&& value) {
    return emplace(std::move(value));
  }
  // 使用 emplace 方法插入带提示位置的键值对
  template <typename... Args>
  iterator emplace_hint(const_iterator, Args&&... args) {
    return emplace(std::forward<Args>(args)...).first;
  }
  // 插入给定值 value 的副本，带有位置提示
  iterator insert(const_iterator, const value_type& value) {
    return emplace(value).first;
  }
  // 插入给定值 value 的移动语义版本，带有位置提示
  iterator insert(const_iterator, value_type&& value) {
    return emplace(std::move(value)).first;
  }

  // 插入迭代器范围内的元素
  template <typename It>
  void insert(It begin, It end) {
    for (; begin != end; ++begin) {
      emplace(*begin);
    }
  }
  // 插入初始化列表 il 中的元素
  void insert(std::initializer_list<value_type> il) {
    insert(il.begin(), il.end());
  }

  // 重新设置哈希表大小为 num_buckets
  void rehash(uint64_t num_buckets) {
    // 计算所需的桶数，至少为当前元素数量与最大负载因子的比值向上取整
    num_buckets = std::max(
        num_buckets,
        static_cast<uint64_t>(std::ceil(
            static_cast<double>(num_elements) /
            static_cast<double>(_max_load_factor))));
    // 如果请求的桶数为 0，重置为空状态并返回
    if (num_buckets == 0) {
      reset_to_empty_state();
      return;
    }
    // 计算新的桶数并判断是否需要进行 rehash 操作
    auto new_prime_index = hash_policy.next_size_over(num_buckets);
    if (num_buckets == bucket_count())
      return;
    // 计算新的最大查找次数
    int8_t new_max_lookups = compute_max_lookups(num_buckets);
    // 分配新的桶数组并初始化
    EntryPointer new_buckets(
        AllocatorTraits::allocate(*this, num_buckets + new_max_lookups));
    EntryPointer special_end_item =
        new_buckets + static_cast<ptrdiff_t>(num_buckets + new_max_lookups - 1);
    for (EntryPointer it = new_buckets; it != special_end_item; ++it)
      it->distance_from_desired = -1;
    special_end_item->distance_from_desired = Entry::special_end_value;
    // 交换旧的桶数组和新的桶数组，并更新相关状态
    std::swap(entries, new_buckets);
    std::swap(num_slots_minus_one, num_buckets);
    --num_slots_minus_one;
    hash_policy.commit(new_prime_index);
    int8_t old_max_lookups = max_lookups;
    max_lookups = new_max_lookups;
    num_elements = 0;

    // 备份链表的起始位置
    auto start = sentinel->next;
    // 将 sentinel 指向自身，重置链表状态
    reset_list();
    // 重新插入链表中的元素
    for (EntryPointer it = start; it != sentinel;) {
      auto next = it->next;
      emplace(std::move(it->value));
      it->destroy_value();
      it = next;
    }

    // 释放旧的桶数组和相关资源
    deallocate_data(new_buckets, num_buckets, old_max_lookups);
  }

  // 预留至少能容纳 num_elements_ 个元素的空间
  void reserve(uint64_t num_elements_) {
    uint64_t required_buckets = num_buckets_for_reserve(num_elements_);
    // 如果所需的桶数大于当前桶数，则重新分配哈希表空间
    if (required_buckets > bucket_count())
      rehash(required_buckets);
  }

  // 替换链表中特定位置的节点为新的节点
  void replace_linked_list_position(
      EntryPointer to_be_replaced,
      EntryPointer new_node) {
  // 从链表中移除指定的节点
  remove_from_list(new_node);

  // 将新节点插入到指定节点的前面
  insert_after(new_node, to_be_replaced->prev);

  // 从链表中移除待替换节点
  remove_from_list(to_be_replaced);
}

// erase 函数的返回值可以转换为迭代器类型
// 这么做的原因是查找指向下一个元素的迭代器并非免费的。
// 如果关心下一个迭代器，请将返回值转换为迭代器。
convertible_to_iterator erase(const_iterator to_erase) {
  // 获取待删除元素的指针
  EntryPointer current = to_erase.current;

  // 从链表中移除当前节点
  remove_from_list(current);

  // 销毁当前节点的值
  current->destroy_value();

  // 减少元素数量计数
  --num_elements;

  // 遍历当前节点之后的节点，将它们移动到它们期望的位置
  for (EntryPointer next = current + ptrdiff_t(1);
       !next->is_at_desired_position();
       ++current, ++next) {
    // 如果正在移除一个条目，并且有其他具有相同哈希值的条目，
    // 则通过重新插入将其他条目移动到它们的期望位置。
    current->emplace(next->distance_from_desired - 1, std::move(next->value));
    replace_linked_list_position(next, current);
    next->destroy_value();
  }

  // 返回一个包含被删除元素指针的可转换对象
  return {to_erase.current};
}

// erase 函数，删除给定范围内的元素
iterator erase(const_iterator begin_it, const_iterator end_it) {
  // 每当删除一个条目，并且有其他具有相同哈希值的条目时，必须将其他条目移动到它们的期望位置。
  // 任何对已移动条目的引用都会失效。
  // 在这里，我们遍历范围，并确保更新指向列表中下一个条目或迭代器末尾的指针，以防失效。

  auto curr_iter = begin_it.current;
  auto next_iter = curr_iter->next;
  auto end_iter = end_it.current;

  while (curr_iter != end_iter) {
    // 从链表中移除当前节点
    remove_from_list(curr_iter);

    // 销毁当前节点的值
    curr_iter->destroy_value();

    // 减少元素数量计数
    --num_elements;

    // 遍历当前节点之后的节点，将它们移动到它们期望的位置
    for (EntryPointer next_hash_slot = curr_iter + ptrdiff_t(1);
         !next_hash_slot->is_at_desired_position();
         ++curr_iter, ++next_hash_slot) {
      curr_iter->emplace(
          next_hash_slot->distance_from_desired - 1,
          std::move(next_hash_slot->value));
      replace_linked_list_position(next_hash_slot, curr_iter);
      next_hash_slot->destroy_value();

      // 如果正在失效 next_iter 或 end_iter 的引用
      if (next_hash_slot == end_iter) {
        end_iter = curr_iter;
      } else if (next_hash_slot == next_iter) {
        next_iter = curr_iter;
      }
    }
    curr_iter = next_iter;
    next_iter = curr_iter->next;
  }

  // 返回一个包含指向结束迭代器的迭代器对象
  return {end_iter};
}

// 根据给定的 key 删除元素，并返回删除的元素数量
uint64_t erase(const FindKey& key) {
  // 查找指定 key 的元素位置
  auto found = find(key);

  // 如果找不到，则返回删除 0 个元素
  if (found == end())
    return 0;
  else {
    // 否则，删除找到的元素并返回删除 1 个元素
    erase(found);
    return 1;
  }
}

// 清空容器，销毁所有元素
void clear() {
  // 遍历所有条目，销毁具有值的条目
  for (EntryPointer it = entries,
                    end = it +
           static_cast<ptrdiff_t>(num_slots_minus_one + max_lookups);
       it != end;
       ++it) {
    if (it->has_value())
      it->destroy_value();
  }

  // 重置链表状态
  reset_list();

  // 重置元素数量计数
  num_elements = 0;
}

void shrink_to_fit() {
  }

  // 交换两个sherwood_v3_table对象的内容
  void swap(sherwood_v3_table& other) noexcept {
    using std::swap;
    // 交换指针和基类的哈希函数对象
    swap_pointers(other);
    swap(static_cast<ArgumentHash&>(*this), static_cast<ArgumentHash&>(other));
    swap(
        static_cast<ArgumentEqual&>(*this), static_cast<ArgumentEqual&>(other));
    // 如果分配器支持容器交换，则交换分配器
    if (AllocatorTraits::propagate_on_container_swap::value)
      swap(static_cast<EntryAlloc&>(*this), static_cast<EntryAlloc&>(other));
  }

  // 返回当前表中的元素个数
  uint64_t size() const {
    return num_elements;
  }
  
  // 返回表的最大容量
  uint64_t max_size() const {
    return (AllocatorTraits::max_size(*this)) / sizeof(Entry);
  }
  
  // 返回当前表的桶数量
  uint64_t bucket_count() const {
    return num_slots_minus_one ? num_slots_minus_one + 1 : 0;
  }
  
  // 返回表的最大桶容量
  size_type max_bucket_count() const {
    return (AllocatorTraits::max_size(*this) - min_lookups) / sizeof(Entry);
  }
  
  // 返回给定键在表中的桶索引
  uint64_t bucket(const FindKey& key) const {
    return hash_policy.index_for_hash(hash_object(key), num_slots_minus_one);
  }
  
  // 返回当前加载因子
  float load_factor() const {
    uint64_t buckets = bucket_count();
    if (buckets)
      return static_cast<float>(num_elements) / bucket_count();
    else
      return 0;
  }
  
  // 设置或返回最大加载因子
  void max_load_factor(float value) {
    _max_load_factor = value;
  }
  
  // 返回当前最大加载因子
  float max_load_factor() const {
    return _max_load_factor;
  }

  // 返回是否为空表
  bool empty() const {
    return num_elements == 0;
  }

 private:
  // 表的条目指针
  EntryPointer entries = empty_default_table();
  
  // 桶的数量减一
  uint64_t num_slots_minus_one = 0;
  
  // 哈希策略对象
  typename HashPolicySelector<ArgumentHash>::type hash_policy;
  
  // 最大探测次数
  int8_t max_lookups = detailv3::min_lookups - 1;
  
  // 最大加载因子
  float _max_load_factor = 0.5f;
  
  // 元素数量
  uint64_t num_elements = 0;
  
  // 表的哨兵值的独特指针
  std::unique_ptr<sherwood_v3_entry<T>> sentinel_val;

  // 双向链表的头部
  EntryPointer sentinel = initSentinel();

  // 初始化哨兵值并返回其指针
  EntryPointer initSentinel() {
    // 必须使用指针，以便哈希映射可以与前向声明的类型一起使用
    sentinel_val = std::make_unique<sherwood_v3_entry<T>>();
    sentinel = sentinel_val.get();
    reset_list();
    return sentinel;
  }

  // 返回一个默认空表的条目指针
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

  // 计算给定桶数量的最大探测次数
  static int8_t compute_max_lookups(uint64_t num_buckets) {
    int8_t desired = detailv3::log2(num_buckets);
    return std::max(detailv3::min_lookups, desired);
  }

  // 为给定元素数量计算预留桶的数量
  uint64_t num_buckets_for_reserve(uint64_t num_elements_) const {
    return static_cast<uint64_t>(std::ceil(
        static_cast<double>(num_elements_) /
        std::min(0.5, static_cast<double>(_max_load_factor))));
  }

  // 为另一个表重新哈希以匹配当前表
  void rehash_for_other_container(const sherwood_v3_table& other) {
  // 重新计算哈希表大小并重新哈希元素
  rehash(
      std::min(num_buckets_for_reserve(other.size()), other.bucket_count()));
}

// 交换两个哈希表的指针
void swap_pointers(sherwood_v3_table& other) {
  using std::swap;
  swap(hash_policy, other.hash_policy);
  swap(entries, other.entries);
  swap(num_slots_minus_one, other.num_slots_minus_one);
  swap(num_elements, other.num_elements);
  swap(max_lookups, other.max_lookups);
  swap(_max_load_factor, other._max_load_factor);
  swap(sentinel, other.sentinel);
  swap(sentinel_val, other.sentinel_val);
}

// 重置链表
void reset_list() {
  sentinel->next = sentinel;
  sentinel->prev = sentinel;
}

// 从链表中移除元素
void remove_from_list(EntryPointer elem) {
  elem->prev->next = elem->next;
  elem->next->prev = elem->prev;
}

// 在指定元素后插入新元素
void insert_after(EntryPointer new_elem, EntryPointer prev) {
  auto next = prev->next;

  prev->next = new_elem;
  new_elem->prev = prev;

  new_elem->next = next;
  next->prev = new_elem;
}

// 交换相邻节点的位置
void swap_adjacent_nodes(EntryPointer before, EntryPointer after) {
  // sentinel保持不变，因此before->prev不可能等于after
  auto before_prev = before->prev;
  auto after_next = after->next;

  before_prev->next = after;
  after->prev = before_prev;

  after_next->prev = before;
  before->next = after_next;

  before->prev = after;
  after->next = before;
}

// 交换两个节点的位置
void swap_positions(EntryPointer p1, EntryPointer p2) {
  if (p1 == p2) {
    return;
  }
  if (p1->next == p2) {
    return swap_adjacent_nodes(p1, p2);
  } else if (p2->next == p1) {
    return swap_adjacent_nodes(p2, p1);
  }

  auto p1_prev = p1->prev;
  auto p1_next = p1->next;

  auto p2_prev = p2->prev;
  auto p2_next = p2->next;

  p1_prev->next = p2;
  p2->prev = p1_prev;

  p1_next->prev = p2;
  p2->next = p1_next;

  p2_prev->next = p1;
  p1->prev = p2_prev;

  p2_next->prev = p1;
  p1->next = p2_next;
}

// 将新尾部元素追加到链表
void append_to_list(EntryPointer new_tail) {
  insert_after(new_tail, sentinel->prev);
}

// 插入新键值对
template <typename Key, typename... Args>
SKA_NOINLINE(std::pair<iterator, bool>)
emplace_new_key(
    int8_t distance_from_desired,
    EntryPointer current_entry,
    Key&& key,
    Args&&... args) {
  using std::swap;
  if (num_slots_minus_one == 0 || distance_from_desired == max_lookups ||
      static_cast<double>(num_elements + 1) >
          static_cast<double>(num_slots_minus_one + 1) *
              static_cast<double>(_max_load_factor)) {
    grow();
    return emplace(std::forward<Key>(key), std::forward<Args>(args)...);
  } else if (current_entry->is_empty()) {
    current_entry->emplace(
        distance_from_desired,
        std::forward<Key>(key),
        std::forward<Args>(args)...);
    ++num_elements;
    append_to_list(current_entry);
    return {{current_entry}, true};
  }
  value_type to_insert(std::forward<Key>(key), std::forward<Args>(args)...);
  swap(distance_from_desired, current_entry->distance_from_desired);
}
    // 我们保持的不变条件是：
    // - result.current_entry 包含我们正在插入的新值，并且在要插入的 LinkedList 位置上
    // - to_insert 包含表示 result.current_entry 位置的值
    swap(to_insert, current_entry->value);
    iterator result = {current_entry};
    for (++distance_from_desired, ++current_entry;; ++current_entry) {
      if (current_entry->is_empty()) {
        current_entry->emplace(distance_from_desired, std::move(to_insert));
        append_to_list(current_entry);
        // 现在我们可以将被替换的值交换回其正确的位置，
        // 将我们正在插入的新值放在列表的最前面
        swap_positions(current_entry, result.current);
        ++num_elements;
        return {result, true};
      } else if (current_entry->distance_from_desired < distance_from_desired) {
        swap(distance_from_desired, current_entry->distance_from_desired);
        swap(to_insert, current_entry->value);
        // 为了保持我们的不变条件，我们需要交换 result.current 和 current_entry 的位置
        swap_positions(result.current, current_entry);
        ++distance_from_desired;
      } else {
        ++distance_from_desired;
        if (distance_from_desired == max_lookups) {
          // 被替换的元素被放回其正确的位置
          // 我们扩展哈希表，然后尝试重新插入新元素
          swap(to_insert, result.current->value);
          grow();
          return emplace(std::move(to_insert));
        }
      }
    }
  }
};
} // namespace detailv3

// 结构体，定义了一系列静态成员函数，用于不同的质数取模操作
struct prime_number_hash_policy {
  // 取模 0 的操作
  static uint64_t mod0(uint64_t) {
    return 0llu;
  }
  // 取模 2 的操作
  static uint64_t mod2(uint64_t hash) {
    return hash % 2llu;
  }
  // 取模 3 的操作
  static uint64_t mod3(uint64_t hash) {
    return hash % 3llu;
  }
  // 取模 5 的操作
  static uint64_t mod5(uint64_t hash) {
    return hash % 5llu;
  }
  // 取模 7 的操作
  static uint64_t mod7(uint64_t hash) {
    return hash % 7llu;
  }
  // 取模 11 的操作
  static uint64_t mod11(uint64_t hash) {
    return hash % 11llu;
  }
  // 取模 13 的操作
  static uint64_t mod13(uint64_t hash) {
    return hash % 13llu;
  }
  // 取模 17 的操作
  static uint64_t mod17(uint64_t hash) {
    return hash % 17llu;
  }
  // 取模 23 的操作
  static uint64_t mod23(uint64_t hash) {
    return hash % 23llu;
  }
  // 取模 29 的操作
  static uint64_t mod29(uint64_t hash) {
    return hash % 29llu;
  }
  // 取模 37 的操作
  static uint64_t mod37(uint64_t hash) {
    return hash % 37llu;
  }
  // 取模 47 的操作
  static uint64_t mod47(uint64_t hash) {
    return hash % 47llu;
  }
  // 取模 59 的操作
  static uint64_t mod59(uint64_t hash) {
    return hash % 59llu;
  }
  // 取模 73 的操作
  static uint64_t mod73(uint64_t hash) {
    return hash % 73llu;
  }
  // 取模 97 的操作
  static uint64_t mod97(uint64_t hash) {
    return hash % 97llu;
  }
  // 取模 127 的操作
  static uint64_t mod127(uint64_t hash) {
    return hash % 127llu;
  }
  // 取模 151 的操作
  static uint64_t mod151(uint64_t hash) {
    return hash % 151llu;
  }
  // 取模 197 的操作
  static uint64_t mod197(uint64_t hash) {
    return hash % 197llu;
  }
  // 取模 251 的操作
  static uint64_t mod251(uint64_t hash) {
    return hash % 251llu;
  }
  // 取模 313 的操作
  static uint64_t mod313(uint64_t hash) {
    return hash % 313llu;
  }
  // 取模 397 的操作
  static uint64_t mod397(uint64_t hash) {
    return hash % 397llu;
  }
  // 取模 499 的操作
  static uint64_t mod499(uint64_t hash) {
    return hash % 499llu;
  }
  // 取模 631 的操作
  static uint64_t mod631(uint64_t hash) {
    return hash % 631llu;
  }
  // 取模 797 的操作
  static uint64_t mod797(uint64_t hash) {
    return hash % 797llu;
  }
  // 取模 1009 的操作
  static uint64_t mod1009(uint64_t hash) {
    return hash % 1009llu;
  }
  // 取模 1259 的操作
  static uint64_t mod1259(uint64_t hash) {
    return hash % 1259llu;
  }
  // 取模 1597 的操作
  static uint64_t mod1597(uint64_t hash) {
    return hash % 1597llu;
  }
  // 取模 2011 的操作
  static uint64_t mod2011(uint64_t hash) {
    return hash % 2011llu;
  }
  // 取模 2539 的操作
  static uint64_t mod2539(uint64_t hash) {
    return hash % 2539llu;
  }
  // 取模 3203 的操作
  static uint64_t mod3203(uint64_t hash) {
    return hash % 3203llu;
  }
  // 取模 4027 的操作
  static uint64_t mod4027(uint64_t hash) {
    return hash % 4027llu;
  }
  // 取模 5087 的操作
  static uint64_t mod5087(uint64_t hash) {
    return hash % 5087llu;
  }
  // 取模 6421 的操作
  static uint64_t mod6421(uint64_t hash) {
    return hash % 6421llu;
  }
  // 取模 8089 的操作
  static uint64_t mod8089(uint64_t hash) {
    return hash % 8089llu;
  }
  // 取模 10193 的操作
  static uint64_t mod10193(uint64_t hash) {
    return hash % 10193llu;
  }
  // 取模 12853 的操作
  static uint64_t mod12853(uint64_t hash) {
    return hash % 12853llu;
  }
  // 取模 16193 的操作
  static uint64_t mod16193(uint64_t hash) {
    return hash % 16193llu;
  }
  // 取模 20399 的操作
  static uint64_t mod20399(uint64_t hash) {
    return hash % 20399llu;
  }
  // 取模 25717 的操作
  static uint64_t mod25717(uint64_t hash) {
    return hash % 25717llu;
  }
  // 取模 32401 的操作
  static uint64_t mod32401(uint64_t hash) {
    return hash % 32401llu;
  }
  // 取模 40823 的操作
  static uint64_t mod40823(uint64_t hash) {
  // 计算 hash 值对 40823 取模的结果
  return hash % 40823llu;
}
// 计算 hash 值对 51437 取模的结果
static uint64_t mod51437(uint64_t hash) {
  return hash % 51437llu;
}
// 计算 hash 值对 64811 取模的结果
static uint64_t mod64811(uint64_t hash) {
  return hash % 64811llu;
}
// 计算 hash 值对 81649 取模的结果
static uint64_t mod81649(uint64_t hash) {
  return hash % 81649llu;
}
// 计算 hash 值对 102877 取模的结果
static uint64_t mod102877(uint64_t hash) {
  return hash % 102877llu;
}
// 计算 hash 值对 129607 取模的结果
static uint64_t mod129607(uint64_t hash) {
  return hash % 129607llu;
}
// 计算 hash 值对 163307 取模的结果
static uint64_t mod163307(uint64_t hash) {
  return hash % 163307llu;
}
// 计算 hash 值对 205759 取模的结果
static uint64_t mod205759(uint64_t hash) {
  return hash % 205759llu;
}
// 计算 hash 值对 259229 取模的结果
static uint64_t mod259229(uint64_t hash) {
  return hash % 259229llu;
}
// 计算 hash 值对 326617 取模的结果
static uint64_t mod326617(uint64_t hash) {
  return hash % 326617llu;
}
// 计算 hash 值对 411527 取模的结果
static uint64_t mod411527(uint64_t hash) {
  return hash % 411527llu;
}
// 计算 hash 值对 518509 取模的结果
static uint64_t mod518509(uint64_t hash) {
  return hash % 518509llu;
}
// 计算 hash 值对 653267 取模的结果
static uint64_t mod653267(uint64_t hash) {
  return hash % 653267llu;
}
// 计算 hash 值对 823117 取模的结果
static uint64_t mod823117(uint64_t hash) {
  return hash % 823117llu;
}
// 计算 hash 值对 1037059 取模的结果
static uint64_t mod1037059(uint64_t hash) {
  return hash % 1037059llu;
}
// 计算 hash 值对 1306601 取模的结果
static uint64_t mod1306601(uint64_t hash) {
  return hash % 1306601llu;
}
// 计算 hash 值对 1646237 取模的结果
static uint64_t mod1646237(uint64_t hash) {
  return hash % 1646237llu;
}
// 计算 hash 值对 2074129 取模的结果
static uint64_t mod2074129(uint64_t hash) {
  return hash % 2074129llu;
}
// 计算 hash 值对 2613229 取模的结果
static uint64_t mod2613229(uint64_t hash) {
  return hash % 2613229llu;
}
// 计算 hash 值对 3292489 取模的结果
static uint64_t mod3292489(uint64_t hash) {
  return hash % 3292489llu;
}
// 计算 hash 值对 4148279 取模的结果
static uint64_t mod4148279(uint64_t hash) {
  return hash % 4148279llu;
}
// 计算 hash 值对 5226491 取模的结果
static uint64_t mod5226491(uint64_t hash) {
  return hash % 5226491llu;
}
// 计算 hash 值对 6584983 取模的结果
static uint64_t mod6584983(uint64_t hash) {
  return hash % 6584983llu;
}
// 计算 hash 值对 8296553 取模的结果
static uint64_t mod8296553(uint64_t hash) {
  return hash % 8296553llu;
}
// 计算 hash 值对 10453007 取模的结果
static uint64_t mod10453007(uint64_t hash) {
  return hash % 10453007llu;
}
// 计算 hash 值对 13169977 取模的结果
static uint64_t mod13169977(uint64_t hash) {
  return hash % 13169977llu;
}
// 计算 hash 值对 16593127 取模的结果
static uint64_t mod16593127(uint64_t hash) {
  return hash % 16593127llu;
}
// 计算 hash 值对 20906033 取模的结果
static uint64_t mod20906033(uint64_t hash) {
  return hash % 20906033llu;
}
// 计算 hash 值对 26339969 取模的结果
static uint64_t mod26339969(uint64_t hash) {
  return hash % 26339969llu;
}
// 计算 hash 值对 33186281 取模的结果
static uint64_t mod33186281(uint64_t hash) {
  return hash % 33186281llu;
}
// 计算 hash 值对 41812097 取模的结果
static uint64_t mod41812097(uint64_t hash) {
  return hash % 41812097llu;
}
// 计算 hash 值对 52679969 取模的结果
static uint64_t mod52679969(uint64_t hash) {
  return hash % 52679969llu;
}
// 计算 hash 值对 66372617 取模的结果
static uint64_t mod66372617(uint64_t hash) {
  return hash % 66372617llu;
}
// 计算 hash 值对 83624237 取模的结果
static uint64_t mod83624237(uint64_t hash) {
  return hash % 83624237llu;
}
// 计算 hash 值对 105359939 取模的结果
static uint64_t mod105359939(uint64_t hash) {
  return hash % 105359939llu;
}
// 计算 hash 值对 132745199 取模的结果
static uint64_t mod132745199(uint64_t hash) {
  return hash % 132745199llu;
}
// 计算 hash 值对 167248483 取模的结果
static uint64_t mod167248483(uint64_t hash) {
  return hash % 167248483llu;
}
// 计算 hash 值对 210719881 取模的结果
static uint64_t mod210719881(uint64_t hash) {
    // 对给定的哈希值取模，使用常数 210719881llu 作为模数
    static uint64_t mod210719881(uint64_t hash) {
        return hash % 210719881llu;
    }
    
    // 对给定的哈希值取模，使用常数 265490441llu 作为模数
    static uint64_t mod265490441(uint64_t hash) {
        return hash % 265490441llu;
    }
    
    // 对给定的哈希值取模，使用常数 334496971llu 作为模数
    static uint64_t mod334496971(uint64_t hash) {
        return hash % 334496971llu;
    }
    
    // 对给定的哈希值取模，使用常数 421439783llu 作为模数
    static uint64_t mod421439783(uint64_t hash) {
        return hash % 421439783llu;
    }
    
    // 对给定的哈希值取模，使用常数 530980861llu 作为模数
    static uint64_t mod530980861(uint64_t hash) {
        return hash % 530980861llu;
    }
    
    // 对给定的哈希值取模，使用常数 668993977llu 作为模数
    static uint64_t mod668993977(uint64_t hash) {
        return hash % 668993977llu;
    }
    
    // 对给定的哈希值取模，使用常数 842879579llu 作为模数
    static uint64_t mod842879579(uint64_t hash) {
        return hash % 842879579llu;
    }
    
    // 对给定的哈希值取模，使用常数 1061961721llu 作为模数
    static uint64_t mod1061961721(uint64_t hash) {
        return hash % 1061961721llu;
    }
    
    // 对给定的哈希值取模，使用常数 1337987929llu 作为模数
    static uint64_t mod1337987929(uint64_t hash) {
        return hash % 1337987929llu;
    }
    
    // 对给定的哈希值取模，使用常数 1685759167llu 作为模数
    static uint64_t mod1685759167(uint64_t hash) {
        return hash % 1685759167llu;
    }
    
    // 对给定的哈希值取模，使用常数 2123923447llu 作为模数
    static uint64_t mod2123923447(uint64_t hash) {
        return hash % 2123923447llu;
    }
    
    // 对给定的哈希值取模，使用常数 2675975881llu 作为模数
    static uint64_t mod2675975881(uint64_t hash) {
        return hash % 2675975881llu;
    }
    
    // 对给定的哈希值取模，使用常数 3371518343llu 作为模数
    static uint64_t mod3371518343(uint64_t hash) {
        return hash % 3371518343llu;
    }
    
    // 对给定的哈希值取模，使用常数 4247846927llu 作为模数
    static uint64_t mod4247846927(uint64_t hash) {
        return hash % 4247846927llu;
    }
    
    // 对给定的哈希值取模，使用常数 5351951779llu 作为模数
    static uint64_t mod5351951779(uint64_t hash) {
        return hash % 5351951779llu;
    }
    
    // 对给定的哈希值取模，使用常数 6743036717llu 作为模数
    static uint64_t mod6743036717(uint64_t hash) {
        return hash % 6743036717llu;
    }
    
    // 对给定的哈希值取模，使用常数 8495693897llu 作为模数
    static uint64_t mod8495693897(uint64_t hash) {
        return hash % 8495693897llu;
    }
    
    // 对给定的哈希值取模，使用常数 10703903591llu 作为模数
    static uint64_t mod10703903591(uint64_t hash) {
        return hash % 10703903591llu;
    }
    
    // 对给定的哈希值取模，使用常数 13486073473llu 作为模数
    static uint64_t mod13486073473(uint64_t hash) {
        return hash % 13486073473llu;
    }
    
    // 对给定的哈希值取模，使用常数 16991387857llu 作为模数
    static uint64_t mod16991387857(uint64_t hash) {
        return hash % 16991387857llu;
    }
    
    // 对给定的哈希值取模，使用常数 21407807219llu 作为模数
    static uint64_t mod21407807219(uint64_t hash) {
        return hash % 21407807219llu;
    }
    
    // 对给定的哈希值取模，使用常数 26972146961llu 作为模数
    static uint64_t mod26972146961(uint64_t hash) {
        return hash % 26972146961llu;
    }
    
    // 对给定的哈希值取模，使用常数 33982775741llu 作为模数
    static uint64_t mod33982775741(uint64_t hash) {
        return hash % 33982775741llu;
    }
    
    // 对给定的哈希值取模，使用常数 42815614441llu 作为模数
    static uint64_t mod42815614441(uint64_t hash) {
        return hash % 42815614441llu;
    }
    
    // 对给定的哈希值取模，使用常数 53944293929llu 作为模数
    static uint64_t mod53944293929(uint64_t hash) {
        return hash % 53944293929llu;
    }
    
    // 对给定的哈希值取模，使用常数 67965551447llu 作为模数
    static uint64_t mod67965551447(uint64_t hash) {
        return hash % 67965551447llu;
    }
    
    // 对给定的哈希值取模，使用常数 85631228929llu 作为模数
    static uint64_t mod85631228929(uint64_t hash) {
        return hash % 85631228929llu;
    }
    
    // 对给定的哈希值取模，使用常数 107888587883llu 作为模数
    static uint64_t mod107888587883(uint64_t hash) {
        return hash % 107888587883llu;
    }
    
    // 对给定的哈希值取模，使用常数 135931102921llu 作为模数
    static uint64_t mod135931102921(uint64_t hash) {
        return hash % 135931102921llu;
    }
    
    // 对给定的哈希值取模，使用常数 171262457903llu 作为模数
    static uint64_t mod171262457903(uint64_t hash) {
        return hash % 171262457903llu;
    }
    
    // 对给定的哈希值取模，使用常数 215777175787llu 作为模数
    static uint64_t mod215777175787(uint64_t hash) {
        return hash % 215777175787llu;
    }
    
    // 对给定的哈希值取模，使用常数 271862205833llu 作为模数
    static uint64_t mod271862205833(uint64_t hash) {
        return hash % 271862205833llu;
    }
    
    // 对给定的哈希值取模，使用常数 342524915839llu 作为模数
    static uint64_t mod342524915839(uint64_t hash) {
        return hash % 342524915839llu;
    }
    
    // 对给定的哈希值取模，使用常数 431554351609llu 作为模数
    static uint64_t mod431554351609(uint64_t hash) {
        return hash % 431554351609llu;
    }
    
    // 对给定的哈希值取模，使用常数 543724411781llu 作为模数
    static uint64_t mod543724411781(uint64_t hash) {
        return hash % 543724411781llu;
    }
  static uint64_t mod883823312134381(uint64_t hash) {
    // 计算给定哈希值对 883823312134381llu 取模后的结果
    return hash % 883823312134381llu;
  }
    // 对输入的哈希值取模，返回结果与 883823312134381llu 取模后的余数
    static uint64_t mod883823312134381(uint64_t hash) {
        return hash % 883823312134381llu;
    }
    
    // 对输入的哈希值取模，返回结果与 1113547595345903llu 取模后的余数
    static uint64_t mod1113547595345903(uint64_t hash) {
        return hash % 1113547595345903llu;
    }
    
    // 对输入的哈希值取模，返回结果与 1402982055436147llu 取模后的余数
    static uint64_t mod1402982055436147(uint64_t hash) {
        return hash % 1402982055436147llu;
    }
    
    // 对输入的哈希值取模，返回结果与 1767646624268779llu 取模后的余数
    static uint64_t mod1767646624268779(uint64_t hash) {
        return hash % 1767646624268779llu;
    }
    
    // 对输入的哈希值取模，返回结果与 2227095190691797llu 取模后的余数
    static uint64_t mod2227095190691797(uint64_t hash) {
        return hash % 2227095190691797llu;
    }
    
    // 对输入的哈希值取模，返回结果与 2805964110872297llu 取模后的余数
    static uint64_t mod2805964110872297(uint64_t hash) {
        return hash % 2805964110872297llu;
    }
    
    // 对输入的哈希值取模，返回结果与 3535293248537579llu 取模后的余数
    static uint64_t mod3535293248537579(uint64_t hash) {
        return hash % 3535293248537579llu;
    }
    
    // 对输入的哈希值取模，返回结果与 4454190381383713llu 取模后的余数
    static uint64_t mod4454190381383713(uint64_t hash) {
        return hash % 4454190381383713llu;
    }
    
    // 对输入的哈希值取模，返回结果与 5611928221744609llu 取模后的余数
    static uint64_t mod5611928221744609(uint64_t hash) {
        return hash % 5611928221744609llu;
    }
    
    // 对输入的哈希值取模，返回结果与 7070586497075177llu 取模后的余数
    static uint64_t mod7070586497075177(uint64_t hash) {
        return hash % 7070586497075177llu;
    }
    
    // 对输入的哈希值取模，返回结果与 8908380762767489llu 取模后的余数
    static uint64_t mod8908380762767489(uint64_t hash) {
        return hash % 8908380762767489llu;
    }
    
    // 对输入的哈希值取模，返回结果与 11223856443489329llu 取模后的余数
    static uint64_t mod11223856443489329(uint64_t hash) {
        return hash % 11223856443489329llu;
    }
    
    // 对输入的哈希值取模，返回结果与 14141172994150357llu 取模后的余数
    static uint64_t mod14141172994150357(uint64_t hash) {
        return hash % 14141172994150357llu;
    }
    
    // 对输入的哈希值取模，返回结果与 17816761525534927llu 取模后的余数
    static uint64_t mod17816761525534927(uint64_t hash) {
        return hash % 17816761525534927llu;
    }
    
    // 对输入的哈希值取模，返回结果与 22447712886978529llu 取模后的余数
    static uint64_t mod22447712886978529(uint64_t hash) {
        return hash % 22447712886978529llu;
    }
    
    // 对输入的哈希值取模，返回结果与 28282345988300791llu 取模后的余数
    static uint64_t mod28282345988300791(uint64_t hash) {
        return hash % 28282345988300791llu;
    }
    
    // 对输入的哈希值取模，返回结果与 35633523051069991llu 取模后的余数
    static uint64_t mod35633523051069991(uint64_t hash) {
        return hash % 35633523051069991llu;
    }
    
    // 对输入的哈希值取模，返回结果与 44895425773957261llu 取模后的余数
    static uint64_t mod44895425773957261(uint64_t hash) {
        return hash % 44895425773957261llu;
    }
    
    // 对输入的哈希值取模，返回结果与 56564691976601587llu 取模后的余数
    static uint64_t mod56564691976601587(uint64_t hash) {
        return hash % 56564691976601587llu;
    }
    
    // 对输入的哈希值取模，返回结果与 71267046102139967llu 取模后的余数
    static uint64_t mod71267046102139967(uint64_t hash) {
        return hash % 71267046102139967llu;
    }
    
    // 对输入的哈希值取模，返回结果与 89790851547914507llu 取模后的余数
    static uint64_t mod89790851547914507(uint64_t hash) {
        return hash % 89790851547914507llu;
    }
    
    // 对输入的哈希值取模，返回结果与 113129383953203213llu 取模后的余数
    static uint64_t mod113129383953203213(uint64_t hash) {
        return hash % 113129383953203213llu;
    }
    
    // 对输入的哈希值取模，返回结果与 142534092204280003llu 取模后的余数
    static uint64_t mod142534092204280003(uint64_t hash) {
        return hash % 142534092204280003llu;
    }
    
    // 对输入的哈希值取模，返回结果与 179581703095829107llu 取模后的余数
    static uint64_t mod179581703095829107(uint64_t hash) {
        return hash % 179581703095829107llu;
    }
    
    // 对输入的哈希值取模，返回结果与 226258767906406483llu 取模后的余数
    static uint64_t mod226258767906406483(uint64_t hash) {
        return hash % 226258767906406483llu;
    }
    
    // 对输入的哈希值取模，返回结果与 285068184408560057llu 取模后的余数
    static uint64_t mod285068184408560057(uint64_t hash) {
        return hash % 285068184408560057llu;
    }
    
    // 对输入的哈希值取模，返回结果与 359163406191658253llu 取模后的余数
    static uint64_t mod359163406191658253(uint64_t hash) {
        return hash % 359163406191658253llu;
    }
    
    // 对输入的哈希值取模，返回结果与 452517535812813007llu 取模后的余数
    static uint64_t mod452517535812813007(uint64_t hash) {
        return hash % 452517535812813007llu;
    }
    
    // 对输入的哈希值取模，返回结果与 570136368817120201llu 取模后的余数
    static uint64_t mod570136368817120201(uint64_t hash) {
        return hash % 570136368817120201llu;
    }
    
    // 对输入的哈希值取模，返回结果与 718326812383316683llu 取模后的余数
    static uint64_t mod718326812383316683(uint64_t hash) {
        return hash % 718326812383316683llu;
    }
    
    // 对输入的哈希值取模，返回结果与 905035071625626043llu 取模后的余数
    static uint64_t mod905035071625626043(uint64_t hash) {
        return hash % 905035071625626043llu;
    }
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 905035071625626043llu;
}
static uint64_t mod1140272737634240411(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 1140272737634240411llu;
}
static uint64_t mod1436653624766633509(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 1436653624766633509llu;
}
static uint64_t mod1810070143251252131(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 1810070143251252131llu;
}
static uint64_t mod2280545475268481167(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 2280545475268481167llu;
}
static uint64_t mod2873307249533267101(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 2873307249533267101llu;
}
static uint64_t mod3620140286502504283(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 3620140286502504283llu;
}
static uint64_t mod4561090950536962147(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 4561090950536962147llu;
}
static uint64_t mod5746614499066534157(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 5746614499066534157llu;
}
static uint64_t mod7240280573005008577(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 7240280573005008577llu;
}
static uint64_t mod9122181901073924329(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 9122181901073924329llu;
}
static uint64_t mod11493228998133068689(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 11493228998133068689llu;
}
static uint64_t mod14480561146010017169(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 14480561146010017169llu;
}
static uint64_t mod18446744073709551557(uint64_t hash) {
  // 使用给定的素数进行哈希值的模运算，返回结果
  return hash % 18446744073709551557llu;
}

using mod_function = uint64_t (*)(uint64_t);

mod_function next_size_over(uint64_t& size) const {
  // 通过一系列素数生成方法获取适合大小的哈希函数
  // 这些素数保证了在调用 reserve() 时不会出现不幸的大小
  // 最后一个素数是 2^64 的前一个质数
  const uint64_t* found = std::lower_bound(
      std::begin(prime_list), std::end(prime_list) - 1, size);
  size = *found;
  return mod_functions[1 + found - prime_list];
}
void commit(mod_function new_mod_function) {
  // 更新当前使用的哈希函数
  current_mod_function = new_mod_function;
}
void reset() {
  // 重置为默认的哈希函数
  current_mod_function = &mod0;
}

uint64_t index_for_hash(uint64_t hash, uint64_t /*num_slots_minus_one*/) const {
  // 使用当前选择的哈希函数计算哈希值的索引
  return current_mod_function(hash);
}
uint64_t keep_in_range(uint64_t index, uint64_t num_slots_minus_one) const {
  // 如果索引超出范围，则使用当前选择的哈希函数进行修正
  return index > num_slots_minus_one ? current_mod_function(index) : index;
}
};

// 哈希策略：幂次方为二的哈希策略
struct power_of_two_hash_policy {
  // 根据哈希和槽位数计算索引
  uint64_t index_for_hash(uint64_t hash, uint64_t num_slots_minus_one) const {
    return hash & num_slots_minus_one;
  }
  // 确保索引在有效范围内
  uint64_t keep_in_range(uint64_t index, uint64_t num_slots_minus_one) const {
    return index_for_hash(index, num_slots_minus_one);
  }
  // 计算大于当前大小的下一个幂次方的大小，并返回偏移量
  int8_t next_size_over(uint64_t& size) const {
    size = detailv3::next_power_of_two(size);
    return 0;
  }
  // 提交操作，无特定功能
  void commit(int8_t) {}
  // 重置操作，无特定功能
  void reset() {}
};

// 哈希策略：斐波那契哈希策略
struct fibonacci_hash_policy {
  // 根据哈希计算索引
  uint64_t index_for_hash(uint64_t hash, uint64_t /*num_slots_minus_one*/) const {
    return (11400714819323198485ull * hash) >> shift;
  }
  // 确保索引在有效范围内
  uint64_t keep_in_range(uint64_t index, uint64_t num_slots_minus_one) const {
    return index & num_slots_minus_one;
  }

  // 计算大于当前大小的下一个幂次方的大小，并返回偏移量
  int8_t next_size_over(uint64_t& size) const {
    size = std::max(uint64_t(2), detailv3::next_power_of_two(size));
    return static_cast<int8_t>(64 - detailv3::log2(size));
  }
  // 提交操作，设置斐波那契哈希的位移量
  void commit(int8_t shift_) {
    shift = shift_;
  }
  // 重置操作，将位移量重置为默认值 63
  void reset() {
    shift = 63;
  }

 private:
  int8_t shift = 63; // 初始位移量为 63
};

// 有序保持扁平哈希映射类模板
template <
    typename K,
    typename V,
    typename H = std::hash<K>,
    typename E = std::equal_to<K>,
    typename A = std::allocator<std::pair<K, V>>>
class order_preserving_flat_hash_map
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
  using key_type = K; // 键类型为 K
  using mapped_type = V; // 值类型为 V

  using Table::Table; // 继承基类的构造函数

  order_preserving_flat_hash_map() = default; // 默认构造函数

  // 下标运算符重载，插入或访问元素
  inline V& operator[](const K& key) {
    return emplace(key, convertible_to_value()).first->second;
  }
  // 移动下标运算符重载，插入或访问元素
  inline V& operator[](K&& key) {
    return emplace(std::move(key), convertible_to_value()).first->second;
  }

  // 访问指定键的值，若键不存在则抛出异常
  V& at(const K& key) {
    auto found = this->find(key);
    if (found == this->end())
      throw std::out_of_range("Argument passed to at() was not in the map.");
    return found->second;
  }
  // 访问指定键的值的常量版本，若键不存在则抛出异常
  const V& at(const K& key) const {
    auto found = this->find(key);
    if (found == this->end())
      throw std::out_of_range("Argument passed to at() was not in the map.");
    return found->second;
  }

  using Table::emplace; // 继承基类的 emplace 方法

  // emplace 方法的重载，插入默认构造的键和值
  std::pair<typename Table::iterator, bool> emplace() {
    return emplace(key_type(), convertible_to_value());
  }
  // insert_or_assign 方法的模板重载，插入或分配给定键的值
  template <typename M>
  std::pair<typename Table::iterator, bool> insert_or_assign(
      const key_type& key,
      M&& m) {
    // 调用 emplace 方法插入键值对，返回插入结果
    auto emplace_result = emplace(key, std::forward<M>(m));
    // 如果插入未成功（键已存在），则更新已存在的键对应的值
    if (!emplace_result.second)
      emplace_result.first->second = std::forward<M>(m);
    // 返回 emplace 的结果（插入结果）
    return emplace_result;
  }

  // 插入或更新键值对，返回插入结果和是否更新标志
  template <typename M>
  std::pair<typename Table::iterator, bool> insert_or_assign(
      key_type&& key,
      M&& m) {
    // 调用 emplace 方法插入键值对，返回插入结果
    auto emplace_result = emplace(std::move(key), std::forward<M>(m));
    // 如果插入未成功（键已存在），则更新已存在的键对应的值
    if (!emplace_result.second)
      emplace_result.first->second = std::forward<M>(m);
    // 返回 emplace 的结果（插入结果和是否更新标志）
    return emplace_result;
  }

  // 插入或更新键值对，返回插入结果的迭代器
  template <typename M>
  typename Table::iterator insert_or_assign(
      typename Table::const_iterator,
      const key_type& key,
      M&& m) {
    // 调用上述 insert_or_assign 方法进行插入或更新，并返回插入结果的迭代器
    return insert_or_assign(key, std::forward<M>(m)).first;
  }

  // 插入或更新键值对，返回插入结果的迭代器
  template <typename M>
  typename Table::iterator insert_or_assign(
      typename Table::const_iterator,
      key_type&& key,
      M&& m) {
    // 调用上述 insert_or_assign 方法进行插入或更新，并返回插入结果的迭代器
    return insert_or_assign(std::move(key), std::forward<M>(m)).first;
  }

  // 比较两个 order_preserving_flat_hash_map 对象是否相等
  friend bool operator==(
      const order_preserving_flat_hash_map& lhs,
      const order_preserving_flat_hash_map& rhs) {
    // 如果两个对象大小不同，则它们不相等
    if (lhs.size() != rhs.size())
      return false;
    // 遍历左侧对象中的每个键值对
    for (const typename Table::value_type& value : lhs) {
      // 在右侧对象中查找当前键对应的值
      auto found = rhs.find(value.first);
      // 如果找不到该键或者对应的值不相等，则两个对象不相等
      if (found == rhs.end() || value.second != found->second)
        return false;
    }
    // 所有键值对都匹配，两个对象相等
    return true;
  }

  // 比较两个 order_preserving_flat_hash_map 对象是否不相等
  friend bool operator!=(
      const order_preserving_flat_hash_map& lhs,
      const order_preserving_flat_hash_map& rhs) {
    // 调用 == 运算符判断两个对象是否相等，返回相反的结果
    return !(lhs == rhs);
  }

 private:
  // 结构体，用于隐式转换为 V 类型
  struct convertible_to_value {
    // 隐式转换为 V 类型的操作符重载
    operator V() const {
      return V();
    }
  };
};

// 结束 flat_hash_set 类的定义

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
  using key_type = T;

  // 使用基类的构造函数
  using Table::Table;

  // 默认构造函数
  flat_hash_set() = default;

  // emplace 函数，用于插入元素
  template <typename... Args>
  std::pair<typename Table::iterator, bool> emplace(Args&&... args) {
    return Table::emplace(T(std::forward<Args>(args)...));
  }

  // emplace 函数的重载，接受 const key_type&
  std::pair<typename Table::iterator, bool> emplace(const key_type& arg) {
    return Table::emplace(arg);
  }

  // emplace 函数的重载，接受 key_type&
  std::pair<typename Table::iterator, bool> emplace(key_type& arg) {
    return Table::emplace(arg);
  }

  // emplace 函数的重载，接受 const key_type&&
  std::pair<typename Table::iterator, bool> emplace(const key_type&& arg) {
    return Table::emplace(std::move(arg));
  }

  // emplace 函数的重载，接受 key_type&&
  std::pair<typename Table::iterator, bool> emplace(key_type&& arg) {
    return Table::emplace(std::move(arg));
  }

  // 友元函数：判断两个 flat_hash_set 对象是否相等
  friend bool operator==(const flat_hash_set& lhs, const flat_hash_set& rhs) {
    // 如果两个集合大小不同，直接返回 false
    if (lhs.size() != rhs.size())
      return false;
    // 遍历左侧集合中的元素
    for (const T& value : lhs) {
      // 如果右侧集合中不存在该元素，则返回 false
      if (rhs.find(value) == rhs.end())
        return false;
    }
    // 全部元素比较完成，返回 true
    return true;
  }

  // 友元函数：判断两个 flat_hash_set 对象是否不相等
  friend bool operator!=(const flat_hash_set& lhs, const flat_hash_set& rhs) {
    // 利用 operator== 实现 operator!=
    return !(lhs == rhs);
  }
};

// 结束 flat_hash_set 类模板的定义

// 结构体 power_of_two_std_hash，继承自 std::hash<T>，使用 ska_ordered::power_of_two_hash_policy 作为 hash 策略
template <typename T>
struct power_of_two_std_hash : std::hash<T> {
  typedef ska_ordered::power_of_two_hash_policy hash_policy;
};

// 结束 power_of_two_std_hash 结构体的定义

} // namespace ska_ordered

// 结束 ska_ordered 命名空间的定义
```