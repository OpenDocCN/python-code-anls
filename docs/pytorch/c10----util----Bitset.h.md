# `.\pytorch\c10\util\Bitset.h`

```py
#pragma once

#include <cstddef>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace c10::utils {

/**
 * This is a simple bitset class with sizeof(long long int) bits.
 * You can set bits, unset bits, query bits by index,
 * and query for the first set bit.
 * Before using this class, please also take a look at std::bitset,
 * which has more functionality and is more generic. It is probably
 * a better fit for your use case. The sole reason for c10::utils::bitset
 * to exist is that std::bitset misses a find_first_set() method.
 */
struct bitset final {
 private:
#if defined(_MSC_VER)
  // MSVCs _BitScanForward64 expects int64_t
  using bitset_type = int64_t;
#else
  // POSIX ffsll expects long long int
  using bitset_type = long long int;
#endif

 public:
  // Returns the number of bits in the bitset type
  static constexpr size_t NUM_BITS() {
    return 8 * sizeof(bitset_type);
  }

  // Default constructor
  constexpr bitset() noexcept = default;

  // Copy constructor
  constexpr bitset(const bitset&) noexcept = default;

  // Move constructor
  constexpr bitset(bitset&&) noexcept = default;

  // Copy assignment operator
  bitset& operator=(const bitset&) noexcept = default;

  // Move assignment operator
  bitset& operator=(bitset&&) noexcept = default;

  // Set the bit at the given index
  constexpr void set(size_t index) noexcept {
    bitset_ |= (static_cast<long long int>(1) << index);
  }

  // Unset the bit at the given index
  constexpr void unset(size_t index) noexcept {
    bitset_ &= ~(static_cast<long long int>(1) << index);
  }

  // Get the value of the bit at the given index
  constexpr bool get(size_t index) const noexcept {
    return bitset_ & (static_cast<long long int>(1) << index);
  }

  // Check if all bits are unset
  constexpr bool is_entirely_unset() const noexcept {
    return 0 == bitset_;
  }

  // Call the given functor with the index of each bit that is set
  template <class Func>
  void for_each_set_bit(Func&& func) const {
    bitset cur = *this;
    size_t index = cur.find_first_set();
    while (0 != index) {
      // -1 because find_first_set() is not one-indexed.
      index -= 1;
      func(index);
      cur.unset(index);
      index = cur.find_first_set();
    }
  }

 private:
  // Return the index of the first set bit. The returned index is one-indexed
  // (i.e. if the very first bit is set, this function returns '1'), and a
  // return of '0' means that there was no bit set.
  size_t find_first_set() const {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long result;
    bool has_bits_set = (0 != _BitScanForward64(&result, bitset_));
    if (!has_bits_set) {
      return 0;
    }
    return result + 1;
#elif defined(_MSC_VER) && defined(_M_IX86)
    unsigned long result;
    if (static_cast<uint32_t>(bitset_) != 0) {
      bool has_bits_set =
          (0 != _BitScanForward(&result, static_cast<uint32_t>(bitset_)));
      if (!has_bits_set) {
        return 0;
      }
      return result + 1;
#endif
  }

 private:
  bitset_type bitset_;  // Internal storage for the bitset
};

}  // namespace c10::utils
    } else {
      // 检查位集合中是否有位被设置，并返回最低位的位置
      bool has_bits_set =
          (0 != _BitScanForward(&result, static_cast<uint32_t>(bitset_ >> 32)));
      // 如果没有位被设置，返回32（表示没有有效的位）
      if (!has_bits_set) {
        return 32;
      }
      // 返回最低位被设置的位置加上33（因为最低位的位置从0开始，但要返回的是从1开始的位数）
      return result + 33;
    }
#else
    // 如果不在编译环境中定义了 __builtin_ffsll，则返回 bitset_ 的最低位设置为 1 的索引（从 1 开始），否则执行对应平台的操作
    return __builtin_ffsll(bitset_);
#endif
  }

  // 比较操作符重载：比较两个 bitset 是否相等，如果相等则返回 true，否则返回 false
  friend bool operator==(bitset lhs, bitset rhs) noexcept {
    // 比较两个 bitset 的内部存储 bitset_ 是否相等
    return lhs.bitset_ == rhs.bitset_;
  }

  // bitset 类的不等于操作符重载：如果两个 bitset 不相等则返回 true，否则返回 false
  bitset_type bitset_{0};  // 内部存储的 bitset 数据，默认初始化为 0
};

// 全局作用域下的不等于操作符重载：如果两个 bitset 不相等则返回 true，否则返回 false
inline bool operator!=(bitset lhs, bitset rhs) noexcept {
  // 使用相等操作符重载函数来实现不等于操作
  return !(lhs == rhs);
}

} // namespace c10::utils
```