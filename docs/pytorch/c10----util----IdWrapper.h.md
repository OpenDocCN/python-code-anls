# `.\pytorch\c10\util\IdWrapper.h`

```
#pragma once

#include <cstddef>
#include <functional>
#include <utility>

namespace c10 {

/**
 * This template simplifies generation of simple classes that wrap an id
 * in a typesafe way. Namely, you can use it to create a very lightweight
 * type that only offers equality comparators and hashing. Example:
 *
 *   struct MyIdType final : IdWrapper<MyIdType, uint32_t> {
 *     constexpr explicit MyIdType(uint32_t id): IdWrapper(id) {}
 *   };
 *
 * Then in the global top level namespace:
 *
 *   C10_DEFINE_HASH_FOR_IDWRAPPER(MyIdType);
 *
 * That's it - equality operators and hash functions are automatically defined
 * for you, given the underlying type supports it.
 */
template <class ConcreteType, class UnderlyingType>
class IdWrapper {
 public:
  using underlying_type = UnderlyingType;
  using concrete_type = ConcreteType;

 protected:
  // Constructor to initialize the id_
  constexpr explicit IdWrapper(underlying_type id) noexcept(
      noexcept(underlying_type(std::declval<underlying_type>())))
      : id_(id) {}

  // Getter function to retrieve the underlying id_
  constexpr underlying_type underlyingId() const
      noexcept(noexcept(underlying_type(std::declval<underlying_type>()))) {
    return id_;
  }

 private:
  // Friend declaration for hash_value function
  friend size_t hash_value(const concrete_type& v) {
    return std::hash<underlying_type>()(v.id_);
  }

  // Friend declaration for equality operator ==
  // Note: Not noexcept for compatibility with older GCC versions
  friend constexpr bool operator==(
      const concrete_type& lhs,
      const concrete_type& rhs) noexcept {
    return lhs.id_ == rhs.id_;
  }

  // Friend declaration for inequality operator !=
  // Note: Not noexcept for compatibility with older GCC versions
  friend constexpr bool operator!=(
      const concrete_type& lhs,
      const concrete_type& rhs) noexcept {
    return !(lhs == rhs);
  }

  // Member variable to hold the underlying id
  underlying_type id_;
};

} // namespace c10

// Macro to define hash function for IdWrapper classes
#define C10_DEFINE_HASH_FOR_IDWRAPPER(ClassName) \
  namespace std {                                \
  template <>                                    \
  struct hash<ClassName> {                       \
    size_t operator()(ClassName x) const {       \
      return hash_value(x);                      \
    }                                            \
  };                                             \
  }
```