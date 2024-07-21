# `.\pytorch\c10\util\strong_type.h`

```py
/*
 * strong_type C++14/17/20 strong typedef library
 *
 * Copyright (C) Björn Fahller
 *
 *  Use, modification and distribution is subject to the
 *  Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 *
 * Project home: https://github.com/rollbear/strong_type
 */

#ifndef ROLLBEAR_STRONG_TYPE_HPP_INCLUDED
#define ROLLBEAR_STRONG_TYPE_HPP_INCLUDED

#include <functional>
#include <istream>
#include <ostream>
#include <type_traits>
#include <utility>

#ifndef STRONG_HAS_STD_FORMAT
#define STRONG_HAS_STD_FORMAT 0
#endif

#ifndef STRONG_HAS_FMT_FORMAT
#define STRONG_HAS_FMT_FORMAT 0
#endif

#if STRONG_HAS_STD_FORMAT
#include <format>
#if !defined(__cpp_lib_format) || __cpp_lib_format < 201907
#undef STRONG_HAS_STD_FORMAT
#define STRONG_HAS_STD_FORMAT 0
#endif
#endif

#if STRONG_HAS_FMT_FORMAT
#include <fmt/format.h>
#endif

namespace strong
{

namespace impl
{
  // 定义模板别名 WhenConstructible，用于检查类型 T 是否可以使用参数 V... 构造
  template <typename T, typename ... V>
  using WhenConstructible = std::enable_if_t<std::is_constructible<T, V...>::value>;
}

// modifier 模板别名，表示类型 M 对应的修改器，可以应用于类型 T
template <typename M, typename T>
using modifier = typename M::template modifier<T>;

// 未初始化标记类型 uninitialized_t 的定义
struct uninitialized_t {};
static constexpr uninitialized_t uninitialized{};

// 默认可构造类型 default_constructible 的定义
struct default_constructible
{
  // modifier 类模板，用于标记可应用于类型 T 的默认构造修改器
  template <typename T>
  class modifier
  {
  };
};

namespace impl {
  // supports_default_construction 函数模板，检查类型 T 是否支持默认构造
  template <typename T>
  constexpr bool supports_default_construction(const ::strong::default_constructible::modifier<T>*)
  {
    return true;
  }
}

// strong::type 类模板定义，包含类型 T、标签 Tag 和一组修改器 M...
template <typename T, typename Tag, typename ... M>
class type : public modifier<M, type<T, Tag, M...>>...
{
public:
  // 默认构造函数模板，当 T 是平凡构造的时候启用
  template <typename TT = T, typename = std::enable_if_t<std::is_trivially_constructible<TT>{}>>
  explicit type(uninitialized_t)
    noexcept
  {
  }

  // 默认构造函数模板，当支持默认构造时启用
  template <typename type_ = type,
            bool = impl::supports_default_construction(static_cast<type_*>(nullptr))>
  constexpr type()
    noexcept(noexcept(T{}))
  : val{}
  {
  }

  // 列表初始化构造函数模板，接受 std::initializer_list<U> 初始化列表
  template <typename U,
    typename = impl::WhenConstructible<T, std::initializer_list<U>>>
  constexpr explicit type(std::initializer_list<U> us)
    noexcept(noexcept(T{us}))
  : val{us}
  {
  }

  // 可变参数模板构造函数，接受参数 U&& ... u，并转发到类型 T 的构造函数
  template <typename ... U,
            typename = std::enable_if_t<std::is_constructible<T, U&&...>::value && (sizeof...(U) > 0)>>
  constexpr explicit type(U&& ... u)
    noexcept(std::is_nothrow_constructible<T, U...>::value)
  : val(std::forward<U>(u)...)
  {}

  // 友元函数 swap，用于交换两个 strong::type 对象的值
  friend constexpr void swap(type& a, type& b) noexcept(
                                                        std::is_nothrow_move_constructible<T>::value &&
                                                        std::is_nothrow_move_assignable<T>::value
                                                      )
  {
    using std::swap;
    // 使用 std::swap 对象 a 和 b 的值
    swap(a.val, b.val);
  }

private:
  T val; // 存储的实际值的成员变量
};

} // namespace strong

#endif // ROLLBEAR_STRONG_TYPE_HPP_INCLUDED
    // 交换两个变量的值（这里假设 swap 是一个自定义的函数或宏）
    swap(a.val, b.val);
    
    [[nodiscard]]
    // 返回当前对象的左值引用（当对象本身是左值时调用，且不抛出异常）
    constexpr T& value_of() & noexcept { return val;}
    [[nodiscard]]
    // 返回当前对象的常量左值引用（当对象本身是左值时调用，且不抛出异常）
    constexpr const T& value_of() const & noexcept { return val;}
    [[nodiscard]]
    // 返回当前对象的右值引用（当对象本身是右值时调用，且不抛出异常）
    constexpr T&& value_of() && noexcept { return std::move(val);}
    
    [[nodiscard]]
    // 友元函数：返回传入对象的左值引用（当对象本身是左值时调用，且不抛出异常）
    friend constexpr T& value_of(type& t) noexcept { return t.val;}
    [[nodiscard]]
    // 友元函数：返回传入对象的常量左值引用（当对象本身是左值时调用，且不抛出异常）
    friend constexpr const T& value_of(const type& t) noexcept { return t.val;}
    [[nodiscard]]
    // 友元函数：返回传入对象的右值引用（当对象本身是右值时调用，且不抛出异常）
    friend constexpr T&& value_of(type&& t) noexcept { return std::move(t).val;}
private:
  T val;
};

// 命名空间 impl 开始
namespace impl {
  // is_strong_type_func 模板函数：检查是否为强类型
  template <typename T, typename Tag, typename ... Ms>
  constexpr bool is_strong_type_func(const strong::type<T, Tag, Ms...>*) { return true;}
  // is_strong_type_func 模板函数的默认情况：不是强类型
  constexpr bool is_strong_type_func(...) { return false;}
  // underlying_type 函数模板：获取强类型的基础类型
  template <typename T, typename Tag, typename ... Ms>
  constexpr T underlying_type(strong::type<T, Tag, Ms...>*);

}

// is_strong_type 结构体模板：检查是否为强类型的特化
template <typename T>
struct is_strong_type : std::integral_constant<bool, impl::is_strong_type_func(static_cast<T *>(nullptr))> {};

// 命名空间 impl 开始
namespace impl {
  // WhenStrongType 模板别名：当为强类型时使用的 enable_if
  template <typename T>
  using WhenStrongType = std::enable_if_t<is_strong_type<std::decay_t<T>>::value>;
  // WhenNotStrongType 模板别名：当不是强类型时使用的 enable_if
  template <typename T>
  using WhenNotStrongType = std::enable_if_t<!is_strong_type<std::decay_t<T>>::value>;
}

// underlying_type 结构体模板：获取类型的基础类型
template <typename T, bool = is_strong_type<T>::value>
struct underlying_type
{
  using type = decltype(impl::underlying_type(static_cast<T*>(nullptr)));
};

// underlying_type 结构体模板的偏特化：当不是强类型时，基础类型即为自身
template <typename T>
struct underlying_type<T, false>
{
  using type = T;
};

// underlying_type_t 模板别名：获取类型的基础类型
template <typename T>
using underlying_type_t = typename underlying_type<T>::type;


// 命名空间 impl 开始
namespace impl {
  // access 函数模板：当不是强类型时，直接返回参数
  template<
    typename T,
    typename = impl::WhenNotStrongType<T>>
  constexpr
  T &&
  access(T &&t)
  noexcept {
    return std::forward<T>(t);
  }
  // access 函数模板的特化：当是强类型时，调用 value_of 函数获取值
  template <
    typename T,
    typename = impl::WhenStrongType<T>>
  [[nodiscard]]
  constexpr
  auto
  access(T&& t)
  noexcept
  -> decltype(value_of(std::forward<T>(t)))
  {
    return value_of(std::forward<T>(t));
  }

}

// equality 结构体定义开始
struct equality
{
  // modifier 模板类定义：用于强类型的比较操作符重载
  template <typename T>
  class modifier;
};

// equality::modifier 类模板特化：为 strong::type 的比较操作符重载
template <typename T, typename Tag, typename ... M>
class equality::modifier<::strong::type<T, Tag, M...>>
{
  // type 别名定义：strong::type 的类型
  using type = ::strong::type<T, Tag, M...>;
public:
  // operator== 操作符重载：强类型的相等比较
  [[nodiscard]]
  friend
  constexpr
  auto
  operator==(
    const type& lh,
    const type& rh)
  noexcept(noexcept(std::declval<const T&>() == std::declval<const T&>()))
  -> decltype(std::declval<const T&>() == std::declval<const T&>())
  {
    return value_of(lh) == value_of(rh);
  }

  // operator!= 操作符重载：强类型的不等比较
  [[nodiscard]]
  friend
  constexpr
  auto
  operator!=(
    const type& lh,
    const type& rh)
  noexcept(noexcept(std::declval<const T&>() != std::declval<const T&>()))
  -> decltype(std::declval<const T&>() != std::declval<const T&>())
  {
    return value_of(lh) != value_of(rh);
  }
};

// 命名空间 impl 开始
namespace impl
{
  // typed_equality 类模板定义：用于强类型与其他类型的比较操作
  template <typename T, typename Other>
  class typed_equality
  {
  private:
    // TT 别名定义：T 的基础类型
    using TT = underlying_type_t<T>;
    // OT 别名定义：Other 的基础类型
    using OT = underlying_type_t<Other>;
  public:
    // operator== 操作符重载：强类型与其他类型的相等比较
    [[nodiscard]]
    friend
    constexpr
    auto operator==(const T& lh, const Other& rh)
    noexcept(noexcept(std::declval<const TT&>() == std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() == std::declval<const OT&>())
    {
      return value_of(lh) == impl::access(rh);
    }
    // operator== 操作符重载（反向）：其他类型与强类型的相等比较
    [[nodiscard]]
    friend
    constexpr
    auto operator==(const Other& lh, const T& rh)
    noexcept(noexcept(std::declval<const OT&>() == std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() == std::declval<const TT&>())
    {
      return impl::access(lh) == value_of(rh);
    }
    {
      // 返回左操作数的实现访问结果是否等于右操作数的值
      return impl::access(lh) == value_of(rh) ;
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator!=(const T& lh, const Other rh)
    // noexcept 指定操作符不抛出异常，依赖于模板参数的不相等判断是否抛出异常
    noexcept(noexcept(std::declval<const TT&>() != std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() != std::declval<const OT&>())
    {
      // 返回左操作数的值不等于右操作数的实现访问结果
      return value_of(lh) != impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator!=(const Other& lh, const T& rh)
    // noexcept 指定操作符不抛出异常，依赖于模板参数的不相等判断是否抛出异常
    noexcept(noexcept(std::declval<const OT&>() != std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() != std::declval<const TT&>())
    {
      // 返回左操作数的实现访问结果不等于右操作数的值
      return impl::access(lh) != value_of(rh) ;
    }
  };
}
template <typename ... Ts>
struct equality_with
{
  // 模板结构，支持多个类型的等式比较
  template <typename T>
  // 修改器类，继承自所有参数类型的具体实现类
  class modifier : public impl::typed_equality<T, Ts>...
  {
  };
};

namespace impl
{
  // 模板结构，支持两个类型的比较
  template <typename T, typename Other>
  class typed_ordering
  {
  private:
    // 使用基础类型的别名
    using TT = underlying_type_t<T>;
    using OT = underlying_type_t<Other>;
  public:
    [[nodiscard]]
    friend
    constexpr
    auto operator<(const T& lh, const Other& rh)
    // 指定 noexcept 条件
    noexcept(noexcept(std::declval<const TT&>() < std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() < std::declval<const OT&>())
    {
      // 比较左值和右值
      return value_of(lh) < impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator<(const Other& lh, const T& rh)
    // 指定 noexcept 条件
    noexcept(noexcept(std::declval<const OT&>() < std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() < std::declval<const TT&>())
    {
      // 比较左值和右值
      return impl::access(lh) < value_of(rh) ;
    }

    [[nodiscard]]
    friend
    constexpr
    auto operator<=(const T& lh, const Other& rh)
    // 指定 noexcept 条件
    noexcept(noexcept(std::declval<const TT&>() <= std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() <= std::declval<const OT&>())
    {
      // 比较左值和右值
      return value_of(lh) <= impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator<=(const Other& lh, const T& rh)
    // 指定 noexcept 条件
    noexcept(noexcept(std::declval<const OT&>() <= std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() <= std::declval<const TT&>())
    {
      // 比较左值和右值
      return impl::access(lh) <= value_of(rh) ;
    }

    [[nodiscard]]
    friend
    constexpr
    auto operator>(const T& lh, const Other& rh)
    // 指定 noexcept 条件
    noexcept(noexcept(std::declval<const TT&>() > std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() > std::declval<const OT&>())
    {
      // 比较左值和右值
      return value_of(lh) > impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator>(const Other& lh, const T& rh)
    // 指定 noexcept 条件
    noexcept(noexcept(std::declval<const OT&>() > std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() > std::declval<const TT&>())
    {
      // 比较左值和右值
      return impl::access(lh) > value_of(rh) ;
    }

    [[nodiscard]]
    friend
    constexpr
    auto operator>=(const T& lh, const Other& rh)
    // 指定 noexcept 条件
    noexcept(noexcept(std::declval<const TT&>() >= std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() >= std::declval<const OT&>())
    {
      // 比较左值和右值
      return value_of(lh) >= impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator>=(const Other& lh, const T& rh)
    // 指定 noexcept 条件
    noexcept(noexcept(std::declval<const OT&>() >= std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() >= std::declval<const TT&>())
    {
      // 比较左值和右值
      return impl::access(lh) >= value_of(rh) ;
    }
  };
}

// 模板结构，支持多个类型的有序比较
template <typename ... Ts>
struct ordered_with
{
  // 修改器类，继承自所有参数类型的具体排序类
  template <typename T>
  class modifier : public impl::typed_ordering<T, Ts>...
  {
  };
};

namespace impl
{
  // 模板结构，要求类型支持复制构造
  template <typename T>
  struct require_copy_constructible
  {
    // 定义一个静态常量，检查底层类型是否可复制构造
    static constexpr bool value = std::is_copy_constructible<underlying_type_t<T>>::value;
    // 使用 static_assert 断言，确保底层类型必须是可复制构造的
    static_assert(value, "underlying type must be copy constructible");
  };

  // 模板定义，要求底层类型必须是可移动构造的
  template <typename T>
  struct require_move_constructible
  {
    static constexpr bool value = std::is_move_constructible<underlying_type_t<T>>::value;
    // 使用 static_assert 断言，确保底层类型必须是可移动构造的
    static_assert(value, "underlying type must be move constructible");
  };

  // 模板定义，要求底层类型必须是可复制赋值的
  template <typename T>
  struct require_copy_assignable
  {
    static constexpr bool value = std::is_copy_assignable<underlying_type_t<T>>::value;
    // 使用 static_assert 断言，确保底层类型必须是可复制赋值的
    static_assert(value, "underlying type must be copy assignable");
  };

  // 模板定义，要求底层类型必须是可移动赋值的
  template <typename T>
  struct require_move_assignable
  {
    static constexpr bool value = std::is_move_assignable<underlying_type_t<T>>::value;
    // 使用 static_assert 断言，确保底层类型必须是可移动赋值的
    static_assert(value, "underlying type must be move assignable");
  };

  // 模板定义，要求类型必须是半正则类型（支持复制构造、移动构造、复制赋值和移动赋值）
  template <bool> struct valid_type;
  template <>
  struct valid_type<true> {};

  template <typename T>
  struct require_semiregular
    : valid_type<require_copy_constructible<T>::value &&
                 require_move_constructible<T>::value &&
                 require_copy_assignable<T>::value &&
                 require_move_assignable<T>::value>
  {
  };
};

// 结构体定义结束

template <typename T>
class incrementable::modifier
{
public:
    // 前置自增运算符重载
    friend constexpr T& operator++(T& t)
    {
        // 调用 T 类型的自增运算符
        return ++value_of(t);
    }

    // 后置自增运算符重载
    friend constexpr T operator++(T& t, int)
    {
        // 调用 T 类型的自增运算符
        T result = t;
        ++t;
        return result;
    }
};

// 增量类型定义结束


这段代码定义了一系列的模板类和模板结构体，每个都有一个名为`modifier`的内部类，用于为特定的类型添加功能修改器。以下是详细的注释解释：

1. `};`
   - 结构体定义的结束标记。

2. `template <typename T>`
   - 定义一个模板类 `modifier`，接受一个类型参数 `T`。

3. `friend constexpr T& operator++(T& t)`
   - 定义了前置自增运算符重载，返回类型为 `T&`，参数为一个类型 `T` 的引用 `t`。

4. `return ++value_of(t);`
   - 调用 `value_of` 函数对 `t` 进行自增操作，并返回自增后的结果。

5. `friend constexpr T operator++(T& t, int)`
   - 定义了后置自增运算符重载，返回类型为 `T`，参数为一个类型 `T` 的引用 `t` 和一个整数。

6. `T result = t;`
   - 复制 `t` 到 `result`，准备返回未自增前的值。

7. `++t;`
   - 调用 `t` 的自增操作。

8. `return result;`
   - 返回未自增前的值 `result`。

9. `};`
   - 模板类 `modifier` 的定义结束标记。

10. `// 增量类型定义结束`
    - 注释标记，表示增量类型定义的结束。

这些注释提供了对每个代码片段的详细解释，包括类的作用以及每个方法的功能和实现细节。
    noexcept(noexcept(++std::declval<T&>().value_of()))
    {
      // 调用 value_of() 方法获取 t 对象的值并执行前缀递增操作，返回递增后的 t 对象
      ++value_of(t);
      return t;
    }

    friend
    constexpr
    T
    operator++(T& t, int)
    {
      // 复制 t 对象，然后执行 t 对象的后缀递增操作，并返回复制的原始 t 对象
      auto copy = t;
      ++t;
      return copy;
    }
  };
};

// 结构体 decrementable 的定义
struct decrementable
{
  // 模板类 modifier 的定义
  template <typename T>
  class modifier
  {
  public:
    // 声明友元函数 operator--，用于前缀递减操作符重载
    friend
    constexpr
    // 返回类型为 T 的引用
    T&
    operator--(T& t)
    // noexcept 声明，表明不抛出异常
    noexcept(noexcept(--std::declval<T&>().value_of()))
    {
      // 调用 value_of 函数对 t 进行前缀递减操作
      --value_of(t);
      return t;
    }

    // 声明友元函数 operator--，用于后缀递减操作符重载
    friend
    constexpr
    // 返回类型为 T
    T
    operator--(T& t, int)
    {
      // 复制 t 的值
      auto copy = t;
      // 对 t 进行前缀递减操作
      --t;
      // 返回复制的值
      return copy;
    }
  };
};

// 结构体 bicrementable 的定义
struct bicrementable
{
  // 模板类 modifier 的定义，继承自 incrementable::modifier<T> 和 decrementable::modifier<T>
  template <typename T>
  class modifier
    : public incrementable::modifier<T>
    , public decrementable::modifier<T>
  {
  };
};

// 结构体 boolean 的定义
struct boolean
{
  // 模板类 modifier 的定义
  template <typename T>
  class modifier
  {
  public:
    // 显式转换运算符，将类型 T 转换为 bool 类型
    explicit constexpr operator bool() const
    // noexcept 声明，表明不抛出异常
    noexcept(noexcept(static_cast<bool>(value_of(std::declval<const T&>()))))
    {
      // 获取当前对象的常量引用 self
      const auto& self = static_cast<const T&>(*this);
      // 调用 value_of 函数并将其结果转换为 bool 类型
      return static_cast<bool>(value_of(self));
    }
  };
};

// 结构体 hashable 的定义
struct hashable
{
  // 模板类 modifier 的定义，但未实现任何内容
  template <typename T>
  class modifier{};
};

// 结构体 difference 的定义
struct difference
{
  // 模板类 modifier 的声明
  template <typename T>
  class modifier;
};

// difference::modifier 模板类对 ::strong::type<T, Tag, M...> 的特化定义
template <typename T, typename Tag, typename ... M>
class difference::modifier<::strong::type<T, Tag, M...>>
// 继承自 ordered::modifier<::strong::type<T, Tag, M...>> 和 equality::modifier<::strong::type<T, Tag, M...>>
: public ordered::modifier<::strong::type<T, Tag, M...>>
, public equality::modifier<::strong::type<T, Tag, M...>>
{
  // 别名定义，表示 ::strong::type<T, Tag, M...>
  using type = ::strong::type<T, Tag, M...>;
public:
  // 声明友元函数 operator+=，用于复合赋值加法操作符重载
  friend
  constexpr
  // 返回类型为 type 的引用
  type& operator+=(type& lh, const type& rh)
  // noexcept 声明，表明不抛出异常
  noexcept(noexcept(value_of(lh) += value_of(rh)))
  {
    // 对 lh 的值执行复合赋值加法操作
    value_of(lh) += value_of(rh);
    return lh;
  }

  // 声明友元函数 operator-=，用于复合赋值减法操作符重载
  friend
  constexpr
  // 返回类型为 type 的引用
  type& operator-=(type& lh, const type& rh)
  // noexcept 声明，表明不抛出异常
  noexcept(noexcept(value_of(lh) -= value_of(rh)))
  {
    // 对 lh 的值执行复合赋值减法操作
    value_of(lh) -= value_of(rh);
    return lh;
  }

  // 声明友元函数 operator*=，用于复合赋值乘法操作符重载
  friend
  constexpr
  // 返回类型为 type 的引用
  type& operator*=(type& lh, const T& rh)
  // noexcept 声明，表明不抛出异常
  {
    // 对 lh 的值执行复合赋值乘法操作
    value_of(lh) *= rh;
    return lh;
  }

  // 声明友元函数 operator/=，用于复合赋值除法操作符重载
  friend
  constexpr
  // 返回类型为 type 的引用
  type& operator/=(type& lh, const T& rh)
  // noexcept 声明，表明不抛出异常
  noexcept(noexcept(value_of(lh) /= rh))
  {
    // 对 lh 的值执行复合赋值除法操作
    value_of(lh) /= rh;
    return lh;
  }

  // 声明友元函数 operator%=，用于复合赋值取模操作符重载
  template <typename TT = T, typename = decltype(std::declval<TT&>()%= std::declval<const TT&>())>
  friend
  constexpr
  // 返回类型为 type 的引用
  type& operator%=(type& lh, const T& rh)
  // noexcept 声明，表明不抛出异常
  {
    // 对 lh 的值执行复合赋值取模操作
    value_of(lh) %= rh;
    return lh;
  }

  // 声明友元函数 operator+，用于加法操作符重载
  friend
  constexpr
  // 返回类型为 type
  type operator+(type lh, const type& rh)
  {
    // 执行 lh += rh 操作
    lh += rh;
    return lh;
  }

  // 声明友元函数 operator-，用于减法操作符重载
  friend
  constexpr
  // 返回类型为 type
  type operator-(type lh, const type& rh)
  {
    // 执行 lh -= rh 操作
    lh -= rh;
    return lh;
  }

  // 声明友元函数 operator*，用于乘法操作符重载
  friend
  constexpr
  // 返回类型为 type
  type operator*(type lh, const T& rh)
  {
    // 执行 lh *= rh 操作
    lh *= rh;
    return lh;
  }

  // 声明友元函数 operator*，用于乘法操作符重载（反向）
  friend
  constexpr
  // 返回类型为 type
  type operator*(const T& lh, type rh)
  {
    // 执行 rh *= lh 操作
    rh *= lh;
    return rh;
  }

  // 声明友元函数 operator/，用于除法操作符重载
  friend
  constexpr
  // 返回类型为 type
  type operator/(type lh, const T& rh)
  {
    // 执行 lh /= rh 操作
    lh /= rh;
    return lh;
  }

  // 声明友元函数 operator/，用于除法操作符重载
  friend
  constexpr
  // 返回类型为 T
  T operator/(const type& lh, const type& rh)
  {
    // 返回 lh 的值除以 rh 的值
    return value_of(lh) / value_of(rh);
  }

  // 声明友元函数 operator%，用于取模操作符重载
  template <typename TT = T, typename = decltype(std::declval<TT&>() %= std::declval<const TT&>())>
  friend
  constexpr
  // 返回类型为 type
  type operator%(type lh, const T& rh)
    noexcept(noexcept(lh%= rh))
  {
      lh %= rh;
      return lh;
  }


// 定义模板函数 operator% ，用于重载取模运算符%
template <typename TT = T, typename = decltype(std::declval<TT>() % std::declval<TT>())>
friend
// 声明为 constexpr 函数，允许在编译时计算
constexpr
// 返回类型为 T
T operator%(type lh, type rh)
  // 指定 noexcept 条件，表示如果 value_of(lh) % value_of(rh) 不会抛出异常，则该函数也不会抛出异常
  noexcept(noexcept(value_of(lh) % value_of(rh)))
{
    // 对左操作数 lh 和右操作数 rh 执行取模运算
    return value_of(lh) % value_of(rh);
}
};

// 模板结构体 affine_point 的默认模板参数 D
template <typename D = void>
struct affine_point
{
  // 内部模板类 modifier，接受 strong::type<T, Tag, M...> 类型的模板参数
  template <typename T>
  class modifier;
};

// 命名空间 impl 中的实用工具模板定义
namespace impl
{
  // void_t 模板定义，接受任意数量的类型参数并返回 void
  template <typename ...>
  using void_t = void;

  // subtractable 结构体模板，用于检查类型 T 是否支持减法操作
  template <typename T, typename = void>
  struct subtractable : std::false_type {};

  // 针对支持减法操作的类型 T 的部分特化
  template <typename T>
  struct subtractable<T, void_t<decltype(std::declval<const T&>() - std::declval<const T&>())>>
  : std::true_type {};
}

// affine_point 结构体模板的 modifier 特化，针对 strong::type<T, Tag, M...> 类型
template <typename D>
template <typename T, typename Tag, typename ... M>
class affine_point<D>::modifier<::strong::type<T, Tag, M...>>
{
  // 使用 strong::type<T, Tag, M...> 定义别名 type
  using type = ::strong::type<T, Tag, M...>;

  // 静态断言，确保类型 T 支持减法操作
  static_assert(impl::subtractable<T>::value, "it must be possible to subtract instances of your underlying type");

  // 使用 decltype 推导出基础减法操作结果的类型，并定义为 base_diff_type
  using base_diff_type = decltype(std::declval<const T&>() - std::declval<const T&>());

public:
  // 差异类型 difference 的定义，根据 D 的类型条件性地选择 strong::type<base_diff_type, Tag, strong::difference>
  using difference = std::conditional_t<std::is_same<D, void>{}, strong::type<base_diff_type, Tag, strong::difference>, D>;

  // 静态断言，确保可以从 base_diff_type 构造 difference 类型
  static_assert(std::is_constructible<difference, base_diff_type>::value, "");

  // 返回差值的减法运算符重载，使用 constexpr 保证在编译时计算
  [[nodiscard]]
  friend constexpr difference operator-(const type& lh, const type& rh)
  {
    return difference(value_of(lh) - value_of(rh));
  }

  // 加法赋值运算符重载，增加差值到左操作数上
  friend constexpr type& operator+=(type& lh, const difference& d) noexcept(noexcept(value_of(lh) += impl::access(d)))
  {
    value_of(lh) += impl::access(d);
    return lh;
  }

  // 减法赋值运算符重载，减去差值从左操作数上
  friend constexpr type& operator-=(type& lh, const difference& d) noexcept(noexcept(value_of(lh) -= impl::access(d)))
  {
    value_of(lh) -= impl::access(d);
    return lh;
  }

  // 加法运算符重载，返回左操作数增加差值后的新对象
  [[nodiscard]]
  friend constexpr type operator+(type lh, const difference& d)
  {
    return lh += d;
  }

  // 加法运算符重载，返回右操作数增加差值后的新对象
  [[nodiscard]]
  friend constexpr type operator+(const difference& d, type rh)
  {
    return rh += d;
  }

  // 减法运算符重载，返回左操作数减去差值后的新对象
  [[nodiscard]]
  friend constexpr type operator-(type lh, const difference& d)
  {
    return lh -= d;
  }
};

// 结构体 pointer 的定义
struct pointer
{
  // pointer 结构体的 modifier 模板特化，针对 strong::type<T, Tag, M...> 类型
  template <typename T>
  class modifier;
};

// pointer 结构体模板的 modifier 特化，针对 strong::type<T, Tag, M...> 类型
template <typename T, typename Tag, typename ... M>
class pointer::modifier<::strong::type<T, Tag, M...>>
{
  // 使用 strong::type<T, Tag, M...> 定义别名 type
  using type = strong::type<T, Tag, M...>;

public:
  // 等于运算符重载，比较 type 对象是否等于 nullptr
  template <typename TT = T>
  [[nodiscard]]
  friend constexpr auto operator==(const type& t, std::nullptr_t) noexcept(noexcept(std::declval<const TT&>() == nullptr))
  -> decltype(std::declval<const TT&>() == nullptr)
  {
    return value_of(t) == nullptr;
  }

  // 等于运算符重载，比较 nullptr 是否等于 type 对象
  template <typename TT = T>
  [[nodiscard]]
  friend constexpr auto operator==(std::nullptr_t, const type& t) noexcept(noexcept(nullptr == std::declval<const TT&>()))
  -> decltype(nullptr == std::declval<const TT&>())
  {
    return value_of(t) == nullptr;
  }

  // 不等于运算符重载，比较 type 对象是否不等于 nullptr
  template <typename TT = T>
  [[nodiscard]]
  friend constexpr auto operator!=(const type& t, std::nullptr_t) noexcept(noexcept(std::declval<const TT&>() != nullptr))
  -> decltype(std::declval<const TT&>() != nullptr)
  {
    // 返回类型为 bool，判断给定类型 T 的值是否不等于 nullptr
    return value_of(t) != nullptr;
  }

  template <typename TT = T>
  [[nodiscard]]
  friend
  constexpr
  auto
  // 定义 != 运算符重载，左操作数为 nullptr，右操作数为类型的常量引用
  operator!=(
    std::nullptr_t,
    const type& t)
  // noexcept 指明 noexcept 条件
  noexcept(noexcept(nullptr != std::declval<const TT&>()))
  -> decltype(nullptr != std::declval<const TT&>())
  {
    // 调用 value_of 函数判断类型 T 的值是否不等于 nullptr
    return value_of(t) != nullptr;
  }

  // 返回类型为 T 的引用的解引用操作符重载
  [[nodiscard]]
  constexpr
  decltype(*std::declval<const T&>())
  operator*()
  const
  {
    // 获取 this 指针所指向对象的常量引用，并进行值的解引用
    auto& self = static_cast<const type&>(*this);
    return *value_of(self);  // 返回解引用后的值
  }

  // 返回指向类型 T 的指针成员访问操作符重载
  [[nodiscard]]
  constexpr
  decltype(&(*std::declval<const T&>())) operator->() const { return &operator*(); }
};

// 结构体 `arithmetic` 定义
struct arithmetic
{
  // 模板类 `modifier` 的定义
  template <typename T>
  class modifier
  {
  public:
    // 重载操作符 `-`，返回类型为 `T`，使用 `[[nodiscard]]` 属性
    friend
    constexpr
    T
    operator-(
      const T &lh)
    {
      // 返回 `lh` 的相反数
      return T{-value_of(lh)};
    }

    // 重载操作符 `+=`，修改 `lh` 的值并返回引用 `T&`
    friend
    constexpr
    T&
    operator+=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) += value_of(rh)))
    {
      // 将 `lh` 增加 `rh` 的值
      value_of(lh) += value_of(rh);
      return lh;
    }

    // 重载操作符 `-=`，修改 `lh` 的值并返回引用 `T&`
    friend
    constexpr
    T&
    operator-=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) -= value_of(rh)))
    {
      // 将 `lh` 减去 `rh` 的值
      value_of(lh) -= value_of(rh);
      return lh;
    }

    // 重载操作符 `*=`，修改 `lh` 的值并返回引用 `T&`
    friend
    constexpr
    T&
    operator*=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) *= value_of(rh)))
    {
      // 将 `lh` 乘以 `rh` 的值
      value_of(lh) *= value_of(rh);
      return lh;
    }

    // 重载操作符 `/=`，修改 `lh` 的值并返回引用 `T&`
    friend
    constexpr
    T&
    operator/=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) /= value_of(rh)))
    {
      // 将 `lh` 除以 `rh` 的值
      value_of(lh) /= value_of(rh);
      return lh;
    }

    // 重载操作符 `%=`，修改 `lh` 的值并返回引用 `T&`
    template <typename TT = T, typename = decltype(value_of(std::declval<TT>()) % value_of(std::declval<TT>()))>
    friend
    constexpr
    T&
    operator%=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) %= value_of(rh)))
    {
      // 将 `lh` 取模 `rh` 的值
      value_of(lh) %= value_of(rh);
      return lh;
    }

    // 重载操作符 `+`，返回类型为 `T`，使用 `[[nodiscard]]` 属性
    [[nodiscard]]
    friend
    constexpr
    T
    operator+(
      T lh,
      const T &rh)
    {
      // 返回 `lh` 加上 `rh` 的值
      lh += rh;
      return lh;
    }

    // 重载操作符 `-`，返回类型为 `T`，使用 `[[nodiscard]]` 属性
    [[nodiscard]]
    friend
    constexpr
    T
    operator-(
      T lh,
      const T &rh)
    {
      // 返回 `lh` 减去 `rh` 的值
      lh -= rh;
      return lh;
    }

    // 重载操作符 `*`，返回类型为 `T`，使用 `[[nodiscard]]` 属性
    [[nodiscard]]
    friend
    constexpr
    T
    operator*(
      T lh,
      const T &rh)
    {
      // 返回 `lh` 乘以 `rh` 的值
      lh *= rh;
      return lh;
    }

    // 重载操作符 `/`，返回类型为 `T`，使用 `[[nodiscard]]` 属性
    [[nodiscard]]
    friend
    constexpr
    T
    operator/(
      T lh,
      const T &rh)
    {
      // 返回 `lh` 除以 `rh` 的值
      lh /= rh;
      return lh;
    }

    // 重载操作符 `%`，返回类型为 `T`，使用 `[[nodiscard]]` 属性
    template <typename TT = T, typename = decltype(value_of(std::declval<TT>()) % value_of(std::declval<TT>()))>
    [[nodiscard]]
    friend
    constexpr
    T
    operator%(
      T lh,
      const T &rh)
    {
      // 返回 `lh` 取模 `rh` 的值
      lh %= rh;
      return lh;
    }

  };
};

// 结构体 `bitarithmetic` 定义
struct bitarithmetic
{
  // 模板类 `modifier` 的定义
  template <typename T>
  class modifier
  {
  public:
    // 重载操作符 `&=`，修改 `lh` 的值并返回引用 `T&`
    friend
    constexpr
    T&
    operator&=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) &= value_of(rh)))
    {
      // 将 `lh` 按位与 `rh` 的值
      value_of(lh) &= value_of(rh);
      return lh;
    }

    // 重载操作符 `|=`，修改 `lh` 的值并返回引用 `T&`
    friend
    constexpr
    T&
    operator|=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) |= value_of(rh)))
    {
      // 将 `lh` 按位或 `rh` 的值
      value_of(lh) |= value_of(rh);
      return lh;
    }

    // 重载操作符 `^=`，修改 `lh` 的值并返回引用 `T&`
    friend
    constexpr
    T&
    operator^=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) ^= value_of(rh)))
    {
      // 将 `lh` 按位异或 `rh` 的值
      value_of(lh) ^= value_of(rh);
      return lh;
    }

    // 重载操作符 `<<=`，修改 `lh` 的值并返回引用 `T&`
    template <typename C>
    friend
    constexpr
    T&
    operator<<=(
      T &lh,
      C c)
    noexcept(noexcept(value_of(lh) <<= c)))
    {
      // 左移操作符的重载函数，将左操作数按右操作数指定的位数左移，并返回左操作数的引用
      value_of(lh) <<= c;
      return lh;
    }

    template <typename C>
    friend
    constexpr
    T&
    operator>>=(
      T &lh,
      C c)
    noexcept(noexcept(value_of(lh) >>= c))
    {
      // 右移赋值操作符的重载函数，将左操作数按右操作数指定的位数右移，并返回左操作数的引用
      value_of(lh) >>= c;
      return lh;
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator~(
      const T &lh)
    {
      // 按位取反操作符的重载函数，返回左操作数按位取反后的结果
      auto v = value_of(lh);
      v = ~v;
      return T(v);
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator&(
      T lh,
      const T &rh)
    {
      // 按位与操作符的重载函数，返回左右操作数按位与的结果
      lh &= rh;
      return lh;
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator|(
      T lh,
      const T &rh)
    {
      // 按位或操作符的重载函数，返回左右操作数按位或的结果
      lh |= rh;
      return lh;
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator^(
      T lh,
      const T &rh)
    {
      // 按位异或操作符的重载函数，返回左右操作数按位异或的结果
      lh ^= rh;
      return lh;
    }

    template <typename C>
    [[nodiscard]]
    friend
    constexpr
    T
    operator<<(
      T lh,
      C c)
    {
      // 左移操作符的重载函数，返回左操作数按右操作数指定的位数左移后的结果
      lh <<= c;
      return lh;
    }

    template <typename C>
    [[nodiscard]]
    friend
    constexpr
    T
    operator>>(
      T lh,
      C c)
    {
      // 右移操作符的重载函数，返回左操作数按右操作数指定的位数右移后的结果
      lh >>= c;
      return lh;
    }
  };
};
template <typename I = void>
struct indexed
{
  // 声明一个模板类 modifier，用于处理特定类型 T 的索引操作
  template <typename T>
  class modifier;
};

template <>
struct indexed<void> {
  // 对于没有模板参数的 indexed 特化版本，定义 modifier 模板类
  template<typename>
  class modifier;

  // 对于具体类型 T, Tag, Ms... 的 indexed::modifier 特化版本
  template <typename T, typename Tag, typename ... Ms>
  class modifier<type<T, Tag, Ms...>> {
    // 定义几种引用类型
    using ref = T&;
    using cref = const T&;
    using rref = T&&;
    // 使用 strong 命名空间中的 type 类型
    using type = strong::type<T, Tag, Ms...>;
  public:
    // 索引操作符的 const & 版本
    template<typename I>
    [[nodiscard]]
    auto
    operator[](
      const I &i)
    const &
    // 使用 impl::access 访问索引 i，返回相应的值，不抛出异常
    noexcept(noexcept(std::declval<cref>()[impl::access(i)]))
    -> decltype(std::declval<cref>()[impl::access(i)]) {
      auto& self = static_cast<const type&>(*this);
      return value_of(self)[impl::access(i)];
    }

    // 索引操作符的 & 版本
    template<typename I>
    [[nodiscard]]
    auto
    operator[](
      const I &i)
    &
    // 使用 impl::access 访问索引 i，返回相应的值，不抛出异常
    noexcept(noexcept(std::declval<ref>()[impl::access(i)]))
    -> decltype(std::declval<ref>()[impl::access(i)]) {
      auto& self = static_cast<type&>(*this);
      return value_of(self)[impl::access(i)];
    }

    // 索引操作符的 && 版本
    template<typename I>
    [[nodiscard]]
    auto
    operator[](
      const I &i)
    &&
    // 使用 impl::access 访问索引 i，返回相应的值，不抛出异常
    noexcept(noexcept(std::declval<rref>()[impl::access(i)]))
    -> decltype(std::declval<rref>()[impl::access(i)]) {
      auto& self = static_cast<type&>(*this);
      return value_of(std::move(self))[impl::access(i)];
    }

    // at 方法的 const & 版本
    template<typename I, typename C = cref>
    [[nodiscard]]
    auto
    at(
      const I &i)
    const &
    // 使用 impl::access 访问索引 i，返回相应的值，不抛出异常
    -> decltype(std::declval<C>().at(impl::access(i))) {
      auto& self = static_cast<const type&>(*this);
      return value_of(self).at(impl::access(i));
    }

    // at 方法的 & 版本
    template<typename I, typename R = ref>
    [[nodiscard]]
    auto
    at(
      const I &i)
    &
    // 使用 impl::access 访问索引 i，返回相应的值，不抛出异常
    -> decltype(std::declval<R>().at(impl::access(i))) {
      auto& self = static_cast<type&>(*this);
      return value_of(self).at(impl::access(i));
    }

    // at 方法的 && 版本
    template<typename I, typename R = rref>
    [[nodiscard]]
    auto
    at(
      const I &i)
    &&
    // 使用 impl::access 访问索引 i，返回相应的值，不抛出异常
    -> decltype(std::declval<R>().at(impl::access(i))) {
      auto& self = static_cast<type&>(*this);
      return value_of(std::move(self)).at(impl::access(i));
    }
  };
};

template <typename I>
template <typename T, typename Tag, typename ... M>
class indexed<I>::modifier<type<T, Tag, M...>>
{
  // 使用 strong 命名空间中的 type 类型
  using type = ::strong::type<T, Tag, M...>;
public:
  // 索引操作符的 const & 版本
  [[nodiscard]]
  auto
  operator[](
    const I& i)
  const &
  // 使用 impl::access 访问索引 i，返回相应的值，不抛出异常
  noexcept(noexcept(std::declval<const T&>()[impl::access(i)]))
  -> decltype(std::declval<const T&>()[impl::access(i)])
  {
    auto& self = static_cast<const type&>(*this);
    return value_of(self)[impl::access(i)];
  }

  // 索引操作符的 & 版本
  [[nodiscard]]
  auto
  operator[](
    const I& i)
  &
  // 使用 impl::access 访问索引 i，返回相应的值，不抛出异常
  noexcept(noexcept(std::declval<T&>()[impl::access(i)]))
  -> decltype(std::declval<T&>()[impl::access(i)])
  {
    auto& self = static_cast<type&>(*this);
    return value_of(self)[impl::access(i)];
  }

  // 索引操作符的 && 版本
  [[nodiscard]]
  auto
  operator[](
    const I& i)
  &&
  // 使用 impl::access 访问索引 i，返回相应的值，不抛出异常
  -> decltype(std::declval<rref>()[impl::access(i)]) {
    auto& self = static_cast<type&>(*this);
    return value_of(std::move(self))[impl::access(i)];
  }
};
    // 定义成员函数 operator[]，接受常量引用参数 i，并保证不抛出异常
    // 返回值类型为 decltype(std::declval<T&&>()[impl::access(i)])
    template <typename I>
    auto operator[](
      const I& i)
      &&
      noexcept(noexcept(std::declval<T&&>()[impl::access(i)]))
      -> decltype(std::declval<T&&>()[impl::access(i)])
    {
      // 将 this 转换为 type& 类型的引用，并赋值给 self
      auto& self = static_cast<type&>(*this);
      // 调用 value_of(std::move(self))[impl::access(i)] 并返回结果
      return value_of(std::move(self))[impl::access(i)];
    }
    
    // 定义成员函数 at，接受常量引用参数 i，且是 const & 成员函数
    // 返回值类型为 decltype(std::declval<const TT&>().at(impl::access(i)))
    template <typename TT = T>
    [[nodiscard]]
    auto
    at(
      const I& i)
    const &
    -> decltype(std::declval<const TT&>().at(impl::access(i)))
    {
      // 将 this 转换为 const type& 类型的引用，并赋值给 self
      auto& self = static_cast<const type&>(*this);
      // 调用 value_of(self).at(impl::access(i)) 并返回结果
      return value_of(self).at(impl::access(i));
    }
    
    // 定义成员函数 at，接受常量引用参数 i，且是 & 成员函数
    // 返回值类型为 decltype(std::declval<TT&>().at(impl::access(i)))
    template <typename TT = T>
    [[nodiscard]]
    auto
    at(
      const I& i)
    &
    -> decltype(std::declval<TT&>().at(impl::access(i)))
    {
      // 将 this 转换为 type& 类型的引用，并赋值给 self
      auto& self = static_cast<type&>(*this);
      // 调用 value_of(self).at(impl::access(i)) 并返回结果
      return value_of(self).at(impl::access(i));
    }
    
    // 定义成员函数 at，接受常量引用参数 i，且是 && 成员函数
    // 返回值类型为 decltype(std::declval<TT&&>().at(impl::access(i)))
    template <typename TT = T>
    [[nodiscard]]
    auto
    at(
      const I& i)
    &&
    -> decltype(std::declval<TT&&>().at(impl::access(i)))
    {
      // 将 this 转换为 type& 类型的引用，并赋值给 self
      auto& self = static_cast<type&>(*this);
      // 调用 value_of(std::move(self)).at(impl::access(i)) 并返回结果
      return value_of(std::move(self)).at(impl::access(i));
    }
};

// 定义一个名为 iterator 的类
class iterator
{
public:
  // 定义模板类 modifier，参数 I 是迭代器类型，category 是迭代器类型的标签
  template <typename I, typename category = typename std::iterator_traits<underlying_type_t<I>>::iterator_category>
  class modifier
    : public pointer::modifier<I> // 继承自 pointer::modifier<I>
    , public equality::modifier<I> // 继承自 equality::modifier<I>
    , public incrementable::modifier<I> // 继承自 incrementable::modifier<I>
  {
  public:
    // 定义迭代器特性类型
    using difference_type = typename std::iterator_traits<underlying_type_t<I>>::difference_type;
    using value_type = typename std::iterator_traits<underlying_type_t<I>>::value_type;
    using pointer = typename std::iterator_traits<underlying_type_t<I>>::value_type;
    using reference = typename std::iterator_traits<underlying_type_t<I>>::reference;
    using iterator_category = typename std::iterator_traits<underlying_type_t<I>>::iterator_category;
  };

  // 对于双向迭代器标签的特化
  template <typename I>
  class modifier<I, std::bidirectional_iterator_tag>
    : public modifier<I, std::forward_iterator_tag> // 继承自前向迭代器的 modifier
      , public decrementable::modifier<I> // 继承自 decrementable::modifier<I>
  {
  };

  // 对于随机访问迭代器标签的特化
  template <typename I>
  class modifier<I, std::random_access_iterator_tag>
    : public modifier<I, std::bidirectional_iterator_tag> // 继承自双向迭代器的 modifier
      , public affine_point<typename std::iterator_traits<underlying_type_t<I>>::difference_type>::template modifier<I> // 继承自 affine_point
      , public indexed<>::modifier<I> // 继承自 indexed::modifier<I>
      , public ordered::modifier<I> // 继承自 ordered::modifier<I>
  {
  };
};

// 定义一个名为 range 的类
class range
{
public:
  // 定义模板类 modifier，参数 R 是范围类型
  template <typename R>
  class modifier;
};

// 对于 strong::type<T, Tag, M...> 类型的范围 modifier 特化
template <typename T, typename Tag, typename ... M>
class range::modifier<type<T, Tag, M...>>
{
  using type = ::strong::type<T, Tag, M...>; // 定义 type 别名为 strong::type<T, Tag, M...>
  using r_iterator = decltype(std::declval<T&>().begin()); // 定义范围迭代器类型
  using r_const_iterator = decltype(std::declval<const T&>().begin()); // 定义常量范围迭代器类型
public:
  using iterator = ::strong::type<r_iterator, Tag, strong::iterator>; // 定义迭代器类型为 strong::type<r_iterator, Tag, strong::iterator>
  using const_iterator = ::strong::type<r_const_iterator, Tag, strong::iterator>; // 定义常量迭代器类型为 strong::type<r_const_iterator, Tag, strong::iterator>

  // 返回范围的起始迭代器
  iterator
  begin()
  noexcept(noexcept(std::declval<T&>().begin()))
  {
    auto& self = static_cast<type&>(*this);
    return iterator{value_of(self).begin()};
  }

  // 返回范围的结束迭代器
  iterator
  end()
  noexcept(noexcept(std::declval<T&>().end()))
  {
    auto& self = static_cast<type&>(*this);
    return iterator{value_of(self).end()};
  }

  // 返回常量范围的起始迭代器
  const_iterator
  cbegin()
    const
  noexcept(noexcept(std::declval<const T&>().begin()))
  {
    auto& self = static_cast<const type&>(*this);
    return const_iterator{value_of(self).begin()};
  }

  // 返回常量范围的结束迭代器
  const_iterator
  cend()
    const
  noexcept(noexcept(std::declval<const T&>().end()))
  {
    auto& self = static_cast<const type&>(*this);
    return const_iterator{value_of(self).end()};
  }

  // 返回常量范围的起始迭代器
  const_iterator
  begin()
  const
  noexcept(noexcept(std::declval<const T&>().begin()))
  {
    auto& self = static_cast<const type&>(*this);
    return const_iterator{value_of(self).begin()};
  }

  // 返回常量范围的结束迭代器
  const_iterator
  end()
  const
  noexcept(noexcept(std::declval<const T&>().end()))
  {
    auto& self = static_cast<const type&>(*this);
    return const_iterator{value_of(self).end()};
  }
};

// 命名空间 impl 的结构体 converter，参数 T 是类型，D 是转换器
namespace impl {

  template<typename T, typename D>
  struct converter
  {
    # 定义一个 constexpr 成员函数，将当前类 T 转换为目标类型 D
    constexpr explicit operator D() const
    # 指定 noexcept 修饰符，表明转换操作不会抛出异常，依赖于底层类型 T 的转换操作的 noexcept 特性
    noexcept(noexcept(static_cast<D>(std::declval<const underlying_type_t<T>&>())))
    {
      # 将当前对象 static_cast 为 T 类型的常量引用
      auto& self = static_cast<const T&>(*this);
      # 调用 value_of 函数获取 self 的值，然后将其转换为类型 D 并返回
      return static_cast<D>(value_of(self));
    }
  };
  template<typename T, typename D>
  struct implicit_converter
  {
    # 定义一个 constexpr 类型转换操作符，将当前类 T 转换为目标类型 D
    constexpr operator D() const
    # 指定 noexcept 修饰符，表明转换操作不会抛出异常，依赖于底层类型 T 的转换操作的 noexcept 特性
    noexcept(noexcept(static_cast<D>(std::declval<const underlying_type_t<T>&>())))
    {
      # 将当前对象 static_cast 为 T 类型的常量引用
      auto& self = static_cast<const T&>(*this);
      # 调用 value_of 函数获取 self 的值，然后将其转换为类型 D 并返回
      return static_cast<D>(value_of(self));
    }
  };
}
template <typename ... Ts>
struct convertible_to
{
  // 可转换为结构体模板，定义了一个修改器模板，继承自多个转换器模板
  template <typename T>
  struct modifier : impl::converter<T, Ts>...
  {
  };
};

template <typename ... Ts>
struct implicitly_convertible_to
{
  // 隐式可转换为结构体模板，定义了一个修改器模板，继承自多个隐式转换器模板
  template <typename T>
  struct modifier : impl::implicit_converter<T, Ts>...
  {
  };

};

struct formattable
{
    // 可格式化结构体，定义了一个修改器模板
    template <typename T>
    class modifier{};
};

}

namespace std {
template <typename T, typename Tag, typename ... M>
struct hash<::strong::type<T, Tag, M...>>
  : std::conditional_t<
    std::is_base_of<
      ::strong::hashable::modifier<
        ::strong::type<T, Tag, M...>
      >,
      ::strong::type<T, Tag, M...>
    >::value,
    // 如果类型可哈希，使用标准库中的 hash<T> 进行哈希
    hash<T>,
    // 否则使用 std::false_type
    std::false_type>
{
  using type = ::strong::type<T, Tag, M...>;
  decltype(auto)
  operator()(
    const ::strong::hashable::modifier<type>& t)
  const
  // 声明不抛出异常的哈希运算符重载
  noexcept(noexcept(std::declval<hash<T>>()(value_of(std::declval<const type&>()))))
  {
    auto& tt = static_cast<const type&>(t);
    // 调用 hash<T> 的运算符重载计算哈希值
    return hash<T>::operator()(value_of(tt));
  }
};
template <typename T, typename Tag, typename ... M>
struct is_arithmetic<::strong::type<T, Tag, M...>>
  : is_base_of<::strong::arithmetic::modifier<::strong::type<T, Tag, M...>>,
               ::strong::type<T, Tag, M...>>
{
};

#if STRONG_HAS_STD_FORMAT
template<typename T, typename Tag, typename... M, typename Char>
struct formatter<::strong::type<T, Tag, M...>, Char,
                 std::enable_if_t<
                     std::is_base_of<
                         ::strong::formattable::modifier<
                             ::strong::type<T, Tag, M...>
                             >,
                         ::strong::type<T, Tag, M...>
                         >::value
                     >>
    : formatter<T>
{
  using type = ::strong::type<T, Tag, M...>;
  template<typename FormatContext>
  constexpr
  decltype(auto)
  format(const ::strong::formattable::modifier<type>& t, FormatContext& fc)
      // 声明不抛出异常的格式化函数
      noexcept(noexcept(std::declval<formatter<T, Char>>().format(value_of(std::declval<const type&>()), fc)))
  {
    const auto& tt = static_cast<const type&>(t);
    // 调用 formatter<T, Char> 的格式化函数
    return formatter<T, Char>::format(value_of(tt), fc);
  }
};
#endif

}

#if STRONG_HAS_FMT_FORMAT
namespace fmt
{
template<typename T, typename Tag, typename... M, typename Char>
struct formatter<::strong::type<T, Tag, M...>, Char,
                 std::enable_if_t<
                   std::is_base_of<
                     ::strong::formattable::modifier<
                       ::strong::type<T, Tag, M...>
                     >,
                     ::strong::type<T, Tag, M...>
                   >::value
                 >>
  : formatter<T>
{
  using type = ::strong::type<T, Tag, M...>;
  template<typename FormatContext>
  constexpr
  decltype(auto)
  format(const ::strong::formattable::modifier<type>& t, FormatContext& fc)
      // 声明不抛出异常的格式化函数
      noexcept(noexcept(std::declval<formatter<T, Char>>().format(value_of(std::declval<const type&>()), fc)))
  {
    const auto& tt = static_cast<const type&>(t);
    // 使用模板函数 formatter<T, Char>::format 对给定的 value_of(tt) 进行格式化处理，并返回结果
    return formatter<T, Char>::format(value_of(tt), fc);
};
}
#endif
#endif //ROLLBEAR_STRONG_TYPE_HPP_INCLUDED


// 结束命名空间的闭合和条件编译指令
};
}
// 结束第一个条件编译指令
#endif
// 结束第二个条件编译指令，并注释标识符ROLLBEAR_STRONG_TYPE_HPP_INCLUDED
#endif //ROLLBEAR_STRONG_TYPE_HPP_INCLUDED
```