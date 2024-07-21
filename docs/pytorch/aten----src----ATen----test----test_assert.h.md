# `.\pytorch\aten\src\ATen\test\test_assert.h`

```
#pragma once
// 如果尚未定义 __func__ 宏，并且编译器版本小于等于 1900，则将 __func__ 定义为 __FUNCTION__
#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

// 如果编译器为 GCC、Intel C++ 或者 Clang，则定义 AT_EXPECT 宏为 __builtin_expect((x),(y))，否则为 x
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define AT_EXPECT(x, y) (__builtin_expect((x),(y)))
#else
#define AT_EXPECT(x, y) (x)
#endif

// ASSERT 宏，用于检查条件是否成立，否则抛出异常
#define ASSERT(cond) \
  if (AT_EXPECT(!(cond), 0)) { \
    barf("%s:%u: %s: Assertion `%s` failed.", __FILE__, __LINE__, __func__, #cond); \
  }

// TRY_CATCH_ELSE 宏，用于封装执行 fn 并捕获异常，然后执行 catc 或 els
#define TRY_CATCH_ELSE(fn, catc, els)                           \
  {                                                             \
    /* 避免在 els 代码块抛出异常时错误地认为通过了 */           \
    bool _passed = false;                                       \
    try {                                                       \
      fn;                                                       \
      _passed = true;                                           \
      els;                                                      \
    } catch (const std::exception &e) {                         \
      ASSERT(!_passed);                                         \
      catc;                                                     \
    }                                                           \
  }

// ASSERT_THROWSM 宏，用于验证 fn 是否抛出特定的异常消息 message
#define ASSERT_THROWSM(fn, message)     \
  TRY_CATCH_ELSE(fn, ASSERT(std::string(e.what()).find(message) != std::string::npos), ASSERT(false))

// ASSERT_THROWS 宏，用于验证 fn 是否抛出任何异常
#define ASSERT_THROWS(fn)  \
  ASSERT_THROWSM(fn, "");

// ASSERT_EQUAL 宏，用于验证 t1 和 t2 是否相等
#define ASSERT_EQUAL(t1, t2) \
  ASSERT(t1.equal(t2));

// ASSERT_ALLCLOSE 宏，用于验证 t1 和 t2 是否在给定的 tolerance 下近似相等
#define ASSERT_ALLCLOSE(t1, t2)   \
  ASSERT(t1.is_same_size(t2));    \
  ASSERT(t1.allclose(t2));

// ASSERT_ALLCLOSE_TOLERANCES 宏，用于验证 t1 和 t2 是否在给定的 tolerance (atol, rtol) 下近似相等
#define ASSERT_ALLCLOSE_TOLERANCES(t1, t2, atol, rtol)   \
  ASSERT(t1.is_same_size(t2));    \
  ASSERT(t1.allclose(t2, atol, rtol));
```