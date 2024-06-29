# `.\numpy\numpy\_core\src\common\numpy_tag.h`

```py
#ifndef _NPY_COMMON_TAG_H_
#define _NPY_COMMON_TAG_H_

#include "../npysort/npysort_common.h"

namespace npy {

// 定义一个模板结构体，用于存储多个标签类型，tags 是模板参数包
template<typename... tags>
struct taglist {
  // 计算标签数量的静态成员变量
  static constexpr unsigned size = sizeof...(tags);
};

// 各种标签类型的结构体定义

// 整数标签
struct integral_tag {
};

// 浮点数标签
struct floating_point_tag {
};

// 复数标签
struct complex_tag {
};

// 日期标签
struct date_tag {
};

// 布尔类型标签，继承自整数标签
struct bool_tag : integral_tag {
    // 使用 npy_bool 作为类型别名
    using type = npy_bool;
    // 标识这是布尔类型的 NPY_TYPES 值
    static constexpr NPY_TYPES type_value = NPY_BOOL;
    // 比较操作：小于
    static int less(type const& a, type const& b) {
      return BOOL_LT(a, b);
    }
    // 比较操作：小于等于
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

// 字节标签，继承自整数标签
struct byte_tag : integral_tag {
    // 使用 npy_byte 作为类型别名
    using type = npy_byte;
    // 标识这是字节类型的 NPY_TYPES 值
    static constexpr NPY_TYPES type_value = NPY_BYTE;
    // 比较操作：小于
    static int less(type const& a, type const& b) {
      return BYTE_LT(a, b);
    }
    // 比较操作：小于等于
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

// 无符号字节标签，继承自整数标签
struct ubyte_tag : integral_tag {
    // 使用 npy_ubyte 作为类型别名
    using type = npy_ubyte;
    // 标识这是无符号字节类型的 NPY_TYPES 值
    static constexpr NPY_TYPES type_value = NPY_UBYTE;
    // 比较操作：小于
    static int less(type const& a, type const& b) {
      return UBYTE_LT(a, b);
    }
    // 比较操作：小于等于
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

// 短整数标签，继承自整数标签
struct short_tag : integral_tag {
    // 使用 npy_short 作为类型别名
    using type = npy_short;
    // 标识这是短整数类型的 NPY_TYPES 值
    static constexpr NPY_TYPES type_value = NPY_SHORT;
    // 比较操作：小于
    static int less(type const& a, type const& b) {
      return SHORT_LT(a, b);
    }
    // 比较操作：小于等于
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

// 无符号短整数标签，继承自整数标签
struct ushort_tag : integral_tag {
    // 使用 npy_ushort 作为类型别名
    using type = npy_ushort;
    // 标识这是无符号短整数类型的 NPY_TYPES 值
    static constexpr NPY_TYPES type_value = NPY_USHORT;
    // 比较操作：小于
    static int less(type const& a, type const& b) {
      return USHORT_LT(a, b);
    }
    // 比较操作：小于等于
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

// 整数标签，继承自整数标签
struct int_tag : integral_tag {
    // 使用 npy_int 作为类型别名
    using type = npy_int;
    // 标识这是整数类型的 NPY_TYPES 值
    static constexpr NPY_TYPES type_value = NPY_INT;
    // 比较操作：小于
    static int less(type const& a, type const& b) {
      return INT_LT(a, b);
    }
    // 比较操作：小于等于
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

// 无符号整数标签，继承自整数标签
struct uint_tag : integral_tag {
    // 使用 npy_uint 作为类型别名
    using type = npy_uint;
    // 标识这是无符号整数类型的 NPY_TYPES 值
    static constexpr NPY_TYPES type_value = NPY_UINT;
    // 比较操作：小于
    static int less(type const& a, type const& b) {
      return UINT_LT(a, b);
    }
    // 比较操作：小于等于
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

// 长整数标签，继承自整数标签
struct long_tag : integral_tag {
    // 使用 npy_long 作为类型别名
    using type = npy_long;
    // 标识这是长整数类型的 NPY_TYPES 值
    static constexpr NPY_TYPES type_value = NPY_LONG;
    // 比较操作：小于
    static int less(type const& a, type const& b) {
      return LONG_LT(a, b);
    }
    // 比较操作：小于等于
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

// 无符号长整数标签，继承自整数标签
struct ulong_tag : integral_tag {
    // 使用 npy_ulong 作为类型别名
    using type = npy_ulong;
    // 标识这是无符号长整数类型的 NPY_TYPES 值
    static constexpr NPY_TYPES type_value = NPY_ULONG;
    // 比较操作：小于
    static int less(type const& a, type const& b) {
      return ULONG_LT(a, b);
    }
    // 比较操作：小于等于
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

#endif // _NPY_COMMON_TAG_H_
struct longlong_tag : integral_tag {
    // 定义类型为 npy_longlong 的别名 type
    using type = npy_longlong;
    // 设置 type_value 为 NPY_LONGLONG
    static constexpr NPY_TYPES type_value = NPY_LONGLONG;
    // 比较函数，比较 a 是否小于 b
    static int less(type const& a, type const& b) {
      return LONGLONG_LT(a, b);
    }
    // 小于等于函数，利用 less 函数实现
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

struct ulonglong_tag : integral_tag {
    // 定义类型为 npy_ulonglong 的别名 type
    using type = npy_ulonglong;
    // 设置 type_value 为 NPY_ULONGLONG
    static constexpr NPY_TYPES type_value = NPY_ULONGLONG;
    // 比较函数，比较 a 是否小于 b
    static int less(type const& a, type const& b) {
      return ULONGLONG_LT(a, b);
    }
    // 小于等于函数，利用 less 函数实现
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

struct half_tag {
    // 定义类型为 npy_half 的别名 type
    using type = npy_half;
    // 设置 type_value 为 NPY_HALF
    static constexpr NPY_TYPES type_value = NPY_HALF;
    // 比较函数，比较 a 是否小于 b
    static int less(type const& a, type const& b) {
      return HALF_LT(a, b);
    }
    // 小于等于函数，利用 less 函数实现
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

struct float_tag : floating_point_tag {
    // 定义类型为 npy_float 的别名 type
    using type = npy_float;
    // 设置 type_value 为 NPY_FLOAT
    static constexpr NPY_TYPES type_value = NPY_FLOAT;
    // 比较函数，比较 a 是否小于 b
    static int less(type const& a, type const& b) {
      return FLOAT_LT(a, b);
    }
    // 小于等于函数，利用 less 函数实现
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

struct double_tag : floating_point_tag {
    // 定义类型为 npy_double 的别名 type
    using type = npy_double;
    // 设置 type_value 为 NPY_DOUBLE
    static constexpr NPY_TYPES type_value = NPY_DOUBLE;
    // 比较函数，比较 a 是否小于 b
    static int less(type const& a, type const& b) {
      return DOUBLE_LT(a, b);
    }
    // 小于等于函数，利用 less 函数实现
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

struct longdouble_tag : floating_point_tag {
    // 定义类型为 npy_longdouble 的别名 type
    using type = npy_longdouble;
    // 设置 type_value 为 NPY_LONGDOUBLE
    static constexpr NPY_TYPES type_value = NPY_LONGDOUBLE;
    // 比较函数，比较 a 是否小于 b
    static int less(type const& a, type const& b) {
      return LONGDOUBLE_LT(a, b);
    }
    // 小于等于函数，利用 less 函数实现
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

struct cfloat_tag : complex_tag {
    // 定义类型为 npy_cfloat 的别名 type
    using type = npy_cfloat;
    // 设置 type_value 为 NPY_CFLOAT
    static constexpr NPY_TYPES type_value = NPY_CFLOAT;
    // 比较函数，比较 a 是否小于 b
    static int less(type const& a, type const& b) {
      return CFLOAT_LT(a, b);
    }
    // 小于等于函数，利用 less 函数实现
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

struct cdouble_tag : complex_tag {
    // 定义类型为 npy_cdouble 的别名 type
    using type = npy_cdouble;
    // 设置 type_value 为 NPY_CDOUBLE
    static constexpr NPY_TYPES type_value = NPY_CDOUBLE;
    // 比较函数，比较 a 是否小于 b
    static int less(type const& a, type const& b) {
      return CDOUBLE_LT(a, b);
    }
    // 小于等于函数，利用 less 函数实现
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

struct clongdouble_tag : complex_tag {
    // 定义类型为 npy_clongdouble 的别名 type
    using type = npy_clongdouble;
    // 设置 type_value 为 NPY_CLONGDOUBLE
    static constexpr NPY_TYPES type_value = NPY_CLONGDOUBLE;
    // 比较函数，比较 a 是否小于 b
    static int less(type const& a, type const& b) {
      return CLONGDOUBLE_LT(a, b);
    }
    // 小于等于函数，利用 less 函数实现
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

struct datetime_tag : date_tag {
    // 定义类型为 npy_datetime 的别名 type
    using type = npy_datetime;
    // 设置 type_value 为 NPY_DATETIME
    static constexpr NPY_TYPES type_value = NPY_DATETIME;
    // 比较函数，比较 a 是否小于 b
    static int less(type const& a, type const& b) {
      return DATETIME_LT(a, b);
    }
    // 定义静态函数 `less_equal`，用于比较两个常量引用类型 `a` 和 `b`，检查是否 `a` 不小于等于 `b`
    static int less_equal(type const& a, type const& b) {
      // 返回 `b` 不小于 `a` 的结果取反的布尔值（即 `a` 是否不小于等于 `b`）
      return !less(b, a);
    }
};
// 结构体定义：timedelta_tag，继承自date_tag
struct timedelta_tag : date_tag {
    // 使用npy_timedelta作为类型
    using type = npy_timedelta;
    // 定义类型值为NPY_TIMEDELTA
    static constexpr NPY_TYPES type_value = NPY_TIMEDELTA;
    // 比较函数：比较两个类型为type的对象a和b是否满足a < b
    static int less(type const& a, type const& b) {
      return TIMEDELTA_LT(a, b);
    }
    // 比较函数：比较两个类型为type的对象a和b是否满足a <= b
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

// 结构体定义：string_tag
struct string_tag {
    // 使用npy_char作为类型
    using type = npy_char;
    // 定义类型值为NPY_STRING
    static constexpr NPY_TYPES type_value = NPY_STRING;
    // 比较函数：比较两个类型为type*的对象a和b（长度为len）是否满足a < b
    static int less(type const* a, type const* b, size_t len) {
      return STRING_LT(a, b, len);
    }
    // 比较函数：比较两个类型为type*的对象a和b（长度为len）是否满足a <= b
    static int less_equal(type const* a, type const* b, size_t len) {
      return !less(b, a, len);
    }
    // 交换函数：交换两个类型为type*的对象a和b（长度为len）
    static void swap(type* a, type* b, size_t len) {
      STRING_SWAP(a, b, len);
    }
    // 复制函数：将类型为type*的对象b（长度为len）复制到a
    static void copy(type * a, type const* b, size_t len) {
      STRING_COPY(a, b, len);
    }
};

// 结构体定义：unicode_tag
struct unicode_tag {
    // 使用npy_ucs4作为类型
    using type = npy_ucs4;
    // 定义类型值为NPY_UNICODE
    static constexpr NPY_TYPES type_value = NPY_UNICODE;
    // 比较函数：比较两个类型为type*的对象a和b（长度为len）是否满足a < b
    static int less(type const* a, type const* b, size_t len) {
      return UNICODE_LT(a, b, len);
    }
    // 比较函数：比较两个类型为type*的对象a和b（长度为len）是否满足a <= b
    static int less_equal(type const* a, type const* b, size_t len) {
      return !less(b, a, len);
    }
    // 交换函数：交换两个类型为type*的对象a和b（长度为len）
    static void swap(type* a, type* b, size_t len) {
      UNICODE_SWAP(a, b, len);
    }
    // 复制函数：将类型为type*的对象b（长度为len）复制到a
    static void copy(type * a, type const* b, size_t len) {
      UNICODE_COPY(a, b, len);
    }
};

}  // namespace npy

#endif
```