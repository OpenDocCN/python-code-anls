# `.\pytorch\c10\core\Scalar.h`

```py
#pragma once
  // 将传入的值 vv 转换为 int64_t 类型，并存储在 v.i 中
  v.i = convert<int64_t, bool>(vv);
}

template <
    typename T,
    // 当模板参数 T 为 c10::SymBool 类型时，启用此构造函数
    typename std::enable_if_t<std::is_same_v<T, c10::SymBool>, bool>* =
        nullptr>
// 初始化 Scalar 对象，设置 tag 为 Tag::HAS_sb
Scalar(T vv) : tag(Tag::HAS_sb) {
  // 将传入的值 vv 转换为 int64_t 类型，并存储在 v.i 中
  v.i = convert<int64_t, c10::SymBool>(vv);
}
// 定义宏 DEFINE_ACCESSOR，用于生成类型访问器函数
#define DEFINE_ACCESSOR(type, name)                                   \
  // 定义返回 type 类型的成员函数 to##name()
  type to##name() const {                                             \
    // 根据当前对象的 tag 属性判断类型
    if (Tag::HAS_d == tag) {                                          \
      // 如果是 double 类型，则使用 checked_convert 转换为 type 类型
      return checked_convert<type, double>(v.d, #type);               \
    } else if (Tag::HAS_z == tag) {                                   \
      // 如果是复数类型，则使用 checked_convert 转换为 type 类型
      return checked_convert<type, c10::complex<double>>(v.z, #type); \
    }                                                                 \
    // 根据不同的 tag 属性，转换为对应的 type 类型
    if (Tag::HAS_b == tag) {                                          \
      return checked_convert<type, bool>(v.i, #type);                 \
    } else if (Tag::HAS_i == tag) {                                   \
      return checked_convert<type, int64_t>(v.i, #type);              \
    } else if (Tag::HAS_u == tag) {                                   \
      return checked_convert<type, uint64_t>(v.u, #type);             \
    } else if (Tag::HAS_si == tag) {                                  \
      // 如果是符号整数类型，则使用 toSymInt() 转换后再进行类型转换
      return checked_convert<type, int64_t>(                          \
          toSymInt().guard_int(__FILE__, __LINE__), #type);           \
    } else if (Tag::HAS_sd == tag) {                                  \
      // 如果是符号浮点类型，则使用 toSymFloat() 转换后再进行类型转换
      return checked_convert<type, int64_t>(                          \
          toSymFloat().guard_float(__FILE__, __LINE__), #type);       \
    } else if (Tag::HAS_sb == tag) {                                  \
      // 如果是符号布尔类型，则使用 toSymBool() 转换后再进行类型转换
      return checked_convert<type, int64_t>(                          \
          toSymBool().guard_bool(__FILE__, __LINE__), #type);         \
    }                                                                 \
    // 如果 tag 属性未知，则抛出错误
    TORCH_CHECK(false)                                                \
  }

// TODO: 支持 ComplexHalf 访问器
// 为所有标量类型生成访问器函数
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ACCESSOR)
// 生成 UInt16 的访问器函数
DEFINE_ACCESSOR(uint16_t, UInt16)
// 生成 UInt32 的访问器函数
DEFINE_ACCESSOR(uint32_t, UInt32)
// 生成 UInt64 的访问器函数
DEFINE_ACCESSOR(uint64_t, UInt64)

// 取消宏 DEFINE_ACCESSOR 的定义
#undef DEFINE_ACCESSOR

// 定义成员函数 toSymInt()，根据 tag 属性返回符号整数类型
SymInt toSymInt() const {
  if (Tag::HAS_si == tag) {
    // 如果 tag 属性为符号整数类型，则返回 SymInt 对象
    return c10::SymInt(intrusive_ptr<SymNodeImpl>::reclaim_copy(
        static_cast<SymNodeImpl*>(v.p)));
  } else {
    // 否则返回基本类型 long
    return toLong();
  }
}

// 定义成员函数 toSymFloat()，根据 tag 属性返回符号浮点类型
SymFloat toSymFloat() const {
  if (Tag::HAS_sd == tag) {
    // 如果 tag 属性为符号浮点类型，则返回 SymFloat 对象
    return c10::SymFloat(intrusive_ptr<SymNodeImpl>::reclaim_copy(
        static_cast<SymNodeImpl*>(v.p)));
  } else {
    // 否则返回基本类型 double
    return toDouble();
  }
}

// 定义成员函数 toSymBool()，根据 tag 属性返回符号布尔类型
SymBool toSymBool() const {
  if (Tag::HAS_sb == tag) {
    // 如果 tag 属性为符号布尔类型，则返回 SymBool 对象
    return c10::SymBool(intrusive_ptr<SymNodeImpl>::reclaim_copy(
        static_cast<SymNodeImpl*>(v.p)));
  } else {
    // 否则返回基本类型 bool
    return toBool();
  }
}

// 专门化的模板函数 to<T>()，禁止调用以不支持的类型
template <typename T>
T to() const = delete;

// 返回数据指针的常量版本，确保不是符号类型
const void* data_ptr() const {
  TORCH_INTERNAL_ASSERT(!isSymbolic());
  // 返回当前对象的地址
  return static_cast<const void*>(&v);
}

// 判断是否为浮点数类型
bool isFloatingPoint() const {
  // 检查标签是否表示具有 'd' 或 'sd' 的标量类型
  return Tag::HAS_d == tag || Tag::HAS_sd == tag;
}

C10_DEPRECATED_MESSAGE(
    "isIntegral is deprecated. Please use the overload with 'includeBool' parameter instead.")
bool isIntegral() const {
  // 检查标签是否表示整数类型，不包括布尔值
  return Tag::HAS_i == tag || Tag::HAS_si == tag || Tag::HAS_u == tag;
}
bool isIntegral(bool includeBool) const {
  // 检查标签是否表示整数类型，包括布尔值（如果 includeBool 为 true）
  return Tag::HAS_i == tag || Tag::HAS_si == tag || Tag::HAS_u == tag ||
      (includeBool && isBoolean());
}

bool isComplex() const {
  // 检查标签是否表示复数类型
  return Tag::HAS_z == tag;
}
bool isBoolean() const {
  // 检查标签是否表示布尔类型
  return Tag::HAS_b == tag || Tag::HAS_sb == tag;
}

// you probably don't actually want these; they're mostly for testing
bool isSymInt() const {
  // 检查标签是否表示符号整数类型
  return Tag::HAS_si == tag;
}
bool isSymFloat() const {
  // 检查标签是否表示符号浮点数类型
  return Tag::HAS_sd == tag;
}
bool isSymBool() const {
  // 检查标签是否表示符号布尔类型
  return Tag::HAS_sb == tag;
}

bool isSymbolic() const {
  // 检查标签是否表示符号类型（包括符号整数、符号浮点数和符号布尔）
  return Tag::HAS_si == tag || Tag::HAS_sd == tag || Tag::HAS_sb == tag;
}

C10_ALWAYS_INLINE Scalar& operator=(Scalar&& other) noexcept {
  if (&other == this) {
    return *this;
  }

  // 销毁当前对象，并从移动构造的对象中获取内容
  destroy();
  moveFrom(std::move(other));
  return *this;
}

C10_ALWAYS_INLINE Scalar& operator=(const Scalar& other) {
  if (&other == this) {
    return *this;
  }

  // 使用移动语义赋值给当前对象
  *this = Scalar(other);
  return *this;
}

Scalar operator-() const;
Scalar conj() const;
Scalar log() const;

template <
    typename T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
bool equal(T num) const {
  if (isComplex()) {
    TORCH_INTERNAL_ASSERT(!isSymbolic());
    auto val = v.z;
    // 对比复数的实部是否等于给定数值，虚部是否为零
    return (val.real() == num) && (val.imag() == T());
  } else if (isFloatingPoint()) {
    TORCH_CHECK(!isSymbolic(), "NYI SymFloat equality");
    // 对比浮点数是否等于给定数值
    return v.d == num;
  } else if (tag == Tag::HAS_i) {
    if (overflows<T>(v.i, /* strict_unsigned */ true)) {
      return false;
    } else {
      // 对比整数是否等于给定数值
      return static_cast<T>(v.i) == num;
    }
  } else if (tag == Tag::HAS_u) {
    if (overflows<T>(v.u, /* strict_unsigned */ true)) {
      return false;
    } else {
      // 对比无符号整数是否等于给定数值
      return static_cast<T>(v.u) == num;
    }
  } else if (tag == Tag::HAS_si) {
    TORCH_INTERNAL_ASSERT(false, "NYI SymInt equality");
  } else if (isBoolean()) {
    // 布尔标量不等于非布尔数值
    TORCH_INTERNAL_ASSERT(!isSymbolic());
    return false;
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }
}

template <
    typename T,
    typename std::enable_if_t<c10::is_complex<T>::value, int> = 0>
bool equal(T num) const {
  if (isComplex()) {
    TORCH_INTERNAL_ASSERT(!isSymbolic());
    // 对比复数是否等于给定复数
    return v.z == num;
  } else if (isFloatingPoint()) {
    TORCH_CHECK(!isSymbolic(), "NYI SymFloat equality");
    // 对比浮点数是否等于给定复数的实部，虚部是否为零
    return (v.d == num.real()) && (num.imag() == T());
  }
}
    } else if (tag == Tag::HAS_i) {
      // 如果标签为 HAS_i，检查是否整数溢出（严格无符号模式）
      if (overflows<T>(v.i, /* strict_unsigned */ true)) {
        return false;
      } else {
        // 将 v.i 转换为 T 类型，并且虚部为零时返回真
        return static_cast<T>(v.i) == num.real() && num.imag() == T();
      }
    } else if (tag == Tag::HAS_u) {
      // 如果标签为 HAS_u，检查是否整数溢出（严格无符号模式）
      if (overflows<T>(v.u, /* strict_unsigned */ true)) {
        return false;
      } else {
        // 将 v.u 转换为 T 类型，并且虚部为零时返回真
        return static_cast<T>(v.u) == num.real() && num.imag() == T();
      }
    } else if (tag == Tag::HAS_si) {
      // 如果标签为 HAS_si，目前不支持符号整数相等性的断言，抛出错误信息
      TORCH_INTERNAL_ASSERT(false, "NYI SymInt equality");
    } else if (isBoolean()) {
      // 如果是布尔值，标量不等于非布尔值，断言不是符号值
      TORCH_INTERNAL_ASSERT(!isSymbolic());
      return false;
    } else {
      // 如果以上条件均不满足，断言失败
      TORCH_INTERNAL_ASSERT(false);
    }
  }

  bool equal(bool num) const {
    if (isBoolean()) {
      // 如果是布尔值，断言不是符号值，返回与 num 相等
      TORCH_INTERNAL_ASSERT(!isSymbolic());
      return static_cast<bool>(v.i) == num;
    } else {
      // 其他情况返回假
      return false;
    }
  }

  ScalarType type() const {
    if (isComplex()) {
      // 如果是复数，返回复数类型为 ComplexDouble
      return ScalarType::ComplexDouble;
    } else if (isFloatingPoint()) {
      // 如果是浮点数，返回浮点数类型为 Double
      return ScalarType::Double;
    } else if (isIntegral(/*includeBool=*/false)) {
      // 如果是整数（不包括布尔值），表示为长整型，除非是无符号整数，因此无法表示为长整型
      if (Tag::HAS_u == tag) {
        return ScalarType::UInt64;
      }
      return ScalarType::Long;
    } else if (isBoolean()) {
      // 如果是布尔值，返回布尔类型
      return ScalarType::Bool;
    } else {
      // 如果以上条件均不满足，抛出未知标量类型的运行时错误
      throw std::runtime_error("Unknown scalar type.");
    }
  }

  Scalar(Scalar&& rhs) noexcept : tag(rhs.tag) {
    // 移动构造函数，从 rhs 移动资源
    moveFrom(std::move(rhs));
  }

  Scalar(const Scalar& rhs) : tag(rhs.tag), v(rhs.v) {
    if (isSymbolic()) {
      // 如果是符号值，增加引用计数
      c10::raw::intrusive_ptr::incref(v.p);
    }
  }

  Scalar(c10::SymInt si) {
    if (auto m = si.maybe_as_int()) {
      // 如果可以作为整数，使用 HAS_i 标签存储整数值
      tag = Tag::HAS_i;
      v.i = *m;
    } else {
      // 否则使用 HAS_si 标签存储符号整数
      tag = Tag::HAS_si;
      v.p = std::move(si).release();
    }
  }

  Scalar(c10::SymFloat sd) {
    if (sd.is_symbolic()) {
      // 如果是符号浮点数，使用 HAS_sd 标签存储符号浮点数
      tag = Tag::HAS_sd;
      v.p = std::move(sd).release();
    } else {
      // 否则使用 HAS_d 标签存储实数部分
      tag = Tag::HAS_d;
      v.d = sd.as_float_unchecked();
    }
  }

  Scalar(c10::SymBool sb) {
    if (auto m = sb.maybe_as_bool()) {
      // 如果可以作为布尔值，使用 HAS_b 标签存储布尔值
      tag = Tag::HAS_b;
      v.i = *m;
    } else {
      // 否则使用 HAS_sb 标签存储符号布尔值
      tag = Tag::HAS_sb;
      v.p = std::move(sb).release();
    }
  }
  // 关闭匿名命名空间

  // 由于 MSVC 不支持 v{ .member = ... } 的初始化列表语法，我们无法在初始化列表中设置 v
 private:
  // 定义枚举类型 Tag，表示 Scalar 可能的数据类型
  enum class Tag { HAS_d, HAS_i, HAS_u, HAS_z, HAS_b, HAS_sd, HAS_si, HAS_sb };

  // 注意 [Meaning of HAS_u]
  // ~~~~~~~~~~~~~~~~~~~~~~~
  // HAS_u 有些特殊。表面上看，它只表示我们持有一个无符号整数。
  // 然而，我们通常不区分 Scalar 中不同位大小的数据类型（例如，我们将 float 表示为 double），
  // 而是使用它来表示某些数量的数学概念（整数与浮点数）。因此，HAS_u 实际上仅用于表示无法
  // 作为有符号整数表示的无符号整数。这意味着只有 uint64_t 可能会使用这个标签；较小的类型如
  // uint8_t 可以适应普通的 int，所以出于向后兼容性的考虑，我们将其保留为 int 类型。

  // 注意：假设 self 已经被清除
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  // 强制内联函数，在 noexcept 条件下从 rhs 移动数据到当前对象
  C10_ALWAYS_INLINE void moveFrom(Scalar&& rhs) noexcept {
    v = rhs.v;  // 移动 rhs 的数据到当前对象的 v
    tag = rhs.tag;  // 将 rhs 的标签赋给当前对象的 tag
    if (rhs.tag == Tag::HAS_si || rhs.tag == Tag::HAS_sd ||
        rhs.tag == Tag::HAS_sb) {
      // 从标量中移出数据
      rhs.tag = Tag::HAS_i;  // 将 rhs 的标签重置为 HAS_i
      rhs.v.i = 0;  // 将 rhs 的整数数据重置为 0
    }
  }

  Tag tag;  // 标量的数据类型标签

  // 联合体 v_t，用于存储标量的各种可能类型
  union v_t {
    double d{};  // 双精度浮点数
    int64_t i;   // 64 位有符号整数
    // See Note [Meaning of HAS_u]
    uint64_t u;  // 64 位无符号整数
    c10::complex<double> z;  // 复数
    c10::intrusive_ptr_target* p;  // 指针
    // NOLINTNEXTLINE(modernize-use-equals-default)
    v_t() {}  // 默认构造函数
  } v;  // 标量的值

  template <
      typename T,
      typename std::enable_if_t<
          std::is_integral_v<T> && !std::is_same_v<T, bool>,
          bool>* = nullptr>
  // 标量的构造函数，用于整数类型（不包括 bool 类型）
  Scalar(T vv, bool) : tag(Tag::HAS_i) {
    v.i = convert<decltype(v.i), T>(vv);  // 将整数 vv 转换为 v.i 的类型并赋值
  }

  template <
      typename T,
      typename std::enable_if_t<
          !std::is_integral_v<T> && !c10::is_complex<T>::value,
          bool>* = nullptr>
  // 标量的构造函数，用于非整数类型且非复数类型
  Scalar(T vv, bool) : tag(Tag::HAS_d) {
    v.d = convert<decltype(v.d), T>(vv);  // 将值 vv 转换为 v.d 的类型并赋值
  }

  template <
      typename T,
      typename std::enable_if_t<c10::is_complex<T>::value, bool>* = nullptr>
  // 标量的构造函数，用于复数类型
  Scalar(T vv, bool) : tag(Tag::HAS_z) {
    v.z = convert<decltype(v.z), T>(vv);  // 将值 vv 转换为 v.z 的类型并赋值
  }
};

using OptionalScalarRef = c10::OptionalRef<Scalar>;
// 定义了一个别名 OptionalScalarRef，表示 c10::OptionalRef<Scalar>

// 定义了 Scalar 类的 to<int64_t>() 的特化模板
#define DEFINE_TO(T, name)         \
  template <>                      \
  inline T Scalar::to<T>() const { \
    return to##name();             \
  }
// 使用宏 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX 对所有标量类型进行遍历，并调用 DEFINE_TO 宏定义的模板
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_TO)
// 分别定义了 uint16_t、uint32_t 和 uint64_t 类型的 toUInt16()、toUInt32() 和 toUInt64() 特化模板
DEFINE_TO(uint16_t, UInt16)
DEFINE_TO(uint32_t, UInt32)
DEFINE_TO(uint64_t, UInt64)
#undef DEFINE_TO

} // namespace c10
// 结束 c10 命名空间
```