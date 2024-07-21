# `.\pytorch\aten\src\ATen\core\jit_type_base.h`

```
#pragma once

// 包含必要的头文件，用于类型定义和功能声明
#include <functional>
#include <memory>
#include <string>
#include <utility>

// 包含 ATen 库的特定头文件
#include <ATen/core/qualified_name.h>
#include <ATen/core/type_ptr.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

// 定义 C10 命名空间
namespace c10 {

// 定义 C10_FORALL_TYPES 宏，用于列举所有类型
#define C10_FORALL_TYPES(_) \
  _(AnyType)                \
  _(EnumType)               \
  _(AnyEnumType)            \
  _(TensorType)             \
  _(StorageType)            \
  _(TupleType)              \
  _(ListType)               \
  _(DictType)               \
  _(NumberType)             \
  _(FloatType)              \
  _(ComplexType)            \
  _(FutureType)             \
  _(AwaitType)              \
  _(RRefType)               \
  _(IntType)                \
  _(NoneType)               \
  _(StringType)             \
  _(GeneratorType)          \
  _(QuantizerType)          \
  _(BoolType)               \
  _(OptionalType)           \
  _(VarType)                \
  _(DeviceObjType)          \
  _(StreamObjType)          \
  _(FunctionType)           \
  _(ClassType)              \
  _(PyObjectType)           \
  _(CapsuleType)            \
  _(InterfaceType)          \
  _(QSchemeType)            \
  _(ScalarTypeType)         \
  _(LayoutType)             \
  _(MemoryFormatType)       \
  _(AnyListType)            \
  _(AnyTupleType)           \
  _(AnyClassType)           \
  _(SymIntType)             \
  _(SymFloatType)           \
  _(SymBoolType)            \
  _(UnionType)              \
  _(DynamicType)

// 枚举类型 TypeKind，包含所有定义的类型
enum class TypeKind {
#define DEFINE_TYPE(T) T,
  C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

// 声明一个函数，将 TypeKind 转换为对应的字符串表示
TORCH_API const char* typeKindToString(TypeKind kind);

// 声明 Type 结构体和 SharedType 结构体
struct Type;
struct SharedType;

// 定义 TypePrinter 类型，用于定制化 Type 的打印输出
using TypePrinter = std::function<std::optional<std::string>(const Type&)>;

namespace detail {
// 模板类 IsSingletonType 用于检测是否为单例类型
template <typename T>
struct IsSingletonType : public std::integral_constant<bool, false> {};
} // namespace detail

// 定义 TORCH_DECLARE_SINGLETON 宏，用于声明单例类型
#define TORCH_DECLARE_SINGLETON(Type) \
  struct Type;                                                          \
  namespace detail { \
  template <> struct IsSingletonType<Type> : public std::integral_constant<bool, true> {}; \
  }

// 声明各种单例类型
TORCH_DECLARE_SINGLETON(AnyType);
TORCH_DECLARE_SINGLETON(AnyEnumType);
TORCH_DECLARE_SINGLETON(NumberType);
TORCH_DECLARE_SINGLETON(FloatType);
TORCH_DECLARE_SINGLETON(ComplexType);
TORCH_DECLARE_SINGLETON(IntType);
TORCH_DECLARE_SINGLETON(BoolType);
TORCH_DECLARE_SINGLETON(StringType);
TORCH_DECLARE_SINGLETON(StorageType);
TORCH_DECLARE_SINGLETON(NoneType);
TORCH_DECLARE_SINGLETON(GeneratorType);
TORCH_DECLARE_SINGLETON(QuantizerType);
TORCH_DECLARE_SINGLETON(QSchemeType);

} // namespace c10
// 声明并定义全局单例对象 DeviceObjType
TORCH_DECLARE_SINGLETON(DeviceObjType);
// 声明并定义全局单例对象 StreamObjType
TORCH_DECLARE_SINGLETON(StreamObjType);
// 声明并定义全局单例对象 CapsuleType
TORCH_DECLARE_SINGLETON(CapsuleType);
// 声明并定义全局单例对象 PyObjectType
TORCH_DECLARE_SINGLETON(PyObjectType);
// 声明并定义全局单例对象 ScalarTypeType
TORCH_DECLARE_SINGLETON(ScalarTypeType);
// 声明并定义全局单例对象 LayoutType
TORCH_DECLARE_SINGLETON(LayoutType);
// 声明并定义全局单例对象 MemoryFormatType
TORCH_DECLARE_SINGLETON(MemoryFormatType);
// 声明并定义全局单例对象 AnyListType
TORCH_DECLARE_SINGLETON(AnyListType);
// 声明并定义全局单例对象 AnyTupleType
TORCH_DECLARE_SINGLETON(AnyTupleType);
// 声明并定义全局单例对象 AnyClassType
TORCH_DECLARE_SINGLETON(AnyClassType);

// 进入 detail 命名空间
namespace detail {
// 模板类 CastReturnType
template <typename T, typename Enable = void>
struct CastReturnType {
  // 默认情况下使用 std::shared_ptr<T>
  using type = std::shared_ptr<T>;
};

// 特化模板类 CastReturnType，当 T 是单例类型时
template <typename T>
struct CastReturnType<T, std::enable_if_t<IsSingletonType<T>::value>> {
  // 使用 SingletonTypePtr<T>
  using type = SingletonTypePtr<T>;
};

// 模板类 CastConstReturnType
template <typename T, typename Enable = void>
struct CastConstReturnType {
  // 默认情况下使用 std::shared_ptr<const T>
  using type = std::shared_ptr<const T>;
};

// 特化模板类 CastConstReturnType，当 T 是单例类型时
template <typename T>
struct CastConstReturnType<T, std::enable_if_t<IsSingletonType<T>::value>> {
  // 使用 SingletonTypePtr<const T>
  using type = SingletonTypePtr<const T>;
};

// 模板类 as_shared_type
template <typename T>
struct as_shared_type {
  // 默认情况下使用 SharedType*
  using type = SharedType*;
};

// 特化模板类 as_shared_type，处理 const T* 类型
template <typename T>
struct as_shared_type<const T*> {
  // 使用 const SharedType*
  using type = const SharedType *;
};
} // namespace detail

// 结构体 Type 的定义
struct TORCH_API Type {
  // 声明友元函数 operator==，用于比较两个 Type 对象是否相等
  friend TORCH_API bool operator==(const Type& lhs, const Type& rhs);
  // 私有成员变量 kind_
  private:
  TypeKind kind_;

  // 受保护构造函数，初始化 kind_
  protected:
  Type(TypeKind kind) : kind_(kind) {}

  // 默认复制构造函数和赋值运算符
  Type(const Type&) = default;
  Type& operator=(const Type&) = default;
  // 移动构造函数和移动赋值运算符
  Type(Type&&) noexcept = default;
  Type& operator=(Type&&) noexcept = default;

  // 虚函数，返回类型注解的字符串表示
  virtual std::string annotation_str_impl(const TypePrinter& /*printer*/) const {
    return str();
  }
  // 纯虚函数，比较两个 Type 对象是否相等
  // a == b
  virtual bool equals(const Type& rhs) const = 0;
  // 虚函数，对称性检查，通常为 true
  // a == b <=> b == a
  virtual bool symmetric() const {
    return true;
  }

 // 公共部分开始
 public:
  // 模板类 SingletonOrSharedTypePtr
  template <typename T>
  class SingletonOrSharedTypePtr {
   public:
    // 使用 std::shared_ptr<T>::element_type
    using element_type = typename std::shared_ptr<T>::element_type;

    // 默认构造函数
    SingletonOrSharedTypePtr() = default;

    // 隐式转换构造函数，从 std::shared_ptr<T> 转换
    /* implicit */ SingletonOrSharedTypePtr(std::shared_ptr<T> x)
        : repr_(std::move(x)) {}

    // 模板构造函数，支持从 U* 转换到 T*，条件为可转换
    template <typename U, std::enable_if_t<std::is_convertible_v<U*, T*>, bool> = true>
    /* implicit */ SingletonOrSharedTypePtr(std::shared_ptr<U> x)
        : repr_(std::move(x)) {}

    // 隐式转换构造函数，从 nullptr 转换
    /* implicit */ SingletonOrSharedTypePtr(std::nullptr_t)
        : repr_(nullptr) {}

    // 隐式转换构造函数，从 SingletonTypePtr<T> 转换
    /* implicit */ SingletonOrSharedTypePtr(SingletonTypePtr<T> p)
        : repr_(p) {}

    // 模板构造函数，支持从 SingletonTypePtr<U> 转换到 SingletonTypePtr<T>，条件为可转换
    template <typename U, std::enable_if_t<std::is_convertible_v<U*, T*>, bool> = true>
    /* implicit */ SingletonOrSharedTypePtr(SingletonTypePtr<U> p)
        : repr_(SingletonTypePtr<T>(p.get())) {}

    // 由于 pybind 的需求，需要支持从 T* 构造
    // 问题在于不清楚是否应该共享所有权。
    //
    // 情况1：如果 T 在静态上下文中被认为是 SharedType 的派生类，则应使用 shared_from_this() 并共享所有权。
    //
    // 情况2：如果 T 恰好是 Type，我们需要进行 dynamic_cast 来检查它是否是 SharedType，并做正确的操作。
    //
    // Case 3: Otherwise, T is not a SharedType. (debug-check this
    // assumption!) Use a singleton pointer.

    template <typename U = T, std::enable_if_t<std::is_base_of_v<SharedType, U>, bool> = true>
    /* implicit */ SingletonOrSharedTypePtr(T* p) : SingletonOrSharedTypePtr(static_cast<typename detail::as_shared_type<U>::type>(p)->shared_from_this()) {}

    template <typename U = T, std::enable_if_t<std::is_same_v<Type, U>, bool> = true>
    /* implicit */ SingletonOrSharedTypePtr(T* p) {
      if (auto* shared_p = dynamic_cast<typename detail::as_shared_type<U>::type>(p)) {
        // If 'p' can be casted to a shared pointer of 'U', store its shared representation
        repr_ = Repr(shared_p->shared_from_this());
      } else {
        // Otherwise, store 'p' directly as its representation
        repr_ = Repr(p);
      }
    }

    template <typename U = T, std::enable_if_t<!std::is_same_v<Type, U> && !std::is_base_of_v<SharedType, U>, bool> = true>
    /* implicit */ SingletonOrSharedTypePtr(T* p)
        : repr_(p) {
      // For types that are neither 'Type' nor derived from SharedType,
      // store 'p' directly and assert that 'p' cannot be casted to a shared pointer of 'U'.
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dynamic_cast<typename detail::as_shared_type<U>::type>(p) == nullptr);
    }

    // Default copy and move constructors and assignment operators
    SingletonOrSharedTypePtr(const SingletonOrSharedTypePtr&) = default;
    SingletonOrSharedTypePtr(SingletonOrSharedTypePtr&&) noexcept = default;
    SingletonOrSharedTypePtr& operator=(const SingletonOrSharedTypePtr&) = default;
    SingletonOrSharedTypePtr& operator=(SingletonOrSharedTypePtr&&) noexcept = default;

    // Get the pointer to the stored object
    T* get() const {
      // If stored representation is a shared pointer and not null, return its pointer,
      // otherwise return the raw pointer.
      return repr_.isSharedAndNonNull() ? repr_.shared_.repr_.get() : static_cast<T*>(repr_.rawRepr().first);
    }

    // Conversion to bool
    operator bool() const {
      // Check if the stored representation is non-null
      return repr_.isNonNull();
    }

    // Comparison with nullptr
    bool operator==(std::nullptr_t) const {
      // Check if the stored representation is null
      return !repr_.isNonNull();
    }

    // Comparison with nullptr
    bool operator!=(std::nullptr_t) const {
      // Check if the stored representation is non-null
      return repr_.isNonNull();
    }

    // Dereference operator
    template <typename U = T, std::enable_if_t<!std::is_same_v<std::remove_const_t<U>, void>, bool> = true>
    U& operator*() const {
      // Dereference the stored pointer
      return *get();
    }

    // Arrow operator
    T* operator->() const {
      // Access the stored pointer
      return get();
    }

  private:
    // NOTE: SharedPtrWrapper exists to work around a baffling bug in
    // nvcc; see comment in destroy() below.
    // Wrapper struct for holding a shared_ptr to T
    struct SharedPtrWrapper {
      SharedPtrWrapper(std::shared_ptr<T> &&x)
          : repr_(std::move(x)) {}
      std::shared_ptr<T> repr_;
    };
  } repr_;
  };

  using TypePtr = SingletonOrSharedTypePtr<Type>;
  using Ptr = TypePtr;
  using ElementType = Type;

  // subtyping relation. By default, we return true for the case
  // when the type is exactly equal or if this <: T where rhs = Optional[T]

  // if this returns false and the why_not stream is non-null, it contains
  // additional details that describe why this is not a subtype of 'rhs'.
  // This additional information should only contain details that are not
  // obvious from the annotation_str() that describes the type. For instance it
  // is clear that `int <: str` is false but not clear why `Foo <: InterfaceBar`
  // might be false.
  
  // 声明虚函数，用于判断是否为rhs的子类型，如果是则返回true，否则返回false，并通过why_not流返回详细的不匹配信息
  virtual bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const;
  
  // 返回当前类型是否为模块类型的布尔值
  virtual bool is_module() const;
  
  // 用于判断当前类型是否为rhs的子类型的简便方法，不返回详细信息
  bool isSubtypeOf(const Type& rhs) const {
    return isSubtypeOfExt(rhs, nullptr);
  }
  
  // 兼容性适配器，接受shared_ptr类型参数并调用对应的isSubtypeOf方法
  template <typename T>
  std::enable_if_t<std::is_base_of_v<Type, T>, bool>
  isSubtypeOf(const std::shared_ptr<T>& rhs) const {
    return isSubtypeOf(*rhs);
  }

  // 兼容性适配器，接受SingletonOrSharedTypePtr类型参数并调用对应的isSubtypeOf方法
  template <typename T>
  std::enable_if_t<std::is_base_of_v<Type, T>, bool>
  isSubtypeOf(const SingletonOrSharedTypePtr<T>& rhs) const {
    return isSubtypeOf(*rhs);
  }

  // 兼容性适配器，接受SingletonTypePtr类型参数并调用对应的isSubtypeOf方法
  template <typename T>
  std::enable_if_t<std::is_base_of_v<Type, T>, bool>
  isSubtypeOf(SingletonTypePtr<T> rhs) const {
    return isSubtypeOf(*rhs);
  }

  // 兼容性适配器，接受SingletonOrSharedTypePtr类型参数并调用对应的isSubtypeOfExt方法
  template <typename T>
  std::enable_if_t<std::is_base_of_v<Type, T>, bool>
  isSubtypeOfExt(const SingletonOrSharedTypePtr<T>& rhs, std::ostream* why_not) const {
    return isSubtypeOfExt(*rhs, why_not);
  }

  // 兼容性适配器，接受std::shared_ptr类型参数并调用对应的isSubtypeOfExt方法
  template <typename T>
  std::enable_if_t<std::is_base_of_v<Type, T>, bool>
  isSubtypeOfExt(const std::shared_ptr<T>& rhs, std::ostream* why_not) const {
    return isSubtypeOfExt(*rhs, why_not);
  }

  // 兼容性适配器，接受SingletonTypePtr类型参数并调用对应的isSubtypeOfExt方法
  template <typename T>
  std::enable_if_t<std::is_base_of_v<Type, T>, bool>
  isSubtypeOfExt(SingletonTypePtr<T> rhs, std::ostream* why_not) const {
    return isSubtypeOfExt(*rhs, why_not);
  }

  // 用于在FunctionSchema声明中描述当前类型的字符串表示形式
  virtual std::string str() const = 0;

  // 用于在Python类型注释中描述当前类型的字符串表示形式，与声明中的可能有所不同
  //
  // 接受一个自定义打印器，允许用户自定义输出方式
  std::string annotation_str(const TypePrinter& printer) const {
    if (printer) {
      // 打印器可以返回nullopt以使用默认实现
      if (auto renamed = printer(*this)) {
        return *renamed;
      }
    }
    return annotation_str_impl(printer);
  }
  
  // 重载方法，无参数版本，帮助调试器
  std::string annotation_str() const {
    // 为`printer`定义默认值的重载方法
    // 以帮助调试器
    // 从此处开始到下一行为止。
被
    // 调用 annotation_str 函数，并返回其结果
    return annotation_str(nullptr);
  }

  // 返回一个包含额外信息的人类可读字符串，如“类型是推断而非显式定义”，以帮助构建更用户友好的消息。
  virtual std::string repr_str() const {
    // 调用 annotation_str 函数，并返回其结果
    return annotation_str();
  }

  // 返回该对象的类型种类
  TypeKind kind() const {
    return kind_;
  }

  // 检查是否是联合类型，始终返回 false
  virtual bool isUnionType() const {
    return false;
  }

  // 检查是否需要梯度，递归检查所有包含类型中是否有需要梯度的类型
  virtual bool requires_grad() const {
    for (const auto& ct : containedTypes()) {
      if (ct->requires_grad()) {
        return true;
      }
    }
    return false;
  }

  // 动态将此对象向下转型为模板变量指示的子类，如果转型无效则返回 nullptr
  template <typename T, std::enable_if_t<!detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastReturnType<T>::type cast() {
    if (T::Kind == kind()) {
      return std::static_pointer_cast<T>(static_cast<T*>(this)->shared_from_this());
    }
    return nullptr;
  }

  // 同上，针对单例类型的特化版本
  template <typename T, std::enable_if_t<detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastReturnType<T>::type cast() {
    if (T::Kind == kind()) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this == T::get().get());
      return typename detail::CastReturnType<T>::type(static_cast<T*>(this));
    }
    return nullptr;
  }

  // 同上，针对 const 对象的非常量版本
  template <typename T, std::enable_if_t<!detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastConstReturnType<T>::type cast() const {
    if (T::Kind == kind()) {
      return std::static_pointer_cast<const T>(static_cast<const T*>(this)->shared_from_this());
    }
    return nullptr;
  }

  // 同上，针对 const 对象的常量版本
  template <typename T, std::enable_if_t<detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastConstReturnType<T>::type cast() const {
    if (T::Kind == kind()) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this == T::get().get());
      return typename detail::CastConstReturnType<T>::type(static_cast<const T*>(this));
    }
    return nullptr;
  }

  // 将此对象转换为指定的原始指针类型 T
  template <typename T>
  T* castRaw() {
    if (T::Kind == kind()) {
      return static_cast<T*>(this);
    }
    return nullptr;
  }

  // 将此对象转换为指定的 const 原始指针类型 T
  template <typename T>
  const T* castRaw() const {
    if (T::Kind == kind()) {
      return static_cast<const T*>(this);
    }
    return nullptr;
  }

  // 返回 cast<T>() 的结果，并要求返回值不能为空指针
  template <typename T>
  auto expect() {
    auto r = cast<T>();
    AT_ASSERT(r);
    return r;
  }

  // 返回 cast<const T>() 的结果，并要求返回值不能为空指针
  template <typename T>
  auto expect() const {
    auto r = cast<const T>();
    AT_ASSERT(r);
    return r;
  }

  // 返回 castRaw<T>() 的结果，并要求返回值不能为空指针引用
  template <typename T>
  T& expectRef() {
    auto* r = castRaw<T>();
    AT_ASSERT(r);
    return *r;
  }

  // 返回 castRaw<const T>() 的结果，并要求返回值不能为空指针引用
  template <typename T>
  const T& expectRef() const {
    auto* r = castRaw<const T>();
    AT_ASSERT(r);
    return *r;
  }

  // 默认析构函数
  virtual ~Type() = default;

  // 检查是否具有自由变量，此处函数体未完整
  virtual bool hasFreeVariables() const {
    // 返回 false
    return false;
  }
  // 返回此类型包含的类型列表，例如对于列表来说，返回列表元素的类型；对于元组来说，返回元组元素的类型列表
  virtual at::ArrayRef<TypePtr> containedTypes() const {
    // 返回空数组
    return {};
  }
  // 返回第 i 个包含的类型指针
  virtual TypePtr containedType(size_t i) const {
    // 返回 containedTypes() 中第 i 个元素
    return containedTypes().at(i);
  }
  // 返回包含的类型的数量
  virtual size_t containedTypeSize() const {
    // 返回 containedTypes() 的大小
    return containedTypes().size();
  }
  // 创建此类型的新版本，用 contained_types 替换其中的包含类型
  TypePtr withContained(std::vector<TypePtr> contained_types);
  // 对于每种类型的构造函数，只有在 containedTypes() 不为空时才需要重写此函数
  virtual TypePtr createWithContained(
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      std::vector<TypePtr> /*contained_types*/) const {
    // 抛出错误，提示未重载包含类型的 createWithContained 函数
    AT_ERROR(
        "type with contained types did not overload createWithContained: ",
        str());
  }
};

// 使用模板定义一个别名 SingletonOrSharedTypePtr，用于表示指向 Type 类型对象的指针或共享指针
template <typename T>
using SingletonOrSharedTypePtr = Type::SingletonOrSharedTypePtr<T>;

// 定义比较运算符重载函数，用于比较两个 SingletonOrSharedTypePtr 对象是否相等
template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

// 定义比较运算符重载函数，用于比较 SingletonOrSharedTypePtr 和 std::shared_ptr 是否相等
template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x, const std::shared_ptr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

// 定义比较运算符重载函数，用于比较 std::shared_ptr 和 SingletonOrSharedTypePtr 是否相等
template <typename T, typename U>
bool operator==(const std::shared_ptr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

// 定义比较运算符重载函数，用于比较 SingletonOrSharedTypePtr 和 SingletonTypePtr 是否相等
template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x, const SingletonTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

// 定义比较运算符重载函数，用于比较 SingletonTypePtr 和 SingletonOrSharedTypePtr 是否相等
template <typename T, typename U>
bool operator==(const SingletonTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

// 定义不相等运算符重载函数，用于比较两个 SingletonOrSharedTypePtr 对象是否不相等
template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

// 定义不相等运算符重载函数，用于比较 SingletonOrSharedTypePtr 和 std::shared_ptr 是否不相等
template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x, const std::shared_ptr<U>& y) {
  return !(x == y);
}

// 定义不相等运算符重载函数，用于比较 std::shared_ptr 和 SingletonOrSharedTypePtr 是否不相等
template <typename T, typename U>
bool operator!=(const std::shared_ptr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

// 定义不相等运算符重载函数，用于比较 SingletonOrSharedTypePtr 和 SingletonTypePtr 是否不相等
template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x, const SingletonTypePtr<U>& y) {
  return !(x == y);
}

// 定义不相等运算符重载函数，用于比较 SingletonTypePtr 和 SingletonOrSharedTypePtr 是否不相等
template <typename T, typename U>
bool operator!=(const SingletonTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

// 定义别名 TypePtr，表示 SingletonOrSharedTypePtr<Type>
using TypePtr = SingletonOrSharedTypePtr<Type>;

// 定义别名 ConstTypePtr，表示 SingletonOrSharedTypePtr<const Type>
using ConstTypePtr = SingletonOrSharedTypePtr<const Type>;

// 显式地启用 MaybeOwned<shared_ptr<T>>，而不是允许任何类型的 MaybeOwned 使用
template <typename T>
struct MaybeOwnedTraits<SingletonOrSharedTypePtr<T>>
    : public MaybeOwnedTraitsGenericImpl<SingletonOrSharedTypePtr<T>> {};

// 基类 SharedType，表示确保由 std::shared_ptr 持有的类型
struct TORCH_API SharedType : public Type, public std::enable_shared_from_this<SharedType> {
  using Type::Type;
};

// Type 类的成员函数 withContained，用于处理包含的 TypePtr
inline TypePtr Type::withContained(std::vector<TypePtr> contained_types) {
  auto current_contained = containedTypes();
  // 对于没有包含类型的 Type，不需要调用此函数。在调用前进行检查！
  //
  // (由于没有包含类型的类型可能是单例，因此 shared_from_this 将导致崩溃；
  // 我们必须提供一个虚拟的 typeptr_from_this 或 isSingleton。)
  TORCH_INTERNAL_ASSERT(!current_contained.empty() && current_contained.size() == contained_types.size());
  if (current_contained.equals(contained_types)) {
    return std::static_pointer_cast<Type>(static_cast<SharedType *>(this)->shared_from_this());
  }
  return createWithContained(std::move(contained_types));
}
# 定义了一个内联函数，用于比较两个 Type 对象是否相等
TORCH_API inline bool operator==(const Type& lhs, const Type& rhs) {
  # 如果 rhs 不对称（symmetric），则调用 rhs 的 equals 方法与 lhs 比较
  if (C10_UNLIKELY(!rhs.symmetric())) {
    return rhs.equals(lhs);
  }
  # 否则调用 lhs 的 equals 方法与 rhs 比较
  return lhs.equals(rhs);
}

# 声明了一个结构体 NamedType
struct NamedType;
# 使用 std::shared_ptr 定义了 NamedTypePtr 类型，指向 NamedType 的共享指针
using NamedTypePtr = std::shared_ptr<NamedType>;
# 使用 std::shared_ptr 定义了 ConstNamedTypePtr 类型，指向 const NamedType 的共享指针
using ConstNamedTypePtr = std::shared_ptr<const NamedType>;

# 定义了一个 NamedType 结构体，继承自 SharedType
struct TORCH_API NamedType : public SharedType {
  # 构造函数，接受类型种类 tk 和可选的限定名称 name
  NamedType(TypeKind tk, std::optional<QualifiedName> name)
      : SharedType(tk), name_(std::move(name)) {
    # 断言：如果 tk 是 TupleType、FunctionType、ClassType、InterfaceType 或 EnumType 中的一种
    TORCH_INTERNAL_ASSERT(
        tk == TypeKind::TupleType || tk == TypeKind::FunctionType ||
            tk == TypeKind::ClassType || tk == TypeKind::InterfaceType ||
            tk == TypeKind::EnumType,
        "If you add a new kind of NamedType, ",
        "please update the cast<NamedType> specialization and this assert");
  }

  # 返回类型的完全限定名称的可选值
  # 形如："foo.bar.Baz"。
  const std::optional<QualifiedName>& name() const {
    return name_;
  }

 private:
  std::optional<QualifiedName> name_;
};

} // namespace c10

# 在 std 命名空间中定义了一个模板特化：hash 函数，用于计算 c10::SingletonOrSharedTypePtr<T> 的哈希值
namespace std {
template <typename T>
struct hash<c10::SingletonOrSharedTypePtr<T>> {
  size_t operator()(const c10::SingletonOrSharedTypePtr<T>& x) const {
    return std::hash<T*>()(x.get());
  }
};
} // namespace std
```