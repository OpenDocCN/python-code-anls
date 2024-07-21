# `.\pytorch\c10\util\typeid.h`

```
#pragma once

#include <array>  // 引入标准数组库
#include <atomic>  // 引入原子操作库
#include <cstddef>  // 引入标准库，包含 size_t 和 nullptr_t 的定义
#include <cstdint>  // 引入标准整数类型定义
#include <memory>  // 引入智能指针和动态内存管理相关库
#include <mutex>  // 引入互斥锁库
#include <ostream>  // 引入输出流库
#include <string>  // 引入字符串库
#include <type_traits>  // 引入类型特性库
#include <vector>  // 引入动态数组库

#include <c10/macros/Export.h>  // 引入 C10 库的导出宏定义
#include <c10/macros/Macros.h>  // 引入 C10 库的宏定义
#include <c10/util/Exception.h>  // 引入 C10 库的异常处理相关工具
#include <c10/util/Half.h>  // 引入 C10 库的半精度浮点数处理工具
#include <c10/util/IdWrapper.h>  // 引入 C10 库的 ID 封装工具
#include <c10/util/TypeIndex.h>  // 引入 C10 库的类型索引工具
#include <c10/util/TypeTraits.h>  // 引入 C10 库的类型特性工具
#include <c10/util/irange.h>  // 引入 C10 库的整数范围工具
#include <c10/util/string_view.h>  // 引入 C10 库的字符串视图工具

#include <c10/core/ScalarType.h>  // 引入 C10 核心库的标量类型定义

/*
 * TypeIdentifier 是一个包含类型 ID 的小型类型。
 * 必须使用 CAFFE_DECLARE_KNOWN_TYPE()（在头文件中）和 CAFFE_DEFINE_KNOWN_TYPE()
 * （在 .cpp 文件中）注册类型，以便它们具有类型 ID。如果类型已注册，还可以调用
 * TypeMeta::Make<T> 创建包含有关该类型的元数据（如构造函数、析构函数、字符串化名称等）
 * 的对象。此调用返回一个 TypeMeta() 对象，实际上只是指向类型信息的指针，因此在各处传递起来很便宜。
 */

// TODO: 尽管位于 ATen 目录中，但此文件仍处于 caffe2 命名空间中。
// 原因在于 CAFFE_KNOWN_TYPE（和 CAFFE_DECLARE_KNOWN_TYPE）定义了一个模板特化，
// 其中依赖于 TypeMeta 的命名空间匹配宏调用的命名空间。这要求我们修复所有调用点，
// 这是我稍后要做的事情。因此，命名空间目前未固定。

// 使 at::Half 成为基本类型。

namespace c10::guts {
template <>
struct is_fundamental<at::Half> : std::true_type {};  // 将 at::Half 设置为基本类型
} // namespace c10::guts

namespace caffe2 {

/**
 * 类型 ID 是给定 C++ 类型的唯一 ID。
 * 您需要使用 CAFFE_KNOWN_TYPE(MyType) 注册您的类型，以便能够使用 TypeIdentifier
 * 处理自定义类型。例如，用于存储张量的数据类型。
 */
class C10_API TypeIdentifier final
    : public at::IdWrapper<TypeIdentifier, c10::util::type_index> {
 public:
  friend std::ostream& operator<<(std::ostream& stream, TypeIdentifier typeId);  // 友元函数，用于流输出 TypeIdentifier
  friend constexpr bool operator<(TypeIdentifier lhs, TypeIdentifier rhs);  // 友元函数，用于比较 TypeIdentifier

  /**
   * 返回给定类型 T 的唯一 ID。ID 对于不同类型是唯一的；对于相同类型 T，在不同函数调用间 ID 保持不变。
   * 然而，不能保证在不同运行间 ID 不变，因为 ID 在运行时生成。不要序列化 ID 以便存储。
   */
  template <typename T>
  static C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA TypeIdentifier Get() noexcept {
    return TypeIdentifier(c10::util::get_type_index<T>());  // 返回类型 T 的 TypeIdentifier
  }

  static constexpr TypeIdentifier uninitialized() {
    return TypeIdentifier(c10::util::type_index{0});  // 返回未初始化的 TypeIdentifier
  }

 private:
  constexpr explicit TypeIdentifier(c10::util::type_index id) : IdWrapper(id) {}  // 构造函数，使用类型索引初始化 TypeIdentifier
};

// 允许在 std::map / std::set 中使用
// TODO 禁止此操作，应在所有地方使用 std::unordered_map/set
inline constexpr bool operator<(TypeIdentifier lhs, TypeIdentifier rhs) {
  // 比较两个 TypeIdentifier 对象的底层 ID，用于排序
  return lhs.underlyingId() < rhs.underlyingId();
}

// 输出流操作符重载，用于将 TypeIdentifier 对象输出到流中
inline std::ostream& operator<<(
    std::ostream& stream,
    caffe2::TypeIdentifier typeId) {
  // 将 TypeIdentifier 对象的底层 ID 输出到流中
  return stream << typeId.underlyingId();
}

} // namespace caffe2

namespace at {
// 定义 DataType 别名为 caffe2::TypeIdentifier
using DataType = caffe2::TypeIdentifier;
}

// 为 caffe2::TypeIdentifier 定义哈希函数
C10_DEFINE_HASH_FOR_IDWRAPPER(caffe2::TypeIdentifier)

namespace caffe2 {

namespace detail {

// 此结构体保存实际的类型信息，每种类型将分配一个该结构体实例
struct TypeMetaData final {
  using New = void*();
  using PlacementNew = void(void*, size_t);
  using Copy = void(const void*, void*, size_t);
  using PlacementDelete = void(void*, size_t);
  using Delete = void(void*);

  // 默认构造函数，初始化所有成员变量
  constexpr TypeMetaData() noexcept
      : itemsize_(0),
        new_(nullptr),
        placementNew_(nullptr),
        copy_(nullptr),
        placementDelete_(nullptr),
        delete_(nullptr),
        id_(TypeIdentifier::uninitialized()),
        name_("nullptr (uninitialized)") {}

  // 构造函数，初始化所有成员变量
  constexpr TypeMetaData(
      size_t itemsize,
      New* newFn,
      PlacementNew* placementNew,
      Copy* copy,
      PlacementDelete* placementDelete,
      Delete* deleteFn,
      TypeIdentifier id,
      c10::string_view name) noexcept
      : itemsize_(itemsize),
        new_(newFn),
        placementNew_(placementNew),
        copy_(copy),
        placementDelete_(placementDelete),
        delete_(deleteFn),
        id_(id),
        name_(name) {}

  size_t itemsize_; // 类型的大小
  New* new_; // 指向类型的新建函数的指针
  PlacementNew* placementNew_; // 指向类型的放置新建函数的指针
  Copy* copy_; // 指向类型的复制函数的指针
  PlacementDelete* placementDelete_; // 指向类型的放置删除函数的指针
  Delete* delete_; // 指向类型的删除函数的指针
  TypeIdentifier id_; // 类型的标识符
  c10::string_view name_; // 类型的名称
};

// 抛出运行时类型逻辑错误的机制，用于处理无法在编译时预防的类型擦除问题
[[noreturn]] C10_API void _ThrowRuntimeTypeLogicError(const std::string& msg);

/**
 * 为类型定义的放置新建函数。
 */
template <typename T>
inline void _PlacementNew(void* ptr, size_t n) {
  // 针对类型 T，在给定内存位置 ptr 上放置 n 个对象
  T* typed_ptr = static_cast<T*>(ptr);
  for (const auto i : c10::irange(n)) {
    new (typed_ptr + i) T;
  }
}

/**
 * 对于不可默认构造的类型，使用这个函数抛出异常。
 */
template <typename T>
inline void _PlacementNewNotDefault(void* /*ptr*/, size_t /*n*/) {
  // 抛出类型不可默认构造的运行时错误
  _ThrowRuntimeTypeLogicError(
      "Type " + std::string(c10::util::get_fully_qualified_type_name<T>()) +
      " is not default-constructible.");
}

/**
 * 选择适当的放置新建函数，如果类型可默认构造，返回对应函数指针。
 */
template <
    typename T,
    std::enable_if_t<std::is_default_constructible_v<T>>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  return (c10::guts::is_fundamental<T>::value || std::is_pointer_v<T>)
      ? nullptr
      : &_PlacementNew<T>;
}

/**
 * 如果类型不可默认构造，则返回空指针。
 */
template <
    typename T,
    std::enable_if_t<!std::is_default_constructible_v<T>>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  return &_PlacementNewNotDefault<T>;
}
    // 如果类型 T 不可默认构造，则启用 SFINAE 技术，用于函数模板重载的条件判断
    std::enable_if_t<!std::is_default_constructible_v<T>>* = nullptr>
/**
 * Selects and returns the appropriate placement new function for type T based on SFINAE.
 */
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  static_assert(
      !c10::guts::is_fundamental<T>::value && !std::is_pointer_v<T>,
      "this should have picked the other SFINAE case");
  return &_PlacementNewNotDefault<T>;
}

/**
 * Default new operator for type T.
 */
template <typename T>
inline void* _New() {
  return new T;
}

/**
 * Throws a runtime error indicating type T is not default-constructible.
 */
template <typename T>
inline void* _NewNotDefault() {
  _ThrowRuntimeTypeLogicError(
      "Type " + std::string(c10::util::get_fully_qualified_type_name<T>()) +
      " is not default-constructible.");
}

/**
 * Selects and returns the appropriate new function for type T based on default constructibility.
 */
template <
    typename T,
    std::enable_if_t<std::is_default_constructible_v<T>>* = nullptr>
inline constexpr TypeMetaData::New* _PickNew() {
  return &_New<T>;
}

/**
 * Selects and returns the appropriate new function for type T based on non-default constructibility.
 */
template <
    typename T,
    std::enable_if_t<!std::is_default_constructible_v<T>>* = nullptr>
inline constexpr TypeMetaData::New* _PickNew() {
  return &_NewNotDefault<T>;
}

/**
 * Typed copy function for classes.
 */
template <typename T>
inline void _Copy(const void* src, void* dst, size_t n) {
  const T* typed_src = static_cast<const T*>(src);
  T* typed_dst = static_cast<T*>(dst);
  for (const auto i : c10::irange(n)) {
    typed_dst[i] = typed_src[i];
  }
}

/**
 * Throws a runtime error indicating type T does not allow assignment.
 */
template <typename T>
inline void _CopyNotAllowed(const void* /*src*/, void* /*dst*/, size_t /*n*/) {
  _ThrowRuntimeTypeLogicError(
      "Type " + std::string(c10::util::get_fully_qualified_type_name<T>()) +
      " does not allow assignment.");
}

/**
 * Selects and returns the appropriate copy function for type T based on copy assignability.
 */
template <typename T, std::enable_if_t<std::is_copy_assignable_v<T>>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
  return (c10::guts::is_fundamental<T>::value || std::is_pointer_v<T>)
      ? nullptr
      : &_Copy<T>;
}

/**
 * Selects and returns the appropriate copy function for type T based on non-copy assignability.
 */
template <
    typename T,
    std::enable_if_t<!std::is_copy_assignable_v<T>>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
  static_assert(
      !c10::guts::is_fundamental<T>::value && !std::is_pointer_v<T>,
      "this should have picked the other SFINAE case");
  return &_CopyNotAllowed<T>;
}

/**
 * Destructor for non-fundamental types.
 */
template <typename T>
inline void _PlacementDelete(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (const auto i : c10::irange(n)) {
    typed_ptr[i].~T();
  }
}

/**
 * Selects and returns the appropriate placement delete function for type T based on fundamental check.
 */
template <typename T>
inline constexpr TypeMetaData::PlacementDelete* _PickPlacementDelete() {
  return (c10::guts::is_fundamental<T>::value || std::is_pointer_v<T>)
      ? nullptr
      : &_PlacementDelete<T>;
}

/**
 * Default delete operator for type T.
 */
template <typename T>
inline void _Delete(void* ptr) {
  T* typed_ptr = static_cast<T*>(ptr);
  delete typed_ptr;
}

/**
 * Selects and returns the appropriate delete function for type T.
 */
template <class T>
inline constexpr TypeMetaData::Delete* _PickDelete() noexcept {
  return &_Delete<T>;
}

/**
 * Placeholder class for uninitialized types.
 */
class _Uninitialized final {};

} // namespace detail

//
// note: this is outside TypeMeta bc gcc seems to have trouble
// with scalarTypeItemSizes as a constexpr static member used by
// a public inline instance method
//

// item sizes for TypeMeta::itemsize() fast path
/**
 * 定义了一个静态 constexpr 数组，存储各种标量类型的大小，通过宏展开将每种类型的大小放入数组中
 */
static constexpr std::array<uint8_t, NumScalarTypes> scalarTypeItemSizes = {
#define SCALAR_TYPE_SIZE(T, name) sizeof(T),
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_SIZE)
#undef SCALAR_TYPE_SIZE
        0, // Undefined
};

/**
 * TypeMeta 是一个轻量级类，用于存储容器（如 blob）或张量的数据类型，带有唯一的运行时 ID。
 * 还存储一些额外数据，如项目大小和类型的名称，用于运行时检查。
 */
class C10_API TypeMeta final {
 public:
  using New = detail::TypeMetaData::New;
  using PlacementNew = detail::TypeMetaData::PlacementNew;
  using Copy = detail::TypeMetaData::Copy;
  using PlacementDelete = detail::TypeMetaData::PlacementDelete;
  using Delete = detail::TypeMetaData::Delete;

  /** 
   * 创建一个虚拟的 TypeMeta 对象。要为特定类型创建 TypeMeta 对象，请使用 TypeMeta::Make<T>()。
   */
  TypeMeta() noexcept;

  /**
   * 复制构造函数。
   */
  TypeMeta(const TypeMeta& src) noexcept = default;

  /**
   * 赋值运算符。
   */
  TypeMeta& operator=(const TypeMeta& src) noexcept = default;

  TypeMeta(TypeMeta&& rhs) noexcept = default;

  /**
   * 将当前对象设置为指定的 ScalarType 类型。
   */
  inline TypeMeta& operator=(ScalarType scalar_type) noexcept {
    index_ = static_cast<uint16_t>(scalar_type);
    return *this;
  }

 private:
  // TypeMeta 只能通过 Make 函数创建，确保不会创建错误混合的 TypeMeta 对象。
  explicit TypeMeta(const uint16_t index) noexcept : index_(index) {}

 public:
  /**
   * 返回类型 ID。
   */
  TypeIdentifier id() const noexcept {
    return data().id_;
  }
  
  /**
   * 如果当前对象表示某种 ScalarType 类型，则返回 true。
   */
  inline bool isScalarType() const noexcept {
    return index_ < NumScalarTypes;
  }

  /**
   * 如果当前对象表示指定的 ScalarType 类型，则返回 true。
   */
  inline bool isScalarType(ScalarType scalar_type) const noexcept {
    return index_ == static_cast<uint16_t>(scalar_type);
  }

  /**
   * 返回项目的大小。
   * 如果是标量类型，则使用预定义的数组 scalarTypeItemSizes 获取大小。
   */
  inline size_t itemsize() const noexcept {
    if (C10_LIKELY(isScalarType())) {
      return scalarTypeItemSizes[index_];
    }
    return data().itemsize_;
  }

  /**
   * 返回单个项目的新建函数指针。
   */
  New* newFn() const noexcept {
    return data().new_;
  }

  /**
   * 返回单个项目的就地新建函数指针。
   */
  PlacementNew* placementNew() const noexcept {
    return data().placementNew_;
  }

  /**
   * 返回单个项目的复制函数指针。
   */
  Copy* copy() const noexcept {
    return data().copy_;
  }

  /**
   * 返回单个项目的析构函数指针。
   */
  PlacementDelete* placementDelete() const noexcept {
    return data().placementDelete_;
  }

  /**
   * 返回单个项目的删除函数指针。
   */
  Delete* deleteFn() const noexcept {
    return data().delete_;
  }

  /**
   * 返回类型的可打印名称。
   */
  c10::string_view name() const noexcept {
    // 返回数据对象的名称
    return data().name_;
  }

  // 定义 TypeMeta 类的相等比较运算符，判断两个对象是否相等
  friend bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept;

  // 模板函数，用于检查当前 TypeMeta 对象是否匹配模板类型 T
  template <typename T>
  bool Match() const noexcept {
    // 调用全局的相等比较运算符判断当前 TypeMeta 对象是否与类型 T 匹配
    return (*this == Make<T>());
  }

  // 以下是可以通过传递特定类型调用的静态函数

  // 返回类型 T 对应的 TypeIdentifier
  template <class T>
  static C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA TypeIdentifier Id() noexcept {
    return TypeIdentifier::Get<T>();
  }

  // 返回类型 T 的名称视图
  template <class T>
  static c10::string_view TypeName() noexcept {
    return c10::util::get_fully_qualified_type_name<T>();
  }

  // 返回类型 T 的大小
  template <class T>
  static constexpr size_t ItemSize() noexcept {
    return sizeof(T);
  }

  /**
   * 返回与类型 T 对应的 TypeMeta 对象。
   */
  template <typename T>
  static TypeMeta Make() {
    // 在这里声明实例的指针，但是它的定义在 .cpp 文件中。
    // 我们需要消除编译器关于使用未定义变量模板的警告。
    // 对于那些不识别 '-Wundefined-var-template' 的编译器，我们需要禁用 '-Wpragmas' 和 '-Wunknown-warning-option'，
    // 否则会在我们尝试禁用警告时出错。
    // 详见 https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html
#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wundefined-var-template"
#endif
    // 返回 TypeMeta 对象，使用模板函数 _typeMetaData<T>() 获取 T 类型的元数据
    return TypeMeta(_typeMetaData<T>());
#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif
  }

  /**
   * 将 ScalarType 枚举值转换为 TypeMeta 句柄
   */
  static inline caffe2::TypeMeta fromScalarType(ScalarType scalar_type) {
    // 将枚举值转换为 uint16_t 类型的索引
    const auto index = static_cast<uint16_t>(scalar_type);
    // 在调试模式下断言索引小于 NumScalarTypes
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        index < NumScalarTypes,
        "Unrecognized Scalartype ",
        scalar_type,
        " (please report this error)");
    // 返回对应的 TypeMeta 句柄
    return TypeMeta(index);
  }

  /**
   * 将 TypeMeta 句柄转换为 ScalarType 枚举值
   */
  inline ScalarType toScalarType() {
    // 如果是标量类型，直接将索引转换为 ScalarType 枚举值返回
    if (C10_LIKELY(isScalarType())) {
      return static_cast<ScalarType>(index_);
    }
    // 否则调用错误处理函数
    error_unsupported_typemeta(*this);
  }

 private:
  // 声明一个静态方法，指明该方法不会返回
  [[noreturn]] static void error_unsupported_typemeta(caffe2::TypeMeta dtype);

  // 硬限制注册类型的数量
  // 注意: constexpr 在 Windows 编译中引发错误 "member may not be initialized"
  // static constexpr size_t MaxTypeIndex = 32;
  //
#if defined C10_MOBILE
// 之所以不使用 UINT8_MAX，是因为数组初始化占用空间与数组大小成比例。
// 编译器似乎会添加代码（或数据填充）以使用空元素来初始化数组。详细信息请参见
// https://github.com/pytorch/pytorch/pull/51881
//
#define MaxTypeIndex                                                           \
  (NumScalarTypes + 15 /* number of CAFFE_DEFINE_KNOWN_TYPE in typeid.cpp */ + \
   1 /* 1 more for caffe2 tensor */)
#else
#define MaxTypeIndex UINT8_MAX
#endif

  // 保护类型元数据分配的互斥锁
  // NOLINTNEXTLINE(facebook-hte-NonPodStaticDeclaration)
  static std::mutex& getTypeMetaDatasLock();
  // 下一个类型索引
  static uint16_t nextTypeIndex;

  // 返回类型元数据的指针
  static detail::TypeMetaData* typeMetaDatas();

  // 返回给定类型的现有元数据索引
  static uint16_t existingMetaDataIndexForType(TypeIdentifier identifier);

 public:
#ifdef __CUDACC__
  // 注意 [TypeIdentifier::Get nvcc/clang 差异]
  // nvcc 和 clang 对 TypeIdentifier::Get 的结果不一致，因为 TypeIdentifier::Get 依赖于
  // __PRETTY_FUNCTION__，它们对类型的规范名称不一样（例如，nvcc 规范为 `short unsigned int`，但 clang 称其为 `unsigned short`）。
  // 隐藏这个函数的实现，使得 TypeIdentifier::Get 始终使用 clang（或主机 C++ 编译器）。
  template <class T>
  C10_EXPORT static uint16_t addTypeMetaData();
#else
  template <class T>
  // 添加类型 T 的元数据
  C10_EXPORT static uint16_t addTypeMetaData() {
    // 获取类型 T 的标识符
    const auto identifier = TypeIdentifier::Get<T>();
    // 在函数的其余部分需要持有此标识符，以保护：
    // 1) existingMetaDataIndexForType()
    // 2) nextTypeIndex++
    // ...
    // 获取类型元数据的锁，确保线程安全地访问
    std::lock_guard<std::mutex> lock(getTypeMetaDatasLock());
    // 检查给定标识符的类型元数据是否已存在
    // 如果在不同的动态共享库中已经存在，则直接返回其索引
    const uint16_t existing_index = existingMetaDataIndexForType(identifier);
    if (existing_index != MaxTypeIndex) {
      // 如果已存在，则直接返回已存在的索引
      return existing_index;
    }
    // 计算新的类型索引，并递增到下一个可用的索引
    const uint16_t index = nextTypeIndex++;
    // 检查新索引是否超出最大允许的类型索引值
    TORCH_CHECK(
        index <= MaxTypeIndex,
        "Maximum number of CAFFE_KNOWN_TYPE declarations has been exceeded. ",
        "Please report this issue.");
    // 将新类型的元数据写入类型元数据数组中
    typeMetaDatas()[index] = detail::TypeMetaData{
        sizeof(T),                                       // 记录类型大小
        detail::_PickNew<T>(),                           // 选择适当的 new 函数
        detail::_PickPlacementNew<T>(),                  // 选择适当的 placement new 函数
        detail::_PickCopy<T>(),                          // 选择适当的拷贝函数
        detail::_PickPlacementDelete<T>(),               // 选择适当的 placement delete 函数
        detail::_PickDelete<T>(),                        // 选择适当的 delete 函数
        identifier,                                      // 记录类型标识符
        c10::util::get_fully_qualified_type_name<T>()};  // 记录类型的全限定名
    // 返回新类型的索引
    return index;
  }


这段代码主要用于将新类型的元数据写入到一个数组中，并返回该类型的索引。
#endif

 private:
  // specializations return indexes into typeMetaDataInstances()
  // 特化返回到 typeMetaDataInstances() 的索引
  template <class T>
  C10_API static uint16_t _typeMetaData() noexcept;

  //
  // TypeMeta just wraps this index
  //
  // TypeMeta 只是简单封装了这个索引

  uint16_t index_;

  // 返回对应索引处的 TypeMetaData 引用
  inline const detail::TypeMetaData& data() const {
    return typeMetaDatas()[index_];
  }
};

// specializations of TypeMeta::_typeMetaData for ScalarType types

#define DEFINE_SCALAR_METADATA_INSTANCE(T, name)             \
  template <>                                                \
  constexpr uint16_t TypeMeta::_typeMetaData<T>() noexcept { \
    return static_cast<uint16_t>(ScalarType::name);          \
  }
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_METADATA_INSTANCE)
#undef DEFINE_SCALAR_METADATA_INSTANCE

template <>
C10_EXPORT constexpr uint16_t TypeMeta::_typeMetaData<
    detail::_Uninitialized>() noexcept {
  return static_cast<uint16_t>(ScalarType::Undefined);
}

// TypeMeta 的默认构造函数，初始化为未初始化状态的 TypeMetaData 索引
inline TypeMeta::TypeMeta() noexcept
    : index_(_typeMetaData<detail::_Uninitialized>()) {}

// TypeMeta 的相等运算符重载，比较两个 TypeMeta 对象的索引是否相等
inline bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
  return (lhs.index_ == rhs.index_);
}
// TypeMeta 的不等运算符重载，比较两个 TypeMeta 对象的索引是否不相等
inline bool operator!=(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
  return !operator==(lhs, rhs);
}

// TypeMeta 的流输出运算符重载，输出 TypeMeta 对象的名称到输出流
inline std::ostream& operator<<(
    std::ostream& stream,
    caffe2::TypeMeta typeMeta) {
  return stream << typeMeta.name();
}

/**
 * Register unique id for a type so it can be used in TypeMeta context, e.g. be
 * used as a type for Blob or for Tensor elements.
 *
 * CAFFE_KNOWN_TYPE is deprecated; prefer CAFFE_DECLARE_KNOWN_TYPE and
 * CAFFE_DEFINE_KNOWN_TYPE.
 *
 * CAFFE_KNOWN_TYPE does explicit instantiation of TypeIdentifier::Get<T>
 * template function and thus needs to be put in a single translation unit (.cpp
 * file) for a given type T. Other translation units that use type T as a type
 * of the caffe2::Blob or element type of caffe2::Tensor need to depend on the
 * translation unit that contains CAFFE_KNOWN_TYPE declaration via regular
 * linkage dependencies.
 *
 * NOTE: the macro needs to be invoked in ::caffe2 namespace
 */
// Implementation note: in MSVC, we will need to prepend the C10_API
// keyword in order to get things compiled properly. in Linux, gcc seems to
// create attribute ignored error for explicit template instantiations, see
//   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0537r0.html
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51930
// and as a result, we define these two macros slightly differently.
#if defined(_MSC_VER) || defined(__clang__)
#define EXPORT_IF_NOT_GCC C10_EXPORT
#else
#define EXPORT_IF_NOT_GCC
#endif

// CAFFE_KNOWN_TYPE is deprecated! Use CAFFE_DECLARE_KNOWN_TYPE and
// CAFFE_DEFINE_KNOWN_TYPE instead.
#define CAFFE_KNOWN_TYPE(T)                                          \
  template uint16_t TypeMeta::addTypeMetaData<T>();                  \
  template <>                                                        \
  EXPORT_IF_NOT_GCC uint16_t TypeMeta::_typeMetaData<T>() noexcept { \
    static const uint16_t index = addTypeMetaData<T>();              \
    return index;                                                    \
  }


// 宏定义：CAFFE_KNOWN_TYPE(T)
// 功能：定义模板特化，用于为类型 T 添加类型元数据
// 注意：此宏用于声明和定义 TypeMeta 类中的特定类型元数据函数。



#define CAFFE_DEFINE_KNOWN_TYPE(T, ident)                   \
  template uint16_t TypeMeta::addTypeMetaData<T>();         \
  namespace detail {                                        \
  EXPORT_IF_NOT_GCC const uint16_t ident##_metadata_index = \
      TypeMeta::addTypeMetaData<T>();                       \
  } // namespace detail


// 宏定义：CAFFE_DEFINE_KNOWN_TYPE(T, ident)
// 功能：定义模板特化，并在细节命名空间中导出类型元数据索引
// 注意：此宏用于在 TypeMeta 类中定义特定类型的元数据，并将其索引导出到命名空间 detail 中。



// Unlike CAFFE_KNOWN_TYPE, CAFFE_DECLARE_KNOWN_TYPE avoids a function
// call to access _typeMetaData in the common case.
#define CAFFE_DECLARE_KNOWN_TYPE(T, ident)                 \
  extern template uint16_t TypeMeta::addTypeMetaData<T>(); \
  namespace detail {                                       \
  extern C10_API const uint16_t ident##_metadata_index;    \
  } /* namespace detail */                                 \
  template <>                                              \
  EXPORT_IF_NOT_GCC C10_ALWAYS_INLINE uint16_t             \
  TypeMeta::_typeMetaData<T>() noexcept {                  \
    return detail::ident##_metadata_index;                 \
  }


// 宏定义：CAFFE_DECLARE_KNOWN_TYPE(T, ident)
// 功能：声明模板特化，并定义内联函数用于返回类型元数据索引
// 注意：此宏用于声明 TypeMeta 类中特定类型的元数据，并定义内联函数，直接返回元数据索引。



#define CAFFE_KNOWN_TYPE_NOEXPORT(T)                    \
  template <>                                           \
  uint16_t TypeMeta::_typeMetaData<T>() noexcept {      \
    static const uint16_t index = addTypeMetaData<T>(); \
    return index;                                       \
  }


// 宏定义：CAFFE_KNOWN_TYPE_NOEXPORT(T)
// 功能：定义不导出的模板特化，用于为类型 T 添加类型元数据
// 注意：此宏用于在 TypeMeta 类中定义特定类型的元数据，但不导出任何符号。



CAFFE_DECLARE_KNOWN_TYPE(std::string, std_string)
CAFFE_DECLARE_KNOWN_TYPE(char, char)
CAFFE_DECLARE_KNOWN_TYPE(std::unique_ptr<std::mutex>, std_unique_ptr_std_mutex)
CAFFE_DECLARE_KNOWN_TYPE(
    std::unique_ptr<std::atomic<bool>>,
    std_unique_ptr_std_atomic_bool)
CAFFE_DECLARE_KNOWN_TYPE(std::vector<int32_t>, std_vector_int32_t)
CAFFE_DECLARE_KNOWN_TYPE(std::vector<int64_t>, std_vector_int64_t)
CAFFE_DECLARE_KNOWN_TYPE(std::vector<unsigned long>, std_vector_unsigned_long)
CAFFE_DECLARE_KNOWN_TYPE(bool*, bool_ptr)
CAFFE_DECLARE_KNOWN_TYPE(char*, char_ptr)
CAFFE_DECLARE_KNOWN_TYPE(int*, int_ptr)


// CAFFE_DECLARE_KNOWN_TYPE 宏的多次调用
// 每次调用声明或定义了 TypeMeta 类中特定类型的元数据。
// 每个宏调用对应不同的类型，如 std::string、char、std::unique_ptr<std::mutex> 等。
// 每个声明或定义确保特定类型的元数据可以在代码中使用。



// For some of the compilers, long is defined separately from int32_t and
// int64_t. As a result we will need to actually define them separately.
// It is recommended that one does NOT use long - use int32_t and int64_t
// explicitly. Explicit long type annotation may go away in the future.
// details: This hack works by defining a _guard_long_unique type, which is
// long iff the compiler has a separate long type and is a dummy type otherwise.
// we then allocate a type id to that _guard_long_unique. If the compiler has a
// separate long type, this allocates a type id for long. Otherwise, it


// 注释说明：
// 一段注释，解释了长整型（long）在某些编译器中与 int32_t 和 int64_t 定义不同的情况。
// 建议不要直接使用 long，而是显式地使用 int32_t 和 int64_t。
// 此段注释提醒开发者有关长整型的使用建议，并对其未来可能的去除进行了讨论。
// 在 detail 命名空间中定义一个模板类 _guard_long_unique_dummy，用作占位符类型
namespace detail {
template <class T>
class _guard_long_unique_dummy final {};
// 定义一个模板别名 _guard_long_unique，根据条件选择 _guard_long_unique_dummy 或原类型 T
template <class T>
using _guard_long_unique = std::conditional_t<
    std::is_same_v<long, int32_t> || std::is_same_v<long, int64_t>,
    _guard_long_unique_dummy<T>,
    T>;
} // namespace detail

// 声明一个名为 detail_guard_long_unique_long 的类型，这是 _guard_long_unique<long> 的类型别名
CAFFE_DECLARE_KNOWN_TYPE(
    detail::_guard_long_unique<long>,
    detail_guard_long_unique_long);

// 声明一个名为 detail_guard_long_unique_std_vector_long 的类型，这是 _guard_long_unique<std::vector<long>> 的类型别名
CAFFE_DECLARE_KNOWN_TYPE(
    detail::_guard_long_unique<std::vector<long>>,
    detail_guard_long_unique_std_vector_long)

// 声明一个名为 float_ptr 的类型，这是 float* 的类型别名
CAFFE_DECLARE_KNOWN_TYPE(float*, float_ptr)

// 声明一个名为 at_Half 的类型，这是 at::Half* 的类型别名
CAFFE_DECLARE_KNOWN_TYPE(at::Half*, at_Half)

// 结束 caffe2 命名空间
} // namespace caffe2
```