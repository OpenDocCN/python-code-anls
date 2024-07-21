# `.\pytorch\aten\src\ATen\core\ivalue.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/core/DimVector.h>
// 包含 ATen 库中的 DimVector 头文件

#include <ATen/core/TensorBody.h>
// 包含 ATen 库中的 TensorBody 头文件

#include <ATen/core/blob.h>
// 包含 ATen 库中的 blob 头文件

#include <ATen/core/custom_class.h>
// 包含 ATen 库中的 custom_class 头文件

#include <ATen/core/ivalue_to.h>
// 包含 ATen 库中的 ivalue_to 头文件

#include <ATen/core/jit_type_base.h>
// 包含 ATen 库中的 jit_type_base 头文件

#include <ATen/core/type_factory.h>
// 包含 ATen 库中的 type_factory 头文件

#include <c10/core/SymBool.h>
// 包含 c10 库中的 SymBool 头文件

#include <c10/core/SymFloat.h>
// 包含 c10 库中的 SymFloat 头文件

#include <c10/macros/Export.h>
// 包含 c10 库中的 Export 头文件

#include <c10/util/MaybeOwned.h>
// 包含 c10 库中的 MaybeOwned 头文件

#include <c10/util/intrusive_ptr.h>
// 包含 c10 库中的 intrusive_ptr 头文件

#include <type_traits>
// 包含 type_traits 标准库头文件

#include <unordered_map>
// 包含 unordered_map 标准库头文件

#include <unordered_set>
// 包含 unordered_set 标准库头文件

#include <utility>
// 包含 utility 标准库头文件

namespace torch {
class TORCH_API CustomClassHolder : public c10::intrusive_ptr_target {};
// 定义了一个 TORCH_API 类 CustomClassHolder，继承自 c10::intrusive_ptr_target

namespace jit {
using ::torch::CustomClassHolder;
// 使用 torch 命名空间下的 CustomClassHolder

struct Function;
// 声明一个结构体 Function

struct CompilationUnit;
// 声明一个结构体 CompilationUnit

struct Module;
// 声明一个结构体 Module
} // namespace jit
} // namespace torch

namespace c10 {
template <class Key, class Value>
class Dict;
// 定义一个模板类 Dict，用于表示键值对的字典

template <class T>
class List;
// 定义一个模板类 List，用于表示列表

template <class T>
class IListRef;
// 定义一个模板类 IListRef，用于表示列表的引用

struct IValue;
// 声明一个结构体 IValue

struct ClassType;
// 声明一个结构体 ClassType

struct Type;
// 声明一个结构体 Type

class RRefInterface;
// 声明一个类 RRefInterface

struct ClassType;
using ClassTypePtr = std::shared_ptr<ClassType>;
// 使用共享指针定义 ClassTypePtr 类型

TORCH_API bool _fastEqualsForContainer(const IValue& lhs, const IValue& rhs);
// 声明一个名为 _fastEqualsForContainer 的函数，用于比较两个 IValue 是否相等

TORCH_API torch::jit::Function* checkObjectSortSchema(
    const c10::ClassTypePtr& t,
    std::stringstream& why_not);
// 声明一个名为 checkObjectSortSchema 的函数，用于检查对象的排序模式

// A comparator that checks ordering of two IValues of same type.
// 一个比较器，用于检查两个相同类型的 IValue 的顺序。
typedef std::function<bool(const IValue& a, const IValue& b)> IValueComparator;

TORCH_API IValueComparator getLessThanComparator(const IValue& v);
// 声明一个名为 getLessThanComparator 的函数，返回一个用于比较小于关系的 IValueComparator

TORCH_API IValueComparator getGreaterThanComparator(const IValue& v);
// 声明一个名为 getGreaterThanComparator 的函数，返回一个用于比较大于关系的 IValueComparator

namespace ivalue {
struct Tuple;
// 声明一个结构体 Tuple

struct Future;
// 声明一个结构体 Future

struct Await;
// 声明一个结构体 Await

struct ConstantString;
// 声明一个结构体 ConstantString

struct GenericDict;
// 声明一个结构体 GenericDict

struct Object;
// 声明一个结构体 Object

struct PyObjectHolder;
// 声明一个结构体 PyObjectHolder

struct EnumHolder;
// 声明一个结构体 EnumHolder

// We need a ComplexHolder because currently the payloads in the Union
// only take 64 bits. Since ComplexDouble takes up 128 bits, and is too big
// to fit in the IValue directly, we indirect complex numbers through an
// intrusive pointer to ComplexHolder (which contains a c10::complex).
// 我们需要 ComplexHolder，因为当前 Union 中的有效负载只能容纳 64 位。
// 由于 ComplexDouble 占用 128 位，太大无法直接放入 IValue 中，我们通过指向 ComplexHolder 的侵入式指针间接引用复数（其中包含 c10::complex）。

struct ComplexHolder : c10::intrusive_ptr_target {
 public:
  template <typename T>
  ComplexHolder(c10::complex<T> c) {
    val = convert<decltype(val), c10::complex<T>>(c);
  }
  ComplexHolder() = default;
  c10::complex<double> val;
};

// Similar to ComplexHolder, for StreamData3
// 类似于 ComplexHolder，用于 StreamData3

struct StreamData3Holder : c10::intrusive_ptr_target {
 public:
  StreamData3Holder(struct c10::StreamData3 d) : val(d) {}
  StreamData3Holder() = delete;
  struct c10::StreamData3 val;
};

} // namespace ivalue

// This is an owning wrapper for a std::optional<std::vector<T>>
// that can be implicitly converted to a (non-owning) optional<ArrayRef<T>>.
// Its purpose is to be used in generated code to keep the vector alive
// either until the end of a statement (as a temporary), or as a saved arg
// in autograd.
// 这是 std::optional<std::vector<T>> 的拥有包装器，
// 可以隐式转换为（非拥有）optional<ArrayRef<T>>。
// 其目的是在生成的代码中使用，以保持向量在语句结束时（作为临时变量）或在自动求导中作为保存的参数。

template <typename T>
// 定义一个结构体 OptionalArray，用于包装可选的 std::vector<T> 对象
struct OptionalArray {
  // 存储一个可选的 std::vector<T> 对象
  std::optional<std::vector<T>> list;

  // 默认构造函数，默认初始化 list
  OptionalArray() = default;
  
  // 接受一个 std::vector<T> 参数的构造函数，移动赋值给 list
  OptionalArray(std::vector<T> val) : list(std::move(val)) {}

  // 当保存反向传递参数时使用
  // 将 std::optional<ArrayRef<T>> 赋值给当前对象的 operator=
  OptionalArray& operator=(std::optional<ArrayRef<T>> ref) {
    if (ref) {
      // 如果 ref 有值，则从 ref 构造一个新的 std::vector<T> 并赋给 list
      list = std::vector<T>(ref->begin(), ref->end());
    } else {
      // 如果 ref 为空，则将 list 置为 nullopt
      list = nullopt;
    }
    return *this;
  }

  // 当保存反向传递参数时使用
  // 将 c10::OptionalArrayRef<T> 赋值给当前对象的 operator=
  OptionalArray& operator=(c10::OptionalArrayRef<T> ref) {
    if (ref) {
      // 如果 ref 有值，则从 ref 构造一个新的 std::vector<T> 并赋给 list
      list = std::vector<T>(ref->begin(), ref->end());
    } else {
      // 如果 ref 为空，则将 list 置为 nullopt
      list = nullopt;
    }
    return *this;
  }

  // 将当前对象转换为 std::optional<c10::ArrayRef<T>> 类型
  operator std::optional<c10::ArrayRef<T>>() {
    if (!list) {
      // 如果 list 为空，则返回 nullopt
      return nullopt;
    }
    // 否则返回 list 的值
    return *list;
  }

  // 将当前对象转换为 c10::OptionalArrayRef<T> 类型
  operator c10::OptionalArrayRef<T>() {
    if (!list) {
      // 如果 list 为空，则返回 nullopt
      return nullopt;
    }
    // 否则返回 list 的值
    return *list;
  }
};

// Capsule 是自定义 C++ 类的内部实现细节。我们将其定义为
// c10::intrusive_ptr<torch::CustomClassHolder> 的拥有包装器。
// 这个包装器作为类型擦除的自定义类对象指针的抽象存在。它还允许 pybind11
// 将其视为一个独立的类进行注册，而不是一个自定义指针持有器，
// 后者的类型转换器会尝试自动“解包”它。
struct Capsule {
  // 持有 c10::intrusive_ptr<torch::CustomClassHolder> 的对象指针
  c10::intrusive_ptr<torch::CustomClassHolder> obj_ptr;
  
  // 显式构造函数，接受一个 c10::intrusive_ptr<torch::CustomClassHolder> 指针，并移动赋值给 obj_ptr
  explicit Capsule(c10::intrusive_ptr<torch::CustomClassHolder> ptr)
      : obj_ptr(std::move(ptr)) {}
};

// IValue 是解释器用于保存所有值类型的通用标签联合体。
// 它是一个16字节的对象，具有8字节的有效载荷和8字节的标签。
// 标签目前占据4字节用于确定类型，另1字节标记是否为 c10::intrusive_ptr_target 的子类型，
// 需要保留/释放调用。

// 定义一个宏，用于列出所有可能的 IValue 标签
#define TORCH_FORALL_TAGS(_) \
  _(None)                    \
  _(Tensor)                  \
  _(Storage)                 \
  _(Double)                  \
  _(ComplexDouble)           \
  _(Int)                     \
  _(SymInt)                  \
  _(SymFloat)                \
  _(SymBool)                 \
  _(Bool)                    \
  _(Tuple)                   \
  _(String)                  \
  _(Blob)                    \
  _(GenericList)             \
  _(GenericDict)             \
  _(Future)                  \
  _(Await)                   \
  _(Device)                  \
  _(Stream)                  \
  _(Object)                  \
  _(PyObject)                \
  _(Uninitialized)           \
  _(Capsule)                 \
  _(RRef)                    \
  _(Quantizer)               \
  _(Generator)               \
  _(Enum)

// [doxygen private]
// 这些方法实际上并不是私有的，但我们不希望对它们进行文档化，
// 因此它们被标记为 `@private`，在这个页面的 doxygen 文档中被隐藏起来。

/// IValue（Interpreter Value）是类型标签联合体，用于描述不同值类型。
//
/// Definition of the IValue struct, which represents a value in TorchScript.
/// This struct is used to encapsulate various types including primitive types
/// (int64_t, bool, double, Device), Tensor objects, and other types using
/// c10::intrusive_ptr.
///
/// The IValue struct supports efficient memory management by ensuring that
/// the destructor and related operations handle both Tensor and
/// c10::intrusive_ptr paths consistently. A null c10::intrusive_ptr is
/// represented by UndefinedTensorImpl::singleton() rather than nullptr.
///
/// IValues are crucial as inputs to and outputs from the TorchScript interpreter.
/// To extract the contained value from an IValue, utilize the `.toX()` methods,
/// where `X` denotes the type you intend to retrieve. It's important to note
/// that these methods do not perform any casting; they merely unwrap the
/// contained value.
///
/// Example usage:
///
/// \rst
/// .. code-block:: cpp
///
///   // Create an IValue holding an integer
///   torch::IValue my_ivalue(26);
///   std::cout << my_ivalue << "\n";
///
///   // Unwrap the integer from the IValue
///   int64_t my_int = my_ivalue.toInt();
///   std::cout << my_int << "\n";
///
///   // Attempting to convert to a Tensor will raise an error
///   // since my_ivalue is tagged as an int and cannot be interpreted as a Tensor
///   torch::Tensor my_tensor = my_ivalue.toTensor();
/// \endrst
struct TORCH_API IValue final {
  /// Copy constructor for IValue, initializes from another IValue instance.
  IValue(const IValue& rhs) : IValue(rhs.payload, rhs.tag) {
    // Increase reference count for intrusive_ptr if it's not nullptr
    if (isIntrusivePtr() &&
        payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton()) {
      c10::raw::intrusive_ptr::incref(payload.u.as_intrusive_ptr);
    }
  }

  /// Move constructor for IValue, noexcept to ensure safe move semantics.
  IValue(IValue&& rhs) noexcept : tag(rhs.tag) {
    moveFrom(std::move(rhs));
  }

  /// Destructor for IValue, ensures proper cleanup of resources.
  ~IValue() {
    destroy();
  }

  /// Move assignment operator for IValue.
  C10_ALWAYS_INLINE IValue& operator=(IValue&& rhs) & noexcept {
    if (&rhs == this) {
      return *this;
    }

    // Clean up existing resources
    destroy();
    // Move resources from rhs
    moveFrom(std::move(rhs));
    return *this;
  }

  /// Copy assignment operator for IValue.
  IValue& operator=(IValue const& rhs) & {
    // Delegate to move assignment operator using copy constructor
    *this = IValue(rhs);
    return *this;
  }
    return *this;
  }

  void dump() const;

  /**
   * Equality comparison. The semantics are the same as Python's `==`:
   * 1. Numerical types are compared by value.
   * 2. Tensors compute element-wise equality, returning a BoolTensor (see:
   * `torch.eq()`)
   * 3. Strings are compared by value.
   * 4. Sequence types (list, tuple) are compared lexicographically by
   *    comparing their elements. Different sequence types never compare equal.
   * 5. Mappings (dict) must have equal (key, value) pairs.
   * 6. If not listed above, the default behavior for is to test identity
   * equality (e.g. pointer equality).
   *
   * Why does this return an IValue instead of a bool? Because in PyTorch,
   * `tensor1 == tensor2` returns a `BoolTensor`, not a bool.
   *
   * NOTE: we (like Python) assume that identity equality implies value equality
   * for efficiency.
   * TODO: need to support customizing equality
   */
  // 定义了equals方法，用于比较两个IValue对象的相等性，返回结果为IValue对象
  IValue equals(const IValue& rhs) const;
  
  /**
   * This implements the same semantics as `bool(lhs == rhs)` in Python. which
   * is the same as `equals()` except for Tensor types.
   */
  // 定义了IValue对象之间的等于运算符重载函数，实现了与equals相同的语义，但对于Tensor类型有所区别
  TORCH_API friend bool operator==(const IValue& lhs, const IValue& rhs);
  TORCH_API friend bool operator!=(const IValue& lhs, const IValue& rhs);

  /**
   * Identity comparison. Checks if `this` is the same object as `rhs`. The
   * semantics are the same as Python's `is` operator.
   *
   * NOTE: Like in Python, this operation is poorly defined for primitive types
   * like numbers and strings. Prefer to use `==` unless you really want to
   * check identity equality.
   */
  // 检查当前对象是否与rhs相同的身份。语义与Python的`is`运算符相同。
  // 注意：与Python类似，对于数字和字符串等基本类型，此操作定义不明确。建议使用`==`除非真的需要检查身份相等性。
  bool is(const IValue& rhs) const;

  /**
   * Hashing for IValues. Returns an IValue-boxed int.
   *
   * Some notes:
   * - Like eager, Tensors are hashed by looking at the pointer. This is not
   *   strictly correct because two value-equal tensors with different tensor
   *   pointers will hash differently, but we choose to reproduce the eager
   *   semantics.
   * - Hashing is not defined on all built-in IValue types (e.g. list and
   *   dict), following Python. Calling `hash()` on these types will throw.
   */
  // 计算IValue的哈希值，返回一个IValue对象封装的整数。
  // 一些注意事项：
  // - 类似eager模式，张量通过查看指针来进行哈希。这不是严格正确的，因为两个值相等的张量具有不同的张量指针会产生不同的哈希值，但我们选择复制eager的语义。
  // - 并非所有内置的IValue类型都定义了哈希（例如列表和字典），遵循Python的行为。在这些类型上调用`hash()`会抛出异常。
  IValue hash() const {
    return (int64_t)IValue::hash(*this);
  }
  // 这里定义是因为 `c10::hash` 调度到了这样一个函数签名。详见成员函数 `hash()`。
  static size_t hash(const IValue& iv);

  /**
   * @private [doxygen private]
   * [container equality]
   * 这是一个相等性实现，假设具有相同标识的对象相等，出于效率考虑。
   * 我们主要出于一致性而这样做，因为 Python 也是这样。由于 torch 的特殊情况，
   * 这实际上会引发用户可见的行为变化：
   *      [tensor1] == [tensor1] -> True（因为容器相等性首先比较标识）
   *      [tensor1] == [tensor1_copy] -> RuntimeError:
   * Boolean value of Tensor with more than one value is ambiguous
   */
  TORCH_API friend bool _fastEqualsForContainer(
      const IValue& lhs,
      const IValue& rhs);

 private:
  static bool isAliasOf(const at::Tensor& a, const at::Tensor& b) {
    if (a.is_sparse()) {
      return isAliasOf(a._values(), b) || isAliasOf(a._indices(), b);
    }
    if (b.is_sparse()) {
      return isAliasOf(a, b._values()) || isAliasOf(a, b._indices());
    }
    if (a.is_sparse_csr()) {
      return isAliasOf(a.values(), b) || isAliasOf(a.crow_indices(), b) ||
          isAliasOf(a.col_indices(), b);
    }
    if (b.is_sparse_csr()) {
      return isAliasOf(a, b.values()) || isAliasOf(a, b.crow_indices()) ||
          isAliasOf(a, b.col_indices());
    }

    // 不透明张量（例如由 MKL-DNN 后端构造的张量）没有存储，因此我们只比较它们的 TensorImpl。
    // TODO: 找到一种方法来暴露不透明张量的别名信息。
    if (!a.has_storage() || !b.has_storage()) {
      return a.unsafeGetTensorImpl() == b.unsafeGetTensorImpl();
    }

    return a.is_alias_of(b);
  }

  template <typename T>
  bool isListOf() const;

 public:
  /// @private [doxygen private]
  bool isAliasOf(const IValue& rhs) const {
    if (this->tag != rhs.tag) {
      // 类型不同时显然不会有别名
      return false;
    }

    // 张量应该基于内部存储进行比较
    if (this->isTensor()) {
      return isAliasOf(this->toTensor(), rhs.toTensor());
    }

    if (!isIntrusivePtr()) {
      // 原始类型不会有别名
      return false;
    }

    AT_ASSERT(rhs.isIntrusivePtr());

    // 其他类型可以通过它们的指针值进行比较
    return this->payload.u.as_intrusive_ptr == rhs.payload.u.as_intrusive_ptr;
  }

  /// @private [doxygen private]
  size_t use_count() const noexcept {
    if (isTensor()) {
      return payload.as_tensor.use_count();
    }

    if (!isIntrusivePtrLegacyBehavior()) {
      return 1;
    }

    if (payload.u.as_intrusive_ptr == c10::UndefinedTensorImpl::singleton()) {
      return 0;
    }
    return c10::raw::intrusive_ptr::use_count(payload.u.as_intrusive_ptr);
  }

  /// @private [doxygen private]
  void swap(IValue& rhs) noexcept {
    // 如果当前对象和 rhs 都是 Tensor 类型
    if (isTensor() && rhs.isTensor()) {
      // 交换当前对象和 rhs 的 Tensor 数据
      std::swap(payload.as_tensor, rhs.payload.as_tensor);
    } else if (isTensor()) {
      // 如果当前对象是 Tensor 类型而 rhs 不是
      // 移动当前对象的 Tensor 数据到临时变量 t
      at::Tensor t = std::move(payload.as_tensor);
      // 从性能角度考虑，这里省略了通常的显式析构函数调用
      // 这并不会导致未定义行为，因为被移动后的 Tensor 是处于空状态的 intrusive_ptr
      // 不需要显式调用析构函数确保正确性，这里保留注释说明这一点
      //
      // payload.as_tensor.~Tensor();
      // 将 rhs 的数据赋值给当前对象
      payload.u = rhs.payload.u;
      // 使用 t 构造新的 Tensor 对象，放置在 rhs 的位置
      new (&rhs.payload.as_tensor) at::Tensor(std::move(t));
    } else if (rhs.isTensor()) {
      // 如果 rhs 是 Tensor 类型而当前对象不是，直接交换对象内容
      rhs.swap(*this);
      return;
    } else {
      // 如果当前对象和 rhs 都不是 Tensor 类型，交换它们的 payload.u 数据
      std::swap(payload.u, rhs.payload.u);
    }
    // 最后交换当前对象的 tag 和 rhs 的 tag
    std::swap(tag, rhs.tag);
  }

  // Accessors for subtypes are arranged together below
  // While some of these accessors could be generated through templates,
  // we prefer to write them manually for clarity

  // 接下来是各种子类型的访问器函数，为了清晰起见手动编写而非使用模板生成

  // 使用 at::TensorBase 类型构造函数，初始化为 Tensor 类型
  IValue(at::TensorBase t) : tag(Tag::Tensor) {
    new (&payload.as_tensor) at::Tensor(std::move(t));
  }

  // 返回当前对象是否为 Tensor 类型
  bool isTensor() const {
    return Tag::Tensor == tag;
  }

 private:
  // Outlined error path so that toTensor() can be inlined.
  // 内联错误处理路径，以便 toTensor() 可以内联展开

  // 声明不会返回的函数标记，用于指示编译器该函数不会正常返回
  [[noreturn]] void reportToTensorTypeError() const;

 public:
  // 移动语义版本的 toTensor()，返回一个 Tensor
  at::Tensor toTensor() &&;
  // 左值引用版本的 toTensor()，返回当前对象的 Tensor 引用
  at::Tensor& toTensor() &;
  // 常量左值引用版本的 toTensor()，返回当前对象的 Tensor 常量引用
  const at::Tensor& toTensor() const&;

  // 返回当前对象的 TensorImpl 指针，如果当前对象不是 Tensor 报错
  at::TensorImpl* unsafeToTensorImpl() const {
    TORCH_INTERNAL_ASSERT(isTensor());
    return payload.as_tensor.unsafeGetTensorImpl();
  }

  // 使用 at::Storage 类型构造函数，初始化为 Storage 类型
  IValue(at::Storage s) : tag(Tag::Storage) {
    // 释放 StorageImpl 并转为对应的 u.as_intrusive_ptr
    payload.u.as_intrusive_ptr =
        null_to_undefined_tensor(s.unsafeReleaseStorageImpl());
  }

  // 返回当前对象是否为 Storage 类型
  bool isStorage() const {
    return Tag::Storage == tag;
  }

  // 移动语义版本的 toStorage()，返回一个 Storage
  c10::Storage toStorage() &&;
  // 常量左值引用版本的 toStorage()，返回当前对象的 Storage 常量引用
  c10::Storage toStorage() const&;

  // 返回当前对象自身的常量引用
  const IValue& toIValue() const {
    return *this;
  }
  // 返回当前对象自身的引用
  IValue& toIValue() {
    return *this;
  }

  /// @private [doxygen private]
  // 使用 intrusive_ptr<caffe2::Blob> 类型构造函数，初始化为 Blob 类型
  IValue(intrusive_ptr<caffe2::Blob> blob) : tag(Tag::Blob) {
    // TODO (after Tensor merge) If we pass in a Blob holding a Tensor, extract
    // and store it as a Tensor instead.
    // 释放 Blob 并转为对应的 u.as_intrusive_ptr
    payload.u.as_intrusive_ptr = null_to_undefined_tensor(blob.release());
  }

  /// @private [doxygen private]
  // 返回当前对象是否为 Blob 类型
  bool isBlob() const {
    return Tag::Blob == tag;
  }

  /// @private [doxygen private]
  // 移动语义版本的 toBlob()，返回一个 Blob
  c10::intrusive_ptr<caffe2::Blob> toBlob() &&;

  /// @private [doxygen private]
  // 常量左值引用版本的 toBlob()，返回当前对象的 Blob 常量引用
  c10::intrusive_ptr<caffe2::Blob> toBlob() const&;

  // Capsule. No new callsites of these APIs should
  // be introduced.
  // Capsule 类型相关的访问器函数

  // 创建一个 Capsule 类型的 IValue 对象
  static inline IValue make_capsule(
      intrusive_ptr<torch::CustomClassHolder> blob);
  // 返回当前对象是否为 Capsule 类型
  bool isCapsule() const {
  // 检查标签是否为 Tag::Capsule
  return Tag::Capsule == tag;
}

// 移动语义版本，将当前对象转换为 Capsule
c10::intrusive_ptr<torch::CustomClassHolder> toCapsule() &&;

// 常量引用版本，返回一个 Capsule 对象
c10::intrusive_ptr<torch::CustomClassHolder> toCapsule() const&;

// Custom C++ classes

// 接受继承自 torch::CustomClassHolder 的自定义类的指针
template <
    typename T,
    std::enable_if_t<std::is_base_of_v<torch::CustomClassHolder, T>, int> = 0>
IValue(intrusive_ptr<T> custom_class);

// 检查当前对象是否为自定义类
bool isCustomClass() const;

// 移动语义版本，将当前对象转换为指定类型的自定义类对象
template <typename T>
c10::intrusive_ptr<T> toCustomClass() &&;

// 常量引用版本，返回一个指定类型的自定义类对象
template <typename T>
c10::intrusive_ptr<T> toCustomClass() const&;

// Tuple

// 接受 ivalue::Tuple 类型的指针作为参数
IValue(c10::intrusive_ptr<ivalue::Tuple> v);

// 接受 std::tuple 类型的参数，条件为非左值引用且可构造为 IValue
template <
    typename... Args,
    std::enable_if_t<
        !std::disjunction_v<
            std::is_lvalue_reference<Args>...,
            std::negation<std::is_constructible<IValue, Args>>...>,
        std::nullptr_t> = nullptr>
IValue(const std::tuple<Args...>& t);

// 接受 std::tuple 类型的参数，条件为非左值引用且可构造为 IValue
template <
    typename... Args,
    std::enable_if_t<
        !std::disjunction_v<
            std::is_lvalue_reference<Args>...,
            std::negation<std::is_constructible<IValue, Args>>...>,
        std::nullptr_t> = nullptr>
IValue(std::tuple<Args...>&& t);

// 检查当前对象是否为 Tuple
bool isTuple() const {
  return Tag::Tuple == tag;
}

// 移动语义版本，将当前对象转换为 Tuple
c10::intrusive_ptr<ivalue::Tuple> toTuple() &&;

// 常量引用版本，返回一个 Tuple 对象
c10::intrusive_ptr<ivalue::Tuple> toTuple() const&;

// 返回 Tuple 的非丢弃引用
C10_NODISCARD ivalue::Tuple& toTupleRef() const;

// Double

// 接受 double 类型的参数，初始化为 Double 标签
IValue(double d) : tag(Tag::Double) {
  payload.u.as_double = d;
}

// 检查当前对象是否为 Double
bool isDouble() const {
  return Tag::Double == tag;
}

// 返回当前对象的 double 值
double toDouble() const {
  if (isDouble()) {
    return payload.u.as_double;
  } else if (isSymFloat()) {
    return toSymFloat().guard_float(__FILE__, __LINE__);
  } else {
    TORCH_INTERNAL_ASSERT(0, "expected double");
  }
}

// ComplexDouble

// 接受 c10::complex<T> 类型的参数
template <typename T>
IValue(c10::complex<T> c);

// 检查当前对象是否为 ComplexDouble
bool isComplexDouble() const {
  return Tag::ComplexDouble == tag;
}

// 返回当前对象转换为 c10::complex<double> 类型
c10::complex<double> toComplexDouble() const;

// Future

// 接受 ivalue::Future 类型的指针作为参数
IValue(c10::intrusive_ptr<ivalue::Future> v);

// 检查当前对象是否为 Future
bool isFuture() const {
  return Tag::Future == tag;
}

// 移动语义版本，将当前对象转换为 Future
c10::intrusive_ptr<ivalue::Future> toFuture() &&;

// 常量引用版本，返回一个 Future 对象
c10::intrusive_ptr<ivalue::Future> toFuture() const&;

// 接受 ivalue::Await 类型的指针作为参数
IValue(c10::intrusive_ptr<ivalue::Await> v);

// 检查当前对象是否为 Await
bool isAwait() const {
  return Tag::Await == tag;
}

// 移动语义版本，将当前对象转换为 Await
c10::intrusive_ptr<ivalue::Await> toAwait() &&;

// 常量引用版本，返回一个 Await 对象
c10::intrusive_ptr<ivalue::Await> toAwait() const&;

// RRef

// 接受 c10::intrusive_ptr<c10::RRefInterface> 类型的参数
IValue(c10::intrusive_ptr<c10::RRefInterface> v);

// 检查当前对象是否为 RRef
bool isRRef() const {
  return Tag::RRef == tag;
}

// 移动语义版本，将当前对象转换为 RRef
c10::intrusive_ptr<c10::RRefInterface> toRRef() &&;

// 常量引用版本，返回一个 RRef 对象
c10::intrusive_ptr<c10::RRefInterface> toRRef() const&;

// Quantizer

// 接受 c10::intrusive_ptr<at::Quantizer> 类型的参数
IValue(c10::intrusive_ptr<at::Quantizer> v);

// 检查当前对象是否为 Quantizer
bool isQuantizer() const {
  return Tag::Quantizer == tag;
}

// 移动语义版本，将当前对象转换为 Quantizer
c10::intrusive_ptr<at::Quantizer> toQuantizer() &&;

// 常量引用版本，返回一个 Quantizer 对象
c10::intrusive_ptr<at::Quantizer> toQuantizer() const&;

// Int

// 接受 int64_t 类型的参数，初始化为 Int 标签
IValue(int64_t i) : tag(Tag::Int) {
  payload.u.as_int = i;
}

// 接受 c10::SymInt 类型的参数
IValue(const c10::SymInt& i) {
    // 如果 `i` 可以转换为整数，则设置标签为 Tag::Int，并将整数值存储在 payload 中
    if (auto mi = i.maybe_as_int()) {
      tag = Tag::Int;
      payload.u.as_int = *mi;
    } else {
      // 否则设置标签为 Tag::SymInt，并将符号整数节点释放后的指针存储在 payload 中
      tag = Tag::SymInt;
      payload.u.as_intrusive_ptr = i.toSymNode().release();
    }
  }

  // 检查当前对象是否为符号整数类型
  bool isSymInt() const {
    return Tag::SymInt == tag;
  }

  // 移动语义的符号整数转换函数声明
  c10::SymInt toSymInt() &&;
  // 常量引用语义的符号整数转换函数声明
  c10::SymInt toSymInt() const&;

  // 构造函数：根据符号浮点数 `i` 初始化 IValue 对象
  IValue(const c10::SymFloat& i) {
    // 如果 `i` 是符号类型，则设置标签为 Tag::SymFloat，并存储符号节点释放后的指针
    if (i.is_symbolic()) {
      tag = Tag::SymFloat;
      payload.u.as_intrusive_ptr = i.toSymNodeImpl().release();
    } else {
      // 否则设置标签为 Tag::Double，并存储浮点数值在 payload 中
      tag = Tag::Double;
      payload.u.as_double = i.as_float_unchecked();
    }
  }

  // 检查当前对象是否为符号浮点数类型
  bool isSymFloat() const {
    return Tag::SymFloat == tag;
  }

  // 移动语义的符号浮点数转换函数声明
  c10::SymFloat toSymFloat() &&;
  // 常量引用语义的符号浮点数转换函数声明
  c10::SymFloat toSymFloat() const&;

  // 构造函数：根据符号布尔值 `i` 初始化 IValue 对象
  IValue(const c10::SymBool& i) {
    // 如果 `i` 可以转换为布尔值，则设置标签为 Tag::Bool，并存储布尔值在 payload 中
    if (auto mi = i.maybe_as_bool()) {
      tag = Tag::Bool;
      payload.u.as_int = *mi;
    } else {
      // 否则设置标签为 Tag::SymBool，并存储符号布尔节点释放后的指针
      tag = Tag::SymBool;
      payload.u.as_intrusive_ptr = i.toSymNodeImpl().release();
    }
  }

  // 检查当前对象是否为符号布尔类型
  bool isSymBool() const {
    return Tag::SymBool == tag;
  }

  // 移动语义的符号布尔转换函数声明
  c10::SymBool toSymBool() &&;
  // 常量引用语义的符号布尔转换函数声明
  c10::SymBool toSymBool() const&;

  // 构造函数：允许通过整数字面量初始化 IValue 对象，避免歧义
  IValue(int32_t i) : IValue(static_cast<int64_t>(i)) {}

  // 检查当前对象是否为整数类型
  bool isInt() const {
    return Tag::Int == tag;
  }

  // 获取当前对象的整数值
  int64_t toInt() const {
    if (isInt()) {
      // 如果是整数类型，直接返回整数值
      return payload.u.as_int;
    } else if (isSymInt()) {
      // 如果是符号整数类型，调用符号整数对象的 guard_int 方法获取整数值
      return toSymInt().guard_int(__FILE__, __LINE__);
    } else {
      // 如果既不是整数也不是符号整数类型，抛出断言错误
      TORCH_INTERNAL_ASSERT(0, "expected int");
    }
  }

  // 布尔类型构造函数：根据布尔值 `b` 初始化 IValue 对象，设置标签为 Tag::Bool
  IValue(bool b) : tag(Tag::Bool) {
#if defined(__clang__) && defined(__x86_64__)
    // 如果编译器是 Clang 并且目标平台是 x86_64
    // 初始化整个 payload 可以阻止 Valgrind 报告在 IValue 拷贝构造函数中的
    // "jump or move depends on uninitialised value" 错误
    // 参见 https://github.com/pytorch/pytorch/issues/37117
    payload.u.as_int = b;
#else
    // 否则，初始化 payload 的布尔值字段
    payload.u.as_bool = b;
#endif
  }
  // 检查当前 IValue 是否为布尔类型
  bool isBool() const {
    return Tag::Bool == tag;
  }
  // 获取当前 IValue 的布尔值
  bool toBool() const {
    if (isBool()) {
      return payload.u.as_bool; // 如果是布尔类型，直接返回布尔值
    } else if (isSymBool()) {
      return toSymBool().guard_bool(__FILE__, __LINE__); // 如果是符号布尔类型，调用其方法获取布尔值
    } else {
      TORCH_INTERNAL_ASSERT(0, "expected bool"); // 否则，报告错误，预期的类型是布尔值
    }
  }

  // IntList
  bool isIntList() const; // 检查当前 IValue 是否为整数列表类型
  bool isSymIntList() const; // 检查当前 IValue 是否为符号整数列表类型
  c10::List<int64_t> toIntList() &&; // 获取当前 IValue 的整数列表（移动语义版本）
  c10::List<int64_t> toIntList() const&; // 获取当前 IValue 的整数列表（常量引用版本）
  std::vector<int64_t> toIntVector() const; // 获取当前 IValue 的整数向量
  at::DimVector toDimVector() const; // 获取当前 IValue 的维度向量

  // ConstantString
  IValue(c10::intrusive_ptr<ivalue::ConstantString> v); // 构造函数，用常量字符串创建 IValue
  IValue(std::string v); // 构造函数，用字符串创建 IValue
  IValue(const char* v) : IValue(std::string(v)) {} // 构造函数，用 C 字符串创建 IValue
  IValue(c10::string_view v) : IValue(std::string(v)){}; // 构造函数，用 string_view 创建 IValue
  // 检查当前 IValue 是否为字符串类型
  bool isString() const {
    return Tag::String == tag;
  }
  // 获取当前 IValue 的字符串（移动语义版本）
  c10::intrusive_ptr<ivalue::ConstantString> toString() &&;
  // 获取当前 IValue 的字符串（常量引用版本）
  c10::intrusive_ptr<ivalue::ConstantString> toString() const&;
  // 获取当前 IValue 的字符串引用
  const std::string& toStringRef() const;
  // 获取当前 IValue 的可选字符串引用
  std::optional<std::reference_wrapper<const std::string>> toOptionalStringRef() const;
  // 获取当前 IValue 的字符串视图
  c10::string_view toStringView() const;

  // DoubleList
  bool isDoubleList() const; // 检查当前 IValue 是否为双精度浮点数列表类型
  c10::List<double> toDoubleList() &&; // 获取当前 IValue 的双精度浮点数列表（移动语义版本）
  c10::List<double> toDoubleList() const&; // 获取当前 IValue 的双精度浮点数列表（常量引用版本）
  std::vector<double> toDoubleVector() const; // 获取当前 IValue 的双精度浮点数向量

  // ComplexDoubleList
  bool isComplexDoubleList() const; // 检查当前 IValue 是否为复数双精度浮点数列表类型
  c10::List<c10::complex<double>> toComplexDoubleList() &&; // 获取当前 IValue 的复数双精度浮点数列表（移动语义版本）
  c10::List<c10::complex<double>> toComplexDoubleList() const&; // 获取当前 IValue 的复数双精度浮点数列表（常量引用版本）
  std::vector<c10::complex<double>> toComplexDoubleVector() const; // 获取当前 IValue 的复数双精度浮点数向量

  // BoolList
  bool isBoolList() const; // 检查当前 IValue 是否为布尔列表类型
  c10::List<bool> toBoolList() &&; // 获取当前 IValue 的布尔列表（移动语义版本）
  c10::List<bool> toBoolList() const&; // 获取当前 IValue 的布尔列表（常量引用版本）

  // TensorList
  bool isTensorList() const; // 检查当前 IValue 是否为张量列表类型
  c10::List<at::Tensor> toTensorList() &&; // 获取当前 IValue 的张量列表（移动语义版本）
  c10::List<at::Tensor> toTensorList() const&; // 获取当前 IValue 的张量列表（常量引用版本）
  std::vector<at::Tensor> toTensorVector() const; // 获取当前 IValue 的张量向量

  // OptionalTensorList
  bool isOptionalTensorList() const; // 检查当前 IValue 是否为可选张量列表类型
  c10::List<std::optional<at::Tensor>> toOptionalTensorList() &&; // 获取当前 IValue 的可选张量列表（移动语义版本）
  c10::List<std::optional<at::Tensor>> toOptionalTensorList() const&; // 获取当前 IValue 的可选张量列表（常量引用版本）
  std::vector<std::optional<at::Tensor>> toOptionalTensorVector() const; // 获取当前 IValue 的可选张量向量

  // GenericList
  IValue(c10::List<IValue> v); // 构造函数，用 IValue 列表创建 IValue
  // 检查当前 IValue 是否为列表类型
  bool isList() const {
    // 检查 Tag::GenericList 是否等于当前标签，返回布尔结果
    return Tag::GenericList == tag;
    }
    
    // 将当前对象转换为右值引用的列表类型
    c10::List<IValue> toList() &&;
    
    // 将当前对象转换为常量左值引用的列表类型
    c10::List<IValue> toList() const&;
    
    // 将当前对象转换为数组引用的列表类型
    c10::ArrayRef<IValue> toListRef() const;
    
    // 模板构造函数，用于 IValue，递归调用另一个构造函数
    // 通过 SFINAE 检查调用的构造函数是否存在
    template <class T>
    using enable_if_ivalue_constructible =
        std::enable_if_t<std::is_constructible_v<IValue, T>, std::nullptr_t>;
    
    // 对列表的规则更为复杂；通用构造函数只有在元素不是 SymInt 时才接受
    // 如果有 SymInt 元素，则必须在构造时检查是否可以将列表衰减为 int 列表
    // 对于 SymIntArrayRef，这是强制性的，因为在使用场景中，我们可能期望 toIntList
    // 能够正常工作，即使在调用点处有 SymIntArrayRef 参数
    // 实际上，只有 SymIntArrayRef 以这种方式使用，因此我们没有费力使其适用于其他构造函数，
    // 我们只确保它们不可选。
    template <class T>
    using enable_if_list_is_ivalue_constructible = std::enable_if_t<
        std::is_constructible_v<IValue, T> && !std::is_same_v<T, c10::SymInt>,
        std::nullptr_t>;
    
    // 接受右值引用的列表类型参数并构造为 IValue 对象
    template <class T, enable_if_list_is_ivalue_constructible<T> = nullptr>
    IValue(c10::List<T>&& v);
    
    // 接受常量左值引用的列表类型参数并构造为 IValue 对象
    template <class T, enable_if_list_is_ivalue_constructible<T> = nullptr>
    IValue(const c10::List<T>& v);
    
    // 接受数组引用类型参数并构造为 IValue 对象
    template <class T, enable_if_list_is_ivalue_constructible<T> = nullptr>
    IValue(at::ArrayRef<T> v);
    
    // 接受常量引用的 std::vector 类型参数并构造为 IValue 对象
    template <class T, enable_if_list_is_ivalue_constructible<T> = nullptr>
    IValue(const std::vector<T>& v);
    
    // 接受右值引用的 std::vector 类型参数并构造为 IValue 对象
    template <class T, enable_if_list_is_ivalue_constructible<T> = nullptr>
    IValue(std::vector<T>&& v);
    
    // 接受固定大小数组类型参数并构造为 IValue 对象
    template <class T, size_t N>
    IValue(std::array<T, N> v);
    
    // 手动构造 SymInt 类型的列表，如果可能，将其衰减为 int 列表
    // 为避免歧义的重载情况，使用模板来阻止隐式转换
    template <class T>
    using enable_if_symint =
        std::enable_if_t<std::is_same_v<T, c10::SymInt>, std::nullptr_t>;
    
    // 接受数组引用类型参数并构造为 IValue 对象，仅适用于 SymInt 类型
    template <class T, enable_if_symint<T> = nullptr>
    IValue(at::ArrayRef<T> v);
    
    // 接受可选数组引用类型参数并构造为 IValue 对象，仅适用于 SymInt 类型
    template <class T, enable_if_symint<T> = nullptr>
    IValue(at::OptionalArrayRef<T> v);
    
    // 接受常量引用的 std::vector 类型参数并构造为 IValue 对象，仅适用于 SymInt 类型
    template <class T, enable_if_symint<T> = nullptr>
    IValue(const std::vector<T>& v);
    
    // 接受右值引用的 std::vector 类型参数并构造为 IValue 对象，仅适用于 SymInt 类型
    template <class T, enable_if_symint<T> = nullptr>
    IValue(std::vector<T>&& v);
    
    // 对于列表类型 T，满足 IValue 构造的条件，同时也满足转换为 IListRef<T>::boxed_type 类型的条件
    // 并且不是 SymInt 类型
    template <class T>
    using enable_if_ilist_is_ivalue_constructible = std::enable_if_t<
        std::is_constructible_v<IValue, T> &&
            std::is_constructible_v<IValue, typename IListRef<T>::boxed_type> &&
            !std::is_same_v<T, c10::SymInt>,
        std::nullptr_t>;
    
    // 接受 IListRef 类型参数并构造为 IValue 对象
    template <class T, enable_if_ilist_is_ivalue_constructible<T> = nullptr>
    IValue(c10::IListRef<T> v);
    
    // 接受 c10::Dict<IValue, IValue> 类型参数并构造为 IValue 对象，用于通用字典类型
    IValue(c10::Dict<IValue, IValue> v);
    
    // 检查当前对象是否为通用字典类型，返回布尔结果
    bool isGenericDict() const {
  // 检查标签是否为 GenericDict，返回布尔值
  return Tag::GenericDict == tag;
}

// 移动语义方法，将当前对象转换为 c10::Dict<IValue, IValue>
c10::Dict<IValue, IValue> toGenericDict() &&;

// 常量引用方法，将当前对象转换为 c10::Dict<IValue, IValue>
c10::Dict<IValue, IValue> toGenericDict() const&;

// 根据给定的 c10::Dict<Key, Value> 对象构造 IValue
template <class Key, class Value>
IValue(c10::Dict<Key, Value> v);

// 根据 std::unordered_map<Key, Value> 构造 IValue
template <class Key, class Value>
/// \cond
/// DOXYGEN_CANNOT_HANDLE_CONSTRUCTORS_WITH_MACROS_SO_EXCLUDE_THIS_LINE_FROM_DOXYGEN
C10_DEPRECATED_MESSAGE(
    "IValues based on std::unordered_map<K, V> are slow and deprecated. Please use c10::Dict<K, V> instead.")
/// \endcond
IValue(std::unordered_map<Key, Value> v);

// 根据 std::optional<T> 构造 IValue，要求 T 可以通过 IValue 构造
template <class T, enable_if_ivalue_constructible<T> = nullptr>
IValue(std::optional<T> v);

// 根据 c10::OptionalArrayRef<T> 构造 IValue，要求 T 可以通过 IValue 构造
template <class T, enable_if_list_is_ivalue_constructible<T> = nullptr>
IValue(c10::OptionalArrayRef<T> v);

// 构造一个空值的 IValue
IValue(c10::nullopt_t);

// 根据 c10::intrusive_ptr<ivalue::Object> 构造 IValue
IValue(c10::intrusive_ptr<ivalue::Object> v);

// 检查标签是否为 Object，返回布尔值
bool isObject() const {
  return tag == Tag::Object;
}

// 移动语义方法，将当前对象转换为 c10::intrusive_ptr<ivalue::Object>
c10::intrusive_ptr<ivalue::Object> toObject() &&;

// 常量引用方法，将当前对象转换为 c10::intrusive_ptr<ivalue::Object>
c10::intrusive_ptr<ivalue::Object> toObject() const&;

// 返回当前对象的 ivalue::Object 引用
ivalue::Object& toObjectRef() const;

// 将当前对象转换为 torch::jit::Module
torch::jit::Module toModule() const;

// 检查标签是否为 Module，返回布尔值
bool isModule() const;

// 根据 c10::intrusive_ptr<ivalue::PyObjectHolder> 构造 IValue
IValue(c10::intrusive_ptr<ivalue::PyObjectHolder> v);

// 检查标签是否为 PyObject，返回布尔值
bool isPyObject() const {
  return tag == Tag::PyObject;
}

// 移动语义方法，将当前对象转换为 c10::intrusive_ptr<ivalue::PyObjectHolder>
c10::intrusive_ptr<ivalue::PyObjectHolder> toPyObjectHolder() &&;

// 常量引用方法，将当前对象转换为 c10::intrusive_ptr<ivalue::PyObjectHolder>
c10::intrusive_ptr<ivalue::PyObjectHolder> toPyObjectHolder() const&;

// 返回当前对象的 PyObject 指针
PyObject* toPyObject() const;

// 根据 c10::intrusive_ptr<ivalue::EnumHolder> 构造 IValue，强制显式
explicit IValue(c10::intrusive_ptr<ivalue::EnumHolder> v);

// 检查标签是否为 Enum，返回布尔值
bool isEnum() const {
  return tag == Tag::Enum;
}

// 移动语义方法，将当前对象转换为 c10::intrusive_ptr<ivalue::EnumHolder>
c10::intrusive_ptr<ivalue::EnumHolder> toEnumHolder() &&;

// 常量引用方法，将当前对象转换为 c10::intrusive_ptr<ivalue::EnumHolder>
c10::intrusive_ptr<ivalue::EnumHolder> toEnumHolder() const&;

// 默认构造函数，构造一个空值的 IValue
IValue() = default;

// 检查标签是否为 None，返回布尔值
bool isNone() const {
  return Tag::None == tag;
}

// 返回当前对象表示的 None 的字符串形式
std::string toNone() const {
  AT_ASSERT(isNone());
  return "None";
}

// 静态方法，返回一个未初始化的 IValue 对象
static IValue uninitialized() {
  auto i = IValue();
  i.tag = Tag::Uninitialized;
  return i;
}

// 根据 at::Scalar 对象构造 IValue
// NB: 这里采用了委托构造函数的方式
IValue(const at::Scalar& s) : IValue() {
  // 根据 Scalar 的类型不同，设置对应的标签和数据
  if (s.isSymInt()) {
    tag = Tag::SymInt;
    payload.u.as_intrusive_ptr = s.toSymInt().toSymNode().release();
  } else if (s.isSymFloat()) {
    tag = Tag::SymFloat;
    payload.u.as_intrusive_ptr = s.toSymFloat().toSymNodeImpl().release();
  } else if (s.isSymBool()) {
    tag = Tag::SymBool;
    payload.u.as_intrusive_ptr = s.toSymBool().toSymNodeImpl().release();
  } else if (s.isFloatingPoint()) {
    tag = Tag::Double;
    payload.u.as_double = s.toDouble();
  } else if (s.isComplex()) {
    *this = s.toComplexDouble();
  } else if (s.isBoolean()) {
    tag = Tag::Bool;
    payload.u.as_bool = s.toBool();
  // 如果不属于已知类型，则抛出错误
  } else {
    // 在调试模式下，确保标量是整数类型
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        s.isIntegral(false), "Unknown type in Scalar");
    // 设置标签为整数类型
    tag = Tag::Int;
    // 将标量值转换为长整型并存储在负载的整数字段中
    payload.u.as_int = s.toLong();
  }

  // 判断当前值是否为标量
  bool isScalar() const {
    return isDouble() || isInt() || isComplexDouble() || isBool() ||
        isSymInt() || isSymFloat() || isSymBool();
  }

  // 将当前值转换为标量
  at::Scalar toScalar() const {
    if (isDouble())
      return toDouble();
    else if (isInt())
      return toInt();
    else if (isComplexDouble())
      return toComplexDouble();
    else if (isBool())
      return toBool();
    else if (isSymInt())
      return toSymInt();
    else if (isSymFloat())
      return toSymFloat();
    else if (isSymBool())
      return toSymBool();
    // 如果当前值不是标量，则抛出运行时错误
    throw std::runtime_error("IValue is not a Scalar");
  }

  // Device
  // 使用给定的设备类型和索引构造一个Device类型的IValue
  IValue(c10::Device d) : tag(Tag::Device) {
    payload.u.as_device.type = d.type();
    payload.u.as_device.index = d.index();
  }
  // 判断当前值是否为Device类型
  bool isDevice() const {
    return Tag::Device == tag;
  }
  // 将当前值转换为Device类型
  c10::Device toDevice() const {
    AT_ASSERT(isDevice());
    return c10::Device(payload.u.as_device.type, payload.u.as_device.index);
  }

  // Stream
  // 使用给定的Stream对象构造一个Stream类型的IValue
  IValue(c10::Stream s) : tag(Tag::Stream) {
    auto v = c10::make_intrusive<ivalue::StreamData3Holder>(s.pack3());
    payload.u.as_intrusive_ptr = v.release();
  }
  // 判断当前值是否为Stream类型
  bool isStream() const {
    return Tag::Stream == tag;
  }

  // ScalarType
  // 使用给定的标量类型构造一个ScalarType类型的IValue
  IValue(ScalarType t)
      : IValue(static_cast<std::underlying_type_t<ScalarType>>(t)) {}
  // 将当前值转换为ScalarType类型
  at::ScalarType toScalarType() const {
    return static_cast<at::ScalarType>(toInt());
  }

  // Layout
  // 使用给定的布局类型构造一个Layout类型的IValue
  IValue(Layout l) : IValue(static_cast<std::underlying_type_t<Layout>>(l)) {}
  // 将当前值转换为Layout类型
  at::Layout toLayout() const {
    return static_cast<at::Layout>(toInt());
  }

  // MemoryFormat
  // 使用给定的内存格式构造一个MemoryFormat类型的IValue
  IValue(MemoryFormat m)
      : IValue(static_cast<std::underlying_type_t<MemoryFormat>>(m)) {}
  // 将当前值转换为MemoryFormat类型
  at::MemoryFormat toMemoryFormat() const {
    return static_cast<at::MemoryFormat>(toInt());
  }

  // QScheme
  // 使用给定的量化方案构造一个整数类型的IValue
  IValue(at::QScheme qscheme) : tag(Tag::Int) {
    payload.u.as_int = static_cast<int64_t>(qscheme);
  }
  // 将当前值转换为QScheme类型
  at::QScheme toQScheme() const {
    return static_cast<at::QScheme>(toInt());
  }

  // Dimname
  // 使用给定的Dimname对象构造一个字符串类型的IValue
  IValue(at::Dimname dimname) : IValue(dimname.symbol().toQualString()) {}

  // 将当前值转换为Dimname类型
  at::Dimname toDimname() const {
    return at::Dimname::fromSymbol(Symbol::fromQualString(toStringRef()));
  }

  // Generator
  // 使用给定的Generator对象构造一个Generator类型的IValue
  IValue(at::Generator g) : tag(Tag::Generator) {
    payload.u.as_intrusive_ptr =
        null_to_undefined_tensor(g.unsafeReleaseGeneratorImpl());
  }
  // 判断当前值是否为Generator类型
  bool isGenerator() const {
    return Tag::Generator == tag;
  }
  // 将当前值转换为Generator类型
  at::Generator toGenerator() &&;
  at::Generator toGenerator() const&;

  // for debugging
  // 返回当前标签的类型名称，用于调试目的
  std::string tagKind() const {
    switch (tag) {
  // 定义宏DEFINE_CASE，用于生成每个标签的case语句，并返回其字符串表示
  #define DEFINE_CASE(x) \
    case Tag::x:         \
      return #x;
  
  // 使用TORCH_FORALL_TAGS宏展开DEFINE_CASE宏，生成一系列的case语句
  TORCH_FORALL_TAGS(DEFINE_CASE)
  
  // 清除之前定义的DEFINE_CASE宏，避免宏定义的污染
  #undef DEFINE_CASE
  
  // 如果传入的标签tag不在预定义的范围内，则返回一个指示无效标签的字符串
  }
  return "InvalidTag(" + std::to_string(static_cast<int>(tag)) + ")";
}

// 实现针对特定函数（如pop/push）中使用的通用的v.to<at::Tensor>()方法
// 使用模板元编程技术，尽可能优先使用直接命名的方法，因为它们更易于理解

// 注意：如果出现链接器错误指出某个方法缺失，
// 将其修改为 ... && = delete; 可以获得更好的错误信息
// 然而，由于某些编译器版本对此操作的支持不一致，因此无法在代码中永久使用该修改

template <typename T>
T to() &&;

template <typename T>
typename c10::detail::ivalue_to_const_ref_overload_return<T>::type to() const&;

// 将IValue转换为Optional对象，接受类型T和None
template <typename T>
optional<T> toOptional();

template <typename T>
optional<T> toOptional() const;

/// @private [doxygen private]
/// 浅层比较两个IValue对象，测试它们是否具有相同的对象标识
bool isSameIdentity(const IValue& rhs) const;

// 计算IValue的官方字符串表示
// 生成一个TorchScript表达式，可以用来重建具有相同值的IValue对象
// 在序列化器中打印常量时特别有用

// 调用者可以使用customFormatter函数来覆盖repr()打印IValue时的行为
// 如果你有某种环境可以查找值，并且想要打印指向该环境的引用（如序列化器的常量表）
// repr()并不一定在所有对象上都定义！

std::ostream& repr(
    std::ostream& stream,
    std::function<bool(std::ostream&, const IValue& v)> customFormatter)
    const;

// 计算IValue的非正式字符串表示
// 用于调试或类似print()函数的服务

// 与repr()不同之处在于，不保证能够从输出精确重建IValue对象
// 可以使用简洁/漂亮的形式打印

TORCH_API friend std::ostream& operator<<(std::ostream& out, const IValue& v);

// 如果当前IValue对象表示一个指针类型，则返回true
// 对于Tensor类型的对象，检查其payload中的定义情况
// 对于其他类型，则检查其是否具有遗留的内存管理行为

bool isPtrType() const {
  if (isTensor()) {
    return payload.as_tensor.defined();
  }
  return isIntrusivePtrLegacyBehavior();
}

/// @private [doxygen private]
// 返回当前IValue对象的内部指针
// 仅当对象表示一个指针类型时调用有效，否则抛出断言错误
const void* internalToPointer() const {
  TORCH_INTERNAL_ASSERT(
      isPtrType(), "Can only call internalToPointer() for pointer types");
  if (isTensor()) {
    return payload.as_tensor.unsafeGetTensorImpl();
  } else {
    return payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton()
        ? payload.u.as_intrusive_ptr
        : nullptr;
  }
}

template <typename T = c10::PlatformType>
TypePtr type() const;

// 检测是否为别名张量
struct HashAliasedIValue {
    // 定义一个哈希函数对象，用于计算给定 Tensor 对象的哈希值
    size_t hashTensor(const at::Tensor& ten) const {
      // 如果 Tensor 是稀疏张量
      if (ten.is_sparse()) {
        // COO 稀疏张量包含 "values" 和 "indices" 张量
        // 这段代码会检测共享 "values" 张量的稀疏张量的重叠情况，
        // 但不会检测共享 "indices" 张量的稀疏张量的重叠情况。
        return hashTensor(ten._values());
      } else if (ten.is_sparse_csr()) {
        // 如果是 CSR 稀疏张量
        // COO 稀疏张量包含 "values" 和 "indices" 张量
        // 这段代码会检测共享 "values" 张量的稀疏张量的重叠情况，
        // 但不会检测共享 "indices" 张量的稀疏张量的重叠情况。
        return hashTensor(ten.values());
      } else if (!ten.has_storage()) {
        // 不透明张量，例如由 MKL-DNN 后端构造的张量
        // 没有存储，因此直接使用它们的 TensorImpls。
        // TODO: 找到一种方法来暴露不透明张量的别名信息。
        return reinterpret_cast<size_t>(ten.unsafeGetTensorImpl());
      } else {
        // 普通张量，具有存储
        return reinterpret_cast<size_t>(ten.storage().unsafeGetStorageImpl());
      }
    }
    
    // 定义一个哈希函数对象，用于计算给定 IValue 对象的哈希值
    size_t operator()(const IValue& val) const {
      // 如果 IValue 是 Tensor 类型
      if (val.isTensor()) {
        // 调用 hashTensor 计算 Tensor 对象的哈希值
        return hashTensor(val.toTensor());
      }
      // 如果不是 Tensor 类型，则两个可变的 IValue 对象只有在它们是相同指针时才别名。
      return val.payload.u.as_int;
    }
  };

  // 定义一个比较函数对象，用于比较两个 IValue 对象是否是别名
  struct CompAliasedIValues {
    bool operator()(const IValue& lhs, const IValue& rhs) const {
      // 调用 IValue 的 isAliasOf 方法比较两个对象是否是别名
      return lhs.isAliasOf(rhs);
    }
  };

  // 使用哈希函数 HashAliasedIValue 和比较函数 CompAliasedIValues 定义 unordered_set 容器
  using HashAliasedIValues =
      std::unordered_set<IValue, HashAliasedIValue, CompAliasedIValues>;
  
  // 使用哈希函数 HashAliasedIValue 和比较函数 CompAliasedIValues 定义 unordered_map 容器
  using HashAliasedIValueMap =
      std::unordered_map<IValue, IValue, HashAliasedIValue, CompAliasedIValues>;

  // 定义一个哈希函数对象，用于计算给定 IValue 对象的哈希值（这里的计算方式是使用 IValue 的 payload.u.as_int）
  struct HashIdentityIValue {
    size_t operator()(const IValue& val) const {
      return val.payload.u.as_int;
    }
  };

  // 定义一个比较函数对象，用于比较两个 IValue 对象是否是相同的值（即 IValue 的 is 方法）
  struct CompIdentityIValues {
    bool operator()(const IValue& lhs, const IValue& rhs) const {
      return lhs.is(rhs);
    }
  };
  }
};

using HashIdentityIValues =
    std::unordered_set<IValue, HashIdentityIValue, CompIdentityIValues>;
using HashIdentityIValueMap =
    std::unordered_map<IValue, IValue, HashIdentityIValue, CompIdentityIValues>;

// 检查当前 IValue 对象与 rhs 是否有共同的子值。
// 例如，[t1,t2] 和 [t2, t3] 返回 true。
bool overlaps(const IValue& rhs) const;

// 将当前 IValue 对象的所有子值插入到 subValues 中。
void getSubValues(HashAliasedIValues& subValues) const;

// 对每个子值应用访问者函数。
// TODO: 存在多个递归遍历 IValue 的地方，这是脆弱的。
// 应该使用此访问者函数递归遍历 ivalues。
void visit(const std::function<bool(const IValue&)>& visitor) const;

// 深度复制当前 IValue 对象。
// 可选地指定设备参数。
IValue deepcopy(std::optional<at::Device> device = c10::nullopt) const;

// 在 memo 中进行当前 IValue 对象的深度复制。
// 可选地指定设备参数。
IValue deepcopy(
    HashIdentityIValueMap& memo,
    std::optional<at::Device> device = c10::nullopt) const;

private:
static c10::intrusive_ptr_target* null_to_undefined_tensor(
    c10::intrusive_ptr_target* p) {
  // 如果 p 不为空，则返回 p；否则返回 UndefinedTensorImpl 的单例指针。
  return p ? p
           : static_cast<c10::intrusive_ptr_target*>(
                 c10::UndefinedTensorImpl::singleton());
}

static bool ptrEqual(const IValue& lhs, const IValue& rhs);

// 注意：IValue 的标签意图上是私有的。将来我们可能会以不同方式编码此值（例如使用 NaN 包装），
// 这将使得确定所有类型的标签变得更加昂贵，而不仅仅是确定某些类型。
// 相反，我们希望客户端在可能的情况下使用 `isX` 方法。
// 如果出于性能原因确实绝对必须使用跳转表，则我们可以重新审视这一点。
enum class Tag : uint32_t {
  // 定义宏 DEFINE_TAG(x)，将 x 作为标签定义
#define DEFINE_TAG(x) x,
    // 对于 TORCH_FORALL_TAGS 宏中的每个标签调用 DEFINE_TAG 宏
    TORCH_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
  };

  // 定义宏 COUNT_TAG(x)，用于计算标签数量
#define COUNT_TAG(x) 1 +
  // 使用 TORCH_FORALL_TAGS 宏计算标签的总数并存储在 kNumTags 中
  static constexpr auto kNumTags = TORCH_FORALL_TAGS(COUNT_TAG) 0;
#undef COUNT_TAG

  // 模板函数声明，将对象转换为具有内部引用计数的指针
  template <
      class T,
      class NullType = c10::detail::intrusive_target_default_null_type<T>>
  c10::intrusive_ptr<T, NullType> moveToIntrusivePtr();
  // 模板函数声明，将对象转换为常量引用计数的指针
  template <
      typename T,
      class NullType = c10::detail::intrusive_target_default_null_type<T>>
  c10::intrusive_ptr<T, NullType> toIntrusivePtr() const;

  // 销毁对象的方法
  void destroy() {
    // 通过 payload 中的信息安全地执行销毁操作，避免未定义行为并保持代码生成的一致性
    if (isTensor() || isIntrusivePtr()) {
      // 根据对象类型选择正确的指针 p
      c10::intrusive_ptr_target* p = isTensor()
          ? payload.as_tensor.unsafeGetTensorImpl()
          : payload.u.as_intrusive_ptr;
      // 使用特定的析构函数 reclaim 安全地释放指针 p 的资源
      c10::intrusive_ptr<intrusive_ptr_target, c10::UndefinedTensorImpl>::
          reclaim(p);
      // 不需要调用析构函数来释放 payload.as_tensor
      // payload.as_tensor.~Tensor();
    }
  }

  // 将传入的右值引用 rhs 移动到当前对象中
  // 在性能上略微优化，避免不必要的析构函数调用
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  C10_ALWAYS_INLINE void moveFrom(IValue&& rhs) noexcept {
    if (rhs.isTensor()) {
      // 使用移动构造函数将 rhs 中的 Tensor 移动到当前对象的 payload.as_tensor 中
      new (&payload.as_tensor) at::Tensor(std::move(rhs.payload.as_tensor));
      // 由于移动后的 Tensor 对象是无效的，不需要显式调用析构函数
      // rhs.payload.as_tensor.~Tensor();
    } else {
      // 否则直接移动 rhs 的 payload.u 到当前对象的 payload.u
      payload.u = rhs.payload.u;
    }
    // 将标签 tag 设为 rhs 的标签
    tag = rhs.tag;
    // 清空 rhs 对象，使其变为 None 状态
    rhs.clearToNone();
  }

  // 将当前对象清空为 None 状态
  void clearToNone() noexcept {
    // 将 payload.u 的整型部分置零
    payload.u.as_int = 0;
    // 将标签 tag 设为 None
    tag = Tag::None;
  }

 private:
  // 用于确定对象是否为内部引用计数指针的静态方法
  // NOLINTBEGIN(bugprone-branch-clone)
  static constexpr bool isIntrusivePtrConstexpr(Tag tag) {
    // 根据给定的标签(tag)判断是否属于某些特定类型，并返回相应的布尔值
    switch (tag) {
      // 如果标签为None，则返回false
      case Tag::None:
        return false;
      // 如果标签为Tensor，则返回false
      case Tag::Tensor:
        return false;
      // 如果标签为Storage，则返回true
      case Tag::Storage:
        return true;
      // 如果标签为Generator，则返回true
      case Tag::Generator:
        return true;
      // 如果标签为Double，则返回false
      case Tag::Double:
        return false;
      // 如果标签为ComplexDouble，则返回true
      case Tag::ComplexDouble:
        return true;
      // 如果标签为Int，则返回false
      case Tag::Int:
        return false;
      // 如果标签为SymInt，则返回true
      case Tag::SymInt:
        return true;
      // 如果标签为SymFloat，则返回true
      case Tag::SymFloat:
        return true;
      // 如果标签为SymBool，则返回true
      case Tag::SymBool:
        return true;
      // 如果标签为Bool，则返回false
      case Tag::Bool:
        return false;
      // 如果标签为Tuple，则返回true
      case Tag::Tuple:
        return true;
      // 如果标签为String，则返回true
      case Tag::String:
        return true;
      // 如果标签为Blob，则返回true
      case Tag::Blob:
        return true;
      // 如果标签为GenericList，则返回true
      case Tag::GenericList:
        return true;
      // 如果标签为GenericDict，则返回true
      case Tag::GenericDict:
        return true;
      // 如果标签为Future，则返回true
      case Tag::Future:
        return true;
      // 如果标签为Await，则返回true
      case Tag::Await:
        return true;
      // 如果标签为Device，则返回false
      case Tag::Device:
        return false;
      // 如果标签为Stream，则返回true
      case Tag::Stream:
        return true;
      // 如果标签为Object，则返回true
      case Tag::Object:
        return true;
      // 如果标签为PyObject，则返回true
      case Tag::PyObject:
        return true;
      // 如果标签为Uninitialized，则返回false
      case Tag::Uninitialized:
        return false;
      // 如果标签为Capsule，则返回true
      case Tag::Capsule:
        return true;
      // 如果标签为RRef，则返回true
      case Tag::RRef:
        return true;
      // 如果标签为Quantizer，则返回true
      case Tag::Quantizer:
        return true;
      // 如果标签为Enum，则返回true
      case Tag::Enum:
        return true;
    }
    // 如果标签不匹配上述任何一种情况，则默认返回false
    return false;
  }
  // NOLINTEND(bugprone-branch-clone)

 public:
  // 不要编辑这部分以添加新标签的结果；编辑 isIntrusivePtrConstexpr 来实现这个目的。
  // 检查当前对象是否是侵入式指针的常量版本
  bool isIntrusivePtr() const {
    // 实现注意事项：isIntrusivePtrConstexpr 函数中的 switch 是这个函数之前的实现版本。
    // 我们观察到，在 x86_64 上生成的指令序列与我们下面手动实现的位向量测试非常相似，
    // 除了多了一个“边界检查”分支，确认 `tag < kNumTags`，在这种情况下提供一致的结果。
    // 如果 tag 超出范围，我们不关心结果，因此我们想消除该比较和分支；
    // 通过位测试手动实现这个函数是我能找到的最简单的方法来实现这个消除。
    static constexpr uint32_t kTruthTableBitVector =
#define TRUTH_TABLE_ENTRY(tag) \
  (uint32_t(isIntrusivePtrConstexpr(Tag::tag)) << uint32_t(Tag::tag)) | \
        TORCH_FORALL_TAGS(TRUTH_TABLE_ENTRY)
#undef TRUTH_TABLE_ENTRY
            0;

这段代码定义了一个宏 `TRUTH_TABLE_ENTRY`，它用于生成一个按位与的结果。这个宏的展开包括对 `isIntrusivePtrConstexpr` 和 `TORCH_FORALL_TAGS` 的调用，最后加上一个常量 `0`。


TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
    static_cast<uint32_t>(tag) < kNumTags,
    "unexpected tag ",
    static_cast<int>(tag));

这是一个调试用的内部断言，用于确保 `tag` 的值在有效范围内。如果 `tag` 的值超出了预期的范围，将会输出错误信息。


return kTruthTableBitVector & (1 << (uint32_t(tag) % 32));

这个函数返回一个位掩码 `kTruthTableBitVector` 和 `(1 << (uint32_t(tag) % 32))` 的按位与操作结果。它用于检查给定 `tag` 是否在位掩码中设置了相应的位。


// Storage and Generator were treated specially when
// is_intrusive_ptr was stored as explicit state. This getter
// preserves the old behavior for use with WeakIValue for now.
bool isIntrusivePtrLegacyBehavior() const {
  if (tag == Tag::Storage || tag == Tag::Generator) {
    return payload.u.as_intrusive_ptr !=
        c10::UndefinedTensorImpl::singleton();
  } else {
    return isIntrusivePtr();
  }
}

这个函数检查当前对象是否具有旧的内部行为（legacy behavior）以处理 `Storage` 和 `Generator` 标签的情况。它根据 `tag` 的值判断是否应该返回 `payload.u.as_intrusive_ptr` 是否等于 `c10::UndefinedTensorImpl::singleton()`，否则调用 `isIntrusivePtr()`。


union Payload {
  // [TriviallyCopyablePayload]
  // We use a nested union here so that we can make the copy easy
  // and efficient in the non-tensor (i.e., trivially copyable)
  // case. Specifically, we do not have to do a switch-on-tag to
  // figure out which union member to assign; we can just use
  // TriviallyCopyablePayload::operator=.
  union TriviallyCopyablePayload {
    TriviallyCopyablePayload() : as_int(0) {}
    int64_t as_int;
    double as_double;
    bool as_bool;
    // Invariant: never nullptr; null state is represented as
    // c10::UndefinedTensorImpl::singleton() for consistency of
    // representation with Tensor.
    c10::intrusive_ptr_target* as_intrusive_ptr;
    struct {
      c10::DeviceType type;
      DeviceIndex index;
    } as_device;
  } u;
  at::Tensor as_tensor;
  Payload() : u() {}
  ~Payload() {}
};

这段代码定义了一个联合体 `Payload`，它包含了两个主要部分：`TriviallyCopyablePayload` 和 `as_tensor`。`TriviallyCopyablePayload` 是一个内部联合体，用于存储基本类型数据，包括整数、双精度浮点数、布尔值和指向 `intrusive_ptr_target` 的指针。`as_tensor` 是一个 `at::Tensor` 对象，用于存储张量数据。


IValue(const Payload& p, Tag t) : tag(t) {
  if (isTensor()) {
    new (&payload.as_tensor) at::Tensor(p.as_tensor);
  } else {
    payload.u = p.u;
  }
}

这是一个构造函数 `IValue`，它接受 `Payload` 和 `Tag` 作为参数。根据对象是否是张量（由 `isTensor()` 判断），它会选择性地初始化 `payload.as_tensor` 或者 `payload.u`。


WeakIValue(const WeakIValue& rhs)
    : payload(rhs.payload),
      tag(rhs.tag),
      is_intrusive_ptr(rhs.is_intrusive_ptr) {
  if (is_intrusive_ptr &&
      payload.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton()) {
    c10::raw::weak_intrusive_ptr::incref(payload.as_intrusive_ptr);
  }
}

这是 `WeakIValue` 的拷贝构造函数，用于复制另一个 `WeakIValue` 对象。它会复制 `payload`、`tag` 和 `is_intrusive_ptr` 的值，并在 `is_intrusive_ptr` 为真且 `payload.as_intrusive_ptr` 不为 `c10::UndefinedTensorImpl::singleton()` 时增加引用计数。


WeakIValue(const IValue& rhs)
    : tag(rhs.tag), is_intrusive_ptr(rhs.isIntrusivePtrLegacyBehavior()) {
  if (rhs.isTensor()) {
    payload.as_intrusive_ptr = rhs.unsafeToTensorImpl();
    is_intrusive_ptr = true;
  } else {
    payload = rhs.payload.u;
  }
  if (is_intrusive_ptr) {
    if (payload.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton()) {
      c10::raw::weak_intrusive_ptr::incref(payload.as_intrusive_ptr);
    }
  }
}

这是 `WeakIValue` 的从 `IValue` 转换构造函数。它根据 `rhs` 的类型和状态初始化 `tag`、`is_intrusive_ptr` 和 `payload`。


WeakIValue(WeakIValue&& rhs) noexcept : WeakIValue() {

这是 `WeakIValue` 的移动构造函数，使用了移动语义，并标记为 `noexcept`。
    swap(rhs);
  }
  // 析构函数，释放资源
  ~WeakIValue() {
    // 如果是指向内部共享指针且不是空指针
    if (is_intrusive_ptr &&
        payload.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton()) {
      // 减少内部共享指针的弱引用计数
      c10::raw::weak_intrusive_ptr::decref(payload.as_intrusive_ptr);
    }
  }
  // 移动赋值运算符，交换当前对象与rhs的内容
  WeakIValue& operator=(WeakIValue&& rhs) & noexcept {
    // 创建临时对象并交换其内容到当前对象，同时设置rhs为None
    WeakIValue(std::move(rhs)).swap(*this);
    return *this;
  }
  // 拷贝赋值运算符，将rhs内容交换到当前对象
  WeakIValue& operator=(WeakIValue const& rhs) & {
    // 创建临时对象并交换其内容到当前对象
    WeakIValue(rhs).swap(*this);
    return *this;
  }
  // 交换函数，交换当前对象与rhs对象的成员变量
  void swap(WeakIValue& rhs) noexcept {
    std::swap(payload, rhs.payload);
    std::swap(is_intrusive_ptr, rhs.is_intrusive_ptr);
    std::swap(tag, rhs.tag);
  }

  // 判断是否具有相同的标识
  bool isSameIdentity(const WeakIValue& rhs) const {
    return payload.as_int == rhs.payload.as_int && tag == rhs.tag &&
        is_intrusive_ptr == rhs.is_intrusive_ptr;
  }

  // 锁定函数，返回当前对象的IValue表示
  IValue lock() const {
    // 如果不是内部共享指针，直接复制payload并返回
    if (!is_intrusive_ptr) {
      IValue::Payload newPayload;
      newPayload.u = payload;
      return IValue(newPayload, tag);
    }
    // 如果是Tensor类型的内部共享指针
    if (IValue::Tag::Tensor == tag) {
      // 从内部共享指针中获取TensorImpl对象并锁定它
      auto temp =
          c10::weak_intrusive_ptr<at::TensorImpl, c10::UndefinedTensorImpl>::
              reclaim(static_cast<at::TensorImpl*>(payload.as_intrusive_ptr));
      c10::intrusive_ptr<at::TensorImpl, c10::UndefinedTensorImpl> ip(
          temp.lock());
      temp.release();
      // 如果锁定失败返回空的IValue，否则将Tensor对象包装成IValue返回
      if (!ip) {
        return IValue();
      } else {
        return IValue(at::Tensor(std::move(ip)));
      }
    } else {
      // 对于其他类型的内部共享指针，锁定并创建对应的IValue返回
      auto temp = c10::weak_intrusive_ptr<c10::intrusive_ptr_target>::reclaim(
          payload.as_intrusive_ptr == c10::UndefinedTensorImpl::singleton()
              ? nullptr
              : payload.as_intrusive_ptr);
      IValue::Payload pl;
      pl.u.as_intrusive_ptr = temp.lock().release();
      temp.release();
      // 如果锁定失败返回空的IValue，否则创建对应的IValue返回
      if (!pl.u.as_intrusive_ptr) {
        return IValue();
      } else {
        return IValue(pl, tag);
      }
    }
  }

  // 返回当前对象的引用计数
  size_t use_count() const noexcept {
    // 如果不是内部共享指针，引用计数为1
    if (!is_intrusive_ptr) {
      return 1;
    }
    // 获取内部共享指针对象并返回其引用计数
    auto temp = c10::weak_intrusive_ptr<
        c10::intrusive_ptr_target,
        c10::UndefinedTensorImpl>::reclaim(payload.as_intrusive_ptr);
    size_t result = temp.use_count();
    temp.release();
    return result;
  }

  // 返回当前对象的弱引用计数
  size_t weak_use_count() const noexcept {
    // 如果不是内部共享指针，弱引用计数为1
    if (!is_intrusive_ptr) {
      return 1;
    }
    // 获取内部共享指针对象并返回其弱引用计数
    auto temp = c10::weak_intrusive_ptr<
        c10::intrusive_ptr_target,
        c10::UndefinedTensorImpl>::reclaim(payload.as_intrusive_ptr);
    size_t result = temp.weak_use_count();
    temp.release();
    return result;
  }
  // 返回当前对象的哈希值
  size_t hash() const {
    return payload.as_int;
  }

 private:
  // 使用IValue的TriviallyCopyablePayload作为Payload类型
  using Payload = IValue::Payload::TriviallyCopyablePayload;
  // Payload成员变量
  Payload payload;
  // 标签成员变量，默认为None
  IValue::Tag tag{IValue::Tag::None};
  // 是否为内部共享指针的标志，默认为false
  bool is_intrusive_ptr{false};
};

// An owning pointer to a type. When the type is class type, it requires a pair
// of shared_ptrs to the class type and its owning CU, so that the class type is
// guaranteed to stay alive as long as we hold this object.
struct TORCH_API StrongTypePtr {
  StrongTypePtr(std::shared_ptr<torch::jit::CompilationUnit> cu, TypePtr type);

  std::shared_ptr<torch::jit::CompilationUnit> cu_;   // 持有 CompilationUnit 的 shared_ptr
  TypePtr type_;                                       // 指向类型的智能指针
};

// [Constant Object Weak CompilationUnit Reference]
// A non owning pointer to a type. When a class get inserted as a constant
// into a graph, if we used a strong pointer we would have a circular reference
// from Object -> CompilationUnit and CompilationUnit -> Graph (which owns the
// Constant Object)
struct TORCH_API WeakTypePtr {
  WeakTypePtr(std::weak_ptr<torch::jit::CompilationUnit> cu, TypePtr type);

  std::weak_ptr<torch::jit::CompilationUnit> cu_;      // 弱引用指向 CompilationUnit 的 weak_ptr
  TypePtr type_;                                       // 指向类型的智能指针
};

// internal build errors with std::variant :/
struct WeakOrStrongCompilationUnit {
  explicit WeakOrStrongCompilationUnit(
      std::shared_ptr<torch::jit::CompilationUnit> shared_cu)
      : strong_ptr_(std::move(shared_cu)), weak_ptr_(c10::nullopt) {}

  explicit WeakOrStrongCompilationUnit(
      std::weak_ptr<torch::jit::CompilationUnit> weak_cu)
      : strong_ptr_(c10::nullopt), weak_ptr_(std::move(weak_cu)) {}

  std::shared_ptr<torch::jit::CompilationUnit> getStrongRefOrThrow() const {
    TORCH_INTERNAL_ASSERT(strong_ptr_ != c10::nullopt);
    return *strong_ptr_;
  }

  std::weak_ptr<torch::jit::CompilationUnit> getWeakRefOrThrow() const {
    TORCH_INTERNAL_ASSERT(weak_ptr_ != c10::nullopt);
    return *weak_ptr_;
  }

  bool holdingStrongRef() const {
    return strong_ptr_ != c10::nullopt;
  }

  bool holdingEmptyStrongRef() const {
    return holdingStrongRef() && *strong_ptr_ == nullptr;
  }

  std::optional<std::shared_ptr<torch::jit::CompilationUnit>> strong_ptr_;  // 可能持有 CompilationUnit 的 shared_ptr
  std::optional<std::weak_ptr<torch::jit::CompilationUnit>> weak_ptr_;      // 可能持有 CompilationUnit 的 weak_ptr
};

// An Object will hold a non-owning Compilation Unit reference if it is a
// Constant in the graph and a Owning reference otherwise
struct TORCH_API WeakOrStrongTypePtr {
  explicit WeakOrStrongTypePtr(WeakTypePtr weak)
      : cu_(WeakOrStrongCompilationUnit(std::move(weak.cu_))),
        type_(std::move(weak.type_)) {}
  explicit WeakOrStrongTypePtr(StrongTypePtr strong)
      : cu_(WeakOrStrongCompilationUnit(std::move(strong.cu_))),
        type_(std::move(strong.type_)) {}
  explicit WeakOrStrongTypePtr(WeakOrStrongCompilationUnit cu, TypePtr type)
      : cu_(std::move(cu)), type_(std::move(type)) {}

  WeakTypePtr asWeakTypePtr() const;

  WeakOrStrongCompilationUnit cu_;  // 可能持有 CompilationUnit 的强引用或者弱引用
  TypePtr type_;                    // 指向类型的智能指针

  bool holds_strong_ref() const {   // 检查是否持有强引用
    return cu_.holdingStrongRef();
  }

  bool holds_empty_strong_ref() const {  // 检查是否持有空的强引用
    return cu_.holdingEmptyStrongRef();
  }
};

} // namespace c10

#include <ATen/core/ivalue_inl.h> // IWYU pragma: keep
```