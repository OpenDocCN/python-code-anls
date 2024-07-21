# `.\pytorch\aten\src\ATen\core\ivalue_inl.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <condition_variable>
#include <memory>
#include <type_traits>
#include <utility>

#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/functional.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <ATen/core/rref_interface.h>
#include <ATen/core/symbol.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/core/Scalar.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/FunctionRef.h>
#include <c10/util/Logging.h>
#include <c10/util/hash.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>

// 命名空间 torch 下的命名空间 jit
namespace torch {
namespace jit {
struct Function;
struct CompilationUnit;
} // namespace jit
TORCH_API bool isCustomClass(const c10::IValue& v);
} // namespace torch

// 命名空间 c10 下的结构体和类的声明
namespace c10 {
struct IValue;
struct ClassType;
struct TupleType;
struct EnumType;
struct InferredType;

// 自定义类的初始化注册，需要传入一个函数，其形式类似于 [](IValue x, args...)
// make_boxed_from_unboxed_functor.h 自动根据函子的类型设置函数的输入类型
// 我们需要的是其绑定到 Foo 类型

// 因此，我们传入一个 lambda [](ivalue_holder<CurClass> x, args...)，通过 getTypePtr 可以恢复原始类指针
template <typename TaggedCapsuleType>
struct tagged_capsule {
  IValue ivalue;
};

// 将 IValue 转移到具有引用计数的指针
template <class T, class NullType>
c10::intrusive_ptr<T, NullType> IValue::moveToIntrusivePtr() {
  auto t = c10::intrusive_ptr<T, NullType>::reclaim(
      payload.u.as_intrusive_ptr == c10::UndefinedTensorImpl::singleton()
      ? NullType::singleton()
      : static_cast<T*>(payload.u.as_intrusive_ptr));
  clearToNone();
  return t;
}

// 将 IValue 转换为具有引用计数的指针
template <typename T, class NullType>
c10::intrusive_ptr<T, NullType> IValue::toIntrusivePtr() const {
  if (payload.u.as_intrusive_ptr == c10::UndefinedTensorImpl::singleton()) {
    return c10::intrusive_ptr<T, NullType>();
  }
  c10::raw::intrusive_ptr::incref(payload.u.as_intrusive_ptr);
  return c10::intrusive_ptr<T, NullType>::reclaim(
      static_cast<T*>(payload.u.as_intrusive_ptr));
}

// 静态引用计数指针的类型转换
template <class T, class U>
intrusive_ptr<T> static_intrusive_pointer_cast(intrusive_ptr<U> r) {
  return intrusive_ptr<T>::reclaim(static_cast<T*>(r.release()));
}

// 动态引用计数指针的类型转换
template <class T, class U>
intrusive_ptr<T> dynamic_intrusive_pointer_cast(intrusive_ptr<U> r) {
  return intrusive_ptr<T>::reclaim(dynamic_cast<T*>(r.release()));
}

// 将 IValue 转换为 Future 的引用计数指针
inline c10::intrusive_ptr<ivalue::Future> IValue::toFuture() && {
  AT_ASSERT(isFuture(), "Expected Future but got ", tagKind());
  return moveToIntrusivePtr<ivalue::Future>();
}

// 声明结束
// 返回一个指向 Future 对象的智能指针，用于常规引用的方法
inline c10::intrusive_ptr<ivalue::Future> IValue::toFuture() const& {
  // 断言当前对象确实是 Future 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isFuture(), "Expected Future but got ", tagKind());
  // 调用通用方法将当前对象转换为 Future 类型的智能指针并返回
  return toIntrusivePtr<ivalue::Future>();
}

// 返回一个指向 Await 对象的智能指针，用于右值引用的方法
inline c10::intrusive_ptr<ivalue::Await> IValue::toAwait() && {
  // 断言当前对象确实是 Await 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isAwait(), "Expected Await but got ", tagKind());
  // 调用移动语义方法将当前对象转换为 Await 类型的智能指针并返回
  return moveToIntrusivePtr<ivalue::Await>();
}

// 返回一个指向 Await 对象的智能指针，用于常规引用的方法
inline c10::intrusive_ptr<ivalue::Await> IValue::toAwait() const& {
  // 断言当前对象确实是 Await 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isAwait(), "Expected Await but got ", tagKind());
  // 调用通用方法将当前对象转换为 Await 类型的智能指针并返回
  return toIntrusivePtr<ivalue::Await>();
}

// 返回一个指向 RRefInterface 对象的智能指针，用于右值引用的方法
inline c10::intrusive_ptr<c10::RRefInterface> IValue::toRRef() && {
  // 断言当前对象确实是 RRef 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isRRef(), "Expected RRef but got ", tagKind());
  // 调用移动语义方法将当前对象转换为 RRef 类型的智能指针并返回
  return moveToIntrusivePtr<c10::RRefInterface>();
}

// 返回一个指向 RRefInterface 对象的智能指针，用于常规引用的方法
inline c10::intrusive_ptr<c10::RRefInterface> IValue::toRRef() const& {
  // 断言当前对象确实是 RRef 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isRRef(), "Expected RRef but got ", tagKind());
  // 调用通用方法将当前对象转换为 RRef 类型的智能指针并返回
  return toIntrusivePtr<c10::RRefInterface>();
}

// 返回一个指向 Quantizer 对象的智能指针，用于右值引用的方法
inline c10::intrusive_ptr<at::Quantizer> IValue::toQuantizer() && {
  // 断言当前对象确实是 Quantizer 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isQuantizer(), "Expected Quantizer but got ", tagKind());
  // 调用移动语义方法将当前对象转换为 Quantizer 类型的智能指针并返回
  return moveToIntrusivePtr<at::Quantizer>();
}

// 返回一个指向 Quantizer 对象的智能指针，用于常规引用的方法
inline c10::intrusive_ptr<at::Quantizer> IValue::toQuantizer() const& {
  // 断言当前对象确实是 Quantizer 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isQuantizer(), "Expected Quantizer but got ", tagKind());
  // 调用通用方法将当前对象转换为 Quantizer 类型的智能指针并返回
  return toIntrusivePtr<at::Quantizer>();
}

// 返回一个指向 ConstantString 对象的智能指针，用于右值引用的方法
inline c10::intrusive_ptr<ivalue::ConstantString> IValue::toString() && {
  // 断言当前对象确实是 String 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isString(), "Expected String but got ", tagKind());
  // 调用移动语义方法将当前对象转换为 String 类型的智能指针并返回
  return moveToIntrusivePtr<ivalue::ConstantString>();
}

// 返回一个指向 ConstantString 对象的智能指针，用于常规引用的方法
inline c10::intrusive_ptr<ivalue::ConstantString> IValue::toString() const& {
  // 断言当前对象确实是 String 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isString(), "Expected String but got ", tagKind());
  // 调用通用方法将当前对象转换为 String 类型的智能指针并返回
  return toIntrusivePtr<ivalue::ConstantString>();
}

// 返回一个指向 Object 对象的智能指针，用于右值引用的方法
inline c10::intrusive_ptr<ivalue::Object> IValue::toObject() && {
  // 断言当前对象确实是 Object 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  // 调用移动语义方法将当前对象转换为 Object 类型的智能指针并返回
  return moveToIntrusivePtr<ivalue::Object>();
}

// 返回一个指向 Object 对象的智能指针，用于常规引用的方法
inline c10::intrusive_ptr<ivalue::Object> IValue::toObject() const& {
  // 断言当前对象确实是 Object 类型，否则输出错误信息和当前对象的标签类型
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  // 调用通用方法将当前对象转换为 Object 类型的智能指针并返回
  return toIntrusivePtr<ivalue::Object>();
}

// 返回一个指向 PyObjectHolder 对象的智能指针，用于右值引用的方法
inline c10::intrusive_ptr<ivalue::PyObjectHolder> IValue::
    toPyObjectHolder() && {
  // 内部断言当前对象确实是 PyObject 类型，否则输出错误信息和当前对象的标签类型
  TORCH_INTERNAL_ASSERT(isPyObject(), "Expected PyObject but got ", tagKind());
  // 调用移动语义方法将当前对象转换为 PyObject 类型的智能指针并返回
  return moveToIntrusivePtr<ivalue::PyObjectHolder>();
}

// 返回一个指向 PyObjectHolder 对象的智能指针，用于常规引用的方法
inline c10::intrusive_ptr<ivalue::PyObjectHolder> IValue::toPyObjectHolder()
    const& {
  // 内部断言当前对象确实是 PyObject 类型，否则输出错误信息和当前对象的标签类型
  TORCH_INTERNAL_ASSERT(isPyObject(), "Expected PyObject but got ", tagKind());
  // 调用通用方法将当前对象转换为 PyObject 类型的智能指针并返回
  return toIntrusivePtr<ivalue::PyObjectHolder>();
}

// 返回一个指向 EnumHolder 对象的智能指针，用于右值引用的方法
inline c10::intrusive_ptr<ivalue::EnumHolder> IValue::toEnumHolder() && {
  // 内部断言当前对象确实是 Enum 类型，否则输出错误信息和当前对象的标签类型
  TORCH_INTERNAL_ASSERT(isEnum(), "Expected Enum but got ", tagKind());
  // 调用移动语义方法将当前对象转换为 Enum 类型的智能指针并返回
  return moveToIntrusivePtr<ivalue::EnumHolder>();
}

// 返回一个指向 EnumHolder 对象的智能指针，用于常规引用的方法
inline c10::intrusive_ptr<ivalue::EnumHolder> IValue::toEnumHolder() const& {
  // 内部断言当前对象确实是 Enum 类型，否则输出错误信息和当前对象的标签类型
  TORCH_INTERNAL_ASSERT(isEnum(), "Expected Enum but got ", tagKind());
  // 调用通用方法将当前对象转换为 Enum 类型的智能指针并返回
  return toIntrusivePtr<ivalue::EnumHolder>();
}
// 转换 IValue 对象为 complex<double> 类型
inline c10::complex<double> IValue::toComplexDouble() const {
  // 断言当前 IValue 对象是否为 ComplexDouble 类型，否则报错并打印类型信息
  TORCH_INTERNAL_ASSERT(isComplexDouble(), "Expected ComplexDouble but got ", tagKind());
  // 获取指向 ComplexHolder 的智能指针，然后返回其值
  auto ptr = toIntrusivePtr<ivalue::ComplexHolder>();
  return (*ptr).val;
}

// 转移 IValue 对象为 Tensor 类型
inline at::Tensor IValue::toTensor() && {
  // 如果当前对象不是 Tensor 类型，则报告类型错误
  if (C10_UNLIKELY(!isTensor())) {
    reportToTensorTypeError();
  }
  // 移动 payload 中的 Tensor，注释掉析构函数调用以提升性能
  auto result = std::move(payload.as_tensor);
  // 清除当前 IValue 对象的状态，将其设置为 None
  clearToNone();
  return result;
}

// 获取 IValue 对象的 Tensor 引用（非右值引用）
inline at::Tensor& IValue::toTensor() & {
  // 如果当前对象不是 Tensor 类型，则报告类型错误
  if (C10_UNLIKELY(!isTensor())) {
    reportToTensorTypeError();
  }
  // 返回 payload 中的 Tensor 引用
  return payload.as_tensor;
}

// 获取常量引用的 IValue 对象的 Tensor
inline const at::Tensor& IValue::toTensor() const& {
  // 如果当前对象不是 Tensor 类型，则报告类型错误
  if (C10_UNLIKELY(!isTensor())) {
    reportToTensorTypeError();
  }
  // 返回 payload 中的 Tensor 的常量引用
  return payload.as_tensor;
}

// 转移 IValue 对象为 Storage 类型
inline c10::Storage IValue::toStorage() && {
  // 断言当前 IValue 对象是否为 Storage 类型，否则报错并打印类型信息
  AT_ASSERT(isStorage(), "Expected Storage but got ", tagKind());
  // 调用 moveToIntrusivePtr 将 StorageImpl 转为智能指针，并创建 Storage 对象返回
  return c10::Storage(moveToIntrusivePtr<at::StorageImpl>());
}

// 获取常量引用的 IValue 对象的 Storage
inline c10::Storage IValue::toStorage() const& {
  // 断言当前 IValue 对象是否为 Storage 类型，否则报错并打印类型信息
  AT_ASSERT(isStorage(), "Expected Storage but got ", tagKind());
  // 调用 toIntrusivePtr 将 StorageImpl 转为智能指针，并创建 Storage 对象返回
  return c10::Storage(toIntrusivePtr<at::StorageImpl>());
}

// 转移 IValue 对象为 Stream 类型
inline c10::Stream IValue::toStream() && {
  // 断言当前 IValue 对象是否为 Stream 类型，否则报错并打印类型信息
  AT_ASSERT(isStream(), "Expected Stream but got ", tagKind());
  // 获取指向 StreamData3Holder 的智能指针，解包成 Stream 对象并返回
  auto ptr = toIntrusivePtr<ivalue::StreamData3Holder>();
  return c10::Stream::unpack3((*ptr).val.stream_id,
                              (*ptr).val.device_index,
                              (*ptr).val.device_type);
}

// 获取常量引用的 IValue 对象的 Stream
inline c10::Stream IValue::toStream() const& {
  // 断言当前 IValue 对象是否为 Stream 类型，否则报错并打印类型信息
  AT_ASSERT(isStream(), "Expected Stream but got ", tagKind());
  // 获取指向 StreamData3Holder 的智能指针，解包成 Stream 对象并返回
  auto ptr = toIntrusivePtr<ivalue::StreamData3Holder>();
  return c10::Stream::unpack3((*ptr).val.stream_id,
                              (*ptr).val.device_index,
                              (*ptr).val.device_type);
}

// 转移 IValue 对象为 Blob 类型
inline c10::intrusive_ptr<caffe2::Blob> IValue::toBlob() && {
  // 断言当前 IValue 对象是否为 Blob 类型，否则报错并打印类型信息
  AT_ASSERT(isBlob(), "Expected Blob but got ", tagKind());
  // 调用 moveToIntrusivePtr 将 Blob 转为智能指针，并返回
  return moveToIntrusivePtr<caffe2::Blob>();
}

// 获取常量引用的 IValue 对象的 Blob
inline c10::intrusive_ptr<caffe2::Blob> IValue::toBlob() const& {
  // 断言当前 IValue 对象是否为 Blob 类型，否则报错并打印类型信息
  AT_ASSERT(isBlob(), "Expected Blob but got ", tagKind());
  // 调用 toIntrusivePtr 将 Blob 转为智能指针，并返回
  return toIntrusivePtr<caffe2::Blob>();
  ;
}

// 转移 IValue 对象为 CustomClassHolder 类型（capsule）
inline c10::intrusive_ptr<torch::CustomClassHolder> IValue::toCapsule() && {
  // 内部断言当前 IValue 对象是否为 Capsule 类型，否则报错
  TORCH_INTERNAL_ASSERT(isCapsule());
  // 调用 moveToIntrusivePtr 将 CustomClassHolder 转为智能指针，并返回
  return moveToIntrusivePtr<torch::CustomClassHolder>();
}

// 获取常量引用的 IValue 对象的 CustomClassHolder（capsule）
inline c10::intrusive_ptr<torch::CustomClassHolder> IValue::toCapsule() const& {
  // 内部断言当前 IValue 对象是否为 Capsule 类型，否则报错
  TORCH_INTERNAL_ASSERT(isCapsule());
  // 调用 toIntrusivePtr 将 CustomClassHolder 转为智能指针，并返回
  return toIntrusivePtr<torch::CustomClassHolder>();
}
// 移动语义：将当前对象作为右值引用，转换为 Generator 对象并返回
inline at::Generator IValue::toGenerator() && {
  // 断言当前对象类型为 Generator
  AT_ASSERT(isGenerator(), "Expected Generator but got ", tagKind());
  // 调用 moveToIntrusivePtr 将内部指针移动到 GeneratorImpl 对象并返回
  return at::Generator(moveToIntrusivePtr<at::GeneratorImpl>());
}

// 左值引用：将当前对象作为左值引用，转换为 Generator 对象并返回
inline at::Generator IValue::toGenerator() const& {
  // 断言当前对象类型为 Generator
  AT_ASSERT(isGenerator(), "Expected Generator but got ", tagKind());
  // 调用 toIntrusivePtr 将内部指针转换为 GeneratorImpl 对象并返回
  return at::Generator(toIntrusivePtr<at::GeneratorImpl>());
}

// 移动语义：将当前对象作为右值引用，转换为 SymInt 对象并返回
inline c10::SymInt IValue::toSymInt() && {
  // 断言当前对象类型为 SymInt 或 int
  AT_ASSERT(isSymInt() || isInt(), "Expected SymInt or int but got ", tagKind());
  if (isSymInt()) {
    // 如果是 SymInt 类型，则移动内部指针到 SymNodeImpl 对象并返回
    return c10::SymInt(moveToIntrusivePtr<c10::SymNodeImpl>());
  } else {
    // 否则，直接返回 payload 中的整数值作为 SymInt 对象
    return c10::SymInt(payload.u.as_int);
  }
}

// 左值引用：将当前对象作为左值引用，转换为 SymInt 对象并返回
inline c10::SymInt IValue::toSymInt() const& {
  // 断言当前对象类型为 SymInt 或 int
  AT_ASSERT(isSymInt() || isInt(), "Expected SymInt or int but got ", tagKind());
  if (isSymInt()) {
    // 如果是 SymInt 类型，则转换内部指针为 SymNodeImpl 对象并返回
    return c10::SymInt(toIntrusivePtr<c10::SymNodeImpl>());
  } else {
    // 否则，直接返回 payload 中的整数值作为 SymInt 对象
    return c10::SymInt(payload.u.as_int);
  }
}

// 移动语义：将当前对象作为右值引用，转换为 SymFloat 对象并返回
inline c10::SymFloat IValue::toSymFloat() && {
  // 断言当前对象类型为 SymFloat 或 double
  AT_ASSERT(isSymFloat() || isDouble(), "Expected SymFloat or double but got ", tagKind());
  if (isSymFloat()) {
    // 如果是 SymFloat 类型，则移动内部指针到 SymNodeImpl 对象并返回
    return c10::SymFloat(moveToIntrusivePtr<c10::SymNodeImpl>());
  } else {
    // 否则，直接返回 payload 中的双精度浮点数值作为 SymFloat 对象
    return c10::SymFloat(payload.u.as_double);
  }
}

// 左值引用：将当前对象作为左值引用，转换为 SymFloat 对象并返回
inline c10::SymFloat IValue::toSymFloat() const& {
  // 断言当前对象类型为 SymFloat 或 double
  AT_ASSERT(isSymFloat() || isDouble(), "Expected SymFloat or double but got ", tagKind());
  if (isSymFloat()) {
    // 如果是 SymFloat 类型，则转换内部指针为 SymNodeImpl 对象并返回
    return c10::SymFloat(toIntrusivePtr<c10::SymNodeImpl>());
  } else {
    // 否则，直接返回 payload 中的双精度浮点数值作为 SymFloat 对象
    return c10::SymFloat(payload.u.as_double);
  }
}

// 移动语义：将当前对象作为右值引用，转换为 SymBool 对象并返回
inline c10::SymBool IValue::toSymBool() && {
  // 断言当前对象类型为 SymBool 或 boolean
  AT_ASSERT(isSymBool() || isBool(), "Expected SymBool or boolean but got ", tagKind());
  if (isSymBool()) {
    // 如果是 SymBool 类型，则移动内部指针到 SymNodeImpl 对象并返回
    return c10::SymBool(moveToIntrusivePtr<c10::SymNodeImpl>());
  } else {
    // 否则，直接返回 payload 中的布尔值作为 SymBool 对象
    return c10::SymBool(payload.u.as_bool);
  }
}

// 左值引用：将当前对象作为左值引用，转换为 SymBool 对象并返回
inline c10::SymBool IValue::toSymBool() const& {
  // 断言当前对象类型为 SymBool 或 boolean
  AT_ASSERT(isSymBool() || isBool(), "Expected SymBool or boolean but got ", tagKind());
  if (isSymBool()) {
    // 如果是 SymBool 类型，则转换内部指针为 SymNodeImpl 对象并返回
    return c10::SymBool(toIntrusivePtr<c10::SymNodeImpl>());
  } else {
    // 否则，直接返回 payload 中的布尔值作为 SymBool 对象
    return c10::SymBool(payload.u.as_bool);
  }
}
    // 返回一个空的字符串对象
    return string();
  }
  
  // 定义友元函数，用于将 ConstantString 对象输出到流中
  TORCH_API friend std::ostream& operator<<(
      std::ostream& out,
      const ConstantString& v);
};

// 声明一个名为 Future 的结构体，但没有在这段代码中定义其内容

struct Future;

// TupleElements 类定义开始
struct TORCH_API TupleElements {
 private:
  size_t inlineSize_;  // 存储 TupleElements 内元素的数量
  // 我们这样表示 TupleElements，以避免在常见情况（至少对于反序列化来说）下进行堆分配，
  // 即仅有 3 个元素的情况。我们使用自己的 union 而不是 c10::SmallVector<IValue>，
  // 因为 c10::SmallVector<IValue> 总是存储 begin/end/capacity 指针，
  // 在我们的用例中会浪费空间。

  union {
    std::vector<IValue> elementsVector_;  // 使用 vector 存储元素
    // 不想声明 std::array，因为方便的迭代和大小成员在这种情况下是一个隐患 ——
    // 数组的实际大小可能小于 3！
    // NOLINTNEXTLINE(*c-arrays*)
    IValue elementsInline_[3];  // 使用固定大小数组存储元素
  };

  // 销毁内联元素
  void destroyInline() {
    for (const auto ii : c10::irange(inlineSize_)) {
      elementsInline_[ii].~IValue();  // 逐个析构内联元素
    }
  }

 public:
  // 公有成员定义开始

  using iterator = IValue*;  // 迭代器类型为指向 IValue 的指针
  using const_iterator = const IValue*;  // const 迭代器类型为指向 const IValue 的指针

  // 默认构造函数，初始化 inlineSize_ 为 0
  TupleElements() : inlineSize_(0) {
    new (&elementsVector_) std::vector<IValue>();  // 在 elementsVector_ 上构造新的 vector
  }

  // 从 vector 构造 TupleElements
  explicit TupleElements(std::vector<IValue> elements)
      : inlineSize_(0), elementsVector_(std::move(elements)) {}

  // 从 c10::ArrayRef<IValue> 构造 TupleElements
  explicit TupleElements(c10::ArrayRef<IValue> elements)
      : inlineSize_(elements.size() <= 3 ? elements.size() : 0) {
    switch (inlineSize_) {
      case 3:
        new (&elementsInline_[2]) IValue(elements[2]);  // 构造第三个元素
        [[fallthrough]];  // 允许 fallthrough，即向下执行
      case 2:
        new (&elementsInline_[1]) IValue(elements[1]);  // 构造第二个元素
        [[fallthrough]];
      case 1:
        new (&elementsInline_[0]) IValue(elements[0]);  // 构造第一个元素
        break;
      case 0:
        new (&elementsVector_) std::vector<IValue>(elements.begin(), elements.end());  // 构造 vector
        break;
    }
  }

  // 从移动构造一个元素构造 TupleElements
  explicit TupleElements(IValue&& e1)
      : inlineSize_(1) {
    new (&elementsInline_[0]) IValue(std::move(e1));  // 移动构造第一个元素
  }

  // 从移动构造两个元素构造 TupleElements
  explicit TupleElements(IValue&& e1, IValue&& e2)
      : inlineSize_(2) {
    new (&elementsInline_[0]) IValue(std::move(e1));  // 移动构造第一个元素
    new (&elementsInline_[1]) IValue(std::move(e2));  // 移动构造第二个元素
  }

  // 从移动构造三个元素构造 TupleElements
  explicit TupleElements(IValue&& e1, IValue&& e2, IValue&& e3)
      : inlineSize_(3) {
    new (&elementsInline_[0]) IValue(std::move(e1));  // 移动构造第一个元素
    new (&elementsInline_[1]) IValue(std::move(e2));  // 移动构造第二个元素
    new (&elementsInline_[2]) IValue(std::move(e3));  // 移动构造第三个元素
  }

  // 析构函数，根据 inlineSize_ 的值选择销毁方式
  ~TupleElements() {
    if (inlineSize_) {
      destroyInline();  // 销毁内联元素
    } else {
      elementsVector_.~vector();  // 销毁 vector
    }
  }

  // 拷贝构造函数，复制 rhs 的内容到当前对象
  TupleElements(const TupleElements& rhs)
      : inlineSize_(rhs.inlineSize_) {

    // 拷贝构造函数，复制 rhs 的内容到当前对象
    inlineSize_(rhs.inlineSize_) {
    // 具体实现根据 inlineSize_ 的值选择不同的路径
    if (inlineSize_) {
      // 如果 inlineSize_ 大于 0，则复制内联元素
      for (const auto ii : c10::irange(inlineSize_)) {
        new (&elementsInline_[ii]) IValue(rhs.elementsInline_[ii]);
      }
    } else {
      // 否则复制 vector 中的元素
      new (&elementsVector_) std::vector<IValue>(rhs.elementsVector_);
    }
  }

  // 这里可以看到一段注释，描述了为什么没有把这个类设计成不可拷贝的。
  // 我们可以读到代码，解释在这个例子中，为什么以后的实现会依赖于拷贝复制，而不是构建出一个新的指
  // 如果右操作数的 inlineSize_ 不为零，执行以下代码块
  if (rhs.inlineSize_) {
    // 遍历从 0 到 inlineSize_ 的范围
    for (const auto ii : c10::irange(inlineSize_)) {
      // 使用拷贝构造函数在 elementsInline_ 数组中构造新的 IValue 对象
      new (&elementsInline_[ii]) IValue(rhs.elementsInline_[ii]);
    }
  } else {
    // 如果右操作数的 inlineSize_ 为零，则执行以下代码块
    // 在 elementsVector_ 中使用拷贝构造函数创建新的 std::vector<IValue> 对象
    new (&elementsVector_) std::vector<IValue>(rhs.elementsVector_);
  }
}

// 复制赋值运算符重载函数
TupleElements& operator=(const TupleElements& rhs) {
  // 如果当前对象的 inlineSize_ 不为零，执行以下代码块
  if (inlineSize_) {
    // 如果右操作数的 inlineSize_ 不为零，执行以下代码块
    if (rhs.inlineSize_) {
      // 遍历从 0 到 std::min(inlineSize_, rhs.inlineSize_) 的范围
      for (const auto ii : c10::irange(std::min(inlineSize_, rhs.inlineSize_))) {
        // 对 elementsInline_ 数组中的对象执行赋值操作
        elementsInline_[ii] = rhs.elementsInline_[ii];
      }
      // 如果 rhs 的 inlineSize_ 大于当前对象的 inlineSize_
      if (rhs.inlineSize_ > inlineSize_) {
        // 遍历从 inlineSize_ 到 rhs.inlineSize_ 的范围
        for (const auto ii : c10::irange(inlineSize_, rhs.inlineSize_)) {
          // 使用移动构造函数在 elementsInline_ 数组中构造新的 IValue 对象
          new (&elementsInline_[ii]) IValue(rhs.elementsInline_[ii]);
        }
      } else {
        // 如果当前对象的 inlineSize_ 大于 rhs 的 inlineSize_
        // 遍历从 rhs.inlineSize_ 到 inlineSize_ 的范围
        for (const auto ii : c10::irange(rhs.inlineSize_, inlineSize_)) {
          // 调用析构函数销毁 elementsInline_ 数组中的对象
          elementsInline_[ii].~IValue();
        }
      }
    } else {
      // 如果 rhs 的 inlineSize_ 为零，销毁当前对象的 inline 数据
      destroyInline();
      // 在 elementsVector_ 中使用拷贝构造函数创建新的 std::vector<IValue> 对象
      new (&elementsVector_) std::vector<IValue>(rhs.elementsVector_);
    }
  } else {
    // 如果当前对象的 inlineSize_ 为零，执行以下代码块
    if (rhs.inlineSize_) {
      // 销毁当前对象的 elementsVector_ 数据
      elementsVector_.~vector();
      // 遍历从 0 到 rhs.inlineSize_ 的范围
      for (const auto ii : c10::irange(rhs.inlineSize_)) {
        // 使用移动构造函数在 elementsInline_ 数组中构造新的 IValue 对象
        new (&elementsInline_[ii]) IValue(rhs.elementsInline_[ii]);
      }
    } else {
      // 如果 rhs 的 inlineSize_ 为零，执行以下代码块
      // 使用移动赋值运算符将 rhs 的 elementsVector_ 赋值给当前对象的 elementsVector_
      elementsVector_ = rhs.elementsVector_;
    }
  }
  // 将当前对象的 inlineSize_ 更新为 rhs 的 inlineSize_
  inlineSize_ = rhs.inlineSize_;
  // 返回当前对象的引用
  return *this;
}

// 移动构造函数，使用 std::move 将 rhs 的资源移动到当前对象
TupleElements(TupleElements&& rhs) noexcept
: inlineSize_(rhs.inlineSize_) {
  // 如果当前对象的 inlineSize_ 不为零，执行以下代码块
  if (inlineSize_) {
    // 遍历从 0 到 inlineSize_ 的范围
    for (const auto ii : c10::irange(inlineSize_)) {
      // 使用移动构造函数在 elementsInline_ 数组中构造新的 IValue 对象
      new (&elementsInline_[ii]) IValue(std::move(rhs.elementsInline_[ii]));
    }
  } else {
    // 如果当前对象的 inlineSize_ 为零
    // 在 elementsVector_ 中使用移动构造函数创建新的 std::vector<IValue> 对象
    new (&elementsVector_) std::vector<IValue>(std::move(rhs.elementsVector_));
  }
}

// 移动赋值运算符重载函数
TupleElements& operator=(TupleElements&& rhs) noexcept {
  // 如果当前对象的 inlineSize_ 不为零，执行以下代码块
  if (inlineSize_) {
    // 如果 rhs 的 inlineSize_ 不为零，执行以下代码块
    if (rhs.inlineSize_) {
      // 遍历从 0 到 std::min(inlineSize_, rhs.inlineSize_) 的范围
      for (const auto ii : c10::irange(std::min(inlineSize_, rhs.inlineSize_))) {
        // 使用移动赋值运算符将 rhs 的数据移动到当前对象的 elementsInline_ 数组
        elementsInline_[ii] = std::move(rhs.elementsInline_[ii]);
      }
      // 如果 rhs 的 inlineSize_ 大于当前对象的 inlineSize_
      if (rhs.inlineSize_ > inlineSize_) {
        // 遍历从 inlineSize_ 到 rhs.inlineSize_ 的范围
        for (const auto ii : c10::irange(inlineSize_, rhs.inlineSize_)) {
          // 使用移动构造函数在 elementsInline_ 数组中构造新的 IValue 对象
          new (&elementsInline_[ii]) IValue(std::move(rhs.elementsInline_[ii]));
        }
      } else {
        // 如果当前对象的 inlineSize_ 大于 rhs 的 inlineSize_
        // 遍历从 rhs.inlineSize_ 到 inlineSize_ 的范围
        for (const auto ii : c10::irange(rhs.inlineSize_, inlineSize_)) {
          // 调用析构函数销毁 elementsInline_ 数组中的对象
          elementsInline_[ii].~IValue();
        }
      }
    } else {
      // 如果 rhs 的 inlineSize_ 为零，销毁当前对象的 inline 数据
      destroyInline();
      // 在 elementsVector_ 中使用移动构造函数创建新的 std::vector<IValue> 对象
      new (&elementsVector_) std::vector<IValue>(std::move(rhs.elementsVector_));
    }
  } else {
    // 如果当前对象的 inlineSize_ 为零，执行以下代码块
    if (rhs.inlineSize_) {
      // 销毁当前对象的 elementsVector_ 数据
      elementsVector_.~vector();
      // 遍历从 0 到 rhs.inlineSize_ 的范围
      for (const auto ii : c10::irange(rhs.inlineSize_)) {
        // 使用移动构造函数在 elementsInline_ 数组中构造新的 IValue 对象
        new (&elementsInline_[ii]) IValue(std::move(rhs.elementsInline_[ii]));
      }
    } else {
      // 如果 rhs 的 inlineSize_ 为零，执行以下代码块
      // 使用移动赋值运算符将 rhs 的 elementsVector_ 移动到当前对象的 elementsVector_
      elementsVector_ = std::move(rhs.elementsVector_);
    }
  }
  // 将当前对象的 inlineSize_ 更新为 rhs 的 inlineSize_
  inlineSize_ = rhs.inlineSize_;
  // 返回当前对象的引用
  return *this;
}

// asArrayRef 方法，返回当前对象作为 c10::ArrayRef<IValue> 的引用
C10_NODISCARD c10::ArrayRef<IValue> asArrayRef() const {
  // 如果 inlineSize_ 非零，则返回内联元素的 ArrayRef
  if (inlineSize_) {
    return c10::ArrayRef<IValue>(elementsInline_, inlineSize_);
  } else {
    // 否则返回 elementsVector_ 的 ArrayRef
    return elementsVector_;
  }
}

// 将 TupleElements 转换为 c10::ArrayRef<IValue> 的隐式转换
operator c10::ArrayRef<IValue>() const {
  return asArrayRef();
}

// 计算 TupleElements 的哈希值
static size_t hash(const TupleElements& v) {
  return c10::hash<c10::ArrayRef<IValue>>()(v.asArrayRef());
}

// 设置 TupleElements 的内容为移动后的 std::vector<IValue>
void setContents(std::vector<IValue>&& contents) {
  // 如果 inlineSize_ 非零，则销毁内联元素并创建新的 elementsVector_
  if (inlineSize_) {
    destroyInline();
    new (&elementsVector_) std::vector<IValue>(std::move(contents));
    inlineSize_ = 0;
  } else {
    // 否则直接移动赋值给 elementsVector_
    elementsVector_ = std::move(contents);
  }
}

// 返回 TupleElements 是否为空
C10_NODISCARD bool empty() const {
  return inlineSize_ ? false : elementsVector_.empty();
}

// 返回 TupleElements 的大小
C10_NODISCARD size_t size() const {
  return inlineSize_ ? inlineSize_ : elementsVector_.size();
}

// 访问 TupleElements 中指定位置的元素（可变版本）
C10_NODISCARD IValue& operator[](size_t idx) {
  if (inlineSize_) {
    return elementsInline_[idx];
  } else {
    return elementsVector_[idx];
  }
}

// 访问 TupleElements 中指定位置的元素（常量版本）
C10_NODISCARD const IValue& operator[](size_t idx) const {
  if (inlineSize_) {
    return elementsInline_[idx];
  } else {
    return elementsVector_[idx];
  }
}

// 安全地访问 TupleElements 中指定位置的元素（可变版本）
C10_NODISCARD IValue& at(size_t idx) {
  if (inlineSize_) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inlineSize_ <= 3);
    TORCH_CHECK(idx < inlineSize_, "TupleElements: invalid index Index = ", idx, "; Length = ", inlineSize_);
    return elementsInline_[idx];
  } else {
    return elementsVector_.at(idx);
  }
}

// 安全地访问 TupleElements 中指定位置的元素（常量版本）
C10_NODISCARD const IValue& at(size_t idx) const {
  if (inlineSize_) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inlineSize_ <= 3);
    TORCH_CHECK(idx < inlineSize_, "TupleElements: invalid index Index = ", idx, "; Length = ", inlineSize_);
    return elementsInline_[idx];
  } else {
    TORCH_CHECK(idx < elementsVector_.size(), "TupleElements: invalid index Index = ", idx, "; Length = ", elementsVector_.size());
    return elementsVector_.at(idx);
  }
}

// 返回 TupleElements 的起始迭代器（可变版本）
C10_NODISCARD iterator begin() {
  if (inlineSize_) {
    return elementsInline_;
  } else {
    return elementsVector_.data();
  }
}

// 返回 TupleElements 的结束迭代器（可变版本）
C10_NODISCARD iterator end() {
  if (inlineSize_) {
    return elementsInline_ + inlineSize_;
  } else {
    return elementsVector_.data() + elementsVector_.size();
  }
}

// 返回 TupleElements 的起始迭代器（常量版本）
C10_NODISCARD const_iterator begin() const {
  if (inlineSize_) {
    return elementsInline_;
  } else {
    return elementsVector_.data();
  }
}

// 返回 TupleElements 的结束迭代器（常量版本）
C10_NODISCARD const_iterator end() const {
  if (inlineSize_) {
    return elementsInline_ + inlineSize_;
  } else {
    return elementsVector_.data() + elementsVector_.size();
  }
}

// 返回 TupleElements 的常量起始迭代器
C10_NODISCARD const_iterator cbegin() const {
  return begin();
}

// 返回 TupleElements 的常量结束迭代器
C10_NODISCARD const_iterator cend() const {
  return end();
}

// 返回 TupleElements 的内容作为 std::vector<IValue>
C10_NODISCARD std::vector<IValue> vec() const & {
  return asArrayRef().vec();
}

// 返回 TupleElements 中的最后一个元素（可变版本）
C10_NODISCARD IValue& back() {
    // 返回当前容器的最后一个元素（引用），通过减去1来获取末尾元素的迭代器，然后解引用获取元素
    return *(end() - 1);
  }

  // 返回当前容器的最后一个元素（常量引用），通过减去1来获取末尾元素的迭代器，然后解引用获取元素
  C10_NODISCARD const IValue& back() const {
    return *(end() - 1);
  }

  // 转移语义的向量化转换操作，将当前对象的所有元素转移到新的 std::vector<IValue> 对象中
  C10_NODISCARD std::vector<IValue> vec() && {
    // 创建一个新的 std::vector<IValue> 对象
    std::vector<IValue> result;
    // 预留容器空间以避免频繁的重新分配
    result.reserve(size());
    // 遍历当前对象的每个元素，并使用 std::move 将元素移动到结果向量中
    for (auto&& iv : *this) {
      result.push_back(std::move(iv));
    }
    // 返回包含当前对象元素的 std::vector<IValue> 对象
    return result;
  }

  // 将当前对象转换为 std::vector<IValue> 对象的常量引用
  // 这是为了与需要将元组元素复制到向量中的大量代码兼容
  operator std::vector<IValue>() const & {
    return vec();
  }

  // 将当前对象转换为 std::vector<IValue> 对象的移动语义版本
  // 这也是为了与需要将元组元素复制到向量中的大量代码兼容
  operator std::vector<IValue>() && {
    return vec();
  }
};

// 模板类：TupleTypeFactory，用于生成 TupleType 对象
template <typename T>
struct TupleTypeFactory {};

// 特化模板类 TupleTypeFactory，针对 TupleType 类型
template <>
struct TORCH_API TupleTypeFactory<TupleType> {
  // 创建 TupleTypePtr 对象，使用给定的类型列表
  static TupleTypePtr create(std::vector<TypePtr> types) {
    return TupleType::create(std::move(types));
  }
  // 提供回退选项，用于处理 Type 类型
  static TupleTypePtr fallback(const Type& type);
};

// 特化模板类 TupleTypeFactory，针对 c10::DynamicType 类型
template <>
struct TORCH_API TupleTypeFactory<c10::DynamicType> {
  // 创建 DynamicTypePtr 对象，使用给定的元素类型列表
  static DynamicTypePtr create(const std::vector<TypePtr>& elemTypes);
  // 提供回退选项，用于处理 Type 类型
  static DynamicTypePtr fallback(const Type&);
};

// 类：Tuple，继承自 c10::intrusive_ptr_target
struct TORCH_API Tuple : c10::intrusive_ptr_target {
 private:
  TupleElements elements_; // 元组的元素列表
  mutable c10::TypePtr type_; // 惰性计算的类型信息，对于无名元组

 public:
  // 创建具名元组，直接使用给定的元素和类型创建
  static c10::intrusive_ptr<Tuple> createNamed(
      std::vector<IValue> elements_,
      c10::TypePtr type_) {
    return c10::make_intrusive<Tuple>(std::move(elements_), std::move(type_));
  }

  // 创建具名元组，使用给定的元素和类型创建
  static c10::intrusive_ptr<Tuple> createNamed(
      TupleElements elements_,
      std::shared_ptr<TupleType> type_) {
    return c10::make_intrusive<Tuple>(std::move(elements_), std::move(type_));
  }

  // 创建具名元组，使用给定的初始化列表的元素和类型创建
  static c10::intrusive_ptr<Tuple> createNamed(
      std::initializer_list<IValue> elements_,
      std::shared_ptr<TupleType> type_) {
    return createNamed(TupleElements(c10::ArrayRef<IValue>(elements_)), std::move(type_));
  }

  // MSVC 明显无法在没有这个的情况下消除对 create 的其他两个重载的歧义
  // 创建元组，使用给定的初始化列表的元素创建
  static c10::intrusive_ptr<Tuple> create(std::initializer_list<IValue> elements_) {
    return create(c10::ArrayRef<IValue>(elements_));
  }

  // 创建元组，使用给定的元素列表创建
  static c10::intrusive_ptr<Tuple> create(std::vector<IValue> elements_) {
    return c10::make_intrusive<Tuple>(std::move(elements_));
  }

  // 创建元组，使用给定的元素列表创建
  static c10::intrusive_ptr<Tuple> create(TupleElements elements_) {
    return c10::make_intrusive<Tuple>(std::move(elements_));
  }

  // 创建元组，使用给定的元素数组引用创建
  static c10::intrusive_ptr<Tuple> create(c10::ArrayRef<IValue> elements_) {
    return create(TupleElements(elements_));
  }

  // 创建元组，使用给定的单个元素创建
  static c10::intrusive_ptr<Tuple> create(IValue e1) {
    return c10::make_intrusive<Tuple>(std::move(e1));
  }

  // 创建元组，使用给定的两个元素创建
  static c10::intrusive_ptr<Tuple> create(IValue e1, IValue e2) {
    return c10::make_intrusive<Tuple>(std::move(e1), std::move(e2));
  }

  // 创建元组，使用给定的三个元素创建
  static c10::intrusive_ptr<Tuple> create(IValue e1, IValue e2, IValue e3) {
    return c10::make_intrusive<Tuple>(std::move(e1), std::move(e2), std::move(e3));
  }

 private:
  // 用于处理在模板参数列表中无法使用 ">" 操作符的问题的函数模板
  template <typename... Args>
  static constexpr bool hasMoreThanThreeArgs() {
    return sizeof...(Args) > 3;
  }

 public:
  // 创建元组，使用给定数量的参数创建，支持变长参数列表
  template <typename... Args>
  static c10::intrusive_ptr<Tuple> create(Args&&... elements_) {
  // 根据可变模板参数数量，选择性地创建 Tuple 对象
  switch (sizeof...(Args)) {
    // 如果参数数量为 1、2 或 3，直接创建 Tuple 对象
    case 1:
    case 2:
    case 3:
      return create(IValue(std::forward<Args>(elements_))...);
    // 如果参数数量大于 3，创建包含所有参数的 Tuple 对象，作为一个包含元素的 std::vector
    default:
      return create(
          std::vector<IValue>{IValue(std::forward<Args>(elements_))...});
  }
}

// 再次说明，尽管希望让 Tuple 类不可复制，但是现存的大量代码依赖于复制功能。
// Tuple(const Tuple& rhs) = delete;

// 返回元组的成员变量 elements_ 的引用（const 限定）
const TupleElements& elements() const& {
  return elements_;
}

// 将元组的成员变量 elements_ 以移动方式返回
TupleElements elements() && {
  return std::move(elements_);
}

// 设置元组的成员变量 elements_，接收一个右值引用的 std::vector<IValue>
void setElements(std::vector<IValue>&& elements) {
  elements_.setContents(std::move(elements));
}

// 设置元组的成员变量 elements_，接收一个右值引用的 TupleElements
void setElements(TupleElements&& elements) {
  elements_ = std::move(elements);
}

// 在指定索引处不安全地设置元组的成员变量 elements_
void unsafeSetElement(size_t idx, const IValue& element) {
  elements_[idx] = element;
}

// 在指定索引处不安全地设置元组的成员变量 elements_，接收一个右值引用的 IValue
void unsafeSetElement(size_t idx, IValue&& element) {
  elements_[idx] = std::move(element);
}

// 返回元组的大小，即 elements_ 的大小
size_t size() const {
  return elements_.size();
}

// 返回元组的类型，通过模板 T 确定类型，首次调用时动态创建类型并缓存
template <typename T = c10::TupleType>
std::shared_ptr<T> type() const {
  if (!type_) {
    // 如果类型还未创建，则使用 elements_ 中的数据创建对应的 TupleType
    type_ = TupleTypeFactory<T>::create(fmap(elements(), [&](const IValue& v) {
      return v.type<typename T::ElementType>();
    }));
  }
  // 如果类型可以转换为 T，则返回类型指针；否则使用默认的回退类型
  if (auto t = type_->cast<T>()) {
    return t;
  }
  return TupleTypeFactory<T>::fallback(*type_);
}

// 静态方法，计算 Tuple 的哈希值，使用 elements_ 中的数据计算哈希
static size_t hash(const Tuple& t) {
  return c10::get_hash(t.elements());
}

// 声明友元函数，用于比较两个 Tuple 对象是否相等
TORCH_API friend bool operator==(
    const ivalue::Tuple& lhs,
    const ivalue::Tuple& rhs);

private:
// 注意事项：如果尝试省略默认的 std::shared_ptr<TupleType> type，我们
// 必须将其默认为 nullptr，但实际上静态地知道它不会做任何操作时，
// 我们仍然需要调用 shared_ptr 的（部分）析构函数。
explicit Tuple(std::vector<IValue> elements)
  : elements_(std::move(elements)){}

explicit Tuple(std::vector<IValue> elements, c10::TypePtr type)
  : elements_(std::move(elements)), type_(std::move(type)) {}

explicit Tuple(TupleElements&& elements)
  : elements_(std::move(elements)) {}

explicit Tuple(TupleElements&& elements, std::shared_ptr<TupleType> type)
  : elements_(std::move(elements)), type_(std::move(type)) {}

explicit Tuple(IValue&& e1)
  : elements_(std::move(e1)) {}

explicit Tuple(IValue&& e1, std::shared_ptr<TupleType> type)
  : elements_(std::move(e1)), type_(std::move(type)) {}

explicit Tuple(IValue&& e1, IValue&& e2)
  : elements_(std::move(e1), std::move(e2)) {}

explicit Tuple(IValue&& e1, IValue&& e2, std::shared_ptr<TupleType> type)
  : elements_(std::move(e1), std::move(e2)), type_(std::move(type)) {}

explicit Tuple(IValue&& e1, IValue&& e2, IValue&& e3)
  : elements_(std::move(e1), std::move(e2), std::move(e3)) {}

explicit Tuple(IValue&& e1, IValue&& e2, IValue&& e3, std::shared_ptr<TupleType> type)
    : elements_(std::move(e1), std::move(e2), std::move(e3)), type_(std::move(type)) {}


// 使用成员初始化列表初始化对象的 elements_ 和 type_
// std::move 用于将 e1, e2, e3 和 type 转移赋值给相应的成员变量
// 这里假设 elements_ 是一个包含 e1, e2, e3 的对象，type_ 是一个类型 type 的对象

  friend class c10::intrusive_ptr<Tuple>;


// 声明 Tuple 类为 c10::intrusive_ptr 的友元类
// 这意味着 Tuple 类可以访问 c10::intrusive_ptr 的私有和受保护成员
};

// 声明一个未定义的结构体 Object
struct Object;

// 声明一个未定义的结构体 PyObjectHolder
struct PyObjectHolder;

// 声明一个未定义的结构体 EnumHolder
struct EnumHolder;

// 定义 ivalue 命名空间，用于包含下面的内容
namespace ivalue {

// Future 结构体的定义，继承自 c10::intrusive_ptr_target
struct C10_EXPORT Future final : c10::intrusive_ptr_target {
 private:
  // 构造函数私有化，只能通过 make_intrusive 方法创建 Future 对象，
  // 禁止直接创建未被 intrusive_ptr 持有的 Future 对象
  explicit Future(TypePtr type, std::vector<c10::Device> devices={})
      : type_(std::move(type)),
        impl_(getTypeOfDevices(devices)),
        devices_(sortAndDeduplicateDevices(impl_, std::move(devices))) {}

  friend c10::intrusive_ptr<Future>;

  // FutureCallback 结构体定义
  struct FutureCallback {
    std::function<void(Future&)> callback; // 回调函数
    bool uses_future; // 是否使用传入的 Future& 引用

    // 模板化构造函数，初始化回调函数和使用标志
    template <typename T>
    FutureCallback(T callback, bool uses_future)
        : callback(std::move(callback)), uses_future(uses_future) {}
  };

 public:
  // 禁用拷贝构造函数、移动构造函数、拷贝赋值运算符、移动赋值运算符
  Future(const Future&) = delete;
  Future(Future&&) = delete;
  Future& operator=(const Future&) = delete;
  Future& operator=(Future&&) = delete;

  // FutureError 结构体定义，继承自 std::exception
  struct TORCH_API FutureError final : public std::exception {
    explicit FutureError(std::string&& error_msg_)
        : error_msg(std::move(error_msg_)) {}

    FutureError() = default;

    const char* what() const noexcept override {
      return error_msg.c_str(); // 返回错误消息的 C 字符串形式
    }

    std::string error_msg; // 错误消息字符串
  };

  /**
   * 等待 Future 完成。
   */
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_cv_.wait(lock, [&]() -> bool { return completed_; });
    synchronizeWithCurrentStreams(); // 同步当前 CUDA 流
  }

  /**
   * 等待 Future 完成，并在存在错误时抛出异常。
   */
  void waitAndThrow() {
    wait(); // 等待 Future 完成

    if (eptr_) {
      std::rethrow_exception(eptr_); // 抛出保存的异常指针
    }
  }

  /**
   * 显式标记 Future 完成，并设置输出值。可选地，可以传递所有张量在 IValue 中的存储。
   * 这些存储的 DataPtr 用于同步 CUDA 流。如果未提供 storages，我们将尝试从值中提取，
   * 如果需要（当构造函数传入非空设备集时）。因此，只有在以下情况下需要提供 storages：
   * 1）无法通过 IValue::getSubValues() 或 Python 对象的 pickling 提取它们；
   * 2）自定义存储提取效率更高。
   */
  using WeakStorage = c10::weak_intrusive_ptr<c10::StorageImpl>;
  void markCompleted(
      IValue value,
      std::optional<std::vector<WeakStorage>> storages = c10::nullopt) {
    // 首先执行所有可能引发异常的步骤，然后再设置任何字段。
    // 在获取互斥锁之前执行此操作，因为 extractStorages 可能会获取 GIL，
    // 这可能导致互斥锁反转。参见 https://github.com/pytorch/pytorch/issues/58239.
    std::vector<WeakStorage> actualStorages;
    std::vector<c10::Device> usedDevices;
    try {
      // FIXME We should always extract DataPtrs, in order to catch the case of
      // users using CUDA values but forgetting to set devices, which currently
      // leads to a silent synchronization/correctness issue. However, as this
      // might worsen perf in CPU-only cases, we should only do so after careful
      // benchmarks.
      // 如果实现类型不是 CPU，我们应该始终提取 DataPtrs，以便捕捉用户使用 CUDA 值但忘记设置设备的情况，
      // 这种情况目前会导致潜在的同步/正确性问题。然而，在只有 CPU 的情况下，这样做可能会降低性能，
      // 因此我们应该在仔细进行基准测试之后再决定是否这样做。
      if (impl_.type() != c10::kCPU) {
        actualStorages =
            storages.has_value() ? std::move(*storages) : extractStorages(value);
        usedDevices = getDevicesOfStorages(impl_, actualStorages);
        ensureIsSubsetOfDevices(usedDevices, devices_);
      }
    } catch (const std::exception&) {
      setError(std::current_exception());
      return;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    TORCH_CHECK(
        !completed(),
        "Attempting to mark a completed Future as complete again. Note that "
        "a Future can only be marked completed once.");

    // Only set value_ and completed_ flag once all checks and preparation steps
    // have returned successfully to allow for proper error propagation.
    // 只有在所有检查和准备步骤成功返回后，才设置 value_ 和 completed_ 标志，
    // 以确保正确地传播错误。
    value_ = std::move(value);
    completed_ = true;

    currentDevice_ = impl_.getDevice();
    storages_ = std::move(actualStorages);
    for (const c10::Device& device : usedDevices) {
      c10::Event event(impl_.type());
      event.record(impl_.getStream(device));
      events_.push_back(std::move(event));
    }

    std::vector<FutureCallback> cbs;
    cbs.swap(callbacks_);
    lock.unlock();

    finished_cv_.notify_all();
    for (auto& callback : cbs) {
      invokeCallback(std::move(callback.callback), callback.uses_future);
    }
  }

  // 标记当前 Future 为完成状态
  void markCompleted() {
    markCompleted(IValue{});
  }

  // 设置 Future 的错误状态
  void setError(std::exception_ptr eptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    setErrorInternal(std::move(eptr), lock);
  }

  // 如果需要，设置 Future 的错误状态
  void setErrorIfNeeded(std::exception_ptr eptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed_) {
      // This should be rare and shouldn't cause log spew. Its important to
      // log errors and thats why we have this log here.
      // 这种情况应该很少见，并且不应导致大量日志输出。记录错误非常重要，这就是为什么我们在这里记录日志。
      std::string msg = c10::str(
          "Skipping setting following error on the Future since "
          "it is already marked completed (this is not necessarily "
          "an error):\n",
          tryRetrieveErrorMessageInternal(std::move(eptr)));
      if (eptr_) {
        msg += c10::str(
            ", \nOriginal exception:\n",
            tryRetrieveErrorMessageInternal(eptr_));
      }
      LOG(INFO) << msg;
      return;
    } else {
      setErrorInternal(std::move(eptr), lock);
    }
  }

  // 获取当前 Future 的结果
  IValue value() {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(completed());
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
    return value_;
  }

  // 如果我们知道 Future 已完成且没有错误，才能使用此访问器
  const IValue& constValue() const {
  // 使用互斥锁对共享资源进行加锁，确保线程安全访问
  std::unique_lock<std::mutex> lock(mutex_);
  // 确保当前 future 已完成
  AT_ASSERT(completed());
  // 确保当前 future 没有异常
  TORCH_INTERNAL_ASSERT(
    !eptr_,
    "value() accessor should only be used when future is not completed with ",
    "an error, but future had the following error: ",
    tryRetrieveErrorMessageInternal(eptr_)
  );
  // 返回当前 future 的值
  return value_;
}

// 仅当确定 future 已完成且没有错误时才使用此访问器
const std::vector<WeakStorage>& storages() const {
  // 使用互斥锁对共享资源进行加锁，确保线程安全访问
  std::unique_lock<std::mutex> lock(mutex_);
  // 确保当前 future 已完成
  AT_ASSERT(completed());
  // 确保当前 future 没有异常
  AT_ASSERT(!eptr_);
  // 返回存储的数据结构
  return storages_;
}

/**
 * 向 future 添加回调函数。
 * 当 future 完成时，这些回调将被执行。
 * 如果 future 已经完成，则立即执行回调。
 */
template <typename T>
void addCallback(T callback, bool uses_future = true) {
  // 使用互斥锁对共享资源进行加锁，确保线程安全操作
  std::unique_lock<std::mutex> lock(mutex_);
  // 如果 future 已经完成
  if (completed()) {
    lock.unlock();
    // 调用回调函数
    invokeCallback(std::move(callback), uses_future);
    return;
  }
  // 将回调函数加入到队列中
  callbacks_.emplace_back(std::move(callback), uses_future);
}

/**
 * 向 future 添加回调函数，并返回另一个 Future 以持有回调的返回值。
 * 当回调完成时，需要另一个 Future 来保存返回值。
 */
template <typename T>
c10::intrusive_ptr<Future> then(T callback, TypePtr type) {
  // 确保回调函数的签名符合要求
  using IValueWithStorages = std::tuple<IValue, std::vector<WeakStorage>>;
  static_assert(
      std::disjunction<
          std::is_invocable_r<IValue, T, Future&>,
          std::is_invocable_r<IValueWithStorages, T, Future&>>::value,
      "The callback must have signature IValue(Future&) or "
      "std::tuple<IValue, std::vector<Storage>>(Future&)");

  // 创建子 Future 实例
  auto childFut = createInstance(::std::move(type));
  // 添加回调函数到当前 future
  addCallback([childFut,
               cb = std::move(callback)](Future& parentFut) mutable {
    try {
      // 根据回调函数返回类型的不同处理不同的情况
      if constexpr (::std::is_convertible_v<typename std::invoke_result_t<T &&, Future&>, IValueWithStorages>) {
        auto [ivalue, storages] = cb(parentFut);
        childFut->markCompleted(::std::move(ivalue), ::std::move(storages));
      } else {
        childFut->markCompleted(cb(parentFut));
      }
    } catch (std::exception&) {
      // 如果发生异常，设置子 Future 的错误状态
      childFut->setError(std::current_exception());
    }
  });
  // 返回子 Future
  return childFut;
}

template <typename T>
c10::intrusive_ptr<Future> thenAsync(T callback, TypePtr type) {
  // 确保回调函数的签名符合要求
  static_assert(
      std::is_invocable_r<c10::intrusive_ptr<Future>, T, Future&>::value,
      "The callback must have signature c10::intrusive_ptr<Future>(Future&)");

  // 创建子 Future 实例
  auto childFut = createInstance(std::move(type));
    // 添加回调函数到当前 Future 对象
    addCallback(
        // 使用移动语义将 childFut 和 callback 传递给 lambda 表达式
        [childFut, cb = std::move(callback)](Future& parentFut) mutable {
          // 声明中间 Future 指针
          c10::intrusive_ptr<Future> intermediateFut;
          try {
            // 调用回调函数 cb 处理 parentFut，获取中间 Future
            intermediateFut = cb(parentFut);
          } catch (std::exception&) {
            // 如果捕获到异常，设置 childFut 的错误状态并返回
            childFut->setError(std::current_exception());
            return;
          }
          // 给中间 Future 添加回调函数
          intermediateFut->addCallback(
              // 使用移动语义将 childFut 传递给 lambda 表达式
              [childFut = std::move(childFut)](Future& intermediateFut) {
                // 检查中间 Future 是否有错误
                if (intermediateFut.hasError()) {
                  // 如果有错误，设置 childFut 的错误状态
                  childFut->setError(intermediateFut.exception_ptr());
                } else {
                  // 如果没有错误，标记 childFut 已完成，并使用中间 Future 的值和存储
                  childFut->markCompleted(
                      intermediateFut.value(), intermediateFut.storages());
                }
              });
        });
    // 返回 childFut 对象
    return childFut;
  }

  // 尝试从 std::exception_ptr 中检索错误消息
  std::string tryRetrieveErrorMessage() const {
    TORCH_CHECK(hasError(), "No error present on the future.");
    std::unique_lock<std::mutex> lock(mutex_);
    return tryRetrieveErrorMessageInternal(eptr_);
  }

  // 检查当前 Future 是否已完成
  bool completed() const {
    return completed_;
  }

  // 检查当前 Future 是否有值
  bool hasValue() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return completed_ && !eptr_;
  }

  // 检查当前 Future 是否有错误
  bool hasError() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return eptr_ ? true : false;
  }

  // 返回异常指针 eptr_
  std::exception_ptr exception_ptr() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return eptr_;
  }

  // 友元函数，用于输出 Future 对象
  TORCH_API friend std::ostream& operator<<(
      std::ostream& out,
      const Future& v);

  // 返回 Future 对象的元素类型
  const TypePtr& elementType() const {
    return type_;
  }

  // 返回 Future 对象的设备列表
  const std::vector<c10::Device>& devices() const {
    return devices_;
  }

  // 创建一个新的 Future 实例，用于自定义 then() 方法的实现
  c10::intrusive_ptr<Future> createInstance(at::TypePtr type) {
    return c10::make_intrusive<Future>(std::move(type), devices_);
  }

 private:

  // 当调用回调时，确保环境正确设置的方法
  // 例如设置 CUDA 流、同步值等操作
  template<typename T>
  void invokeCallback(T callback, bool uses_future) {
    static_assert(
        std::is_invocable_r<void, T, Future&>::value,
        "The callback must have signature void(Future&)");

    // 当回调不使用 Future 时，下面的同步操作不应该执行
    if (uses_future) {
      // 如果使用了 future，则创建一个 OptionalDeviceGuard 对象，用于设置当前设备
      c10::OptionalDeviceGuard deviceGuard(currentDevice_);

      // 创建一个流的向量，预留足够的空间以容纳所有设备的流
      std::vector<c10::Stream> streams;
      streams.reserve(devices_.size());
      // 遍历设备列表，为每个设备获取对应的流并添加到流的向量中
      for (const c10::Device& device : devices_) {
        streams.push_back(impl_.getStreamFromGlobalPool(device));
      }
      // 创建 MultiStreamGuard 对象，用于管理多个流的当前状态
      c10::MultiStreamGuard streamGuard(streams);
      // 同步当前的 CUDA 流
      synchronizeWithCurrentStreams();
      // 调用回调函数，将当前对象作为参数传递给回调函数
      callback(*this);
    } else {
      // 如果未使用 future，则直接调用回调函数，将当前对象作为参数传递给回调函数
      callback(*this);
    }
  }

  // 此方法应在使用此 future 的值之前调用，确保在调用点处的 CUDA 流与值正确同步
  void synchronizeWithCurrentStreams() {
    // 遍历事件列表，对每个事件阻塞在相应的流上
    for (c10::Event& event : events_) {
      event.block(impl_.getStream(event.device()));
    }

    // 遍历存储器列表，对每个存储器的数据指针进行记录，如果存储器不在 CPU 上
    for (const WeakStorage& weak_storage : storages_) {
      c10::intrusive_ptr<c10::StorageImpl> storage = weak_storage.lock();
      if (!storage) {
        continue;
      }
      if (!storage->device().is_cpu()) {
        // 在存储器对应的流上记录存储器的数据指针
        impl_.recordDataPtrOnStream(
            storage->data_ptr(), impl_.getStream(storage->device()));
      }
    }
  }

  // 设置内部错误状态，并在给定的锁上解锁
  void setErrorInternal(
      std::exception_ptr eptr,
      std::unique_lock<std::mutex>& lock) {
    // 检查当前 future 是否已经设置了错误状态
    TORCH_CHECK(
        !eptr_,
        "Error already set on this Future: ",
        tryRetrieveErrorMessageInternal(eptr_),
        ", trying to set error: ",
        tryRetrieveErrorMessageInternal(eptr));
    // 内部断言，确保 future 尚未完成
    TORCH_INTERNAL_ASSERT(!completed(), "Future is already marked completed");
    // 标记 future 已完成
    completed_ = true;
    // 将异常指针移动到成员变量中
    eptr_ = std::move(eptr);

    // 交换回调函数的向量，清空当前回调函数列表，并释放锁
    std::vector<FutureCallback> cbs;
    cbs.swap(callbacks_);
    lock.unlock();

    // 通知所有等待的线程条件变量已完成
    finished_cv_.notify_all();
    // 遍历回调函数列表，并调用每个回调函数
    for (auto& callback : cbs) {
      invokeCallback(std::move(callback.callback), callback.uses_future);
    }
  }

  // 尝试从 std::exception_ptr 中获取错误消息
  std::string tryRetrieveErrorMessageInternal(std::exception_ptr eptr) const {
    try {
      std::rethrow_exception(std::move(eptr));
    } catch (const std::exception& e) {
      // 捕获到标准异常，返回异常的 what() 方法的结果
      return e.what();
    } catch (...) {
      // 捕获到未知异常类型，返回固定的错误消息
      return "Unknown Exception Type";
    }
  }

  // 在 ivalue.cpp 文件中定义
  static std::vector<WeakStorage> extractStorages(
      const at::IValue& value);

  // 从存储器列表中提取设备列表
  static std::vector<c10::Device> getDevicesOfStorages(
      const c10::impl::VirtualGuardImpl& impl,
      const std::vector<WeakStorage>& storages) {
    // 获取虚拟保护实现中的设备数量
    c10::DeviceIndex deviceCount = impl.deviceCount();
    // 创建布尔向量，表示每个设备是否已使用
    std::vector<bool> isDeviceUsed(deviceCount, false);
    // 遍历存储列表中的每个弱引用存储对象
    for (const WeakStorage& weak_storage : storages) {
      // 尝试获取弱引用指向的存储对象，转换为指针类型
      c10::intrusive_ptr<c10::StorageImpl> storage = weak_storage.lock();
      // 如果存储对象为空指针，则跳过当前循环
      if (!storage) {
        continue;
      }
      // 获取存储对象的设备信息
      c10::Device device = storage->device();
      // 如果设备不是 CPU 设备
      if (!device.is_cpu()) {
        // 检查设备类型是否与目标类型一致，如果不一致则抛出错误信息
        TORCH_CHECK_VALUE(
            device.type() == impl.type(),
            "Expected all data ptrs to be on a device of type ",
            impl.type(),
            ", got one on device ",
            device);
        // 标记使用了该设备
        isDeviceUsed[device.index()] = true;
      }
    }
    // 声明设备列表
    std::vector<c10::Device> devices;
    // 遍历设备使用标记列表
    for (c10::DeviceIndex idx = 0; idx < deviceCount; idx++) {
      // 如果当前设备被使用
      if (isDeviceUsed[idx]) {
        // 将该设备添加到设备列表中
        devices.emplace_back(impl.type(), idx);
      }
    }
    // 返回筛选后的设备列表
    return devices;
  }

  // 格式化一组设备列表的输出字符串
  static std::string formatSetOfDevices(
      const std::vector<c10::Device>& devices) {
    // 如果设备列表为空，返回"(none)"
    if (devices.empty()) {
      return "(none)";
    }
    // 创建输出流对象
    std::ostringstream oss;
    // 将第一个设备信息输出到流中
    oss << devices[0];
    // 遍历剩余设备列表
    for (const auto idx : c10::irange(1, devices.size())) {
      // 如果是最后一个设备
      if (idx == devices.size() - 1) {
        oss << " and ";
      } else {
        oss << ", ";
      }
      // 将设备信息输出到流中
      oss << devices[idx];
    }
    // 返回流中的字符串表示
    return oss.str();
  }

  // 获取一组设备列表的设备类型
  static c10::DeviceType getTypeOfDevices(
      const std::vector<c10::Device>& devices) {
    // 如果设备列表为空，返回 CPU 设备类型
    if (devices.empty()) {
      return c10::kCPU;
    }
    // 获取第一个设备的类型
    c10::DeviceType deviceType = devices[0].type();
    // 检查所有设备类型是否一致
    for (const auto idx : c10::irange(1, devices.size())) {
      // 如果设备类型不一致，则抛出错误信息
      TORCH_CHECK_VALUE(
          devices[idx].type() == deviceType,
          "Expected all devices to be of the same type, but got a mismatch between ",
          devices[0],
          " and ",
          devices[idx]);
    }
    // 返回设备类型
    return deviceType;
  }

  // 对设备列表进行排序和去重处理
  // 由于需要使用 ensureIsSubsetOfDevices 函数，设备列表需要先排序
  static std::vector<c10::Device> sortAndDeduplicateDevices(
      const c10::impl::VirtualGuardImpl& /*impl*/,
      std::vector<c10::Device> devices) {
    // 对设备列表按索引进行排序
    std::sort(
      devices.begin(), devices.end(),
      [](const c10::Device& a, const c10::Device& b) { return a.index() < b.index(); });
    // 通过压缩去重设备列表
    size_t targetIdx = 0;
    for (const auto sourceIdx : c10::irange(devices.size())) {
      // 检查设备是否具有有效索引
      TORCH_CHECK_VALUE(
          devices[sourceIdx].has_index(),
          "Expected devices to have indices, got ", devices[sourceIdx]);
      // 如果是重复设备，则跳过
      if (targetIdx > 0 && devices[targetIdx - 1].index() == devices[sourceIdx].index()) {
        continue;
      }
      // 将不重复的设备移动到目标位置
      if (sourceIdx != targetIdx) {
        devices[targetIdx] = devices[sourceIdx];
      }
      // 更新目标位置索引
      targetIdx++;
    }
    // 如果有重复设备，则截断列表
    devices.resize(targetIdx, c10::Device(c10::kCPU));
    // 返回处理后的设备列表
    return devices;
  }
  // 返回 devices 数组
  return devices;
}

static void ensureIsSubsetOfDevices(
    const std::vector<c10::Device>& subset,
    const std::vector<c10::Device>& superset) {
  // 我们假设这两个向量中的设备具有相同的一致类型，它们的索引是唯一的并且已排序。
  std::vector<c10::Device> excessDevices;
  // 计算 subset 中存在但在 superset 中不存在的设备
  std::set_difference(
      subset.begin(),
      subset.end(),
      superset.begin(),
      superset.end(),
      std::back_inserter(excessDevices),
      [](const c10::Device& a, const c10::Device& b) { return a.index() < b.index(); });
  // 检查是否存在多余的设备，并给出相应的错误信息
  TORCH_CHECK_VALUE(
      excessDevices.empty(),
      "The result contained tensors residing on device(s) ",
      formatSetOfDevices(excessDevices),
      " which are not among the expected device(s) ",
      formatSetOfDevices(superset));
}

mutable std::mutex mutex_;
std::atomic_bool completed_ = {false}; // 表示此 future 是否完成

std::condition_variable finished_cv_;

IValue value_; // future 完成时的值
TypePtr type_;
std::vector<FutureCallback> callbacks_;
std::exception_ptr eptr_;

// 一个向上转型的指针，指向一个虚拟类，允许我们以一种通用的方式处理事件、流等，而不需要显式依赖于 CUDA。
// NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
const c10::impl::VirtualGuardImpl impl_;

// 当调用 markCompleted 时当前的设备，我们在调用回调时将其恢复。这是可选的，因为仅当 future 成功完成时才会存储它。
optional<c10::Device> currentDevice_;

// 异步 I/O 内核完成时对应的事件。它们在 future 被标记为完成时记录在适当的流上，然后可以查询/等待/阻塞它们。
// 每个值的张量所在的设备都有一个事件。
std::vector<c10::Event> events_;

// 当 future 第一次被标记为完成时，从值中提取的存储的缓存版本。
std::vector<WeakStorage> storages_;

// 此 future 及其任何子 future 允许使用的设备的边界集合。
// 这是上述事件使用设备集合的超集。我们需要这个信息来知道在调用回调时设置哪些设备的流，从而允许回调使用父 future 没有使用的设备。
// NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
const std::vector<c10::Device> devices_;
};

// ivalue 命名空间中的 Await 结构体定义，继承自 c10::intrusive_ptr_target
struct C10_EXPORT ivalue::Await final : c10::intrusive_ptr_target {
 private:
  // 私有构造函数，用于有回调函数的情况下创建 Await 对象
  explicit Await(TypePtr elType, std::function<IValue()> fn)
      : elType_(std::move(elType)), type_(AwaitType::create(elType_)), fn_(std::move(fn)) {}

  // 私有构造函数，用于没有回调函数的情况下创建 Await 对象
  explicit Await(TypePtr elType) : elType_(std::move(elType)), type_(AwaitType::create(elType_)) { }

  // 声明 c10::intrusive_ptr<Await> 为友元，允许访问私有构造函数
  friend c10::intrusive_ptr<Await>;

 public:
  // 禁用拷贝和移动构造函数以及赋值操作符
  Await(const Await&) = delete;
  Await(Await&&) = delete;
  Await& operator=(const Await&) = delete;
  Await& operator=(Await&&) = delete;

  // 等待异步操作完成，并返回结果值
  IValue wait() {
    if (!completed_) {
      TORCH_CHECK(fn_, "Incompleted Await: fn can't be None");
      value_ = fn_();
      completed_ = true;
      args_ = {};
    }
    return value_;
  }

  // 返回已完成的异步操作的值，若未完成则抛出异常
  IValue value() {
    TORCH_CHECK(completed_, "Await must be completed");
    return value_;
  }

  // 设置异步操作的回调函数
  void setFn(std::function<IValue()> fn) {
    fn_ = std::move(fn);
  }

  // 检查异步操作是否已完成
  bool completed() {
    return completed_;
  }

  // 标记异步操作已完成，并设置其返回值
  void markCompleted(IValue value) {
    value_ = std::move(value);
    completed_ = true;
  }

  // 友元声明，允许将 Await 对象输出到流中
  TORCH_API friend std::ostream& operator<<(
      std::ostream& out,
      const Await& v);

  // 返回元素类型的 TypePtr
  const TypePtr& elementType() const {
    return elType_;
  }

  // 返回 Await 对象的类型 TypePtr
  const TypePtr& type() const {
    return type_;
  }

  // 设置异步操作的参数列表
  void setArgs(std::vector<IValue> args) {
    args_ = std::move(args);
  }

  // 返回异步操作的参数列表的引用
  std::vector<IValue>& args() {
    return args_;
  }

 private:
  TypePtr elType_;                  // 元素类型的 TypePtr
  TypePtr type_;                    // Await 对象的类型 TypePtr
  std::vector<IValue> args_;        // 参数列表
  std::function<IValue()> fn_;      // 异步操作的回调函数
  IValue value_;                    // 异步操作的返回值
  bool completed_{};                // 异步操作是否已完成的标志
};

// 输入为相同目标类型的 Future 列表，输出为已完成 Future 列表的 Future
TORCH_API intrusive_ptr<ivalue::Future> collectAll(
    const c10::List<c10::intrusive_ptr<ivalue::Future>>& srcs);

// 输入为相同目标类型的 Future 列表，输出为将更新为已看到的值的 Future
TORCH_API intrusive_ptr<ivalue::Future> collectAny(
    const c10::List<c10::intrusive_ptr<ivalue::Future>>& srcs);

// ivalue 命名空间中的 Object 结构体定义，继承自 c10::intrusive_ptr_target
struct C10_EXPORT ivalue::Object final : c10::intrusive_ptr_target {
 public:
  // 构造函数，创建一个 Object 对象，持有一个类型的弱或强引用指针和指定数量的槽位
  Object(WeakOrStrongTypePtr type, size_t numSlots) : type_(std::move(type)) {
    slots_.resize(numSlots);
  }

  // 构造函数，创建一个 Object 对象，持有一个类型的强引用指针和指定数量的槽位
  Object(StrongTypePtr type, size_t numSlots)
      : type_(WeakOrStrongTypePtr(std::move(type))) {
    slots_.resize(numSlots);
  }

  // 静态工厂方法，创建一个 Object 对象，持有一个类型的弱或强引用指针和指定数量的槽位
  static c10::intrusive_ptr<Object> create(
      WeakOrStrongTypePtr type,
      size_t numSlots) {
      // In general, class types hold a shared_ptr to its owning CompilationUnit,
      // so that its type and methods do not get deallocated while the class exists.
      // However, the CompilationUnit holds ownership of the type's graphs, so
      // inserting a constant object into a Graph would create a reference cycle if
      // that constant object held a shared_ptr to its CU. For these objects we
      // instatiate them with non-owning references to its CU
  }

 private:
  WeakOrStrongTypePtr type_;       // 弱或强引用的类型指针
  std::vector<IValue> slots_;      // 槽位数组
};
  }

  /**
   * 使用给定类型和槽位数量创建一个新对象。
   */
  static c10::intrusive_ptr<Object> create(
      StrongTypePtr type,
      size_t numSlots) {
    return c10::make_intrusive<Object>(std::move(type), numSlots);
  }

  /**
   * 使用给定类类型和槽位数量创建一个新对象。
   */
  static c10::intrusive_ptr<Object> create(ClassTypePtr classType, size_t numSlots);

  /**
   * 槽位 API。
   *
   * 属性以简单的向量形式存储，以便运行时查找速度快。
   * "槽位"只是向量的索引，如果有类类型的访问权限，可以静态计算。
   * 如果你在编写编译器相关的内容，请使用这个 API。
   */
  void setSlot(size_t slot, IValue v) {
    if (slot >= slots_.size()) {
      // 对于模块类型，对象创建后可能会扩展类成员。在这种情况下，我们会调整槽位大小以匹配扩展后的长度。
      resizeObject(slot);
    }
    slots_[slot] = std::move(v);
  }

  /**
   * 获取指定槽位的值。
   *
   * 注意：这个查找操作非常频繁，因此我们使用未检查的访问来访问向量。可以使用 ASan 检测到错误。
   */
  const IValue& getSlot(size_t slot) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(slot < slots_.size());
    // 注意：这个查找操作非常频繁，因此我们使用未检查的访问来访问向量。可以使用 ASan 检测到错误。
    return slots_[slot];
  }

  /**
   * 不安全地移除指定槽位的值。
   *
   * 调用者需要确保操作的安全性。
   */
  void unsafeRemoveSlot(size_t slot) {
    TORCH_CHECK(slot < slots_.size());
    slots_.erase(slots_.begin() + static_cast<std::ptrdiff_t>(slot));
  }

  /**
   * 属性 API。
   *
   * 封装了槽位操作，以便用户可以直接访问属性。
   * 如果你是用户，请使用这个 API。
   *
   * 注意：与 Python 不同，TorchScript 必须区分属性（IValues）和方法（Methods）。
   * 如果需要方法，请使用 `obj.type()->getMethod()`。
   */
  IValue getAttr(const std::string& name) const;

  /**
   * 设置指定名称的属性值。
   */
  void setAttr(const std::string& name, IValue v);

  /**
   * 不安全地移除指定名称的属性。
   *
   * 调用者需负责此操作的安全性。
   * 我们没有在类型中移除属性，因为类型可能被多个对象共享。
   * 因此，在移除属性后，对象处于不一致状态，有更多的属性类型而槽位数量不匹配。
   * 用户需要确保通过在类型中也移除属性使对象保持一致。
   */
  void unsafeRemoveAttr(const std::string& name);

  /**
   * 返回对象的名称。
   */
  std::string name() const;

  /**
   * 返回对象的类型。
   */
  std::shared_ptr<ClassType> type() const;

  /**
   * 返回对象关联的编译单元。
   *
   * 如果对象关联的类型持有强引用，则返回其关联的编译单元。
   * 否则，返回关联编译单元的弱引用。
   */
  std::shared_ptr<torch::jit::CompilationUnit> compilation_unit() {
    if (type_.holds_strong_ref()) {
      return type_.cu_.getStrongRefOrThrow();
    } else {
      auto weak_ptr = type_.cu_.getWeakRefOrThrow();
      return std::shared_ptr<torch::jit::CompilationUnit>(weak_ptr);
    }
  }

  /**
   * 将对象复制为弱引用的编译单元引用。
   */
  c10::intrusive_ptr<Object> copy_to_weak_compilation_ref() const;

  /**
   * 不安全地将对象转换为弱引用的编译单元引用。
   */
  void unsafe_make_weak_compilation_ref() {
  // 将 type_ 转换为弱类型指针或强类型指针，并赋值给 type_
  type_ = WeakOrStrongTypePtr(type_.asWeakTypePtr());
}

// 声明一个返回类型为 intrusive_ptr<Object> 的成员函数 copy
c10::intrusive_ptr<Object> copy() const;

// 声明一个返回类型为 intrusive_ptr<Object> 的成员函数 deepcopy，可选地指定设备
c10::intrusive_ptr<Object> deepcopy(
    std::optional<at::Device> device = c10::nullopt) const;

// 声明一个返回类型为 intrusive_ptr<Object> 的成员函数 deepcopy，接受 memo 和可选的设备参数
c10::intrusive_ptr<Object> deepcopy(
    IValue::HashIdentityIValueMap& memo,
    std::optional<at::Device> device = c10::nullopt) const;

// 检查对象是否是弱编译引用
bool is_weak_compilation_ref() const {
  return !type_.holds_strong_ref();
}

// 检查对象是否是空的强编译引用
bool is_empty_strong_compilation_ref() const {
  return type_.holds_empty_strong_ref();
}

private:
// 私有成员函数：调整对象大小，接受一个 slot 大小的参数
void resizeObject(size_t slot);

// 弱或强类型指针，用于指向对象的类型信息
WeakOrStrongTypePtr type_;

// 存储对象的数据值的容器
std::vector<IValue> slots_;
};

// virtual ivalue PyObjectHolder that hold a py::object, we make this virtual
// because the py::object and refcounting logic should happen in libtorch_python
// see concrete implementation in python_ivalue.h
// 定义一个虚拟的 ivalue::PyObjectHolder 类，用于保存 py::object 对象，
// 将其设计为虚拟类是因为 py::object 和引用计数逻辑应该在 libtorch_python 中实现，
// 具体实现可见 python_ivalue.h

struct ivalue::PyObjectHolder : c10::intrusive_ptr_target {
 public:
  // 获取持有的 PyObject 指针
  virtual PyObject* getPyObject() = 0;
  // 尝试推断类型
  virtual c10::InferredType tryToInferType() = 0;
  // 转换为指定类型的 IValue
  virtual IValue toIValue(const TypePtr& type, std::optional<int32_t> N = c10::nullopt) = 0;
  // 转换为字符串表示
  virtual std::string toStr() = 0;
  // 提取其中的张量对象
  virtual std::vector<at::Tensor> extractTensors() = 0;

  // 虚析构函数
  ~PyObjectHolder() override = default;
};

struct ivalue::EnumHolder : c10::intrusive_ptr_target {
 public:
  // 构造函数，初始化枚举类型、名称和值
  EnumHolder(std::shared_ptr<EnumType> type, std::string name, IValue value)
      : type_(std::move(type)),
        name_(std::move(name)),
        value_(std::move(value)) {}

  // 判断两个 EnumHolder 对象是否相等
  bool is(const ivalue::EnumHolder& rhs) {
    return *this == rhs;
  }

  // 友元函数重载相等操作符
  friend bool operator==(
      const ivalue::EnumHolder& lhs,
      const ivalue::EnumHolder& rhs);

  // 友元函数，输出 EnumHolder 对象信息到流中
  TORCH_API friend std::ostream& operator<<(
      std::ostream& out,
      const ivalue::EnumHolder& v);

  // 获取类的限定名
  TORCH_API const std::string& qualifiedClassName() const;

  // 获取类的非限定名
  const std::string& unqualifiedClassName() const;

  // 获取枚举值的名称
  const std::string& name() const {
    return name_;
  }

  // 获取枚举值的 IValue 对象
  const IValue& value() const {
    return value_;
  }

  // 获取枚举类型的 shared_ptr
  std::shared_ptr<EnumType> type() const {
    return type_;
  }

 private:
  std::shared_ptr<EnumType> type_; // 枚举类型指针
  std::string name_; // 枚举名称
  IValue value_; // 枚举值
};

#undef TORCH_FORALL_TAGS

namespace detail {

// 用于条件编译的无符号长整型别名选择
struct _guarded_unsigned_long_unique_dummy final {
  _guarded_unsigned_long_unique_dummy(int64_t){};
};
using _guarded_unsigned_long = std::conditional_t<
    std::is_same_v<unsigned long, uint32_t> ||
        std::is_same_v<unsigned long, uint64_t>,
    _guarded_unsigned_long_unique_dummy,
    unsigned long>;

} // namespace detail

// 返回 IValue 对象的对象引用
inline ivalue::Object& IValue::toObjectRef() const {
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(), "Attempted to create null reference");
  return *static_cast<c10::ivalue::Object*>(payload.u.as_intrusive_ptr);
}

// 注意：在此处添加 DEFINE_TO 宏定义时，需要在 IValue 类中添加对应的 toX 方法。
// 这些命名的方法比模板函数更易于发现。

#define DEFINE_TO(T, method_name)                          \
  template <>                                              \
  inline T IValue::to<T>()&& {                             \
    return static_cast<T>(std::move(*this).method_name()); \
  }                                                        \
  template <>                                              \
  inline c10::detail::ivalue_to_const_ref_overload_return<T>::type IValue::to<T>() const& { \
    typedef c10::detail::ivalue_to_const_ref_overload_return<T>::type return_type;          \
    // 返回方法名调用的结果，并进行静态类型转换为指定的返回类型
    return static_cast<return_type>(this->method_name());                                   \
  }
// 定义宏DEFINE_TO，用于为不同类型定义类型转换函数，如将at::Tensor转换为toTensor
DEFINE_TO(at::Tensor, toTensor)
DEFINE_TO(at::Storage, toStorage)
DEFINE_TO(c10::Stream, toStream)
DEFINE_TO(float, toDouble)
DEFINE_TO(double, toDouble)
DEFINE_TO(c10::complex<double>, toComplexDouble)
DEFINE_TO(unsigned char, toInt)
DEFINE_TO(signed char, toInt)
DEFINE_TO(unsigned short, toInt)
DEFINE_TO(short, toInt)
DEFINE_TO(int, toInt)
DEFINE_TO(uint32_t, toInt)
DEFINE_TO(uint64_t, toInt)
DEFINE_TO(detail::_guarded_unsigned_long, toInt)
DEFINE_TO(int64_t, toInt)
DEFINE_TO(bool, toBool)
DEFINE_TO(c10::intrusive_ptr<caffe2::Blob>, toBlob);
DEFINE_TO(c10::intrusive_ptr<ivalue::ConstantString>, toString)
DEFINE_TO(c10::intrusive_ptr<ivalue::Object>, toObject)
DEFINE_TO(at::Scalar, toScalar)
DEFINE_TO(c10::List<int64_t>, toIntList)
DEFINE_TO(c10::List<double>, toDoubleList)
DEFINE_TO(c10::List<c10::complex<double>>, toComplexDoubleList)
DEFINE_TO(c10::List<bool>, toBoolList)
DEFINE_TO(c10::List<at::Tensor>, toTensorList)
DEFINE_TO(c10::impl::GenericList, toList)
DEFINE_TO(c10::impl::GenericDict, toGenericDict)
DEFINE_TO(c10::intrusive_ptr<ivalue::Tuple>, toTuple)
DEFINE_TO(std::string, toStringRef)
DEFINE_TO(c10::string_view, toStringView)
DEFINE_TO(c10::intrusive_ptr<ivalue::Future>, toFuture)
DEFINE_TO(c10::intrusive_ptr<ivalue::Await>, toAwait)
DEFINE_TO(c10::intrusive_ptr<c10::RRefInterface>, toRRef)
DEFINE_TO(c10::intrusive_ptr<at::Quantizer>, toQuantizer)
DEFINE_TO(IValue, toIValue)
DEFINE_TO(c10::Device, toDevice)
DEFINE_TO(at::ScalarType, toScalarType)
DEFINE_TO(at::Layout, toLayout)
DEFINE_TO(at::MemoryFormat, toMemoryFormat)
DEFINE_TO(at::QScheme, toQScheme)
DEFINE_TO(at::Dimname, toDimname)
DEFINE_TO(at::Generator, toGenerator)
DEFINE_TO(c10::SymInt, toSymInt)
DEFINE_TO(c10::SymFloat, toSymFloat)
DEFINE_TO(c10::SymBool, toSymBool)

template <class T>
struct _fake_type {};

// generic_to<T> converts an IValue from a generic list or generic dict
// to a concrete list/dict type likelike List<T>, Dict<...> or optional<T>.
// Note that in the case of lists, this only works for IValue-based lists,
// i.e. not for int64_t, double, ...
// generic_to<T> is an implementation detail of IValue::to<T> and not
// supposed to be called directly.
// The _fake_type<T> parameter allows us to overload
// based on the return type.
template <class Elem>
// TODO this is deprecated but we don't throw a warning because a lot of ops in
// native_functions.yaml still return std::vector.
// C10_DEPRECATED_MESSAGE("IValues based on std::vector<T> are potentially slow
// and deprecated. Please use torch::List<T> instead.")
// generic_to<T>函数定义，用于将泛型的IValue转换为特定类型的std::vector<Elem>
std::vector<Elem> generic_to(IValue ivalue, _fake_type<std::vector<Elem>>) {
  // We need to do a deep copy of the vector because there might be other
  // references to this same IValue that also use the list. We can't just
  // move the elements out.
  // 将传入的IValue转换为List<Elem>类型的变量list
  auto list = std::move(ivalue).to<List<Elem>>();
  // 创建一个std::vector<Elem>类型的结果变量result，预留足够的空间
  std::vector<Elem> result;
  result.reserve(list.size());
  // 遍历list，将元素逐个移动到result中
  for (Elem v : list) {
    result.push_back(std::move(v));
  }
  // 返回转换后的std::vector<Elem>
  return result;
}
// 转换 IValue 到具体自定义类对象的右值引用版本
template <typename T>
c10::intrusive_ptr<T> IValue::toCustomClass() && {
  // 断言：确保模板参数 T 必须继承自 torch::CustomClassHolder
  static_assert(
      std::is_base_of<torch::CustomClassHolder, T>::value == true,
      "toCustomClass 需要模板参数 T 必须从 torch::CustomClassHolder 继承");

  // 获取对象的普通对象表示
  auto obj = toObject();
  
  // 检查对象的槽的数量是否为 1，否则抛出异常
  TORCH_CHECK(
      obj->slots().size() == 1,
      "尝试将 IValue 转换为自定义类，但它不包含自定义类!");

  // 获取 T 类型的预期类型
  const auto* expected_type = c10::getCustomClassType<c10::intrusive_ptr<T>>().get();
  
  // 检查对象类型是否匹配预期类型
  ivalue::checkCustomClassType(expected_type, type().get());

  // 将对象的第一个槽转换为 T 类型的静态内存引用计数指针
  auto userObj =
      c10::static_intrusive_pointer_cast<T>(obj->getSlot(0).toCapsule());

  // 返回转换后的对象
  return userObj;
}

// 转换 IValue 到具体自定义类对象的常量左值引用版本
template <typename T>
c10::intrusive_ptr<T> IValue::toCustomClass() const& {
  // 断言：确保模板参数 T 必须继承自 torch::CustomClassHolder
  static_assert(
      std::is_base_of<torch::CustomClassHolder, T>::value == true,
      "toCustomClass 需要模板参数 T 必须从 torch::CustomClassHolder 继承");

  // 获取对象的普通对象表示
  auto obj = toObject();
  
  // 检查对象的槽的数量是否为 1，否则抛出异常
  TORCH_CHECK(
      obj->slots().size() == 1,
      "尝试将 IValue 转换为自定义类，但它不包含自定义类!");

  // 获取 T 类型的预期类型
  const auto* expected_type = c10::getCustomClassType<c10::intrusive_ptr<T>>().get();
  
  // 检查对象类型是否匹配预期类型
  ivalue::checkCustomClassType(expected_type, type().get());

  // 将对象的第一个槽转换为 T 类型的静态内存引用计数指针
  auto userObj =
      c10::static_intrusive_pointer_cast<T>(obj->getSlot(0).toCapsule());

  // 返回转换后的对象
  return userObj;
}

// 通用函数：从 IValue 转换为指定类型的对象
template <typename T>
T generic_to(IValue ivalue, _fake_type<T>) {
  // 定义 ElemType 为 T 类型去掉指针后的元素类型
  using ElemType = typename std::remove_pointer<T>::type::element_type;
  
  // 调用右值引用版本的 toCustomClass 转换 IValue 到 ElemType 类型的自定义类对象
  return std::move(ivalue).toCustomClass<ElemType>();
}

// 通用函数：从 IValue 转换为 tagged_capsule<T> 类型对象
template <typename T>
tagged_capsule<T> generic_to(IValue ivalue, _fake_type<tagged_capsule<T>>) {
  // 直接封装 ivalue 为 tagged_capsule<T> 返回
  return tagged_capsule<T>{std::move(ivalue)};
}

// 通用函数：从 IValue 转换为 c10::List<Elem> 类型对象
template <typename Elem>
c10::List<Elem> generic_to(IValue ivalue, _fake_type<c10::List<Elem>>) {
  // 调用 toTypedList 将 IValue 转换为 c10::List<Elem>，再返回
  return impl::toTypedList<Elem>(std::move(ivalue).toList());
}

// 静态函数：从 c10::detail::ListImpl* 创建类似于 T 的容器对象
template <typename T>
static T createVectorLikeFromList(const c10::detail::ListImpl* impl) {
  // 创建 T 类型的结果对象
  T result;
  
  // 预留足够容量以容纳 impl 列表的所有元素
  result.reserve(impl->list.size());
  
  // 遍历 impl 列表，将每个元素转换为 T::value_type 类型后添加到结果对象中
  for (const auto & i : impl->list) {
    result.push_back(i.to<typename T::value_type>());
  }
  
  // 返回填充完毕的结果对象
  return result;
}

// 通用函数：从 c10::detail::ListImpl* 转换为 std::vector<T> 类型对象
template <typename T>
static std::vector<T> createVectorFromList(const c10::detail::ListImpl* impl) {
  // 调用 createVectorLikeFromList 将 c10::detail::ListImpl* 转换为 std::vector<T>，再返回
  return createVectorLikeFromList<std::vector<T>>(impl);
}

// 通用函数：从 c10::List<T> 转换为 std::vector<T> 类型对象
template <typename T>
std::vector<T> createVectorFromList(const c10::List<T>& impl) {
  // 创建 std::vector<T> 类型的结果对象
  std::vector<T> result;
  
  // 预留足够容量以容纳 impl 列表的所有元素
  result.reserve(impl.size());
  
  // 遍历 impl 列表，将每个元素添加到结果对象中
  for (size_t i = 0, N = impl.size(); i < N; ++i) {
    result.push_back(impl[i]);
  }
  
  // 返回填充完毕的结果对象
  return result;
}

// 通用函数：从 IValue 转换为 OptionalArray<T> 类型对象
template <typename T>
OptionalArray<T> generic_to(IValue ivalue, _fake_type<OptionalArray<T>>) {
  // 如果 ivalue 是 None，则返回空的 OptionalArray<T>
  if (ivalue.isNone()) {
    return {};
  }
  
  // 调用 createVectorFromList 将 ivalue 转换为 c10::List<T>，再转换为 OptionalArray<T> 返回
  return createVectorFromList<T>(
    std::move(ivalue).to<c10::List<T>>()
  );
}

namespace detail {
// 通用函数：从 IValue 转换为 std::array<Elem, sizeof...(I)> 类型对象
template <typename Elem, size_t... I>
std::array<Elem, sizeof...(I)> generic_to_array(
    IValue ivalue,
    _fake_type<std::array<Elem, sizeof...(I)>>,
    // 对给定的 ivalue (一个泛型对象) 进行类型转换，转换成一个 List<Elem> 类型的对象，并通过 std::move 转移语义将其移动到变量 list 中
    auto list = std::move(ivalue).to<List<Elem>>();
    
    // 使用 TORCH_CHECK 断言检查 list 的大小是否等于模板参数包 sizeof...(I)，即模板参数中指定的固定大小
    TORCH_CHECK(
        list.size() == sizeof...(I),
        "Tried to convert a List with ",
        list.size(),
        " elements to a fixed-size array of size ",
        sizeof...(I));
    
    // 通过折叠表达式 (pack expansion) 将 list 中的元素按照索引 I 展开成一个固定大小数组的初始化列表，并返回
    return {list[I]...};
} // namespace detail

// 将泛型值转换为 std::array<Elem, N> 类型的特化函数
template <typename Elem, size_t N>
std::array<Elem, N> generic_to(
    IValue ivalue,
    _fake_type<std::array<Elem, N>> ft) {
  return detail::generic_to_array(ivalue, ft, std::make_index_sequence<N>());
}

// 将泛型值转换为 c10::Dict<Key, Value> 类型的特化函数
template <typename Key, typename Value>
c10::Dict<Key, Value> generic_to(
    IValue ivalue,
    _fake_type<c10::Dict<Key, Value>>) {
  return impl::toTypedDict<Key, Value>(std::move(ivalue).toGenericDict());
}

// 弃用消息：基于 std::unordered_map 的 IValues 较慢且已弃用，建议使用 c10::Dict<K, V>
template <typename K, typename V>
C10_DEPRECATED_MESSAGE(
    "IValues based on std::unordered_map are slow and deprecated. Please use c10::Dict<K, V> instead.")
std::unordered_map<K, V> generic_to(
    IValue ivalue,
    _fake_type<std::unordered_map<K, V>>) {
  std::unordered_map<K, V> specialized_dict;

  // 遍历泛型字典，将其转换为特定类型的 unordered_map
  for (const auto& item : std::move(ivalue).toGenericDict()) {
    specialized_dict[item.key().template to<K>()] = item.value().template to<V>();
  }

  return specialized_dict;
}

// 将泛型值转换为 std::optional<T> 类型的特化函数
template <typename T>
std::optional<T> generic_to(IValue ivalue, _fake_type<std::optional<T>>) {
  if (ivalue.isNone()) {
    return c10::nullopt;
  }
  return std::move(ivalue).to<T>();
}

namespace detail {

// 用于实现泛型值到 tuple 的转换
template <typename Tuple, std::size_t... INDEX>
Tuple generic_to_tuple_impl(
    const ivalue::TupleElements& t,
    std::index_sequence<INDEX...>) {
  return std::make_tuple(
      t[INDEX].to<typename std::tuple_element<INDEX, Tuple>::type>()...);
}

} // namespace detail

// 将泛型值转换为 std::tuple<Args...> 类型的特化函数
template <
    typename... Args,
    typename Indices = std::make_index_sequence<sizeof...(Args)>,
    std::enable_if_t<
        !std::disjunction_v<
            std::is_lvalue_reference<Args>...,
            std::negation<std::is_constructible<IValue, Args>>...>,
        std::nullptr_t> = nullptr>
std::tuple<Args...> generic_to(const IValue& ivalue, _fake_type<std::tuple<Args...>>) {
  const auto& vals = ivalue.toTupleRef().elements();
  TORCH_CHECK(vals.size() == sizeof...(Args));
  return detail::generic_to_tuple_impl<std::tuple<Args...>>(vals, Indices{});
}

// IValue 类型的移动右值引用转换函数，返回特化类型 T
template <typename T>
inline T IValue::to() && {
  return generic_to(std::move(*this), _fake_type<T>{});
}

// 特化：返回 std::optional<c10::string_view> 类型的右值引用转换函数
template <>
inline std::optional<c10::string_view> IValue::to() && {
  // 默认实现中，IValue 会被 std::move 销毁。但如果非装箱类型是 optional<string_view>，则不能销毁 IValue。
  return generic_to(*this, _fake_type<std::optional<c10::string_view>>{});
}

// 返回常量引用的泛型值转换函数，返回类型为 T
template <typename T>
inline typename c10::detail::ivalue_to_const_ref_overload_return<T>::type IValue::to() const& {
  return generic_to(*this, _fake_type<T>{});
}

// 返回右值引用的 IntList 转换函数，检查标签类型并返回 c10::List<int64_t>
inline c10::List<int64_t> IValue::toIntList() && {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  return c10::List<int64_t>(moveToIntrusivePtr<c10::detail::ListImpl>());
}

// 返回常量引用的 IntList 转换函数，检查标签类型并返回 c10::List<int64_t>
inline c10::List<int64_t> IValue::toIntList() const& {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  return c10::List<int64_t>(toIntrusivePtr<c10::detail::ListImpl>());
}
// 返回一个包含 int64_t 元素的 std::vector，从当前 IValue 转换而来
inline std::vector<int64_t> IValue::toIntVector() const {
  // 确保当前 IValue 类型为 IntList，否则抛出错误
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  // 断言当前指针不为空，防止空指针异常，仅在调试模式下有效
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toIntVector on null intrusive_ptr IValue");
  // 调用模板函数 createVectorFromList，将 ListImpl 转换为 int64_t 的 std::vector 并返回
  return createVectorFromList<int64_t>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}

// 返回一个包含 c10::SymInt 元素的 std::vector，从当前 IValue 转换而来
inline std::vector<c10::SymInt> IValue::toSymIntVector() const {
  // 确保当前 IValue 类型为 SymIntList 或 IntList，否则抛出错误
  AT_ASSERT(isSymIntList() || isIntList(), "Expected SymIntList or IntList but got ", tagKind());
  // 断言当前指针不为空，防止空指针异常，仅在调试模式下有效
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toSymIntVector on null intrusive_ptr IValue");
  // 调用模板函数 createVectorFromList，将 ListImpl 转换为 c10::SymInt 的 std::vector 并返回
  return createVectorFromList<c10::SymInt>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}

// 返回一个包含 at::DimVector 元素的 at::DimVector，从当前 IValue 转换而来
inline at::DimVector IValue::toDimVector() const {
  // 确保当前 IValue 类型为 IntList，否则抛出错误
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  // 断言当前指针不为空，防止空指针异常，仅在调试模式下有效
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toDimVector on null intrusive_ptr IValue");
  // 调用模板函数 createVectorLikeFromList，将 ListImpl 转换为 at::DimVector 并返回
  return createVectorLikeFromList<at::DimVector>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}

// 将当前 IValue 转换为 c10::List<double>，并移动语义返回
inline c10::List<double> IValue::toDoubleList() && {
  // 确保当前 IValue 类型为 DoubleList，否则抛出错误
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  // 使用 moveToIntrusivePtr 将当前 IValue 转换为 c10::List<double> 并移动语义返回
  return c10::List<double>(moveToIntrusivePtr<c10::detail::ListImpl>());
}

// 将当前 IValue 转换为 c10::List<double>，并通过常引用返回
inline c10::List<double> IValue::toDoubleList() const& {
  // 确保当前 IValue 类型为 DoubleList，否则抛出错误
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  // 使用 toIntrusivePtr 将当前 IValue 转换为 c10::List<double> 并通过常引用返回
  return c10::List<double>(toIntrusivePtr<c10::detail::ListImpl>());
}

// 返回一个包含 double 元素的 std::vector，从当前 IValue 转换而来
inline std::vector<double> IValue::toDoubleVector() const {
  // 确保当前 IValue 类型为 DoubleList，否则抛出错误
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  // 断言当前指针不为空，防止空指针异常，仅在调试模式下有效
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toDoubleVector on null intrusive_ptr IValue");
  // 调用模板函数 createVectorFromList，将 ListImpl 转换为 std::vector<double> 并返回
  return createVectorFromList<double>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}

// 将当前 IValue 转换为 c10::List<c10::complex<double>>，并移动语义返回
inline c10::List<c10::complex<double>> IValue::toComplexDoubleList() && {
  // 确保当前 IValue 类型为 ComplexDoubleList，否则抛出错误
  AT_ASSERT(isComplexDoubleList(), "Expected ComplexDoubleList but got ", tagKind());
  // 使用 moveToIntrusivePtr 将当前 IValue 转换为 c10::List<c10::complex<double>> 并移动语义返回
  return c10::List<c10::complex<double>>(moveToIntrusivePtr<c10::detail::ListImpl>());
}

// 将当前 IValue 转换为 c10::List<c10::complex<double>>，并通过常引用返回
inline c10::List<c10::complex<double>> IValue::toComplexDoubleList() const& {
  // 确保当前 IValue 类型为 ComplexDoubleList，否则抛出错误
  AT_ASSERT(isComplexDoubleList(), "Expected ComplexDoubleList but got ", tagKind());
  // 使用 toIntrusivePtr 将当前 IValue 转换为 c10::List<c10::complex<double>> 并通过常引用返回
  return c10::List<c10::complex<double>>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline std::vector<c10::complex<double>> IValue::toComplexDoubleVector() const {
  // 断言当前对象是否为 ComplexDoubleList 类型，否则抛出异常并显示当前类型
  AT_ASSERT(isComplexDoubleList(), "Expected ComplexDoubleList but got ", tagKind());
  // 内部断言：确保 intrusive_ptr 不为空，否则输出调试信息并终止程序
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toComplexDoubleVector on null intrusive_ptr IValue");
  // 从 ComplexDoubleList 转换为 complex<double> 类型的 vector 并返回
  return createVectorFromList<c10::complex<double>>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}

inline c10::List<bool> IValue::toBoolList() && {
  // 断言当前对象是否为 BoolList 类型，否则抛出异常并显示当前类型
  AT_ASSERT(isBoolList(), "Expected BoolList but got ", tagKind());
  // 将当前对象作为右值移动到新的 c10::List<bool> 对象并返回
  return c10::List<bool>(moveToIntrusivePtr<c10::detail::ListImpl>());
}

inline c10::List<bool> IValue::toBoolList() const& {
  // 断言当前对象是否为 BoolList 类型，否则抛出异常并显示当前类型
  AT_ASSERT(isBoolList(), "Expected BoolList but got ", tagKind());
  // 将当前对象作为常量左值转换为 c10::List<bool> 对象并返回
  return c10::List<bool>(toIntrusivePtr<c10::detail::ListImpl>());
}

inline c10::List<at::Tensor> IValue::toTensorList() && {
  // 断言当前对象是否为 TensorList 类型，否则抛出异常并显示当前类型
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  // 将当前对象作为右值移动到新的 c10::List<at::Tensor> 对象并返回
  return c10::List<at::Tensor>(moveToIntrusivePtr<c10::detail::ListImpl>());
}

inline c10::List<at::Tensor> IValue::toTensorList() const& {
  // 断言当前对象是否为 TensorList 类型，否则抛出异常并显示当前类型
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  // 将当前对象作为常量左值转换为 c10::List<at::Tensor> 对象并返回
  return c10::List<at::Tensor>(toIntrusivePtr<c10::detail::ListImpl>());
}

inline std::vector<at::Tensor> IValue::toTensorVector() const {
  // 断言当前对象是否为 TensorList 类型，否则抛出异常并显示当前类型
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  // 内部断言：确保 intrusive_ptr 不为空，否则输出调试信息并终止程序
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toTensorVector on null intrusive_ptr IValue");
  // 从 TensorList 转换为 vector<at::Tensor> 并返回
  return createVectorFromList<at::Tensor>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}

inline c10::List<std::optional<at::Tensor>> IValue::toOptionalTensorList() && {
  // 断言当前对象是否为 OptionalTensorList 类型，否则抛出异常并显示当前类型
  AT_ASSERT(isOptionalTensorList(), "Expected OptionalTensorList but got ", tagKind());
  // 将当前对象作为右值移动到新的 c10::List<std::optional<at::Tensor>> 对象并返回
  return c10::List<std::optional<at::Tensor>>(moveToIntrusivePtr<c10::detail::ListImpl>());
}

inline c10::List<std::optional<at::Tensor>> IValue::toOptionalTensorList() const& {
  // 断言当前对象是否为 OptionalTensorList 类型，否则抛出异常并显示当前类型
  AT_ASSERT(isOptionalTensorList(), "Expected OptionalTensorList but got ", tagKind());
  // 将当前对象作为常量左值转换为 c10::List<std::optional<at::Tensor>> 对象并返回
  return c10::List<std::optional<at::Tensor>>(toIntrusivePtr<c10::detail::ListImpl>());
}

inline std::vector<std::optional<at::Tensor>> IValue::toOptionalTensorVector() const {
  // 断言当前对象是否为 OptionalTensorList 类型，否则抛出异常并显示当前类型
  AT_ASSERT(isOptionalTensorList(), "Expected OptionalTensorList but got ", tagKind());
  // 内部断言：确保 intrusive_ptr 不为空，否则输出调试信息并终止程序
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toOptionalTensorVector on null intrusive_ptr IValue");
  // 从 OptionalTensorList 转换为 vector<std::optional<at::Tensor>> 并返回
  return createVectorFromList<std::optional<at::Tensor>>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}

inline c10::List<IValue> IValue::toList() && {
  // 断言当前对象是否为 GenericList 类型，否则抛出异常并显示当前类型
  AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
  // 将当前对象作为右值移动到新的 c10::List<IValue> 对象并返回
  return c10::List<IValue>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
// 返回当前 IValue 实例的列表视图，要求实例是列表类型，否则抛出断言错误
inline c10::List<IValue> IValue::toList() const& {
  AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
  // 使用指向 ListImpl 的内部指针创建 c10::List<IValue> 对象并返回
  return c10::List<IValue>(toIntrusivePtr<c10::detail::ListImpl>());
}

// 返回当前 IValue 实例的不可变数组引用，要求实例是列表类型，否则抛出断言错误
inline c10::ArrayRef<IValue> IValue::toListRef() const {
  AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
  // 在调试模式下，确保 payload.u.as_intrusive_ptr 不为 UndefinedTensorImpl 的单例，
  // 否则抛出内部断言错误
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toListRef on null intrusive_ptr IValue");
  // 返回当前 IValue 实例持有的 ListImpl 内部列表的不可变引用
  return static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr)
      ->list;
}

// 移动语义版本：返回当前 IValue 实例的通用字典，要求实例是字典类型，否则抛出断言错误
inline c10::Dict<IValue, IValue> IValue::toGenericDict() && {
  AT_ASSERT(isGenericDict(), "Expected GenericDict but got ", tagKind());
  // 使用移动语义将当前 IValue 实例的 DictImpl 内部指针转移，并返回 Dict<IValue, IValue> 对象
  return c10::Dict<IValue, IValue>(moveToIntrusivePtr<c10::detail::DictImpl>());
}

// 常量引用版本：返回当前 IValue 实例的通用字典，要求实例是字典类型，否则抛出断言错误
inline c10::Dict<IValue, IValue> IValue::toGenericDict() const& {
  AT_ASSERT(isGenericDict(), "Expected GenericDict but got ", tagKind());
  // 使用常量引用将当前 IValue 实例的 DictImpl 内部指针转移，并返回 Dict<IValue, IValue> 对象
  return c10::Dict<IValue, IValue>(toIntrusivePtr<c10::detail::DictImpl>());
}

// 移动语义版本：返回当前 IValue 实例的元组，要求实例是元组类型，否则抛出断言错误
inline c10::intrusive_ptr<ivalue::Tuple> IValue::toTuple() && {
  AT_ASSERT(isTuple(), "Expected Tuple but got ", tagKind());
  // 使用移动语义返回当前 IValue 实例的 Tuple 内部指针
  return moveToIntrusivePtr<ivalue::Tuple>();
}

// 常量引用版本：返回当前 IValue 实例的元组，要求实例是元组类型，否则抛出断言错误
inline c10::intrusive_ptr<ivalue::Tuple> IValue::toTuple() const& {
  AT_ASSERT(isTuple(), "Expected Tuple but got ", tagKind());
  // 使用常量引用返回当前 IValue 实例的 Tuple 内部指针
  return toIntrusivePtr<ivalue::Tuple>();
}

// 返回当前 IValue 实例的元组引用，要求实例是元组类型，否则抛出断言错误
inline ivalue::Tuple& IValue::toTupleRef() const {
  AT_ASSERT(isTuple(), "Expected Tuple but got ", tagKind());
  // 在调试模式下，确保 payload.u.as_intrusive_ptr 不为 UndefinedTensorImpl 的单例，
  // 否则抛出内部断言错误
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toTupleRef on null intrusive_ptr IValue");
  // 返回当前 IValue 实例持有的 Tuple 指针的引用
  return *static_cast<c10::ivalue::Tuple*>(payload.u.as_intrusive_ptr);
}

// 构造函数：用给定的 Tuple 指针创建一个新的 IValue 实例，标记为 Tuple 类型
inline IValue::IValue(c10::intrusive_ptr<ivalue::Tuple> v)
    : tag(Tag::Tuple) {
  // 将给定的 Tuple 指针转换为内部的 undefined_tensor 单例并存储在 payload 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

// 模板构造函数：用给定的 std::tuple 创建一个新的 IValue 实例
template <
    typename... Args,
    std::enable_if_t<
        !std::disjunction_v<
            std::is_lvalue_reference<Args>...,
            std::negation<std::is_constructible<IValue, Args>>...>,
        std::nullptr_t>>
inline IValue::IValue(const std::tuple<Args...>& t)
    : IValue(c10::guts::apply(c10::ivalue::Tuple::create<const Args&...>, t)) {
}

// 移动语义模板构造函数：用给定的 std::tuple 创建一个新的 IValue 实例
template <
    typename... Args,
    std::enable_if_t<
        !std::disjunction_v<
            std::is_lvalue_reference<Args>...,
            std::negation<std::is_constructible<IValue, Args>>...>,
        std::nullptr_t>>
inline IValue::IValue(std::tuple<Args...>&& t)
    : IValue(c10::guts::apply(c10::ivalue::Tuple::create<Args&&...>, std::move(t))) {
}

// 构造函数：用给定的 ConstantString 指针创建一个新的 IValue 实例，标记为 String 类型
inline IValue::IValue(c10::intrusive_ptr<ivalue::ConstantString> v)
    : tag(Tag::String) {
  // 将给定的 ConstantString 指针转换为内部的 undefined_tensor 单例并存储在 payload 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

// 构造函数：用给定的 std::string 创建一个新的 IValue 实例，标记为 String 类型
inline IValue::IValue(std::string v)
    : IValue(ivalue::ConstantString::create(std::move(v))) {}

// 未完成的构造函数实现，用于接受 c10::impl::GenericList 类型的参数
inline IValue::IValue(c10::impl::GenericList v)
    // 初始化 tag 成员为 Tag::GenericList
    : tag(Tag::GenericList) {
  // 调用 null_to_undefined_tensor 函数将 v.impl_ 的所有权释放，并将结果包装成一个 intrusive_ptr 存放在 payload.u 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.impl_.release());
// 结束当前类定义的语法块
}

// 模板构造函数定义，接受右值引用的 c10::List<T> 参数
template <class T, IValue::enable_if_list_is_ivalue_constructible<T>>
inline IValue::IValue(c10::List<T>&& v) : IValue(impl::toList<T>(std::move(v))) {}

// 模板构造函数定义，接受 const 引用的 c10::List<T> 参数
template <class T, IValue::enable_if_list_is_ivalue_constructible<T>>
inline IValue::IValue(const c10::List<T>& v) : IValue(impl::toList<T>(v)) {}

// 模板构造函数定义，接受 at::ArrayRef<T> 参数
template <class T, IValue::enable_if_list_is_ivalue_constructible<T>>
inline IValue::IValue(at::ArrayRef<T> v) : IValue(c10::List<T>()) {
  // 创建 c10::List<T> 对象，并将 v 中的元素逐个添加到 list 中
  auto list = to<c10::List<T>>();
  list.reserve(v.size());
  for (const auto& e : v) {
    list.push_back(e);
  }
}

// 模板构造函数定义，接受 at::ArrayRef<T> 参数，并且 T 满足 IValue::enable_if_symint
template <class T, IValue::enable_if_symint<T>>
inline IValue::IValue(at::ArrayRef<T> v) : IValue() {
  auto vi = c10::asIntArrayRefSlowOpt(v);
  if (vi.has_value()) {
    // 如果数组中的元素全部为整数，将其转换为 IntList
    *this = IValue(*vi);
  } else {
    // 如果数组中的元素包含 SymInt，将其转换为 SymIntList
    *this = IValue(impl::toList<c10::SymInt>(c10::List<c10::SymInt>()));
    auto list = to<c10::List<c10::SymInt>>();
    list.reserve(v.size());
    for (const auto& e : v) {
      list.push_back(e);
    }
  }
}

// 模板构造函数定义，接受 at::OptionalArrayRef<T> 参数，并且 T 满足 IValue::enable_if_symint
template <class T, IValue::enable_if_symint<T>>
inline IValue::IValue(at::OptionalArrayRef<T> mb_v) : IValue() {
  // 如果 mb_v 为空，直接返回
  if (!mb_v.has_value()) return;
  // 否则，调用另一个构造函数处理 mb_v
  *this = IValue(*mb_v);
}

// 模板构造函数定义，接受 const std::vector<T>& 参数，并且 T 满足 IValue::enable_if_symint
template <class T, IValue::enable_if_symint<T>>
inline IValue::IValue(const std::vector<T>& v) : IValue() {
  // 调用另一个构造函数处理 std::vector 转换为 at::ArrayRef<T>
  *this = IValue(at::ArrayRef<T>(v));
}

// 模板构造函数定义，接受 std::vector<T>&& 参数，并且 T 满足 IValue::enable_if_symint
template <class T, IValue::enable_if_symint<T>>
inline IValue::IValue(std::vector<T>&& v) : IValue() {
  auto vi = c10::asIntArrayRefSlowOpt(v);
  if (vi.has_value()) {
    // 如果 vector 中的元素全部为整数，将其转换为 IntList
    *this = IValue(*vi);
  } else {
    // 如果 vector 中的元素包含 SymInt，将其转换为 SymIntList
    *this = IValue(impl::toList<c10::SymInt>(c10::List<c10::SymInt>()));
    auto list = to<c10::List<c10::SymInt>>();
    list.reserve(v.size());
    for (auto&& e : std::move(v)) {
      list.push_back(std::move(e));
    }
  }
}

// 模板构造函数定义，接受 c10::OptionalArrayRef<T> 参数，并且 T 满足 IValue::enable_if_list_is_ivalue_constructible
template <class T, IValue::enable_if_list_is_ivalue_constructible<T>>
inline IValue::IValue(c10::OptionalArrayRef<T> v) : IValue() {
  // 如果 v 有值，则调用另一个构造函数处理
  if (v.has_value()) {
    *this = IValue(std::move(*v));
  }
}

// 模板构造函数定义，接受 size_t N 参数
template <class T, size_t N>
inline IValue::IValue(std::array<T, N> v) : IValue(c10::List<T>()) {
  // 使用默认构造函数创建一个空的 c10::List<T> 对象
  auto list = to<c10::List<T>>();
  // 预留足够的空间以容纳数组 v 的所有元素
  list.reserve(v.size());
  // 将数组 v 中的每个元素移动到列表 list 中
  for (auto& e : v) {
    list.push_back(std::move(e));
  }
}

template <class T, IValue::enable_if_ilist_is_ivalue_constructible<T>>
inline IValue::IValue(c10::IListRef<T> v) : IValue() {
  // 检查是否能够使用 boxed 类型来构造 IValue
  constexpr bool boxed_type_constructs_ivalue =
      std::is_constructible<IValue, typename c10::IListRef<T>::boxed_type>::value;
  // 首先尝试使用 boxed 值
  // 如果失败（可能是因为不在 boxed 状态，或者其 boxed 类型无法构造 IValue），则回退到复制列表
  if (boxed_type_constructs_ivalue && v.isBoxed()) {
    *this = IValue(impl::toList(v.toBoxed()));
  } else {
    // 创建一个 c10::List<T> 对象，并预留足够的空间以容纳 v 的所有元素
    c10::List<T> list;
    list.reserve(v.size());
    // 将 v 中的每个元素复制到列表 list 中
    for (const auto& t : v) {
      list.push_back(t);
    }
    // 将构造出的列表 list 转换为通用字典，并构造出对应的 IValue
    *this = IValue(impl::toList(std::move(list)));
  }
}

inline IValue::IValue(c10::impl::GenericDict v)
    : tag(Tag::GenericDict) {
  // 将传入的 GenericDict 对象转换为 undefined tensor，并存储在 payload 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.impl_.release());
}

template <class Key, class Value>
inline IValue::IValue(c10::Dict<Key, Value> v)
    : IValue(impl::toGenericDict(std::move(v))) {}

template <class Key, class Value>
inline IValue::IValue(std::unordered_map<Key, Value> v)
    : IValue(Dict<Key, Value>()) {
  // 将传入的无序映射转换为 c10::Dict<Key, Value> 对象
  auto dict = to<c10::Dict<Key, Value>>();
  // 预留足够的空间以容纳无序映射 v 的所有元素
  dict.reserve(v.size());
  // 将无序映射 v 中的每个键值对插入到 dict 中
  for (auto& e : v) {
    dict.insert(std::move(e.first), std::move(e.second));
  }
}

template <class T, IValue::enable_if_ivalue_constructible<T>>
inline IValue::IValue(std::optional<T> v) : IValue() {
  // 如果 optional 对象有值，则移动该值构造对应的 IValue
  if (v.has_value()) {
    *this = IValue(std::move(*v));
  }
}

inline IValue::IValue(c10::nullopt_t) : IValue() {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::Object> v)
    : tag(Tag::Object) {
  // 将传入的 intrusive_ptr<ivalue::Object> 对象转换为 undefined tensor，并存储在 payload 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::PyObjectHolder> v)
    : tag(Tag::PyObject) {
  // 将传入的 intrusive_ptr<ivalue::PyObjectHolder> 对象转换为 undefined tensor，并存储在 payload 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::EnumHolder> v)
    : tag(Tag::Enum) {
  // 将传入的 intrusive_ptr<ivalue::EnumHolder> 对象转换为 undefined tensor，并存储在 payload 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue IValue::make_capsule(
    intrusive_ptr<torch::CustomClassHolder> blob) {
  // 创建一个 Tag 为 Capsule 的新 IValue 对象，并将传入的 CustomClassHolder 对象转换为 undefined tensor 存储在 payload 中
  IValue iv;
  iv.tag = Tag::Capsule;
  iv.payload.u.as_intrusive_ptr = null_to_undefined_tensor(blob.release());
  return iv;
}

template <
    typename T,
    std::enable_if_t<std::is_base_of_v<torch::CustomClassHolder, T>, int>>
IValue::IValue(c10::intrusive_ptr<T> custom_class) : tag(Tag::Object) {
  // 获取 T 类型的自定义类类型，并尝试构造相应的 IValue 对象
  auto classType = []() {
    try {
      return c10::getCustomClassType<c10::intrusive_ptr<T>>();
    } catch (const c10::Error&) {
      throw c10::Error(
          "Trying to instantiate a class that isn't a registered custom class: " +
          std::string(c10::util::get_fully_qualified_type_name<T>()));
    }
  }();
  // 使用匿名函数创建一个临时对象，并执行该函数，返回结果作为对象初始化
  auto ivalue_obj = c10::ivalue::Object::create(std::move(classType), /* numSlots */1);
  // 将自定义类对象作为指针封装到ivalue_obj对象的第一个插槽中
  ivalue_obj->setSlot(0, IValue::make_capsule(std::move(custom_class)));
  // 释放ivalue_obj对象的所有权，并将其转换为undefined类型的张量，并赋值给payload的u字段
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(ivalue_obj.release());
}

# IValue 类的构造函数，接受一个 ivalue::Future 类型的指针作为参数
inline IValue::IValue(c10::intrusive_ptr<ivalue::Future> v)
    : tag(Tag::Future) {
  # 将传入的 Future 指针释放并转换为空值或未定义的张量，然后将其存储在联合体 payload 中的 intrusive_ptr 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

# IValue 类的构造函数，接受一个 ivalue::Await 类型的指针作为参数
inline IValue::IValue(c10::intrusive_ptr<ivalue::Await> v)
    : tag(Tag::Await) {
  # 将传入的 Await 指针释放并转换为空值或未定义的张量，然后将其存储在联合体 payload 中的 intrusive_ptr 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

# IValue 类的构造函数，接受一个 c10::RRefInterface 类型的指针作为参数
inline IValue::IValue(c10::intrusive_ptr<c10::RRefInterface> v)
    : tag(Tag::RRef) {
  # 将传入的 RRefInterface 指针释放并转换为空值或未定义的张量，然后将其存储在联合体 payload 中的 intrusive_ptr 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

# IValue 类的构造函数，接受一个 at::Quantizer 类型的指针作为参数
inline IValue::IValue(c10::intrusive_ptr<at::Quantizer> v)
    : tag(Tag::Quantizer) {
  # 将传入的 Quantizer 指针释放并转换为空值或未定义的张量，然后将其存储在联合体 payload 中的 intrusive_ptr 中
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

# IValue 类的模板构造函数，接受一个 c10::complex<T> 类型的参数
template <typename T>
inline IValue::IValue(c10::complex<T> c)
    : tag(Tag::ComplexDouble) {
  # 创建一个 ivalue::ComplexHolder 对象，并将其存储在联合体 payload 中的 intrusive_ptr 中
  auto v = c10::make_intrusive<ivalue::ComplexHolder>(c);
  payload.u.as_intrusive_ptr = v.release();
}

# 返回 IValue 对象中存储的字符串的常量引用
inline const std::string& IValue::toStringRef() const {
  # 断言当前对象的类型为字符串，如果不是则触发断言错误并输出错误信息
  AT_ASSERT(isString(), "Expected String but got ", tagKind());
  # 内部断言，确保存储在 intrusive_ptr 中的指针不为空，否则输出调试信息并触发断言错误
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toStringRef on null intrusive_ptr IValue");
  # 返回静态转换为 ConstantString 指针后的字符串引用
  return static_cast<const c10::ivalue::ConstantString*>(
             payload.u.as_intrusive_ptr)
      ->string();
}

# 返回 IValue 对象中存储的可选字符串的常量引用包装类
inline std::optional<std::reference_wrapper<const std::string>> IValue::
    toOptionalStringRef() const {
  # 如果当前对象为空值，则返回空的 std::optional
  if (isNone()) {
    return c10::nullopt;
  }
  # 断言当前对象的类型为字符串，如果不是则触发断言错误并输出错误信息
  AT_ASSERT(isString(), "Expected optional<string> but got ", tagKind());
  # 内部断言，确保存储在 intrusive_ptr 中的指针不为空，否则输出调试信息并触发断言错误
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toOptionalStringRef on null intrusive_ptr IValue");
  # 返回 ConstantString 指针转换后的字符串引用包装类
  return std::reference_wrapper<const std::string>(
      static_cast<const c10::ivalue::ConstantString*>(payload.u.as_intrusive_ptr)
          ->string());
}

# 返回 IValue 对象中存储的字符串的视图
inline c10::string_view IValue::toStringView() const {
  # 断言当前对象的类型为字符串，如果不是则触发断言错误并输出错误信息
  AT_ASSERT(isString(), "Expected String but got ", tagKind());
  # 内部断言，确保存储在 intrusive_ptr 中的指针不为空，否则输出调试信息并触发断言错误
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toStringView on null intrusive_ptr IValue");
  # 返回 ConstantString 指针转换后的字符串视图
  return static_cast<const c10::ivalue::ConstantString*>(
        payload.u.as_intrusive_ptr)
    ->string_view();
}

# 返回 IValue 对象中存储的 PyObject 指针
inline PyObject* IValue::toPyObject() const {
  # 调用内部方法获取存储的 PyObject 指针并返回
  return toPyObjectHolder()->getPyObject();
}

# 返回 IValue 对象中存储的类型 T 的可选值
template <typename T>
inline optional<T> IValue::toOptional() {
  # 如果当前对象为空值，则返回空的 optional
  if (this->isNone()) {
    return nullopt;
  }
  # 调用 to<T> 方法返回转换后的值
  return this->to<T>();
}

# 返回 IValue 对象中存储的类型 T 的常量可选值
template <typename T>
inline optional<T> IValue::toOptional() const {
  # 如果当前对象为空值，则返回空的 optional
  if (this->isNone()) {
    return nullopt;
  }
  # 调用 to<T> 方法返回转换后的值
  return this->to<T>();
}

# 判断当前 IValue 对象是否为自定义类
inline bool IValue::isCustomClass() const {
  # 调用 torch::isCustomClass 方法判断当前对象是否为自定义类
  return torch::isCustomClass(*this);
}
inline bool IValue::isSameIdentity(const IValue& rhs) const {
  // We choose to not use memcmp for payload check due to potential random
  // padding characters on union type

  // Semantics:
  // 1. Immutable primitive values of the same type (Int, Double, None, Bool,
  // Str) return value equality
  // 2. If it is a tensor type, we need to take undefined tensor into account
  // 3. Undefined_tensor is None and vice versa should be true
  // 4. If it is a reference type (i.e. isIntrusivePtr()), then is True when
  // the pointed-to object is the same.
  // 5. False for all other comparisons.

  // Check if both are None
  if (this->isNone() && rhs.isNone()) {
    return true;
  } else if (this->isBool() && rhs.isBool()) {
    // Compare boolean values
    return this->toBool() == rhs.toBool();
  } else if (this->isTensor() && rhs.isTensor()) {
    // Compare tensors by identity
    return this->payload.as_tensor.is_same(rhs.payload.as_tensor);
  } else if (this->isTensor() && rhs.isNone()) {
    // Special case: undefined tensor and None are considered identical
    return !this->payload.as_tensor.defined();
  } else if (this->isNone() && rhs.isTensor()) {
    // Special case: undefined tensor and None are considered identical
    return !rhs.payload.as_tensor.defined();
  } else if (this->isInt() && rhs.isInt()) {
    // Compare integer values
    return this->toInt() == rhs.toInt();
  } else if (this->isDouble() && rhs.isDouble()) {
    // Compare double values
    return this->toDouble() == rhs.toDouble();
  } else if (this->isString() && rhs.isString()) {
    // Compare string values
    return this->toStringRef() == rhs.toStringRef();
  } else {
    // For objects held in IValue, perform shallow comparison on pointer address
    // to determine identity
    return this->isIntrusivePtr() && rhs.isIntrusivePtr() &&
        this->payload.u.as_intrusive_ptr == rhs.payload.u.as_intrusive_ptr;
  }
}

namespace ivalue {
namespace detail {

// Template specialization for constructing IValue from different types

template <typename T>
IValue from_(T&& x, std::true_type) {
  return IValue(std::forward<T>(x));
}

template <typename T>
IValue from_(c10::intrusive_ptr<T> x, std::false_type) {
  return IValue(std::move(x));
}

template <typename T>
IValue from_(T&& /*x*/, std::false_type) {
  // Static assertion for unsupported types, expected to be unreachable
  static_assert(
      guts::false_t<T>::value,
      "You are calling from with a type that it doesn't support, and isn't a potential custom class (ie: is an intrusive_ptr)");
  return IValue();
}

} // namespace detail

// Generic from function that dispatches to specialized from_ based on type traits

template <typename T>
IValue from(T&& x) {
  return detail::from_(
      std::forward<T>(x), typename std::is_constructible<IValue, T>::type{});
}

} // namespace ivalue

// Template specialization for MaybeOwnedTraits for IValue type

template <>
struct MaybeOwnedTraits<IValue> {
  using owned_type = IValue;
  using borrow_type = IValue;

  // Create borrow_type from owned_type based on certain conditions

  static borrow_type createBorrow(const owned_type& from) {
    if (!from.isPtrType()) {
      // If not a pointer type, return as is
      return from;
    }
    if (from.isTensor()) {
      // If it's a tensor, create a borrow for tensor types
      return IValue(MaybeOwnedTraits<at::Tensor>::createBorrow(from.toTensor()));
    } else {
      // For other pointer types, create a borrow with payload and tag
      return IValue(from.payload, from.tag);
    }
  }

  // Assign one borrow_type to another

  static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
    lhs.clearToNone();  // Clear lhs to None state
    // Assignment logic not fully provided in the snippet
  }

};
    // 如果 rhs 不是指针类型，则直接赋值给 lhs
    if (!rhs.isPtrType()) {
      lhs = rhs;
    } else if (rhs.isTensor()) {  // 如果 rhs 是 Tensor 类型
      // 使用 MaybeOwnedTraits<at::Tensor> 创建一个 borrow，然后包装成 IValue 对象赋给 lhs
      lhs = IValue(MaybeOwnedTraits<at::Tensor>::createBorrow(rhs.toTensor()));
    } else {
      // 使用 rhs 的 payload 和 tag 创建一个新的 IValue 对象赋给 lhs
      lhs = IValue(rhs.payload, rhs.tag);
    }
  }

  // 清空 borrow 对象，使其不再持有任何值
  static void destroyBorrow(borrow_type& toDestroy) {
    toDestroy.clearToNone();
  }

  // 返回 borrow 中所持有的 owned_type 对象的引用
  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  // 返回 borrow 中所持有的 owned_type 对象的指针
  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return &borrow;
  }

  // 调试用函数，始终返回 true，表示 borrow 是有效的
  static bool debugBorrowIsValid(const borrow_type&) {
    return true;
  }
};

// 结构模板特化：当模板参数为 c10::Type 时
template <>
struct IValue::TagType<c10::Type> {
    // 获取给定 IValue 对象的类型信息，返回类型指针
    static TORCH_API c10::TypePtr get(const IValue&);
};

// 结构模板特化：当模板参数为 c10::DynamicType 时
template <>
struct IValue::TagType<c10::DynamicType> {
    // 获取给定 IValue 对象的类型信息，返回类型指针
    static TORCH_API c10::TypePtr get(const IValue&);
};

// 模板函数定义：获取 IValue 对象的类型信息，返回类型指针
template <typename T>
TypePtr IValue::type() const {
    return IValue::TagType<T>::get(*this);
}

} // namespace c10
```