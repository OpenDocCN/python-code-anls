# `.\pytorch\test\edge\Evalue.h`

```py
#pragma once
// 包含 ATen 库的头文件
#include <ATen/ATen.h>

/**
 * WARNING: EValue is a class used by Executorch, for its boxed operators. It
 * contains similar logic as `IValue` in PyTorch, by providing APIs to convert
 * boxed values to unboxed values.
 *
 * It's mirroring a fbcode internal source file
 * [`EValue.h`](https://www.internalfb.com/code/fbsource/xplat/executorch/core/values/Evalue.h).
 *
 * The reason why we are mirroring this class, is to make sure we have CI job
 * coverage on torchgen logic, given that torchgen is used for both Executorch
 * and PyTorch.
 *
 * If any of the logic here needs to be changed, please update fbcode version of
 * `Evalue.h` as well. These two versions will be merged as soon as Executorch
 * is in OSS (hopefully by Q2 2023).
 */

namespace torch {
namespace executor {

// 定义宏 ET_CHECK_MSG 为 TORCH_CHECK_MSG
#define ET_CHECK_MSG TORCH_CHECK_MSG

// 枚举类型 Tag，列出所有可能的标签
#define EXECUTORCH_FORALL_TAGS(_) \
  _(None)                         \
  _(Tensor)                       \
  _(String)                       \
  _(Double)                       \
  _(Int)                          \
  _(Bool)                         \
  _(ListBool)                     \
  _(ListDouble)                   \
  _(ListInt)                      \
  _(ListTensor)                   \
  _(ListScalar)                   \
  _(ListOptionalTensor)

enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
  EXECUTORCH_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};

// 结构体 EValue 声明
struct EValue;

// 模板结构体 evalue_to_const_ref_overload_return，用于确定返回值类型
template <typename T>
struct evalue_to_const_ref_overload_return {
  using type = T;
};

// 特化模板结构体 evalue_to_const_ref_overload_return，针对 at::Tensor 类型
template <>
struct evalue_to_const_ref_overload_return<at::Tensor> {
  using type = const at::Tensor&;
};

// 模板结构体 evalue_to_ref_overload_return，用于确定返回值类型
template <typename T>
struct evalue_to_ref_overload_return {
  using type = T;
};

// 特化模板结构体 evalue_to_ref_overload_return，针对 at::Tensor 类型
template <>
struct evalue_to_ref_overload_return<at::Tensor> {
  using type = at::Tensor&;
};

/*
 * Helper class used to correlate EValues in the executor table, with the
 * unwrapped list of the proper type. Because values in the runtime's values
 * table can change during execution, we cannot statically allocate list of
 * objects at deserialization. Imagine the serialized list says index 0 in the
 * value table is element 2 in the list, but during execution the value in
 * element 2 changes (in the case of tensor this means the TensorImpl* stored in
 * the tensor changes). To solve this instead they must be created dynamically
 * whenever they are used.
 */

// 模板类模板参数为 T 的帮助类，用于在执行表中关联 EValues 和适当类型的解包列表
template <typename T>
class EValObjectList {
 public:
  // 默认构造函数
  EValObjectList() = default;
  /*
   * wrapped_vals 是一个指向运行时值表的指针列表，其目标与列表中元素相关联；
   * unwrapped_vals 是相同大小的容器，用作构造未包装值的内存。
   */
  // 带参构造函数，接受包装值指针数组、未包装值数组和大小作为参数
  EValObjectList(EValue** wrapped_vals, T* unwrapped_vals, int size)
      : wrapped_vals_(wrapped_vals, size), unwrapped_vals_(unwrapped_vals) {}
  /*
   * 构造并返回由 EValue 指针指定的 T 类型列表
   */
  // 返回当前 wrapped_vals 对象中的 ArrayRef<T> 类型数据
  at::ArrayRef<T> get() const;

 private:
  // 列表的真实数据来源
  // 包装值的 ArrayRef
  at::ArrayRef<EValue*> wrapped_vals_;
  // 与 wrapped_vals_ 大小相同的可变未包装值指针数组
  mutable T* unwrapped_vals_;
};

// 类似于 IValue 的聚合类型系统，只是简化了，功能更少，没有原子依赖，并且支持的类型更少，以更好地适应嵌入式系统（例如没有 intrusive ptr）
struct EValue {
  union Payload {
    // 当处于 ATen 模式时，at::Tensor 不是平凡可复制的，这个嵌套联合允许我们处理张量作为特例，同时将其余字段保持在简单状态，而不是在所有地方都需要标签切换。
    union TriviallyCopyablePayload {
      TriviallyCopyablePayload() : as_int(0) {}
      // 通过这三种类型支持标量
      int64_t as_int;
      double as_double;
      bool as_bool;
      // TODO(jakeszwe): 转换回指针以优化此结构的大小
      at::ArrayRef<char> as_string;
      at::ArrayRef<int64_t> as_int_list;
      at::ArrayRef<double> as_double_list;
      at::ArrayRef<bool> as_bool_list;
      // 对象列表，支持 at::Tensor 类型
      EValObjectList<at::Tensor> as_tensor_list;
      // 支持 at::optional<at::Tensor> 类型的对象列表
      EValObjectList<at::optional<at::Tensor>> as_list_optional_tensor;
    } copyable_union;

    // 因为 Tensor 只持有一个 TensorImpl*，所以这里不使用 Tensor*
    at::Tensor as_tensor;

    Payload() {}
    ~Payload() {}
  };

  // 数据存储和类型标签
  Payload payload;
  Tag tag;

  // 基本构造函数和赋值操作符
  // 复制构造函数
  EValue(const EValue& rhs) : EValue(rhs.payload, rhs.tag) {}

  // 移动构造函数
  EValue(EValue&& rhs) noexcept : tag(rhs.tag) {
    moveFrom(std::move(rhs));
  }

  // 移动赋值操作符
  EValue& operator=(EValue&& rhs) & noexcept {
    if (&rhs == this) {
      return *this;
    }

    destroy();
    moveFrom(std::move(rhs));
    return *this;
  }

  // 拷贝赋值操作符
  EValue& operator=(EValue const& rhs) & {
    // 通过复制构造函数和移动赋值操作符定义拷贝赋值操作符
    *this = EValue(rhs);
    return *this;
  }

  // 析构函数
  ~EValue() {
    destroy();
  }

  /****** None 类型 ******/
  // 默认构造函数，初始化为 None 类型
  EValue() : tag(Tag::None) {
    payload.copyable_union.as_int = 0;
  }

  // 判断是否为 None 类型
  bool isNone() const {
    return tag == Tag::None;
  }

  /****** Int 类型 ******/
  /*implicit*/ EValue(int64_t i) : tag(Tag::Int) {
    payload.copyable_union.as_int = i;
  }

  // 判断是否为 Int 类型
  bool isInt() const {
    return tag == Tag::Int;
  }

  // 获取 Int 值
  int64_t toInt() const {
    ET_CHECK_MSG(isInt(), "EValue is not an int.");
    return payload.copyable_union.as_int;
  }

  /****** Double Type ******/
  /*implicit*/ EValue(double d) : tag(Tag::Double) {
    // 初始化一个 EValue 对象为 Double 类型，并保存 double 类型的数据
    payload.copyable_union.as_double = d;
  }

  bool isDouble() const {
    // 检查当前 EValue 对象是否为 Double 类型
    return tag == Tag::Double;
  }

  double toDouble() const {
    // 将当前 EValue 对象转换为 double 类型数据，如果不是 Double 类型则抛出错误
    ET_CHECK_MSG(isDouble(), "EValue is not a Double.");
    return payload.copyable_union.as_double;
  }

  /****** Bool Type ******/
  /*implicit*/ EValue(bool b) : tag(Tag::Bool) {
    // 初始化一个 EValue 对象为 Bool 类型，并保存 bool 类型的数据
    payload.copyable_union.as_bool = b;
  }

  bool isBool() const {
    // 检查当前 EValue 对象是否为 Bool 类型
    return tag == Tag::Bool;
  }

  bool toBool() const {
    // 将当前 EValue 对象转换为 bool 类型数据，如果不是 Bool 类型则抛出错误
    ET_CHECK_MSG(isBool(), "EValue is not a Bool.");
    return payload.copyable_union.as_bool;
  }

  /****** Scalar Type ******/
  /// Construct an EValue using the implicit value of a Scalar.
  /*implicit*/ EValue(at::Scalar s) {
    // 根据 at::Scalar 类型的数据初始化 EValue 对象
    if (s.isIntegral(false)) {
      // 如果是整数类型，保存为 Int 类型数据
      tag = Tag::Int;
      payload.copyable_union.as_int = s.to<int64_t>();
    } else if (s.isFloatingPoint()) {
      // 如果是浮点数类型，保存为 Double 类型数据
      tag = Tag::Double;
      payload.copyable_union.as_double = s.to<double>();
    } else if (s.isBoolean()) {
      // 如果是布尔类型，保存为 Bool 类型数据
      tag = Tag::Bool;
      payload.copyable_union.as_bool = s.to<bool>();
    } else {
      // 如果未初始化，抛出错误
      ET_CHECK_MSG(false, "Scalar passed to EValue is not initialized.");
    }
  }

  bool isScalar() const {
    // 检查当前 EValue 对象是否为 Scalar 类型（Int、Double、Bool）
    return tag == Tag::Int || tag == Tag::Double || tag == Tag::Bool;
  }

  at::Scalar toScalar() const {
    // 将当前 EValue 对象转换为 at::Scalar 类型数据
    // 使用隐式构造函数进行转换

    if (isDouble()) {
      // 如果是 Double 类型，返回其对应的 double 数据
      return toDouble();
    } else if (isInt()) {
      // 如果是 Int 类型，返回其对应的 int64_t 数据
      return toInt();
    } else if (isBool()) {
      // 如果是 Bool 类型，返回其对应的 bool 数据
      return toBool();
    } else {
      // 如果不是 Scalar 类型，抛出错误
      ET_CHECK_MSG(false, "EValue is not a Scalar.");
      return c10::Scalar();
    }
  }

  /****** Tensor Type ******/
  /*implicit*/ EValue(at::Tensor t) : tag(Tag::Tensor) {
    // 初始化一个 EValue 对象为 Tensor 类型，并保存 at::Tensor 数据
    // 在 aten 模式下，at::Tensor 有非平凡的构造函数和析构函数，因此必须通过 placement new 来进行赋值
    new (&payload.as_tensor) at::Tensor(t);
  }

  bool isTensor() const {
    // 检查当前 EValue 对象是否为 Tensor 类型
    return tag == Tag::Tensor;
  }

  at::Tensor toTensor() && {
    // 将当前 EValue 对象转换为 at::Tensor 类型数据（移动语义版本）
    ET_CHECK_MSG(isTensor(), "EValue is not a Tensor.");
    return std::move(payload.as_tensor);
  }

  at::Tensor& toTensor() & {
    // 将当前 EValue 对象转换为 at::Tensor 类型数据（左值引用版本）
    ET_CHECK_MSG(isTensor(), "EValue is not a Tensor.");
    return payload.as_tensor;
  }

  const at::Tensor& toTensor() const& {
    // 将当前 EValue 对象转换为 at::Tensor 类型数据（常量左值引用版本）
    ET_CHECK_MSG(isTensor(), "EValue is not a Tensor.");
    return payload.as_tensor;
  }

  /****** String Type ******/
  /*implicit*/ EValue(const char* s, size_t size) : tag(Tag::String) {
    // 初始化一个 EValue 对象为 String 类型，并保存字符串数据
    payload.copyable_union.as_string = at::ArrayRef<char>(s, size);
  }

  bool isString() const {
    // 检查当前 EValue 对象是否为 String 类型
    return tag == Tag::String;
  }

  at::string_view toString() const {
    // 将当前 EValue 对象转换为 string_view 类型数据（字符串视图）
    ET_CHECK_MSG(isString(), "EValue is not a String.");
    // 省略部分，未完全显示出来
  return at::string_view(
      payload.copyable_union.as_string.data(),
      payload.copyable_union.as_string.size());
}

/****** Int List Type ******/
/*implicit*/ EValue(at::ArrayRef<int64_t> i) : tag(Tag::ListInt) {
  // 构造函数，接受整数数组引用，设置标签为 ListInt，并存储在联合体中
  payload.copyable_union.as_int_list = i;
}

bool isIntList() const {
  // 返回当前对象是否是整数列表类型（ListInt）
  return tag == Tag::ListInt;
}

at::ArrayRef<int64_t> toIntList() const {
  // 将当前对象转换为整数列表类型（ListInt），并返回其数据
  ET_CHECK_MSG(isIntList(), "EValue is not an Int List.");
  return payload.copyable_union.as_int_list;
}

/****** Bool List Type ******/
/*implicit*/ EValue(at::ArrayRef<bool> b) : tag(Tag::ListBool) {
  // 构造函数，接受布尔数组引用，设置标签为 ListBool，并存储在联合体中
  payload.copyable_union.as_bool_list = b;
}

bool isBoolList() const {
  // 返回当前对象是否是布尔列表类型（ListBool）
  return tag == Tag::ListBool;
}

at::ArrayRef<bool> toBoolList() const {
  // 将当前对象转换为布尔列表类型（ListBool），并返回其数据
  ET_CHECK_MSG(isBoolList(), "EValue is not a Bool List.");
  return payload.copyable_union.as_bool_list;
}

/****** Double List Type ******/
/*implicit*/ EValue(at::ArrayRef<double> d) : tag(Tag::ListDouble) {
  // 构造函数，接受双精度浮点数数组引用，设置标签为 ListDouble，并存储在联合体中
  payload.copyable_union.as_double_list = d;
}

bool isDoubleList() const {
  // 返回当前对象是否是双精度浮点数列表类型（ListDouble）
  return tag == Tag::ListDouble;
}

at::ArrayRef<double> toDoubleList() const {
  // 将当前对象转换为双精度浮点数列表类型（ListDouble），并返回其数据
  ET_CHECK_MSG(isDoubleList(), "EValue is not a Double List.");
  return payload.copyable_union.as_double_list;
}

/****** Tensor List Type ******/
/*implicit*/ EValue(EValObjectList<at::Tensor> t) : tag(Tag::ListTensor) {
  // 构造函数，接受张量对象列表，设置标签为 ListTensor，并存储在联合体中
  payload.copyable_union.as_tensor_list = t;
}

bool isTensorList() const {
  // 返回当前对象是否是张量列表类型（ListTensor）
  return tag == Tag::ListTensor;
}

at::ArrayRef<at::Tensor> toTensorList() const {
  // 将当前对象转换为张量列表类型（ListTensor），并返回其数据
  ET_CHECK_MSG(isTensorList(), "EValue is not a Tensor List.");
  return payload.copyable_union.as_tensor_list.get();
}

/****** List Optional Tensor Type ******/
/*implicit*/ EValue(EValObjectList<at::optional<at::Tensor>> t)
    : tag(Tag::ListOptionalTensor) {
  // 构造函数，接受可选张量对象列表，设置标签为 ListOptionalTensor，并存储在联合体中
  payload.copyable_union.as_list_optional_tensor = t;
}

bool isListOptionalTensor() const {
  // 返回当前对象是否是可选张量对象列表类型（ListOptionalTensor）
  return tag == Tag::ListOptionalTensor;
}

at::ArrayRef<at::optional<at::Tensor>> toListOptionalTensor() {
  // 将当前对象转换为可选张量对象列表类型（ListOptionalTensor），并返回其数据
  return payload.copyable_union.as_list_optional_tensor.get();
}

/****** ScalarType Type ******/
at::ScalarType toScalarType() const {
  // 将当前对象转换为标量类型（ScalarType），并返回其数据
  ET_CHECK_MSG(isInt(), "EValue is not a ScalarType.");
  return static_cast<at::ScalarType>(payload.copyable_union.as_int);
}

/****** MemoryFormat Type ******/
at::MemoryFormat toMemoryFormat() const {
  // 将当前对象转换为内存格式类型（MemoryFormat），并返回其数据
  ET_CHECK_MSG(isInt(), "EValue is not a MemoryFormat.");
  return static_cast<at::MemoryFormat>(payload.copyable_union.as_int);
}

template <typename T>
T to() &&;

template <typename T>
typename evalue_to_ref_overload_return<T>::type to() &;

/**
 * Converts the EValue to an optional object that can represent both T and
 * an uninitialized state.
 */
template <typename T>
inline at::optional<T> toOptional() {
  // 将当前对象转换为可选对象类型，如果当前对象为 None，则返回空值
  if (this->isNone()) {
    return at::nullopt;
  }
  // 返回当前对象转换为类型 T 后的结果
  return this->to<T>();
}

private:
// 前提条件：payload 值已调用其析构函数
void clearToNone() noexcept {
  // 将 payload 的 copyable_union 成员置为 0
  payload.copyable_union.as_int = 0;
  // 将 tag 置为 Tag::None
  tag = Tag::None;
}

// 共享的移动逻辑
void moveFrom(EValue&& rhs) noexcept {
  // 如果 rhs 是 Tensor 类型
  if (rhs.isTensor()) {
    // 在 payload.as_tensor 上构造新的 at::Tensor 对象，移动 rhs 的内容进来
    new (&payload.as_tensor) at::Tensor(std::move(rhs.payload.as_tensor));
    // 调用 rhs.payload.as_tensor 的析构函数，释放其资源
    rhs.payload.as_tensor.~Tensor();
  } else {
    // 否则直接复制 rhs 的 payload.copyable_union 到当前对象的 payload
    payload.copyable_union = rhs.payload.copyable_union;
  }
  // 将当前对象的 tag 设置为 rhs 的 tag
  tag = rhs.tag;
  // 清空 rhs 对象，将其置为 None 状态
  rhs.clearToNone();
}

// 销毁存储的 tensor（如果存在）
void destroy() {
  // 如果当前对象存储了 Tensor
  if (isTensor()) {
    // 调用 payload.as_tensor 的析构函数，释放 Tensor 的资源
    payload.as_tensor.~Tensor();
  } else if (isTensorList()) {
    // 如果当前对象存储了 Tensor 的列表
    for (auto& tensor : toTensorList()) {
      // 逐个调用列表中每个 Tensor 的析构函数，释放资源
      tensor.~Tensor();
    }
  } else if (isListOptionalTensor()) {
    // 如果当前对象存储了 optional 的 Tensor 列表
    for (auto& optional_tensor : toListOptionalTensor()) {
      // 逐个调用列表中每个 optional 的析构函数，释放资源
      optional_tensor.~optional();
    }
  }
}

// 使用给定的 Payload 对象 p 和 Tag 值 t 进行构造
EValue(const Payload& p, Tag t) : tag(t) {
  // 如果当前对象是 Tensor 类型
  if (isTensor()) {
    // 在 payload.as_tensor 上构造新的 at::Tensor 对象，使用 p.as_tensor 进行初始化
    new (&payload.as_tensor) at::Tensor(p.as_tensor);
  } else {
    // 否则直接将 p 的 copyable_union 复制给当前对象的 payload
    payload.copyable_union = p.copyable_union;
  }
}
};

// 定义一个宏来简化类型转换函数的定义，模板参数 T 和方法名 method_name 作为参数
#define EVALUE_DEFINE_TO(T, method_name)                           \
  template <>                                                      \
  inline evalue_to_ref_overload_return<T>::type EValue::to<T>()& { \
    // 调用当前对象的 method_name 方法，并将其静态转换为类型 T
    return static_cast<T>(this->method_name());                    \
  }

// 为类型 at::Tensor 定义特化版本的类型转换函数
template <>
inline at::Tensor& EValue::to<at::Tensor>() & {
  // 调用当前对象的 toTensor 方法，并返回其结果
  return this->toTensor();
}

// 以下为一系列类型转换函数的定义，使用宏 EVALUE_DEFINE_TO 简化重复的模板代码
EVALUE_DEFINE_TO(at::Scalar, toScalar)
EVALUE_DEFINE_TO(int64_t, toInt)
EVALUE_DEFINE_TO(bool, toBool)
EVALUE_DEFINE_TO(double, toDouble)
EVALUE_DEFINE_TO(at::string_view, toString)
EVALUE_DEFINE_TO(at::ScalarType, toScalarType)
EVALUE_DEFINE_TO(at::MemoryFormat, toMemoryFormat)
EVALUE_DEFINE_TO(at::optional<at::Tensor>, toOptional<at::Tensor>)
EVALUE_DEFINE_TO(at::ArrayRef<int64_t>, toIntList)
EVALUE_DEFINE_TO(
    at::optional<at::ArrayRef<int64_t>>,
    toOptional<at::ArrayRef<int64_t>>)
EVALUE_DEFINE_TO(
    at::optional<at::ArrayRef<double>>,
    toOptional<at::ArrayRef<double>>)
EVALUE_DEFINE_TO(at::ArrayRef<at::optional<at::Tensor>>, toListOptionalTensor)
EVALUE_DEFINE_TO(at::ArrayRef<double>, toDoubleList)
#undef EVALUE_DEFINE_TO

// 定义模板类 EValObjectList<T> 的成员函数 get，用于获取包装值列表的解包值列表
template <typename T>
at::ArrayRef<T> EValObjectList<T>::get() const {
  // 遍历包装值列表 wrapped_vals_
  for (size_t i = 0; i < wrapped_vals_.size(); i++) {
    // 将 wrapped_vals_ 中的每个元素通过类型转换函数 template to<T>() 解包到 unwrapped_vals_ 中
    unwrapped_vals_[i] = wrapped_vals_[i]->template to<T>();
  }
  // 返回解包后的值列表 unwrapped_vals_ 作为 at::ArrayRef<T> 类型的引用
  return at::ArrayRef<T>{unwrapped_vals_, wrapped_vals_.size()};
}

// 命名空间 executor 的结束
} // namespace executor

// 命名空间 torch 的结束
} // namespace torch
```