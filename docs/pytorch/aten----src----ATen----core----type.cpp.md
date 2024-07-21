# `.\pytorch\aten\src\ATen\core\type.cpp`

```
// 包含 ATen 库中所需的头文件
#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dynamic_type.h>
#include <ATen/core/enum_type.h>
#include <ATen/core/function.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/grad_mode.h>
#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <array>
#include <iostream>
#include <utility>

// 声明 std 命名空间下的自定义哈希函数
namespace std {
// 用于元组 (string, TypePtr, TypePtr) 的哈希函数
template<>
struct hash<std::tuple<std::string, c10::TypePtr, c10::TypePtr>> {
  size_t operator()(std::tuple<std::string, c10::TypePtr, c10::TypePtr> const& t) const {
    // 通过哈希结合函数对元组的每个成员进行哈希计算
    auto hash = std::hash<std::string>()(std::get<0>(t));
    hash = at::hash_combine(hash, std::hash<c10::TypePtr>()(std::get<1>(t)));
    hash = at::hash_combine(hash, std::hash<c10::TypePtr>()(std::get<2>(t)));
    return hash;
  }
};

// 用于元组 (string, TypePtr) 的哈希函数
template<>
struct hash<std::tuple<std::string, c10::TypePtr>> {
  size_t operator()(std::tuple<std::string, c10::TypePtr> const& t) const {
    // 通过哈希结合函数对元组的每个成员进行哈希计算
    auto hash = std::hash<std::string>()(std::get<0>(t));
    hash = at::hash_combine(hash, std::hash<c10::TypePtr>()(std::get<1>(t)));
    return hash;
  }
};
} // namespace std

// 声明 c10 命名空间
namespace c10 {

// 静态断言检查 std::shared_ptr 在当前平台上的内存布局是否符合预期
static_assert(
    sizeof(SingletonOrSharedTypePtr<void>) == sizeof(std::shared_ptr<void>) && sizeof(std::shared_ptr<void>) == 2 * sizeof(void*),
    "std::shared_ptr has an unexpected representation on this platform!");

// 静态断言检查 getTypePtr<std::tuple<int64_t, int64_t>>() 返回值是否为 const 引用
static_assert(
    std::is_same_v<decltype(getTypePtr<std::tuple<int64_t, int64_t>>()), const TupleTypePtr&>,
    "getTypePtr<std::tuple<int64_t, int64_t>> not returning const ref!");

// 获取环境变量 PYTORCH_JIT_TYPE_VERBOSITY，确定类型信息的详细程度
TypeVerbosity type_verbosity() {
  static const char* c_verbosity = std::getenv("PYTORCH_JIT_TYPE_VERBOSITY");
  // 如果环境变量存在，将其转换为整数并返回对应的类型详细程度
  static TypeVerbosity verbosity = c_verbosity ?
    static_cast<TypeVerbosity>(std::stoi(c_verbosity)) : TypeVerbosity::Default;
  return verbosity;
}

// 重载流输出运算符，用于将 Type 对象输出到流中
std::ostream& operator<<(std::ostream & out, const Type & t) {
  // 尝试将 Type 对象转换为 TensorType
  if (auto value = t.cast<TensorType>()) {
    // 如果转换成功，检查是否存在标量类型信息
    if (value->scalarType().has_value()) {
      // 如果存在标量类型，将其转换为字符串并输出
      out << toString(*value->scalarType());
      // 检查是否存在张量的维度信息
      if (!value->sizes().size().has_value()) {
        // 如果不存在维度信息，追加 "Tensor" 到输出流
        out << "Tensor";
      }
    } else {
      // 如果不存在标量类型信息，直接输出 "Tensor"
      out << "Tensor";
    }
    // 如果对象不是 TensorType，直接输出 "Tensor"
    // 如果 value 是一个 Tensor，并且具有确定的维度信息
    if (auto ndim = value->sizes().size()) {
      // 检查是否有有效的步长信息：维度大于零，步长信息完整，并且步长信息的数量与维度相同
      bool has_valid_strides_info = *ndim > 0 &&
          value->strides().isComplete() && value->strides().size() == ndim;

      // 输出左括号作为起始
      out << "(";
      size_t i = 0;
      // 确定是否使用符号类型的名称
      bool symbolic = type_verbosity() == TypeVerbosity::Symbolic;
      // 遍历维度信息
      for (i = 0; i < *ndim; ++i) {
        // 如果不是第一个维度，输出逗号和空格
        if (i > 0) {
          out << ", ";
        }
        // 如果存在第 i 维的大小信息，则输出该大小
        if (auto s = value->sizes()[i]) {
          out << *s;
        } else if (symbolic) {
          // 否则，如果是符号类型，输出符号大小信息
          out << value->symbolic_sizes().at(i);
        } else {
          // 否则输出星号表示未知大小
          out << "*";
        }
      }
      // 如果有有效的步长信息并且类型显示模式至少是 TypeAndStride
      if (has_valid_strides_info &&
          type_verbosity() >= TypeVerbosity::TypeAndStride) {
        // 输出步长信息
        out << ", strides=[";
        for (size_t i = 0; i < *ndim; ++i) {
          if (i > 0) {
            out << ", ";
          }
          out << *value->strides()[i];
        }
        out << "]";
      }
      // 如果类型显示模式至少是 Full
      if (type_verbosity() >= TypeVerbosity::Full) {
        // 如果需要梯度
        if (value->requiresGrad()) {
          // 如果不是第一个附加信息，输出逗号和空格
          if (i++ > 0) {
            out << ", ";
          }
          // 输出 requires_grad 信息
          out << "requires_grad=" << *value->requiresGrad();
        }
        // 如果有设备信息
        if (value->device()) {
          // 如果不是第一个附加信息，输出逗号和空格
          if (i++ > 0) {
            out << ", ";
          }
          // 输出设备信息
          out << "device=" << *value->device();
        }
      }
      // 输出右括号作为结束
      out << ")";
    } else {
      // 如果类型显示模式至少是 Full
      if (type_verbosity() >= TypeVerbosity::Full) {
        size_t i = 0;
        // 如果需要梯度
        if (value->requiresGrad()) {
          // 输出 requires_grad 信息
          out << "(" << "requires_grad=" << *value->requiresGrad();
          i++;
        }
        // 如果有设备信息
        if (value->device()) {
          // 如果不是第一个附加信息，输出逗号和空格
          out << ((i++ > 0) ? ", " : "(") << "device=" << *value->device();
        }
        // 如果有附加信息，输出右括号
        if (i > 0) {
          out << ")";
        }
      }
    }

    // 如果 value 是未定义的，并且标记为未定义
    if (value->undefined() && *value->undefined()) {
      // 输出 "[Undefined]"
      out << "[Undefined]";
    }
  } else if(t.kind() == TypeKind::ListType) {
    // 如果类型是 ListType
    auto prim = t.castRaw<ListType>()->getElementType();
    // 输出元素类型后跟 "[]"
    out << *prim << "[]";
  } else if (t.kind() == TypeKind::OptionalType) {
    // 如果类型是 OptionalType
    auto prim = t.castRaw<OptionalType>()->getElementType();
    // 输出元素类型后跟 "?"
    out << *prim << "?";
  } else if(t.kind() == TypeKind::FutureType) {
    // 如果类型是 FutureType
    auto elem = t.castRaw<FutureType>()->getElementType();
    // 输出 "Future[" 后跟元素类型信息
    out << "Future[" << *elem << "]";
  } else if(t.kind() == TypeKind::RRefType) {
    // 如果类型是 RRefType
    auto elem = t.castRaw<RRefType>()->getElementType();
    // 输出 "RRef[" 后跟元素类型信息
    out << "RRef[" << *elem << "]";
  } else if(auto tup = t.cast<TupleType>()) {
    // 如果类型是 TupleType
    if (tup->schema()) {
      // 如果有定义的模式，输出 "NamedTuple"
      out << "NamedTuple";
    }
    // 输出左括号作为起始
    out << "(";
    // 遍历元素列表
    for(size_t i = 0; i < tup->elements().size(); ++i) {
      // 如果不是第一个元素，输出逗号和空格
      if(i > 0)
        out << ", ";
      // 如果有定义的模式
      if (tup->schema()) {
        // 获取参数名称和类型信息
        auto arg = tup->schema()->arguments()[i];
        out << arg.name() << " : ";
        out << *(tup->elements()[i]);
        // 如果有默认值，输出默认值信息
        if (arg.default_value()) {
          out << " = " << *arg.default_value();
        }
      }
      else {
        // 否则输出元素类型信息
        out << *(tup->elements()[i]);
      }
    }
    // 输出右括号作为结束
    out << ")";
  } else if (t.kind() == TypeKind::FunctionType) {
    // 如果类型是 FunctionType，输出 "Function"
    out << "Function";
  } else {
     // 否则输出类型的字符串表示
     out << t.str();
  }
  // 返回输出流
  return out;
}

# 获取任意类型的单例对象指针
AnyTypePtr AnyType::get() {
    # 静态局部变量，存储 AnyType 类型的单例对象，并初始化
    static AnyTypePtr value(new AnyType());
    # 返回单例对象指针
    return value;
}

# 获取 NumberType 类型的单例对象指针
NumberTypePtr NumberType::get() {
    # 静态局部变量，存储 NumberType 类型的单例对象，并初始化
    static NumberTypePtr value(new NumberType());
    # 返回单例对象指针
    return value;
}

# 获取 IntType 类型的单例对象指针
IntTypePtr IntType::get() {
    # 静态局部变量，存储 IntType 类型的单例对象，并初始化
    static IntTypePtr value(new IntType());
    # 返回单例对象指针
    return value;
}

# 获取 FloatType 类型的单例对象指针
FloatTypePtr FloatType::get() {
    # 静态局部变量，存储 FloatType 类型的单例对象，并初始化
    static FloatTypePtr value(new FloatType());
    # 返回单例对象指针
    return value;
}

# 获取 ComplexType 类型的单例对象指针
ComplexTypePtr ComplexType::get() {
    # 静态局部变量，存储 ComplexType 类型的单例对象，并初始化
    static ComplexTypePtr value(new ComplexType());
    # 返回单例对象指针
    return value;
}

# 获取 BoolType 类型的单例对象指针
BoolTypePtr BoolType::get() {
    # 静态局部变量，存储 BoolType 类型的单例对象，并初始化
    static BoolTypePtr value(new BoolType());
    # 返回单例对象指针
    return value;
}

# 获取 StorageType 类型的单例对象指针
StorageTypePtr StorageType::get() {
    # 静态局部变量，存储 StorageType 类型的单例对象，并初始化
    static StorageTypePtr value(new StorageType());
    # 返回单例对象指针
    return value;
}

# 获取 NoneType 类型的单例对象指针
NoneTypePtr NoneType::get() {
    # 静态局部变量，存储 NoneType 类型的单例对象，并初始化
    static NoneTypePtr value(new NoneType());
    # 返回单例对象指针
    return value;
}

# 获取 GeneratorType 类型的单例对象指针
GeneratorTypePtr GeneratorType::get() {
    # 静态局部变量，存储 GeneratorType 类型的单例对象，并初始化
    static GeneratorTypePtr value(new GeneratorType());
    # 返回单例对象指针
    return value;
}

# 获取 QuantizerType 类型的单例对象指针
QuantizerTypePtr QuantizerType::get() {
    # 静态局部变量，存储 QuantizerType 类型的单例对象，并初始化
    static QuantizerTypePtr value(new QuantizerType());
    # 返回单例对象指针
    return value;
}

# 获取 QSchemeType 类型的单例对象指针
QSchemeTypePtr QSchemeType::get() {
    # 静态局部变量，存储 QSchemeType 类型的单例对象，并初始化
    static QSchemeTypePtr value(new QSchemeType());
    # 返回单例对象指针
    return value;
}

# 获取 StringType 类型的单例对象指针
StringTypePtr StringType::get() {
    # 静态局部变量，存储 StringType 类型的单例对象，并初始化
    static StringTypePtr value(new StringType());
    # 返回单例对象指针
    return value;
}

# 获取 DeviceObjType 类型的单例对象指针
DeviceObjTypePtr DeviceObjType::get() {
    # 静态局部变量，存储 DeviceObjType 类型的单例对象，并初始化
    static DeviceObjTypePtr value(new DeviceObjType());
    # 返回单例对象指针
    return value;
}

# 获取 StreamObjType 类型的单例对象指针
StreamObjTypePtr StreamObjType::get() {
    # 静态局部变量，存储 StreamObjType 类型的单例对象，并初始化
    static StreamObjTypePtr value(new StreamObjType());
    # 返回单例对象指针
    return value;
}

# 获取 ScalarTypeType 类型的单例对象指针
ScalarTypeTypePtr ScalarTypeType::get() {
    # 静态局部变量，存储 ScalarTypeType 类型的单例对象，并初始化
    static ScalarTypeTypePtr value(new ScalarTypeType());
    # 返回单例对象指针
    return value;
}

# 获取 LayoutType 类型的单例对象指针
LayoutTypePtr LayoutType::get() {
    # 静态局部变量，存储 LayoutType 类型的单例对象，并初始化
    static LayoutTypePtr value(new LayoutType());
    # 返回单例对象指针
    return value;
}

# 获取 MemoryFormatType 类型的单例对象指针
MemoryFormatTypePtr MemoryFormatType::get() {
    # 静态局部变量，存储 MemoryFormatType 类型的单例对象，并初始化
    static MemoryFormatTypePtr value(new MemoryFormatType());
    # 返回单例对象指针
    return value;
}

# 获取 PyObjectType 类型的单例对象指针
PyObjectTypePtr PyObjectType::get() {
    # 静态局部变量，存储 PyObjectType 类型的单例对象，并初始化
    static PyObjectTypePtr value(new PyObjectType());
    # 返回单例对象指针
    return value;
}

# 获取 CapsuleType 类型的单例对象指针
CapsuleTypePtr CapsuleType::get() {
    # 静态局部变量，存储 CapsuleType 类型的单例对象，并初始化
    static CapsuleTypePtr value(new CapsuleType());
    # 返回单例对象指针
    return value;
}

# 获取整数类型列表 ListTypePtr 的单例对象指针
ListTypePtr ListType::ofInts() {
    # 静态局部变量，存储整数类型列表 ListTypePtr 的单例对象，并初始化
    static auto value = ListType::create(IntType::get());
    # 返回单例对象指针
    return value;
}

# 获取符号整数类型列表 ListTypePtr 的单例对象指针
ListTypePtr ListType::ofSymInts() {
    # 静态局部变量，存储符号整数类型列表 ListTypePtr 的单例对象，并初始化
    static auto value = ListType::create(SymIntType::get());
    # 返回单例对象指针
    return value;
}

# 获取复数双精度浮点类型列表 ListTypePtr 的单例对象指针
ListTypePtr ListType::ofComplexDoubles() {
    # 静态局部变量，存储复数双精度浮点类型列表 ListTypePtr 的单例对象，并初始化
    static auto value = ListType::create(ComplexType::get());
    # 返回单例对象指针
    return value;
}

# 获取浮点数类型列表 ListTypePtr 的单例对象指针
ListTypePtr ListType::ofFloats() {
    # 静态局部变量，存储浮点数类型列表 ListTypePtr 的单例对象，并初始化
    static auto value = ListType::create(FloatType::get());
    # 返回单例对象指针
    return value;
}

# 获取布尔类型列表 ListTypePtr 的单例对象指针
ListTypePtr ListType::ofBools() {
    # 静态局部变量，存储布尔类型列表 ListTypePtr 的单例对象，并初始化
    static auto value = ListType::create(BoolType::get());
    # 返回单例对象指针
    return value;
}

# 获取字符串类型列表 ListTypePtr 的单例对象指针
ListTypePtr ListType::ofStrings() {
    # 静态局部变量，存储字符串类型列表 ListTypePtr 的单例对象，并初始化
    static auto value = ListType::create(StringType::get());
    # 返回单例对象指针
    return value;
}

# 获取数字类型列表 ListTypePtr 的单例对象指针
ListTypePtr ListType::ofNumbers() {
    # 静态局部变量，存储数字类型列表 ListTypePtr 的单例对象，并初始化
    static auto value = ListType::create(NumberType::get());
    # 返回单例对象指针
    return value;
}
// 获取一个 OptionalType 类型的指针，如果尚未存在，则创建并存储在静态哈希映射 containerTypePtrs 中
TypePtr OptionalType::get(TypePtr inner) {
  // 静态哈希映射，存储每个 inner 对应的 OptionalType 类型指针
  static ska::flat_hash_map<TypePtr, TypePtr> containerTypePtrs;
  // 静态互斥量，用于线程安全地访问 containerTypePtrs
  static std::mutex mutex;
  // 使用互斥量进行加锁，保证线程安全性
  std::lock_guard<std::mutex> lock(mutex);
  // 如果 inner 在 containerTypePtrs 中不存在，则创建新的 OptionalType 对象并存储
  if (containerTypePtrs.find(inner) == containerTypePtrs.end()) {
    TypePtr t = TypeFactory::create<OptionalType>(inner);
    containerTypePtrs.emplace(inner, std::move(t));
  }
  // 返回 inner 对应的 OptionalType 类型指针
  return containerTypePtrs[inner];
}

// 获取一个 ListType 类型的指针，如果尚未存在，则创建并存储在静态哈希映射 containerTypePtrs 中
TypePtr ListType::get(const std::string& identifier, TypePtr inner) {
  // 静态哈希映射，存储每个 (identifier, inner) 对应的 ListType 类型指针
  static ska::flat_hash_map<std::tuple<std::string, TypePtr>, TypePtr> containerTypePtrs;
  // 静态互斥量，用于线程安全地访问 containerTypePtrs
  static std::mutex mutex;
  // 构造哈希键
  auto key = std::make_tuple(identifier, inner);
  // 使用互斥量进行加锁，保证线程安全性
  std::lock_guard<std::mutex> lock(mutex);
  // 如果 (identifier, inner) 在 containerTypePtrs 中不存在，则创建新的 ListType 对象并存储
  if (containerTypePtrs.find(key) == containerTypePtrs.end()) {
    TypePtr t = ListType::create(inner);
    containerTypePtrs.emplace(key, std::move(t));
  }
  // 返回 (identifier, inner) 对应的 ListType 类型指针
  return containerTypePtrs[key];
}

// 获取一个 DictType 类型的指针，如果尚未存在，则创建并存储在静态哈希映射 containerTypePtrs 中
TypePtr DictType::get(const std::string& identifier, TypePtr key, TypePtr value) {
  // 静态哈希映射，存储每个 (identifier, key, value) 对应的 DictType 类型指针
  static ska::flat_hash_map<std::tuple<std::string, TypePtr, TypePtr>, TypePtr> containerTypePtrs;
  // 静态互斥量，用于线程安全地访问 containerTypePtrs
  static std::mutex mutex;
  // 构造哈希键
  auto map_key = std::make_tuple(identifier, key, value);
  // 使用互斥量进行加锁，保证线程安全性
  std::lock_guard<std::mutex> lock(mutex);
  // 如果 (identifier, key, value) 在 containerTypePtrs 中不存在，则创建新的 DictType 对象并存储
  if (containerTypePtrs.find(map_key) == containerTypePtrs.end()) {
    TypePtr t = DictType::create(std::move(key), std::move(value));
    containerTypePtrs.emplace(map_key, std::move(t));
  }
  // 返回 (identifier, key, value) 对应的 DictType 类型指针
  return containerTypePtrs[map_key];
}

// 返回 DictType 的注释字符串表示，包括键和值的类型注释
std::string DictType::annotation_str_impl(const TypePrinter& printer) const {
  // 获取键的类型注释
  auto keyAnnotation = getKeyType()->annotation_str(printer);
  // 获取值的类型注释
  auto valueAnnotation = getValueType()->annotation_str(printer);

  // 构造结果字符串
  std::string result;
  // 预留足够的空间来容纳 "Dict[keyAnnotation, valueAnnotation]" 的字符串
  result.reserve(5 /* "Dict[" */ + keyAnnotation.size() + 2 /* ", " */ + valueAnnotation.size() + 1 /* "]" */);
  result = "Dict[";
  result += keyAnnotation;
  result.push_back(',');
  result.push_back(' ');
  result += valueAnnotation;
  result.push_back(']');
  // 返回构造的结果字符串
  return result;
}

// 返回静态的 AnyListTypePtr 实例
AnyListTypePtr AnyListType::get() {
  static AnyListTypePtr value(new AnyListType());
  return value;
}

// 返回静态的 AnyTupleTypePtr 实例
AnyTupleTypePtr AnyTupleType::get() {
  static AnyTupleTypePtr value(new AnyTupleType());
  return value;
}

// 返回静态的 AnyClassTypePtr 实例
AnyClassTypePtr AnyClassType::get() {
  static AnyClassTypePtr value(new AnyClassType());
  return value;
}

// 返回静态的 AnyEnumTypePtr 实例
AnyEnumTypePtr AnyEnumType::get() {
  static AnyEnumTypePtr value(new AnyEnumType());
  return value;
}

// 返回静态的 SymIntTypePtr 实例
SymIntTypePtr SymIntType::get() {
  static SymIntTypePtr value(new SymIntType());
  return value;
}

// 返回静态的 SymFloatTypePtr 实例
SymFloatTypePtr SymFloatType::get() {
  static SymFloatTypePtr value(new SymFloatType());
  return value;
}
SymBoolTypePtr SymBoolType::get() {
  // 返回静态指针，如果尚未初始化，则创建一个新的SymBoolType对象
  static SymBoolTypePtr value(new SymBoolType());
  return value;
}

static std::optional<TypePtr> unifyTypesImpl(const TypePtr& t1, const TypePtr& t2, bool default_to_union=false, const TypePtr& type_hint=nullptr) {
  // 检查直接子类型关系
  if (t1->isSubtypeOf(*t2)) {
    return t2;
  } else if (t2->isSubtypeOf(*t1)) {
    return t1;
  }

  // 处理非容器类型的统一问题，它们不相互子类型化
  if (t1->kind() == TensorType::Kind && t2->kind() == TensorType::Kind) {
    // 合并两个TensorType对象
    return t1->expectRef<TensorType>().merge(t2->expectRef<TensorType>());
  }

  if (t1->isSubtypeOf(*NoneType::get()) && !t2->isSubtypeOf(*NoneType::get())) {
    // 统一Optional[t2] => OptionalType::create(t2)
    return OptionalType::create(t2);
  } else if (t2->isSubtypeOf(*NoneType::get()) && !t1->isSubtypeOf(*NoneType::get())) {
    // 统一Optional[t1] => OptionalType::create(t1)
    return OptionalType::create(t1);
  }

  // 不返回NumberType，因为目前运算符支持不足够

  // 尝试统一完整的Tensor类型，用于不可变类型容器

  // unify(Optional[t1], t2) => Optional[unify(t1, t2)]
  if (auto opt_t1 = t1->cast<OptionalType>()) {
    if (auto elem = unifyTypes(opt_t1->getElementType(), t2)) {
      return OptionalType::create(*std::move(elem));
    }
  } else if (auto opt_t2 = t2->cast<OptionalType>()) {
    if (auto elem = unifyTypes(opt_t2->getElementType(), t1)) {
      return OptionalType::create(*std::move(elem));
    }
  }

  if (t1->castRaw<TupleType>() && t2->castRaw<TupleType>()) {
    // 统一TupleType类型
    auto tuple1 = t1->castRaw<TupleType>();
    auto tuple2 = t2->castRaw<TupleType>();
    if (tuple1->elements().size() != tuple2->elements().size()) {
      return c10::nullopt;
    }
    std::vector<TypePtr> elements;
    for (size_t i = 0; i < tuple1->elements().size(); i++) {
      if (auto elem = unifyTypes(tuple1->elements().at(i), tuple2->elements().at(i), default_to_union)) {
        elements.push_back(*std::move(elem));
      } else {
        return c10::nullopt;
      }
    }
    return static_cast<TypePtr>(TupleType::create(std::move(elements)));
  }

  if (t1->castRaw<FutureType>() && t2->castRaw<FutureType>()) {
    // 统一FutureType类型
    if (auto elem = unifyTypes(
            t1->castRaw<FutureType>()->getElementType(),
            t2->castRaw<FutureType>()->getElementType())) {
      return FutureType::create(*elem);
    }
  }

  // 再次检查Unshaped类型的直接子类型关系，以处理可能包含两种不同专门化张量的可变容器类型（ListType / DictType）
  auto t1_unshaped = unshapedType(t1);
  auto t2_unshaped = unshapedType(t2);

  if (t1_unshaped->isSubtypeOf(*t2_unshaped)) {
    return t2_unshaped;
  } else if (t2_unshaped->isSubtypeOf(*t1_unshaped)) {
    return t1_unshaped;
  }
    return t1_unshaped;
  }

  // Check whether or not `type_hint` is a common parent. This case
  // could occur if we had two class types that had been annotated with
  // a common interface
  // 检查 `type_hint` 是否为两个类型 `t1` 和 `t2` 的共同父类。
  // 这种情况可能发生在我们有两个类类型，它们都被注解为一个共同的接口时。
  if (type_hint && t1->isSubtypeOf(*type_hint) && t2->isSubtypeOf(*type_hint)) {
    // 如果 `type_hint` 是 `t1` 和 `t2` 的共同父类，则返回 `type_hint`
    return type_hint;
  }

  // 如果以上条件不满足，则返回空值（`c10::nullopt`）
  return c10::nullopt;
}

// unifyTypes 函数：尝试统一两个类型指针 t1 和 t2，返回一个包含统一结果的 optional 对象
std::optional<TypePtr> unifyTypes(const TypePtr& t1, const TypePtr& t2, bool default_to_union, const TypePtr& type_hint) {
  // 调用 unifyTypesImpl 函数尝试统一类型 t1 和 t2
  auto unified = unifyTypesImpl(t1, t2, default_to_union, type_hint);

  // 如果设置了 default_to_union 并且未统一成功，则返回一个包含 t1 和 t2 的联合类型
  if (default_to_union && !unified) {
    return UnionType::create({t1, t2});
  }

  // 返回统一的结果
  return unified;
}

// unifyTypeList 函数：尝试统一一个类型指针数组 elements，返回统一后的类型指针或空 optional 对象
std::optional<TypePtr> unifyTypeList(
    at::ArrayRef<TypePtr> elements,
    std::ostream& why_not,
    bool default_to_union,
    const TypePtr& type_hint) {
  // 如果 elements 为空，输出错误信息并返回空 optional 对象
  if (elements.empty()) {
    why_not << "Cannot get unified type from empty list";
    return c10::nullopt;
  }

  // 初始化返回类型为第一个元素的类型
  TypePtr ret_type = elements.at(0);
  // 遍历 elements 中的每个元素，尝试统一类型
  for (size_t i = 1; i < elements.size() && ret_type; ++i) {
    // 尝试统一 ret_type 和当前元素 elements.at(i)，得到统一后的类型或空 optional 对象
    std::optional<TypePtr> maybe_unified = unifyTypes(ret_type, elements.at(i), default_to_union, type_hint);
    // 如果统一失败，输出错误信息并返回空 optional 对象
    if (!maybe_unified) {
      why_not << "Could not unify type list since element " << i << " of type "
              << elements.at(i)->repr_str()
              << " did not match the types before it ("
              << ret_type->repr_str() << ")";
      return c10::nullopt;
    }
    // 更新 ret_type 为统一后的类型
    ret_type = *maybe_unified;
  }

  // 返回统一后的类型
  return ret_type;
}

// matchTypeVariables 函数：匹配类型变量 formal 和 actual，更新 type_env，并返回匹配结果
MatchTypeReturn matchTypeVariables(
    const TypePtr& formal,
    const TypePtr& actual,
    TypeEnv& type_env) {
  // 如果 formal 没有自由变量，尝试匹配动态类型的 fallback，或直接成功
  if (!formal->hasFreeVariables()) {
    if (auto dyn = formal->castRaw<c10::DynamicType>()) {
      return matchTypeVariables(dyn->fallback(), actual, type_env);
    }
    return MatchTypeReturn::Success();
  }

  // 如果 formal 是变量类型 VarType
  if (auto vt = formal->castRaw<VarType>()) {
    // 查找变量名 vt->name() 在 type_env 中的绑定
    auto it = type_env.find(vt->name());
    // 如果没有找到，将 vt->name() 和 actual 绑定到 type_env 中，并返回成功
    if (it == type_env.end()) {
      type_env[vt->name()] = actual;
      return MatchTypeReturn::Success();
    }
    // 如果找到绑定，尝试统一当前绑定和 actual
    else if (unifyTypes(it->second, actual)) {
      // 注意：unifyTypes 允许两个类型之间的子类型化，所以 actual 可能是当前绑定的超类型
      // 这里我们只需要保持 type_env 的稳定，不需要报告错误
      return MatchTypeReturn::Success();
    }
    // 统一失败，输出错误信息
    std::stringstream ss;
    ss << "Type variable '" << vt->name() << "' previously matched to type "
       << it->second->repr_str() << " is matched to type "
       << actual->repr_str();
    return ss.str();
  }
  // 如果 formal 是列表类型 ListType
  else if (auto lt_formal = formal->castRaw<ListType>()) {
    // 如果 actual 也是列表类型 ListType
    if (auto lt_actual = actual->castRaw<ListType>()) {
      // 递归匹配列表元素类型
      auto innerMatch = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      // 如果内部匹配不成功，向外传播错误信息
      if (!innerMatch.success()) {
        return innerMatch;
      }
      // 内部匹配成功，返回成功
      return MatchTypeReturn::Success();
    }
    // actual 不是列表类型，返回匹配错误信息
    else {
      return "Type mismatch: expected ListType but got " + actual->repr_str();
    }
  }

  // 其它类型，返回错误信息
  return "Unsupported type encountered during type matching";
}
    } else if (auto tup_type = actual->castRaw<TupleType>()) {
      // 如果实际类型是元组类型，进入条件分支
      std::stringstream ss;
      // 创建字符串流对象 ss
      auto maybe_tuple_unified = unifyTypeList(tup_type->elements(), ss);
      // 尝试统一元组类型列表，将结果存储在 maybe_tuple_unified 中
      if (maybe_tuple_unified) {
        // 如果统一成功，则递归匹配元素类型变量
        return matchTypeVariables(
            lt_formal->getElementType(), *maybe_tuple_unified, type_env);
      }
    }

    // 如果前面的条件都不符合，则生成错误信息字符串
    std::stringstream ss;
    ss << "Cannot match " << lt_formal->repr_str() << " to "
       << actual->repr_str();
    // 返回生成的错误信息字符串
    return ss.str();
  } else if (auto tp_formal = formal->castRaw<TupleType>()) {
    // 如果形式类型是元组类型
    if (auto tp_actual = actual->castRaw<TupleType>()) {
      // 如果实际类型也是元组类型
      if (tp_formal->elements().size() != tp_actual->elements().size()) {
        // 如果元组元素数量不匹配，则返回错误信息
        return MatchTypeReturn("Cannot match tuples of mismatched size");
      }
      // 遍历元组的每个元素，逐一匹配类型变量
      for (size_t i = 0; i < tp_formal->elements().size(); ++i) {
        auto result = matchTypeVariables(
            tp_formal->elements()[i], tp_actual->elements()[i], type_env);
        if (!result.success()) {
          // 如果匹配失败，则返回失败结果
          return result;
        }
      }
      // 如果所有元素匹配成功，则返回成功结果
      return MatchTypeReturn::Success();
    } else {
      // 如果实际类型不是元组类型，则生成错误信息字符串
      std::stringstream ss;
      ss << "Cannot match a tuple to " << actual->repr_str();
      // 返回生成的错误信息字符串
      return MatchTypeReturn(ss.str());
    }
  } else if (auto lt_formal = formal->castRaw<FutureType>()) {
    // 如果形式类型是 FutureType
    if (auto lt_actual = actual->castRaw<FutureType>()) {
      // 如果实际类型也是 FutureType，则递归匹配内部元素类型
      auto innerMatch = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerMatch.success()) {
        // 如果匹配失败，则返回失败结果
        return innerMatch;
      }
      // 如果匹配成功，则返回成功结果
      return MatchTypeReturn::Success();
    } else {
      // 如果实际类型不是 FutureType，则生成错误信息字符串
      std::stringstream ss;
      ss << "Cannot match a future to " << actual->repr_str();
      // 返回生成的错误信息字符串
      return ss.str();
    }
  } else if (auto lt_formal = formal->castRaw<AwaitType>()) {
    // 如果形式类型是 AwaitType
    if (auto lt_actual = actual->castRaw<AwaitType>()) {
      // 如果实际类型也是 AwaitType，则递归匹配内部元素类型
      auto innerMatch = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerMatch.success()) {
        // 如果匹配失败，则返回失败结果
        return innerMatch;
      }
      // 如果匹配成功，则返回成功结果
      return MatchTypeReturn::Success();
    } else {
      // 如果实际类型不是 AwaitType，则生成错误信息字符串
      std::stringstream ss;
      ss << "Cannot match an await to " << actual->repr_str();
      // 返回生成的错误信息字符串
      return ss.str();
    }
  } else if (auto lt_formal = formal->castRaw<RRefType>()) {
    // 如果形式类型是 RRefType
    if (auto lt_actual = actual->castRaw<RRefType>()) {
      // 如果实际类型也是 RRefType，则递归匹配内部元素类型
      auto innerMatch = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerMatch.success()) {
        // 如果匹配失败，则返回失败结果
        return innerMatch;
      }
      // 如果匹配成功，则返回成功结果
      return MatchTypeReturn::Success();
    } else {
      // 如果实际类型不是 RRefType，则生成错误信息字符串
      std::stringstream ss;
      ss << "Cannot match a rref to " << actual->repr_str();
      // 返回生成的错误信息字符串
      return ss.str();
    }
  } else if (auto opt_formal = formal->castRaw<OptionalType>()) {
    // 如果形式类型是 OptionalType
    if (auto opt_actual = actual->castRaw<OptionalType>()) {
      // 如果实际类型也是 OptionalType，则递归匹配内部元素类型
      auto optionedMatch = matchTypeVariables(
          opt_formal->getElementType(), opt_actual->getElementType(), type_env);
      if (!optionedMatch.success()) {
        // 如果匹配失败，则返回失败结果
        return optionedMatch;
      }
      // 如果匹配成功，则返回成功结果
      return MatchTypeReturn::Success();
    }

          // 如果实际类型不是 OptionalType，则生成错误信息字符串
          std::stringstream ss;
          ss << "Cannot match an optional to " << actual->repr_str();
          // 返回生成的错误信息字符串
          return ss.str();
        }
      }
    }
    } else if (!actual->isSubtypeOf(*NoneType::get())) {
      // 如果实际类型不是 NoneType 的子类型
      // 允许将非可选类型匹配到形式参数，前提是其元素类型与实际类型匹配。
      // 不匹配 None，因为它已经是一个可选类型（但类型未知）。
      return matchTypeVariables(opt_formal->getElementType(), actual, type_env);
    }
    // 注意：如果 actual 是 None，则在这里我们可能没有填充形式参数中的类型变量。
    // 这仍然是一个有效的匹配，因为 None 可以匹配到 Optional[T]，稍后的错误检查 tryEvalTypeVariables 将在不匹配类型 T 变量时报告问题。
    return MatchTypeReturn::Success();
  } else if (auto dict_formal = formal->castRaw<DictType>()) {
    // 如果形式参数是字典类型
    if (auto dict_actual = actual->castRaw<DictType>()) {
      // 如果实际参数也是字典类型
      // 匹配字典类型的键类型
      auto key_match = matchTypeVariables(
          dict_formal->getKeyType(), dict_actual->getKeyType(), type_env);
      if (!key_match.success()) {
        return key_match;
      }
      // 匹配字典类型的值类型
      auto value_match = matchTypeVariables(
          dict_formal->getValueType(), dict_actual->getValueType(), type_env);
      if (!value_match.success()) {
        return value_match;
      }
      return MatchTypeReturn::Success();
    } else {
      // 如果实际参数不是字典类型，报错
      std::stringstream ss;
      ss << "Cannot match a dict to " << actual->repr_str();
      return ss.str();
    }
  }

  // 如果形式参数既不是 Optional 类型也不是字典类型，报错
  AT_ERROR("Unhandled free variable container: ", formal->repr_str());
}

// 此函数尝试将类型中的自由变量解析为具体类型，使用给定的类型环境
TORCH_API TypePtr tryEvalTypeVariables(const TypePtr& type, std::unordered_map<std::string, TypePtr>& type_env) {
  // 如果类型没有自由变量，则直接返回类型本身
  if (!type->hasFreeVariables()) {
    // 如果类型是动态类型，则尝试使用其回退类型再次调用此函数
    if (auto dyn = type->castRaw<c10::DynamicType>()) {
      return tryEvalTypeVariables(dyn->fallback(), type_env);
    }
    // 否则返回原始类型
    return type;
  }

  // 如果类型是变量类型，则根据类型环境尝试解析变量名为具体类型
  if (auto vt = type->castRaw<VarType>()) {
    auto it = type_env.find(vt->name());
    if (it == type_env.end()) {
      return nullptr;  // 如果类型环境中找不到对应的变量名，则返回空指针
    }
    return it->second;  // 返回找到的具体类型
  } else {
    // 如果类型不是变量类型，则递归处理其包含的类型
    at::ArrayRef<TypePtr> contained = type->containedTypes();
    if (contained.empty()) {
      return type;  // 如果没有包含的类型，则直接返回原始类型
    }
    std::vector<TypePtr> new_contained;
    new_contained.reserve(contained.size());
    // 对每个包含的类型进行尝试解析自由变量
    for (const TypePtr& t : contained) {
      TypePtr r = tryEvalTypeVariables(t, type_env);
      if (!r) {
        return nullptr;  // 如果解析失败，则返回空指针
      }
      new_contained.push_back(std::move(r));
    }
    // 使用解析后的类型构建新的类型对象
    return type->withContained(std::move(new_contained));
  }
}

// 判断元素类型是否可以从其成员推断出来
TORCH_API bool elementTypeCanBeInferredFromMembers(const TypePtr& elem_type) {
  // 如果元素类型是内置的联合类型、可选类型或数值类型，则无法推断元素类型
  if (elem_type->kind() == UnionType::Kind
      || elem_type->kind() == OptionalType::Kind
      || elem_type->kind() == NumberType::Kind) {
    return false;
  }
  // 如果元素类型是接口类型，则无法从其成员确定列表持有的接口类型
  if (elem_type->kind() == InterfaceType::Kind) {
    return false;
  }
  // 如果元素类型是任意类型，则列表可以包含异构类型
  if (elem_type->kind() == AnyType::Kind) {
    return false;
  }
  // 其他情况下可以推断元素类型
  return true;
}

// 将类型种类转换为字符串表示
const char * typeKindToString(TypeKind kind) {
#define CASE_TYPE(T) case TypeKind::T: return #T;
  switch(kind) {
    C10_FORALL_TYPES(CASE_TYPE)  // 使用C10_FORALL_TYPES宏展开所有类型
  }
#undef CASE_TYPE
  return "";  // 默认返回空字符串
}

// 扩展的类型子类型判断函数，检查当前类型是否是rhs的子类型
bool Type::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  // 如果rhs是任意类型或者两个类型相等，则当前类型是rhs的子类型
  if (rhs.kind() == TypeKind::AnyType || *this == rhs) {
    return true;
  }
  // 如果rhs是可选类型，则递归检查当前类型是否是其元素类型的子类型
  if (auto opt_rhs = rhs.castRaw<OptionalType>()) {
    return this->isSubtypeOfExt(*opt_rhs->getElementType(), why_not);
  }
  // 如果rhs是联合类型，则检查当前类型是否是其包含类型中任意一个的子类型
  if (auto union_rhs = rhs.castRaw<UnionType>()) {
    return std::any_of(union_rhs->containedTypes().begin(),
                       union_rhs->containedTypes().end(),
                       [&](const TypePtr& inner) {
                         return this->isSubtypeOfExt(*inner, why_not);
                       });
  }
  // 如果rhs是动态类型，则创建当前类型的动态类型并检查是否是rhs的子类型
  if (auto dyn = rhs.castRaw<DynamicType>()) {
    return DynamicType::create(*this)->isSubtypeOf(*dyn);
  }
  return false;  // 其他情况下不是子类型
}

// 判断类型是否是模块类型的函数，默认返回false
bool Type::is_module() const {
  return false;
}

// 创建具名元组类型的静态方法
TupleTypePtr TupleType::createNamed(
    const std::optional<c10::QualifiedName>& qualName,
    const std::vector<std::string>& field_names,
    const std::vector<TypePtr>& field_types) {
  std::vector<IValue> empty_defaults;  // 创建空的默认值向量
  return TupleType::createNamed(qualName, field_names, field_types, empty_defaults);  // 调用另一个重载的静态方法
}
// 创建具名元组类型，并返回指向该类型的智能指针
TupleTypePtr TupleType::createNamed(
    const std::optional<c10::QualifiedName>& qualName,  // 可选的限定名，用于标识元组类型
    const std::vector<c10::string_view>& field_names,   // 元组字段名的视图数组
    const std::vector<TypePtr>& field_types) {          // 元组字段类型的智能指针数组
  std::vector<IValue> empty_defaults;                   // 空的默认值数组
  return createWithSpec(qualName, field_names, field_types, empty_defaults);  // 调用带规格的创建函数
}

// 创建具名元组类型，并返回指向该类型的智能指针（重载函数，包含默认值）
TupleTypePtr TupleType::createNamed(
    const std::optional<c10::QualifiedName>& qualName,  // 可选的限定名，用于标识元组类型
    const std::vector<std::string>& field_names,        // 元组字段名的字符串数组
    const std::vector<TypePtr>& field_types,            // 元组字段类型的智能指针数组
    std::vector<IValue>& field_defaults) {              // 元组字段的默认值数组
  return createWithSpec(qualName, field_names, field_types, field_defaults);  // 调用带规格的创建函数
}

// 使用特定规格创建元组类型，并返回指向该类型的智能指针
template <typename S>
TupleTypePtr TupleType::createWithSpec(const std::optional<c10::QualifiedName>& qualName,
    const std::vector<S>& field_names,                  // 元组字段名数组
    const std::vector<TypePtr>& field_types,            // 元组字段类型的智能指针数组
    std::vector<IValue>& field_defaults) {              // 元组字段的默认值数组
  TORCH_INTERNAL_ASSERT(field_names.size() == field_types.size());  // 断言确保字段名和字段类型数组大小相同

  std::vector<Argument> arguments;                      // 参数列表
  arguments.reserve(field_names.size());                // 预留参数空间
  auto min_default_idx = field_names.size() - field_defaults.size();  // 最小默认值索引
  for (size_t i = 0; i < field_names.size(); ++i) {     // 遍历字段名数组
    if (i < min_default_idx) {                          // 如果索引小于最小默认值索引
      Argument arg{
          /*name=*/std::string{field_names[i]},         // 参数名
          /*type=*/field_types[i],                      // 参数类型
          /*N=*/i};                                     // 参数序号
      arguments.emplace_back(std::move(arg));           // 添加参数到参数列表
    }
    else {                                              // 否则（有默认值的情况）
      size_t j = i - min_default_idx;                   // 计算默认值数组索引
      TORCH_CHECK(field_defaults[j].tagKind() != "Tensor", "Tensors are "
                  "not supported as default NamedTuple fields. Their "
                  "mutability could lead to potential memory aliasing "
                  "problems");                          // 检查默认值不是张量类型
      Argument arg{
          /*name=*/std::string{field_names[i]},         // 参数名
          /*type=*/field_types[i],                      // 参数类型
          /*N=*/i,
          /*default_value=*/field_defaults[j]};          // 默认值
      arguments.emplace_back(std::move(arg));           // 添加参数到参数列表
    }
  }

  auto schema = std::make_shared<FunctionSchema>(
      /*name=*/qualName.value_or(c10::QualifiedName()).name(),  // 构造函数模式的名称
      /*overload_name=*/std::string(""),                // 重载名称为空字符串
      /*arguments=*/std::move(arguments),               // 参数列表
      /*returns=*/std::vector<Argument>{});             // 返回值为空的参数列表
  return std::shared_ptr<TupleType>(new TupleType(
      field_types, qualName, std::move(schema)));       // 返回具名元组类型的智能指针
}

// 返回元组类型的字段名的视图数组（如果存在）
std::optional<std::vector<c10::string_view>> TupleType::names() const {
  if (!schema_) {                                       // 如果模式为空
    return {};                                          // 返回空的可选值
  }
  std::vector<c10::string_view> ret;                    // 结果数组
  for (const auto& arg : schema_->arguments()) {        // 遍历模式中的参数
    ret.emplace_back(arg.name());                       // 将参数名添加到结果数组
  }
  return ret;                                           // 返回结果数组
}

// 检查当前类型是否是 rhs 的子类型（扩展版本，包括 OptionalType 类型）
bool NoneType::isSubtypeOfExt(const Type& rhs, std::ostream *why_not) const {
  if (rhs.kind() == OptionalType::Kind) {               // 如果 rhs 是 OptionalType 类型
    return true;                                        // 返回 true
  }
  return Type::isSubtypeOfExt(rhs, why_not);            // 否则调用基类的类型扩展检查
}

// 检查当前类型是否与 rhs 类型相等（如果 rhs 是 UnionType，则检查包含的类型）
bool NumberType::equals(const Type& rhs) const {
  if (auto union_type = rhs.cast<UnionType>()) {        // 如果 rhs 能够转换为 UnionType 类型
    return union_type->containedTypes().size() == 3 && union_type->canHoldType(*NumberType::get());  // 检查包含的类型数量和能否容纳当前类型
  } else {
    return rhs.kind() == this->kind();                 // 否则比较类型的种类
  }
}
bool NumberType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
    // 检查是否可以转换为 UnionType
    if (auto union_type = rhs.cast<UnionType>()) {
        // 如果可以转换为 UnionType，则检查 UnionType 是否可以容纳 NumberType
        return union_type->canHoldType(*NumberType::get());
    } else {
        // 否则调用基类的 isSubtypeOfExt 方法进行处理
        return Type::isSubtypeOfExt(rhs, why_not);
    }
}

TupleType::TupleType(
    std::vector<TypePtr> elements,
    std::optional<c10::QualifiedName> name,
    std::shared_ptr<FunctionSchema> schema)
    : NamedType(TypeKind::TupleType, std::move(name)),
      elements_(std::move(elements)),
      // 检查元组中是否包含 None 类型，并设置 has_free_variables_ 标志
      has_free_variables_(std::any_of(elements_.begin(), elements_.end(), [](const TypePtr& v) {
        if (!v) {
          throw std::runtime_error("Can not create tuple with None type");
        }
        return v->hasFreeVariables();
      })),
      schema_(std::move(schema)) {

  // 如果存在 schema，则检查其中的参数是否满足特定的规则
  if (schema_) {
    for (const Argument& arg : schema_->arguments()) {
      checkNoAny(*this, "attribute", arg.name(), arg.type());
    }
  }
}

bool TupleType::isSubtypeOfExt(const Type& rhs_, std::ostream* why_not) const {
    // 首先调用基类的 isSubtypeOfExt 方法，如果返回 true，则直接返回 true
    if (Type::isSubtypeOfExt(rhs_, why_not)) {
        return true;
    }
    // 如果 rhs_ 是 AnyTupleType 类型，则返回 true
    if (rhs_.kind() == AnyTupleType::Kind) {
        return true;
    }
    // 尝试将 rhs_ 转换为 TupleType
    auto rhs = rhs_.cast<TupleType>();
    if (!rhs)
        return false;
    
    // 未命名元组不是命名元组的子类型
    if (!schema() && rhs->schema())
        return false;
    
    // 命名元组可能是未命名元组的子类型，检查参数名是否匹配
    auto test_names_match = [&](const std::shared_ptr<FunctionSchema>& lhs, const std::shared_ptr<FunctionSchema>& rhs) {
        const auto& args_lhs = lhs->arguments();
        const auto& args_rhs = rhs->arguments();
        if (args_lhs.size() != args_rhs.size()) {
            return false;
        }

        for (size_t i = 0; i < args_lhs.size(); ++i) {
            if (args_lhs[i].name() != args_rhs[i].name()) {
                return false;
            }
        }
        return true;
    };
    
    // 检查参数名是否匹配，并且比较元组的结构
    bool names_match = !rhs->schema() || test_names_match(schema(), rhs->schema());
    // 协变规则适用于元组
    return names_match && compare(*rhs, [&](const Type& a, const Type& b) {
        return a.isSubtypeOfExt(b, why_not);
    });
}

bool ListType::isSubtypeOfExt(const Type& rhs_, std::ostream* why_not) const {
    // 调用基类的 isSubtypeOfExt 方法，如果返回 true，则直接返回 true
    if (Type::isSubtypeOfExt(rhs_, why_not)) {
        return true;
    }
    // 如果 rhs_ 是 AnyListType 类型，则返回 true
    if (rhs_.kind() == AnyListType::Kind) {
        return true;
    }
    // 其它情况返回 false
    return false;
}

bool TupleType::equals(const Type& rhs) const {
    // 使用 compare 函数比较两个类型是否相同
    bool typesSame =
        compare(rhs, [](const Type& a, const Type& b) { return a == b; });
    if (!typesSame) {
        return false;
    }

    // 根据 compare 函数的保证，rhs 一定是 TupleType 类型
    auto rhsTuple = rhs.expect<TupleType>();
    // 如果两者的 schema 都为 nullptr，则返回 typesSame
    if (schema_ == nullptr && rhsTuple->schema_ == nullptr) {
        return typesSame;
    }
    // 如果两者的 schema 有一个为 nullptr，则返回 false
    if (schema_ == nullptr || rhsTuple->schema_ == nullptr) {
        return false;
    }
    // 比较两个 schema 是否相同
    return *schema_ == *rhsTuple->schema_;
}

std::string TupleType::str() const {
    std::stringstream ss;
    // 如果存在 schema 和 name，则输出 name 的限定名称
    if (schema_ && name()) {
        ss << name()->qualifiedName();
    } else {
        ss << "(";
    # 循环遍历 elements() 返回的元素列表
    for(size_t i = 0; i < elements().size(); ++i) {
      # 如果 i 大于 0，则在字符串流 ss 中添加逗号和空格
      if(i > 0)
        ss << ", ";
      # 将 elements() 返回的第 i 个元素的字符串表示追加到字符串流 ss 中
      ss << elements()[i]->str();
    }
    # 在字符串流 ss 中添加右括号，表示元素列表结束
    ss << ")";
  }
  # 返回字符串流 ss 中构建的最终字符串表示
  return ss.str();
}
// 实现获取类型注解字符串的函数，返回值为 std::string 类型
std::string TupleType::annotation_str_impl(const TypePrinter& printer) const {
  // 如果存在 schema_ 并且有名称，则返回其限定名称
  if (schema_ && name()) {
    return name()->qualifiedName();
  }

  // 如果元素为空，特殊处理空元组的注解语法
  if (elements().empty()) {
    // `typing.Tuple` 会特殊处理空元组的注解语法为 `typing.Tuple[()]`，参见官方文档
    return "Tuple[()]";
  }

  // 对于预期较小的元组，采用快速路径处理
  const auto elts = elements();
  if (elts.size() <= 3) {
    std::array<std::string, 3> elements_strs;
    size_t total_length = 0;
    int idx = 0;
    for (const auto& element: elts) {
      elements_strs[idx] = element->annotation_str(printer);
      total_length += elements_strs[idx].size();
      idx++;
    }
    std::string result;
    result.reserve(strlen("Tuple[") + strlen(", ") * (elts.size() - 1) + total_length + 1);
    result.append("Tuple[");
    for (const auto ii : c10::irange(elts.size())) {
      if (ii > 0) {
        result.push_back(',');
        result.push_back(' ');
      }
      result.append(elements_strs[ii]);
    }
    result.push_back(']');
    return result;
  }

  // 对于较大的元组，使用 stringstream 进行构建结果字符串
  std::ostringstream ss;
  ss << "Tuple[";
  size_t i = 0;
  for (const auto& element: elts) {
    if (i > 0) {
      ss << ", ";
    }
    ss << element->annotation_str(printer);
    i++;
  }
  ss << ']';
  return std::move(ss).str();
}

// 创建接口类型对象的静态方法，返回值为 InterfaceTypePtr
InterfaceTypePtr InterfaceType::create(QualifiedName qualifiedName, bool is_module) {
  // 使用移动语义创建 InterfaceType 对象并返回指针
  return InterfaceTypePtr(
      new InterfaceType(std::move(qualifiedName), is_module));
}

// FunctionType 的构造函数，初始化 NamedType 基类和 function_ 成员变量
FunctionType::FunctionType(torch::jit::Function* function)
  : NamedType(TypeKind::FunctionType, function->qualname()),
    function_(function) {}

// 接口类型的子类型判断方法
bool InterfaceType::isSubTypeImpl(
    const InterfaceType& lhs,
    const InterfaceType& rhs,
    std::ostream* why_not) {
  // 如果左侧接口不是模块而右侧接口是模块，则报错并返回 false
  if (!lhs.is_module() && rhs.is_module()) {
    if (why_not) {
      *why_not << "Interface '" << lhs.repr_str() << "' is not a subtype of "
               << "the module interface '" << rhs.repr_str() << "'.\n";
    }
    return false;
  }

  // 遍历右侧接口的方法列表
  for (const FunctionSchema& schema : *rhs.methods_) {
    // 获取左侧接口对应方法的 schema
    auto self_schema = lhs.getMethod(schema.name());
    if (!self_schema) {
      // 如果左侧接口缺少右侧接口的方法，则报错并返回 false
      if (why_not) {
        *why_not << "Interface '" << lhs.repr_str()
                 << "' does not have method '" << schema.name() << "' but interface '"
                 << rhs.repr_str() << "' does.\n";
      }
      return false;
    }
    // 检查方法的子类型关系，若不符合则报错并返回 false
    // NOLINTNEXTLINE(bugprone-argument-comment)
    if (!self_schema->isSubtypeOf(schema, /*is_method=*/true, why_not)) {
      if (why_not) {
        *why_not << "Method on interface '" << lhs.repr_str()
                 << "' (1) is not compatible with interface '"
                 << rhs.repr_str() << "' (2)\n"
                 << "  (1) " << *self_schema << "\n"
                 << "  (2) " << schema << "\n";
        return false;
      }
      return false;
    }
  }
  return true;
}
bool InterfaceType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  // 如果 rhs 是 InterfaceType 类型，则进行具体的子类型检查
  if (auto iface = rhs.castRaw<InterfaceType>()) {
    return isSubTypeImpl(*this, *iface, why_not);
  }
  // 否则调用父类的 isSubtypeOfExt 方法进行检查
  return Type::isSubtypeOfExt(rhs, why_not);
}

const FunctionSchema* InterfaceType::getMethod(const std::string& name) const {
  // 遍历存储在 methods_ 中的所有方法，查找名称匹配的方法
  for (const FunctionSchema& method : *methods_) {
    if (method.name() == name) {
      // 如果找到则返回该方法的指针
      return &method;
    }
  }
  // 如果找不到则返回空指针
  return nullptr;
}

void InterfaceType::addMethod(FunctionSchema schema) {
  // 向 methods_ 中添加新的方法 schema
  methods_->emplace_back(std::move(schema));
}

InterfaceType::InterfaceType(QualifiedName name, bool is_module)
    : NamedType(InterfaceType::Kind, std::move(name)),
      methods_(std::make_shared<std::vector<FunctionSchema>>()),
      is_module_(is_module) {
  // InterfaceType 构造函数，初始化名称、方法集合和是否为模块标志
}

InterfaceType::~InterfaceType() = default;
// InterfaceType 析构函数，默认实现

bool containsAnyType(const TypePtr& type) {
  // 检查给定类型是否包含 AnyType 类型
  std::vector<TypePtr> to_scan = { type };
  while (!to_scan.empty()) {
    const auto typ = to_scan.back();
    to_scan.pop_back();
    if (typ->kind() == AnyType::Kind) {
      return true;
    }
    // 将该类型包含的所有子类型加入待检查列表
    for (const TypePtr& sub : typ->containedTypes()) {
      to_scan.emplace_back(sub);
    }
  }
  return false;
}

void checkNoAny(const Type& base, const char* what, const std::string& attrname, const TypePtr& attrtype) {
  // 检查要添加的属性类型是否包含 AnyType 类型，如果包含则抛出异常
  TORCH_CHECK(
      !containsAnyType(attrtype),
      "attempting to add ",
      what,
      " '",
      attrname,
      "' of type ",
      attrtype->repr_str(),
      " to '",
      base.repr_str(),
      "' but it contains an Any type. Any types cannot be members of modules, classes, or named tuples.");
}

SymbolicShape SymbolicShape::merge(const SymbolicShape& other) const {
  // 合并两个 SymbolicShape 对象的维度信息
  if (!dims_ || !other.dims_ || dims_->size() != other.dims_->size()) {
    return SymbolicShape(); // 如果维度信息不一致则返回空的 SymbolicShape
  }
  std::vector<ShapeSymbol> dims;
  // 逐个合并对应位置的维度信息
  for (size_t i = 0, n = dims_->size(); i < n; i++) {
    dims.push_back(merge_primitive((*dims_)[i], (*other.dims_)[i]));
  }
  return SymbolicShape(std::move(dims)); // 返回合并后的 SymbolicShape 对象
}

void SymbolicShape::dump() const {
  // 打印当前 SymbolicShape 对象的信息到标准输出
  std::cout << *this << "\n";
}

bool EnumType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  // 检查当前枚举类型是否是 rhs 的子类型，或者 rhs 是 AnyType 或 AnyEnumType
  return rhs.kind() == TypeKind::AnyType ||
      rhs.kind() == TypeKind::AnyEnumType ||
      *this == rhs ||
      Type::isSubtypeOfExt(rhs, why_not);
}
```