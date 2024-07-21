# `.\pytorch\aten\src\ATen\core\dynamic_type.cpp`

```
#include <ATen/core/dynamic_type.h>

#include <string>

#include <ATen/core/class_type.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/type_factory.h>
#include <c10/util/Exception.h>

namespace c10 {

namespace {

// 检查 DynamicType::Tag 是否包含指定的 DynamicTypeBits 标志
bool contains(DynamicType::Tag lhs, DynamicTypeBits rhs) {
  return (static_cast<DynamicTypeBits>(lhs) | rhs) ==
      static_cast<DynamicTypeBits>(lhs);
}

// 检查 DynamicType::Tag 是否包含指定的 DynamicType::Tag 标志
bool contains(DynamicType::Tag lhs, DynamicType::Tag rhs) {
  return contains(lhs, static_cast<DynamicTypeBits>(rhs));
}

} // namespace

namespace detail {

// 创建一个基础的 DynamicTypePtr，使用给定的 tag
DynamicTypePtr makeBaseType(DynamicType::Tag tag) {
  return std::make_shared<DynamicType>(tag, DynamicType::Arguments{});
}

} // namespace detail

// 返回 DynamicType 的字符串表示形式
std::string DynamicType::str() const {
  if (name_) {
    return *name_;
  }
  std::string ret = "Dynamic<";
  ret += std::to_string(static_cast<DynamicTypeBits>(tag_));
  ret += ">";
  if (tag_ != Tag::Class && !arguments_.elems.empty()) {
    ret += "[";
    for (const auto& arg : arguments_.elems) {
      if (arg.label) {
        ret += *arg.label + ":";
      }
      ret += arg.ty->str();
      ret += ",";
    }
    ret += "]";
  }
  return ret;
}

// 构造函数：通过 TypePtr 列表创建 Arguments 对象
DynamicType::Arguments::Arguments(c10::ArrayRef<TypePtr> args) {
  elems.reserve(args.size());
  for (const auto& arg : args) {
    elems.emplace_back(create(*arg));
  }
}

// 构造函数：通过 names 和 TypePtr 列表创建 Arguments 对象
DynamicType::Arguments::Arguments(
    const std::vector<c10::string_view>& names,
    c10::ArrayRef<TypePtr> args)
    : Arguments(args) {
  TORCH_INTERNAL_ASSERT(names.size() == args.size());
  for (size_t i = 0; i < args.size(); i++) {
    elems[i].label = std::string{names[i]};
  }
}

// 析构函数：根据 tag_ 的不同情况进行处理
DynamicType::~DynamicType() {
  if (tag_ == Tag::Class) {
    class_.~ClassTypePtr();
    return;
  }

  arguments_.~Arguments();
}

// 静态函数：根据另一个 Type 对象创建对应的 DynamicType 共享指针
std::shared_ptr<const DynamicType> DynamicType::create(const Type& other) {
  if (auto dynRaw = other.castRaw<DynamicType>()) {
    TORCH_INTERNAL_ASSERT(!dynRaw->weak_from_this().expired(),
        "Error creating dynamic type instance not managed by shared_ptr: ",
        other.str());
  }
  if (auto dyn = other.cast<DynamicType>()) {
    return dyn;
  }
  return std::shared_ptr<const DynamicType>(new DynamicType{other});
}

// 静态函数：根据另一个 Type 对象创建对应的 DynamicType 共享指针
DynamicTypePtr DynamicType::create(Type& other) {
  if (auto dynRaw = other.castRaw<DynamicType>()) {
    TORCH_INTERNAL_ASSERT(!dynRaw->weak_from_this().expired(),
        "Error creating dynamic type instance not managed by shared_ptr: ",
        other.str());
  }
  if (auto dyn = other.cast<DynamicType>()) {
    return dyn;
  }
  return std::shared_ptr<DynamicType>(new DynamicType{other});
}

// 构造函数：使用给定的 tag 和 arguments 创建 DynamicType 对象
DynamicType::DynamicType(Tag tag, Arguments arguments)
    : SharedType(Kind), tag_(tag), arguments_(std::move(arguments)) {}

// 构造函数：使用给定的 tag、name 和 arguments 创建 DynamicType 对象
DynamicType::DynamicType(Tag tag, c10::string_view name, Arguments arguments)
    : SharedType(Kind),
      tag_(tag),
      name_(std::string{name}),
      arguments_(std::move(arguments)) {}
// 复制构造函数，从另一个 Type 对象创建 DynamicType 对象，并初始化为 SharedType
DynamicType::DynamicType(const Type& other) : SharedType(DynamicType::Kind) {
  // 获取其他对象的类型
  auto kind = other.kind();
  // 断言确保类型不同
  TORCH_INTERNAL_ASSERT(kind != Kind);
  // 如果其他对象是 NamedType 类型
  if (auto n = other.castRaw<NamedType>()) {
    // 获取命名类型的限定名称，并存储到当前对象的 name_ 成员变量中
    if (const auto& qn = n->name()) {
      name_ = qn->qualifiedName();
    }
  } else if (auto v = other.castRaw<VarType>()) {
    // 如果其他对象是 VarType 类型，则将其名称存储到当前对象的 name_ 成员变量中
    name_ = v->name();
  }

  // 如果其他对象是 ClassType 类型
  if (auto cls = other.cast<ClassType>()) {
    // 将 ClassType 对象移动构造到当前对象的 class_ 成员变量中
    new (&class_) ClassTypePtr(std::move(cls));
    // 设置当前对象的 tag_ 为 Tag::Class
    tag_ = Tag::Class;
    return;
  }

  // 根据类型种类进行判断和处理
  switch (kind) {
#define CASE_TYPE(T, _, __) \
  case T##Type::Kind:       \
    // 设置当前对象的 tag_ 为对应的 Tag::T
    tag_ = Tag::T;          \
    break;
    // 遍历所有动态类型，生成对应的 case 语句
    FORALL_DYNAMIC_TYPES(CASE_TYPE)
    // FORALL_DYNAMIC_TYPES_FAKE 不在此处处理，因为这些动态类型映射到相同的 tag，
    // 所以它们始终解析为整数
#undef CASE_TYPE
    // 默认情况下，如果未知类型，则断言失败，并输出不支持的动态类型信息
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported dynamic type: ", other.str());
  }

  // 获取其他对象中包含的类型参数
  auto args = other.containedTypes();
  // 如果参数列表为空，则创建一个空的 Arguments 对象
  if (args.empty()) {
    new (&arguments_) Arguments();
    return;
  }

  // 如果其他对象是 TupleType 类型
  if (auto tup = other.castRaw<TupleType>()) {
    // 获取元组类型的名称，并基于元组的名称和参数创建 Arguments 对象
    if (auto names = tup->names()) {
      new (&arguments_) Arguments(*names, args);
      return;
    }
  }

  // 基于参数直接创建 Arguments 对象
  new (&arguments_) Arguments(args);
}

// 比较两个 DynamicType 对象是否相等
bool DynamicType::equals(const DynamicType& other) const {
  // 如果比较的是同一个对象，则返回 true
  if (this == &other) {
    return true;
  }
  // 如果标签不同，则返回 false
  if (tag_ != other.tag_) {
    return false;
  }
  // 根据标签类型进行具体的比较
  switch (tag_) {
    // 如果是 Class 类型，则比较其中的 ClassTypePtr 对象是否相等
    case Tag::Class:
      return *class_ == *other.class_;
    // 对于其他类型，比较它们的参数是否相等
    default:
      // 调用 compareArguments 函数比较参数列表中的每个 LabeledDynamicType 对象是否相等
      return compareArguments(
          other, [](const LabeledDynamicType& a, const LabeledDynamicType& b) {
            return a.equals(b);
          });
  }
}

// 比较当前 DynamicType 对象与给定 Type 对象是否相等
bool DynamicType::equals(const Type& rhs) const {
  // 创建一个新的 DynamicType 对象，并调用 equals 函数比较它们
  return equals(*create(rhs));
}

// 判断当前 DynamicType 对象是否是给定 Type 对象的子类型
bool DynamicType::isSubtypeOfExt(const Type& rhs, std::ostream*) const {
  // 创建一个新的 DynamicType 对象，并判断其标签与当前对象是否相同
  auto other = create(rhs);
  if (tag_ == other->tag_) {
    // 如果标签相同，则进一步比较它们是否相等
    if (equals(*other)) {
      return true;
    }
    // 如果标签支持动态协变类型位，则比较参数列表中的每个 LabeledDynamicType 对象
    if (contains(tag_, kDynamicCovariantTypeBit)) {
      if (compareArguments(
              *other,
              [](const LabeledDynamicType& a, const LabeledDynamicType& b) {
                return a.isSubtypeOf(b);
              })) {
        return true;
      };
    }
  } else if (contains(other->tag_, tag_)) {
    // 如果其他对象的标签包含当前对象的标签，则认为是子类型
    return true;
  }

  // 如果其他对象的标签为 Optional，则检查其参数列表第一个元素的类型是否是当前对象的子类型
  if (other->tag_ == Tag::Optional) {
    if (isSubtypeOf(other->arguments_.elems[0].ty)) {
      return true;
    }
  }

  // 否则返回 false
  return false;
}

// 获取当前 DynamicType 对象中第 i 个包含的类型
TypePtr DynamicType::containedType(size_t i) const {
  // 断言当前对象的标签不是 Class，因为 Class 类型没有包含的类型
  TORCH_INTERNAL_ASSERT(tag_ != Tag::Class);
  // 返回参数列表中第 i 个元素的类型指针
  return arguments_.elems.at(i).ty;
}

// 获取当前 DynamicType 对象中包含的类型数量
size_t DynamicType::containedTypeSize() const {
  // 断言当前对象的标签不是 Class，因为 Class 类型没有包含的类型
  TORCH_INTERNAL_ASSERT(tag_ != Tag::Class);
  // 返回参数列表的大小
  return arguments_.elems.size();
}

// 获取当前 DynamicType 对象的动态类型种类
TypeKind DynamicType::dynamicKind() const {
  // 根据当前对象的 tag_ 返回对应的 TypeKind
  switch (tag_) {
#define CASE_TYPE(T, _, __) \
  case Tag::T:              \
    return TypeKind::T##Type;
    // 遍历所有动态类型，生成对应的 case 语句
    FORALL_DYNAMIC_TYPES(CASE_TYPE)
    // FORALL_DYNAMIC_TYPES_FAKE 在此处被故意省略，因为这些动态类型映射到相同的 tag，
    // 所以它们始终解析为整数
#undef CASE_TYPE
    // 默认情况下，如果未知类型，则断言失败，并返回 AnyType
    default:
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
      return TypeKind::AnyType;
  }
}
// 返回当前动态类型对象的回退类型指针
TypePtr DynamicType::fallback() const {
  // 根据标签选择不同的类型并返回其类型指针
  switch (tag_) {
    case Tag::Tensor:
      return TensorType::get();
    case Tag::None:
      return NoneType::get();
    case Tag::Bool:
      return BoolType::get();
    case Tag::Int:
      return IntType::get();
    case Tag::Float:
      return FloatType::get();
    case Tag::Complex:
      return ComplexType::get();
    case Tag::Number:
      return NumberType::get();
    case Tag::String:
      return StringType::get();
    case Tag::List:
      // 返回列表类型的回退类型指针，其中元素类型是第一个元素的回退类型
      return ListType::create(arguments_.elems[0].ty->fallback());
    case Tag::Tuple: {
      // 准备存放各元素回退类型的向量
      std::vector<TypePtr> fallbacks;
      fallbacks.reserve(arguments_.elems.size());
      // 遍历每个元组元素并获取其回退类型，存入fallbacks向量
      for (const auto& elem : arguments_.elems) {
        fallbacks.push_back(elem.ty->fallback());
      }
      // 如果存在元组名称，则创建带有字段名的元组类型
      if (name_) {
        std::vector<c10::string_view> fields;
        fields.reserve(arguments_.elems.size());
        // 获取每个元素的字段名，并存入fields向量
        for (const auto& elem : arguments_.elems) {
          // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
          fields.emplace_back(*elem.label);
        }
        // 创建带有名称和字段的元组类型并返回
        return TupleType::createNamed(*name_, fields, fallbacks);
      }
      // 创建无名称的元组类型并返回
      return TupleType::create(std::move(fallbacks));
    }
    case Tag::Dict:
      // 返回字典类型的回退类型指针，其中键和值的类型分别为第一个和第二个元素的回退类型
      return DictType::create(
          arguments_.elems[0].ty->fallback(),
          arguments_.elems[1].ty->fallback());
    case Tag::Class:
      // 返回类类型的回退类型指针，使用已存储的类信息
      return std::make_shared<ClassType>(*class_);
    case Tag::Optional:
      // 返回可选类型的回退类型指针，其中元素类型是第一个元素的回退类型
      return OptionalType::create(arguments_.elems[0].ty->fallback());
    case Tag::AnyList:
      return AnyListType::get();
    case Tag::AnyTuple:
      return AnyTupleType::get();
    case Tag::DeviceObj:
      return DeviceObjType::get();
    case Tag::StreamObj:
      return StreamObjType::get();
    case Tag::Capsule:
      return CapsuleType::get();
    case Tag::Generator:
      return GeneratorType::get();
    case Tag::Storage:
      return StorageType::get();
    case Tag::Var:
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      // 返回变量类型的回退类型指针，使用存储的变量名称
      return VarType::create(*name_);
    case Tag::AnyClass:
      return AnyClassType::get();
    case Tag::QScheme:
      return QSchemeType::get();
    case Tag::Quantizer:
      return QuantizerType::get();
    case Tag::AnyEnum:
      return AnyEnumType::get();
    case Tag::RRef:
      // 返回远程引用类型的回退类型指针，其中元素类型是第一个元素的回退类型
      return RRefType::create(arguments_.elems[0].ty->fallback());
    case Tag::Future:
      // 返回异步类型的回退类型指针，其中元素类型是第一个元素的回退类型
      return FutureType::create(arguments_.elems[0].ty->fallback());
    case Tag::Await:
      // 返回等待类型的回退类型指针，其中元素类型是第一个元素的回退类型
      return AwaitType::create(arguments_.elems[0].ty->fallback());
    case Tag::Any:
      return AnyType::get();
  }
  // 如果未匹配任何标签，引发调试断言错误
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
  // 默认返回空指针
  return nullptr;
}

// 检查当前标签化动态类型对象是否是给定对象的子类型
bool DynamicType::LabeledDynamicType::isSubtypeOf(
    const LabeledDynamicType& other) const {
  // 如果给定对象没有标签或者当前标签等于给定对象的标签，则检查类型是否是子类型
  if (!other.label || (label == other.label)) {
    return ty->isSubtypeOf(other.ty);
  }
  // 否则返回false，表示不是子类型
  return false;
}

// 检查当前标签化动态类型对象是否与给定对象相等
bool DynamicType::LabeledDynamicType::equals(
    const LabeledDynamicType& other) const {
  // 返回当前标签和类型与给定对象的标签和类型是否完全相等
  return (label == other.label) && (*ty == *other.ty);
}
}

// 定义函数模板特化，用于从 IValue 中获取动态类型指针
DynamicType::Ptr IValue::TagType<c10::DynamicType>::get(const c10::IValue& v) {
  // 根据 IValue 的标签选择合适的动态类型
  switch (v.tag) {
    case Tag::None:
      return DynamicTypeTrait<NoneType>::getBaseType();  // 返回 NoneType 的基础类型指针
    case Tag::Tensor:
      return DynamicTypeTrait<TensorType>::getBaseType();  // 返回 TensorType 的基础类型指针
    case Tag::Double:
      return DynamicTypeTrait<FloatType>::getBaseType();  // 返回 FloatType 的基础类型指针
    case Tag::ComplexDouble:
      return DynamicTypeTrait<ComplexType>::getBaseType();  // 返回 ComplexType 的基础类型指针
    case Tag::Int:
      return DynamicTypeTrait<IntType>::getBaseType();  // 返回 IntType 的基础类型指针
    case Tag::Bool:
      return DynamicTypeTrait<BoolType>::getBaseType();  // 返回 BoolType 的基础类型指针
    case Tag::String:
      return DynamicTypeTrait<StringType>::getBaseType();  // 返回 StringType 的基础类型指针
    case Tag::GenericDict: {
      auto d = v.toGenericDict();
      return DynamicTypeFactory::create<DictType>(d.keyType(), d.valueType());  // 返回由 GenericDict 创建的 DictType 类型指针
    }
    case Tag::GenericList:
      return DynamicTypeFactory::create<ListType>(v.toList().elementType());  // 返回由 GenericList 创建的 ListType 类型指针
    case Tag::Device:
      return DynamicTypeTrait<DeviceObjType>::getBaseType();  // 返回 DeviceObjType 的基础类型指针
    case Tag::Stream:
      return DynamicTypeTrait<StreamObjType>::getBaseType();  // 返回 StreamObjType 的基础类型指针
    case Tag::Object:
      return v.toObjectRef().type();  // 返回对象引用中的类型指针
    case Tag::Capsule:
      return DynamicTypeTrait<CapsuleType>::getBaseType();  // 返回 CapsuleType 的基础类型指针
    case Tag::Tuple:
      return v.toTupleRef().type<c10::DynamicType>();  // 返回元组引用中的 DynamicType 类型指针
    default:
      return DynamicTypeTrait<AnyType>::getBaseType();  // 默认返回 AnyType 的基础类型指针
  }
}

// 创建 TupleType 对象的工厂函数，根据元素类型创建 TupleType 指针
DynamicTypePtr ivalue::TupleTypeFactory<c10::DynamicType>::create(
    const std::vector<TypePtr>& elemTypes) {
  return DynamicTypeFactory::create<TupleType>(elemTypes);  // 使用元素类型创建 TupleType 类型指针
}

// TupleTypeFactory 的回退函数，当创建失败时调用，抛出错误并返回空指针
DynamicTypePtr ivalue::TupleTypeFactory<c10::DynamicType>::fallback(
    const Type&) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);  // 断言，用于调试，表明不应该执行到这里
  return nullptr;  // 返回空指针
}

// 在 Torch API 中创建 TupleType 的回退函数，用于移动设备上不支持的情况
TORCH_API TupleTypePtr
ivalue::TupleTypeFactory<TupleType>::fallback(C10_UNUSED const Type& type) {
#ifdef C10_MOBILE
  return nullptr;  // 移动设备上返回空指针
#else
  const auto& dyn = type.expectRef<DynamicType>();
  std::vector<c10::string_view> fields;
  std::vector<TypePtr> types;

  // 遍历动态类型的参数，收集字段名和类型
  for (const auto& elem : dyn.arguments().elems) {
    types.emplace_back(elem.ty);
    if (const auto& name = elem.label) {
      fields.emplace_back(*name);
    }
  }
  // 如果动态类型有名称，创建带名称的 TupleType
  if (const auto& name = dyn.name()) {
    return TupleType::createNamed(*name, fields, types);
  }
  // 否则创建匿名 TupleType
  return TupleType::create(std::move(types));
#endif
}

} // namespace c10  // 结束命名空间 c10
```