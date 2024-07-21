# `.\pytorch\aten\src\ATen\core\ivalue.cpp`

```
namespace c10 {
    // 定义用于比较 IValue 对象的函数，判断是否是同一个对象或者具有相同的值
    bool _fastEqualsForContainer(const IValue& lhs, const IValue& rhs) {
        // 如果两个 IValue 对象的类型相同，则返回 true
        if (lhs.is(rhs)) {
            // 像 Python 一样，对于容器，我们认为身份相等足以但不必要是值相等的充分条件
            return true;
        }
        // 否则，使用默认的相等比较函数
        return lhs == rhs;
    }

    namespace ivalue {

        // 由于需要访问 Type::annotation_str，该函数定义在 jit_type.h 中
        // 检查实际类型是否与期望的自定义类类型相符
        void checkCustomClassType(const ClassType* expected_type, const Type* actual_type) {
            // 注意：这里进行指针比较
            // 如果未来需要在自定义类类型上调用 operator==，则需要修改此处！
            TORCH_CHECK(actual_type == static_cast<const Type*>(expected_type),
                        "Tried to convert an IValue of type ",
                        actual_type ? actual_type->repr_str() : std::string("*NULL*"),
                        " to custom class type ",
                        expected_type ? expected_type->repr_str() : std::string("*NULL*"));
        }

        // 创建 ConstantString 对象的静态方法，接受不同类型的字符串参数
        TORCH_API c10::intrusive_ptr<ConstantString> ConstantString::create(
            std::string str_) {
            return c10::make_intrusive<ConstantString>(std::move(str_));
        }

        TORCH_API c10::intrusive_ptr<ConstantString> ConstantString::create(
            c10::string_view str_) {
            return c10::make_intrusive<ConstantString>(std::string(str_));
        }

        TORCH_API c10::intrusive_ptr<ConstantString> ConstantString::create(
            const char* str_) {
            return c10::make_intrusive<ConstantString>(std::string(str_));
        }

        // 比较两个 ivalue::Tuple 对象是否相等的操作符重载函数
        bool operator==(const ivalue::Tuple& lhs, const ivalue::Tuple& rhs) {
            // 判断元组大小是否相等，并逐个比较元素
            return lhs.size() == rhs.size() &&
                std::equal(
                    lhs.elements().cbegin(),
                    lhs.elements().cend(),
                    rhs.elements().cbegin(),
                    _fastEqualsForContainer);
        }

        // 输出 ivalue::EnumHolder 对象的输出流操作符重载函数
        std::ostream& operator<<(std::ostream& out, const ivalue::EnumHolder& v) {
            // 输出枚举的完全限定类名和名称
            out << v.qualifiedClassName() << "." << v.name();
            return out;
        }

        // 比较两个 ivalue::EnumHolder 对象是否相等的操作符重载函数
        bool operator==(const ivalue::EnumHolder& lhs, const ivalue::EnumHolder& rhs) {
            // 判断枚举名称和类型是否相同
            return lhs.name() == rhs.name() && *rhs.type() == *lhs.type();
        }

        // 返回 ivalue::EnumHolder 对象的完全限定类名
        const std::string& ivalue::EnumHolder::qualifiedClassName() const {
            return type_->qualifiedClassName().qualifiedName();
        }

        // 返回 ivalue::EnumHolder 对象的非限定类名
        const std::string& ivalue::EnumHolder::unqualifiedClassName() const {
            return type_->qualifiedClassName().name();
        }

    } // namespace ivalue

} // namespace c10
c10::TypePtr IValue::TagType<c10::Type>::get(const IValue& v) {
  // 根据 IValue 的 tag 值返回相应的类型指针
  switch (v.tag) {
      case Tag::None:
        // 返回 None 类型的 TypePtr
        return NoneType::get();
      case Tag::Tensor:
        // 返回 Tensor 类型的 TypePtr，使用 v 中的 Tensor 创建
        return TensorType::create(v.toTensor());
      case Tag::Storage:
        // 返回 Storage 类型的 TypePtr
        return StorageType::get();
      case Tag::Double:
        // 返回 Float 类型的 TypePtr
        return FloatType::get();
      case Tag::ComplexDouble:
        // 返回 Complex 类型的 TypePtr
        return ComplexType::get();
      case Tag::Int:
        // 返回 Int 类型的 TypePtr
        return IntType::get();
      case Tag::SymInt:
        // 返回 SymInt 类型的 TypePtr
        return c10::SymIntType::get();
      case Tag::SymFloat:
        // 返回 SymFloat 类型的 TypePtr
        return c10::SymFloatType::get();
      case Tag::SymBool:
        // 返回 SymBool 类型的 TypePtr
        return c10::SymBoolType::get();
      case Tag::Bool:
        // 返回 Bool 类型的 TypePtr
        return BoolType::get();
      case Tag::String:
        // 返回 String 类型的 TypePtr
        return StringType::get();
      case Tag::Blob:
        // 返回 Any 类型的 TypePtr
        return AnyType::get();
      case Tag::GenericDict: {
        // 处理 GenericDict 类型，返回对应的 DictType::create
        auto d = v.toGenericDict();
        return DictType::create(d.keyType(), d.valueType());
      }
      case Tag::GenericList:
        // 处理 GenericList 类型，返回对应的 ListType::create
        return ListType::create(v.toList().elementType());
      case Tag::Await:
        // 处理 Await 类型，返回对应的 AwaitType::create
        return AwaitType::create(v.toAwait()->elementType());
      case Tag::Future:
        // 处理 Future 类型，返回对应的 FutureType::create
        return FutureType::create(v.toFuture()->elementType());
      case Tag::RRef:
        // 处理 RRef 类型，返回对应的 RRefType::create
        return RRefType::create(v.toRRef()->type());
      case Tag::Device:
        // 返回 DeviceObjType 类型的 TypePtr
        return DeviceObjType::get();
      case Tag::Stream:
        // 返回 StreamObjType 类型的 TypePtr
        return StreamObjType::get();
      case Tag::Object:
        // 返回存储在 IValue 对象中的 Object 类型的 TypePtr
        return v.toObjectRef().type();
      case Tag::PyObject:
        // 返回 PyObjectType 类型的 TypePtr
        return PyObjectType::get();
      case Tag::Uninitialized:
        // 返回 Any 类型的 TypePtr
        return AnyType::get();
      case Tag::Capsule:
        // 返回 CapsuleType 类型的 TypePtr
        return CapsuleType::get();
      case Tag::Tuple:
        // 处理 Tuple 类型，返回对应的 Tuple 类型的 TypePtr
        return v.toTupleRef().type();
      case Tag::Generator:
        // 返回 GeneratorType 类型的 TypePtr
        return GeneratorType::get();
      case Tag::Quantizer:
        // 返回 QuantizerType 类型的 TypePtr
        return QuantizerType::get();
      case Tag::Enum:
        // 返回枚举类型的 TypePtr
        return v.toEnumHolder()->type();
  }
  // 上面的 switch 已经涵盖了所有情况，这里是为了消除编译器警告
  TORCH_INTERNAL_ASSERT(false, "unhandled case in IValue::type");

  // 这个 static_assert 需要放在某个 IValue 成员函数中，这里选择了这个位置。
  // 它不在类体中，因为 ivalue.h 是一个高扇出的头文件，我们希望尽量减少构建时间。
  static_assert(
      kNumTags <= 32,
      "IValue::isIntrusivePtr 需要更新，因为它假设最多有 32 个标签");
}

void IValue::visit(const std::function<bool (const IValue &)>& visitor) const {
  // 如果 visitor 返回 true，则提前结束访问。
  if (visitor(*this)) {
    // 简化处理：提前返回
    return;
  }
  // 根据 IValue 的 tag 值进行访问
  switch (this->tag) {
    case Tag::Tuple:
    case Tag::GenericList: {
      c10::ArrayRef<IValue> elems;
      // 根据类型不同，获取元素数组的引用
      if (isTuple()) {
        elems = this->toTupleRef().elements();
      } else {
        elems = this->toListRef();
      }
      // 逐个访问元素
      for (auto& elem : elems) {
        elem.visit(visitor);
      }
      break;
    }
    // 对于标签为 GenericDict 的情况，遍历 GenericDict，并访问其中每个键值对的值和键
    case Tag::GenericDict:
      for (const auto& pair : this->toGenericDict()) {
        pair.value().visit(visitor);  // 访问键值对的值
        pair.key().visit(visitor);    // 访问键值对的键
      }
      break;
    // 对于标签为 Object 的情况
    case Tag::Object: {
      auto obj_type = type()->expect<ClassType>();  // 获取对象的类型
      auto obj_value = toObject();                  // 获取对象的值
      auto attributes = obj_type->getAttributes();  // 获取对象类型的所有属性
      // 遍历对象的属性列表，并访问每个属性的值
      for (const auto& attr: attributes) {
        auto attribute = obj_value->getAttr(attr.getName());  // 获取属性的值
        attribute.visit(visitor);                             // 访问属性的值
      }
      break;
    }
    // 对于标签为 PyObject 的情况
    case Tag::PyObject: {
      c10::intrusive_ptr<at::ivalue::PyObjectHolder> py_obj = toPyObjectHolder();  // 获取 PyObject 的持有者
      auto match = py_obj->tryToInferType();                                      // 尝试推断 PyObject 的类型
      if (match.success()) {
        auto contained_value = py_obj->toIValue(match.type());  // 转换 PyObject 到 IValue
        contained_value.visit(visitor);                         // 访问转换后的值
      }
      break;
    }
    default:
      break;
}

void IValue::getSubValues(HashAliasedIValues& subValues) const {
  // 根据当前 IValue 的标签类型进行分支处理
  switch (this->tag) {
    // 如果是 Tensor 类型，直接将当前对象插入到 subValues 中并返回
    case Tag::Tensor:
      subValues.insert(*this);
      return;
    // 如果是 Tuple 或者 GenericList 类型
    case Tag::Tuple:
    case Tag::GenericList: {
      // 将当前对象插入到 subValues 中
      subValues.insert(*this);
      c10::ArrayRef<IValue> elems;
      // 根据具体类型获取元素列表
      if (isTuple()) {
        elems = this->toTupleRef().elements();
      } else {
        elems = this->toListRef();
      }
      // 遍历元素列表，递归调用 getSubValues 获取子值
      for (auto& elem : elems) {
        elem.getSubValues(subValues);
      }
      break;
    }
    // 如果是 GenericDict 类型
    case Tag::GenericDict:
      // 将当前对象插入到 subValues 中
      subValues.insert(*this);
      // 遍历字典的键值对，递归调用 getSubValues 获取子值
      for (const auto& pair : this->toGenericDict()) {
        pair.value().getSubValues(subValues);
        pair.key().getSubValues(subValues);
      }
      break;
    // 如果是 Object 类型
    case Tag::Object: {
      // 记录对象类型和对象的属性
      subValues.insert(*this);
      auto obj_type = type()->expect<ClassType>();
      auto obj_value = toObject();
      auto attributes = obj_type->getAttributes();
      // 遍历对象的属性，递归调用 getSubValues 获取子值
      for (const auto& attr: attributes) {
        auto attribute = obj_value->getAttr(attr.getName());
        attribute.getSubValues(subValues);
      }
      break;
    }
    // 如果是 PyObject 类型
    case Tag::PyObject: {
      // 将当前对象插入到 subValues 中
      subValues.insert(*this);
      c10::intrusive_ptr<at::ivalue::PyObjectHolder> py_obj = toPyObjectHolder();
      auto match = py_obj->tryToInferType();
      TORCH_CHECK_TYPE(match.success(),
            "Cannot infer type of ", py_obj->toStr(), ": ", match.reason());
      auto contained_value = py_obj->toIValue(match.type());
      // 递归调用 getSubValues 获取子值
      contained_value.getSubValues(subValues);
      break;
    }
    // 对于其他类型，报错，无法处理
    case Tag::Future:
    case Tag::Await:
    case Tag::Device:
    case Tag::Uninitialized:
    case Tag::Capsule:
      TORCH_CHECK_TYPE(
          false, "Cannot inspect value of type ", this->tagKind());
    default:
      // 对于标量类型，不记录任何内容
      break;
  }
}

bool IValue::overlaps(const IValue& rhs) const {
  HashAliasedIValues rhsSubValues, thisSubValues;
  // 获取当前对象和 rhs 对象的所有子值
  rhs.getSubValues(rhsSubValues);
  getSubValues(thisSubValues);
  // 检查两者的子值集合是否有交集
  for (auto& sub : thisSubValues) {
    if (rhsSubValues.count(sub)) {
      return true;
    }
  }
  return false;
}

bool operator!=(const IValue& lhs, const IValue& rhs) {
  // 使用 == 运算符的相反结果作为 != 的结果
  return !(lhs == rhs);
}

bool operator==(const IValue& lhs, const IValue& rhs) {
  // 调用 equals 方法获取比较结果
  IValue eq = lhs.equals(rhs);
  if (eq.isBool()) {
    return eq.toBool();
  }
  // 对于 Tensor 类型的特殊处理
  TORCH_INTERNAL_ASSERT(eq.isTensor());
  return eq.toTensor().is_nonzero();
}

bool IValue::ptrEqual(const IValue& lhs, const IValue& rhs) {
  // 断言两个对象都是 IntrusivePtr 类型
  TORCH_INTERNAL_ASSERT(lhs.isIntrusivePtr());
  TORCH_INTERNAL_ASSERT(rhs.isIntrusivePtr());
  // 检查两个对象的指针是否相等
  return lhs.tag == rhs.tag &&
      lhs.payload.u.as_intrusive_ptr == rhs.payload.u.as_intrusive_ptr;
}

IValue IValue::equals(const IValue& rhs) const {
  const IValue& lhs = *this;
  // 使用当前对象的 tag 属性进行分支处理
  switch (lhs.tag) {
    // 其他类型使用默认的 equals 方法进行比较
    default:
      // 不应该到达这里，因为每种类型都应该有具体的 equals 实现
      TORCH_INTERNAL_ASSERT(false, "equals not implemented for this type");
  }
}
    case Tag::None:
      // 如果标签是 None，比较 rhs 是否也是 None
      // Python 中不推荐这样的比较，可能需要警告
      return rhs.isNone();
    case Tag::Tensor: {
      // 如果标签是 Tensor，确保 rhs 也是 Tensor，并比较它们的值
      if (!rhs.isTensor()) {
        return false;
      }
      return lhs.toTensor().eq(rhs.toTensor());
    }
    case Tag::Storage:
      // 如果标签是 Storage，比较 lhs 和 rhs 是否指向相同的存储实现
      return rhs.isStorage() && lhs.toStorage().unsafeGetStorageImpl() == rhs.toStorage().unsafeGetStorageImpl();
    case Tag::Double:
      // 如果标签是 Double，比较 lhs 和 rhs 的双精度浮点值
      return rhs.isDouble() && lhs.toDouble() == rhs.toDouble();
    case Tag::ComplexDouble:
      // 如果标签是 ComplexDouble，比较 lhs 和 rhs 的复数双精度值
      return rhs.isComplexDouble() && lhs.toComplexDouble() == rhs.toComplexDouble();
    case Tag::Int:
      // 如果标签是 Int，比较 lhs 和 rhs 的整数值
      return rhs.isInt() && lhs.toInt() == rhs.toInt();
    case Tag::SymInt:
      // 如果标签是 SymInt，比较 lhs 和 rhs 的符号整数值
      return rhs.isSymInt() && lhs.toSymInt() == rhs.toSymInt();
    case Tag::SymFloat:
      // 如果标签是 SymFloat，比较 lhs 和 rhs 的符号浮点值
      return rhs.isSymFloat() && lhs.toSymFloat() == rhs.toSymFloat();
    case Tag::SymBool:
      // 如果标签是 SymBool，比较 lhs 和 rhs 的符号布尔值
      return rhs.isSymBool() && lhs.toSymBool() == rhs.toSymBool();
    case Tag::Bool:
      // 如果标签是 Bool，比较 lhs 和 rhs 的布尔值
      return rhs.isBool() && lhs.toBool() == rhs.toBool();
    case Tag::String:
      // 如果标签是 String，比较 lhs 和 rhs 的字符串值
      return rhs.isString() && lhs.toStringRef() == rhs.toStringRef();
    case Tag::GenericDict:
      // 如果标签是 GenericDict，比较 lhs 和 rhs 的通用字典值
      return rhs.isGenericDict() && lhs.toGenericDict() == rhs.toGenericDict();
    case Tag::Tuple:
      // 如果标签是 Tuple，比较 lhs 和 rhs 的元组值
      return rhs.isTuple() && *lhs.toTuple() == *rhs.toTuple();
    case Tag::Stream:
      // 如果标签是 Stream，比较 lhs 和 rhs 的流对象
      return rhs.isStream() && lhs.toStream() == rhs.toStream();
    case Tag::Device:
      // 如果标签是 Device，比较 lhs 和 rhs 的设备对象
      return rhs.isDevice() && lhs.toDevice() == rhs.toDevice();
    case Tag::GenericList:
      // 如果标签是 GenericList，比较 lhs 和 rhs 的通用列表值
      return rhs.isList() && lhs.toList() == rhs.toList();
    case Tag::Blob:
    case Tag::Future:
    case Tag::Await:
    case Tag::RRef:
    case Tag::Object:
    case Tag::PyObject:
    case Tag::Capsule:
    case Tag::Generator:
    case Tag::Quantizer:
      // 对于这些标签，直接使用指针比较函数 ptrEqual 比较 lhs 和 rhs
      return ptrEqual(lhs, rhs);
    case Tag::Enum:
      // 如果标签是 Enum，调用枚举持有者的方法判断是否相等
      return lhs.toEnumHolder()->is(*rhs.toEnumHolder());
    case Tag::Uninitialized:
      // 如果标签是 Uninitialized，始终返回 false，表示未初始化的值不相等
      // 这种情况通常在编译器能够证明值永远不会被使用时出现
      return false;
  }
  // 上面的 switch 应该覆盖所有情况，如果到达此处则抛出内部断言错误
  TORCH_INTERNAL_ASSERT(false, "we should never reach here")
}

size_t IValue::hash(const IValue& v) {
  switch (v.tag) {
    case Tag::None:
      return 0;  // 如果值的标签为 None，则返回哈希值 0
    case Tag::Bool:
      return c10::get_hash(v.payload.u.as_bool);  // 对布尔值进行哈希处理
    case Tag::Double:
      return c10::get_hash(v.payload.u.as_double);  // 对双精度浮点数进行哈希处理
    case Tag::Tensor:
      // Tensor 的哈希等同于其指针值，因此取得张量的指针以模拟哈希值
      return c10::get_hash(v.payload.as_tensor.unsafeGetTensorImpl());
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case Tag::Storage:
      return c10::get_hash(v.payload.u.as_int);  // 对存储对象进行哈希处理
    case Tag::Int:
      return c10::get_hash(v.payload.u.as_int);  // 对整数进行哈希处理
    // NB: these are technically strict aliasing violations
    case Tag::SymInt:
      return c10::get_hash(v.payload.u.as_int);  // 对符号整数进行哈希处理
    case Tag::SymFloat:
      return c10::get_hash(v.payload.u.as_int);  // 对符号浮点数进行哈希处理
    case Tag::SymBool:
      return c10::get_hash(v.payload.u.as_int);  // 对符号布尔值进行哈希处理
    case Tag::String:
      return c10::get_hash(v.toStringRef());  // 对字符串进行哈希处理
    case Tag::Tuple:
      return c10::get_hash(*v.toTuple());  // 对元组进行哈希处理
    case Tag::Device:
      return c10::get_hash(v.toDevice());  // 对设备对象进行哈希处理
    case Tag::GenericDict:
    case Tag::GenericList:
    case Tag::Blob:
    case Tag::Future:
    case Tag::Await:
    case Tag::RRef:
    case Tag::Object:
    case Tag::PyObject:
    case Tag::Capsule:
    case Tag::Generator:
    case Tag::Quantizer:
    case Tag::ComplexDouble:
    case Tag::Enum:
    case Tag::Stream:
    case Tag::Uninitialized:
      throw std::runtime_error(
          "unhashable type: '" + v.type()->repr_str() + "'");  // 抛出异常，表示不可哈希的类型
  }
  // the above switch should be exhaustive
  TORCH_INTERNAL_ASSERT(false, "we should never reach here")  // 断言，不应该执行到这里
}

static bool isUndefinedTensor(const IValue& iv) {
  return iv.isTensor() && !iv.toTensor().defined();  // 检查是否为未定义的张量
}

bool IValue::is(const IValue& rhs) const {
  const IValue& lhs = *this;
  // Special handling for undefined tensors:
  // 1. Undefined_tensor is None and vice versa.
  if ((isUndefinedTensor(lhs) && rhs.isNone()) ||
      (lhs.isNone() && isUndefinedTensor(rhs))) {
    return true;  // 处理未定义的张量与 None 相等的情况
  }
  // 2. Undefined_tensor is Undefined_tensor.
  if (isUndefinedTensor(lhs) && isUndefinedTensor(rhs)) {
    return true;  // 处理两个未定义的张量相等的情况
  }

  if (lhs.isTensor()) {
    // Use the standard way of comparing two tensors for identity
    return rhs.isTensor() && lhs.toTensor().is_same(rhs.toTensor());  // 比较两个张量是否相同
  }

  if (lhs.isIntrusivePtr()) {
    return rhs.isIntrusivePtr() && ptrEqual(lhs, rhs);  // 比较两个内部指针是否相同
  }
  return lhs == rhs;  // 使用默认的相等性比较
}

template <typename T>
inline bool IValue::isListOf() const {
  // note: avoids calling type() to avoid extra referencing counting for the returned type.
  if (!isList()) {
    return false;  // 如果不是列表类型，则返回 false
  }
  const auto& ty = static_cast<detail::ListImpl*>(payload.u.as_intrusive_ptr)->elementType;
  if (ty->kind() == T::Kind) {
    return true;  // 如果列表元素类型匹配，则返回 true
  }
  return *ty == *TypeFactory::get<T>();  // 比较列表元素类型是否与指定类型相同
}

bool IValue::isDoubleList() const {
  return isListOf<c10::FloatType>();  // 判断是否是浮点数列表
}

bool IValue::isComplexDoubleList() const {
  return isListOf<c10::ComplexType>();  // 判断是否是复数浮点数列表
}
// 检查当前 IValue 是否为 Tensor 类型的列表
bool IValue::isTensorList() const {
  return isListOf<c10::TensorType>();
}

// 检查当前 IValue 是否为可选的 Tensor 类型的列表
bool IValue::isOptionalTensorList() const {
  if (!isList()) {  // 如果当前 IValue 不是列表类型，则返回 false
    return false;
  }
  // 获取列表的元素类型
  const auto& ty = static_cast<detail::ListImpl*>(payload.u.as_intrusive_ptr)->elementType;
  // 获取预期的元素类型为 optional<at::Tensor>
  const auto& expected_ty = c10::getTypePtr<std::optional<at::Tensor>>();
  return expected_ty == ty;  // 返回列表的元素类型是否符合预期
}

// 检查当前 IValue 是否为 Int 类型的列表
bool IValue::isIntList() const {
  return isListOf<c10::IntType>();
}

// 检查当前 IValue 是否为 SymInt 类型的列表
bool IValue::isSymIntList() const {
  return isListOf<c10::SymIntType>();
}

// 检查当前 IValue 是否为 Bool 类型的列表
bool IValue::isBoolList() const {
  return isListOf<c10::BoolType>();
}

namespace {

using IValueFormatter = std::function<void(std::ostream&, const IValue&)>;

// 打印列表的通用函数，支持自定义起始和结束字符串以及格式化函数
template <class T>
std::ostream& printList(
    std::ostream& out,
    const T& list,
    const std::string& start,
    const std::string& finish,
    const IValueFormatter& formatter) {
  out << start;  // 输出起始字符串
  for (const auto i : c10::irange(list.size())) {  // 遍历列表
    if (i > 0) {
      out << ", ";  // 在元素之间添加逗号和空格
    }
    formatter(out, IValue(list[i]));  // 使用给定的格式化函数输出列表元素的值
  }
  out << finish;  // 输出结束字符串
  return out;
}

// 打印可能带注释的列表，根据列表的元素类型判断是否需要注释
std::ostream& printMaybeAnnotatedList(
    std::ostream& out,
    const IValue& the_list,
    const IValueFormatter& formatter) {
  auto list_elem_type = the_list.type()->containedType(0);  // 获取列表的元素类型
  if (the_list.toListRef().empty() ||  // 如果列表为空
      !elementTypeCanBeInferredFromMembers(list_elem_type)) {  // 或无法从成员推断出元素类型
    out << "annotate(" << the_list.type<c10::Type>()->annotation_str() << ", ";  // 添加注释
    printList(out, the_list.toListRef(), "[", "]", formatter);  // 打印带注释的列表
    out << ")";  // 输出注释结束
    return out;
  } else {
    return printList(out, the_list.toListRef(), "[", "]", formatter);  // 直接打印列表
  }
}

// 打印字典的通用函数，支持自定义格式化函数
template <typename Dict>
std::ostream& printDict(
    std::ostream& out,
    const Dict& v,
    const IValueFormatter& formatter) {
  out << "{";  // 输出字典的起始符号

  bool first = true;
  for (const auto& pair : v) {
    if (!first) {
      out << ", ";  // 在每对键值对之间添加逗号和空格
    }

    formatter(out, pair.key());  // 格式化输出键
    out << ": ";
    formatter(out, pair.value());  // 格式化输出值
    first = false;
  }

  out << "}";  // 输出字典的结束符号
  return out;
}
}

// 打印可能带注释的字典，根据字典值的类型判断是否需要注释
static std::ostream& printMaybeAnnotatedDict(
    std::ostream& out,
    const IValue& the_dict,
    const IValueFormatter& formatter) {
  auto value_type = the_dict.type()->castRaw<DictType>()->getValueType();  // 获取字典值的类型
  if (the_dict.toGenericDict().empty() ||  // 如果字典为空
      !elementTypeCanBeInferredFromMembers(value_type)) {  // 或无法从成员推断出值的类型
    out << "annotate(" << the_dict.type<c10::Type>()->annotation_str() << ",";  // 添加注释
    printDict(out, the_dict.toGenericDict(), formatter) << ")";  // 打印带注释的字典
  } else {
    return printDict(out, the_dict.toGenericDict(), formatter);  // 直接打印字典
  }
  return out;
}

// 打印复数值
static std::ostream& printComplex(std::ostream & out, const IValue & v) {
  c10::complex<double> d = v.toComplexDouble();  // 将 IValue 转换为复数值
  IValue real(d.real()), imag(std::abs(d.imag()));  // 获取实部和绝对值的虚部
  auto sign = "";
  if (d.imag() >= 0) {
    sign = "+";  // 设置正号
  } else {
    sign = "-";  // 设置负号
  }
  return out << real << sign << imag << "j";  // 输出复数值的字符串表示
}

// 打印 IValue 对象的字符串表示形式
std::ostream& IValue::repr(
    std::ostream& out,
    std::function<bool(std::ostream&, const IValue& v)>
        customFormatter) const {


// 定义一个成员函数，接受一个自定义格式化函数作为参数，并返回一个布尔值
// 这个函数用于将对象的字符串表示输出到给定的输出流中
// customFormatter 是一个函数对象，用于自定义格式化输出
// 返回值是一个 std::function 对象，可以接受一个输出流和一个 IValue 对象作为参数
// const 保证这个函数不会修改当前对象的状态



  // First check if the caller has provided a custom formatter. Use that if possible.
  if (customFormatter(out, *this)) {
    return out;
  }


  // 首先检查调用者是否提供了自定义的格式化函数，如果有则使用它
  // 调用 customFormatter 函数对象，传入当前对象和输出流 out 作为参数
  // 如果 customFormatter 返回 true，则直接返回输出流 out，表示已经完成格式化输出



  const IValue& v = *this;


  // 将当前对象 *this 赋值给常量引用 v
  // 这样可以在后续代码中方便地引用当前对象的值



  // continue to use custom formatter in recursion
  auto formatter = [&](std::ostream& out, const IValue& input) {
    input.repr(out, customFormatter);
  };


  // 定义一个 lambda 函数 formatter，用于递归调用自定义格式化函数 customFormatter
  // 这个 lambda 函数接受一个输出流 out 和一个 IValue 对象 input 作为参数
  // 调用 input 对象的 repr 方法，并传入输出流 out 和自定义格式化函数 customFormatter



  switch (v.tag) {


  // 根据当前对象 v 的标签（tag）进行不同的处理分支
  // 标签是一个枚举类型，用于表示当前对象的具体类型或类别



    case IValue::Tag::None:
      return out << v.toNone();


    // 如果当前对象的标签是 None 类型
    // 使用输出流 out 输出当前对象的 None 类型的值
    // v.toNone() 返回 None 类型的具体值



    case IValue::Tag::Double: {
      double d = v.toDouble();
      int c = std::fpclassify(d);
      if ((c == FP_NORMAL || c == FP_ZERO ) && std::abs(d) < 1e10) {
        int64_t i = int64_t(d);
        if (double(i) == d) {
          // -0.0 (signed zero) needs to be parsed as -0.
          if (i == 0 && std::signbit(d)) {
            return out << "-" << i << ".";
          }
          return out << i << ".";
        }
      }
      auto orig_prec = out.precision();
      return out << std::setprecision(std::numeric_limits<double>::max_digits10)
                 << d << std::setprecision(static_cast<int>(orig_prec));
    }


    // 如果当前对象的标签是 Double 类型
    // 从当前对象中获取 Double 类型的具体值 d
    // 根据 d 的特性进行不同的输出处理：
    // - 如果 d 是有限浮点数且绝对值小于 1e10，则尝试输出整数形式（如果是整数的话）
    // - 否则，按照浮点数的最大精度输出 d



    case IValue::Tag::ComplexDouble: {
      return printComplex(out, v);
    }


    // 如果当前对象的标签是 ComplexDouble 类型
    // 调用 printComplex 函数，将当前对象 v 输出为复数的字符串表示形式



    case IValue::Tag::Int:
      return out << v.toInt();


    // 如果当前对象的标签是 Int 类型
    // 使用输出流 out 输出当前对象的 Int 类型的值
    // v.toInt() 返回 Int 类型的具体值



    case IValue::Tag::SymInt:
      return out << v.toSymInt();


    // 如果当前对象的标签是 SymInt 类型
    // 使用输出流 out 输出当前对象的 SymInt 类型的值
    // v.toSymInt() 返回 SymInt 类型的具体值



    case IValue::Tag::SymFloat:
      return out << v.toSymFloat();


    // 如果当前对象的标签是 SymFloat 类型
    // 使用输出流 out 输出当前对象的 SymFloat 类型的值
    // v.toSymFloat() 返回 SymFloat 类型的具体值



    case IValue::Tag::SymBool:
      return out << v.toSymBool();


    // 如果当前对象的标签是 SymBool 类型
    // 使用输出流 out 输出当前对象的 SymBool 类型的值
    // v.toSymBool() 返回 SymBool 类型的具体值



    case IValue::Tag::Bool:
      return out << (v.toBool() ? "True" : "False");


    // 如果当前对象的标签是 Bool 类型
    // 使用输出流 out 输出当前对象的 Bool 类型的值
    // v.toBool() 返回 Bool 类型的具体值，输出 "True" 或者 "False"



    case IValue::Tag::Tuple: {
      const auto& elements = v.toTupleRef().elements();
      const auto& finish = elements.size() == 1 ? ",)" : ")";
      return printList(out, elements, "(", finish, formatter);
    }


    // 如果当前对象的标签是 Tuple 类型
    // 获取当前对象 v 的元组引用，并获取其元素列表 elements
    // 根据元素个数选择不同的结束标记 finish
    // 调用 printList 函数，将元组元素输出为字符串列表形式，使用自定义的 formatter 函数



    case IValue::Tag::String:
      c10::printQuotedString(out, v.toStringRef());
      return out;


    // 如果当前对象的标签是 String 类型
    // 使用 c10::printQuotedString 函数，将当前对象 v 的字符串引用输出为带引号的字符串形式
    // 然后将结果输出到输出流 out 中



    case IValue::Tag::GenericList: {
      return printMaybeAnnotatedList(out, *this, formatter);
    }


    // 如果当前对象的标签是 GenericList 类型
    // 调用 printMaybeAnnotatedList 函数，将当前对象 *this 输出为可能带注释的列表形式
    // 使用自定义的 formatter 函数来格式化列表中的元素



    case IValue::Tag::Device: {
      std::stringstream device_stream;
      device_stream << v.toDevice();
      out << "torch.device(";
      c10::printQuotedString(out, device_stream.str());
      return out << ")";
    }


    // 如果当前对象的标签是 Device 类型
    // 获取当前对象 v 的设备表示，并将其转换为字符串流 device_stream
    // 使用输出流 out 输出设备类型的字符串表示，格式为 "torch.device('device_string')"



    case IValue::Tag::Generator: {
      auto generator = v.toGenerator();
      out << "torch.Generator(device=";
      c10::printQuotedString(out, generator.device().str());
      out << ", seed=" << generator.current_seed() << ")";
      return out;
    }


    // 如果当前对象的标签是 Generator 类型
    // 获取当前对象 v 的生成器对象 generator
    // 使用输出流 out 输出生成器的字符串表示形式，格式为 "torch.Generator(device='device_string', seed=seed_value)"



    case IValue::Tag::GenericDict:
      return printMaybeAnnotatedDict(out, v, formatter);


    // 如果当前对象的标签是 GenericDict 类型
    // 调用 printMaybeAnnotatedDict 函数，将当前对象 v 输出为可能带注释的字典形式
    // 使用自定义的 formatter 函数来格式化字典中的键值对



    case IValue::Tag::Enum: {
      auto enum_holder = v.toEnumHolder();
      return out << enum_holder->qualifiedClassName() <<
}

// 检查简单的类类型参数是否与给定的类类型匹配，同时不是关键字参数且没有默认值
static bool simpleClassTypeArg(const Argument& arg, const ClassTypePtr& type) {
  return arg.type() == type && !arg.kwarg_only() && !arg.default_value();
}

// 检查对象类型是否定义了 "__lt__" 方法的模式，并返回该方法对象，如果不符合则返回 nullptr
torch::jit::Function* checkObjectSortSchema(const c10::ClassTypePtr& t, std::stringstream& why_not) {
  // 查找类类型 t 是否包含 "__lt__" 方法
  if (auto method = t->findMethod("__lt__")) {
      // 获取 "__lt__" 方法的 schema
      const auto& lt_schema = method->getSchema();
      // 获取方法的参数列表
      const auto& schema_args = lt_schema.arguments();
      // 检查方法的参数和返回类型是否符合预期
      bool error =
          (schema_args.size() != 2 ||
           !simpleClassTypeArg(schema_args[0], t) ||
           !simpleClassTypeArg(schema_args[1], t) ||
           lt_schema.returns().size() != 1 ||
           lt_schema.returns()[0].type() != BoolType::get());
      // 如果检查通过，则返回该方法对象
      if (!error) {
        return method;
      }
    }

    // 如果不符合要求，记录错误原因到 why_not，并返回 nullptr
    why_not << "To sort a list of " << t->repr_str()
            << " it must define a "
            << "__lt__ method with two inputs of type "
            << t->repr_str() << " that "
            << "returns a bool";
    return nullptr;
}

// 根据给定的 IValue 类型返回对应的比较器
IValueComparator getLessThanComparator(const IValue& v) {
  // 如果是 Tensor 类型，返回对应的比较器
  if (v.isTensor()) {
      return [](const IValue& a, const IValue& b) {
        return a.toTensor().lt(b.toTensor()).is_nonzero();
      };
  }

  // 如果是 Double 类型，返回对应的比较器
  if (v.isDouble()) {
      return [](const IValue& a, const IValue& b) {
        return a.toDouble() < b.toDouble();
      };
  }

  // 如果是 Int 类型，返回对应的比较器
  if (v.isInt()) {
      return [](const IValue& a, const IValue& b) {
        return a.toInt() < b.toInt();
      };
  }

  // 如果是 Bool 类型，返回对应的比较器
  if (v.isBool()) {
      return [](const IValue& a, const IValue& b) {
        return a.toBool() == false && b.toBool() == true;
      };
  }

  // 如果是 String 类型，返回对应的比较器
  if (v.isString()) {
      return [](const IValue& a, const IValue& b) {
       return a.toStringRef() < b.toStringRef();
      };
  }

  // 如果是 Tuple 类型，返回对应的比较器
  if (v.isTuple()) {
      const auto& elements = v.toTupleRef().elements();
      size_t n = elements.size();

      // 递归获取每个元素的比较器
      std::vector<IValueComparator> elements_lts;
      elements_lts.reserve(n);
      for (const auto i : c10::irange(n)) {
        elements_lts.push_back(getLessThanComparator(elements[i]));
      }

      // 返回一个比较两个 Tuple 的比较器
      return [elements_lts=std::move(elements_lts), n](const IValue& a, const IValue& b) {
        const auto& a_elements = a.toTupleRef().elements();
        const auto& b_elements = b.toTupleRef().elements();

        for (const auto i : c10::irange(n)) {
          if (elements_lts[i](a_elements[i], b_elements[i])) {
            return true;
          }
          if (a_elements[i] == b_elements[i]) {
            continue;
          }
          return false;
        }
        // 如果两个 Tuple 完全相等，则返回 false
        return false;
      };
  }

  // 如果是 Object 类型，检查其排序模式是否符合要求，并返回相应的比较器
  if (v.isObject()) {
    std::stringstream why_not;
    torch::jit::Function* lt_func =
        checkObjectSortSchema(v.type()->expect<ClassType>(), why_not);
    // 如果未找到合适的排序方法，抛出错误
    if (!lt_func) {
      AT_ERROR(why_not.str());
    }
    // 返回一个 lambda 函数，该函数接受两个常量引用参数 a 和 b，类型为 IValue
    return [lt_func](const IValue& a, const IValue& b) {
      // 快速检查，满足“严格弱排序”要求
      if (a.is(b)) {
        // 如果 a 和 b 是相同的对象，则返回 false
        return false;
      }
      // 创建一个 JIT 栈对象用于排序
      torch::jit::Stack sort_stack;
      // 将 a 和 b 添加到 JIT 栈中
      sort_stack.push_back(a);
      sort_stack.push_back(b);
      // 运行 lt_func，即比较函数，对 JIT 栈进行排序操作
      lt_func->run(sort_stack);
      // 从 JIT 栈中弹出结果并转换为布尔类型返回
      return torch::jit::pop(sort_stack).toBool();
    };
  }

  // 抛出错误，指明不可比较的 IValue 类型
  AT_ERROR("IValues of type: ", v.tagKind(), " are not comparable");
}

// 返回一个大于给定值的比较器函数
IValueComparator getGreaterThanComparator(const IValue& v) {
  // 调用获取小于给定值的比较器函数
  auto lt = getLessThanComparator(v);
  // 返回一个 lambda 函数，该函数使用小于比较器实现大于比较
  return [lt = std::move(lt)](const IValue& a, const IValue& b) {
    return lt(b, a);  // gt(a, b) === lt(b, a)
  };
}

// 重载流输出运算符，用于将 IValue 对象输出到 ostream
std::ostream& operator<<(std::ostream & out, const IValue & v) {
  // 定义格式化函数对象 formatter
  auto formatter = [&](std::ostream& out, const IValue& v) {
    out << v;
  };
  // 根据 IValue 的标签执行不同的输出操作
  switch(v.tag) {
    case IValue::Tag::None:
      return out << v.toNone();
    case IValue::Tag::Tensor:
      return out << v.toTensor();
    case IValue::Tag::Storage:
      return out << v.toStorage().unsafeGetStorageImpl();
    case IValue::Tag::Double: {
      double d = v.toDouble();
      int c = std::fpclassify(d);
      if (c == FP_NORMAL || c == FP_ZERO) {
        int64_t i = int64_t(d);
        if (double(i) == d) {
          return out << i << ".";
        }
      }
      auto orig_prec = out.precision();
      return out
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << v.toDouble()
        << std::setprecision(static_cast<int>(orig_prec));
    } case IValue::Tag::ComplexDouble: {
      return printComplex(out, v);
    } case IValue::Tag::Int:
      return out << v.toInt();
    case IValue::Tag::SymInt:
      return out << v.toSymInt();
    case IValue::Tag::SymFloat:
      return out << v.toSymFloat();
    case IValue::Tag::SymBool:
      return out << v.toSymBool();
    case IValue::Tag::Bool:
      return out << (v.toBool() ? "True" : "False");
    case IValue::Tag::Tuple: {
      // 获取元组的元素，并打印为列表形式
      const auto& elements = v.toTupleRef().elements();
      const auto& finish = elements.size() == 1 ? ",)" : ")";
      return printList(out, elements, "(", finish, formatter);
    }
    case IValue::Tag::String:
      return out << v.toStringRef();
    case IValue::Tag::Blob:
      return out << *v.toBlob();
    case IValue::Tag::Capsule:
      return out << "Capsule";
    case IValue::Tag::GenericList:
      // 打印通用列表
      return printList(out, v.toList(), "[", "]", formatter);
    case IValue::Tag::RRef:
      return out << "RRef";
    case IValue::Tag::Future:
      return out << "Future";
    case IValue::Tag::Await:
      return out << "Await";
    case IValue::Tag::Uninitialized:
      return out << "Uninitialized";
    case IValue::Tag::Device:
      return out << v.toDevice();
    case IValue::Tag::Stream:
      return out << v.toStream();
    case IValue::Tag::GenericDict:
      // 打印通用字典
      return printDict(out, v.toGenericDict(), formatter);
    case IValue::Tag::PyObject: {
      auto py_obj = v.toPyObject();
      return out << "<PyObject at" << py_obj << ">";
    }
    case IValue::Tag::Generator:
      return out << "Generator";
    case IValue::Tag::Quantizer:
      return out << "Quantizer";
    case IValue::Tag::Object: {
      // 如果对象定义了 __str__ 方法，调用该方法输出
      auto obj = v.toObject();
      return out << "<" << obj->name() << " object at " << obj.get() << ">";
    }
    case IValue::Tag::Enum: {
      // 如果值是枚举类型，则获取枚举的持有者对象
      auto enum_holder = v.toEnumHolder();
      // 输出枚举类型的信息，包括其未限定类名和枚举名称
      return out << "Enum<" << enum_holder->unqualifiedClassName() << "." <<
          enum_holder->name() << ">";
    }

  }
  // 如果值的标签无效，则输出无效标签的信息
  return out << "<Invalid IValue tag=" << std::to_string(static_cast<uint32_t>(v.tag)) << ">";
}

#undef TORCH_FORALL_TAGS



// 取消定义宏 TORCH_FORALL_TAGS

void IValue::dump() const {
  // 打印当前对象到标准输出流
  std::cout << *this << "\n";
}

std::shared_ptr<ClassType> ivalue::Object::type() const {
  // 返回对象的类型，期望结果是 ClassType 的智能指针
  return type_.type_->expect<ClassType>();
}

c10::intrusive_ptr<ivalue::Object> ivalue::Object::create(
    ClassTypePtr classType, size_t numSlots) {
  // 创建一个 ivalue::Object 的实例，使用给定的 classType 和槽位数量
  return ivalue::Object::create(
      StrongTypePtr(nullptr, std::move(classType)), numSlots);
}

IValue IValue::deepcopy(std::optional<at::Device> device) const {
  // 创建一个深拷贝对象，初始化 memo 用于跟踪已复制的对象
  IValue::HashIdentityIValueMap memo;
  return deepcopy(memo, device);
}

IValue IValue::deepcopy(
    IValue::HashIdentityIValueMap& memo,
    std::optional<at::Device> device) const {
  // 如果 memo 中已存在当前对象，则直接返回复制的对象
  if (memo.count(*this)) {
    return memo.at(*this);
  }
  IValue copy;
  // 根据对象的标签类型执行不同的深拷贝操作
  switch(tag) {
    case IValue::Tag::Tensor: {
      // 如果是 Tensor 类型，根据设备选项进行深拷贝
      const at::Tensor& src_tensor = toTensor();
      copy = device.has_value() && !src_tensor.device().is_meta()
          ? IValue(src_tensor.to(*device))
          : IValue(src_tensor.clone());
    } break;
    case IValue::Tag::Tuple: {
      // 如果是 Tuple 类型，逐个深拷贝其中的元素
      std::vector<IValue> copied_tuple;
      for (const auto& e : toTupleRef().elements()) {
        copied_tuple.emplace_back(e.deepcopy(memo, device));
      }
      copy = IValue(ivalue::Tuple::create(std::move(copied_tuple)));
    }
      break;
    case IValue::Tag::GenericList: {
      // 如果是 GenericList 类型，深拷贝其中的元素到新的列表
      auto list = toList();
      auto copied_list = c10::impl::GenericList(list.elementType());
      for (IValue v : list) {
        copied_list.push_back(v.deepcopy(memo, device));
      }
      copy = IValue(copied_list);
    }
      break;
    case IValue::Tag::GenericDict: {
      // 如果是 GenericDict 类型，深拷贝其中的键值对到新的字典
      auto dict = toGenericDict();
      auto copied_dict = c10::impl::GenericDict(dict.keyType(), dict.valueType());
      for (const auto& entry : dict) {
        copied_dict.insert(
            entry.key().deepcopy(memo, device),
            entry.value().deepcopy(memo, device));
      }
      copy = IValue(copied_dict);
    }
      break;
    case IValue::Tag::Object: {
      // 如果是 Object 类型，根据类是否有指定方法进行状态拷贝或整体拷贝
      auto class_type = type()->expect<ClassType>();
      if (class_type->hasMethod("__getstate__") &&
          class_type->hasMethod("__setstate__")) {
        copy = ivalue::Object::create(
            c10::StrongTypePtr(class_type->compilation_unit(), type()),
            class_type->numAttributes());
        auto state = class_type->getMethod("__getstate__")({*this});
        class_type->getMethod("__setstate__")({copy, std::move(state)});
      } else {
        copy = IValue(toObject()->deepcopy(memo, device));
      }
    } break;
    case IValue::Tag::Enum: {
      // 如果是 Enum 类型，深拷贝其包含的数据和名称
      auto enum_holder = toEnumHolder();
      copy = IValue(c10::make_intrusive<ivalue::EnumHolder>(
          enum_holder->type(),
          enum_holder->name(),
          enum_holder->value().deepcopy(memo, device)));
    } break;
    case IValue::Tag::String:
    case IValue::Tag::None:
    case IValue::Tag::Double:
    case IValue::Tag::Int:
    case IValue::Tag::SymInt:
    case IValue::Tag::SymFloat:
    case IValue::Tag::SymBool:
      // 对于基本类型和符号类型，直接复制
      copy = *this;
      break;
  }
  return copy;
}


这样的注释可以帮助阅读者理解每个函数和代码块的具体功能和作用。
    // 处理不同的 IValue 类型标签，进行深拷贝或报错处理
    case IValue::Tag::Bool:
    case IValue::Tag::Device:
    case IValue::Tag::Generator:
    case IValue::Tag::Uninitialized: {
      // 对于 Bool、Device、Generator 和 Uninitialized 类型的 IValue，执行浅拷贝
      copy = *this;
    } break;
    default: {
      // 如果遇到未知的 IValue 类型标签，抛出错误并显示标签类型
      AT_ERROR("Can't deepcopy IValue with tag: ", tagKind());
    }
  }
  // 注意：如果对象包含对自身的引用，则以下代码不起作用；
  // 当我们扩展对象系统并遇到此类问题时，我们将提交后续 PR 来解决。
  // 将当前对象及其拷贝加入 memo 字典，以避免循环引用问题
  if (!isAliasOf(copy)) {
    memo[*this] = copy;
  }
  // 返回深拷贝后的对象副本
  return copy;
}

void IValue::reportToTensorTypeError() const {
  // 报告张量类型错误的方法
  TORCH_CHECK(false, "Expected Tensor but got ", tagKind());
}

std::string ivalue::Object::name() const {
  // 返回对象类型的限定名称
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return type()->name()->qualifiedName();
}

IValue ivalue::Object::getAttr(const std::string& name) const {
  // 获取对象的属性值
  const size_t slot = type()->getAttributeSlot(name);
  return getSlot(slot);
}

void ivalue::Object::setAttr(const std::string& name, IValue v) {
  // 设置对象的属性值
  const size_t slot = type()->getAttributeSlot(name);
  setSlot(slot, std::move(v));
}

void ivalue::Object::unsafeRemoveAttr(const std::string& name) {
  // 不安全地移除对象的属性
  const size_t slot = type()->getAttributeSlot(name);
  unsafeRemoveSlot(slot);
}

void ivalue::Object::resizeObject(size_t slot) {
  // 调整对象的大小
  AT_ASSERT(slot < type()->numAttributes());
  slots_.resize(type()->numAttributes());
}


c10::intrusive_ptr<ivalue::Object> ivalue::Object::copy() const {
  // 复制对象
  auto object = ivalue::Object::create(type_, type()->numAttributes());
  for (const auto i : c10::irange(slots_.size())) {
    object->setSlot(i, slots_[i]);
  }
  return object;
}

c10::intrusive_ptr<ivalue::Object> ivalue::Object::copy_to_weak_compilation_ref() const {
  // 将对象复制到弱编译引用
  auto object = ivalue::Object::create(
      WeakOrStrongTypePtr(type_.asWeakTypePtr()), type()->numAttributes());
  for (const auto i : c10::irange(slots_.size())) {
    object->setSlot(i, slots_[i]);
  }
  return object;
}

c10::intrusive_ptr<ivalue::Object> ivalue::Object::deepcopy(
    std::optional<at::Device> device) const {
  // 深度复制对象
  IValue::HashIdentityIValueMap memo;
  return deepcopy(memo, device);
}

c10::intrusive_ptr<ivalue::Object> ivalue::Object::deepcopy(
    IValue::HashIdentityIValueMap& memo,
    std::optional<at::Device> device) const {
  // 深度复制对象，支持设备选择
  auto cu = type_.cu_;
  auto object = ivalue::Object::create(WeakOrStrongTypePtr(type_.cu_, type_.type_), type()->numAttributes());
  for (const auto i : c10::irange(slots_.size())) {
    if (*slots_[i].type() == *c10::TypeFactory::get<CapsuleType>()) {
      // 如果到达这里，表示未通过__getstate__和__setstate__复制类
      // 并且存在Capsule属性，表明这是一个未定义序列化方法的自定义C++类
      std::stringstream err;
      err << "Cannot serialize custom bound C++ class";
      if (auto qualname = type()->name()) {
        err << " " << qualname->qualifiedName();
      }
      err << ". Please define serialization methods via def_pickle() for "
            "this class.";
      AT_ERROR(err.str());
    }
    object->setSlot(i, slots_[i].deepcopy(memo, device));
  }
  return object;
}

StrongTypePtr::StrongTypePtr(
    std::shared_ptr<torch::jit::CompilationUnit> cu,
    TypePtr type) : cu_(std::move(cu)), type_(std::move(type)) {
  // 强类型指针的构造函数
  TORCH_INTERNAL_ASSERT(type_);
}

WeakTypePtr::WeakTypePtr(
    std::weak_ptr<torch::jit::CompilationUnit> cu,
    TypePtr type) : cu_(std::move(cu)), type_(std::move(type)) {
  // 弱类型指针的构造函数
}
// 将当前对象转换为弱类型指针
WeakTypePtr WeakOrStrongTypePtr::asWeakTypePtr() const {
  // 如果不持有强引用，则使用当前 CompilationUnit 的弱引用构造 WeakTypePtr
  if (!holds_strong_ref()) {
    return WeakTypePtr(cu_.getWeakRefOrThrow(), type_);
  } else {
    // 否则，获取当前 CompilationUnit 的强引用，并移动构造 WeakTypePtr
    std::weak_ptr<torch::jit::CompilationUnit> weak_cu =
        cu_.getStrongRefOrThrow();
    return WeakTypePtr(std::move(weak_cu), type_);
  }
}

// 需要在此 .cpp 文件中以便访问 PyObjectHolder 的完整定义
std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> ivalue::Future::extractStorages(
    const at::IValue& value) {
  // 存储弱引用的 StorageImpl 对象的向量
  std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> weakStorageImpls;
  // 如果 value 是 PyObject 类型
  if (value.isPyObject()) {
    // 通过 PyObjectHolder 提取张量
    std::vector<at::Tensor> tensors =
        value.toPyObjectHolder()->extractTensors();
    size_t num_storages = 0;
    // 遍历提取的张量，计算总的 storage 数量
    for (const at::Tensor& tensor : tensors) {
      if (tensor.is_sparse()) {
        // 稀疏张量包含索引和值两个 storage，因此增加 2
        num_storages += 2;
      } else {
        // 密集/分步张量包含一个 storage
        num_storages += 1;
      }
    }
    // 预留空间以容纳提取的 storages
    weakStorageImpls.reserve(num_storages);
    // 再次遍历张量，提取 storage 并添加到 weakStorageImpls
    for (const at::Tensor& tensor : tensors) {
      if (tensor.is_sparse()) {
        // 对于稀疏张量，获取索引和值的 storage，并添加到 weakStorageImpls
        weakStorageImpls.emplace_back(tensor.indices().storage().getWeakStorageImpl());
        weakStorageImpls.emplace_back(tensor.values().storage().getWeakStorageImpl());
      } else {
        // 对于密集/分步张量，获取 storage 并添加到 weakStorageImpls
        weakStorageImpls.emplace_back(tensor.storage().getWeakStorageImpl());
      }
    }
  } else {
    // 如果 value 不是 PyObject 类型，获取其子值
    at::IValue::HashAliasedIValues sub_values;
    value.getSubValues(sub_values);
    // 遍历子值
    for (const at::IValue& sub_value : sub_values) {
      if (sub_value.isTensor()) {
        // 如果子值是张量，获取其 storage 并添加到 weakStorageImpls
        auto const & tens = sub_value.toTensor();
        if (tens.is_sparse()) {
          // 对于稀疏张量，获取索引和值的 storage，并添加到 weakStorageImpls
          auto coalesced = tens.coalesce();
          weakStorageImpls.emplace_back(coalesced.indices().storage().getWeakStorageImpl());
          weakStorageImpls.emplace_back(coalesced.values().storage().getWeakStorageImpl());
        } else {
          // 对于密集/分步张量，获取 storage 并添加到 weakStorageImpls
          weakStorageImpls.emplace_back(tens.storage().getWeakStorageImpl());
        }
      }
    }
  }
  // 返回包含所有提取 storage 弱引用的向量
  return weakStorageImpls;
}
    // 定义上下文结构体 Ctx，用于处理多个异步 Future 的情况
    struct Ctx {
      // Ctx 结构体的构造函数，接收异步 Future 列表作为参数
      explicit Ctx(const List<intrusive_ptr<ivalue::Future>>& srcs)
          // 初始化剩余任务数为异步 Future 列表的大小
          : remaining(srcs.size()),
            // 复制异步 Future 列表
            srcFutures(srcs),
            // 将异步 Future 转换为 IValue 类型
            asIvalue(srcFutures),
            // 创建目标 Future 对象，类型与 asIvalue 保持一致
            dstFuture(make_intrusive<ivalue::Future>(asIvalue.type())) {}
    
      // 原子变量，用于记录剩余待完成的任务数
      std::atomic<size_t> remaining{0};
      // 存储输入的异步 Future 列表
      List<intrusive_ptr<ivalue::Future>> srcFutures;
      // 将 srcFutures 转换后的 IValue
      IValue asIvalue;
      // 目标 Future，用于存储最终的结果
      intrusive_ptr<ivalue::Future> dstFuture;
    };
    
    // 创建共享指针 ctx，指向上下文结构体 Ctx 的实例，传入异步 Future 列表作为参数
    auto ctx = std::make_shared<Ctx>(srcs);
    
    // 如果输入的异步 Future 列表为空
    if (ctx->srcFutures.empty()) {
      // 直接将 dstFuture 标记为已完成，并设置其值为 asIvalue
      ctx->dstFuture->markCompleted(ctx->asIvalue);
    } else {
      // 遍历异步 Future 列表中的每一个 Future
      for (const auto i : c10::irange(ctx->srcFutures.size())) {
        // 定义一个 lambda 函数 func，处理每个异步 Future
        std::function<void(ivalue::Future&)> func = [ctx](ivalue::Future& fut) {
          // 如果异步 Future fut 发生错误
          if (fut.hasError()) {
            // 将目标 Future 的错误状态设置为 fut 的异常指针
            ctx->dstFuture->setErrorIfNeeded(fut.exception_ptr());
            return;  // 退出函数
          }
    
          // 减少剩余任务数，并且如果剩余任务数为 0 且目标 Future 未完成
          if (--ctx->remaining == 0 && !ctx->dstFuture->completed()) {
            // 将目标 Future 标记为已完成，并设置其值为 asIvalue
            ctx->dstFuture->markCompleted(ctx->asIvalue);
          }
        };
    
        // 为异步 Future 添加回调函数 func
        ctx->srcFutures.get(i)->addCallback(func);
      }
    }
    
    // 返回处理后的目标 Future 对象
    return ctx->dstFuture;
}

namespace {

// 格式化设备集合为字符串
std::string formatSetOfDevices(const std::vector<c10::Device>& devices) {
  // 创建一个字符串流
  std::ostringstream oss;
  // 将设备集合中的设备逐个写入字符串流，用逗号和空格分隔
  std::copy(
      devices.begin(),
      devices.end(),
      std::ostream_iterator<c10::Device>(oss, ", "));
  // 返回字符串流中的字符串表示
  return oss.str();
}

}

// 在 Torch 的 API 中，收集任意一个 Future 对象
TORCH_API intrusive_ptr<ivalue::Future> collectAny(
    const List<intrusive_ptr<ivalue::Future>>& srcs) {
  // 如果源列表为空，创建一个空 Future 并标记为完成状态，然后返回
  if (srcs.empty()) {
    auto res = make_intrusive<ivalue::Future>(NoneType::get());
    res->markCompleted();
    return res;
  }
  // 获取第一个 Future 对象的类型指针和设备列表
  const TypePtr& typePtr = srcs.get(0)->elementType();
  const std::vector<c10::Device>& devices = srcs.get(0)->devices();
  // 遍历源列表中的所有 Future 对象
  for (const auto i : c10::irange(srcs.size())) {
    // 如果当前 Future 对象已经完成，直接返回该 Future 对象
    if (srcs.get(i)->completed()) {
      return srcs.get(i);
    }
    // 检查所有 Future 对象是否具有相同的类型
    TORCH_CHECK_TYPE(
        i == 0 || (*typePtr == *srcs.get(i)->elementType()),
        "Expected all futures to have the same type, but found ", *typePtr,
        " in position 0 and ", *srcs.get(i)->elementType(), " in position ", i);
    // 检查所有 Future 对象是否具有相同的设备
    TORCH_CHECK_VALUE(
        i == 0 || (devices == srcs.get(i)->devices()),
        "Expected all futures to have the same devices, but found ",
        formatSetOfDevices(devices), " in position 0 and ",
        formatSetOfDevices(srcs.get(i)->devices()), " in position ", i);
  }
  // 定义上下文结构体，用于处理 Future 对象的回调和状态
  struct Ctx {
    explicit Ctx(
        const List<intrusive_ptr<ivalue::Future>>& srcs,
        TypePtr typePtr,
        std::vector<c10::Device> devices)
        : srcFutures(srcs),
          dstFuture(make_intrusive<ivalue::Future>(std::move(typePtr), std::move(devices))) {}
    std::atomic<bool> done{false}; // 原子布尔值，表示操作是否完成
    List<intrusive_ptr<ivalue::Future>> srcFutures; // 源 Future 对象列表
    intrusive_ptr<ivalue::Future> dstFuture; // 目标 Future 对象
  };
  // 创建上下文对象，用于管理回调函数和状态
  auto ctx = std::make_shared<Ctx>(srcs, typePtr, devices);
  // 定义回调函数，一旦某个 Future 对象完成，将触发该函数
  std::function<void(ivalue::Future&)> func = [ctx](ivalue::Future& src) {
    if (!ctx->done.exchange(true)) { // 如果操作尚未完成，则执行以下代码
      intrusive_ptr<ivalue::Future> dst = ctx->dstFuture; // 获取目标 Future 对象
      ctx->dstFuture.reset(); // 一旦 Future 对象被满足，清除引用
      ctx->srcFutures =
          List<intrusive_ptr<ivalue::Future>>(ctx->srcFutures.elementType()); // 重置源 Future 对象列表
      // 如果源 Future 对象出现错误，设置目标 Future 对象的错误状态
      if (src.hasError()) {
        dst->setError(src.exception_ptr());
      } else { // 否则，标记目标 Future 对象为完成状态，并传递值和存储
        dst->markCompleted(src.constValue(), src.storages());
      }
    }
  };
  // 为所有源 Future 对象添加回调函数
  for (const auto i : c10::irange(ctx->srcFutures.size())) {
    ctx->srcFutures.get(i)->addCallback(func);
  }
  // 返回目标 Future 对象
  return ctx->dstFuture;
}

} // namespace c10
```