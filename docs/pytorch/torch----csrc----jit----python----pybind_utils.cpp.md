# `.\pytorch\torch\csrc\jit\python\pybind_utils.cpp`

```py
// 引入 Torch 库中的一些模块和头文件
#include <torch/csrc/jit/ir/graph_utils.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_dict.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/python_list.h>
#include <torch/csrc/jit/python/utf8_decoding_ignore.h>

// 引入 ATen 库中的标量操作
#include <ATen/ScalarOps.h>

// 引入 C10 库中的 QSchem 类和 irange 函数
#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <torch/csrc/utils/python_arg_parser.h>

// 引入 C++ 标准库中的一些头文件
#include <limits>
#include <optional>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 定义线程局部变量，用于控制是否允许将数字作为张量处理
static thread_local bool allow_numbers_as_tensors = false;

// 控制是否允许将数字作为张量处理的 RAII 类
ToIValueAllowNumbersAsTensors::ToIValueAllowNumbersAsTensors(bool enable)
    : old_(allow_numbers_as_tensors) {
  allow_numbers_as_tensors = enable;
}

// 析构函数，恢复旧的允许状态
ToIValueAllowNumbersAsTensors::~ToIValueAllowNumbersAsTensors() {
  allow_numbers_as_tensors = old_;
}

// 清除注册在 PyBind 缓存中的已删除实例
void clear_registered_instances(void* ptr) {
  auto& registered_instances =
      pybind11::detail::get_internals().registered_instances;
  auto range = registered_instances.equal_range(ptr);
  for (auto it = range.first; it != range.second; ++it) {
    auto vh = it->second->get_value_and_holder();
    vh.set_instance_registered(false);
  }
  registered_instances.erase(ptr);
}

// 警告：此函数的前提条件是已经确认 SymIntList 只包含整数，如果是，则将其转换为 IValue 类型
// 在运行时不会检查此前提条件。
template <typename T>
IValue listToIValue(py::handle obj) {
  c10::List<T> rs;
  for (auto it = obj.begin(); it != obj.end(); it++) {
    auto elm = *it;
    rs.push_back(py::cast<T>(elm));
  }
  // 承诺已适当地转换了列表
  return c10::impl::toList<T>(rs);
}

// 将 Python 对象转换为 IValue 类型，考虑给定的类型和可选的大小限制
IValue toIValue(py::handle obj, const TypePtr& type, std::optional<int32_t> N) {
  switch (type->kind()) {
    case TypeKind::TensorType: {
      // 如果输入对象是 None，则返回一个未定义的 Tensor
      if (obj.ptr() == Py_None) {
        return autograd::Variable();
      }
      // 如果输入对象是 autograd::Variable 类型，则直接返回它
      if (THPVariable_Check(obj.ptr())) {
        auto var = py::cast<autograd::Variable>(obj);
        // 检查并确保变量不是命名张量
        guardAgainstNamedTensor<autograd::Variable>(var);
        return var;
      } else {
        // 如果不允许将数字作为 Tensor 处理，则抛出异常
        if (!allow_numbers_as_tensors) {
          throw py::cast_error(
              c10::str("Unable to cast ", py::str(obj), " to Tensor"));
        }
        bool save_symint = false;
        at::Scalar scalar;
        // 根据不同的 Python 对象类型转换为标量 Scalar
        if (PyBool_Check(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackBool(obj.ptr()));
        } else if (THPUtils_checkLong(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackLong(obj.ptr()));
        } else if (PyComplex_Check(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackComplexDouble(obj.ptr()));
        } else if (THPUtils_checkDouble(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackDouble(obj.ptr()));
        } else if (torch::is_symint(py::handle(obj))) {
          // 对于符号整数类型，保存标志并使用特定值构造 Scalar
          save_symint = true;
          scalar = at::Scalar(7777777);
        } else if (torch::is_symfloat(py::handle(obj))) {
          // 对于符号浮点数类型，保存标志并使用 NaN 值构造 Scalar
          save_symint = true;
          scalar = at::Scalar(std::numeric_limits<double>::quiet_NaN());
        } else if (torch::is_symbool(py::handle(obj))) {
          // 对于符号布尔类型，保存标志并使用 true 构造 Scalar
          save_symint = true;
          scalar = at::Scalar(true);
        } else {
          // 如果无法将对象转换为 Tensor，则抛出异常
          throw py::cast_error(
              c10::str("Unable to cast ", py::str(obj), " to Tensor"));
        }
        // 将 Scalar 转换为 Tensor
        at::Tensor tensor = at::scalar_to_tensor(scalar);
        // 标记该 Tensor 包装了一个数字
        tensor.unsafeGetTensorImpl()->set_wrapped_number(true);

        // 如果保存了符号整数标志，将原始 Python 对象作为属性附加到 Tensor 上
        if (save_symint) {
          auto py_tensor = py::cast(tensor);
          if (PyObject_SetAttrString(
                  py_tensor.ptr(), "_wrapped_number", obj.ptr()) < 0) {
            throw python_error();
          }
        }

        return tensor;
      }
    }
    case TypeKind::StorageType:
      // 如果输入对象是 Storage 类型，则直接返回对应的 C++ at::Storage 对象
      return py::cast<at::Storage>(obj);
    case TypeKind::FloatType:
      // 如果输入对象是符号浮点数类型，返回对应的 SymFloat 对象
      if (torch::is_symfloat(py::handle(obj))) {
        return py::cast<c10::SymFloat>(obj).guard_float(__FILE__, __LINE__);
      }
      // 如果输入对象是 autograd::Variable 类型，并且它的存储是 meta 类型，则抛出异常
      if (THPVariable_Check(obj.ptr())) {
        auto var = py::cast<autograd::Variable>(obj);
        // 注意：我们仔细检查存储是否是 meta 类型，因为这总是准确的，即使你有一个伪张量（这是我们试图检测的主要情况）
        if (var.storage().device_type() == c10::kMeta) {
          throw py::cast_error(
              "cannot extract float from tensor with meta storage");
        }
      }
      // 否则，将输入对象转换为双精度浮点数并返回
      return py::cast<double>(obj);
    case TypeKind::ComplexType: {
      // 如果输入对象是 Python 复数对象，则转换为 C++ 复数类型并返回
      auto c_obj = py::cast<std::complex<double>>(obj.ptr());
      return static_cast<c10::complex<double>>(c_obj);
    }
    case TypeKind::IntType:
      // 处理整数类型的情况
      // TODO: Properly fake this type (TODO: 适当伪造此类型)
      // 检查是否为 THPQScheme 对象，返回其 qscheme 属性的值
      if (THPQScheme_Check(obj.ptr())) {
        auto qscheme = reinterpret_cast<THPQScheme*>(obj.ptr());
        return static_cast<uint8_t>(qscheme->qscheme);
      }
      // 用于向后兼容性
      // 检查是否为 THPDtype 对象，返回其 scalar_type 属性的值
      if (THPDtype_Check(obj.ptr())) {
        auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
        return static_cast<int64_t>(dtype->scalar_type);
      }
      // 再次检查是否为 THPQScheme 对象，返回其 qscheme 属性的值
      if (THPQScheme_Check(obj.ptr())) {
        auto qscheme = reinterpret_cast<THPQScheme*>(obj.ptr());
        return static_cast<uint8_t>(qscheme->qscheme);
      }
      // 检查是否为 THPLayout 对象，返回其 layout 属性的值
      if (THPLayout_Check(obj.ptr())) {
        auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
        return static_cast<int8_t>(layout->layout);
      }
      // 检查是否为 THPMemoryFormat 对象，返回其 memory_format 属性的值
      if (THPMemoryFormat_Check(obj.ptr())) {
        auto memory_format = reinterpret_cast<THPMemoryFormat*>(obj.ptr());
        return static_cast<int8_t>(memory_format->memory_format);
      }
      // 检查是否为 torch 的符号整数类型，转换为 SymInt 类型
      if (torch::is_symint(py::handle(obj))) {
        return py::cast<c10::SymInt>(obj).guard_int(__FILE__, __LINE__);
      }
      // 检查是否为 THPVariable 对象，处理特殊情况
      if (THPVariable_Check(obj.ptr())) {
        auto var = py::cast<autograd::Variable>(obj);
        // 如果变量存储的设备类型是元设备，抛出类型转换错误
        if (var.storage().device_type() == c10::kMeta) {
          throw py::cast_error(
              "cannot extract int from tensor with meta storage");
        }
      }
      // 将 obj 强制转换为 int64_t 类型并返回
      return py::cast<int64_t>(obj);

    case TypeKind::LayoutType: {
      // 处理布局类型的情况
      // 检查是否为 THPLayout 对象，返回其 layout 属性的值
      if (THPLayout_Check(obj.ptr())) {
        auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
        return static_cast<int8_t>(layout->layout);
      }
      // 向后兼容性处理，将 obj 强制转换为 int64_t 类型并返回
      return py::cast<int64_t>(obj);
    }

    case TypeKind::ScalarTypeType: {
      // 处理标量类型的情况
      // 检查是否为 THPDtype 对象，返回其 scalar_type 属性的值
      if (THPDtype_Check(obj.ptr())) {
        auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
        return static_cast<int64_t>(dtype->scalar_type);
      }
      // 向后兼容性处理，将 obj 强制转换为 int64_t 类型并返回
      return py::cast<int64_t>(obj);
    }

    case TypeKind::MemoryFormatType: {
      // 处理内存格式类型的情况
      // 检查是否为 THPMemoryFormat 对象，返回其 memory_format 属性的值
      if (THPMemoryFormat_Check(obj.ptr())) {
        auto memory_format = reinterpret_cast<THPMemoryFormat*>(obj.ptr());
        return static_cast<int8_t>(memory_format->memory_format);
      }
      // 向后兼容性处理，将 obj 强制转换为 int64_t 类型并返回
      return py::cast<int64_t>(obj);
    }

    case TypeKind::SymIntType:
      // 处理符号整数类型的情况
      // 检查是否为 torch 的符号整数类型，转换为 SymInt 类型
      if (torch::is_symint(obj.ptr())) {
        return py::cast<c10::SymInt>(obj);
      }
      // 将 obj 强制转换为 int64_t 类型并返回
      return py::cast<int64_t>(obj);

    case TypeKind::SymFloatType:
      // 处理符号浮点数类型的情况
      // 检查是否为 torch 的符号浮点数类型，转换为 SymFloat 类型
      if (torch::is_symfloat(obj.ptr())) {
        return py::cast<c10::SymFloat>(obj);
      }
      // 将 obj 强制转换为 double 类型并返回
      return py::cast<double>(obj);

    case TypeKind::SymBoolType:
      // 处理符号布尔类型的情况
      // 检查是否为 torch 的符号布尔类型，转换为 SymBool 类型
      if (torch::is_symbool(obj.ptr())) {
        return py::cast<c10::SymBool>(obj);
      }
      // 将 obj 强制转换为 bool 类型并返回
      return py::cast<bool>(obj);

    case TypeKind::NoneType:
      // 处理 None 类型的情况
      // 如果 obj 不是 None，则抛出类型转换错误
      if (!obj.is_none()) {
        throw py::cast_error(
            c10::str("Cannot cast ", py::str(obj), " to None"));
      }
      // 返回空的字典
      return {};
    case TypeKind::BoolType:
      // 如果类型为布尔类型
      if (torch::is_symbool(obj.ptr())) {
        // 检查是否为符号布尔类型，如果是，则返回相应的 SymBool 对象
        return py::cast<c10::SymBool>(obj).guard_bool(__FILE__, __LINE__);
      }
      // 如果是 THPVariable 类型
      if (THPVariable_Check(obj.ptr())) {
        // 尝试将对象转换为 autograd::Variable 类型
        auto var = py::cast<autograd::Variable>(obj);
        // 如果存储类型为元数据，抛出类型转换错误
        if (var.storage().device_type() == c10::kMeta) {
          throw py::cast_error(
              "cannot extract bool from tensor with meta storage");
        }
      }
      // 否则，将对象转换为普通的布尔类型并返回
      return py::cast<bool>(obj);
    case TypeKind::TupleType: {
      // 如果类型为元组类型
      py::tuple tuple = py::cast<py::tuple>(obj);
      size_t tuple_size = tuple.size();
      auto tuple_type = type->cast<TupleType>();
      const auto& elem_types = tuple_type->elements();
      // 检查元组中元素的数量是否与类型中声明的元素数量一致
      if (elem_types.size() != tuple_size) {
        throw py::cast_error(c10::str(
            "Object ",
            py::str(obj),
            " had a different number of elements than type ",
            type->repr_str()));
      }
      std::vector<IValue> values;
      values.reserve(tuple_size);
      // 遍历元组中的每个元素，将其转换为 IValue，并添加到 values 中
      for (const auto i : c10::irange(tuple_size)) {
        values.push_back(toIValue(tuple[i], elem_types[i]));
      }
      // 如果元组类型有名称，则创建带有名称的 Tuple::create，否则创建普通 Tuple::create
      return tuple_type->name()
          ? c10::ivalue::Tuple::createNamed(std::move(values), tuple_type)
          : c10::ivalue::Tuple::create(std::move(values));
    }
    case TypeKind::UnionType: {
      // 如果类型为联合类型
      auto actual_type = toTypeInferredIValue(obj);
      auto actual_type_ptr = actual_type.type();
      auto union_type = type->expect<UnionType>();
      // 检查实际类型是否是联合类型的子类型
      if (!actual_type_ptr->isSubtypeOf(union_type)) {
        throw py::cast_error(c10::str(
            "Expected a member of ",
            union_type->annotation_str(),
            " but instead found type ",
            actual_type.type()->annotation_str()));
      }
      // 返回实际类型的 IValue
      return actual_type;
    }
    case TypeKind::StringType:
      // 如果类型为字符串类型，创建 ConstantString 对象并返回
      return ConstantString::create(py::cast<std::string>(obj));
    case TypeKind::DeviceObjType: {
      // 如果类型为设备对象类型
      if (THPDevice_Check(obj.ptr())) {
        // 如果对象是 THPDevice 类型，从中获取设备信息并返回
        auto device = reinterpret_cast<THPDevice*>(obj.ptr());
        return device->device;
      }
      // 否则，根据字符串创建 c10::Device 对象并返回
      return c10::Device(py::cast<std::string>(obj.ptr()));
    }
    case TypeKind::StreamObjType: {
      // 如果类型为流对象类型
      auto thp_stream = reinterpret_cast<THPStream*>(obj.ptr());
      // 根据 THPStream 对象的属性解包成 c10::Stream 对象并返回
      auto stream = c10::Stream::unpack3(
          thp_stream->stream_id,
          thp_stream->device_index,
          static_cast<c10::DeviceType>(thp_stream->device_type));
      return stream;
    }
    }
    case TypeKind::DictType: {
      // 如果类型为字典类型
      const auto& dict_type = type->expect<DictType>();

      // 如果对象是 ScriptDict 类型，则返回其中的 c10::Dict 实例
      try {
        auto script_dict = py::cast<ScriptDict>(obj);
        return script_dict.dict_;
      } catch (py::cast_error& e) {
      }

      // 如果对象是普通 Python 字典，则创建一个新的 c10::Dict 返回
      return createGenericDict(
          py::cast<py::dict>(obj),
          dict_type->getKeyType(),
          dict_type->getValueType());
    }
    case TypeKind::OptionalType: {
      // 检查是否为 None 对象，因为 Optional 类型可以接受 NoneType
      if (obj.is_none()) {
        // 如果是 None 对象，返回一个空的 IValue，表示 NoneType
        return {};
      }
      // 否则，将对象转换为 IValue，传递给它的元素类型，继续处理
      return toIValue(obj, type->expectRef<OptionalType>().getElementType(), N);
    }
    case TypeKind::ClassType: {
      auto classType = type->expect<ClassType>();
      auto object = py::cast<py::object>(obj);
      
      // 如果 obj 已经是 ScriptModule，则直接返回其对应的 ivalue
      if (auto mod = as_module(object)) {
        return mod.value()._ivalue();
      }

      // 检查 obj 是否为 ScriptObject
      if (auto script_obj = as_object(object)) {
        return script_obj.value()._ivalue();
      }

      // 否则，obj 是普通的类对象，需要创建一个新的 ivalue::Object 来使用
      // 1. 创建一个空的 ivalue
      const size_t numAttrs = classType->numAttributes();
      auto cu = classType->compilation_unit();
      auto userObj = c10::ivalue::Object::create(
          c10::StrongTypePtr(cu, classType), numAttrs);

      // 2. 复制所有包含的属性
      for (const auto slot : c10::irange(numAttrs)) {
        const auto& attrType = classType->getAttribute(slot);
        const auto& attrName = classType->getAttributeName(slot);

        // 检查 obj 是否具有该属性
        if (!py::hasattr(obj, attrName.c_str())) {
          throw py::cast_error(c10::str(
              "Tried to cast object to type ",
              type->repr_str(),
              " but object",
              " was missing attribute ",
              attrName));
        }

        try {
          // 获取属性的值并将其转换为相应的 IValue，然后设置到 userObj 的 slot 中
          const auto& contained = py::getattr(obj, attrName.c_str());
          userObj->setSlot(slot, toIValue(contained, attrType));
        } catch (std::exception& e) {
          throw py::cast_error(c10::str(
              "Could not cast attribute '",
              attrName,
              "' to type ",
              attrType->repr_str(),
              ": ",
              e.what()));
        }
      }
      // 返回创建的 userObj
      return userObj;
    }
    case TypeKind::InterfaceType: {
      auto interfaceType = type->expect<InterfaceType>();
      // 当转换一个 pyobj 到接口时，我们检查 rhs 是否是模块或普通的 torchscript 类，
      // 并相应地从它们中获取类型和 ivalue。
      c10::ClassTypePtr classType = nullptr;
      IValue res;
      if (auto mod = as_module(py::cast<py::object>(obj))) {
        // 如果 obj 是模块，则获取其类型和 ivalue
        classType = mod.value().type();
        res = mod.value()._ivalue();
      } else if (auto object = as_object(py::cast<py::object>(obj))) {
        // 如果 obj 是对象，则获取其类型和 ivalue
        classType = object.value().type();
        res = object.value()._ivalue();
      } else {
        // 否则，我们检查值以找到编译的 TorchScript 类，
        // 然后从该类类型创建一个 ivalue::Object。
        py::str qualified_name = py::module::import("torch._jit_internal")
                                     .attr("_qualified_name")(obj.get_type());
        auto pyCu = get_python_cu();
        // 从 Python 的 CompilationUnit 中获取类类型
        classType = pyCu->get_class(c10::QualifiedName(qualified_name));
        if (!classType) {
          // 如果找不到类类型，则抛出运行时错误
          throw std::runtime_error(c10::str(
              "Assigning the object ",
              py::str(obj),
              " to an interface fails because the value is not "
              "a TorchScript compatible type, did you forget to",
              "turn it into a user defined TorchScript class?"));
        }
        // 将 obj 转换为对应的 IValue
        res = toIValue(obj, classType);
      }
      // 检查 classType 是否符合接口的要求
      std::stringstream why_not;
      if (!classType->isSubtypeOfExt(*interfaceType, &why_not)) {
        // 如果不符合接口要求，则抛出类型转换错误
        throw py::cast_error(c10::str(
            "Object of type ",
            classType->repr_str(),
            " is not compatible with interface ",
            interfaceType->repr_str(),
            "\n",
            why_not.str()));
      }
      // 返回转换后的 IValue
      return res;
    }
    case TypeKind::NumberType: {
      // 如果对象的类型是 NumberType
      if (THPDtype_Check(obj.ptr())) {
        // 如果对象是 THPDtype 类型，将其转换为 THPDtype 指针
        auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
        // 返回 dtype 的 scalar_type 强制转换为 int64_t 类型
        return static_cast<int64_t>(dtype->scalar_type);
      }
      // 如果对象是 THPQScheme 类型
      if (THPQScheme_Check(obj.ptr())) {
        // 将对象转换为 THPQScheme 指针
        auto qscheme = reinterpret_cast<THPQScheme*>(obj.ptr());
        // 返回 qscheme 的 qscheme 强制转换为 uint8_t 类型
        return static_cast<uint8_t>(qscheme->qscheme);
      }
      // 如果对象是 THPLayout 类型
      if (THPLayout_Check(obj.ptr())) {
        // 将对象转换为 THPLayout 指针
        auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
        // 返回 layout 的 layout 强制转换为 int8_t 类型
        return static_cast<int8_t>(layout->layout);
      }
      // 如果对象是 Python 的 bool 类型
      if (py::isinstance<py::bool_>(obj)) {
        // 将对象转换为 bool 类型并返回
        return py::cast<bool>(obj);
      } else if (py::isinstance<py::int_>(obj)) {
        // 如果对象是 Python 的 int 类型，将其转换为 int64_t 类型并返回
        return py::cast<int64_t>(obj);
      } else if (py::isinstance<py::float_>(obj)) {
        // 如果对象是 Python 的 float 类型，将其转换为 double 类型并返回
        return py::cast<double>(obj);
      } else if (PyComplex_CheckExact(obj.ptr())) {
        // 如果对象是 Python 的 complex 类型
        auto c_obj = py::cast<std::complex<double>>(obj.ptr());
        // 将 complex 对象转换为 c10::complex<double> 类型并返回
        return static_cast<c10::complex<double>>(c_obj);
      } else if (torch::is_symint(obj)) {
        // 如果对象是 torch 的 SymInt 类型，将其转换为 c10::SymInt 类型并返回
        return py::cast<c10::SymInt>(obj);
      } else if (torch::is_symfloat(obj)) {
        // 如果对象是 torch 的 SymFloat 类型，将其转换为 c10::SymFloat 类型并返回
        return py::cast<c10::SymFloat>(obj);
      } else if (torch::is_symbool(obj)) {
        // 如果对象是 torch 的 SymBool 类型，将其转换为 c10::SymBool 类型并返回
        return py::cast<c10::SymBool>(obj);
      } else {
        // 如果对象无法转换，抛出异常
        throw py::cast_error(
            c10::str("Cannot cast ", py::str(obj), " to ", type->repr_str()));
      }
    }
    case TypeKind::RRefType: {
#ifdef USE_RPC
      // 如果定义了 USE_RPC 宏，则返回 PyRRef 对象的 IValue 表示
      return obj.cast<torch::distributed::rpc::PyRRef>().toIValue();
#else
      // 否则，抛出错误，因为 RRef 仅在分布式包中支持
      AT_ERROR("RRef is only supported with the distributed package");
#endif
    } break;
    case TypeKind::PyObjectType: {
      // 如果是 Python 对象类型，则创建 ConcretePyObjectHolder 包装对象并返回
      return c10::ivalue::ConcretePyObjectHolder::create(obj);
    }
    case TypeKind::CapsuleType: {
      // 如果是 Capsule 类型，则使用 Capsule 对象指针创建 IValue
      return IValue::make_capsule(py::cast<c10::Capsule>(obj).obj_ptr);
    }
    case TypeKind::FutureType: {
      // 如果是 Future 类型，则返回 PythonFutureWrapper 的共享指针
      return obj.cast<std::shared_ptr<PythonFutureWrapper>>()->fut;
    }
    case TypeKind::AwaitType: {
      // 如果是 Await 类型，则返回 PythonAwaitWrapper 的 aw_ 成员
      return obj.cast<std::shared_ptr<PythonAwaitWrapper>>()->aw_;
    }
    case TypeKind::AnyType:
      // 如果是 Any 类型，则调用 toTypeInferredIValue 函数进行推断转换
      return toTypeInferredIValue(obj);
    case TypeKind::QSchemeType: {
      // 如果是 QScheme 类型，且 obj 是 py::int_ 类型，则转换为 at::QScheme
      if (py::isinstance<py::int_>(obj)) {
        return static_cast<at::QScheme>(py::cast<int64_t>(obj));
      }
      // 否则抛出转换错误
      throw py::cast_error(
          c10::str("Cannot cast ", py::str(obj), " to ", type->repr_str()));
    }
    case TypeKind::GeneratorType:
      // 如果是 Generator 类型，则转换为 at::Generator 类型并返回
      return py::cast<at::Generator>(obj);
    case TypeKind::DynamicType:
    case TypeKind::FunctionType:
    case TypeKind::QuantizerType:
    case TypeKind::VarType:
    case TypeKind::AnyListType:
    case TypeKind::AnyTupleType:
    case TypeKind::AnyClassType:
    case TypeKind::AnyEnumType:
      // 对于这些类型，不进行处理，直接跳过
      break;
    case TypeKind::EnumType:
      // 如果是 Enum 类型，则处理为 EnumHolder 对象并返回其 IValue 表示
      EnumTypePtr enum_type = type->expect<EnumType>();
      py::object py_obj = py::reinterpret_borrow<py::object>(obj);
      std::string name = py::cast<std::string>(obj.attr("name"));
      IValue value = toIValue(obj.attr("value"), enum_type->getValueType(), {});
      auto enum_holder =
          c10::make_intrusive<c10::ivalue::EnumHolder>(enum_type, name, value);
      return IValue(enum_holder);
  }
  // 如果没有匹配到任何类型，抛出转换错误，指明转换类型不支持
  throw py::cast_error(c10::str(
      "toIValue() cannot handle converting to type: ", type->repr_str()));
}

py::object toPyObject(IValue ivalue) {
  if (ivalue.isNone()) {
    // 如果是 None 类型的 IValue，则返回 Python 的 None 对象
    return py::none();
  } else if (ivalue.isTensor()) {
    // 如果是 Tensor 类型的 IValue，则转换为 Tensor 对象并返回
    auto tensor = std::move(ivalue).toTensor();
    // 检查张量是否包装了数字
    if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      // 断言张量在 CPU 上
      TORCH_INTERNAL_ASSERT(tensor.device().is_cpu());
      // 将张量转换为 Python 对象
      auto py_tensor = py::cast(tensor);
      // 检查 Python 对象是否有 "_wrapped_number" 属性
      if (PyObject_HasAttrString(py_tensor.ptr(), "_wrapped_number")) {
        // 返回 Python 对象的 "_wrapped_number" 属性
        return py_tensor.attr("_wrapped_number");
      }
      // 获取张量的标量类型
      auto scalar_type = tensor.scalar_type();
      // 根据标量类型进行处理
      switch (scalar_type) {
        // 布尔类型
        case at::ScalarType::Bool:
          return py::cast(*tensor.const_data_ptr<bool>());
        // 长整型
        case at::ScalarType::Long:
          return py::cast(*tensor.const_data_ptr<int64_t>());
        // 双精度浮点型
        case at::ScalarType::Double:
          return py::cast(*tensor.const_data_ptr<double>());
        // 复数双精度浮点型
        case at::ScalarType::ComplexDouble:
          // TODO: https://github.com/pytorch/pytorch/issues/77134
          // 将复数双精度浮点数转换为 Python 对象
          return py::cast(static_cast<std::complex<double>>(
              *tensor.const_data_ptr<c10::complex<double>>()));
        // 默认情况，抛出异常
        default:
          TORCH_CHECK(
              false,
              "Missing cases in 'toPyObject' wrapped number handling! Can't convert ",
              scalar_type,
              " to a Python object");
      }
    } else {
      // 如果不是包装的数字，防止命名张量
      guardAgainstNamedTensor<at::Tensor>(tensor);
      // 转换为 PyTorch 自动求导变量并返回
      return py::cast(autograd::Variable(std::move(tensor)));
    }
  } else if (ivalue.isStorage()) {
    // 如果是存储类型，转换为 Python 对象并返回
    return py::cast(std::move(ivalue).toStorage());
  } else if (ivalue.isGenerator()) {
    // 如果是生成器类型，转换为 Python 对象并返回
    return py::cast(std::move(ivalue).toGenerator());
  } else if (ivalue.isDouble()) {
    // 如果是双精度浮点数类型，转换为 Python 对象并返回
    return py::cast(std::move(ivalue).toDouble());
  } else if (ivalue.isComplexDouble()) {
    // 如果是复数双精度浮点数类型，转换为 Python 对象并返回
    return py::cast(
        static_cast<std::complex<double>>(std::move(ivalue).toComplexDouble()));
  } else if (ivalue.isInt()) {
    // 如果是整型，转换为 Python 对象并返回
    return py::cast(std::move(ivalue).toInt());
  } else if (ivalue.isBool()) {
    // 如果是布尔类型，转换为 Python 对象并返回
    return py::cast(std::move(ivalue).toBool());
  } else if (ivalue.isString()) {
    // 如果是字符串类型
    if (getUTF8DecodingIgnore()) {
      // 如果需要忽略 UTF-8 解码错误
      std::string s = std::move(ivalue).toStringRef();
      // 将字符串解码为 Python Unicode 对象并返回
      PyObject* pyObj = PyUnicode_DecodeUTF8(s.data(), s.length(), "ignore");
      return py::reinterpret_steal<py::object>(pyObj);
    } else {
      // 否则，直接转换为 Python 字符串对象并返回
      return py::cast(std::move(ivalue).toStringRef());
    }
  } else if (ivalue.isList()) {
    // 如果是列表类型
    auto list = std::move(ivalue).toList();
    // 创建 Python 列表对象
    py::list t{list.size()};
    // 遍历列表中的元素，递归调用 toPyObject 转换每个元素，并存入 Python 列表
    for (const auto i : c10::irange(list.size())) {
      t[i] = toPyObject(IValue{list.get(i)});
    }
    // 返回 Python 列表对象
    return std::move(t);
  } else if (ivalue.isTuple()) {
    // 如果是元组类型
    auto tuple = std::move(ivalue).toTuple();
    // 获取元组的所有元素
    const auto& elements = tuple->elements();
    // 创建 Python 元组对象
    py::tuple t{elements.size()};
    // 遍历元组中的元素，递归调用 toPyObject 转换每个元素，并存入 Python 元组
    for (const auto i : c10::irange(elements.size())) {
      t[i] = toPyObject(IValue{elements.at(i)});
    }
    // 返回 Python 元组对象
    return std::move(t);
    // 如果我们有一个命名元组
    // 检查 tuple 是否有类型，并且类型有 schema，并且 schema 的名字不为空
    if (tuple->type() && tuple->type()->schema() &&
        !tuple->type()->schema()->name().empty()) {
      // 获取未限定的类型名称
      auto unqualName = tuple->type()->name()->name();

      // 获取 tuple 类型的参数列表
      std::vector<Argument> tuple_args = tuple->type()->schema()->arguments();

      // 准备存储参数默认值的向量
      std::vector<pybind11::object> defaults;
      
      // 查找第一个带有默认值的参数，并将其转换为 Python 对象
      auto it = std::find_if(
          tuple_args.begin(), tuple_args.end(), [](const Argument& arg) {
            return arg.default_value().has_value();
          });
      std::transform(
          it,
          tuple_args.end(),
          std::back_inserter(defaults),
          [](const Argument& arg) { return toPyObject(*arg.default_value()); });

      // 获取字段名列表
      std::vector<std::string> fieldNames =
          fmap(tuple_args, [](const Argument& arg) { return arg.name(); });

      // 调用 torch._jit_internal 模块中的 _create_named_tuple 函数，创建命名元组
      return py::module::import("torch._jit_internal")
          .attr("_create_named_tuple")(
              t, unqualName, fieldNames, py::make_tuple(defaults));
    } else {
      // 如果 tuple 不符合上述条件，直接返回移动后的 t
      return std::move(t);
    }
  } else if (ivalue.isDevice()) {
    // 如果 ivalue 是 Device 类型，则转换为对应的 Python 对象并返回
    return py::cast(std::move(ivalue).toDevice());
  } else if (ivalue.isStream()) {
    // 如果 ivalue 是 Stream 类型，则转换为对应的 Python 对象并返回
    return py::cast(std::move(ivalue).toStream());
  } else if (ivalue.isGenericDict()) {
    // 如果 ivalue 是 GenericDict 类型，则将其转换为 Python 字典对象并返回
    auto dict = std::move(ivalue).toGenericDict();
    py::dict py_dict;
    for (auto& pair : dict) {
      // 将 GenericDict 中的键值对转换为对应的 Python 对象，并存储到 py_dict 中
      py_dict[toPyObject(IValue{pair.key()})] =
          toPyObject(IValue{pair.value()});
    }
    return std::move(py_dict);
  } else if (ivalue.isRRef()) {
#ifdef USE_RPC
    // 如果定义了 USE_RPC 宏，则执行以下代码块
    auto RRefPtr =
        c10::dynamic_intrusive_pointer_cast<torch::distributed::rpc::RRef>(
            std::move(ivalue).toRRef());
    // 将 IValue 转换为 RPC 的 RRef，并将其封装为 PyRRef 返回给 Python
    return py::cast(torch::distributed::rpc::PyRRef(RRefPtr));
#else
    // 如果未定义 USE_RPC 宏，则抛出错误信息
    AT_ERROR("RRef is only supported with the distributed package");
#endif
  } else if (ivalue.isObject()) {
    // 如果 IValue 是一个对象
    const auto obj = std::move(ivalue).toObject();
    if (obj->type()->is_module()) {
      // 如果对象是一个模块类型，则封装为 Python 的 Module 类并返回
      return py::cast(Module(obj));
    }

    auto pyCu = get_python_cu();
    if (obj->name().find("__torch__.torch.classes") == 0) {
      // 如果对象的名称以 "__torch__.torch.classes" 开头，则封装为 Python 的 Object 类并返回
      return py::cast(Object(obj));
    }
    const auto classType = pyCu->get_class(c10::QualifiedName(obj->name()));
    AT_ASSERT(classType);
    auto pyClass = getScriptedClassOrError(obj->type());
    auto pyObj = pyClass.attr("__new__")(pyClass);

    const auto numAttrs = classType->numAttributes();

    for (const auto slot : c10::irange(numAttrs)) {
      // 遍历对象的属性并转换为 Python 对象，然后设置到新创建的 Python 对象中
      const auto& attrName = classType->getAttributeName(slot);
      IValue v = obj->getSlot(slot);
      py::setattr(pyObj, attrName.c_str(), toPyObject(std::move(v)));
    }
    // 返回转换后的 Python 对象
    return pyObj;
  } else if (ivalue.isPyObject()) {
    // 如果 IValue 是一个 Python 对象，则将其转换为 py::object 并返回
    // 使用 reinterpret_borrow 来确保正确增加引用计数
    return py::reinterpret_borrow<py::object>(ivalue.toPyObject());
  } else if (ivalue.isCapsule()) {
    // 如果 IValue 是一个 Capsule，则转换为 c10::Capsule 并封装为 Python 对象返回
    return py::cast(c10::Capsule(ivalue.toCapsule()));
  } else if (ivalue.isFuture()) {
    // 如果 IValue 是一个 Future，则封装为 PythonFutureWrapper 并返回
    return py::cast(std::make_shared<PythonFutureWrapper>(ivalue.toFuture()));
  } else if (ivalue.isAwait()) {
    // 如果 IValue 是一个 Await，则封装为 PythonAwaitWrapper 并返回
    return py::cast(std::make_shared<PythonAwaitWrapper>(ivalue.toAwait()));
  } else if (ivalue.isEnum()) {
    // 如果 IValue 是一个枚举类型，则获取其对应的 Python 类并返回
    auto enum_holder = ivalue.toEnumHolder();
    auto py_class = getScriptedClassOrError(enum_holder->type());
    return py_class.attr(enum_holder->name().c_str());
  } else if (ivalue.isRRef()) {
#ifdef USE_RPC
    // 如果定义了 USE_RPC 宏，则将 RRef 转换为 PyRRef 并返回
    return py::cast(torch::distributed::rpc::PyRRef(
        c10::static_intrusive_pointer_cast<distributed::rpc::RRef>(
            ivalue.toRRef())));
#else
    // 如果未定义 USE_RPC 宏，则抛出错误信息
    TORCH_CHECK(false, "RRef is only supported with the distributed package");
#endif
  } else if (ivalue.isSymInt()) {
    // 如果 IValue 是一个符号整数，则转换为 Python 的 int 类型并返回
    return py::cast(std::move(ivalue).toSymInt());
  } else if (ivalue.isSymFloat()) {
    // 如果 IValue 是一个符号浮点数，则转换为 Python 的 float 类型并返回
    return py::cast(std::move(ivalue).toSymFloat());
  } else if (ivalue.isSymBool()) {
    // 如果 IValue 是一个符号布尔值，则转换为 Python 的 bool 类型并返回
    return py::cast(std::move(ivalue).toSymBool());
  } else {
    // 如果 IValue 类型未知，则抛出错误信息
    AT_ERROR(
        "Missing cases in 'toPyObject'! Can't convert ",
        ivalue.tagKind(),
        " to a Python object");
  }
}

std::pair<std::shared_ptr<Operator>, Stack> getOpWithStack(
    const std::vector<std::shared_ptr<Operator>>& operations,
    py::args args,
    const py::kwargs& kwargs) {
  // 函数用途：从一组操作中获取操作符和操作栈

  Stack stack;
  if (operations.size() == 1) {
    // 如果操作数的大小为 1
    std::shared_ptr<Operator> op = operations.at(0);
    // 获取第一个操作符
    // 创建一个栈，其中包含参数和关键字参数
    stack = createStackForSchema(
        op->schema(), std::move(args), kwargs, c10::nullopt);
    // 返回一个包含操作符和堆栈的 std::pair 对象
    return std::make_pair(std::move(op), std::move(stack));
  } else {
    // 如果未找到匹配的操作符，则记录错误信息
    std::vector<schema_match_error> errors;
    // 初始化一个空的共享指针，用于存储找到的操作符
    std::shared_ptr<Operator> found_op = nullptr;
    // 遍历所有的操作符
    for (const auto& op : operations) {
      try {
        // 尝试为当前操作符 op 创建与指定模式匹配的堆栈
        stack = createStackForSchema(op->schema(), args, kwargs, c10::nullopt);
        // 如果成功匹配模式，则记录该操作符，并退出循环
        found_op = op;
        break;
      } catch (schema_match_error& error) {
        // 如果捕获到模式匹配错误，将错误存储到 errors 向量中
        errors.push_back(std::move(error));
      }
    }
    // 如果未找到任何匹配的操作符，则抛出运行时异常，包含所有捕获的错误信息
    if (!found_op) {
      std::stringstream ss;
      ss << "Overloaded torch operator invoked from Python failed to match any schema:\n";
      for (const auto& err : errors) {
        ss << err.what() << "\n\n";
      }
      throw std::runtime_error(ss.str());
    }

    // 返回一个包含找到的操作符和堆栈的 std::pair 对象
    return std::make_pair(std::move(found_op), std::move(stack));
  }
// 检查给定的函数模式对象是否允许使用 FakeScriptObject 进行匹配
// 函数返回一个布尔值，指示是否匹配成功
bool checkSchemaAllowFakeScriptObject(
    const FunctionSchema& schema,  // 函数模式对象
    py::args args,                  // 位置参数
    const py::kwargs& kwargs) {     // 关键字参数
  bool match = false;  // 初始化匹配结果为 false
  try {
    // 调用 matchSchemaAllowFakeScriptObject 函数进行模式匹配
    match = matchSchemaAllowFakeScriptObject(schema, std::move(args), kwargs);
  } catch (schema_match_error& error) {  // 捕获模式匹配错误
    throw std::runtime_error(error.what());  // 抛出运行时错误，包含错误信息
  }
  return match;  // 返回匹配结果
}

// 从 Python 中调用操作符函数，并返回其结果对象
py::object invokeOperatorFromPython(
    const std::vector<std::shared_ptr<Operator>>& operations,  // 操作符列表
    py::args args,                  // 位置参数
    const py::kwargs& kwargs,       // 关键字参数
    std::optional<c10::DispatchKey> dk) {  // 分发键的可选参数
  auto [found_op, stack] = getOpWithStack(operations, args, kwargs);  // 调用操作符获取操作对象和堆栈
  {
    pybind11::gil_scoped_release no_gil_guard;  // 释放 GIL，允许多线程操作
    if (dk) {
      found_op->getOperationForDispatchKey(*dk)(stack);  // 根据分发键调用操作
    } else {
      found_op->getOperation()(stack);  // 调用默认操作
    }
  }

  return createPyObjectForStack(std::move(stack));  // 根据堆栈创建 Python 对象并返回
}

// 处理可能存在的 torch 函数调用
// 返回一个可选的 Python 对象，表示处理后的结果
std::optional<py::object> _maybe_handle_torch_function(
    const std::string& ns,          // 命名空间
    const std::string& method_name, // 方法名
    const std::string& overload_name,  // 重载名
    bool is_overload,               // 是否是重载
    py::args args,                  // 位置参数
    const py::kwargs& kwargs) {     // 关键字参数
  std::vector<PyObject*> overloaded_args;  // 重载参数列表
  size_t total_arg_num = args.size() + kwargs.size();  // 总参数数量

  // 检查位置参数是否是张量类型，并将满足条件的参数加入重载参数列表
  for (const auto i : c10::irange(args.size())) {
    is_tensor_and_append_overloaded(args[i].ptr(), &overloaded_args);
    is_tensor_list_and_append_overloaded(
        args[i].ptr(),
        &overloaded_args,
        static_cast<int>(total_arg_num),
        false /* throw_error */);
  }

  // 检查关键字参数是否是张量类型，并将满足条件的参数加入重载参数列表
  // 注意：关键字参数的顺序不一定与操作符模式中的参数顺序相同
  for (auto item : kwargs) {
    is_tensor_and_append_overloaded(item.second.ptr(), &overloaded_args);
    is_tensor_list_and_append_overloaded(
        item.second.ptr(),
        &overloaded_args,
        total_arg_num,
        false /* throw_error */);
  }

  // 如果有重载参数或者 torch 函数模式启用了 torch_function 模式
  if (!overloaded_args.empty() || at::impl::torch_function_mode_enabled()) {
    // 导入对应的 torch 操作符函数
    auto self_func = py::module::import("torch")
                         .attr("ops")
                         .attr(ns.c_str())
                         .attr(method_name.c_str());

    // 如果是重载函数，则根据重载名选择具体的重载版本，否则选择默认版本
    if (is_overload) {
      if (overload_name.empty()) {
        self_func = self_func.attr("default");
      } else {
        self_func = self_func.attr(overload_name.c_str());
      }
    }
    std::string module_name("torch.ops");
    module_name.append(ns);
    // 调用 handle_torch_function_no_python_arg_parser 处理 Torch 函数重载的参数，
    // 返回一个 PyObject* 对象，并通过 pybind11::reinterpret_steal 转换为 py::object。
    return {pybind11::reinterpret_steal<py::object>(
        handle_torch_function_no_python_arg_parser(
            overloaded_args,  // 重载的参数列表
            args.ptr(),       // 传递的位置参数
            kwargs.ptr(),     // 传递的关键字参数
            method_name.c_str(),  // 方法名的 C 字符串
            self_func.ptr(),   // self 对象的 PyObject*
            module_name.c_str()  // 模块名的 C 字符串
        ))};
    // 如果 handle_torch_function_no_python_arg_parser 返回 std::nullopt，则返回空的 optional 对象。
  }
  // 如果没有返回语句执行，则默认返回 std::nullopt。
  return std::nullopt;
}

py::object _get_operation_for_overload_or_packet(
    const std::vector<std::shared_ptr<Operator>>& operations, // 参数 operations：操作符的共享指针向量
    Symbol symbol,                                             // 参数 symbol：表示操作符的符号
    py::args args,                                             // 参数 args：Python 中的位置参数
    const py::kwargs& kwargs,                                   // 参数 kwargs：Python 中的关键字参数
    bool is_overload,                                          // 参数 is_overload：指示是否是重载操作的布尔值
    std::optional<c10::DispatchKey> dk) {                      // 参数 dk：C10 的 DispatchKey 的可选类型
  std::string ns = symbol.ns().toUnqualString();               // 从符号获取命名空间的非限定字符串表示
  std::string method_name = symbol.toUnqualString();           // 从符号获取方法名的非限定字符串表示
  std::string overload_name = operations[0]->schema().overload_name();  // 获取第一个操作符的模式重载名称
  auto res = _maybe_handle_torch_function(
      ns, method_name, overload_name, is_overload, args, kwargs);   // 调用 _maybe_handle_torch_function 处理 Torch 函数的可能性
  auto torch_function_called = res.has_value();                // 检查是否成功调用了 Torch 函数
  return torch_function_called
      ? *res                                                  // 如果成功调用了 Torch 函数，则返回结果
      : invokeOperatorFromPython(operations, args, kwargs, dk);  // 否则调用 invokeOperatorFromPython 来执行操作符
}

} // namespace torch::jit  // 结束 torch::jit 命名空间
```