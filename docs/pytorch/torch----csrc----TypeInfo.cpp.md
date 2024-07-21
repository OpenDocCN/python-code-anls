# `.\pytorch\torch\csrc\TypeInfo.cpp`

```
#include <torch/csrc/TypeInfo.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_dtypes.h>

#include <ATen/Dispatch_v2.h>

#include <c10/util/Exception.h>

#include <structmember.h>
#include <cstring>
#include <limits>
#include <sstream>

// 创建一个新的 THPFInfo 对象，用于浮点数类型信息
PyObject* THPFInfo_New(const at::ScalarType& type) {
  auto finfo = (PyTypeObject*)&THPFInfoType;  // 获取 THPFInfoType 的类型对象指针
  auto self = THPObjectPtr{finfo->tp_alloc(finfo, 0)};  // 分配 THPFInfo 对象内存
  if (!self)
    throw python_error();  // 如果分配失败，抛出异常
  auto self_ = reinterpret_cast<THPDTypeInfo*>(self.get());  // 将 self 转换为 THPDTypeInfo 指针
  self_->type = c10::toRealValueType(type);  // 设置对象的数据类型
  return self.release();  // 返回 PyObject*，释放 self 所有权
}

// 创建一个新的 THPIInfo 对象，用于整数类型信息
PyObject* THPIInfo_New(const at::ScalarType& type) {
  auto iinfo = (PyTypeObject*)&THPIInfoType;  // 获取 THPIInfoType 的类型对象指针
  auto self = THPObjectPtr{iinfo->tp_alloc(iinfo, 0)};  // 分配 THPIInfo 对象内存
  if (!self)
    throw python_error();  // 如果分配失败，抛出异常
  auto self_ = reinterpret_cast<THPDTypeInfo*>(self.get());  // 将 self 转换为 THPDTypeInfo 指针
  self_->type = type;  // 设置对象的数据类型
  return self.release();  // 返回 PyObject*，释放 self 所有权
}

// Python 的构造函数，用于创建 THPFInfo 对象
PyObject* THPFInfo_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS  // 开始错误处理块
  static torch::PythonArgParser parser({
      "finfo(ScalarType type)",  // 构造函数参数的文档字符串
      "finfo()",  // 构造函数参数的文档字符串
  });

  torch::ParsedArgs<1> parsed_args;  // 解析参数对象
  auto r = parser.parse(args, kwargs, parsed_args);  // 解析输入参数
  TORCH_CHECK(r.idx < 2, "Not a type");  // 检查是否为有效类型

  at::ScalarType scalar_type = at::ScalarType::Undefined;  // 初始化标量类型
  if (r.idx == 1) {
    scalar_type = torch::tensors::get_default_scalar_type();  // 获取默认张量类型
    // 默认张量类型只能设置为浮点数类型
    AT_ASSERT(at::isFloatingType(scalar_type));
  } else {
    scalar_type = r.scalartype(0);  // 获取指定的标量类型
    // 如果标量类型既不是浮点数类型也不是复数类型，返回类型错误
    if (!at::isFloatingType(scalar_type) && !at::isComplexType(scalar_type)) {
      return PyErr_Format(
          PyExc_TypeError,
          "torch.finfo() requires a floating point input type. Use torch.iinfo to handle '%s'",
          type->tp_name);
    }
  }
  return THPFInfo_New(scalar_type);  // 调用创建 THPFInfo 对象的函数并返回结果
  END_HANDLE_TH_ERRORS  // 结束错误处理块
}

// Python 的构造函数，用于创建 THPIInfo 对象
PyObject* THPIInfo_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS  // 开始错误处理块
  static torch::PythonArgParser parser({
      "iinfo(ScalarType type)",  // 构造函数参数的文档字符串
  });
  torch::ParsedArgs<1> parsed_args;  // 解析参数对象
  auto r = parser.parse(args, kwargs, parsed_args);  // 解析输入参数
  TORCH_CHECK(r.idx == 0, "Not a type");  // 检查是否为有效类型

  at::ScalarType scalar_type = r.scalartype(0);  // 获取指定的标量类型
  // 如果标量类型为布尔类型，返回类型错误
  if (scalar_type == at::ScalarType::Bool) {
    return PyErr_Format(
        PyExc_TypeError, "torch.bool is not supported by torch.iinfo");
  }
  // 如果既不是整数类型也不是量化整数类型，返回类型错误
  if (!at::isIntegralType(scalar_type, /*includeBool=*/false) &&
      !at::isQIntType(scalar_type)) {
    return PyErr_Format(
        PyExc_TypeError,
        "torch.iinfo() requires an integer input type. Use torch.finfo to handle '%s'",
        type->tp_name);
  }
  return THPIInfo_New(scalar_type);  // 调用创建 THPIInfo 对象的函数并返回结果
  END_HANDLE_TH_ERRORS  // 结束错误处理块
}

// 比较两个 THPDTypeInfo 对象的函数
PyObject* THPDTypeInfo_compare(THPDTypeInfo* a, THPDTypeInfo* b, int op) {
  switch (op) {
    # 对比操作：相等性比较
    case Py_EQ:
      # 如果对象 a 和 b 的类型相同，则返回 Python 的 True 对象
      if (a->type == b->type) {
        Py_RETURN_TRUE;
      } else {
        # 否则返回 Python 的 False 对象
        Py_RETURN_FALSE;
      }
    
    # 对比操作：不等性比较
    case Py_NE:
      # 如果对象 a 和 b 的类型不同，则返回 Python 的 True 对象
      if (a->type != b->type) {
        Py_RETURN_TRUE;
      } else {
        # 否则返回 Python 的 False 对象
        Py_RETURN_FALSE;
      }
  }
  # 默认情况下，增加对 Py_NotImplemented 的引用计数并返回它，表示操作未实现
  return Py_INCREF(Py_NotImplemented), Py_NotImplemented;
}

// 定义静态函数 THPDTypeInfo_bits，返回包含数据类型位数的 Python 对象
static PyObject* THPDTypeInfo_bits(THPDTypeInfo* self, void*) {
  // 计算数据类型的位数，乘以每个字节的位数 CHAR_BIT
  uint64_t bits = elementSize(self->type) * CHAR_BIT;
  // 将 bits 打包为 Python 的 UInt64 对象并返回
  return THPUtils_packUInt64(bits);
}

// 定义宏 _AT_DISPATCH_FINFO_TYPES，用于调度不同浮点数类型的函数
#define _AT_DISPATCH_FINFO_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND6(    \
      at::kHalf,                                  \
      at::ScalarType::BFloat16,                   \
      at::ScalarType::Float8_e5m2,                \
      at::ScalarType::Float8_e5m2fnuz,            \
      at::ScalarType::Float8_e4m3fn,              \
      at::ScalarType::Float8_e4m3fnuz,            \
      TYPE,                                       \
      NAME,                                       \
      __VA_ARGS__)

// 定义静态函数 THPFInfo_eps，返回指定浮点数类型的 epsilon（最小增量）值的 Python 对象
static PyObject* THPFInfo_eps(THPFInfo* self, void*) {
  HANDLE_TH_ERRORS
  // 使用宏 _AT_DISPATCH_FINFO_TYPES 调度并执行 lambda 函数来获取 epsilon 值
  return _AT_DISPATCH_FINFO_TYPES(self->type, "epsilon", [] {
    return PyFloat_FromDouble(
        std::numeric_limits<at::scalar_value_type<scalar_t>::type>::epsilon());
  });
  END_HANDLE_TH_ERRORS
}

// 定义静态函数 THPFInfo_max，返回指定浮点数类型的最大值的 Python 对象
static PyObject* THPFInfo_max(THPFInfo* self, void*) {
  HANDLE_TH_ERRORS
  // 使用宏 _AT_DISPATCH_FINFO_TYPES 调度并执行 lambda 函数来获取最大值
  return _AT_DISPATCH_FINFO_TYPES(self->type, "max", [] {
    return PyFloat_FromDouble(
        std::numeric_limits<at::scalar_value_type<scalar_t>::type>::max());
  });
  END_HANDLE_TH_ERRORS
}

// 定义静态函数 THPFInfo_min，返回指定浮点数类型的最小值的 Python 对象
static PyObject* THPFInfo_min(THPFInfo* self, void*) {
  HANDLE_TH_ERRORS
  // 使用宏 _AT_DISPATCH_FINFO_TYPES 调度并执行 lambda 函数来获取最小值
  return _AT_DISPATCH_FINFO_TYPES(self->type, "lowest", [] {
    return PyFloat_FromDouble(
        std::numeric_limits<at::scalar_value_type<scalar_t>::type>::lowest());
  });
  END_HANDLE_TH_ERRORS
}

// 定义宏 AT_DISPATCH_IINFO_TYPES，用于调度不同整数类型的函数
#define AT_DISPATCH_IINFO_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_V2(                                \
      TYPE, NAME, AT_WRAP(__VA_ARGS__), AT_EXPAND(AT_INTEGRAL_TYPES_V2))

// 定义静态函数 THPIInfo_max，返回指定整数类型的最大值的 Python 对象
static PyObject* THPIInfo_max(THPIInfo* self, void*) {
  HANDLE_TH_ERRORS
  // 检查数据类型是否为整数类型（不包括布尔型）
  if (at::isIntegralType(self->type, /*includeBool=*/false)) {
    // 使用宏 AT_DISPATCH_IINFO_TYPES 调度并执行 lambda 函数来获取最大值
    return AT_DISPATCH_IINFO_TYPES(self->type, "max", [] {
      // 如果是无符号整数类型，返回 UInt64 类型的最大值
      if (std::is_unsigned_v<scalar_t>) {
        return THPUtils_packUInt64(std::numeric_limits<scalar_t>::max());
      } else {
        // 否则返回 Int64 类型的最大值
        return THPUtils_packInt64(std::numeric_limits<scalar_t>::max());
      }
    });
  }
  // 如果是量化类型，返回它的底层类型的最大值
  return AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(self->type, "max", [] {
    return THPUtils_packInt64(std::numeric_limits<underlying_t>::max());
  });
  END_HANDLE_TH_ERRORS
}

// 定义静态函数 THPIInfo_min，返回指定整数类型的最小值的 Python 对象
static PyObject* THPIInfo_min(THPIInfo* self, void*) {
  HANDLE_TH_ERRORS
  // 检查数据类型是否为整数类型（不包括布尔型）
  if (at::isIntegralType(self->type, /*includeBool=*/false)) {
    // 使用宏 AT_DISPATCH_IINFO_TYPES 调度并执行 lambda 函数来获取最小值
    return AT_DISPATCH_IINFO_TYPES(self->type, "min", [] {
      // 如果是无符号整数类型，返回 UInt64 类型的最小值
      if (std::is_unsigned_v<scalar_t>) {
        return THPUtils_packUInt64(std::numeric_limits<scalar_t>::lowest());
      } else {
        // 否则返回 Int64 类型的最小值
        return THPUtils_packInt64(std::numeric_limits<scalar_t>::lowest());
      }
    });
  }
  // 如果是量化类型，返回它的底层类型的最小值
  return AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(self->type, "min", [] {
    return THPUtils_packInt64(std::numeric_limits<underlying_t>::lowest());
  });
  END_HANDLE_TH_ERRORS
}
// 返回包含自定义数据类型名称的 Python 对象
static PyObject* THPIInfo_dtype(THPIInfo* self, void*) {
  HANDLE_TH_ERRORS
  // 获取主要的数据类型名称
  auto primary_name = c10::getDtypeNames(self->type).first;
  // 调用 AT_DISPATCH_IINFO_TYPES 宏，并使用 primary_name 来创建 Python 字符串对象
  return AT_DISPATCH_IINFO_TYPES(self->type, "dtype", [&primary_name] {
    return PyUnicode_FromString(primary_name.data());
  });
  END_HANDLE_TH_ERRORS
}

// 返回包含最小正常值的 Python 浮点数对象
static PyObject* THPFInfo_smallest_normal(THPFInfo* self, void*) {
  HANDLE_TH_ERRORS
  // 调用 _AT_DISPATCH_FINFO_TYPES 宏，返回一个 Python 浮点数对象，表示最小正常值
  return _AT_DISPATCH_FINFO_TYPES(self->type, "min", [] {
    return PyFloat_FromDouble(
        std::numeric_limits<at::scalar_value_type<scalar_t>::type>::min());
  });
  END_HANDLE_TH_ERRORS
}

// 返回包含最小正常值的 Python 对象，调用 THPFInfo_smallest_normal 函数
static PyObject* THPFInfo_tiny(THPFInfo* self, void*) {
  // see gh-70909, essentially the array_api prefers smallest_normal over tiny
  // 查看 gh-70909，通常 array_api 更喜欢 smallest_normal 而不是 tiny
  return THPFInfo_smallest_normal(self, nullptr);
}

// 返回包含分辨率的 Python 浮点数对象
static PyObject* THPFInfo_resolution(THPFInfo* self, void*) {
  HANDLE_TH_ERRORS
  // 调用 _AT_DISPATCH_FINFO_TYPES 宏，返回一个 Python 浮点数对象，表示分辨率
  return _AT_DISPATCH_FINFO_TYPES(self->type, "digits10", [] {
    return PyFloat_FromDouble(std::pow(
        10,
        -std::numeric_limits<at::scalar_value_type<scalar_t>::type>::digits10));
  });
  END_HANDLE_TH_ERRORS
}

// 返回包含自定义数据类型名称的 Python 对象
static PyObject* THPFInfo_dtype(THPFInfo* self, void*) {
  HANDLE_TH_ERRORS
  // 获取主要的数据类型名称
  auto primary_name = c10::getDtypeNames(self->type).first;
  // 调用 _AT_DISPATCH_FINFO_TYPES 宏，并使用 primary_name 来创建 Python 字符串对象
  return _AT_DISPATCH_FINFO_TYPES(self->type, "dtype", [&primary_name] {
    return PyUnicode_FromString(primary_name.data());
  });
  END_HANDLE_TH_ERRORS
}

// 返回包含 THPFInfo 对象的字符串表示形式的 Python 对象
PyObject* THPFInfo_str(THPFInfo* self) {
  std::ostringstream oss;
  // 获取数据类型的字符串表示形式
  const auto dtypeStr = THPFInfo_dtype(self, nullptr);
  oss << "finfo(resolution="
      << PyFloat_AsDouble(THPFInfo_resolution(self, nullptr));
  oss << ", min=" << PyFloat_AsDouble(THPFInfo_min(self, nullptr));
  oss << ", max=" << PyFloat_AsDouble(THPFInfo_max(self, nullptr));
  oss << ", eps=" << PyFloat_AsDouble(THPFInfo_eps(self, nullptr));
  oss << ", smallest_normal="
      << PyFloat_AsDouble(THPFInfo_smallest_normal(self, nullptr));
  oss << ", tiny=" << PyFloat_AsDouble(THPFInfo_tiny(self, nullptr));
  if (dtypeStr != nullptr) {
    oss << ", dtype=" << PyUnicode_AsUTF8(dtypeStr) << ")";
  }
  // 如果没有错误发生，返回 THPUtils_packString 创建的 Python 字符串对象，否则返回 nullptr
  return !PyErr_Occurred() ? THPUtils_packString(oss.str().c_str()) : nullptr;
}

// 返回包含 THPIInfo 对象的字符串表示形式的 Python 对象
PyObject* THPIInfo_str(THPIInfo* self) {
  std::ostringstream oss;
  // 获取数据类型的字符串表示形式
  const auto dtypeStr = THPIInfo_dtype(self, nullptr);
  oss << "iinfo(min=" << PyLong_AsDouble(THPIInfo_min(self, nullptr));
  oss << ", max=" << PyLong_AsDouble(THPIInfo_max(self, nullptr));
  if (dtypeStr) {
    oss << ", dtype=" << PyUnicode_AsUTF8(dtypeStr) << ")";
  }
  // 如果没有错误发生，返回 THPUtils_packString 创建的 Python 字符串对象，否则返回 nullptr
  return !PyErr_Occurred() ? THPUtils_packString(oss.str().c_str()) : nullptr;
}

// 定义 THPFInfo 对象的属性列表
static struct PyGetSetDef THPFInfo_properties[] = {
    {"bits", (getter)THPDTypeInfo_bits, nullptr, nullptr, nullptr},
    {"eps", (getter)THPFInfo_eps, nullptr, nullptr, nullptr},
    {"max", (getter)THPFInfo_max, nullptr, nullptr, nullptr},
    {"min", (getter)THPFInfo_min, nullptr, nullptr, nullptr},
    {
        "smallest_normal",
        // 键："smallest_normal"，对应的值是一个函数指针，指向 THPFInfo_smallest_normal 函数
        (getter)THPFInfo_smallest_normal,
        // 没有 setter，因此为 nullptr
        nullptr,
        // 没有 docstring，因此为 nullptr
        nullptr,
        // 没有 closure，因此为 nullptr
        nullptr
    },
    {
        "tiny",
        // 键："tiny"，对应的值是一个函数指针，指向 THPFInfo_tiny 函数
        (getter)THPFInfo_tiny,
        // 没有 setter，因此为 nullptr
        nullptr,
        // 没有 docstring，因此为 nullptr
        nullptr,
        // 没有 closure，因此为 nullptr
        nullptr
    },
    {
        "resolution",
        // 键："resolution"，对应的值是一个函数指针，指向 THPFInfo_resolution 函数
        (getter)THPFInfo_resolution,
        // 没有 setter，因此为 nullptr
        nullptr,
        // 没有 docstring，因此为 nullptr
        nullptr,
        // 没有 closure，因此为 nullptr
        nullptr
    },
    {
        "dtype",
        // 键："dtype"，对应的值是一个函数指针，指向 THPFInfo_dtype 函数
        (getter)THPFInfo_dtype,
        // 没有 setter，因此为 nullptr
        nullptr,
        // 没有 docstring，因此为 nullptr
        nullptr,
        // 没有 closure，因此为 nullptr
        nullptr
    },
    // 最后一个条目为 nullptr，表示结束
    {nullptr}
// 定义一个静态的方法列表 THPFInfo_methods，用于存储 torch.finfo 类型对象的方法
static PyMethodDef THPFInfo_methods[] = {
    {nullptr} /* Sentinel */  // Sentinel，用于标记方法列表的结尾
};

// 定义 PyTypeObject 结构体 THPFInfoType，表示 torch.finfo 类型对象
PyTypeObject THPFInfoType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.finfo", /* tp_name */  // 类型对象的名称
    sizeof(THPFInfo), /* tp_basicsize */  // 对象基本大小
    0, /* tp_itemsize */  // 每个元素的大小（对于变长对象，如列表，为0）
    nullptr, /* tp_dealloc */  // 对象销毁函数
    0, /* tp_vectorcall_offset */  // Vectorcall 协议的偏移量
    nullptr, /* tp_getattr */  // 获取属性的函数
    nullptr, /* tp_setattr */  // 设置属性的函数
    nullptr, /* tp_reserved */  // 保留字段
    (reprfunc)THPFInfo_str, /* tp_repr */  // repr() 函数
    nullptr, /* tp_as_number */  // 数值类型方法集合
    nullptr, /* tp_as_sequence */  // 序列类型方法集合
    nullptr, /* tp_as_mapping */  // 映射类型方法集合
    nullptr, /* tp_hash  */  // 哈希函数
    nullptr, /* tp_call */  // 调用对象时的操作
    (reprfunc)THPFInfo_str, /* tp_str */  // str() 函数
    nullptr, /* tp_getattro */  // 获取属性操作
    nullptr, /* tp_setattro */  // 设置属性操作
    nullptr, /* tp_as_buffer */  // 缓冲区接口
    Py_TPFLAGS_DEFAULT, /* tp_flags */  // 类型对象的标志
    nullptr, /* tp_doc */  // 文档字符串
    nullptr, /* tp_traverse */  // 遍历对象引用
    nullptr, /* tp_clear */  // 清除对象资源
    (richcmpfunc)THPDTypeInfo_compare, /* tp_richcompare */  // 富比较函数
    0, /* tp_weaklistoffset */  // 弱引用列表偏移量
    nullptr, /* tp_iter */  // 迭代器方法
    nullptr, /* tp_iternext */  // 迭代器下一个元素方法
    THPFInfo_methods, /* tp_methods */  // 方法集合
    nullptr, /* tp_members */  // 成员变量集合
    THPFInfo_properties, /* tp_getset */  // 属性集合
    nullptr, /* tp_base */  // 基类
    nullptr, /* tp_dict */  // 字典
    nullptr, /* tp_descr_get */  // 获取描述器
    nullptr, /* tp_descr_set */  // 设置描述器
    0, /* tp_dictoffset */  // 字典偏移量
    nullptr, /* tp_init */  // 初始化函数
    nullptr, /* tp_alloc */  // 分配函数
    THPFInfo_pynew, /* tp_new */  // 创建新对象函数
};

// 定义一个属性集合 THPIInfo_properties，用于存储 torch.iinfo 类型对象的属性
static struct PyGetSetDef THPIInfo_properties[] = {
    {"bits", (getter)THPDTypeInfo_bits, nullptr, nullptr, nullptr},  // bits 属性及其 getter 函数
    {"max", (getter)THPIInfo_max, nullptr, nullptr, nullptr},  // max 属性及其 getter 函数
    {"min", (getter)THPIInfo_min, nullptr, nullptr, nullptr},  // min 属性及其 getter 函数
    {"dtype", (getter)THPIInfo_dtype, nullptr, nullptr, nullptr},  // dtype 属性及其 getter 函数
    {nullptr}  // Sentinel，用于标记属性集合的结尾
};

// 定义 PyTypeObject 结构体 THPIInfoType，表示 torch.iinfo 类型对象
PyTypeObject THPIInfoType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.iinfo", /* tp_name */  // 类型对象的名称
    sizeof(THPIInfo), /* tp_basicsize */  // 对象基本大小
    0, /* tp_itemsize */  // 每个元素的大小（对于变长对象，如列表，为0）
    nullptr, /* tp_dealloc */  // 对象销毁函数
    0, /* tp_vectorcall_offset */  // Vectorcall 协议的偏移量
    nullptr, /* tp_getattr */  // 获取属性的函数
    nullptr, /* tp_setattr */  // 设置属性的函数
    nullptr, /* tp_reserved */  // 保留字段
    (reprfunc)THPIInfo_str, /* tp_repr */  // repr() 函数
    nullptr, /* tp_as_number */  // 数值类型方法集合
    nullptr, /* tp_as_sequence */  // 序列类型方法集合
    nullptr, /* tp_as_mapping */  // 映射类型方法集合
    nullptr, /* tp_hash  */  // 哈希函数
    nullptr, /* tp_call */  // 调用对象时的操作
    (reprfunc)THPIInfo_str, /* tp_str */  // str() 函数
    nullptr, /* tp_getattro */  // 获取属性操作
    nullptr, /* tp_setattro */  // 设置属性操作
    nullptr, /* tp_as_buffer */  // 缓冲区接口
    Py_TPFLAGS_DEFAULT, /* tp_flags */  // 类型对象的标志
    nullptr, /* tp_doc */  // 文档字符串
    nullptr, /* tp_traverse */  // 遍历对象引用
    nullptr, /* tp_clear */  // 清除对象资源
    (richcmpfunc)THPDTypeInfo_compare, /* tp_richcompare */  // 富比较函数
    0, /* tp_weaklistoffset */  // 弱引用列表偏移量
    nullptr, /* tp_iter */  // 迭代器方法
    nullptr, /* tp_iternext */  // 迭代器下一个元素方法
    THPIInfo_methods, /* tp_methods */  // 方法集合
    nullptr, /* tp_members */  // 成员变量集合
    THPIInfo_properties, /* tp_getset */  // 属性集合
    nullptr, /* tp_base */  // 基类
    nullptr, /* tp_dict */  // 字典
    nullptr, /* tp_descr_get */  // 获取描述器
    nullptr, /* tp_descr_set */  // 设置描述器
    0, /* tp_dictoffset */  // 字典偏移量
    nullptr, /* tp_init */  // 初始化函数
    nullptr, /* tp_alloc */  // 分配函数
    THPIInfo_pynew, /* tp_new */  // 创建新对象函数
};
    nullptr, /* tp_iter */
    // 指定对象的迭代器函数，此处为 nullptr 表示没有特定的迭代器
    nullptr, /* tp_iternext */
    // 指定迭代器的下一个函数，此处为 nullptr 表示没有迭代器的下一个函数
    THPIInfo_methods, /* tp_methods */
    // 指定类型对象的方法列表，THPIInfo_methods 是一个方法列表的指针
    nullptr, /* tp_members */
    // 指定类型对象的成员变量列表，此处为 nullptr 表示没有成员变量
    THPIInfo_properties, /* tp_getset */
    // 指定类型对象的属性(getter/setter)列表，THPIInfo_properties 是一个属性列表的指针
    nullptr, /* tp_base */
    // 指定类型对象的基类，此处为 nullptr 表示没有指定基类
    nullptr, /* tp_dict */
    // 保留字段，用于支持动态属性
    nullptr, /* tp_descr_get */
    // 属性描述符的获取函数，此处为 nullptr 表示没有特定的属性描述符获取函数
    nullptr, /* tp_descr_set */
    // 属性描述符的设置函数，此处为 nullptr 表示没有特定的属性描述符设置函数
    0, /* tp_dictoffset */
    // 类型对象的字典偏移量，通常为 0
    nullptr, /* tp_init */
    // 类型对象的初始化函数，此处为 nullptr 表示没有特定的初始化函数
    nullptr, /* tp_alloc */
    // 类型对象的内存分配函数，此处为 nullptr 表示使用默认的内存分配函数
    THPIInfo_pynew, /* tp_new */
    // 类型对象的构造函数，指定为 THPIInfo_pynew，用于创建新的对象
};

void THPDTypeInfo_init(PyObject* module) {
  // 准备 THPFInfoType 类型对象，如果失败则抛出 Python 错误异常
  if (PyType_Ready(&THPFInfoType) < 0) {
    throw python_error();
  }
  // 增加对 THPFInfoType 类型对象的引用计数
  Py_INCREF(&THPFInfoType);
  // 将 THPFInfoType 类型对象添加到指定模块中，名称为 "finfo"，如果失败则抛出 Python 错误异常
  if (PyModule_AddObject(module, "finfo", (PyObject*)&THPFInfoType) != 0) {
    throw python_error();
  }
  // 准备 THPIInfoType 类型对象，如果失败则抛出 Python 错误异常
  if (PyType_Ready(&THPIInfoType) < 0) {
    throw python_error();
  }
  // 增加对 THPIInfoType 类型对象的引用计数
  Py_INCREF(&THPIInfoType);
  // 将 THPIInfoType 类型对象添加到指定模块中，名称为 "iinfo"，如果失败则抛出 Python 错误异常
  if (PyModule_AddObject(module, "iinfo", (PyObject*)&THPIInfoType) != 0) {
    throw python_error();
  }
}
```