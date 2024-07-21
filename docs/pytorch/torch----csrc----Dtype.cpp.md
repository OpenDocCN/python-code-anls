# `.\pytorch\torch\csrc\Dtype.cpp`

```py
#include <torch/csrc/Dtype.h>

#include <c10/core/ScalarType.h>
#include <structmember.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/utils/tensor_types.h>
#include <cstring>

// 创建并返回一个新的 THPDtype 对象，包含标量类型和名称
PyObject* THPDtype_New(at::ScalarType scalar_type, const std::string& name) {
  HANDLE_TH_ERRORS
  // 确保名称长度不超过 DTYPE_NAME_LEN
  AT_ASSERT(name.length() < DTYPE_NAME_LEN);
  auto type = (PyTypeObject*)&THPDtypeType;
  // 分配一个新的 THPDtype 对象
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPDtype*>(self.get());
  // 设置对象的标量类型和名称
  self_->scalar_type = scalar_type;
  std::strncpy(self_->name, name.c_str(), DTYPE_NAME_LEN);
  return self.release();
  END_HANDLE_TH_ERRORS
}

// 检查 THPDtype 对象是否为浮点类型，并返回对应的 Python 布尔值对象
PyObject* THPDtype_is_floating_point(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::isFloatingType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// 返回 THPDtype 对象的标量类型的字节大小
PyObject* THPDtype_itemsize(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 获取标量类型对应的元数据，并返回其字节大小的 Python 整数对象
  return THPUtils_packUInt64(
      scalarTypeToTypeMeta(self->scalar_type).itemsize());
  END_HANDLE_TH_ERRORS
}

// 检查 THPDtype 对象是否为复数类型，并返回对应的 Python 布尔值对象
PyObject* THPDtype_is_complex(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::isComplexType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// 检查 THPDtype 对象的标量类型是否为有符号类型，并返回对应的 Python 布尔值对象
PyObject* THPDtype_is_signed(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::isSignedType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// 将 THPDtype 对象序列化为其名称字符串，并返回对应的 Python 字符串对象
PyObject* THPDtype_reduce(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回 THPDtype 对象的名称字符串
  auto self = (THPDtype*)_self;
  return THPUtils_packString(self->name);
  END_HANDLE_TH_ERRORS
}

// 将 THPDtype 对象转换为实部类型的 THPDtype 对象，并返回对应的 Python 对象
PyObject* THPDtype_to_real(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = (THPDtype*)_self;
  auto scalar_type = self->scalar_type;
  // 如果不是浮点类型，则将其转换为实部类型
  if (!at::isFloatingType(self->scalar_type)) {
    scalar_type = at::toRealValueType(self->scalar_type);
  }
  return Py_NewRef(torch::getTHPDtype(scalar_type));
  END_HANDLE_TH_ERRORS
}

// 将 THPDtype 对象转换为复数类型的 THPDtype 对象，并返回对应的 Python 对象
PyObject* THPDtype_to_complex(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = (THPDtype*)_self;
  auto scalar_type = self->scalar_type;
  // 如果不是复数类型，则将其转换为复数类型
  if (!at::isComplexType(self->scalar_type)) {
    scalar_type = at::toComplexType(self->scalar_type);
  }
  return Py_NewRef(torch::getTHPDtype(scalar_type));
  END_HANDLE_TH_ERRORS
}

// 定义 getter 类型，用于获取属性的回调函数指针
typedef PyObject* (*getter)(PyObject*, void*);

// 定义 THPDtype 对象的属性描述结构体，包含属性名和获取函数的映射
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static struct PyGetSetDef THPDtype_properties[] = {
    {"is_floating_point",     // 键为字符串 "is_floating_point"
     (getter)THPDtype_is_floating_point,  // 值为 THPDtype_is_floating_point 函数的 getter
     nullptr,                 // 未指定 setter
     nullptr,                 // 未指定 deleter
     nullptr},                // 结束符，表示此项结束

    {"is_complex",            // 键为字符串 "is_complex"
     (getter)THPDtype_is_complex,  // 值为 THPDtype_is_complex 函数的 getter
     nullptr,                 // 未指定 setter
     nullptr,                 // 未指定 deleter
     nullptr},                // 结束符，表示此项结束

    {"is_signed",             // 键为字符串 "is_signed"
     (getter)THPDtype_is_signed,  // 值为 THPDtype_is_signed 函数的 getter
     nullptr,                 // 未指定 setter
     nullptr,                 // 未指定 deleter
     nullptr},                // 结束符，表示此项结束

    {"itemsize",              // 键为字符串 "itemsize"
     (getter)THPDtype_itemsize,  // 值为 THPDtype_itemsize 函数的 getter
     nullptr,                 // 未指定 setter
     nullptr,                 // 未指定 deleter
     nullptr},                // 结束符，表示此项结束

    {nullptr}                 // 最后一项，表示字典列表结束的标志
};
// 定义静态数组 THPDtype_methods，包含一系列 PyMethodDef 结构体，每个结构体表示一个方法及其相关信息
static PyMethodDef THPDtype_methods[] = {
    {"__reduce__", THPDtype_reduce, METH_NOARGS, nullptr},  // 方法名为 "__reduce__"，对应的 C 函数为 THPDtype_reduce，不接受参数，无额外信息
    {"to_real", THPDtype_to_real, METH_NOARGS, nullptr},    // 方法名为 "to_real"，对应的 C 函数为 THPDtype_to_real，不接受参数，无额外信息
    {"to_complex", THPDtype_to_complex, METH_NOARGS, nullptr},  // 方法名为 "to_complex"，对应的 C 函数为 THPDtype_to_complex，不接受参数，无额外信息
    {nullptr} /* Sentinel */  // 结束标志，PyMethodDef 数组以 nullptr 结束
};

// 定义函数 THPDtype_repr，接受一个指向 THPDtype 结构体的指针 self，返回一个 PyObject* 类型的对象
PyObject* THPDtype_repr(THPDtype* self) {
  return THPUtils_packString(std::string("torch.") + self->name);  // 返回以 "torch." 加上 self->name 构成的字符串的 PyObject*
}

// 定义 PyTypeObject 结构体 THPDtypeType，表示一个 Python 类型对象
PyTypeObject THPDtypeType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.dtype", /* tp_name */  // 对象的名称为 "torch.dtype"
    sizeof(THPDtype), /* tp_basicsize */  // 对象的基本大小为 sizeof(THPDtype)
    0, /* tp_itemsize */  // 对象的每个元素的大小为 0（不适用于变长对象）
    nullptr, /* tp_dealloc */  // 对象的析构函数为 nullptr（无特定的析构函数）
    0, /* tp_vectorcall_offset */  // 向量调用的偏移量为 0（不使用向量调用）
    nullptr, /* tp_getattr */  // 获取属性的函数为 nullptr
    nullptr, /* tp_setattr */  // 设置属性的函数为 nullptr
    nullptr, /* tp_reserved */  // 保留字段为 nullptr
    (reprfunc)THPDtype_repr, /* tp_repr */  // repr 方法为 THPDtype_repr 函数
    nullptr, /* tp_as_number */  // 数字类型方法为 nullptr
    nullptr, /* tp_as_sequence */  // 序列类型方法为 nullptr
    nullptr, /* tp_as_mapping */  // 映射类型方法为 nullptr
    nullptr, /* tp_hash  */  // 哈希方法为 nullptr
    nullptr, /* tp_call */  // 调用方法为 nullptr
    nullptr, /* tp_str */  // 字符串表示方法为 nullptr
    nullptr, /* tp_getattro */  // 获取属性对象的属性方法为 nullptr
    nullptr, /* tp_setattro */  // 设置属性对象的属性方法为 nullptr
    nullptr, /* tp_as_buffer */  // 缓冲区方法为 nullptr
    Py_TPFLAGS_DEFAULT, /* tp_flags */  // 默认的标志位
    nullptr, /* tp_doc */  // 文档字符串为 nullptr
    nullptr, /* tp_traverse */  // 遍历对象的方法为 nullptr
    nullptr, /* tp_clear */  // 清理对象的方法为 nullptr
    nullptr, /* tp_richcompare */  // 比较对象的方法为 nullptr
    0, /* tp_weaklistoffset */  // 弱引用列表的偏移量为 0
    nullptr, /* tp_iter */  // 迭代方法为 nullptr
    nullptr, /* tp_iternext */  // 迭代下一个方法为 nullptr
    THPDtype_methods, /* tp_methods */  // 方法集合为 THPDtype_methods 数组
    nullptr, /* tp_members */  // 成员变量为 nullptr
    THPDtype_properties, /* tp_getset */  // 属性获取和设置为 THPDtype_properties
    nullptr, /* tp_base */  // 基类为 nullptr
    nullptr, /* tp_dict */  // 字典为 nullptr
    nullptr, /* tp_descr_get */  // 获取描述符方法为 nullptr
    nullptr, /* tp_descr_set */  // 设置描述符方法为 nullptr
    0, /* tp_dictoffset */  // 字典的偏移量为 0
    nullptr, /* tp_init */  // 初始化方法为 nullptr
    nullptr, /* tp_alloc */  // 分配方法为 nullptr
    nullptr, /* tp_new */  // 新建方法为 nullptr
};

// 定义函数 THPDtype_init，接受一个 PyObject* 类型的 module 参数，用于初始化 dtype 类型
void THPDtype_init(PyObject* module) {
  // 设置一个 __dict__，其中 '__module__' = 'torch'。这意味着 '__module__' 的值将被实例继承
  // （例如 `torch.float32.__module__ == "torch"`）。这将防止 Pickle 在尝试 pickle dtype 实例时
  // 需要搜索所有的 sys.modules。
  //
  // 我们必须在 C++ 中执行此操作，因为扩展类型无法从 Python 代码中修改。
  //
  // 参见 https://github.com/pytorch/pytorch/issues/65077
  TORCH_INTERNAL_ASSERT(THPDtypeType.tp_dict == nullptr);  // 断言确保 THPDtypeType 的字典为空
  auto dict = THPObjectPtr(PyDict_New());  // 创建一个新的 Python 字典对象
  if (!dict)
    throw python_error();  // 如果创建字典失败，则抛出异常
  auto torch = THPUtils_packString("torch");  // 将字符串 "torch" 打包成 PyObject*
  if (!torch)
    throw python_error();  // 如果打包字符串失败，则抛出异常
  if (PyDict_SetItemString(dict, "__module__", torch) < 0) {  // 将 "__module__" = torch 添加到字典中
    throw python_error();  // 如果设置失败，则抛出异常
  }
  THPDtypeType.tp_dict = dict.release();  // 将创建的字典赋值给 THPDtypeType 的 tp_dict 字段，并释放所有权

  if (PyType_Ready(&THPDtypeType) < 0) {  // 准备类型对象 THPDtypeType
    throw python_error();  // 如果准备失败，则抛出异常
  }
  Py_INCREF(&THPDtypeType);  // 增加类型对象的引用计数
  if (PyModule_AddObject(module, "dtype", (PyObject*)&THPDtypeType) != 0) {  // 将类型对象添加到模块中
    throw python_error();  // 如果添加失败，则抛出异常
  }
}
```