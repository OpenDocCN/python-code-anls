# `.\pytorch\torch\csrc\QScheme.cpp`

```py
#include <torch/csrc/QScheme.h>  // 包含 Torch 框架中的 QScheme 头文件

#include <torch/csrc/Exceptions.h>  // 包含 Torch 框架中的异常处理头文件
#include <torch/csrc/utils/object_ptr.h>  // 包含 Torch 框架中的智能指针对象头文件
#include <torch/csrc/utils/python_strings.h>  // 包含 Torch 框架中处理 Python 字符串的头文件

#include <c10/core/QScheme.h>  // 包含 C10 核心库中的 QScheme 头文件

#include <structmember.h>  // 包含 Python C API 中的结构成员定义头文件
#include <cstring>  // 包含 C 标准字符串操作的头文件
#include <string>  // 包含 C++ 标准字符串处理的头文件

// 创建一个新的 THPQScheme 对象，并初始化其属性 qscheme 和 name
PyObject* THPQScheme_New(at::QScheme qscheme, const std::string& name) {
  auto type = (PyTypeObject*)&THPQSchemeType;  // 获取 THPQSchemeType 的类型对象指针
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};  // 分配并初始化一个 THPQScheme 对象
  if (!self)
    throw python_error();  // 如果分配失败，则抛出 Python 异常
  auto self_ = reinterpret_cast<THPQScheme*>(self.get());  // 将 self 转换为 THPQScheme 指针
  self_->qscheme = qscheme;  // 设置 self_ 指向对象的 qscheme 属性
  std::strncpy(self_->name, name.c_str(), QSCHEME_NAME_LEN);  // 将 name 字符串复制到 self_->name 中
  self_->name[QSCHEME_NAME_LEN] = '\0';  // 确保 self_->name 以 null 结尾
  return self.release();  // 返回创建的 THPQScheme 对象的 PyObject 指针
}

// 减少 THPQScheme 对象的表示形式
PyObject* THPQScheme_reduce(PyObject* _self, PyObject* noargs) {
  auto self = (THPQScheme*)_self;  // 将 PyObject 指针转换为 THPQScheme 指针
  return THPUtils_packString(self->name);  // 返回包含 self->name 字符串的 Python 对象
}

// 定义 THPQScheme 对象的方法列表
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static PyMethodDef THPQScheme_methods[] = {
    {"__reduce__", THPQScheme_reduce, METH_NOARGS, nullptr},  // 定义 __reduce__ 方法
    {nullptr} /* Sentinel */  // 结束方法列表的标志
};

// 返回 THPQScheme 对象的字符串表示形式
PyObject* THPQScheme_repr(THPQScheme* self) {
  std::string name = self->name;  // 将 self->name 转换为 std::string
  return THPUtils_packString("torch." + name);  // 返回包含 "torch." + name 字符串的 Python 对象
}

// 定义 THPQSchemeType 类型对象
PyTypeObject THPQSchemeType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.qscheme", /* tp_name */  // 类型对象的名称
    sizeof(THPQScheme), /* tp_basicsize */  // 类型对象的基本大小
    0, /* tp_itemsize */  // 类型对象的每个项的大小
    nullptr, /* tp_dealloc */  // 类型对象的内存释放函数
    0, /* tp_vectorcall_offset */  // 类型对象的矢量调用偏移量
    nullptr, /* tp_getattr */  // 类型对象的获取属性函数
    nullptr, /* tp_setattr */  // 类型对象的设置属性函数
    nullptr, /* tp_reserved */  // 保留字段
    (reprfunc)THPQScheme_repr, /* tp_repr */  // 类型对象的表示函数
    nullptr, /* tp_as_number */  // 类型对象的数值计算函数
    nullptr, /* tp_as_sequence */  // 类型对象的序列操作函数
    nullptr, /* tp_as_mapping */  // 类型对象的映射操作函数
    nullptr, /* tp_hash  */  // 类型对象的哈希函数
    nullptr, /* tp_call */  // 类型对象的调用函数
    nullptr, /* tp_str */  // 类型对象的字符串表示函数
    nullptr, /* tp_getattro */  // 类型对象的获取属性函数
    nullptr, /* tp_setattro */  // 类型对象的设置属性函数
    nullptr, /* tp_as_buffer */  // 类型对象的缓冲区操作函数
    Py_TPFLAGS_DEFAULT, /* tp_flags */  // 类型对象的标志位
    nullptr, /* tp_doc */  // 类型对象的文档字符串
    nullptr, /* tp_traverse */  // 类型对象的遍历函数
    nullptr, /* tp_clear */  // 类型对象的清理函数
    nullptr, /* tp_richcompare */  // 类型对象的富比较函数
    0, /* tp_weaklistoffset */  // 弱引用列表的偏移量
    nullptr, /* tp_iter */  // 迭代器函数
    nullptr, /* tp_iternext */  // 迭代器下一个函数
    THPQScheme_methods, /* tp_methods */  // 类型对象的方法列表
    nullptr, /* tp_members */  // 类型对象的成员列表
    nullptr, /* tp_getset */  // 类型对象的获取/设置函数列表
    nullptr, /* tp_base */  // 类型对象的基类
    nullptr, /* tp_dict */  // 类型对象的字典
    nullptr, /* tp_descr_get */  // 获取描述符的函数
    nullptr, /* tp_descr_set */  // 设置描述符的函数
    0, /* tp_dictoffset */  // 字典偏移量
    nullptr, /* tp_init */  // 初始化函数
    nullptr, /* tp_alloc */  // 分配函数
    nullptr, /* tp_new */  // 新建函数
};

// 初始化 THPQSchemeType 类型对象
void THPQScheme_init(PyObject* module) {
  if (PyType_Ready(&THPQSchemeType) < 0) {  // 如果初始化类型对象失败
    throw python_error();  // 抛出 Python 异常
  }
  Py_INCREF(&THPQSchemeType);  // 增加 THPQSchemeType 的引用计数
  if (PyModule_AddObject(module, "qscheme", (PyObject*)&THPQSchemeType) != 0) {  // 将 THPQSchemeType 添加到模块中
    throw python_error();  // 如果添加失败，则抛出 Python 异常
  }
}
```