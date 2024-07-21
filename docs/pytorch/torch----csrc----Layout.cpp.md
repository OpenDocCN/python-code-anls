# `.\pytorch\torch\csrc\Layout.cpp`

```
# 包含 Torch 库中 Layout 相关的头文件
#include <torch/csrc/Layout.h>

# 包含 Torch 异常处理相关的头文件
#include <torch/csrc/Exceptions.h>
# 包含 Torch 中对象指针工具的头文件
#include <torch/csrc/utils/object_ptr.h>
# 包含 Torch 中处理 Python 字符串的头文件
#include <torch/csrc/utils/python_strings.h>

# 包含 ATen 库中 Layout 相关的头文件
#include <ATen/Layout.h>

# 包含 Python C API 中处理结构体成员的头文件
#include <structmember.h>
# 包含 C 标准库中处理字符串相关操作的头文件
#include <cstring>
# 包含 C++ 标准库中处理字符串相关操作的头文件
#include <string>

# 创建并返回一个新的 THPLayout 对象，该对象包含给定的布局信息和名称
PyObject* THPLayout_New(at::Layout layout, const std::string& name) {
  # 获取 THPLayoutType 的类型对象指针
  auto type = (PyTypeObject*)&THPLayoutType;
  # 使用 tp_alloc 分配内存以创建 Python 对象
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  # 如果分配失败，则抛出 Python 异常
  if (!self)
    throw python_error();
  # 将 self 转换为 THPLayout 指针类型
  auto self_ = reinterpret_cast<THPLayout*>(self.get());
  # 设置 self_ 对象的布局信息
  self_->layout = layout;
  # 将名称 name 复制到 self_ 对象的 name 字段中，确保不超过 LAYOUT_NAME_LEN 长度
  std::strncpy(self_->name, name.c_str(), LAYOUT_NAME_LEN);
  # 确保 self_->name 以 null 结尾
  self_->name[LAYOUT_NAME_LEN] = '\0';
  # 返回创建的 THPLayout 对象
  return self.release();
}

# 返回 THPLayout 对象的字符串表示形式
PyObject* THPLayout_repr(THPLayout* self) {
  return THPUtils_packString(self->name);
}

# 定义 THPLayoutType 类型对象的结构
PyTypeObject THPLayoutType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.layout", /* tp_name */
    sizeof(THPLayout), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPLayout_repr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr, /* tp_new */
};

# 初始化 THPLayoutType 类型对象，并将其添加到指定的 Python 模块中
void THPLayout_init(PyObject* module) {
  # 如果类型对象初始化失败，则抛出 Python 异常
  if (PyType_Ready(&THPLayoutType) < 0) {
    throw python_error();
  }
  # 增加类型对象的引用计数，以确保不被垃圾回收
  Py_INCREF(&THPLayoutType);
  # 将 THPLayoutType 添加到指定的 Python 模块中
  if (PyModule_AddObject(module, "layout", (PyObject*)&THPLayoutType) != 0) {
    throw python_error();
  }
}
```