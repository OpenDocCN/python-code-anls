# `.\pytorch\torch\csrc\MemoryFormat.cpp`

```
// 包含 Torch 的 MemoryFormat 相关头文件

#include <torch/csrc/MemoryFormat.h>

// 包含 Torch 的异常处理相关头文件和实用工具

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_strings.h>

// 包含 C10 的 MemoryFormat 相关头文件

#include <c10/core/MemoryFormat.h>

// 包含 Python 的结构成员定义头文件

#include <structmember.h>

// 包含 C 标准库的字符串处理和内存操作相关头文件

#include <cstring>
#include <string>

// 定义一个函数 THPMemoryFormat_New，用于创建新的 THPMemoryFormat 对象
PyObject* THPMemoryFormat_New(
    at::MemoryFormat memory_format,         // 传入的 MemoryFormat 枚举值
    const std::string& name) {              // 传入的名称字符串引用
  auto type = (PyTypeObject*)&THPMemoryFormatType;  // 获取 THPMemoryFormatType 的 PyTypeObject 指针
  auto self = THPObjectPtr{type->tp_alloc(type, 0)}; // 分配一个新的 THPMemoryFormat 对象
  if (!self)                                // 如果分配失败
    throw python_error();                   // 抛出 Python 异常
  auto self_ = reinterpret_cast<THPMemoryFormat*>(self.get()); // 获取 THPMemoryFormat 对象指针
  self_->memory_format = memory_format;     // 设置对象的 memory_format 成员
  std::strncpy(self_->name, name.c_str(), MEMORY_FORMAT_NAME_LEN); // 将名称复制到对象的 name 数组中
  self_->name[MEMORY_FORMAT_NAME_LEN] = '\0'; // 确保字符串以 NULL 结尾
  return self.release();                    // 返回创建的对象
}

// 定义一个函数 THPMemoryFormat_repr，用于返回 THPMemoryFormat 对象的字符串表示
PyObject* THPMemoryFormat_repr(THPMemoryFormat* self) {
  return THPUtils_packString(self->name);   // 将对象的 name 字符串打包成 Python 字符串对象并返回
}

// 定义一个函数 THPMemoryFormat_reduce，用于序列化 THPMemoryFormat 对象
PyObject* THPMemoryFormat_reduce(PyObject* _self, PyObject* noargs) {
  auto* self = (THPMemoryFormat*)_self;     // 将输入参数转换为 THPMemoryFormat 对象指针
  return THPUtils_packString(self->name);   // 将对象的 name 字符串打包成 Python 字符串对象并返回
}

// 定义一个静态的 PyMethodDef 数组 THPMemoryFormat_methods，包含对象的方法信息
// NOLINTNEXTLINE 指定不检查特定的 lint 问题
static PyMethodDef THPMemoryFormat_methods[] = {
    {"__reduce__", THPMemoryFormat_reduce, METH_NOARGS, nullptr}, // 定义 __reduce__ 方法
    {nullptr} /* Sentinel */                 // 声明方法列表的结束
};

// 定义 PyTypeObject 结构 THPMemoryFormatType，表示 THPMemoryFormat 对象的类型信息
PyTypeObject THPMemoryFormatType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.memory_format", /* tp_name */ // 类型名称
    sizeof(THPMemoryFormat),         /* tp_basicsize */    // 对象基本大小
    0,                               /* tp_itemsize */     // 对象每个元素的大小
    nullptr,                         /* tp_dealloc */      // 释放对象的函数指针
    0,                               /* tp_vectorcall_offset */ // 向量调用偏移量
    nullptr,                         /* tp_getattr */      // 获取对象属性的函数指针
    nullptr,                         /* tp_setattr */      // 设置对象属性的函数指针
    nullptr,                         /* tp_reserved */     // 保留字段
    (reprfunc)THPMemoryFormat_repr,   /* tp_repr */         // 返回对象字符串表示的函数指针
    nullptr,                         /* tp_as_number */    // 数字对象协议
    nullptr,                         /* tp_as_sequence */  // 序列对象协议
    nullptr,                         /* tp_as_mapping */   // 映射对象协议
    nullptr,                         /* tp_hash */         // 哈希值计算函数指针
    nullptr,                         /* tp_call */         // 调用对象为函数的函数指针
    nullptr,                         /* tp_str */          // 返回对象的字符串表示的函数指针
    nullptr,                         /* tp_getattro */     // 获取对象属性的函数指针
    nullptr,                         /* tp_setattro */     // 设置对象属性的函数指针
    nullptr,                         /* tp_as_buffer */    // 缓冲区协议
    Py_TPFLAGS_DEFAULT,              /* tp_flags */        // 类型标志
    nullptr,                         /* tp_doc */          // 类型文档字符串
    nullptr,                         /* tp_traverse */     // 遍历对象的函数指针
    nullptr,                         /* tp_clear */        // 清除对象的函数指针
    nullptr,                         /* tp_richcompare */  // 对象比较的函数指针
    0,                               /* tp_weaklistoffset */ // 弱引用列表偏移量
    nullptr,                         /* tp_iter */         // 迭代协议
    nullptr,                         /* tp_iternext */     // 迭代的下一个元素的函数指针
    THPMemoryFormat_methods,         /* tp_methods */      // 方法列表
    nullptr,                         /* tp_members */      // 成员列表
    nullptr,                         /* tp_getset */       // 获取和设置属性的函数指针列表
    nullptr,                         /* tp_base */         // 基类类型指针
    nullptr,                         /* tp_dict */         // 类字典
    nullptr,                         /* tp_descr_get */    // 获取描述符的函数指针
    nullptr,                         /* tp_descr_set */    // 设置描述符的函数指针
    0,                               /* tp_dictoffset */   // 字典偏移量
    nullptr,                         /* tp_init */         // 初始化对象的函数指针
    nullptr,                         /* tp_alloc */        // 分配对象的函数指针
    nullptr                          /* tp_new */          // 创建新对象的函数指针
};

// 定义 THPMemoryFormat_init 函数，用于初始化 THPMemoryFormat 类型并将其添加到模块中
void THPMemoryFormat_init(PyObject* module) {
  if (PyType_Ready(&THPMemoryFormatType) < 0) {    // 准备 THPMemoryFormatType 类型，失败则抛出异常
    throw python_error();
  }
  Py_INCREF(&THPMemoryFormatType);                 // 增加类型的引用计数
  if (PyModule_AddObject(
          module, "memory_format", (PyObject*)&THPMemoryFormatType) != 0) { // 将类型添加到模块中，失败则抛出异常
    throw python_error();
  }
}
```