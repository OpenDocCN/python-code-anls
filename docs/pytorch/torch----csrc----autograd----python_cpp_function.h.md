# `.\pytorch\torch\csrc\autograd\python_cpp_function.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <torch/csrc/python_headers.h>
// 包含 Torch 的 Python 头文件

#include <memory>
// 包含标准库中的内存管理相关头文件

#include <typeinfo>
// 包含标准库中的类型信息头文件

#include <torch/csrc/Exceptions.h>
// 包含 Torch 中的异常处理相关头文件

#include <torch/csrc/autograd/function.h>
// 包含 Torch 中的自动求导功能相关头文件

#include <torch/csrc/utils/object_ptr.h>
// 包含 Torch 中的对象指针工具相关头文件

namespace torch::autograd {
// Torch 的自动求导命名空间

struct THPCppFunction {
  PyObject_HEAD std::shared_ptr<Node> cdata;
  // 定义结构体 THPCppFunction，包含一个 Python 对象头和一个 Node 的共享指针 cdata
};

template <typename Ctor>
PyObject* CppFunction_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  // 定义模板函数 CppFunction_pynew，用于创建新的 C++ 函数对象
  THPObjectPtr obj(type->tp_alloc(type, 0));
  // 分配 Python 对象内存空间
  if (!obj)
    return nullptr;
  THPCppFunction* f = (THPCppFunction*)obj.get();
  // 获取 THPCppFunction 结构体指针
  HANDLE_TH_ERRORS
  // 错误处理开始
  new (&f->cdata) std::shared_ptr<Node>(Ctor()(args));
  // 在 f->cdata 处构造一个新的 Node 共享指针，调用 Ctor 构造函数并传入 args
  END_HANDLE_TH_ERRORS
  // 错误处理结束
  if (!f->cdata) {
    return nullptr;
  }
  // 如果共享指针为空，则返回空指针
  return obj.release();
  // 返回创建的 Python 对象
}

#define THP_FUNCTION_DEFAULT_METHODS                                           \
  {(char*)"_register_hook_dict",                                               \
   THPCppFunction_register_hook_dict,                                          \
   METH_O,                                                                     \
   nullptr},                                                                   \
      {(char*)"register_hook", THPCppFunction_register_hook, METH_O, nullptr}, \
      {(char*)"register_prehook",                                              \
       THPCppFunction_register_prehook,                                        \
       METH_O,                                                                 \
       nullptr},                                                               \
      {(char*)"name", THPCppFunction_name, METH_NOARGS, nullptr},              \
      {(char*)"_sequence_nr",                                                  \
       THPCppFunction_sequence_nr,                                             \
       METH_NOARGS,                                                            \
       nullptr},                                                               \
  {                                                                            \
    (char*)"_set_sequence_nr", THPCppFunction_set_sequence_nr, METH_O, nullptr \
  }
// 定义 THP_FUNCTION_DEFAULT_METHODS 宏，包含函数名及其对应的 C++ 函数指针、方法标志和可选的文档字符串
# 定义 THP_FUNCTION_DEFAULT_PROPERTIES 宏，包含了函数默认属性的列表
#define THP_FUNCTION_DEFAULT_PROPERTIES                                   \
  {(char*)"next_functions",                                               \  # 属性：下一个函数列表，指向下一个函数的指针
   THPCppFunction_next_functions,                                         \  # 指向获取下一个函数列表的函数指针
   nullptr,                                                               \  # 保留字段：未使用
   nullptr,                                                               \  # 保留字段：未使用
   nullptr},                                                              \  # 保留字段：未使用
      {(char*)"requires_grad",                                            \  # 属性：是否需要梯度
       THPCppFunction_requires_grad,                                      \  # 指向获取是否需要梯度的函数指针
       nullptr,                                                           \  # 保留字段：未使用
       nullptr,                                                           \  # 保留字段：未使用
       nullptr},                                                          \  # 保留字段：未使用
  {                                                                       \  # 属性：元数据
    (char*)"metadata", THPCppFunction_metadata, nullptr, nullptr, nullptr \  # 元数据属性，指向获取元数据的函数指针
  }

# 声明获取下一个函数列表的函数的原型
PyObject* THPCppFunction_next_functions(PyObject* self, void* _unused);

# 声明获取元数据的函数的原型
PyObject* THPCppFunction_metadata(PyObject* self, void* _unused);

# 声明获取是否需要梯度的函数的原型
PyObject* THPCppFunction_requires_grad(PyObject* self, void* _unused);

# 声明注册钩子字典的函数的原型
PyObject* THPCppFunction_register_hook_dict(PyObject* self, PyObject* _var);

# 声明注册钩子的函数的原型
PyObject* THPCppFunction_register_hook(PyObject* self, PyObject* hook);

# 声明注册预钩子的函数的原型
PyObject* THPCppFunction_register_prehook(PyObject* self, PyObject* hook);

# 声明获取函数名称的函数的原型
PyObject* THPCppFunction_name(PyObject* self, PyObject* noargs);

# 声明获取序列号的函数的原型
PyObject* THPCppFunction_sequence_nr(PyObject* self, PyObject* noargs);

# 初始化函数类型对象并返回其指针
PyTypeObject* _initFunctionPyTypeObject(
    PyTypeObject& type,
    const char* name,
    PyGetSetDef* function_properties,
    PyMethodDef* function_methods);

# 注册函数钩子
PyObject* registerFunctionHook(Node& fn, PyObject* hook);

# 注册函数预钩子
PyObject* registerFunctionPreHook(Node& fn, PyObject* hook);

# 创建前向函数的 Python 类型对象的模板
template <typename Ctor>
PyTypeObject* createForwardFunctionPyTypeObject(
    PyTypeObject& type,
    const char* name,
    PyGetSetDef* function_properties = nullptr,
    PyMethodDef* function_methods = nullptr) {
  type.tp_new = &CppFunction_pynew<Ctor>;  // 设置类型对象的创建函数
  return _initFunctionPyTypeObject(
      type, name, function_properties, function_methods);  // 初始化函数类型对象并返回其指针
}

# 注册 C++ 函数类型
void registerCppFunction(const std::type_info& type, PyTypeObject* pytype);

# 将函数转换为 Python 对象
PyObject* functionToPyObject(const std::shared_ptr<Node>& cdata);

# 检查对象是否为 THP C++ 函数类型
bool THPCppFunction_Check(PyObject* obj);

} // namespace torch::autograd  # 命名空间 torch::autograd 的结束
```