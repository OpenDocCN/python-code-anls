# `.\pytorch\torch\csrc\autograd\functions\init.cpp`

```py
// 包含 Python C API 头文件
#include <Python.h>
// 包含 C10 库中的工具函数和数据结构头文件
#include <c10/util/irange.h>
// 包含 Torch 自动微分模块中的梯度累积函数头文件
#include <torch/csrc/autograd/functions/accumulate_grad.h>
// 包含 Torch 自动微分模块中的基本操作函数头文件
#include <torch/csrc/autograd/functions/basic_ops.h>
// 包含 Torch 自动微分模块与 Python 绑定相关的头文件
#include <torch/csrc/autograd/functions/pybind.h>
// 包含 Torch 自动微分模块中的张量相关函数头文件
#include <torch/csrc/autograd/functions/tensor.h>
// 包含 Torch 自动微分模块中生成的 Python 函数头文件
#include <torch/csrc/autograd/generated/python_functions.h>
// 包含 Torch 自动微分模块中的 Python C++ 函数头文件
#include <torch/csrc/autograd/python_cpp_function.h>
// 包含 Torch 自动微分模块中的 Python 变量头文件
#include <torch/csrc/autograd/python_variable.h>
#ifdef USE_DISTRIBUTED
// 如果定义了 USE_DISTRIBUTED，则包含 Torch 分布式自动微分模块中的发送 RPC 反向传播函数头文件
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#endif
// 包含 Torch JIT 编译器中的 Python 追踪器头文件
#include <torch/csrc/jit/python/python_tracer.h>
// 包含 Torch 工具函数中与 Python 绑定相关的头文件
#include <torch/csrc/utils/pybind.h>
// 包含 Torch 工具函数中与 Python 数字相关的头文件
#include <torch/csrc/utils/python_numbers.h>
// 包含 Torch 工具函数中与 Python 字符串相关的头文件
#include <torch/csrc/utils/python_strings.h>

#include <utility>  // 包含标准库中的实用程序函数头文件

using namespace torch::autograd;  // 使用 Torch 自动微分命名空间

// 定义一个结构体 DelayedErrorCtor
struct DelayedErrorCtor {
  DelayedError* operator()(PyObject* args) {  // 延迟错误构造函数
    TORCH_CHECK(
        PyTuple_GET_SIZE(args) == 2,  // 检查参数元组大小是否为2
        "Requires two arguments, got ",  // 错误信息前缀
        PyTuple_GET_SIZE(args));  // 打印实际参数元组大小
    auto arg1 = PyTuple_GET_ITEM(args, 0);  // 获取第一个参数
    TORCH_CHECK(THPUtils_checkString(arg1), "argument 'msg' must be a string");  // 检查第一个参数是否为字符串
    std::string msg = THPUtils_unpackString(arg1);  // 解包第一个参数为字符串
    auto arg2 = PyTuple_GET_ITEM(args, 1);  // 获取第二个参数
    TORCH_CHECK(
        THPUtils_checkLong(arg2), "argument 'num_inputs' must be an int");  // 检查第二个参数是否为整数
    auto num_inputs = THPUtils_unpackLong(arg2);  // 解包第二个参数为整数
    return new DelayedError(std::move(msg), num_inputs);  // 返回 DelayedError 类的新实例
  }
};

// 定义一个结构体 UndefinedGradCtor
struct UndefinedGradCtor {
  UndefinedGrad* operator()(PyObject* args) {  // 未定义梯度构造函数
    TORCH_CHECK(
        PyTuple_GET_SIZE(args) == 0,  // 检查参数元组大小是否为0
        "Requires zero arguments, got ",  // 错误信息前缀
        PyTuple_GET_SIZE(args));  // 打印实际参数元组大小
    return new UndefinedGrad();  // 返回 UndefinedGrad 类的新实例
  }
};

// 定义一个结构体 NoCtor
struct NoCtor {
  Node* operator()(PyObject* args) {  // 无构造函数
    throw std::runtime_error("Cannot construct");  // 抛出运行时错误，无法构造
  }
};

// 定义一个模板函数 addClass
template <typename C, typename T>
static void addClass(
    PyObject* module,  // Python 模块对象
    PyTypeObject& type,  // Python 类型对象
    const char* name,  // 类型名称
    PyGetSetDef* function_properties = nullptr,  // 属性函数定义（默认为空）
    PyMethodDef* function_methods = nullptr) {  // 方法函数定义（默认为空）
  createForwardFunctionPyTypeObject<T>(
      type, name, function_properties, function_methods);  // 创建前向函数 Python 类型对象
  Py_INCREF(&type);  // 增加 Python 类型对象的引用计数
  PyModule_AddObject(module, name, (PyObject*)&type);  // 向 Python 模块中添加类型对象
  registerCppFunction(typeid(C), &type);  // 注册 C++ 函数
}

// 定义一个模板函数 getTupleAttr
template <
    typename T,
    typename ValueT,
    typename ParamsT,
    ValueT ParamsT::*ptr,
    typename ConvertArgT,
    PyObject* (*Convert)(ConvertArgT)>
PyObject* getTupleAttr(PyObject* obj, void* _unused) {  // 获取元组属性函数模板
  HANDLE_TH_ERRORS  // 处理 Torch 错误
  THPCppFunction* self = (THPCppFunction*)obj;  // 将 Python 对象转换为 THPCppFunction 指针
  auto& arr = ((T*)(self->cdata.get()))->*ptr;  // 获取数组引用
  auto num_elems = arr.size();  // 获取数组大小
  THPObjectPtr py_tuple(PyTuple_New(num_elems));  // 创建 Python 元组对象
  if (!py_tuple)  // 如果 Python 元组创建失败
    return nullptr;  // 返回空指针
  for (const auto i : c10::irange(num_elems)) {  // 遍历数组元素范围
    PyTuple_SET_ITEM(py_tuple.get(), i, Convert(arr[i]));  // 将数组元素转换并添加到 Python 元组中
  }
  return py_tuple.release();  // 返回 Python 元组并释放所有权
  END_HANDLE_TH_ERRORS  // 结束 Torch 错误处理
}

// 最后一部分模板函数的声明被省略，无法提供完整的注释
PyObject* getValueAttr(PyObject* obj, void* _unused) {
  HANDLE_TH_ERRORS
  // 将传入的 PyObject 转换为 THPCppFunction 对象
  THPCppFunction* self = (THPCppFunction*)obj;
  // 通过 self->cdata 获取 T 类型对象的指针，并访问其成员变量 *ptr
  auto& val = ((T*)(self->cdata.get()))->*ptr;
  // 将获取到的值转换为 PyObject 并返回
  return Convert(val);
  END_HANDLE_TH_ERRORS
}

static PyObject* accumulateGradVar(PyObject* _self, void* _unused) {
  // 将传入的 PyObject 转换为 THPCppFunction 对象
  THPCppFunction* self = (THPCppFunction*)_self;
  // 获取 self->cdata 指向的 AccumulateGrad 类型对象的指针
  auto grad_acc = (AccumulateGrad*)self->cdata.get();
  // 返回包装了 grad_acc->variable 的 THPVariable 对象
  return THPVariable_Wrap(grad_acc->variable);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static struct PyGetSetDef accumulate_grad_properties[] = {
    THP_FUNCTION_DEFAULT_PROPERTIES,
    // 默认的属性定义，用于 Python 的 getattr 和 setattr 操作
    {(char*)"variable", accumulateGradVar, nullptr, nullptr, nullptr},
    {nullptr}};

void THPAutograd_initFunctions() {
  // 创建名为 "torch._C._functions" 的新 Python 模块对象
  THPObjectPtr module(PyModule_New("torch._C._functions"));
  if (!module)
    throw python_error();

  static PyTypeObject AccumulateGradClass;
  // 在模块中注册 AccumulateGrad 类型，包含特定的属性
  addClass<AccumulateGrad, NoCtor>(
      module,
      AccumulateGradClass,
      "AccumulateGrad",
      accumulate_grad_properties);

  static PyTypeObject ErrorClass;
  // 在模块中注册 Error 类型，无构造函数
  addClass<Error, NoCtor>(module, ErrorClass, "Error");

  static PyTypeObject NotImplementedClass;
  // 在模块中注册 NotImplemented 类型，无构造函数
  addClass<NotImplemented, NoCtor>(
      module, NotImplementedClass, "NotImplemented");

  static PyTypeObject DelayedErrorClass;
  // 在模块中注册 DelayedError 类型，带有 DelayedErrorCtor 构造函数
  addClass<DelayedError, DelayedErrorCtor>(
      module, DelayedErrorClass, "DelayedError");

  static PyTypeObject UndefinedGradBackwardClass;
  // 在模块中注册 UndefinedGradBackward 类型，无构造函数
  addClass<UndefinedGradBackward, NoCtor>(
      module, UndefinedGradBackwardClass, "UndefinedGradBackward");

  static PyTypeObject UndefinedGradClass;
  // 在模块中注册 UndefinedGrad 类型，带有 UndefinedGradCtor 构造函数
  addClass<UndefinedGrad, UndefinedGradCtor>(
      module, UndefinedGradClass, "UndefinedGrad");

  static PyTypeObject CopyBackwardsClass;
  // 在模块中注册 CopyBackwards 类型，无构造函数
  addClass<CopyBackwards, NoCtor>(module, CopyBackwardsClass, "CopyBackwards");

#ifdef USE_DISTRIBUTED
  static PyTypeObject SendRpcBackwardClass;
  // 在模块中注册 SendRpcBackward 类型，无构造函数
  addClass<torch::distributed::autograd::SendRpcBackward, NoCtor>(
      module, SendRpcBackwardClass, "SendRpcBackward");
#endif

  static PyTypeObject CopySlicesClass;
  // 在模块中注册 CopySlices 类型，无构造函数
  addClass<CopySlices, NoCtor>(module, CopySlicesClass, "CopySlices");

  // 初始化自动生成的函数
  generated::initialize_autogenerated_functions(module);

  // 导入 "torch._C" 模块
  auto c_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!c_module)
    throw python_error();

  // 增加新创建的 module 到 c_module 中，并增加其引用计数
  Py_INCREF(module.get());
  if (PyModule_AddObject(c_module, "_functions", module) < 0) {
    Py_DECREF(module.get());
    throw python_error();
  }
}
```