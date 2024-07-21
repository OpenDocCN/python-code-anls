# `.\pytorch\torch\csrc\multiprocessing\init.cpp`

```
// 包含C++头文件，用于线程命名和异常处理
#include <c10/util/thread_name.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

// 包含标准异常处理类
#include <stdexcept>

// 根据定义操作系统类型，包含对应的系统调用头文件
#if defined(__linux__)
#include <sys/prctl.h>
#endif

// 定义宏，用于检查系统调用的返回值，若小于0则抛出系统错误异常
#define SYSASSERT(rv, ...)                                                 \
  if ((rv) < 0) {                                                          \
    throw std::system_error(errno, std::system_category(), ##__VA_ARGS__); \
  }

// torch::multiprocessing 命名空间
namespace torch::multiprocessing {

// 私有命名空间，用于封装内部函数和变量
namespace {

// 初始化 multiprocessing 模块的回调函数
PyObject* multiprocessing_init(PyObject* _unused, PyObject* noargs) {
  // 导入 torch.multiprocessing 模块
  auto multiprocessing_module =
      THPObjectPtr(PyImport_ImportModule("torch.multiprocessing"));
  // 检查模块导入是否成功
  if (!multiprocessing_module) {
    throw python_error();
  }

  // 将 PyObject 转换为 py::module 类型
  auto module = py::handle(multiprocessing_module).cast<py::module>();

  // 定义 _prctl_pr_set_pdeathsig 方法，用于设置进程死亡信号
  module.def("_prctl_pr_set_pdeathsig", [](int signal) {
#if defined(__linux__)
    // 调用 prctl 系统调用设置进程死亡信号
    auto rv = prctl(PR_SET_PDEATHSIG, signal);
    // 检查系统调用返回值，若小于0则抛出异常
    SYSASSERT(rv, "prctl");
#endif
  });

  // 返回 True 表示成功
  Py_RETURN_TRUE;
}

// 设置线程名称的回调函数
PyObject* set_thread_name(PyObject* _unused, PyObject* arg) {
  // 检查参数是否为有效字符串对象
  TORCH_CHECK(THPUtils_checkString(arg), "invalid argument to setDevice");

  // 解包字符串参数
  auto name = THPUtils_unpackString(arg);
  // 调用 C++ 库函数设置当前线程名称
  c10::setThreadName(name);

  // 返回 True 表示成功
  Py_RETURN_TRUE;
}

// 获取线程名称的回调函数
PyObject* get_thread_name(PyObject* _unused, PyObject* noargs) {
  // 返回当前线程的名称字符串
  return THPUtils_packString(c10::getThreadName());
}

} // namespace

// 定义导出的 Python 方法列表
static PyMethodDef methods[] = {
    {
        "_multiprocessing_init",   // 方法名称
        multiprocessing_init,      // 方法实现函数
        METH_NOARGS,               // 方法接受的参数类型
        nullptr,                   // 方法的文档字符串，此处为空
    },
    {
        "_set_thread_name",        // 方法名称
        set_thread_name,           // 方法实现函数
        METH_O,                    // 方法接受的参数类型
        nullptr,                   // 方法的文档字符串，此处为空
    },
    {
        "_get_thread_name",        // 方法名称
        get_thread_name,           // 方法实现函数
        METH_NOARGS,               // 方法接受的参数类型
        nullptr,                   // 方法的文档字符串，此处为空
    },
    {nullptr, nullptr, 0, nullptr}, // 方法列表结束标志
};

// 返回 Python 方法定义的函数
PyMethodDef* python_functions() {
  return methods;
}

} // namespace torch::multiprocessing
```