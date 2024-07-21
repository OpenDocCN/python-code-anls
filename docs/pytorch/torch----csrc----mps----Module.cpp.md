# `.\pytorch\torch\csrc\mps\Module.cpp`

```py
// 包含 ATen 库头文件
#include <ATen/ATen.h>
// 包含 C10 实用工具中的 CallOnce 头文件
#include <c10/util/CallOnce.h>
// 包含 Torch 的 Generator 头文件
#include <torch/csrc/Generator.h>
// 包含 Torch 的 THP 头文件
#include <torch/csrc/THP.h>
// 包含 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>
// 包含 Torch 的 Python 数字工具头文件
#include <torch/csrc/utils/python_numbers.h>
// 包含 Torch 的 Python 字符串工具头文件
#include <torch/csrc/utils/python_strings.h>

// 用于追踪不良分叉的 pthread.h 头文件（仅限非 Windows 平台）
#ifndef WIN32
#include <pthread.h>
#endif

// 定义 torch::mps 命名空间
namespace torch::mps {

// 匿名命名空间，内部静态变量和函数
namespace {
// 用于标记在 mps 初始化后派生的子进程
static bool in_bad_fork = false;

// 当 mps 已经初始化后，在派生的子进程中调用
static void forked_mps_child() {
  in_bad_fork = true;
}

// 在第一次调用 mps 之前应该调用
static void track_bad_mps_fork() {
#ifndef WIN32
  // 初始化静态的 call_once 旗标，调用 pthread_atfork 函数来设置 forked_mps_child 作为 fork 时的处理函数
  static c10::once_flag flag;
  c10::call_once(
      flag, [] { pthread_atfork(nullptr, nullptr, forked_mps_child); });
#endif
}
} // namespace

// MPSModule_isInBadFork 函数，返回 in_bad_fork 的 Python 对象表示
static PyObject* MPSModule_isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

// MPSModule_getDefaultMPSGenerator 函数，初始化默认的 MPS Generator
static PyObject* MPSModule_getDefaultMPSGenerator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  track_bad_mps_fork(); // 调用追踪不良分叉的函数
  // 调用 THPGenerator_initDefaultGenerator 初始化默认的 Generator
  return THPGenerator_initDefaultGenerator(
      at::detail::getMPSHooks().getDefaultMPSGenerator());
  END_HANDLE_TH_ERRORS
}

// MPSModule_isAvailable 函数，检查 MPS 是否可用
static PyObject* MPSModule_isAvailable(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  track_bad_mps_fork(); // 调用追踪不良分叉的函数
  // 检查当前是否有 MPS 可用，并返回相应的 Python 布尔值
  if (at::detail::getMPSHooks().hasMPS()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// MPSModule_isMacOSorNewer 函数，检查当前是否在 macOS 或更新版本上运行
static PyObject* MPSModule_isMacOSorNewer(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  size_t major = 0;
  size_t minor = 0;
  // 解析参数，获取 major 和 minor 版本号
  if (!PyArg_ParseTuple(args, "LL", &major, &minor)) {
    return nullptr;
  }
  // 检查当前系统版本是否符合要求，并返回相应的 Python 布尔值
  if (at::detail::getMPSHooks().isOnMacOSorNewer(major, minor)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// MPSModule_deviceSynchronize 函数，同步设备
static PyObject* MPSModule_deviceSynchronize(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 调用 MPS 钩子中的设备同步函数
  at::detail::getMPSHooks().deviceSynchronize();
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// MPSModule_emptyCache 函数，清空缓存
static PyObject* MPSModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 调用 MPS 钩子中的清空缓存函数
  at::detail::getMPSHooks().emptyCache();
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// MPSModule_setMemoryFraction 函数，设置内存分配比例
static PyObject* MPSModule_setMemoryFraction(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  // 检查参数是否为有效的 double 类型
  TORCH_CHECK(
      THPUtils_checkDouble(args), "invalid argument to setMemoryFraction()");
  // 解包 double 类型的参数
  double fraction = THPUtils_unpackDouble(args);
  // 调用 MPS 钩子中的设置内存分配比例函数
  at::detail::getMPSHooks().setMemoryFraction(fraction);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// MPSModule_currentAllocatedMemory 函数，获取当前分配的内存量
static PyObject* MPSModule_currentAllocatedMemory(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回当前分配的内存量，打包成 Python 的 UInt64 对象
  return THPUtils_packUInt64(
      at::detail::getMPSHooks().getCurrentAllocatedMemory());
  END_HANDLE_TH_ERRORS
}

// MPSModule_driverAllocatedMemory 函数（未完整列出，省略部分）
    // 定义一个函数 PyObject* noargs)，使用宏 HANDLE_TH_ERRORS 处理异常
    PyObject* noargs) {
      // 调用 at::detail::getMPSHooks().getDriverAllocatedMemory() 获取驱动程序分配的内存，
      // 并使用 THPUtils_packUInt64 封装为 Python 对象
      HANDLE_TH_ERRORS
      return THPUtils_packUInt64(
          at::detail::getMPSHooks().getDriverAllocatedMemory());
      // 结束异常处理
      END_HANDLE_TH_ERRORS
    }
static PyObject* MPSModule_recommendedMaxMemory(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 调用ATen库中的MPS钩子获取推荐的最大内存值，并将其打包为一个无符号64位整数对象返回
  return THPUtils_packUInt64(
      at::detail::getMPSHooks().getRecommendedMaxMemory());
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_profilerStartTrace(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  // 定义两个PyObject指针，用于存储模式和等待完成标志的Python对象
  PyObject* mode_string_o = nullptr;
  PyObject* wait_until_completed_string_o = nullptr;
  // 解析Python元组参数，获取模式和等待完成标志
  if (!PyArg_ParseTuple(
          args, "OO", &mode_string_o, &wait_until_completed_string_o)) {
    // 解析失败时返回空指针
    return nullptr;
  }
  // 将模式字符串解包成C++ std::string
  const std::string mode = THPUtils_unpackString(mode_string_o);
  // 将等待完成标志解包成布尔值
  const bool waitUntilCompleted =
      THPUtils_unpackBool(wait_until_completed_string_o);
  // 调用ATen库中的MPS钩子启动性能分析器的跟踪
  at::detail::getMPSHooks().profilerStartTrace(mode, waitUntilCompleted);
  // 返回Python None对象表示成功执行
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_profilerStopTrace(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 调用ATen库中的MPS钩子停止性能分析器的跟踪
  at::detail::getMPSHooks().profilerStopTrace();
  // 返回Python None对象表示成功执行
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_acquireEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // 解包Python参数，获取一个布尔值，指示是否启用计时
  const bool enable_timing = THPUtils_unpackBool(args);
  // 调用ATen库中的MPS钩子获取事件的ID，并将其打包为一个无符号32位整数对象返回
  return THPUtils_packUInt32(
      at::detail::getMPSHooks().acquireEvent(enable_timing));
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_releaseEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // 解包Python参数，获取事件的ID
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  // 调用ATen库中的MPS钩子释放指定ID的事件
  at::detail::getMPSHooks().releaseEvent(event_id);
  // 返回Python None对象表示成功执行
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_recordEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // 解包Python参数，获取事件的ID
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  // 调用ATen库中的MPS钩子记录指定ID的事件
  at::detail::getMPSHooks().recordEvent(event_id);
  // 返回Python None对象表示成功执行
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_waitForEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // 解包Python参数，获取事件的ID
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  // 调用ATen库中的MPS钩子等待指定ID的事件完成
  at::detail::getMPSHooks().waitForEvent(event_id);
  // 返回Python None对象表示成功执行
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_synchronizeEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // 解包Python参数，获取事件的ID
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  // 调用ATen库中的MPS钩子同步指定ID的事件
  at::detail::getMPSHooks().synchronizeEvent(event_id);
  // 返回Python None对象表示成功执行
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_queryEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // 解包Python参数，获取事件的ID
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  // 调用ATen库中的MPS钩子查询指定ID的事件，根据结果返回对应的Python布尔值对象
  if (at::detail::getMPSHooks().queryEvent(event_id)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_elapsedTimeOfEvents(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  // 定义两个PyObject指针，用于存储起始事件和结束事件的Python对象
  PyObject* start_event_o = nullptr;
  PyObject* end_event_o = nullptr;
  // 解析Python元组参数，获取起始和结束事件的Python对象
  if (!PyArg_ParseTuple(args, "OO", &start_event_o, &end_event_o)) {
    // 返回空指针（nullptr），表示函数提前结束并无返回值
    return nullptr;
  }
  // 从 Python 对象中解包出起始事件 ID，并转换为 uint32_t 类型
  const uint32_t start_event_id = THPUtils_unpackUInt32(start_event_o);
  // 从 Python 对象中解包出结束事件 ID，并转换为 uint32_t 类型
  const uint32_t end_event_id = THPUtils_unpackUInt32(end_event_o);
  // 调用 ATen 库中的函数，计算指定事件 ID 之间的经过时间，并将结果转换为 Python 的浮点数对象
  return PyFloat_FromDouble(at::detail::getMPSHooks().elapsedTimeOfEvents(
      start_event_id, end_event_id));
  // 处理 Torch 异常，并返回计算结果或异常对象
  END_HANDLE_TH_ERRORS
// 结束 _MPSModule_methods 数组的声明

// NOLINTNEXTLINE(*-c-arrays, *-global-variables)
// 静态声明 _MPSModule_methods 数组，包含多个 PyMethodDef 结构体初始化
static struct PyMethodDef _MPSModule_methods[] = {
    // 定义 _mps_deviceSynchronize 方法，指向 MPSModule_deviceSynchronize 函数，无参数
    {"_mps_deviceSynchronize",
     MPSModule_deviceSynchronize,
     METH_NOARGS,
     nullptr},
    // 定义 _mps_is_in_bad_fork 方法，指向 MPSModule_isInBadFork 函数，无参数
    {"_mps_is_in_bad_fork", MPSModule_isInBadFork, METH_NOARGS, nullptr},
    // 定义 _mps_is_available 方法，指向 MPSModule_isAvailable 函数，无参数
    {"_mps_is_available", MPSModule_isAvailable, METH_NOARGS, nullptr},
    // 定义 _mps_is_on_macos_or_newer 方法，指向 MPSModule_isMacOSorNewer 函数，可变参数
    {"_mps_is_on_macos_or_newer",
     MPSModule_isMacOSorNewer,
     METH_VARARGS,
     nullptr},
    // 定义 _mps_get_default_generator 方法，指向 MPSModule_getDefaultMPSGenerator 函数，无参数
    {"_mps_get_default_generator",
     MPSModule_getDefaultMPSGenerator,
     METH_NOARGS,
     nullptr},
    // 定义 _mps_emptyCache 方法，指向 MPSModule_emptyCache 函数，无参数
    {"_mps_emptyCache", MPSModule_emptyCache, METH_NOARGS, nullptr},
    // 定义 _mps_setMemoryFraction 方法，指向 MPSModule_setMemoryFraction 函数，接收一个参数
    {"_mps_setMemoryFraction", MPSModule_setMemoryFraction, METH_O, nullptr},
    // 定义 _mps_currentAllocatedMemory 方法，指向 MPSModule_currentAllocatedMemory 函数，无参数
    {"_mps_currentAllocatedMemory",
     MPSModule_currentAllocatedMemory,
     METH_NOARGS,
     nullptr},
    // 定义 _mps_driverAllocatedMemory 方法，指向 MPSModule_driverAllocatedMemory 函数，无参数
    {"_mps_driverAllocatedMemory",
     MPSModule_driverAllocatedMemory,
     METH_NOARGS,
     nullptr},
    // 定义 _mps_recommendedMaxMemory 方法，指向 MPSModule_recommendedMaxMemory 函数，无参数
    {"_mps_recommendedMaxMemory",
     MPSModule_recommendedMaxMemory,
     METH_NOARGS,
     nullptr},
    // 定义 _mps_profilerStartTrace 方法，指向 MPSModule_profilerStartTrace 函数，可变参数
    {"_mps_profilerStartTrace",
     MPSModule_profilerStartTrace,
     METH_VARARGS,
     nullptr},
    // 定义 _mps_profilerStopTrace 方法，指向 MPSModule_profilerStopTrace 函数，无参数
    {"_mps_profilerStopTrace",
     MPSModule_profilerStopTrace,
     METH_NOARGS,
     nullptr},
    // 定义 _mps_acquireEvent 方法，指向 MPSModule_acquireEvent 函数，接收一个参数
    {"_mps_acquireEvent", MPSModule_acquireEvent, METH_O, nullptr},
    // 定义 _mps_releaseEvent 方法，指向 MPSModule_releaseEvent 函数，接收一个参数
    {"_mps_releaseEvent", MPSModule_releaseEvent, METH_O, nullptr},
    // 定义 _mps_recordEvent 方法，指向 MPSModule_recordEvent 函数，接收一个参数
    {"_mps_recordEvent", MPSModule_recordEvent, METH_O, nullptr},
    // 定义 _mps_waitForEvent 方法，指向 MPSModule_waitForEvent 函数，接收一个参数
    {"_mps_waitForEvent", MPSModule_waitForEvent, METH_O, nullptr},
    // 定义 _mps_synchronizeEvent 方法，指向 MPSModule_synchronizeEvent 函数，接收一个参数
    {"_mps_synchronizeEvent", MPSModule_synchronizeEvent, METH_O, nullptr},
    // 定义 _mps_queryEvent 方法，指向 MPSModule_queryEvent 函数，接收一个参数
    {"_mps_queryEvent", MPSModule_queryEvent, METH_O, nullptr},
    // 定义 _mps_elapsedTimeOfEvents 方法，指向 MPSModule_elapsedTimeOfEvents 函数，可变参数
    {"_mps_elapsedTimeOfEvents",
     MPSModule_elapsedTimeOfEvents,
     METH_VARARGS,
     nullptr},
    // 数组的结束标志
    {nullptr}};

// 返回指向 _MPSModule_methods 数组的指针作为 python_functions 函数的返回值
PyMethodDef* python_functions() {
  return _MPSModule_methods;
}

// 结束 torch::mps 命名空间
} // namespace torch::mps
```