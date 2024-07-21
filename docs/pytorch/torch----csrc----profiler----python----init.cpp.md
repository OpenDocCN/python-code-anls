# `.\pytorch\torch\csrc\profiler\python\init.cpp`

```
// 包含 Torch Profiler 初始化相关的头文件
#include <torch/csrc/profiler/python/init.h>

// 包含 ATen 记录函数相关的头文件
#include <ATen/record_function.h>

// 包含 C10 Python 解释器实现的头文件
#include <c10/core/impl/PyInterpreter.h>

// 包含 C10 overloaded 实用工具的头文件
#include <c10/util/overloaded.h>

// 包含 Torch 动态类型相关的头文件
#include <torch/csrc/DynamicTypes.h>

// 包含 Torch 自动求导工具函数封装输出的头文件
#include <torch/csrc/autograd/utils/wrap_outputs.h>

// 包含 Torch JIT Python 绑定工具的头文件
#include <torch/csrc/jit/python/pybind_utils.h>

// 包含 Torch Profiler 数据收集相关的头文件
#include <torch/csrc/profiler/collection.h>

// 包含 Torch Profiler 合并回溯信息的头文件
#include <torch/csrc/profiler/python/combined_traceback.h>

// 包含 Torch Profiler 独立执行跟踪观察器的头文件
#include <torch/csrc/profiler/standalone/execution_trace_observer.h>

// 包含 Torch 工具类 Python 绑定的头文件
#include <torch/csrc/utils/pybind.h>

// 定义一个结构体 THPCapturedTraceback，表示捕获的 Python 回溯信息
struct THPCapturedTraceback {
  PyObject_HEAD
  std::shared_ptr<torch::CapturedTraceback> data;  // 持有 Torch CapturedTraceback 类的共享指针数据
};

// 定义 THPCapturedTraceback_traverse 函数，用于遍历 Python 对象的子对象
static int THPCapturedTraceback_traverse(
    PyObject* self,
    visitproc visit,
    void* arg) {
  return ((THPCapturedTraceback*)self)->data->traversePython((int (*)(void*, void*))visit, arg);
}

// 定义 THPCapturedTraceback_clear 函数，用于清理 Python 对象的数据
static int THPCapturedTraceback_clear(PyObject* self) {
  return ((THPCapturedTraceback*)self)->data->clearPython();
}

// 定义 THPCapturedTraceback_dealloc 函数，用于销毁 Python 对象并释放资源
static void THPCapturedTraceback_dealloc(PyObject* self_) {
  auto* self = (THPCapturedTraceback*)self_;
  PyObject_GC_UnTrack(self);
  self->data.~shared_ptr<torch::CapturedTraceback>();  // 手动调用析构函数释放 CapturedTraceback 的共享指针
  // 立即触发延迟释放的回溯帧，因为此时已经获得了 GIL
  torch::freeDeadCapturedTracebackFrames();
  PyObject_GC_Del(self);  // 从 Python 垃圾回收器中删除对象
}

// 定义 Python 类型对象 THPCapturedTracebackType，表示 THPCapturedTraceback 类的 Python 类型
PyTypeObject THPCapturedTracebackType = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0)
    "torch._C._profiler.CapturedTraceback",  // 类型名称
    sizeof(THPCapturedTraceback),  // 类型大小
    0,  // 类型每个元素的大小
    THPCapturedTraceback_dealloc,  // 类型销毁时的回调函数
    0,  // tp_vectorcall_offset
    nullptr,  // tp_getattr
    nullptr,  // tp_setattr
    nullptr,  // tp_reserved
    nullptr,  // tp_repr
    nullptr,  // tp_as_number
    nullptr,  // tp_as_sequence
    nullptr,  // tp_as_mapping
    nullptr,  // tp_hash
    nullptr,  // tp_call
    nullptr,  // tp_str
    nullptr,  // tp_getattro
    nullptr,  // tp_setattro
    nullptr,  // tp_as_buffer
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,  // 类型标志，支持 Python 垃圾回收
    nullptr,  // 类型文档字符串
    (traverseproc)THPCapturedTraceback_traverse,  // 类型遍历时的回调函数
    (inquiry)THPCapturedTraceback_clear,  // 类型清理时的回调函数
    nullptr,  // tp_richcompare
    0,  // tp_weaklistoffset
    nullptr,  // tp_iter
    nullptr,  // tp_iternext
    nullptr,  // tp_methods
    nullptr,  // tp_members
    nullptr,  // tp_getset
    nullptr,  // tp_base
    nullptr,  // tp_dict
    nullptr,  // tp_descr_get
    nullptr,  // tp_descr_set
    0,  // tp_dictoffset
    nullptr,  // tp_init
    nullptr,  // tp_alloc
    nullptr,  // tp_new
};

}  // namespace pybind11::detail
    # 检查 src 指针指向的对象是否是 THPCapturedTracebackType 类型
    if (Py_TYPE(src.ptr()) == &THPCapturedTracebackType) {
      # 如果是，则获取 THPCapturedTraceback 对象中的数据
      value = reinterpret_cast<THPCapturedTraceback*>(src.ptr())->data;
      # 返回 true 表示成功进行类型转换并获取数据
      return true;
    }
    # 如果不是 THPCapturedTracebackType 类型，则返回 false
    return false;
  }

  # 定义静态类型转换函数 cast
  static handle cast(
      # 参数为 std::shared_ptr<torch::CapturedTraceback> 类型的 src
      std::shared_ptr<torch::CapturedTraceback> src,
      # 返回值策略为忽略，父对象为默认
      return_value_policy /* policy */,
      handle /* parent */) {
    # 使用 PyObject_GC_New 分配 THPCapturedTraceback 对象的内存，并初始化
    auto* r = PyObject_GC_New(THPCapturedTraceback, &THPCapturedTracebackType);
    # 在新分配的 THPCapturedTraceback 对象中构造一个 std::shared_ptr
    new (&r->data) std::shared_ptr<torch::CapturedTraceback>(std::move(src));
    # 返回新创建的 THPCapturedTraceback 对象的 Python 句柄
    return py::handle((PyObject*)r);
  }
// 定义了一个匿名命名空间，内部包含一个结构体 RecordFunctionFast，用于实现一个较快的记录函数功能
namespace {

// 定义 RecordFunctionFast 结构体，用于封装记录函数的相关信息
struct RecordFunctionFast {
  PyObject_HEAD PyObject* name; // 记录函数名
  PyObject* input_values; // 记录输入参数的列表或元组
  PyObject* keyword_values; // 记录关键字参数的字典
  std::unique_ptr<at::RecordFunction> guard; // 记录函数的唯一指针保护
};

// RecordFunctionFast 类的构造函数，用于创建一个新的 RecordFunctionFast 实例
PyObject* RecordFunctionFast_new(
    PyTypeObject* subtype,
    PyObject* args,
    PyObject* kwargs) {
  RecordFunctionFast* self = (RecordFunctionFast*)subtype->tp_alloc(subtype, 0);
  if (self != nullptr) {
    self->name = nullptr;
    self->input_values = nullptr;
    self->keyword_values = nullptr;
    self->guard.reset();
  }
  return (PyObject*)self;
}

// RecordFunctionFast 类的初始化函数，用于初始化 RecordFunctionFast 实例的属性
int RecordFunctionFast_init(
    PyObject* selfGeneric,
    PyObject* args,
    PyObject* kwargs) {
  auto self = (RecordFunctionFast*)selfGeneric;
  // NOLINTNEXTLINE(*-c-arrays*)
  constexpr const char* kwlist[] = {
      "name", "input_values", "keyword_values", nullptr};
  PyObject* name = nullptr;
  PyObject* input_values = nullptr;
  PyObject* keyword_values = nullptr;
  // 解析传入的参数和关键字参数，以及它们的类型
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OO", // name 是必需的 PyObject，args 和 kwargs 是可选的 PyObject
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &name,
          &input_values,
          &keyword_values)) {
    return -1;
  }
  // 检查并设置记录函数名
  if (name) {
    TORCH_CHECK(
        THPUtils_checkString(name),
        "The name passed to RecordFunctionFast must be a string");
    Py_INCREF(name);
    self->name = name;
  }
  // 检查并设置输入参数列表或元组
  if (input_values) {
    TORCH_CHECK(
        PyList_Check(input_values) || PyTuple_Check(input_values),
        "input_values must be a list or tuple");
    Py_INCREF(input_values);
    self->input_values = input_values;
  }
  // 检查并设置关键字参数字典
  if (keyword_values) {
    TORCH_CHECK(PyDict_Check(keyword_values), "keyword_values must be dict");
    Py_INCREF(keyword_values);
    self->keyword_values = keyword_values;
  }
  return 0;
}
}

// 释放 RecordFunctionFast 对象的内存
void RecordFunctionFast_dealloc(PyObject* selfGeneric) {
  // 将通用指针转换为 RecordFunctionFast 指针
  auto self = (RecordFunctionFast*)selfGeneric;
  // 清除 Python 对象引用，防止内存泄漏
  Py_CLEAR(self->name);
  Py_CLEAR(self->input_values);
  Py_CLEAR(self->keyword_values);
  // 如果 guard 存在，重置它
  if (self->guard) {
    self->guard.reset();
  }
  // 调用类型对象的 tp_free 方法释放内存
  Py_TYPE(self)->tp_free(self);
}

// 进入 RecordFunctionFast 上下文
PyObject* RecordFunctionFast_enter(PyObject* selfGeneric, PyObject* unused) {
  HANDLE_TH_ERRORS
  // 检查当前是否启用了分析器状态
  if (torch::profiler::impl::ProfilerStateBase::get() != nullptr) {
    // 将通用指针转换为 RecordFunctionFast 指针
    auto self = (RecordFunctionFast*)selfGeneric;
    // 断言 guard 不存在，确保没有重复设置上下文
    TORCH_INTERNAL_ASSERT(
        !self->guard,
        "Trying to enter a new record_function_fast context but the guard is unexpectedly already set");
    // 创建 RecordFunction 对象并设置 guard
    self->guard =
        std::make_unique<at::RecordFunction>(at::RecordScope::FUNCTION);
    // 准备存储位置参数和关键字参数
    std::vector<at::IValue> args;
    std::unordered_map<std::string, at::IValue> kwargs;
    // 检查是否需要分析输入形状
    bool profiler_need_input = torch::autograd::profiler::profilerEnabled() &&
        torch::autograd::profiler::getProfilerConfig().report_input_shapes;
    // 如果存在输入值并且需要分析输入形状
    if (self->input_values != nullptr && profiler_need_input) {
      // 快速获取输入值序列
      THPObjectPtr input_fast(
          PySequence_Fast(self->input_values, "input must be a sequence"));
      // 获取输入值数组
      PyObject** input_items = PySequence_Fast_ITEMS(input_fast.get());
      // 遍历输入值数组
      for (int i = 0; i < PySequence_Fast_GET_SIZE(input_fast.get()); i++) {
        PyObject* item = input_items[i];
        // 尝试推断 Python 对象的 Torch 类型
        auto match = torch::jit::tryToInferType(item);
        // 如果推断成功，将值转换为 IValue 存入 args 中
        if (match.success()) {
          args.push_back(torch::jit::toIValue(item, match.type()));
        }
      }
    }

    // 如果存在关键字值并且需要分析输入形状
    if (self->keyword_values != nullptr && profiler_need_input) {
      // 遍历关键字值字典
      Py_ssize_t pos = 0;
      PyObject *key = nullptr, *value = nullptr;
      while (PyDict_Next(self->keyword_values, &pos, &key, &value)) {
        // 获取关键字和对应值的字符串表示
        std::string key_str = THPUtils_unpackString(key);
        at::IValue ivalue;
        // 如果值是字符串类型，直接转换为 IValue
        if (THPUtils_checkString(value)) {
          ivalue = at::IValue(THPUtils_unpackString(value));
        } else {
          // 否则，尝试推断 Python 对象的原始类型
          auto match = torch::jit::tryToInferPrimitiveType(value);
          // 如果推断成功，将值转换为 IValue 存入 kwargs 中
          if (match.success()) {
            ivalue = torch::jit::toIValue(value, match.type());
          } else {
            // 如果推断失败，发出警告并将值标记为 "NULL"
            TORCH_WARN("Unable to infer type of value for keyword: ", key_str);
            ivalue = at::IValue("NULL");
          }
        }
        // 将关键字和对应的 IValue 存入 kwargs 中
        kwargs[key_str] = ivalue;
      }
    }
    // 在 guard 上调用 before 方法，设置函数名、位置参数和关键字参数
    self->guard->before(THPUtils_unpackString(self->name), &args, &kwargs);
  }
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 退出 RecordFunctionFast 上下文
PyObject* RecordFunctionFast_exit(PyObject* selfGeneric, PyObject* unused) {
  HANDLE_TH_ERRORS
  // 检查当前是否启用了分析器状态
  if (torch::profiler::impl::ProfilerStateBase::get() != nullptr) {
    // 将通用指针转换为 RecordFunctionFast 指针
    auto self = (RecordFunctionFast*)selfGeneric;
    // 断言 guard 存在，确保在退出上下文时有设置 guard
    TORCH_INTERNAL_ASSERT(
        self->guard,
        "Trying to exit an active record_function_fast context but no guard is set");
    self->guard.reset();

# 调用自定义对象 `self` 的 `guard` 成员函数 `reset()`，用于重置某种保护机制或状态。


  }

# 关闭之前的代码块。


  Py_RETURN_NONE;

# 在 Python C API 中，返回一个 `None` 对象给调用者。


  END_HANDLE_TH_ERRORS

# 结束处理可能出现的 Torch 错误，这通常是一个宏，用于捕获和处理 Torch 引发的异常。
} // namespace
```