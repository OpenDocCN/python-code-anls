# `.\pytorch\torch\csrc\cuda\Stream.cpp`

```py
// 包含必要的头文件来定义 Python 绑定
#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/cuda/Module.h>
#include <torch/csrc/cuda/Stream.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>

// 包含 CUDA 相关的头文件
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <structmember.h>

// 全局变量，用于存储 THCPStream 类的 Python 类型对象
PyObject* THCPStreamClass = nullptr;

// 定义 THCPStream 类的构造函数
static PyObject* THCPStream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  // 获取当前 CUDA 设备索引
  const auto current_device = c10::cuda::current_device();

  // 初始化参数的默认值
  int priority = 0;
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;
  uint64_t stream_ptr = 0;

  // 定义参数列表的字符串数组
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "priority",
      "stream_id",
      "device_index",
      "device_type",
      "stream_ptr",
      nullptr};

  // 解析 Python 函数参数，并赋值给对应的变量
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|iLLLK",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &priority,
          &stream_id,
          &device_index,
          &device_type,
          &stream_ptr)) {
    return nullptr;
  }

  // 分配内存以创建新的 THCPStream 对象
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  // 检查是否为外部流设置了优先级
  if (stream_ptr) {
    TORCH_CHECK(
        priority == 0, "Priority was explicitly set for a external stream")
  }

  // 根据参数创建 CUDA 流对象
  at::cuda::CUDAStream stream = (stream_id || device_index || device_type)
      ? at::cuda::CUDAStream::unpack3(
            stream_id,
            static_cast<c10::DeviceIndex>(device_index),
            static_cast<c10::DeviceType>(device_type))
      : stream_ptr ? at::cuda::getStreamFromExternal(
                         // NOLINTNEXTLINE(performance-no-int-to-ptr)
                         reinterpret_cast<cudaStream_t>(stream_ptr),
                         current_device)
                   : at::cuda::getStreamFromPool(priority);

  // 获取 self 指针，并设置其属性
  THCPStream* self = (THCPStream*)ptr.get();
  self->stream_id = static_cast<int64_t>(stream.id());
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  new (&self->cuda_stream) at::cuda::CUDAStream(stream);

  // 返回创建的 Python 对象
  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

// 定义 THCPStream 对象的析构函数
static void THCPStream_dealloc(THCPStream* self) {
  self->cuda_stream.~CUDAStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// 获取 THCPStream 对象的 CUDA 流
static PyObject* THCPStream_get_cuda_stream(THCPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->cuda_stream.stream());
  END_HANDLE_TH_ERRORS
}

// 获取 THCPStream 对象的优先级
static PyObject* THCPStream_get_priority(THCPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(self->cuda_stream.priority());
  END_HANDLE_TH_ERRORS
}

// 定义 THCPStream 优先级范围的函数
static PyObject* THCPStream_priority_range(
    PyObject* _unused,
    // 定义一个函数 PyObject* noargs)，该函数不接受任何参数
    HANDLE_TH_ERRORS
    // 调用宏 HANDLE_TH_ERRORS，用于处理 Torch 错误并设置错误处理的上下文

    // 调用 at::cuda::CUDAStream::priority_range() 获取 CUDA 流的优先级范围，返回值是一个 std::pair
    auto [least_priority, greatest_priority] =
        at::cuda::CUDAStream::priority_range();

    // 使用 Py_BuildValue 构建一个 Python 元组对象，包含 least_priority 和 greatest_priority 两个整数
    return Py_BuildValue("(ii)", least_priority, greatest_priority);

    // 结束 Torch 错误处理上下文
    END_HANDLE_TH_ERRORS
// 定义了一个静态函数 THCPStream_query，用于查询 CUDA 流的状态
static PyObject* THCPStream_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THCPStream*)_self; // 将 _self 转换为 THCPStream 对象
  return PyBool_FromLong(self->cuda_stream.query()); // 调用 cuda_stream 对象的 query 方法，返回查询结果
  END_HANDLE_TH_ERRORS // 结束错误处理
}

// 定义了一个静态函数 THCPStream_synchronize，用于同步 CUDA 流
static PyObject* THCPStream_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil; // 释放全局解释器锁
    auto self = (THCPStream*)_self; // 将 _self 转换为 THCPStream 对象
    self->cuda_stream.synchronize(); // 调用 cuda_stream 对象的 synchronize 方法，同步 CUDA 流
  }
  Py_RETURN_NONE; // 返回 None
  END_HANDLE_TH_ERRORS // 结束错误处理
}

// 定义了一个静态函数 THCPStream_eq，用于比较两个 THCPStream 对象的 cuda_stream 是否相等
static PyObject* THCPStream_eq(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = (THCPStream*)_self; // 将 _self 转换为 THCPStream 对象
  auto other = (THCPStream*)_other; // 将 _other 转换为 THCPStream 对象
  return PyBool_FromLong(self->cuda_stream == other->cuda_stream); // 比较两个 cuda_stream 是否相等，并返回比较结果
  END_HANDLE_TH_ERRORS // 结束错误处理
}

// 定义了一个静态成员结构体 PyMemberDef，用于描述 THCPStream 类型的成员，此处为空
// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMemberDef THCPStream_members[] = {{nullptr}};

// 定义了一个静态属性结构体 PyGetSetDef，用于描述 THCPStream 类型的属性
// 包括 cuda_stream 和 priority 两个属性的获取方法
// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyGetSetDef THCPStream_properties[] = {
    {"cuda_stream",
     (getter)THCPStream_get_cuda_stream, // 获取 cuda_stream 的方法
     nullptr,
     nullptr,
     nullptr},
    {"priority", (getter)THCPStream_get_priority, nullptr, nullptr, nullptr},
    {nullptr}};

// 定义了一个静态方法结构体 PyMethodDef，用于描述 THCPStream 类型的方法
// 包括 query、synchronize、priority_range 和 __eq__ 四个方法的定义
static PyMethodDef THCPStream_methods[] = {
    {"query", THCPStream_query, METH_NOARGS, nullptr}, // query 方法：查询 CUDA 流状态
    {"synchronize", THCPStream_synchronize, METH_NOARGS, nullptr}, // synchronize 方法：同步 CUDA 流
    {"priority_range",
     THCPStream_priority_range, // priority_range 方法：优先级范围（静态方法）
     METH_STATIC | METH_NOARGS,
     nullptr},
    {"__eq__", THCPStream_eq, METH_O, nullptr}, // __eq__ 方法：比较两个 THCPStream 对象的 cuda_stream 是否相等
    {nullptr}};

// 定义了一个 PyTypeObject 结构体 THCPStreamType，描述了 THCPStream 类型对象的属性和方法
PyTypeObject THCPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._CudaStreamBase", /* tp_name */ // 类型名称
    sizeof(THCPStream), /* tp_basicsize */ // 基本大小
    0, /* tp_itemsize */ // 每个元素大小
    (destructor)THCPStream_dealloc, /* tp_dealloc */ // 析构函数
    0, /* tp_vectorcall_offset */ // 向量调用偏移量
    nullptr, /* tp_getattr */ // 获取属性方法
    nullptr, /* tp_setattr */ // 设置属性方法
    nullptr, /* tp_reserved */ // 保留字段
    nullptr, /* tp_repr */ // repr 方法
    nullptr, /* tp_as_number */ // 数值类型协议
    nullptr, /* tp_as_sequence */ // 序列类型协议
    nullptr, /* tp_as_mapping */ // 映射类型协议
    nullptr, /* tp_hash  */ // 哈希方法
    nullptr, /* tp_call */ // 调用方法
    nullptr, /* tp_str */ // str 方法
    nullptr, /* tp_getattro */ // 获取属性方法（更广义）
    nullptr, /* tp_setattro */ // 设置属性方法（更广义）
    nullptr, /* tp_as_buffer */ // 缓冲区协议
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */ // 类型标志
    nullptr, /* tp_doc */ // 文档字符串
    nullptr, /* tp_traverse */ // 遍历对象方法
    nullptr, /* tp_clear */ // 清除对象方法
    nullptr, /* tp_richcompare */ // 富比较方法
    0, /* tp_weaklistoffset */ // 弱引用列表偏移量
    nullptr, /* tp_iter */ // 迭代器协议
    nullptr, /* tp_iternext */ // 迭代器下一个方法
    THCPStream_methods, /* tp_methods */ // 方法集合
    THCPStream_members, /* tp_members */ // 成员集合
    THCPStream_properties, /* tp_getset */ // 属性集合
    nullptr, /* tp_base */ // 基类
    nullptr, /* tp_dict */ // 字典
    nullptr, /* tp_descr_get */ // 描述符获取方法
    nullptr, /* tp_descr_set */ // 描述符设置方法
    0, /* tp_dictoffset */ // 字典偏移量
    nullptr, /* tp_init */ // 初始化方法
    nullptr, /* tp_alloc */ // 分配方法
    THCPStream_pynew, /* tp_new */ // 新建对象方法
};
// 声明一个名为 THCPStream_init 的函数，初始化 CUDA 流对象类型在 Python 模块中的定义
void THCPStream_init(PyObject* module) {
  // 增加对 THPStreamClass 的引用计数，确保其在函数生命周期内有效
  Py_INCREF(THPStreamClass);
  // 设置 THCPStreamType 的基类为 THPStreamClass，即指定 CUDA 流对象类型的基类
  THCPStreamType.tp_base = THPStreamClass;
  // 将 THCPStreamType 转换为 PyObject 类型，并赋值给 THCPStreamClass
  THCPStreamClass = (PyObject*)&THCPStreamType;
  // 准备 THCPStreamType 类型，如果失败则抛出 Python 异常
  if (PyType_Ready(&THCPStreamType) < 0) {
    throw python_error();
  }
  // 增加对 THCPStreamType 的引用计数，确保其在函数生命周期内有效
  Py_INCREF(&THCPStreamType);
  // 将 THCPStreamType 对象添加到指定的 Python 模块中，名字为 "_CudaStreamBase"
  if (PyModule_AddObject(
          module, "_CudaStreamBase", (PyObject*)&THCPStreamType) < 0) {
    throw python_error();
  }
}
```