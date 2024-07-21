# `.\pytorch\torch\csrc\xpu\Module.cpp`

```py
// 包含 ATen 库的头文件，用于张量操作和设备管理
#include <ATen/ATen.h>
// 包含 XPU 上下文管理相关的头文件
#include <ATen/xpu/XPUContext.h>
// 包含 XPU 生成器实现的头文件
#include <ATen/xpu/XPUGeneratorImpl.h>
// 包含调用一次初始化工具的头文件
#include <c10/util/CallOnce.h>
// 包含 XPU 缓存分配器的头文件
#include <c10/xpu/XPUCachingAllocator.h>
// 包含 XPU 功能函数的头文件
#include <c10/xpu/XPUFunctions.h>
// 包含 PyTorch 模块的头文件
#include <torch/csrc/Module.h>
// 包含 THP 的头文件
#include <torch/csrc/THP.h>
// 包含设备懒初始化工具的头文件
#include <torch/csrc/utils/device_lazy_init.h>
// 包含 Python C 函数帮助工具的头文件
#include <torch/csrc/utils/pycfunction_helpers.h>
// 包含 Python 数字处理的头文件
#include <torch/csrc/utils/python_numbers.h>
// 包含 Python 字符串处理的头文件
#include <torch/csrc/utils/python_strings.h>

// 在非 Windows 系统下引入 POSIX 线程库头文件
#ifndef WIN32
#include <pthread.h>
#endif

// 使用 torch 命名空间
using namespace torch;

// 标记在 XPU 初始化后派生的子进程为 true
static bool in_bad_fork = false;

// 在非 Windows 系统下定义，用于在 XPU 已初始化的派生子进程中调用
#ifndef WIN32
static void forked_child() {
  // 设置在派生的子进程中 XPU 初始化失败的标志
  in_bad_fork = true;
  // 强制 ATen 使用 XPU 需要初始化设备
  torch::utils::set_requires_device_init(at::kXPU, true);
}
#endif

// 在首次调用 XPU 之前应调用此函数，主要用于 lazy_init 中
// 注意：这与 initExtension 不同，因为存根 XPU 实现具有一些工作函数（例如 device_count），但不能完全初始化
static void poison_fork() {
#ifndef WIN32
  // 声明静态的一次初始化标志
  static c10::once_flag flag;
  // 使用 call_once 保证只调用一次 pthread_atfork 函数
  c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

// XPU 管理方法

// 封装的 Python 函数：THXPModule_isInBadFork_wrap
// 返回当前进程是否在初始化 XPU 后派生的子进程
static PyObject* THXPModule_isInBadFork_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回标志位 in_bad_fork，表示是否在坏的 fork 中
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

// 封装的 Python 函数：THXPModule_setDevice_wrap
// 设置当前 XPU 设备
PyObject* THXPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为有效的设备索引
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to set_device");

  // 解包设备索引
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  // 调用 ATen 的函数设置设备
  c10::xpu::set_device(device_index);

  // 返回 None 表示成功
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 封装的 Python 函数：THXPModule_exchangeDevice_wrap
// 切换当前 XPU 设备并返回之前的设备
PyObject* THXPModule_exchangeDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为有效的设备索引
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchange_device");

  // 解包设备索引
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  // 如果设备索引小于 0，则返回 -1
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  // 懒初始化 XPU 设备
  torch::utils::device_lazy_init(at::kXPU);
  // 调用 ATen 的函数切换设备并返回当前设备索引
  auto current_device = c10::xpu::exchange_device(device_index);

  // 将当前设备索引打包成 Python 对象返回
  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

// 封装的 Python 函数：THXPModule_maybeExchangeDevice_wrap
// 可能切换当前 XPU 设备并返回之前的设备
PyObject* THXPModule_maybeExchangeDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为有效的设备索引
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to maybe_exchange_device");

  // 解包设备索引
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  // 如果设备索引小于 0，则返回 -1
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  // 懒初始化 XPU 设备
  torch::utils::device_lazy_init(at::kXPU);
  // 调用 ATen 的函数可能切换设备并返回当前设备索引
  auto current_device = c10::xpu::maybe_exchange_device(device_index);

  // 将当前设备索引打包成 Python 对象返回
  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

// 封装的 Python 函数：THXPModule_getDevice_wrap
// 获取当前 XPU 设备索引
PyObject* THXPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  // 调用 ATen 的函数获取当前设备索引
  auto device_index = c10::xpu::current_device();

  // 将当前设备索引打包成 Python 对象返回
  return THPUtils_packDeviceIndex(device_index);
  END_HANDLE_TH_ERRORS
}
PyObject* THXPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 在调用之前，确保 fork 后不会被修改
  poison_fork();
  // 返回当前设备数量的 Python 对象封装
  return THPUtils_packUInt64(at::xpu::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_getCurrentStream_wrap(
    PyObject* self,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  // 检查输入的设备索引是否为长整型，如果不是，则抛出异常
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to current_stream");
  // 将 Python 设备索引对象转换为 C++ 设备索引对象
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  // 获取当前设备上的流对象
  auto stream = at::xpu::getCurrentXPUStream(c10_device_index);
  // 创建一个包含流信息的 Python 元组对象
  PyObject* output_tuple = PyTuple_New(3);
  // 将流的 ID 打包为 Python 的整数对象，并添加到元组中
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  // 将流的设备索引打包为 Python 的设备索引对象，并添加到元组中
  PyTuple_SetItem(
      output_tuple, 1, THPUtils_packDeviceIndex(stream.device_index()));
  // 将流的设备类型打包为 Python 的整数对象，并添加到元组中
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  // 返回包含流信息的 Python 元组对象
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_getCurrentStream_raw(
    PyObject* self,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  // 检查输入的设备索引是否为长整型，如果不是，则抛出异常
  TORCH_CHECK(
      THPUtils_checkLong(device_index),
      "invalid argument to getCurrentRawStream");
  // 将 Python 设备索引对象转换为 C++ 设备索引对象
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  // 返回当前设备上原始流队列的地址作为 Python 长整型对象
  return PyLong_FromVoidPtr(
      &at::xpu::getCurrentXPUStream(c10_device_index).queue());
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_setStream_wrap(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 初始化流 ID、设备索引和设备类型为零
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // 定义关键字参数列表
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  // 解析 Python 元组和关键字参数，并检查解析结果
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          // 忽略现代化建议以避免 C 数组
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  // 将流 ID、设备索引和设备类型解包为 XPUStream 对象
  auto stream = at::xpu::XPUStream::unpack3(
      stream_id,
      static_cast<c10::DeviceIndex>(device_index),
      static_cast<c10::DeviceType>(device_type));

  // 获取当前设备对象
  auto device = c10::xpu::current_device();
  // 如果当前设备与流的设备索引不匹配，则切换设备
  if (device != stream.device_index()) {
    c10::xpu::set_device(stream.device_index());
  }
  // 设置当前 XPU 流
  at::xpu::setCurrentXPUStream(stream);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_xpuSynchronize(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查输入的参数是否为长整型，如果不是，则抛出异常
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to synchronize");
  // 将 Python 设备索引对象解包为 C++ 设备索引
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  {
    // 释放 GIL，允许 Python 解释器在同步期间继续执行其他线程
    pybind11::gil_scoped_release no_gil;
    // 同步指定设备上的所有 SYCL 队列
    // 参见注释 [Synchronize Streams on Device]，仅同步我们保留的 SYCL 队列
    c10::xpu::syncStreamsOnDevice(device_index);
  }
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
// 清空 XPU 缓存的函数，返回 None
PyObject* THXPModule_emptyCache(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 调用 XPUCachingAllocator 的 emptyCache 方法清空缓存
  c10::xpu::XPUCachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  // 返回 Python 中的 None 对象
  Py_RETURN_NONE;
}

// XPU 模块初始化

static void registerXpuDeviceProperties(PyObject* module) {
  // 向 torch._C 添加 _xpuDeviceProperties 类
  using namespace c10::xpu;
  // 定义函数 get_device_type，根据设备类型返回对应字符串
  auto get_device_type = [](const DeviceProp& prop) {
    std::ostringstream stream;
    using namespace sycl::info;
    switch (prop.device_type) {
      case device_type::cpu:
        stream << "cpu";
        break;
      case device_type::gpu:
        stream << "gpu";
        break;
      case device_type::accelerator:
        stream << "accelerator";
        break;
      case device_type::host:
        stream << "host";
        break;
      default:
        // 处理未知设备类型的情况，输出其整数值
        stream << "unknown device type:"
               << static_cast<typename std::underlying_type<device_type>::type>(
                      prop.device_type);
        break;
    }
    return stream.str();  // 返回设备类型的字符串描述
  };
  // 定义函数 gpu_subslice_count，返回设备的子分片数量
  auto gpu_subslice_count = [](const DeviceProp& prop) {
    // 省略部分代码，未完整展示
    # 定义一个函数对象 m，将 Python 模块 module 转换为 py::module 类型
    auto m = py::handle(module).cast<py::module>();
    # 在 Python 模块 m 中创建一个名为 _XpuDeviceProperties 的 Python 类，并绑定 C++ 类 DeviceProp 的成员到 Python 的只读属性
    py::class_<DeviceProp>(m, "_XpuDeviceProperties")
        .def_readonly("name", &DeviceProp::name)  // 绑定 DeviceProp 的 name 属性为只读
        .def_readonly("platform_name", &DeviceProp::platform_name)  // 绑定 platform_name 属性为只读
        .def_readonly("vendor", &DeviceProp::vendor)  // 绑定 vendor 属性为只读
        .def_readonly("driver_version", &DeviceProp::driver_version)  // 绑定 driver_version 属性为只读
        .def_readonly("version", &DeviceProp::version)  // 绑定 version 属性为只读
        .def_readonly("total_memory", &DeviceProp::global_mem_size)  // 绑定 global_mem_size 属性为只读，命名为 total_memory
        .def_readonly("max_compute_units", &DeviceProp::max_compute_units)  // 绑定 max_compute_units 属性为只读
        .def_readonly("gpu_eu_count", &DeviceProp::gpu_eu_count)  // 绑定 gpu_eu_count 属性为只读
        .def_property_readonly("gpu_subslice_count", gpu_subslice_count)  // 绑定 gpu_subslice_count 属性为只读属性，调用函数 gpu_subslice_count 获取值
        .def_readonly("max_work_group_size", &DeviceProp::max_work_group_size)  // 绑定 max_work_group_size 属性为只读
        .def_readonly("max_num_sub_groups", &DeviceProp::max_num_sub_groups)  // 绑定 max_num_sub_groups 属性为只读
        .def_readonly("sub_group_sizes", &DeviceProp::sub_group_sizes)  // 绑定 sub_group_sizes 属性为只读
        .def_readonly("has_fp16", &DeviceProp::has_fp16)  // 绑定 has_fp16 属性为只读
        .def_readonly("has_fp64", &DeviceProp::has_fp64)  // 绑定 has_fp64 属性为只读
        .def_readonly("has_atomic64", &DeviceProp::has_atomic64)  // 绑定 has_atomic64 属性为只读
        .def_property_readonly("type", get_device_type)  // 绑定 type 属性为只读属性，调用函数 get_device_type 获取值
        .def(
            "__repr__",
            // 定义 __repr__ 方法，用 lambda 表达式生成字符串表示对象的详细信息
            [&get_device_type, &gpu_subslice_count](const DeviceProp& prop) {
              std::ostringstream stream;
              stream << "_XpuDeviceProperties(name='" << prop.name
                     << "', platform_name='" << prop.platform_name << "', type='"
                     << get_device_type(prop) << "', driver_version='"
                     << prop.driver_version << "', total_memory="
                     << prop.global_mem_size / (1024ull * 1024)
                     << "MB, max_compute_units=" << prop.max_compute_units
                     << ", gpu_eu_count=" << prop.gpu_eu_count
                     << ", gpu_subslice_count=" << gpu_subslice_count(prop)
                     << ", max_work_group_size=" << prop.max_work_group_size
                     << ", max_num_sub_groups=" << prop.max_num_sub_groups
                     << ", sub_group_sizes=[" << prop.sub_group_sizes
                     << "], has_fp16=" << prop.has_fp16
                     << ", has_fp64=" << prop.has_fp64
                     << ", has_atomic64=" << prop.has_atomic64 << ")";
              return stream.str();
            });
}

// 定义静态函数，将获取设备属性的方法绑定到 torch.xpu 模块中
static void bindGetDeviceProperties(PyObject* module) {
  // 将传入的 module 转换为 py::module 类型
  auto m = py::handle(module).cast<py::module>();
  // 在 torch.xpu 模块中注册名为 "_get_device_properties" 的方法
  m.def(
      "_get_device_properties",
      [](c10::DeviceIndex device) -> c10::xpu::DeviceProp* {
        // 调用 at::xpu::getDeviceProperties 方法获取指定设备的属性
        return at::xpu::getDeviceProperties(device);
      },
      py::return_value_policy::reference);
}

// Python 部分的回调函数，用于初始化 python 类的附加部分
static PyObject* THXPModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 断言确保不在错误的 fork 中
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level
  // 毒害 fork
  poison_fork();
  // 惰性初始化 XPU 全局上下文
  at::globalContext().lazyInitXPU();

  // 导入 torch.xpu 模块
  auto m = THPObjectPtr(PyImport_ImportModule("torch.xpu"));
  if (!m)
    throw python_error();

  // 辅助函数，用于设置模块属性
  auto set_module_attr = [&](const char* name, PyObject* v) {
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  // 获取 XPU 设备数量
  auto num_gpus = c10::xpu::device_count();
  // 创建默认 XPU 生成器元组对象
  THPObjectPtr default_xpu_generators(
      PyTuple_New(static_cast<Py_ssize_t>(num_gpus)));
  // 遍历设备索引范围，为每个设备设置默认生成器
  for (const auto i : c10::irange(num_gpus)) {
    const auto& gen = at::xpu::detail::getDefaultXPUGenerator(i);
    auto* cast_gen = THPGenerator_initDefaultGenerator(gen);
    PyTuple_SetItem(default_xpu_generators.get(), i, cast_gen);
  }
  // 设置 torch.xpu 模块的 "default_generators" 属性
  set_module_attr("default_generators", default_xpu_generators.get());
  // 将设备属性绑定到 torch.xpu 模块中
  bindGetDeviceProperties(m);

  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
// 定义静态 PyMethodDef 结构体数组 _THXPModule_methods
static struct PyMethodDef _THXPModule_methods[] = {
    {"_xpu_init", THXPModule_initExtension, METH_NOARGS, nullptr},
    {"_xpu_setDevice", THXPModule_setDevice_wrap, METH_O, nullptr},
    {"_xpu_exchangeDevice", THXPModule_exchangeDevice_wrap, METH_O, nullptr},
    {"_xpu_maybeExchangeDevice",
     THXPModule_maybeExchangeDevice_wrap,
     METH_O,
     nullptr},
    {"_xpu_getDevice", THXPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_xpu_getDeviceCount",
     THXPModule_getDeviceCount_wrap,
     METH_NOARGS,
     nullptr},
    {"_xpu_isInBadFork", THXPModule_isInBadFork_wrap, METH_NOARGS, nullptr},
    {"_xpu_getCurrentStream",
     THXPModule_getCurrentStream_wrap,
     METH_O,
     nullptr},
    {"_xpu_getCurrentRawStream",
     THXPModule_getCurrentStream_raw,
     METH_O,
     nullptr},
    {"_xpu_setStream",
     castPyCFunctionWithKeywords(THXPModule_setStream_wrap),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_xpu_synchronize", THXPModule_xpuSynchronize, METH_O, nullptr},
    {"_xpu_emptyCache", THXPModule_emptyCache, METH_NOARGS, nullptr},
    {nullptr}};

// 返回 _THXPModule_methods 数组的指针
PyMethodDef* THXPModule_methods() {
  return _THXPModule_methods;
}

// 定义 torch::xpu 命名空间下的初始化函数，注册 XPU 设备属性
namespace torch::xpu {

void initModule(PyObject* module) {
  registerXpuDeviceProperties(module);
}

} // namespace torch::xpu
```