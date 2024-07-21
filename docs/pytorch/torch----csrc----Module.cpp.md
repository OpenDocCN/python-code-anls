# `.\pytorch\torch\csrc\Module.cpp`

```py
#include <ATen/DeviceAccelerator.h>
#include <c10/util/Optional.h>
#include <fmt/core.h>
#include <sys/types.h>
#include <torch/csrc/python_headers.h>

#ifndef _MSC_VER
#include <sys/socket.h>
#endif

#include <ATen/ATen.h>
#include <ATen/BlasBackend.h>
#include <ATen/DLConvertor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/LegacyVmapMode.h>
#include <ATen/LinalgBackend.h>
#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <ATen/core/Vitals.h>
#include <ATen/detail/AcceleratorHooksInterface.h>
#include <ATen/dlpack.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/Normalization.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/AbortHandler.h>
#include <c10/util/Backtrace.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>
#include <libshm.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/THConcat.h>
#include <torch/csrc/utils/pybind.h>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

#include <ATen/ThreadLocalPythonObjects.h>
#include <torch/csrc/DataLoader.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Event.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/TypeInfo.h>
#include <torch/csrc/api/include/torch/python/init.h>
#include <torch/csrc/autograd/generated/python_return_types.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_enum_tag.h>
#include <torch/csrc/autograd/python_fft_functions.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_legacy_variable.h>
#include <torch/csrc/autograd/python_linalg_functions.h>
#include <torch/csrc/autograd/python_nested_functions.h>
#include <torch/csrc/autograd/python_nn_functions.h>
#include <torch/csrc/autograd/python_sparse_functions.h>
#include <torch/csrc/autograd/python_special_functions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/dynamo/init.h>
#include <torch/csrc/functorch/init.h>
#include <torch/csrc/fx/node.h>
#include <torch/csrc/inductor/aoti_runner/pybind.h>
#include <torch/csrc/jit/python/init.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/lazy/python/init.h>
#include <torch/csrc/monitor/python_init.h>
#include <torch/csrc/mps/Module.h>
#include <torch/csrc/mtia/Module.h>
#include <torch/csrc/multiprocessing/init.h>
#include <torch/csrc/onnx/init.h>
#include <torch/csrc/profiler/python/init.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/disable_torch_function.h>



// 包含了 ATen 和 C10 库的头文件，用于张量计算和设备管理
#include <ATen/DeviceAccelerator.h>
#include <c10/util/Optional.h>
// 提供了格式化输出的函数接口
#include <fmt/core.h>
// 系统级的类型定义
#include <sys/types.h>
// Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>

// 如果不是在 Visual Studio 编译环境下，包含套接字操作的头文件
#ifndef _MSC_VER
#include <sys/socket.h>
#endif

// ATen 库的核心功能和工具函数
#include <ATen/ATen.h>
#include <ATen/BlasBackend.h>
#include <ATen/DLConvertor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/LegacyVmapMode.h>
#include <ATen/LinalgBackend.h>
#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <ATen/core/Vitals.h>
#include <ATen/detail/AcceleratorHooksInterface.h>
#include <ATen/dlpack.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/Normalization.h>
// C10 库的核心功能和工具函数
#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/AbortHandler.h>
#include <c10/util/Backtrace.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>
// 用于共享内存管理的库
#include <libshm.h>
// pybind11 库，用于 Python 与 C++ 之间的接口
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Torch 中的 THConcat 相关头文件
#include <torch/csrc/THConcat.h>
// Torch 中的 Python 绑定工具
#include <torch/csrc/utils/pybind.h>
// 标准 C 库，包含基本的函数和宏定义
#include <cstdlib>
// 标准输入输出流
#include <iostream>
// 无序映射的实现，用于管理一组键值对
#include <unordered_map>

// ATen 内部的 Python 对象管理
#include <ATen/ThreadLocalPythonObjects.h>
// Torch 数据加载器模块
#include <torch/csrc/DataLoader.h>
// Torch 设备相关的功能
#include <torch/csrc/Device.h>
// Torch 数据类型
#include <torch/csrc/Dtype.h>
// Torch 动态类型支持
#include <torch/csrc/DynamicTypes.h>
// Torch 事件处理
#include <torch/csrc/Event.h>
// Torch 随机数生成器
#include <torch/csrc/Generator.h>
// Torch 张量的布局
#include <torch/csrc/Layout.h>
// Torch 内存格式
#include <torch/csrc/MemoryFormat.h>
// Torch 量化方案
#include <torch/csrc/QScheme.h>
// Torch 流对象
#include <torch/csrc/Stream.h>
// Torch Python 桥接头文件
#include <torch/csrc/THP.h>
// Torch 类型信息
#include <torch/csrc/TypeInfo.h>
// Torch Python 接口初始化
#include <torch/csrc/api/include/torch/python/init.h>
// Torch 自动求导生成的 Python 返回类型
#include <torch/csrc/autograd/generated/python_return_types.h>
// Torch 自动求导的 C++ 函数接口
#include <torch/csrc/autograd/python_cpp_function.h>
// Torch 自动求导的 Python 枚举标签
#include <torch/csrc/autograd/python_enum_tag.h>
// Torch 自动求导的 FFT 函数
#include <torch/csrc/autograd/python_fft_functions.h>
// Torch 自动求导的 Python 函数
#include <torch/csrc/autograd/python_function.h>
// Torch 自动求导的 Python 旧版变量支持
#include <torch/csrc/autograd/python_legacy_variable.h>
// Torch 自动求导的线性代数函数
#include <torch/csrc/autograd/python_linalg_functions.h>
// Torch 自动求导的嵌套函数
#include <torch/csrc/autograd/python_nested_functions.h>
// Torch 自动求导的神经网络函数
#include <torch/csrc/autograd/python_nn_functions.h>
// Torch 自动求导的稀疏函数支持
#include <torch/csrc/autograd/python_sparse_functions.h>
// Torch 自动求导的特殊函数支持
#include <torch/csrc/autograd/python_special_functions.h>
// Torch 自动求导的变量支持
#include <torch/csrc/autograd/python_variable.h>
// Torch CPU 模块
#include <torch/csrc/cpu/Module.h>
// Torch 动态图模块的初始化
#include <torch/csrc/dynamo/init.h>
// Torch Functorch 模块的初始化
#include <torch/csrc/functorch/init.h>
// Torch 函数图节点
#include <torch/csrc/fx/node.h>
// Torch 自动化测试接口的 Python 绑定
#include <torch/csrc/inductor/aoti_runner/pybind.h>
// Torch JIT 模块的初始化
#include <torch/csrc/jit/python/init.h>
// Torch
// 包含 Torch 相关的头文件
#include <torch/csrc/utils/init.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/utils/tensor_layouts.h>
#include <torch/csrc/utils/tensor_memoryformats.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/utils/tensor_qschemes.h>
#include <torch/csrc/utils/verbose.h>

// 包含 Torch 中与 CUDA 相关的头文件，条件编译检查 USE_CUDA 宏
#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#ifdef __HIP_PLATFORM_AMD__
#include <ATen/native/cudnn/hip/BatchNorm.h>
#else
#include <ATen/native/cudnn/BatchNorm.h>
#endif
#endif

// 包含与分布式训练相关的头文件，条件编译检查 USE_DISTRIBUTED 宏
#ifdef USE_DISTRIBUTED
#ifdef USE_C10D
#include <torch/csrc/distributed/autograd/python_autograd.h>
#include <torch/csrc/distributed/c10d/c10d.h>
#include <torch/csrc/distributed/rpc/rpc.h>
#include <torch/csrc/distributed/rpc/testing/testing.h>
#endif
#endif

// 包含与 Valgrind 工具相关的头文件，条件编译检查 USE_VALGRIND 宏
#if defined(USE_VALGRIND)
#include <callgrind.h>
#endif

// 命名空间别名 py 表示 pybind11
namespace py = pybind11;

// 定义模块对象指针
PyObject* module;

// Torch 默认的 CPU 生成器指针
THPGenerator* THPDefaultCPUGenerator = nullptr;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// 初始化 Python 类名字的函数，接受一个 PyObject* 类型的参数
static PyObject* THPModule_initNames(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 静态变量，存储类名的字符串向量
  static std::vector<std::string> names;

  // 获取传入参数的 PyObject* 类型，期望是一个序列
  THPObjectPtr types(PySequence_Fast(arg, "expected a sequence"));
  if (!types)
    return nullptr;

  // 获取序列中元素的数量
  auto num_classes = PySequence_Fast_GET_SIZE(types.get());
  // 预留足够的空间以存储类名
  names.reserve(names.size() + num_classes);
  // 遍历每个类
  for (Py_ssize_t i = 0; i < num_classes; i++) {
    // 获取序列中的对象
    PyObject* obj = PySequence_Fast_GET_ITEM(types.get(), i);
    // 检查对象是否为 PyTypeObject 类型
    TORCH_CHECK(PyType_Check(obj), "expected a PyTypeObject");
    PyTypeObject* type = (PyTypeObject*)obj;

    // 获取类的模块名
    THPObjectPtr module_name(PyObject_GetAttrString(obj, "__module__"));
    if (!module_name)
      return nullptr;
    // 检查模块名是否为字符串
    TORCH_CHECK(
        THPUtils_checkString(module_name.get()),
        "expected __module__ to be a string");
    // 解包模块名字符串
    std::string name = THPUtils_unpackString(module_name.get());
    // 构造完整的类名
    names.emplace_back(name + "." + type->tp_name);
    // 更新类的类型名称
    type->tp_name = names.back().c_str();
  }
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

//
// Python 部分的回调函数，用于对 Python 类进行额外初始化
static PyObject* THPModule_initExtension(
    PyObject* _unused,
    PyObject* shm_manager_path) {
  HANDLE_TH_ERRORS
  // 如果启用了 CPP 堆栈跟踪
#if !defined(FBCODE_CAFFE2)
  if (torch::get_cpp_stacktraces_enabled()) {
    // 设置堆栈跟踪获取器，捕获C++异常时调用
    c10::SetStackTraceFetcher([]() -> std::string {
      // 收集当前的Torch捕获堆栈跟踪信息，不包括原始C++异常
      auto tb = torch::CapturedTraceback::gather(false, false, true);
      // 如果地址符号化模式为addr2line，则打印警告信息
      if (torch::get_symbolize_mode() == torch::unwind::Mode::addr2line) {
        LOG(WARNING)
            << "symbolizing C++ stack trace for exception; if this hangs, rerun with TORCH_DISABLE_ADDR2LINE=1..."
            << std::endl;
      }
      // 符号化堆栈跟踪信息
      auto s_tbs = torch::symbolize({tb.get()});
      std::stringstream oss;
      // 添加C++捕获堆栈跟踪的描述
      oss << "C++ CapturedTraceback:" << std::endl;
      // 获取第一个跟踪信息
      const auto& s_tb = s_tbs.tracebacks.at(0);
      // 遍历所有帧信息
      for (auto idx : c10::irange(s_tb.size())) {
        // 跳过前几个帧：
        //  #1 torch::CapturedTraceback::gather(bool, bool, bool)
        //  #2 THPModule_initExtension
        //  #3 THPModule_initExtension(_object*, _object*)::{lambda()#1}
        if (idx <= 3) {
          continue;
        }
        // 获取帧的标识符
        auto frame_id = s_tb[idx];
        // 获取帧信息
        const auto& frame = s_tbs.all_frames.at(frame_id);
        // 输出帧号、函数名、文件名和行号信息
        oss << "#" << idx << " " << frame.funcname << " from " << frame.filename
            << ":" << frame.lineno << std::endl;
      }
      // 返回描述完整的堆栈跟踪信息
      return oss.str();
    });
  }
#endif
  // 如果 shm_manager_path 不是字符串或字节对象，设置错误并返回空指针
  if (!THPUtils_checkString(shm_manager_path)) {
    THPUtils_setError(
        "initialization error - expected bytes/string object as shm_manager_path!");
    return nullptr;
  }
  // 初始化 Torch 的布局
  torch::utils::initializeLayouts();
  // 初始化 Torch 的内存格式
  torch::utils::initializeMemoryFormats();
  // 初始化 Torch 的量化方案
  torch::utils::initializeQSchemes();
  // 初始化 Torch 的数据类型
  torch::utils::initializeDtypes();
  // 初始化 Torch 张量的 Python 绑定
  torch::tensors::initialize_python_bindings();
  // 解包 shm_manager_path 字符串，转换为 std::string 类型
  std::string path = THPUtils_unpackString(shm_manager_path);
  // 使用路径初始化共享内存库
  libshm_init(path.c_str());

  // 主线程通常启动 CPU/GPU/加速器内核，因此对延迟敏感。如果线程命名，可以更容易调试性能问题。
  c10::setThreadName("pt_main_thread");

  // 导入 torch 模块
  auto module = THPObjectPtr(PyImport_ImportModule("torch"));
  // 如果导入失败，则抛出 Python 错误
  if (!module)
    throw python_error();

  // 在 Torch 存储初始化完成后执行后处理
  THPStorage_postInit(module);
  // 初始化自动微分函数
  THPAutograd_initFunctions();
  // 返回 Python None 对象
  Py_RETURN_NONE;
  // 处理 Torch 错误
  END_HANDLE_TH_ERRORS
}

// 这两个函数的思想是简化测试是否使用 ASAN 编译的过程：它们设计为如果 ASAN 未启用，则不会崩溃，但如果启用了 ASAN，则会触发。
// 这允许我们运行一个“canary”测试，检查我们的构建环境是否配置正确。

static PyObject* THPModule_crashIfCsrcASAN(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为整数，否则抛出错误
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "crash_if_csrc_asan expects an int, but got ",
      THPUtils_typename(arg));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
  // 声明一个易失性字符数组，并尝试写入其第一个元素，用于测试 ASAN
  volatile char x[3];
  x[THPUtils_unpackInt(arg)] = 0;
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  // 返回试图写入的第一个元素的整数表示
  return THPUtils_packInt32(x[0]);
  // 处理 Torch 错误
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_crashIfCsrcUBSAN(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为整数，否则抛出错误
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "crash_if_csrc_ubsan expects an int, but got ",
      THPUtils_typename(arg));
  // 尝试除以整数参数，用于触发 UBSAN
  int32_t x = THPUtils_unpackInt(arg);
  double y = 1.0 / x;
  // 返回除法结果的整数部分
  return THPUtils_packInt32((int)y);
  // 处理 Torch 错误
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_crashIfvptrUBSAN(PyObject* module, PyObject* noarg) {
  // 这段代码应该完全正常工作，因为在未启用 RTTI 和 UBSAN 的情况下，Foo 和 Baz 的虚表是相同的。
  struct Foo {
    // Foo 结构的虚函数，需要在派生类中实现
    virtual int bar() = 0;
    // 默认析构函数
    virtual ~Foo() = default;
  };
  struct Baz {
    // Baz 结构的虚函数实现，返回固定整数值
    virtual int bar() {
      return 17;
    }
    // 默认析构函数
    virtual ~Baz() = default;
  };
  // 创建 Baz 结构的实例
  Baz x{};
  // 将 Baz 实例的指针转换为 Foo 类型指针，调用其虚函数
  auto y = static_cast<Foo*>(static_cast<void*>(&x));
  // 调用虚函数 bar() 返回结果
  auto rc = y->bar();
  // 返回虚函数 bar() 的结果
  return THPUtils_packInt32(rc);
}

static PyObject* THPModule_crashIfATenASAN(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为整数，否则抛出错误
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "crash_if_aten_asan expects an int, "
      "but got ",
      THPUtils_typename(arg));
  // 调用 ATen 库的特定函数，用于测试 ASAN
  return THPUtils_packInt32(at::_crash_if_asan(THPUtils_unpackInt(arg)));
  // 处理 Torch 错误
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_abort(PyObject* module, PyObject* noargs) {
  // 终止程序运行
  std::terminate();
  // 返回 Python None 对象
  Py_RETURN_NONE;
}
// 定义一个名为 THPModule_crashIfDebugAssertsFail 的函数，接受两个参数：模块对象和参数对象
static PyObject* THPModule_crashIfDebugAssertsFail(
    PyObject* module,
    PyObject* arg) {
  HANDLE_TH_ERRORS  // 开始处理 Torch 异常

  // 使用 TORCH_CHECK 来验证参数 arg 是否为长整型，否则抛出错误信息
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "crash_if_debug_asserts_fail expects an int, but got ",
      THPUtils_typename(arg));

  // 在调试构建中，使用 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 来验证 arg 不等于 424242
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      THPUtils_unpackInt(arg) != 424242,
      "Expect anything but 424242 as an input for debug builds");

  // 将整数 0 打包成 PyObject* 并返回
  return THPUtils_packInt32(0);

  END_HANDLE_TH_ERRORS  // 结束 Torch 异常处理
}

// 定义一个名为 THPModule_getNumThreads 的函数，接受模块对象和无参数
static PyObject* THPModule_getNumThreads(PyObject* module, PyObject* noargs) {
  // 返回当前线程数，并打包成 PyObject* 返回
  return THPUtils_packInt32(at::get_num_threads());
}

// 定义一个名为 THPModule_setNumThreads 的函数，接受模块对象和参数
static PyObject* THPModule_setNumThreads(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS  // 开始处理 Torch 异常

  // 使用 TORCH_CHECK 来验证参数 arg 是否为长整型，否则抛出错误信息
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "set_num_threads expects an int, but got ",
      THPUtils_typename(arg));

  // 将参数解包成整数 nthreads
  int nthreads = (int)THPUtils_unpackLong(arg);

  // 验证 nthreads 必须为正数，否则抛出错误信息
  TORCH_CHECK(nthreads > 0, "set_num_threads expects a positive integer");

  // 设置 Torch 的线程数为 nthreads
  at::set_num_threads(nthreads);

  // 返回 None
  Py_RETURN_NONE;

  END_HANDLE_TH_ERRORS  // 结束 Torch 异常处理
}

// 定义一个名为 THPModule_getNumInteropThreads 的函数，接受模块对象和无参数
static PyObject* THPModule_getNumInteropThreads(
    PyObject* module,
    PyObject* noargs) {
  // 返回当前互操作线程数，并打包成 PyObject* 返回
  return THPUtils_packInt32(at::get_num_interop_threads());
}

// 定义一个名为 THPModule_setNumInteropThreads 的函数，接受模块对象和参数
static PyObject* THPModule_setNumInteropThreads(
    PyObject* module,
    PyObject* arg) {
  HANDLE_TH_ERRORS  // 开始处理 Torch 异常

  // 使用 TORCH_CHECK 来验证参数 arg 是否为长整型，否则抛出错误信息
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "set_num_interop_threads expects an int, "
      "but got ",
      THPUtils_typename(arg));

  // 将参数解包成整数 nthreads
  int nthreads = (int)THPUtils_unpackLong(arg);

  // 验证 nthreads 必须为正数，否则抛出错误信息
  TORCH_CHECK(
      nthreads > 0, "set_num_interop_threads expects a positive integer");

  // 设置 Torch 的互操作线程数为 nthreads
  at::set_num_interop_threads(nthreads);

  // 返回 None
  Py_RETURN_NONE;

  END_HANDLE_TH_ERRORS  // 结束 Torch 异常处理
}

// 定义一个名为 THPModule_setDefaultTensorType 的函数，接受两个参数：未使用的模块和类型对象
PyObject* THPModule_setDefaultTensorType(PyObject* _unused, PyObject* type) {
  HANDLE_TH_ERRORS  // 开始处理 Torch 异常

  // 调用 Torch C++ API 设置默认张量类型为 type
  torch::tensors::py_set_default_tensor_type(type);

  // 返回 None
  Py_RETURN_NONE;

  END_HANDLE_TH_ERRORS  // 结束 Torch 异常处理
}

// 定义一个名为 THPModule_setDefaultDtype 的函数，接受两个参数：未使用的模块和 dtype 对象
PyObject* THPModule_setDefaultDtype(PyObject* _unused, PyObject* dtype) {
  HANDLE_TH_ERRORS  // 开始处理 Torch 异常

  // 调用 Torch C++ API 设置默认数据类型为 dtype
  torch::tensors::py_set_default_dtype(dtype);

  // 返回 None
  Py_RETURN_NONE;

  END_HANDLE_TH_ERRORS  // 结束 Torch 异常处理
}

// 定义一个名为 THPModule_swap_tensor_impl 的函数，接受两个参数：未使用的模块和参数元组 args
PyObject* THPModule_swap_tensor_impl(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS  // 开始处理 Torch 异常

  // 定义两个 PyObject* 类型的指针，并初始化为 nullptr
  PyObject* a_ = nullptr;
  PyObject* b_ = nullptr;

  // 尝试从 args 解析两个参数为 PyObject* 类型的对象 a_ 和 b_
  if (!PyArg_ParseTuple(args, "OO", &a_, &b_)) {
      // 解析失败时，直接返回 nullptr
      // 在 Python 层面会抛出异常


这段代码的注释解释了每个函数的作用以及每行代码的具体功能，包括异常处理和参数验证等。
    // 返回空指针表示操作失败
    return nullptr;
  }

  // 确保参数 a_ 和 b_ 都是有效的 Tensors
  TORCH_CHECK(THPVariable_Check(a_));
  TORCH_CHECK(THPVariable_Check(b_));

  // 将参数 a_ 和 b_ 转换为 THPVariable 指针
  THPVariable* a = reinterpret_cast<THPVariable*>(a_);
  THPVariable* b = reinterpret_cast<THPVariable*>(b_);

  // 检查 Tensor 对象的弱引用计数，确保没有其他对象持有其引用
  TORCH_CHECK(
      a->cdata->weak_use_count() == 1,
      "Expected no weakrefs to t1's Tensor object but got  ",
      a->cdata->weak_use_count() - 1);
  TORCH_CHECK(
      b->cdata->weak_use_count() == 1,
      "Expected no weakrefs to t2's Tensor object but got  ",
      b->cdata->weak_use_count() - 1);

  // 交换 Tensor 对象的实现指针
  c10::MaybeOwned<at::Tensor> tmp = a->cdata;

  // TensorImpl 包含 PyObjectSlots，这些 slots 引用与 TensorImpl 关联的 PyObject。
  // 也需要交换这些引用。
  std::optional<PyObject*> mb_obj_a =
      a->cdata->unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
          getPyInterpreter(), /*ignore_hermetic_tls=*/false);
  std::optional<PyObject*> mb_obj_b =
      b->cdata->unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
          getPyInterpreter(), /*ignore_hermetic_tls=*/false);
  TORCH_INTERNAL_ASSERT(
      mb_obj_a.has_value() && mb_obj_b.has_value(),
      "Both tensors should have PyObjects tagged by the current python interpreter");
  TORCH_CHECK(mb_obj_a.value() == a_);
  TORCH_CHECK(mb_obj_b.value() == b_);

  // 执行实际的 Tensor 实现指针交换
  a->cdata = b->cdata;
  b->cdata = tmp;

  // 重新初始化 PyObjectSlots 的引用，标记为由我们标记的状态
  a->cdata->unsafeGetTensorImpl()->pyobj_slot()->init_pyobj(
      getPyInterpreter(), a_, c10::impl::PyInterpreterStatus::TAGGED_BY_US);
  b->cdata->unsafeGetTensorImpl()->pyobj_slot()->init_pyobj(
      getPyInterpreter(), b_, c10::impl::PyInterpreterStatus::TAGGED_BY_US);

  // 返回 Python 中的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_addDocStr(PyObject* _unused, PyObject* args) {
  // 添加一个 __doc__ 字符串到函数中，类似于 numpy 的 arr_add_docstring 函数
  static std::vector<std::string> all_docs;  // 静态变量，存储所有文档字符串
  PyObject* obj = nullptr;                   // Python 对象指针，初始化为 nullptr
  PyObject* doc_obj = nullptr;               // 文档字符串的 Python 对象指针，初始化为 nullptr
  if (!PyArg_ParseTuple(args, "OO", &obj, &doc_obj)) {  // 解析输入参数，要求两个对象
    return nullptr;                           // 解析失败时返回空指针
  }

  const char* doc_str = "<invalid string>";   // 默认的文档字符串，无效字符串
  if (THPUtils_checkString(doc_obj)) {        // 检查文档对象是否为字符串
    all_docs.push_back(THPUtils_unpackString(doc_obj));  // 将解包后的字符串加入静态容器中
    doc_str = all_docs.back().c_str();        // 获取最新加入的文档字符串的 C 字符串形式
  }

  if (Py_TYPE(obj) == &PyCFunction_Type) {    // 如果对象是 C 函数类型
    PyCFunctionObject* f = (PyCFunctionObject*)obj;  // 转换为 C 函数对象指针
    if (f->m_ml->ml_doc) {                    // 如果函数已经有文档字符串
      return PyErr_Format(
          PyExc_RuntimeError,
          "function '%s' already has a docstring",
          f->m_ml->ml_name);                  // 返回运行时错误，函数已经有文档字符串
    }
    f->m_ml->ml_doc = doc_str;                // 设置函数的文档字符串
  } else if (strcmp(Py_TYPE(obj)->tp_name, "method_descriptor") == 0) {
    PyMethodDescrObject* m = (PyMethodDescrObject*)obj;  // 转换为方法描述符对象指针
    if (m->d_method->ml_doc) {                // 如果方法已经有文档字符串
      return PyErr_Format(
          PyExc_RuntimeError,
          "method '%s' already has a docstring",
          m->d_method->ml_name);              // 返回运行时错误，方法已经有文档字符串
    }
    m->d_method->ml_doc = doc_str;            // 设置方法的文档字符串
  } else if (strcmp(Py_TYPE(obj)->tp_name, "getset_descriptor") == 0) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
    PyGetSetDescrObject* m = (PyGetSetDescrObject*)obj;  // 转换为 get/set 描述符对象指针
    if (m->d_getset->doc) {                   // 如果属性已经有文档字符串
      return PyErr_Format(
          PyExc_RuntimeError,
          "attribute '%s' already has a docstring",
          m->d_getset->name);                 // 返回运行时错误，属性已经有文档字符串
    }
    m->d_getset->doc = doc_str;               // 设置属性的文档字符串
  } else if (Py_TYPE(obj) == &PyType_Type) {  // 如果对象是类型对象
    PyTypeObject* t = (PyTypeObject*)obj;     // 转换为类型对象指针
    if (t->tp_doc) {                          // 如果类型已经有文档字符串
      return PyErr_Format(
          PyExc_RuntimeError, "Type '%s' already has a docstring", t->tp_name);  // 返回运行时错误，类型已经有文档字符串
    }
    t->tp_doc = doc_str;                      // 设置类型的文档字符串
  } else {
    return PyErr_Format(
        PyExc_TypeError,
        "don't know how to add docstring to type '%s'",
        Py_TYPE(obj)->tp_name);               // 返回类型错误，不知道如何添加文档字符串到该类型
  }

  Py_INCREF(obj);                             // 增加对象的引用计数
  return obj;                                 // 返回设置了文档字符串后的对象
}

PyObject* THPModule_inferSize(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? (Py_ssize_t)PyTuple_Size(args) : 0;  // 获取参数元组的大小
  TORCH_CHECK(num_args == 2, "expected exactly 2 arguments");       // 断言参数数量为 2
  PyObject* arg1 = PyTuple_GET_ITEM(args, 0);  // 获取参数元组的第一个参数
  TORCH_CHECK(THPSize_Check(arg1), "expected a torch.Size as argument 1");  // 断言第一个参数是 torch.Size 类型
  PyObject* arg2 = PyTuple_GET_ITEM(args, 1);  // 获取参数元组的第二个参数
  TORCH_CHECK(THPSize_Check(arg2), "expected a torch.Size as argument 2");  // 断言第二个参数是 torch.Size 类型

  auto size1 = THPUtils_unpackLongs(arg1);     // 解包第一个参数为长整数数组
  auto size2 = THPUtils_unpackLongs(arg2);     // 解包第二个参数为长整数数组
  auto sizes = at::infer_size(size1, size2);   // 推断尺寸大小
  return THPSize_NewFromSizes(static_cast<int64_t>(sizes.size()), sizes.data());  // 返回推断后的尺寸作为新的 torch.Size 对象
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setBackcompatBroadcastWarn(
    PyObject* module,
    PyObject* arg) {
      HANDLE_TH_ERRORS
      // 检查参数是否为布尔类型
      TORCH_CHECK(
          PyBool_Check(arg),
          "set_backcompat_broadcast_warn expects a bool, "
          "but got ",
          THPUtils_typename(arg));
      // 根据参数设置向后兼容的广播警告
      setBackCompatBroadcastWarn(arg == Py_True);
      // 返回 Python None 对象
      Py_RETURN_NONE;
      END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getBackcompatBroadcastWarn(
    PyObject* module,
    PyObject* noargs) {
  // 检查是否需要向后兼容广播警告，返回相应的布尔对象
  if (getBackCompatBroadcastWarn())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setBackcompatKeepdimWarn(
    PyObject* module,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔类型，如果不是，抛出错误信息
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_backcompat_keepdim_warn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置是否保留向后兼容的保持维度警告
  setBackCompatKeepdimWarn(arg == Py_True);
  // 返回空对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getBackcompatKeepdimWarn(
    PyObject* module,
    PyObject* noargs) {
  // 检查是否需要向后兼容保持维度警告，返回相应的布尔对象
  if (getBackCompatKeepdimWarn())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_hasDistributed(PyObject* _unused, PyObject* noargs) {
#ifdef USE_DISTRIBUTED
  // 如果定义了 USE_DISTRIBUTED 宏，则返回真值
  Py_RETURN_TRUE;
#else
  // 否则返回假值
  Py_RETURN_FALSE;
#endif
}

static PyObject* THPModule_showConfig(PyObject* module, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回当前 ATen 库的配置信息字符串
  return THPUtils_packString(at::show_config());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_cxxFlags(PyObject* module, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回当前编译 ATen 库时使用的 C++ 编译标志字符串
  return THPUtils_packString(at::get_cxx_flags());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_parallelInfo(PyObject* module, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回 ATen 库的并行信息字符串
  return THPUtils_packString(at::get_parallel_info());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getCpuCapability(
    PyObject* module,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回当前 CPU 的计算能力字符串
  return THPUtils_packString(at::get_cpu_capability());
  END_HANDLE_TH_ERRORS
}

void DLPack_Capsule_Destructor(PyObject* data) {
  // 如果 PyCapsule 不是有效的 "dltensor" 类型，提前返回
  if (C10_LIKELY(!PyCapsule_IsValid(data, "dltensor"))) {
    // 提前退出，参考 DLPack 规范：如果消费库将胶囊重命名为其他内容，则它们拥有它，我们无需执行任何操作
    return;
  }
  HANDLE_TH_ERRORS
  // 获取 DLManagedTensor 指针
  DLManagedTensor* dlMTensor =
      (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  // 调用 dlMTensor 的删除函数进行清理
  dlMTensor->deleter(dlMTensor);
  END_HANDLE_TH_ERRORS_RET()
}

PyObject* THPModule_toDLPack(PyObject* _unused, PyObject* data) {
  HANDLE_TH_ERRORS
  // 检查输入数据是否为 Tensor 类型
  TORCH_CHECK(THPVariable_Check(data), "data must be a Tensor");
  // 转换 PyTorch 的 Tensor 到 DLManagedTensor，并封装为 PyCapsule
  DLManagedTensor* dlMTensor = at::toDLPack(THPVariable_Unpack(data));
  // 返回 DLManagedTensor 的 PyCapsule 对象
  return PyCapsule_New(dlMTensor, "dltensor", DLPack_Capsule_Destructor);
  END_HANDLE_TH_ERRORS
}
PyObject* THPModule_fromDLPack(PyObject* _unused, PyObject* data) {
  // 导入 torch::autograd 命名空间
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  // 调用 torch::utils::tensor_fromDLPack 函数将 DLPack 数据转换为 PyTorch Tensor
  auto tensor = torch::utils::tensor_fromDLPack(data);
  // 将 PyTorch Tensor 封装成 THPVariable 对象返回给 Python
  return THPVariable_Wrap(tensor);
  END_HANDLE_TH_ERRORS
}

PyObject* THModule_getCppBacktrace(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // 定义需要跳过的堆栈帧数和最大堆栈帧数
  size_t frames_to_skip = 0;
  size_t maximum_number_of_frames = 0;
  // 解析 Python 参数，获取 frames_to_skip 和 maximum_number_of_frames 的值
  if (!PyArg_ParseTuple(
          args, "LL", &frames_to_skip, &maximum_number_of_frames)) {
    return nullptr;
  }
  // 获取 C++ 的堆栈回溯信息，并返回包装成字符串的结果给 Python
  return THPUtils_packString(
      c10::get_backtrace(frames_to_skip, maximum_number_of_frames, true));
  END_HANDLE_TH_ERRORS
}

static PyObject* THModule_rename_privateuse1_backend(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为字符串
  TORCH_CHECK(
      THPUtils_checkString(arg),
      "_rename_privateuse1_backend expects a str, but got ",
      THPUtils_typename(arg));
  // 将 Python 字符串解包成 C++ 标准字符串
  const std::string backend_name = THPUtils_unpackString(arg);
  // 注册一个私有的后端使用指定名称
  c10::register_privateuse1_backend(backend_name);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THModule_get_privateuse1_backend_name(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 获取当前私有后端的名称并返回为 Python 字符串
  return THPUtils_packString(c10::get_privateuse1_backend());
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_setAllowTF32CuDNN(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔类型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_allow_tf32_cublas expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中的 TF32 CuDNN 允许状态
  at::globalContext().setAllowTF32CuDNN(arg == Py_True);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_allowTF32CuDNN(PyObject* _unused, PyObject* noargs) {
  // 检查全局上下文中 TF32 CuDNN 允许状态，返回对应的 Python 布尔值
  if (at::globalContext().allowTF32CuDNN())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_setFloat32MatmulPrecision(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为字符串类型
  TORCH_CHECK(
      THPUtils_checkString(arg),
      "set_float32_matmul_precision expects a str, "
      "but got ",
      THPUtils_typename(arg));
  // 解包 Python 字符串成 C++ 标准字符串
  std::string s = THPUtils_unpackString(arg);
  // 设置全局上下文中的浮点32位矩阵乘法精度
  at::globalContext().setFloat32MatmulPrecision(s);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_float32MatmulPrecision(
    PyObject* _unused,
    PyObject* noargs) {
  // 获取当前全局上下文中的浮点32位矩阵乘法精度，并转换成相应的字符串返回给 Python
  std::string s = "highest";
  auto p = at::globalContext().float32MatmulPrecision();
  if (p == at::Float32MatmulPrecision::HIGH) {
    s = "high";
  } else if (p == at::Float32MatmulPrecision::MEDIUM) {
    s = "medium";
  }
  return THPUtils_packString(s);
}

PyObject* THPModule_setSDPUseFlash(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔类型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中 SDP 使用 Flash 的状态
  at::globalContext().setSDPUseFlash(arg == Py_True);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_userEnabledFlashSDP(PyObject* _unused, PyObject* noargs) {
  // 检查全局上下文中用户是否启用了 Flash SDP，并返回对应的 Python 布尔值
  if (at::globalContext().userEnabledFlashSDP())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}
    # 使用 Python/C API 返回一个布尔值 False 的对象
    Py_RETURN_FALSE;
PyObject* THPModule_setSDPUseMemEfficient(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中的 SDP 内存高效标志位
  at::globalContext().setSDPUseMemEfficient(arg == Py_True);
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* userEnabledMemEfficientSDP(PyObject* _unused, PyObject* noargs) {
  // 检查是否启用了 SDP 内存高效模式
  if (at::globalContext().userEnabledMemEfficientSDP())
    // 返回 Python 中的 True 对象
    Py_RETURN_TRUE;
  else
    // 返回 Python 中的 False 对象
    Py_RETURN_FALSE;
}

PyObject* THPModule_setSDPUseMath(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中的 SDP 数学模式标志位
  at::globalContext().setSDPUseMath(arg == Py_True);
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_userEnabledMathSDP(PyObject* _unused, PyObject* noargs) {
  // 检查是否启用了 SDP 数学模式
  if (at::globalContext().userEnabledMathSDP())
    // 返回 Python 中的 True 对象
    Py_RETURN_TRUE;
  else
    // 返回 Python 中的 False 对象
    Py_RETURN_FALSE;
}

PyObject* THPModule_setSDPUseOverrideable(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_overrideable expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中的 SDP 可重写标志位
  at::globalContext().setSDPUseOverrideable(arg == Py_True);
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_userEnabledOverrideableSDP(
    PyObject* _unused,
    PyObject* noargs) {
  // 检查是否启用了可重写的 SDP
  if (at::globalContext().userEnabledOverrideableSDP())
    // 返回 Python 中的 True 对象
    Py_RETURN_TRUE;
  else
    // 返回 Python 中的 False 对象
    Py_RETURN_FALSE;
}

PyObject* THPModule_setSDPUseCuDNN(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_cudnn expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  // 设置全局上下文中的 SDP 使用 CuDNN 标志位
  at::globalContext().setSDPUseCuDNN(arg == Py_True);
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_userEnabledCuDNNSDP(PyObject* _unused, PyObject* noargs) {
  // 检查是否启用了 CuDNN 的 SDP
  if (at::globalContext().userEnabledCuDNNSDP())
    // 返回 Python 中的 True 对象
    Py_RETURN_TRUE;
  else
    // 返回 Python 中的 False 对象
    Py_RETURN_FALSE;
}

PyObject* THPModule_setUserEnabledCuDNN(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_enabled_cudnn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中用户是否启用 CuDNN 的标志位
  at::globalContext().setUserEnabledCuDNN(arg == Py_True);
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_userEnabledCuDNN(PyObject* _unused, PyObject* noargs) {
  // 检查用户是否启用了 CuDNN
  if (at::globalContext().userEnabledCuDNN())
    // 返回 Python 中的 True 对象
    Py_RETURN_TRUE;
  else
    // 返回 Python 中的 False 对象
    Py_RETURN_FALSE;
}

PyObject* THPModule_setUserEnabledMkldnn(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_enabled_mkldnn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中用户是否启用 MKL-DNN 的标志位
  at::globalContext().setUserEnabledMkldnn(arg == Py_True);
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
PyObject* THPModule_userEnabledMkldnn(PyObject* _unused, PyObject* noargs) {
  // 检查是否启用了用户自定义的 MKLDNN
  if (at::globalContext().userEnabledMkldnn())
    // 返回 Python 中的 True
    Py_RETURN_TRUE;
  else
    // 返回 Python 中的 False
    Py_RETURN_FALSE;
}

PyObject* THPModule_setDeterministicCuDNN(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔类型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_deterministic_cudnn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中 CuDNN 的确定性模式
  at::globalContext().setDeterministicCuDNN(arg == Py_True);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_deterministicCuDNN(PyObject* _unused, PyObject* noargs) {
  // 检查当前是否启用了 CuDNN 的确定性模式
  if (at::globalContext().deterministicCuDNN())
    // 返回 Python 中的 True
    Py_RETURN_TRUE;
  else
    // 返回 Python 中的 False
    Py_RETURN_FALSE;
}

PyObject* THPModule_setDeterministicMkldnn(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔类型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_deterministic_mkldnn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中 MKLDNN 的确定性模式
  at::globalContext().setDeterministicMkldnn(arg == Py_True);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_deterministicMkldnn(PyObject* _unused, PyObject* noargs) {
  // 检查当前是否启用了 MKLDNN 的确定性模式
  if (at::globalContext().deterministicMkldnn())
    // 返回 Python 中的 True
    Py_RETURN_TRUE;
  else
    // 返回 Python 中的 False
    Py_RETURN_FALSE;
}

PyObject* THPModule_setDeterministicAlgorithms(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"_set_deterministic_algorithms(bool mode, *, bool warn_only=False)"});
  torch::ParsedArgs<2> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  // 解析参数中的模式和是否仅警告的标志
  bool mode = r.toBool(0);
  bool warn_only = r.toBool(1);
  // 设置全局上下文中算法的确定性模式
  at::globalContext().setDeterministicAlgorithms(mode, warn_only);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_deterministicAlgorithms(
    PyObject* _unused,
    PyObject* noargs) {
  // 检查当前是否启用了算法的确定性模式
  if (at::globalContext().deterministicAlgorithms()) {
    // 返回 Python 中的 True
    Py_RETURN_TRUE;
  }
  // 返回 Python 中的 False
  Py_RETURN_FALSE;
}

PyObject* THPModule_deterministicAlgorithmsWarnOnly(
    PyObject* _unused,
    PyObject* noargs) {
  // 检查当前是否仅对算法的确定性模式进行警告
  if (at::globalContext().deterministicAlgorithmsWarnOnly()) {
    // 返回 Python 中的 True
    Py_RETURN_TRUE;
  }
  // 返回 Python 中的 False
  Py_RETURN_FALSE;
}

PyObject* THPModule_setDeterministicFillUninitializedMemory(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔类型
  TORCH_CHECK(
      PyBool_Check(arg), "expected a bool, but got ", THPUtils_typename(arg));
  // 设置全局上下文中是否填充未初始化内存的确定性模式
  at::globalContext().setDeterministicFillUninitializedMemory(arg == Py_True);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_deterministicFillUninitializedMemory(
    PyObject* _unused,
    PyObject* noargs) {
  // 检查当前是否启用了填充未初始化内存的确定性模式
  if (at::globalContext().deterministicFillUninitializedMemory())
    // 返回 Python 中的 True
    Py_RETURN_TRUE;
  else
    // 返回 Python 中的 False
    Py_RETURN_FALSE;
}
// 设置用户是否启用 NNPACK，接受一个布尔型参数
PyObject* THPModule_setUserEnabledNNPACK(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_enabled_NNPACK expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中 NNPACK 的启用状态
  at::globalContext().setUserEnabledNNPACK(arg == Py_True);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 返回当前用户是否启用 NNPACK 的布尔值
PyObject* THPModule_userEnabledNNPACK(PyObject* _unused, PyObject* noargs) {
  // 检查全局上下文中 NNPACK 是否被启用
  if (at::globalContext().userEnabledNNPACK())
    // 返回 Python 的 True 对象
    Py_RETURN_TRUE;
  else
    // 返回 Python 的 False 对象
    Py_RETURN_FALSE;
}

// 设置是否总是发出警告，接受一个布尔型参数
PyObject* THPModule_setWarnAlways(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "setWarnOnlyOnce expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置是否总是发出警告
  c10::WarningUtils::set_warnAlways(arg == Py_True);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 返回是否总是发出警告的布尔值
PyObject* THPModule_warnAlways(PyObject* _unused, PyObject* noargs) {
  // 检查是否总是发出警告
  if (c10::WarningUtils::get_warnAlways()) {
    // 返回 Python 的 True 对象
    Py_RETURN_TRUE;
  }
  // 返回 Python 的 False 对象
  Py_RETURN_FALSE;
}

// 用于测试 C++ 到 Python 警告转换的测试函数，发出一条警告消息
PyObject* THPModule_warn(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 发出一条警告消息
  TORCH_WARN("Test message for TORCH_WARN");
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 用于测试 C++ 到 Python 警告转换的测试函数，发出一条过时警告消息
PyObject* THPModule_warnDeprecation(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 发出一条过时警告消息
  TORCH_WARN_DEPRECATION("Test message for TORCH_WARN_DEPRECATION");
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 设置是否启用 CuDNN 的基准测试，接受一个布尔型参数
PyObject* THPModule_setBenchmarkCuDNN(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_benchmark_cudnn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中 CuDNN 的基准测试状态
  at::globalContext().setBenchmarkCuDNN(arg == Py_True);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 返回当前是否启用 CuDNN 的基准测试的布尔值
PyObject* THPModule_benchmarkCuDNN(PyObject* _unused, PyObject* noargs) {
  // 检查全局上下文中 CuDNN 的基准测试是否启用
  if (at::globalContext().benchmarkCuDNN()) {
    // 返回 Python 的 True 对象
    Py_RETURN_TRUE;
  }
  // 返回 Python 的 False 对象
  Py_RETURN_FALSE;
}

// 设置是否允许 TF32 CuBLAS，接受一个布尔型参数
PyObject* THPModule_setAllowTF32CuBLAS(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_allow_tf32_cublas expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中是否允许 TF32 CuBLAS
  at::globalContext().setAllowTF32CuBLAS(arg == Py_True);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 返回当前是否允许 TF32 CuBLAS 的布尔值
PyObject* THPModule_allowTF32CuBLAS(PyObject* _unused, PyObject* noargs) {
  // 检查全局上下文中是否允许 TF32 CuBLAS
  if (at::globalContext().allowTF32CuBLAS()) {
    // 返回 Python 的 True 对象
    Py_RETURN_TRUE;
  }
  // 返回 Python 的 False 对象
  Py_RETURN_FALSE;
}

// 设置是否允许 FP16 Reducation CuBLAS，接受一个布尔型参数
PyObject* THPModule_setAllowFP16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_allow_fp16_reduction_cublas expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中是否允许 FP16 Reducation CuBLAS
  at::globalContext().setAllowFP16ReductionCuBLAS(arg == Py_True);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 返回当前是否允许 FP16 Reduction CuBLAS 的布尔值
PyObject* THPModule_allowFP16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* noargs) {
  // 检查全局上下文中是否允许 FP16 Reduction CuBLAS
  if (at::globalContext().allowFP16ReductionCuBLAS()) {
    // 返回 Python 的 True 对象
    Py_RETURN_TRUE;
  }
  // 返回 Python 的 False 对象
  Py_RETURN_FALSE;
}
    # 返回Python中的True对象，表示函数执行成功
    Py_RETURN_TRUE;
  }
  # 返回Python中的False对象，表示函数执行失败或未满足条件
  Py_RETURN_FALSE;
PyObject* THPModule_setAllowBF16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔类型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_allow_bf16_reduction_cublas expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中的 CuBLAS BF16 降维标志
  at::globalContext().setAllowBF16ReductionCuBLAS(arg == Py_True);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_allowBF16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* noargs) {
  // 如果允许 CuBLAS BF16 降维，则返回 True；否则返回 False
  if (at::globalContext().allowBF16ReductionCuBLAS()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_setAllowFP16ReductionCPU(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔类型
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_allow_fp16_reduction_cpu expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中的 CPU FP16 降维标志
  at::globalContext().setAllowFP16ReductionCPU(arg == Py_True);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_allowFP16ReductionCPU(PyObject* _unused, PyObject* noargs) {
  // 如果允许 CPU FP16 降维，则返回 True；否则返回 False
  if (at::globalContext().allowFP16ReductionCPU()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_setFlushDenormal(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔类型
  TORCH_CHECK(
      PyBool_Check(arg),
      "flush_denormal expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中的 denormal flush 标志
  if (!at::globalContext().setFlushDenormal(arg == Py_True)) {
    // 如果设置失败，则返回 False
    Py_RETURN_FALSE;
  };
  // 设置成功，则返回 True
  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getDefaultDtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 获取默认的张量数据类型并返回其 THP 类型
  auto scalar_type = torch::tensors::get_default_scalar_type();
  return Py_NewRef(torch::getTHPDtype(scalar_type));
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getDefaultDevice(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 获取默认的设备类型名称并返回小写格式的字符串
  return THPUtils_packString(c10::DeviceTypeName(
      dispatchKeyToDeviceType(torch::tensors::get_default_dispatch_key()),
      /*lower_case=*/true));
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_setQEngine(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为长整型
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "set_qengine expects an int, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中的量化引擎
  auto qengine = THPUtils_unpackLong(arg);
  at::globalContext().setQEngine(static_cast<at::QEngine>(qengine));
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_qEngine(PyObject* _unused, PyObject* noargs) {
  // 返回当前全局上下文中的量化引擎的整数表示
  return THPUtils_packInt64(
      static_cast<int64_t>(at::globalContext().qEngine()));
}

PyObject* THPModule_supportedQEngines(PyObject* _unused, PyObject* noargs) {
  // 获取支持的量化引擎列表并封装成 Python 列表返回
  auto qengines = at::globalContext().supportedQEngines();
  auto list =
      THPObjectPtr(PyList_New(static_cast<Py_ssize_t>(qengines.size())));
  if (!list)
    return nullptr;
  for (const auto i : c10::irange(qengines.size())) {
    PyObject* i64 = THPUtils_packInt64(static_cast<int64_t>(qengines[i]));
    if (!i64)
      return nullptr;
    PyList_SET_ITEM(list.get(), i, i64);
  }
  return list.release();
}
PyObject* THPModule_isEnabledXNNPACK(PyObject* _unused, PyObject* noargs) {
  // 检查全局上下文是否启用了XNNPACK，并返回对应的Python布尔值
  if (at::globalContext().isXNNPACKAvailable())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_setCheckSparseTensorInvariants(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为布尔类型，否则抛出异常
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_check_sparse_tensor_invariants expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  // 设置全局上下文中是否检查稀疏张量不变量的标志位
  at::globalContext().setCheckSparseTensorInvariants(arg == Py_True);
  // 返回None对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_checkSparseTensorInvariants(
    PyObject* _unused,
    PyObject* noargs) {
  // 检查全局上下文中是否开启了稀疏张量不变量的检查，并返回对应的Python布尔值
  if (at::globalContext().checkSparseTensorInvariants())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_willEngineExecuteNode(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查传入的参数是否为THPFunction或THPCppFunction类型，否则抛出异常
  bool isTHPFunction = THPFunction_Check(arg);
  bool isTHPCppFunction = torch::autograd::THPCppFunction_Check(arg);
  TORCH_CHECK(
      isTHPFunction || isTHPCppFunction,
      "_will_engine_execute_node expects an grad_fn, "
      "but got ",
      THPUtils_typename(arg));
  // 获取当前图任务的执行信息
  const auto exec_info = torch::autograd::get_current_graph_task_exec_info();
  TORCH_CHECK(
      exec_info,
      "_get_should_execute_nodes should only be called during the backward pass");
  torch::autograd::Node* node = nullptr;
  std::shared_ptr<torch::autograd::Node> node_sp;
  if (isTHPFunction) {
    // 如果是THPFunction类型，获取其关联的Node对象
    node_sp = ((THPFunction*)arg)->cdata.lock();
    node = node_sp.get();
  } else {
    // 如果是THPCppFunction类型，获取其关联的Node对象
    node = ((torch::autograd::THPCppFunction*)arg)->cdata.get();
  }
  // 获取当前图任务中所有节点的集合
  const auto nodes_in_graph =
      torch::autograd::get_current_graph_task_nodes_in_graph();
  // 检查传入的Node对象是否在当前图任务的节点集合中
  bool ret = nodes_in_graph->find(node) != nodes_in_graph->end();
  if (ret && !exec_info->empty()) {
    // 如果节点存在且执行信息非空，进一步检查节点的执行状态
    auto it = exec_info->find(node);
    if (it == exec_info->end() || !it->second.should_execute()) {
      ret = false;
    } else {
      // 如果节点为叶节点且存在捕获状态，则抛出异常
      TORCH_CHECK(
          !(node->topological_nr() == 0 && it->second.captures_),
          "A leaf node was passed to _will_engine_execute_node but we are "
          "currently running autograd.grad(). This is currently not supported.");
    }
  }
  // 返回节点是否执行的Python布尔值
  if (ret) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getCurrentGraphTaskExecutionOrder(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 获取当前图任务中节点的执行顺序，并返回为Python列表
  std::vector<torch::autograd::Node*> nodes =
      torch::autograd::get_current_graph_task_execution_order();
  TORCH_CHECK(
      !nodes.empty(),
      "_current_graph_task_execution_order should only be called during the backward pass");
  auto list = THPObjectPtr(PyList_New(static_cast<Py_ssize_t>(nodes.size())));
  if (!list)
    return nullptr;
  for (const auto i : c10::irange(nodes.size())) {
    // 将节点对象转换为Python对象，并加入到列表中
    // 注意：此节点对象在后向传播过程中保证有效
    PyObject* pyobj_node =
        torch::autograd::functionToPyObject(nodes[i]->getptr());
        PyList_SET_ITEM(list, i, pyobj_node);
    // 使用 PyList_SET_ITEM 将 pyobj_node 设置为 list 中的第 i 个元素
    PyList_SET_ITEM(list.get(), i, pyobj_node);
  }
  // 释放 list 对象的所有权，并返回其指针
  return list.release();
  // 在处理异常时结束，进行错误处理
  END_HANDLE_TH_ERRORS
}

// 获取当前图任务的ID并封装成 Python 对象返回
PyObject* THPModule_getCurrentGraphTaskId(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(torch::autograd::get_current_graph_task_id());
  END_HANDLE_TH_ERRORS
}

// 获取当前节点并封装成 Python 对象返回
PyObject* THPModule_getCurrentNode(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return torch::autograd::functionToPyObject(
      torch::autograd::get_current_node());
  END_HANDLE_TH_ERRORS
}

// 设置默认的移动端 CPU 分配器
PyObject* THPModule_setDefaultMobileCPUAllocator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::globalContext().setDefaultMobileCPUAllocator();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 取消设置默认的移动端 CPU 分配器
PyObject* THPModule_unsetDefaultMobileCPUAllocator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::globalContext().unsetDefaultMobileCPUAllocator();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 增加 Vmap 模式的嵌套层数并返回结果
static PyObject* THPModule_vmapmode_increment_nesting(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::impl::VmapMode::increment_nesting());
  END_HANDLE_TH_ERRORS
}

// 减少 Vmap 模式的嵌套层数并返回结果
static PyObject* THPModule_vmapmode_decrement_nesting(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::impl::VmapMode::decrement_nesting());
  END_HANDLE_TH_ERRORS
}

// 设置是否显示 Vmap 回退警告信息
static PyObject* THPModule_set_display_vmap_fallback_warnings_mode(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "enabled must be a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setDisplayVmapFallbackWarnings(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 检查是否启用了 Vmap 回退警告信息
static PyObject* THPModule_are_vmap_fallback_warnings_enabled(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  if (at::globalContext().areVmapFallbackWarningsEnabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// Torch 方法定义数组，包含了各种 Torch C++ 扩展函数
static PyMethodDef TorchMethods[] = { // NOLINT
    {"_initExtension", THPModule_initExtension, METH_O, nullptr},
    {"_autograd_init", THPAutograd_initExtension, METH_NOARGS, nullptr},
    {"_add_docstr", THPModule_addDocStr, METH_VARARGS, nullptr},
    {"_swap_tensor_impl", THPModule_swap_tensor_impl, METH_VARARGS, nullptr},
    {"_init_names", THPModule_initNames, METH_O, nullptr},
    {"_has_distributed", THPModule_hasDistributed, METH_NOARGS, nullptr},
    {"_set_default_tensor_type",
     THPModule_setDefaultTensorType,
     METH_O,
     nullptr},
    {"_set_default_dtype", THPModule_setDefaultDtype, METH_O, nullptr},
    {"_infer_size", THPModule_inferSize, METH_VARARGS, nullptr},
    {"_abort", THPModule_abort, METH_NOARGS, nullptr},
    {"_crash_if_csrc_asan", THPModule_crashIfCsrcASAN, METH_O, nullptr},
    {"_crash_if_csrc_ubsan", THPModule_crashIfCsrcUBSAN, METH_O, nullptr},
    {"_crash_if_vptr_ubsan", THPModule_crashIfvptrUBSAN, METH_NOARGS, nullptr},
    {"_crash_if_aten_asan", THPModule_crashIfATenASAN, METH_O, nullptr},
};
    {"_crash_if_debug_asserts_fail",
     THPModule_crashIfDebugAssertsFail,
     METH_O,
     nullptr},
    # 注册一个名为 "_crash_if_debug_asserts_fail" 的函数，使用 THPModule_crashIfDebugAssertsFail 函数处理，接受一个参数，返回值为空
    {"_show_config", THPModule_showConfig, METH_NOARGS, nullptr},
    # 注册一个名为 "_show_config" 的函数，使用 THPModule_showConfig 函数处理，不接受参数，返回值为空
    {"_cxx_flags", THPModule_cxxFlags, METH_NOARGS, nullptr},
    # 注册一个名为 "_cxx_flags" 的函数，使用 THPModule_cxxFlags 函数处理，不接受参数，返回值为空
    {"_parallel_info", THPModule_parallelInfo, METH_NOARGS, nullptr},
    # 注册一个名为 "_parallel_info" 的函数，使用 THPModule_parallelInfo 函数处理，不接受参数，返回值为空
    {"_get_cpu_capability", THPModule_getCpuCapability, METH_NOARGS, nullptr},
    # 注册一个名为 "_get_cpu_capability" 的函数，使用 THPModule_getCpuCapability 函数处理，不接受参数，返回值为空
    {"_set_backcompat_broadcast_warn",
     THPModule_setBackcompatBroadcastWarn,
     METH_O,
     nullptr},
    # 注册一个名为 "_set_backcompat_broadcast_warn" 的函数，使用 THPModule_setBackcompatBroadcastWarn 函数处理，接受一个参数，返回值为空
    {"_get_backcompat_broadcast_warn",
     THPModule_getBackcompatBroadcastWarn,
     METH_NOARGS,
     nullptr},
    # 注册一个名为 "_get_backcompat_broadcast_warn" 的函数，使用 THPModule_getBackcompatBroadcastWarn 函数处理，不接受参数，返回值为空
    {"_set_backcompat_keepdim_warn",
     THPModule_setBackcompatKeepdimWarn,
     METH_O,
     nullptr},
    # 注册一个名为 "_set_backcompat_keepdim_warn" 的函数，使用 THPModule_setBackcompatKeepdimWarn 函数处理，接受一个参数，返回值为空
    {"_get_backcompat_keepdim_warn",
     THPModule_getBackcompatKeepdimWarn,
     METH_NOARGS,
     nullptr},
    # 注册一个名为 "_get_backcompat_keepdim_warn" 的函数，使用 THPModule_getBackcompatKeepdimWarn 函数处理，不接受参数，返回值为空
    {"get_num_threads", THPModule_getNumThreads, METH_NOARGS, nullptr},
    # 注册一个名为 "get_num_threads" 的函数，使用 THPModule_getNumThreads 函数处理，不接受参数，返回值为空
    {"set_num_threads", THPModule_setNumThreads, METH_O, nullptr},
    # 注册一个名为 "set_num_threads" 的函数，使用 THPModule_setNumThreads 函数处理，接受一个参数，返回值为空
    {"get_num_interop_threads",
     THPModule_getNumInteropThreads,
     METH_NOARGS,
     nullptr},
    # 注册一个名为 "get_num_interop_threads" 的函数，使用 THPModule_getNumInteropThreads 函数处理，不接受参数，返回值为空
    {"set_num_interop_threads",
     THPModule_setNumInteropThreads,
     METH_O,
     nullptr},
    # 注册一个名为 "set_num_interop_threads" 的函数，使用 THPModule_setNumInteropThreads 函数处理，接受一个参数，返回值为空
    {"_get_flash_sdp_enabled",
     THPModule_userEnabledFlashSDP,
     METH_NOARGS,
     nullptr},
    # 注册一个名为 "_get_flash_sdp_enabled" 的函数，使用 THPModule_userEnabledFlashSDP 函数处理，不接受参数，返回值为空
    {"_set_sdp_use_flash", THPModule_setSDPUseFlash, METH_O, nullptr},
    # 注册一个名为 "_set_sdp_use_flash" 的函数，使用 THPModule_setSDPUseFlash 函数处理，接受一个参数，返回值为空
    {"_get_mem_efficient_sdp_enabled",
     userEnabledMemEfficientSDP,
     METH_NOARGS,
     nullptr},
    # 注册一个名为 "_get_mem_efficient_sdp_enabled" 的函数，使用 userEnabledMemEfficientSDP 函数处理，不接受参数，返回值为空
    {"_set_sdp_use_mem_efficient",
     THPModule_setSDPUseMemEfficient,
     METH_O,
     nullptr},
    # 注册一个名为 "_set_sdp_use_mem_efficient" 的函数，使用 THPModule_setSDPUseMemEfficient 函数处理，接受一个参数，返回值为空
    {"_get_math_sdp_enabled",
     THPModule_userEnabledMathSDP,
     METH_NOARGS,
     nullptr},
    # 注册一个名为 "_get_math_sdp_enabled" 的函数，使用 THPModule_userEnabledMathSDP 函数处理，不接受参数，返回值为空
    {"_set_sdp_use_math", THPModule_setSDPUseMath, METH_O, nullptr},
    # 注册一个名为 "_set_sdp_use_math" 的函数，使用 THPModule_setSDPUseMath 函数处理，接受一个参数，返回值为空
    {"_get_overrideable_sdp_enabled",
     THPModule_userEnabledOverrideableSDP,
     METH_NOARGS,
     nullptr},
    # 注册一个名为 "_get_overrideable_sdp_enabled" 的函数，使用 THPModule_userEnabledOverrideableSDP 函数处理，不接受参数，返回值为空
    {"_set_sdp_use_overrideable",
     THPModule_setSDPUseOverrideable,
     METH_O,
     nullptr},
    # 注册一个名为 "_set_sdp_use_overrideable" 的函数，使用 THPModule_setSDPUseOverrideable 函数处理，接受一个参数，返回值为空
    {"_get_cudnn_sdp_enabled",
     THPModule_userEnabledCuDNNSDP,
     METH_NOARGS,
     nullptr},
    # 注册一个名为 "_get_cudnn_sdp_enabled" 的函数，使用 THPModule_userEnabledCuDNNSDP 函数处理，不接受参数，返回值为空
    {"_set_sdp_use_cudnn", THPModule_setSDPUseCuDNN, METH_O, nullptr},
    # 注册一个名为 "_set_sdp_use_cudnn" 的函数，使用 THPModule_setSDPUseCuDNN 函数处理，接受一个参数，返回值为空
    {"_get_cudnn_enabled", THPModule_userEnabledCuDNN, METH_NOARGS, nullptr},
    # 注册一个名为 "_get_cudnn_enabled" 的函数，使用 THPModule_userEnabledCuDNN 函数处理，不接受参数，返回值为空
    {"_set_cudnn_enabled", THPModule_setUserEnabledCuDNN, METH_O, nullptr},
    # 注册一个名为 "_set_cudnn_enabled" 的函数，使用 THPModule_setUserEnabledCuDNN 函数处理，接受一个参数，返回值为空
    {"_get_mkldnn_enabled", THPModule_userEnabledMkldnn, METH_NOARGS, nullptr},
    # 注册一个名为 "_get_mkldnn_enabled" 的函数，使用 THPModule_userEnabledMkldnn 函数处理，不接受参数，返回值为空
    {"_set_mkldnn_enabled", THPModule_setUserEnabledMkldnn, METH_O, nullptr},
    # 注册一个名为 "_set_mkldnn_enabled" 的函数，使用 THPModule_setUserEnabledMkldnn 函数处理，接受一个参数，返回值为空
    {"_get_cudnn_allow_tf32", THPModule_allowTF32CuDNN, METH_NOARGS, nullptr},
    # 注册一个名为 "_get_cudnn_allow_tf32" 的函数，使用 THPModule_allowTF32CuDNN 函数处理，不接受参数，返回值为空
    {"_set_cudnn_allow_tf32", THPModule_setAllowTF32CuDNN, METH_O, nullptr},
    # 注册一个名为 "_set_cudnn_allow_tf32" 的函数，使用 THPModule_setAllowTF32CuDNN 函数处理，接受一个参数，返回值为空
    {"_get_cudnn_benchmark", THPModule_benchmarkCuDNN, METH_NOARGS, nullptr},
    # 注册一个名为 "_get_cudnn_benchmark" 的函数，使用 THPModule_benchmarkCuDNN 函数处理，不接受参数，返回值为空
    {"_set_cudnn_benchmark", THPModule_setBenchmarkCuDNN, METH_O, nullptr},
    # 注册一个名为 "_set_cudnn_benchmark" 的函数，使用 THPModule_setBenchmarkCuDNN 函数处理，接受一个参数，返回值为空
    {"_get_cudnn_deterministic",
     THPModule_deterministicCuDNN,
    {"_get_mkldnn_deterministic",
     THPModule_deterministicMkldnn,
     METH_NOARGS,
     nullptr},
    // 定义一个名为 "_get_mkldnn_deterministic" 的 C 函数，它在 Python 中映射到 THPModule_deterministicMkldnn，不接受任何参数
    {"_set_mkldnn_deterministic",
     THPModule_setDeterministicMkldnn,
     METH_O,
     nullptr},
    // 定义一个名为 "_set_mkldnn_deterministic" 的 C 函数，它在 Python 中映射到 THPModule_setDeterministicMkldnn，接受一个对象参数
    {"_get_deterministic_algorithms",
     THPModule_deterministicAlgorithms,
     METH_NOARGS,
     nullptr},
    // 定义一个名为 "_get_deterministic_algorithms" 的 C 函数，它在 Python 中映射到 THPModule_deterministicAlgorithms，不接受任何参数
    {"_get_deterministic_algorithms_warn_only",
     THPModule_deterministicAlgorithmsWarnOnly,
     METH_NOARGS,
     nullptr},
    // 定义一个名为 "_get_deterministic_algorithms_warn_only" 的 C 函数，它在 Python 中映射到 THPModule_deterministicAlgorithmsWarnOnly，不接受任何参数
    {"_set_deterministic_algorithms",
     castPyCFunctionWithKeywords(THPModule_setDeterministicAlgorithms),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    // 定义一个名为 "_set_deterministic_algorithms" 的 C 函数，它在 Python 中映射到 castPyCFunctionWithKeywords(THPModule_setDeterministicAlgorithms)，接受可变位置参数和关键字参数
    {"_get_deterministic_fill_uninitialized_memory",
     THPModule_deterministicFillUninitializedMemory,
     METH_NOARGS,
     nullptr},
    // 定义一个名为 "_get_deterministic_fill_uninitialized_memory" 的 C 函数，它在 Python 中映射到 THPModule_deterministicFillUninitializedMemory，不接受任何参数
    {"_set_deterministic_fill_uninitialized_memory",
     THPModule_setDeterministicFillUninitializedMemory,
     METH_O,
     nullptr},
    // 定义一个名为 "_set_deterministic_fill_uninitialized_memory" 的 C 函数，它在 Python 中映射到 THPModule_setDeterministicFillUninitializedMemory，接受一个对象参数
    {"_get_nnpack_enabled", THPModule_userEnabledNNPACK, METH_NOARGS, nullptr},
    // 定义一个名为 "_get_nnpack_enabled" 的 C 函数，它在 Python 中映射到 THPModule_userEnabledNNPACK，不接受任何参数
    {"_set_nnpack_enabled", THPModule_setUserEnabledNNPACK, METH_O, nullptr},
    // 定义一个名为 "_set_nnpack_enabled" 的 C 函数，它在 Python 中映射到 THPModule_setUserEnabledNNPACK，接受一个对象参数
    {"_get_warnAlways", THPModule_warnAlways, METH_NOARGS, nullptr},
    // 定义一个名为 "_get_warnAlways" 的 C 函数，它在 Python 中映射到 THPModule_warnAlways，不接受任何参数
    {"_set_warnAlways", THPModule_setWarnAlways, METH_O, nullptr},
    // 定义一个名为 "_set_warnAlways" 的 C 函数，它在 Python 中映射到 THPModule_setWarnAlways，接受一个对象参数
    {"_warn", THPModule_warn, METH_NOARGS, nullptr},
    // 定义一个名为 "_warn" 的 C 函数，它在 Python 中映射到 THPModule_warn，不接受任何参数
    {"_warn_deprecation", THPModule_warnDeprecation, METH_NOARGS, nullptr},
    // 定义一个名为 "_warn_deprecation" 的 C 函数，它在 Python 中映射到 THPModule_warnDeprecation，不接受任何参数
    {"_get_cublas_allow_tf32", THPModule_allowTF32CuBLAS, METH_NOARGS, nullptr},
    // 定义一个名为 "_get_cublas_allow_tf32" 的 C 函数，它在 Python 中映射到 THPModule_allowTF32CuBLAS，不接受任何参数
    {"_set_cublas_allow_tf32", THPModule_setAllowTF32CuBLAS, METH_O, nullptr},
    // 定义一个名为 "_set_cublas_allow_tf32" 的 C 函数，它在 Python 中映射到 THPModule_setAllowTF32CuBLAS，接受一个对象参数
    {"_get_float32_matmul_precision",
     THPModule_float32MatmulPrecision,
     METH_NOARGS,
     nullptr},
    // 定义一个名为 "_get_float32_matmul_precision" 的 C 函数，它在 Python 中映射到 THPModule_float32MatmulPrecision，不接受任何参数
    {"_set_float32_matmul_precision",
     THPModule_setFloat32MatmulPrecision,
     METH_O,
     nullptr},
    // 定义一个名为 "_set_float32_matmul_precision" 的 C 函数，它在 Python 中映射到 THPModule_setFloat32MatmulPrecision，接受一个对象参数
    {"_get_cublas_allow_fp16_reduced_precision_reduction",
     THPModule_allowFP16ReductionCuBLAS,
     METH_NOARGS,
     nullptr},
    // 定义一个名为 "_get_cublas_allow_fp16_reduced_precision_reduction" 的 C 函数，它在 Python 中映射到 THPModule_allowFP16ReductionCuBLAS，不接受任何参数
    {"_set_cublas_allow_fp16_reduced_precision_reduction",
     THPModule_setAllowFP16ReductionCuBLAS,
     METH_O,
     nullptr},
    // 定义一个名为 "_set_cublas_allow_fp16_reduced_precision_reduction" 的 C 函数，它在 Python 中映射到 THPModule_setAllowFP16ReductionCuBLAS，接受一个对象参数
    {"_get_cublas_allow_bf16_reduced_precision_reduction",
     THPModule_allowBF16ReductionCuBLAS,
     METH_NOARGS,
     nullptr},
    // 定义一个名为 "_get_cublas_allow_bf16_reduced_precision_reduction" 的 C 函数，它在 Python 中映射到 THPModule_allowBF16ReductionCuBLAS，不接受任何参数
    {"_set_cublas_allow_bf16_reduced_precision_reduction",
     THPModule_setAllowBF16ReductionCuBLAS,
     METH_O,
     nullptr},
    // 定义一个名为 "_set_cublas_allow_bf16_reduced_precision_reduction" 的 C 函数，它在 Python 中映射到 THPModule_setAllowBF16ReductionCuBLAS，接受一个对象参数
    {"_get_cpu_allow_fp16_reduced_precision_reduction",
     THPModule_allowFP16ReductionCPU,
     METH_NOARGS,
     nullptr},
    // 定义一个名为 "_get_cpu_allow_fp16_reduced_precision_reduction" 的 C 函数，它在 Python 中映射到 THPModule_allowFP16ReductionCPU，不接受任何参数
    {"_set_cpu_allow_fp16_reduced_precision_reduction",
     THPModule_setAllowFP16ReductionCPU,
     METH_O,
     nullptr},
    // 定义一个名为 "_set_cpu_allow_fp16_reduced_precision_reduction" 的 C 函数，它在 Python 中映射到 THPModule_setAllowFP16ReductionCPU，接受一个对象参数
    {"_vmapmode_increment_nesting",
     THPModule_vmapmode_increment_nesting,
     METH_NOARGS,
     nullptr},
    // 定义一个名为 "_vmapmode_increment_nesting" 的 C 函数，它在 Python 中映射到 THPModule_vmapmode_increment_nesting，不接受任何参数
    {"_vmapmode_decrement_nesting",
     THPModule_vmapmode_decrement_nesting,
     METH_NOARGS,
     nullptr},
    // 定义一个名为 "_vmapmode_decrement_nesting" 的 C 函数，它在 Python 中映射到 THPModule_vmapmode_decrement_nesting，不接受任何参数
    {"_debug_only_display_vmap_fallback_warnings",
     THPModule_set_display_vmap_fallback_warnings_mode,
     METH_O,
     nullptr},
    // 定义一个名为 "_debug_only_display_vmap_fallback_warnings" 的 C 函数，它在 Python 中映射到 THPModule_set_display_vmap_fallback_warnings_mode
    // 定义一个结构体数组，描述了Python方法名到C函数指针的映射关系
    {"_to_dlpack", THPModule_toDLPack, METH_O, nullptr},
    {"_from_dlpack", THPModule_fromDLPack, METH_O, nullptr},
    // 获取C++调用堆栈的函数指针及其参数类型信息
    {"_get_cpp_backtrace", THModule_getCppBacktrace, METH_VARARGS, nullptr},
    // 重命名私有使用的第一后端的函数指针及其参数类型信息
    {"_rename_privateuse1_backend",
     THModule_rename_privateuse1_backend,
     METH_O,
     nullptr},
    // 获取私有使用的第一后端名称的函数指针及其参数类型信息
    {"_get_privateuse1_backend_name",
     THModule_get_privateuse1_backend_name,
     METH_NOARGS,
     nullptr},
    // 设置刷新非规范数的函数指针及其参数类型信息
    {"set_flush_denormal", THPModule_setFlushDenormal, METH_O, nullptr},
    // 获取默认数据类型的函数指针及其参数类型信息
    {"get_default_dtype", THPModule_getDefaultDtype, METH_NOARGS, nullptr},
    // 获取默认设备的函数指针及其参数类型信息
    {"_get_default_device", THPModule_getDefaultDevice, METH_NOARGS, nullptr},
    // 获取量化引擎的函数指针及其参数类型信息
    {"_get_qengine", THPModule_qEngine, METH_NOARGS, nullptr},
    // 设置量化引擎的函数指针及其参数类型信息
    {"_set_qengine", THPModule_setQEngine, METH_O, nullptr},
    // 获取支持的量化引擎列表的函数指针及其参数类型信息
    {"_supported_qengines", THPModule_supportedQEngines, METH_NOARGS, nullptr},
    // 检查是否启用XNNPACK加速的函数指针及其参数类型信息
    {"_is_xnnpack_enabled", THPModule_isEnabledXNNPACK, METH_NOARGS, nullptr},
    // 设置检查稀疏张量不变性的函数指针及其参数类型信息
    {"_set_check_sparse_tensor_invariants",
     THPModule_setCheckSparseTensorInvariants,
     METH_O,
     nullptr},
    // 检查稀疏张量不变性的函数指针及其参数类型信息
    {"_check_sparse_tensor_invariants",
     THPModule_checkSparseTensorInvariants,
     METH_NOARGS,
     nullptr},
    // 预测引擎是否将执行节点的函数指针及其参数类型信息
    {"_will_engine_execute_node",
     THPModule_willEngineExecuteNode,
     METH_O,
     nullptr},
    // 获取当前图任务执行顺序的函数指针及其参数类型信息
    {"_current_graph_task_execution_order",
     THPModule_getCurrentGraphTaskExecutionOrder,
     METH_NOARGS,
     nullptr},
    // 获取当前图任务ID的函数指针及其参数类型信息
    {"_current_graph_task_id",
     THPModule_getCurrentGraphTaskId,
     METH_NOARGS,
     nullptr},
    // 获取当前自动求导节点的函数指针及其参数类型信息
    {"_current_autograd_node", THPModule_getCurrentNode, METH_NOARGS, nullptr},
    // 设置默认的移动端CPU分配器的函数指针及其参数类型信息
    {"_set_default_mobile_cpu_allocator",
     THPModule_setDefaultMobileCPUAllocator,
     METH_NOARGS,
     nullptr},
    // 取消默认的移动端CPU分配器的函数指针及其参数类型信息
    {"_unset_default_mobile_cpu_allocator",
     THPModule_unsetDefaultMobileCPUAllocator,
     METH_NOARGS,
     nullptr},
    // 检查是否启用Torch函数的函数指针及其参数类型信息
    {"_is_torch_function_enabled",
     THPModule_isEnabledTorchFunction,
     METH_NOARGS,
     nullptr},
    // 禁用Torch函数实现的函数指针及其参数类型信息
    {"_disabled_torch_function_impl",
     THPModule_disable_torch_function,
     METH_VARARGS,
     nullptr},
    // 禁用Torch分派实现的函数指针及其参数类型信息
    {"_disabled_torch_dispatch_impl",
     THPModule_disable_torch_dispatch,
     METH_VARARGS,
     nullptr},
    // 检查是否有Torch函数的函数指针及其参数类型信息
    {"_has_torch_function", THPModule_has_torch_function, METH_O, nullptr},
    // 检查是否有Torch一元函数的函数指针及其参数类型信息
    {"_has_torch_function_unary",
     THPModule_has_torch_function_unary,
     METH_O,
     nullptr},
    // 检查是否有Torch多元函数的函数指针及其参数类型信息
    {"_has_torch_function_variadic",
     (PyCFunction)(void (*)())THPModule_has_torch_function_variadic,
     METH_FASTCALL,
     nullptr},
    // 数组的结尾标记，指示结束
    {nullptr, nullptr, 0, nullptr}};
// 初始化 THCPStream 相关 Python 模块方法
void THCPStream_init(PyObject* module);

// 初始化 THCPEvent 相关 Python 模块方法
void THCPEvent_init(PyObject* module);

// 初始化 THCPGraph 相关 Python 模块方法
void THCPGraph_init(PyObject* module);

#ifdef USE_CUDA
// 返回 THCPModule 的 Python 方法定义数组
PyMethodDef* THCPModule_methods();

// torch::cuda 命名空间中初始化模块的 Python 接口
namespace torch::cuda {
void initModule(PyObject* module);
} // namespace torch::cuda
#endif

#ifdef USE_XPU
// 返回 THXPModule 的 Python 方法定义数组
PyMethodDef* THXPModule_methods();

// 初始化 THXPStream 相关 Python 模块方法
void THXPStream_init(PyObject* module);

// 初始化 THXPEvent 相关 Python 模块方法
void THXPEvent_init(PyObject* module);

// torch::xpu 命名空间中初始化模块的 Python 接口
namespace torch::xpu {
void initModule(PyObject* module);
} // namespace torch::xpu
#endif

#ifdef USE_ITT
// torch::profiler 命名空间中初始化 ITT 绑定的 Python 接口
namespace torch::profiler {
void initIttBindings(PyObject* module);
} // namespace torch::profiler
#endif

// 静态方法定义的数组
static std::vector<PyMethodDef> methods;

// 在 Python 中调用，无需锁定映射，保证在 GIL 下调用
static void LogAPIUsageOnceFromPython(const std::string& event) {
  static std::unordered_set<std::string> seen;
  // 检查事件是否已经记录过，若未记录则插入并调用 c10::LogAPIUsage
  if (!seen.count(event)) {
    seen.insert(event);
    c10::LogAPIUsage(event);
  }
}

// 在 Python 中记录 API 使用元数据的方法
static void LogAPIUsageMetadataFromPython(
    const std::string& event,
    const std::map<std::string, std::string>& metadata_map) {
  // 调用 c10::LogAPIUsageMetadata 记录 API 使用的元数据
  c10::LogAPIUsageMetadata(event, metadata_map);
}

// 弱引用 tensor 的类，用于测试 tensor 是否泄漏
class WeakTensorRef {
  c10::weak_intrusive_ptr<c10::TensorImpl> weakref_;

 public:
  // 构造函数，创建 tensor 的弱引用
  WeakTensorRef(const at::Tensor& t) : weakref_(t.getIntrusivePtr()) {}

  // 检查弱引用是否已过期（即 tensor 是否已释放）
  bool expired() {
    return weakref_.expired();
  }
};

// 导出的 C 函数，用于初始化模块
extern "C" C10_EXPORT PyObject* initModule();

// MSVC 中分离声明和定义以避免错误 C2491
// 初始化模块的函数定义
PyObject* initModule() {
  HANDLE_TH_ERRORS

  // 初始化 C10 日志系统
  c10::initLogging();
  // 设置异常处理程序
  c10::set_terminate_handler();
  // 惰性初始化线程数量
  at::internal::lazy_init_num_threads();

  // 记录 "torch.python.import" 的 API 使用
  C10_LOG_API_USAGE_ONCE("torch.python.import");

  // 定义 ASSERT_TRUE 宏，若条件不满足则返回 nullptr
#define ASSERT_TRUE(cmd) \
  if (!(cmd))            \
  return nullptr

  // 添加 Torch 相关方法定义到 methods 中
  THPUtils_addPyMethodDefs(methods, TorchMethods);
  // 添加 DataLoader 相关方法定义到 methods 中
  THPUtils_addPyMethodDefs(methods, DataLoaderMethods);
  // 添加 torch::autograd 的 Python 函数方法定义到 methods 中
  THPUtils_addPyMethodDefs(methods, torch::autograd::python_functions());
  // 添加 torch::multiprocessing 的 Python 函数方法定义到 methods 中
  THPUtils_addPyMethodDefs(methods, torch::multiprocessing::python_functions());
  // 添加 torch::mps 的 Python 函数方法定义到 methods 中
  THPUtils_addPyMethodDefs(methods, torch::mps::python_functions());
#ifdef USE_CUDA
  // 添加 THCPModule 的 Python 方法定义到 methods 中
  THPUtils_addPyMethodDefs(methods, THCPModule_methods());
#endif
#ifdef USE_XPU
  // 添加 THXPModule 的 Python 方法定义到 methods 中
  THPUtils_addPyMethodDefs(methods, THXPModule_methods());
#endif
#if defined(USE_DISTRIBUTED) && defined(USE_C10D)
  // 添加 torch::distributed::c10d 的 Python 函数方法定义到 methods 中
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::c10d::python_functions());
#ifndef _WIN32
  // 添加 torch::distributed::rpc 的 Python 函数方法定义到 methods 中
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::rpc::python_functions());
  // 添加 torch::distributed::autograd 的 Python 函数方法定义到 methods 中
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::autograd::python_functions());
  // 添加 torch::distributed::rpc::testing 的 Python 函数方法定义到 methods 中
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::rpc::testing::python_functions());
#endif
#endif

  // 定义一个静态的 PyModuleDef 结构体，用于描述 torchmodule 模块
  static struct PyModuleDef torchmodule = {
      PyModuleDef_HEAD_INIT, "torch._C", nullptr, -1, methods.data()};
  // 创建一个 Python 模块对象并将其赋值给 module
  module = PyModule_Create(&torchmodule);
  // 断言 module 不为空，确保成功创建 Python 模块对象
  ASSERT_TRUE(module);
  // 初始化 THPGenerator 模块
  ASSERT_TRUE(THPGenerator_init(module));
  // 初始化 THPException 模块
  ASSERT_TRUE(THPException_init(module));
  // 初始化 THPSize 模块
  THPSize_init(module);
  // 初始化 THPDtype 模块
  THPDtype_init(module);
  // 初始化 THPDTypeInfo 模块
  THPDTypeInfo_init(module);
  // 初始化 THPLayout 模块
  THPLayout_init(module);
  // 初始化 THPMemoryFormat 模块
  THPMemoryFormat_init(module);
  // 初始化 THPQScheme 模块
  THPQScheme_init(module);
  // 初始化 THPDevice 模块
  THPDevice_init(module);
  // 初始化 THPStream 模块
  THPStream_init(module);
  // 初始化 THPEvent 模块
  THPEvent_init(module);
  // 初始化 NodeBase 模块
  NodeBase_init(module);
  // 初始化 NodeIter 模块
  NodeIter_init(module);
  // 初始化 THPVariable 模块
  ASSERT_TRUE(THPVariable_initModule(module));
  // 初始化 THPFunction 模块
  ASSERT_TRUE(THPFunction_initModule(module));
  // 初始化 THPEngine 模块
  ASSERT_TRUE(THPEngine_initModule(module));
  // 初始化 ONNX 绑定，用于 JIT 的导出功能
  // 在 JIT 初始化之前，需要确保可以从 ONNX 中访问 OperatorExportTypes
  torch::onnx::initONNXBindings(module);
  // 初始化 autograd 中的枚举标签
  torch::autograd::initEnumTag(module);
  // 初始化 JIT 绑定
  torch::jit::initJITBindings(module);
  // 初始化监视器绑定
  torch::monitor::initMonitorBindings(module);
  // 初始化分发绑定
  torch::impl::dispatch::initDispatchBindings(module);
  // 初始化动态绑定
  torch::dynamo::initDynamoBindings(module);
  // 初始化 functorch 绑定
  torch::functorch::impl::initFuncTorchBindings(module);
  // 初始化吞吐量基准绑定
  torch::throughput_benchmark::initThroughputBenchmarkBindings(module);
  // 初始化 autograd 返回类型
  torch::autograd::initReturnTypes(module);
  // 初始化 autograd 中的 NN 函数
  torch::autograd::initNNFunctions(module);
  // 初始化 autograd 中的 FFT 函数
  torch::autograd::initFFTFunctions(module);
  // 初始化 autograd 中的线性代数函数
  torch::autograd::initLinalgFunctions(module);
  // 初始化 autograd 中的嵌套函数
  torch::autograd::initNestedFunctions(module);
  // 初始化 autograd 中的稀疏函数
  torch::autograd::initSparseFunctions(module);
  // 初始化 autograd 中的特殊函数
  torch::autograd::initSpecialFunctions(module);
  // 初始化旧版变量支持
  torch::autograd::init_legacy_variable(module);
  // 初始化分析器的 Python 绑定
  torch::profiler::initPythonBindings(module);
  // 初始化 Python 绑定
  torch::python::init_bindings(module);
  // 初始化懒惰绑定
  torch::lazy::initLazyBindings(module);
  // 初始化 AOTI 运行器绑定
  torch::inductor::initAOTIRunnerBindings(module);
#ifdef USE_ITT
  // 初始化 Intel VTune Profiler 绑定
  torch::profiler::initIttBindings(module);
#endif
#ifdef USE_CUDA
  // 如果使用 CUDA，则初始化 CUDA 模块
  torch::cuda::initModule(module);
#endif
#ifdef USE_XPU
  // 如果使用 XPU，则初始化 XPU 模块
  torch::xpu::initModule(module);
#endif
  // 初始化 MTIA 模块
  torch::mtia::initModule(module);
  // 初始化 CPU 模块
  torch::cpu::initModule(module);
  // 初始化详细绑定
  torch::initVerboseBindings(module);
  // 初始化 THPStorage 模块
  ASSERT_TRUE(THPStorage_init(module));

#ifdef USE_CUDA
  // 如果使用 CUDA，则初始化 CUDA 流模块
  // 这些模块只初始化基础类并将它们附加到库命名空间
  // 在导入 CUDA 模块之前，它们不会准备好真正使用
  // 但它们会在调用 C 之前定义 Python 类，所以这些行必须首先执行
  THCPStream_init(module);
  THCPEvent_init(module);
  THCPGraph_init(module);
#endif

#ifdef USE_XPU
  // 如果使用 XPU，则初始化 XPU 流模块
  THXPStream_init(module);
  THXPEvent_init(module);
#endif

  // 定义一个 lambda 函数，用于设置模块的属性
  auto set_module_attr =
      [&](const char* name, PyObject* v, bool incref = true) {
        // PyModule_AddObject 偷取引用
        if (incref) {
          Py_INCREF(v);
        }

        // 将对象 v 添加到模块中
        int ret = PyModule_AddObject(module, name, v);
        // 如果添加失败，则释放对象 v 的引用
        if (ret != 0) {
          Py_DECREF(v);
        }

        return ret == 0;
      };
#if defined(USE_CUDNN) || defined(USE_ROCM)
  // 如果定义了 USE_CUDNN 或 USE_ROCM，则设置 has_cudnn 为 Python 中的 True
  PyObject* has_cudnn = Py_True;
#else
  // 否则设置 has_cudnn 为 Python 中的 False
  PyObject* has_cudnn = Py_False;
#endif
  // 将 _has_cudnn 属性设置为上述结果
  ASSERT_TRUE(set_module_attr("_has_cudnn", has_cudnn));

#if AT_MKL_ENABLED() || AT_POCKETFFT_ENABLED()
  // 如果 AT_MKL_ENABLED() 或 AT_POCKETFFT_ENABLED() 返回 True，则设置 has_spectral 为 Python 中的 True
  PyObject* has_spectral = Py_True;
#else
  // 否则设置 has_spectral 为 Python 中的 False
  PyObject* has_spectral = Py_False;
#endif
  // 将 has_spectral 属性设置为上述结果
  ASSERT_TRUE(set_module_attr("has_spectral", has_spectral));

  // 强制 ATen 进行初始化，因为它处理设置 TH（Torch/THC）错误，使其抛出 C++ 异常
  at::init();

  // 自动翻译从 pybind11 函数抛出的错误
  py::register_exception_translator([](std::exception_ptr e) { // NOLINT
    try {
      if (e) {
        // 尝试重新抛出异常
        std::rethrow_exception(e);
      }
    }
    // 捕获 Torch 错误
    CATCH_TH_ERRORS()
  });

  // 将 Python 模块转换为 py::module 类型
  auto py_module = py::reinterpret_borrow<py::module>(module);
  // 定义 _demangle 函数
  py_module.def("_demangle", &c10::demangle);
  // 定义 _log_api_usage_once 函数
  py_module.def("_log_api_usage_once", &LogAPIUsageOnceFromPython);
  // 定义 _log_api_usage_metadata 函数
  py_module.def("_log_api_usage_metadata", &LogAPIUsageMetadataFromPython);

  // 定义 vitals_enabled 函数
  py_module.def("vitals_enabled", &at::vitals::torchVitalEnabled);
  // 定义 set_vital 函数
  py_module.def(
      "set_vital",
      [](const std::string& vital,
         const std::string& attr,
         const std::string& value) {
        return at::vitals::VitalsAPI.setVital(vital, attr, value);
      });
  // 定义 read_vitals 函数
  py_module.def(
      "read_vitals", []() { return at::vitals::VitalsAPI.readVitals(); });

  // 定义 init_num_threads 函数，并提供文档字符串
  py_module.def(
      "init_num_threads",
      torch::wrap_pybind_function(at::init_num_threads),
      R"(
init_num_threads()

Initializes the number of parallel threads used on the current thread.

Call this whenever a new thread is created in order to propagate values from
:func:`torch.set_num_threads` onto the new thread.
)");

  // 设置 has_openmp 属性，根据 at::hasOpenMP() 的结果选择 Py_True 或 Py_False
  ASSERT_TRUE(
      set_module_attr("has_openmp", at::hasOpenMP() ? Py_True : Py_False));
  // 设置 has_mkl 属性，根据 at::hasMKL() 的结果选择 Py_True 或 Py_False
  ASSERT_TRUE(set_module_attr("has_mkl", at::hasMKL() ? Py_True : Py_False));
  // 设置 has_lapack 属性，根据 at::hasLAPACK() 的结果选择 Py_True 或 Py_False
  ASSERT_TRUE(
      set_module_attr("has_lapack", at::hasLAPACK() ? Py_True : Py_False));

  // 定义 _valgrind_supported_platform 函数，根据定义 USE_VALGRIND 的情况返回 True 或 False
  py_module.def("_valgrind_supported_platform", []() {
#if defined(USE_VALGRIND)
    return true;
#else
      return false;
#endif
  });

  // 定义 _valgrind_toggle 函数，根据定义 USE_VALGRIND 的情况执行 CALLGRIND_TOGGLE_COLLECT 或抛出错误
  py_module.def("_valgrind_toggle", []() {
#if defined(USE_VALGRIND)
    CALLGRIND_TOGGLE_COLLECT;
#else
      TORCH_CHECK(false, "Valgrind is not supported.");
#endif
  });

  // 定义 _valgrind_toggle_and_dump_stats 函数，根据定义 USE_VALGRIND 的情况执行 CALLGRIND_TOGGLE_COLLECT 和 CALLGRIND_DUMP_STATS 或抛出错误
  py_module.def("_valgrind_toggle_and_dump_stats", []() {
#if defined(USE_VALGRIND)
    // 注意：如果不在 dump stats 前后进行 toggle collect，callgrind_annotate 将无法正确处理结果
    CALLGRIND_TOGGLE_COLLECT;
    CALLGRIND_DUMP_STATS;
#else
      TORCH_CHECK(false, "Valgrind is not supported.");
#endif
  });

  // 定义 _can_use_mem_efficient_attention 函数，接受参数并调用 at::vitals::VitalsAPI 的方法
  py_module.def(
      "_can_use_mem_efficient_attention",
      [](const sdp::sdp_params& params, bool debug) {
#ifdef USE_CUDA
        return sdp::can_use_flash_attention(params, debug);
#else
        return false;
#endif
      });
  # 如果定义了 USE_CUDA 宏，则调用 sdp::can_use_mem_efficient_attention 函数，返回其结果；否则返回 false
  # 以下代码块为条件返回语句的结尾
#ifdef USE_CUDA
        return sdp::can_use_mem_efficient_attention(params, debug);
#else
        return false;
#endif

  # 在 Python 模块中定义一个枚举类型 _LinalgBackend，包含 Default、Cusolver 和 Magma 三个值
  py::enum_<at::LinalgBackend>(py_module, "_LinalgBackend")
      .value("Default", at::LinalgBackend::Default)
      .value("Cusolver", at::LinalgBackend::Cusolver)
      .value("Magma", at::LinalgBackend::Magma);

  # 定义一个 Python 绑定函数 _set_linalg_preferred_backend，用于设置全局上下文的线性代数优选后端
  py_module.def("_set_linalg_preferred_backend", [](at::LinalgBackend b) {
    at::globalContext().setLinalgPreferredBackend(b);
  });

  # 定义一个 Python 绑定函数 _get_linalg_preferred_backend，用于获取全局上下文的线性代数优选后端
  py_module.def("_get_linalg_preferred_backend", []() {
    return at::globalContext().linalgPreferredBackend();
  });

  # 在 Python 模块中定义一个枚举类型 _BlasBackend，包含 Cublas 和 Cublaslt 两个值
  py::enum_<at::BlasBackend>(py_module, "_BlasBackend")
      .value("Cublas", at::BlasBackend::Cublas)
      .value("Cublaslt", at::BlasBackend::Cublaslt);

  # 定义一个 Python 绑定函数 _set_blas_preferred_backend，用于设置全局上下文的 BLAS 优选后端
  py_module.def("_set_blas_preferred_backend", [](at::BlasBackend b) {
    at::globalContext().setBlasPreferredBackend(b);
  });

  # 定义一个 Python 绑定函数 _get_blas_preferred_backend，用于获取全局上下文的 BLAS 优选后端
  py_module.def("_get_blas_preferred_backend", []() {
    return at::globalContext().blasPreferredBackend();
  });

  # 定义一个 Python 绑定函数 _construct_storage_from_data_pointer，用于根据数据指针、设备和字节大小构造 Storage 对象
  py_module.def(
      "_construct_storage_from_data_pointer",
      [](int64_t data_ptr, c10::Device device, size_t size_bytes) {
        return c10::Storage(
            c10::Storage::use_byte_size_t(),
            size_bytes,
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            at::DataPtr(reinterpret_cast<void*>(data_ptr), device));
      });

  # 定义一个 Python 绑定函数 _stash_obj_in_tls，用于将 Python 对象存储在线程本地存储中
  py_module.def(
      "_stash_obj_in_tls", [](const std::string& key, py::handle arg) {
        at::impl::ThreadLocalPythonObjects::get_state().set(
            key,
            std::make_shared<c10::SafePyObject>(arg.ptr(), getPyInterpreter()));
      });

  # 定义一个 Python 绑定函数 _get_obj_in_tls，用于从线程本地存储中获取存储的 Python 对象
  py_module.def("_get_obj_in_tls", [](const std::string& key) -> py::handle {
    auto safe_pyobject =
        at::impl::ThreadLocalPythonObjects::get_state().get(key);
    auto obj = safe_pyobject->ptr(getPyInterpreter());
    return py::handle(obj);
  });

  # 定义一个 Python 绑定函数 _is_key_in_tls，用于检查线程本地存储中是否存在指定的键
  py_module.def("_is_key_in_tls", [](const std::string& key) -> bool {
    return at::impl::ThreadLocalPythonObjects::get_state().contains(key);
  });

  # 定义一个 Python 绑定函数 _accelerator_hooks_device_count，用于获取当前加速器设备的数量
  py_module.def("_accelerator_hooks_device_count", []() {
    auto device_type = at::getAccelerator();
    if (device_type.has_value()) {
      return at::globalContext()
          .getAcceleratorHooksInterface(device_type.value())
          .deviceCount();
    }
    return c10::DeviceIndex(-1);
  });

  # 定义一个 Python 绑定函数 _accelerator_hooks_set_current_device，用于设置当前加速器设备
  py_module.def(
      "_accelerator_hooks_set_current_device",
      [](c10::DeviceIndex device_index) {
        auto device_type = at::getAccelerator();
        if (device_type.has_value()) {
          at::globalContext()
              .getAcceleratorHooksInterface(device_type.value())
              .setCurrentDevice(device_index);
        }
      });

  # 定义一个 Python 绑定函数 _accelerator_hooks_get_current_device，用于获取当前加速器设备
  py_module.def("_accelerator_hooks_get_current_device", []() {
    auto device_type = at::getAccelerator();
    if (device_type.has_value()) {
      return at::globalContext()
          .getAcceleratorHooksInterface(device_type.value())
          .getCurrentDevice();
    }
  return c10::DeviceIndex(-1);
  // 返回一个表示设备索引的对象，值为 -1
});

py_module.def(
    "_accelerator_hooks_exchange_device", [](c10::DeviceIndex device_index) {
      // 获取当前加速器类型
      auto device_type = at::getAccelerator();
      // 如果加速器类型存在
      if (device_type.has_value()) {
        // 获取加速器钩子接口，并交换设备
        return at::globalContext()
            .getAcceleratorHooksInterface(device_type.value())
            .exchangeDevice(device_index);
      }
      // 如果加速器类型不存在，返回 -1
      return c10::DeviceIndex(-1);
    });

py_module.def(
    "_accelerator_hooks_maybe_exchange_device",
    [](c10::DeviceIndex device_index) {
      // 获取当前加速器类型
      auto device_type = at::getAccelerator();
      // 如果加速器类型存在
      if (device_type.has_value()) {
        // 获取加速器钩子接口，并可能交换设备
        return at::globalContext()
            .getAcceleratorHooksInterface(device_type.value())
            .maybeExchangeDevice(device_index);
      }
      // 如果加速器类型不存在，返回 -1
      return c10::DeviceIndex(-1);
    });

py_module.def(
    "_get_accelerator",
    [](std::optional<bool> check = c10::nullopt) {
      // 获取加速器类型，并根据参数检查是否初始化
      return c10::Device(
          at::getAccelerator(check.value_or(false))
              .value_or(c10::DeviceType::CPU),
          -1);
    },
    py::arg("check") = nullptr);
#ifdef USE_CUDA
  // 如果定义了 USE_CUDA 宏，则将 has_cuda 设置为 Python 中的 True
  PyObject* has_cuda = Py_True;
#else
  // 如果未定义 USE_CUDA 宏，则将 has_cuda 设置为 Python 中的 False
  PyObject* has_cuda = Py_False;
#endif

#ifdef USE_MPS
  // 如果定义了 USE_MPS 宏，则将 has_mps 设置为 Python 中的 True
  PyObject* has_mps = Py_True;
#else
  // 如果未定义 USE_MPS 宏，则将 has_mps 设置为 Python 中的 False
  PyObject* has_mps = Py_False;
#endif

#ifdef USE_XPU
  // 如果定义了 USE_XPU 宏，则将 has_xpu 设置为 Python 中的 True
  PyObject* has_xpu = Py_True;
#else
  // 如果未定义 USE_XPU 宏，则将 has_xpu 设置为 Python 中的 False
  PyObject* has_xpu = Py_False;
#endif

// 将 _has_cuda 属性设置为 has_cuda 对象
ASSERT_TRUE(set_module_attr("_has_cuda", has_cuda));
// 将 _has_magma 属性设置为 at::hasMAGMA() 的结果（True 或 False）
ASSERT_TRUE(
    set_module_attr("_has_magma", at::hasMAGMA() ? Py_True : Py_False));
// 将 _has_mps 属性设置为 has_mps 对象
ASSERT_TRUE(set_module_attr("_has_mps", has_mps));
// 将 _has_xpu 属性设置为 has_xpu 对象
ASSERT_TRUE(set_module_attr("_has_xpu", has_xpu));
// 将 _has_mkldnn 属性设置为 at::hasMKLDNN() 的结果（True 或 False）
ASSERT_TRUE(
    set_module_attr("_has_mkldnn", at::hasMKLDNN() ? Py_True : Py_False));

#ifdef _GLIBCXX_USE_CXX11_ABI
  // 如果定义了 _GLIBCXX_USE_CXX11_ABI 宏，则将其设置为 Python 中的 True 或 False
  ASSERT_TRUE(set_module_attr(
      "_GLIBCXX_USE_CXX11_ABI", _GLIBCXX_USE_CXX11_ABI ? Py_True : Py_False));
#else
  // 如果未定义 _GLIBCXX_USE_CXX11_ABI 宏，则将其设置为 Python 中的 False
  ASSERT_TRUE(set_module_attr("_GLIBCXX_USE_CXX11_ABI", Py_False));
#endif

// 定义宏 SET_STR_DEFINE，用于设置字符串属性
// 根据 PYBIND11_COMPILER_TYPE 宏设置对应的属性
#ifdef PYBIND11_COMPILER_TYPE
  SET_STR_DEFINE(PYBIND11_COMPILER_TYPE);
#else
  // 如果未定义 PYBIND11_COMPILER_TYPE 宏，则将属性设置为 Python 中的 None
  ASSERT_TRUE(
      set_module_attr("_" C10_STRINGIZE(PYBIND11_COMPILER_TYPE), Py_None));
#endif

// 根据 PYBIND11_STDLIB 宏设置对应的属性
#ifdef PYBIND11_STDLIB
  SET_STR_DEFINE(PYBIND11_STDLIB);
#else
  // 如果未定义 PYBIND11_STDLIB 宏，则将属性设置为 Python 中的 None
  ASSERT_TRUE(set_module_attr("_" C10_STRINGIZE(PYBIND11_STDLIB), Py_None));
#endif

// 根据 PYBIND11_BUILD_ABI 宏设置对应的属性
#ifdef PYBIND11_BUILD_ABI
  SET_STR_DEFINE(PYBIND11_BUILD_ABI);
#else
  // 如果未定义 PYBIND11_BUILD_ABI 宏，则将属性设置为 Python 中的 None
  ASSERT_TRUE(set_module_attr("_" C10_STRINGIZE(PYBIND11_BUILD_ABI), Py_None));
#endif
#undef SET_STR_DEFINE

// 定义 Python 函数 _set_conj，将 C++ 函数 x._set_conj 包装为 Python 函数
py_module.def(
    "_set_conj", [](const at::Tensor& x, bool conj) { x._set_conj(conj); });
// 定义 Python 函数 _set_neg，将 C++ 函数 x._set_neg 包装为 Python 函数
py_module.def(
    "_set_neg", [](const at::Tensor& x, bool neg) { x._set_neg(neg); });
// 定义 Python 函数 _get_tensor_metadata，将 C++ 函数 torch::jit::getTensorMetadata 包装为 Python 函数
py_module.def("_get_tensor_metadata", &torch::jit::getTensorMetadata);
// 定义 Python 函数 _set_tensor_metadata，将 C++ 函数 torch::jit::setTensorMetadata 包装为 Python 函数
py_module.def(
    "_set_tensor_metadata",
    static_cast<void (*)(
        const at::Tensor&, std::unordered_map<std::string, bool>)>(
        torch::jit::setTensorMetadata));
// 定义 Python 函数 _dispatch_key_set，将 C++ 函数 x.key_set 包装为 Python 函数
py_module.def("_dispatch_key_set", [](const at::Tensor& x) {
  return toString(x.key_set());
});
// 定义 Python 函数 _has_storage，将 C++ 函数 x.has_storage 包装为 Python 函数
py_module.def(
    "_has_storage", [](const at::Tensor& x) { return x.has_storage(); });

// 定义 Python 函数 _set_meta_in_tls_dispatch_include，设置本地 TLS 中的分发键集合
py_module.def("_set_meta_in_tls_dispatch_include", [](bool meta_in_tls) {
  auto local_keyset = c10::impl::tls_local_dispatch_key_set();
  c10::DispatchKeySet key_set({at::DispatchKey::Meta});
  if (meta_in_tls) {
    local_keyset.included_ = local_keyset.included_ | key_set;
  } else {
    local_keyset.included_ =
        local_keyset.included_.remove_backend(c10::BackendComponent::MetaBit);
  }
  c10::impl::_force_tls_local_dispatch_key_set(local_keyset);
});

// 定义 Python 函数 _meta_in_tls_dispatch_include，返回本地 TLS 中是否包含元数据分发键
py_module.def("_meta_in_tls_dispatch_include", []() {
  auto local_keyset = c10::impl::tls_local_dispatch_key_set();
  return local_keyset.included_.has_backend(c10::BackendComponent::MetaBit);
});

// 定义 Python 函数 _dump_local_tls_set，打印本地 TLS 中的分发键集合
py_module.def("_dump_local_tls_set", []() {
  auto local_keyset = c10::impl::tls_local_dispatch_key_set();
    // 输出 "Included: " 后接本地密钥集合的包含项，转换为字符串输出
    std::cout << "Included: " << toString(local_keyset.included_) << "\n";
    // 输出 "Excluded: " 后接本地密钥集合的排除项，转换为字符串输出
    std::cout << "Excluded: " << toString(local_keyset.excluded_) << "\n";
  });

  // 定义 Python 模块的函数 "_should_allow_numbers_as_tensors"，返回是否允许将数字作为张量
  py_module.def(
      "_should_allow_numbers_as_tensors", [](const std::string& name) {
        return torch::should_allow_numbers_as_tensors(name);
      });

  // 定义 Python 模块的函数 "_group_tensors_by_device_and_dtype"，按设备和数据类型对张量进行分组
  py_module.def(
      "_group_tensors_by_device_and_dtype",
      [](const std::vector<std::vector<std::optional<at::Tensor>>>&
             nested_tensorlist,
         const bool with_indices) {
        return at::native::_group_tensors_by_first_tensors_device_and_dtype(
            nested_tensorlist, with_indices);
      });

  // 定义 Python 模块的函数 "_storage_address"，返回张量存储的内存地址
  py_module.def(
      "_storage_address",
      [](const at::Tensor& tensor) {
        return reinterpret_cast<std::intptr_t>(
            tensor.storage().unsafeGetStorageImpl());
      },
      "Gets the memory address of the Tensor's StorageImpl.");

  // 定义 Python 模块的函数 "_data_address"，返回张量数据指针的内存地址
  py_module.def(
      "_data_address",
      [](const at::Tensor& tensor) {
        return reinterpret_cast<std::intptr_t>(tensor.storage().data());
      },
      "Gets the memory address of the Tensor's data pointer.");

  // 定义 Python 模块的函数 "_is_cow_tensor"，检查张量数据指针是否是 COW（写时复制）的
  py_module.def(
      "_is_cow_tensor",
      [](const at::Tensor& tensor) {
        return c10::impl::cow::is_cow_data_ptr(tensor.storage().data_ptr());
      },
      "Checks if a tensor's data pointer is COW");

  // 定义 Python 模块的函数 "_get_cudnn_batch_norm_reserve_space_size"，获取 CuDNN 批量归一化保留空间大小
  py_module.def(
      "_get_cudnn_batch_norm_reserve_space_size",
      [](const at::Tensor& input, bool training) {
#ifdef USE_CUDA
        // 如果使用 CUDA，调用 CUDA 版本的批量归一化的保留空间大小计算函数
        return at::native::_get_cudnn_batch_norm_reserve_space_size(
            input, training);
#else
        // 如果没有使用 CUDA，抛出错误信息，指示 PyTorch 没有使用 CUDA 构建
        TORCH_CHECK(false, "PyTorch was not built with cuda");
#endif
      },
      py::arg("input"),
      py::arg("training"));

  // 在 Python 模块中定义一个枚举，用于表示批量归一化的后端实现方式
  py::enum_<at::native::BatchNormBackend>(py_module, "_BatchNormBackend")
      .value("Native", at::native::BatchNormBackend::Native)
      .value("Cudnn", at::native::BatchNormBackend::Cudnn)
      .value("Miopen", at::native::BatchNormBackend::Miopen);

  // 在 Python 模块中定义一个函数，根据输入参数选择合适的批量归一化后端实现
  py_module.def(
      "_select_batch_norm_backend",
      [](const at::Tensor& input,
         const at::Tensor& weight,
         const at::Tensor& bias,
         const at::Tensor& running_mean,
         const at::Tensor& running_var,
         bool training,
         double eps) {
        return at::native::_select_batch_norm_backend(
            input, weight, bias, running_mean, running_var, training, eps);
      },
      py::arg("input"),
      py::arg("weight"),
      py::arg("bias"),
      py::arg("running_mean"),
      py::arg("running_var"),
      py::arg("training"),
      py::arg("eps"));

  // 获取默认的 CPU 生成器，并初始化 Python 全局变量 THPDefaultCPUGenerator
  const auto& defaultGenerator = at::detail::getDefaultCPUGenerator();
  THPDefaultCPUGenerator =
      (THPGenerator*)THPGenerator_initDefaultGenerator(defaultGenerator);
  // 不需要增加引用计数，因为这个引用会被传递出去
  ASSERT_TRUE(set_module_attr(
      "default_generator",
      (PyObject*)THPDefaultCPUGenerator,
      /* incref= */ false));
  // 设置 Python 模块的属性，用于禁用 Torch Function 的子类化
  ASSERT_TRUE(set_module_attr(
      "DisableTorchFunctionSubclass",
      (PyObject*)THPModule_DisableTorchFunctionSubclassType(),
      /* incref= */ false));
  // 设置 Python 模块的属性，用于禁用 Torch Function
  ASSERT_TRUE(set_module_attr(
      "DisableTorchFunction",
      (PyObject*)THPModule_DisableTorchFunctionType(),
      /* incref= */ false));
  // 设置全局变量，用于指定禁用 Torch Function 实现的函数
  torch::set_disabled_torch_function_impl(
      PyObject_GetAttrString(module, "_disabled_torch_function_impl"));
  // 确保禁用 Torch Function 实现的函数不为空
  ASSERT_TRUE(torch::disabled_torch_function_impl() != nullptr);
  // 设置全局变量，用于指定禁用 Torch Dispatch 实现的函数
  torch::set_disabled_torch_dispatch_impl(
      PyObject_GetAttrString(module, "_disabled_torch_dispatch_impl"));
  // 确保禁用 Torch Dispatch 实现的函数不为空
  ASSERT_TRUE(torch::disabled_torch_dispatch_impl() != nullptr);
  // 返回 Python 模块
  return module;
  END_HANDLE_TH_ERRORS
}

// 检查 _C 共享库是否已经被初始化多次的辅助函数
// 如果多次初始化，则打印错误信息并终止程序
inline void pytorch_duplicate_guard() {
  // 静态变量，用于记录是否已经初始化
  static int initialized = 0;
  // 如果已经初始化，则输出错误信息并终止程序
  if (initialized) {
    fmt::print(stderr, "pytorch: _C shared library re-initialized\n");
    abort();
  }
  // 将初始化标志设置为已初始化
  initialized = 1;
  ;
}

// 结构体，用于在构造函数中调用 _C 共享库初始化检查函数
struct call_duplicate_guard {
  call_duplicate_guard() {
    // 调用 _C 共享库初始化检查函数
    pytorch_duplicate_guard();
  }
};

// 静态对象，用于在程序启动时调用 _C 共享库初始化检查函数
static call_duplicate_guard _call_duplicate_guard;
```