# `.\pytorch\torch\csrc\cuda\Module.cpp`

```
#ifndef WIN32
// 全局变量，用于标记是否处于不良的 fork 环境中
static bool in_bad_fork = false; // True for children forked after cuda init
#endif

#ifndef WIN32
// 在 CUDA 已经初始化的情况下，当子进程 fork 时调用此函数
static void forked_child() {
  in_bad_fork = true;
  // 设置需要延迟设备初始化为真，适用于 CUDA 设备
  torch::utils::set_requires_device_init(at::kCUDA, true);
}
#endif

// 在第一次 CUDA 调用之前应该调用此函数
// 注意：这与 initExtension 不同，因为存根 CUDA 实现具有一些工作函数（例如 device_count），但无法完全初始化
static void poison_fork() {
#ifndef WIN32
  // 保证只调用一次 forked_child 函数
  static c10::once_flag flag;
  c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

////////////////////////////////////////////////////////////////////////////////
// CUDA 管理方法
////////////////////////////////////////////////////////////////////////////////

// 设置 CUDA 设备的 Python 封装函数
PyObject* THCPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否是长整型
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  // 解包参数为设备索引
  auto device = THPUtils_unpackLong(arg);

  // 懒惰初始化 CUDA 设备
  torch::utils::device_lazy_init(at::kCUDA);
  // 设置当前设备为指定设备
  c10::cuda::set_device(static_cast<c10::DeviceIndex>(device));

  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
PyObject* THCPModule_exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为长整型，否则抛出异常
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  // 解包设备索引
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  // 如果设备索引小于0，返回包装的-1
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  // CUDA 设备延迟初始化
  torch::utils::device_lazy_init(at::kCUDA);
  // 交换当前设备为给定的设备索引
  auto current_device = c10::cuda::ExchangeDevice(device_index);

  // 返回当前设备的包装对象
  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_maybeExchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为长整型，否则抛出异常
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  // 解包设备索引
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  // 如果设备索引小于0，返回包装的-1
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  // CUDA 设备延迟初始化
  torch::utils::device_lazy_init(at::kCUDA);
  // 可能交换当前设备为给定的设备索引
  auto current_device = c10::cuda::MaybeExchangeDevice(device_index);

  // 返回当前设备的包装对象
  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // CUDA 设备延迟初始化
  torch::utils::device_lazy_init(at::kCUDA);
  // 获取当前 CUDA 设备索引并转换为 int32_t 类型
  auto device = static_cast<int32_t>(c10::cuda::current_device());
  // 返回当前设备索引的包装对象
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_canDeviceAccessPeer_wrap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 解析参数元组，期望两个整数参数
  PyObject* arg1 = nullptr;
  PyObject* arg2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
    // 如果解析失败，抛出异常并返回空指针
    THPUtils_invalidArguments(
        args,
        nullptr,
        "can_device_peer_access",
        1,
        "(int device, int peer_device);");
    return nullptr;
  }
  // 检查参数是否为长整型，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkLong(arg1), "invalid argument to canDeviceAccessPeer");
  TORCH_CHECK(
      THPUtils_checkLong(arg2), "invalid argument to canDeviceAccessPeer");
  // 解包设备索引
  int64_t device = THPUtils_unpackLong(arg1);
  int64_t peer_device = THPUtils_unpackLong(arg2);

  // CUDA 设备延迟初始化
  torch::utils::device_lazy_init(at::kCUDA);
  // 检查设备之间是否可以相互访问
  auto can_access = at::cuda::canDeviceAccessPeer(device, peer_device);
  // 返回布尔值的包装对象
  return PyBool_FromLong(can_access);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 禁止在分叉中使用
  poison_fork();
  // 返回当前 CUDA 设备数量的无符号 64 位整数包装对象
  return THPUtils_packUInt64(at::cuda::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getArchFlags(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 禁止在分叉中使用
  poison_fork();
#ifdef CUDA_ARCH_FLAGS
  // 返回 CUDA 架构标志的字符串包装对象
  static const char* flags = C10_STRINGIZE(CUDA_ARCH_FLAGS);
  return THPUtils_packString(flags);
#else
  // 如果未定义 CUDA_ARCH_FLAGS，则返回 None
  Py_RETURN_NONE;
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPModule_isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回当前是否处于错误的分叉状态的布尔值包装对象
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}
    // 检查传入的设备索引是否为长整型，否则抛出错误
    PyObject* device_index) {
      HANDLE_TH_ERRORS
      TORCH_CHECK(
          THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
      // 将Python对象的设备索引解包成C++的设备索引类型
      auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
      // 获取当前CUDA流对象
      auto stream = at::cuda::getCurrentCUDAStream(c10_device_index);
      // 创建一个Python元组对象，用于存放输出结果，共3个元素
      PyObject* output_tuple = PyTuple_New(3);
      // 将CUDA流的ID转换成int64_t并封装为Python整数对象，设置为元组的第一个元素
      PyTuple_SetItem(
          output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
      // 将CUDA流的设备索引封装为Python设备索引对象，设置为元组的第二个元素
      PyTuple_SetItem(
          output_tuple, 1, THPUtils_packDeviceIndex(stream.device_index()));
      // 将CUDA流的设备类型转换成int64_t并封装为Python整数对象，设置为元组的第三个元素
      PyTuple_SetItem(
          output_tuple,
          2,
          THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
      // 返回存放输出结果的元组对象
      return output_tuple;
      // 处理Torch和TH错误的宏结束
      END_HANDLE_TH_ERRORS
    }
}

PyObject* THCPModule_getCurrentStream_raw(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  // 检查传入的 device_index 是否是一个长整型数值
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  // 解包 device_index 转换为 C10 设备索引
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  // 返回当前 CUDA 流的句柄指针
  return PyLong_FromVoidPtr(
      at::cuda::getCurrentCUDAStream(c10_device_index).stream());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getDefaultStream_wrap(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  // 检查传入的 device_index 是否是一个长整型数值
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  // 解包 device_index 转换为 C10 设备索引
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  // 获取默认的 CUDA 流对象
  auto stream = at::cuda::getDefaultCUDAStream(c10_device_index);
  // 创建一个 Python 元组对象，包含流的 id、设备索引和设备类型
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple, 1, THPUtils_packDeviceIndex(stream.device_index()));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_setStream_wrap(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 初始化默认值
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  // 解析传入的参数列表，支持可选的 stream_id、device_index 和 device_type
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  // 解包参数，构建 CUDA 流对象
  auto stream = at::cuda::CUDAStream::unpack3(
      stream_id,
      static_cast<c10::DeviceIndex>(device_index),
      static_cast<c10::DeviceType>(device_type));

  // 获取当前 CUDA 设备
  auto device = c10::cuda::current_device();
  // 如果当前设备与流的设备索引不同，则切换到流的设备
  if (device != stream.device_index()) {
    c10::cuda::set_device(stream.device_index());
  }
  // 设置当前 CUDA 流
  at::cuda::setCurrentCUDAStream(stream);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getCompiledVersion(PyObject* self, PyObject* noargs) {
#if defined(USE_ROCM)
  // 如果定义了 USE_ROCM，返回 ROCM 版本号
  return THPUtils_packInt64((int64_t)ROCM_VERSION);
#else
  // 否则返回 CUDA 版本号
  return THPUtils_packInt64((int64_t)CUDA_VERSION);
#endif
}

PyObject* THCPModule_cudaHostAllocator(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 获取 CUDA 的缓存主机分配器
  c10::Allocator* allocator = at::cuda::getCachingHostAllocator();
  // 返回分配器的指针
  return PyLong_FromVoidPtr(allocator);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaCachingAllocator_raw_alloc(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* size_o = nullptr;
  PyObject* stream_o = nullptr;
  // 解析传入的参数，期望两个对象参数：size_o 和 stream_o
  if (!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
    # 调用 THPUtils_invalidArguments 函数，用于处理无效参数情况
    THPUtils_invalidArguments(
        args,
        nullptr,
        "caching_allocator_alloc",
        1,
        "(ssize_t size, intptr_t stream);");
    # 如果参数无效，返回空指针
    return nullptr;
  }
  # 将 Python 对象 size_o 转换为 ssize_t 类型的整数
  auto size = PyLong_AsSsize_t(size_o);
  # 将 Python 对象 stream_o 转换为 cudaStream_t 类型的指针
  cudaStream_t stream = static_cast<cudaStream_t>(PyLong_AsVoidPtr(stream_o));
  # 分配内存的指针初始化为 nullptr
  void* mem = nullptr;
  {
    # 释放全局解释器锁，允许 GIL 被释放
    pybind11::gil_scoped_release no_gil;
    # 使用 CUDACachingAllocator 类的 raw_alloc_with_stream 方法分配指定大小和流的内存
    mem = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, stream);
  }
  # 将分配的内存指针转换为 Python 对象返回
  return PyLong_FromVoidPtr(mem);
  # 处理 Torch 的异常错误
  END_HANDLE_TH_ERRORS
// Unpack a PyObject to at::Scalar, throw an exception if it fails
at::Scalar as_scalar(PyObject* arg) {
  // Zero-dim tensors are converted to Scalars as-is. Note this doesn't
  // currently handle most NumPy scalar types except np.float64.
  if (THPVariable_Check(arg)) {
    // If the argument is a THPVariable, unpack it and return its item as a Scalar
    return THPVariable_Unpack(arg).item();
  }

  if (THPUtils_checkLong(arg)) {
    // If the argument is a Python long integer, cast it to int64_t and return as a Scalar
    return at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(arg)));
  }

  if (PyBool_Check(arg)) {
    // If the argument is a Python boolean, unpack it and return as a Scalar
    return at::Scalar(THPUtils_unpackBool(arg));
  }

  if (PyComplex_Check(arg)) {
    // If the argument is a Python complex number, unpack it to double and return as a Scalar
    return at::Scalar(THPUtils_unpackComplexDouble(arg));
  }
  // If none of the above types match, unpack the argument to double and return as a Scalar
  return at::Scalar(THPUtils_unpackDouble(arg));
}

// Entrypoint for the callable created by torch.cuda.jiterator
// See jiterator.py for more details
PyObject* THCPModule_cudaJiteratorCompileAndLaunchKernel(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS

  PyObject* code_string_o = nullptr;
  PyObject* kernel_name_o = nullptr;
  PyObject* return_by_ref_o = nullptr;
  PyObject* num_outputs_o = nullptr;
  PyObject* tensors_o = nullptr;
  PyObject* kwargs_o = nullptr;
  if (!PyArg_ParseTuple(
          args,
          "OOOOO|O",
          &code_string_o,
          &kernel_name_o,
          &return_by_ref_o,
          &num_outputs_o,
          &tensors_o,
          &kwargs_o)) {
    // Parsing the input arguments tuple failed, return nullptr
    return nullptr;
  }

  // Unpack Python string objects from input arguments
  const std::string code_string = THPUtils_unpackString(code_string_o);
  const std::string kernel_name = THPUtils_unpackString(kernel_name_o);
  // Unpack boolean and integer values from input arguments
  const bool return_by_ref = THPUtils_unpackBool(return_by_ref_o);
  const int num_outputs = static_cast<int>(THPUtils_unpackLong(num_outputs_o));

  // Check that tensors_o is a tuple
  TORCH_CHECK(
      PyTuple_Check(tensors_o),
      "tensors argument is expected to "
      "be a tuple, but got ",
      THPUtils_typename(tensors_o));
  // Get the number of tensors in the tuple
  Py_ssize_t num_tensors = PyTuple_GET_SIZE(tensors_o);

  // Vector to store unpacked tensors
  c10::SmallVector<at::Tensor> tensors;
  for (const auto i : c10::irange(num_tensors)) {
    // Get the i-th tensor from the tuple
    PyObject* _tensor = PyTuple_GET_ITEM(tensors_o, i);
    // Check that _tensor is a THPVariable
    TORCH_CHECK(
        THPVariable_Check(_tensor),
        i,
        " of input tensors tuple is not a Tensor");
    // Unpack _tensor and store in the tensors vector
    tensors.emplace_back(THPVariable_Unpack(_tensor));
  }

  // Vector to store unpacked keyword arguments
  c10::SmallVector<at::Scalar> extra_args;
  PyObject* key = nullptr;
  PyObject* value = nullptr;
  Py_ssize_t pos = 0;
  // Iterate through the kwargs_o dictionary
  while (PyDict_Next(kwargs_o, &pos, &key, &value)) {
    // Convert each keyword argument value to a Scalar and store in extra_args vector
    extra_args.emplace_back(as_scalar(value));
  }

  // Call the CUDA backend function to compile and launch the kernel
  c10::SmallVector<at::Tensor> outputs = at::cuda::CompileAndLaunchKernel(
      code_string,
      kernel_name,
      num_outputs,
      tensors,
      extra_args,
      return_by_ref);

  // Depending on the number of outputs, wrap and return them accordingly
  if (num_outputs == 1) {
    // If there's only one output, wrap it as a THPVariable and return
    return THPVariable_Wrap(outputs[0]);
  } else {
    // If there are multiple outputs, create a tuple of THPVariable and return
    PyObject* output_tuple = PyTuple_New(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      PyTuple_SetItem(output_tuple, i, THPVariable_Wrap(outputs[i]));
    }
    return output_tuple;
  }

  END_HANDLE_TH_ERRORS
}
    PyObject* obj) {
```  
# 接受一个 PyObject 类型的参数 `obj`


  HANDLE_TH_ERRORS
```  
# 开始处理异常，宏定义的错误处理开始


  void* mem_ptr = PyLong_AsVoidPtr(obj);
```  
# 将 PyObject 转换为 void* 类型的指针 `mem_ptr`


  {
    pybind11::gil_scoped_release no_gil;
```  
# 创建一个 `no_gil` 的对象，用于在其生命周期内释放全局解释器锁 (GIL)


    c10::cuda::CUDACachingAllocator::raw_delete(mem_ptr);
```  
# 使用 CUDA 缓存分配器删除 `mem_ptr` 指向的内存块


  }
```  
# `no_gil` 对象的生命周期结束，GIL 重新获取


  Py_RETURN_NONE;
```  
# 返回一个 Python 的 None 对象


  END_HANDLE_TH_ERRORS
```  
# 结束异常处理宏定义的作用域
PyObject* THCPModule_cudaCachingAllocator_set_allocator_settings(
    PyObject* _unused,
    PyObject* env) {
  HANDLE_TH_ERRORS
  // 调用 CUDACachingAllocator 的 setAllocatorSettings 方法，设置分配器的环境变量
  c10::cuda::CUDACachingAllocator::setAllocatorSettings(
      THPUtils_unpackString(env));
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getAllocatorBackend(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回当前 CUDA 缓存分配器的后端名称
  return THPUtils_packString(c10::cuda::CUDACachingAllocator::name());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaSynchronize(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    // 释放全局解释器锁（GIL），确保 CUDA 设备同步时不会被 Python 线程调度打断
    pybind11::gil_scoped_release no_gil;
    // 执行 CUDA 设备同步操作
    c10::cuda::device_synchronize();
  }
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaIPCCollect(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 执行 CUDA IPC 收集操作
  torch::CudaIPCCollect();
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaSleep(PyObject* _unused, PyObject* cycles) {
  HANDLE_TH_ERRORS
  // 检查 cycles 参数是否为整数，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkLong(cycles), "torch.cuda._sleep(): expected 'int'");
  // 解包 cycles 参数为 int64_t 类型
  int64_t unpacked_cycles = THPUtils_unpackLong(cycles);
  {
    // 释放全局解释器锁（GIL），确保 CUDA 睡眠操作不会被 Python 线程打断
    pybind11::gil_scoped_release no_gil;
    // 执行 CUDA 睡眠操作，参数为解包后的 cycles 数值
    at::cuda::sleep(unpacked_cycles);
  }
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// We need to ensure that as long as a thread will NEVER loose the GIL as long
// as it holds the CUDA mutex. Otherwise another thread might be scheduled and
// try to e.g. allocate a new tensor which will cause a deadlock. It's enough to
// have a single global, because it can be only set once (cudaMutex is not
// recursive) by the thread that owns the mutex (obviously there can be only one
// such thread).
// 需要确保在持有 CUDA 互斥锁期间绝对不会丢失全局解释器锁（GIL）。否则，其他线程可能会被调度，
// 尝试分配新的张量，从而导致死锁。只需要一个全局变量即可，因为它只能由拥有互斥锁的线程设置一次
// （cudaMutex 不是递归的），显然只能有一个这样的线程。
static PyGILState_STATE cudaMutexGILState;

PyObject* THCPModule_cudaLockMutex(PyObject* module, PyObject* noargs) {
  auto mutex = c10::cuda::getFreeMutex();
  // 这必须是一个忙循环，因为我们绝对需要持有 GIL，否则会导致死锁（如果我们在持有 cudaMutex
  // 的同时允许其他 Python 线程运行，但没有 GIL，它们可能会尝试释放 CUDA 张量并获取 cudaMutex，
  // 而不放弃 GIL，因为这些操作深入到 THC 中）。
  while (true) {
    if (mutex->try_lock())
      break;
    {
      // 释放全局解释器锁（GIL），让其他 Python 线程有机会运行
      pybind11::gil_scoped_release no_gil;
      // 线程休眠 10 微秒，避免忙等待时 CPU 占用过高
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  // 保存当前 GIL 状态，并确保获取 GIL
  cudaMutexGILState = PyGILState_Ensure();
  // 返回 None 对象
  Py_RETURN_NONE;
}

PyObject* THCPModule_cudaUnlockMutex(PyObject* module, PyObject* noargs) {
  auto mutex = c10::cuda::getFreeMutex();
  // 释放先前保存的 GIL 状态
  PyGILState_Release(cudaMutexGILState);
  // 释放 CUDA 互斥锁
  mutex->unlock();
  // 返回 None 对象
  Py_RETURN_NONE;
}

PyObject* THCPModule_hasPrimaryContext(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为整数，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to has_primary_context");
  // 解包设备索引
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  // 检查是否存在主 CUDA 上下文
  if (c10::cuda::hasPrimaryContext(device_index)) {
    // 返回 Python 的 True 对象
    Py_RETURN_TRUE;
  } else {
    // 返回 Python 的 False 对象
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}
PyObject* THCPModule_setMemoryFraction(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // 定义两个 PyObject 指针，用于接收函数参数
  PyObject* fraction_o = nullptr;
  PyObject* device_o = nullptr;
  // 解析 Python 函数参数，期望参数类型为 double 和 int
  if (!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
    // 参数解析失败，抛出异常并返回空指针
    THPUtils_invalidArguments(
        args,
        nullptr,
        "set_memory_fraction",
        1,
        "(double fraction, int device);");
    return nullptr;
  }
  // 将 fraction_o 转换为 double 类型
  double fraction = PyFloat_AsDouble(fraction_o);
  // 解析 device_o 获取设备索引
  auto device_index = THPUtils_unpackDeviceIndex(device_o);

  // 调用 CUDACachingAllocator 类的静态方法设置内存分配比例
  c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, device_index);
  END_HANDLE_TH_ERRORS
  // 返回 None
  Py_RETURN_NONE;
}

PyObject* THCPModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 调用 CUDACachingAllocator 类的静态方法清空缓存
  c10::cuda::CUDACachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  // 返回 None
  Py_RETURN_NONE;
}

PyObject* THCPModule_memoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数类型是否为 long
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  // 解析参数获取设备索引
  const auto device_index = THPUtils_unpackDeviceIndex(arg);

  // 使用别名简化 CUDACachingAllocator 类中的类型和方法
  using c10::cuda::CUDACachingAllocator::DeviceStats;
  using c10::cuda::CUDACachingAllocator::Stat;
  using c10::cuda::CUDACachingAllocator::StatArray;
  using c10::cuda::CUDACachingAllocator::StatType;

  // 将 Stat 结构体转换为 Python 字典的 Lambda 函数
  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  // 将 StatArray 转换为 Python 字典的 Lambda 函数
  const auto statArrayToDict = [=](const StatArray& statArray) {
    // 定义一个静态数组存储统计类型的名称
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    // 遍历统计类型数组，转换为对应的 Python 字典
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
  return dict;
};

const DeviceStats stats =
    c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
// 使用 CUDA 缓存分配器获取指定设备的统计信息

py::dict result;
// 创建一个 Python 字典用于存储结果

result["num_alloc_retries"] = stats.num_alloc_retries;
// 将统计信息中的 num_alloc_retries 存入结果字典

result["num_ooms"] = stats.num_ooms;
// 将统计信息中的 num_ooms 存入结果字典

result["max_split_size"] = stats.max_split_size;
// 将统计信息中的 max_split_size 存入结果字典

result["num_sync_all_streams"] = stats.num_sync_all_streams;
// 将统计信息中的 num_sync_all_streams 存入结果字典

result["num_device_alloc"] = stats.num_device_alloc;
// 将统计信息中的 num_device_alloc 存入结果字典

result["num_device_free"] = stats.num_device_free;
// 将统计信息中的 num_device_free 存入结果字典

result["allocation"] = statArrayToDict(stats.allocation);
// 将统计信息中的 allocation 数组转换为字典，并存入结果字典的 "allocation" 键

result["segment"] = statArrayToDict(stats.segment);
// 将统计信息中的 segment 数组转换为字典，并存入结果字典的 "segment" 键

result["active"] = statArrayToDict(stats.active);
// 将统计信息中的 active 数组转换为字典，并存入结果字典的 "active" 键

result["inactive_split"] = statArrayToDict(stats.inactive_split);
// 将统计信息中的 inactive_split 数组转换为字典，并存入结果字典的 "inactive_split" 键

result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
// 将统计信息中的 allocated_bytes 数组转换为字典，并存入结果字典的 "allocated_bytes" 键

result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
// 将统计信息中的 reserved_bytes 数组转换为字典，并存入结果字典的 "reserved_bytes" 键

result["active_bytes"] = statArrayToDict(stats.active_bytes);
// 将统计信息中的 active_bytes 数组转换为字典，并存入结果字典的 "active_bytes" 键

result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
// 将统计信息中的 inactive_split_bytes 数组转换为字典，并存入结果字典的 "inactive_split_bytes" 键

result["requested_bytes"] = statArrayToDict(stats.requested_bytes);
// 将统计信息中的 requested_bytes 数组转换为字典，并存入结果字典的 "requested_bytes" 键

result["oversize_allocations"] = statToDict(stats.oversize_allocations);
// 将统计信息中的 oversize_allocations 转换为字典，并存入结果字典的 "oversize_allocations" 键

result["oversize_segments"] = statToDict(stats.oversize_segments);
// 将统计信息中的 oversize_segments 转换为字典，并存入结果字典的 "oversize_segments" 键

return result.release().ptr();
// 释放 result 对象的所有权，并返回其指针给调用者
END_HANDLE_TH_ERRORS
// 结束错误处理
}

PyObject* THCPModule_resetAccumulatedMemoryStats(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为长整型，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "invalid argument to reset_accumulated_memory_stats");
  // 解析设备索引
  const auto device_index = THPUtils_unpackDeviceIndex(arg);
  // 调用 CUDA 缓存分配器的方法重置累积内存统计信息
  c10::cuda::CUDACachingAllocator::resetAccumulatedStats(device_index);
  END_HANDLE_TH_ERRORS
  // 返回 None
  Py_RETURN_NONE;
}

PyObject* THCPModule_resetPeakMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为长整型，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  // 解析设备索引
  const auto device_index = THPUtils_unpackDeviceIndex(arg);
  // 调用 CUDA 缓存分配器的方法重置内存峰值统计信息
  c10::cuda::CUDACachingAllocator::resetPeakStats(device_index);
  END_HANDLE_TH_ERRORS
  // 返回 None
  Py_RETURN_NONE;
}

CapturedTraceback* getFromContext(
    const std::shared_ptr<c10::GatheredContext>& x) {
  // 如果 x 可以转换为 CapturedTraceback 指针，则返回该指针
  if (CapturedTraceback* sc = dynamic_cast<CapturedTraceback*>(x.get())) {
    return sc;
  }
  // 否则抛出异常，表明尝试从错误的 StackContext 类型中获取堆栈上下文
  TORCH_CHECK(
      false,
      "attempting to gather stack context from the wrong StackContext type.");
}

PyObject* THCPModule_memorySnapshot(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS

  // 导入 CUDA 缓存分配器的 BlockInfo 和 SegmentInfo 类
  using c10::cuda::CUDACachingAllocator::BlockInfo;
  using c10::cuda::CUDACachingAllocator::SegmentInfo;

  // 定义 Python 字符串常量
  py::str device_s = "device";
  py::str address_s = "address";
  py::str total_size_s = "total_size";
  py::str allocated_size_s = "allocated_size";
  py::str active_size_s = "active_size";
  py::str requested_size_s = "requested_size";
  py::str stream_s = "stream";
  py::str segment_type_s = "segment_type";
  py::str segment_pool_id = "segment_pool_id";
  py::str large_s = "large";
  py::str small_s = "small";
  py::str size_s = "size";
  py::str state_s = "state";
  py::str active_allocated_s = "active_allocated";
  py::str active_pending_free_s = "active_pending_free";
  py::str inactive_s = "inactive";
  py::str addr_s = "addr";
  py::str cpp_frames_s = "cpp_frames";
  py::str blocks_s = "blocks";
  py::str is_expandable_s = "is_expandable";
  py::str frames_s = "frames";
  py::str time_us_s = "time_us";

  // 创建空的 Python 列表和向量
  py::list empty_frames;
  std::vector<CapturedTraceback*> to_gather_frames;
  std::vector<py::dict> to_gather_dest;

  // 定义添加帧关键字的函数
  auto add_frame_key = [&](const py::dict& d,
                           const std::shared_ptr<c10::GatheredContext>& ctx) {
    // 如果上下文不为空，则获取堆栈上下文并添加到相应的向量中
    if (ctx) {
      auto sc = getFromContext(ctx);
      to_gather_frames.emplace_back(sc);
      to_gather_dest.emplace_back(d);
    } else {
      // 否则将空帧列表添加到字典中
      d[frames_s] = empty_frames;
    }
  };

  // 将 SegmentInfo 转换为 Python 字典的函数
  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict[device_s] = segmentInfo.device;
    segmentDict[address_s] = segmentInfo.address;
    segmentDict[total_size_s] = segmentInfo.total_size;
    segmentDict[allocated_size_s] = segmentInfo.allocated_size;
    segmentDict[active_size_s] = segmentInfo.active_size;
    segmentDict[requested_size_s] = segmentInfo.requested_size;
    segmentDict[stream_s] = segmentInfo.stream;
    segmentDict[segment_type_s] = segmentInfo.segment_type;
    segmentDict[segment_pool_id] = segmentInfo.segment_pool_id;
    segmentDict[large_s] = segmentInfo.large;
    segmentDict[small_s] = segmentInfo.small;
    segmentDict[size_s] = segmentInfo.size;
    segmentDict[state_s] = segmentInfo.state;
    segmentDict[active_allocated_s] = segmentInfo.active_allocated;
    segmentDict[active_pending_free_s] = segmentInfo.active_pending_free;
    segmentDict[inactive_s] = segmentInfo.inactive;
    segmentDict[addr_s] = segmentInfo.addr;
    segmentDict[cpp_frames_s] = empty_frames;  // 初始化为空帧列表
    return segmentDict;
  };
    // 将流用 int 表示，以便 Python 对象能够轻松 pickle
    segmentDict[stream_s] = int64_t(segmentInfo.stream);
    // 根据 segmentInfo 的 is_large 属性选择 large_s 或 small_s
    segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);
    // 将 segmentInfo 的 owner_private_pool_id 分配给 segmentDict
    segmentDict[segment_pool_id] = segmentInfo.owner_private_pool_id;
    // 将 segmentInfo 的 is_expandable 属性分配给 segmentDict
    segmentDict[is_expandable_s] = segmentInfo.is_expandable;
    // 将 segmentInfo.context_when_allocated 添加到 segmentDict 中
    add_frame_key(segmentDict, segmentInfo.context_when_allocated);

    // 将 segmentInfo 的 address 赋给 address
    auto address = segmentInfo.address;
    // 创建 py::list blocks
    py::list blocks;
    // 遍历 segmentInfo 的 blocks
    for (const auto& blockInfo : segmentInfo.blocks) {
      // 创建 py::dict blockDict
      py::dict blockDict;
      // 将 address 分配给 blockDict[address_s]
      blockDict[address_s] = address;
      // 将 blockInfo 的 size 分配给 blockDict[size_s]
      blockDict[size_s] = blockInfo.size;
      // 将 blockInfo 的 requested_size 分配给 blockDict[requested_size_s]
      blockDict[requested_size_s] = blockInfo.requested_size;
      // 根据 blockInfo 的 allocated 和 active 属性选择状态分配给 blockDict[state_s]
      blockDict[state_s] =
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s));
      // 将 blockInfo.context_when_allocated 添加到 blockDict 中
      add_frame_key(blockDict, blockInfo.context_when_allocated);
      // 将 blockDict 添加到 blocks 中
      blocks.append(blockDict);
      // 更新 address 以包含当前 block 的大小
      address += blockInfo.size;
    }
    // 将 blocks 添加到 segmentDict 中
    segmentDict[blocks_s] = blocks;

    // 返回 segmentDict
    return segmentDict;
  };

  // 获得当前 CUDA 缓存分配器的快照
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();

  // 创建 py::list segments
  py::list segments;

  // 遍历快照中的每个 segmentInfo，并将其转换为字典添加到 segments 中
  for (const auto& segmentInfo : snapshot.segments) {
    segments.append(segmentInfoToDict(segmentInfo));
  }

  // 创建 py::list traces 和字符串表示的动作类型
  py::list traces;
  py::str action_s = "action";
  py::str alloc_s = "alloc";
  py::str free_requested_s = "free_requested";
  py::str free_completed_s = "free_completed";
  py::str segment_alloc_s = "segment_alloc";
  py::str segment_free_s = "segment_free";
  py::str segment_map_s = "segment_map";
  py::str segment_unmap_s = "segment_unmap";

  py::str snapshot_s = "snapshot";
  py::str oom_s = "oom";
  py::str device_free_s = "device_free";
  py::str user_defined_s = "user_defined";

  // 使用命名空间 c10::cuda::CUDACachingAllocator
  using namespace c10::cuda::CUDACachingAllocator;

  // Lambda 函数 action_to_str 将 TraceEntry::Action 转换为字符串表示
  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      // 根据不同的 TraceEntry::Action 返回相应的字符串表示
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
      case TraceEntry::SEGMENT_UNMAP:
        return segment_unmap_s;
      case TraceEntry::SEGMENT_MAP:
        return segment_map_s;
      case TraceEntry::USER_DEFINED:
        return user_defined_s;
    }
    // 如果出现未知的 action，抛出运行时错误
    throw std::runtime_error("unreachable");
  };

  // 遍历快照中的每个 traceInfo，并将其转换为列表形式的 trace 添加到 traces 中
  for (const auto& traceInfo : snapshot.device_traces) {
    py::list trace;
    // 遍历跟踪信息列表，每个元素为 traceInfo 中的一个对象 te
    for (const auto& te : traceInfo) {
        // 创建 Python 字典 trace_entry，用于存储每个跟踪条目的信息
        py::dict trace_entry;
        if (te.context_) {
            // 如果 te 的 context_ 不为空，则获取其对应的栈信息
            auto sc = getFromContext(te.context_);
            // 将获取到的栈信息 sc 添加到 to_gather_frames 列表中
            to_gather_frames.emplace_back(sc);
            // 将空的 trace_entry 添加到 to_gather_dest 列表中
            to_gather_dest.emplace_back(trace_entry);
        }
        // 将 te 的动作 action_ 转换为字符串并存储到 trace_entry 中
        trace_entry[action_s] = action_to_str(te.action_);
        // 根据 te 的动作类型，存储地址或设备释放标识到 trace_entry 中
        trace_entry[TraceEntry::OOM == te.action_ ? device_free_s : addr_s] =
            te.addr_;
        // 存储 te 的大小 size_ 到 trace_entry 中
        trace_entry[size_s] = te.size_;
        // 存储 te 的流 stream_ 到 trace_entry 中
        trace_entry[stream_s] = int64_t(te.stream_);
        // 存储 te 的时间戳 te.time_.t_ 到 trace_entry 中
        trace_entry[time_us_s] = te.time_.t_;
        // 将 trace_entry 添加到 trace 列表中
        trace.append(trace_entry);
    }
    // 将 trace 添加到 traces 列表中
    traces.append(trace);

    // 创建 Python 字典 allocator_settings，用于存储内存分配器的设置信息
    py::dict allocator_settings;
    // 定义字符串变量 last_allocator_settings_s
    py::str last_allocator_settings_s = "PYTORCH_CUDA_ALLOC_CONF";
    // 定义字符串变量 max_split_size_s
    py::str max_split_size_s = "max_split_size";
    // 定义字符串变量 garbage_collection_threshold_s
    py::str garbage_collection_threshold_s = "garbage_collection_threshold";
    // 定义字符串变量 expandable_segments_s
    py::str expandable_segments_s = "expandable_segments";
    // 定义字符串变量 pinned_num_register_threads_s
    py::str pinned_num_register_threads_s = "pinned_num_register_threads";
    // 定义字符串变量 release_lock_on_malloc_s
    py::str release_lock_on_malloc_s = "release_lock_on_cudamalloc";
    // 定义字符串变量 pinned_use_host_register_s
    py::str pinned_use_host_register_s = "pinned_use_cuda_host_register";
    // 定义字符串变量 roundup_power2_divisions_s
    py::str roundup_power2_divisions_s = "roundup_power2_divisions";

    // 将内存分配器的配置信息存储到 allocator_settings 中
    allocator_settings[last_allocator_settings_s] =
        snapshot.config_metadata.last_allocator_settings;
    allocator_settings[max_split_size_s] =
        int64_t(snapshot.config_metadata.max_split_size);
    allocator_settings[garbage_collection_threshold_s] =
        snapshot.config_metadata.garbage_collection_threshold;
    allocator_settings[expandable_segments_s] =
        snapshot.config_metadata.expandable_segments;
    allocator_settings[pinned_num_register_threads_s] =
        int64_t(snapshot.config_metadata.pinned_num_register_threads);
    allocator_settings[release_lock_on_malloc_s] =
        snapshot.config_metadata.release_lock_on_malloc;
    allocator_settings[pinned_use_host_register_s] =
        snapshot.config_metadata.pinned_use_host_register;
    
    // 创建 Python 字典 roundup_settings，存储 roundup_power2_divisions_s 的设置信息
    unsigned int roundup_key = 1;
    py::dict roundup_settings;
    // 遍历 snapshot.config_metadata.roundup_power2_divisions 中的值，并存储到 roundup_settings 中
    for (const auto& v : snapshot.config_metadata.roundup_power2_divisions) {
        // 将 roundup_key 转换为字符串，并作为键存储在 roundup_settings 中
        py::str roundup_key_s = std::to_string(roundup_key);
        // 将 v 转换为 int64_t 类型后存储在 roundup_settings 中
        roundup_settings[roundup_key_s] = int64_t(v);
        // 更新 roundup_key 的值
        roundup_key *= 2;
    }
    // 将 roundup_settings 存储到 allocator_settings 中
    allocator_settings[roundup_power2_divisions_s] = roundup_settings;

    // 创建 Python 字典 result，用于存储最终结果
    py::dict result;
    // 将 segments 存储到 result 中
    result["segments"] = segments;
    // 将 traces 存储到 result 中
    result["device_traces"] = traces;
    // 将 allocator_settings 存储到 result 中
    result["allocator_settings"] = allocator_settings;

    // 使用 py_symbolize 函数对 to_gather_frames 中的数据进行符号化处理，并存储到 frames 中
    auto frames = py_symbolize(to_gather_frames);
    // 遍历 frames，并将其添加到 to_gather_dest 中对应位置的 trace_entry 中
    for (auto i : c10::irange(frames.size())) {
        to_gather_dest.at(i)[frames_s] = frames.at(i);
    }

    // 返回 result 的指针
    return result.release().ptr();
    // 处理 Torch 异常结束
    END_HANDLE_TH_ERRORS
// 定义一个名为 registerCudaDeviceProperties 的静态函数，接受一个 PyObject* 类型的参数 module
static void registerCudaDeviceProperties(PyObject* module) {
    // 将参数 module 转换为 py::module 类型的对象 m
    auto m = py::handle(module).cast<py::module>();
    // 在 torch._C 中添加 _cudaDevicePropertires 类
    // 该类用于表示 CUDA 设备的属性
    // 这是一个占位注释，具体内容需要根据函数实现的最终细节来补充
}
#ifndef FBCODE_CAFFE2
  // 如果未定义 FBCODE_CAFFE2，则定义以下内容：

  // 在 Python 中创建 _CUuuid 类，用于封装 CUuuid 结构体
  py::class_<CUuuid>(m, "_CUuuid")
      // 定义 bytes 属性，返回 CUuuid 结构体的字节表示为 std::vector<uint8_t>
      .def_property_readonly(
          "bytes",
          [](const CUuuid& uuid) {
            return std::vector<uint8_t>(uuid.bytes, uuid.bytes + 16);
          })
      // 定义 __str__ 方法，返回 CUuuid 结构体的十六进制字符串表示形式
      .def("__str__", [](const CUuuid& uuid) {
        // UUID 是一个128位标签。CUDA 和 HIP 将其存储为 char[16]。
        // 为了字符串表示，这段代码将其扩展为 8-4-4-4-12 的十六进制格式，
        // 每个字节转换为2个十六进制字符。
        // 字符串的大小为 16x2 个十六进制字符 + 4 个连字符 + 1 个空字节。
        constexpr size_t size = sizeof(CUuuid) * 2 + 4 + 1;
        char device_path_str[size] = {0};
        snprintf(
            device_path_str,
            sizeof(device_path_str),
            "%02x%02x%02x%02x-"
            "%02x%02x-"
            "%02x%02x-"
            "%02x%02x-"
            "%02x%02x%02x%02x%02x%02x",
            (uint8_t)uuid.bytes[0],
            (uint8_t)uuid.bytes[1],
            (uint8_t)uuid.bytes[2],
            (uint8_t)uuid.bytes[3],
            (uint8_t)uuid.bytes[4],
            (uint8_t)uuid.bytes[5],
            (uint8_t)uuid.bytes[6],
            (uint8_t)uuid.bytes[7],
            (uint8_t)uuid.bytes[8],
            (uint8_t)uuid.bytes[9],
            (uint8_t)uuid.bytes[10],
            (uint8_t)uuid.bytes[11],
            (uint8_t)uuid.bytes[12],
            (uint8_t)uuid.bytes[13],
            (uint8_t)uuid.bytes[14],
            (uint8_t)uuid.bytes[15]);
        return std::string(device_path_str);
      });
#endif

// 在 Python 中创建 _CudaDeviceProperties 类，用于封装 cudaDeviceProp 结构体
py::class_<cudaDeviceProp>(m, "_CudaDeviceProperties")
      // 定义只读属性 name，返回 cudaDeviceProp 结构体的设备名称
      .def_readonly("name", &cudaDeviceProp::name)
      // 定义只读属性 major，返回 cudaDeviceProp 结构体的主版本号
      .def_readonly("major", &cudaDeviceProp::major)
      // 定义只读属性 minor，返回 cudaDeviceProp 结构体的次版本号
      .def_readonly("minor", &cudaDeviceProp::minor)
      // 定义只读属性 is_multi_gpu_board，返回 cudaDeviceProp 结构体的多 GPU 板标志
      .def_readonly("is_multi_gpu_board", &cudaDeviceProp::isMultiGpuBoard)
      // 定义只读属性 is_integrated，返回 cudaDeviceProp 结构体的集成 GPU 标志
      .def_readonly("is_integrated", &cudaDeviceProp::integrated)
      // 定义只读属性 multi_processor_count，返回 cudaDeviceProp 结构体的多处理器数量
      .def_readonly(
          "multi_processor_count", &cudaDeviceProp::multiProcessorCount)
      // 定义只读属性 total_memory，返回 cudaDeviceProp 结构体的总全局内存量
      .def_readonly("total_memory", &cudaDeviceProp::totalGlobalMem)
      // 定义只读属性 max_threads_per_multi_processor，返回 cudaDeviceProp 结构体的每个多处理器的最大线程数
      .def_readonly(
          "max_threads_per_multi_processor",
          &cudaDeviceProp::maxThreadsPerMultiProcessor)
#if !USE_ROCM
      // 如果不使用 ROCm，则定义只读属性 regs_per_multiprocessor，返回 cudaDeviceProp 结构体的每个多处理器的寄存器数量
      .def_readonly(
          "regs_per_multiprocessor", &cudaDeviceProp::regsPerMultiprocessor)
#endif // USE_ROCM
      // 如果不使用 ROCm，则定义只读属性 gcnArchName，返回 cudaDeviceProp 结构体的 GPU 架构名称
      // 如果使用 ROCm，则重用 name 属性作为 CUDA 构建的 GPU 架构名称
      .def_readonly(
          "gcnArchName",
#if USE_ROCM
          &cudaDeviceProp::gcnArchName
#else
          &cudaDeviceProp::name
#endif // USE_ROCM
          )
#ifndef FBCODE_CAFFE2
      // 如果未定义 FBCODE_CAFFE2，则定义只读属性 uuid，返回 cudaDeviceProp 结构体的设备 UUID
      .def_readonly("uuid", &cudaDeviceProp::uuid)
#endif
      // 定义 __repr__ 方法，返回 cudaDeviceProp 结构体的字符串表示形式
      .def("__repr__", [](const cudaDeviceProp& prop) {
        std::ostringstream stream;
        stream << "_CudaDeviceProperties(name='" << prop.name
               << "', major=" << prop.major << ", minor=" << prop.minor
               // 继续串联其它属性到输出流中
// 如果定义了 USE_ROCM 宏，则将 prop.gcnArchName 信息附加到输出流中
#if USE_ROCM
               << ", gcnArchName='" << prop.gcnArchName << "'"
#endif // USE_ROCM

// 输出 GPU 属性信息：总内存大小、多处理器数量，以及可能的设备 UUID
               << ", total_memory=" << prop.totalGlobalMem / (1024ull * 1024)
               << "MB, multi_processor_count=" << prop.multiProcessorCount
#ifndef FBCODE_CAFFE2
               << ", uuid=" << std::string(prop.uuid.bytes, 16)
#endif

// 返回描述 GPU 设备的字符串
               << ")";

// 返回构建的描述 GPU 设备的完整字符串
        return stream.str();
      });

// 定义一个绑定到 Python 的函数 "_cuda_record_memory_history_legacy"，用于记录内存历史
  m.def(
      "_cuda_record_memory_history_legacy",
      static_cast<void (*)(bool, bool, int64_t, bool, bool)>(
          torch::cuda::_record_memory_history));

// 定义一个绑定到 Python 的函数 "_cuda_record_memory_history"，用于记录内存历史
  m.def(
      "_cuda_record_memory_history",
      static_cast<void (*)(
          std::optional<std::string>,
          std::optional<std::string>,
          const std::string&,
          size_t)>(torch::cuda::_record_memory_history));

// 定义一个绑定到 Python 的函数 "_cuda_isHistoryEnabled"，用于检查内存历史记录是否启用
  m.def("_cuda_isHistoryEnabled", []() {
    return c10::cuda::CUDACachingAllocator::isHistoryEnabled();
  });

// 定义一个绑定到 Python 的函数 "_cuda_get_conv_benchmark_empty_cache"，获取空缓存时的卷积性能基准
  m.def("_cuda_get_conv_benchmark_empty_cache", []() {
    return at::native::_cudnn_get_conv_benchmark_empty_cache();
  });

// 定义一个绑定到 Python 的函数 "_cudnn_set_conv_benchmark_empty_cache"，设置是否启用空缓存时的卷积性能基准
  m.def("_cudnn_set_conv_benchmark_empty_cache", [](bool enable) {
    return at::native::_cudnn_set_conv_benchmark_empty_cache(enable);
  });
}

// 移除一些当前已分配的块，以备份到其检查点时，对应的删除器函数需进行交换
void removeStorageDeleterFns(
    const std::vector<c10::StorageImpl*>& stale_live_storages,
    std::unordered_set<void*> definitely_stale_pointers) {
  for (c10::StorageImpl* stale_storage : stale_live_storages) {
    auto ptr = stale_storage->data_ptr().get();
    auto allocated_pointer = definitely_stale_pointers.find(ptr);
    TORCH_CHECK(allocated_pointer != definitely_stale_pointers.end());
    auto t = c10::cuda::CUDACachingAllocator::get();
    bool succeeded = stale_storage->mutable_data_ptr().compare_exchange_deleter(
        t->raw_deleter(), &c10::detail::deleteNothing);

    TORCH_CHECK(
        succeeded,
        "Unexpected deleter function on storage, could not swap function");
  }
}

// 为一组存储添加删除器函数，使用给定的检查点增量对象
void addStorageDeleterFns(
    std::vector<c10::StorageImpl*>& storages_to_add_deleters_to,
    c10::cuda::CUDACachingAllocator::CheckpointDelta& delta) {
  std::unordered_map<void*, c10::StorageImpl*> storages;
  for (auto& storage : storages_to_add_deleters_to) {
    storages[storage->data_ptr().get()] = storage;
  }

  for (auto& data_ptr : delta.dataptrs_allocd) {
    auto storage_pair = storages.find(data_ptr.get());
    if (storage_pair != storages.end()) {
      auto ctx = storage_pair->second->data_ptr().get_context();
      TORCH_CHECK(ctx == nullptr, " Not expecting deleter function");
      storage_pair->second->set_data_ptr_noswap(std::move(data_ptr));
    } else {
      data_ptr.release_context();
    }
  }
}
static void registerCudaPluggableAllocator(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // 创建名为 _cuda_CUDAAllocator 的 Python 类，表示 CUDA 分配器
  py::class_<
      c10::cuda::CUDACachingAllocator::CUDAAllocator,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>>(
      m, "_cuda_CUDAAllocator");

  // 定义 Python 函数 _cuda_getAllocator，返回自定义 CUDA 分配器
  m.def("_cuda_getAllocator", []() {
    // 定义 malloc_fn 和 free_fn 函数指针类型，分别指向 malloc_ptr 和 free_ptr
    using MallocFuncType = void*(size_t, int, cudaStream_t);
    using FreeFuncType = void(void*, size_t, int, cudaStream_t);
    std::function<MallocFuncType> malloc_fn =
        // 将 malloc_ptr 转换为 MallocFuncType* 类型
        reinterpret_cast<MallocFuncType*>(malloc_ptr);
    std::function<FreeFuncType> free_fn =
        // 将 free_ptr 转换为 FreeFuncType* 类型
        reinterpret_cast<FreeFuncType*>(free_ptr);
    // 使用 torch::cuda::CUDAPluggableAllocator 创建自定义分配器
    return torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
        malloc_fn, free_fn);
  });

  // 创建名为 _cuda_CUDAAllocator_AllocatorState 的 Python 类，表示 CUDA 分配器状态
  py::class_<
      c10::cuda::CUDACachingAllocator::AllocatorState,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>>(
      m, "_cuda_CUDAAllocator_AllocatorState");

  // 定义 Python 函数 _cuda_getCheckpointState，获取指定设备和内存池的检查点状态
  m.def(
      "_cuda_getCheckpointState",
      [](c10::DeviceIndex device, c10::cuda::MempoolId_t id) {
        return c10::cuda::CUDACachingAllocator::getCheckpointState(device, id);
      });

  // 定义 Python 函数 _free_And_Remove_DeleterFn，释放并移除存储实现的自定义删除器函数
  m.def("_free_And_Remove_DeleterFn", [](size_t storage_impl_ptr) {
    // 将 storage_impl_ptr 转换为 StorageImpl* 类型
    c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
    auto alloc = c10::cuda::CUDACachingAllocator::get();
    // 获取存储实现的数据指针并执行自定义删除器的比较交换
    auto data_ptr = storage_impl->data_ptr().get();
    bool succeeded = storage_impl->mutable_data_ptr().compare_exchange_deleter(
        alloc->raw_deleter(), c10::detail::deleteNothing);
    TORCH_CHECK(succeeded, "Expected standard deleter");
    // 使用 CUDA 缓存分配器的 raw_delete 方法释放数据指针
    c10::cuda::CUDACachingAllocator::raw_delete(data_ptr);
  });

  // 定义 Python 函数 _set_storage_access_error_msg，设置存储访问错误消息
  m.def(
      "_set_storage_access_error_msg", [](const at::Tensor& t, std::string s) {
        // 释放存储并设置自定义数据指针错误消息
        t.unsafeGetTensorImpl()
            ->release_storage_and_set_meta_custom_data_ptr_error_msg_(s);
      });

  // 定义 Python 函数 _has_Standard_Deleter，检查存储实现是否使用标准删除器
  m.def("_has_Standard_Deleter", [](size_t storage_impl_ptr) {
    // 将 storage_impl_ptr 转换为 StorageImpl* 类型
    c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
    auto alloc = c10::cuda::CUDACachingAllocator::get();
    // 检查存储实现的数据指针是否使用标准删除器
    return (storage_impl->data_ptr().get_deleter() == alloc->raw_deleter());
  });

  // 定义 Python 函数 _set_cached_tensors_enabled，启用或禁用缓存张量
  m.def("_set_cached_tensors_enabled", [](bool enabled) {
    at::caching::set_cached_tensors_enabled(enabled);
  });

  // 定义 Python 函数 _add_cached_tensor，向缓存中添加张量
  m.def("_add_cached_tensor", [](const at::Tensor& t) {
    at::caching::add_cached_tensor(t);
  });

  // 定义 Python 函数 _remove_cached_tensor，从缓存中移除张量
  m.def("_remove_cached_tensor", [](const at::Tensor& t) {
    at::caching::remove_cached_tensor(t);
  });

  // 定义 Python 函数 _is_cached_tensor，检查张量是否在缓存中
  m.def("_is_cached_tensor", [](const at::Tensor& t) {
    return at::caching::is_cached_tensor(t);
  });

  // 定义 Python 函数 _storage_Use_Count，获取存储实现的引用计数
  m.def("_storage_Use_Count", [](size_t storage_impl_ptr) {
    // 将 storage_impl_ptr 转换为 StorageImpl* 类型
    c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
    // 返回存储实现的数据指针的引用计数
    return storage_impl->use_count();
  });
}
    # 将 void 指针 storage_impl_ptr 强制转换为 c10::StorageImpl* 类型，存储在 storage_impl 中
    c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
}

// 定义静态方法，将 _get_device_properties 方法绑定到 torch.cuda 模块中
static void bindGetDeviceProperties(PyObject* module) {
  // 将 module 转换为 py::module 类型
  auto m = py::handle(module).cast<py::module>();
  // 定义 _get_device_properties 方法，返回 CUDA 设备的属性指针
  m.def(
      "_get_device_properties",
      [](c10::DeviceIndex device) -> cudaDeviceProp* {
        return at::cuda::getDeviceProperties(device);
      },
      py::return_value_policy::reference);
}

// Python 初始化回调函数，用于扩展初始化
static PyObject* THCPModule_initExtension(PyObject* self, PyObject* noargs) {
#if C10_ASAN_ENABLED
  // 如果启用地址检查器，则发出警告
  TORCH_WARN(
      "torch.cuda: your pytorch binary has address sanitizer (asan) built in, "
      "asan is currently not compatible with torch.cuda module, "
      "you might get unexpected behavior (eg. out of memory, crash, etc.), "
      "please rebuild pytorch without asan if you need to use this module");
#endif
  HANDLE_TH_ERRORS
  // 断言当前没有处于坏的 fork 状态
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level
  // 执行 fork 毒化操作
  poison_fork();
  // 惰性初始化 CUDA 上下文
  at::globalContext().lazyInitCUDA();

  // 导入 torch.cuda 模块
  auto m = THPObjectPtr(PyImport_ImportModule("torch.cuda"));
  if (!m)
    throw python_error();

  // 设置模块属性的 lambda 函数
  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString 不会窃取引用，因此不需要增加引用计数
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  // 获取 CUDA 设备数量
  auto num_gpus = c10::cuda::device_count();
  // 创建默认 CUDA 生成器的元组对象
  auto default_cuda_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  // 为每个 CUDA 设备初始化默认生成器，并放入元组中
  for (const auto i : c10::irange(num_gpus)) {
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(
        at::cuda::detail::getDefaultCUDAGenerator(i));
    // 这里的引用是要交出去的，所以不需要在此增加引用计数
    PyTuple_SetItem(default_cuda_generators, i, (PyObject*)cast_gen);
  }
  // 设置模块属性 default_generators
  set_module_attr("default_generators", default_cuda_generators);
  // 绑定 _get_device_properties 方法到 torch.cuda 模块
  bindGetDeviceProperties(m);

  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 获取当前 BLAS 句柄的 Python 封装函数
PyObject* THCPModule_getCurrentBlasHandle_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 获取当前 CUDA BLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 将句柄转换为 PyLong 对象并返回
  return PyLong_FromVoidPtr(handle);
  END_HANDLE_TH_ERRORS
}

// 清除 BLAS 工作空间的 Python 封装函数
static PyObject* THCPModule_clearBlasWorkspaces_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 清除 CUDA BLAS 工作空间
  at::cuda::clearCublasWorkspaces();
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 检查是否处于 ROCm 后向传递的 Python 封装函数
PyObject* THCPModule_rocm_is_backward_pass(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
#if USE_ROCM
  // 如果是 ROCm 后向传递，则返回 True
  if (at::ROCmBackwardPassGuard::is_backward_pass()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
#else
  // 如果不是 ROCm 后向传递，则返回 False
  Py_RETURN_FALSE;
#endif
  END_HANDLE_TH_ERRORS
}
PyObject* THCPModule_cuda_tunableop_enable(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔型，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkBool(arg),
      "cuda_tunableop_enable expects a bool, but got ",
      THPUtils_typename(arg));
  // 调用CUDA的可调优操作接口，根据参数设置是否启用
  at::cuda::tunable::getTuningContext()->EnableTunableOp(
      THPUtils_unpackBool(arg));
  // 返回None对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_is_enabled(
    PyObject* _unused,
    PyObject* noarg) {
  HANDLE_TH_ERRORS
  // 检查CUDA的可调优操作是否已启用，返回相应的Python布尔值
  if (at::cuda::tunable::getTuningContext()->IsTunableOpEnabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_tuning_enable(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔型，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkBool(arg),
      "cuda_tunableop_tuning_enable expects a bool, but got ",
      THPUtils_typename(arg));
  // 设置CUDA的可调优操作是否启用
  at::cuda::tunable::getTuningContext()->EnableTuning(THPUtils_unpackBool(arg));
  // 返回None对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_tuning_is_enabled(
    PyObject* _unused,
    PyObject* noarg) {
  HANDLE_TH_ERRORS
  // 检查CUDA的可调优操作是否已启用，返回相应的Python布尔值
  if (at::cuda::tunable::getTuningContext()->IsTuningEnabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_write_file_on_exit(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为布尔型，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkBool(arg),
      "cuda_tunableop_write_file_on_exit expects a bool, but got ",
      THPUtils_typename(arg));
  // 设置CUDA的可调优操作是否在退出时写文件
  at::cuda::tunable::getTuningContext()->WriteFileOnExit(
      THPUtils_unpackBool(arg));
  // 返回None对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_set_max_tuning_duration(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为长整型，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "cuda_tunableop_set_max_tuning_duration expects an int, but got ",
      THPUtils_typename(arg));
  // 将参数解包为整型，并设置为CUDA的最大调优持续时间（毫秒）
  auto duration = static_cast<int>(THPUtils_unpackLong(arg));
  at::cuda::tunable::getTuningContext()->SetMaxTuningDurationMs(duration);
  // 返回None对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_get_max_tuning_duration(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回CUDA的最大调优持续时间（毫秒）的整型封装
  return THPUtils_packInt32(
      at::cuda::tunable::getTuningContext()->GetMaxTuningDurationMs());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_set_max_tuning_iterations(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为长整型，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "cuda_tunableop_set_max_tuning_iterations expects an int, but got ",
      THPUtils_typename(arg));
  // 将参数解包为整型，并设置为CUDA的最大调优迭代次数
  auto iterations = static_cast<int>(THPUtils_unpackLong(arg));
  at::cuda::tunable::getTuningContext()->SetMaxTuningIterations(iterations);
  // 返回None对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_get_max_tuning_iterations(
    PyObject* _unused,
    ```
    // 定义一个函数 PyObject* noargs)，这里 PyObject* 是返回类型，表示返回一个 PyObject 对象指针
    PyObject* noargs) {
  // 开始捕获可能抛出的异常
  HANDLE_TH_ERRORS
  // 调用 THPUtils_packInt32 函数，将 at::cuda::tunable::getTuningContext()->GetMaxTuningIterations() 的结果打包成一个 int32 的 PyObject 对象，并返回
  return THPUtils_packInt32(
      at::cuda::tunable::getTuningContext()->GetMaxTuningIterations());
  // 结束对异常的处理
  END_HANDLE_TH_ERRORS
PyObject* THCPModule_cuda_tunableop_set_filename(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  // 定义存储字符串对象和布尔对象的指针
  PyObject* obj_str = nullptr;
  PyObject* obj_ord = nullptr;
  // 解析 Python 元组参数，期望第一个参数是字符串，第二个参数是可选的布尔值
  if (!PyArg_ParseTuple(args, "O|O", &obj_str, &obj_ord)) {
  }
  // 检查第一个参数是否为字符串类型，否则抛出异常
  TORCH_CHECK(
      THPUtils_checkString(obj_str),
      "cuda_tunableop_set_filename expects a string, but got ",
      THPUtils_typename(obj_str));
  // 解包字符串对象，获取文件名
  auto filename = THPUtils_unpackString(obj_str);
  // 初始化一个布尔变量，用于指示是否应用于特定设备
  bool dev = false;
  // 如果第二个参数存在，检查其是否为布尔类型，如果是则解包为布尔值
  if (obj_ord) {
    TORCH_CHECK(
        THPUtils_checkBool(obj_ord),
        "cuda_tunableop_set_filename expects a bool, but got ",
        THPUtils_typename(obj_ord));
    dev = THPUtils_unpackBool(obj_ord);
  }
  // 调用 CUDA 可调整操作的设置文件名方法，传递文件名和设备布尔值
  at::cuda::tunable::getTuningContext()->SetFilename(filename, dev);
  // 返回 None 对象表示成功执行
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_get_filename(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 返回当前 CUDA 可调整操作的文件名作为 Python 字符串对象
  return THPUtils_packString(
      at::cuda::tunable::getTuningContext()->GetFilename());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_write_file(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  // 定义一个指向 Python 对象的指针，用于存储文件名字符串
  PyObject* str = nullptr;
  // 初始化成功标志为假
  bool success = false;
  // 解析 Python 元组参数，期望第一个参数是可选的字符串对象
  if (!PyArg_ParseTuple(args, "|O", &str)) {
  }
  // 如果字符串对象存在
  if (str) {
    // 检查字符串对象是否为字符串类型，否则抛出异常
    TORCH_CHECK(
        THPUtils_checkString(str),
        "cuda_tunableop_write_file expects a string, but got ",
        THPUtils_typename(str));
    // 解包字符串对象，获取文件名
    auto filename = THPUtils_unpackString(str);
    // 调用 CUDA 可调整操作的写入文件方法，传递文件名
    success = at::cuda::tunable::getTuningContext()->WriteFile(filename);
  } else {
    // 如果字符串对象不存在，调用无参数版本的写入文件方法
    success = at::cuda::tunable::getTuningContext()->WriteFile();
  }
  // 如果写入成功，返回 Python 的 True 对象，否则返回 False 对象
  if (success) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_read_file(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  // 定义一个指向 Python 对象的指针，用于存储文件名字符串
  PyObject* str = nullptr;
  // 初始化成功标志为假
  bool success = false;
  // 解析 Python 元组参数，期望第一个参数是可选的字符串对象
  if (!PyArg_ParseTuple(args, "|O", &str)) {
  }
  // 如果字符串对象存在
  if (str) {
    // 检查字符串对象是否为字符串类型，否则抛出异常
    TORCH_CHECK(
        THPUtils_checkString(str),
        "cuda_tunableop_read_file expects a string, but got ",
        THPUtils_typename(str));
    // 解包字符串对象，获取文件名
    auto filename = THPUtils_unpackString(str);
    // 调用 CUDA 可调整操作的读取文件方法，传递文件名
    success = at::cuda::tunable::getTuningContext()->ReadFile(filename);
  } else {
    // 如果字符串对象不存在，调用无参数版本的读取文件方法
    success = at::cuda::tunable::getTuningContext()->ReadFile();
  }
  // 如果读取成功，返回 Python 的 True 对象，否则返回 False 对象
  if (success) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cuda_tunableop_get_results(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 调用 CUDA 可调整操作的获取调整结果管理器的 Dump 方法，获取所有结果
  auto results =
      at::cuda::tunable::getTuningContext()->GetTuningResultsManager().Dump();
  // 初始化结果大小为 0
  size_t result_size = 0;
  // 遍历结果，计算总共的内核映射数
  for (const auto& [op_sig, kernelmap] : results) {
    result_size += kernelmap.size();
  }
  // 创建一个 Python 元组，用于存储所有的调整结果
  THPObjectPtr outer_tuple(PyTuple_New(result_size));
  // 如果创建元组失败，抛出 Python 异常
  if (!outer_tuple)
    throw python_error();
  // 初始化结果索引为 0
  size_t result_index = 0;
  // 遍历结果
  for (const auto& [op_sig, kernelmap] : results) {
    // 遍历 kernelmap 中的每对键值对 (param_sig, result)
    for (const auto& [param_sig, result] : kernelmap) {
      // 创建一个长度为 4 的 Python 元组对象 inner_tuple
      THPObjectPtr inner_tuple(PyTuple_New(4));
      if (!inner_tuple)
        throw python_error();
      // 将操作符签名 op_sig 打包成 Python 字符串对象
      PyObject* obj_op_sig = THPUtils_packString(op_sig);
      if (!obj_op_sig)
        throw python_error();
      // 将参数签名 param_sig 打包成 Python 字符串对象
      PyObject* obj_param_sig = THPUtils_packString(param_sig);
      if (!obj_param_sig)
        throw python_error();
      // 将结果的键 (result.GetKey()) 打包成 Python 字符串对象
      PyObject* obj_result_key = THPUtils_packString(result.GetKey());
      if (!obj_result_key)
        throw python_error();
      // 将结果的时间 (result.GetTime()) 打包成 Python 浮点数对象
      PyObject* obj_result_time = PyFloat_FromDouble(result.GetTime());
      if (!obj_result_time)
        throw python_error();
      // 将打包好的对象依次放入 inner_tuple 中
      PyTuple_SET_ITEM(inner_tuple.get(), 0, obj_op_sig);
      PyTuple_SET_ITEM(inner_tuple.get(), 1, obj_param_sig);
      PyTuple_SET_ITEM(inner_tuple.get(), 2, obj_result_key);
      PyTuple_SET_ITEM(inner_tuple.get(), 3, obj_result_time);
      // 将 inner_tuple 的所有权释放并存入 outer_tuple 的 result_index 位置
      PyTuple_SET_ITEM(
          outer_tuple.get(), result_index++, inner_tuple.release());
    }
  }
  // 返回包含结果的 outer_tuple，所有权转移给调用者
  return outer_tuple.release();
  // 处理 Python 中的异常并结束处理
  END_HANDLE_TH_ERRORS
// 返回 CUDA 可调优操作的验证器元组
PyObject* THCPModule_cuda_tunableop_get_validators(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 获取调优上下文中的验证器列表
  auto validators = at::cuda::tunable::getTuningContext()
                        ->GetTuningResultsValidator()
                        .GetAllValidators();
  // 创建外部元组，其长度为验证器列表的大小
  THPObjectPtr outer_tuple(PyTuple_New(validators.size()));
  if (!outer_tuple)
    throw python_error();
  size_t validator_index = 0;
  // 遍历验证器列表，创建并填充内部元组
  for (const auto& [key, val] : validators) {
    THPObjectPtr inner_tuple(PyTuple_New(2));
    if (!inner_tuple)
      throw python_error();
    // 打包验证器的键为字符串对象
    PyObject* obj_key = THPUtils_packString(key);
    if (!obj_key)
      throw python_error();
    // 打包验证器的值为字符串对象
    PyObject* obj_val = THPUtils_packString(val);
    if (!obj_val)
      throw python_error();
    // 设置内部元组的第一个和第二个元素为键和值对象
    PyTuple_SET_ITEM(inner_tuple.get(), 0, obj_key);
    PyTuple_SET_ITEM(inner_tuple.get(), 1, obj_val);
    // 将填充好的内部元组放入外部元组的对应位置
    PyTuple_SET_ITEM(
        outer_tuple.get(), validator_index++, inner_tuple.release());
  }
  // 返回外部元组对象
  return outer_tuple.release();
  END_HANDLE_TH_ERRORS
}

// 包装函数，用于检查当前 CUDA 流是否处于捕获状态
static PyObject* THCPModule_isCurrentStreamCapturing_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 如果没有 CUDA 上下文或当前流不在捕获状态，返回 False
  if (at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None) {
    Py_RETURN_FALSE;
  } else {
    // 否则返回 True
    Py_RETURN_TRUE;
  }
  END_HANDLE_TH_ERRORS
}

// 设置 cuDNN 的基准限制
PyObject* THCPModule_setBenchmarkLimitCuDNN(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为整数类型
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "set_benchmark_limit_cudnn expects an int, "
      "but got ",
      THPUtils_typename(arg));
  // 如果是 ROCm 平台，发出一次警告，因为 MIOpen 不支持 cuDNN 的基准限制
#if defined(USE_ROCM)
  TORCH_WARN_ONCE(
      "cuDNN Benchmark limit is not supported in MIOpen and will have no effect.");
#endif
  // 将参数转换为整数并设置为 cuDNN 的基准限制
  auto benchmark_limit = static_cast<int>(THPUtils_unpackLong(arg));
  at::globalContext().setBenchmarkLimitCuDNN(benchmark_limit);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 获取 cuDNN 的基准限制
PyObject* THCPModule_benchmarkLimitCuDNN(PyObject* _unused, PyObject* noargs) {
  // 返回当前 cuDNN 的基准限制作为整数对象
  return THPUtils_packInt32(at::globalContext().benchmarkLimitCuDNN());
}
    {"_cuda_getCurrentStream",
     THCPModule_getCurrentStream_wrap,
     METH_O,
     nullptr},
    // 函数名: "_cuda_getCurrentStream"
    // 绑定的 C 函数: THCPModule_getCurrentStream_wrap
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_getCurrentRawStream",
     THCPModule_getCurrentStream_raw,
     METH_O,
     nullptr},
    // 函数名: "_cuda_getCurrentRawStream"
    // 绑定的 C 函数: THCPModule_getCurrentStream_raw
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_getDefaultStream",
     THCPModule_getDefaultStream_wrap,
     METH_O,
     nullptr},
    // 函数名: "_cuda_getDefaultStream"
    // 绑定的 C 函数: THCPModule_getDefaultStream_wrap
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_getCurrentBlasHandle",
     THCPModule_getCurrentBlasHandle_wrap,
     METH_NOARGS,
     nullptr},
    // 函数名: "_cuda_getCurrentBlasHandle"
    // 绑定的 C 函数: THCPModule_getCurrentBlasHandle_wrap
    // 参数类型: METH_NOARGS (不接受任何参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_clearCublasWorkspaces",
     THCPModule_clearBlasWorkspaces_wrap,
     METH_NOARGS,
     nullptr},
    // 函数名: "_cuda_clearCublasWorkspaces"
    // 绑定的 C 函数: THCPModule_clearBlasWorkspaces_wrap
    // 参数类型: METH_NOARGS (不接受任何参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_isCurrentStreamCapturing",
     THCPModule_isCurrentStreamCapturing_wrap,
     METH_NOARGS,
     nullptr},
    // 函数名: "_cuda_isCurrentStreamCapturing"
    // 绑定的 C 函数: THCPModule_isCurrentStreamCapturing_wrap
    // 参数类型: METH_NOARGS (不接受任何参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_setStream",
     castPyCFunctionWithKeywords(THCPModule_setStream_wrap),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    // 函数名: "_cuda_setStream"
    // 绑定的 C 函数: castPyCFunctionWithKeywords(THCPModule_setStream_wrap)
    // 参数类型: METH_VARARGS | METH_KEYWORDS (接受可变位置参数和关键字参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_getCompiledVersion",
     THCPModule_getCompiledVersion,
     METH_NOARGS,
     nullptr},
    // 函数名: "_cuda_getCompiledVersion"
    // 绑定的 C 函数: THCPModule_getCompiledVersion
    // 参数类型: METH_NOARGS (不接受任何参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_hasPrimaryContext", THCPModule_hasPrimaryContext, METH_O, nullptr},
    // 函数名: "_cuda_hasPrimaryContext"
    // 绑定的 C 函数: THCPModule_hasPrimaryContext
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_setMemoryFraction",
     THCPModule_setMemoryFraction,
     METH_VARARGS,
     nullptr},
    // 函数名: "_cuda_setMemoryFraction"
    // 绑定的 C 函数: THCPModule_setMemoryFraction
    // 参数类型: METH_VARARGS (接受可变位置参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_emptyCache", THCPModule_emptyCache, METH_NOARGS, nullptr},
    // 函数名: "_cuda_emptyCache"
    // 绑定的 C 函数: THCPModule_emptyCache
    // 参数类型: METH_NOARGS (不接受任何参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_memoryStats", THCPModule_memoryStats, METH_O, nullptr},
    // 函数名: "_cuda_memoryStats"
    // 绑定的 C 函数: THCPModule_memoryStats
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_resetAccumulatedMemoryStats",
     THCPModule_resetAccumulatedMemoryStats,
     METH_O,
     nullptr},
    // 函数名: "_cuda_resetAccumulatedMemoryStats"
    // 绑定的 C 函数: THCPModule_resetAccumulatedMemoryStats
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_resetPeakMemoryStats",
     THCPModule_resetPeakMemoryStats,
     METH_O,
     nullptr},
    // 函数名: "_cuda_resetPeakMemoryStats"
    // 绑定的 C 函数: THCPModule_resetPeakMemoryStats
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_memorySnapshot", THCPModule_memorySnapshot, METH_NOARGS, nullptr},
    // 函数名: "_cuda_memorySnapshot"
    // 绑定的 C 函数: THCPModule_memorySnapshot
    // 参数类型: METH_NOARGS (不接受任何参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_attach_out_of_memory_observer",
     THCPModule_attachOutOfMemoryObserver,
     METH_O,
     nullptr},
    // 函数名: "_cuda_attach_out_of_memory_observer"
    // 绑定的 C 函数: THCPModule_attachOutOfMemoryObserver
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_cudaHostAllocator",
     THCPModule_cudaHostAllocator,
     METH_NOARGS,
     nullptr},
    // 函数名: "_cuda_cudaHostAllocator"
    // 绑定的 C 函数: THCPModule_cudaHostAllocator
    // 参数类型: METH_NOARGS (不接受任何参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_cudaCachingAllocator_raw_alloc",
     THCPModule_cudaCachingAllocator_raw_alloc,
     METH_VARARGS,
     nullptr},
    // 函数名: "_cuda_cudaCachingAllocator_raw_alloc"
    // 绑定的 C 函数: THCPModule_cudaCachingAllocator_raw_alloc
    // 参数类型: METH_VARARGS (接受可变位置参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_cudaCachingAllocator_raw_delete",
     THCPModule_cudaCachingAllocator_raw_delete,
     METH_O,
     nullptr},
    // 函数名: "_cuda_cudaCachingAllocator_raw_delete"
    // 绑定的 C 函数: THCPModule_cudaCachingAllocator_raw_delete
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_cudaCachingAllocator_set_allocator_settings",
     THCPModule_cudaCachingAllocator_set_allocator_settings,
     METH_O,
     nullptr},
    // 函数名: "_cuda_cudaCachingAllocator_set_allocator_settings"
    // 绑定的 C 函数: THCPModule_cudaCachingAllocator_set_allocator_settings
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_getAllocatorBackend",
     THCPModule_getAllocatorBackend,
     METH_NOARGS,
     nullptr},
    // 函数名: "_cuda_getAllocatorBackend"
    // 绑定的 C 函数: THCPModule_getAllocatorBackend
    // 参数类型: METH_NOARGS (不接受任何参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_synchronize", THCPModule_cudaSynchronize, METH_NOARGS, nullptr},
    // 函数名: "_cuda_synchronize"
    // 绑定的 C 函数: THCPModule_cudaSynchronize
    // 参数类型: METH_NOARGS (不接受任何参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_ipc_collect", THCPModule_cudaIPCCollect, METH_NOARGS, nullptr},
    // 函数名: "_cuda_ipc_collect"
    // 绑定的 C 函数: THCPModule_cudaIPCCollect
    // 参数类型: METH_NOARGS (不接受任何参数)
    // 默认值: nullptr (无默认值)
    {"_cuda_sleep", THCPModule_cudaSleep, METH_O, nullptr},
    // 函数名: "_cuda_sleep"
    // 绑定的 C 函数: THCPModule_cudaSleep
    // 参数类型: METH_O (接受一个 Python 对象作为参数)
    // 默认值: nullptr (无默认值)
    {"_cuda
    {
        "_cuda_get_cudnn_benchmark_limit",
        THCPModule_benchmarkLimitCuDNN,
        METH_NOARGS,
        nullptr
    },
    
    
    
    # 键为 "_cuda_get_cudnn_benchmark_limit"，值为 THCPModule_benchmarkLimitCuDNN 函数的指针，
    # 该函数没有参数 (METH_NOARGS)，nullptr 表示末尾标记为空
    
    
    
    {
        "_cuda_set_cudnn_benchmark_limit",
        THCPModule_setBenchmarkLimitCuDNN,
        METH_O,
        nullptr
    },
    
    
    
    # 键为 "_cuda_set_cudnn_benchmark_limit"，值为 THCPModule_setBenchmarkLimitCuDNN 函数的指针，
    # 该函数接受一个参数 (METH_O)，nullptr 表示末尾标记为空
#ifdef USE_NCCL
    {"_nccl_version", THCPModule_nccl_version, METH_NOARGS, nullptr},
    // 定义了一个名为 "_nccl_version" 的 Python 方法，使用 THCPModule_nccl_version 函数，不接受参数，返回空，无额外说明
    {"_nccl_version_suffix",
     THCPModule_nccl_version_suffix,
     METH_NOARGS,
     nullptr},
    // 定义了一个名为 "_nccl_version_suffix" 的 Python 方法，使用 THCPModule_nccl_version_suffix 函数，不接受参数，返回空，无额外说明
    {"_nccl_unique_id", THCPModule_nccl_unique_id, METH_NOARGS, nullptr},
    // 定义了一个名为 "_nccl_unique_id" 的 Python 方法，使用 THCPModule_nccl_unique_id 函数，不接受参数，返回空，无额外说明
    {"_nccl_init_rank", THCPModule_nccl_init_rank, METH_VARARGS, nullptr},
    // 定义了一个名为 "_nccl_init_rank" 的 Python 方法，使用 THCPModule_nccl_init_rank 函数，接受可变参数，返回空，无额外说明
    {"_nccl_reduce", THCPModule_nccl_reduce, METH_VARARGS, nullptr},
    // 定义了一个名为 "_nccl_reduce" 的 Python 方法，使用 THCPModule_nccl_reduce 函数，接受可变参数，返回空，无额外说明
    {"_nccl_all_reduce", THCPModule_nccl_all_reduce, METH_VARARGS, nullptr},
    // 定义了一个名为 "_nccl_all_reduce" 的 Python 方法，使用 THCPModule_nccl_all_reduce 函数，接受可变参数，返回空，无额外说明
    {"_nccl_broadcast", THCPModule_nccl_broadcast, METH_VARARGS, nullptr},
    // 定义了一个名为 "_nccl_broadcast" 的 Python 方法，使用 THCPModule_nccl_broadcast 函数，接受可变参数，返回空，无额外说明
    {"_nccl_all_gather", THCPModule_nccl_all_gather, METH_VARARGS, nullptr},
    // 定义了一个名为 "_nccl_all_gather" 的 Python 方法，使用 THCPModule_nccl_all_gather 函数，接受可变参数，返回空，无额外说明
    {"_nccl_reduce_scatter",
     THCPModule_nccl_reduce_scatter,
     METH_VARARGS,
     nullptr},
    // 定义了一个名为 "_nccl_reduce_scatter" 的 Python 方法，使用 THCPModule_nccl_reduce_scatter 函数，接受可变参数，返回空，无额外说明
#endif
    {"_rocm_is_backward_pass",
     THCPModule_rocm_is_backward_pass,
     METH_NOARGS,
     nullptr},
    // 定义了一个名为 "_rocm_is_backward_pass" 的 Python 方法，使用 THCPModule_rocm_is_backward_pass 函数，不接受参数，返回空，无额外说明
    {"_cuda_tunableop_enable",
     THCPModule_cuda_tunableop_enable,
     METH_O,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_enable" 的 Python 方法，使用 THCPModule_cuda_tunableop_enable 函数，接受一个参数，返回空，无额外说明
    {"_cuda_tunableop_is_enabled",
     THCPModule_cuda_tunableop_is_enabled,
     METH_NOARGS,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_is_enabled" 的 Python 方法，使用 THCPModule_cuda_tunableop_is_enabled 函数，不接受参数，返回空，无额外说明
    {"_cuda_tunableop_tuning_enable",
     THCPModule_cuda_tunableop_tuning_enable,
     METH_O,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_tuning_enable" 的 Python 方法，使用 THCPModule_cuda_tunableop_tuning_enable 函数，接受一个参数，返回空，无额外说明
    {"_cuda_tunableop_tuning_is_enabled",
     THCPModule_cuda_tunableop_tuning_is_enabled,
     METH_NOARGS,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_tuning_is_enabled" 的 Python 方法，使用 THCPModule_cuda_tunableop_tuning_is_enabled 函数，不接受参数，返回空，无额外说明
    {"_cuda_tunableop_write_file_on_exit",
     THCPModule_cuda_tunableop_write_file_on_exit,
     METH_O,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_write_file_on_exit" 的 Python 方法，使用 THCPModule_cuda_tunableop_write_file_on_exit 函数，接受一个参数，返回空，无额外说明
    {"_cuda_tunableop_set_max_tuning_duration",
     THCPModule_cuda_tunableop_set_max_tuning_duration,
     METH_O,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_set_max_tuning_duration" 的 Python 方法，使用 THCPModule_cuda_tunableop_set_max_tuning_duration 函数，接受一个参数，返回空，无额外说明
    {"_cuda_tunableop_get_max_tuning_duration",
     THCPModule_cuda_tunableop_get_max_tuning_duration,
     METH_NOARGS,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_get_max_tuning_duration" 的 Python 方法，使用 THCPModule_cuda_tunableop_get_max_tuning_duration 函数，不接受参数，返回空，无额外说明
    {"_cuda_tunableop_set_max_tuning_iterations",
     THCPModule_cuda_tunableop_set_max_tuning_iterations,
     METH_O,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_set_max_tuning_iterations" 的 Python 方法，使用 THCPModule_cuda_tunableop_set_max_tuning_iterations 函数，接受一个参数，返回空，无额外说明
    {"_cuda_tunableop_get_max_tuning_iterations",
     THCPModule_cuda_tunableop_get_max_tuning_iterations,
     METH_NOARGS,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_get_max_tuning_iterations" 的 Python 方法，使用 THCPModule_cuda_tunableop_get_max_tuning_iterations 函数，不接受参数，返回空，无额外说明
    {"_cuda_tunableop_set_filename",
     THCPModule_cuda_tunableop_set_filename,
     METH_VARARGS,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_set_filename" 的 Python 方法，使用 THCPModule_cuda_tunableop_set_filename 函数，接受可变参数，返回空，无额外说明
    {"_cuda_tunableop_get_filename",
     THCPModule_cuda_tunableop_get_filename,
     METH_NOARGS,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_get_filename" 的 Python 方法，使用 THCPModule_cuda_tunableop_get_filename 函数，不接受参数，返回空，无额外说明
    {"_cuda_tunableop_write_file",
     THCPModule_cuda_tunableop_write_file,
     METH_VARARGS,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_write_file" 的 Python 方法，使用 THCPModule_cuda_tunableop_write_file 函数，接受可变参数，返回空，无额外说明
    {"_cuda_tunableop_read_file",
     THCPModule_cuda_tunableop_read_file,
     METH_VARARGS,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_read_file" 的 Python 方法，使用 THCPModule_cuda_tunableop_read_file 函数，接受可变参数，返回空，无额外说明
    {"_cuda_tunableop_get_results",
     THCPModule_cuda_tunableop_get_results,
     METH_NOARGS,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_get_results" 的 Python 方法，使用 THCPModule_cuda_tunableop_get_results 函数，不接受参数，返回空，无额外说明
    {"_cuda_tunableop_get_validators",
     THCPModule_cuda_tunableop_get_validators,
     METH_NOARGS,
     nullptr},
    // 定义了一个名为 "_cuda_tunableop_get_validators" 的 Python 方法，使用 THCPModule_cuda_tunableop_get_validators 函数，不接受参数，返回空，无额外说明
    {nullptr}};
// 方法列表的最后一个元素，表示方法定义结束，为 nullptr

PyMethodDef* THCPModule_methods() {
  return _THCPModule_methods;
}
// 返回 THCPModule_methods 的指针，该指针指向上述定义的方法列表数组

namespace torch::cuda {

namespace shared {

void initCudartBindings(PyObject* module);
void initNvtxBindings(PyObject* module);
#if defined(USE_CUDNN) || defined(USE_ROCM)
void initCudnnBindings(PyObject* module);
#endif

} // namespace shared
void initModule(PyObject* module) {
  // 初始化通信方法，传入 Python 模块对象
  python::initCommMethods(module);
  
  // 初始化 CUDA runtime 绑定，这个文件可能会被用于 ROCm 平台编译，
  // 所以这个条件并不总是成立…
  shared::initCudartBindings(module);
  
  // 初始化 NVTX 绑定
  shared::initNvtxBindings(module);
  
  // 如果定义了 USE_CUDNN 或者 USE_ROCM，则初始化 CUDNN 绑定
#if defined(USE_CUDNN) || defined(USE_ROCM)
  shared::initCudnnBindings(module);
#endif
  
  // 注册 CUDA 设备属性到 Python 模块
  registerCudaDeviceProperties(module);
  
  // 注册可插拔的 CUDA 分配器到 Python 模块
  registerCudaPluggableAllocator(module);
}

} // namespace torch::cuda
```