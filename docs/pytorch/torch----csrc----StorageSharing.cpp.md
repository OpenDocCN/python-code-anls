# `.\pytorch\torch\csrc\StorageSharing.cpp`

```py
// 包含 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>
// 如果是在 MSVC 下编译，包含 Win32 头文件
#ifdef _MSC_VER
#include <c10/util/win32-headers.h>
#endif
// 包含 Python 结构成员头文件
#include <structmember.h>

// 包含 CPU 分配器的头文件
#include <c10/core/CPUAllocator.h>
// 包含共享内存库的头文件
#include <libshm.h>
// 包含 CUDA IPC 类型的头文件
#include <torch/csrc/CudaIPCTypes.h>
// 包含设备类型的头文件
#include <torch/csrc/Device.h>
// 包含动态类型的头文件
#include <torch/csrc/DynamicTypes.h>
// 包含 THP 头文件
#include <torch/csrc/THP.h>
// 包含自动微分工具的头文件
#include <torch/csrc/autograd/utils/wrap_outputs.h>
// 包含复制工具的头文件
#include <torch/csrc/copy_utils.h>

// 包含 C10 的侵入式指针头文件
#include <c10/util/intrusive_ptr.h>
// 使用 fmt 库的格式化功能
#include <fmt/format.h>

// 包含 Tensor 存储的头文件
#include <torch/csrc/Storage.h>
// 包含 Tensor 存储共享的头文件
#include <torch/csrc/StorageSharing.h>

#ifdef USE_CUDA
// 如果使用 CUDA，包含 CUDA 守卫头文件
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// 包含 Map 分配器的头文件
#include <ATen/MapAllocator.h>
// 包含存储工具的头文件
#include <ATen/StorageUtils.h>
// 包含 Python 数字处理工具的头文件
#include <torch/csrc/utils/python_numbers.h>
// 包含原子操作的头文件
#include <atomic>
// 包含字符串处理的头文件
#include <string>

// 定义静态函数 THPStorage_sharedDecref，用于减少共享引用计数
static PyObject* THPStorage_sharedDecref(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 确保 self 不为空
  THPStorage_assertNotNull(self);
  // 从 Python 对象中解包得到存储对象
  const auto& storage = THPStorage_Unpack(self);
  // 获取存储的设备类型
  c10::DeviceType device_type = storage.device_type();
  // 如果设备类型为 CPU
  if (device_type == at::kCPU) {
    // 从数据指针获取 THManagedMapAllocator 上下文
    THManagedMapAllocator* ctx =
        THManagedMapAllocator::fromDataPtr(storage.data_ptr());
    // 如果上下文不为空，减少引用计数
    if (ctx) {
      ctx->decref();
    }
  }
  // 增加 Python 对象的引用计数，避免被垃圾回收
  Py_INCREF(self);
  // 返回 Python 对象本身
  return self;
  END_HANDLE_TH_ERRORS
}

// 定义静态函数 THPStorage_sharedIncref，用于增加共享引用计数
static PyObject* THPStorage_sharedIncref(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 确保 self 不为空
  THPStorage_assertNotNull(self);
  // 从 Python 对象中解包得到存储对象
  const auto& storage = THPStorage_Unpack(self);
  // 获取存储的设备类型
  c10::DeviceType device_type = storage.device_type();
  // 如果设备类型为 CPU
  if (device_type == at::kCPU) {
    // 从数据指针获取 THManagedMapAllocator 上下文
    THManagedMapAllocator* ctx =
        THManagedMapAllocator::fromDataPtr(storage.data_ptr());
    // 如果上下文不为空，增加引用计数
    if (ctx) {
      ctx->incref();
    }
  }
  // 返回 None 表示操作成功
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 定义静态函数 THPStorage_pyNewFilenameStorage，用于创建新的文件名存储
static PyObject* THPStorage_pyNewFilenameStorage(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  long long size = 0;
  // 解析 Python 参数元组，获取存储大小
  if (!PyArg_ParseTuple(args, "L", &size)) {
    return nullptr;
  }
  // 如果存储大小小于 0，返回空指针
  if (size < 0) {
    return nullptr;
  }

  // 定义共享内存标志
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
  // 生成新的进程共享内存句柄
  std::string handle = at::NewProcessWideShmHandle();
  // 使用 THPStorage_NewWithStorage 创建新的存储对象，并返回
  return THPStorage_NewWithStorage(
      THPStorageClass,
      c10::make_intrusive<at::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size,
          // 使用 THManagedMapAllocator 创建数据指针
          THManagedMapAllocator::makeDataPtr(
              "", handle.c_str(), flags, static_cast<size_t>(size)),
          /*allocator=*/nullptr,
          /*resizable=*/false),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
  END_HANDLE_TH_ERRORS
}

// 定义静态函数 THPStorage_shareFilename，用于共享文件名存储
static PyObject* THPStorage_shareFilename(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 确保 self 不为空
  THPStorage_assertNotNull(self);
  // 从 Python 对象中解包得到存储对象
  const auto& storage = THPStorage_Unpack(self);
  // 检查存储的设备类型必须为 CPU
  TORCH_CHECK(
      storage.device_type() == at::kCPU,
      "_share_filename_: only available on CPU");
  // 从数据指针获取 THManagedMapAllocator 上下文
  THManagedMapAllocator* ctx =
      THManagedMapAllocator::fromDataPtr(storage.data_ptr());
  // 如果上下文已存在，直接返回句柄
  if (ctx) {
    // done
  } else {
    // 设置标志，指示要在共享内存中分配存储空间，并且独占该内存
    // TODO: 在发生冲突时重试
    // TODO: 释放全局解释器锁（GIL），但记得在抛出异常时重新获取它
    int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
    
    // 创建一个新的进程范围共享内存句柄
    std::string handle = at::NewProcessWideShmHandle();
    
    // 在共享内存中创建一个新的存储空间
    at::Storage new_storage(c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),                // 使用字节大小
        storage.nbytes(),                                   // 存储空间的字节大小
        THManagedMapAllocator::makeDataPtr(
            "", handle.c_str(), flags, storage.nbytes()),   // 使用 THManagedMapAllocator 创建数据指针
        /*allocator=*/nullptr,                              // 分配器为空
        /*resizable=*/false));                              // 不可调整大小

    {
      // 由于将数据复制到共享内存可能较慢，释放全局解释器锁
      pybind11::gil_scoped_release no_gil;
      
      // 将旧存储中的数据复制到新存储中
      at::storage_copy(new_storage, storage);
    }

    // 替换旧数据指针和分配器为新的数据指针和分配器
    storage.set_data_ptr(std::move(new_storage.mutable_data_ptr()));
    storage.unsafeGetStorageImpl()->set_allocator(new_storage.allocator());

    // 从数据指针中获取 THManagedMapAllocator 上下文
    ctx = THManagedMapAllocator::fromDataPtr(storage.data_ptr());
    AT_ASSERT(ctx);
  }

  // 使用 manager_handle 创建一个 Python 字节对象
  THPObjectPtr manager_handle(PyBytes_FromString(ctx->manager_handle()));
  if (!manager_handle)
    return nullptr;
  
  // 使用 storage_handle 创建一个 Python 字节对象
  THPObjectPtr storage_handle(PyBytes_FromString(ctx->filename()));
  if (!storage_handle)
    return nullptr;
  
  // 使用 storage.nbytes() 创建一个 Python 整数对象
  THPObjectPtr size(THPUtils_packUInt64(storage.nbytes()));
  if (!size)
    return nullptr;
  
  // 创建一个包含 manager_handle、storage_handle 和 size 的元组对象
  THPObjectPtr tuple(PyTuple_New(3));
  if (!tuple)
    return nullptr;
  PyTuple_SET_ITEM(tuple.get(), 0, manager_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 2, size.release());
  
  // 返回元组对象
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

// 创建新的共享文件名存储的 Python 对象
static PyObject* THPStorage_newSharedFilename(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  // 检查参数数量是否为3
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 3, "tuple of 3 items expected");
  // 获取参数中的管理器句柄、对象句柄和大小
  PyObject* _manager_handle = PyTuple_GET_ITEM(args, 0);
  PyObject* _object_handle = PyTuple_GET_ITEM(args, 1);
  PyObject* _size = PyTuple_GET_ITEM(args, 2);
  // 检查句柄是否为字节字符串，大小是否为整数
  if (!PyBytes_Check(_manager_handle) || !PyBytes_Check(_object_handle) ||
      !THPUtils_checkLong(_size)) {
    // 参数无效时抛出异常
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in file system mode",
        1,
        "a handle (string/bytes) and storage size (int)");
    return nullptr;
  }
  // 获取管理器句柄和对象句柄的 C 字符串表示
  const char* manager_handle = PyBytes_AS_STRING(_manager_handle);
  const char* object_handle = PyBytes_AS_STRING(_object_handle);
  // 解包并获取存储大小
  uint64_t size = THPUtils_unpackUInt64(_size);
  // 设置标志以指示分配器使用共享内存映射且不创建新映射
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
  // 使用给定的存储大小和数据指针创建新的存储实现，并返回 Python 对象
  return THPStorage_NewWithStorage(
      THPStorageClass,
      c10::make_intrusive<at::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size,
          THManagedMapAllocator::makeDataPtr(
              manager_handle, object_handle, flags, size),
          /*allocator=*/nullptr,
          /*resizable=*/false),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
  END_HANDLE_TH_ERRORS
}

// 创建新的文件描述符存储的 Python 对象
static PyObject* THPStorage_pyNewFdStorage(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  long long size = 0;
  // 解析参数并获取存储大小
  if (!PyArg_ParseTuple(args, "L", &size)) {
    return nullptr;
  }
  // 确保存储大小为非负数
  if (size < 0) {
    return nullptr;
  }
  // 使用给定的存储大小创建新的共享内存文件描述符存储，并返回 Python 对象
  return THPStorage_NewWithStorage(
      THPStorageClass,
      at::new_shm_fd_storage(size),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
  END_HANDLE_TH_ERRORS
}

// 在当前存储上共享文件描述符
static PyObject* THPStorage_shareFd(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 确保存储对象不为空
  THPStorage_assertNotNull(self);
  // 解包存储对象
  const auto& storage = THPStorage_Unpack(self);
  // 检查存储设备类型是否为 CPU
  TORCH_CHECK(
      storage.device_type() == at::kCPU, "_share_fd_: only available on CPU");
  // 获取当前数据指针关联的映射分配器
  at::MapAllocator* ctx = at::MapAllocator::fromDataPtr(storage.data_ptr());
  // 如果存储已在共享内存中，则直接返回句柄
  if (ctx) {
    // done
  } else {
    // 创建新的共享内存文件描述符存储，并复制数据到新存储中
    at::Storage new_storage(at::new_shm_fd_storage(storage.nbytes()));
    {
      // 释放 GIL，因为在共享内存中进行数据复制可能会很慢
      pybind11::gil_scoped_release no_gil;
      // 将旧存储中的数据复制到新存储中
      at::storage_copy(new_storage, storage);
    }
    // 替换旧数据指针和分配器为新的
    storage.set_data_ptr(std::move(new_storage.mutable_data_ptr()));
    storage.unsafeGetStorageImpl()->set_allocator(new_storage.allocator());
    // 获取新数据指针关联的映射分配器
    ctx = at::MapAllocator::fromDataPtr(storage.data_ptr());
    AT_ASSERT(ctx);
  }
  // 将文件描述符打包为 Python 对象并返回
  THPObjectPtr storage_handle(THPUtils_packInt32(ctx->fd()));
  if (!storage_handle)
    return nullptr;
  // 将存储大小打包为 Python 对象并返回
  THPObjectPtr size(THPUtils_packUInt64(storage.nbytes()));
  if (!size)
    return nullptr;
    # 返回空指针，表示函数执行失败或无返回值
        return nullptr;
    
    # 创建一个包含两个元素的 Python 元组对象
      THPObjectPtr tuple(PyTuple_New(2));
      if (!tuple)
        return nullptr;
    
    # 将 storage_handle.release() 的所有权转移给元组的第一个元素
      PyTuple_SET_ITEM(tuple.get(), 0, storage_handle.release());
    
    # 将 size.release() 的所有权转移给元组的第二个元素
      PyTuple_SET_ITEM(tuple.get(), 1, size.release());
    
    # 返回元组对象，释放所有权
      return tuple.release();
      END_HANDLE_TH_ERRORS
  // 开始错误处理宏，捕获并处理异常
  HANDLE_TH_ERRORS
  // 检查参数元组是否包含两个项
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  // 获取第一个参数作为文件描述符对象
  PyObject* _tmp_fd = PyTuple_GET_ITEM(args, 0);
  // 获取第二个参数作为存储大小对象
  PyObject* _size = PyTuple_GET_ITEM(args, 1);
  // 检查参数类型是否为整数，否则抛出错误
  if (!THPUtils_checkLong(_tmp_fd) || !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in file descriptor mode",
        1,
        "a file descriptor (int) and storage size (int)");
    return nullptr;
  }
  // 将临时文件描述符和存储大小转换为相应的整数类型
  int tmp_fd = (int)THPUtils_unpackLong(_tmp_fd);
  int64_t size = THPUtils_unpackLong(_size);
  // 复制临时文件描述符以获取共享内存文件描述符
  int fd = dup(tmp_fd);
  // 如果复制失败，则设置错误并返回空指针
  if (fd == -1) {
    THPUtils_setError("could not duplicate a shared memory file descriptor");
    return nullptr;
  }

  // 设置共享内存的标志
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE |
      at::ALLOCATOR_MAPPED_KEEPFD | at::ALLOCATOR_MAPPED_FROMFD;
  // 使用共享内存文件描述符创建新的存储对象，并返回 Python 对象
  return THPStorage_NewWithStorage(
      THPStorageClass,
      c10::make_intrusive<at::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size,
          at::MapAllocator::makeDataPtr(
              at::WITH_FD, "", fd, flags, size, nullptr),
          /*allocator=*/nullptr,
          /*resizable=*/false),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
  // 结束错误处理宏
  END_HANDLE_TH_ERRORS
}
  
static PyObject* THPStorage_shareCuda(PyObject* self, PyObject* noargs) {
  // 开始错误处理宏，捕获并处理异常
  HANDLE_TH_ERRORS
  // 确保 self 不为空
  THPStorage_assertNotNull(self);
  // 如果使用 CUDA，从 self 中解包存储对象
#ifdef USE_CUDA
  const auto& storage = THPStorage_Unpack(self);
  // 检查存储对象是否位于 CUDA 设备上
  TORCH_CHECK(
      storage.device_type() == at::kCUDA,
      "_share_cuda_: only available on CUDA");
  // 获取存储实现指针
  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();

  // 如果存储对象已从 CUDA 接收，抛出错误
  if (storage_impl->received_cuda()) {
    AT_ERROR(
        "Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending.");
  }

  // 设置设备为存储对象所在设备
  at::DeviceGuard device_guard(storage.device());
  // 创建一个 Python 元组对象，包含共享 CUDA 所需的信息
  THPObjectPtr tuple(PyTuple_New(8));
  // 封装设备索引到 Python 整数对象
  THPObjectPtr device(THPUtils_packInt32(storage.device().index()));
  // 初始化共享句柄为 Python None 对象
  THPObjectPtr _handle(Py_None);
  Py_INCREF(Py_None);
  // 封装存储字节大小到 Python 整数对象
  THPObjectPtr size_bytes(THPUtils_packUInt64(storage.nbytes()));
  // 初始化偏移量为 Python 整数对象 0
  THPObjectPtr _offset_bytes(THPUtils_packInt32(0));
  // 初始化引用计数为 Python None 对象
  THPObjectPtr _ref_counter(Py_None);
  Py_INCREF(Py_None);
  // 初始化引用计数偏移量为 Python 整数对象 0
  THPObjectPtr _ref_counter_offset(THPUtils_packInt32(0));
  // 初始化事件句柄为 Python None 对象
  THPObjectPtr _event_handle(Py_None);
  Py_INCREF(Py_None);
  // 初始化事件同步标志为 Python None 对象
  THPObjectPtr _event_sync_required(Py_None);
  Py_INCREF(Py_None);
  // 如果存储对象有数据
  if (storage.data()) {
    // 获取存储基础分配的基地址和大小
    size_t base_size;
    void* base_ptr = c10::cuda::CUDACachingAllocator::getBaseAllocation(
        storage.mutable_data(), &base_size);
    // 计算数据在基地址中的偏移量
    ptrdiff_t offset_bytes = (char*)storage.data() - (char*)base_ptr;

    // 获取 CUDA IPC 内存句柄
    cudaIpcMemHandle_t handle;
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, base_ptr));

    // 将 CUDA IPC 句柄转换为 Python 字节对象
    _handle = PyBytes_FromStringAndSize((char*)&handle, CUDA_IPC_HANDLE_SIZE);
    // 将 offset_bytes 转换为 Python 的 PyLong 对象
    _offset_bytes = PyLong_FromSsize_t((Py_ssize_t)offset_bytes);

    // 将存储数据放置在新的引用计数上下文中
    // 参见注释 [CUDA IPC Refcounting implementation explained]
    at::DataPtr sent_data_ptr = torch::GetNewRefCountedSentData(
        storage.mutable_data(), storage.device());
    // 设置存储的数据指针为新创建的 sent_data_ptr
    auto old_data_ptr = storage.set_data_ptr(std::move(sent_data_ptr));
    // 获取 sent_data_ptr 的实际类型为 torch::CudaIPCSentData*，并设置原始指针
    auto sent_data =
        static_cast<torch::CudaIPCSentData*>(storage.data_ptr().get_context());
    sent_data->set_original_ptr(std::move(old_data_ptr));
    // 用 sent_data 的句柄创建一个 PyBytes 对象
    _ref_counter = PyBytes_FromString((sent_data->handle()).c_str());
    // 将 sent_data 的偏移量转换为 PyLong 对象
    _ref_counter_offset = THPUtils_packUInt64(sent_data->offset());

    // 定义 CUDA IPC 事件句柄
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    cudaIpcEventHandle_t ipc_event_handle;

    // 如果需要同步事件，获取事件的 IPC 句柄
    if (sent_data->event_sync_required_) {
      C10_CUDA_CHECK(
          cudaIpcGetEventHandle(&ipc_event_handle, sent_data->event_));
    }

    // 使用 ipc_event_handle 创建一个 PyBytes 对象
    _event_handle = PyBytes_FromStringAndSize(
        (char*)&ipc_event_handle, CUDA_IPC_HANDLE_SIZE);
    // 创建一个 PyBool 对象，表示是否需要同步事件
    _event_sync_required = PyBool_FromLong(sent_data->event_sync_required_);
  }

  // 如果有任何参数为空，则返回空指针
  if (!tuple || !device || !_handle || !size_bytes || !_offset_bytes ||
      !_event_handle) {
    return nullptr;
  }

  // 将各个参数设置到 tuple 中，并释放其占有的资源
  PyTuple_SET_ITEM(tuple.get(), 0, device.release()); // 设备对象
  PyTuple_SET_ITEM(tuple.get(), 1, _handle.release()); // 句柄对象
  PyTuple_SET_ITEM(tuple.get(), 2, size_bytes.release()); // 存储的大小
  PyTuple_SET_ITEM(tuple.get(), 3, _offset_bytes.release()); // 存储的偏移量
  PyTuple_SET_ITEM(tuple.get(), 4, _ref_counter.release()); // 引用计数器
  PyTuple_SET_ITEM(tuple.get(), 5, _ref_counter_offset.release()); // 引用计数器偏移量
  PyTuple_SET_ITEM(tuple.get(), 6, _event_handle.release()); // CUDA IPC 事件句柄
  PyTuple_SET_ITEM(tuple.get(), 7, _event_sync_required.release()); // 是否需要同步事件
  return tuple.release(); // 返回 tuple 对象的指针
#else
  // 如果没有定义 USE_CUDA，执行以下代码块
  TORCH_CHECK(false, "CUDA is not available");
#endif
  // 结束处理 Torch 错误的宏
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_releaseIPCCounter(
    PyObject* _unused,
    PyObject* args) {
  // 处理 Torch 错误的宏
  HANDLE_TH_ERRORS
#ifdef USE_CUDA
  // 如果定义了 USE_CUDA，执行以下代码块
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  // 获取参数元组中的引用计数器和偏移量对象
  PyObject* _ref_counter = PyTuple_GET_ITEM(args, 0);
  PyObject* _ref_counter_offset = PyTuple_GET_ITEM(args, 1);
  // 检查参数类型是否正确
  if (!(PyBytes_Check(_ref_counter) &&
        THPUtils_checkLong(_ref_counter_offset))) {
    // 参数类型不正确时抛出异常并返回空指针
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_release_ipc_counter in CUDA mode",
        1,
        "(bytes _ref_counter, int _ref_counter_offset)");
    return nullptr;
  }
  // 将引用计数器转换为字符串
  std::string ref_counter_handle = PyBytes_AS_STRING(_ref_counter);
  // 将偏移量转换为 ptrdiff_t 类型
  ptrdiff_t ref_counter_offset =
      (ptrdiff_t)THPUtils_unpackLong(_ref_counter_offset);
  // 尝试创建共享内存映射指定大小的数据区域，以减少引用计数
  // 设置标志为 ALLOCATOR_MAPPED_SHAREDMEM 和 ALLOCATOR_MAPPED_NOCREATE
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
  try {
    auto sptr = at::RefcountedMapAllocator::makeDataPtr(
        ref_counter_handle.c_str(),
        flags,
        sizeof(int64_t) * torch::CUDA_IPC_REF_COUNTER_FILE_SIZE,
        nullptr);
    // 减少指定偏移量的引用计数
    *(static_cast<int64_t*>(sptr.get()) + ref_counter_offset) -= 1;
  } catch (c10::Error& err) {
    // 在生产者进程中已经发出警告
  }
  // 返回 None
  Py_RETURN_NONE;
#else
  // 如果没有定义 USE_CUDA，执行以下代码块
  TORCH_CHECK(false, "CUDA is not available");
#endif
  // 结束处理 Torch 错误的宏
  END_HANDLE_TH_ERRORS
}

#ifdef USE_CUDA
static std::string THPStorage_bytesAsHandleString(PyObject* handle) {
  // 处理 Torch 错误的宏
  HANDLE_TH_ERRORS
  // 将 Python 字符串对象转换为 C++ 字符串并返回
  char* buffer = nullptr;
  Py_ssize_t handle_size = 0;
  // 获取 Python 字符串对象的指针和大小
  if (PyBytes_AsStringAndSize(handle, &buffer, &handle_size) == -1) {
    // 检查获取的句柄大小是否正确
    TORCH_CHECK(handle_size == CUDA_IPC_HANDLE_SIZE, "incorrect handle");
  }
  // 检查获取的句柄大小是否正确
  TORCH_CHECK(handle_size == CUDA_IPC_HANDLE_SIZE, "incorrect handle size");
  // 返回 C++ 标准字符串对象
  return std::string(buffer, handle_size);
  // 结束处理 Torch 错误的宏，并返回空字符串
  END_HANDLE_TH_ERRORS_RET("")
}
#endif

static PyObject* THPStorage_newSharedCuda(PyObject* _unused, PyObject* args) {
  // 处理 Torch 错误的宏
  HANDLE_TH_ERRORS
#ifdef USE_CUDA
  // 如果定义了 USE_CUDA，执行以下代码块
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 8, "tuple of 8 items expected");
  // 获取参数元组中的各个参数对象
  PyObject* _device = PyTuple_GET_ITEM(args, 0);
  PyObject* _handle = PyTuple_GET_ITEM(args, 1);
  PyObject* _size_bytes = PyTuple_GET_ITEM(args, 2);
  PyObject* _offset_bytes = PyTuple_GET_ITEM(args, 3);
  PyObject* _ref_counter = PyTuple_GET_ITEM(args, 4);
  PyObject* _ref_counter_offset = PyTuple_GET_ITEM(args, 5);
  PyObject* _event_handle = PyTuple_GET_ITEM(args, 6);
  PyObject* _event_sync_required = PyTuple_GET_ITEM(args, 7);
  // 检查参数类型是否正确
  if (!(THPUtils_checkLong(_device) && THPUtils_checkLong(_size_bytes) &&
        PyBytes_Check(_handle) && PyBytes_Check(_ref_counter) &&
        PyBytes_Check(_event_handle) && THPUtils_checkLong(_offset_bytes) &&
        THPUtils_checkLong(_ref_counter_offset) &&
        PyBool_Check(_event_sync_required))) {
   `
    # 调用 THPUtils_invalidArguments 函数，处理无效参数情况，并提供错误信息，返回 nullptr
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in CUDA mode",
        1,
        "(int device, bytes handle, int storage_size_bytes, int storage_offset_bytes, bytes _ref_counter, int _ref_counter_offset, bytes event_handle, bool event_sync_required)");
    return nullptr;
  }

  # 将 _size_bytes 转换为 size_t 类型，计算存储的字节数
  size_t storage_size =
      (size_t)THPUtils_unpackLong(_size_bytes) / sizeof(uint8_t);
  # 将 _offset_bytes 转换为 ptrdiff_t 类型，获取存储的偏移量
  ptrdiff_t storage_offset_bytes =
      (ptrdiff_t)THPUtils_unpackLong(_offset_bytes);

  # 将 _device 字节数据转换为 c10::DeviceIndex 类型，并检查转换是否成功
  const auto device = c10::checked_convert<c10::DeviceIndex>(
      THPUtils_unpackLong(_device), "c10::DeviceIndex");
  # 创建 CUDAGuard 对象，设置当前线程使用指定设备
  at::cuda::CUDAGuard device_guard(device);

  # 检查 _event_sync_required 是否为真，若真则执行以下代码块
  if (PyObject_IsTrue(_event_sync_required)) {
    # 将 _event_handle 转换为 IPC 事件句柄的字符串表示
    std::string s_ipc_event_handle =
        THPStorage_bytesAsHandleString(_event_handle);
    # 若事件句柄为空，返回 nullptr
    if (s_ipc_event_handle.empty()) {
      return nullptr;
    }
    # 将字符串表示的事件句柄转换为 cudaIpcEventHandle_t 指针
    auto ipc_event_handle = reinterpret_cast<const cudaIpcEventHandle_t*>(
        s_ipc_event_handle.c_str());
    # 声明一个 CUDA 事件变量 event
    cudaEvent_t event;
    # 打开 IPC 事件句柄
    cudaIpcOpenEventHandle(&event, *ipc_event_handle);
    # 等待当前 CUDA 流中的事件完成
    C10_CUDA_CHECK(
        cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(device), event, 0));
  }

  # 将 _handle 字节数据转换为字符串表示，若为空，返回 nullptr
  std::string s_handle = THPStorage_bytesAsHandleString(_handle);
  if (s_handle.empty()) {
    return nullptr;
  }
  # 使用 CUDACachingAllocator 获取共享内存指针
  std::shared_ptr<void> basePtr =
      c10::cuda::CUDACachingAllocator::getIpcDevPtr(s_handle);

  # 将 basePtr 偏移 storage_offset_bytes，计算实际的设备指针
  void* devPtr = basePtr.get();
  devPtr = (char*)devPtr + storage_offset_bytes;

  # 将 _ref_counter 转换为字符串，获取引用计数器的句柄
  std::string ref_counter_handle = PyBytes_AS_STRING(_ref_counter);
  # 将 _ref_counter_offset 转换为 ptrdiff_t 类型，获取引用计数器的偏移量
  ptrdiff_t ref_counter_offset =
      (ptrdiff_t)THPUtils_unpackLong(_ref_counter_offset);

  # 定义一个结构体 IpcDeleterContext，用于保存 IPC 删除器的上下文信息
  struct
    // 创建一个结构体对象 `received_data`，用于存储 CUDA IPC 接收到的数据
    torch::CudaIPCReceivedData received_data;
  };

  // 创建一个独占指针 `ctx`，用于管理 IPC 数据的释放上下文
  auto ctx = std::make_unique<IpcDeleterContext>();
  // 将引用计数句柄移动给上下文对象
  ctx->ref_counter_handle = std::move(ref_counter_handle);
  // 设置引用计数偏移量
  ctx->ref_counter_offset = ref_counter_offset;
  // 设置设备信息
  ctx->device = device;
  // 移动基础指针到接收数据的共享指针
  ctx->received_data.shared_ptr_ = std::move(basePtr);

  // 获取当前 CUDA 设备
  auto cur_device = at::cuda::current_device();
  // 创建数据指针 `data_ptr`，并指定释放函数
  c10::DataPtr data_ptr(
      devPtr,
      ctx.release(), // 释放函数上下文
      +[](void* ctx_) {
        // 将上下文对象转换为正确的类型
        std::unique_ptr<IpcDeleterContext> ctx(
            static_cast<IpcDeleterContext*>(ctx_));
        // 重置接收数据的共享指针
        ctx->received_data.shared_ptr_.reset();

        // 同步默认流，确保所有与存储相关的操作完成
        // （否则另一个进程可能会重用内存并损坏数据）

        // 理想情况下，所有共享内存的引用计数可以通过从生产者到消费者发送未触发的 CUDA 事件来替换，
        // 并使用此事件作为内存释放的标准。然而，CUDA 目前（截至 10.1 版本）不支持创建未触发的事件，
        // 并且有数千个共享事件的性能影响是未知的。

        // TODO: 可以考虑添加流回调而不是 cudaStreamSynchronize，以在其中释放计数（需要检查性能影响）
        at::cuda::stream_synchronize(
            c10::cuda::getCurrentCUDAStream(ctx->device));

        // 我们不希望破坏现有代码，因此资源删除是尽力而为的。如果生产者进程在消费者释放数据之前终止，则预期会出现异常。
        int flags =
            at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
        try {
          // 创建共享内存数据指针 `sptr`
          auto sptr = at::RefcountedMapAllocator::makeDataPtr(
              ctx->ref_counter_handle.c_str(),
              flags,
              sizeof(int64_t) * torch::CUDA_IPC_REF_COUNTER_FILE_SIZE,
              nullptr);
          // 减少引用计数
          *(static_cast<int64_t*>(sptr.get()) + ctx->ref_counter_offset) -= 1;
        } catch (c10::Error& err) {
          // 在生产者进程内部已经发出警告
        }
      },
      at::Device(at::DeviceType::CUDA, cur_device));

  // 使用存储大小、数据指针和其他参数创建新的存储实现 `base`
  auto base = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      storage_size,
      std::move(data_ptr),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  // 设置存储不可调整大小
  base->set_resizable(false);
  // 标记存储已从 CUDA 接收到
  base->set_received_cuda(true);

  // 返回用存储实现 `base` 创建的新 THPStorage 对象
  return THPStorage_NewWithStorage(
      THPStorageClass,
      std::move(base),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
#else
  TORCH_CHECK(false, "CUDA is not available");
#endif
  END_HANDLE_TH_ERRORS
}

// 如果条件不满足，则触发 TORCH_CHECK 条件断言，输出错误信息表示 CUDA 不可用
static PyObject* THPStorage_weakRef(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 从 Python 对象中解包出 c10::StorageImpl 对象的强引用
  c10::StorageImpl* storage = THPStorage_Unpack(self).unsafeGetStorageImpl();
  // 返回一个持有 c10::StorageImpl 弱引用的 Python 对象
  return PyLong_FromVoidPtr(c10::raw::intrusive_ptr::make_weak(storage));
  END_HANDLE_TH_ERRORS
}

// 创建一个持有 c10::StorageImpl 弱引用的 Python 对象
static PyObject* THPStorage_newWithWeakPtr(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为长整型，否则触发 TORCH_CHECK 条件断言
  TORCH_CHECK(
      THPUtils_checkLong(arg), "_new_with_weak_ptr(): arg must be an 'int'");
  // 将 Python 中的长整型参数转换为 c10::StorageImpl* 类型
  c10::StorageImpl* weak_storage = (c10::StorageImpl*)PyLong_AsVoidPtr(arg);
  // 如果能够通过弱引用锁定到存储对象，则返回一个封装了 c10::StorageImpl 的 Python 对象
  if (auto* storage = c10::raw::weak_intrusive_ptr::lock(weak_storage)) {
    return THPStorage_Wrap(
        c10::intrusive_ptr<c10::StorageImpl>::reclaim(storage));
  }
  // 否则返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 释放 c10::StorageImpl 的弱引用
static PyObject* THPStorage_freeWeakRef(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 如果参数为 None，则直接返回 None
  if (arg == Py_None) {
    Py_RETURN_NONE;
  }
  // 检查参数是否为长整型，否则触发 TORCH_CHECK 条件断言
  TORCH_CHECK(
      THPUtils_checkLong(arg), "_free_weak_ref(): arg must be an 'int'");
  // 将 Python 中的长整型参数转换为 c10::StorageImpl* 类型
  c10::StorageImpl* weak_storage = (c10::StorageImpl*)PyLong_AsVoidPtr(arg);
  // 释放 c10::StorageImpl 的弱引用
  c10::raw::weak_intrusive_ptr::decref(weak_storage);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 检查 c10::StorageImpl 弱引用是否过期
static PyObject* THPStorage_expired(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 检查参数是否为长整型，否则触发 TORCH_CHECK 条件断言
  TORCH_CHECK(THPUtils_checkLong(arg), "_expired(): arg must be an 'int'");
  // 将 Python 中的长整型参数转换为 c10::StorageImpl* 类型
  c10::StorageImpl* weak_storage = (c10::StorageImpl*)PyLong_AsVoidPtr(arg);
  // 返回一个布尔值，表示 c10::StorageImpl 弱引用是否已经过期（use_count 为 0）
  return PyBool_FromLong(
      c10::raw::weak_intrusive_ptr::use_count(weak_storage) == 0);
  END_HANDLE_TH_ERRORS
}

// 获取共享文件描述符
static PyObject* THPStorage_sharedFd(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 确保 self 引用的 c10::StorageImpl 对象不为空
  THPStorage_assertNotNull(self);
  at::MapAllocator* ctx = nullptr;
  // 从 THPStorage_Unpack 中获取 c10::Storage 对象的引用
  const auto& storage = THPStorage_Unpack(self);
  // 如果存储对象在 CPU 设备上，则获取其对应的 MapAllocator 对象
  if (storage.device_type() == at::kCPU) {
    ctx = at::MapAllocator::fromDataPtr(storage.data_ptr());
  }

  // 确保获取到了 MapAllocator 对象，否则触发 TORCH_CHECK 条件断言
  TORCH_CHECK(ctx, "couldn't retrieve a shared file descriptor");
  // 返回 MapAllocator 对象的文件描述符
  return THPUtils_packInt32(ctx->fd());
  END_HANDLE_TH_ERRORS
}

// 检查 c10::StorageImpl 是否是共享的
static PyObject* THPStorage_isShared(PyObject* self, PyObject* noargs) {
  const auto& storage = THPStorage_Unpack(self);
  // 如果存储对象在 CUDA 设备上，则返回 True
  if (storage.device_type() == at::kCUDA) {
    Py_RETURN_TRUE;
  }
  // 如果存储对象关联了 MapAllocator 或 THManagedMapAllocator，则也返回 True；否则返回 False
  if (at::MapAllocator::fromDataPtr(storage.data_ptr()) ||
      THManagedMapAllocator::fromDataPtr(storage.data_ptr())) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// 定义 THPStorage_sharingMethods 数组，包含了 THPStorage 模块中的共享方法
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef THPStorage_sharingMethods[] = {
    {"_new_with_weak_ptr",
     THPStorage_newWithWeakPtr,
     METH_O | METH_CLASS,
     nullptr},
    {"_share_cuda_", THPStorage_shareCuda, METH_NOARGS, nullptr},
    ...
};
    {"_new_shared_cuda",
     THPStorage_newSharedCuda,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_release_ipc_counter_cuda",
     THPStorage_releaseIPCCounter,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_share_fd_cpu_", THPStorage_shareFd, METH_NOARGS, nullptr},
    {"_new_shared_fd_cpu",
     THPStorage_newSharedFd,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_new_using_fd_cpu",
     THPStorage_pyNewFdStorage,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_share_filename_cpu_", THPStorage_shareFilename, METH_NOARGS, nullptr},
    {"_new_shared_filename_cpu",
     THPStorage_newSharedFilename,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_new_using_filename_cpu",
     THPStorage_pyNewFilenameStorage,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_weak_ref", THPStorage_weakRef, METH_NOARGS, nullptr},
    {"_free_weak_ref", THPStorage_freeWeakRef, METH_O | METH_STATIC, nullptr},
    {"_expired", THPStorage_expired, METH_O | METH_STATIC, nullptr},
    {"_shared_decref", THPStorage_sharedDecref, METH_NOARGS, nullptr},
    {"_shared_incref", THPStorage_sharedIncref, METH_NOARGS, nullptr},
    {"_get_shared_fd", THPStorage_sharedFd, METH_NOARGS, nullptr},
    {"is_shared", THPStorage_isShared, METH_NOARGS, nullptr},
    {nullptr}};



# 注册存储相关的函数到 Python 模块中
{"_new_shared_cuda",           // 函数名字符串
 THPStorage_newSharedCuda,      // 对应的 C 函数指针
 METH_VARARGS | METH_STATIC,    // 函数接受参数并且是静态方法
 nullptr},                      // 表示函数定义结束

{"_release_ipc_counter_cuda",   // 函数名字符串
 THPStorage_releaseIPCCounter,  // 对应的 C 函数指针
 METH_VARARGS | METH_STATIC,    // 函数接受参数并且是静态方法
 nullptr},                      // 表示函数定义结束

{"_share_fd_cpu_",             // 函数名字符串
 THPStorage_shareFd,            // 对应的 C 函数指针
 METH_NOARGS,                   // 函数不接受参数
 nullptr},                      // 表示函数定义结束

{"_new_shared_fd_cpu",         // 函数名字符串
 THPStorage_newSharedFd,        // 对应的 C 函数指针
 METH_VARARGS | METH_STATIC,    // 函数接受参数并且是静态方法
 nullptr},                      // 表示函数定义结束

{"_new_using_fd_cpu",          // 函数名字符串
 THPStorage_pyNewFdStorage,     // 对应的 C 函数指针
 METH_VARARGS | METH_STATIC,    // 函数接受参数并且是静态方法
 nullptr},                      // 表示函数定义结束

{"_share_filename_cpu_",       // 函数名字符串
 THPStorage_shareFilename,      // 对应的 C 函数指针
 METH_NOARGS,                   // 函数不接受参数
 nullptr},                      // 表示函数定义结束

{"_new_shared_filename_cpu",   // 函数名字符串
 THPStorage_newSharedFilename,  // 对应的 C 函数指针
 METH_VARARGS | METH_STATIC,    // 函数接受参数并且是静态方法
 nullptr},                      // 表示函数定义结束

{"_new_using_filename_cpu",    // 函数名字符串
 THPStorage_pyNewFilenameStorage,// 对应的 C 函数指针
 METH_VARARGS | METH_STATIC,    // 函数接受参数并且是静态方法
 nullptr},                      // 表示函数定义结束

{"_weak_ref",                  // 函数名字符串
 THPStorage_weakRef,            // 对应的 C 函数指针
 METH_NOARGS,                   // 函数不接受参数
 nullptr},                      // 表示函数定义结束

{"_free_weak_ref",             // 函数名字符串
 THPStorage_freeWeakRef,        // 对应的 C 函数指针
 METH_O | METH_STATIC,          // 函数接受一个参数并且是静态方法
 nullptr},                      // 表示函数定义结束

{"_expired",                   // 函数名字符串
 THPStorage_expired,            // 对应的 C 函数指针
 METH_O | METH_STATIC,          // 函数接受一个参数并且是静态方法
 nullptr},                      // 表示函数定义结束

{"_shared_decref",             // 函数名字符串
 THPStorage_sharedDecref,       // 对应的 C 函数指针
 METH_NOARGS,                   // 函数不接受参数
 nullptr},                      // 表示函数定义结束

{"_shared_incref",             // 函数名字符串
 THPStorage_sharedIncref,       // 对应的 C 函数指针
 METH_NOARGS,                   // 函数不接受参数
 nullptr},                      // 表示函数定义结束

{"_get_shared_fd",             // 函数名字符串
 THPStorage_sharedFd,           // 对应的 C 函数指针
 METH_NOARGS,                   // 函数不接受参数
 nullptr},                      // 表示函数定义结束

{"is_shared",                   // 函数名字符串
 THPStorage_isShared,           // 对应的 C 函数指针
 METH_NOARGS,                   // 函数不接受参数
 nullptr},                      // 表示函数定义结束

{nullptr}};                     // 表示函数列表结束的标志，空指针
// 返回 THPStorage_sharingMethods 指针，它是 PyMethodDef* 类型的全局变量，用于定义在 Python 中共享行为的方法列表。
PyMethodDef* THPStorage_getSharingMethods() {
    return THPStorage_sharingMethods;
}
```