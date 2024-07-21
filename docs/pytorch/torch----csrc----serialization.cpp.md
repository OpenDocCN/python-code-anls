# `.\pytorch\torch\csrc\serialization.cpp`

```
// 包含 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>
// 包含系统错误处理头文件
#include <system_error>
// 包含向量容器头文件
#include <vector>

// 包含 ATen 的从 blob 中创建操作头文件
#include <ATen/ops/from_blob.h>
// 包含 C10 的 CPU 分配器头文件
#include <c10/core/CPUAllocator.h>
// 包含 Torch 的 THP 头文件
#include <torch/csrc/THP.h>
// 包含 Torch 的序列化头文件
#include <torch/csrc/serialization.h>

// 部分读取模板函数声明，针对不同类型的 io 参数
template <class io>
Py_ssize_t doPartialRead(io fildes, void* buf, size_t nbytes);

// 部分写入模板函数声明，针对不同类型的 io 参数
template <class io>
Py_ssize_t doPartialWrite(io fildes, void* buf, size_t nbytes);

// 部分 Python 读取缓冲函数声明，针对 PyObject* 参数
static Py_ssize_t doPartialPythonReadBuffered(
    PyObject* fildes,
    void* buf,
    size_t nbytes);

// 部分 Python 读取到函数声明，针对 PyObject* 参数
static Py_ssize_t doPartialPythonReadInto(
    PyObject* fildes,
    void* buf,
    size_t nbytes);

// 部分 Python 写入函数声明，针对 PyObject* 参数
static Py_ssize_t doPartialPythonWrite(
    PyObject* fildes,
    void* buf,
    size_t nbytes);

// 特化模板函数，针对 int 类型的 fildes 参数进行部分读取操作
template <>
Py_ssize_t doPartialRead<int>(int fildes, void* buf, size_t nbytes) {
  return read(fildes, buf, nbytes);
}

// 特化模板函数，针对 PyObject* 类型的 fildes 参数进行部分读取操作
template <>
Py_ssize_t doPartialRead<PyObject*>(
    PyObject* fildes,
    void* buf,
    size_t nbytes) {
  // 尝试使用 fildes.readinto() 而不是 fildes.read()，因为它在内存上更有效率
  // TODO: 在我们的读取循环中停止对 PyObject_HasAttrString() 的反复调用
  auto has_readinto = PyObject_HasAttrString(fildes, "readinto") == 1;
  if (has_readinto) {
    return doPartialPythonReadInto(fildes, buf, nbytes);
  }
  return doPartialPythonReadBuffered(fildes, buf, nbytes);
}

// 特化模板函数，针对 int 类型的 fildes 参数进行部分写入操作
template <>
Py_ssize_t doPartialWrite<int>(int fildes, void* buf, size_t nbytes) {
  return write(fildes, buf, nbytes);
}

// 特化模板函数，针对 PyObject* 类型的 fildes 参数进行部分写入操作
template <>
Py_ssize_t doPartialWrite<PyObject*>(
    PyObject* fildes,
    void* buf,
    size_t nbytes) {
  return doPartialPythonWrite(fildes, buf, nbytes);
}

// 内联函数，判断是否为不支持的操作
static inline bool isUnsupportedOperation() {
  // 导入 io 模块
  THPObjectPtr io(PyImport_ImportModule("io"));
  if (!io)
    throw python_error();
  // 获取 io 模块中的 UnsupportedOperation 异常类
  THPObjectPtr exception(PyObject_GetAttrString(io, "UnsupportedOperation"));
  if (!exception)
    throw python_error();
  // 检查当前异常是否匹配 UnsupportedOperation 类
  return PyErr_ExceptionMatches(exception.get());
}

// 调用 Python 的 fildes.read(nbytes) 并将数据复制到 buf 中
static inline Py_ssize_t doPartialPythonReadBuffered(
    PyObject* fildes,
    void* buf,
    size_t raw_nbytes) {
  // 如果请求的数据量较大，f.read() 将尝试在内部分配该大小的缓冲区。
  // 这是适得其反的，因为它不是我们最终要写入数据的缓冲区。限制分配的额外内存量。
  // TODO: 或许 260 KB 太小了…
  const size_t nbytes = std::min<size_t>(raw_nbytes, 262144u); // 2^18 (~260 KB)

  // 调用 fildes.read(nbytes) 方法
  THPObjectPtr r(PyObject_CallMethod(fildes, "read", "i", nbytes));
  if (!r)
    throw python_error();

  // 获取 Python 字节对象的大小和数据指针
  auto size = PyBytes_GET_SIZE(r.get());
  const void* py_buf = PyBytes_AsString(r.get());

  // 如果读取到 EOF
  if (size == 0) {
    return 0;
  }

  // 将数据拷贝到目标缓冲区中
  memcpy(buf, py_buf, size);

  return size;
}

// 要么调用 fildes.readinto(buf)，要么调用 fildes.write(buf)
static inline Py_ssize_t doPartialPythonIO(
    PyObject* fildes,
    void* buf,
    size_t nbytes,
    // 根据传入的 is_read 参数确定内存视图的读写权限
    auto rw_flag = is_read ? PyBUF_WRITE : PyBUF_READ;
    // 使用 PyMemoryView_FromMemory 从给定的 buf 创建内存视图对象
    THPObjectPtr memview(PyMemoryView_FromMemory(
        reinterpret_cast<char*>(buf), static_cast<Py_ssize_t>(nbytes), rw_flag));
    // 如果创建内存视图失败，抛出 python_error 异常
    if (!memview)
        throw python_error();

    // 默认使用 "write" 方法
    std::string method = "write";
    // 如果 is_read 为真，切换到 "readinto" 方法
    if (is_read) {
        method = "readinto";
    }
    // 调用 fildes 对象的指定方法（"write" 或 "readinto"）并传入内存视图对象
    THPObjectPtr r(
        PyObject_CallMethod(fildes, method.c_str(), "O", memview.get()));
    // 如果调用成功，将返回值转换为 Py_ssize_t 类型并返回
    if (r) {
        return PyLong_AsSsize_t(r.get());
    }

    // 如果 is_read 为真且调用结果为 UnsupportedOperation，通过缓冲读取部分数据
    // 否则抛出 python_error 异常
    if (is_read && isUnsupportedOperation()) {
        PyErr_Clear();
        return doPartialPythonReadBuffered(fildes, buf, nbytes);
    }
    throw python_error();
// 结束静态函数 doPartialPythonReadInto 的定义

// 调用 Python 的 fildes.readinto(buf) 方法
static Py_ssize_t doPartialPythonReadInto(
    PyObject* fildes,
    void* buf,
    size_t nbytes) {
  return doPartialPythonIO(fildes, buf, nbytes, /* is_read */ true);
}

// 调用 Python 的 fildes.write(buf) 方法
static Py_ssize_t doPartialPythonWrite(
    PyObject* fildes,
    void* buf,
    size_t nbytes) {
  return doPartialPythonIO(fildes, buf, nbytes, /* is_read */ false);
}

// 要求确切读取 nbytes；如果未达到要求则失败。
template <typename io>
void doRead(io fildes, void* raw_buf, size_t nbytes) {
  char* buf = static_cast<char*>(raw_buf);
  while (nbytes > 0) {
    errno = 0; // doPartialRead 可能不会设置 errno
    // 以 1GB 块大小读取，以避免在 Mac OS X Lion 上出现错误
    // 参见 https://github.com/pytorch/pytorch/issues/1031 获取更多细节
    Py_ssize_t r =
        doPartialRead(fildes, buf, std::min<size_t>(nbytes, 1073741824));
    if (r < 0) {
      int err = errno;
      TORCH_INTERNAL_ASSERT(
          err != 0, "read(): impossible! r < 0, but no errno was set");
      TORCH_INTERNAL_ASSERT(
          err != EAGAIN,
          "read(): non-blocking fd ",
          fildes,
          " read EAGAIN; cowardly refusing to spin-wait");
      if (err == EINTR) {
        continue;
      } else {
        AT_ERROR("read(): fd ", fildes, " failed with ", strerror(err));
      }
    } else if (r == 0) {
      break;
    }
    buf += r;
    // 这由 POSIX 保证，但我想要确保不会发生有符号整数下溢。
    AT_ASSERT(static_cast<size_t>(r) <= nbytes);
    nbytes -= r;
  }
  if (nbytes != 0) {
    AT_ERROR(
        "unexpected EOF, expected ",
        nbytes,
        " more bytes. The file might be corrupted.");
  }
}

// 要求确切写入 nbytes；如果未达到要求则失败。
template <typename io>
void doWrite(io fildes, void* raw_buf, size_t nbytes) {
  char* buf = static_cast<char*>(raw_buf);
  while (nbytes > 0) {
    errno = 0; // doPartialWrite 可能不会设置 errno
    // 以 1GB 块大小写入，以避免在 Mac OS X Lion 上出现错误
    // 参见 https://github.com/pytorch/pytorch/issues/1031 获取更多细节
    Py_ssize_t r =
        doPartialWrite(fildes, buf, std::min<size_t>(nbytes, 1073741824));
    if (r < 0) {
      int err = errno;
      TORCH_INTERNAL_ASSERT(
          err != 0, "write(): impossible! r < 0, but no errno was set");
      TORCH_INTERNAL_ASSERT(
          err != EAGAIN,
          "write(): non-blocking fd ",
          fildes,
          " read EAGAIN; cowardly refusing to spin-wait");
      if (err == EINTR) {
        continue;
      } else {
        AT_ERROR("write(): fd ", fildes, " failed with ", strerror(err));
      }
    }
    buf += r;
    AT_ASSERT(static_cast<size_t>(r) <= nbytes);
    nbytes -= r;
  }
}

// save_save 是必需的，因为旧的 eager 格式将存储保存为 [size + data]，但 v1.5 eager 格式移除了 size，因为 size 已经保存在 filesize 中。
template <class io>
void THPStorage_writeFileRaw(
    c10::StorageImpl* self,
    io fd,
    bool save_size,
    // 设置设备保护，确保在操作期间使用正确的设备
    c10::DeviceGuard guard(self->device());
    // 初始化数据指针
    uint8_t* data{};
    // 创建一个用于 CPU 的张量
    at::Tensor cpu_tensor;
    // 计算张量总字节数
    size_t size_bytes = self->nbytes();
    // 计算张量元素个数
    size_t numel = size_bytes / element_size;
    
    // 如果张量在 CPU 上
    if (self->device_type() == at::kCPU) {
        // 使用可变数据指针，因为最终会调用一个需要它的 Python API，尽管不会改变数据
        data = static_cast<uint8_t*>(self->mutable_data());
    } else {
        // 对于非 CPU 设备，使用 from_blob() 方法将数据转换到 CPU 上
        auto device_tensor = at::from_blob(
            self->mutable_data(),
            {static_cast<int64_t>(size_bytes)},
            {1},
            nullptr,
            at::device(self->device()).dtype(c10::kByte),
            {self->device()});
        cpu_tensor = device_tensor.to(at::kCPU);
        // 获取 CPU 张量的数据指针
        data = (uint8_t*)cpu_tensor.data_ptr();
    }
    
    // 如果需要保存大小信息
    if (save_size) {
        // 如果本地字节序为小端序，则直接写入元素数量
        if (torch::utils::THP_nativeByteOrder() ==
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN)
            doWrite(fd, &numel, sizeof(int64_t));
        else {
            // 否则，将大端序 CPU 数据转换为小端序存储，并写入文件
            int64_t nsize{};
            torch::utils::THP_encodeInt64Buffer(
                (uint8_t*)&nsize,
                (const int64_t*)&numel,
                torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
                1);
            doWrite(fd, &nsize, sizeof(int64_t));
        }
    }
    
    // 对于字节大小为 1 或者本地字节序为小端序的情况，使用快速路径直接写入数据
    if (element_size == 1 ||
        torch::utils::THP_nativeByteOrder() ==
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
        doWrite(fd, data, size_bytes);
    } else {
        // 否则，逐块进行大小端序转换，并写入文件
        size_t buffer_size = std::min(numel, (size_t)5000);
        std::vector<uint8_t> le_buffer;
        le_buffer.resize(buffer_size * element_size);
        for (size_t i = 0; i < numel; i += buffer_size) {
            size_t to_convert = std::min(numel - i, buffer_size);
            if (element_size == 2) {
                torch::utils::THP_encodeInt16Buffer(
                    le_buffer.data(),
                    (const int16_t*)data + i,
                    torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
                    to_convert);
            } else if (element_size == 4) {
                torch::utils::THP_encodeInt32Buffer(
                    le_buffer.data(),
                    (const int32_t*)data + i,
                    torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
                    to_convert);
            } else if (element_size == 8) {
                torch::utils::THP_encodeInt64Buffer(
                    le_buffer.data(),
                    (const int64_t*)data + i,
                    torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
                    to_convert);
            }
            // 将转换后的数据块写入文件
            doWrite(fd, le_buffer.data(), to_convert * element_size);
        }
    }
}

// 实例化模板，用于将 int 类型数据写入文件
template void THPStorage_writeFileRaw<int>(
    c10::StorageImpl* self,
    int fd,
    bool save_size,
    uint64_t element_size);

// 实例化模板，用于将 PyObject* 类型数据写入文件
template void THPStorage_writeFileRaw<PyObject*>(
    c10::StorageImpl* self,
    PyObject* fd,
    bool save_size,
    uint64_t element_size);

// 从文件中读取原始数据到 StorageImpl 对象的函数模板
template <class io>
c10::intrusive_ptr<c10::StorageImpl> THPStorage_readFileRaw(
    io file,  // 输入的文件流对象
    c10::intrusive_ptr<c10::StorageImpl> storage,  // 存储数据的对象指针
    uint64_t element_size) {  // 每个元素的大小

  c10::OptionalDeviceGuard guard;  // 可选的设备守护，用于设备管理
  if (storage.defined()) {
    guard.reset_device(storage->device());  // 如果 storage 已定义，重置设备到 storage 的设备
  }

  int64_t size{};  // 数据大小初始化为零
  doRead(file, &size, sizeof(int64_t));  // 从文件中读取 int64_t 类型的数据到 size

  // 如果当前系统是大端字节序，则将存储的数据转换为大端字节序
  if (torch::utils::THP_nativeByteOrder() ==
      torch::utils::THPByteOrder::THP_BIG_ENDIAN) {
    int64_t tsize = size;  // 将 size 转换为大端字节序
    torch::utils::THP_decodeInt64Buffer(&size, (const uint8_t*)&tsize, true, 1);  // 解码 int64_t 缓冲区
  }

  size_t nbytes = element_size * size;  // 计算总字节数

  // 如果 storage 未定义，则创建一个新的 StorageImpl 对象
  if (!storage.defined()) {
    storage = c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        c10::GetDefaultCPUAllocator(),
        /*resizable=*/true);
  } else {
    size_t _storage_nbytes = storage->nbytes();
    // 检查存储的字节大小是否与预期的 nbytes 相符
    TORCH_CHECK(
        _storage_nbytes == nbytes,
        "storage has wrong byte size: expected %ld got %ld",
        nbytes,
        _storage_nbytes);
  }

  // 创建指向数据的指针，并根据存储设备类型分配内存
  std::unique_ptr<char[]> cpu_data;

  uint8_t* data{};
  if (storage->device_type() == at::kCPU) {
    data = static_cast<uint8_t*>(storage->mutable_data());  // 获取 CPU 上的数据指针
  } else {
    // 在非 CPU 设备上分配内存并获取数据指针
    cpu_data = std::unique_ptr<char[]>(new char[nbytes]);
    data = (uint8_t*)cpu_data.get();
  }

  // 如果元素大小为 1 或者系统字节序为小端，采用快速路径
  if (element_size == 1 ||
      torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    doRead(file, data, storage->nbytes());  // 从文件中读取数据到存储指针
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    // 分配用于存储小端数据的缓冲区
    std::unique_ptr<uint8_t[]> le_buffer(
        new uint8_t[buffer_size * element_size]);

    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      doRead(file, le_buffer.get(), element_size * to_convert);

      // 根据元素大小进行相应的解码操作
      if (element_size == 2) {
        torch::utils::THP_decodeInt16Buffer(
            (int16_t*)data + i, le_buffer.get(), true, to_convert);
      } else if (element_size == 4) {
        torch::utils::THP_decodeInt32Buffer(
            (int32_t*)data + i, le_buffer.get(), true, to_convert);
      } else if (element_size == 8) {
        torch::utils::THP_decodeInt64Buffer(
            (int64_t*)data + i, le_buffer.get(), true, to_convert);
      }
    }
  }

  // 如果存储不在 CPU 上，则进行数据拷贝到 CPU
  if (storage->device_type() != at::kCPU) {
    // 使用 tensor.copy_() 在所有非 CPU 设备上实现从主机到设备的数据传输
    auto cpu_tensor = at::from_blob(
        (void*)data,                                    // 从指定数据创建 CPU 上的 Tensor
        {static_cast<int64_t>(nbytes)},                 // Tensor 的大小为数据字节数
        at::device(at::kCPU).dtype(c10::kByte));        // 在 CPU 上创建字节类型的 Tensor
    auto device_tensor = at::from_blob(
        storage->mutable_data(),                        // 从存储器中获取可变数据指针
        {static_cast<int64_t>(nbytes)},                 // Tensor 的大小为数据字节数
        {1},                                            // 数据的步长
        nullptr,                                        // 自动计算存储器位置
        at::device(storage->device()).dtype(c10::kByte), // 在存储器所在设备上创建字节类型的 Tensor
        {storage->device()});                           // 指定存储器所在的设备
    device_tensor.copy_(cpu_tensor);                    // 将 CPU 上的 Tensor 数据复制到设备上的 Tensor
  }
  return storage;                                       // 返回存储器指针
}



// 结束了一个 C++ 的模板函数的定义，该函数接受 int 类型的 fd、StorageImpl 类型的 storage 和 element_size 参数，并返回一个 StorageImpl 类型的 intrusive_ptr 对象。
template c10::intrusive_ptr<c10::StorageImpl> THPStorage_readFileRaw<int>(
    int fd,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size);



// 结束了一个 C++ 的模板函数的定义，该函数接受 PyObject* 类型的 fd、StorageImpl 类型的 storage 和 element_size 参数，并返回一个 StorageImpl 类型的 intrusive_ptr 对象。
template c10::intrusive_ptr<c10::StorageImpl> THPStorage_readFileRaw<PyObject*>(
    PyObject* fd,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size);
```