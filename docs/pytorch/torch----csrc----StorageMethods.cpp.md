# `.\pytorch\torch\csrc\StorageMethods.cpp`

```py
// 包含 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>
// 对于 MSC 编译器，包含 Win32 头文件
#ifdef _MSC_VER
#include <c10/util/win32-headers.h>
#endif
// 包含 Python 结构成员定义头文件
#include <structmember.h>

// 包含 CPU 分配器相关头文件
#include <c10/core/CPUAllocator.h>
// 包含共享内存库头文件
#include <libshm.h>
// 包含 CUDA IPC 类型定义头文件
#include <torch/csrc/CudaIPCTypes.h>
// 包含设备相关定义头文件
#include <torch/csrc/Device.h>
// 包含动态类型定义头文件
#include <torch/csrc/DynamicTypes.h>
// 包含 Torch 的 C API 头文件
#include <torch/csrc/THP.h>
// 包含自动微分工具函数头文件
#include <torch/csrc/autograd/utils/wrap_outputs.h>
// 包含数据复制工具函数头文件
#include <torch/csrc/copy_utils.h>

// 包含 C10 库的智能指针头文件
#include <c10/util/intrusive_ptr.h>
// 使用 fmt 库进行格式化输出
#include <fmt/format.h>

// 包含 Tensor 存储相关头文件
#include <torch/csrc/Storage.h>
// 包含 Tensor 存储方法头文件
#include <torch/csrc/StorageMethods.h>

// 包含 ATen 库的核心头文件
#include <ATen/ATen.h>
// 包含 ATen 的映射分配器头文件
#include <ATen/MapAllocator.h>
// 包含 ATen 的存储工具头文件
#include <ATen/StorageUtils.h>
// 包含 Python C 函数帮助函数头文件
#include <torch/csrc/utils/pycfunction_helpers.h>
// 包含 Python 参数解析器头文件
#include <torch/csrc/utils/python_arg_parser.h>
// 包含 Python 数字处理头文件
#include <torch/csrc/utils/python_numbers.h>

// 如果使用 CUDA，则包含 CUDA 相关的 Resize 头文件
#ifdef USE_CUDA
#include <ATen/native/cuda/Resize.h>
#include <cuda_runtime.h>
#endif

// 包含 ATen 的私有扩展接口头文件
#include <ATen/detail/PrivateUse1HooksInterface.h>
// 包含 ATen 的 Resize 头文件
#include <ATen/native/Resize.h>

// 如果是 MSC 编译器，则定义 LSEEK 为 _lseeki64，否则为 lseek
#ifdef _MSC_VER
#define LSEEK _lseeki64
#else
#define LSEEK lseek
#endif

// 定义静态函数 THPStorage_nbytes
static PyObject* THPStorage_nbytes(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 确保 THPStorage 不为空
  THPStorage_assertNotNull(self);
  // 返回封装在 THPStorage_Unpack 中的数据指向的字节数量
  return py::cast(THPStorage_Unpack(self).sym_nbytes()).release().ptr();
  END_HANDLE_TH_ERRORS
}

// 定义静态函数 THPStorage_dataPtr
static PyObject* THPStorage_dataPtr(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 使用 THPStorage_Unpack 解包 self
  auto self_ = THPStorage_Unpack(self);
  // 检查是否为无效 Python 存储，见注释 [Invalid Python Storages]
  auto invalid = self_.data() == nullptr &&
      self_.device_type() != c10::DeviceType::Meta && self_.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid,
      "Attempted to access the data pointer on an invalid python storage.")
  // 返回 self 的可变数据指针的 PyLong 对象
  return PyLong_FromVoidPtr(self_.mutable_data());
  END_HANDLE_TH_ERRORS
}

// 定义静态函数 THPStorage_resizable
static PyObject* THPStorage_resizable(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 确保 THPStorage 不为空
  THPStorage_assertNotNull(self);
  // 返回 THPStorage_Unpack 中的 resizable 属性值的 PyBool 对象
  return PyBool_FromLong(THPStorage_Unpack(self).resizable());
  END_HANDLE_TH_ERRORS
}

// 定义静态函数 THPStorage_copy_
static PyObject* THPStorage_copy_(
    PyObject* self,
    PyObject* args,
    ```
    // 定义函数 copy_，接受两个参数：Storage src 和可选的布尔值 non_blocking
    PyObject* copy_(Storage src, bool? non_blocking=None) {
      // 处理可能出现的异常
      HANDLE_TH_ERRORS
      // 确保 self（即调用该函数的对象）不为 null
      THPStorage_assertNotNull(self);
    
      // 创建一个 C++ 的 at::Storage 对象 self_，并使用 torch::createStorage 创建它
      at::Storage self_ = torch::createStorage(self);
    
      // 创建静态的 PythonArgParser 对象 parser，用于解析参数
      static torch::PythonArgParser parser({
          "copy_(Storage src, bool? non_blocking=None)",
      });
    
      // 解析传入的参数 args 和 kwargs，并存储到 parsed_args 中
      torch::ParsedArgs<2> parsed_args;
      auto r = parser.parse(args, kwargs, parsed_args);
    
      // 从解析结果中获取第一个参数作为 src 的存储对象
      at::Storage src = r.storage(0);
      // 从解析结果中获取第二个参数作为是否非阻塞复制的标志位，如果未提供则默认为 false
      bool non_blocking = r.toBoolOptional(1).value_or(false);
    
      // 检查 src 的数据指针是否为 null，且其设备类型不是 Meta，且符号化字节数不为 0
      // 参考文档中的 "Note [Invalid Python Storages]"
      auto invalid = src.data() == nullptr &&
          src.device_type() != c10::DeviceType::Meta && src.sym_nbytes() != 0;
      // 如果发现 src 是无效的 Python 存储，则抛出异常
      TORCH_CHECK(
          !invalid, "Attempted to call copy_() on an invalid python storage.")
    
      // 检查 self_ 和 src 的存储字节数是否相等
      TORCH_CHECK(
          self_.nbytes() == src.nbytes(),
          "size does not match, self was ",
          self_.nbytes(),
          " bytes but src was ",
          src.nbytes(),
          " bytes");
    
      // 调用 at::storage_copy 函数，将 src 的数据复制到 self_ 中
      at::storage_copy(self_, src, non_blocking);
    
      // 增加 Python 对象 self 的引用计数，确保返回后不被释放
      Py_INCREF(self);
      // 返回 Python 对象 self
      return self;
    
      // 处理可能出现的异常结束
      END_HANDLE_TH_ERRORS
    }
static PyObject* THPStorage_elementSize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 确保参数不为空
  THPStorage_assertNotNull(_self);
  // 返回存储单元的大小作为 int64 的包装对象
  return THPUtils_packInt64(sizeof(uint8_t));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_new(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 确保参数不为空
  THPStorage_assertNotNull(self);
  // 从对象中提取存储器实例
  c10::Allocator* allocator = THPStorage_Unpack(self).allocator();
  // 创建一个新的可调整大小的 StorageImpl 实例
  auto new_storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      0,
      allocator,
      /*resizable=*/true);

  // 将新创建的 StorageImpl 实例封装为 Python 对象并返回
  return THPStorage_Wrap(std::move(new_storage));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_resize_(PyObject* self, PyObject* number_arg) {
  HANDLE_TH_ERRORS
  // 确保参数不为空
  THPStorage_assertNotNull(self);
  // 从对象中提取存储器实例
  const auto& storage = THPStorage_Unpack(self);
  // 检查存储器是否有效
  auto invalid = storage.data() == nullptr &&
      storage.device_type() != c10::DeviceType::Meta &&
      storage.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call resize_() on an invalid python storage.")
  TORCH_CHECK(
      THPUtils_checkLong(number_arg),
      "resize_ expects an int, "
      "but got ",
      THPUtils_typename(number_arg));
  // 解析并获取新的存储器大小
  int64_t newsize = THPUtils_unpackLong(number_arg);
  // 获取存储器的设备类型
  c10::DeviceType device_type = storage.device_type();
  // 根据设备类型调整存储器的大小
  if (device_type == at::kCUDA) {
#ifdef USE_CUDA
    // 检查是否溢出，并调整 CUDA 存储器的大小
    ptrdiff_t size_bytes_i = newsize;
    TORCH_CHECK(
        !c10::overflows<size_t>(size_bytes_i),
        "Requested storage size (",
        size_bytes_i,
        ") cannot be represented as a size_t");
    const auto size_bytes = static_cast<size_t>(size_bytes_i);
    at::native::resize_bytes_cuda(storage.unsafeGetStorageImpl(), size_bytes);
#else
    TORCH_CHECK(false, "built without USE_CUDA");
#endif
  } else {
    // 调整非 CUDA 存储器的大小
    at::native::resize_bytes_nocuda(storage, newsize);
  }
  // 增加 Python 对象的引用计数并返回自身
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_fill_(PyObject* self, PyObject* number_arg) {
  HANDLE_TH_ERRORS
  // 确保参数不为空
  THPStorage_assertNotNull(self);
  // 从对象中提取存储器实例
  const auto& storage = THPStorage_Unpack(self);
  // 检查存储器是否有效
  auto invalid = storage.data() == nullptr &&
      storage.device_type() != c10::DeviceType::Meta &&
      storage.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call fill_() on an invalid python storage.")
  TORCH_CHECK(
      THPByteUtils_checkReal(number_arg),
      "fill_ expects int, "
      "but got ",
      THPUtils_typename(number_arg));
  // 使用给定值填充存储器
  storage_fill(storage, THPByteUtils_unpackReal(number_arg));
  // 增加 Python 对象的引用计数并返回自身
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_fromBuffer(
    PyObject* _unused,
    PyObject* args,
    //
    PyObject* keywds) {
    // 定义函数，接受 Python 参数对象和关键字参数对象作为输入
    HANDLE_TH_ERRORS
    // 处理 Torch 异常错误
    
    // 初始化变量
    PyObject* obj = nullptr;
    const char* byte_order_str = nullptr;
    Py_ssize_t count = -1, offset = 0;
    PyObject* dtype_obj = nullptr;
    c10::ScalarType scalar_type = at::kByte;
    Py_buffer buffer = {};
    
    // 定义关键字列表和参数类型字符串
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    constexpr const char* kwlist[] = {
        "buffer", "byte_order", "count", "offset", "dtype", nullptr};
    constexpr const char* argtypes = "O|snnO";
    
    // 解析传入的 Python 参数
    if (!PyArg_ParseTupleAndKeywords(
            args,
            keywds,
            argtypes,
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            const_cast<char**>(kwlist),
            &obj,
            &byte_order_str,
            &count,
            &offset,
            &dtype_obj)) {
      return nullptr;
    }
    
    // 检查 'dtype' 参数不为 None
    TORCH_CHECK(dtype_obj != nullptr, "argument 'dtype' cannot be None");
    // 检查 'dtype' 参数是 torch.dtype 类型
    TORCH_CHECK(
        THPDtype_Check(dtype_obj),
        "argument 'dtype' must be of type torch.dtype");
    
    // 将 'dtype' 参数转换为 THPDtype 对象，并获取其标量类型
    auto dtype = reinterpret_cast<THPDtype*>(dtype_obj);
    scalar_type = dtype->scalar_type;
    
    // 检查标量类型是否是与字节顺序无关的类型
    const bool is_endian_independent = (scalar_type == at::kByte) ||
        (scalar_type == at::kChar) || (scalar_type == at::kFloat8_e5m2) ||
        (scalar_type == at::kFloat8_e5m2fnuz) ||
        (scalar_type == at::kFloat8_e4m3fn) ||
        (scalar_type == at::kFloat8_e4m3fnuz);
    
    // 如果不是与字节顺序无关的类型，则检查 'byte_order' 参数是否为 nullptr
    TORCH_CHECK(
        is_endian_independent || (byte_order_str != nullptr),
        "function missing required argument 'byte_order' (pos 2)");
    
    // 计算元素大小
    size_t element_size = c10::elementSize(scalar_type);
    
    // 检查是否需要进行字节交换
    bool do_byte_swap = false;
    if (!is_endian_independent) {
      if (strcmp(byte_order_str, "native") == 0) {
        do_byte_swap = false;
      } else if (strcmp(byte_order_str, "big") == 0) {
        do_byte_swap =
            (torch::utils::THP_LITTLE_ENDIAN ==
             torch::utils::THP_nativeByteOrder());
      } else if (strcmp(byte_order_str, "little") == 0) {
        do_byte_swap =
            (torch::utils::THP_BIG_ENDIAN == torch::utils::THP_nativeByteOrder());
      } else {
        // 抛出异常，说明 'byte_order' 参数无效
        PyErr_Format(
            PyExc_ValueError,
            "invalid byte_order '%s' (expected 'big', 'little', or 'native')",
            byte_order_str);
        return nullptr;
      }
    }
    
    // 获取 Python 对象的缓冲区
    if (PyObject_GetBuffer(obj, &buffer, PyBUF_SIMPLE) < 0)
      return nullptr;
    
    // 检查偏移量是否合法
    if (offset < 0 || offset > buffer.len) {
      PyErr_SetString(
          PyExc_ValueError,
          fmt::format(
              "offset must be non-negative and no greater than buffer length ({}) , but got {}",
              offset,
              buffer.len));
      PyBuffer_Release(&buffer);
      return nullptr;
    }
    
    // 计算需要处理的字节数
    size_t size_bytes = 0;
    if (count < 0) {
      if ((buffer.len - offset) % element_size != 0) {
        PyErr_SetString(
            PyExc_ValueError,
            fmt::format(
                "buffer size ({}) must be a multiple of element size ({})",
                buffer.len,
                element_size));
        PyBuffer_Release(&buffer);
        return nullptr;
      }
      size_bytes = buffer.len - offset;
    // 计算元素个数，如果未指定则根据缓冲区大小和元素大小计算
    count = static_cast<Py_ssize_t>(size_bytes / element_size);
  } else {
    // 根据给定的元素个数计算总字节数
    size_bytes = count * element_size;
  }

  // 检查读取的数据是否超出缓冲区范围
  if (offset + (count * (Py_ssize_t)element_size) > buffer.len) {
    // 设置异常信息并释放缓冲区
    PyErr_SetString(
        PyExc_ValueError,
        fmt::format(
            "buffer has only {} elements after offset {}, but specified a size of {}",
            buffer.len - offset,
            offset,
            count));
    PyBuffer_Release(&buffer);
    return nullptr;
  }

  // 将缓冲区的起始地址转换为 uint8_t 指针
  uint8_t* src = (uint8_t*)buffer.buf;
  // 创建一个新的 StorageImpl 对象来存储解码后的数据
  auto storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      c10::GetDefaultCPUAllocator(),
      /*resizable=*/true);

  // 根据数据类型和大小端信息选择相应的解码函数进行数据解析
  if (is_endian_independent) {
    // 使用 memcpy 将数据从源缓冲区复制到新创建的 Storage 中
    memcpy(storage->mutable_data(), src + offset, count);
  } else if (scalar_type == at::kBool) {
    // 解码布尔类型的数据
    torch::utils::THP_decodeBoolBuffer(
        static_cast<bool*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kShort) {
    // 解码 int16 类型的数据
    torch::utils::THP_decodeInt16Buffer(
        static_cast<int16_t*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kInt) {
    // 解码 int32 类型的数据
    torch::utils::THP_decodeInt32Buffer(
        static_cast<int32_t*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kLong) {
    // 解码 int64 类型的数据
    torch::utils::THP_decodeInt64Buffer(
        static_cast<int64_t*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kHalf) {
    // 解码半精度浮点数数据
    torch::utils::THP_decodeHalfBuffer(
        static_cast<c10::Half*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kBFloat16) {
    // 解码 BF16 浮点数数据
    torch::utils::THP_decodeBFloat16Buffer(
        static_cast<c10::BFloat16*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kFloat) {
    // 解码单精度浮点数数据
    torch::utils::THP_decodeFloatBuffer(
        static_cast<float*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kDouble) {
    // 解码双精度浮点数数据
    torch::utils::THP_decodeDoubleBuffer(
        static_cast<double*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kComplexFloat) {
    // 解码复数类型的单精度浮点数数据
    torch::utils::THP_decodeComplexFloatBuffer(
        static_cast<c10::complex<float>*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kComplexDouble) {
    // 解码复数类型的双精度浮点数数据
    // 以下部分未提供完整代码，需在此处继续填写注释
    // 调用 Torch C++ 库中的函数 THP_decodeComplexDoubleBuffer，
    // 解码复杂双精度缓冲区数据到给定的 storage 中
    torch::utils::THP_decodeComplexDoubleBuffer(
        static_cast<c10::complex<double>*>(storage->mutable_data()),  // 将 storage 的数据转换为双精度复数类型指针
        src + offset,   // 源数据的偏移量
        do_byte_swap,   // 是否需要进行字节交换的标志
        count);         // 要处理的元素数量
  } else {
    // 如果未知的数据类型，抛出错误信息
    TORCH_CHECK(false, "Unknown type: ", scalar_type);
  }

  // 释放 Python 缓冲区对象
  PyBuffer_Release(&buffer);

  // 包装 Torch 的 storage 并返回给 Python
  return THPStorage_Wrap(storage);

  // 处理 Torch 错误的尾部
  END_HANDLE_TH_ERRORS
static PyObject* THPStorage_fromFile(
    PyObject* _unused,
    PyObject* args,
    PyObject* keywds) {
  HANDLE_TH_ERRORS
  // 声明文件名、字节数和共享标志
  const char* filename = nullptr;
  Py_ssize_t nbytes = 0;
  int shared = 0;
  // 定义关键字参数列表，包括文件名、共享标志和字节数
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {"filename", "shared", "nbytes", nullptr};
  // 解析传入的参数，支持字符串、整数和可选的整数参数
  if (!PyArg_ParseTupleAndKeywords(
          args,
          keywds,
          "s|in",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &filename,
          &shared,
          &nbytes)) {
    return nullptr;
  }
  // 如果 shared 标志为真，则设置为对应的映射共享内存标志
  if (shared)
    shared = at::ALLOCATOR_MAPPED_SHARED;

  // 实际读取的字节数，默认为-1
  size_t actual_nbytes = -1;
  // 创建一个存储实例，使用给定的文件名、共享标志和字节数，返回数据指针并设置实际读取的字节数
  auto storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
      at::MapAllocator::makeDataPtr(filename, shared, nbytes, &actual_nbytes),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  // 如果传入的字节数小于等于0，则设置实际读取的字节数
  if (nbytes <= 0) {
    storage->set_nbytes(actual_nbytes);
  }

  // 使用新创建的存储实例构造一个 Python 对象，并返回该对象
  return THPStorage_NewWithStorage(
      THPStorageClass,
      std::move(storage),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
  END_HANDLE_TH_ERRORS
}

PyObject* THPStorage_writeFile(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 确保存储对象不为空
  THPStorage_assertNotNull(self);
  // 解包存储对象
  const auto& storage = THPStorage_Unpack(self);
  // 检查存储对象是否为有效 Python 存储，不是则抛出错误
  // See Note [Invalid Python Storages]
  auto invalid = storage.data() == nullptr &&
      storage.device_type() != c10::DeviceType::Meta &&
      storage.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call _write_file() on an invalid python storage.")
  // 获取参数元组中的文件对象、是否真实文件、是否保存大小和元素大小对象
  PyObject* file = PyTuple_GetItem(args, 0);
  bool is_real_file = PyTuple_GetItem(args, 1) == Py_True;
  bool save_size = PyTuple_GetItem(args, 2) == Py_True;
  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 3);

  // 确保元素大小对象不为空
  TORCH_CHECK(
      element_size_obj != Py_None, "_write_file: need to specify element size");
  // 解包并获取元素大小
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  // 如果不是真实文件，则调用原始文件写入函数，返回 None
  if (!is_real_file) {
    THPStorage_writeFileRaw<PyObject*>(
        storage.unsafeGetStorageImpl(), file, save_size, element_size);
    Py_RETURN_NONE;
  }

  // 获取文件描述符，并确保获取成功
  int fd = PyObject_AsFileDescriptor(file);
  TORCH_CHECK(
      fd != -1,
      "_write_file couldn't retrieve a file descriptor "
      "from given object");
  // 调用原始文件写入函数，传入文件描述符，返回 None
  THPStorage_writeFileRaw(
      storage.unsafeGetStorageImpl(), fd, save_size, element_size);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
PyObject* THPStorage_newWithFile(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  // 检查参数数量是否为2
  TORCH_CHECK(
      PyTuple_Size(args) == 2, "_new_with_file takes exactly two arguments");
  // 将第一个参数转换为文件描述符
  int fd = PyObject_AsFileDescriptor(PyTuple_GetItem(args, 0));
  // 检查文件描述符是否有效
  TORCH_CHECK(
      fd != -1,
      "_new_with_file couldn't retrieve a file "
      "descriptor from given object");
  // 获取第二个参数（元素大小）
  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 1);
  // 检查元素大小参数不为None
  TORCH_CHECK(
      element_size_obj != Py_None,
      "_new_with_file: need to specify element size");
  // 解包元素大小参数
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  // 调用 THPStorage_readFileRaw 函数读取文件中的数据
  auto storage = THPStorage_readFileRaw<int>(fd, {}, element_size);
  // 如果读取失败，返回空指针
  if (!storage.defined())
    return nullptr;
  // 封装存储对象并返回
  return THPStorage_Wrap(std::move(storage));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_setFromFile(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 确保 self 不为空
  THPStorage_assertNotNull(self);
  // 解包存储对象
  const auto& storage = THPStorage_Unpack(self);
  // 获取函数参数
  PyObject* file = PyTuple_GET_ITEM(args, 0);
  PyObject* offset = PyTuple_GET_ITEM(args, 1);
  bool is_real_file = PyTuple_GET_ITEM(args, 2) == Py_True;

  // 获取元素大小参数
  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 3);
  // 检查元素大小参数不为None
  TORCH_CHECK(
      element_size_obj != Py_None,
      "_set_from_file: need to specify element size");
  // 解包元素大小参数
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  if (!is_real_file) {
    // 对于类文件对象，暂不支持偏移，因此报错
    TORCH_CHECK(
        offset == Py_None,
        "_set_from_file: offset is NYI for filelike objects");

    // 从类文件对象中读取数据
    auto self_storage_impl = c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(
        storage.unsafeGetStorageImpl());
    auto storage_impl = THPStorage_readFileRaw<PyObject*>(
        file, std::move(self_storage_impl), element_size);
    // 如果读取失败，返回空指针
    if (!storage_impl.defined()) {
      return nullptr;
    }
    // 增加对象的引用计数并返回 self
    Py_INCREF(self);
    return (PyObject*)self;
  }

  // 对于真实文件，获取文件描述符
  const int fd = PyObject_AsFileDescriptor(file);
  // 记录文件当前位置
  const auto fd_original_pos = LSEEK(fd, 0, SEEK_CUR);
  // 如果指定了偏移量，将文件指针移到指定位置
  if (offset != Py_None) {
    LSEEK(fd, THPUtils_unpackLong(offset), SEEK_SET);
  }
  // 检查文件描述符是否有效
  TORCH_CHECK(
      fd != -1,
      "_set_from_file couldn't retrieve a file "
      "descriptor from given object");
  // 从文件中读取数据到存储对象
  auto self_storage_impl = c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(
      storage.unsafeGetStorageImpl());
  auto storage_impl =
      THPStorage_readFileRaw<int>(fd, self_storage_impl, element_size);
  // 如果读取失败，返回空指针
  if (!storage_impl.defined())
    return nullptr;
  // 增加对象的引用计数并返回 self
  Py_INCREF(self);

  // 将文件描述符恢复到原始位置，并更新 Python 端的文件对象位置
  const auto fd_current_pos = LSEEK(fd, 0, SEEK_CUR);
  LSEEK(fd, fd_original_pos, SEEK_SET);
  const auto seek_return =
      PyObject_CallMethod(file, "seek", "Li", (long long)fd_current_pos, 0);
  // 如果 seek 失败，返回空指针
  if (seek_return == nullptr) {
    // 返回空指针（nullptr），表示函数执行失败或异常情况
    return nullptr;
  }
  // 减少 Python 对象的引用计数，避免内存泄漏
  Py_DECREF(seek_return);

  // 返回当前对象指针，表示函数成功执行完成
  return self;
  // 结束异常处理块，确保异常能够被适当地处理或传播
  END_HANDLE_TH_ERRORS
}

// 函数：设置存储对象的 cdata 值
PyObject* THPStorage__setCdata(PyObject* _self, PyObject* new_cdata) {
  HANDLE_TH_ERRORS
  // 将 PyObject 转换为 THPStorage 对象
  auto self = (THPStorage*)_self;
  // 检查传入的 new_cdata 是否为长整型
  TORCH_CHECK(
      THPUtils_checkLong(new_cdata),
      "given an invalid argument to "
      "_set_cdata - expected an int or long, but got ",
      THPUtils_typename(new_cdata));
  // 将 new_cdata 转换为 StorageImpl 指针
  c10::StorageImpl* ptr = (c10::StorageImpl*)PyLong_AsVoidPtr(new_cdata);
  // 销毁原有的 cdata 对象并创建新的 MaybeOwned<c10::Storage> 对象
  self->cdata.~MaybeOwned<c10::Storage>();
  self->cdata = c10::MaybeOwned<c10::Storage>::owned(
      c10::Storage(c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(ptr)));
  // 增加自身的引用计数
  Py_INCREF(self);
  // 返回自身对象的 PyObject 指针
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

// 函数：对存储对象进行字节交换
PyObject* THPStorage_byteswap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 检查参数 args 是否包含一个元素
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 1, "tuple of 1 item expected");
  // 获取 args 中的第一个元素
  PyObject* _elem_size = PyTuple_GET_ITEM(args, 0);
  // 检查 _elem_size 是否为长整型
  TORCH_CHECK(
      THPUtils_checkLong(_elem_size), "_byteswap(): arg must be an 'int'");
  // 将 _elem_size 转换为整型
  auto elem_size = THPUtils_unpackLong(_elem_size);
  // 检查 elem_size 是否为 1, 2, 4, 或 8
  TORCH_CHECK(
      elem_size == 1 || elem_size == 2 || elem_size == 4 || elem_size == 8,
      "elem_size must be 1, 2, 4, or 8");

  // 解包存储对象
  const auto& storage = THPStorage_Unpack(self);
  // 计算存储对象的字节数
  const auto nbytes = static_cast<uint64_t>(storage.nbytes());
  const uint64_t count = nbytes / elem_size;

  // 如果 elem_size 为 1，则直接返回 None
  if (elem_size == 1) {
    Py_RETURN_NONE;
  }
  // 检查数据长度是否为 elem_size 的整数倍
  TORCH_CHECK(
      nbytes % elem_size == 0,
      "the length of data is not a multiple of ",
      elem_size);

  // 根据 elem_size 执行对应的字节交换操作
  if (elem_size == 2) {
    auto buffer = static_cast<uint16_t*>(storage.mutable_data());
    for (uint64_t i = 0; i < count; i++, buffer++) {
      *buffer = thp_bswap16(*buffer);
    }
  } else if (elem_size == 4) {
    auto buffer = static_cast<uint32_t*>(storage.mutable_data());
    for (uint64_t i = 0; i < count; i++, buffer++) {
      *buffer = thp_bswap32(*buffer);
    }
  } else if (elem_size == 8) {
    auto buffer = static_cast<uint64_t*>(storage.mutable_data());
    for (uint64_t i = 0; i < count; i++, buffer++) {
      *buffer = thp_bswap64(*buffer);
    }
  }

  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 函数：修复存储对象的弱引用
static PyObject* THPStorage_fix_weakref(PyObject* self, PyObject* noargs) {
  // 解包存储对象
  const auto& storage = THPStorage_Unpack(self);
  // 减少存储对象的 Python 引用计数
  Py_DECREF(THPStorage_Wrap(storage));
  // 返回 None
  Py_RETURN_NONE;
}

// 函数：获取存储对象的文件名
static PyObject* THPStorage__get_filename(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  // 解包存储对象
  const auto& self_ = THPStorage_Unpack(self);
  // 获取存储对象的数据指针
  const c10::DataPtr& data_ptr = self_.data_ptr();
  // 从数据指针获取 MapAllocator
  at::MapAllocator* map_allocator = at::MapAllocator::fromDataPtr(data_ptr);

  // 如果 MapAllocator 为 nullptr 或不是共享映射的标志，则返回 None
  if (map_allocator == nullptr ||
      !(map_allocator->flags() & at::ALLOCATOR_MAPPED_SHARED)) {
    Py_RETURN_NONE;
  }
  // 获取 MapAllocator 的文件名
  std::string filename = map_allocator->filename();

  // 将文件名打包成 Python 字符串对象并返回
  return THPUtils_packString(filename);
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
// 定义存储对象的方法列表
static PyMethodDef THPStorage_methods[] = {
    // 注册一个名为 "copy_" 的方法，其实现为 THPStorage_copy_，接受可变参数和关键字参数，无返回值
    {"copy_", castPyCFunctionWithKeywords(THPStorage_copy_), METH_VARARGS | METH_KEYWORDS, nullptr},
    // 注册一个名为 "element_size" 的方法，其实现为 THPStorage_elementSize，不接受参数，无返回值
    {"element_size", THPStorage_elementSize, METH_NOARGS, nullptr},
    // 注册一个名为 "fill_" 的方法，其实现为 THPStorage_fill_，接受一个参数对象，无返回值
    {"fill_", THPStorage_fill_, METH_O, nullptr},
    // 注册一个名为 "new" 的方法，其实现为 THPStorage_new，不接受参数，无返回值
    {"new", THPStorage_new, METH_NOARGS, nullptr},
    // 注册一个名为 "resize_" 的方法，其实现为 THPStorage_resize_，接受一个参数对象，无返回值
    {"resize_", THPStorage_resize_, METH_O, nullptr},
    // 注册一个名为 "nbytes" 的方法，其实现为 THPStorage_nbytes，不接受参数，无返回值
    {"nbytes", THPStorage_nbytes, METH_NOARGS, nullptr},
    // 注册一个名为 "data_ptr" 的方法，其实现为 THPStorage_dataPtr，不接受参数，无返回值
    {"data_ptr", THPStorage_dataPtr, METH_NOARGS, nullptr},
    // 注册一个名为 "resizable" 的方法，其实现为 THPStorage_resizable，不接受参数，无返回值
    {"resizable", THPStorage_resizable, METH_NOARGS, nullptr},
    // 注册一个名为 "_write_file" 的方法，其实现为 THPStorage_writeFile，接受可变参数，无返回值
    {"_write_file", THPStorage_writeFile, METH_VARARGS, nullptr},
    // 注册一个名为 "_new_with_file" 的方法，其实现为 THPStorage_newWithFile，接受可变参数和静态方法标记，无返回值
    {"_new_with_file", THPStorage_newWithFile, METH_VARARGS | METH_STATIC, nullptr},
    // 注册一个名为 "_set_from_file" 的方法，其实现为 THPStorage_setFromFile，接受可变参数，无返回值
    {"_set_from_file", THPStorage_setFromFile, METH_VARARGS, nullptr},
    // 注册一个名为 "from_buffer" 的方法，其实现为 THPStorage_fromBuffer，接受可变参数、关键字参数和静态方法标记，无返回值
    {"from_buffer", castPyCFunctionWithKeywords(THPStorage_fromBuffer), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
    // 注册一个名为 "from_file" 的方法，其实现为 THPStorage_fromFile，接受可变参数、关键字参数和静态方法标记，无返回值
    {"from_file", castPyCFunctionWithKeywords(THPStorage_fromFile), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
    // 注册一个名为 "_set_cdata" 的方法，其实现为 THPStorage__setCdata，接受一个参数对象，无返回值
    {"_set_cdata", THPStorage__setCdata, METH_O, nullptr},
    // 注册一个名为 "_byteswap" 的方法，其实现为 THPStorage_byteswap，接受可变参数，无返回值
    {"_byteswap", THPStorage_byteswap, METH_VARARGS, nullptr},
    // 注册一个名为 "_fix_weakref" 的方法，其实现为 THPStorage_fix_weakref，不接受参数，无返回值
    {"_fix_weakref", THPStorage_fix_weakref, METH_NOARGS, nullptr},
    // 注册一个名为 "_get_filename" 的方法，其实现为 THPStorage__get_filename，不接受参数，无返回值
    {"_get_filename", THPStorage__get_filename, METH_NOARGS, nullptr},
    // 结束方法注册列表的标志
    {nullptr}
};
# 返回 THPStorage_methods 指针，该指针指向 PyMethodDef 结构体数组
PyMethodDef* THPStorage_getMethods() {
    return THPStorage_methods;
}
```