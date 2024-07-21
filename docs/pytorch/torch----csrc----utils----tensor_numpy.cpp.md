# `.\pytorch\torch\csrc\utils\tensor_numpy.cpp`

```py
#ifndef USE_NUMPY
// 如果没有启用 NumPy 支持，则定义以下命名空间
namespace torch::utils {
// 将 Torch 张量转换为 NumPy 数组，如果未启用 NumPy 支持则抛出异常
PyObject* tensor_to_numpy(const at::Tensor&, bool) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}
// 从 NumPy 数组创建 Torch 张量，如果未启用 NumPy 支持则抛出异常
at::Tensor tensor_from_numpy(
    PyObject* obj,
    bool warn_if_not_writeable /*=true*/) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}

// 检查是否可用 NumPy，如果未启用 NumPy 支持则抛出异常
bool is_numpy_available() {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}

// 检查对象是否为 NumPy 整数类型，如果未启用 NumPy 支持则抛出异常
bool is_numpy_int(PyObject* obj) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}

// 检查对象是否为 NumPy 标量类型，如果未启用 NumPy 支持则抛出异常
bool is_numpy_scalar(PyObject* obj) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}

// 从 CUDA 数组接口创建 Torch 张量，如果未启用 NumPy 支持则抛出异常
at::Tensor tensor_from_cuda_array_interface(PyObject* obj) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}

// 如果未启用 NumPy 支持则抛出异常
void warn_numpy_not_writeable() {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}

// 检查是否需要验证 NumPy 的 dlpack 删除器问题，这是一个空操作
void validate_numpy_for_dlpack_deleter_bug() {}

// 返回 NumPy 的 dlpack 删除器是否存在问题的标志，始终返回 false
bool is_numpy_dlpack_deleter_bugged() {
  return false;
}
} // namespace torch::utils

#else
// 如果启用了 NumPy 支持，则包含以下头文件和命名空间
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/object_ptr.h>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace at;
using namespace torch::autograd;

namespace torch {
namespace utils {

// 检查是否可用 NumPy
bool is_numpy_available() {
  // 使用 Lambda 表达式检查 NumPy 是否可用，如果初始化失败则打印警告并返回 false
  static bool available = []() {
    if (_import_array() >= 0) {
      return true;
    }
    // 尝试获取异常消息，打印警告并返回 false
    std::string message = "Failed to initialize NumPy";
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    if (auto str = value ? PyObject_Str(value) : nullptr) {
      if (auto enc_str = PyUnicode_AsEncodedString(str, "utf-8", "strict")) {
        if (auto byte_str = PyBytes_AS_STRING(enc_str)) {
          message += ": " + std::string(byte_str);
        }
        Py_XDECREF(enc_str);
      }
      Py_XDECREF(str);
    }
    PyErr_Clear();
    TORCH_WARN(message);
    return false;
  }();
  return available;
}

// 将 IntArrayRef 转换为 NumPy 的形状信息（npy_intp 数组）
static std::vector<npy_intp> to_numpy_shape(IntArrayRef x) {
  // 将 IntArrayRef 中的形状信息转换为 npy_intp 数组
  auto nelem = x.size();
  auto result = std::vector<npy_intp>(nelem);
  for (const auto i : c10::irange(nelem)) {
    result[i] = static_cast<npy_intp>(x[i]);
  }
  return result;
}

// 将 npy_intp 数组转换为 ATen 的形状信息（int64_t 数组）
static std::vector<int64_t> to_aten_shape(int ndim, npy_intp* values) {
  // 将 npy_intp 数组中的形状信息转换为 int64_t 数组
  auto result = std::vector<int64_t>(ndim);
  for (const auto i : c10::irange(ndim)) {
    # 将 values 数组中的每个元素转换为 int64_t 类型，并存入 result 数组的相应位置
    result[i] = static_cast<int64_t>(values[i]);
  }
  # 返回存有转换后数据的 result 数组
  return result;
}

// 将 Python 对象转换为 ATen 张量的形状
static std::vector<int64_t> seq_to_aten_shape(PyObject* py_seq) {
  // 获取序列的长度作为张量的维度数
  int ndim = PySequence_Length(py_seq);
  if (ndim == -1) {
    // 如果长度获取失败，抛出类型错误异常
    throw TypeError("shape and strides must be sequences");
  }
  // 创建一个存储维度大小的向量
  auto result = std::vector<int64_t>(ndim);
  // 遍历每一个维度
  for (const auto i : c10::irange(ndim)) {
    // 获取序列中的每一项
    auto item = THPObjectPtr(PySequence_GetItem(py_seq, i));
    if (!item)
      // 如果获取项失败，抛出 Python 错误
      throw python_error();

    // 将项转换为 int64_t 类型的数值，并存储到结果中
    result[i] = PyLong_AsLongLong(item);
    // 检查转换是否失败，并抛出 Python 错误
    if (result[i] == -1 && PyErr_Occurred())
      throw python_error();
  }
  // 返回转换后的结果向量
  return result;
}

// 将 ATen 张量转换为 NumPy 数组
PyObject* tensor_to_numpy(const at::Tensor& tensor, bool force /*=false*/) {
  // 检查是否支持 NumPy
  TORCH_CHECK(is_numpy_available(), "Numpy is not available");

  // 检查是否支持具有 Python 调度的张量
  TORCH_CHECK(
      !tensor.unsafeGetTensorImpl()->is_python_dispatch(),
      ".numpy() is not supported for tensor subclasses.");

  // 检查张量布局是否为 Strided
  TORCH_CHECK_TYPE(
      tensor.layout() == Layout::Strided,
      "can't convert ",
      c10::str(tensor.layout()).c_str(),
      " layout tensor to numpy. ",
      "Use Tensor.dense() first.");

  // 如果不强制转换
  if (!force) {
    // 检查张量设备是否为 CPU
    TORCH_CHECK_TYPE(
        tensor.device().type() == DeviceType::CPU,
        "can't convert ",
        tensor.device().str().c_str(),
        " device type tensor to numpy. Use Tensor.cpu() to ",
        "copy the tensor to host memory first.");

    // 检查张量是否需要梯度
    TORCH_CHECK(
        !(at::GradMode::is_enabled() && tensor.requires_grad()),
        "Can't call numpy() on Tensor that requires grad. "
        "Use tensor.detach().numpy() instead.");

    // 检查张量是否具有共轭位
    TORCH_CHECK(
        !tensor.is_conj(),
        "Can't call numpy() on Tensor that has conjugate bit set. ",
        "Use tensor.resolve_conj().numpy() instead.");

    // 检查张量是否具有负位
    TORCH_CHECK(
        !tensor.is_neg(),
        "Can't call numpy() on Tensor that has negative bit set. "
        "Use tensor.resolve_neg().numpy() instead.");
  }

  // 备份处理后的张量，确保不影响原始张量
  auto prepared_tensor = tensor.detach().cpu().resolve_conj().resolve_neg();

  // 获取 NumPy 数组的数据类型
  auto dtype = aten_to_numpy_dtype(prepared_tensor.scalar_type());
  // 获取 NumPy 数组的形状
  auto sizes = to_numpy_shape(prepared_tensor.sizes());
  // 获取 NumPy 数组的步长
  auto strides = to_numpy_shape(prepared_tensor.strides());

  // 转换 Torch 的步长为 NumPy 的步长（以字节为单位）
  auto element_size_in_bytes = prepared_tensor.element_size();
  for (auto& stride : strides) {
    stride *= element_size_in_bytes;
  }

  // 创建一个 NumPy 数组对象
  auto array = THPObjectPtr(PyArray_New(
      &PyArray_Type,
      static_cast<int>(prepared_tensor.dim()),
      sizes.data(),
      dtype,
      strides.data(),
      prepared_tensor.data_ptr(),
      0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
      nullptr));
  // 检查数组对象是否创建成功
  if (!array)
    return nullptr;

  // TODO: 这里尝试通过设置 ndarray 的基础对象来保持底层内存的存活，
  // 禁止对存储进行重新调整。这可能不足够，例如，张量的存储可以通过
  // Tensor.set_ 进行更改，从而释放底层内存。
  // 将张量包装为 Python 对象，以保持内存的存活
  PyObject* py_tensor = THPVariable_Wrap(prepared_tensor);
  if (!py_tensor)
    // 抛出一个 Python 错误对象
    throw python_error();
  // 将 PyArrayObject 对象的 base object 设置为 py_tensor，如果失败则返回空指针
  if (PyArray_SetBaseObject((PyArrayObject*)array.get(), py_tensor) == -1) {
    return nullptr;
  }
  // 使用私有存储 API，设置准备好的张量的存储为不可调整大小
  prepared_tensor.storage().unsafeGetStorageImpl()->set_resizable(false);

  // 释放 array 持有的 PyArrayObject 对象并返回其所有权
  return array.release();
}

// 函数：warn_numpy_not_writeable
// 功能：输出警告，指示给定的 NumPy 数组不可写
void warn_numpy_not_writeable() {
  TORCH_WARN_ONCE(
      "The given NumPy array is not writable, and PyTorch does "
      "not support non-writable tensors. This means writing to this tensor "
      "will result in undefined behavior. "
      "You may want to copy the array to protect its data or make it writable "
      "before converting it to a tensor. This type of warning will be "
      "suppressed for the rest of this program.");
}

// 函数：tensor_from_numpy
// 参数：
//   - obj: PyObject*，指向要转换为 Tensor 的 NumPy 数组
//   - warn_if_not_writeable: bool，如果为 true，当 NumPy 数组不可写时输出警告
// 返回：转换后的 ATen Tensor
at::Tensor tensor_from_numpy(
    PyObject* obj,
    bool warn_if_not_writeable /*=true*/) {
  if (!is_numpy_available()) {
    throw std::runtime_error("Numpy is not available");
  }
  TORCH_CHECK_TYPE(
      PyArray_Check(obj),
      "expected np.ndarray (got ",
      Py_TYPE(obj)->tp_name,
      ")");
  auto array = (PyArrayObject*)obj;

  // 如果 NumPy 数组不可写，并且 warn_if_not_writable 为 true，则输出警告
  if (!PyArray_ISWRITEABLE(array) && warn_if_not_writeable) {
    warn_numpy_not_writeable();
  }

  int ndim = PyArray_NDIM(array);
  auto sizes = to_aten_shape(ndim, PyArray_DIMS(array));
  auto strides = to_aten_shape(ndim, PyArray_STRIDES(array));
  // NumPy 中的 strides 是以字节为单位，而 Torch 中是以元素个数为单位
  auto element_size_in_bytes = PyArray_ITEMSIZE(array);
  for (auto& stride : strides) {
    TORCH_CHECK_VALUE(
        stride % element_size_in_bytes == 0,
        "given numpy array strides not a multiple of the element byte size. "
        "Copy the numpy array to reallocate the memory.");
    stride /= element_size_in_bytes;
  }

  // 检查所有维度的 stride 是否为非负数
  for (const auto i : c10::irange(ndim)) {
    TORCH_CHECK_VALUE(
        strides[i] >= 0,
        "At least one stride in the given numpy array is negative, "
        "and tensors with negative strides are not currently supported. "
        "(You can probably work around this by making a copy of your array "
        " with array.copy().) ");
  }

  // 获取 NumPy 数组的数据指针
  void* data_ptr = PyArray_DATA(array);
  // 检查 NumPy 数组的字节序是否与本机字节序一致
  TORCH_CHECK_VALUE(
      PyArray_EquivByteorders(PyArray_DESCR(array)->byteorder, NPY_NATIVE),
      "given numpy array has byte order different from the native byte order. "
      "Conversion between byte orders is currently not supported.");
  // 在 INCREF 之前执行，以防映射类型不存在并抛出异常
  auto torch_dtype = numpy_dtype_to_aten(PyArray_TYPE(array));
  Py_INCREF(obj);
  // 使用 from_blob 创建 ATen Tensor，并确保在释放时递减引用计数
  return at::lift_fresh(at::from_blob(
      data_ptr,
      sizes,
      strides,
      [obj](void* data) {
        pybind11::gil_scoped_acquire gil;
        Py_DECREF(obj);
      },
      at::device(kCPU).dtype(torch_dtype)));
}

// 函数：aten_to_numpy_dtype
// 功能：将 ATen 标量类型转换为对应的 NumPy 数据类型
// 参数：
//   - scalar_type: ScalarType，ATen 的标量类型
// 返回：对应的 NumPy 数据类型
int aten_to_numpy_dtype(const ScalarType scalar_type) {
  switch (scalar_type) {
    case kDouble:
      return NPY_DOUBLE;
    case kFloat:
      return NPY_FLOAT;
    case kHalf:
      return NPY_HALF;
    case kComplexDouble:
      return NPY_COMPLEX128;
    case kComplexFloat:
      return NPY_COMPLEX64;
    case kLong:
      return NPY_INT64;


这样可以确保代码的每一行都有详细的注释，清晰地解释了其功能和作用。
    # 如果 scalar_type 是 kInt，则返回相应的 NumPy 类型 NPY_INT32
    case kInt:
      return NPY_INT32;
    # 如果 scalar_type 是 kShort，则返回相应的 NumPy 类型 NPY_INT16
    case kShort:
      return NPY_INT16;
    # 如果 scalar_type 是 kChar，则返回相应的 NumPy 类型 NPY_INT8
    case kChar:
      return NPY_INT8;
    # 如果 scalar_type 是 kByte，则返回相应的 NumPy 类型 NPY_UINT8
    case kByte:
      return NPY_UINT8;
    # 如果 scalar_type 是 kUInt16，则返回相应的 NumPy 类型 NPY_UINT16
    case kUInt16:
      return NPY_UINT16;
    # 如果 scalar_type 是 kUInt32，则返回相应的 NumPy 类型 NPY_UINT32
    case kUInt32:
      return NPY_UINT32;
    # 如果 scalar_type 是 kUInt64，则返回相应的 NumPy 类型 NPY_UINT64
    case kUInt64:
      return NPY_UINT64;
    # 如果 scalar_type 是 kBool，则返回相应的 NumPy 类型 NPY_BOOL
    case kBool:
      return NPY_BOOL;
    # 如果 scalar_type 是其他未支持的类型，抛出类型错误并附带错误信息
    default:
      throw TypeError("Got unsupported ScalarType %s", toString(scalar_type));
}

// 将 NumPy 数据类型转换为 ATen 数据类型
ScalarType numpy_dtype_to_aten(int dtype) {
  switch (dtype) {
    case NPY_DOUBLE:
      return kDouble;  // 双精度浮点数对应 ATen 的 kDouble 类型
    case NPY_FLOAT:
      return kFloat;   // 单精度浮点数对应 ATen 的 kFloat 类型
    case NPY_HALF:
      return kHalf;    // 半精度浮点数对应 ATen 的 kHalf 类型
    case NPY_COMPLEX64:
      return kComplexFloat;    // 复数（单精度）对应 ATen 的 kComplexFloat 类型
    case NPY_COMPLEX128:
      return kComplexDouble;   // 复数（双精度）对应 ATen 的 kComplexDouble 类型
    case NPY_INT16:
      return kShort;   // 16 位整数对应 ATen 的 kShort 类型
    case NPY_INT8:
      return kChar;    // 8 位整数对应 ATen 的 kChar 类型
    case NPY_UINT8:
      return kByte;    // 无符号 8 位整数对应 ATen 的 kByte 类型
    case NPY_UINT16:
      return kUInt16;  // 无符号 16 位整数对应 ATen 的 kUInt16 类型
    case NPY_UINT32:
      return kUInt32;  // 无符号 32 位整数对应 ATen 的 kUInt32 类型
    case NPY_UINT64:
      return kUInt64;  // 无符号 64 位整数对应 ATen 的 kUInt64 类型
    case NPY_BOOL:
      return kBool;    // 布尔类型对应 ATen 的 kBool 类型
    default:
      // Workaround: MSVC 不支持具有相同值的两个 case 分支
      if (dtype == NPY_INT || dtype == NPY_INT32) {
        // 覆盖所有情况，使用 NPY_INT，因为 NPY_INT32 是别名，可能等于：
        // - NPY_INT（当 sizeof(int) = 4 且 sizeof(long) = 8）
        // - NPY_LONG（当 sizeof(int) = 4 且 sizeof(long) = 4）
        return kInt;
      } else if (dtype == NPY_LONGLONG || dtype == NPY_INT64) {
        // NPY_INT64 是别名，可能等于：
        // - NPY_LONG（当 sizeof(long) = 8 且 sizeof(long long) = 8）
        // - NPY_LONGLONG（当 sizeof(long) = 4 且 sizeof(long long) = 8）
        return kLong;
      } else {
        break;  // 这里只是一个 workaround 的标记，因为它与上述 case 分支类似
      }
  }
  auto pytype = THPObjectPtr(PyArray_TypeObjectFromType(dtype));
  if (!pytype)
    throw python_error();  // 抛出 Python 错误，表示无法转换给定的 NumPy 类型
  throw TypeError(
      "can't convert np.ndarray of type %s. The only supported types are: "
      "float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.",
      ((PyTypeObject*)pytype.get())->tp_name);  // 抛出类型错误，指出不支持的 NumPy 数据类型
}

// 检查给定对象是否为 NumPy 的整数类型
bool is_numpy_int(PyObject* obj) {
  return is_numpy_available() && PyArray_IsScalar((obj), Integer);
}

// 检查给定对象是否为 NumPy 的布尔类型
bool is_numpy_bool(PyObject* obj) {
  return is_numpy_available() && PyArray_IsScalar((obj), Bool);
}

// 检查给定对象是否为 NumPy 的标量类型
bool is_numpy_scalar(PyObject* obj) {
  return is_numpy_available() &&
      (is_numpy_int(obj) || PyArray_IsScalar(obj, Bool) ||
       PyArray_IsScalar(obj, Floating) ||
       PyArray_IsScalar(obj, ComplexFloating));
}

// 根据 CUDA 数组接口创建 ATen 张量
at::Tensor tensor_from_cuda_array_interface(PyObject* obj) {
  if (!is_numpy_available()) {
    throw std::runtime_error("Numpy is not available");  // 如果 NumPy 不可用，抛出运行时错误
  }
  auto cuda_dict =
      THPObjectPtr(PyObject_GetAttrString(obj, "__cuda_array_interface__"));  // 获取对象的 __cuda_array_interface__ 属性
  TORCH_INTERNAL_ASSERT(cuda_dict);  // 确保获取成功

  if (!PyDict_Check(cuda_dict.get())) {
    throw TypeError("`__cuda_array_interface__` must be a dict");  // 如果不是字典类型，抛出类型错误
  }

  // 提取 `obj.__cuda_array_interface__['shape']` 属性
  std::vector<int64_t> sizes;
  {
    PyObject* py_shape = PyDict_GetItemString(cuda_dict, "shape");
    if (py_shape == nullptr) {
      throw TypeError("attribute `shape` must exist");  // 如果不存在 `shape` 属性，抛出类型错误
    }
  // 将 Python 中的形状转换为 ATen 的形状
  sizes = seq_to_aten_shape(py_shape);
}

// 提取 `obj.__cuda_array_interface__['typestr']` 属性
// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
ScalarType dtype;
// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
int dtype_size_in_bytes;
{
  // 从 cuda_dict 字典中获取 `typestr` 属性
  PyObject* py_typestr = PyDict_GetItemString(cuda_dict, "typestr");
  if (py_typestr == nullptr) {
    // 如果找不到 `typestr` 属性，则抛出类型错误异常
    throw TypeError("attribute `typestr` must exist");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  PyArray_Descr* descr;
  // 检查并转换 `typestr` 为 PyArray_Descr 结构体
  TORCH_CHECK_VALUE(
      PyArray_DescrConverter(py_typestr, &descr), "cannot parse `typestr`");
  // 将 numpy 的数据类型转换为 ATen 的数据类型
  dtype = numpy_dtype_to_aten(descr->type_num);
#if NPY_ABI_VERSION >= 0x02000000
    // 如果 NPY_ABI_VERSION 大于或等于 0x02000000，则使用 PyDataType_ELSIZE(descr) 计算 dtype_size_in_bytes
    dtype_size_in_bytes = PyDataType_ELSIZE(descr);
#else
    // 否则，使用 descr->elsize 计算 dtype_size_in_bytes
    dtype_size_in_bytes = descr->elsize;
#endif
    // 断言 dtype_size_in_bytes 大于 0
    TORCH_INTERNAL_ASSERT(dtype_size_in_bytes > 0);
  }

  // 提取 `obj.__cuda_array_interface__['data']` 属性
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* data_ptr;
  {
    // 获取字典 cuda_dict 中键为 "data" 的值
    PyObject* py_data = PyDict_GetItemString(cuda_dict, "data");
    // 如果找不到该键或值为空指针，则抛出 TypeError 异常
    if (py_data == nullptr) {
      throw TypeError("attribute `shape` data exist");
    }
    // 如果值不是二元组或者长度不为 2，则抛出 TypeError 异常
    if (!PyTuple_Check(py_data) || PyTuple_GET_SIZE(py_data) != 2) {
      throw TypeError("`data` must be a 2-tuple of (int, bool)");
    }
    // 获取二元组的第一个元素并转换为 void* 类型
    data_ptr = PyLong_AsVoidPtr(PyTuple_GET_ITEM(py_data, 0));
    // 如果转换失败，则抛出 python_error 异常
    if (data_ptr == nullptr && PyErr_Occurred()) {
      throw python_error();
    }
    // 获取二元组的第二个元素并转换为 bool 类型
    int read_only = PyObject_IsTrue(PyTuple_GET_ITEM(py_data, 1));
    // 如果转换失败，则抛出 python_error 异常
    if (read_only == -1) {
      throw python_error();
    }
    // 如果 read_only 为真，则抛出 TypeError 异常
    if (read_only) {
      throw TypeError(
          "the read only flag is not supported, should always be False");
    }
  }

  // 提取 `obj.__cuda_array_interface__['strides']` 属性
  std::vector<int64_t> strides;
  {
    // 获取字典 cuda_dict 中键为 "strides" 的值
    PyObject* py_strides = PyDict_GetItemString(cuda_dict, "strides");
    // 如果值不为空且不为 None
    if (py_strides != nullptr && py_strides != Py_None) {
      // 如果长度为 -1 或长度不等于 sizes.size()，则抛出 TypeError 异常
      if (PySequence_Length(py_strides) == -1 ||
          static_cast<size_t>(PySequence_Length(py_strides)) != sizes.size()) {
        throw TypeError(
            "strides must be a sequence of the same length as shape");
      }
      // 将 py_strides 转换为 ATen 格式的 strides
      strides = seq_to_aten_shape(py_strides);

      // 校验 strides 是否符合要求
      // __cuda_array_interface__ 中的 strides 使用字节，而 Torch 中的 strides 使用元素个数
      for (auto& stride : strides) {
        TORCH_CHECK_VALUE(
            stride % dtype_size_in_bytes == 0,
            "given array strides not a multiple of the element byte size. "
            "Make a copy of the array to reallocate the memory.");
        stride /= dtype_size_in_bytes;
      }
    } else {
      // 如果未提供 strides，则使用默认的 ATen strides
      strides = at::detail::defaultStrides(sizes);
    }
  }

  // 确定目标设备，如果 data_ptr 不为空，则不指定目标设备，否则使用当前 CUDA 设备
  const auto target_device = [&]() -> std::optional<Device> {
    // zero-size arrays 会带有 nullptr
    // 参考：https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html#cuda-array-interface-version-3
    if (data_ptr != nullptr) {
      return {};
    } else {
      const auto current_device = at::detail::getCUDAHooks().current_device();
      return Device(
          kCUDA,
          static_cast<DeviceIndex>(current_device > -1 ? current_device : 0));
    }
  }();

  // 增加对象 obj 的引用计数
  Py_INCREF(obj);
  // 从给定的数据指针、尺寸、步幅、析构器、设备类型和目标设备创建 Torch 张量
  return at::from_blob(
      data_ptr,
      sizes,
      strides,
      [obj](void* data) {
        // 获取全局解释器锁
        pybind11::gil_scoped_acquire gil;
        // 减少对象 obj 的引用计数
        Py_DECREF(obj);
      },
      at::device(kCUDA).dtype(dtype),
      target_device);
}

// 仅在模块初始化期间变异一次；在此后行为如不可变变量一样。
bool numpy_with_dlpack_deleter_bug_installed = false;
// NumPy 在版本 1.22.0 中实现了对 Dlpack capsule 的支持。然而，初始实现未能正确处理在无 GIL（全局解释器锁）上下文中调用 `DLManagedTensor::deleter` 的情况。
// 直到 PyTorch 1.13.0 版本，我们在调用 deleter 时隐式地持有 GIL，但这在释放大张量的内存映射时会带来显著的性能开销。
// 从 PyTorch 1.13.0 开始，在释放之前的 `THPVariable_clear` 中释放 GIL，但这导致了 NumPy 中上述的 bug。
//
// NumPy 的 bug 应该在 1.24.0 版本中修复，但在 1.22.0 和 1.23.5 之间的所有版本都会导致内部断言失败，进而导致段错误。
// 为了解决这个问题，我们需要在检测到有问题的 NumPy 安装时有选择性地禁用此优化。
// 我们理想情况下希望将此“修复”限制在由 NumPy 衍生的 Dlpack-backed 张量上，但由于难以准确检测这些张量的来源，我们只能采用更一般化的方法。
//
// 参考：
//  https://github.com/pytorch/pytorch/issues/88082
//  https://github.com/pytorch/pytorch/issues/77139
//  https://github.com/numpy/numpy/issues/22507
void validate_numpy_for_dlpack_deleter_bug() {
  // 确保每个会话中只调用一次此函数。
  static bool validated = false;
  TORCH_INTERNAL_ASSERT(validated == false);
  validated = true;

  // 导入 NumPy 模块
  THPObjectPtr numpy_module(PyImport_ImportModule("numpy"));
  if (!numpy_module) {
    PyErr_Clear();
    return;
  }

  // 获取 NumPy 模块的版本属性
  THPObjectPtr version_attr(
      PyObject_GetAttrString(numpy_module.get(), "__version__"));
  if (!version_attr) {
    PyErr_Clear();
    return;
  }

  // 获取 NumPy 模块版本的 UTF-8 字符串表示
  Py_ssize_t version_utf8_size = 0;
  const char* version_utf8 =
      PyUnicode_AsUTF8AndSize(version_attr.get(), &version_utf8_size);
  if (!version_utf8_size) {
    PyErr_Clear();
    return;
  }
  std::string version(version_utf8, version_utf8_size);

  // 如果版本号长度小于 4，则直接返回
  if (version_utf8_size < 4)
    return;

  // 截取版本号的前四位
  std::string truncated_version(version.substr(0, 4));

  // 检查 NumPy 是否存在 Dlpack deleter 的 bug
  numpy_with_dlpack_deleter_bug_installed =
      truncated_version == "1.22" || truncated_version == "1.23";
}

// 检查 NumPy 是否安装了 Dlpack deleter 的 bug
bool is_numpy_dlpack_deleter_bugged() {
  return numpy_with_dlpack_deleter_bug_installed;
}
```