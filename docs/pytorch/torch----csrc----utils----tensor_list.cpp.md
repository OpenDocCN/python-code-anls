# `.\pytorch\torch\csrc\utils\tensor_list.cpp`

```
// 包含头文件 <torch/csrc/utils/tensor_list.h>
#include <torch/csrc/utils/tensor_list.h>

// 包含以下标准库和第三方库头文件
#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_scalars.h>

// 使用命名空间 at
using namespace at;

// 声明了命名空间 torch::utils
namespace torch::utils {

// 定义了静态函数 PyObject* recursive_to_list
static PyObject* recursive_to_list(
    const char* data,               // 数据指针，指向内存数据
    IntArrayRef sizes,              // 大小数组的引用
    IntArrayRef strides,            // 步幅数组的引用
    int64_t dim,                    // 当前处理的维度
    ScalarType scalarType,          // 标量类型
    size_t elementSize) {           // 元素大小
  // 确定数据的维度数
  int64_t ndim = static_cast<int64_t>(sizes.size());
  // 如果当前维度与数据的维度数相同，返回标量值
  if (dim == ndim) {
    return torch::utils::load_scalar(data, scalarType);
  }
  // 否则创建一个长度为 n 的 Python 列表对象
  auto n = sizes[dim];
  auto list = THPObjectPtr(PyList_New(n));
  // 如果创建失败，则抛出 Python 异常
  if (!list)
    throw python_error();
  // 遍历列表的每个元素
  for (const auto i : c10::irange(n)) {
    // 递归调用 recursive_to_list 函数
    PyObject* obj = recursive_to_list(
        data, sizes, strides, dim + 1, scalarType, elementSize);
    // 如果递归调用失败，则抛出 Python 异常
    if (!obj)
      throw python_error();
    // 将返回的对象设置为列表的第 i 个元素
    PyList_SET_ITEM(list.get(), i, obj);
    // 计算数据指针的偏移量
    auto advance_data_ptr = strides[dim] * elementSize;
    // 检查数据指针是否有效
    TORCH_INTERNAL_ASSERT(data || (advance_data_ptr == 0));
    // 更新数据指针位置
    data += advance_data_ptr;
  }
  // 返回 Python 列表对象
  return list.release();
}

// 定义了函数 PyObject* tensor_to_list，将张量转换为 Python 列表对象
PyObject* tensor_to_list(const Tensor& tensor) {
  {
    // 转换张量为 Python 对象
    py::object pytensor =
        py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));
    // 检查张量是否为 Python 派发的子类，如果是则抛出异常
    TORCH_CHECK(
        !tensor.unsafeGetTensorImpl()->is_python_dispatch(),
        ".tolist() is not supported for tensor subclasses, got ",
        Py_TYPE(pytensor.ptr())->tp_name);
  }
  // 解析共轭和负数的影响后的张量
  Tensor data = tensor.resolve_conj().resolve_neg();
  // 如果数据不在 CPU 上，则转换到 CPU 上处理
  if (!data.device().is_cpu()) {
    pybind11::gil_scoped_release no_gil;
    data = data.toBackend(Backend::CPU);
  }
  // 检查张量是否分配了存储空间，或者张量元素数量为零
  TORCH_CHECK(
      tensor.numel() == 0 || data.const_data_ptr(),
      "tolist() shouldn't be called on a tensor with unallocated storage");
  // 调用 recursive_to_list 函数将张量转换为 Python 列表对象并返回
  return recursive_to_list(
      (const char*)data.const_data_ptr(), // 数据指针转为 char*
      data.sizes(),                       // 张量的尺寸
      data.strides(),                     // 张量的步幅
      0,                                  // 初始维度为 0
      data.scalar_type(),                 // 张量的标量类型
      tensor.numel() == 0 ? 0 : data.dtype().itemsize()); // 元素大小
}

} // 命名空间 torch::utils
```