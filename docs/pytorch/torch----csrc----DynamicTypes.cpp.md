# `.\pytorch\torch\csrc\DynamicTypes.cpp`

```py
// 引入 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>

// 引入 Torch 的 C++ 头文件
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/cuda_enabled.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>

// 引入 ATen 库的头文件
#include <ATen/ATen.h>
#include <ATen/FunctionalStorageImpl.h>

// 引入标准库的头文件
#include <array>
#include <stdexcept>

// 定义 Torch 的命名空间
namespace torch {
// 匿名命名空间，用于定义内部静态数组
namespace {
// 用于存储 Torch 中的数据类型对象指针的静态数组，大小为 ScalarType 的选项数量
std::array<THPDtype*, static_cast<int>(at::ScalarType::NumOptions)>
    dtype_registry = {};

// 用于存储 Torch 中布局对象指针的静态数组，大小为 Layout 的选项数量
std::array<THPLayout*, static_cast<int>(at::Layout::NumOptions)>
    layout_registry = {};

} // namespace

// 注册给定的 dtype 对象到对应的 ScalarType
void registerDtypeObject(THPDtype* dtype, at::ScalarType scalarType) {
  dtype_registry[static_cast<int>(scalarType)] = dtype;
}

// 注册给定的布局对象到对应的 Layout
void registerLayoutObject(THPLayout* thp_layout, at::Layout layout) {
  layout_registry[static_cast<int>(layout)] = thp_layout;
}

// 根据 ScalarType 获取对应的 THPDtype 对象指针
THPDtype* getTHPDtype(at::ScalarType scalarType) {
  auto dtype = dtype_registry[static_cast<int>(scalarType)];
  if (!dtype) {
    throw std::invalid_argument("unsupported scalarType");
  }
  return dtype;
}

// 根据 Layout 获取对应的 THPLayout 对象指针
THPLayout* getTHPLayout(at::Layout layout) {
  auto thp_layout = layout_registry[static_cast<int>(layout)];
  if (!thp_layout) {
    throw std::invalid_argument("unsupported at::Layout");
  }
  return thp_layout;
}

// 创建 Python 对象，包装给定的 ATen Storage 对象
PyObject* createPyObject(const at::Storage& storage) {
  // 注释 [Invalid Python Storages]
  // 当用户创建一个 Python 张量包装子类时，该子类是一个张量对象，其存储为 nullptr。
  // 我们仍允许用户调用 `my_subclass.untyped_storage()`，并获得一个有效的存储对象
  // （这在从 Python 检测存储的别名信息时很有用）。然而，任何访问 data_ptr 的操作是不允许的，
  // 例如通过 `x.untyped_storage().data_ptr()` 方法。
  PyObject* obj = THPStorage_Wrap(storage);
  if (!obj)
    throw python_error();
  return obj;
}

// 加载 TypedStorage 类型的 Python 类对象
PyTypeObject* loadTypedStorageTypeObject() {
  PyObject* storage_module = PyImport_ImportModule("torch.storage");
  TORCH_INTERNAL_ASSERT(storage_module && PyModule_Check(storage_module));

  PyObject* typed_storage_obj =
      PyObject_GetAttrString(storage_module, "TypedStorage");
  TORCH_INTERNAL_ASSERT(typed_storage_obj && PyType_Check(typed_storage_obj));
  return reinterpret_cast<PyTypeObject*>(
      PyObject_GetAttrString(storage_module, "TypedStorage"));
}

// 获取 TypedStorage 类型的 Python 类对象
PyTypeObject* getTypedStorageTypeObject() {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static PyTypeObject* typed_storage_type_obj = loadTypedStorageTypeObject();
  return typed_storage_type_obj;
}

// 检查给定的 Python 对象是否为 Storage 对象
bool isStorage(PyObject* obj) {
  if (PyObject_TypeCheck(obj, getTypedStorageTypeObject())) {
    return true;
  }
  return THPStorage_Check(obj);
}

// 创建 Storage 对象、获取其类型和是否成功的元组
std::tuple<at::Storage, at::ScalarType, bool> createStorageGetType(
    PyObject* obj) {
  // 定义一个变量来存储标量类型，默认为Undefined
  at::ScalarType scalar_type = at::ScalarType::Undefined;
  // 检查对象是否为特定类型的存储对象
  bool is_typed_storage = PyObject_TypeCheck(obj, getTypedStorageTypeObject());
  // 未经类型化的存储对象的指针
  PyObject* untyped_storage_obj = nullptr;

  // 如果对象是特定类型的存储对象
  if (is_typed_storage) {
    // 获取对象的dtype属性，增加dtype和_untyped_storage的引用计数
    PyObject* dtype_obj = PyObject_GetAttrString(obj, "dtype");
    TORCH_INTERNAL_ASSERT(dtype_obj);
    // 断言dtype_obj是THPDtype类型的对象
    TORCH_INTERNAL_ASSERT(THPDtype_Check(dtype_obj));
    // 获取标量类型并赋值给scalar_type
    scalar_type = reinterpret_cast<THPDtype*>(dtype_obj)->scalar_type;
    // 减少dtype_obj的引用计数
    Py_DECREF(dtype_obj);

    // 获取对象的_untyped_storage属性，增加其引用计数
    untyped_storage_obj = PyObject_GetAttrString(obj, "_untyped_storage");
    TORCH_INTERNAL_ASSERT(untyped_storage_obj);
    // 减少_untyped_storage属性的引用计数
    Py_DECREF(untyped_storage_obj);

  } else {
    // 如果不是特定类型的存储对象，则默认标量类型为kByte，使用传入的对象作为未经类型化的存储对象
    scalar_type = at::kByte;
    untyped_storage_obj = obj;
  }

  // 断言untyped_storage_obj是THPStorage类型的对象
  TORCH_CHECK(
      THPStorage_Check(untyped_storage_obj),
      "not a storage '",
      Py_TYPE(obj)->tp_name,
      "'");
  
  // 解包THPStorage对象并返回存储、标量类型和是否为类型化存储的元组
  auto storage = THPStorage_Unpack(untyped_storage_obj);
  return std::make_tuple(storage, scalar_type, is_typed_storage);
}

// 结束 torch 命名空间定义

at::Storage createStorage(PyObject* obj) {
    // 调用 createStorageGetType 函数并获取返回的元组的第一个元素，返回作为 at::Storage 对象
    return std::get<0>(createStorageGetType(obj));
}

} // namespace torch
```