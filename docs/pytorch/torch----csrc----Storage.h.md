# `.\pytorch\torch\csrc\Storage.h`

```
#ifndef THP_STORAGE_INC
#define THP_STORAGE_INC

// 包含必要的头文件
#include <Python.h>
#include <c10/core/Storage.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/Types.h>

// 定义宏，用于字符串表示 Torch 的未类型化存储
#define THPStorageStr "torch.UntypedStorage"

// THPStorage 结构体，表示 Python 中的 Torch 存储对象
struct THPStorage {
  PyObject_HEAD; // Python 对象头部

  c10::MaybeOwned<c10::Storage> cdata; // Torch 的存储对象，可能是所有权对象
  bool is_hermetic; // 表示存储是否封闭的布尔值
};

// 声明函数 THPStorage_Wrap，用于将 C++ 中的 Storage 对象包装成 Python 对象
TORCH_PYTHON_API PyObject* THPStorage_Wrap(c10::Storage storage);

// 声明函数 THPStorage_NewWithStorage，用于创建新的 THPStorage 对象，关联给定的 Storage
TORCH_PYTHON_API PyObject* THPStorage_NewWithStorage(
    PyTypeObject* type,
    c10::Storage _storage,
    c10::impl::PyInterpreterStatus status,
    bool allow_preexisting_pyobj = false);

// 声明 THPStorageClass，表示 THPStorage 对象的 Python 类型
extern PyTypeObject* THPStorageClass;

// 内联函数，检查给定的 PyTypeObject 是否精确匹配 THPStorageClass
inline bool THPStorage_CheckTypeExact(PyTypeObject* tp) {
  return tp == THPStorageClass;
}

// 内联函数，检查给定的 PyObject 是否精确匹配 THPStorageClass
inline bool THPStorage_CheckExact(PyObject* obj) {
  return THPStorage_CheckTypeExact(Py_TYPE(obj));
}

// 内联函数，检查给定的 PyObject 是否为 THPStorageClass 或其子类的实例
inline bool THPStorage_Check(PyObject* obj) {
  if (!THPStorageClass)
    return false;

  // 使用 PyObject_IsInstance 检查实例
  const auto result = PyObject_IsInstance(obj, (PyObject*)THPStorageClass);
  if (result == -1)
    throw python_error(); // 如果检查失败，抛出异常
  return result; // 返回检查结果
}

// 初始化函数声明，初始化 THPStorage 模块
bool THPStorage_init(PyObject* module);

// 后期初始化函数声明，处理 THPStorage 模块的后续初始化
void THPStorage_postInit(PyObject* module);

// 断言函数声明，用于确保 THPStorage 对象非空
void THPStorage_assertNotNull(THPStorage* storage);

// 断言函数声明，用于确保 PyObject 是非空的 THPStorage 对象
void THPStorage_assertNotNull(PyObject* obj);

// THPStorageType 的声明，表示 THPStorage 在 Python 中的类型对象
extern PyTypeObject THPStorageType;

// 内联函数，用于从 THPStorage 中提取 c10::Storage 引用
inline const c10::Storage& THPStorage_Unpack(THPStorage* storage) {
  return *storage->cdata;
}

// 内联函数，用于从 PyObject 中提取 c10::Storage 引用
inline const c10::Storage& THPStorage_Unpack(PyObject* obj) {
  return THPStorage_Unpack(reinterpret_cast<THPStorage*>(obj));
}

#endif // THP_STORAGE_INC
```