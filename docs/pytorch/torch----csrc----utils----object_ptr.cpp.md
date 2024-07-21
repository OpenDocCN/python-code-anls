# `.\pytorch\torch\csrc\utils\object_ptr.cpp`

```
// 定义模板特化，用于释放 PyObject 指针的资源
template <>
void THPPointer<PyObject>::free() {
  // 检查指针是否非空且 Python 解释器已初始化，然后减少引用计数
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

// 实例化 PyObject 的模板类
template class THPPointer<PyObject>;

// 定义模板特化，用于释放 PyCodeObject 指针的资源
template <>
void THPPointer<PyCodeObject>::free() {
  // 检查指针是否非空且 Python 解释器已初始化，然后减少引用计数
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

// 实例化 PyCodeObject 的模板类
template class THPPointer<PyCodeObject>;

// 定义模板特化，用于释放 PyFrameObject 指针的资源
template <>
void THPPointer<PyFrameObject>::free() {
  // 检查指针是否非空且 Python 解释器已初始化，然后减少引用计数
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

// 实例化 PyFrameObject 的模板类
template class THPPointer<PyFrameObject>;
```