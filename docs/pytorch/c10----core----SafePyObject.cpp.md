# `.\pytorch\c10\core\SafePyObject.cpp`

```
#include <c10/core/SafePyObject.h>
// 引入 SafePyObject 类的头文件

namespace c10 {
// 进入 c10 命名空间

PyObject* SafePyObject::ptr(const c10::impl::PyInterpreter* interpreter) const {
    // 定义 SafePyObject 类的成员函数 ptr，返回一个 PyObject 指针
    TORCH_INTERNAL_ASSERT(interpreter == pyinterpreter_);
    // 使用内部断言确保传入的解释器与存储的解释器匹配
    return data_;
    // 返回存储的 PyObject 数据指针
}

PyObject* SafePyHandle::ptr(const c10::impl::PyInterpreter* interpreter) const {
    // 定义 SafePyHandle 类的成员函数 ptr，返回一个 PyObject 指针
    TORCH_INTERNAL_ASSERT(interpreter == pyinterpreter_);
    // 使用内部断言确保传入的解释器与存储的解释器匹配
    return data_;
    // 返回存储的 PyObject 数据指针
}

} // namespace c10
// 退出 c10 命名空间
```