# `.\numpy\numpy\_core\src\common\npy_ctypes.h`

```
/*
 * 检查一个 Python 类型是否是 ctypes 类型。
 *
 * 类似于 Py<type>_Check 函数，如果参数看起来像一个 ctypes 对象，则返回 true。
 *
 * 这个整个函数只是围绕同名的 Python 函数的一个包装器。
 */
static inline int
npy_ctypes_check(PyTypeObject *obj)
{
    PyObject *ret_obj;  // 用于存储函数调用的返回对象
    int ret;  // 存储最终的返回结果

    // 导入并缓存 numpy._core._internal 模块中的 npy_ctypes_check 函数
    npy_cache_import("numpy._core._internal", "npy_ctypes_check",
                     &npy_thread_unsafe_state.npy_ctypes_check);
    if (npy_thread_unsafe_state.npy_ctypes_check == NULL) {
        goto fail;  // 如果导入失败，则跳转到错误处理
    }

    // 调用 npy_ctypes_check 函数来检查是否是 ctypes 类型
    ret_obj = PyObject_CallFunctionObjArgs(npy_thread_unsafe_state.npy_ctypes_check,
                                           (PyObject *)obj, NULL);
    if (ret_obj == NULL) {
        goto fail;  // 如果调用失败，则跳转到错误处理
    }

    // 将返回对象转换为布尔值并赋给 ret
    ret = PyObject_IsTrue(ret_obj);
    Py_DECREF(ret_obj);  // 减少返回对象的引用计数
    if (ret == -1) {
        goto fail;  // 如果转换失败，则跳转到错误处理
    }

    return ret;  // 返回最终的检查结果

fail:
    /* 如果上述步骤失败，则假设该类型不是 ctypes 类型 */
    PyErr_Clear();  // 清除错误信息
    return 0;  // 返回 false
}
```