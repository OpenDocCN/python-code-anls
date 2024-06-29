# `.\numpy\numpy\_core\src\common\npy_cpu_dispatch.c`

```py
// 定义宏，禁用过时的 NumPy API，并设置为当前 API 版本
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义宏，标识该文件属于多维数组模块
#define _MULTIARRAYMODULE

// 包含必要的头文件
#include "npy_cpu_dispatch.h"
#include "numpy/ndarraytypes.h"
#include "npy_static_data.h"

// 初始化 CPU 分发追踪器
NPY_VISIBILITY_HIDDEN int
npy_cpu_dispatch_tracer_init(PyObject *mod)
{
    // 如果 CPU 分发注册表已经初始化，则抛出运行时错误
    if (npy_static_pydata.cpu_dispatch_registry != NULL) {
        PyErr_Format(PyExc_RuntimeError, "CPU dispatcher tracer already initlized");
        return -1;
    }
    
    // 获取模块的字典
    PyObject *mod_dict = PyModule_GetDict(mod);
    if (mod_dict == NULL) {
        return -1;
    }
    
    // 创建一个新的字典作为注册表
    PyObject *reg_dict = PyDict_New();
    if (reg_dict == NULL) {
        return -1;
    }
    
    // 将注册表添加到模块字典中
    int err = PyDict_SetItemString(mod_dict, "__cpu_targets_info__", reg_dict);
    Py_DECREF(reg_dict);  // 减少字典的引用计数
    if (err != 0) {
        return -1;
    }
    
    // 将注册表赋给静态数据结构中的 CPU 分发注册表
    npy_static_pydata.cpu_dispatch_registry = reg_dict;
    return 0;
}

// CPU 分发追踪函数
NPY_VISIBILITY_HIDDEN void
npy_cpu_dispatch_trace(const char *fname, const char *signature,
                       const char **dispatch_info)
{
    // 获取函数名对应的函数字典
    PyObject *func_dict = PyDict_GetItemString(npy_static_pydata.cpu_dispatch_registry, fname);
    if (func_dict == NULL) {
        // 如果函数字典不存在，则创建一个新的函数字典
        func_dict = PyDict_New();
        if (func_dict == NULL) {
            return;
        }
        // 将新创建的函数字典添加到注册表中
        int err = PyDict_SetItemString(npy_static_pydata.cpu_dispatch_registry, fname, func_dict);
        Py_DECREF(func_dict);  // 减少函数字典的引用计数
        if (err != 0) {
            return;
        }
    }
    
    // 为每个签名创建目标信息的字典
    PyObject *sig_dict = PyDict_New();
    if (sig_dict == NULL) {
        return;
    }
    // 将签名信息字典添加到函数字典中
    int err = PyDict_SetItemString(func_dict, signature, sig_dict);
    Py_DECREF(sig_dict);  // 减少签名信息字典的引用计数
    if (err != 0) {
        return;
    }
    
    // 添加当前调度的目标到签名信息字典中
    PyObject *current_target = PyUnicode_FromString(dispatch_info[0]);
    if (current_target == NULL) {
        return;
    }
    err = PyDict_SetItemString(sig_dict, "current", current_target);
    Py_DECREF(current_target);  // 减少当前目标字符串的引用计数
    if (err != 0) {
        return;
    }
    
    // 添加可用目标信息到签名信息字典中
    PyObject *available = PyUnicode_FromString(dispatch_info[1]);
    if (available == NULL) {
        return;
    }
    err = PyDict_SetItemString(sig_dict, "available", available);
    Py_DECREF(available);  // 减少可用目标字符串的引用计数
    if (err != 0) {
        return;
    }
}
```