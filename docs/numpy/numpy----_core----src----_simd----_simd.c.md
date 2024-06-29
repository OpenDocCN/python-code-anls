# `.\numpy\numpy\_core\src\_simd\_simd.c`

```py
#include "_simd.h"

#include "numpy/npy_math.h"

// 定义函数：获取当前浮点状态的 Python 对象
static PyObject *
get_floatstatus(PyObject* NPY_UNUSED(self), PyObject *NPY_UNUSED(args))
{
    // 调用 numpy 库函数获取当前浮点状态并返回相应的 Python 对象
    return PyLong_FromLong(npy_get_floatstatus());
}

// 定义函数：清除当前浮点状态
static PyObject *
clear_floatstatus(PyObject* NPY_UNUSED(self), PyObject *NPY_UNUSED(args))
{
    // 调用 numpy 库函数清除当前浮点状态
    npy_clear_floatstatus();
    // 直接返回 None 对象
    Py_RETURN_NONE;
}

// 定义模块方法列表
static PyMethodDef _simd_methods[] = {
    {"get_floatstatus", get_floatstatus, METH_NOARGS, NULL},  // 注册获取浮点状态函数
    {"clear_floatstatus", clear_floatstatus, METH_NOARGS, NULL},  // 注册清除浮点状态函数
    {NULL, NULL, 0, NULL}  // 结束方法列表的标记
};

// 模块初始化函数
PyMODINIT_FUNC PyInit__simd(void)
{
    // 定义模块结构体
    static struct PyModuleDef defs = {
        .m_base = PyModuleDef_HEAD_INIT,  // 设置模块结构的基本属性
        .m_name = "numpy._core._simd",  // 模块名称
        .m_size = -1,  // 模块大小
        .m_methods = _simd_methods  // 模块方法列表
    };

    // 初始化 CPU 相关功能，若失败则返回 NULL
    if (npy_cpu_init() < 0) {
        return NULL;
    }

    // 创建 Python 模块对象，如果失败则返回 NULL
    PyObject *m = PyModule_Create(&defs);
    if (m == NULL) {
        return NULL;
    }

    // 创建一个空的字典对象用于存储目标特性
    PyObject *targets = PyDict_New();
    if (targets == NULL) {
        goto err;  // 如果创建失败则跳转到错误处理
    }

    // 将 targets 字典对象添加到模块中，如果失败则释放 targets 并跳转到错误处理
    if (PyModule_AddObject(m, "targets", targets) < 0) {
        Py_DECREF(targets);
        goto err;
    }

    // 宏定义，用于根据特性测试情况添加模块
    #define ATTACH_MODULE(TESTED_FEATURES, TARGET_NAME, MAKE_MSVC_HAPPY)       \
        {                                                                      \
            PyObject *simd_mod;                                                \
            if (!TESTED_FEATURES) {                                            \
                Py_INCREF(Py_None);                                            \
                simd_mod = Py_None;                                            \
            } else {                                                           \
                simd_mod = NPY_CAT(simd_create_module_, TARGET_NAME)();        \
                if (simd_mod == NULL) {                                        \
                    goto err;                                                  \
                }                                                              \
            }                                                                  \
            const char *target_name = NPY_TOSTRING(TARGET_NAME);               \
            if (PyDict_SetItemString(targets, target_name, simd_mod) < 0) {    \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
            Py_INCREF(simd_mod);                                               \
            if (PyModule_AddObject(m, target_name, simd_mod) < 0) {            \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
        }

    // ATTACH_MODULE 宏的作用是根据特性情况添加模块对象到 targets 字典和模块中

    // 返回创建的模块对象
    return m;

// 错误处理代码块
err:
    // 在错误处理时释放所有已分配的资源，并返回 NULL
    Py_XDECREF(m);
    return NULL;
}
    #define ATTACH_BASELINE_MODULE(MAKE_MSVC_HAPPY)                            \
        {                                                                      \
            // 创建名为 baseline 的 Python 模块对象
            PyObject *simd_mod = simd_create_module();                         \
            // 如果创建失败，则跳转到错误处理标签 err
            if (simd_mod == NULL) {                                            \
                goto err;                                                      \
            }                                                                  \
            // 将 baseline 模块对象添加到 targets 字典中，如果失败则跳转到 err
            if (PyDict_SetItemString(targets, "baseline", simd_mod) < 0) {     \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
            // 增加 baseline 模块对象的引用计数
            Py_INCREF(simd_mod);                                               \
            // 将 baseline 模块对象添加到 m 模块中，如果失败则跳转到 err
            if (PyModule_AddObject(m, "baseline", simd_mod) < 0) {             \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
        }
    #ifdef NPY__CPU_MESON_BUILD
        // 使用 NPY_MTARGETS_CONF_DISPATCH 宏调用 ATTACH_MODULE 宏和 MAKE_MSVC_HAPPY 宏
        NPY_MTARGETS_CONF_DISPATCH(NPY_CPU_HAVE, ATTACH_MODULE, MAKE_MSVC_HAPPY)
        // 使用 NPY_MTARGETS_CONF_BASELINE 宏调用 ATTACH_BASELINE_MODULE 宏和 MAKE_MSVC_HAPPY 宏
        NPY_MTARGETS_CONF_BASELINE(ATTACH_BASELINE_MODULE, MAKE_MSVC_HAPPY)
    #else
        // 使用 NPY__CPU_DISPATCH_CALL 宏调用 ATTACH_MODULE 宏和 MAKE_MSVC_HAPPY 宏
        NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, ATTACH_MODULE, MAKE_MSVC_HAPPY)
        // 使用 NPY__CPU_DISPATCH_BASELINE_CALL 宏调用 ATTACH_BASELINE_MODULE 宏和 MAKE_MSVC_HAPPY 宏
        NPY__CPU_DISPATCH_BASELINE_CALL(ATTACH_BASELINE_MODULE, MAKE_MSVC_HAPPY)
    #endif
    // 返回模块对象 m
    return m;
err:
    # 减少 Python 对象 m 的引用计数，释放其占用的内存
    Py_DECREF(m);
    # 返回 NULL 表示函数执行失败
    return NULL;
}
```