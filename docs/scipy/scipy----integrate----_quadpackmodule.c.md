# `D:\src\scipysrc\scipy\scipy\integrate\_quadpackmodule.c`

```
/*
  从 Multipack 项目中引入
 */
#include "__quadpack.h"

// 定义 quadpack 模块的方法列表，每个方法包含方法名、对应的 C 函数、参数类型、文档字符串
static struct PyMethodDef quadpack_module_methods[] = {
    {"_qagse", quadpack_qagse, METH_VARARGS, doc_qagse},
    {"_qagie", quadpack_qagie, METH_VARARGS, doc_qagie},
    {"_qagpe", quadpack_qagpe, METH_VARARGS, doc_qagpe},
    {"_qawoe", quadpack_qawoe, METH_VARARGS, doc_qawoe},
    {"_qawfe", quadpack_qawfe, METH_VARARGS, doc_qawfe},
    {"_qawse", quadpack_qawse, METH_VARARGS, doc_qawse},
    {"_qawce", quadpack_qawce, METH_VARARGS, doc_qawce},
    {NULL, NULL, 0, NULL}  // 结束方法列表的标志
};

// 定义 quadpack 模块的模块定义结构体，包括模块名、文档字符串、方法列表、全局变量等
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  // 必须的初始化宏
    "_quadpack",  // 模块名
    NULL,  // 模块的文档字符串
    -1,  // 模块状态，-1 表示使用全局解释器锁
    quadpack_module_methods,  // 模块的方法列表
    NULL,
    NULL,
    NULL,
    NULL
};

// Python 解释器调用的初始化函数，返回一个新创建的 Python 模块对象
PyMODINIT_FUNC
PyInit__quadpack(void)
{
    PyObject *module, *mdict;

    // 导入 NumPy 的 C API
    import_array();

    // 创建一个新的 Python 模块对象
    module = PyModule_Create(&moduledef);
    if (module == NULL) {
        return NULL;
    }

    // 获取模块的字典对象
    mdict = PyModule_GetDict(module);
    if (mdict == NULL) {
        return NULL;
    }

    // 创建一个新的 _quadpack.error 异常对象
    quadpack_error = PyErr_NewException("_quadpack.error", NULL, NULL);
    if (quadpack_error == NULL) {
        return NULL;
    }
    // 将异常对象添加到模块的字典中
    if (PyDict_SetItemString(mdict, "error", quadpack_error)) {
        return NULL;
    }

    // 返回 Python 模块对象
    return module;
}
```