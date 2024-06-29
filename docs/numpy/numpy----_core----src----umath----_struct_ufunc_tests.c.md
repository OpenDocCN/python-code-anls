# `.\numpy\numpy\_core\src\umath\_struct_ufunc_tests.c`

```py
/*
 * struct_ufunc_test.c
 * This is the C code for creating your own
 * NumPy ufunc for a structured array dtype.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

// 定义一个静态函数，用于将三个 uint64 类型的数组相加
static void add_uint64_triplet(char **args,
                               npy_intp const *dimensions,
                               npy_intp const* steps,
                               void* data)
{
    npy_intp i;
    npy_intp is1=steps[0];  // 第一个输入数组的步长
    npy_intp is2=steps[1];  // 第二个输入数组的步长
    npy_intp os=steps[2];   // 输出数组的步长
    npy_intp n=dimensions[0];  // 数组的维度大小
    npy_uint64 *x, *y, *z;   // 定义指向输入和输出数组的指针

    char *i1=args[0];   // 第一个输入数组的起始地址
    char *i2=args[1];   // 第二个输入数组的起始地址
    char *op=args[2];   // 输出数组的起始地址

    for (i = 0; i < n; i++) {
        // 将输入数组的地址强制转换为 uint64 类型的指针
        x = (npy_uint64*)i1;
        y = (npy_uint64*)i2;
        z = (npy_uint64*)op;

        // 执行相加操作，并将结果存入输出数组
        z[0] = x[0] + y[0];
        z[1] = x[1] + y[1];
        z[2] = x[2] + y[2];

        // 更新输入和输出数组的地址，移动到下一个元素
        i1 += is1;
        i2 += is2;
        op += os;
    }
}

// 定义一个 Python 函数，用于注册自定义的 ufunc
static PyObject*
register_fail(PyObject* NPY_UNUSED(self), PyObject* NPY_UNUSED(args))
{
    PyObject *add_triplet;   // 定义一个 PyObject 类型的对象，用于存储 ufunc
    PyObject *dtype_dict;    // 用于存储结构化数据类型的字典对象
    PyArray_Descr *dtype;    // 用于存储结构化数据类型的描述符
    PyArray_Descr *dtypes[3];  // 定义一个数组，存储三个结构化数据类型的描述符
    int retval;              // 用于存储函数调用返回值的变量

    // 创建一个 ufunc 对象，名称为 "add_triplet"，无文档字符串
    add_triplet = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 2, 1,
                                    PyUFunc_None, "add_triplet",
                                    "add_triplet_docstring", 0);

    // 创建一个包含结构化数据类型信息的字典对象
    dtype_dict = Py_BuildValue("[(s, s), (s, s), (s, s)]",
                               "f0", "u8", "f1", "u8", "f2", "u8");
    // 将 dtype_dict 转换为 PyArray_Descr 结构，并存储在 dtype 中
    PyArray_DescrConverter(dtype_dict, &dtype);
    // 释放 dtype_dict 占用的内存
    Py_DECREF(dtype_dict);

    // 将 dtype 分别赋值给 dtypes 数组中的三个元素
    dtypes[0] = dtype;
    dtypes[1] = dtype;
    dtypes[2] = dtype;

    // 将 add_uint64_triplet 函数注册到 add_triplet ufunc 中
    retval = PyUFunc_RegisterLoopForDescr((PyUFuncObject *)add_triplet,
                                dtype,
                                &add_uint64_triplet,
                                dtypes,
                                NULL);

    // 如果注册失败，则释放内存并返回 NULL
    if (retval < 0) {
        Py_DECREF(add_triplet);
        Py_DECREF(dtype);
        return NULL;
    }

    // 再次尝试注册，以确保成功
    retval = PyUFunc_RegisterLoopForDescr((PyUFuncObject *)add_triplet,
                                dtype,
                                &add_uint64_triplet,
                                dtypes,
                                NULL);
    // 释放内存
    Py_DECREF(add_triplet);
    Py_DECREF(dtype);
    // 如果再次注册失败，则返回 NULL
    if (retval < 0) {
        return NULL;
    }
    // 注册成功，返回 Py_None
    Py_RETURN_NONE;
}

// 定义模块中的方法列表
static PyMethodDef StructUfuncTestMethods[] = {
    {"register_fail",
        register_fail,
        METH_NOARGS, NULL},  // 注册失败时调用的方法
    {NULL, NULL, 0, NULL}   // 方法列表结束
};

// 定义模块的结构体
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,   // 模块定义的头部初始化
    "_struct_ufunc_tests",   // 模块名称
    NULL,                    // 模块文档字符串
    -1,                      // 模块状态
    StructUfuncTestMethods,  // 模块中定义的方法列表
    NULL,                    // 模块的全局状态
    NULL,                    // 模块中的内存分配函数
    NULL,                    // 模块中的内存释放函数
    NULL                     // 模块中的状态清理函数
};

// 模块初始化函数
PyMODINIT_FUNC PyInit__struct_ufunc_tests(void)
{
    PyObject *m, *add_triplet, *d;

    // 创建一个新模块对象
    m = PyModule_Create(&moduledef);
    // 返回创建的模块对象
    return m;
}
    // 声明 PyObject 类型的变量 dtype_dict，用于存储描述符字典
    PyObject *dtype_dict;
    // 声明 PyArray_Descr 类型的变量 dtype，用于存储数组描述符
    PyArray_Descr *dtype;
    // 声明 PyArray_Descr 类型的数组 dtypes，用于存储多个数组描述符
    PyArray_Descr *dtypes[3];

    // 创建 Python 模块对象 m，使用给定的 moduledef 结构体
    m = PyModule_Create(&moduledef);

    // 如果创建模块失败，返回 NULL
    if (m == NULL) {
        return NULL;
    }

    // 导入 NumPy 的数组支持模块
    import_array();
    // 导入 NumPy 的数学函数支持模块
    import_umath();

    // 创建一个通用函数对象 add_triplet，用于执行三个输入操作数的特定功能
    add_triplet = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 2, 1,
                                    PyUFunc_None, "add_triplet",
                                    "add_triplet_docstring", 0);

    // 使用 Py_BuildValue 函数创建一个 Python 字典对象 dtype_dict，包含三个键值对
    dtype_dict = Py_BuildValue("[(s, s), (s, s), (s, s)]",
                               "f0", "u8", "f1", "u8", "f2", "u8");
    // 将 dtype_dict 转换为 PyArray_Descr 结构体，存储在 dtype 中
    PyArray_DescrConverter(dtype_dict, &dtype);
    // 释放 dtype_dict 对象的引用计数
    Py_DECREF(dtype_dict);

    // 将同一个 dtype 描述符分配给数组 dtypes 的所有三个元素
    dtypes[0] = dtype;
    dtypes[1] = dtype;
    dtypes[2] = dtype;

    // 将描述符 dtype 注册到通用函数 add_triplet 中，并指定用于每个操作数的描述符数组 dtypes
    PyUFunc_RegisterLoopForDescr((PyUFuncObject *)add_triplet,
                                dtype,
                                &add_uint64_triplet,
                                dtypes,
                                NULL);

    // 释放描述符 dtype 对象的引用计数
    Py_DECREF(dtype);
    // 获取模块 m 的字典对象
    d = PyModule_GetDict(m);

    // 将 add_triplet 对象添加到模块字典 d 中，键为 "add_triplet"
    PyDict_SetItemString(d, "add_triplet", add_triplet);
    // 释放 add_triplet 对象的引用计数
    Py_DECREF(add_triplet);
    // 返回创建的 Python 模块对象 m
    return m;
}


注释：


# 这是一个单独的右括号 '}'，用于结束一个代码块或数据结构的定义。
```