# `D:\src\scipysrc\scipy\scipy\ndimage\src\_ctest.c`

```
static void
_destructor(PyObject *obj)
{
    // 从 PyCapsule 中获取回调数据并释放内存
    void *callback_data = PyCapsule_GetContext(obj);
    PyMem_Free(callback_data);
}


static int
_filter1d(double *input_line, npy_intp input_length, double *output_line,
      npy_intp output_length, void *callback_data)
{
    // 获取滤波器大小
    npy_intp filter_size = *(npy_intp *)callback_data;

    // 对输出线进行滤波操作
    for (npy_intp i = 0; i < output_length; i++) {
        output_line[i] = 0;
        // 计算滤波结果
        for (npy_intp j = 0; j < filter_size; j++) {
            output_line[i] += input_line[i+j];
        }
        output_line[i] /= filter_size; // 求平均值
    }
    return 1;
}


static PyObject *
py_filter1d(PyObject *obj, PyObject *args)
{
    npy_intp *callback_data = NULL;
    PyObject *capsule = NULL;

    // 分配内存存储滤波器大小
    callback_data = PyMem_Malloc(sizeof(npy_intp));
    if (!callback_data) {
        PyErr_NoMemory();
        goto error;
    }
    // 解析输入参数获取滤波器大小
    if (!PyArg_ParseTuple(args, "n", callback_data)) goto error;

    // 创建 PyCapsule 包装 _filter1d 函数，并设置回调数据
    capsule = PyCapsule_New(_filter1d, NULL, _destructor);
    if (!capsule) goto error;
    if (PyCapsule_SetContext(capsule, callback_data) != 0) {
        Py_DECREF(capsule);
        goto error;
    }
    return capsule;

error:
    PyMem_Free(callback_data);
    return NULL;
}


static int
_filter2d(double *buffer, npy_intp filter_size, double *res,
      void *callback_data)
{
    // 获取权重数组
    double *weights = (double *)callback_data;

    // 对输入缓冲区进行二维滤波操作
    *res = 0;
    for (npy_intp i = 0; i < filter_size; i++) {
        *res += weights[i]*buffer[i];
    }
    return 1;
}


static PyObject *
py_filter2d(PyObject *obj, PyObject *args)
{
    Py_ssize_t i, size;
    double *callback_data = NULL;
    PyObject *seq = NULL, *item = NULL, *capsule = NULL;

    // 解析输入参数获取权重序列
    if (!PyArg_ParseTuple(args, "O", &seq)) goto error;

    // 获取序列长度
    size = PySequence_Length(seq);
    if (size == -1) goto error;
    // 分配内存存储权重数组
    callback_data = PyMem_Malloc(size*sizeof(double));
    if (!callback_data) {
        PyErr_NoMemory();
        goto error;
    }

    // 从序列中读取权重数据
    for (i = 0; i < size; i++) {
        item = PySequence_GetItem(seq, i);
        if (!item) {
            PyErr_SetString(PyExc_IndexError, "failed to get item");
            goto error;
        }
        callback_data[i] = PyFloat_AsDouble(item); // 转换为 double 类型
        Py_DECREF(item);
        item = NULL;
        if (PyErr_Occurred()) goto error;
    }

    // 创建 PyCapsule 包装 _filter2d 函数，并设置回调数据
    capsule = PyCapsule_New(_filter2d, NULL, _destructor);
    if (!capsule) goto error;
    if (PyCapsule_SetContext(capsule, callback_data) != 0) {
        Py_DECREF(capsule);
        goto error;
    }
    return capsule;

error:
    PyMem_Free(callback_data);
    return NULL;
}


static int
_transform(npy_intp *output_coordinates, double *input_coordinates,
       npy_intp output_rank, npy_intp input_rank, void *callback_data)
{
    // 获取偏移量
    double shift = *(double *)callback_data;

    // 根据输出坐标和偏移量计算输入坐标
    for (npy_intp i = 0; i < input_rank; i++) {
        input_coordinates[i] = output_coordinates[i] - shift;
    }
    return 1;
}


static PyObject *
py_transform(PyObject *obj, PyObject *args)
{
    // 分配内存存储偏移量
    double *callback_data = PyMem_Malloc(sizeof(double));
    PyObject *capsule = NULL;

    // 创建 PyCapsule 包装 _transform 函数，并设置回调数据
    capsule = PyCapsule_New(_transform, NULL, _destructor);
    if (!capsule) goto error;
    if (PyCapsule_SetContext(capsule, callback_data) != 0) {
        Py_DECREF(capsule);
        goto error;
    }
    return capsule;

error:
    PyMem_Free(callback_data);
    return NULL;
}
    # 如果 callback_data 为空指针，则设置内存不足错误，并跳转到 error 标签处处理错误
    if (!callback_data) {
        PyErr_NoMemory();
        goto error;
    }
    # 使用 PyArg_ParseTuple 解析传入参数 args，将其解析为一个双精度浮点数并存储在 callback_data 中，如果解析失败则跳转到 error 标签处处理错误
    if (!PyArg_ParseTuple(args, "d", callback_data)) goto error;

    # 创建一个 PyCapsule 对象，封装 _transform 函数，不使用上下文信息，使用 _destructor 函数释放资源，如果创建失败则跳转到 error 标签处处理错误
    capsule = PyCapsule_New(_transform, NULL, _destructor);
    if (!capsule) goto error;
    # 将 callback_data 设置为 PyCapsule 对象的上下文信息，如果设置失败则释放 PyCapsule 对象并跳转到 error 标签处处理错误
    if (PyCapsule_SetContext(capsule, callback_data) != 0) {
        Py_DECREF(capsule);
        goto error;
    }
    # 返回创建的 PyCapsule 对象
    return capsule;

error:
    # 释放 callback_data 分配的内存
    PyMem_Free(callback_data);
    # 返回空指针，表示发生错误
    return NULL;
}

static PyMethodDef _CTestMethods[] = {
    {"transform", (PyCFunction)py_transform, METH_VARARGS, ""},
    {"filter1d", (PyCFunction)py_filter1d, METH_VARARGS, ""},
    {"filter2d", (PyCFunction)py_filter2d, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


// 定义一个静态数组 _CTestMethods，用于存储模块中的方法信息
// 每个条目包括方法名、对应的C函数指针、方法调用约定和空字符串的文档字符串
static PyMethodDef _CTestMethods[] = {
    {"transform", (PyCFunction)py_transform, METH_VARARGS, ""},
    {"filter1d", (PyCFunction)py_filter1d, METH_VARARGS, ""},
    {"filter2d", (PyCFunction)py_filter2d, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}  // 数组的结尾，以NULL作为标志
};



/* Initialize the module */
static struct PyModuleDef _ctest = {
    PyModuleDef_HEAD_INIT,
    "_ctest",
    NULL,
    -1,
    _CTestMethods,
    NULL,
    NULL,
    NULL,
    NULL
};


// 定义一个静态 PyModuleDef 结构 _ctest，用于描述 Python 模块的信息
// 包括模块定义头的初始化、模块名、模块文档字符串、模块状态（-1表示全局状态）、
// 模块方法列表、槽函数创建实例、模块清理函数、模块的内存管理操作和遇到错误时的处理函数
static struct PyModuleDef _ctest = {
    PyModuleDef_HEAD_INIT,  // 使用宏初始化模块定义头
    "_ctest",               // 模块名为 "_ctest"
    NULL,                   // 模块的文档字符串为空
    -1,                     // 全局状态为 -1，表示一个全局模块
    _CTestMethods,          // 模块包含的方法列表
    NULL,                   // 槽函数创建实例为空
    NULL,                   // 模块清理函数为空
    NULL,                   // 模块的内存管理操作为空
    NULL                    // 遇到错误时的处理函数为空
};



PyMODINIT_FUNC
PyInit__ctest(void)
{
    return PyModule_Create(&_ctest);
}


// 模块初始化函数 PyInit__ctest
// 创建并返回一个 Python 模块对象，模块的信息由之前定义的 _ctest 结构提供
PyMODINIT_FUNC
PyInit__ctest(void)
{
    return PyModule_Create(&_ctest);
}
```