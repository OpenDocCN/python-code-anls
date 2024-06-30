# `D:\src\scipysrc\scipy\scipy\_lib\_uarray\vectorcall.cxx`

```
#ifdef PYPY_VERSION
/* 如果是在 PyPy 环境下编译此代码 */

/* 创建参数元组，用于转换为 PyObject_Call */
static PyObject * build_arg_tuple(PyObject * const * args, Py_ssize_t nargs) {
    PyObject * tuple = PyTuple_New(nargs);
    if (!tuple) {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < nargs; ++i) {
        Py_INCREF(args[i]); /* SET_ITEM steals a reference */
        PyTuple_SET_ITEM(tuple, i, args[i]);
    }
    return tuple;
}

/* 创建关键字参数字典，用于转换为 PyObject_Call */
static PyObject * build_kwarg_dict(
    PyObject * const * args, PyObject * names, Py_ssize_t nargs) {
    PyObject * dict = PyDict_New();
    if (!dict) {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < nargs; ++i) {
        PyObject * key = PyTuple_GET_ITEM(names, i);
        int success = PyDict_SetItem(dict, key, args[i]);
        if (success == -1) {
            Py_DECREF(dict);
            return NULL;
        }
    }
    return dict;
}
#endif /* PYPY_VERSION */


/* 计算 Vectorcall 的参数个数 */
Py_ssize_t Q_PyVectorcall_NARGS(size_t n) {
    return n & (~Q_PY_VECTORCALL_ARGUMENTS_OFFSET);
}

/* 调用 PyObject 的 Vectorcall 函数 */
PyObject * Q_PyObject_Vectorcall(
    PyObject * callable, PyObject * const * args, size_t nargsf,
    PyObject * kwnames) {
#ifdef PYPY_VERSION
    PyObject * dict = NULL;
    Py_ssize_t nargs = Q_PyVectorcall_NARGS(nargsf);
    if (kwnames) {
        Py_ssize_t nkwargs = PyTuple_GET_SIZE(kwnames);
        /* 创建关键字参数字典 */
        dict = build_kwarg_dict(&args[nargs - nkwargs], kwnames, nkwargs);
        if (!dict) {
            return NULL;
        }
        nargs -= nkwargs;
    }
    /* 调用带有字典参数的 Vectorcall 函数 */
    PyObject * ret = Q_PyObject_VectorcallDict(callable, args, nargs, dict);
    Py_XDECREF(dict);
    return ret;
#else
    /* 调用 PyObject 的 Vectorcall 函数 */
    return PyObject_Vectorcall(callable, args, nargsf, kwnames);
#endif
}

/* 调用 PyObject 的 Vectorcall 函数 */
PyObject * Q_PyObject_VectorcallDict(
    PyObject * callable, PyObject * const * args, size_t nargsf,
    PyObject * kwdict) {
#ifdef PYPY_VERSION
    Py_ssize_t nargs = Q_PyVectorcall_NARGS(nargsf);
    /* 创建参数元组 */
    PyObject * tuple = build_arg_tuple(args, nargs);
    if (!tuple) {
        return NULL;
    }
    /* 调用带有字典参数的 PyObject_Call 函数 */
    PyObject * ret = PyObject_Call(callable, tuple, kwdict);
    Py_DECREF(tuple);
    return ret;
#else
    /* 调用 PyObject 的 Vectorcall 函数 */
    return PyObject_VectorcallDict(callable, args, nargsf, kwdict);
#endif
}

/* 调用 PyObject 的 VectorcallMethod 函数 */
PyObject * Q_PyObject_VectorcallMethod(
    PyObject * name, PyObject * const * args, size_t nargsf,
    PyObject * kwnames) {
#ifdef PYPY_VERSION
    /* 获取对象的属性作为可调用对象 */
    PyObject * callable = PyObject_GetAttr(args[0], name);
    if (!callable) {
        return NULL;
    }
    /* 调用带有对象调用的 Vectorcall 函数 */
    PyObject * result =
        Q_PyObject_Vectorcall(callable, &args[1], nargsf - 1, kwnames);
    Py_DECREF(callable);
    return result;
#else
    /* 调用 PyObject 的 VectorcallMethod 函数 */
    return PyObject_VectorcallMethod(name, args, nargsf, kwnames);
#endif
}
```