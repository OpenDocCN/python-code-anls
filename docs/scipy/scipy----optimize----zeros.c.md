# `D:\src\scipysrc\scipy\scipy\optimize\zeros.c`

```
/*
 * Written by Charles Harris charles.harris@sdl.usu.edu
 */

/*
 * Modifications by Travis Oliphant to separate Python code from C routines
 */

#include "Python.h"
#include <setjmp.h>
#include "Zeros/zeros.h"

/*
 * Caller entry point functions
 */

#ifdef PYPY_VERSION
    /*
     * As described in http://doc.pypy.org/en/latest/cpython_differences.html#c-api-differences,
     * "assignment to a PyTupleObject is not supported after the tuple is used internally,
     * even by another C-API function call."
     */
    #define PyArgs(Operation) PyList_##Operation
#else
    /*
     * Using a list in CPython raises "TypeError: argument list must be a tuple"
     */
    #define PyArgs(Operation) PyTuple_##Operation
#endif

typedef struct {
    PyObject *function;  // Python 函数对象
    PyObject *xargs;     // 扩展参数元组
    jmp_buf env;         // 非局部跳转环境
} scipy_zeros_parameters;


static double
scipy_zeros_functions_func(double x, void *params)
{
    scipy_zeros_parameters *myparams = params;
    PyObject *args, *xargs, *item, *f, *retval=NULL;
    Py_ssize_t i, len;
    double val;

    xargs = myparams->xargs;
    /* Need to create a new 'args' tuple on each call in case 'f' is
       stateful and keeps references to it (e.g. functools.lru_cache) */
    len = PyTuple_Size(xargs);
    /* Make room for the double as first argument */
    args = PyArgs(New)(len + 1);  // 创建新的参数元组
    if (args == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate arguments");
        longjmp(myparams->env, 1);  // 分配失败，跳转到错误处理
    }
    PyArgs(SET_ITEM)(args, 0, Py_BuildValue("d", x));  // 将 double 类型的 x 添加到参数元组中
    for (i = 0; i < len; i++) {
        item = PyTuple_GetItem(xargs, i);  // 获取 xargs 元组中的每个元素
        if (item == NULL) {
            Py_DECREF(args);
            longjmp(myparams->env, 1);  // 获取元素失败，跳转到错误处理
        }
        Py_INCREF(item);  // 增加元素的引用计数
        PyArgs(SET_ITEM)(args, i+1, item);  // 将元素添加到参数元组中
    }

    f = myparams->function;  // 获取函数对象
    retval = PyObject_CallObject(f,args);  // 调用 Python 函数
    Py_DECREF(args);  // 释放参数元组的引用计数
    if (retval == NULL) {
        longjmp(myparams->env, 1);  // 调用函数失败，跳转到错误处理
    }
    val = PyFloat_AsDouble(retval);  // 将返回值转换为 double 类型
    Py_XDECREF(retval);  // 释放返回值对象的引用计数
    return val;  // 返回 double 类型的返回值
}


/*
 * Helper function that calls a Python function with extended arguments
 */

static PyObject *
call_solver(solver_type solver, PyObject *self, PyObject *args)
{
    double a, b, xtol, rtol, zero;
    int iter, fulloutput, disp=1, flag=0;
    scipy_zeros_parameters params;
    scipy_zeros_info solver_stats;
    PyObject *f, *xargs;

    if (!PyArg_ParseTuple(args, "OddddiOi|i",
                &f, &a, &b, &xtol, &rtol, &iter, &xargs, &fulloutput, &disp)) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to parse arguments");
        return NULL;
    }
    if (xtol < 0) {
        PyErr_SetString(PyExc_ValueError, "xtol must be >= 0");
        return NULL;
    }
    if (iter < 0) {
        PyErr_SetString(PyExc_ValueError, "maxiter should be > 0");
        return NULL;
    }

    params.function = f;  // 将 Python 函数对象赋值给参数结构体
    params.xargs = xargs;  // 将扩展参数元组赋值给参数结构体
    # 如果不是通过 longjmp 跳转过来，执行以下代码块
    if (!setjmp(params.env)) {
        /* 直接返回 */
        solver_stats.error_num = 0;  // 设置错误号为 0，表示没有错误
        zero = solver(scipy_zeros_functions_func, a, b, xtol, rtol,
                      iter, (void*)&params, &solver_stats);
    } else {
        /* 从 Python 函数中返回错误 */
        return NULL;  // 直接返回 NULL，表示函数执行出错
    }

    // 如果求解器返回的错误号不是 CONVERGED
    if (solver_stats.error_num != CONVERGED) {
        // 如果错误号是 SIGNERR，设置一个错误字符串并返回 NULL
        if (solver_stats.error_num == SIGNERR) {
            PyErr_SetString(PyExc_ValueError,
                    "f(a) and f(b) must have different signs");
            return NULL;
        }
        // 如果错误号是 CONVERR
        if (solver_stats.error_num == CONVERR) {
            // 如果 disp 为真，构造一个包含迭代次数的错误信息字符串并返回 NULL
            if (disp) {
                char msg[100];
                PyOS_snprintf(msg, sizeof(msg),
                        "Failed to converge after %d iterations.",
                        solver_stats.iterations);
                PyErr_SetString(PyExc_RuntimeError, msg);
                return NULL;
            }
            // 否则将 flag 设置为 CONVERR
            flag = CONVERR;
        }
    }
    else {
        // 如果错误号是 CONVERGED，将 flag 设置为 CONVERGED
        flag = CONVERGED;
    }

    // 如果 fulloutput 为真，返回包含多个值的元组
    if (fulloutput) {
        return Py_BuildValue("diii",
                zero, solver_stats.funcalls, solver_stats.iterations, flag);
    }
    // 否则只返回一个值
    else {
        return Py_BuildValue("d", zero);
    }
/*
 * These routines interface with the solvers through call_solver
 */

// 定义一个静态函数 _bisect，接收 self 和 args 参数，调用 call_solver 函数，并返回结果
static PyObject *
_bisect(PyObject *self, PyObject *args)
{
        return call_solver(bisect,self,args);
}

// 定义一个静态函数 _ridder，接收 self 和 args 参数，调用 call_solver 函数，并返回结果
static PyObject *
_ridder(PyObject *self, PyObject *args)
{
        return call_solver(ridder,self,args);
}

// 定义一个静态函数 _brenth，接收 self 和 args 参数，调用 call_solver 函数，并返回结果
static PyObject *
_brenth(PyObject *self, PyObject *args)
{
        return call_solver(brenth,self,args);
}

// 定义一个静态函数 _brentq，接收 self 和 args 参数，调用 call_solver 函数，并返回结果
static PyObject *
_brentq(PyObject *self, PyObject *args)
{
        return call_solver(brentq,self,args);
}

/*
 * Standard Python module interface
 */

// 定义一个 PyMethodDef 数组 Zerosmethods，包含了模块的四个方法名及其对应的处理函数和文档字符串
static PyMethodDef
Zerosmethods[] = {
    {"_bisect", _bisect, METH_VARARGS, "a"},
    {"_ridder", _ridder, METH_VARARGS, "a"},
    {"_brenth", _brenth, METH_VARARGS, "a"},
    {"_brentq", _brentq, METH_VARARGS, "a"},
    {NULL, NULL}
};

// 定义一个 PyModuleDef 结构体 moduledef，描述了模块的初始化方法、名称和方法列表
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  // 初始化 Python 模块定义
    "_zeros",                // 模块名
    NULL,                    // 模块文档字符串
    -1,                      // 模块状态，-1 表示没有全局状态
    Zerosmethods,            // 模块的方法列表
    NULL,                    // 模块的槽，可选
    NULL,                    // 模块的查询插槽，可选
    NULL,                    // 模块的清理函数，可选
    NULL                     // 模块的自定义数据，可选
};

// Python 模块初始化函数 PyInit__zeros
PyMODINIT_FUNC
PyInit__zeros(void)
{
    PyObject *m;

    // 创建一个新的 Python 模块对象，使用定义的 moduledef
    m = PyModule_Create(&moduledef);

    // 返回创建的 Python 模块对象
    return m;
}
```