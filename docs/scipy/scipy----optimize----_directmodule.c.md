# `D:\src\scipysrc\scipy\scipy\optimize\_directmodule.c`

```
/*
 * 包含必要的头文件以及_directmodule.h文件
 */
#include <Python.h>
#include <numpy/arrayobject.h>
#include "_directmodule.h"

/*
 * direct函数的实现，作为Python C扩展中的入口函数
 */
static PyObject *
direct(PyObject *self, PyObject *args)
{
    /*
     * 声明变量
     */
    PyObject *f, *f_args, *lb, *ub, *callback;
    int dimension, max_feval, max_iter, force_stop, disp;
    const double *lower_bounds, *upper_bounds;
    double minf, magic_eps, magic_eps_abs, *x;
    double volume_reltol, sigma_reltol;
    double fglobal, fglobal_reltol;
    FILE *logfile = NULL;
    direct_algorithm algorithm;
    direct_return_code ret_code;

    /*
     * 解析Python传入的参数元组，参数顺序为:
     * f, lb, ub, f_args, disp, magic_eps, max_feval, max_iter, algorithm,
     * fglobal, fglobal_reltol, volume_reltol, sigma_reltol, callback
     */
    if (!PyArg_ParseTuple(args, "OOOOidiiiddddO",
                          &f, &lb, &ub, &f_args, &disp, &magic_eps,
                          &max_feval, &max_iter, (int*) &algorithm,
                          &fglobal, &fglobal_reltol,
                          &volume_reltol, &sigma_reltol, &callback))
    {
        return NULL;
    }

    /*
     * 如果disp为真，则将日志文件指向标准输出
     */
    if (disp) {
        logfile = stdout;
    }

    /*
     * 获取lb数组的维度，作为问题的维度
     */
    dimension = PyArray_DIMS((PyArrayObject*)lb)[0];

    /*
     * 分配一个数组来存储优化器结果的解向量x
     */
    x = (double *) malloc(sizeof(double) * (dimension + 1));
    if (!(x)) {
        ret_code = DIRECT_OUT_OF_MEMORY;
    }

    /*
     * 创建一个Python列表对象来存储每次迭代的解向量序列x_seq
     */
    PyObject *x_seq = PyList_New(dimension);

    /*
     * 获取lb和ub数组的数据指针作为问题的下界和上界
     */
    lower_bounds = (double*)PyArray_DATA((PyArrayObject*)lb);
    upper_bounds = (double*)PyArray_DATA((PyArrayObject*)ub);

    /*
     * 初始化其他变量
     */
    magic_eps_abs = 0.0;
    force_stop = 0;
    direct_return_info info;

    /*
     * 调用优化器函数direct_optimize进行优化过程
     */
    if (!direct_optimize(f, x, x_seq, f_args, dimension, lower_bounds,
                         upper_bounds, &minf, max_feval, max_iter,
                         magic_eps, magic_eps_abs, volume_reltol,
                         sigma_reltol, &force_stop, fglobal, fglobal_reltol,
                         logfile, algorithm, &info, &ret_code, callback)) {
        if (x)
            free(x);
        return NULL;
    }

    /*
     * 构建返回值元组ret_py，包括优化结果x_seq、最小函数值minf、返回代码ret_code、
     * 函数调用次数info.numfunc和迭代次数info.numiter
     */
    PyObject* ret_py = Py_BuildValue("Odiii", x_seq, minf, (int) ret_code,
                                     info.numfunc, info.numiter);
    if (x)
        free(x);

    return ret_py;
}

/*
 * 定义Python模块的方法表
 */
static PyMethodDef
DIRECTMethods[] = {
    {"direct", direct, METH_VARARGS, "DIRECT Optimization Algorithm"},
    {NULL, NULL, 0, NULL}
};

/*
 * 定义Python模块的结构体
 */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_direct",
    NULL,
    -1,
    DIRECTMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

/*
 * Python模块的初始化函数，导入numpy数组接口并创建模块
 */
PyMODINIT_FUNC
PyInit__direct(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
```