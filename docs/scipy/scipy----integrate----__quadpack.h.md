# `D:\src\scipysrc\scipy\scipy\integrate\__quadpack.h`

```
/* This file should be included in the _multipackmodule file */
/* $Revision$ */
/* module_methods:
  {"_qagse", quadpack_qagse, METH_VARARGS, doc_qagse},
  {"_qagie", quadpack_qagie, METH_VARARGS, doc_qagie},
  {"_qagpe", quadpack_qagpe, METH_VARARGS, doc_qagpe},
  {"_qawoe", quadpack_qawoe, METH_VARARGS, doc_qawoe},
  {"_qawfe", quadpack_qawfe, METH_VARARGS, doc_qawfe},
  {"_qawse", quadpack_qawse, METH_VARARGS, doc_qawse},
  {"_qawce", quadpack_qawce, METH_VARARGS, doc_qawce},
 */
/* link libraries: (should be listed in separate lines)
   quadpack
   linpack_lite
   blas
   mach
 */
/* Python files: (to be imported to Multipack.py)
   quadpack.py
 */


#include <Python.h>
#include <setjmp.h>

#include "ccallback.h"

#include "numpy/arrayobject.h"

#ifdef HAVE_BLAS_ILP64

#define F_INT npy_int64
#define F_INT_NPY NPY_INT64

#if NPY_BITSOF_SHORT == 64
#define F_INT_PYFMT   "h"
#elif NPY_BITSOF_INT == 64
#define F_INT_PYFMT   "i"
#elif NPY_BITSOF_LONG == 64
#define F_INT_PYFMT   "l"
#elif NPY_BITSOF_LONGLONG == 64
#define F_INT_PYFMT   "L"
#else
#error No compatible 64-bit integer size. \
       Please contact NumPy maintainers and give detailed information about your \
       compiler and platform, or set NPY_USE_BLAS64_=0
#endif

#else

#define F_INT int
#define F_INT_NPY NPY_INT
#define F_INT_PYFMT   "i"

#endif

#define PYERR(errobj,message) {PyErr_SetString(errobj,message); goto fail;}
#define PYERR2(errobj,message) {PyErr_Print(); PyErr_SetString(errobj, message); goto fail;}
#define ISCONTIGUOUS(m) ((m)->flags & CONTIGUOUS)

static PyObject *quadpack_error;


#if defined(NO_APPEND_FORTRAN)
  #if defined(UPPERCASE_FORTRAN)
  /* nothing to do here */
  #else
    #define DQAGSE dqagse
    #define DQAGIE dqagie
    #define DQAGPE dqagpe
    #define DQAWOE dqawoe
    #define DQAWFE dqawfe
    #define DQAWSE dqawse
    #define DQAWCE dqawce
  #endif
#else
  #if defined(UPPERCASE_FORTRAN)
    #define DQAGSE DQAGSE_
    #define DQAGIE DQAGIE_
    #define DQAGPE DQAGPE_
    #define DQAWOE DQAWOE_
    #define DQAWFE DQAWFE_
    #define DQAWSE DQAWSE_
    #define DQAWCE DQAWCE_
#else
    #define DQAGSE dqagse_
    #define DQAGIE dqagie_
    #define DQAGPE dqagpe_
    #define DQAWOE dqawoe_
    #define DQAWFE dqawfe_
    #define DQAWSE dqawse_
    #define DQAWCE dqawce_
  #endif
#endif

 
typedef double quadpack_f_t(double *);

/* Declaration of Fortran subroutines for numerical integration */

void DQAGSE(quadpack_f_t f, double *a, double *b, double *epsabs, double *epsrel, F_INT *limit, double *result,
            double *abserr, F_INT *neval, F_INT *ier, double *alist, double *blist, double *rlist, double *elist,
            F_INT *iord, F_INT *last);
void DQAGIE(quadpack_f_t f, double *bound, F_INT *inf, double *epsabs, double *epsrel, F_INT *limit,
            double *result, double *abserr, F_INT *neval, F_INT *ier, double *alist, double *blist,
            double *rlist, double *elist, F_INT *iord, F_INT *last);
// 声明一个函数原型，该函数使用 Quadpack 标准接口，计算定积分，返回结果和误差估计等信息
void DQAGPE(quadpack_f_t f,          // 回调函数指针，用于计算被积函数的值
            double *a,               // 积分下限
            double *b,               // 积分上限
            F_INT *npts2,            // 传递给回调函数的参数数量
            double *points,          // 传递给回调函数的参数数组
            double *epsabs,          // 绝对误差容限
            double *epsrel,          // 相对误差容限
            F_INT *limit,            // 控制策略的极限
            double *result,          // 积分结果
            double *abserr,          // 积分结果的绝对误差估计
            F_INT *neval,            // 积分计算的函数调用次数
            F_INT *ier,              // 积分返回的状态指示
            double *alist,           // 积分区间分割点
            double *blist,           // 积分区间分割点
            double *rlist,           // 最后的积分结果
            double *elist,           // 积分结果的绝对误差估计
            double *pts,             // 传递给回调函数的参数数组
            F_INT *iord,             // 最后使用的存储方案
            F_INT *level,            // 深度
            F_INT *ndin,             // 子程序调用过程中内存空间的使用情况
            F_INT *last);            // 最后一个使用的子程序

// 声明一个函数原型，该函数使用 Quadpack 标准接口，计算积分并加权，返回结果和误差估计等信息
void DQAWOE(quadpack_f_t f,          // 回调函数指针，用于计算被积函数的值
            double *a,               // 积分下限
            double *b,               // 积分上限
            double *omega,           // 角频率参数
            F_INT *integr,           // 确定积分的类型
            double *epsabs,          // 绝对误差容限
            double *epsrel,          // 相对误差容限
            F_INT *limit,            // 控制策略的极限
            F_INT *icall,            // 第一次调用
            F_INT *maxp1,            // 维度
            double *result,          // 积分结果
            double *abserr,          // 积分结果的绝对误差估计
            F_INT *neval,            // 积分计算的函数调用次数
            F_INT *ier,              // 积分返回的状态指示
            F_INT *last,             // 最后一个使用的子程序
            double *alist,           // 积分区间分割点
            double *blist,           // 积分区间分割点
            double *rlist,           // 最后的积分结果
            double *elist,           // 积分结果的绝对误差估计
            F_INT *iord,             // 最后使用的存储方案
            F_INT *nnlog,            // 数组长度
            F_INT *momcom,           // 多项式
            double *chebmo);         // 多项式

// 声明一个函数原型，该函数使用 Quadpack 标准接口，计算积分并加权，返回结果和误差估计等信息
void DQAWFE(quadpack_f_t f,          // 回调函数指针，用于计算被积函数的值
            double *a,               // 积分下限
            double *omega,           // 角频率参数
            F_INT *integr,           // 确定积分的类型
            double *epsabs,          // 绝对误差容限
            F_INT *limlst,           // 传递给回调函数的参数数量
            F_INT *limit,            // 控制策略的极限
            F_INT *maxp1,            // 维度
            double *result,          // 积分结果
            double *abserr,          // 积分结果的绝对误差估计
            F_INT *neval,            // 积分计算的函数调用次数
            F_INT *ier,              // 积分返回的状态指示
            double *rslst,           // 传递给回调函数的参数数组
            double *erlst,           // 传递给回调函数的参数数组
            F_INT *ierlst,           // 最后使用的存储方案
            F_INT *lst,              // 深度
            double *alist,           // 积分区间分割点
            double *blist,           // 积分区间分割点
            double *rlist,           // 最后的积分结果
            double *elist,           // 积分结果的绝对误差估计
            F_INT *iord,             // 最后使用的存储方案
            F_INT *nnlog,            // 数组长度
            double *chebmo);         // 多项式

// 声明一个函数原型，该函数使用 Quadpack 标准接口，计算积分并加权，返回结果和误差估计等信息
void DQAWSE(quadpack_f_t f,          // 回调函数指针，用于计算被积函数的值
            double *a,               // 积分下限
            double *b,               // 积分上限
            double *alfa,            // 阿尔法
            double *beta,            // 贝塔
            F_INT *integr,           // 确定积分的类型
            double *epsabs,          // 绝对误差容限
            double *epsrel,          // 相对误差容限
            F_INT *limit,            // 控制策略的极限
            double *result,          // 积分结果
            double *abserr,          // 积分结果的绝对误差估计
            F_INT *neval,            // 积分计算的函数调用次数
            F_INT *ier,              // 积分返回的状态指示
            double *alist,           // 积分区间分割点
            double *blist,           // 积分区间分割点
            double *rlist,           // 最后的积分结果
            double *elist,           // 积分结果的绝对误差估计
            F_INT *iord,             // 最后使用的存储方案
            F_INT *last);            // 最后一个使用的子程序

// 声明一个函数原型，该函数使用 Quadpack 标准接口，计算积分并加权，返回结果和误差估计等信息
void DQAWCE(quadpack_f_t f,          // 回调函数指针，用于计算被积函数的值
            double *a,               // 积分下限
            double *b,               // 积分上限
            double *c,               // 参数 C
            double *epsabs,          // 绝对误差容限
            double *epsrel,          // 相对误差容限
            F_INT *limit,            // 控制策略的极限
            double *result,          // 积分结果
            double *abserr,          // 积分结果的绝对误差估计
            F_INT *neval,            // 积分计算的函数调用次数
            F_INT *ier,              // 积分返回的状态指示
            double *alist,           // 积分区间分割点
            double *blist,           // 积分区间分割点
            double *rlist,           // 最后的积分结果
            double *elist,           // 积分结果的绝对误差估计
            F_INT *iord,             // 最后使用的存储方案
            F
    // 如果指针 p 为 NULL，则说明内存分配失败，需要释放 p，并设置内存错误异常，返回 -1
    if (p == NULL) {
        free(p);
        PyErr_SetString(PyExc_MemoryError, "failed to allocate memory");
        return -1;
    }

    // 获取额外参数的元组大小
    size = PyTuple_Size(extra_arguments);
    // 如果额外参数的数量与 ndim - 1 不匹配，则释放 p，设置值错误异常，返回 -1
    if (size != ndim - 1) {
        free(p);
        PyErr_SetString(PyExc_ValueError, "extra arguments don't match ndim");
        return -1;
    }

    // 将 p 数组的第一个元素设置为 0
    p[0] = 0;
    // 遍历额外参数的元组
    for (i = 0; i < size; ++i) {
        PyObject *item;
        
        // 获取元组中第 i 个元素
        item = PyTuple_GET_ITEM(extra_arguments, i);
        // 将该元素转换为 double 类型，并赋值给 p 数组对应位置
        p[i+1] = PyFloat_AsDouble(item);
        // 如果转换过程中发生错误，则释放 p 并返回 -1
        if (PyErr_Occurred()) {
            free(p);
            return -1;
        }
    }

    // 将 p 数组转换为 void 指针，并赋值给 callback 结构体的 info_p 字段
    callback->info_p = (void *)p;
    // 返回成功标志 0
    return 0;
static int
init_callback(ccallback_t *callback, PyObject *func, PyObject *extra_arguments)
{
    static PyObject *cfuncptr_type = NULL;

    int ret;
    int ndim;
    int flags = CCALLBACK_OBTAIN;
    ccallback_signature_t *signatures = quadpack_call_signatures;

    // 检查是否已经导入 ctypes 模块，如果未导入则导入它
    if (cfuncptr_type == NULL) {
        PyObject *module;

        module = PyImport_ImportModule("ctypes");
        if (module == NULL) {
            return -1;
        }

        // 获取 ctypes 模块中的 _CFuncPtr 属性
        cfuncptr_type = PyObject_GetAttrString(module, "_CFuncPtr");
        Py_DECREF(module);
        if (cfuncptr_type == NULL) {
            return -1;
        }
    }

    // 检查 func 是否为 ctypes 指针类型，如果是，则设置相应的标志位和签名
    if (PyObject_TypeCheck(func, (PyTypeObject *)cfuncptr_type)) {
        /* Legacy support --- ctypes objects can be passed in as-is */
        flags |= CCALLBACK_PARSE;
        signatures = quadpack_call_legacy_signatures;
    }

    // 准备回调函数，根据签名和给定的参数
    ret = ccallback_prepare(callback, signatures, func, flags);
    if (ret == -1) {
        return -1;
    }

    // 根据回调函数的签名类型，设置额外参数信息
    if (callback->signature == NULL) {
        /* pure-Python */
        callback->info_p = (void *)extra_arguments;
    }
    else if (callback->signature->value == CB_1D || callback->signature->value == CB_1D_USER) {
        /* extra_arguments is just ignored */
        callback->info_p = NULL;
    }
    else {
        // 检查额外参数的合法性，并初始化多维数据结构
        if (!PyTuple_Check(extra_arguments)) {
            PyErr_SetString(PyExc_ValueError, "multidimensional integrand but invalid extra args");
            return -1;
        }

        ndim = PyTuple_GET_SIZE(extra_arguments) + 1;

        callback->info = ndim;

        // 初始化多维数据结构
        if (init_multivariate_data(callback, ndim, extra_arguments) == -1) {
            return -1;
        }
    }

    return 0;
}


static int
free_callback(ccallback_t *callback)
{
    // 如果回调函数签名类型为多维，则释放额外参数信息
    if (callback->signature && (callback->signature->value == CB_ND ||
                                callback->signature->value == CB_ND_USER)) {
        free(callback->info_p);
        callback->info_p = NULL;
    }

    // 释放回调函数资源
    if (ccallback_release(callback) != 0) {
        return -1;
    }

    return 0;
}


double quad_thunk(double *x)
{
    // 获取回调函数对象
    ccallback_t *callback = ccallback_obtain();
    double result = 0;
    int error = 0;
    // 检查回调函数是否存在
    if (callback->py_function) {
        // 初始化变量
        PyObject *arg1 = NULL, *argobj = NULL, *arglist = NULL, *res = NULL;
        // 获取额外的参数对象
        PyObject *extra_arguments = (PyObject *)callback->info_p;

        // 将浮点数*x转换为Python的float对象
        argobj = PyFloat_FromDouble(*x);
        if (argobj == NULL) {
            error = 1;  // 设置错误标志
            goto done;  // 跳转到清理阶段
        }

        // 创建一个包含一个元素的元组，用于作为函数的参数
        arg1 = PyTuple_New(1);
        if (arg1 == NULL) {
            error = 1;  // 设置错误标志
            goto done;  // 跳转到清理阶段
        }

        // 将argobj放入arg1元组中的第一个位置
        PyTuple_SET_ITEM(arg1, 0, argobj);
        argobj = NULL;

        // 将arg1元组和额外的参数对象合并成一个新的参数列表
        arglist = PySequence_Concat(arg1, extra_arguments);
        if (arglist == NULL) {
            error = 1;  // 设置错误标志
            goto done;  // 跳转到清理阶段
        }

        // 调用Python回调函数，并获取返回值
        res = PyObject_CallObject(callback->py_function, arglist);
        if (res == NULL) {
            error = 1;  // 设置错误标志
            goto done;  // 跳转到清理阶段
        }

        // 将Python对象res转换为双精度浮点数
        result = PyFloat_AsDouble(res);
        if (PyErr_Occurred()) {
            error = 1;  // 设置错误标志
            goto done;  // 跳转到清理阶段
        }

    done:
        // 释放所有Python对象的引用
        Py_XDECREF(arg1);
        Py_XDECREF(argobj);
        Py_XDECREF(arglist);
        Py_XDECREF(res);
    }
    else {
        // 根据回调函数的类型执行相应的C函数调用
        switch (callback->signature->value) {
        case CB_1D_USER:
            result = ((double(*)(double, void *))callback->c_function)(*x, callback->user_data);
            break;
        case CB_1D:
            result = ((double(*)(double))callback->c_function)(*x);
            break;
        case CB_ND_USER:
            ((double *)callback->info_p)[0] = *x;
            result = ((double(*)(int, double *, void *))callback->c_function)(
                (int)callback->info, (double *)callback->info_p, callback->user_data);
            break;
        case CB_ND:
            ((double *)callback->info_p)[0] = *x;
            result = ((double(*)(int, double *))callback->c_function)(
                (int)callback->info, (double *)callback->info_p);
            break;
        default:
            error = 1;  // 设置错误标志
            // 如果默认情况下，发生了无效的回调类型，抛出致命错误
            Py_FatalError("scipy.integrate.quad: internal error (this is a bug!): invalid callback type");
            break;
        }
    }

    // 如果发生了错误，通过长跳转回调错误处理程序
    if (error) {
        longjmp(callback->error_buf, 1);
    }

    // 返回计算得到的结果
    return result;
}

// 函数文档字符串，描述函数签名和返回值
static char doc_qagse[] = "[result,abserr,infodict,ier] = _qagse(fun, a, b, | args, full_output, epsabs, epsrel, limit)";

// quadpack_qagse 函数定义，处理 Python 中的函数调用和参数解析
static PyObject *quadpack_qagse(PyObject *dummy, PyObject *args) {

  // 定义 PyArrayObject 类型的变量，用于存储传入和计算过程中的数组数据
  PyArrayObject *ap_alist = NULL, *ap_iord = NULL;
  PyArrayObject *ap_blist = NULL, *ap_elist = NULL;
  PyArrayObject *ap_rlist = NULL;

  // 定义额外的参数和函数对象
  PyObject *extra_args = NULL;
  PyObject *fcn;

  // 初始化默认参数和变量
  F_INT limit=50;
  npy_intp limit_shape[1];
  int      full_output = 0;
  double   a, b, epsabs=1.49e-8, epsrel=1.49e-8;
  F_INT neval=0, ier=6, last=0, *iord;
  double   result=0.0, abserr=0.0;
  double   *alist, *blist, *rlist, *elist;
  int      ret;
  ccallback_t callback;

  // 解析传入的 Python 参数，并进行错误检查
  if (!PyArg_ParseTuple(args, ("Odd|Oidd" F_INT_PYFMT), &fcn, &a, &b, &extra_args, &full_output, &epsabs, &epsrel, &limit)) return NULL;
  limit_shape[0] = limit;

  // 检查 limit 值是否大于 1，如果不是则返回默认值
  if (limit < 1)
      return Py_BuildValue(("dd" F_INT_PYFMT),result,abserr,ier);

  // 初始化回调函数并检查初始化是否成功
  ret = init_callback(&callback, fcn, extra_args);
  if (ret == -1) {
      return NULL;
  }

  // 设置并初始化工作数组和排序数组
  ap_iord = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,F_INT_NPY);
  ap_alist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_blist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_rlist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_elist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  if (ap_iord == NULL || ap_alist == NULL || ap_blist == NULL || ap_rlist == NULL || ap_elist == NULL) goto fail;
  iord = (F_INT *)PyArray_DATA(ap_iord);
  alist = (double *)PyArray_DATA(ap_alist);
  blist = (double *)PyArray_DATA(ap_blist);
  rlist = (double *)PyArray_DATA(ap_rlist);
  elist = (double *)PyArray_DATA(ap_elist);

  // 设置错误跳转点，如果发生错误则释放资源
  if (setjmp(callback.error_buf) != 0) {
      goto fail;
  }

  // 调用数值积分函数 DQAGSE 进行积分计算
  DQAGSE(quad_thunk, &a, &b, &epsabs, &epsrel, &limit, &result, &abserr, &neval, &ier, alist,
         blist, rlist, elist, iord, &last);

  // 释放回调函数资源
  if (free_callback(&callback) != 0) {
      goto fail_free;
  }

  // 如果需要完整输出，则返回详细结果
  if (full_output) {
      return Py_BuildValue(("dd{s:" F_INT_PYFMT ",s:" F_INT_PYFMT ",s:N,s:N,s:N,s:N,s:N}" F_INT_PYFMT), result, abserr, "neval", neval, "last", last, "iord", PyArray_Return(ap_iord), "alist", PyArray_Return(ap_alist), "blist", PyArray_Return(ap_blist), "rlist", PyArray_Return(ap_rlist), "elist", PyArray_Return(ap_elist),ier);
  }
  // 否则，只返回结果值，释放数组对象
  else {
    Py_DECREF(ap_alist);
    Py_DECREF(ap_blist);
    Py_DECREF(ap_rlist);
    Py_DECREF(ap_elist);
    Py_DECREF(ap_iord);
    return Py_BuildValue(("dd" F_INT_PYFMT),result,abserr,ier);
  }

 fail:
  // 失败时释放回调函数资源
  free_callback(&callback);
 fail_free:
  // 释放所有数组对象
  Py_XDECREF(ap_alist);
  Py_XDECREF(ap_blist);
  Py_XDECREF(ap_rlist);
  Py_XDECREF(ap_elist);
  Py_XDECREF(ap_iord);
  return NULL;
}

// 函数文档字符串，描述函数签名和返回值
static char doc_qagie[] = "[result,abserr,infodict,ier] = _qagie(fun, bound, inf, | args, full_output, epsabs, epsrel, limit)";
static PyObject *quadpack_qagie(PyObject *dummy, PyObject *args) {

  PyArrayObject *ap_alist = NULL, *ap_iord = NULL;
  PyArrayObject *ap_blist = NULL, *ap_elist = NULL;
  PyArrayObject *ap_rlist = NULL;

  PyObject *extra_args = NULL;
  PyObject *fcn;

  F_INT limit=50;
  npy_intp limit_shape[1];
  int      full_output = 0;
  double   bound, epsabs=1.49e-8, epsrel=1.49e-8;
  F_INT inf, neval=0, ier=6, last=0, *iord;
  double   result=0.0, abserr=0.0;
  double   *alist, *blist, *rlist, *elist;
  int ret;
  ccallback_t callback;

  // 解析 Python 函数调用的参数
  if (!PyArg_ParseTuple(args, ("Od" F_INT_PYFMT "|Oidd" F_INT_PYFMT), &fcn, &bound, &inf, &extra_args,
                        &full_output, &epsabs, &epsrel, &limit))
    return NULL;
  limit_shape[0] = limit;

  /* Need to check that limit is greater than 1 */
  // 检查 limit 是否大于1，否则返回默认值
  if (limit < 1)
      return Py_BuildValue(("dd" F_INT_PYFMT),result,abserr,ier);

  // 初始化回调函数
  ret = init_callback(&callback, fcn, extra_args);
  if (ret == -1) {
      return NULL;
  }

  // 设置 iord 和工作数组
  ap_iord = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,F_INT_NPY);
  ap_alist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_blist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_rlist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_elist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  if (ap_iord == NULL || ap_alist == NULL || ap_blist == NULL || ap_rlist == NULL
      || ap_elist == NULL) goto fail;
  iord = (F_INT *)PyArray_DATA(ap_iord);
  alist = (double *)PyArray_DATA(ap_alist);
  blist = (double *)PyArray_DATA(ap_blist);
  rlist = (double *)PyArray_DATA(ap_rlist);
  elist = (double *)PyArray_DATA(ap_elist);

  // 使用长跳转来处理可能的错误
  if (setjmp(callback.error_buf) != 0) {
      goto fail;
  }

  // 调用数值积分函数 DQAGIE 进行积分计算
  DQAGIE(quad_thunk, &bound, &inf, &epsabs, &epsrel, &limit, &result, &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);

  // 释放回调函数资源
  if (free_callback(&callback) != 0) {
      goto fail_free;
  }

  // 根据 full_output 返回结果
  if (full_output) {
      return Py_BuildValue(("dd{s:" F_INT_PYFMT ",s:" F_INT_PYFMT ",s:N,s:N,s:N,s:N,s:N}" F_INT_PYFMT), result, abserr, "neval", neval, "last", last, "iord", PyArray_Return(ap_iord), "alist", PyArray_Return(ap_alist), "blist", PyArray_Return(ap_blist), "rlist", PyArray_Return(ap_rlist), "elist", PyArray_Return(ap_elist), ier);
  }
  else {
    // 清理内存并返回结果
    Py_DECREF(ap_alist);
    Py_DECREF(ap_blist);
    Py_DECREF(ap_rlist);
    Py_DECREF(ap_elist);
    Py_DECREF(ap_iord);
    return Py_BuildValue(("dd" F_INT_PYFMT), result, abserr, ier);
  }

 fail:
  // 发生错误时释放资源
  free_callback(&callback);
 fail_free:
  // 释放所有数组资源
  Py_XDECREF(ap_alist);
  Py_XDECREF(ap_blist);
  Py_XDECREF(ap_rlist);
  Py_XDECREF(ap_elist);
  Py_XDECREF(ap_iord);
  return NULL;
}


static char doc_qagpe[] = "[result,abserr,infodict,ier] = _qagpe(fun, a, b, points, | args, full_output, epsabs, epsrel, limit)";

    Py_DECREF(ap_alist);
    Py_DECREF(ap_blist);
    Py_DECREF(ap_rlist);
    Py_DECREF(ap_elist);
    Py_DECREF(ap_pts);
    // 递减对象引用计数，释放对 ap_iord 的引用
    Py_DECREF(ap_iord);
    // 递减对象引用计数，释放对 ap_ndin 的引用
    Py_DECREF(ap_ndin);
    // 递减对象引用计数，释放对 ap_level 的引用
    Py_DECREF(ap_level);
    // 使用 Py_BuildValue 函数构建一个 Python 对象，包含两个双精度浮点数和一个整数
    return Py_BuildValue(("dd" F_INT_PYFMT), result, abserr, ier);
  }

 fail:
  // 释放回调函数相关资源
  free_callback(&callback);
 fail_free:
  // 递减对象引用计数，释放对 ap_alist 的引用
  Py_XDECREF(ap_alist);
  // 递减对象引用计数，释放对 ap_blist 的引用
  Py_XDECREF(ap_blist);
  // 递减对象引用计数，释放对 ap_rlist 的引用
  Py_XDECREF(ap_rlist);
  // 递减对象引用计数，释放对 ap_elist 的引用
  Py_XDECREF(ap_elist);
  // 递减对象引用计数，释放对 ap_iord 的引用
  Py_XDECREF(ap_iord);
  // 递减对象引用计数，释放对 ap_pts 的引用
  Py_XDECREF(ap_pts);
  // 递减对象引用计数，释放对 ap_points 的引用
  Py_XDECREF(ap_points);
  // 递减对象引用计数，释放对 ap_ndin 的引用
  Py_XDECREF(ap_ndin);
  // 递减对象引用计数，释放对 ap_level 的引用
  Py_XDECREF(ap_level);
  // 返回空指针表示失败
  return NULL;
}



static char doc_qawoe[] = "[result,abserr,infodict,ier] = _qawoe(fun, a, b, omega, integr, | args, full_output, epsabs, epsrel, limit, maxp1, icall, momcom, chebmo)";

注释：

# 定义函数的文档字符串，描述了函数 _qawoe 的参数和返回值
static char doc_qawoe[] = "[result,abserr,infodict,ier] = _qawoe(fun, a, b, omega, integr, | args, full_output, epsabs, epsrel, limit, maxp1, icall, momcom, chebmo)";



static PyObject *quadpack_qawoe(PyObject *dummy, PyObject *args) {

注释：

# 定义了 quadpack_qawoe 函数，这是一个 Python C 扩展函数
static PyObject *quadpack_qawoe(PyObject *dummy, PyObject *args) {



  PyArrayObject *ap_alist = NULL, *ap_iord = NULL;
  PyArrayObject *ap_blist = NULL, *ap_elist = NULL;
  PyArrayObject *ap_rlist = NULL, *ap_nnlog = NULL;
  PyArrayObject *ap_chebmo = NULL;

注释：

# 定义了多个 PyArrayObject 类型的指针变量，用于存储 NumPy 数组对象
PyArrayObject *ap_alist = NULL, *ap_iord = NULL;
PyArrayObject *ap_blist = NULL, *ap_elist = NULL;
PyArrayObject *ap_rlist = NULL, *ap_nnlog = NULL;
PyArrayObject *ap_chebmo = NULL;



  PyObject *extra_args = NULL, *o_chebmo = NULL;
  PyObject *fcn;

注释：

# 定义了多个 PyObject 类型的变量，用于存储 Python 对象
PyObject *extra_args = NULL, *o_chebmo = NULL;
PyObject *fcn;



  F_INT limit=50;
  npy_intp limit_shape[1], sz[2];
  int      full_output = 0;
  F_INT maxp1=50, icall=1;
  double   a, b, epsabs=1.49e-8, epsrel=1.49e-8;
  F_INT neval=0, ier=6, integr=1, last=0, momcom=0, *iord;
  F_INT *nnlog;
  double   result=0.0, abserr=0.0, omega=0.0;
  double   *chebmo;
  double   *alist, *blist, *rlist, *elist;
  int ret;
  ccallback_t callback;

注释：

# 定义了多个变量，包括整型、双精度浮点型、指针等，用于存储函数参数和结果
F_INT limit=50;                             // 上限值，默认为 50
npy_intp limit_shape[1], sz[2];              // NumPy 整数类型的数组和大小数组
int full_output = 0;                        // 是否输出完整结果，默认为否
F_INT maxp1=50, icall=1;                     // 最大值加一，调用次数，均默认为 50 和 1
double a, b, epsabs=1.49e-8, epsrel=1.49e-8; // 积分上下限以及绝对和相对误差，默认为 1.49e-8
F_INT neval=0, ier=6, integr=1, last=0, momcom=0, *iord; // 评估次数，错误代码，积分方法等，默认为 0、6、1、0、0
F_INT *nnlog;                                // 指向整数的指针
double result=0.0, abserr=0.0, omega=0.0;    // 积分结果、绝对误差、权重，默认为 0.0
double *chebmo;                              // 指向双精度浮点数的指针
double *alist, *blist, *rlist, *elist;       // 指向双精度浮点数数组的指针
int ret;                                     // 返回值
ccallback_t callback;                        // 回调函数



  if (!PyArg_ParseTuple(args,
                        ("Oddd" F_INT_PYFMT "|Oidd" F_INT_PYFMT
                         F_INT_PYFMT F_INT_PYFMT F_INT_PYFMT "O"),
                        &fcn, &a, &b, &omega, &integr, &extra_args, &full_output, &epsabs, &epsrel, &limit, &maxp1, &icall, &momcom, &o_chebmo)) return NULL;

注释：

# 解析传入的参数元组，如果解析失败则返回空指针
if (!PyArg_ParseTuple(args,
                      ("Oddd" F_INT_PYFMT "|Oidd" F_INT_PYFMT
                       F_INT_PYFMT F_INT_PYFMT F_INT_PYFMT "O"),
                      &fcn, &a, &b, &omega, &integr, &extra_args, &full_output, &epsabs, &epsrel, &limit, &maxp1, &icall, &momcom, &o_chebmo)) return NULL;



  limit_shape[0] = limit;

注释：

# 将 limit 的值赋给 limit_shape 数组的第一个元素
limit_shape[0] = limit;



  /* Need to check that limit is greater than 1 */
  if (limit < 1)
      return Py_BuildValue(("dd" F_INT_PYFMT),result,abserr,ier);

注释：

# 检查 limit 是否大于 1，如果不是则返回包含结果、误差和错误代码的 Python 对象
/* Need to check that limit is greater than 1 */
if (limit < 1)
    return Py_BuildValue(("dd" F_INT_PYFMT),result,abserr,ier);



  ret = init_callback(&callback, fcn, extra_args);
  if (ret == -1) {
      return NULL;
  }

注释：

# 调用 init_callback 函数初始化回调函数，如果返回值为 -1 则返回空指针
ret = init_callback(&callback, fcn, extra_args);
if (ret == -1) {
    return NULL;
}



  if (o_chebmo != NULL) {
    ap_chebmo = (PyArrayObject *)PyArray_ContiguousFromObject(o_chebmo, NPY_DOUBLE, 2, 2);
    if (ap_chebmo == NULL) goto fail;
    if (PyArray_DIMS(ap_chebmo)[1] != maxp1 || PyArray_DIMS(ap_chebmo)[0] != 25)
      PYERR(quadpack_error,"Chebyshev moment array has the wrong size.");
  }
  else {
    sz[0] = 25;
    sz[1] = maxp1;
    ap_chebmo = (PyArrayObject *)PyArray_SimpleNew(2,sz,NPY_DOUBLE);

注释：

# 如果 o_chebmo 不为空，则将其转换为 PyArrayObject 类型的对象 ap_chebmo，
# 并检查其维度是否符合要求，否则创建一个新的 PyArrayObject 对象 ap_chebmo
if (o_chebmo != NULL) {
  ap_chebmo = (PyArrayObject *)PyArray_ContiguousFromObject(o_chebmo, NPY_DOUBLE, 2, 2);
  if (ap_chebmo == NULL) goto fail;
  if (PyArray_DIMS(ap_chebmo)[1] != maxp1 || PyArray_DIMS(ap_chebmo)[0] != 25)
    PYERR(quadpack_error,"Chebyshev moment array has the wrong size.");
}
else {
  sz[0] = 25;
  sz[1] = maxp1;
  ap_chebmo = (PyArrayObject *)PyArray_SimpleNew(2,sz,NPY_DOUBLE);





注释：

# 空行，结束函数 quadpack_qawoe 的定义
    # 检查指针变量 ap_chebmo 是否为 NULL，如果是则跳转到失败处理代码块
    if (ap_chebmo == NULL) goto fail;
  }
  # 将 PyArrayObject 对象 ap_chebmo 的数据转换为 double 类型的指针，并赋给 chebmo
  chebmo = (double *) PyArray_DATA(ap_chebmo);

  /* 设置 iwork 和 work 数组 */
  # 使用 PyArray_SimpleNew 创建新的 PyArrayObject 对象，分别对应 ap_iord, ap_nnlog, ap_alist, ap_blist, ap_rlist, ap_elist
  ap_iord = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,F_INT_NPY);
  ap_nnlog = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,F_INT_NPY);
  ap_alist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_blist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_rlist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_elist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  # 检查以上数组是否成功创建，如果有任何一个创建失败则跳转到失败处理代码块
  if (ap_iord == NULL || ap_nnlog == NULL || ap_alist == NULL || ap_blist == NULL || ap_rlist == NULL || ap_elist == NULL) goto fail;
  # 将以上数组对象的数据转换为对应类型的指针，并赋给对应的变量 iord, nnlog, alist, blist, rlist, elist
  iord = (F_INT *)PyArray_DATA(ap_iord);
  nnlog = (F_INT *)PyArray_DATA(ap_nnlog);
  alist = (double *)PyArray_DATA(ap_alist);
  blist = (double *)PyArray_DATA(ap_blist);
  rlist = (double *)PyArray_DATA(ap_rlist);
  elist = (double *)PyArray_DATA(ap_elist);

  # 设置长跳转的错误处理点，并执行 DQAWOE 函数调用
  if (setjmp(callback.error_buf) != 0) {
      goto fail;
  }

  # 调用 DQAWOE 函数，传递所需参数和数组指针
  DQAWOE(quad_thunk, &a, &b, &omega, &integr, &epsabs, &epsrel, &limit, &icall, &maxp1, &result, &abserr, &neval, &ier, &last, alist, blist, rlist, elist, iord, nnlog, &momcom, chebmo);

  # 释放 callback 并检查是否成功
  if (free_callback(&callback) != 0) {
      goto fail_free;
  }

  # 如果设置了 full_output 标志，则构建包含详细输出的 Python 对象并返回
  if (full_output) {
      return Py_BuildValue(("dd{s:" F_INT_PYFMT ",s:" F_INT_PYFMT ",s:N,s:N,s:N,s:N,s:N,s:N,s:" F_INT_PYFMT ",s:N}" F_INT_PYFMT ""), result, abserr, "neval", neval, "last", last, "iord", PyArray_Return(ap_iord), "alist", PyArray_Return(ap_alist), "blist", PyArray_Return(ap_blist), "rlist", PyArray_Return(ap_rlist), "elist", PyArray_Return(ap_elist), "nnlog", PyArray_Return(ap_nnlog), "momcom", momcom, "chebmo", PyArray_Return(ap_chebmo),ier);
  }
  else {
    # 否则，释放所有数组对象，并构建简单的 Python 对象返回
    Py_DECREF(ap_alist);
    Py_DECREF(ap_blist);
    Py_DECREF(ap_rlist);
    Py_DECREF(ap_elist);
    Py_DECREF(ap_iord);
    Py_DECREF(ap_nnlog);
    Py_DECREF(ap_chebmo);
    return Py_BuildValue(("dd" F_INT_PYFMT),result,abserr,ier);
  }

 fail:
  # 处理失败情况，释放 callback 并返回 NULL
  free_callback(&callback);
 fail_free:
  # 释放所有数组对象，并返回 NULL
  Py_XDECREF(ap_alist);
  Py_XDECREF(ap_blist);
  Py_XDECREF(ap_rlist);
  Py_XDECREF(ap_elist);
  Py_XDECREF(ap_iord);
  Py_XDECREF(ap_nnlog);
  Py_XDECREF(ap_chebmo);
  return NULL;
}

static char doc_qawfe[] = "[result,abserr,infodict,ier] = _qawfe(fun, a, omega, integr, | args, full_output, epsabs, limlst, limit, maxp1)";
    // 定义文档字符串，描述 _qawfe 函数的参数和返回值

    Py_DECREF(ap_rslst);
    // 释放 Python 对象 ap_rslst 的引用计数

    Py_DECREF(ap_erlst);
    // 释放 Python 对象 ap_erlst 的引用计数

    Py_DECREF(ap_ierlst);
    // 释放 Python 对象 ap_ierlst 的引用计数

    return Py_BuildValue(("dd" F_INT_PYFMT),result,abserr,ier);
    // 使用 result, abserr, ier 构建一个 Python 元组对象并返回，使用指定的格式字符串

  }

 fail:
  free_callback(&callback);
  // 跳转标签 fail:，释放回调函数 callback 所占用的资源

 fail_free:
  Py_XDECREF(ap_alist);
  // 跳转标签 fail_free:，释放 Python 对象 ap_alist 的引用计数

  Py_XDECREF(ap_blist);
  // 释放 Python 对象 ap_blist 的引用计数

  Py_XDECREF(ap_rlist);
  // 释放 Python 对象 ap_rlist 的引用计数

  Py_XDECREF(ap_elist);
  // 释放 Python 对象 ap_elist 的引用计数

  Py_XDECREF(ap_iord);
  // 释放 Python 对象 ap_iord 的引用计数

  Py_XDECREF(ap_nnlog);
  // 释放 Python 对象 ap_nnlog 的引用计数

  Py_XDECREF(ap_chebmo);
  // 释放 Python 对象 ap_chebmo 的引用计数

  Py_XDECREF(ap_rslst);
  // 释放 Python 对象 ap_rslst 的引用计数

  Py_XDECREF(ap_erlst);
  // 释放 Python 对象 ap_erlst 的引用计数

  Py_XDECREF(ap_ierlst);
  // 释放 Python 对象 ap_ierlst 的引用计数

  return NULL;
  // 函数结束，返回空指针表示执行失败

}


static char doc_qawce[] = "[result,abserr,infodict,ier] = _qawce(fun, a, b, c, | args, full_output, epsabs, epsrel, limit)";
    // 定义文档字符串，描述 _qawce 函数的参数和返回值
static PyObject *quadpack_qawce(PyObject *dummy, PyObject *args) {

  PyArrayObject *ap_alist = NULL, *ap_iord = NULL;
  PyArrayObject *ap_blist = NULL, *ap_elist = NULL;
  PyArrayObject *ap_rlist = NULL;

  PyObject *extra_args = NULL;
  PyObject *fcn;

  F_INT limit;
  npy_intp limit_shape[1];
  int      full_output = 0;
  double   a, b, c, epsabs=1.49e-8, epsrel=1.49e-8;
  F_INT neval=0, ier=6, last=0, *iord;
  double   result=0.0, abserr=0.0;
  double   *alist, *blist, *rlist, *elist;
  int ret;
  ccallback_t callback;

  // 解析输入参数并初始化变量
  if (!PyArg_ParseTuple(args, ("Oddd|Oidd" F_INT_PYFMT), &fcn, &a, &b, &c, &extra_args, &full_output, &epsabs, &epsrel, &limit)) return NULL;
  limit_shape[0] = limit;

  /* Need to check that limit is greater than 1 */
  // 检查 limit 是否大于1，如果不是，则返回结果和错误信息
  if (limit < 1)
      return Py_BuildValue(("dd" F_INT_PYFMT),result,abserr,ier);

  // 初始化回调函数
  ret = init_callback(&callback, fcn, extra_args);
  if (ret == -1) {
      return NULL;
  }

  /* Set up iwork and work arrays */
  // 分配并初始化用于计算的数组
  ap_iord = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,F_INT_NPY);
  ap_alist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_blist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_rlist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_elist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  if (ap_iord == NULL || ap_alist == NULL || ap_blist == NULL || ap_rlist == NULL || ap_elist == NULL) goto fail;
  iord = (F_INT *)PyArray_DATA(ap_iord);
  alist = (double *)PyArray_DATA(ap_alist);
  blist = (double *)PyArray_DATA(ap_blist);
  rlist = (double *)PyArray_DATA(ap_rlist);
  elist = (double *)PyArray_DATA(ap_elist);

  // 设置错误处理跳转点
  if (setjmp(callback.error_buf) != 0) {
      goto fail;
  }

  // 调用数值积分函数 DQAWCE 计算积分结果
  DQAWCE(quad_thunk, &a, &b, &c, &epsabs, &epsrel, &limit, &result, &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);

  // 释放回调函数资源
  if (free_callback(&callback) != 0) {
      goto fail_free;
  }

  // 根据 full_output 参数选择返回结果的格式
  if (full_output) {
      return Py_BuildValue(("dd{s:" F_INT_PYFMT ",s:" F_INT_PYFMT ",s:N,s:N,s:N,s:N,s:N}" F_INT_PYFMT), result, abserr, "neval", neval, "last", last, "iord", PyArray_Return(ap_iord), "alist", PyArray_Return(ap_alist), "blist", PyArray_Return(ap_blist), "rlist", PyArray_Return(ap_rlist), "elist", PyArray_Return(ap_elist), ier);
  }
  else {
    // 释放不需要返回的数组资源并返回简化格式的结果
    Py_DECREF(ap_alist);
    Py_DECREF(ap_blist);
    Py_DECREF(ap_rlist);
    Py_DECREF(ap_elist);
    Py_DECREF(ap_iord);
    return Py_BuildValue(("dd" F_INT_PYFMT), result, abserr, ier);
  }

 fail:
  // 失败时释放回调函数资源
  free_callback(&callback);
 fail_free:
  // 失败时释放所有数组资源并返回空指针
  Py_XDECREF(ap_alist);
  Py_XDECREF(ap_blist);
  Py_XDECREF(ap_rlist);
  Py_XDECREF(ap_elist);
  Py_XDECREF(ap_iord);
  return NULL;
}


static char doc_qawse[] = "[result,abserr,infodict,ier] = _qawse(fun, a, b, (alfa, beta), integr, | args, full_output, epsabs, epsrel, limit)";
static PyObject *quadpack_qawse(PyObject *dummy, PyObject *args) {

静态函数 `quadpack_qawse` 接受一个 `PyObject` 类型的参数 `args`，返回一个 `PyObject` 指针。


  PyArrayObject *ap_alist = NULL, *ap_iord = NULL;
  PyArrayObject *ap_blist = NULL, *ap_elist = NULL;
  PyArrayObject *ap_rlist = NULL;

  PyObject *extra_args = NULL;
  PyObject *fcn;

定义了几个 `PyArrayObject` 和 `PyObject` 类型的变量，并初始化为 `NULL`。


  int      full_output = 0;
  F_INT integr;
  F_INT limit=50;
  npy_intp limit_shape[1];
  double   a, b, epsabs=1.49e-8, epsrel=1.49e-8;
  double   alfa, beta;
  F_INT neval=0, ier=6, last=0, *iord;
  double   result=0.0, abserr=0.0;
  double   *alist, *blist, *rlist, *elist;
  int ret;
  ccallback_t callback;

声明了一系列基本数据类型的变量，包括整数、双精度浮点数和数组指针，并初始化一些变量的默认值。


  if (!PyArg_ParseTuple(args, ("Odd(dd)" F_INT_PYFMT "|Oidd" F_INT_PYFMT),
                        &fcn, &a, &b, &alfa, &beta, &integr, &extra_args, &full_output, &epsabs, &epsrel, &limit)) return NULL;

解析传入的参数 `args`，如果解析失败则返回 `NULL`。


  limit_shape[0] = limit;

  /* Need to check that limit is greater than 1 */
  if (limit < 1)
      return Py_BuildValue(("dd" F_INT_PYFMT),result,abserr,ier);

设置数组 `limit_shape` 的大小，并检查 `limit` 是否大于 1，若不是则返回结果 `result`、`abserr` 和 `ier`。


  ret = init_callback(&callback, fcn, extra_args);
  if (ret == -1) {
      return NULL;
  }

初始化回调函数 `callback`，并检查初始化是否成功。


  /* Set up iwork and work arrays */
  ap_iord = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,F_INT_NPY);
  ap_alist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_blist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_rlist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  ap_elist = (PyArrayObject *)PyArray_SimpleNew(1,limit_shape,NPY_DOUBLE);
  if (ap_iord == NULL || ap_alist == NULL || ap_blist == NULL || ap_rlist == NULL || ap_elist == NULL) goto fail;

创建并初始化 `iord`、`alist`、`blist`、`rlist` 和 `elist` 数组，检查创建是否成功，若有任何一个为空则跳转到 `fail` 标签。


  iord = (F_INT *)PyArray_DATA(ap_iord);
  alist = (double *)PyArray_DATA(ap_alist);
  blist = (double *)PyArray_DATA(ap_blist);
  rlist = (double *)PyArray_DATA(ap_rlist);
  elist = (double *)PyArray_DATA(ap_elist);

获取数组 `ap_iord`、`ap_alist`、`ap_blist`、`ap_rlist` 和 `ap_elist` 的数据指针。


  if (setjmp(callback.error_buf) != 0) {
      goto fail;
  }

设置错误处理点，并在出现错误时跳转到 `fail` 标签。


  DQAWSE(quad_thunk, &a, &b, &alfa, &beta, &integr, &epsabs, &epsrel, &limit, &result, &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);

调用 `DQAWSE` 函数，传入参数执行数值积分计算。


  if (free_callback(&callback) != 0) {
      goto fail_free;
  }

释放回调函数 `callback`，并检查是否成功释放。


  if (full_output) {
      return Py_BuildValue(("dd{s:" F_INT_PYFMT ",s:" F_INT_PYFMT ",s:N,s:N,s:N,s:N,s:N}" F_INT_PYFMT), result, abserr, "neval", neval, "last", last, "iord", PyArray_Return(ap_iord), "alist", PyArray_Return(ap_alist), "blist", PyArray_Return(ap_blist), "rlist", PyArray_Return(ap_rlist), "elist", PyArray_Return(ap_elist), ier);
  }
  else {
    Py_DECREF(ap_alist);
    Py_DECREF(ap_blist);
    Py_DECREF(ap_rlist);
    Py_DECREF(ap_elist);
    Py_DECREF(ap_iord);
    return Py_BuildValue(("dd" F_INT_PYFMT), result, abserr, ier);
  }

根据 `full_output` 的值返回不同的结果：如果为真则返回详细输出，否则只返回简单结果。在返回前释放相应的数组对象。


 fail:
  free_callback(&callback);
 fail_free:
  Py_XDECREF(ap_alist);
  Py_XDECREF(ap_blist);
  Py_XDECREF(ap_rlist);
  Py_XDECREF(ap_elist);
  Py_XDECREF(ap_iord);
  return NULL;
}

处理失败的情况，释放资源并返回 `NULL`。
```