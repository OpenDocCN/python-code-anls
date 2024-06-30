# `D:\src\scipysrc\scipy\scipy\optimize\minpack.h`

```
/* MULTIPACK module by Travis Oliphant

Copyright (c) 2002 Travis Oliphant all rights reserved
Oliphant.Travis@altavista.net
Permission to use, modify, and distribute this software is given under the 
terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
*/

# 包含必要的头文件
#include "Python.h"
#include "numpy/arrayobject.h"
#include "ccallback.h"

# 定义宏，用于抛出错误
#define PYERR(errobj,message) {PyErr_SetString(errobj,message); goto fail;}
#define PYERR2(errobj,message) {PyErr_Print(); PyErr_SetString(errobj, message); goto fail;}

# 定义宏，用于存储变量和回调信息
#define STORE_VARS() ccallback_t callback; int callback_inited = 0; jac_callback_info_t jac_callback_info;
#define STORE_VARS_NO_INFO() ccallback_t callback; int callback_inited = 0;

# 初始化函数宏，用于设置回调函数和额外参数
#define INIT_FUNC(fun,arg,errobj) do { /* Get extra arguments or set to zero length tuple */ \
  if (arg == NULL) { \
    if ((arg = PyTuple_New(0)) == NULL) goto fail_free; \
  } \
  else \
    Py_INCREF(arg);   /* We decrement on exit. */ \
  if (!PyTuple_Check(arg))  \
    PYERR(errobj,"Extra Arguments must be in a tuple"); \
  /* Set up callback functions */ \
  if (!PyCallable_Check(fun)) \
    PYERR(errobj,"First argument must be a callable function."); \
  if (init_callback(&callback, fun, arg) != 0) \
    PYERR(errobj,"Could not init callback");\
  callback_inited = 1; \
  } while(0)

# 初始化雅可比函数宏，用于设置雅可比回调函数和额外参数
#define INIT_JAC_FUNC(fun,Dfun,arg,col_deriv,errobj) do { \
  if (arg == NULL) { \
    if ((arg = PyTuple_New(0)) == NULL) goto fail_free; \
  } \
  else \
    Py_INCREF(arg);   /* We decrement on exit. */ \
  if (!PyTuple_Check(arg))  \
    PYERR(errobj,"Extra Arguments must be in a tuple"); \
  /* Set up callback functions */ \
  if (!PyCallable_Check(fun) || (Dfun != Py_None && !PyCallable_Check(Dfun))) \
    PYERR(errobj,"The function and its Jacobian must be callable functions."); \
  if (init_jac_callback(&callback, &jac_callback_info, fun, Dfun, arg, col_deriv) != 0) \
    PYERR(errobj,"Could not init callback");\
  callback_inited = 1; \
} while(0)
#define RESTORE_JAC_FUNC() do { \
  // 如果回调函数已初始化并且释放回调函数失败，则跳转到fail_free标签处
  if (callback_inited && release_callback(&callback) != 0) { \
    goto fail_free; \
  }} while(0)

#define RESTORE_FUNC() do { \
  // 如果回调函数已初始化并且释放回调函数失败，则跳转到fail_free标签处
  if (callback_inited && release_callback(&callback) != 0) { \
    goto fail_free; \
  }} while(0)

#define SET_DIAG(ap_diag,o_diag,mode) { /* Set the diag vector from input */ \
  // 如果输入的o_diag为NULL或者Py_None，则创建一个新的一维数组对象ap_diag，并设置diag指向其数据，mode设为1
  if (o_diag == NULL || o_diag == Py_None) { \
    ap_diag = (PyArrayObject *)PyArray_SimpleNew(1,&n,NPY_DOUBLE); \
    if (ap_diag == NULL) goto fail; \
    diag = (double *)PyArray_DATA(ap_diag); \
    mode = 1; \
  } \
  // 否则，将输入o_diag转换成连续的一维数组对象ap_diag，并设置diag指向其数据，mode设为2
  else { \
    ap_diag = (PyArrayObject *)PyArray_ContiguousFromObject(o_diag, NPY_DOUBLE, 1, 1); \
    if (ap_diag == NULL) goto fail; \
    diag = (double *)PyArray_DATA(ap_diag); \
    mode = 2; } }

#define MATRIXC2F(jac,data,m,n) {double *p1=(double *)(jac), *p2, *p3=(double *)(data);\
int i,j;\
for (j=0;j<(m);p3++,j++) \
  for (p2=p3,i=0;i<(n);p2+=(m),i++,p1++) \
    // 将data中的列向量转置写入jac中
    *p1 = *p2; }

typedef struct {
  PyObject *Dfun;
  PyObject *extra_args;
  int jac_transpose;
} jac_callback_info_t;

static PyObject *call_python_function(PyObject *func, npy_intp n, double *x, PyObject *args, int dim, PyObject *error_obj, npy_intp out_size)
{
  /*
    This is a generic function to call a python function that takes a 1-D
    sequence as a first argument and optional extra_arguments (should be a
    zero-length tuple if none desired).  The result of the function is 
    returned in a multiarray object.
        -- build sequence object from values in x.
    -- add extra arguments (if any) to an argument list.
    -- call Python callable object
     -- check if error occurred:
             if so return NULL
    -- if no error, place result of Python code into multiarray object.
  */

  PyArrayObject *sequence = NULL;
  PyObject *arglist = NULL;
  PyObject *arg1 = NULL;
  PyObject *result = NULL;
  PyArrayObject *result_array = NULL;
  npy_intp fvec_sz = 0;

  /* Build sequence argument from inputs */
  // 从输入值x创建一个新的一维数组对象sequence
  sequence = (PyArrayObject *)PyArray_SimpleNewFromData(1, &n, NPY_DOUBLE, (char *)x);
  if (sequence == NULL) PYERR2(error_obj,"Internal failure to make an array of doubles out of first\n                 argument to function call.");

  /* Build argument list */
  // 创建一个包含一个元素的元组arg1
  if ((arg1 = PyTuple_New(1)) == NULL) {
    Py_DECREF(sequence);
    return NULL;
  }
  PyTuple_SET_ITEM(arg1, 0, (PyObject *)sequence); 
                /* arg1 now owns sequence reference */
  // 将args和arg1拼接成一个新的元组arglist
  if ((arglist = PySequence_Concat( arg1, args)) == NULL)
    PYERR2(error_obj,"Internal error constructing argument list.");

  Py_DECREF(arg1);    /* arglist has a reference to sequence, now. */
  arg1 = NULL;

  /* Call function object --- variable passed to routine.  Extra
          arguments are in another passed variable.
   */
  // 调用Python函数对象func，传递arglist作为参数列表，将结果存储在result中
  if ((result = PyObject_CallObject(func, arglist))==NULL) {
      goto fail;
  }

  if ((result_array = (PyArrayObject *)PyArray_ContiguousFromObject(result, NPY_DOUBLE, dim-1, dim))==NULL) 
  // 将错误信息设置为指定错误对象的描述信息，并返回 NULL
  PYERR2(error_obj,"Result from function call is not a proper array of floats.");

  // 获取返回数组的大小
  fvec_sz = PyArray_SIZE(result_array);
  // 如果指定了输出大小且返回数组大小与指定的输出大小不相等
  if(out_size != -1 && fvec_sz != out_size){
      // 设置异常并描述异常信息为“函数返回的数组在调用之间发生了大小变化”
      PyErr_SetString(PyExc_ValueError, "The array returned by a function changed size between calls");
      // 释放 result_array 的引用计数
      Py_DECREF(result_array);
      // 跳转到错误处理标签 fail
      goto fail;
  }

  // 释放 result 和 arglist 的引用计数
  Py_DECREF(result);
  Py_DECREF(arglist);
  // 返回转换为 PyObject* 类型的 result_array
  return (PyObject *)result_array;

 fail:
  // 释放 arglist 的引用计数
  Py_XDECREF(arglist);
  // 释放 result 的引用计数
  Py_XDECREF(result);
  // 释放 arg1 的引用计数
  Py_XDECREF(arg1);
  // 返回 NULL，表示错误处理失败
  return NULL;
}



# 这行代码是一个单独的右大括号 '}'，通常用于结束代码块或语句块。
# 在没有上下文的情况下，它本身没有实际作用，需要与代码的其他部分结合起来理解其用途。
```