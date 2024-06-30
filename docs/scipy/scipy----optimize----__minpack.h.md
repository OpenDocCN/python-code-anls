# `D:\src\scipysrc\scipy\scipy\optimize\__minpack.h`

```
/* This file is used to make _multipackmodule.c */

/* $Revision$ */

/* module_methods:
  {"_hybrd", minpack_hybrd, METH_VARARGS, doc_hybrd},
  {"_hybrj", minpack_hybrj, METH_VARARGS, doc_hybrj},
  {"_lmdif", minpack_lmdif, METH_VARARGS, doc_lmdif},
  {"_lmder", minpack_lmder, METH_VARARGS, doc_lmder},
  {"_chkder", minpack_chkder, METH_VARARGS, doc_chkder},
 */

/* link libraries:
   minpack
   linpack_lite
   blas
*/

/* python files:
   minpack.py
*/

#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
/* nothing to do in that case */
#else
/* Define aliases for functions with lowercase Fortran naming convention */
#define CHKDER chkder
#define HYBRD  hybrd
#define HYBRJ  hybrj
#define LMDIF  lmdif
#define LMDER  lmder
#define LMSTR  lmstr
#endif
#else
#if defined(UPPERCASE_FORTRAN)
/* Define aliases for functions with uppercase Fortran naming convention */
#define CHKDER CHKDER_
#define HYBRD  HYBRD_
#define HYBRJ  HYBRJ_
#define LMDIF  LMDIF_
#define LMDER  LMDER_
#define LMSTR  LMSTR_
#else
/* Define aliases for functions with lowercase Fortran naming convention */
#define CHKDER chkder_
#define HYBRD  hybrd_
#define HYBRJ  hybrj_
#define LMDIF  lmdif_
#define LMDER  lmder_
#define LMSTR  lmstr_
#endif
#endif

/* Declaration of external Fortran functions */

extern void CHKDER(int*,int*,double*,double*,double*,int*,double*,double*,int*,double*);
extern void HYBRD(void*,int*,double*,double*,double*,int*,int*,int*,double*,double*,int*,double*,int*,int*,int*,double*,int*,double*,int*,double*,double*,double*,double*,double*);
extern void HYBRJ(void*,int*,double*,double*,double*,int*,double*,int*,double*,int*,double*,int*,int*,int*,int*,double*,int*,double*,double*,double*,double*,double*);
extern void LMDIF(void*,int*,int*,double*,double*,double*,double*,double*,int*,double*,double*,int*,double*,int*,int*,int*,double*,int*,int*,double*,double*,double*,double*,double*);
extern void LMDER(void*,int*,int*,double*,double*,double*,int*,double*,double*,double*,int*,double*,int*,double*,int*,int*,int*,int*,int*,double*,double*,double*,double*,double*);
extern void LMSTR(void*,int*,int*,double*,double*,double*,int*,double*,double*,double*,int*,double*,int*,double*,int*,int*,int*,int*,int*,double*,double*,double*,double*,double*);

/* Declaration of ccallback signatures used with Python functions */
static ccallback_signature_t call_signatures[] = {
  {NULL}
};

/* Function to initialize a ccallback */
static int init_callback(ccallback_t *callback, PyObject *fcn, PyObject *extra_args)
{
  int ret;
  int flags = CCALLBACK_OBTAIN;

  /* Prepare the callback with specified function and flags */
  ret = ccallback_prepare(callback, call_signatures, fcn, flags);
  if (ret == -1) {
    return -1;
  }
  
  /* Store extra arguments in the callback's info_p field */
  callback->info_p = (void *)extra_args;

  return 0;
}

/* Function to release a ccallback */
static int release_callback(ccallback_t *callback)
{
  /* Release the callback and return non-zero on failure */
  return ccallback_release(callback) != 0;
}

/* Function to initialize a jacobian ccallback */
static int init_jac_callback(ccallback_t *callback, jac_callback_info_t *jac_callback_info, PyObject *fcn, PyObject *Dfun, PyObject *extra_args, int col_deriv)
{
  int ret;
  int flags = CCALLBACK_OBTAIN;

  /* Prepare the jacobian callback with specified function and flags */
  ret = ccallback_prepare(callback, call_signatures, fcn, flags);
  if (ret == -1) {
    // 如果输入参数 Dfun 为 NULL，返回错误代码 -1
    return -1;
  }

  // 将函数指针 Dfun 和额外参数 extra_args 分配给 jac_callback_info 结构体
  jac_callback_info->Dfun = Dfun;
  jac_callback_info->extra_args = extra_args;
  // 根据 col_deriv 的值设置 jac_transpose，如果 col_deriv 为 false，则 jac_transpose 为 true
  jac_callback_info->jac_transpose = !col_deriv;
  
  // 将 jac_callback_info 结构体的指针转换为 void* 类型，赋值给 callback->info_p
  callback->info_p = (void *)jac_callback_info;

  // 成功分配并设置回调信息后返回成功代码 0
  return 0;
}

# raw_multipack_calling_function函数被调用的接口函数，用于与Fortran代码交互
int raw_multipack_calling_function(int *n, double *x, double *fvec, int *iflag)
{
  /* This is the function called from the Fortran code it should
        -- use call_python_function to get a multiarrayobject result
    -- check for errors and return -1 if any
    -- otherwise place result of calculation in *fvec
  */

  # 从C代码获取回调对象
  ccallback_t *callback = ccallback_obtain();
  # 获取Python中的多维数组函数和额外参数
  PyObject *multipack_python_function = callback->py_function;
  PyObject *multipack_extra_arguments = (PyObject *)callback->info_p;

  PyArrayObject *result_array = NULL;
 
  # 调用Python函数获取多维数组对象作为结果
  result_array = (PyArrayObject *)call_python_function(multipack_python_function, *n, x, multipack_extra_arguments, 1, minpack_error, *n);
  # 如果调用失败，设置iflag为-1并返回-1
  if (result_array == NULL) {
    *iflag = -1;
    return -1;
  }
  # 将结果复制到fvec中
  memcpy(fvec, PyArray_DATA(result_array), (*n)*sizeof(double));
  # 释放结果数组对象
  Py_DECREF(result_array);
  return 0;
}

# jac_multipack_calling_function函数被调用的接口函数，用于与Fortran代码交互
int jac_multipack_calling_function(int *n, double *x, double *fvec, double *fjac, int *ldfjac, int *iflag)
{
  /* This is the function called from the Fortran code it should
        -- use call_python_function to get a multiarrayobject result
    -- check for errors and return -1 if any
    -- otherwise place result of calculation in *fvec or *fjac.

     If iflag = 1 this should compute the function.
     If iflag = 2 this should compute the jacobian (derivative matrix)
  */

  # 从C代码获取回调对象
  ccallback_t *callback = ccallback_obtain();
  # 获取Python中的函数和Jacobi矩阵函数及其额外参数
  PyObject *multipack_python_function = callback->py_function,
           *multipack_python_jacobian = ((jac_callback_info_t *)callback->info_p)->Dfun;
  PyObject *multipack_extra_arguments = ((jac_callback_info_t *)callback->info_p)->extra_args;
  int multipack_jac_transpose = ((jac_callback_info_t *)callback->info_p)->jac_transpose;

  PyArrayObject *result_array;

  # 如果iflag为1，调用Python函数获取结果作为fvec
  if (*iflag == 1) {
    result_array = (PyArrayObject *)call_python_function(multipack_python_function, *n, x, multipack_extra_arguments, 1, minpack_error, *n);
    # 如果调用失败，设置iflag为-1并返回-1
    if (result_array == NULL) {
      *iflag = -1;
      return -1;
    }
    # 将结果复制到fvec中
    memcpy(fvec, PyArray_DATA(result_array), (*n)*sizeof(double));
  }
  else {         /* iflag == 2 */
    # 调用Python函数获取Jacobi矩阵结果
    result_array = (PyArrayObject *)call_python_function(multipack_python_jacobian, *n, x, multipack_extra_arguments, 2, minpack_error, (*n)*(*ldfjac));
    # 如果调用失败，设置iflag为-1并返回-1
    if (result_array == NULL) {
      *iflag = -1;
      return -1;
    }
    # 根据multipack_jac_transpose标志决定是否转置Jacobi矩阵
    if (multipack_jac_transpose == 1)
      MATRIXC2F(fjac, PyArray_DATA(result_array), *n, *ldfjac)
    else
      memcpy(fjac, PyArray_DATA(result_array), (*n)*(*ldfjac)*sizeof(double));
  }

  # 释放结果数组对象
  Py_DECREF(result_array);
  return 0;
}

# raw_multipack_lm_function函数被调用的接口函数，用于与Fortran代码交互
int raw_multipack_lm_function(int *m, int *n, double *x, double *fvec, int *iflag)
{
  /* This is the function called from the Fortran code it should
        -- use call_python_function to get a multiarrayobject result
    -- check for errors and return -1 if any
  */
  /*
  -- otherwise place result of calculation in *fvec
  */

  // 获取回调函数对象
  ccallback_t *callback = ccallback_obtain();
  // 获取多重打包 Python 函数对象
  PyObject *multipack_python_function = callback->py_function;
  // 获取多重打包额外参数对象
  PyObject *multipack_extra_arguments = (PyObject *)callback->info_p;

  // 定义结果数组指针，并初始化为 NULL
  PyArrayObject *result_array = NULL;
 
  // 调用 Python 函数处理，并将结果赋给 result_array
  result_array = (PyArrayObject *)call_python_function(multipack_python_function, *n, x, multipack_extra_arguments, 1, minpack_error, *m);
  // 如果调用失败，设置错误标志并返回 -1
  if (result_array == NULL) {
    *iflag = -1;
    return -1;
  }
  // 将结果数组的数据拷贝至 fvec 数组中
  memcpy(fvec, PyArray_DATA(result_array), (*m)*sizeof(double));
  // 释放结果数组对象
  Py_DECREF(result_array);
  // 返回成功状态
  return 0;
  /*
  -- otherwise place result of calculation in *fvec
  */
}

# 定义一个名为 jac_multipack_lm_function 的函数，该函数作为 Fortran 代码调用的接口
int jac_multipack_lm_function(int *m, int *n, double *x, double *fvec, double *fjac, int *ldfjac, int *iflag)
{
  /* This is the function called from the Fortran code it should
        -- use call_python_function to get a multiarrayobject result
    -- check for errors and return -1 if any
    -- otherwise place result of calculation in *fvec or *fjac.

     If iflag = 1 this should compute the function.
     If iflag = 2 this should compute the jacobian (derivative matrix)
  */

  # 从 ccallback_obtain() 获取回调对象
  ccallback_t *callback = ccallback_obtain();
  # 从回调对象中获取 Python 函数和雅可比函数
  PyObject *multipack_python_function = callback->py_function,
           *multipack_python_jacobian = ((jac_callback_info_t *)callback->info_p)->Dfun;
  # 获取额外参数
  PyObject *multipack_extra_arguments = ((jac_callback_info_t *)callback->info_p)->extra_args;
  # 获取雅可比矩阵是否需要转置的标志
  int multipack_jac_transpose = ((jac_callback_info_t *)callback->info_p)->jac_transpose;

  # 声明一个 PyArrayObject 类型的变量
  PyArrayObject *result_array;

  # 根据 iflag 的值进行不同操作
  if (*iflag == 1) {
    # 调用 Python 函数计算结果数组
    result_array = (PyArrayObject *)call_python_function(multipack_python_function, *n, x, multipack_extra_arguments, 1, minpack_error, *m);
    # 检查是否有错误发生，若有则返回 -1
    if (result_array == NULL) {
      *iflag = -1;
      return -1;
    }
    # 将计算结果拷贝到 fvec 中
    memcpy(fvec, PyArray_DATA(result_array), (*m)*sizeof(double));
  }
  else {         /* iflag == 2 */
    # 调用 Python 雅可比函数计算结果数组
    result_array = (PyArrayObject *)call_python_function(multipack_python_jacobian, *n, x, multipack_extra_arguments, 2, minpack_error, (*n)*(*ldfjac));
    # 检查是否有错误发生，若有则返回 -1
    if (result_array == NULL) {
      *iflag = -1;
      return -1;
    }
    # 根据是否需要转置进行不同操作
    if (multipack_jac_transpose == 1) 
      MATRIXC2F(fjac, PyArray_DATA(result_array), *n, *ldfjac)
    else
      memcpy(fjac, PyArray_DATA(result_array), (*n)*(*ldfjac)*sizeof(double));
  }

  # 释放结果数组对象
  Py_DECREF(result_array);
  # 返回成功标志 0
  return 0;
}

# 定义静态字符数组 doc_hybrd，该数组包含了 _hybrd 函数的文档字符串
static char doc_hybrd[] = "[x,infodict,info] = _hybrd(fun, x0, args, full_output, xtol, maxfev, ml, mu, epsfcn, factor, diag)";
static PyObject *minpack_hybrd(PyObject *dummy, PyObject *args) {
  PyObject *fcn, *x0, *extra_args = NULL, *o_diag = NULL;
  int      full_output = 0, maxfev = -10, ml = -10, mu = -10;
  double   xtol = 1.49012e-8, epsfcn = 0.0, factor = 1.0e2;
  int      mode = 2, nprint = 0, info, nfev, ldfjac;
  npy_intp n,lr;
  int      n_int, lr_int;  /* for casted storage to pass int into HYBRD */
  double   *x, *fvec, *diag, *fjac, *r, *qtf;

  PyArrayObject *ap_x = NULL, *ap_fvec = NULL;
  PyArrayObject *ap_fjac = NULL, *ap_r = NULL, *ap_qtf = NULL;
  PyArrayObject *ap_diag = NULL;

  npy_intp dims[2];
  int      allocated = 0;
  double   *wa = NULL;

  STORE_VARS_NO_INFO();    /* Define storage variables for global variables. */

  // 解析传入的参数
  if (!PyArg_ParseTuple(args, "OO|OidiiiddO", &fcn, &x0, &extra_args, &full_output, &xtol, &maxfev, &ml, &mu, &epsfcn, &factor, &o_diag)) return NULL;

  // 初始化函数，检查是否出错
  INIT_FUNC(fcn,extra_args,minpack_error);

  /* Initial input vector */
  // 将输入向量转换为连续的 NumPy 数组
  ap_x = (PyArrayObject *)PyArray_ContiguousFromObject(x0, NPY_DOUBLE, 1, 1);
  if (ap_x == NULL) goto fail;
  x = (double *) PyArray_DATA(ap_x);
  n = PyArray_DIMS(ap_x)[0];

  // 计算矩阵 r 的长度 lr
  lr = n * (n + 1) / 2;
  if (ml < 0) ml = n-1;
  if (mu < 0) mu = n-1;
  if (maxfev < 0) maxfev = 200*(n+1);

  /* Setup array to hold the function evaluations */
  // 调用 Python 函数计算 fvec 数组
  ap_fvec = (PyArrayObject *)call_python_function(fcn, n, x, extra_args, 1, minpack_error, -1);
  if (ap_fvec == NULL) goto fail;
  fvec = (double *) PyArray_DATA(ap_fvec);
  if (PyArray_NDIM(ap_fvec) == 0)
    n = 1;
  else if (PyArray_DIMS(ap_fvec)[0] < n)
    n = PyArray_DIMS(ap_fvec)[0];

  // 设置对角线 diag 数组
  SET_DIAG(ap_diag,o_diag,mode);

  dims[0] = n; dims[1] = n;
  // 创建数组 ap_r, ap_qtf, ap_fjac
  ap_r = (PyArrayObject *)PyArray_SimpleNew(1,&lr,NPY_DOUBLE);
  ap_qtf = (PyArrayObject *)PyArray_SimpleNew(1,&n,NPY_DOUBLE);
  ap_fjac = (PyArrayObject *)PyArray_SimpleNew(2,dims,NPY_DOUBLE);

  if (ap_r == NULL || ap_qtf == NULL || ap_fjac ==NULL) goto fail;

  r = (double *) PyArray_DATA(ap_r);
  qtf = (double *) PyArray_DATA(ap_qtf);
  fjac = (double *) PyArray_DATA(ap_fjac);
  ldfjac = dims[1];

  if ((wa = malloc(4*n * sizeof(double)))==NULL) {
    PyErr_NoMemory();
    goto fail;
  }
  allocated = 1;

  /* Call the underlying FORTRAN routines. */
  // 调用底层的 FORTRAN 程序 HYBRD
  n_int = n; lr_int = lr; /* cast/store/pass into HYBRD */
  HYBRD(raw_multipack_calling_function, &n_int, x, fvec, &xtol, &maxfev, &ml, &mu, &epsfcn, diag, &mode, &factor, &nprint, &info, &nfev, fjac, &ldfjac, r, &lr_int, qtf, wa, wa+n, wa+2*n, wa+3*n);

  RESTORE_FUNC();

  // 检查返回的信息是否小于 0，如果是则失败
  if (info < 0) goto fail;            /* Python Terminated */

  // 释放内存
  free(wa);
  Py_DECREF(extra_args);
  Py_DECREF(ap_diag);

  // 如果需要完整的输出
  if (full_output) {
    // 返回完整的输出结果
    return Py_BuildValue("N{s:N,s:i,s:N,s:N,s:N}i",PyArray_Return(ap_x),"fvec",PyArray_Return(ap_fvec),"nfev",nfev,"fjac",PyArray_Return(ap_fjac),"r",PyArray_Return(ap_r),"qtf",PyArray_Return(ap_qtf),info);
  }
  else {
    // 释放其余的数组对象
    Py_DECREF(ap_fvec);
    Py_DECREF(ap_fjac);
    Py_DECREF(ap_r);
    Py_DECREF(ap_qtf);
    // 使用 Py_BuildValue 函数构建一个 Python 对象，这里是一个元组 ("Ni")，
    // 包含 PyArray_Return(ap_x) 的返回值和 info 变量的值
    return Py_BuildValue("Ni", PyArray_Return(ap_x), info);
  }

 fail:
  // 进入失败标签，恢复之前的函数调用状态
  RESTORE_FUNC();
 fail_free:
  // 释放所有之前分配的 Python 对象的引用
  Py_XDECREF(extra_args);
  Py_XDECREF(ap_x);
  Py_XDECREF(ap_fvec);
  Py_XDECREF(ap_diag);
  Py_XDECREF(ap_fjac);
  Py_XDECREF(ap_r);
  Py_XDECREF(ap_qtf);
  // 如果之前分配了内存，释放它
  if (allocated) free(wa);
  // 返回 NULL 表示函数执行失败
  return NULL;
/* 定义静态字符数组，描述了 _hybrj 函数的文档字符串 */
static char doc_hybrj[] = "[x,infodict,info] = _hybrj(fun, Dfun, x0, args, full_output, col_deriv, xtol, maxfev, factor, diag)";

/* 定义 minpack_hybrj 函数，用于 Python 到 C 的接口 */
static PyObject *minpack_hybrj(PyObject *dummy, PyObject *args) {
  PyObject *fcn, *Dfun, *x0, *extra_args = NULL, *o_diag = NULL;
  int      full_output = 0, maxfev = -10, col_deriv = 1;
  double   xtol = 1.49012e-8, factor = 1.0e2;
  int      mode = 2, nprint = 0, info, nfev, njev, ldfjac;
  npy_intp n, lr;
  int n_int, lr_int;
  double   *x, *fvec, *diag, *fjac, *r, *qtf;

  PyArrayObject *ap_x = NULL, *ap_fvec = NULL;
  PyArrayObject *ap_fjac = NULL, *ap_r = NULL, *ap_qtf = NULL;
  PyArrayObject *ap_diag = NULL;

  npy_intp dims[2];
  int      allocated = 0;
  double   *wa = NULL;

  /* 定义宏函数 STORE_VARS，用于保存局部变量 */
  STORE_VARS();

  /* 解析传入参数并初始化函数及其导数 */
  if (!PyArg_ParseTuple(args, "OOO|OiididO", &fcn, &Dfun, &x0, &extra_args, &full_output, &col_deriv, &xtol, &maxfev, &factor, &o_diag)) return NULL;

  /* 初始化雅可比矩阵的函数 */
  INIT_JAC_FUNC(fcn,Dfun,extra_args,col_deriv,minpack_error);

  /* 初始化输入向量 */
  ap_x = (PyArrayObject *)PyArray_ContiguousFromObject(x0, NPY_DOUBLE, 1, 1);
  if (ap_x == NULL) goto fail;
  x = (double *) PyArray_DATA(ap_x);
  n = PyArray_DIMS(ap_x)[0];
  lr = n * (n + 1) / 2;

  /* 如果 maxfev 小于 0，则重新计算其值 */
  if (maxfev < 0) maxfev = 100*(n+1);

  /* 设置数组以保存函数的评估结果 */
  ap_fvec = (PyArrayObject *)call_python_function(fcn, n, x, extra_args, 1, minpack_error, -1);
  if (ap_fvec == NULL) goto fail;
  fvec = (double *) PyArray_DATA(ap_fvec);
  if (PyArray_NDIM(ap_fvec) == 0)
    n = 1;
  else if (PyArray_DIMS(ap_fvec)[0] < n)
    n = PyArray_DIMS(ap_fvec)[0];

  /* 设置对角线数组 */
  SET_DIAG(ap_diag,o_diag,mode);

  /* 初始化数组以保存雅可比矩阵、r 向量和 qtf 向量 */
  dims[0] = n; dims[1] = n;
  ap_r = (PyArrayObject *)PyArray_SimpleNew(1,&lr,NPY_DOUBLE);
  ap_qtf = (PyArrayObject *)PyArray_SimpleNew(1,&n,NPY_DOUBLE);
  ap_fjac = (PyArrayObject *)PyArray_SimpleNew(2,dims,NPY_DOUBLE);

  if (ap_r == NULL || ap_qtf == NULL || ap_fjac ==NULL) goto fail;

  /* 获取数组的数据指针 */
  r = (double *) PyArray_DATA(ap_r);
  qtf = (double *) PyArray_DATA(ap_qtf);
  fjac = (double *) PyArray_DATA(ap_fjac);

  ldfjac = dims[1];

  /* 分配内存以保存工作数组 */
  if ((wa = malloc(4*n * sizeof(double)))==NULL) {
    PyErr_NoMemory();
    goto fail;
  }
  allocated = 1;

  /* 调用底层的 FORTRAN 程序 */
  n_int = n; lr_int = lr; /* cast/store/pass into HYBRJ */
  HYBRJ(jac_multipack_calling_function, &n_int, x, fvec, fjac, &ldfjac, &xtol, &maxfev, diag, &mode, &factor, &nprint, &info, &nfev, &njev, r, &lr_int, qtf, wa, wa+n, wa+2*n, wa+3*n);

  /* 恢复雅可比矩阵的函数 */
  RESTORE_JAC_FUNC();

  /* 如果 info 小于 0，则跳转到错误处理 */
  if (info < 0) goto fail;            /* Python Terminated */

  /* 释放工作数组的内存 */
  free(wa);
  Py_DECREF(extra_args);
  Py_DECREF(ap_diag);

  /* 如果 full_output 为真，则返回详细输出 */
  if (full_output) {
    return Py_BuildValue("N{s:N,s:i,s:i,s:N,s:N,s:N}i",PyArray_Return(ap_x),"fvec",PyArray_Return(ap_fvec),"nfev",nfev,"njev",njev,"fjac",PyArray_Return(ap_fjac),"r",PyArray_Return(ap_r),"qtf",PyArray_Return(ap_qtf),info);
  }
  else {
    /* 否则，只返回 x 向量 */
    Py_DECREF(ap_fvec);
    Py_DECREF(ap_fjac);
    Py_DECREF(ap_r);
    Py_DECREF(ap_qtf);
    return Py_BuildValue("Ni",PyArray_Return(ap_x),info);
  }

返回一个 Python 对象，构建为一个元组，包含两个元素：一个是 PyArray
/************************ Levenberg-Marquardt *******************/

// 定义函数文档字符串，描述 _lmdif 函数的输入输出及其作用
static char doc_lmdif[] = "[x,infodict,info] = _lmdif(fun, x0, args, full_output, ftol, xtol, gtol, maxfev, epsfcn, factor, diag)";

// Python C 扩展函数，实现 Levenberg-Marquardt 算法的接口
static PyObject *minpack_lmdif(PyObject *dummy, PyObject *args) {
  // 定义函数输入参数和局部变量
  PyObject *fcn, *x0, *extra_args = NULL, *o_diag = NULL;
  int      full_output = 0, maxfev = -10;
  double   xtol = 1.49012e-8, ftol = 1.49012e-8;
  double   gtol = 0.0, epsfcn = 0.0, factor = 1.0e2;
  int      m, mode = 2, nprint = 0, info = 0, nfev, ldfjac, *ipvt;
  npy_intp n;
  int      n_int;  /* for casted storage to pass int into LMDIF */
  double   *x, *fvec, *diag, *fjac, *qtf;

  // 定义用于存储 Python 数组对象的指针
  PyArrayObject *ap_x = NULL, *ap_fvec = NULL;
  PyArrayObject *ap_fjac = NULL, *ap_ipvt = NULL, *ap_qtf = NULL;
  PyArrayObject *ap_diag = NULL;

  // 定义数组维度
  npy_intp dims[2];
  int      allocated = 0;
  double   *wa = NULL;

  // 宏定义，用于在函数返回出错时释放内存和恢复 Python 函数
  STORE_VARS_NO_INFO();

  // 解析 Python 函数输入参数
  if (!PyArg_ParseTuple(args, "OO|OidddiddO", &fcn, &x0, &extra_args, &full_output, &ftol, &xtol, &gtol, &maxfev, &epsfcn, &factor, &o_diag)) return NULL;

  // 初始化函数及其附加参数
  INIT_FUNC(fcn, extra_args, minpack_error);

  /* Initial input vector */
  // 将输入的初始向量 x0 转换为连续的双精度数组对象
  ap_x = (PyArrayObject *)PyArray_ContiguousFromObject(x0, NPY_DOUBLE, 1, 1);
  if (ap_x == NULL) goto fail;
  x = (double *) PyArray_DATA(ap_x);
  n = PyArray_DIMS(ap_x)[0];
  dims[0] = n;

  // 设置对角线数组 diag
  SET_DIAG(ap_diag, o_diag, mode);

  // 如果 maxfev 小于零，设置其值为 200*(n+1)
  if (maxfev < 0) maxfev = 200 * (n + 1);

  /* Setup array to hold the function evaluations and find it's size*/
  // 调用 Python 函数计算函数向量，并获取其尺寸
  ap_fvec = (PyArrayObject *)call_python_function(fcn, n, x, extra_args, 1, minpack_error, -1);
  if (ap_fvec == NULL) goto fail;
  fvec = (double *) PyArray_DATA(ap_fvec);
  m = (PyArray_NDIM(ap_fvec) > 0 ? PyArray_DIMS(ap_fvec)[0] : 1);

  dims[0] = n; dims[1] = m;
  // 创建包含 Jacobi 矩阵的数组对象
  ap_ipvt = (PyArrayObject *)PyArray_SimpleNew(1, &n, NPY_INT);
  ap_qtf = (PyArrayObject *)PyArray_SimpleNew(1, &n, NPY_DOUBLE);
  ap_fjac = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);

  if (ap_ipvt == NULL || ap_qtf == NULL || ap_fjac == NULL) goto fail;

  ipvt = (int *) PyArray_DATA(ap_ipvt);
  qtf = (double *) PyArray_DATA(ap_qtf);
  fjac = (double *) PyArray_DATA(ap_fjac);
  ldfjac = dims[1];
  wa = (double *) malloc((3 * n + m) * sizeof(double));
  if (wa == NULL) {
    PyErr_NoMemory();
    goto fail;
  }
  allocated = 1;

  // 调用底层 FORTRAN 程序
  n_int = n; /* to provide int*-pointed storage for int argument of LMDIF */
  LMDIF(raw_multipack_lm_function, &m, &n_int, x, fvec, &ftol, &xtol, &gtol, &maxfev, &epsfcn, diag, &mode, &factor, &nprint, &info, &nfev, fjac, &ldfjac, ipvt, qtf, wa, wa + n, wa + 2 * n, wa + 3 * n);

  // 恢复 Python 函数及其附加参数
  RESTORE_FUNC();

  // 如果返回的 info 小于零，触发 Python 错误
  if (info < 0) goto fail;           /* Python error */

  // 释放内存和减少 Python 对象的引用计数
  free(wa);
  Py_DECREF(extra_args);
  Py_DECREF(ap_diag);

  // 如果需要完整输出，返回多个对象构成的 Python 元组
  if (full_output) {
    return Py_BuildValue("N{s:N,s:i,s:N,s:N,s:N}i", PyArray_Return(ap_x), "fvec", PyArray_Return(ap_fvec), "nfev", nfev, "fjac", PyArray_Return(ap_fjac), "ipvt", PyArray_Return(ap_ipvt), "qtf", PyArray_Return(ap_qtf), info);
  }
  else {

    return Py_BuildValue("N", PyArray_Return(ap_x));
    Py_DECREF(ap_fvec);
    Py_DECREF(ap_fjac);
    Py_DECREF(ap_ipvt);
    Py_DECREF(ap_qtf);
    return Py_BuildValue("Ni",PyArray_Return(ap_x),info);
  }

 fail:
  RESTORE_FUNC();
 fail_free:
  Py_XDECREF(extra_args);
  Py_XDECREF(ap_x);
  Py_XDECREF(ap_fvec);
  Py_XDECREF(ap_fjac);
  Py_XDECREF(ap_diag);
  Py_XDECREF(ap_ipvt);
  Py_XDECREF(ap_qtf);
  if (allocated) free(wa);
  return NULL;  



    // 递减对象的引用计数，释放对 ap_fvec 的引用
    Py_DECREF(ap_fvec);
    // 递减对象的引用计数，释放对 ap_fjac 的引用
    Py_DECREF(ap_fjac);
    // 递减对象的引用计数，释放对 ap_ipvt 的引用
    Py_DECREF(ap_ipvt);
    // 递减对象的引用计数，释放对 ap_qtf 的引用
    Py_DECREF(ap_qtf);
    // 使用 Py_BuildValue 构建一个 Python 对象，并返回给调用者
    // 返回一个包含 PyArray_Return(ap_x) 和 info 的元组对象
    return Py_BuildValue("Ni", PyArray_Return(ap_x), info);
  }

 fail:
  // 失败时恢复函数指针
  RESTORE_FUNC();
 fail_free:
  // 释放 extra_args 的引用
  Py_XDECREF(extra_args);
  // 释放 ap_x 的引用
  Py_XDECREF(ap_x);
  // 释放 ap_fvec 的引用
  Py_XDECREF(ap_fvec);
  // 释放 ap_fjac 的引用
  Py_XDECREF(ap_fjac);
  // 释放 ap_diag 的引用
  Py_XDECREF(ap_diag);
  // 释放 ap_ipvt 的引用
  Py_XDECREF(ap_ipvt);
  // 释放 ap_qtf 的引用
  Py_XDECREF(ap_qtf);
  // 如果分配了内存，则释放 wa 指向的内存块
  if (allocated) free(wa);
  // 返回 NULL 表示失败
  return NULL;
/* 结束 minpack_lmder 函数，即 Python 对 C 函数的调用 */
}

/* 定义文档字符串 doc_lmder，用于描述 _lmder 函数的参数和返回值 */
static char doc_lmder[] = "[x,infodict,info] = _lmder(fun, Dfun, x0, args, full_output, col_deriv, ftol, xtol, gtol, maxfev, factor, diag)";

/* 定义 Python C 扩展模块中的 minpack_lmder 函数 */
static PyObject *minpack_lmder(PyObject *dummy, PyObject *args) {
    PyObject *fcn, *x0, *Dfun, *extra_args = NULL, *o_diag = NULL;
    int      full_output = 0, maxfev = -10, col_deriv = 1;
    double   xtol = 1.49012e-8, ftol = 1.49012e-8;
    double   gtol = 0.0, factor = 1.0e2;
    int      m, mode = 2, nprint = 0, info, nfev, njev, ldfjac, *ipvt;
    npy_intp n;
    int n_int;
    double   *x, *fvec, *diag, *fjac, *qtf;

    PyArrayObject *ap_x = NULL, *ap_fvec = NULL;
    PyArrayObject *ap_fjac = NULL, *ap_ipvt = NULL, *ap_qtf = NULL;
    PyArrayObject *ap_diag = NULL;

    npy_intp dims[2];
    int      allocated = 0;
    double   *wa = NULL;

    /* 调用 STORE_VARS 宏，可能是用于保存变量的宏定义 */

    /* 使用 PyArg_ParseTuple 解析 Python 传入的参数，如果解析失败则返回 NULL */
    if (!PyArg_ParseTuple(args, "OOO|OiidddidO", &fcn, &Dfun, &x0, &extra_args, &full_output, &col_deriv, &ftol, &xtol, &gtol, &maxfev, &factor, &o_diag))
        return NULL;

    /* 调用 INIT_JAC_FUNC 宏，可能是用于初始化 Jacobian 函数的宏定义 */

    /* 将输入的 x0 转换为 PyArrayObject 类型的 ap_x，如果转换失败则跳转到 fail 标签 */
    ap_x = (PyArrayObject *)PyArray_ContiguousFromObject(x0, NPY_DOUBLE, 1, 1);
    if (ap_x == NULL) goto fail;
    x = (double *) PyArray_DATA(ap_x);
    n = PyArray_DIMS(ap_x)[0];

    /* 如果 maxfev 小于 0，则将其设为 100*(n+1) */
    if (maxfev < 0) maxfev = 100*(n+1);

    /* 调用 call_python_function 函数，获取函数 fcn 在点 x 处的函数值，保存在 ap_fvec 中 */
    ap_fvec = (PyArrayObject *)call_python_function(fcn, n, x, extra_args, 1, minpack_error, -1);
    if (ap_fvec == NULL) goto fail;
    fvec = (double *) PyArray_DATA(ap_fvec);

    /* 设置对角线矩阵 diag，具体方法由 SET_DIAG 宏决定 */

    /* 计算函数向量 fvec 的长度 m */
    m = (PyArray_NDIM(ap_fvec) > 0 ? PyArray_DIMS(ap_fvec)[0] : 1);

    /* 创建用于存储 ipvt、qtf 和 fjac 的数组 */
    dims[0] = n; dims[1] = m;
    ap_ipvt = (PyArrayObject *)PyArray_SimpleNew(1,&n,NPY_INT);
    ap_qtf = (PyArrayObject *)PyArray_SimpleNew(1,&n,NPY_DOUBLE);
    ap_fjac = (PyArrayObject *)PyArray_SimpleNew(2,dims,NPY_DOUBLE);

    /* 检查数组是否成功创建，如果有任何一个失败则跳转到 fail 标签 */
    if (ap_ipvt == NULL || ap_qtf == NULL || ap_fjac == NULL) goto fail;

    /* 将数组转换为相应的 C 数组 */
    ipvt = (int *) PyArray_DATA(ap_ipvt);
    qtf = (double *) PyArray_DATA(ap_qtf);
    fjac = (double *) PyArray_DATA(ap_fjac);
    ldfjac = dims[1];

    /* 分配工作数组 wa，如果分配失败则抛出内存错误 */
    wa = (double *)malloc((3*n + m)* sizeof(double));
    if (wa == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    allocated = 1;

    /* 调用 LMDER 函数，执行最小化算法的核心部分 */
    n_int = n;
    LMDER(jac_multipack_lm_function, &m, &n_int, x, fvec, fjac, &ldfjac, &ftol, &xtol, &gtol, &maxfev, diag, &mode, &factor, &nprint, &info, &nfev, &njev, ipvt, qtf, wa, wa+n, wa+2*n, wa+3*n);

    /* 调用 RESTORE_JAC_FUNC 宏，可能是用于恢复 Jacobian 函数的宏定义 */

    /* 如果返回的 info 小于 0，则跳转到 fail 标签，可能是出现 Python 错误 */
    if (info < 0) goto fail;

    /* 释放工作数组 wa 的内存 */
    free(wa);
    Py_DECREF(extra_args);
    Py_DECREF(ap_diag);

    /* 如果 full_output 为真，则返回详细输出；否则只返回必要的结果 */
    if (full_output) {
        return Py_BuildValue("N{s:N,s:i,s:i,s:N,s:N,s:N}i",PyArray_Return(ap_x),"fvec",PyArray_Return(ap_fvec),"nfev",nfev,"njev",njev,"fjac",PyArray_Return(ap_fjac),"ipvt",PyArray_Return(ap_ipvt),"qtf",PyArray_Return(ap_qtf),info);
    }
    else {
        /* 释放内存并返回简化的结果 */
        Py_DECREF(ap_fvec);
        Py_DECREF(ap_fjac);
        Py_DECREF(ap_ipvt);
        Py_DECREF(ap_qtf);
        return Py_None;
    }

fail:
    /* 处理错误情况的跳转标签，释放分配的内存和对象引用 */
    if (allocated) free(wa);
    Py_XDECREF(ap_x);
    Py_XDECREF(ap_fvec);
    Py_XDECREF(ap_fjac);
    Py_XDECREF(ap_ipvt);
    Py_XDECREF(ap_qtf);
    Py_XDECREF(ap_diag);
    return NULL;
}
    return Py_BuildValue("Ni",PyArray_Return(ap_x),info);
  }


    // 使用 Py_BuildValue 函数构建一个 Python 对象，格式为 "Ni"
    // 第一个参数 PyArray_Return(ap_x) 返回一个 PyArray 对象
    // 第二个参数 info 是一个整数值
    return Py_BuildValue("Ni", PyArray_Return(ap_x), info);
  }

 fail:
  RESTORE_JAC_FUNC();
 fail_free:
  Py_XDECREF(extra_args);
  Py_XDECREF(ap_x);
  Py_XDECREF(ap_fvec);
  Py_XDECREF(ap_fjac);
  Py_XDECREF(ap_diag);
  Py_XDECREF(ap_ipvt);
  Py_XDECREF(ap_qtf);
  if (allocated) free(wa);
  // 返回空指针表示函数执行失败
  return NULL;  
/** Check gradient function **/

/** 
 * 定义函数的文档字符串，描述了函数 _chkder 的参数和返回值
 **/
static char doc_chkder[] = "_chkder(m,n,x,fvec,fjac,ldfjac,xp,fvecp,mode,err)";

/**
 * minpack_chkder 函数，Python C 扩展中定义的函数，用于检查梯度
 * 
 * 参数:
 *   self: 指向自身的指针，Python C 扩展函数的标准参数
 *   args: Python 传递给函数的参数元组对象
 * 返回值:
 *   返回一个 PyObject 类型的对象，通常为 None
 **/
static PyObject *minpack_chkder(PyObject *self, PyObject *args)
{
  /**
   * 用于保存输入参数的 PyArrayObject 对象，具体是各种数组对象
   **/
  PyArrayObject *ap_fvecp = NULL, *ap_fjac = NULL, *ap_err = NULL;
  PyArrayObject *ap_x = NULL, *ap_fvec = NULL, *ap_xp = NULL;
  /**
   * 用于保存输入参数的 Python 对象
   **/
  PyObject *o_x, *o_fvec, *o_fjac, *o_fvecp;
  /**
   * 用于保存各种数据类型的指针
   **/
  double *xp, *fvecp, *fjac, *fvec, *x;
  double *err;
  int mode, m, n, ldfjac;

  /**
   * 解析 Python 传入的参数元组，按照指定的格式解析各个参数
   **/
  if (!PyArg_ParseTuple(args,"iiOOOiO!OiO!",&m, &n, &o_x, &o_fvec, &o_fjac, &ldfjac, &PyArray_Type, (PyObject **)&ap_xp, &o_fvecp, &mode, &PyArray_Type, (PyObject **)&ap_err)) return NULL;

  /**
   * 将 o_x 转换为 PyArrayObject 类型的 ap_x，且数据类型为 NPY_DOUBLE
   **/
  ap_x = (PyArrayObject *)PyArray_ContiguousFromObject(o_x,NPY_DOUBLE,1,1);
  if (ap_x == NULL) goto fail;
  /**
   * 检查 x 的维度是否为 n
   **/
  if (n != PyArray_DIMS(ap_x)[0])
     PYERR(minpack_error,"Input data array (x) must have length n");
  /**
   * 获取 ap_x 的数据指针，赋值给 x
   **/
  x = (double *) PyArray_DATA(ap_x);
  /**
   * 检查 ap_xp 是否是连续的数组且数据类型为 NPY_DOUBLE
   **/
  if (!PyArray_IS_C_CONTIGUOUS(ap_xp) || (PyArray_TYPE(ap_xp) != NPY_DOUBLE))
     PYERR(minpack_error,"Seventh argument (xp) must be contiguous array of type Float64.");

  /**
   * 根据 mode 的值选择不同的处理分支
   **/
  if (mode == 1) {
    /**
     * 如果 mode 为 1，设置 fvec 和 fjac 为 NULL
     **/
    fvec = NULL;
    fjac = NULL;
    /**
     * 获取 ap_xp 的数据指针，赋值给 xp
     **/
    xp = (double *)PyArray_DATA(ap_xp);
    /**
     * 设置 fvecp 和 err 为 NULL
     **/
    fvecp = NULL;
    err = NULL;
    /**
     * 调用 CHKDER 函数，用于检查梯度
     **/
    CHKDER(&m, &n, x, fvec, fjac, &ldfjac, xp, fvecp, &mode, err);
  }
  else if (mode == 2) {
    /**
     * 如果 mode 为 2，检查 ap_err 是否是连续的数组且数据类型为 NPY_DOUBLE
     **/
    if (!PyArray_IS_C_CONTIGUOUS(ap_err) || (PyArray_TYPE(ap_err) != NPY_DOUBLE))
       PYERR(minpack_error,"Last argument (err) must be contiguous array of type Float64.");
    /**
     * 将 o_fvec, o_fjac, o_fvecp 转换为 PyArrayObject 类型的数组对象
     **/
    ap_fvec = (PyArrayObject *)PyArray_ContiguousFromObject(o_fvec,NPY_DOUBLE,1,1);
    ap_fjac = (PyArrayObject *)PyArray_ContiguousFromObject(o_fjac,NPY_DOUBLE,2,2);
    ap_fvecp = (PyArrayObject *)PyArray_ContiguousFromObject(o_fvecp,NPY_DOUBLE,1,1);
    if (ap_fvec == NULL || ap_fjac == NULL || ap_fvecp == NULL) goto fail;

    /**
     * 获取各个数组对象的数据指针
     **/
    fvec = (double *)PyArray_DATA(ap_fvec);
    fjac = (double *)PyArray_DATA(ap_fjac);
    xp = (double *)PyArray_DATA(ap_xp);
    fvecp = (double *)PyArray_DATA(ap_fvecp);
    err = (double *)PyArray_DATA(ap_err);

    /**
     * 调用 CHKDER 函数，用于检查梯度
     **/
    CHKDER(&m, &n, x, fvec, fjac, &m, xp, fvecp, &mode, err);

    /**
     * 释放临时创建的数组对象
     **/
    Py_DECREF(ap_fvec);
    Py_DECREF(ap_fjac);
    Py_DECREF(ap_fvecp);
  }
  else 
    PYERR(minpack_error,"Invalid mode, must be 1 or 2.");

  /**
   * 释放 ap_x 对象
   **/
  Py_DECREF(ap_x);

  /**
   * 返回 None 对象
   **/
  Py_INCREF(Py_None);
  return Py_None;

 fail:
  /**
   * 如果出现错误，释放所有临时创建的数组对象，并返回 NULL
   **/
  Py_XDECREF(ap_fvec);
  Py_XDECREF(ap_fjac);
  Py_XDECREF(ap_fvecp);
  Py_XDECREF(ap_x);
  return NULL;
}
```