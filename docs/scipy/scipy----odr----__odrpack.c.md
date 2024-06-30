# `D:\src\scipysrc\scipy\scipy\odr\__odrpack.c`

```
/*
 * Anti-Copyright
 *
 * I hereby release this code into the PUBLIC DOMAIN AS IS. There is no
 * support, warranty, or guarantee. I will gladly accept comments, bug
 * reports, and patches, however.
 *
 * Robert Kern
 * kern@caltech.edu
 *
 */

#define PY_SSIZE_T_CLEAN
#include "odrpack.h"

/* 定义一个双精度的回调函数，用于 DODRC */
void F_FUNC(dodrc,DODRC)(void (*fcn)(F_INT *n, F_INT *m, F_INT *np, F_INT *nq, F_INT *ldn, F_INT *ldm,
            F_INT *ldnp, double *beta, double *xplusd, F_INT *ifixb, F_INT *ifixx,
            F_INT *ldifx, F_INT *ideval, double *f, double *fjacb, double *fjacd,
            F_INT *istop),
           F_INT *n, F_INT *m, F_INT *np, F_INT *nq, double *beta, double *y, F_INT *ldy,
           double *x, F_INT *ldx, double *we, F_INT *ldwe, F_INT *ld2we, double *wd,
           F_INT *ldwd, F_INT *ld2wd, F_INT *ifixb, F_INT *ifixx, F_INT *ldifx, F_INT *job,
           F_INT *ndigit, double *taufac, double *sstol, double *partol,
           F_INT *maxit, F_INT *iprint, F_INT *lunerr, F_INT *lunrpt, double *stpb,
           double *stpd, F_INT *ldstpd, double *sclb, double *scld, F_INT *ldscld,
           double *work, F_INT *lwork, F_INT *iwork, F_INT *liwork, F_INT *info) {
    /* 实现代码省略 */
}

/* 定义一个整型函数，用于计算 ODR 参数信息 */
void F_FUNC(dwinf,DWINF)(F_INT *n, F_INT *m, F_INT *np, F_INT *nq, F_INT *ldwe, F_INT *ld2we, F_INT *isodr,
        F_INT *delta, F_INT *eps, F_INT *xplus, F_INT *fn, F_INT *sd, F_INT *vcv, F_INT *rvar,
        F_INT *wss, F_INT *wssde, F_INT *wssep, F_INT *rcond, F_INT *eta, F_INT *olmav,
        F_INT *tau, F_INT *alpha, F_INT *actrs, F_INT *pnorm, F_INT *rnors, F_INT *prers,
        F_INT *partl, F_INT *sstol, F_INT *taufc, F_INT *apsma, F_INT *betao, F_INT *betac,
        F_INT *betas, F_INT *betan, F_INT *s, F_INT *ss, F_INT *ssf, F_INT *qraux, F_INT *u,
        F_INT *fs, F_INT *fjacb, F_INT *we1, F_INT *diff, F_INT *delts, F_INT *deltn,
        F_INT *t, F_INT *tt, F_INT *omega, F_INT *fjacd, F_INT *wrk1, F_INT *wrk2,
        F_INT *wrk3, F_INT *wrk4, F_INT *wrk5, F_INT *wrk6, F_INT *wrk7, F_INT *lwkmn) {
    /* 实现代码省略 */
}

/* 定义一个函数，用于打开文件并绑定到特定的 I/O 缓冲区 */
void F_FUNC(dluno,DLUNO)(F_INT *lun, char *fn, int fnlen) {
    /* 实现代码省略 */
}

/* 定义一个函数，用于关闭文件 */
void F_FUNC(dlunc,DLUNC)(F_INT *lun) {
    /* 实现代码省略 */
}

/* 回调函数，用于调用全局结构 |odr_global| 中的 Python 函数 */
void fcn_callback(F_INT *n, F_INT *m, F_INT *np, F_INT *nq, F_INT *ldn, F_INT *ldm,
                  F_INT *ldnp, double *beta, double *xplusd, F_INT *ifixb,
                  F_INT *ifixx, F_INT *ldfix, F_INT *ideval, double *f,
                  double *fjacb, double *fjacd, F_INT *istop) {
    PyObject *arg01, *arglist;
    PyObject *result = NULL;
    PyArrayObject *result_array = NULL;
    PyArrayObject *pyXplusD;
    void *beta_dst;

    /* 如果 m 不等于 1，则创建一个二维的 PyArrayObject 对象 */
    if (*m != 1) {
        npy_intp dim2[2];
        dim2[0] = *m;
        dim2[1] = *n;
        pyXplusD = (PyArrayObject *) PyArray_SimpleNew(2, dim2, NPY_DOUBLE);
        /* 将 xplusd 中的数据复制到 PyArrayObject 对象中 */
        memcpy(PyArray_DATA(pyXplusD), (void *)xplusd, (*m) * (*n) * sizeof(double));
    } else {
        /* 实现代码省略 */
    }
}
    {
      npy_intp dim1[1];  // 定义一个整型数组，用于存储维度信息
      dim1[0] = *n;  // 将指针 n 指向的值赋给 dim1 的第一个元素，即设定数组的大小
      pyXplusD = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);  // 创建一个新的一维 NumPy 数组对象，存储类型为双精度浮点型
      memcpy(PyArray_DATA(pyXplusD), (void *)xplusd, (*n) * sizeof(double));  // 将数组 xplusd 的内容复制到 pyXplusD 对应的数据区域
    }

  arg01 = PyTuple_Pack(2, odr_global.pyBeta, (PyObject *) pyXplusD);  // 创建一个包含两个元素的元组 arg01，第一个元素是 odr_global.pyBeta，第二个是 pyXplusD
  Py_DECREF(pyXplusD);  // 减少 pyXplusD 的引用计数

  if (arg01 == NULL) {  // 检查 arg01 是否为 NULL
    return;  // 如果是，则直接返回
  }

  if (odr_global.extra_args != NULL)
      arglist = PySequence_Concat(arg01, odr_global.extra_args);  // 如果存在额外的参数，将它们拼接到 arg01 中
  else
      arglist = PySequence_Tuple(arg01);        /* make a copy */  // 否则，创建 arg01 的一个副本作为 arglist
  Py_DECREF(arg01);  // 减少 arg01 的引用计数

  *istop = 0;  // 将 istop 指向的值设置为 0

  beta_dst = (PyArray_DATA((PyArrayObject *) odr_global.pyBeta));  // 获取 odr_global.pyBeta 对应的 NumPy 数组的数据区域的指针
  if (beta != beta_dst) {  // 检查 beta 是否与 beta_dst 指向的数据区域相同
      memcpy(beta_dst, (void *)beta, (*np) * sizeof(double));  // 如果不同，则将 beta 指向的数据复制到 beta_dst 指向的数据区域
  }

  if ((*ideval % 10) >= 1)
    {
      /* compute f with odr_global.fcn */

      if (odr_global.fcn == NULL)
        {
          /* we don't have a function to call */
          PYERR2(odr_error, "Function has not been initialized");  // 如果 odr_global.fcn 为 NULL，则抛出一个带有错误消息 "Function has not been initialized" 的异常
        }

      if ((result = PyObject_CallObject(odr_global.fcn, arglist)) == NULL)
        {
          if (PyErr_ExceptionMatches(odr_stop))
            {
              /* stop, don't fail */
              *istop = 1;  // 设置 istop 指向的值为 1，表示停止计算

              Py_DECREF(arglist);  // 减少 arglist 的引用计数
              return;  // 直接返回
            }
          goto fail;  // 如果调用失败，则跳转到 fail 标签处处理错误
        }

      if ((result_array =
           (PyArrayObject *) PyArray_ContiguousFromObject(result,
                                                          NPY_DOUBLE, 0,
                                                          2)) == NULL)
        {
          PYERR2(odr_error,
                 "Result from function call is not a proper array of floats.");  // 如果无法从 result 创建一个连续的 NumPy 数组对象，则抛出一个带有错误消息 "Result from function call is not a proper array of floats." 的异常
        }

      memcpy((void *)f, PyArray_DATA(result_array), (*n) * (*nq) * sizeof(double));  // 将 result_array 对应的数据复制到 f 指向的数据区域
      Py_DECREF(result_array);  // 减少 result_array 的引用计数
    }

  if (((*ideval) / 10) % 10 >= 1)
    {
      /* 计算 fjacb，使用 odr_global.fjacb */

      if (odr_global.fjacb == NULL)
        {
          /* 如果函数指针为空，表示未初始化 */
          PYERR2(odr_error, "Function has not been initialized");
        }

      /* 调用 odr_global.fjacb 函数对象 */
      if ((result = PyObject_CallObject(odr_global.fjacb, arglist)) == NULL)
        {
          /* 如果调用失败 */
          if (PyErr_ExceptionMatches(odr_stop))
            {
              /* 若遇到停止异常，设置 istop 标志 */
              *istop = 1;

              Py_DECREF(arglist);
              return;
            }
          goto fail;
        }

      /* 将调用结果转换为连续存储的 NPY_DOUBLE 类型的 PyArrayObject */
      if ((result_array =
           (PyArrayObject *) PyArray_ContiguousFromObject(result,
                                                          NPY_DOUBLE, 0,
                                                          3)) == NULL)
        {
          /* 如果结果不是合适的浮点数数组 */
          PYERR2(odr_error,
                 "Result from function call is not a proper array of floats.");
        }

      /* 检查结果数组的维度是否符合预期 */
      if (*nq != 1 && *np != 1)
        {
          /* 如果 *nq 和 *np 不同时为 1，结果数组应为三维 */
          if (PyArray_NDIM(result_array) != 3)
            {
              Py_DECREF(result_array);
              PYERR2(odr_error, "Beta Jacobian is not rank-3");
            }
        }
      else if (*nq == 1)
        {
          /* 如果 *nq 为 1，结果数组应为二维 */
          if (PyArray_NDIM(result_array) != 2)
            {
              Py_DECREF(result_array);
              PYERR2(odr_error, "Beta Jacobian is not rank-2");
            }
        }

      /* 将结果数组的数据复制到 fjacb 数组中 */
      memcpy((void *)fjacb, PyArray_DATA(result_array),
             (*n) * (*nq) * (*np) * sizeof(double));
      Py_DECREF(result_array);

    }

  /* 检查 ideval 是否满足特定条件 */
  if (((*ideval) / 100) % 10 >= 1)
    {
      /* 使用全局变量 odr_global.fjacd 计算 fjacd */
    
      if (odr_global.fjacd == NULL)
      {
        /* 如果未初始化 fjcad 函数，则报错 */
        PYERR2(odr_error, "fjcad has not been initialized");
      }
    
      /* 调用 odr_global.fjacd 函数，并获取返回结果 */
      if ((result = PyObject_CallObject(odr_global.fjacd, arglist)) == NULL)
      {
        /* 处理函数调用异常情况 */
        if (PyErr_ExceptionMatches(odr_stop))
        {
          /* 如果遇到 odr_stop 异常，设置 istop 为 1，并返回 */
          *istop = 1;
    
          Py_DECREF(arglist);
          return;
        }
        /* 其他异常情况转到 fail 标签处理 */
        goto fail;
      }
    
      /* 将返回的结果转换为连续的双精度浮点数数组 */
      if ((result_array =
           (PyArrayObject *) PyArray_ContiguousFromObject(result,
                                                          NPY_DOUBLE, 0,
                                                          3)) == NULL)
      {
        /* 如果转换失败，则报错 */
        PYERR2(odr_error,
               "Result from function call is not a proper array of floats.");
      }
    
      /* 根据不同的条件检查结果数组的维度 */
      if (*nq != 1 && *m != 1)
      {
        /* 当 *nq 和 *m 不同时，result_array 应为三维 */
        if (PyArray_NDIM(result_array) != 3)
        {
          /* 如果不是三维数组，则报错 */
          Py_DECREF(result_array);
          PYERR2(odr_error, "xplusd Jacobian is not rank-3");
        }
      }
      else if (*nq == 1 && *m != 1)
      {
        /* 当 *nq 为 1 而 *m 不为 1 时，result_array 应为二维 */
        if (PyArray_NDIM(result_array) != 2)
        {
          /* 如果不是二维数组，则报错 */
          Py_DECREF(result_array);
          PYERR2(odr_error, "xplusd Jacobian is not rank-2");
        }
      }
      else if (*nq == 1 && *m == 1)
      {
        /* 当 *nq 和 *m 都为 1 时，result_array 应为一维 */
        if (PyArray_NDIM(result_array) != 1)
        {
          /* 如果不是一维数组，则报错 */
          Py_DECREF(result_array);
          PYERR2(odr_error, "xplusd Jacobian is not rank-1");
        }
      }
    
      /* 将 result_array 的数据拷贝到 fjacd 数组中 */
      memcpy((void *)fjacd, PyArray_DATA(result_array),
             (*n) * (*nq) * (*m) * sizeof(double));
      Py_DECREF(result_array);
    }
    
    /* 释放资源并返回 */
    Py_DECREF(result);
    Py_DECREF(arglist);
    
    return;
/* 释放结果对象和参数列表对象的引用计数 */
Py_XDECREF(result);
Py_XDECREF(arglist);

/* 设置istop指针指向的值为-1，表示程序异常终止 */
*istop = -1;

/* 函数结束，无返回值 */
return;
}

/* 从DODRC的原始输出生成Python输出 */
PyObject *gen_output(F_INT n, F_INT m, F_INT np, F_INT nq, F_INT ldwe, F_INT ld2we,
                     PyArrayObject * beta, PyArrayObject * work,
                     PyArrayObject * iwork, F_INT isodr, F_INT info,
                     int full_output)
{
  PyArrayObject *sd_beta, *cov_beta;

  F_INT delta, eps, xplus, fn, sd, vcv, rvar, wss, wssde, wssep, rcond;
  F_INT eta, olmav, tau, alpha, actrs, pnorm, rnors, prers, partl, sstol;
  F_INT taufc, apsma, betao, betac, betas, betan, s, ss, ssf, qraux, u;
  F_INT fs, fjacb, we1, diff, delts, deltn, t, tt, omega, fjacd;
  F_INT wrk1, wrk2, wrk3, wrk4, wrk5, wrk6, wrk7, lwkmn;

  PyObject *retobj;

  npy_intp dim1[1], dim2[2];

  /* 检查info值，如果为50005，表示函数调用中出现致命错误，返回NULL以传播异常 */
  if (info == 50005) {
      return NULL;
  }

  /* 获取work数组的第一个维度大小，存入lwkmn变量 */
  lwkmn = PyArray_DIMS(work)[0];

  /* 调用F_FUNC(dwinf,DWINF)函数，填充大量变量 */
  F_FUNC(dwinf,DWINF)(&n, &m, &np, &nq, &ldwe, &ld2we, &isodr,
        &delta, &eps, &xplus, &fn, &sd, &vcv, &rvar, &wss, &wssde,
        &wssep, &rcond, &eta, &olmav, &tau, &alpha, &actrs, &pnorm,
        &rnors, &prers, &partl, &sstol, &taufc, &apsma, &betao, &betac,
        &betas, &betan, &s, &ss, &ssf, &qraux, &u, &fs, &fjacb, &we1,
        &diff, &delts, &deltn, &t, &tt, &omega, &fjacd, &wrk1, &wrk2,
        &wrk3, &wrk4, &wrk5, &wrk6, &wrk7, &lwkmn);

  /* 将FORTRAN风格的索引转换为C风格的索引 */
  delta--;
  eps--;
  xplus--;
  fn--;
  sd--;
  vcv--;
  rvar--;
  wss--;
  wssde--;
  wssep--;
  rcond--;
  eta--;
  olmav--;
  tau--;
  alpha--;
  actrs--;
  pnorm--;
  rnors--;
  prers--;
  partl--;
  sstol--;
  taufc--;
  apsma--;
  betao--;
  betac--;
  betas--;
  betan--;
  s--;
  ss--;
  ssf--;
  qraux--;
  u--;
  fs--;
  fjacb--;
  we1--;
  diff--;
  delts--;
  deltn--;
  t--;
  tt--;
  omega--;
  fjacd--;
  wrk1--;
  wrk2--;
  wrk3--;
  wrk4--;
  wrk5--;
  wrk6--;
  wrk7--;

  /* 设置sd_beta和cov_beta的维度 */
  dim1[0] = PyArray_DIMS(beta)[0];
  sd_beta = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
  dim2[0] = PyArray_DIMS(beta)[0];
  dim2[1] = PyArray_DIMS(beta)[0];
  cov_beta = (PyArrayObject *) PyArray_SimpleNew(2, dim2, NPY_DOUBLE);

  /* 从work数组中复制数据到sd_beta和cov_beta */
  memcpy(PyArray_DATA(sd_beta), (void *)((double *)(PyArray_DATA(work)) + sd),
         np * sizeof(double));
  memcpy(PyArray_DATA(cov_beta), (void *)((double *)(PyArray_DATA(work)) + vcv),
         np * np * sizeof(double));

  /* 如果不需要完整输出，则构建包含beta、sd_beta和cov_beta的元组返回 */
  if (!full_output) {
      retobj = Py_BuildValue("OOO", PyArray_Return(beta),
                             PyArray_Return(sd_beta),
                             PyArray_Return(cov_beta));
      Py_DECREF((PyObject *) sd_beta);
      Py_DECREF((PyObject *) cov_beta);

      return retobj;
  } else {
      /* 如果需要完整输出，则省略具体实现 */
  }
}
{
  // 定义 Python 对象指针，用于存储函数和各种参数
  PyObject *fcn, *initbeta, *py, *px, *pwe = NULL, *pwd = NULL, *fjacb = NULL;
  PyObject *fjacd = NULL, *pifixb = NULL, *pifixx = NULL;
  PyObject *pstpb = NULL, *pstpd = NULL, *psclb = NULL, *pscld = NULL;
  PyObject *pwork = NULL, *piwork = NULL, *extra_args = NULL;
  // 定义整型变量和双精度浮点数，设置默认值
  F_INT job = 0, ndigit = 0, maxit = -1, iprint = 0;
  int full_output = 0;
  double taufac = 0.0, sstol = -1.0, partol = -1.0;
  // 定义文件名字符串指针，并初始化为 NULL
  char *errfile = NULL, *rptfile = NULL;
  // 定义文件名字符串长度变量，初始化为 0
  Py_ssize_t lerrfile = 0, lrptfile = 0;
  // 定义 NumPy 数组对象指针，用于存储数据
  PyArrayObject *beta = NULL, *y = NULL, *x = NULL, *we = NULL, *wd = NULL;
  PyArrayObject *ifixb = NULL, *ifixx = NULL;
  PyArrayObject *stpb = NULL, *stpd = NULL, *sclb = NULL, *scld = NULL;
  PyArrayObject *work = NULL, *iwork = NULL;
  // 定义整型变量，用于表示数据的维度和长度等信息
  F_INT n, m, np, nq, ldy, ldx, ldwe, ld2we, ldwd, ld2wd, ldifx;
  F_INT lunerr = -1, lunrpt = -1, ldstpd, ldscld, lwork, liwork, info = 0;
  // 定义静态字符指针数组，用于指定关键字参数名
  static char *kw_list[] = { "fcn", "initbeta", "y", "x", "we", "wd", "fjacb",
    "fjacd", "extra_args", "ifixb", "ifixx", "job", "iprint", "errfile",
    "rptfile", "ndigit", "taufac", "sstol", "partol",
    "maxit", "stpb", "stpd", "sclb", "scld", "work",
    "iwork", "full_output", NULL
  };
  // 定义整型变量，用于表示模型是否是隐式模型的标志
  F_INT isodr = 1;
  // 定义 Python 对象指针，用于存储函数调用的结果
  PyObject *result;
  // 定义 NumPy 数组的维度信息数组
  npy_intp dim1[1], dim2[2], dim3[3];
  // 定义整型变量，用于表示模型是否是隐式模型的标志
  F_INT implicit; /* flag for implicit model */

  // 如果关键字参数为空，则使用非关键字方式解析参数
  if (kwds == NULL)
    {
        // 解析传入参数为函数和各种参数对象，并设置各个变量的值
        if (!PyArg_ParseTuple(args, ("OOOO|OOOOOOO" F_INT_PYFMT F_INT_PYFMT
                                     "z#z#" F_INT_PYFMT "ddd" F_INT_PYFMT
                                     "OOOOOOi:odr"),
                            &fcn, &initbeta, &py, &px, &pwe, &pwd,
                            &fjacb, &fjacd, &extra_args, &pifixb, &pifixx,
                            &job, &iprint, &errfile, &lerrfile, &rptfile,
                            &lrptfile, &ndigit, &taufac, &sstol, &partol,
                            &maxit, &pstpb, &pstpd, &psclb, &pscld, &pwork,
                            &piwork, &full_output))
        {
          return NULL;
        }
    }
  else
    {
      // 解析传入参数为函数和各种参数对象，并设置各个变量的值，使用关键字方式解析
      if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                       ("OOOO|OOOOOOO" F_INT_PYFMT "" F_INT_PYFMT
                                        "z#z#" F_INT_PYFMT "ddd" F_INT_PYFMT
                                        "OOOOOOi:odr"),
                                       kw_list, &fcn, &initbeta, &py, &px,
                                       &pwe, &pwd, &fjacb, &fjacd,
                                       &extra_args, &pifixb, &pifixx, &job,
                                       &iprint, &errfile, &lerrfile, &rptfile,
                                       &lrptfile, &ndigit, &taufac, &sstol,
                                       &partol, &maxit, &pstpb, &pstpd,
                                       &psclb, &pscld, &pwork, &piwork,
                                       &full_output))
        {
          return NULL;
        }
    }

  // 检查函数对象是否可调用
  if (!PyCallable_Check(fcn))
    {
      PYERR(PyExc_TypeError, "fcn must be callable");
    }
    如果参数 `fcn` 不是可调用对象，则抛出类型错误异常。
    
    if (!PySequence_Check(initbeta))
    {
      PYERR(PyExc_TypeError, "initbeta must be a sequence");
    }
    如果参数 `initbeta` 不是序列类型，则抛出类型错误异常。
    
    if (!PySequence_Check(py) && !PyNumber_Check(py))
    {
      PYERR(PyExc_TypeError,
            "y must be a sequence or integer (if model is implicit)");
    }
    如果参数 `py` 不是序列类型且也不是整数类型（在模型为隐式时），则抛出类型错误异常。
    
    if (!PySequence_Check(px))
    {
      PYERR(PyExc_TypeError, "x must be a sequence");
    }
    如果参数 `px` 不是序列类型，则抛出类型错误异常。
    
    if (pwe != NULL && !PySequence_Check(pwe) && !PyNumber_Check(pwe))
    {
      PYERR(PyExc_TypeError, "we must be a sequence or a number");
    }
    如果参数 `pwe` 不为 NULL 且既不是序列类型也不是数值类型，则抛出类型错误异常。
    
    if (pwd != NULL && !PySequence_Check(pwd) && !PyNumber_Check(pwd))
    {
      PYERR(PyExc_TypeError, "wd must be a sequence or a number");
    }
    如果参数 `pwd` 不为 NULL 且既不是序列类型也不是数值类型，则抛出类型错误异常。
    
    if (fjacb != NULL && !PyCallable_Check(fjacb))
    {
      PYERR(PyExc_TypeError, "fjacb must be callable");
    }
    如果参数 `fjacb` 不为 NULL 且不是可调用对象，则抛出类型错误异常。
    
    if (fjacd != NULL && !PyCallable_Check(fjacd))
    {
      PYERR(PyExc_TypeError, "fjacd must be callable");
    }
    如果参数 `fjacd` 不为 NULL 且不是可调用对象，则抛出类型错误异常。
    
    if (extra_args != NULL && !PySequence_Check(extra_args))
    {
      PYERR(PyExc_TypeError, "extra_args must be a sequence");
    }
    如果参数 `extra_args` 不为 NULL 且不是序列类型，则抛出类型错误异常。
    
    if (pifixx != NULL && !PySequence_Check(pifixx))
    {
      PYERR(PyExc_TypeError, "ifixx must be a sequence");
    }
    如果参数 `pifixx` 不为 NULL 且不是序列类型，则抛出类型错误异常。
    
    if (pifixb != NULL && !PySequence_Check(pifixb))
    {
      PYERR(PyExc_TypeError, "ifixb must be a sequence");
    }
    如果参数 `pifixb` 不为 NULL 且不是序列类型，则抛出类型错误异常。
    
    if (pstpb != NULL && !PySequence_Check(pstpb))
    {
      PYERR(PyExc_TypeError, "stpb must be a sequence");
    }
    如果参数 `pstpb` 不为 NULL 且不是序列类型，则抛出类型错误异常。
    
    if (pstpd != NULL && !PySequence_Check(pstpd))
    {
      PYERR(PyExc_TypeError, "stpd must be a sequence");
    }
    如果参数 `pstpd` 不为 NULL 且不是序列类型，则抛出类型错误异常。
    
    if (psclb != NULL && !PySequence_Check(psclb))
    {
      PYERR(PyExc_TypeError, "sclb must be a sequence");
    }
    如果参数 `psclb` 不为 NULL 且不是序列类型，则抛出类型错误异常。
    
    if (pscld != NULL && !PySequence_Check(pscld))
    {
      PYERR(PyExc_TypeError, "scld must be a sequence");
    }
    如果参数 `pscld` 不为 NULL 且不是序列类型，则抛出类型错误异常。
    
    if (pwork != NULL && !PyArray_Check(pwork))
    {
      PYERR(PyExc_TypeError, "work must be an array");
    }
    如果参数 `pwork` 不为 NULL 且不是数组类型，则抛出类型错误异常。
    
    if (piwork != NULL && !PyArray_Check(piwork))
    {
      PYERR(PyExc_TypeError, "iwork must be an array");
    }
    如果参数 `piwork` 不为 NULL 且不是数组类型，则抛出类型错误异常。
    
    /* start processing the arguments and check for errors on the way */
    
    /* check for implicit model */
    
    implicit = (job % 10 == 1);
    判断是否为隐式模型，通过取模运算 `job % 10 == 1`。
    
    if (!implicit)
    {
      // 如果没有显式的模型
      if ((y =
           (PyArrayObject *) PyArray_CopyFromObject(py, NPY_DOUBLE, 1,
                                                    2)) == NULL)
        {
          // 报错：y 无法转换成适当的数组
          PYERR(PyExc_ValueError,
                "y could not be made into a suitable array");
        }
      // 获取数组 y 的最后一个维度的大小
      n = PyArray_DIMS(y)[PyArray_NDIM(y) - 1];     /* pick the last dimension */
      // 如果没有显式的模型
      if ((x =
           (PyArrayObject *) PyArray_CopyFromObject(px, NPY_DOUBLE, 1,
                                                    2)) == NULL)
        {
          // 报错：x 无法转换成适当的数组
          PYERR(PyExc_ValueError,
                "x could not be made into a suitable array");
        }
      // 如果 x 和 y 的观测数量不匹配
      if (n != PyArray_DIMS(x)[PyArray_NDIM(x) - 1])
        {
          // 报错：x 和 y 的观测数量不匹配
          PYERR(PyExc_ValueError,
                "x and y don't have matching numbers of observations");
        }
      // 如果数组 y 是一维的
      if (PyArray_NDIM(y) == 1)
        {
          // 设置 nq 为 1
          nq = 1;
        }
      else
        {
          // 否则，设置 nq 为数组 y 的第一个维度大小
          nq = PyArray_DIMS(y)[0];
        }

      // 设置 ldx 和 ldy 的值为 n
      ldx = ldy = n;
    }
  else
    {                           /* we *do* have an implicit model */
      // 否则，我们有一个隐式模型
      ldy = 1;
      // 将 py 转换为长整型，并将结果赋给 nq
      nq = (F_INT)PyLong_AsLong(py);
      dim1[0] = 1;

      /* 初始化 y 为一个虚拟数组；从未被引用 */
      // 创建一个维度为 1 的双精度浮点数数组 y
      y = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);

      // 如果没有显式的模型
      if ((x =
           (PyArrayObject *) PyArray_CopyFromObject(px, NPY_DOUBLE, 1,
                                                    2)) == NULL)
        {
          // 报错：x 无法转换成适当的数组
          PYERR(PyExc_ValueError,
                "x could not be made into a suitable array");
        }

      // 获取数组 x 的最后一个维度的大小
      n = PyArray_DIMS(x)[PyArray_NDIM(x) - 1];
      // 设置 ldx 的值为 n
      ldx = n;
    }

  // 如果数组 x 是一维的
  if (PyArray_NDIM(x) == 1)
    {
      // 设置 m 为 1
      m = 1;
    }
  else
    {
      // 否则，设置 m 为数组 x 的第一个维度大小
      m = PyArray_DIMS(x)[0];
    }                           /* x, y */

  // 将 initbeta 转换为双精度浮点数数组 beta，并将结果赋给 beta
  if ((beta =
       (PyArrayObject *) PyArray_CopyFromObject(initbeta, NPY_DOUBLE, 1,
                                                1)) == NULL)
    {
      // 报错：initbeta 无法转换成适当的数组
      PYERR(PyExc_ValueError,
            "initbeta could not be made into a suitable array");
    }
  // 获取数组 beta 的第一个维度的大小，并将结果赋给 np
  np = PyArray_DIMS(beta)[0];

  // 如果 pwe 为 NULL
  if (pwe == NULL)
    {
      // 设置 ldwe 和 ld2we 的值为 1
      ldwe = ld2we = 1;
      // 创建一个维度为 n 的双精度浮点数数组 we，并将其第一个元素设置为 -1.0
      dim1[0] = n;
      we = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
      ((double *)(PyArray_DATA(we)))[0] = -1.0;
    }
  // 否则，如果 pwe 是一个数字并且不是数组
  else if (PyNumber_Check(pwe) && !PyArray_Check(pwe))
    {
      /* we is a single weight, set the first value of we to -pwe */
      // 将 pwe 转换为浮点数，并将结果赋给 val
      PyObject *tmp;
      double val;

      tmp = PyNumber_Float(pwe);
      if (tmp == NULL)
        // 报错：无法将 pwe 转换为适当的数组
        PYERR(PyExc_ValueError, "could not convert we to a suitable array");
      val = PyFloat_AsDouble(tmp);
      Py_DECREF(tmp);

      // 设置维度数组 dim3 的各个维度大小
      dim3[0] = nq;
      dim3[1] = 1;
      dim3[2] = 1;
      // 创建一个三维双精度浮点数数组 we，并根据 implicit 的值设置其第一个元素
      we = (PyArrayObject *) PyArray_SimpleNew(3, dim3, NPY_DOUBLE);
      if (implicit)
        {
          ((double *)(PyArray_DATA(we)))[0] = val;
        }
      else
        {
          ((double *)(PyArray_DATA(we)))[0] = -val;
        }
      // 设置 ldwe 和 ld2we 的值为 1
      ldwe = ld2we = 1;
    }
  else if (PySequence_Check(pwe))
    {
      /* 将 pwe 转换为一个数组 */

      // 将 Python 对象 pwe 转换为双精度浮点类型的一维 PyArrayObject 数组，存储在 we 中
      if ((we =
           (PyArrayObject *) PyArray_CopyFromObject(pwe, NPY_DOUBLE, 1,
                                                    3)) == NULL)
        {
          // 如果转换失败，抛出 ValueError 异常
          PYERR(PyExc_ValueError, "could not convert we to a suitable array");
        }

      // 如果 we 是一维数组且 nq 等于 1
      if (PyArray_NDIM(we) == 1 && nq == 1)
        {

          // ldwe 和 ld2we 分别设置为 n 和 1
          ldwe = n;
          ld2we = 1;
        }
      // 如果 we 是一维数组且数组长度为 nq
      else if (PyArray_NDIM(we) == 1 && PyArray_DIMS(we)[0] == nq)
        {
          /* we 是一个一维数组，对角线权重将广播到所有观测值 */
          // ldwe 和 ld2we 设置为 1
          ldwe = 1;
          ld2we = 1;
        }
      // 如果 we 是三维数组且维度分别为 nq、nq、1
      else if (PyArray_NDIM(we) == 3 && PyArray_DIMS(we)[0] == nq
               && PyArray_DIMS(we)[1] == nq && PyArray_DIMS(we)[2] == 1)
        {
          /* we 是一个三维数组，协方差权重将广播到所有观测值 */
          // ldwe 设置为 1，ld2we 设置为 nq
          ldwe = 1;
          ld2we = nq;
        }
      // 如果 we 是二维数组且维度分别为 nq、nq
      else if (PyArray_NDIM(we) == 2 && PyArray_DIMS(we)[0] == nq
               && PyArray_DIMS(we)[1] == nq)
        {
          /* we 是一个二维数组，完整的协方差权重将广播到所有观测值 */
          // ldwe 设置为 1，ld2we 设置为 nq
          ldwe = 1;
          ld2we = nq;
        }

      // 如果 we 是二维数组且维度分别为 nq、n
      else if (PyArray_NDIM(we) == 2 && PyArray_DIMS(we)[0] == nq
               && PyArray_DIMS(we)[1] == n)
        {
          /* we 是一个二维数组，每个观测的协方差权重的对角线元素 */
          // ldwe 设置为 n，ld2we 设置为 1
          ldwe = n;
          ld2we = 1;
        }
      // 如果 we 是三维数组且维度分别为 nq、nq、n
      else if (PyArray_NDIM(we) == 3 && PyArray_DIMS(we)[0] == nq
               && PyArray_DIMS(we)[1] == nq && PyArray_DIMS(we)[2] == n)
        {
          /* we 是完整规范的每个观测的协方差权重 */
          // ldwe 设置为 n，ld2we 设置为 nq
          ldwe = n;
          ld2we = nq;
        }
      else
        {
          // 如果无法将 we 转换为合适的数组，抛出 ValueError 异常
          PYERR(PyExc_ValueError, "could not convert we to a suitable array");
        }
    }                           /* we */

  // 如果 pwd 为 NULL
  if (pwd == NULL)
    {
      // ldwd 和 ld2wd 设置为 1
      ldwd = ld2wd = 1;

      // 设置 wd 为一个一维双精度浮点数数组，长度为 m，第一个元素设为 -1.0
      dim1[0] = m;
      wd = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
      ((double *)(PyArray_DATA(wd)))[0] = -1.0;
    }
  // 如果 pwd 是数字并且不是数组
  else if (PyNumber_Check(pwd) && !PyArray_Check(pwd))
    {
      /* wd 是一个单一权重，将 wd 的第一个值设为 -pwd */
      PyObject *tmp;
      double val;

      // 将 pwd 转换为浮点数
      tmp = PyNumber_Float(pwd);
      if (tmp == NULL)
        // 如果转换失败，抛出 ValueError 异常
        PYERR(PyExc_ValueError, "could not convert wd to a suitable array");
      val = PyFloat_AsDouble(tmp);
      Py_DECREF(tmp);

      // 设置 wd 为一个三维双精度浮点数数组，形状为 (1, 1, m)，第一个元素设为 -val
      dim3[0] = 1;
      dim3[1] = 1;
      dim3[2] = m;
      wd = (PyArrayObject *) PyArray_SimpleNew(3, dim3, NPY_DOUBLE);
      ((double *)(PyArray_DATA(wd)))[0] = -val;
      // ldwd 和 ld2wd 设置为 1
      ldwd = ld2wd = 1;
    }
  // 如果 pwd 是序列
    {
      /* wd needs to be turned into an array */
    
      // 将 pwd 转换为双精度浮点数数组（rank-1），存储在 wd 中
      if ((wd =
           (PyArrayObject *) PyArray_CopyFromObject(pwd, NPY_DOUBLE, 1,
                                                    3)) == NULL)
        {
          // 若转换失败，则抛出值错误异常
          PYERR(PyExc_ValueError, "could not convert wd to a suitable array");
        }
    
      // 检查 wd 是否为 rank-1 数组且 m 等于 1
      if (PyArray_NDIM(wd) == 1 && m == 1)
        {
          // 设置 ldwd 和 ld2wd
          ldwd = n;
          ld2wd = 1;
        }
      // 检查 wd 是否为 rank-1 数组且长度为 m
      else if (PyArray_NDIM(wd) == 1 && PyArray_DIMS(wd)[0] == m)
        {
          /* wd 是一个 rank-1 数组，其中包含要广播到所有观测值的对角线权重 */
          // 设置 ldwd 和 ld2wd
          ldwd = 1;
          ld2wd = 1;
        }
    
      // 检查 wd 是否为 rank-3 数组，且尺寸为 (m, m, 1)
      else if (PyArray_NDIM(wd) == 3 && PyArray_DIMS(wd)[0] == m
               && PyArray_DIMS(wd)[1] == m && PyArray_DIMS(wd)[2] == 1)
        {
          /* wd 是一个 rank-3 数组，其中包含要广播到所有观测值的协变权重 */
          // 设置 ldwd 和 ld2wd
          ldwd = 1;
          ld2wd = m;
        }
      // 检查 wd 是否为 rank-2 数组，且尺寸为 (m, m)
      else if (PyArray_NDIM(wd) == 2 && PyArray_DIMS(wd)[0] == m
               && PyArray_DIMS(wd)[1] == m)
        {
          /* wd 是一个 rank-2 数组，其中包含要广播到所有观测值的完整协变权重 */
          // 设置 ldwd 和 ld2wd
          ldwd = 1;
          ld2wd = m;
        }
    
      // 检查 wd 是否为 rank-2 数组，且尺寸为 (m, n)
      else if (PyArray_NDIM(wd) == 2 && PyArray_DIMS(wd)[0] == m
               && PyArray_DIMS(wd)[1] == n)
        {
          /* wd 是一个 rank-2 数组，其中包含每个观测的协变权重的对角线元素 */
          // 设置 ldwd 和 ld2wd
          ldwd = n;
          ld2wd = 1;
        }
      // 检查 wd 是否为 rank-3 数组，且尺寸为 (m, m, n)
      else if (PyArray_NDIM(wd) == 3 && PyArray_DIMS(wd)[0] == m
               && PyArray_DIMS(wd)[1] == m && PyArray_DIMS(wd)[2] == n)
        {
          /* wd 是每个观测的协变权重的完整规范 */
          // 设置 ldwd 和 ld2wd
          ldwd = n;
          ld2wd = m;
        }
      else
        {
          // 若未找到合适的数组类型，则抛出值错误异常
          PYERR(PyExc_ValueError, "could not convert wd to a suitable array");
        }
    
    }                           /* wd */
    
    
    if (pifixb == NULL)
    {
      // 如果 pifixb 为空指针，则创建一个一维整型数组 ifixb，尺寸为 np
      dim1[0] = np;
      ifixb = (PyArrayObject *) PyArray_SimpleNew(1, dim1, F_INT_NPY);
      // 将数组的第一个元素设为负数
      *(F_INT *)(PyArray_DATA(ifixb)) = -1;      /* set first element negative */
    }
    else
    {
      /* pifixb is a sequence as checked before */
    
      // 如果 pifixb 不为空，则将其转换为一维整型数组 ifixb
      if ((ifixb =
           (PyArrayObject *) PyArray_CopyFromObject(pifixb, F_INT_NPY, 1,
                                                    1)) == NULL)
        {
          // 若转换失败，则抛出值错误异常
          PYERR(PyExc_ValueError,
                "could not convert ifixb to a suitable array");
        }
    
      // 检查 ifixb 的长度是否为 np
      if (PyArray_DIMS(ifixb)[0] != np)
        {
          // 若长度不符合预期，则抛出值错误异常
          PYERR(PyExc_ValueError,
                "could not convert ifixb to a suitable array");
        }
    }                           /* ifixb */
    
    // 如果 pifixx 为空指针
    if (pifixx == NULL)
    {
      // 创建一个二维整型数组 ifixx，尺寸为 (m, 1)
      dim2[0] = m;
      dim2[1] = 1;
      ifixx = (PyArrayObject *) PyArray_SimpleNew(2, dim2, F_INT_NPY);
      // 将数组的第一个元素设为负数
      *(F_INT *)(PyArray_DATA(ifixx)) = -1;      /* set first element negative */
      // 设置 ldifx
      ldifx = 1;
    }  /* 结束 else 块 */

  else
    {
      /* 如果 pifixx 是之前检查过的序列 */

      // 将 pifixx 转换为一个整型数组对象
      if ((ifixx =
           (PyArrayObject *) PyArray_CopyFromObject(pifixx, F_INT_NPY, 1,
                                                    2)) == NULL)
        {
          // 如果转换失败，抛出值错误异常
          PYERR(PyExc_ValueError,
                "could not convert ifixx to a suitable array");
        }

      // 检查 ifixx 是否是一维数组且长度为 m
      if (PyArray_NDIM(ifixx) == 1 && PyArray_DIMS(ifixx)[0] == m)
        {
          ldifx = 1;
        }
      // 检查 ifixx 是否是一维数组且长度为 n，且 m 为 1
      else if (PyArray_NDIM(ifixx) == 1 && PyArray_DIMS(ifixx)[0] == n && m == 1)
        {
          ldifx = n;
        }
      // 检查 ifixx 是否是二维数组且形状为 (m, n)
      else if (PyArray_NDIM(ifixx) == 2 && PyArray_DIMS(ifixx)[0] == m
               && PyArray_DIMS(ifixx)[1] == n)
        {
          ldifx = n;
        }
      else
        {
          // 如果都不满足，则抛出值错误异常
          PYERR(PyExc_ValueError,
                "could not convert ifixx to a suitable array");
        }
    }  /* 结束 ifixx 块 */

  // 如果 errfile 不为空，则执行下面的操作
  if (errfile != NULL)
    {
      // 调用 FORTRAN 的 OPEN 函数打开文件，逻辑单元号为 18
      lunerr = 18;
      F_FUNC(dluno,DLUNO)(&lunerr, errfile, lerrfile);
    }

  // 如果 rptfile 不为空，则执行下面的操作
  if (rptfile != NULL)
    {
      // 调用 FORTRAN 的 OPEN 函数打开文件，逻辑单元号为 19
      lunrpt = 19;
      F_FUNC(dluno,DLUNO)(&lunrpt, rptfile, lrptfile);
    }

  // 如果 pstpb 为空，则执行下面的操作
  if (pstpb == NULL)
    {
      // 创建一个形状为 (np,) 的双精度数组 stpb，并将第一个元素设为 0.0
      dim1[0] = np;
      stpb = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
      *(double *)(PyArray_DATA(stpb)) = 0.0;
    }
  else                          /* pstpb 是一个序列 */
    {
      // 将 pstpb 转换为一个双精度数组对象
      if ((stpb =
           (PyArrayObject *) PyArray_CopyFromObject(pstpb, NPY_DOUBLE, 1,
                                                    1)) == NULL
          || PyArray_DIMS(stpb)[0] != np)
        {
          // 如果转换失败或者形状不匹配 np，则抛出值错误异常
          PYERR(PyExc_ValueError,
                "could not convert stpb to a suitable array");
        }
    }  /* 结束 stpb 块 */

  // 如果 pstpd 为空，则执行下面的操作
  if (pstpd == NULL)
    {
      // 创建一个形状为 (1, m) 的双精度数组 stpd，并将第一个元素设为 0.0，ldstpd 设为 1
      dim2[0] = 1;
      dim2[1] = m;
      stpd = (PyArrayObject *) PyArray_SimpleNew(2, dim2, NPY_DOUBLE);
      *(double *)(PyArray_DATA(stpd)) = 0.0;
      ldstpd = 1;
    }
  else
    {
      // 将 pstpd 转换为一个双精度数组对象
      if ((stpd =
           (PyArrayObject *) PyArray_CopyFromObject(pstpd, NPY_DOUBLE, 1,
                                                    2)) == NULL)
        {
          // 如果转换失败，则抛出值错误异常
          PYERR(PyExc_ValueError,
                "could not convert stpb to a suitable array");
        }

      // 根据 stpd 的形状确定 ldstpd 的值
      if (PyArray_NDIM(stpd) == 1 && PyArray_DIMS(stpd)[0] == m)
        {
          ldstpd = 1;
        }
      else if (PyArray_NDIM(stpd) == 1 && PyArray_DIMS(stpd)[0] == n && m == 1)
        {
          ldstpd = n;
        }
      else if (PyArray_NDIM(stpd) == 2 && PyArray_DIMS(stpd)[0] == n &&
               PyArray_DIMS(stpd)[1] == m)
        {
          ldstpd = n;
        }
    }  /* 结束 stpd 块 */

  // 如果 psclb 为空，则执行下面的操作
  if (psclb == NULL)
    {
      // 创建一个形状为 (np,) 的双精度数组 sclb，并将第一个元素设为 0.0
      dim1[0] = np;
      sclb = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
      *(double *)(PyArray_DATA(sclb)) = 0.0;
    }
  else                          /* psclb is a sequence */
    {
      if ((sclb =
           (PyArrayObject *) PyArray_CopyFromObject(psclb, NPY_DOUBLE, 1,
                                                    1)) == NULL
          || PyArray_DIMS(sclb)[0] != np)
        {
          PYERR(PyExc_ValueError,
                "could not convert sclb to a suitable array");
        }
    }                           /* sclb */

  if (pscld == NULL)
    {
      dim2[0] = 1;
      dim2[1] = n;
      // 创建一个双精度的二维 NumPy 数组 scld，维度为 (1, n)，初始值为 0.0
      scld = (PyArrayObject *) PyArray_SimpleNew(2, dim2, NPY_DOUBLE);
      *(double *)(PyArray_DATA(scld)) = 0.0;
      ldscld = 1;
    }
  else
    {
      // 从 Python 对象 pscld 创建一个双精度的一维 NumPy 数组 scld
      if ((scld =
           (PyArrayObject *) PyArray_CopyFromObject(pscld, NPY_DOUBLE, 1,
                                                    2)) == NULL)
        {
          PYERR(PyExc_ValueError,
                "could not convert stpb to a suitable array");
        }

      // 根据 scld 的维度情况设置 ldscld 的值
      if (PyArray_NDIM(scld) == 1 && PyArray_DIMS(scld)[0] == m)
        {
          ldscld = 1;
        }
      else if (PyArray_NDIM(scld) == 1 && PyArray_DIMS(scld)[0] == n && m == 1)
        {
          ldscld = n;
        }
      else if (PyArray_NDIM(scld) == 2 && PyArray_DIMS(scld)[0] == n &&
               PyArray_DIMS(scld)[1] == m)
        {
          ldscld = n;
        }
    }                           /* scld */

  if (job % 10 < 2)
    {
      /* ODR, not OLS */

      // 计算 lwork 的值，用于 ODR 情况
      lwork =
        18 + 11 * np + np * np + m + m * m + 4 * n * nq + 6 * n * m +
        2 * n * nq * np + 2 * n * nq * m + nq * nq + 5 * nq + nq * (np + m) +
        ldwe * ld2we * nq;

      // 设置 isodr 为 1，表示进行 ODR
      isodr = 1;
    }
  else
    {
      /* OLS, not ODR */

      // 计算 lwork 的值，用于 OLS 情况
      lwork =
        18 + 11 * np + np * np + m + m * m + 4 * n * nq + 2 * n * m +
        2 * n * nq * np + 5 * nq + nq * (np + m) + ldwe * ld2we * nq;

      // 设置 isodr 为 0，表示进行 OLS
      isodr = 0;
    }

  // 计算 liwork 的值
  liwork = 20 + np + nq * (np + m);

  if ((job / 10000) % 10 >= 1)
    {
      /* fit is a restart, make sure work and iwork are input */

      // 如果 job 表示 fit 是一个重新开始，确保输入了 work 和 iwork 数组
      if (pwork == NULL || piwork == NULL)
        {
          PYERR(PyExc_ValueError,
                "need to input work and iwork arrays to restart");
        }
    }

  if ((job / 1000) % 10 >= 1)
    {
      /* delta should be supplied, make sure the user does */

      // 如果 job 表示 delta 应该被提供，确保用户已经提供了 delta
      if (pwork == NULL)
        {
          PYERR(PyExc_ValueError,
                "need to input work array for delta initialization");
        }
    }

  if (pwork != NULL)
    {
      // 从 Python 对象 pwork 创建一个双精度的一维 NumPy 数组 work
      if ((work =
           (PyArrayObject *) PyArray_CopyFromObject(pwork, NPY_DOUBLE, 1,
                                                    1)) == NULL)
        {
          PYERR(PyExc_ValueError,
                "could not convert work to a suitable array");
        }
      // 检查 work 数组的长度是否足够大
      if (PyArray_DIMS(work)[0] < lwork)
        {
          // 打印出现在的长度和需要的长度，然后抛出异常
          printf("%lld %lld\n", (long long)PyArray_DIMS(work)[0], (long long)lwork);
          PYERR(PyExc_ValueError, "work is too small");
        }
    }
  else
    {
      /* initialize our own work array */
      # 初始化自定义的工作数组维度
      dim1[0] = lwork;
      # 创建一个新的一维 NumPy 数组对象，用于存储双精度浮点数
      work = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
    }                           /* work */

  if (piwork != NULL)
    {
      # 如果 piwork 不为空，则将其转换为一个适合的整数类型 NumPy 数组对象
      if ((iwork =
           (PyArrayObject *) PyArray_CopyFromObject(piwork, F_INT_NPY, 1,
                                                    1)) == NULL)
        {
          # 转换失败，抛出值错误异常
          PYERR(PyExc_ValueError,
                "could not convert iwork to a suitable array");
        }

      # 检查转换后的 iwork 数组是否足够大
      if (PyArray_DIMS(iwork)[0] < liwork)
        {
          # 如果不够大，抛出值错误异常
          PYERR(PyExc_ValueError, "iwork is too small");
        }
    }
  else
    {
      /* initialize our own iwork array */
      # 初始化自定义的 iwork 数组维度
      dim1[0] = liwork;
      # 创建一个新的一维整数类型 NumPy 数组对象
      iwork = (PyArrayObject *) PyArray_SimpleNew(1, dim1, F_INT_NPY);
    }                           /* iwork */

  /* check if what JOB requests can be done with what the user has
     input into the function */

  # 检查 JOB 参数要求的操作是否可以使用用户输入的函数进行

  if ((job / 10) % 10 >= 2)
    {
      /* derivatives are supposed to be supplied */
      
      # 如果 JOB 参数要求提供导数

      if (fjacb == NULL || fjacd == NULL)
        {
          # 如果 fjacb 或 fjacd 为空，抛出值错误异常
          PYERR(PyExc_ValueError,
                "need fjacb and fjacd to calculate derivatives");
        }
    }

  /* setup the global data for the callback */
  # 设置回调函数的全局数据
  odr_global.fcn = fcn;
  Py_INCREF(fcn);
  odr_global.fjacb = fjacb;
  Py_XINCREF(fjacb);
  odr_global.fjacd = fjacd;
  Py_XINCREF(fjacd);
  odr_global.pyBeta = (PyObject *) beta;
  Py_INCREF(beta);
  odr_global.extra_args = extra_args;
  Py_XINCREF(extra_args);
   /* now call DODRC */
   # 现在调用 DODRC 函数
   F_FUNC(dodrc,DODRC)(fcn_callback, &n, &m, &np, &nq, (double *)(PyArray_DATA(beta)),
         (double *)(PyArray_DATA(y)), &ldy, (double *)(PyArray_DATA(x)), &ldx,
         (double *)(PyArray_DATA(we)), &ldwe, &ld2we,
         (double *)(PyArray_DATA(wd)), &ldwd, &ld2wd,
         (F_INT *)(PyArray_DATA(ifixb)), (F_INT *)(PyArray_DATA(ifixx)), &ldifx,
         &job, &ndigit, &taufac, &sstol, &partol, &maxit,
         &iprint, &lunerr, &lunrpt,
         (double *)(PyArray_DATA(stpb)), (double *)(PyArray_DATA(stpd)), &ldstpd,
         (double *)(PyArray_DATA(sclb)), (double *)(PyArray_DATA(scld)), &ldscld,
         (double *)(PyArray_DATA(work)), &lwork, (F_INT *)(PyArray_DATA(iwork)), &liwork,
         &info);

  # 调用 DODRC 函数完成后，得到的结果存储在 result 变量中
  result = gen_output(n, m, np, nq, ldwe, ld2we,
                      beta, work, iwork, isodr, info, full_output);

  # 如果结果为空，抛出运行时错误异常
  if (result == NULL)
    PYERR(PyExc_RuntimeError, "could not generate output");

  # 如果 lunerr 不等于 -1，则调用 DLUNC 函数处理 lunerr
  if (lunerr != -1)
    {
      F_FUNC(dlunc,DLUNC)(&lunerr);
    }
  # 如果 lunrpt 不等于 -1，则调用 DLUNC 函数处理 lunrpt
  if (lunrpt != -1)
    {
      F_FUNC(dlunc,DLUNC)(&lunrpt);
    }

  # 释放引用的对象，防止内存泄漏
  Py_DECREF(odr_global.fcn);
  Py_XDECREF(odr_global.fjacb);
  Py_XDECREF(odr_global.fjacd);
  Py_DECREF(odr_global.pyBeta);
  Py_XDECREF(odr_global.extra_args);

  # 清空全局数据
  odr_global.fcn = odr_global.fjacb = odr_global.fjacd = odr_global.pyBeta =
    odr_global.extra_args = NULL;

# 将 odr_global 结构体中的 extra_args 成员设置为 NULL


  Py_DECREF(beta);
  Py_DECREF(y);
  Py_DECREF(x);
  Py_DECREF(we);
  Py_DECREF(wd);
  Py_DECREF(ifixb);
  Py_DECREF(ifixx);
  Py_DECREF(stpb);
  Py_DECREF(stpd);
  Py_DECREF(sclb);
  Py_DECREF(scld);
  Py_DECREF(work);
  Py_DECREF(iwork);

# 逐个减少 Python 对象的引用计数，这些对象分别是 beta, y, x, we, wd, ifixb, ifixx, stpb, stpd, sclb, scld, work, iwork


  return result;

# 返回 result 变量作为函数的结果
fail:

// 如果 lunerr 不等于 -1，则调用 dlunc 函数来释放 lunerr 指向的资源
if (lunerr != -1)
{
    F_FUNC(dlunc,DLUNC)(&lunerr);
}

// 如果 lunrpt 不等于 -1，则调用 dlunc 函数来释放 lunrpt 指向的资源
if (lunrpt != -1)
{
    F_FUNC(dlunc,DLUNC)(&lunrpt);
}

// 逐一释放所有 Python 对象的引用
Py_XDECREF(beta);
Py_XDECREF(y);
Py_XDECREF(x);
Py_XDECREF(we);
Py_XDECREF(wd);
Py_XDECREF(ifixb);
Py_XDECREF(ifixx);
Py_XDECREF(stpb);
Py_XDECREF(stpd);
Py_XDECREF(sclb);
Py_XDECREF(scld);
Py_XDECREF(work);
Py_XDECREF(iwork);

// 返回空指针，表明函数执行失败
return NULL;
}


PyObject *set_exceptions(PyObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *exc_error, *exc_stop;

    // 解析 Python 元组参数，获取 exc_error 和 exc_stop
    if (!PyArg_ParseTuple(args, "OO", &exc_error, &exc_stop))
        return NULL;

    // 增加 exc_stop 和 exc_error 的引用计数，确保它们在函数生命周期内有效
    Py_INCREF(exc_stop);
    Py_INCREF(exc_error);
    
    // 将 exc_stop 和 exc_error 分别赋值给全局变量 odr_stop 和 odr_error
    odr_stop = exc_stop;
    odr_error = exc_error;

    // 增加 Py_None 的引用计数，并返回 Py_None 表示函数成功执行
    Py_INCREF(Py_None);
    return Py_None;
}

// 定义 Python 扩展模块的方法列表
static PyMethodDef methods[] = {
    {"_set_exceptions", (PyCFunction) set_exceptions, METH_VARARGS, NULL},
    {"odr", (PyCFunction) odr, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL},  // 方法列表结束标志
};

// 定义 Python 扩展模块的结构体
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  // 初始化 Python 模块结构体
    "_odrpack",             // 模块名称
    NULL,                   // 模块文档字符串
    -1,                     // 模块状态，-1 表示使用全局状态
    methods,                // 方法列表
    NULL,                   // 模块的槽函数，此处为 NULL
    NULL,                   // 模块的清理函数，此处为 NULL
    NULL,                   // 模块的内存分配函数，此处为 NULL
    NULL                    // 模块的状态对象，此处为 NULL
};

// Python 模块初始化函数
PyMODINIT_FUNC
PyInit___odrpack(void)
{
    PyObject *m;
    
    // 导入 NumPy 数组支持
    import_array();
    
    // 创建 Python 模块对象 m，使用 moduledef 定义的模块结构
    m = PyModule_Create(&moduledef);
    
    // 返回创建的 Python 模块对象 m
    return m;
}
```