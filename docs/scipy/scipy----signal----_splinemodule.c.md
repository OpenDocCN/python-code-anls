# `D:\src\scipysrc\scipy\scipy\signal\_splinemodule.c`

```
#include "_splinemodule.h"

#define PyArray_MIN(a,b) (((a)<(b))?(a):(b))

// 定义一个静态函数，用于将输入数组的步幅转换为输出数组的步幅
static void
convert_strides(npy_intp* instrides, npy_intp* convstrides, int size, int N)
{
  int n;
  npy_intp bitshift;

  bitshift = -1;

  // 计算步幅转换所需的位移量
  while (size != 0) {
    size >>= 1;
    bitshift++;
  }

  // 根据位移量转换每个输入步幅到输出步幅
  for (n = 0; n < N; n++) {
    convstrides[n] = instrides[n] >> bitshift;
  }
}

// 定义文档字符串，描述了函数 FIRsepsym2d 的作用、参数和返回值
static char doc_FIRsepsym2d[] = "out = sepfir2d(input, hrow, hcol)\n"
"\n"
"    Convolve with a 2-D separable FIR filter.\n"
"\n"
"    Convolve the rank-2 input array with the separable filter defined by the\n"
"    rank-1 arrays hrow, and hcol. Mirror symmetric boundary conditions are\n"
"    assumed. This function can be used to find an image given its B-spline\n"
"    representation.\n"
"\n"
"    Parameters\n"
"    ----------\n"
"    input : ndarray\n"
"        The input signal. Must be a rank-2 array.\n"
"    hrow : ndarray\n"
"        A rank-1 array defining the row direction of the filter.\n"
"        Must be odd-length\n"
"    hcol : ndarray\n"
"        A rank-1 array defining the column direction of the filter.\n"
"        Must be odd-length\n"
"\n"
"    Returns\n"
"    -------\n"
"    output : ndarray\n"
"        The filtered signal.\n"
"\n"
"    Examples\n"
"    --------\n"
"    Examples are given :ref:`in the tutorial <tutorial-signal-bsplines>`.\n"
"\n";

// 定义 Python C 扩展函数 FIRsepsym2d，实现了图像与二维分离 FIR 滤波器的卷积
static PyObject *FIRsepsym2d(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
  PyObject *image=NULL, *hrow=NULL, *hcol=NULL;
  PyArrayObject *a_image=NULL, *a_hrow=NULL, *a_hcol=NULL, *out=NULL;
  int thetype, M, N, ret;
  npy_intp outstrides[2], instrides[2];

  // 解析 Python 函数参数，获取输入图像和两个滤波器数组
  if (!PyArg_ParseTuple(args, "OOO", &image, &hrow, &hcol)) return NULL;

  // 确定输入图像的数据类型，并创建相应的 NumPy 数组对象
  thetype = PyArray_ObjectType(image, NPY_FLOAT);
  thetype = PyArray_MIN(thetype, NPY_CDOUBLE);
  a_image = (PyArrayObject *)PyArray_FromObject(image, thetype, 2, 2);
  if (a_image == NULL) goto fail;

  // 从 Python 对象创建滤波器数组对象，并确保它们是连续的
  a_hrow = (PyArrayObject *)PyArray_ContiguousFromObject(hrow, thetype, 1, 1);
  if (a_hrow == NULL) goto fail;

  a_hcol = (PyArrayObject *)PyArray_ContiguousFromObject(hcol, thetype, 1, 1);
  if (a_hcol == NULL) goto fail;

  // 根据输入图像的尺寸创建输出数组对象
  out = (PyArrayObject *)PyArray_SimpleNew(2, PyArray_DIMS(a_image), thetype);
  if (out == NULL) goto fail;

  // 获取输入图像的行列数
  M = PyArray_DIMS(a_image)[0];
  N = PyArray_DIMS(a_image)[1];

  // 将输入图像的步幅转换为输出数组的步幅
  convert_strides(PyArray_STRIDES(a_image), instrides, PyArray_ITEMSIZE(a_image), 2);
  outstrides[0] = N;
  outstrides[1] = 1;

  // 检查滤波器数组是否为奇数长度，若不是则抛出异常
  if (PyArray_DIMS(a_hrow)[0] % 2 != 1 ||
      PyArray_DIMS(a_hcol)[0] % 2 != 1) {
    PYERR("hrow and hcol must be odd length");
  }

  // 根据输入图像和滤波器数组的数据类型选择合适的卷积函数进行计算
  switch (thetype) {
  case NPY_FLOAT:
    ret = S_separable_2Dconvolve_mirror((float *)PyArray_DATA(a_image),
                    (float *)PyArray_DATA(out), M, N,
                    (float *)PyArray_DATA(a_hrow),
                    (float *)PyArray_DATA(a_hcol),
                    PyArray_DIMS(a_hrow)[0], PyArray_DIMS(a_hcol)[0],
                    instrides, outstrides);
    break;
  case NPY_DOUBLE:
    // 对于双精度浮点数类型，执行相应的卷积计算函数
    // （未完整提供，需根据实际代码补充）
    break;
  // 其他数据类型的处理（未完整提供，需根据实际代码补充）
  }

  // 返回卷积结果的 NumPy 数组对象
  return (PyObject *)out;

  // 出错处理标签，释放已分配的资源并返回 NULL
fail:
  Py_XDECREF(a_image);
  Py_XDECREF(a_hrow);
  Py_XDECREF(a_hcol);
  Py_XDECREF(out);
  return NULL;
}
    # 调用名为 D_separable_2Dconvolve_mirror 的函数，进行镜像分离卷积操作
    ret = D_separable_2Dconvolve_mirror((double *)PyArray_DATA(a_image),
                    (double *)PyArray_DATA(out), M, N,
                    (double *)PyArray_DATA(a_hrow),
                    (double *)PyArray_DATA(a_hcol),
                    PyArray_DIMS(a_hrow)[0], PyArray_DIMS(a_hcol)[0],
                    instrides, outstrides);
    # 跳出当前循环或 switch 语句，终止循环体执行
    break;
#ifdef __GNUC__
  // 如果编译器是 GCC，根据数组元素类型选择相应的复数类型处理
  case NPY_CFLOAT:
    // 对于单精度复数数组，调用镜像对称的二维卷积函数
    ret = C_separable_2Dconvolve_mirror((__complex__ float *)PyArray_DATA(a_image),
                    (__complex__ float *)PyArray_DATA(out), M, N,
                    (__complex__ float *)PyArray_DATA(a_hrow),
                    (__complex__ float *)PyArray_DATA(a_hcol),
                    PyArray_DIMS(a_hrow)[0], PyArray_DIMS(a_hcol)[0],
                    instrides, outstrides);
    break;
  case NPY_CDOUBLE:
    // 对于双精度复数数组，调用镜像对称的二维卷积函数
    ret = Z_separable_2Dconvolve_mirror((__complex__ double *)PyArray_DATA(a_image),
                    (__complex__ double *)PyArray_DATA(out), M, N,
                    (__complex__ double *)PyArray_DATA(a_hrow),
                    (__complex__ double *)PyArray_DATA(a_hcol),
                    PyArray_DIMS(a_hrow)[0], PyArray_DIMS(a_hcol)[0],
                    instrides, outstrides);
    break;
#endif
  // 默认情况，若类型不匹配则抛出错误信息
  default:
    PYERR("Incorrect type.");
  }

  // 检查返回值，若小于 0 则抛出错误信息
  if (ret < 0) PYERR("Problem occurred inside routine.");

  // 释放 Python 对象引用
  Py_DECREF(a_image);
  Py_DECREF(a_hrow);
  Py_DECREF(a_hcol);
  // 返回输出数组对象
  return PyArray_Return(out);

 fail:
  // 处理错误情况，释放 Python 对象引用并返回 NULL
  Py_XDECREF(a_image);
  Py_XDECREF(a_hrow);
  Py_XDECREF(a_hcol);
  Py_XDECREF(out);
  return NULL;
}


static char doc_IIRsymorder1_ic[] = "out = symiirorder1_ic(input, z1, precision=-1.0)\n"
"\n"
"    Compute the (forward) mirror-symmetric boundary conditions for a smoothing\n"
"    IIR filter that is composed of cascaded first-order sections.\n"
"\n"
"    The starting condition returned by this function is computed based on\n"
"    the following transfer function::\n"
"\n"
"                       1         \n"
"           H(z) = ------------   \n"
"                   (1 - z1/z)    \n"
"\n"
"\n"
"    Parameters\n"
"    ----------\n"
"    input : ndarray\n"
"        The input signal. If 2D, then it will find the initial conditions \n"
"        for each of the elements on the last axis.\n"
"    z1 : scalar\n"
"        Parameter in the transfer function.\n"
"    precision :\n"
"        Specifies the precision for calculating initial conditions\n"
"        of the recursive filter based on mirror-symmetric input.\n"
"\n"
"    Returns\n"
"    -------\n"
"    z_0 : ndarray\n"
"        The mirror-symmetric initial condition for the forward IIR filter.";

static PyObject *IIRsymorder1_ic(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
  PyObject *sig=NULL;
  PyArrayObject *a_sig=NULL, *out=NULL;
  npy_intp* in_size;

  Py_complex z1;
  double precision = -1.0;
  int thetype, ret;
  npy_intp M, N;
  PyArray_Descr* dtype;

  // 解析输入参数，获取信号、参数 z1 和 precision
  if (!PyArg_ParseTuple(args, "OD|d", &sig, &z1, &precision))
    return NULL;

  // 获取信号的数据类型
  thetype = PyArray_ObjectType(sig, NPY_FLOAT);
  thetype = PyArray_MIN(thetype, NPY_CDOUBLE);
  // 根据数据类型创建相应的数组对象
  a_sig = (PyArrayObject *)PyArray_FromObject(sig, thetype, 1, 2);

  // 检查数组对象是否创建成功
  if (a_sig == NULL) goto fail;

  // 获取数组的尺寸
  in_size = PyArray_DIMS(a_sig);
  M = 1;
  N = in_size[0];

  // 如果数组是二维的，更新 M 的值
  if(PyArray_NDIM(a_sig) > 1) {
    M = in_size[0];
  }
    N = in_size[1];
  }

  # 定义一个包含两个元素的数组，用于指定输出数组的形状
  const npy_intp sz[2] = {M, 1};
  # 从给定的数据类型创建一个描述器对象
  dtype = PyArray_DescrFromType(thetype);
  # 创建一个指定形状和数据类型的空 NumPy 数组
  out = (PyArrayObject *)PyArray_Empty(2, sz, dtype, 0);
  # 如果创建数组失败，跳转到错误处理标签
  if (out == NULL) goto fail;

  # 根据不同的数据类型执行不同的操作
  switch (thetype) {
  case NPY_FLOAT:
    {
      # 获取 z1 的实部作为 rz1 的值
      float rz1 = z1.real;

      # 如果 precision 不在合理范围内，设置默认精度
      if ((precision <= 0.0) || (precision > 1.0)) precision = 1e-6;
      # 调用 S_SYM_IIR1_initial 函数初始化操作
      ret = S_SYM_IIR1_initial(rz1, (float *)PyArray_DATA(a_sig),
                               (float *)PyArray_DATA(out), M, N,
                               (float)precision);
    }
    break;
  case NPY_DOUBLE:
    {
      # 获取 z1 的实部作为 rz1 的值
      double rz1 = z1.real;

      # 如果 precision 不在合理范围内，设置默认精度
      if ((precision <= 0.0) || (precision > 1.0)) precision = 1e-11;
      # 调用 D_SYM_IIR1_initial 函数初始化操作
      ret = D_SYM_IIR1_initial(rz1, (double *)PyArray_DATA(a_sig),
                               (double *)PyArray_DATA(out), M, N,
                               precision);
    }
    break;
#ifdef __GNUC__
  case NPY_CFLOAT:
    {
      // 定义一个复数 zz1，实部为 z1.real，虚部为 z1.imag
      __complex__ float zz1 = z1.real + 1.0i*z1.imag;
      // 如果 precision 小于等于 0.0 或者大于 1.0，则将 precision 设为 1e-6
      if ((precision <= 0.0) || (precision > 1.0)) precision = 1e-6;
      // 调用 C_SYM_IIR1_initial 函数进行初始化
      ret = C_SYM_IIR1_initial (zz1, (__complex__ float *)PyArray_DATA(a_sig),
                (__complex__ float *)PyArray_DATA(out), M, N,
                (float )precision);
    }
    break;
  case NPY_CDOUBLE:
    {
      // 定义一个双精度复数 zz1，实部为 z1.real，虚部为 z1.imag
      __complex__ double zz1 = z1.real + 1.0i*z1.imag;
      // 如果 precision 小于等于 0.0 或者大于 1.0，则将 precision 设为 1e-11
      if ((precision <= 0.0) || (precision > 1.0)) precision = 1e-11;
      // 调用 Z_SYM_IIR1_initial 函数进行初始化
      ret = Z_SYM_IIR1_initial (zz1, (__complex__ double *)PyArray_DATA(a_sig),
                (__complex__ double *)PyArray_DATA(out), M, N,
                precision);
    }
    break;
#endif
  // 默认情况，当 type 不是 NPY_CFLOAT 或者 NPY_CDOUBLE 时，抛出错误信息
  default:
    PYERR("Incorrect type.");
  }

  // 如果 ret 等于 0，表示初始化成功，释放 a_sig，返回处理后的 out 数组
  if (ret == 0) {
    Py_DECREF(a_sig);
    return PyArray_Return(out);
  }

  // 如果 ret 等于 -1，表示内存分配失败，抛出错误信息
  if (ret == -1) PYERR("Could not allocate enough memory.");
  // 如果 ret 等于 -2，表示 |z1| 必须小于 1.0，抛出错误信息
  if (ret == -2) PYERR("|z1| must be less than 1.0");
  // 如果 ret 等于 -3，表示对称边界条件的求和没有收敛，抛出错误信息
  if (ret == -3) PYERR("Sum to find symmetric boundary conditions did not converge.");

  // 如果 ret 不是 0、-1、-2、-3，则抛出未知错误信息
  PYERR("Unknown error.");


 fail:
  // 失败处理，释放 a_sig 和 out，返回 NULL
  Py_XDECREF(a_sig);
  Py_XDECREF(out);
  return NULL;

}

// 函数 doc_IIRsymorder2_ic_fwd 的文档字符串，描述了函数 symiirorder2_ic_fwd 的作用、参数和返回值
static char doc_IIRsymorder2_ic_fwd[] = "out = symiirorder2_ic_fwd(input, r, omega, precision=-1.0)\n"
"\n"
"    Compute the (forward) mirror-symmetric boundary conditions for a smoothing\n"
"    IIR filter that is composed of cascaded second-order sections.\n"
"\n"
"    The starting condition returned by this function is computed based on\n"
"    the following transfer function::\n"
"\n"
"                         cs\n"
"         H(z) = -------------------\n"
"                (1 - a2/z - a3/z^2)\n"
"\n"
"    where::\n"
"\n"
"          a2 = (2 r cos omega)\n"
"          a3 = - r^2\n"
"          cs = 1 - 2 r cos omega + r^2\n"
"\n"
"    Parameters\n"
"    ----------\n"
"    input : ndarray\n"
"        The input signal.\n"
"    r, omega : float\n"
"        Parameters in the transfer function.\n"
"    precision : float\n"
"        Specifies the precision for calculating initial conditions\n"
"        of the recursive filter based on mirror-symmetric input.\n"
"\n"
"    Returns\n"
"    -------\n"
"    zi : ndarray\n"
"        The mirror-symmetric initial condition for the forward IIR filter.";

// 函数 IIRsymorder2_ic_fwd 的实现，计算镜像对称的初始条件
static PyObject *IIRsymorder2_ic_fwd(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
  PyObject *sig=NULL;
  PyArrayObject *a_sig=NULL, *out=NULL;
  npy_intp* in_size;
  double r, omega;
  double precision = -1.0;
  int thetype, ret;
  npy_intp N, M;
  PyArray_Descr* dtype;

  // 解析参数，获取输入信号 sig、参数 r、omega 和 precision
  if (!PyArg_ParseTuple(args, "Odd|d", &sig, &r, &omega, &precision))
    return NULL;

  // 确定输入信号的类型
  thetype = PyArray_ObjectType(sig, NPY_FLOAT);
  // 确保类型不低于 NPY_FLOAT，如果 sig 不是数组，则返回错误
  thetype = PyArray_MIN(thetype, NPY_DOUBLE);
  // 将 sig 转换为相应类型的一维数组 a_sig
  a_sig = (PyArrayObject *)PyArray_FromObject(sig, thetype, 1, 2);

  // 如果转换失败，跳转到失败处理标签 fail
  if (a_sig == NULL) goto fail;

  // 获取 a_sig 的维度信息
  in_size = PyArray_DIMS(a_sig);
  M = 1;
  N = in_size[0];

  // 如果 a_sig 的维度大于 1，将 M 设置为 in_size[0]
  if(PyArray_NDIM(a_sig) > 1) {
    M = in_size[0];
    N = in_size[1];
  }

  dtype = PyArray_DescrFromType(thetype);
  const npy_intp sz[2] = {M, 2};
  out = (PyArrayObject *)PyArray_Empty(2, sz, dtype, 0);
  if (out == NULL) goto fail;

  switch (thetype) {
  case NPY_FLOAT:
    {
      // 如果精度小于等于0或大于1，则设置默认精度为1e-6
      if ((precision <= 0.0) || (precision > 1.0)) precision = 1e-6;
      // 调用 S_SYM_IIR2_initial_fwd 函数执行单精度浮点数的初始化前向对称IIR滤波器操作
      ret = S_SYM_IIR2_initial_fwd(r, omega, (float *)PyArray_DATA(a_sig),
                                  (float *)PyArray_DATA(out), M, N,
                                  (float)precision);
    }
    break;
  case NPY_DOUBLE:
    {
      // 如果精度小于等于0或大于1，则设置默认精度为1e-11
      if ((precision <= 0.0) || (precision > 1.0)) precision = 1e-11;
      // 调用 D_SYM_IIR2_initial_fwd 函数执行双精度浮点数的初始化前向对称IIR滤波器操作
      ret = D_SYM_IIR2_initial_fwd(r, omega, (double *)PyArray_DATA(a_sig),
                                  (double *)PyArray_DATA(out), M, N,
                                  precision);
    }
    break;
  default:
    // 类型不正确，抛出异常信息
    PYERR("Incorrect type.");
  }

  // 如果操作成功完成
  if (ret == 0) {
    Py_DECREF(a_sig);
    // 返回处理后的输出数组对象
    return PyArray_Return(out);
  }

  // 根据返回的错误代码进行相应处理
  if (ret == -1) PYERR("Could not allocate enough memory.");
  if (ret == -2) PYERR("|z1| must be less than 1.0");
  if (ret == -3) PYERR("Sum to find symmetric boundary conditions did not converge.");

  // 未知错误，抛出异常信息
  PYERR("Unknown error.");


 fail:
  // 发生失败时，释放信号数组和输出数组对象的引用
  Py_XDECREF(a_sig);
  Py_XDECREF(out);
  // 返回空指针表示处理失败
  return NULL;
static char doc_IIRsymorder2_ic_bwd[] = "out = symiirorder2_ic_bwd(input, r, omega, precision=-1.0)\n"
"\n"
"    Compute the (backward) mirror-symmetric boundary conditions for a smoothing\n"
"    IIR filter that is composed of cascaded second-order sections.\n"
"\n"
"    The starting condition returned by this function is computed based on\n"
"    the following transfer function::\n"
"\n"
"                         cs\n"
"         H(z) = -------------------\n"
"                (1 - a2 z - a3 z^2)\n"
"\n"
"    where::\n"
"\n"
"          a2 = (2 r cos omega)\n"
"          a3 = - r^2\n"
"          cs = 1 - 2 r cos omega + r^2\n"
"\n"
"    Parameters\n"
"    ----------\n"
"    input : ndarray\n"
"        The input signal.\n"
"    r, omega : float\n"
"        Parameters in the transfer function.\n"
"    precision : float\n"
"        Specifies the precision for calculating initial conditions\n"
"        of the recursive filter based on mirror-symmetric input.\n"
"\n"
"    Returns\n"
"    -------\n"
"    zi : ndarray\n"
"        The mirror-symmetric initial condition for the forward IIR filter.";

// 定义一个名为IIRsymorder2_ic_bwd的Python函数，接受sig, r, omega, precision作为参数
static PyObject *IIRsymorder2_ic_bwd(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
  PyObject *sig=NULL; // Python对象，表示输入信号
  PyArrayObject *a_sig=NULL, *out=NULL; // NumPy数组对象，表示输入信号和输出数组
  npy_intp* in_size; // 用于存储输入信号的尺寸
  double r, omega; // 浮点数参数r和omega
  double precision = -1.0; // 默认精度为-1.0
  int thetype, ret; // 整数类型和返回值
  npy_intp M, N; // 数组维度的整数值
  PyArray_Descr* dtype; // NumPy数组描述符

  // 解析Python传入的参数，格式为(sig, r, omega, precision)
  if (!PyArg_ParseTuple(args, "Odd|d", &sig, &r, &omega, &precision))
    return NULL;

  // 获取输入信号的NumPy数据类型
  thetype = PyArray_ObjectType(sig, NPY_FLOAT);
  thetype = PyArray_MIN(thetype, NPY_DOUBLE);
  // 根据输入信号创建NumPy数组对象a_sig
  a_sig = (PyArrayObject *)PyArray_FromObject(sig, thetype, 1, 2);

  // 如果a_sig创建失败，则跳转到fail标签
  if (a_sig == NULL) goto fail;

  // 获取输入信号的维度大小
  in_size = PyArray_DIMS(a_sig);
  M = 1;
  N = in_size[0];

  // 如果输入信号的维度大于1，则重新分配M和N
  if(PyArray_NDIM(a_sig) > 1) {
    M = in_size[0];
    N = in_size[1];
  }

  // 根据thetype创建NumPy数组的描述符
  dtype = PyArray_DescrFromType(thetype);
  // 创建一个维度为[M, 2]的零数组out，使用dtype描述符
  const npy_intp sz[2] = {M, 2};
  out = (PyArrayObject *)PyArray_Zeros(2, sz, dtype, 0);
  // 如果创建out数组失败，则跳转到fail标签
  if (out == NULL) goto fail;

  // 根据thetype选择性地设置precision的值
  switch (thetype) {
  case NPY_FLOAT:
    {
      if ((precision <= 0.0) || (precision > 1.0)) precision = 1e-6;
      // 调用S_SYM_IIR2_initial_bwd函数计算输出，返回结果存储在ret中
      ret = S_SYM_IIR2_initial_bwd(r, omega, (float *)PyArray_DATA(a_sig),
                                  (float *)PyArray_DATA(out), M, N,
                                  (float)precision);
    }
    break;
  case NPY_DOUBLE:
    {
      if ((precision <= 0.0) || (precision > 1.0)) precision = 1e-11;
      // 调用D_SYM_IIR2_initial_bwd函数计算输出，返回结果存储在ret中
      ret = D_SYM_IIR2_initial_bwd(r, omega, (double *)PyArray_DATA(a_sig),
                                   (double *)PyArray_DATA(out), M, N,
                                   precision);
    }
    break;
  default:
    PYERR("Incorrect type.");
  }

  // 如果ret为0，则释放a_sig对象
  if (ret == 0) {
    Py_DECREF(a_sig);
    // 返回一个 PyArray_Return 函数调用的结果，将 out 指针作为参数
    return PyArray_Return(out);
  }

  // 如果 ret 的值为 -1，则输出错误信息 "Could not allocate enough memory."
  if (ret == -1) PYERR("Could not allocate enough memory.");
  // 如果 ret 的值为 -2，则输出错误信息 "|z1| must be less than 1.0"
  if (ret == -2) PYERR("|z1| must be less than 1.0");
  // 如果 ret 的值为 -3，则输出错误信息 "Sum to find symmetric boundary conditions did not converge."
  if (ret == -3) PYERR("Sum to find symmetric boundary conditions did not converge.");

  // 如果以上条件都不满足，则输出默认错误信息 "Unknown error."
  PYERR("Unknown error.");


 fail:
  // 清理并释放 a_sig 和 out 指针所指向的对象
  Py_XDECREF(a_sig);
  Py_XDECREF(out);
  // 返回 NULL 指示函数执行失败
  return NULL;
}

static struct PyMethodDef toolbox_module_methods[] = {
    {"sepfir2d", FIRsepsym2d, METH_VARARGS, doc_FIRsepsym2d},
    // 定义 Python 模块中的方法表，包括方法名、函数指针、调用方式、文档字符串
    {"symiirorder1_ic", IIRsymorder1_ic, METH_VARARGS, doc_IIRsymorder1_ic},
    // 添加第二个方法到方法表，对应函数指针为 IIRsymorder1_ic，调用方式为 METH_VARARGS，文档为 doc_IIRsymorder1_ic
    {"symiirorder2_ic_fwd", IIRsymorder2_ic_fwd, METH_VARARGS, doc_IIRsymorder2_ic_fwd},
    // 添加第三个方法到方法表，对应函数指针为 IIRsymorder2_ic_fwd，调用方式为 METH_VARARGS，文档为 doc_IIRsymorder2_ic_fwd
    {"symiirorder2_ic_bwd", IIRsymorder2_ic_bwd, METH_VARARGS, doc_IIRsymorder2_ic_bwd },
    // 添加第四个方法到方法表，对应函数指针为 IIRsymorder2_ic_bwd，调用方式为 METH_VARARGS，文档为 doc_IIRsymorder2_ic_bwd
    {NULL, NULL, 0, NULL}        /* sentinel */
    // 方法表的结尾标志，表明方法列表结束
};

/* Initialization function for the module */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_spline",
    NULL,
    -1,
    toolbox_module_methods, // 将方法表 toolbox_module_methods 分配给模块
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__spline(void)
{
    import_array(); // 导入 NumPy 数组处理库
    return PyModule_Create(&moduledef); // 创建 Python 模块并返回
}
```