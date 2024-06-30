# `D:\src\scipysrc\scipy\scipy\odr\odrpack.h`

```
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"

#ifdef HAVE_BLAS_ILP64

#define F_INT npy_int64               // 定义 F_INT 为 numpy 中的 64 位整型
#define F_INT_NPY NPY_INT64           // 定义 F_INT_NPY 为 numpy 中的 64 位整型常量

#if NPY_BITSOF_SHORT == 64
#define F_INT_PYFMT   "h"             // 如果 short 是 64 位，则定义 F_INT_PYFMT 为短整型的格式化字符串
#elif NPY_BITSOF_INT == 64
#define F_INT_PYFMT   "i"             // 如果 int 是 64 位，则定义 F_INT_PYFMT 为整型的格式化字符串
#elif NPY_BITSOF_LONG == 64
#define F_INT_PYFMT   "l"             // 如果 long 是 64 位，则定义 F_INT_PYFMT 为长整型的格式化字符串
#elif NPY_BITSOF_LONGLONG == 64
#define F_INT_PYFMT   "L"             // 如果 long long 是 64 位，则定义 F_INT_PYFMT 为长长整型的格式化字符串
#else
#error No compatible 64-bit integer size. \
       Please contact NumPy maintainers and give detailed information about your \
       compiler and platform, or set NPY_USE_BLAS64_=0
#endif

#else

#define F_INT int                     // 如果没有定义 HAVE_BLAS_ILP64，则将 F_INT 定义为普通的 int
#define F_INT_NPY NPY_INT             // 定义 F_INT_NPY 为普通的整型常量
#define F_INT_PYFMT   "i"             // 定义 F_INT_PYFMT 为普通整型的格式化字符串

#endif

#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F                 // 如果定义了 NO_APPEND_FORTRAN 和 UPPERCASE_FORTRAN，则定义 F_FUNC 为大写 Fortran 函数名
#else
#define F_FUNC(f,F) f                 // 如果定义了 NO_APPEND_FORTRAN 但未定义 UPPERCASE_FORTRAN，则定义 F_FUNC 为普通函数名
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_              // 如果未定义 NO_APPEND_FORTRAN 但定义了 UPPERCASE_FORTRAN，则定义 F_FUNC 为大写 Fortran 函数名加下划线
#else
#define F_FUNC(f,F) f##_              // 否则定义 F_FUNC 为普通函数名加下划线
#endif
#endif

#define PYERR(errobj,message) {PyErr_SetString(errobj,message); goto fail;}   // 定义宏 PYERR，设置 Python 异常并跳转到 fail 标签处
#define PYERR2(errobj,message) {PyErr_Print(); PyErr_SetString(errobj, message); goto fail;}  // 定义宏 PYERR2，打印当前 Python 异常并设置新的异常消息，并跳转到 fail 标签处
#define ISCONTIGUOUS(m) ((m)->flags & CONTIGUOUS)   // 定义宏 ISCONTIGUOUS，检查数组是否是连续的

#define MAX(n1,n2) ((n1) > (n2))?(n1):(n2);           // 定义宏 MAX，返回两个数中较大的数
#define MIN(n1,n2) ((n1) > (n2))?(n2):(n1);           // 定义宏 MIN，返回两个数中较小的数

struct ODR_info_ {
  PyObject* fcn;
  PyObject* fjacb;
  PyObject* fjacd;
  PyObject* pyBeta;
  PyObject* extra_args;
};

typedef struct ODR_info_ ODR_info;                   // 定义结构体 ODR_info 并用 typedef 定义为 ODR_info

static ODR_info odr_global;                         // 定义静态变量 odr_global，类型为 ODR_info 结构体

static PyObject *odr_error=NULL;                    // 定义静态 PyObject 指针变量 odr_error，初始化为 NULL
static PyObject *odr_stop=NULL;                     // 定义静态 PyObject 指针变量 odr_stop，初始化为 NULL

void fcn_callback(F_INT *n, F_INT *m, F_INT *np, F_INT *nq, F_INT *ldn, F_INT *ldm,
          F_INT *ldnp, double *beta, double *xplusd, F_INT *ifixb,
          F_INT *ifixx, F_INT *ldfix, F_INT *ideval, double *f,
          double *fjacb, double *fjacd, F_INT *istop);
          // 声明函数 fcn_callback，接受多个参数，有些为整型指针，有些为双精度浮点数指针

PyObject *gen_output(F_INT n, F_INT m, F_INT np, F_INT nq, F_INT ldwe, F_INT ld2we,
             PyArrayObject *beta, PyArrayObject *work, PyArrayObject *iwork,
             F_INT isodr, F_INT info, int full_output);
             // 声明函数 gen_output，返回 PyObject 指针，接受多个参数，包括整型和 PyArrayObject 指针

PyObject *odr(PyObject *self, PyObject *args, PyObject *kwds);
             // 声明函数 odr，返回 PyObject 指针，接受 self、args 和 kwds 三个 PyObject 指针参数

#define PyArray_CONTIGUOUS(m) (ISCONTIGUOUS(m) ? Py_INCREF(m), m : \
(PyArrayObject *)(PyArray_ContiguousFromObject((PyObject *)(m), \
(m)->descr->type_num, 0,0)))
             // 定义宏 PyArray_CONTIGUOUS，检查数组是否连续，如果是则增加其引用计数并返回该数组，否则将其转换为连续的数组并返回

#define D(dbg) printf("we're here: %i\n", dbg)         // 定义宏 D，打印调试信息

#define EXIST(name,obj) if (obj==NULL){printf("%s\n",name);}   // 定义宏 EXIST，检查对象是否为 NULL，如果是则打印指定名称的信息
```