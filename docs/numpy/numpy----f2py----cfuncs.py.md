# `.\numpy\numpy\f2py\cfuncs.py`

```py
"""
C declarations, CPP macros, and C functions for f2py2e.
Only required declarations/macros/functions will be used.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
# 导入系统模块
import sys
# 导入复制模块
import copy

# 导入当前包的版本信息
from . import __version__

# 获取 f2py2e 的版本号
f2py_version = __version__.version
# 定义错误信息写入函数
errmess = sys.stderr.write

##################### Definitions ##################

# 输出所需的各种定义
outneeds = {'includes0': [], 'includes': [], 'typedefs': [], 'typedefs_generated': [],
            'userincludes': [],
            'cppmacros': [], 'cfuncs': [], 'callbacks': [], 'f90modhooks': [],
            'commonhooks': []}
# 空字典，用于存放需要的定义
needs = {}
# 初始化 includes0 的字典
includes0 = {'includes0': '/*need_includes0*/'}
# 初始化 includes 的字典
includes = {'includes': '/*need_includes*/'}
# 初始化 userincludes 的字典
userincludes = {'userincludes': '/*need_userincludes*/'}
# 初始化 typedefs 的字典
typedefs = {'typedefs': '/*need_typedefs*/'}
# 初始化 typedefs_generated 的字典
typedefs_generated = {'typedefs_generated': '/*need_typedefs_generated*/'}
# 初始化 cppmacros 的字典
cppmacros = {'cppmacros': '/*need_cppmacros*/'}
# 初始化 cfuncs 的字典
cfuncs = {'cfuncs': '/*need_cfuncs*/'}
# 初始化 callbacks 的字典
callbacks = {'callbacks': '/*need_callbacks*/'}
# 初始化 f90modhooks 的字典
f90modhooks = {'f90modhooks': '/*need_f90modhooks*/',
               'initf90modhooksstatic': '/*initf90modhooksstatic*/',
               'initf90modhooksdynamic': '/*initf90modhooksdynamic*/',
               }
# 初始化 commonhooks 的字典
commonhooks = {'commonhooks': '/*need_commonhooks*/',
               'initcommonhooks': '/*need_initcommonhooks*/',
               }

############ Includes ###################

# 定义 includes0 中的一些常用头文件引用
includes0['math.h'] = '#include <math.h>'
includes0['string.h'] = '#include <string.h>'
includes0['setjmp.h'] = '#include <setjmp.h>'

# 引入数组对象头文件并定义 PyArray_API 的唯一符号
includes['arrayobject.h'] = '''#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "arrayobject.h"'''
# 引入 numpy 的数学头文件
includes['npy_math.h'] = '#include "numpy/npy_math.h"'

# 重新定义 arrayobject.h 的包含，引入 fortranobject.h
includes['arrayobject.h'] = '#include "fortranobject.h"'
# 引入可变参数头文件
includes['stdarg.h'] = '#include <stdarg.h>'

############# Type definitions ###############

# 定义各种数据类型的 typedef

typedefs['unsigned_char'] = 'typedef unsigned char unsigned_char;'
typedefs['unsigned_short'] = 'typedef unsigned short unsigned_short;'
typedefs['unsigned_long'] = 'typedef unsigned long unsigned_long;'
typedefs['signed_char'] = 'typedef signed char signed_char;'
typedefs['long_long'] = """
#if defined(NPY_OS_WIN32)
typedef __int64 long_long;
#else
typedef long long long_long;
typedef unsigned long long unsigned_long_long;
#endif
"""
typedefs['unsigned_long_long'] = """
#if defined(NPY_OS_WIN32)
typedef __uint64 long_long;
#else
typedef unsigned long long unsigned_long_long;
#endif
"""
typedefs['long_double'] = """
#ifndef _LONG_DOUBLE
typedef long double long_double;
#endif
"""
typedefs[
    'complex_long_double'] = 'typedef struct {long double r,i;} complex_long_double;'
typedefs['complex_float'] = 'typedef struct {float r,i;} complex_float;'
typedefs['complex_double'] = 'typedef struct {double r,i;} complex_double;'
# 定义 C 语言中的 typedef，将 'string' 映射为 char *
typedefs['string'] = """typedef char * string;"""
# 定义 C 语言中的 typedef，将 'character' 映射为 char
typedefs['character'] = """typedef char character;"""


############### CPP macros ####################

# 定义一个名为 CFUNCSMESS 的宏，用于在调试模式下打印调试信息到 stderr
# 使用 fprintf 函数打印格式化字符串到 stderr
# DEBUGCFUNCS 宏开启时，会打印消息，并使用 PyObject_Print 打印 Python 对象到 stderr
cppmacros['CFUNCSMESS'] = """
#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,\"debug-capi:\"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \\
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\
    fprintf(stderr,\"\\n\");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif
"""

# 定义名为 F_FUNC 的宏，根据不同的预处理器宏组合生成 Fortran 函数名称
cppmacros['F_FUNC'] = """
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif
"""

# 定义名为 F_WRAPPEDFUNC 的宏，用于生成 Fortran 函数的包装函数名称
cppmacros['F_WRAPPEDFUNC'] = """
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F
#else
#define F_WRAPPEDFUNC(f,F) _f2pywrap##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F##_
#else
#define F_WRAPPEDFUNC(f,F) _f2pywrap##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F
#else
#define F_WRAPPEDFUNC(f,F) f2pywrap##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F##_
#else
#define F_WRAPPEDFUNC(f,F) f2pywrap##f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f##_,F##_)
#else
#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f,F)
#endif
"""

# 定义名为 F_MODFUNC 的宏，用于生成 Fortran 模块函数的名称，根据不同的编译器设置不同的命名规则
cppmacros['F_MODFUNC'] = """
#if defined(F90MOD2CCONV1) /*E.g. Compaq Fortran */
#if defined(NO_APPEND_FORTRAN)
#define F_MODFUNCNAME(m,f) $ ## m ## $ ## f
#else
#define F_MODFUNCNAME(m,f) $ ## m ## $ ## f ## _
#endif
#endif

#if defined(F90MOD2CCONV2) /*E.g. IBM XL Fortran, not tested though */
#if defined(NO_APPEND_FORTRAN)
#define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f
#else
#define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f ## _
#endif
#endif

#if defined(F90MOD2CCONV3) /*E.g. MIPSPro Compilers */
#if defined(NO_APPEND_FORTRAN)
#define F_MODFUNCNAME(m,f)  f ## .in. ## m
#else
#define F_MODFUNCNAME(m,f)  f ## .in. ## m ## _
#endif
#endif
/*
#if defined(UPPERCASE_FORTRAN)
#define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(M,F)
#else
#define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(m,f)
#endif
*/

#define F_MODFUNC(m,f) (*(f2pymodstruct##m##.##f))
"""

# 定义一个名为 SWAPUNSAFE 的宏，用于不安全地交换两个变量的值
# 使用位运算来交换两个变量的值
cppmacros['SWAPUNSAFE'] = """
#define SWAP(a,b) (size_t)(a) = ((size_t)(a) ^ (size_t)(b));\\
 (size_t)(b) = ((size_t)(a) ^ (size_t)(b));\\
 (size_t)(a) = ((size_t)(a) ^ (size_t)(b))
"""
cppmacros['SWAP'] = """
#define SWAP(a,b,t) {\\
    t *c;\\
    c = a;\\
    a = b;\\
    b = c;}
"""
# 定义宏 SWAP(a, b, t)，用于交换变量 a 和 b 的值，其中 t 是类型
cppmacros['PRINTPYOBJERR'] = """
#define PRINTPYOBJERR(obj)\\
    fprintf(stderr,\"#modulename#.error is related to \");\\
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\
    fprintf(stderr,\"\\n\");
"""
# 定义宏 PRINTPYOBJERR(obj)，用于在标准错误流输出关于 obj 的 Python 对象的错误信息
cppmacros['MINMAX'] = """
#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif
"""
# 定义宏 MIN 和 MAX，分别返回两个值中的最小值和最大值
cppmacros['len..'] = """
/* See fortranobject.h for definitions. The macros here are provided for BC. */
#define rank f2py_rank
#define shape f2py_shape
#define fshape f2py_shape
#define len f2py_len
#define flen f2py_flen
#define slen f2py_slen
#define size f2py_size
"""
# 定义一系列宏，用于处理 Fortran 对象的长度、维度等
cppmacros['pyobj_from_char1'] = r"""
#define pyobj_from_char1(v) (PyLong_FromLong(v))
"""
# 定义宏 pyobj_from_char1(v)，将字符转换为 Python 的整数对象
cppmacros['pyobj_from_short1'] = r"""
#define pyobj_from_short1(v) (PyLong_FromLong(v))
"""
# 定义宏 pyobj_from_short1(v)，将短整型转换为 Python 的整数对象
needs['pyobj_from_int1'] = ['signed_char']
cppmacros['pyobj_from_int1'] = r"""
#define pyobj_from_int1(v) (PyLong_FromLong(v))
"""
# 定义宏 pyobj_from_int1(v)，将整型转换为 Python 的整数对象
cppmacros['pyobj_from_long1'] = r"""
#define pyobj_from_long1(v) (PyLong_FromLong(v))
"""
# 定义宏 pyobj_from_long1(v)，将长整型转换为 Python 的整数对象
needs['pyobj_from_long_long1'] = ['long_long']
cppmacros['pyobj_from_long_long1'] = """
#ifdef HAVE_LONG_LONG
#define pyobj_from_long_long1(v) (PyLong_FromLongLong(v))
#else
#warning HAVE_LONG_LONG is not available. Redefining pyobj_from_long_long.
#define pyobj_from_long_long1(v) (PyLong_FromLong(v))
#endif
"""
# 定义宏 pyobj_from_long_long1(v)，根据是否有长长整型支持，将值转换为 Python 的整数对象
needs['pyobj_from_long_double1'] = ['long_double']
cppmacros['pyobj_from_long_double1'] = """
#define pyobj_from_long_double1(v) (PyFloat_FromDouble(v))"""
# 定义宏 pyobj_from_long_double1(v)，将长双精度浮点数转换为 Python 的浮点数对象
cppmacros['pyobj_from_double1'] = """
#define pyobj_from_double1(v) (PyFloat_FromDouble(v))"""
# 定义宏 pyobj_from_double1(v)，将双精度浮点数转换为 Python 的浮点数对象
cppmacros['pyobj_from_float1'] = """
#define pyobj_from_float1(v) (PyFloat_FromDouble(v))"""
# 定义宏 pyobj_from_float1(v)，将单精度浮点数转换为 Python 的浮点数对象
needs['pyobj_from_complex_long_double1'] = ['complex_long_double']
cppmacros['pyobj_from_complex_long_double1'] = """
#define pyobj_from_complex_long_double1(v) (PyComplex_FromDoubles(v.r,v.i))"""
# 定义宏 pyobj_from_complex_long_double1(v)，将长双精度复数转换为 Python 的复数对象
needs['pyobj_from_complex_double1'] = ['complex_double']
cppmacros['pyobj_from_complex_double1'] = """
#define pyobj_from_complex_double1(v) (PyComplex_FromDoubles(v.r,v.i))"""
# 定义宏 pyobj_from_complex_double1(v)，将双精度复数转换为 Python 的复数对象
needs['pyobj_from_complex_float1'] = ['complex_float']
cppmacros['pyobj_from_complex_float1'] = """
#define pyobj_from_complex_float1(v) (PyComplex_FromDoubles(v.r,v.i))"""
# 定义宏 pyobj_from_complex_float1(v)，将单精度复数转换为 Python 的复数对象
needs['pyobj_from_string1'] = ['string']
cppmacros['pyobj_from_string1'] = """
#define pyobj_from_string1(v) (PyUnicode_FromString((char *)v))"""
# 定义宏 pyobj_from_string1(v)，将字符串转换为 Python 的 Unicode 对象
needs['pyobj_from_string1size'] = ['string']
cppmacros['pyobj_from_string1size'] = """
#define pyobj_from_string1size(v,len) (PyUnicode_FromStringAndSize((char *)v, len))"""
# 定义宏 pyobj_from_string1size(v, len)，根据长度将字符串转换为 Python 的 Unicode 对象
needs['TRYPYARRAYTEMPLATE'] = ['PRINTPYOBJERR']
cppmacros['TRYPYARRAYTEMPLATE'] = """
/* New SciPy */
"""
# 定义 TRYPYARRAYTEMPLATE 宏，用于新的 SciPy
# 定义一个宏，用于处理 NPY_STRING 类型的数组赋值
#define TRYPYARRAYTEMPLATECHAR case NPY_STRING: *(char *)(PyArray_DATA(arr))=*v; break;

# 定义一个宏，用于处理 NPY_LONG 类型的数组赋值
#define TRYPYARRAYTEMPLATELONG case NPY_LONG: *(long *)(PyArray_DATA(arr))=*v; break;

# 定义一个宏，用于处理 NPY_OBJECT 类型的数组赋值
#define TRYPYARRAYTEMPLATEOBJECT case NPY_OBJECT: PyArray_SETITEM(arr,PyArray_DATA(arr),pyobj_from_ ## ctype ## 1(*v)); break;

# 定义一个宏，用于处理不同类型的数组赋值
# 参数 ctype：数组类型
# 参数 typecode：数组类型代码
#define TRYPYARRAYTEMPLATE(ctype,typecode) \\
        PyArrayObject *arr = NULL;\\  # 初始化一个 PyArrayObject 对象
        if (!obj) return -2;\\  # 如果 obj 为空，则返回 -2
        if (!PyArray_Check(obj)) return -1;\\  # 如果 obj 不是数组对象，则返回 -1
        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,\"TRYPYARRAYTEMPLATE:\");PRINTPYOBJERR(obj);return 0;}\\  # 将 obj 转换成 PyArrayObject 对象
        if (PyArray_DESCR(arr)->type==typecode)  {*(ctype *)(PyArray_DATA(arr))=*v; return 1;}\\  # 如果数组的类型与给定的类型代码相同，则直接赋值返回1
        switch (PyArray_TYPE(arr)) {\\  # 根据数组类型代码进行对应的赋值操作
                case NPY_DOUBLE: *(npy_double *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_INT: *(npy_int *)(PyArray_DATA(arr))=*v; break;\\
                ...
                ...  # 其他类型的赋值操作
                ...
        default: return -2;\\  # 默认情况下，返回 -2
        };\\
        return 1  # 返回 1 表示成功
"""

# 需要的其他函数
needs['TRYCOMPLEXPYARRAYTEMPLATE'] = ['PRINTPYOBJERR']

# 宏定义，用于处理复数类型的数组赋值
cppmacros['TRYCOMPLEXPYARRAYTEMPLATE'] = """
#define TRYCOMPLEXPYARRAYTEMPLATEOBJECT case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_complex_ ## ctype ## 1((*v))); break;
#define TRYCOMPLEXPYARRAYTEMPLATE(ctype,typecode)\\
        // 定义宏 TRYCOMPLEXPYARRAYTEMPLATE，用于处理复数数组
        PyArrayObject *arr = NULL;\\
        // 初始化 PyArrayObject 指针 arr，置空
        if (!obj) return -2;\\
        // 如果 obj 为 NULL，则返回错误码 -2
        if (!PyArray_Check(obj)) return -1;\\
        // 如果 obj 不是 NumPy 数组，则返回错误码 -1
        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,\"TRYCOMPLEXPYARRAYTEMPLATE:\");PRINTPYOBJERR(obj);return 0;}\\
        // 将 obj 转换为 PyArrayObject 类型，若转换失败则打印错误信息并返回 0
        if (PyArray_DESCR(arr)->type==typecode) {\\
            // 如果数组的数据类型与指定的 typecode 相同
            *(ctype *)(PyArray_DATA(arr))=(*v).r;\\
            // 将实部 (*v).r 写入数组的首地址
            *(ctype *)(PyArray_DATA(arr)+sizeof(ctype))=(*v).i;\\
            // 将虚部 (*v).i 写入数组的首地址加上一个 ctype 类型的大小后的地址
            return 1;\\
            // 返回 1，表示成功写入复数到数组中
        }\\
        switch (PyArray_TYPE(arr)) {\\
                // 根据数组的数据类型进行进一步处理
                case NPY_CDOUBLE: *(npy_double *)(PyArray_DATA(arr))=(*v).r;\\
                                  *(npy_double *)(PyArray_DATA(arr)+sizeof(npy_double))=(*v).i;\\
                                  break;\\
                                  // 处理复数类型为双精度浮点型
                case NPY_CFLOAT: *(npy_float *)(PyArray_DATA(arr))=(*v).r;\\
                                 *(npy_float *)(PyArray_DATA(arr)+sizeof(npy_float))=(*v).i;\\
                                 break;\\
                                 // 处理复数类型为单精度浮点型
                case NPY_DOUBLE: *(npy_double *)(PyArray_DATA(arr))=(*v).r; break;\\
                                 // 处理复数类型为双精度浮点型
                case NPY_LONG: *(npy_long *)(PyArray_DATA(arr))=(*v).r; break;\\
                               // 处理复数类型为长整型
                case NPY_FLOAT: *(npy_float *)(PyArray_DATA(arr))=(*v).r; break;\\
                                // 处理复数类型为单精度浮点型
                case NPY_INT: *(npy_int *)(PyArray_DATA(arr))=(*v).r; break;\\
                              // 处理复数类型为整型
                case NPY_SHORT: *(npy_short *)(PyArray_DATA(arr))=(*v).r; break;\\
                                // 处理复数类型为短整型
                case NPY_UBYTE: *(npy_ubyte *)(PyArray_DATA(arr))=(*v).r; break;\\
                                // 处理复数类型为无符号字节型
                case NPY_BYTE: *(npy_byte *)(PyArray_DATA(arr))=(*v).r; break;\\
                               // 处理复数类型为字节型
                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=((*v).r!=0 && (*v).i!=0); break;\\
                               // 处理复数类型为布尔型
                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=(*v).r; break;\\
                                 // 处理复数类型为无符号短整型
                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=(*v).r; break;\\
                               // 处理复数类型为无符号整型
                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=(*v).r; break;\\
                                // 处理复数类型为无符号长整型
                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=(*v).r; break;\\
                                   // 处理复数类型为长长整型
                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=(*v).r; break;\\
                                    // 处理复数类型为无符号长长整型
                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r; break;\\
                                    // 处理复数类型为长双精度浮点型
                case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r;\\
                                      *(npy_longdouble *)(PyArray_DATA(arr)+sizeof(npy_longdouble))=(*v).i;\\
                                      break;\\
                                      // 处理复数类型为长双精度浮点型
                case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_complex_ ## ctype ## 1((*v))); break;\\
                                 // 处理复数类型为对象型
                default: return -2;\\
                         // 默认情况，返回错误码 -2
        };\\
        return -1;
        // 返回错误码 -1，表示未能处理复数类型数组
/*
GETSTRFROMPYTUPLE宏用于从Python元组中获取字符串。

#define GETSTRFROMPYTUPLE(tuple,index,str,len) {\
        PyObject *rv_cb_str = PyTuple_GetItem((tuple),(index));\
        // 检查返回的对象是否为NULL，如果是则跳转到错误处理标签capi_fail
        if (rv_cb_str == NULL)\
            goto capi_fail;\
        // 检查返回的对象是否为字节对象（PyBytesObject）
        if (PyBytes_Check(rv_cb_str)) {\
            // 将字符串的最后一个字符设置为'\0'，然后使用STRINGCOPYN宏将PyBytesObject的内容复制到str中
            str[len-1]='\0';\
            STRINGCOPYN((str),PyBytes_AS_STRING((PyBytesObject*)rv_cb_str),(len));\
        } else {\
            // 如果返回的对象不是字节对象，则调用PRINTPYOBJERR宏输出错误信息
            PRINTPYOBJERR(rv_cb_str);\
            // 设置Python异常字符串为模块名_error，并跳转到错误处理标签capi_fail
            PyErr_SetString(#modulename#_error,"string object expected");\
            goto capi_fail;\
        }\
    }
*/


/*
GETSCALARFROMPYTUPLE宏用于从Python元组中获取标量值。

#define GETSCALARFROMPYTUPLE(tuple,index,var,ctype,mess) {\
        // 从元组中获取指定索引位置的对象，存储在capi_tmp中
        if ((capi_tmp = PyTuple_GetItem((tuple),(index)))==NULL) goto capi_fail;\
        // 使用ctype##_from_pyobj宏将capi_tmp转换为ctype类型的标量，并检查转换是否成功
        if (!(ctype##_from_pyobj((var),capi_tmp,mess)))\
            goto capi_fail;\
    }
*/


/*
FAILNULL宏用于检查指针是否为NULL，如果是则设置内存错误异常，并跳转到错误处理标签capi_fail。

#define FAILNULL(p) do {                                            \
    // 如果指针p为NULL，则设置Python内存错误异常，并跳转到错误处理标签capi_fail\
    if ((p) == NULL) {                                              \
        PyErr_SetString(PyExc_MemoryError, "NULL pointer found");   \
        goto capi_fail;                                             \
    }                                                               \
} while (0)
*/


/*
MEMCOPY宏用于执行内存拷贝操作，并在拷贝过程中检查目标地址和源地址是否为NULL。

#define MEMCOPY(to,from,n)\
    do { FAILNULL(to); FAILNULL(from); (void)memcpy(to,from,n); } while (0)
*/


/*
STRINGMALLOC宏用于分配字符串内存，并在分配失败时设置内存错误异常并跳转到错误处理标签capi_fail。

#define STRINGMALLOC(str,len)\
    if ((str = (string)malloc(len+1)) == NULL) {\
        PyErr_SetString(PyExc_MemoryError, "out of memory");\
        goto capi_fail;\
    } else {\
        (str)[len] = '\0';\
    }
*/


/*
STRINGFREE宏用于释放字符串内存。

#define STRINGFREE(str) do {if (!(str == NULL)) free(str);} while (0)
*/
/*
STRINGPADN(to, N, NULLVALUE, PADDING) is a macro that pads the string `to` with `PADDING`
character up to `N` bytes if the end of `to` is filled with `NULLVALUE` characters.

Parameters:
- `to`: Pointer to the string to pad.
- `N`: Size in bytes of the buffer `to`.
- `NULLVALUE`: The character considered as null (to be replaced).
- `PADDING`: The character used for padding.

The macro iterates backwards through the string `to` and replaces `NULLVALUE` characters
with `PADDING` until it reaches the beginning of the string or encounters a non-NULLVALUE
character.
*/
#define STRINGPADN(to, N, NULLVALUE, PADDING)                   \\
    do {                                                        \\
        int _m = (N);                                           \\
        char *_to = (to);                                       \\
        for (_m -= 1; _m >= 0 && _to[_m] == NULLVALUE; _m--) {  \\
             _to[_m] = PADDING;                                 \\
        }                                                       \\
    } while (0)



/*
STRINGCOPYN(to, from, N) is a macro that copies `N` bytes from the buffer `from` to the buffer `to`.

Parameters:
- `to`: Pointer to the destination buffer.
- `from`: Pointer to the source buffer.
- `N`: Number of bytes to copy.

The macro uses `strncpy` to copy up to `N` bytes from `from` to `to`. It also checks for NULL
pointers using the `FAILNULL` macro for `to` and `from` before copying.
*/
#define STRINGCOPYN(to, from, N)                                  \\
    do {                                                        \\
        int _m = (N);                                           \\
        char *_to = (to);                                       \\
        char *_from = (from);                                   \\
        FAILNULL(_to); FAILNULL(_from);                         \\
        (void)strncpy(_to, _from, _m);                           \\
    } while (0)
    'ARRSIZE'] = '#define ARRSIZE(dims,rank) (_PyArray_multiply_list(dims,rank))'


# 将字符串 '#define ARRSIZE(dims,rank) (_PyArray_multiply_list(dims,rank))' 赋给字典的键 'ARRSIZE'
# 定义宏 'OLDPYNUM'，当 'OLDPYNUM' 已定义时报错，提示需要安装 NumPy 0.13 或更高版本
cppmacros['OLDPYNUM'] = """
#ifdef OLDPYNUM
#error You need to install NumPy version 0.13 or higher. See https://scipy.org/install.html
#endif
"""

# 定义宏 'F2PY_THREAD_LOCAL_DECL'，根据不同平台和编译器设置线程局部存储的声明
cppmacros["F2PY_THREAD_LOCAL_DECL"] = """
#ifndef F2PY_THREAD_LOCAL_DECL
#if defined(_MSC_VER)
#define F2PY_THREAD_LOCAL_DECL __declspec(thread)
#elif defined(NPY_OS_MINGW)
#define F2PY_THREAD_LOCAL_DECL __thread
#elif defined(__STDC_VERSION__) \\
      && (__STDC_VERSION__ >= 201112L) \\
      && !defined(__STDC_NO_THREADS__) \\
      && (!defined(__GLIBC__) || __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 12)) \\
      && !defined(NPY_OS_OPENBSD) && !defined(NPY_OS_HAIKU)
/* __STDC_NO_THREADS__ was first defined in a maintenance release of glibc 2.12,
   see https://lists.gnu.org/archive/html/commit-hurd/2012-07/msg00180.html,
   so `!defined(__STDC_NO_THREADS__)` may give false positive for the existence
   of `threads.h` when using an older release of glibc 2.12
   See gh-19437 for details on OpenBSD */
#include <threads.h>
#define F2PY_THREAD_LOCAL_DECL thread_local
#elif defined(__GNUC__) \\
      && (__GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 4)))
#define F2PY_THREAD_LOCAL_DECL __thread
#endif
#endif
"""

################# C functions ###############

# 定义静态 C 函数 'calcarrindex'，计算多维数组中索引对应的线性位置
cfuncs['calcarrindex'] = """
static int calcarrindex(int *i,PyArrayObject *arr) {
    int k,ii = i[0];
    for (k=1; k < PyArray_NDIM(arr); k++)
        ii += (ii*(PyArray_DIM(arr,k) - 1)+i[k]); /* assuming contiguous arr */
    return ii;
}"""

# 定义静态 C 函数 'calcarrindextr'，计算转置后的多维数组中索引对应的线性位置
cfuncs['calcarrindextr'] = """
static int calcarrindextr(int *i,PyArrayObject *arr) {
    int k,ii = i[PyArray_NDIM(arr)-1];
    for (k=1; k < PyArray_NDIM(arr); k++)
        ii += (ii*(PyArray_DIM(arr,PyArray_NDIM(arr)-k-1) - 1)+i[PyArray_NDIM(arr)-k-1]); /* assuming contiguous arr */
    return ii;
}"""

# 定义静态 C 函数 'forcomb'，用于生成多维数组的组合索引
cfuncs['forcomb'] = """
static struct { int nd;npy_intp *d;int *i,*i_tr,tr; } forcombcache;
static int initforcomb(npy_intp *dims,int nd,int tr) {
  int k;
  if (dims==NULL) return 0;
  if (nd<0) return 0;
  forcombcache.nd = nd;
  forcombcache.d = dims;
  forcombcache.tr = tr;
  if ((forcombcache.i = (int *)malloc(sizeof(int)*nd))==NULL) return 0;
  if ((forcombcache.i_tr = (int *)malloc(sizeof(int)*nd))==NULL) return 0;
  for (k=1;k<nd;k++) {
    forcombcache.i[k] = forcombcache.i_tr[nd-k-1] = 0;
  }
  forcombcache.i[0] = forcombcache.i_tr[nd-1] = -1;
  return 1;
}

static int *nextforcomb(void) {
  int j,*i,*i_tr,k;
  int nd=forcombcache.nd;
  if ((i=forcombcache.i) == NULL) return NULL;
  if ((i_tr=forcombcache.i_tr) == NULL) return NULL;
  if (forcombcache.d == NULL) return NULL;
  i[0]++;
  if (i[0]==forcombcache.d[0]) {
    j=1;
    while ((j<nd) && (i[j]==forcombcache.d[j]-1)) j++;
    if (j==nd) {
      free(i);
      free(i_tr);
      return NULL;
    }
    for (k=0;k<j;k++) i[k] = i_tr[nd-k-1] = 0;
    i[j]++;
    i_tr[nd-j-1]++;
  } else
    i_tr[nd-1]++;
  if (forcombcache.tr) return i_tr;
  return i;
}"""
/*
  Add the function `try_pyarr_from_string` to the needs dictionary, requiring these functionalities: 'STRINGCOPYN', 'PRINTPYOBJERR', 'string'.
*/
needs['try_pyarr_from_string'] = ['STRINGCOPYN', 'PRINTPYOBJERR', 'string']
/*
  Define the C function `try_pyarr_from_string` which attempts to copy `str[:len(obj)]` to the data of an `ndarray`.

  If `obj` is an `ndarray`, it is expected to be contiguous.

  If `len` is set to -1, `str` must be null-terminated.
*/
cfuncs['try_pyarr_from_string'] = """
static int try_pyarr_from_string(PyObject *obj,
                                 const string str, const int len) {
#ifdef DEBUGCFUNCS
fprintf(stderr, "try_pyarr_from_string(str='%s', len=%d, obj=%p)\\n",
        (char*)str,len, obj);
#endif
    if (!obj) return -2; /* Object missing */
    if (obj == Py_None) return -1; /* None */
    if (!PyArray_Check(obj)) goto capi_fail; /* not an ndarray */
    if (PyArray_Check(obj)) {
        PyArrayObject *arr = (PyArrayObject *)obj;
        assert(ISCONTIGUOUS(arr));
        string buf = PyArray_DATA(arr);
        npy_intp n = len;
        if (n == -1) {
            /* Assuming null-terminated str. */
            n = strlen(str);
        }
        if (n > PyArray_NBYTES(arr)) {
            n = PyArray_NBYTES(arr);
        }
        STRINGCOPYN(buf, str, n); /* Copy at most `n` bytes from `str` to `buf` */
        return 1;
    }
capi_fail:
    PRINTPYOBJERR(obj); /* Print error message for Python object `obj` */
    PyErr_SetString(#modulename#_error, "try_pyarr_from_string failed"); /* Set error message for module-specific error */
    return 0;
}
"""
/*
  Add the function `string_from_pyobj` to the needs dictionary, requiring these functionalities: 'string', 'STRINGMALLOC', 'STRINGCOPYN'.
*/
needs['string_from_pyobj'] = ['string', 'STRINGMALLOC', 'STRINGCOPYN']
/*
  Define the C function `string_from_pyobj` which creates a new string buffer `str` of maximum length `len` from a Python string-like object `obj`.

  The string buffer has the size `len` or the size of `inistr` when `len==-1`.

  The string buffer is padded with blanks. In Fortran, trailing blanks are insignificant unlike C nulls.
 */
cfuncs['string_from_pyobj'] = """
static int
string_from_pyobj(string *str, int *len, const string inistr, PyObject *obj,
                  const char *errmess)
{
    PyObject *tmp = NULL;
    string buf = NULL;
    npy_intp n = -1;
#ifdef DEBUGCFUNCS
fprintf(stderr,"string_from_pyobj(str='%s',len=%d,inistr='%s',obj=%p)\\n",
               (char*)str, *len, (char *)inistr, obj);
#endif
    if (obj == Py_None) {
        n = strlen(inistr);
        buf = inistr;
    }
    else if (PyArray_Check(obj)) {
        PyArrayObject *arr = (PyArrayObject *)obj;
        if (!ISCONTIGUOUS(arr)) {
            PyErr_SetString(PyExc_ValueError,
                            "array object is non-contiguous.");
            goto capi_fail;
        }
        n = PyArray_NBYTES(arr);
        buf = PyArray_DATA(arr);
        n = strnlen(buf, n);
    }
    else {
        // 如果对象是字节对象，则直接引用它
        if (PyBytes_Check(obj)) {
            tmp = obj;
            Py_INCREF(tmp);
        }
        // 如果对象是 Unicode 对象，则转换为 ASCII 字符串
        else if (PyUnicode_Check(obj)) {
            tmp = PyUnicode_AsASCIIString(obj);
        }
        // 对象既不是字节对象也不是 Unicode 对象，将其转换为字符串表示
        else {
            PyObject *tmp2;
            tmp2 = PyObject_Str(obj);
            // 如果转换成功，则再将其转换为 ASCII 字符串
            if (tmp2) {
                tmp = PyUnicode_AsASCIIString(tmp2);
                Py_DECREF(tmp2);
            }
            else {
                tmp = NULL;
            }
        }
        // 如果转换失败，跳转到错误处理标签
        if (tmp == NULL) goto capi_fail;
        // 获取 tmp 对象的字节大小
        n = PyBytes_GET_SIZE(tmp);
        // 获取 tmp 对象的字节内容
        buf = PyBytes_AS_STRING(tmp);
    }
    // 如果 *len 为 -1，检查 n 是否超过 32 位整数的最大值
    if (*len == -1) {
        /* TODO: change the type of `len` so that we can remove this */
        if (n > NPY_MAX_INT) {
            // 如果 n 超过 32 位整数的最大值，设置 OverflowError 并跳转到错误处理标签
            PyErr_SetString(PyExc_OverflowError,
                            "object too large for a 32-bit int");
            goto capi_fail;
        }
        // 将 n 赋值给 *len
        *len = n;
    }
    // 如果 *len 小于 n，截断输入 buf 的末尾 (len-n) 字节
    else if (*len < n) {
        /* discard the last (len-n) bytes of input buf */
        n = *len;
    }
    // 检查 n 和 *len 是否小于 0，buf 是否为空，若满足条件则跳转到错误处理标签
    if (n < 0 || *len < 0 || buf == NULL) {
        goto capi_fail;
    }
    // 分配 *str 的内存，大小为 (*len + 1)
    STRINGMALLOC(*str, *len);
    // 如果 n 小于 *len，用 '\0' 填充固定宽度字符串的末尾
    if (n < *len) {
        /*
          Pad fixed-width string with nulls. The caller will replace
          nulls with blanks when the corresponding argument is not
          intent(c).
        */
        memset(*str + n, '\0', *len - n);
    }
    // 将 buf 的前 n 个字节复制到 *str 中
    STRINGCOPYN(*str, buf, n);
    // 释放 tmp 对象
    Py_XDECREF(tmp);
    // 返回成功
    return 1;
# 定义错误处理标签，用于处理错误时清理临时变量并设置异常
capi_fail:
    # 释放临时对象 tmp 的 Python 引用
    Py_XDECREF(tmp);
    {
        # 获取当前的 Python 异常对象
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            # 如果没有异常发生，则将 #modulename#_error 赋给 err
            err = #modulename#_error;
        }
        # 设置异常，并使用给定的错误消息 errmess
        PyErr_SetString(err, errmess);
    }
    # 返回 0 表示失败
    return 0;
}


```  
# 为 C 函数 'character_from_pyobj' 添加注释  
static int
character_from_pyobj(character* v, PyObject *obj, const char *errmess) {
    if (PyBytes_Check(obj)) {
        /* 空字节总是有一个结尾的 null 字符，因此可以安全地解引用 */
        *v = PyBytes_AS_STRING(obj)[0];
        return 1;
    } else if (PyUnicode_Check(obj)) {
        # 将 Python Unicode 对象转换为 ASCII 字符串
        PyObject* tmp = PyUnicode_AsASCIIString(obj);
        if (tmp != NULL) {
            # 从临时对象中获取第一个字符，并释放临时对象
            *v = PyBytes_AS_STRING(tmp)[0];
            Py_DECREF(tmp);
            return 1;
        }
    } else if (PyArray_Check(obj)) {
        # 将 Python 数组对象转换为 PyArrayObject
        PyArrayObject* arr = (PyArrayObject*)obj;
        if (F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(arr)) {
            # 获取数组的第一个字节作为字符
            *v = PyArray_BYTES(arr)[0];
            return 1;
        } else if (F2PY_IS_UNICODE_ARRAY(arr)) {
            // TODO: 当 numpy 支持 1 字节和 2 字节的 Unicode 数据类型时更新
            # 创建 Unicode 对象，从数组中获取数据
            PyObject* tmp = PyUnicode_FromKindAndData(
                              PyUnicode_4BYTE_KIND,
                              PyArray_BYTES(arr),
                              (PyArray_NBYTES(arr)>0?1:0));
            if (tmp != NULL) {
                if (character_from_pyobj(v, tmp, errmess)) {
                    # 释放临时对象
                    Py_DECREF(tmp);
                    return 1;
                }
                Py_DECREF(tmp);
            }
        }
    } else if (PySequence_Check(obj)) {
        # 从序列对象中获取第一个元素
        PyObject* tmp = PySequence_GetItem(obj,0);
        if (tmp != NULL) {
            if (character_from_pyobj(v, tmp, errmess)) {
                # 释放临时对象
                Py_DECREF(tmp);
                return 1;
            }
            Py_DECREF(tmp);
        }
    }
    {
        /* TODO: 需要清理这种错误处理（以及大多数其他）的方法。 */
        # 定义错误消息缓冲区
        char mess[F2PY_MESSAGE_BUFFER_SIZE];
        strcpy(mess, errmess);
        # 获取当前的 Python 异常对象
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            # 如果没有异常发生，则使用 PyExc_TypeError
            err = PyExc_TypeError;
            Py_INCREF(err);
        }
        else {
            Py_INCREF(err);
            PyErr_Clear();
        }
        # 格式化错误消息，描述预期的对象类型
        sprintf(mess + strlen(mess),
                " -- expected str|bytes|sequence-of-str-or-bytes, got ");
        f2py_describe(obj, mess + strlen(mess));
        # 设置异常并释放异常对象
        PyErr_SetString(err, mess);
        Py_DECREF(err);
    }
    # 返回 0 表示失败
    return 0;
}


```py  
# 为 C 函数 'char_from_pyobj' 添加注释  
static int
char_from_pyobj(char* v, PyObject *obj, const char *errmess) {
    int i = 0;
    # 将 Python 对象转换为整数，并将其转换为 char 类型
    if (int_from_pyobj(&i, obj, errmess)) {
        *v = (char)i;
        return 1;
    }
    # 返回 0 表示失败
    return 0;
}
/*
 * 从 Python 对象中提取 signed_char 类型的值，并存入 v 中
 *
 * Args:
 *     v: 指向 signed_char 的指针，用于存储提取的值
 *     obj: 要提取值的 PyObject 对象
 *     errmess: 错误消息，如果提取失败时设置的错误信息
 *
 * Returns:
 *     提取成功返回 1，失败返回 0
 */
signed_char_from_pyobj(signed_char* v, PyObject *obj, const char *errmess) {
    int i = 0;
    // 调用 int_from_pyobj 函数尝试从 obj 中提取整数值 i
    if (int_from_pyobj(&i, obj, errmess)) {
        // 将整数值 i 转换为 signed_char 类型并存入 v
        *v = (signed_char)i;
        return 1;
    }
    return 0;
}
    # 检查是否存在临时对象
    if (tmp) {
        # 将临时对象转换为长长整型并赋值给*v，释放临时对象
        *v = PyLong_AsLongLong(tmp);
        Py_DECREF(tmp);
        # 返回条件：如果*v为-1且发生了异常则返回假，否则返回真
        return !(*v == -1 && PyErr_Occurred());
    }

    # 如果对象是复数类型，清除当前异常状态并获取对象的'real'属性
    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj, "real");
    }
    # 如果对象是字节字符串或Unicode字符串类型，则无操作
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    # 如果对象是序列类型，则清除当前异常状态并获取序列的第一个元素
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    # 如果存在临时对象
    if (tmp) {
        # 尝试将临时对象转换为长长整型，如果失败则释放临时对象并返回1
        if (long_long_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }
    {
        # 检查是否发生了异常
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            # 如果没有异常，则设置错误字符串为#modulename#_error
            err = #modulename#_error;
        }
        # 设置异常并指定错误信息
        PyErr_SetString(err, errmess);
    }
    # 返回0，表示函数执行完成
    return 0;
# 'long_double_from_pyobj' 函数的实现，将 Python 对象转换为 long_double 类型
cfuncs['long_double_from_pyobj'] = """
static int
long_double_from_pyobj(long_double* v, PyObject *obj, const char *errmess)
{
    double d=0;
    # 检查是否是 NumPy 数组标量
    if (PyArray_CheckScalar(obj)){
        # 如果是长双精度数组标量，直接转换为 long_double 类型
        if PyArray_IsScalar(obj, LongDouble) {
            PyArray_ScalarAsCtype(obj, v);
            return 1;
        }
        # 如果是长双精度数组，从数据中读取并存入 v
        else if (PyArray_Check(obj) && PyArray_TYPE(obj) == NPY_LONGDOUBLE) {
            (*v) = *((npy_longdouble *)PyArray_DATA(obj));
            return 1;
        }
    }
    # 如果无法直接转换为长双精度数组，则尝试转换为双精度浮点数
    if (double_from_pyobj(&d, obj, errmess)) {
        *v = (long_double)d;
        return 1;
    }
    # 转换失败，返回 0
    return 0;
}
"""

# 'double_from_pyobj' 函数的实现，将 Python 对象转换为 double 类型
cfuncs['double_from_pyobj'] = """
static int
double_from_pyobj(double* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;
    # 如果是 Python 浮点数对象，则直接转换为 double 类型
    if (PyFloat_Check(obj)) {
        *v = PyFloat_AsDouble(obj);
        return !(*v == -1.0 && PyErr_Occurred());
    }

    # 否则尝试将对象转换为浮点数对象并获取其 double 值
    tmp = PyNumber_Float(obj);
    if (tmp) {
        *v = PyFloat_AsDouble(tmp);
        Py_DECREF(tmp);
        return !(*v == -1.0 && PyErr_Occurred());
    }

    # 如果是复数对象，则尝试获取其实部
    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj, "real");
    }
    # 如果是字节对象或者 Unicode 对象，则不进行处理
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    # 如果是序列对象，则获取序列的第一个元素
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    # 如果成功获取到 tmp，则尝试将其转换为 double 类型
    if (tmp) {
        if (double_from_pyobj(v, tmp, errmess)) { Py_DECREF(tmp); return 1; }
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) err = #modulename#_error;
        PyErr_SetString(err, errmess);
    }
    # 转换失败，返回 0
    return 0;
}
"""

# 'float_from_pyobj' 函数的实现，将 Python 对象转换为 float 类型
cfuncs['float_from_pyobj'] = """
static int
float_from_pyobj(float* v, PyObject *obj, const char *errmess)
{
    double d=0.0;
    # 尝试将对象转换为双精度浮点数，并存入 v
    if (double_from_pyobj(&d, obj, errmess)) {
        *v = (float)d;
        return 1;
    }
    # 转换失败，返回 0
    return 0;
}
"""

# 'complex_long_double_from_pyobj' 函数的实现，将 Python 对象转换为 complex_long_double 类型
cfuncs['complex_long_double_from_pyobj'] = """
static int
complex_long_double_from_pyobj(complex_long_double* v, PyObject *obj, const char *errmess)
{
    complex_double cd = {0.0, 0.0};
    # 检查是否是 NumPy 数组标量
    if (PyArray_CheckScalar(obj)){
        # 如果是复杂长双精度数组标量，直接转换为 complex_long_double 类型
        if PyArray_IsScalar(obj, CLongDouble) {
            PyArray_ScalarAsCtype(obj, v);
            return 1;
        }
        # 如果是复杂长双精度数组，则从数据中获取实部和虚部存入 v
        else if (PyArray_Check(obj) && PyArray_TYPE(obj)==NPY_CLONGDOUBLE) {
            (*v).r = npy_creall(*(((npy_clongdouble *)PyArray_DATA(obj))));
            (*v).i = npy_cimagl(*(((npy_clongdouble *)PyArray_DATA(obj))));
            return 1;
        }
    }
    # 如果无法直接转换为复杂长双精度数组，则尝试转换为复杂双精度数组
    if (complex_double_from_pyobj(&cd, obj, errmess)) {
        (*v).r = (long_double)cd.r;
        (*v).i = (long_double)cd.i;
        return 1;
    }
    # 转换失败，返回 0
    return 0;
}
"""
cfuncs['complex_double_from_pyobj'] = """
static int
complex_double_from_pyobj(complex_double* v, PyObject *obj, const char *errmess) {
    Py_complex c;
    // 检查给定对象是否是 Python 复数对象
    if (PyComplex_Check(obj)) {
        // 将 Python 复数对象转换为 C 复数结构体
        c = PyComplex_AsCComplex(obj);
        // 将实部和虚部分别存入复数结构体
        (*v).r = c.real;
        (*v).i = c.imag;
        return 1;
    }
    // 检查给定对象是否是 NumPy 的复数类型标量
    if (PyArray_IsScalar(obj, ComplexFloating)) {
        // 如果是 CFloat 类型的标量
        if (PyArray_IsScalar(obj, CFloat)) {
            npy_cfloat new;
            // 将标量对象转换为 CFloat 类型
            PyArray_ScalarAsCtype(obj, &new);
            // 提取实部和虚部，并转换为 double 存入复数结构体
            (*v).r = (double)npy_crealf(new);
            (*v).i = (double)npy_cimagf(new);
        }
        // 如果是 CLongDouble 类型的标量
        else if (PyArray_IsScalar(obj, CLongDouble)) {
            npy_clongdouble new;
            // 将标量对象转换为 CLongDouble 类型
            PyArray_ScalarAsCtype(obj, &new);
            // 提取实部和虚部，并转换为 double 存入复数结构体
            (*v).r = (double)npy_creall(new);
            (*v).i = (double)npy_cimagl(new);
        }
        else { /* 如果是 CDouble 类型的标量 */
            // 将标量对象直接转换为 CDouble 类型，存入复数结构体
            PyArray_ScalarAsCtype(obj, v);
        }
        return 1;
    }
    // 检查给定对象是否是 NumPy 的标量对象
    if (PyArray_CheckScalar(obj)) { /* 0-dim array or still array scalar */
        PyArrayObject *arr;
        // 如果是 NumPy 数组对象
        if (PyArray_Check(obj)) {
            // 将对象转换为复数类型的 NumPy 数组
            arr = (PyArrayObject *)PyArray_Cast((PyArrayObject *)obj, NPY_CDOUBLE);
        }
        else {
            // 将标量对象转换为复数类型的 NumPy 数组
            arr = (PyArrayObject *)PyArray_FromScalar(obj, PyArray_DescrFromType(NPY_CDOUBLE));
        }
        // 如果转换失败，则返回 0
        if (arr == NULL) {
            return 0;
        }
        // 提取数组中的复数值，存入复数结构体
        (*v).r = npy_creal(*(((npy_cdouble *)PyArray_DATA(arr))));
        (*v).i = npy_cimag(*(((npy_cdouble *)PyArray_DATA(arr))));
        // 减少数组对象的引用计数
        Py_DECREF(arr);
        return 1;
    }
    // Python 不提供 PyNumber_Complex 函数 :-(
    (*v).i = 0.0;
    // 如果给定对象是 Python 浮点数
    if (PyFloat_Check(obj)) {
        // 将浮点数对象转换为 double 存入复数结构体
        (*v).r = PyFloat_AsDouble(obj);
        return !((*v).r == -1.0 && PyErr_Occurred());
    }
    // 如果给定对象是 Python 长整型
    if (PyLong_Check(obj)) {
        // 将长整型对象转换为 double 存入复数结构体
        (*v).r = PyLong_AsDouble(obj);
        return !((*v).r == -1.0 && PyErr_Occurred());
    }
    // 如果给定对象是 Python 序列对象（不包括字节串或 Unicode 字符串）
    if (PySequence_Check(obj) && !(PyBytes_Check(obj) || PyUnicode_Check(obj))) {
        PyObject *tmp = PySequence_GetItem(obj,0);
        if (tmp) {
            // 递归调用 complex_double_from_pyobj 处理序列中的第一个元素
            if (complex_double_from_pyobj(v,tmp,errmess)) {
                Py_DECREF(tmp);
                return 1;
            }
            Py_DECREF(tmp);
        }
    }
    {
        // 获取当前的异常对象
        PyObject* err = PyErr_Occurred();
        if (err==NULL)
            err = PyExc_TypeError;
        // 设置类型错误异常信息
        PyErr_SetString(err,errmess);
    }
    return 0;
}
"""

needs['complex_float_from_pyobj'] = [
    'complex_float', 'complex_double_from_pyobj']
cfuncs['complex_float_from_pyobj'] = """
static int
complex_float_from_pyobj(complex_float* v,PyObject *obj,const char *errmess)
{
    complex_double cd={0.0,0.0};
    // 调用 complex_double_from_pyobj 处理给定对象
    if (complex_double_from_pyobj(&cd,obj,errmess)) {
        // 将返回的复数结构体转换为 float 存入复数结构体
        (*v).r = (float)cd.r;
        (*v).i = (float)cd.i;
        return 1;
    }
    return 0;
}
"""


cfuncs['try_pyarr_from_character'] = """
static int try_pyarr_from_character(PyObject* obj, character* v) {
    // 将对象转换为 PyArrayObject 类型
    PyArrayObject *arr = (PyArrayObject*)obj;
    // 如果对象为空，则返回 -2
    if (!obj) return -2;
    # 检查对象是否为 NumPy 数组
    if (PyArray_Check(obj)) {
        # 如果对象是字符兼容的 NumPy 数组
        if (F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(arr))  {
            # 将字符指针 v 的值复制给数组 arr 的第一个字符位置
            *(character *)(PyArray_DATA(arr)) = *v;
            # 返回成功标志 1
            return 1;
        }
    }
    {
        # 声明一个字符数组作为错误消息的缓冲区
        char mess[F2PY_MESSAGE_BUFFER_SIZE];
        # 检查是否发生了 Python 异常
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            # 如果未发生异常，设置错误类型为 ValueError
            err = PyExc_ValueError;
            # 构造错误消息，说明期望的数据类型
            strcpy(mess, "try_pyarr_from_character failed"
                         " -- expected bytes array-scalar|array, got ");
            # 将对象 obj 的描述信息添加到错误消息中
            f2py_describe(obj, mess + strlen(mess));
            # 设置 Python 异常对象及其错误消息
            PyErr_SetString(err, mess);
        }
    }
    # 返回失败标志 0
    return 0;
}
"""

# 将 'pyobj_from_char1' 和 'TRYPYARRAYTEMPLATE' 添加到 needs 字典的 'try_pyarr_from_char' 键中
needs['try_pyarr_from_char'] = ['pyobj_from_char1', 'TRYPYARRAYTEMPLATE']

# 将字符串 'static int try_pyarr_from_char(PyObject* obj,char* v) {\n    TRYPYARRAYTEMPLATE(char,\'c\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_char' 键
cfuncs['try_pyarr_from_char'] = 'static int try_pyarr_from_char(PyObject* obj,char* v) {\n    TRYPYARRAYTEMPLATE(char,\'c\');\n}\n'

# 将 'TRYPYARRAYTEMPLATE' 和 'unsigned_char' 添加到 needs 字典的 'try_pyarr_from_unsigned_char' 键中
needs['try_pyarr_from_unsigned_char'] = ['TRYPYARRAYTEMPLATE', 'unsigned_char']

# 将字符串 'static int try_pyarr_from_unsigned_char(PyObject* obj,unsigned_char* v) {\n    TRYPYARRAYTEMPLATE(unsigned_char,\'b\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_unsigned_char' 键
cfuncs['try_pyarr_from_unsigned_char'] = 'static int try_pyarr_from_unsigned_char(PyObject* obj,unsigned_char* v) {\n    TRYPYARRAYTEMPLATE(unsigned_char,\'b\');\n}\n'

# 将 'TRYPYARRAYTEMPLATE' 和 'signed_char' 添加到 needs 字典的 'try_pyarr_from_signed_char' 键中
needs['try_pyarr_from_signed_char'] = ['TRYPYARRAYTEMPLATE', 'signed_char']

# 将字符串 'static int try_pyarr_from_signed_char(PyObject* obj,signed_char* v) {\n    TRYPYARRAYTEMPLATE(signed_char,\'1\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_signed_char' 键
cfuncs['try_pyarr_from_signed_char'] = 'static int try_pyarr_from_signed_char(PyObject* obj,signed_char* v) {\n    TRYPYARRAYTEMPLATE(signed_char,\'1\');\n}\n'

# 将 'pyobj_from_short1' 和 'TRYPYARRAYTEMPLATE' 添加到 needs 字典的 'try_pyarr_from_short' 键中
needs['try_pyarr_from_short'] = ['pyobj_from_short1', 'TRYPYARRAYTEMPLATE']

# 将字符串 'static int try_pyarr_from_short(PyObject* obj,short* v) {\n    TRYPYARRAYTEMPLATE(short,\'s\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_short' 键
cfuncs['try_pyarr_from_short'] = 'static int try_pyarr_from_short(PyObject* obj,short* v) {\n    TRYPYARRAYTEMPLATE(short,\'s\');\n}\n'

# 将 'pyobj_from_int1' 和 'TRYPYARRAYTEMPLATE' 添加到 needs 字典的 'try_pyarr_from_int' 键中
needs['try_pyarr_from_int'] = ['pyobj_from_int1', 'TRYPYARRAYTEMPLATE']

# 将字符串 'static int try_pyarr_from_int(PyObject* obj,int* v) {\n    TRYPYARRAYTEMPLATE(int,\'i\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_int' 键
cfuncs['try_pyarr_from_int'] = 'static int try_pyarr_from_int(PyObject* obj,int* v) {\n    TRYPYARRAYTEMPLATE(int,\'i\');\n}\n'

# 将 'pyobj_from_long1' 和 'TRYPYARRAYTEMPLATE' 添加到 needs 字典的 'try_pyarr_from_long' 键中
needs['try_pyarr_from_long'] = ['pyobj_from_long1', 'TRYPYARRAYTEMPLATE']

# 将字符串 'static int try_pyarr_from_long(PyObject* obj,long* v) {\n    TRYPYARRAYTEMPLATE(long,\'l\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_long' 键
cfuncs['try_pyarr_from_long'] = 'static int try_pyarr_from_long(PyObject* obj,long* v) {\n    TRYPYARRAYTEMPLATE(long,\'l\');\n}\n'

# 将 'pyobj_from_long_long1', 'TRYPYARRAYTEMPLATE' 和 'long_long' 添加到 needs 字典的 'try_pyarr_from_long_long' 键中
needs['try_pyarr_from_long_long'] = ['pyobj_from_long_long1', 'TRYPYARRAYTEMPLATE', 'long_long']

# 将字符串 'static int try_pyarr_from_long_long(PyObject* obj,long_long* v) {\n    TRYPYARRAYTEMPLATE(long_long,\'L\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_long_long' 键
cfuncs['try_pyarr_from_long_long'] = 'static int try_pyarr_from_long_long(PyObject* obj,long_long* v) {\n    TRYPYARRAYTEMPLATE(long_long,\'L\');\n}\n'

# 将 'pyobj_from_float1' 和 'TRYPYARRAYTEMPLATE' 添加到 needs 字典的 'try_pyarr_from_float' 键中
needs['try_pyarr_from_float'] = ['pyobj_from_float1', 'TRYPYARRAYTEMPLATE']

# 将字符串 'static int try_pyarr_from_float(PyObject* obj,float* v) {\n    TRYPYARRAYTEMPLATE(float,\'f\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_float' 键
cfuncs['try_pyarr_from_float'] = 'static int try_pyarr_from_float(PyObject* obj,float* v) {\n    TRYPYARRAYTEMPLATE(float,\'f\');\n}\n'

# 将 'pyobj_from_double1' 和 'TRYPYARRAYTEMPLATE' 添加到 needs 字典的 'try_pyarr_from_double' 键中
needs['try_pyarr_from_double'] = ['pyobj_from_double1', 'TRYPYARRAYTEMPLATE']

# 将字符串 'static int try_pyarr_from_double(PyObject* obj,double* v) {\n    TRYPYARRAYTEMPLATE(double,\'d\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_double' 键
cfuncs['try_pyarr_from_double'] = 'static int try_pyarr_from_double(PyObject* obj,double* v) {\n    TRYPYARRAYTEMPLATE(double,\'d\');\n}\n'

# 将 'pyobj_from_complex_float1', 'TRYCOMPLEXPYARRAYTEMPLATE' 和 'complex_float' 添加到 needs 字典的 'try_pyarr_from_complex_float' 键中
needs['try_pyarr_from_complex_float'] = ['pyobj_from_complex_float1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_float']

# 将字符串 'static int try_pyarr_from_complex_float(PyObject* obj,complex_float* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(float,\'F\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_complex_float' 键
cfuncs['try_pyarr_from_complex_float'] = 'static int try_pyarr_from_complex_float(PyObject* obj,complex_float* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(float,\'F\');\n}\n'

# 将 'pyobj_from_complex_double1', 'TRYCOMPLEXPYARRAYTEMPLATE' 和 'complex_double' 添加到 needs 字典的 'try_pyarr_from_complex_double' 键中
needs['try_pyarr_from_complex_double'] = ['pyobj_from_complex_double1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_double']

# 将字符串 'static int try_pyarr_from_complex_double(PyObject* obj,complex_double* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(double,\'D\');\n}\n' 关联到 cfuncs 字典的 'try_pyarr_from_complex_double' 键
cfuncs['try_pyarr_from_complex_double'] = 'static int try_pyarr_from_complex_double(PyObject* obj,complex_double* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(double,\'D\');\n}\n'

# 将 'CFUNCSMESS', 'PRINTPYOBJERR' 和 'MINMAX' 添加到 needs 字典的 'create_cb_arglist' 键中
needs['create_cb_arglist'] = ['CFUNCSMESS', 'PRINTPYOBJERR', 'MINMAX']

# 将字符串 'static int\n' 关联到 cfuncs 字典的 'create_cb_arglist' 键
cfuncs['create_cb_arglist'] = """
static int
    // 打印调试信息到标准错误输出
    CFUNCSMESS("create_cb_arglist\n");
    // 初始化变量 tot, opt, ext, siz, i, di
    Py_ssize_t tot, opt, ext, siz, i, di = 0;
    // 初始化临时对象指针 tmp 和 tmp_fun
    PyObject *tmp = NULL;
    PyObject *tmp_fun = NULL;
    // 初始化参数 maxnofargs, nofoptargs, nofargs, args, errmess
    // 初始化 tot, opt, ext, siz 为 0
    tot = opt = ext = siz = 0;

    // 获取函数的总参数数目
    if (PyFunction_Check(fun)) {
        // 如果是 Python 函数对象，直接将其赋给 tmp_fun
        tmp_fun = fun;
        Py_INCREF(tmp_fun);
    }
    else {
        di = 1;
        // 如果不是 Python 函数对象，则根据不同类型的对象来处理
        if (PyObject_HasAttrString(fun, "im_func")) {
            // 如果是实例方法对象，获取其对应的函数对象赋给 tmp_fun
            tmp_fun = PyObject_GetAttrString(fun, "im_func");
        }
        else if (PyObject_HasAttrString(fun, "__call__")) {
            // 如果是可调用对象，获取其 __call__ 方法的函数对象赋给 tmp_fun
            tmp = PyObject_GetAttrString(fun, "__call__");
            if (PyObject_HasAttrString(tmp, "im_func"))
                tmp_fun = PyObject_GetAttrString(tmp, "im_func");
            else {
                // 如果是内置函数，直接使用原始对象 fun
                tmp_fun = fun;
                Py_INCREF(tmp_fun);
                // 设置 tot 为 maxnofargs，如果是 PyCFunction_Check，设置 di 为 0
                tot = maxnofargs;
                if (PyCFunction_Check(fun)) {
                    // 如果函数有 co_argcount（如 PyPy 中的情况）
                    di = 0;
                }
                // 如果传入了参数 xa，则将其长度加到 tot 上
                if (xa != NULL)
                    tot += PyTuple_Size((PyObject *)xa);
            }
            Py_XDECREF(tmp);
        }
        else if (PyFortran_Check(fun) || PyFortran_Check1(fun)) {
            // 如果是 Fortran 函数对象，直接使用原始对象 fun
            tot = maxnofargs;
            if (xa != NULL)
                tot += PyTuple_Size((PyObject *)xa);
            tmp_fun = fun;
            Py_INCREF(tmp_fun);
        }
        else if (F2PyCapsule_Check(fun)) {
            // 如果是 F2PyCapsule 对象，直接使用原始对象 fun
            tot = maxnofargs;
            if (xa != NULL)
                ext = PyTuple_Size((PyObject *)xa);
            if (ext > 0) {
                // 如果有额外的参数，输出错误信息到标准错误输出并跳转到错误处理标签
                fprintf(stderr, "extra arguments tuple cannot be used with PyCapsule call-back\n");
                goto capi_fail;
            }
            tmp_fun = fun;
            Py_INCREF(tmp_fun);
        }
    }

    // 如果没有有效的 tmp_fun 对象，输出错误信息到标准错误输出并跳转到错误处理标签
    if (tmp_fun == NULL) {
        fprintf(stderr,
                "Call-back argument must be function|instance|instance.__call__|f2py-function "
                "but got %s.\n",
                ((fun == NULL) ? "NULL" : Py_TYPE(fun)->tp_name));
        goto capi_fail;
    }

    // 如果 tmp_fun 有 __code__ 属性
    if (PyObject_HasAttrString(tmp_fun, "__code__")) {
        // 如果 tmp_fun.__code__ 有 co_argcount 属性
        if (PyObject_HasAttrString(tmp = PyObject_GetAttrString(tmp_fun, "__code__"), "co_argcount")) {
            // 获取 tmp.__code__.co_argcount 属性的值，并赋给 tmp_argcount
            PyObject *tmp_argcount = PyObject_GetAttrString(tmp, "co_argcount");
            Py_DECREF(tmp);
            if (tmp_argcount == NULL) {
                // 如果获取失败，跳转到错误处理标签
                goto capi_fail;
            }
            // 将 tmp_argcount 转换为 Py_ssize_t 类型，减去 di，并赋给 tot
            tot = PyLong_AsSsize_t(tmp_argcount) - di;
            Py_DECREF(tmp_argcount);
        }
    }
    // 获取可选参数的数目
    /* 检查临时函数对象是否具有 "__defaults__" 属性 */
    if (PyObject_HasAttrString(tmp_fun, "__defaults__")) {
        /* 如果临时函数对象的 "__defaults__" 是一个元组，则获取其长度作为可选参数的数量 */
        if (PyTuple_Check(tmp = PyObject_GetAttrString(tmp_fun, "__defaults__")))
            opt = PyTuple_Size(tmp);
        /* 释放临时对象 */
        Py_XDECREF(tmp);
    }
    /* 获取额外参数的数量 */
    if (xa != NULL)
        ext = PyTuple_Size((PyObject *)xa);
    /* 计算回调参数列表的大小 */
    siz = MIN(maxnofargs + ext, tot);
    /* 设置返回的参数数量，排除额外参数后的剩余数量 */
    *nofargs = MAX(0, siz - ext);
#ifdef DEBUGCFUNCS
    # 如果定义了 DEBUGCFUNCS 宏，则输出调试信息到标准错误流
    fprintf(stderr,
            "debug-capi:create_cb_arglist:maxnofargs(-nofoptargs),"
            "tot,opt,ext,siz,nofargs = %d(-%d), %zd, %zd, %zd, %zd, %d\\n",
            maxnofargs, nofoptargs, tot, opt, ext, siz, *nofargs);
#endif

    # 检查参数 siz 是否小于 tot-opt，若是则输出错误信息到标准错误流并跳转到 capi_fail 标签处
    if (siz < tot-opt) {
        fprintf(stderr,
                "create_cb_arglist: Failed to build argument list "
                "(siz) with enough arguments (tot-opt) required by "
                "user-supplied function (siz,tot,opt=%zd, %zd, %zd).\\n",
                siz, tot, opt);
        goto capi_fail;
    }

    /* Initialize argument list */
    # 初始化参数列表 args，创建一个大小为 siz 的 PyTupleObject 对象
    *args = (PyTupleObject *)PyTuple_New(siz);
    for (i=0;i<*nofargs;i++) {
        # 为每个参数设置为 Py_None，并增加其引用计数
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM((PyObject *)(*args),i,Py_None);
    }
    # 如果 xa 不为 NULL，则将其内容复制到 args 的后续位置
    if (xa != NULL)
        for (i=(*nofargs);i<siz;i++) {
            tmp = PyTuple_GetItem((PyObject *)xa,i-(*nofargs));
            Py_INCREF(tmp);
            PyTuple_SET_ITEM(*args,i,tmp);
        }
    # 输出调试信息到 CFUNCSMESS 宏所指定的位置
    CFUNCSMESS("create_cb_arglist-end\\n");
    # 减少 tmp_fun 的引用计数
    Py_DECREF(tmp_fun);
    # 返回成功状态码 1
    return 1;

capi_fail:
    # 如果没有发生 Python 异常，则设置一个异常字符串，并释放 tmp_fun 的引用
    if (PyErr_Occurred() == NULL)
        PyErr_SetString(#modulename#_error, errmess);
    Py_XDECREF(tmp_fun);
    # 返回失败状态码 0
    return 0;
}
    elif isinstance(need, str):
        # 如果 need 是字符串类型
        if not need:
            # 如果 need 是空字符串，则直接返回
            return
        # 根据 need 的值分别进行判断
        if need in includes0:
            n = 'includes0'
        elif need in includes:
            n = 'includes'
        elif need in typedefs:
            n = 'typedefs'
        elif need in typedefs_generated:
            n = 'typedefs_generated'
        elif need in cppmacros:
            n = 'cppmacros'
        elif need in cfuncs:
            n = 'cfuncs'
        elif need in callbacks:
            n = 'callbacks'
        elif need in f90modhooks:
            n = 'f90modhooks'
        elif need in commonhooks:
            n = 'commonhooks'
        else:
            # 如果 need 不属于已知的分类，则输出错误信息并返回
            errmess('append_needs: unknown need %s\n' % (repr(need)))
            return
        # 检查 need 是否已经在 outneeds[n] 中，如果是，则直接返回
        if need in outneeds[n]:
            return
        # 如果 flag 为真，则进行以下操作
        if flag:
            tmp = {}
            # 如果 need 在 needs 中存在
            if need in needs:
                # 遍历 needs[need] 中的每个 nn，递归调用 append_needs 获取 t
                for nn in needs[need]:
                    t = append_needs(nn, 0)
                    # 如果 t 是字典类型，则将其合并到 tmp 中
                    if isinstance(t, dict):
                        for nnn in t.keys():
                            if nnn in tmp:
                                tmp[nnn] = tmp[nnn] + t[nnn]
                            else:
                                tmp[nnn] = t[nnn]
            # 将 tmp 中的每个 nn 和 nnn 添加到对应的 outneeds[nn] 中
            for nn in tmp.keys():
                for nnn in tmp[nn]:
                    if nnn not in outneeds[nn]:
                        outneeds[nn] = [nnn] + outneeds[nn]
            # 将 need 添加到 outneeds[n] 中
            outneeds[n].append(need)
        else:
            # 如果 flag 不为真，则进行以下操作
            tmp = {}
            # 如果 need 在 needs 中存在
            if need in needs:
                # 遍历 needs[need] 中的每个 nn，递归调用 append_needs 获取 t
                for nn in needs[need]:
                    t = append_needs(nn, flag)
                    # 如果 t 是字典类型，则将其合并到 tmp 中
                    if isinstance(t, dict):
                        for nnn in t.keys():
                            if nnn in tmp:
                                tmp[nnn] = t[nnn] + tmp[nnn]
                            else:
                                tmp[nnn] = t[nnn]
            # 如果 n 不在 tmp 中，则初始化为空列表
            if n not in tmp:
                tmp[n] = []
            # 将 need 添加到 tmp[n] 中
            tmp[n].append(need)
            return tmp
    else:
        # 如果 need 不是字符串类型，则输出错误信息并返回
        errmess('append_needs: expected list or string but got :%s\n' %
                (repr(need)))
# 获取需求的函数，修改全局变量 `outneeds` 字典的内容。
def get_needs():
    # 初始化空字典 res，用于存储最终结果
    res = {}
    # 遍历 outneeds 字典的键列表
    for n in outneeds.keys():
        # 初始化空列表 out，并复制一份 outneeds[n] 的内容到 saveout
        out = []
        saveout = copy.copy(outneeds[n])
        
        # 当 outneeds[n] 非空时执行循环
        while len(outneeds[n]) > 0:
            # 如果 outneeds[n] 的第一个元素不在 needs 字典中
            if outneeds[n][0] not in needs:
                # 将 outneeds[n] 的第一个元素添加到 out 列表末尾，并从 outneeds[n] 中删除
                out.append(outneeds[n][0])
                del outneeds[n][0]
            else:
                # 否则，设置标志 flag 为 0
                flag = 0
                # 遍历 outneeds[n] 的除第一个元素外的剩余元素
                for k in outneeds[n][1:]:
                    # 如果 k 在 needs[outneeds[n][0]] 中，则将 flag 置为 1 并跳出循环
                    if k in needs[outneeds[n][0]]:
                        flag = 1
                        break
                # 如果 flag 为真
                if flag:
                    # 将 outneeds[n] 的第一个元素移到末尾，并继续处理
                    outneeds[n] = outneeds[n][1:] + [outneeds[n][0]]
                else:
                    # 否则，将 outneeds[n] 的第一个元素添加到 out 列表末尾，并从 outneeds[n] 中删除
                    out.append(outneeds[n][0])
                    del outneeds[n][0]
            
            # 检查是否存在循环依赖，若无进展，则输出错误信息并合并 out 和 saveout 列表
            if saveout and (0 not in map(lambda x, y: x == y, saveout, outneeds[n])) \
                    and outneeds[n] != []:
                print(n, saveout)
                errmess(
                    'get_needs: no progress in sorting needs, probably circular dependence, skipping.\n')
                out = out + saveout
                break
            
            # 保存当前 outneeds[n] 的副本到 saveout
            saveout = copy.copy(outneeds[n])
        
        # 如果 out 列表为空，则将 n 添加为其自身的依赖
        if out == []:
            out = [n]
        
        # 将结果记录到 res 字典中，键为 n，值为 out 列表
        res[n] = out
    
    # 返回最终结果字典 res
    return res
```