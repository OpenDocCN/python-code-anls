# `.\pytorch\torch\csrc\utils.h`

```
// 如果 THP_UTILS_H 未定义，则定义 THP_UTILS_H，用于防止多次包含
#ifndef THP_UTILS_H
#define THP_UTILS_H

// 包含 ATen 库中的 ATen.h 头文件，提供张量操作的支持
#include <ATen/ATen.h>

// 包含 C10 库中的 Exception.h 头文件，提供异常处理支持
#include <c10/util/Exception.h>

// 包含 torch 库中的 Storage.h 头文件，提供张量存储支持
#include <torch/csrc/Storage.h>

// 包含 torch 库中的 THConcat.h 头文件，提供张量拼接支持
#include <torch/csrc/THConcat.h>

// 包含 torch 库中的 object_ptr.h 头文件，提供智能指针支持
#include <torch/csrc/utils/object_ptr.h>

// 包含 torch 库中的 python_compat.h 头文件，提供 Python 兼容性支持
#include <torch/csrc/utils/python_compat.h>

// 包含 torch 库中的 python_numbers.h 头文件，提供 Python 数字支持
#include <torch/csrc/utils/python_numbers.h>

// 包含标准库中的 string 头文件，提供字符串操作支持
#include <string>

// 包含标准库中的 type_traits 头文件，提供类型特性支持
#include <type_traits>

// 包含标准库中的 vector 头文件，提供动态数组支持
#include <vector>

// 如果定义了 USE_CUDA，则包含 CUDAStream.h 头文件，提供 CUDA 流支持
#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

// 定义 THPUtils_(NAME) 宏，用于生成实际的函数名
#define THPUtils_(NAME) TH_CONCAT_4(THP, Real, Utils_, NAME)

// 定义 THPUtils_typename(obj) 宏，返回对象类型名
#define THPUtils_typename(obj) (Py_TYPE(obj)->tp_name)

// 根据编译器类型定义 THP_EXPECT 宏，用于优化条件判断
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define THP_EXPECT(x, y) (__builtin_expect((x), (y)))
#else
#define THP_EXPECT(x, y) (x)
#endif

// 定义 THPUtils_checkReal_FLOAT 宏，检查对象是否为 Python 中的浮点数类型
#define THPUtils_checkReal_FLOAT(object) (PyFloat_Check(object) || PyLong_Check(object))

// 定义 THPUtils_unpackReal_FLOAT 宏，从 Python 对象中解包浮点数数据
#define THPUtils_unpackReal_FLOAT(object)           \
  (PyFloat_Check(object) ? PyFloat_AsDouble(object) \
       : PyLong_Check(object)                       \
       ? PyLong_AsLongLong(object)                  \
       : (throw std::runtime_error("Could not parse real"), 0))

// 定义 THPUtils_checkReal_INT 宏，检查对象是否为 Python 中的整数类型
#define THPUtils_checkReal_INT(object) PyLong_Check(object)

// 定义 THPUtils_unpackReal_INT 宏，从 Python 对象中解包整数数据
#define THPUtils_unpackReal_INT(object) \
  (PyLong_Check(object)                 \
       ? PyLong_AsLongLong(object)      \
       : (throw std::runtime_error("Could not parse real"), 0))

// 定义 THPUtils_unpackReal_BOOL 宏，从 Python 对象中解包布尔值数据
#define THPUtils_unpackReal_BOOL(object) \
  (PyBool_Check(object)                  \
       ? object                          \
       : (throw std::runtime_error("Could not parse real"), Py_False))

// 定义 THPUtils_unpackReal_COMPLEX 宏，从 Python 对象中解包复数数据
#define THPUtils_unpackReal_COMPLEX(object)                                   \
  (PyComplex_Check(object)                                                    \
       ? (c10::complex<double>(                                               \
             PyComplex_RealAsDouble(object), PyComplex_ImagAsDouble(object))) \
       : PyFloat_Check(object)                                                \
       ? (c10::complex<double>(PyFloat_AsDouble(object), 0))                  \
       : PyLong_Check(object)                                                 \
       ? (c10::complex<double>(PyLong_AsLongLong(object), 0))                 \
       : (throw std::runtime_error("Could not parse real"),                   \
          c10::complex<double>(0, 0)))

// 定义 THPUtils_checkReal_BOOL 宏，检查对象是否为 Python 中的布尔类型
#define THPUtils_checkReal_BOOL(object) PyBool_Check(object)

// 定义 THPUtils_checkReal_COMPLEX 宏，检查对象是否为 Python 中的复数类型
#define THPUtils_checkReal_COMPLEX(object)                                    \
  PyComplex_Check(object) || PyFloat_Check(object) || PyLong_Check(object) || \
      PyInt_Check(object)

// 定义 THPUtils_newReal_FLOAT 宏，创建一个新的 Python 浮点数对象
#define THPUtils_newReal_FLOAT(value) PyFloat_FromDouble(value)

// 定义 THPUtils_newReal_INT 宏，创建一个新的 Python 整数对象
#define THPUtils_newReal_INT(value) PyInt_FromLong(value)

// 定义 THPUtils_newReal_BOOL 宏，创建一个新的 Python 布尔对象
#define THPUtils_newReal_BOOL(value) PyBool_FromLong(value)

// 定义 THPUtils_newReal_COMPLEX 宏，创建一个新的 Python 复数对象
#define THPUtils_newReal_COMPLEX(value) \
  PyComplex_FromDoubles(value.real(), value.imag())

// 定义 THPDoubleUtils_checkReal 宏，检查对象是否为 Python 中的浮点数类型
#define THPDoubleUtils_checkReal(object) THPUtils_checkReal_FLOAT(object)

// 定义 THPDoubleUtils_unpackReal 宏，从 Python 对象中解包浮点数数据并转换为 double 类型
#define THPDoubleUtils_unpackReal(object) \
  (double)THPUtils_unpackReal_FLOAT(object)

// 结束 THP_UTILS_H 的定义
#endif
#define THPDoubleUtils_newReal(value) THPUtils_newReal_FLOAT(value)
// 定义宏 THPDoubleUtils_newReal(value)，用于将输入值转换为浮点数并创建新的对象

#define THPFloatUtils_checkReal(object) THPUtils_checkReal_FLOAT(object)
// 定义宏 THPFloatUtils_checkReal(object)，用于检查给定对象是否为浮点数类型

#define THPFloatUtils_unpackReal(object) \
  (float)THPUtils_unpackReal_FLOAT(object)
// 定义宏 THPFloatUtils_unpackReal(object)，用于解包浮点数对象并将其转换为单精度浮点数

#define THPFloatUtils_newReal(value) THPUtils_newReal_FLOAT(value)
// 定义宏 THPFloatUtils_newReal(value)，用于将输入值转换为浮点数并创建新的对象

#define THPHalfUtils_checkReal(object) THPUtils_checkReal_FLOAT(object)
// 定义宏 THPHalfUtils_checkReal(object)，用于检查给定对象是否为浮点数类型

#define THPHalfUtils_unpackReal(object) \
  (at::Half) THPUtils_unpackReal_FLOAT(object)
// 定义宏 THPHalfUtils_unpackReal(object)，用于解包浮点数对象并将其转换为半精度浮点数

#define THPHalfUtils_newReal(value) PyFloat_FromDouble(value)
// 定义宏 THPHalfUtils_newReal(value)，使用给定的双精度浮点数值创建新的 Python 浮点数对象

#define THPHalfUtils_newAccreal(value) THPUtils_newReal_FLOAT(value)
// 定义宏 THPHalfUtils_newAccreal(value)，用于将输入值转换为浮点数并创建新的对象

#define THPComplexDoubleUtils_checkReal(object) \
  THPUtils_checkReal_COMPLEX(object)
// 定义宏 THPComplexDoubleUtils_checkReal(object)，用于检查给定对象是否为复数类型

#define THPComplexDoubleUtils_unpackReal(object) \
  THPUtils_unpackReal_COMPLEX(object)
// 定义宏 THPComplexDoubleUtils_unpackReal(object)，用于解包复数对象

#define THPComplexDoubleUtils_newReal(value) THPUtils_newReal_COMPLEX(value)
// 定义宏 THPComplexDoubleUtils_newReal(value)，用于将输入值转换为复数并创建新的对象

#define THPComplexFloatUtils_checkReal(object) \
  THPUtils_checkReal_COMPLEX(object)
// 定义宏 THPComplexFloatUtils_checkReal(object)，用于检查给定对象是否为复数类型

#define THPComplexFloatUtils_unpackReal(object) \
  (c10::complex<float>)THPUtils_unpackReal_COMPLEX(object)
// 定义宏 THPComplexFloatUtils_unpackReal(object)，用于解包复数对象并将其转换为单精度浮点数复数

#define THPComplexFloatUtils_newReal(value) THPUtils_newReal_COMPLEX(value)
// 定义宏 THPComplexFloatUtils_newReal(value)，用于将输入值转换为复数并创建新的对象

#define THPBFloat16Utils_checkReal(object) THPUtils_checkReal_FLOAT(object)
// 定义宏 THPBFloat16Utils_checkReal(object)，用于检查给定对象是否为浮点数类型

#define THPBFloat16Utils_unpackReal(object) \
  (at::BFloat16) THPUtils_unpackReal_FLOAT(object)
// 定义宏 THPBFloat16Utils_unpackReal(object)，用于解包浮点数对象并将其转换为 BF16 类型

#define THPBFloat16Utils_newReal(value) PyFloat_FromDouble(value)
// 定义宏 THPBFloat16Utils_newReal(value)，使用给定的双精度浮点数值创建新的 Python 浮点数对象

#define THPBFloat16Utils_newAccreal(value) THPUtils_newReal_FLOAT(value)
// 定义宏 THPBFloat16Utils_newAccreal(value)，用于将输入值转换为浮点数并创建新的对象

#define THPBoolUtils_checkReal(object) THPUtils_checkReal_BOOL(object)
// 定义宏 THPBoolUtils_checkReal(object)，用于检查给定对象是否为布尔类型

#define THPBoolUtils_unpackReal(object) THPUtils_unpackReal_BOOL(object)
// 定义宏 THPBoolUtils_unpackReal(object)，用于解包布尔对象

#define THPBoolUtils_newReal(value) THPUtils_newReal_BOOL(value)
// 定义宏 THPBoolUtils_newReal(value)，用于将输入值转换为布尔类型并创建新的对象

#define THPBoolUtils_checkAccreal(object) THPUtils_checkReal_BOOL(object)
// 定义宏 THPBoolUtils_checkAccreal(object)，用于检查给定对象是否为布尔类型

#define THPBoolUtils_unpackAccreal(object) \
  (int64_t) THPUtils_unpackReal_BOOL(object)
// 定义宏 THPBoolUtils_unpackAccreal(object)，用于解包布尔对象并转换为整数类型

#define THPBoolUtils_newAccreal(value) THPUtils_newReal_BOOL(value)
// 定义宏 THPBoolUtils_newAccreal(value)，用于将输入值转换为布尔类型并创建新的对象

#define THPLongUtils_checkReal(object) THPUtils_checkReal_INT(object)
// 定义宏 THPLongUtils_checkReal(object)，用于检查给定对象是否为整数类型

#define THPLongUtils_unpackReal(object) \
  (int64_t) THPUtils_unpackReal_INT(object)
// 定义宏 THPLongUtils_unpackReal(object)，用于解包整数对象并转换为 64 位整数

#define THPLongUtils_newReal(value) THPUtils_newReal_INT(value)
// 定义宏 THPLongUtils_newReal(value)，用于将输入值转换为整数并创建新的对象

#define THPIntUtils_checkReal(object) THPUtils_checkReal_INT(object)
// 定义宏 THPIntUtils_checkReal(object)，用于检查给定对象是否为整数类型

#define THPIntUtils_unpackReal(object) (int)THPUtils_unpackReal_INT(object)
// 定义宏 THPIntUtils_unpackReal(object)，用于解包整数对象并转换为整数

#define THPIntUtils_newReal(value) THPUtils_newReal_INT(value)
// 定义宏 THPIntUtils_newReal(value)，用于将输入值转换为整数并创建新的对象

#define THPShortUtils_checkReal(object) THPUtils_checkReal_INT(object)
// 定义宏 THPShortUtils_checkReal(object)，用于检查给定对象是否为整数类型

#define THPShortUtils_unpackReal(object) (short)THPUtils_unpackReal_INT(object)
// 定义宏 THPShortUtils_unpackReal(object)，用于解包整数对象并转换为短整数

#define THPShortUtils_newReal(value) THPUtils_newReal_INT(value)
// 定义宏 THPShortUtils_newReal(value)，用于将输入值转换为整数并创建新的对象

#define THPCharUtils_checkReal(object) THPUtils_checkReal_INT(object)
// 定义宏 THPCharUtils_checkReal(object)，用于检查给定对象是否为整数类型

#define THPCharUtils_unpackReal(object) (char)THPUtils_unpackReal_INT(object)
// 定义宏 THPCharUtils_unpackReal(object)，用于解包整数对象并转换为字符

#define THPCharUtils_newReal(value) THPUtils_newReal_INT(value)
// 定义宏 THPCharUtils_newReal(value)，用于将输入值转换为整数并创建新的对象

#define THPByteUtils_checkReal(object) THPUtils_checkReal_INT(object)
// 定义宏 THPByteUtils_checkReal(object)，用于检查给定对象是否为整数类型

#define THPByteUtils_unpackReal(object) \
  (unsigned char)THPUtils_unpackReal_INT(object)
// 定义宏 THPByteUtils_unpackReal(object)，用于解包整数对象并转换为无符号字符

#define THPByteUtils_newReal(value) THPUtils_newReal_INT(value)
// 定义宏 THPByteUtils_newReal(value)，用于将输入值转换为整数并创建新的对象

// quantized types
// 量化类型的宏定义
/*
   定义宏，用于检查和操作不同类型的整数和浮点数对象
   THPQUInt8Utils_checkReal: 检查对象是否为无符号8位整数
   THPQUInt8Utils_unpackReal: 解包对象为整数，假定为无符号8位整数
   THPQUInt8Utils_newReal: 创建新的整数对象，假定为无符号8位整数
   THPQInt8Utils_checkReal: 检查对象是否为有符号8位整数
   THPQInt8Utils_unpackReal: 解包对象为整数，假定为有符号8位整数
   THPQInt8Utils_newReal: 创建新的整数对象，假定为有符号8位整数
   THPQInt32Utils_checkReal: 检查对象是否为有符号32位整数
   THPQInt32Utils_unpackReal: 解包对象为整数，假定为有符号32位整数
   THPQInt32Utils_newReal: 创建新的整数对象，假定为有符号32位整数
   THPQUInt4x2Utils_checkReal: 检查对象是否为无符号4x2位整数
   THPQUInt4x2Utils_unpackReal: 解包对象为整数，假定为无符号4x2位整数
   THPQUInt4x2Utils_newReal: 创建新的整数对象，假定为无符号4x2位整数
   THPQUInt2x4Utils_checkReal: 检查对象是否为无符号2x4位整数
   THPQUInt2x4Utils_unpackReal: 解包对象为整数，假定为无符号2x4位整数
   THPQUInt2x4Utils_newReal: 创建新的整数对象，假定为无符号2x4位整数
*/

/*
   从 https://github.com/python/cpython/blob/v3.7.0/Modules/xxsubtype.c
   如果编译为共享库，一些编译器不允许在静态 PyTypeObject 初始化器中使用其他库中定义的 Python 对象的地址。
   DEFERRED_ADDRESS 宏用于标记这些地址出现的插槽；模块初始化函数在运行时必须填写标记的插槽。
   参数用于文档 -- 宏会忽略它。
*/

/*
   定义了几个函数原型，以及一些用于处理 Python 对象的实用函数和模板结构。
*/

/*
   模板结构 mod_traits 用于根据类型 _real 的属性（整数或浮点数）选择合适的取模操作。
   对于浮点数类型 _real，使用 fmod 函数取模；
   对于整数类型 _real，使用 % 操作符取模。
*/

/*
   函数声明：
   setBackCompatBroadcastWarn: 设置广播警告的后向兼容性
   getBackCompatBroadcastWarn: 获取广播警告的后向兼容性设置
   setBackCompatKeepdimWarn: 设置 keepdim 警告的后向兼容性
*/
// 返回一个布尔值，表示是否获取到了向后兼容的保留维度警告
bool getBackCompatKeepdimWarn();

// 可能会抛出向后兼容的保留维度警告，函数参数为字符指针
bool maybeThrowBackCompatKeepdimWarn(char* func);

// 如果定义了 USE_CUDA 宏，则声明一个函数，将 Python 对象转换为 CUDA 流列表
// 函数参数为 PyObject 指针，返回一个包含 CUDA 流的 optional 向量
#ifdef USE_CUDA
std::vector<std::optional<at::cuda::CUDAStream>>
THPUtils_PySequence_to_CUDAStreamList(PyObject* obj);
#endif

// 填充存储器 self，将其所有元素设为指定的 uint8_t 值
void storage_fill(const at::Storage& self, uint8_t value);

// 设置存储器 self 中索引为 idx 的元素值为指定的 uint8_t 值
void storage_set(const at::Storage& self, ptrdiff_t idx, uint8_t value);

// 获取存储器 self 中索引为 idx 的元素值，返回 uint8_t 类型的值
uint8_t storage_get(const at::Storage& self, ptrdiff_t idx);
```