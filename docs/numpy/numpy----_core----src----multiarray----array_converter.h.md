# `.\numpy\numpy\_core\src\multiarray\array_converter.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAY_CONVERTER_H_
// 如果未定义 NUMPY_CORE_SRC_MULTIARRAY_ARRAY_CONVERTER_H_ 宏，则开始条件编译
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAY_CONVERTER_H_

// 包含 ndarraytypes.h 文件，定义了与 ndarray 相关的类型和宏
#include "numpy/ndarraytypes.h"

// 声明 PyArrayArrayConverter_Type 变量，不导出给外部
extern NPY_NO_EXPORT PyTypeObject PyArrayArrayConverter_Type;

// 定义枚举类型 npy_array_converter_flags，用于标志数组转换器的特性
typedef enum {
    NPY_CH_ALL_SCALARS = 1 << 0,         // 表示所有对象都是标量
    NPY_CH_ALL_PYSCALARS = 1 << 1,       // 表示所有对象都是 Python 标量对象
} npy_array_converter_flags;

// 定义结构体 creation_item，描述创建对象的元数据
typedef struct {
    PyObject *object;                    // Python 对象
    PyArrayObject *array;                // NumPy 数组对象
    PyArray_DTypeMeta *DType;            // NumPy 数据类型元信息
    PyArray_Descr *descr;                // NumPy 数据描述符
    int scalar_input;                    // 是否标量输入的标志
} creation_item;

// 定义结构体 PyArrayArrayConverterObject，表示数组转换器对象
typedef struct {
    PyObject_VAR_HEAD                     // 可变大小对象的头部
    int narrs;                            // 数组的数量
    npy_array_converter_flags flags;      // 数组转换器的特性标志
    PyObject *wrap;                       // __array_wrap__ 缓存对象
    PyObject *wrap_type;                  // __array_wrap__ 方法的类型
    creation_item items[];                // 创建对象的元数据数组
} PyArrayArrayConverterObject;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAY_CONVERTER_H_ */
```