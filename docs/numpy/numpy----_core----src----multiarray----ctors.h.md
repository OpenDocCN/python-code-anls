# `.\numpy\numpy\_core\src\multiarray\ctors.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_CTORS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CTORS_H_

// 声明一个在多维数组模块中不导出的错误消息字符串常量
extern NPY_NO_EXPORT const char *npy_no_copy_err_msg;

// 根据给定的描述符和参数创建一个新的数组对象
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj);

// 根据描述符、维度、步长、数据等创建一个新的数组对象，同时传递一个基础对象
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescrAndBase(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base);


/* NewFromDescriptor 的私有选项 */
typedef enum {
    /*
     * 指示数组应该被置零，可以使用 calloc 来实现
     * （与其他选项类似，但具有特定的零化语义）。
     */
    _NPY_ARRAY_ZEROED = 1 << 0,
    /* 是否允许空字符串（由保证 dtype 一致性隐含） */
    _NPY_ARRAY_ALLOW_EMPTY_STRING = 1 << 1,
    /*
     * 如果我们从现有数组中获取视图并使用其 dtype，则必须保留该 dtype
     * （例如对于子数组和 S0 类型，也可能适用于将来存储更多元数据的 dtype）。
     */
    _NPY_ARRAY_ENSURE_DTYPE_IDENTITY = 1 << 2,
} _NPY_CREATION_FLAGS;

// 根据描述符和参数创建新的数组对象，同时传递一个自定义的标志位
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr_int(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base, _NPY_CREATION_FLAGS cflags);

// 根据原型数组创建一个具有指定形状的新数组对象
NPY_NO_EXPORT PyObject *
PyArray_NewLikeArrayWithShape(
        PyArrayObject *prototype, NPY_ORDER order,
        PyArray_Descr *descr, PyArray_DTypeMeta *dtype,
        int ndim, npy_intp const *dims, int subok);

// 创建一个新的数组对象，使用给定的参数和标志位
NPY_NO_EXPORT PyObject *
PyArray_New(
        PyTypeObject *, int nd, npy_intp const *,
        int, npy_intp const*, void *, int, int, PyObject *);

// 根据类似数组对象创建一个数组，支持从不同的对象类型创建
NPY_NO_EXPORT PyObject *
_array_from_array_like(PyObject *op,
        PyArray_Descr *requested_dtype, npy_bool writeable, PyObject *context,
        int copy, int *was_copied_by__array__);

// 尝试从任意对象创建一个数组对象，支持从不同的对象类型创建
NPY_NO_EXPORT PyObject *
PyArray_FromAny_int(PyObject *op, PyArray_Descr *in_descr,
                    PyArray_DTypeMeta *in_DType, int min_depth, int max_depth,
                    int flags, PyObject *context, int *was_scalar);

// 尝试从任意对象创建一个数组对象，支持从不同的对象类型创建
NPY_NO_EXPORT PyObject *
PyArray_FromAny(PyObject *op, PyArray_Descr *newtype, int min_depth,
                int max_depth, int flags, PyObject *context);

// 尝试从任意对象创建一个数组对象，支持从不同的对象类型创建，强制满足指定要求
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny_int(PyObject *op, PyArray_Descr *in_descr,
                         PyArray_DTypeMeta *in_DType, int min_depth,
                         int max_depth, int requires, PyObject *context);

// 尝试从任意对象创建一个数组对象，支持从不同的对象类型创建，强制满足指定要求
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny(PyObject *op, PyArray_Descr *descr, int min_depth,
                     int max_depth, int requires, PyObject *context);

// 从现有数组对象创建一个新的数组对象，可以指定新的数据类型和标志位
NPY_NO_EXPORT PyObject *
PyArray_FromArray(PyArrayObject *arr, PyArray_Descr *newtype, int flags);

// 从结构接口对象创建一个新的数组对象
NPY_NO_EXPORT PyObject *
PyArray_FromStructInterface(PyObject *input);

#endif  // NUMPY_CORE_SRC_MULTIARRAY_CTORS_H_
/* 
   从给定的接口对象创建一个 NumPy 数组对象，并返回该对象
   输入参数 input: 接口对象的指针
   返回值：新创建的 NumPy 数组对象的指针
*/
NPY_NO_EXPORT PyObject *
PyArray_FromInterface(PyObject *input);

/*
   根据给定的数组对象 op、数据描述符 descr、复制标志 copy，
   以及指向整型变量的指针 was_copied_by__array__，
   创建一个新的 NumPy 数组对象并返回该对象
   输入参数 op: 数组对象的指针
   输入参数 descr: 数据描述符的指针
   输入参数 copy: 复制标志，表示是否需要复制数组数据
   输入输出参数 was_copied_by__array__: 指向整型变量的指针，用于记录是否通过 __array__ 复制了数组
   返回值：新创建的 NumPy 数组对象的指针
*/
NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr_int(PyObject *op, PyArray_Descr *descr, int copy,
                          int *was_copied_by__array__);

/*
   根据给定的数组对象 op、数据类型描述符 typecode 和上下文 context，
   创建一个新的 NumPy 数组对象并返回该对象
   输入参数 op: 数组对象的指针
   输入参数 typecode: 数据类型描述符的指针
   输入参数 context: 上下文对象，用于创建数组的环境信息
   返回值：新创建的 NumPy 数组对象的指针
*/
NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr(PyObject *op, PyArray_Descr *typecode,
                      PyObject *context);

/*
   确保给定对象 op 是一个 NumPy 数组对象，如果不是则尝试将其转换为数组对象
   输入参数 op: 要确保的对象的指针
   返回值：如果成功转换为数组，则返回转换后的数组对象的指针；否则返回 NULL
*/
NPY_NO_EXPORT PyObject *
PyArray_EnsureArray(PyObject *op);

/*
   确保给定对象 op 是一个任意类型的 NumPy 数组对象，如果不是则尝试将其转换为数组对象
   输入参数 op: 要确保的对象的指针
   返回值：如果成功转换为任意类型的数组，则返回转换后的数组对象的指针；否则返回 NULL
*/
NPY_NO_EXPORT PyObject *
PyArray_EnsureAnyArray(PyObject *op);

/*
   将源数组对象 src 的数据复制到目标数组对象 dest 中
   输入参数 dest: 目标数组对象的指针
   输入参数 src: 源数组对象的指针
   返回值：如果成功复制则返回 0；否则返回 -1
*/
NPY_NO_EXPORT int
PyArray_CopyAnyInto(PyArrayObject *dest, PyArrayObject *src);

/*
   检查给定数组对象 arr 上的轴参数 axis，并根据指定的 flags 进行检查和调整
   输入参数 arr: 数组对象的指针
   输入输出参数 axis: 指向整型变量的指针，表示轴参数
   输入参数 flags: 标志位，用于控制检查行为
   返回值：如果成功检查并调整轴参数则返回 0；否则返回 -1
*/
NPY_NO_EXPORT PyObject *
PyArray_CheckAxis(PyArrayObject *arr, int *axis, int flags);

/*
   将源数组对象 src 的数据按照指定的顺序 order 复制到目标数组对象 dst 中
   输入参数 dst: 目标数组对象的指针
   输入参数 src: 源数组对象的指针
   输入参数 order: 指定复制顺序的参数
   返回值：如果成功复制则返回 0；否则返回 -1
*/
NPY_NO_EXPORT int
PyArray_CopyAsFlat(PyArrayObject *dst, PyArrayObject *src,
                   NPY_ORDER order);

/*
   根据给定的数组维度 dims、维度数量 nd、元素大小 itemsize、标志 inflag
   以及对象标志 objflags，填充数组的步长 strides
   输入参数 strides: 数组的步长数组的指针
   输入参数 dims: 数组的维度数组的指针
   输入参数 nd: 数组的维度数量
   输入参数 itemsize: 数组元素的大小
   输入参数 inflag: 输入标志，用于指示输入数据的布局特性
   输入输出参数 objflags: 指向整型变量的指针，用于记录对象的标志
*/
NPY_NO_EXPORT void
_array_fill_strides(npy_intp *strides, npy_intp const *dims, int nd, size_t itemsize,
                    int inflag, int *objflags);

/*
   将未对齐的源数据区域 src 复制到未对齐的目标数据区域 dst 中
   输入参数 dst: 目标数据区域的指针
   输入参数 outstrides: 目标数据区域的步长
   输入参数 src: 源数据区域的指针
   输入参数 instrides: 源数据区域的步长
   输入参数 N: 复制的数据元素数量
   输入参数 elsize: 数据元素的大小
*/
NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                             npy_intp instrides, npy_intp N, int elsize);

/*
   对给定数据区域 p 进行字节交换操作，交换的数据数量为 n，每个数据元素的大小为 size
   输入参数 p: 数据区域的指针
   输入参数 stride: 数据区域的步长
   输入参数 n: 数据区域中的数据元素数量
   输入参数 size: 每个数据元素的大小
*/
NPY_NO_EXPORT void
_strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size);

/*
   将源数据区域 src 复制并交换字节顺序后存储到目标数据区域 dst 中
   输入参数 dst: 目标数据区域的指针
   输入参数 src: 源数据区域的指针
   输入参数 itemsize: 数据元素的大小
   输入参数 numitems: 复制的数据元素数量
   输入参数 srcstrides: 源数据区域的步长
   输入参数 swap: 指示是否进行字节交换的标志
*/
NPY_NO_EXPORT void
copy_and_swap(void *dst, void *src, int itemsize, npy_intp numitems,
              npy_intp srcstrides, int swap);

/*
   对给定数据区域 p 中的每个数据元素进行字节交换操作，每个数据元素的大小为 size
   输入参数 p: 数据区域的指针
   输入参数 n: 数据区域中的数据元素数量
   输入参数 size: 每个数据元素的大小
*/
NPY_NO_EXPORT void
byte_swap_vector(void *p, npy_intp n, int size);

/*
   根据给定的维度 dims、维度数量 nd、数据描述符 descr、数据类型 dtype
   以及布局标志 is_f_order，创建一个所有元素为零的 NumPy 数组对象
   输入参数 nd: 数组的维度数量
   输入参数 dims: 数组的维度数组的指针
   输入参数 descr: 数据描述符的指针
   输入参数 dtype: 数据类型的元数据描述符
   输入参数 is_f_order: 布局标志，表示数组是否以 Fortran 顺序存储
   返回值：新创建的 NumPy 数组对象的指针
*/
NPY_NO_EXPORT PyObject *
PyArray_Zeros_int(int nd, npy_intp const *dims, PyArray_Descr *descr,
                  PyArray_DTypeMeta *dtype, int is_f_order);

/*
   根据给定的维度 dims、维度数量 nd、数据描述符 descr、数据类型 dtype
   以及布局标志 is_f_order，创建一个未初始化的 NumPy 数组对象
   输入参数 nd: 数组的维度数量
   输入参数 dims: 数组的维度数组的指针
   输入参数 descr: 数据描述符的指针
   输入参数 dtype: 数据类型的元
```