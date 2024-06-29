# `.\numpy\numpy\_core\include\numpy\ndarrayobject.h`

```
/*
 * DON'T INCLUDE THIS DIRECTLY.
 */
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NDARRAYOBJECT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NDARRAYOBJECT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include "ndarraytypes.h"
#include "dtype_api.h"

/* Includes the "function" C-API -- these are all stored in a
   list of pointers --- one for each file
   The two lists are concatenated into one in multiarray.

   They are available as import_array()
*/

#include "__multiarray_api.h"

/*
 * Include any definitions which are defined differently for 1.x and 2.x
 * (Symbols only available on 2.x are not there, but rather guarded.)
 */
#include "npy_2_compat.h"

/* C-API that requires previous API to be defined */

// 定义宏，检查对象是否为 PyArray_Descr 类型
#define PyArray_DescrCheck(op) PyObject_TypeCheck(op, &PyArrayDescr_Type)

// 定义宏，检查对象是否为 PyArray 类型
#define PyArray_Check(op) PyObject_TypeCheck(op, &PyArray_Type)

// 定义宏，精确检查对象是否为 PyArray 类型
#define PyArray_CheckExact(op) (((PyObject*)(op))->ob_type == &PyArray_Type)

// 定义宏，检查对象是否有数组接口类型
#define PyArray_HasArrayInterfaceType(op, type, context, out)                 \
        ((((out)=PyArray_FromStructInterface(op)) != Py_NotImplemented) ||    \
         (((out)=PyArray_FromInterface(op)) != Py_NotImplemented) ||          \
         (((out)=PyArray_FromArrayAttr(op, type, context)) !=                 \
          Py_NotImplemented))

// 定义宏，检查对象是否有数组接口
#define PyArray_HasArrayInterface(op, out)                                    \
        PyArray_HasArrayInterfaceType(op, NULL, NULL, out)

// 定义宏，检查对象是否为零维数组
#define PyArray_IsZeroDim(op) (PyArray_Check(op) && \
                               (PyArray_NDIM((PyArrayObject *)(op)) == 0))

// 定义宏，检查对象是否为标量
#define PyArray_IsScalar(obj, cls)                                            \
        (PyObject_TypeCheck(obj, &Py##cls##ArrType_Type))

// 定义宏，检查对象是否为标量或零维数组
#define PyArray_CheckScalar(m) (PyArray_IsScalar(m, Generic) ||               \
                                PyArray_IsZeroDim(m))

// 定义宏，检查对象是否为 Python 数字类型
#define PyArray_IsPythonNumber(obj)                                           \
        (PyFloat_Check(obj) || PyComplex_Check(obj) ||                        \
         PyLong_Check(obj) || PyBool_Check(obj))

// 定义宏，检查对象是否为整数标量
#define PyArray_IsIntegerScalar(obj) (PyLong_Check(obj)                       \
              || PyArray_IsScalar((obj), Integer))

// 定义宏，检查对象是否为 Python 标量类型
#define PyArray_IsPythonScalar(obj)                                           \
        (PyArray_IsPythonNumber(obj) || PyBytes_Check(obj) ||                 \
         PyUnicode_Check(obj))

// 定义宏，检查对象是否为任何类型的标量
#define PyArray_IsAnyScalar(obj)                                              \
        (PyArray_IsScalar(obj, Generic) || PyArray_IsPythonScalar(obj))

// 定义宏，检查对象是否为任何类型的标量或零维数组
#define PyArray_CheckAnyScalar(obj) (PyArray_IsPythonScalar(obj) ||           \
                                     PyArray_CheckScalar(obj))

// 定义宏，获取对象的连续副本
#define PyArray_GETCONTIGUOUS(m) (PyArray_ISCONTIGUOUS(m) ?                   \
                                  Py_INCREF(m), (m) :                         \
                                  (PyArrayObject *)(PyArray_Copy(m)))
# 定义宏 PyArray_SAMESHAPE(a1,a2)，用于比较两个数组的形状是否相同
#define PyArray_SAMESHAPE(a1,a2) ((PyArray_NDIM(a1) == PyArray_NDIM(a2)) &&   \
                                  PyArray_CompareLists(PyArray_DIMS(a1),      \
                                                       PyArray_DIMS(a2),      \
                                                       PyArray_NDIM(a1)))

# 定义宏 PyArray_SIZE(m)，计算数组 m 的总元素数
#define PyArray_SIZE(m) PyArray_MultiplyList(PyArray_DIMS(m), PyArray_NDIM(m))

# 定义宏 PyArray_NBYTES(m)，计算数组 m 的总字节数
#define PyArray_NBYTES(m) (PyArray_ITEMSIZE(m) * PyArray_SIZE(m))

# 定义宏 PyArray_FROM_O(m)，根据对象 m 创建一个 NumPy 数组
#define PyArray_FROM_O(m) PyArray_FromAny(m, NULL, 0, 0, 0, NULL)

# 定义宏 PyArray_FROM_OF(m,flags)，根据对象 m 和标志 flags 创建一个 NumPy 数组
#define PyArray_FROM_OF(m,flags) PyArray_CheckFromAny(m, NULL, 0, 0, flags,   \
                                                      NULL)

# 定义宏 PyArray_FROM_OT(m,type)，根据对象 m 和数据类型 type 创建一个 NumPy 数组
#define PyArray_FROM_OT(m,type) PyArray_FromAny(m,                            \
                                PyArray_DescrFromType(type), 0, 0, 0, NULL)

# 定义宏 PyArray_FROM_OTF(m, type, flags)，根据对象 m、数据类型 type 和标志 flags 创建一个 NumPy 数组
#define PyArray_FROM_OTF(m, type, flags) \
        PyArray_FromAny(m, PyArray_DescrFromType(type), 0, 0, \
                        (((flags) & NPY_ARRAY_ENSURECOPY) ? \
                         ((flags) | NPY_ARRAY_DEFAULT) : (flags)), NULL)

# 定义宏 PyArray_FROMANY(m, type, min, max, flags)，根据对象 m、数据类型 type、最小深度 min、最大深度 max 和标志 flags 创建一个 NumPy 数组
#define PyArray_FROMANY(m, type, min, max, flags) \
        PyArray_FromAny(m, PyArray_DescrFromType(type), min, max, \
                        (((flags) & NPY_ARRAY_ENSURECOPY) ? \
                         (flags) | NPY_ARRAY_DEFAULT : (flags)), NULL)

# 定义宏 PyArray_ZEROS(m, dims, type, is_f_order)，创建一个元素类型为 type、形状为 dims、是否 F 风格的全零 NumPy 数组
#define PyArray_ZEROS(m, dims, type, is_f_order) \
        PyArray_Zeros(m, dims, PyArray_DescrFromType(type), is_f_order)

# 定义宏 PyArray_EMPTY(m, dims, type, is_f_order)，创建一个元素类型为 type、形状为 dims、是否 F 风格的空 NumPy 数组
#define PyArray_EMPTY(m, dims, type, is_f_order) \
        PyArray_Empty(m, dims, PyArray_DescrFromType(type), is_f_order)

# 定义宏 PyArray_FILLWBYTE(obj, val)，使用值 val 填充数组 obj 的每个字节
#define PyArray_FILLWBYTE(obj, val) memset(PyArray_DATA(obj), val, \
                                           PyArray_NBYTES(obj))

# 定义宏 PyArray_ContiguousFromAny(op, type, min_depth, max_depth)，从对象 op 创建一个连续的 NumPy 数组，元素类型为 type，最小深度 min_depth，最大深度 max_depth
#define PyArray_ContiguousFromAny(op, type, min_depth, max_depth) \
        PyArray_FromAny(op, PyArray_DescrFromType(type), min_depth, \
                              max_depth, NPY_ARRAY_DEFAULT, NULL)

# 定义宏 PyArray_EquivArrTypes(a1, a2)，比较两个数组 a1 和 a2 是否类型等效
#define PyArray_EquivArrTypes(a1, a2) \
        PyArray_EquivTypes(PyArray_DESCR(a1), PyArray_DESCR(a2))

# 定义宏 PyArray_EquivByteorders(b1, b2)，比较两个字节序 b1 和 b2 是否等效
#define PyArray_EquivByteorders(b1, b2) \
        (((b1) == (b2)) || (PyArray_ISNBO(b1) == PyArray_ISNBO(b2)))

# 定义宏 PyArray_SimpleNew(nd, dims, typenum)，简单地创建一个 NumPy 数组，元素类型为 typenum，形状为 dims，维度数为 nd
#define PyArray_SimpleNew(nd, dims, typenum) \
        PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, NULL, 0, 0, NULL)

# 定义宏 PyArray_SimpleNewFromData(nd, dims, typenum, data)，从给定的数据 data 创建一个 NumPy 数组，元素类型为 typenum，形状为 dims，维度数为 nd
#define PyArray_SimpleNewFromData(nd, dims, typenum, data) \
        PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, \
                    data, 0, NPY_ARRAY_CARRAY, NULL)

# 定义宏 PyArray_SimpleNewFromDescr(nd, dims, descr)，根据描述符 descr 创建一个 NumPy 数组，形状为 dims，维度数为 nd
#define PyArray_SimpleNewFromDescr(nd, dims, descr) \
        PyArray_NewFromDescr(&PyArray_Type, descr, nd, dims, \
                             NULL, NULL, 0, NULL)

# 定义宏 PyArray_ToScalar(data, arr)，将数据 data 转换为与数组 arr 兼容的标量
#define PyArray_ToScalar(data, arr) \
        PyArray_Scalar(data, PyArray_DESCR(arr), (PyObject *)arr)

/* 这些宏可能在循环内部性能更高，因为避免了 obj 内部的解引用操作 */
/* 宏定义：根据给定索引 i 计算一维数组中的元素指针 */
#define PyArray_GETPTR1(obj, i) ((void *)(PyArray_BYTES(obj) + \
                                         (i)*PyArray_STRIDES(obj)[0]))

/* 宏定义：根据给定索引 i, j 计算二维数组中的元素指针 */
#define PyArray_GETPTR2(obj, i, j) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1]))

/* 宏定义：根据给定索引 i, j, k 计算三维数组中的元素指针 */
#define PyArray_GETPTR3(obj, i, j, k) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2]))

/* 宏定义：根据给定索引 i, j, k, l 计算四维数组中的元素指针 */
#define PyArray_GETPTR4(obj, i, j, k, l) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2] + \
                                            (l)*PyArray_STRIDES(obj)[3]))

/* 函数：丢弃写回备份（若存在） */
static inline void
PyArray_DiscardWritebackIfCopy(PyArrayObject *arr)
{
    PyArrayObject_fields *fa = (PyArrayObject_fields *)arr;
    if (fa && fa->base) {
        if (fa->flags & NPY_ARRAY_WRITEBACKIFCOPY) {
            PyArray_ENABLEFLAGS((PyArrayObject*)fa->base, NPY_ARRAY_WRITEABLE);
            Py_DECREF(fa->base);
            fa->base = NULL;
            PyArray_CLEARFLAGS(arr, NPY_ARRAY_WRITEBACKIFCOPY);
        }
    }
}

/* 宏定义：替换数组描述符 */
#define PyArray_DESCR_REPLACE(descr) do { \
                PyArray_Descr *_new_; \
                _new_ = PyArray_DescrNew(descr); \
                Py_XDECREF(descr); \
                descr = _new_; \
        } while(0)

/* 宏定义：复制数组并保证连续性 */
#define PyArray_Copy(obj) PyArray_NewCopy(obj, NPY_CORDER)

/* 宏定义：从 Python 对象创建数组（保证数组行为和数组化） */
#define PyArray_FromObject(op, type, min_depth, max_depth) \
        PyArray_FromAny(op, PyArray_DescrFromType(type), min_depth, \
                              max_depth, NPY_ARRAY_BEHAVED | \
                                         NPY_ARRAY_ENSUREARRAY, NULL)

/* 宏定义：从 Python 对象创建连续数组 */
#define PyArray_ContiguousFromObject(op, type, min_depth, max_depth) \
        PyArray_FromAny(op, PyArray_DescrFromType(type), min_depth, \
                              max_depth, NPY_ARRAY_DEFAULT | \
                                         NPY_ARRAY_ENSUREARRAY, NULL)

/* 宏定义：从 Python 对象复制数组 */
#define PyArray_CopyFromObject(op, type, min_depth, max_depth) \
        PyArray_FromAny(op, PyArray_DescrFromType(type), min_depth, \
                        max_depth, NPY_ARRAY_ENSURECOPY | \
                                   NPY_ARRAY_DEFAULT | \
                                   NPY_ARRAY_ENSUREARRAY, NULL)

/* 宏定义：将数组转换为指定类型 */
#define PyArray_Cast(mp, type_num)                                            \
        PyArray_CastToType(mp, PyArray_DescrFromType(type_num), 0)

/* 宏定义：在指定轴上采取数组元素 */
#define PyArray_Take(ap, items, axis)                                         \
        PyArray_TakeFrom(ap, items, axis, NULL, NPY_RAISE)
/*
   定义一个宏，用于在数组中放置元素。此宏调用 PyArray_PutTo 函数，
   将 items 和 values 放置到 ap 中，如果出现问题则会抛出异常。
*/
#define PyArray_Put(ap, items, values) \
        PyArray_PutTo(ap, items, values, NPY_RAISE)


/*
   检查字典中的某个键是否是元组的“标题”条目（即在字段字典中的重复条目）。
   如果元组的长度不为3，则返回0。
   否则，获取元组的第三个元素作为 title，然后比较 key 和 title 的对象身份。
   如果它们相等，则返回1。
   在 PyPy 版本下，由于字典键不总是保留对象身份，使用值比较作为备选方案。
*/
static inline int
NPY_TITLE_KEY_check(PyObject *key, PyObject *value)
{
    PyObject *title;
    if (PyTuple_Size(value) != 3) {
        return 0;
    }
    title = PyTuple_GetItem(value, 2);
    if (key == title) {
        return 1;
    }
#ifdef PYPY_VERSION
    /*
     * 在 PyPy 中，字典键并不总是保持对象身份。
     * 因此，如果 key 和 title 都是 Unicode 对象，则通过值比较判断它们是否相等。
     * 如果相等，则返回1；否则返回0。
     */
    if (PyUnicode_Check(title) && PyUnicode_Check(key)) {
        return PyUnicode_Compare(title, key) == 0 ? 1 : 0;
    }
#endif
    return 0;
}

/* 宏，用于向后兼容“if NPY_TITLE_KEY(key, value) { ...” 的旧代码 */
#define NPY_TITLE_KEY(key, value) (NPY_TITLE_KEY_check((key), (value)))

/*
   定义一个宏 DEPRECATE，用于发出 DeprecationWarning 警告。
   使用 PyErr_WarnEx 函数，警告消息为 msg，级别为 1。
*/
#define DEPRECATE(msg) PyErr_WarnEx(PyExc_DeprecationWarning,msg,1)
/*
   定义一个宏 DEPRECATE_FUTUREWARNING，用于发出 FutureWarning 警告。
   使用 PyErr_WarnEx 函数，警告消息为 msg，级别为 1。
*/
#define DEPRECATE_FUTUREWARNING(msg) PyErr_WarnEx(PyExc_FutureWarning,msg,1)


/*
 * 这些宏和函数需要运行时版本检查，而这些检查只在 `npy_2_compat.h` 中定义。
 * 因此它们不能成为 `ndarraytypes.h` 的一部分，后者试图自包含。
 */

/*
   返回给定数组对象的元素大小（以字节为单位）。
   通过 PyArrayObject 的 descr 成员访问 PyDataType_ELSIZE 宏得到结果。
*/
static inline npy_intp
PyArray_ITEMSIZE(const PyArrayObject *arr)
{
    return PyDataType_ELSIZE(((PyArrayObject_fields *)arr)->descr);
}

/*
   检查给定数据类型对象是否具有字段（即是否是传统的数据类型描述符且具有非空名称）。
   使用 PyDataType_ISLEGACY 和 PyDataType_NAMES 宏来实现。
*/
#define PyDataType_HASFIELDS(obj) (PyDataType_ISLEGACY((PyArray_Descr*)(obj)) && PyDataType_NAMES((PyArray_Descr*)(obj)) != NULL)
/*
   检查给定数据类型对象是否具有子数组。
   使用 PyDataType_ISLEGACY 和 PyDataType_SUBARRAY 宏来实现。
*/
#define PyDataType_HASSUBARRAY(dtype) (PyDataType_ISLEGACY(dtype) && PyDataType_SUBARRAY(dtype) != NULL)
/*
   检查给定数据类型对象是否为未定大小（即其元素大小为0且没有字段）。
*/
#define PyDataType_ISUNSIZED(dtype) ((dtype)->elsize == 0 && \
                                      !PyDataType_HASFIELDS(dtype))

/*
   检查给定数据类型对象是否具有特定标志。
   使用 PyDataType_FLAGS 宏来获取标志位，然后与指定的标志进行比较。
*/
#define PyDataType_FLAGCHK(dtype, flag) \
        ((PyDataType_FLAGS(dtype) & (flag)) == (flag))

/*
   检查给定数据类型对象是否具有引用计数。
   使用 PyDataType_FLAGCHK 宏来检查 NPY_ITEM_REFCOUNT 标志位。
*/
#define PyDataType_REFCHK(dtype) \
        PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)

/*
   在处理给定数据类型对象时，开始线程（如果没有需要 Python API 支持的标志位）。
   使用 NPY_BEGIN_THREADS 宏来启动线程。
*/
#define NPY_BEGIN_THREADS_DESCR(dtype) \
        do {if (!(PyDataType_FLAGCHK((dtype), NPY_NEEDS_PYAPI))) \
                NPY_BEGIN_THREADS;} while (0);

/*
   在处理给定数据类型对象时，结束线程（如果没有需要 Python API 支持的标志位）。
   使用 NPY_END_THREADS 宏来结束线程。
*/
#define NPY_END_THREADS_DESCR(dtype) \
        do {if (!(PyDataType_FLAGCHK((dtype), NPY_NEEDS_PYAPI))) \
                NPY_END_THREADS; } while (0);

/*
   如果不是在 NPY_INTERNAL_BUILD 且 NPY_INTERNAL_BUILD 未定义时，
   此处的内部副本现在在 `dtypemeta.h` 中定义。
 */

/*
 * `PyArray_Scalar` 与此函数相同，但会将大多数 NumPy 类型转换为 Python 标量。
 */
static inline PyObject *
PyArray_GETITEM(const PyArrayObject *arr, const char *itemptr)
{
    /*
       使用 PyArrayObject 的 descr 成员来获取 PyDataType_GetArrFuncs 函数，
       然后调用其 getitem 方法，以 itemptr 作为参数获取数组对象 arr 的元素。
    */
    return PyDataType_GetArrFuncs(((PyArrayObject_fields *)arr)->descr)->getitem(
                                        (void *)itemptr, (PyArrayObject *)arr);
}
/*
 * SETITEM should only be used if it is known that the value is a scalar
 * and of a type understood by the arrays dtype.
 * Use `PyArray_Pack` if the value may be of a different dtype.
 */
static inline int
PyArray_SETITEM(PyArrayObject *arr, char *itemptr, PyObject *v)
{
    // 获取数组描述符的类型函数集合，并调用其 setitem 方法设置值 v 到数组 arr 的指定位置 itemptr
    return PyDataType_GetArrFuncs(((PyArrayObject_fields *)arr)->descr)->setitem(v, itemptr, arr);
}
#endif  /* not internal */


#ifdef __cplusplus
}
#endif


#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NDARRAYOBJECT_H_ */
```