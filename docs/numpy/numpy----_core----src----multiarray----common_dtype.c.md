# `.\numpy\numpy\_core\src\multiarray\common_dtype.c`

```
/*
 * 定义以下宏，以确保使用 NumPy 的最新 API 版本并禁用过时的 API。
 * _MULTIARRAYMODULE 是为了在编译期间定义多维数组模块。
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 引入必要的头文件和库文件。
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/npy_common.h"
#include "numpy/arrayobject.h"

#include "convert_datatype.h"
#include "dtypemeta.h"
#include "abstractdtypes.h"
#include "npy_static_data.h"

/*
 * 该文件定义了通用的“common dtype”操作逻辑。
 * 这是复杂的，因为 NumPy 使用基于值的逻辑，并且没有明确的类型提升层次结构。
 * 不像大多数语言中的 `int32 + float32 -> float64`，而是变为 `float32`。
 * 另一个复杂的地方是基于值的提升，这意味着在许多情况下，Python 1 可能会成为 `int8` 或 `uint8`。
 *
 * 该文件实现了必要的逻辑，以便 `np.result_type(...)` 可以针对任何输入顺序给出正确的结果，并可以进一步推广到用户 DTypes。
 */

/*NUMPY_API
 * 该函数定义了通用的 DType 运算符。
 *
 * 注意，通用的 DType 不会是 "object"（除非其中一个 DType 是 object），即使 object 可以在技术上正确表示所有值。
 * 类似于 `np.result_type`，但适用于类而不是实例。
 *
 * TODO: 在暴露之前，我们应该审查返回值（例如，当没有找到通用的 DType 时不应报错）。
 *
 * @param dtype1 要找到通用类型的第一个 DType 类。
 * @param dtype2 第二个 DType 类。
 * @return 通用的 DType 或者 NULL 并设置错误。
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_CommonDType(PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2)
{
    // 如果 dtype1 等于 dtype2，则增加引用计数并返回 dtype1
    if (dtype1 == dtype2) {
        Py_INCREF(dtype1);
        return dtype1;
    }

    PyArray_DTypeMeta *common_dtype;

    // 调用 NPY_DT_CALL_common_dtype 函数尝试找到 dtype1 和 dtype2 的通用 DType
    common_dtype = NPY_DT_CALL_common_dtype(dtype1, dtype2);
    // 如果未找到通用 DType，则尝试反转参数顺序再次查找
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(common_dtype);
        common_dtype = NPY_DT_CALL_common_dtype(dtype2, dtype1);
    }
    // 如果仍然未找到通用 DType，则返回 NULL
    if (common_dtype == NULL) {
        return NULL;
    }
    // 如果找到了 Py_NotImplemented，表示 dtype1 和 dtype2 没有通用 DType
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(Py_NotImplemented);
        // 报错并返回 NULL，指示没有找到通用 DType
        PyErr_Format(npy_static_pydata.DTypePromotionError,
                "The DTypes %S and %S do not have a common DType. "
                "For example they cannot be stored in a single array unless "
                "the dtype is `object`.", dtype1, dtype2);
        return NULL;
    }
    // 返回找到的通用 DType
    return common_dtype;
}
/**
 * This function takes a list of dtypes and "reduces" them (in a sense,
 * it finds the maximal dtype). Note that "maximum" here is defined by
 * knowledge (or category or domain). A user DType must always "know"
 * about all NumPy dtypes, floats "know" about integers, integers "know"
 * about unsigned integers.
 *
 *           c
 *          / \
 *         a   \    <-- The actual promote(a, b) may be c or unknown.
 *        / \   \
 *       a   b   c
 *
 * The reduction is done "pairwise". In the above `a.__common_dtype__(b)`
 * has a result (so `a` knows more) and `a.__common_dtype__(c)` returns
 * NotImplemented (so `c` knows more).  You may notice that the result
 * `res = a.__common_dtype__(b)` is not important.  We could try to use it
 * to remove the whole branch if `res is c` or by checking if
 * `c.__common_dtype__(res) is c`.
 * Right now, we only clear initial elements in the most simple case where
 * `a.__common_dtype__(b) is a` (and thus `b` cannot alter the end-result).
 * Clearing means, we do not have to worry about them later.
 *
 * Abstract dtypes are not handled specially here.  In a first
 * version they were but this version also tried to be able to do value-based
 * behavior.
 * There may be some advantage to special casing the abstract ones (e.g.
 * so that the concrete ones do not have to deal with it), but this would
 * require more complex handling later on. See the logic in
 * default_builtin_common_dtype
 *
 * @param length Number of DTypes
 * @param dtypes List of DTypes
 */
static PyArray_DTypeMeta *
reduce_dtypes_to_most_knowledgeable(
        npy_intp length, PyArray_DTypeMeta **dtypes)
{
    assert(length >= 2); // Ensure the length is at least 2

    npy_intp half = length / 2; // Calculate the half of the length

    PyArray_DTypeMeta *res = NULL; // Initialize the result as NULL

    for (npy_intp low = 0; low < half; low++) { // Loop through the first half of the length
        npy_intp high = length - 1 - low; // Calculate the index of the opposite element
        if (dtypes[high] == dtypes[low]) { // Check if the dtypes at the high and low indices are the same
            /* Fast path for identical dtypes: do not call common_dtype */
            Py_INCREF(dtypes[low]); // Increment the reference count of dtypes[low]
            Py_XSETREF(res, dtypes[low]); // Set res to dtypes[low]
        }
        else {
            Py_XSETREF(res, NPY_DT_CALL_common_dtype(dtypes[low], dtypes[high])); // Set res to the common dtype of dtypes[low] and dtypes[high]
            if (res == NULL) { // If res is NULL
                return NULL; // Return NULL
            }
        }

        if (res == (PyArray_DTypeMeta *)Py_NotImplemented) { // Check if res is Py_NotImplemented
            /* guess at other being more "knowledgable" */
            PyArray_DTypeMeta *tmp = dtypes[low]; // Temporarily store dtypes[low]
            dtypes[low] = dtypes[high]; // Set dtypes[low] to dtypes[high]
            dtypes[high] = tmp; // Set dtypes[high] to the temporary value
        }
        else if (res == dtypes[low]) { // Check if res is equal to dtypes[low]
            /* `dtypes[high]` cannot influence result: clear */
            dtypes[high] = NULL; // Set dtypes[high] to NULL
        }
    }

    if (length == 2) { // If the length is 2
        return res; // Return res
    }
    Py_DECREF(res); // Decrease the reference count of res
    return reduce_dtypes_to_most_knowledgeable(length - half, dtypes); // Recursively call the function with the updated length and dtypes
}
/*NUMPY_API
 * Promotes a list of DTypes with each other in a way that should guarantee
 * stable results even when changing the order.  This function is smarter and
 * can often return successful and unambiguous results when
 * `common_dtype(common_dtype(dt1, dt2), dt3)` would depend on the operation
 * order or fail.  Nevertheless, DTypes should aim to ensure that their
 * common-dtype implementation is associative and commutative!  (Mainly,
 * unsigned and signed integers are not.)
 *
 * For guaranteed consistent results DTypes must implement common-Dtype
 * "transitively".  If A promotes B and B promotes C, than A must generally
 * also promote C; where "promotes" means implements the promotion.  (There
 * are some exceptions for abstract DTypes)
 *
 * In general this approach always works as long as the most generic dtype
 * is either strictly larger, or compatible with all other dtypes.
 * For example promoting float16 with any other float, integer, or unsigned
 * integer again gives a floating point number. And any floating point number
 * promotes in the "same way" as `float16`.
 * If a user inserts more than one type into the NumPy type hierarchy, this
 * can break. Given:
 *     uint24 + int32 -> int48  # Promotes to a *new* dtype!
 *
 * The following becomes problematic (order does not matter):
 *         uint24 +      int16  +           uint32  -> int64
 *    <==      (uint24 + int16) + (uint24 + uint32) -> int64
 *    <==                int32  +           uint32  -> int64
 *
 * It is impossible to achieve an `int48` result in the above.
 *
 * This is probably only resolvable by asking `uint24` to take over the
 * whole reduction step; which we currently do not do.
 * (It may be possible to notice the last up-cast and implement use something
 * like: `uint24.nextafter(int32).__common_dtype__(uint32)`, but that seems
 * even harder to grasp.)
 *
 * Note that a case where two dtypes are mixed (and know nothing about each
 * other) will always generate an error:
 *     uint24 + int48 + int64 -> Error
 *
 * Even though `int64` is a safe solution, since `uint24 + int64 -> int64` and
 * `int48 + int64 -> int64` and `int64` and there cannot be a smaller solution.
 *
 * //TODO: Maybe this function should allow not setting an error?
 *
 * @param length Number of dtypes (and values) must be at least 1
 * @param dtypes The concrete or abstract DTypes to promote
 * @return NULL or the promoted DType.
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_PromoteDTypeSequence(
        npy_intp length, PyArray_DTypeMeta **dtypes_in)
{
    // 如果只有一个dtype，直接返回该dtype，增加其引用计数
    if (length == 1) {
        Py_INCREF(dtypes_in[0]);
        return dtypes_in[0];
    }
    
    // 否则初始化结果为NULL
    PyArray_DTypeMeta *result = NULL;

    /* Copy dtypes so that we can reorder them (only allocate when many) */
    // 使用栈上的数组或堆上的数组来复制dtypes，以便重新排序（只有在长度大时才分配堆上内存）
    PyObject *_scratch_stack[NPY_MAXARGS];
    PyObject **_scratch_heap = NULL;
    PyArray_DTypeMeta **dtypes = (PyArray_DTypeMeta **)_scratch_stack;
    # 如果传入的长度超过了预定义的最大参数个数 NPY_MAXARGS，则分配一个临时内存空间 _scratch_heap 来存储 PyObject 指针数组
    if (length > NPY_MAXARGS) {
        _scratch_heap = PyMem_Malloc(length * sizeof(PyObject *));
        // 如果分配内存失败，则设置内存错误并返回 NULL
        if (_scratch_heap == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        // 将 _scratch_heap 强制转换为 PyArray_DTypeMeta 指针数组，用于存储传入的 dtypes
        dtypes = (PyArray_DTypeMeta **)_scratch_heap;
    }

    // 将 dtypes_in 数组中的数据拷贝到 dtypes 数组中，拷贝的字节数为 length * sizeof(PyObject *)
    memcpy(dtypes, dtypes_in, length * sizeof(PyObject *));

    /*
     * `result` 是最后的推广结果，通常情况下可以重复使用，除非它是 NotImplemneted。
     * 传入的 dtypes 已部分排序，并在不再相关时已清除。
     * `dtypes[0]` 将是最有知识（最高类别）的 dtype，这里我们称之为 "main_dtype"。
     */
    // 调用 reduce_dtypes_to_most_knowledgeable 函数处理 dtypes，返回推广后的结果给 result
    result = reduce_dtypes_to_most_knowledgeable(length, dtypes);
    // 如果 result 为 NULL，则跳转至 finish 标签处
    if (result == NULL) {
        goto finish;
    }
    // 将 dtypes[0] 赋值给 main_dtype，表示最有知识的 dtype
    PyArray_DTypeMeta *main_dtype = dtypes[0];

    // reduce_start 初始化为 1
    npy_intp reduce_start = 1;
    // 如果 result 是 Py_NotImplemented，则将 result 设置为 NULL
    if (result == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_SETREF(result, NULL);
    }
    else {
        /* (new) first value is already taken care of in `result` */
        // 否则，reduce_start 设置为 2，表示处理的起始位置
        reduce_start = 2;
    }
    /*
     * 到此为止，我们最多只查看了每个 DType 一次。
     * `main_dtype` 必须了解所有其他 dtype（否则将会失败），
     * 并且其 `common_dtype` 返回的所有 dtype 必须保证能够互相推广成功。
     * 在这一点上，"main DType" 的任务是确保顺序无关紧要。
     * 如果这证明是一个限制，这种 "reduction" 将必须变成一个默认版本，并允许 DType 来覆盖它。
     */
    // prev 初始化为 NULL
    PyArray_DTypeMeta *prev = NULL;
   `
    for (npy_intp i = reduce_start; i < length; i++) {
        // 循环遍历从 reduce_start 开始的 dtypes 数组
        if (dtypes[i] == NULL || dtypes[i] == prev) {
            // 如果当前 dtypes[i] 是 NULL 或者与前一个相同，则跳过当前循环
            continue;
        }
        /*
         * 将当前 dtype 与主 dtype 进行"提升"（promotion），假设结果不会低于主 dtype 的类别。
         */
        PyArray_DTypeMeta *promotion = NPY_DT_CALL_common_dtype(
                main_dtype, dtypes[i]);
        if (promotion == NULL) {
            // 如果提升失败，则设置 result 为 NULL，并跳转至 finish 标签处
            Py_XSETREF(result, NULL);
            goto finish;
        }
        else if ((PyObject *)promotion == Py_NotImplemented) {
            // 如果提升操作返回 Py_NotImplemented，则处理错误情况
            Py_DECREF(Py_NotImplemented);
            Py_XSETREF(result, NULL);
            PyObject *dtypes_in_tuple = PyTuple_New(length);
            if (dtypes_in_tuple == NULL) {
                goto finish;
            }
            // 构建一个包含所有 dtypes 的元组
            for (npy_intp l=0; l < length; l++) {
                Py_INCREF(dtypes_in[l]);
                PyTuple_SET_ITEM(dtypes_in_tuple, l, (PyObject *)dtypes_in[l]);
            }
            // 设置错误信息并跳转至 finish 标签处
            PyErr_Format(npy_static_pydata.DTypePromotionError,
                    "The DType %S could not be promoted by %S. This means that "
                    "no common DType exists for the given inputs. "
                    "For example they cannot be stored in a single array unless "
                    "the dtype is `object`. The full list of DTypes is: %S",
                    dtypes[i], main_dtype, dtypes_in_tuple);
            Py_DECREF(dtypes_in_tuple);
            goto finish;
        }
        if (result == NULL) {
            // 如果 result 为 NULL，则将其设置为当前的 promotion
            result = promotion;
            continue;
        }

        /*
         * 以上步骤完成提升后，现在与当前的 result 进行"减少"（reduce）操作；
         * 注意在典型情况下，我们预期这一步骤不会产生实际操作。
         */
        Py_SETREF(result, PyArray_CommonDType(result, promotion));
        Py_DECREF(promotion);
        if (result == NULL) {
            // 如果操作后 result 为 NULL，则跳转至 finish 标签处
            goto finish;
        }
    }

  finish:
    // 释放临时内存
    PyMem_Free(_scratch_heap);
    // 返回最终的 result
    return result;
}



# 这是一个单独的右大括号 '}'，通常用于结束一个代码块或者数据结构的定义。
```