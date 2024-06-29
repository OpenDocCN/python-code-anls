# `.\numpy\numpy\_core\src\umath\special_integer_comparisons.cpp`

```py
/*
 * 包含 Python.h 头文件，提供 Python C API 功能支持。
 */
#include <Python.h>

/*
 * 避免使用过时的 NumPy API，并设置使用的 NumPy API 版本。
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

/*
 * 包含 NumPy 的相关头文件，定义了 ndarray 类型、数学运算、通用函数等。
 */
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

/*
 * 包含抽象数据类型、分派功能、数据类型元信息、数据类型转换等相关头文件。
 */
#include "abstractdtypes.h"
#include "dispatching.h"
#include "dtypemeta.h"
#include "convert_datatype.h"

/*
 * 包含 legacy_array_method.h 头文件，用于获取 legacy ufunc 循环的包装函数。
 */
#include "legacy_array_method.h"

/*
 * 包含 special_integer_comparisons.h 头文件，用于特殊整数比较的函数。
 */
#include "special_integer_comparisons.h"


/*
 * 用于模板中的辅助函数，避免未覆盖的 switch 分支导致的警告。
 */
enum class COMP {
    EQ, NE, LT, LE, GT, GE,
};

/*
 * 根据 COMP 枚举值返回对应的比较名称字符串。
 */
static char const *
comp_name(COMP comp) {
    switch(comp) {
        case COMP::EQ: return "equal";
        case COMP::NE: return "not_equal";
        case COMP::LT: return "less";
        case COMP::LE: return "less_equal";
        case COMP::GT: return "greater";
        case COMP::GE: return "greater_equal";
        default:
            assert(0);  // 如果出现未知的 COMP 值，触发断言错误。
            return nullptr;
    }
}

/*
 * 模板函数，根据 result 参数设置输出数组的布尔值结果。
 * 返回值为 0 表示成功执行。
 */
template <bool result>
static int
fixed_result_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];  // 获取数组的第一个维度大小
    char *out = data[2];  // 获取输出数组的起始地址
    npy_intp stride = strides[2];  // 获取输出数组的步幅

    while (N--) {
        *reinterpret_cast<npy_bool *>(out) = result;  // 将 result 赋值给输出数组的当前元素
        out += stride;  // 移动到下一个输出数组元素
    }
    return 0;  // 返回表示成功执行
}

/*
 * 内联函数，根据数据类型 typenum 获取其最小值和最大值。
 */
static inline void
get_min_max(int typenum, long long *min, unsigned long long *max)
{
    *min = 0;  // 默认最小值为 0

    switch (typenum) {
        case NPY_BYTE:
            *min = NPY_MIN_BYTE;  // 设置为 NPY_BYTE 类型的最小值
            *max = NPY_MAX_BYTE;  // 设置为 NPY_BYTE 类型的最大值
            break;
        case NPY_UBYTE:
            *max = NPY_MAX_UBYTE;  // 设置为 NPY_UBYTE 类型的最大值
            break;
        case NPY_SHORT:
            *min = NPY_MIN_SHORT;  // 设置为 NPY_SHORT 类型的最小值
            *max = NPY_MAX_SHORT;  // 设置为 NPY_SHORT 类型的最大值
            break;
        case NPY_USHORT:
            *max = NPY_MAX_USHORT;  // 设置为 NPY_USHORT 类型的最大值
            break;
        case NPY_INT:
            *min = NPY_MIN_INT;  // 设置为 NPY_INT 类型的最小值
            *max = NPY_MAX_INT;  // 设置为 NPY_INT 类型的最大值
            break;
        case NPY_UINT:
            *max = NPY_MAX_UINT;  // 设置为 NPY_UINT 类型的最大值
            break;
        case NPY_LONG:
            *min = NPY_MIN_LONG;  // 设置为 NPY_LONG 类型的最小值
            *max = NPY_MAX_LONG;  // 设置为 NPY_LONG 类型的最大值
            break;
        case NPY_ULONG:
            *max = NPY_MAX_ULONG;  // 设置为 NPY_ULONG 类型的最大值
            break;
        case NPY_LONGLONG:
            *min = NPY_MIN_LONGLONG;  // 设置为 NPY_LONGLONG 类型的最小值
            *max = NPY_MAX_LONGLONG;  // 设置为 NPY_LONGLONG 类型的最大值
            break;
        case NPY_ULONGLONG:
            *max = NPY_MAX_ULONGLONG;  // 设置为 NPY_ULONGLONG 类型的最大值
            break;
        default:
            *max = 0;  // 默认最大值为 0
            assert(0);  // 如果出现未知的 typenum 值，触发断言错误
    }
}

/*
 * 内联函数，检查 Python 的长整型对象是否在给定类型的范围内。
 * 返回 -1 表示出错。
 */
static inline int
get_value_range(PyObject *value, int type_num, int *range)
{
    long long min;
    unsigned long long max;
    get_min_max(type_num, &min, &max);  // 获取指定类型的最小值和最大值

    int overflow;
    long long val = PyLong_AsLongLongAndOverflow(value, &overflow);  // 将 Python 长整型转换为 long long
    if (val == -1 && overflow == 0 && PyErr_Occurred()) {
        return -1;  // 如果转换出错，返回 -1
    }
    // 如果转换成功，检查值是否在指定类型的范围内

    return 0;  // 返回表示成功执行
}
    // 如果溢出标志为0
    if (overflow == 0) {
        // 如果值小于最小值，设置范围为-1
        if (val < min) {
            *range = -1;
        }
        // 如果值大于0且大于最大值，设置范围为1
        else if (val > 0 && (unsigned long long)val > max) {
            *range = 1;
        }
        // 否则，设置范围为0
        else {
            *range = 0;
        }
    }
    // 如果溢出标志为负数，设置范围为-1
    else if (overflow < 0) {
        *range = -1;
    }
    // 如果最大值小于等于最大长长整数值
    else if (max <= NPY_MAX_LONGLONG) {
        // 设置范围为1
        *range = 1;
    }
    // 否则，处理大于长长整数的情况
    else {
        /*
        * 如果我们正在检查无符号长长整数，则值可能大于长长整数，
        * 但在无符号长长整数的范围内。通过正常的Python整数比较来检查这一点。
        */
        // 创建一个无符号长长整数的Python对象
        PyObject *obj = PyLong_FromUnsignedLongLong(max);
        // 如果对象创建失败，返回-1
        if (obj == NULL) {
            return -1;
        }
        // 进行Python对象的大于比较
        int cmp = PyObject_RichCompareBool(value, obj, Py_GT);
        Py_DECREF(obj);
        // 如果比较出错，返回-1
        if (cmp < 0) {
            return -1;
        }
        // 根据比较结果设置范围：1表示大于，0表示否
        if (cmp) {
            *range = 1;
        }
        else {
            *range = 0;
        }
    }
    // 返回处理结果，成功返回0
    return 0;
/*
 * Find the type resolution for any numpy_int with pyint comparison.  This
 * function supports *both* directions for all types.
 */
static NPY_CASTING
resolve_descriptors_with_scalars(
    PyArrayMethodObject *self, PyArray_DTypeMeta **dtypes,
    PyArray_Descr **given_descrs, PyObject *const *input_scalars,
    PyArray_Descr **loop_descrs, npy_intp *view_offset)
{
    // 初始化变量 value_range 为 0
    int value_range = 0;

    // 检查第一个 dtypes 是否为 PyArray_PyLongDType 类型
    npy_bool first_is_pyint = dtypes[0] == &PyArray_PyLongDType;
    // 根据 first_is_pyint 的值选择数组和标量的索引
    int arr_idx = first_is_pyint ? 1 : 0;
    int scalar_idx = first_is_pyint ? 0 : 1;
    // 获取输入的标量
    PyObject *scalar = input_scalars[scalar_idx];
    // 断言数组的类型是整数类型
    assert(PyTypeNum_ISINTEGER(dtypes[arr_idx]->type_num));
    // 获取数组的数据类型元数据
    PyArray_DTypeMeta *arr_dtype = dtypes[arr_idx];

    /*
     * Three way decision (with hack) on value range:
     *  0: The value fits within the range of the dtype.
     *  1: The value came second and is larger or came first and is smaller.
     * -1: The value came second and is smaller or came first and is larger
     */
    // 如果标量不为空且是精确的长整型
    if (scalar != NULL && PyLong_CheckExact(scalar)) {
        // 获取标量值的范围，并根据结果调整 value_range
        if (get_value_range(scalar, arr_dtype->type_num, &value_range) < 0) {
            return _NPY_ERROR_OCCURRED_IN_CAST;
        }
        // 如果 first_is_pyint 为真，则将 value_range 取反
        if (first_is_pyint == 1) {
            value_range *= -1;
        }
    }

    /*
     * Very small/large values always need to be encoded as `object` dtype
     * in order to never fail casting (NumPy will store the Python integer
     * in a 0-D object array this way -- even if we never inspect it).
     *
     * TRICK: We encode the value range by whether or not we use the object
     *        singleton!  This information is then available in `get_loop()`
     *        to pick a loop that returns always True or False.
     */
    // 根据 value_range 的值选择相应的描述符类型
    if (value_range == 0) {
        // 对于范围为 0 的情况，使用数组数据类型的 singleton
        Py_INCREF(arr_dtype->singleton);
        loop_descrs[scalar_idx] = arr_dtype->singleton;
    }
    else if (value_range < 0) {
        // 对于范围小于 0 的情况，使用 NPY_OBJECT 类型的描述符
        loop_descrs[scalar_idx] = PyArray_DescrFromType(NPY_OBJECT);
    }
    else {
        // 对于范围大于 0 的情况，创建一个新的 NPY_OBJECT 类型的描述符
        loop_descrs[scalar_idx] = PyArray_DescrNewFromType(NPY_OBJECT);
        if (loop_descrs[scalar_idx] == NULL) {
            return _NPY_ERROR_OCCURRED_IN_CAST;
        }
    }
    // 增加数组数据类型的 singleton 的引用计数
    Py_INCREF(arr_dtype->singleton);
    // 将数组数据类型的 singleton 设置为数组的描述符
    loop_descrs[arr_idx] = arr_dtype->singleton;
    // 将 NP_BOOL 类型的描述符设置为第三个描述符
    loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);

    // 返回无需转换的状态
    return NPY_NO_CASTING;
}

/*
 * Template function to retrieve a loop based on the comparison type.
 */
template<COMP comp>
static int
get_loop(PyArrayMethod_Context *context,
        int aligned, int move_references, const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 如果第一个描述符的类型与第二个相同，则回退到旧的循环实现
    if (context->descriptors[1]->type_num == context->descriptors[0]->type_num) {
        /*
         * Fall back to the current implementation, which wraps legacy loops.
         */
        return get_wrapped_legacy_ufunc_loop(
                context, aligned, move_references, strides,
                out_loop, out_transferdata, flags);
    }
}
    // 否则情况的处理分支，开始
    else {
        // 定义指向另一个描述符的指针
        PyArray_Descr *other_descr;
        // 如果第二个描述符的类型是 NPY_OBJECT，则将其赋给 other_descr
        if (context->descriptors[1]->type_num == NPY_OBJECT) {
            other_descr = context->descriptors[1];
        }
        // 否则，假设第一个描述符的类型是 NPY_OBJECT
        else {
            assert(context->descriptors[0]->type_num == NPY_OBJECT);
            other_descr = context->descriptors[0];
        }
        // HACK: 如果描述符是单例，则结果较小
        // 创建一个指向 NPY_OBJECT 类型的单例描述符对象
        PyArray_Descr *obj_singleton = PyArray_DescrFromType(NPY_OBJECT);
        // 如果 other_descr 和 obj_singleton 相等
        if (other_descr == obj_singleton) {
            // 根据比较操作类型设置输出循环的指针
            switch (comp) {
                case COMP::EQ:
                case COMP::LT:
                case COMP::LE:
                    *out_loop = &fixed_result_loop<false>;
                    break;
                case COMP::NE:
                case COMP::GT:
                case COMP::GE:
                    *out_loop = &fixed_result_loop<true>;
                    break;
            }
        }
        // 否则，other_descr 和 obj_singleton 不相等
        else {
            // 根据比较操作类型设置输出循环的指针
            switch (comp) {
                case COMP::EQ:
                case COMP::GT:
                case COMP::GE:
                    *out_loop = &fixed_result_loop<false>;
                    break;
                case COMP::NE:
                case COMP::LT:
                case COMP::LE:
                    *out_loop = &fixed_result_loop<true>;
                    break;
            }
        }
        // 释放单例描述符对象的引用
        Py_DECREF(obj_singleton);
    }
    // 设置标志位，指示无浮点错误
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    // 返回 0 表示成功
    return 0;
/*
 * 结束函数，返回成功状态码
 */
}


/*
 * 用于将 Python 整数添加到 NumPy 整数比较中的机制，
 * 以及用于特殊情况下的 Python 整数与 Python 整数比较的特殊推广。
 */

/*
 * 简单的推广器，确保在输入仅为 Python 整数时使用对象循环。
 * 注意，如果用户显式传递 Python `int` 抽象的 DType，则承诺实际传递的是 Python 整数，
 * 我们接受这一点，并不做检查。
 */
static int
pyint_comparison_promoter(PyUFuncObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    new_op_dtypes[0] = NPY_DT_NewRef(&PyArray_ObjectDType);  // 第一个操作数类型设为对象类型
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_ObjectDType);  // 第二个操作数类型设为对象类型
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_BoolDType);    // 结果类型设为布尔类型
    return 0;  // 返回成功状态码
}


/*
 * 此函数使用传入的循环替换步进循环，并将其注册到给定的 ufunc 中。
 * 它还为 (pyint, pyint, bool) 添加推广器，以使用 (object, object, bool) 实现。
 */
template<COMP comp>
static int
add_dtype_loops(PyObject *umath, PyArrayMethod_Spec *spec, PyObject *info)
{
    PyArray_DTypeMeta *PyInt = &PyArray_PyLongDType;  // Python 整数类型

    PyObject *name = PyUnicode_FromString(comp_name(comp));
    if (name == nullptr) {  // 如果创建名称对象失败
        return -1;  // 返回错误状态码
    }
    PyUFuncObject *ufunc = (PyUFuncObject *)PyObject_GetItem(umath, name);  // 获取 ufunc 对象
    Py_DECREF(name);  // 释放名称对象的引用计数
    if (ufunc == nullptr) {  // 如果获取 ufunc 对象失败
        return -1;  // 返回错误状态码
    }
    if (Py_TYPE(ufunc) != &PyUFunc_Type) {  // 如果 ufunc 对象类型不是 PyUFunc_Type
        PyErr_SetString(PyExc_RuntimeError,
                "internal NumPy error: comparison not a ufunc");  // 设置错误信息
        goto fail;  // 跳转到失败处理代码块
    }

    /* 
     * 注意：迭代所有类型号码，希望减少此次迭代。
     *       （如果我们总体上统一 int DTypes 将更容易。）
     */
    for (int typenum = NPY_BYTE; typenum <= NPY_ULONGLONG; typenum++) {
        spec->slots[0].pfunc = (void *)get_loop<comp>;  // 设置特定比较操作的循环函数

        PyArray_DTypeMeta *Int = PyArray_DTypeFromTypeNum(typenum);  // 根据类型号码获取类型对象

        /* 注册正向和反向方向的 spec/loop */
        spec->dtypes[0] = Int;     // 第一个操作数类型设为当前类型对象
        spec->dtypes[1] = PyInt;   // 第二个操作数类型设为 Python 整数类型
        int res = PyUFunc_AddLoopFromSpec_int((PyObject *)ufunc, spec, 1);  // 添加 spec/loop 到 ufunc
        if (res < 0) {
            Py_DECREF(Int);  // 减少类型对象的引用计数
            goto fail;  // 跳转到失败处理代码块
        }
        spec->dtypes[0] = PyInt;   // 第一个操作数类型设为 Python 整数类型
        spec->dtypes[1] = Int;     // 第二个操作数类型设为当前类型对象
        res = PyUFunc_AddLoopFromSpec_int((PyObject *)ufunc, spec, 1);  // 添加 spec/loop 到 ufunc
        Py_DECREF(Int);  // 减少类型对象的引用计数
        if (res < 0) {
            goto fail;  // 跳转到失败处理代码块
        }
    }

    /*
     * 安装推广信息以允许两个 Python 整数进行比较。
     */
    return PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);  // 添加循环到 ufunc 并返回成功状态码

    Py_DECREF(ufunc);  // 减少 ufunc 对象的引用计数
    return 0;  // 返回成功状态码

  fail:
    Py_DECREF(ufunc);  // 减少 ufunc 对象的引用计数
    return -1;  // 返回错误状态码
}


template<COMP...>
struct add_loops;

template<>
struct add_loops<> {
    int operator()(PyObject*, PyArrayMethod_Spec*, PyObject *) {
        return 0;  // 返回成功状态码
    }
};
/*
 * 定义模板结构体 `add_loops`，接受多个比较器作为模板参数，并重载函数调用操作符
 */
struct add_loops<comp, comps...> {
    /*
     * 重载函数调用操作符，接受三个指针参数，并返回整数结果
     */
    int operator()(PyObject* umath, PyArrayMethod_Spec* spec, PyObject *info) {
        // 如果调用 `add_dtype_loops` 失败，返回 -1
        if (add_dtype_loops<comp>(umath, spec, info) < 0) {
            return -1;
        }
        else {
            // 递归调用 `add_loops` 结构体的实例
            return add_loops<comps...>()(umath, spec, info);
        }
    }
};

/*
 * 初始化特殊整数比较操作
 */
NPY_NO_EXPORT int
init_special_int_comparisons(PyObject *umath)
{
    int res = -1;
    PyObject *info = NULL, *promoter = NULL;
    PyArray_DTypeMeta *Bool = &PyArray_BoolDType;

    /* 所有循环具有布尔输出 DType（其它类型稍后填充） */
    PyArray_DTypeMeta *dtypes[] = {NULL, NULL, Bool};
    /*
     * 目前我们只有一个循环，即步进循环。默认类型解析器确保原生字节顺序/规范表示。
     */
    PyType_Slot slots[] = {
        {NPY_METH_get_loop, nullptr},
        {_NPY_METH_resolve_descriptors_with_scalars,
             (void *)&resolve_descriptors_with_scalars},
        {0, NULL},
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_pyint_to_integers_comparisons";
    spec.nin = 2;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /*
     * 以下设置正确的促进器，以使类似 `np.equal(2, 4)`（使用两个 Python 整数）的比较使用对象循环。
     */
    PyObject *dtype_tuple = PyTuple_Pack(3,
            &PyArray_PyLongDType, &PyArray_PyLongDType, Bool);
    if (dtype_tuple == NULL) {
        goto finish;
    }
    promoter = PyCapsule_New(
            (void *)&pyint_comparison_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter == NULL) {
        Py_DECREF(dtype_tuple);
        goto finish;
    }
    info = PyTuple_Pack(2, dtype_tuple, promoter);
    Py_DECREF(dtype_tuple);
    Py_DECREF(promoter);
    if (info == NULL) {
        goto finish;
    }

    /* 添加所有 PyInt 和 NumPy 整数比较的组合 */
    using comp_looper = add_loops<COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    if (comp_looper()(umath, &spec, info) < 0) {
        goto finish;
    }

    res = 0;
  finish:

    Py_XDECREF(info);
    return res;
}
```