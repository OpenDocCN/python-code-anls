# `.\numpy\numpy\_core\src\multiarray\legacy_dtype_implementation.c`

```
/*
 * 此部分代码定义了一些用于比较和判断旧版数据类型的函数和宏。
 * 这些函数和宏主要用于处理旧版数据类型，可能在将来被弃用和移除。
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "numpy/arrayobject.h"
#include "scalartypes.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "can_cast_table.h"
#include "convert_datatype.h"
#include "dtypemeta.h"

#include "legacy_dtype_implementation.h"

/*
 * 比较两个数据类型的字段字典是否相等。
 * 如果两个描述符的字段类型和字段名称相等且顺序相同，则返回1，否则返回0。
 */
static int
_equivalent_fields(_PyArray_LegacyDescr *type1, _PyArray_LegacyDescr *type2) {

    int val;

    if (type1->fields == type2->fields && type1->names == type2->names) {
        return 1;
    }
    if (type1->fields == NULL || type2->fields == NULL) {
        return 0;
    }

    val = PyObject_RichCompareBool(type1->fields, type2->fields, Py_EQ);
    if (val != 1 || PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }

    val = PyObject_RichCompareBool(type1->names, type2->names, Py_EQ);
    if (val != 1 || PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }

    return 1;
}

/*
 * 比较两个数据类型的子数组数据是否相等。
 * 如果相等则返回1，否则返回0。
 */
static int
_equivalent_subarrays(PyArray_ArrayDescr *sub1, PyArray_ArrayDescr *sub2)
{
    int val;

    if (sub1 == sub2) {
        return 1;
    }
    if (sub1 == NULL || sub2 == NULL) {
        return 0;
    }

    val = PyObject_RichCompareBool(sub1->shape, sub2->shape, Py_EQ);
    if (val != 1 || PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }

    return PyArray_EquivTypes(sub1->base, sub2->base);
}

/*
 * 判断两个数据描述符是否等价。
 * 如果是旧版数据类型且各种条件满足则返回NPY_TRUE，否则返回NPY_FALSE。
 */
static unsigned char
PyArray_LegacyEquivTypes(PyArray_Descr *type1, PyArray_Descr *type2)
{
    int type_num1, type_num2, size1, size2;

    if (type1 == type2) {
        return NPY_TRUE;
    }
    if (!PyDataType_ISLEGACY(type1) || !PyDataType_ISLEGACY(type2)) {
        return NPY_FALSE;
    }

    type_num1 = type1->type_num;
    type_num2 = type2->type_num;
    size1 = type1->elsize;
    size2 = type2->elsize;

    if (size1 != size2) {
        return NPY_FALSE;
    }
    if (PyArray_ISNBO(type1->byteorder) != PyArray_ISNBO(type2->byteorder)) {
        return NPY_FALSE;
    }
    if (PyDataType_SUBARRAY(type1) || PyDataType_SUBARRAY(type2)) {
        return ((type_num1 == type_num2)
                && _equivalent_subarrays(PyDataType_SUBARRAY(type1), PyDataType_SUBARRAY(type2)));
    }
    if (type_num1 == NPY_VOID || type_num2 == NPY_VOID) {
        return ((type_num1 == type_num2) && _equivalent_fields(
                    (_PyArray_LegacyDescr *)type1, (_PyArray_LegacyDescr *)type2));
    }
    # 如果 type_num1 或 type_num2 是 NPY_DATETIME 或 NPY_TIMEDELTA 类型之一，则执行以下操作
    if (type_num1 == NPY_DATETIME
        || type_num1 == NPY_TIMEDELTA
        || type_num2 == NPY_DATETIME
        || type_num2 == NPY_TIMEDELTA) {
        # 如果 type_num1 和 type_num2 的类型相同，并且具有等效的日期时间元数据，则返回真
        return ((type_num1 == type_num2)
                && has_equivalent_datetime_metadata(type1, type2));
    }
    
    # 如果不满足上述条件，则比较 type1 和 type2 的 kind 属性是否相同，然后返回结果
    return type1->kind == type2->kind;
/*
 * 比较两个类型描述符的等价性。
 *
 * 如果 typenum1 和 typenum2 相等，则返回 NPY_SUCCEED；否则返回 NPY_FAIL。
 */
static unsigned char
PyArray_LegacyEquivTypenums(int typenum1, int typenum2)
{
    PyArray_Descr *d1, *d2;
    npy_bool ret;

    // 检查 typenum1 和 typenum2 是否相等
    if (typenum1 == typenum2) {
        return NPY_SUCCEED;
    }

    // 获取 typenum1 和 typenum2 对应的类型描述符
    d1 = PyArray_DescrFromType(typenum1);
    d2 = PyArray_DescrFromType(typenum2);

    // 判断两个类型描述符是否等价
    ret = PyArray_LegacyEquivTypes(d1, d2);

    // 释放类型描述符的引用
    Py_DECREF(d1);
    Py_DECREF(d2);

    return ret;
}


/*
 * 检查是否可以安全地从一个类型转换到另一个类型。
 *
 * 如果 fromtype 和 totype 相等，则返回 1；否则根据类型转换表格判断是否可以安全转换。
 * 如果可以安全转换，则返回 1；否则返回 0。
 */
static int
PyArray_LegacyCanCastSafely(int fromtype, int totype)
{
    PyArray_Descr *from;

    /* 快速查找小类型编号的表格 */
    if ((unsigned int)fromtype < NPY_NTYPES_LEGACY &&
        (unsigned int)totype < NPY_NTYPES_LEGACY) {
        return _npy_can_cast_safely_table[fromtype][totype];
    }

    /* 判断是否同一类型 */
    if (fromtype == totype) {
        return 1;
    }

    // 获取 fromtype 对应的类型描述符
    from = PyArray_DescrFromType(fromtype);

    /*
     * cancastto 是一个以 NPY_NOTYPE 结尾的 C 整型数组，表示该数据类型可以安全转换的目标类型。
     * 遍历数组，查找是否存在可以安全转换到 totype 的目标类型。
     */
    if (PyDataType_GetArrFuncs(from)->cancastto) {
        int *curtype = PyDataType_GetArrFuncs(from)->cancastto;

        while (*curtype != NPY_NOTYPE) {
            if (*curtype++ == totype) {
                Py_DECREF(from);
                return 1;
            }
        }
    }

    // 释放类型描述符的引用
    Py_DECREF(from);
    return 0;
}


/*
 * 检查是否可以安全地将一个类型描述符转换为另一个类型描述符。
 *
 * 根据 from 和 to 的类型描述符，判断是否可以安全转换。
 * 如果可以安全转换，则返回 1；否则返回 0。
 */
static npy_bool
PyArray_LegacyCanCastTo(PyArray_Descr *from, PyArray_Descr *to)
{
    int from_type_num = from->type_num;
    int to_type_num = to->type_num;
    npy_bool ret;

    // 调用 PyArray_LegacyCanCastSafely 函数判断是否可以安全转换
    ret = (npy_bool) PyArray_LegacyCanCastSafely(from_type_num, to_type_num);

    return ret;
}


/*
 * 比较两个字段字典是否可以安全地进行类型转换。
 *
 * 如果 field1 和 field2 相同，则返回 1；
 * 如果 field1 或 field2 为空，则返回 0；
 * 如果 field1 和 field2 中的字段数量不同，则返回 0；
 * 遍历 field1 和 field2 中的每个字段，并比较其类型是否可以安全转换。
 * 如果所有字段都可以安全转换，则返回 1；否则返回 0。
 */
static int
can_cast_fields(PyObject *field1, PyObject *field2, NPY_CASTING casting)
{
    Py_ssize_t ppos;
    PyObject *key;
    PyObject *tuple1, *tuple2;

    // 检查 field1 和 field2 是否是同一个对象
    if (field1 == field2) {
        return 1;
    }

    // 检查 field1 和 field2 是否为空
    if (field1 == NULL || field2 == NULL) {
        return 0;
    }

    // 检查 field1 和 field2 的字段数量是否相同
    if (PyDict_Size(field1) != PyDict_Size(field2)) {
        return 0;
    }

    /* 迭代比较所有字段并检查是否可以安全转换 */
    ppos = 0;
    while (PyDict_Next(field1, &ppos, &key, &tuple1)) {
        // 获取 field2 中与 key 对应的值
        if ((tuple2 = PyDict_GetItem(field2, key)) == NULL) {
            return 0;
        }
        /* 比较字段的 dtype 是否可以安全转换 */
        if (!PyArray_CanCastTypeTo(
                        (PyArray_Descr *)PyTuple_GET_ITEM(tuple1, 0),
                        (PyArray_Descr *)PyTuple_GET_ITEM(tuple2, 0),
                        casting)) {
            return 0;
        }
    }

    return 1;
}


/*
 * 检查是否可以安全地将一个类型描述符转换为另一个类型描述符。
 *
 * 根据 from 和 to 的类型描述符，判断是否可以安全转换。
 * 如果可以安全转换，则返回 NPY_SUCCEED；否则返回 NPY_FAIL。
 */
NPY_NO_EXPORT npy_bool
PyArray_LegacyCanCastTypeTo(PyArray_Descr *from, PyArray_Descr *to,
        NPY_CASTING casting)
{
    // 将 from 和 to 强制转换为 _PyArray_LegacyDescr 结构体指针
    _PyArray_LegacyDescr *lfrom = (_PyArray_LegacyDescr *)from;
    _PyArray_LegacyDescr *lto = (_PyArray_LegacyDescr *)to;

    /*
     * 快速路径：处理相等性和基本类型的情况。
     */
    if (from == to ||
        ((NPY_LIKELY(PyDataType_ISNUMBER(from)) ||
          PyDataType_ISOBJECT(from)) &&
         NPY_LIKELY(from->type_num == to->type_num) &&
         NPY_LIKELY(from->byteorder == to->byteorder))) {
        return 1;
    }
    // 如果 from 或 to 不是遗留数据类型，则返回不可转换
    if (!PyDataType_ISLEGACY(from) || !PyDataType_ISLEGACY(to)) {
        return 0;
    }
    /*
     * 处理包含子数组和字段的情况需要特殊处理。
     */
    if (PyDataType_HASFIELDS(from)) {
        /*
         * 如果 from 是结构化数据类型，则只能在不安全转换的情况下将其转换为非对象类型，
         * 前提是它只有一个字段；递归处理，以防单个字段本身是结构化的。
         */
        if (!PyDataType_HASFIELDS(to) && !PyDataType_ISOBJECT(to)) {
            if (casting == NPY_UNSAFE_CASTING &&
                    PyDict_Size(lfrom->fields) == 1) {
                Py_ssize_t ppos = 0;
                PyObject *tuple;
                PyArray_Descr *field;
                PyDict_Next(lfrom->fields, &ppos, NULL, &tuple);
                field = (PyArray_Descr *)PyTuple_GET_ITEM(tuple, 0);
                /*
                 * 对于子数组，我们需要获取其基础类型；
                 * 由于我们已经在进行不安全转换，可以忽略其形状。
                 */
                if (PyDataType_HASSUBARRAY(field)) {
                    field = PyDataType_SUBARRAY(field)->base;
                }
                return PyArray_LegacyCanCastTypeTo(field, to, casting);
            }
            else {
                return 0;
            }
        }
        /*
         * 从一个结构化数据类型到另一个的转换依赖于字段；
         * 我们将此情况传递给下面的 EquivTypenums 情况处理。
         *
         * TODO: 将上面的部分移到这里？需要检查等价类型编号是否是必需的附加约束。
         *
         * TODO/FIXME: 目前，对于不安全转换，始终允许结构化到结构化的转换；
         * 这是不正确的，但因为 can_cast 下面的处理与 astype 不同步，所以需要；参见 gh-13667。
         */
        if (casting == NPY_UNSAFE_CASTING) {
            return 1;
        }
    }
    else if (PyDataType_HASFIELDS(to)) {
        /*
         * 如果 "from" 是简单数据类型而 "to" 具有字段，则仅不安全转换有效
         * （即使对多个字段也是如此）。
         */
        return casting == NPY_UNSAFE_CASTING;
    }
    /*
     * 对于其他情况，我们暂时认为可以进行不安全转换。
     * FIXME: 确保这里的操作与 "astype" 一致，
     * 即更正确地处理子数组和用户定义的数据类型。
     */
    else if (casting == NPY_UNSAFE_CASTING) {
        // 如果 casting 等于 NPY_UNSAFE_CASTING，则返回 1，表示允许不安全的类型转换
        return 1;
    }
    /*
     * Equivalent simple types can be cast with any value of 'casting', but
     * we need to be careful about structured to structured.
     */
    }
    // 如果允许安全或者同种类的类型转换
    else if (casting == NPY_SAFE_CASTING || casting == NPY_SAME_KIND_CASTING) {
        // 检查是否可以使用旧的 PyArray_LegacyCanCastTo 函数进行转换
        if (PyArray_LegacyCanCastTo(from, to)) {
            return 1; // 可以转换则返回 1
        }
        else if(casting == NPY_SAME_KIND_CASTING) {
            /*
             * Also allow casting from lower to higher kinds, according
             * to the ordering provided by dtype_kind_to_ordering.
             * Some kinds, like datetime, don't fit in the hierarchy,
             * and are special cased as -1.
             */
            int from_order, to_order;

            // 获取源类型和目标类型的类型顺序
            from_order = dtype_kind_to_ordering(from->kind);
            to_order = dtype_kind_to_ordering(to->kind);

            if (to->kind == 'm') {
                // 对于 'm' 类型 (timedelta)，直接返回是否源类型在整数类型之前
                int integer_order = dtype_kind_to_ordering('i');
                return (from_order != -1) && (from_order <= integer_order);
            }

            // 一般情况下返回是否源类型顺序在目标类型顺序之前
            return (from_order != -1) && (from_order <= to_order);
        }
        else {
            return 0; // 其他情况返回 0，表示不允许类型转换
        }
    }
    // 如果 casting 设置为 NPY_NO_CASTING 或者 NPY_EQUIV_CASTING
    else {
        return 0; // 返回 0，表示不允许类型转换
    }
}



# 这行代码结束了一个 Python 函数的定义，闭合了函数的代码块
```