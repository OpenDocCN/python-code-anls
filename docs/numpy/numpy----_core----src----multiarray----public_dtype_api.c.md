# `.\numpy\numpy\_core\src\multiarray\public_dtype_api.c`

```
/*NUMPY_API
 *
 * Initialize a new DType.  It must currently be a static Python C type that
 * is declared as `PyArray_DTypeMeta` and not `PyTypeObject`.  Further, it
 * must subclass `np.dtype` and set its type to `PyArrayDTypeMeta_Type`
 * (before calling `PyType_Ready()`). The DTypeMeta object has additional
 * fields compared to a normal PyTypeObject!
 */
int
PyArrayInitDTypeMeta_FromSpec(
        PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec)
{
    /* 检查传入的 DType 是否是有效的 DTypeMeta 实例 */
    if (!PyObject_TypeCheck(DType, &PyArrayDTypeMeta_Type)) {
        PyErr_SetString(PyExc_RuntimeError,
                "Passed in DType must be a valid (initialized) DTypeMeta "
                "instance!");
        return -1;
    }

    /* 检查自定义 DType 是否实现了 `__repr__` 和 `__str__` 方法 */
    if (((PyTypeObject *)DType)->tp_repr == PyArrayDescr_Type.tp_repr
            || ((PyTypeObject *)DType)->tp_str == PyArrayDescr_Type.tp_str) {
        PyErr_SetString(PyExc_TypeError,
                "A custom DType must implement `__repr__` and `__str__` since "
                "the default inherited version (currently) fails.");
        return -1;
    }

    /* 检查 spec->typeobj 是否为有效的类型对象 */
    if (spec->typeobj == NULL || !PyType_Check(spec->typeobj)) {
        PyErr_SetString(PyExc_TypeError,
                "Not giving a type object is currently not supported, but "
                "is expected to be supported eventually.  This would mean "
                "that e.g. indexing a NumPy array will return a 0-D array "
                "and not a scalar.");
        return -1;
    }

    /* 检查和处理 flags */
    int allowed_flags = NPY_DT_PARAMETRIC | NPY_DT_ABSTRACT | NPY_DT_NUMERIC;
    if (spec->flags & ~(allowed_flags)) {
        PyErr_SetString(PyExc_RuntimeError,
                "invalid DType flags specified, only NPY_DT_PARAMETRIC, "
                "NPY_DT_ABSTRACT, and NPY_DT_NUMERIC are valid flags for "
                "user DTypes.");
        return -1;
    }

    /* 检查 spec->casts 是否为 NULL */
    if (spec->casts == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "DType must at least provide a function to cast (or just copy) "
            "between its own instances!");
        return -1;
    }

    /* 初始化 DTypeMeta 结构体 */
    dtypemeta_initialize_struct_from_spec(DType, spec, 0);

    /* 检查 DType 是否提供了 setitem 和 getitem 方法 */
    if (NPY_DT_SLOTS(DType)->setitem == NULL
            || NPY_DT_SLOTS(DType)->getitem == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "A DType must provide a getitem/setitem (there may be an "
                "exception here in the future if no scalar type is provided)");
        return -1;
    }
    # 如果 ensure_canonical 函数指针为空，说明 DType 没有提供 ensure_canonical 的实现
    if (NPY_DT_SLOTS(DType)->ensure_canonical == NULL) {
        # 抛出运行时错误，说明一个 DType 必须提供 ensure_canonical 的实现
        PyErr_SetString(PyExc_RuntimeError,
                        "A DType must provide an ensure_canonical implementation.");
        # 返回 -1 表示错误退出
        return -1;
    }

    """
     * 现在规范已经读取，我们可以检查用户是否定义了所有必需的函数。
     """
    # 如果 spec 的标志包含 NPY_DT_PARAMETRIC
    if (spec->flags & NPY_DT_PARAMETRIC) {
        # 如果 common_instance 函数指针为空，或者 discover_descr_from_pyobject 指针指向默认函数
        if (NPY_DT_SLOTS(DType)->common_instance == NULL ||
                NPY_DT_SLOTS(DType)->discover_descr_from_pyobject
                        == &dtypemeta_discover_as_default) {
            # 抛出运行时错误，说明 parametric DType 必须定义 common-instance 和 descriptor discovery 函数
            PyErr_SetString(PyExc_RuntimeError,
                    "Parametric DType must define a common-instance and "
                    "descriptor discovery function!");
            # 返回 -1 表示错误退出
            return -1;
        }
    }

    # 如果 within_dtype_castingimpl 函数指针为空
    if (NPY_DT_SLOTS(DType)->within_dtype_castingimpl == NULL) {
        """
         * 我们暂时期望这样。对于只支持简单复制（可能还涉及字节顺序）的 DType，我们应该有一个默认值。
         """
        # 抛出运行时错误，说明 DType 必须提供一个用于在其实例之间转换（或仅复制）的函数
        PyErr_SetString(PyExc_RuntimeError,
                "DType must provide a function to cast (or just copy) between "
                "its own instances!");
        # 返回 -1 表示错误退出
        return -1;
    }

    """
     * 最后，我们必须注册所有的类型转换！
     """
    # 返回 0 表示成功退出，没有错误
    return 0;
# 填充数据类型 API 表
void
_fill_dtype_api(void *full_api_table[])
{
    # 将 API 表的起始位置移动到第 320 个位置后
    void **api_table = full_api_table + 320;

    /* The type of the DType metaclass */
    # DType 元类的类型
    api_table[0] = &PyArrayDTypeMeta_Type;
    /* Boolean */
    # 布尔类型
    api_table[1] = &PyArray_BoolDType;
    /* Integers */
    # 整数类型
    api_table[2] = &PyArray_ByteDType;
    api_table[3] = &PyArray_UByteDType;
    api_table[4] = &PyArray_ShortDType;
    api_table[5] = &PyArray_UShortDType;
    api_table[6] = &PyArray_IntDType;
    api_table[7] = &PyArray_UIntDType;
    api_table[8] = &PyArray_LongDType;
    api_table[9] = &PyArray_ULongDType;
    api_table[10] = &PyArray_LongLongDType;
    api_table[11] = &PyArray_ULongLongDType;
    /* Integer aliases */
    # 整数别名
    api_table[12] = &PyArray_Int8DType;
    api_table[13] = &PyArray_UInt8DType;
    api_table[14] = &PyArray_Int16DType;
    api_table[15] = &PyArray_UInt16DType;
    api_table[16] = &PyArray_Int32DType;
    api_table[17] = &PyArray_UInt32DType;
    api_table[18] = &PyArray_Int64DType;
    api_table[19] = &PyArray_UInt64DType;
    api_table[20] = &PyArray_IntpDType;
    api_table[21] = &PyArray_UIntpDType;
    /* Floats */
    # 浮点数类型
    api_table[22] = &PyArray_HalfDType;
    api_table[23] = &PyArray_FloatDType;
    api_table[24] = &PyArray_DoubleDType;
    api_table[25] = &PyArray_LongDoubleDType;
    /* Complex */
    # 复数类型
    api_table[26] = &PyArray_CFloatDType;
    api_table[27] = &PyArray_CDoubleDType;
    api_table[28] = &PyArray_CLongDoubleDType;
    /* String/Bytes */
    # 字符串/字节类型
    api_table[29] = &PyArray_BytesDType;
    api_table[30] = &PyArray_UnicodeDType;
    /* Datetime/Timedelta */
    # 日期时间/时间增量类型
    api_table[31] = &PyArray_DatetimeDType;
    api_table[32] = &PyArray_TimedeltaDType;
    /* Object and Structured */
    # 对象和结构化类型
    api_table[33] = &PyArray_ObjectDType;
    api_table[34] = &PyArray_VoidDType;
    /* Abstract */
    # 抽象类型
    api_table[35] = &PyArray_PyLongDType;
    api_table[36] = &PyArray_PyFloatDType;
    api_table[37] = &PyArray_PyComplexDType;
    api_table[38] = &PyArray_DefaultIntDType;
    /* Non-legacy DTypes that are built in to NumPy */
    # 内置于 NumPy 中的非旧版数据类型
    api_table[39] = &PyArray_StringDType;

    /* Abstract ones added directly: */
    # 直接添加的抽象类型
    full_api_table[366] = &PyArray_IntAbstractDType;
    full_api_table[367] = &PyArray_FloatAbstractDType;
    full_api_table[368] = &PyArray_ComplexAbstractDType;
}
```