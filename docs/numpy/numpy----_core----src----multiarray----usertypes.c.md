# `.\numpy\numpy\_core\src\multiarray\usertypes.c`

```py
/*
  提供多维数组作为 Python 中的基本对象类型。

  基于原始的 Numeric 实现
  版权所有 (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  同时也包含了许多 Numeric Python 开发者在 1995-2004 年间的贡献

  2005 年大幅修改，受 Numarray 启发

  作者：Travis Oliphant, oliphant@ee.byu.edu
  所属：Brigham Young University

  维护者邮箱：oliphant.travis@ieee.org

  Numarray 的设计思路（提供了指导）来自于
  Space Science Telescope Institute
  (J. Todd Miller, Perry Greenfield, Rick White)
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "common.h"

#include "usertypes.h"
#include "dtypemeta.h"
#include "scalartypes.h"
#include "array_method.h"
#include "convert_datatype.h"
#include "dtype_traversal.h"
#include "legacy_dtype_implementation.h"


/* 
   NPY_NO_EXPORT 表示此函数不会导出到公共 API
   userdescrs 用于保存用户定义的数据描述符
*/
NPY_NO_EXPORT _PyArray_LegacyDescr **userdescrs = NULL;

/*
   功能：向整数数组添加新元素
   参数：p_types - 指向整数数组的指针
         insert - 要插入的整数值
   返回值：成功返回 0，内存分配失败返回 -1
*/
static int
_append_new(int **p_types, int insert)
{
    int n = 0;
    int *newtypes;
    int *types = *p_types;

    // 找到数组中的最后一个元素
    while (types[n] != NPY_NOTYPE) {
        n++;
    }
    // 重新分配内存，以添加新元素
    newtypes = (int *)realloc(types, (n + 2)*sizeof(int));
    if (newtypes == NULL) {
        PyErr_NoMemory(); // 内存分配失败，抛出内存错误异常
        return -1;
    }
    newtypes[n] = insert; // 在数组末尾添加新元素
    newtypes[n + 1] = NPY_NOTYPE;

    /* 替换传入的指针 */
    *p_types = newtypes;
    return 0;
}

/*
   默认的非零判断函数
   功能：检查数组中的元素是否存在非零值
   参数：ip - 数组元素的指针
         arr - 数组对象的指针
   返回值：存在非零元素返回 NPY_TRUE，否则返回 NPY_FALSE
*/
static npy_bool
_default_nonzero(void *ip, void *arr)
{
    int elsize = PyArray_ITEMSIZE(arr); // 获取数组元素的大小
    char *ptr = ip;
    while (elsize--) {
        if (*ptr++ != 0) { // 检查元素是否为非零
            return NPY_TRUE;
        }
    }
    return NPY_FALSE;
}

/*
   默认的复制和交换函数
   功能：将源数组的部分内容复制到目标数组，支持交换字节序
   参数：dst - 目标数组的起始地址
         dstride - 目标数组中相邻元素之间的字节偏移量
         src - 源数组的起始地址
         sstride - 源数组中相邻元素之间的字节偏移量
         n - 要复制的元素个数
         swap - 是否交换字节序的标志
         arr - 数组对象的指针
*/
static void
_default_copyswapn(void *dst, npy_intp dstride, void *src,
                   npy_intp sstride, npy_intp n, int swap, void *arr)
{
    npy_intp i;
    PyArray_CopySwapFunc *copyswap;
    char *dstptr = dst;
    char *srcptr = src;

    copyswap = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->copyswap; // 获取复制和交换函数

    for (i = 0; i < n; i++) {
        copyswap(dstptr, srcptr, swap, arr); // 调用复制和交换函数
        dstptr += dstride; // 移动到下一个目标元素的位置
        srcptr += sstride; // 移动到下一个源元素的位置
    }
}

/*NUMPY_API
  初始化 arrfuncs 结构体，将所有函数指针初始化为 NULL
*/
NPY_NO_EXPORT void
PyArray_InitArrFuncs(PyArray_ArrFuncs *f)
{
    int i;

    for(i = 0; i < NPY_NTYPES_ABI_COMPATIBLE; i++) {
        f->cast[i] = NULL; // 将类型转换函数指针数组初始化为 NULL
    }
    f->getitem = NULL; // 获取元素函数指针初始化为 NULL
    f->setitem = NULL; // 设置元素函数指针初始化为 NULL
    f->copyswapn = NULL; // 复制和交换多个元素函数指针初始化为 NULL
    f->copyswap = NULL; // 复制和交换单个元素函数指针初始化为 NULL
    f->compare = NULL; // 比较函数指针初始化为 NULL
    f->argmax = NULL; // 最大值索引函数指针初始化为 NULL
    f->argmin = NULL; // 最小值索引函数指针初始化为 NULL
    f->dotfunc = NULL; // 点积函数指针初始化为 NULL
    f->scanfunc = NULL; // 扫描函数指针初始化为 NULL
    f->fromstr = NULL; // 字符串转数组函数指针初始化为 NULL
    f->nonzero = NULL; // 非零判断函数指针初始化为 NULL
    f->fill = NULL; // 填充函数指针初始化为 NULL
    f->fillwithscalar = NULL; // 使用标量填充函数指针初始化为 NULL
    for(i = 0; i < NPY_NSORTS; i++) {
        f->sort[i] = NULL; // 排序函数指针数组初始化为 NULL
        f->argsort[i] = NULL; // 获取排序索引函数指针数组初始化为 NULL
    }
    f->castdict = NULL; // 类型转换字典指针初始化为 NULL
    f->scalarkind = NULL; // 标量类型指针初始化为 NULL
    f->cancastscalarkindto = NULL; // 标量类型转换函数指针初始化为 NULL
    f->cancastto = NULL; // 类型转换函数指针初始化为 NULL
    f->_unused1 = NULL; // 未使用的字段1初始化为 NULL
    f->_unused2 = NULL; // 未使用的字段2初始化为 NULL
}
    f->_unused3 = NULL;


    // 将结构体指针 f 的成员变量 _unused3 设置为 NULL
/*
  返回与此类型相关联的类型编号 >=NPY_USERDEF。
  需要arraytypes.inc中定义的userdecrs表和PyArray_NUMUSER变量。
*/
/*NUMPY_API
 * 注册数据类型
 *
 * 从原型创建一个新的描述符。
 *
 * 该原型与NumPy 1.x ABI兼容，在1.x中将用作实际描述符。但由于ABI已更改，这在2.0中无法工作，
 * 我们将所有字段复制到新的结构体中。
 *
 * 成功注册后，代码必须使用`descr = PyArray_DescrFromType(num);`。这与1.x中的使用兼容。
 *
 * 此函数仅在2.x中复制所有内部引用。这应该是不相关的，因为任何内部引用都是不朽的。
*/
NPY_NO_EXPORT int
PyArray_RegisterDataType(PyArray_DescrProto *descr_proto)
{
    int typenum;
    int i;
    PyArray_ArrFuncs *f;

    /* 查看是否已经注册了此类型 */
    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        if (userdescrs[i]->type_num == descr_proto->type_num) {
            return descr_proto->type_num;
        }
    }
    typenum = NPY_USERDEF + NPY_NUMUSERTYPES;
    if (typenum >= NPY_VSTRING) {
        PyErr_SetString(PyExc_ValueError,
                "Too many user defined dtypes registered");
        return -1;
    }
    descr_proto->type_num = -1;
    if (PyDataType_ISUNSIZED(descr_proto)) {
        PyErr_SetString(PyExc_ValueError, "cannot register a" \
                        "flexible data-type");
        return -1;
    }
    f = descr_proto->f;
    if (f->nonzero == NULL) {
        f->nonzero = _default_nonzero;
    }
    if (f->copyswapn == NULL) {
        f->copyswapn = _default_copyswapn;
    }
    if (f->copyswap == NULL || f->getitem == NULL ||
        f->setitem == NULL) {
        PyErr_SetString(PyExc_ValueError, "a required array function"   \
                        " is missing.");
        return -1;
    }
    if (descr_proto->typeobj == NULL) {
        PyErr_SetString(PyExc_ValueError, "missing typeobject");
        return -1;
    }

    int use_void_clearimpl = 0;
    /*
     * 如果描述符的标志包含 NPY_ITEM_IS_POINTER 或 NPY_ITEM_REFCOUNT 标志位，
     * 表明用户自定义的数据类型实际上不能执行引用计数。然而，存在某些已有的
     * 手段（例如 xpress），它们使用了结构化的方法：
     *     dtype((xpress.var, [('variable', 'O')]))
     * 因此我们必须支持这种情况。但这样的结构必须是常量的（即在注册时固定，
     * 这是对于 `xpress` 而言的情况）。
     */
    use_void_clearimpl = 1;

    /*
     * 如果描述符的 names 或 fields 为 NULL，或者 fields 不是 PyDict 类型，
     * 则抛出错误。这是因为不支持使用 `NPY_ITEM_IS_POINTER` 或 `NPY_ITEM_REFCOUNT`
     * 的旧用户数据类型。只有在注册时硬编码了 names 和 fields 的结构化数据类型才能被创建。
     */
    if (descr_proto->names == NULL || descr_proto->fields == NULL ||
        !PyDict_CheckExact(descr_proto->fields)) {
        PyErr_Format(PyExc_ValueError,
                "Failed to register dtype for %S: Legacy user dtypes "
                "using `NPY_ITEM_IS_POINTER` or `NPY_ITEM_REFCOUNT` are "
                "unsupported.  It is possible to create such a dtype only "
                "if it is a structured dtype with names and fields "
                "hardcoded at registration time.\n"
                "Please contact the NumPy developers if this used to work "
                "but now fails.", descr_proto->typeobj);
        return -1;
    }

    /*
     * 扩展 userdescrs 数组，以便容纳新的用户数据类型描述符。
     * 如果内存分配失败，则返回内存错误。
     */
    userdescrs = realloc(userdescrs,
                         (NPY_NUMUSERTYPES+1)*sizeof(void *));
    if (userdescrs == NULL) {
        PyErr_SetString(PyExc_MemoryError, "RegisterDataType");
        return -1;
    }

    /*
     * 由于旧用户数据类型类无法具有名称（因为用户从未定义过名称），
     * 我们在这里为其创建一个名称。这些数据类型在本质上是静态类型。
     *
     * 注意：我们没有意图再次释放这段内存，因为这与静态类型定义的行为完全一致。
     */
    const char *scalar_name = descr_proto->typeobj->tp_name;

    /*
     * 为了获得一个合理的 __name__，我们必须仅获取名称，并忽略模块部分，
     * 这是因为静态类型在这方面受到限制（尽管这不是理想的，但在实践中并不是一个大问题）。
     * 这就是 Python 用于打印静态类型 __name__ 的方法。
     */
    const char *dot = strrchr(scalar_name, '.');
    if (dot) {
        scalar_name = dot + 1;
    }
    Py_ssize_t name_length = strlen(scalar_name) + 14;

    /*
     * 分配内存并格式化名称字符串，格式为 "numpy.dtype[类型名]"。
     * 如果内存分配失败，则返回内存错误。
     */
    char *name = PyMem_Malloc(name_length);
    if (name == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    snprintf(name, name_length, "numpy.dtype[%s]", scalar_name);

    /*
     * 将用户提供的描述符结构体复制到一个新的结构体中。这样做是为了允许两者之间的布局不同。
     * 如果内存分配失败，则释放已分配的名称内存并返回内存错误。
     */
    _PyArray_LegacyDescr *descr = PyObject_Malloc(sizeof(_PyArray_LegacyDescr));
    if (descr == NULL) {
        PyMem_FREE(name);
        PyErr_NoMemory();
        return -1;
    }
    PyObject_INIT(descr, Py_TYPE(descr_proto));

    /*
     * 简单地按名称复制所有字段：
     * 增加类型对象的引用计数，设置类型对象和类型的种类。
     */
    Py_XINCREF(descr_proto->typeobj);
    descr->typeobj = descr_proto->typeobj;
    descr->kind = descr_proto->kind;
    // 将描述符的类型设置为原型描述符的类型
    descr->type = descr_proto->type;
    // 将描述符的字节顺序设置为原型描述符的字节顺序
    descr->byteorder = descr_proto->byteorder;
    // 将描述符的标志设置为原型描述符的标志
    descr->flags = descr_proto->flags;
    // 将描述符的元素大小设置为原型描述符的元素大小
    descr->elsize = descr_proto->elsize;
    // 将描述符的对齐方式设置为原型描述符的对齐方式
    descr->alignment = descr_proto->alignment;
    // 将描述符的子数组设置为原型描述符的子数组
    descr->subarray = descr_proto->subarray;
    // 增加原型描述符的字段的引用计数并将其赋给描述符的字段
    Py_XINCREF(descr_proto->fields);
    descr->fields = descr_proto->fields;
    // 增加原型描述符的名称的引用计数并将其赋给描述符的名称
    Py_XINCREF(descr_proto->names);
    descr->names = descr_proto->names;
    // 增加原型描述符的元数据的引用计数并将其赋给描述符的元数据
    Py_XINCREF(descr_proto->metadata);
    descr->metadata = descr_proto->metadata;
    // 如果原型描述符的 C 元数据不为空，则克隆它并赋给描述符的 C 元数据；否则置空描述符的 C 元数据
    if (descr_proto->c_metadata != NULL) {
        descr->c_metadata = NPY_AUXDATA_CLONE(descr_proto->c_metadata);
    }
    else {
        descr->c_metadata = NULL;
    }
    // 将描述符的哈希值置为无效值（假定字段未设置）
    descr->hash = -1;

    // 将描述符添加到用户描述符数组中，并增加用户描述符数量计数
    userdescrs[NPY_NUMUSERTYPES++] = descr;

    // 将描述符的类型编号设置为指定的类型编号
    descr->type_num = typenum;
    // 更新原型描述符的类型编号以便于检测重复注册
    descr_proto->type_num = typenum;
    // 尝试包装旧版本的描述符以处理遗留描述符，如果失败则进行回滚
    if (dtypemeta_wrap_legacy_descriptor(
            descr, descr_proto->f, &PyArrayDescr_Type, name, NULL) < 0) {
        descr->type_num = -1;
        NPY_NUMUSERTYPES--;
        // 覆盖描述符的类型，防止错误导致解除引用崩溃
        Py_SET_TYPE(descr, &PyArrayDescr_Type);
        Py_DECREF(descr);
        PyMem_Free(name);  // 只有在失败时释放名称
        return -1;
    }
    // 如果使用了 void_clearimpl，则设置相应的清除和零填充循环
    if (use_void_clearimpl) {
        // 设置清除循环函数指针
        NPY_DT_SLOTS(NPY_DTYPE(descr))->get_clear_loop = (
                (PyArrayMethod_GetTraverseLoop *)&npy_get_clear_void_and_legacy_user_dtype_loop);
        // 设置零填充循环函数指针
        NPY_DT_SLOTS(NPY_DTYPE(descr))->get_fill_zero_loop = (
                (PyArrayMethod_GetTraverseLoop *)&npy_get_zerofill_void_and_legacy_user_dtype_loop);
    }

    // 返回设置的类型编号
    return typenum;
/*
 * 检查是否已经使用新的转换实现机制缓存了转换。
 * 如果是，则不清空缓存（但是会默默地继续）。用户在使用后不应修改转换，
 * 但这可能在设置过程中意外发生（也可能从未发生）。参见 https://github.com/numpy/numpy/issues/20009
 */
static int _warn_if_cast_exists_already(
        PyArray_Descr *descr, int totype, char *funcname)
{
    // 从类型编号获取对应的数据类型元数据对象
    PyArray_DTypeMeta *to_DType = PyArray_DTypeFromTypeNum(totype);
    if (to_DType == NULL) {
        return -1;
    }
    // 从字典中获取对应转换实现的对象
    PyObject *cast_impl = PyDict_GetItemWithError(
            NPY_DT_SLOTS(NPY_DTYPE(descr))->castingimpls, (PyObject *)to_DType);
    Py_DECREF(to_DType);
    if (cast_impl == NULL) {
        if (PyErr_Occurred()) {
            return -1;
        }
    }
    else {
        // 如果获取到转换实现对象
        char *extra_msg;
        if (cast_impl == Py_None) {
            extra_msg = "the cast will continue to be considered impossible.";
        }
        else {
            extra_msg = "the previous definition will continue to be used.";
        }
        Py_DECREF(cast_impl);
        // 根据类型编号获取对应的描述符对象
        PyArray_Descr *to_descr = PyArray_DescrFromType(totype);
        // 发出运行时警告，说明转换注册/修改发生在使用后
        int ret = PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                "A cast from %R to %R was registered/modified using `%s` "
                "after the cast had been used.  "
                "This registration will have (mostly) no effect: %s\n"
                "The most likely fix is to ensure that casts are the first "
                "thing initialized after dtype registration.  "
                "Please contact the NumPy developers with any questions!",
                descr, to_descr, funcname, extra_msg);
        Py_DECREF(to_descr);
        if (ret < 0) {
            return -1;
        }
    }
    return 0;
}

/*
 * NUMPY_API
 * 注册转换函数
 * 替换当前存储的任何函数。
 */
NPY_NO_EXPORT int
PyArray_RegisterCastFunc(PyArray_Descr *descr, int totype,
                         PyArray_VectorUnaryFunc *castfunc)
{
    PyObject *cobj, *key;
    int ret;

    // 检查类型编号是否有效
    if (totype >= NPY_NTYPES_LEGACY && !PyTypeNum_ISUSERDEF(totype)) {
        PyErr_SetString(PyExc_TypeError, "invalid type number.");
        return -1;
    }
    // 检查是否已经存在相同转换的警告
    if (_warn_if_cast_exists_already(
            descr, totype, "PyArray_RegisterCastFunc") < 0) {
        return -1;
    }

    // 如果类型编号小于 NPY_NTYPES_ABI_COMPATIBLE，直接设置转换函数
    if (totype < NPY_NTYPES_ABI_COMPATIBLE) {
        PyDataType_GetArrFuncs(descr)->cast[totype] = castfunc;
        return 0;
    }
    // 如果转换函数字典尚未初始化，初始化它
    if (PyDataType_GetArrFuncs(descr)->castdict == NULL) {
        PyDataType_GetArrFuncs(descr)->castdict = PyDict_New();
        if (PyDataType_GetArrFuncs(descr)->castdict == NULL) {
            return -1;
        }
    }
    // 创建用于作为字典键的 Python 整数对象
    key = PyLong_FromLong(totype);
    if (PyErr_Occurred()) {
        return -1;
    }
    // 创建转换函数对象的 Capsule 对象
    cobj = PyCapsule_New((void *)castfunc, NULL, NULL);
    if (cobj == NULL) {
        Py_DECREF(key);
        return -1;
    }
    # 调用 PyDataType_GetArrFuncs 函数获取描述符的数组函数字典（castdict），并向其中添加键值对
    ret = PyDict_SetItem(PyDataType_GetArrFuncs(descr)->castdict, key, cobj);
    # 减少键对象 key 的引用计数，可能会释放其内存
    Py_DECREF(key);
    # 减少值对象 cobj 的引用计数，可能会释放其内存
    Py_DECREF(cobj);
    # 返回 PyDict_SetItem 函数的执行结果，通常表示操作是否成功
    return ret;
/*NUMPY_API
 * Register a type number indicating that a descriptor can be cast
 * to it safely
 */
NPY_NO_EXPORT int
PyArray_RegisterCanCast(PyArray_Descr *descr, int totype,
                        NPY_SCALARKIND scalar)
{
    /*
     * 如果允许这样做，内置类型的类型转换查找表需要修改，
     * 因为对于它们，不会检查 cancastto。
     */
    if (!PyTypeNum_ISUSERDEF(descr->type_num) &&
                                        !PyTypeNum_ISUSERDEF(totype)) {
        PyErr_SetString(PyExc_ValueError,
                        "At least one of the types provided to "
                        "RegisterCanCast must be user-defined.");
        return -1;
    }
    if (_warn_if_cast_exists_already(
            descr, totype, "PyArray_RegisterCanCast") < 0) {
        return -1;
    }

    if (scalar == NPY_NOSCALAR) {
        /*
         * 使用 cancastto 进行注册
         * 这些列表一旦创建就不会被释放
         * —— 它们成为数据类型的一部分
         */
        if (PyDataType_GetArrFuncs(descr)->cancastto == NULL) {
            PyDataType_GetArrFuncs(descr)->cancastto = (int *)malloc(1*sizeof(int));
            if (PyDataType_GetArrFuncs(descr)->cancastto == NULL) {
                PyErr_NoMemory();
                return -1;
            }
            PyDataType_GetArrFuncs(descr)->cancastto[0] = NPY_NOTYPE;
        }
        return _append_new(&PyDataType_GetArrFuncs(descr)->cancastto, totype);
    }
    else {
        /* 使用 cancastscalarkindto 进行注册 */
        if (PyDataType_GetArrFuncs(descr)->cancastscalarkindto == NULL) {
            int i;
            PyDataType_GetArrFuncs(descr)->cancastscalarkindto =
                (int **)malloc(NPY_NSCALARKINDS* sizeof(int*));
            if (PyDataType_GetArrFuncs(descr)->cancastscalarkindto == NULL) {
                PyErr_NoMemory();
                return -1;
            }
            for (i = 0; i < NPY_NSCALARKINDS; i++) {
                PyDataType_GetArrFuncs(descr)->cancastscalarkindto[i] = NULL;
            }
        }
        if (PyDataType_GetArrFuncs(descr)->cancastscalarkindto[scalar] == NULL) {
            PyDataType_GetArrFuncs(descr)->cancastscalarkindto[scalar] =
                (int *)malloc(1*sizeof(int));
            if (PyDataType_GetArrFuncs(descr)->cancastscalarkindto[scalar] == NULL) {
                PyErr_NoMemory();
                return -1;
            }
            PyDataType_GetArrFuncs(descr)->cancastscalarkindto[scalar][0] =
                NPY_NOTYPE;
        }
        return _append_new(&PyDataType_GetArrFuncs(descr)->cancastscalarkindto[scalar], totype);
    }
}
// 定义 legacy_userdtype_common_dtype_function 函数，参数为两个 PyArray_DTypeMeta 类型的指针
legacy_userdtype_common_dtype_function(
        PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    // 初始化 skind1 和 skind2 为 NPY_NOSCALAR
    int skind1 = NPY_NOSCALAR, skind2 = NPY_NOSCALAR, skind;

    // 如果 other 不是 legacy 类型，则可以始终退化为新样式类型
    if (!NPY_DT_is_legacy(other)) {
        // 增加 Py_NotImplemented 的引用计数，并返回 Py_NotImplemented 指针
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    
    // 推迟处理，以便只有一种类型处理转换
    if (cls->type_num < other->type_num) {
        // 增加 Py_NotImplemented 的引用计数，并返回 Py_NotImplemented 指针
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }

    // 检查是否可以安全地从一种类型转换为另一种类型
    if (PyArray_CanCastSafely(cls->type_num, other->type_num)) {
        // 增加 other 的引用计数，并返回 other 指针
        Py_INCREF(other);
        return other;
    }
    if (PyArray_CanCastSafely(other->type_num, cls->type_num)) {
        // 增加 cls 的引用计数，并返回 cls 指针
        Py_INCREF(cls);
        return cls;
    }

    /*
     * 以下代码曾是 PyArray_PromoteTypes() 的一部分。
     * 可以预期这段代码不会被使用。
     * 实际上，它允许将两种不同的用户定义类型提升为相同 "kind" 的 NumPy 类型。
     * 在实践中，由于 PyArray_EquivTypes(descr1, descr2) 的简化，使用与 NumPy 相同的 `kind` 从未可能。
     * 如果 kind 和元素大小匹配，则 PyArray_EquivTypes(descr1, descr2) 返回 True（例如，bfloat16 和 float16 将等效）。
     * 这个选项也非常难以理解，并且在示例中没有使用。
     */

    // 将 'kind' 字符转换为标量 kind
    switch (cls->singleton->kind) {
        case 'b':
            skind1 = NPY_BOOL_SCALAR;
            break;
        case 'u':
            skind1 = NPY_INTPOS_SCALAR;
            break;
        case 'i':
            skind1 = NPY_INTNEG_SCALAR;
            break;
        case 'f':
            skind1 = NPY_FLOAT_SCALAR;
            break;
        case 'c':
            skind1 = NPY_COMPLEX_SCALAR;
            break;
    }
    switch (other->singleton->kind) {
        case 'b':
            skind2 = NPY_BOOL_SCALAR;
            break;
        case 'u':
            skind2 = NPY_INTPOS_SCALAR;
            break;
        case 'i':
            skind2 = NPY_INTNEG_SCALAR;
            break;
        case 'f':
            skind2 = NPY_FLOAT_SCALAR;
            break;
        case 'c':
            skind2 = NPY_COMPLEX_SCALAR;
            break;
    }

    // 如果两者都是标量，可能会存在一种提升的可能性
    # 如果两个标量类型都不是 NPY_NOSCALAR
    if (skind1 != NPY_NOSCALAR && skind2 != NPY_NOSCALAR) {

        # 选择较大的标量类型作为起点
        skind = (skind1 > skind2) ? skind1 : skind2;
        
        # 根据较大标量类型确定最小的可以表示该类型的数据类型
        int ret_type_num = _npy_smallest_type_of_kind_table[skind];

        # 无限循环，直到找到合适的数据类型或者确认没有更大的类型
        for (;;) {

            # 如果没有该种类的更大数据类型，则尝试下一个更大的种类
            if (ret_type_num < 0) {
                ++skind;
                # 使用 -1 表示未找到更大的推广类型
                if (skind < NPY_NSCALARKINDS) {
                    ret_type_num = _npy_smallest_type_of_kind_table[skind];
                }
                else {
                    break;
                }
            }

            # 如果找到一个既能安全转换当前类的类型又能安全转换另一个类的类型，则完成
            if (PyArray_CanCastSafely(cls->type_num, ret_type_num) &&
                PyArray_CanCastSafely(other->type_num, ret_type_num)) {
                return PyArray_DTypeFromTypeNum(ret_type_num);
            }

            # 尝试当前种类的下一个更大的数据类型
            ret_type_num = _npy_next_larger_type_table[ret_type_num];
        }
    }

    # 如果两个标量类型中有任意一个是 NPY_NOSCALAR，则返回 Py_NotImplemented
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
/**
 * This function wraps a legacy cast into an array-method. This is mostly
 * used for legacy user-dtypes, but for example numeric to/from datetime
 * casts were only defined that way as well.
 *
 * @param from Source data type metadata for casting.
 * @param to Target data type metadata for casting.
 * @param casting Casting behavior specifier: `NPY_NO_CASTING` checks the legacy
 *        registered cast, otherwise uses the provided cast.
 */
NPY_NO_EXPORT int
PyArray_AddLegacyWrapping_CastingImpl(
        PyArray_DTypeMeta *from, PyArray_DTypeMeta *to, NPY_CASTING casting)
{
    // Determine the casting behavior if not explicitly provided
    if (casting < 0) {
        // Check if source and target types are the same
        if (from == to) {
            casting = NPY_NO_CASTING;
        }
        // Check for safe casting capability
        else if (PyArray_LegacyCanCastTypeTo(
                from->singleton, to->singleton, NPY_SAFE_CASTING)) {
            casting = NPY_SAFE_CASTING;
        }
        // Check for same-kind casting capability
        else if (PyArray_LegacyCanCastTypeTo(
                from->singleton, to->singleton, NPY_SAME_KIND_CASTING)) {
            casting = NPY_SAME_KIND_CASTING;
        }
        // Default to unsafe casting if no other options are suitable
        else {
            casting = NPY_UNSAFE_CASTING;
        }
    }

    // Prepare an array method specification for legacy casting
    PyArray_DTypeMeta *dtypes[2] = {from, to};
    PyArrayMethod_Spec spec = {
            /* Name is not actually used, but allows identifying these. */
            .name = "legacy_cast",
            .nin = 1,
            .nout = 1,
            .casting = casting,
            .dtypes = dtypes,
    };

    // Define method flags and slots based on whether types are identical or not
    if (from == to) {
        spec.flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_SUPPORTS_UNALIGNED;
        // Define slots for identical type casting
        PyType_Slot slots[] = {
            {NPY_METH_get_loop, &legacy_cast_get_strided_loop},
            {NPY_METH_resolve_descriptors, &legacy_same_dtype_resolve_descriptors},
            {0, NULL}};
        spec.slots = slots;
        // Add the casting implementation using the specified method specification
        return PyArray_AddCastingImplementation_FromSpec(&spec, 1);
    }
    else {
        spec.flags = NPY_METH_REQUIRES_PYAPI;
        // Define slots for different type casting
        PyType_Slot slots[] = {
            {NPY_METH_get_loop, &legacy_cast_get_strided_loop},
            {NPY_METH_resolve_descriptors, &simple_cast_resolve_descriptors},
            {0, NULL}};
        spec.slots = slots;
        // Add the casting implementation using the specified method specification
        return PyArray_AddCastingImplementation_FromSpec(&spec, 1);
    }
}
```