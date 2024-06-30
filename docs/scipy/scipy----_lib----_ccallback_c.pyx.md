# `D:\src\scipysrc\scipy\scipy\_lib\_ccallback_c.pyx`

```
# 关闭 Cython 性能提示，确保在编译时不显示性能提示信息
# cython: show_performance_hints=False

# 导入需要使用的 CPython 扩展库中的函数和类型
from cpython.pycapsule cimport (
    PyCapsule_CheckExact, PyCapsule_New, PyCapsule_SetContext, PyCapsule_GetName, PyCapsule_GetPointer,
    PyCapsule_GetContext
)
# 导入 CPython 中的整型类型转换函数
from cpython.long cimport PyLong_AsVoidPtr
# 导入标准 C 库中的内存管理和字符串操作函数
from libc.stdlib cimport free
from libc.string cimport strdup
# 导入标准 C 库中的数学函数
from libc.math cimport sin

# 导入当前包中的 C 扩展模块中的函数和类型
from .ccallback cimport (ccallback_t, ccallback_prepare, ccallback_release, CCALLBACK_DEFAULTS,
                         ccallback_signature_t)


#
# PyCapsule helpers
#

# 定义用于释放 PyCapsule 中数据的析构函数
cdef void raw_capsule_destructor(object capsule) noexcept:
    cdef const char *name
    # 获取 PyCapsule 的名称
    name = PyCapsule_GetName(capsule)
    # 释放名称对应的内存
    free(<char*>name)


def get_raw_capsule(func_obj, name_obj, context_obj):
    """
    get_raw_capsule(ptr, name, context)

    根据给定的指针、名称和上下文创建一个新的 PyCapsule。

    Parameters
    ----------
    ptr : {PyCapsule, int}
        指针的内存地址。
    name : str
        包含签名的 Python 字符串。
    context : {PyCapsule, int}
        上下文的内存地址。
        如果为 NULL 且 ptr 是 PyCapsule，则使用 ptr 的上下文。

    """
    cdef:
        void *func
        void *context
        const char *capsule_name
        const char *name
        const char *name_copy

    # 如果 name_obj 为 None，则将 name 设为 NULL
    if name_obj is None:
        name = NULL
    # 如果 name_obj 不是 bytes 类型，则将其编码为 ASCII 字符串
    elif not isinstance(name_obj, bytes):
        name_obj = name_obj.encode('ascii')
        name = <char*>name_obj
    else:
        name = <char*>name_obj

    # 如果 context_obj 是 PyCapsule 类型，则获取其名称和指针
    if PyCapsule_CheckExact(context_obj):
        capsule_name = PyCapsule_GetName(context_obj)
        context = PyCapsule_GetPointer(context_obj, capsule_name)
    # 如果 context_obj 是 None，则将 context 设为 NULL
    elif context_obj is None:
        context = NULL
    # 否则，将 context_obj 转换为指针类型
    else:
        context = PyLong_AsVoidPtr(int(context_obj))

    # 如果 func_obj 是 PyCapsule 类型，则获取其名称和指针
    if PyCapsule_CheckExact(func_obj):
        capsule_name = PyCapsule_GetName(func_obj)
        func = PyCapsule_GetPointer(func_obj, capsule_name)

        # 如果 context 为 NULL，则使用 func_obj 的上下文
        if context == NULL:
            context = PyCapsule_GetContext(func_obj)

        # 如果 name 为 NULL，则使用 func_obj 的名称
        if name == NULL:
            name = capsule_name
    # 否则，将 func_obj 转换为指针类型
    else:
        func = PyLong_AsVoidPtr(int(func_obj))

    # 如果 name 为 NULL，则将 name_copy 设为 NULL，否则复制 name 到 name_copy
    if name == NULL:
        name_copy = name
    else:
        name_copy = strdup(name)

    # 创建一个新的 PyCapsule，将 func、name_copy 和析构函数关联起来
    capsule = PyCapsule_New(func, name_copy, &raw_capsule_destructor)
    # 如果 context 不为 NULL，则设置 PyCapsule 的上下文
    if context != NULL:
        PyCapsule_SetContext(capsule, context)
    return capsule


def get_capsule_signature(capsule_obj):
    """
    get_capsule_signature(capsule_obj)

    获取 PyCapsule 的签名字符串。

    Parameters
    ----------
    capsule_obj : object
        PyCapsule 对象。

    Returns
    -------
    str
        PyCapsule 的签名字符串。

    Raises
    ------
    ValueError
        如果 PyCapsule 没有签名。

    """
    cdef const char *name
    # 获取 PyCapsule 的名称
    name = PyCapsule_GetName(capsule_obj)
    # 如果名称为 NULL，则抛出 ValueError 异常
    if name == NULL:
        raise ValueError("Capsule has no signature")
    # 将名称转换为 ASCII 字符串并返回
    return bytes(name).decode('ascii')


def check_capsule(item):
    """
    check_capsule(item)

    检查给定的对象是否为 PyCapsule。

    Parameters
    ----------
    item : object
        待检查的对象。

    Returns
    -------
    bool
        如果对象是 PyCapsule 则返回 True，否则返回 False。

    """
    # 检查对象是否为 PyCapsule 类型
    if PyCapsule_CheckExact(item):
        return True
    return False

# 定义一组函数签名的列表
sigs = [
    (b"double (double, int *, void *)", 0),
    (b"double (double, double, int *, void *)", 1)
]

# 如果 int 和 long 的大小相等，则添加另一个函数签名到列表中
if sizeof(int) == sizeof(long):
    sigs.append((b"double (double, long *, void *)", 0))
    # 在列表 sigs 中追加一个元组，元组包含一个字节字符串和一个整数
    sigs.append((b"double (double, double, long *, void *)", 1))
# 如果 int 和 short 的大小相等，执行以下操作
if sizeof(int) == sizeof(short):
    # 将函数签名及其值添加到 sigs 列表中
    sigs.append((b"double (double, short *, void *)", 0))
    sigs.append((b"double (double, double, short *, void *)", 1))

# 定义长度为 7 的 signatures 数组
cdef ccallback_signature_t signatures[7]

# 遍历 sigs 列表中的函数签名及其对应的值
for idx, sig in enumerate(sigs):
    # 将当前签名和值分配给 signatures 数组中对应的元素
    signatures[idx].signature = sig[0]
    signatures[idx].value = sig[1]

# 将 signatures 数组中的下一个元素的签名设置为 NULL
signatures[idx + 1].signature = NULL


# 定义了一个 Cython 函数 test_thunk_cython，用于执行 thunk 操作
cdef double test_thunk_cython(double a, int *error_flag, void *data) except? -1.0 nogil:
    """
    Implementation of a thunk routine in Cython
    """
    cdef:
        ccallback_t *callback = <ccallback_t *>data
        double result = 0

    # 如果回调函数指针不为 NULL
    if callback.c_function != NULL:
        # 根据回调函数的签名值调用相应的 C 函数指针
        if callback.signature.value == 0:
            result = (<double(*)(double, int *, void *) nogil>callback.c_function)(
                a, error_flag, callback.user_data)
        else:
            result = (<double(*)(double, double, int *, void *) nogil>callback.c_function)(
                a, 0.0, error_flag, callback.user_data)

        # 如果 error_flag 标志被设置，则返回 -1.0 表示错误
        if error_flag[0]:
            # Python 中的异常由回调函数设置
            return -1.0
    else:
        # 如果回调函数指针为 NULL，在全局解释锁下尝试执行 Python 回调
        with gil:
            try:
                return float((<object>callback.py_function)(a))
            except:  # noqa: E722
                # 捕获异常并设置 error_flag 标志
                error_flag[0] = 1
                raise

    # 返回计算结果
    return result


# 定义了一个 Cython 函数 test_call_cython，用于调用 thunk 函数
def test_call_cython(callback_obj, double value):
    """
    Implementation of a caller routine in Cython
    """
    cdef:
        ccallback_t callback
        int error_flag = 0
        double result

    # 准备回调结构体并设置默认值
    ccallback_prepare(&callback, signatures, callback_obj, CCALLBACK_DEFAULTS)

    # 在无全局解释锁环境下调用 test_thunk_cython 函数
    with nogil:
        result = test_thunk_cython(value, &error_flag, <void *>&callback)

    # 释放回调结构体
    ccallback_release(&callback)

    # 返回结果
    return result


# 定义了一个 Cython 函数 plus1_cython，实现简单的加法操作
cdef double plus1_cython(double a, int *error_flag, void *user_data) except * nogil:
    """
    Implementation of a callable in Cython
    """
    # 如果 a 的值为 2.0，则设置 error_flag 标志并抛出 ValueError 异常
    if a == 2.0:
        error_flag[0] = 1
        with gil:
            raise ValueError("failure...")

    # 如果 user_data 为 NULL，则返回 a + 1
    if user_data == NULL:
        return a + 1
    else:
        # 否则返回 a + user_data 指向的值
        return a + (<double *>user_data)[0]


# 定义了一个 Cython 函数 plus1b_cython，实现带有额外参数的加法操作
cdef double plus1b_cython(double a, double b, int *error_flag, void *user_data) except * nogil:
    # 调用 plus1_cython 函数，并将结果与 b 相加返回
    return plus1_cython(a, error_flag, user_data) + b


# 定义了一个 Cython 函数 plus1bc_cython，实现带有两个额外参数的加法操作
cdef double plus1bc_cython(double a, double b, double c, int *error_flag, void *user_data) except * nogil:
    # 调用 plus1_cython 函数，并将结果与 b、c 相加返回
    return plus1_cython(a, error_flag, user_data) + b + c


# 定义了一个 Cython 函数 sine，实现对数值的正弦函数计算
cdef double sine(double x, void *user_data) except * nogil:
    # 返回 x 的正弦值
    return sin(x)


# 导入 ctypes，声明上述 Cython 函数的 Ctypes 类型
import ctypes

# 声明了一个 ctypes 函数指针类型 plus1_t，用于表示 plus1_cython 函数指针
plus1_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.c_void_p)
plus1_ctypes = ctypes.cast(<size_t>&plus1_cython, plus1_t)

# 声明了一个 ctypes 函数指针类型 plus1b_t，用于表示 plus1b_cython 函数指针
plus1b_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p)
plus1b_ctypes = ctypes.cast(<size_t>&plus1b_cython, plus1b_t)
# 定义一个 ctypes 函数类型 plus1bc_t，接受三个 c_double 类型参数和一个指向 c_int 类型的指针，返回一个 c_double 类型值
plus1bc_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p)
# 使用 ctypes.cast 将 plus1bc_cython 的地址转换为 plus1bc_t 类型的函数指针
plus1bc_ctypes = ctypes.cast(<size_t>&plus1bc_cython, plus1bc_t)

# 定义一个 ctypes 函数类型 sine_t，接受一个 c_double 类型参数和一个 c_void_p 类型参数，返回一个 c_double 类型值
sine_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_void_p)
# 使用 ctypes.cast 将 sine 的地址转换为 sine_t 类型的函数指针
sine_ctypes = ctypes.cast(<size_t>&sine, sine_t)
```