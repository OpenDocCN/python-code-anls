# `D:\src\scipysrc\pandas\pandas\_libs\testing.pyx`

```
import cmath  # 导入复数数学运算模块
import math  # 导入数学运算模块

import numpy as np  # 导入NumPy库，并使用np作为别名

from numpy cimport import_array  # 导入NumPy C API中的import_array函数

import_array()  # 调用NumPy C API中的import_array函数，导入数组对象

from pandas._libs.missing cimport (  # 从pandas库的C扩展模块中导入指定函数
    checknull,  # 导入checknull函数，用于检查缺失值
    is_matching_na,  # 导入is_matching_na函数，用于检查匹配的缺失值
)
from pandas._libs.util cimport (  # 从pandas库的C扩展模块中导入指定函数
    is_array,  # 导入is_array函数，用于检查对象是否为数组
    is_complex_object,  # 导入is_complex_object函数，用于检查对象是否为复数对象
    is_real_number_object,  # 导入is_real_number_object函数，用于检查对象是否为实数对象
)


from pandas.core.dtypes.missing import array_equivalent  # 从pandas库中导入array_equivalent函数，用于比较数组对象的等价性


cdef bint isiterable(obj):  # 定义Cython函数isiterable，用于检查对象是否可迭代
    return hasattr(obj, "__iter__")


cdef bint has_length(obj):  # 定义Cython函数has_length，用于检查对象是否具有长度
    return hasattr(obj, "__len__")


cdef bint is_dictlike(obj):  # 定义Cython函数is_dictlike，用于检查对象是否类似字典
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__")


cpdef assert_dict_equal(a, b, bint compare_keys=True):  # 定义Cython函数assert_dict_equal，用于比较两个字典是否相等
    assert is_dictlike(a) and is_dictlike(b), (
        "Cannot compare dict objects, one or both is not dict-like"
    )

    a_keys = frozenset(a.keys())  # 获取字典a的键集合
    b_keys = frozenset(b.keys())  # 获取字典b的键集合

    if compare_keys:
        assert a_keys == b_keys  # 检查字典的键集合是否相等

    for k in a_keys:
        assert_almost_equal(a[k], b[k])  # 对字典的每个键进行近似相等性断言

    return True  # 返回True，表示字典相等性断言通过


cpdef assert_almost_equal(a, b,  # 定义Cython函数assert_almost_equal，用于比较两个对象是否近似相等
                          rtol=1.e-5, atol=1.e-8,  # 相对容差和绝对容差的默认值
                          bint check_dtype=True,  # 是否检查数据类型的标志位
                          obj=None, lobj=None, robj=None, index_values=None):  # 用于显示断言消息的相关参数
    """
    Check that left and right objects are almost equal.

    Parameters
    ----------
    a : object
    b : object
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    check_dtype: bool, default True
        check dtype if both a and b are np.ndarray.
    obj : str, default None
        Specify object name being compared, internally used to show
        appropriate assertion message.
    lobj : str, default None
        Specify left object name being compared, internally used to show
        appropriate assertion message.
    robj : str, default None
        Specify right object name being compared, internally used to show
        appropriate assertion message.
    index_values : Index | ndarray, default None
        Specify shared index values of objects being compared, internally used
        to show appropriate assertion message.

    """
    cdef:
        double diff = 0.0  # 存储两个对象的差值
        Py_ssize_t i, na, nb  # Py_ssize_t类型的变量，用于迭代计数和对象长度比较
        double fa, fb  # 存储a和b的转换为double类型后的值
        bint is_unequal = False  # 存储比较结果的布尔变量，初始为False，表示相等
        str first_diff = ""  # 存储首次不等的字符串描述

    if lobj is None:
        lobj = a  # 如果未指定左对象名称，使用a的值作为名称
    if robj is None:
        robj = b  # 如果未指定右对象名称，使用b的值作为名称

    if isinstance(a, set) or isinstance(b, set):
        assert a == b, f"{a} != {b}"  # 如果a或b是集合，则直接比较它们是否相等
        return True

    if isinstance(a, dict) or isinstance(b, dict):
        return assert_dict_equal(a, b)  # 如果a或b是字典，则调用字典比较函数assert_dict_equal

    if isinstance(a, str) or isinstance(b, str):
        assert a == b, f"{a} != {b}"  # 如果a或b是字符串，则直接比较它们是否相等
        return True

    a_is_ndarray = is_array(a)  # 检查a是否为NumPy数组
    b_is_ndarray = is_array(b)  # 检查b是否为NumPy数组

    if obj is None:
        if a_is_ndarray or b_is_ndarray:
            obj = "numpy array"  # 如果a或b是NumPy数组，则将obj设置为numpy array
        else:
            obj = "Iterable"  # 否则将obj设置为Iterable
    # 检查变量a是否可迭代
    if isiterable(a):

        # 如果变量b不可迭代，引入assert_class_equal函数来确保a和b的类型不同，从而引发错误
        if not isiterable(b):
            from pandas._testing import assert_class_equal

            # 类型不能相同，抛出错误
            assert_class_equal(a, b, obj=obj)

        # 确保a和b都具有长度，否则抛出异常
        assert has_length(a) and has_length(b), (
            f"Can't compare objects without length, one or both is invalid: ({a}, {b})"
        )

        # 如果a和b都是ndarray，并且a_is_ndarray和b_is_ndarray为True
        if a_is_ndarray and b_is_ndarray:
            # 获取a和b的元素个数
            na, nb = a.size, b.size

            # 如果a和b的形状不同，引发assert_detail异常
            if a.shape != b.shape:
                from pandas._testing import raise_assert_detail
                raise_assert_detail(
                    obj, f"{obj} shapes are different", a.shape, b.shape)

            # 如果需要检查数据类型，并且a和b的dtype不同，引入assert_attr_equal函数来确保它们的dtype相同
            if check_dtype and a.dtype != b.dtype:
                from pandas._testing import assert_attr_equal
                assert_attr_equal("dtype", a, b, obj=obj)

            # 使用array_equivalent函数检查a和b是否严格等价
            if array_equivalent(a, b, strict_nan=True):
                return True

        else:
            # 如果a和b不是ndarray，则分别获取它们的长度
            na, nb = len(a), len(b)

        # 如果a和b的长度不相等，引发assert_detail异常
        if na != nb:
            from pandas._testing import raise_assert_detail

            # 如果差异较小，打印差异集合；否则设为None
            if abs(na - nb) < 10:
                r = list(set(a) ^ set(b))
            else:
                r = None

            raise_assert_detail(obj, f"{obj} length are different", na, nb, r)

        # 遍历a的每个元素，逐个使用assert_almost_equal函数检查它们的近似相等性
        for i in range(len(a)):
            try:
                assert_almost_equal(a[i], b[i], rtol=rtol, atol=atol)
            except AssertionError:
                is_unequal = True
                diff += 1
                if not first_diff:
                    first_diff = (
                        f"At positional index {i}, first diff: {a[i]} != {b[i]}"
                    )

        # 如果发现不等，引入raise_assert_detail函数来引发异常，说明对象值不同
        if is_unequal:
            from pandas._testing import raise_assert_detail
            msg = (f"{obj} values are different "
                   f"({np.round(diff * 100.0 / na, 5)} %)")
            raise_assert_detail(
                obj, msg, lobj, robj, first_diff=first_diff, index_values=index_values
            )

        # 如果所有元素都近似相等，则返回True
        return True

    # 如果a不可迭代但b可迭代，引入assert_class_equal函数来确保a和b的类型不同，从而引发错误
    elif isiterable(b):
        from pandas._testing import assert_class_equal

        # 类型不能相同，抛出错误
        assert_class_equal(a, b, obj=obj)

    # 如果a或b是NaN或None，检查它们是否匹配，否则引发异常
    if checknull(a):
        # 检查NaN或None的比较情况，如果匹配则返回True，否则引发异常
        if is_matching_na(a, b, nan_matches_none=False):
            return True
        elif checknull(b):
            # 如果a和b都是NaN或None，则引发AssertionError异常，说明它们不匹配
            # GH#18463
            raise AssertionError(f"Mismatched null-like values {a} != {b}")
        raise AssertionError(f"{a} != {b}")
    elif checknull(b):
        # 如果b是NaN或None，引发AssertionError异常，说明a和b不匹配
        raise AssertionError(f"{a} != {b}")

    # 如果a和b是普通对象且相等，则返回True
    if a == b:
        # 对象比较相等
        return True

    # 如果a和b是实数对象，使用math.isclose检查它们的近似相等性
    if is_real_number_object(a) and is_real_number_object(b):
        fa, fb = a, b

        # 如果fa和fb不近似相等，引发错误并提供详细信息
        if not math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol):
            assert False, (f"expected {fb:.5f} but got {fa:.5f}, "
                           f"with rtol={rtol}, atol={atol}")
        return True
    # 如果 a 和 b 都是复杂对象（complex object），则进行以下操作
    if is_complex_object(a) and is_complex_object(b):
        # 使用相对容差（rel_tol=rtol）和绝对容差（abs_tol=atol）检查 a 和 b 是否相似
        if not cmath.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            # 如果不相似，触发断言错误，显示期望值和实际值，以及使用的容差值
            assert False, (f"expected {b:.5f} but got {a:.5f}, "
                           f"with rtol={rtol}, atol={atol}")
        # 如果相似，返回 True 表示通过测试
        return True

    # 如果 a 和 b 不是都是复杂对象，则触发断言错误，显示 a 不等于 b 的具体值
    raise AssertionError(f"{a} != {b}")
```