# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_testing.py`

```
# 导入模块：atexit、os、unittest、warnings、numpy、pytest、scipy.sparse
import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse

# 导入 sklearn 中的具体类和函数
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import (
    TempMemmap,
    _convert_container,
    _delete_folder,
    _get_warnings_filters_info_list,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_no_warnings,
    assert_raise_message,
    assert_raises,
    assert_raises_regex,
    assert_run_python_script_without_output,
    check_docstring_parameters,
    create_memmap_backed_data,
    ignore_warnings,
    raises,
    set_random_state,
    turn_warnings_into_errors,
)
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
    _IS_WASM,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    parse_version,
    sp_version,
)
from sklearn.utils.metaestimators import available_if

# 测试函数：验证 set_random_state 函数的随机状态设置功能
def test_set_random_state():
    # 创建线性判别分析对象和决策树分类器对象
    lda = LinearDiscriminantAnalysis()
    tree = DecisionTreeClassifier()
    # 对线性判别分析对象设置随机状态为 3
    set_random_state(lda, 3)
    # 对决策树分类器对象设置随机状态为 3
    set_random_state(tree, 3)
    # 验证决策树分类器对象的随机状态是否为 3
    assert tree.random_state == 3

# 参数化测试：测试 assert_allclose_dense_sparse 函数对稠密和稀疏数组的比较
@pytest.mark.parametrize("csr_container", CSC_CONTAINERS)
def test_assert_allclose_dense_sparse(csr_container):
    # 创建一个 3x3 的数组
    x = np.arange(9).reshape(3, 3)
    # 错误信息
    msg = "Not equal to tolerance "
    # 将数组 x 转换为稀疏格式
    y = csr_container(x)
    # 对 x 和 y 进行比较
    for X in [x, y]:
        # 基本比较，预期会触发 AssertionError
        with pytest.raises(AssertionError, match=msg):
            assert_allclose_dense_sparse(X, X * 2)
        # 正确比较
        assert_allclose_dense_sparse(X, X)
    
    # 预期会触发 ValueError
    with pytest.raises(ValueError, match="Can only compare two sparse"):
        assert_allclose_dense_sparse(x, y)
    
    # 创建稀疏对角矩阵 A 和 csr_container 格式的稀疏数组 B
    A = sparse.diags(np.ones(5), offsets=0).tocsr()
    B = csr_container(np.ones((1, 5)))
    # 预期会触发 AssertionError
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        assert_allclose_dense_sparse(B, A)

# 测试函数：验证 assert_raises_regex 函数是否捕获指定异常和消息
def test_assert_raises_msg():
    # 使用 assert_raises_regex 捕获 AssertionError 异常，并验证消息 "Hello world"
    with assert_raises_regex(AssertionError, "Hello world"):
        # 使用 assert_raises 捕获 ValueError 异常，并验证消息 "Hello world"
        with assert_raises(ValueError, msg="Hello world"):
            pass

# 测试函数：验证 assert_raise_message 函数是否正确捕获指定异常和消息
def test_assert_raise_message():
    # 抛出 ValueError 异常，带有特定消息
    def _raise_ValueError(message):
        raise ValueError(message)

    # 不抛出异常的函数
    def _no_raise():
        pass

    # 验证 assert_raise_message 能够捕获 ValueError 异常和指定消息
    assert_raise_message(ValueError, "test", _raise_ValueError, "test")

    # 预期会触发 AssertionError
    assert_raises(
        AssertionError,
        assert_raise_message,
        ValueError,
        "something else",
        _raise_ValueError,
        "test",
    )

    # 预期会触发 ValueError
    assert_raises(
        ValueError,
        assert_raise_message,
        TypeError,
        "something else",
        _raise_ValueError,
        "test",
    )

    # 预期会触发 AssertionError
    assert_raises(AssertionError, assert_raise_message, ValueError, "test", _no_raise)

    # 多个异常的情况，预期会触发 AssertionError
    assert_raises(
        AssertionError,
        assert_raise_message,
        (ValueError, AttributeError),
        "test",
        _no_raise,
    )

# 测试函数：
def test_ignore_warning():
    # 这段代码用于验证 ignore_warnings 装饰器和上下文管理器的预期功能是否正常工作
    
    # 定义一个发出警告的函数
    def _warning_function():
        warnings.warn("deprecation warning", DeprecationWarning)
    
    # 定义一个发出多个警告的函数
    def _multiple_warning_function():
        warnings.warn("deprecation warning", DeprecationWarning)
        warnings.warn("deprecation warning")
    
    # 直接检查函数的行为
    assert_no_warnings(ignore_warnings(_warning_function))
    assert_no_warnings(ignore_warnings(_warning_function, category=DeprecationWarning))
    # 使用 pytest 的 warns 上下文来检查特定类型的警告
    with pytest.warns(DeprecationWarning):
        ignore_warnings(_warning_function, category=UserWarning)()
    
    # 使用 pytest 的 warns 上下文记录多个警告
    with pytest.warns() as record:
        ignore_warnings(_multiple_warning_function, category=FutureWarning)()
    assert len(record) == 2
    assert isinstance(record[0].message, DeprecationWarning)
    assert isinstance(record[1].message, UserWarning)
    
    with pytest.warns() as record:
        ignore_warnings(_multiple_warning_function, category=UserWarning)()
    assert len(record) == 1
    assert isinstance(record[0].message, DeprecationWarning)
    
    # 检查多种警告类型的情况
    assert_no_warnings(
        ignore_warnings(_warning_function, category=(DeprecationWarning, UserWarning))
    )
    
    # 检查装饰器的行为
    @ignore_warnings
    def decorator_no_warning():
        _warning_function()
        _multiple_warning_function()
    
    @ignore_warnings(category=(DeprecationWarning, UserWarning))
    def decorator_no_warning_multiple():
        _multiple_warning_function()
    
    @ignore_warnings(category=DeprecationWarning)
    def decorator_no_deprecation_warning():
        _warning_function()
    
    @ignore_warnings(category=UserWarning)
    def decorator_no_user_warning():
        _warning_function()
    
    @ignore_warnings(category=DeprecationWarning)
    def decorator_no_deprecation_multiple_warning():
        _multiple_warning_function()
    
    @ignore_warnings(category=UserWarning)
    def decorator_no_user_multiple_warning():
        _multiple_warning_function()
    
    assert_no_warnings(decorator_no_warning)
    assert_no_warnings(decorator_no_warning_multiple)
    assert_no_warnings(decorator_no_deprecation_warning)
    # 使用 pytest 的 warns 上下文来检查特定类型的警告
    with pytest.warns(DeprecationWarning):
        decorator_no_user_warning()
    # 使用 pytest 的 warns 上下文来检查特定类型的警告
    with pytest.warns(UserWarning):
        decorator_no_deprecation_multiple_warning()
    # 使用 pytest 的 warns 上下文来检查特定类型的警告
    with pytest.warns(DeprecationWarning):
        decorator_no_user_multiple_warning()
    
    # 检查上下文管理器的行为
    def context_manager_no_warning():
        with ignore_warnings():
            _warning_function()
    
    def context_manager_no_warning_multiple():
        with ignore_warnings(category=(DeprecationWarning, UserWarning)):
            _multiple_warning_function()
    
    def context_manager_no_deprecation_warning():
        with ignore_warnings(category=DeprecationWarning):
            _warning_function()
    
    def context_manager_no_user_warning():
        with ignore_warnings(category=UserWarning):
            _warning_function()
    # 定义一个使用 contextlib 模块中的 ignore_warnings 上下文管理器来忽略 DeprecationWarning 类别的警告的函数
    def context_manager_no_deprecation_multiple_warning():
        # 在 ignore_warnings 上下文中忽略 DeprecationWarning 警告
        with ignore_warnings(category=DeprecationWarning):
            # 调用 _multiple_warning_function 函数
            _multiple_warning_function()

    # 定义一个使用 contextlib 模块中的 ignore_warnings 上下文管理器来忽略 UserWarning 类别的警告的函数
    def context_manager_no_user_multiple_warning():
        # 在 ignore_warnings 上下文中忽略 UserWarning 警告
        with ignore_warnings(category=UserWarning):
            # 调用 _multiple_warning_function 函数
            _multiple_warning_function()

    # 调用 assert_no_warnings 函数，验证 context_manager_no_warning 函数不产生警告
    assert_no_warnings(context_manager_no_warning)
    # 调用 assert_no_warnings 函数，验证 context_manager_no_warning_multiple 函数不产生警告
    assert_no_warnings(context_manager_no_warning_multiple)
    # 调用 assert_no_warnings 函数，验证 context_manager_no_deprecation_warning 函数不产生 DeprecationWarning 警告
    assert_no_warnings(context_manager_no_deprecation_warning)

    # 在 pytest.warns 上下文中，检查 context_manager_no_user_warning 函数是否产生 DeprecationWarning 警告
    with pytest.warns(DeprecationWarning):
        context_manager_no_user_warning()

    # 在 pytest.warns 上下文中，检查 context_manager_no_deprecation_multiple_warning 函数是否产生 UserWarning 警告
    with pytest.warns(UserWarning):
        context_manager_no_deprecation_multiple_warning()

    # 在 pytest.warns 上下文中，检查 context_manager_no_user_multiple_warning 函数是否产生 DeprecationWarning 警告
    with pytest.warns(DeprecationWarning):
        context_manager_no_user_multiple_warning()

    # 定义一个 UserWarning 类的变量
    warning_class = UserWarning
    # 定义一个字符串匹配模式，用于匹配错误信息
    match = "'obj' should be a callable.+you should use 'category=UserWarning'"

    # 在 pytest.raises 上下文中，检查 silence_warnings_func 函数是否产生 ValueError 错误并匹配指定的错误信息
    with pytest.raises(ValueError, match=match):
        # 使用 ignore_warnings 函数创建一个新函数 silence_warnings_func，并在其上调用 _warning_function 函数
        silence_warnings_func = ignore_warnings(warning_class)(_warning_function)
        silence_warnings_func()

    # 在 pytest.raises 上下文中，检查 test 函数是否产生 ValueError 错误并匹配指定的错误信息
    with pytest.raises(ValueError, match=match):
        # 使用 ignore_warnings 装饰器修饰 test 函数
        @ignore_warnings(warning_class)
        def test():
            pass
class TestWarns(unittest.TestCase):
    # 定义测试类 TestWarns，继承自 unittest.TestCase

    def test_warn(self):
        # 测试方法 test_warn，用于测试警告

        def f():
            # 定义内部函数 f

            warnings.warn("yo")
            # 发出警告消息 "yo"

            return 3
            # 返回整数 3

        with pytest.raises(AssertionError):
            # 使用 pytest 检测是否会抛出 AssertionError 异常

            assert_no_warnings(f)
            # 调用 assert_no_warnings 函数，期望在函数 f 执行时不会有警告

        assert assert_no_warnings(lambda x: x, 1) == 1
        # 使用 lambda 函数调用 assert_no_warnings，检查输入参数为 1 时返回值是否为 1


# Tests for docstrings:


def f_ok(a, b):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a
    b : float
        Parameter b

    Returns
    -------
    c : list
        Parameter c
    """
    # 函数 f_ok，接受整数 a 和浮点数 b 作为参数，返回 c 列表
    c = a + b
    # 计算 a + b 并赋值给变量 c
    return c
    # 返回变量 c


def f_bad_sections(a, b):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a
    b : float
        Parameter b

    Results
    -------
    c : list
        Parameter c
    """
    # 函数 f_bad_sections，接受整数 a 和浮点数 b 作为参数，返回 c 列表
    c = a + b
    # 计算 a + b 并赋值给变量 c
    return c
    # 返回变量 c


def f_bad_order(b, a):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a
    b : float
        Parameter b

    Returns
    -------
    c : list
        Parameter c
    """
    # 函数 f_bad_order，接受整数 a 和浮点数 b 作为参数，返回 c 列表
    c = a + b
    # 计算 a + b 并赋值给变量 c
    return c
    # 返回变量 c


def f_too_many_param_docstring(a, b):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a
    b : int
        Parameter b
    c : int
        Parameter c

    Returns
    -------
    d : list
        Parameter c
    """
    # 函数 f_too_many_param_docstring，接受整数 a 和整数 b 作为参数，返回 d 列表
    d = a + b
    # 计算 a + b 并赋值给变量 d
    return d
    # 返回变量 d


def f_missing(a, b):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a

    Returns
    -------
    c : list
        Parameter c
    """
    # 函数 f_missing，接受整数 a 和整数 b 作为参数，返回 c 列表
    c = a + b
    # 计算 a + b 并赋值给变量 c
    return c
    # 返回变量 c


def f_check_param_definition(a, b, c, d, e):
    """Function f

    Parameters
    ----------
    a: int
        Parameter a
    b:
        Parameter b
    c :
        This is parsed correctly in numpydoc 1.2
    d:int
        Parameter d
    e
        No typespec is allowed without colon
    """
    # 函数 f_check_param_definition，接受五个参数 a, b, c, d, e
    return a + b + c + d
    # 返回参数 a, b, c, d 的和


class Klass:
    # 类 Klass

    def f_missing(self, X, y):
        # 类方法 f_missing，接受参数 self, X, y
        pass
        # 仅占位，无具体实现

    def f_bad_sections(self, X, y):
        """Function f

        Parameter
        ---------
        a : int
            Parameter a
        b : float
            Parameter b

        Results
        -------
        c : list
            Parameter c
        """
        # 类方法 f_bad_sections，接受参数 self, X, y
        pass
        # 仅占位，无具体实现


class MockEst:
    # 类 MockEst

    def __init__(self):
        """MockEstimator"""
        # 构造函数，初始化 MockEstimator 实例

    def fit(self, X, y):
        # 方法 fit，接受参数 self, X, y
        return X
        # 返回参数 X

    def predict(self, X):
        # 方法 predict，接受参数 self, X
        return X
        # 返回参数 X

    def predict_proba(self, X):
        # 方法 predict_proba，接受参数 self, X
        return X
        # 返回参数 X

    def score(self, X):
        # 方法 score，接受参数 self, X
        return 1.0
        # 返回浮点数 1.0


class MockMetaEstimator:
    # 类 MockMetaEstimator

    def __init__(self, delegate):
        """MetaEstimator to check if doctest on delegated methods work.

        Parameters
        ---------
        delegate : estimator
            Delegated estimator.
        """
        # 构造函数，初始化 MetaEstimator 实例，接受参数 delegate

        self.delegate = delegate
        # 将参数 delegate 赋值给实例变量 self.delegate

    @available_if(lambda self: hasattr(self.delegate, "predict"))
    # 使用装饰器 @available_if 检查 self.delegate 是否具有 predict 方法
    def predict(self, X):
        """This is available only if delegate has predict.

        Parameters
        ----------
        y : ndarray
            Parameter y
        """
        # 方法 predict，接受参数 self, X
        return self.delegate.predict(X)
        # 如果 self.delegate 有 predict 方法，则调用其 predict 方法并返回结果
    # 根据条件检查是否可用，如果委托对象有 score 方法，则可用
    @available_if(lambda self: hasattr(self.delegate, "score"))
    # 使用 deprecated 装饰器标记为已弃用，提供额外信息 "Testing a deprecated delegated method"
    @deprecated("Testing a deprecated delegated method")
    # score 方法，接收参数 X，但注释错误，应该接收参数 y
    def score(self, X):
        """This is available only if delegate has score.
    
        Parameters
        ---------
        y : ndarray
            Parameter y
        """
    
    # 根据条件检查是否可用，如果委托对象有 predict_proba 方法，则可用
    @available_if(lambda self: hasattr(self.delegate, "predict_proba"))
    # predict_proba 方法，接收参数 X，返回参数 X
    def predict_proba(self, X):
        """This is available only if delegate has predict_proba.
    
        Parameters
        ---------
        X : ndarray
            Parameter X
        """
        return X
    
    # 使用 deprecated 装饰器标记为已弃用，提供额外信息 "Testing deprecated function with wrong params"
    @deprecated("Testing deprecated function with wrong params")
    # fit 方法，接收参数 X 和 y，但注释错误，应该接收参数 X 和 y
    def fit(self, X, y):
        """Incorrect docstring but should not be tested"""
# 定义一个测试函数，用于检查文档字符串中的参数格式是否正确
def test_check_docstring_parameters():
    # 导入 pytest 模块，如果导入失败则跳过测试，并给出原因和最低版本要求
    pytest.importorskip(
        "numpydoc",
        reason="numpydoc is required to test the docstrings",
        minversion="1.2.0",
    )

    # 检查没有忽略任何参数的文档字符串参数格式是否正确
    incorrect = check_docstring_parameters(f_ok)
    assert incorrect == []

    # 检查忽略参数 'b' 后的文档字符串参数格式是否正确
    incorrect = check_docstring_parameters(f_ok, ignore=["b"])
    assert incorrect == []

    # 检查忽略参数 'b' 后，缺失参数的文档字符串参数格式是否正确
    incorrect = check_docstring_parameters(f_missing, ignore=["b"])
    assert incorrect == []

    # 测试异常情况：文档字符串中包含未知的 section 'Results'，预期抛出 RuntimeError 异常
    with pytest.raises(RuntimeError, match="Unknown section Results"):
        check_docstring_parameters(f_bad_sections)

    # 测试异常情况：类 Klass 的方法 f_bad_sections 中包含未知的 section 'Parameter'，预期抛出 RuntimeError 异常
    with pytest.raises(RuntimeError, match="Unknown section Parameter"):
        check_docstring_parameters(Klass.f_bad_sections)

    # 检查文档字符串参数定义格式是否正确，并验证错误信息列表
    incorrect = check_docstring_parameters(f_check_param_definition)

    # 创建一个 MockMetaEstimator 实例，用于模拟测试环境
    mock_meta = MockMetaEstimator(delegate=MockEst())
    mock_meta_name = mock_meta.__class__.__name__

    # 验证错误信息列表是否符合预期格式和内容
    assert incorrect == [
        (
            "sklearn.utils.tests.test_testing.f_check_param_definition There "
            "was no space between the param name and colon ('a: int')"
        ),
        (
            "sklearn.utils.tests.test_testing.f_check_param_definition There "
            "was no space between the param name and colon ('b:')"
        ),
        (
            "sklearn.utils.tests.test_testing.f_check_param_definition There "
            "was no space between the param name and colon ('d:int')"
        ),
    ]
    messages = [
        # 第一个错误消息条目
        [
            "In function: sklearn.utils.tests.test_testing.f_bad_order",
            (
                "There's a parameter name mismatch in function docstring w.r.t."
                " function signature, at index 0 diff: 'b' != 'a'"
            ),
            "Full diff:",
            "- ['b', 'a']",  # 函数签名中参数顺序错误的部分
            "+ ['a', 'b']",  # 函数文档字符串中正确的参数顺序
        ],
        # 第二个错误消息条目
        [
            "In function: "
            + "sklearn.utils.tests.test_testing.f_too_many_param_docstring",
            (
                "Parameters in function docstring have more items w.r.t. function"
                " signature, first extra item: c"
            ),
            "Full diff:",
            "- ['a', 'b']",  # 函数签名中参数列表
            "+ ['a', 'b', 'c']",  # 函数文档字符串中包含额外参数 'c'
            "?          +++++",  # 表示输出中有额外参数的指示
        ],
        # 第三个错误消息条目
        [
            "In function: sklearn.utils.tests.test_testing.f_missing",
            (
                "Parameters in function docstring have less items w.r.t. function"
                " signature, first missing item: b"
            ),
            "Full diff:",
            "- ['a', 'b']",  # 函数签名中参数列表
            "+ ['a']",  # 函数文档字符串中缺少参数 'b'
        ],
        # 第四个错误消息条目
        [
            "In function: sklearn.utils.tests.test_testing.Klass.f_missing",
            (
                "Parameters in function docstring have less items w.r.t. function"
                " signature, first missing item: X"
            ),
            "Full diff:",
            "- ['X', 'y']",  # 函数签名中参数列表
            "+ []",  # 函数文档字符串中没有参数
        ],
        # 第五个错误消息条目
        [
            "In function: "
            + f"sklearn.utils.tests.test_testing.{mock_meta_name}.predict",
            (
                "There's a parameter name mismatch in function docstring w.r.t."
                " function signature, at index 0 diff: 'X' != 'y'"
            ),
            "Full diff:",
            "- ['X']",  # 函数签名中参数名
            "?   ^",
            "+ ['y']",  # 函数文档字符串中的参数名
            "?   ^",
        ],
        # 第六个错误消息条目
        [
            "In function: "
            + f"sklearn.utils.tests.test_testing.{mock_meta_name}."
            + "predict_proba",
            "potentially wrong underline length... ",
            "Parameters ",
            "--------- in ",
        ],
        # 第七个错误消息条目
        [
            "In function: "
            + f"sklearn.utils.tests.test_testing.{mock_meta_name}.score",
            "potentially wrong underline length... ",
            "Parameters ",
            "--------- in ",
        ],
        # 第八个错误消息条目
        [
            "In function: " + f"sklearn.utils.tests.test_testing.{mock_meta_name}.fit",
            (
                "Parameters in function docstring have less items w.r.t. function"
                " signature, first missing item: X"
            ),
            "Full diff:",
            "- ['X', 'y']",  # 函数签名中参数列表
            "+ []",  # 函数文档字符串中没有参数
        ],
    ]
    # 使用 zip 函数逐个迭代 messages 和 f 列表中的元素
    # messages 是一组字符串列表
    # f 是包含函数和方法的列表，用于检查它们的参数文档字符串
    for msg, f in zip(
        messages,
        [
            f_bad_order,                 # 第一个函数或方法，检查其参数文档字符串
            f_too_many_param_docstring,  # 第二个函数或方法，检查其参数文档字符串
            f_missing,                   # 第三个函数或方法，检查其参数文档字符串
            Klass.f_missing,             # Klass 类的 f_missing 方法，检查其参数文档字符串
            mock_meta.predict,           # mock_meta 对象的 predict 方法，检查其参数文档字符串
            mock_meta.predict_proba,     # mock_meta 对象的 predict_proba 方法，检查其参数文档字符串
            mock_meta.score,             # mock_meta 对象的 score 方法，检查其参数文档字符串
            mock_meta.fit,               # mock_meta 对象的 fit 方法，检查其参数文档字符串
        ],
    ):
        # 检查文档字符串参数是否正确，如果不正确则引发异常
        incorrect = check_docstring_parameters(f)
        # 使用断言来验证消息与不正确的参数文档字符串是否匹配，如果不匹配则抛出异常
        assert msg == incorrect, '\n"%s"\n not in \n"%s"' % (msg, incorrect)
class RegistrationCounter:
    # 注册计数器类，用于记录调用次数
    def __init__(self):
        # 初始化调用次数为0
        self.nb_calls = 0

    # 定义__call__方法，使实例对象可调用
    def __call__(self, to_register_func):
        # 每次调用增加计数器值
        self.nb_calls += 1
        # 断言注册函数是_delete_folder
        assert to_register_func.func is _delete_folder


def check_memmap(input_array, mmap_data, mmap_mode="r"):
    # 断言mmap_data是np.memmap类型
    assert isinstance(mmap_data, np.memmap)
    # 根据mmap_mode确定是否可写
    writeable = mmap_mode != "r"
    # 断言mmap_data的可写属性与mmap_mode一致
    assert mmap_data.flags.writeable is writeable
    # 检查input_array与mmap_data是否相等
    np.testing.assert_array_equal(input_array, mmap_data)


def test_tempmemmap(monkeypatch):
    # 创建RegistrationCounter实例
    registration_counter = RegistrationCounter()
    # 使用monkeypatch设置atexit.register为registration_counter
    monkeypatch.setattr(atexit, "register", registration_counter)

    # 创建全1数组作为input_array
    input_array = np.ones(3)
    # 使用TempMemmap上下文管理器创建临时内存映射数据
    with TempMemmap(input_array) as data:
        # 检查内存映射数据
        check_memmap(input_array, data)
        # 获取临时文件夹路径
        temp_folder = os.path.dirname(data.filename)
    # 如果操作系统不是Windows，断言临时文件夹不存在
    if os.name != "nt":
        assert not os.path.exists(temp_folder)
    # 断言注册计数为1
    assert registration_counter.nb_calls == 1

    # 设置mmap_mode为'r+'
    mmap_mode = "r+"
    # 再次使用TempMemmap创建内存映射数据
    with TempMemmap(input_array, mmap_mode=mmap_mode) as data:
        # 检查内存映射数据
        check_memmap(input_array, data, mmap_mode=mmap_mode)
        # 获取临时文件夹路径
        temp_folder = os.path.dirname(data.filename)
    # 如果操作系统不是Windows，断言临时文件夹不存在
    if os.name != "nt":
        assert not os.path.exists(temp_folder)
    # 断言注册计数为2
    assert registration_counter.nb_calls == 2


@pytest.mark.xfail(_IS_WASM, reason="memmap not fully supported")
def test_create_memmap_backed_data(monkeypatch):
    # 创建RegistrationCounter实例
    registration_counter = RegistrationCounter()
    # 使用monkeypatch设置atexit.register为registration_counter
    monkeypatch.setattr(atexit, "register", registration_counter)

    # 创建全1数组作为input_array
    input_array = np.ones(3)
    # 创建基于内存映射的数据
    data = create_memmap_backed_data(input_array)
    # 检查内存映射数据
    check_memmap(input_array, data)
    # 断言注册计数为1
    assert registration_counter.nb_calls == 1

    # 获取带有临时文件夹的内存映射数据
    data, folder = create_memmap_backed_data(input_array, return_folder=True)
    # 检查内存映射数据
    check_memmap(input_array, data)
    # 断言文件夹路径与内存映射数据文件名的文件夹路径一致
    assert folder == os.path.dirname(data.filename)
    # 断言注册计数为2
    assert registration_counter.nb_calls == 2

    # 设置mmap_mode为'r+'
    mmap_mode = "r+"
    # 创建基于内存映射的数据
    data = create_memmap_backed_data(input_array, mmap_mode=mmap_mode)
    # 检查内存映射数据
    check_memmap(input_array, data, mmap_mode)
    # 断言注册计数为3
    assert registration_counter.nb_calls == 3

    # 创建包含多个数组的基于内存映射的数据列表
    input_list = [input_array, input_array + 1, input_array + 2]
    mmap_data_list = create_memmap_backed_data(input_list)
    # 遍历input_list和mmap_data_list，检查每个数组与对应的内存映射数据
    for input_array, data in zip(input_list, mmap_data_list):
        check_memmap(input_array, data)
    # 断言注册计数为4
    assert registration_counter.nb_calls == 4

    # 创建基于内存映射的输出数据和其他数据
    output_data, other = create_memmap_backed_data([input_array, "not-an-array"])
    # 检查内存映射数据
    check_memmap(input_array, output_data)
    # 断言other为"not-an-array"
    assert other == "not-an-array"
    # 创建一个包含不同数据类型和函数的元组列表
    [
        # ("list", list) 将字符串 "list" 与内置函数 list 绑定为元组的一部分
        ("list", list),
        # ("tuple", tuple) 将字符串 "tuple" 与内置函数 tuple 绑定为元组的一部分
        ("tuple", tuple),
        # ("array", np.ndarray) 将字符串 "array" 与 numpy 库中的 ndarray 类型绑定为元组的一部分
        ("array", np.ndarray),
        # ("sparse", sparse.csr_matrix) 将字符串 "sparse" 与 SciPy 库中的 CSR 稀疏矩阵类型绑定为元组的一部分
        ("sparse", sparse.csr_matrix),
        # using `zip` will only keep the available sparse containers
        # depending of the installed SciPy version
        # 使用 `zip` 函数，根据已安装的 SciPy 版本保留可用的稀疏容器
        *zip(["sparse_csr", "sparse_csr_array"], CSR_CONTAINERS),
        # 使用 `zip` 函数，将字符串 "sparse_csr" 与 CSR_CONTAINERS 中的稀疏容器类型绑定为元组的一部分
        # 使用 `zip` 函数，将字符串 "sparse_csr_array" 与 CSR_CONTAINERS 中的稀疏容器类型绑定为元组的一部分
        *zip(["sparse_csc", "sparse_csc_array"], CSC_CONTAINERS),
        # ("dataframe", lambda: pytest.importorskip("pandas").DataFrame) 将字符串 "dataframe" 与通过 pytest 动态导入的 pandas 库中的 DataFrame 类型的 lambda 函数绑定为元组的一部分
        ("dataframe", lambda: pytest.importorskip("pandas").DataFrame),
        # ("series", lambda: pytest.importorskip("pandas").Series) 将字符串 "series" 与通过 pytest 动态导入的 pandas 库中的 Series 类型的 lambda 函数绑定为元组的一部分
        ("series", lambda: pytest.importorskip("pandas").Series),
        # ("index", lambda: pytest.importorskip("pandas").Index) 将字符串 "index" 与通过 pytest 动态导入的 pandas 库中的 Index 类型的 lambda 函数绑定为元组的一部分
        ("index", lambda: pytest.importorskip("pandas").Index),
        # ("slice", slice) 将字符串 "slice" 与内置函数 slice 绑定为元组的一部分
        ("slice", slice),
    ],
# 使用 pytest 模块中的 mark.parametrize 装饰器，为以下的 test_convert_container 函数参数化测试用例
@pytest.mark.parametrize(
    "dtype, superdtype",
    [
        (np.int32, np.integer),  # 测试参数 dtype 是 np.int32 时，superdtype 是 np.integer
        (np.int64, np.integer),  # 测试参数 dtype 是 np.int64 时，superdtype 是 np.integer
        (np.float32, np.floating),  # 测试参数 dtype 是 np.float32 时，superdtype 是 np.floating
        (np.float64, np.floating),  # 测试参数 dtype 是 np.float64 时，superdtype 是 np.floating
    ],
)
# 定义测试函数 test_convert_container，用于验证容器转换为正确类型的数组以及正确的数据类型
def test_convert_container(
    constructor_name,
    container_type,
    dtype,
    superdtype,
):
    """Check that we convert the container to the right type of array with the
    right data type."""
    if constructor_name in ("dataframe", "polars", "series", "polars_series", "index"):
        # 延迟在函数内部导入 pandas/polars，以便仅跳过此测试而不是整个文件
        container_type = container_type()
    container = [0, 1]

    # 调用 _convert_container 函数，将 container 转换为指定类型的容器 container_converted
    container_converted = _convert_container(
        container,
        constructor_name,
        dtype=dtype,
    )
    # 断言 container_converted 的类型是 container_type
    assert isinstance(container_converted, container_type)

    if constructor_name in ("list", "tuple", "index"):
        # 对于 list 和 tuple 使用 Python 类型 dtype: int, float
        # 对于 pandas index 总是使用高精度类型: np.int64 和 np.float64
        assert np.issubdtype(type(container_converted[0]), superdtype)
    elif hasattr(container_converted, "dtype"):
        # 断言 container_converted 的数据类型是 dtype
        assert container_converted.dtype == dtype
    elif hasattr(container_converted, "dtypes"):
        # 断言 container_converted 的第一个数据类型是 dtype
        assert container_converted.dtypes[0] == dtype


# 定义测试函数 test_convert_container_categories_pandas，用于验证 pandas 中的类别转换
def test_convert_container_categories_pandas():
    # 导入 pytest 模块中的 importorskip 函数，确保 pandas 可用，否则跳过测试
    pytest.importorskip("pandas")
    # 调用 _convert_container 函数，将 [["x"]] 转换为 pandas 中的 dataframe
    df = _convert_container(
        [["x"]], "dataframe", ["A"], categorical_feature_names=["A"]
    )
    # 断言 dataframe 的第一个列的数据类型是 category
    assert df.dtypes.iloc[0] == "category"


# 定义测试函数 test_convert_container_categories_polars，用于验证 polars 中的类别转换
def test_convert_container_categories_polars():
    # 导入 pytest 模块中的 importorskip 函数，确保 polars 可用，否则跳过测试
    pl = pytest.importorskip("polars")
    # 调用 _convert_container 函数，将 [["x"]] 转换为 polars 中的 dataframe
    df = _convert_container([["x"]], "polars", ["A"], categorical_feature_names=["A"])
    # 断言 dataframe 的列名为 "A" 的数据类型是 polars 中的 Categorical
    assert df.schema["A"] == pl.Categorical()


# 定义测试函数 test_convert_container_categories_pyarrow，用于验证 pyarrow 中的类别转换
def test_convert_container_categories_pyarrow():
    # 导入 pytest 模块中的 importorskip 函数，确保 pyarrow 可用，否则跳过测试
    pa = pytest.importorskip("pyarrow")
    # 调用 _convert_container 函数，将 [["x"]] 转换为 pyarrow 中的 dataframe
    df = _convert_container([["x"]], "pyarrow", ["A"], categorical_feature_names=["A"])
    # 断言 dataframe 的第一个列的类型是 pyarrow 中的 DictionaryType
    assert type(df.schema[0].type) is pa.DictionaryType


# 使用 pytest.mark.skipif 装饰器，根据条件跳过测试
@pytest.mark.skipif(
    sp_version >= parse_version("1.8"),
    reason="sparse arrays are available as of scipy 1.8.0",
)
# 使用 pytest.mark.parametrize 装饰器，为以下的 test_convert_container_raise_when_sparray_not_available 函数参数化测试用例
@pytest.mark.parametrize("constructor_name", ["sparse_csr_array", "sparse_csc_array"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
# 定义测试函数 test_convert_container_raise_when_sparray_not_available，用于验证当稀疏数组不可用时是否抛出明确错误
def test_convert_container_raise_when_sparray_not_available(constructor_name, dtype):
    """Check that if we convert to sparse array but sparse array are not supported
    (scipy<1.8.0), we should raise an explicit error."""
    container = [0, 1]

    # 使用 pytest.raises 上下文管理器，断言在指定条件下会抛出 ValueError 错误
    with pytest.raises(
        ValueError,
        match=f"only available with scipy>=1.8.0, got {sp_version}",
    ):
        _convert_container(container, constructor_name, dtype=dtype)


# 定义测试函数 test_raises，用于测试 raises 上下文管理器的使用
def test_raises():
    # Tests for the raises context manager

    # Proper type, no match
    with raises(TypeError):
        raise TypeError()

    # Proper type, proper match
    # 使用 pytest 的 raises 上下文管理器来测试异常抛出情况，预期抛出 TypeError 异常并匹配特定错误信息
    with raises(TypeError, match="how are you") as cm:
        # 抛出一个 TypeError 异常并包含特定的错误消息
        raise TypeError("hello how are you")
    # 断言异常已经被抛出并且匹配预期的错误消息
    assert cm.raised_and_matched

    # 正确的异常类型，匹配包含多个模式的错误信息
    with raises(TypeError, match=["not this one", "how are you"]) as cm:
        # 抛出一个 TypeError 异常并包含特定的错误消息
        raise TypeError("hello how are you")
    # 断言异常已经被抛出并且匹配预期的错误消息
    assert cm.raised_and_matched

    # 错误的异常类型，不匹配任何错误消息
    with pytest.raises(ValueError, match="this will be raised"):
        with raises(TypeError) as cm:
            # 抛出一个 ValueError 异常
            raise ValueError("this will be raised")
    # 断言异常没有被抛出或者没有匹配预期的错误消息
    assert not cm.raised_and_matched

    # 错误的异常类型，不匹配任何错误消息，并且有一个错误消息
    with pytest.raises(AssertionError, match="the failure message"):
        with raises(TypeError, err_msg="the failure message") as cm:
            # 抛出一个 ValueError 异常
            raise ValueError()
    # 断言异常没有被抛出或者没有匹配预期的错误消息
    assert not cm.raised_and_matched

    # 错误的异常类型，匹配了一个错误消息但实际上是被忽略的
    with pytest.raises(ValueError, match="this will be raised"):
        with raises(TypeError, match="this is ignored") as cm:
            # 抛出一个 ValueError 异常
            raise ValueError("this will be raised")
    # 断言异常没有被抛出或者没有匹配预期的错误消息
    assert not cm.raised_and_matched

    # 正确的异常类型，但匹配了一个错误消息，但实际上错误消息不匹配
    with pytest.raises(
        AssertionError, match="should contain one of the following patterns"
    ):
        with raises(TypeError, match="hello") as cm:
            # 抛出一个 TypeError 异常但错误消息不匹配
            raise TypeError("Bad message")
    # 断言异常没有被抛出或者没有匹配预期的错误消息
    assert not cm.raised_and_matched

    # 正确的异常类型，但错误消息不匹配，并且有一个错误消息
    with pytest.raises(AssertionError, match="the failure message"):
        with raises(TypeError, match="hello", err_msg="the failure message") as cm:
            # 抛出一个 TypeError 异常但错误消息不匹配
            raise TypeError("Bad message")
    # 断言异常没有被抛出或者没有匹配预期的错误消息
    assert not cm.raised_and_matched

    # 没有抛出异常，使用默认的 may_pass=False
    with pytest.raises(AssertionError, match="Did not raise"):
        with raises(TypeError) as cm:
            # 什么都不做，没有抛出异常
            pass
    # 断言异常没有被抛出或者没有匹配预期的错误消息
    assert not cm.raised_and_matched

    # 没有抛出异常，但 may_pass=True
    with raises(TypeError, match="hello", may_pass=True) as cm:
        # 什么都不做，没有抛出异常
        pass  # 仍然是正确的
    # 断言异常没有被抛出或者没有匹配预期的错误消息
    assert not cm.raised_and_matched

    # 多个异常类型的情况下：
    with raises((TypeError, ValueError)):
        # 抛出一个 TypeError 异常
        raise TypeError()
    with raises((TypeError, ValueError)):
        # 抛出一个 ValueError 异常
        raise ValueError()
    with pytest.raises(AssertionError):
        with raises((TypeError, ValueError)):
            # 什么都不做，没有抛出异常
            pass
# 定义一个测试函数，用于测试带有特定相对容差的 assert_allclose 函数
def test_float32_aware_assert_allclose():
    # 对于 float32 类型的输入，相对容差设置为 1e-4
    assert_allclose(np.array([1.0 + 2e-5], dtype=np.float32), 1.0)
    # 使用 pytest 来检查是否抛出 AssertionError
    with pytest.raises(AssertionError):
        assert_allclose(np.array([1.0 + 2e-4], dtype=np.float32), 1.0)

    # 对于其他类型的输入，相对容差保持为 1e-7，与原始 numpy 版本相同
    assert_allclose(np.array([1.0 + 2e-8], dtype=np.float64), 1.0)
    with pytest.raises(AssertionError):
        assert_allclose(np.array([1.0 + 2e-7], dtype=np.float64), 1.0)

    # 默认情况下，即使对于 float32 类型，atol（绝对容差）也保持为 0.0
    with pytest.raises(AssertionError):
        assert_allclose(np.array([1e-5], dtype=np.float32), 0.0)
    # 使用特定的 atol（2e-5）来测试 assert_allclose 函数
    assert_allclose(np.array([1e-5], dtype=np.float32), 0.0, atol=2e-5)


# 使用 pytest 的 xfails 标记来定义一个测试函数，测试在 WASM 环境下无法启动子进程的情况
@pytest.mark.xfail(_IS_WASM, reason="cannot start subprocess")
def test_assert_run_python_script_without_output():
    # 测试一个不会有输出的 Python 脚本
    code = "x = 1"
    assert_run_python_script_without_output(code)

    # 测试一个会有输出的 Python 脚本，并使用 AssertionError 来验证期望没有输出
    code = "print('something to stdout')"
    with pytest.raises(AssertionError, match="Expected no output"):
        assert_run_python_script_without_output(code)

    # 测试一个会有输出的 Python 脚本，并使用 AssertionError 来验证输出内容不符合预期
    code = "print('something to stdout')"
    with pytest.raises(
        AssertionError,
        match="output was not supposed to match.+got.+something to stdout",
    ):
        assert_run_python_script_without_output(code, pattern="to.+stdout")

    # 测试一个会有输出的 Python 脚本（输出到 stderr），并使用 AssertionError 来验证输出内容不符合预期
    code = "\n".join(["import sys", "print('something to stderr', file=sys.stderr)"])
    with pytest.raises(
        AssertionError,
        match="output was not supposed to match.+got.+something to stderr",
    ):
        assert_run_python_script_without_output(code, pattern="to.+stderr")


# 使用 pytest 的 parametrize 标记来定义一个测试函数，测试将稀疏容器从一种格式转换为另一种格式的功能
@pytest.mark.parametrize(
    "constructor_name",
    [
        "sparse_csr",
        "sparse_csc",
        pytest.param(
            "sparse_csr_array",
            marks=pytest.mark.skipif(
                sp_version < parse_version("1.8"),
                reason="sparse arrays are available as of scipy 1.8.0",
            ),
        ),
        pytest.param(
            "sparse_csc_array",
            marks=pytest.mark.skipif(
                sp_version < parse_version("1.8"),
                reason="sparse arrays are available as of scipy 1.8.0",
            ),
        ),
    ],
)
def test_convert_container_sparse_to_sparse(constructor_name):
    """非回归测试，检查我们是否能够将稀疏容器从一种格式转换为另一种格式。"""
    # 创建一个稀疏矩阵 X_sparse，以 CSR 格式存储，并调用 _convert_container 函数
    X_sparse = sparse.random(10, 10, density=0.1, format="csr")
    _convert_container(X_sparse, constructor_name)


# 定义一个函数，用于检查警告信息是否应当被视作错误处理
def check_warnings_as_errors(warning_info, warnings_as_errors):
    # 如果警告信息的处理动作为 "error" 并且 warnings_as_errors 为 True，则期望抛出特定的警告类别和消息内容的异常
    if warning_info.action == "error" and warnings_as_errors:
        with pytest.raises(warning_info.category, match=warning_info.message):
            # 发出警告，同时指定警告的类别和消息内容
            warnings.warn(
                message=warning_info.message,
                category=warning_info.category,
            )
    # 如果警告信息的操作为 "ignore"
    if warning_info.action == "ignore":
        # 使用警告捕获上下文管理器，记录警告信息
        with warnings.catch_warnings(record=True) as record:
            # 获取警告消息
            message = warning_info.message
            # 当消息中包含 "Pyarrow" 时，进行特殊处理
            if "Pyarrow" in message:
                message = "\nPyarrow will become a required dependency"
    
            # 发出警告
            warnings.warn(
                message=message,
                category=warning_info.category,
            )
            
            # 如果设置了将警告视为错误，确保没有记录到任何警告
            assert len(record) == 0 if warnings_as_errors else 1
            # 如果有记录到警告
            if record:
                # 确保第一个记录的消息与发出的消息一致
                assert str(record[0].message) == message
                # 确保第一个记录的警告类别与给定的类别一致
                assert record[0].category == warning_info.category
# 使用 pytest 的 mark.parametrize 装饰器为函数 test_sklearn_warnings_as_errors 添加参数化测试
@pytest.mark.parametrize("warning_info", _get_warnings_filters_info_list())
def test_sklearn_warnings_as_errors(warning_info):
    # 从环境变量中获取 SKLEARN_WARNINGS_AS_ERRORS 的值，将其转换成布尔类型
    warnings_as_errors = os.environ.get("SKLEARN_WARNINGS_AS_ERRORS", "0") != "0"
    # 调用函数 check_warnings_as_errors 来检查警告是否作为错误处理
    check_warnings_as_errors(warning_info, warnings_as_errors=warnings_as_errors)


# 使用 pytest 的 mark.parametrize 装饰器为函数 test_turn_warnings_into_errors 添加参数化测试
@pytest.mark.parametrize("warning_info", _get_warnings_filters_info_list())
def test_turn_warnings_into_errors(warning_info):
    # 使用 warnings.catch_warnings() 上下文管理器来捕获警告
    with warnings.catch_warnings():
        # 调用 turn_warnings_into_errors 函数，将所有警告转换成错误
        turn_warnings_into_errors()
        # 调用函数 check_warnings_as_errors 来检查警告是否作为错误处理，此处预期设置为 True
        check_warnings_as_errors(warning_info, warnings_as_errors=True)
```