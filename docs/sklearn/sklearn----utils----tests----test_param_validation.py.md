# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_param_validation.py`

```
# 从 numbers 模块导入 Integral（整数）和 Real（实数）类型
from numbers import Integral, Real

# 导入 numpy 库，并用 np 别名表示
import numpy as np

# 导入 pytest 库，用于单元测试
import pytest

# 导入 scipy.sparse 库中的 csr_matrix 类
from scipy.sparse import csr_matrix

# 导入 sklearn 库中的配置和获取配置函数
from sklearn._config import config_context, get_config

# 导入 sklearn 库中的基本估算器类 BaseEstimator 和 _fit_context 函数
from sklearn.base import BaseEstimator, _fit_context

# 导入 sklearn 库中的 LeaveOneOut 类，用于交叉验证
from sklearn.model_selection import LeaveOneOut

# 导入 sklearn 库中的工具函数 deprecated 和 _param_validation 模块下的各类
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
    HasMethods,
    Hidden,
    Interval,
    InvalidParameterError,
    MissingValues,
    Options,
    RealNotInt,
    StrOptions,
    _ArrayLikes,
    _Booleans,
    _Callables,
    _CVObjects,
    _InstancesOf,
    _IterablesNotString,
    _NanConstraint,
    _NoneConstraint,
    _PandasNAConstraint,
    _RandomStates,
    _SparseMatrices,
    _VerboseHelper,
    generate_invalid_param_val,
    generate_valid_param,
    make_constraint,
    validate_params,
)

# 从 sklearn.utils.fixes 模块导入 CSR_CONTAINERS 常量
from sklearn.utils.fixes import CSR_CONTAINERS


# 以下是测试辅助函数和类定义

# 使用 validate_params 装饰器验证函数参数，指定参数约束为 a, b, c, d 均为实数
@validate_params(
    {"a": [Real], "b": [Real], "c": [Real], "d": [Real]},
    prefer_skip_nested_validation=True,
)
def _func(a, b=0, *args, c, d=0, **kwargs):
    """A function to test the validation of functions."""


# 定义一个用于测试 _InstancesOf 约束和方法验证的类
class _Class:
    """A class to test the _InstancesOf constraint and the validation of methods."""

    # 使用 validate_params 装饰器验证方法参数，指定参数约束为 a 为实数
    @validate_params({"a": [Real]}, prefer_skip_nested_validation=True)
    def _method(self, a):
        """A validated method"""

    # 使用 deprecated 装饰器标记为弃用方法，并使用 validate_params 装饰器验证方法参数，指定参数约束为 a 为实数
    @deprecated()
    @validate_params({"a": [Real]}, prefer_skip_nested_validation=True)
    def _deprecated_method(self, a):
        """A deprecated validated method"""


# 定义一个用于测试估算器参数验证的类
class _Estimator(BaseEstimator):
    """An estimator to test the validation of estimator parameters."""

    # 定义参数约束字典，a 必须为实数
    _parameter_constraints: dict = {"a": [Real]}

    # 初始化方法，接受参数 a
    def __init__(self, a):
        self.a = a

    # 使用 _fit_context 装饰器标记 fit 方法，表示跳过嵌套验证
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X=None, y=None):
        pass


# 使用 pytest 的参数化标记，测试 Interval 类的不同参数类型（整数和实数）下的取值范围
@pytest.mark.parametrize("interval_type", [Integral, Real])
def test_interval_range(interval_type):
    """Check the range of values depending on closed."""

    # 创建 Interval 对象，指定取值范围和闭合方式为左闭
    interval = Interval(interval_type, -2, 2, closed="left")
    assert -2 in interval  # 断言 -2 在 interval 内
    assert 2 not in interval  # 断言 2 不在 interval 内

    # 创建 Interval 对象，指定取值范围和闭合方式为右闭
    interval = Interval(interval_type, -2, 2, closed="right")
    assert -2 not in interval  # 断言 -2 不在 interval 内
    assert 2 in interval  # 断言 2 在 interval 内

    # 创建 Interval 对象，指定取值范围和闭合方式为两侧闭合
    interval = Interval(interval_type, -2, 2, closed="both")
    assert -2 in interval  # 断言 -2 在 interval 内
    assert 2 in interval  # 断言 2 在 interval 内

    # 创建 Interval 对象，指定取值范围和闭合方式为两侧开放
    interval = Interval(interval_type, -2, 2, closed="neither")
    assert -2 not in interval  # 断言 -2 不在 interval 内
    assert 2 not in interval  # 断言 2 不在 interval 内


# 使用 pytest 的参数化标记，测试 Interval 类对大整数的处理
@pytest.mark.parametrize("interval_type", [Integral, Real])
def test_interval_large_integers(interval_type):
    """Check that Interval constraint work with large integers.

    non-regression test for #26648.
    """

    # 创建 Interval 对象，指定取值范围和闭合方式为两侧开放，验证大整数是否在范围内
    interval = Interval(interval_type, 0, 2, closed="neither")
    assert 2**65 not in interval  # 断言 2 的 65 次方不在 interval 内
    assert 2**128 not in interval  # 断言 2 的 128 次方不在 interval 内
    assert float(2**65) not in interval  # 断言 float(2 的 65 次方) 不在 interval 内
    assert float(2**128) not in interval  # 断言 float(2 的 128 次方) 不在 interval 内

    # 创建 Interval 对象，指定取值范围和闭合方式为两侧开放，验证超大整数是否在范围内
    interval = Interval(interval_type, 0, 2**128, closed="neither")
    assert 2**65 in interval  # 断言 2 的 65 次方在 interval 内
    # 断言：检查 2 的 128 次方不在 interval 中
    assert 2**128 not in interval
    
    # 断言：检查浮点数 2 的 65 次方在 interval 中
    assert float(2**65) in interval
    
    # 断言：检查浮点数 2 的 128 次方不在 interval 中
    assert float(2**128) not in interval
    
    # 断言：检查 2 的 1024 次方不在 interval 中
    assert 2**1024 not in interval
# 定义一个测试函数，用于验证实数区间对象的基本属性和行为
def test_interval_inf_in_bounds():
    """Check that inf is included iff a bound is closed and set to None.

    Only valid for real intervals.
    """
    # 创建一个右闭区间对象，左边界为0，右边界为无穷大，右边界闭合
    interval = Interval(Real, 0, None, closed="right")
    assert np.inf in interval  # 断言无穷大在区间内

    # 创建一个左闭区间对象，左边界为无穷小，右边界为0，左边界闭合
    interval = Interval(Real, None, 0, closed="left")
    assert -np.inf in interval  # 断言负无穷大在区间内

    # 创建一个无穷区间对象，无具体边界设定，非闭合
    interval = Interval(Real, None, None, closed="neither")
    assert np.inf not in interval  # 断言无穷大不在区间内
    assert -np.inf not in interval  # 断言负无穷大不在区间内


# 使用pytest的参数化装饰器，对不同的区间对象执行相同的测试
@pytest.mark.parametrize(
    "interval",
    [Interval(Real, 0, 1, closed="left"), Interval(Real, None, None, closed="both")],
)
def test_nan_not_in_interval(interval):
    """Check that np.nan is not in any interval."""
    assert np.nan not in interval  # 断言NaN不在区间内


# 使用pytest的参数化装饰器，对不同参数组合执行区间对象构造错误测试
@pytest.mark.parametrize(
    "params, error, match",
    [
        (
            {"type": Integral, "left": 1.0, "right": 2, "closed": "both"},
            TypeError,
            r"Expecting left to be an int for an interval over the integers",
        ),
        (
            {"type": Integral, "left": 1, "right": 2.0, "closed": "neither"},
            TypeError,
            "Expecting right to be an int for an interval over the integers",
        ),
        (
            {"type": Integral, "left": None, "right": 0, "closed": "left"},
            ValueError,
            r"left can't be None when closed == left",
        ),
        (
            {"type": Integral, "left": 0, "right": None, "closed": "right"},
            ValueError,
            r"right can't be None when closed == right",
        ),
        (
            {"type": Integral, "left": 1, "right": -1, "closed": "both"},
            ValueError,
            r"right can't be less than left",
        ),
    ],
)
def test_interval_errors(params, error, match):
    """Check that informative errors are raised for invalid combination of parameters"""
    # 使用pytest断言检查期望的错误和匹配的错误消息
    with pytest.raises(error, match=match):
        Interval(**params)


# 定义一个简单的测试函数，用于验证字符串选项约束的基本行为
def test_stroptions():
    """Sanity check for the StrOptions constraint"""
    # 创建一个字符串选项对象，包含 {"a", "b", "c"}，其中 "c" 被标记为过时选项
    options = StrOptions({"a", "b", "c"}, deprecated={"c"})
    assert options.is_satisfied_by("a")  # 断言 "a" 在选项内
    assert options.is_satisfied_by("c")  # 断言 "c" 在选项内
    assert not options.is_satisfied_by("d")  # 断言 "d" 不在选项内

    assert "'c' (deprecated)" in str(options)  # 断言字符串中包含 "'c' (deprecated)"


# 定义一个简单的测试函数，用于验证实数选项约束的基本行为
def test_options():
    """Sanity check for the Options constraint"""
    # 创建一个实数选项对象，包含 {-0.5, 0.5, np.inf}，其中 -0.5 被标记为过时选项
    options = Options(Real, {-0.5, 0.5, np.inf}, deprecated={-0.5})
    assert options.is_satisfied_by(-0.5)  # 断言 -0.5 在选项内
    assert options.is_satisfied_by(np.inf)  # 断言 np.inf 在选项内
    assert not options.is_satisfied_by(1.23)  # 断言 1.23 不在选项内

    assert "-0.5 (deprecated)" in str(options)  # 断言字符串中包含 "-0.5 (deprecated)"


# 使用pytest的参数化装饰器，对不同类型及其名称执行实例类型约束的字符串表示测试
@pytest.mark.parametrize(
    "type, expected_type_name",
    [
        (int, "int"),
        (Integral, "int"),
        (Real, "float"),
        (np.ndarray, "numpy.ndarray"),
    ],
)
def test_instances_of_type_human_readable(type, expected_type_name):
    """Check the string representation of the _InstancesOf constraint."""
    constraint = _InstancesOf(type)
    assert str(constraint) == f"an instance of '{expected_type_name}'"


# 定义一个未完成的测试函数，以后可以添加更多功能测试
def test_hasmethods():
    """Placeholder for additional tests on HasMethods constraint."""
    pass
    """Check the HasMethods constraint."""

    # 创建一个 HasMethods 约束对象，指定需要检查的方法列表为 ["a", "b"]
    constraint = HasMethods(["a", "b"])

    # 定义一个满足约束的类 _Good
    class _Good:
        # 实现约束要求的方法 a
        def a(self):
            pass  # pragma: no cover

        # 实现约束要求的方法 b
        def b(self):
            pass  # pragma: no cover

    # 定义一个不完全满足约束的类 _Bad
    class _Bad:
        # 只实现了约束要求的方法 a，没有实现方法 b
        def a(self):
            pass  # pragma: no cover

    # 断言 _Good 类的实例满足约束
    assert constraint.is_satisfied_by(_Good())

    # 断言 _Bad 类的实例不满足约束
    assert not constraint.is_satisfied_by(_Bad())

    # 断言约束对象的字符串表示应为 "an object implementing 'a' and 'b'"
    assert str(constraint) == "an object implementing 'a' and 'b'"
@pytest.mark.parametrize(
    "constraint",
    [  # 参数化测试，依次传入不同的约束条件
        Interval(Real, None, 0, closed="left"),  # 实数区间约束，左闭右开
        Interval(Real, 0, None, closed="left"),  # 实数区间约束，左闭右开
        Interval(Real, None, None, closed="neither"),  # 实数区间约束，两端均不闭
        StrOptions({"a", "b", "c"}),  # 字符串选项约束，限定在集合 {"a", "b", "c"} 内
        MissingValues(),  # 缺失值约束，可以是任何缺失值
        MissingValues(numeric_only=True),  # 缺失值约束，仅限于数值类型的缺失值
        _VerboseHelper(),  # 详细帮助类，辅助测试
        HasMethods("fit"),  # 具有方法 "fit" 的对象约束
        _IterablesNotString(),  # 非字符串的可迭代对象约束
        _CVObjects(),  # 交叉验证对象约束
    ],
)
def test_generate_invalid_param_val(constraint):
    """Check that the value generated does not satisfy the constraint"""
    bad_value = generate_invalid_param_val(constraint)  # 生成不满足约束条件的值
    assert not constraint.is_satisfied_by(bad_value)  # 断言生成的值不满足约束条件


@pytest.mark.parametrize(
    "integer_interval, real_interval",
    # 创建一个包含多个元组的列表，每个元组包含两个Interval对象
    [
        # 第一个元组
        (
            # 创建一个整数区间，从无穷小到3（右闭区间）
            Interval(Integral, None, 3, closed="right"),
            # 创建一个非整数实数区间，从-5到5（两端闭区间）
            Interval(RealNotInt, -5, 5, closed="both"),
        ),
        # 第二个元组
        (
            # 创建一个整数区间，从无穷小到3（右闭区间）
            Interval(Integral, None, 3, closed="right"),
            # 创建一个非整数实数区间，从-5到5（左开右闭区间）
            Interval(RealNotInt, -5, 5, closed="neither"),
        ),
        # 第三个元组
        (
            # 创建一个整数区间，从无穷小到3（右闭区间）
            Interval(Integral, None, 3, closed="right"),
            # 创建一个非整数实数区间，从4到5（两端闭区间）
            Interval(RealNotInt, 4, 5, closed="both"),
        ),
        # 后续元组按此类推...
    ],
)
def test_generate_invalid_param_val_2_intervals(integer_interval, real_interval):
    """检查生成的值不满足任何一个区间约束条件。"""
    # 生成一个不满足实数区间约束条件的值
    bad_value = generate_invalid_param_val(constraint=real_interval)
    # 断言该值不满足实数区间约束条件
    assert not real_interval.is_satisfied_by(bad_value)
    # 断言该值不满足整数区间约束条件
    assert not integer_interval.is_satisfied_by(bad_value)

    # 生成一个不满足整数区间约束条件的值
    bad_value = generate_invalid_param_val(constraint=integer_interval)
    # 断言该值不满足实数区间约束条件
    assert not real_interval.is_satisfied_by(bad_value)
    # 断言该值不满足整数区间约束条件
    assert not integer_interval.is_satisfied_by(bad_value)


@pytest.mark.parametrize(
    "constraint",
    [
        _ArrayLikes(),
        _InstancesOf(list),
        _Callables(),
        _NoneConstraint(),
        _RandomStates(),
        _SparseMatrices(),
        _Booleans(),
        Interval(Integral, None, None, closed="neither"),
    ],
)
def test_generate_invalid_param_val_all_valid(constraint):
    """检查当约束条件下没有无效值时，函数是否引发 NotImplementedError。"""
    # 使用 pytest 断言来检测是否引发 NotImplementedError 异常
    with pytest.raises(NotImplementedError):
        generate_invalid_param_val(constraint)


@pytest.mark.parametrize(
    "constraint",
    [
        _ArrayLikes(),
        _Callables(),
        _InstancesOf(list),
        _NoneConstraint(),
        _RandomStates(),
        _SparseMatrices(),
        _Booleans(),
        _VerboseHelper(),
        MissingValues(),
        MissingValues(numeric_only=True),
        StrOptions({"a", "b", "c"}),
        Options(Integral, {1, 2, 3}),
        Interval(Integral, None, None, closed="neither"),
        Interval(Integral, 0, 10, closed="neither"),
        Interval(Integral, 0, None, closed="neither"),
        Interval(Integral, None, 0, closed="neither"),
        Interval(Real, 0, 1, closed="neither"),
        Interval(Real, 0, None, closed="both"),
        Interval(Real, None, 0, closed="right"),
        HasMethods("fit"),
        _IterablesNotString(),
        _CVObjects(),
    ],
)
def test_generate_valid_param(constraint):
    """检查生成的值是否满足约束条件。"""
    # 生成符合约束条件的值
    value = generate_valid_param(constraint)
    # 断言生成的值确实满足约束条件
    assert constraint.is_satisfied_by(value)


@pytest.mark.parametrize(
    "constraint_declaration, value",
    [
        # 元组1：包含实数区间对象（闭区间[0, 1]），关联值为浮点数0.42
        (Interval(Real, 0, 1, closed="both"), 0.42),
        # 元组2：包含整数区间对象（半开区间[0, ∞)，左开右闭），关联值为整数42
        (Interval(Integral, 0, None, closed="neither"), 42),
        # 元组3：包含字符串选项对象（包含 {"a", "b", "c"}），关联值为字符串"b"
        (StrOptions({"a", "b", "c"}), "b"),
        # 元组4：包含选项对象（关联到类型type，包含 {np.float32, np.float64}），关联值为np.float64类型
        (Options(type, {np.float32, np.float64}), np.float64),
        # 元组5：包含可调用对象callable，关联值为一个将输入加1的lambda函数
        (callable, lambda x: x + 1),
        # 元组6：空值元组，关联值为None
        (None, None),
        # 元组7：描述为"array-like"，关联值为包含两个列表的列表 [[1, 2], [3, 4]]
        ("array-like", [[1, 2], [3, 4]]),
        # 元组8：描述为"array-like"，关联值为包含两个列表的NumPy数组
        ("array-like", np.array([[1, 2], [3, 4]])),
        # 元组9：描述为"sparse matrix"，关联值为稀疏矩阵CSR格式表示的[[1, 2], [3, 4]]
        ("sparse matrix", csr_matrix([[1, 2], [3, 4]])),
        # 元组10-12：描述为"sparse matrix"，关联值为通过CSR_CONTAINERS中的不同容器生成的稀疏矩阵
        *[
            ("sparse matrix", container([[1, 2], [3, 4]]))
            for container in CSR_CONTAINERS
        ],
        # 元组13：描述为"random_state"，关联值为整数0
        ("random_state", 0),
        # 元组14：描述为"random_state"，关联值为使用种子0创建的NumPy随机状态对象
        ("random_state", np.random.RandomState(0)),
        # 元组15：描述为"random_state"，关联值为None
        ("random_state", None),
        # 元组16：描述为_Class类的实例化对象，关联值为_Class类的一个实例
        (_Class, _Class()),
        # 元组17：描述为整数类型，关联值为整数1
        (int, 1),
        # 元组18：描述为实数类型，关联值为浮点数0.5
        (Real, 0.5),
        # 元组19：描述为"boolean"，关联值为布尔值False
        ("boolean", False),
        # 元组20：描述为"verbose"，关联值为整数1
        ("verbose", 1),
        # 元组21-27：描述为"nan"或"MissingValues"，关联值为不同类型的缺失值表示
        ("nan", np.nan),
        (MissingValues(), -1),
        (MissingValues(), -1.0),
        (MissingValues(), 2**1028),
        (MissingValues(), None),
        (MissingValues(), float("nan")),
        (MissingValues(), np.nan),
        (MissingValues(), "missing"),
        # 元组28：描述为具有"fit"方法的对象，关联值为具有属性a=0的估计器对象
        (HasMethods("fit"), _Estimator(a=0)),
        # 元组29：描述为"cv_object"，关联值为整数5
        ("cv_object", 5),
    ],
# 测试函数，用于检查给定约束是否满足
def test_is_satisfied_by(constraint_declaration, value):
    # 创建约束对象
    constraint = make_constraint(constraint_declaration)
    # 断言给定值是否满足约束
    assert constraint.is_satisfied_by(value)


# 使用 pytest 的 parametrize 装饰器多参数化测试用例
@pytest.mark.parametrize(
    "constraint_declaration, expected_constraint_class",
    [
        # 测试不同约束声明的情况下，make_constraint 返回对应的约束类
        (Interval(Real, 0, 1, closed="both"), Interval),
        (StrOptions({"option1", "option2"}), StrOptions),
        (Options(Real, {0.42, 1.23}), Options),
        ("array-like", _ArrayLikes),
        ("sparse matrix", _SparseMatrices),
        ("random_state", _RandomStates),
        (None, _NoneConstraint),
        (callable, _Callables),
        (int, _InstancesOf),
        ("boolean", _Booleans),
        ("verbose", _VerboseHelper),
        (MissingValues(numeric_only=True), MissingValues),
        (HasMethods("fit"), HasMethods),
        ("cv_object", _CVObjects),
        ("nan", _NanConstraint),
    ],
)
def test_make_constraint(constraint_declaration, expected_constraint_class):
    # 根据约束声明创建约束对象
    constraint = make_constraint(constraint_declaration)
    # 断言创建的约束对象类型是否与预期的约束类相同
    assert constraint.__class__ is expected_constraint_class


# 测试当传递未知约束时是否引发适当的错误信息
def test_make_constraint_unknown():
    # 使用 pytest 检查是否引发值错误并包含"Unknown constraint"的错误信息
    with pytest.raises(ValueError, match="Unknown constraint"):
        make_constraint("not a valid constraint")


# 测试 validate_params 函数在不同参数传递方式下是否正常工作
def test_validate_params():
    # 测试在错误参数情况下是否引发 InvalidParameterError 错误
    with pytest.raises(
        InvalidParameterError, match="The 'a' parameter of _func must be"
    ):
        _func("wrong", c=1)

    with pytest.raises(
        InvalidParameterError, match="The 'b' parameter of _func must be"
    ):
        _func(*[1, "wrong"], c=1)

    with pytest.raises(
        InvalidParameterError, match="The 'c' parameter of _func must be"
    ):
        _func(1, **{"c": "wrong"})

    with pytest.raises(
        InvalidParameterError, match="The 'd' parameter of _func must be"
    ):
        _func(1, c=1, d="wrong")

    # 检查在存在额外位置参数和关键字参数时的错误情况
    with pytest.raises(
        InvalidParameterError, match="The 'b' parameter of _func must be"
    ):
        _func(0, *["wrong", 2, 3], c=4, **{"e": 5})

    with pytest.raises(
        InvalidParameterError, match="The 'c' parameter of _func must be"
    ):
        _func(0, *[1, 2, 3], c="four", **{"e": 5})


# 测试 validate_params 函数在存在无约束的参数情况下是否正常工作
def test_validate_params_missing_params():
    # 定义一个带有 validate_params 装饰器的函数，测试调用是否正常
    @validate_params({"a": [int]}, prefer_skip_nested_validation=True)
    def func(a, b):
        pass

    func(1, 2)


# 测试 validate_params 函数是否能够装饰函数
def test_decorate_validated_function():
    # 使用 deprecated 装饰器来装饰 _func 函数，测试是否正常工作
    decorated_function = deprecated()(_func)
    # 使用 pytest 的 warn 函数检查未来警告，并验证是否匹配指定的警告消息
    with pytest.warns(FutureWarning, match="Function _func is deprecated"):
        # 调用被装饰后的函数 decorated_function，并期望触发未来警告
        decorated_function(1, 2, c=3)
    
    # 外层装饰器不会干扰验证过程
    with pytest.warns(FutureWarning, match="Function _func is deprecated"):
        # 使用 pytest 的 raises 函数检查是否引发了指定异常，并验证异常消息
        with pytest.raises(
            InvalidParameterError, match=r"The 'c' parameter of _func must be"
        ):
            # 调用 decorated_function 函数，传入参数 1, 2, c="wrong"，期望引发 InvalidParameterError 异常
            decorated_function(1, 2, c="wrong")
# 定义一个测试函数，验证 validate_params 方法能够处理方法参数
def test_validate_params_method():
    """Check that validate_params works with methods"""
    # 使用 pytest 的断言检查，验证当传递错误参数时，是否会抛出 InvalidParameterError 异常，并且异常消息匹配特定的模式
    with pytest.raises(
        InvalidParameterError, match="The 'a' parameter of _Class._method must be"
    ):
        _Class()._method("wrong")

    # 对已废弃的方法进行验证，使用 pytest.warns 检查是否会收到 FutureWarning 警告，并且在参数错误时是否会抛出异常
    with pytest.warns(FutureWarning, match="Function _deprecated_method is deprecated"):
        with pytest.raises(
            InvalidParameterError,
            match="The 'a' parameter of _Class._deprecated_method must be",
        ):
            _Class()._deprecated_method("wrong")


# 定义一个测试函数，验证 validate_params 方法能够处理 Estimator 实例参数
def test_validate_params_estimator():
    """Check that validate_params works with Estimator instances"""
    # 创建一个 _Estimator 实例，传入错误的参数，检查是否会抛出 InvalidParameterError 异常，并且异常消息匹配特定的模式
    est = _Estimator("wrong")
    with pytest.raises(
        InvalidParameterError, match="The 'a' parameter of _Estimator must be"
    ):
        est.fit()


# 定义一个测试函数，验证在初始化时 deprecated 参数必须是选项集的子集
def test_stroptions_deprecated_subset():
    """Check that the deprecated parameter must be a subset of options."""
    # 使用 pytest.raises 检查是否会抛出 ValueError 异常，并且异常消息匹配特定的模式
    with pytest.raises(ValueError, match="deprecated options must be a subset"):
        StrOptions({"a", "b", "c"}, deprecated={"a", "d"})


# 定义一个测试函数，验证内部约束不会在错误消息中暴露出来
def test_hidden_constraint():
    """Check that internal constraints are not exposed in the error message."""

    # 使用 validate_params 装饰器，验证参数 param 可以是列表或字典类型
    @validate_params(
        {"param": [Hidden(list), dict]}, prefer_skip_nested_validation=True
    )
    def f(param):
        pass

    # 调用函数 f，传递有效参数字典和列表，确保不会抛出异常
    f({"a": 1, "b": 2, "c": 3})
    f([1, 2, 3])

    # 使用 pytest.raises 检查是否会抛出 InvalidParameterError 异常，并且异常消息匹配特定的模式
    with pytest.raises(
        InvalidParameterError, match="The 'param' parameter"
    ) as exc_info:
        f(param="bad")

    # 检查错误消息确保不包含 list 选项
    err_msg = str(exc_info.value)
    assert "an instance of 'dict'" in err_msg
    assert "an instance of 'list'" not in err_msg


# 定义一个测试函数，验证 StrOptions 约束可以有两个，其中一个是隐藏的
def test_hidden_stroptions():
    """Check that we can have 2 StrOptions constraints, one being hidden."""

    # 使用 validate_params 装饰器，验证参数 param 可以是 "auto" 或者 "warn"，"warn" 选项是隐藏的
    @validate_params(
        {"param": [StrOptions({"auto"}), Hidden(StrOptions({"warn"}))]},
        prefer_skip_nested_validation=True,
    )
    def f(param):
        pass

    # 调用函数 f，传递有效参数 "auto" 和 "warn"，确保不会抛出异常
    f("auto")
    f("warn")

    # 使用 pytest.raises 检查是否会抛出 InvalidParameterError 异常，并且异常消息匹配特定的模式
    with pytest.raises(
        InvalidParameterError, match="The 'param' parameter"
    ) as exc_info:
        f(param="bad")

    # 检查错误消息确保不包含 "warn" 选项
    err_msg = str(exc_info.value)
    assert "auto" in err_msg
    assert "warn" not in err_msg


# 定义一个测试函数，验证 validate_params 装饰器能够正确设置参数约束属性
def test_validate_params_set_param_constraints_attribute():
    """Check that the validate_params decorator properly sets the parameter constraints
    as attribute of the decorated function/method.
    """
    # 使用 assert 检查 _func 和 _Class()._method 是否有 _skl_parameter_constraints 属性
    assert hasattr(_func, "_skl_parameter_constraints")
    assert hasattr(_Class()._method, "_skl_parameter_constraints")


# 定义一个测试函数，验证在接受布尔值参数时，当传入整数会抛出废弃消息但仍然通过验证
def test_boolean_constraint_deprecated_int():
    """Check that validate_params raise a deprecation message but still passes
    validation when using an int for a parameter accepting a boolean.
    """
    # 定义装饰器函数，用于验证参数，参数结构为{"param": ["boolean"]}，允许跳过嵌套验证
    @validate_params({"param": ["boolean"]}, prefer_skip_nested_validation=True)
    # 定义函数f，接收一个布尔类型的参数param
    def f(param):
        # 函数体暂时为空，因为pass表示什么也不做
    
    # 调用函数f，传入True作为参数
    f(True)
    # 调用函数f，传入np.bool_(False)作为参数，np.bool_是numpy库中的布尔类型
    f(np.bool_(False))
def test_no_validation():
    """Check that validation can be skipped for a parameter."""

    @validate_params(
        {"param1": [int, None], "param2": "no_validation"},
        prefer_skip_nested_validation=True,
    )
    def f(param1=None, param2=None):
        pass

    # param1 is validated
    with pytest.raises(InvalidParameterError, match="The 'param1' parameter"):
        f(param1="wrong")

    # param2 is not validated: any type is valid.
    class SomeType:
        pass

    # Calling f with SomeType class as param2 argument
    f(param2=SomeType)

    # Calling f with an instance of SomeType class as param2 argument
    f(param2=SomeType())


def test_pandas_na_constraint_with_pd_na():
    """Add a specific test for checking support for `pandas.NA`."""
    pd = pytest.importorskip("pandas")

    # Creating a _PandasNAConstraint object
    na_constraint = _PandasNAConstraint()

    # Asserting that pandas.NA satisfies the constraint
    assert na_constraint.is_satisfied_by(pd.NA)

    # Asserting that numpy array does not satisfy the constraint
    assert not na_constraint.is_satisfied_by(np.array([1, 2, 3]))


def test_iterable_not_string():
    """Check that a string does not satisfy the _IterableNotString constraint."""
    constraint = _IterablesNotString()

    # Asserting that list satisfies the constraint
    assert constraint.is_satisfied_by([1, 2, 3])

    # Asserting that range object satisfies the constraint
    assert constraint.is_satisfied_by(range(10))

    # Asserting that string does not satisfy the constraint
    assert not constraint.is_satisfied_by("some string")


def test_cv_objects():
    """Check that the _CVObjects constraint accepts all current ways
    to pass cv objects."""
    constraint = _CVObjects()

    # Asserting different objects satisfy the constraint
    assert constraint.is_satisfied_by(5)
    assert constraint.is_satisfied_by(LeaveOneOut())
    assert constraint.is_satisfied_by([([1, 2], [3, 4]), ([3, 4], [1, 2])])
    
    # Asserting None does not satisfy the constraint
    assert constraint.is_satisfied_by(None) 

    # Asserting that a string does not satisfy the constraint
    assert not constraint.is_satisfied_by("not a CV object")


def test_third_party_estimator():
    """Check that the validation from a scikit-learn estimator inherited by a third
    party estimator does not impose a match between the dict of constraints and the
    parameters of the estimator.
    """

    class ThirdPartyEstimator(_Estimator):
        def __init__(self, b):
            self.b = b
            super().__init__(a=0)

        def fit(self, X=None, y=None):
            super().fit(X, y)

    # Instantiating ThirdPartyEstimator with 'b=0' and calling fit method
    # No error is raised despite 'b' not being in the constraints dict and 'a' not being a parameter of the estimator.
    ThirdPartyEstimator(b=0).fit()


def test_interval_real_not_int():
    """Check for the type RealNotInt in the Interval constraint."""
    constraint = Interval(RealNotInt, 0, 1, closed="both")

    # Asserting that float satisfies the RealNotInt constraint
    assert constraint.is_satisfied_by(1.0)

    # Asserting that integer does not satisfy the RealNotInt constraint
    assert not constraint.is_satisfied_by(1)


def test_real_not_int():
    """Check for the RealNotInt type."""
    
    # Asserting various types against RealNotInt
    assert isinstance(1.0, RealNotInt)
    assert not isinstance(1, RealNotInt)
    assert isinstance(np.float64(1), RealNotInt)
    assert not isinstance(np.int64(1), RealNotInt)


def test_skip_param_validation():
    """Check that param validation can be skipped using config_context."""

    @validate_params({"a": [int]}, prefer_skip_nested_validation=True)
    def f(a):
        pass

    # Asserting that InvalidParameterError is raised with message about 'a' parameter
    with pytest.raises(InvalidParameterError, match="The 'a' parameter"):
        f(a="1")
    # 在不引发异常的情况下，使用特定的配置上下文，跳过参数验证
    with config_context(skip_parameter_validation=True):
        # 调用函数 f，并传入参数 'a'='1'
        f(a="1")
# 使用 pytest.mark.parametrize 装饰器为 test_skip_nested_validation 函数参数 prefer_skip_nested_validation 添加参数化测试
@pytest.mark.parametrize("prefer_skip_nested_validation", [True, False])
def test_skip_nested_validation(prefer_skip_nested_validation):
    """Check that nested validation can be skipped."""

    # 定义一个带参数验证的装饰器，参数为字典 {"a": [int]}，并设置 prefer_skip_nested_validation 为 True 或 False
    @validate_params({"a": [int]}, prefer_skip_nested_validation=True)
    def f(a):
        pass

    # 定义一个带参数验证的装饰器，参数为字典 {"b": [int]}，并设置 prefer_skip_nested_validation 为传入的参数 prefer_skip_nested_validation
    @validate_params(
        {"b": [int]},
        prefer_skip_nested_validation=prefer_skip_nested_validation,
    )
    def g(b):
        # 调用 f 函数并传入一个错误的参数类型
        return f(a="invalid_param_value")

    # 验证 g 函数的参数验证不会被跳过
    with pytest.raises(InvalidParameterError, match="The 'b' parameter"):
        g(b="invalid_param_value")

    # 如果 prefer_skip_nested_validation 为 True，则 g 函数的参数验证会被跳过
    if prefer_skip_nested_validation:
        g(b=1)  # does not raise because inner f is not validated
    else:
        # 否则验证 g 函数的参数验证不会被跳过
        with pytest.raises(InvalidParameterError, match="The 'a' parameter"):
            g(b=1)


# 使用 pytest.mark.parametrize 装饰器为 test_skip_nested_validation_and_config_context 函数参数 skip_parameter_validation, prefer_skip_nested_validation, expected_skipped 添加参数化测试
@pytest.mark.parametrize(
    "skip_parameter_validation, prefer_skip_nested_validation, expected_skipped",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_skip_nested_validation_and_config_context(
    skip_parameter_validation, prefer_skip_nested_validation, expected_skipped
):
    """Check interaction between global skip and local skip."""

    # 定义一个带参数验证的装饰器，参数为字典 {"a": [int]}，并设置 prefer_skip_nested_validation 为传入的参数 prefer_skip_nested_validation
    @validate_params(
        {"a": [int]}, prefer_skip_nested_validation=prefer_skip_nested_validation
    )
    def g(a):
        return get_config()["skip_parameter_validation"]

    # 在配置上下文中设置 skip_parameter_validation 为传入的参数 skip_parameter_validation
    with config_context(skip_parameter_validation=skip_parameter_validation):
        actual_skipped = g(1)

    # 断言实际跳过的结果与预期结果相同
    assert actual_skipped == expected_skipped
```