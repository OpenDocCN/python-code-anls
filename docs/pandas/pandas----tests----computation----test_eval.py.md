# `D:\src\scipysrc\pandas\pandas\tests\computation\test_eval.py`

```
# 从未来版本导入注解功能
from __future__ import annotations

# 导入 functools 模块中的 reduce 函数
from functools import reduce
# 导入 itertools 模块中的 product 函数
from itertools import product
# 导入 operator 模块
import operator

# 导入 NumPy 库，并使用 np 别名
import numpy as np
# 导入 pytest 库
import pytest

# 从 pandas.compat 模块导入 PY312
from pandas.compat import PY312
# 从 pandas.errors 模块导入多个异常类
from pandas.errors import (
    NumExprClobberingError,
    PerformanceWarning,
    UndefinedVariableError,
)
# 导入 pandas.util._test_decorators 模块，并使用 td 别名
import pandas.util._test_decorators as td

# 从 pandas.core.dtypes.common 模块导入多个函数
from pandas.core.dtypes.common import (
    is_bool,
    is_float,
    is_list_like,
    is_scalar,
)

# 导入 pandas 库，并使用 pd 别名
import pandas as pd
# 从 pandas 库中导入多个对象和函数
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
    period_range,
    timedelta_range,
)
# 导入 pandas._testing 模块，并使用 tm 别名
import pandas._testing as tm
# 从 pandas.core.computation 模块导入 expr 和 pytables
from pandas.core.computation import (
    expr,
    pytables,
)
# 从 pandas.core.computation.engines 模块导入 ENGINES
from pandas.core.computation.engines import ENGINES
# 从 pandas.core.computation.expr 模块导入多个类
from pandas.core.computation.expr import (
    BaseExprVisitor,
    PandasExprVisitor,
    PythonExprVisitor,
)
# 从 pandas.core.computation.expressions 模块导入 NUMEXPR_INSTALLED 和 USE_NUMEXPR
from pandas.core.computation.expressions import (
    NUMEXPR_INSTALLED,
    USE_NUMEXPR,
)
# 从 pandas.core.computation.ops 模块导入多个变量和函数
from pandas.core.computation.ops import (
    ARITH_OPS_SYMS,
    _binary_math_ops,
    _binary_ops_dict,
    _unary_math_ops,
)
# 从 pandas.core.computation.scope 模块导入 DEFAULT_GLOBALS
from pandas.core.computation.scope import DEFAULT_GLOBALS

# 定义 pytest 的 fixture，参数为 engine
@pytest.fixture(
    params=(
        pytest.param(
            engine,
            marks=[
                # 如果引擎为 "numexpr" 并且 USE_NUMEXPR 为 False，则跳过测试
                pytest.mark.skipif(
                    engine == "numexpr" and not USE_NUMEXPR,
                    reason=f"numexpr enabled->{USE_NUMEXPR}, "
                    f"installed->{NUMEXPR_INSTALLED}",
                ),
                # 使用 td.skip_if_no("numexpr") 标记
                td.skip_if_no("numexpr"),
            ],
        )
        for engine in ENGINES
    )
)
def engine(request):
    return request.param

# 定义 pytest 的 fixture，参数为 parser
@pytest.fixture(params=expr.PARSERS)
def parser(request):
    return request.param

# 定义函数 _eval_single_bin，接受 lhs、cmp1、rhs 和 engine 参数
def _eval_single_bin(lhs, cmp1, rhs, engine):
    # 获取 _binary_ops_dict 中的操作函数 c
    c = _binary_ops_dict[cmp1]
    # 如果 ENGINES[engine].has_neg_frac 为真
    if ENGINES[engine].has_neg_frac:
        try:
            # 使用 c 计算 lhs 和 rhs
            return c(lhs, rhs)
        except ValueError as e:
            # 捕获 ValueError 异常
            if str(e).startswith(
                "negative number cannot be raised to a fractional power"
            ):
                return np.nan  # 返回 NaN
            raise  # 抛出原始异常
    return c(lhs, rhs)  # 返回 c(lhs, rhs)

# TODO: using range(5) here is a kludge
# 定义 pytest 的 fixture，使用 range(5) 作为参数列表
@pytest.fixture(
    params=list(range(5)),
    ids=["DataFrame", "Series", "SeriesNaN", "DataFrameNaN", "float"],
)
def lhs(request):
    # 生成一个具有随机标准正态分布数据的 DataFrame nan_df1
    nan_df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    nan_df1[nan_df1 > 0.5] = np.nan  # 将大于 0.5 的值设为 NaN

    # 定义 opts 列表，包含不同类型的对象
    opts = (
        DataFrame(np.random.default_rng(2).standard_normal((10, 5))),  # DataFrame
        Series(np.random.default_rng(2).standard_normal(5)),  # Series
        Series([1, 2, np.nan, np.nan, 5]),  # 包含 NaN 的 Series
        nan_df1,  # 包含 NaN 的 DataFrame
        np.random.default_rng(2).standard_normal(),  # 随机标准正态分布值
    )
    return opts[request.param]  # 返回根据 param 选择的对象

rhs = lhs  # 将 rhs 设为 lhs 的值
midhs = lhs  # 将 midhs 设为 lhs 的值

# 定义 pytest 的 fixture，名称为 idx_func_dict，无参数
@pytest.fixture
def idx_func_dict():
    # 返回一个字典，包含不同类型索引的生成器函数
    return {
        # 返回一个整数索引的生成器函数，范围从0到n-1
        "i": lambda n: Index(np.arange(n), dtype=np.int64),
        # 返回一个浮点数索引的生成器函数，范围从0到n-1
        "f": lambda n: Index(np.arange(n), dtype=np.float64),
        # 返回一个字符串索引的生成器函数，格式为"索引_对应的ASCII字符"
        "s": lambda n: Index([f"{i}_{chr(i)}" for i in range(97, 97 + n)]),
        # 返回一个日期索引的生成器函数，从"2020-01-01"开始，生成n个日期
        "dt": lambda n: date_range("2020-01-01", periods=n),
        # 返回一个时间间隔索引的生成器函数，从"1 day"开始，生成n个时间间隔
        "td": lambda n: timedelta_range("1 day", periods=n),
        # 返回一个周期索引的生成器函数，从"2020-01-01"开始，频率为天，生成n个周期
        "p": lambda n: period_range("2020-01-01", periods=n, freq="D"),
    }
    # 定义一个测试类 TestEval，用于测试复杂的比较运算
    class TestEval:
        # 使用 pytest 的参数化装饰器，对测试方法 test_complex_cmp_ops 参数化
        @pytest.mark.parametrize(
            "cmp1",
            ["!=", "==", "<=", ">=", "<", ">"],  # 参数化比较运算符
            ids=["ne", "eq", "le", "ge", "lt", "gt"],  # 指定参数化标识
        )
        @pytest.mark.parametrize("cmp2", [">", "<"], ids=["gt", "lt"])  # 参数化比较运算符
        @pytest.mark.parametrize("binop", expr.BOOL_OPS_SYMS)  # 使用表达式模块中的布尔运算符参数化
        # 定义测试复杂比较运算的方法，接受参数 cmp1, cmp2, binop, lhs, rhs, engine, parser
        def test_complex_cmp_ops(self, cmp1, cmp2, binop, lhs, rhs, engine, parser):
            # 如果使用的解析器是 "python"，并且 binop 是 "and" 或 "or"
            if parser == "python" and binop in ["and", "or"]:
                msg = "'BoolOp' nodes are not implemented"
                ex = f"(lhs {cmp1} rhs) {binop} (lhs {cmp2} rhs)"
                # 断言抛出 NotImplementedError 异常，匹配给定的消息
                with pytest.raises(NotImplementedError, match=msg):
                    pd.eval(ex, engine=engine, parser=parser)
                return  # 结束测试方法

            # 分别用给定的比较运算符计算左右两侧表达式的值
            lhs_new = _eval_single_bin(lhs, cmp1, rhs, engine)
            rhs_new = _eval_single_bin(lhs, cmp2, rhs, engine)
            # 计算预期结果
            expected = _eval_single_bin(lhs_new, binop, rhs_new, engine)

            ex = f"(lhs {cmp1} rhs) {binop} (lhs {cmp2} rhs)"
            # 使用 pandas 的 eval 函数计算表达式 ex 的结果
            result = pd.eval(ex, engine=engine, parser=parser)
            # 断言计算结果与预期结果相等
            tm.assert_equal(result, expected)

        # 使用 pytest 的参数化装饰器，对测试方法 test_simple_cmp_ops 参数化
        @pytest.mark.parametrize("cmp_op", expr.CMP_OPS_SYMS)  # 使用表达式模块中的比较运算符参数化
        # 定义测试简单比较运算的方法，接受参数 cmp_op, lhs, rhs, engine, parser
        def test_simple_cmp_ops(self, cmp_op, lhs, rhs, engine, parser):
            # 将 lhs 和 rhs 转换为布尔值
            lhs = lhs < 0
            rhs = rhs < 0

            # 如果使用的解析器是 "python"，并且 cmp_op 是 "in" 或 "not in"
            if parser == "python" and cmp_op in ["in", "not in"]:
                msg = "'(In|NotIn)' nodes are not implemented"
                ex = f"lhs {cmp_op} rhs"
                # 断言抛出 NotImplementedError 异常，匹配给定的消息
                with pytest.raises(NotImplementedError, match=msg):
                    pd.eval(ex, engine=engine, parser=parser)
                return  # 结束测试方法

            ex = f"lhs {cmp_op} rhs"
            # 构造匹配的错误消息
            msg = "|".join(
                [
                    r"only list-like( or dict-like)? objects are allowed to be "
                    r"passed to (DataFrame\.)?isin\(\), you passed a "
                    r"(`|')bool(`|')",
                    "argument of type 'bool' is not iterable",
                ]
            )
            # 如果 cmp_op 是 "in" 或 "not in" 并且 rhs 不是类列表对象
            if cmp_op in ("in", "not in") and not is_list_like(rhs):
                # 断言抛出 TypeError 异常，匹配给定的错误消息
                with pytest.raises(TypeError, match=msg):
                    pd.eval(
                        ex,
                        engine=engine,
                        parser=parser,
                        local_dict={"lhs": lhs, "rhs": rhs},
                    )
            else:
                # 计算预期结果
                expected = _eval_single_bin(lhs, cmp_op, rhs, engine)
                # 使用 pandas 的 eval 函数计算表达式 ex 的结果
                result = pd.eval(ex, engine=engine, parser=parser)
                # 断言计算结果与预期结果相等
                tm.assert_equal(result, expected)

        @pytest.mark.parametrize("op", expr.CMP_OPS_SYMS)
    # 定义测试函数，用于测试复合反转操作
    def test_compound_invert_op(self, op, lhs, rhs, request, engine, parser):
        # 如果解析器为"python"且操作符为"in"或"not in"
        if parser == "python" and op in ["in", "not in"]:
            # 抛出未实现错误，指定相应消息
            msg = "'(In|NotIn)' nodes are not implemented"
            ex = f"~(lhs {op} rhs)"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex, engine=engine, parser=parser)
            return

        # 如果左操作数为浮点数且右操作数不是浮点数，且操作符为"in"或"not in"，
        # 引擎为"python"，解析器为"pandas"
        if (
            is_float(lhs)
            and not is_float(rhs)
            and op in ["in", "not in"]
            and engine == "python"
            and parser == "pandas"
        ):
            # 标记为预期失败，给出理由说明
            mark = pytest.mark.xfail(
                reason="Looks like expected is negative, unclear whether "
                "expected is incorrect or result is incorrect"
            )
            request.applymarker(mark)

        # 跳过检查的操作符列表
        skip_these = ["in", "not in"]
        ex = f"~(lhs {op} rhs)"

        # 构造匹配消息
        msg = "|".join(
            [
                r"only list-like( or dict-like)? objects are allowed to be "
                r"passed to (DataFrame\.)?isin\(\), you passed a "
                r"(`|')float(`|')",
                "argument of type 'float' is not iterable",
            ]
        )

        # 如果右操作数为标量且操作符在跳过列表中
        if is_scalar(rhs) and op in skip_these:
            # 使用 pytest 检查是否抛出类型错误，匹配相应消息
            with pytest.raises(TypeError, match=msg):
                pd.eval(
                    ex,
                    engine=engine,
                    parser=parser,
                    local_dict={"lhs": lhs, "rhs": rhs},
                )
        else:
            # 复合操作
            # 如果左操作数和右操作数都是标量
            if is_scalar(lhs) and is_scalar(rhs):
                lhs, rhs = (np.array([x]) for x in (lhs, rhs))
            # 计算单一二元操作的预期结果
            expected = _eval_single_bin(lhs, op, rhs, engine)
            # 如果预期结果是标量，将其取反
            if is_scalar(expected):
                expected = not expected
            else:
                expected = ~expected
            # 使用 pandas 执行表达式，并比较预期结果与实际结果的近似性
            result = pd.eval(ex, engine=engine, parser=parser)
            tm.assert_almost_equal(expected, result)

    # 使用参数化标记定义测试函数，用于测试链式比较操作
    @pytest.mark.parametrize("cmp1", ["<", ">"])
    @pytest.mark.parametrize("cmp2", ["<", ">"])
    def test_chained_cmp_op(self, cmp1, cmp2, lhs, midhs, rhs, engine, parser):
        # 将 midhs 赋值给 mid
        mid = midhs
        # 如果解析器为"python"
        if parser == "python":
            # 构造表达式 ex1，抛出未实现错误，指定相应消息
            ex1 = f"lhs {cmp1} mid {cmp2} rhs"
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex1, engine=engine, parser=parser)
            return

        # 计算左操作数经过单一二元操作的结果，使用给定的比较符 cmp1
        lhs_new = _eval_single_bin(lhs, cmp1, mid, engine)
        # 计算右操作数经过单一二元操作的结果，使用给定的比较符 cmp2
        rhs_new = _eval_single_bin(mid, cmp2, rhs, engine)

        # 如果左右操作数均非空
        if lhs_new is not None and rhs_new is not None:
            # 构造不同的表达式 ex1, ex2, ex3，并计算预期结果
            ex1 = f"lhs {cmp1} mid {cmp2} rhs"
            ex2 = f"lhs {cmp1} mid and mid {cmp2} rhs"
            ex3 = f"(lhs {cmp1} mid) & (mid {cmp2} rhs)"
            expected = _eval_single_bin(lhs_new, "&", rhs_new, engine)

            # 对每个表达式 ex 执行 pandas 计算，并比较结果与预期结果的近似性
            for ex in (ex1, ex2, ex3):
                result = pd.eval(ex, engine=engine, parser=parser)
                tm.assert_almost_equal(result, expected)
    @pytest.mark.parametrize(
        "arith1", sorted(set(ARITH_OPS_SYMS).difference({"**", "//", "%"}))
    )
    # 使用 pytest 的 parametrize 装饰器，对 test_binary_arith_ops 方法进行参数化测试，
    # 参数 arith1 从 ARITH_OPS_SYMS 中筛选出不包含 "**", "//", "%" 的运算符集合

    def test_binary_arith_ops(self, arith1, lhs, rhs, engine, parser):
        # 构建表达式 ex，如 "lhs + rhs"，执行 pd.eval 运算
        ex = f"lhs {arith1} rhs"
        result = pd.eval(ex, engine=engine, parser=parser)
        # 调用 _eval_single_bin 函数计算预期结果
        expected = _eval_single_bin(lhs, arith1, rhs, engine)

        # 使用 tm.assert_almost_equal 断言 result 与 expected 的近似相等性
        tm.assert_almost_equal(result, expected)

        # 构建带有两个相同运算符的表达式 ex，如 "lhs + rhs + rhs"
        ex = f"lhs {arith1} rhs {arith1} rhs"
        result = pd.eval(ex, engine=engine, parser=parser)
        # 计算新的 nlhs 值
        nlhs = _eval_single_bin(lhs, arith1, rhs, engine)
        
        try:
            # 尝试对 nlhs 和 rhs 进行对齐操作
            nlhs, ghs = nlhs.align(rhs)
        except (ValueError, TypeError, AttributeError):
            # 处理对齐操作可能出现的异常：ValueError, TypeError, AttributeError
            # 返回空，跳过后续处理
            return
        else:
            if engine == "numexpr":
                import numexpr as ne

                # 使用 numexpr 直接计算预期结果
                expected = ne.evaluate(f"nlhs {arith1} ghs")
                # 更新断言语句以应对数值精度问题
                # TODO: 更新测试代码，以便可以再次使用 assert_numpy_array_equal 语句替换 assert_almost_equal 语句
                tm.assert_almost_equal(result.values, expected)
            else:
                # 否则使用 eval 计算预期结果
                expected = eval(f"nlhs {arith1} ghs")
                tm.assert_almost_equal(result, expected)

    # modulus, pow, and floor division require special casing

    def test_modulus(self, lhs, rhs, engine, parser):
        # 创建取模运算表达式 ex
        ex = r"lhs % rhs"
        result = pd.eval(ex, engine=engine, parser=parser)
        # 计算预期的取模结果
        expected = lhs % rhs
        # 使用 tm.assert_almost_equal 断言 result 与 expected 的近似相等性
        tm.assert_almost_equal(result, expected)

        if engine == "numexpr":
            import numexpr as ne

            # 使用 numexpr 计算预期的取模结果
            expected = ne.evaluate(r"expected % rhs")
            # 如果 result 是 DataFrame 或 Series，则对比其值与预期值
            if isinstance(result, (DataFrame, Series)):
                tm.assert_almost_equal(result.values, expected)
            else:
                tm.assert_almost_equal(result, expected.item())
        else:
            # 否则使用 _eval_single_bin 计算预期结果
            expected = _eval_single_bin(expected, "%", rhs, engine)
            tm.assert_almost_equal(result, expected)

    def test_floor_division(self, lhs, rhs, engine, parser):
        # 创建整除运算表达式 ex
        ex = "lhs // rhs"

        if engine == "python":
            # 如果引擎是 Python，则执行 pd.eval 运算
            res = pd.eval(ex, engine=engine, parser=parser)
            # 计算预期的整除结果
            expected = lhs // rhs
            # 使用 tm.assert_equal 断言 res 与 expected 的相等性
            tm.assert_equal(res, expected)
        else:
            # 否则抛出特定错误信息
            msg = (
                r"unsupported operand type\(s\) for //: 'VariableNode' and "
                "'VariableNode'"
            )
            # 使用 pytest.raises 断言引发特定异常
            with pytest.raises(TypeError, match=msg):
                pd.eval(
                    ex,
                    local_dict={"lhs": lhs, "rhs": rhs},
                    engine=engine,
                    parser=parser,
                )

    @td.skip_if_windows
    # 使用 test_decorators 模块中的 skip_if_windows 装饰器标记跳过 Windows 系统的测试
    # 定义一个测试函数，用于测试幂运算
    def test_pow(self, lhs, rhs, engine, parser):
        # 在win32平台上有时会出现奇怪的失败，因此跳过测试
        ex = "lhs ** rhs"  # 构造幂运算的表达式字符串
        expected = _eval_single_bin(lhs, "**", rhs, engine)  # 计算预期结果
        result = pd.eval(ex, engine=engine, parser=parser)  # 使用pandas进行表达式求值

        # 如果lhs和rhs都是标量，并且预期结果是复数或复数浮点数，并且结果是NaN
        if (
            is_scalar(lhs)
            and is_scalar(rhs)
            and isinstance(expected, (complex, np.complexfloating))
            and np.isnan(result)
        ):
            msg = "(DataFrame.columns|numpy array) are different"  # 错误消息
            # 断言抛出AssertionError，并匹配特定的错误消息
            with pytest.raises(AssertionError, match=msg):
                tm.assert_numpy_array_equal(result, expected)
        else:
            # 对于一般情况，比较结果和预期结果的近似性
            tm.assert_almost_equal(result, expected)

            ex = "(lhs ** rhs) ** rhs"  # 嵌套幂运算的表达式字符串
            result = pd.eval(ex, engine=engine, parser=parser)  # 再次使用pandas进行表达式求值

            middle = _eval_single_bin(lhs, "**", rhs, engine)  # 计算中间结果
            expected = _eval_single_bin(middle, "**", rhs, engine)  # 计算预期结果
            tm.assert_almost_equal(result, expected)  # 断言结果和预期结果的近似性

    # 定义一个测试函数，用于检查单个反转操作
    def test_check_single_invert_op(self, lhs, engine, parser):
        # 简单处理，将lhs转换为布尔类型
        try:
            elb = lhs.astype(bool)
        except AttributeError:
            elb = np.array([bool(lhs)])
        expected = ~elb  # 计算预期结果，即取反操作
        result = pd.eval("~elb", engine=engine, parser=parser)  # 使用pandas进行表达式求值
        tm.assert_almost_equal(expected, result)  # 断言结果和预期结果的近似性
    def test_frame_invert(self, engine, parser):
        # 设置测试表达式
        expr = "~lhs"

        # 测试浮点数情况下的按位取反操作
        lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)))
        if engine == "numexpr":
            # 当引擎为 numexpr 时，预期抛出 NotImplementedError，并匹配特定错误消息
            msg = "couldn't find matching opcode for 'invert_dd'"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            # 其他引擎情况下，预期 TypeError，并进行结果比较
            msg = "ufunc 'invert' not supported for the input types"
            with pytest.raises(TypeError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)

        # 测试整数情况下的按位取反操作
        lhs = DataFrame(np.random.default_rng(2).integers(5, size=(5, 2)))
        if engine == "numexpr":
            # 当引擎为 numexpr 时，预期抛出 NotImplementedError，并匹配特定错误消息
            msg = "couldn't find matching opcode for 'invert'"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            # 其他引擎情况下，预期按位取反后的结果，并进行结果比较
            expect = ~lhs
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_frame_equal(expect, result)

        # 测试布尔类型情况下的按位取反操作
        lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)) > 0.5)
        expect = ~lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_frame_equal(expect, result)

        # 测试对象类型情况下的按位取反操作
        lhs = DataFrame(
            {"b": ["a", 1, 2.0], "c": np.random.default_rng(2).standard_normal(3) > 0.5}
        )
        if engine == "numexpr":
            # 当引擎为 numexpr 时，预期抛出 ValueError，并匹配特定错误消息
            with pytest.raises(ValueError, match="unknown type object"):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            # 其他引擎情况下，预期 TypeError，并匹配特定错误消息
            msg = "bad operand type for unary ~: 'str'"
            with pytest.raises(TypeError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
    ```python`
        def test_series_invert(self, engine, parser):
            # 定义表达式，使用位运算符“取反”
            expr = "~lhs"
    
            # 创建一个包含浮点数的 Series 对象，随机生成 5 个符合标准正态分布的数据
            lhs = Series(np.random.default_rng(2).standard_normal(5))
            # 如果引擎为 numexpr，预期在 eval 时抛出 NotImplementedError 异常
            if engine == "numexpr":
                msg = "couldn't find matching opcode for 'invert_dd'"
                with pytest.raises(NotImplementedError, match=msg):
                    result = pd.eval(expr, engine=engine, parser=parser)
            else:
                msg = "ufunc 'invert' not supported for the input types"
                with pytest.raises(TypeError, match=msg):
                    pd.eval(expr, engine=engine, parser=parser)
    
            # 创建一个包含整数的 Series 对象，随机生成 5 个在 0 到 5 之间的整数
            lhs = Series(np.random.default_rng(2).integers(5, size=5))
            # 如果引擎为 numexpr，预期在 eval 时抛出 NotImplementedError 异常
            if engine == "numexpr":
                msg = "couldn't find matching opcode for 'invert"
                with pytest.raises(NotImplementedError, match=msg):
                    pd.eval(expr, engine=engine, parser=parser)
            else:
                expect = ~lhs  # 计算取反的预期结果
                result = pd.eval(expr, engine=engine, parser=parser)
                tm.assert_series_equal(expect, result)  # 验证结果与预期相同
    
            # 创建一个包含布尔值的 Series 对象，随机生成 5 个符合正态分布的数据，并进行大于 0.5 的比较
            lhs = Series(np.random.default_rng(2).standard_normal(5) > 0.5)
            expect = ~lhs  # 计算取反的预期结果
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_series_equal(expect, result)  # 验证结果与预期相同
    
            # 创建一个包含混合数据类型的 Series 对象，包含字符串、整数和浮点数
            lhs = Series(["a", 1, 2.0])
            # 如果引擎为 numexpr，预期在 eval 时抛出 ValueError 异常，匹配错误信息 "unknown type object"
            if engine == "numexpr":
                with pytest.raises(ValueError, match="unknown type object"):
                    pd.eval(expr, engine=engine, parser=parser)
            else:
                msg = "bad operand type for unary ~: 'str'"
                # 如果引擎不是 numexpr，预期在 eval 时抛出 TypeError 异常，匹配错误信息 "bad operand type for unary ~: 'str'"
                with pytest.raises(TypeError, match=msg):
                    pd.eval(expr, engine=engine, parser=parser)
    
        def test_frame_negate(self, engine, parser):
            # 定义表达式，使用负号运算符进行取负
            expr = "-lhs"
    
            # 创建一个包含浮点数的 DataFrame 对象，随机生成 5 行 2 列的数据，符合标准正态分布
            lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)))
            expect = -lhs  # 计算取负的预期结果
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_frame_equal(expect, result)  # 验证结果与预期相同
    
            # 创建一个包含整数的 DataFrame 对象，随机生成 5 行 2 列的整数数据，在 0 到 5 之间
            lhs = DataFrame(np.random.default_rng(2).integers(5, size=(5, 2)))
            expect = -lhs  # 计算取负的预期结果
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_frame_equal(expect, result)  # 验证结果与预期相同
    
            # 创建一个包含布尔值的 DataFrame 对象，随机生成 5 行 2 列的数据，符合正态分布，并进行大于 0.5 的比较
            lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)) > 0.5)
            # 如果引擎为 numexpr，预期在 eval 时抛出 NotImplementedError 异常
            if engine == "numexpr":
                msg = "couldn't find matching opcode for 'neg_bb'"
                with pytest.raises(NotImplementedError, match=msg):
                    pd.eval(expr, engine=engine, parser=parser)
            else:
                expect = -lhs  # 计算取负的预期结果
                result = pd.eval(expr, engine=engine, parser=parser)
                tm.assert_frame_equal(expect, result)  # 验证结果与预期相同
    # 定义测试函数，用于测试 Series 对象的取反操作
    def test_series_negate(self, engine, parser):
        # 定义测试表达式
        expr = "-lhs"

        # 浮点数 Series
        lhs = Series(np.random.default_rng(2).standard_normal(5))
        # 期望结果是 Series 的取反
        expect = -lhs
        # 使用 pd.eval() 执行表达式求值
        result = pd.eval(expr, engine=engine, parser=parser)
        # 断言结果与期望一致
        tm.assert_series_equal(expect, result)

        # 整数 Series
        lhs = Series(np.random.default_rng(2).integers(5, size=5))
        # 期望结果是 Series 的取反
        expect = -lhs
        # 使用 pd.eval() 执行表达式求值
        result = pd.eval(expr, engine=engine, parser=parser)
        # 断言结果与期望一致
        tm.assert_series_equal(expect, result)

        # 布尔 Series 在 numexpr 引擎下不支持取反操作，但在其他引擎下可以工作
        lhs = Series(np.random.default_rng(2).standard_normal(5) > 0.5)
        if engine == "numexpr":
            # 如果是 numexpr 引擎，验证是否抛出预期的异常信息
            msg = "couldn't find matching opcode for 'neg_bb'"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            # 否则，期望结果是 Series 的取反
            expect = -lhs
            # 使用 pd.eval() 执行表达式求值
            result = pd.eval(expr, engine=engine, parser=parser)
            # 断言结果与期望一致
            tm.assert_series_equal(expect, result)

    # 使用参数化测试装饰器，测试 DataFrame 对象的正数加操作
    @pytest.mark.parametrize(
        "lhs",
        [
            # 浮点数数组
            np.random.default_rng(2).standard_normal((5, 2)),
            # 整数数组
            np.random.default_rng(2).integers(5, size=(5, 2)),
            # 布尔数组在 numexpr 引擎下不支持，但在其他引擎下可以工作
            np.array([True, False, True, False, True], dtype=np.bool_),
        ],
    )
    def test_frame_pos(self, lhs, engine, parser):
        # 将参数转换为 DataFrame
        lhs = DataFrame(lhs)
        # 定义测试表达式
        expr = "+lhs"
        # 期望结果与输入的 DataFrame 相同
        expect = lhs

        # 使用 pd.eval() 执行表达式求值
        result = pd.eval(expr, engine=engine, parser=parser)
        # 断言结果与期望一致
        tm.assert_frame_equal(expect, result)

    # 使用参数化测试装饰器，测试 Series 对象的正数加操作
    @pytest.mark.parametrize(
        "lhs",
        [
            # 浮点数 Series
            np.random.default_rng(2).standard_normal(5),
            # 整数 Series
            np.random.default_rng(2).integers(5, size=5),
            # 布尔 Series 在 numexpr 引擎下不支持，但在其他引擎下可以工作
            np.array([True, False, True, False, True], dtype=np.bool_),
        ],
    )
    def test_series_pos(self, lhs, engine, parser):
        # 将参数转换为 Series
        lhs = Series(lhs)
        # 定义测试表达式
        expr = "+lhs"
        # 期望结果与输入的 Series 相同
        expect = lhs

        # 使用 pd.eval() 执行表达式求值
        result = pd.eval(expr, engine=engine, parser=parser)
        # 断言结果与期望一致
        tm.assert_series_equal(expect, result)
    # 测试一元运算符在 pd.eval() 中的行为
    def test_scalar_unary(self, engine, parser):
        # 定义当操作数类型为浮点数时的异常消息
        msg = "bad operand type for unary ~: 'float'"
        warn = None
        # 如果是 Python 3.12 并且不是使用 engine="numexpr" 和 parser="pandas"，则设置 DeprecationWarning
        if PY312 and not (engine == "numexpr" and parser == "pandas"):
            warn = DeprecationWarning
        # 断言对 "~1.0" 表达式执行时会抛出 TypeError 异常，并且匹配预设的异常消息
        with pytest.raises(TypeError, match=msg):
            pd.eval("~1.0", engine=engine, parser=parser)

        # 断言对 "-1.0" 表达式执行的结果应为 -1.0
        assert pd.eval("-1.0", parser=parser, engine=engine) == -1.0
        # 断言对 "+1.0" 表达式执行的结果应为 +1.0
        assert pd.eval("+1.0", parser=parser, engine=engine) == +1.0
        # 断言对 "~1" 表达式执行的结果应为按位取反的结果
        assert pd.eval("~1", parser=parser, engine=engine) == ~1
        # 断言对 "-1" 表达式执行的结果应为 -1
        assert pd.eval("-1", parser=parser, engine=engine) == -1
        # 断言对 "+1" 表达式执行的结果应为 +1
        assert pd.eval("+1", parser=parser, engine=engine) == +1
        # 断言对 "~True" 表达式执行时会产生 DeprecationWarning 警告，并且结果应为按位取反的结果
        with tm.assert_produces_warning(
            warn, match="Bitwise inversion", check_stacklevel=False
        ):
            assert pd.eval("~True", parser=parser, engine=engine) == ~True
        # 断言对 "~False" 表达式执行时会产生 DeprecationWarning 警告，并且结果应为按位取反的结果
        with tm.assert_produces_warning(
            warn, match="Bitwise inversion", check_stacklevel=False
        ):
            assert pd.eval("~False", parser=parser, engine=engine) == ~False
        # 断言对 "-True" 表达式执行的结果应为 -True
        assert pd.eval("-True", parser=parser, engine=engine) == -True
        # 断言对 "-False" 表达式执行的结果应为 -False
        assert pd.eval("-False", parser=parser, engine=engine) == -False
        # 断言对 "+True" 表达式执行的结果应为 +True
        assert pd.eval("+True", parser=parser, engine=engine) == +True
        # 断言对 "+False" 表达式执行的结果应为 +False
        assert pd.eval("+False", parser=parser, engine=engine) == +False

    # 测试数组中的一元运算符行为
    def test_unary_in_array(self):
        # GH 11235
        # TODO: 2022-01-29: 在 CI 中使用 numexpr 2.7.3 可以返回列表，但本地无法重现
        # 使用 pd.eval() 计算表达式生成的数组，并设定数据类型为 np.object_
        result = np.array(
            pd.eval("[-True, True, +True, -False, False, +False, -37, 37, ~37, +37]"),
            dtype=np.object_,
        )
        # 预期结果的数组，数据类型也为 np.object_
        expected = np.array(
            [
                -True,
                True,
                +True,
                -False,
                False,
                +False,
                -37,
                37,
                ~37,
                +37,
            ],
            dtype=np.object_,
        )
        # 断言计算结果与预期结果的数组相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试浮点数比较的二元运算符行为
    @pytest.mark.parametrize("expr", ["x < -0.1", "-5 > x"])
    def test_float_comparison_bin_op(self, float_numpy_dtype, expr):
        # GH 16363
        # 创建包含浮点数的 DataFrame
        df = DataFrame({"x": np.array([0], dtype=float_numpy_dtype)})
        # 使用 df.eval() 计算表达式的结果
        res = df.eval(expr)
        # 断言结果数组中的值为预期的 False
        assert res.values == np.array([False])

    # 测试函数中的一元运算符行为
    def test_unary_in_function(self):
        # GH 46471
        # 创建包含整数和 NaN 值的 DataFrame
        df = DataFrame({"x": [0, 1, np.nan]})

        # 使用 df.eval() 计算表达式的结果
        result = df.eval("x.fillna(-1)")
        # 预期的填充 NaN 后的结果，如果引擎不是 numexpr，则检查列名
        expected = df.x.fillna(-1)
        tm.assert_series_equal(result, expected, check_names=not USE_NUMEXPR)

        # 使用 df.eval() 计算表达式的结果
        result = df.eval("x.shift(1, fill_value=-1)")
        # 预期的移位后的结果，如果引擎不是 numexpr，则检查列名
        expected = df.x.shift(1, fill_value=-1)
        tm.assert_series_equal(result, expected, check_names=not USE_NUMEXPR)
    @pytest.mark.parametrize(
        "ex",
        (
            "1 or 2",  # 第一个测试表达式：逻辑或运算
            "1 and 2",  # 第二个测试表达式：逻辑与运算
            "a and b",  # 第三个测试表达式：变量间的逻辑与运算
            "a or b",  # 第四个测试表达式：变量间的逻辑或运算
            "1 or 2 and (3 + 2) > 3",  # 第五个测试表达式：复合逻辑表达式
            "2 * x > 2 or 1 and 2",  # 第六个测试表达式：复合逻辑表达式
            "2 * df > 3 and 1 or a",  # 第七个测试表达式：复合逻辑表达式
        ),
    )
    def test_disallow_scalar_bool_ops(self, ex, engine, parser):
        x, a, b = np.random.default_rng(2).standard_normal(3), 1, 2  # 定义变量 x, a, b，初始化 x 为标准正态分布的随机数
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))  # 创建 DataFrame df，初始化为标准正态分布的随机数

        msg = "cannot evaluate scalar only bool ops|'BoolOp' nodes are not"
        # 使用 pytest 的断言检查 pd.eval 在给定引擎和解析器下是否抛出 NotImplementedError 异常，匹配特定消息
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex, engine=engine, parser=parser)

    def test_identical(self, engine, parser):
        # 查看问题编号 gh-10546
        x = 1
        # 使用 pd.eval 检查对于变量 x 的表达式求值结果是否为 1
        result = pd.eval("x", engine=engine, parser=parser)
        assert result == 1  # 断言结果为 1
        assert is_scalar(result)  # 断言结果为标量

        x = 1.5
        # 使用 pd.eval 检查对于变量 x 的表达式求值结果是否为 1.5
        result = pd.eval("x", engine=engine, parser=parser)
        assert result == 1.5  # 断言结果为 1.5
        assert is_scalar(result)  # 断言结果为标量

        x = False
        # 使用 pd.eval 检查对于变量 x 的表达式求值结果是否为 False
        result = pd.eval("x", engine=engine, parser=parser)
        assert not result  # 断言结果为 False
        assert is_bool(result)  # 断言结果为布尔值
        assert is_scalar(result)  # 断言结果为标量

        x = np.array([1])
        # 使用 pd.eval 检查对于变量 x 的表达式求值结果是否与数组 [1] 相等
        result = pd.eval("x", engine=engine, parser=parser)
        tm.assert_numpy_array_equal(result, np.array([1]))  # 使用测试框架的数组比较函数
        assert result.shape == (1,)  # 断言结果数组形状为 (1,)

        x = np.array([1.5])
        # 使用 pd.eval 检查对于变量 x 的表达式求值结果是否与数组 [1.5] 相等
        result = pd.eval("x", engine=engine, parser=parser)
        tm.assert_numpy_array_equal(result, np.array([1.5]))  # 使用测试框架的数组比较函数
        assert result.shape == (1,)  # 断言结果数组形状为 (1,)

        x = np.array([False])  # 定义 x 为包含 False 的数组
        # 使用 pd.eval 检查对于变量 x 的表达式求值结果是否与数组 [False] 相等
        result = pd.eval("x", engine=engine, parser=parser)
        tm.assert_numpy_array_equal(result, np.array([False]))  # 使用测试框架的数组比较函数
        assert result.shape == (1,)  # 断言结果数组形状为 (1,)

    def test_line_continuation(self, engine, parser):
        # GH 11149
        exp = """1 + 2 * \
        5 - 1 + 2 """
        # 使用 pd.eval 检查多行表达式 exp 的求值结果是否为 12
        result = pd.eval(exp, engine=engine, parser=parser)
        assert result == 12  # 断言结果为 12

    def test_float_truncation(self, engine, parser):
        # GH 14241
        exp = "1000000000.006"
        # 使用 pd.eval 检查对于表达式 exp 的求值结果是否与 np.float64(exp) 相等
        result = pd.eval(exp, engine=engine, parser=parser)
        expected = np.float64(exp)
        assert result == expected  # 断言结果与预期值相等

        df = DataFrame({"A": [1000000000.0009, 1000000000.0011, 1000000000.0015]})
        cutoff = 1000000000.0006
        # 使用 DataFrame 的 query 方法查询满足条件 A < 1000000000.0006 的结果，期望结果为空
        result = df.query(f"A < {cutoff:.4f}")
        assert result.empty  # 断言结果为空

        cutoff = 1000000000.0010
        # 使用 DataFrame 的 query 方法查询满足条件 A > 1000000000.0010 的结果，期望结果与预期相等
        result = df.query(f"A > {cutoff:.4f}")
        expected = df.loc[[1, 2], :]
        tm.assert_frame_equal(expected, result)  # 使用测试框架的数据框比较函数

        exact = 1000000000.0011
        # 使用 DataFrame 的 query 方法查询满足条件 A == 1000000000.0011 的结果，期望结果与预期相等
        result = df.query(f"A == {exact:.4f}")
        expected = df.loc[[1], :]
        tm.assert_frame_equal(expected, result)  # 使用测试框架的数据框比较函数
    # 测试禁止使用Python关键字作为标识符的行为
    def test_disallow_python_keywords(self):
        # 创建一个包含一行数据的DataFrame，列名分别为"foo", "bar", "class"
        df = DataFrame([[0, 0, 0]], columns=["foo", "bar", "class"])
        # 定义用于匹配的错误消息
        msg = "Python keyword not valid identifier in numexpr query"
        # 使用 pytest 检查是否会抛出 SyntaxError，并匹配预期的错误消息
        with pytest.raises(SyntaxError, match=msg):
            # 尝试在DataFrame上执行query操作，使用了Python关键字"class"作为标识符
            df.query("class == 0")

        # 创建一个空的DataFrame
        df = DataFrame()
        # 设置DataFrame的索引名称为"lambda"
        df.index.name = "lambda"
        # 再次使用 pytest 检查是否会抛出 SyntaxError，并匹配预期的错误消息
        with pytest.raises(SyntaxError, match=msg):
            # 尝试在DataFrame上执行query操作，使用了Python关键字"lambda"作为标识符
            df.query("lambda == 0")

    # 测试逻辑运算中True和False的行为
    def test_true_false_logic(self):
        # 根据 GitHub issue 25823 描述，这种行为在 Python 3.12 中已被弃用
        with tm.maybe_produces_warning(
            DeprecationWarning, PY312, check_stacklevel=False
        ):
            # 测试 pd.eval("not True") 的结果是否为 -2
            assert pd.eval("not True") == -2
            # 测试 pd.eval("not False") 的结果是否为 -1
            assert pd.eval("not False") == -1
            # 测试 pd.eval("True and not True") 的结果是否为 0
            assert pd.eval("True and not True") == 0

    # 测试字符串匹配逻辑运算中的and操作
    def test_and_logic_string_match(self):
        # 根据 GitHub issue 25823 描述
        event = Series({"a": "hello"})
        # 使用 pd.eval 进行字符串匹配，并断言结果为True
        assert pd.eval(f"{event.str.match('hello').a}")
        # 使用 pd.eval 进行两次字符串匹配的and操作，并断言结果为True
        assert pd.eval(f"{event.str.match('hello').a and event.str.match('hello').a}")

    # 测试在 eval 函数中保留列名
    def test_eval_keep_name(self, engine, parser):
        # 创建一个包含数据的Series，并转换为DataFrame
        df = Series([2, 15, 28], name="a").to_frame()
        # 使用 eval 函数计算"a + a"，并传入引擎和解析器参数
        res = df.eval("a + a", engine=engine, parser=parser)
        # 创建一个预期的Series对象
        expected = Series([4, 30, 56], name="a")
        # 断言 eval 函数的结果与预期的Series对象相等
        tm.assert_series_equal(expected, res)

    # 测试在 eval 函数中使用不匹配的列名
    def test_eval_unmatching_names(self, engine, parser):
        # 创建一个包含数据的Series，并指定列名为"series_name"
        variable_name = Series([42], name="series_name")
        # 使用 eval 函数计算"variable_name + 0"，并传入引擎和解析器参数
        res = pd.eval("variable_name + 0", engine=engine, parser=parser)
        # 断言 eval 函数的结果与原始Series对象相等
        tm.assert_series_equal(variable_name, res)
# -------------------------------------
# gh-12388: Typecasting rules consistency with python

# 定义一个测试类 TestTypeCasting，用于测试类型转换的一致性
class TestTypeCasting:
    
    # 参数化装饰器，测试以下操作符的类型转换规则："+", "-", "*", "**", "/"
    @pytest.mark.parametrize("op", ["+", "-", "*", "**", "/"])
    # 有一天也许... numexpr 现在有太多的类型提升规则了
    # 链式参数化，使用 np.core.sctypes 中的数据类型进行参数化
    @pytest.mark.parametrize("left_right", [("df", "3"), ("3", "df")])
    # 测试二元操作符的类型转换
    def test_binop_typecasting(self, engine, parser, op, float_numpy_dtype, left_right):
        # 创建一个 DataFrame，包含随机数值，默认数据类型为 float_numpy_dtype
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), dtype=float_numpy_dtype
        )
        left, right = left_right
        s = f"{left} {op} {right}"
        # 使用 pd.eval 执行字符串表达式 s，使用给定的引擎和解析器
        res = pd.eval(s, engine=engine, parser=parser)
        # 断言 DataFrame 的值的数据类型与 float_numpy_dtype 相同
        assert df.values.dtype == float_numpy_dtype
        # 断言结果 res 的值的数据类型与 float_numpy_dtype 相同
        assert res.values.dtype == float_numpy_dtype
        # 断言 DataFrame 和 eval(s) 的结果相等
        tm.assert_frame_equal(res, eval(s))


# -------------------------------------
# Basic and complex alignment

# 定义一个 should_warn 函数，用于检查是否应发出警告
def should_warn(*args):
    # 如果没有单调递增的属性
    not_mono = not any(map(operator.attrgetter("is_monotonic_increasing"), args))
    # 只有一个是 np.datetime64 类型的日期时间
    only_one_dt = reduce(
        operator.xor, (issubclass(x.dtype.type, np.datetime64) for x in args)
    )
    return not_mono and only_one_dt


# 定义一个测试类 TestAlignment，用于测试基本和复杂的对齐
class TestAlignment:
    index_types = ["i", "s", "dt"]
    lhs_index_types = index_types + ["s"]  # 'p'

    # 测试嵌套的一元操作符
    def test_align_nested_unary_op(self, engine, parser):
        s = "df * ~2"
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        res = pd.eval(s, engine=engine, parser=parser)
        tm.assert_frame_equal(res, df * ~2)

    # 参数化装饰器，测试基本框架对齐
    @pytest.mark.parametrize("lr_idx_type", lhs_index_types)
    @pytest.mark.parametrize("rr_idx_type", index_types)
    @pytest.mark.parametrize("c_idx_type", index_types)
    def test_basic_frame_alignment(
        self, engine, parser, lr_idx_type, rr_idx_type, c_idx_type, idx_func_dict
    ):
        # 创建两个 DataFrame，分别用不同的索引类型初始化
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            index=idx_func_dict[lr_idx_type](10            columns=idx_func_dict[c_idx_type](10        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((20, 10)),
            index=idx_func_dict[rr_idx_type](20            columns=idx_func_dict[c_idx_type](10        )
        # 如果应该发出警告，则使用 tm.assert_produces_warning 检查 RuntimeWarning
        if should_warn(df.index, df2.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval("df + df2", engine=engine, parser=parser)
        else:
            res = pd.eval("df + df2", engine=engine, parser=parser)
        # 断言 DataFrame 和 df + df2 的结果相等
        tm.assert_frame_equal(res, df + df2)

    # 参数化装饰器，测试框架比较
    @pytest.mark.parametrize("r_idx_type", lhs_index_types)
    @pytest.mark.parametrize("c_idx_type", lhs_index_types)
    def test_frame_comparison(
        self, engine, parser, r_idx_type, c_idx_type, idx_func_dict
    ):
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    # 标记：忽略特定类型的运行时警告
    @pytest.mark.parametrize("r1", lhs_index_types)
    # 使用参数化测试，对 r1 参数进行多次测试，使用的参数在 lhs_index_types 中定义
    @pytest.mark.parametrize("c1", index_types)
    # 使用参数化测试，对 c1 参数进行多次测试，使用的参数在 index_types 中定义
    @pytest.mark.parametrize("r2", index_types)
    # 使用参数化测试，对 r2 参数进行多次测试，使用的参数在 index_types 中定义
    @pytest.mark.parametrize("c2", index_types)
    # 使用参数化测试，对 c2 参数进行多次测试，使用的参数在 index_types 中定义
    def test_medium_complex_frame_alignment(
        self, engine, parser, r1, c1, r2, c2, idx_func_dict
    ):
        # 定义一个测试函数，测试中等复杂的 DataFrame 对齐操作
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 2)),
            # 使用随机数生成器创建一个 3x2 的标准正态分布的 DataFrame
            index=idx_func_dict[r1](3            # 使用 idx_func_dict 中 r1 对应的函数生成索引
            columns=idx_func_dict[c1](2            # 使用 idx_func_dict 中 c1 对应的函数生成列名
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)),
            # 使用随机数生成器创建一个 4x2 的标准正态分布的 DataFrame
            index=idx_func_dict[r2](4            # 使用 idx_func_dict 中 r2 对应的函数生成索引
            columns=idx_func_dict[c2](2            # 使用 idx_func_dict 中 c2 对应的函数生成列名
        )
        df3 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)),
            # 使用随机数生成器创建一个 5x2 的标准正态分布的 DataFrame
            index=idx_func_dict[r2](5            # 使用 idx_func_dict 中 r2 对应的函数生成索引
            columns=idx_func_dict[c2](2            # 使用 idx_func_dict 中 c2 对应的函数生成列名
        )
        if should_warn(df.index, df2.index, df3.index):
            # 如果应该发出警告（根据索引类型判断）
            with tm.assert_produces_warning(RuntimeWarning):
                # 使用 pytest 的 assert_produces_warning 上下文管理器检查是否产生 RuntimeWarning
                res = pd.eval("df + df2 + df3", engine=engine, parser=parser)
        else:
            # 如果不应该发出警告
            res = pd.eval("df + df2 + df3", engine=engine, parser=parser)
        tm.assert_frame_equal(res, df + df2 + df3)
        # 使用 pytest 的 assert_frame_equal 断言比较 res 和 df + df2 + df3 是否相等

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    # 标记：忽略特定类型的运行时警告
    @pytest.mark.parametrize("index_name", ["index", "columns"])
    # 使用参数化测试，对 index_name 参数进行多次测试，参数为 "index" 和 "columns"
    @pytest.mark.parametrize("c_idx_type", index_types)
    # 使用参数化测试，对 c_idx_type 参数进行多次测试，使用的参数在 index_types 中定义
    @pytest.mark.parametrize("r_idx_type", lhs_index_types)
    # 使用参数化测试，对 r_idx_type 参数进行多次测试，使用的参数在 lhs_index_types 中定义
    def test_basic_frame_series_alignment(
        self, engine, parser, index_name, r_idx_type, c_idx_type, idx_func_dict
    ):
        # 定义一个测试函数，测试基本的 DataFrame 和 Series 对齐操作
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            # 使用随机数生成器创建一个 10x10 的标准正态分布的 DataFrame
            index=idx_func_dict[r_idx_type](10            # 使用 idx_func_dict 中 r_idx_type 对应的函数生成索引
            columns=idx_func_dict[c_idx_type](10            # 使用 idx_func_dict 中 c_idx_type 对应的函数生成列名
        )
        index = getattr(df, index_name)
        # 获取 DataFrame 中的索引或列名，根据 index_name 参数决定
        s = Series(np.random.default_rng(2).standard_normal(5), index[:5])
        # 创建一个长度为 5 的 Series，使用随机数生成器生成标准正态分布的数据，索引为 df 的前 5 个

        if should_warn(df.index, s.index):
            # 如果应该发出警告（根据索引类型判断）
            with tm.assert_produces_warning(RuntimeWarning):
                # 使用 pytest 的 assert_produces_warning 上下文管理器检查是否产生 RuntimeWarning
                res = pd.eval("df + s", engine=engine, parser=parser)
        else:
            # 如果不应该发出警告
            res = pd.eval("df + s", engine=engine, parser=parser)

        if r_idx_type == "dt" or c_idx_type == "dt":
            # 如果 r_idx_type 或 c_idx_type 中包含 "dt"
            expected = df.add(s) if engine == "numexpr" else df + s
            # 如果引擎是 "numexpr"，使用 DataFrame 的 add 方法，否则直接相加
        else:
            expected = df + s
            # 直接将 Series 加到 DataFrame 上

        tm.assert_frame_equal(res, expected)
        # 使用 pytest 的 assert_frame_equal 断言比较 res 和 expected 是否相等
    @pytest.mark.parametrize("index_name", ["index", "columns"])
    # 使用pytest的parametrize装饰器，为index_name参数指定两个取值：'index'和'columns'

    @pytest.mark.parametrize(
        "r_idx_type, c_idx_type",
        list(product(["i", "s"], ["i", "s"])) + [("dt", "dt")],
    )
    # 使用pytest的parametrize装饰器，为r_idx_type和c_idx_type参数指定多组参数组合：
    # - r_idx_type取值为'i'或's'，c_idx_type同样取值为'i'或's'，以及一组特殊组合("dt", "dt")

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    # 使用pytest的filterwarnings装饰器，忽略RuntimeWarning类型的警告

    def test_basic_series_frame_alignment(
        self, request, engine, parser, index_name, r_idx_type, c_idx_type, idx_func_dict
    ):
        # 测试方法：test_basic_series_frame_alignment，参数包括self, request, engine, parser,
        # index_name, r_idx_type, c_idx_type, idx_func_dict

        if (
            engine == "numexpr"
            and parser in ("pandas", "python")
            and index_name == "index"
            and r_idx_type == "i"
            and c_idx_type == "s"
        ):
            # 如果engine为'numexpr'，且parser为'pandas'或'python'，并且index_name为'index'，
            # r_idx_type为'i'，c_idx_type为's'
            reason = (
                f"Flaky column ordering when engine={engine}, "
                f"parser={parser}, index_name={index_name}, "
                f"r_idx_type={r_idx_type}, c_idx_type={c_idx_type}"
            )
            # 生成reason字符串，描述测试失败的原因，包括engine, parser, index_name, r_idx_type, c_idx_type的值
            request.applymarker(pytest.mark.xfail(reason=reason, strict=False))
            # 应用pytest的xfail标记，标记该测试为预期失败，reason参数描述预期失败的原因

        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 7)),
            index=idx_func_dict[r_idx_type](10            columns=idx_func_dict[c_idx_type](7        )
        # 创建一个DataFrame对象df，包含随机生成的标准正态分布数据，行数为10，列数为7，
        # index和columns参数通过idx_func_dict[r_idx_type]函数生成

        index = getattr(df, index_name)
        # 获取df对象的属性index_name，赋值给index变量

        s = Series(np.random.default_rng(2).standard_normal(5), index[:5])
        # 创建一个Series对象s，包含随机生成的标准正态分布数据，长度为5，使用df的前5个索引作为index

        if should_warn(s.index, df.index):
            # 如果应当发出警告，根据s的索引和df的索引进行判断
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval("s + df", engine=engine, parser=parser)
                # 执行pd.eval("s + df")，使用给定的engine和parser求解表达式"s + df"，期望发出RuntimeWarning警告
        else:
            res = pd.eval("s + df", engine=engine, parser=parser)
            # 执行pd.eval("s + df")，使用给定的engine和parser求解表达式"s + df"

        if r_idx_type == "dt" or c_idx_type == "dt":
            expected = df.add(s) if engine == "numexpr" else s + df
            # 如果r_idx_type或c_idx_type为'dt'，则执行df.add(s)，否则执行s + df，根据engine选择求解方式
        else:
            expected = s + df
            # 否则，执行s + df

        tm.assert_frame_equal(res, expected)
        # 使用tm.assert_frame_equal比较res和expected，验证其是否相等

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    # 使用pytest的filterwarnings装饰器，忽略RuntimeWarning类型的警告

    @pytest.mark.parametrize("c_idx_type", index_types)
    # 使用pytest的parametrize装饰器，为c_idx_type参数指定多组参数组合，参数来源于index_types

    @pytest.mark.parametrize("r_idx_type", lhs_index_types)
    # 使用pytest的parametrize装饰器，为r_idx_type参数指定多组参数组合，参数来源于lhs_index_types

    @pytest.mark.parametrize("index_name", ["index", "columns"])
    # 使用pytest的parametrize装饰器，为index_name参数指定两个取值：'index'和'columns'

    @pytest.mark.parametrize("op", ["+", "*"])
    # 使用pytest的parametrize装饰器，为op参数指定两个取值：'+'和'*'

    def test_series_frame_commutativity(
        self, engine, parser, index_name, op, r_idx_type, c_idx_type, idx_func_dict
    ):
        # 测试方法：test_series_frame_commutativity，参数包括self, engine, parser, index_name,
        # op, r_idx_type, c_idx_type, idx_func_dict
    ):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            index=idx_func_dict[r_idx_type](10            columns=idx_func_dict[c_idx_type](10        )
        # 获取DataFrame的特定索引
        index = getattr(df, index_name)
        # 创建一个Series对象，包含部分DataFrame的索引和随机生成的数据
        s = Series(np.random.default_rng(2).standard_normal(5), index[:5])

        # 构建左操作数和右操作数的字符串表示
        lhs = f"s {op} df"
        rhs = f"df {op} s"
        
        # 如果应该发出警告，则使用with语句块执行eval，并检查是否产生RuntimeWarning
        if should_warn(df.index, s.index):
            with tm.assert_produces_warning(RuntimeWarning):
                a = pd.eval(lhs, engine=engine, parser=parser)
            with tm.assert_produces_warning(RuntimeWarning):
                b = pd.eval(rhs, engine=engine, parser=parser)
        else:
            # 否则直接执行eval
            a = pd.eval(lhs, engine=engine, parser=parser)
            b = pd.eval(rhs, engine=engine, parser=parser)

        # 如果r_idx_type和c_idx_type均不为"dt"，则进一步比较a和b
        if r_idx_type != "dt" and c_idx_type != "dt":
            if engine == "numexpr":
                # 使用测试工具函数检查a和b是否相等
                tm.assert_frame_equal(a, b)

    @pytest.mark.filterwarnings("always::RuntimeWarning")
    @pytest.mark.parametrize("r1", lhs_index_types)
    @pytest.mark.parametrize("c1", index_types)
    @pytest.mark.parametrize("r2", index_types)
    @pytest.mark.parametrize("c2", index_types)
    def test_complex_series_frame_alignment(
        self, engine, parser, r1, c1, r2, c2, idx_func_dict
    ):
        n = 3
        m1 = 5
        m2 = 2 * m1
        # 创建两个DataFrame对象，随机填充数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((m1, n)),
            index=idx_func_dict[r1](m1),
            columns=idx_func_dict[c1](n),
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((m2, n)),
            index=idx_func_dict[r2](m2),
            columns=idx_func_dict[c2](n),
        )
        # 获取df2的列索引
        index = df2.columns
        # 创建一个Series对象，包含随机生成的数据和df2的部分列索引
        ser = Series(np.random.default_rng(2).standard_normal(n), index[:n])

        # 根据r2和c2的值，计算expected2
        if r2 == "dt" or c2 == "dt":
            if engine == "numexpr":
                expected2 = df2.add(ser)
            else:
                expected2 = df2 + ser
        else:
            expected2 = df2 + ser

        # 根据r1和c1的值，计算expected
        if r1 == "dt" or c1 == "dt":
            if engine == "numexpr":
                expected = expected2.add(df)
            else:
                expected = expected2 + df
        else:
            expected = expected2 + df

        # 如果应该发出警告，则使用with语句块执行eval，并检查是否产生RuntimeWarning
        if should_warn(df2.index, ser.index, df.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval("df2 + ser + df", engine=engine, parser=parser)
        else:
            # 否则直接执行eval
            res = pd.eval("df2 + ser + df", engine=engine, parser=parser)
        
        # 断言结果的形状和期望的形状相等
        assert res.shape == expected.shape
        # 使用测试工具函数检查res和expected是否相等
        tm.assert_frame_equal(res, expected)

    def test_performance_warning_for_poor_alignment(
        self, performance_warning, engine, parser
    ):
        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 10)))
        # 创建一个包含随机数据的 DataFrame，形状为 (1000, 10)
        s = Series(np.random.default_rng(2).standard_normal(10000))
        # 创建一个包含随机数据的 Series，长度为 10000
        if engine == "numexpr" and performance_warning:
            seen = PerformanceWarning
        else:
            seen = False

        msg = "Alignment difference on axis 1 is larger than an order of magnitude"
        # 设置警告信息，用于后续警告匹配
        with tm.assert_produces_warning(seen, match=msg):
            # 使用 pd.eval() 执行表达式 "df + s"，选择引擎为 engine，解析器为 parser
            pd.eval("df + s", engine=engine, parser=parser)

        s = Series(np.random.default_rng(2).standard_normal(1000))
        # 创建一个新的 Series，长度为 1000
        with tm.assert_produces_warning(False):
            # 确保没有警告被触发
            pd.eval("df + s", engine=engine, parser=parser)

        df = DataFrame(np.random.default_rng(2).standard_normal((10, 10000)))
        # 创建一个包含随机数据的 DataFrame，形状为 (10, 10000)
        s = Series(np.random.default_rng(2).standard_normal(10000))
        # 创建一个包含随机数据的 Series，长度为 10000
        with tm.assert_produces_warning(False):
            # 确保没有警告被触发
            pd.eval("df + s", engine=engine, parser=parser)

        df = DataFrame(np.random.default_rng(2).standard_normal((10, 10)))
        # 创建一个包含随机数据的 DataFrame，形状为 (10, 10)
        s = Series(np.random.default_rng(2).standard_normal(10000))
        # 创建一个包含随机数据的 Series，长度为 10000

        is_python_engine = engine == "python"

        if not is_python_engine and performance_warning:
            wrn = PerformanceWarning
        else:
            wrn = False

        with tm.assert_produces_warning(wrn, match=msg) as w:
            # 使用 pd.eval() 执行表达式 "df + s"，选择引擎为 engine，解析器为 parser
            pd.eval("df + s", engine=engine, parser=parser)

            if not is_python_engine and performance_warning:
                # 如果使用非 Python 引擎且开启性能警告，则执行以下断言
                assert len(w) == 1
                # 确保只有一个警告被触发
                msg = str(w[0].message)
                # 获取警告信息的字符串表示
                logged = np.log10(s.size - df.shape[1])
                # 计算预期的警告信息
                expected = (
                    f"Alignment difference on axis 1 is larger "
                    f"than an order of magnitude on term 'df', "
                    f"by more than {logged:.4g}; performance may suffer."
                )
                # 断言实际警告信息与预期信息相同
                assert msg == expected
# ------------------------------------
# Slightly more complex ops

class TestOperations:
    def eval(self, *args, **kwargs):
        # 将关键字参数中的 "level" 放到 kwargs 中，并增加其值，如果没有指定，则默认为 0
        kwargs["level"] = kwargs.pop("level", 0) + 1
        # 调用 pandas 的 eval 函数，传入参数和更新后的 kwargs，并返回结果
        return pd.eval(*args, **kwargs)

    def test_simple_arith_ops(self, engine, parser):
        # 根据 parser 类型决定要排除的算术操作符列表
        exclude_arith = []
        if parser == "python":
            exclude_arith = ["in", "not in"]

        # 合并算术和比较操作符列表，并根据需要排除指定操作符
        arith_ops = [
            op
            for op in expr.ARITH_OPS_SYMS + expr.CMP_OPS_SYMS
            if op not in exclude_arith
        ]

        # 生成一个迭代器，过滤掉 "//" 操作符
        ops = (op for op in arith_ops if op != "//")

        # 遍历所有操作符
        for op in ops:
            # 构建简单的算术表达式
            ex = f"1 {op} 1"
            # 构建带变量的算术表达式
            ex2 = f"x {op} 1"
            # 构建包含表达式的算术表达式
            ex3 = f"1 {op} (x + 1)"

            if op in ("in", "not in"):
                # 如果操作符是 "in" 或 "not in"，验证是否会抛出 TypeError 异常
                msg = "argument of type 'int' is not iterable"
                with pytest.raises(TypeError, match=msg):
                    pd.eval(ex, engine=engine, parser=parser)
            else:
                # 否则，计算表达式的预期值并验证结果
                expec = _eval_single_bin(1, op, 1, engine)
                x = self.eval(ex, engine=engine, parser=parser)
                assert x == expec

                expec = _eval_single_bin(x, op, 1, engine)
                y = self.eval(ex2, local_dict={"x": x}, engine=engine, parser=parser)
                assert y == expec

                expec = _eval_single_bin(1, op, x + 1, engine)
                y = self.eval(ex3, local_dict={"x": x}, engine=engine, parser=parser)
                assert y == expec

    @pytest.mark.parametrize("rhs", [True, False])
    @pytest.mark.parametrize("lhs", [True, False])
    @pytest.mark.parametrize("op", expr.BOOL_OPS_SYMS)
    def test_simple_bool_ops(self, rhs, lhs, op):
        # 构建简单的布尔表达式
        ex = f"{lhs} {op} {rhs}"

        if parser == "python" and op in ["and", "or"]:
            # 如果 parser 是 "python" 并且操作符是 "and" 或 "or"，验证是否会抛出 NotImplementedError 异常
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                self.eval(ex)
            return

        # 否则，计算表达式的预期值并验证结果
        res = self.eval(ex)
        exp = eval(ex)
        assert res == exp

    @pytest.mark.parametrize("rhs", [True, False])
    @pytest.mark.parametrize("lhs", [True, False])
    @pytest.mark.parametrize("op", expr.BOOL_OPS_SYMS)
    def test_bool_ops_with_constants(self, rhs, lhs, op):
        # 构建布尔表达式
        ex = f"{lhs} {op} {rhs}"

        if parser == "python" and op in ["and", "or"]:
            # 如果 parser 是 "python" 并且操作符是 "and" 或 "or"，验证是否会抛出 NotImplementedError 异常
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                self.eval(ex)
            return

        # 否则，计算表达式的预期值并验证结果
        res = self.eval(ex)
        exp = eval(ex)
        assert res == exp

    def test_4d_ndarray_fails(self):
        # 创建一个随机生成的 4 维数组和一个 Series 对象
        x = np.random.default_rng(2).standard_normal((3, 4, 5, 6))
        y = Series(np.random.default_rng(2).standard_normal(10))
        # 验证对于 4 维数组执行 eval 操作会抛出 NotImplementedError 异常
        msg = "N-dimensional objects, where N > 2, are not supported with eval"
        with pytest.raises(NotImplementedError, match=msg):
            self.eval("x + y", local_dict={"x": x, "y": y})
    # 测试常量表达式的求值
    def test_constant(self):
        # 使用 eval 方法对字符串 "1" 进行求值，返回结果给 x
        x = self.eval("1")
        # 断言 x 的值等于 1
        assert x == 1

    # 测试单个变量的求值
    def test_single_variable(self):
        # 创建一个 DataFrame，包含 10 行 2 列的随机标准正态分布数据
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        # 使用 eval 方法对字符串 "df" 进行求值，使用本地字典传入 df 变量，返回结果给 df2
        df2 = self.eval("df", local_dict={"df": df})
        # 断言 df 和 df2 的内容相等
        tm.assert_frame_equal(df, df2)

    # 测试带有 NameError 异常的子脚本求值
    def test_failing_subscript_with_name_error(self):
        # 创建一个 DataFrame，包含 5 行 3 列的随机标准正态分布数据
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))  # noqa: F841
        # 使用 pytest 检测是否抛出 NameError 异常，异常信息应包含 "name 'x' is not defined"
        with pytest.raises(NameError, match="name 'x' is not defined"):
            # 使用 eval 方法对字符串 "df[x > 2] > 2" 进行求值
            self.eval("df[x > 2] > 2")

    # 测试左手边表达式子脚本求值
    def test_lhs_expression_subscript(self):
        # 创建一个 DataFrame，包含 5 行 3 列的随机标准正态分布数据
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        # 使用 eval 方法对字符串 "(df + 1)[df > 2]" 进行求值，使用本地字典传入 df 变量，返回结果给 result
        result = self.eval("(df + 1)[df > 2]", local_dict={"df": df})
        # 创建预期结果，应为 (df + 1)[df > 2]
        expected = (df + 1)[df > 2]
        # 断言 result 和 expected 的内容相等
        tm.assert_frame_equal(result, expected)

    # 测试属性表达式求值
    def test_attr_expression(self):
        # 创建一个 DataFrame，包含 5 行 3 列的随机标准正态分布数据，列标签为 'a', 'b', 'c'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=list("abc")
        )
        # 定义多个表达式和它们的预期结果
        expr1 = "df.a < df.b"
        expec1 = df.a < df.b
        expr2 = "df.a + df.b + df.c"
        expec2 = df.a + df.b + df.c
        expr3 = "df.a + df.b + df.c[df.b < 0]"
        expec3 = df.a + df.b + df.c[df.b < 0]
        exprs = expr1, expr2, expr3
        expecs = expec1, expec2, expec3
        # 遍历表达式和预期结果，使用 eval 方法对每个表达式求值，使用本地字典传入 df 变量，断言结果与预期一致
        for e, expec in zip(exprs, expecs):
            tm.assert_series_equal(expec, self.eval(e, local_dict={"df": df}))

    # 测试赋值操作失败的情况
    def test_assignment_fails(self):
        # 创建一个 DataFrame，包含 5 行 3 列的随机标准正态分布数据，列标签为 'a', 'b', 'c'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=list("abc")
        )
        # 创建另一个 DataFrame，包含 5 行 3 列的随机标准正态分布数据
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        # 定义表达式 "df = df2" 和预期的错误信息
        expr1 = "df = df2"
        msg = "cannot assign without a target object"
        # 使用 pytest 检测是否抛出 ValueError 异常，异常信息应包含上述错误信息
        with pytest.raises(ValueError, match=msg):
            # 使用 eval 方法对表达式进行求值，传入本地字典 df 和 df2
            self.eval(expr1, local_dict={"df": df, "df2": df2})

    # 测试赋值操作中出现多个赋值目标的情况
    def test_assignment_column_multiple_raise(self):
        # 创建一个 DataFrame，包含 5 行 2 列的随机标准正态分布数据，列标签为 'a', 'b'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # 使用 pytest 检测是否抛出 SyntaxError 异常，异常信息应包含 "invalid syntax"
        with pytest.raises(SyntaxError, match="invalid syntax"):
            # 在 DataFrame 上使用 eval 方法尝试执行 "d c = a + b" 操作
            df.eval("d c = a + b")

    # 测试赋值操作中出现无效赋值目标的情况
    def test_assignment_column_invalid_assign(self):
        # 创建一个 DataFrame，包含 5 行 2 列的随机标准正态分布数据，列标签为 'a', 'b'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # 定义错误信息
        msg = "left hand side of an assignment must be a single name"
        # 使用 pytest 检测是否抛出 SyntaxError 异常，异常信息应包含上述错误信息
        with pytest.raises(SyntaxError, match=msg):
            # 在 DataFrame 上使用 eval 方法尝试执行 "d,c = a + b" 操作
            df.eval("d,c = a + b")

    # 测试赋值操作中出现函数调用作为赋值目标的情况
    def test_assignment_column_invalid_assign_function_call(self):
        # 创建一个 DataFrame，包含 5 行 2 列的随机标准正态分布数据，列标签为 'a', 'b'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # 定义错误信息
        msg = "cannot assign to function call"
        # 使用 pytest 检测是否抛出 SyntaxError 异常，异常信息应包含上述错误信息
        with pytest.raises(SyntaxError, match=msg):
            # 在 DataFrame 上使用 eval 方法尝试执行 'Timestamp("20131001") = a + b' 操作
            df.eval('Timestamp("20131001") = a + b')
    def test_assignment_single_assign_existing(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # single assignment - existing variable
        expected = df.copy()  # 创建一个原始数据框的副本作为预期结果
        expected["a"] = expected["a"] + expected["b"]  # 在预期结果中计算列 'a' 的新值
        df.eval("a = a + b", inplace=True)  # 使用 eval 方法在原始数据框上直接计算列 'a' 的新值
        tm.assert_frame_equal(df, expected)  # 断言原始数据框和预期结果是否相等

    def test_assignment_single_assign_new(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # single assignment - new variable
        expected = df.copy()  # 创建一个原始数据框的副本作为预期结果
        expected["c"] = expected["a"] + expected["b"]  # 在预期结果中创建新的列 'c'
        df.eval("c = a + b", inplace=True)  # 使用 eval 方法在原始数据框上直接创建新的列 'c'
        tm.assert_frame_equal(df, expected)  # 断言原始数据框和预期结果是否相等

    def test_assignment_single_assign_local_overlap(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        df = df.copy()  # 创建原始数据框的副本
        a = 1  # noqa: F841
        df.eval("a = 1 + b", inplace=True)  # 使用 eval 方法在原始数据框上直接计算列 'a' 的新值

        expected = df.copy()  # 创建一个原始数据框的副本作为预期结果
        expected["a"] = 1 + expected["b"]  # 在预期结果中计算列 'a' 的新值
        tm.assert_frame_equal(df, expected)  # 断言原始数据框和预期结果是否相等

    def test_assignment_single_assign_name(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )

        a = 1  # noqa: F841
        old_a = df.a.copy()  # 创建列 'a' 的副本
        df.eval("a = a + b", inplace=True)  # 使用 eval 方法在原始数据框上直接计算列 'a' 的新值
        result = old_a + df.b  # 计算预期结果列 'a' 的值
        tm.assert_series_equal(result, df.a, check_names=False)  # 断言预期结果列和实际计算结果列是否相等，忽略列名检查
        assert result.name is None  # 断言结果列没有名称

    def test_assignment_multiple_raises(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # multiple assignment
        df.eval("c = a + b", inplace=True)  # 尝试在 eval 中进行多重赋值操作
        msg = "can only assign a single expression"
        with pytest.raises(SyntaxError, match=msg):  # 使用 pytest 断言捕获 SyntaxError 异常
            df.eval("c = a = b")  # 尝试在 eval 中进行多重赋值操作

    def test_assignment_explicit(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # explicit targets
        self.eval("c = df.a + df.b", local_dict={"df": df}, target=df, inplace=True)
        # 使用 self.eval 方法，显式计算目标 'c' 的值，并将结果存储在原始数据框中
        expected = df.copy()  # 创建一个原始数据框的副本作为预期结果
        expected["c"] = expected["a"] + expected["b"]  # 在预期结果中计算列 'c' 的新值
        tm.assert_frame_equal(df, expected)  # 断言原始数据框和预期结果是否相等

    def test_column_in(self, engine):
        # GH 11235
        df = DataFrame({"a": [11], "b": [-32]})
        result = df.eval("a in [11, -32]", engine=engine)  # 在数据框中使用 eval 方法进行条件判断
        expected = Series([True], name="a")  # 预期结果是一个布尔序列
        tm.assert_series_equal(result, expected)  # 断言实际结果和预期结果是否相等

    @pytest.mark.xfail(reason="Unknown: Omitted test_ in name prior.")
    def test_assignment_not_inplace(self):
        # 用例见 gh-9297
        # 创建一个具有随机标准正态分布数据的 DataFrame，包含 5 行 2 列，列名为 'a' 和 'b'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )

        # 使用 eval 方法计算新列 'c'，不改变原始 DataFrame，返回计算结果
        actual = df.eval("c = a + b", inplace=False)
        # 断言确保返回值不为 None
        assert actual is not None

        # 复制预期结果的 DataFrame
        expected = df.copy()
        # 计算预期结果中的列 'c'
        expected["c"] = expected["a"] + expected["b"]
        # 断言两个 DataFrame 相等
        tm.assert_frame_equal(df, expected)

    def test_multi_line_expression(self):
        # GH 11149
        # 创建一个包含两列 'a' 和 'b' 的 DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # 复制预期结果的 DataFrame
        expected = df.copy()

        # 计算预期结果中的列 'c' 和 'd'，并进行断言
        expected["c"] = expected["a"] + expected["b"]
        expected["d"] = expected["c"] + expected["b"]
        # 使用 eval 方法计算多行表达式，直接修改原始 DataFrame
        answer = df.eval(
            """
        c = a + b
        d = c + b""",
            inplace=True,
        )
        # 断言两个 DataFrame 相等
        tm.assert_frame_equal(expected, df)
        # 断言返回值为 None
        assert answer is None

        # 修改预期结果的 DataFrame 中的列 'a' 和 'e'
        expected["a"] = expected["a"] - 1
        expected["e"] = expected["a"] + 2
        # 使用 eval 方法计算多行表达式，直接修改原始 DataFrame
        answer = df.eval(
            """
        a = a - 1
        e = a + 2""",
            inplace=True,
        )
        # 断言两个 DataFrame 相等
        tm.assert_frame_equal(expected, df)
        # 断言返回值为 None
        assert answer is None

        # 如果不是所有表达式都包含赋值，则多行表达式无效
        msg = "Multi-line expressions are only valid if all expressions contain"
        # 使用 pytest 断言引发 ValueError，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            df.eval(
                """
            a = b + 2
            b - 2""",
                inplace=False,
            )

    def test_multi_line_expression_not_inplace(self):
        # GH 11149
        # 创建一个包含两列 'a' 和 'b' 的 DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # 复制预期结果的 DataFrame
        expected = df.copy()

        # 计算预期结果中的列 'c' 和 'd'，并进行断言
        expected["c"] = expected["a"] + expected["b"]
        expected["d"] = expected["c"] + expected["b"]
        # 使用 eval 方法计算多行表达式，返回新的 DataFrame，不改变原始 DataFrame
        df = df.eval(
            """
        c = a + b
        d = c + b""",
            inplace=False,
        )
        # 断言两个 DataFrame 相等
        tm.assert_frame_equal(expected, df)

        # 修改预期结果的 DataFrame 中的列 'a' 和 'e'
        expected["a"] = expected["a"] - 1
        expected["e"] = expected["a"] + 2
        # 使用 eval 方法计算多行表达式，返回新的 DataFrame，不改变原始 DataFrame
        df = df.eval(
            """
        a = a - 1
        e = a + 2""",
            inplace=False,
        )
        # 断言两个 DataFrame 相等
        tm.assert_frame_equal(expected, df)

    def test_multi_line_expression_local_variable(self):
        # GH 15342
        # 创建一个包含两列 'a' 和 'b' 的 DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # 复制预期结果的 DataFrame
        expected = df.copy()

        # 定义本地变量 local_var，并计算预期结果中的列 'c' 和 'd'，并进行断言
        local_var = 7
        expected["c"] = expected["a"] * local_var
        expected["d"] = expected["c"] + local_var
        # 使用 eval 方法计算多行表达式，直接修改原始 DataFrame
        answer = df.eval(
            """
        c = a * @local_var
        d = c + @local_var
        """,
            inplace=True,
        )
        # 断言两个 DataFrame 相等
        tm.assert_frame_equal(expected, df)
        # 断言返回值为 None
        assert answer is None
    def test_multi_line_expression_callable_local_variable(self):
        # 测试多行表达式中使用可调用的局部变量
        # 创建一个包含两列的数据框
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # 定义一个局部函数，返回第二个参数
        def local_func(a, b):
            return b

        # 复制数据框以备后用
        expected = df.copy()

        # 计算预期的"c"列，将"a"列的每个元素乘以局部函数返回值
        expected["c"] = expected["a"] * local_func(1, 7)

        # 计算预期的"d"列，将"c"列的每个元素加上局部函数返回值
        expected["d"] = expected["c"] + local_func(1, 7)

        # 在原数据框上使用eval方法进行计算，将结果存储在answer中
        answer = df.eval(
            """
        c = a * @local_func(1, 7)
        d = c + @local_func(1, 7)
        """,
            inplace=True,
        )

        # 使用测试工具检查预期结果和数据框是否相等
        tm.assert_frame_equal(expected, df)

        # 断言answer为None，表明eval方法成功执行且没有返回值
        assert answer is None

    def test_multi_line_expression_callable_local_variable_with_kwargs(self):
        # 测试多行表达式中使用带关键字参数的可调用局部变量
        # 创建一个包含两列的数据框
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # 定义一个局部函数，返回第二个参数
        def local_func(a, b):
            return b

        # 复制数据框以备后用
        expected = df.copy()

        # 计算预期的"c"列，将"a"列的每个元素乘以局部函数返回值（使用关键字参数）
        expected["c"] = expected["a"] * local_func(b=7, a=1)

        # 计算预期的"d"列，将"c"列的每个元素加上局部函数返回值（使用关键字参数）
        expected["d"] = expected["c"] + local_func(b=7, a=1)

        # 在原数据框上使用eval方法进行计算，将结果存储在answer中
        answer = df.eval(
            """
        c = a * @local_func(b=7, a=1)
        d = c + @local_func(b=7, a=1)
        """,
            inplace=True,
        )

        # 使用测试工具检查预期结果和数据框是否相等
        tm.assert_frame_equal(expected, df)

        # 断言answer为None，表明eval方法成功执行且没有返回值
        assert answer is None

    def test_assignment_in_query(self):
        # 测试query方法中的赋值操作
        # 创建一个包含两列的数据框
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        # 复制数据框以备后用
        df_orig = df.copy()

        # 准备一个错误消息用于异常断言
        msg = "cannot assign without a target object"

        # 使用pytest检查query方法中赋值操作是否会引发预期的异常
        with pytest.raises(ValueError, match=msg):
            df.query("a = 1")

        # 使用测试工具检查数据框是否未被修改
        tm.assert_frame_equal(df, df_orig)

    def test_query_inplace(self):
        # 测试query方法中的inplace参数
        # 创建一个包含两列的数据框
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # 复制数据框的子集作为预期结果
        expected = df.copy()
        expected = expected[expected["a"] == 2]

        # 使用query方法过滤数据框，同时修改原数据框
        df.query("a == 2", inplace=True)

        # 使用测试工具检查预期结果和数据框是否相等
        tm.assert_frame_equal(expected, df)

        # 重新赋值df，将其设为空字典
        df = {}

        # 准备一个预期的字典
        expected = {"a": 3}

        # 使用eval方法进行计算，将结果存储在df中
        self.eval("a = 1 + 2", target=df, inplace=True)

        # 使用测试工具检查预期结果和df是否相等
        tm.assert_dict_equal(df, expected)

    @pytest.mark.parametrize("invalid_target", [1, "cat", [1, 2], np.array([]), (1, 3)])
    def test_cannot_item_assign(self, invalid_target):
        # 测试不能对目标对象进行赋值的情况
        # 准备一个错误消息用于异常断言
        msg = "Cannot assign expression output to target"

        # 准备一个表达式用于测试
        expression = "a = 1 + 2"

        # 使用pytest检查对不合法目标对象执行eval方法时是否会引发预期的异常
        with pytest.raises(ValueError, match=msg):
            self.eval(expression, target=invalid_target, inplace=True)

        # 如果不合法目标对象具有copy方法
        if hasattr(invalid_target, "copy"):
            # 使用pytest检查对不合法目标对象执行eval方法时是否会引发预期的异常（inplace=False）
            with pytest.raises(ValueError, match=msg):
                self.eval(expression, target=invalid_target, inplace=False)

    @pytest.mark.parametrize("invalid_target", [1, "cat", (1, 3)])
    def test_cannot_copy_item(self, invalid_target):
        # 测试不能返回目标对象的副本的情况
        # 准备一个错误消息用于异常断言
        msg = "Cannot return a copy of the target"

        # 准备一个表达式用于测试
        expression = "a = 1 + 2"

        # 使用pytest检查对不合法目标对象执行eval方法时是否会引发预期的异常（inplace=False）
        with pytest.raises(ValueError, match=msg):
            self.eval(expression, target=invalid_target, inplace=False)

    @pytest.mark.parametrize("target", [1, "cat", [1, 2], np.array([]), (1, 3), {1: 2}])
    # 测试非原位操作且没有赋值的情况
    def test_inplace_no_assignment(self, target):
        # 定义一个表达式字符串
        expression = "1 + 2"

        # 断言使用 eval 方法计算表达式的结果，预期结果应为 3
        assert self.eval(expression, target=target, inplace=False) == 3

        # 在原位操作为 True 的情况下，使用 eval 方法计算表达式，预期引发 ValueError 异常，并匹配特定错误消息
        msg = "Cannot operate inplace if there is no assignment"
        with pytest.raises(ValueError, match=msg):
            self.eval(expression, target=target, inplace=True)

    # 测试基本的期间索引布尔表达式
    def test_basic_period_index_boolean_expression(self):
        # 创建一个包含随机数据的 DataFrame，列名为日期期间索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((2, 2)),
            columns=period_range("2020-01-01", freq="D", periods=2),
        )

        # 生成预期的布尔表达式结果
        e = df < 2

        # 使用 eval 方法计算表达式 "df < 2"，传入本地字典以替换 df，得到计算结果 r
        r = self.eval("df < 2", local_dict={"df": df})

        # 生成另一个预期的布尔表达式结果
        x = df < 2

        # 使用 assert_frame_equal 断言 r 与 e 相等
        tm.assert_frame_equal(r, e)
        # 使用 assert_frame_equal 断言 x 与 e 相等
        tm.assert_frame_equal(x, e)

    # 测试基本的期间索引下标表达式
    def test_basic_period_index_subscript_expression(self):
        # 创建一个包含随机数据的 DataFrame，列名为日期期间索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((2, 2)),
            columns=period_range("2020-01-01", freq="D", periods=2),
        )

        # 使用 eval 方法计算表达式 "df[df < 2 + 3]"，传入本地字典以替换 df，得到计算结果 r
        r = self.eval("df[df < 2 + 3]", local_dict={"df": df})

        # 生成预期的索引表达式结果
        e = df[df < 2 + 3]

        # 使用 assert_frame_equal 断言 r 与 e 相等
        tm.assert_frame_equal(r, e)

    # 测试嵌套的期间索引下标表达式
    def test_nested_period_index_subscript_expression(self):
        # 创建一个包含随机数据的 DataFrame，列名为日期期间索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((2, 2)),
            columns=period_range("2020-01-01", freq="D", periods=2),
        )

        # 使用 eval 方法计算表达式 "df[df[df < 2] < 2] + df * 2"，传入本地字典以替换 df，得到计算结果 r
        r = self.eval("df[df[df < 2] < 2] + df * 2", local_dict={"df": df})

        # 生成预期的嵌套索引表达式结果
        e = df[df[df < 2] < 2] + df * 2

        # 使用 assert_frame_equal 断言 r 与 e 相等
        tm.assert_frame_equal(r, e)

    # 测试日期布尔表达式
    def test_date_boolean(self, engine, parser):
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        
        # 添加日期列
        df["dates1"] = date_range("1/1/2012", periods=5)
        
        # 使用 eval 方法计算表达式 "df.dates1 < 20130101"，传入本地字典以替换 df，以及引擎和解析器参数，得到计算结果 res
        res = self.eval(
            "df.dates1 < 20130101",
            local_dict={"df": df},
            engine=engine,
            parser=parser,
        )

        # 生成预期的日期布尔表达式结果
        expec = df.dates1 < "20130101"

        # 使用 assert_series_equal 断言 res 与 expec 相等
        tm.assert_series_equal(res, expec)
    # 定义测试方法，测试简单的 'in' 操作
    def test_simple_in_ops(self, engine, parser):
        # 如果解析器不是 "python"，则执行以下测试
        if parser != "python":
            # 执行 pd.eval 函数，判断 1 是否在 [1, 2] 中，预期结果为 True
            res = pd.eval("1 in [1, 2]", engine=engine, parser=parser)
            assert res

            # 执行 pd.eval 函数，判断 2 是否在 (1, 2) 中，预期结果为 True
            res = pd.eval("2 in (1, 2)", engine=engine, parser=parser)
            assert res

            # 执行 pd.eval 函数，判断 3 是否在 (1, 2) 中，预期结果为 False
            res = pd.eval("3 in (1, 2)", engine=engine, parser=parser)
            assert not res

            # 执行 pd.eval 函数，判断 3 是否不在 (1, 2) 中，预期结果为 True
            res = pd.eval("3 not in (1, 2)", engine=engine, parser=parser)
            assert res

            # 执行 pd.eval 函数，判断 [3] 是否不在 (1, 2) 中，预期结果为 True
            res = pd.eval("[3] not in (1, 2)", engine=engine, parser=parser)
            assert res

            # 执行 pd.eval 函数，判断 [3] 是否在 ([3], 2) 中，预期结果为 True
            res = pd.eval("[3] in ([3], 2)", engine=engine, parser=parser)
            assert res

            # 执行 pd.eval 函数，判断 [[3]] 是否在 [[[3]], 2] 中，预期结果为 True
            res = pd.eval("[[3]] in [[[3]], 2]", engine=engine, parser=parser)
            assert res

            # 执行 pd.eval 函数，判断 (3,) 是否在 [(3,), 2] 中，预期结果为 True
            res = pd.eval("(3,) in [(3,), 2]", engine=engine, parser=parser)
            assert res

            # 执行 pd.eval 函数，判断 (3,) 是否不在 [(3,), 2] 中，预期结果为 False
            res = pd.eval("(3,) not in [(3,), 2]", engine=engine, parser=parser)
            assert not res

            # 执行 pd.eval 函数，判断 [(3,)] 是否在 [[(3,)], 2] 中，预期结果为 True
            res = pd.eval("[(3,)] in [[(3,)], 2]", engine=engine, parser=parser)
            assert res
        else:
            # 如果解析器是 "python"，则执行以下测试
            # 抛出 NotImplementedError 异常，因为 'In' 节点没有实现
            msg = "'In' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("1 in [1, 2]", engine=engine, parser=parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("2 in (1, 2)", engine=engine, parser=parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("3 in (1, 2)", engine=engine, parser=parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("[(3,)] in (1, 2, [(3,)])", engine=engine, parser=parser)
            # 抛出 NotImplementedError 异常，因为 'NotIn' 节点没有实现
            msg = "'NotIn' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("3 not in (1, 2)", engine=engine, parser=parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("[3] not in (1, 2, [[3]])", engine=engine, parser=parser)

    # 定义测试方法，检查多个表达式
    def test_check_many_exprs(self, engine, parser):
        # 定义变量 a 并赋值为 1
        a = 1  # noqa: F841
        # 构建表达式，重复字符串 "a" 33 次，以 * 连接
        expr = " * ".join("a" * 33)
        # 期望的结果为 1
        expected = 1
        # 执行 pd.eval 函数，计算表达式的值，使用给定的引擎和解析器
        res = pd.eval(expr, engine=engine, parser=parser)
        # 断言计算结果等于期望的结果
        assert res == expected

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试
    @pytest.mark.parametrize(
        "expr",
        [
            "df > 2 and df > 3",    # 表达式 1：df 大于 2 并且 df 大于 3
            "df > 2 or df > 3",     # 表达式 2：df 大于 2 或者 df 大于 3
            "not df > 2",           # 表达式 3：不是 df 大于 2
        ],
    )
    # 定义一个测试方法，用于测试表达式中的布尔运算符在特定解析器和引擎下的行为
    def test_fails_and_or_not(self, expr, engine, parser):
        # 创建一个包含随机数据的 DataFrame，用于测试
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        
        # 如果解析器是 "python"
        if parser == "python":
            # 设置默认错误消息
            msg = "'BoolOp' nodes are not implemented"
            
            # 如果表达式中包含 "not" 关键字，更新错误消息
            if "not" in expr:
                msg = "'Not' nodes are not implemented"

            # 使用 pytest 来确保 pd.eval 在特定条件下会引发 NotImplementedError，并匹配特定的错误消息
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(
                    expr,
                    local_dict={"df": df},
                    parser=parser,
                    engine=engine,
                )
        else:
            # 对于非 "python" 解析器，进行烟雾测试，预期不会引发异常
            pd.eval(
                expr,
                local_dict={"df": df},
                parser=parser,
                engine=engine,
            )

    # 使用 pytest 的参数化标记，定义一个测试方法，用于测试在不同解析器和引擎下的布尔运算符 "&" 和 "|" 的行为
    @pytest.mark.parametrize("char", ["|", "&"])
    def test_fails_ampersand_pipe(self, char, engine, parser):
        # 创建一个包含随机数据的 DataFrame，用于测试
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))  # noqa: F841
        
        # 构建测试表达式，包含特定的布尔运算符 "&" 或 "|"
        ex = f"(df + 2)[df > 1] > 0 {char} (df > 0)"
        
        # 如果解析器是 "python"
        if parser == "python":
            # 设置特定的错误消息，指示无法评估标量布尔运算
            msg = "cannot evaluate scalar only bool ops"
            
            # 使用 pytest 来确保 pd.eval 在特定条件下会引发 NotImplementedError，并匹配特定的错误消息
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex, parser=parser, engine=engine)
        else:
            # 对于非 "python" 解析器，进行烟雾测试，预期不会引发异常
            pd.eval(ex, parser=parser, engine=engine)
    # 定义一个名为 TestMath 的测试类，用于数学表达式的测试
    class TestMath:
        
        # 定义一个方法 eval，用于执行数学表达式的评估
        def eval(self, *args, **kwargs):
            # 将 kwargs 中的 "level" 键值对移除并添加到 kwargs 中，如果不存在则默认为 0
            kwargs["level"] = kwargs.pop("level", 0) + 1
            # 调用 pandas 的 eval 方法执行表达式求值
            return pd.eval(*args, **kwargs)

        # 使用 pytest 标记忽略 RuntimeWarning，对 _unary_math_ops 中的每个函数进行测试
        @pytest.mark.filterwarnings("ignore::RuntimeWarning")
        @pytest.mark.parametrize("fn", _unary_math_ops)
        # 定义测试一元数学函数的方法
        def test_unary_functions(self, fn, engine, parser):
            # 创建一个 DataFrame 包含随机数据列 "a"
            df = DataFrame({"a": np.random.default_rng(2).standard_normal(10)})
            a = df.a

            # 构建表达式，调用 eval 方法求值
            expr = f"{fn}(a)"
            got = self.eval(expr, engine=engine, parser=parser)
            # 使用 numpy 计算期望结果
            with np.errstate(all="ignore"):
                expect = getattr(np, fn)(a)
            # 断言得到的结果与期望结果相等
            tm.assert_series_equal(got, expect)

        # 对 _binary_math_ops 中的每个函数进行测试
        @pytest.mark.parametrize("fn", _binary_math_ops)
        # 定义测试二元数学函数的方法
        def test_binary_functions(self, fn, engine, parser):
            # 创建一个 DataFrame 包含随机数据列 "a" 和 "b"
            df = DataFrame(
                {
                    "a": np.random.default_rng(2).standard_normal(10),
                    "b": np.random.default_rng(2).standard_normal(10),
                }
            )
            a = df.a
            b = df.b

            # 构建表达式，调用 eval 方法求值
            expr = f"{fn}(a, b)"
            got = self.eval(expr, engine=engine, parser=parser)
            # 使用 numpy 计算期望结果
            with np.errstate(all="ignore"):
                expect = getattr(np, fn)(a, b)
            # 断言得到的结果与期望结果几乎相等
            tm.assert_almost_equal(got, expect)

        # 定义测试 DataFrame 用例的方法
        def test_df_use_case(self, engine, parser):
            # 创建一个 DataFrame 包含随机数据列 "a" 和 "b"
            df = DataFrame(
                {
                    "a": np.random.default_rng(2).standard_normal(10),
                    "b": np.random.default_rng(2).standard_normal(10),
                }
            )
            # 使用 eval 方法在 DataFrame 上执行表达式并修改原地
            df.eval(
                "e = arctan2(sin(a), b)",
                engine=engine,
                parser=parser,
                inplace=True,
            )
            got = df.e
            # 计算期望结果
            expect = np.arctan2(np.sin(df.a), df.b).rename("e")
            # 断言得到的结果与期望结果相等
            tm.assert_series_equal(got, expect)

        # 定义测试 DataFrame 算术子表达式的方法
        def test_df_arithmetic_subexpression(self, engine, parser):
            # 创建一个 DataFrame 包含随机数据列 "a" 和 "b"
            df = DataFrame(
                {
                    "a": np.random.default_rng(2).standard_normal(10),
                    "b": np.random.default_rng(2).standard_normal(10),
                }
            )
            # 使用 eval 方法在 DataFrame 上执行表达式并修改原地
            df.eval("e = sin(a + b)", engine=engine, parser=parser, inplace=True)
            got = df.e
            # 计算期望结果
            expect = np.sin(df.a + df.b).rename("e")
            # 断言得到的结果与期望结果相等
            tm.assert_series_equal(got, expect)

        # 使用 pytest 参数化多种数据类型进行测试
        @pytest.mark.parametrize(
            "dtype, expect_dtype",
            [
                (np.int32, np.float64),
                (np.int64, np.float64),
                (np.float32, np.float32),
                (np.float64, np.float64),
                pytest.param(np.complex128, np.complex128, marks=td.skip_if_windows),
            ],
        )
    # 定义一个测试函数，用于测试 DataFrame 的 eval 方法返回结果的数据类型是否符合预期
    def test_result_types(self, dtype, expect_dtype, engine, parser):
        # xref https://github.com/pandas-dev/pandas/issues/12293
        #  this fails on Windows, apparently a floating point precision issue
        # 引用外部链接，解释这段代码的背景，指出在 Windows 平台可能会因为浮点数精度问题导致失败

        # Did not test complex64 because DataFrame is converting it to
        # complex128. Due to https://github.com/pandas-dev/pandas/issues/10952
        # 由于 DataFrame 将 complex64 类型转换为 complex128，因此没有测试 complex64 类型的情况。
        # 原因参考 https://github.com/pandas-dev/pandas/issues/10952
        df = DataFrame(
            {"a": np.random.default_rng(2).standard_normal(10).astype(dtype)}
        )
        # 断言 DataFrame 列 'a' 的数据类型符合预期的 dtype
        assert df.a.dtype == dtype
        # 使用 eval 方法计算 'sin(a)'，并将结果赋值给新列 'b'，在原地进行操作
        df.eval("b = sin(a)", engine=engine, parser=parser, inplace=True)
        # 获取计算后的列 'b'
        got = df.b
        # 生成预期的 'sin(a)' 列，并命名为 'b'
        expect = np.sin(df.a).rename("b")
        # 断言预期结果的数据类型与实际结果的数据类型相符
        assert expect.dtype == got.dtype
        # 断言预期的结果数据类型与期望的结果数据类型相符
        assert expect_dtype == got.dtype
        # 使用测试工具函数确保实际结果与预期结果一致
        tm.assert_series_equal(got, expect)

    # 定义一个测试函数，用于测试 DataFrame 的 eval 方法在使用未定义的函数时是否会抛出 ValueError
    def test_undefined_func(self, engine, parser):
        df = DataFrame({"a": np.random.default_rng(2).standard_normal(10)})
        # 定义未定义函数 'mysin' 抛出的错误信息
        msg = '"mysin" is not a supported function'

        # 使用 pytest 的上下文管理器检查是否会抛出 ValueError，并匹配预期的错误信息
        with pytest.raises(ValueError, match=msg):
            df.eval("mysin(a)", engine=engine, parser=parser)

    # 定义一个测试函数，用于测试 DataFrame 的 eval 方法在使用不支持关键字参数的函数时是否会抛出 TypeError
    def test_keyword_arg(self, engine, parser):
        df = DataFrame({"a": np.random.default_rng(2).standard_normal(10)})
        # 定义函数 'sin' 不支持关键字参数时抛出的错误信息
        msg = 'Function "sin" does not support keyword arguments'

        # 使用 pytest 的上下文管理器检查是否会抛出 TypeError，并匹配预期的错误信息
        with pytest.raises(TypeError, match=msg):
            df.eval("sin(x=a)", engine=engine, parser=parser)
# 创建一个包含10个标准正态分布随机数的 NumPy 数组
_var_s = np.random.default_rng(2).standard_normal(10)


class TestScope:
    def test_global_scope(self, engine, parser):
        # 定义表达式字符串
        e = "_var_s * 2"
        # 使用 pd.eval 计算表达式并断言结果与 _var_s * 2 相等
        tm.assert_numpy_array_equal(
            _var_s * 2, pd.eval(e, engine=engine, parser=parser)
        )

    def test_no_new_locals(self, engine, parser):
        # 定义局部变量 x
        x = 1
        # 复制当前的局部变量字典
        lcls = locals().copy()
        # 使用 pd.eval 计算表达式 "x + 1"，传入局部变量字典，不影响原始的局部变量字典
        pd.eval("x + 1", local_dict=lcls, engine=engine, parser=parser)
        # 再次复制当前的局部变量字典
        lcls2 = locals().copy()
        # 移除 "lcls" 键，因为 pd.eval 中未引入新的局部变量
        lcls2.pop("lcls")
        # 断言两个局部变量字典相等
        assert lcls == lcls2

    def test_no_new_globals(self, engine, parser):
        # 定义全局变量 x
        x = 1  # noqa: F841
        # 复制当前的全局变量字典
        gbls = globals().copy()
        # 使用 pd.eval 计算表达式 "x + 1"，传入全局变量字典，不影响原始的全局变量字典
        pd.eval("x + 1", engine=engine, parser=parser)
        # 再次复制当前的全局变量字典
        gbls2 = globals().copy()
        # 断言两个全局变量字典相等
        assert gbls == gbls2

    def test_empty_locals(self, engine, parser):
        # GH 47084
        # 定义局部变量 x
        x = 1  # noqa: F841
        # 预期的错误消息
        msg = "name 'x' is not defined"
        # 使用 pd.eval 计算表达式 "x + 1"，传入空的局部变量字典，预期抛出 UndefinedVariableError 异常，匹配错误消息
        with pytest.raises(UndefinedVariableError, match=msg):
            pd.eval("x + 1", engine=engine, parser=parser, local_dict={})

    def test_empty_globals(self, engine, parser):
        # GH 47084
        # 预期的错误消息
        msg = "name '_var_s' is not defined"
        # 定义表达式字符串
        e = "_var_s * 2"
        # 使用 pd.eval 计算表达式，传入空的全局变量字典，预期抛出 UndefinedVariableError 异常，匹配错误消息
        with pytest.raises(UndefinedVariableError, match=msg):
            pd.eval(e, engine=engine, parser=parser, global_dict={})


@td.skip_if_no("numexpr")
def test_invalid_engine():
    # 预期的错误消息
    msg = "Invalid engine 'asdf' passed"
    # 使用 pd.eval 计算表达式 "x + y"，传入自定义的局部变量字典和无效的计算引擎，预期抛出 KeyError 异常，匹配错误消息
    with pytest.raises(KeyError, match=msg):
        pd.eval("x + y", local_dict={"x": 1, "y": 2}, engine="asdf")


@td.skip_if_no("numexpr")
@pytest.mark.parametrize(
    ("use_numexpr", "expected"),
    (
        (True, "numexpr"),
        (False, "python"),
    ),
)
def test_numexpr_option_respected(use_numexpr, expected):
    # GH 32556
    # 导入检查引擎的函数
    from pandas.core.computation.eval import _check_engine

    # 设置计算选项上下文，用于设定是否使用 numexpr 引擎
    with pd.option_context("compute.use_numexpr", use_numexpr):
        # 检查当前使用的引擎
        result = _check_engine(None)
        # 断言检查结果与预期是否一致
        assert result == expected


@td.skip_if_no("numexpr")
def test_numexpr_option_incompatible_op():
    # GH 32556
    # 设置计算选项上下文，强制不使用 numexpr 引擎
    with pd.option_context("compute.use_numexpr", False):
        # 创建 DataFrame 对象
        df = DataFrame(
            {"A": [True, False, True, False, None, None], "B": [1, 2, 3, 4, 5, 6]}
        )
        # 使用 query 方法查询满足条件 A.isnull() 的行
        result = df.query("A.isnull()")
        # 预期的 DataFrame 结果
        expected = DataFrame({"A": [None, None], "B": [5, 6]}, index=[4, 5])
        # 断言结果 DataFrame 与预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numexpr")
def test_invalid_parser():
    # 预期的错误消息
    msg = "Invalid parser 'asdf' passed"
    # 使用 pd.eval 计算表达式 "x + y"，传入自定义的局部变量字典和无效的解析器，预期抛出 KeyError 异常，匹配错误消息
    with pytest.raises(KeyError, match=msg):
        pd.eval("x + y", local_dict={"x": 1, "y": 2}, parser="asdf")


# 创建一个字典，映射不同的解析器名称到对应的访问器类
_parsers: dict[str, type[BaseExprVisitor]] = {
    "python": PythonExprVisitor,
    "pytables": pytables.PyTablesExprVisitor,
    "pandas": PandasExprVisitor,
}


# 使用参数化测试，组合不同的引擎和解析器
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("parser", _parsers)
def test_disallowed_nodes(engine, parser):
    # 获取当前解析器对应的访问器类
    VisitorClass = _parsers[parser]
    # 创建访问器类的实例，用于处理表达式 "x + 1"
    inst = VisitorClass("x + 1", engine, parser)
    # 遍历 VisitorClass.unsupported_nodes 中的每个元素
    for ops in VisitorClass.unsupported_nodes:
        # 设置错误消息文本
        msg = "nodes are not implemented"
        # 使用 pytest 来检查调用 inst 对象的 ops 方法时是否会抛出 NotImplementedError，并匹配指定的错误消息
        with pytest.raises(NotImplementedError, match=msg):
            # 使用 getattr 获取 inst 对象的 ops 方法，并调用该方法
            getattr(inst, ops)()
# 测试语法错误表达式的函数
def test_syntax_error_exprs(engine, parser):
    # 定义一个含有语法错误的表达式
    e = "s +"  
    # 使用 pytest 检查是否会抛出 SyntaxError 异常，匹配特定错误信息
    with pytest.raises(SyntaxError, match="invalid syntax"):
        pd.eval(e, engine=engine, parser=parser)


# 测试名称错误表达式的函数
def test_name_error_exprs(engine, parser):
    # 定义一个含有未定义变量的表达式
    e = "s + t"
    msg = "name 's' is not defined"
    # 使用 pytest 检查是否会抛出 NameError 异常，匹配特定错误信息
    with pytest.raises(NameError, match=msg):
        pd.eval(e, engine=engine, parser=parser)


# 使用参数化测试多种情况下的无效局部变量引用
@pytest.mark.parametrize("express", ["a + @b", "@a + b", "@a + @b"])
def test_invalid_local_variable_reference(engine, parser, express):
    # 定义局部变量 a 和 b
    a, b = 1, 2  # noqa: F841

    # 根据不同的解析器执行不同的测试
    if parser != "pandas":
        # 当解析器不是 pandas 时，检查是否会抛出 SyntaxError 异常，匹配特定错误信息
        with pytest.raises(SyntaxError, match="The '@' prefix is only"):
            pd.eval(express, engine=engine, parser=parser)
    else:
        # 当解析器是 pandas 时，检查是否会抛出 SyntaxError 异常，匹配特定错误信息
        with pytest.raises(SyntaxError, match="The '@' prefix is not"):
            pd.eval(express, engine=engine, parser=parser)


# 测试 numexpr 引擎中的内置函数引发异常
def test_numexpr_builtin_raises(engine, parser):
    sin, dotted_line = 1, 2
    if engine == "numexpr":
        msg = "Variables in expression .+"
        # 使用 pytest 检查是否会抛出 NumExprClobberingError 异常，匹配特定错误信息
        with pytest.raises(NumExprClobberingError, match=msg):
            pd.eval("sin + dotted_line", engine=engine, parser=parser)
    else:
        # 使用默认引擎执行表达式，确保结果正确
        res = pd.eval("sin + dotted_line", engine=engine, parser=parser)
        assert res == sin + dotted_line


# 测试无法解析的解析器类型引发异常
def test_bad_resolver_raises(engine, parser):
    cannot_resolve = 42, 3.0
    # 使用 pytest 检查是否会抛出 TypeError 异常，匹配特定错误信息
    with pytest.raises(TypeError, match="Resolver of type .+"):
        pd.eval("1 + 2", resolvers=cannot_resolve, engine=engine, parser=parser)


# 测试空字符串表达式引发异常
def test_empty_string_raises(engine, parser):
    # GH 13139
    # 使用 pytest 检查是否会抛出 ValueError 异常，匹配特定错误信息
    with pytest.raises(ValueError, match="expr cannot be an empty string"):
        pd.eval("", engine=engine, parser=parser)


# 测试多于一个表达式的情况引发异常
def test_more_than_one_expression_raises(engine, parser):
    # 使用 pytest 检查是否会抛出 SyntaxError 异常，匹配特定错误信息
    with pytest.raises(SyntaxError, match="only a single expression is allowed"):
        pd.eval("1 + 1; 2 + 2", engine=engine, parser=parser)


# 使用参数化测试各种情况下的布尔操作在标量上失败
@pytest.mark.parametrize("cmp", ("and", "or"))
@pytest.mark.parametrize("lhs", (int, float))
@pytest.mark.parametrize("rhs", (int, float))
def test_bool_ops_fails_on_scalars(lhs, cmp, rhs, engine, parser):
    # 定义生成不同类型的随机数的函数
    gen = {
        int: lambda: np.random.default_rng(2).integers(10),
        float: np.random.default_rng(2).standard_normal,
    }

    mid = gen[lhs]()  # noqa: F841
    lhs = gen[lhs]()
    rhs = gen[rhs]()

    ex1 = f"lhs {cmp} mid {cmp} rhs"
    ex2 = f"lhs {cmp} mid and mid {cmp} rhs"
    ex3 = f"(lhs {cmp} mid) & (mid {cmp} rhs)"
    # 针对每个表达式执行测试，检查是否会抛出 NotImplementedError 异常，匹配特定错误信息
    for ex in (ex1, ex2, ex3):
        msg = "cannot evaluate scalar only bool ops|'BoolOp' nodes are not"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex, engine=engine, parser=parser)


# 使用参数化测试不同情况下的相等性测试
@pytest.mark.parametrize(
    "other",
    [
        "'x'",
        "...",
    ],
)
def test_equals_various(other):
    # 创建包含对象的 DataFrame
    df = DataFrame({"A": ["a", "b", "c"]}, dtype=object)
    # 执行等式测试，确保结果与预期相符
    result = df.eval(f"A == {other}")
    expected = Series([False, False, False], name="A")
    tm.assert_series_equal(result, expected)


# 测试无穷大数值运算
def test_inf(engine, parser):
    # 定义包含无穷大数值的表达式
    s = "inf + 1"
    # 设置期望结果为正无穷大
    expected = np.inf
    # 使用指定的引擎和解析器对字符串进行求值，将结果赋给result
    result = pd.eval(s, engine=engine, parser=parser)
    # 断言结果与期望结果相等
    assert result == expected
@pytest.mark.parametrize("column", ["Temp(°C)", "Capacitance(μF)"])
# 使用 pytest 的 parametrize 装饰器，对参数 column 进行多组测试
def test_query_token(engine, column):
    # 创建一个 DataFrame，包含两列随机正态分布数据，其中一列名为 column，另一列为 "b"
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 2)), columns=[column, "b"]
    )
    # 根据指定列 column 的条件筛选出符合条件的行
    expected = df[df[column] > 5]
    # 生成查询字符串，用于在 DataFrame 中查询大于 5 的 column 列数据
    query_string = f"`{column}` > 5"
    # 使用 query 方法进行查询，指定引擎 engine
    result = df.query(query_string, engine=engine)
    # 使用 pytest 的 assert_frame_equal 方法比较查询结果和预期结果
    tm.assert_frame_equal(result, expected)


def test_negate_lt_eq_le(engine, parser):
    # 创建一个 DataFrame，包含两列数据，列名分别为 "cat" 和 "count"
    df = DataFrame([[0, 10], [1, 20]], columns=["cat", "count"])
    # 根据条件 ~(df.cat > 0) 对 DataFrame 进行筛选，即取反操作
    expected = df[~(df.cat > 0)]

    # 使用 query 方法对 "~(cat > 0)" 进行查询，指定引擎 engine 和解析器 parser
    result = df.query("~(cat > 0)", engine=engine, parser=parser)
    # 使用 pytest 的 assert_frame_equal 方法比较查询结果和预期结果
    tm.assert_frame_equal(result, expected)

    # 如果解析器为 "python"，则测试抛出 NotImplementedError 异常
    if parser == "python":
        msg = "'Not' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            df.query("not (cat > 0)", engine=engine, parser=parser)
    else:
        # 否则继续使用 query 方法对 "not (cat > 0)" 进行查询
        result = df.query("not (cat > 0)", engine=engine, parser=parser)
        # 使用 pytest 的 assert_frame_equal 方法比较查询结果和预期结果
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "column",
    DEFAULT_GLOBALS.keys(),
)
# 使用 pytest 的 parametrize 装饰器，对 DEFAULT_GLOBALS 字典的键进行测试
def test_eval_no_support_column_name(request, column):
    # 如果 column 在指定列表中，则对测试标记为失败，抛出 KeyError 异常
    if column in ["True", "False", "inf", "Inf"]:
        request.applymarker(
            pytest.mark.xfail(
                raises=KeyError,
                reason=f"GH 47859 DataFrame eval not supported with {column}",
            )
        )

    # 创建一个 DataFrame，包含两列随机整数数据，列名分别为 column 和 "col1"
    df = DataFrame(
        np.random.default_rng(2).integers(0, 100, size=(10, 2)),
        columns=[column, "col1"],
    )
    # 根据 column 列的条件筛选出大于 6 的行
    expected = df[df[column] > 6]
    # 使用 query 方法进行查询，查询条件为 f"{column}>6"
    result = df.query(f"{column}>6")

    # 使用 pytest 的 assert_frame_equal 方法比较查询结果和预期结果
    tm.assert_frame_equal(result, expected)


def test_set_inplace():
    # 创建一个 DataFrame，包含三列数据，列名分别为 "A"、"B"、"C"
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    # 创建一个 DataFrame 的视图，复制 df 的所有内容
    result_view = df[:]
    # 创建一个 Series，引用 df 中的 "A" 列数据
    ser = df["A"]
    # 使用 eval 方法对 df 中的 "A" 列进行计算并更新，同时 inplace=True
    df.eval("A = B + C", inplace=True)
    # 创建一个预期的 DataFrame，包含更新后的 "A" 列和其他列数据
    expected = DataFrame({"A": [11, 13, 15], "B": [4, 5, 6], "C": [7, 8, 9]})
    # 使用 pytest 的 assert_frame_equal 方法比较 df 和预期的 DataFrame
    tm.assert_frame_equal(df, expected)
    # 创建一个预期的 Series，包含未修改的 "A" 列数据
    expected = Series([1, 2, 3], name="A")
    # 使用 pytest 的 assert_series_equal 方法比较 ser 和预期的 Series
    tm.assert_series_equal(ser, expected)
    # 使用 pytest 的 assert_series_equal 方法比较 result_view 中的 "A" 列和预期的 Series
    tm.assert_series_equal(result_view["A"], expected)


@pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
# 使用 pytest 的 parametrize 装饰器，对参数 value 进行多组测试
def test_validate_bool_args(value):
    # 准备用于抛出异常的消息字符串
    msg = 'For argument "inplace" expected type bool, received type'
    # 使用 pytest 的 raises 方法断言 pd.eval("2+2", inplace=value) 抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        pd.eval("2+2", inplace=value)
```