# `D:\src\scipysrc\pandas\pandas\tests\test_expressions.py`

```
# 导入运算符模块，用于动态获取操作函数
import operator
# 导入正则表达式模块
import re

# 导入NumPy库，使用别名np
import numpy as np
# 导入pytest测试框架
import pytest

# 导入pandas库中的option_context函数
from pandas import option_context
# 导入pandas库中的_testing模块，使用别名tm
import pandas._testing as tm
# 导入pandas库中的DataFrame类
from pandas.core.api import DataFrame
# 导入pandas库中的expressions模块，使用别名expr
from pandas.core.computation import expressions as expr


@pytest.fixture
# 定义_fixture修饰的函数_frame，返回一个包含随机数的DataFrame对象
def _frame():
    return DataFrame(
        np.random.default_rng(2).standard_normal((10001, 4)),
        columns=list("ABCD"),
        dtype="float64",
    )


@pytest.fixture
# 定义_fixture修饰的函数_frame2，返回一个包含随机数的DataFrame对象
def _frame2():
    return DataFrame(
        np.random.default_rng(2).standard_normal((100, 4)),
        columns=list("ABCD"),
        dtype="float64",
    )


@pytest.fixture
# 定义_fixture修饰的函数_mixed，返回一个包含不同数据类型的DataFrame对象
def _mixed(_frame):
    return DataFrame(
        {
            "A": _frame["A"],
            "B": _frame["B"].astype("float32"),
            "C": _frame["C"].astype("int64"),
            "D": _frame["D"].astype("int32"),
        }
    )


@pytest.fixture
# 定义_fixture修饰的函数_mixed2，返回一个包含不同数据类型的DataFrame对象
def _mixed2(_frame2):
    return DataFrame(
        {
            "A": _frame2["A"],
            "B": _frame2["B"].astype("float32"),
            "C": _frame2["C"].astype("int64"),
            "D": _frame2["D"].astype("int32"),
        }
    )


@pytest.fixture
# 定义_fixture修饰的函数_integer，返回一个包含随机整数的DataFrame对象
def _integer():
    return DataFrame(
        np.random.default_rng(2).integers(1, 100, size=(10001, 4)),
        columns=list("ABCD"),
        dtype="int64",
    )


@pytest.fixture
# 定义_fixture修饰的函数_integer_integers，返回一个根据_integer生成的包含零的DataFrame对象
def _integer_integers(_integer):
    # integers to get a case with zeros
    return _integer * np.random.default_rng(2).integers(0, 2, size=np.shape(_integer))


@pytest.fixture
# 定义_fixture修饰的函数_integer2，返回一个包含随机整数的DataFrame对象
def _integer2():
    return DataFrame(
        np.random.default_rng(2).integers(1, 100, size=(101, 4)),
        columns=list("ABCD"),
        dtype="int64",
    )


@pytest.fixture
# 定义_fixture修饰的函数_array，返回_frame中'A'列的NumPy数组
def _array(_frame):
    return _frame["A"].to_numpy()


@pytest.fixture
# 定义_fixture修饰的函数_array2，返回_frame2中'A'列的NumPy数组
def _array2(_frame2):
    return _frame2["A"].to_numpy()


@pytest.fixture
# 定义_fixture修饰的函数_array_mixed，返回_mixed中'D'列的NumPy数组
def _array_mixed(_mixed):
    return _mixed["D"].to_numpy()


@pytest.fixture
# 定义_fixture修饰的函数_array_mixed2，返回_mixed2中'D'列的NumPy数组
def _array_mixed2(_mixed2):
    return _mixed2["D"].to_numpy()


@pytest.mark.skipif(not expr.USE_NUMEXPR, reason="not using numexpr")
# 定义一个测试类TestExpressions，标记为跳过如果不使用numexpr
class TestExpressions:
    @staticmethod
    # 定义静态方法call_op，接受DataFrame对象df、other，布尔值flex和字符串opname作为参数
    def call_op(df, other, flex: bool, opname: str):
        if flex:
            # 如果flex为True，定义一个匿名函数op，使用getattr获取操作函数
            op = lambda x, y: getattr(x, opname)(y)
            op.__name__ = opname
        else:
            # 如果flex为False，直接从operator模块获取操作函数
            op = getattr(operator, opname)

        # 使用option_context函数设置上下文，设置"compute.use_numexpr"为False
        with option_context("compute.use_numexpr", False):
            # 执行操作函数op，并将结果赋给变量expected
            expected = op(df, other)

        # 调用expressions模块的get_test_result函数
        expr.get_test_result()

        # 再次执行操作函数op，并将结果赋给变量result
        result = op(df, other)
        return result, expected

    @pytest.mark.parametrize(
        "fixture",
        [
            "_integer",
            "_integer2",
            "_integer_integers",
            "_frame",
            "_frame2",
            "_mixed",
            "_mixed2",
        ],
    )
    @pytest.mark.parametrize("flex", [True, False])
    @pytest.mark.parametrize(
        "arith", ["add", "sub", "mul", "mod", "truediv", "floordiv"]
    )
    # 定义测试函数，用于测试算术操作
    def test_run_arithmetic(self, request, fixture, flex, arith, monkeypatch):
        # 获取测试数据集
        df = request.getfixturevalue(fixture)
        
        # 使用 monkeypatch 修改表达式模块的_MIN_ELEMENTS属性为0
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)
            
            # 调用算术操作函数，并获取结果和期望值
            result, expected = self.call_op(df, df, flex, arith)
            
            # 对于 'truediv' 操作，验证期望值的所有数据类型为浮点型
            if arith == "truediv":
                assert all(x.kind == "f" for x in expected.dtypes.values)
            
            # 使用测试模块的断言函数验证结果与期望值是否相等
            tm.assert_equal(expected, result)
            
            # 遍历数据集的列
            for i in range(len(df.columns)):
                # 再次调用算术操作函数，对单列进行操作，获取结果和期望值
                result, expected = self.call_op(
                    df.iloc[:, i], df.iloc[:, i], flex, arith
                )
                # 对于 'truediv' 操作，验证期望值的数据类型为浮点型
                if arith == "truediv":
                    assert expected.dtype.kind == "f"
                
                # 使用测试模块的断言函数验证结果与期望值是否相等
                tm.assert_equal(expected, result)

    # 参数化测试函数，用于测试二元操作
    @pytest.mark.parametrize(
        "fixture",
        [
            "_integer",
            "_integer2",
            "_integer_integers",
            "_frame",
            "_frame2",
            "_mixed",
            "_mixed2",
        ],
    )
    @pytest.mark.parametrize("flex", [True, False])
    def test_run_binary(self, request, fixture, flex, comparison_op, monkeypatch):
        """
        tests solely that the result is the same whether or not numexpr is
        enabled.  Need to test whether the function does the correct thing
        elsewhere.
        """
        # 获取测试数据集
        df = request.getfixturevalue(fixture)
        
        # 获取操作名称
        arith = comparison_op.__name__
        
        # 禁用 numexpr 后，计算 df + 1
        with option_context("compute.use_numexpr", False):
            other = df + 1
        
        # 使用 monkeypatch 修改表达式模块的_MIN_ELEMENTS属性为0
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)
            expr.set_test_mode(True)
            
            # 调用二元操作函数，获取结果和期望值
            result, expected = self.call_op(df, other, flex, arith)
            
            # 获取测试模块中的测试结果
            used_numexpr = expr.get_test_result()
            
            # 验证是否按预期使用了 numexpr
            assert used_numexpr, "Did not use numexpr as expected."
            
            # 使用测试模块的断言函数验证结果与期望值是否相等
            tm.assert_equal(expected, result)
            
            # 遍历数据集的列
            for i in range(len(df.columns)):
                # 计算 other 的单列加 1
                binary_comp = other.iloc[:, i] + 1
                # 再次调用二元操作函数，验证结果与期望值是否相等
                self.call_op(df.iloc[:, i], binary_comp, flex, "add")

    # 测试无效情况
    def test_invalid(self):
        # 创建随机数数组
        array = np.random.default_rng(2).standard_normal(1_000_001)
        array2 = np.random.default_rng(2).standard_normal(100)
        
        # 测试无操作情况
        result = expr._can_use_numexpr(operator.add, None, array, array, "evaluate")
        assert not result
        
        # 测试最小元素情况
        result = expr._can_use_numexpr(operator.add, "+", array2, array2, "evaluate")
        assert not result
        
        # 只检查表达式的第一部分是否可用
        result = expr._can_use_numexpr(operator.add, "+", array, array2, "evaluate")
        assert result

    # 参数化测试函数，用于测试不同操作的运行
    @pytest.mark.filterwarnings("ignore:invalid value encountered in:RuntimeWarning")
    @pytest.mark.parametrize(
        "opname,op_str",
        [("add", "+"), ("sub", "-"), ("mul", "*"), ("truediv", "/"), ("pow", "**")],
    )
    @pytest.mark.parametrize(
        "left_fix,right_fix", [("_array", "_array2"), ("_array_mixed", "_array_mixed2")]
    )
    # 定义一个测试二进制运算的方法，接受多个参数，包括请求、操作名、操作字符串、左侧修复和右侧修复
    def test_binary_ops(self, request, opname, op_str, left_fix, right_fix):
        # 使用 request 对象获取左侧和右侧修复的值
        left = request.getfixturevalue(left_fix)
        right = request.getfixturevalue(right_fix)

        # 定义一个内部函数 testit，用于执行测试
        def testit(left, right, opname, op_str):
            # 如果操作名是 "pow"，则将左侧值取绝对值
            if opname == "pow":
                left = np.abs(left)

            # 获取操作符对应的函数对象
            op = getattr(operator, opname)

            # 用 numexpr 执行表达式的评估，返回结果
            result = expr.evaluate(op, left, left, use_numexpr=True)
            # 不使用 numexpr 执行表达式的评估，返回期望结果
            expected = expr.evaluate(op, left, left, use_numexpr=False)
            # 断言两个 numpy 数组相等
            tm.assert_numpy_array_equal(result, expected)

            # 使用 numexpr 判断是否可以使用 numexpr 进行计算
            result = expr._can_use_numexpr(op, op_str, right, right, "evaluate")
            assert not result  # 断言结果为 False

        # 关闭 compute.use_numexpr 上下文，执行 testit 函数
        with option_context("compute.use_numexpr", False):
            testit(left, right, opname, op_str)

        # 设置 numexpr 线程数为 1，执行 testit 函数
        expr.set_numexpr_threads(1)
        testit(left, right, opname, op_str)
        # 重置 numexpr 线程数，执行 testit 函数
        expr.set_numexpr_threads()
        testit(left, right, opname, op_str)

    # 使用 pytest 的参数化装饰器，定义测试比较操作的方法，接受请求、比较操作、左侧修复和右侧修复作为参数
    @pytest.mark.parametrize(
        "left_fix,right_fix", [("_array", "_array2"), ("_array_mixed", "_array_mixed2")]
    )
    def test_comparison_ops(self, request, comparison_op, left_fix, right_fix):
        # 使用 request 对象获取左侧和右侧修复的值
        left = request.getfixturevalue(left_fix)
        right = request.getfixturevalue(right_fix)

        # 定义一个内部函数 testit，用于执行测试
        def testit():
            # 对左侧和右侧的值分别加 1
            f12 = left + 1
            f22 = right + 1

            # 获取比较操作对应的函数对象
            op = comparison_op

            # 用 numexpr 执行表达式的评估，返回结果
            result = expr.evaluate(op, left, f12, use_numexpr=True)
            # 不使用 numexpr 执行表达式的评估，返回期望结果
            expected = expr.evaluate(op, left, f12, use_numexpr=False)
            # 断言两个 numpy 数组相等
            tm.assert_numpy_array_equal(result, expected)

            # 使用 numexpr 判断是否可以使用 numexpr 进行计算
            result = expr._can_use_numexpr(op, op, right, f22, "evaluate")
            assert not result  # 断言结果为 False

        # 关闭 compute.use_numexpr 上下文，执行 testit 函数
        with option_context("compute.use_numexpr", False):
            testit()

        # 设置 numexpr 线程数为 1，执行 testit 函数
        expr.set_numexpr_threads(1)
        testit()
        # 重置 numexpr 线程数，执行 testit 函数
        expr.set_numexpr_threads()
        testit()

    # 使用 pytest 的参数化装饰器，定义测试 where 函数的方法，接受请求、条件和数据框的修复作为参数
    @pytest.mark.parametrize("cond", [True, False])
    @pytest.mark.parametrize("fixture", ["_frame", "_frame2", "_mixed", "_mixed2"])
    def test_where(self, request, cond, fixture):
        # 使用 request 对象获取数据框的值
        df = request.getfixturevalue(fixture)

        # 定义一个内部函数 testit，用于执行测试
        def testit():
            # 创建一个布尔类型的空数组，填充为条件值
            c = np.empty(df.shape, dtype=np.bool_)
            c.fill(cond)
            # 使用 expr 的 where 函数进行条件操作，返回结果
            result = expr.where(c, df.values, df.values + 1)
            # 使用 numpy 的 where 函数进行条件操作，返回期望结果
            expected = np.where(c, df.values, df.values + 1)
            # 断言两个 numpy 数组相等
            tm.assert_numpy_array_equal(result, expected)

        # 关闭 compute.use_numexpr 上下文，执行 testit 函数
        with option_context("compute.use_numexpr", False):
            testit()

        # 设置 numexpr 线程数为 1，执行 testit 函数
        expr.set_numexpr_threads(1)
        testit()
        # 重置 numexpr 线程数，执行 testit 函数
        expr.set_numexpr_threads()
        testit()

    # 使用 pytest 的参数化装饰器，定义测试不同数学运算操作的方法，接受操作字符串和操作名作为参数
    @pytest.mark.parametrize(
        "op_str,opname", [("/", "truediv"), ("//", "floordiv"), ("**", "pow")]
    )
    # 定义一个测试函数，用于测试布尔运算符在 DataFrame 上是否能正确抛出算术操作未实现的错误
    def test_bool_ops_raise_on_arithmetic(self, op_str, opname):
        # 创建一个包含随机布尔值的 DataFrame 对象 df
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(10) > 0.5,
                "b": np.random.default_rng(2).random(10) > 0.5,
            }
        )

        # 准备错误消息，指明当布尔数据类型的操作未实现时应抛出 NotImplementedError
        msg = f"operator '{opname}' not implemented for bool dtypes"
        # 获取操作符对应的函数对象
        f = getattr(operator, opname)
        # 将错误消息转义，以便在 pytest 的异常检测中进行匹配
        err_msg = re.escape(msg)

        # 以下为使用 pytest 的 assert_raises 检测各种情况下是否抛出了正确的异常

        # 在整个 DataFrame 上应用操作符函数 f，并期望抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=err_msg):
            f(df, df)

        # 在单独列上应用操作符函数 f，并期望抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=err_msg):
            f(df.a, df.b)

        # 在单列与标量布尔值上应用操作符函数 f，并期望抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=err_msg):
            f(df.a, True)

        # 在标量布尔值与单列上应用操作符函数 f，并期望抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=err_msg):
            f(False, df.a)

        # 在标量布尔值与整个 DataFrame 上应用操作符函数 f，并期望抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=err_msg):
            f(False, df)

        # 在整个 DataFrame 与标量布尔值上应用操作符函数 f，并期望抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=err_msg):
            f(df, True)

    # 使用 pytest 的参数化装饰器，测试布尔运算符在算术操作中是否能正确产生警告
    @pytest.mark.parametrize(
        "op_str,opname", [("+", "add"), ("*", "mul"), ("-", "sub")]
    )
    def test_bool_ops_warn_on_arithmetic(self, op_str, opname, monkeypatch):
        # 准备数据集大小
        n = 10
        # 创建包含随机布尔值的 DataFrame 对象 df
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(n) > 0.5,
                "b": np.random.default_rng(2).random(n) > 0.5,
            }
        )

        # 定义替换字典，用于在特定操作下替换操作符和函数
        subs = {"+": "|", "*": "&", "-": "^"}
        sub_funcs = {"|": "or_", "&": "and_", "^": "xor"}

        # 获取原始操作符对应的函数对象和替换操作符后对应的函数对象
        f = getattr(operator, opname)
        fe = getattr(operator, sub_funcs[subs[op_str]])

        # 对于减法操作，直接返回，不进行警告检测
        if op_str == "-":
            return

        # 准备警告消息，指明在使用 numexpr 时不支持的操作
        msg = "operator is not supported by numexpr"
        # 使用 monkeypatch 设置 numexpr 相关的上下文，并检测警告消息
        with monkeypatch.context() as m:
            # 设置模拟表达式对象的最小元素数量
            m.setattr(expr, "_MIN_ELEMENTS", 5)
            # 使用 numexpr 执行计算，并期望产生 UserWarning 警告，匹配警告消息
            with option_context("compute.use_numexpr", True):
                with tm.assert_produces_warning(UserWarning, match=msg):
                    # 使用原始操作符函数 f 在整个 DataFrame 上执行计算
                    r = f(df, df)
                    # 使用替换后的操作符函数 fe 在整个 DataFrame 上执行计算
                    e = fe(df, df)
                    # 检测计算结果是否相等
                    tm.assert_frame_equal(r, e)

                # 使用 numexpr 执行计算，并期望产生 UserWarning 警告，匹配警告消息
                with tm.assert_produces_warning(UserWarning, match=msg):
                    # 使用原始操作符函数 f 在单独列上执行计算
                    r = f(df.a, df.b)
                    # 使用替换后的操作符函数 fe 在单独列上执行计算
                    e = fe(df.a, df.b)
                    # 检测计算结果是否相等
                    tm.assert_series_equal(r, e)

                # 使用 numexpr 执行计算，并期望产生 UserWarning 警告，匹配警告消息
                with tm.assert_produces_warning(UserWarning, match=msg):
                    # 使用原始操作符函数 f 在单列与标量布尔值上执行计算
                    r = f(df.a, True)
                    # 使用替换后的操作符函数 fe 在单列与标量布尔值上执行计算
                    e = fe(df.a, True)
                    # 检测计算结果是否相等
                    tm.assert_series_equal(r, e)

                # 使用 numexpr 执行计算，并期望产生 UserWarning 警告，匹配警告消息
                with tm.assert_produces_warning(UserWarning, match=msg):
                    # 使用原始操作符函数 f 在标量布尔值与单列上执行计算
                    r = f(False, df.a)
                    # 使用替换后的操作符函数 fe 在标量布尔值与单列上执行计算
                    e = fe(False, df.a)
                    # 检测计算结果是否相等
                    tm.assert_series_equal(r, e)

                # 使用 numexpr 执行计算，并期望产生 UserWarning 警告，匹配警告消息
                with tm.assert_produces_warning(UserWarning, match=msg):
                    # 使用原始操作符函数 f 在标量布尔值与整个 DataFrame 上执行计算
                    r = f(False, df)
                    # 使用替换后的操作符函数 fe 在标量布尔值与整个 DataFrame 上执行计算
                    e = fe(False, df)
                    # 检测计算结果是否相等
                    tm.assert_frame_equal(r, e)

                # 使用 numexpr 执行计算，并期望产生 UserWarning 警告，匹配警告消息
                with tm.assert_produces_warning(UserWarning, match=msg):
                    # 使用原始操作符函数 f 在整个 DataFrame 与标量布尔值上执行计算
                    r = f(df, True)
                    # 使用替换后的操作符函数 fe 在整个 DataFrame 与标量布尔值上执行计算
                    e = fe(df, True)
                    # 检测计算结果是否相等
                    tm.assert_frame_equal(r, e)
    @pytest.mark.parametrize(
        "test_input,expected",
        [  # 参数化测试输入和期望输出
            (
                DataFrame(
                    [[0, 1, 2, "aa"], [0, 1, 2, "aa"]], columns=["a", "b", "c", "dtype"]
                ),
                DataFrame([[False, False], [False, False]], columns=["a", "dtype"]),
            ),
            (
                DataFrame(
                    [[0, 3, 2, "aa"], [0, 4, 2, "aa"], [0, 1, 1, "bb"]],
                    columns=["a", "b", "c", "dtype"],
                ),
                DataFrame(
                    [[False, False], [False, False], [False, False]],
                    columns=["a", "dtype"],
                ),
            ),
        ],
    )
    def test_bool_ops_column_name_dtype(self, test_input, expected):
        # 测试布尔运算对含有列名 'dtype' 的DataFrame的影响
        result = test_input.loc[:, ["a", "dtype"]].ne(test_input.loc[:, ["a", "dtype"]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "arith", ("add", "sub", "mul", "mod", "truediv", "floordiv")
    )
    @pytest.mark.parametrize("axis", (0, 1))
    def test_frame_series_axis(self, axis, arith, _frame, monkeypatch):
        # GH#26736 Dataframe.floordiv(Series, axis=1) 失败的问题

        df = _frame
        if axis == 1:
            other = df.iloc[0, :]
        else:
            other = df.iloc[:, 0]

        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 0)

            op_func = getattr(df, arith)

            with option_context("compute.use_numexpr", False):
                expected = op_func(other, axis=axis)

            result = op_func(other, axis=axis)
            tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize(
        "op",
        [
            "__mod__",
            "__rmod__",
            "__floordiv__",
            "__rfloordiv__",
        ],
    )
    @pytest.mark.parametrize("scalar", [-5, 5])
    def test_python_semantics_with_numexpr_installed(
        self, op, box_with_array, scalar, monkeypatch
    ):
        # 测试使用安装了 numexpr 的Python语义
        # 使用 monkeypatch 上下文来模拟设置表达式的最小元素数为 0
        with monkeypatch.context() as m:
            # 设置表达式对象的 _MIN_ELEMENTS 属性为 0
            m.setattr(expr, "_MIN_ELEMENTS", 0)
            
            # 创建一个包含从 -50 到 49 的整数的 NumPy 数组
            data = np.arange(-50, 50)
            
            # 使用数据创建一个 box_with_array 对象
            obj = box_with_array(data)
            
            # 获取 box_with_array 对象上的特定操作的方法
            method = getattr(obj, op)
            
            # 使用 scalar 执行特定操作
            result = method(scalar)

            # 与 NumPy 计算得到的期望结果进行比较
            with option_context("compute.use_numexpr", False):
                # 使用 scalar 执行特定操作得到期望结果
                expected = method(scalar)

            # 使用 TestManager 的 assert_equal 方法比较结果与期望值
            tm.assert_equal(result, expected)

            # 逐个元素地与 Python 计算结果进行比较
            for i, elem in enumerate(data):
                # 如果 box_with_array 是 DataFrame 类型
                if box_with_array == DataFrame:
                    # 获取结果 DataFrame 中的特定标量结果
                    scalar_result = result.iloc[i, 0]
                else:
                    # 否则获取结果数组中的特定标量结果
                    scalar_result = result[i]
                
                try:
                    # 尝试计算 Python 中整数 elem 和 scalar 执行特定操作的结果
                    expected = getattr(int(elem), op)(scalar)
                except ZeroDivisionError:
                    # 如果遇到 ZeroDivisionError 则忽略异常
                    pass
                else:
                    # 否则断言 scalar_result 等于期望的结果
                    assert scalar_result == expected
```