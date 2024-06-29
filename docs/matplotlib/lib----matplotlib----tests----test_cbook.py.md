# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_cbook.py`

```
# 从未来版本导入注解功能，用于类型提示
from __future__ import annotations

# 导入系统模块
import sys
# 导入迭代工具模块
import itertools
# 导入 pickle 序列化模块
import pickle

# 导入类型提示模块中的 Any 类型
from typing import Any
# 导入 unittest.mock 模块中的 patch 和 Mock 类
from unittest.mock import patch, Mock

# 导入日期时间模块中的 datetime, date, timedelta 类
from datetime import datetime, date, timedelta

# 导入 NumPy 库及其测试工具
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
                           assert_array_almost_equal)
# 导入 pytest 测试框架
import pytest

# 导入 Matplotlib 库中的 _api, cbook 模块
from matplotlib import _api, cbook
# 导入 Matplotlib 颜色模块
import matplotlib.colors as mcolors
# 从 Matplotlib.cbook 模块导入 delete_masked_points, strip_math 函数
from matplotlib.cbook import delete_masked_points, strip_math

# 导入 ModuleType 类型
from types import ModuleType


class Test_delete_masked_points:
    # 测试 delete_masked_points 函数的异常情况
    def test_bad_first_arg(self):
        # 断言调用 delete_masked_points 函数时抛出 ValueError 异常
        with pytest.raises(ValueError):
            delete_masked_points('a string', np.arange(1.0, 7.0))

    # 测试 delete_masked_points 函数对字符串序列的操作
    def test_string_seq(self):
        # 定义字符串序列 a1 和混合序列 a2
        a1 = ['a', 'b', 'c', 'd', 'e', 'f']
        a2 = [1, 2, 3, np.nan, np.nan, 6]
        # 调用 delete_masked_points 函数，返回结果 result1, result2
        result1, result2 = delete_masked_points(a1, a2)
        # 定义预期的索引 ind
        ind = [0, 1, 2, 5]
        # 断言 result1 是从 a1 中选取 ind 索引的结果
        assert_array_equal(result1, np.array(a1)[ind])
        # 断言 result2 是从 a2 中选取 ind 索引的结果
        assert_array_equal(result2, np.array(a2)[ind])

    # 测试 delete_masked_points 函数对 datetime 对象的操作
    def test_datetime(self):
        # 定义日期时间列表 dates
        dates = [datetime(2008, 1, 1), datetime(2008, 1, 2),
                 datetime(2008, 1, 3), datetime(2008, 1, 4),
                 datetime(2008, 1, 5), datetime(2008, 1, 6)]
        # 创建一个带掩码的数组 a_masked
        a_masked = np.ma.array([1, 2, 3, np.nan, np.nan, 6],
                               mask=[False, False, True, True, False, False])
        # 调用 delete_masked_points 函数，返回 actual
        actual = delete_masked_points(dates, a_masked)
        # 定义预期的索引 ind
        ind = [0, 1, 5]
        # 断言 actual[0] 是从 dates 中选取 ind 索引的结果
        assert_array_equal(actual[0], np.array(dates)[ind])
        # 断言 actual[1] 是从 a_masked 中选取 ind 索引并压缩的结果
        assert_array_equal(actual[1], a_masked[ind].compressed())

    # 测试 delete_masked_points 函数对 RGBA 颜色的操作
    def test_rgba(self):
        # 创建一个带掩码的数组 a_masked
        a_masked = np.ma.array([1, 2, 3, np.nan, np.nan, 6],
                               mask=[False, False, True, True, False, False])
        # 将颜色字符串数组转换为 RGBA 数组 a_rgba
        a_rgba = mcolors.to_rgba_array(['r', 'g', 'b', 'c', 'm', 'y'])
        # 调用 delete_masked_points 函数，返回 actual
        actual = delete_masked_points(a_masked, a_rgba)
        # 定义预期的索引 ind
        ind = [0, 1, 5]
        # 断言 actual[0] 是从 a_masked 中选取 ind 索引并压缩的结果
        assert_array_equal(actual[0], a_masked[ind].compressed())
        # 断言 actual[1] 是从 a_rgba 中选取 ind 索引的结果
        assert_array_equal(actual[1], a_rgba[ind])


class Test_boxplot_stats:
    # 设置测试方法的初始化操作
    def setup_method(self):
        # 设置随机种子，确保结果可重复
        np.random.seed(937)
        # 设置数据的行数和列数
        self.nrows = 37
        self.ncols = 4
        # 生成服从对数正态分布的随机数据矩阵
        self.data = np.random.lognormal(size=(self.nrows, self.ncols),
                                        mean=1.5, sigma=1.75)
        # 预定义的已知键列表，按字母顺序排列
        self.known_keys = sorted([
            'mean', 'med', 'q1', 'q3', 'iqr',
            'cilo', 'cihi', 'whislo', 'whishi',
            'fliers', 'label'
        ])
        # 对数据进行箱线图统计分析，返回结果
        self.std_results = cbook.boxplot_stats(self.data)

        # 预定义的非自举置信区间结果字典
        self.known_nonbootstrapped_res = {
            'cihi': 6.8161283264444847,
            'cilo': -0.1489815330368689,
            'iqr': 13.492709959447094,
            'mean': 13.00447442387868,
            'med': 3.3335733967038079,
            'fliers': np.array([
                92.55467075,  87.03819018,  42.23204914,  39.29390996
            ]),
            'q1': 1.3597529879465153,
            'q3': 14.85246294739361,
            'whishi': 27.899688243699629,
            'whislo': 0.042143774965502923
        }

        # 预定义的自举置信区间结果字典
        self.known_bootstrapped_ci = {
            'cihi': 8.939577523357828,
            'cilo': 1.8692703958676578,
        }

        # 预定义的调整后箱线图结果字典（设定whis参数为3）
        self.known_whis3_res = {
            'whishi': 42.232049135969874,
            'whislo': 0.042143774965502923,
            'fliers': np.array([92.55467075, 87.03819018]),
        }

        # 预定义的百分位数结果字典（设定whis参数为[0, 100]）
        self.known_res_percentiles = {
            'whislo':   0.1933685896907924,
            'whishi':  42.232049135969874
        }

        # 预定义的范围结果字典（设定whis参数为[0, 100]）
        self.known_res_range = {
            'whislo': 0.042143774965502923,
            'whishi': 92.554670752188699
        }

    # 测试函数：确认标准箱线图统计结果是一个列表
    def test_form_main_list(self):
        assert isinstance(self.std_results, list)

    # 测试函数：确认标准箱线图统计结果中的每个元素是一个字典
    def test_form_each_dict(self):
        for res in self.std_results:
            assert isinstance(res, dict)

    # 测试函数：确认标准箱线图统计结果中的每个字典的键是已知的键集合的子集
    def test_form_dict_keys(self):
        for res in self.std_results:
            assert set(res) <= set(self.known_keys)

    # 测试函数：比较非自举置信区间结果和标准箱线图统计结果中第一个元素的对应键的值是否近似相等
    def test_results_baseline(self):
        res = self.std_results[0]
        for key, value in self.known_nonbootstrapped_res.items():
            assert_array_almost_equal(res[key], value)

    # 测试函数：比较自举置信区间结果和标准箱线图统计结果中第一个元素的对应键的值是否近似相等
    def test_results_bootstrapped(self):
        results = cbook.boxplot_stats(self.data, bootstrap=10000)
        res = results[0]
        for key, value in self.known_bootstrapped_ci.items():
            assert_approx_equal(res[key], value)

    # 测试函数：比较调整后箱线图结果和标准箱线图统计结果中第一个元素的对应键的值是否近似相等
    def test_results_whiskers_float(self):
        results = cbook.boxplot_stats(self.data, whis=3)
        res = results[0]
        for key, value in self.known_whis3_res.items():
            assert_array_almost_equal(res[key], value)

    # 测试函数：比较范围结果和标准箱线图统计结果中第一个元素的对应键的值是否近似相等
    def test_results_whiskers_range(self):
        results = cbook.boxplot_stats(self.data, whis=[0, 100])
        res = results[0]
        for key, value in self.known_res_range.items():
            assert_array_almost_equal(res[key], value)
    # 测试箱线图统计结果的百分位数
    def test_results_whiskers_percentiles(self):
        # 使用 cbook.boxplot_stats 计算数据的箱线图统计信息，指定上下分位数为5%和95%
        results = cbook.boxplot_stats(self.data, whis=[5, 95])
        # 获取第一个结果的统计信息
        res = results[0]
        # 对已知的百分位数进行断言比较
        for key, value in self.known_res_percentiles.items():
            assert_array_almost_equal(res[key], value)

    # 测试带标签的箱线图统计结果
    def test_results_withlabels(self):
        # 指定标签
        labels = ['Test1', 2, 'Aardvark', 4]
        # 使用 cbook.boxplot_stats 计算数据的箱线图统计信息，带有标签信息
        results = cbook.boxplot_stats(self.data, labels=labels)
        # 检查每个标签和相应结果的关联性
        for lab, res in zip(labels, results):
            assert res['label'] == lab

        # 对于没有指定标签的情况，再次计算箱线图统计信息
        results = cbook.boxplot_stats(self.data)
        # 断言结果中不包含标签信息
        for res in results:
            assert 'label' not in res

    # 测试标签参数错误的情况
    def test_label_error(self):
        # 错误的标签格式
        labels = [1, 2]
        # 使用 pytest 检测是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            cbook.boxplot_stats(self.data, labels=labels)

    # 测试维度错误的情况
    def test_bad_dims(self):
        # 生成维度错误的数据
        data = np.random.normal(size=(34, 34, 34))
        # 使用 pytest 检测是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            cbook.boxplot_stats(data)

    # 测试 autorange 参数为 False 的情况
    def test_boxplot_stats_autorange_false(self):
        # 生成数据
        x = np.zeros(shape=140)
        x = np.hstack([-25, x, 25])
        # 使用 cbook.boxplot_stats 计算数据的箱线图统计信息，关闭自动范围调整
        bstats_false = cbook.boxplot_stats(x, autorange=False)
        # 使用 cbook.boxplot_stats 计算数据的箱线图统计信息，开启自动范围调整
        bstats_true = cbook.boxplot_stats(x, autorange=True)

        # 断言不使用自动范围调整时的箱线图统计结果
        assert bstats_false[0]['whislo'] == 0
        assert bstats_false[0]['whishi'] == 0
        assert_array_almost_equal(bstats_false[0]['fliers'], [-25, 25])

        # 断言使用自动范围调整时的箱线图统计结果
        assert bstats_true[0]['whislo'] == -25
        assert bstats_true[0]['whishi'] == 25
        assert_array_almost_equal(bstats_true[0]['fliers'], [])
# 定义一个名为 Test_callback_registry 的测试类
class Test_callback_registry:
    # 初始化方法，在每个测试方法运行前执行
    def setup_method(self):
        # 设置测试信号名称为 'test'
        self.signal = 'test'
        # 创建一个 CallbackRegistry 实例作为回调注册器
        self.callbacks = cbook.CallbackRegistry()

    # 连接回调函数到回调注册器的方法
    def connect(self, s, func, pickle):
        # 如果 pickle 为真，则调用回调注册器的 connect 方法连接回调函数
        if pickle:
            return self.callbacks.connect(s, func)
        else:
            # 否则调用回调注册器的 _connect_picklable 方法连接回调函数
            return self.callbacks._connect_picklable(s, func)

    # 断开指定回调 ID 的回调函数连接的方法
    def disconnect(self, cid):
        return self.callbacks.disconnect(cid)

    # 计算当前信号注册的回调函数数量的方法
    def count(self):
        # 获取当前信号在 _func_cid_map 中注册的回调函数数量
        count1 = len(self.callbacks._func_cid_map.get(self.signal, []))
        # 获取当前信号在 callbacks 中注册的回调函数数量
        count2 = len(self.callbacks.callbacks.get(self.signal))
        # 断言两个数量相等
        assert count1 == count2
        return count1

    # 检查回调注册器是否为空的方法
    def is_empty(self):
        # 断言 _func_cid_map 和 callbacks 都为空字典或集合
        np.testing.break_cycles()
        assert self.callbacks._func_cid_map == {}
        assert self.callbacks.callbacks == {}
        assert self.callbacks._pickled_cids == set()

    # 检查回调注册器是否不为空的方法
    def is_not_empty(self):
        # 断言 _func_cid_map 和 callbacks 都不为空
        np.testing.break_cycles()
        assert self.callbacks._func_cid_map != {}
        assert self.callbacks.callbacks != {}

    # 测试回调 ID 恢复的方法
    def test_cid_restore(self):
        # 创建一个新的 CallbackRegistry 实例
        cb = cbook.CallbackRegistry()
        # 连接一个简单的回调函数 'a'
        cb.connect('a', lambda: None)
        # 对 cb 对象进行序列化和反序列化操作
        cb2 = pickle.loads(pickle.dumps(cb))
        # 使用反序列化后的对象连接一个新的回调函数 'c'
        cid = cb2.connect('c', lambda: None)
        # 断言新回调函数的 ID 为 1
        assert cid == 1

    # 参数化测试方法，测试回调函数的完整性
    @pytest.mark.parametrize('pickle', [True, False])
    def test_callback_complete(self, pickle):
        # 确保开始时回调注册器为空
        self.is_empty()

        # 创建一个 Test_callback_registry 的实例对象
        mini_me = Test_callback_registry()

        # 测试能否添加一个回调函数
        cid1 = self.connect(self.signal, mini_me.dummy, pickle)
        assert type(cid1) is int
        self.is_not_empty()

        # 测试不能再添加第二个相同的回调函数
        cid2 = self.connect(self.signal, mini_me.dummy, pickle)
        assert cid1 == cid2
        self.is_not_empty()
        assert len(self.callbacks._func_cid_map) == 1
        assert len(self.callbacks.callbacks) == 1

        del mini_me

        # 确认现在已经没有回调函数注册
        self.is_empty()

    # 参数化测试方法，测试回调函数的断开连接
    @pytest.mark.parametrize('pickle', [True, False])
    def test_callback_disconnect(self, pickle):
        # 确保开始时回调注册器为空
        self.is_empty()

        # 创建一个 Test_callback_registry 的实例对象
        mini_me = Test_callback_registry()

        # 测试能否添加一个回调函数
        cid1 = self.connect(self.signal, mini_me.dummy, pickle)
        assert type(cid1) is int
        self.is_not_empty()

        # 断开连接指定的回调函数
        self.disconnect(cid1)

        # 确认现在已经没有回调函数注册
        self.is_empty()

    # 参数化测试方法的起始点
    @pytest.mark.parametrize('pickle', [True, False])
    # 测试回调函数在断开连接时的行为，验证开始时注册表为空
    def test_callback_wrong_disconnect(self, pickle):
        # 确保开始时注册表为空
        self.is_empty()

        # 创建用于测试的类实例
        mini_me = Test_callback_registry()

        # 测试添加回调函数的功能
        cid1 = self.connect(self.signal, mini_me.dummy, pickle)
        # 断言返回的连接 ID 类型为整数
        assert type(cid1) is int
        # 验证注册表不为空
        self.is_not_empty()

        # 尝试断开一个不存在的连接
        self.disconnect("foo")

        # 验证断开连接后注册表仍然不为空
        self.is_not_empty()

    # 使用参数化测试测试在非空注册表上注册回调函数的行为
    @pytest.mark.parametrize('pickle', [True, False])
    def test_registration_on_non_empty_registry(self, pickle):
        # 确保开始时注册表为空
        self.is_empty()

        # 设置带有回调函数的注册表
        mini_me = Test_callback_registry()
        self.connect(self.signal, mini_me.dummy, pickle)

        # 添加第二个回调函数
        mini_me2 = Test_callback_registry()
        self.connect(self.signal, mini_me2.dummy, pickle)

        # 删除并重新添加第二个回调函数
        mini_me2 = Test_callback_registry()
        self.connect(self.signal, mini_me2.dummy, pickle)

        # 验证仍然有两个引用
        self.is_not_empty()
        assert self.count() == 2

        # 移除最后两个引用
        mini_me = None
        mini_me2 = None
        self.is_empty()

    # 定义一个空的方法 dummy
    def dummy(self):
        pass

    # 测试回调函数对象的序列化和反序列化
    def test_pickling(self):
        assert hasattr(pickle.loads(pickle.dumps(cbook.CallbackRegistry())),
                       "callbacks")
# 定义一个测试函数，用于测试回调注册表的默认异常处理器
def test_callbackregistry_default_exception_handler(capsys, monkeypatch):
    # 创建一个回调注册表对象
    cb = cbook.CallbackRegistry()
    # 向回调注册表对象连接一个名为 "foo" 的回调函数，这里使用了一个空的 lambda 函数
    cb.connect("foo", lambda: None)

    # 使用 monkeypatch 修改 cbook 模块中的 _get_running_interactive_framework 函数，使其返回 None
    monkeypatch.setattr(
        cbook, "_get_running_interactive_framework", lambda: None)
    # 断言在处理 "foo" 信号时会抛出 TypeError 异常
    with pytest.raises(TypeError):
        cb.process("foo", "argument mismatch")
    # 读取捕获的标准输出和标准错误流
    outerr = capsys.readouterr()
    # 断言标准输出和标准错误流均为空
    assert outerr.out == outerr.err == ""

    # 使用 monkeypatch 修改 _get_running_interactive_framework 函数，使其返回 "not-none"
    monkeypatch.setattr(
        cbook, "_get_running_interactive_framework", lambda: "not-none")
    # 在 "foo" 信号的处理中，不应该抛出异常
    cb.process("foo", "argument mismatch")
    # 再次读取捕获的标准输出和标准错误流
    outerr = capsys.readouterr()
    # 断言标准输出为空
    assert outerr.out == ""
    # 断言标准错误流中包含特定字符串 "takes 0 positional arguments but 1 was given"
    assert "takes 0 positional arguments but 1 was given" in outerr.err


# 定义一个装饰器函数 raising_cb_reg，用于生成测试函数，并设置自定义的异常处理器
def raising_cb_reg(func):
    # 定义一个自定义异常类 TestException，继承自内置异常类 Exception
    class TestException(Exception):
        pass

    # 定义一个函数 raise_runtime_error，用于抛出 RuntimeError 异常
    def raise_runtime_error():
        raise RuntimeError

    # 定义一个函数 raise_value_error，用于抛出 ValueError 异常
    def raise_value_error():
        raise ValueError

    # 定义一个转换器函数 transformer，用于处理异常
    def transformer(excp):
        if isinstance(excp, RuntimeError):
            raise TestException
        raise excp

    # 创建一个回调注册表对象 cb_old，使用默认的异常处理器
    cb_old = cbook.CallbackRegistry(exception_handler=None)
    # 向 cb_old 对象连接一个名为 'foo' 的回调函数 raise_runtime_error
    cb_old.connect('foo', raise_runtime_error)

    # 创建一个回调注册表对象 cb_filt，使用自定义的异常处理器 transformer
    cb_filt = cbook.CallbackRegistry(exception_handler=transformer)
    # 向 cb_filt 对象连接一个名为 'foo' 的回调函数 raise_runtime_error
    cb_filt.connect('foo', raise_runtime_error)

    # 创建一个回调注册表对象 cb_filt_pass，使用自定义的异常处理器 transformer
    cb_filt_pass = cbook.CallbackRegistry(exception_handler=transformer)
    # 向 cb_filt_pass 对象连接一个名为 'foo' 的回调函数 raise_value_error
    cb_filt_pass.connect('foo', raise_value_error)

    # 使用 pytest.mark.parametrize 装饰 func 函数，参数化测试 cb, excp
    return pytest.mark.parametrize('cb, excp',
                                   [[cb_old, RuntimeError],
                                    [cb_filt, TestException],
                                    [cb_filt_pass, ValueError]])(func)


# 使用 raising_cb_reg 装饰的测试函数 test_callbackregistry_custom_exception_handler
@raising_cb_reg
def test_callbackregistry_custom_exception_handler(monkeypatch, cb, excp):
    # 使用 monkeypatch 修改 _get_running_interactive_framework 函数，使其返回 None
    monkeypatch.setattr(
        cbook, "_get_running_interactive_framework", lambda: None)
    # 断言在处理 'foo' 信号时会抛出 excp 指定的异常类型
    with pytest.raises(excp):
        cb.process('foo')


# 定义一个测试函数，测试回调注册表的信号处理功能
def test_callbackregistry_signals():
    # 创建一个带有预设信号 "foo" 的回调注册表对象
    cr = cbook.CallbackRegistry(signals=["foo"])
    # 定义一个结果列表 results
    results = []
    # 定义一个回调函数 cb，将参数 x 添加到 results 列表中
    def cb(x): results.append(x)
    # 向回调注册表对象 cr 连接一个名为 "foo" 的回调函数 cb
    cr.connect("foo", cb)

    # 断言向 "bar" 信号连接相同的回调函数会抛出 ValueError 异常
    with pytest.raises(ValueError):
        cr.connect("bar", cb)

    # 处理 "foo" 信号，传入参数 1
    cr.process("foo", 1)
    # 断言 results 列表中的内容为 [1]
    assert results == [1]

    # 断言向 "bar" 信号处理时会抛出 ValueError 异常
    with pytest.raises(ValueError):
        cr.process("bar", 1)


# 定义一个测试函数，测试回调注册表的阻塞功能
def test_callbackregistry_blocking():
    # 定义一个异常处理器函数 raise_handler，用于在交互式测试环境中处理异常
    def raise_handler(excp):
        raise excp

    # 创建一个回调注册表对象 cb，使用自定义的异常处理器 raise_handler
    cb = cbook.CallbackRegistry(exception_handler=raise_handler)

    # 定义一个测试函数 test_func1，抛出 ValueError 异常
    def test_func1():
        raise ValueError("1 should be blocked")

    # 定义一个测试函数 test_func2，抛出 ValueError 异常
    def test_func2():
        raise ValueError("2 should be blocked")

    # 向回调注册表对象 cb 连接名为 "test1" 的回调函数 test_func1
    cb.connect("test1", test_func1)
    # 向回调注册表对象 cb 连接名为 "test2" 的回调函数 test_func2

    # 使用 cb.blocked() 上下文管理器，阻塞所有回调函数的处理
    with cb.blocked():
        # 处理 "test1" 信号
        cb.process("test1")
        # 处理 "test2" 信号
        cb.process("test2")
    # 使用 cb 对象的 blocked 方法，阻塞信号 "test1" 的回调函数
    with cb.blocked(signal="test1"):
        # 调用 cb 对象的 process 方法，处理信号 "test1"
        cb.process("test1")
        # 使用 pytest 的 raises 方法验证是否抛出 ValueError 异常，并检查异常消息是否包含 "2 should be blocked"
        with pytest.raises(ValueError, match="2 should be blocked"):
            # 再次调用 cb 对象的 process 方法，处理信号 "test2"，预期会触发异常
            cb.process("test2")

    # 确保在阻塞后，原始的回调函数仍然存在
    with pytest.raises(ValueError, match="1 should be blocked"):
        # 调用 cb 对象的 process 方法，处理信号 "test1"，预期会触发异常
        cb.process("test1")
    with pytest.raises(ValueError, match="2 should be blocked"):
        # 再次调用 cb 对象的 process 方法，处理信号 "test2"，预期会触发异常
        cb.process("test2")
@pytest.mark.parametrize('line, result', [
    # 测试用例：无注释
    ('a : no_comment', 'a : no_comment'),
    # 测试用例：带双引号的字符串
    ('a : "quoted str"', 'a : "quoted str"'),
    # 测试用例：带注释的字符串，应删除注释部分
    ('a : "quoted str" # comment', 'a : "quoted str"'),
    # 测试用例：带注释的字符串，但注释符号在引号内，应保留
    ('a : "#000000"', 'a : "#000000"'),
    # 测试用例：带注释的字符串，应删除注释部分
    ('a : "#000000" # comment', 'a : "#000000"'),
    # 测试用例：带注释的字符串列表，应删除注释部分
    ('a : ["#000000", "#FFFFFF"]', 'a : ["#000000", "#FFFFFF"]'),
    # 测试用例：带注释的字符串列表，应删除注释部分
    ('a : ["#000000", "#FFFFFF"] # comment', 'a : ["#000000", "#FFFFFF"]'),
    # 测试用例：带注释的值，应删除注释部分
    ('a : val  # a comment "with quotes"', 'a : val'),
    # 测试用例：仅注释行，应返回空字符串
    ('# only comment "with quotes" xx', ''),
])
def test_strip_comment(line, result):
    """Strip everything from the first unquoted #."""
    assert cbook._strip_comment(line) == result


def test_strip_comment_invalid():
    # 测试异常情况：缺少闭合引号
    with pytest.raises(ValueError, match="Missing closing quote"):
        cbook._strip_comment('grid.color: "aa')


def test_sanitize_sequence():
    d = {'a': 1, 'b': 2, 'c': 3}
    k = ['a', 'b', 'c']
    v = [1, 2, 3]
    i = [('a', 1), ('b', 2), ('c', 3)]
    # 测试字典键的处理，应返回排序后的列表
    assert k == sorted(cbook.sanitize_sequence(d.keys()))
    # 测试字典值的处理，应返回排序后的列表
    assert v == sorted(cbook.sanitize_sequence(d.values()))
    # 测试字典项的处理，应返回排序后的列表
    assert i == sorted(cbook.sanitize_sequence(d.items()))
    # 测试列表的处理，应返回原列表
    assert i == cbook.sanitize_sequence(i)
    # 测试单个列表的处理，应返回原列表
    assert k == cbook.sanitize_sequence(k)


fail_mapping: tuple[tuple[dict, dict], ...] = (
    # 测试失败的关键字参数规范化，应抛出类型错误异常
    ({'a': 1, 'b': 2}, {'alias_mapping': {'a': ['b']}}),
    # 测试失败的关键字参数规范化，应抛出类型错误异常
    ({'a': 1, 'b': 2}, {'alias_mapping': {'a': ['a', 'b']}}),
)

pass_mapping: tuple[tuple[Any, dict, dict], ...] = (
    # 测试成功的关键字参数规范化，应返回原参数字典
    (None, {}, {}),
    # 测试成功的关键字参数规范化，应返回原参数字典
    ({'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {}),
    # 测试成功的关键字参数规范化，应返回经过调整后的参数字典
    ({'b': 2}, {'a': 2}, {'alias_mapping': {'a': ['a', 'b']}}),
)


@pytest.mark.parametrize('inp, kwargs_to_norm', fail_mapping)
def test_normalize_kwargs_fail(inp, kwargs_to_norm):
    # 测试异常情况：关键字参数规范化失败，应抛出类型错误异常
    with pytest.raises(TypeError), \
         _api.suppress_matplotlib_deprecation_warning():
        cbook.normalize_kwargs(inp, **kwargs_to_norm)


@pytest.mark.parametrize('inp, expected, kwargs_to_norm',
                         pass_mapping)
def test_normalize_kwargs_pass(inp, expected, kwargs_to_norm):
    # 测试正常情况：关键字参数规范化成功，应返回期望的参数字典
    with _api.suppress_matplotlib_deprecation_warning():
        # 不应产生其他警告
        assert expected == cbook.normalize_kwargs(inp, **kwargs_to_norm)


def test_warn_external_frame_embedded_python():
    with patch.object(cbook, "sys") as mock_sys:
        mock_sys._getframe = Mock(return_value=None)
        # 测试外部警告，应触发用户警告
        with pytest.warns(UserWarning, match=r"\Adummy\Z"):
            _api.warn_external("dummy")


def test_to_prestep():
    x = np.arange(4)
    y1 = np.arange(4)
    y2 = np.arange(4)[::-1]

    xs, y1s, y2s = cbook.pts_to_prestep(x, y1, y2)

    x_target = np.asarray([0, 0, 1, 1, 2, 2, 3], dtype=float)
    y1_target = np.asarray([0, 1, 1, 2, 2, 3, 3], dtype=float)
    y2_target = np.asarray([3, 2, 2, 1, 1, 0, 0], dtype=float)

    # 测试数据预处理函数，应返回预期的处理结果
    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)
    assert_array_equal(y2_target, y2s)

    xs, y1s = cbook.pts_to_prestep(x, y1)
    # 测试数据预处理函数，应返回预期的处理结果
    assert_array_equal(x_target, xs)
    # 使用 NumPy 提供的 assert_array_equal 函数比较 y1_target 和 y1s 两个数组是否相等
    assert_array_equal(y1_target, y1s)
# 测试空步骤转换函数 `pts_to_prestep`，传入空列表，返回空的步骤数组
def test_to_prestep_empty():
    steps = cbook.pts_to_prestep([], [])
    # 断言返回的步骤数组形状为 (2, 0)
    assert steps.shape == (2, 0)


# 测试后步骤转换函数 `pts_to_poststep`
def test_to_poststep():
    # 创建长度为 4 的数组
    x = np.arange(4)
    y1 = np.arange(4)
    y2 = np.arange(4)[::-1]

    # 执行后步骤转换
    xs, y1s, y2s = cbook.pts_to_poststep(x, y1, y2)

    # 预期的转换结果
    x_target = np.asarray([0, 1, 1, 2, 2, 3, 3], dtype=float)
    y1_target = np.asarray([0, 0, 1, 1, 2, 2, 3], dtype=float)
    y2_target = np.asarray([3, 3, 2, 2, 1, 1, 0], dtype=float)

    # 断言转换后的结果与预期结果一致
    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)
    assert_array_equal(y2_target, y2s)

    # 再次测试只转换 x 和 y1 的情况
    xs, y1s = cbook.pts_to_poststep(x, y1)
    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)


# 测试空后步骤转换函数 `pts_to_poststep`
def test_to_poststep_empty():
    steps = cbook.pts_to_poststep([], [])
    # 断言返回的步骤数组形状为 (2, 0)
    assert steps.shape == (2, 0)


# 测试中间步骤转换函数 `pts_to_midstep`
def test_to_midstep():
    # 创建长度为 4 的数组
    x = np.arange(4)
    y1 = np.arange(4)
    y2 = np.arange(4)[::-1]

    # 执行中间步骤转换
    xs, y1s, y2s = cbook.pts_to_midstep(x, y1, y2)

    # 预期的转换结果
    x_target = np.asarray([0, .5, .5, 1.5, 1.5, 2.5, 2.5, 3], dtype=float)
    y1_target = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=float)
    y2_target = np.asarray([3, 3, 2, 2, 1, 1, 0, 0], dtype=float)

    # 断言转换后的结果与预期结果一致
    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)
    assert_array_equal(y2_target, y2s)

    # 再次测试只转换 x 和 y1 的情况
    xs, y1s = cbook.pts_to_midstep(x, y1)
    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)


# 测试空中间步骤转换函数 `pts_to_midstep`
def test_to_midstep_empty():
    steps = cbook.pts_to_midstep([], [])
    # 断言返回的步骤数组形状为 (2, 0)
    assert steps.shape == (2, 0)


# 参数化测试步骤失败的情况 `pts_to_prestep`
@pytest.mark.parametrize(
    "args",
    [(np.arange(12).reshape(3, 4), 'a'),
     (np.arange(12), 'a'),
     (np.arange(12), np.arange(3))])
def test_step_fails(args):
    with pytest.raises(ValueError):
        cbook.pts_to_prestep(*args)


# 测试 Grouper 类的功能
def test_grouper():
    # 定义 Dummy 类，并创建 5 个实例
    class Dummy:
        pass
    a, b, c, d, e = objs = [Dummy() for _ in range(5)]

    # 创建 Grouper 实例
    g = cbook.Grouper()
    g.join(*objs)

    # 断言第一个组中包含所有对象
    assert set(list(g)[0]) == set(objs)

    # 断言获取 siblings 的结果与所有对象集合相同
    assert set(g.get_siblings(a)) == set(objs)

    # 对每对对象进行 joined 断言
    for other in objs[1:]:
        assert g.joined(a, other)

    # 移除对象 a，并断言其与其他对象不再 joined
    g.remove(a)
    for other in objs[1:]:
        assert not g.joined(a, other)

    # 对所有对象的每对组合，断言它们都是 joined 的
    for A, B in itertools.product(objs[1:], objs[1:]):
        assert g.joined(A, B)


# 测试 Grouper 类的私有方法
def test_grouper_private():
    # 定义 Dummy 类，并创建 5 个实例
    class Dummy:
        pass
    objs = [Dummy() for _ in range(5)]

    # 创建 Grouper 实例并加入所有对象
    g = cbook.Grouper()
    g.join(*objs)

    # 获取私有属性 _mapping
    mapping = g._mapping

    # 断言所有对象都在 _mapping 中
    for o in objs:
        assert o in mapping

    # 断言所有对象共享同一基本集合
    base_set = mapping[objs[0]]
    for o in objs[1:]:
        assert mapping[o] is base_set


# 测试 flatiter 功能
def test_flatiter():
    # 创建长度为 5 的数组
    x = np.arange(5)
    # 获取 flat 迭代器
    it = x.flat
    assert 0 == next(it)
    assert 1 == next(it)
    # 测试 _safe_first_finite 函数返回第一个有限元素的情况
    ret = cbook._safe_first_finite(it)
    assert ret == 0

    assert 0 == next(it)
    assert 1 == next(it)


# 测试 _safe_first_finite 函数处理全为 NaN 的情况
def test__safe_first_finite_all_nan():
    arr = np.full(2, np.nan)
    ret = cbook._safe_first_finite(arr)
    assert np.isnan(ret)


# 测试 _safe_first_finite 函数处理全为 Inf 的情况
def test__safe_first_finite_all_inf():
    arr = np.full(2, np.inf)
    # 调用 cbook 模块中的 _safe_first_finite 函数，传入 arr 作为参数，返回第一个有限的元素
    ret = cbook._safe_first_finite(arr)
    
    # 使用 NumPy 的 assert 函数检查 ret 是否为无穷大
    assert np.isinf(ret)
def test_reshape2d():

    class Dummy:
        pass

    # 调用 cbook 模块的 _reshape_2D 函数，传入空列表和字符串 'x'，返回结果赋给 xnew
    xnew = cbook._reshape_2D([], 'x')
    # 断言 xnew 的形状应为 (1, 0)
    assert np.shape(xnew) == (1, 0)

    # 创建包含 5 个 Dummy 实例的列表 x
    x = [Dummy() for _ in range(5)]
    # 调用 _reshape_2D 函数，传入列表 x 和字符串 'x'，返回结果赋给 xnew
    xnew = cbook._reshape_2D(x, 'x')
    # 断言 xnew 的形状应为 (1, 5)
    assert np.shape(xnew) == (1, 5)

    # 创建一个包含 0 到 4 的整数的 NumPy 数组 x
    x = np.arange(5)
    # 调用 _reshape_2D 函数，传入数组 x 和字符串 'x'，返回结果赋给 xnew
    xnew = cbook._reshape_2D(x, 'x')
    # 断言 xnew 的形状应为 (1, 5)
    assert np.shape(xnew) == (1, 5)

    # 创建一个包含 3 个列表，每个列表包含 5 个 Dummy 实例的二维列表 x
    x = [[Dummy() for _ in range(5)] for _ in range(3)]
    # 调用 _reshape_2D 函数，传入二维列表 x 和字符串 'x'，返回结果赋给 xnew
    xnew = cbook._reshape_2D(x, 'x')
    # 断言 xnew 的形状应为 (3, 5)
    assert np.shape(xnew) == (3, 5)

    # 创建一个形状为 (3, 5) 的随机数 NumPy 数组 x
    x = np.random.rand(3, 5)
    # 调用 _reshape_2D 函数，传入数组 x 和字符串 'x'，返回结果赋给 xnew
    xnew = cbook._reshape_2D(x, 'x')
    # 断言 xnew 的形状应为 (5, 3)，表现出特定行为
    assert np.shape(xnew) == (5, 3)

    # 测试包含多个内部长度为 1 的列表的情况
    x = [[1], [2], [3]]
    xnew = cbook._reshape_2D(x, 'x')
    # 断言 xnew 是列表类型，并且每个元素是形状为 (1,) 的 NumPy 数组
    assert isinstance(xnew, list)
    assert isinstance(xnew[0], np.ndarray) and xnew[0].shape == (1,)
    assert isinstance(xnew[1], np.ndarray) and xnew[1].shape == (1,)
    assert isinstance(xnew[2], np.ndarray) and xnew[2].shape == (1,)

    # 测试包含零维数组的列表的情况
    x = [np.array(0), np.array(1), np.array(2)]
    xnew = cbook._reshape_2D(x, 'x')
    # 断言 xnew 是列表类型，并且包含一个形状为 (3,) 的 NumPy 数组
    assert isinstance(xnew, list)
    assert len(xnew) == 1
    assert isinstance(xnew[0], np.ndarray) and xnew[0].shape == (3,)

    # 测试包含不同长度子列表的列表的情况，会导致内部数组转换为 1D 对象数组的列表
    x = [[1, 2, 3], [3, 4], [2]]
    xnew = cbook._reshape_2D(x, 'x')
    # 断言 xnew 是列表类型，并且每个元素是具有特定形状的 NumPy 数组
    assert isinstance(xnew, list)
    assert isinstance(xnew[0], np.ndarray) and xnew[0].shape == (3,)
    assert isinstance(xnew[1], np.ndarray) and xnew[1].shape == (2,)
    assert isinstance(xnew[2], np.ndarray) and xnew[2].shape == (1,)

    # 测试 Numpy 的子类，确保即使是标量，也能正确处理，避免由于 _reshape_2D 的错误而分割数组
    class ArraySubclass(np.ndarray):

        def __iter__(self):
            for value in super().__iter__():
                yield np.array(value)

        def __getitem__(self, item):
            return np.array(super().__getitem__(item))

    v = np.arange(10, dtype=float)
    x = ArraySubclass((10,), dtype=float, buffer=v.data)
    xnew = cbook._reshape_2D(x, 'x')

    # 断言 xnew 的长度为 1，并且包含一个 ArraySubclass 实例
    assert len(xnew) == 1
    assert isinstance(xnew[0], ArraySubclass)

    # 测试包含字符串列表的情况
    x = ['a', 'b', 'c', 'c', 'dd', 'e', 'f', 'ff', 'f']
    xnew = cbook._reshape_2D(x, 'x')
    # 断言 xnew 的第一个元素长度与 x 的长度相同，并且是一个 NumPy 数组
    assert len(xnew[0]) == len(x)
    assert isinstance(xnew[0], np.ndarray)
    # 创建一个包含30个元素的一维 NumPy 数组，然后将其重新形状为 10 行 3 列的二维数组
    X = np.arange(30).reshape(10, 3)
    
    # 使用 Pandas 创建一个 DataFrame，列名为 "a", "b", "c"，数据来源于数组 X
    x = pd.DataFrame(X, columns=["a", "b", "c"])
    
    # 使用 matplotlib 的 cbook 模块中的 _reshape_2D 函数，将 DataFrame x 重新组织为新的结构 Xnew
    Xnew = cbook._reshape_2D(x, 'x')
    
    # 逐行检查重新组织后的数组 Xnew，因为 _reshape_2D 返回一个数组列表
    for x, xnew in zip(X.T, Xnew):
        # 使用 NumPy 的 assert_array_equal 函数检查每一行的数组 x 是否等于对应的 xnew
        np.testing.assert_array_equal(x, xnew)
# 分离以便在没有 xarray 的情况下运行其余测试...
def test_reshape2d_xarray(xr):
    # 创建一个 10x3 的 NumPy 数组 X，并进行形状重塑，以便创建 xarray.DataArray 对象
    X = np.arange(30).reshape(10, 3)
    x = xr.DataArray(X, dims=["x", "y"])
    # 调用 cbook._reshape_2D 函数对 x 进行 'x' 方向的重塑，并返回结果 Xnew
    Xnew = cbook._reshape_2D(x, 'x')
    # 需要逐行检查，因为 _reshape_2D 返回一个数组列表
    for x, xnew in zip(X.T, Xnew):
        np.testing.assert_array_equal(x, xnew)


def test_index_of_pandas(pd):
    # 分离以便在没有 pandas 的情况下运行其余测试...
    # 创建一个 10x3 的 NumPy 数组 X，并使用其创建 pandas.DataFrame 对象 x
    X = np.arange(30).reshape(10, 3)
    x = pd.DataFrame(X, columns=["a", "b", "c"])
    # 调用 cbook.index_of 函数对 x 进行处理，返回索引 Idx 和处理后的数组 Xnew
    Idx, Xnew = cbook.index_of(x)
    np.testing.assert_array_equal(X, Xnew)
    # 创建一个参考索引数组 IdxRef
    IdxRef = np.arange(10)
    np.testing.assert_array_equal(Idx, IdxRef)


def test_index_of_xarray(xr):
    # 分离以便在没有 xarray 的情况下运行其余测试...
    # 创建一个 10x3 的 NumPy 数组 X，并使用 xarray.DataArray 创建 x 对象
    X = np.arange(30).reshape(10, 3)
    x = xr.DataArray(X, dims=["x", "y"])
    # 调用 cbook.index_of 函数对 x 进行处理，返回索引 Idx 和处理后的数组 Xnew
    Idx, Xnew = cbook.index_of(x)
    np.testing.assert_array_equal(X, Xnew)
    # 创建一个参考索引数组 IdxRef
    IdxRef = np.arange(10)
    np.testing.assert_array_equal(Idx, IdxRef)


def test_contiguous_regions():
    a, b, c = 3, 4, 5
    # 创建一个布尔掩码 mask，以不同方式进行测试
    mask = [True]*a + [False]*b + [True]*c
    expected = [(0, a), (a+b, a+b+c)]
    # 调用 cbook.contiguous_regions 函数，检查其返回结果与预期是否相同
    assert cbook.contiguous_regions(mask) == expected
    d, e = 6, 7
    # 在 mask 后添加 False，继续测试不同情况下的返回结果
    mask = mask + [False]*e
    assert cbook.contiguous_regions(mask) == expected
    # 在 mask 前添加 False，继续测试不同情况下的返回结果
    mask = [False]*d + mask[:-e]
    expected = [(d, d+a), (d+a+b, d+a+b+c)]
    assert cbook.contiguous_regions(mask) == expected
    # 测试 mask 中不存在 True 的情况
    assert cbook.contiguous_regions([False]*5) == []
    # 测试空的 mask 的情况
    assert cbook.contiguous_regions([]) == []


def test_safe_first_element_pandas_series(pd):
    # 故意创建一个 pandas.Series，其索引不是从 0 开始
    s = pd.Series(range(5), index=range(10, 15))
    # 调用 cbook._safe_first_finite 函数，返回第一个有限元素的值
    actual = cbook._safe_first_finite(s)
    assert actual == 0


def test_warn_external(recwarn):
    # 发出一个外部警告消息 "oops"
    _api.warn_external("oops")
    # 检查 recwarn 中的警告数量是否为 1
    assert len(recwarn) == 1
    # 检查 recwarn 中的第一个警告的文件名是否为当前文件名
    assert recwarn[0].filename == __file__


def test_array_patch_perimeters():
    # 这里比较旧的实现作为向量化实现的参考。
    pass
    # 定义一个函数 check，用于检查多边形分割结果是否正确
    def check(x, rstride, cstride):
        # 获取数组 x 的行数和列数
        rows, cols = x.shape
        # 根据步长 rstride 生成行索引
        row_inds = [*range(0, rows-1, rstride), rows-1]
        # 根据步长 cstride 生成列索引
        col_inds = [*range(0, cols-1, cstride), cols-1]
        # 初始化多边形列表
        polys = []
        # 遍历行索引和列索引的组合，生成多边形
        for rs, rs_next in zip(row_inds[:-1], row_inds[1:]):
            for cs, cs_next in zip(col_inds[:-1], col_inds[1:]):
                # +1 确保多边形之间共享边界
                # 从数组 x 中提取子数组，并计算其周长
                ps = cbook._array_perimeter(x[rs:rs_next+1, cs:cs_next+1]).T
                # 将生成的多边形添加到列表中
                polys.append(ps)
        # 将多边形列表转换为 NumPy 数组
        polys = np.asarray(polys)
        # 使用断言检查生成的多边形是否与预期的边界多边形相等
        assert np.array_equal(polys,
                              cbook._array_patch_perimeters(
                                  x, rstride=rstride, cstride=cstride))

    # 定义一个函数 divisors，返回一个整数 n 的所有除数列表
    def divisors(n):
        return [i for i in range(1, n + 1) if n % i == 0]

    # 遍历给定的行数和列数组合
    for rows, cols in [(5, 5), (7, 14), (13, 9)]:
        # 创建一个行数为 rows，列数为 cols 的 NumPy 数组 x
        x = np.arange(rows * cols).reshape(rows, cols)
        # 遍历行数和列数的所有可能的步长组合
        for rstride, cstride in itertools.product(divisors(rows - 1),
                                                  divisors(cols - 1)):
            # 调用 check 函数，检查当前步长组合下的多边形分割情况
            check(x, rstride=rstride, cstride=cstride)
def test_setattr_cm():
    # 定义类 A，包含类属性和实例属性，以及各种方法和装饰器
    class A:
        cls_level = object()  # 类属性 cls_level
        override = object()   # 类属性 override

        def __init__(self):
            self.aardvark = 'aardvark'  # 实例属性 aardvark
            self.override = 'override'  # 实例属性 override
            self._p = 'p'               # 实例属性 _p

        def meth(self):
            ...  # 方法 meth，未实现

        @classmethod
        def classy(cls):
            ...  # 类方法 classy，未实现

        @staticmethod
        def static():
            ...  # 静态方法 static，未实现

        @property
        def prop(self):
            return self._p  # 属性 prop 的 getter 方法

        @prop.setter
        def prop(self, val):
            self._p = val  # 属性 prop 的 setter 方法

    # 定义类 B，继承自 A，无额外属性或方法
    class B(A):
        ...

    other = A()  # 创建 A 类的实例 other

    def verify_pre_post_state(obj):
        # 验证对象状态的函数，包括方法和属性的比较
        assert obj.meth is not obj.meth  # 方法 meth 返回不同的实例
        assert obj.aardvark is obj.aardvark  # 普通属性返回相同实例
        assert a.aardvark == 'aardvark'  # 确保属性值为 'aardvark'
        assert obj.prop is obj.prop  # 属性 prop 返回相同实例
        assert obj.cls_level is A.cls_level  # 类属性 cls_level 保持一致
        assert obj.override == 'override'  # 实例属性 override 为 'override'
        assert not hasattr(obj, 'extra')  # 确保对象没有额外属性 'extra'
        assert obj.prop == 'p'  # 属性 prop 的值为 'p'
        assert obj.monkey == other.meth  # 属性 monkey 与 other 的 meth 方法相同
        assert obj.cls_level is A.cls_level  # 类属性 cls_level 保持一致
        assert 'cls_level' not in obj.__dict__  # 类属性不在实例的 __dict__ 中
        assert 'classy' not in obj.__dict__  # 类方法不在实例的 __dict__ 中
        assert 'static' not in obj.__dict__  # 静态方法不在实例的 __dict__ 中

    a = B()  # 创建 B 类的实例 a

    a.monkey = other.meth  # 将 other 的 meth 方法赋给实例 a 的 monkey 属性
    verify_pre_post_state(a)  # 验证设置前后的对象状态

    with cbook._setattr_cm(
            a, prop='squirrel',
            aardvark='moose', meth=lambda: None,
            override='boo', extra='extra',
            monkey=lambda: None, cls_level='bob',
            classy='classy', static='static'):
        # 使用 _setattr_cm 上下文管理器设置对象属性
        # 因为设置了 lambda 函数，所以属性访问是正常的
        assert a.meth is a.meth  # 方法 meth 返回相同实例
        assert a.aardvark is a.aardvark  # 普通属性返回相同实例
        assert a.aardvark == 'moose'  # 确保属性值为 'moose'
        assert a.override == 'boo'  # 实例属性 override 为 'boo'
        assert a.extra == 'extra'  # 验证额外属性 extra 的设置
        assert a.prop == 'squirrel'  # 属性 prop 的值为 'squirrel'
        assert a.monkey != other.meth  # 属性 monkey 与 other 的 meth 方法不同
        assert a.cls_level == 'bob'  # 实例属性 cls_level 的值为 'bob'
        assert a.classy == 'classy'  # 验证类方法 classy 的设置
        assert a.static == 'static'  # 验证静态方法 static 的设置

    verify_pre_post_state(a)  # 再次验证设置前后的对象状态


def test_format_approx():
    f = cbook._format_approx  # 获取函数 _format_approx 的引用
    assert f(0, 1) == '0'  # 确保 f(0, 1) 返回 '0'
    assert f(0, 2) == '0'  # 确保 f(0, 2) 返回 '0'
    assert f(0, 3) == '0'  # 确保 f(0, 3) 返回 '0'
    assert f(-0.0123, 1) == '-0'  # 确保 f(-0.0123, 1) 返回 '-0'
    assert f(1e-7, 5) == '0'  # 确保 f(1e-7, 5) 返回 '0'
    assert f(0.0012345600001, 5) == '0.00123'  # 确保 f(0.0012345600001, 5) 返回 '0.00123'
    assert f(-0.0012345600001, 5) == '-0.00123'  # 确保 f(-0.0012345600001, 5) 返回 '-0.00123'
    assert f(0.0012345600001, 8) == f(0.0012345600001, 10) == '0.00123456'  # 确保精确的小数格式化


def test_safe_first_element_with_none():
    datetime_lst = [date.today() + timedelta(days=i) for i in range(10)]  # 创建日期列表
    datetime_lst[0] = None  # 将第一个元素设置为 None
    actual = cbook._safe_first_finite(datetime_lst)  # 调用函数 _safe_first_finite 获取第一个非 None 元素
    assert actual is not None and actual == datetime_lst[1]  # 确保返回值不为 None，且与列表的第二个元素相等
def test_strip_math():
    # 检查 strip_math 函数对于非数学表达式的不变性
    assert strip_math(r'1 \times 2') == r'1 \times 2'
    # 检查 strip_math 函数对于带美元符号的数学表达式的处理
    assert strip_math(r'$1 \times 2$') == '1 x 2'
    # 检查 strip_math 函数对于带美元符号的单词表达式的处理
    assert strip_math(r'$\rm{hi}$') == 'hi'


@pytest.mark.parametrize('fmt, value, result', [
    ('%.2f m', 0.2, '0.20 m'),
    ('{:.2f} m', 0.2, '0.20 m'),
    ('{} m', 0.2, '0.2 m'),
    ('const', 0.2, 'const'),
    ('%d or {}', 0.2, '0 or {}'),
    ('{{{:,.0f}}}', 2e5, '{200,000}'),
    ('{:.2%}', 2/3, '66.67%'),
    ('$%g', 2.54, '$2.54'),
])
def test_auto_format_str(fmt, value, result):
    """Apply *value* to the format string *fmt*."""
    # 检查 _auto_format_str 函数在给定格式和值时的输出是否符合预期
    assert cbook._auto_format_str(fmt, value) == result
    # 检查 _auto_format_str 函数在给定格式和 np.float64 类型值时的输出是否符合预期
    assert cbook._auto_format_str(fmt, np.float64(value)) == result


def test_unpack_to_numpy_from_torch():
    """
    Test that torch tensors are converted to NumPy arrays.

    We don't want to create a dependency on torch in the test suite, so we mock it.
    """
    class Tensor:
        def __init__(self, data):
            self.data = data

        def __array__(self):
            return self.data

    # 创建虚拟的 torch 模块
    torch = ModuleType('torch')
    torch.Tensor = Tensor
    sys.modules['torch'] = torch

    # 创建测试数据
    data = np.arange(10)
    torch_tensor = torch.Tensor(data)

    # 调用待测试函数，并比较结果
    result = cbook._unpack_to_numpy(torch_tensor)
    # 比较转换结果是否符合预期
    assert_array_equal(result, data)


def test_unpack_to_numpy_from_jax():
    """
    Test that jax arrays are converted to NumPy arrays.

    We don't want to create a dependency on jax in the test suite, so we mock it.
    """
    class Array:
        def __init__(self, data):
            self.data = data

        def __array__(self):
            return self.data

    # 创建虚拟的 jax 模块
    jax = ModuleType('jax')
    jax.Array = Array
    sys.modules['jax'] = jax

    # 创建测试数据
    data = np.arange(10)
    jax_array = jax.Array(data)

    # 调用待测试函数，并比较结果
    result = cbook._unpack_to_numpy(jax_array)
    # 比较转换结果是否符合预期
    assert_array_equal(result, data)


def test_unpack_to_numpy_from_tensorflow():
    """
    Test that tensorflow arrays are converted to NumPy arrays.

    We don't want to create a dependency on tensorflow in the test suite, so we mock it.
    """
    class Tensor:
        def __init__(self, data):
            self.data = data

        def __array__(self):
            return self.data

    # 创建虚拟的 tensorflow 模块
    tensorflow = ModuleType('tensorflow')
    tensorflow.is_tensor = lambda x: isinstance(x, Tensor)
    tensorflow.Tensor = Tensor
    sys.modules['tensorflow'] = tensorflow

    # 创建测试数据
    data = np.arange(10)
    tf_tensor = tensorflow.Tensor(data)

    # 调用待测试函数，并比较结果
    result = cbook._unpack_to_numpy(tf_tensor)
    # 比较转换结果是否符合预期
    assert_array_equal(result, data)
    # 使用 assert_array_equal 函数来比较 result 和 data 两个对象是否相等
    assert_array_equal(result, data)
```