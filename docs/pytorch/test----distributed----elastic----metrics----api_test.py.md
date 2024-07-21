# `.\pytorch\test\distributed\elastic\metrics\api_test.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs

import abc  # 导入抽象基类模块
import unittest.mock as mock  # 导入 mock 模块

from torch.distributed.elastic.metrics.api import (  # 导入 Torch Elastic 相关模块
    _get_metric_name,
    MetricData,
    MetricHandler,
    MetricStream,
    prof,
)
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试相关模块


def foo_1():
    pass


class TestMetricsHandler(MetricHandler):
    def __init__(self):
        self.metric_data = {}  # 初始化一个空的 metric_data 字典

    def emit(self, metric_data: MetricData):
        self.metric_data[metric_data.name] = metric_data  # 将 MetricData 对象存入 metric_data 字典


class Parent(abc.ABC):
    @abc.abstractmethod
    def func(self):
        raise NotImplementedError  # 抽象方法 func()，子类需实现

    def base_func(self):
        self.func()  # 调用抽象方法 func()


class Child(Parent):
    # 使用 @prof 装饰 func 方法的实现
    @prof
    def func(self):
        pass


class MetricsApiTest(TestCase):
    def foo_2(self):
        pass

    @prof
    def bar(self):
        pass

    @prof
    def throw(self):
        raise RuntimeError  # 抛出 RuntimeError 异常

    @prof(group="torchelastic")
    def bar2(self):
        pass

    def test_get_metric_name(self):
        # 注意：由于 PyTorch 使用主方法启动测试，模块名称在 fb 和 oss 之间可能不同，
        # 这里保持模块名称一致。
        foo_1.__module__ = "api_test"
        self.assertEqual("api_test.foo_1", _get_metric_name(foo_1))  # 断言获取函数名的度量名称
        self.assertEqual("MetricsApiTest.foo_2", _get_metric_name(self.foo_2))  # 断言获取方法名的度量名称

    def test_profile(self):
        handler = TestMetricsHandler()  # 创建 TestMetricsHandler 对象
        stream = MetricStream("torchelastic", handler)  # 创建 MetricStream 对象
        # 使用 mock.patch 来替换 getStream 方法，避免并行测试时的冲突
        with mock.patch(
            "torch.distributed.elastic.metrics.api.getStream", return_value=stream
        ):
            self.bar()  # 调用 bar 方法

            # 断言检查度量数据中的成功计数
            self.assertEqual(1, handler.metric_data["MetricsApiTest.bar.success"].value)
            # 断言检查度量数据中的失败计数未包含
            self.assertNotIn("MetricsApiTest.bar.failure", handler.metric_data)
            # 断言检查度量数据中的持续时间包含在内
            self.assertIn("MetricsApiTest.bar.duration.ms", handler.metric_data)

            # 断言检查抛出 RuntimeError 时的行为
            with self.assertRaises(RuntimeError):
                self.throw()

            # 断言检查度量数据中抛出异常的计数
            self.assertEqual(
                1, handler.metric_data["MetricsApiTest.throw.failure"].value
            )
            # 断言检查度量数据中未包含 bar_raise 方法的成功计数
            self.assertNotIn("MetricsApiTest.bar_raise.success", handler.metric_data)
            # 断言检查度量数据中的抛出异常的持续时间包含在内
            self.assertIn("MetricsApiTest.throw.duration.ms", handler.metric_data)

            self.bar2()  # 调用 bar2 方法
            # 断言检查度量数据中 bar2 方法的成功计数的组名为 torchelastic
            self.assertEqual(
                "torchelastic",
                handler.metric_data["MetricsApiTest.bar2.success"].group_name,
            )
    # 定义一个测试方法，用于测试继承情况
    def test_inheritance(self):
        # 创建一个测试用的指标处理器对象
        handler = TestMetricsHandler()
        # 创建一个指标流对象，关联到特定的处理器
        stream = MetricStream("torchelastic", handler)
        # 使用 mock.patch 来替换配置，以避免并行测试时的冲突
        with mock.patch(
            "torch.distributed.elastic.metrics.api.getStream", return_value=stream
        ):
            # 创建一个 Child 类的实例
            c = Child()
            # 调用 Child 类的方法 base_func()
            c.base_func()

            # 断言验证 Child 类的函数成功执行的指标值为 1
            self.assertEqual(1, handler.metric_data["Child.func.success"].value)
            # 断言验证 Child 类的函数执行时间指标存在于指标数据中
            self.assertIn("Child.func.duration.ms", handler.metric_data)
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```