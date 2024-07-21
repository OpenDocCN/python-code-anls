# `.\pytorch\test\profiler\test_record_function.py`

```
# Owner(s): ["oncall: profiler"]

# 如果 tqdm 没有正确关闭，会导致监视线程仍然存在。
# 这在多线程测试中会出现问题，因为我们检查所有事件及其线程 ID。
# 那些对应于这些残留线程的事件的 TID 都是 (uint64_t)(-1)，这是无效的。
# 解决方法是在加载 tqdm 时关闭监视线程。
# 由于这些是单元测试，关闭监视线程是安全的。

try:
    import tqdm

    # 将 tqdm 的监视间隔设置为 0，关闭监视线程
    tqdm.tqdm.monitor_interval = 0
except ImportError:
    None

from typing import Any, Dict

import torch
import torch.optim
import torch.utils.data
import torch.utils.data.datapipes as dp
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
from torch.autograd.profiler import profile as _profile
from torch.profiler import kineto_available, record_function
from torch.testing._internal.common_utils import run_tests, TestCase

Json = Dict[str, Any]


class TestRecordFunction(TestCase):
    def _record_function_with_param(self):
        # 创建一个需要梯度的随机张量 u
        u = torch.randn(3, 4, 5, requires_grad=True)
        # 使用 _profile 进行性能分析，同时记录堆栈信息和形状信息
        with _profile(
            with_stack=True, use_kineto=kineto_available(), record_shapes=True
        ) as prof:
            # 进入记录函数 "## TEST 1 ##"，带参数 "1, 2, 3"
            with record_function("## TEST 1 ##", "1, 2, 3"):
                # 进入记录函数 "## TEST 2 ##"，带参数 1, False, 2.5, [u, u], "hello", u
                rf_handle = _record_function_with_args_enter(
                    "## TEST 2 ##", 1, False, 2.5, [u, u], "hello", u
                )
                # 退出记录函数 "## TEST 2 ##"
                _record_function_with_args_exit(rf_handle)
            # 进入记录函数 "## TEST 3 ##"
            with record_function("## TEST 3 ##"):
                # 进入记录函数 "## TEST 4 ##"
                rf_handle = _record_function_with_args_enter("## TEST 4 ##")
                # 退出记录函数 "## TEST 4 ##"
                _record_function_with_args_exit(rf_handle)
        # 返回性能分析结果 prof
        return prof

    def test_record_function(self):
        # 调用 _record_function_with_param 方法
        prof_result = self._record_function_with_param()
        # 初始化标志变量，用于检查是否找到了各个测试记录函数
        found_test_1 = False
        found_test_2 = False
        found_test_3 = False
        found_test_4 = False
        # 遍历性能分析结果中的每个事件
        for e in prof_result.function_events:
            # 检查是否找到 "## TEST 1 ##" 记录函数
            if "## TEST 1 ##" == e.name:
                found_test_1 = True
                # 断言输入形状为空列表
                self.assertTrue(e.input_shapes == [[]])
            # 检查是否找到 "## TEST 2 ##" 记录函数
            elif "## TEST 2 ##" == e.name:
                found_test_2 = True
                # 断言输入形状为 [[], [], [], [], [], [3, 4, 5]]
                self.assertTrue(e.input_shapes == [[], [], [], [], [], [3, 4, 5]])
            # 检查是否找到 "## TEST 3 ##" 记录函数
            elif "## TEST 3 ##" == e.name:
                found_test_3 = True
                # 断言输入形状为空列表
                self.assertTrue(e.input_shapes == [])
            # 检查是否找到 "## TEST 4 ##" 记录函数
            elif "## TEST 4 ##" == e.name:
                found_test_4 = True
                # 断言输入形状为空列表
                self.assertTrue(e.input_shapes == [])
        # 断言所有测试记录函数都被找到
        self.assertTrue(found_test_1)
        self.assertTrue(found_test_2)
        self.assertTrue(found_test_3)
        self.assertTrue(found_test_4)
    # 测试数据管道的记录功能，使用记录函数
    def test_datapipe_with_record_function(self):
        # 使用性能分析器进行上下文管理
        with _profile(
            with_stack=True, use_kineto=kineto_available(), record_shapes=True
        ) as prof:
            # 创建三个输入数据管道包装迭代器对象
            input_dp1 = dp.iter.IterableWrapper(range(4))
            input_dp2 = dp.iter.IterableWrapper(range(4, 8))
            input_dp3 = dp.iter.IterableWrapper(range(8, 12))
            # 将三个输入数据管道对象合并成一个输出数据管道对象
            output_dp = input_dp1.mux(input_dp2, input_dp3)
            # 将输出数据管道对象转换为列表
            output = list(output_dp)

        # 初始化迭代器和复用器是否存在的标志
        has_iter = False
        has_mux = False
        # 遍历性能分析器中的函数事件
        for e in prof.function_events:
            # 如果已经找到迭代器和复用器则退出循环
            if has_iter and has_mux:
                break

            # 如果还未找到迭代器且事件名称中包含 "IterableWrapper" 则标记为找到迭代器
            if not has_iter and "IterableWrapper" in e.name:
                has_iter = True
            # 如果还未找到复用器且事件名称中包含 "Multiplexer" 则标记为找到复用器
            if not has_mux and "Multiplexer" in e.name:
                has_mux = True
        # 断言已找到迭代器和复用器
        self.assertTrue(has_iter)
        self.assertTrue(has_mux)

    # 测试数据管道的委托与性能分析器
    def test_datapipe_delegation_with_profiler(self):
        # 定义一个继承自 IterDataPipe 的迭代器类
        class IDPIterator(torch.utils.data.IterDataPipe):
            def __init__(self):
                self.data = list(range(10))
                self._idx = 0

            def __iter__(self):
                return self

            def __next__(self):
                # 当迭代到末尾时重置索引并抛出 StopIteration 异常
                if self._idx >= 10:
                    self._idx = 0
                    raise StopIteration
                self._idx += 1
                return self.data[self._idx - 1]

            # 获取指定索引处的数据
            def get_value(self, idx):
                return self.data[idx]

        # 创建 IDPIterator 类的实例对象 dp1
        dp1 = IDPIterator()
        # 断言获取索引 5 处的值为 5
        self.assertEqual(5, dp1.get_value(5))
        # 创建 dp1 的迭代器对象 it_dp1
        it_dp1 = iter(dp1)
        # 断言获取索引 5 处的值为 5，类型忽略属性定义
        self.assertEqual(5, it_dp1.get_value(5))  # type: ignore[attr-defined]
        # 断言迭代 it_dp1 得到的列表与预期列表相等
        self.assertEqual(list(range(10)), list(it_dp1))

        # 定义一个继承自 IterDataPipe 的委托类
        class IDPDelegator(torch.utils.data.IterDataPipe):
            def __init__(self, datapipe):
                self.datapipe = datapipe

            def __iter__(self):
                return iter(self.datapipe)

        # 创建 IDPDelegator 类的实例对象 dp2，委托给 dp1
        dp2 = IDPDelegator(dp1)
        # 创建 dp2 的迭代器对象 it_dp2
        it_dp2 = iter(dp2)
        # 断言获取索引 5 处的值为 5
        self.assertEqual(5, it_dp2.get_value(5))
        # 断言迭代 it_dp2 得到的列表与预期列表相等
        self.assertEqual(list(range(10)), list(it_dp2))

    # 测试数据管道的记录函数分支功能
    def test_datapipe_with_record_function_fork(self):
        # 使用性能分析器进行上下文管理
        with _profile(
            with_stack=True, use_kineto=kineto_available(), record_shapes=True
        ) as prof:
            # 创建一个输入数据管道包装迭代器对象 input_dp
            input_dp = dp.iter.IterableWrapper(range(10))
            # 将 input_dp 分叉成三个子数据管道对象 dp1, dp2, dp3
            dp1, dp2, dp3 = input_dp.fork(num_instances=3)
            # 对 dp1 进行迭代并转换为列表 output1
            output1 = list(dp1)

        # 初始化迭代器和子数据管道是否存在的标志
        has_iter = False
        has_child = False
        # 遍历性能分析器中的函数事件
        for e in prof.function_events:
            # 如果已经找到迭代器和子数据管道则退出循环
            if has_iter and has_child:
                break

            # 如果还未找到迭代器且事件名称中包含 "IterableWrapper" 则标记为找到迭代器
            if not has_iter and "IterableWrapper" in e.name:
                has_iter = True
            # 如果还未找到子数据管道且事件名称中包含 "_ChildDataPipe" 则标记为找到子数据管道
            if not has_child and "_ChildDataPipe" in e.name:
                has_child = True
        # 断言已找到迭代器和子数据管道
        self.assertTrue(has_iter)
        self.assertTrue(has_child)
# 如果当前脚本被直接执行（而不是被导入为模块），则执行下面的代码
if __name__ == "__main__":
    # 调用运行测试函数，此处假设其定义在其他地方
    run_tests()
```