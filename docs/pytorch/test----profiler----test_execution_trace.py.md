# `.\pytorch\test\profiler\test_execution_trace.py`

```
# Owner(s): ["oncall: profiler"]

# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turnning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    # 设置 tqdm 的监控间隔为0，即关闭监控线程
    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

import json
import sys
import tempfile
import unittest
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch import _dynamo as torchdynamo
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
from torch.profiler import (
    ExecutionTraceObserver,
    kineto_available,
    profile,
    record_function,
    supported_activities,
)

from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)

from torch.utils._triton import has_triton

Json = Dict[str, Any]


class TestExecutionTrace(TestCase):
    def payload(self, use_cuda=False):
        u = torch.randn(3, 4, 5, requires_grad=True)
        with record_function("## TEST 1 ##", "1, 2, 3"):
            inf_val = float("inf")
            neg_inf_val = float("-inf")
            nan_val = float("nan")
            # 进入记录函数，记录特定参数的进入状态
            rf_handle = _record_function_with_args_enter(
                "## TEST 2 ##",
                1,
                False,
                2.5,
                [u, u],
                (u, u),
                "hello",
                u,
                inf_val,
                neg_inf_val,
                nan_val,
            )
            x = torch.randn(10, 10, requires_grad=True)
            if use_cuda:
                x = x.cuda()
            y = torch.randn(10, 10, requires_grad=True)
            if use_cuda:
                y = y.cuda()
            z = x + y + x * y + x * y
            z.backward(z)
            gelu = nn.GELU()
            m = torch.randn(2)
            _ = gelu(m)
            if use_cuda:
                z = z.cpu()
            # 退出记录函数，结束对特定参数的记录状态
            _record_function_with_args_exit(rf_handle)

    def get_execution_trace_root(self, output_file_name) -> Json:
        nodes = []
        with open(output_file_name) as f:
            et_graph = json.load(f)
            assert "nodes" in et_graph
            nodes = et_graph["nodes"]
        return nodes
    # 定义一个方法，用于获取执行轨迹中的 rf_id（记录函数 ID）列表，并按升序排序后返回
    def get_execution_trace_rf_ids(self, nodes: List[Json]) -> List[int]:
        """Returns a sorted list of rf_id (record function ids) in execution trace"""

        # 定义一个内部函数，用于从节点的属性中获取 rf_id
        def get_rf_id(node):
            attrs = node["attrs"]  # 获取节点的属性列表
            for a in attrs:
                if a["name"] == "rf_id":  # 查找属性名为 "rf_id" 的属性
                    return a["value"]  # 返回 rf_id 的值
            return None  # 如果找不到 "rf_id" 属性，则返回 None

        # 生成器表达式，遍历节点列表 nodes，筛选出非特定名称的节点，并获取其 rf_id
        rf_ids_ = (
            get_rf_id(n)
            for n in nodes
            if n["name"] != "[pytorch|profiler|execution_trace|process]"  # 排除特定名称的节点
            and n["name"] != "[pytorch|profiler|execution_trace|thread]"   # 排除特定名称的节点
        )
        # 返回按升序排序后的 rf_id 列表，去除 None 值
        return sorted(rf_id for rf_id in rf_ids_ if rf_id is not None)

    # 定义一个方法，用于获取 Kineto 事件中的 rf_id（记录函数 ID）列表，并按升序排序后返回
    def get_kineto_rf_ids(self, events: List[Json]) -> List[int]:
        """Returns a sorted list of Record function IDs for CPU operators and user annotations"""
        
        # 生成器表达式，筛选出类别为 "cpu_op" 或 "user_annotation" 的事件
        ops_and_annotations = (
            e for e in events if e.get("cat", "") in ["cpu_op", "user_annotation"]
        )
        
        # 返回按升序排序后的 Record function id 列表，如果没有对应参数，则返回 -1
        return sorted(
            e.get("args", {}).get("Record function id", -1) for e in ops_and_annotations
        )

    # 根据 Kineto 可用性进行单元测试跳过标记
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_execution_trace_with_kineto(self):
        # 跟踪调用次数的计数器初始化
        trace_called_num = 0

        # 定义一个回调函数，用于处理跟踪事件
        def trace_handler(p):
            nonlocal trace_called_num
            trace_called_num += 1

        # 检查是否支持 CUDA 加速
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()

        # 创建一个临时文件以保存执行追踪和 kineto 数据
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
        fp.close()

        # 创建一个临时文件以保存 kineto 数据
        kt = tempfile.NamedTemporaryFile(
            mode="w+t", suffix=".kineto.json", delete=False
        )
        kt.close()

        # 使用 profile 上下文管理器进行性能分析
        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
            on_trace_ready=trace_handler,
            execution_trace_observer=(
                ExecutionTraceObserver().register_callback(fp.name)
            ),
        ) as p:
            # 执行循环，记录性能数据
            for idx in range(10):
                with record_function(f"## LOOP {idx} ##"):
                    self.payload(use_cuda=use_cuda)
                p.step()

            # 断言执行追踪文件名与实际的输出文件路径相符
            self.assertEqual(fp.name, p.execution_trace_observer.get_output_file_path())

        # 导出 Chrome 追踪文件
        p.export_chrome_trace(kt.name)

        # 断言 trace_handler 被调用的次数为 1
        self.assertEqual(trace_called_num, 1)

        # 获取执行追踪根节点
        nodes = self.get_execution_trace_root(fp.name)
        loop_count = 0
        found_root_node = False

        # 遍历节点，验证包含特定名称的根节点和循环节点的数量
        for n in nodes:
            assert "name" in n
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1

        # 断言找到了根节点
        self.assertTrue(found_root_node)

        # 断言循环节点的数量为 2
        self.assertEqual(loop_count, 2)

        # 比较执行追踪和 kineto 追踪的记录函数 ID（rf_id）和外部 ID
        with open(kt.name) as f:
            kineto = json.load(f)
            events = kineto["traceEvents"]

        # 获取执行追踪中的记录函数 ID 列表
        rf_ids_et = self.get_execution_trace_rf_ids(nodes)

        # 获取 kineto 追踪中的记录函数 ID 列表
        rf_ids_kineto = self.get_kineto_rf_ids(events)

        # 断言两个列表应完全匹配
        self.assertCountEqual(rf_ids_et, rf_ids_kineto)

        # 断言两个列表应完全匹配，并提供详细的错误消息
        self.assertListEqual(
            rf_ids_et,
            rf_ids_kineto,
            msg=f"ET and kineto rf_id should exactly match\n"
            f"  rf_ids_et = {rf_ids_et}\n"
            f"  rf_ids_kineto = {rf_ids_kineto}\n",
        )
    def test_execution_trace_alone(self):
        # 检查是否支持 CUDA 作为性能分析的一部分
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        
        # 创建一个临时文件来保存执行跟踪数据
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
        fp.close()
        
        # 预期的循环事件数初始化为0
        expected_loop_events = 0
        
        # 创建 ExecutionTraceObserver 实例并注册回调函数，用于保存执行跟踪数据
        et = ExecutionTraceObserver().register_callback(fp.name)
        
        # 开始记录执行跟踪
        et.start()
        for idx in range(5):
            # 每次循环增加预期的循环事件数
            expected_loop_events += 1
            
            # 使用记录函数记录当前循环的信息
            with record_function(f"## LOOP {idx} ##"):
                self.payload(use_cuda=use_cuda)
        
        # 停止记录执行跟踪
        et.stop()
        
        # 断言临时文件的路径与获取的执行跟踪文件路径相同
        assert fp.name == et.get_output_file_path()
        
        # 取消执行跟踪回调函数的注册
        et.unregister_callback()
        
        # 获取执行跟踪根节点
        nodes = self.get_execution_trace_root(fp.name)
        
        # 初始化循环计数器
        loop_count = 0
        
        # 预期的张量对象元组大小
        tensor_tuple_size = 6
        
        # 标记是否找到根节点
        found_root_node = False
        
        # 遍历执行跟踪节点
        for n in nodes:
            # 断言每个节点中包含名称属性
            assert "name" in n
            
            # 检查是否找到根节点
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            
            # 如果节点名称以 "## LOOP " 开头，增加循环计数
            if n["name"].startswith("## LOOP "):
                loop_count += 1
            
            # 检查张量对象元组表示大小是否正确
            if n["name"] == "## TEST 2 ##":
                assert len(n["inputs"]["values"][3][0]) == tensor_tuple_size
        
        # 断言已找到根节点
        assert found_root_node
        
        # 断言循环计数等于预期的循环事件数
        assert loop_count == expected_loop_events
    # 定义一个测试函数，用于检查执行跟踪功能是否正常
    def test_execution_trace_with_pt2(self):
        # 使用 torchdynamo 库对 fn 函数进行优化，标记为 "inductor"
        @torchdynamo.optimize("inductor")
        def fn(a, b, c):
            # 计算线性函数操作
            x = torch.nn.functional.linear(a, b)
            # 加上常数向量 c
            x = x + c
            # 返回 x 的余弦值
            return x.cos()

        # 生成三个随机张量 a, b, c，并将它们移动到 CUDA 设备
        a, b, c = (torch.randn(4, 4, requires_grad=True).to("cuda") for _ in range(3))

        # 将输入张量打包成列表
        inputs = [a, b, c]
        
        # 使用 torch._inductor.config.patch 进行配置，设置编译线程数为 1
        with torch._inductor.config.patch(compile_threads=1):
            # 调用优化后的 fn 函数，对输入张量执行计算
            fn(*inputs)

        # 创建一个临时文件以保存执行跟踪数据
        fp = tempfile.NamedTemporaryFile("w+t", suffix="_et.json", delete=False)
        fp.close()

        # 使用 torch.profiler 进行性能分析和执行跟踪
        with profile(
            activities=torch.profiler.supported_activities(),
            record_shapes=True,
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
            # 注册执行跟踪观察器，将结果写入临时文件 fp.name
            execution_trace_observer=(
                ExecutionTraceObserver().register_callback(fp.name)
            ),
        ) as p:
            # 执行 10 次循环
            for idx in range(10):
                # 记录当前循环的函数调用
                with record_function(f"## LOOP {idx} ##"):
                    fn(*inputs)
                # 在性能分析器中记录当前步骤
                p.step()

        # 从临时文件中获取执行跟踪根节点的列表
        nodes = self.get_execution_trace_root(fp.name)
        # 标记是否找到捕获的 Triton 内核节点
        found_captured_triton_kernel_node = False
        # 遍历每个节点
        for n in nodes:
            # 确保节点中有 "name" 属性
            assert "name" in n
            # 如果节点名中包含 "triton_"，则说明找到了 Triton 内核节点
            if "triton_" in n["name"]:
                # 遍历节点的所有属性
                for attr in n["attrs"]:
                    # 如果属性的名称是 "kernel_file" 并且值不为空，则表示找到了捕获的 Triton 内核节点
                    if attr["name"] == "kernel_file" and attr["value"] != "":
                        found_captured_triton_kernel_node = True
                        # 确保节点的输入值列表不为空
                        assert len(n["inputs"]["values"]) > 0
                        # 确保节点的输出值列表为空
                        assert len(n["outputs"]["values"]) == 0
        # 最终断言是否找到了捕获的 Triton 内核节点
        assert found_captured_triton_kernel_node
    # 定义一个测试方法，用于测试执行跟踪的启动和停止功能
    def test_execution_trace_start_stop(self):
        # 检查当前环境是否支持 CUDA 加速
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        
        # 创建一个临时文件来保存执行跟踪数据
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
        fp.close()
        
        # 预期的循环事件数量
        expected_loop_events = 0
        
        # 创建执行跟踪观察者对象，并注册回调函数以便写入数据到临时文件
        et = ExecutionTraceObserver().register_callback(fp.name)
        
        # 循环10次
        for idx in range(10):
            # 根据索引选择执行跟踪的启动或停止
            if idx == 3:
                et.start()
            elif idx == 5:
                et.stop()
            elif idx == 8:
                et.start()
            elif idx == 9:
                et.stop()
            
            # 如果执行跟踪正在运行，则增加预期的循环事件数量
            if et._execution_trace_running:
                expected_loop_events += 1
            
            # 记录当前循环的函数调用，包含循环索引信息
            with record_function(f"## LOOP {idx} ##"):
                self.payload(use_cuda=use_cuda)

        # 断言临时文件的名称与执行跟踪对象中的输出文件路径相匹配
        assert fp.name == et.get_output_file_path()
        
        # 注销执行跟踪回调函数
        et.unregister_callback()
        
        # 获取执行跟踪数据的根节点信息
        nodes = self.get_execution_trace_root(fp.name)
        
        # 初始化循环计数器和根节点发现标志
        loop_count = 0
        found_root_node = False
        
        # 遍历所有节点
        for n in nodes:
            # 断言节点中包含"name"键
            assert "name" in n
            
            # 检查是否找到了根节点
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            
            # 统计以"## LOOP "开头的节点数量，即循环事件数量
            if n["name"].startswith("## LOOP "):
                loop_count += 1
        
        # 断言已找到根节点
        assert found_root_node
        
        # 断言循环事件数量与预期的一致
        assert loop_count == expected_loop_events

    # 定义另一个测试方法，用于测试循环中重复执行跟踪的功能
    def test_execution_trace_repeat_in_loop(self):
        # 检查当前环境是否支持 CUDA 加速
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        
        # 初始化索引列表，表示需要执行跟踪的迭代索引
        iter_list = {3, 4, 6, 8}
        
        # 预期的循环事件数量等于迭代列表的长度
        expected_loop_events = len(iter_list)
        
        # 存储所有输出文件的名称
        output_files = []
        
        # 循环10次
        for idx in range(10):
            # 如果当前索引在迭代列表中
            if idx in iter_list:
                # 创建一个临时文件来保存执行跟踪数据
                fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
                fp.close()
                output_files.append(fp.name)
                
                # 创建执行跟踪观察者对象，并注册回调函数以便写入数据到临时文件
                et = ExecutionTraceObserver().register_callback(fp.name)
                et.start()
            
            # 记录当前循环的函数调用，包含循环索引信息
            with record_function(f"## LOOP {idx} ##"):
                self.payload(use_cuda=use_cuda)
            
            # 如果当前索引在迭代列表中
            if idx in iter_list:
                # 停止执行跟踪并注销回调函数
                et.stop()
                et.unregister_callback()

        # 初始化事件计数器
        event_count = 0
        
        # 遍历所有输出文件
        for et_file in output_files:
            # 获取执行跟踪数据的根节点信息
            nodes = self.get_execution_trace_root(et_file)
            found_root_node = False
            
            # 遍历所有节点
            for n in nodes:
                # 断言节点中包含"name"键
                assert "name" in n
                
                # 检查是否找到了根节点，并验证其ID为1
                if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                    assert n["id"] == 1
                    found_root_node = True
                
                # 统计以"## LOOP "开头的节点数量，即事件数量
                if n["name"].startswith("## LOOP "):
                    event_count += 1
            
            # 断言已找到根节点
            assert found_root_node
        
        # 断言事件数量与预期的一致
        assert event_count == expected_loop_events
    # 定义一个测试方法，用于验证执行跟踪的结果是否未被捕获
    def test_execution_trace_no_capture(self):
        # 创建一个临时文件对象，以"w+t"模式打开（可读可写文本模式），文件名以.et.json结尾，不立即删除
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
        fp.close()
        # 注册一个回调函数，用于执行跟踪观察器，获取其输出文件的路径
        et = ExecutionTraceObserver().register_callback(fp.name)

        # 断言临时文件的名称与执行跟踪观察器的输出文件路径相同
        assert fp.name == et.get_output_file_path()

        # 注销回调函数，停止执行跟踪
        et.unregister_callback()

        # 获取执行跟踪的根节点列表
        nodes = self.get_execution_trace_root(fp.name)

        # 遍历根节点列表
        for n in nodes:
            # 断言节点中包含名为"name"的字段
            assert "name" in n
            # 如果节点的"name"字段包含特定字符串，则标记为找到根节点
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True

        # 断言已找到根节点
        assert found_root_node

    # 在特定条件下跳过测试，条件为Torch Dynamo问题链接
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/124500")
    def test_execution_trace_nested_tensor(self):
        # 创建一个临时文件对象，以"w+t"模式打开（可读可写文本模式），文件名以.et.json结尾，不立即删除
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
        fp.close()

        # 注册一个回调函数，用于执行跟踪观察器，获取其输出文件的路径
        observer = ExecutionTraceObserver().register_callback(fp.name)

        # 定义一个函数，接受嵌套张量并返回其sin和cos的乘积
        def fn(nt):
            return nt.sin().cos()

        # 使用Torch的性能分析器，启动一个性能分析会话，将执行跟踪观察器作为参数传入
        with torch.profiler.profile(execution_trace_observer=observer) as prof:
            # 多次循环
            for i in range(3):
                # 创建随机张量和偏移张量
                values = torch.rand((8 + i, 4 + i))
                offsets = torch.tensor([0, 2, 4, 6, 8 + i])
                # 从不规则张量创建嵌套张量
                nt = torch.nested.nested_tensor_from_jagged(values, offsets)
                # 调用定义的函数处理嵌套张量
                fn(nt)

        # 获取执行跟踪的根节点列表
        nodes = self.get_execution_trace_root(fp.name)

        # 初始化标志变量，用于指示是否找到包含"cos"字符串的节点
        found_cos = False

        # 遍历根节点列表
        for n in nodes:
            # 断言节点中包含名为"name"的字段
            assert "name" in n
            # 如果节点的"name"字段包含"cos"字符串，则标记为找到cos节点
            if "cos" in n["name"]:
                found_cos = True

        # 断言已找到"cos"节点
        assert found_cos
# 如果当前模块是主程序（而不是被导入的模块），则执行以下代码块
if __name__ == "__main__":
    # 调用运行测试函数
    run_tests()
```