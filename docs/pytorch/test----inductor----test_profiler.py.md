# `.\pytorch\test\inductor\test_profiler.py`

```py
# Owner(s): ["module: inductor"]

# 导入必要的库
import json
import unittest

# 导入 Torch 相关模块
import torch
import torch._inductor.test_case
import torch._inductor.utils

# 导入 Torch 内部配置和性能分析模块
from torch._inductor import config
from torch.profiler import ProfilerActivity

# 导入 Torch 内部测试和工具
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.inductor_utils import HAS_CUDA

# 导入 Triton 相关模块
from torch.utils._triton import has_triton

# 检查是否有 Triton 支持
HAS_TRITON = has_triton()

# 定义 DynamoProfilerTests 测试类，继承自 torch._inductor.test_case.TestCase
class DynamoProfilerTests(torch._inductor.test_case.TestCase):
    
    # 测试方法，当没有 Triton 支持时跳过测试
    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_triton_launch(self):
        # 验证在性能分析中是否能够捕捉到 Triton 内核启动的 CPU 端指示
        # 目前，这些事件显示为 `cuLaunchKernel`。如果这个细节改变了，测试可能需要更新或移除。

        # 定义一个编译函数 fn，用于计算 x 和 y 的 sin 和 cos
        @torch.compile
        def fn(x, y):
            return (x + y).sin().cos()

        # 在 CUDA 设备上生成随机数据 x 和 y
        x, y = (torch.rand((4, 4), device="cuda") for _ in range(2))

        # 使用性能分析器捕捉代码段 fn 的性能数据
        with torch.profiler.profile() as prof:
            fn(x, y)

        # 使用临时文件名保存 Chrome 追踪文件
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace_json = json.load(f)

        # 断言追踪 JSON 中包含 "traceEvents" 字段
        self.assertTrue("traceEvents" in trace_json)
        events = trace_json["traceEvents"]

        # 根据 Torch 版本选择合适的内核名称
        kernel_name = "hipModuleLaunchKernel" if torch.version.hip else "cuLaunchKernel"

        # 定义函数，用于检查事件名称是否匹配启动内核的标记
        def nameMatchesLaunchKernel(event_name):
            return kernel_name in event_name

        # 断言在事件列表中至少存在一个事件名称匹配内核启动的标记
        self.assertTrue(
            any(("name" in event and kernel_name == event["name"]) for event in events)
        )
    def _test_profiling_kernel_names(self, fn, args, kernel_name_str: str):
        """
        We expect a record_function event to be added on the CPU side, surrounding
        the launch of each triton kernel.
        """
        # 编译优化后的函数
        fn_opt = torch.compile(fn)

        # 执行两次优化后的函数，预热
        for _ in range(2):
            fn_opt(*args)

        # 使用 Torch Profiler 进行性能分析，记录 CPU 活动和形状信息
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            fn_opt(*args)

        # 断言：确保每个事件的名称中包含 "triton" 和 kernel_name_str
        self.assertTrue(
            any(
                (
                    hasattr(event, "name")
                    and kernel_name_str in event.name
                    and "triton" in event.name
                )
                for event in prof.events()
            )
        )
        # 返回分析事件列表
        return prof.events()

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_kernel_names_pointwise(self):
        # 定义测试函数 fn，进行点操作(sin -> cos)
        def fn(x, y):
            return (x + y).sin().cos()

        # 准备参数：在 CUDA 设备上生成随机数据
        args = [torch.rand((4, 4), device="cuda") for _ in range(2)]

        # 测试函数 _test_profiling_kernel_names，验证 sin 是否出现在内核名称中
        events = self._test_profiling_kernel_names(fn, args, "sin")
        event_found = False
        # 遍历事件列表，查找指定的内核名称
        for event in events:
            if event.name == "triton_poi_fused_add_cos_sin_0":
                event_found = True
                # 断言：验证输入形状是否匹配
                self.assertTrue(event.input_shapes == [[4, 4], [4, 4], [4, 4], []])
        # 断言：确保找到了指定的事件
        self.assertTrue(event_found)

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_kernel_names_template(self):
        # 在设置中启用自动调优并选择 TRITON 作为后端
        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        ):
            # 定义测试函数 fn，执行矩阵乘法操作
            def fn(x, y):
                return x @ y

            # 准备参数：在 CUDA 设备上生成随机数据
            args = [torch.rand((4, 4), device="cuda") for _ in range(2)]

            # 测试函数 _test_profiling_kernel_names，验证 mm 是否出现在内核名称中
            events = self._test_profiling_kernel_names(fn, args, "mm")
            event_found = False
            # 遍历事件列表，查找指定的内核名称
            for event in events:
                if event.name == "triton_tem_fused_mm_0":
                    event_found = True
                    # 断言：验证输入形状是否匹配
                    self.assertTrue(event.input_shapes == [[4, 4], [4, 4], [4, 4]])
            # 断言：确保找到了指定的事件
            self.assertTrue(event_found)

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    # 定义一个测试函数，用于测试在自动调整最大值和 TRITON 后端情况下的感应器和内核名称
    def test_inductor_profiling_kernel_names_foreach(self):
        # 设置配置，包括启用最大自动调谐和指定 TRITON 后端
        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        ):
        
            # 定义一个函数 fn，使用 torch._foreach_add 函数对两个输入进行元素级相加
            def fn(x, y):
                return torch._foreach_add(x, y)
            
            # 生成三个在 CUDA 设备上的随机矩阵作为输入 x 和 y
            x = [torch.rand((4, 4), device="cuda") for _ in range(3)]
            y = [torch.rand((4, 4), device="cuda") for _ in range(3)]
            
            # 将 x 和 y 组成一个参数元组
            args = (x, y)
            
            # 调用 self._test_profiling_kernel_names 函数，传入 fn 函数和 args 参数，
            # 并指定 "_for_" 作为标识符，返回事件列表
            events = self._test_profiling_kernel_names(fn, args, "_for_")
            
            # 初始化事件是否被找到的标志为 False
            event_found = False
            
            # 遍历事件列表中的每个事件
            for event in events:
                # 如果事件的名称为 "triton_for_fused_0"
                if event.name == "triton_for_fused_0":
                    # 将事件找到的标志设置为 True
                    event_found = True
                    # 断言事件的输入形状是否符合预期
                    self.assertTrue(
                        event.input_shapes
                        == [
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                        ]
                    )
            
            # 最后断言事件是否被找到
            self.assertTrue(event_found)
    
    # 如果没有 TRITON 模块，则跳过此测试
    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_triton_hooks(self):
        # 导入 TRITON 编译器模块中的 CompiledKernel 类
        from triton.compiler import CompiledKernel
        
        # 创建一个字典，用于记录进入和退出钩子是否被调用
        hooks_called = {"enter": False, "exit": False}
        
        # 定义一个进入钩子函数，更新 hooks_called 字典中的 "enter" 键
        def launch_enter_hook(lazy_dict):
            hooks_called["enter"] = True
        
        # 定义一个退出钩子函数，更新 hooks_called 字典中的 "exit" 键
        def launch_exit_hook(lazy_dict):
            hooks_called["exit"] = True
        
        # 将进入和退出钩子函数绑定到 CompiledKernel 类上
        CompiledKernel.launch_enter_hook = launch_enter_hook
        CompiledKernel.launch_exit_hook = launch_exit_hook
        
        # 定义一个函数 fn，使用 torch._foreach_add 函数对两个输入进行元素级相加
        def fn(x, y):
            return torch._foreach_add(x, y)
        
        # 生成三个在 CUDA 设备上的随机矩阵作为输入 x 和 y
        x = [torch.rand((4, 4), device="cuda") for _ in range(3)]
        y = [torch.rand((4, 4), device="cuda") for _ in range(3)]
        
        # 将 x 和 y 组成一个参数元组
        args = (x, y)
        
        # 编译函数 fn 以进行优化
        fn_opt = torch.compile(fn)
        
        # 调用优化后的函数 fn_opt，并传入参数 args
        fn_opt(*args)
        
        # 断言进入和退出钩子函数是否被调用
        self.assertTrue(hooks_called["enter"])
        self.assertTrue(hooks_called["exit"])
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 从 torch 库中的 _inductor 模块导入 run_tests 函数
    from torch._inductor.test_case import run_tests

    # 如果 CUDA 可用（假设 HAS_CUDA 是一个标识 CUDA 是否可用的变量）
    if HAS_CUDA:
        # 运行测试函数，通常用于测试 CUDA 相关功能
        run_tests()
```