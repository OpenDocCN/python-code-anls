# `.\pytorch\test\dynamo\test_structured_trace.py`

```
# Owner(s): ["module: dynamo"]

# 导入必要的库
import copy                # 复制对象的库
import functools           # 提供工具函数的库
import io                  # 用于处理流的库
import json                # 处理 JSON 数据的库
import logging             # 记录日志的库
import os                  # 提供与操作系统交互的功能的库
import shutil              # 提供高级文件操作功能的库
import subprocess          # 运行外部命令和访问系统 shell 的库
import tempfile            # 创建临时文件和目录的库
import unittest.mock       # 单元测试的模拟功能的库

import torch               # PyTorch 深度学习库
import torch._dynamo.test_case  # PyTorch 内部测试用例支持库
import torch._dynamo.testing    # PyTorch 动力学测试支持库
import torch._logging.structured  # PyTorch 结构化日志记录库
import torch.distributed as dist  # PyTorch 分布式支持库

from torch._inductor.test_case import TestCase  # 导入 PyTorch 测试用例基类

from torch._logging._internal import TorchLogsFormatter  # 导入 PyTorch 日志格式化器
from torch.nn.parallel import DistributedDataParallel as DDP  # PyTorch 分布式数据并行模块
from torch.testing._internal.common_utils import find_free_port  # 导入查找空闲端口函数
from torch.testing._internal.inductor_utils import HAS_CUDA  # 导入是否有 CUDA 支持的标志

# 创建装饰器，标记需要 CUDA 支持的测试用例
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")

# 创建装饰器，标记需要分布式支持的测试用例
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


def example_fn(a):
    # 执行矩阵元素对应位置相乘操作
    output = a.mul(torch.ones(1000, 1000))
    # 执行矩阵元素对应位置相加操作
    output = output.add(torch.ones(1000, 1000))
    return output


def dynamo_error_fn(a):
    # 执行矩阵元素对应位置相乘操作
    output = a.mul(torch.ones(1000, 1000))
    # 尝试执行矩阵元素对应位置相加操作，但是形状不匹配，可能会导致错误
    output = output.add(torch.ones(10, 10))
    return output


def inductor_error_fn(a):
    # 执行矩阵元素四舍五入操作
    output = torch.round(a)
    return output


def inductor_schedule_fn(a):
    # 执行矩阵元素对应位置相加操作，使用 CUDA 加速
    output = a.add(torch.ones(1000, 1000, device="cuda"))
    return output


ARGS = (torch.ones(1000, 1000, requires_grad=True),)

# 自定义日志过滤器，用于结构化跟踪测试
class StructuredTraceTestingFilter(logging.Filter):
    def filter(self, record):
        # 如果日志记录中包含 "str" 键，则过滤掉该记录
        if "str" in record.metadata:
            return False
        return True

# 自定义日志格式化器，用于结构化跟踪测试
class StructuredTraceTestingFormatter(logging.Formatter):
    def format(self, record):
        metadata = copy.deepcopy(record.metadata)

        # 替换不稳定的日志信息为固定值
        # TODO: 检查这些值是否匹配特定的模式
        if "has_payload" in metadata:
            metadata["has_payload"] = "HASH"
        if "dynamo_start" in metadata:
            metadata["dynamo_start"]["stack"] = "STACK"
        if "inductor_output_code" in metadata:
            metadata["inductor_output_code"]["filename"] = "FILENAME"
        if "stack" in metadata:
            metadata["stack"] = "STACK"
        if "compilation_metrics" in metadata:
            metadata["compilation_metrics"] = "METRICS"
        if "describe_storage" in metadata:
            metadata["describe_storage"]["describer_id"] = "ID"
        if "describe_tensor" in metadata:
            metadata["describe_tensor"]["describer_id"] = "ID"
            if "view_func" in metadata["describe_tensor"]:
                metadata["describe_tensor"]["view_func"] = "VIEW_FUNC"
        if "describe_source" in metadata:
            metadata["describe_source"]["describer_id"] = "ID"

        # 将处理后的 metadata 转换为 JSON 格式的字符串
        return json.dumps(metadata)

# 获取名为 "torch.__trace" 的日志记录器
trace_log = logging.getLogger("torch.__trace")

# 继承自 PyTorch 测试用例的结构化跟踪测试类
class StructuredTraceTest(TestCase):
    # 在每个测试运行前执行的设置方法，继承父类的设置方法
    def setUp(self):
        # 调用父类的设置方法
        super().setUp()
        # 重置 torch 内部的 Dynamo 状态
        torch._dynamo.reset()
        # 清空 torch 内部的结构化日志的国际化表
        torch._logging.structured.INTERN_TABLE.clear()
        # 创建一个用于存储日志的内存缓冲区
        self.buffer = io.StringIO()
        # 保存当前的日志级别，并设置为 DEBUG 级别
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        # 创建一个流处理器，并设置特定的格式化器和过滤器
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(StructuredTraceTestingFormatter())
        self.handler.addFilter(StructuredTraceTestingFilter())
        # 将流处理器添加到日志记录器中
        trace_log.addHandler(self.handler)

        # 创建一个临时文件对象，用于存储原始日志数据
        self.raw_file = tempfile.NamedTemporaryFile(
            mode="w", delete=True
        )  # 设置 delete=False 可以保留临时文件
        # 创建一个流处理器，并设置特定的格式化器
        self.raw_handler = logging.StreamHandler(self.raw_file)
        self.raw_handler.setFormatter(TorchLogsFormatter(trace=True))
        # 将流处理器添加到日志记录器中
        trace_log.addHandler(self.raw_handler)

    # 在每个测试运行后执行的清理方法
    def tearDown(self):
        # 移除流处理器
        trace_log.removeHandler(self.handler)
        trace_log.removeHandler(self.raw_handler)
        # 关闭临时文件
        self.raw_file.close()
        # 恢复原始的日志级别设置
        trace_log.setLevel(self.old_level)

    # 断言解析方法
    def assertParses(self):
        # 创建一个临时目录
        out = tempfile.mkdtemp()
        try:
            # 调用外部命令 tlparse，解析 self.raw_file 中的日志数据到指定目录
            subprocess.check_call(
                [
                    "tlparse",
                    "-o",
                    out,
                    "--overwrite",
                    "--no-browser",
                    "--strict",
                    self.raw_file.name,
                ]
            )
        finally:
            # 删除临时目录及其内容，即使出现错误也忽略
            shutil.rmtree(out, ignore_errors=True)

    # 要求 CUDA 的测试方法
    @requires_cuda
    def test_schedule(self):
        # 使用 torch._dynamo.optimize 优化器调度 inductor_schedule_fn 函数
        fn_opt = torch._dynamo.optimize("inductor")(inductor_schedule_fn)
        # 执行优化后的函数，传入 CUDA 设备上的全为 1 的张量
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        # 断言内联预期结果与 self.buffer 中的日志数据匹配
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
# 定义一个包含多个字典的列表，每个字典描述了不同的数据结构和属性
[
    {"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0},
    {"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0},
    {"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1000, 1000], "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0},
    {"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0},
    {"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"},
    {"aot_forward_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"},
    {"artifact": {"name": "fx_graph_cache_hash", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"},
    {"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"},
    {"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"},
    {"dynamo_guards": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"},
    {"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"},
    {"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
]
{"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
# 表示一个包含空字典和几个固定键值对的 JSON 对象

{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
# 包含一个键为 'filename' 的字符串值的 JSON 对象

{"dynamo_guards": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
# 包含空字典的 JSON 对象

{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
# 包含空字典的 JSON 对象

{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
# 包含字符串值 'METRICS' 的 JSON 对象
# 创建一个包含字典元素的列表，每个字典描述了不同的数据结构或功能
[
    {"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0},
    # 描述存储器的属性，包括标识符、描述者 ID 和大小
    {"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0},
    # 描述张量的属性，包括 ID、维度、数据类型、设备、大小等信息
    {"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0},
    # 描述源码的属性，包括描述者 ID 和源码内容
    {"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0},
    # 描述动态输出图的属性，包括不同部分的大小
    {"dynamo_output_graph": {"sizes": {"l_x_": [1000, 1000], "add": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"},
    # 描述预先编译的前向图的属性
    {"aot_forward_graph": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"},
    # 描述艺术品的属性，包括名称和编码方式
    {"artifact": {"name": "fx_graph_cache_hash", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"},
    # 描述感应器后处理梯度图的属性
    {"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"},
    # 描述感应器输出代码的属性，包括文件名
    {"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"},
    # 描述动态守卫的属性
    {"dynamo_guards": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"},
    # 描述动态 CPP 守卫字符串的属性
    {"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"},
    # 描述编译指标的属性，包括度量值
    {"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
]
# 测试类的定义，包含多个测试方法来验证不同的功能
class TestOptimization(unittest.TestCase):

    def test_dynamo_optimize(self):
        # 尝试优化"inductor"相关的函数，并调用带有特定参数的优化函数
        fn_opt = torch._dynamo.optimize("inductor")(inductor_optimize_fn)
        fn_opt(*ARGS)

        # 断言测试缓冲区中的输出与期望值匹配
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_guards": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        # 确保解析测试输出的正确性
        self.assertParses()

    def test_dynamo_error(self):
        try:
            # 优化"inductor"相关函数的错误处理测试
            fn_opt = torch._dynamo.optimize("inductor")(dynamo_error_fn)
            fn_opt(*ARGS)
        except Exception:
            pass

        # 断言测试缓冲区中的输出与期望值匹配
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        # 确保解析测试输出的正确性
        self.assertParses()

    def test_inductor_error(self):
        import torch._inductor.lowering

        def throw(x):
            raise AssertionError

        # 在"lowerings"中注入错误，用于测试在lowerings中引发的错误
        dict_entries = {}
        for x in list(torch._inductor.lowering.lowerings.keys()):
            if "round" in x.__name__:
                dict_entries[x] = throw

        with unittest.mock.patch.dict(torch._inductor.lowering.lowerings, dict_entries):
            try:
                # 优化"inductor"相关函数的错误处理测试
                fn_opt = torch._dynamo.optimize("inductor")(inductor_error_fn)
                fn_opt(*ARGS)
            except Exception:
                pass

        # 断言测试缓冲区中的输出与期望值匹配
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        # 确保解析测试输出的正确性
        self.assertParses()
{
    "dynamo_output_graph": {  // 定义一个名为 "dynamo_output_graph" 的字典
        "sizes": {  // 在 "dynamo_output_graph" 字典中定义一个 "sizes" 字典
            "l_a_": [1000, 1000],  // "l_a_" 键对应值为列表 [1000, 1000]
            "output": [1000, 1000]  // "output" 键对应值为列表 [1000, 1000]
        }
    },
    "frame_id": 0,  // 指定帧 ID 为 0
    "frame_compile_id": 0,  // 指定帧编译 ID 为 0
    "attempt": 0,  // 设置尝试次数为 0
    "has_payload": "HASH"  // 指定 "has_payload" 键的值为 "HASH"
},
{
    "aot_joint_graph": {},  // 定义一个空字典 "aot_joint_graph"
    "frame_id": 0,  // 指定帧 ID 为 0
    "frame_compile_id": 0,  // 指定帧编译 ID 为 0
    "attempt": 0,  // 设置尝试次数为 0
    "has_payload": "HASH"  // 指定 "has_payload" 键的值为 "HASH"
},
{
    "aot_forward_graph": {},  // 定义一个空字典 "aot_forward_graph"
    "frame_id": 0,  // 指定帧 ID 为 0
    "frame_compile_id": 0,  // 指定帧编译 ID 为 0
    "attempt": 0,  // 设置尝试次数为 0
    "has_payload": "HASH"  // 指定 "has_payload" 键的值为 "HASH"
},
{
    "aot_backward_graph": {},  // 定义一个空字典 "aot_backward_graph"
    "frame_id": 0,  // 指定帧 ID 为 0
    "frame_compile_id": 0,  // 指定帧编译 ID 为 0
    "attempt": 0,  // 设置尝试次数为 0
    "has_payload": "HASH"  // 指定 "has_payload" 键的值为 "HASH"
},
{
    "artifact": {  // 定义一个名为 "artifact" 的字典
        "name": "fx_graph_cache_hash",  // "name" 键对应值为 "fx_graph_cache_hash"
        "encoding": "json"  // "encoding" 键对应值为 "json"
    },
    "frame_id": 0,  // 指定帧 ID 为 0
    "frame_compile_id": 0,  // 指定帧编译 ID 为 0
    "attempt": 0,  // 设置尝试次数为 0
    "has_payload": "HASH"  // 指定 "has_payload" 键的值为 "HASH"
},
{
    "inductor_post_grad_graph": {},  // 定义一个空字典 "inductor_post_grad_graph"
    "frame_id": 0,  // 指定帧 ID 为 0
    "frame_compile_id": 0,  // 指定帧编译 ID 为 0
    "attempt": 0,  // 设置尝试次数为 0
    "has_payload": "HASH"  // 指定 "has_payload" 键的值为 "HASH"
},
{
    "compilation_metrics": "METRICS",  // 定义一个名为 "compilation_metrics" 的字典，值为 "METRICS"
    "frame_id": 0,  // 指定帧 ID 为 0
    "frame_compile_id": 0,  // 指定帧编译 ID 为 0
    "attempt": 0  // 设置尝试次数为 0
}
{
    "dynamo_output_graph": {
        "sizes": {
            "l_x_": [1024, 1024],  // 定义 l_x_ 尺寸为 [1024, 1024]
            "l__self___layers_0": [1024, 1024],  // 定义 l__self___layers_0 尺寸为 [1024, 1024]
            "l__self___layers_1": [1024, 1024]   // 定义 l__self___layers_1 尺寸为 [1024, 1024]
        }
    },
    "rank": 0,  // 设置 rank 为 0
    "frame_id": 1,  // 设置 frame_id 为 1
    "frame_compile_id": 0,  // 设置 frame_compile_id 为 0
    "attempt": 0,  // 设置 attempt 为 0
    "has_payload": "HASH"  // 设置 has_payload 为 "HASH"
}
{
    "optimize_ddp_split_graph": {},  // 定义空的 optimize_ddp_split_graph 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "optimize_ddp_split_child": {
        "name": "submod_0"  // 设置 optimize_ddp_split_child 的 name 为 "submod_0"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "optimize_ddp_split_child": {
        "name": "submod_1"  // 设置 optimize_ddp_split_child 的 name 为 "submod_1"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "describe_storage": {
        "id": 0,
        "describer_id": "ID",
        "size": 4194304  // 设置 describe_storage 的 size 为 4194304
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0
}
{
    "describe_tensor": {
        "id": 0,
        "ndim": 2,
        "dtype": "torch.float32",
        "device": "device(type='cuda', index=0)",
        "size": [1024, 1024],  // 设置 describe_tensor 的 size 为 [1024, 1024]
        "is_leaf": true,
        "stride": [1024, 1],
        "storage": 0,
        "view_func": "VIEW_FUNC",
        "describer_id": "ID"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0
}
{
    "describe_source": {
        "describer_id": "ID",
        "id": 0,
        "source": "L['x']"  // 设置 describe_source 的 source 为 "L['x']"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0
}
{
    "aot_joint_graph": {},  // 定义空的 aot_joint_graph 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "aot_forward_graph": {},  // 定义空的 aot_forward_graph 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "aot_backward_graph": {},  // 定义空的 aot_backward_graph 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "artifact": {
        "name": "fx_graph_cache_hash",
        "encoding": "json"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "inductor_post_grad_graph": {},  // 定义空的 inductor_post_grad_graph 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "inductor_output_code": {
        "filename": "FILENAME"  // 设置 inductor_output_code 的 filename 为 "FILENAME"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "aot_joint_graph": {},  // 定义空的 aot_joint_graph 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "aot_forward_graph": {},  // 定义空的 aot_forward_graph 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "aot_backward_graph": {},  // 定义空的 aot_backward_graph 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "artifact": {
        "name": "fx_graph_cache_hash",
        "encoding": "json"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "inductor_post_grad_graph": {},  // 定义空的 inductor_post_grad_graph 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "inductor_output_code": {
        "filename": "FILENAME"  // 设置 inductor_output_code 的 filename 为 "FILENAME"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "dynamo_guards": {},  // 定义空的 dynamo_guards 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "dynamo_cpp_guards_str": {},  // 定义空的 dynamo_cpp_guards_str 对象
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}
{
    "compilation_metrics": "METRICS",  // 设置 compilation_metrics 为 "METRICS"
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0
}
# 如果条件为真，执行以下代码块
if dynamic_s:
    # 断言内联方法的预期输出，验证当前缓冲区的值是否符合预期
    self.assertExpectedInline(
        # 获取当前缓冲区的值
        self.buffer.getvalue(),
        # 依次输出每个 JSON 对象，每个对象表示一个描述，包含描述的详细信息
        """\
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_guards": {}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['self']._modules['layers']._modules['0']._parameters['weight']"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 4096}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['self']._modules['layers']._modules['0']._parameters['bias']"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 2, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 2, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "stride": [1024, 1], "storage": 2, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 2, "source": "L['x']"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 3, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
"""
    )
# 定义一个包含多个字典的列表，每个字典描述了不同的数据结构
{
    "describe_tensor": {
        "id": 8,  # 张量的唯一标识符
        "ndim": 2,  # 张量的维度数
        "dtype": "torch.float32",  # 张量的数据类型
        "device": "device(type='cuda', index=0)",  # 张量所在的设备
        "size": [1024, 1024],  # 张量的尺寸
        "is_leaf": true,  # 张量是否是叶子节点
        "requires_grad": true,  # 是否需要梯度计算
        "is_parameter": true,  # 是否是模型的参数
        "stride": [1024, 1],  # 张量的步幅
        "storage": 3,  # 存储的具体信息
        "view_func": "VIEW_FUNC",  # 张量的视图函数
        "describer_id": "ID"  # 描述该张量的唯一标识符
    },
    "rank": 0,  # 该数据在某种排名中的位置
    "frame_id": 1,  # 框架的唯一标识符
    "frame_compile_id": 0,  # 编译框架的唯一标识符
    "attempt": 0  # 执行尝试次数
}

# 定义一个包含多个字典的列表，每个字典描述了不同的数据源
{
    "describe_source": {
        "describer_id": "ID",  # 描述该数据源的唯一标识符
        "id": 8,  # 数据源的唯一标识符
        "source": "L['self']._modules['layers']._modules['1']._parameters['weight']"  # 数据源的具体来源
    },
    "rank": 0,  # 该数据在某种排名中的位置
    "frame_id": 1,  # 框架的唯一标识符
    "frame_compile_id": 0,  # 编译框架的唯一标识符
    "attempt": 0  # 执行尝试次数
}

# 定义一个包含多个字典的列表，每个字典描述了不同的存储空间
{
    "describe_storage": {
        "id": 4,  # 存储空间的唯一标识符
        "describer_id": "ID",  # 描述该存储空间的唯一标识符
        "size": 4096  # 存储空间的尺寸
    },
    "rank": 0,  # 该数据在某种排名中的位置
    "frame_id": 1,  # 框架的唯一标识符
    "frame_compile_id": 0,  # 编译框架的唯一标识符
    "attempt": 0  # 执行尝试次数
}

# 定义一个包含多个字典的列表，每个字典描述了不同的张量
{
    "describe_tensor": {
        "id": 9,  # 张量的唯一标识符
        "ndim": 1,  # 张量的维度数
        "dtype": "torch.float32",  # 张量的数据类型
        "device": "device(type='cuda', index=0)",  # 张量所在的设备
        "size": [1024],  # 张量的尺寸
        "is_leaf": true,  # 张量是否是叶子节点
        "requires_grad": true,  # 是否需要梯度计算
        "is_parameter": true,  # 是否是模型的参数
        "stride": [1],  # 张量的步幅
        "storage": 4,  # 存储的具体信息
        "view_func": "VIEW_FUNC",  # 张量的视图函数
        "describer_id": "ID"  # 描述该张量的唯一标识符
    },
    "rank": 0,  # 该数据在某种排名中的位置
    "frame_id": 1,  # 框架的唯一标识符
    "frame_compile_id": 0,  # 编译框架的唯一标识符
    "attempt": 0  # 执行尝试次数
}

# 定义一个包含多个字典的列表，每个字典描述了不同的数据源
{
    "describe_source": {
        "describer_id": "ID",  # 描述该数据源的唯一标识符
        "id": 9,  # 数据源的唯一标识符
        "source": "L['self']._modules['layers']._modules['1']._parameters['bias']"  # 数据源的具体来源
    },
    "rank": 0,  # 该数据在某种排名中的位置
    "frame_id": 1,  # 框架的唯一标识符
    "frame_compile_id": 0,  # 编译框架的唯一标识符
    "attempt": 0  # 执行尝试次数
}

# 定义一个包含多个字典的列表，每个字典描述了不同的输出图形
{
    "dynamo_output_graph": {
        "sizes": {
            "l_self_modules_layers_modules_0_parameters_weight_": [1024, 1024],  # 第一个权重参数的尺寸
            "l_self_modules_layers_modules_0_parameters_bias_": [1024],  # 第一个偏置参数的尺寸
            "l_x_": [1024, 1024],  # x 数据的尺寸
            "l_self_modules_layers_modules_1_parameters_weight_": [1024, 1024],  # 第二个权重参数的尺寸
            "l_self_modules_layers_modules_1_parameters_bias_": [1024],  # 第二个偏置参数的尺寸
            "input_1": [1024, 1024],  # 输入1的尺寸
            "input_2": [1024, 1024]  # 输入2的尺寸
        }
    },
    "rank": 0,  # 该数据在某种排名中的位置
    "frame_id": 1,  # 框架的唯一标识符
    "frame_compile_id": 0,  # 编译框架的唯一标识符
    "attempt": 0,  # 执行尝试次数
    "has_payload": "HASH"  # 是否有有效载荷
}

# 定义一个包含多个字典的列表，每个字典描述了优化分布数据平行处理的过程
{
    "optimize_ddp_split_graph": {},
    "rank": 0,  # 该数据在某种排名中的位置
    "frame_id": 1,  # 框架的唯一标识符
    "frame_compile_id": 0,  # 编译框架的唯一标识符
    "attempt": 0,  # 执行尝试次数
    "has_payload": "HASH"  # 是否有有效载荷
}

# 定义一个包含多个字典的列表，每个字典描述了优化分布数据平行处理的子过程
{
    "optimize_ddp_split_child": {
        "name": "submod_0"  # 子模块的名称
    },
    "rank": 0,  # 该数据在某种排名中的位置
    "frame_id": 1,  # 框架的唯一标识符
    "frame_compile_id": 0,  # 编译框架的唯一标识符
    "attempt": 0,  # 执行尝试次数
    "has_payload": "HASH"  # 是否有有效载荷
}
# 第一个字典，描述一个张量
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}

# 第二个字典，描述一个源（source）
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['self']._modules['layers']._modules['0']._parameters['weight']"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}

# 第三个字典，描述一个存储（storage）
{"describe_storage": {"id": 2, "describer_id": "ID", "size": 4096}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}

# 第四个字典，描述一个张量
{"describe_tensor": {"id": 2, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 2, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}

# 第五个字典，描述一个源（source）
{"describe_source": {"describer_id": "ID", "id": 2, "source": "L['self']._modules['layers']._modules['0']._parameters['bias']"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}

# 第六个字典，描述一个联合图（aot_joint_graph）
{"aot_joint_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}

# 第七个字典，描述一个前向图（aot_forward_graph）
{"aot_forward_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}

# 第八个字典，描述一个反向图（aot_backward_graph）
{"aot_backward_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}

# 第九个字典，描述一个工件（artifact）
{"artifact": {"name": "fx_graph_cache_hash", "encoding": "json"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}

# 第十个字典，描述一个诱导器后梯度图（inductor_post_grad_graph）
{"inductor_post_grad_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}

# 第十一个字典，描述一个诱导器输出代码（inductor_output_code）
{"inductor_output_code": {"filename": "FILENAME"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}

# 第十二个字典，描述一个存储（storage）
{"describe_storage": {"id": 16, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}

# 第十三个字典，描述一个张量
{"describe_tensor": {"id": 31, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 16, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}

# 第十四个字典，描述一个源（source）
{"describe_source": {"describer_id": "ID", "id": 31, "source": "L['self']._modules['layers']._modules['1']._parameters['weight']"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}

# 第十五个字典，描述一个存储（storage）
{"describe_storage": {"id": 17, "describer_id": "ID", "size": 4096}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{
    "describe_tensor": {
        "id": 32,
        "ndim": 1,
        "dtype": "torch.float32",
        "device": "device(type='cuda', index=0)",
        "size": [1024],
        "is_leaf": true,
        "requires_grad": true,
        "is_parameter": true,
        "stride": [1],
        "storage": 17,
        "view_func": "VIEW_FUNC",
        "describer_id": "ID"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0
}



# 描述一个张量的特征，包括 ID、数据类型、设备信息等
{
    "describe_tensor": {
        "id": 32,
        "ndim": 1,
        "dtype": "torch.float32",
        "device": "device(type='cuda', index=0)",
        "size": [1024],
        "is_leaf": true,
        "requires_grad": true,
        "is_parameter": true,
        "stride": [1],
        "storage": 17,
        "view_func": "VIEW_FUNC",
        "describer_id": "ID"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0
}



# 描述一个源数据，包括描述者 ID 和源路径
{
    "describe_source": {
        "describer_id": "ID",
        "id": 32,
        "source": "L['self']._modules['layers']._modules['1']._parameters['bias']"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0
}



# 包含一个空的 AOT (Ahead-of-Time) 联合图
{
    "aot_joint_graph": {},
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}



# 包含一个空的 AOT (Ahead-of-Time) 前向图
{
    "aot_forward_graph": {},
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}



# 包含一个空的 AOT (Ahead-of-Time) 反向图
{
    "aot_backward_graph": {},
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}



# 描述一个名为 fx_graph_cache_hash 的工件，采用 JSON 编码
{
    "artifact": {
        "name": "fx_graph_cache_hash",
        "encoding": "json"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}



# 包含一个空的感应器后梯度图
{
    "inductor_post_grad_graph": {},
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}



# 包含一个感应器输出代码的工件，文件名为 "FILENAME"
{
    "inductor_output_code": {
        "filename": "FILENAME"
    },
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}



# 包含一个空的 Dynamo 卫士
{
    "dynamo_guards": {},
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}



# 包含一个空的 Dynamo C++ 卫士字符串
{
    "dynamo_cpp_guards_str": {},
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0,
    "has_payload": "HASH"
}



# 描述一个编译指标为 "METRICS"
{
    "compilation_metrics": "METRICS",
    "rank": 0,
    "frame_id": 1,
    "frame_compile_id": 0,
    "attempt": 0
}



# 末尾的一个空字典，忽略了编码规范建议 (B950)
{
}
# 定义一系列包含不同数据的字典
{"dynamo_output_graph": {"sizes": {"l_x_": [1], "add": [1]}}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_hash", "encoding": "json"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_guards": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
)

# 断言解析结果
self.assertParses()

# 定义一个函数，对两个输入进行矩阵乘法操作
def test_graph_sizes_dynamic():
    def fn(a, b):
        return a @ b

    # 对函数进行优化，设置动态为 False
    fn_opt = torch._dynamo.optimize("eager", dynamic=False)(fn)
    fn_opt(torch.randn(10, 20), torch.randn(20, 30))

    # 对函数进行优化，设置动态为 True
    fn_opt2 = torch._dynamo.optimize("eager", dynamic=True)(fn)
    fn_opt2(torch.randn(5, 10), torch.randn(10, 15))

    # 断言内联结果
    self.assertExpectedInline(
        self.buffer.getvalue(),
        """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 800}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [10, 20], "is_leaf": true, "stride": [20, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 2400}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [20, 30], "is_leaf": true, "stride": [30, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['b']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [10, 20], "l_b_": [20, 30], "matmul": [10, 30]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_guards": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
"""
    )
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}


# 定义一个包含编译指标的字典，用于记录编译过程的相关信息
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}


# 定义一个包含 Dynamo 开始信息的字典，记录了堆栈信息
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 200}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}


# 定义一个包含描述存储信息的字典，记录了存储的标识符、描述者 ID 和大小
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [5, 10], "is_leaf": true, "stride": [10, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}


# 定义一个包含描述张量信息的字典，记录了张量的各种属性，如维度、数据类型、设备类型等
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}


# 定义一个包含描述数据源信息的字典，记录了数据源的描述者 ID 和源信息
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 600}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}


# 定义一个包含描述存储信息的字典，记录了另一个存储的标识符、描述者 ID 和大小
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [10, 15], "is_leaf": true, "stride": [15, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}


# 定义一个包含描述张量信息的字典，记录了另一个张量的各种属性
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['b']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}


# 定义一个包含描述数据源信息的字典，记录了另一个数据源的描述者 ID 和源信息
{"dynamo_output_graph": {"sizes": {"l_a_": ["s0", "s1"], "l_b_": ["s1", "s3"], "matmul": ["s0", "s3"]}}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}


# 定义一个包含动态输出图信息的字典，记录了图的大小和相关标识
{"dynamo_guards": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}


# 定义一个包含动态守卫信息的空字典，记录了守卫的相关信息和标识
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}


# 定义一个包含动态 CPP 守卫字符串信息的空字典，记录了守卫的相关信息和标识
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}


# 定义一个包含编译指标的字典，用于记录另一个编译过程的相关信息
""",  # noqa: B950
# 定义一个包含多个字典的元组
(
    # 第一个字典
    {
        "dynamo_guards": {},  # 动态守卫为空字典
        "frame_id": 0,         # 帧 ID 为 0
        "frame_compile_id": 0, # 帧编译 ID 为 0
        "attempt": 0,          # 尝试次数为 0
        "has_payload": "HASH"  # 是否有有效载荷为 "HASH"
    },
    # 第二个字典
    {
        "dynamo_cpp_guards_str": {},  # 动态 CPP 守卫字符串为空字典
        "frame_id": 0,                 # 帧 ID 为 0
        "frame_compile_id": 0,         # 帧编译 ID 为 0
        "attempt": 0,                  # 尝试次数为 0
        "has_payload": "HASH"          # 是否有有效载荷为 "HASH"
    },
    # 第三个字典
    {
        "compilation_metrics": "METRICS",  # 编译度量为 "METRICS"
        "frame_id": 0,                     # 帧 ID 为 0
        "frame_compile_id": 0,             # 帧编译 ID 为 0
        "attempt": 0                       # 尝试次数为 0
    },
    # 第四个字典
    {
        "dynamo_start": {"stack": "STACK"},  # 动态启动，包含一个键为 "stack" 值为 "STACK" 的字典
        "frame_id": 0,                       # 帧 ID 为 0
        "frame_compile_id": 1,               # 帧编译 ID 为 1
        "attempt": 0                         # 尝试次数为 0
    },
    # 第五个字典
    {
        "describe_storage": {"id": 0, "describer_id": "ID", "size": 4},  # 描述存储，包含 id 为 0，描述者 id 为 "ID"，大小为 4
        "frame_id": 0,                                                    # 帧 ID 为 0
        "frame_compile_id": 1,                                            # 帧编译 ID 为 1
        "attempt": 0                                                      # 尝试次数为 0
    },
    # 第六个字典
    {
        "describe_tensor": {                               # 描述张量，包含多个属性
            "id": 0,
            "ndim": 1,
            "dtype": "torch.float32",
            "device": "device(type='cpu')",
            "size": [1],
            "is_leaf": True,
            "stride": [1],
            "storage": 0,
            "view_func": "VIEW_FUNC",
            "describer_id": "ID"
        },
        "frame_id": 0,                 # 帧 ID 为 0
        "frame_compile_id": 1,         # 帧编译 ID 为 1
        "attempt": 0                   # 尝试次数为 0
    },
    # 第七个字典
    {
        "describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"},  # 描述源，包含描述者 id 为 "ID"，id 为 0，源为 "L['x']"
        "frame_id": 0,                                                           # 帧 ID 为 0
        "frame_compile_id": 1,                                                   # 帧编译 ID 为 1
        "attempt": 0                                                             # 尝试次数为 0
    },
    # 第八个字典
    {
        "dynamo_output_graph": {"sizes": {"l_x_": [1], "x": [1]}},  # 动态输出图，包含 "sizes" 字典，其中包含 "l_x_" 和 "x"，每个值为 [1]
        "frame_id": 0,                                               # 帧 ID 为 0
        "frame_compile_id": 1,                                       # 帧编译 ID 为 1
        "attempt": 0,                                                # 尝试次数为 0
        "has_payload": "HASH"                                        # 是否有有效载荷为 "HASH"
    },
    # 第九个字典
    {
        "dynamo_guards": {},  # 动态守卫为空字典
        "frame_id": 0,         # 帧 ID 为 0
        "frame_compile_id": 1, # 帧编译 ID 为 1
        "attempt": 0,          # 尝试次数为 0
        "has_payload": "HASH"  # 是否有有效载荷为 "HASH"
    },
    # 第十个字典
    {
        "dynamo_cpp_guards_str": {},  # 动态 CPP 守卫字符串为空字典
        "frame_id": 0,                 # 帧 ID 为 0
        "frame_compile_id": 1,         # 帧编译 ID 为 1
        "attempt": 0,                  # 尝试次数为 0
        "has_payload": "HASH"          # 是否有有效载荷为 "HASH"
    },
    # 第十一个字典
    {
        "compilation_metrics": "METRICS",  # 编译度量为 "METRICS"
        "frame_id": 0,                     # 帧 ID 为 0
        "frame_compile_id": 1,             # 帧编译 ID 为 1
        "attempt": 0                       # 尝试次数为 0
    },
    """,  # noqa: B950
)
# 第一个字典对象
{
    "inductor_output_code": {"filename": "FILENAME"},  # 键 'inductor_output_code' 对应的值是一个包含 'filename' 键的字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0,  # 键 'attempt' 对应的值为整数 0
    "has_payload": "HASH"  # 键 'has_payload' 对应的值是字符串 'HASH'
}

# 第二个字典对象
{
    "dynamo_guards": {},  # 键 'dynamo_guards' 对应的值是一个空字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0,  # 键 'attempt' 对应的值为整数 0
    "has_payload": "HASH"  # 键 'has_payload' 对应的值是字符串 'HASH'
}

# 第三个字典对象
{
    "dynamo_cpp_guards_str": {},  # 键 'dynamo_cpp_guards_str' 对应的值是一个空字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0,  # 键 'attempt' 对应的值为整数 0
    "has_payload": "HASH"  # 键 'has_payload' 对应的值是字符串 'HASH'
}

# 第四个字典对象
{
    "compilation_metrics": "METRICS",  # 键 'compilation_metrics' 对应的值是字符串 'METRICS'
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0  # 键 'attempt' 对应的值为整数 0
}

# 第五个字典对象
{
    "dynamo_start": {"stack": "STACK"},  # 键 'dynamo_start' 对应的值是一个包含 'stack' 键的字典，其值是字符串 'STACK'
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0  # 键 'attempt' 对应的值为整数 0
}

# 第六个字典对象
{
    "describe_storage": {"id": 0, "describer_id": "ID", "size": 4},  # 键 'describe_storage' 对应的值是一个包含多个键的字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0  # 键 'attempt' 对应的值为整数 0
}

# 第七个字典对象
{
    "describe_tensor": {  # 键 'describe_tensor' 对应的值是一个包含多个键的字典
        "id": 0,
        "ndim": 1,
        "dtype": "torch.float32",
        "device": "device(type='cpu')",
        "size": [1],
        "is_leaf": True,
        "stride": [1],
        "storage": 0,
        "view_func": "VIEW_FUNC",
        "describer_id": "ID"
    },
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0  # 键 'attempt' 对应的值为整数 0
}

# 第八个字典对象
{
    "describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"},  # 键 'describe_source' 对应的值是一个包含多个键的字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0  # 键 'attempt' 对应的值为整数 0
}

# 第九个字典对象
{
    "dynamo_output_graph": {"sizes": {"l_a_": [1], "sin": [1]}},  # 键 'dynamo_output_graph' 对应的值是一个包含 'sizes' 键的字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0,  # 键 'attempt' 对应的值为整数 0
    "has_payload": "HASH"  # 键 'has_payload' 对应的值是字符串 'HASH'
}

# 第十个字典对象
{
    "aot_forward_graph": {},  # 键 'aot_forward_graph' 对应的值是一个空字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0,  # 键 'attempt' 对应的值为整数 0
    "has_payload": "HASH"  # 键 'has_payload' 对应的值是字符串 'HASH'
}

# 第十一个字典对象
{
    "artifact": {"name": "fx_graph_cache_hash", "encoding": "json"},  # 键 'artifact' 对应的值是一个包含多个键的字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0,  # 键 'attempt' 对应的值为整数 0
    "has_payload": "HASH"  # 键 'has_payload' 对应的值是字符串 'HASH'
}

# 第十二个字典对象
{
    "inductor_output_code": {"filename": "FILENAME"},  # 键 'inductor_output_code' 对应的值是一个包含 'filename' 键的字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0,  # 键 'attempt' 对应的值为整数 0
    "has_payload": "HASH"  # 键 'has_payload' 对应的值是字符串 'HASH'
}

# 第十三个字典对象
{
    "dynamo_guards": {},  # 键 'dynamo_guards' 对应的值是一个空字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
    "attempt": 0,  # 键 'attempt' 对应的值为整数 0
    "has_payload": "HASH"  # 键 'has_payload' 对应的值是字符串 'HASH'
}

# 第十四个字典对象
{
    "dynamo_cpp_guards_str": {},  # 键 'dynamo_cpp_guards_str' 对应的值是一个空字典
    "frame_id": 0,  # 键 'frame_id' 对应的值为整数 0
    "frame_compile_id": 0,  # 键 'frame_compile_id' 对应的值为整数 0
```