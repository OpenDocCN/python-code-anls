# `.\pytorch\test\inductor\test_cudagraph_trees.py`

```
# Owner(s): ["module: inductor"]

import contextlib  # 导入上下文管理模块
import functools  # 导入函数工具模块
import gc  # 导入垃圾回收模块
import importlib  # 导入模块导入工具
import sys  # 导入系统模块
import unittest  # 导入单元测试框架
import warnings  # 导入警告模块

import torch  # 导入PyTorch库

import torch._dynamo.config as dynamo_config  # 导入动态编译配置模块
import torch.nn as nn  # 导入神经网络模块
from torch._dynamo.utils import counters  # 从动态编译工具中导入计数器模块
from torch._inductor import config  # 导入电感器模块的配置
from torch._inductor.compile_fx import compile_fx_inner  # 导入电感器模块的编译函数
from torch._inductor.cudagraph_trees import cudagraphify_impl as tree_cudagraphify_impl  # 导入电感器模块的CUDA图函数
from torch._inductor.test_case import TestCase as InductorTestCase  # 导入电感器模块的测试用例类
from torch.fx.experimental.proxy_tensor import make_fx  # 导入代理张量的FX模块
from torch.testing import FileCheck  # 导入文件检查模块

from torch.testing._internal.common_cuda import TEST_MULTIGPU  # 导入测试多GPU模块
from torch.testing._internal.common_utils import (  # 导入内部常用工具模块
    instantiate_parametrized_tests,
    IS_CI,
    IS_LINUX,
    IS_WINDOWS,
    parametrize,
    skipIfRocm,
    TEST_CUDA_GRAPH,
    TEST_WITH_ASAN,
)
from torch.utils._python_dispatch import TorchDispatchMode  # 导入Python调度模式

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

importlib.import_module("functorch")  # 导入functorch模块
importlib.import_module("filelock")  # 导入filelock模块

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA  # 导入电感器内部工具模块中的CUDA和CPU判断

aten = torch.ops.aten  # 设置aten为torch的运算符接口
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")  # 条件装饰器：需要CUDA环境
requires_multigpu = functools.partial(  # 设置部分函数：需要多GPU环境
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)
from io import StringIO  # 导入字符串IO模块


def get_compile_fn(backend):
    if backend == "cudagraphs":
        return functools.partial(torch.compile, backend="cudagraphs")  # 返回部分函数：使用CUDA图形模式编译
    else:
        return functools.partial(torch.compile, mode="reduce-overhead")  # 返回部分函数：使用降低开销模式编译


class capture_stderr(list):
    """
    Replace sys.stderr with a temporary StringIO
    """

    def __enter__(self):
        self.sys_stderr = sys.stderr  # 保存当前sys.stderr引用
        self.stringio = StringIO()  # 创建一个临时的StringIO对象
        sys.stderr = self.stringio  # 替换sys.stderr为StringIO对象
        return self

    def __exit__(self, *args):
        self.append(str(self.stringio.getvalue()))  # 将StringIO中的内容转换为字符串并存入列表
        del self.stringio  # 删除StringIO对象
        sys.stderr = self.sys_stderr  # 恢复原始的sys.stderr引用


def cdata(t):
    return t.untyped_storage()._cdata  # 返回张量的未类型化存储的_cdata属性


class TestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()  # 调用父类的类级别setUp方法
        cls._stack = contextlib.ExitStack()  # 创建上下文管理的堆栈对象
        cls._stack.enter_context(  # 进入上下文管理，使用配置补丁
            config.patch(
                {
                    "debug": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # 禁用自动调整
                    "implicit_fallbacks": False,  # 禁用隐式回退
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()  # 关闭上下文管理的堆栈
        super().tearDownClass()  # 调用父类的类级别tearDown方法

    def setUp(self):
        torch._dynamo.reset()  # 重置动态编译环境
        super().setUp()  # 调用父类的setUp方法

    def tearDown(self):
        super().tearDown()  # 调用父类的tearDown方法
        torch._dynamo.reset()  # 重置动态编译环境


if HAS_CUDA and not TEST_WITH_ASAN:
    # 如果有CUDA并且不是在ASAN环境下
    # 获取所有 CUDA 图形段的快照
    def get_all_cudagraph_segments():
        # 使用 PyTorch 提供的方法获取当前 CUDA 内存的快照信息
        segments = torch.cuda.memory_snapshot()
        # 返回一个列表，其中每个元素是一个段（segment），表示 CUDA 图形段
        return [segment for segment in segments if segment["segment_pool_id"] != (0, 0)]
    
    # 获取所有活跃的内存块地址列表
    def all_live_blocks():
        # 初始化一个空列表，用于存储所有活跃块的地址
        blocks_addrs = []
        # 遍历所有 CUDA 图形段
        for segment in get_all_cudagraph_segments():
            # 获取当前段的起始地址
            addr = segment["address"]
            # 遍历当前段中的所有块
            for block in segment["blocks"]:
                # 如果块的状态为 "active_allocated"，将其地址添加到列表中
                if block["state"] == "active_allocated":
                    blocks_addrs.append(addr)
                # 更新下一个块的起始地址
                addr += block["size"]
    
        # 返回所有活跃块的地址列表
        return blocks_addrs
    
    # 获取所有活跃内存块的数量
    def all_live_block_count():
        # 返回活跃内存块地址列表的长度，即活跃内存块的数量
        return len(all_live_blocks())
    
    # 实例化参数化测试，传入 CudaGraphTreeTests 类
    instantiate_parametrized_tests(CudaGraphTreeTests)
if __name__ == "__main__":
    # 检查当前模块是否作为主程序运行
    from torch._inductor.test_case import run_tests

    # 如果不需要进行 CUDA 图测试，则退出程序
    if not TEST_CUDA_GRAPH:
        if __name__ == "__main__":
            # 如果当前模块是主程序，退出并返回状态码 0
            sys.exit(0)
        # 如果不需要进行 CUDA 图测试，抛出 unittest 跳过测试的异常
        raise unittest.SkipTest("cuda graph test is skipped")

    # 如果具有 CPU 或 CUDA 的环境存在，则运行测试
    if HAS_CPU or HAS_CUDA:
        # 运行测试，需要使用 filelock
        run_tests(needs="filelock")
```