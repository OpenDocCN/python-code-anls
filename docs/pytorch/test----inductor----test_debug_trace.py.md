# `.\pytorch\test\inductor\test_debug_trace.py`

```
# Owner(s): ["module: inductor"]
# 导入必要的库和模块
import logging  # 导入日志模块
import os  # 导入操作系统相关功能模块
import re  # 导入正则表达式模块
import shutil  # 导入文件操作模块
import sys  # 导入系统相关模块
import unittest  # 导入单元测试模块
from pathlib import Path  # 导入路径处理模块

import torch  # 导入PyTorch深度学习框架
from torch._inductor import config, test_operators  # 导入PyTorch内部工具
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU  # 导入测试相关工具

# 尝试导入本地模块或全局模块，用于测试
try:
    try:
        from . import test_torchinductor  # 尝试从当前包导入测试模块
    except ImportError:
        import test_torchinductor  # 如果失败，则从全局环境导入测试模块
except unittest.SkipTest:
    # 如果单元测试被标记为跳过，则根据情况退出或抛出异常
    if __name__ == "__main__":
        sys.exit(0)  # 如果是在主程序中运行，直接退出
    raise  # 否则抛出未捕获的unittest.SkipTest异常

# 定义一个函数，用于获取指定文件的大小
def filesize(filename: Path):
    assert filename.exists(), f"{filename} is missing"  # 断言文件存在，否则报错
    return os.stat(filename).st_size  # 返回文件大小

# 使用装饰器配置调试参数为启用状态，并定义测试类
@config.patch("trace.enabled", True)
class TestDebugTrace(test_torchinductor.TestCase):
    # 定义测试函数，用于测试调试追踪功能
    def test_debug_trace(self):
        # 定义一个编译函数
        @torch.compile
        def fn(a, b):
            a = test_operators.realize(a + 1) + 2  # 对a进行实现操作，并加上2
            return torch.matmul(a, b)  # 返回a和b的矩阵乘积

        # 使用assertLogs检查日志输出，并设置日志级别为WARNING
        with self.assertLogs(
            logging.getLogger("torch._inductor.debug"), level=logging.WARNING
        ) as cm:
            fn(torch.randn(16, 16), torch.randn(16, 16))  # 执行编译函数

        self.assertEqual(len(cm.output), 1)  # 断言日志输出只有一条
        m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])  # 使用正则表达式匹配日志内容
        self.assertTrue(m)  # 断言匹配成功
        filename = Path(m.group(1))  # 获取匹配到的文件名路径
        self.assertTrue(filename.is_dir())  # 断言文件名对应的路径是一个目录
        self.assertGreater(filesize(filename / "fx_graph_readable.py"), 512)  # 断言各个文件的大小满足最小要求
        self.assertGreater(filesize(filename / "fx_graph_runnable.py"), 512)
        self.assertGreater(filesize(filename / "fx_graph_transformed.py"), 512)
        self.assertGreater(filesize(filename / "output_code.py"), 1024)
        self.assertExpectedInline(
            open(filename / "ir_pre_fusion.txt").read().rstrip(),
            """\
buf0: SchedulerNode(ComputedBuffer)
buf0.writes = [MemoryDep('buf0', c0, {c0: 256}, None)]
buf0.unmet_dependencies = []
buf0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256}, None)]
buf0.users = [NodeUser(node=SchedulerNode(name='buf1'), can_inplace=True, is_weak=False)]
buf0.group.device = cpu
buf0.group.iteration = ((256,), ())
buf0.sizes = ([256], [])
arg0_1_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
buf0_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
class buf0_loop_body:
    var_ranges = {z0: 256}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(1.0, torch.float32)
        add = ops.add(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, add, None)
        return store


buf1: SchedulerNode(ComputedBuffer)
buf1.writes = [MemoryDep('buf1', c0, {c0: 256}, None)]
buf1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 256}, None)]
buf1.met_dependencies = []
buf1.users = [NodeUser(node=ExternKernelSchedulerNode(name='buf2'), can_inplace=False, is_weak=False)]
buf1.group.device = cpu
buf1.group.iteration = ((256,), ())
buf1.sizes = ([256], [])
"""
        )
# 定义 buf0_layout，使用 FixedLayout 类创建一个固定布局对象，表示存储在 CPU 上的浮点数张量，
# 大小为 16x16，步长为 [16, 1]
buf0_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])

# 定义 buf1_layout，使用 FixedLayout 类创建另一个固定布局对象，表示存储在 CPU 上的浮点数张量，
# 大小为 16x16，步长为 [16, 1]
buf1_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])

# 定义 buf1_loop_body 类，表示一个循环体对象
class buf1_loop_body:
    # var_ranges 定义变量范围，这里指定了变量 z0 的范围为 256
    var_ranges = {z0: 256}
    # index0 变量设置为 z0，用于循环体中的索引操作
    index0 = z0
    
    # body 方法定义循环体的操作
    def body(self, ops):
        # 获取 index0 变量的索引
        get_index = self.get_index('index0')
        # 从 buf0 中加载数据到 load
        load = ops.load('buf0', get_index)
        # 创建一个常数张量，数值为 2.0，数据类型为 torch.float32
        constant = ops.constant(2.0, torch.float32)
        # 将 load 和 constant 相加
        add = ops.add(load, constant)
        # 获取 index0 变量的索引
        get_index_1 = self.get_index('index0')
        # 将 add 的结果存储到 buf1 中
        store = ops.store('buf1', get_index_1, add, None)
        # 返回存储操作的结果
        return store

# 定义 buf2 对象，表示一个外部内核调度节点
buf2: ExternKernelSchedulerNode(ExternKernelOut)
buf2.writes = [StarDep(name='buf2', mode=None)]  # 指定 buf2 的写入依赖关系
buf2.unmet_dependencies = [StarDep(name='buf1', mode=None)]  # 指定 buf2 的未满足依赖关系
buf2.met_dependencies = [StarDep(name='arg1_1', mode=None)]  # 指定 buf2 的已满足依赖关系
buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]  # 指定使用 buf2 的用户
buf2.node.kernel = extern_kernels.mm  # 指定 buf2 的内核为 extern_kernels.mm

# 使用 self.assertExpectedInline 方法断言预期的内联输出
self.assertExpectedInline(
    open(filename / "ir_post_fusion.txt").read().rstrip(),
    """\
buf0_buf1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
buf0_buf1.writes = [MemoryDep('buf0', c0, {c0: 256}, None), MemoryDep('buf1', c0, {c0: 256}, None)]
buf0_buf1.unmet_dependencies = []
buf0_buf1.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256}, None)]
buf0_buf1.users = []

buf0_buf1.snodes[0] =
buf0: SchedulerNode(ComputedBuffer)
buf0.writes = [MemoryDep('buf0', c0, {c0: 256}, None)]
buf0.unmet_dependencies = []
buf0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256}, None)]
buf0.users = [NodeUser(node=SchedulerNode(name='buf1'), can_inplace=True, is_weak=False)]
buf0.group.device = cpu
buf0.group.iteration = ((256,), ())
buf0.sizes = ([256], [])
arg0_1_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
buf0_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
class buf0_loop_body:
    var_ranges = {z0: 256}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(1.0, torch.float32)
        add = ops.add(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, add, None)
        return store
buf0_buf1.snodes[1] =
buf1: SchedulerNode(ComputedBuffer)
buf1.writes = [MemoryDep('buf1', c0, {c0: 256}, None)]
buf1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 256}, None)]
buf1.met_dependencies = []
buf1.users = [NodeUser(node=ExternKernelSchedulerNode(name='buf2'), can_inplace=False, is_weak=False)]
buf1.group.device = cpu
buf1.group.iteration = ((256,), ())
buf1.sizes = ([256], [])
buf0_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
buf1_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
"""
)
    # 定义一个名为 buf1_loop_body 的类，表示循环体 buf1 的操作
    class buf1_loop_body:
        # 设置变量范围，z0 的取值范围为 0 到 255
        var_ranges = {z0: 256}
        # 初始化 index0 变量，其值为 z0
        index0 = z0
        
        # 定义 body 方法，接收 ops 参数，表示循环体的具体操作
        def body(self, ops):
            # 获取 index0 变量的当前索引
            get_index = self.get_index('index0')
            # 从 buf0 中加载对应索引的值
            load = ops.load('buf0', get_index)
            # 创建一个常量值 2.0，类型为 torch.float32
            constant = ops.constant(2.0, torch.float32)
            # 将加载的值与常量相加
            add = ops.add(load, constant)
            # 获取 index0 变量的当前索引
            get_index_1 = self.get_index('index0')
            # 将相加的结果存储到 buf1 中的相同索引位置
            store = ops.store('buf1', get_index_1, add, None)
            # 返回存储操作的结果
            return store
buf2: ExternKernelSchedulerNode(ExternKernelOut)
# 创建一个名为buf2的ExternKernelSchedulerNode对象，参数为ExternKernelOut
buf2.writes = [StarDep(name='buf2', mode=None)]
# 设置buf2对象的writes属性为一个StarDep对象的列表，包含一个StarDep(name='buf2', mode=None)
buf2.unmet_dependencies = [StarDep(name='buf1', mode=None)]
# 设置buf2对象的unmet_dependencies属性为一个StarDep对象的列表，包含一个StarDep(name='buf1', mode=None)
buf2.met_dependencies = [StarDep(name='arg1_1', mode=None)]
# 设置buf2对象的met_dependencies属性为一个StarDep对象的列表，包含一个StarDep(name='arg1_1', mode=None)
buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
# 设置buf2对象的users属性为一个NodeUser对象的列表，包含一个NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)
buf2.node.kernel = extern_kernels.mm""",
# 设置buf2对象的node属性的kernel属性为字符串extern_kernels.mm"""

        )
# 字符串常量的结束

# 故意只在成功时清理，以简化调试测试
# 在成功时删除指定的文件或目录
shutil.rmtree(filename)

@unittest.skipIf(not HAS_GPU, "requires GPU")
# 如果没有GPU，则跳过测试
def test_debug_multi_tempalte(self):
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(100, 100)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.l(x))

    # 没有失败

    from torch._inductor.utils import fresh_inductor_cache
    # 导入fresh_inductor_cache函数从torch._inductor.utils模块

    with self.assertLogs(
        logging.getLogger("torch._inductor.debug"), level=logging.WARNING
    ), fresh_inductor_cache():
        # 使用assertLogs断言日志输出，并调整日志级别到WARNING
        m = ToyModel().to(device=GPU_TYPE)
        # 创建ToyModel对象并将其移动到GPU设备
        m = torch.compile(m, mode="max-autotune")
        # 编译模型m，模式为"max-autotune"
        input_tensor = torch.randn(100).to(device=GPU_TYPE)
        # 生成一个100维的随机张量，并将其移动到GPU设备
        m(input_tensor)
        # 对模型m使用输入张量input_tensor进行前向传播


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU:
        run_tests(needs="filelock")
        # 如果有CPU，运行测试，需要"filelock"依赖
```