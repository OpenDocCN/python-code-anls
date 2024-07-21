# `.\pytorch\test\inductor\test_snode_runtime.py`

```
# Owner(s): ["module: inductor"]

# 从 unittest 模块中导入 skipIf 函数
from unittest import skipIf

# 导入 torch 库
import torch
# 导入 torch 分布式通信模块
import torch.distributed as dist

# 从 torch._inductor 包中导入 metrics 模块
from torch._inductor import metrics
# 从 torch._inductor.comm_analysis 包中导入 estimate_nccl_collective_runtime 函数
from torch._inductor.comm_analysis import estimate_nccl_collective_runtime
# 从 torch._inductor.compile_fx 包中导入 compile_fx 和 compile_fx_inner 函数
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
# 从 torch._inductor.test_case 包中导入 TestCase 类，并取别名为 InductorTestCase
from torch._inductor.test_case import TestCase as InductorTestCase
# 从 torch._inductor.utils 包中导入 is_collective 函数
from torch._inductor.utils import is_collective
# 从 torch.testing._internal.inductor_utils 包中导入 HAS_CUDA 常量
from torch.testing._internal.inductor_utils import HAS_CUDA

# 导入 torch 操作符的 aten 模块
aten = torch.ops.aten
# 导入 torch 分布式 c10d_functional 模块
c10d = torch.ops.c10d_functional
# 导入 torch 分布式 _c10d_functional 模块
_c10d = torch.ops._c10d_functional

# 定义函数 compile_but_use_eager，用于编译模型图并使用 eager 模式
def compile_but_use_eager(gm, example_inputs):
    def inner_compile(gm, *args, **kwargs):
        compile_fx_inner(gm, *args, **kwargs)
        return gm

    return compile_fx(gm, example_inputs, inner_compile=inner_compile)

# 定义函数 calculate_runtime，计算给定函数的运行时间
def calculate_runtime(f, *args) -> float:
    """
    Assumes all inputs are fp32
    """
    # 重置 metrics
    metrics.reset()
    # 使用 torch.compile 函数编译给定的函数 f，使用 compile_but_use_eager 作为后端
    torch.compile(f, backend=compile_but_use_eager)(*args)
    # 打印节点运行时间信息
    print(metrics.node_runtimes)

    ret = 0.0
    # 将所有节点的运行时间累加到 ret 变量中
    for pair in metrics.node_runtimes:
        ret += pair[1]

    return ret

# 设备类型为 cuda
DEVICE = "cuda"

# 定义 T 函数，用于创建指定大小和类型的张量
def T(*size, dtype=torch.float32, device=DEVICE, grad=False) -> torch.Tensor:
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)

# 定义 TestCase 类，继承自 InductorTestCase，用于测试用例
class TestCase(InductorTestCase):
    device = DEVICE

    """
    Helper methods to compare runtime estimate against 0. Since this estimate is hardware dependent,
    stronger comparisons may fail dependending on the host's specs.

    atol/rtol must be provided explicitly with each call, since precision/rel_tol overrides are not always utilized
    """

    # 断言结果为 0 的辅助方法
    def assertZero(self, x: float):
        assert isinstance(x, float)
        super().assertEqual(x, 0.0, atol=0, rtol=0)

    # 断言结果不为 0 的辅助方法
    def assertNotZero(self, x):
        assert isinstance(x, float)
        super().assertNotEqual(x, 0.0, atol=0, rtol=0)

# 定义 UnsupportedTests 类，继承自 TestCase，用于测试不支持的情况
class UnsupportedTests(TestCase):
    # 测试函数：无操作
    def test_no_op(self):
        def f(a):
            return a

        inp = (T(10, 10),)
        # 断言运行时为 0
        self.assertZero(calculate_runtime(f, *inp))

    # 测试函数：无 CUDA 支持
    def test_no_cuda(self):
        def f(a):
            return a

        inp = (torch.randn((10, 10), device="cpu"),)
        # 断言运行时为 0
        self.assertZero(calculate_runtime(f, *inp))

# 定义 ComputeBoundedTests 类，继承自 TestCase，用于测试有限条件下的计算
class ComputeBoundedTests(TestCase):
    # 测试函数：一维卷积
    def test_conv1d(self):
        def f(x, y):
            return torch.nn.functional.conv1d(x, y)

        inp = (T(33, 16, 30), T(20, 16, 5))
        # 断言运行时不为 0
        self.assertNotZero(calculate_runtime(f, *inp))

    # 测试函数：二维卷积
    def test_conv2d(self):
        def f(x, y):
            return torch.nn.functional.conv2d(x, y, padding=1)

        inp = (T(8, 4, 3, 3), T(1, 4, 5, 5))
        # 断言运行时不为 0
        self.assertNotZero(calculate_runtime(f, *inp))

    # 测试函数：二维转置卷积
    def test_conv2d_transpose(self):
        def f(x, y):
            return torch.nn.functional.conv_transpose2d(x, y, padding=1)

        inp = (T(8, 1, 1, 1), T(1, 4, 5, 5))
        # 断言运行时不为 0
        self.assertNotZero(calculate_runtime(f, *inp))
    # 定义测试函数 test_conv3d，用于测试 torch.nn.functional.conv3d 函数
    def test_conv3d(self):
        # 定义内部函数 f，接受两个参数 x 和 y，并调用 torch.nn.functional.conv3d 函数进行卷积操作
        def f(x, y):
            return torch.nn.functional.conv3d(x, y)

        # 定义输入数据 inp，包含两个张量参数
        inp = (T(20, 16, 50, 10, 20), T(33, 16, 3, 3, 3))
        # 断言调用 calculate_runtime 函数返回值不为零
        self.assertNotZero(calculate_runtime(f, *inp))

    # 定义测试函数 test_mm，用于测试 torch.mm 函数
    def test_mm(self):
        # 定义内部函数 f，接受两个参数 a 和 b，并调用 torch.mm 函数执行矩阵乘法
        def f(a, b):
            return torch.mm(a, b)

        # 定义输入数据 inp，包含两个相同大小的矩阵
        inp = (
            T(10, 10),
            T(10, 10),
        )
        # 断言调用 calculate_runtime 函数返回值不为零
        self.assertNotZero(calculate_runtime(f, *inp))

    # 定义测试函数 test_addmm，用于测试 torch.addmm 函数
    def test_addmm(self):
        # 定义内部函数 f，接受三个参数 a、b 和 c，并调用 torch.addmm 函数执行矩阵相加和矩阵乘法
        def f(a, b, c):
            return torch.addmm(a, b, c)

        # 定义输入数据 inp，包含三个相同大小的矩阵
        inp = (
            T(10, 10),
            T(10, 10),
            T(10, 10),
        )
        # 断言调用 calculate_runtime 函数返回值不为零
        self.assertNotZero(calculate_runtime(f, *inp))

    # 定义测试函数 test_bmm，用于测试 torch.bmm 函数
    def test_bmm(self):
        # 定义内部函数 f，接受两个参数 a 和 b，并调用 torch.bmm 函数执行批量矩阵乘法
        def f(a, b):
            return torch.bmm(a, b)

        # 定义输入数据 inp，包含两个三维矩阵
        inp = (
            T(10, 10, 10),
            T(10, 10, 10),
        )
        # 断言调用 calculate_runtime 函数返回值不为零
        self.assertNotZero(calculate_runtime(f, *inp))
class MemoryBoundedTests(TestCase):
    # 测试用例类，用于测试内存限制场景

    def test_relu(self):
        # 测试ReLU函数
        def f(a):
            return torch.nn.functional.relu(a)
        # 准备输入数据
        inp = (T(10, 10),)
        # 断言运行时间不为零
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_horizontal_reduction_pointwise(self):
        # 测试水平缩减和逐点操作
        def f(a):
            # 沿着第一维度求和
            b = a.sum(dim=1)
            # 对输入数据执行余弦函数
            c = a.cos()
            return b, c
        # 准备输入数据
        inp = (T(10, 10),)
        # 断言运行时间不为零
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_pointwise(self):
        # 测试逐点操作
        def f(x):
            # 计算输入张量的余弦函数
            return x.cos()
        # 准备输入数据
        inp = (T(10),)
        # 断言运行时间不为零
        self.assertNotZero(calculate_runtime(f, *inp))

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic(self):
        # 测试动态场景
        def f(x):
            # 计算输入张量的余弦函数
            return x.cos()
        # 准备输入数据
        inp = (T(10),)
        # 断言运行时间不为零
        self.assertNotZero(calculate_runtime(f, *inp))


@skipIf(not dist.is_available(), "requires distributed")
class TestCommAnalysis(TestCase):
    # 测试通信分析类，要求分布式环境可用

    WORLD_SIZE: int = 8
    RANKS = list(range(8))

    def _verify_runtime_estimation(self, fn, inps):
        # 验证运行时估计函数
        from torch.testing._internal.distributed.fake_pg import FakeStore

        # 创建虚拟存储
        store = FakeStore()
        # 初始化进程组
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.WORLD_SIZE, store=store
        )
        try:
            # 重置度量数据
            metrics.reset()
            # 编译并运行函数
            torch.compile(fn)(*inps)
            found_collective = False
            # 遍历节点运行时间度量
            for snode, runtime in metrics.node_runtimes:
                if not is_collective(snode.node):
                    continue
                found_collective = True
                # 估算NCCL集合运行时
                est = estimate_nccl_collective_runtime(snode.node)
                self.assertNotZero(est)
                # 确保估算函数正常工作
                self.assertNotZero(runtime)
            # 确保图中存在集合核心
            self.assertTrue(found_collective)
        finally:
            # 销毁进程组
            dist.destroy_process_group()

    def test_legacy_all_reduce(self):
        # 测试传统的全局归约操作
        def fn(x):
            r = c10d.all_reduce(x, "sum", "", self.RANKS, self.WORLD_SIZE)
            return c10d.wait_tensor(r)
        # 准备输入数据
        inp = T(10, 10)
        # 验证运行时估计
        self._verify_runtime_estimation(fn, (inp,))

    def test_legacy_all_reduce_coalesced(self):
        # 测试传统的合并全局归约操作
        def fn(x):
            rs = c10d.all_reduce_coalesced(x, "sum", "", self.RANKS, self.WORLD_SIZE)
            return [c10d.wait_tensor(r) for r in rs]
        # 准备输入数据
        inp = [T(10, 10), T(15, 15)]
        # 验证运行时估计
        self._verify_runtime_estimation(fn, (inp,))
    # 定义测试方法：测试使用all_gather_into_tensor_coalesced函数
    def test_legacy_all_gather_into_tensor_coalesced(self):
        # 定义内部函数fn，接受参数x
        def fn(x):
            # 调用c10d.all_gather_into_tensor_coalesced函数，将结果存储在rs中
            rs = c10d.all_gather_into_tensor_coalesced(
                x,
                "",  # 空字符串作为目标设备（device）的标识符
                self.RANKS,  # 变量RANKS表示进程的排名
                self.WORLD_SIZE,  # 变量WORLD_SIZE表示进程组中的总进程数
            )
            # 返回一个列表，其中每个元素都是c10d.wait_tensor函数对rs中对应张量的等待结果
            return [c10d.wait_tensor(r) for r in rs]

        # 定义输入inp为包含两个T(10, 10)和T(15, 15)张量的列表
        inp = [T(10, 10), T(15, 15)]
        # 调用_verify_runtime_estimation方法验证fn的运行时间估计，传入参数inp的元组
        self._verify_runtime_estimation(fn, (inp,))

    # 定义测试方法：测试使用all_reduce函数
    def test_all_reduce(self):
        # 定义内部函数fn，接受参数x
        def fn(x):
            # 调用_c10d.all_reduce函数，将结果存储在r中，使用"sum"操作进行张量归约，目标设备标识符为"0"
            r = _c10d.all_reduce(x, "sum", "0")
            # 返回_c10d.wait_tensor函数对r张量的等待结果
            return _c10d.wait_tensor(r)

        # 定义输入inp为T(10, 10)张量
        inp = T(10, 10)
        # 调用_verify_runtime_estimation方法验证fn的运行时间估计，传入参数inp的元组
        self._verify_runtime_estimation(fn, (inp,))

    # 定义测试方法：测试使用all_reduce_coalesced函数
    def test_all_reduce_coalesced(self):
        # 定义内部函数fn，接受参数x
        def fn(x):
            # 调用_c10d.all_reduce_coalesced函数，将结果存储在rs中，使用"sum"操作进行张量归约，目标设备标识符为"0"
            rs = _c10d.all_reduce_coalesced(x, "sum", "0")
            # 返回一个列表，其中每个元素都是_c10d.wait_tensor函数对rs中对应张量的等待结果
            return [_c10d.wait_tensor(r) for r in rs]

        # 定义输入inp为包含两个T(10, 10)和T(15, 15)张量的列表
        inp = [T(10, 10), T(15, 15)]
        # 调用_verify_runtime_estimation方法验证fn的运行时间估计，传入参数inp的元组
        self._verify_runtime_estimation(fn, (inp,))

    # 定义测试方法：测试使用all_gather_into_tensor函数
    def test_all_gather_into_tensor(self):
        # 定义内部函数fn，接受参数x
        def fn(x):
            # 调用_c10d.all_gather_into_tensor函数，将结果存储在rs中
            rs = _c10d.all_gather_into_tensor(
                x,
                self.WORLD_SIZE,  # 参数表示进程组中的总进程数
                "0",  # 目标设备标识符为"0"
            )
            # 返回一个列表，其中每个元素都是_c10d.wait_tensor函数对rs中对应张量的等待结果
            return [_c10d.wait_tensor(r) for r in rs]

        # 定义输入inp为T(10, 10)张量
        inp = T(10, 10)
        # 调用_verify_runtime_estimation方法验证fn的运行时间估计，传入参数inp的元组
        self._verify_runtime_estimation(fn, (inp,))

    # 定义测试方法：测试使用all_gather_into_tensor_coalesced函数
    def test_all_gather_into_tensor_coalesced(self):
        # 定义内部函数fn，接受参数x
        def fn(x):
            # 调用_c10d.all_gather_into_tensor_coalesced函数，将结果存储在rs中
            rs = _c10d.all_gather_into_tensor_coalesced(
                x,
                self.WORLD_SIZE,  # 参数表示进程组中的总进程数
                "0",  # 目标设备标识符为"0"
            )
            # 返回一个列表，其中每个元素都是_c10d.wait_tensor函数对rs中对应张量的等待结果
            return [_c10d.wait_tensor(r) for r in rs]

        # 定义输入inp为包含两个T(10, 10)和T(15, 15)张量的列表
        inp = [T(10, 10), T(15, 15)]
        # 调用_verify_runtime_estimation方法验证fn的运行时间估计，传入参数inp的元组
        self._verify_runtime_estimation(fn, (inp,))

    # 定义测试方法：测试使用reduce_scatter_tensor函数
    def test_reduce_scatter_tensor(self):
        # 定义内部函数fn，接受参数x
        def fn(x):
            # 调用_c10d.reduce_scatter_tensor函数，将结果存储在rs中
            rs = _c10d.reduce_scatter_tensor(
                x,
                "sum",  # 使用"sum"操作进行张量归约
                self.WORLD_SIZE,  # 参数表示进程组中的总进程数
                "0",  # 目标设备标识符为"0"
            )
            # 返回一个列表，其中每个元素都是_c10d.wait_tensor函数对rs中对应张量的等待结果
            return [_c10d.wait_tensor(r) for r in rs]

        # 定义输入inp为T(self.WORLD_SIZE, 10)张量
        inp = T(self.WORLD_SIZE, 10)
        # 调用_verify_runtime_estimation方法验证fn的运行时间估计，传入参数inp的元组
        self._verify_runtime_estimation(fn, (inp,))

    # 定义测试方法：测试使用reduce_scatter_tensor_coalesced函数
    def test_reduce_scatter_tensor_coalesced(self):
        # 定义内部函数fn，接受参数x
        def fn(x):
            # 调用_c10d.reduce_scatter_tensor_coalesced函数，将结果存储在rs中
            rs = _c10d.reduce_scatter_tensor_coalesced(
                x,
                "sum",  # 使用"sum"操作进行张量归约
                self.WORLD_SIZE,  # 参数表示进程组中的总进程数
                "0",  # 目标设备标识符为"0"
            )
            # 返回一个列表，其中每个元素都是_c10d.wait_tensor函数对rs中对应张量的等待结果
            return [_c10d.wait_tensor(r) for r in rs]

        # 定义输入inp为包含两个T(self.WORLD_SIZE, 10)和T(self.WORLD_SIZE, 15)张量的列表
        inp = [T(self.WORLD_SIZE, 10), T(self.WORLD_SIZE, 15)]
        # 调用_verify_runtime_estimation方法验证fn的运行时间估计，传入参数inp的元组
        self._verify_runtime_estimation(fn, (inp,))
# 如果该脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch 库中导入 _inductor 模块的 test_case 子模块中的 run_tests 函数
    from torch._inductor.test_case import run_tests

    # 如果存在 CUDA 支持（假设 HAS_CUDA 是一个表示 CUDA 是否可用的变量）
    if HAS_CUDA:
        # 运行测试，确保测试需要使用文件锁（filelock）
        run_tests(needs="filelock")
```