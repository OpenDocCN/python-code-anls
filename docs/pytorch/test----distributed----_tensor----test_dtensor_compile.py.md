# `.\pytorch\test\distributed\_tensor\test_dtensor_compile.py`

```py
# 引入必要的模块和库
import copy  # 引入 copy 模块，用于复制对象
import functools  # 引入 functools 模块，用于高阶函数的操作
import unittest  # 引入 unittest 模块，用于编写和运行单元测试
from unittest.mock import patch  # 从 unittest.mock 模块中引入 patch 函数，用于模拟对象和函数调用

import torch  # 引入 PyTorch 深度学习库
import torch._dynamo  # 引入 PyTorch 的动态图模块
import torch._dynamo.testing  # 引入 PyTorch 动态图测试模块
import torch.distributed as dist  # 引入 PyTorch 分布式模块
import torch.nn as nn  # 引入 PyTorch 神经网络模块
from torch._C import FileCheck  # 从 PyTorch _C 模块中引入 FileCheck 类
from torch._inductor.utils import run_and_get_triton_code  # 从 PyTorch _inductor.utils 模块中引入运行和获取 Triton 代码的函数
from torch.distributed._tensor import (  # 引入 PyTorch 分布式张量模块中的多个类和函数
    DeviceMesh,
    DTensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed._tensor.placement_types import (  # 引入 PyTorch 分布式张量的放置类型模块中的多个类
    DTensorSpec,
    TensorMeta,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # 引入 PyTorch 分布式算法的检查点包装器模块中的类和函数
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 引入 PyTorch 全分片数据并行模块中的 FSDP 类
from torch.distributed.tensor.parallel import (  # 引入 PyTorch 分布式张量并行模块中的多个类和函数
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 从 PyTorch 测试内部分布式模块中引入跳过 GPU 数量小于指定值的函数
from torch.testing._internal.common_utils import (  # 引入 PyTorch 测试内部通用工具模块中的多个函数
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 引入 PyTorch 内部分布式张量常见张量模块中的多个类和函数
    DTensorTestBase,
    MLPModule,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore  # 从 PyTorch 内部分布式模块中引入虚假进程组存储模块
from torch.utils._triton import has_triton  # 从 PyTorch Triton 工具模块中引入检查是否有 Triton 的函数
from torch.utils.checkpoint import checkpoint  # 引入 PyTorch 检查点模块中的 checkpoint 函数

class SimpleModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)  # 初始化一个 MLPModule 类的实例并赋值给 self.mlp_0
        self.mlp_1 = MLPModule(device)  # 初始化一个 MLPModule 类的实例并赋值给 self.mlp_1

    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))  # 前向传播函数，先通过 self.mlp_0 处理输入，再通过 self.mlp_1 处理结果

def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g.code  # 将传入的 fx_g 对象的代码保存到 graph_cell 的第一个元素中
    return fx_g

# 创建两个空列表，用于存储前向和反向计算的图表达式
fw_graph_cell = [None]
bw_graph_cell = [None]
# 使用 functools.partial 创建两个自定义编译器，分别处理前向和反向计算的图表达式提取
fw_compiler = functools.partial(extract_graph, graph_cell=fw_graph_cell)
bw_compiler = functools.partial(extract_graph, graph_cell=bw_graph_cell)

from functorch.compile import min_cut_rematerialization_partition  # 从 functorch.compile 模块中引入最小剪切重材料化分区函数
from torch._dynamo.backends.common import aot_autograd  # 从 PyTorch _dynamo.backends.common 模块中引入 aot_autograd 函数

# 使用 aot_autograd 函数生成 AOT 自动求导图
aot_eager_graph = aot_autograd(
    fw_compiler=fw_compiler,  # 前向编译器使用自定义的 fw_compiler 函数
    bw_compiler=bw_compiler,  # 反向编译器使用自定义的 bw_compiler 函数
    partition_fn=min_cut_rematerialization_partition,  # 使用最小剪切重材料化分区函数
)

class TestDTensorCompile(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        fake_store = FakeStore()  # 创建虚假存储对象
        dist.init_process_group(  # 初始化分布式进程组
            "fake", store=fake_store, rank=0, world_size=self.world_size
        )

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()  # 销毁分布式进程组

    @property
    def device_type(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"  # 如果有 CUDA 设备，则返回 "cuda"，否则返回 "cpu"

    @property
    def world_size(self) -> int:
        return 2  # 设置分布式环境的总进程数为 2
    # 定义测试函数 test_placement_compile
    def test_placement_compile(self):
        # 定义内部函数 fn，接受参数 x
        def fn(x):
            # 初始化变量 a 为 0
            a = 0
            # 如果 x 是 Replicate 类型，增加 a 的值 1
            if x.is_replicate():
                a += 1
            # 如果 x 是 Shard 类型，增加 a 的值 2
            if x.is_shard():
                a += 2
                # 如果 x 的维度 dim 小于 0，抛出 RuntimeError 异常
                if x.dim < 0:
                    raise RuntimeError("dim < 0")
            # 如果 x 是 Shard 类型且参数为 0，增加 a 的值 2
            if x.is_shard(0):
                a += 2
            # 如果 x 是 Shard 类型且指定维度为 0，增加 a 的值 2
            if x.is_shard(dim=0):
                a += 2
            # 如果 x 是 Shard 类型且维度为 None，增加 a 的值 2
            if x.is_shard(dim=None):
                a += 2
            # 如果 x 是 Partial 类型，增加 a 的值 3
            if x.is_partial():
                a += 3
            # 返回计算后的 a 值
            return a

        # 使用 torch.compile 进行 AOT 编译，设置 backend="aot_eager", fullgraph=True
        compiled_fn = torch.compile(backend="aot_eager", fullgraph=True)(fn)

        # 遍历对象列表 [Shard(0), Replicate(), Partial()]
        for x in [Shard(0), Replicate(), Partial()]:
            # 获取未编译版本 fn(x) 的结果
            opt_fn = fn(x)
            # 获取编译版本 compiled_fn(x) 的结果
            compiled_out = compiled_fn(x)
            # 断言未编译和编译结果相等
            self.assertEqual(opt_fn, compiled_out)

    # 定义测试函数 test_device_mesh_compile
    def test_device_mesh_compile(self):
        # 定义内部函数 fn，接受参数 x
        def fn(x):
            # 调用 x 的 size() 方法，获取返回值并赋给变量 a
            a = x.size()
            # 调用 x 的 size(0) 方法，获取返回值并赋给变量 b
            b = x.size(0)
            # 调用 x 的 size(mesh_dim=0) 方法，获取返回值并赋给变量 c
            c = x.size(mesh_dim=0)
            # 计算 size 的和
            size = a + b + c
            # 调用 x 的 get_coordinate() 方法，获取返回值并赋给变量 coord
            coord = x.get_coordinate()
            # 调用 x 的 get_group() 方法，获取返回值并赋给变量 group
            group = x.get_group()
            # 返回 size, coord, group 三个变量
            return size, coord, group

        # 使用 torch.compile 进行 AOT 编译，设置 backend="aot_eager", fullgraph=True
        compiled_fn = torch.compile(backend="aot_eager", fullgraph=True)(fn)

        # 创建 DeviceMesh 对象 mesh
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # 获取未编译版本 fn(mesh) 的结果
        opt_fn = fn(mesh)
        # 获取编译版本 compiled_fn(mesh) 的结果
        compiled_out = compiled_fn(mesh)
        # 断言未编译和编译结果相等
        self.assertEqual(opt_fn, compiled_out)

    # 定义测试函数 test_fakify_dtensor
    def test_fakify_dtensor(self):
        # 创建 DeviceMesh 对象 mesh
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 定义内部函数 fn，接受参数 x
        # 直接返回参数 x
        def fn(x):
            return x

        # 创建 DTensor 对象 x，使用 torch.rand(1) 初始化数据
        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        # 获取 fn(x) 的返回值，赋给 ref
        ref = fn(x)

        # 使用 torch.compile 进行 AOT 编译，设置 backend="aot_eager", fullgraph=True
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        # 获取编译版本 opt_fn(x) 的结果，赋给 res
        res = opt_fn(x)
        # 断言未编译和编译结果相等
        self.assertEqual(res, ref)

    # 定义测试函数 test_dynamo_dtensor
    def test_dynamo_dtensor(self):
        # 创建 DeviceMesh 对象 mesh
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 定义内部函数 fn，接受参数 x
        # 对参数 x 进行平方后加 2 的运算，并返回结果
        def fn(x):
            return x * x + 2

        # 创建 DTensor 对象 x，使用 torch.rand(1) 初始化数据
        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        # 获取 fn(x) 的返回值，赋给 ref
        ref = fn(x)

        # 使用 torch.compile 进行 AOT 编译，设置 backend="aot_eager", fullgraph=True
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        # 获取编译版本 opt_fn(x) 的结果，赋给 res
        res = opt_fn(x)
        # 断言未编译和编译结果相等
        self.assertEqual(res, ref)

    # 定义测试函数 test_dtensor_attribute_access_on_intermediate
    def test_dtensor_attribute_access_on_intermediate(self):
        # 创建 DeviceMesh 对象 mesh
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 定义内部函数 fn，接受参数 x
        def fn(x):
            # 对 x 进行乘法操作后乘以 2，结果赋给 tmp
            tmp = x * 2
            # 如果 tmp 的 placements[0] 是 Shard 类型
            if tmp.placements[0].is_shard():
                # 返回 tmp 的 _local_tensor 加 2
                return tmp._local_tensor + 2
            else:
                # 返回 tmp 的 _local_tensor 加 3
                return tmp._local_tensor + 3

        # 创建 DTensor 对象 x，使用 torch.ones(4) 初始化数据
        x = DTensor.from_local(torch.ones(4), mesh, [Shard(0)], run_check=False)
        # 获取 fn(x) 的返回值，赋给 ref
        ref = fn(x)

        # 使用 torch.compile 进行 AOT 编译，设置 backend="aot_eager", fullgraph=True
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        # 获取编译版本 opt_fn(x) 的结果，赋给 res
        res = opt_fn(x)
        # 断言未编译和编译结果相等
        self.assertEqual(res, ref)
    def test_dtensor_constructor_w_graph_break(self):
        # 创建一个设备网格对象，使用给定的设备类型和全局大小创建
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # 创建一个形状为 (64, 32) 的张量 x，要求其梯度信息被跟踪
        x = torch.randn(64, 32, requires_grad=True)
        # 创建一个 DTensorSpec 对象，指定了网格、复制和分片的处理方式，以及张量的元数据
        spec = DTensorSpec(
            mesh,
            (Replicate(), Shard(0)),
            tensor_meta=TensorMeta(
                shape=torch.Size([128, 32]), stride=(32, 1), dtype=x.dtype
            ),
        )

        # 定义一个函数 fn，将输入 x 封装成 DTensor 对象，并打印一条消息
        def fn(x):
            print("graph break!")
            return DTensor(
                x,
                spec,
                requires_grad=x.requires_grad,
            )

        # 调用函数 fn，生成一个 DTensor 对象 out
        out = fn(x)
        # 使用 eager 模式编译函数 fn，并运行，生成另一个 DTensor 对象 out2
        out2 = torch.compile(fn, backend="eager")(x)

    def test_dtensor_constructor_w_dynamo_disable(self):
        # 创建一个设备网格对象，使用给定的设备类型和全局大小创建
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # 创建一个形状为 (32,) 的张量 x，要求其梯度信息被跟踪
        x = torch.randn(32, requires_grad=True)
        # 创建一个 DTensorSpec 对象，指定了网格、复制的处理方式，以及张量的元数据
        spec = DTensorSpec(
            mesh,
            (Replicate(),),
            tensor_meta=TensorMeta(shape=torch.Size([32]), stride=(1,), dtype=x.dtype),
        )

        # 使用 torch._dynamo.disable 禁用动态图递归，定义函数 fn，将输入 x 封装成 DTensor 对象，并打印一条消息
        @torch._dynamo.disable(recursive=False)
        def fn(x):
            print("foo")
            return DTensor(
                x,
                spec,
                requires_grad=x.requires_grad,
            )

        # 调用函数 fn，生成一个 DTensor 对象 out
        out = fn(x)
        # 使用 eager 模式编译函数 fn，并运行，生成另一个 DTensor 对象 out2
        out2 = torch.compile(fn, backend="eager")(x)
        # 断言 out 和 out2 的值相等
        self.assertEqual(out, out2)

    def test_dtensor_noncontiguous_output(self):
        # 创建一个设备网格对象，使用给定的设备类型和全局大小创建
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 定义函数 fn，接受 x, y, z 三个输入，进行一系列操作，并返回结果
        def fn(x, y, z):
            # 对 x 进行维度变换和连续化处理
            x_transposed = x.permute(0, 2, 1).contiguous()
            # 使用底层 C++ 函数 linear 进行线性操作
            tmp = torch._C._nn.linear(x_transposed, y, z)
            # 对 tmp 再次进行维度变换
            return tmp.permute(0, 2, 1)

        # 创建三个张量 x_inner, y_inner, z_inner，要求其梯度信息被跟踪
        x_inner = torch.randn(4, 16, 4, requires_grad=True)
        y_inner = torch.randn(4, 16, requires_grad=True)
        z_inner = torch.randn(4, requires_grad=True)
        # 使用 DTensor.from_local 方法，根据输入张量和网格创建 DTensor 对象 x, y, z
        x = DTensor.from_local(x_inner, mesh, [Shard(1)], run_check=False)
        y = DTensor.from_local(y_inner, mesh, [Shard(1)], run_check=False)
        z = DTensor.from_local(z_inner, mesh, [Replicate()], run_check=False)
        # 使用 aot_eager 模式编译函数 fn，并运行，生成一个 DTensor 对象 out
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, y, z)
        # 对 out 进行连续化处理，求和并反向传播梯度
        out.contiguous().sum().backward()
    # 定义一个测试方法，用于测试在本地创建 DTensor
    def test_dynamo_dtensor_from_local(self):
        # 创建一个设备网格对象，使用给定的设备类型和世界大小
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 在函数内部创建 DTensor，并进行一些计算操作
        def fn(x):
            # 使用 from_local 方法创建 DTensor，不进行检查运行
            dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)
            # 将 DTensor 转换为本地并加上常数 2
            return dt.to_local() + 2

        # 下面是引用操作的方法作为参考
        # from torch.distributed._tensor.api import _FromTorchTensor
        # def from_local_tensor(x):
        #     return _FromTorchTensor.apply(x, mesh, [Replicate()], False)

        # 创建一个名为 _dt_lib_def 的库对象，用于定义 dtensor 的操作
        _dt_lib_def = torch.library.Library("dtensor", "DEF")
        _dt_lib_def.define("from_local(Tensor self) -> Tensor")

        # 创建一个名为 _dt_lib_impl 的库对象，用于实现 dtensor 的 from_local 操作
        _dt_lib_impl = torch.library.Library("dtensor", "IMPL")
        _dt_lib_impl.impl("from_local", from_local_tensor, "Autograd")

        # 创建一个 requires_grad 为 True 的张量 x
        x = torch.ones(1, requires_grad=True)
        # 调用 fn 函数并得到其结果 ref
        ref = fn(x)
        # 创建一个带有后端计数器的编译对象 cnt
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        # 使用 cnt 后端编译 fn 函数，并设置 fullgraph 为 True
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        # 使用编译后的函数 opt_fn 对 x 进行计算得到 res
        res = opt_fn(x)
        # 反向传播计算梯度
        res.sum().backward()

        # 断言 res 等于 ref
        self.assertEqual(res, ref)
        # 断言 frame_count 等于 1
        self.assertEqual(cnt.frame_count, 1)

        # 测试用户是否可以调用带有网格和放置方式参数的 from_local 方法，并保证其正常工作
        def from_local_kwargs_fn(x):
            # 使用关键字参数调用 from_local 方法创建 DTensor
            dt = DTensor.from_local(
                x, device_mesh=mesh, placements=[Replicate()], run_check=False
            )
            # 将 DTensor 转换为本地并加上常数 2
            return dt.to_local() + 2

        # 更新 ref 为 from_local_kwargs_fn(x) 的结果
        ref = from_local_kwargs_fn(x)
        # 使用相同的计数器 cnt 编译 from_local_kwargs_fn 函数
        opt_kwargs_fn = torch.compile(from_local_kwargs_fn, backend=cnt, fullgraph=True)
        # 使用编译后的函数 opt_kwargs_fn 对 x 进行计算得到 res
        res = opt_kwargs_fn(x)
        # 断言 res 等于 ref
        self.assertEqual(res, ref)
        # 断言 frame_count 等于 2
        self.assertEqual(cnt.frame_count, 2)
    def test_dtensor_partial_placement_redistribute_unbalanced_correct_strides(self):
        # Partial -> Shard on an unbalanced tensor results in:
        # - A contiguous DTensor
        # - where the inner _local_tensor is noncontiguous
        
        # 定义一个 Shard 类型的放置策略对象
        placement = Shard(1)

        def fn(x):
            # 将输入张量 x 在 mesh 上使用 placement 放置策略进行重新分配
            out = x.redistribute(mesh, [placement])
            return out

        # 暂时忽略 setUp()，并在跟踪期间使用 rank3 图
        dist.destroy_process_group()
        # 创建一个虚拟存储 FakeStore 对象
        fake_store = FakeStore()
        # 初始化一个名为 "fake" 的虚拟进程组，rank 为 3，总共 2 个进程
        dist.init_process_group("fake", store=fake_store, rank=3, world_size=2)
        # 创建一个 DeviceMesh 对象，使用 self.device_type 和 [1, 3] 作为参数
        mesh = DeviceMesh(self.device_type, [1, 3])

        # 创建一个形状为 (10, 257, 160) 的随机张量 x，需要梯度
        x = torch.randn(10, 257, 160, requires_grad=True)
        # 使用 DTensor 类的 from_local 方法创建一个 DTensor 对象 x_dt
        x_dt = DTensor.from_local(
            x,
            mesh,
            [Partial()],  # 使用 Partial 放置策略
            run_check=False,
            shape=(10, 257, 160),
            stride=(41120, 160, 1),
        )

        # 调用 fn 函数，对 x_dt 进行操作，得到 tmp_dt
        tmp_dt = fn(x_dt)
        # 创建一个 Torch 的子类 FakeTensorMode 对象
        fake_mode = torch._subclasses.FakeTensorMode()
        # 使用 fake_mode 的 from_tensor 方法创建 tmp_dt 的伪造对象 tmp_dt_fake
        tmp_dt_fake = fake_mode.from_tensor(tmp_dt)
        
        # 断言 tmp_dt 与 tmp_dt_fake 的形状相同
        self.assertEqual(tmp_dt.shape, tmp_dt_fake.shape)
        # 断言 tmp_dt 与 tmp_dt_fake 的步长相同
        self.assertEqual(tmp_dt.stride(), tmp_dt_fake.stride())
        # 断言 tmp_dt 的 _local_tensor 属性与 tmp_dt_fake 的 _local_tensor 属性的形状相同
        self.assertEqual(tmp_dt._local_tensor.shape, tmp_dt_fake._local_tensor.shape)
        # 断言 tmp_dt 的 _local_tensor 属性与 tmp_dt_fake 的 _local_tensor 属性的步长相同
        self.assertEqual(
            tmp_dt._local_tensor.stride(), tmp_dt_fake._local_tensor.stride()
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_contiguous_dtensor_noncontiguous_local_as_tangent(self):
        # Partial -> Shard on an unbalanced tensor results in:
        # - A contiguous DTensor
        # - where the inner _local_tensor is noncontiguous
        # When this tensor is a fwd graph output,
        # AOTAutograd needs to make sure we trace the backward
        # with a contiguous tangent
        
        # 定义一个 Shard 类型的放置策略对象
        placement = Shard(1)

        def fn(x):
            # 将输入张量 x 在 mesh 上使用 placement 放置策略进行重新分配
            out = x.redistribute(mesh, [placement])
            return out

        # 暂时忽略 setUp()，并在跟踪期间使用 rank3 图
        dist.destroy_process_group()
        # 创建一个虚拟存储 FakeStore 对象
        fake_store = FakeStore()
        # 初始化一个名为 "fake" 的虚拟进程组，rank 为 3，总共 2 个进程
        dist.init_process_group("fake", store=fake_store, rank=3, world_size=2)
        # 创建一个 DeviceMesh 对象，使用 self.device_type 和 [1, 3] 作为参数
        mesh = DeviceMesh(self.device_type, [1, 3])

        # 创建一个形状为 (10, 257, 160) 的随机张量 x，需要梯度
        x = torch.randn(10, 257, 160, requires_grad=True)
        # 使用 DTensor 类的 from_local 方法创建一个 DTensor 对象 x_dt
        x_dt = DTensor.from_local(
            x,
            mesh,
            [Partial()],  # 使用 Partial 放置策略
            run_check=False,
            shape=(10, 257, 160),
            stride=(41120, 160, 1),
        )

        # 对 fn 函数进行 torch 编译，得到 out_dt
        out_dt = torch.compile(fn)(x_dt)
        # 如果我们没有正确处理跟踪时的连续切线，将会失败并引发感应器步长断言
        # 将 out_dt 转为本地张量，求和并进行反向传播
        out_dt.to_local().sum().backward()
    # 定义测试函数，测试从 Dynamo 到本地关键字参数的转换
    def test_dynamo_to_local_kwargs(self):
        # 创建设备网格，使用设备类型和全局大小创建
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 定义内部函数 fn，接受参数 x，返回 dt.to_local 的结果加 2
        def fn(x):
            return dt.to_local(grad_placements=[Shard(0)]) + 2

        # 编译函数 fn，使用 "aot_eager" 后端，完整图形模式
        fn_opt = torch.compile(fn, backend="aot_eager", fullgraph=True)
        # 创建张量 x，全为 1
        x = torch.ones(4)
        # 使用 x 创建 DTensor，指定网格和复制副本，不运行检查
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        # 计算 fn(dt) 和 fn_opt(dt)，比较两者结果是否相等
        out_ref = fn(dt)
        out_test = fn_opt(dt)
        self.assertEqual(out_ref, out_test)

    # 定义测试函数，测试从 Dynamo 到本地关键字参数的转换，并使用前向钩子
    def test_dynamo_to_local_kwargs_forward_hook(self):
        # 创建设备网格，使用设备类型和全局大小创建
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 定义前向钩子 fw_hook，处理模块、输入和输出
        def fw_hook(module, inp, out):
            # 将输出 out 转换为本地，使用其梯度位置加 2
            tmp = out.to_local(grad_placements=out.placements) + 2
            # 返回从本地创建的 DTensor，使用网格和输出位置，不运行检查
            return DTensor.from_local(tmp, mesh, out.placements, run_check=False)

        # 创建线性模块 mod，输入维度 4，输出维度 4
        mod = torch.nn.Linear(4, 4)
        # 注册前向钩子 fw_hook 到模块 mod
        mod.register_forward_hook(fw_hook)

        # 创建另一个线性模块 mod，输入维度 4，输出维度 4
        mod = torch.nn.Linear(4, 4)
        # 再次注册前向钩子 fw_hook 到模块 mod
        mod.register_forward_hook(fw_hook)
        # 使用 DTensor 从本地创建权重和偏置参数，使用网格和复制副本，不运行检查
        mod.weight = torch.nn.Parameter(
            DTensor.from_local(mod.weight, mesh, [Replicate()], run_check=False)
        )
        mod.bias = torch.nn.Parameter(
            DTensor.from_local(mod.bias, mesh, [Replicate()], run_check=False)
        )
        # 编译模块 mod，使用 "aot_eager" 后端，完整图形模式
        opt_mod = torch.compile(mod, backend="aot_eager", fullgraph=True)

        # 创建输入张量 x，维度为 4x4，全为 1
        x = torch.ones(4, 4)
        # 使用 x 创建 DTensor，指定网格和复制副本，不运行检查
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        # 计算 mod(dt) 和 opt_mod(dt)，比较两者结果是否相等
        out_ref = mod(dt)
        out_test = opt_mod(dt)
        self.assertEqual(out_ref, out_test)

    # 跳过测试，如果没有 Triton 或 GPU 需要，Inductor+gpu 需要 Triton 和最新的 GPU 架构
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_different_gradient_placement(self):
        # 创建设备网格，使用设备类型和全局大小创建
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 定义函数 fn，接受参数 x, y, z，执行一系列操作并返回输出
        def fn(x, y, z):
            # 对 x 进行维度置换
            permute = x.permute(0, 2, 1)
            # 进行连续内存布局
            permute2 = permute.contiguous()
            # 使用 layer_norm 函数对 permute2 进行层归一化处理
            layer_norm = torch.nn.functional.layer_norm(permute2, (4,), y, z, 1e-05)
            # 对 layer_norm 的输出进行维度置换
            out = layer_norm.permute(0, 2, 1)
            # 返回输出 out
            return out

        # 创建具有梯度的输入张量 x，维度为 4x2x4，设备为 cuda
        x = torch.randn(4, 2, 4, requires_grad=True, device="cuda")
        # 使用 DTensor 从本地创建 x，使用网格和分片 1，不运行检查
        x_dt = DTensor.from_local(x, mesh, [Shard(1)], run_check=False)

        # 创建具有梯度的输入张量 y，维度为 4，设备为 cuda
        y = torch.randn(4, requires_grad=True, device="cuda")
        # 使用 DTensor 从本地创建 y，使用网格和复制副本，不运行检查
        y_dt = DTensor.from_local(y, mesh, [Replicate()], run_check=False)

        # 创建具有梯度的输入张量 z，维度为 4，设备为 cuda
        z = torch.randn(4, requires_grad=True, device="cuda")
        # 使用 DTensor 从本地创建 z，使用网格和复制副本，不运行检查
        z_dt = DTensor.from_local(z, mesh, [Replicate()], run_check=False)

        # 编译函数 fn，使用 "inductor" 后端，完整图形模式
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        # 执行优化后的函数 opt_fn，传入 x_dt, y_dt, z_dt 作为参数
        tmp_dt = opt_fn(x_dt, y_dt, z_dt)
        # 对 tmp_dt 与 x_dt 进行矩阵乘法，并对结果进行维度置换
        out_dt = torch.matmul(tmp_dt, x_dt).permute(0, 2, 1)
        # 对 out_dt 的和进行反向传播
        out_dt.sum().backward()
    def test_dynamo_dtensor_from_local_redistribute(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x):
            # 创建 DTensor 并在函数内部运行 redistribute (allgather collective)
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            # 将结果重新分布到指定的设备网格，并返回到本地后加上 2
            return dt.redistribute(mesh, [Replicate()]).to_local() + 2

        x = torch.ones(1)
        ref = fn(x)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        def redistribute_kwargs_fn(x):
            # 创建 DTensor 并使用关键字参数进行重新分布
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return (
                dt.redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()
                + 2
            )

        x = torch.ones(1)
        ref = redistribute_kwargs_fn(x)
        opt_kwargs_fn = torch.compile(
            redistribute_kwargs_fn, backend=cnt, fullgraph=True
        )
        res = opt_kwargs_fn(x)
        self.assertEqual(res, ref)

    def test_dtensor_dynamo_device_mesh_attrs(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x_dt):
            # 检查设备网格类型，根据类型返回不同计算结果
            if x_dt.device_mesh.device_type == "cuda":
                return x_dt + 1
            else:
                return x_dt + 2

        x = torch.ones(4, 4)
        x_dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        ref = fn(x_dt)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x_dt)
        self.assertEqual(ref, res)

    def test_graph_input_is_async(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # 对输入张量执行链式正弦函数
            return x.sin().sin()

        opt_fn = torch.compile(fn, backend=aot_eager_graph, fullgraph=True)

        x = torch.randn(4, 4, requires_grad=True)
        x_dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        x2 = x_dt.redistribute(mesh, [Replicate()], async_op=True)
        x2 = x2.to_local()
        out = opt_fn(x2)
        # 重要部分：图中包含 wait_tensor()
        # 在运行时，图的输入是 AsyncCollectiveTensor，
        # 在图中需要发出 wait() 来同步
        self.assertExpectedInline(
            str(fw_graph_cell[0]).strip(),
            """\
def forward(self, primals_1):
    # 调用 C++ 函数库中的 wait_tensor 函数，处理 primals_1 并返回等待的张量
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(primals_1)
    # 计算 wait_tensor 中每个元素的正弦值
    sin = torch.ops.aten.sin.default(wait_tensor)
    # 对 sin 张量再次计算正弦值，并将原 sin 张量设为 None
    sin_1 = torch.ops.aten.sin.default(sin);  sin = None
    # 返回包含 sin_1、primals_1 和 wait_tensor 的列表
    return [sin_1, primals_1, wait_tensor]



@unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
def test_dtensor_partial_placement_graph_output(self):
    # 创建一个包含指定设备类型和全局设备索引的 DeviceMesh 对象
    mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

    def fn(x):
        return x + x

    # 生成一个形状为 (4, 4) 的随机张量 x，并标记其需要梯度
    x = torch.randn(4, 4, requires_grad=True)
    # 将本地张量 x 转换为 DTensor 对象，并在指定的 mesh 上使用 Partial 策略进行分布
    x_dt = DTensor.from_local(x, mesh, [Partial()], run_check=False)

    # 生成一个形状为 (4, 4) 的随机张量 y，并标记其需要梯度
    y = torch.randn(4, 4, requires_grad=True)
    # 将本地张量 y 转换为 DTensor 对象，并在指定的 mesh 上使用 Replicate 策略进行分布
    y_dt = DTensor.from_local(y, mesh, [Replicate()], run_check=False)

    # 使用 inductor 后端编译函数 fn，并生成优化后的函数对象 opt_fn
    opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
    # 将 x_dt 作为输入调用优化后的函数 opt_fn，并将结果保存在 tmp_dt 中
    tmp_dt = opt_fn(x_dt)
    # 计算 tmp_dt 和 y_dt 的矩阵乘积，结果保存在 out_dt 中
    out_dt = torch.matmul(tmp_dt, y_dt)
    # 对 out_dt 所有元素求和，并进行反向传播
    out_dt.sum().backward()



@unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
@skip_if_lt_x_gpu(1)
# TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
@patch.object(torch._inductor.config, "compile_threads", 1)
@patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    def test_tp_compile_comm_reordering(self):
        # 定义一个名为test_tp_compile_comm_reordering的测试方法
        class FakeAttention(nn.Module):
            # 定义一个虚拟的注意力模块
            def __init__(self):
                super().__init__()
                # 线性层 wq 用于查询
                self.wq = nn.Linear(16, 16)
                # 线性层 wk 用于键
                self.wk = nn.Linear(16, 16)
                # 线性层 wv 用于值
                self.wv = nn.Linear(16, 16)
                # 线性层 wo 用于输出
                self.wo = nn.Linear(16, 16)

            def forward(self, x):
                # 前向传播函数
                xq = self.wq(x)  # 查询向量
                xk = self.wk(x)  # 键向量
                xv = self.wv(x)  # 值向量
                # 假的注意力机制：简单地将查询、键和值相加
                xo = xq + xk + xv
                return self.wo(xo)  # 输出结果经过线性层处理后返回

        class FakeTransformerBlock(nn.Module):
            # 定义一个虚拟的变换器块
            def __init__(self):
                super().__init__()
                self.attn = FakeAttention()  # 使用虚拟的注意力模块

            def forward(self, x):
                return self.attn(x)  # 前向传播通过注意力模块处理输入数据

        class FakeTransformer(nn.Module):
            # 定义一个虚拟的变换器模型
            def __init__(self):
                super().__init__()
                self.block = FakeTransformerBlock()  # 使用虚拟的变换器块

            def forward(self, input):
                return self.block(input)  # 前向传播通过变换器块处理输入数据

        model = FakeTransformer().to(self.device_type)  # 创建并将模型移动到指定设备类型

        tp_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
        # 初始化一个具有2个 GPU 的设备网格，使用 "tp" 作为维度名称

        # 应用序列并行处理计划
        parallel_plan = {
            "attn": PrepareModuleInput(
                input_layouts=Shard(0), desired_input_layouts=Replicate()
            ),
            "attn.wq": ColwiseParallel(),
            "attn.wk": ColwiseParallel(),
            "attn.wv": ColwiseParallel(),
            "attn.wo": RowwiseParallel(output_layouts=Shard(0)),
        }

        # 并行化模型的指定模块和设备网格
        parallelize_module(
            module=model.block,
            device_mesh=tp_mesh,
            parallelize_plan=parallel_plan,
        )

        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        # 使用 "inductor" 作为后端计数编译次数
        compiled_model = torch.compile(model, backend=cnt, fullgraph=True)
        # 编译模型并获得编译后的模型对象，使用完整图形进行编译

        inp = torch.rand(20, 16).to(self.device_type)
        # 创建一个随机输入张量，形状为 (20, 16)，并将其移动到指定设备类型
        out = compiled_model(inp)  # 使用编译后的模型处理输入数据
        out.sum().backward()  # 对输出结果求和并反向传播梯度
        self.assertEqual(cnt.frame_count, 1)  # 断言编译计数器的帧数为1

        code = run_and_get_triton_code(compiled_model, inp)
        # 运行并获取 Triton 代码，用于后续检查

        FileCheck().check(
            "buf0 = torch.ops._c10d_functional.all_gather_into_tensor.default(primal"
        ).check("torch.ops._c10d_functional.wait_tensor.default(buf0").check(
            "extern_kernels.mm(buf0,"
        ).run(
            code
        )
        # 使用 FileCheck 对 Triton 代码进行检查，确保特定的字符串存在
# 定义一个测试类，用于测试分布式张量的编译端到端情况，继承自DTensorTestBase
@instantiate_parametrized_tests
class TestDTensorCompileE2E(DTensorTestBase):
    
    # 定义一个属性，返回测试的世界大小为4
    @property
    def world_size(self):
        return 4

    # 使用通信装饰器，参数化测试函数，测试张量编译的全图情况
    @with_comms
    @parametrize("is_seq_parallel", [True, False])
    def test_tp_compile_fullgraph(self, is_seq_parallel):
        # 创建设备网格对象，使用torch.arange生成长度为world_size的张量
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 创建SimpleModel模型对象，使用self.device_type作为设备类型
        model = SimpleModel(self.device_type)

        # 根据is_seq_parallel条件选择不同的列并行风格
        colwise_style = (
            ColwiseParallel(input_layouts=Shard(0))
            if is_seq_parallel
            else ColwiseParallel()
        )
        # 根据is_seq_parallel条件选择不同的行并行风格
        rowwise_style = (
            RowwiseParallel(output_layouts=Shard(0))
            if is_seq_parallel
            else RowwiseParallel()
        )

        if is_seq_parallel:
            # 如果is_seq_parallel为True，使用输入准备来测试其编译
            prepare_module_input = PrepareModuleInput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
            )
            prepare_module_out = PrepareModuleOutput(
                output_layouts=Replicate(),
                desired_output_layouts=Shard(0),
            )
            # 定义并行计划，包含不同模块和并行风格的映射关系
            plan = {
                "mlp_0": prepare_module_input,
                "mlp_0.net1": ColwiseParallel(),
                "mlp_0.net2": rowwise_style,
                "mlp_1.net1": colwise_style,
                "mlp_1.net2": RowwiseParallel(),
                "mlp_1": prepare_module_out,
            }
        else:
            # 如果is_seq_parallel为False，简化的并行计划，直接映射并行风格到各网络
            plan = {
                "mlp_0.net1": colwise_style,
                "mlp_0.net2": rowwise_style,
                "mlp_1.net1": colwise_style,
                "mlp_1.net2": rowwise_style,
            }

        # 将模型并行化处理，使用给定的mesh网格和计划
        model = parallelize_module(
            model,
            mesh,
            parallelize_plan=plan,
        )
        
        # 根据is_seq_parallel选择随机数生成种子
        rng_seed = self.rank if is_seq_parallel else 0
        # 设置随机数种子
        torch.manual_seed(rng_seed)
        # 创建输入张量inp，形状为(20, 10)，位于指定设备上
        inp = torch.rand(20, 10, device=self.device_type)
        # 使用模型进行推理
        out = model(inp)
        # 创建编译计数器对象，用于统计编译帧数，使用后端"aot_eager"
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        # 对模型进行全图编译，返回编译后的模型对象
        compiled_mod = torch.compile(model, backend=cnt, fullgraph=True)
        # 使用编译后的模型进行推理
        compiled_out = compiled_mod(inp)
        # 计算编译输出的梯度
        compiled_out.sum().backward()
        # 断言编译输出与未编译输出相等
        self.assertEqual(compiled_out, out)
        # 断言编译帧数为1
        self.assertEqual(cnt.frame_count, 1)

    # 使用通信装饰器，跳过GPU数目少于4的设备
    @with_comms
    @skip_if_lt_x_gpu(4)
    # 定义一个测试方法，用于测试二维分布数据并行处理的编译
    def test_2d_fsdp_tp_compile(self):
        # 设置数据并行大小为2
        data_parallel_size = 2
        # 创建一个简单模型实例，使用设备类型 self.device_type
        model = SimpleModel(self.device_type)
        # 深度复制模型，得到一个副本
        model_copy = copy.deepcopy(model)

        # 初始化设备网格为二维，格式为 [dp, tp]
        twod_mesh = init_device_mesh(
            "cuda",
            (data_parallel_size, self.world_size // data_parallel_size),
            mesh_dim_names=["dp", "tp"],
        )

        # 获取第一个维度为 dp 的设备组
        fsdp_pg = twod_mesh.get_group(mesh_dim=0)

        # 创建输入张量，形状为 (20, 10)，使用设备类型 self.device_type
        inp = torch.rand(20, 10, device=self.device_type)

        # 定义并行化计划，将指定的模块并行化处理
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        
        # 使用 parallelize_module 函数对模型进行并行化处理，针对第二个维度 tp
        tp_model = parallelize_module(model, twod_mesh["tp"], parallelize_plan)
        
        # 创建 FSDP 对象，用于支持分布式训练和稀疏梯度的优化处理
        eager_2d = FSDP(
            tp_model,
            device_id=self.rank,
            use_orig_params=True,
            device_mesh=twod_mesh["dp"],
        )
        
        # 对输入数据 inp 进行处理
        out = eager_2d(inp)

        # 再次使用 parallelize_module 函数对模型副本进行并行化处理，针对第二个维度 tp
        tp_model2 = parallelize_module(
            model_copy,
            twod_mesh["tp"],
            parallelize_plan,
        )
        
        # 创建第二个 FSDP 对象，用于支持分布式训练和稀疏梯度的优化处理
        fsdp_2d = FSDP(
            tp_model2,
            device_id=self.rank,
            use_orig_params=True,
            device_mesh=twod_mesh["dp"],
        )

        # TODO: 一旦 aot autograd 支持准备就绪，可以使用默认后端
        
        # 使用 CompileCounterWithBackend 类创建计数器 cnt，用于统计编译次数
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        
        # 对 fsdp_2d 模型进行编译，使用计数器 cnt 进行统计
        compiled_2d = torch.compile(fsdp_2d, backend=cnt)
        
        # 对编译后的模型进行推断处理，得到编译输出
        compiled_output = compiled_2d(inp)

        # 断言计算结果 out 与编译输出 compiled_output 相等
        self.assertEqual(out, compiled_output)
        
        # 断言编译帧计数 cnt.frame_count 等于 1
        self.assertEqual(cnt.frame_count, 1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    # 测试2D FSDP TP AC编译
    def test_2d_fsdp_tp_ac_compile(self):
        # 设置DP度数为2
        dp_degree = 2
        # 计算TP度数
        tp_degree = self.world_size // dp_degree
        # 创建SimpleModel对象
        model = SimpleModel(self.device_type)
        # 深度复制model对象
        model_copy = copy.deepcopy(model)

        # 初始化2D网格为[dp, tp]
        mesh_2d = init_device_mesh(
            "cuda", mesh_shape=(dp_degree, tp_degree), mesh_dim_names=("dp", "tp")
        )

        # 生成随机输入张量
        inp = torch.rand(20, 10, device=self.device_type)
        # 并行化计划
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        # 在TP维度上并行化模型
        tp_model = parallelize_module(model, mesh_2d["tp"], parallelize_plan)
        # 使用检查点包装器
        tp_model = checkpoint_wrapper(
            tp_model,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
        )
        # 创建2D FSDP对象
        eager_2d = FSDP(tp_model, device_mesh=mesh_2d["dp"], use_orig_params=True)

        # 在TP维度上并行化模型的深度复制
        tp_model2 = parallelize_module(model_copy, mesh_2d["tp"], parallelize_plan)
        # 创建2D FSDP对象
        fsdp_2d = FSDP(
            tp_model2,
            device_mesh=mesh_2d["dp"],
            use_orig_params=True,
        )
        # 编译2D FSDP对象
        compiled_2d = torch.compile(fsdp_2d, backend="aot_eager")

        # 前向传播
        out = eager_2d(inp)
        compiled_output = compiled_2d(inp)
        # 断言前向传播结果一致
        self.assertEqual(out, compiled_output)

        # 反向传播
        out.sum().backward()
        compiled_output.sum().backward()

        # 比较梯度
        for n, p in zip(fsdp_2d.parameters(), compiled_2d.parameters()):
            self.assertEqual(n.grad, p.grad)

    # 测试编译DTensor重新分配反向传播
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_compile_dtensor_redistribute_backward(self):
        # 创建设备网格
        mesh = DeviceMesh(device_type="cuda", mesh=torch.arange(self.world_size))

        def fn(x, y):
            # 从本地数据创建DTensor对象
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            # 矩阵相乘
            dt_out = torch.matmul(dt, dt2)
            # 重新分配DTensor对象
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out_redistribute.to_local()

        # 编译函数
        opt_fn = torch.compile(fn, backend=aot_eager_graph, fullgraph=True)

        # 创建输入张量
        x_ref = torch.arange(8, requires_grad=True, dtype=torch.float32)
        y_ref = torch.arange(8, requires_grad=True, dtype=torch.float32)
        ref = fn(x_ref, y_ref)

        x = torch.arange(8, requires_grad=True, dtype=torch.float32)
        y = torch.arange(8, requires_grad=True, dtype=torch.float32)
        res = opt_fn(x, y)

        # 断言结果一致
        self.assertEqual(res, ref)

        # 运行并断言反向传播和梯度
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(x_ref.grad, x.grad)
        self.assertEqual(y_ref.grad, y.grad)
# 如果当前脚本作为主程序执行（而非被导入为模块），则运行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```