# `.\pytorch\test\distributed\fsdp\test_fsdp_misc.py`

```
# Owner(s): ["oncall: distributed"]

import functools  # 导入 functools 模块，用于高阶函数、函数修饰器等功能
import os  # 导入 os 模块，提供了访问操作系统功能的接口
import sys  # 导入 sys 模块，提供了对 Python 解释器的访问
import warnings  # 导入 warnings 模块，用于管理警告信息的显示
from collections import namedtuple  # 导入 namedtuple 类型，创建具名元组对象
from contextlib import nullcontext  # 导入 nullcontext 上下文管理器，用于创建一个空的上下文
from copy import deepcopy  # 导入 deepcopy 函数，用于深拷贝对象
from itertools import chain  # 导入 chain 函数，用于串联多个可迭代对象
from typing import Any, Tuple  # 导入类型提示，声明函数参数和返回值的类型

import torch  # 导入 PyTorch 深度学习框架
import torch.distributed as dist  # 导入 PyTorch 分布式通信模块
import torch.distributed.fsdp._traversal_utils as traversal_utils  # 导入 FSDP 模块的辅助函数
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.distributed.fsdp import (  # 导入 FSDP 模块的相关类和函数
    CPUOffload,  # CPUOffload 类
    FlatParameter,  # FlatParameter 类
    FullyShardedDataParallel as FSDP,  # FullyShardedDataParallel 类，并用 FSDP 别名
    ShardingStrategy,  # ShardingStrategy 类
)
from torch.distributed.fsdp._flat_param import _FSDP_USE_UNSAFE_SETATTR  # 导入 FSDP 模块的特定变量
from torch.distributed.fsdp._runtime_utils import HOMOGENEOUS_ATTR_NAMES  # 导入 FSDP 运行时工具函数
from torch.distributed.fsdp.wrap import (  # 导入 FSDP 包装模块中的函数和策略
    always_wrap_policy,  # always_wrap_policy 函数
    ModuleWrapPolicy,  # ModuleWrapPolicy 类
    transformer_auto_wrap_policy,  # transformer_auto_wrap_policy 函数
)
from torch.distributed.optim import _apply_optimizer_in_backward  # 导入优化器相关函数
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer  # 导入 Transformer 编解码层类
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行类，并用 DDP 别名
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试相关函数
from torch.testing._internal.common_fsdp import (  # 导入 FSDP 模块的测试相关函数和类
    _assert_module_states,  # _assert_module_states 函数
    CUDAInitMode,  # CUDAInitMode 类
    FSDPInitMode,  # FSDPInitMode 类
    FSDPTest,  # FSDPTest 类
    FSDPTestMultiThread,  # FSDPTestMultiThread 类
    MLP,  # MLP 类
    NestedWrappedModule,  # NestedWrappedModule 类
    TransformerWithSharedParams,  # TransformerWithSharedParams 类
)
from torch.testing._internal.common_utils import (  # 导入通用测试工具函数
    instantiate_parametrized_tests,  # instantiate_parametrized_tests 函数
    parametrize,  # parametrize 装饰器
    run_tests,  # run_tests 函数
    TEST_WITH_DEV_DBG_ASAN,  # TEST_WITH_DEV_DBG_ASAN 变量
)

if not dist.is_available():  # 如果分布式通信不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 打印消息到标准错误流
    sys.exit(0)  # 退出程序

if TEST_WITH_DEV_DBG_ASAN:  # 如果测试标记为 dev-asan
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )  # 打印消息到标准错误流，说明为什么跳过 dev-asan 测试
    sys.exit(0)  # 退出程序


class MyModel(nn.Module):  # 定义自定义神经网络模型类 MyModel
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        self.a = nn.Linear(2, 2)  # 添加名为 a 的线性层
        self.b = nn.Linear(2, 2)  # 添加名为 b 的线性层

    def forward(self, x, y):  # 定义前向传播方法，接收输入 x 和 y
        return self.b(self.a(x + y))  # 返回 b 层对 a 层输入求和后的输出


class TestFSDPMiscMultiProcess(FSDPTest):  # 定义测试类 TestFSDPMiscMultiProcess，继承自 FSDPTest
    @property
    def world_size(self):  # 定义属性 world_size，返回进程组大小为 2
        return 2

    @property
    def process_group(self):  # 定义属性 process_group，返回默认的分布式进程组
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)  # 跳过 GPU 少于 2 个的情况
    @parametrize("use_index", [True, False])  # 参数化测试，测试 use_index 参数为 True 和 False 的情况
    # 定义测试方法，用于测试 FSDP 中的 `device_id` 参数
    def test_fsdp_device_id(self, use_index):
        """
        Tests the FSDP ``device_id`` argument:
          - Wrapping a CPU module should move the module to the GPU matching
          ``device_id``
          - Wrapping a GPU module already on the GPU matching ``device_id``
          should not raise an error
          - Wrapping a GPU module already on GPU and passing a GPU device
          without specifying a device ID (i.e. ``torch.device("cuda")``) warns
        """
        # 根据 use_index 决定是否使用当前 CUDA 设备作为 device_id
        dev_id = (
            torch.cuda.current_device()
            if use_index
            else torch.device("cuda", torch.cuda.current_device())
        )

        # 内部函数，检查模块中的 `FlatParameter` 是否与指定的 `device_id` 匹配
        def _check_device_matches(module, device_id):
            """Checks that the ``FlatParameter``s in ``module`` have device
            matching ``device_id``."""
            # 获取所有 `FlatParameter` 的设备，并确保至少有一个
            devices = {
                p.device for p in module.parameters() if isinstance(p, FlatParameter)
            }
            assert len(devices) > 0
            # 断言只有一个设备
            self.assertEqual(1, len(devices))
            found_device = devices.pop()
            # 根据 use_index 和 device_id 类型的不同，确定最终设备
            if use_index and not isinstance(device_id, torch.device):
                device = torch.device("cuda", device_id)
            else:
                device = device_id
            self.assertEqual(found_device, device)

        # 测试用例：将 FSDP 参数移动到 `device_id` 对应的 CPU 模块
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_NEVER,
            fsdp_kwargs={"device_id": dev_id},
        )
        # 检查参数是否正确移动到 `device_id`
        _check_device_matches(nested_wrapped_module, dev_id)

        # 测试用例：为已在指定 GPU 设备上的 GPU 模块指定 `device_id`，不应引发错误
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            fsdp_kwargs={"device_id": dev_id},
        )
        # 检查参数是否正确移动到 `device_id`
        _check_device_matches(nested_wrapped_module, dev_id)

        # 测试用例：为已在 GPU 上的 GPU 模块传递 `torch.device("cuda")`，应产生警告
        regex = "does not have an explicit index"
        context = self.assertWarnsRegex(
            expected_warning=UserWarning, expected_regex=regex
        )
        with context:
            nested_wrapped_module = NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_BEFORE,
                fsdp_kwargs={"device_id": torch.device("cuda")},
            )
        # 检查参数是否正确移动到当前 CUDA 设备对应的 `torch.device("cuda")`
        _check_device_matches(
            nested_wrapped_module, torch.device("cuda", torch.cuda.current_device())
        )

    @skip_if_lt_x_gpu(2)
    @skip_if_lt_x_gpu(2)
    @parametrize("use_second_layer", [True, False])
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD, None])
    # 定义一个测试函数，测试不同的 FSDP 配置下模型的行为，不涉及梯度计算
    def test_fsdp_module_no_compute_grad(self, use_second_layer, sharding_strategy):
        # 当 use_second_layer=True 时，b 在前向计算中被使用，但在反向传播中不接收梯度。
        # 否则，b 不参与前向计算。

        # 定义一个简单的神经网络模型
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(10, 10)  # 定义一个线性层 a
                self.b = nn.Linear(10, 10)  # 定义另一个线性层 b

            def forward(self, x, y):
                out1 = self.a(x)  # 计算线性层 a 的输出
                if use_second_layer:
                    out2 = self.b(y)  # 若 use_second_layer=True，则计算线性层 b 的输出
                    return out1, out2
                else:
                    return out1

        # 创建 FSDP 对象，包裹在 CUDA 上的 MyModel 实例，并指定分片策略和自动包裹策略
        fsdp = FSDP(
            MyModel().cuda(),
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=always_wrap_policy,
        )
        # 在 CUDA 上生成随机张量 x 和 y
        x = torch.randn(10, 10, device="cuda")
        y = torch.randn(10, 10, device="cuda")
        for i in range(4):
            if use_second_layer:
                a, b = fsdp(x, y)  # 若 use_second_layer=True，则返回 a 和 b
            else:
                a = fsdp(x, y)  # 否则，只返回 a
            loss = a.sum()  # 计算输出 a 的和作为损失
            loss.backward()  # 反向传播计算梯度

            # 检查梯度是否正确传播：self.a 应接收梯度，self.b 不应接收梯度
            a_grad = fsdp.module.a._handle.flat_param.grad
            b_grad = fsdp.module.b._handle.flat_param.grad
            self.assertIsNotNone(a_grad)  # 断言确保 self.a 的梯度不为 None
            self.assertIsNone(b_grad)     # 断言确保 self.b 的梯度为 None

    # 当 GPU 数量小于 2 时跳过该测试
    @skip_if_lt_x_gpu(2)
    def test_fsdp_not_all_outputs_used_in_loss(self):
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,    # 全部分片策略
                    ShardingStrategy.SHARD_GRAD_OP,  # 梯度操作分片策略
                    ShardingStrategy.NO_SHARD,       # 无分片策略
                ]
            },
            self._test_fsdp_not_all_outputs_used_in_loss,
        )

    # 测试 FSDP 在损失函数中未使用全部输出时的行为
    def _test_fsdp_not_all_outputs_used_in_loss(
        self, sharding_strategy: ShardingStrategy
        ):
            class MyModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lin1 = nn.Linear(4, 4)
                    self.lin2 = nn.Linear(4, 4)

                def forward(self, x):
                    # 计算第一层线性变换结果
                    a = self.lin1(x)
                    # 计算第二层线性变换结果
                    b = self.lin2(x)
                    return (a, b)

            def _check_resharded(fsdp_module):
                # 获取 FSDP 模块的句柄
                handle = fsdp_module._handle
                if not handle:
                    return
                # 获取平坦化的参数
                param = handle.flat_param
                # 如果使用了分片策略
                if handle.uses_sharded_strategy:
                    # 获取完整参数，并检查其存储大小为零
                    full_param = param._full_param_padded
                    self.assertEqual(full_param.storage().size(), 0)

                # 检查参数的数据指针是否与本地分片的数据指针相同
                self.assertEqual(param.data_ptr(), param._local_shard.data_ptr())

            def _check_equal(local, fsdp):
                # 使用 FSDP 召唤完整参数
                with FSDP.summon_full_params(fsdp):
                    # 遍历比较 FSDP 和本地模块的参数
                    for p1, p2 in zip(fsdp.parameters(), local.parameters()):
                        torch.testing.assert_close(p1, p2)

            # 使用 functools.partial 创建 FSDP 构造函数
            fsdp_ctor = functools.partial(FSDP, sharding_strategy=sharding_strategy)
            # 创建并将 MyModule 模块移到 CUDA 设备上
            m = MyModule().cuda()
            # 创建 MyModule 的深拷贝
            m_local = deepcopy(m)
            # 将本地模块设置为 MyModule
            local_m = m_local
            # 复制本地模块的参数
            prev_params = [p.clone() for p in m_local.parameters()]

            # 对第一层线性变换应用 FSDP
            m.lin1 = fsdp_ctor(m.lin1)
            # 对整个模块应用 FSDP
            m = fsdp_ctor(m)
            # 检查模块 m 和 m_local 是否相等
            _check_equal(m_local, m)

            # 使用 SGD 优化器优化 m 的参数
            opt = torch.optim.SGD(m.parameters(), lr=1e-3)
            # 使用 SGD 优化器优化 local_m 的参数
            opt_local = torch.optim.SGD(local_m.parameters(), lr=1e-3)

            # 进行 6 次迭代
            for i in range(6):
                t = torch.ones(4, device="cuda")
                # 在模块 m 上执行前向传播
                a, b = m(t)
                # 在本地模块上执行前向传播
                local_a, local_b = local_m(t)
                if i < 2:
                    # 在损失计算中使用两个参数。稍后，b 将不再使用，我们检查梯度是否与本地训练相同
                    loss = (a @ b).sum()
                    loss_local = (local_a @ local_b).sum()
                else:
                    # 仅在损失计算中使用第一个参数
                    loss = a.sum()
                    loss_local = local_a.sum()

                # 计算损失的反向传播
                loss.backward()
                loss_local.backward()
                # 检查并重整化模块 m
                _check_resharded(m)
                # 在优化器上执行一步优化
                opt.step()
                opt_local.step()
                # 检查模块 m 和 m_local 是否相等
                _check_equal(m_local, m)
                # 确保至少有些变化从先前的参数中产生，否则上面的检查会空洞地为真
                self.assertTrue(
                    any(
                        not torch.equal(p1, p2)
                        for p1, p2 in zip(prev_params, m_local.parameters())
                    )
                )
                # 更新先前参数为本地模块的深拷贝
                prev_params = [p.clone() for p in local_m.parameters()]
                # 清空优化器的梯度
                opt.zero_grad()
                opt_local.zero_grad()

            # 使用分布式通信库进行同步
            dist.barrier()

        @skip_if_lt_x_gpu(2)
    def test_fsdp_optim_overlap_no_use_orig_params_error(self):
        # 创建一个使用FSDP的实例fsdp_overlap，用于测试重叠优化器错误情况
        fsdp_overlap = FSDP(
            MyModel().cuda(),
            auto_wrap_policy=always_wrap_policy,
            use_orig_params=False,
        )
        # 指定优化器类为SGD
        optim_cls = torch.optim.SGD
        # 定义优化器的参数字典
        optim_kwargs = {"lr": 0.03}
        # 调用_apply_optimizer_in_backward函数，将SGD优化器应用到fsdp_overlap的参数上
        _apply_optimizer_in_backward(
            optimizer_class=optim_cls,
            params=fsdp_overlap.parameters(),
            optimizer_kwargs=optim_kwargs,
            register_hook=False,
        )

        # 生成一个CUDA上的随机输入张量
        inp = torch.randn(10, 10, device="cuda")
        # 断言捕获RuntimeError异常，其消息为"only supported with use_orig_params=True"
        with self.assertRaisesRegex(
            RuntimeError, "only supported with use_orig_params=True"
        ):
            # 调用fsdp_overlap模型两次，预期抛出异常
            fsdp_overlap(inp, inp)

    @skip_if_lt_x_gpu(2)
    @skip_if_lt_x_gpu(2)
    def test_fsdp_cpu_training(self):
        """Tests FSDP training on CPU."""
        # 创建一个使用Gloo后端的新进程组gloo_pg
        gloo_pg = dist.new_group(backend="gloo")
        # 遍历不同的分片策略
        for ss in [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
            ShardingStrategy.HYBRID_SHARD,
            ShardingStrategy._HYBRID_SHARD_ZERO2,
        ]:
            # 设定随机种子
            torch.manual_seed(42)
            # 创建一个MyModel模型实例
            model = MyModel()
            # 使用DDP对模型进行深拷贝，并设定使用gloo_pg进程组
            ref_model = DDP(deepcopy(model), process_group=gloo_pg)
            # 使用FSDP对模型进行包装，设定自动包装策略和cpu设备
            model = FSDP(
                model,
                auto_wrap_policy=always_wrap_policy,
                process_group=gloo_pg,
                device_id=torch.device("cpu"),
            )
            # 创建Adam优化器实例，用于参考模型
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
            # 创建Adam优化器实例，用于FSDP模型
            optim = torch.optim.Adam(model.parameters(), lr=1e-2)
            # 设定随机种子，加上rank属性
            torch.manual_seed(42 + self.rank)
            # 创建一个2x2的随机输入张量
            inp = torch.randn(2, 2)
            # 进行10次迭代训练
            for _ in range(10):
                losses = []
                # 对参考模型和FSDP模型分别进行优化
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    # 计算模型输出的损失，并求和
                    loss = _model(inp, inp).sum()
                    losses.append(loss)
                    # 反向传播求梯度
                    loss.backward()
                    # 执行优化步骤
                    _optim.step()
                    # 清空梯度
                    _optim.zero_grad()
                # 断言两个模型的损失值相等
                self.assertEqual(losses[0], losses[1])
    def test_fsdp_cpu_init_stays_on_cpu(self):
        # 将此测试移至 MT 测试，一旦警告日志和向后集体问题解决。
        """Tests that passing a CPU module to FSDP preserves that the wrapped
        module is on CPU after FSDP initialization, albeit after logging a
        warning, and that FSDP moves CPU input to GPU before the forward."""
        # 设置当前 CUDA 设备为 self.rank
        torch.cuda.set_device(self.rank)
        # 定义预期的警告消息正则表达式
        regex = "passed-in `module` is on CPU"
        # 断言将会产生特定的警告消息
        context = self.assertWarnsRegex(
            expected_warning=UserWarning, expected_regex=regex
        )
        # 在警告上下文中执行以下代码块
        with context:
            # 初始化 NestedWrappedModule
            nested_wrapped_module = NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_NEVER,
            )
            # 使用 nested_wrapped_module 和 process_group 创建 FSDP 模型
            fsdp_model = FSDP(nested_wrapped_module, self.process_group)
        # 收集模型参数所在的设备集合
        devices = {p.device for p in fsdp_model.parameters()}
        # 断言模型参数只存在于一个设备上
        self.assertEqual(1, len(devices))
        # 断言模型参数在 CPU 设备上
        self.assertEqual(torch.device("cpu"), devices.pop())
        # 将 fsdp_model 移动到 CUDA 设备
        fsdp_model = fsdp_model.cuda()
        # 确保在移动到 CUDA 设备后可以进行前向和后向传播
        # CPU 输入也测试确保输入正确移动到适当的 CUDA 设备
        inp = fsdp_model.module.get_input(device=torch.device("cpu"))
        fsdp_model(*inp).sum().backward()

    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于测试在同步模块状态时初始化 CPU 模块的行为
    def test_cpu_init_with_sync_module_states(self):
        """
        Tests that passing ``sync_module_states=True`` raises an error for
        a CPU module since the synchronization requires GPU communication,
        while additionally passing ``device_id`` does not raise an error, even
        when the model has CPU buffers.
        """

        # 定义一个内部函数，用于初始化嵌套的 WrappedModule
        def init_nested_wrapped_module():
            return NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_NEVER,
            )

        # 断言：当 `sync_module_states=True` 时，如果模块含有 CPU 参数或缓冲区，应该抛出 ValueError 异常
        with self.assertRaisesRegex(
            ValueError,
            "The module has CPU parameters or buffers when `sync_module_states=True`",
        ):
            # 初始化 FSDP 对象时传入参数，期望抛出上述异常
            FSDP(
                init_nested_wrapped_module(),
                self.process_group,
                sync_module_states=True,
            )

        # 验证：即使模型含有 CPU 缓冲区，使用 `sync_module_states=True` 和 `device_id` 也不应该抛出异常
        nested_wrapped_module = init_nested_wrapped_module()
        nested_wrapped_module.register_buffer(
            "buf", torch.ones((2, 2), device="cpu") * self.rank
        )
        nested_wrapped_module.module[0].register_buffer(
            "buf", torch.ones((3, 2), device="cpu") * self.rank
        )
        nested_wrapped_module = FSDP(
            nested_wrapped_module,
            self.process_group,
            auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
        )
        # 验证：每个 rank 的缓冲区应为全零，因为 rank 0 是源头，并且缓冲区应位于 GPU 上，因为我们指定了 `device_id`
        self.assertEqual(
            nested_wrapped_module.buf.device,
            torch.device("cuda", torch.cuda.current_device()),
        )
        self.assertEqual(nested_wrapped_module.buf, torch.zeros((2, 2)))
        self.assertEqual(
            nested_wrapped_module.module.module[0].buf.device,
            torch.device("cuda", torch.cuda.current_device()),
        )
        self.assertEqual(
            nested_wrapped_module.module.module[0].buf, torch.zeros((3, 2))
        )
class TestFSDPMiscMultiThread(FSDPTestMultiThread):
    @property
    def world_size(self):
        # 返回并行运行时的进程数量为2
        return 2

    @property
    def process_group(self):
        # 返回默认的分布式进程组
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_namedtuple(self):
        # 定义一个简单的神经网络模块，包含一个线性层
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(100, 100)

            def forward(self, x):
                return x

        # 将神经网络模块移动到 GPU 上
        m = MyModule().cuda()
        # 使用FSDP对模块进行包装
        m = FSDP(m)
        # 创建一个张量并移动到 GPU 上，设置需要梯度计算
        t = torch.ones(1, device="cuda", requires_grad=True)

        # 定义一个命名元组类型
        MyOutputType = namedtuple(
            "MyOutputType", ["a", "b", "c", "d"], defaults=(t, t, t, t)
        )

        inp = MyOutputType()
        # 对模块进行前向传播
        out = m(inp)
        # 确保注册了钩子函数
        for x in out:
            self.assertNotEqual([], list(x._backward_hooks.values()))

        # TODO: 我们应该检查 backward() 和参数是否被重新分片，
        # 但由于以下问题而被阻塞：
        # https://github.com/pytorch/pytorch/issues/83107 和
        # https://github.com/pytorch/pytorch/issues/83129

    @skip_if_lt_x_gpu(2)
    def test_device_id_auto_wrap(self):
        """测试 ``auto_wrap_policy`` 是否将 ``device_id`` 传播到所有嵌套的 FSDP 实例中。"""
        self.run_subtests(
            {"use_callable": [False, True]},
            self._test_device_id_auto_wrap,
        )

    def _test_device_id_auto_wrap(self, use_callable: bool):
        # 定义模块类的集合
        module_classes = {TransformerEncoderLayer, TransformerDecoderLayer}
        if use_callable:
            # 如果使用可调用方式，则设置自动包装策略
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=module_classes,
            )
        else:
            # 否则，使用默认的模块包装策略
            auto_wrap_policy = ModuleWrapPolicy(module_classes)
        # 设置 FSDP 的参数
        fsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "device_id": torch.cuda.current_device(),
        }
        # 初始化带有共享参数的 Transformer 模型
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            fsdp_kwargs,
        )
        # 对于 FSDP 模块的每一个实例，检查计算设备是否正确设置
        for fsdp_module in FSDP.fsdp_modules(fsdp_model):
            self.assertEqual(
                fsdp_module.compute_device,
                torch.device("cuda", torch.cuda.current_device()),
            )

    @skip_if_lt_x_gpu(2)
    def test_fsdp_device_id_cpu_offload(self):
        """
        测试在指定 ``device_id`` 和参数 CPU 卸载时的 FSDP 行为。
        """
        self.run_subtests(
            {"use_orig_params": [False, True]},
            self._test_fsdp_device_id_cpu_offload,
        )
    def _test_fsdp_device_id_cpu_offload(self, use_orig_params: bool):
        # 定义一个内部的神经网络模型类 MyModel
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个包含两个线性层的序列模块
                self.seq = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.Linear(10, 10),
                )
                # 定义一个单独的线性层
                self.lin = nn.Linear(10, 10)

            # 前向传播函数，对输入 x 应用 seq 和 lin 层
            def forward(self, x):
                return self.lin(self.seq(x))

        # 创建 MyModel 实例
        model = MyModel()
        
        # 定义一个自动封装策略，以确保存在嵌套的 FSDP 实例并且父级 FSDP 实例管理参数
        auto_wrap_policy = ModuleWrapPolicy({nn.Sequential})
        
        # 使用 FSDP 封装模型，设置 CPU 降载模式，设备 ID 为当前 CUDA 设备，使用指定的参数设置
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            device_id=torch.cuda.current_device(),
            use_orig_params=use_orig_params,
        )
        
        # 定义一个 CPU 设备对象
        cpu_device = torch.device("cpu")
        
        # 对于 FSDP 模型的所有句柄，验证其扁平参数的设备是否为 CPU
        for handle in traversal_utils._get_fsdp_handles(fsdp_model):
            self.assertEqual(handle.flat_param.device, cpu_device)

    @skip_if_lt_x_gpu(2)
    def test_module_device_mismatches_device_id(self):
        """Tests that specifying a ``device_id`` argument to FSDP for a GPU
        module that does not match the GPU device ID raises an error."""
        # 设置当前 CUDA 设备为指定的排名 self.rank
        torch.cuda.set_device(self.rank)
        
        # 根据排名 self.rank 判断是否抛出 ValueError 异常，显示设备 ID 不匹配
        context = (
            self.assertRaisesRegex(ValueError, f"cuda:{self.rank} vs cuda:0")
            if self.rank != 0
            else nullcontext()
        )
        
        # 在上下文中执行代码块，用于初始化 NestedWrappedModule
        with context:
            NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                # 将包装模块移动到 CUDA 设备之前，然后再用 FSDP 进行包装
                cuda_init_mode=CUDAInitMode.CUDA_BEFORE,
                # 因为模型位于 cuda:1，所以给定 device_id=0 应该引发错误
                fsdp_kwargs={"device_id": 0},
            )

    @skip_if_lt_x_gpu(2)
    def test_cpu_gpu_module(self):
        """Tests a CPU + GPU module supported if device_id is passed
        in, errors if device_id is not.
        """
        # 设置当前 CUDA 设备为指定的排名 self.rank
        torch.cuda.set_device(self.rank)

        # 定义一个包含 CPU 和 GPU 的模块 CPUGPUModule
        class CPUGPUModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 在 GPU 上创建一个线性层 a
                self.a = nn.Linear(1, 1).cuda()
                # 在 CPU 上创建一个线性层 b
                self.b = nn.Linear(1, 1)

        # 创建 CPUGPUModule 实例
        cpu_gpu = CPUGPUModule()
        
        # 使用 FSDP 封装 CPU + GPU 模块，设备 ID 为当前 CUDA 设备
        fsdp = FSDP(cpu_gpu, device_id=torch.cuda.current_device())
        
        # 验证所有参数的设备是否为当前 CUDA 设备
        for param in fsdp.parameters():
            self.assertEqual(param.device, torch.device(torch.cuda.current_device()))

        # 当不指定 device_id 时，应该抛出 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "please pass in device_id"):
            FSDP(CPUGPUModule())

    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于测试忽略模块元数据的行为
    def test_fsdp_ignored_module_meta(self):
        # 设置当前 CUDA 设备为指定的设备编号
        torch.cuda.set_device(self.rank)

        # 定义一个继承自 nn.Module 的类 CPUGPUModule
        class CPUGPUModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 在模块中定义两个线性层
                self.a = nn.Linear(1, 1)
                self.b = nn.Linear(1, 1)

        # 使用 "meta" 设备环境创建 CPUGPUModule 实例 m
        with torch.device("meta"):
            m = CPUGPUModule()
        
        # 使用 FSDP 对象包装模块 m，设置设备 ID，并指定忽略的模块 [m.a]
        m = FSDP(m, device_id=self.rank, ignored_modules=[m.a], use_orig_params=True)
        
        # 获取第一个线性层 a 的参数设备，并断言其设备为 "meta"
        meta_device = torch.device("meta")
        self.assertEqual(meta_device, next(m.a.parameters()).device)

        # 使用 "meta" 设备环境再次创建 CPUGPUModule 实例 m
        with torch.device("meta"):
            m = CPUGPUModule()
        
        # 使用 FSDP 对象包装模块 m，设置当前 CUDA 设备 ID，并指定忽略的模块 [m.a]
        # 同时使用 param_init_fn 初始化函数来配置参数
        m = FSDP(
            m,
            device_id=torch.cuda.current_device(),
            ignored_modules=[m.a],
            use_orig_params=True,
            param_init_fn=lambda m: m.to_empty(
                device=torch.cuda.current_device(), recurse=False
            ),
        )
        
        # 获取第一个线性层 a 的参数设备，并断言其设备为 "meta"
        self.assertEqual(meta_device, next(m.a.parameters()).device)

    # 如果当前 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_fsdp_device_id_no_move_ignored_params_and_bufs(self):
        # 定义一个继承自 nn.Module 的类 CPUGPUModule
        class CPUGPUModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 在模块中定义两个线性层 a 和 b
                self.a = nn.Linear(1, 1)
                self.b = nn.Linear(1, 1)
                # 向模块的线性层 a 注册一个名为 "buf" 的缓冲区，并初始化为全 1
                self.a.register_buffer("buf", torch.ones(1))

        # 创建 CPUGPUModule 实例 m
        m = CPUGPUModule()
        
        # 使用 FSDP 对象包装模块 m，设置设备 ID，并指定忽略的模块 [m.a]
        m = FSDP(m, device_id=self.rank, ignored_modules=[m.a], use_orig_params=True)
        
        # 获取被忽略的模块 m.a 的参数和缓冲区，并断言它们的设备为 "cpu"
        ignored_params = m.a.parameters()
        ignored_bufs = m.a.buffers()
        for t in chain(ignored_params, ignored_bufs):
            self.assertEqual(torch.device("cpu"), t.device)

    # 如果当前 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_multigpu_module(self):
        """
        Module on multiple GPUs wrapped in FSDP should raise an error.
        """
        
        # 定义一个多 GPU 环境下的模块 MultiGPUModule
        class MultiGPUModule(nn.Module):
            def __init__(self, rank):
                super().__init__()
                self.rank = rank
                # 在模块中定义两个线性层 a 和 b，分别放置在不同的 GPU 设备上
                self.a = nn.Linear(1, 1).cuda(self.rank)
                self.b = nn.Linear(1, 1).cuda((self.rank + 1) % dist.get_world_size())

        # 断言在 FSDP 中包装多 GPU 设备模块会引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, "FSDP only supports single device modules"
        ):
            FSDP(MultiGPUModule(self.rank))
    def test_no_params(self):
        """
        Test that device_id and cpu init work if module has no params
        (they are effective noops, but ensure FSDP does not assume module
        has parameters during init)
        """
        # TODO: override FSDP MT Thread _run to set this instead of here for
        # every test.
        
        # 设置当前 CUDA 设备为 self.rank
        torch.cuda.set_device(self.rank)
        
        # 测试在 CPU 上初始化
        no_params = nn.ReLU()
        module = FSDP(no_params)
        
        # 测试在 CUDA 上初始化
        no_params = nn.ReLU().cuda()
        module = FSDP(no_params)
        
        # 测试在 CPU + device_id 上初始化
        no_params = nn.ReLU()
        module = FSDP(no_params, device_id=torch.cuda.current_device())
        
        # 对于没有参数的模块，在错误的 device_id 下会引发错误，报告计算设备与 device_id 不一致
        no_params = nn.ReLU().cuda()
        context = (
            (
                self.assertRaisesRegex(
                    ValueError, f"Inconsistent.*cuda:{self.rank} vs cuda:0"
                )
            )
            if self.rank != 0
            else nullcontext()
        )
        with context:
            FSDP(no_params, device_id=0)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_same_model_across_ranks(self):
        """
        FSDP broadcasts model from rank 0 to ensure it starts off with the same
        values.
        """

        class MyModel(nn.Module):
            def __init__(self, rank):
                super().__init__()
                # 使用 rank 来设置随机种子，以确保不同的 rank 有不同的模型
                torch.manual_seed(rank)
                torch.cuda.manual_seed(rank)
                self.lin = nn.Linear(10, 10, bias=False)
                self.register_buffer("buffer", torch.ones(1) * rank)

        # 创建 MyModel 的实例，并将其移至 GPU 上
        m = MyModel(self.rank).cuda()
        
        # 使用 _assert_module_states 函数验证模型状态，确保不同于初始状态
        _assert_module_states(
            m, process_group=self.process_group, assert_fn=self.assertNotEqual
        )
        
        # 将 sync_module_states 参数传递给 FSDP，使模型在初始化期间保持一致
        fsdp = FSDP(m, sync_module_states=True)
        
        # 使用 fsdp.summon_full_params 方法进入全参数模式，并再次验证模型状态，确保与初始状态相同
        with fsdp.summon_full_params(fsdp):
            _assert_module_states(
                fsdp, process_group=self.process_group, assert_fn=self.assertEqual
            )

        # sync_module_states 参数同样适用于带有指定 device_id 的 CPU 模块
        m = MyModel(self.rank)
        
        # 使用 _assert_module_states 函数验证模型状态，确保不同于初始状态
        _assert_module_states(
            m, process_group=self.process_group, assert_fn=self.assertNotEqual
        )
        
        # 将 sync_module_states 参数传递给 FSDP，使模型在初始化期间保持一致
        fsdp = FSDP(m, device_id=torch.cuda.current_device(), sync_module_states=True)
        
        # 使用 fsdp.summon_full_params 方法进入全参数模式，并再次验证模型状态，确保与初始状态相同
        with fsdp.summon_full_params(fsdp):
            _assert_module_states(
                fsdp, process_group=self.process_group, assert_fn=self.assertEqual
            )
    def test_homogeneous_attributes(self):
        """
        Tests that passing heterogeneous values for attributes designated as
        homogeneous raises an error.
        """
        # 手动构造属性名和值的列表，并与全局的同质属性名列表进行验证
        all_attr_name_and_values = [
            ("_use_orig_params", False, True),
            ("limit_all_gathers", False, True),
            ("_use_full_prec_in_eval", False, True),
        ]
        # 断言列表推导式，验证属性名列表和全局同质属性名列表相等
        self.assertEqual(
            [
                attr_name_and_values[0]
                for attr_name_and_values in all_attr_name_and_values
            ],
            HOMOGENEOUS_ATTR_NAMES,
        )

        # 运行子测试，传入属性名和值的列表及测试函数
        self.run_subtests(
            {"attr_name_and_values": all_attr_name_and_values},
            self._test_homogeneous_attributes,
        )

    def _test_homogeneous_attributes(self, attr_name_and_values: Tuple[str, Any, Any]):
        # 使用给定的参数初始化 NestedWrappedModule 对象
        model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            {},
        )
        # 获取属性名
        attr_name = attr_name_and_values[0]

        # 根据属性名进行条件判断和处理
        if "_use_full_prec_in_eval" == attr_name:
            # 修改模型的第二个模块并设置环境变量
            model.module[1] = FSDP(model.module[1])
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = "1"
            fsdp_model = FSDP(model)
        else:
            # 准备用于内外层 FSDP 初始化的参数字典
            fsdp_kwargs_inner = {attr_name.lstrip("_"): attr_name_and_values[1]}
            fsdp_kwargs_outer = {attr_name.lstrip("_"): attr_name_and_values[2]}
            # 修改模型的第二个模块，并使用指定参数初始化 FSDP 对象
            model.module[1] = FSDP(model.module[1], **fsdp_kwargs_inner)
            fsdp_model = FSDP(model, **fsdp_kwargs_outer)

        # 运行前向传播以触发惰性初始化和错误
        with self.assertRaisesRegex(
            ValueError, f"Expects one homogeneous value for {attr_name}"
        ):
            inp = fsdp_model.module.get_input(torch.device("cuda"))
            fsdp_model(*inp)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_unsupported_module_cls(self):
        # 定义用于匹配的正则表达式
        regex = r"FSDP will not all-gather parameters for containers that do not implement forward"
        
        # 创建包含多个 MLP 模块的 ModuleList，并验证警告信息
        model = nn.ModuleList([MLP(8, torch.device("cpu")) for _ in range(3)])
        with self.assertWarnsRegex(UserWarning, regex):
            FSDP(model, device_id="cuda")
        
        # 创建包含多个 MLP 模块的 ModuleDict，并验证警告信息
        model = nn.ModuleDict(
            {"1": MLP(8, torch.device("cpu")), "2": MLP(8, torch.device("cpu"))}
        )
        with self.assertWarnsRegex(UserWarning, regex):
            FSDP(model)
class TestFSDPMiscWorldSize1(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        # 返回当前测试环境的进程数，这里固定为1
        return 1

    @skip_if_lt_x_gpu(1)
    def test_world_size_1_sharding_strategy_warning(self):
        """
        Tests that FSDP issues a warning when it switches to using ``NO_SHARD``
        when the world size is 1.
        """
        warning_prefix = "FSDP is switching to use `NO_SHARD` instead of"
        
        # 如果用户已经指定了 `NO_SHARD`，则不应该发出警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # 触发所有警告
            # 创建一个使用 FSDP 的线性层，并指定 `NO_SHARD` 策略
            FSDP(nn.Linear(3, 3).cuda(), sharding_strategy=ShardingStrategy.NO_SHARD)
            for warning in w:
                self.assertTrue(
                    warning.category != UserWarning
                    or not str(warning.message).startswith(warning_prefix)
                )

        # 检查是否发出了警告
        warning_suffix = " since the world size is 1."
        
        # - 当传递 `FULL_SHARD` 或 `None` 时
        expected_regex_full_shard = (
            warning_prefix + " " + str(ShardingStrategy.FULL_SHARD) + warning_suffix
        )
        with self.assertWarnsRegex(UserWarning, expected_regex_full_shard):
            # 创建一个使用 FSDP 的线性层，并指定 `FULL_SHARD` 策略
            FSDP(nn.Linear(3, 3).cuda(), sharding_strategy=ShardingStrategy.FULL_SHARD)
        with self.assertWarnsRegex(UserWarning, expected_regex_full_shard):
            # 创建一个使用 FSDP 的线性层，默认使用 `FULL_SHARD` 策略
            FSDP(nn.Linear(3, 3).cuda())
        
        # - 当传递 `SHARD_GRAD_OP` 时
        expected_regex_shard_grad_op = (
            warning_prefix + " " + str(ShardingStrategy.SHARD_GRAD_OP) + warning_suffix
        )
        with self.assertWarnsRegex(UserWarning, expected_regex_shard_grad_op):
            # 创建一个使用 FSDP 的线性层，并指定 `SHARD_GRAD_OP` 策略
            FSDP(
                nn.Linear(3, 3).cuda(), sharding_strategy=ShardingStrategy.SHARD_GRAD_OP
            )
    
    @skip_if_lt_x_gpu(1)
    def test_training_device_mismatch_errors(self):
        """
        Tests that, when training starts, if FSDP parameters are not on the
        expected device, then an informative error is raised. This applies for
        both no parameter CPU offloading and parameter CPU offloading.
        """
        # 测试在训练开始时，如果 FSDP 参数不在预期设备上，则引发详细的错误信息。
        
        # 情况一：错误地没有从 CPU -> GPU 迁移
        model = torch.nn.Linear(10, 10)
        fsdp_model = FSDP(model)
        inp = torch.randn((2, 10))
        with self.assertRaisesRegex(
            RuntimeError,
            "An FSDP-managed module unexpectedly has parameters on cpu. Make "
            "sure to move the module to cuda:0 before training.",
        ):
            fsdp_model(inp)

        # 情况二：错误地从 CPU -> GPU 迁移
        model = torch.nn.Linear(10, 10)
        fsdp_model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
        fsdp_model.to(torch.device("cuda"))
        inp = torch.randn((2, 10))
        with self.assertRaisesRegex(
            RuntimeError,
            "An FSDP-managed module with parameter CPU offloading enabled has "
            "parameters on cuda:0. Make sure to not move the module from CPU "
            "when offloading parameters.",
        ):
            fsdp_model(inp)

    @skip_if_lt_x_gpu(2)
    def test_unsafe_setattr(self):
        """
        Tests that the environment variable for using unsafe setattr gates as
        expected.
        """
        # 测试环境变量用于按预期设置不安全的 setattr 门。

        self.run_subtests(
            {"use_orig_params": [False, True]},
            self._test_unsafe_setattr,
        )
    # 定义一个测试方法 `_test_unsafe_setattr`，接受一个布尔类型参数 `use_orig_params`，用于测试 `setattr` 方法的安全性设置

        # 初始化一个标志位 `called_setattr_override`，用于跟踪是否调用了 `__setattr__` 方法

        # 定义一个内部类 `SetattrLinear`，继承自 `nn.Module`，用于创建一个线性层模型
        class SetattrLinear(nn.Module):
            # 构造方法，接受输入维度 `in_dim`，输出维度 `out_dim`，设备 `device`
            def __init__(self, in_dim: int, out_dim: int, device: torch.device) -> None:
                super().__init__()
                # 创建一个参数 `weight`，使用正态分布随机初始化，维度为 `(in_dim, out_dim)`，在指定设备上
                self.weight = nn.Parameter(
                    torch.randn((in_dim, out_dim), device=device)
                )

            # 前向传播方法，接受输入张量 `x`，返回 `x` 与 `weight` 矩阵相乘的结果
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weight

            # 自定义 `__setattr__` 方法，重写父类方法，在设置属性时调用
            def __setattr__(self, name: str, value: Any) -> None:
                nonlocal called_setattr_override
                # 标记 `called_setattr_override` 为 `True`，表示调用了 `__setattr__` 方法
                called_setattr_override = True
                # 调用父类的 `__setattr__` 方法，实际设置属性
                return super().__setattr__(name, value)

        # 创建一个 `SetattrLinear` 类的实例 `module`，输入参数为 (5, 5, cuda)
        module = SetattrLinear(5, 5, torch.device("cuda"))
        
        # 使用 `FSDP` 包装 `module`，设置 `use_orig_params` 参数为 `use_orig_params`
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        
        # 创建一个随机张量 `inp`，维度为 (8, 5)，在 cuda 设备上
        inp = torch.randn((8, 5), device=torch.device("cuda"))
        
        # 重置 `called_setattr_override` 标志位为 `False`
        called_setattr_override = False
        
        # 调用 `fsdp_module` 的前向传播方法，触发 `__setattr__` 方法的调用
        fsdp_module(inp)
        
        # 断言 `called_setattr_override` 已被设置为 `True`
        self.assertTrue(called_setattr_override)

        # 设置环境变量 `_FSDP_USE_UNSAFE_SETATTR` 为 "1"，启用不安全的 `setattr` 设置
        os.environ[_FSDP_USE_UNSAFE_SETATTR] = "1"
        
        # 重新创建 `module` 实例，与上述过程类似
        module = SetattrLinear(5, 5, torch.device("cuda"))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        called_setattr_override = False
        fsdp_module(inp)
        
        # 断言 `called_setattr_override` 未被设置为 `True`
        self.assertFalse(called_setattr_override)

        # 设置环境变量 `_FSDP_USE_UNSAFE_SETATTR` 为 "0"，禁用不安全的 `setattr` 设置
        os.environ[_FSDP_USE_UNSAFE_SETATTR] = "0"
        
        # 重新创建 `module` 实例，与上述过程类似
        module = SetattrLinear(5, 5, torch.device("cuda"))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        called_setattr_override = False
        fsdp_module(inp)
        
        # 断言 `called_setattr_override` 已被设置为 `True`
        self.assertTrue(called_setattr_override)
# 调用函数 instantiate_parametrized_tests，实例化参数化测试类 TestFSDPMiscMultiThread
instantiate_parametrized_tests(TestFSDPMiscMultiThread)

# 调用函数 instantiate_parametrized_tests，实例化参数化测试类 TestFSDPMiscMultiProcess
instantiate_parametrized_tests(TestFSDPMiscMultiProcess)

# 如果当前脚本作为主程序执行，则调用 run_tests 函数运行测试
if __name__ == "__main__":
    run_tests()
```