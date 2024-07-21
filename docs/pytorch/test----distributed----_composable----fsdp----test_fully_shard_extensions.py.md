# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_extensions.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import contextlib  # 提供上下文管理器的支持
import copy  # 提供对象的浅拷贝和深拷贝操作
import functools  # 提供装饰器和高阶函数的支持
import threading  # 提供多线程支持
import unittest  # 提供单元测试框架
from typing import Any, List, Optional, Tuple, Union  # 引入类型提示支持

import torch  # PyTorch主库
import torch.distributed as dist  # PyTorch分布式通信模块
import torch.nn as nn  # PyTorch神经网络模块
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy  # 引入分片策略和混合精度策略

from torch.distributed.device_mesh import DeviceMesh  # 引入设备网格模块
from torch.testing._internal.common_cuda import TEST_CUDA  # 引入CUDA测试支持
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 引入分布式测试支持
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,  # 引入检查分片奇偶性的函数
    FSDPTest,  # 引入FSDP测试类
    FSDPTestMultiThread,  # 引入多线程FSDP测试类
    MLP,  # 引入多层感知机模型
)
from torch.testing._internal.common_utils import run_tests  # 引入运行测试的函数
from torch.testing._internal.two_tensor import TwoTensor  # 引入两个张量的封装类


def two_tensor_fsdp_pre_all_gather(
    self, mesh: DeviceMesh
) -> Tuple[Tuple[torch.Tensor, ...], Any]:
    # 准备用于全聚合操作的输入张量元组和元数据
    all_gather_inputs = (self.a, self.b)
    metadata = None
    return all_gather_inputs, metadata


def two_tensor_fsdp_post_all_gather(
    self,
    all_gather_outputs: Tuple[torch.Tensor, ...],
    metadata: Any,
    param_dtype: torch.dtype,
    *,
    out: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], None]:
    # 检查元数据是否为None
    assert metadata is None, f"{metadata}"
    # 解包全聚合操作的输出张量
    a, b = all_gather_outputs
    if out is not None:
        # 如果输出不为None，则进行相关的类型和内存地址断言
        assert isinstance(out, TwoTensor), f"{type(out)}"
        if a.dtype == param_dtype:
            assert a.untyped_storage().data_ptr() == out.a.untyped_storage().data_ptr()
            assert b.untyped_storage().data_ptr() == out.b.untyped_storage().data_ptr()
        else:
            assert out.a.dtype == param_dtype, f"{out.a.dtype} {param_dtype}"
            assert out.b.dtype == param_dtype, f"{out.b.dtype} {param_dtype}"
            # 将全聚合的输出复制到指定的输出张量
            out.a.copy_(a)
            out.b.copy_(b)
        return
    # 若输出为None，则准备需要释放的张量元组
    tensors_to_free = (a, b)
    # 如果需要类型转换，则创建一个新的TwoTensor对象，并返回
    two_tensor = TwoTensor(a, b).to(param_dtype)
    return two_tensor, tensors_to_free


class TestFullyShardAllGatherExtensionsCommon:
    @property
    def world_size(self) -> int:
        # 返回世界大小为2
        return 2

    @contextlib.contextmanager
    def _patch_two_tensor_fsdp_all_gather(self):
        # 创建一个线程锁
        lock = threading.Lock()
        # 设置TwoTensor类的全聚合前和后处理函数
        TwoTensor.fsdp_pre_all_gather = two_tensor_fsdp_pre_all_gather
        TwoTensor.fsdp_post_all_gather = two_tensor_fsdp_post_all_gather
        # 分布式操作中的同步屏障
        dist.barrier()
        try:
            yield
        finally:
            # 再次执行分布式操作中的同步屏障
            dist.barrier()
            with lock:  # 只有一个线程需要删除
                # 删除TwoTensor类的全聚合前处理函数和全聚合后处理函数
                if hasattr(TwoTensor, "fsdp_pre_all_gather"):
                    delattr(TwoTensor, "fsdp_pre_all_gather")
                if hasattr(TwoTensor, "fsdp_post_all_gather"):
                    delattr(TwoTensor, "fsdp_post_all_gather")
    def _init_two_tensor_mlp(self) -> nn.Module:
        # 创建一个由三个MLP组成的序列模型，每个MLP都禁用偏置项
        model = nn.Sequential(*[MLP(8, bias=False) for _ in range(3)])
        # 遍历模型中的每个MLP
        for mlp in model:
            # 使用TwoTensor包装MLP的输入投影层权重，并将其设为可学习参数
            mlp.in_proj.weight = nn.Parameter(
                TwoTensor(mlp.in_proj.weight, mlp.in_proj.weight.clone())
            )
            # 使用TwoTensor包装MLP的输出投影层权重，并将其设为可学习参数
            mlp.out_proj.weight = nn.Parameter(
                TwoTensor(mlp.out_proj.weight, mlp.out_proj.weight.clone())
            )
        # 返回组装好的模型
        return model
class TestFullyShardAllGatherExtensionsMultiProcess(
    TestFullyShardAllGatherExtensionsCommon, FSDPTest
):
    # 继承自 TestFullyShardAllGatherExtensionsCommon 和 FSDPTest 的测试类
    @skip_if_lt_x_gpu(2)
    def test_all_gather_extensions_train_parity(self):
        # 如果 GPU 数量少于 2，则跳过测试
        with self._patch_two_tensor_fsdp_all_gather():
            # 使用 _patch_two_tensor_fsdp_all_gather 方法进行上下文管理
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_train_parity,
            )

    def _test_all_gather_extensions_train_parity(self, reshard_after_forward: bool):
        # 测试方法，检验 all_gather 扩展的训练一致性
        torch.manual_seed(42)
        # 设置随机种子
        model = self._init_two_tensor_mlp()
        # 初始化包含两个张量的 MLP 模型
        ref_model = copy.deepcopy(model).cuda()
        # 深拷贝模型并将其移到 GPU 上
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=True)
        # 初始化参考模型的 Adam 优化器，foreach 参数为 True
        fully_shard_fn = functools.partial(
            fully_shard, reshard_after_forward=reshard_after_forward
        )
        # 创建 partially applied 函数 fully_shard_fn，传入 reshard_after_forward 参数
        for mlp in model:
            fully_shard_fn(mlp)
        # 对模型中的每个 MLP 应用 fully_shard_fn 函数
        fully_shard_fn(model)
        # 对整个模型应用 fully_shard_fn 函数
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        # 初始化模型的 Adam 优化器，foreach 参数为 True
        check_sharded_parity(self, ref_model, model)
        # 使用 check_sharded_parity 方法检查参考模型和当前模型的一致性

        torch.manual_seed(42 + self.rank + 1)
        # 根据当前进程的等级设置新的随机种子
        inp = torch.randn((2, 8), device="cuda")
        # 在 GPU 上生成随机输入张量
        for iter_idx in range(10):
            # 迭代 10 次
            losses: List[torch.Tensor] = []
            # 创建一个列表 losses 用于存储损失张量
            for _model in (ref_model, model):
                # 针对参考模型和当前模型
                losses.append(_model(inp).sum())
                # 计算模型对输入的输出并求和得到损失
                losses[-1].backward()
                # 对最新添加的损失张量进行反向传播
                if _model is ref_model:
                    # 如果是参考模型
                    for param_name, param in _model.named_parameters():
                        dist.all_reduce(param.grad)
                        # 对参数的梯度进行全局归约操作
                        param.grad.detach().div_(self.world_size)
                        # 分离并归一化参数的梯度
            self.assertEqual(losses[0], losses[1])
            # 检查两个模型的损失是否相等
            check_sharded_parity(self, ref_model, model)
            # 再次检查参考模型和当前模型的一致性
            for _optim in (ref_optim, optim):
                # 针对参考优化器和当前优化器
                _optim.step()
                # 执行优化步骤
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                # 清除梯度，可选地将梯度置为 None（根据迭代次数的奇偶性）
            check_sharded_parity(self, ref_model, model)
            # 再次检查参考模型和当前模型的一致性


class TestFullyShardAllGatherExtensionsMultiThread(
    TestFullyShardAllGatherExtensionsCommon, FSDPTestMultiThread
):
    # 继承自 TestFullyShardAllGatherExtensionsCommon 和 FSDPTestMultiThread 的多线程测试类
    @property
    def device(self) -> torch.device:
        # 返回一个 CUDA 设备
        return torch.device("cuda:0")

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_all_gather_extensions_end_to_end(self):
        # 如果没有 CUDA，则跳过测试
        with self._patch_two_tensor_fsdp_all_gather():
            # 使用 _patch_two_tensor_fsdp_all_gather 方法进行上下文管理
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_end_to_end,
            )
            # 运行子测试，测试 all_gather 扩展的端到端功能
    def _test_all_gather_extensions_end_to_end(self, reshard_after_forward: bool):
        # 检查我们能否运行元设备初始化流程
        with torch.device("meta"):
            # 使用_meta_设备初始化一个双张量 MLP 模型
            model = self._init_two_tensor_mlp()
        for param in model.parameters():
            # 断言模型参数的设备为_meta_
            self.assertEqual(param.device, torch.device("meta"))
        # 部分应用完全分片函数，其中包括重新前向传播后的完全分片和混合精度策略
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        # 对模型中的每个 MLP 执行完全分片
        for mlp in model:
            fully_shard_fn(mlp)
        # 对整个模型执行完全分片
        fully_shard_fn(model)
        # 将模型移动到空设备，这里的设备由self.device指定
        model.to_empty(device=self.device)
        # 对模型参数初始化为截断正态分布
        for param in model.parameters():
            nn.init.trunc_normal_(param)
        # 使用Adam优化器优化模型参数，lr为学习率，foreach=True表示批处理
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)

        # 运行几次迭代以检查是否出现错误
        torch.manual_seed(42 + self.rank + 1)
        # 生成一个在CUDA设备上的随机张量作为输入
        inp = torch.randn((2, 8), device="cuda")
        for _ in range(3):
            # 对模型进行前向传播，计算损失并反向传播
            model(inp).sum().backward()
            # 执行优化步骤
            optim.step()
            # 清空梯度
            optim.zero_grad()

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_all_gather_extensions_monkey_patch(self):
        # 定义一个前/后全收集对，用于将参数量化为 bf16 进行全收集，然后再将其反量化为参数的数据类型
        def fsdp_pre_all_gather(self) -> Tuple[Tuple[torch.Tensor, ...], Any]:
            return (self.to(torch.bfloat16),), None

        def fsdp_post_all_gather(
            self,
            all_gather_outputs: Tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: Optional[torch.Tensor] = None,
        ) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], None]:
            (tensor,) = all_gather_outputs
            assert metadata is None, f"{metadata}"
            assert tensor.dtype == torch.bfloat16, f"{tensor.dtype}"
            if out is not None:
                out.copy_(tensor)
                return
            return tensor.to(param_dtype), (tensor,)

        # 使用元设备 "meta" 来初始化一个包含两个张量的 MLP 模型
        with torch.device("meta"):
            model = self._init_two_tensor_mlp()
        
        # 对模型中的每个 MLP 应用完全分片
        for mlp in model:
            fully_shard(mlp)
        
        # 对整个模型应用完全分片
        fully_shard(model)
        
        # 将模型转移到空设备，使用 self.device 指定的设备
        model.to_empty(device=self.device)
        
        # 对模型的每个参数进行截断正态初始化
        for param in model.parameters():
            nn.init.trunc_normal_(param)
        
        # 在 `to_empty()` 后，对预/后全收集函数进行 monkey patch，
        # 因为本地张量对象在材料化后会发生变化
        self.assertGreater(sum("weight" in n for n, _ in model.named_parameters()), 0)
        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                local_param = param.to_local()
                # Monkey patch `torch.Tensor`，以显示扩展即使在没有子类的情况下也可以工作
                local_param.fsdp_pre_all_gather = fsdp_pre_all_gather
                local_param.fsdp_post_all_gather = fsdp_post_all_gather
        
        # 使用 foreach=True 参数初始化 Adam 优化器
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)

        # 运行几次迭代以检查是否存在错误
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 8), device="cuda")
        for _ in range(3):
            model(inp).sum().backward()
            optim.step()
            optim.zero_grad()
# 如果当前脚本作为主程序运行（而非被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```