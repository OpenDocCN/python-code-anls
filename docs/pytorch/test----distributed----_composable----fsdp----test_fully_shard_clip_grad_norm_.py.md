# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_clip_grad_norm_.py`

```
# Owner(s): ["oncall: distributed"]
# 导入所需的库和模块
import copy  # 导入深拷贝函数
import functools  # 导入函数装饰器相关模块
from typing import Optional, Union  # 导入类型提示相关模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._composable import replicate  # 导入分布式相关模块
from torch.distributed._composable.fsdp import fully_shard  # 导入FSDP相关模块
from torch.distributed._tensor import Shard  # 导入张量分片相关模块
from torch.distributed._tensor.debug import CommDebugMode  # 导入通信调试模式相关模块
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh  # 导入设备网格相关模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试相关模块
from torch.testing._internal.common_fsdp import FSDPTest, MLPStack  # 导入FSDP测试相关模块
from torch.testing._internal.common_utils import run_tests  # 导入测试运行相关模块
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入分布式张量通用模块
    ModelArgs,
    Transformer,
    TransformerBlock,
)

# 定义一个测试类，继承自FSDPTest类
class _TestClipGradNormBase(FSDPTest):
    # 定义测试方法，用于测试梯度裁剪
    def _test_clip_grad_norm(
        self,
        max_norm: Union[float, int],  # 最大范数的阈值，可以是浮点数或整数
        norm_type: Union[float, int],  # 范数的类型，可以是浮点数或整数
        ref_model: nn.Module,  # 参考模型，继承自nn.Module的对象
        ref_optim: torch.optim.Optimizer,  # 参考优化器，继承自torch.optim.Optimizer的对象
        model: nn.Module,  # 待测试的模型，继承自nn.Module的对象
        optim: torch.optim.Optimizer,  # 待测试的优化器，继承自torch.optim.Optimizer的对象
        inp: torch.Tensor,  # 输入张量，继承自torch.Tensor的对象
        dp_mesh: Optional[DeviceMesh] = None,  # 设备网格，可选参数，默认为None
    ):
        # 创建一个部分应用了 torch.linalg.vector_norm 的函数，指定了 norm_type 参数
        vector_norm_fn = functools.partial(torch.linalg.vector_norm, ord=norm_type)
        
        # 如果 dp_mesh 为空，则使用函数 init_device_mesh 创建一个带有 "cuda" 设备的 dp_mesh
        dp_mesh = dp_mesh or init_device_mesh("cuda", (self.world_size,))
        
        # 设置随机种子为 43 + dp_mesh 的本地排名加 1
        torch.manual_seed(42 + dp_mesh.get_local_rank() + 1)
        
        # 迭代 10 次
        for iter_idx in range(10):
            # 清空 ref_optim 的梯度
            ref_optim.zero_grad()
            # 对 ref_model 的输出求和并反向传播
            ref_model(inp).sum().backward()
            
            # 清空 optim 的梯度
            optim.zero_grad()
            # 对 model 的输出求和并反向传播
            model(inp).sum().backward()

            # 获取 ref_model 的梯度，并创建其副本
            ref_grads = [p.grad.detach().clone() for p in ref_model.parameters()]
            
            # 获取 model 的本地梯度，并创建其副本
            local_grads = [
                p.grad.to_local().detach().clone() for p in model.parameters()
            ]
            
            # 遍历 ref_grads 和 model.parameters()，检查是否需要跳过参数检查
            for ref_grad, param in zip(ref_grads, model.parameters()):
                # 如果 param.placements 的元组为 (Shard(0), Shard(0))，则跳过检查
                if tuple(param.placements) == (Shard(0), Shard(0)):
                    continue
                # 使用 self.assertEqual 检查 ref_grad 是否等于 param.grad 的完整张量
                self.assertEqual(ref_grad, param.grad.full_tensor())

            # 检查至少有一个梯度的范数大于 max_norm，以确保剪裁不是无效的
            self.assertTrue(any(vector_norm_fn(g).item() > max_norm for g in ref_grads))
            self.assertTrue(
                any(vector_norm_fn(g).item() > max_norm for g in local_grads)
            )

            # 检查通过总范数和剪裁后的各个梯度范数进行梯度范数剪裁
            ref_total_norm = torch.nn.utils.clip_grad_norm_(
                ref_model.parameters(), max_norm=max_norm, norm_type=norm_type
            )
            
            # 使用 CommDebugMode() 模式进行通信调试
            comm_mode = CommDebugMode()
            with comm_mode:
                # 使用 torch.nn.utils.clip_grad_norm_ 进行梯度范数剪裁，并对 total_norm 进行赋值
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_norm,
                    norm_type=norm_type,
                )
            
            # 使用 self.assertEqual 检查 ref_total_norm 是否等于 total_norm 的完整张量
            self.assertEqual(ref_total_norm, total_norm.full_tensor())
            
            # 预期每个网格维度都会进行一个 all-reduce 操作，以便进行部分到复制的传递
            expected_all_reduces = len(total_norm.placements)
            # 使用 self.assertEqual 检查 comm_mode 中的通信计数是否与 expected_all_reduces 相等
            self.assertEqual(
                comm_mode.get_comm_counts()[torch.ops.c10d_functional.all_reduce],
                expected_all_reduces,
            )
            
            # 对于零梯度，剪裁没有效果
            for param, grad in zip(ref_model.parameters(), ref_grads):
                # 使用 self.assertTrue 检查 param.grad 的范数是否小于等于 max_norm
                self.assertTrue(vector_norm_fn(param.grad).item() <= max_norm)
                # 如果 grad 中有非零元素，则使用 self.assertFalse 检查 param.grad 是否等于 grad
                if torch.count_nonzero(grad):
                    self.assertFalse(torch.equal(param.grad, grad))
            for param, grad in zip(model.parameters(), local_grads):
                # 使用 self.assertTrue 检查 param.grad.to_local() 的范数是否小于等于 max_norm
                self.assertTrue(
                    vector_norm_fn(param.grad.to_local()).item() <= max_norm
                )
                # 如果 grad 中有非零元素，则使用 self.assertFalse 检查 param.grad.to_local() 是否等于 grad
                if torch.count_nonzero(grad):
                    self.assertFalse(torch.equal(param.grad.to_local(), grad))
class TestClipGradNormWorldSize2(_TestClipGradNormBase):
    @property
    def world_size(self) -> int:
        # 返回当前 CUDA 设备数量与 2 的较小值作为并行测试的世界大小
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_clip_grad_norm_1d(self):
        # 对于不同的梯度范数类型进行测试
        for norm_type in (2, 1, float("inf")):
            # 设置随机种子
            torch.manual_seed(42)
            # 创建 Transformer 模型参数对象，关闭 dropout
            model_args = ModelArgs(dropout_p=0.0)
            # 创建 Transformer 模型
            model = Transformer(model_args)
            # 复制并放置模型到 CUDA 设备上，作为参考模型
            ref_model = replicate(copy.deepcopy(model).cuda())
            # 创建参考模型的优化器
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
            # 对于每个 TransformerBlock，进行完全分片
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    fully_shard(module)
            # 对整个模型进行完全分片
            fully_shard(model)
            # 创建模型的优化器
            optim = torch.optim.Adam(model.parameters(), lr=1e-2)
            # 在 CUDA 设备上生成输入数据
            inp = torch.randint(0, model.model_args.vocab_size, (3, 16), device="cuda")
            # 执行梯度裁剪范数测试
            self._test_clip_grad_norm(
                1, norm_type, ref_model, ref_optim, model, optim, inp
            )


class TestClipGradNormWorldSize4(_TestClipGradNormBase):
    @property
    def world_size(self) -> int:
        # 返回当前 CUDA 设备数量与 4 的较小值作为并行测试的世界大小
        return min(torch.cuda.device_count(), 4)

    @skip_if_lt_x_gpu(4)
    def test_clip_grad_norm_2d(self):
        # 对于不同的梯度范数类型进行测试
        for norm_type in (2, 1, 3, float("inf")):
            # 设定每个处理器的大小
            dp_size = 2
            # 初始化 CUDA 设备网格
            global_mesh = init_device_mesh(
                "cuda",
                (dp_size, self.world_size // dp_size),
                mesh_dim_names=("dp", "tp"),
            )
            # 获取处理器网格 dp 和 tp
            dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
            # 设置随机种子
            torch.manual_seed(42)
            # 使用 MLP 堆栈进行测试，而不是 Transformer，因为 Transformer 对于 TP 存在较大的数值差异
            model = MLPStack(16, with_seq_parallel=True)
            # 复制并放置模型到 CUDA 设备上，使用处理器网格的进程组
            ref_model = replicate(
                copy.deepcopy(model).cuda(), process_group=dp_mesh.get_group()
            )
            # 创建参考模型的优化器
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
            # 并行化模型的层次结构
            model.parallelize(
                tp_mesh,
                dp_mesh,
                use_activation_checkpointing=False,
                reshard_after_forward=True,
            )
            # 创建模型的优化器
            optim = torch.optim.Adam(model.parameters(), lr=1e-2)
            # 在 CUDA 设备上生成输入数据
            inp = torch.randn(2, 16, device="cuda")
            # 执行梯度裁剪范数测试
            self._test_clip_grad_norm(
                0.5, norm_type, ref_model, ref_optim, model, optim, inp, dp_mesh
            )


if __name__ == "__main__":
    run_tests()
```