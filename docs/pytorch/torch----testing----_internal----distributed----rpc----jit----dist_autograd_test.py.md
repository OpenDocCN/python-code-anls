# `.\pytorch\torch\testing\_internal\distributed\rpc\jit\dist_autograd_test.py`

```
# 忽略类型检查错误，由于特定的代码约束或限制，忽略类型检查有时是必要的
# 导入需要的模块和类
from typing import Dict, Tuple  # 导入类型提示相关的模块

import torch  # 导入PyTorch库
import torch.distributed.autograd as dist_autograd  # 导入分布式自动求导模块
import torch.distributed.rpc as rpc  # 导入分布式RPC模块
from torch import Tensor  # 导入张量类
from torch.distributed.rpc import rpc_async  # 导入RPC异步调用函数
from torch.testing import FileCheck  # 导入测试工具中的文件检查模块
from torch.testing._internal.dist_utils import dist_init, worker_name  # 导入分布式测试工具中的初始化函数和工作节点名称函数
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,  # 导入RPC代理测试基础类
)


@torch.jit.script
def local_add(t1, t2):
    return torch.add(t1, t2)  # 对两个张量进行本地加法操作


@torch.jit.script
def remote_add(t1, t2, dst: str):  # noqa: E999
    return rpc_async(dst, local_add, (t1, t2)).wait()  # 异步RPC调用远程节点上的local_add函数并等待返回结果


@torch.jit.script
def fork_add(t1, t2, dst: str):
    fut = torch.jit._fork(remote_add, t1, t2, dst)  # 在分布式环境中fork一个远程的add操作
    return torch.jit._wait(fut)  # 等待fork的操作完成并返回结果


class JitDistAutogradTest(RpcAgentTestFixture):
    @dist_init
    def test_get_gradients(self):
        dst_rank = self.rank  # 获取当前进程的排名

        @torch.jit.script
        def dist_get_gradients(context_id: int) -> (Dict[Tensor, Tensor]):
            return dist_autograd.get_gradients(context_id)  # 获取指定上下文中的梯度信息

        FileCheck().check("get_gradients").run(str(dist_get_gradients.graph))  # 检查梯度获取函数的图形表示
        with dist_autograd.context() as context_id:  # 使用分布式自动求导的上下文
            t1 = torch.rand((3, 3), requires_grad=True)  # 创建一个需要梯度的随机张量
            t2 = torch.rand((3, 3), requires_grad=True)  # 创建另一个需要梯度的随机张量
            t3 = torch.add(t1, t2)  # 对两个张量进行加法操作

            dist_autograd.backward(context_id, [t3.sum()])  # 对加法结果的和进行反向传播
            grads = dist_get_gradients(context_id)  # 获取反向传播后的梯度信息

            self.assertEqual(2, len(grads))  # 断言梯度数量为2
            self.assertIn(t1, grads)  # 断言t1在梯度字典中
            self.assertIn(t2, grads)  # 断言t2在梯度字典中
            self.assertEqual(torch.ones(3, 3), grads[t1])  # 断言t1的梯度为全1张量
            self.assertEqual(torch.ones(3, 3), grads[t2])  # 断言t2的梯度为全1张量

    @dist_init
    def test_dist_backward(self):
        if self.rank != 0:
            return  # 如果当前进程不是rank为0的进程，直接返回

        @torch.jit.script
        def dist_backward_script(context_id: int, loss: torch.Tensor):
            dist_autograd.backward(context_id, [loss])  # 在指定上下文中进行反向传播操作

        FileCheck().check("dist_backward").run(str(dist_backward_script.graph))  # 检查分布式反向传播脚本的图形表示
        with dist_autograd.context() as context_id:  # 使用分布式自动求导的上下文
            t1 = torch.rand(3, 3, requires_grad=True)  # 创建一个需要梯度的随机张量
            t2 = torch.rand(3, 3, requires_grad=True)  # 创建另一个需要梯度的随机张量
            dst_worker_name = worker_name((self.rank + 1) % self.world_size)  # 获取下一个工作节点的名称
            loss = rpc.rpc_sync(dst_worker_name, torch.add, args=(t1, t2)).sum()  # 在远程节点上进行同步RPC调用并求和
            dist_backward_script(context_id, loss)  # 在指定上下文中进行分布式反向传播操作

    @dist_init
    def test_jit_fork_within_context(self):
        with dist_autograd.context() as context_id:  # 使用分布式自动求导的上下文
            t1 = torch.rand((3, 3), requires_grad=True)  # 创建一个需要梯度的随机张量
            t2 = torch.rand((3, 3), requires_grad=True)  # 创建另一个需要梯度的随机张量
            dst_worker_name = worker_name((self.rank + 1) % self.world_size)  # 获取下一个工作节点的名称
            res = fork_add(t1, t2, dst_worker_name)  # 调用分布式fork加法操作
            loss = res.sum()  # 对结果求和作为损失
            dist_autograd.backward(context_id, [loss])  # 在指定上下文中进行反向传播操作

            grads = dist_autograd.get_gradients(context_id)  # 获取反向传播后的梯度信息
            self.assertEqual(2, len(grads))  # 断言梯度数量为2
            self.assertIn(t1, grads)  # 断言t1在梯度字典中
            self.assertIn(t2, grads)  # 断言t2在梯度字典中
    # 使用 dist_init 装饰器初始化分布式设置
    @dist_init
    # 定义一个测试函数，用于测试在切换到 JIT 线程后是否能恢复上下文
    def test_restore_context_after_swtich_to_jit_thread(self):
        # 如果当前进程的 rank 不是 0，则直接返回，不进行后续操作
        if self.rank != 0:
            return

        # 定义一个 TorchScript 函数 forward_script，接受四个参数并返回两个 Tensor 的元组
        @torch.jit.script
        def forward_script(
            context_id: int, dst_worker_name: str, t1: Tensor, t2: Tensor
        ) -> Tuple[Tensor, Tensor]:
            # 发送异步 RPC 请求到指定的远程 worker，执行 local_add 函数对 t1 进行操作
            res1_fut = rpc.rpc_async(dst_worker_name, local_add, (t1, t1))
            # 等待异步操作完成，并获取结果
            res1 = res1_fut.wait()  # After this, the script runs in a new JIT thread.
            # 计算结果的和作为 loss1
            loss1 = res1.sum()

            # 由于此处丢失了 DistAutogradContext，因此不会附加 SendRpcBackward 操作
            # 发送异步 RPC 请求到指定的远程 worker，执行 local_add 函数对 t2 进行操作
            res2_fut = rpc.rpc_async(dst_worker_name, local_add, (t2, t2))
            # 等待异步操作完成，并获取结果
            res2 = res2_fut.wait()
            # 计算结果的和作为 loss2
            loss2 = res2.sum()

            # 返回 loss1 和 loss2
            return loss1, loss2

        # 使用 dist_autograd.context() 创建一个新的 autograd 上下文
        with dist_autograd.context() as context_id:
            # 创建两个 shape 为 (2, 3) 的全 1 Tensor，并标记为需要梯度计算
            t1 = torch.ones((2, 3), requires_grad=True)
            t2 = torch.ones((2, 3), requires_grad=True)
            # 计算目标 worker 的名称，用于后续的 RPC 调用
            dst_worker_name = worker_name((self.rank + 1) % self.world_size)
            # 调用 forward_script 函数进行前向计算，并获取返回的 loss0 和 loss1
            loss0, loss1 = forward_script(context_id, dst_worker_name, t1, t2)
            # 使用 dist_autograd.backward() 执行反向传播，计算梯度
            dist_autograd.backward(context_id, [loss0, loss1])
            # 获取计算得到的梯度值
            grad0, grad1 = dist_autograd.get_gradients(context_id)
            # 断言两个梯度值相等
            self.assertEqual(grad0, grad1)
```