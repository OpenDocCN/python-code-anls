# `.\pytorch\torch\testing\_internal\distributed\rpc\dist_autograd_test.py`

```py
# mypy: ignore-errors

import sys  # 导入sys模块，用于系统相关操作
import threading  # 导入threading模块，支持多线程编程
import time  # 导入time模块，处理时间相关操作
from enum import Enum  # 导入Enum类，支持枚举类型的定义
import random  # 导入random模块，生成随机数
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from datetime import timedelta  # 从datetime模块导入timedelta类，处理时间差
import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.distributed.autograd as dist_autograd  # 导入PyTorch分布式自动求导模块
import torch.distributed.rpc as rpc  # 导入PyTorch分布式RPC模块
import torch.testing._internal.dist_utils  # 导入PyTorch内部分布式测试工具
from torch.autograd import Function  # 从torch.autograd导入Function类
from torch.autograd.function import once_differentiable  # 导入once_differentiable装饰器
from torch.distributed.rpc import RRef  # 导入PyTorch分布式RPC中的RRef类
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if  # 导入测试工具和平台相关的函数
from torch.testing._internal.dist_utils import (  # 导入内部分布式测试工具中的特定函数
    dist_init,
    initialize_pg,
    wait_until_node_failure,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (  # 导入RPC测试工具中的TestFixture类
    RpcAgentTestFixture,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式相关的测试工具函数


# Right now we test up to 3-layer nested rpc calls.
# rpc_done[1] and ctx_ids[1] represent rpc is done in prev rank, and context id
# sent from prev rank respectively.
# rpc_done[2] and ctx_ids[2] represents for prev of prev rank.
# rpc_done[3] and ctx_ids[3] represents for prev of prev of prev rank.
# rpc_done[0] and ctx_ids[0] represents for current rank, but mostly not used.
rpc_done = [False, False, False, False]  # 定义用于标记RPC是否完成的列表
ctx_ids = [-1, -1, -1, -1]  # 定义用于存储上下文ID的列表

known_context_ids = set()  # 定义空集合，用于存储已知的上下文ID

requires_grad_tensor = torch.ones(3, 3, requires_grad=True)  # 创建一个需要梯度的3x3张量


# Send rpc done info and context_id to
# dst_rank = (self.rank + rank_distance) % self.world_size
# we don't need a lock here since the GIL is held while executing remote
# python UDFs, so access is serialized across several workers.
def _set_rpc_done(ctx_id, rank_distance):
    global rpc_done  # 声明使用全局变量rpc_done
    global ctx_ids  # 声明使用全局变量ctx_ids
    global known_context_ids  # 声明使用全局变量known_context_ids
    rpc_done[rank_distance] = True  # 设置指定rank_distance位置的rpc_done为True，表示RPC完成
    ctx_ids[rank_distance] = ctx_id  # 将上下文ID存储到指定rank_distance位置的ctx_ids列表中
    known_context_ids.add(ctx_id)  # 将上下文ID添加到已知上下文ID集合中


def _check_rpc_done(rank_distance):
    while not rpc_done[rank_distance]:  # 当指定rank_distance位置的rpc_done不为True时
        time.sleep(0.1)  # 等待0.1秒，检查RPC是否完成


def _torch_ones(sizes, requires_grad=False):
    return torch.ones(sizes, requires_grad=requires_grad)  # 返回一个指定大小的张量，可以指定是否需要梯度


# This method must be called on the rref owner, and verifies that the grad of
# rref tensor equals to the given grad.
def _compare_owner_value(context_id, rref, grad):
    grads = dist_autograd.get_gradients(context_id)  # 获取指定上下文ID的梯度信息
    x = grads[rref.local_value()]  # 获取rref对应的本地值的梯度
    if x.is_sparse:  # 如果x是稀疏张量
        assert grad.is_sparse  # 断言给定的梯度grad也是稀疏张量
        x = x.to_dense()  # 将稀疏张量转换为密集张量
        grad = grad.to_dense()  # 将给定的梯度grad转换为密集张量
    else:  # 如果x不是稀疏张量
        assert not grad.is_sparse  # 断言给定的梯度grad不是稀疏张量
    return torch.equal(x, grad)  # 比较x和grad是否相等


def create_tensor():
    return torch.ones((3, 3), requires_grad=True)  # 创建一个需要梯度的3x3张量


def build_sparse_tensor(coalesce=False, requires_grad=True, dtype=torch.float32):
    i = [[0, 1, 1], [2, 0, 2]]  # 稀疏张量的索引
    v = [3.2, 4.1, 5.3]  # 稀疏张量的值
    tensor = torch.sparse_coo_tensor(
        i, v, (3, 3), requires_grad=requires_grad, dtype=dtype
    )  # 创建稀疏张量
    if coalesce:  # 如果需要合并
        tensor = tensor.coalesce()  # 合并稀疏张量
    return tensor  # 返回构建好的稀疏张量


@torch.jit.script
def create_torchscript_tensor() -> torch.Tensor:
    return torch.ones((3, 3)).requires_grad_()  # 创建一个TorchScript张量，需要梯度
# 定义一个函数，使用 PyTorch 提供的 torch.add() 函数实现张量 t1 和 t2 的加法
def my_py_add(t1, t2):
    return torch.add(t1, t2)


# 定义一个函数，使用标准的加法运算符实现两个数 a 和 b 的加法
def my_scalar_add(a, b):
    return a + b


# 定义一个函数，通过调用 torch.add() 函数对一个 RRef 引用的本地值 rref_t1 和张量 t2 进行加法运算
def my_rref_add(rref_t1, t2):
    ret = torch.add(rref_t1.local_value(), t2)
    return ret


# 使用 @torch.jit.script 装饰器定义一个 TorchScript 脚本函数，实现张量 t1 和 t2 的加法
@torch.jit.script
def my_script_add(t1, t2):
    return torch.add(t1, t2)


# 使用 @torch.jit.script 装饰器定义一个 TorchScript 脚本函数，实现 RRef 引用 ref_t1 和张量 t2 的加法
def my_script_ref_add(ref_t1: RRef[torch.Tensor], t2: torch.Tensor) -> torch.Tensor:
    # 将 RRef 引用的张量数据移到本地，并对其和张量 t2 进行加法运算
    t1 = ref_t1.to_here()
    return torch.add(t1, t2)


# 定义一个函数，通过 RPC 调用远程服务器上的 my_rref_add 函数，实现在远程执行 rref_t1 和 t2 的加法
def my_nested_rref_add(dst, rref_t1, t2):
    return rpc.rpc_sync(dst, my_rref_add, args=(rref_t1, t2))


# 定义一个函数，返回一个全局张量 requires_grad_tensor，其梯度计算已启用
def ret_requires_grad():
    return requires_grad_tensor


# 定义一个递归函数，通过 RPC 在不同的 worker 上调用 my_py_add 函数，实现 t1 和 t2 的加法
def my_py_nested_call(t1, t2, dst, world_size, hops):
    next_dst = (dst + 1) % world_size
    if hops > 0:
        # 如果 hops 大于 0，则递归调用 my_py_nested_call 函数
        return rpc.rpc_sync(
            worker_name(next_dst),
            my_py_nested_call,
            args=(t1, t2, next_dst, world_size, hops - 1),
        )
    else:
        # 如果 hops 不大于 0，则调用 my_py_add 函数
        return rpc.rpc_sync(worker_name(next_dst), my_py_add, args=(t1, t2))


# 定义一个函数，用于清理分布式自动求导上下文，等待所有节点的上下文在给定的超时时间内清理完毕
def _all_contexts_cleaned_up(timeout_seconds=10):
    global known_context_ids
    start = time.time()
    context_id_to_raised = set()
    while (
        time.time() - start < timeout_seconds
        and context_id_to_raised != known_context_ids
    ):
        for context_id in known_context_ids:
            try:
                # 尝试检索指定的分布式自动求导上下文
                dist_autograd._retrieve_context(context_id)
            except RuntimeError:
                context_id_to_raised.add(context_id)
    # 如果尝试检索任何上下文导致 RuntimeError，则认为所有上下文已清理完毕
    success = context_id_to_raised == known_context_ids
    return success


# 定义一个函数，创建一个分布式自动求导上下文，运行在给定的 ps 上通过 rpc_sync 调用 my_rref_add 函数，
# 并阻塞直到 ps 验证梯度正确累积
def _run_trainer(rref_t1, t2, ps, rank_diff, sparse):
    with dist_autograd.context() as context_id:
        ret = rpc.rpc_sync(ps, my_rref_add, args=(rref_t1, t2))
        if sparse:
            # 如果 sparse 为 True，则对稀疏张量 ret 求和
            loss = torch.sparse.sum(ret)
        else:
            # 否则对 ret 张量求和
            loss = ret.sum()
        # 执行分布式自动求导反向传播，以 loss 为损失
        dist_autograd.backward(context_id, [loss])
        # 防止删除分布式自动求导上下文
        rpc.rpc_sync(ps, _set_rpc_done, args=(context_id, rank_diff))
        rpc.rpc_sync(ps, _check_rpc_done, args=(0,))


# 定义一个函数，与 _run_trainer 类似，不同之处在于调用 TorchScript 函数 "my_script_ref_add" 替代 Python 函数 "my_rref_add"
def _run_trainer_torchscript(rref_t1, t2, ps, rank_diff, sparse):
    # 使用分布式自动求导上下文管理器，创建一个新的上下文
    with dist_autograd.context() as context_id:
        # 通过 RPC 调用执行异步远程过程调用，获取结果
        ret = rpc.rpc_sync(ps, my_script_ref_add, args=(rref_t1, t2))
        
        # 根据稀疏标志选择计算损失的方式
        if sparse:
            # 如果稀疏为真，则计算稀疏张量的和作为损失
            loss = torch.sparse.sum(ret)
        else:
            # 否则计算普通张量的和作为损失
            loss = ret.sum()
        
        # 在指定的上下文中进行反向传播，传递损失作为参数
        dist_autograd.backward(context_id, [loss])
        
        # 防止删除分布式自动求导上下文
        rpc.rpc_sync(ps, _set_rpc_done, args=(context_id, rank_diff))
        
        # 检查异步远程过程调用是否完成，参数为0表示检查指定上下文中的完成状态
        rpc.rpc_sync(ps, _check_rpc_done, args=(0,))
# 定义一个模拟反向传播错误的函数，继承自Function类
class SimulateBackwardError(Function):
    # 类变量，用于标识是否模拟错误
    _simulate_error = True

    # 静态方法：前向传播函数，直接返回输入
    @staticmethod
    def forward(ctx, input):
        return input

    # 静态方法：反向传播函数
    @staticmethod
    @once_differentiable
    def backward(ctx, input):
        # 如果_simulate_error为True，抛出异常模拟反向传播错误
        if SimulateBackwardError._simulate_error:
            raise Exception("Simulate error on backward pass")  # noqa: TRY002
        else:
            return input


# 枚举类，定义执行模式
class ExecMode(Enum):
    LOCAL = 1  # 在本地运行操作
    RPC_SYNC = 2  # 使用rpc_sync运行操作
    REMOTE = 3  # 使用远程方式运行操作
    RPC_ASYNC = 4  # 使用rpc_async运行操作


# 用于CPU和CUDA测试套件的通用工具类
class CommonDistAutogradTest(RpcAgentTestFixture):
    # 执行带有目标dst的函数，根据exec_mode选择执行方式
    def _exec_func_with_dst(self, dst, exec_mode, method, *args):
        if ExecMode.LOCAL == exec_mode:  # 如果执行模式为本地
            if len(args) == 1 and isinstance(args[0], list):
                return method(*args[0])
            return method(*args)
        elif ExecMode.RPC_SYNC == exec_mode:  # 如果执行模式为rpc_sync
            return rpc.rpc_sync(worker_name(dst), method, args=(args))
        elif ExecMode.REMOTE == exec_mode:  # 如果执行模式为远程
            return rpc.remote(worker_name(dst), method, args=(args)).to_here()
        elif ExecMode.RPC_ASYNC == exec_mode:  # 如果执行模式为rpc_async
            fut = rpc.rpc_async(worker_name(dst), method, args=(args))
            return fut.wait()
        else:
            # 抛出数值错误，表示未识别的执行模式
            raise ValueError(f"Unrecognized ExecMode {exec_mode}")

    # 执行函数，根据exec_mode选择执行方式，并带有下一个目标rank
    def _exec_func(self, exec_mode, method, *args):
        return self._exec_func_with_dst(
            self._next_rank(), exec_mode, method, *args
        )

    # 获取下一个rank值，用于RPC操作
    def _next_rank(self):
        if hasattr(self, "dst_rank"):
            self.dst_rank = (self.dst_rank + 1) % self.world_size
            if self.dst_rank == self.rank:
                return self._next_rank()
        else:
            self.dst_rank = (self.rank + 1) % self.world_size
        return self.dst_rank

    # 检查RPC是否完成，根据rank_distance进行检查
    def _check_rpc_done(self, rank_distance):
        _check_rpc_done(rank_distance)

    # 验证反向传播的结果，根据exec_mode不同选择本地或远程验证方式
    def _verify_backwards(self, exec_mode, tensors, context_id, local_grads, *args):
        if exec_mode == ExecMode.LOCAL:  # 如果执行模式为本地
            torch.autograd.backward(tensors)
            return [arg.grad for arg in args]
        else:
            self._verify_backwards_remote(tensors, context_id, local_grads, *args)

    # 远程验证反向传播的结果，使用dist_autograd进行远程操作
    def _verify_backwards_remote(self, tensors, context_id, local_grads, *args):
        dist_autograd.backward(context_id, tensors)

        # 验证梯度是否被正确累积
        grads = dist_autograd.get_gradients(context_id)
        nargs = len(args)
        ngrads = 0
        for i in range(0, nargs):
            if local_grads[i] is not None:
                self.assertIn(args[i], grads)
                self.assertEqual(local_grads[i], grads[args[i]])
                ngrads += 1
            else:
                self.assertNotIn(args[i], grads)

        self.assertEqual(ngrads, len(grads))
    # 定义一个测试方法，用于测试分布式图计算
    def _test_graph(self, fn, exec_mode, sparse):
        # 计算目标排名，使其为当前排名加一后对总排名取余数
        dst_rank = (self.rank + 1) % self.world_size

        # 使用指定的初始化方法初始化进程组，传入当前排名和总排名
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 进入分布式自动微分的上下文
        with dist_autograd.context() as context_id:
            # 如果指定使用稀疏张量
            if sparse:
                # 构建稀疏张量 t1 和 t2
                t1 = build_sparse_tensor()
                t2 = build_sparse_tensor()
            else:
                # 否则构建普通张量 t1 和 t2，设置 requires_grad 为 True
                t1 = torch.ones(3, 3, requires_grad=True)
                t2 = torch.zeros(3, 3, requires_grad=True)
            
            # 根据执行模式选择执行远程过程调用
            if ExecMode.RPC_SYNC == exec_mode:
                # 同步 RPC 调用，将函数 fn 以 t1 和 t2 作为参数传递给目标排名的工作节点
                ret = rpc.rpc_sync(worker_name(dst_rank), fn, args=(t1, t2))
            elif ExecMode.REMOTE == exec_mode:
                # 远程 RPC 调用，将函数 fn 以 t1 和 t2 作为参数传递给目标排名的工作节点，并在本地等待结果返回
                ret = rpc.remote(
                    worker_name(dst_rank), fn, args=(t1, t2)
                ).to_here()
            else:
                # 抛出异常，表示未识别的执行模式
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            # 在目标排名的工作节点上执行同步 RPC 调用，标记当前上下文为完成状态
            rpc.rpc_sync(
                worker_name(dst_rank), _set_rpc_done, args=(context_id, 1)
            )

            # 验证当前上下文 ID 的图结构
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            # 确保发送函数集合只包含一个函数
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            # 确保接收函数集合只包含一个函数
            self.assertEqual(1, len(recv_functions))
            # 验证第一个 RPC 调用的图结构
            self._verify_graph_for_first_rpc_call(
                next(iter(send_functions.values())),
                next(iter(recv_functions.values())),
                t1,
                t2,
                ret,
            )

            # 等待前一排名的工作节点完成 RPC 调用
            self._check_rpc_done(1)
            # 根据上一个上下文 ID 验证 RPC 调用的图结构
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            send_functions = ctx._send_functions()
            # 确保发送函数集合只包含一个函数
            self.assertEqual(1, len(send_functions))
            # 验证执行 RPC 调用的图结构
            self._verify_graph_for_rpc_call_exec(next(iter(send_functions.values())))
            # 此处需要屏障，以确保一个工作节点在另一个尝试访问它之前不清理其自动微分上下文
            dist.barrier()

        # 现在应该清理自动微分上下文
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._retrieve_context(context_id)

        # 没有可用的自动微分上下文
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._current_context()

    # 三层嵌套调用示例
    # Rank0->Rank1->Rank0
    # 定义测试方法，验证在不需要梯度的张量上执行的情况，包括执行模式和稀疏性
    def _test_no_graph_with_tensors_not_require_grad(self, exec_mode, sparse):
        # 使用给定的初始化方法、进程编号和总进程数初始化分布式进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # 计算下一个目标进程的排名
        dst_rank = (self.rank + 1) % self.world_size
        # 使用dist_autograd上下文进行操作
        with dist_autograd.context() as context_id:
            # 根据稀疏性选择构建稀疏张量（不需要梯度）或者构建全一或全零张量（不需要梯度）
            if sparse:
                t1 = build_sparse_tensor(requires_grad=False)
                t2 = build_sparse_tensor(requires_grad=False)
            else:
                t1 = torch.ones(3, 3, requires_grad=False)
                t2 = torch.zeros(3, 3, requires_grad=False)
            # 根据执行模式调用远程过程调用（RPC），执行张量的加法操作
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(
                    worker_name(dst_rank), torch.add, args=(t1, t2)
                )
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(
                    worker_name(dst_rank), torch.add, args=(t1, t2)
                ).to_here()
            else:
                # 抛出异常，如果执行模式未知
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            # 在目标进程上同步RPC，通知其完成任务
            rpc.rpc_sync(
                worker_name(dst_rank), _set_rpc_done, args=(context_id, 1)
            )

            # 获取当前的dist_autograd上下文
            ctx = dist_autograd._current_context()
            # 检查发送函数列表是否为空
            send_functions = ctx._send_functions()
            self.assertEqual(len(send_functions), 0)
            # 检查接收函数列表是否为空
            recv_functions = ctx._recv_functions()
            self.assertEqual(len(recv_functions), 0)

            # 等待前一个进程完成RPC调用
            self._check_rpc_done(1)
            # 注意事项：RRef.to_here()总是将自动求导上下文传递给调用者，
            # 因为调用者不知道返回值是否包含需要梯度的张量。
            #
            # 对于带有用户定义函数（例如这里的_set_rpc_done）的rpc/remote调用，
            # 同样由于同样的原因，总是将自动求导上下文传递给被调用者。
            self.assertNotEqual(-1, dist_autograd._retrieve_context(ctx_ids[1]))
            # 在所有进程上进行同步栅栏操作
            dist.barrier()
    # 定义一个测试方法，用于测试远程过程调用（RPC）中处理复杂参数的情况
    def _test_rpc_complex_args(self, exec_mode, sparse):
        # 使用分布式自动求导上下文管理器获取上下文ID
        with dist_autograd.context() as context_id:
            # 定义张量的数量
            num_tensors = 10
            # 初始化一个空列表，用于存储生成的张量
            tensors = []
            # 循环创建张量
            for i in range(num_tensors):
                # 根据稀疏标志选择是创建稀疏张量还是全1张量
                if sparse:
                    tensor = build_sparse_tensor(requires_grad=(i % 2 == 0))
                else:
                    tensor = torch.ones(3, 3, requires_grad=(i % 2 == 0))
                # 将创建的张量加入到列表中
                tensors.append(tensor)
            
            # 获取下一个目标计算节点的排名
            dst_rank = self._next_rank()
            
            # 根据执行模式选择不同的RPC调用方式
            if ExecMode.RPC_SYNC == exec_mode:
                # 使用rpc_sync同步调用远程计算节点上的torch.stack方法
                ret = rpc.rpc_sync(
                    worker_name(dst_rank), torch.stack, args=(tensors,)
                )
            elif ExecMode.REMOTE == exec_mode:
                # 使用remote异步调用远程计算节点上的torch.stack方法，并将结果同步回本地
                ret = rpc.remote(
                    worker_name(dst_rank), torch.stack, args=(tensors,)
                ).to_here()
            else:
                # 如果执行模式未知，则抛出异常
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            # 断言返回的结果与本地计算的torch.stack结果相同
            self.assertEqual(torch.stack(tensors), ret)

            # 验证哪些张量已经被附加到自动求导图中
            next_funcs = next(iter(dist_autograd._current_context()._send_functions().values())).next_functions
            idx = 0
            for i in range(len(next_funcs)):
                # 断言下一个函数是torch::autograd::AccumulateGrad类型
                self.assertEqual(
                    "torch::autograd::AccumulateGrad", next_funcs[i][0].name()
                )
                # 断言张量与下一个函数的变量相匹配
                self.assertEqual(tensors[i], next_funcs[i][0].variable)

            # 验证该工作节点ID已记录在上下文中
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(len(worker_ids), 1)
            self.assertEqual(worker_ids, {dst_rank})
    # 定义一个辅助函数，用于测试上下文清理
    def context_cleanup_test_helper(self, rpc_args, func, nested=False):
        # 调用初始化函数，设定文件初始化方法、当前进程的排名和世界大小
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 在分布式自动求导中测试，在 RPC 通信期间，即使张量参数不需要梯度，
        # 我们仍然清理在其他节点上创建的分布式自动求导上下文。
        # 这是因为即使张量参数不需要梯度，自动求导上下文仍然会通过 RPC 进行通信，
        # 因为响应可能需要梯度。
        if nested:
            # 如果是嵌套调用，确定目标排名和嵌套目标排名
            dst_rank = (self.rank + 1) % self.world_size
            nested_dst_rank = (dst_rank + 1) % self.world_size
            dst_ranks = {dst_rank}
        else:
            # 如果不是嵌套调用，目标排名为除了当前排名之外的所有排名
            dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}

        # 使用分布式自动求导上下文，返回上下文 ID
        with dist_autograd.context() as context_id:
            for dst_rank in dst_ranks:
                # 使用 RPC 同步调用远程工作节点的函数 func，并传递参数 rpc_args
                rpc.rpc_sync(worker_name(dst_rank), func, args=rpc_args)
                # 在远程工作节点上设置 RPC 完成状态
                rpc.rpc_sync(
                    worker_name(dst_rank), _set_rpc_done, args=(context_id, 1)
                )
                if nested:
                    # 如果是嵌套调用，还需要在嵌套目标排名上设置 RPC 完成状态
                    rpc.rpc_sync(
                        worker_name(nested_dst_rank),
                        _set_rpc_done,
                        args=(context_id, 2),
                    )
        # 确保线程的上下文 ID 被清理
        with self.assertRaises(RuntimeError):
            dist_autograd._retrieve_context(context_id)
        # 确保所有节点都完成了对 `known_context_ids` 集合的突变
        dist.barrier()
        # 检查所有上下文是否已被清理
        success = _all_contexts_cleaned_up()
        self.assertTrue(success)

    # 测试在张量不需要梯度的情况下的反向传播
    def _backward_no_grad_on_tensor(self, t1, t2, sparse):
        # 使用分布式自动求导上下文，返回上下文 ID
        with dist_autograd.context() as context_id:
            # 在下一个排名的工作节点上同步调用 torch.add 函数，并传递参数 t1, t2
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()),
                torch.add,
                args=(t1, t2))
            if sparse:
                # 如果稀疏标志为真，对 loss 使用稀疏求和
                loss = torch.sparse.sum(loss)
            else:
                # 否则对 loss 使用求和
                loss = loss.sum()
            # 在给定的上下文 ID 下进行反向传播，计算梯度，保留计算图
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            # 检查 t1 和 t2 的梯度是否为 None
            self.assertIsNone(t1.grad)
            self.assertIsNone(t2.grad)

            # 现在使用本地自动求导引擎对 loss 进行反向传播
            loss_local = torch.add(t1, t2)
            if sparse:
                loss_local = torch.sparse.sum(loss_local)
            else:
                loss_local = loss_local.sum()
            loss_local.backward()
            # 检查 t1 和 t2 的梯度是否不为 None
            self.assertIsNotNone(t1.grad)
            self.assertIsNotNone(t2.grad)

            # 记录当前 t1 和 t2 的梯度
            t1_grad_before = t1.grad
            t2_grad_before = t2.grad
            # 在给定的上下文 ID 下再次进行反向传播，计算梯度
            dist_autograd.backward(context_id, [loss])
            # 检查 t1 和 t2 的梯度是否与之前记录的一致
            self.assertEqual(t1_grad_before, t1.grad)
            self.assertEqual(t2_grad_before, t2.grad)

    # 当前排名首先在 rref_owner 上创建张量，然后传递
    # 将两个张量 t1 和 t2 相加，结果存储在 local_ret 中
    local_ret = torch.add(t1, t2)

    # 如果 sparse 为 True，则对稀疏张量进行求和操作
    if sparse:
        local_ret = torch.sparse.sum(local_ret)
    else:
        # 否则对稠密张量进行求和操作
        local_ret = local_ret.sum()

    # 对 local_ret 执行反向传播
    local_ret.backward()

    # 使用 dist_autograd.context() 创建上下文 id
    with dist_autograd.context() as context_id:
        # 根据 sparse 的值选择创建远程的稀疏张量或稠密张量 rref_t1
        if sparse:
            rref_t1 = rpc.remote(
                rref_owner, build_sparse_tensor, args=(False, True,)
            )
        else:
            rref_t1 = rpc.remote(
                rref_owner, _torch_ones, args=((3, 3),), kwargs={"requires_grad": True}
            )

        # 根据 callee 是否为 rref_owner，选择调用 my_rref_add 或 my_nested_rref_add
        if callee == rref_owner:
            rref = rpc.remote(callee, my_rref_add, args=(rref_t1, t2))
        else:
            rref = rpc.remote(
                callee, my_nested_rref_add, args=(rref_owner, rref_t1, t2)
            )

        # 将远程计算结果从 rref 拉回本地
        ret = rref.to_here()

        # 根据 sparse 的值对 ret 进行求和操作
        if sparse:
            ret = torch.sparse.sum(ret)
        else:
            ret = ret.sum()

        # 在 dist_autograd 上下文中执行反向传播，计算梯度
        dist_autograd.backward(context_id, [ret])

        # 在调用者端验证梯度是否正确
        grads = dist_autograd.get_gradients(context_id)
        self.assertIn(t2, grads)
        self.assertEqual(grads[t2], t2.grad)

        # 在 rref 持有者端验证梯度是否正确
        self.assertTrue(
            rpc.rpc_sync(
                rref_owner,
                _compare_owner_value,
                args=(context_id, rref_t1, t1.grad),
            )
        )

    # 在此测试中，每个 rank 充当参数服务器（ps）和驱动程序，并启动其他三个 rank 的训练器。
    # 因此，我们有以下关系：
    # ps = rank0，训练器 = rank1/2/3
    # ps = rank2，训练器 = rank2/3/0
    # ps = rank3，训练器 = rank3/0/1
    # ps = rank4，训练器 = rank0/1/2
    #
    # 这四组 ps-trainer 测试在完全独立的 autograd 图上运行，但它们共享相同的 RpcAgents 集合。
    # 定义测试方法，用于测试分布式训练过程中的参数服务器功能
    def _test_trainer_ps(self, create_ref_fn, trainer_fn, sparse):
        # 如果 sparse 为 True，创建稀疏张量 t1 和 t2，并标记需要梯度
        if sparse:
            t1 = build_sparse_tensor(requires_grad=True)
            t2 = build_sparse_tensor(requires_grad=True)
        else:
            # 否则创建普通张量 t1 和 t2，形状为 (3, 3)，并标记需要梯度
            t1 = torch.ones((3, 3), requires_grad=True)
            t2 = torch.zeros((3, 3), requires_grad=True)

        # 对 t1 和 t2 执行元素级的加法操作，结果保存在 local_ret 中
        local_ret = torch.add(t1, t2)
        # 如果 sparse 为 True，对稀疏张量 local_ret 执行 sparse.sum() 操作，并反向传播梯度
        if sparse:
            torch.sparse.sum(local_ret).backward()
        else:
            # 否则对 local_ret 执行 sum() 操作，并反向传播梯度
            local_ret.sum().backward()

        # 在当前 worker 上创建 t1 的远程引用 rref_t1
        rref_t1 = rpc.remote(
            worker_name(self.rank),
            create_ref_fn,
            args=())

        # 向三个其他 worker 发起远程过程调用（RPC），执行训练函数 trainer_fn
        rank_diffs = [1, 2, 3]
        futures = []
        for rank_diff in rank_diffs:
            futures.append(
                rpc.rpc_async(
                    worker_name((self.rank + rank_diff) % self.world_size),
                    trainer_fn,
                    args=(rref_t1, t2, worker_name(self.rank), rank_diff, sparse),
                )
            )

        # 检查所有训练 worker 是否完成了其 backward pass
        for rank_diff in rank_diffs:
            self._check_rpc_done(rank_diff)

        # 所有训练 worker 完成，并持有上下文以供验证
        accumulate_grad_func = None
        for rank_diff in rank_diffs:
            # 确保梯度已经累积到相同的张量上，并且值都是正确的
            ctx_id = ctx_ids[rank_diff]  # ctx_ids 未定义在代码中
            grads = dist_autograd.get_gradients(ctx_id)  # 获取梯度信息
            local_t1 = rref_t1.to_here()  # 将远程引用 rref_t1 拉取到当前 worker
            self.assertIn(local_t1, grads)  # 断言 local_t1 在梯度字典中
            self.assertEqual(grads[local_t1], t1.grad)  # 断言梯度值与 t1.grad 相等

        # 解除训练 worker 的阻塞状态
        _set_rpc_done(None, 0)

        # 等待所有训练 worker 完成
        torch.futures.wait_all(futures)
    # 在不同的执行模式下循环执行多次 RPC 通信。
    def _backward_multiple_round_trips(self, t1, t2, t3, t4, t5, local_grads, sparse):
        # 遍历执行模式，包括本地执行、同步 RPC 和远程执行
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            # 使用分布式自动求导的上下文管理器
            with dist_autograd.context() as context_id:
                # 使用指定执行模式调用 _exec_func 方法执行 torch.add 操作
                val = self._exec_func(exec_mode, torch.add, t1, t2)
                # 使用指定执行模式调用 _exec_func 方法执行 torch.mul 操作
                val = self._exec_func(exec_mode, torch.mul, t3, val)
                # 使用指定执行模式调用 _exec_func 方法执行 torch.stack 操作
                s1 = self._exec_func(exec_mode, torch.stack, (t4, val))
                # 使用指定执行模式调用 _exec_func 方法执行 torch.stack 操作
                s2 = self._exec_func(exec_mode, torch.stack, (t5, val))
                # 根据 sparse 参数选择计算路径
                if sparse:
                    # 使用指定执行模式调用 _exec_func 方法执行 torch.mul 操作
                    val = self._exec_func(exec_mode, torch.mul, s1, s2)
                    # 进行两次平方操作
                    val = self._exec_func(exec_mode, torch.mul, val, val)
                    # 计算稀疏张量的和作为损失
                    loss = torch.sparse.sum(val)
                else:
                    # 使用指定执行模式调用 _exec_func 方法执行 torch.bmm 操作
                    val = self._exec_func(exec_mode, torch.bmm, s1, s2)
                    # 使用指定执行模式调用 _exec_func 方法执行 torch.matmul 操作
                    val = self._exec_func(exec_mode, torch.matmul, val, val)
                    # 计算张量的和作为损失
                    loss = val.sum()

                # 调用 _verify_backwards 方法验证反向传播结果，并更新本地梯度
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2, t3, t4, t5
                )
                # 如果验证结果不为空，则更新本地梯度
                local_grads = ret if ret else local_grads

    # 在不同数据类型下进行反向传播的计算
    def _backward_different_dtypes(self, t1, t2, sparse):
        local_grads = None
        # 遍历执行模式，包括本地执行和远程执行
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            # 使用分布式自动求导的上下文管理器
            with dist_autograd.context() as context_id:
                # 使用指定执行模式调用 _exec_func 方法执行 torch.add 操作
                loss = self._exec_func(exec_mode, torch.add, t1, t2)
                # 根据 sparse 参数选择计算路径
                if sparse:
                    # 计算稀疏张量的和作为损失
                    loss = torch.sparse.sum(loss)
                else:
                    # 计算张量的和作为损失
                    loss = loss.sum()
                # 调用 _verify_backwards 方法验证反向传播结果，并更新本地梯度
                local_grads = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )

    # 在简单的 Python UDF 下进行反向传播的计算，并验证梯度是否一致
    def _backward_simple_python_udf(self, t1, t2, sparse):
        local_grads = None
        # 遍历执行模式，包括本地执行和远程执行
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            # 使用分布式自动求导的上下文管理器
            with dist_autograd.context() as context_id:
                # 使用指定执行模式调用 _exec_func 方法执行自定义函数 my_py_add 操作
                ret = self._exec_func(exec_mode, my_py_add, t1, t2)
                # 根据 sparse 参数选择计算路径
                if sparse:
                    # 计算稀疏张量的和作为损失
                    loss = torch.sparse.sum(ret)
                else:
                    # 计算张量的和作为损失
                    loss = ret.sum()
                # 调用 _verify_backwards 方法验证反向传播结果，并更新本地梯度
                local_grads = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )
    # 在当前对象上执行简单脚本调用的反向传播
    def _backward_simple_script_call(self, t1, t2, sparse):
        # 初始化本地梯度为空
        local_grads = None
        # 遍历不同的执行模式：本地、RPC同步、RPC异步、远程
        for exec_mode in [
            ExecMode.LOCAL,
            ExecMode.RPC_SYNC,
            ExecMode.RPC_ASYNC,
            ExecMode.REMOTE,
        ]:
            # 使用分布式自动微分的上下文环境
            with dist_autograd.context() as context_id:
                # 在指定执行模式下执行函数，并返回前向传播结果
                forward_ret = self._exec_func(exec_mode, my_script_add, t1, t2)
                # 根据稀疏标志选择计算损失的方法
                if sparse:
                    loss = torch.sparse.sum(forward_ret)
                else:
                    loss = forward_ret.sum()
                # 验证反向传播结果，并更新本地梯度
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )
                local_grads = ret if ret else local_grads

    # 嵌套执行梯度累积的反向传播
    def _nested_backward_accumulate_grads(self, t1, t2, sparse):
        # 使用分布式自动微分的上下文环境
        with dist_autograd.context() as context_id:
            # 使用RPC同步调用远程函数，获取返回值ret
            ret = rpc.rpc_sync(
                worker_name(self._next_rank()),
                DistAutogradTest._test_nested_backward_accumulate_grads,
                args=(t1, t2, self._next_rank()),
            )
            # 根据稀疏标志选择计算损失的方法
            if sparse:
                loss = torch.sparse.sum(ret)
            else:
                loss = ret.sum()
            # 执行两次反向传播
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            dist_autograd.backward(context_id, [loss])

    # 使用嵌套Python用户定义函数的反向传播
    def _backwards_nested_python_udf(self, t1, t2, sparse):
        # 计算两个张量的乘积和相加结果
        t3 = t1 * t2
        t4 = t1 + t2
        # 计算最终结果
        res = t3 + t4
        # 计算损失函数
        loss = t1 * t2 * t3 * t4 * res
        # 根据稀疏标志选择计算损失的方法
        if sparse:
            loss = torch.sparse.sum(loss)
        else:
            loss = loss.sum()
        # 执行反向传播
        torch.autograd.backward([loss])

        # 运行分布式自动微分
        with dist_autograd.context() as context_id:
            # 使用RPC同步调用远程函数，并传入参数t1, t2, self._next_rank()
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()),
                DistAutogradTest._nested_python_udf,
                args=(t1, t2, self._next_rank()),
            )
            # 根据稀疏标志选择计算损失的方法
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            # 执行分布式反向传播，并获取梯度
            dist_autograd.backward(context_id, [loss])
            grads = dist_autograd.get_gradients(context_id)
            # 断言梯度结果是否正确
            self.assertEqual(t1.grad, grads[t1])
            self.assertEqual(t2.grad, grads[t2])
    # 检查在两种执行模式下的混合梯度需求
    def _mixed_requires_grad(self, t1, t2, sparse):
        # 遍历两种执行模式：RPC_SYNC 和 REMOTE
        for exec_mode in [ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            # 创建分布式自动求导的上下文管理器
            with dist_autograd.context() as context_id:
                # 执行函数，并获取返回结果
                ret = self._exec_func(
                    exec_mode, DistAutogradTest._mixed_requires_grad_operaton, t1, t2
                )
                # 断言 t1 * t2 等于返回结果 ret
                self.assertEqual(t1 * t2, ret)
                # 根据稀疏标志计算损失
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    loss = ret.sum()
                # 执行分布式自动求导的反向传播
                dist_autograd.backward(context_id, [loss])
                # 确认 t1 需要梯度，t2 不需要梯度
                self.assertTrue(t1.requires_grad)
                self.assertFalse(t2.requires_grad)
                # 获取梯度信息
                grads = dist_autograd.get_gradients(context_id)
                # 断言 t1 在梯度字典中，t2 不在梯度字典中
                self.assertIn(t1, grads)
                self.assertNotIn(t2, grads)
                # 断言 t1 的梯度等于 t2
                self.assertEqual(t2, grads[t1])

    # 执行多次反向传播，并保留计算图
    def _multiple_backward(self, t1, t2, sparse):
        # 创建分布式自动求导的上下文管理器
        with dist_autograd.context() as context_id:
            # 在远程 RPC 上同步执行 torch.add 函数
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()),
                torch.add,
                args=(t1, t2))
            # 根据稀疏标志计算损失
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            # 多次执行反向传播，并保留计算图
            for i in range(1000):
                dist_autograd.backward(context_id, [loss], retain_graph=True)

    # 验证第一个 RPC 调用的计算图结构
    # 对于当前上下文，在此排发送 t1 和 t2 张量给目标排，
    # 然后获取 t3 = torch.add(t1, t2) 的结果张量。
    # 对于此排的当前上下文，期望的计算图如下：
    #  发送函数:
    #              rpcSendBackward
    #                  /          \
    #  t1.AccumulateGrad         t2.AccumulateGrad
    #
    #  接收函数:
    #
    #            |
    #          t3.rpcRecvBackward
    #
    def _verify_graph_for_first_rpc_call(
        self, send_function, recv_function, t1, t2, ret
    ):
        # 检索计算图中的下一个函数
        next_funcs = send_function.next_functions
        # 断言下一个函数的数量为2
        self.assertEqual(2, len(next_funcs))

        # 在自动求导图中应当找到 t1 和 t2
        self.assertEqual("torch::autograd::AccumulateGrad", next_funcs[0][0].name())
        self.assertEqual(t1, next_funcs[0][0].variable)
        self.assertEqual(0, next_funcs[0][1])
        self.assertEqual("torch::autograd::AccumulateGrad", next_funcs[1][0].name())
        self.assertEqual(t2, next_funcs[1][0].variable)
        self.assertEqual(0, next_funcs[1][1])

        # 测试接收函数
        self.assertEqual(ret.grad_fn, recv_function)

    # 在本地和分布式自动求导中运行相同的代码，并验证梯度是否相同。
    # 定义一个方法 `_backward_simple`，接受参数 `dst`, `t1`, `t2`, `local_grads`, `sparse`
    def _backward_simple(self, dst, t1, t2, local_grads, sparse):
        # 遍历执行模式：LOCAL、RPC_SYNC、REMOTE
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            # 使用 `dist_autograd.context()` 创建上下文环境，并获取上下文ID
            with dist_autograd.context() as context_id:
                # 调用 `_exec_func_with_dst` 方法执行 `torch.add(t1, t2)`，并返回结果 `ret`
                ret = self._exec_func_with_dst(
                    dst, exec_mode, torch.add, t1, t2
                )
                # 如果 `sparse` 为真，计算稀疏张量 `ret` 的和作为损失 `loss`
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    # 否则，计算 `ret` 的总和作为损失 `loss`
                    loss = ret.sum()
                # 调用 `_verify_backwards` 方法验证反向传播，传入 `exec_mode`、损失列表 `[loss]`、
                # 上下文ID、本地梯度 `local_grads`、`t1` 和 `t2`，并获取返回值 `ret`
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )
                # 如果 `ret` 不为空，更新 `local_grads`；否则保持不变
                local_grads = ret if ret else local_grads

    # 对于从前一个嵌套链式调用传递的上下文，在此排名中，接收两个张量 `t1` 和 `t2`，
    # 执行 `torch.add(t1, t2)` 并发送结果张量 `t3` 回去。
    # 对于此排名中的上下文，期望的图形如下：
    #  发送和接收函数：
    #       rpcSendBackward
    #           |
    #          t3.AddBackward0
    #          /             \
    # t1.recvRpcBackward    t2.recvRpcBackward
    def _verify_graph_for_rpc_call_exec(self, send_function):
        # 验证下一个函数是 `AddBackward0`
        next_funcs = send_function.next_functions
        self.assertEqual(1, len(next_funcs))
        add_backward_fn = next_funcs[0][0]
        self.assertEqual("AddBackward0", add_backward_fn.name())

        # 验证下两个函数是相同的接收反向函数
        next_funcs = add_backward_fn.next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
        )
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[1][0].name()
        )
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])

    # 对于从前一个嵌套链式调用传递的上下文，在此排名中，接收两个张量 `t1` 和 `t2`，
    # 使用嵌套的 RPC 调用将 `t1` 和 `t2` 张量转发到下一个目标 `dst`。
    # 在返回路由中，从下一个目标接收结果张量 `t3` 并将其转发回到之前的调用。
    # 对于此排名中的上下文，期望的图形如下：
    #  用于接收和转发 `t1` 和 `t2` 的发送和接收函数：
    #       rpcSendBackward
    #          /          \
    # t1.recvRpcBackward    t2.recvRpcBackward
    #  用于接收和转发 `t3` 的发送和接收函数：
    #       rpcSendBackward
    #             |
    #           t3.recvRpcBackward
    # 验证嵌套的远程过程调用（RPC）图中的发送函数
    send_functions = ctx._send_functions()
    # 断言发送函数的数量为2
    self.assertEqual(2, len(send_functions))

    # 对于发送函数，在进行嵌套 RPC 调用时，
    # 发送函数的下一个函数应该是两个接收函数，
    # 用于接收前一调用返回的两个张量
    next_funcs = next(iter(send_functions.values())).next_functions
    self.assertEqual(2, len(next_funcs))
    self.assertEqual(
        "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
    )
    self.assertEqual(
        "torch::distributed::autograd::RecvRpcBackward", next_funcs[1][0].name()
    )
    self.assertEqual(next_funcs[0][0], next_funcs[1][0])

    # 对于发送函数，在返回响应给前一调用时，
    # 发送函数的下一个函数应该是接收函数，
    # 用于接收嵌套调用返回的张量结果
    next_funcs = list(send_functions.values())[1].next_functions
    self.assertEqual(1, len(next_funcs))
    self.assertEqual(
        "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
    )
# 定义一个测试类，继承自 CommonDistAutogradTest 类
class TensorPipeAgentDistAutogradTest(CommonDistAutogradTest):

    # 使用装饰器 dist_init 标记下面的方法作为分布式初始化的测试函数
    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的内置函数调用图
    def test_graph_for_builtin_call_sparse(self):
        self._test_graph(torch.add, ExecMode.RPC_SYNC, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的自定义 Python 函数调用图
    def test_graph_for_python_call_sparse(self):
        self._test_graph(my_py_add, ExecMode.RPC_SYNC, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的内置函数远程调用图
    def test_graph_for_builtin_remote_call_sparse(self):
        self._test_graph(torch.add, ExecMode.REMOTE, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的自定义 Python 函数远程调用图
    def test_graph_for_python_remote_call_sparse(self):
        self._test_graph(my_py_add, ExecMode.REMOTE, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的嵌套 Python 函数调用图
    def test_graph_for_py_nested_call_sparse(self):
        self._test_graph_for_py_nested_call(ExecMode.RPC_SYNC, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的嵌套 Python 函数远程调用图
    def test_graph_for_py_nested_remote_call_sparse(self):
        self._test_graph_for_py_nested_call(ExecMode.REMOTE, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的嵌套 Python 函数自身调用图
    def test_graph_for_py_nested_call_itself_sparse(self):
        self._test_graph_for_py_nested_call_itself(ExecMode.RPC_SYNC, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的嵌套 Python 函数自身远程调用图
    def test_graph_for_py_nested_remote_call_itself_sparse(self):
        self._test_graph_for_py_nested_call_itself(ExecMode.REMOTE, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的不需要梯度的张量上下文清理
    def test_no_graph_with_tensors_not_require_grad_sparse(self):
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.RPC_SYNC, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的不需要梯度的张量远程上下文清理
    def test_no_graph_with_tensors_not_require_grad_remote_sparse(self):
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.REMOTE, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的复杂 RPC 参数
    def test_rpc_complex_args_sparse(self):
        self._test_rpc_complex_args(ExecMode.RPC_SYNC, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的复杂远程参数
    def test_remote_complex_args_sparse(self):
        self._test_rpc_complex_args(ExecMode.REMOTE, True)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的张量梯度上下文清理
    def test_context_cleanup_tensor_with_grad_sparse(self):
        t1 = build_sparse_tensor(requires_grad=True)
        t2 = build_sparse_tensor(requires_grad=True)
        self.context_cleanup_test_helper(rpc_args=(t1, t2), func=torch.add)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的无梯度张量上下文清理
    def test_context_cleanup_tensor_no_grad_sparse(self):
        t1 = build_sparse_tensor(requires_grad=False)
        self.context_cleanup_test_helper(rpc_args=(t1, t1), func=torch.add)

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的嵌套 RPC 上下文清理
    def test_context_cleanup_nested_rpc_sparse(self):
        t1 = build_sparse_tensor(requires_grad=True)
        t2 = build_sparse_tensor(requires_grad=True)
        dst_rank = (self.rank + 1) % self.world_size
        args = (t1, t2, dst_rank, self.world_size, 0)
        self.context_cleanup_test_helper(
            rpc_args=args, func=my_py_nested_call, nested=True
        )

    @dist_init
    # 测试使用 TensorPipeAgent 执行稀疏张量的无梯度张量反向传播
    def test_backward_no_grad_on_tensor_sparse(self):
        self._backward_no_grad_on_tensor(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True
        )

    @dist_init
    # 继续上面的测试
    def test_backward_no_grad_on_tensor_remote_sparse(self):
        self._backward_no_grad_on_tensor(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True
        )
    # 定义一个测试方法，用于测试简单稀疏张量的反向传播
    def test_backward_simple_sparse(self):
        # 调用 _backward_simple 方法，传入以下参数：
        # - self._next_rank(): 获取下一个排名
        # - build_sparse_tensor(requires_grad=True): 构建一个需要梯度的稀疏张量
        # - build_sparse_tensor(requires_grad=True): 构建另一个需要梯度的稀疏张量
        # - None: 空值，表示没有额外的参数
        # - True: 布尔值参数，表示真值
        self._backward_simple(
            self._next_rank(),
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True
        )

    # 使用 @dist_init 装饰器初始化分布式测试环境
    @dist_init
    # 定义一个测试方法，用于测试简单自身稀疏张量的反向传播
    def test_backward_simple_self_sparse(self):
        # 调用 _backward_simple 方法，传入以下参数：
        # - self.rank: 当前进程的排名
        # - build_sparse_tensor(requires_grad=True): 构建一个需要梯度的稀疏张量
        # - build_sparse_tensor(requires_grad=True): 构建另一个需要梯度的稀疏张量
        # - None: 空值，表示没有额外的参数
        # - True: 布尔值参数，表示真值
        self._backward_simple(
            self.rank,
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True
        )

    # 使用 @dist_init 装饰器初始化分布式测试环境
    @dist_init
    # 定义一个测试方法，用于测试多个稀疏张量的远程引用反向传播
    def test_backward_rref_multi_sparse(self):
        # 如果当前进程的排名大于 0，则执行以下代码块
        if self.rank > 0:
            callee = "worker0"  # 设置被调用者名称
            rref_owner = callee  # 设置远程引用所有者名称
            # 调用 _backward_rref 方法，传入以下参数：
            # - callee: 被调用者名称
            # - rref_owner: 远程引用所有者名称
            # - build_sparse_tensor(requires_grad=True): 构建一个需要梯度的稀疏张量
            # - build_sparse_tensor(requires_grad=True): 构建另一个需要梯度的稀疏张量
            # - None: 空值，表示没有额外的参数
            # - True: 布尔值参数，表示真值
            self._backward_rref(
                callee,
                rref_owner,
                build_sparse_tensor(requires_grad=True),
                build_sparse_tensor(requires_grad=True),
                None,
                True
            )

    # 使用 @dist_init 装饰器初始化分布式测试环境
    @dist_init
    # 定义一个测试方法，用于测试稀疏张量的远程引用反向传播
    def test_backward_rref_sparse(self):
        # 根据下一个排名获取工作进程的名称
        callee = worker_name(self._next_rank())
        rref_owner = callee  # 远程引用所有者名称与被调用者名称相同
        # 调用 _backward_rref 方法，传入以下参数：
        # - callee: 工作进程的名称
        # - rref_owner: 远程引用所有者名称
        # - build_sparse_tensor(requires_grad=True): 构建一个需要梯度的稀疏张量
        # - build_sparse_tensor(requires_grad=True): 构建另一个需要梯度的稀疏张量
        # - None: 空值，表示没有额外的参数
        # - True: 布尔值参数，表示真值
        self._backward_rref(
            callee,
            rref_owner,
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True
        )

    # 使用 @dist_init 装饰器初始化分布式测试环境
    @dist_init
    # 定义一个测试方法，用于测试嵌套稀疏张量的远程引用反向传播
    def test_backward_rref_nested_sparse(self):
        # 根据 (self.rank + 1) % self.world_size 计算被调用者名称
        callee = worker_name((self.rank + 1) % self.world_size)
        # 根据 (self.rank + 2) % self.world_size 计算远程引用所有者名称
        rref_owner = worker_name((self.rank + 2) % self.world_size)
        # 调用 _backward_rref 方法，传入以下参数：
        # - callee: 被调用者名称
        # - rref_owner: 远程引用所有者名称
        # - build_sparse_tensor(requires_grad=True): 构建一个需要梯度的稀疏张量
        # - build_sparse_tensor(requires_grad=True): 构建另一个需要梯度的稀疏张量
        # - None: 空值，表示没有额外的参数
        # - True: 布尔值参数，表示真值
        self._backward_rref(
            callee,
            rref_owner,
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True
        )

    # 使用 @dist_init 装饰器初始化分布式测试环境
    @dist_init
    # 定义一个测试方法，用于测试参数服务器训练器与稀疏张量的结合
    def test_trainer_ps_sparse(self):
        # 调用 _test_trainer_ps 方法，传入以下参数：
        # - build_sparse_tensor: 构建稀疏张量的函数
        # - _run_trainer: 运行训练器的函数
        # - True: 布尔值参数，表示真值
        self._test_trainer_ps(
            build_sparse_tensor,
            _run_trainer,
            True
        )

    # 使用 @dist_init 装饰器初始化分布式测试环境
    @dist_init
    # 定义一个测试方法，用于测试多次往返反向传播与稀疏张量的结合
    def test_backward_multiple_round_trips_sparse(self):
        # 调用 _backward_multiple_round_trips 方法，传入以下参数：
        # - build_sparse_tensor(requires_grad=True): 构建一个需要梯度的稀疏张量
        # - build_sparse_tensor(requires_grad=False): 构建一个不需要梯度的稀疏张量
        # - build_sparse_tensor(requires_grad=True): 构建另一个需要梯度的稀疏张量
        # - build_sparse_tensor(requires_grad=False): 构建另一个不需要梯度的稀疏张量
        # - build_sparse_tensor(requires_grad=True): 构建另一个需要梯度的稀疏张量
        # - None: 空值，表示没有额外的参数
        # - True: 布尔值参数，表示真值
        self._backward_multiple_round_trips(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=False),
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=False),
            build_sparse_tensor(requires_grad=True),
            None,
            True
        )

    # 使用 @dist_init 装饰器初始化分布式测试环境
    @dist_init
    # 定义一个测试方法，用于测试不同数据类型的稀疏张量的反向传播
    def test_backward_different_dtypes_sparse(self):
        # 调用 _backward_different_dtypes 方法，传入以下参数：
        # - build_sparse_tensor(requires_grad=True, dtype=torch.float32): 构建一个需要梯度的单精度浮点数稀疏张量
        # - build_sparse_tensor(requires_grad=True, dtype=torch.float64): 构建一个需要梯度的双精度浮点数稀疏张
    # 调用 _backward_simple_script_call 方法，传入两个需要梯度的稀疏张量及参数 True
    def test_backward_simple_script_call_sparse(self):
        self._backward_simple_script_call(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True
        )

    # 使用 dist_init 装饰器初始化分布式环境，调用 _nested_backward_accumulate_grads 方法
    # 传入两个需要梯度的稀疏张量及参数 True
    @dist_init
    def test_nested_backward_accumulate_grads_sparse(self):
        self._nested_backward_accumulate_grads(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True
        )

    # 使用 dist_init 装饰器初始化分布式环境，运行本地等效于 _nested_python_udf 的方法
    # 调用 _backwards_nested_python_udf，传入两个需要梯度的稀疏张量及参数 True
    @dist_init
    def test_backwards_nested_python_udf_sparse(self):
        self._backwards_nested_python_udf(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True
        )

    # 使用 dist_init 装饰器初始化分布式环境，调用 _mixed_requires_grad 方法
    # 传入一个需要梯度的稀疏张量和一个不需要梯度的稀疏张量，及参数 True
    @dist_init
    def test_mixed_requires_grad_sparse(self):
        self._mixed_requires_grad(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=False),
            True
        )

    # 使用 dist_init 装饰器初始化分布式环境，调用 _multiple_backward 方法
    # 传入两个需要梯度的稀疏张量及参数 True
    @dist_init
    def test_multiple_backward_sparse(self):
        self._multiple_backward(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True
        )

    # 使用 dist_init 装饰器初始化分布式环境，测试不需要梯度的张量的嵌入包处理
    def test_embedding_bag_with_no_grad_tensors(self):
        # 获取下一个排名的目标
        dst = self._next_rank()
        # 在远程工作节点上创建一个远程嵌入包对象
        remote_embedding = rpc.remote(
            worker_name(dst),
            torch.nn.EmbeddingBag,
            args=(16, 16),
            kwargs={"mode": "sum", "sparse": True},
        )
        # 在本地创建一个嵌入包对象
        local_embedding = torch.nn.EmbeddingBag(16, 16, mode="sum", sparse=True)

        # 输入张量
        input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
        # 为了记录发送/接收函数，设置 requires_grad=True
        per_sample_weights = torch.rand((8), requires_grad=True)
        # 偏移量张量
        offsets = torch.LongTensor([0, 4])

        # 在本地执行嵌入包操作
        local_res = local_embedding(input, offsets, per_sample_weights)

        # 运行两次反向传播
        torch.autograd.backward([local_res.sum()], retain_graph=True)
        torch.autograd.backward([local_res.sum()])
        # 获取本地嵌入包权重的梯度
        local_grad = local_embedding.weight.grad

        # 使用 dist_autograd.context() 上下文管理器
        with dist_autograd.context() as context_id:
            # 在远程工作节点上同步调用远程嵌入包对象的方法
            res = rpc.rpc_sync(
                worker_name(dst),
                DistAutogradTest._call_remote_embedding,
                args=(remote_embedding, input, offsets, per_sample_weights),
            )

            # 运行两次反向传播以测试稀疏梯度的累积
            dist_autograd.backward(context_id, [res.sum()], retain_graph=True)
            dist_autograd.backward(context_id, [res.sum()])

            # 在远程工作节点上同步获取远程嵌入包对象的梯度
            remote_grad = rpc.rpc_sync(
                worker_name(dst),
                DistAutogradTest._get_grad,
                args=(remote_embedding, context_id),
            )

            # 断言本地嵌入包的梯度与远程嵌入包的梯度相等
            self.assertEqual(local_grad, remote_grad)
# 继承自 CommonDistAutogradTest 类的 DistAutogradTest 类
class DistAutogradTest(CommonDistAutogradTest):

    # 使用 dist_init 装饰器初始化测试函数 test_autograd_context
    @dist_init
    def test_autograd_context(self):
        # 设置最大可能的自增 ID
        max_auto_increment = 281474976710655
        # 断言最大自增 ID 加上当前 worker_id 左移 48 位后的结果与 dist_autograd._get_max_id() 相等
        self.assertEqual(
            max_auto_increment + (self.worker_id << 48), dist_autograd._get_max_id()
        )

        # 初始化上下文 ID 列表
        context_ids = []
        # 循环 200 次
        for i in range(200):
            # 使用 dist_autograd.context() 创建上下文管理器，并获取 context_id
            with dist_autograd.context() as context_id:
                # 断言当前 context_id 等于从 dist_autograd._retrieve_context(context_id)._context_id() 获取的 context_id
                self.assertEqual(
                    context_id,
                    dist_autograd._retrieve_context(context_id)._context_id(),
                )
                # 断言 context_id 的前 16 位应该是 worker_id
                self.assertEqual(self.worker_id, context_id >> 48)
                # 将 context_id 添加到列表中
                context_ids.append(context_id)

        # 遍历 context_ids 列表中的每一个 context_id
        for context_id in context_ids:
            # 使用 self.assertRaisesRegex 断言在调用 dist_autograd._retrieve_context(context_id) 时会抛出 RuntimeError 异常，
            # 并且异常信息中包含特定的 context_id
            with self.assertRaisesRegex(
                RuntimeError,
                f"Could not find autograd context with id: {context_id}",
            ):
                dist_autograd._retrieve_context(context_id)

    # 使用 dist_init 装饰器初始化测试函数 test_nested_context
    @dist_init
    def test_nested_context(self):
        # 使用 dist_autograd.context() 创建上下文管理器，并获取 context_id
        with dist_autograd.context() as context_id:
            # 断言在嵌套上下文中会抛出 RuntimeError 异常，异常信息中包含特定文本
            with self.assertRaisesRegex(
                RuntimeError, "Already have an autograd context id for this thread"
            ):
                # 再次调用 dist_autograd.context() 创建上下文管理器，此时应该会抛出异常
                with dist_autograd.context() as context_id:
                    pass

    # 使用 dist_init 装饰器初始化测试函数 test_graph_for_builtin_call
    @dist_init
    def test_graph_for_builtin_call(self):
        # 调用 _test_graph 方法测试 torch.add 函数的图生成，同步执行，不远程
        self._test_graph(torch.add, ExecMode.RPC_SYNC, False)

    # 使用 dist_init 装饰器初始化测试函数 test_graph_for_python_call
    @dist_init
    def test_graph_for_python_call(self):
        # 调用 _test_graph 方法测试 my_py_add 函数的图生成，同步执行，不远程
        self._test_graph(my_py_add, ExecMode.RPC_SYNC, False)

    # 使用 dist_init 装饰器初始化测试函数 test_graph_for_builtin_remote_call
    @dist_init
    def test_graph_for_builtin_remote_call(self):
        # 调用 _test_graph 方法测试 torch.add 函数的图生成，远程执行，不同步
        self._test_graph(torch.add, ExecMode.REMOTE, False)

    # 使用 dist_init 装饰器初始化测试函数 test_graph_for_python_remote_call
    @dist_init
    def test_graph_for_python_remote_call(self):
        # 调用 _test_graph 方法测试 my_py_add 函数的图生成，远程执行，不同步
        self._test_graph(my_py_add, ExecMode.REMOTE, False)

    # 使用 dist_init 装饰器初始化测试函数 test_graph_for_py_nested_call
    @dist_init
    def test_graph_for_py_nested_call(self):
        # 调用 _test_graph_for_py_nested_call 方法测试 Python 嵌套调用的图生成，同步执行，不远程
        self._test_graph_for_py_nested_call(ExecMode.RPC_SYNC, False)

    # 使用 dist_init 装饰器初始化测试函数 test_graph_for_py_nested_remote_call
    @dist_init
    def test_graph_for_py_nested_remote_call(self):
        # 调用 _test_graph_for_py_nested_call 方法测试 Python 嵌套调用的图生成，远程执行，不同步
        self._test_graph_for_py_nested_call(ExecMode.REMOTE, False)

    # 使用 dist_init 装饰器初始化测试函数 test_graph_for_py_nested_call_itself
    @dist_init
    def test_graph_for_py_nested_call_itself(self):
        # 调用 _test_graph_for_py_nested_call_itself 方法测试 Python 嵌套自身调用的图生成，同步执行，不远程
        self._test_graph_for_py_nested_call_itself(ExecMode.RPC_SYNC, False)

    # 使用 dist_init 装饰器初始化测试函数 test_graph_for_py_nested_remote_call_itself
    @dist_init
    def test_graph_for_py_nested_remote_call_itself(self):
        # 调用 _test_graph_for_py_nested_call_itself 方法测试 Python 嵌套自身调用的图生成，远程执行，不同步
        self._test_graph_for_py_nested_call_itself(ExecMode.REMOTE, False)

    # 使用 dist_init 装饰器初始化测试函数 test_no_graph_with_tensors_not_require_grad
    @dist_init
    def test_no_graph_with_tensors_not_require_grad(self):
        # 调用 _test_no_graph_with_tensors_not_require_grad 方法测试不需要梯度的张量的图生成，同步执行，不远程
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.RPC_SYNC, False)

    # 使用 dist_init 装饰器初始化测试函数 test_no_graph_with_tensors_not_require_grad_remote
    @dist_init
    def test_no_graph_with_tensors_not_require_grad_remote(self):
        # 调用 _test_no_graph_with_tensors_not_require_grad 方法测试不需要梯度的张量的图生成，远程执行，不同步
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.REMOTE, False)
    # 在测试函数中，仅对返回值的梯度进行测试
    def _test_grad_only_on_return_value(self, exec_mode):
        # 初始化分布式组，根据指定的初始化方法、进程排名和总进程数进行初始化
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # 计算目标进程的排名
        dst_rank = (self.rank + 1) % self.world_size
        # 进入分布式自动求导的上下文管理器
        with dist_autograd.context() as context_id:
            # 根据执行模式选择不同的RPC方式获取返回值
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), ret_requires_grad)
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(
                    worker_name(dst_rank), ret_requires_grad
                ).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            # 对返回值进行反向传播
            dist_autograd.backward(context_id, [ret.sum()])

            # 在目标进程上同步RPC调用，标记RPC调用完成
            rpc.rpc_sync(
                worker_name(dst_rank), _set_rpc_done, args=(context_id, 1)
            )

            # 等待前一个进程完成RPC调用
            self._check_rpc_done(1)
            # 获取梯度信息
            grads = dist_autograd.get_gradients(ctx_ids[1])
            # 断言只有一个梯度返回
            self.assertEqual(1, len(grads))
            # 断言所需梯度张量在梯度字典中
            self.assertIn(requires_grad_tensor, grads)
            # 断言梯度张量的值为全1张量
            self.assertEqual(torch.ones_like(ret), grads[requires_grad_tensor])
            # 由于上述get_gradients调用，确保分布式自动求导上下文在所有工作进程退出上下文管理器前不会被清理
            dist.barrier()

    @dist_init
    # 测试RPC同步模式下的梯度仅在返回值上
    def test_grad_only_on_return_value(self):
        self._test_grad_only_on_return_value(ExecMode.RPC_SYNC)

    @dist_init
    # 测试远程执行模式下的梯度仅在返回值上
    def test_grad_only_on_return_value_remote(self):
        self._test_grad_only_on_return_value(ExecMode.REMOTE)

    @dist_init
    # 测试RPC复杂参数
    def test_rpc_complex_args(self):
        self._test_rpc_complex_args(ExecMode.RPC_SYNC, False)

    @dist_init
    # 测试远程执行复杂参数
    def test_remote_complex_args(self):
        self._test_rpc_complex_args(ExecMode.REMOTE, False)

    @dist_init
    # 测试在有梯度的张量情况下的上下文清理
    def test_context_cleanup_tensor_with_grad(self):
        # 创建一个需要梯度的全1张量
        t1 = torch.ones(3, 3, requires_grad=True)
        # 创建一个需要梯度的全0张量
        t2 = torch.zeros(3, 3, requires_grad=True)
        # 使用RPC参数和函数进行上下文清理测试
        self.context_cleanup_test_helper(rpc_args=(t1, t2), func=torch.add)

    @dist_init
    # 测试在无梯度的张量情况下的上下文清理
    def test_context_cleanup_tensor_no_grad(self):
        # 创建一个无需梯度的全1张量
        t1 = torch.ones(3, 3, requires_grad=False)
        # 使用RPC参数和函数进行上下文清理测试
        self.context_cleanup_test_helper(rpc_args=(t1, t1), func=torch.add)

    @dist_init
    # 测试在无张量的情况下的上下文清理
    def test_context_cleanup_no_tensors(self):
        # 使用RPC参数和函数进行上下文清理测试
        self.context_cleanup_test_helper(rpc_args=(1, 1), func=my_scalar_add)

    @dist_init
    # 测试嵌套RPC调用的上下文清理
    def test_context_cleanup_nested_rpc(self):
        # 创建一个需要梯度的全1张量
        t1 = torch.ones(3, 3, requires_grad=True)
        # 创建一个需要梯度的全0张量
        t2 = torch.zeros(3, 3, requires_grad=True)
        # 计算目标进程的排名
        dst_rank = (self.rank + 1) % self.world_size
        # 设置RPC参数和函数进行上下文清理测试，包括嵌套调用
        args = (t1, t2, dst_rank, self.world_size, 0)
        self.context_cleanup_test_helper(
            rpc_args=args, func=my_py_nested_call, nested=True
        )
    def test_worker_ids_recorded(self):
        # 创建排除当前进程rank的目标rank集合
        dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}
        # 在分布式自动求导上下文中执行以下代码块
        with dist_autograd.context() as context_id:
            # 如果没有张量需要梯度，仍然记录worker_ids，
            # 因为autograd上下文ID仍会传递给其他worker。
            t1 = torch.ones(3, 3, requires_grad=False)
            t2 = torch.zeros(3, 3, requires_grad=False)
            for dst_rank in dst_ranks:
                # 在目标rank上同步执行torch.add操作
                rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
                # 在目标rank上同步执行_set_rpc_done函数
                rpc.rpc_sync(
                    worker_name(dst_rank), _set_rpc_done, args=(context_id, 1)
                )
            # dst_ranks中所有worker_ids应该被记录。
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(worker_ids, dst_ranks)

            # 当张量需要梯度时应该记录worker_ids
            t1.requires_grad = True
            t2.requires_grad = True
            for dst_rank in dst_ranks:
                # 在目标rank上同步执行torch.add操作
                ret = rpc.rpc_sync(
                    worker_name(dst_rank), torch.add, args=(t1, t2)
                )
                # 在目标rank上同步执行_set_rpc_done函数
                rpc.rpc_sync(
                    worker_name(dst_rank), _set_rpc_done, args=(context_id, 1)
                )
            # dst_ranks中所有worker_ids应该被记录。
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(worker_ids, dst_ranks)

    @dist_init
    def test_dist_autograd_profiling(self):
        # 在分布式自动求导上下文中执行以下代码块
        with dist_autograd.context() as context_id:
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(3, 3, requires_grad=True)
            # 在下一个rank的worker上执行torch.add，并计算总和作为损失
            loss = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2)).sum()
            # 使用torch.autograd.profiler.profile()进行性能分析
            with torch.autograd.profiler.profile() as p:
                dist_autograd.backward(context_id, [loss])

        function_events = p.function_events

        def get_event(partial_key):
            # 获取事件列表中包含指定关键字的事件
            return next(event for event in function_events if partial_key in event.name)

        # 获取特定事件
        send_event = get_event("SendRpcBackward")
        recv_event = get_event("RecvRpcBackward")
        backward_event = get_event("torch::distributed::autograd::backward")
        # 至少应该有一个send和recv事件，对应执行的send/recv函数。
        self.assertEqual(send_event.count, 1)
        self.assertEqual(recv_event.count, 1)
        # 在backward事件的CPU总时间应该大于send和recv事件，
        # 因为在反向传播中应用这些函数是整个反向传播过程的子集。
        self.assertGreater(backward_event.cpu_time_total, send_event.cpu_time_total)
        self.assertGreater(backward_event.cpu_time_total, recv_event.cpu_time_total)
    @dist_init
    # 定义测试函数，用于测试在上下文中发生错误的情况
    def test_error_in_context(self):
        # 创建分布式自动求导的上下文，并获取其上下文 ID
        with dist_autograd.context() as context_id:
            # 创建随机的大小为 3x3 的张量 t1，并设置 requires_grad=True
            t1 = torch.rand(3, 3, requires_grad=True)
            # 创建随机的大小为 6x6 的张量 t2，并设置 requires_grad=True
            t2 = torch.rand(6, 6, requires_grad=True)

            # 使用 RPC 进行同步调用，期望在矩阵尺寸不匹配时抛出 RuntimeError
            with self.assertRaises(RuntimeError):
                # 调用 rpc_sync 函数，传递 worker 的名称和 torch.matmul 函数以及参数 t1 和 t2
                rpc.rpc_sync(
                    worker_name(self._next_rank()), torch.matmul, args=(t1, t2)
                )

    @dist_init
    # 在分布式初始化的情况下，测试在张量上没有梯度时的反向传播
    def test_backward_no_grad_on_tensor(self):
        # 调用 _backward_no_grad_on_tensor 方法，传递一个大小为 3x3 的随机张量，requires_grad=True
        self._backward_no_grad_on_tensor(
            torch.rand((3, 3), requires_grad=True),
            # 再传递一个大小为 3x3 的随机张量，requires_grad=True
            torch.rand((3, 3), requires_grad=True),
            # 设置是否是简单模式为 False
            False
        )

    @dist_init
    # 在分布式初始化的情况下，测试简单的反向传播
    def test_backward_simple(self):
        # 获取下一个 worker 的 rank，并调用 _backward_simple 方法
        self._backward_simple(
            self._next_rank(),
            # 传递一个大小为 3x3 的随机张量，requires_grad=True
            torch.rand((3, 3), requires_grad=True),
            # 再传递一个大小为 3x3 的随机张量，requires_grad=True
            torch.rand((3, 3), requires_grad=True),
            # 传递 None 作为额外参数
            None,
            # 设置是否是简单模式为 False
            False
        )

    @dist_init
    # 在分布式初始化的情况下，测试在当前 worker 上的简单反向传播
    def test_backward_simple_self(self):
        # 获取当前 worker 的 rank，并调用 _backward_simple 方法
        self._backward_simple(
            self.rank,
            # 传递一个大小为 3x3 的随机张量，requires_grad=True
            torch.rand((3, 3), requires_grad=True),
            # 再传递一个大小为 3x3 的随机张量，requires_grad=True
            torch.rand((3, 3), requires_grad=True),
            # 传递 None 作为额外参数
            None,
            # 设置是否是简单模式为 False
            False
        )

    @dist_init
    # 在分布式初始化的情况下，测试使用 RRef 的反向传播
    def test_backward_rref(self):
        # 获取下一个 worker 的名称，并设为 callee 和 rref_owner，然后调用 _backward_rref 方法
        callee = worker_name(self._next_rank())
        rref_owner = callee
        self._backward_rref(
            callee,
            rref_owner,
            # 传递一个大小为 3x3 的随机张量，requires_grad=True
            torch.rand((3, 3), requires_grad=True),
            # 再传递一个大小为 3x3 的随机张量，requires_grad=True
            torch.rand((3, 3), requires_grad=True),
            # 传递 None 作为额外参数
            None,
            # 设置是否是简单模式为 False
            False
        )

    @dist_init
    # 在分布式初始化的情况下，测试使用多个 RRef 的反向传播
    def test_backward_rref_multi(self):
        # 如果当前 worker 的 rank 大于 0，则执行以下代码块
        if self.rank > 0:
            # 将 callee 设为 "worker0"，并将 rref_owner 设为 callee，然后调用 _backward_rref 方法
            callee = "worker0"
            rref_owner = callee
            self._backward_rref(
                callee,
                rref_owner,
                # 传递一个大小为 3x3 的随机张量，requires_grad=True
                torch.rand((3, 3), requires_grad=True),
                # 再传递一个大小为 3x3 的随机张量，requires_grad=True
                torch.rand((3, 3), requires_grad=True),
                # 传递 None 作为额外参数
                None,
                # 设置是否是简单模式为 False
                False
            )

    @dist_init
    # 在分布式初始化的情况下，测试嵌套使用 RRef 的反向传播
    def test_backward_rref_nested(self):
        # 获取下一个 worker 的名称作为 callee，再下一个 worker 的名称作为 rref_owner，然后调用 _backward_rref 方法
        callee = worker_name((self.rank + 1) % self.world_size)
        rref_owner = worker_name((self.rank + 2) % self.world_size)
        self._backward_rref(
            callee,
            rref_owner,
            # 传递一个大小为 3x3 的随机张量，requires_grad=True
            torch.rand((3, 3), requires_grad=True),
            # 再传递一个大小为 3x3 的随机张量，requires_grad=True
            torch.rand((3, 3), requires_grad=True),
            # 传递 None 作为额外参数
            None,
            # 设置是否是简单模式为 False
            False
        )

    @dist_init
    # 在分布式初始化的情况下，测试 PS（参数服务器）的训练器
    def test_trainer_ps(self):
        # 调用 _test_trainer_ps 方法，传递 create_tensor 和 _run_trainer 函数以及 False 作为参数
        self._test_trainer_ps(
            create_tensor,
            _run_trainer,
            False
        )
    def test_trainer_ps_torchscript_functions(self):
        """
        # TODO, need more investigation
        # there is rref leak when shutting down, suspect it is because
        # ref as arg is passed to pybind boundary, and the ref is not garbage
        # collected by python when calling shutdown()
        测试 PyTorch Script 函数的训练器功能

        设置忽略 RRef 泄漏，可能是由于将引用作为参数传递到 pybind 边界而导致的，
        在调用 shutdown() 时，Python 没有对引用进行垃圾回收。
        """

        import torch.distributed.rpc.api as api
        api._ignore_rref_leak = True

        self._test_trainer_ps(create_torchscript_tensor, _run_trainer_torchscript, False)

    @dist_init
    def test_backward_multiple_round_trips(self):
        """
        执行多次反向传播的测试，用于验证多次模型参数更新的正确性

        参数：
        - torch.rand((3, 3), requires_grad=True): 随机生成的张量，需要梯度计算
        - torch.rand((3, 3)): 随机生成的张量
        - torch.rand((3, 3), requires_grad=True): 随机生成的张量，需要梯度计算
        - torch.rand((3, 3)): 随机生成的张量
        """

        self._backward_multiple_round_trips(
            torch.rand((3, 3), requires_grad=True),
            torch.rand((3, 3)),
            torch.rand((3, 3), requires_grad=True),
            torch.rand((3, 3)),
            torch.rand((3, 3), requires_grad=True),
            None,
            False
        )

    @dist_init
    def test_backward_different_tensor_dims(self):
        """
        执行不同维度张量的反向传播测试

        参数：
        - t1: torch.rand((4, 6), requires_grad=True) 随机生成的张量，需要梯度计算
        - t2: torch.rand((6, 5)) 随机生成的张量
        - t3: torch.rand((5, 7), requires_grad=True) 随机生成的张量，需要梯度计算
        - t4: torch.rand((7, 9)) 随机生成的张量
        """

        local_grads = None
        t1 = torch.rand((4, 6), requires_grad=True)
        t2 = torch.rand((6, 5))
        t3 = torch.rand((5, 7), requires_grad=True)
        t4 = torch.rand((7, 9))

        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                val = self._exec_func(exec_mode, torch.matmul, t1, t2)
                val = self._exec_func(exec_mode, torch.linalg.multi_dot, (val, t3, t4))
                loss = val.sum()

                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2, t2, t3, t4
                )
                local_grads = ret if ret else local_grads

    @dist_init
    def test_backward_unused_tensors(self):
        """
        执行未使用张量的反向传播测试

        参数：
        - t1: torch.rand((3, 3), requires_grad=True) 随机生成的张量，需要梯度计算
        - t2: torch.rand((3, 3), requires_grad=True) 随机生成的张量，需要梯度计算
        - t3: torch.rand((3, 3), requires_grad=True) 随机生成的张量，需要梯度计算
        """

        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        t3 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                s = self._exec_func(exec_mode, torch.stack, (t1, t2, t3))
                val = self._exec_func(
                    exec_mode,
                    torch.matmul,
                    torch.narrow(s, 0, 0, 1),
                    torch.narrow(s, 0, 2, 1),
                )

                loss = val.sum()
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2, t3
                )
                local_grads = ret if ret else local_grads
    def test_backward_multiple_output_tensors(self):
        local_grads = None  # 初始化一个变量local_grads，用于保存局部梯度
        t = torch.rand((10, 2), requires_grad=True)  # 创建一个形状为(10, 2)的张量t，要求计算梯度

        # 针对三种执行模式依次进行测试：本地执行、RPC同步执行、远程执行
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:  # 开启分布式自动求导的上下文管理器，获取上下文ID
                # 使用_exec_func方法执行torch.split函数，将张量t分割成多个子张量，存储在tensor_list中
                tensor_list = self._exec_func(exec_mode, torch.split, t, 2)
                t1 = tensor_list[0]  # 获取分割后的第一个子张量t1
                t2 = tensor_list[2]  # 获取分割后的第三个子张量t2
                t3 = tensor_list[4]  # 获取分割后的第五个子张量t3

                # 使用_exec_func方法执行torch.linalg.multi_dot函数，计算t1、t2、t3的多点乘积并返回结果val
                val = self._exec_func(exec_mode, torch.linalg.multi_dot, (t1, t2, t3))

                loss = val.sum()  # 计算val的所有元素的和作为损失值loss
                # 调用_verify_backwards方法，验证反向传播过程，检查是否正确返回梯度
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t
                )
                local_grads = ret if ret else local_grads  # 如果ret不为空，则更新local_grads为ret

    def _run_test_backward_unused_send_function_in_thread(self):
        with dist_autograd.context() as context_id:  # 开启分布式自动求导的上下文管理器，获取上下文ID
            t1 = torch.rand((3, 3), requires_grad=True)  # 创建一个形状为(3, 3)的张量t1，要求计算梯度
            t2 = torch.rand((3, 3), requires_grad=True)  # 创建一个形状为(3, 3)的张量t2，要求计算梯度

            # 使用rpc_sync方法调用torch.add函数，将t1和t2相加，结果存储在res中
            res = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.add, args=(t1, t2)
            )

            val = torch.mul(t1, t2)  # 计算t1和t2的逐元素乘积，结果存储在val中

            # 执行反向传播，这里会导致永久挂起，因为没有使用RPC函数的结果
            dist_autograd.backward(context_id, [val.sum()])

    @dist_init
    def test_backward_unused_send_function(self):
        # 在一个线程中运行_run_test_backward_unused_send_function_in_thread方法，该方法永远不会完成
        t = threading.Thread(
            target=self._run_test_backward_unused_send_function_in_thread
        )
        t.daemon = True  # 将线程设置为守护线程
        t.start()  # 启动线程
        t.join(10)  # 等待线程最多10秒钟

        # 验证线程仍然存活（表示反向传播尚未完成）
        self.assertTrue(t.is_alive())
    def test_backward_autograd_engine_error(self):
        # 使用 dist_autograd 上下文管理器创建上下文 ID
        with dist_autograd.context() as context_id:
            # 创建两个随机张量 t1 和 t2，并指定需要计算梯度
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            
            # 执行一些操作以模拟错误之前的过程
            tmp = (t1 + t2) * (t1 + t2)
            
            # 调用 SimulateBackwardError.apply 函数，模拟后向传播时的错误
            t3 = SimulateBackwardError.apply(tmp)

            # 在不同节点之间运行多轮 RPC，并验证原始节点是否在链中的深层节点上抛出错误
            val = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.add, args=(t2, t3)
            )
            val = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.mul, args=(val, t2)
            )
            val = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.matmul, args=(val, t2)
            )
            val = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.div, args=(val, t2)
            )

            # 使用 assertRaisesRegex 验证 RuntimeError 异常是否包含指定的错误信息
            with self.assertRaisesRegex(
                RuntimeError, "Error on Node [0-9]+: Simulate error on backward pass"
            ):
                # 执行反向传播，并验证是否接收到错误
                dist_autograd.backward(context_id, [val.sum()])

    @dist_init(clean_shutdown=False)
    @skip_but_pass_in_sandcastle_if(
        IS_MACOS,
        "Test is flaky on MacOS since libuv error handling is not as robust as TCP",
    )
    def test_backward_node_failure(self):
        # 设置 RPC 超时时间为 5 秒
        rpc._set_rpc_timeout(5)
        # 使用指定的文件初始化方法、排名和世界大小初始化进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 使用 dist_autograd 上下文管理器创建上下文 ID
        with dist_autograd.context() as context_id:
            # 创建两个随机张量 t1 和 t2，并指定需要计算梯度
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            
            # 在远程节点使用 RPC 调用 torch.add 函数
            res = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.add, args=(t1, t2)
            )

            # 等待所有 RPC 调用完成
            dist.barrier()

            # 杀死所有奇数排名的节点
            if self.rank % 2 == 0:
                # 获取用于关机错误的正则表达式
                shutdown_error_regex = self.get_shutdown_error_regex()
                
                # 等待所有其他节点死亡
                for rank in range(self.world_size):
                    if rank % 2 != 0:
                        wait_until_node_failure(rank, shutdown_error_regex)

                # 由于关机顺序不太明确定义，可能会看到由 get_shutdown_error_regex() 给出的任何错误
                # 使用 assertRaisesRegex 验证 RuntimeError 异常是否包含指定的错误信息
                with self.assertRaisesRegex(RuntimeError, shutdown_error_regex):
                    # 执行反向传播，并验证是否因为所有其他节点都已死亡而接收到错误
                    dist_autograd.backward(context_id, [res.sum()])
            else:
                # 退出所有其他节点
                pass

    @dist_init
    def test_backward_without_context(self):
        # 创建两个大小为3x3的张量，允许计算梯度
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        context_id = 100  # 设置虚拟的上下文ID
        # 使用断言检查是否引发指定异常
        with self.assertRaisesRegex(
            RuntimeError,
            f"Could not find autograd context with id: {context_id}",
        ):
            # 使用RPC同步调用，在远程执行torch.add操作
            res = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.add, args=(t1, t2)
            )
            # 对RPC调用结果求和的结果进行反向传播
            dist_autograd.backward(context_id, [res.sum()])

    @dist_init
    def test_backward_without_rpc(self):
        # 获取当前进程的等级作为目标等级
        dst_rank = self.rank
        # 创建分布式自动求导上下文，并获取上下文ID
        with dist_autograd.context() as context_id:
            # 创建两个大小为3x3的张量，允许计算梯度
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            # 计算t1和t2的加法结果
            t3 = torch.add(t1, t2)

            # 对t3的和进行分布式自动求导
            dist_autograd.backward(context_id, [t3.sum()])
            # 获取梯度信息
            grads = dist_autograd.get_gradients(context_id)
            # 使用断言检查是否符合预期的梯度数量和内容
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(torch.ones(3, 3), grads[t1])
            self.assertEqual(torch.ones(3, 3), grads[t2])

    @dist_init
    def test_backward_invalid_args(self):
        # 创建分布式自动求导上下文，并获取上下文ID
        with dist_autograd.context() as context_id:

            # 使用断言检查是否引发指定异常（类型错误）
            with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
                dist_autograd.backward(context_id, None)

            # 使用断言检查是否引发指定异常（类型错误）
            with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
                dist_autograd.backward(None, None)

            # 使用断言检查是否引发指定异常（运行时错误）
            with self.assertRaisesRegex(
                RuntimeError, "No tensors provided for gradient computation"
            ):
                # 创建空列表作为参数，尝试执行分布式自动求导
                dist_autograd.backward(context_id, [])

            # 使用断言检查是否引发指定异常（运行时错误）
            with self.assertRaisesRegex(RuntimeError, "requires_grad not set on"):
                # 创建不允许计算梯度的张量
                t = torch.rand(3, 3)
                dist_autograd.backward(context_id, [t])

            # 使用断言检查是否引发指定异常（运行时错误）
            with self.assertRaisesRegex(
                RuntimeError, "is not a scalar, all roots need to be scalar"
            ):
                # 创建不是标量的张量，尝试执行分布式自动求导
                t = torch.rand(3, 3, requires_grad=True)
                dist_autograd.backward(context_id, [t])

            # 使用断言检查是否引发指定异常（运行时错误）
            with self.assertRaisesRegex(
                RuntimeError, "does not have a valid gradient function"
            ):
                # 创建不具备有效梯度函数的张量
                t = torch.rand(1, requires_grad=True)
                dist_autograd.backward(context_id, [t])

    @dist_init
    # 定义一个测试方法，用于测试多个根的反向传播情况
    def test_backward_multiple_roots(self):
        local_grads = None  # 初始化本地梯度为空
        t1 = torch.rand((3, 3), requires_grad=True)  # 创建一个3x3的张量t1，需要计算梯度
        t2 = torch.rand((3, 3), requires_grad=True)  # 创建一个3x3的张量t2，需要计算梯度
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC]:  # 遍历执行模式：本地和RPC同步
            with dist_autograd.context() as context_id:  # 使用分布式自动求导上下文管理器
                # 执行加法操作并对结果求和，然后返回结果的梯度和
                r1 = self._exec_func(exec_mode, torch.add, t1, t2).sum()
                # 执行乘法操作并对结果求和，然后返回结果的梯度和
                r2 = self._exec_func(exec_mode, torch.mul, t1, t2).sum()
                # 执行余弦函数操作并对结果求和，然后返回结果的梯度和
                r3 = self._exec_func(exec_mode, torch.cos, t1).sum()
                # 执行除法操作并对结果求和，然后返回结果的梯度和
                r4 = self._exec_func(exec_mode, torch.div, t1, t2).sum()

                # 验证反向传播结果，并更新本地梯度
                local_grads = self._verify_backwards(
                    exec_mode, [r1, r2, r3, r4], context_id, local_grads, t1, t2
                )

    @dist_init
    # 初始化分布式环境后，定义一个测试方法，用于测试不同数据类型的反向传播
    def test_backward_different_dtypes(self):
        self._backward_different_dtypes(
            torch.rand((3, 3), requires_grad=True, dtype=torch.float32),  # 创建一个3x3的浮点数张量，需要计算梯度
            torch.rand((3, 3), requires_grad=True, dtype=torch.float64),  # 创建一个3x3的双精度浮点数张量，需要计算梯度
            False  # 指示是否进行验证
        )

    @dist_init
    # 初始化分布式环境后，定义一个测试方法，用于测试简单的Python用户定义函数的反向传播
    def test_backward_simple_python_udf(self):
        self._backward_simple_python_udf(
            torch.rand(3, 3, requires_grad=True),  # 创建一个3x3的张量，需要计算梯度
            torch.rand(3, 3, requires_grad=True),  # 创建一个3x3的张量，需要计算梯度
            False  # 指示是否进行验证
        )

    @dist_init
    # 初始化分布式环境后，定义一个测试方法，用于测试简单的脚本调用的反向传播
    def test_backward_simple_script_call(self):
        self._backward_simple_script_call(
            torch.rand(3, 3, requires_grad=True),  # 创建一个3x3的张量，需要计算梯度
            torch.rand(3, 3, requires_grad=True),  # 创建一个3x3的张量，需要计算梯度
            False  # 指示是否进行验证
        )

    @staticmethod
    # 静态方法：定义一个复杂的Python用户定义函数，接受两个张量作为输入
    def _complex_python_udf(t1, t2):
        t3 = torch.nn.functional.linear(t1, t2)  # 对t1和t2进行线性变换
        t4 = torch.nn.functional.linear(t2, t3)  # 对t2和t3进行线性变换
        t5 = torch.nn.functional.linear(t3, t4)  # 对t3和t4进行线性变换
        return torch.linalg.multi_dot([t1, t2, t3, t4, t5])  # 返回五个张量的多点乘积

    @dist_init
    # 初始化分布式环境后，定义一个测试方法，用于测试复杂的Python用户定义函数的反向传播
    def test_backward_complex_python_udf(self):
        # 在本地和分布式自动求导模式下运行相同的代码，并验证梯度是否一致
        local_grads = None  # 初始化本地梯度为空
        t1 = torch.rand((3, 3), requires_grad=True)  # 创建一个3x3的张量t1，需要计算梯度
        t2 = torch.rand((3, 3), requires_grad=True)  # 创建一个3x3的张量t2，需要计算梯度
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:  # 遍历执行模式：本地和远程
            with dist_autograd.context() as context_id:  # 使用分布式自动求导上下文管理器
                # 执行复杂的Python用户定义函数，并返回结果
                ret = self._exec_func(
                    exec_mode, DistAutogradTest._complex_python_udf, t1, t2
                )
                loss = ret.sum()  # 对结果求和，作为损失值
                # 验证反向传播结果，并更新本地梯度
                local_grads = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )

    @staticmethod
    # 静态方法：定义一个带有反向传播错误的Python用户定义函数，接受两个张量作为输入
    def _python_udf_with_backward_error(t1, t2):
        t3 = t1 + t2  # 执行张量的加法运算
        t4 = SimulateBackwardError.apply(t3)  # 模拟反向传播错误
        return torch.linalg.multi_dot([t1, t2, t3, t4])  # 返回四个张量的多点乘积

    @staticmethod
    # 静态方法：定义一个带有嵌套RPC调用和反向传播错误的函数，接受两个张量和目标作为输入
    def _nested_rpc_call_backward_error(t1, t2, dst):
        t1 = t1 * t2  # 执行张量的乘法运算
        t2 = t1 + t2  # 执行张量的加法运算
        res = rpc.rpc_sync(
            worker_name(dst),  # 获取目标工作节点名称
            DistAutogradTest._python_udf_with_backward_error,  # 在目标工作节点上执行带有反向传播错误的Python用户定义函数
            args=(t1, t2),  # 将t1和t2作为参数传递给函数
        )
        return torch.linalg.multi_dot([t1, t2, res])  # 返回三个张量的多点乘积
    def test_backward_python_udf_error(self):
        # 创建两个3x3的张量，允许计算梯度
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        # 使用分布式自动求导上下文
        with dist_autograd.context() as context_id:
            # 在指定的工作节点上同步执行远程过程调用
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()),
                DistAutogradTest._nested_rpc_call_backward_error,
                args=(t1, t2, self._next_rank()),
            )
            # 断言捕获到运行时错误，消息为 "Simulate error on backward pass"
            with self.assertRaisesRegex(
                RuntimeError, "Simulate error on backward pass"
            ):
                # 在指定上下文中执行反向传播，传入损失的总和
                dist_autograd.backward(context_id, [loss.sum()])

    _backward_done = False

    @dist_init(clean_shutdown=False)
    @skip_but_pass_in_sandcastle_if(
        IS_MACOS,
        "Test is flaky on MacOS since libuv error handling is not as robust as TCP",
    )
    def test_backward_node_failure_python_udf(self):
        # 设置短暂的超时时间以快速超时失败的RPC调用
        rpc._set_rpc_timeout(5)  # 5秒
        # 初始化进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        with dist_autograd.context() as context_id:
            # 创建两个3x3的张量，允许计算梯度
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)

            dst = self._next_rank()
            # 在指定的工作节点上同步执行远程过程调用
            res = rpc.rpc_sync(
                worker_name(dst),
                my_py_nested_call,
                args=(t1, t2, dst, self.world_size, 1),
            )

            # 分布式环境下的屏障同步
            dist.barrier()

            # 终止排名为2的进程（嵌套RPC的最后一跳），并验证排名为0是否接收到错误
            if self.rank == 2:
                return

            store = dist.distributed_c10d._get_default_store()
            if self.rank == 0:
                # 等待排名为2的进程终止
                shutdown_error_regex = self.get_shutdown_error_regex()
                wait_until_node_failure(2, shutdown_error_regex)
                # 由于排名为2已死，期望捕获到由get_shutdown_error_regex()返回的运行时错误
                with self.assertRaisesRegex(RuntimeError, shutdown_error_regex):
                    # 执行反向传播，并验证我们是否收到错误，因为排名为2已经死亡
                    dist_autograd.backward(context_id, [res.sum()])

                # 在存储中标记排名为0的节点完成
                store.set('test_backward_node_failure_python_udf_rank0_done', "True")
            else:
                # 等待排名为0的节点上的反向传播完成
                store.wait(['test_backward_node_failure_python_udf_rank0_done'], timedelta(seconds=10))

    @staticmethod
    def _nested_python_udf(t1, t2, dst):
        # 计算张量的逐元素乘积和加法
        t3 = t1 * t2
        t4 = t1 + t2
        # 在指定的工作节点上同步执行远程过程调用
        res = rpc.rpc_sync(worker_name(dst), my_py_add, args=(t3, t4))
        # 返回多个张量的复杂计算结果
        return t1 * t2 * t3 * t4 * res

    @dist_init
    def test_backwards_nested_python_udf(self):
        # 测试反向嵌套 Python 用户定义函数（UDF），在本地运行等效于 _nested_python_udf。
        self._backwards_nested_python_udf(
            torch.rand(3, 3, requires_grad=True),  # 创建一个随机的3x3张量，要求计算梯度
            torch.rand(3, 3, requires_grad=True),  # 创建另一个随机的3x3张量，要求计算梯度
            False  # 指定一个布尔值参数
        )

    _test_clean_context_backward_context_id = None

    class MyBackwardFunc(Function):
        @staticmethod
        def forward(ctx, input):
            return input  # 返回输入的张量

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            assert DistAutogradTest._test_clean_context_backward_context_id is not None

            # 释放上下文以模拟错误（在释放上下文之前使用屏障确保所有节点执行反向函数）。
            dist.barrier()  # 调用分布式通信库的屏障函数
            dist_autograd._release_context(
                DistAutogradTest._test_clean_context_backward_context_id
            )  # 释放指定的分布式自动求导上下文

            # 验证所有上下文是否已清理。
            assert _all_contexts_cleaned_up()  # 验证所有上下文是否已经清理完毕

            return input  # 返回输入的张量

    @dist_init
    def test_clean_context_during_backward(self):
        """
        This test simulates the situation where the 'backward' call might throw
        an exception locally which would lead to the autograd context being
        cleaned up if we're using the context manager. As a result, the autograd
        context might be cleaned up while some threads are still using the
        autograd context.

        It is fine for the 'backward' call to throw an exception in this test,
        but the process should not crash.
        """
        # 初始化进程组，设定初始方法和排名
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 创建一个新的分布式自动微分上下文
        context = dist_autograd._new_context()
        # 获取上下文的 ID
        context_id = context._context_id()
        # 将上下文 ID 存储到类变量中
        DistAutogradTest._test_clean_context_backward_context_id = context_id

        # 将上下文 ID 发送到所有节点
        for i in range(0, self.world_size):
            if i != self.rank:
                rank_distance = (i - self.rank + self.world_size) % self.world_size
                # 使用 RPC 同步发送上下文 ID 到指定节点
                rpc.rpc_sync(
                    worker_name(i),
                    _set_rpc_done,
                    args=(context_id, rank_distance),
                )

        # 等待所有节点完成
        dist.barrier()

        # 验证已接收到的上下文 ID 数量
        self.assertEqual(self.world_size - 1, len(known_context_ids))

        # 创建一个需要梯度的张量
        t1 = torch.rand((3, 3), requires_grad=True)
        for i in range(0, 100):
            dst = self._next_rank()
            # 使用 RPC 同步将张量 t1 发送给下一个节点，并返回结果
            t1 = rpc.rpc_sync(worker_name(dst), torch.add, args=(t1, t1))

        # 调用 MyBackwardFunc 作为反向传播过程中的第一个操作，以确保提前释放上下文
        t1 = DistAutogradTest.MyBackwardFunc.apply(t1)
        # 断言上下文中发送函数的数量为 100
        self.assertEqual(100, len(context._send_functions()))

        # 设置一个虚拟的上下文 ID
        context_id = 100  # dummy context_id
        # 使用上下文管理器断言 RuntimeError 异常信息包含指定的上下文 ID
        with self.assertRaisesRegex(
            RuntimeError,
            f"Could not find autograd context with id: {context_id}",
        ):
            # 调用分布式自动微分的 backward 函数，传入虚拟的上下文 ID 和张量 t1 的和作为参数
            dist_autograd.backward(context_id, [t1.sum()])

        # HACK: 强制关闭工作进程，因为否则自动微分引擎会在其他节点上卡住
        # 正确的修复方法应该解决 https://github.com/pytorch/pytorch/issues/27643，通知其他节点有关错误
        # 自动微分引擎在其他节点上卡住，因为它们正在等待从接收到错误的节点接收梯度
        dist.barrier()
        rpc.shutdown(graceful=False)
        sys.exit(0)


    @classmethod
    def _call_remote_embedding(cls, embedding_rref, input, offsets, per_sample_weights):
        # 获取嵌入值
        embedding = embedding_rref.local_value()
        # 调用嵌入函数并返回结果
        return embedding(input, offsets, per_sample_weights)

    @classmethod
    def _get_grad(cls, embedding_rref, context_id):
        # 获取嵌入值
        embedding = embedding_rref.local_value()
        # 使用分布式自动微分模块获取指定上下文 ID 的梯度映射
        grad_map = dist_autograd.get_gradients(context_id)
        # 返回嵌入权重对应的梯度
        return grad_map[embedding.weight]
    # 定义一个类方法，用于执行混合 requires_grad 操作
    @classmethod
    def _mixed_requires_grad_operaton(cls, t1, t2):
        # 如果 t2 需要梯度，则执行 t1 - t2 操作
        if t2.requires_grad:
            return t1 - t2
        else:
            # 如果 t2 不需要梯度，则执行 t1 * t2 操作
            return t1 * t2

    # 使用分布式初始化装饰器，定义一个测试方法 test_mixed_requires_grad
    @dist_init
    def test_mixed_requires_grad(self):
        # 调用 _mixed_requires_grad 方法，传入两个张量，其中一个需要梯度，另一个不需要
        self._mixed_requires_grad(
            torch.rand(3, 3, requires_grad=True),
            torch.rand(3, 3, requires_grad=False),
            False
        )

    # 定义一个继承自 Function 的测试调试信息功能类 TestDebugInfoFunc
    class TestDebugInfoFunc(Function):
        # 前向传播方法的静态函数，直接返回输入
        @staticmethod
        def forward(ctx, input):
            return input

        # 反向传播方法的静态函数，用 once_differentiable 装饰，获取调试信息并断言其不为空
        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            # 获取分布式自动求导模块的调试信息
            debug_info = dist_autograd._get_debug_info()
            # 断言调试信息不为空
            assert debug_info is not None
            # 获取当前反向传播次数
            backward_passes = int(debug_info["num_current_backward_passes"])

            # 难以验证确切的数字，因为分布式特性的影响
            # 不能在这里使用 barrier()，因为会阻塞自动求导的单个 CPU 线程，可能导致死锁
            assert backward_passes >= 1 and backward_passes <= 4
            return input

    # 使用分布式初始化装饰器，未完全提供的类和方法声明
    @dist_init
    # 定义测试方法，用于调试信息的测试
    def test_debug_info(self):
        # 调用初始化函数，设置参数和节点数量
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 创建两个随机张量，并且要求计算梯度
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        # 进入分布式自动求导的上下文管理器
        with dist_autograd.context() as context_id:
            # 初始化索引变量
            i = 0
            # 创建空字典以存储结果
            res = {}
            # 将第一个张量存入字典中
            res[i] = t1

            # 遍历所有节点的rank
            for rank in range(self.world_size):
                # 如果当前rank不等于本节点的rank
                if rank != self.rank:
                    # 使用RPC同步调用远程节点的torch.add函数，并将结果存入字典中
                    res[i + 1] = rpc.rpc_sync(
                        worker_name(rank), torch.add, args=(res[i], t2)
                    )
                    # 更新索引
                    i += 1

            # 调用自定义函数，在反向传播过程中执行以确保所有节点仍在等待backward()
            res[i + 1] = DistAutogradTest.TestDebugInfoFunc.apply(res[i])
            # 更新索引
            i += 1

            # 再次遍历所有节点的rank
            for rank in range(self.world_size):
                # 如果当前rank不等于本节点的rank
                if rank != self.rank:
                    # 使用RPC同步调用远程节点的torch.add函数，并将结果存入字典中
                    res[i + 1] = rpc.rpc_sync(
                        worker_name(rank), torch.add, args=(res[i], t2)
                    )
                    # 更新索引
                    i += 1

            # 执行分布式自动求导的反向传播过程，求和后反向传播
            dist_autograd.backward(context_id, [res[i].sum()])

            # 获取调试信息
            debug_info = dist_autograd._get_debug_info()
            # 获取自动求导上下文的数量并转换为整数
            num_autograd_context = int(debug_info["num_autograd_contexts"])
            # 断言自动求导上下文数量至少为1且最多为4
            self.assertTrue(num_autograd_context >= 1 and num_autograd_context <= 4)

        # 对于每个远程节点执行RPC同步调用，标记RPC已完成
        for rd in range(self.world_size - 1):
            rpc.rpc_sync(
                worker_name((self.rank + rd + 1) % self.world_size),
                _set_rpc_done,
                args=(context_id, rd + 1),
            )

        # 同步所有节点
        dist.barrier()

        # 验证信息
        debug_info = dist_autograd._get_debug_info()
        # 断言调试信息不为空
        assert debug_info is not None
        # 断言当前反向传播过程数为0
        self.assertEqual(0, int(debug_info["num_current_backward_passes"]))
        # 断言调试信息字段数量为2
        self.assertTrue(len(debug_info) == 2)

        # 断言所有上下文都已清理
        self.assertTrue(_all_contexts_cleaned_up())

        # 断言自动求导上下文数量为0
        debug_info = dist_autograd._get_debug_info()
        self.assertEqual(0, int(debug_info["num_autograd_contexts"]))

    # 静态方法：工作负载线程
    @staticmethod
    def _workload_thread():
        # 创建两个随机张量，并且要求计算梯度
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        # 进入分布式自动求导的上下文管理器
        with dist_autograd.context() as context_id:
            # 使用RPC同步调用worker0节点上的torch.add函数，并将结果存入t3
            t3 = rpc.rpc_sync("worker0", torch.add, args=(t1, t2))
            # 使用RPC同步调用worker0节点上的torch.mul函数，并将结果存入t4
            t4 = rpc.rpc_sync("worker0", torch.mul, args=(t2, t3))
            # 使用RPC同步调用worker0节点上的torch.matmul函数，并将结果存入t5
            t5 = rpc.rpc_sync("worker0", torch.matmul, args=(t3, t4))
            # 使用RPC同步调用worker0节点上的torch.add函数，并将结果存入t6
            t6 = rpc.rpc_sync("worker0", torch.add, args=(t4, t5))

            # 执行分布式自动求导的反向传播过程，求和后反向传播
            dist_autograd.backward(context_id, [t6.sum()])

    # 分布式初始化装饰器
    @dist_init
    def test_async_dist_autograd(self):
        """
        This test ensures async processing for distributed autograd works
        appropriately. This is achieved by spawning multiple threads and
        hammering a single node with a lot of backward() calls.
        """

        # 初始化进程组，设置初始方法和当前进程在组中的排名
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        
        # 如果当前进程不是排名为0的进程
        if self.rank != 0:
            # 其他所有排名的进程都在排名为0的进程上调度工作
            threads = []
            for i in range(20):
                # 创建线程，目标函数是_workload_thread()
                t = threading.Thread(target=DistAutogradTest._workload_thread)
                # 启动线程
                t.start()
                threads.append(t)

            # 等待所有线程执行完毕
            for thread in threads:
                thread.join()

        # 等待所有进程到达同步点
        dist.barrier()

    @dist_init
    def test_backward_accumulate_grads(self):
        # 创建具有梯度需求的随机张量
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        
        # 使用分布式自动求导上下文
        with dist_autograd.context() as context_id:
            t3 = torch.matmul(t1, t2)
            # 执行两次反向传播
            torch.autograd.backward([t3.sum()], retain_graph=True)
            torch.autograd.backward([t3.sum()])

            # 在远程工作节点上执行RPC调用
            t3 = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.matmul, args=(t1, t2)
            )
            # 再次执行两次反向传播
            dist_autograd.backward(context_id, [t3.sum()], retain_graph=True)
            dist_autograd.backward(context_id, [t3.sum()])

            # 验证本地和远程执行的梯度是否一致
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(t1.grad, grads[t1])
            self.assertEqual(t2.grad, grads[t2])

    @staticmethod
    def _test_nested_backward_accumulate_grads(t1, t2, dst_rank):
        return rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))

    @dist_init
    def test_nested_backward_accumulate_grads(self):
        # 调用嵌套的反向传播累积梯度函数
        self._nested_backward_accumulate_grads(
            torch.rand(3, 3, requires_grad=True),
            torch.rand(3, 3, requires_grad=True),
            False
        )

    @dist_init
    def test_multiple_backward(self):
        # 调用多次反向传播函数
        self._multiple_backward(
            torch.rand(3, 3, requires_grad=True),
            torch.rand(3, 3, requires_grad=True),
            False
        )

    @dist_init(clean_shutdown=False)
    # 定义一个测试方法，用于测试在多个后向传播中可能出现的错误情况
    def test_multiple_backward_with_errors(self):
        # 使用指定的初始化方法、排名和世界大小初始化进程组
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # 创建两个随机张量，要求计算梯度
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        # 进入分布式自动求导上下文
        with dist_autograd.context() as context_id:
            # 在指定的远程 worker 上执行远程过程调用，调用一个带有后向传播错误的 Python 函数
            loss = rpc.rpc_sync(
                f'worker{self._next_rank()}',
                DistAutogradTest._python_udf_with_backward_error,
                args=(t1, t2)).sum()

            try:
                # 多次循环运行反向传播
                for i in range(100):
                    if i < 50:
                        # 在保留计算图的情况下运行反向传播，并期望捕获特定的运行时错误
                        with self.assertRaisesRegex(RuntimeError, "Simulate error on backward pass"):
                            dist_autograd.backward(context_id, [loss], retain_graph=True)
                    elif i > 50:
                        # 从错误中恢复，继续运行反向传播
                        dist_autograd.backward(context_id, [loss], retain_graph=True)
                    else:
                        # 同步所有进程
                        dist.barrier()
                        # 关闭模拟后向传播错误的标志
                        SimulateBackwardError._simulate_error = False
                        dist.barrier()
            finally:
                # 在重置标志之前，再次同步所有进程
                dist.barrier()

                # 重置模拟后向传播错误的标志
                SimulateBackwardError._simulate_error = True

    @dist_init
    # 定义一个用于验证后向传播钩子的测试方法
    def test_backward_verify_hooks(self):
        # 创建一个全为 1 的张量，要求计算梯度，并注册一个钩子函数使梯度翻倍
        t1 = torch.ones((3, 3), requires_grad=True)
        t1.register_hook(lambda grad: grad * 2)
        # 创建另一个全为 1 的张量，要求计算梯度
        t2 = torch.ones((3, 3), requires_grad=True)
        # 初始化本地梯度为 None
        local_grads = None
        # 遍历执行模式：本地、RPC 同步、远程
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            # 进入分布式自动求导上下文
            with dist_autograd.context() as context_id:
                # 执行指定的执行模式下的函数，并获取返回值
                ret = self._exec_func(exec_mode, torch.matmul, t1, t2)
                # 计算损失并求和
                loss = ret.sum()
                # 验证反向传播的结果，并返回本地梯度
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )
                # 更新本地梯度变量
                local_grads = ret if ret else local_grads
    def test_no_grad_copy(self):
        '''
        Similar to test in test_autograd.py.
        '''
        # 定义一个自动求导函数，将梯度指针保存为类的静态变量
        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad.data_ptr()
                return grad, grad

        # 定义另一个自动求导函数，同样将梯度指针保存为类的静态变量
        class MyFuncSingleGrad(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp):
                return inp

            @staticmethod
            def backward(ctx, grad):
                MyFuncSingleGrad.static_grad_ptr = grad.data_ptr()
                return grad

        # 定义一个不连续梯度的自动求导函数
        class NonContGradFunc(Function):
            @staticmethod
            def forward(ctx, inp1):
                ctx.size = inp1.size()
                return torch.tensor([1.])

            @staticmethod
            def backward(ctx, grad):
                return torch.ones(1).expand(ctx.size)

        # 创建两个随机张量a和b，需要计算梯度
        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        
        # 在分布式自动求导上下文中，执行自动求导操作
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [NonContGradFunc.apply(MyFunc.apply(a, b))])
            # 获取梯度字典
            grads = dist_autograd.get_gradients(context_id)
            # 检查是否对张量a和b进行了梯度拷贝
            self.assertFalse(grads[a].data_ptr() == MyFunc.static_grad_ptr)
            self.assertFalse(grads[b].data_ptr() == MyFunc.static_grad_ptr)

        # 另一个测试用例，预期不会对张量a进行梯度拷贝
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [MyFuncSingleGrad.apply(a)[1][0]])
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFuncSingleGrad.static_grad_ptr
            p_a = grads[a].data_ptr()
            # 验证没有进行梯度拷贝
            self.assertTrue(p_a == p_g)

        # 测试应该对张量a和b都进行梯度拷贝的情况
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [MyFunc.apply(a, b)[1][0]])
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFunc.static_grad_ptr
            p_a = grads[a].data_ptr()
            p_b = grads[b].data_ptr()
            # 检查a和b是否使用不同的梯度缓冲区
            self.assertFalse(p_a == p_b)
            # 都应该进行梯度拷贝
            self.assertFalse(grads[a].data_ptr() == MyFunc.static_grad_ptr)
            self.assertFalse(grads[b].data_ptr() == MyFunc.static_grad_ptr)
    def test_grad_copy_sparse_indices_extra_ref(self):
        # 创建一个自动求导函数，将梯度指针保存为类的静态属性
        class MyFunc(Function):
            static_grad_ptr = None  # 静态属性：保存梯度指针
            static_grad_indices_ref = None  # 静态属性：保存稀疏张量的索引引用
            static_grad_values_ref = None  # 静态属性：保存稀疏张量的值引用

            @staticmethod
            def forward(ctx, inp):
                return inp

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                # indices() 和 values() 返回视图，因此保持它们的引用不会增加稀疏张量内部索引和值的引用计数。
                MyFunc.static_grad_indices_ref = grad._indices()
                MyFunc.static_grad_values_ref = grad._values()
                return grad

        a = torch.randn(10, 3, requires_grad=True)  # 创建一个需要梯度的张量 a
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])  # 创建输入张量 input
        offsets = torch.tensor([0, 4])  # 创建偏移量张量 offsets
        import torch.nn.functional as F  # 导入 PyTorch 的 nn.functional 模块

        with dist_autograd.context() as context_id:
            emb_matrix = MyFunc.apply(a)  # 应用自定义的自动求导函数 MyFunc 到张量 a 上
            loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()  # 计算稀疏的嵌入 Bag，并求和作为损失
            dist_autograd.backward(context_id, [loss], retain_graph=True)  # 在分布式自动求导上下文中进行反向传播
            grads = dist_autograd.get_gradients(context_id)  # 获取梯度
            p_g = MyFunc.static_grad_ptr  # 获取静态梯度指针
            p_a = grads[a]._values().data_ptr()  # 获取张量 a 的梯度值的数据指针
            self.assertIsNotNone(MyFunc.static_grad_indices_ref)  # 断言静态属性 static_grad_indices_ref 不为空
            self.assertIsNotNone(MyFunc.static_grad_values_ref)  # 断言静态属性 static_grad_values_ref 不为空
            # 由于 static_grad_indices_ref 和 static_grad_values_ref 保持了视图，所以梯度会被窃取。
            self.assertTrue(p_g == p_a)  # 断言静态梯度指针与张量 a 的梯度值的数据指针相等

    @dist_init
    def test_post_hooks(self):
        self.hook_called_times = 0  # 初始化钩子调用次数为 0

        def post_hook_add_one(output_grads, input_grads):
            self.hook_called_times += 1  # 钩子函数，每次调用增加钩子调用次数 1
            return output_grads

        def post_hook_add_two(output_grads, input_grads):
            self.hook_called_times += 2  # 钩子函数，每次调用增加钩子调用次数 2
            return output_grads

        t = torch.rand(10, 10, requires_grad=True)  # 创建一个需要梯度的张量 t
        a = t + t  # 创建张量 a，是 t 的两倍

        # 注册后处理钩子
        accumulate_grad_0 = a.grad_fn.next_functions[0][0]
        accumulate_grad_0.register_hook(post_hook_add_one)  # 向第一个梯度函数注册钩子 post_hook_add_one
        accumulate_grad_0.register_hook(post_hook_add_two)  # 向第一个梯度函数注册钩子 post_hook_add_two

        accumulate_grad_1 = a.grad_fn.next_functions[1][0]
        accumulate_grad_1.register_hook(post_hook_add_two)  # 向第二个梯度函数注册钩子 post_hook_add_two

        with dist_autograd.context() as context_id:
            loss = a.sum()  # 计算张量 a 的和作为损失
            dist_autograd.backward(context_id, [loss])  # 在分布式自动求导上下文中进行反向传播
            self.assertEqual(5, self.hook_called_times)  # 断言钩子调用次数为 5
            grads = dist_autograd.get_gradients(context_id)  # 获取梯度
            self.assertEqual(1, len(grads))  # 断言梯度字典长度为 1
            self.assertTrue(t in grads)  # 断言张量 t 在梯度字典中

    @staticmethod
    # 定义一个函数 `_slow_add`，用于模拟耗时的加法操作
    def _slow_add(t1, t2):
        # 模拟函数执行时的等待时间
        time.sleep(1)
        # 计算两个张量的和
        t3 = t1 + t2
        # 设置张量的梯度属性为 True，以便后续进行自动微分
        t3.requires_grad = True
        return t3

    # 使用装饰器 `dist_init` 标记的方法 `test_thread_local_context_id`
    def test_thread_local_context_id(self):
        # 创建两个随机张量
        t1 = torch.rand((3, 3))
        t2 = torch.rand((3, 3))

        # 计算两个张量的和
        t3 = t1 + t2
        # 设置张量的梯度属性为 True，以便后续进行自动微分
        t3.requires_grad = True
        # 对张量的和进行求和并执行反向传播
        t3.sum().backward()

        # 根据工作节点的名称确定远程目标
        dst = worker_name((self.rank + 1) % self.world_size)
        # 在远程工作节点上调用 `_slow_add` 函数，并传递参数 t1 和 t2
        rref = rpc.remote(dst, DistAutogradTest._slow_add, args=(t1, t2))

        # 使用 `dist_autograd.context()` 创建一个上下文环境
        with dist_autograd.context() as context_id:
            # 从远程获取结果，并求和
            loss = rref.to_here().sum()
            # 由于 _slow_add 是一个耗时操作，其后的反向传播可能由之前的
            # rpc.remote 线程调用，而该线程可能没有有效的 context_id。
            # 因此，这里测试我们是否正确地在服务器端线程之间传播 thread_local 状态。
            # 使用给定的 context_id 对 loss 进行反向传播
            dist_autograd.backward(context_id, [loss])
            # 断言验证，通过 rpc_sync 同步调用远程方法 `_compare_owner_value`，
            # 检查梯度传播的正确性
            self.assertTrue(
                rpc.rpc_sync(
                    dst,
                    _compare_owner_value,
                    args=(context_id, rref, t3.grad)
                )
            )
class CudaDistAutogradTest(CommonDistAutogradTest):
    # CUDA分布式自动求导测试类，继承自通用分布式自动求导测试类

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_gpu_simple(self):
        # 测试简单的GPU操作
        t1 = torch.rand(3, 3, requires_grad=True, device="cuda:0")
        # 创建一个3x3的随机张量t1，要求计算梯度，存储在cuda:0设备上
        t2 = torch.rand(3, 3, requires_grad=True, device="cuda:0")
        # 创建一个3x3的随机张量t2，要求计算梯度，存储在cuda:0设备上
        (t1 + t2).sum().backward()
        # 计算t1 + t2的和，对结果求和，并进行反向传播

        with dist_autograd.context() as context_id:
            # 使用分布式自动求导上下文管理器创建上下文ID
            t3 = t1 + t2
            # 计算t1 + t2的结果
            dist_autograd.backward(context_id, [t3.sum()])
            # 在指定的上下文ID上进行反向传播，传入需要求导的张量列表
            grads = dist_autograd.get_gradients(context_id)
            # 获取指定上下文ID中的梯度
            self.assertEqual(2, len(grads))
            # 断言：梯度字典的长度应为2
            self.assertEqual(t1.grad, grads[t1])
            # 断言：t1的梯度应与从梯度字典中获取的t1的梯度相等
            self.assertEqual(t2.grad, grads[t2])
            # 断言：t2的梯度应与从梯度字典中获取的t2的梯度相等

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_gpu_to_cpu_continuation(self):
        # 测试GPU到CPU的持续操作
        t1 = torch.rand(3, 3, requires_grad=True, device="cuda:0")
        # 创建一个3x3的随机张量t1，要求计算梯度，存储在cuda:0设备上
        t2 = torch.rand(3, 3, requires_grad=True)
        # 创建一个3x3的随机张量t2，要求计算梯度，存储在默认设备上（CPU）
        
        # 运行几次迭代
        for i in range(3):
            t1.grad = None
            t2.grad = None
            # 清空t1和t2的梯度

            # 根节点是CPU
            local_grads = None
            for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC]:
                # 在本地模式和RPC同步模式下执行
                with dist_autograd.context() as context_id:
                    # 使用分布式自动求导上下文管理器创建上下文ID
                    t3 = self._exec_func(exec_mode, torch.add, t2, t2)
                    # 执行_exec_func函数，传入exec_mode，torch.add，t2，t2
                    t4 = t3.cuda(0) + t1
                    # 将t3移动到cuda:0设备上，并与t1相加
                    t5 = self._exec_func(exec_mode, torch.add, t4.cpu(), t2)
                    # 将t4移动到CPU设备上，并与t2相加
                    t6 = t5.cuda(0) + t4
                    # 将t5移动到cuda:0设备上，并与t4相加
                    t7 = self._exec_func(exec_mode, torch.add, t6.cpu(), t5)
                    # 将t6移动到CPU设备上，并与t5相加
                    # 自动求导图包含CPU -> GPU -> CPU的执行路径
                    ret = self._verify_backwards(
                        exec_mode, [t7.sum()], context_id, local_grads, t1, t2
                    )
                    # 使用_verify_backwards函数验证反向传播结果
                    local_grads = ret if ret else local_grads

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_gpu_to_cpu_continuation_gpu_root(self):
        # 测试GPU到CPU的持续操作，根节点为GPU
        t1 = torch.rand(3, 3, requires_grad=True, device="cuda:0")
        # 创建一个3x3的随机张量t1，要求计算梯度，存储在cuda:0设备上
        t2 = torch.rand(3, 3, requires_grad=True)
        # 创建一个3x3的随机张量t2，要求计算梯度，存储在默认设备上（CPU）

        # 运行几次迭代
        for i in range(3):
            t1.grad = None
            t2.grad = None
            # 清空t1和t2的梯度

            # 根节点是CPU
            local_grads = None
            for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC]:
                # 在本地模式和RPC同步模式下执行
                with dist_autograd.context() as context_id:
                    # 使用分布式自动求导上下文管理器创建上下文ID
                    t3 = self._exec_func(exec_mode, torch.add, t2, t2)
                    # 执行_exec_func函数，传入exec_mode，torch.add，t2，t2
                    t4 = t3.cuda(0) + t1
                    # 将t3移动到cuda:0设备上，并与t1相加
                    t5 = self._exec_func(exec_mode, torch.add, t4.cpu(), t2)
                    # 将t4移动到CPU设备上，并与t2相加
                    t6 = t5.cuda(0) + t4
                    # 将t5移动到cuda:0设备上，并与t4相加
                    # 自动求导图包含CPU -> GPU -> CPU的执行路径
                    ret = self._verify_backwards(
                        exec_mode, [t6.sum()], context_id, local_grads, t1, t2
                    )
                    # 使用_verify_backwards函数验证反向传播结果
                    local_grads = ret if ret else local_grads


class FaultyAgentDistAutogradTest(RpcAgentTestFixture):
    # 故障代理分布式自动求导测试类，继承自RpcAgentTestFixture
    # 从DistAutogradTest中复用一个简化的辅助函数，确保即使RPC失败，自动求导上下文也能成功清理
    # 定义一个测试辅助函数，用于清理上下文
    def context_cleanup_test_helper(self, rpc_args, func):
        # 使用指定的文件初始化方法和进程信息进行 PG 初始化
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # 在分布式自动求导环境中测试，即使 RPC 传输的张量不需要梯度，
        # 我们仍然清理在其他节点上创建的分布式自动求导上下文。
        # 这是因为即使张量参数不需要梯度，自动求导上下文仍然会通过 RPC 传输，
        # 因为响应可能需要。
        dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}

        # 使用分布式自动求导上下文
        with dist_autograd.context() as context_id:
            # 对每个目标节点进行迭代
            for dst_rank in dst_ranks:
                # 同步执行 RPC 调用，向目标节点发送函数调用
                rpc.rpc_sync(worker_name(dst_rank), func, args=rpc_args)
                # 同步执行 RPC 调用，标记 RPC 已完成
                rpc.rpc_sync(
                    worker_name(dst_rank), _set_rpc_done, args=(context_id, 1)
                )

        # 线程的上下文 ID 应该被清理
        with self.assertRaises(RuntimeError):
            # 检索指定上下文 ID 的分布式自动求导上下文
            dist_autograd._retrieve_context(context_id)

        # 确保所有同级完成对 `known_context_ids` 集合的变异
        dist.barrier()

        # 检查所有上下文是否已清理
        success = _all_contexts_cleaned_up()
        self.assertTrue(success)

    # 没有定义 `faulty_messages`，因此所有可重试消息都会失败 - 参见
    # faulty_rpc_agent_test_fixture.py 中的可重试消息列表。
    @dist_init
    def test_context_cleanup_tensor_with_grad(self):
        # 创建需要梯度的张量 t1 和 t2
        t1 = torch.ones(3, 3, requires_grad=True)
        t2 = torch.zeros(3, 3, requires_grad=True)
        # 使用测试辅助函数进行测试，传递张量参数和函数 torch.add
        self.context_cleanup_test_helper(rpc_args=(t1, t2), func=torch.add)

    @dist_init
    def test_verify_backend_options(self):
        # 断言 RPC 后端类型为 FAULTY_TENSORPIPE
        self.assertEqual(self.rpc_backend, rpc.backend_registry.BackendType.FAULTY_TENSORPIPE)
        # 断言 RPC 后端选项中的工作线程数为 8
        self.assertEqual(self.rpc_backend_options.num_worker_threads, 8)
        # 断言 RPC 后端选项中的失败发送次数为 3
        self.assertEqual(self.rpc_backend_options.num_fail_sends, 3)
        # 断言 RPC 后端选项中的失败消息列表长度为 4
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 4)
# 定义一个继承自 nn.Module 的包装模块，用于包裹传入的模型并将其移到指定设备上
class WrapperModule(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model.to(device)

    # 定义前向传播方法，直接调用被包裹模型的前向传播方法
    def forward(self, *args):
        return self.model(*args)

    # 定义获取梯度的方法，使用 dist_autograd 获取当前上下文中模型参数的梯度
    def gradients(self, ctx_id):
        grads = dist_autograd.get_gradients(ctx_id)
        return [grads[p] for p in self.model.parameters()]


# 定义一个测试类，继承自 RpcAgentTestFixture，用于测试分布式自动求导和 RPC 通信
class TensorPipeCudaDistAutogradTest(RpcAgentTestFixture):

    # 装饰器函数，如果 GPU 数量少于 4，则跳过此测试
    @skip_if_lt_x_gpu(4)
    def test_device_maps_backward_pass(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)

        # 设置设备映射，用于反向传播时使用对应的设备映射
        options.set_device_map(dst, {self.rank: (self.rank + 1) % self.world_size})

        # 初始化 RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 创建两个带梯度的张量，并放置在当前 rank 指定的设备上
        t1 = torch.rand(10, device=self.rank, requires_grad=True)
        t2 = torch.rand(10, device=self.rank, requires_grad=True)

        # 在分布式自动求导的上下文中进行 RPC 调用
        with dist_autograd.context() as context_id:
            res = rpc.rpc_sync(dst, torch.add, args=(t1, t2))
            dist_autograd.backward(context_id, [res.sum()])

            # 获取梯度信息
            grads = dist_autograd.get_gradients(context_id)

            # 断言两个张量的梯度为全一向量
            self.assertEqual(torch.ones(10), grads[t1])
            self.assertEqual(torch.ones(10), grads[t2])

            # 断言梯度张量与原张量在同一设备上
            self.assertEqual(t1.device, grads[t1].device)
            self.assertEqual(t2.device, grads[t2].device)

        # 关闭 RPC
        rpc.shutdown()

    # 内部定义的远程计算模块，继承自 torch.nn.Module
    class MyRemoteCompute(torch.nn.Module):
        def forward(self, input):
            input = input * 2.0
            return input

    # 内部定义的本地计算模块，接受一个下一阶段计算模块作为参数
    class MyLocalCompute(torch.nn.Module):
        def __init__(self, next_stage):
            super().__init__()
            self.next_stage = next_stage

        def forward(self, input):
            # 使用 RPC 同步调用下一阶段计算模块的前向传播方法
            return self.next_stage.rpc_sync().forward(input)

    # 装饰器函数，如果 GPU 数量少于 4，则跳过此测试
    @skip_if_lt_x_gpu(4)
    # 定义一个测试方法，用于测试分布式自动求导的同步流操作
    def test_dist_autograd_sync_streams(self):

        # 获取RPC后端选项
        options = self.rpc_backend_options
        # 计算目标worker的名称，使用环形方式确定
        dst = worker_name((self.rank + 1) % self.world_size)

        # 设定反向传播时应使用的设备映射，以确保正确的设备对应关系
        options.set_device_map(dst, {self.rank: (self.rank + 1) % self.world_size})

        # 初始化RPC环境
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        # 创建远程计算对象
        remote_compute = rpc.remote(dst, TensorPipeCudaDistAutogradTest.MyRemoteCompute)
        # 创建本地计算对象
        local_compute = TensorPipeCudaDistAutogradTest.MyLocalCompute(remote_compute)

        # 循环执行10次
        for _ in range(10):
            # 创建具有梯度的随机输入数据张量
            input = torch.rand([1000, 10000], device=self.rank, requires_grad=True)
            # 运行本地自动求导操作
            result = input * 2.0
            r = random.random()
            loss = result.sum() * r
            loss.backward()

            # 运行分布式自动求导操作
            with dist_autograd.context() as context_id:
                result = local_compute(input)
                loss = result.sum() * r
                dist_autograd.backward(context_id, [loss])

                # 比较梯度
                grads = dist_autograd.get_gradients(context_id)
                self.assertEqual(input.grad, grads[input])

        # 关闭RPC环境
        rpc.shutdown()

    @skip_if_lt_x_gpu(4)


这段代码是一个用于测试分布式自动求导同步流操作的方法。代码中涉及了RPC的初始化与关闭，设备映射的设置，远程和本地计算的创建，以及本地和分布式自动求导的执行与梯度比较。
    def test_gradients_synchronizations(self):
        # 获取RPC后端选项
        options = self.rpc_backend_options
        # 遍历每个对等节点
        for peer_rank in range(self.world_size):
            # 设置设备映射，将本节点与每个对等节点的映射关系添加到选项中
            options.set_device_map(worker_name(peer_rank), {self.rank: peer_rank})

        # 初始化RPC
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        if self.rank == 0:
            # 主节点处理
            # 创建多个线性层模型，每个节点一个，除了主节点
            layers = [nn.Linear(2000, 2000) for _ in range(self.world_size - 1)]
            # 将本地线性层移到设备 0
            local_layers = [l.to(0) for l in layers]
            remote_layers = []
            # 遍历远程节点，为每个节点创建远程模块代理对象
            for rank in range(1, self.world_size):
                remote_layers.append(rpc.remote(
                    worker_name(rank),
                    WrapperModule,
                    args=(layers[rank - 1], rank)
                ))

            x = torch.randn(5000, 2000).to(0)
            # 本地迭代
            local_model = nn.Sequential(*local_layers)
            local_model(x).sum().backward()

            # 远程迭代
            with dist_autograd.context() as context_id:
                for remote_layer in remote_layers:
                    x = remote_layer.rpc_sync().forward(x)

                # 执行远程自动求导的反向传播
                dist_autograd.backward(context_id, [x.sum()])

                futs = []
                # 为每个远程模块获取梯度
                for remote_layer in remote_layers:
                    futs.append(remote_layer.rpc_async().gradients(context_id))

                # 检查每个远程模块的梯度与本地模块的梯度是否一致
                for i in range(len(futs)):
                    local_gradients = [p.grad for p in local_layers[i].parameters()]
                    for g1, g2 in zip(futs[i].wait(), local_gradients):
                        self.assertEqual(g1, g2)

        # 关闭RPC
        rpc.shutdown()
```