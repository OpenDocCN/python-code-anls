# `.\pytorch\torch\testing\_internal\distributed\multi_threaded_pg.py`

```py
# 忽略 mypy 的错误提示
# 导入必要的库和模块
import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce

# 导入 PyTorch 相关模块
import torch
import torch.distributed as dist
import weakref
from torch._C._distributed_c10d import (
    _create_work_from_future,
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
    ReduceOp,
)
from torch.distributed.distributed_c10d import _CollOp, _store_based_barrier, P2POp
from torch.futures import Future
from torch.utils import _pytree as pytree

"""
TODO:
Lots of missing collectives.
Collectives validation.
Make timeout robust by making collectives respect the test deadline.
Make tests robust by making collectives interruptible.
We need some synchronization around cleanup to ensure that timedout ranks don't cause spurious failures.

"""

# 定义函数 flatten_list，用于展开列表
def flatten_list(lst):
    return pytree.tree_leaves(lst)

# 定义函数 ret_work，创建并返回一个包含结果的工作对象
def ret_work(ret):
    fut = Future()
    fut.set_result(ret)
    return _create_work_from_future(fut)

# 定义函数 binop_reduce，对输入的张量列表执行二元操作并返回结果
def binop_reduce(tensors, op):
    res = op(torch.stack(tensors), dim=0)
    if isinstance(res, torch.Tensor):
        return res
    # min/max 返回一个命名元组，这里返回其 values 属性
    return res.values

# 定义函数 bitwise_reduce，对输入的张量列表执行位运算操作并返回结果
def bitwise_reduce(tensors, op):
    return reduce(op, tensors)

# 定义字典 _reduce_ops，将 ReduceOp 枚举值映射到相应的操作函数
_reduce_ops = {
    ReduceOp.SUM: partial(binop_reduce, op=torch.sum),
    ReduceOp.AVG: partial(binop_reduce, op=torch.mean),
    ReduceOp.PRODUCT: partial(binop_reduce, op=torch.prod),
    ReduceOp.MIN: partial(binop_reduce, op=torch.min),
    ReduceOp.MAX: partial(binop_reduce, op=torch.max),
    ReduceOp.BAND: partial(bitwise_reduce, op=torch.bitwise_and),
    ReduceOp.BOR: partial(bitwise_reduce, op=torch.bitwise_or),
    ReduceOp.BXOR: partial(bitwise_reduce, op=torch.bitwise_xor),
}

# 定义类 AllToAll，实现全互联操作
class AllToAll:
    @torch.no_grad()
    def work(self, data):
        world_size = len(data)
        for dest_rank in range(world_size):
            output_tensor_list, _ = data[dest_rank]
            for src_rank in range(world_size):
                _, input_tensor_list = data[src_rank]
                output_tensor_list[src_rank].copy_(input_tensor_list[dest_rank])

# 定义类 AllToAllBase，实现全互联基础操作
class AllToAllBase:
    @torch.no_grad()
    def work(self, data):
        world_size = len(data)
        for dest_rank in range(world_size):
            output_buffer, _, output_split_sizes, _ = data[dest_rank]

            # 计算输出索引
            output_indexes = self._size_cumsum(output_buffer.size(0), output_split_sizes, world_size)

            for src_rank in range(world_size):
                _, input_buffer, _, input_split_sizes = data[src_rank]
                # 计算输入索引
                input_indexes = self._size_cumsum(input_buffer.size(0), input_split_sizes, world_size)

                # 执行数据交换
                output_buffer[output_indexes[src_rank]:output_indexes[src_rank + 1]].copy_(
                    input_buffer[input_indexes[dest_rank]:input_indexes[dest_rank + 1]]
                )
    # 计算累积大小并返回作为张量
    def _size_cumsum(self, buf_size: int, sizes: Union[torch.Tensor, List[int], None], world_size: int) -> torch.Tensor:
        # 如果sizes为None或者空列表，则创建一个全为buf_size // world_size的张量
        if sizes is None or len(sizes) == 0:
            sizes = torch.full(
                (world_size,), buf_size // world_size, dtype=torch.int64
            )
        
        # 如果sizes不是torch.Tensor类型，则将其转换为torch.Tensor
        if not isinstance(sizes, torch.Tensor):
            sizes = torch.tensor(sizes, dtype=torch.int64)
        
        # 确保sizes的数据类型为torch.int64
        assert sizes.dtype == torch.int64
        
        # 计算sizes张量的累积和，首先在sizes张量的开头插入一个零元素，并在给定维度上进行累积和计算
        sizes = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], dtype=torch.int64, device=sizes.device), sizes
                ),
                dim=0
            ),
            dim=0
        )
        
        # 返回计算后的sizes张量
        return sizes
class AllReduce:
    def __init__(self, op):
        # 检查给定的操作是否在支持的操作列表中
        if op.op not in _reduce_ops:
            # 如果不支持，则抛出未实现错误
            raise NotImplementedError(
                f"AllReduce op {op.op} not supported on multithreaded pg for now."
            )
        # 保存操作类型
        self.op = op.op

    @torch.no_grad()
    def work(self, data):
        # 遍历数据的每个元素
        for i in range(len(data[0])):
            tensors = []
            # 将第一个进程的设备作为求和的设备
            rank_0_device = data[0][i].device
            # 收集所有数据到列表，并确保它们在第一个进程的设备上
            for src_rank in range(0, len(data)):
                tensors.append(data[src_rank][i].to(rank_0_device))

            # 执行指定操作（如求和、平均等）以减少数据
            res = _reduce_ops[self.op](tensors)

            # 将减少后的值复制到每个进程的对应数据中
            for src_rank in range(len(data)):
                data[src_rank][i].copy_(res.to(data[src_rank][i].device))


class AllGather:
    @torch.no_grad()
    def work(self, data):
        # 遍历数据中的每个进程
        for src_rank in range(len(data)):
            in_tensor_list = data[src_rank][1]
            # 确保输入的张量列表只有一个张量
            assert len(in_tensor_list) == 1
            src_tensor = in_tensor_list[0]

            # 将源张量广播到所有进程的所有数据中
            for dest in data:
                dest_tensor = dest[0][0][src_rank]
                dest_tensor.copy_(src_tensor)


class Scatter:
    def __init__(self, src):
        # 保存源进程的索引
        self.src = src

    @torch.no_grad()
    def work(self, data):
        # 获取源进程的输入张量列表
        src_in_tensor_list = data[self.src][1]
        # 确保源进程的输入张量列表只有一个张量
        assert len(src_in_tensor_list) == 1
        src_in_tensors = src_in_tensor_list[0]

        # 将源进程的输入张量分发到所有进程的对应输出张量中
        for rank, each_rank_data in enumerate(data):
            out_tensor_list = each_rank_data[0]
            # 确保每个进程的输出张量列表只有一个张量
            assert len(out_tensor_list) == 1
            dest_tensor = out_tensor_list[0]
            dest_tensor.copy_(src_in_tensors[rank])


class Gather:
    def __init__(self, dst):
        # 保存目标进程的索引
        self.dst = dst

    @torch.no_grad()
    def work(self, data):
        # 确保目标进程的输出张量列表只有一个张量
        assert len(data[self.dst][0]) == 1
        out_tensor_list = data[self.dst][0][0]
        # 遍历所有进程的数据
        for rank, each_rank_data in enumerate(data):
            src_in_tensor_list = each_rank_data[1]
            # 确保每个进程的输入张量列表只有一个张量
            assert len(src_in_tensor_list) == 1
            dest_tensor = out_tensor_list[rank]
            dest_tensor.copy_(src_in_tensor_list[0])

class ReduceScatter:
    def __init__(self, op):
        # 检查给定的操作是否为支持的归约操作（如求和、平均）
        if op != dist.ReduceOp.SUM and op != dist.ReduceOp.AVG:
            # 如果不支持，则抛出未实现错误
            raise NotImplementedError(f"ReduceScatter does not support {op}")
        # 保存操作类型
        self.op = op

    @torch.no_grad()
    def work(self, data):
        # 初始化一个布尔列表，用于标记每个rank是否开始了减少操作
        start_reduction = [False for _ in range(len(data))]
        # 遍历每个rank的数据
        for each_rank_data in data:
            # 检查是否只有一个scatter列表
            assert len(each_rank_data[1]) == 1
            # 获取要scatter的数据
            to_scatter = each_rank_data[1][0]
            # 遍历要scatter的数据
            for i in range(len(to_scatter)):
                # 获取目标rank i上的目标张量
                dest_tensor_on_rank_i = data[i][0]
                # 检查目标张量是否只有一个
                assert len(dest_tensor_on_rank_i) == 1
                # 获取目标张量所在设备
                dst_tensor_device = dest_tensor_on_rank_i[0].device
                # 如果还没有开始减少操作，则复制第一个scatter到目标张量
                if not start_reduction[i]:
                    dest_tensor_on_rank_i[0].copy_(to_scatter[i].to(dst_tensor_device))
                    start_reduction[i] = True
                else:
                    # 否则，累加scatter到目标张量
                    dest_tensor_on_rank_i[0].add_(to_scatter[i].to(dst_tensor_device))
        # 如果操作是平均值ReduceOp.AVG
        if self.op == dist.ReduceOp.AVG:
            # 计算rank数量
            num_ranks = len(data)
            # 对每个rank的数据执行平均值操作
            for each_rank_data in data:
                each_rank_data[0][0] /= num_ranks
class Broadcast:
    def __init__(self, src):
        self.src = src

    @torch.no_grad()
    def work(self, data):
        # 将输入数据展平成列表
        in_tensor_list = flatten_list(data[self.src])
        # 遍历数据集的长度
        for i in range(len(data)):
            # 将输出数据展平成列表
            out_tensor_list = flatten_list(data[i])
            # 遍历输入数据列表长度
            for j in range(len(in_tensor_list)):
                # 将输入数据复制到输出数据中
                out_tensor_list[j].copy_(in_tensor_list[j])


class Collective:
    def __init__(self, world_size, collective, pg):
        self._world_size = world_size
        self._collective = collective

        # 创建两个条件变量
        self._start_cond = threading.Condition()
        self._done_cond = threading.Condition()

        # 初始化数据和计数器
        self._data = [None] * world_size
        self._count = 0
        self._done = False

        self._pg = pg

    def join(self, rank, data):
        with self._start_cond:
            # 将数据存入集合
            self._data[rank] = data
            self._count += 1

            # 通知 rank 0
            if self._count == self._world_size:
                if rank > 0:
                    self._start_cond.notify()

            if rank == 0:
                # 等待条件变量，直到所有进程都完成或者终止事件被设置
                self._start_cond.wait_for(
                    lambda: self._count == self._world_size or self._pg._terminate.is_set()
                )
                # 如果发生终止事件，则退出程序
                if self._pg._terminate.is_set():
                    sys.exit("Test termination event occurs.")

        with self._done_cond:
            if rank > 0:
                # 等待条件变量，直到 rank 0 完成或者终止事件被设置
                self._done_cond.wait_for(lambda: self._done or self._pg._terminate.is_set())
                # 如果发生终止事件，则退出程序
                if self._pg._terminate.is_set():
                    sys.exit("Test termination event occurs.")
            else:
                # 执行集合操作，将数据传递到 collective 对象中
                self._collective.work(self._data)
                # 标记完成状态
                self._done = True
                # 通知所有等待的线程完成
                self._done_cond.notify_all()
        # 返回工作的结果数据
        return ret_work(data)


class ProcessLocalGroup(dist.ProcessGroup):
    _coll_lock = threading.Lock()
    _cur_coll_on_pgs = {}

    _terminate = threading.Event()

    @classmethod
    def _start_coll(cls, collective, pg):
        with cls._coll_lock:
            # 如果 pg_name 不在记录中，则创建新的 Collective 实例
            if pg.pg_name not in cls._cur_coll_on_pgs:
                cls._cur_coll_on_pgs[pg.pg_name] = Collective(pg.size(), collective, cls)
            # 返回当前的 Collective 实例
            return cls._cur_coll_on_pgs[pg.pg_name]

    @classmethod
    def _end_coll(cls, collective, pg):
        # 由所有进程竞争性地调用，只有一个进程会执行下面的代码
        with cls._coll_lock:
            # 如果 pg_name 在记录中并且对应的 Collective 实例与给定的 collective 相同，则移除记录
            if pg.pg_name in cls._cur_coll_on_pgs and cls._cur_coll_on_pgs[pg.pg_name] == collective:
                cls._cur_coll_on_pgs.pop(pg.pg_name)
    # 异常处理方法，清除终止标志并通知所有当前正在进行的协程集合
    def exception_handle(cls, exc):
        cls._terminate.set()
        # 遍历当前正在进行的协程集合，并通知其开始条件
        for coll in cls._cur_coll_on_pgs.values():
            with coll._start_cond:
                coll._start_cond.notify()
            # 通知所有协程完成条件
            with coll._done_cond:
                coll._done_cond.notify_all()

    # 类方法，重置类的状态，清空当前协程集合并清除终止标志
    @classmethod
    def reset(cls):
        with cls._coll_lock:
            # 清空当前协程集合
            cls._cur_coll_on_pgs = {}
            # 清除终止标志
            cls._terminate.clear()

    # 执行基本的全互联操作，使用给定的输出和输入缓冲区以及可选的分割大小
    def alltoall_base(
        self,
        output_buffer: torch.Tensor,
        input_buffer: torch.Tensor,
        output_split_sizes: Optional[List[int]],
        input_split_sizes: Optional[List[int]],
        opts=AllToAllOptions()
    ) -> torch.Tensor:
        # 创建一个新的集合，并加入当前进程
        coll = ProcessLocalGroup._start_coll(AllToAllBase(), self)
        # 在集合中当前进程上执行操作，并返回结果
        res = coll.join(self._rank, (output_buffer, input_buffer, output_split_sizes, input_split_sizes))
        # 结束集合
        ProcessLocalGroup._end_coll(coll, self)
        return res

    # 执行全互联操作，用于多个输出和输入张量列表
    def alltoall(self, output_tensor_list, input_tensor_list, opts=AllToAllOptions()):
        coll = ProcessLocalGroup._start_coll(AllToAll(), self)
        res = coll.join(self._rank, (output_tensor_list, input_tensor_list))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    # 执行全局归约操作，用于多个张量列表
    def allreduce(self, tensor_list, opts=AllreduceOptions()):
        coll = ProcessLocalGroup._start_coll(AllReduce(opts.reduceOp), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    # 执行集合的全局归约操作，用于多个张量列表
    def allreduce_coalesced(self, tensor_list, opts=AllreduceOptions()):
        coll = ProcessLocalGroup._start_coll(AllReduce(opts.reduceOp), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    # 执行屏障同步操作，默认发送一个长度为1的张量以触发全局归约
    def barrier(self, opts=BarrierOptions()):
        return self.allreduce(tensor_list=[torch.ones(1)])

    # 执行全收集操作，用于多个输出张量和一个输入张量
    def allgather(self, output_tensors, input_tensor, opts=AllgatherOptions()):
        coll = ProcessLocalGroup._start_coll(AllGather(), self)
        res = coll.join(self._rank, (output_tensors, input_tensor))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    # 执行基本的全收集操作，将一个输出张量分片后进行全收集，并结合一个输入张量
    def _allgather_base(self, output_tensor, input_tensor, opts=AllgatherOptions()):
        tensor_list = list(torch.chunk(output_tensor, self._world_size))
        return self.allgather([tensor_list], [input_tensor], opts)

    # 执行广播操作，将多个张量列表广播给所有进程
    def broadcast(self, tensor_list, opts=BroadcastOptions()):
        coll = ProcessLocalGroup._start_coll(Broadcast(opts.rootRank), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    # 执行分散操作，将多个输出张量分散给所有进程，同时收集多个输入张量
    def scatter(self, output_tensors, input_tensors, opts=ScatterOptions()):
        coll = ProcessLocalGroup._start_coll(Scatter(opts.rootRank), self)
        res = coll.join(self._rank, (output_tensors, input_tensors))
        ProcessLocalGroup._end_coll(coll, self)
        return res
    def gather(self, output_tensors, input_tensors, opts=ScatterOptions()):
        # 使用指定的根节点进行 gather 操作，返回聚集结果
        coll = ProcessLocalGroup._start_coll(Gather(opts.rootRank), self)
        # 将当前进程的输出张量和输入张量传递给协调器对象，获取结果
        res = coll.join(self._rank, (output_tensors, input_tensors))
        # 结束协调操作
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def reduce_scatter(self, output_tensor, scatter_list, opts=ReduceScatterOptions()):
        # 使用指定的 reduce 操作进行 reduce scatter，返回结果
        coll = ProcessLocalGroup._start_coll(ReduceScatter(opts.reduceOp), self)
        # 将当前进程的输出张量和散列列表传递给协调器对象，获取结果
        res = coll.join(self._rank, (output_tensor, scatter_list))
        # 结束协调操作
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts=ReduceScatterOptions()):
        # 将输入张量分块后进行 reduce scatter 操作，并返回结果
        tensor_list = list(torch.chunk(input_tensor, self._world_size))
        return self.reduce_scatter([output_tensor], [tensor_list], opts)

    def reduce_scatter_tensor_coalesced(self, output_tensors, input_tensors, opts=ReduceScatterOptions()):
        # 对多个输出张量和输入张量进行 reduce scatter 操作，并等待完成
        works = [
            self._reduce_scatter_base(output_tensor, input_tensor, opts)
            for output_tensor, input_tensor
            in zip(output_tensors, input_tensors)
        ]
        # 等待所有 reduce scatter 操作完成
        for work in works[:-1]:
            work.wait()
        return works[-1]

    def allgather_into_tensor_coalesced(self, output_tensor_list, input_tensor_list, opts=AllgatherOptions()):
        # 对多个输出张量和输入张量进行 allgather 操作
        res = None
        for o_t, i_t in zip(output_tensor_list, input_tensor_list):
            res = self._allgather_base(o_t, i_t)
        return res

    def __init__(self, rank, world_size):
        # 初始化 ThreadedPG 对象，设置 rank 和 world_size
        super().__init__(rank, world_size)
        self._rank = rank
        self._world_size = world_size
        # 获取全局的分布式通信 world 对象的引用
        world = dist.distributed_c10d._world
        if isinstance(world, ThreadLocalWorld):
            world = world._get_world()
        self._world = weakref.ref(world)
        # 禁用 Torch 的多线程自动求导上下文
        self._ctx = torch.autograd.set_multithreading_enabled(False)

    def size(self):
        # 返回当前分布式组的大小
        return self._world_size

    @property
    def pg_name(self):
        """
        返回当前分布式组在 world 中全局注册的名称
        """
        return self._world().pg_names[self]

    @property
    def group_name(self):
        # 返回当前分布式组的名称
        return self.pg_name

    def getBackendName(self):
        # 返回当前后端名称为 "threaded"
        return "threaded"

    def __repr__(self):
        # 返回对象的字符串表示，包含 world_size 和 rank 信息
        return f"ThreadedPG world_size:{self._world_size} rank:{self._rank}"
# 定义一个函数用于创建线程化的进程组
def _create_threaded_pg(prefix_store, rank, world_size, timeout):
    # 创建一个本地进程组对象，用于管理当前进程的角色和数量
    pg = ProcessLocalGroup(rank, world_size)

    # 根据 GitHub 上的说明，当设备网格涉及子组并且 c10d 中未启用基于存储的障碍时，
    # 即使线程化的进程组假定为单线程，不同的线程可能会独立地初始化不同的组，
    # 从而导致竞态条件。
    # 例如，如果我们有一个网格 [[0, 1], [2, 3]]，子组（维度 0 和 1）将在不同的线程中独立初始化。
    # 在这种情况下，不能再依赖类或全局变量，而必须依赖基于存储的障碍，
    # 确保每个组在可以调用任何组中的集体操作之前都是单独就绪的。
    
    # 在此处调用基于存储的障碍函数，确保每个组都单独就绪
    _store_based_barrier(rank, prefix_store, "", world_size, timeout)

    # 返回创建的进程组对象
    return pg


# 注册 "threaded" 后端到分布式通信框架中，允许在 "cpu" 和 "cuda" 设备上使用线程化进程组
dist.Backend.register_backend("threaded", _create_threaded_pg, devices=["cpu", "cuda"])


# 定义一个数据类，用于存储全局线程相关的数据
@dataclass
class WorldData:
    default_pg: dist.ProcessGroup  # 默认的进程组对象
    pg_map: Dict[dist.ProcessGroup, Tuple[str, Optional[Store]]]  # 进程组映射到名称和存储的字典
    pg_names: Dict[dist.ProcessGroup, str]  # 进程组到名称的映射字典
    pg_group_ranks: Dict[dist.ProcessGroup, Dict[int, int]]  # 进程组到组内角色与全局角色的映射字典
    pg_backend_config: Dict[dist.ProcessGroup, str]  # 进程组到后端配置的映射字典
    group_count: int  # 进程组的数量
    tags_to_pg: Dict[str, List[dist.ProcessGroup]]  # 标签到进程组列表的映射字典
    pg_to_tag: Dict[dist.ProcessGroup, str]  # 进程组到标签的映射字典
    pg_coalesce_state: Dict[dist.ProcessGroup, List[Union[_CollOp, P2POp]]]  # 进程组到协同状态列表的映射字典
    pg_default_device: Dict[dist.ProcessGroup, torch.device]  # 进程组到默认设备的映射字典


# 定义一个线程本地类，用于管理线程局部的世界数据
class ThreadLocalWorld:
    _world = threading.local()  # 使用 threading.local() 创建线程局部变量

    # 获取当前线程的世界数据对象
    def _get_world(self) -> WorldData:
        if not hasattr(ThreadLocalWorld._world, "world"):
            ThreadLocalWorld._world.world = WorldData(None, {}, {}, {}, {}, 0, {}, {}, {}, {})
        return ThreadLocalWorld._world.world

    # 获取默认的进程组对象
    @property
    def default_pg(self):
        return self._get_world().default_pg

    # 设置默认的进程组对象
    @default_pg.setter
    def default_pg(self, value):
        self._get_world().default_pg = value

    # 获取进程组映射字典
    @property
    def pg_map(self):
        return self._get_world().pg_map

    # 获取进程组名称映射字典
    @property
    def pg_names(self):
        return self._get_world().pg_names

    # 获取进程组角色映射字典
    @property
    def pg_group_ranks(self):
        return self._get_world().pg_group_ranks

    # 获取进程组后端配置映射字典
    @property
    def pg_backend_config(self):
        return self._get_world().pg_backend_config

    # 获取进程组数量
    @property
    def group_count(self) -> int:
        return self._get_world().group_count

    # 设置进程组数量
    @group_count.setter
    def group_count(self, value):
        self._get_world().group_count = value

    # 获取标签到进程组列表的映射字典
    @property
    def tags_to_pg(self):
        return self._get_world().tags_to_pg

    # 获取进程组到标签的映射字典
    @property
    def pg_to_tag(self):
        return self._get_world().pg_to_tag

    # 获取进程组协同状态映射字典
    @property
    def pg_coalesce_state(self) -> Dict[dist.ProcessGroup, List[Union[_CollOp, P2POp]]]:
        return self._get_world().pg_coalesce_state
    # 定义一个属性方法，用于获取默认的进程组和对应的设备映射关系
    @property
    def pg_default_device(self) -> Dict[dist.ProcessGroup, torch.device]:
        # 调用内部方法 _get_world() 获取当前环境的全局信息
        return self._get_world().pg_default_device
# 全局变量，用于存储旧的 PyTorch 分布式进程组世界
_old_pg_world = None

# 全局变量，用于存储上下文管理器
_ctx_manager = None

# 定义函数，安装多线程 PyTorch 分布式进程组
def _install_threaded_pg():
    # 声明使用全局变量
    global _old_pg_world
    global _ctx_manager
    # 将当前的分布式进程组世界保存到_old_pg_world中
    _old_pg_world = dist.distributed_c10d._world
    # 设置新的分布式进程组世界为 ThreadLocalWorld 的实例
    dist.distributed_c10d._world = ThreadLocalWorld()
    # 禁用 PyTorch 的多线程支持，并将上下文管理器赋值给_ctx_manager
    _ctx_manager = torch.autograd.set_multithreading_enabled(False)

    # 返回新的分布式进程组世界对象
    return dist.distributed_c10d._world


# 定义函数，卸载多线程 PyTorch 分布式进程组
def _uninstall_threaded_pg():
    # 将保存的旧的分布式进程组世界恢复回dist.distributed_c10d._world
    dist.distributed_c10d._world = _old_pg_world
```