# `.\pytorch\torch\distributed\pipelining\schedules.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import logging  # 导入日志记录模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from collections import defaultdict  # 导入默认字典
from enum import Enum  # 导入枚举类型支持
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union  # 导入类型提示支持

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入分布式训练模块
from torch.profiler import record_function  # 导入性能分析模块

from .microbatch import merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec  # 导入微批处理相关功能
from .stage import _PipelineStageBase  # 导入管道阶段基类


__all__ = [  # 定义可以通过 `from module import *` 导入的符号列表
    "PipelineScheduleSingle",
    "PipelineScheduleMulti",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
]

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class _ComputationType(Enum):  # 定义计算类型枚举类
    FORWARD = 1  # 前向传播
    BACKWARD = 2  # 反向传播
    WEIGHT = 3  # 权重更新

    def __str__(self):  # 定义枚举对象转换为字符串的方法
        str_map = {
            _ComputationType.FORWARD: "F",
            _ComputationType.BACKWARD: "B",
            _ComputationType.WEIGHT: "W",
        }
        return str_map[self]


class _Action(NamedTuple):  # 定义动作命名元组
    computation_type: _ComputationType  # 计算类型
    microbatch_index: int  # 微批次索引
    stage_index: int  # 阶段索引

    def __repr__(self):  # 定义对象的字符串表示形式
        return f"{self.computation_type}{self.microbatch_index}_s{self.stage_index}"


class _PipelineSchedule(ABC):  # 定义管道调度抽象基类
    def __init__(  # 初始化方法
        self,
        n_microbatches: int,  # 微批次数目
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,  # 损失函数（可选）
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,  # 位置参数的分块规格（可选）
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,  # 关键字参数的分块规格（可选）
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,  # 输出合并规格（可选）
    ):
        # From arguments
        self._n_microbatches = n_microbatches  # 设置微批次数目
        self._loss_fn = loss_fn  # 设置损失函数
        self._args_chunk_spec = args_chunk_spec  # 设置位置参数的分块规格
        self._kwargs_chunk_spec = kwargs_chunk_spec  # 设置关键字参数的分块规格
        self._output_merge_spec = output_merge_spec  # 设置输出合并规格
        """
        # args_chunk_spec and kwargs_chunk_spec specify how to chunk inputs.
        # They are used to convert batch to microbatches in `step(x)`.  See
        # `TensorChunkSpec` for helper methods for creating them.
        """
        # args_chunk_spec 和 kwargs_chunk_spec 指定了如何分块输入。
        # 它们用于将批次转换为微批次，见 `step(x)`。参见 `TensorChunkSpec` 获取创建它们的辅助方法。

        # Derived
        self._has_backward = self._loss_fn is not None  # 根据是否有损失函数判断是否有反向传播

        # Holds the losses for each microbatch.
        self._internal_losses: List[torch.Tensor] = []  # 存储每个微批次的损失列表
        logger.info(f"Using {self.__class__.__name__}")  # 记录使用当前类的日志信息，不应出现指令 G004

    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):  # 或许计算损失的方法
        if stage.is_last and self._has_backward:  # 如果是最后一个阶段且有反向传播
            loss = self._compute_loss(output, target_mbs[mb_index])  # 计算损失
            self._internal_losses.append(loss)  # 将损失添加到内部损失列表中
    def _maybe_get_loss(self, stage, mb_index):
        # 检查 microbatch 索引是否在有效范围内，且当前阶段为最后阶段并且支持反向传播
        valid_index = 0 <= mb_index < len(self._internal_losses)
        if stage.is_last and self._has_backward and valid_index:
            # 返回对应 microbatch 索引的损失值
            return self._internal_losses[mb_index]
        elif len(self._internal_losses) != 0 and not valid_index:
            # 若损失列表非空但索引无效，则引发运行时错误
            raise RuntimeError(
                f"Loss for microbatch {mb_index} is not available. "
                f"Available losses for microbatches: {self._internal_losses}"
            )
        else:
            # 其他情况返回 None
            return None

    def _update_losses(self, stages, losses):
        """
        Update the losses to those in the internal state
        """
        # 如果 stages 不是列表，则转换为列表
        if not isinstance(stages, list):
            stages = [stages]
        # 检查 stages 中是否包含最后阶段
        contains_last_stage = any(stage.is_last for stage in stages)

        # 如果包含最后阶段并且 losses 不为 None，则处理损失值
        if contains_last_stage and losses is not None:
            # 检查内部损失列表长度是否与 microbatch 数量一致
            if len(self._internal_losses) != self._n_microbatches:
                # 若长度不一致，则引发运行时错误
                raise RuntimeError(
                    f"Expecting {self._n_microbatches} losses but got {len(self._internal_losses)}"
                )

            # 清空外部 losses 容器
            losses.clear()
            # 将内部损失列表复制到外部 losses 容器中
            losses.extend(self._internal_losses)

        # 清空内部损失列表
        self._internal_losses.clear()

    @abstractmethod
    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the schedule
        implementation.

        Args:
            microbatches: list of microbatch args.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        raise NotImplementedError

    def _check_inputs(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        # 检查输入参数是否为列表或 None，并不作其他具体处理
        pass
    ):
        """
        Pre-process/check inputs
        """

        def check_type_and_len(mbs, name: str):
            # 检查输入的数据类型是否为列表，如果不是则抛出类型错误
            if not isinstance(mbs, list):
                raise TypeError(f"{name} must be a list but got a {type(mbs)}")
            # 检查列表长度是否符合预期的微批次数，如果不符合则抛出值错误
            if len(mbs) != self._n_microbatches:
                raise ValueError(
                    f"Expecting {self._n_microbatches} {name} but got {len(mbs)}"
                )

        if arg_mbs is not None:
            # 如果参数 arg_mbs 不为 None，则调用检查函数检查其类型和长度
            check_type_and_len(arg_mbs, "arg_mbs")
        else:
            # 如果 arg_mbs 为 None，则初始化一个空元组的列表，长度为 self._n_microbatches
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            # 如果参数 kwarg_mbs 不为 None，则调用检查函数检查其类型和长度
            check_type_and_len(kwarg_mbs, "kwarg_mbs")
        else:
            # 如果 kwarg_mbs 为 None，则初始化一个空字典的列表，长度为 self._n_microbatches
            kwarg_mbs = [{}] * self._n_microbatches

        if target_mbs is not None:
            # 如果参数 target_mbs 不为 None，则调用检查函数检查其类型和长度
            check_type_and_len(target_mbs, "target_mbs")

        if losses is not None:
            # 如果参数 losses 不为 None，则检查其类型是否为列表，如果不是则抛出类型错误
            if not isinstance(losses, list):
                raise TypeError(f"losses must be a list but got a {type(losses)}")

        # 返回经过检查和处理后的参数列表和关键字参数列表
        return arg_mbs, kwarg_mbs

    def _compute_loss(self, output, target):
        # 调用内部损失函数计算损失值
        return self._loss_fn(output, target)  # type: ignore[misc]

    def _split_inputs(
        self,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        if args or kwargs:
            # 如果参数 args 或 kwargs 存在，则调用函数将输入拆分成微批次，并返回拆分后的结果
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self._n_microbatches,
                self._args_chunk_spec,
                self._kwargs_chunk_spec,
            )
            return args_split, kwargs_split
        else:
            # 如果 args 和 kwargs 都不存在，则返回一个空元组的列表和空字典的列表，长度为 self._n_microbatches
            # 表示在中间阶段调用时的空输入情况
            return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def _merge_outputs(self, output_chunks: List[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """
        # 调用函数将输出块合并回批处理状态，并根据输出合并规范指定的方式进行合并
        return merge_chunks(
            output_chunks,
            self._output_merge_spec,
        )
def _batch_p2p(p2p_ops: List[dist.P2POp], desc: Optional[str] = None):
    """
    Simple wrapper over batch_isend_irecv from torch.distributed, which just adds a descriptive logger on top.
    """
    # 如果没有操作，则直接返回 None
    if len(p2p_ops) == 0:
        return None
    # 根据描述字符串是否存在，构造描述信息
    desc_str = f"{desc}, " if desc else ""
    # 记录调试信息，包括描述和 p2p_ops 参数
    logger.debug(f"batch_p2p {desc_str}{p2p_ops}")  # noqa: G004
    # 调用 torch.distributed 的 batch_isend_irecv 方法，并弹出结果
    return dist.batch_isend_irecv(p2p_ops).pop()


def _sorted_batch_p2p(
    p2p_ops: List[dist.P2POp], desc: Optional[str] = None
) -> Dict[int, dist.Work]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # 按照 peer rank 对 P2P 操作进行排序
    #   int 是 peer rank；
    #   List 是向该 peer 的操作列表
    ops_by_peer: Dict[int, List[dist.P2POp]] = defaultdict(list)
    work_by_peer: Dict[int, dist.Work] = {}
    # 如果没有操作，则直接返回空字典
    if len(p2p_ops) == 0:
        return work_by_peer

    # 根据 peer rank 对操作进行分类
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # 按照 peer rank 的顺序调用 batch_isend_irecv，并保存每个 peer 的工作结果
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = _batch_p2p(ops, desc=desc)

    return work_by_peer


class PipelineScheduleSingle(_PipelineSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.
    """

    def __init__(
        self,
        stage: _PipelineStageBase,
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # 调用父类构造函数进行初始化
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
        )
        # 设置对象的特定属性
        self._stage = stage
        self._num_stages = stage.num_stages
        # 为 stage 对象设置相同的 has_backward 标志
        self._stage.has_backward = self._has_backward

        # TODO: later replace this with lazy shape inference during forward
        # 准备用于阶段前向传播发送/接收的基础设施
        stage._prepare_forward_infra(n_microbatches)
        # 如果具有后向传播，则为阶段准备后向传播基础设施
        if self._has_backward:
            stage._prepare_backward_infra(n_microbatches)
    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """

        # 清除每次迭代时的运行时状态
        self._stage.clear_runtime_states()

        # 将输入拆分成微批次
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # 如果存在目标，则将目标拆分成微批次
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # 运行微批次
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # 如果是最后一个阶段，则返回按原始格式合并的结果
        if self._stage.is_last:
            return self._merge_outputs(self._stage.output_chunks)
        else:
            return None
    """
    The GPipe schedule.
    Will go through all the microbatches in a fill-drain manner.
    """

    # 定义 ScheduleGPipe 类，继承自 PipelineScheduleSingle 类
    class ScheduleGPipe(PipelineScheduleSingle):

        # 定义 _step_microbatches 方法，用于执行一次 GPipe 调度的迭代过程
        def _step_microbatches(
            self,
            arg_mbs: Optional[List] = None,
            kwarg_mbs: Optional[List] = None,
            target_mbs: Optional[List] = None,
            losses: Optional[List] = None,
        ):
            """
            Run one iteration of the pipeline schedule with list of microbatches.
            Will go through all the microbatches according to the GPipe schedule.

            Args:
                microbatches: list of microbatch args.
            """

            # 检查输入参数并赋值给本地变量
            arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

            # 延迟发送等待列表
            fwd_sends_to_wait: List[dist.Work] = []

            # 执行微批次
            for i in range(self._n_microbatches):
                # 记录前向传播操作的记录函数
                with record_function(f"Forward {i}"):
                    # 获取前向接收操作并按照顺序排序
                    ops = self._stage.get_fwd_recv_ops(i)
                    works = _sorted_batch_p2p(ops, desc="fwd_recv")
                    for work in works.values():
                        work.wait()  # 等待操作完成

                    # 执行当前微批次的前向传播，并获取输出结果
                    output = self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                    # 获取前向发送操作并按照顺序排序
                    ops = self._stage.get_fwd_send_ops(i)
                    works = _sorted_batch_p2p(ops, desc="fwd_send")
                    fwd_sends_to_wait.extend(works.values())  # 将发送操作加入等待列表

                # 记录调试信息，指示已完成微批次的前向传播
                logger.debug(
                    f"[{self._stage.stage_index}] Forwarded microbatch {i}"  # noqa: G004
                )

                # 可能计算损失函数
                self._maybe_compute_loss(self._stage, output, target_mbs, i)

            # 等待所有前向发送操作完成
            for work in fwd_sends_to_wait:
                work.wait()

            # 如果没有后向传播，直接返回
            if not self._has_backward:
                return

            # 执行后向传播
            # 延迟发送等待列表
            bwd_sends_to_wait: List[dist.Work] = []
            for i in range(self._n_microbatches):
                # 记录后向传播操作的记录函数
                with record_function(f"Backward {i}"):
                    # 获取后向接收操作并按照顺序排序
                    ops = self._stage.get_bwd_recv_ops(i)
                    works = _sorted_batch_p2p(ops, desc="bwd_recv")
                    for work in works.values():
                        work.wait()  # 等待操作完成

                    # 获取当前微批次的损失值（如果有）
                    loss = self._maybe_get_loss(self._stage, i)
                    # 执行当前微批次的后向传播
                    self._stage.backward_one_chunk(i, loss=loss)

                    # 获取后向发送操作并按照顺序排序
                    ops = self._stage.get_bwd_send_ops(i)
                    works = _sorted_batch_p2p(ops, desc="bwd_send")
                    bwd_sends_to_wait.extend(works.values())  # 将发送操作加入等待列表

                # 记录调试信息，指示已完成微批次的后向传播
                logger.debug(
                    f"[{self._stage.stage_index}] Backwarded microbatch {i}"  # noqa: G004
                )

            # 更新损失函数容器中的损失值（如果有）
            self._update_losses(self._stage, losses)

            # 等待所有后向发送操作完成
            for work in bwd_sends_to_wait:
                work.wait()
class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        # 实现 PipelineScheduleSingle 类的 _step_microbatches 方法
        # 接受多个可选的微批次参数和损失列表
        pass


class PipelineScheduleMulti(_PipelineSchedule):
    """
    Base class for multi-stage schedules.
    Implements the `step` method.
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        stage_index_to_group_rank: Optional[Dict[int, int]] = None,
    ):
        # 如果提供的阶段数量少于等于1，则抛出值错误
        if len(stages) <= 1:
            raise ValueError(
                f"Multi-stage schedule expects at least two stages but got {len(stages)}"
            )
        # 调用父类的构造方法，初始化基本参数
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
        )
        # 设置对象的属性
        self._stages = stages  # 存储阶段列表
        self._num_stages = stages[0].num_stages  # 存储阶段数量
        self.pp_group_size = stages[0].group_size  # 存储阶段的组大小
        self.rank = stages[0].group_rank  # 存储阶段的组排名

        # 如果提供了阶段索引到组排名的映射，则对每个阶段设置映射
        if stage_index_to_group_rank is not None:
            for stage in self._stages:
                stage.stage_index_to_group_rank = stage_index_to_group_rank
        self.stage_index_to_group_rank = stages[0].stage_index_to_group_rank  # 存储第一个阶段的阶段索引到组排名的映射

        # 对每个阶段设置相同的 has_backward 标志
        for stage in self._stages:
            stage.has_backward = self._has_backward

        # 定义一个函数，用于确定是否应计算损失，仅在最后一个阶段且设置了损失函数时返回 True
        self._should_compute_loss = (
            lambda stage: stage.is_last and self._loss_fn is not None
        )

        # 这将在派生调度程序的初始化期间设置
        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}  # 存储管道的执行顺序
        self.use_full_backward = True  # 使用完整的反向传播标志

        # TODO: 后续替换为在前向传播期间进行懒惰形状推断
        # 为每个阶段准备前向发送/接收基础设施
        for stage in self._stages:
            stage._prepare_forward_infra(n_microbatches)
            if self._has_backward:
                stage._prepare_backward_infra(n_microbatches)
    # 定义一个方法用于执行单个步骤的管道调度，接收多个位置参数和关键字参数
    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """

        # 每次迭代前清理各阶段的运行时状态
        for stage in self._stages:
            stage.clear_runtime_states()

        # 将输入拆分成微批次
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # 如果有指定目标，则将目标数据也拆分成微批次
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # 运行微批次
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # 根据原始格式返回合并的结果
        for stage in self._stages:
            if stage.is_last:
                return self._merge_outputs(stage.output_chunks)
        # 如果不包含最后一个阶段，则返回空值
        return None

    # 用于执行微批次处理的内部方法
    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
        ):
class ScheduleLoopedBFS(PipelineScheduleMulti):
    """
    Breadth-First Pipeline Parallelism.
    See https://arxiv.org/abs/2211.05953 for details.
    Simliar to Interleaved 1F1B, Looped BFS supports multiple stages per rank.
    What is different is that when microbatches are ready for multiple local
    stages, Loops BFS will prioritizes the earlier stage, running all available
    microbatches at once.
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}
        # ========================================================================
        # Initialize pipeline_order for each rank
        for rank in range(self.pp_group_size):
            # Calculate operations for each rank
            rank_ops = self._calculate_single_rank_operations(rank)
            # Assign calculated operations to pipeline_order
            self.pipeline_order[rank] = rank_ops

    def _calculate_single_rank_operations(self, rank):
        # Determine number of local stages
        n_local_stages = len(self._stages)
        # Calculate stage indices for the current rank
        stage_indices = range(
            rank, self.pp_group_size * n_local_stages, self.pp_group_size
        )

        # Store the list of operations used for that rank
        rank_ops: List[Optional[_Action]] = []
        
        # Pre-padding: Insert no-ops based on the warmup phase
        for _ in range(rank):
            rank_ops.append(None)

        # Generate operations for each stage and microbatch index
        for stage_index in stage_indices:
            for mb_index in range(self._n_microbatches):
                rank_ops.append(
                    _Action(_ComputationType.FORWARD, mb_index, stage_index)
                )

        # Calculate post-warmup operations based on the rank
        post_warmup_ops = 2 * (self.pp_group_size - 1 - rank)
        rank_ops.extend([None] * post_warmup_ops)

        # Generate backward operations in reverse order for each stage and microbatch index
        for stage_index in reversed(stage_indices):
            for mb_index in reversed(range(self._n_microbatches)):
                rank_ops.append(
                    _Action(_ComputationType.BACKWARD, mb_index, stage_index)
                )
        return rank_ops


class ScheduleInterleaved1F1B(PipelineScheduleMulti):
    """
    The Interleaved 1F1B schedule.
    See https://arxiv.org/pdf/2104.04473 for details.
    Will perform one forward and one backward on the microbatches in steady
    state and supports multiple stages per rank. When microbatches are ready for
    multiple local stages, Interleaved 1F1B prioritizes the earlier microbatch
    (also called "depth first").
    """
    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # 初始化方法，接受多个参数用于配置管道和微批处理
        # 设置管道阶段的组大小
        self.pp_group_size = stages[0].group_size
        # 检查微批处理数是否能被组大小整除，否则抛出异常
        if n_microbatches % self.pp_group_size != 0:
            raise ValueError(
                f"Interleaved 1F1B schedule requires the number of microbatches ({n_microbatches}) \
                to be a multiple of the number of pipeline ranks ({self.pp_group_size})."
            )

        # 调用父类的初始化方法，传递参数设置管道
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
        )

        # 设置本地阶段数目和第一个阶段的组等级和组信息
        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank
        self.group = stages[0].group

        # 1. 创建管道顺序 (所有等级都执行此计算)
        # 这将用于跟踪整个管道的当前状态
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        # 初始化管道顺序的字典结构
        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}

        # 遍历组大小，为每个等级计算单个等级的操作
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops
```