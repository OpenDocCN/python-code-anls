# `.\pytorch\torch\distributed\pipelining\stage.py`

```
    """
    Base class for pipeline stages.
    Defines or implements common methods used by the `_PipelineStage` used by
    the tracing frontend and `PipelineStage` used by manual frontend.
    """
    def __init__(
        self,
        submodule: torch.nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[[], None]]] = None,
    ):
        """
        Constructor for _PipelineStageBase.

        Args:
            submodule (torch.nn.Module): The submodule associated with this stage.
            stage_index (int): Index of this stage in the pipeline.
            num_stages (int): Total number of stages in the pipeline.
            device (torch.device): Device on which the stage operates.
            group (Optional[dist.ProcessGroup], optional): Process group for distributed training. Defaults to None.
            dw_builder (Optional[Callable[[], Callable[[], None]]], optional):
                Function builder for distributed worker processes. Defaults to None.
        """
        self.submodule = submodule
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device
        self.group = group
        self.dw_builder = dw_builder
        self._has_backward = False

    @property
    def has_backward(self) -> bool:
        """
        Returns true if this stage has a backward pass.
        """
        return self._has_backward

    @has_backward.setter
    def has_backward(self, has_backward: bool):
        """
        Setter for the `has_backward` property.

        Args:
            has_backward (bool): True if this stage has a backward pass, False otherwise.
        """
        self._has_backward = has_backward

    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
        return self.stage_index == 0
    # 判断当前阶段是否是管道中的最后一个阶段，返回布尔值
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
        return self.stage_index == self.num_stages - 1

    # 检查给定的块ID是否有效，如果块未配置则引发运行时错误
    def _check_chunk_id(self, chunk_id: int):
        if self.chunks is None:
            raise RuntimeError(
                "Attempted to access chunk_id before chunks have been configured."
            )
        if chunk_id >= self.chunks:
            raise RuntimeError(
                f"Chunk id {chunk_id} is out of range [0, {self.chunks})"
            )

    # 配置输出元数据，跟踪此阶段的输出形状/数据类型，以确保与下一阶段的接收操作匹配
    def _configure_outputs_meta(self, outputs_meta: Tuple[torch.Tensor, ...]):
        """
        Track the output shapes/dtype of this stage since they determine the send operation(s) which must match
        recv operations of the next stage.  The next stage _will_ be freezing its recv buffers based on its initial
        configuration, so it's important to also freeze/validate the output side to avoid any send/recv mismatches
        which could show up as hangs, silent corruption, or other errors.
        """
        assert (
            self._outputs_meta is None
        ), "Attempting to reconfigure output_meta, which is not supported"
        self._outputs_meta = tuple(outputs_meta)  # type: ignore[assignment]

    # 获取输出元数据，表示此阶段输出的元数据（元张量）
    def get_outputs_meta(self) -> Tuple[torch.Tensor, ...]:
        """Get the output metadata (meta tensors) representing the outputs of this stage"""
        assert (
            self._outputs_meta is not None
        ), "Attempted to get_outputs_meta() without configuring output meta"
        return self._outputs_meta

    # 创建梯度发送信息列表，指示将梯度发送给哪些阶段索引
    def _create_grad_send_info(
        self,
        args_recv_info: Tuple,
    ) -> List[Optional[int]]:
        """
        Create a list of stage indices to send gradients to.
        """
        grad_send_info: List[Optional[int]] = []

        def map_recv_to_send(a):
            # 注意：我们将梯度发送回上一个阶段，只要在前向传播中它是一个接收到的输入，无论是否需要梯度。
            # 由上一个阶段决定是否丢弃此梯度。
            if isinstance(a, _RecvInfo):
                grad_send_info.append(a.source)
                return a.source
            else:
                grad_send_info.append(None)
                return None

        map_aggregate(args_recv_info, map_recv_to_send)

        logger.debug(
            f"{self.log_prefix} Grad send info: {grad_send_info}"  # noqa: G004
        )
        return grad_send_info

    # 抽象方法：准备前向推断的基础设施，使用微批次的数量作为参数
    @abstractmethod
    def _prepare_forward_infra(self, num_microbatches: int):
        raise NotImplementedError
    def _prepare_backward_infra(self, num_microbatches: int):
        # TODO: this is needed for backward_maybe_with_nosync
        self.chunks = num_microbatches
        # 设置对象的 `chunks` 属性为给定的微批次数量

        for mb_index in range(num_microbatches):
            # `grad_recv_info` 是 `act_send_info` 的镜像
            self.grad_recv_info[mb_index] = self._create_grad_recv_info(
                self.act_send_info
            )
            # 使用 `act_send_info` 创建 `grad_recv_info` 中的接收梯度信息

    @abstractmethod
    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ) -> Tuple[_RecvInfo, ...]:
        raise NotImplementedError
        # 抽象方法：根据 `act_send_info` 创建接收梯度信息的元组

    def _get_recv_ops(
        self,
        recv_infos: Tuple[InputInfo, ...],
    ) -> List[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: List[dist.P2POp] = []
        # 初始化空的操作列表

        for info in recv_infos:
            if not isinstance(info, _RecvInfo):
                continue
            # 遍历接收信息元组，筛选出 `_RecvInfo` 类型的信息

            peer_rank = self.stage_index_to_group_rank[info.source]
            peer_global_rank = (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )  # TODO
            # 获取发送者的全局等级

            ops.append(
                dist.P2POp(dist.irecv, info.buffer, peer_global_rank, self.group)
            )
            # 将接收操作添加到操作列表中

        return ops
        # 返回操作列表

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: Tuple[InputInfo, ...] = self.args_recv_info[fwd_chunk_id]
        # 获取前向传递阶段的接收信息元组

        # In case there is backward pass, set requires_grad for receive buffers
        # before first forward
        if self.has_backward and not self.set_requires_grad[fwd_chunk_id]:
            for a in recv_infos:
                if isinstance(a, _RecvInfo):
                    a.buffer.requires_grad_(True)
        # 如果存在反向传递，并且接收缓冲区尚未设置梯度要求，则在第一次前向传递前设置接收缓冲区的梯度要求为真

        return self._get_recv_ops(recv_infos)
        # 返回根据接收信息元组获取的操作列表

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []
        # 如果不存在反向传递或者当前阶段是最后一个阶段，则返回空列表

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        # 获取反向传递阶段的接收梯度信息

        return self._get_recv_ops(recv_infos)
        # 返回根据接收梯度信息获取的操作列表
    def get_fwd_send_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Get the activation send operations for the forward pass of the current stage.
        """
        # Retrieve the output corresponding to the forward chunk ID
        output = self.output_chunks[fwd_chunk_id]
        
        # Ensure output is in tuple form for consistent handling with `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)
        
        # Initialize an empty list to store P2POp operations
        ops: List[dist.P2POp] = []

        # Iterate over each index and corresponding output tensor in output_tuple
        for idx, out in enumerate(output_tuple):
            # Retrieve destinations for sending activation data from act_send_info
            dst_stages = self.act_send_info[idx]
            
            # Iterate over each destination stage
            for dst in dst_stages:
                if dst is None:
                    continue
                
                # Log debug information about sending tensor to the destination stage
                logger.debug(
                    f"{self.log_prefix} "
                    f"Sending tensor to Stage {dst}: {out.size()}"
                )
                
                # Determine the rank of the destination stage in the communication group
                peer_rank = self.stage_index_to_group_rank[dst]
                
                # Obtain global rank if a communication group is defined
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )  # TODO
                
                # Create a P2POp instance for sending the tensor to the destination
                ops.append(dist.P2POp(dist.isend, out, peer_global_rank, self.group))

        # Return the list of send operations
        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Get the gradient send operations for the backward pass of the current stage.
        """
        # Validate the backward chunk ID against internal checks
        self._check_chunk_id(bwd_chunk_id)

        # If no backward pass is required or it's the first stage, return empty list
        if not self.has_backward or self.is_first:
            return []

        # Lazily create the gradient send infrastructure if not already initialized
        if self.grad_send_info is None:
            # Determine destinations for gradient sending based on received arguments
            # Mirror of args_recv_info, used during backward propagation
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        # Initialize an empty list to store P2POp operations
        ops: List[dist.P2POp] = []

        # Iterate over each input gradient and its corresponding receiving stage
        for grad, grad_recv_stage in zip(self.grads_input, self.grad_send_info):
            # Check if the gradient is a tensor and if there's a valid receiving stage
            if isinstance(grad, torch.Tensor) and grad_recv_stage is not None:
                # Log debug information about sending gradient to the receiving stage
                logger.debug(
                    f"{self.log_prefix} "
                    f"Sending gradient to Stage {grad_recv_stage}: {grad.size()}"
                )
                
                # Determine the rank of the receiving stage in the communication group
                peer_rank = self.stage_index_to_group_rank[grad_recv_stage]
                
                # Obtain global rank if a communication group is defined
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )  # TODO
                
                # Create a P2POp instance for sending the gradient tensor
                ops.append(dist.P2POp(dist.isend, grad, peer_global_rank, self.group))
            else:
                # Raise an error if there's a mismatch in gradient and expected behavior
                if not (grad is None and grad_recv_stage is None):
                    raise RuntimeError(
                        f"[{self.stage_index}] for chunk {bwd_chunk_id - 1} has gradients {grad} "
                        f"and is expecting to send gradients to stage {grad_recv_stage}"
                    )

        # Return the list of send operations
        return ops
    def clear_runtime_states(self) -> None:
        """
        Clear runtime states of the stage.
        """
        # 清空正向计算缓存，将微批次 ID 映射到正向张量参数列表
        self.fwd_cache.clear()
        # 清空输出块缓存，用于最终输出合并或减少
        self.output_chunks.clear()
        # 重置反向计算块计数器
        self._seen_bwd_chunks = 0

        # 在调度步骤之间清除输入缓冲区的梯度。这是因为 `torch.autograd.backward()` 默认会累积梯度到叶子张量中。
        # 为了使梯度传递回前面的阶段，我们不希望进行这样的累积。
        for recv_tuple in self.args_recv_info.values():  # 遍历所有块
            for a in recv_tuple:  # 遍历所有输入参数
                if isinstance(a, _RecvInfo):
                    # 将梯度设置为 None 是清除梯度的推荐方式，而不是使用 `zero_()`。
                    # 参考 https://github.com/pytorch/pytorch/pull/92731
                    a.buffer.grad = None

    def _map_tensor_from_recv_info(
        self,
        recv_infos: Tuple[InputInfo, ...],
    ):
        """
        Map tensors from recv infos to a list.
        """

        def get_recv_tensor(info):
            if isinstance(info, _RecvInfo):
                return info.buffer
            else:
                raise AssertionError(f"Expected _RecvInfo but got {type(info)}")

        # 使用 `map_aggregate` 函数将接收信息映射为张量列表
        tensors = map_aggregate(
            recv_infos,
            get_recv_tensor,
        )

        return tensors

    def _retrieve_recv_activations(self, fwd_chunk_id: int):
        """
        Retrieve the activations received for the current stage during forward.
        """
        recv_infos = self.args_recv_info[fwd_chunk_id]
        # 获取从前向阶段接收到的激活值
        activations = self._map_tensor_from_recv_info(recv_infos)
        return activations

    def _retrieve_recv_grads(
        self,
        bwd_chunk_id: int,
    ):
        """
        Retrieve the gradients received for the current stage during backward.
        """
        recv_infos = self.grad_recv_info[bwd_chunk_id]
        # 获取从反向阶段接收到的梯度
        grads = self._map_tensor_from_recv_info(recv_infos)
        return grads

    def forward_maybe_with_nosync(self, *args, **kwargs):
        # 如果子模块被包装为分布式数据并行模型（DDP），使用 `no_sync` 上下文管理器来避免每个微批次的梯度全局归约
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # 使用 `no_sync` 上下文管理器
                out_val = self.submod(*args, **kwargs)
        else:
            out_val = self.submod(*args, **kwargs)
        return out_val
    def backward_maybe_with_nosync(self, bwd_kwargs: Dict):
        """
        Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """
        # 判断是否是最后一个反向传播步骤
        last_backward = self._seen_bwd_chunks == self.chunks - 1  # type: ignore[operator]

        # 如果子模块被包装为 DDP（分布式数据并行）
        if isinstance(self.submod, DistributedDataParallel):
            if last_backward:
                # 最后一个分块，准备进行梯度归约
                # HACK: 这里涉及到 DDP 实现的内部细节。是否有更好的方式？
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                # 执行反向传播阶段
                grads_input = stage_backward(**bwd_kwargs)
            else:
                # 在非最后一个分块时，使用 no_sync 上下文管理器来执行反向传播
                with self.submod.no_sync():  # type: ignore[operator]
                    grads_input = stage_backward(**bwd_kwargs)
        # 如果子模块是 FSDP 模块
        elif isinstance(self.submod, FSDPModule):
            # 设置是否是最后一个反向传播步骤
            self.submod.set_is_last_backward(last_backward)
            # 设置是否需要梯度同步
            self.submod.set_requires_gradient_sync(last_backward)
            # 执行反向传播阶段
            grads_input = stage_backward(**bwd_kwargs)
        else:
            # 非数据并行的子模块，执行常规的反向传播
            grads_input = stage_backward(**bwd_kwargs)

        # 更新已经处理的反向传播分块数目
        self._seen_bwd_chunks += 1
        # 返回计算得到的梯度输入
        return grads_input
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage. They
        applies only to the first stage in most cases.
        """

        if self.is_first:
            # 如果是第一个阶段，则不需要接收任何输入
            composite_args = args
            composite_kwargs = kwargs or {}
        else:
            # 接收本分块的激活值
            # 激活值仅以 args 形式传入
            composite_args = self._retrieve_recv_activations(fwd_chunk_id)
            composite_kwargs = {}

        self._validate_fwd_input(args, kwargs)

        # 计算前向传播
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        if type(output) is list:
            # HACK: 这是对导出创建列表格式输出的一种临时解决方法
            output = tuple(output)

        # 统一输出形式为元组，以便与 `act_send_info` 对应
        output_tuple = output if type(output) is tuple else (output,)
        # 准备最终输出的合并或减少
        self.output_chunks.append(output)

        # 保存反向传播所需的激活值和输入值
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        logger.debug(
            f"{self.log_prefix} Forwarded chunk {fwd_chunk_id}, outputs: {map_debug_info(output)}"  # noqa: G004
        )
        self._validate_fwd_outputs(output_tuple)
        return output
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is required that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.
        """

        # 如果 full_backward 参数为 False，则需要确保在 PipelineStage 的初始化阶段提供了 dw_builder
        if not full_backward:
            assert self.dw_builder, "Must provide dw_builder to run partial backward"

        # 检查 bwd_chunk_id 的有效性
        self._check_chunk_id(bwd_chunk_id)

        # 从缓存中取出前向传播阶段计算得到的输出和输入值
        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # 计算反向传播
        if self.is_last:
            # 如果是最后一个阶段，从损失计算梯度，没有来自下一阶段的梯度信息
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
            # 否则，从下一阶段接收梯度信息
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)
            # 如果流水线中的输入需要梯度，`torch.autograd.backward` 将把梯度累积到这些输入的 `.grad` 字段中
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        # 调用反向传播函数，并可能不同步更新梯度
        self.grads_input = self.backward_maybe_with_nosync(bwd_kwargs)
        logger.debug(f"{self.log_prefix} Backwarded chunk {bwd_chunk_id}")  # noqa: G004

        # 如果不是完整的反向传播，确保提供了 dw_builder，并且还未运行过 `backward_weight_one_chunk`
        if not full_backward:
            assert self.dw_builder, "Must provide dw_builder to run partial backward"
            assert bwd_chunk_id not in self.dw_runner, (
                f"{self.log_prefix} Attempted to run partial backward for chunk {bwd_chunk_id}"
                " repeatedly without calling `backward_weight_one_chunk`"
            )
            # 创建并存储权重更新的运行器 dw_runner
            dw_runner = self.dw_builder()
            self.dw_runner[bwd_chunk_id] = dw_runner
        # 如果是完整的反向传播，并且提供了 dw_builder，则直接调用 dw_builder
        elif self.dw_builder:
            self.dw_builder()()

    # 运行单个 chunk 的权重更新反向传播
    def backward_weight_one_chunk(self, bwd_chunk_id: int):
        assert bwd_chunk_id in self.dw_runner, (
            f"{self.log_prefix} Attempted to run backward_weight_one_chunk for chunk {bwd_chunk_id}"
            " without first calling `backward_one_chunk(full_backward=False)`"
        )
        # 移除并执行指定 chunk 的权重更新运行器
        self.dw_runner.pop(bwd_chunk_id)()
    def _validate_fwd_input(self, args, kwargs):
        """Raises a RuntimeError if shapes of input args/kwargs do not match the shapes configured for this stage."""
        
        # 如果这个阶段是第一个阶段
        if self.is_first:
            # TODO 为什么每个管道块都有一个单独的 recv_info？
            # kwen2501: 为了避免将 fwd_chunk_id 传递给这个函数，我们
            # 检查所有块是否与 args_recv_info[0] 匹配
            expected_args = self.args_recv_info[0]
        else:
            # 在典型的管道场景中，我们假设非第一个阶段不会接受用户输入，因此不做输入验证
            return

        if len(kwargs):
            # TODO- 需要一个将 kwarg 映射到 self.args_recv_info 中位置的映射
            # 如果没有这个映射，我们只验证 args 的形状，而忽略 kwargs
            expected_args = expected_args[: len(expected_args) - len(kwargs)]

        # TODO- 需要一个将 kwarg 映射到 self.args_recv_info 中位置的映射
        # 或许很难确定长度不匹配是因为：
        # (a) 用户传递了额外的参数或者漏掉了参数
        # (b) 用户没有传递一个带有默认值的 kwargs，该值已经包含在 expected_args 中
        expected_tensors_meta = [
            e.meta if isinstance(e, _RootArgPlaceholder) else e.buffer
            for e in expected_args
        ]
        validate_tensors_metadata(
            f"Stage {self.stage_index} forward inputs", expected_tensors_meta, args
        )

    def _validate_fwd_outputs(self, outputs: Tuple[torch.Tensor, ...]):
        """Raises a RuntimeError if this stage produces an output of unexpected shape/dtype.
        Most likely, this could be cause either by incorrect user specification of output shapes, or because
        shape inference was done on the original model but then at runtime the model is wrapped with something like
        mixed precision which changes output dtype.
        """
        expected_tensors_meta = self.get_outputs_meta()
        validate_tensors_metadata(
            f"Stage {self.stage_index} forward outputs", expected_tensors_meta, outputs
        )
class _`
class _PipelineStage(_PipelineStageBase):
    # 初始化 PipelineStage 类，接收必要参数和配置
    def __init__(
        self,
        stage_module: torch.nn.Module,  # 要包装的模块
        stage_index: int,  # 当前阶段的索引
        pipe_info: PipeInfo,  # 管道信息，描述管道的阶段关系
        device: torch.device,  # 用于该阶段的设备
        group: Optional[dist.ProcessGroup] = None,  # 可选的进程组
    ):
        """
        创建一个管道阶段，给定一个要由此阶段包装的 stage_module 
        和描述该阶段关系的 pipe_info。

        Args:
            stage_module (torch.nn.Module): 要被此阶段包装的模块
            stage_index (int): 该阶段在管道中的索引
            pipe_info (PipeInfo): 管道信息，可以通过 `pipe.info()` 获取
            device (torch.device): 此阶段要使用的设备
            group (Optional[dist.ProcessGroup]): 此阶段要使用的进程组
        """
        # 调用基类初始化方法，传入相关参数
        _PipelineStageBase.__init__(
            self,
            stage_module,
            stage_index,
            pipe_info.num_stages,  # 管道的总阶段数
            device,
            group,
        )
        # 存储管道信息
        self.pipe_info = pipe_info

        # 在图中查找阶段节点
        submod_nodes = [
            node for node in pipe_info.graph.nodes if node.op == "call_module"
        ]
        # 检查图中的子模块节点数是否与阶段数匹配
        if len(submod_nodes) != self.num_stages:
            raise AssertionError(
                f"Number of submodules in pipe graph {len(submod_nodes)} does not match number of stages {self.num_stages}"
            )

        # 找到当前阶段的节点
        self.node = submod_nodes[self.stage_index]
        self.name = self.node.name
        # 记录日志，标明创建了一个新的 PipelineStage
        logger.info(
            f"[{self.group_rank}] "  # noqa: G004
            f"Creating PipelineStage {stage_index} for {self.name}"
        )

        # 创建阶段名称到阶段索引的映射
        self.submod_to_stage_index: Dict[str, int] = {}
        for i, node in enumerate(submod_nodes):
            self.submod_to_stage_index.setdefault(node.name, i)

        # 移动子模块到指定设备
        self._move_submod_to_device()

    def _move_submod_to_device(self):
        # 尝试将子模块移动到指定设备
        # 注意：由于元模块的参数不支持 to() 方法，不能将元模块移动到实际设备，需要进行就地张量交换
        has_meta_param = any(
            isinstance(p, FakeTensor) or p.is_meta for p in self.submod.parameters()
        )
        if has_meta_param:
            logger.debug(f"{self.log_prefix} Found meta parameters!")  # noqa: G004
        else:
            self.submod.to(self.device)
    def _prepare_forward_infra(self, num_microbatches: int):
        """
        Create send/recv infrastructures for activations (during forward)
        """
        # 对每个微批次的标志进行追踪，记录是否已经设置了接收缓冲区的`requires_grad`属性。格式：{chunk : Boolean}
        for chunk in range(num_microbatches):
            self.args_recv_info[chunk] = self._create_act_recv_info()
            self.set_requires_grad[chunk] = False

        # 在前向传播期间为每个激活创建发送信息
        self.act_send_info = self._create_act_send_info()

    def get_stage_index_of_submod(
        self,
        submod_name: str,
    ):
        """
        Given a submodule name, return the stage index of the submodule.
        """
        # 给定子模块名称，返回子模块的阶段索引
        if submod_name not in self.submod_to_stage_index:
            raise AssertionError(f"Stage id of {submod_name} not found")

        return self.submod_to_stage_index[submod_name]

    def _create_act_recv_info(
        self,
    ):
        """
        Create a tuple of `_RecvInfo` for inputs to the stage.
        """

        def create_recv_tensor(placeholder, arg_node):
            """
            Create a receive buffer for a placeholder.
            """
            example_value = placeholder.meta["val"]
            if arg_node.op == "placeholder":
                # This is a root level placeholder, thus an input argument to the entire model.
                # We are likely at stage 0, hence no need to create a receive buffer.
                return _RootArgPlaceholder(example_value)

            # Figure out the source stage of this input
            while arg_node.target is operator.getitem:
                # If the input is a getitem, we need to go deeper
                arg_node = arg_node.args[0]

            assert (
                arg_node.op == "call_module"
            ), f"Expecting call_module, got {arg_node.op}"
            src_stage = self.get_stage_index_of_submod(arg_node.name)

            # Create a receive buffer for this placeholder
            logger.debug(
                f"{self.log_prefix} "  # noqa: G004
                f"Creating recv buffer for input '{placeholder.name}' "
                f": {example_value.shape}, {example_value.dtype}"
            )
            buffer = _make_tensor_from_meta(example_value, self.device)

            return _RecvInfo(
                arg_node.name,
                src_stage,
                buffer,
            )

        args_recv_info: List[InputInfo] = []
        # Filter out placeholder nodes from `self.submod` (a GraphModule)
        placeholders = filter(
            lambda node: node.op == "placeholder", self.submod.graph.nodes
        )
        # `placeholders` are nodes internal to submod.
        # `self.node.args` are dependency nodes in the outer graph.
        # The two are 1:1.
        for placeholder, arg_node in zip(placeholders, self.node.args):
            # Create a receive buffer for this placeholder
            recv_info = create_recv_tensor(placeholder, arg_node)
            args_recv_info.append(recv_info)

        logger.debug(
            f"{self.log_prefix} "  # noqa: G004
            f"Activation recv / args info: {args_recv_info}"
        )
        # `args` is a Tuple, hence we will return a Tuple[InputInfo]
        return tuple(args_recv_info)


注释：

# 创建一个 `_RecvInfo` 元组，用于描述输入到该阶段的信息。

# 为给定占位符创建一个接收缓冲区。
# 如果 `arg_node.op` 是 "placeholder"，则这是一个根级占位符，是整个模型的输入参数。
# 我们可能处于阶段 0，因此不需要创建接收缓冲区。

# 确定此输入的源阶段。
# 当 `arg_node.target` 是 operator.getitem 时，需要继续深入。
# 断言 `arg_node.op` 应为 "call_module"，如果不是则抛出异常。

# 为该占位符创建一个接收缓冲区。
# 记录调试信息，显示正在创建输入 '{placeholder.name}' 的接收缓冲区：{example_value.shape}, {example_value.dtype}。

# `args` 是一个元组，因此我们将返回一个 Tuple[InputInfo]。
    ) -> Optional[int]:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
        if user.op == "call_module":
            # 用户节点是一个阶段 (`call_module`)
            return self.get_stage_index_of_submod(user.name)
        else:
            # 如果 user.op == "output":
            # 不需要发送到 rank 0
            # 如果 user.target 是 stage_backward:
            # 假设子模块输出在本地存储或在激活检查点情况下应重新计算，则不需要发送
            return None

    def _create_act_send_info(self):
        """
        Create a dict of send info for activations.
        The dict is of the form:
        {
            output_index: [dst_rank_0, dst_rank_1, ...],
            ...
        }
        where the list of `dst_rank`s covers the case where an output value may
        be consumed by multiple stages.
        """
        # 输出索引: 接收者 rank 的列表
        act_send_info: Dict[int, List] = {}
        out_idx = 0

        for user in self.node.users:
            if user.target is operator.getitem:
                # 递归查找真正的目的地
                gi_dsts = act_send_info.setdefault(out_idx, [])
                for gi_user in user.users:
                    dst_rank = self.find_dst_rank(gi_user)
                    if dst_rank is not None:
                        gi_dsts.append(dst_rank)
                # 下一个 `getitem` 将指向下一个输出索引
                out_idx += 1
            else:
                # 对于单个输出值的情况，`out_idx` 不会增加
                dsts = act_send_info.setdefault(out_idx, [])
                dst_rank = self.find_dst_rank(user)
                if dst_rank is not None:
                    dsts.append(dst_rank)

        output_node = self._get_output_node()
        output_vals: Tuple[torch.Tensor] = tuple(
            v.meta["val"] for v in flatten_args(output_node.args)
        )
        self._configure_outputs_meta(output_vals)

        logger.debug(f"{self.log_prefix} " f"Send info: {act_send_info}")  # noqa: G004
        return act_send_info

    def _get_output_node(self):
        output_nodes = [node for node in self.submod.graph.nodes if node.op == "output"]
        assert len(output_nodes) == 1
        output_node = output_nodes[0]
        return output_node

    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
        ) -> Tuple[_RecvInfo, ...]:
        """
        Create a tuple of `_RecvInfo` for gradients.
        """
        # 创建一个空的字典，用于存储梯度接收信息，键为输出索引，值为 `_RecvInfo` 对象
        grad_recv_info: Dict[int, _RecvInfo] = {}
        
        # 获取模型的输出节点
        output_node = self._get_output_node()

        # 将输出节点的参数展平，以便处理可能的多个输出值
        output_vals = flatten_args(output_node.args)

        # 遍历激活发送信息的条目
        for out_idx, dst_list in act_send_info.items():
            # 如果目标列表为空，则表示没有接收到对应激活信号，因此不需要梯度回传
            if not dst_list:
                continue

            # 获取当前输出索引对应的输出值
            output = output_vals[out_idx]
            # 从输出的元数据中获取示例值
            example_value = output.meta["val"]
            
            # 记录调试信息，显示正在为输出创建梯度接收缓冲区的过程
            logger.debug(
                f"{self.log_prefix} Creating grad recv buffer for output {output.name} "
                f": {example_value.shape}, {example_value.dtype}"
            )

            # TODO: 目前不支持跳跃连接的反向传播，需要进行梯度累积处理
            assert len(dst_list) == 1, "Backward of skip connections not supported yet"
            # 获取梯度来源信息
            grad_src = dst_list[0]
            
            # 创建 `_RecvInfo` 对象，并将其添加到 `grad_recv_info` 字典中
            grad_recv_info[out_idx] = _RecvInfo(
                f"{grad_src}",  # noqa: G004
                grad_src,
                _make_tensor_from_meta(example_value, self.device),
            )

        # 将 `grad_recv_info` 字典转换为元组，以便在 `get_ops` 和检索张量时方便使用
        grad_recv_info_tuple = tuple(grad_recv_info.values())
        
        # 记录调试信息，显示已创建的梯度接收信息元组
        logger.debug(
            f"{self.log_prefix} Grad recv info: {grad_recv_info_tuple}"  # noqa: G004
        )
        
        # 返回梯度接收信息的元组
        return grad_recv_info_tuple
# A helper function to create a pipeline stage based on traced pipeline information
def build_stage(
    stage_module: torch.nn.Module,
    stage_index: int,
    pipe_info: PipeInfo,
    device: torch.device,
    group: Optional[dist.ProcessGroup] = None,
) -> _PipelineStage:
    """
    Create a pipeline stage given a stage_module to be wrapped by this stage
    and pipeline information.

    Args:
        stage_module (torch.nn.Module): the module to be wrapped by this stage
        stage_index (int): the index of this stage in the pipeline
        pipe_info (PipeInfo): information about the pipeline, can be retrieved by `pipe.info()`
        device (torch.device): the device to be used by this stage
        group (Optional[dist.ProcessGroup]): the process group to be used by this stage

    Returns:
        _PipelineStage: a pipeline stage that can run with `PipelineSchedules`.
    """
    return _PipelineStage(
        stage_module,
        stage_index,
        pipe_info,
        device,
        group,
    )


# Manual PipelineStage functions and definition

# Length of the metadata tensor
METADATA_TENSOR_LEN = 100
# Placeholder value used in the metadata tensor
PLACEHOLDER_VAL = -1


def _create_empty_tensors(
    tensor: Union[torch.Tensor, Iterable[torch.Tensor]], device: torch.device
) -> List[torch.Tensor]:
    """
    Creates a list of empty tensors with the same properties (like shape and dtype) as the input tensor(s),
    and places them on the specified device.

    Args:
        tensor (Union[torch.Tensor, List[torch.Tensor]]): The input tensor(s).
        device (torch.device): The device where the new tensors will be placed.

    Returns:
        List[torch.Tensor]: A list of empty tensors with the same properties as the input tensor(s).
    """
    if isinstance(tensor, torch.Tensor):
        # Create a list with one empty tensor having the same properties as `tensor`
        return [torch.empty_like(tensor, device=device)]
    elif isinstance(tensor, (list, tuple)):
        # Create a list of empty tensors, each having properties similar to tensors in the input list
        return [torch.empty_like(t, device=device) for t in tensor]
    # Raise an error if the input type is unsupported
    raise TypeError(f"Unsupported type {type(tensor)} cannot create empty tensors")


def _create_metadata_tensor(
    tensors: Optional[List[torch.Tensor]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a metadata tensor that can be sent over the wire.
    This tensor contains the number of dimensions and the shape of each tensor being sent.

    The data is of format [num_dims, dim1, dim2, ...].
    If the tensor is None, a tensor of only placeholder values will be returned.

    Args:
        tensors (Optional[List[torch.Tensor]]): A list of tensors whose shapes will be concatenated into the metadata tensor.
        device (Optional[torch.device]): The device where the metadata tensor will be created.

    Returns:
        torch.Tensor: A tensor containing metadata about the dimensions of the input tensors.
    """
    # Create a metadata tensor filled with placeholder values
    metadata_tensor = torch.full(
        (METADATA_TENSOR_LEN,),  # Shape of the tensor
        PLACEHOLDER_VAL,  # Fill value
        dtype=torch.int32,  # Data type of the tensor
        device=device,  # Device where the tensor will reside
    )
    # 如果输入的张量列表非空，则执行以下操作
    if tensors:
        # 创建一个列表，其中包含每个张量的维数和形状信息
        data = [
            # data 是以 [num_dims, dim1, dim2, ...] 的格式存储
            torch.tensor(
                [len(tensor.shape)] + list(tensor.shape),
                dtype=torch.int32,
                device=device,
            )
            for tensor in tensors
        ]
        # 将所有数据张量拼接成一个单独的张量
        data_tensor = torch.cat(data)
        # 获取拼接后张量的长度
        dt_shape = data_tensor.shape[0]
        # 如果拼接后张量长度超过了预定义的最大长度 METADATA_TENSOR_LEN，则抛出数值错误
        if dt_shape > METADATA_TENSOR_LEN:
            raise ValueError(
                f"Metadata tensor size ({dt_shape}) exceeds maximum allowed length ({METADATA_TENSOR_LEN})."
            )
        # 将拼接后的数据张量写入 metadata_tensor 的前 dt_shape 个元素中
        metadata_tensor[:dt_shape] = data_tensor
    # 返回 metadata_tensor
    return metadata_tensor
# 从张量中提取元数据，包括张量的维数和每个张量的形状
def _extract_metadata_from_tensor(tensor: torch.Tensor) -> List[torch.Size]:
    """
    Extract the number of dimensions and the shape of each tensor from a metadata tensor.
    """
    # 初始化一个空列表，用于存储张量的形状信息
    metadata: List[torch.Size] = []
    # 初始化索引 i 为 0
    i = 0
    # 循环直到索引 i 超出张量长度或者遇到占位符值
    while i < len(tensor) and tensor[i] != PLACEHOLDER_VAL:
        # 第 i 个元素是张量的维数，转换为整数
        num_dims = int(tensor[i].item())
        # 接下来 num_dims 个元素是张量的形状，构造成 torch.Size 对象
        shape = torch.Size(tensor[i + 1 : i + 1 + num_dims].tolist())
        # 将形状信息加入到 metadata 列表中
        metadata.append(shape)
        # 更新索引，跳到下一个张量的起始位置
        i += num_dims + 1
    # 返回所有张量的形状信息列表
    return metadata


# 获取各个管道阶段模块的输入输出形状
def _get_stage_shapes(
    stage_modules: List[nn.Module],
    stage_ids: List[int],
    num_stages: int,
    rank: int,
    world_size: int,
    device: torch.device,
    microbatch: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
):
    """
    Performs a dry run through all the pipeline stages (a rank can have multiple pipeline stages in the case of
    virtual pipelining) and returns the shape of the inputs and outputs of the module.
    Only the first stage must pass in a microbatch.

    Each rank must call _get_stage_shapes or the program will hang.

    Args:
        stage_modules: The chunks assigned to this rank. Rhe length should be 1 for any
                non-interleaved schedules and >1 for any interleaved schedules.
        stage_ids: The id of the stages assigned to this rank.
        num_stages: Total number of stages.
        rank: Rank of the current process.
        world_size: Number of processes participating in the pipeline.
        device: Device where the tensors are allocated.

    Returns a dictionary containing the following keys:
        "inputs": Shape of the inputs to the module
        "outputs": Shape of the outputs of the module
    """

    # 初始化一个空字典，用于存储各个阶段模块的输入和输出形状信息
    stage_id_to_shapes: Dict[int, Dict[str, list[torch.Size]]] = {}
    # 对于每个阶段的标识和模型，依次执行以下操作
    for stage_id, model in zip(stage_ids, stage_modules):
        # 创建用于存储输入形状信息的元数据张量
        input_shape_metadata_tensor = _create_metadata_tensor(device=device)
        # TODO: 假设 prev_stage == rank - 1，next_stage == rank + 1
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size
        shapes = {}

        # 第一个阶段不接收任何输入，使用微批处理
        if stage_id == 0:
            if microbatch is None:
                # 如果缺少微批处理数据，则抛出运行时错误
                raise RuntimeError("Microbatch is required for first stage")
            example_fwd_inputs = microbatch
            if isinstance(example_fwd_inputs, torch.Tensor):
                example_fwd_inputs = [example_fwd_inputs]
        else:
            # 其他阶段需要接收形状信息
            # TODO: send/recv 应使用指定的通信组，而非使用默认通信组
            dist.recv(input_shape_metadata_tensor, prev_rank)
            metadata = _extract_metadata_from_tensor(input_shape_metadata_tensor)
            example_fwd_inputs = [
                torch.empty(shape_list, device=device) for shape_list in metadata
            ]
        shapes["inputs"] = [fwd_input.shape for fwd_input in example_fwd_inputs]

        # 执行前向传播
        # TODO: 如果前向传播失败，抛出更详细的错误，指出是哪个阶段失败了
        fwd_outputs = model(*example_fwd_inputs)
        fwd_outputs = _create_empty_tensors(fwd_outputs, device)
        shapes["outputs"] = [fwd_output.shape for fwd_output in fwd_outputs]

        # 发送形状维度信息
        if stage_id != num_stages - 1:
            output_shape_metadata_tensor = _create_metadata_tensor(
                fwd_outputs, device=device
            )
            dist.send(output_shape_metadata_tensor, next_rank)
        stage_id_to_shapes[stage_id] = shapes
    
    # 记录阶段与其对应的输入输出形状信息
    logger.info(stage_id_to_shapes)
    # 返回阶段与其对应的输入输出形状信息的字典
    return stage_id_to_shapes
# PipelineStage 类表示管道并行设置中的一个阶段。
# 此类是通过提供示例输入（和可选的输出）手动创建的，而不是从 pipeline() 函数输出的 PipelineStage 类。
# 它扩展了 `_PipelineStageBase` 类，并可以类似地在 `PipelineSchedule` 中使用。

Args:
    submodule (nn.Module): 由此阶段包装的 PyTorch 模块。
    stage_index (int): 此阶段的ID。
    num_stages (int): 总阶段数。
    device (torch.device): 此阶段所在的设备。
    input_args (Union[torch.Tensor, Tuple[torch.tensor]]): 子模块的输入参数。
    output_args (Union[torch.Tensor, Tuple[torch.tensor]], optional): 子模块的输出参数，可选。
    group (dist.ProcessGroup, optional): 分布式训练的进程组。如果为 None，则使用默认组。
    dw_builder: TODO 清理注释
        ):
            # 调用父类的初始化方法，传入子模块、阶段索引、阶段总数、设备、组和构建器
            super().__init__(submodule, stage_index, num_stages, device, group, dw_builder)
            # 将子模块移到指定设备上
            self.submod.to(self.device)
            # 当在 CUDA 上实例化模型分区时，如果可用，调用 reset_parameters()
            self.inputs: List[torch.Tensor] = []
            self.outputs: List[torch.Tensor] = []

            # 使用输入参数列表创建空张量列表作为输入
            self.inputs = _create_empty_tensors(input_args, device)

            if output_args is None:
                logger.info("output_args not provided, performing forward using input_args")
                # 使用输入参数调用子模块进行前向传播，并将结果保存在 outputs 中
                self.outputs = self.submod(*self.inputs)
                # 为输出创建空张量列表，以便在后续的 p2p 操作中使用
                self.outputs = _create_empty_tensors(self.outputs, device)
            else:
                # 使用输出参数列表创建空张量列表作为输出
                self.outputs = _create_empty_tensors(output_args, device)

            # 配置输出元数据
            self._configure_outputs_meta(tuple(self.outputs))

            # 用于反向发送/接收的缓冲区，稍后分配
            self.outputs_grad: List[torch.Tensor] = []

            # 定义一个函数，用于获取全局阶段等级
            def stage_global_rank(peer_rank):
                return (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )

            # 计算上一个阶段和下一个阶段的全局等级
            self.prev_stage = stage_global_rank((self.group_rank - 1) % self.group_size)
            self.next_stage = stage_global_rank((self.group_rank + 1) % self.group_size)

            # 记录调试信息，包括阶段索引、是否为第一个阶段、是否为最后一个阶段、阶段总数、输入形状和输出形状
            logger.debug(
                f"finished pipeline stage init, {self.stage_index=}, {self.is_first=}, "
                f"{self.is_last=}, {self.num_stages=}, "
                f"inputs: {[inp.shape for inp in self.inputs]}, "
                f"output: {[output.shape for output in self.outputs]}"
            )
    def _prepare_forward_infra(self, num_microbatches: int) -> None:
        # 在前向传播期间接收信息
        # TODO: 是否需要延迟创建 args_recv_info？（PipelineStage 需要相同的处理）
        for chunk_id in range(num_microbatches):
            # 设置当前块是否需要梯度计算为 False
            self.set_requires_grad[chunk_id] = False
            if not self.is_first:
                # 假设我们总是从上一阶段接收信息
                recv_infos = tuple(
                    [
                        _RecvInfo(
                            f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                            self.stage_index - 1,
                            _make_tensor_from_meta(inp, self.device),
                        )
                        for inp in self.inputs
                    ]
                )

                self.args_recv_info[chunk_id] = recv_infos
            else:
                # 如果是第一阶段，则将输入数据占位符化处理
                self.args_recv_info[chunk_id] = tuple(
                    [_RootArgPlaceholder(i) for i in self.inputs]
                )

        # 在前向传播期间为每个激活项发送信息
        # 只需要发送到目标阶段的排名
        self.act_send_info: Dict[int, List] = {}
        for idx in range(len(self.outputs)):
            # 假设我们总是发送到下一阶段
            if not self.is_last:
                self.act_send_info[idx] = [self.stage_index + 1]
            else:
                self.act_send_info[idx] = []

    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ) -> Tuple[_RecvInfo, ...]:
        grad_recv_info: Tuple[_RecvInfo, ...] = ()
        if not self.is_last:
            # 不支持从多个来源接收梯度，因此我们只取第一个目标
            grad_recv_info = tuple(
                [
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{dst_list[0]}",
                        dst_list[0],
                        _make_tensor_from_meta(self.outputs[idx], self.device),
                    )
                    for idx, dst_list in act_send_info.items()
                ]
            )
        return grad_recv_info
    def _init_p2p_neighbors(self):
        """
        Set up p2p communicators between previous and next stages
        by sending a dummy tensor.

        If this is used, must be called for all pipeline stages.
        """
        # 初始化一个空操作列表
        ops = []
        # 创建一个在GPU上的零张量作为接收张量
        recv_tensor = torch.zeros(1, device="cuda")
        # 创建一个在GPU上的全一张量作为发送张量
        send_tensor = torch.ones(1, device="cuda")
        
        # 向前传输
        # 如果当前不是第一个阶段
        if not self.is_first:
            # 向前阶段发送接收张量的 P2P 操作，并将操作加入操作列表
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.prev_stage, self.group))
        
        # 如果当前不是最后一个阶段
        if not self.is_last:
            # 向后阶段发送发送张量的 P2P 操作，并将操作加入操作列表
            ops.append(dist.P2POp(dist.isend, send_tensor, self.next_stage, self.group))

        # 向后传输
        # 如果当前不是第一个阶段
        if not self.is_first:
            # 向前阶段发送发送张量的 P2P 操作，并将操作加入操作列表
            ops.append(dist.P2POp(dist.isend, send_tensor, self.prev_stage, self.group))
        
        # 如果当前不是最后一个阶段
        if not self.is_last:
            # 向后阶段发送接收张量的 P2P 操作，并将操作加入操作列表
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.next_stage, self.group))

        # 返回 True，表示初始化成功
        return True
# 确保管道阶段的缓冲区形状符合预期，通过在所有阶段之间执行全局收集来实现
def _validate_stage_shapes(pipeline_stages: List[PipelineStage]):
    """
    检查缓冲区形状是否与预期匹配，通过在所有阶段之间执行全局收集来验证。
    """
    # 如果没有提供管道阶段，则抛出值错误异常
    if len(pipeline_stages) == 0:
        raise ValueError("No pipeline stages provided.")

    # 确定虚拟管道的大小，即管道阶段的数量
    virtual_pipeline_size = len(pipeline_stages)
    
    # 初始化存储所有输入和输出的列表
    all_inputs = []
    all_outputs = []
    
    # 获取管道的全局大小，假定第一个阶段的组大小即为全局大小
    world_size = pipeline_stages[0].group_size
    
    # 获取管道的阶段数量，假定第一个阶段的阶段数即为整个管道的阶段数
    num_stages = pipeline_stages[0].num_stages
    
    # 执行所有阶段之间的全局收集操作
    # 在这里执行具体的全局收集操作，将所有阶段的数据进行集中处理
    # 使用 enumerate() 遍历 pipeline_stages 列表，获取虚拟的 ID 和每个阶段的对象 stage
    for virtual_id, stage in enumerate(pipeline_stages):
        # 获取当前阶段的 group_size 属性，表示该阶段中的进程组大小
        world_size = stage.group_size
        # 获取当前阶段的 stage_index 属性，表示该阶段的索引号
        stage_id: int = stage.stage_index
        # 获取当前阶段的 group_rank 属性，表示该阶段中当前进程的排名
        rank = stage.group_rank

        # 检查所有阶段中 world_size 和 num_stages 是否一致
        if stage.group_size != world_size:
            # 如果不一致，抛出 ValueError 异常
            raise ValueError(
                f"Stage id {stage_id} has world size ({stage.group_size}) \
                which does not match world size ({world_size}) of other stages."
            )

        if stage.num_stages != num_stages:
            # 检查所有阶段中 num_stages 是否一致，如果不一致，抛出 ValueError 异常
            raise ValueError(
                f"Stage id {stage_id} has num stages ({stage.num_stages}) \
                which does not match num stages ({num_stages}) of other stages."
            )

        # 使用 dist.get_rank() 获取当前进程在 stage.group 中的排名
        pg_rank = dist.get_rank(stage.group)
        # 检查当前进程在 stage.group 中的排名是否与 stage.group_rank 属性一致
        if rank != pg_rank:
            # 如果不一致，抛出 ValueError 异常
            raise ValueError(
                f"Rank {rank} is not equal to process group rank {pg_rank}"
            )

        # 检查当前阶段的 num_stages 是否是 world_size 的倍数
        if (num_stages := stage.num_stages) % world_size != 0:
            # 如果不是，抛出 ValueError 异常
            raise ValueError(
                f"Number of stages ({num_stages}) must be a multiple of the world_size ({world_size})"
            )

        # 为每个阶段中的每个进程创建一个元数据张量列表，使用 _create_metadata_tensor() 函数
        tensor_list = [
            _create_metadata_tensor(device=stage.device)
            for _ in range(stage.group_size)
        ]

        # 获取当前阶段的输入期望值，并创建对应的元数据张量 stage_input
        expected_inputs = stage.inputs
        stage_input = _create_metadata_tensor(expected_inputs, device=stage.device)
        # 使用 dist.all_gather() 收集所有进程的 stage_input 数据到 tensor_list 中
        dist.all_gather(tensor_list, stage_input)
        # 提取 tensor_list 中每个张量的元数据，存储到 stage_input_shapes 列表中
        stage_input_shapes = [
            _extract_metadata_from_tensor(tensor) for tensor in tensor_list
        ]

        # 为每个阶段中的每个进程创建一个元数据张量列表，使用 _create_metadata_tensor() 函数
        tensor_list = [
            _create_metadata_tensor(device=stage.device)
            for _ in range(stage.group_size)
        ]

        # 获取当前阶段的输出期望值，并创建对应的元数据张量 stage_output
        expected_outputs = stage.outputs
        stage_output = _create_metadata_tensor(expected_outputs, device=stage.device)
        # 使用 dist.all_gather() 收集所有进程的 stage_output 数据到 tensor_list 中
        dist.all_gather(tensor_list, stage_output)
        # 提取 tensor_list 中每个张量的元数据，存储到 stage_output_shapes 列表中
        stage_output_shapes = [
            _extract_metadata_from_tensor(tensor) for tensor in tensor_list
        ]

        # 记录调试信息，包括当前进程的排名、阶段 ID、阶段的 num_stages、进程的排名、进程组大小、输入和输出形状
        logger.debug(
            f"Rank: {pg_rank}"  # noqa: G004
            f"Stage id: {stage_id}"
            f"Stage num stages: {stage.num_stages}"
            f"Stage rank: {rank}"
            f"Stage world size: {world_size}"
            f"Stage {virtual_id * world_size}-{(virtual_id + 1) * world_size - 1} input shapes: {stage_input_shapes}"  # noqa: G003
            f"Stage {virtual_id * world_size}-{(virtual_id + 1) * world_size - 1} output shapes: {stage_output_shapes}"  # noqa: G003
        )

        # 将当前阶段的输入形状添加到 all_inputs 列表中
        all_inputs.extend(stage_input_shapes)
        # 将当前阶段的输出形状添加到 all_outputs 列表中
        all_outputs.extend(stage_output_shapes)

        # 如果当前进程是排名为 0 的进程，则记录信息，包括所有阶段的输入和输出形状
        if pg_rank == 0:
            logger.info(
                f"all stage inputs: {all_inputs}"  # noqa: G004
                f"all stage outputs: {all_outputs}"
            )
    # 循环检查每个阶段的输出是否与下一个阶段的输入匹配
    for i in range(virtual_pipeline_size * world_size - 1):
        # 使用 walrus 运算符同时比较当前阶段的输出和下一个阶段的输入
        if (out := all_outputs[i]) != (inp := all_inputs[i + 1]):
            # 如果输出形状与下一个阶段的输入形状不匹配，则抛出数值错误异常
            raise ValueError(
                f"Stage_id {i} output shape {out} at does not match stage_id {i + 1} input shape {inp}."
            )
```