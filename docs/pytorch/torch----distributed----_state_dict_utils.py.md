# `.\pytorch\torch\distributed\_state_dict_utils.py`

```
    """Iterate through the state dict, applying the given functions to each tensor type.

    Args:
        iter_object (Any): Object to iterate over (typically a state dict).
        sharded_tensor_func (Callable): Function to apply to ShardedTensor types.
        dtensor_func (Callable): Function to apply to DTensor types.
        tensor_func (Callable): Function to apply to regular Tensor types.
        pg (Optional[dist.ProcessGroup]): Process group for distributed operations.
        device (Optional[torch.device]): Device for computation.
        cpu_offload (bool): Flag indicating if computation should be offloaded to CPU.
        companion_obj (Any): Companion object for additional context.
        ranks_only (Tuple[int, ...]): Tuple of ranks to apply functions to.
        type_check (bool): Flag to indicate if type checking should be enforced.
        non_blocking (bool): Flag for non-blocking operations.

    Returns:
        Dict[str, Any]: Dictionary containing results after applying functions to each tensor.

    Raises:
        CompanionMismatch: Raised if there is a mismatch in companion objects.

    """
    Args:
        iter_object (Any): 要处理的对象，可以是任意类型。
        sharded_tensor_func (Callable): 应用于 ShardedTensor 的函数。
        dtensor_func (Callable): 应用于 DTensor 的函数。
        tensor_func (Callable): 应用于 Tensor 的函数。
        pg (Optional[dist.ProcessGroup]): 传递给张量函数的进程组。
        device (Optional[torch.device]): 传递给张量函数的设备。
        cpu_offload (bool): 是否将张量卸载到 CPU 内存。如果提供了 companion_obj，则忽略此选项。
        companion_obj (Any): 状态字典的伴随对象。如果提供了此对象，则尝试将张量复制到该对象。
        ranks_only (Tuple[int, ...]): 如果为空元组，则所有 ranks 将具有相同的状态字典。否则，只有在 ranks_only 中的 ranks 才会有相同的状态字典，其他 ranks 将得到空状态字典。
        type_check (bool): 检查实例数据类型是否为 DCP 可保存的支持类型。当前支持的数据类型包括 torch.Tensor, DTensor, int, float, str, list, dict, None。
        non_blocking (bool): 在复制到伴随对象时是否使用非阻塞复制。

    """
    # TODO: should we use pytree?
    # 判断 iter_object 是否为 ShardedTensor 类型
    cpu_device = torch.device("cpu")
    if isinstance(iter_object, ShardedTensor):
        # 对 ShardedTensor 类型的对象应用 sharded_tensor_func 函数
        ret = sharded_tensor_func(iter_object, pg, device, companion_obj)
    # 判断 iter_object 是否为 DTensor 类型
    elif isinstance(iter_object, DTensor):
        # 对 DTensor 类型的对象应用 dtensor_func 函数
        ret = dtensor_func(iter_object, pg, device, companion_obj)
    # 判断 iter_object 是否为 torch.Tensor 类型
    elif isinstance(iter_object, torch.Tensor):
        # 对 torch.Tensor 类型的对象应用 tensor_func 函数
        ret = tensor_func(iter_object, pg, device, companion_obj)
    # 判断 iter_object 是否为 int, float, str, bytes, io.BytesIO 或 None 类型
    elif (
        isinstance(iter_object, (int, float, str, bytes, io.BytesIO))
        or iter_object is None
    ):
        # 直接返回 iter_object，因为它是基本数据类型或者 None
        ret = iter_object
    # 判断 iter_object 是否为 dict 类型
    elif isinstance(iter_object, dict):
        # 如果 companion_obj 不为 None，并且 companion_obj 不是 dict 类型，或者其键集合与 iter_object 的键集合不相等，抛出 CompanionMismatch 异常
        if companion_obj is not None and (
            not isinstance(companion_obj, dict)
            or set(companion_obj.keys()) != set(iter_object.keys())
        ):
            msg = (
                ""
                if isinstance(companion_obj, dict)
                else f"{set(companion_obj.keys())=} {set(iter_object.keys())=}"
            )
            raise CompanionMismatch(msg)

        # 递归调用 _iterate_state_dict 处理 iter_object 中的每个键值对
        ret = {
            key: _iterate_state_dict(
                value,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[key] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for key, value in iter_object.items()
        }
    # 如果 iter_object 是 list 或 tuple 类型
    elif isinstance(iter_object, (list, tuple)):
        # 如果 companion_obj 不为 None，并且 companion_obj 不是 list 或 tuple，或者长度不等于 iter_object
        if companion_obj is not None and (
            not isinstance(companion_obj, (list, tuple))
            or len(companion_obj) != len(iter_object)
        ):
            # 抛出 CompanionMismatch 异常
            raise CompanionMismatch

        # 递归调用 _iterate_state_dict 函数处理 iter_object 中的每个元素
        ret = [
            _iterate_state_dict(
                v,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[idx] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for idx, v in enumerate(iter_object)
        ]
        # 如果 iter_object 是 tuple 类型，则将 ret 转换为 tuple
        if isinstance(iter_object, tuple):
            ret = tuple(ret)
    
    # 如果不进行类型检查
    elif not type_check:
        # 对 iter_object 执行深拷贝操作
        ret = copy.deepcopy(iter_object)
    
    else:
        # 如果不符合以上任何条件，则抛出 ValueError 异常，指明意外的值类型
        raise ValueError(f"Unexpected value type {type(iter_object)}")

    # 如果不仅仅输出 ranks_only 中指定的 rank，或者当前进程在 ranks_only 中
    if not ranks_only or dist.get_rank(pg) in ranks_only:
        # 如果 ret 是 torch.Tensor 类型
        if isinstance(ret, torch.Tensor):
            # 如果启用了 cpu_offload 且 companion_obj 为 None，则将 ret 转移到 cpu_device
            if cpu_offload and companion_obj is None:
                ret = ret.to(cpu_device)

            # 如果 companion_obj 不为 None
            if companion_obj is not None:
                # TODO: 支持 DTensor
                # 将 ret 的数据拷贝到 companion_obj，支持非阻塞操作
                companion_obj.copy_(ret, non_blocking=non_blocking)
                ret = companion_obj
    else:
        # 如果 ret 是 dict 类型，则设置为空字典，否则设为 None
        ret = {} if isinstance(ret, dict) else None

    # 返回处理后的结果 ret
    return ret
def _gather_state_dict(
    state_dict: Dict[str, Any],
    *,
    pg: Optional[dist.ProcessGroup] = None,  # 可选参数：用于收集 ShardedTensor 的进程组
    device: Optional[torch.device] = None,   # 可选参数：用于 ShardedTensor 的 allgather 的设备
    cpu_offload: bool = False,               # 是否将张量转移到 CPU 内存的标志，默认为 False
    ranks_only: Tuple[int, ...] = tuple(),   # 仅包含相同状态字典的特定排名的元组
    type_check: bool = True,                 # 是否检查数据类型是否支持保存到 DCP 中

) -> Dict[str, Any]:
    """
    Given a state_dict, this API gathers all the ShardedTensors or DTensors in
    the state_dict.

    Args:
        state_dict (Dict[str, Any]): 目标分片状态字典。
        pg (Optional[dist.ProcessGroup]): 用于收集 ShardedTensor 的进程组。注意，收集 DTensor 时将使用 DeviceMesh，因此在收集 DTensor 时将忽略此参数。
        device: (Optional[torch.device]): 用于执行 ShardedTensor 的 allgather 的设备。注意，收集 DTensor 时将使用 DeviceMesh，因此在收集 DTensor 时将忽略此参数。
        cpu_offload (bool): 是否将张量转移到 CPU 内存。默认值为 False。
        ranks_only: (Tuple[int, ...]): 如果此元组为空，则所有排名将具有相同的状态字典。否则，只有在 ranks_only 中的排名将具有相同的状态字典，其他排名将获得空状态字典。
        type_check: (bool): 检查实例数据类型是否是 DCP 可保存的支持类型。当前支持的数据类型包括 torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """

    def sharded_tensor_func(value, pg, device, companion_obj):
        # ShardedTensor 似乎没有记录原始设备类型。
        # 因此，如果张量被移动到 CPU，我们无法知道原始类型。
        # 结果，我们必须依赖用户告诉我们正确的类型。
        cpu_device = torch.device("cpu")
        output_tensor = _all_gather_sharded_tensor(value, pg, device)
        local_shard_device = (
            value.local_shards()[0].tensor.device  # 获取本地分片的设备类型
            if value.local_shards()
            else cpu_device  # 如果没有本地分片，则默认为 CPU 设备
        )
        if output_tensor.device != local_shard_device:
            value = output_tensor.to(local_shard_device)  # 如果输出张量的设备与本地分片设备不同，则将输出张量转移到本地分片设备
        else:
            value = output_tensor  # 否则直接使用输出张量
        return value
    # 定义一个名为 dtensor_func 的函数，用于处理分布式张量的操作
    def dtensor_func(value, pg, device, companion_obj):
        # 如果 value 的设备类型与 value.device_mesh.device_type 不一致，则将 value 转移到指定的设备类型上
        if value.device != value.device_mesh.device_type:
            value = value.to(value.device_mesh.device_type)
        
        # 根据不同的分布式设置，调整数据分布
        # FSDP all_gather: [Shard(0)] -> [Replicate()]
        # HSDP all_gather: [Replicate(), Shard(0)] -> [Replicate(), Replicate()]
        # 2D FSDP + TP all_gather:
        # - [Shard(0), Shard(n)] -> [Replicate(), Replicate()]
        # - [Shard(0), Replicate()] -> [Replicate(), Replicate()]
        placements = [Replicate() for _ in value.placements]
        value = value.redistribute(
            device_mesh=value.device_mesh,
            placements=placements,
        )
        
        # 调用 `wait()` 方法，强制张量与主流同步
        # 参考：https://github.com/pytorch/pytorch/pull/117799
        value = value.to_local()
        
        # 如果 value 是 AsyncCollectiveTensor 类型，则等待其异步操作完成
        if isinstance(value, AsyncCollectiveTensor):
            value = value.wait()
        
        # 返回处理后的 value
        return value

    # 调用 _iterate_state_dict 函数，传入参数进行状态字典的迭代处理
    return _iterate_state_dict(
        state_dict,
        sharded_tensor_func,
        dtensor_func,
        _identity_func,
        pg=pg,
        device=device,
        cpu_offload=cpu_offload,
        ranks_only=ranks_only,
        type_check=type_check,
    )
def _offload_state_dict_to_cpu(
    state_dict: Dict[str, Any],
    *,
    ranks_only: Tuple[int, ...] = tuple(),  # 定义可选参数 ranks_only，默认为空元组
    type_check: bool = True,  # 定义布尔类型参数 type_check，默认为 True
) -> Dict[str, Any]:  # 函数返回一个字典，键为字符串，值为任意类型

    """
    Given a state_dict, this API offload all the tensors to CPU memory.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check: (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """

    ret = _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        _identity_func,
        pg=None,  # 设定参数 pg 为 None
        device=None,  # 设定参数 device 为 None
        cpu_offload=True,  # 启用 CPU 内存卸载
        ranks_only=ranks_only,  # 将 ranks_only 参数传递给 _iterate_state_dict 函数
        type_check=type_check,  # 将 type_check 参数传递给 _iterate_state_dict 函数
    )
    return ret


def _copy_state_dict(
    state_dict: Dict[str, Any],
    copy_state_dict: Dict[str, Any],
    non_blocking: bool = False,  # 定义布尔类型参数 non_blocking，默认为 False
    type_check: bool = True,  # 定义布尔类型参数 type_check，默认为 True
) -> Dict[str, Any]:  # 函数返回一个字典，键为字符串，值为任意类型

    """
    Copies all tensors in a given state dict into a different state_dict with the
    same structure. Additionally, a copied state dict with the same value references
    is returned. Editing the keys on this state dict will not affect the
    passed in copy_state_dict (but the value references are the same).

    .. warning::
        It is expected by this function that state_dict and copy_state_dict share
        the same structure and data types.

    .. warning::
        The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        copy_state_dict (Dict[str, Any]):
            The state dict we are copying into. This state_dict must have exactly
             the same structure as the source `state_dict`.
        non_blocking: (bool): Whether copy ops should be performed asynchronously
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP. The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        State Dict copy
    """
    # 调用函数 `_iterate_state_dict`，并返回其结果
    return _iterate_state_dict(
        # 将 state_dict 作为参数传递给 `_iterate_state_dict` 函数
        state_dict,
        # 将 `_identity_func` 作为第二个参数传递给 `_iterate_state_dict` 函数
        _identity_func,
        # 将 `_identity_func` 作为第三个参数传递给 `_iterate_state_dict` 函数
        _identity_func,
        # 将 `_identity_func` 作为第四个参数传递给 `_iterate_state_dict` 函数
        _identity_func,
        # 将 pg 参数设置为 None
        pg=None,
        # 将 device 参数设置为 None
        device=None,
        # 将 cpu_offload 参数设置为 False
        cpu_offload=False,
        # 将 ranks_only 参数设置为一个空元组
        ranks_only=tuple(),
        # 将 companion_obj 参数设置为 copy_state_dict
        companion_obj=copy_state_dict,
        # 将 type_check 参数设置为 type_check
        type_check=type_check,
        # 将 non_blocking 参数设置为 non_blocking
        non_blocking=non_blocking,
    )
def _create_cpu_state_dict(
    state_dict: Dict[str, Any], pin_memory: bool = False, share_memory: bool = False
) -> Dict[str, Any]:
    """
    给定一个 state_dict，创建一个具有相同结构和元素的新 state_dict。
    然而，返回的 state_dict 中的所有张量都是在 CPU 上的新张量。
    这些张量可以根据提供的参数放置在 pin_memory 或 share_memory 上。

    .. warning::
        如果同时设置 `pin_memory` 和 `share_memory` 为 True，则由于需要直接注册内存作为固定内存，
        此方法的延迟显著增加。这个选项应仅用于需要共享的长期存活的张量。
        只要 `pin_memory` 或 `share_memory` 中至少有一个设置为 False，就不需要使用这个选项。

    """

    def tensor_func(
        obj: torch.Tensor,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        _: Any,
    ) -> torch.Tensor:
        if len(obj.size()) == 0:
            return torch.tensor(0, dtype=obj.dtype)

        if share_memory:
            # 创建一个共享内存的张量
            t = torch.empty(*tuple(obj.size()), dtype=obj.dtype).share_memory_()
            if pin_memory:
                # 尝试将共享内存固定到CUDA中
                succ = torch.cuda.cudart().cudaHostRegister(
                    t.data_ptr(),
                    t.numel() * t.element_size(),
                    1,  # 对应 'cudaHostRegisterPortable'
                )
                assert (
                    succ == 0
                ), f"Pinning shared memory failed with error-code: {succ}"
            return t
        elif pin_memory:
            # 创建一个固定内存的张量
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype).pin_memory()
        else:
            # 创建一个普通的张量
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype)

    ret = _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        tensor_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=tuple(),
        type_check=False,
    )
    return ret


def _check_state_dict_similarity(
    state_dict: Dict[str, Any],
    compared_state_dict: Dict[str, Any],
) -> bool:
    """
    给定两个 state_dict，检查它们的结构是否相同。
    如果一个 state_dict 中存在 [key, tensor] 对，则另一个 state_dict 中必须存在对应的 [key, other_tensor] 对，
    其中 tensor 和 other_tensor 具有相同的大小和数据类型。

    返回检查结果。
    """

    def tensor_func(
        obj: torch.Tensor,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        companion_obj: Any,
    ) -> torch.Tensor:
        # 检查张量的数据类型和大小是否匹配
        if companion_obj.dtype != obj.dtype or companion_obj.size() != obj.size():
            raise CompanionMismatch
        return obj
    # 尝试调用 _iterate_state_dict 函数，用于比较两个状态字典的内容
    try:
        _iterate_state_dict(
            state_dict,              # 第一个状态字典
            _identity_func,          # 用于标识的函数（无操作）
            _identity_func,          # 用于标识的函数（无操作）
            tensor_func,             # 处理张量的函数
            pg=None,                 # 进程组参数（未指定）
            device=None,             # 设备参数（未指定）
            cpu_offload=False,       # 是否启用 CPU 卸载（默认禁用）
            ranks_only=tuple(),      # 仅限排名的元组（空元组）
            companion_obj=compared_state_dict,  # 与比较状态字典对应的对象
            type_check=False,        # 是否进行类型检查（默认禁用）
        )
    # 如果 CompanionMismatch 异常抛出
    except CompanionMismatch:
        # 返回 False
        return False
    
    # 如果没有异常抛出，则返回 True
    return True
# 定义一个命名元组 `_TensorInfo`，包含两个字段 `size` 和 `dtype`，分别表示张量的大小和数据类型
class _TensorInfo(NamedTuple):
    size: torch.Size
    dtype: torch.dtype


# 广播张量数据到各个分布式进程
# `full_state_dict`: 全局状态字典，包含所有张量的完整状态
# `local_state_dict`: 本地状态字典，包含每个进程所需的部分张量状态
# `keys`: 要广播的张量的键列表
# `device`: 目标设备，张量将被发送到该设备
# `pg`: 进程组，用于分布式通信的进程组对象，可选
def _broadcast_tensors(
    full_state_dict: Dict[str, Any],
    local_state_dict: Dict[str, Any],
    keys: List[str],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
) -> None:
    tensors = []
    for key in keys:
        if dist.get_rank() == 0:
            full_state = full_state_dict[key]
            assert isinstance(full_state, torch.Tensor)
            full_tensor = full_state.detach().to(device)
        else:
            tensor_info = full_state_dict[key]
            full_tensor = torch.empty(
                size=tensor_info.size,
                device=device,
                dtype=tensor_info.dtype,
            )

        tensors.append(full_tensor)
        local_state = local_state_dict.get(key, None)
        if local_state is None:
            continue
        elif isinstance(local_state, DTensor):
            local_state_dict[key] = (local_state, full_tensor)
        else:
            local_state_dict[key] = full_tensor

    # 如果进程组未指定，则使用默认的分布式进程组
    if pg is None:
        pg = dist.distributed_c10d._get_default_group()

    # 如果张量数量大于1，则采用合并广播；否则，直接广播单个张量
    if len(tensors) > 1:
        dist._broadcast_coalesced(pg, tensors, 500, 0)
    else:
        dist.broadcast(tensors[0], src=0, group=pg)

    # 将本地状态字典中的张量分发到各个设备
    _distribute_tensors(local_state_dict, keys, device, pg)


# 分发张量到指定设备
# `local_state_dict`: 本地状态字典，包含每个进程所需的部分张量状态
# `keys`: 要分发的张量的键列表
# `device`: 目标设备，张量将被发送到该设备
# `pg`: 进程组，用于分布式通信的进程组对象，可选
def _distribute_tensors(
    local_state_dict: Dict[str, Any],
    keys: List[str],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
) -> None:
    # 如果进程组未指定，则使用默认的分布式进程组
    if pg is None:
        pg = dist.distributed_c10d._get_default_group()
    for key in keys:
        _local_state = local_state_dict.get(key, None)
        if _local_state is None or torch.is_tensor(_local_state):
            continue

        # 获取本地状态和对应的全局张量，并将全局张量分发到指定设备上
        local_state = _local_state[0]
        full_tensor = _local_state[1]
        local_state_dict[key] = distribute_tensor(
            full_tensor, local_state.device_mesh, local_state.placements
        )


# 广播完整状态字典 `full_state_dict` 到所有进程的 `local_state_dict`
# `full_state_dict`: 全局状态字典，包含所有张量的完整状态
# `local_state_dict`: 本地状态字典，将从全局状态中接收更新
# `device`: 目标设备，张量将被发送到该设备
# `pg`: 进程组，用于分布式通信的进程组对象，可选
# `strict`: 是否严格模式，如果为 True，本地状态字典中不在全局状态中的键将被移除
def _broadcast_state_dict(
    full_state_dict: Dict[str, Any],
    local_state_dict: Dict[str, Any],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
    strict: bool = False,
) -> None:
    # 从 rank0 的 `full_state_dict` 广播到所有 rank 的 `local_state_dict`
    # 如果 `strict` 为 True，那么 `local_state_dict` 中存在但 `full_state_dict` 中不存在的键将被移除
    ret = {}
    if dist.get_rank() == 0:
        for key, value in full_state_dict.items():
            if not torch.is_tensor(value):
                ret[key] = value
            elif value.dim() == 0:
                ret[key] = value.cpu()
            else:
                ret[key] = _TensorInfo(value.size(), value.dtype)

    broadcast_list = [ret]
    dist.broadcast_object_list(broadcast_list, src=0, group=pg)
    ret = broadcast_list[0]

    # 收集键集合
    keys = []
    local_state_dict_keys = set(local_state_dict.keys())
    global_keys = set()
    # 遍历字典 ret 中的每个键值对
    for key, value in ret.items():
        # 将当前键 key 添加到全局键集合 global_keys 中
        global_keys.add(key)
        
        # 检查值 value 是否不是 _TensorInfo 类型的实例
        if not isinstance(value, _TensorInfo):
            # 如果 key 已经存在于 local_state_dict 中，则更新其对应的值为 value
            if key in local_state_dict:
                local_state_dict[key] = value
            # 继续下一个循环
            continue
        
        # 如果当前进程的 rank 是 0
        if dist.get_rank() == 0:
            # 将 ret 中键为 key 的值更新为 full_state_dict 中对应的值
            ret[key] = full_state_dict[key]
        
        # 将 key 添加到 keys 列表中
        keys.append(key)
        
        # 广播每个张量，目的是暂时避免内存溢出（OOM）
        if len(keys) >= 1:
            _broadcast_tensors(ret, local_state_dict, keys, device, pg)
            # 清空 keys 列表
            keys.clear()

    # 如果 strict 为 True
    if strict:
        # 计算 local_state_dict_keys 和 global_keys 的差集，找出丢失的键
        if missing_keys := (local_state_dict_keys - global_keys):
            # 遍历丢失的键列表
            for key in missing_keys:
                # 从 local_state_dict 中移除对应的键
                local_state_dict.pop(key)

    # 如果 keys 列表非空
    if keys:
        # 广播 keys 列表中的张量
        _broadcast_tensors(ret, local_state_dict, keys, device, pg)
# 将完整的状态字典分发到本地状态字典中
def _distribute_state_dict(
    full_state_dict: Dict[str, Any],
    local_state_dict: Dict[str, Any],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
) -> None:
    # 遍历完整状态字典中的每个键值对
    for key, value in full_state_dict.items():
        # 如果键不在完整状态字典中，则跳过
        if key not in full_state_dict:
            continue
        # 如果值不是张量，则直接将其复制到本地状态字典中
        if not torch.is_tensor(value):
            local_state_dict[key] = value
        # 如果值是零维张量，则将其转移到CPU上
        elif value.dim() == 0:
            local_state_dict[key] = value.cpu()
        else:
            assert isinstance(value, torch.Tensor)
            # 将完整张量分离并移到指定设备上
            full_tensor = value.detach().to(device)
            local_state = local_state_dict.get(key, None)
            # 如果本地状态为空，则继续下一个键
            if local_state is None:
                continue
            # 如果本地状态是自定义张量类型 DTensor，则存储为元组
            elif isinstance(local_state, DTensor):
                local_state_dict[key] = (local_state, full_tensor)
            else:
                # 否则直接存储完整张量到本地状态字典中
                local_state_dict[key] = full_tensor

            # 调用 _distribute_tensors 函数，分发本地状态字典中的张量
            _distribute_tensors(local_state_dict, [key], device, pg)


# 这些API来自 torch.distributed.checkpoint 模块。
# TODO: 我们应该在这里整合代码，因为并非所有模块都可以依赖于 DCP。
PATH_ITEM = Union[str, int]
OBJ_PATH = Tuple[PATH_ITEM, ...]
FLATTEN_MAPPING = Dict[str, OBJ_PATH]
STATE_DICT_TYPE = Dict[str, Any]
CONTAINER_TYPE = MutableMapping[PATH_ITEM, Any]


def _traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, Any], None],
) -> None:
    """
    递归地遍历状态字典中的每个值，并对其应用访问者函数。

    映射、列表和元组会被展开，其他值类型被视为终端值，并对其应用访问者函数。
    """

    def _traverse_obj(path: OBJ_PATH, value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)
        else:
            visitor(path, value)

    # 遍历状态字典中的每个键值对
    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def _flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> Tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """
    将由嵌套字典和列表组成的状态字典展平为顶级字典。

    使用 unflatten_state_dict 函数可以还原此过程。
    返回:
        一个元组，包含展平后的状态字典和从原始状态字典到新状态字典的映射。
    注意: 新键由对象路径派生，路径中的元素用点连接起来。
        例如: {'a': {'b':...}} 将得到键 `a.b`。
    """
    flattened: STATE_DICT_TYPE = {}
    mappings: FLATTEN_MAPPING = {}
    # 定义一个函数 `flat_copy`，用于将给定路径和对应的值添加到 `flattened` 和 `mappings` 字典中
    def flat_copy(path: OBJ_PATH, value: Any) -> None:
        # 将路径转换为点分隔的字符串形式
        new_fqn = ".".join(map(str, path))
        # 如果新路径已经存在于 `flattened` 字典中，则抛出值错误异常
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        # 否则将新路径和对应的值添加到 `flattened` 和 `mappings` 字典中
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    # 使用递归方式遍历 `state_dict`，并将每个路径和值传递给 `flat_copy` 函数处理
    _traverse_state_dict(state_dict, flat_copy)
    # 返回处理后的 `flattened` 和 `mappings` 字典
    return flattened, mappings
# 设置函数_set_element，将value设置到root_dict的path指定的对象路径上
def _set_element(root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: Any) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    # 将root_dict强制类型转换为CONTAINER_TYPE类型，并赋值给cur_container
    cur_container = cast(CONTAINER_TYPE, root_dict)

    # 定义一个函数extend_list，用于扩展列表至指定长度
    def extend_list(lst: List[Any], idx: int) -> None:
        while len(lst) <= idx:
            lst.append(None)

    # 遍历path中除了第一个元素之外的每个元素
    for i in range(1, len(path)):
        # 获取前一个键(prev_key)和当前键(key)
        prev_key = path[i - 1]
        key = path[i]
        # 根据key的类型选择默认值：如果key是字符串，则默认值为字典{}；否则默认值为空列表[]
        def_val: Union[CONTAINER_TYPE, List[Any]] = {} if type(key) == str else []

        # 如果当前容器cur_container是映射类型的，则设置prev_key对应的值为def_val，并更新cur_container
        if isinstance(cur_container, Mapping):
            cur_container = cast(
                CONTAINER_TYPE, cur_container.setdefault(prev_key, def_val)
            )
        else:
            # 如果不是映射类型，则扩展cur_container列表至prev_key索引，如果之前该位置为None，则置为def_val
            extend_list(cur_container, prev_key)
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            # 更新cur_container为prev_key索引处的值
            cur_container = cur_container[prev_key]

    # 处理path中的最后一个键key
    key = path[-1]
    # 如果key的类型是整数，则将cur_container强制类型转换为列表，并扩展至key索引位置
    if type(key) == int:
        extend_list(cast(List[Any], cur_container), key)

    # 最终将value设置到cur_container[key]位置
    cur_container[key] = value


# 设置函数_unflatten_state_dict，根据映射mapping和扁平化的state_dict恢复原始的嵌套状态字典state_dict
def _unflatten_state_dict(
    state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING
) -> STATE_DICT_TYPE:
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
    # 初始化一个空的嵌套字典nested
    nested: STATE_DICT_TYPE = {}
    # 遍历扁平化的state_dict中的每个键值对
    for key, value in state_dict.items():
        # 调用_set_element函数，根据mapping[key]将value设置到nested中
        _set_element(nested, mapping[key], value)
    # 返回恢复后的嵌套状态字典nested
    return nested
```