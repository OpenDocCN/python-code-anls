# `.\pytorch\torch\distributed\_tensor\api.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import inspect  # 导入 inspect 模块，用于获取对象信息
import warnings  # 导入 warnings 模块，用于发出警告
from typing import Any, Callable, cast, Optional, Sequence, Tuple  # 导入类型提示相关的类和函数

import torch  # 导入 PyTorch 库
import torch.distributed._tensor._dispatch as op_dispatch  # 导入分布式张量分发模块
import torch.distributed._tensor.random as random  # 导入分布式张量随机数模块
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torch.distributed._tensor._collective_utils import mesh_broadcast  # 导入分布式张量的集合通信函数
from torch.distributed._tensor._redistribute import (
    Redistribute,
    redistribute_local_tensor,
)  # 导入张量重分布相关函数和类
from torch.distributed._tensor._utils import compute_global_tensor_info  # 导入计算全局张量信息的函数
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
    TensorMeta,
)  # 导入张量放置类型相关的类
from torch.distributed._tensor.random import (
    is_rng_supported_mesh,
    OffsetBasedRNGTracker,
)  # 导入随机数生成相关的函数和类
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh  # 导入设备网格相关的函数和类

__all__ = ["DTensor", "distribute_tensor", "distribute_module"]  # 指定模块的公开接口

aten = torch.ops.aten  # 获取 PyTorch 的 aten 操作对象


# NOTE [Autograd interaction between torch.Tensor]
#
# The autograd functions defined below are being used by the public
# facing APIs (i.e. from_local, to_local) to ensure our DTensor
# works together with torch.Tensor within autograd engine. This
# allows DistributedTensor to exist on part of the module hierarchy
# and still able to calculate gradients across the torch.Tensor and
# DistributedTensor boundary.
# As an example, we have the a module that consists of submodules
# A, B, and C, the execution flow would be like:
#  input(torch.Tensor) -> Module A -> Module B -> Module C -> output (torch.Tensor)
#
# Suppose I only want to make Module B be a sharded module with
# DistributedTensor params, we would need to make the following
# flow to work:
#
#  input(torch.Tensor) -> Module A
#       -> DTensor input -> Sharded Module B -> DTensor output
#           -> output (torch.Tensor) -> Module C -> output (torch.Tensor)
#
# We need the conversion from Module A to DTensor input, which is
# `from_local`, and conversion from DTensor output to output, which
# is `to_local`, thus these two functions must be Autograd functions.
#
class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: "DTensor",
        grad_placements: Optional[Sequence[Placement]],
    ):
        ctx.dtensor_spec = input._spec  # 保存 DTensor 的规范信息到上下文
        ctx.grad_placements = grad_placements  # 保存梯度放置信息到上下文
        local_tensor = input._local_tensor  # 获取 DTensor 的本地张量

        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to pollute the Tensor
        # object stored in the _local_tensor of this DTensor.
        return local_tensor.view_as(local_tensor)  # 返回与本地张量形状相同的新张量对象

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        # 获取上下文中的数据张量规范
        dtensor_spec = ctx.dtensor_spec
        # 获取数据张量规范中的网格信息
        mesh = dtensor_spec.mesh
        # 获取上下文中的梯度放置信息
        grad_placements = ctx.grad_placements
        # 获取数据张量规范中的张量元数据信息
        dtensor_meta = dtensor_spec.tensor_meta

        # 计算全局张量信息，包括张量步长
        _, tensor_stride = compute_global_tensor_info(
            grad_output, mesh, dtensor_spec.placements
        )
        # 将张量步长转换为元组形式
        tensor_stride = tuple(tensor_stride)
        # 如果梯度放置信息为空，则使用数据张量规范中的放置信息
        grad_placements = grad_placements or dtensor_spec.placements
        # 构建梯度规范对象
        grad_spec = DTensorSpec(
            mesh,
            grad_placements,
            tensor_meta=TensorMeta(
                shape=dtensor_meta.shape,
                stride=tensor_stride,
                dtype=dtensor_meta.dtype,
            ),
        )

        # 返回梯度张量对象及空值
        return (
            DTensor(
                grad_output,
                grad_spec,
                requires_grad=grad_output.requires_grad,
            ),
            None,
        )
    # 定义一个用于从 Torch 张量生成分布式张量的自定义函数
    class _FromTorchTensor(torch.autograd.Function):
        @staticmethod
        def forward(  # type: ignore[override]
            ctx,  # pyre-ignore[2]: Parameter must be annotated.
            input: torch.Tensor,
            device_mesh: DeviceMesh,
            placements: Tuple[Placement, ...],
            run_check: bool,
            shape: Optional[torch.Size] = None,
            stride: Optional[Tuple[int, ...]] = None,
        ) -> "DTensor":
            # 将设备布局和放置信息存储在上下文中，以备反向传播使用
            ctx.previous_placement = placements
            ctx.previous_device_mesh = device_mesh

            # 根据是否提供了形状和步长信息来确定张量的形状和步长
            if shape and stride:
                tensor_shape, tensor_stride = shape, stride
            elif not shape and not stride:
                # 如果没有提供形状和步长，默认情况下根据输入计算全局形状和步长
                global_shape, global_stride = compute_global_tensor_info(
                    input, device_mesh, placements
                )
                tensor_shape, tensor_stride = torch.Size(global_shape), tuple(global_stride)
            else:
                # 如果提供了形状或步长但不完整，抛出运行时错误
                raise RuntimeError(
                    f"Found shape:{shape}, stride:{stride}.",
                    "Please pass both shape and stride at the same time.",
                )

            if device_mesh.get_coordinate() is None:
                # 如果全局排名不参与设备布局，将本地张量设置为空张量
                input = input.new_empty(0, requires_grad=input.requires_grad)
            elif run_check:
                # 如果需要运行检查，根据放置信息进行张量元信息的广播检查
                # 仅当 run_check 为 True 时才广播到所有副本
                for idx, placement in enumerate(placements):
                    if placement.is_replicate():
                        # 将输入张量转换为连续存储，并在设备布局中进行广播
                        input = input.contiguous()
                        mesh_broadcast(input, device_mesh, mesh_dim=idx)

            # 构造分布式张量的规格说明
            dist_spec = DTensorSpec(
                device_mesh,
                placements,
                tensor_meta=TensorMeta(
                    tensor_shape,
                    tensor_stride,
                    input.dtype,
                ),
            )

            # 创建一个新的分布式张量对象，与输入张量共享内存
            dist_tensor = DTensor(
                input.view_as(input),
                dist_spec,
                # 分布式张量的 requires_grad 取决于输入张量的 requires_grad
                requires_grad=input.requires_grad,
            )
            return dist_tensor

        @staticmethod
    def backward(ctx, grad_output: "DTensor"):  # type: ignore[override]
        # 获取上下文中保存的先前的放置信息和设备网格信息
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh

        # 重新分布到创建分布式张量时的放置，以保证梯度布局匹配，可以直接返回本地梯度
        if grad_output.placements != previous_placement:
            current_spec = grad_output._spec
            # 创建目标规格，指定先前的设备网格和放置，保留梯度的张量元数据
            target_spec = DTensorSpec(
                previous_device_mesh,
                previous_placement,
                tensor_meta=grad_output._spec.tensor_meta,
            )
            local_tensor = grad_output._local_tensor
            # 重新分布本地张量到目标规格，用于反向传播，标记为反向传播
            output = redistribute_local_tensor(
                local_tensor, current_spec, target_spec, is_backward=True
            )
            # TODO: 直接返回重新分布的本地张量，而不进行可微的反向传播。查看这是否对所有情况都有意义。
            return output, None, None, None, None, None

        # TODO: 现在 backward 也是可微的，添加一个测试来测试更高级别的梯度。
        return grad_output.to_local(), None, None, None, None, None
# 定义一个名为 DTensor 的类，继承自 torch.Tensor 类
class DTensor(torch.Tensor):  # pyre-ignore[13]: pyre is bad at __new__
    # 实例变量 _local_tensor，用于存储本地的 torch.Tensor 对象
    _local_tensor: torch.Tensor
    # 实例变量 _spec，用于存储 DTensorSpec 对象
    _spec: DTensorSpec
    # __slots__ 定义了实例只能有 _local_tensor 和 _spec 两个属性，节省内存
    __slots__ = ["_local_tensor", "_spec"]

    # class attribute that handles operator placements propagation
    # rules, keyed by aten op name, value is propagation func
    # _op_dispatcher 是 OpDispatcher 类的实例，用于处理运算符的传播规则
    _op_dispatcher: op_dispatch.OpDispatcher = op_dispatch.OpDispatcher()

    @staticmethod
    # 禁用动态图功能，该装饰器用于静态方法
    @torch._disable_dynamo
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: DTensorSpec,
        *,
        requires_grad: bool,
    ) -> "DTensor":
        """
        Construct a DTensor from a local tensor, device mesh, and placement and
        other tensor properties (i.e. shape, requires_grad, strides, etc).
        Note: This is not a public API and it's only supposed to be used by the
            operator implementations and internals. If you want to construct a
            DTensor from a local tensor, consider using `DTensor.from_local`, if
            you want to construct a DTensor from a "global" tensor (where you
            already have tensor initialized and want to shard this tensor),
            consider using `distribute_tensor`.
        """
        # 如果 local_tensor 需要梯度但 requires_grad 为 False，发出警告信息
        if local_tensor.requires_grad and not requires_grad:
            warnings.warn(
                "To construct DTensor from torch.Tensor, it's recommended to "
                "use local_tensor.detach() and make requires_grad consistent."
            )

        # 使用 torch.Tensor._make_wrapper_subclass 方法创建包装子类 r
        # 该方法不会实际进行张量的分发，只是为本地张量添加包装器和位置规范
        assert spec.tensor_meta is not None, "TensorMeta should not be None!"
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            spec.tensor_meta.shape,
            strides=spec.tensor_meta.stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )

        # 设置 r 的 _spec 属性为传入的 spec
        r._spec = spec
        # 设置 r 的 _local_tensor 属性为传入的 local_tensor
        r._local_tensor = local_tensor
        return r

    # pyre-fixme[14]: `__repr__` overrides method defined in `DTensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def __repr__(self):
        # 返回 DTensor 对象的字符串表示形式，包括 _local_tensor、_spec.mesh 和 _spec.placements
        # TODO: consider all_gather the local tensors for better debugging
        return f"DTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        # 返回一个元组，包含 "_local_tensor" 和 (self._spec, self.requires_grad)
        # 用于指示如何将 DTensor 展平为本地张量，以进行 PT2 追踪
        return ["_local_tensor"], (self._spec, self.requires_grad)

    @staticmethod
    # 将扁平化后的张量重新展开为 `DTensor` 对象
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        # 断言确保扁平化规范不为 None，来自于 `__tensor_flatten__` 的返回值
        assert (
            flatten_spec is not None
        ), "Expecting spec to be not None from `__tensor_flatten__` return value!"
        # 获取内部张量 `_local_tensor`
        local_tensor = inner_tensors["_local_tensor"]
        # 解包扁平化规范和梯度要求
        spec, requires_grad = flatten_spec
        # 创建未展开的张量元信息 `TensorMeta`
        unflatten_tensor_meta = TensorMeta(
            shape=outer_size,
            stride=outer_stride,
            dtype=spec.tensor_meta.dtype,
        )
        # 创建未展开的规范 `DTensorSpec`
        unflatten_spec = DTensorSpec(
            spec.mesh,
            spec.placements,
            tensor_meta=unflatten_tensor_meta,
        )
        # 返回创建的 `DTensor` 对象
        return DTensor(
            local_tensor,
            unflatten_spec,
            requires_grad=requires_grad,
        )

    # 强制转换与切线相关的元数据
    def __coerce_tangent_metadata__(self):
        # 如果 `self.placements` 中包含任何 `Partial` 类型的元素
        if not any(isinstance(p, Partial) for p in self.placements):
            return self
        # 将所有 `Partial` 类型的元素替换为 `Replicate` 类型
        placements = [
            Replicate() if isinstance(p, Partial) else p for p in self.placements
        ]
        # 重新分配元数据并返回结果
        return self.redistribute(device_mesh=self.device_mesh, placements=placements)

    # 强制转换为与切线相关的相同元数据
    def __coerce_same_metadata_as_tangent__(self, flatten_spec):
        # 解包扁平化规范的结果
        (spec, _) = flatten_spec  # Result of tensor_flatten()
        # 根据规范中的放置信息重新分配元数据
        return self.redistribute(
            device_mesh=self.device_mesh,
            placements=spec.placements,
        )

    # 类方法：处理 Torch 分发的方法
    @classmethod
    @torch._disable_dynamo
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 使用 `_op_dispatcher` 分发方法调用
        return DTensor._op_dispatcher.dispatch(
            func,
            args,
            kwargs or {},
        )

    # 静态方法：从本地张量创建 `DTensor` 对象
    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        *,
        run_check: bool = True,
        shape: Optional[torch.Size] = None,
        stride: Optional[Tuple[int, ...]] = None,
    # 将 `DTensor` 对象转换为本地张量
    def to_local(
        self, *, grad_placements: Optional[Sequence[Placement]] = None
    ) -> torch.Tensor:
        """
        Get the local tensor of this DTensor on its current rank. For sharding it returns
        a local shard of the logical tensor view, for replication it returns the replica on
        its current rank.

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the Tensor returned from this
                function.
                `to_local` converts DTensor to local tensor and the returned local tensor
                might not be used as the original DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original DTensor layout.
                If not specified, we will assume the gradient layout remains the same
                as the original DTensor and use that for gradient computation.

        Returns:
            A :class:`torch.Tensor` or `AsyncCollectiveTensor` object. it represents the
            local tensor on its current rank.

        .. note:: `to_local` is differentiable, the `requires_grad` of the local tensor returned
            will depend on if the `DTensor` requires_grad or not.
        """
        # Check if gradient computation is enabled in torch
        if not torch.is_grad_enabled():
            # If not enabled, return the local tensor without any gradient tracking
            return self._local_tensor

        # Ensure grad_placements is converted to a tuple if provided and not already
        if grad_placements is not None and not isinstance(grad_placements, tuple):
            grad_placements = tuple(grad_placements)

        # Apply custom autograd function _ToTorchTensor.apply to convert DTensor to torch.Tensor
        return _ToTorchTensor.apply(
            self, grad_placements
        )  # pyre-ignore[16]: autograd func

    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        *,
        async_op: bool = False,
    ) -> "DTensor":
        """
        `redistribute` performs necessary collective operations that redistribute the current
        DTensor from its current placements to a new placements, or from is current DeviceMesh
        to a new DeviceMesh. i.e. we can turn a Sharded DTensor to a Replicated DTensor by
        specifying a Replicate placement for each dimension of the DeviceMesh.

        Args:
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                DTensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the new placements that
                describes how to place the DTensor into the DeviceMesh, must
                have the same number of elements as `device_mesh.ndim`.

        Keyword args:
            async_op (bool, optional): whether to perform the DTensor redistribute operation
                asynchronously or not. Default: False

        Returns:
            A :class:`DTensor` object

        .. note:: `redistribute` is differentiable.
        """
        # NOTE: This redistribute API currently only supports out
        # of place redistribution, i.e. it always create a new
        # DTensor object and leave the original one unchanged.

        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self.device_mesh
        # raise error if new placements not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")

        placements = list(placements)
        for i, placement in enumerate(placements):
            if placement.is_partial():
                raise RuntimeError(
                    "Can not redistribute to Partial, redistributing to Partial is for internal use only!"
                )
            elif isinstance(placement, Shard) and placement.dim < 0:
                # normalize shard dim to be positive
                placements[i] = Shard(placement.dim + self.ndim)
        placements = tuple(placements)

        # pyre-fixme[16]: `Redistribute` has no attribute `apply`.
        return Redistribute.apply(self, device_mesh, placements, async_op)
    ) -> torch.Tensor:
        """
        Return the full tensor of this DTensor. It will perform necessary collectives
        to gather the local tensors from other ranks in its DeviceMesh and concatenate
        them together. It's a syntatic sugar of the following code:

        `dtensor.redistribute(placements=[Replicate()] * mesh.ndim).to_local()`

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the full Tensor returned from this
                function.
                `full_tensor` converts DTensor to a full torch.Tensor and the returned torch.tensor
                might not be used as the original replicated DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original replicated DTensor layout.
                If not specified, we will assume the gradient layout of the full tensor be replicated.

        Returns:
            A :class:`torch.Tensor` object that represents the full tensor of this DTensor.

        .. note:: `full_tensor` is differentiable.
        """

        # Redistribute the DTensor across all replicas in the mesh and gather the result
        redist_res = self.redistribute(
            placements=[Replicate()] * self.device_mesh.ndim, async_op=False
        )
        # Convert the redistributed DTensor into a torch.Tensor
        return _ToTorchTensor.apply(redist_res, grad_placements)

    @property
    def device_mesh(self) -> DeviceMesh:
        """
        The :class:`DeviceMesh` attribute that associates with this DTensor object.

        .. note:: device_mesh is a read-only property, it can not be set.
        """
        # Return the device mesh associated with this DTensor
        return self._spec.mesh

    @property
    def placements(self) -> Sequence[Placement]:
        """
        The placements attribute of this DTensor that describes the layout of this
        DTensor on the its DeviceMesh.

        .. note:: placements is a read-only property, it can not be set.
        """
        # Return the placements that describe the layout of this DTensor on its mesh
        return self._spec.placements

    def __create_write_items__(self, fqn: str, object: Any):
        from torch.distributed.checkpoint.planner_helpers import (
            _create_write_items_for_dtensor,
        )

        if hasattr(self._local_tensor, "__create_write_items__"):
            # Call the __create_write_items__ method on the local tensor
            return self._local_tensor.__create_write_items__(fqn, object)  # type: ignore[attr-defined]
        elif isinstance(self._local_tensor, torch.Tensor):
            # Create write items for checkpointing the DTensor if it's a torch.Tensor
            return [_create_write_items_for_dtensor(fqn, object)]
        else:
            # Raise an error if the tensor type is unsupported
            raise RuntimeError("Unsupported tensor type!")
    # 如果本地张量对象有定义 "__create_chunk_list__" 方法，则调用该方法返回结果
    from torch.distributed.checkpoint.planner_helpers import (
        _create_chunk_from_dtensor,
    )

    if hasattr(self._local_tensor, "__create_chunk_list__"):
        return self._local_tensor.__create_chunk_list__()  # type: ignore[attr-defined]
    # 如果本地张量对象是 torch.Tensor 类型，则创建一个由该张量生成的分块列表
    elif isinstance(self._local_tensor, torch.Tensor):
        return [_create_chunk_from_dtensor(self)]
    else:
        # 如果类型不支持以上操作，则抛出运行时错误
        raise RuntimeError("Unsupported tensor type!")

    # 如果本地张量对象有定义 "__get_tensor_shard__" 方法，则调用该方法返回结果
    def __get_tensor_shard__(self, index):
        if hasattr(self._local_tensor, "__get_tensor_shard__"):
            return self._local_tensor.__get_tensor_shard__(index)  # type: ignore[attr-defined]
        # 如果本地张量对象是 torch.Tensor 类型，则返回其本地副本的数据分片
        elif isinstance(self._local_tensor, torch.Tensor):
            return self.to_local()
        else:
            # 如果类型不支持以上操作，则抛出运行时错误
            raise RuntimeError("Unsupported tensor type!")
def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Distribute a leaf torch.Tensor (i.e. nn.Parameter) to the ``device_mesh`` according
    to the ``placements`` specified. The rank of ``device_mesh`` and ``placements`` must be
    the same. If you want to construct a DTensor in the middle of the Autograd computation,
    please use ``DTensor.from_local`` instead.

    Args:
        tensor (torch.Tensor): torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use ``torch.chunk``
            semantic to shard the tensor and scatter the shards.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as `device_mesh.ndim`. If not specified, we will
            by default replicate the tensor across the `device_mesh` from the
            first rank of each dimension of the `device_mesh`.

    Returns:
        A :class:`DTensor` or `XLAShardedTensor` object.

    Note:
        When initialize the DeviceMesh with the `xla` device_type, `distribute_tensor`
        return `XLAShardedTensor` instead. see [link](https://github.com/pytorch/pytorch/issues/92909)
        for more details. The XLA integration is experimental and subject to change.
    """

    # Log the usage of this API in Torch
    torch._C._log_api_usage_once("torch.dtensor.distribute_tensor")

    # get default device mesh if there's nothing specified
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type
    
    # Check if the device type is 'xla'; if so, distribute tensor using xla_distribute_tensor
    if device_type == "xla":
        try:
            # Import and use the XLA SPMD for distributing tensor
            from torch_xla.distributed.spmd import (
                xla_distribute_tensor,
            )
            return xla_distribute_tensor(
                tensor, device_mesh, placements
            )  # type:ignore[return-value]
        except ImportError as e:
            # Raise an ImportError if torch_xla package is not installed
            msg = "To use DTensor API with xla, you must install the torch_xla package!"
            raise ImportError(msg) from e

    # Instantiate a RNG tracker if not already instantiated and supported by the mesh
    if not random._rng_tracker and is_rng_supported_mesh(device_mesh):
        # Use OffsetBasedRNGTracker for random operators in DTensor
        random._rng_tracker = OffsetBasedRNGTracker(device_type)
    # 检查是否为叶子节点张量，如果不是，则抛出错误
    if not tensor.is_leaf:
        raise RuntimeError(
            "`distribute_tensor` should be used to distribute leaf tensors! but found non-leaf tensor!"
        )

    # 如果张量不在指定的设备类型上且不是元张量，将其转换为对应的设备类型
    if device_type != tensor.device.type and not tensor.is_meta:
        tensor = tensor.to(device_type)

    # 如果未指定放置策略，默认设置为复制
    if placements is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]

    # 检查放置策略与设备网格维度是否匹配
    if len(placements) != device_mesh.ndim:
        raise ValueError(
            f"`placements` must have the same length as `device_mesh.ndim`! "
            f"Found placements length: {len(placements)}, and device_mesh.ndim: {device_mesh.ndim}."
        )

    if isinstance(tensor, DTensor):
        # 如果张量已经是一个DTensor，需要检查：
        # 1. 如果两个设备网格属于同一个父网格且进一步分片是可能的
        # 2. 检查设备网格和放置策略是否相同
        if tensor.device_mesh != device_mesh:
            raise ValueError(
                f"Cannot distribute a DTensor with device mesh {tensor.device_mesh} "
                f"to a different device mesh {device_mesh}."
            )
        if tensor.placements != tuple(placements):
            raise ValueError(
                f"Cannot distribute a DTensor with placements {tensor.placements} "
                f"to a different placements {placements}. do you want to call "
                f"`redistribute` instead?"
            )
        return tensor

    local_tensor = tensor.detach()

    # 根据放置策略分片张量
    placements = list(placements)
    for idx, placement in enumerate(placements):
        if placement.is_shard():
            placement = cast(Shard, placement)
            if placement.dim < 0:
                # 规范化分片放置维度
                placement = Shard(placement.dim + tensor.ndim)
                placements[idx] = placement
            local_tensor = placement._shard_tensor(local_tensor, device_mesh, idx)
        elif placement.is_replicate():
            placement = cast(Replicate, placement)
            local_tensor = placement._replicate_tensor(local_tensor, device_mesh, idx)
        else:
            raise RuntimeError(
                f"Trying to distribute tensor with unsupported placements {placement} on device mesh dimension {idx}!"
            )
    placements = tuple(placements)

    assert local_tensor is not None, "distributing a tensor should not be None"
    # 将传递给DTensor的本地张量分离，因为在构造DTensor之后，autograd会在DTensor的基础上工作，而不是本地张量
    # 创建一个 DTensorSpec 对象，用于描述张量的规格
    spec = DTensorSpec(
        # 使用指定的设备网格来描述张量的分布
        mesh=device_mesh,
        # 指定张量的放置方案
        placements=placements,
        # 使用 TensorMeta 对象描述张量的元信息，包括形状、步长和数据类型
        tensor_meta=TensorMeta(
            shape=tensor.size(),    # 张量的形状
            stride=tensor.stride(),  # 张量的步长
            dtype=tensor.dtype,      # 张量的数据类型
        ),
    )
    
    # 创建一个 DTensor 对象，封装了本地张量的梯度信息和规格信息
    return DTensor(
        local_tensor.requires_grad_(tensor.requires_grad),  # 将本地张量设置为需要梯度计算
        spec,  # 使用之前创建的规格描述对象
        requires_grad=tensor.requires_grad,  # 标记张量是否需要梯度计算
    )
# 定义函数 distribute_module，用于将 nn.Module 中的参数进行分发和控制
def distribute_module(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]] = None,
    input_fn: Optional[Callable[[nn.Module, Any, DeviceMesh], None]] = None,
    output_fn: Optional[Callable[[nn.Module, Any, DeviceMesh], None]] = None,
) -> nn.Module:
    """
    This function expose three functions to control the Tensors inside the module:
    1. To perform sharding on the module before runtime execution by specifying the
        ``partition_fn`` (i.e. allow user to convert Module parameters to :class:`DTensor`
        parameters according to the `partition_fn` specified).
    2. To control the inputs or outputs of the module during runtime execution by
        specifying the ``input_fn`` and ``output_fn``. (i.e. convert the input to
        :class:`DTensor`, convert the output back to torch.Tensor)

    Args:
        module (:class:`nn.Module`): user module to be partitioned.
        device_mesh (:class:`DeviceMesh`): the device mesh to place the module.
        partition_fn (Callable): the function to partition parameters (i.e. shard certain
            parameters across the `device_mesh`). If `partition_fn` is not specified,
            by default we replicate all module parameters of `module` across the mesh.
        input_fn (Callable): specify the input distribution, i.e. could control how the
            input of the module is sharded. `input_fn` will be installed as a module
            `forward_pre_hook` (pre forward hook).
        output_fn (Callable): specify the output distribution, i.e. could control how the
            output is sharded, or convert it back to torch.Tensor. output_fn will be
            installed as a module `forward_hook` (post forward hook).

    Returns:
        A module that contains parameters/buffers that are all `DTensor`s.

    Note:
        When initialize the DeviceMesh with the `xla` device_type, `distribute_module`
        return nn.Module with PyTorch/XLA SPMD annotated parameters. See [link](https://github.com/pytorch/pytorch/issues/92909)
        for more details. The XLA integration is experimental and subject to change.
    """

    # 记录 API 使用，一次性日志记录
    torch._C._log_api_usage_once("torch.dtensor.distribute_module")

    # 如果未指定 device_mesh，则使用当前的设备网格资源
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    # 获取设备类型
    device_type = device_mesh.device_type
    if device_type == "xla":
        try:
            # 尝试导入 torch_xla.distributed.spmd 模块，用于自动或显式分区模块参数
            from torch_xla.distributed.spmd import (
                xla_distribute_module,
            )
            
            # 调用 xla_distribute_module 函数，分区模块并返回分区后的模块
            return xla_distribute_module(
                module, device_mesh, partition_fn, input_fn, output_fn
            )  # type:ignore[return-value]
        except ImportError as e:
            # 如果导入失败，则抛出 ImportError，提醒用户安装 torch_xla 包
            msg = "To use DTensor API with xla, you must install the torch_xla package!"
            raise ImportError(msg) from e

    def replicate_module_params_buffers(m: nn.Module, mesh: DeviceMesh) -> None:
        # 这个函数循环遍历模块的参数和缓冲区，复制所有非 DTensor 参数/缓冲区到 DTensor 参数/缓冲区
        # 如果它们没有在 partition_fn 中分区，我们不能简单地使用 `module._apply`，
        # 因为我们不知道 partition_fn 内部发生了什么，用户可能会执行任何操作，比如安装钩子，
        # 我们希望保留这些操作。

        # 创建一个 Replicate 对象列表，长度为 mesh.ndim
        full_replicate = [Replicate()] * mesh.ndim
        
        # 遍历模块的参数，如果参数不是 None 并且不是 DTensor 类型，则将其复制并注册为 DTensor 类型的参数
        for key, param in m._parameters.items():
            if param is not None and not isinstance(param, DTensor):
                m.register_parameter(
                    key,
                    nn.Parameter(distribute_tensor(param.data, mesh, full_replicate)),
                )
        
        # 遍历模块的缓冲区，如果缓冲区不是 None 并且不是 DTensor 类型，则将其复制并分发为 DTensor 类型的缓冲区
        for key, buffer in m._buffers.items():
            if buffer is not None and not isinstance(buffer, DTensor):
                m._buffers[key] = distribute_tensor(buffer, mesh, full_replicate)

    if partition_fn is None:
        # 如果未指定 partition_fn，则默认复制所有模块的参数和缓冲区
        # 遍历模块的所有子模块，对每个子模块调用 replicate_module_params_buffers 函数
        for name, submod in module.named_modules():
            replicate_module_params_buffers(submod, device_mesh)
    else:
        # 如果指定了 partition_fn，则将 partition_fn 应用于每个子模块
        # 遍历模块的所有子模块，对每个子模块先调用 partition_fn，再调用 replicate_module_params_buffers 函数
        for name, submod in module.named_modules():
            partition_fn(name, submod, device_mesh)
            replicate_module_params_buffers(submod, device_mesh)

    # 将 input_fn 注册为模块的前向预钩子
    if input_fn is not None:
        # 检查 input_fn 的参数签名
        num_args = len(inspect.signature(input_fn).parameters)
        if num_args == 2:
            # input_fn 只接受 inputs 和 device_mesh 两个参数
            warnings.warn(
                "Deprecating input_fn that takes two arguments (inputs, device_mesh), "
                "please use input_fn that takes in (module, inputs, device_mesh) instead!",
                FutureWarning,
                stacklevel=2,
            )
            # 注册前向钩子，将 lambda 函数绑定到 input_fn
            module.register_forward_pre_hook(lambda _, inputs: input_fn(inputs, device_mesh))  # type: ignore[call-arg]
        elif num_args == 3:
            # input_fn 接受 module, inputs, device_mesh 三个参数
            module.register_forward_pre_hook(
                lambda mod, inputs: input_fn(mod, inputs, device_mesh)
            )
        else:
            # 抛出异常，input_fn 应该接受 3 个参数
            raise ValueError(
                f"input_fn should take in 3 arguments, but got {num_args} arguments!"
            )
    # 将 output_fn 注册为模块的前向钩子
    if output_fn is not None:
        num_args = len(inspect.signature(output_fn).parameters)
        if num_args == 2:
            # output_fn 只接受 outputs 和 device_mesh 两个参数
            warnings.warn(
                "Deprecating output_fn that takes two arguments (inputs, device_mesh), "
                "please use output_fn that takes in (module, inputs, device_mesh) instead!",
                FutureWarning,
                stacklevel=2,
            )
            # 注册前向钩子，将 lambda 函数绑定到 output_fn
            module.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(outputs, device_mesh)  # type: ignore[call-arg]
            )
        elif num_args == 3:
            # output_fn 接受 module, outputs, device_mesh 三个参数
            module.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(mod, outputs, device_mesh)
            )
        else:
            # 抛出异常，output_fn 应该接受 3 个参数
            raise ValueError(
                f"output_fn should take in 3 arguments, but got {num_args} arguments!"
            )

    return module
```