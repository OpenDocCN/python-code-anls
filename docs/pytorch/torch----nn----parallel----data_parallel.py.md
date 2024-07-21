# `.\pytorch\torch\nn\parallel\data_parallel.py`

```py
# mypy: allow-untyped-defs
# 引入操作符模块，用于对比设备属性
import operator
# 引入警告模块，用于在出现不平衡时发出警告
import warnings
# 引入 itertools 中的 chain 函数，用于合并可迭代对象
from itertools import chain
# 引入类型提示相关模块
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

# 引入 PyTorch 库
import torch
# 从 torch._utils 中导入若干函数
from torch._utils import (
    _get_all_device_indices,           # 获取所有设备索引
    _get_available_device_type,        # 获取可用设备类型
    _get_device_index,                 # 获取设备索引
    _get_devices_properties,           # 获取设备属性
)

# 从上级目录的 modules 模块中导入 Module 类
from ..modules import Module
# 从当前目录的 parallel_apply 模块中导入 parallel_apply 函数
from .parallel_apply import parallel_apply
# 从当前目录的 replicate 模块中导入 replicate 函数
from .replicate import replicate
# 从当前目录的 scatter_gather 模块中导入 gather 和 scatter_kwargs 函数
from .scatter_gather import gather, scatter_kwargs

# 定义模块中公开的接口
__all__ = ["DataParallel", "data_parallel"]


def _check_balance(device_ids: Sequence[Union[int, torch.device]]) -> None:
    # 不平衡警告信息模板
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    # 获取设备索引列表
    device_ids = [_get_device_index(x, True) for x in device_ids]
    # 获取设备属性列表
    dev_props = _get_devices_properties(device_ids)

    # 定义函数，用于发出不平衡警告
    def warn_imbalance(get_prop):
        # 获取所有设备的指定属性值
        values = [get_prop(props) for props in dev_props]
        # 找到具有最小和最大属性值的设备索引及其值
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        # 如果最小值与最大值之比小于0.75，则发出警告
        if min_val / max_val < 0.75:
            warnings.warn(
                imbalance_warn.format(device_ids[min_pos], device_ids[max_pos])
            )
            return True
        return False

    # 如果内存不平衡警告触发，则返回
    if warn_imbalance(lambda props: props.total_memory):
        return
    # 如果处理器核心数不平衡警告触发，则返回
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


# 定义类型变量 T，它是 Module 类的子类
T = TypeVar("T", bound=Module)


class DataParallel(Module, Generic[T]):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting the input across the specified devices by chunking in the batch
    dimension (other objects will be copied once per device). In the forward
    pass, the module is replicated on each device, and each replica handles a
    portion of the input. During the backwards pass, gradients from each replica
    are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    .. warning::
        It is recommended to use :class:`~torch.nn.parallel.DistributedDataParallel`,
        instead of this class, to do multi-GPU training, even if there is only a single
        node. See: :ref:`cuda-nn-ddp-instead` and :ref:`ddp`.

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel but some types are specially handled. tensors will be
    **scattered** on dim specified (default 0). tuple, list and dict types will
    be shallow copied. The other types will be shared among different threads
    and can be corrupted if written to in the model's forward pass.

    The parallelized :attr:`module` must have its parameters and buffers on
    # 在运行此:class:`~torch.nn.DataParallel`模块之前，请确保``device_ids[0]``正确设置。

    .. warning::
        在每次前向传播过程中，:attr:`module`会被**复制**到每个设备上，因此在``forward``过程中对运行的模块的任何更新都会丢失。例如，如果:attr:`module`具有一个计数器属性，在每次``forward``中递增，那么它始终会保持初始值，因为更新是在被销毁的副本上执行的。但是，:class:`~torch.nn.DataParallel`保证在``device[0]``上的副本将其参数和缓冲区与基础并行化的:attr:`module`共享存储。因此，在``device[0]``上的参数或缓冲区的原地更新将被记录下来。例如，:class:`~torch.nn.BatchNorm2d`和:func:`~torch.nn.utils.spectral_norm`依赖于此行为来更新缓冲区。

    .. warning::
        在:attr:`module`及其子模块上定义的前向和后向钩子将被调用``len(device_ids)``次，每次输入位于特定设备上。特别是，钩子只能保证按照对应设备上的操作正确顺序执行。例如，不能保证通过:meth:`~torch.nn.Module.register_forward_pre_hook`设置的钩子在所有``len(device_ids)``个:meth:`~torch.nn.Module.forward`调用之前执行，但可以保证每个这样的钩子在该设备的对应:meth:`~torch.nn.Module.forward`调用之前执行。

    .. warning::
        当:attr:`module`在:func:`forward`中返回一个标量（即0维张量）时，此包装器将返回一个长度等于使用数据并行处理的设备数的向量，其中包含每个设备的结果。

    .. note::
        在使用``pack sequence -> recurrent network -> unpack sequence``模式时，使用:class:`~torch.nn.DataParallel`包装的:class:`~torch.nn.Module`中存在一些微妙之处。有关详细信息，请参阅FAQ中的:ref:`pack-rnn-unpack-with-data-parallelism`部分。

    Args:
        module (Module): 要并行化的模块
        device_ids (list of int or torch.device): CUDA设备（默认为所有设备）
        output_device (int or torch.device): 输出位置的设备（默认为device_ids[0]）

    Attributes:
        module (Module): 要并行化的模块

    Example::

        >>> # xdoctest: +SKIP
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)  # input_var可以位于任何设备上，包括CPU
    # 初始化函数，用于初始化 DataParallel 对象
    def __init__(
        self,
        module: T,  # 参数module是泛型类型T的对象，即要并行处理的模块
        device_ids: Optional[Sequence[Union[int, torch.device]]] = None,  # 设备ID列表，可以是整数或torch.device对象的可选序列，默认为None
        output_device: Optional[Union[int, torch.device]] = None,  # 输出设备，可以是整数或torch.device对象的可选值，默认为None
        dim: int = 0,  # 指定数据切分的维度，默认为0
    ) -> None:
        super().__init__()  # 调用父类的初始化方法
        torch._C._log_api_usage_once("torch.nn.parallel.DataParallel")  # 记录 API 使用情况
        device_type = _get_available_device_type()  # 获取可用设备的类型（如"cuda"或"cpu"）
        if device_type is None:  # 如果没有可用的设备
            self.module = module  # 设置模块
            self.device_ids = []  # 设备ID列表为空
            return  # 返回

        if device_ids is None:  # 如果设备ID列表为None
            device_ids = _get_all_device_indices()  # 获取所有设备的索引

        if device_ids is None:  # 如果设备ID列表仍为None
            raise RuntimeError("no available devices were found")  # 抛出运行时错误，表示找不到可用设备

        if output_device is None:  # 如果输出设备为None
            output_device = device_ids[0]  # 设置输出设备为设备ID列表的第一个设备

        self.dim = dim  # 设置数据切分的维度
        self.module = module  # 设置模块
        self.device_ids = [_get_device_index(x, True) for x in device_ids]  # 获取每个设备的索引，并存储为列表
        self.output_device = _get_device_index(output_device, True)  # 获取输出设备的索引
        self.src_device_obj = torch.device(device_type, self.device_ids[0])  # 创建源设备对象，使用设备类型和第一个设备的索引

        if device_type == "cuda":  # 如果设备类型是CUDA
            _check_balance(self.device_ids)  # 检查设备的负载均衡情况

        if len(self.device_ids) == 1:  # 如果设备ID列表长度为1
            self.module.to(self.src_device_obj)  # 将模块移动到源设备上

    # 前向传播函数，重写父类的前向传播方法
    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        with torch.autograd.profiler.record_function("DataParallel.forward"):  # 使用自动求导分析器记录函数执行时间
            if not self.device_ids:  # 如果设备ID列表为空
                return self.module(*inputs, **kwargs)  # 直接在单设备上执行模块的前向传播

            for t in chain(self.module.parameters(), self.module.buffers()):  # 遍历模块的参数和缓冲区
                if t.device != self.src_device_obj:  # 如果参数或缓冲区不在源设备上
                    raise RuntimeError(
                        "module must have its parameters and buffers "
                        f"on device {self.src_device_obj} (device_ids[0]) but found one of "
                        f"them on device: {t.device}"
                    )  # 抛出运行时错误，要求参数和缓冲区必须在源设备上

            inputs, module_kwargs = self.scatter(inputs, kwargs, self.device_ids)  # 将输入数据和关键字参数分散到各个设备上
            # 处理没有任何输入的情况，创建空列表和字典，以便在第一个设备上执行模块
            if not inputs and not module_kwargs:
                inputs = ((),)
                module_kwargs = ({},)

            if len(self.device_ids) == 1:  # 如果设备ID列表长度为1
                return self.module(*inputs[0], **module_kwargs[0])  # 在单设备上执行模块的前向传播

            replicas = self.replicate(self.module, self.device_ids[: len(inputs)])  # 复制模块到各个设备
            outputs = self.parallel_apply(replicas, inputs, module_kwargs)  # 并行执行模块的前向传播
            return self.gather(outputs, self.output_device)  # 收集并返回输出到指定的输出设备

    # 复制模块到多个设备上
    def replicate(
        self, module: T, device_ids: Sequence[Union[int, torch.device]]
    ) -> List[T]:
        return replicate(module, device_ids, not torch.is_grad_enabled())  # 调用replicate函数进行模块复制

    # 将输入数据和关键字参数分散到多个设备上
    def scatter(
        self,
        inputs: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]],
        device_ids: Sequence[Union[int, torch.device]],
    ) -> Any:
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)  # 调用scatter_kwargs函数进行数据分散
    # 并行应用函数，将输入数据并行应用到多个副本上，并返回结果列表
    def parallel_apply(
        self, replicas: Sequence[T], inputs: Sequence[Any], kwargs: Any
    ) -> List[Any]:
        # 调用外部的并行应用函数 parallel_apply，传入副本列表、输入数据、额外参数和设备 ID 列表
        return parallel_apply(
            replicas, inputs, kwargs, self.device_ids[: len(replicas)]
        )
    
    # 聚集函数，将多个副本的输出聚集到指定的设备上，并返回聚集后的结果
    def gather(self, outputs: Any, output_device: Union[int, torch.device]) -> Any:
        # 调用外部的聚集函数 gather，传入输出数据、输出设备和聚集维度
        return gather(outputs, output_device, dim=self.dim)
def data_parallel(
    module: Module,
    inputs: Any,
    device_ids: Optional[Sequence[Union[int, torch.device]]] = None,
    output_device: Optional[Union[int, torch.device]] = None,
    dim: int = 0,
    module_kwargs: Optional[Any] = None,
) -> torch.Tensor:
    r"""Evaluate module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output. Use -1 to indicate the CPU.
            (default: device_ids[0])
        dim (int): dimension over which to scatter inputs to devices (default: 0)
        module_kwargs (Any): any additional keyword arguments to be passed to the module (default: None)

    Returns:
        torch.Tensor: a Tensor containing the result of module(input) located on output_device
    """
    # 如果输入不是元组，则转换为元组，确保能够迭代
    if not isinstance(inputs, tuple):
        inputs = (inputs,) if inputs is not None else ()

    # 获取可用的设备类型
    device_type = _get_available_device_type()

    # 如果设备类型无法确定，则引发运行时错误
    if device_type is None:
        raise RuntimeError("device type could not be determined")

    # 如果没有指定设备 IDs，则获取所有可用设备的索引
    if device_ids is None:
        device_ids = _get_all_device_indices()

    # 如果找不到可用设备，则引发运行时错误
    if device_ids is None:
        raise RuntimeError("no available devices were found")

    # 如果未指定输出设备，则默认为第一个设备 ID
    if output_device is None:
        output_device = device_ids[0]

    # 将设备 IDs 转换为索引形式
    device_ids = [_get_device_index(x, True) for x in device_ids]
    output_device = _get_device_index(output_device, True)

    # 获取源设备对象
    src_device_obj = torch.device(device_type, device_ids[0])

    # 检查模块的参数和缓冲区是否在源设备上，如果不在，则引发运行时错误
    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError(
                "module must have its parameters and buffers "
                f"on device {src_device_obj} (device_ids[0]) but found one of "
                f"them on device: {t.device}"
            )

    # 根据设备 IDs 和维度对输入数据进行分散
    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)

    # 如果没有输入和模块参数，则创建空列表和字典，使模块可以在第一个设备上执行
    if not inputs and not module_kwargs:
        inputs = ((),)
        module_kwargs = ({},)

    # 确保模块参数不为 None
    assert module_kwargs is not None

    # 如果设备 IDs 的长度为 1，则直接在该设备上执行模块
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])

    # 生成模块的副本并在多个设备上应用并行执行
    used_device_ids = device_ids[: len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)

    # 收集并聚合输出结果到指定的输出设备
    return gather(outputs, output_device, dim)
```