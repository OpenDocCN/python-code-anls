# `.\pytorch\torch\_lazy\extract_compiled_graph.py`

```
# mypy: allow-untyped-defs
# 导入需要的模块
import copy  # 导入copy模块，用于对象复制
import dataclasses  # 导入dataclasses模块，用于创建数据类
import itertools  # 导入itertools模块，用于创建迭代器的函数
import os  # 导入os模块，用于与操作系统进行交互
from typing import Any, Callable, Dict, List  # 导入类型提示模块

import torch  # 导入PyTorch库
import torch._lazy as lazy  # 导入torch._lazy模块，用于延迟加载
import torch._lazy.metrics as metrics  # 导入torch._lazy.metrics模块，用于延迟加载的度量
from torch import fx  # 从torch模块导入fx子模块
from torch._lazy import computation, debug as lazy_debug  # 导入torch._lazy的computation和debug子模块
from torch._lazy.tensor_factory_functions import tensor_factory_functions  # 导入torch._lazy.tensor_factory_functions模块中的tensor_factory_functions函数

debug = os.environ.get("debug_extract_compiled_graph") is not None  # 根据环境变量设置debug标志


@dataclasses.dataclass
class GraphInputMatcher:
    """
    The GraphInputMatcher class setup the graph inputs for future calls after lazy tracing.
    Specifically, those graph inputs corresponding to method parameters should be replaced with the
    arguments for the current call.

    tensor_id_to_arg_idx maps the tensor id to the parameter index.
    graph_input_tensor_ids, graph_input_ivalues list the tensor_id and ivalue for each of the
    TS/XLA graph inputs.
    """

    tensor_id_to_arg_idx: Dict[int, int]  # 字典，将张量ID映射到参数索引
    graph_input_tensor_ids: List[int]  # 列表，存储图输入的张量ID
    # 两种图输入张量的类别：
    # 类别1：那些在tensor_id_to_arg_idx中找不到ID的张量，这些大多数情况下是常量张量，可以从graph_input_tensors中获取其内容
    # 类别2：那些在tensor_id_to_arg_idx中找到ID的张量，应从方法参数中获取张量
    graph_input_ivalues: List[Any]

    # 获取真实的图输入张量
    def __call__(self, args):
        real_input = []
        for tensor_id, traced_ivalue in zip(
            self.graph_input_tensor_ids, self.graph_input_ivalues
        ):
            arg_idx = self.tensor_id_to_arg_idx.get(tensor_id, None)  # 获取张量ID对应的参数索引
            if arg_idx is None:
                inp = traced_ivalue  # 如果找不到对应的参数索引，则使用追踪到的ivalue
            else:
                inp = args[arg_idx]  # 否则从方法参数中获取张量
            real_input.append(inp)  # 将获取到的张量添加到真实输入列表中
        return real_input  # 返回真实的输入列表


class ReturnValueHandler:
    r"""
    When ltc_sync_multi is called on multi tensors, the compiled graph
    will contain output only for unique tensors - if a tensor appears multiple
    times in the input to _ltc_sync_multi, only the first occurance matters.

    However from python level, we still expect multi tensors returned with duplciation
    even if the TS graph dedup the output. e.g. for method:

      def forward(self, a):
        return a, a

    the TS graph captured by LTC will return a single tensor, but Python method expects 2.

    This class dedup the lazy tensors first to get the index that will be used
    to duplicate the eager tensors later.
    """
    # 初始化方法，接受一个 lazy_out_list 参数
    def __init__(self, lazy_out_list):
        # 初始化索引列表为一个空列表
        self.index: List[List[int]] = []
        # 记录 lazy_out_list 的总长度
        self.total_count = len(lazy_out_list)

        # 创建一个字典，用于存储 lazy_tensor 对象的 id 到其在 self.index 中的唯一索引的映射关系
        tensor_id_to_idx: Dict[int, int] = {}
        
        # 遍历 lazy_out_list 中的每个 lazy_tensor 和其对应的索引
        for dup_idx, lazy_tensor in enumerate(lazy_out_list):
            # 获取 lazy_tensor 对象的 id 在 tensor_id_to_idx 中的唯一索引
            uniq_idx = tensor_id_to_idx.get(id(lazy_tensor), None)
            
            # 如果该 lazy_tensor 的 id 已经在字典中有了唯一索引，则将当前 dup_idx 加入到对应的索引列表中
            if uniq_idx is not None:
                self.index[uniq_idx].append(dup_idx)
            else:
                # 否则，为新的 lazy_tensor 创建一个新的唯一索引，并在 self.index 中添加一个新的列表，将当前 dup_idx 加入其中
                uniq_idx = len(self.index)
                self.index.append([dup_idx])
                tensor_id_to_idx[id(lazy_tensor)] = uniq_idx

    # 复制 eager_tensor_list 中的张量，以替代 lazy_out_list 中的张量
    def duplicate_eager_tensors(self, eager_tensor_list):
        # 创建一个与 lazy_out_list 长度相同的列表，用于存储替代后的张量
        duplicated_list = [None] * self.total_count
        # 断言 eager_tensor_list 的长度与 self.index 的长度相等
        assert len(eager_tensor_list) == len(self.index)

        # 遍历 eager_tensor_list 中的每个 eager_tensor 和其对应的唯一索引
        for uniq_idx, eager_tensor in enumerate(eager_tensor_list):
            # 遍历 self.index 中索引为 uniq_idx 的所有重复索引 dup_idx
            for dup_idx in self.index[uniq_idx]:
                # 将 eager_tensor 替代 lazy_out_list 中索引为 dup_idx 的张量
                duplicated_list[dup_idx] = eager_tensor
        # 返回替代后的列表
        return duplicated_list
def force_lazy_device(model: fx.GraphModule):
    """
    Factory methods in a Fx graph may create tensors for a specific eager devices.
    If we take no actions, those eager tensors will be mixed with lazy tensors and
    cause crash. This method overwrite those eager device to lazy device.
    """

    def tolazydevice(dev):
        if isinstance(dev, torch.device):
            return torch.device("lazy", index=dev.index)
        return dev

    def hasDeviceArg(args, kwargs):
        return any(
            isinstance(arg, torch.device)
            for arg in itertools.chain(args, kwargs.values())
        )

    # Iterate over all nodes in the Fx graph model
    for nd in model.graph.nodes:
        # Convert all arguments of the node to lazy devices
        nd.args = tuple(tolazydevice(arg) for arg in nd.args)
        nd.kwargs = {k: tolazydevice(v) for k, v in nd.kwargs.items()}

        # Check if the node corresponds to specific tensor factory functions
        if nd.target in tensor_factory_functions and not hasDeviceArg(
            nd.args, nd.kwargs
        ):
            # If no device argument is present, add a "device" argument with lazy device
            kwargs = dict(nd.kwargs)  # Make a mutable copy since nd.kwargs is immutable
            kwargs["device"] = torch.device("lazy")
            nd.kwargs = kwargs

    # Recompile the model after modifying the device assignments
    model.recompile()


def get_fallback_ops():
    """
    Generates a list of fallback operations based on the metric counters.
    """
    fallback_ops = []
    for opname in metrics.counter_names():
        # Consider only operations prefixed with "aten::"
        if "aten::" not in opname:
            continue
        val = int(metrics.counter_value(opname))
        # Add operations with non-zero counts to the fallback list
        if val > 0:
            fallback_ops.append(f"{opname}={val}")

    return fallback_ops


def extract_compiled_graph(model: fx.GraphModule, example_inputs) -> Callable:
    """
    Optimize an eager model with LTC and returns a wrapper to execute the
    compiled graph directly without retracing. It depends on other mechanisms
    like TorchDynamo guards to guarantee the returned wrapper is only called
    when it's safe.
    """
    # Convert example inputs to lazy devices
    lazy_args = [arg.to(device="lazy") for arg in example_inputs]
    # Retrieve tensor IDs for lazy arguments
    args_tensor_ids = [lazy.get_tensor_id(lazy_arg) for lazy_arg in lazy_args]
    # Map tensor IDs to their corresponding argument indices
    tensor_id_to_arg_idx = {tensor_id: i for i, tensor_id in enumerate(args_tensor_ids)}
    # Deep copy the model and set its device to lazy
    lazy_model = copy.deepcopy(model).to(device=torch.device("lazy"))
    # Apply the function to force all eager devices to lazy devices
    force_lazy_device(lazy_model)

    # Reset metrics to start fresh for lazy tracing
    metrics.reset()
    # 使用 lazy_model 和 lazy_args 调用 lazy_model 函数，获取 lazy 模型的输出
    lazy_out = lazy_model(*lazy_args)
    # 获取后备操作函数列表
    fallback_ops = get_fallback_ops()
    # 重置指标数据
    metrics.reset()

    # 如果存在后备操作函数，则抛出运行时错误，指示无法提取编译后的图形
    if len(fallback_ops) > 0:
        raise RuntimeError(
            f"Fail to extact the compiled graph because of fallback: {','.join(fallback_ops)}"
        )

    # 如果 lazy_out 不是 tuple 或 list 类型，则转换为单元素的元组
    if not isinstance(lazy_out, (tuple, list)):
        lazy_out = (lazy_out,)

    # 将 lazy_args 和 lazy_out 合并成一个元组作为参数，并创建 ReturnValueHandler 实例处理返回值
    args_and_out = tuple(lazy_args) + tuple(lazy_out)
    return_value_handler = ReturnValueHandler(args_and_out)

    # 如果处于调试模式，则打印模型代码和惰性调试信息
    if debug:
        print("Fx code:\n", model.code)
        print("LTC IR:", lazy_debug.dump_ir(args_and_out, "text"))

    # 获取图输入张量的标识符和值，使用 TS 后端特定的计算方法，准备后续计算
    (
        graph_input_tensor_ids,
        graph_input_ivalues,
    ) = computation.get_tensors_ts_device_data_node(args_and_out)
    assert len(graph_input_tensor_ids) == len(graph_input_ivalues)
    # 创建 GraphInputMatcher 实例，用于匹配图输入张量
    graph_input_matcher = GraphInputMatcher(
        tensor_id_to_arg_idx, graph_input_tensor_ids, graph_input_ivalues
    )

    # 获取当前计算图的哈希值
    graph_hash = computation.get_graph_hash(args_and_out)

    # 如果处于调试模式，则打印计算图的哈希值和相关张量的标识符
    if debug:
        print("graph_hash", graph_hash)
        print(f"args_tensor_ids {args_tensor_ids}")
        print("tensor ids from device data:", graph_input_tensor_ids)

    # 同步输出张量列表，以便后续通过图哈希获取计算图
    lazy.sync_multi(args_and_out, [])

    # 定义优化后的模型函数 optimized_mod
    def optimized_mod(*args):
        # 如果 args_and_out 为空，则返回空元组
        if len(args_and_out) == 0:
            return ()
        # 使用 graph_input_matcher 匹配输入参数 args，并运行缓存的计算图，获取结果
        graph_input = graph_input_matcher(args)
        res = return_value_handler.duplicate_eager_tensors(
            computation.run_cached_graph(graph_hash, graph_input)
        )

        # 断言结果长度与 args_and_out 相等
        assert len(res) == len(args_and_out)
        # 将结果中需要更新的张量复制到对应的参数中
        for i, arg in enumerate(args):
            if arg is not res[i]:
                arg.copy_(res[i])

        # 返回除参数外的结果
        return res[len(args):]

    # 返回优化后的模型函数 optimized_mod
    return optimized_mod
```