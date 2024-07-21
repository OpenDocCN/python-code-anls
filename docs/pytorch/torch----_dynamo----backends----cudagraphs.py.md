# `.\pytorch\torch\_dynamo\backends\cudagraphs.py`

```
# 忽略类型检查错误
# 导入 functools 模块，用于高阶函数操作
# 导入 operator 模块，用于操作符操作
# 导入 defaultdict 类，创建默认字典，用于按键进行分组
# 导入 Dict、List、Optional 类型提示，用于类型检查
import functools
import operator
from collections import defaultdict
from typing import Dict, List, Optional

# 导入 torch 库
import torch
# 导入 torch._dynamo.config 模块
from torch._dynamo import config
# 导入 torch._dynamo.backends.common.aot_autograd 模块
from torch._dynamo.backends.common import aot_autograd
# 导入 torch._dynamo.backends.debugging.boxed_nop 模块
from torch._dynamo.backends.debugging import boxed_nop
# 导入 torch._inductor.cudagraph_utils 模块中的多个函数和类
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    check_multiple_devices_or_any_cpu_nodes,
    format_default_skip_message,
    get_mutation_stack_trace,
    get_placeholders,
    log_cudagraph_skip_and_bump_counter,
)
# 导入 torch._inductor.utils 模块中的多个函数和类
from torch._inductor.utils import (
    BoxedBool,
    count_tangents,
    get_first_incompatible_cudagraph_node,
    num_fw_fixed_arguments,
    output_node,
)
# 导入 torch.multiprocessing.reductions.StorageWeakRef 类
from torch.multiprocessing.reductions import StorageWeakRef
# 从当前包中导入 registry 模块中的 register_backend 函数
from .registry import register_backend

# 定义函数 find_input_mutations，接收参数 g（图对象）
def find_input_mutations(g):
    # 定义内部函数 meta_fk，接收参数 meta，返回 meta 字典中 "val" 或 "fake_result" 的值
    def meta_fk(meta):
        return meta["val"] if "val" in meta else meta["fake_result"]

    # 创建默认字典 inputs，用于存储输入节点和其索引的集合
    inputs = defaultdict(set)
    # 初始化输入索引为 0
    input_idx = 0
    # 创建空集合 mutated_inputs，用于存储变异输入的集合
    mutated_inputs = set()
    
    # 遍历图对象中的节点
    for n in g.nodes:
        # 如果节点的操作为 "placeholder"
        if n.op == "placeholder":
            # 如果节点的 meta 值为 torch.Tensor 类型
            if isinstance(meta_fk(n.meta), torch.Tensor):
                # 将输入存储到 inputs 字典中对应的存储引用的集合中
                inputs[StorageWeakRef(meta_fk(n.meta)._typed_storage())].add(input_idx)
            # 输入索引加一
            input_idx += 1
        # 如果节点的操作为 "call_function"
        elif n.op == "call_function":
            # 如果调用的目标函数是 operator.getitem，则跳过
            if n.target is operator.getitem:
                continue
            # 获取调用函数的模式（schema）
            schema = n.target._schema
            # 遍历模式的参数列表
            for i, arg in enumerate(schema.arguments):
                # 如果索引 i 小于节点的参数长度，则使用位置参数
                if i < len(n.args):
                    argument = n.args[i]
                else:
                    # 否则，如果参数名不在关键字参数中，继续下一个循环
                    if arg.name not in n.kwargs:
                        continue
                    # 否则，使用关键字参数
                    argument = n.kwargs[arg.name]
                # 初始化 mut_arg 为 False
                mut_arg = False
                # 如果参数有别名信息
                if arg.alias_info:
                    # 如果别名信息表示写操作，则 mut_arg 置为 True
                    if arg.alias_info.is_write:
                        mut_arg = True
                # 如果 mut_arg 为 True
                if mut_arg:
                    # 根据参数的 meta 值的存储引用，更新 mutated_inputs 集合
                    mutated_inputs |= inputs[
                        StorageWeakRef(meta_fk(argument.meta)._typed_storage())
                    ]

        # TODO: 对于未识别节点报错处理
    
    # 返回 mutated_inputs 集合，包含所有变异输入的索引
    return mutated_inputs


# 定义函数 get_device_node_mapping，接收参数 gm（torch.fx.GraphModule 类型）
def get_device_node_mapping(gm: torch.fx.GraphModule):
    # 创建空字典 device_node_mapping，用于存储设备到节点的映射关系
    device_node_mapping: Dict[torch.device, torch.fx.Node] = {}
    # 遍历图对象中的节点
    for n in gm.graph.nodes:
        # 获取节点的 meta 中的值 t
        t = n.meta.get("val", None)
        # 如果 t 是 torch.Tensor 类型且其设备不在 device_node_mapping 中
        if isinstance(t, torch.Tensor) and t.device not in device_node_mapping:
            # 将设备与节点的映射关系存储到 device_node_mapping 中
            device_node_mapping[t.device] = n
    # 返回设备到节点的映射关系字典
    return device_node_mapping


# 定义函数 check_for_mutation_ignore_cuda_graph_managed_tensor，接收参数 aot_model（torch.fx.GraphModule 类型）和 num_fixed
def check_for_mutation_ignore_cuda_graph_managed_tensor(
    aot_model: torch.fx.GraphModule, num_fixed
) -> Optional[str]:
    # 查找输入节点的变异索引集合（排除前 num_fixed 个固定节点）
    mutation_indices = find_input_mutations(aot_model.graph) - set(range(num_fixed))
    # 如果没有找到变异索引，则返回 None
    if not mutation_indices:
        return None

    # 获取所有占位符节点列表
    placeholders = [node for node in aot_model.graph.nodes if node.op == "placeholder"]
    # 返回占位符节点的变异堆栈跟踪信息
    return get_mutation_stack_trace(placeholders, mutation_indices)
# 检查是否禁用了 CUDA 图的输入变异支持，如果是，则返回相应的跳过消息；否则返回 None
def check_for_skip(aot_model: torch.fx.GraphModule, num_fixed) -> Optional[str]:
    if not config.cudagraph_backend_support_input_mutation:
        # 检查是否存在忽略 CUDA 图管理张量变异的情况，如果有，则返回相应的跳过消息
        if mut_skip := check_for_mutation_ignore_cuda_graph_managed_tensor(
            aot_model, num_fixed
        ):
            return mut_skip

    # 检查是否存在多设备或任何 CPU 节点，如果是，则返回相应的跳过消息
    if skip := check_multiple_devices_or_any_cpu_nodes(
        get_device_node_mapping(aot_model)
    ):
        return skip

    # 获取第一个不兼容 CUDA 图节点，如果存在，则返回格式化的默认跳过消息
    if node := get_first_incompatible_cudagraph_node(aot_model):
        return format_default_skip_message(f"incompatible op ({node.name})")

    return None


# 获取图模型中第一个 CUDA 设备的索引
def get_device_index(gm) -> int:
    device = next(iter(get_device_node_mapping(gm)))
    assert device.type == "cuda"
    return device.index


# 获取图模型的输出节点的堆栈跟踪列表，如果节点不是 torch.fx.node.Node 类型则返回 None
def get_stack_traces(gm) -> List[Optional[str]]:
    output = output_node(gm)
    assert len(output.args) == 1
    return [
        (arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None)
        for arg in output.args[0]
    ]


# 定义 cudagraphs 函数，处理动态模型和动态输入
def cudagraphs(dynamo_model, dynamo_inputs):
    # 导入 cudagraphify_impl 函数
    from torch._inductor.cudagraph_trees import cudagraphify_impl

    # 创建一个布尔型的盒子对象，默认为 True
    do_cudagraphs = BoxedBool(True)
    # 创建一个装载设备索引的盒子对象，默认为 None
    boxed_device_index = BoxedDeviceIndex(None)

    # 定义 forward_cudagraphs 函数，用于执行 cudagraphs 的前向过程
    def forward_cudagraphs(aot_model, aot_inputs, is_inference=False):
        # 使用 boxed_nop 函数初始化 interp
        interp = boxed_nop(aot_model, aot_inputs)
        # 计算固定参数的数量
        fixed = num_fw_fixed_arguments(len(dynamo_inputs), len(aot_inputs))
        
        # 检查是否需要跳过 cudagraphs，如果是则记录日志并返回 interp
        if skip_msg := check_for_skip(aot_model, fixed):
            BoxedBool.disable(do_cudagraphs)
            log_cudagraph_skip_and_bump_counter(
                f"skipping cudagraphs due to {skip_msg}"
            )
            return interp

        # 设置 boxed_device_index 的值为当前模型的设备索引
        boxed_device_index.set(get_device_index(aot_model))
        
        # 执行 cudagraphify_impl 函数，生成 cudagraphs 并返回结果
        out = cudagraphify_impl(
            interp,
            aot_inputs,
            range(fixed),
            device_index=boxed_device_index.value,
            is_backward=False,
            is_inference=False,
            stack_traces=get_stack_traces(aot_model),
            placeholders=get_placeholders(aot_model.graph),
            mutated_input_idxs=find_input_mutations(aot_model.graph),
        )
        # 设置 out 对象的 _boxed_call 属性为 True
        out._boxed_call = True
        return out
    # 定义一个函数，用于生成反向传播的计算图
    def backward_cudagraphs(aot_model, aot_inputs):
        # 调用 boxed_nop 函数，将模型和输入参数封装
        interp = boxed_nop(aot_model, aot_inputs)
        
        # 如果不需要生成 cudagraphs，则直接返回原始模型
        if not do_cudagraphs:
            return aot_model
        
        # 计算模型中固定梯度张量的数量
        fixed = count_tangents(aot_model)
        
        # 检查是否需要跳过生成 cudagraphs 的消息
        if skip_msg := check_for_skip(aot_model, fixed):
            # 记录跳过 cudagraphs 的原因并增加计数器
            log_cudagraph_skip_and_bump_counter(
                "skipping cudagraphs due to %s", skip_msg
            )
            
            # 创建 cudagraph manager 对象，用于管理 cudagraphs 的生成
            # 参见 [Backward Generation Handling]
            manager = torch._inductor.cudagraph_trees.get_manager(
                boxed_device_index.value, create_if_none_exists=False
            )
            assert manager is not None
            
            # 定义一个函数 fn，用于在 cudagraph 管理器设置为反向运行时调用模型
            def fn(inputs):
                manager.set_to_running_backward()
                return aot_model(inputs)
            
            fn._boxed_call = True
            return fn
        
        # 生成 cudagraphs 的实现
        out = cudagraphify_impl(
            interp,
            aot_inputs,
            range(fixed),
            device_index=get_device_index(aot_model),
            is_backward=True,
            is_inference=False,
            stack_traces=get_stack_traces(aot_model),
            placeholders=get_placeholders(aot_model.graph),
            mutated_input_idxs=find_input_mutations(aot_model.graph),
        )
        out._boxed_call = True
        return out

    # 使用 aot_autograd 函数生成自动求导的 AOT 编译模型
    aot_cudagraphs = aot_autograd(
        fw_compiler=forward_cudagraphs,
        bw_compiler=backward_cudagraphs,
        inference_compiler=functools.partial(forward_cudagraphs, is_inference=True),
        keep_inference_input_mutations=torch._dynamo.config.cudagraph_backend_keep_input_mutation,
    )
    
    # 返回动态模型 dynamo_model 和输入 dynamo_inputs 的 AOT cudagraphs
    return aot_cudagraphs(dynamo_model, dynamo_inputs)
# 定义名为 CudagraphsBackend 的类，用于处理 cudagraphs 后端的相关功能
class CudagraphsBackend:
    # 设定编译器名称为 "cudagraphs"
    compiler_name = "cudagraphs"

    # 静态方法 reset，用于重置 cudagraph 树
    @staticmethod
    def reset():
        # 导入并调用 reset_cudagraph_trees 函数，来自 torch._inductor.cudagraph_trees 模块
        from torch._inductor.cudagraph_trees import reset_cudagraph_trees
        reset_cudagraph_trees()

    # 静态方法 __call__，接受 model 和 inputs 作为参数，调用 cudagraphs 函数并返回结果
    @staticmethod
    def __call__(model, inputs):
        return cudagraphs(model, inputs)


# 注册 cudagraphs 后端，命名为 "cudagraphs"，使用 CudagraphsBackend 类作为编译器
register_backend(name="cudagraphs", compiler_fn=CudagraphsBackend())


# 定义 cudagraphs_inner 函数，用于执行 CUDA 图相关操作，非后端注册函数，主要用于基准测试
def cudagraphs_inner(model, inputs, copy_outputs=True, copy_inputs=True):
    """This isn't registered as a backend, but is used in some benchmarks"""
    # 断言 inputs 是 list 或 tuple 类型
    assert isinstance(inputs, (list, tuple))

    # 根据 copy_inputs 参数，创建静态输入数据 static_inputs
    if copy_inputs:
        static_inputs = [torch.zeros_like(x) for x in inputs]
    else:
        static_inputs = list(inputs)

    # 预热阶段
    torch.cuda.synchronize()  # 同步 CUDA 设备
    stream = torch.cuda.Stream()  # 创建 CUDA 流
    stream.wait_stream(torch.cuda.current_stream())  # 等待当前流
    with torch.cuda.stream(stream):  # 在指定流上下文中执行
        model(*inputs)  # 执行模型前向传播
    stream.synchronize()  # 同步流
    torch.cuda.current_stream().wait_stream(stream)  # 等待当前流
    torch.cuda.synchronize()  # 再次同步 CUDA 设备

    # 记录阶段
    graph = torch.cuda.CUDAGraph()  # 创建 CUDA 图对象
    with torch.cuda.graph(graph, stream=stream):  # 在指定流上下文中创建图
        static_outputs = model(*static_inputs)  # 执行模型前向传播并记录到静态输出中
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)  # 确保 static_outputs 是 list 或 tuple 类型

    # 定义 run 函数，用于执行新的输入数据并重放图
    def run(*new_inputs):
        assert len(static_inputs) == len(new_inputs)  # 断言静态输入和新输入数据长度相同
        if copy_inputs:
            for dst, src in zip(static_inputs, new_inputs):
                dst.copy_(src)  # 如果需要，复制新输入数据到静态输入中
        graph.replay()  # 重放 CUDA 图
        if copy_outputs:
            return [x.clone() for x in static_outputs]  # 如果需要，克隆静态输出数据并返回
        else:
            return static_outputs  # 否则直接返回静态输出

    return run  # 返回 run 函数作为结果
```