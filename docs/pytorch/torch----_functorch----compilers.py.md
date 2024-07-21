# `.\pytorch\torch\_functorch\compilers.py`

```
# 忽略类型检查错误
# 导入必要的库
import copy  # 导入copy模块，用于对象的浅拷贝和深拷贝操作
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，提供了操作系统相关的功能
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import random  # 导入random模块，用于生成随机数
from contextlib import contextmanager  # 导入contextmanager模块，用于创建上下文管理器
from functools import partial  # 导入partial模块，用于创建偏函数
from typing import Callable, Union  # 导入类型提示，指定参数和返回值的类型

import sympy  # 导入sympy库，用于符号计算

import torch  # 导入PyTorch库
import torch.fx as fx  # 导入PyTorch FX模块
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.utils._pytree as pytree  # 导入PyTorch的pytree工具模块
from torch import SymInt  # 导入SymInt类
from torch._decomp import get_decompositions  # 导入get_decompositions函数
from torch.fx.experimental.symbolic_shapes import bind_symbols  # 导入bind_symbols函数

from .aot_autograd import aot_function, aot_module, make_boxed_compiler  # 导入自定义模块中的函数和类
from .compile_utils import strip_overloads  # 导入自定义模块中的strip_overloads函数
from .partitioners import (  # 导入自定义模块中的分区相关函数
    default_partition,
    draw_graph,
    min_cut_rematerialization_partition,
)

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


# These canonicalizations are needed here (and not decompositions), as the ops
# we're trying to canonicalize to CompositeImplicitAutograd.
# 这些规范化操作在这里是必要的（而不是分解），因为我们试图将操作规范化为CompositeImplicitAutograd。

def _canonicalize(fx_g):
    # 遍历FX图中所有调用函数为torch.ops.aten._to_copy的节点
    for node in fx_g.graph.find_nodes(
        op="call_function", target=torch.ops.aten._to_copy
    ):
        node.target = torch.ops.aten.to  # 将节点的目标函数修改为torch.ops.aten.to
    fx_g.recompile()  # 重新编译FX图
    return fx_g  # 返回修改后的FX图对象


@contextmanager
def _disable_jit_autocast():
    old_jit_autocast_flag = torch._C._jit_set_autocast_mode(False)  # 关闭JIT的自动转换模式
    try:
        yield  # 返回上下文管理器的控制权
    finally:
        torch._C._jit_set_autocast_mode(old_jit_autocast_flag)  # 恢复JIT的自动转换模式设置


@make_boxed_compiler
def ts_compile(fx_g: fx.GraphModule, inps) -> Callable:
    """
    Compiles the :attr:`fx_g` with Torchscript compiler.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fx_g(fx.GraphModule): The input Fx graph module to be compiled.

    Returns:
        Torch scripted model.
    """

    with _disable_jit_autocast():  # 使用上下文管理器禁用JIT自动转换
        strip_overloads(fx_g)  # 剥离FX图中的重载函数

        # 遍历FX图中所有节点
        for node in fx_g.graph.find_nodes(
            op="call_function", target=torch.ops.aten._to_copy
        ):
            # 如果节点的参数长度为1，关键字参数长度为1，并且包含"dtype"关键字
            if len(node.args) == 1 and len(node.kwargs) == 1 and "dtype" in node.kwargs:
                node.target = torch.ops.aten.to  # 将节点的目标函数修改为torch.ops.aten.to

        # 遍历FX图中所有节点
        for node in fx_g.graph.nodes:
            new_kwargs = {}
            # 遍历节点的关键字参数
            for k, v in node.kwargs.items():
                if isinstance(v, torch.device):
                    v = v.type  # 如果值是torch.device类型，则将其转换为对应的类型
                new_kwargs[k] = v
            node.kwargs = new_kwargs  # 更新节点的关键字参数

        fx_g.graph.lint()  # 对FX图进行Lint检查

        fx_g.recompile()  # 重新编译FX图

        f = torch.jit.script(fx_g)  # 将FX图编译为Torch脚本模型

        torch._C._jit_pass_remove_mutation(f.graph)  # 移除脚本图中的突变操作

        f = torch.jit.freeze(f.eval())  # 冻结脚本模型并评估其表达式
        f = torch.jit.optimize_for_inference(f)  # 优化脚本模型以提升推理性能
        if not any(isinstance(t, torch._subclasses.FakeTensor) for t in inps):
            f(*inps)  # 如果输入不包含FakeTensor类型，则调用脚本模型
    return f  # 返回编译后的脚本模型


def _draw_graph_compile(fx_g, _, name, clear_meta=True):
    print(fx_g.code)  # 打印FX图的代码
    draw_graph(fx_g, name, clear_meta=clear_meta)  # 绘制FX图并显示
    return fx_g  # 返回FX图对象


def draw_graph_compile(name):
    return make_boxed_compiler(partial(_draw_graph_compile, name=name))


@make_boxed_compiler
def nop(fx_g: fx.GraphModule, _) -> Callable:
    """
    Returns the :attr:`fx_g` Fx graph module as it is. This is a no-op compiler
    and can be used to check accuracy.
    """
    return fx_g  # 返回未经任何操作的FX图模块
    .. warning::
        This API is experimental and likely to change.

    """
    返回 fx_g 变量。
    这里的文档字符串可能是一个占位符或者忘记删除的部分。
class DebugInterpreter(fx.Interpreter):
    # 定义一个调试解释器，继承自fx.Interpreter类
    def run(self, *args):
        # 运行函数，绑定符号映射到模块中的参数
        self.symbol_mapping = bind_symbols(self.module, *args)
        # 调用父类的运行方法，传递参数
        super().run(*args)

    def run_node(self, n):
        # 运行单个节点的函数
        def subst_symint(ni):
            # 替换符号整数
            if not isinstance(ni, SymInt):
                return ni
            # 扩展并替换节点表达式中的符号映射
            r = sympy.expand(ni.node.expr.xreplace(self.symbol_mapping))
            # 断言结果为数字
            assert r.is_number, r
            return int(r)

        def subst_symint_tuple(nis):
            # 替换符号整数元组
            return tuple(subst_symint(ni) for ni in nis)

        def check_significant_strides(a, b):
            # 检查重要的步长
            if subst_symint(a.numel()) > 0:
                for idx in range(a.ndim):
                    if (
                        subst_symint(a.stride(idx)) != b.stride(idx)
                        and subst_symint(a.size(idx)) > 1
                    ):
                        return False
            return True

        def check(nv, rv, desc):
            # 检查两个值是否相等
            assert callable(desc)
            assert nv.dtype == rv.dtype, f"{desc()}: {nv.dtype} != {rv.dtype}"
            assert (
                subst_symint_tuple(nv.size()) == rv.size()
            ), f"{desc()}: {nv.size()} aka {subst_symint_tuple(nv.size())} != {rv.size()}"
            same_strides = check_significant_strides(nv, rv)
            assert (
                same_strides
            ), f"{desc()}: {nv.stride()} aka {subst_symint_tuple(nv.stride())} != {rv.stride()}"

        r = super().run_node(n)
        # 如果节点具有“val”元数据
        if "val" in n.meta:
            # 展平节点的值和规范
            n_vals, n_spec = pytree.tree_flatten(n.meta["val"])
            r_vals, r_spec = pytree.tree_flatten(r)
            # TODO: 存在一些问题，即记录运算符返回元组/列表，但实际上版本可能返回列表/元组。需要进一步了解实际情况。
            # assert n_spec == r_spec, f"{n_spec} != {r_spec}"
            assert len(n_vals) == len(r_vals), f"{len(n_vals)} != {len(r_vals)}"
            for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
                if not isinstance(rv, torch.Tensor):
                    continue
                check(nv, rv, lambda: f"output {i} where {self.symbol_mapping}")
        return r


@make_boxed_compiler
def debug_nop(fx_g: fx.GraphModule, _) -> Callable:
    """
    返回一个（慢速）解释器，用于 FX 图模块，同时检查各种调试属性（例如，跟踪步长是否与实际步长匹配）。
    """
    return DebugInterpreter(fx_g).run


@make_boxed_compiler
def simple_ts_compile(fx_g, _):
    # 剥离重载
    strip_overloads(fx_g)
    # 转换为 Torch 脚本
    f = torch.jit.script(fx_g)
    # 冻结和评估脚本
    f = torch.jit.freeze(f.eval())
    return f


def nnc_jit(f):
    # 将函数编译为 AOT 函数，使用简单的 Torch 脚本编译器
    return aot_function(f, simple_ts_compile)


aten = torch.ops.aten
default_decompositions = {
    aten.detach,
    aten.gelu_backward,
    aten.leaky_relu_backward,
    # 访问 PyTorch ATen 模块的 sigmoid_backward 函数
    aten.sigmoid_backward,
    # 访问 PyTorch ATen 模块的 threshold_backward 函数
    aten.threshold_backward,
    # 访问 PyTorch ATen 模块的 hardtanh_backward 函数
    aten.hardtanh_backward,
    # 访问 PyTorch ATen 模块的 hardsigmoid_backward 函数
    aten.hardsigmoid_backward,
    # 访问 PyTorch ATen 模块的 hardswish_backward 函数
    aten.hardswish_backward,
    # 访问 PyTorch ATen 模块的 tanh_backward 函数
    aten.tanh_backward,
    # 访问 PyTorch ATen 模块的 silu_backward 函数
    aten.silu_backward,
    # 访问 PyTorch ATen 模块的 elu_backward 函数
    aten.elu_backward,
    # 访问 PyTorch ATen 模块的 cudnn_batch_norm 函数
    aten.cudnn_batch_norm,
    # 访问 PyTorch ATen 模块的 cudnn_batch_norm_backward 函数
    aten.cudnn_batch_norm_backward,
    # 访问 PyTorch ATen 模块的 masked_fill 函数，用于标量参数
    aten.masked_fill.Scalar,
    # 访问 PyTorch ATen 模块的 masked_fill 函数，用于张量参数
    aten.masked_fill.Tensor,
    # 访问 PyTorch ATen 模块的 elu 函数
    aten.elu,
    # 访问 PyTorch ATen 模块的 leaky_relu 函数
    aten.leaky_relu,
    # 访问 PyTorch ATen 模块的 hardtanh 函数
    aten.hardtanh,
    # 访问 PyTorch ATen 模块的 hardswish 函数
    aten.hardswish,
    # 访问 PyTorch ATen 模块的 hardsigmoid 函数
    aten.hardsigmoid,
    # 访问 PyTorch ATen 模块的 conj_physical 函数
    aten.conj_physical,
    # 访问 PyTorch ATen 模块的 is_same_size 函数
    aten.is_same_size,
}

# 将默认的分解方式获取并存储在 default_decompositions 变量中
default_decompositions = get_decompositions(default_decompositions)

# 使用装饰器 make_boxed_compiler 包装 print_compile 函数
@make_boxed_compiler
def print_compile(fx_g, _):
    # 打印 fx_g 对象的代码表示
    print(fx_g.code)
    return fx_g

# 定义 memory_efficient_fusion 函数，用于内存高效融合操作
def memory_efficient_fusion(
    fn: Union[Callable, nn.Module],
    **kwargs,
):
    """
    Wrapper function over :func:`aot_function` and :func:`aot_module` to perform
    memory efficient fusion. It uses the
    :func:`min_cut_rematerialization_partition` partitioner to perform efficient
    recomputation. It uses NVFuser to compile the generated forward and backward
    graphs.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Union[Callable, nn.Module]): A Python function or a ``nn.Module``
            that takes one ore more arguments. Must return one or more Tensors.
        **kwargs: Any other overrides you want to make to the settings

    Returns:
        Returns a ``Callable``  or ``nn.Module`` that retains the eager behavior
        of the original :attr:`fn`, but whose forward and backward graphs have
        gone through recomputation optimizations, and the graphs have been
        compiled with nvfuser.

    """
    # 配置参数字典，设置前向和后向编译器为 ts_compile，分区函数为 min_cut_rematerialization_partition，分解方式为默认设置
    config = {
        "fw_compiler": ts_compile,
        "bw_compiler": ts_compile,
        "partition_fn": min_cut_rematerialization_partition,
        "decompositions": default_decompositions,
    }
    config.update(kwargs)
    # 如果 fn 是 torch.nn.Module 类型，则调用 aot_module 函数，传入配置参数进行 Ahead-of-Time 编译
    if isinstance(fn, torch.nn.Module):
        return aot_module(fn, **config)
    else:
        # 否则调用 aot_function 函数，同样传入配置参数进行 Ahead-of-Time 编译
        return aot_function(fn, **config)


# 定义 debug_compile 函数，用于调试编译过程
def debug_compile(fx_g, inps):
    # 将 fx_g 对象保存到名为 "foo" 的文件夹中
    fx_g.to_folder("foo")
    # 打印调试信息，包括输入张量的形状和数据类型
    print(
        f"""
##############################################################
# To minimize FX graph, copy and paste the below and run it  #
##############################################################

import torch
import torch.fx as fx
from functorch.compile import minifier, check_nvfuser_subprocess, check_nvfuser_correctness_subprocess

inps = {[(i.shape, i.dtype) for i in inps]}
inps = [torch.ones(shape, dtype=dtype, device='cuda') for (shape, dtype) in inps]
from foo import FxModule
mod = FxModule().cuda()

with torch.jit.fuser("fuser2"):
  # check_nvfuser_subprocess can be replaced with check_nvfuser_correctness_subprocess
  minifier(fx.symbolic_trace(mod), inps, check_nvfuser_subprocess)
"""
    )
    # 导入 FxModule，并在 CUDA 上执行模型推断
    from foo import FxModule
    FxModule().cuda()(*inps)

    # 调用 ts_compile 函数编译 fx_g 对象
    return ts_compile(fx_g, inps)

# 初始化全局变量 graph_index
graph_index = 0

# 定义 get_inputs 函数，用于获取给定输入元数据的随机输入数据
def get_inputs(input_data_path):
    """
    Return a random input for the given inputs meta generated from _save_fx_default.
    """
    inputs = []
    # 使用二进制模式打开指定路径的数据文件
    with open(input_data_path, "rb") as f:
        # 从文件中加载 pickle 格式的元数据并存储在 inputs_meta 变量中
        inputs_meta = pickle.load(f)
        # 初始化空列表 inputs 用于存储处理后的输入数据
        inputs = []
        # 遍历 inputs_meta 中的每个元素
        for meta in inputs_meta:
            # 检查 meta 元素的长度，以确定其类型
            if len(meta) == 1:
                # 如果 meta 的长度为 1，将其视为类型，并使用随机数生成一个该类型的输入
                type = meta
                input = type(random.rand())
            else:
                # 如果 meta 的长度大于 1，解包元组并获取类型、形状、步幅、数据类型和设备信息
                type, shape, stride, dtype, device = meta
                # 检查数据类型是否为整数、布尔值或浮点数的一种
                if dtype in {
                    torch.int,
                    torch.int32,
                    torch.int64,
                    torch.bool,
                    torch.int,
                    torch.uint8,
                    int,
                    float,
                }:
                    # 如果数据类型符合条件，生成一个指定形状和数据类型的随机整数张量
                    input = torch.randint(0, 1, shape, dtype=dtype, device=device)
                else:
                    # 否则，生成一个指定形状和数据类型的随机张量
                    input = torch.rand(shape, dtype=dtype, device=device)
            # 将生成的输入数据添加到 inputs 列表中
            inputs.append(input)
    # 返回处理后的输入数据列表
    return inputs
def _save_fx_default(current_name, folder_name, dump_example_input, gm, example_inputs):
    """
    The forward, backward, and joint computation graph will be stored in
    {folder_name}/{current_name}/{current_name}_forward_{graph_index},
    {folder_name}/{current_name}/{current_name}_backward_{graph_index}, and
    {folder_name}/{current_name}/{current_name}_joint_{graph_index} respectively.
    The input shape of the graphs will be stored in the .input files.
    These files can be loaded with pickle,
    and is a list of format (type, shape, stride, dtype, device).
    In the case of type = int or float, it is just (type,).
    For joint graph input, it is a nested list [[],[]]
    where the two inner lists have the same format.
    If dump_example_input is True, example_inputs will be stored in .pt file.
    Since each function might produce multiple graphs,
    the graph_index is used to distinguish difference graphs
    """
    from functorch.compile import aot_module_simplified

    # Define a function to recursively gather input metadata
    def get_input_meta(args):
        input_meta = []
        if len(args) > 0 and isinstance(args[0], tuple):  # Check if args represent joint input
            # If joint input, recursively gather metadata from each part
            input_meta += get_input_meta(args[0])
            input_meta += get_input_meta(args[1])
            return input_meta
        for arg in args:
            # Append metadata tuple for each argument based on its type
            if type(arg) == int or type(arg) == float:
                input_meta.append((type(arg),))
            else:
                input_meta.append(
                    (type(arg), arg.shape, arg.stride(), arg.dtype, arg.device)
                )
        return input_meta

    # Define a helper function to save computation graphs
    def graph_saver_helper(gm_to_save, args, type_name):
        global graph_index  # Use the global variable graph_index
        # Check if the graph to save has nodes; log a warning if not
        if len(gm_to_save.graph.nodes) == 0:
            log.log(
                logging.WARNING,
                "No nodes in graph {%s}_{%s}_{%s}.",
                current_name,
                type_name,
                graph_index,
            )
            return
        
        # Deepcopy the graph module and recompile it
        gm = copy.deepcopy(gm_to_save)
        gm.graph.set_codegen(torch.fx.graph.CodeGen())  # Remove codegen from the graph
        gm.recompile()

        # Gather input metadata
        input_meta = get_input_meta(args)

        # Create directories if they don't exist to store graph files
        os.makedirs(f"{folder_name}/{current_name}", exist_ok=True)
        # Save the graph to a folder with appropriate naming convention
        gm.to_folder(
            f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}"
        )
        # Save input metadata to a .input file using pickle
        pickle.dump(
            input_meta,
            open(
                f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.input",  # noqa: B950
                "wb",
            ),
        )  # noqa: E501
        # Optionally save example inputs to a .pt file if dump_example_input is True
        if dump_example_input:
            torch.save(
                args,
                f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.pt",  # noqa: B950
            )  # noqa: E501

    # Define a function to save the forward computation graph
    def graph_saver_forward(gm, fw_args):
        # Call the helper function with appropriate type_name for forward graphs
        graph_saver_helper(gm, fw_args, "forward")
        return gm  # Return the modified graph module
    # 定义一个函数，用于保存反向图的操作
    def graph_saver_backward(gm, bw_args):
        # 调用辅助函数，将反向图模型gm和相关参数bw_args保存为"backward"类型的图
        graph_saver_helper(gm, bw_args, "backward")
        # 增加全局变量graph_index的值，用于标识图的索引
        global graph_index
        graph_index += 1
        # 返回保存后的图模型gm
        return gm
    
    # 定义一个函数，用于保存联合图的操作
    def graph_saver_joint(gm, joint_args):
        # 调用辅助函数，将联合图模型gm和相关参数joint_args保存为"joint"类型的图
        graph_saver_helper(gm, joint_args, "joint")
        # 返回对图模型gm进行默认分区操作的结果
        return default_partition(gm, joint_args)
    
    # 返回简化后的AOT模块，包括输入示例、前向编译器、后向编译器、分区函数和默认分解方式
    return aot_module_simplified(
        gm,  # 图模型
        example_inputs,  # 输入示例
        fw_compiler=graph_saver_forward,  # 前向编译器函数
        bw_compiler=graph_saver_backward,  # 后向编译器函数
        partition_fn=graph_saver_joint,  # 分区函数
        decompositions=default_decompositions,  # 默认分解方式
    )
# WARNING: This isn't tested anywhere!!
# 定义一个函数，用于将前向、反向和联合计算图进行转储
# Example Usage:
# save_fx_func = graph_dumper_aot(current_name, folder_name, dump_example_input=False)
# optimize_ctx = torchdynamo.optimize(
#     save_fx_func
# )
# 使用全局变量 graph_index 进行索引初始化
global graph_index = 0
# 返回一个函数_partial对象，将_save_fx_default函数与参数current_name, folder_name, dump_example_input绑定
return partial(_save_fx_default, current_name, folder_name, dump_example_input)
```