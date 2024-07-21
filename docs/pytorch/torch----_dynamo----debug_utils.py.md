# `.\pytorch\torch\_dynamo\debug_utils.py`

```py
# mypy: allow-untyped-defs
# mypy: disable-error-code="method-assign"

# 导入必要的模块和库
import copy                      # 导入深拷贝模块
import functools                 # 导入函数工具模块
import getpass                   # 导入获取用户名模块
import inspect                   # 导入检查模块
import itertools                 # 导入迭代工具模块
import logging                   # 导入日志记录模块
import os                        # 导入操作系统模块
import re                        # 导入正则表达式模块
import subprocess                # 导入子进程管理模块
import tempfile                  # 导入临时文件模块
import textwrap                  # 导入文本包装模块
from collections import Counter  # 从集合模块导入计数器类
from importlib import import_module  # 从导入库模块导入动态导入函数
from typing import Any, Callable, Dict, List, Optional, TypeVar  # 导入类型提示相关模块

import torch                     # 导入PyTorch深度学习库
import torch._prims_common as utils  # 导入PyTorch基本公共库
import torch._subclasses.meta_utils  # 导入PyTorch子类元工具库
from torch import Tensor          # 从PyTorch导入张量类

from torch._dynamo.testing import rand_strided  # 从PyTorch导入动态测试随机步长模块
from torch._prims_common import is_float_dtype  # 从PyTorch基本公共库导入浮点数数据类型检查函数
from torch.multiprocessing.reductions import StorageWeakRef  # 从PyTorch多进程模块导入存储弱引用类
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter  # 从PyTorch导入内容存储读写模块

from . import config              # 从当前目录导入配置模块
from .utils import clone_inputs, get_debug_dir  # 从当前目录的工具模块导入克隆输入函数和获取调试目录函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

T = TypeVar("T")  # 定义一个类型变量T


inductor_config = import_module("torch._inductor.config")  # 动态导入torch._inductor.config模块
use_buck = inductor_config.is_fbcode()  # 检查是否使用了Buck构建系统

if use_buck:
    import libfb.py.build_info  # 如果使用Buck，则导入libfb.py.build_info模块


extra_deps = []  # 初始化额外的依赖列表为空
extra_imports = ""  # 初始化额外导入的字符串为空
if use_buck:
    # 如果使用Buck，则配置额外依赖和导入字符串
    extra_deps = [
        "//caffe2/torch/fb/sparsenn:sparsenn_operators_gpu",
        "//caffe2/torch/fb/sparsenn:sparsenn_operators",
        "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu",
        "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops",
    ]
    cur_target = libfb.py.build_info.BuildInfo.get_build_rule().replace("fbcode:", "//")  # 获取当前目标构建规则
    extra_imports = "\n".join([f'torch.ops.load_library("{x}")' for x in extra_deps])  # 构建额外导入的字符串


BUCK_CMD_PREFIX = ["buck2", "run", "@mode/dev-nosan"]  # Buck命令前缀


class BuckTargetWriter:
    def __init__(self, filename):
        self.subdir, self.py_file = os.path.split(os.path.abspath(filename))  # 获取文件名和路径
        self.target = self.py_file.replace(".py", "")  # 去掉.py后缀作为目标名

        # 从fbcode获取主模块路径
        self.path = f'{self.subdir.replace("/", ".")}.{self.target}'
        self.path = self.path[self.path.find("fbcode.") :]
        self.path = self.path[7:]

        # 获取命令行路径
        tmp = self.subdir
        tmp = tmp[tmp.find("fbcode/") :][7:]
        self.cmd_line_path = f"//{tmp}:{self.target}"

    def build(self):
        extra_cpp_deps = "\n".join([f'        "{x}",' for x in extra_deps])  # 构建额外的C++依赖字符串
        return textwrap.dedent(
            f"""
            load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")

            python_binary(
                name="{self.target}",
                srcs = ["{self.py_file}"],
                compile = False,
                deps = [
                    "//caffe2:torch",
                    "//caffe2/functorch:functorch",
                    "//triton:triton",
                    "{cur_target}",
                ],
                cpp_deps = [
    {extra_cpp_deps}
                ],
                main_module = "{self.path}",
                par_style = "xar",
            )
            """
        )
    # 定义一个方法 `write`，接受一个布尔参数 `print_msg` 默认为 True
    def write(self, print_msg=True):
        # 构建目标文件路径，在当前对象的子目录下命名为 "TARGETS"
        target_file = os.path.join(self.subdir, "TARGETS")
        # 打开目标文件，以写入模式打开文件描述符 `fd`
        with open(target_file, "w") as fd:
            # 将 `self.build()` 方法的返回值写入文件
            fd.write(self.build())
        
        # 执行以下命令生成 `cmd_split` 列表变量，其中包含了 `BUCK_CMD_PREFIX` 和 `self.cmd_line_path`
        cmd_split = BUCK_CMD_PREFIX + [self.cmd_line_path]
        
        # 如果 `print_msg` 为 True，则记录警告信息，说明如何复现错误的示例命令
        if print_msg:
            log.warning(
                "Found an example that reproduces the error. Run this cmd to repro - %s",
                " ".join(cmd_split),
            )
        
        # 返回 `cmd_split` 列表，其中包含了构建好的命令
        return cmd_split
# 定义一个类，用于将神经网络模块转换为字符串表示
class NNModuleToString:
    # 安全的模块类型列表，可以被安全地转换为字符串表示
    safe_reprs = [
        torch.nn.Linear,             # 线性层
        torch.nn.Conv1d,             # 一维卷积层
        torch.nn.Conv2d,             # 二维卷积层
        torch.nn.Conv3d,             # 三维卷积层
        torch.nn.BatchNorm1d,        # 一维批标准化层
        torch.nn.BatchNorm2d,        # 二维批标准化层
        torch.nn.BatchNorm3d,        # 三维批标准化层
        torch.nn.LayerNorm,          # 归一化层
        torch.nn.Dropout,            # 随机失活层
        torch.nn.Softmax,            # Softmax 激活函数
        torch.nn.ReLU,               # ReLU 激活函数
        torch.nn.GELU,               # GELU 激活函数
        torch.nn.Identity,           # 恒等映射层
        torch.nn.MaxPool2d,          # 二维最大池化层
        torch.nn.Embedding,          # 嵌入层
        torch.nn.Tanh,               # Tanh 激活函数
        torch.nn.ConvTranspose1d,    # 一维转置卷积层
        torch.nn.GLU,                # GLU 激活函数
        torch.nn.LSTM,               # LSTM 层
        torch.nn.Flatten,            # 展平层
        torch.nn.AdaptiveAvgPool2d,  # 自适应平均池化层
    ]

    @staticmethod
    # 判断是否可以将神经网络模块 gm 转换为字符串表示
    def can_convert_to_string(gm):
        # 无法转换的模块集合
        cant_convert = set()
        # 遍历 gm 的每个子模块
        for _, module in gm.named_children():
            # 如果模块类型不在安全的模块类型列表中
            if type(module) not in NNModuleToString.safe_reprs:
                cant_convert.add(module)

        # 如果存在无法转换的模块，记录警告日志
        if len(cant_convert) > 0:
            log.warning("We have not tested reprs of some modules - %s", cant_convert)
        
        # TODO - 假设所有模块都可以安全地转换为字符串表示。检查此假设是否正确。
        return True
    # 定义一个函数convert，接受一个torch.nn.Module类型的参数gm
    def convert(gm):
        # 导入_addindent函数，用于增加缩进
        from torch.nn.modules.module import _addindent
    
        # 创建4个空格的tab字符串
        tab = " " * 4
    
        # 使用textwrap.dedent移除model_str字符串开头的额外缩进
        # 创建一个模板字符串model_str，引入torch.nn下的所有模块，并定义一个Repro类继承自torch.nn.Module
        model_str = textwrap.dedent(
            """
            from torch.nn import *
            class Repro(torch.nn.Module):
                def __init__(self):
                    super().__init__()
            """
        )
    
        # 遍历gm模型的每个子模块，获取其模块名和模块对象
        for module_name, module in gm.named_children():
            # 获取模块的字符串表示
            module_str = f"{module.__repr__()}"
            # 如果模块的参数在GPU上，则将模块移到CUDA设备上
            if next(module.parameters(), None) is not None and next(module.parameters()).is_cuda:
                module_str = f"{module_str}.cuda()"
            # 将当前模块的字符串表示添加到model_str中，形成模块的初始化语句
            model_str += f"{tab*2}self.{module_name} = {module_str}\n"
    
        # 遍历gm模型的每个缓冲区，获取其缓冲区名和缓冲区对象
        for buffer_name, buffer in gm._buffers.items():
            # 跳过空缓冲区
            if buffer is None:
                continue
            # 如果缓冲区的元素数量小于等于MAX_CONSTANT_NUMEL_INLINE，则序列化整个数据
            if buffer.numel() <= MAX_CONSTANT_NUMEL_INLINE:
                # 导入PRINT_OPTS，并确保其阈值大于等于MAX_CONSTANT_NUMEL_INLINE
                from torch._tensor_str import PRINT_OPTS
                assert PRINT_OPTS.threshold >= MAX_CONSTANT_NUMEL_INLINE
                # 获取缓冲区的字符串表示
                tensor_str = repr(buffer)
            # 如果缓冲区是浮点类型，则创建一个随机数据张量
            elif torch.is_floating_point(buffer):
                tensor_str = f"torch.randn({list(buffer.shape)}, dtype={buffer.dtype})"
            # 否则创建一个随机整数张量
            else:
                tensor_str = f"torch.randint(1, size={list(buffer.shape)}, dtype={buffer.dtype})"
            # 如果缓冲区在CUDA上，则将其移到CUDA设备上
            if buffer.is_cuda:
                tensor_str = f"{tensor_str}.cuda()"
            # 将当前缓冲区的字符串表示添加到model_str中，形成缓冲区注册语句
            model_str += f"{tab*2}self.register_buffer('{buffer_name}', {tensor_str})\n"
    
        # 遍历gm模型的每个参数，获取其参数名和参数对象
        for param_name, param in gm._parameters.items():
            # 跳过空参数
            if param is None:
                continue
            # 根据参数是否在CUDA上创建参数张量的字符串表示
            maybe_device = ", device='cuda'" if param.is_cuda else ""
            tensor_str = f"torch.nn.Parameter(torch.randn({list(param.shape)}, dtype={param.dtype}{maybe_device}))"
            # 将当前参数的字符串表示添加到model_str中，形成参数初始化语句
            model_str += f"{tab*2}self.{param_name} = {tensor_str}\n"
    
        # 添加gm.code的缩进后的字符串表示到model_str中
        model_str += f"{_addindent(gm.code, 4)}\n"
        # 返回形成的模型字符串表示
        return model_str
# 使用 functools.lru_cache(None) 装饰器来缓存 _cuda_system_info_comment 函数的结果，以提高性能
@functools.lru_cache(None)  # subprocess is expensive
def _cuda_system_info_comment():
    # 检查是否有可用的 CUDA 设备，如果没有则返回信息表示无 GPU 信息被收集
    if not torch.cuda.is_available():
        return "# torch.cuda.is_available()==False, no GPU info collected\n"

    # 初始化 CUDA 相关信息字符串
    model_str = "# CUDA Info: \n"
    try:
        # 使用 subprocess 调用 nvcc 命令获取 CUDA 版本信息
        cuda_version_out = subprocess.check_output(["nvcc", "--version"])
        cuda_version_lines = cuda_version_out.decode().split("\n")
        # 将 nvcc 输出的每行信息格式化为注释并添加到 model_str 中
        comment = "".join([f"# {s} \n" for s in cuda_version_lines if s not in [""]])
        model_str += f"{comment}\n"
    except (FileNotFoundError, subprocess.CalledProcessError):
        # 如果 nvcc 命令未找到或出现其他错误，记录信息表示 nvcc 未找到
        model_str += "# nvcc not found\n"

    # 使用 Counter 统计每个 CUDA 设备的名称和数量
    gpu_names = Counter(
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    )

    # 添加 GPU 硬件信息的标题
    model_str += "# GPU Hardware Info: \n"
    # 遍历 GPU 名称及其数量，并将其格式化为注释添加到 model_str 中
    for name, count in gpu_names.items():
        model_str += f"# {name} : {count} \n"
    model_str += "\n"
    # 返回构建好的 CUDA 系统信息字符串
    return model_str


# 生成配置信息字符串，根据 stable_output 参数决定是否输出稳定输出信息
def generate_config_string(*, stable_output=False):
    import torch._functorch.config
    import torch._inductor.config

    if stable_output:
        # 如果 stable_output 为 True，则返回一条指示配置被省略的注释
        return "# config omitted due to stable_output=True"

    # 否则，收集各个模块的配置信息并返回格式化后的配置字符串
    experimental_config = torch.fx.experimental._config.codegen_config()  # type: ignore[attr-defined]
    return f"""\
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
{torch._dynamo.config.codegen_config()}
{torch._inductor.config.codegen_config()}
{torch._functorch.config.codegen_config()}
{experimental_config}
"""


# 返回用于 minifier 的路径信息，连接 minifier 目录和其启动脚本名
def get_minifier_repro_path():
    return os.path.join(minifier_dir(), "minifier_launcher.py")


# 辅助函数，将内容写入到 minifier 的复现路径中
def helper_for_dump_minify(contents):
    # 获取 minifier 复现路径
    minified_repro_path = get_minifier_repro_path()
    # 使用日志警告记录写入的 minified 复现路径
    log.warning("Writing minified repro to:\n%s", minified_repro_path)

    # 如果 use_buck 为 True，使用 BuckTargetWriter 将路径写入到 Buck 目标
    if use_buck:
        BuckTargetWriter(minified_repro_path).write()
    
    # 尝试打开复现路径并写入内容
    try:
        with open(minified_repro_path, "w") as fd:
            fd.write(contents)

    # 处理可能的操作系统错误
    except OSError as e:
        log.exception("")
        raise NotImplementedError(f"Could not write to {minified_repro_path}") from e


# 自定义异常类，用于指示精度错误
class AccuracyError(Exception):
    pass


# 克隆输入并保持梯度信息的函数
def clone_inputs_retaining_gradness(example_inputs):
    """
    This clone inputs is different from utils clone_input. In case of minifier,
    all the tensors are leaf tensors while creating a new graph. So, we set the
    requires_grad field w/o checking the leafness of the tensor.
    """
    # 克隆输入，同时保留其梯度信息
    cloned_inputs = clone_inputs(example_inputs)
    # 遍历克隆后的输入，如果是 Tensor 对象，则根据原始输入的 requires_grad 设置 requires_grad
    for idx in range(len(example_inputs)):
        if isinstance(cloned_inputs[idx], torch.Tensor):
            cloned_inputs[idx].requires_grad_(example_inputs[idx].requires_grad)
    return cloned_inputs


# 运行前向和可能的后向传播的函数
def run_fwd_maybe_bwd(gm, args, only_fwd=False, disable_clone=False):
    """
    Runs a forward and possibly backward iteration for a given mod and args.

    When disable_clone is True, we will use args as-is without cloning.
    This is higher fidelity but we may destroy the args in the process.
    """
    # 导入测试模块中的 collect_results, reduce_to_scalar_loss, requires_bwd_pass 函数
    from .testing import collect_results, reduce_to_scalar_loss, requires_bwd_pass

    # 深度复制 gm 对象并将其赋值给 gm 变量
    gm = copy.deepcopy(gm)

    # 如果不禁用克隆操作
    if not disable_clone:
        # 使用保留梯度信息的方式克隆输入参数 args
        args = clone_inputs_retaining_gradness(args)

    # 如果 gm 对象具有 "zero_grad" 方法，则调用其并清空梯度信息
    if hasattr(gm, "zero_grad"):
        gm.zero_grad(True)

    # 如果 gm 对象具有 "_boxed_call" 属性，说明 TorchInductor 返回的可调用对象期望接收列表参数
    # 因此可能需要采用装箱的调用约定
    out = gm(args) if hasattr(gm, "_boxed_call") else gm(*args)

    # 如果仅需要前向传播结果，则直接返回 out
    if only_fwd:
        return out

    # 如果 out 需要进行反向传播
    if requires_bwd_pass(out):
        # 将输出 out 转化为标量损失值
        loss = reduce_to_scalar_loss(out)
        # 执行反向传播
        loss.backward()

    # 收集运行结果并返回
    return collect_results(gm, out, None, args)
# 检查两个模型是否具有相同的精度。
def same_two_models(
    gm,  # 第一个模型
    opt_gm,  # 第二个优化后的模型
    example_inputs,  # 示例输入
    only_fwd=False,  # 是否仅执行前向传播，默认为False
    *,  # 后续参数为关键字参数
    require_fp64=False,  # 如果为True，当无法计算fp64参考时会引发错误
    ignore_non_fp=False,  # 如果为True，不比较非浮点数输出
):
    """
    Check two models have same accuracy.

    require_fp64: if True, raise an error if we unable to calculate the fp64 reference
    ignore_non_fp: if True, do not compare outputs which are not floating point.  This
        is mostly useful for the minifier (which wants to avoid quantizing floating point
        error into integer/boolean error)
    """
    from .utils import same  # 导入同一性比较函数

    ref = run_fwd_maybe_bwd(gm, example_inputs, only_fwd)  # 运行模型的前向或前向后向传播

    fp64_ref = None
    if config.same_two_models_use_fp64:  # 如果配置指定使用fp64
        try:
            fp64_model, fp64_examples = cast_to_fp64(
                copy.deepcopy(gm), clone_inputs_retaining_gradness(example_inputs)
            )  # 将模型和输入复制并转换为fp64精度
            fp64_ref = run_fwd_maybe_bwd(fp64_model, fp64_examples, only_fwd)  # 获取fp64精度的输出
        except Exception:
            if require_fp64:
                raise RuntimeError("Could not generate fp64 outputs")  # 如果需要fp64但无法生成，则引发运行时错误
            log.warning("Could not generate fp64 outputs")  # 记录警告信息，无法生成fp64输出

    try:
        res = run_fwd_maybe_bwd(opt_gm, example_inputs, only_fwd)  # 运行优化后模型的前向或前向后向传播
    except Exception as e:
        # 如果出现异常，通常是优化后的图形问题，记录异常并返回True（跳过此图形）
        log.exception(
            "While minifying the program in accuracy minification mode, "
            "ran into a runtime exception which is likely an unrelated issue."
            " Skipping this graph."
        )
        return True

    # 使用same函数比较ref和res的输出结果是否相同，包括fp64_ref的比较
    passing = same(
        ref,
        res,
        fp64_ref,
        tol=config.repro_tolerance,  # 允许的误差范围
        equal_nan=True,  # 是否认为NaN相等
        ignore_non_fp=ignore_non_fp,  # 是否忽略非浮点数输出
    )
    return passing  # 返回比较结果的布尔值


def cast_dtype_args_to_fp64(model):
    # 将模型中的dtype参数转换为fp64
    for node in model.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.prims.convert_element_type.default
        ):
            assert len(node.args) == 2
            if is_float_dtype(node.args[1]) and node.args[1] != torch.float64:
                node.args = (node.args[0], torch.float64)  # 将第二个参数转换为fp64
        if node.op == "call_function":
            dtype = node.kwargs.get("dtype")
            if dtype is not None and is_float_dtype(dtype):
                new_kwargs = dict(node.kwargs)
                new_kwargs["dtype"] = torch.float64  # 将dtype参数转换为fp64
                node.kwargs = new_kwargs

    model.graph.lint()  # 检查模型图形的一致性
    model.recompile()  # 重新编译模型
    return model  # 返回转换后的模型


def cast_to(dtype, model, inputs):
    from torch.utils._pytree import tree_map  # 导入树映射工具

    model = model.to(dtype)  # 将模型转换为指定的数据类型
    if dtype == torch.float64:
        # 如果转换为fp64用于精度比较，则需要替换图中嵌入的dtype参数为fp64
        model = cast_dtype_args_to_fp64(model)

    # 使用树映射函数，将输入中的张量转换为指定的数据类型
    inputs = tree_map(
        lambda x: x.to(dtype)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x,
        inputs,
    )
    )
    # 返回模型和输入数据的元组
    return model, inputs
# 使用 torch.float64 将模型和输入数据类型转换为 64 位浮点数
def cast_to_fp64(model, inputs):
    return cast_to(torch.float64, model, inputs)


# 对比编译后的图形与原始图形的精度，检查是否有精度损失
# 如果编译后的图形与原始图形不同，则返回 True，否则返回 False
# 如果出现异常，则记录异常信息并返回 False
def backend_accuracy_fails(
    gm,
    example_inputs,
    compiler_fn,
    only_fwd=False,
    *,
    require_fp64=False,
    ignore_non_fp=False,
):
    try:
        # 深拷贝原始图形并且保留输入梯度信息后，编译图形
        compiled_gm = compiler_fn(
            copy.deepcopy(gm), clone_inputs_retaining_gradness(example_inputs)
        )
        # 检查编译后的图形与原始图形在前向传播上是否一致
        return not same_two_models(
            gm,
            compiled_gm,
            example_inputs,
            only_fwd,
            require_fp64=require_fp64,
            ignore_non_fp=ignore_non_fp,
        )
    except Exception as e:
        # 这表示精度缩小模式下图形有问题/显示出了不同的问题。
        # 由于我们在此检查精度，因此记录异常信息并返回 False。
        log.exception(
            "While minifying the program in accuracy minification mode, "
            "ran into a runtime exception which is likely an unrelated issue."
            " Skipping this graph"
        )
        return False


# 辅助函数，根据形状计算默认的张量步长值
def _stride_or_default(
    stride: Optional["torch._prims_common.StrideType"],
    *,
    shape: "torch._prims_common.ShapeType",
) -> "torch._prims_common.StrideType":
    return stride if stride is not None else utils.make_contiguous_strides_for(shape)


# 创建默认值生成器函数，根据给定的默认值生成一个返回函数
def _mk_defaulter(d: T) -> Callable[[Optional[T]], T]:
    return lambda x: x if x is not None else d


# 返回默认的数据类型 torch.float32，如果未指定则返回该默认值
_dtype_or_default = _mk_defaulter(torch.float32)
# 返回默认的设备类型 torch.device("cpu")，如果未指定则返回该默认值
_device_or_default = _mk_defaulter(torch.device("cpu"))
# 返回默认的存储偏移量 0，如果未指定则返回该默认值
_storage_offset_or_default = _mk_defaulter(0)
# 返回默认的 requires_grad 属性 False，如果未指定则返回该默认值
_requires_grad_or_default = _mk_defaulter(False)
# 返回默认的 is_leaf 属性 False，如果未指定则返回该默认值
_is_leaf_or_default = _mk_defaulter(False)


# 用于创建无操作的输入读取器类
class NopInputReader:
    def __init__(self):
        self.total = 0

    # 存储方法，用于增加总数计数
    def storage(self, storage_hash, nbytes, *, device=None, dtype_hint=None):
        self.total += 1

    # 返回空操作的张量
    def tensor(self, *args, **kwargs):
        pass

    # 返回空操作的符号整数
    def symint(self, *args, **kwargs):
        pass


# TODO: 支持将整个重现环境打包成 ZIP 文件，以便于传输
# 输入读取器类，用于读取输入数据并可选保存为 ZIP 文件
class InputReader:
    def __init__(self, save_dir=None, *, pbar=None):
        # 如果 save_dir 为 None，则生成随机数据，支持无需包含真实数据的情况
        if save_dir is None:
            log.warning("no save_dir specified, will generate random data")
        # 根据指定的 save_dir 创建内容存储读取器，如果未指定则为 None
        self.store = ContentStoreReader(save_dir) if save_dir is not None else None
        # 初始化参数列表为空
        self.args = []
        # 进度条对象，用于显示读取进度
        self.pbar = pbar
    # 如果进度条对象存在，则更新进度条
    def storage(self, storage_hash, nbytes, *, device=None, dtype_hint=None):
        if self.pbar is not None:
            self.pbar.update(1)
        
        # 根据传入的参数获取设备对象
        device = _device_or_default(device)
        
        # 根据传入的参数获取数据类型提示对象
        dtype_hint = _dtype_or_default(dtype_hint)
        
        # 如果存储对象和存储哈希值都存在，则尝试读取存储对象
        if self.store is not None and storage_hash is not None:
            try:
                storage = self.store.read_storage(storage_hash)
            except FileNotFoundError:
                pass
            else:
                # 如果指定的设备与存储对象的设备不匹配，发出警告
                if device != storage.device:
                    log.warning("device mismatch: %s != %s", device, storage.device)
                    # TODO: 将存储对象转移到正确的设备上？但是如果操作失败将会非常神秘！
                    # 最好在序列化格式中不存储设备信息...
                return storage
        
        # 如果无法加载存储对象，则记录警告信息并生成随机数据
        log.warning("could not load %s, generating random data instead", storage_hash)
        
        # 根据传入的字节数计算形状
        shape = (nbytes // dtype_hint.itemsize,)
        
        # 根据形状获取默认步幅
        stride = _stride_or_default(None, shape=shape)
        
        # 生成指定类型和设备的随机数据，并返回其未类型化存储
        return rand_strided(shape, stride, dtype_hint, device).untyped_storage()

    # 创建张量对象并配置其存储、形状、步幅等属性
    def tensor(
        self,
        storage,
        shape,
        stride=None,
        *,
        storage_offset=None,
        dtype=None,
        requires_grad=None,
        is_leaf=None,
        **metadata,
    ):
        # 根据传入的参数获取默认步幅
        stride = _stride_or_default(stride, shape=shape)
        
        # 根据传入的参数获取存储偏移量
        storage_offset = _storage_offset_or_default(storage_offset)
        
        # 根据传入的参数获取数据类型
        dtype = _dtype_or_default(dtype)
        
        # 根据传入的参数获取是否需要梯度
        requires_grad = _requires_grad_or_default(requires_grad)
        
        # 根据传入的参数获取是否为叶子节点
        is_leaf = _is_leaf_or_default(is_leaf)
        
        # 使用传入的存储对象、存储偏移量、形状和步幅创建张量
        t = torch.tensor(
            [], dtype=dtype, device=storage.device, requires_grad=requires_grad
        )
        
        # 使用无梯度环境设置张量的存储、存储偏移量、形状和步幅
        with torch.no_grad():
            t.set_(storage, storage_offset, shape, stride)
        
        # 如果张量不是叶子节点，则通过克隆和再次设置张量来创建假的自动求导历史
        if not is_leaf:
            # 以一种不良的方式伪造一些自动求导历史
            with torch.enable_grad():
                t = t.clone(memory_format=torch.preserve_format)
            with torch.no_grad():
                t.set_(storage, storage_offset, shape, stride)
        
        # 断言张量是否为安全的叶子节点
        assert torch._subclasses.meta_utils.safe_is_leaf(t) == is_leaf
        
        # 设置张量的元数据
        torch._utils.set_tensor_metadata(t, metadata)
        
        # 将张量对象添加到参数列表中
        self.args.append(t)
        
        # 返回创建的张量对象，用于向后兼容
        return t  # for BC

    # 将符号整数值添加到参数列表中并返回该值，用于向后兼容
    def symint(self, val):
        self.args.append(val)
        return val  # for BC
# 这是我们的写入策略：
#  1. 我们将把所有输入流写入磁盘
#  2. 现在可以确定性地随机化输入，或者重新从磁盘加载输入
#  3. 如果不提供输入，可以直接运行脚本，此时我们将用随机数据填充输入并祈祷。这是旧有行为，但如果要找出即使随机输入也会触发错误的情况，这也很有用
#  4. 我们可以提供一个进程内的“检查随机化的功能是否也有效”的选项，但这很微妙，所以我们暂时不这样做

class InputWriter:
    def __init__(self, save_dir, *, stable_hash=False):
        self._lines = []
        # TODO: 考虑确保张量和存储计数器的对齐？
        self.storage_counter = itertools.count()
        self.save_dir = save_dir
        self.store = (
            ContentStoreWriter(save_dir, stable_hash=stable_hash)
            if save_dir is not None
            else None
        )
        self.seen_storages = {}

    def lines(self):
        r = [
            "def load_args(reader):",
        ]
        r.extend(f"    {l}" for l in self._lines)
        # 如果我们需要以一种破坏性的方式改变load_args的内部格式
        r.append("load_args._version = 0")
        return r

    # 存储器是无类型的，但如果没有真实数据，我们需要使用数据进行初始化，
    # 因此我们给出一个提示，说明可能适合的初始化方式
    #
    # 如果有一个FakeTensor，device_hint告诉我们应该使用什么设备
    def storage(self, untyped_storage, *, dtype_hint=None, device_hint=None) -> str:
        ws = StorageWeakRef(untyped_storage)
        v = self.seen_storages.get(ws)
        if v is not None:
            return v
        v = f"buf{next(self.storage_counter)}"
        maybe_dtype_hint = ""
        if _dtype_or_default(None) != _dtype_or_default(dtype_hint):
            maybe_dtype_hint = f", dtype_hint={dtype_hint!r}"
        # TODO: 在设备上可选是有点无意义的，因为默认是CPU，但我们关心的大多数重现情况是CUDA
        maybe_device = ""
        device = untyped_storage.device
        if device.type == "meta":
            assert device_hint is not None
            device = device_hint
        if _device_or_default(None) != device:
            maybe_device = f", device={device!r}"
        nbytes = untyped_storage.nbytes()
        storage_hash = None
        if self.store is not None and untyped_storage.device.type != "meta":
            storage_hash = self.store.write_storage(untyped_storage)
        self._lines.append(
            f"{v} = reader.storage({storage_hash!r}, {nbytes!r}{maybe_device}{maybe_dtype_hint})"
        )
        self.seen_storages[ws] = v
        return v
    # 定义一个方法 `tensor`，用于将张量 `t` 添加到阅读器对象中
    def tensor(self, name, t) -> None:
        # 获取张量的存储对象，推测数据类型和设备信息
        storage = self.storage(
            t.untyped_storage(), dtype_hint=t.dtype, device_hint=t.device
        )
        args = []
        # 注意：这是位置参数，必须放在最前面
        if _stride_or_default(None, shape=t.shape) != t.stride():
            args.append(str(tuple(t.stride())))
        # 如果数据类型不是默认值，则添加到参数列表中
        if _dtype_or_default(None) != t.dtype:
            args.append(f"dtype={t.dtype!r}")
        # 如果存储偏移量不是默认值，则添加到参数列表中
        if _storage_offset_or_default(None) != t.storage_offset():
            args.append(f"storage_offset={t.storage_offset()!r}")
        # 获取张量的元数据
        tensor_metadata = torch._utils.get_tensor_metadata(t)
        if tensor_metadata:
            # 将元数据键值对转换为字符串，并添加到参数列表中
            args.extend(f"{k}={v!r}" for k, v in tensor_metadata.items())
        # 如果需要梯度不是默认值，则添加到参数列表中
        if _requires_grad_or_default(None) != t.requires_grad:
            args.append(f"requires_grad={t.requires_grad!r}")
        # 检查张量是否是叶子节点
        is_leaf = torch._subclasses.meta_utils.safe_is_leaf(t)
        if _is_leaf_or_default(None) != is_leaf:
            args.append(f"is_leaf={is_leaf!r}")
        # 构建阅读器行，将各参数拼接成字符串，并添加到 `_lines` 中
        self._lines.append(
            "reader.tensor("
            + ", ".join([storage, str(tuple(t.shape)), *args])
            + f")  # {name}"
        )

    # TODO: 这个函数目前实际上不会进行符号整数化
    def symint(self, name, val) -> None:
        # 如果 `val` 是 `torch.SymInt` 类型，则获取其节点的提示信息
        if isinstance(val, torch.SymInt):
            val = val.node.hint
        # 将 `val` 添加到 `_lines` 中，作为 `reader.symint` 方法的调用参数，并添加注释 `name`
        self._lines.append(f"reader.symint({val!r})  # {name}")
    """
    Takes in a function which has been printed with print_readable() and constructs kwargs to run it.

    Handles Tensor inputs, Symints, and a graph module which might have tensor constants.

    Consider a function `forward` defined as follows:

    def forward(self, primals_1: "f32[1001, 6]", primals_2: "f32[s0]", primals_3: "Sym(s0)",):
        _tensor_constant0: "i64[4190]" = self._tensor_constant0
        # Further implementation

    kwargs = aot_graph_input_parser(forward)
    forward(**kwargs)
    """

    from torch.fx.graph import dtype_abbrs  # 导入 dtype_abbrs 模块，用于处理数据类型缩写

    # 创建 dtype_map 字典，将数据类型缩写与其全名对应起来
    dtype_map = {value: key for key, value in dtype_abbrs.items()}
    
    # 构建正则表达式，用于匹配数据类型的缩写
    dtype_pattern = "|".join(dtype_abbrs.values())

    # 提取函数的源代码
    source = inspect.getsource(func)

    # 定义正则表达式，用于匹配张量赋值语句
    tensor_assignment_regex = rf"(_tensor_constant\d+): \"({dtype_pattern})\[\s*(.*?)\s*\]\" = self\.(_tensor_constant\d+)"
    
    # 定义正则表达式，用于匹配张量类型的注解
    tensor_regex = rf"({dtype_pattern})\[\s*(.*?)\s*\]"
    
    # 定义正则表达式，用于匹配 Sym(s0) 形式的注解
    sym_shape_regex = r"Sym\((s\d+)\)"

    # 定义一个空类 TensorContainer，用于承载作为属性的张量
    class TensorContainer:
        "Container for tensors as attributes"
        pass

    # 创建 kwargs 字典，用于存储从注解中解析得到的张量信息
    kwargs: Dict[str, Any] = {}

    # 如果未提供 sym_shapes 参数，则设为一个空字典
    sym_shapes = sym_shapes or {}

    # 定义函数 get_sym_int，用于获取符号整数的具体值
    def get_sym_int(symint):
        torch._check(
            symint in sym_shapes or default_sym_shape is not None,
            lambda: f"{symint} not in symbolic_shapes and default sym shape not passed in",
        )
        return sym_shapes.get(symint, default_sym_shape)

    # 定义函数 gen_tensor，用于生成张量，并解析其中的符号形状为具体值
    def gen_tensor(shape, dtype) -> Tensor:
        resolved_shape = []
        dynamic_dims = []
        for i, dim in enumerate(shape):
            dim = dim.strip()
            if "s" in dim:  # 如果维度标识中包含 's'，则表示这是一个符号形状
                s = get_sym_int(dim)
                resolved_shape.append(s)
                dynamic_dims.append(i)
            else:
                resolved_shape.append(int(dim))  # 否则将其解析为整数

        constructor = torch.randn if dtype.is_floating_point else torch.zeros  # 根据数据类型选择构造函数
        out = constructor(resolved_shape, dtype=dtype, device=device)  # 生成张量
        for d in dynamic_dims:
            torch._dynamo.mark_dynamic(out, d)  # 标记动态维度
        return out

    # 解析函数的注解，获取参数类型信息
    annotations = func.__annotations__
    # 遍历参数注解字典中的每一项
    for param, annotation in annotations.items():
        # 跳过 'return' 参数的注解
        if param == "return":
            continue
        
        # 在注解中查找匹配 tensor_regex 的内容
        match = re.search(tensor_regex, annotation)
        if match:
            # 如果找到匹配项，提取数据类型和形状字符串
            data_type, shape_str = match.groups()
            # 将形状字符串按逗号分隔为元组形状
            shape = tuple(shape_str.split(","))
            # 根据数据类型映射表确定数据类型
            dtype = dtype_map[data_type]
            # 使用 gen_tensor 函数生成张量，并将其存入 kwargs 字典
            kwargs[param] = gen_tensor(shape, dtype)

        # 在注解中查找匹配 sym_shape_regex 的内容
        match = re.search(sym_shape_regex, annotation)
        if match:
            # 如果找到匹配项，调用 get_sym_int 函数获取符号整数，并存入 kwargs 字典
            kwargs[param] = get_sym_int(match.group(1))

    # 如果函数签名中包含 'self' 参数
    if "self" in inspect.signature(func).parameters:
        # 创建一个 TensorContainer 实例
        container = TensorContainer()
        # 将 'self' 参数设置为 TensorContainer 实例
        kwargs["self"] = container
        # 遍历源码中匹配 tensor_assignment_regex 的每一个匹配项
        for match in re.finditer(tensor_assignment_regex, source):
            # 提取属性名、数据类型、形状字符串和其它信息
            attr_name, data_type, shape_str, _ = match.groups()
            # 将形状字符串按逗号分隔为元组形状
            shape = tuple(shape_str.split(","))
            # 根据数据类型映射表确定数据类型
            dtype = dtype_map[data_type]
            # 使用 setattr 函数将生成的张量设置为 TensorContainer 实例的属性
            setattr(container, attr_name, gen_tensor(shape, dtype))

    # 返回填充好的 kwargs 字典
    return kwargs
```