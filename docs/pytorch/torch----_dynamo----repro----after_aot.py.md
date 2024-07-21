# `.\pytorch\torch\_dynamo\repro\after_aot.py`

```py
# mypy: allow-untyped-defs
# 导入所需模块
import argparse  # 解析命令行参数
import copy  # 复制对象
import functools  # 创建装饰器
import io  # 提供 I/O 操作的工具
import logging  # 记录日志
import os  # 提供与操作系统交互的功能
import shutil  # 提供文件和目录操作的工具
import subprocess  # 运行外部命令
import sys  # 提供对 Python 运行时环境的访问
import textwrap  # 格式化文本的工具
import uuid  # 生成唯一标识符
from importlib import import_module  # 动态导入模块
from tempfile import TemporaryFile  # 创建临时文件
from typing import Any, Callable, Dict, Union  # 类型提示

import torch  # PyTorch 深度学习库
import torch.fx as fx  # PyTorch 的特效图库
import torch.nn as nn  # PyTorch 神经网络模块
from torch._dynamo.debug_utils import (
    _cuda_system_info_comment,  # CUDA 系统信息的注释
    AccuracyError,  # 精度错误异常类
    backend_accuracy_fails,  # 后端精度失败检查
    BuckTargetWriter,  # Buck 目标写入器
    cast_to_fp64,  # 转换为双精度浮点数
    extra_imports,  # 额外的导入
    generate_config_string,  # 生成配置字符串
    helper_for_dump_minify,  # 用于转储和缩小的辅助函数
    InputReader,  # 输入读取器
    InputWriter,  # 输入写入器
    MAX_CONSTANT_NUMEL_INLINE,  # 内联的最大常量元素数
    minifier_dir,  # 缩小器目录
    NNModuleToString,  # 神经网络模块转换为字符串
    NopInputReader,  # 空操作的输入读取器
    same_two_models,  # 比较两个模型是否相同
)
from torch._dynamo.utils import clone_inputs, counters, same  # 工具函数
from torch.fx.experimental.proxy_tensor import make_fx  # 创建特效图的代理张量
from torch.fx.experimental.symbolic_shapes import (
    fx_placeholder_targets,  # 特效图占位目标
    has_free_symbols,  # 是否有自由符号
)
from torch.hub import tqdm  # 显示进度条

from .. import config  # 导入自定义配置

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

inductor_config = import_module("torch._inductor.config")  # 导入 PyTorch Inductor 的配置模块
use_buck = inductor_config.is_fbcode()  # 检查是否在 FBCode 环境下运行

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           MAIN ENTRY POINT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def wrap_compiler_debug(unconfigured_compiler_fn, compiler_name: str):
    """
    Minifier for Fx Graph modules after Aot Autograd has finished. We wrap both
    forward and backward call separately with the backend compiler_fn - like
    inductor or nvfuser. Intercepting after Aot Autograd presents neat
    abstraction, where all the params are lifted as graph inputs, making it easy
    to save the graph as a string.
    """
    @functools.wraps(unconfigured_compiler_fn)
    return debug_wrapper  # 返回 debug_wrapper 函数的包装

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP REPROS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def generate_compiler_repro_string(gm, args, *, stable_output=False, save_dir=None):
    model_str = textwrap.dedent(
        f"""
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

{generate_config_string(stable_output=stable_output)}  # 生成配置字符串

isolate_fails_code_str = None  # 初始化隔离失败代码字符串为 None

{extra_imports}  # 插入额外的导入

        """
    )
    if not stable_output:
        model_str += f"# torch version: {torch.version.__version__}\n"  # 若不是稳定输出，记录 Torch 版本信息
        if hasattr(torch.version, "cuda"):
            model_str += f"# torch cuda version: {torch.version.cuda}\n"  # 若有 CUDA 版本信息，记录 CUDA 版本
        if hasattr(torch.version, "git_version"):
            model_str += f"# torch git version: {torch.version.git_version}\n\n\n"  # 记录 Torch 的 Git 版本信息
        model_str += _cuda_system_info_comment()  # 添加 CUDA 系统信息的注释

    model_str += NNModuleToString.convert(gm)  # 转换神经网络模块为字符串形式

    # get hint shape/stride when dynamic shape enabled
    # 定义一个函数，用于处理输入列表 x 中的元素：
    # 如果元素是 torch.SymInt 类型，则返回其 node 的提示信息；
    # 否则直接返回元素本身
    def hint_if_symint(x):
        return tuple(i.node.hint if isinstance(i, torch.SymInt) else i for i in x)

    # 创建一个输入写入器对象，指定保存目录为 save_dir
    writer = InputWriter(save_dir)

    # 遍历 gm 的 fx 占位符和 args 列表的元素，一一对应处理
    for placeholder, arg in zip(fx_placeholder_targets(gm), args):
        # 如果 arg 是 int 或者 torch.SymInt 类型之一
        if isinstance(arg, (int, torch.SymInt)):
            # 使用输入写入器将占位符和 arg 写入
            writer.symint(placeholder, arg)
        # 如果 arg 是 torch.Tensor 类型
        elif isinstance(arg, torch.Tensor):
            # TODO: 改进这些名称使用完全限定名（Fully Qualified Name）
            # 使用输入写入器将占位符和 arg 写入
            writer.tensor(placeholder, arg)
        else:
            # 如果 arg 类型不是 torch.SymInt/int 或 torch.Tensor，则抛出类型错误异常
            raise TypeError(f"arg is neither SymInt/int nor torch.Tensor, {arg}")

    # 将 writer 中的所有行连接成字符串，并添加到 model_str 中
    model_str += "\n".join(writer.lines()) + "\n"

    # 将字符串 "mod = Repro()\n" 添加到 model_str 中
    model_str += "mod = Repro()\n"

    # 返回拼接后的 model_str 字符串作为函数的结果
    return model_str
def dump_to_minify(gm, args, compiler_name: str):
    out = io.StringIO()
    # 创建输出的字符串流对象
    subdir = os.path.join(minifier_dir(), "checkpoints")
    # 检查并创建保存检查点的子目录
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    # 调用 save_graph_repro 函数，将输出写入字符串流中
    save_graph_repro(out, gm, args, compiler_name, save_dir=subdir, command="minify")
    # 将字符串流中的内容传递给辅助函数处理，并返回处理后的结果
    return helper_for_dump_minify(out.getvalue())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP MINIFIER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # 定义函数的参数 compiler_name，类型为字符串
    compiler_name: str,
    # 定义可选参数 env，默认为 None
    env=None,
    # 定义可选参数 save_dir，默认为 None
    save_dir=None,
    # 定义可选参数 accuracy，默认为 None
    accuracy=None,
    # 定义可选参数 tracing_mode，默认为 None
    tracing_mode=None,
    # 定义可选参数 check_str，默认为 None
    check_str=None,
def isolate_test(
    fx_g, args, env=None, compiler_name=None, save_dir=None, use_buck=False,
    accuracy=None, tracing_mode=False, check_str=None
):
    # 如果环境变量未提供，则使用空字典
    if env is None:
        env = {}
    
    # 创建子目录路径，用于保存临时文件
    subdir = os.path.join(os.getcwd(), "isolate")
    
    # 如果子目录不存在，则创建它，如果存在则保持其状态
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    
    # 生成一个随机的文件名，并将其设置为在子目录中保存的 Python 文件
    file_name = os.path.join(subdir, f"{str(uuid.uuid4())[:5]}.py")
    
    # 打开文件名对应的文件对象，并将图表重现保存到文件中
    with open(file_name, "w") as fd:
        save_graph_repro(
            fd,
            fx_g,
            args,
            compiler_name,
            save_dir=save_dir,
            command="minifier-query",
            accuracy=accuracy,
            tracing_mode=tracing_mode,
            check_str=check_str,
        )
    
    # 复制当前进程的环境变量，并将额外提供的环境变量合并进去
    new_env = os.environ.copy()
    new_env = {**new_env, **env}
    
    # 创建临时文件对象用于接收子进程的输出
    stdout, stderr = TemporaryFile(), TemporaryFile()

    # 根据是否使用 Buck 工具生成命令，或者直接运行生成的 Python 文件
    if use_buck:
        cmd = BuckTargetWriter(file_name).write(print_msg=False)
    else:
        cmd = ["python", file_name]

    # 启动一个子进程来执行生成的命令或 Python 文件
    p = subprocess.Popen(
        cmd,
        cwd=subdir,
        stdout=stdout,
        stderr=stderr,
        env=new_env,
    )
    p.wait()

    # 将子进程的标准输出和标准错误输出读取出来并打印到控制台
    stdout.seek(0)
    stderr.seek(0)
    print(
        textwrap.indent(stdout.read().decode("utf-8"), prefix=">>  "), file=sys.stdout
    )
    print(
        textwrap.indent(stderr.read().decode("utf-8"), prefix=">>  "), file=sys.stderr
    )
    
    # 返回子进程的退出码是否为非零，以表示测试是否失败
    return p.returncode != 0
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# 定义一个名为 repro_common 的函数，用于处理复现脚本的公共逻辑
def repro_common(options, mod, load_args):
    # 断言，检查模型中是否没有命名参数
    assert not any(mod.named_parameters())
    # 遍历模型中的所有缓冲区（buffers）
    for n, b in mod.named_buffers():
        # 如果缓冲区中元素数量大于 MAX_CONSTANT_NUMEL_INLINE，则记录警告信息
        if b.numel() > MAX_CONSTANT_NUMEL_INLINE:
            log.warning(
                "Constant %s was not serialized, generated random data instead. "
                "If you think this is affecting you, please comment on "
                "https://github.com/pytorch/pytorch/issues/100468",
                n,
            )

    # 检查 load_args 是否没有 "_version" 属性，记录警告信息
    if not hasattr(load_args, "_version"):
        log.warning(
            "load_args does not have a _version attribute, please file a bug to PyTorch "
            "and describe how you generate this repro script"
        )
    else:
        # 如果 load_args 的版本大于 0，则记录警告信息，因为当前版本仅支持版本 0
        if load_args._version > 0:
            log.warning(
                "load_args is version %s, but this version of PyTorch only supports "
                "version 0.  We will try to run it anyway but there may be an incompatibility; "
                "if so, try upgrading your version of PyTorch.",
                load_args._version,
            )

    # 创建一个 NopInputReader 实例
    nop_reader = NopInputReader()
    # 调用 load_args 方法传入 nop_reader
    load_args(nop_reader)

    # 使用 tqdm 创建进度条，加载输入数据
    with tqdm(desc="Loading inputs", total=nop_reader.total) as pbar:
        # 创建 InputReader 实例
        input_reader = InputReader(save_dir=options.save_dir, pbar=pbar)
        # 调用 load_args 方法传入 input_reader
        load_args(input_reader)
        # 获取 input_reader 的参数 args
        args = input_reader.args

    # 将 mod 转换为 GraphModule，使用 make_fx 函数
    # TODO: 加速这一过程
    mod = make_fx(mod, tracing_mode=options.tracing_mode)(*args)

    # 设置 torch._inductor.config.generate_intermediate_hooks 为 True
    torch._inductor.config.generate_intermediate_hooks = True

    # 返回修改后的模型和参数
    return mod, args


# 定义一个名为 ACCURACY_FAILS 的字典，包含字符串键和回调函数值
ACCURACY_FAILS: Dict[str, Callable[[nn.Module, Any], bool]] = {
    "": inductor_fails,
    # 下面的注释解释了 "accuracy" 和 "strict_accuracy" 的区别
    # 这可能看起来是反过来的，但实际上不是。strict_accuracy 意味着 "每当我们看到任何偏离时，我们都会最小化"，
    # 而 accuracy 更保守，只有在有意义的 fp64 偏离时才会最小化
    "accuracy": functools.partial(
        inductor_accuracy_fails, require_fp64=True, ignore_non_fp=True
    ),
    "strict_accuracy": inductor_accuracy_fails,
}


# 定义一个名为 repro_minifier_query 的函数，用于处理复现脚本的查询逻辑
def repro_minifier_query(options, mod, load_args):
    # 调用 repro_common 函数获取处理后的模型和参数
    mod, args = repro_common(options, mod, load_args)
    # 根据 options.accuracy 获取对应的 ACCURACY_FAILS 函数
    fail_fn = functools.partial(
        ACCURACY_FAILS[options.accuracy], check_str=options.check_str
    )
    # 如果 fail_fn 返回 True，则退出程序并返回 1；否则退出程序并返回 0
    if fail_fn(mod, args):
        sys.exit(1)
    else:
        sys.exit(0)


# 定义一个名为 repro_minify 的函数，用于处理复现脚本的最小化逻辑
def repro_minify(options, mod, load_args):
    # 导入 minifier 模块中的 minifier 函数
    from functorch.compile import minifier

    # 调用 repro_common 函数获取处理后的模型和参数
    mod, args = repro_common(options, mod, load_args)
    # 根据 options.accuracy 确定使用的编译器名称
    compiler_name = "inductor_accuracy" if options.accuracy != "" else "inductor"

    # 如果有多个 CUDA 设备，选择第二个设备作为偏好设备
    favored_device = 1 if torch.cuda.device_count() >= 2 else 0
    # 设置环境变量 CUDA_VISIBLE_DEVICES 为偏好设备的索引
    env_variables = {"CUDA_VISIBLE_DEVICES": str(favored_device)}

    # 定义 module_fails 变量，类型未指定
    module_fails: Any
    # 如果选项中包含隔离标志，则创建部分应用的函数module_fails，
    # 其中包含isolate_fails函数的部分参数，并设置其它固定的参数
    if options.isolate:
        module_fails = functools.partial(
            isolate_fails,
            env=env_variables,
            compiler_name=compiler_name,
            save_dir=options.save_dir,
            accuracy=options.accuracy,
            tracing_mode=options.tracing_mode,
        )
    else:
        # 如果选项中没有隔离标志，则直接使用ACCURACY_FAILS中对应精度的函数
        module_fails = ACCURACY_FAILS[options.accuracy]

    # 调用minifier函数，对模块mod进行压缩和优化处理
    # module_fails是一个部分应用的函数，其中包含了检查字符串check_str的参数
    # dump_state是一个部分应用的函数，其中包含了dump_compiler_graph_state函数的部分参数
    # 同时设置其它可选参数，如保存目录、是否将中间数据写入磁盘、是否跳过保存中间数据的操作、是否跳过健全性检查、最大粒度等
    minifier(
        mod,
        args,
        module_fails=functools.partial(module_fails, check_str=options.check_str),
        dump_state=functools.partial(
            dump_compiler_graph_state, compiler_name=compiler_name
        ),
        save_dir=options.save_dir,
        offload_to_disk=options.offload_to_disk,
        skip_offload=options.skip_saving_eager_intermediates,
        skip_sanity=options.skip_sanity,
        max_granularity=options.max_granularity,
    )
def repro_analyze(options, mod, load_args):
    # 导入必要的模块和函数
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.hooks import intermediate_hook

    # 调用 repro_common 函数处理选项、模型和加载参数
    mod, args = repro_common(options, mod, load_args)

    # 使用 tqdm 显示编译进度
    with tqdm(desc="Compiling"):
        compiled = compile_fx_inner(mod, args)
    # 获取计数器中的中间钩子数目
    total = counters["inductor"]["intermediate_hooks"]

    # 初始化一个空集合，用于存储已知的名称
    known_names = set()

    # 定义一个保存钩子函数，将名称加入已知集合并保存中间结果
    def save_hook(name, val):
        known_names.add(name)
        # 如果选项不跳过保存感应器中间结果，则使用写入器保存张量
        if not options.skip_saving_inductor_intermediates:
            writer.write_tensor(os.path.join("inductor", name), val)
        pbar.update(1)  # type: ignore[has-type]

    # 创建写入器和读取器对象
    writer = torch.utils._content_store.ContentStoreWriter(
        options.save_dir, stable_hash=options.stable_hash
    )
    reader = torch.utils._content_store.ContentStoreReader(options.save_dir)

    # 克隆输入参数
    new_args = clone_inputs(args)
    # 注册中间钩子函数，用于保存感应器中间结果，并使用 tqdm 显示保存进度
    with intermediate_hook(save_hook), tqdm(
        desc="Saving inductor intermediates", total=total
    ) as pbar:
        compiled(new_args)
        assert not new_args

    # 定义比较元组的函数
    def compare_tuples(tuple1, tuple2):
        diff_indices = [i for i in range(len(tuple1)) if tuple1[i] != tuple2[i]]
        diff_values = [(tuple1[i], tuple2[i]) for i in diff_indices]

        if not diff_values:
            return None
        else:
            return " and ".join(f"{a} != {b}" for a, b in diff_values)

    # 定义检查钩子函数，用于比较张量的元数据，检查感应器的确定性
    def check_hook(name, val):
        meta = writer.compute_tensor_metadata(val)
        meta2 = reader.read_tensor_metadata(os.path.join("inductor", name))
        reason = compare_tuples(meta, meta2)
        if reason is not None:
            pbar.write(f"NONDETERMINISTIC INDUCTOR at {name} ({reason})")
        pbar.update(1)

    # 如果未跳过检查感应器的确定性选项，则进行检查
    if not options.skip_check_deterministic:
        new_args = clone_inputs(args)
        # 注册中间钩子函数，用于检查感应器的确定性，并使用 tqdm 显示检查进度
        with intermediate_hook(check_hook), tqdm(
            desc="Checking inductor determinism", total=total
        ) as pbar:
            compiled(new_args)
            assert not new_args

    # 定义 WriterInterp 类，继承自 fx.Interpreter，用于运行节点并保存中间结果
    class WriterInterp(fx.Interpreter):
        def __init__(self, mod, subdir):
            super().__init__(mod)
            self.subdir = subdir

        def run_node(self, n):
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                pbar.update(1)
                writer.write_tensor(os.path.join(self.subdir, name), r)
            return r

    # 注意：模块转换操作实际上不执行任何操作，因为模块上没有参数/缓冲区
    # 如果选项中未跳过保存 float64 中间结果的步骤，则执行以下操作
    if not options.skip_saving_float64_intermediates:
        # 复制模型和输入参数，将它们转换为 float64 类型
        new_mod, new_args = cast_to_fp64(copy.deepcopy(mod), clone_inputs(args))
        # 使用进度条显示保存 float64 中间结果的进度
        with tqdm(desc="Saving float64 intermediates", total=total) as pbar:
            # 使用 WriterInterp 类执行 float64 模型的推理
            WriterInterp(new_mod, "float64").boxed_run(new_args)
        # 确保 new_args 为空，即无剩余输入参数
        assert not new_args

    # 定义一个 ExactReaderInterp 类，继承自 fx.Interpreter 类
    class ExactReaderInterp(fx.Interpreter):
        # 重写 run_node 方法，用于执行节点的计算
        def run_node(self, n):
            # 调用父类的 run_node 方法执行节点计算
            r = super().run_node(n)
            # 获取节点的名称
            name = n.name
            # 如果该名称在已知名称列表中
            if name in known_names:
                # 计算当前结果的张量元数据
                meta = writer.compute_tensor_metadata(r)
                # 从 "float64" 目录下读取与当前节点名称对应的张量元数据
                meta2 = reader.read_tensor_metadata(os.path.join("float64", name))
                # 比较两个张量元数据，确定是否存在非确定性的 float64 结果
                reason = compare_tuples(meta, meta2)
                # 若存在非确定性结果，则记录相关信息到进度条中
                if reason is not None:
                    pbar.write(f"NONDETERMINISTIC FLOAT64 at {name} ({reason})")
                # 更新进度条
                pbar.update(1)
            return r

    # TODO: check eager determinism
    # 如果未跳过检查浮点64位确定性的选项，则执行以下操作
    if not options.skip_check_deterministic:
        # 复制模型和输入参数，将它们转换为 float64 类型
        new_mod, new_args = cast_to_fp64(copy.deepcopy(mod), clone_inputs(args))
        # 使用进度条显示检查 float64 确定性的进度
        with tqdm(desc="Checking float64 determinism", total=total) as pbar:
            # 使用 ExactReaderInterp 类执行 float64 模型的推理
            ExactReaderInterp(new_mod).boxed_run(new_args)
            # 确保 new_args 为空，即无剩余输入参数
            assert not new_args

    # 现在我们保存了所有结果，通过 eager 图进行解释并进行比较
    # 定义一个 ReaderInterp 类，继承自 fx.Interpreter 类
    class ReaderInterp(fx.Interpreter):
        # 重写 run_node 方法，用于执行节点的计算
        def run_node(self, n):
            # 调用父类的 run_node 方法执行节点计算
            r = super().run_node(n)
            # 获取节点的名称
            name = n.name
            # 如果该名称在已知名称列表中
            if name in known_names:
                # 从 "inductor" 目录下读取与当前节点名称对应的张量
                inductor = reader.read_tensor(os.path.join("inductor", name))
                # 从 "float64" 目录下读取与当前节点名称对应的张量
                float64 = reader.read_tensor(os.path.join("float64", name))
                # 初始化 logged 变量为 False
                logged = False

                # 定义一个日志错误信息的函数
                def log_error(msg, *args):
                    nonlocal logged
                    # 将 logged 设置为 True，表示已记录错误信息
                    logged = True
                    # 将错误信息写入进度条中
                    pbar.write(f"DIVERGED at {name}: {msg % args}")

                # 使用 same 函数比较三个张量的值
                if not same(
                    r,
                    inductor,
                    float64,
                    tol=torch._dynamo.config.repro_tolerance,
                    equal_nan=True,
                    log_error=log_error,
                ):
                    # 如果存在差异，则断言 logged 必须为 True
                    assert logged
                # 更新进度条
                pbar.update(1)
            return r

    # 使用进度条显示检查模型输出是否存在差异的进度
    with tqdm(desc="Checking divergence", total=total) as pbar:
        # 使用 ReaderInterp 类执行模型的推理
        ReaderInterp(mod).boxed_run(args)
    # 确保 args 为空，即无剩余输入参数
    assert not args
# 获取参数并调用共同复现函数以准备模型和参数
def repro_get_args(options, mod, load_args):
    mod, args = repro_common(options, mod, load_args)
    return mod, args


# 运行复现，编译模型并返回一个函数以执行编译后的模型
def repro_run(options, mod, load_args):
    # 导入编译内部函数
    from torch._inductor.compile_fx import compile_fx_inner

    # 调用共同复现函数以准备模型和参数
    mod, args = repro_common(options, mod, load_args)

    # 编译模型
    compiled = compile_fx_inner(mod, args)

    # 如果设置了精度选项
    if options.accuracy != "":
        # 不明确支持 --accuracy 与 --strict-accuracy 的区别，这似乎有些反直觉
        # 检查两个模型是否相同，仅对前向传播进行检查，并根据配置决定是否忽略非浮点数
        if not same_two_models(
            mod,
            compiled,
            args,
            only_fwd=True,
            ignore_non_fp=config.repro_ignore_non_fp,
        ):
            # 如果检测到精度问题，则抛出精度错误异常
            raise AccuracyError("Bad accuracy detected")
    else:
        # 检查是否需要同步
        need_sync = False
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                need_sync = True
                break
        # 运行编译后的模型，根据需要执行同步操作以确保异常被捕获
        ref = compiled(list(args))
        if need_sync:
            synchronize()  # 确保出现段错误时可以被捕获

    # 返回一个 lambda 函数，用于执行编译后的模型
    return lambda: compiled(list(args))


# 运行复现脚本，初始化参数并返回一个参数解析器
def run_repro(
    mod,
    load_args,
    *,
    command="run",
    accuracy: Union[bool, str] = "",
    save_dir=None,
    tracing_mode=None,
    patch_code=None,
    check_str=None,
    **kwargs,
):
    # 对于所有未识别的关键字参数，记录警告信息
    for k in kwargs:
        log.warning(
            "Unrecognized kwarg %s; perhaps this repro was made on a newer version of PyTorch",
            k,
        )

    # 将布尔类型的精度参数转换为字符串
    if accuracy is True:
        accuracy = "accuracy"
    elif accuracy is False:
        accuracy = ""

    # 如果存在 patch_code 参数，则记录警告信息，因为在当前版本的 PyTorch 中不再起作用
    if patch_code is not None:
        log.warning(
            "patch_code no longer works on this version of PyTorch, silently ignoring"
        )

    # 创建参数解析器，描述当前复现脚本的功能和默认设置
    parser = argparse.ArgumentParser(
        description=f"""\
An after_aot repro script, typically triggering a bug in PyTorch Inductor.
When run with no arguments, this script defaults to running '{command}'.
Extra flags may be available; to find out more, try '{command} --help'.
There are also alternate subcommands available, see below.

default settings on this script:
  {accuracy=}
  {tracing_mode=}
  {save_dir=}
  {check_str=}
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 定义公共的命令行参数
    def common_flags(parser):
        accuracy_group = parser.add_mutually_exclusive_group()
        accuracy_group.add_argument(
            "--no-accuracy",
            dest="accuracy",
            action="store_const",
            const="",
            default=accuracy,
            help="do not test accuracy, just run the module and see if it errors",
        )
        accuracy_group.add_argument(
            "--accuracy",
            action="store_const",
            const="accuracy",
            default=accuracy,
            help="""\
test if the RMSE between the compiled module and the fp64 reference is greater
than eager and the fp64 reference. This is usually more reliable than the
# 定义标准的 allclose 测试，我们期望编译时会有数值差异，通常会提高准确性。RMSE 测试允许编译模块与 eager 模块有很大的差异，只要这种差异将其更接近网络的“真实”数学值。注意事项：(1) 双精度仍可能受到舍入误差的影响，因此它不是完美的参考（例如参见 'Herbie: Automatically Improving Floating Point Accuracy'），对于检测所需工作精度并在任意精度浮点数中计算的方法，这对于张量计算来说不太实际；(2) 如果要比较的输出中样本不足，我们可能会不幸地得到一个比 eager 更大的 RMSE，这可以通过在某个 p 值上应用更严格的统计测试来克服，这部分留给未来的工作。

accuracy_group.add_argument(
    "--strict-accuracy",
    dest="accuracy",
    action="store_const",
    const="strict_accuracy",
    default=accuracy,
    help="""\
默认情况下，在进行准确性最小化时，我们将拒绝改变从浮点数差异到整数/布尔差异的缩减。这是因为一些操作（如 ReLU）涉及暂时的尖锐边界，之后再次平滑；如果不要求浮点数上的差异，缩小器通常会专注于不同的布尔张量，尽管这不是差异的真正来源。然而，拒绝这些缩减会使缩小器更难以进行处理。使用此选项将允许缩小器处理所有差异--最终可能不会得到有用的重现。

parser.add_argument(
    "--save-dir",
    type=str,
    default=save_dir,
    metavar="DIR",
    help="保存输入的目录",
)
parser.add_argument(
    "--no-save-dir",
    dest="save_dir",
    action="store_const",
    const=None,
    help="不使用任何目录保存输入",
)
parser.add_argument(
    "--tracing-mode",
    type=str,
    metavar="{real,fake,symbolic}",
    default=tracing_mode,
    help="如何将重现模块跟踪为带有元数据的 GraphModule",
)

subparsers = parser.add_subparsers(
    dest="command", metavar="{run,minify,analyze}", required=True
)

parser_run = subparsers.add_parser(
    "run",
    help="只运行重现",
)
common_flags(parser_run)

parser_minify = subparsers.add_parser(
    "minify", help="在重现上运行缩小器"
)
common_flags(parser_minify)
parser_get_args = subparsers.add_parser("get_args", help="获取参数")
common_flags(parser_get_args)
    parser_minify_isolate = parser_minify.add_mutually_exclusive_group()
    # 创建互斥组，用于处理 --isolate 和 --no-isolate 两个参数的互斥选择
    parser_minify_isolate.add_argument(
        "--isolate",
        action="store_true",
        default=True,
        help="run in separate processes to avoid interference (default)",
    )
    # 添加 --isolate 参数选项到互斥组中，设置默认为 True，表示默认情况下在单独的进程中运行以避免干扰
    parser_minify_isolate.add_argument(
        "--no-isolate",
        dest="isolate",
        action="store_false",
        help="speed up by running all compilation in same process",
    )
    # 添加 --no-isolate 参数选项到互斥组中，用于在同一进程中运行所有编译以提高速度
    
    parser_minify.add_argument(
        "--skip-saving-eager-intermediates",
        action="store_true",
        help="skip saving eager intermediates on --minify",
    )
    # 添加 --skip-saving-eager-intermediates 参数选项，用于在 --minify 模式下跳过保存 eager 中间结果
    
    # TODO: make this an option for --analyze too
    # 添加 --offload-to-disk 参数选项，用于在压缩过程中将增量调试中间结果转移到磁盘上。如果内存溢出，请使用此选项。
    parser_minify.add_argument(
        "--offload-to-disk",
        action="store_true",
        help="during minification, offload delta debugging intermediates to disk.  Use if you're OOMing",
    )
    
    parser_minify.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip sanity check at beginning of minification on original graph",
    )
    
    parser_minify.add_argument(
        "--max-granularity",
        type=int,
        default=None,
        help="start at this granularity and work down; must be power of 2",
    )
    
    parser_minify.add_argument(
        "--check-str",
        type=str,
        default=check_str,
        help="require minified program to fail with error containing this string",
    )
    
    parser_analyze = subparsers.add_parser(
        "analyze", help="run the accuracy analyzer on the repro"
    )
    common_flags(parser_analyze)
    # 添加 --skip-saving-inductor-intermediates 参数选项，用于在 --analyze 模式下跳过保存电感器中间结果
    parser_analyze.add_argument(
        "--skip-saving-inductor-intermediates",
        action="store_true",
        help="skip saving inductor intermediates on --analyze",
    )
    
    parser_analyze.add_argument(
        "--skip-saving-float64-intermediates",
        action="store_true",
        help="skip saving float64 intermediates",
    )
    
    parser_analyze.add_argument(
        "--skip-check-deterministic",
        action="store_true",
        help="skip checking that the network is deterministic",
    )
    
    parser_analyze.add_argument(
        "--stable-hash",
        action="store_true",
        help="use SHA-1 checksum instead of fast (but possibly unsound) hash",
    )
    
    # 在 minification 上下文中运行 repro，反转退出代码的含义
    parser_minifier_query = subparsers.add_parser(
        "minifier-query",
    )
    common_flags(parser_minifier_query)
    # 添加 --check-str 参数选项，用于要求被压缩的程序在出错时包含此字符串
    parser_minifier_query.add_argument(
        "--check-str",
        type=str,
        default=check_str,
        help="require minified program to fail with error containing this string",
    )
    
    args = None
    if len(sys.argv) <= 1:
        args = [command, *sys.argv[1:]]
    
    options = parser.parse_args(args)
    # 根据命令选择相应的函数处理
    COMMAND_FNS = {
        "minify": repro_minify,
        "analyze": repro_analyze,
        "minifier-query": repro_minifier_query,
        "run": repro_run,
        "get_args": repro_get_args,
    }
    # 根据选项中的命令查找对应的函数，并执行
    return COMMAND_FNS[options.command](options, mod, load_args)
```