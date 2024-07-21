# `.\pytorch\torch\_dynamo\repro\after_dynamo.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库
import argparse  # 命令行参数解析
import copy  # 复制对象
import functools  # 函数装饰器相关功能
import logging  # 日志记录
import os  # 操作系统功能
import shutil  # 文件和目录操作
import sys  # 系统相关功能
import textwrap  # 文本包装
from importlib import import_module  # 动态导入模块
from typing import Union  # 类型提示

import torch  # PyTorch 深度学习框架
import torch.fx as fx  # PyTorch FX，用于函数级别的图表示

from torch._dynamo.debug_utils import (  # 导入调试工具相关函数和类
    AccuracyError,  # 精度错误异常类
    backend_accuracy_fails,  # 后端精度失败检查函数
    BUCK_CMD_PREFIX,  # Buck 命令前缀
    BuckTargetWriter,  # Buck 目标写入器
    extra_imports,  # 额外导入项
    generate_config_string,  # 生成配置字符串函数
    helper_for_dump_minify,  # 辅助函数，用于转储和缩小
    InputReader,  # 输入读取器
    InputWriter,  # 输入写入器
    minifier_dir,  # 缩小目录
    NNModuleToString,  # 将 NN 模块转换为字符串
    NopInputReader,  # 空操作输入读取器
    run_fwd_maybe_bwd,  # 运行前向（可能反向）过程
    same_two_models,  # 比较两个模型是否相同
)
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets  # FX 符号形状实验性模块
from torch.hub import tqdm  # 进度条显示

from .. import config  # 导入本地配置
from ..backends.registry import lookup_backend, register_debug_backend  # 后端注册和查找相关函数
from ..debug_utils import clone_inputs_retaining_gradness  # 克隆输入并保留梯度信息

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

inductor_config = import_module("torch._inductor.config")  # 动态导入 torch._inductor.config 模块
use_buck = inductor_config.is_fbcode()  # 检查是否在 FBCode 环境中运行

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           MAIN ENTRY POINT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _accuracy_fails(gm, example_inputs, compiler_fn):
    # 检查后端精度是否失败
    return backend_accuracy_fails(
        gm,
        example_inputs,
        compiler_fn,
        only_fwd=config.repro_forward_only,  # 是否仅前向传播
        ignore_non_fp=config.repro_ignore_non_fp,  # 是否忽略非浮点数
    )


class WrapBackendDebug:
    def __init__(self, unconfigured_compiler_fn, compiler_name: str):
        functools.wraps(unconfigured_compiler_fn)(self)  # 使用 unconfigured_compiler_fn 装饰初始化方法
        self._torchdynamo_orig_callable = unconfigured_compiler_fn  # 保存未配置的编译函数引用
        self._compiler_name = compiler_name  # 设置编译器名称属性
        if hasattr(unconfigured_compiler_fn, "__name__"):
            self.__name__ = unconfigured_compiler_fn.__name__  # 如果有 __name__ 属性，设置为同名
        if hasattr(unconfigured_compiler_fn, "compiler_name"):
            self.__name__ = unconfigured_compiler_fn.compiler_name  # 如果有 compiler_name 属性，设置为同名
        if hasattr(unconfigured_compiler_fn, "get_compiler_config"):
            self.get_compiler_config = unconfigured_compiler_fn.get_compiler_config  # 设置获取编译器配置的方法引用
    def __call__(self, gm, example_inputs, **kwargs):
        # 定义编译器函数，部分应用 self._torchdynamo_orig_callable 函数
        compiler_fn = functools.partial(self._torchdynamo_orig_callable, **kwargs)
        
        # 断言 config.repro_after 只能是 "dynamo", "aot" 或者 None
        assert config.repro_after in ("dynamo", "aot", None)

        # 如果 config.repro_after 是 "dynamo"，则执行以下逻辑
        if config.repro_after == "dynamo":

            # 定义函数用于给异常对象添加路径信息
            def add_paths(exc):
                exc.minifier_path = os.path.join(minifier_dir(), "minifier_launcher.py")
                # 如果使用 Buck，添加 Buck 命令信息
                if use_buck:
                    exc.buck_command = " ".join(
                        BUCK_CMD_PREFIX
                        + [BuckTargetWriter(exc.minifier_path).cmd_line_path]
                    )

            # 如果 config.repro_level 是 3，则调用 dump_to_minify_after_dynamo 函数
            if config.repro_level == 3:
                dump_to_minify_after_dynamo(gm, example_inputs, self._compiler_name)

            # 如果 config.repro_level 是 4，则进行精度检查
            if config.repro_level == 4:
                # 编译图形模块并检查精度失败
                compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                if _accuracy_fails(gm, example_inputs, compiler_fn):
                    log.warning(
                        "Accuracy failed for the TorchDynamo produced graph. Creating script to minify the error."
                    )
                    # 创建脚本以减小错误
                    dump_to_minify_after_dynamo(
                        fx.GraphModule(gm, copy.deepcopy(gm.graph)),
                        example_inputs,
                        self._compiler_name,
                    )
                    # 抛出精度错误异常，并添加路径信息
                    exc = AccuracyError("Bad accuracy detected.")
                    add_paths(exc)
                    raise exc
            else:
                try:
                    # 尝试编译图形模块并运行前向和可能的反向传播
                    compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                    run_fwd_maybe_bwd(compiled_gm, example_inputs)
                except Exception as exc:
                    log.warning(
                        "Compiled Fx GraphModule failed. Creating script to minify the error."
                    )
                    # 根据 config.repro_level 的不同，执行不同的错误处理
                    if config.repro_level == 1:
                        dump_state_fn = functools.partial(
                            dump_backend_state, compiler_name=self._compiler_name
                        )
                        dump_state_fn(
                            fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs
                        )
                    elif config.repro_level == 2:
                        dump_to_minify_after_dynamo(
                            fx.GraphModule(gm, copy.deepcopy(gm.graph)),
                            example_inputs,
                            self._compiler_name,
                        )
                    # 添加路径信息到异常对象，并重新抛出异常
                    add_paths(exc)
                    raise
        else:
            # 如果 config.repro_after 不是 "dynamo"，直接使用编译器函数编译 gm 和 example_inputs
            compiled_gm = compiler_fn(gm, example_inputs)

        # 返回编译后的图形模块
        return compiled_gm
# 定义一个装饰器函数，用于包装 TorchDynamo 生成的 Fx 图模块，用于进行代码精简和调试
def wrap_backend_debug(unconfigured_compiler_fn, compiler_name: str):
    """
    A minifier decorator that wraps the TorchDynamo produced Fx graph modules.
    As opposed to wrap_compiler_debug, this wrapper intercepts at the
    TorchDynamo produced Fx Graph Module. This makes it backend-agnostic to some
    level, e.g., it is useful for minifying issues related to Aot Autograd
    tracing.  If an error is found, we minify and save the minified repro in
    repro.tar.gz.
    """
    return WrapBackendDebug(unconfigured_compiler_fn, compiler_name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           REPRO DUMPERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# 生成用于后端不可知的简化版本的复现字符串
def generate_dynamo_fx_repro_string(
    gm,
    args,
    compiler_name,
    check_accuracy=False,
    *,
    stable_output=False,
    save_dir=None,
    command="run",
):
    """
    Generate a repro string for backend-agnostic minified version.
    """
    
    # 将神经网络模块转换为字符串表示
    model_str = NNModuleToString.convert(gm)
    
    # 创建输入写入器对象，用于生成加载参数的字符串
    writer = InputWriter(save_dir, stable_hash=True)
    for placeholder, arg in zip(fx_placeholder_targets(gm), args):
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        elif isinstance(arg, torch.Tensor):
            # 将张量写入输入写入器
            writer.tensor(placeholder, arg)
        else:
            # 抛出类型错误，如果参数既不是 SymInt/int 也不是 torch.Tensor
            raise TypeError(f"arg is neither SymInt/int nor torch.Tensor, {arg}")
    load_args = "\n".join(writer.lines())
    
    # 生成复现代码字符串，包括配置信息、额外的导入模块、模型字符串、加载参数等
    return textwrap.dedent(
        f"""
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

{generate_config_string(stable_output=stable_output)}

{extra_imports}

{model_str}
mod = Repro()

{load_args}

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy={check_accuracy!r}, command={command!r},
        save_dir={save_dir!r}, autocast={torch.is_autocast_enabled()!r}, backend={compiler_name!r})
"""
    )


# 将复现字符串保存到文件中
def dump_backend_repro_as_file(gm, args, compiler_name, check_accuracy=False):
    """
    Saves the repro to a repro.py file
    """
    curdir = os.getcwd()
    subdir = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    
    # 指定保存文件的路径和文件名
    file_name = os.path.join(subdir, f"minified_{len(gm.graph.nodes)}_nodes.py")
    log.warning(
        "Writing checkpoint with %s nodes to %s", len(gm.graph.nodes), file_name
    )
    
    # 将生成的复现字符串写入指定文件
    with open(file_name, "w") as fd:
        fd.write(
            generate_dynamo_fx_repro_string(
                gm, args, compiler_name, check_accuracy, save_dir=subdir
            )
        )
    
    # 更新最新的复现文件路径
    latest_repro = os.path.join(curdir, "repro.py")
    # 输出警告日志，指示正在将 file_name 复制到 latest_repro 以便于操作
    log.warning("Copying %s to %s for convenience", file_name, latest_repro)
    
    # 如果 use_buck 为真，则使用 BuckTargetWriter 类将 latest_repro 写入目标
    if use_buck:
        BuckTargetWriter(latest_repro).write()
    
    # 使用 shutil 模块的 copyfile 函数将 file_name 复制到 latest_repro
    shutil.copyfile(file_name, latest_repro)
# 检查是否可以将 GraphModule 转换为字符串以进行后续处理
assert NNModuleToString.can_convert_to_string(gm)

# 调用函数，将 GraphModule 转换为文件以复现后续问题
return dump_backend_repro_as_file(gm, args, compiler_name, check_accuracy)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER DUMPER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def dump_to_minify_after_dynamo(gm, args, compiler_name):
    # TODO: factor this out

    # 确定保存目录，用于存放检查点文件
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)

    # 调用辅助函数，生成动态图 GraphModule 的复现字符串，并保存到指定目录
    helper_for_dump_minify(
        generate_dynamo_fx_repro_string(
            gm,
            args,
            compiler_name,
            # 根据配置文件设置是否检查精度
            check_accuracy=config.repro_level == 4,
            save_dir=subdir,
            command="minify",
        )
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER BACKENDS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# 注册调试后端的函数装饰器
@register_debug_backend
def dynamo_minifier_backend(gm, example_inputs, compiler_name):
    from functorch.compile import minifier

    # 查找指定编译器名称对应的后端编译函数
    compiler_fn = lookup_backend(compiler_name)

    # TODO: It's inconsistent to pass SymInt inputs but REAL tensors.
    # We should pass ints and look at the GraphModule placeholders
    # to resolve them to SymInt (if necessary)
    # 尝试将 SymInt 输入解析为真实张量，确保传递一致性
    example_inputs = [
        i.node.hint if isinstance(i, torch.SymInt) else i for i in example_inputs
    ]

    try:
        # 使用编译器函数对 GraphModule 进行编译
        compiled_gm = compiler_fn(gm, example_inputs)
        # 执行前向传播和可能的反向传播
        run_fwd_maybe_bwd(compiled_gm, example_inputs)
        # 如果未检测到问题，则引发值错误异常
        raise ValueError("No issue was detected")
    except Exception as exc:
        orig_failure = str(exc)
        # 记录警告信息，指示编译的 Fx GraphModule 失败，创建用于缩小错误的脚本
        log.warning(
            "Compiled Fx GraphModule failed. Creating script to minify the error."
        )

        # 部分转储函数，使用 dump_backend_state 函数
        dump_state_fn = functools.partial(
            dump_backend_state, compiler_name=compiler_name
        )

        # 使用深度复制的 GraphModule 对象和输入示例，调用转储函数
        dump_state_fn(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs)

        # 部分失败函数，与编译器函数和原始失败消息一起调用
        fails_fn = functools.partial(
            backend_fails,
            compiler_fn=compiler_fn,
            orig_failure=orig_failure,
        )

        # 调用 minifier 函数，尝试缩小错误
        minifier(
            gm,
            example_inputs,
            module_fails=fails_fn,
            dump_state=dump_state_fn,
        )

    # 返回处理后的 GraphModule 对象
    return gm


# 注册调试精度后端的函数装饰器
@register_debug_backend
def dynamo_accuracy_minifier_backend(gm, example_inputs, compiler_name):
    from functorch.compile import minifier

    # 查找指定编译器名称对应的后端编译函数
    compiler_fn = lookup_backend(compiler_name)

    # 将模型设置为评估模式，以消除随机性
    gm.eval()
    # 检查准确性
    if _accuracy_fails(gm, example_inputs, compiler_fn):
        # 如果 TorchDynamo 生成的图的准确性失败，则记录警告信息
        log.warning("Accuracy failed for the TorchDynamo produced graph")
        # 部分函数应用，用于导出后端状态，并检查准确性
        dump_state_fn = functools.partial(
            dump_backend_state, compiler_name=compiler_name, check_accuracy=True
        )
        # 部分函数应用，用于检查编译器函数的准确性失败情况
        fails_fn = functools.partial(
            _accuracy_fails,
            compiler_fn=compiler_fn,
        )
        # 导出后端状态，包括 Torch 计算图和输入样本
        dump_state_fn(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs)
        # 最小化处理，包括编译器图模块、输入样本、模块失败和状态导出
        minifier(
            gm,
            example_inputs,
            module_fails=fails_fn,
            dump_state=dump_state_fn,
        )
    else:
        # 如果输入图在准确性测试中未失败，则记录错误信息
        log.error("Input graph does not fail accuracy testing")
    # 返回修改后的 Torch 图模块
    return gm
# Minifier使用此函数来识别是否minified图模块在相同错误下失败。
# 一个注意点是，当生成的图模块因不同原因失败时，Minifier可能会走错方向。
# 为了避免这种情况，我们保存原始异常的字符串，并检查新异常和旧异常之间的相似性。
# 在某些情况下，它们可能稍有不同，这取决于异常字符串是否依赖于失败节点信息。
# 因此，我们使用宽松的相似性度量来指导Minifier的路径。

def backend_fails(gm, example_inputs, compiler_fn, orig_failure):
    """
    Minifier uses this function to identify if the minified graph module fails
    with the same error.

    One caveat is that minifier can potentially go into a wrong direction when
    the resulting graph module fails for a different reason. To avoid this, we
    save the string for the original exception and check similarity between new
    and old exception. They can be somewhat different in some cases, when the
    exception string depends on the failing node information. So, we have a
    loose similarity metric to guide the minifier path.
    """
    from difflib import SequenceMatcher  # 导入SequenceMatcher用于字符串相似性比较

    try:
        # 运行原始gm以检查其快速有效性
        run_fwd_maybe_bwd(gm, clone_inputs_retaining_gradness(example_inputs))
        # 使用编译器函数编译gm和示例输入
        compiled_gm = compiler_fn(gm, example_inputs)
        # 再次运行编译后的gm以检查其快速有效性
        run_fwd_maybe_bwd(compiled_gm, clone_inputs_retaining_gradness(example_inputs))
        return False  # 如果没有异常抛出，则返回False表示不失败
    except Exception as e:
        new_failure = str(e)  # 将新异常转换为字符串
        # 使用SequenceMatcher比较原始失败和新失败之间的相似度，如果相似度大于0.5，则返回True表示失败
        if SequenceMatcher(None, orig_failure, new_failure).ratio() > 0.5:
            return True
        return False  # 否则返回False表示不失败


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           REPRO MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def run_load_args(options, mod, load_args):
    # 检查load_args是否具有"_version"属性，如果没有则发出警告
    if not hasattr(load_args, "_version"):
        log.warning(
            "load_args does not have a _version attribute, please file a bug to PyTorch "
            "and describe how you generate this repro script"
        )
    else:
        # 如果_load_args的版本大于0，则发出警告，因为当前版本只支持版本0
        if load_args._version > 0:
            log.warning(
                "load_args is version %s, but this version of PyTorch only supports "
                "version 0.  We will try to run it anyway but there may be an incompatibility; "
                "if so, try upgrading your version of PyTorch.",
                load_args._version,
            )

    nop_reader = NopInputReader()  # 创建一个NopInputReader对象
    load_args(nop_reader)  # 调用load_args函数，传入nop_reader对象

    with tqdm(desc="Loading inputs", total=nop_reader.total) as pbar:
        # 创建一个InputReader对象，保存目录为options.save_dir，并在进度条中显示总数
        input_reader = InputReader(save_dir=options.save_dir, pbar=pbar)
        load_args(input_reader)  # 调用load_args函数，传入input_reader对象
        args = input_reader.args  # 获取input_reader对象的args属性

    return args  # 返回args变量


def repro_minify(options, mod, load_args):
    args = run_load_args(options, mod, load_args)  # 运行run_load_args函数，获取args

    # 根据选项设置编译器函数
    if not options.accuracy:
        compiler_fn = lookup_backend("dynamo_minifier_backend")
    else:
        compiler_fn = lookup_backend("dynamo_accuracy_minifier_backend")

    if options.backend is None:
        # 如果options.backend为None，则抛出RuntimeError
        raise RuntimeError(
            "Compiler name is None - this likely means that a custom compiler "
            "was called by torchdynamo. Please remove this error, import your "
            "custom compiler function, and replace the backend=None "
            "line in run_repro to backend=<my_imported_custom_function>"
        )
    # 使用 functools.partial 函数创建一个新的函数 dynamo_minifier_backend，
    # 它是 compiler_fn 的部分应用，其中 compiler_name 参数由 options.backend 提供
    dynamo_minifier_backend = functools.partial(
        compiler_fn,
        compiler_name=options.backend,
    )

    # 调用 torch._dynamo.optimize(dynamo_minifier_backend) 函数，返回一个优化器函数，
    # 并将 mod 作为参数传递给该优化器函数，返回优化后的模型 opt_mod
    opt_mod = torch._dynamo.optimize(dynamo_minifier_backend)(mod)

    # 使用 torch.cuda.amp.autocast 上下文管理器，根据 options.autocast 的设置自动进行混合精度计算
    with torch.cuda.amp.autocast(enabled=options.autocast):
        # 调用 opt_mod 函数，传递 args 作为参数
        opt_mod(*args)
# 定义一个函数，用于运行重现脚本
def repro_run(options, mod, load_args):
    # 使用 torch._dynamo.optimize 函数优化模型
    opt_mod = torch._dynamo.optimize(options.backend)(mod)

    # 如果指定了精度选项
    if options.accuracy != "":
        # 将模型和优化后的模型设置为评估模式
        mod.eval()
        opt_mod.eval()

        # 使用 torch.cuda.amp.autocast 自动混合精度计算，根据选项是否启用
        with torch.cuda.amp.autocast(enabled=options.autocast):
            # 运行加载参数函数，获取参数 args
            args = run_load_args(options, mod, load_args)
            # 断言两个模型是否相同，用于验证运行是否正确
            assert same_two_models(mod, mod, args), "Eager itself failed"
            # 检查优化后的模型和原模型是否相同
            if not same_two_models(
                mod,
                opt_mod,
                args,
                only_fwd=config.repro_forward_only,
                ignore_non_fp=config.repro_ignore_non_fp,
            ):
                # 如果不相同，抛出精度错误异常
                raise AccuracyError("Dynamo failed")
    else:
        # 使用 torch.cuda.amp.autocast 自动混合精度计算，根据选项是否启用
        with torch.cuda.amp.autocast(enabled=options.autocast):
            # 运行加载参数函数，获取参数 args
            args = run_load_args(options, mod, load_args)
            # 运行前向（可能包含反向）函数，获取结果 ref，禁用克隆操作
            ref = run_fwd_maybe_bwd(
                mod, args, only_fwd=options.only_fwd, disable_clone=True
            )
            # 删除参数 args
            del args

            # 再次运行加载参数函数，获取参数 args
            args = run_load_args(options, mod, load_args)
            # 运行前向（可能包含反向）函数，获取结果 res，禁用克隆操作
            res = run_fwd_maybe_bwd(
                opt_mod, args, only_fwd=options.only_fwd, disable_clone=True
            )


# 定义一个函数，用于运行重现脚本的入口
def run_repro(
    mod,
    load_args,
    *,
    command="run",
    accuracy: Union[bool, str] = "",
    save_dir=None,
    autocast=False,
    backend="inductor",
    **kwargs,
):
    # 遍历额外的关键字参数 kwargs
    for k in kwargs:
        # 发出警告，指出未识别的关键字参数
        log.warning(
            "Unrecognized kwarg %s; perhaps this repro was made on a newer version of PyTorch",
            k,
        )

    # 如果 accuracy 参数为 True，则设为字符串 "accuracy"
    if accuracy is True:
        accuracy = "accuracy"
    # 如果 accuracy 参数为 False，则设为空字符串
    elif accuracy is False:
        accuracy = ""

    # 创建参数解析器 argparse.ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description=f"""\
An after_dynamo repro script, typically triggering a bug in Dynamo or
AOTAutograd.  When run with no arguments, this script defaults to running
'{command}'.  Extra flags may be available; to find out more, try '{command}
--help'.  There are also alternate subcommands available, see below.

default settings on this script:
  {accuracy=}
  {save_dir=}
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    def common_flags(parser):
        # 定义互斥的参数组，用于设置准确性相关选项
        accuracy_group = parser.add_mutually_exclusive_group()
        # 添加参数 --no-accuracy，如果设置则将 'accuracy' 设置为空字符串
        accuracy_group.add_argument(
            "--no-accuracy",
            dest="accuracy",
            action="store_const",
            const="",
            default=accuracy,
            help="do not test accuracy, just run the module and see if it errors",
        )
        # 添加参数 --accuracy，设置 'accuracy' 参数为 'accuracy'
        accuracy_group.add_argument(
            "--accuracy",
            action="store_const",
            const="accuracy",
            default=accuracy,
            help="test accuracy",
        )
        # 添加参数 --save-dir，设置保存输入数据的目录
        parser.add_argument(
            "--save-dir",
            type=str,
            default=save_dir,
            metavar="DIR",
            help="directory where saved inputs live",
        )
        # 添加参数 --no-save-dir，如果设置则将 'save_dir' 设置为 None
        parser.add_argument(
            "--no-save-dir",
            dest="save_dir",
            action="store_const",
            const=None,
            help="don't use any directory for saved inputs",
        )
        # 添加参数 --no-isolate，如果设置则禁用隔离功能
        parser.add_argument(
            "--no-isolate",
            dest="isolate",
            action="store_false",
            default=False,
            help="no isolate (doesn't do anything for after_dynamo)",
        )
        # 添加参数 --autocast，如果设置则启用 torch.cuda.amp.autocast
        parser.add_argument(
            "--autocast",
            default=autocast,
            action="store_true",
            help="use torch.cuda.amp.autocast",
        )
        # 添加参数 --no-autocast，如果设置则禁用 torch.cuda.amp.autocast
        parser.add_argument(
            "--no-autocast",
            dest="autocast",
            action="store_false",
            help="don't use torch.cuda.amp.autocast",
        )
        # 添加参数 --backend，设置 torch 编译使用的后端
        parser.add_argument(
            "--backend",
            type=str,
            default=backend,
            metavar="BACKEND",
            help="torch.compile backend to use",
        )

    # 添加子命令解析器，设置命令参数为 {run, minify} 中的一个，必须提供一个命令
    subparsers = parser.add_subparsers(
        dest="command", metavar="{run,minify}", required=True
    )

    # 添加 run 子命令解析器，用于运行 repro
    parser_run = subparsers.add_parser(
        "run",
        help="just run the repro",
    )
    # 添加常用参数设置到 run 命令解析器
    common_flags(parser_run)
    # 添加参数 --only-fwd，设置为 True 则不运行反向编译进行测试
    parser_run.add_argument(
        "--only-fwd",
        action="store_true",
        help="don't run backwards compilation for testing",
    )

    # 添加 minify 子命令解析器，用于运行 repro 的缩小版本
    parser_minify = subparsers.add_parser(
        "minify", help="run the minifier on the repro"
    )
    # 添加常用参数设置到 minify 命令解析器
    common_flags(parser_minify)

    # 解析命令行参数
    args = None
    if len(sys.argv) <= 1:
        args = [command, *sys.argv[1:]]

    options = parser.parse_args(args)
    
    # 定义命令与执行函数的映射关系
    COMMAND_FNS = {
        "minify": repro_minify,
        "run": repro_run,
    }
    # 调用对应命令的执行函数，并传递相应的选项、模块和加载参数
    COMMAND_FNS[options.command](options, mod, load_args)
```