# `.\pytorch\benchmarks\instruction_counts\core\expand.py`

```
"""Logic for converting human-readable benchmarks into executable form.

This is mostly string manipulation, with just a bit of importlib magic.
"""
# 导入必要的模块和库
import importlib.abc
import importlib.util
import itertools as it
import os
import re
import textwrap
import uuid
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch  # 导入 PyTorch 库

if TYPE_CHECKING:
    # 如果在类型检查模式下，从 torch.utils.benchmark.utils.timer 导入 Language
    from torch.utils.benchmark.utils.timer import Language
else:
    # 否则直接从 torch.utils.benchmark 导入 Language
    from torch.utils.benchmark import Language

# 导入核心模块和类型定义
from core.api import AutogradMode, AutoLabels, GroupedBenchmark, RuntimeMode, TimerArgs
from core.types import FlatDefinition, FlatIntermediateDefinition, Label
from core.utils import get_temp_dir

# 定义所有可能的运行模式组合
_ALL_MODES = tuple(
    it.product(
        RuntimeMode,
        AutogradMode,
        Language,
    )
)


def _generate_torchscript_file(model_src: str, name: str) -> Optional[str]:
    """Returns the path a saved model if one can be constructed from `spec`.

    Because TorchScript requires actual source code in order to script a
    model, we can't simply `eval` an appropriate model string. Instead, we
    must write the correct source to a temporary Python file and then import
    the TorchScript model from that temporary file.

    `model_src` must contain `jit_model = ...`, which `materialize` will supply.
    """
    # Double check.
    # 确保 model_src 中包含 "jit_model = "，如果不包含则抛出断言错误
    assert "jit_model = " in model_src, f"Missing jit_model definition:\n{model_src}"

    # `torch.utils.benchmark.Timer` 会自动导入 torch，因此我们需要匹配该约定
    # 在 model_src 前加入 import torch
    model_src = f"import torch\n{model_src}"

    # 设置保存 TorchScript 模型文件的根目录
    model_root = os.path.join(get_temp_dir(), "TorchScript_models")
    os.makedirs(model_root, exist_ok=True)
    # 定义模块文件路径和 TorchScript 模型文件路径
    module_path = os.path.join(model_root, f"torchscript_{name}.py")
    artifact_path = os.path.join(model_root, f"torchscript_{name}.pt")

    # 检查模块文件是否已存在，如果存在则抛出值错误
    if os.path.exists(module_path):
        raise ValueError(f"File {module_path} already exists.")

    # 将 model_src 写入模块文件
    with open(module_path, "w") as f:
        f.write(model_src)

    # 使用 importlib 动态加载模块文件
    module_spec = importlib.util.spec_from_file_location(
        f"torchscript__{name}", module_path
    )
    assert module_spec is not None
    module = importlib.util.module_from_spec(module_spec)
    loader = module_spec.loader
    assert loader is not None

    loader.exec_module(module)

    # 断言模块中的 jit_model 属性是 ScriptFunction 或 ScriptModule 类型
    jit_model = module.jit_model  # type: ignore[attr-defined]
    assert isinstance(
        jit_model, (torch.jit.ScriptFunction, torch.jit.ScriptModule)
    ), f"Expected ScriptFunction or ScriptModule, got: {type(jit_model)}"
    
    # 将 TorchScript 模型保存为文件
    jit_model.save(artifact_path)  # type: ignore[call-arg]

    # 清理临时文件
    os.remove(module_path)
    return artifact_path


def _get_stmt(
    benchmark: GroupedBenchmark,
    # 定义一个变量 benchmark，类型为 GroupedBenchmark，可能是用于存储某种基准测试相关的数据或设置

    runtime: RuntimeMode,
    # 定义一个变量 runtime，类型为 RuntimeMode，可能用于控制程序的运行模式或环境配置

    autograd: AutogradMode,
    # 定义一个变量 autograd，类型为 AutogradMode，可能用于控制自动求导的行为或设置

    language: Language,
    # 定义一个变量 language，类型为 Language，可能用于存储或指定程序的语言相关信息
def specialize(benchmark: GroupedBenchmark,
               runtime: RuntimeMode,
               language: Language,
               autograd: AutogradMode,
               model_path: Optional[str]) -> Optional[str]:
    """Specialize a GroupedBenchmark for a particular configuration."""
    # 检查语言是否为 Python
    is_python = language == Language.PYTHON

    # 在 GroupedBenchmark 构建过程中，py_fwd_stmt 和 cpp_fwd_stmt 被设置为即时调用。
    # 因此在 RuntimeMode.EAGER 模式下，我们可以直接复用它们。
    # 在 RuntimeMode.JIT 模式下，我们需要生成适当的 `jit_model(...)` 调用。
    if runtime == RuntimeMode.EAGER:
        # 如果是 EAGER 模式，使用预先设置好的语句
        stmts = (benchmark.py_fwd_stmt, benchmark.cpp_fwd_stmt)
    else:
        # 否则，必须是 JIT 模式
        assert runtime == RuntimeMode.JIT
        assert benchmark.signature_args is not None
        # 调用 GroupedBenchmark 的静态方法生成模型调用语句
        stmts = GroupedBenchmark._make_model_invocation(
            benchmark.signature_args, benchmark.signature_output, RuntimeMode.JIT
        )

    # 根据当前语言选择适当的语句
    stmt = stmts[0 if is_python else 1]

    # 如果是 FORWARD_BACKWARD 自动微分模式，并且存在语句，则生成后向传播语句
    if autograd == AutogradMode.FORWARD_BACKWARD and stmt is not None:
        assert benchmark.signature_output is not None
        # 生成后向传播语句，并根据语言和运行时模式添加额外的调整
        backward = (
            f"{benchmark.signature_output}"
            f"{'.toTensor()' if runtime == RuntimeMode.JIT and language == Language.CPP else ''}"
            f".backward(){';' if language == Language.CPP else ''}"
        )
        stmt = f"{stmt}\n{backward}"
    return stmt


def _get_setup(benchmark: GroupedBenchmark,
               runtime: RuntimeMode,
               language: Language,
               stmt: str,
               model_path: Optional[str]) -> str:
    """Specialize a GroupedBenchmark for a particular configuration.

    Setup requires two extra pieces of information:
      1) The benchmark stmt. This is needed to warm up the model and avoid
         measuring lazy initialization.
      2) The model path so we can load it during the benchmark.

    These are only used when `runtime == RuntimeMode.JIT`.
    """

    # 到达这里时，关于如何设置模型的细节已由 GroupedBenchmark 决定（或在适当情况下设置为 None）。
    # 我们只需收集和打包代码块。
    if language == Language.PYTHON:
        setup = benchmark.setup.py_setup
        model_setup = benchmark.py_model_setup
    else:
        assert language == Language.CPP
        setup = benchmark.setup.cpp_setup
        model_setup = benchmark.cpp_model_setup

    # 如果是 EAGER 模式，返回设置代码和模型设置代码（如果存在）
    if runtime == RuntimeMode.EAGER:
        return "\n".join([setup, model_setup or ""])

    # 否则必须是 JIT 模式
    assert runtime == RuntimeMode.JIT
    assert model_path is not None

    # 模板中可能包含 `"{model_path}"`，因此引号会破坏模型加载。
    # 模型路径由基准测试生成，这只是一种谨慎的做法，而不是实践中预期的事情。
    assert '"' not in model_path

    # `stmt` 可能包含换行符，因此无法直接使用 f-strings，而是需要生成模板以便正确处理缩进。
    # 如果语言是 Python，则使用 Python 特定的模板格式化加载模型的代码
    if language == Language.PYTHON:
        # 设置模板字符串，用于加载 Torch JIT 模型
        setup_template: str = textwrap.dedent(
            f"""
            jit_model = torch.jit.load("{model_path}")

            # 对 `jit_model` 进行预热
            for _ in range(3):
            {{stmt}}
        """
        )

    # 如果语言不是 Python，则默认为 C++，使用 C++ 特定的模板格式化加载模型的代码
    else:
        assert language == Language.CPP
        # 设置模板字符串，用于加载 Torch JIT 模型
        setup_template = textwrap.dedent(
            f"""
            const std::string fpath = "{model_path}";
            auto jit_model = torch::jit::load(fpath);

            // 对 `jit_model` 进行预热
            for (int i = 0; i < 3; i++) {{{{
            {{stmt}}
            }}}}
        """
        )

    # 将 stmt 格式化后插入 setup_template 中，生成完整的模型加载代码
    model_load = setup_template.format(stmt=textwrap.indent(stmt, " " * 4))
    # 将 setup 和 model_load 组合成一个字符串，并以换行符分隔
    return "\n".join([setup, model_load])
# 将异构基准转换为可执行状态的函数
def materialize(benchmarks: FlatIntermediateDefinition) -> FlatDefinition:
    """Convert a heterogeneous benchmark into an executable state.

    This entails generation of TorchScript model artifacts, splitting
    GroupedBenchmarks into multiple TimerArgs, and tagging the results with
    AutoLabels.
    """
    # 存储最终结果的列表，每个元素是一个三元组(label, autolabels, timer_args)
    results: List[Tuple[Label, AutoLabels, TimerArgs]] = []

    # 遍历传入的基准数据字典
    for label, args in benchmarks.items():
        # 如果 args 是 TimerArgs 对象，则直接使用用户提供的参数，无需进一步处理
        if isinstance(args, TimerArgs):
            auto_labels = AutoLabels(
                RuntimeMode.EXPLICIT, AutogradMode.EXPLICIT, args.language
            )
            # 将结果添加到 results 列表中
            results.append((label, auto_labels, args))

        else:
            # 确保 args 是 GroupedBenchmark 类型
            assert isinstance(args, GroupedBenchmark)

            model_path: Optional[str] = None
            # 如果用户提供了 Python 模型设置并且需要 TorchScript
            if args.py_model_setup and args.torchscript:
                # 构建模型设置字符串，并生成 TorchScript 文件
                model_setup = (
                    f"{args.py_model_setup}\njit_model = torch.jit.script(model)"
                )
                # 为了调试方便，生成一个唯一的模型名称，用于 TorchScript 文件
                name: str = re.sub(r"[^a-z0-9_]", "_", "_".join(label).lower())
                name = f"{name}_{uuid.uuid4()}"
                model_path = _generate_torchscript_file(model_setup, name=name)

            # 遍历所有可能的运行时模式、自动微分模式、语言和线程数的组合
            for (runtime, autograd, language), num_threads in it.product(
                _ALL_MODES, args.num_threads
            ):
                # 如果运行时模式为显式或自动微分模式为显式，则跳过
                if runtime == RuntimeMode.EXPLICIT or autograd == AutogradMode.EXPLICIT:
                    continue

                # 如果运行时模式为 JIT 但不需要 TorchScript，则跳过
                if runtime == RuntimeMode.JIT and not args.torchscript:
                    continue

                # 如果自动微分模式为前向后向但不支持自动微分，则跳过
                if autograd == AutogradMode.FORWARD_BACKWARD and not args.autograd:
                    continue

                # 获取当前组合下的基准语句
                stmt = _get_stmt(args, runtime, autograd, language)
                if stmt is None:
                    continue

                # 获取当前组合下的基准设置
                setup = _get_setup(args, runtime, language, stmt, model_path)

                global_setup: str = ""
                # 如果语言是 C++ 并且运行时模式是 JIT，则添加全局设置
                if language == Language.CPP and runtime == RuntimeMode.JIT:
                    global_setup = textwrap.dedent(
                        """
                        #include <string>
                        #include <vector>
                        #include <torch/script.h>
                    """
                    )

                # 创建 AutoLabels 对象，用于描述当前组合的运行时、自动微分模式和语言
                autolabels = AutoLabels(runtime, autograd, language)
                # 创建 TimerArgs 对象，包含当前组合的基准语句、设置、全局设置和线程数
                timer_args = TimerArgs(
                    stmt=stmt,
                    setup=setup,
                    global_setup=global_setup,
                    num_threads=num_threads,
                    language=language,
                )

                # 将结果添加到 results 列表中
                results.append((label, autolabels, timer_args))

    # 返回结果列表
    return tuple(results)
```