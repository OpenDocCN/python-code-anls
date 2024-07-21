# `.\pytorch\torch\_inductor\__init__.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型定义和模块
from typing import Any, Dict, List, Optional, Tuple

# 导入 Torch 的 FX 模块和 pytree 模块
import torch.fx
import torch.utils._pytree as pytree

# 定义可以公开访问的模块成员列表
__all__ = ["compile", "list_mode_options", "list_options", "cudagraph_mark_step_begin"]


def compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    options: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    编译给定的 FX 图形与 TorchInductor。这允许编译
    使用 TorchDynamo 捕获的 FX 图形。

    Args:
        gm: 要编译的 FX 图形。
        example_inputs: 张量输入列表。
        options: 可选的配置选项字典。参见 `torch._inductor.config`。

    Returns:
        与 gm 具有相同行为但更快的可调用对象。
    """
    from .compile_fx import compile_fx

    # 调用编译函数，返回编译后的结果
    return compile_fx(gm, example_inputs, config_patches=options)


def aot_compile(
    gm: torch.fx.GraphModule,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    预先编译给定的 FX 图形与 TorchInductor 到共享库。

    Args:
        gm: 要编译的 FX 图形。
        args: 示例参数
        kwargs: 示例关键字参数
        options: 可选的配置选项字典。参见 `torch._inductor.config`。

    Returns:
        生成的共享库的路径
    """
    from .compile_fx import compile_fx_aot

    # 将 pytree 信息序列化为 .so 文件中的常量字符串
    in_spec = None
    out_spec = None

    # 如果 FX 图形使用 pytree 代码生成器，则提取输入输出规范
    if isinstance(gm.graph._codegen, torch.fx.graph._PyTreeCodeGen):
        codegen = gm.graph._codegen
        gm.graph._codegen = torch.fx.graph.CodeGen()
        gm.recompile()

        if codegen.pytree_info.in_spec is not None:
            in_spec = codegen.pytree_info.in_spec
        if codegen.pytree_info.out_spec is not None:
            out_spec = codegen.pytree_info.out_spec

    else:
        # 否则，尝试提取预定义的输入输出规范
        if hasattr(gm, "_in_spec"):
            in_spec = gm._in_spec
        if hasattr(gm, "_out_spec"):
            out_spec = gm._out_spec

    # 将输入输出规范序列化为字符串
    serialized_in_spec = pytree.treespec_dumps(in_spec) if in_spec is not None else ""
    serialized_out_spec = (
        pytree.treespec_dumps(out_spec) if out_spec is not None else ""
    )

    # 将参数展平，并获取输入张量
    flat_args_with_path, received_spec = pytree.tree_flatten_with_path(
        (args, kwargs or {})
    )
    flat_example_inputs = tuple(
        x[1] for x in flat_args_with_path if isinstance(x[1], torch.Tensor)
    )

    # 检查实际接收的规范是否与预期的输入规范匹配
    if in_spec is not None and received_spec != in_spec:
        raise ValueError(
            "Trying to flatten user inputs with exported input tree spec: \n"
            f"{in_spec}\n"
            "but actually got inputs with tree spec of: \n"
            f"{received_spec}"
        )
    # 根据条件设置选项字典，包括输入和输出的序列化规格
    options = (
        {
            "aot_inductor.serialized_in_spec": serialized_in_spec,
            "aot_inductor.serialized_out_spec": serialized_out_spec,
        }
        # 如果选项为 None，则使用默认选项字典，同时添加序列化规格信息
        if options is None
        else {
            **options,  # 使用 ** 操作符展开已有选项字典的内容
            "aot_inductor.serialized_in_spec": serialized_in_spec,
            "aot_inductor.serialized_out_spec": serialized_out_spec,
        }
    )
    
    # 调用编译 Ahead-of-Time (AOT) 的函数，传入模型、扁平化的示例输入列表以及配置选项
    return compile_fx_aot(
        gm,  # 模型对象
        list(flat_example_inputs),  # 扁平化的示例输入列表
        config_patches=options,  # 配置补丁，包括序列化规格等选项
    )
def list_mode_options(
    mode: Optional[str] = None, dynamic: Optional[bool] = None
) -> Dict[str, Any]:
    r"""Returns a dictionary describing the optimizations that each of the available
    modes passed to `torch.compile()` performs.

    Args:
        mode (str, optional): The mode to return the optimizations for.
        If None, returns optimizations for all modes
        dynamic (bool, optional): Whether dynamic shape is enabled.

    Example::
        >>> torch._inductor.list_mode_options()
    """

    mode_options: Dict[str, Dict[str, bool]] = {
        "default": {},  # 默认模式，无特定优化配置
        # 启用 cudagraphs
        "reduce-overhead": {
            "triton.cudagraphs": True,
        },
        # 启用 max-autotune，但不使用 cudagraphs
        "max-autotune-no-cudagraphs": {
            "max_autotune": True,
        },
        # 同时启用 max-autotune 和 cudagraphs
        "max-autotune": {
            "max_autotune": True,
            "triton.cudagraphs": True,
        },
    }
    return mode_options[mode] if mode else mode_options  # type: ignore[return-value]


def list_options() -> List[str]:
    r"""Returns a dictionary describing the optimizations and debug configurations
    that are available to `torch.compile()`.

    The options are documented in `torch._inductor.config`.

    Example::

        >>> torch._inductor.list_options()
    """

    from torch._inductor import config

    current_config: Dict[str, Any] = config.shallow_copy_dict()  # 获取当前配置的浅拷贝

    return list(current_config.keys())


def cudagraph_mark_step_begin():
    "Indicates that a new iteration of inference or training is about to begin."
    from .cudagraph_trees import mark_step_begin

    mark_step_begin()  # 标记推理或训练的新迭代开始
```