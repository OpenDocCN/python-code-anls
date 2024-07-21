# `.\pytorch\torch\_export\tools.py`

```
# 导入必要的库和模块
# mypy: allow-untyped-defs
import logging
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.export
import torch.export._trace
from torch._utils_internal import log_export_usage

# 获取日志记录器
log = logging.getLogger(__name__)

# 定义公开的函数列表
__all__ = ["report_exportability"]

# 生成用于子模块的输入
def _generate_inputs_for_submodules(
    model: torch.nn.Module,
    target_submodules: Iterable[str],
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[Any, Any]]:
    """
    为给定模型中的目标子模块生成输入。注意，如果两个子模块引用相同的对象，则此函数不起作用。

    Args:
        model: 根模型。
        inputs: 根模型的输入。
        target_submodules: 我们要为其生成输入的子模块。

    Returns:
        一个字典，将子模块名称映射到其输入。
    """
    # 如果kwargs为None，则初始化为空字典
    kwargs = kwargs or {}

    handles = []
    results = {}
    submodule_to_names = {mod: name for name, mod in model.named_modules()}

    def pre_forward(module, module_args, module_kwargs):
        results[submodule_to_names[module]] = (module_args, module_kwargs)

    try:
        for name, mod in model.named_modules():
            if name in target_submodules:
                handles.append(
                    mod.register_forward_pre_hook(pre_forward, with_kwargs=True)
                )
        model(*args, **kwargs)
    except Exception as e:
        # 捕获异常并发出警告
        warnings.warn(
            f"Failed to generate submodule inputs because of the following error:\n{e}"
        )
    finally:
        # 移除注册的hook
        for h in handles:
            h.remove()
    return results

# 报告模块的可导出性问题
def report_exportability(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    strict: bool = True,
    pre_dispatch: bool = False,
) -> Dict[str, Optional[Exception]]:
    """
    一次性报告模块的导出性问题。

    Args:
        mod: 根模块。
        args: 根模块的参数。
        kwargs: 根模块的关键字参数。
    Returns:
        一个字典，将子模块名称映射到尝试导出时引发的异常。
        `None`表示模块可以无问题导出。
    示例输出:
        {
            '': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_1': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_2': None
        }
    """

    # 记录导出使用情况
    log_export_usage(event="export.report_exportability")

    # 如果kwargs为None，则初始化为空字典
    kwargs = kwargs or {}

    # 获取所有子模块的名称
    all_submod_names = [name for name, _ in mod.named_modules() if name != ""]
    # 为所有子模块生成输入
    submod_inputs = _generate_inputs_for_submodules(mod, all_submod_names, args, kwargs)

    # 初始化报告字典
    report: Dict[str, Optional[Exception]] = {}
    # 定义一个函数 `try_export`，用于尝试导出给定模块及其子模块
    def try_export(module, module_name, args, kwargs):
        # 使用 nonlocal 关键字声明内部函数需要引用的外部变量
        nonlocal submod_inputs, report, strict, pre_dispatch
    
        # 如果参数 args 或 kwargs 不为 None
        if args is not None or kwargs is not None:
            try:
                # 调用 torch.export._trace._export 方法来导出模块
                torch.export._trace._export(
                    module,
                    args,
                    kwargs,
                    strict=strict,
                    pre_dispatch=pre_dispatch,
                )
                # 将模块名作为键，导出成功时设置为 None
                report[module_name] = None
                # 记录导出成功的信息
                log.info("Successfully exported `%s`", module_name)
                return  # 成功导出后直接返回
            except Exception as e:
                # 获取异常的简短描述
                short_msg = repr(e).split("\n")[0]
                # 记录导出失败的警告信息
                log.warning(
                    "Failed exporting `%s` with exception: %s", module_name, short_msg
                )
                # 将导出失败的异常记录到报告中
                report[module_name] = e
    
        # 遍历模块的子模块
        for name, submod in module.named_children():
            # 计算子模块的完整名称，以便递归调用 try_export
            sub_module_name = name if module_name == "" else f"{module_name}.{name}"
    
            # 获取子模块可能的输入参数
            submod_args, submod_kwargs = submod_inputs.get(
                sub_module_name, (None, None)
            )
    
            # 递归调用 try_export，尝试导出子模块
            try_export(submod, sub_module_name, submod_args, submod_kwargs)
    
        return  # 结束函数执行
    
    # 调用 try_export 函数，开始导出给定模块及其子模块
    try_export(mod, "", args, kwargs)
    
    # 创建一个集合来存储报告中所有非空异常的唯一描述
    unique_issues = set()
    for exception in report.values():
        if exception is not None:
            # 获取异常的唯一描述并添加到集合中
            key = repr(exception).split("\\n")[0]
            unique_issues.add(key)
    
    # 输出警告信息，指出发现的导出问题数量
    log.warning("Found %d export issues:", len(unique_issues))
    # 遍历集合，输出每个唯一问题的警告信息
    for issue in unique_issues:
        log.warning(issue)
    
    # 返回导出的报告
    return report
```