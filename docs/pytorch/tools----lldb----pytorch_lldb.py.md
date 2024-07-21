# `.\pytorch\tools\lldb\pytorch_lldb.py`

```
from typing import Any

import lldb  # type: ignore[import]


def get_target() -> Any:
    # 获取当前调试器中选定的目标
    target = lldb.debugger.GetSelectedTarget()
    if not target:
        # 如果没有目标可用，则输出错误信息并返回 None
        print("[-] error: no target available. please add a target to lldb.")
        return None
    return target


class DisableBreakpoints:
    """
    Context-manager to temporarily disable all lldb breakpoints, useful if
    there is a risk to hit one during the evaluation of one of our custom
    commands
    """

    def __enter__(self) -> None:
        # 进入上下文管理器时禁用所有断点
        target = get_target()

        if target.DisableAllBreakpoints() is False:
            # 如果禁用断点失败，则输出错误信息
            print("[-] error: failed to disable all breakpoints.")

    def __exit__(self, etype: Any, evalue: Any, tb: Any) -> None:
        # 退出上下文管理器时重新启用所有断点
        target = get_target()

        if target.EnableAllBreakpoints() is False:
            # 如果启用断点失败，则输出错误信息
            print("[-] error: failed to enable all breakpoints.")


def IntArrayRef_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    """Print human readable representation of c10::IntArrayRef"""
    with DisableBreakpoints():
        # 使用上下文管理器禁用断点
        target = get_target()
        tensor = valobj.GetName()
        # 使用 LLDB 执行表达式来获取 IntArrayRef 的字符串表示
        result = target.EvaluateExpression(
            f"torch::gdb::int_array_ref_string({tensor})"
        )
        str_result = str(result)
        str_result = str_result[str_result.find('"') + 1 : -1]
        return str_result


def DispatchKeyset_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    """Print human readable representation of c10::DispatchKeyset"""
    with DisableBreakpoints():
        # 使用上下文管理器禁用断点
        target = get_target()
        keyset = valobj.GetName()
        # 使用 LLDB 执行表达式来获取 DispatchKeyset 的字符串表示
        result = target.EvaluateExpression(
            f"torch::gdb::dispatch_keyset_string({keyset})"
        )
        str_result = str(result)
        str_result = str_result[str_result.find('"') + 1 : -1]
        return str_result


def Tensor_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    """Print a human readable representation of the given at::Tensor.

    at::Tensor instances do not have a C++ implementation of a repr method: in
    pytorch, this is done by pure-Python code. As such, print <tensor>
    internally creates a Python wrapper for the given tensor and call repr()
    on it.
    Usage:
        print self
    """
    with DisableBreakpoints():
        # 使用上下文管理器禁用断点
        target = get_target()
        tensor = valobj.GetName()
        # 使用 LLDB 执行表达式来获取 Tensor 的字符串表示
        result = target.EvaluateExpression(f"torch::gdb::tensor_repr({tensor})")
        str_result = str(result)
        # 释放 result 内存
        target.EvaluateExpression(f"(void)free({result.GetValue()})")
        str_result = "\n" + str_result[str_result.find("tensor") : -1]
        return str_result


# 初始化代码以添加自定义命令
def __lldb_init_module(debugger: Any, internal_dict: Any) -> Any:
    debugger.HandleCommand(
        "type summary add c10::IntArrayRef -F pytorch_lldb.IntArrayRef_summary -w torch"
    )
    debugger.HandleCommand(
        "type summary add c10::DispatchKeySet -F pytorch_lldb.DispatchKeyset_summary -w torch"
    )
    )
    # 使用 LLDB 的调试器命令接口来添加对于 at::Tensor 类型的自定义摘要函数 pytorch_lldb.Tensor_summary，并且指定该摘要函数所属的类型是 torch
    debugger.HandleCommand(
        "type summary add at::Tensor -F pytorch_lldb.Tensor_summary -w torch"
    )
    # 打印消息，提示已经安装了用于 PyTorch AT 类型的 LLDB 摘要打印，并且已经准备好使用
    # 默认情况下，此类别已启用。要禁用，请运行: `type category disable torch`
    print(
        "Pretty Printing lldb summary for PyTorch AT types has been installed and is ready for use. "
        "This category is enabled by default. To disable run: `type category disable torch`"
    )
    # 打印用法说明
    print(
        "Usage:\n\tprint <at::tensor>\n\tprint <c10::IntArrayRef>\n\tprint <c10::DispatchKeySet>"
    )
```