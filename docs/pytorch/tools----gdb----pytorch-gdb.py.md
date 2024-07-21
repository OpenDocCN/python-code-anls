# `.\pytorch\tools\gdb\pytorch-gdb.py`

```
import textwrap  # 导入textwrap模块，用于处理字符串的格式化显示
from typing import Any  # 导入Any类型，用于支持任意类型的数据

import gdb  # type: ignore[import]  # 导入gdb调试器模块，这里忽略类型检查


class DisableBreakpoints:
    """
    Context-manager to temporarily disable all gdb breakpoints, useful if
    there is a risk to hit one during the evaluation of one of our custom
    commands
    """

    def __enter__(self) -> None:
        # 初始化一个空列表，用于存储被禁用的断点对象
        self.disabled_breakpoints = []
        # 遍历当前所有的断点对象
        for b in gdb.breakpoints():
            # 如果断点是启用状态，则禁用它
            if b.enabled:
                b.enabled = False
                # 将禁用的断点对象添加到列表中
                self.disabled_breakpoints.append(b)

    def __exit__(self, etype: Any, evalue: Any, tb: Any) -> None:
        # 恢复之前被禁用的断点状态
        for b in self.disabled_breakpoints:
            b.enabled = True


class TensorRepr(gdb.Command):  # type: ignore[misc, no-any-unimported]
    """
    Print a human readable representation of the given at::Tensor.
    Usage: torch-tensor-repr EXP

    at::Tensor instances do not have a C++ implementation of a repr method: in
    pytorch, this is done by pure-Python code. As such, torch-tensor-repr
    internally creates a Python wrapper for the given tensor and call repr()
    on it.
    """

    __doc__ = textwrap.dedent(__doc__).strip()  # 格式化类的文档字符串，去除首尾空白

    def __init__(self) -> None:
        # 初始化TensorRepr命令，设置名称、命令类型、完成类型
        gdb.Command.__init__(
            self, "torch-tensor-repr", gdb.COMMAND_USER, gdb.COMPLETE_EXPRESSION
        )

    def invoke(self, args: str, from_tty: bool) -> None:
        # 将输入参数解析为列表
        args = gdb.string_to_argv(args)
        # 如果参数数量不等于1，输出用法信息并返回
        if len(args) != 1:
            print("Usage: torch-tensor-repr EXP")
            return
        # 取出第一个参数作为张量名称
        name = args[0]
        # 使用DisableBreakpoints上下文管理器禁用断点
        with DisableBreakpoints():
            # 调用torch::gdb::tensor_repr函数获取张量的Python级别表示
            res = gdb.parse_and_eval(f"torch::gdb::tensor_repr({name})")
            # 打印张量的Python级别表示信息
            print(f"Python-level repr of {name}:")
            # 打印张量的具体字符串表示
            print(res.string())
            # 释放torch::gdb::tensor_repr返回的malloc分配的内存
            gdb.parse_and_eval(f"(void)free({int(res)})")


TensorRepr()  # 创建TensorRepr实例，注册为gdb命令
```