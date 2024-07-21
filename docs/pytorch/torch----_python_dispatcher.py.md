# `.\pytorch\torch\_python_dispatcher.py`

```py
# 导入正则表达式模块
import re

# 导入 torch._C 模块，命名为 C
import torch._C as C

"""
PythonDispatcher 类是一个对 C++ 调度器的轻量级 Python 绑定，用于展示调度器预计算的工作方式。
特别地，它展示了对于某个操作 'foo'，用户在特定的调度键上注册内核后，计算得到的调度表是什么样子的。

在实际的 C++ 调度器中，我们支持多个调度键来支持不同的功能。为简单起见，PythonDispatcher
仅支持每个用例的单个示例调度键。这些用例如下：

- CPU/AutogradCPU: 表示内部后端，在 PyTorch 核心库中通常有专门的推理和自动求导内核。
    例如 CPU, CUDA
- FPGA/AutogradOther: 表示内部后端，在 PyTorch 核心库中通常有特定于后端的推理内核，
    但它们共享 AutogradOther 中指定的相同自动求导内核。
    例如 FPGA, SparseCsrCPU
- XLA/AutogradXLA: 表示外部后端，在 PyTorch 核心库中既没有推理内核也没有自动求导内核。
    后端所有者负责在其扩展（如 torch-xla）中注册操作符所支持的推理和自动求导内核。
    例如 XLA, XPU, MPS
- CompositeExplicitAutograd: 别名键，映射到所有后端的推理内核，如 CPU, CUDA, XLA 等。
    注册到此键的内核必须适用于所有后端的推理。
- Autograd: 别名键，映射到所有后端的自动求导内核，如 AutogradCPU, AutogradXLA, AutogradOther。
    注册到此键的内核必须适用于所有后端的自动求导。
- CompositeImplicitAutograd: 别名键，CompositeImplicitAutograd = CompositeExplicitAutograd + Autograd
    注册到此键的内核必须适用于所有后端的推理和自动求导。

请注意，我们仅允许在 PyTorch 核心库中注册别名键。例如，不应该从 torch-xla 扩展中注册
CompositeImplicitAutograd 或 CompositeExplicitAutograd 内核，而应该将内核贡献到
pytorch/pytorch 仓库，以便所有后端都能使用，并在没有扩展的情况下持续测试。

用法:
  dispatcher = PythonDispatcher()
  dispatcher.register(["CPU", "XLA", "CompositeImplicitAutograd"])
  print(dispatcher.dispatchTable()) # 这会告诉您对于特定后端使用哪个内核。
  # 获取更多调试信息
  # print(dispatcher.keys())
  # print(dispatcher.registrations())
  # print(dispatcher.rawRegistrations())
  # print(dispatcher.rawDispatchTable())
PythonDispatcher 在底层调用 C++ 调度器进行调度表的预计算。
本文件仅为开发者提供简化的 API，相关的测试代码位于 test/test_dispatch.py
"""


class PythonDispatcher:
    # 命名空间设为 "__test__"
    namespace = "__test__"
    # 名称设为 "foo"
    name = "foo"
    # 运行时键列表
    # fmt: off
    runtime_keys = [
        "CPU", "AutogradCPU",
        "FPGA", "AutogradOther",
        "XLA", "AutogradXLA",
        "Lazy", "AutogradLazy",
    ]
    # fmt: on
    # 定义别名键列表，用于标识支持的分发键
    alias_keys = [
        "CompositeExplicitAutograd",
        "Autograd",
        "CompositeImplicitAutograd",
    ]
    # 将运行时键与别名键合并，形成完整的支持键列表
    supported_keys = runtime_keys + alias_keys

    # 初始化方法，检查不变性并设置库引用
    def __init__(self):
        # 检查不变性条件
        C._dispatch_check_invariants(self.name)  # type: ignore[attr-defined]
        # 获取 FRAGMENT 类型的库引用
        self.ref = C._dispatch_library("FRAGMENT", self.namespace, "")
        # 定义一个名为 foo 的函数接口，接收参数为 Tensor 类型并返回 Tensor

    """
    返回 PythonDispatcher 支持的分发键列表。
    您可以向这些键注册内核。
    """

    # 返回支持的分发键列表
    def keys(self):
        return self.supported_keys

    """
    向目标分发键注册内核。
    dispatchKeys(list[str]): 您想要注册自己内核的分发键列表。
      请注意，在 PythonDispatcher 中您无需自己编写内核。例如，对于 CPU 分发键，
      一个针对 CPU 的内核（例如 fn_CPU）将自动生成并注册。
    """

    # 注册内核到目标分发键
    def register(self, dispatchKeys):
        # 检查是否有重复的分发键，重复的情况会触发警告
        if len(set(dispatchKeys)) != len(dispatchKeys):
            raise RuntimeError(
                f"Overriden is not allowed but found duplicates in {dispatchKeys}."
            )
        # 在代码生成阶段禁止以下情况，而非在 C++ 分发器中
        if (
            "CompositeImplicitAutograd" in dispatchKeys
            and "CompositeExplicitAutograd" in dispatchKeys
        ):
            raise RuntimeError(
                "Registration to both CompositeImplicitAutograd and CompositeExplicitAutograd is not allowed."
            )
        # 检查每个分发键是否属于支持的键列表
        for key in dispatchKeys:
            if key not in self.supported_keys:
                raise RuntimeError(
                    f"{key} is not supported, please select a dispatch key in {self.supported_keys}."
                )
            # 在库引用中实现针对特定分发键的函数
            self.ref.impl_t_t("foo", dispatch=key, debug="fn_" + key)

    """
    格式化（键，内核）的辅助函数。
    """

    # 格式化输出键和内核的行
    def _format_line(self, key, kernel):
        return f"{key:<15} {kernel}\n"

    """
    打印表头的辅助函数。
    """

    # 格式化输出表头
    def _format_header(self, header):
        s = f"""
    """
    Returns raw output of all registration info for debugging only.
    Use registrations() for a simplified version.
    """
    # 返回所有注册信息的原始输出，仅用于调试目的
    # 使用 registrations() 获取简化版本的信息

    def rawRegistrations(self):
        # 调用 C 类的 _dispatch_dump 方法，传入命名空间和名称参数
        return C._dispatch_dump(f"{self.namespace}::{self.name}")  # type: ignore[attr-defined]

    """
    Returns raw output of computed dispatch table for debugging only.
    Use dispatchTable() for a simplified version.
    """
    # 返回计算得到的调度表的原始输出，仅用于调试目的
    # 使用 dispatchTable() 获取简化版本的信息

    def rawDispatchTable(self):
        # 调用 C 类的 _dispatch_dump_table 方法，传入命名空间和名称参数
        return C._dispatch_dump_table(f"{self.namespace}::{self.name}")  # type: ignore[attr-defined]

    """
    Returns a table(str) including all the registrations from users.
    Note this includes registrations to both runtime keys and alias keys.
    """
    # 返回一个包含所有用户注册信息的表格字符串
    # 注意，这包括运行时键和别名键的注册信息

    def registrations(self):
        # 生成标题行并初始化输出
        output = self._format_header("Registered Kernels")
        # 获取原始注册信息的输出
        state = self.rawRegistrations()
        # 按行拆分状态信息
        state_entries = state.split("\n")
        # 遍历每一行信息
        for line in state_entries:
            # 提取第一个冒号之前的部分作为键
            first = line.split(":")[0]
            # 检查键是否以支持的任一关键字开头
            if any(first.startswith(k) for k in self.supported_keys):
                # 从行中提取核心名称
                kernel = line.split("::")[0].split(" ")[1]
                # 格式化并添加到输出
                output += self._format_line(first, kernel)
        return output

    """
    Returns the computed dispatch table(str). Note this only include
    runtime keys, registrations to alias keys have been decoded to their
    mapped runtime keys.
    """
    # 返回计算得到的调度表格的字符串表示
    # 注意，这仅包括运行时键，对别名键的注册已解码为它们映射的运行时键

    def dispatchTable(self):
        # 生成标题行并初始化输出
        output = self._format_header("Computed Dispatch Table")
        # 获取原始调度表的输出
        table = self.rawDispatchTable()
        # 按行拆分调度表信息
        table_entries = table.split("\n")
        # 创建用于匹配 FallbackKernel.cpp 的正则表达式
        regex = re.compile(r"registered at .*FallbackKernel\.cpp.*(\[)")
        # 遍历每一行调度表信息
        for line in table_entries:
            # 提取第一个冒号之前的部分作为键
            k = line.split(":")[0]
            # 如果键存在于运行时键中
            if k in self.runtime_keys:
                # 使用正则表达式替换条目中的匹配部分
                entry = regex.sub("[", line)
                # 格式化并添加到输出
                output += self._format_line(k, entry.split(": ")[1])
        return output
```