# `.\pytorch\torch\_inductor\exc.py`

```
# mypy: allow-untyped-defs
# 引入允许未类型化定义的标志，用于类型检查工具mypy

from __future__ import annotations
# 引入将来版本的注解支持，用于支持在类定义中使用自身类作为返回类型

import os
# 引入操作系统接口模块

import tempfile
# 引入用于创建临时文件和目录的模块

import textwrap
# 引入用于格式化文本的模块

from functools import lru_cache
# 从functools模块中引入最近最少使用缓存装饰器

if os.environ.get("TORCHINDUCTOR_WRITE_MISSING_OPS") == "1":
    # 如果环境变量TORCHINDUCTOR_WRITE_MISSING_OPS的值为"1"

    @lru_cache(None)
    # 应用最近最少使用缓存装饰器，缓存无参函数的结果
    def _record_missing_op(target):
        # 定义记录缺失操作的函数，参数为目标操作符
        with open(f"{tempfile.gettempdir()}/missing_ops.txt", "a") as fd:
            # 打开临时目录下的missing_ops.txt文件以追加模式
            fd.write(str(target) + "\n")
            # 写入目标操作符的字符串表示并换行

else:
    # 否则

    def _record_missing_op(target):  # type: ignore[misc]
        # 定义记录缺失操作的函数，忽略类型检查警告
        pass
        # 空操作，什么也不做

class OperatorIssue(RuntimeError):
    # 定义运行时错误的操作符问题类

    @staticmethod
    # 静态方法装饰器，定义静态方法

    def operator_str(target, args, kwargs):
        # 定义静态方法operator_str，接受目标、参数和关键字参数作为输入
        lines = [f"target: {target}"] + [
            f"args[{i}]: {arg}" for i, arg in enumerate(args)
            # 构建行列表，包含目标和参数的字符串表示
        ]
        if kwargs:
            # 如果存在关键字参数
            lines.append(f"kwargs: {kwargs}")
            # 添加关键字参数的字符串表示
        return textwrap.indent("\n".join(lines), "  ")
        # 返回缩进处理后的行列表字符串表示

class MissingOperatorWithoutDecomp(OperatorIssue):
    # 定义没有分解的缺失操作符问题类，继承自OperatorIssue类

    def __init__(self, target, args, kwargs):
        # 定义初始化方法，接受目标、参数和关键字参数作为输入
        _record_missing_op(target)
        # 记录缺失操作
        super().__init__(f"missing lowering\n{self.operator_str(target, args, kwargs)}")
        # 调用父类构造函数，传递生成的错误消息字符串

class MissingOperatorWithDecomp(OperatorIssue):
    # 定义带有分解的缺失操作符问题类，继承自OperatorIssue类

    def __init__(self, target, args, kwargs):
        # 定义初始化方法，接受目标、参数和关键字参数作为输入
        _record_missing_op(target)
        # 记录缺失操作
        super().__init__(
            f"missing decomposition\n{self.operator_str(target, args, kwargs)}"
            + textwrap.dedent(
                f"""
                
                There is a decomposition available for {target} in
                torch._decomp.get_decompositions().  Please add this operator to the
                `decompositions` list in torch._inductor.decomposition
                """
            )
        )
        # 调用父类构造函数，传递生成的错误消息字符串和详细信息

class LoweringException(OperatorIssue):
    # 定义降级异常类，继承自OperatorIssue类

    def __init__(self, exc: Exception, target, args, kwargs):
        # 定义初始化方法，接受异常对象、目标、参数和关键字参数作为输入
        super().__init__(
            f"{type(exc).__name__}: {exc}\n{self.operator_str(target, args, kwargs)}"
        )
        # 调用父类构造函数，传递生成的错误消息字符串和详细信息

class SubgraphLoweringException(RuntimeError):
    # 定义子图降级异常类，继承自运行时错误类

    pass
    # 空操作，什么也不做

class InvalidCxxCompiler(RuntimeError):
    # 定义无效C++编译器异常类，继承自运行时错误类

    def __init__(self):
        from . import config
        # 从当前目录中的config模块中导入

        super().__init__(
            f"No working C++ compiler found in {config.__name__}.cpp.cxx: {config.cpp.cxx}"
        )
        # 调用父类构造函数，传递生成的错误消息字符串

class CppWrapperCodeGenError(RuntimeError):
    # 定义C++包装器代码生成错误异常类，继承自运行时错误类

    def __init__(self, msg: str):
        # 定义初始化方法，接受错误消息字符串作为输入
        super().__init__(f"C++ wrapper codegen error: {msg}")
        # 调用父类构造函数，传递生成的错误消息字符串

class CppCompileError(RuntimeError):
    # 定义C++编译错误异常类，继承自运行时错误类

    def __init__(self, cmd: list[str], output: str):
        # 定义初始化方法，接受命令列表和输出字符串作为输入
        if isinstance(output, bytes):
            # 如果输出为字节流
            output = output.decode("utf-8")
            # 解码为UTF-8字符串

        super().__init__(
            textwrap.dedent(
                """
                    C++ compile error

                    Command:
                    {cmd}

                    Output:
                    {output}
                """
            )
            .strip()
            .format(cmd=" ".join(cmd), output=output)
        )
        # 调用父类构造函数，传递生成的格式化错误消息字符串

class CUDACompileError(CppCompileError):
    # 定义CUDA编译错误异常类，继承自C++编译错误异常类

    pass
    # 空操作，什么也不做
```