# `.\pytorch\tools\coverage_plugins_package\src\coverage_plugins\jit_plugin.py`

```py
"""
This coverage plug-in attempts to cover JIT'd functions and methods that were previously missed in code coverage. Any
function and method that was passed through/decorated with torch.jit.script or torch.jit.script_method should now be
marked covered when coverage is run with this plug-in.

DISCLAIMER: note that this will mark the entire JIT'd function/method as covered without seeking proof that the
compiled code has been executed. This means that even if the code chunk is merely compiled and not run, it will get
marked as covered.
"""

# 从inspect模块中导入以下函数和类，用于检查对象的类型和源代码等信息
from inspect import (
    getsourcefile,
    getsourcelines,
    isclass,
    iscode,
    isfunction,
    ismethod,
    ismodule,
)
# 从time模块中导入time函数，用于生成基于时间的唯一性标识
from time import time
# 从typing模块中导入Any类型，用于指定对象的类型可以是任意类型
from typing import Any

# 从coverage模块中导入CoverageData和CoveragePlugin类，类型提示时忽略未引入的警告
from coverage import CoverageData, CoveragePlugin  # type: ignore[import]

# 创建一个CoverageData对象，使用当前时间作为其唯一标识的一部分，文件名的基础部分为'.coverage.jit.<时间戳>'
cov_data = CoverageData(basename=f".coverage.jit.{time()}")

def is_not_builtin_class(obj: Any) -> bool:
    # 检查对象是否为用户自定义的类且非内置类
    return isclass(obj) and not type(obj).__module__ == "builtins"

class JitPlugin(CoveragePlugin):  # type: ignore[misc, no-any-unimported]
    """
    dynamic_context is an overridden function that gives us access to every frame run during the coverage process. We
    look for when the function being run is `should_drop`, as all functions that get passed into `should_drop` will be
    compiled and thus should be marked as covered.
    """
    # JitPlugin类继承自CoveragePlugin类，用于覆盖其方法以实现特定的代码覆盖行为
    def dynamic_context(self, frame: Any) -> None:
        # 检查当前帧的函数名是否为 "should_drop"
        if frame.f_code.co_name == "should_drop":
            # 从帧的局部变量中获取 "fn" 对象
            obj = frame.f_locals["fn"]
            
            # 根据 inspect.getsourcefile 的文档要求，仅处理模块、类、方法、函数或代码对象，并排除内置模块或函数
            if (
                is_not_builtin_class(obj)
                or ismodule(obj)
                or ismethod(obj)
                or isfunction(obj)
                or iscode(obj)
            ):
                # 获取对象的源文件名
                filename = getsourcefile(obj)
                
                # 如果 filename 不为 None，则继续处理
                if filename:
                    # 尝试获取对象的源码行和起始行号
                    try:
                        sourcelines, starting_lineno = getsourcelines(obj)
                    except OSError:
                        pass
                    else:
                        # 构建行号范围的字典
                        line_data = {
                            filename: range(
                                starting_lineno, starting_lineno + len(sourcelines)
                            )
                        }
                        # 将行号范围添加到 cov_data 中
                        cov_data.add_lines(line_data)
        
        # 调用父类的 dynamic_context 方法处理帧
        super().dynamic_context(frame)
# 定义函数 coverage_init，用于初始化覆盖率分析
def coverage_init(reg: Any, options: Any) -> None:
    # 向给定的注册表 reg 添加动态上下文 JitPlugin 的实例
    reg.add_dynamic_context(JitPlugin())
```