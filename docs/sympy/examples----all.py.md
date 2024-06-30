# `D:\src\scipysrc\sympy\examples\all.py`

```
#!/usr/bin/env python

DESCRIPTION = """
Runs all the examples for testing purposes and reports successes and failures
to stderr.  An example is marked successful if the running thread does not
throw an exception, for threaded examples, such as plotting, one needs to
check the stderr messages as well.
"""

EPILOG = """
Example Usage:
   When no examples fail:
     $ ./all.py > out
     SUCCESSFUL:
       - beginner.basic
       [...]
     NO FAILED EXAMPLES
     $

   When examples fail:
     $ ./all.py -w > out
     Traceback (most recent call last):
       File "./all.py", line 111, in run_examples
     [...]
     SUCCESSFUL:
       - beginner.basic
       [...]
     FAILED:
       - intermediate.mplot2D
       [...]
     $

   Obviously, we want to achieve the first result.
"""

import optparse
import os
import sys
import traceback

# add local sympy to the module path
this_file = os.path.abspath(__file__)
sympy_dir = os.path.join(os.path.dirname(this_file), "..")
sympy_dir = os.path.normpath(sympy_dir)
sys.path.insert(0, sympy_dir)
import sympy

TERMINAL_EXAMPLES = [
    "beginner.basic",
    "beginner.differentiation",
    "beginner.expansion",
    "beginner.functions",
    "beginner.limits_examples",
    "beginner.precision",
    "beginner.print_pretty",
    "beginner.series",
    "beginner.substitution",
    "intermediate.coupled_cluster",
    "intermediate.differential_equations",
    "intermediate.infinite_1d_box",
    "intermediate.partial_differential_eqs",
    "intermediate.trees",
    "intermediate.vandermonde",
    "advanced.curvilinear_coordinates",
    "advanced.dense_coding_example",
    "advanced.fem",
    "advanced.gibbs_phenomenon",
    "advanced.grover_example",
    "advanced.hydrogen",
    "advanced.pidigits",
    "advanced.qft",
    "advanced.relativity",
]

WINDOWED_EXAMPLES = [
    "beginner.plotting_nice_plot",
    "intermediate.mplot2d",
    "intermediate.mplot3d",
    "intermediate.print_gtk",
    "advanced.autowrap_integrators",
    "advanced.autowrap_ufuncify",
    "advanced.pyglet_plotting",
]

EXAMPLE_DIR = os.path.dirname(__file__)

def load_example_module(example):
    """Loads modules based upon the given package name"""
    from importlib import import_module

    # 获取示例目录的基础模块名
    exmod = os.path.split(EXAMPLE_DIR)[1]
    # 构造完整的示例模块名
    modname = exmod + '.' + example
    # 导入并返回模块对象
    return import_module(modname)


def run_examples(*, windowed=False, quiet=False, summary=True):
    """Run all examples in the list of modules.

    Returns a boolean value indicating whether all the examples were
    successful.
    """
    successes = []
    failures = []
    examples = TERMINAL_EXAMPLES
    # 如果指定了 windowed 参数，添加窗口化示例到列表中
    if windowed:
        examples += WINDOWED_EXAMPLES

    if quiet:
        # 如果 quiet 为 True，使用 PyTestReporter 类输出测试结果
        from sympy.testing.runtests import PyTestReporter
        reporter = PyTestReporter()
        reporter.write("Testing Examples\n")
        reporter.write("-" * reporter.terminal_width)
    else:
        reporter = None
    # 遍历给定的例子列表
    for example in examples:
        # 对每个例子运行测试，并根据测试结果将其添加到成功或失败的列表中
        if run_example(example, reporter=reporter):
            successes.append(example)
        else:
            failures.append(example)

    # 如果需要显示总结信息
    if summary:
        # 显示测试成功和失败的总结，使用给定的报告器对象
        show_summary(successes, failures, reporter=reporter)

    # 返回测试失败的例子数量是否为零（True表示所有测试都成功）
    return len(failures) == 0
# 定义函数 `run_example`，用于运行特定的示例代码，并返回是否成功的布尔值
def run_example(example, *, reporter=None):
    """Run a specific example.

    Returns a boolean value indicating whether the example was successful.
    """
    # 如果提供了 reporter 参数，则将 example 写入 reporter
    if reporter:
        reporter.write(example)
    else:
        # 否则，打印分隔线和正在运行的 example 名称
        print("=" * 79)
        print("Running: ", example)

    try:
        # 加载示例对应的模块
        mod = load_example_module(example)
        # 如果提供了 reporter，则抑制 mod.main 函数的输出并将 "[PASS]" 写入 reporter
        if reporter:
            suppress_output(mod.main)
            reporter.write("[PASS]", "Green", align="right")
        else:
            # 否则直接运行 mod.main 函数
            mod.main()
        # 返回示例运行成功的标志 True
        return True
    except KeyboardInterrupt as e:
        # 如果捕获到 KeyboardInterrupt，则重新抛出
        raise e
    except:
        # 捕获所有其他异常
        if reporter:
            # 如果提供了 reporter，则写入 "[FAIL]" 和异常信息到 reporter
            reporter.write("[FAIL]", "Red", align="right")
        # 打印异常的堆栈信息
        traceback.print_exc()
        # 返回示例运行失败的标志 False
        return False
```