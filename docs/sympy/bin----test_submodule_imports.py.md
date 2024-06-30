# `D:\src\scipysrc\sympy\bin\test_submodule_imports.py`

```
#!/usr/bin/env python
"""
Test that

from sympy import *

only imports those sympy submodules that have names that are part of the
top-level namespace.
"""

import sys
import os

# hook in-tree SymPy into Python path, if possible

# 获取当前脚本的绝对路径
this_path = os.path.abspath(__file__)
# 获取当前脚本所在目录的路径
this_dir = os.path.dirname(this_path)
# 获取 sympy 的顶层目录路径
sympy_top = os.path.split(this_dir)[0]
# 构建 sympy 目录的完整路径
sympy_dir = os.path.join(sympy_top, 'sympy')

# 如果 sympy 目录存在，则将其路径添加到系统路径的最前面
if os.path.isdir(sympy_dir):
    sys.path.insert(0, sympy_top)

# 定义允许从 sympy 导入的子模块列表
submodule_whitelist = [
    'algebras',
    'assumptions',
    'calculus',
    'concrete',
    'core',
    'deprecated',
    'discrete',
    'external',
    'functions',
    'geometry',
    'integrals',
    'interactive',
    'logic',
    'matrices',
    'multipledispatch',
    'ntheory',
    'parsing',
    'plotting',
    'polys',
    'printing',
    'release',
    'series',
    'sets',
    'simplify',
    'solvers',
    'strategies',
    'tensor',
    'testing',
    'utilities',
]

# 定义函数，测试通过 'from sympy import *' 是否只导入了预期的子模块
def test_submodule_imports():
    # 如果已经导入了 'sympy' 模块，抛出运行时错误
    if 'sympy' in sys.modules:
        raise RuntimeError("SymPy has already been imported, the test_submodule_imports test cannot run")

    # 使用动态执行方式导入所有 sympy 子模块
    exec("from sympy import *", {})

    # 遍历已导入的所有模块
    for mod in sys.modules:
        # 如果模块名称不以 'sympy' 开头，跳过
        if not mod.startswith('sympy'):
            continue
        
        # 如果模块名称中不包含且仅包含一个点号，则继续
        if not mod.count('.') == 1:
            continue
        
        # 拆分模块名称，获取子模块名称
        _, submodule = mod.split('.')
        
        # 如果子模块不在允许的白名单中，则输出错误信息并退出
        if submodule not in submodule_whitelist:
            sys.exit(f"""\
Error: The submodule {mod} was imported with 'from sympy import *', but it was
not expected to be.

If {mod} is a new module that has functions that are imported at the
top-level, then the whitelist in bin/test_submodule_imports should be updated.
If it is not, the place that imports it should be modified so that it does not
get imported at the top-level, e.g., by moving the 'import {mod}' import
inside the function that uses it.

If you are unsure which code is importing {mod}, it may help to add 'raise
Exception' to sympy/{submodule}/__init__.py and observe the traceback from
running 'from sympy import *'.""")

    # 所有预期之内的子模块都正常导入，输出成功信息
    print("No unexpected submodules were imported with 'from sympy import *'")

# 如果当前脚本作为主程序运行，则执行测试子模块导入的函数
if __name__ == '__main__':
    test_submodule_imports()
```