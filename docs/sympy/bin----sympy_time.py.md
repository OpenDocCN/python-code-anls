# `D:\src\scipysrc\sympy\bin\sympy_time.py`

```
from __future__ import print_function
# 导入 Python 未来支持的 print 函数特性

import time
# 导入时间模块，用于计时操作

from get_sympy import path_hack
# 导入自定义模块 get_sympy 中的 path_hack 函数

path_hack()
# 调用 get_sympy 模块中的 path_hack 函数，可能用于路径配置或环境设置

seen = set()
# 创建一个空集合 seen，用于记录已经导入过的模块名

import_order = []
# 创建一个空列表 import_order，用于记录模块的导入顺序及其层级关系

elapsed_times = {}
# 创建一个空字典 elapsed_times，用于记录每个模块导入所花费的时间

level = 0
# 初始化变量 level，用于记录模块的层级

parent = None
# 初始化变量 parent，用于记录当前模块的父模块

children = {}
# 创建一个空字典 children，用于记录每个模块的子模块

def new_import(name, globals={}, locals={}, fromlist=[]):
    # 自定义的模块导入函数 new_import，用于替换原有的 __import__ 函数
    global level, parent
    # 声明要修改的全局变量
    if name in seen:
        # 如果模块名已经在 seen 集合中，则直接调用原有的导入函数
        return old_import(name, globals, locals, fromlist)
    seen.add(name)
    # 将模块名添加到 seen 集合中，表示已经导入过
    import_order.append((name, level, parent))
    # 将模块名、层级和父模块元组添加到 import_order 列表中，记录导入顺序和层级关系
    t1 = time.time()
    # 记录导入开始时间
    old_parent = parent
    # 保存旧的父模块名
    parent = name
    # 更新当前父模块名为新导入的模块名
    level += 1
    # 增加层级深度
    module = old_import(name, globals, locals, fromlist)
    # 调用原有的 __import__ 函数导入模块
    level -= 1
    # 恢复层级深度
    parent = old_parent
    # 恢复父模块名为旧值
    t2 = time.time()
    # 记录导入结束时间
    elapsed_times[name] = t2 - t1
    # 计算导入时间并保存到 elapsed_times 字典中
    return module
    # 返回导入的模块对象

old_import = __builtins__.__import__
# 保存原始的 __import__ 函数引用

__builtins__.__import__ = new_import
# 将自定义的 new_import 函数赋值给 __import__，实现模块导入的拦截和记录

from sympy import *
# 导入 sympy 模块中的所有内容

parents = {}
# 创建一个空字典 parents，用于记录每个模块的父模块名

is_parent = {}
# 创建一个空字典 is_parent，用于记录哪些模块是其他模块的父模块

for name, level, parent in import_order:
    # 遍历 import_order 列表中的每个元组
    parents[name] = parent
    # 将模块名与父模块名的映射关系存入 parents 字典中
    is_parent[parent] = True
    # 标记父模块名为 True，表示其为其他模块的父模块

print("== Tree ==")
# 输出导入树形结构的提示信息

for name, level, parent in import_order:
    # 再次遍历 import_order 列表中的每个元组
    print("%s%s: %.3f (%s)" % (" "*level, name, elapsed_times.get(name, 0),
            parent))
    # 按照格式输出模块名、层级、导入时间和父模块名

print("\n")
# 输出一个空行

print("== Slowest (including children) ==")
# 输出显示最慢导入模块的提示信息

slowest = sorted((t, name) for (name, t) in elapsed_times.items())[-50:]
# 找出最耗时的 50 个模块，并按时间排序

for elapsed_time, name in slowest[::-1]:
    # 反向遍历耗时最长的模块列表
    print("%.3f %s (%s)" % (elapsed_time, name, parents[name]))
    # 按格式输出耗时、模块名和其父模块名
```