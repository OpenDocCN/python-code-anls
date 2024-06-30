# `D:\src\scipysrc\sympy\bin\generate_module_list.py`

```
# 导入将来版本的 print 函数，确保兼容 Python 2.x 和 Python 3.x
from __future__ import print_function

# 导入 glob 模块，用于文件路径的模式匹配
from glob import glob

# 定义函数 get_paths，生成模块搜索路径集合
def get_paths(level=15):
    """
    生成用于模块搜索的路径集合。

    Examples
    ========

    >>> get_paths(2)
    ['sympy/__init__.py', 'sympy/*/__init__.py', 'sympy/*/*/__init__.py']
    >>> get_paths(6)
    ['sympy/__init__.py', 'sympy/*/__init__.py', 'sympy/*/*/__init__.py',
    'sympy/*/*/*/__init__.py', 'sympy/*/*/*/*/__init__.py',
    'sympy/*/*/*/*/*/__init__.py', 'sympy/*/*/*/*/*/*/__init__.py']

    """
    # 初始化通配符列表，起始于根目录"/"
    wildcards = ["/"]
    # 根据给定的层级生成递归通配符路径
    for i in range(level):
        wildcards.append(wildcards[-1] + "*/")
    # 生成完整的路径列表，加上 "__init__.py" 后缀
    p = ["sympy" + x + "__init__.py" for x in wildcards]
    return p

# 定义函数 generate_module_list，生成模块列表
def generate_module_list():
    # 初始化空列表 g 用于存储匹配到的模块路径
    g = []
    # 获取模块搜索路径集合
    for x in get_paths():
        # 使用 glob 函数获取所有匹配的文件路径
        g.extend(glob(x))
    # 将文件路径转换为模块名，并移除以 '.tests' 结尾的模块名
    g = [".".join(x.split("/")[:-1]) for x in g]
    g = [i for i in g if not i.endswith('.tests')]
    # 移除根模块 'sympy'
    g.remove('sympy')
    # 去重并排序模块列表
    g = list(set(g))
    g.sort()
    return g

# 如果脚本作为主程序执行，则生成模块列表并输出
if __name__ == '__main__':
    g = generate_module_list()
    # 打印模块列表的头部
    print("modules = [")
    # 遍历模块列表，逐行打印模块名
    for x in g:
        print("    '%s'," % x)
    # 打印模块列表的尾部
    print("]")
```