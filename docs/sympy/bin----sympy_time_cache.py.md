# `D:\src\scipysrc\sympy\bin\sympy_time_cache.py`

```
# 引入 __future__ 模块中的 print_function 特性，确保代码同时支持 Python 2 和 Python 3 的 print 函数
from __future__ import print_function

# 引入用于计时的 timeit 模块
import timeit

# 定义 TreeNode 类，表示树结构的节点
class TreeNode(object):
    # 初始化方法，设置节点名称和子节点列表，并初始化时间为 0
    def __init__(self, name):
        self._name = name
        self._children = []
        self._time = 0

    # 返回节点的字符串表示，包含名称和时间
    def __str__(self):
        return "%s: %s" % (self._name, self._time)

    # __repr__ 方法与 __str__ 方法相同
    __repr__ = __str__

    # 添加子节点
    def add_child(self, node):
        self._children.append(node)

    # 返回所有子节点列表
    def children(self):
        return self._children

    # 返回指定索引的子节点
    def child(self, i):
        return self.children()[i]

    # 设置节点的时间
    def set_time(self, time):
        self._time = time

    # 返回节点的时间
    def time(self):
        return self._time

    # 定义 total_time 方法，返回节点的时间
    total_time = time

    # 返回节点的独占时间，即节点的总时间减去所有子节点的时间总和
    def exclusive_time(self):
        return self.total_time() - sum(child.time() for child in self.children())

    # 返回节点的名称
    def name(self):
        return self._name

    # 将树结构展开为列表，按深度优先顺序返回节点和所有子节点的列表
    def linearize(self):
        res = [self]
        for child in self.children():
            res.extend(child.linearize())
        return res

    # 打印树的结构，每个节点根据其深度缩进输出
    def print_tree(self, level=0, max_depth=None):
        print("  "*level + str(self))
        if max_depth is not None and max_depth <= level:
            return
        for child in self.children():
            child.print_tree(level + 1, max_depth=max_depth)

    # 打印节点及其子节点中时间最长的 n 个节点和它们的时间
    def print_generic(self, n=50, method="time"):
        slowest = sorted((getattr(node, method)(), node.name()) for node in self.linearize())[-n:]
        for time, name in slowest[::-1]:
            print("%s %s" % (time, name))

    # 打印耗时最长的 n 个节点及其子节点
    def print_slowest(self, n=50):
        self.print_generic(n=50, method="time")

    # 打印独占时间最长的 n 个节点及其子节点
    def print_slowest_exclusive(self, n=50):
        self.print_generic(n, method="exclusive_time")

    # 将节点及其子节点的运行时间写入到指定文件中，格式为 CacheGrind 格式
    def write_cachegrind(self, f):
        # 如果 f 是字符串，将其打开为写入模式的文件对象
        if isinstance(f, str):
            f = open(f, "w")
            f.write("events: Microseconds\n")
            f.write("fl=sympyallimport\n")
            must_close = True
        else:
            must_close = False

        # 写入当前节点的名称和独占时间
        f.write("fn=%s\n" % self.name())
        f.write("1 %s\n" % self.exclusive_time())

        # 计数器，用于记录子节点的序号
        counter = 2
        # 遍历子节点，写入子节点的相关信息
        for child in self.children():
            f.write("cfn=%s\n" % child.name())
            f.write("calls=1 1\n")
            f.write("%s %s\n" % (counter, child.time()))
            counter += 1

        f.write("\n\n")

        # 递归调用每个子节点的 write_cachegrind 方法，将其运行时间写入文件
        for child in self.children():
            child.write_cachegrind(f)

        # 如果之前打开了文件，关闭文件对象
        if must_close:
            f.close()

# 创建一个 TreeNode 对象 pp，用于表示树的根节点
pp = TreeNode(None)  # We have to use pp since there is a sage function
                     #called parent that gets imported
seen = set()

# 自定义的导入函数，用于记录模块导入时的时间
def new_import(name, globals={}, locals={}, fromlist=[]):
    global pp
    # 如果模块已经导入过，直接返回原始的导入函数结果
    if name in seen:
        return old_import(name, globals, locals, fromlist)
    seen.add(name)

    # 创建一个新的 TreeNode 对象，表示当前导入的模块
    node = TreeNode(name)

    # 将新创建的节点添加为 pp 的子节点
    pp.add_child(node)
    old_pp = pp
    pp = node

    # 执行实际的模块导入操作
    t1 = timeit.default_timer()
    module = old_import(name, globals, locals, fromlist)
    t2 = timeit.default_timer()
    # 设置节点的运行时间，单位为微秒
    node.set_time(int(1000000*(t2 - t1)))

    pp = old_pp

    return module

# 保存原始的 __import__ 函数
old_import = __builtins__.__import__
# 将默认的 __import__ 函数替换为自定义的 new_import 函数
__builtins__.__import__ = new_import

# 保存内置函数 sum 的引用
old_sum = sum

# 导入 sympy 模块，忽略掉任何的导入警告
from sympy import *  # noqa

# 恢复原始的 sum 函数引用
sum = old_sum

# 从 pp 对象的第一个子对象中获取 sageall 对象
sageall = pp.child(0)

# 将 sageall 对象的数据写入到名为 "sympy.cachegrind" 的缓存分析文件中
sageall.write_cachegrind("sympy.cachegrind")

# 打印提示消息，告知用户定时数据已保存，建议如何查看
print("Timings saved. Do:\n$ kcachegrind sympy.cachegrind")
```