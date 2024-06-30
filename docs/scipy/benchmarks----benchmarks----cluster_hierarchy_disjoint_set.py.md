# `D:\src\scipysrc\scipy\benchmarks\benchmarks\cluster_hierarchy_disjoint_set.py`

```
# 导入 NumPy 库并将其命名为 np
import numpy as np

# 尝试导入 scipy.cluster.hierarchy 中的 DisjointSet 类，如果导入失败则跳过
try:
    from scipy.cluster.hierarchy import DisjointSet
except ImportError:
    pass

# 从当前目录下的 common 模块中导入 Benchmark 类
from .common import Benchmark

# 定义 Bench 类，继承自 Benchmark 类
class Bench(Benchmark):
    # 参数化设置，参数是一个包含单个元素列表的列表，参数名称为 'n'
    params = [[100, 1000, 10000]]
    param_names = ['n']

    # 初始化设置方法，接受参数 n
    def setup(self, n):
        # 使用种子值为 0 的随机数生成器创建 rng 对象
        rng = np.random.RandomState(seed=0)
        
        # 创建包含 n 行 2 列的随机整数数组，范围在 [0, 10 * n) 之间
        self.edges = rng.randint(0, 10 * n, (n, 2))
        
        # 从 self.edges 数组中提取唯一的节点并排序
        self.nodes = np.unique(self.edges)
        
        # 使用 self.nodes 创建 DisjointSet 对象并赋值给 self.disjoint_set
        self.disjoint_set = DisjointSet(self.nodes)

        # 使用 self.nodes 创建 DisjointSet 对象并赋值给 self.pre_merged
        self.pre_merged = DisjointSet(self.nodes)
        
        # 遍历 self.edges 数组中的每对节点 a, b
        for a, b in self.edges:
            # 对 self.pre_merged 执行合并操作
            self.pre_merged.merge(a, b)

        # 使用 self.nodes 创建 DisjointSet 对象并赋值给 self.pre_merged_found
        self.pre_merged_found = DisjointSet(self.nodes)
        
        # 再次遍历 self.edges 数组中的每对节点 a, b
        for a, b in self.edges:
            # 对 self.pre_merged_found 执行合并操作
            self.pre_merged_found.merge(a, b)
        
        # 遍历 self.nodes 中的每个节点 x，并查找它们在 self.pre_merged_found 中的代表元素
        for x in self.nodes:
            self.pre_merged_found[x]

    # 定义 time_merge 方法，接受参数 n
    def time_merge(self, n):
        # 将 self.disjoint_set 赋值给 dis
        dis = self.disjoint_set
        # 遍历 self.edges 数组中的每对节点 a, b，并对 dis 执行合并操作
        for a, b in self.edges:
            dis.merge(a, b)

    # 定义 time_merge_already_merged 方法，接受参数 n
    def time_merge_already_merged(self, n):
        # 将 self.pre_merged 赋值给 dis
        dis = self.pre_merged
        # 遍历 self.edges 数组中的每对节点 a, b，并对 dis 执行合并操作
        for a, b in self.edges:
            dis.merge(a, b)

    # 定义 time_find 方法，接受参数 n
    def time_find(self, n):
        # 将 self.pre_merged 赋值给 dis
        dis = self.pre_merged
        # 返回包含所有 self.nodes 节点的代表元素的列表
        return [dis[i] for i in self.nodes]

    # 定义 time_find_already_found 方法，接受参数 n
    def time_find_already_found(self, n):
        # 将 self.pre_merged_found 赋值给 dis
        dis = self.pre_merged_found
        # 返回包含所有 self.nodes 节点的代表元素的列表
        return [dis[i] for i in self.nodes]

    # 定义 time_contains 方法，接受参数 n
    def time_contains(self, n):
        # 断言 self.pre_merged 包含 self.nodes 的第一个节点
        assert self.nodes[0] in self.pre_merged
        # 断言 self.pre_merged 包含 self.nodes 的中间节点
        assert self.nodes[n // 2] in self.pre_merged
        # 断言 self.pre_merged 包含 self.nodes 的最后一个节点
        assert self.nodes[-1] in self.pre_merged

    # 定义 time_absence 方法，接受参数 n
    def time_absence(self, n):
        # 断言 self.pre_merged 不包含 None
        assert None not in self.pre_merged
        # 断言 self.pre_merged 不包含 "dummy"
        assert "dummy" not in self.pre_merged
        # 断言 self.pre_merged 不包含元组 (1, 2, 3)
        assert (1, 2, 3) not in self.pre_merged
```