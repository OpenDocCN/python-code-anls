# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_dbscan_inner.pyx`

```
# Fast inner loop for DBSCAN.
# Author: Lars Buitinck
# License: 3-clause BSD

# 导入必要的 C++ 数据类型和容器
from libcpp.vector cimport vector

# 导入相对路径下的 typedefs 文件中定义的数据类型
from ..utils._typedefs cimport uint8_t, intp_t

# 定义 DBSCAN 内部函数，接收 is_core 数组、neighborhoods 对象数组和 labels 数组
def dbscan_inner(const uint8_t[::1] is_core,
                 object[:] neighborhoods,
                 intp_t[::1] labels):
    # 定义变量 i 用于循环索引，label_num 表示当前聚类的标签编号，v 表示临时整数变量
    cdef intp_t i, label_num = 0, v
    # 定义 neighb 作为邻域数组的引用，stack 使用 vector 定义的整数向量作为栈

    # 循环遍历标签数组的长度
    for i in range(labels.shape[0]):
        # 如果当前数据点已经有标签或者不是核心点，则继续下一次循环
        if labels[i] != -1 or not is_core[i]:
            continue

        # 从当前点 i 开始深度优先搜索，直到没有核心点可扩展
        while True:
            # 如果当前点没有标签，则将其标记为当前聚类的标签号
            if labels[i] == -1:
                labels[i] = label_num
                # 如果当前点是核心点，则扩展其邻域
                if is_core[i]:
                    # 获取当前点 i 的邻域数组
                    neighb = neighborhoods[i]
                    # 遍历邻域数组中的每个点
                    for i in range(neighb.shape[0]):
                        v = neighb[i]
                        # 如果邻域点还没有被标记，则将其入栈
                        if labels[v] == -1:
                            stack.push_back(v)

            # 如果栈为空，则退出当前循环
            if stack.size() == 0:
                break
            # 弹出栈顶元素作为下一个要处理的点 i
            i = stack.back()
            stack.pop_back()

        # 当前聚类的标签号加一，准备处理下一个聚类
        label_num += 1
```