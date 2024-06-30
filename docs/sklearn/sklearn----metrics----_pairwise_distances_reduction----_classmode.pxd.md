# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_pairwise_distances_reduction\_classmode.pxd`

```
# 定义一个枚举类型 WeightingStrategy，包含三个选项 uniform, distance, callable
cpdef enum WeightingStrategy:
    uniform = 0
    # uniform 表示使用均匀权重策略，即所有项目具有相同的权重
    distance = 1
    # distance 表示使用距离权重策略，在加权直方图模式中使用距离作为权重
    callable = 2
    # callable 表示使用可调用对象作为权重策略的选项
```