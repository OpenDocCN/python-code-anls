# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HighsInfo.pxd`

```
# cython: language_level=3

# 从 "HighsInfo.h" 文件中导入 HighsInfo 类型，并且在此声明不使用全局解释器锁（nogil）
cdef extern from "HighsInfo.h" nogil:
    # HighsInfo 类型定义，对应 HiGHS 源码中的 lp_data/HighsInfo.h 头文件
    cdef cppclass HighsInfo:
        # HighsInfo 类型的成员变量，从 HighsInfoStruct 继承而来
        int mip_node_count                       # MIP（混合整数规划）节点计数
        int simplex_iteration_count              # 单纯形迭代计数
        int ipm_iteration_count                  # 内点法迭代计数
        int crossover_iteration_count            # 交叉迭代计数
        int primal_solution_status               # 原始解决方案状态
        int dual_solution_status                 # 对偶解决方案状态
        int basis_validity                       # 基的有效性
        double objective_function_value          # 目标函数值
        double mip_dual_bound                    # MIP 双重界限
        double mip_gap                           # MIP 间隙
        int num_primal_infeasibilities           # 原始不可行性数量
        double max_primal_infeasibility          # 最大原始不可行性
        double sum_primal_infeasibilities        # 原始不可行性总和
        int num_dual_infeasibilities             # 对偶不可行性数量
        double max_dual_infeasibility            # 最大对偶不可行性
        double sum_dual_infeasibilities         # 对偶不可行性总和
```