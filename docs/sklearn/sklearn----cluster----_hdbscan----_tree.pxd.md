# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_hdbscan\_tree.pxd`

```
# 版权声明，声明作者保留所有权利，未经许可不得使用或分发该软件
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. 在源代码的分发中必须保留上述版权声明、此条件列表和以下免责声明。
# 2. 在二进制形式的分发中，必须在文档和/或其他提供的材料中重现上述版权声明、此条件列表和以下免责声明。
# 3. 不得使用版权持有者的名称或其贡献者的名称，来认可或推广从本软件衍生的产品，除非有明确的事先书面许可。

# 本软件由版权持有者和贡献者 "按原样" 提供，不作任何明示或暗示的担保，
# 包括但不限于对适销性和特定用途的适用性的暗示担保。在任何情况下，
# 版权持有者或贡献者均不对任何直接、间接、偶发、特殊、惩罚性或后果性损害负责，
# 包括但不限于采购替代商品或服务的成本、使用数据或利润的损失或业务中断，无论是在合同、严格责任或侵权行为(包括疏忽或其他方式)的任何理论下引起的，即使已被告知可能性。

from ...utils._typedefs cimport intp_t, float64_t, uint8_t
import numpy as cnp

# 定义了 HIERARCHY_t 结构体，用于描述 scipy.cluster.hierarchy 格式的数据
# 结构体包含左节点、右节点、值和簇大小
ctypedef packed struct HIERARCHY_t:
    intp_t left_node
    intp_t right_node
    float64_t value
    intp_t cluster_size

# 定义了 CONDENSED_t 结构体，用于表示边列表，包含父/子节点、值和对应的簇大小
# 每行提供了树结构的信息
ctypedef packed struct CONDENSED_t:
    intp_t parent
    intp_t child
    float64_t value
    intp_t cluster_size

# 从 numpy/arrayobject.h 头文件中引入 PyArray_SHAPE 函数声明
cdef extern from "numpy/arrayobject.h":
    intp_t * PyArray_SHAPE(cnp.PyArrayObject *)
```