# `D:\src\scipysrc\scikit-learn\sklearn\utils\random.py`

```
"""Utilities for random sampling."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import array  # 导入 array 模块，用于处理数组

import numpy as np  # 导入 NumPy 库，用于数值计算
import scipy.sparse as sp  # 导入 SciPy 的稀疏矩阵模块

from . import check_random_state  # 从当前包导入 check_random_state 函数
from ._random import sample_without_replacement  # 从当前包的 _random 模块导入 sample_without_replacement 函数

__all__ = ["sample_without_replacement"]  # 指定当前模块中公开的接口

def _random_choice_csc(n_samples, classes, class_probability=None, random_state=None):
    """Generate a sparse random matrix given column class distributions

    Parameters
    ----------
    n_samples : int,
        Number of samples to draw in each column.

    classes : list of size n_outputs of arrays of size (n_classes,)
        List of classes for each column.

    class_probability : list of size n_outputs of arrays of \
        shape (n_classes,), default=None
        Class distribution of each column. If None, uniform distribution is
        assumed.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the sampled classes.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    random_matrix : sparse csc matrix of size (n_samples, n_outputs)

    """
    data = array.array("i")  # 创建一个整数数组用于存储数据
    indices = array.array("i")  # 创建一个整数数组用于存储列索引
    indptr = array.array("i", [0])  # 创建一个整数数组作为行指针，并初始化第一个元素为0
    # 遍历每个类别集合
    for j in range(len(classes)):
        # 将每个类别转换为NumPy数组
        classes[j] = np.asarray(classes[j])
        
        # 检查类别数组的数据类型是否为整数，如果不是则抛出异常
        if classes[j].dtype.kind != "i":
            raise ValueError("class dtype %s is not supported" % classes[j].dtype)
        
        # 将类别数组的数据类型转换为int64，确保不复制数据
        classes[j] = classes[j].astype(np.int64, copy=False)

        # 如果没有提供class_probability，则使用均匀分布
        if class_probability is None:
            # 创建一个形状与类别数组长度相同的空概率数组，每个元素的概率为1 / 类别数
            class_prob_j = np.empty(shape=classes[j].shape[0])
            class_prob_j.fill(1 / classes[j].shape[0])
        else:
            # 将class_probability[j]转换为NumPy数组
            class_prob_j = np.asarray(class_probability[j])

        # 检查概率数组的总和是否接近于1，如果不是则抛出异常
        if not np.isclose(np.sum(class_prob_j), 1.0):
            raise ValueError(
                "Probability array at index {0} does not sum to one".format(j)
            )

        # 检查类别数组和概率数组的长度是否一致，如果不一致则抛出异常
        if class_prob_j.shape[0] != classes[j].shape[0]:
            raise ValueError(
                "classes[{0}] (length {1}) and "
                "class_probability[{0}] (length {2}) have "
                "different length.".format(
                    j, classes[j].shape[0], class_prob_j.shape[0]
                )
            )

        # 如果类别数组中不包含0，则插入一个值为0的元素，并在概率数组中插入0.0作为其概率
        if 0 not in classes[j]:
            classes[j] = np.insert(classes[j], 0, 0)
            class_prob_j = np.insert(class_prob_j, 0, 0.0)

        # 如果类别数大于1，则根据class_probability随机选择非零类别的样本
        rng = check_random_state(random_state)
        if classes[j].shape[0] > 1:
            # 找到类别数组中值为0的索引位置
            index_class_0 = np.flatnonzero(classes[j] == 0).item()
            # 计算非零类别的概率
            p_nonzero = 1 - class_prob_j[index_class_0]
            # 计算需要采样的非零类别样本数
            nnz = int(n_samples * p_nonzero)
            # 无放回地从总体中抽取样本
            ind_sample = sample_without_replacement(
                n_population=n_samples, n_samples=nnz, random_state=random_state
            )
            # 将抽取的样本索引加入indices列表中
            indices.extend(ind_sample)

            # 对非零元素的概率进行归一化
            classes_j_nonzero = classes[j] != 0
            class_probability_nz = class_prob_j[classes_j_nonzero]
            class_probability_nz_norm = class_probability_nz / np.sum(
                class_probability_nz
            )
            # 根据归一化后的概率随机选择类别索引
            classes_ind = np.searchsorted(
                class_probability_nz_norm.cumsum(), rng.uniform(size=nnz)
            )
            # 将选择的类别数据加入data列表中
            data.extend(classes[j][classes_j_nonzero][classes_ind])
        
        # 将当前索引的长度添加到indptr列表中，表示当前类别数据在稀疏矩阵中的结束位置
        indptr.append(len(indices))

    # 返回稀疏矩阵，数据为data，列索引为indices，指针为indptr，形状为(n_samples, len(classes))
    return sp.csc_matrix((data, indices, indptr), (n_samples, len(classes)), dtype=int)
```