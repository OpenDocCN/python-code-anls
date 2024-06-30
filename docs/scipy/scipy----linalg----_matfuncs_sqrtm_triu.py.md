# `D:\src\scipysrc\scipy\scipy\linalg\_matfuncs_sqrtm_triu.py`

```
# 定义一个函数，该函数支持Pythran导出，用于处理给定的两个二维数组R和T，以及一组起始-终止索引对和整数nblocks作为参数
def within_block_loop(R, T, start_stop_pairs, nblocks):
    # 对每个起始-终止索引对(start, stop)进行循环迭代
    for start, stop in start_stop_pairs:
        # 对于每个索引对(start, stop)，在范围内进行迭代
        for j in range(start, stop):
            # 从j-1到start-1的逆序迭代
            for i in range(j-1, start-1, -1):
                # 初始化s为0
                s = 0
                # 当j - i大于1时，计算累加和s = Σ(R[i, k] * R[k, j])，其中k在(i+1)到(j-1)之间
                if j - i > 1:
                    for k in range(i + 1, j):
                        s += R[i, k] * R[k, j]

                # 计算denom为R[i, i] + R[j, j]，num为T[i, j] - s
                denom = R[i, i] + R[j, j]
                num = T[i, j] - s

                # 根据denom的值进行条件判断
                if denom != 0:
                    # 当denom不为0时，计算R[i, j] = (T[i, j] - s) / denom
                    R[i, j] = (T[i, j] - s) / denom
                elif denom == 0 and num == 0:
                    # 当denom为0且num也为0时，将R[i, j]设为0
                    R[i, j] = 0
                else:
                    # 如果以上条件都不满足，则抛出运行时错误，提示未能找到矩阵的平方根
                    raise RuntimeError('failed to find the matrix square root')
```