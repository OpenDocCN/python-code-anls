# `D:\src\scipysrc\scipy\benchmarks\benchmarks\cutest\dfovec.py`

```
# 导入 NumPy 库，用于数值计算
import numpy as np


# 定义函数 dfovec，实现了算法 dfovec.m 的 Python 版本
def dfovec(m, n, x, nprob):
    # 设置多个常数
    c13 = 1.3e1
    c14 = 1.4e1
    c29 = 2.9e1
    c45 = 4.5e1
    
    # 初始化向量 v、y1、y2、y3、y4、y5
    v = [
        4.0e0,
        2.0e0,
        1.0e0,
        5.0e-1,
        2.5e-1,
        1.67e-1,
        1.25e-1,
        1.0e-1,
        8.33e-2,
        7.14e-2,
        6.25e-2,
    ]
    y1 = [
        1.4e-1,
        1.8e-1,
        2.2e-1,
        2.5e-1,
        2.9e-1,
        3.2e-1,
        3.5e-1,
        3.9e-1,
        3.7e-1,
        5.8e-1,
        7.3e-1,
        9.6e-1,
        1.34e0,
        2.1e0,
        4.39e0,
    ]
    y2 = [
        1.957e-1,
        1.947e-1,
        1.735e-1,
        1.6e-1,
        8.44e-2,
        6.27e-2,
        4.56e-2,
        3.42e-2,
        3.23e-2,
        2.35e-2,
        2.46e-2,
    ]
    y3 = [
        3.478e4,
        2.861e4,
        2.365e4,
        1.963e4,
        1.637e4,
        1.372e4,
        1.154e4,
        9.744e3,
        8.261e3,
        7.03e3,
        6.005e3,
        5.147e3,
        4.427e3,
        3.82e3,
        3.307e3,
        2.872e3,
    ]
    y4 = [
        8.44e-1,
        9.08e-1,
        9.32e-1,
        9.36e-1,
        9.25e-1,
        9.08e-1,
        8.81e-1,
        8.5e-1,
        8.18e-1,
        7.84e-1,
        7.51e-1,
        7.18e-1,
        6.85e-1,
        6.58e-1,
        6.28e-1,
        6.03e-1,
        5.8e-1,
        5.58e-1,
        5.38e-1,
        5.22e-1,
        5.06e-1,
        4.9e-1,
        4.78e-1,
        4.67e-1,
        4.57e-1,
        4.48e-1,
        4.38e-1,
        4.31e-1,
        4.24e-1,
        4.2e-1,
        4.14e-1,
        4.11e-1,
        4.06e-1,
    ]
    y5 = [
        1.366e0,
        1.191e0,
        1.112e0,
        1.013e0,
        9.91e-1,
        8.85e-1,
        8.31e-1,
        8.47e-1,
        7.86e-1,
        7.25e-1,
        7.46e-1,
        6.79e-1,
        6.08e-1,
        6.55e-1,
        6.16e-1,
        6.06e-1,
        6.02e-1,
        6.26e-1,
        6.51e-1,
        7.24e-1,
        6.49e-1,
        6.49e-1,
        6.94e-1,
        6.44e-1,
        6.24e-1,
        6.61e-1,
        6.12e-1,
        5.58e-1,
        5.33e-1,
        4.95e-1,
        5.0e-1,
        4.23e-1,
        3.95e-1,
        3.75e-1,
        3.72e-1,
        3.91e-1,
        3.96e-1,
        4.05e-1,
        4.28e-1,
        4.29e-1,
        5.23e-1,
        5.62e-1,
        6.07e-1,
        6.53e-1,
        6.72e-1,
        7.08e-1,
        6.33e-1,
        6.68e-1,
        6.45e-1,
        6.32e-1,
        5.91e-1,
        5.59e-1,
        5.97e-1,
        6.25e-1,
        7.39e-1,
        7.1e-1,
        7.29e-1,
        7.2e-1,
        6.36e-1,
        5.81e-1,
        4.28e-1,
        2.92e-1,
        1.62e-1,
        9.8e-2,
        5.4e-2,
    ]

    # 初始化向量 fvec，长度为 m，用于存储结果
    fvec = np.zeros(m)
    # 初始化变量 total，用于累加计算
    if nprob == 1:  # 如果问题编号为1，表示线性函数 - 完全秩
        for j in range(n):  # 遍历从0到n-1的范围
            total = total + x[j]  # 计算所有x[j]的总和
        temp = 2 * total / m + 1  # 计算临时变量temp的值
        for i in range(m):  # 遍历从0到m-1的范围
            fvec[i] = -temp  # 设置fvec[i]的值为-temp
            if i < n:  # 如果i小于n
                fvec[i] = fvec[i] + x[i]  # 将fvec[i]增加x[i]
    elif nprob == 2:  # 如果问题编号为2，表示线性函数 - 秩为1
        for j in range(n):  # 遍历从0到n-1的范围
            total = total + (j + 1) * x[j]  # 计算加权和，权重为j+1
        for i in range(m):  # 遍历从0到m-1的范围
            fvec[i] = (i + 1) * total - 1  # 计算fvec[i]的值
    elif nprob == 3:  # 如果问题编号为3，表示线性函数 - 秩为1，并带有零列和零行
        for j in range(1, n - 1):  # 遍历从1到n-2的范围
            total = total + (j + 1) * x[j]  # 计算加权和，权重为j+1
        for i in range(m - 1):  # 遍历从0到m-2的范围
            fvec[i] = i * total - 1  # 计算fvec[i]的值
        fvec[m - 1] = -1  # 设置fvec[m-1]的值为-1
    elif nprob == 4:  # 如果问题编号为4，表示Rosenbrock函数
        fvec[0] = 10 * (x[1] - x[0] * x[0])  # 计算fvec[0]的值
        fvec[1] = 1 - x[0]  # 计算fvec[1]的值
    elif nprob == 5:  # 如果问题编号为5，表示Helical valley函数
        if x[0] > 0:  # 如果x[0]大于0
            th = np.arctan(x[1] / x[0]) / (2 * np.pi)  # 计算th的值
        elif x[0] < 0:  # 如果x[0]小于0
            th = np.arctan(x[1] / x[0]) / (2 * np.pi) + 0.5  # 计算th的值
        elif x[0] == x[1] and x[1] == 0:  # 如果x[0]等于x[1]且它们都为0
            th = 0.0  # 设置th的值为0.0
        else:  # 否则
            th = 0.25  # 设置th的值为0.25
        r = np.sqrt(x[0] * x[0] + x[1] * x[1])  # 计算r的值
        fvec[0] = 10 * (x[2] - 10 * th)  # 计算fvec[0]的值
        fvec[1] = 10 * (r - 1)  # 计算fvec[1]的值
        fvec[2] = x[2]  # 设置fvec[2]的值为x[2]
    elif nprob == 6:  # 如果问题编号为6，表示Powell singular函数
        fvec[0] = x[0] + 10 * x[1]  # 计算fvec[0]的值
        fvec[1] = np.sqrt(5) * (x[2] - x[3])  # 计算fvec[1]的值
        fvec[2] = (x[1] - 2 * x[2]) ** 2  # 计算fvec[2]的值
        fvec[3] = np.sqrt(10) * (x[0] - x[3]) ** 2  # 计算fvec[3]的值
    elif nprob == 7:  # 如果问题编号为7，表示Freudenstein and Roth函数
        fvec[0] = -c13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1]  # 计算fvec[0]的值
        fvec[1] = -c29 + x[0] + ((1 + x[1]) * x[1] - c14) * x[1]  # 计算fvec[1]的值
    elif nprob == 8:  # 如果问题编号为8，表示Bard函数
        for i in range(15):  # 遍历从0到14的范围
            tmp1 = i + 1  # 设置tmp1的值
            tmp2 = 15 - i  # 设置tmp2的值
            tmp3 = tmp1  # 设置tmp3的值为tmp1
            if i > 7:  # 如果i大于7
                tmp3 = tmp2  # 设置tmp3的值为tmp2
            fvec[i] = y1[i] - (x[0] + tmp1 / (x[1] * tmp2 + x[2] * tmp3))  # 计算fvec[i]的值
    elif nprob == 9:  # 如果问题编号为9，表示Kowalik and Osborne函数
        for i in range(11):  # 遍历从0到10的范围
            tmp1 = v[i] * (v[i] + x[1])  # 计算tmp1的值
            tmp2 = v[i] * (v[i] + x[2]) + x[3]  # 计算tmp2的值
            fvec[i] = y2[i] - x[0] * tmp1 / tmp2  # 计算fvec[i]的值
    elif nprob == 10:  # 如果问题编号为10，表示Meyer函数
        for i in range(16):  # 遍历从0到15的范围
            temp = 5 * (i + 1) + c45 + x[2]  # 计算temp的值
            tmp1 = x[1] / temp  # 计算tmp1的值
            tmp2 = np.exp(tmp1)  # 计算tmp2的值
            fvec[i] = x[0] * tmp2 - y3[i]  # 计算fvec[i]的值
    elif nprob == 11:  # 如果问题编号为11，表示Watson函数
        for i in range(29):  # 遍历从0到28的范围
            div = (i + 1) / c29  # 计算div的值
            s1 = 0  # 初始化s1为0
            dx = 1  # 初始化dx为1
            for j in range(1, n):  # 遍历从1到n-1的范围
                s1 = s1 + j * dx * x[j]  # 计算s1的值
                dx = div * dx  # 更新dx的值
            s2 = 0  # 初始化s2为0
            dx = 1  # 初始化dx为1
            for j in range(n):  # 遍历从0到n-1的范围
                s2 = s2 + dx * x[j]  # 计算s2的值
                dx = div * dx  # 更新dx的值
            fvec[i] = s1 - s2 * s2 - 1  # 计算fvec[i]的值
        fvec[29] = x[0]  # 设置fvec[29]的值为x[0]
        fvec[30] = x[1] - x[0] * x[0] - 1  # 计算fvec[30]的值
    elif nprob == 12:  # Box 3-dimensional function.
        # Loop through each component of the function vector
        for i in range(m):
            temp = i + 1  # Compute the index offset
            tmp1 = temp / 10  # Calculate a temporary value
            # Evaluate the i-th component of the function vector
            fvec[i] = (
                np.exp(-tmp1 * x[0])  # Exponential function with adjusted argument
                - np.exp(-tmp1 * x[1])  # Exponential function with adjusted argument
                + (np.exp(-temp) - np.exp(-tmp1)) * x[2]  # Linear combination of exponentials and x[2]
            )
    elif nprob == 13:  # Jennrich and Sampson function.
        # Loop through each component of the function vector
        for i in range(m):
            temp = i + 1  # Compute the index offset
            # Evaluate the i-th component of the function vector
            fvec[i] = 2 + 2 * temp - np.exp(temp * x[0]) - np.exp(temp * x[1])
    elif nprob == 14:  # Brown and Dennis function.
        # Loop through each component of the function vector
        for i in range(m):
            temp = (i + 1) / 5  # Compute the scaled index offset
            # Evaluate the i-th component of the function vector
            tmp1 = x[0] + temp * x[1] - np.exp(temp)
            tmp2 = x[2] + np.sin(temp) * x[3] - np.cos(temp)
            fvec[i] = tmp1 * tmp1 + tmp2 * tmp2  # Combine squares of temporary values
    elif nprob == 15:  # Chebyquad function.
        # Loop through each component of the function vector
        for j in range(n):
            t1 = 1  # Initialize temporary variable
            t2 = 2 * x[j] - 1  # Compute a temporary value
            t = 2 * t2  # Compute another temporary value
            # Nested loop to evaluate components of the function vector
            for i in range(m):
                fvec[i] = fvec[i] + t2  # Accumulate temporary value into function vector
                th = t * t2 - t1  # Compute a temporary value
                t1 = t2  # Update temporary variable
                t2 = th  # Update temporary variable
        iev = -1  # Initialize a temporary variable
        # Loop to finalize evaluation of components of the function vector
        for i in range(m):
            fvec[i] = fvec[i] / n  # Scale component of the function vector
            if iev > 0:
                fvec[i] = fvec[i] + 1 / ((i + 1) ** 2 - 1)  # Adjust component based on index
            iev = -iev  # Update temporary variable
    elif nprob == 16:  # Brown almost-linear function.
        total1 = -(n + 1)  # Initialize a total variable
        prod1 = 1  # Initialize a product variable
        # Loop through each variable
        for j in range(n):
            total1 = total1 + x[j]  # Accumulate sum of variables
            prod1 = x[j] * prod1  # Accumulate product of variables
        # Loop through all but the last component of the function vector
        for i in range(n - 1):
            fvec[i] = x[i] + total1  # Evaluate component based on sum
        fvec[n - 1] = prod1 - 1  # Evaluate last component based on product
    elif nprob == 17:  # Osborne 1 function.
        # Loop through specific indices
        for i in range(33):
            temp = 10 * i  # Compute a scaled index
            tmp1 = np.exp(-x[3] * temp)  # Exponential function with adjusted argument
            tmp2 = np.exp(-x[4] * temp)  # Exponential function with adjusted argument
            fvec[i] = y4[i] - (x[0] + x[1] * tmp1 + x[2] * tmp2)  # Compute function component
    elif nprob == 18:  # Osborne 2 function.
        # Loop through specific indices
        for i in range(65):
            temp = i / 10  # Compute a scaled index
            tmp1 = np.exp(-x[4] * temp)  # Exponential function with adjusted argument
            tmp2 = np.exp(-x[5] * (temp - x[8]) ** 2)  # Exponential function with adjusted argument
            tmp3 = np.exp(-x[6] * (temp - x[9]) ** 2)  # Exponential function with adjusted argument
            tmp4 = np.exp(-x[7] * (temp - x[10]) ** 2)  # Exponential function with adjusted argument
            fvec[i] = y5[i] - (x[0] * tmp1 + x[1] * tmp2 + x[2] * tmp3 + x[3] * tmp4)  # Compute function component
    elif nprob == 19:  # Bdqrtic
        # n >= 5, m = (n-4)*2
        # Loop through a subset of variables
        for i in range(n - 4):
            fvec[i] = -4 * x[i] + 3.0  # Compute specific function components
            fvec[n - 4 + i] = (
                x[i] ** 2
                + 2 * x[i + 1] ** 2
                + 3 * x[i + 2] ** 2
                + 4 * x[i + 3] ** 2
                + 5 * x[n - 1] ** 2
            )  # Compute specific function components
    elif nprob == 20:  # Cube
        # n = 2, m = n
        fvec[0] = x[0] - 1.0  # Evaluate first component of the function vector
        # Loop through subsequent components of the function vector
        for i in range(1, n):
            fvec[i] = 10 * (x[i] - x[i - 1] ** 3)  # Evaluate components based on previous values
    elif nprob == 21:  # 如果 nprob 等于 21，则执行以下代码（Mancino 函数）
        # 设置循环范围为 n 次，其中 n 是输入参数
        for i in range(n):
            # 初始化变量 ss 为 0
            ss = 0
            # 第二个嵌套循环，范围同样为 n 次
            for j in range(n):
                # 计算 v2，使用 x[i] 和 j 的值计算平方根
                v2 = np.sqrt(x[i] ** 2 + (i + 1) / (j + 1))
                # 更新 ss 变量，加上基于 v2 计算的表达式结果
                ss = ss + v2 * ((np.sin(np.log(v2))) ** 5 + (np.cos(np.log(v2))) ** 5)
            # 计算 Mancino 函数中的 fvec[i] 值
            fvec[i] = 1400 * x[i] + (i - 49) ** 3 + ss
    elif nprob == 22:  # 如果 nprob 等于 22，则执行以下代码（Heart8ls 函数）
        # 设置 fvec 的不同索引处的值，这里假设 n = 8
        fvec[0] = x[0] + x[1] + 0.69
        fvec[1] = x[2] + x[3] + 0.044
        fvec[2] = x[4] * x[0] + x[5] * x[1] - x[6] * x[2] - x[7] * x[3] + 1.57
        fvec[3] = x[6] * x[0] + x[7] * x[1] + x[4] * x[2] + x[5] * x[3] + 1.31
        fvec[4] = (
            x[0] * (x[4] ** 2 - x[6] ** 2)
            - 2.0 * x[2] * x[4] * x[6]
            + x[1] * (x[5] ** 2 - x[7] ** 2)
            - 2.0 * x[3] * x[5] * x[7]
            + 2.65
        )
        fvec[5] = (
            x[2] * (x[4] ** 2 - x[6] ** 2)
            + 2.0 * x[0] * x[4] * x[6]
            + x[3] * (x[5] ** 2 - x[7] ** 2)
            + 2.0 * x[1] * x[5] * x[7]
            - 2.0
        )
        fvec[6] = (
            x[0] * x[4] * (x[4] ** 2 - 3.0 * x[6] ** 2)
            + x[2] * x[6] * (x[6] ** 2 - 3.0 * x[4] ** 2)
            + x[1] * x[5] * (x[5] ** 2 - 3.0 * x[7] ** 2)
            + x[3] * x[7] * (x[7] ** 2 - 3.0 * x[5] ** 2)
            + 12.6
        )
        fvec[7] = (
            x[2] * x[4] * (x[4] ** 2 - 3.0 * x[6] ** 2)
            - x[0] * x[6] * (x[6] ** 2 - 3.0 * x[4] ** 2)
            + x[3] * x[5] * (x[5] ** 2 - 3.0 * x[7] ** 2)
            - x[1] * x[7] * (x[7] ** 2 - 3.0 * x[6] ** 2)
            - 9.48
        )
    else:
        # 若 nprob 不是 21 或 22，则打印未识别的函数编号
        print(f"unrecognized function number {nprob}")
        # 返回空值
        return None
    # 返回计算出的 fvec 向量
    return fvec
```