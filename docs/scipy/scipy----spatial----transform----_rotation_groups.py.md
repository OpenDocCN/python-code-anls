# `D:\src\scipysrc\scipy\scipy\spatial\transform\_rotation_groups.py`

```
# 导入 NumPy 库，并指定别名 np
import numpy as np
# 从 SciPy 库中导入常数 golden，并指定别名 phi
from scipy.constants import golden as phi

# 定义一个函数 icosahedral，接受一个类 cls 作为参数
def icosahedral(cls):
    # 调用 tetrahedral 函数，将其返回值转换为四元数，并赋给变量 g1
    g1 = tetrahedral(cls).as_quat()
    # 定义常数 a 和 b，分别赋值为 0.5 和 0.5 除以黄金分割率 phi
    a = 0.5
    b = 0.5 / phi
    # 定义常数 c，赋值为黄金分割率 phi 的一半
    c = phi / 2
    # 创建一个 NumPy 数组 g2，包含多个长度为 4 的子数组，每个子数组代表一个四元数
    g2 = np.array([[+a, +b, +c, 0],
                   [+a, +b, -c, 0],
                   [+a, +c, 0, +b],
                   [+a, +c, 0, -b],
                   [+a, -b, +c, 0],
                   [+a, -b, -c, 0],
                   [+a, -c, 0, +b],
                   [+a, -c, 0, -b],
                   [+a, 0, +b, +c],
                   [+a, 0, +b, -c],
                   [+a, 0, -b, +c],
                   [+a, 0, -b, -c],
                   [+b, +a, 0, +c],
                   [+b, +a, 0, -c],
                   [+b, +c, +a, 0],
                   [+b, +c, -a, 0],
                   [+b, -a, 0, +c],
                   [+b, -a, 0, -c],
                   [+b, -c, +a, 0],
                   [+b, -c, -a, 0],
                   [+b, 0, +c, +a],
                   [+b, 0, +c, -a],
                   [+b, 0, -c, +a],
                   [+b, 0, -c, -a],
                   [+c, +a, +b, 0],
                   [+c, +a, -b, 0],
                   [+c, +b, 0, +a],
                   [+c, +b, 0, -a],
                   [+c, -a, +b, 0],
                   [+c, -a, -b, 0],
                   [+c, -b, 0, +a],
                   [+c, -b, 0, -a],
                   [+c, 0, +a, +b],
                   [+c, 0, +a, -b],
                   [+c, 0, -a, +b],
                   [+c, 0, -a, -b],
                   [0, +a, +c, +b],
                   [0, +a, +c, -b],
                   [0, +a, -c, +b],
                   [0, +a, -c, -b],
                   [0, +b, +a, +c],
                   [0, +b, +a, -c],
                   [0, +b, -a, +c],
                   [0, +b, -a, -c],
                   [0, +c, +b, +a],
                   [0, +c, +b, -a],
                   [0, +c, -b, +a],
                   [0, +c, -b, -a]])
    # 将 g1 和 g2 连接起来，形成一个新的四元数数组，并将其作为结果返回
    return cls.from_quat(np.concatenate((g1, g2)))


# 定义一个函数 octahedral，接受一个类 cls 作为参数
def octahedral(cls):
    # 调用 tetrahedral 函数，将其返回值转换为四元数，并赋给变量 g1
    g1 = tetrahedral(cls).as_quat()
    # 定义常数 c，赋值为根号 2 的一半
    c = np.sqrt(2) / 2
    # 创建一个 NumPy 数组 g2，包含多个长度为 4 的子数组，每个子数组代表一个四元数
    g2 = np.array([[+c, 0, 0, +c],
                   [0, +c, 0, +c],
                   [0, 0, +c, +c],
                   [0, 0, -c, +c],
                   [0, -c, 0, +c],
                   [-c, 0, 0, +c],
                   [0, +c, +c, 0],
                   [0, -c, +c, 0],
                   [+c, 0, +c, 0],
                   [-c, 0, +c, 0],
                   [+c, +c, 0, 0],
                   [-c, +c, 0, 0]])
    # 将 g1 和 g2 连接起来，形成一个新的四元数数组，并将其作为结果返回
    return cls.from_quat(np.concatenate((g1, g2)))


# 定义一个函数 tetrahedral，接受一个类 cls 作为参数
def tetrahedral(cls):
    # 创建一个单位四元数的数组 g1
    g1 = np.eye(4)
    # 定义常数 c，赋值为 0.5
    c = 0.5
    # 创建一个 NumPy 数组 g2，包含多个长度为 4 的子数组，每个子数组代表一个四元数
    g2 = np.array([[c, -c, -c, +c],
                   [c, -c, +c, +c],
                   [c, +c, -c, +c],
                   [c, +c, +c, +c],
                   [c, -c, -c, -c],
                   [c, -c, +c, -c],
                   [c, +c, -c, -c],
                   [c, +c, +c, -c]])
    # 将 g1 和 g2 连接起来，形成一个新的四元数数组，并将其作为结果返回
    return cls.from_quat(np.concatenate((g1, g2)))


# 定义一个函数 dicyclic，接受一个类 cls、整数 n 和轴号 axis 作为参数
def dicyclic(cls, n, axis=2):
    # 调用 cyclic 函数，将其返回值转换为旋转向量，并赋给变量 g1
    g1 = cyclic(cls, n, axis).as_rotvec()
    # 在0到π之间生成均匀间隔的角度值，不包括终点π，共生成n个角度值
    thetas = np.linspace(0, np.pi, n, endpoint=False)
    # 根据角度thetas生成旋转向量，其中第一列为0，第二列为cos(thetas)，第三列为sin(thetas)
    rv = np.pi * np.vstack([np.zeros(n), np.cos(thetas), np.sin(thetas)]).T
    # 将rv数组沿指定轴（axis）进行滚动操作
    g2 = np.roll(rv, axis, axis=1)
    # 调用类方法from_rotvec，将g1和g2连接起来并返回结果
    return cls.from_rotvec(np.concatenate((g1, g2)))
# 定义一个函数用于生成具有循环对称性质的群的旋转操作
def cyclic(cls, n, axis=2):
    # 在0到2π之间均匀分布n个角度，生成θ数组，不包括终点2π
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # 创建一个n行3列的数组，第一列是θ数组，其余两列为零，表示旋转向量(rv)
    rv = np.vstack([thetas, np.zeros(n), np.zeros(n)]).T
    # 使用旋转向量创建旋转矩阵，并在指定轴上进行滚动操作
    return cls.from_rotvec(np.roll(rv, axis, axis=1))


# 定义一个函数用于创建具有特定轴和对称性的群
def create_group(cls, group, axis='Z'):
    # 检查群参数是否为字符串，否则引发值错误
    if not isinstance(group, str):
        raise ValueError("`group` argument must be a string")

    # 定义允许的轴列表
    permitted_axes = ['x', 'y', 'z', 'X', 'Y', 'Z']
    # 检查轴参数是否在允许的轴列表中，否则引发值错误
    if axis not in permitted_axes:
        raise ValueError("`axis` must be one of " + ", ".join(permitted_axes))

    # 根据群的不同性质和轴的不同方向选择符号和顺序
    if group in ['I', 'O', 'T']:
        symbol = group
        order = 1
    elif group[:1] in ['C', 'D'] and group[1:].isdigit():
        symbol = group[:1]
        order = int(group[1:])
    else:
        raise ValueError("`group` must be one of 'I', 'O', 'T', 'Dn', 'Cn'")

    # 确保群的顺序为正数
    if order < 1:
        raise ValueError("Group order must be positive")

    # 将轴参数转换为索引，用于后续的旋转群生成函数调用
    axis = 'xyz'.index(axis.lower())
    
    # 根据不同的群符号选择不同的旋转群生成函数进行返回
    if symbol == 'I':
        return icosahedral(cls)
    elif symbol == 'O':
        return octahedral(cls)
    elif symbol == 'T':
        return tetrahedral(cls)
    elif symbol == 'D':
        return dicyclic(cls, order, axis=axis)
    elif symbol == 'C':
        return cyclic(cls, order, axis=axis)
    else:
        # 如果未能匹配到任何符号，则断言出错
        assert False
```