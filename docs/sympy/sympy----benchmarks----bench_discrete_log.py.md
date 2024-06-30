# `D:\src\scipysrc\sympy\sympy\benchmarks\bench_discrete_log.py`

```
# 导入 sys 模块，用于处理命令行参数
import sys
# 从 time 模块中导入 time 函数，用于计时
from time import time
# 从 sympy.ntheory.residue_ntheory 模块中导入以下函数，用于离散对数计算
from sympy.ntheory.residue_ntheory import (
        discrete_log,
        _discrete_log_trial_mul,
        _discrete_log_shanks_steps,
        _discrete_log_pollard_rho,
        _discrete_log_pohlig_hellman)


# 第一个数据集，包含循环群 (Z/pZ)*，其中 p 是素数，阶数为 p - 1，生成元为 g
data_set_1 = [
        # p, p - 1, g
        [191, 190, 19],
        [46639, 46638, 6],
        [14789363, 14789362, 2],
        [4254225211, 4254225210, 2],
        [432751500361, 432751500360, 7],
        [158505390797053, 158505390797052, 2],
        [6575202655312007, 6575202655312006, 5],
        [8430573471995353769, 8430573471995353768, 3],
        [3938471339744997827267, 3938471339744997827266, 2],
        [875260951364705563393093, 875260951364705563393092, 5],
    ]


# 第二个数据集，包含循环子群 (Z/nZ)*，其中 n 和 p 是素数，n = 2 * p + 1
data_set_2 = [
        # n, p, g
        [227, 113, 3],
        [2447, 1223, 2],
        [24527, 12263, 2],
        [245639, 122819, 2],
        [2456747, 1228373, 3],
        [24567899, 12283949, 3],
        [245679023, 122839511, 2],
        [2456791307, 1228395653, 3],
        [24567913439, 12283956719, 2],
        [245679135407, 122839567703, 2],
        [2456791354763, 1228395677381, 3],
        [24567913550903, 12283956775451, 2],
        [245679135509519, 122839567754759, 2],
    ]


# 第三个数据集，包含循环子群 (Z/nZ)*，其中阶数 o 是光滑数，生成元为 g
data_set_3 = [
        # n, o, g
        [2**118, 2**116, 3],
    ]


# 函数 bench_discrete_log，用于对给定数据集进行离散对数算法的性能评估
def bench_discrete_log(data_set, algo=None):
    # 根据传入的算法名称选择对应的离散对数算法函数
    if algo is None:
        f = discrete_log
    elif algo == 'trial':
        f = _discrete_log_trial_mul
    elif algo == 'shanks':
        f = _discrete_log_shanks_steps
    elif algo == 'rho':
        f = _discrete_log_pollard_rho
    elif algo == 'ph':
        f = _discrete_log_pohlig_hellman
    else:
        raise ValueError("Argument 'algo' should be one"
                " of ('trial', 'shanks', 'rho' or 'ph')")

    # 遍历数据集并执行离散对数算法，记录执行时间并验证结果
    for i, data in enumerate(data_set):
        for j, (n, p, g) in enumerate(data):
            t = time()  # 记录当前时间
            l = f(n, pow(g, p - 1, n), g, p)  # 执行离散对数算法
            t = time() - t  # 计算执行时间
            print('[%02d-%03d] %15.10f' % (i, j, t))  # 打印执行时间
            assert l == p - 1  # 验证计算结果与预期值是否一致


# 当该脚本作为主程序运行时执行以下代码
if __name__ == '__main__':
    # 从命令行参数中获取算法名称（若有）
    algo = sys.argv[1] \
            if len(sys.argv) > 1 else None
    # 将所有数据集合并到一个列表中
    data_set = [
            data_set_1,
            data_set_2,
            data_set_3,
        ]
    # 调用 bench_discrete_log 函数，评估离散对数算法的性能
    bench_discrete_log(data_set, algo)
```