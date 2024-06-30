# `D:\src\scipysrc\scipy\scipy\special\_precompute\hyp2f1_data.py`

```
"""This script evaluates scipy's implementation of hyp2f1 against mpmath's.

Author: Albert Steppi

This script is long running and generates a large output file. With default
arguments, the generated file is roughly 700MB in size and it takes around
40 minutes using an Intel(R) Core(TM) i5-8250U CPU with n_jobs set to 8
(full utilization). There are optional arguments which can be used to restrict
(or enlarge) the computations performed. These are described below.
The output of this script can be analyzed to identify suitable test cases and
to find parameter and argument regions where hyp2f1 needs to be improved.

The script has one mandatory positional argument for specifying the path to
the location where the output file is to be placed, and 4 optional arguments
--n_jobs, --grid_size, --regions, and --parameter_groups. --n_jobs specifies
the number of processes to use if running in parallel. The default value is 1.
The other optional arguments are explained below.

Produces a tab separated values file with 11 columns. The first four columns
contain the parameters a, b, c and the argument z. The next two contain |z| and
a region code for which region of the complex plane belongs to. The regions are

    0) z == 1
    1) |z| < 0.9 and real(z) >= 0
    2) |z| <= 1 and real(z) < 0
    3) 0.9 <= |z| <= 1 and |1 - z| < 0.9:
    4) 0.9 <= |z| <= 1 and |1 - z| >= 0.9 and real(z) >= 0:
    5) 1 < |z| < 1.1 and |1 - z| >= 0.9 and real(z) >= 0
    6) |z| > 1 and not in 5)

The --regions optional argument allows the user to specify a list of regions
to which computation will be restricted.

Parameters a, b, c are taken from a 10 * 10 * 10 grid with values at

    -16, -8, -4, -2, -1, 1, 2, 4, 8, 16

with random perturbations applied.

There are 9 parameter groups handling the following cases.

    1) A, B, C, B - A, C - A, C - B, C - A - B all non-integral.
    2) B - A integral
    3) C - A integral
    4) C - B integral
    5) C - A - B integral
    6) A integral
    7) B integral
    8) C integral
    9) Wider range with c - a - b > 0.

The seventh column of the output file is an integer between 1 and 8 specifying
the parameter group as above.

The --parameter_groups optional argument allows the user to specify a list of
parameter groups to which computation will be restricted.

The argument z is taken from a grid in the box
    -box_size <= real(z) <= box_size, -box_size <= imag(z) <= box_size.
with grid size specified using the optional command line argument --grid_size,
and box_size specified with the command line argument --box_size.
The default value of grid_size is 20 and the default value of box_size is 2.0,
yielding a 20 * 20 grid in the box with corners -2-2j, -2+2j, 2-2j, 2+2j.

The final four columns have the expected value of hyp2f1 for the given
parameters and argument as calculated with mpmath, the observed value
calculated with scipy's hyp2f1, the relative error, and the absolute error.
"""
"""
As special cases of hyp2f1 are moved from the original Fortran implementation
into Cython, this script can be used to ensure that no regressions occur and
to point out where improvements are needed.
"""

import os  # 导入操作系统功能模块
import csv  # 导入CSV文件读写模块
import argparse  # 导入命令行参数解析模块
import numpy as np  # 导入数值计算库numpy
from itertools import product  # 导入迭代工具函数product
from multiprocessing import Pool  # 导入多进程池功能

from scipy.special import hyp2f1  # 导入scipy中的超几何函数hyp2f1
from scipy.special.tests.test_hyp2f1 import mp_hyp2f1  # 导入测试模块中的mp_hyp2f1函数


def get_region(z):
    """Assign numbers for regions where hyp2f1 must be handled differently."""
    if z == 1 + 0j:
        return 0  # 若z为1，则返回0
    elif abs(z) < 0.9 and z.real >= 0:
        return 1  # 若abs(z) < 0.9且z的实部大于等于0，则返回1
    elif abs(z) <= 1 and z.real < 0:
        return 2  # 若abs(z) <= 1且z的实部小于0，则返回2
    elif 0.9 <= abs(z) <= 1 and abs(1 - z) < 0.9:
        return 3  # 若0.9 <= abs(z) <= 1且abs(1-z) < 0.9，则返回3
    elif 0.9 <= abs(z) <= 1 and abs(1 - z) >= 0.9:
        return 4  # 若0.9 <= abs(z) <= 1且abs(1-z) >= 0.9，则返回4
    elif 1 < abs(z) < 1.1 and abs(1 - z) >= 0.9 and z.real >= 0:
        return 5  # 若1 < abs(z) < 1.1且abs(1-z) >= 0.9且z的实部大于等于0，则返回5
    else:
        return 6  # 其他情况返回6


def get_result(a, b, c, z, group):
    """Get results for given parameter and value combination."""
    expected, observed = mp_hyp2f1(a, b, c, z), hyp2f1(a, b, c, z)
    if (
            np.isnan(observed) and np.isnan(expected) or
            expected == observed
    ):
        relative_error = 0.0  # 若expected和observed相等，相对误差为0
        absolute_error = 0.0  # 若expected和observed相等，绝对误差为0
    elif np.isnan(observed):
        # Set error to infinity if result is nan when not expected to be.
        # Makes results easier to interpret.
        relative_error = float("inf")  # 若observed为NaN且不应该为NaN，则设置相对误差为无穷大
        absolute_error = float("inf")  # 若observed为NaN且不应该为NaN，则设置绝对误差为无穷大
    else:
        absolute_error = abs(expected - observed)  # 计算绝对误差
        relative_error = absolute_error / abs(expected)  # 计算相对误差

    return (
        a,
        b,
        c,
        z,
        abs(z),
        get_region(z),
        group,
        expected,
        observed,
        relative_error,
        absolute_error,
    )


def get_result_no_mp(a, b, c, z, group):
    """Get results for given parameter and value combination."""
    expected, observed = complex('nan'), hyp2f1(a, b, c, z)
    relative_error, absolute_error = float('nan'), float('nan')
    return (
        a,
        b,
        c,
        z,
        abs(z),
        get_region(z),
        group,
        expected,
        observed,
        relative_error,
        absolute_error,
    )


def get_results(params, Z, n_jobs=1, compute_mp=True):
    """Batch compute results for multiple parameter and argument values.

    Parameters
    ----------
    params : iterable
        iterable of tuples of floats (a, b, c) specifying parameter values
        a, b, c for hyp2f1
    Z : iterable of complex
        Arguments at which to evaluate hyp2f1
    n_jobs : Optional[int]
        Number of jobs for parallel execution.

    Returns
    -------
    list
        List of tuples of results values. See return value in source code
        of `get_result`.
    """
    input_ = (
        (a, b, c, z, group) for (a, b, c, group), z in product(params, Z)
    )
    # 使用进程池并发处理任务
    with Pool(n_jobs) as pool:
        # 调用进程池的starmap方法，根据compute_mp条件选择处理函数get_result或get_result_no_mp，并传入input_作为参数
        rows = pool.starmap(
            get_result if compute_mp else get_result_no_mp,
            input_
        )
    # 返回处理后的结果行列表
    return rows
# 生成单个测试用例的字符串，用于 test_hyp2f1.py 中的测试
def _make_hyp2f1_test_case(a, b, c, z, rtol):
    """Generate string for single test case as used in test_hyp2f1.py."""
    # 调用 mp_hyp2f1 函数计算期望值
    expected = mp_hyp2f1(a, b, c, z)
    # 返回格式化后的参数字符串，作为 pytest.param 的输入
    return (
        "    pytest.param(\n"
        "        Hyp2f1TestCase(\n"
        f"            a={a},\n"
        f"            b={b},\n"
        f"            c={c},\n"
        f"            z={z},\n"
        f"            expected={expected},\n"
        f"            rtol={rtol},\n"
        "        ),\n"
        "    ),"
    )


# 生成一组测试用例的字符串，用于 test_hyp2f1.py 中的测试
def make_hyp2f1_test_cases(rows):
    """Generate string for a list of test cases for test_hyp2f1.py.

    Parameters
    ----------
    rows : list
        List of lists of the form [a, b, c, z, rtol] where a, b, c, z are
        parameters and the argument for hyp2f1 and rtol is an expected
        relative error for the associated test case.

    Returns
    -------
    str
        String for a list of test cases. The output string can be printed
        or saved to a file and then copied into an argument for
        `pytest.mark.parameterize` within `scipy.special.tests.test_hyp2f1.py`.
    """
    # 初始化结果字符串
    result = "[\n"
    # 使用 _make_hyp2f1_test_case 函数生成每个测试用例的字符串，并将其连接起来
    result += '\n'.join(
        _make_hyp2f1_test_case(a, b, c, z, rtol)
        for a, b, c, z, rtol in rows
    )
    # 结束字符串
    result += "\n]"
    return result


# 主函数，生成不同参数组合的测试用例字符串
def main(
        outpath,
        n_jobs=1,
        box_size=2.0,
        grid_size=20,
        regions=None,
        parameter_groups=None,
        compute_mp=True,
):
    # 将输出路径规范化为绝对路径
    outpath = os.path.realpath(os.path.expanduser(outpath))

    # 初始化随机数生成器
    random_state = np.random.RandomState(1234)
    # 选择参数 a, b, c 的根值附近的数值
    root_params = np.array(
        [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
    )
    # 对根值进行的扰动
    perturbations = 0.1 * random_state.random_sample(
        size=(3, len(root_params))
    )

    params = []
    # 参数组合 1
    # -----------------
    # 没有整数差异，以上述种子进行了确认
    A = root_params + perturbations[0, :]
    B = root_params + perturbations[1, :]
    C = root_params + perturbations[2, :]
    # 将参数组合按最大绝对值排序并添加到参数列表中
    params.extend(
        sorted(
            ((a, b, c, 1) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # 参数组合 2
    # -----------------
    # B - A 是整数
    A = root_params + 0.5
    B = root_params + 0.5
    C = root_params + perturbations[1, :]
    # 将参数组合按最大绝对值排序并添加到参数列表中
    params.extend(
        sorted(
            ((a, b, c, 2) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # 参数组合 3
    # -----------------
    # C - A 是整数
    A = root_params + 0.5
    B = root_params + perturbations[1, :]
    C = root_params + 0.5
    # 将参数组合按最大绝对值排序并添加到参数列表中
    params.extend(
        sorted(
            ((a, b, c, 3) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # 参数组合 4
    # -----------------
    # C - B 是整数
    # 创建新的参数组 A，其中包含原始参数和 perturbations 的第一个元素
    A = root_params + perturbations[0, :]
    # 创建新的参数组 B 和 C，均为原始参数加上 0.5
    B = root_params + 0.5
    C = root_params + 0.5
    # 将由 A、B、C 组成的所有三元组 (a, b, c, 4) 排序后加入到 params 列表中
    params.extend(
        sorted(
            ((a, b, c, 4) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 5
    # -----------------
    # 创建新的参数组 A 和 B，分别为原始参数加上 0.25
    A = root_params + 0.25
    B = root_params + 0.25
    C = root_params + 0.5
    # 将由 A、B、C 组成的所有三元组 (a, b, c, 5) 排序后加入到 params 列表中
    params.extend(
        sorted(
            ((a, b, c, 5) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 6
    # -----------------
    # 创建新的参数组 B 和 C，分别为原始参数和 perturbations 的第一个和第二个元素
    A = root_params
    B = root_params + perturbations[0, :]
    C = root_params + perturbations[1, :]
    # 将由 A、B、C 组成的所有三元组 (a, b, c, 6) 排序后加入到 params 列表中
    params.extend(
        sorted(
            ((a, b, c, 6) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 7
    # -----------------
    # 创建新的参数组 A 和 C，分别为原始参数和 perturbations 的第一个和第二个元素
    A = root_params + perturbations[0, :]
    B = root_params
    C = root_params + perturbations[1, :]
    # 将由 A、B、C 组成的所有三元组 (a, b, c, 7) 排序后加入到 params 列表中
    params.extend(
        sorted(
            ((a, b, c, 7) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 8
    # -----------------
    # 创建新的参数组 A 和 B，分别为原始参数和 perturbations 的第一个和第二个元素
    A = root_params + perturbations[0, :]
    B = root_params + perturbations[1, :]
    C = root_params
    # 将由 A、B、C 组成的所有三元组 (a, b, c, 8) 排序后加入到 params 列表中
    params.extend(
        sorted(
            ((a, b, c, 8) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 9
    # -----------------
    # 计算黄金分割率 phi 并生成长度为 16 的序列 P，包括其相反数
    phi = (1 + np.sqrt(5))/2
    P = phi**np.arange(16)
    P = np.hstack([-P, P])
    # 生成所有满足 c - a - b > 0 条件的三元组 (a, b, c, 9)，并按最大绝对值排序
    group_9_params = sorted(
        (
            (a, b, c, 9) for a, b, c in product(P, P, P) if c - a - b > 0
        ),
        key=lambda x: max(abs(x[0]), abs(x[1])),
    )

    if parameter_groups is not None:
        # 如果指定了 parameter_groups，只添加指定分组的参数到 params 中
        params.extend(group_9_params)
        params = [
            (a, b, c, group) for a, b, c, group in params
            if group in parameter_groups
        ]

    # 生成网格 X 和 Y，分别为在 [-box_size, box_size] 区间内均匀分布的 grid_size 个点
    X, Y = np.meshgrid(
        np.linspace(-box_size, box_size, grid_size),
        np.linspace(-box_size, box_size, grid_size)
    )
    # 构建复数网格 Z，展开为一维数组并添加额外的复数 1+0j
    Z = X + Y * 1j
    Z = Z.flatten().tolist()
    # 如果指定了 regions，只保留 Z 中位于指定区域的复数
    if regions is not None:
        Z = [z for z in Z if get_region(z) in regions]

    # 对于所有参数组合和网格 Z 中的参数，计算 scipy 和 mpmath 的 hyp2f1 函数的结果
    # 使用多线程 n_jobs 加速计算，并考虑是否计算 mpmath 版本
    rows = get_results(params, Z, n_jobs=n_jobs, compute_mp=compute_mp)
    # 使用指定路径创建一个写入文件对象，模式为写入文本（"w"），换行符为默认设置
    with open(outpath, "w", newline="") as f:
        # 创建 CSV 写入器对象，使用制表符作为字段分隔符
        writer = csv.writer(f, delimiter="\t")
        # 写入一行标题行到 CSV 文件，包含以下列名
        writer.writerow(
            [
                "a",
                "b",
                "c",
                "z",
                "|z|",
                "region",
                "parameter_group",
                "expected",  # mpmath 的 hyp2f1
                "observed",  # scipy 的 hyp2f1
                "relative_error",
                "absolute_error",
            ]
        )
        # 遍历每一行数据并写入到 CSV 文件中
        for row in rows:
            writer.writerow(row)
# 如果脚本被直接执行而不是被导入为模块，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description="Test scipy's hyp2f1 against mpmath's on a grid in the"
        " complex plane over a grid of parameter values. Saves output to file"
        " specified in positional argument \"outpath\"."
        " Caution: With default arguments, the generated output file is"
        " roughly 700MB in size. Script may take several hours to finish if"
        " \"--n_jobs\" is set to 1."
    )
    # 添加位置参数：输出文件的路径
    parser.add_argument(
        "outpath", type=str, help="Path to output tsv file."
    )
    # 添加可选参数：用于多进程计算的作业数，默认为1
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs for multiprocessing.",
    )
    # 添加可选参数：定义计算区域的盒子大小，默认为2.0
    parser.add_argument(
        "--box_size",
        type=float,
        default=2.0,
        help="hyp2f1 is evaluated in box of side_length 2*box_size centered"
        " at the origin."
    )
    # 添加可选参数：定义计算网格大小的整数，默认为20
    parser.add_argument(
        "--grid_size",
        type=int,
        default=20,
        help="hyp2f1 is evaluated on grid_size * grid_size grid in box of side"
        " length 2*box_size centered at the origin."
    )
    # 添加可选参数：限制计算的参数组，默认为无限制
    parser.add_argument(
        "--parameter_groups",
        type=int,
        nargs='+',
        default=None,
        help="Restrict to supplied parameter groups. See the Docstring for"
        " this module for more info on parameter groups. Calculate for all"
        " parameter groups by default."
    )
    # 添加可选参数：限制 z 参数的计算区域，默认为无限制
    parser.add_argument(
        "--regions",
        type=int,
        nargs='+',
        default=None,
        help="Restrict to argument z only within the supplied regions. See"
        " the Docstring for this module for more info on regions. Calculate"
        " for all regions by default."
    )
    # 添加可选参数：如果设置此标志，则不使用 mpmath 计算结果
    parser.add_argument(
        "--no_mp",
        action='store_true',
        help="If this flag is set, do not compute results with mpmath. Saves"
        " time if results have already been computed elsewhere. Fills in"
        " \"expected\" column with None values."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 计算标志，如果 --no_mp 没有设置，则为 True，否则为 False
    compute_mp = not args.no_mp
    # 打印参数组信息（用于调试或显示）
    print(args.parameter_groups)
    # 调用主函数 main()，传递解析得到的参数作为参数传入
    main(
        args.outpath,
        n_jobs=args.n_jobs,
        box_size=args.box_size,
        grid_size=args.grid_size,
        parameter_groups=args.parameter_groups,
        regions=args.regions,
        compute_mp=compute_mp,
    )
```