# `.\pytorch\benchmarks\tensorexpr\microbenchmarks.py`

```py
import argparse
import operator
import time

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，并重命名为plt
import numpy as np  # 导入numpy库，并重命名为np
import pandas as pd  # 导入pandas库，并重命名为pd
import seaborn as sns  # 导入seaborn库，并重命名为sns

import torch  # 导入torch库
import torch._C._te as te  # 导入torch._C._te模块，并重命名为te

# 定义一个上下文管理器类kernel_arena_scope
class kernel_arena_scope:
    def __enter__(self):
        self.scope = te.KernelScope()  # 进入时创建一个te.KernelScope对象

    def __exit__(self, typ, val, traceback):
        self.scope = None  # 退出时将scope设置为None


# 包含一元操作的列表
unary_ops = [
    ("sin", torch.sin),
    ("cos", torch.cos),
    ("tan", torch.tan),
    ("asin", torch.asin),
    ("acos", torch.acos),
    ("atan", torch.atan),
    ("sinh", torch.sinh),
    ("cosh", torch.cosh),
    ("tanh", torch.tanh),
    ("sigmoid", torch.sigmoid),
    ("exp", torch.exp),
    ("expm1", torch.expm1),
    ("expm1", torch.expm1),  # 注意，expm1重复了
    ("abs", torch.abs),
    ("log", torch.log),
    ("fast_log", torch.log),  # 注意，fast_log也重复了
    ("log2", torch.log2),
    ("log10", torch.log10),
    ("log1p", torch.log1p),
    ("erf", torch.erf),
    ("erfc", torch.erfc),
    ("sqrt", torch.sqrt),
    ("rsqrt", torch.rsqrt),
    ("ceil", torch.ceil),
    ("floor", torch.floor),
    ("round", torch.round),
    ("trunc", torch.trunc),
    ("lgamma", torch.lgamma),
    # ("frac", torch.frac), # 未实现
    # ("isnan", torch.isnan), # 没有out变体
]

# 生成一个非线性计算的nnc函数
def gen_unary_nnc_fun(nnc_name):
    def nnc_fun(A, B):
        def compute(i, j):
            return getattr(A.load([i, j]), nnc_name)()  # 使用getattr调用A.load([i, j])对象的nnc_name方法

        return compute

    return nnc_fun

# 生成一个torch一元函数的函数
def gen_unary_torch_fun(torch_op):
    def torch_fun(a, b, out):
        def fun():
            return torch_op(a, out=out)

        return fun

    return torch_fun

# 生成一个二元nnc函数
def gen_binary_nnc_fun(fn):
    def nnc_fun(A, B):
        def compute(i, j):
            return fn(A.load([i, j]), B.load([i, j]))  # 使用fn函数处理A.load([i, j])和B.load([i, j])

        return compute

    return nnc_fun

# 生成一个torch二元函数的函数
def gen_binary_torch_fun(fn):
    def pt_fun(a, b, out):
        def fun():
            return fn(a, b, out=out)

        return fun

    return pt_fun

# 生成整数比较张量
def gen_int_comparison_tensors(N, M):
    return (
        torch.randint(0, 3, (N, M)),
        torch.randint(0, 3, (N, M)),
        torch.empty((N, M), dtype=torch.bool),
    )

# 生成浮点数比较张量
def gen_float_comparison_tensors(N, M):
    return (torch.rand(N, M), torch.rand(N, M), torch.empty((N, M), dtype=torch.bool))

# 定义te_bool为te.Dtype.Bool
te_bool = te.Dtype.Bool

# 包含二元操作的列表
binary_ops = [
    ("add", operator.add, torch.add),
    ("mul", operator.mul, torch.mul),
    ("sub", operator.sub, torch.sub),
    ("div", operator.truediv, torch.div),
    (
        "eq",
        (lambda a, b: te.Cast.make(te_bool, a == b)),  # 使用lambda表达式创建一个a == b的比较操作，并使用te.Cast.make转换为te_bool类型
        torch.eq,
        gen_int_comparison_tensors,  # 使用gen_int_comparison_tensors生成整数比较张量
    ),
    (
        "gt",
        (lambda a, b: te.Cast.make(te_bool, a > b)),  # 使用lambda表达式创建一个a > b的比较操作，并使用te.Cast.make转换为te_bool类型
        torch.gt,
        gen_float_comparison_tensors,  # 使用gen_float_comparison_tensors生成浮点数比较张量
    ),
    (
        "lt",
        (lambda a, b: te.Cast.make(te_bool, a < b)),  # 使用lambda表达式创建一个a < b的比较操作，并使用te.Cast.make转换为te_bool类型
        torch.lt,
        gen_float_comparison_tensors,  # 使用gen_float_comparison_tensors生成浮点数比较张量
    ),
    (
        "gte",
        (lambda a, b: te.Cast.make(te_bool, a >= b)),  # 使用lambda表达式创建一个a >= b的比较操作，并使用te.Cast.make转换为te_bool类型
        torch.greater_equal,
        gen_float_comparison_tensors,  # 使用gen_float_comparison_tensors生成浮点数比较张量
    ),
    (
        "lte",  # 操作符标识符，表示小于等于操作
        (lambda a, b: te.Cast.make(te_bool, a <= b)),  # 匿名函数，生成 tvm.expr.Cast 节点，将 a <= b 的结果转换为布尔类型
        torch.less_equal,  # torch 模块中的小于等于运算函数
        gen_float_comparison_tensors,  # 用于生成浮点数比较张量的函数
    ),
    # ('neq', (lambda a, b: a != b), None)), # no one-op equivalent
    # ('&', (lambda a, b: a & b), torch.bitwise_and), # requires more work to test
def nnc_relu(A, B):
    def f(i, j):
        # 如果A[i, j] < 0，则返回0，否则返回A[i, j]
        return torch._C._te.ifThenElse(
            A.load([i, j]) < torch._C._te.ExprHandle.float(0),
            torch._C._te.ExprHandle.float(0),
            A.load([i, j]),
        )

    return f


def pt_relu(a, b, c):
    # 使用PyTorch内置的ReLU函数
    return torch.relu(a)


custom_ops = [
    ("relu", nnc_relu, pt_relu),
    # ('nnc_mul_relu', nnc_mul_relu, pt_mul_relu)
    # ('manual_sigmoid', nnc_manual_sigmoid, lambda a, b, c: torch.sigmoid(a, out=c))
]


def gen_custom_torch_fun(fn):
    def pt_fun(a, b, out):
        def fun():
            # 调用给定的PyTorch函数fn，并返回其结果
            return fn(a, b, out)

        return fun

    return pt_fun


def normalize_benchmarks(ops):
    # 对操作列表进行规范化，确保每个元组都包含三个元素
    return [i + (None,) if len(i) == 3 else i for i in ops]


names = []
nnc_fns = []
pt_fns = []
shape_fns = []

# 处理一元操作
for nnc_name, pt_op in unary_ops:
    names.append(nnc_name)
    nnc_fns.append(gen_unary_nnc_fun(nnc_name))
    pt_fns.append(gen_unary_torch_fun(pt_op))
    shape_fns.append(None)

# 处理二元操作
for name, lmbda, pt_fn, shape_fn in normalize_benchmarks(binary_ops):
    names.append(name)
    nnc_fns.append(gen_binary_nnc_fun(lmbda))
    pt_fns.append(gen_binary_torch_fun(pt_fn))
    shape_fns.append(shape_fn)

# 处理自定义操作
for name, lmbda, pt_fn, shape_fn in normalize_benchmarks(custom_ops):
    names.append(name)
    nnc_fns.append(lmbda)
    pt_fns.append(gen_custom_torch_fun(pt_fn))
    shape_fns.append(shape_fn)

# 将操作函数和相关信息打包成元组列表
benchmarks = list(zip(names, nnc_fns, pt_fns, shape_fns))


def run_benchmarks(benchmarks, sizes):
    # 运行基准测试，返回结果DataFrame
    df = pd.DataFrame(columns=["name", "N", "M", "nnc_time", "torch_time", "ratio"])
    return df


def dump_plot(df, sizes):
    # 绘制热图，比较PyTorch性能和NNC性能的比例
    keys = []
    vals = []
    indexed = df[df["N"] == df["M"]]
    for index, row in indexed.iterrows():
        keys.append(row["name"])
        vals.append(row["ratio"])

    keys = keys[:: len(sizes)]
    sns.set(rc={"figure.figsize": (5.0, len(keys) * 0.5)})

    cmap = sns.diverging_palette(10, 120, n=9, as_cmap=True)
    np_vals = np.array([vals]).reshape(-1, len(sizes))
    g = sns.heatmap(np_vals, annot=True, cmap=cmap, center=1.0, yticklabels=True)
    plt.yticks(rotation=0)
    plt.title("PyTorch performance divided by NNC performance (single core)")
    plt.xlabel("Size of NxN matrix")
    plt.ylabel("Operation")
    g.set_yticklabels(keys)
    g.set_xticklabels(sizes)

    plt.savefig("nnc.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs NNC microbenchmarks")
    parser.add_argument(
        "--multi-threaded",
        "--multi_threaded",
        action="store_true",
        help="Run with more than one thread",
    )
    args = parser.parse_args()
    if not args.multi_threaded:
        # 如果未指定多线程，设置PyTorch线程数为1
        torch.set_num_threads(1)

    sizes = [1, 4, 16, 64, 256, 1024]
    df = run_benchmarks(benchmarks, [(i, i) for i in sizes])
    dump_plot(df, sizes)
```