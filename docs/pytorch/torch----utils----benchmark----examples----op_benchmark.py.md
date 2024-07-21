# `.\pytorch\torch\utils\benchmark\examples\op_benchmark.py`

```py
# mypy: allow-untyped-defs
"""Example use of Timer and op fuzzers to measure kernel performance.

$ python -m examples.op_benchmark
"""

import numpy as np                      # 导入 NumPy 库，用于数值计算
import torch                            # 导入 PyTorch 库，用于深度学习任务

from torch.utils.benchmark import Timer # 导入 Timer 类，用于性能测量
from torch.utils.benchmark.op_fuzzers.binary import BinaryOpFuzzer  # 导入二元操作的模糊器类
from torch.utils.benchmark.op_fuzzers.unary import UnaryOpFuzzer    # 导入一元操作的模糊器类
import operator                         # 导入 operator 模块，用于操作符的函数实现


_MEASURE_TIME = 1.0  # 定义测量时间为1.0秒


def assert_dicts_equal(dict_0, dict_1):
    """Builtin dict comparison will not compare numpy arrays.
    e.g.
        x = {"a": np.ones((2, 1))}
        x == x  # Raises ValueError
    """
    assert set(dict_0.keys()) == set(dict_0.keys())  # 断言两个字典的键集合相等
    assert all(np.all(v == dict_1[k]) for k, v in dict_0.items() if k != "dtype")  # 断言除了键为"dtype"外，所有值相等


def run(n, stmt, fuzzer_cls):
    float_iter = fuzzer_cls(seed=0, dtype=torch.float32).take(n)  # 使用指定种子生成 torch.float32 类型的模糊数据迭代器
    int_iter = fuzzer_cls(seed=0, dtype=torch.int32).take(n)      # 使用指定种子生成 torch.int32 类型的模糊数据迭代器
    raw_results = []
    for i, (float_values, int_values) in enumerate(zip(float_iter, int_iter)):
        float_tensors, float_tensor_params, float_params = float_values  # 解包获取浮点数模糊数据的张量、参数和额外参数
        int_tensors, int_tensor_params, int_params = int_values         # 解包获取整数模糊数据的张量、参数和额外参数

        # This benchmark assumes that the two fuzzers generate identically
        # sized and strided Tensors, since the same seed is used.
        assert_dicts_equal(float_params, int_params)  # 断言浮点数和整数参数字典相等
        assert_dicts_equal(float_tensor_params["x"], int_tensor_params["x"])  # 断言浮点数和整数张量参数字典中"x"键对应的值相等

        float_measurement, int_measurement = (
            Timer(
                stmt,
                globals=tensors,
            ).blocked_autorange(min_run_time=_MEASURE_TIME)
            for tensors in (float_tensors, int_tensors)
        )

        descriptions = []
        for name in float_tensors:
            shape_str = "(" + ", ".join([
                f"2 ** {int(np.log2(i))}"
                if 2 ** int(np.log2(i)) == i and i > 1
                else str(i)
                for i in float_tensors[name].shape
            ]) + ")"
            order = float_tensor_params[name]["order"]
            order_str = ("" if all(order == np.arange(len(order))) else str(tuple(order)))
            steps = float_tensor_params[name]["steps"]
            steps_str = str(steps) if sum(steps) > len(steps) else ""
            descriptions.append((name, shape_str, order_str, steps_str))
        raw_results.append((float_measurement, int_measurement, descriptions))

        print(f"\r{i + 1} / {n}", end="")
    print()

    parsed_results, name_len, shape_len, order_len, steps_len = [], 0, 0, 0, 0
    # 遍历原始结果中的每个元组，每个元组包含浮点数测量值、整数测量值和描述信息列表
    for float_measurement, int_measurement, descriptions in raw_results:
        # 计算浮点数测量值的中位数并乘以 10^6，转换成微秒
        t_float = float_measurement.median * 1e6
        # 计算整数测量值的中位数并乘以 10^6，转换成微秒
        t_int = int_measurement.median * 1e6
        # 计算相对差异，使用浮点数和整数的中位数，计算绝对差异占总和的比例乘以 2
        rel_diff = abs(t_float - t_int) / (t_float + t_int) * 2
        # 将计算结果以及描述信息添加到解析后的结果列表中
        parsed_results.append((t_float, t_int, rel_diff, descriptions))
        # 遍历描述信息列表中的每个元组，更新用于对齐输出的长度变量
        for name, shape, order, steps in descriptions:
            name_len = max(name_len, len(name))
            shape_len = max(shape_len, len(shape))
            order_len = max(order_len, len(order))
            steps_len = max(steps_len, len(steps))

    # 根据相对差异对解析后的结果列表进行排序
    parsed_results.sort(key=operator.itemgetter(2))

    # 输出语句的字符串表示
    print(f"stmt: {stmt}")
    # 输出表头行，并根据名称的最大长度对其进行格式化
    print(f" diff    faster{'':>17}{' ' * name_len} ", end="")
    # 输出形状、顺序和步骤的表头，并根据各自的最大长度进行格式化
    print(f"{'shape'.ljust(shape_len)}{'':>16}{'order'.ljust(order_len)}", end="")
    print(f"          steps\n{'-' * 100}")
    # 遍历解析后的结果列表的前10个和最后10个元组
    for results, spacer in [(parsed_results[:10], "..."), (parsed_results[-10:], "")]:
        # 遍历每个元组的浮点数测量值、整数测量值、相对差异和描述信息
        for t_float, t_int, rel_diff, descriptions in results:
            # 创建时间字符串列表，包含相对差异和哪个实现更快的信息
            time_str = [f"{rel_diff * 100:>4.1f}%    {'int' if t_int < t_float else 'float':<20}"]
            # 根据描述信息的顺序，扩展时间字符串列表
            time_str.extend(["".ljust(len(time_str[0])) for _ in descriptions[:-1]])
            # 遍历时间字符串和描述信息元组，格式化输出每一行
            for t_str, (name, shape, order, steps) in zip(time_str, descriptions):
                name = f"{name}:".ljust(name_len + 1)
                shape = shape.ljust(shape_len + 10)
                order = order.ljust(order_len)
                print(f"{t_str} {name}  {shape}|     {order}      |   {steps}")
        # 输出分隔符行或省略号，以区分解析结果的不同部分
        print(spacer)
# 主函数入口，程序从这里开始执行
def main():
    # 调用 run 函数，运行 UnaryOpFuzzer 类型的模糊测试，测试 torch.median(x, dim=0)
    run(n=100, stmt="torch.median(x, dim=0)", fuzzer_cls=UnaryOpFuzzer)
    # 调用 run 函数，再次运行 UnaryOpFuzzer 类型的模糊测试，测试 torch.square(x)
    run(n=100, stmt="torch.square(x)", fuzzer_cls=UnaryOpFuzzer)
    # 调用 run 函数，运行 BinaryOpFuzzer 类型的模糊测试，测试 x + y

# 如果当前脚本作为主程序执行，则调用 main 函数
if __name__ == "__main__":
    main()
```