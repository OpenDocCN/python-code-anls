# `.\pytorch\torch\utils\benchmark\examples\sparse\op_benchmark.py`

```
# 引入必要的库和模块
import numpy as np
import torch

# 从torch.utils.benchmark中引入计时器Timer以及稀疏操作模糊器
from torch.utils.benchmark import Timer
from torch.utils.benchmark.op_fuzzers.sparse_unary import UnaryOpSparseFuzzer
from torch.utils.benchmark.op_fuzzers.sparse_binary import BinaryOpSparseFuzzer
import operator

# 测量时间的常量设定为1.0秒
_MEASURE_TIME = 1.0

# 定义一个函数用于比较两个字典是否相等，考虑到numpy数组的情况
def assert_dicts_equal(dict_0, dict_1):
    """Builtin dict comparison will not compare numpy arrays.
    e.g.
        x = {"a": np.ones((2, 1))}
        x == x  # Raises ValueError
    """
    # 检查字典的键集合是否相等
    assert set(dict_0.keys()) == set(dict_0.keys())
    # 检查字典中每个值是否相等，对于键名不为"dtype"的项，比较其值是否相等
    assert all(np.all(v == dict_1[k]) for k, v in dict_0.items() if k != "dtype")

# 定义一个函数用于运行指定次数的测试，并测量性能
def run(n, stmt, fuzzer_cls):
    # 创建两个稀疏操作模糊器的实例，一个使用单精度浮点数，一个使用双精度浮点数
    float_iter = fuzzer_cls(seed=0, dtype=torch.float32).take(n)
    double_iter = fuzzer_cls(seed=0, dtype=torch.float64).take(n)
    raw_results = []
    # 对每对单精度和双精度浮点数进行迭代
    for i, (float_values, int_values) in enumerate(zip(float_iter, double_iter)):
        # 分别获取浮点数和整数模糊器生成的张量、参数和其他信息
        float_tensors, float_tensor_params, float_params = float_values
        int_tensors, int_tensor_params, int_params = int_values

        # 比较浮点数和整数参数字典是否相等
        assert_dicts_equal(float_params, int_params)
        # 比较浮点数和整数张量参数字典中"x"键对应的值是否相等
        assert_dicts_equal(float_tensor_params["x"], int_tensor_params["x"])

        # 使用Timer测量给定语句的执行时间，分别在浮点数和整数张量上进行
        float_measurement, int_measurement = (
            Timer(
                stmt,
                globals=tensors,
            ).blocked_autorange(min_run_time=_MEASURE_TIME)
            for tensors in (float_tensors, int_tensors)
        )

        # 构建描述信息列表，描述每个张量的名称、形状、稀疏维度和是否紧凑
        descriptions = []
        for name in float_tensors:
            shape_str = "(" + ", ".join([
                f"2 ** {int(np.log2(i))}"
                if 2 ** int(np.log2(i)) == i and i > 1
                else str(i)
                for i in float_tensors[name].shape
            ]) + ")"
            sparse_dim = float_tensor_params[name]["sparse_dim"]
            sparse_dim_str = str(sparse_dim)
            is_coalesced = float_tensor_params[name]["is_coalesced"]
            is_coalesced_str = "True" if is_coalesced else "False"
            descriptions.append((name, shape_str, sparse_dim_str, is_coalesced_str))
        raw_results.append((float_measurement, int_measurement, descriptions))

        # 打印进度条，显示当前完成的测试次数
        print(f"\r{i + 1} / {n}", end="")
    print()

    # 解析原始结果，计算中位数执行时间并保存描述信息的长度
    parsed_results, name_len, shape_len, sparse_dim_len, is_coalesced_len = [], 0, 0, 0, 0
    for float_measurement, int_measurement, descriptions in raw_results:
        t_float = float_measurement.median * 1e6
        t_int = int_measurement.median * 1e6
        rel_diff = abs(t_float - t_int) / (t_float + t_int) * 2
        parsed_results.append((t_float, t_int, rel_diff, descriptions))
        for name, shape, sparse_dim, is_coalesced in descriptions:
            name_len = max(name_len, len(name))
            shape_len = max(shape_len, len(shape))
            sparse_dim_len = max(sparse_dim_len, len(sparse_dim))
            is_coalesced_len = max(is_coalesced_len, len(is_coalesced))
    # 按照第三个元素（相对差异）对解析结果进行排序
    parsed_results.sort(key=operator.itemgetter(2))
    
    # 打印语句的字符串表示
    print(f"stmt: {stmt}")
    
    # 打印表头，描述解析结果的格式和字段
    print(f" diff    faster{'':>17}{' ' * name_len} ", end="")
    print(f"{'shape'.ljust(shape_len)}{'':>12}{'sparse_dim'.ljust(sparse_dim_len)}", end="")
    print(f"          is_coalesced\n{'-' * 100}")
    
    # 遍历解析结果中的前10个和后10个元素
    for results, spacer in [(parsed_results[:10], "..."), (parsed_results[-10:], "")]:
        # 遍历每个结果中的时间浮点数、时间整数、相对差异和描述
        for t_float, t_int, rel_diff, descriptions in results:
            # 构建时间字符串列表，包含相对差异和更快的数据类型（整数或浮点数）
            time_str = [f"{rel_diff * 100:>4.1f}%    {'int' if t_int < t_float else 'float':<20}"]
            # 填充空字符串以匹配描述列表的长度
            time_str.extend(["".ljust(len(time_str[0])) for _ in descriptions[:-1]])
            # 遍历每个描述项，打印出名称、形状、稀疏维度和是否紧凑
            for t_str, (name, shape, sparse_dim, is_coalesced) in zip(time_str, descriptions):
                name = f"{name}:".ljust(name_len + 1)
                shape = shape.ljust(shape_len + 10)
                sparse_dim = sparse_dim.ljust(sparse_dim_len)
                print(f"{t_str} {name}  {shape}|     {sparse_dim}      |   {is_coalesced}")
        # 打印分隔符（"..."或空字符串）
        print(spacer)
# 主函数入口，程序的执行从这里开始
def main():
    # 调用 run 函数，生成并运行 100 次针对稀疏张量的 torch.sparse.sum 操作的测试用例
    run(n=100, stmt="torch.sparse.sum(x, dim=0)", fuzzer_cls=UnaryOpSparseFuzzer)
    # 调用 run 函数，生成并运行 100 次针对稀疏张量的 torch.sparse.softmax 操作的测试用例
    run(n=100, stmt="torch.sparse.softmax(x, dim=0)", fuzzer_cls=UnaryOpSparseFuzzer)
    # 调用 run 函数，生成并运行 100 次针对稀疏张量的 torch.Tensor 加法操作的测试用例
    run(n=100, stmt="x + y", fuzzer_cls=BinaryOpSparseFuzzer)

# 如果当前脚本被直接执行，则执行 main 函数
if __name__ == "__main__":
    main()
```