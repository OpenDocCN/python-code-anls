# `.\pytorch\benchmarks\operator_benchmark\benchmark_utils.py`

```py
# 导入必要的库
import argparse  # 用于处理命令行参数
import bisect  # 提供二分查找算法
import itertools  # 提供迭代工具，用于生成迭代器
import os  # 提供与操作系统交互的功能
import random  # 提供随机数生成功能

import numpy as np  # 导入 NumPy 库，用于数值计算


"""Performance microbenchmarks's utils.

This module contains utilities for writing microbenchmark tests.
"""

# 定义基准测试套件中的保留关键字
_reserved_keywords = {"probs", "total_samples", "tags"}
_supported_devices = {"cpu", "cuda"}


def shape_to_string(shape):
    # 将形状信息转换为字符串格式
    return ", ".join([str(x) for x in shape])


def str2bool(v):
    # 将字符串转换为布尔值
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def numpy_random(dtype, *shapes):
    """Return a random numpy tensor of the provided dtype.
    Args:
        shapes: int or a sequence of ints to defining the shapes of the tensor
        dtype: use the dtypes from numpy
            (https://docs.scipy.org/doc/numpy/user/basics.types.html)
    Return:
        numpy tensor of dtype
    """
    # 使用随机数生成随机的 NumPy 张量
    # TODO: consider more complex/custom dynamic ranges for
    # comprehensive test coverage.
    return np.random.rand(*shapes).astype(dtype)


def set_omp_threads(num_threads):
    # 设置 OpenMP 的线程数
    existing_value = os.environ.get("OMP_NUM_THREADS", "")
    if existing_value != "":
        print(
            f"Overwriting existing OMP_NUM_THREADS value: {existing_value}; Setting it to {num_threads}."
        )
    os.environ["OMP_NUM_THREADS"] = str(num_threads)


def set_mkl_threads(num_threads):
    # 设置 MKL 的线程数
    existing_value = os.environ.get("MKL_NUM_THREADS", "")
    if existing_value != "":
        print(
            f"Overwriting existing MKL_NUM_THREADS value: {existing_value}; Setting it to {num_threads}."
        )
    os.environ["MKL_NUM_THREADS"] = str(num_threads)


def cross_product(*inputs):
    """
    Return a list of cartesian product of input iterables.
    For example, cross_product(A, B) returns ((x,y) for x in A for y in B).
    """
    # 返回输入迭代器的笛卡尔积列表
    return list(itertools.product(*inputs))


def get_n_rand_nums(min_val, max_val, n):
    # 返回在指定范围内随机抽样的 n 个整数列表
    random.seed((1 << 32) - 1)
    return random.sample(range(min_val, max_val), n)


def generate_configs(**configs):
    """
    Given configs from users, we want to generate different combinations of
    those configs
    For example, given M = ((1, 2), N = (4, 5)) and sample_func being cross_product,
    we will generate (({'M': 1}, {'N' : 4}),
                      ({'M': 1}, {'N' : 5}),
                      ({'M': 2}, {'N' : 4}),
                      ({'M': 2}, {'N' : 5}))
    """
    # 检查是否提供了生成配置的样本函数
    assert "sample_func" in configs, "Missing sample_func to generate configs"
    result = []
    for key, values in configs.items():
        if key == "sample_func":
            continue
        tmp_result = []
        for value in values:
            tmp_result.append({key: value})
        result.append(tmp_result)

    # 使用提供的样本函数生成不同的配置组合
    results = configs["sample_func"](*result)
    return results
def cross_product_configs(**configs):
    """
    Given configs from users, we want to generate different combinations of
    those configs
    For example, given M = ((1, 2), N = (4, 5)),
    we will generate (({'M': 1}, {'N' : 4}),
                      ({'M': 1}, {'N' : 5}),
                      ({'M': 2}, {'N' : 4}),
                      ({'M': 2}, {'N' : 5}))
    """
    # Validate the input configurations against supported devices
    _validate(configs)
    # Initialize an empty list to store attribute configurations
    configs_attrs_list = []
    # Iterate over each key (attribute name) and values (list of values for that attribute)
    for key, values in configs.items():
        # Generate a list of dictionaries where each dictionary contains one attribute and one of its values
        tmp_results = [{key: value} for value in values]
        # Append the list of dictionaries to configs_attrs_list
        configs_attrs_list.append(tmp_results)

    # Generate all possible combinations of attribute dictionaries using itertools.product
    generated_configs = list(itertools.product(*configs_attrs_list))
    # Return the generated configurations
    return generated_configs


def _validate(configs):
    """Validate inputs from users."""
    # Check if 'device' is in the input configurations
    if "device" in configs:
        # Iterate over each value in the 'device' list
        for v in configs["device"]:
            # Ensure each device is in the supported devices list
            assert v in _supported_devices, "Device needs to be a string."


def config_list(**configs):
    """Generate configs based on the list of input shapes.
    This function will take input shapes specified in a list from user. Besides
    that, all other parameters will be cross producted first and each of the
    generated list will be merged with the input shapes list.

    Reserved Args:
        attr_names(reserved): a list of names for input shapes.
        attrs(reserved): a list of values for each input shape.
        corss_product: a dictionary of attributes which will be
                       cross producted with the input shapes.
        tags(reserved): a tag used to filter inputs.

    Here is an example:
    attrs = [
        [1, 2],
        [4, 5],
    ],
    attr_names = ['M', 'N'],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },

    we will generate [[{'M': 1}, {'N' : 2}, {'device' : 'cpu'}],
                      [{'M': 1}, {'N' : 2}, {'device' : 'cuda'}],
                      [{'M': 4}, {'N' : 5}, {'device' : 'cpu'}],
                      [{'M': 4}, {'N' : 5}, {'device' : 'cuda'}]]
    """
    # Initialize an empty list to store generated configurations
    generated_configs = []
    # Define reserved attribute names that should be present in configs
    reserved_names = ["attrs", "attr_names", "tags"]
    # Check if any reserved attribute is missing in configs
    if any(attr not in configs for attr in reserved_names):
        # Raise an error if any reserved attribute is missing
        raise ValueError("Missing attrs in configs")

    # Validate the input configurations against supported devices
    _validate(configs)

    # Initialize cross_configs as None
    cross_configs = None
    # Check if 'cross_product_configs' is present in configs
    if "cross_product_configs" in configs:
        # Generate cross products of configurations using cross_product_configs function
        cross_configs = cross_product_configs(**configs["cross_product_configs"])
    # 遍历配置文件中的"attrs"列表，每个元素被命名为inputs
    for inputs in configs["attrs"]:
        # 对于每个inputs中的元素，创建一个字典包含属性名和对应的值
        tmp_result = [
            {configs["attr_names"][i]: input_value}
            for i, input_value in enumerate(inputs)
        ]
        # 向tmp_result列表中添加一个字典，键为"tags"，值为由configs["tags"]列表元素用"_"连接而成的字符串
        tmp_result.append({"tags": "_".join(configs["tags"])})
        # 如果存在交叉配置项cross_configs
        if cross_configs:
            # 将tmp_result与每个cross_configs元组合并，并将结果添加到generated_configs列表中
            generated_configs += [tmp_result + list(config) for config in cross_configs]
        else:
            # 否则，直接将tmp_result添加到generated_configs列表中
            generated_configs.append(tmp_result)

    # 返回生成的配置列表generated_configs
    return generated_configs
def attr_probs(**probs):
    """返回以字典形式返回输入"""
    return probs


class RandomSample:
    def __init__(self, configs):
        """初始化随机抽样器对象"""
        self.saved_cum_distribution = {}
        self.configs = configs

    def _distribution_func(self, key, weights):
        """这是用于随机抽样输入的累积分布函数"""
        if key in self.saved_cum_distribution:
            return self.saved_cum_distribution[key]

        total = sum(weights)
        result = []
        cumsum = 0
        for w in weights:
            cumsum += w
            result.append(cumsum / total)
        self.saved_cum_distribution[key] = result
        return result

    def _random_sample(self, key, values, weights):
        """给定值和权重，该函数根据权重随机抽样值"""
        # TODO(mingzhe09088): 缓存结果以避免重新计算的开销
        assert len(values) == len(weights)
        _distribution_func_vals = self._distribution_func(key, weights)
        x = random.random()
        idx = bisect.bisect(_distribution_func_vals, x)

        assert idx <= len(values), "返回了错误的索引值"
        # 由于数值属性，累积和中的最后一个值可能略小于1，导致（index == len(values)）。
        if idx == len(values):
            idx -= 1
        return values[idx]

    def get_one_set_of_inputs(self):
        """获取一组随机输入"""
        tmp_attr_list = []
        for key, values in self.configs.items():
            if key in _reserved_keywords:
                continue
            value = self._random_sample(key, values, self.configs["probs"][str(key)])
            tmp_results = {key: value}
            tmp_attr_list.append(tmp_results)
        return tmp_attr_list


def random_sample_configs(**configs):
    """
    这个函数根据权重随机抽样给定输入的<total_samples>个值。
    下面是一个示例展示了此函数的期望输入和输出：
    M = [1, 2],
    N = [4, 5],
    K = [7, 8],
    probs = attr_probs(
        M = [0.7, 0.2],
        N = [0.5, 0.2],
        K = [0.6, 0.2],
    ),
    total_samples=10,
    这个函数将生成
    [
        [{'K': 7}, {'M': 1}, {'N': 4}],
        [{'K': 7}, {'M': 2}, {'N': 5}],
        [{'K': 8}, {'M': 2}, {'N': 4}],
        ...
    ]
    注意：
    probs 是可选的。如果没有提供，则表示所有的概率权重都为1。probs 不必反映实际的归一化概率，
    实现将对其进行归一化处理。
    TODO (mingzhe09088):
    (1): 接受或拒绝配置作为样本的 lambda 函数。例如：对于具有 M、N 和 K 的矩阵乘法，
         此函数可以去掉 (M * N * K > 1e8) 以过滤掉非常慢的基准测试。
    (2): 确保每个样本是唯一的。如果样本数大于总组合数，则返回交叉乘积。否则，如果样本数
    # 检查配置字典中是否包含键"probs"，如果不包含则抛出数值错误异常
    if "probs" not in configs:
        raise ValueError(
            "probs is missing. Consider adding probs or using other config functions"
        )

    # 初始化空列表，用于存储配置属性列表
    configs_attrs_list = []

    # 创建随机抽样对象，传入配置字典
    randomsample = RandomSample(configs)

    # 循环生成指定次数的样本集合
    for i in range(configs["total_samples"]):
        # 获取一个随机生成的输入属性列表
        tmp_attr_list = randomsample.get_one_set_of_inputs()
        
        # 将标签列表连接成字符串，并作为单个元素添加到临时属性列表末尾
        tmp_attr_list.append({"tags": "_".join(configs["tags"])})
        
        # 将生成的临时属性列表添加到总配置属性列表中
        configs_attrs_list.append(tmp_attr_list)
    
    # 返回生成的配置属性列表
    return configs_attrs_list
# 生成一个按特定格式组织的操作列表。
# 接受两个参数：attr_names 和 attr，分别存储操作符的名称和函数。
def op_list(**configs):
    """Generate a list of ops organized in a specific format.
    It takes two parameters which are "attr_names" and "attr".
    attrs stores the name and function of operators.
    Args:
        configs: key-value pairs including the name and function of
        operators. attrs and attr_names must be present in configs.
    Return:
        a sequence of dictionaries which stores the name and function
        of ops in a specifal format
    Example:
    attrs = [
        ["abs", torch.abs],
        ["abs_", torch.abs_],
    ]
    attr_names = ["op_name", "op"].

    With those two examples,
    we will generate (({"op_name": "abs"}, {"op" : torch.abs}),
                      ({"op_name": "abs_"}, {"op" : torch.abs_}))
    """
    # 用于存储生成的配置列表
    generated_configs = []
    
    # 检查 configs 中是否包含 attrs 键
    if "attrs" not in configs:
        # 如果缺少 attrs 键，则抛出数值错误异常
        raise ValueError("Missing attrs in configs")
    
    # 遍历 attrs 中的每一个输入，生成临时结果字典并添加到 generated_configs 中
    for inputs in configs["attrs"]:
        tmp_result = {
            configs["attr_names"][i]: input_value
            for i, input_value in enumerate(inputs)
        }
        generated_configs.append(tmp_result)
    
    # 返回生成的配置列表
    return generated_configs


def get_operator_range(chars_range):
    """Generates the characters from chars_range inclusive."""
    # 如果 chars_range 为 "None" 或者 None，则返回 None
    if chars_range == "None" or chars_range is None:
        return None
    
    # 检查 chars_range 是否符合正确的格式
    if all(item not in chars_range for item in [",", "-"]):
        # 如果不是正确的格式，则抛出值错误异常
        raise ValueError(
            "The correct format for operator_range is "
            "<start>-<end>, or <point>, <start>-<end>"
        )
    
    # 用于存储操作符的起始字符集合
    ops_start_chars_set = set()
    
    # 按逗号分隔 chars_range，并处理每个子项
    ranges = chars_range.split(",")
    for item in ranges:
        # 如果子项长度为 1，则将其小写形式添加到 ops_start_chars_set 中
        if len(item) == 1:
            ops_start_chars_set.add(item.lower())
            continue
        # 否则，按照范围 <start>-<end> 的格式处理，并将生成的字符小写形式添加到 ops_start_chars_set 中
        start, end = item.split("-")
        ops_start_chars_set.update(
            chr(c).lower() for c in range(ord(start), ord(end) + 1)
        )
    
    # 返回操作符的起始字符集合
    return ops_start_chars_set


def process_arg_list(arg_list):
    # 如果 arg_list 为 "None"，则返回 None
    if arg_list == "None":
        return None
    
    # 否则，按逗号分隔 arg_list，并去除每个项两端的空格，生成列表并返回
    return [fr.strip() for fr in arg_list.split(",") if len(fr.strip()) > 0]
```