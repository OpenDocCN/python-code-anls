# `.\pytorch\torch\utils\benchmark\utils\fuzzer.py`

```
# 设置一个类型注解 mpy: allow-untyped-defs，允许未注释的函数定义
import functools  # 导入 functools 模块，用于高阶函数操作
import itertools as it  # 导入 itertools 模块，并使用别名 it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入多种类型提示

import torch  # 导入 PyTorch 库


__all__ = [  # 定义一个列表 __all__，包含在此模块中导出的所有公共接口
    "Fuzzer",  # 将 Fuzzer 导出
    "FuzzedParameter", "ParameterAlias",  # 导出 FuzzedParameter 和 ParameterAlias
    "FuzzedTensor",  # 导出 FuzzedTensor
]


_DISTRIBUTIONS = (  # 定义一个元组 _DISTRIBUTIONS，包含可能的分布名称
    "loguniform",  # 对数均匀分布
    "uniform",  # 均匀分布
)


class FuzzedParameter:
    """Specification for a parameter to be generated during fuzzing."""
    def __init__(  # 构造函数，用于初始化 FuzzedParameter 类
        self,
        name: str,  # 参数名，字符串类型
        minval: Optional[Union[int, float]] = None,  # 最小值，可选的整数或浮点数
        maxval: Optional[Union[int, float]] = None,  # 最大值，可选的整数或浮点数
        distribution: Optional[Union[str, Dict[Any, float]]] = None,  # 分布类型，可选的字符串或分布参数字典
        strict: bool = False,  # 是否严格模式，默认为 False
    ):
        """
        Args:
            name:
                A string name with which to identify the parameter.
                FuzzedTensors can reference this string in their
                specifications.
            minval:
                The lower bound for the generated value. See the description
                of `distribution` for type behavior.
            maxval:
                The upper bound for the generated value. Type behavior is
                identical to `minval`.
            distribution:
                Specifies the distribution from which this parameter should
                be drawn. There are three possibilities:
                    - "loguniform"
                        Samples between `minval` and `maxval` (inclusive) such
                        that the probabilities are uniform in log space. As a
                        concrete example, if minval=1 and maxval=100, a sample
                        is as likely to fall in [1, 10) as it is [10, 100].
                    - "uniform"
                        Samples are chosen with uniform probability between
                        `minval` and `maxval` (inclusive). If either `minval`
                        or `maxval` is a float then the distribution is the
                        continuous uniform distribution; otherwise samples
                        are constrained to the integers.
                    - dict:
                        If a dict is passed, the keys are taken to be choices
                        for the variables and the values are interpreted as
                        probabilities. (And must sum to one.)
                If a dict is passed, `minval` and `maxval` must not be set.
                Otherwise, they must be set.
            strict:
                If a parameter is strict, it will not be included in the
                iterative resampling process which Fuzzer uses to find a
                valid parameter configuration. This allows an author to
                prevent skew from resampling for a given parameter (for
                instance, a low size limit could inadvertently bias towards
                Tensors with fewer dimensions) at the cost of more iterations
                when generating parameters.
        """
        # Initialize instance variables based on provided arguments
        self._name = name
        self._minval = minval
        self._maxval = maxval
        # Check and set the distribution type
        self._distribution = self._check_distribution(distribution)
        # Set strict mode for the parameter
        self.strict = strict

    @property
    def name(self):
        # Getter method for the parameter name
        return self._name

    def sample(self, state):
        # Depending on the distribution type, call the appropriate sampling method
        if self._distribution == "loguniform":
            return self._loguniform(state)

        if self._distribution == "uniform":
            return self._uniform(state)

        if isinstance(self._distribution, dict):
            return self._custom_distribution(state)
    # 检查分布参数是否为字典或者预定义的分布名称
    def _check_distribution(self, distribution):
        if not isinstance(distribution, dict):
            # 如果不是字典，则检查其是否为预定义的分布名称之一
            assert distribution in _DISTRIBUTIONS
        else:
            # 如果是字典，则检查其值是否都为非负数，作为概率值
            assert not any(i < 0 for i in distribution.values()), "Probabilities cannot be negative"
            # 检查分布的总和是否接近1（用于概率分布的标准化检查）
            assert abs(sum(distribution.values()) - 1) <= 1e-5, "Distribution is not normalized"
            # 如果分布是自定义的字典类型，则要求 self._minval 和 self._maxval 都为 None
            assert self._minval is None
            assert self._maxval is None

        return distribution

    # 实现一个对数均匀分布的生成器
    def _loguniform(self, state):
        import numpy as np
        # 根据指定的最小值和最大值生成一个对数均匀分布的整数输出
        output = int(2 ** state.uniform(
            low=np.log2(self._minval) if self._minval is not None else None,
            high=np.log2(self._maxval) if self._maxval is not None else None,
        ))
        # 如果指定了最小值并且生成值小于最小值，则返回最小值
        if self._minval is not None and output < self._minval:
            return self._minval
        # 如果指定了最大值并且生成值大于最大值，则返回最大值
        if self._maxval is not None and output > self._maxval:
            return self._maxval
        # 否则返回生成的输出值
        return output

    # 实现一个均匀分布的生成器
    def _uniform(self, state):
        # 如果最小值和最大值都是整数，则生成一个整数输出
        if isinstance(self._minval, int) and isinstance(self._maxval, int):
            return int(state.randint(low=self._minval, high=self._maxval + 1))
        # 否则生成一个浮点数输出
        return state.uniform(low=self._minval, high=self._maxval)

    # 实现一个自定义分布的生成器
    def _custom_distribution(self, state):
        import numpy as np
        # 通过给定的概率分布进行随机选择一个索引
        index = state.choice(
            np.arange(len(self._distribution)),
            p=tuple(self._distribution.values()))
        # 返回对应于随机选择的键的值
        return list(self._distribution.keys())[index]
class ParameterAlias:
    """Indicates that a parameter should alias the value of another parameter.

    When used in conjunction with a custom distribution, this allows fuzzed
    tensors to represent a broader range of behaviors. For example, the
    following sometimes produces Tensors which broadcast:

    Fuzzer(
        parameters=[
            FuzzedParameter("x_len", 4, 1024, distribution="uniform"),

            # `y` will either be size one, or match the size of `x`.
            FuzzedParameter("y_len", distribution={
                0.5: 1,
                0.5: ParameterAlias("x_len")
            }),
        ],
        tensors=[
            FuzzedTensor("x", size=("x_len",)),
            FuzzedTensor("y", size=("y_len",)),
        ],
    )

    Chains of alias' are allowed, but may not contain cycles.
    """
    def __init__(self, alias_to):
        # 设置当前参数的别名目标
        self.alias_to = alias_to

    def __repr__(self):
        # 返回参数别名对象的字符串表示形式
        return f"ParameterAlias[alias_to: {self.alias_to}]"


def dtype_size(dtype):
    """Return the size of the data type in bytes.

    Determines the size of the given data type in bytes. This is crucial for
    memory allocation and operations involving tensors.

    Args:
        dtype: The torch data type.

    Returns:
        int: The size of the data type in bytes.
    """
    if dtype == torch.bool:
        return 1
    if dtype.is_floating_point or dtype.is_complex:
        return int(torch.finfo(dtype).bits / 8)
    return int(torch.iinfo(dtype).bits / 8)


def prod(values, base=1):
    """Compute the product of values.

    Computes the product of the values in the iterable `values`, starting with
    an initial `base` value. This is used to calculate the total size when
    creating tensors, ensuring it does not overflow and cause memory errors.

    Args:
        values: Iterable of values to multiply.
        base: Initial value for multiplication (default is 1).

    Returns:
        int: The computed product of values.
    """
    return functools.reduce(lambda x, y: int(x) * int(y), values, base)


class FuzzedTensor:
    """Represents a fuzzed tensor object with customizable attributes.

    Attributes:
        name (str): The name of the tensor.
    """
    def __init__(
        self,
        name: str,
        size: Tuple[Union[str, int], ...],
        steps: Optional[Tuple[Union[str, int], ...]] = None,
        probability_contiguous: float = 0.5,
        min_elements: Optional[int] = None,
        max_elements: Optional[int] = None,
        max_allocation_bytes: Optional[int] = None,
        dim_parameter: Optional[str] = None,
        roll_parameter: Optional[str] = None,
        dtype=torch.float32,
        cuda=False,
        tensor_constructor: Optional[Callable] = None
    ):
        # Initialize the attributes of the FuzzedTensor object
        self._name = name

    @property
    def name(self):
        # Getter method for the name attribute of the FuzzedTensor object
        return self._name

    @staticmethod
    def default_tensor_constructor(size, dtype, **kwargs):
        """Default constructor for generating a tensor of specified size and dtype.

        Generates a tensor using specified size and data type (dtype). Depending
        on the dtype, it either generates random values or random integers within
        a certain range.

        Args:
            size: Tuple specifying the size of the tensor.
            dtype: Data type of the tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Generated tensor.
        """
        if dtype.is_floating_point or dtype.is_complex:
            return torch.rand(size=size, dtype=dtype, device="cpu")
        else:
            return torch.randint(1, 127, size=size, dtype=dtype, device="cpu")
    # 根据参数 `params` 和 `state` 创建一个张量
    def _make_tensor(self, params, state):
        import numpy as np
        # 调用 `_get_size_and_steps` 方法获取张量的大小、步长和分配大小
        size, steps, allocation_size = self._get_size_and_steps(params)
        # 确定张量的构造函数，如果未指定则使用默认的构造函数
        constructor = (
            self._tensor_constructor or
            self.default_tensor_constructor
        )

        # 使用构造函数创建原始张量，指定大小、数据类型和额外的参数
        raw_tensor = constructor(size=allocation_size, dtype=self._dtype, **params)
        # 如果启用了 CUDA，将张量移动到 CUDA 设备上
        if self._cuda:
            raw_tensor = raw_tensor.cuda()

        # 随机排列张量，并调用 `.contiguous()` 强制重新排序内存，然后恢复原始形状
        dim = len(size)
        order = np.arange(dim)
        if state.rand() > self._probability_contiguous:
            while dim > 1 and np.all(order == np.arange(dim)):
                order = state.permutation(raw_tensor.dim())

            raw_tensor = raw_tensor.permute(tuple(order)).contiguous()
            raw_tensor = raw_tensor.permute(tuple(np.argsort(order)))

        # 根据给定的大小和步长创建切片，用于获取所需的张量
        slices = [slice(0, size * step, step) for size, step in zip(size, steps)]
        tensor = raw_tensor[slices]

        # 收集张量的属性信息，包括元素数量、排列顺序、步长、是否连续以及数据类型
        properties = {
            "numel": int(tensor.numel()),
            "order": order,
            "steps": steps,
            "is_contiguous": tensor.is_contiguous(),
            "dtype": str(self._dtype),
        }

        # 返回张量和其属性信息
        return tensor, properties

    # 根据参数 `params` 解析张量的大小和步长
    def _get_size_and_steps(self, params):
        dim = (
            params[self._dim_parameter]
            if self._dim_parameter is not None
            else len(self._size)
        )

        # 解析参数中的大小值，将其转换为具体的整数值
        def resolve(values, dim):
            values = tuple(params.get(i, i) for i in values)
            if len(values) > dim:
                values = values[:dim]
            if len(values) < dim:
                values = values + tuple(1 for _ in range(dim - len(values)))
            return values

        # 获取解析后的大小、步长和分配大小，并返回
        size = resolve(self._size, dim)
        steps = resolve(self._steps or (), dim)
        allocation_size = tuple(size_i * step_i for size_i, step_i in zip(size, steps))
        return size, steps, allocation_size

    # 检查是否满足给定的约束条件
    def satisfies_constraints(self, params):
        # 获取张量的大小和分配大小
        size, _, allocation_size = self._get_size_and_steps(params)
        # 计算张量的总元素数量，使用 Python 进行计算以避免整数溢出
        num_elements = prod(size)
        assert num_elements >= 0  # 断言张量的元素数量应该非负

        # 计算分配的总字节数，考虑数据类型的大小
        allocation_bytes = prod(allocation_size, base=dtype_size(self._dtype))

        # 比较函数，用于比较两个可空参数的大小关系
        def nullable_greater(left, right):
            if left is None or right is None:
                return False
            return left > right

        # 检查是否满足所有的约束条件，返回结果
        return not any((
            nullable_greater(num_elements, self._max_elements),
            nullable_greater(self._min_elements, num_elements),
            nullable_greater(allocation_bytes, self._max_allocation_bytes),
        ))
# 定义一个名为 Fuzzer 的类，用于生成模拟数据（fuzzed data）
class Fuzzer:
    # 初始化方法，接收一些参数来配置生成器
    def __init__(
        self,
        parameters: List[Union[FuzzedParameter, List[FuzzedParameter]]],
        tensors: List[Union[FuzzedTensor, List[FuzzedTensor]]],
        constraints: Optional[List[Callable]] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            parameters:
                用于生成参数的 FuzzedParameter 列表，支持可迭代对象但不支持任意嵌套结构。
            tensors:
                定义每一步生成的张量（Tensors）的 FuzzedTensor 列表，支持可迭代对象但不支持任意嵌套结构。
            constraints:
                可调用对象（函数）的列表。它们将被作为关键字参数调用，如果有任何一个返回 False，则当前参数集将被拒绝。
            seed:
                用于 Fuzzer 使用的 RandomState 的种子值。也将用于设置 PyTorch 随机种子，以便随机操作可以创建可重复的张量。
        """
        import numpy as np
        # 如果未提供种子值，则生成一个随机的 64 位整数种子
        if seed is None:
            seed = np.random.RandomState().randint(0, 2 ** 32 - 1, dtype=np.int64)
        self._seed = seed
        # 解包 parameters 列表中的 FuzzedParameter 对象，并保存到实例变量 _parameters 中
        self._parameters = Fuzzer._unpack(parameters, FuzzedParameter)
        # 解包 tensors 列表中的 FuzzedTensor 对象，并保存到实例变量 _tensors 中
        self._tensors = Fuzzer._unpack(tensors, FuzzedTensor)
        # 如果 constraints 为 None，则设置为空元组
        self._constraints = constraints or ()

        # 检查参数和张量的名称是否有重叠，如果有重叠则抛出 ValueError 异常
        p_names = {p.name for p in self._parameters}
        t_names = {t.name for t in self._tensors}
        name_overlap = p_names.intersection(t_names)
        if name_overlap:
            raise ValueError(f"Duplicate names in parameters and tensors: {name_overlap}")

        # 初始化拒绝计数和总生成计数
        self._rejections = 0
        self._total_generated = 0

    # 静态方法：将 values 中的对象解包成 cls 类型的元组
    @staticmethod
    def _unpack(values, cls):
        return tuple(it.chain(
            *[[i] if isinstance(i, cls) else i for i in values]
        ))

    # 生成器方法：生成指定数量的数据
    def take(self, n):
        import numpy as np
        state = np.random.RandomState(self._seed)
        # 设置 PyTorch 随机种子
        torch.manual_seed(state.randint(low=0, high=2 ** 63, dtype=np.int64))
        for _ in range(n):
            # 生成参数
            params = self._generate(state)
            tensors = {}
            tensor_properties = {}
            # 为每个 FuzzedTensor 对象生成张量和属性，并保存到相应的字典中
            for t in self._tensors:
                tensor, properties = t._make_tensor(params, state)
                tensors[t.name] = tensor
                tensor_properties[t.name] = properties
            # 返回生成的张量、属性以及参数
            yield tensors, tensor_properties, params

    # 属性方法：计算拒绝率
    @property
    def rejection_rate(self):
        # 如果总生成计数为 0，则返回拒绝率为 0
        if not self._total_generated:
            return 0.
        # 否则返回拒绝次数除以总生成次数的比率
        return self._rejections / self._total_generated
    # 生成符合指定状态的参数集合的私有方法
    def _generate(self, state):
        # strict_params 是一个空字典，用于存放严格参数的名称和值
        strict_params: Dict[str, Union[float, int, ParameterAlias]] = {}

        # 最多尝试生成1000次候选参数
        for _ in range(1000):
            # candidate_params 是一个空字典，用于存放候选参数的名称和值
            candidate_params: Dict[str, Union[float, int, ParameterAlias]] = {}

            # 遍历所有参数
            for p in self._parameters:
                # 如果参数是严格的
                if p.strict:
                    # 如果严格参数已经在 strict_params 中，则使用其值
                    if p.name in strict_params:
                        candidate_params[p.name] = strict_params[p.name]
                    # 否则从参数对象 p 中获取一个样本值并存入 strict_params
                    else:
                        candidate_params[p.name] = p.sample(state)
                        strict_params[p.name] = candidate_params[p.name]
                else:
                    # 如果参数不是严格的，则直接从参数对象 p 中获取一个样本值
                    candidate_params[p.name] = p.sample(state)

            # 解析候选参数中的别名，将其替换为实际的值
            candidate_params = self._resolve_aliases(candidate_params)

            # 增加生成的总数
            self._total_generated += 1

            # 检查候选参数是否满足所有约束条件
            if not all(f(candidate_params) for f in self._constraints):
                # 如果不满足约束条件，则增加拒绝的计数并继续下一次循环
                self._rejections += 1
                continue

            # 检查候选参数是否满足所有张量约束条件
            if not all(t.satisfies_constraints(candidate_params) for t in self._tensors):
                # 如果不满足张量约束条件，则增加拒绝的计数并继续下一次循环
                self._rejections += 1
                continue

            # 如果候选参数通过了所有约束条件和张量约束条件，则返回候选参数
            return candidate_params

        # 如果尝试了1000次仍未生成有效参数集合，则抛出异常
        raise ValueError("Failed to generate a set of valid parameters.")

    @staticmethod
    # 解析参数中的别名，将别名替换为实际的参数值的静态方法
    def _resolve_aliases(params):
        # 复制参数字典，避免直接修改原始参数
        params = dict(params)
        
        # 统计参数中别名的数量
        alias_count = sum(isinstance(v, ParameterAlias) for v in params.values())

        # 获取参数的键列表
        keys = list(params.keys())
        
        # 当存在别名时循环处理
        while alias_count:
            # 遍历参数键
            for k in keys:
                v = params[k]
                # 如果参数值是别名对象
                if isinstance(v, ParameterAlias):
                    # 将参数值替换为别名指向的实际值
                    params[k] = params[v.alias_to]
            
            # 计算新的别名数量
            alias_count_new = sum(isinstance(v, ParameterAlias) for v in params.values())
            
            # 如果循环前后别名数量没有变化，则可能存在参数别名循环，抛出异常
            if alias_count == alias_count_new:
                raise ValueError(f"ParameterAlias cycle detected\n{params}")

            # 更新别名数量
            alias_count = alias_count_new

        # 返回解析后的参数字典
        return params
```