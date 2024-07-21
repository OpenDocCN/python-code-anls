# `.\pytorch\benchmarks\tensorexpr\reduction.py`

```py
frompython
# 从 benchmark 模块中导入 benchmark 类
from . import benchmark

# 定义 ReduceBench 类，继承自 Benchmark 类
class ReduceBench(benchmark.Benchmark):
    # 初始化方法，接受模式、设备、数据类型、案例、M、N、K 和 skip_input_transform 参数
    def __init__(self, mode, device, dtype, case, M, N, K, skip_input_transform):
        # 调用父类 Benchmark 的初始化方法
        super().__init__(mode, device, dtype)
        # 设置案例、M、N、K 和 skip_input_transform 属性
        self.case = case
        self.M = M
        self.N = N
        self.K = K
        self._set_skip_input_transform(skip_input_transform)

        # 初始化 inputs 属性为一个包含随机数的列表
        self.inputs = [
            self.randn(
                [M, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        ]
        
        # 根据 case 的不同设置 dims 属性
        if case == "row":
            self.dims = [1, 2]
        elif case == "mid":
            self.dims = [0, 2]
        elif case == "col":
            self.dims = [0, 1]
        elif case == "full":
            self.dims = [0, 1, 2]
        else:
            raise ValueError(f"invalid case: {case}")

    # 前向传播方法，接受 inputs 参数
    def forward(self, inputs):
        # 根据 skip_input_transform 属性决定是否跳过输入转换
        if self.skip_input_transform:
            x = inputs
        else:
            x = self.add(inputs, 0.001)  # 对 inputs 添加一个小值
        y = self.sum(x, self.dims)  # 对 x 在指定维度上求和
        return y

    # 返回配置信息的方法
    def config(self):
        if self.case == "full":
            return [self.M * self.N * self.K, self._skip_input_transform_str()]
        return [self.M, self.N, self.K, self._skip_input_transform_str()]

    # 返回默认配置列表的静态方法
    @staticmethod
    def default_configs():
        return [
            # [512, 512, 512],
            [512, 64, 512, "s0"],
        ]

    # 返回模块名称的静态方法
    @staticmethod
    def module():
        return "reduce"

    # 计算内存工作负载的方法
    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = (1) + (1)
            algorithmic_count = 1 + 1

        buffer_size = self.M * self.N * self.K
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    # 设置 skip_input_transform 属性的方法
    def _set_skip_input_transform(self, input_str):
        # 在测试设置中，s1 表示跳过输入转换，s0 表示不跳过
        if input_str == "s0":
            self.skip_input_transform = False
        elif input_str == "s1":
            self.skip_input_transform = True
        else:
            raise ValueError(f"invalid skip_input_transform: {input_str}")

    # 返回 skip_input_transform 属性字符串表示的方法
    def _skip_input_transform_str(self):
        if self.skip_input_transform:
            return "s1"
        else:
            return "s0"


# 定义 ReduceRowBench 类，继承自 ReduceBench 类，案例固定为 "row"
class ReduceRowBench(ReduceBench):
    def __init__(self, mode, device, dtype, M, N, K, skip_input_transform):
        super().__init__(mode, device, dtype, "row", M, N, K, skip_input_transform)

    # 返回模块名称的静态方法
    @staticmethod
    def module():
        return "reduce_row"


# 定义 ReduceMidBench 类，继承自 ReduceBench 类，案例固定为 "mid"
class ReduceMidBench(ReduceBench):
    def __init__(self, mode, device, dtype, M, N, K, skip_input_transform):
        super().__init__(mode, device, dtype, "mid", M, N, K, skip_input_transform)

    # 返回模块名称的静态方法
    @staticmethod
    def module():
        return "reduce_mid"


# 定义 ReduceColBench 类，继承自 ReduceBench 类，案例固定为 "col"
class ReduceColBench(ReduceBench):
    def __init__(self, mode, device, dtype, M, N, K, skip_input_transform):
        super().__init__(mode, device, dtype, "col", M, N, K, skip_input_transform)

    # 返回模块名称的静态方法
    @staticmethod
    def module():
        return "reduce_col"
    # 初始化函数，继承父类的初始化方法，设置模式、设备、数据类型、矩阵维度和是否跳过输入转换
    def __init__(self, mode, device, dtype, M, N, K, skip_input_transform):
        super().__init__(mode, device, dtype, "col", M, N, K, skip_input_transform)

    # 静态方法，返回模块类型为"reduce_col"
    @staticmethod
    def module():
        return "reduce_col"
class ReduceFullBench(ReduceBench):
    # ReduceFullBench 类继承自 ReduceBench 类，用于执行完整的规约基准测试
    def __init__(self, mode, device, dtype, M, skip_input_transform):
        # 调用父类 ReduceBench 的构造函数初始化对象
        super().__init__(mode, device, dtype, "full", M, 1, 1, skip_input_transform)

    def config(self):
        # 返回一个配置列表，包括 M * N * K 的计算量和 _skip_input_transform_str() 的结果
        return [self.M * self.N * self.K, self._skip_input_transform_str()]

    @staticmethod
    def default_configs():
        # 静态方法，返回默认配置列表，包括 [1 << 24, "s1"]
        return [
            [1 << 24, "s1"],
        ]

    @staticmethod
    def module():
        # 静态方法，返回字符串 "reduce_full"，表示模块名称
        return "reduce_full"


class Reduce2DBench(benchmark.Benchmark):
    """
    A benchmark class to validate 2 dimensional reduction performance.
    Only a simple add is fused to induce the fuser and isolate reduction perf.
    """

    def __init__(self, mode, device, dtype, red_dim, dim0, dim1):
        # 构造函数，初始化对象属性，包括规约维度 red_dim 和数组维度 dim0, dim1
        super().__init__(mode, device, dtype)
        self.red_dim = red_dim
        self.dim0 = dim0
        self.dim1 = dim1

        self.inputs = [
            self.randn(
                [dim0, dim1],
                device=device,
                dtype=dtype,
                requires_grad=self.requires_grad,
            )
        ]

        if red_dim != 0 and red_dim != 1:
            # 如果规约维度不为 0 或 1，则抛出异常
            raise ValueError(f"invalid reduction dimension: {red_dim}")

    def forward(self, inputs):
        # 前向传播方法，对输入进行加法操作并返回规约后的结果
        x = self.add(inputs, 0.001)
        y = self.sum(x, [self.red_dim])
        return y

    def config(self):
        # 返回一个配置列表，包括 red_dim, dim0, dim1
        return [self.red_dim, self.dim0, self.dim1]

    @staticmethod
    def default_configs():
        # 静态方法，返回默认配置列表，包括 [1, 640, 524288]
        return [
            [1, 640, 524288],
        ]

    @staticmethod
    def module():
        # 静态方法，返回字符串 "reduce2d"，表示模块名称
        return "reduce2d"

    @staticmethod
    def input_iterable():
        # 静态方法，返回 True，表示输入可以迭代
        return True

    def memory_workload(self):
        # 返回内存工作负载字典，包括 "sol" 和 "algorithmic" 两个键
        assert self.mode == "fwd", "Only the forward operation is modeled!"

        buffer_size = self.dim0 * self.dim1
        if self.red_dim == 0:
            buffer_size += self.dim1
        else:
            buffer_size += self.dim0
        return {
            "sol": buffer_size,
            "algorithmic": buffer_size,
        }


class Reduce2DInnerBench(Reduce2DBench):
    def __init__(self, mode, device, dtype, dim0, dim1):
        # 构造函数，调用父类 Reduce2DBench 的构造函数，设置规约维度为 1
        super().__init__(mode, device, dtype, 1, dim0, dim1)

    @staticmethod
    def default_configs():
        # 静态方法，获取父类 Reduce2DBench 的默认配置，并去除第一个元素
        parent_config = Reduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        # 返回父类配置的子列表
        parent_config = super().config()
        return parent_config[1:]

    @staticmethod
    def module():
        # 静态方法，返回字符串 "reduce2d_inner"，表示模块名称
        return "reduce2d_inner"


class Reduce2DOuterBench(Reduce2DBench):
    def __init__(self, mode, device, dtype, dim0, dim1):
        # 构造函数，调用父类 Reduce2DBench 的构造函数，设置规约维度为 0
        super().__init__(mode, device, dtype, 0, dim0, dim1)

    @staticmethod
    def default_configs():
        # 静态方法，获取父类 Reduce2DBench 的默认配置，并去除第一个元素
        parent_config = Reduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        # 返回父类配置的子列表
        parent_config = super().config()
        return parent_config[1:]

    @staticmethod
    def module():
        # 静态方法，返回字符串 "reduce2d_outer"，表示模块名称
        return "reduce2d_outer"


benchmark.register_benchmark_class(ReduceRowBench)
# 注册 ReduceMidBench 类到基准测试框架
benchmark.register_benchmark_class(ReduceMidBench)
# 注册 ReduceColBench 类到基准测试框架
benchmark.register_benchmark_class(ReduceColBench)
# 注册 Reduce2DInnerBench 类到基准测试框架
benchmark.register_benchmark_class(Reduce2DInnerBench)
# 注册 Reduce2DOuterBench 类到基准测试框架
benchmark.register_benchmark_class(Reduce2DOuterBench)
# 注册 ReduceFullBench 类到基准测试框架
benchmark.register_benchmark_class(ReduceFullBench)


class DynamicReduce2DBench(benchmark.DynamicShape, Reduce2DBench):
    """
    一个用于验证二维减少性能的基准测试类。
    只有一个简单的加法被融合以诱导融合器并隔离减少性能。
    """

    def __init__(self, mode, device, dtype, red_dim, dim0, dim1):
        # 调用 DynamicShape 类的初始化方法
        benchmark.DynamicShape.__init__(self)
        # 调用 Reduce2DBench 类的初始化方法
        Reduce2DBench.__init__(self, mode, device, dtype, red_dim, dim0, dim1)

    def instantiate_input(self):
        # 生成随机形状的 dim0 和 dim1
        dim0, dim1 = self.rand_shape([self.dim0, self.dim1])

        self.inputs = [
            # 生成随机张量，形状为 [dim0, dim1]
            self.randn(
                [dim0, dim1],
                device=self.device,
                dtype=self.dtype,
                requires_grad=self.requires_grad,
            )
        ]

    @staticmethod
    def module():
        return "dynamicreduce2d"


class DynamicReduce2DInnerBench(DynamicReduce2DBench):
    def __init__(self, mode, device, dtype, dim0, dim1):
        # 调用父类 DynamicReduce2DBench 的初始化方法
        super().__init__(mode, device, dtype, 1, dim0, dim1)

    @staticmethod
    def default_configs():
        # 获取父类 DynamicReduce2DBench 的默认配置并返回第一个配置的子集
        parent_config = DynamicReduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        # 获取父类的配置并返回第一个配置的子集
        parent_config = super().config()
        return parent_config[1:]

    @staticmethod
    def module():
        return "reduce2d_dynamic_inner"


class DynamicReduce2DOuterBench(DynamicReduce2DBench):
    def __init__(self, mode, device, dtype, dim0, dim1):
        # 调用父类 DynamicReduce2DBench 的初始化方法
        super().__init__(mode, device, dtype, 0, dim0, dim1)

    @staticmethod
    def default_configs():
        # 获取父类 DynamicReduce2DBench 的默认配置并返回第一个配置的子集
        parent_config = DynamicReduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        # 获取父类的配置并返回第一个配置的子集
        parent_config = super().config()
        return parent_config[1:]

    @staticmethod
    def module():
        return "reduce2d_dynamic_outer"


# 注册 DynamicReduce2DInnerBench 类到基准测试框架
benchmark.register_benchmark_class(DynamicReduce2DInnerBench)
# 注册 DynamicReduce2DOuterBench 类到基准测试框架
benchmark.register_benchmark_class(DynamicReduce2DOuterBench)
```