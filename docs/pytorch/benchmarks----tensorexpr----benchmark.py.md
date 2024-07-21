# `.\pytorch\benchmarks\tensorexpr\benchmark.py`

```
# 导入必要的模块和库
import contextlib  # 上下文管理模块，提供了一些用于管理上下文资源的工具
import json  # JSON 数据的编解码模块
import os  # 提供了访问操作系统功能的接口
import time  # 提供了时间相关的功能

import numpy as np  # 数值计算库，用于处理多维数组和矩阵运算

import torch  # PyTorch 深度学习库

from . import tensor_engine  # 导入自定义的张量引擎模块


class Benchmark:
    def __init__(self, mode, device, dtype):
        # 初始化基准测试类的属性
        self.mode = mode  # 模式，可以是 "both" 或 "fwd"
        self.deterministic = False  # 是否确定性操作的标志
        self.device = device  # 设备类型，如 GPU 或 CPU
        self.dtype = dtype  # 数据类型，例如 torch.float32
        self.output_type = "stdout"  # 输出类型，默认为标准输出
        self.print_ir = False  # 是否打印中间表示(IR)的标志
        self.print_kernel = False  # 是否打印内核代码的标志

        # 根据模式设置是否需要梯度
        if mode == "both":
            self.requires_grad = True
        elif mode == "fwd":
            self.requires_grad = False
        else:
            raise ValueError(f"invalid mode: {mode}")  # 抛出异常，模式不合法

        self.result_grad = None  # 梯度结果，默认为空
        self.grad_variables = []  # 梯度变量列表，默认为空列表

        # 获取张量引擎的实例并重置为指定设备
        self.engine = tensor_engine.get_engine()
        self.engine.reset(device)

        # 将张量引擎的所有成员函数转发到当前 Benchmark 实例
        for method in dir(self.engine):
            if not callable(getattr(self.engine, method)):
                continue
            if hasattr(self, method):
                continue
            if method.startswith("_"):
                continue
            method_engine = getattr(self.engine, method)
            setattr(self, method, method_engine)

    def forward(self):
        """执行一步计算"""
        raise ValueError("this method should be reimplemented by subclass")  # 抛出异常，子类应重新实现该方法

    def check(self):
        # 如果不是确定性操作，直接返回
        if not self.deterministic:
            return
        # 断言当前计算结果与参考结果的所有元素在给定容差下相等
        np.testing.assert_allclose(
            self.reference(), self.numpy(self.compute()), atol=1e-2
        )

    def config(self):
        """返回当前基准测试的配置数组"""
        raise ValueError("this method should be reimplemented by subclass")  # 抛出异常，子类应重新实现该方法

    def desc(self):
        """返回当前基准测试的描述信息"""
        config = self.config()  # 获取配置数组
        config_str = "_".join([str(x) for x in config])  # 将配置数组转换为字符串
        device = self.device  # 获取设备类型
        if "NNC_NUM_THREADS" in os.environ:
            num_threads_str = os.environ["NNC_NUM_THREADS"]  # 获取环境变量中的线程数
            device += num_threads_str  # 在设备类型后加上线程数字符串
        return f"{self.engine.mode}: {self.module()}_{self.mode}_{device}_{config_str}"  # 返回描述信息字符串

    @staticmethod
    def module():
        raise ValueError("this method should be reimplemented by subclass")  # 抛出异常，子类应重新实现该方法

    def memory_workload(self):
        raise ValueError("this method should be reimplemented by subclass")  # 抛出异常，子类应重新实现该方法

    def compute_workload(self):
        """返回完成张量操作所需的标量运算次数"""
        return None  # 默认返回空，子类应重新实现该方法

    @staticmethod
    def input_iterable():
        """如果基准测试子类使用了输入可迭代参数，则返回 True"""
        return False  # 默认不使用输入可迭代参数

    def dtype_to_bytes(self):
        # 返回指定数据类型的每个元素所占字节数
        return torch.tensor(0, dtype=self.dtype).element_size()

    @staticmethod
    # 返回一个此基准测试的默认配置列表的方法
    def default_configs():
        """return a list of defualt configs for this benchmark"""
        # 如果没有在子类中重新实现此方法，则引发值错误异常
        raise ValueError("this method should be reimplemented by subclass")

    # 检查此基准测试是否受支持的方法
    def is_supported(self):
        # 总是返回True，表明此基准测试始终受支持
        return True

    # 生成具有指定形状的随机数张量的方法
    def rand(self, shape, device=None, dtype=None, requires_grad=False):
        # 调用引擎对象的rand方法生成随机数张量v
        v = self.engine.rand(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )
        # 如果requires_grad为True，将生成的张量v添加到梯度变量列表中
        if requires_grad:
            self.grad_variables.append(v)
        # 返回生成的随机数张量v
        return v

    # 生成具有指定形状（NCHW格式）的随机数张量的方法
    def nchw_rand(self, shape, device=None, requires_grad=False):
        # 调用引擎对象的nchw_rand方法生成NCHW格式的随机数张量v
        v = self.engine.nchw_rand(shape, device=device, requires_grad=requires_grad)
        # 如果requires_grad为True，将生成的张量v添加到梯度变量列表中
        if requires_grad:
            self.grad_variables.append(v)
        # 返回生成的NCHW格式的随机数张量v
        return v

    # 执行基准测试计算的方法
    def compute(self):
        # 如果启用了即时编译（bm_jit），则调用bm_jit方法并传递输入参数self.inputs
        if self.bm_jit:
            return self.bm_jit(*self.inputs)
        else:
            # 否则调用forward方法并传递输入参数self.inputs
            return self.forward(*self.inputs)

    # 运行基准测试的方法，接受参数args
    def run(self, args):
        # 设置是否打印IR的标志
        self.print_ir = args.print_ir
        # 根据args中的cuda_fuser参数进行不同的处理逻辑
        if args.cuda_fuser == "old":
            # 设置PyTorch可以在GPU上进行融合操作
            torch._C._jit_override_can_fuse_on_gpu(True)
            # 如果设置了打印kernel的标志，设置环境变量以启用PyTorch融合调试
            if args.print_kernel:
                os.environ["PYTORCH_FUSION_DEBUG"] = "1"
            # 调用run_impl方法并传递True，表明使用旧的CUDA融合器
            return self.run_impl(True)
        elif args.cuda_fuser == "te":
            # 启用新的张量表达式融合器
            torch._C._jit_set_texpr_fuser_enabled(True)
            # 使用指定的CUDA点对点上下文参数设置
            with cuda_pointwise_context(
                args.cuda_pointwise_loop_levels,
                args.cuda_pointwise_block_count,
                args.cuda_pointwise_block_size,
            ):
                # 调用run_impl方法并传递True，表明使用新的张量表达式融合器
                return self.run_impl(True)
        elif args.cuda_fuser == "nvf":
            # 启用NVIDIA的Fusion融合器
            torch._C._jit_set_nvfuser_enabled(True)
            # 启用分析执行器模式和分析模式
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            # 设置CPU和GPU上的融合限制
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(False)
            # 设置融合失败的深度为20
            torch._C._jit_set_bailout_depth(20)
            # 如果设置了打印kernel的标志，设置环境变量以启用CUDA融合调试
            if args.print_kernel:
                os.environ["PYTORCH_CUDA_FUSER_DEBUG"] = "1"
            # 调用run_impl方法并传递True，表明使用NVIDIA的Fusion融合器
            return self.run_impl(True)
        else:
            # 如果没有匹配的args.cuda_fuser参数值，则调用run_impl方法并传递False
            return self.run_impl(False)
    # 实现运行基准测试的方法，接受一个布尔参数 use_fuser
    def run_impl(self, use_fuser):
        # 预热迭代次数
        warmups = 10
        # 如果设备是 CUDA，设置迭代次数为 1000；否则设置为 10
        if self.device == "cuda":
            iters = 1000
        else:
            iters = 10
        # 获取张量计算引擎的实例
        engine = tensor_engine.get_engine()

        # 初始化用于 JIT 编译的对象
        self.bm_jit = None
        # 执行预热和正式迭代
        for i in range(warmups + iters):
            # 当达到预热迭代次数时
            if i == warmups:
                # 如果设备是 CUDA，同步 CUDA 设备
                if self.device == "cuda":
                    engine.sync_cuda()
                # 记录开始时间
                time_start = time.time()

            # 对于第一次迭代
            if i == 0:
                # 如果 JIT 模式是 "trace" 并且使用了 fuser，使用 torch.jit.trace 进行跟踪编译
                if self.jit_mode == "trace" and use_fuser:
                    self.bm_jit = torch.jit.trace(
                        self.forward, example_inputs=self.inputs, check_trace=False
                    )
                # 如果对象有 "reference" 方法，执行检查
                if callable(getattr(self, "reference", None)):
                    self.check()
                else:
                    # 否则打印警告信息
                    print("Warning: no reference result for ", self.module())
            # 对于第二次迭代
            elif i == 1:
                # 如果 JIT 模式是 "trace"，使用 fuser 并且设置了打印 IR 的选项，打印融合图
                if self.jit_mode == "trace" and use_fuser and self.print_ir:
                    print(self.bm_jit.graph_for(*self.inputs))
            # 计算模型的输出
            z = self.compute()
            # 如果模式是 "both"
            if self.mode == "both":
                # 如果结果梯度为空，生成与 z 同样大小的随机梯度
                if self.result_grad is None:
                    self.result_grad = engine.rand_like(z)
                # 执行反向传播
                engine.backward([z], [self.result_grad], self.grad_variables)

        # 如果设备是 CUDA，再次同步 CUDA 设备
        if self.device == "cuda":
            engine.sync_cuda()

        # 计算总运行时间
        duration = time.time() - time_start
        # 计算每次迭代的平均时间
        iter_time = duration / iters
        # 计算内存工作负载
        memory_workload = self.memory_workload()
        # 计算计算工作负载
        compute_workload = self.compute_workload()

        # 构建结果字典
        result_dict = {
            "desc": self.desc(),  # 获取描述信息
            "us": iter_time * 1e6,  # 将迭代时间转换为微秒
            "sol": memory_workload["sol"] * self.dtype_to_bytes() / iter_time / 1e9,  # 计算 SOL 吞吐量
            "algorithmic": memory_workload["algorithmic"]
            * self.dtype_to_bytes()
            / iter_time
            / 1e9,  # 计算算法效率
        }
        # 如果有计算工作负载，添加到结果字典中
        if compute_workload:
            result_dict["compute_workload"] = compute_workload / iter_time / 1e9
        # 输出结果字典
        self.dump_result(result_dict)

    # 将结果字典输出为 JSON 或者标准输出格式
    def dump_result(self, result_dict):
        if self.output_type == "json":
            # 输出 JSON 格式的结果字典
            print(json.dumps(result_dict))
        elif self.output_type == "stdout":
            # 构建标准输出格式的消息
            msg = "{}: {:.2f} us, SOL {:.2f} GB/s, algorithmic {:.2f} GB/s".format(
                result_dict["desc"],
                result_dict["us"],
                result_dict["sol"],
                result_dict["algorithmic"],
            )
            # 如果结果字典中有 compute_workload，添加到消息中
            if "compute_workload" in result_dict:
                msg += f", compute {result_dict['compute_workload']:.2f} Gops/s"
            # 输出消息
            print(msg)
        else:
            # 如果输出类型未知，抛出异常
            raise Exception("Unknown output_type " + self.output_type)  # noqa: TRY002
# 定义一个上下文管理器，用于设置 CUDA pointwise 的执行参数
@contextlib.contextmanager
def cuda_pointwise_context(loop_levels, block_count, block_size):
    # 如果有传入循环级别参数，设置 CUDA pointwise 循环级别，并记录旧值
    if loop_levels:
        old_loop_levels = torch._C._jit_get_te_cuda_pointwise_loop_levels()
        torch._C._jit_set_te_cuda_pointwise_loop_levels(loop_levels)
    # 如果有传入块数量参数，设置 CUDA pointwise 块数量，并记录旧值
    if block_count:
        old_block_count = torch._C._jit_get_te_cuda_pointwise_block_count()
        torch._C._jit_set_te_cuda_pointwise_block_count(block_count)
    # 如果有传入块大小参数，设置 CUDA pointwise 块大小，并记录旧值
    if block_size:
        old_block_size = torch._C._jit_get_te_cuda_pointwise_block_size()
        torch._C._jit_set_te_cuda_pointwise_block_size(block_size)

    try:
        yield
    finally:
        # 在上下文管理器结束时，如果设置过循环级别参数，恢复旧的循环级别设置
        if loop_levels:
            torch._C._jit_set_te_cuda_pointwise_loop_levels(old_loop_levels)
        # 在上下文管理器结束时，如果设置过块数量参数，恢复旧的块数量设置
        if block_count:
            torch._C._jit_set_te_cuda_pointwise_block_count(old_block_count)
        # 在上下文管理器结束时，如果设置过块大小参数，恢复旧的块大小设置
        if block_size:
            torch._C._jit_set_te_cuda_pointwise_block_size(old_block_size)


# 辅助类，用于动态输入形状的基准测试
class DynamicShape:
    r"""
    An Auxiliary class for dynamic shape benchmarks

    Pre-computes input with random shapes and also
    modifies the compute method so in each call the
    fuser sees a different input tensor shape
    """

    # 在一个实例中的随机输入数量
    SAMPLE_SIZE = 100

    def __init__(self, dynamic_range=1.2):
        # 存储随机输入样本的列表
        self._input_samples = []
        # 当前输入样本索引
        self._input_sample_index = 0
        # 动态范围，影响随机形状的生成
        self._dynamic_range = (
            1.0 / dynamic_range if dynamic_range > 1.0 else dynamic_range
        )
        # 是否启用动态形状
        self._enable_dynamic_shapes = True

    # 获取当前索引指向的输入测试用例
    @property
    def inputs(self):
        return self._input_samples[self._input_sample_index]

    # 设置输入测试用例，实际上将其添加到类缓冲区中的测试样例
    @inputs.setter
    def inputs(self, val):
        self._input_samples.append(val)

    # 运行普通的计算方法，同时增加测试用例索引
    def compute(self):
        super().compute()
        self._input_sample_index = (self._input_sample_index + 1) % self.SAMPLE_SIZE

    # 由基准测试定义，基准测试需要在此方法中指定输入张量的构建方式
    def instantiate_input(self):
        raise NotImplementedError

    # 实例化随机形状的输入，并启动基准测试运行
    def run(self, args):
        # 如果命令行禁用了动态形状，强制禁用动态形状
        if args.no_dynamic_shape:
            self._enable_dynamic_shapes = False
        # 加载输入数据
        self.load_inputs()
        super().run(args)

    # 预先计算输入，以便随机张量的创建不会增加计算时间
    def load_inputs(self):
        for i in range(self.SAMPLE_SIZE - 1):
            self.instantiate_input()

    # 返回一个随机形状
    # 定义一个方法，用于生成随机形状的数组
    def rand_shape(self, shape):
        # 如果禁用了动态形状功能，直接返回原始形状
        if not self._enable_dynamic_shapes:
            return shape
        # 生成一个长度与输入形状相同的随机比例数组，每个比例在[_dynamic_range, 1.0)范围内均匀分布
        ratios = np.random.uniform(self._dynamic_range, 1.0, len(shape))
        # 将输入形状与随机比例相乘并转换为整数，生成动态形状数组
        dyn_shape = list(np.multiply(shape, ratios).astype(int))
        # 返回生成的动态形状数组
        return dyn_shape
# 定义一个空列表，用于存储注册的基准测试类
benchmark_classes = []

# 定义一个函数，用于注册基准测试类到 benchmark_classes 列表中
def register_benchmark_class(benchmark_cls):
    # 将传入的 benchmark_cls 参数添加到 benchmark_classes 列表中
    benchmark_classes.append(benchmark_cls)
```