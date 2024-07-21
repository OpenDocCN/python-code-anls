# `.\pytorch\benchmarks\operator_benchmark\benchmark_core.py`

```py
# 导入必要的模块和库
import ast  # 用于处理抽象语法树的模块
import copy  # 用于复制对象的模块
import functools  # 用于创建高阶函数的模块
import json  # 用于处理 JSON 格式的模块
import timeit  # 用于测量小段代码执行时间的模块
from collections import namedtuple  # 从 collections 模块导入 namedtuple 类

import benchmark_utils  # 导入性能基准测试工具
import numpy as np  # 导入 NumPy 数学库

import torch  # 导入 PyTorch 深度学习库

# 需要在导入 torch 后导入
import torch.utils.cpp_extension as cpp_extension  # 导入用于加载 C++ 扩展的模块（不会在代码中使用，但需要保留）

"""性能微基准测试。

这个模块包含性能微基准测试的核心功能。
"""

"""
用于存储测试配置的命名元组。
示例输入如下：
TestConfig(test_name='add_M8_N2_K1', input_config='M: 8, N: 2, K: 1',
    tag='long', run_backward=False)
"""
TestConfig = namedtuple("TestConfig", "test_name input_config tag run_backward")


BENCHMARK_TESTER = []  # 用于存储测试信息的列表


def _register_test(*test_metainfo):
    """保存创建测试所需的元信息。目前 test_metainfo 可以接受两种不同的输入：
    1) 当添加单个操作到基准测试时的输入格式
    _register_test(configs, pt_bench_op, create_pytorch_op_test_case,
                      run_backward=True)
    2) 当添加一组操作到基准测试时的输入格式
    _register_test(configs, pt_bench_op, create_pytorch_op_test_case,
                      run_backward=False,
                      op_name_function=op)
    """
    BENCHMARK_TESTER.append(test_metainfo)


def _create_test(
    bench_op_obj, orig_test_attrs, tags, OperatorTestCase, run_backward, bwd_input
):
    """使用基准测试后端创建测试。
    参数:
        bench_op_obj: 一个由 TorchBenchmarkBase 的子类实例化的对象，包括张量创建和操作执行。
        orig_test_attrs: 包含测试配置的字典。
        tags: 用于筛选输入的测试配置中的一个属性。
        OperatorTestCase: 用于保存测试元数据的命名元组。
        run_backward: 一个布尔参数，指示是否进行反向路径测试。
        bwd_input: 后向路径输入的字符串。
    """
    test_attrs = copy.deepcopy(orig_test_attrs)
    test_attrs = {k: str(v) for k, v in test_attrs.items()}
    ascii_test_attrs = ast.literal_eval(json.dumps(test_attrs))
    input_config = str(ascii_test_attrs)[1:-1].replace("'", "")
    if bwd_input:
        # 当使用 auto_set 时，测试名称需要包括输入。
        test_attrs.update({"bwd": bwd_input})
    test_name = bench_op_obj.test_name(**test_attrs)
    test_config = TestConfig(test_name, input_config, tags, run_backward)
    return OperatorTestCase(bench_op_obj, test_config)


def _build_test(
    configs, bench_op, OperatorTestCase, run_backward, op_name_function=None
):
    """生成具有不同输入的 PyTorch/Caffe2 操作符测试。
    参数:
        configs: 包含输入形状的字典。
        bench_op: TorchBenchmarkBase 的子类，包括张量创建和操作执行。
        OperatorTestCase: 用于保存测试元数据的命名元组。
        run_backward: 一个布尔参数，指示是否进行反向路径测试。
        op_name_function: 包含操作符名称和函数的字典。
    """
    """
    BenchmarkRunner is responsible for benchmarking all the registered
    benchmark test groups.

    Attributes:
        tag_filter (str): control the benchmarks which matches the tag.
        operator (str): only run benchmark test cases that contains
        this filter string in the test case's id.
        test_name (str): only run benchmark test cases that matches this filter,
        this is a case-sensitive substring match and it happens in
        the _keep_test method.
    """
    # BenchmarkRunner 类负责对所有注册的基准测试组进行基准测试

    def __init__(self, args):
        # 初始化方法，接受参数 args
        # TODO: 考虑时间约束条件
        self.args = args
        self.iters = 100  # 迭代次数默认为 100
        self.has_explicit_iteration_count = False  # 是否有显式的迭代次数设定，默认为 False
        self.multiplier = 2  # 迭代次数的倍增因子，默认为 2
        self.predefined_minimum_secs = 1  # 预定义的最小运行时间，默认为 1 秒
        self.max_iters = 1e6  # 最大迭代次数，默认为 1000000
        self.use_jit = args.use_jit  # 是否使用 JIT，由参数 args 决定
        self.num_runs = args.num_runs  # 运行次数，由参数 args 决定
        self.print_per_iter = False  # 是否每次迭代打印信息，默认为 False
        self.operator_range = benchmark_utils.get_operator_range(args.operator_range)
        # 根据参数 args 中的 operator_range 获取操作范围

        # 如果 args 中的 warmup_iterations 为 -1，则设为默认的 100
        if self.args.warmup_iterations == -1:
            self.args.warmup_iterations = 100

        # 如果 args 中的 iterations 存在且不为 -1，则表示有显式的迭代次数设定
        if self.args.iterations and self.args.iterations != -1:
            self.has_explicit_iteration_count = True
            self.iters = self.args.iterations

        # 当用户选择了特定的测试用例时，不再需要匹配标签
        if self.args.test_name is not None:
            self.args.tag_filter = None
    ```
    # _print_header 方法用于打印基准测试报告的标题信息
    def _print_header(self):
        DASH_LINE = "-" * 40
        print(
            f"# {DASH_LINE}\n"
            "# PyTorch/Caffe2 Operator Micro-benchmarks\n"
            f"# {DASH_LINE}\n"
            f"# Tag : {self.args.tag_filter}\n"
        )
        # 如果 args 中的 list_tests 为 True，则打印测试用例列表
        if self.args.list_tests:
            print("# List of tests:")
        # 如果 args 中的 list_ops 为 True，则打印要运行的操作符列表
        elif self.args.list_ops:
            print("# List of Operators to run:")
            self.printed_ops_list = set()
            # 如果 args 中指定了 operators，则打印操作符列表
            if self.args.operators:
                print(f"# {self.args.operators}")
    # 打印性能结果函数，用于输出性能测量结果
    def _print_perf_result(self, reported_run_time_us, test_case):
        if self.args.report_aibench:
            # 如果设置了 report_aibench 参数，则输出适用于 AIBench 的数据格式
            # 不输出通常的性能结果
            return

            # 构造测试名称，用于输出每次迭代的执行时间
            test_name = "_".join([test_case.framework, test_case.test_config.test_name])
            for run in range(self.num_runs):
                print(
                    f"{test_case.framework}Observer "
                    + json.dumps(
                        {
                            "type": test_name,
                            "metric": "latency",
                            "unit": "us",
                            "value": str(reported_run_time_us[run]),
                        }
                    )
                )
        else:
            # 否则输出普通的性能结果
            print(f"# Mode: {'JIT' if self.use_jit else 'Eager'}")
            print(
                f"# Name: {test_case.test_config.test_name}\n# Input: {test_case.test_config.input_config}"
            )

            # 根据测试配置判断是反向还是正向模式
            mode = "Backward" if test_case.test_config.run_backward else "Forward"
            if self.num_runs > 1:
                # 如果运行多次，则输出每次运行的执行时间
                for run in range(self.num_runs):
                    print(
                        f"Run: {run}, {mode} Execution Time (us) : {reported_run_time_us[run]:.3f}"
                    )
                print()
            else:
                # 否则只输出单次运行的执行时间
                print(f"{mode} Execution Time (us) : {reported_run_time_us[0]:.3f}\n")

    # 预测所需的迭代次数，根据给定的乘数进行计算
    def _predict_num_iter_needed(self, i):
        return i * self.multiplier

    # 判断迭代结果是否显著，根据条件判断是否应该报告测量时间
    def _iteration_result_is_significant(
        self, iters, run_time_sec, curr_test_total_time, has_explicit_iteration_count
    ):
        """This function decides whether the measured time can be reported based on the
        following conditions: 1) the number of iterations is larger than the max_iters.
        2) the execution time is larger than the predefined minimum_time
        3) the execution time is larger than user defined minimum_time
        """
        return (
            iters > self.max_iters
            or run_time_sec > self.predefined_minimum_secs
            or has_explicit_iteration_count
        ) and curr_test_total_time > self.args.min_time_per_test

    # 执行正向传播，使用 Python 的 timeit 模块测量执行时间（单位：秒）
    def _launch_forward(self, test_case, iters, print_per_iter):
        cuda_sync = "cuda" in test_case.test_config.test_name
        func = test_case.run_forward
        if self.use_jit:
            func = test_case.run_jit_forward
        # 调用 timeit 模块测量执行时间
        forward_time = timeit.timeit(
            functools.partial(func, iters, print_per_iter, cuda_sync), number=1
        )
        return forward_time
    def _launch_backward(self, test_case, iters, print_per_iter=False):
        """
        运行操作的前向路径以获取输出，然后执行反向路径，并报告执行时间。
        """
        # 运行测试用例的前向路径一次
        test_case.run_forward(num_runs=1, print_per_iter=False, cuda_sync=False)
        # 计算输出的均值
        test_case._output_mean()
        # 使用 timeit 库计时运行反向路径，并记录时间
        backward_time = timeit.timeit(
            functools.partial(test_case.run_backward, iters, print_per_iter), number=1
        )
        return backward_time

    def _measure_time(self, launch_test, test_case, iters, print_per_iter):
        """
        执行操作 <iters> 次迭代，并测量时间。
        如果时间不够显著，将增加迭代次数后重新运行，直到时间显著为止。
        """
        curr_test_total_time = 0
        time_trace = []
        while True:
            # 执行测试，获取运行时间
            run_time_sec = launch_test(test_case, iters, print_per_iter)
            curr_test_total_time += run_time_sec
            # 分析每次运行后的时间，判断结果是否稳定
            results_are_significant = self._iteration_result_is_significant(
                iters,
                run_time_sec,
                curr_test_total_time,
                self.has_explicit_iteration_count,
            )

            # 计算每次迭代的平均运行时间，单位为毫秒
            report_run_time = 1e6 * run_time_sec / iters
            time_trace.append(report_run_time)
            # 如果需要报告 AIBench 的结果
            if self.args.report_aibench:
                mode = "JIT" if self.use_jit else "Eager"
                test_name = "_".join(
                    [test_case.framework, test_case.test_config.test_name, mode]
                )
                # 打印每轮迭代的时间，以毫秒为单位
                print(
                    "PyTorchObserver "
                    + json.dumps(
                        {
                            "type": test_name,
                            "metric": "latency",
                            "unit": "ms",
                            "value": str(report_run_time / 1e3),
                        }
                    )
                )
            # 如果结果已经足够稳定，则结束循环
            if results_are_significant:
                break

            # 重新估计可能需要的迭代次数，并再次运行基准测试
            iters = self._predict_num_iter_needed(iters)
        # 返回时间跟踪列表的中位数，作为报告的运行时间（单位微秒）
        reported_run_time_us = np.percentile(np.array(time_trace), 50)
        return reported_run_time_us

    def _check_keep(self, test_flag, cmd_flag):
        """
        检查是否保留测试标志，根据命令行标志决定。
        """
        return cmd_flag is None or test_flag == cmd_flag

    def _check_operator_first_char(self, test_flag, cmd_flag):
        """
        检查测试标志的第一个字符是否在命令行标志中。
        """
        if cmd_flag is None or test_flag[:1].lower() in cmd_flag:
            return True
        return False

    def _check_keep_list(self, test_flag, cmd_flag_list):
        """
        检查是否保留测试标志，根据命令行标志列表决定。
        """
        if cmd_flag_list is None or any(
            test_flag == cmd_flag for cmd_flag in cmd_flag_list
        ):
            return True
        return False
    # 定义一个方法 `_keep_test`，用于判断是否保留给定的测试用例
    def _keep_test(self, test_case):
        # TODO: consider regex matching for test filtering.
        # 目前，此处使用子字符串匹配，可以考虑使用正则表达式进行测试过滤。
        
        # 获取测试用例的配置信息
        op_test_config = test_case.test_config
        
        # 如果有指定运算符列表，则将其处理为参数列表；否则为 None
        operators = (
            benchmark_utils.process_arg_list(self.args.operators)
            if self.args.operators
            else None
        )
        
        # 根据一系列条件来过滤测试用例：
        # 1. 检查测试名称是否保留
        # 2. 检查运算符是否符合要求
        # 3. 检查运算符模块名的首字符是否符合范围要求
        # 4. 检查标签是否符合要求（如果 `tag_filter` 为 "all" 或与测试标签匹配）
        # 5. 检查是否需要仅保留反向运行的测试用例
        # 6. 检查设备是否匹配测试用例的配置要求
        if (
            self._check_keep(op_test_config.test_name, self.args.test_name)
            and self._check_keep_list(test_case.op_bench.module_name(), operators)
            and self._check_operator_first_char(
                test_case.op_bench.module_name(), self.operator_range
            )
            and (
                self.args.tag_filter == "all"
                or self._check_keep(op_test_config.tag, self.args.tag_filter)
            )
            and (
                not self.args.forward_only
                or op_test_config.run_backward != self.args.forward_only
            )
            and (
                self.args.device == "None"
                or "device" not in test_case.test_config.input_config
                or self.args.device in op_test_config.test_name
            )
        ):
            return True
        
        # 如果以上条件都不符合，则不保留此测试用例
        return False


    # 定义一个方法 `_print_test_case_info`，用于打印测试用例的信息或者跳过真实执行
    def _print_test_case_info(self, test_case):
        # 如果指定了 `list_tests` 参数，则打印测试用例的名称并返回 True
        if self.args.list_tests:
            print(f"# {test_case.test_config.test_name}")
            return True
        
        # 如果指定了 `list_ops` 参数，则打印运算符的名称（如果尚未打印过）
        elif self.args.list_ops:
            if self.args.operators is None:
                op_name = test_case.op_bench.module_name()

                # 如果运算符名称不在已打印列表中，则打印并将其添加到已打印集合中
                if op_name not in self.printed_ops_list:
                    print(f"# {op_name}")
                    self.printed_ops_list.add(op_name)
            return True
        
        # 如果以上条件都不符合，则返回 False，表示不打印信息也不跳过执行
        return False
    def run(self):
        # 执行函数开始，打印头部信息
        self._print_header()

        # 遍历每个测试元信息
        for test_metainfo in BENCHMARK_TESTER:
            # 使用元信息构建测试用例
            for test in _build_test(*test_metainfo):
                # 获取完整测试ID和测试用例对象
                full_test_id, test_case = test
                # 获取测试配置信息
                op_test_config = test_case.test_config

                # 打印测试用例信息，如果打印成功则继续下一个测试用例
                if self._print_test_case_info(test_case):
                    continue

                # 如果不保留该测试用例，则继续下一个测试用例
                if not self._keep_test(test_case):
                    continue

                # 为了减少变异，固定numpy的随机种子到测试用例，
                # 以便每个测试用例的随机生成输入张量保持一致。
                # 随机种子限制在32位以符合numpy的要求。
                np.random.seed(seed=hash(full_test_id) & ((1 << 32) - 1))

                # 打印性能基准信息
                print(
                    f"# Benchmarking {test_case.framework}: {test_case.op_bench.module_name()}"
                )

                # 根据测试配置选择启动函数
                if op_test_config.run_backward:
                    launch_func = self._launch_backward
                else:
                    launch_func = self._launch_forward

                # 预热阶段
                launch_func(
                    test_case, self.args.warmup_iterations, print_per_iter=False
                )
                # 实际执行阶段
                reported_time = [
                    self._measure_time(
                        launch_func, test_case, self.iters, self.print_per_iter
                    )
                    for _ in range(self.num_runs)
                ]

                # 打印性能结果
                self._print_perf_result(reported_time, test_case)
```