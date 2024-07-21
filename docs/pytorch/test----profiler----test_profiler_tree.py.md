# `.\pytorch\test\profiler\test_profiler_tree.py`

```
# Owner(s): ["oncall: profiler"]

# 导入标准库和第三方模块
import functools
import os
import re
import textwrap
import traceback
import unittest

# 导入自定义模块
import expecttest

# 导入PyTorch相关模块
import torch
from torch._C._profiler import _ExtraFields_PyCall, _ExtraFields_PyCCall
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
)
from torch.utils._pytree import tree_map

# 定义常量，用于剪枝函数调用路径
PRUNE_ALL = 1
KEEP_ELLIPSES = 2
KEEP_NAME_AND_ELLIPSES = 3

# 定义需要剪枝的函数及其策略
PRUNE_FUNCTIONS = {
    "torch/utils/_pytree.py(...): tree_map": KEEP_NAME_AND_ELLIPSES,
    "torch/profiler/profiler.py(...): start": KEEP_ELLIPSES,
    "torch/profiler/profiler.py(...): stop_trace": KEEP_ELLIPSES,
    "torch/profiler/profiler.py(...): _transit_action": KEEP_ELLIPSES,
    "<built-in method __exit__ of torch._C.DisableTorchFunctionSubclass object at 0xXXXXXXXXXXXX>": PRUNE_ALL,
    "cudaStreamIsCapturing": PRUNE_ALL,
    # 仅在CUDA时显示，为保持CUDA和CPU的期望结果一致而进行剪枝
    "cudaGetDeviceCount": PRUNE_ALL,
    "cudaGetDeviceProperties_v2": PRUNE_ALL,
}

# ROCTracer目前无法产生分析器可以提取的事件。我们应该使其与CUPTI Kineto / 分析器集成齐头并进，
# 但在此期间，仍然可以运行测试但不检查值是否与预期值匹配。
# 1）我们仍然可以捕获运行时错误和断言失败
# 2）我们可以对输出进行比对，看看与齐头并进有多远
#
# TODO: 在某些平台上，我们也无法捕获Windows上的事件。
ALLOW_CUDA_FAILURE = (torch.version.hip is not None) or IS_WINDOWS


class TorchFunctionTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)


class TorchDispatchTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        # 创建子类张量实例，并记录原始元素及是否需要梯度
        t = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
        t.elem = elem
        return t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 用于将张量封装为特定类型，并在处理前后进行解封
        def unwrap(x):
            return x.elem if isinstance(x, TorchDispatchTensor) else x

        def wrap(x):
            return TorchDispatchTensor(x) if isinstance(x, torch.Tensor) else x

        # 对参数和关键字参数应用unwrap和wrap函数
        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs or {})

        # 应用函数并对结果应用wrap函数
        return tree_map(wrap, func(*args, **kwargs))


class ProfilerTree:
    @staticmethod
    @classmethod
    # 格式化输出给定的 Profiler 对象的事件树结构
    def format(cls, profiler, indent: int = 0):
        # 定义将事件树结构展平为列表的函数
        def flatten(nodes, depth=0, out=None):
            if out is None:
                out = []

            # 遍历每个节点
            for node in nodes:
                # 验证节点的有效性
                cls.validate_node(node)
                # 格式化节点的名称
                name = cls.fmt_name(node.name)
                # 根据节点名称是否在剪枝函数字典中决定处理方式
                prune_level = PRUNE_FUNCTIONS.get(name.strip(), None)
                if prune_level is None:
                    # 如果节点不需要剪枝，则加入到输出列表中
                    out.append((depth, name))
                    # 递归展平子节点
                    flatten(node.children, depth + 1, out)
                elif prune_level == KEEP_NAME_AND_ELLIPSES:
                    # 如果保留名称和省略号，则加入节点和省略号到输出列表
                    out.append((depth, name))
                    if node.children:
                        out.append((depth + 1, "..."))
                elif prune_level == KEEP_ELLIPSES:
                    # 如果只保留省略号，则加入省略号到输出列表
                    out.append((depth, "..."))
                else:
                    # 如果剪枝等级为 PRUNE_ALL，则断言删除该节点及其子节点
                    assert prune_level == PRUNE_ALL

            return out

        # 调用 flatten 函数，展平给定 profiler 对象的实验性事件树
        flat_nodes = flatten(profiler.kineto_results.experimental_event_tree())

        # 如果最后两个节点是 CUDA 同步事件，则删除这两个节点
        if flat_nodes and flat_nodes[-2][1] == "cudaDeviceSynchronize":
            flat_nodes = flat_nodes[:-2]

        # 如果最后一个节点是 CUDA 同步事件，则删除最后一个节点
        if flat_nodes and flat_nodes[-1][1] == "cudaDeviceSynchronize":
            flat_nodes = flat_nodes[:-1]

        # 如果最后一个节点是 HIP 同步事件，则删除最后一个节点
        if flat_nodes and flat_nodes[-1][1] == "hipDeviceSynchronize":
            flat_nodes = flat_nodes[:-1]

        # 计算最小深度，用于缩进调整
        min_depth = min(
            [d + 1 for d, name in flat_nodes if "begin_unit_test_marker" in name] or [0]
        )
        
        # 构建最终格式化的文本，根据缩进调整每行输出
        return textwrap.indent(
            "\n".join(
                [
                    f"{'  ' * (d - min_depth)}{name.rstrip()}"
                    for d, name in flat_nodes
                    if d >= min_depth
                ]
            ),
            " " * indent,
        )
    # 格式化异常或函数名称，返回格式化后的字符串表示
    def fmt_name(name: str) -> str:
        # 使用正则表达式匹配异常或函数名称格式
        match = re.match(r"^(.*)\.py\(([0-9]+)\): (.*)$", name)
        if match:
            # 提取文件名、行号和函数名
            filename, _, fn = match.groups()

            # 获取当前文件名（不带扩展名）
            test_file = os.path.splitext(os.path.split(__file__)[1])[0]
            # 如果文件名以当前运行的测试文件名结尾，则将文件名设为测试文件名
            if filename.endswith(test_file):
                filename = test_file

            # 将文件路径分隔符替换为 POSIX 风格的斜杠
            filename = filename.replace(os.sep, "/")

            # 由于目前不确定行号，所以设为省略号
            lineno = "..."

            return f"{filename}.py({lineno}): {fn}"

        # 替换包含特定内核模式的字符串，隐藏详细信息以避免依赖具体的实现细节
        for kernel_pattern in (
            "void at::native::elementwise_kernel",
            "void at::native::reduce_kernel",
            "void at::native::vectorized_elementwise_kernel",
            "void at::native::unrolled_elementwise_kernel",
            r"void [a-zA-Z0-9]+_kernel",  # Nvidia kernels.
        ):
            name = re.sub(
                rf"{kernel_pattern}<.+>\(.+\)$",
                f"{kernel_pattern.replace('[a-zA-Z0-9]+', '...')}<...>(...)",
                name,
            )

        # 替换对象地址信息，避免暴露内存地址
        return re.sub("object at 0x[0-9a-fA-F]+>", "object at 0xXXXXXXXXXXXX>", name)

    @classmethod
    def validate_node(cls, node):
        # 获取节点的额外字段
        extra_fields = node.extra_fields
        # 如果额外字段属于 PyCall 或者 PyCCall 类型
        if isinstance(extra_fields, (_ExtraFields_PyCall, _ExtraFields_PyCCall)):
            # 检查由分析器建立的谱系是否与 Python 追踪器记录的调用者匹配
            parent = node.parent
            while parent is not None:
                if isinstance(parent.extra_fields, _ExtraFields_PyCall):
                    break
                parent = parent.parent

            # 定义函数，将帧状态转换为字符串形式
            def to_string(frame_state):
                return f"{frame_state.file_name}(...): {frame_state.function_name}"

            # 如果存在父节点，比较父节点和当前节点的调用者信息
            if parent:
                parent_name = to_string(parent.extra_fields.callsite)
                caller_name = to_string(extra_fields.caller)
                assert parent_name == caller_name, f"{parent_name} vs. {caller_name}"
# 根据条件跳过 ARM 架构下的测试，因为在该架构上无法正常工作
@unittest.skipIf(IS_ARM64, "Not working on ARM")
class TestProfilerTree(TestCase):
    # 断言实际输出与预期输出树结构匹配
    def assertTreesMatch(self, actual: str, expected: str, allow_failure: bool = False):
        # 警告：这里可能会遇到问题
        #   不同平台对于 Python 跟踪的行为可能略有不同。观察到的差异包括：
        #     1) Windows 与 posix 下的符号名称解析不同
        #     2) 在某些平台上，对于 Tensor.__pow__ 的 c_call 回调不会触发。这不是函数跟踪器引起的，
        #        而是 cPython 本身的问题。
        #
        # 这些单元测试的目的是确保性能分析器执行合理的操作。当出现这些依赖平台的差异时，
        # 可以将它们转换为平台无关的形式。如果在代码库中进行了更改以更改生成的跟踪，
        # 可以简单地使用 EXPECTTEST_ACCEPT=1 来更新测试以反映新的结构。

        # 如果未通过 expecttest.ACCEPT 环境变量，actual 将用空格填充至与 expected 相同长度
        if not expecttest.ACCEPT:
            actual = actual.ljust(len(expected))
        # 允许输出的最大差异长度无限制
        self.maxDiff = None

        # 获取 self 中的 tree_replicate 属性，用于测试复制性
        replicate = getattr(self, "tree_replicate", None)
        # 断言 tree_replicate 不为 None，提示应使用 `@ProfilerTree.test` 注解测试方法
        self.assertIsNotNone(
            replicate, "Please annotate test with `@ProfilerTree.test`"
        )

        # 性能分析器应生成确定性结果，并且每次运行后应返回清洁状态。因此，
        # 只有第一个复制允许更新 `expected`。如果后续运行不匹配，则是性能分析器的 bug。
        if replicate:
            self.assertEqual(actual, expected)
        else:
            try:
                # 断言实际输出与预期输出行内匹配，跳过第一行
                self.assertExpectedInline(actual, expected, skip=1)
            except AssertionError as e:
                if allow_failure:
                    self.tree_replicate = None
                    # 输出 AssertionError 的详细信息，去除 "AssertionError:" 前缀
                    msg = traceback.format_exception_only(type(e), e)[0]
                    print(msg.split("AssertionError:")[-1])
                else:
                    raise

    # TODO: Add logic for CUDA version of test
    # 标记为性能分析器树的测试方法
    @ProfilerTree.test
    # 根据 CUDA 是否可用跳过 CUDA 版本的测试
    @unittest.skipIf(torch.cuda.is_available(), "Test not working for CUDA")
    # 定义名为 test_profiler_experimental_tree 的测试方法
    def test_profiler_experimental_tree(self):
        # 创建两个元素为 1 的张量，并标记需要梯度计算
        t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        
        # 使用 torch.profiler.profile() 创建性能分析器，并记录以下代码块的性能
        with torch.profiler.profile() as p:
            # 执行张量 t1 和 t2 的加法操作
            z = torch.add(t1, t2)
            # 创建一个元素为 1 的张量 y
            y = torch.ones(1)
            # 计算损失值，即 (y - z)^2
            loss = (y - z) ** 2
            # 根据损失值计算梯度
            loss.backward()
        
        # 使用 self.assertTreesMatch 方法比较分析器的结果与预期的字符串表示形式
        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            aten::add
            aten::ones
              aten::empty
              aten::fill_
            aten::sub
            aten::pow
              aten::result_type
              aten::to
            aten::ones_like
              aten::empty_like
                aten::empty_strided
              aten::fill_
            autograd::engine::evaluate_function: PowBackward0
              PowBackward0
                aten::pow
                  aten::result_type
                  aten::to
                  aten::copy_
                aten::mul
                  aten::mul
                    aten::to
                      aten::_to_copy
                        aten::empty_strided
                        aten::copy_
                aten::mul
            autograd::engine::evaluate_function: SubBackward0
              SubBackward0
                aten::neg
            autograd::engine::evaluate_function: AddBackward0
              AddBackward0
            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
              torch::autograd::AccumulateGrad
                aten::new_empty_strided
                  aten::empty_strided
                aten::copy_
            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
              torch::autograd::AccumulateGrad
                aten::detach
                  detach""",
        )

    # TODO: Add logic for CUDA version of test
    # 使用 ProfilerTree.test 进行测试，如果 CUDA 可用，则跳过该测试
    @ProfilerTree.test
    @unittest.skipIf(torch.cuda.is_available(), "Test not working for CUDA")
    # 定义测试函数 test_profiler_experimental_tree_with_record_function，用于性能分析实验树的记录功能
    def test_profiler_experimental_tree_with_record_function(self):
        # 使用 torch.profiler.profile() 创建性能分析器 p
        with torch.profiler.profile() as p:
            # 使用 torch.autograd.profiler.record_function("Top level Annotation") 记录顶层注解
            with torch.autograd.profiler.record_function("Top level Annotation"):
                # 使用 torch.autograd.profiler.record_function("First Annotation") 记录第一个注解
                with torch.autograd.profiler.record_function("First Annotation"):
                    # 创建一个元素为 1 的张量 x，并要求计算其梯度
                    x = torch.ones((1,), requires_grad=True)

                # 检查用户注解未调用 `__exit__` 的情况
                _ = torch.autograd.profiler.record_function(
                    "Second Annotation"
                ).__enter__()

                # 对 x 加 1
                y = x + 1
                # 使用 torch.autograd.profiler.record_function("Third Annotation") 记录第三个注解
                with torch.autograd.profiler.record_function("Third Annotation"):
                    # 执行反向传播
                    y.backward()

        # NB: `aten::zeros` 出现在记录函数注解之前，由于 `at::cpp_custom_type_hack`。当我们切换到 `torch::CustomClassHolder` 时，它们将消失。
        # 断言性能分析树与预期格式化的树匹配
        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            Top level Annotation
              First Annotation
                aten::ones
                  aten::empty
                  aten::fill_
              Second Annotation
                aten::add
                  aten::to
                    aten::_to_copy
                      aten::empty_strided
                      aten::copy_
                Third Annotation
                  aten::ones_like
                    aten::empty_like
                      aten::empty_strided
                    aten::fill_
                  autograd::engine::evaluate_function: AddBackward0
                    AddBackward0
                  autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
                    torch::autograd::AccumulateGrad
                      aten::new_empty_strided
                        aten::empty_strided
                      aten::copy_""",
        )

    # TODO: Add logic for CUDA version of test
    @ProfilerTree.test
    @unittest.skipIf(torch.cuda.is_available(), "Test not working for CUDA")
    # 定义名为 test_profiler_experimental_tree_with_memory 的测试方法
    def test_profiler_experimental_tree_with_memory(self):
        # 创建两个值为1的张量，并要求计算梯度
        t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        
        # 使用 torch.profiler.profile 进行性能分析，包括内存分析
        with torch.profiler.profile(profile_memory=True) as p:
            # 执行张量加法
            z = torch.add(t1, t2)
            # 创建一个值为1的张量 y
            y = torch.ones(1)
            # 计算损失值并反向传播梯度
            loss = (y - z) ** 2
            loss.backward()

        # 断言性能分析结果与指定的树结构匹配
        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            aten::add
              [memory]
            aten::ones
              aten::empty
                [memory]
              aten::fill_
            aten::sub
              [memory]
            aten::pow
              aten::result_type
              aten::to
              [memory]
            aten::ones_like
              aten::empty_like
                aten::empty_strided
                  [memory]
              aten::fill_
            autograd::engine::evaluate_function: PowBackward0
              PowBackward0
                aten::pow
                  aten::result_type
                  aten::to
                  [memory]
                  aten::copy_
                aten::mul
                  [memory]
                  aten::mul
                    aten::to
                      aten::_to_copy
                        aten::empty_strided
                          [memory]
                        aten::copy_
                    [memory]
                    [memory]
                  [memory]
                aten::mul
                  [memory]
                [memory]
                [memory]
              [memory]
            autograd::engine::evaluate_function: SubBackward0
              SubBackward0
                aten::neg
                  [memory]
              [memory]
            autograd::engine::evaluate_function: AddBackward0
              AddBackward0
            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
              torch::autograd::AccumulateGrad
                aten::new_empty_strided
                  aten::empty_strided
                    [memory]
                aten::copy_
            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
              torch::autograd::AccumulateGrad
                aten::detach
                  detach
            [memory]""",
        )

    # 跳过具有交叉引用的测试用例
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref 拦截调用并更改调用点。"
    )
    # 使用 ProfilerTree.test 装饰器标记测试用例
    @ProfilerTree.test
    # 如果使用 Torch Dynamo，跳过测试用例因为速度太慢
    @skipIfTorchDynamo("too slow")
    # 跳过具有交叉引用的测试用例
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref 拦截调用并更改调用点。"
    )
    # 跳过具有交叉引用的测试用例
    @ProfilerTree.test
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref 拦截调用并更改调用点。"
    )
    @ProfilerTree.test
    # 定义一个测试方法，用于测试带有栈和 torch 函数的实验性树
    def test_profiler_experimental_tree_with_stack_and_torch_function(self):
        # 创建一个 TorchFunctionTensor 对象，包含一个全为1的张量
        x = TorchFunctionTensor(torch.ones((1,)))
        # 创建一个全为1的 torch 张量
        y = torch.ones((1,))

        # 在 __torch_function__ 方法中存在一些惰性初始化。如果不运行这一行，第一次运行将无法匹配复制。
        torch.add(x, y)

        # 使用 torch.profiler.profile 开始性能分析，同时捕获调用栈
        with torch.profiler.profile(with_stack=True) as p:
            # 执行 torch.add 操作
            torch.add(x, y)

        # 使用 self.assertTreesMatch 方法断言性能分析结果与预期输出匹配
        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            test_profiler_tree.py(...): test_profiler_experimental_tree_with_stack_and_torch_function
              torch/profiler/profiler.py(...): __enter__
                ...
              <built-in method add of type object at 0xXXXXXXXXXXXX>
                test_profiler_tree.py(...): __torch_function__
                  torch/_tensor.py(...): __torch_function__
                    <built-in function all>
                      torch/_tensor.py(...): <genexpr>
                        <built-in function issubclass>
                      torch/_tensor.py(...): <genexpr>
                    <built-in method add of type object at 0xXXXXXXXXXXXX>
                      aten::add
                    torch/_tensor.py(...): _convert
                      <built-in function isinstance>
                      <built-in function isinstance>
                      <built-in method as_subclass of Tensor object at 0xXXXXXXXXXXXX>
                        aten::alias
                      <built-in function isinstance>
              torch/profiler/profiler.py(...): __exit__
                torch/profiler/profiler.py(...): stop
                  ...""",
        )

    # 装饰器用于跳过带有跨引用测试的情况，因为跨引用会拦截调用并改变调用点。
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    # 使用 ProfilerTree.test 装饰器定义另一个测试方法，用于测试带有栈和 torch 分派的实验性树
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_stack_and_torch_dispatch(self):
        # 创建一个 TorchDispatchTensor 对象，包含一个全为1的张量
        x = TorchDispatchTensor(torch.ones((1,)))
        # 创建一个全为1的 torch 张量
        y = torch.ones((1,))

        # 使用 torch.profiler.profile 开始性能分析，同时捕获调用栈
        with torch.profiler.profile(with_stack=True) as p:
            # 执行张量加法操作
            x + y

        # 使用 self.assertTreesMatch 方法断言性能分析结果与预期输出匹配
        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            test_profiler_tree.py(...): test_profiler_experimental_tree_with_stack_and_torch_dispatch
              torch/profiler/profiler.py(...): __enter__
                ...
              aten::add
                test_profiler_tree.py(...): __torch_dispatch__
                  torch/utils/_pytree.py(...): tree_map
                    ...
                  torch/utils/_pytree.py(...): tree_map
                    ...
                  torch/_ops.py(...): __call__
                    <built-in method  of PyCapsule object at 0xXXXXXXXXXXXX>
                      aten::add
                  torch/utils/_pytree.py(...): tree_map
                    ...
              torch/profiler/profiler.py(...): __exit__
                torch/profiler/profiler.py(...): stop
                  ...""",
        )
    # 跳过此测试，因为存在已知的 PyTorch GitHub 问题，详情见链接
    @unittest.skip("https://github.com/pytorch/pytorch/issues/83606")
    # 如果 CUDA 不可用，则跳过此测试
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    # 使用 ProfilerTree.test 标记此方法为性能分析测试
    @ProfilerTree.test
    # 再次跳过此测试，因为存在已知的 PyTorch GitHub 问题，详情见链接
    @unittest.skip("https://github.com/pytorch/pytorch/issues/83606")
    # 如果 CUDA 不可用，则再次跳过此测试
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    # 定义名为 test_profiler_experimental_tree_cuda_with_stream 的测试方法
    def test_profiler_experimental_tree_cuda_with_stream(self):
        # 创建三个 CUDA 流对象并存储在 streams 列表中
        streams = [torch.cuda.Stream() for _ in range(3)]
        # 创建一个空列表 results
        results = []
        # 使用 torch.profiler.profile 开始性能分析
        with torch.profiler.profile(profile_memory=True) as p:
            # 在 CUDA 设备上创建一个全为 1 的张量 x
            x = torch.ones((4, 4), device="cuda")
            # 遍历每个 CUDA 流对象
            for stream in streams:
                # 将当前操作放入指定的 CUDA 流中
                with torch.cuda.stream(stream):
                    # 计算 tanh(x) - x 并将结果添加到 results 列表中
                    results.append(torch.tanh(x) - x)
        # 删除 results 列表的引用，释放内存
        del results
        # 等待所有 CUDA 操作完成
        for s in streams:
            torch.cuda.current_stream().wait_stream(s)

        # 使用 self.assertTreesMatch 方法断言性能分析结果与预期的树状结构匹配
        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            aten::ones
              aten::empty
                [memory]
              aten::fill_
                cudaLaunchKernel
                  void at::native::vectorized_elementwise_kernel<...>(...)
            aten::tanh
              cudaMalloc
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            aten::sub
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            [memory]
            aten::tanh
              cudaMalloc
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            aten::sub
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            [memory]
            aten::tanh
              cudaMalloc
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            aten::sub
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            [memory]""",
            allow_failure=ALLOW_CUDA_FAILURE,
        )

    # 跳过此测试，因为存在已知的 PyTorch GitHub 问题，详情见链接
    @unittest.skip("https://github.com/pytorch/pytorch/issues/83606")
    # 如果 TEST_WITH_CROSSREF 为 True，则跳过此测试，因为 crossref 拦截调用并更改调用位置
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    # 如果 CUDA 不可用，则跳过此测试
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    # 使用 ProfilerTree.test 标记此方法为性能分析测试
    @ProfilerTree.test
# 如果当前脚本被直接执行（而非被导入为模块），则执行以下代码
if __name__ == "__main__":
    # 调用运行测试函数
    run_tests()
```