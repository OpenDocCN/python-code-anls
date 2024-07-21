# `.\pytorch\test\profiler\test_memory_profiler.py`

```
# Owner(s): ["oncall: profiler"]
# 导入必要的模块和库
import functools  # 导入 functools 模块
import gc  # 导入 gc 模块，用于垃圾回收
import itertools as it  # 导入 itertools 模块，提供迭代器的函数
import textwrap  # 导入 textwrap 模块，用于文本包装和填充
from typing import Callable, Dict, Iterator, List, Optional, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 深度学习框架
from torch._C._profiler import _EventType, _TensorMetadata  # 导入 _EventType 和 _TensorMetadata 类
from torch.profiler import _memory_profiler, _utils  # 导入内存分析器和工具
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase  # 导入测试相关的函数和类
from torch.utils import _pytree as pytree  # 导入 _pytree 模块，用于树形数据结构的操作


# 使用 functools.partial 创建一个 profile 函数的偏函数，用于性能分析
profile = functools.partial(
    torch.profiler.profile, record_shapes=True, profile_memory=True, with_stack=True
)


# 定义一个测试类 TestMemoryProfiler，继承自 TestCase
@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
class TestMemoryProfiler(TestCase):
    
    # 定义一个测试方法 test_config_check
    def test_config_check(self) -> None:
        # 使用 torch.profiler.profile 上下文管理器创建 prof 对象
        with torch.profiler.profile() as prof:
            pass

        # 设置断言，验证是否引发 ValueError 异常，异常信息包含指定的 pattern
        pattern = r"record_shapes=True, profile_memory=True, with_stack=True"
        with self.assertRaisesRegex(ValueError, pattern):
            prof._memory_profile()

        # 使用 torch.profiler.profile 上下文管理器创建 prof 对象，指定参数 record_shapes 和 with_stack 为 True
        with torch.profiler.profile(record_shapes=True, with_stack=True) as prof:
            pass

        # 设置断言，验证是否引发 ValueError 异常，异常信息符合指定的 pattern
        pattern = r"^profile_memory=True required for memory profiling\.$"
        with self.assertRaisesRegex(ValueError, pattern):
            prof._memory_profile()

        # 使用前面定义的 profile 偏函数创建 prof 对象
        with profile() as prof:
            pass

        # 断言，验证 prof._memory_profile() 的返回类型是 _memory_profiler.MemoryProfile 类型
        self.assertIsInstance(prof._memory_profile(), _memory_profiler.MemoryProfile)


# 定义一个 ScaleLayer 类，继承自 torch.nn.Module
class ScaleLayer(torch.nn.Module):
    
    # 定义初始化方法
    def __init__(self) -> None:
        super().__init__()
        # 创建一个随机初始化的可学习参数 scale
        self.scale = torch.nn.Parameter(torch.rand(()), requires_grad=True)

    # 定义前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回输入张量 x 与参数 scale 的乘积
        return x * self.scale


# 定义一个 LazyLinear 类，继承自 torch.nn.Module
class LazyLinear(torch.nn.Module):
    
    # 定义初始化方法
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数

    # 定义前向传播方法
    def forward(self, x) -> torch.Tensor:
        # 如果权重 weight 属性为 None，则初始化权重和偏置参数
        if getattr(self, "weight", None) is None:
            self.weight = torch.nn.Parameter(
                torch.empty((self.out_features, self.in_features))
            )
            self.bias = torch.nn.Parameter(torch.empty(self.out_features))

        # 使用 torch.nn.functional.linear 执行线性变换操作，并返回结果
        return torch.nn.functional.linear(x, self.weight, self.bias)


# 定义一个 RecordInputOutputDispatchMode 类，继承自 torch.utils._python_dispatch.TorchDispatchMode
class RecordInputOutputDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    
    # 定义初始化方法
    def __init__(self):
        self.results = []  # 初始化结果列表

    # 定义 mark_region 方法
    def mark_region(self, name: str):
        # 将传入的 name 参数和空的元组添加到结果列表
        self.results.append((name, (), ()))

    # 定义静态方法 flat_ids
    @staticmethod
    def flat_ids(args):
        # 获取 args 中所有叶子节点的列表
        flat_args = pytree.tree_leaves(args)
        # 遍历 flat_args 中的每个元素，筛选出是 torch.Tensor 类型且有存储的元素
        return tuple(
            (t._cdata, t.storage().data_ptr())
            for t in flat_args
            if isinstance(t, torch.Tensor) and t.storage()
        )
    # 定义特殊方法 __torch_dispatch__，用于处理函数调度
    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        # 如果 args 为 None，则设为一个空列表
        args = args or []
        # 如果 kwargs 为 None，则设为一个空字典
        kwargs = kwargs or {}
        # 获取 args 和 kwargs 中所有元素的平铺 ID
        flat_inputs = self.flat_ids(args) + self.flat_ids(kwargs)
        # 调用 func 函数，传入 args 和 kwargs，获取输出结果 out
        out = func(*args, **kwargs)
        # 获取 out 中所有元素的平铺 ID
        flat_outputs = self.flat_ids(out)
        # 如果 flat_inputs 或 flat_outputs 非空，并且 func.name() 不包含 "_record_function_enter"
        if (
            flat_inputs or flat_outputs
        ) and "_record_function_enter" not in func.name():
            # 将函数名 func.name()、输入的平铺 ID flat_inputs 和输出的平铺 ID flat_outputs 添加到结果列表 self.results 中
            self.results.append((func.name(), flat_inputs, flat_outputs))
        # 返回函数的输出结果 out
        return out
@skipIfTorchDynamo("TorchDynamo changes Python calls that memory profiling relies on.")
# 装饰器：如果TorchDynamo对内存分析依赖的Python调用有变化，则跳过执行该测试类。

class TestIdentifyGradients(TestCase):
    # 测试类：用于测试梯度识别功能

    def gradient_detected(
        self,
        prof: torch.profiler.profile,
        ctx: _EventType,
        grad_tensor: torch.Tensor,
        parameter: Optional[torch.Tensor] = None,
    ) -> None:
        # 梯度检测函数：用于检测给定上下文中的梯度是否存在

        # This is not an exhaustive check, but for the purpose of unit testing
        # it is sufficient.
        # 这并非详尽的检查，但对于单元测试来说已经足够。

        def key_matches_tensor(key, tensor) -> bool:
            # 确定键是否与张量匹配的辅助函数

            # Vacuous case.
            # 空情况
            if tensor is None:
                return True

            if key is None:
                return False

            return tensor.storage().data_ptr() == key.storage.ptr

        tree = prof.profiler.kineto_results.experimental_event_tree()
        # 获取实验性事件树

        for node in _utils.traverse_dfs(tree):
            # 深度优先遍历实验性事件树

            for p_key, p_grad_key in _memory_profiler.extract_gradients(node):
                # 从节点中提取梯度信息

                if node.tag == ctx and key_matches_tensor(p_grad_key, grad_tensor):
                    # 如果节点标签与给定上下文匹配，并且梯度键与梯度张量匹配

                    if parameter is None:
                        return True  # Don't need to check parameter; we're done.
                        # 不需要检查参数；已完成检测。

                    elif p_key is not None:
                        # For a complex workflow a gradient could correspond to
                        # different parameters at different points in a trace.
                        # However this will not happen in the relatively simple
                        # cases tested here, so if `extract_gradients` identifies
                        # the parameter corresponding to a particular gradient it
                        # must be the one we expect.
                        # 复杂的工作流程中，梯度可能对应于追踪中不同点的不同参数。
                        # 但在这里测试的相对简单情况下，如果`extract_gradients`识别
                        # 出与特定梯度对应的参数，则它必须是我们期望的那个。
                        self.assertTrue(key_matches_tensor(p_key, parameter))
                        return True

        return False
        # 如果未找到匹配的梯度，返回 False

    def assertGradientDetected(self, name: str, *args, **kwargs) -> None:
        # 断言函数：用于确保梯度已被成功识别

        self.assertTrue(
            self.gradient_detected(*args, **kwargs),
            f"Failed to identify gradient `{name}` from profile.",
        )
        # 断言梯度已被成功识别，否则抛出失败信息

    def assertOnlyGradients(
        self, prof: torch.profiler.profile, tensors: Iterator[torch.Tensor]
    ) -> None:
        # 断言函数：确保仅有梯度存在

        allowed_set = {t.storage().data_ptr() for t in tensors}
        # 允许的梯度集合：由输入张量的存储指针组成

        tree = prof.profiler.kineto_results.experimental_event_tree()
        # 获取实验性事件树

        for node in _utils.traverse_dfs(tree):
            # 深度优先遍历实验性事件树

            for _, p_grad_key in _memory_profiler.extract_gradients(node):
                # 从节点中提取梯度信息

                self.assertTrue(
                    p_grad_key.storage.ptr in allowed_set,
                    f"Tensor wrongly marked as gradient: {node.name}: {p_grad_key}",
                )
                # 断言梯度张量是否在允许的集合中，否则抛出错误信息
    # 定义一个测试方法，用于验证从低级别提取梯度的功能
    def test_extract_gradients_low_level(self) -> None:
        # 创建一个张量 x，所有元素均为 1
        x = torch.ones((1,))
        # 创建一个需要梯度的张量 w0，所有元素均为 1
        w0 = torch.ones((1,), requires_grad=True)
        # 创建一个需要梯度的张量 w1，所有元素均为 1
        w1 = torch.ones((1,), requires_grad=True)

        # 定义一个内部函数 check，用于验证梯度提取
        def check(cold_start: bool):
            # 断言在冷启动时 w0 和 w1 的梯度应为 None
            self.assertEqual(w0.grad is None, cold_start)
            self.assertEqual(w1.grad is None, cold_start)
            
            # 使用 profile() 开始性能分析
            with profile() as prof:
                # 创建张量 z，将 x 扩展成 4 个元素，然后与 w0 相乘
                z = x.expand(4) * w0
                # 计算 (z * w1) 的和，并进行反向传播
                (z * w1).sum().backward()

            # 断言通过操作检查梯度，应指向 w0 和 w1 的梯度
            self.assertGradientDetected("w0", prof, _EventType.TorchOp, w0.grad)
            self.assertGradientDetected("w1", prof, _EventType.TorchOp, w1.grad)
            # 断言只检测到了 w0 和 w1 的梯度
            self.assertOnlyGradients(prof, (w0.grad, w1.grad))

        # 验证冷启动和热启动两种情况下的梯度提取
        check(cold_start=True)
        check(cold_start=False)

    # 定义一个测试方法，用于从模块中提取梯度
    def test_extract_gradients_from_module(self) -> None:
        # 创建一个包含线性层和自定义缩放层的序列模型
        model = torch.nn.Sequential(torch.nn.Linear(2, 1), ScaleLayer())
        # 获取模型中所有参数的名称和参数本身，并组成字典
        named_parameters = dict(model.named_parameters())
        # 断言模型中的参数个数为 3
        self.assertEqual(len(named_parameters), 3)

        # 定义一个内部函数 assert_only_gradients，用于断言所有参数均有梯度
        def assert_only_gradients(prof: torch.profiler.profile):
            # 获取所有参数的梯度，并组成元组
            gradients = tuple(i.grad for i in named_parameters.values())
            # 断言所有梯度均不为 None
            self.assertFalse(any(i is None for i in gradients))
            # 断言性能分析中只有这些梯度存在
            self.assertOnlyGradients(prof, gradients)

        # 定义一个内部函数 check，用于验证梯度提取
        def check(cold_start: bool):
            # 创建一个 2x2 的张量 x，所有元素均为 1
            x = torch.ones((2, 2))
            # 使用 profile() 开始性能分析
            with profile() as prof:
                # 对模型进行前向传播并计算损失，然后反向传播
                model(x).sum().backward()

            # 遍历所有命名参数和其名称
            for name, p in named_parameters.items():
                # 第一次运行模块时，所有 `.grad` 字段都未初始化，这是正常的；
                # 在这种情况下，我们可以在分析的部分检测到所有需要的信息。
                self.assertNotEqual(
                    self.gradient_detected(prof, _EventType.PyCall, p.grad, p),
                    cold_start,
                    name,
                )

                # 基于操作的检测仍然应该识别出梯度
                self.assertGradientDetected(name, prof, _EventType.TorchOp, p.grad)
            
            # 断言所有参数均有梯度
            assert_only_gradients(prof)

            # 即使没有调用 `.backward()`，我们仍然可以检测到梯度
            with profile() as prof:
                model(torch.ones((2, 2)))

            # 遍历所有命名参数和其名称
            for name, p in named_parameters.items():
                # 断言在性能分析中通过 PyCall 检测到了梯度
                self.assertGradientDetected(name, prof, _EventType.PyCall, p.grad, p)
                # 断言没有通过 TorchOp 检测到梯度
                self.assertFalse(
                    self.gradient_detected(prof, _EventType.TorchOp, p.grad), name
                )
            
            # 断言所有参数均有梯度
            assert_only_gradients(prof)

        # 验证冷启动和热启动两种情况下的梯度提取
        check(cold_start=True)
        check(cold_start=False)
    # 定义测试方法，用于测试从优化器中提取梯度的行为
    def _test_extract_gradients_from_optimizer(self, set_to_none: bool) -> None:
        # 创建张量 x，全为 1，并且需要计算梯度
        x = torch.ones((1,))
        # 创建张量 w0，全为 1，并且需要计算梯度
        w0 = torch.ones((1,), requires_grad=True)
        # 创建张量 w1，全为 1，并且需要计算梯度
        w1 = torch.ones((1,), requires_grad=True)
        # 创建 SGD 优化器，传入需要优化的张量 w0 和 w1，设定学习率为 0.1，动量为 0.9
        optimizer = torch.optim.SGD((w0, w1), lr=0.1, momentum=0.9)

        # 定义内部函数 check，用于检查梯度的初始化状态
        def check(cold_start: bool):
            # 断言 w0 的梯度是否为 None，与冷启动状态（cold_start）相符
            self.assertEqual(w0.grad is None, cold_start)
            # 断言 w1 的梯度是否为 None，与冷启动状态（cold_start）相符
            self.assertEqual(w1.grad is None, cold_start)
            # 使用 profile() 上下文进行性能分析
            with profile() as prof:
                # 调用优化器的 zero_grad 方法，根据 set_to_none 参数决定是否将梯度置为 None
                optimizer.zero_grad(set_to_none=set_to_none)
                # 执行张量操作，将张量 x 扩展为长度为 4，并与 w0 相乘
                z = x.expand(4) * w0
                # 计算 (z * w1) 的和，并进行反向传播
                (z * w1).sum().backward()
                # 执行优化器的参数更新
                optimizer.step()

            # 在步骤末尾执行优化器的仪表化，因此可以检测到冷启动和热启动的梯度
            self.assertGradientDetected("w0", prof, _EventType.PyCall, w0.grad, w0)
            self.assertGradientDetected("w1", prof, _EventType.PyCall, w1.grad, w1)

            # 检查仪表化过程中是否检测到 Torch 操作引发的梯度
            self.assertGradientDetected("w0", prof, _EventType.TorchOp, w0.grad)
            self.assertGradientDetected("w1", prof, _EventType.TorchOp, w1.grad)
            # 断言仪表化结果只包含 w0 和 w1 的梯度
            self.assertOnlyGradients(prof, (w0.grad, w1.grad))

            # 再次使用 profile() 上下文进行性能分析
            with profile() as prof:
                # 多次执行以下操作：
                for _ in range(2):
                    # 调用优化器的 zero_grad 方法，根据 set_to_none 参数决定是否将梯度置为 None
                    optimizer.zero_grad(set_to_none=set_to_none)
                    # 执行张量操作，将张量 x 扩展为长度为 4，并与 w0 相乘
                    z = x.expand(4) * w0
                    # 计算 (z * w1) 的和，并进行反向传播
                    (z * w1).sum().backward()
                    # 执行优化器的参数更新
                    optimizer.step()

            # 检查内部状态是否已缓存，如果替换梯度（例如 `set_to_none=True`），则 Python 仪表化将无法检测到
            # TODO(robieta): 是否应该在缓存中排除 `.step()`？
            self.assertNotEqual(
                self.gradient_detected(prof, _EventType.PyCall, w0.grad, w0),
                set_to_none,
            )

            self.assertNotEqual(
                self.gradient_detected(prof, _EventType.PyCall, w1.grad, w1),
                set_to_none,
            )

            # 如果 `set_to_none=True`，则使用断言检查是否仅标记了错误的张量
            if set_to_none:
                with self.assertRaisesRegex(AssertionError, "Tensor wrongly marked"):
                    self.assertOnlyGradients(prof, (w0.grad, w1.grad))

        # 分别测试冷启动和热启动的情况
        check(cold_start=True)
        check(cold_start=False)

    # 测试从优化器中提取梯度的行为，将 `set_to_none=False` 传入测试方法 `_test_extract_gradients_from_optimizer`
    def test_extract_gradients_from_optimizer(self) -> None:
        self._test_extract_gradients_from_optimizer(set_to_none=False)

    # 测试从优化器中提取梯度的行为，将 `set_to_none=True` 传入测试方法 `_test_extract_gradients_from_optimizer`
    def test_extract_gradients_from_optimizer_set_to_none(self) -> None:
        self._test_extract_gradients_from_optimizer(set_to_none=True)
    # 定义一个测试方法，用于从模型和优化器中提取梯度信息
    def test_extract_gradients_from_module_and_optimizer(self) -> None:
        # 模块和优化器分别进行了详尽测试，它们应该可以相加使用。
        # 因此我们可以进行轻量级的检查，确保它们不会产生不良互动。
        
        # 创建一个简单的神经网络模型，包括一个线性层和自定义的ScaleLayer
        model = torch.nn.Sequential(torch.nn.Linear(2, 1), ScaleLayer())
        
        # 使用随机梯度下降优化器（SGD），设置学习率和动量参数
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        
        # 使用profile()上下文管理器，记录模型执行时的性能数据
        with profile() as prof:
            # 将输入张量传递给模型，计算输出并进行反向传播
            model(torch.ones((2, 2))).sum().backward()
            # 使用优化器执行一步梯度下降
            optimizer.step()

        # 断言检查，确保模型的权重参数的梯度在性能分析数据中被正确检测到
        self.assertGradientDetected(
            "weight", prof, _EventType.PyCall, model[0].weight.grad, model[0].weight
        )
# 装饰器，用于在满足条件时跳过测试（条件：TorchDynamo移除了性能分析器）
@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
# 定义一个测试类 TestDataFlow，继承自 TestCase 类
class TestDataFlow(TestCase):
    
    # 设置测试环境的准备工作
    def setUp(self) -> None:
        super().setUp()
        # 设置最大差异为无限制，即允许所有差异
        self.maxDiff = None

    # 静态方法：格式化分析器的模式
    @staticmethod
    def formatSchemas(
        prof: torch.profiler.profile, indent: int = 12
    ) -> Tuple[Tuple[str, Tuple[bool, ...]], ...]:
        # 获取实验性事件树
        tree = prof.profiler.kineto_results.experimental_event_tree()
        # 输出列表，元素为元组，包含事件名称和输入是否可变的布尔值元组
        out: List[Tuple[str, Tuple[bool, ...]]] = []
        # 深度优先遍历树中的节点
        for node in _utils.traverse_dfs(tree):
            # 如果节点类型为 Torch 操作
            if node.tag == _EventType.TorchOp:
                # 获取附加字段
                e = node.extra_fields
                # 匹配模式，返回匹配到的模式列表
                schemas = _memory_profiler.SchemaMatcher.match_schemas(e)
                name = node.name
                # 根据模式数量调整节点名称
                if len(schemas) == 1:
                    name = f"{name}.{schemas[0].overload_name}"
                elif len(schemas) > 1:
                    name = f"{name}.{{{', '.join(s.overload_name for s in schemas)}}}"

                # 将结果添加到输出列表
                out.append((name, _memory_profiler.SchemaMatcher.inputs_are_mutable(e)))
        # 返回元组形式的输出列表
        return tuple(out)

    # 静态方法：运行并格式化数据流
    @staticmethod
    def _run_and_format_data_flow(
        inputs: Dict[str, torch.Tensor],
        f: Callable[..., Optional[Dict[str, torch.Tensor]]],
        indent: int = 12,
    ) -> str:
        # 使用性能分析器 profile 来运行函数 f，并记录性能数据
        with profile() as prof:
            outputs = f(**inputs) or {}  # 执行函数 f，获取输出结果
            gc.collect()  # 手动触发垃圾回收

        # 获取内存分析数据
        memory_profile = prof._memory_profile()
        # 获取数据流图
        graph = memory_profile._data_flow_graph
        # 创建存储指针到ID的映射字典
        storage_to_id = {key.storage.ptr: key.id for key in graph._active_version}

        # 输出行列表
        lines: List[str] = []
        # 遍历输入和输出，构建输出行
        for name, t in it.chain(inputs.items(), outputs.items()):
            lines.append(f"{name + ':':<8} T{storage_to_id[t.storage().data_ptr()]}")
            if t.grad is not None:
                grad_id = storage_to_id[t.grad.storage().data_ptr()]
                lines.append(f"{name + '.grad:':<9} T{grad_id}")

        if lines:
            lines.append("")

        # 遍历数据流图中的流节点
        for node in graph.flow_nodes:
            # 获取被删除的键集合
            destroyed = {k for k, v in node._edges.items() if v.is_deletion}

            inputs: List[str] = []
            # 构建输入字符串列表
            for key, (_, v) in node.inputs.items():
                inputs.append(f"T{key.id}(v{v}{'*' if key in destroyed else ''})")

            # 构建输出字符串列表
            outputs = [f"T{key.id}(v{v})" for key, v in node.outputs.items()]
            if inputs or outputs:
                # 获取事件名称，去除前缀并对齐输出格式
                event_name = node._event.name.replace("torch::autograd::", "")
                lines.append(
                    f"{event_name:<25} {', '.join(inputs):<15}  ->  {', '.join(outputs)}"
                )

        # 将行列表转换为缩进的文本块，并返回结果
        return textwrap.indent("\n".join([l.rstrip() for l in lines]), " " * indent)
    def test_match_schemas(self) -> None:
        # 使用 profile() 上下文管理器开始性能分析
        with profile() as prof:
            # 创建一个形状为 (1,) 的全为1的张量 x，乘以2并加2
            x = torch.ones((1,)).mul(2).add_(2)
            # 对 x 应用 sin 函数，结果存储在一个预先分配的张量中
            _ = torch.sin(x, out=torch.empty_like(x))

        # 断言分析结果与预期的格式化模式匹配
        self.assertEqual(
            self.formatSchemas(prof),
            (
                ("aten::ones.", (False,) * 5),
                ("aten::empty.memory_format", (False,) * 6),
                #
                # fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
                ("aten::fill_.Scalar", (True, False)),
                ("aten::mul.Tensor", (False, False)),
                ("aten::to.dtype", (False,) * 5),
                ("aten::_to_copy.", (False,) * 7),
                ("aten::empty_strided.", (False,) * 6),
                #
                # copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
                ("aten::copy_.", (True, False, False)),
                #
                # add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
                ("aten::add_.Tensor", (True, False, False)),
                ("aten::to.dtype", (False,) * 5),
                ("aten::_to_copy.", (False,) * 7),
                ("aten::empty_strided.", (False,) * 6),
                #
                # copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
                ("aten::copy_.", (True, False, False)),
                ("aten::empty_like.", (False,) * 6),
                ("aten::empty_strided.", (False,) * 6),
                #
                # sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
                ("aten::sin.out", (False, True)),
            ),
        )

    def test_match_schemas_backward(self) -> None:
        # 创建形状为 (1,) 的全为1的张量 x，设置 requires_grad=True
        x = torch.ones((1,))
        w = torch.ones((1,), requires_grad=True)
        # 使用 profile() 上下文管理器开始性能分析
        with profile() as prof:
            # 计算 x 和 w 的乘积，并执行反向传播
            torch.mul(x, w).backward()

        # 断言分析结果与预期的格式化模式匹配
        self.assertEqual(
            self.formatSchemas(prof),
            (
                ("aten::mul.Tensor", (False, False)),
                ("aten::ones_like.", (False,) * 6),
                ("aten::empty_like.", (False,) * 6),
                ("aten::empty_strided.", (False,) * 6),
                #
                # fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
                ("aten::fill_.Scalar", (True, False)),
                ("autograd::engine::evaluate_function: MulBackward0", ()),
                ("MulBackward0", (None,)),
                ("aten::mul.Tensor", (False, False)),
                (
                    "autograd::engine::evaluate_function: torch::autograd::AccumulateGrad",
                    (),
                ),
                ("torch::autograd::AccumulateGrad", (None,)),
                ("aten::detach.", (False,)),
                ("detach", (None,)),
            ),
        )
    def test_match_schemas_tensorlist(self) -> None:
        # 创建两个形状为 (1,) 的张量 x 和 y
        x = torch.ones((1,))
        y = torch.ones((1,))
        # 使用 torch.profiler.profile() 开始性能分析
        with profile() as prof:
            # 在轴 0 上对张量 x 和 y 进行拼接
            torch.cat([x, y], axis=0)

        # 断言性能分析结果的格式化后的模式与预期一致
        self.assertEqual(
            self.formatSchemas(prof),
            (("aten::cat.", (False, False)),),
        )

    def test_data_flow_graph_with_annotations(self) -> None:
        def f(x, y):
            # torch._C._jit_get_schemas_for_operator 会拒绝任何没有命名空间的名称
            # 我们想要检查跳过没有模式的注释（从 SchemaMatcher.lookup_schemas 返回空元组）
            # 和无法有模式的注释（从 SchemaMatcher.lookup_schemas 返回 None）。
            with torch.profiler.record_function("Namespaced::Annotation"):
                with torch.profiler.record_function("My Annotation"):
                    # 将张量 x 和 y 的值置零
                    x.zero_()
                    y.zero_()
                    # 返回一个字典，其中包含 x0 和 y0 分别是张量 x 和 y 的相应形状的新张量
                    return {"x0": torch.ones_like(x), "y0": torch.zeros_like(y)}

        inputs = {"x": torch.ones((1,)), "y": torch.ones((1,))}
        # 断言运行数据流并格式化输出与预期一致
        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f),
            """\
            x:       T0
            y:       T1
            x0:      T2
            y0:      T3

            aten::zero_               T0(v0)           ->  T0(v1)
            aten::zero_               T1(v0)           ->  T1(v1)
            aten::ones_like           T0(v1)           ->  T2(v0)
            aten::zeros_like          T1(v1)           ->  T3(v0)""",
        )

    def test_data_flow_graph_non_op_allocations(self) -> None:
        def f(x):
            # 将张量 x 的每个元素乘以 2
            x.mul(2)

        # Python 参数解析器将 Python 标量 `2` 转换为张量，以便传递给 `aten::mul`。
        # 因此，没有一个操作拥有此分配。张量的删除也不是在操作中进行的；它们是由于 Python 对象超出作用域而被收集的。
        self.assertExpectedInline(
            self._run_and_format_data_flow({"x": torch.ones((1,))}, f),
            """\
            x:       T1

            [memory]                                   ->  T0(v0)
            aten::mul                 T0(v0), T1(v0)   ->
            [memory]                  T0(v0*)          ->""",
        )
    def test_data_flow_graph_simple(self) -> None:
        # 定义输入数据字典，包含两个张量 'x' 和 'y'
        inputs = {"x": torch.ones((25,)), "y": torch.ones((25,), requires_grad=True)}

        # 定义函数 f0，接受 x 和 y 作为输入，计算它们的乘积并返回一个包含结果的字典
        def f0(x, y):
            z = x.mul(y)
            return {"z": z.view_as(z)}

        # 定义函数 f1，接受 x 和 y 作为输入，在不计算梯度的情况下调用函数 f0 并返回结果
        def f1(x, y):
            with torch.no_grad():
                return f0(x, y)

        # 断言调用 _run_and_format_data_flow 方法后返回的数据流图与期望的结果一致
        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f0),
            """\
            x:       T0
            y:       T1
            z:       T2

            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::view_as             T2(v0)           ->""",
        )

        # 对于不计算梯度的情况下，输出的数据流图应该与上面一致，因此再次断言结果
        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f0),
            """\
            x:       T0
            y:       T1
            z:       T2

            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::view_as             T2(v0)           ->""",
        )

    def test_data_flow_graph_simple_inplace(self) -> None:
        # 定义输入数据字典，包含两个张量 'x' 和 'y'
        inputs = {"x": torch.ones((25,)), "y": torch.ones((25,), requires_grad=True)}

        # 定义函数 f0，接受 x 和 y 作为输入，对 x 执行原地乘法操作
        def f0(x, y):
            x.mul_(y)

        # 定义函数 f1，接受 x 和 y 作为输入，在不计算梯度的情况下调用函数 f0 并返回结果
        def f1(x, y):
            with torch.no_grad():
                return f0(x, y)

        # 当 Autograd 启用时，执行原地操作会创建额外的张量 `T2` 用于反向传播
        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f0),
            """\
            x:       T0
            y:       T1

            aten::mul_                T0(v0), T1(v0)   ->  T0(v1), T2(v0)""",
        )

        # 当不计算梯度时，输出的数据流图应该与上面一致，因此再次断言结果
        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f1),
            """\
            x:       T0
            y:       T1

            aten::mul_                T0(v0), T1(v0)   ->  T0(v1)""",
        )

    def test_data_flow_graph_simple_backward(self) -> None:
        # 定义输入数据字典，包含两个张量 'x' 和 'w'，其中 'w' 需要计算梯度
        inputs = {
            "x": torch.ones((1,)),
            "w": torch.ones((1,), requires_grad=True),
        }

        # 断言调用 _run_and_format_data_flow 方法后返回的数据流图与期望的结果一致
        self.assertExpectedInline(
            self._run_and_format_data_flow(
                inputs, lambda x, w: (x * w).sin().backward()
            ),
            """\
            x:       T0
            w:       T1
            w.grad:   T7

            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::sin                 T2(v0)           ->  T3(v0)
            aten::ones_like           T3(v0)           ->  T4(v0)
            SinBackward0              T2(v0), T4(v0)   ->  T6(v0)
            [memory]                  T2(v0*)          ->
            MulBackward0              T0(v0), T6(v0)   ->  T7(v0)
            [memory]                  T6(v0*)          ->
            AccumulateGrad            T7(v0)           ->
            [memory]                  T4(v0*)          ->
            [memory]                  T3(v0*)          ->""",
        )
    # 定义测试方法，用于测试复杂数据流图
    def test_data_flow_graph_complicated(self) -> None:
        # 定义函数 f，生成和处理张量数据流，并返回结果字典
        def f():
            # 创建一个包含 25 个元素的全为 1 的张量 x
            x = torch.ones((25,))
            # 将张量 x 中的每个元素乘以 2，并加上 2，结果存入 y
            y = x.mul(2).add_(2)
            # 对张量 y 中的每个元素求正弦，结果存入新创建的张量 z
            z = torch.sin(y, out=torch.empty_like(y))
            # 返回包含 x, y, z 的字典
            return {"x": x, "y": y, "z": z}

        # 断言测试的预期输出与实际运行的数据流结果一致
        self.assertExpectedInline(
            self._run_and_format_data_flow({}, f),
            """\
            x:       T0
            y:       T3
            z:       T6

            aten::ones                                 ->  T0(v0)
            [memory]                                   ->  T1(v0)
            aten::mul                 T0(v0), T1(v0)   ->  T3(v0)
            [memory]                  T1(v0*)          ->
            [memory]                                   ->  T4(v0)
            aten::add_                T3(v0), T4(v0)   ->  T3(v1)
            [memory]                  T4(v0*)          ->
            aten::empty_like          T3(v1)           ->  T6(v0)
            aten::sin                 T3(v1), T6(v0)   ->  T6(v1)""",
        )

        # 使用性能分析器 profile() 记录函数 f() 的执行过程
        with profile() as prof:
            f()

        # 检查流程图中的乘法节点，确认是否创建了临时张量 T2
        mul_node = prof._memory_profile()._data_flow_graph.flow_nodes[2]
        # 断言乘法节点的事件名称为 "aten::mul"
        self.assertEqual(mul_node._event.name, "aten::mul")
        # 断言乘法节点的临时张量列表长度为 1，且第一个临时张量的 ID 为 2
        self.assertEqual(len(mul_node.intermediates), 1)
        self.assertEqual(mul_node.intermediates[0].id, 2)
# 如果 TorchDynamo 换变了内部 Python 调用，依赖内存分析的功能可能受影响，因此跳过该测试。
@skipIfTorchDynamo("TorchDynamo changes Python calls that memory profiling relies on.")
class TestMemoryProfilerE2E(TestCase):

    # 静态方法：查找张量的分类信息
    @staticmethod
    def _lookup_tensor_categories(
        t: torch.Tensor, memory_profile: _memory_profiler.MemoryProfile
    ) -> Dict[_memory_profiler.TensorAndID, Optional[_memory_profiler.Category]]:
        # 获取张量的存储
        storage = t.storage()
        if storage is None:
            raise ValueError("Cannot look up uninitialized Tensor.")

        # 获取内存快照的分类信息
        snapshot = memory_profile._category_snapshot()
        
        # 查找与当前张量关联的所有分配ID
        ids = {
            key.storage.allocation_id
            for key, _ in snapshot
            if key.storage.ptr == storage.data_ptr() and key.device == storage.device
        }

        # 返回张量及其版本的分类信息
        return {
            (key, version): category
            for (key, version), category in memory_profile._category_snapshot().items()
            # 如果张量是活跃的，选择最新的分配ID
            if key.storage.allocation_id == max(ids | {-1})
        }

    # 运行内部函数并检查参数和梯度
    def _run_and_check_parameters_and_gradients(
        self, inner_fn, model, grads_none: bool = False
    ):
        # 使用性能分析器进行性能分析
        with profile() as prof:
            inner_fn()

        # 获取内存分析信息
        memory_profile = prof._memory_profile()

        # 断言张量的分类信息
        def assert_category(
            t: torch.Tensor,
            category: _memory_profiler.Category,
            should_be_none: bool = False,
        ):
            if should_be_none:
                assert t is None, "tensor should be None but is not."
                return
            self.assertIsNotNone(t)
            categories = self._lookup_tensor_categories(t, memory_profile)
            self.assertGreater(len(categories), 0)
            self.assertTrue(all(c == category for c in categories.values()), categories)

        # 对模型的参数和梯度进行分类断言
        for p in model.parameters():
            assert_category(p, _memory_profiler.Category.PARAMETER)
            assert_category(p.grad, _memory_profiler.Category.GRADIENT, grads_none)

        # 依赖内部断言
        _ = memory_profile.timeline
    def _run_and_format_categories(self, fn, indent=12):
        """Generate summary of assigned categories for expecttest."""

        # Use `__torch_dispatch__` to collect ground truth.
        # 使用 `__torch_dispatch__` 收集真实数据。

        with RecordInputOutputDispatchMode() as record_ops, profile() as prof:
            # Record input-output operations and profile execution.
            # 记录输入输出操作并进行性能分析。
            fn(lambda name: record_ops.mark_region(f"-- {name} ".ljust(105, "-")))

        memory_profile = prof._memory_profile()
        # Retrieve memory profiling information.
        # 获取内存分析信息。
        
        ptr_pair_to_key: Dict[Tuple[int, int], _memory_profiler.TensorKey] = {}
        # Dictionary mapping pairs of pointers to TensorKey objects.

        snapshot = memory_profile._category_snapshot()
        # Snapshot of memory categories.

        # Build map from observed live Tensors to the memory profiler's
        # TensorKey representation.
        # 从观察到的活跃张量构建映射到内存分析器的张量键表示。
        for op in memory_profile._op_tree.dfs():
            if op.typed[0] == _EventType.TorchOp:
                inputs = pytree.tree_leaves(op.typed[1].inputs)
                for t in (i for i in inputs if isinstance(i, _TensorMetadata)):
                    key = _memory_profiler.TensorKey.from_tensor(t)
                    if key:
                        ptr_pair_to_key[(t.impl_ptr, t.storage_data_ptr)] = key

        def format_categories(ptr_pair: int):
            # Format categories based on pointer pair.
            # 根据指针对格式化类别信息。
            target_key = ptr_pair_to_key.get(ptr_pair, None)
            if target_key is None:
                return "???"

            matches = tuple(
                (version, category.name if category else "???")
                for (key, version), category in snapshot.items()
                if key == target_key
            )
            assert matches, "Failed to lookup Tensor"

            # Deduplicate version bumps which don't change the category.
            # 去除不改变类别的版本变化。
            categories = [matches[0][1]]
            for _, category in matches:
                if category != categories[-1]:
                    categories.append(category)

            return f"{target_key.storage.allocation_id} ({','.join(categories)})"

        out: List[str] = []
        for name, inputs, outputs in record_ops.results:
            if inputs or outputs:
                # PyTorch ops
                # Handle PyTorch operations
                inputs_str = ", ".join(format_categories(i) for i in inputs)
                outputs_str = ", ".join(format_categories(i) for i in outputs)
                out.append(f"{name:<40} {inputs_str:<45} -> {outputs_str}")

            else:
                # Marked regions.
                # Handle marked regions
                out.append(f"\n{name}")

        return textwrap.indent("\n".join(out), " " * indent)
    def test_parameters_and_gradients(self):
        # 创建一个包含线性层和自定义的ScaleLayer的神经网络模型
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2), ScaleLayer(), torch.nn.Linear(2, 1), ScaleLayer()
        )
        # 使用随机梯度下降优化器来优化模型参数
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def fwd_only():
            # 只进行前向传播，不进行反向传播
            _ = model(torch.ones((2, 2)))

        def fwd_bwd_step():
            # 每次训练前都将梯度清零
            optimizer.zero_grad()
            # 进行前向传播
            y = model(torch.ones((2, 2)))
            # 计算损失并进行反向传播
            torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()
            # 执行优化步骤
            optimizer.step()

        # 如果我们在第一步进行性能分析，那么在调用 `model.forward` 时梯度尚未创建，
        # 所以如果我们不调用 `.backward`，则梯度将永远不会被创建。
        self._run_and_check_parameters_and_gradients(
            inner_fn=fwd_only, model=model, grads_none=True
        )

        # 在第一步，我们必须依赖 `AccumulateGrad`，因为在调用 `model.forward` 时梯度不存在。
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)

        # 经过一步之后，Python 跟踪器也会标记梯度。
        self.assertTrue(not any(p.grad is None for p in model.parameters()))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)

        # 参数梯度没有被使用，但我们仍然通过 Python 跟踪器检测到它们。
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_only, model=model)

    def test_parameters_and_gradients_set_to_none(self):
        # 创建一个包含两个线性层的神经网络模型
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        # 使用随机梯度下降优化器来优化模型参数
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def fwd_bwd_step():
            for _ in range(3):
                # 在开始时将梯度清零，使用 `set_to_none=True` 来使梯度保持活跃以便检查。
                optimizer.zero_grad(set_to_none=True)

                # 进行前向传播
                y = model(torch.ones((2, 2)))
                # 计算损失并进行反向传播
                torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()
                # 执行优化步骤
                optimizer.step()

        fwd_bwd_step()
        # 检查模型参数是否所有梯度都不为 None
        self.assertTrue(not any(p.grad is None for p in model.parameters()))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)

        # 在最后再次将梯度设置为 None
        optimizer.zero_grad(set_to_none=True)
        # 检查模型参数是否所有梯度都为 None
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)
    # 定义一个测试函数，用于测试神经网络模型正向传播时的内存分配情况
    def test_inputs_fwd(self):
        # 创建一个包含两个线性层的神经网络模型
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        # 创建包含两个 2x2 全1张量的输入列表
        inputs = [torch.ones((2, 2)) for _ in range(2)]

        # 使用内存分析器开始分析
        with profile() as prof:
            # 在分析开始前已经分配的输入
            for x in inputs:
                _ = model(x)

            # 在分析开始后继续添加的输入
            for _ in range(2):
                x = torch.ones((2, 2))
                inputs.append(x)
                _ = model(x)

        # 获取内存分析结果的内存分布概要
        memory_profile = prof._memory_profile()
        # 对每一个输入张量，查找其在内存分布中的分类信息
        for x in inputs:
            categories = self._lookup_tensor_categories(x, memory_profile)
            # 断言分类信息列表的长度大于0
            self.assertGreater(len(categories), 0)
            # 断言所有分类信息应为空（因为在正向传播时还无法分类具体的用途）
            self.assertTrue(
                all(i is None for i in categories.values()),
                categories,
            )

        # 获取当前时刻内存分布的快照
        snapshot = memory_profile._category_snapshot()
        # 断言快照中不应包含输入分类
        self.assertFalse(_memory_profiler.Category.INPUT in snapshot.values())

    # 定义另一个测试函数，用于测试懒惰版本的神经网络模型正向传播时的内存分配情况
    def test_inputs_fwd_lazy(self):
        # 创建一个包含两个懒惰线性层的神经网络模型
        model = torch.nn.Sequential(LazyLinear(2, 2), LazyLinear(2, 1))
        # 创建包含两个 2x2 全1张量的输入列表
        inputs = [torch.ones((2, 2)) for _ in range(2)]

        # 使用内存分析器开始分析
        with profile() as prof:
            # 在分析开始前已经分配的输入
            for x in inputs:
                _ = model(x)

            # 在分析开始后继续添加的输入
            for _ in range(2):
                x = torch.ones((2, 2))
                inputs.append(x)
                _ = model(x)

        # 对于现阶段，无法在没有反向传播的情况下作出任何有意义的描述。
        # 此处我们确保测试不会产生误判的分类。
        memory_profile = prof._memory_profile()
        # 对每一个输入张量，查找其在内存分布中的分类信息
        for x in inputs:
            categories = self._lookup_tensor_categories(x, memory_profile)
            # 断言分类信息列表的长度大于0
            self.assertGreater(len(categories), 0)
            # 断言所有分类信息应为空（因为在正向传播时还无法分类具体的用途）
            self.assertTrue(all(i is None for i in categories.values()), categories)

        # 获取当前时刻内存分布的快照
        snapshot = memory_profile._category_snapshot()
        # 断言快照中不应包含输入分类
        self.assertFalse(_memory_profiler.Category.INPUT in snapshot.values())
    def test_inputs_fwd_bwd(self):
        # 创建一个包含两个线性层的神经网络模型，输入维度为2，输出维度为2和1
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        # 使用随机梯度下降（SGD）作为优化器，学习率为0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # 创建包含两组输入和目标张量的列表
        inputs_targets = [(torch.ones((2, 2)), torch.rand((2, 1))) for _ in range(2)]

        def fwd_bwd_step(x, targets):
            # 前向传播：通过模型计算预测值
            y = model(x)
            # 计算均方误差损失，并执行反向传播
            torch.nn.functional.mse_loss(y, targets).backward()
            # 根据梯度更新模型参数
            optimizer.step()
            # 清空梯度，准备下一次迭代
            optimizer.zero_grad()

        with profile() as prof:
            # 在开始性能分析前已分配的输入数据
            for x, targets in inputs_targets:
                fwd_bwd_step(x, targets)

            # 在开始性能分析后动态添加的输入数据
            for _ in range(2):
                x = torch.ones((2, 2))
                targets = torch.rand((2, 1))
                inputs_targets.append((x, targets))
                fwd_bwd_step(x, targets)

        # 获取内存使用情况的性能分析结果
        memory_profile = prof._memory_profile()

        def check(t):
            # 根据内存分析结果确定张量的分类，并验证其是否为输入张量
            categories = self._lookup_tensor_categories(t, memory_profile)
            self.assertGreater(len(categories), 0)
            self.assertTrue(
                all(i == _memory_profiler.Category.INPUT for i in categories.values())
            )

        # 验证所有输入数据张量是否被正确分类
        for x, targets in inputs_targets:
            check(x)
            check(targets)

    def test_lazily_initialized(self) -> None:
        # 创建一个延迟初始化的神经网络模型
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            LazyLinear(2, 2),  # LazyLinear是一个延迟初始化的线性层
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
        )

        # 验证模型参数的数量是否正确
        self.assertEqual(len(list(model.parameters())), 4)

        def inner_fn():
            # 在内部函数中执行模型的前向传播、损失计算、反向传播及参数更新
            y = model(torch.ones((2, 2)))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()
            optimizer.step()

        # 运行并验证内部函数中的模型参数和梯度
        self._run_and_check_parameters_and_gradients(inner_fn=inner_fn, model=model)
        # 验证经过内部函数后模型参数的数量是否增加
        self.assertEqual(len(list(model.parameters())), 6)

    def test_manual_optimizer_step(self) -> None:
        # 创建一个包含两个线性层的神经网络模型
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))

        def inner_fn():
            # 在内部函数中执行模型的前向传播、损失计算、反向传播，但手动更新参数
            y = model(torch.ones((2, 2)))
            torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()

            with torch.no_grad():
                # 手动更新每个模型参数
                for p in model.parameters():
                    grad = p.grad
                    self.assertIsNotNone(grad)
                    p.add_(grad, alpha=-0.1)

        # 运行并验证内部函数中的模型参数和梯度
        self._run_and_check_parameters_and_gradients(inner_fn=inner_fn, model=model)
    def test_categories_e2e_simple_fwd(self) -> None:
        # 创建一个张量 w0，所有元素为 1，并且需要计算梯度
        w0 = torch.ones((1,), requires_grad=True)
        # 创建一个张量 w1，所有元素为 1，并且需要计算梯度
        w1 = torch.ones((1,), requires_grad=True)

        def step_fn(_):
            # 创建一个 2x2 的张量 x，所有元素为 1
            x = torch.ones((2, 2))
            # 将 x 乘以 w0 和 w1，并在第二维度上拼接成一个新的张量 y
            y = torch.cat([x * w0, x * w1], dim=1)

        # 注意：我们期望所有未知的类别。这只是一个健全性检查，确保我们没有过度标记。
        # 运行 step_fn 并格式化输出的类别信息，进行断言
        self.assertExpectedInline(
            self._run_and_format_categories(step_fn),
            """\
            aten::ones                                                                             -> 1 (???)
            aten::mul.Tensor                         1 (???), 2 (???)                              -> 3 (???)
            aten::mul.Tensor                         1 (???), 4 (???)                              -> 5 (???)
            aten::cat                                3 (???), 5 (???)                              -> ???""",
        )

    def test_categories_e2e_simple_module_fwd(self) -> None:
        # 创建一个具有输入维度为 2 和输出维度为 4 的线性模型，并包含偏置
        model = torch.nn.Linear(2, 4, bias=True)
        # 运行模型并格式化输出的类别信息，进行断言
        self.assertExpectedInline(
            self._run_and_format_categories(lambda _: model(torch.ones((2, 2)))),
            """\
            aten::ones                                                                             -> 1 (INPUT)
            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)
            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)""",
        )
    def test_categories_e2e_simple_module_fwd_bwd(self) -> None:
        # 创建一个线性模型，输入维度为2，输出维度为1，带有偏置项
        model = torch.nn.Linear(2, 1, bias=True)

        def step_fn(mark_region):
            # 标记前向传播和计算损失的区域
            mark_region("Forward & loss")
            # 对输入为全1矩阵进行前向传播，并计算损失值
            loss = model(torch.ones((2, 2))).sum()

            # 标记反向传播的区域
            mark_region("Backward")
            # 执行反向传播
            loss.backward()

        # 断言期望的内联输出结果
        self.assertExpectedInline(
            self._run_and_format_categories(step_fn),
            """\

            -- Forward & loss ---------------------------------------------------------------------------------------
            # 创建全1张量作为输入
            aten::ones                                                                             -> 1 (INPUT)
            # 对输入进行转置操作
            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)
            # 执行矩阵相乘并加上偏置，生成激活值
            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)
            # 对激活值进行求和，得到损失值
            aten::sum                                4 (ACTIVATION)                                -> 5 (ACTIVATION)

            -- Backward ---------------------------------------------------------------------------------------------
            # 根据激活值生成全1的张量
            aten::ones_like                          5 (ACTIVATION)                                -> 6 (ACTIVATION)
            # 将全1张量扩展成与激活值相同的形状
            aten::expand                             6 (ACTIVATION)                                -> 6 (ACTIVATION)
            # 对扩展后的张量进行转置操作
            aten::t                                  6 (ACTIVATION)                                -> 6 (ACTIVATION)
            # 执行矩阵相乘，计算得到梯度
            aten::mm                                 6 (ACTIVATION), 1 (INPUT)                     -> 7 (GRADIENT)
            # 对梯度进行转置操作
            aten::t                                  7 (GRADIENT)                                  -> 7 (GRADIENT)
            # 沿着指定维度对张量进行求和，得到梯度值
            aten::sum.dim_IntList                    6 (ACTIVATION)                                -> 9 (GRADIENT)
            # 调整张量的形状
            aten::view                               9 (GRADIENT)                                  -> 9 (GRADIENT)
            # 断开梯度与计算图的连接
            aten::detach                             9 (GRADIENT)                                  -> 9 (GRADIENT)
            # 断开梯度与计算图的连接
            aten::detach                             9 (GRADIENT)                                  -> ???
            # 对梯度进行转置操作
            aten::t                                  7 (GRADIENT)                                  -> 7 (GRADIENT)
            # 断开梯度与计算图的连接
            aten::detach                             7 (GRADIENT)                                  -> 7 (GRADIENT)
            # 断开梯度与计算图的连接
            aten::detach                             7 (GRADIENT)                                  -> ???""",
        )
    def test_categories_e2e_sequential_fwd(self) -> None:
        # 创建一个序列模型，包括两个线性层、ReLU激活函数和一个没有偏置的线性层，以及在第二维上进行softmax操作
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4, bias=True),  # 创建一个输入维度为2，输出维度为4的线性层，带有偏置
            torch.nn.ReLU(),                   # ReLU激活函数
            torch.nn.Linear(4, 4, bias=False), # 创建一个输入维度为4，输出维度为4的线性层，没有偏置
            torch.nn.Softmax(dim=1),           # 在第二维上进行softmax操作
        )
        # 断言内联的预期输出与运行模型并格式化后的类别匹配
        self.assertExpectedInline(
            self._run_and_format_categories(lambda _: model(torch.ones((2, 2)))),  # 运行模型并传入全为1的2x2张量作为输入
            """\
            aten::ones                                                                             -> 1 (INPUT)
            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)
            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)
            aten::relu                               4 (ACTIVATION)                                -> 5 (ACTIVATION)
            aten::detach                             5 (ACTIVATION)                                -> ???  # 分离激活函数输出
            aten::t                                  6 (PARAMETER)                                 -> 6 (PARAMETER)
            aten::mm                                 5 (ACTIVATION), 6 (PARAMETER)                 -> 7 (ACTIVATION)
            aten::_softmax                           7 (ACTIVATION)                                -> 8 (ACTIVATION)
            aten::detach                             8 (ACTIVATION)                                -> ???""",  # 分离softmax输出
        )
    def test_memory_timeline_no_id(self) -> None:
        # On CPU the default behavior is to simply forward to malloc. That
        # means that when we free `x` the allocator doesn't actually know how
        # many bytes are in the allocation, and thus there's no point to
        # calling `c10::reportMemoryUsageToProfiler`. So in order to test that
        # memory profiler processes this case correctly we need to use CUDA
        # where we do always keep a record.
        # 创建一个在 CUDA 或 CPU 上的张量 `x`，根据 GPU 是否可用选择设备
        x = torch.ones((1024,), device="cuda" if torch.cuda.is_available() else "cpu")

        with profile() as prof:
            # We never see `x` used so we don't know the storage is for a
            # Tensor, but we do still see the free event.
            # 删除张量 `x`，观察内存分配和释放事件
            del x

            # For empty we see the allocation and free, but not any use.
            # So this also cannot be identified as a Tensor.
            # 创建一个未初始化的张量 `y`，并删除它
            y = torch.empty((64,))
            del y

            # 创建一个未初始化的张量 `z`，并对其执行视图操作，将其展示给分析器
            z = torch.empty((256,))
            z.view_as(z)  # Show `z` to the profiler
            del z

        memory_profile = prof._memory_profile()

        expected = [
            # x
            (_memory_profiler.Action.PREEXISTING, 4096),  # 对于 `x`，预先存在，大小为 4096
            (_memory_profiler.Action.DESTROY, 4096),      # 销毁 `x`，释放 4096 字节
            #
            # y
            (_memory_profiler.Action.CREATE, 256),         # 创建 `y`，大小为 256
            (_memory_profiler.Action.DESTROY, 256),        # 销毁 `y`，释放 256 字节
            #
            # z
            (_memory_profiler.Action.CREATE, 1024),        # 创建 `z`，大小为 1024
            (_memory_profiler.Action.DESTROY, 1024),       # 销毁 `z`，释放 1024 字节
        ]

        actual = [(action, size) for _, action, _, size in memory_profile.timeline]

        # See above.
        # 如果 CUDA 不可用，则从预期结果中移除 `x` 的部分，并验证其余事件是否在实际结果中
        if not torch.cuda.is_available():
            expected = expected[2:]
            for event in expected:
                self.assertTrue(
                    event in actual, f"event: {event} was not found in actual."
                )
        else:
            # 否则，直接比较预期结果和实际结果
            self.assertEqual(
                actual,
                expected,
                f"expected does not match actual: {actual}",
            )
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```