# `.\pytorch\torch\testing\_internal\common_jit.py`

```py
# mypy: ignore-errors

# 导入 Torch 库及其子模块
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized

# 导入测试工具
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import TestCase, \
    freeze_rng_state, TemporaryFileName, enable_profiling_mode_for_profiling_tests, is_iterable_of_tensors
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401

# 导入标准库
from itertools import chain
from typing import List, Union
from torch._C import TensorType

import io

# 定义测试类方法：检查函数输出的类型是否正确
def check_output_types(self, func, ref_outputs, args, kwargs):
    graph = getattr(func, 'last_graph', None)  # 获取函数的最后一个图形表示
    types = [o.type() for o in graph.outputs()]  # 获取图形输出的类型列表
    self.assertTrue(len(types) == 1)  # 断言图形输出的类型列表长度为1
    t = types[0]  # 获取第一个类型
    torch._C._jit_assert_is_instance(ref_outputs, t)  # 使用 JIT 断言函数检查引用输出与类型是否一致

# 用于只对单个导数进行检查的测试名称集合
nn_functional_single_grad = frozenset('test_nn_' + name for name in [
    'pdist',
    'multilabel_margin_loss',
    'max_unpool3d',
    'multi_margin_loss',
    'binary_cross_entropy',
    'binary_cross_entropy_size_average',
    'ctc_loss',
    'grid_sample',
])

# 检查函数与参考实现的输出是否相符的方法
def check_against_reference(self, func, reference_func, output_func, args, kwargs=None,
                            allow_unused=True, check_types=True, no_grad=False, no_gradgrad=False):
    """Verifies a function performs identically to some reference implementation.

    Commonly, this is used to verify that a JIT implementation
    (output_func) matches the behavior of the eager implementation
    (reference_func).
    """
    kwargs = kwargs if kwargs else {}

    # 计算所有张量的和
    def allSum(vs):
        if isinstance(vs, torch.Tensor):
            vs = (vs,)
        return sum((i + 1) * v.sum().abs() if v.dtype.is_complex else (i + 1) * v.sum()
                   for i, v in enumerate(vs)
                   if v is not None and v.dtype in floating_and_complex_types_and(torch.half, torch.bfloat16))

    # 克隆张量并保留 requires_grad 属性
    def clone_tensor(t, preserve_requires_grad):
        require_grad = preserve_requires_grad and t.requires_grad
        return t.detach().clone().requires_grad_(require_grad)

    # 克隆输入张量及其列表
    def clone_inputs(preserve_requires_grad: bool):
        inputs: List[Union[torch.Tensor, List[torch.Tensor]]] = []

        for arg in args:
            if isinstance(arg, torch.Tensor):
                inputs.append(clone_tensor(arg, preserve_requires_grad))
            elif is_iterable_of_tensors(arg):
                inputs.append([clone_tensor(t, preserve_requires_grad) for t in arg])
            else:
                inputs.append(arg)

        return inputs

    # 返回需要梯度的输入张量的列表，包括 TensorList 中的张量
    # 定义一个函数，用于获取输入参数中所有需要记录梯度的张量
    def get_recording_tensors(args):
        # 初始化一个空列表，用于存储记录梯度的张量
        recording_tensors: List[torch.Tensor] = []

        # 遍历输入参数列表
        for arg in args:
            # 如果参数是 torch.Tensor 类型且需要梯度记录，则将其加入列表
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                recording_tensors.append(arg)
            # 如果参数是张量类型的可迭代对象，则筛选出需要梯度记录的张量加入列表
            elif is_iterable_of_tensors(arg):
                recording_tensors.extend(filter(lambda t: t.requires_grad, arg))

        # 返回记录梯度的张量列表
        return recording_tensors

    # 在没有梯度的情况下进行测试
    nograd_inputs = clone_inputs(preserve_requires_grad=False)
    # 使用参考函数运行并保存随机数生成器状态，返回结果
    outputs = self.runAndSaveRNG(reference_func, nograd_inputs, kwargs)
    # 在性能测试中启用分析模式，并运行函数得到结果
    with enable_profiling_mode_for_profiling_tests():
        outputs_test = self.runAndSaveRNG(func, nograd_inputs, kwargs)
    # 断言两次运行的结果应该相等
    self.assertEqual(outputs, outputs_test)

    # 如果需要检查输出类型，则调用检查函数进行检查
    if check_types:
        check_output_types(self, func, outputs_test, nograd_inputs, kwargs)

    # 如果设置了 no_grad 参数，则跳过梯度相关的测试
    if no_grad:
        # 跳过梯度测试
        return

    # 在性能测试中启用分析模式，并进行单一梯度测试
    with enable_profiling_mode_for_profiling_tests():
        # 复制输入并保留需要梯度的设置
        recording_inputs = clone_inputs(preserve_requires_grad=True)
        # 获取需要记录梯度的张量
        recording_tensors = get_recording_tensors(recording_inputs)
        # 使用参考函数运行并保存随机数生成器状态，返回结果，并应用输出函数
        outputs = output_func(self.runAndSaveRNG(reference_func, recording_inputs, kwargs))
        # 计算输出结果的梯度
        grads = torch.autograd.grad(allSum(outputs), recording_tensors,
                                    allow_unused=allow_unused)
        # 使用函数运行结果作为输入，再次运行函数，返回结果
        outputs_test = output_func(self.runAndSaveRNG(func, recording_inputs, kwargs))
        # 计算新结果的梯度
        grads_test = torch.autograd.grad(allSum(outputs_test), recording_tensors,
                                         allow_unused=allow_unused)
        # 断言两次运行的结果和梯度应该相等
        self.assertEqual(outputs, outputs_test)
        self.assertEqual(grads, grads_test)

        # 如果当前测试方法在单一梯度测试列表中，或者设置了 no_gradgrad 参数，则直接返回
        if self._testMethodName in nn_functional_single_grad or no_gradgrad:
            return

        # 使用参考函数运行并保存随机数生成器状态，返回结果，并应用输出函数
        outputs = output_func(self.runAndSaveRNG(reference_func, recording_inputs, kwargs))
        # 计算输出结果的总和
        l1 = allSum(outputs)
        # 计算输出结果的梯度，并创建计算图
        grads = torch.autograd.grad(l1, recording_tensors, create_graph=True,
                                    allow_unused=allow_unused)

        # 计算梯度的二次导数
        l2 = (allSum(grads) * l1)
        # 计算二次导数的梯度
        grads2 = torch.autograd.grad(l2, recording_tensors, allow_unused=allow_unused)
        # 复制输入并保留需要梯度的设置
        recording_inputs = clone_inputs(preserve_requires_grad=True)
        # 获取需要记录梯度的张量
        recording_tensors = get_recording_tensors(recording_inputs)
        # 使用函数运行结果作为输入，再次运行函数，返回结果
        outputs_test = output_func(self.runAndSaveRNG(func, recording_inputs, kwargs))
        # 计算新结果的总和
        l1_test = allSum(outputs_test)
        # 计算新结果的梯度，并创建计算图
        grads_test = torch.autograd.grad(
            l1_test, recording_tensors, create_graph=True, allow_unused=allow_unused)

        # 计算新结果的二次导数
        l2_test = (allSum(grads_test) * l1_test)
        # 计算新结果的二次导数的梯度
        grads2_test = torch.autograd.grad(l2_test, recording_tensors, allow_unused=allow_unused)

        # 断言两次运行的结果和梯度应该相等
        self.assertEqual(outputs, outputs_test)
        self.assertEqual(grads, grads_test)
        # 比较每一对二次导数的梯度，如果有一个为 None，则继续比较下一对
        for g2, g2_test in zip(grads2, grads2_test):
            if g2 is None and g2_test is None:
                continue
            # 断言二次导数的梯度应该相等，设置误差范围为绝对误差 5e-4 和相对误差 1e-4
            self.assertEqual(g2, g2_test, atol=5e-4, rtol=1e-4)
# 定义一个名为 JitCommonTestCase 的测试用例类，继承自 TestCase 类
class JitCommonTestCase(TestCase):

    # 创建一个函数，从给定的跟踪对象（trace）中获取计算图并创建函数对象
    def createFunctionFromGraph(self, trace):
        # 如果 trace 是 torch._C.Graph 类型，则直接使用，否则获取其计算图
        graph = trace if isinstance(trace, torch._C.Graph) else trace.graph()
        # 调用底层 C++ 接口，根据计算图创建函数对象，并命名为 "forward"
        return torch._C._create_function_from_graph("forward", graph)

    # 断言导出和导入过程，验证模型的序列化和反序列化
    def assertExportImport(self, trace, inputs):
        # 根据跟踪对象创建函数对象
        m = self.createFunctionFromGraph(trace)
        # 调用 assertExportImportModule 方法，验证模型的导出和导入
        self.assertExportImportModule(m, inputs)

    # 断言导出和导入模块，验证模型在导出和导入后的一致性
    def assertExportImportModule(self, m, inputs):
        # 获取模型的导入副本
        m_import = self.getExportImportCopy(m)
        # 分别运行模型并保存随机数生成器的状态，返回结果
        a = self.runAndSaveRNG(m, inputs)
        b = self.runAndSaveRNG(m_import, inputs)
        # 断言两个模型运行的结果应该一致
        self.assertEqual(a, b, "Results of original model and "
                               "exported/imported version of model differed")

    # 运行模型并保存随机数生成器的状态
    def runAndSaveRNG(self, func, inputs, kwargs=None):
        # 如果 kwargs 为 None，则设为一个空字典
        kwargs = kwargs if kwargs else {}
        # 冻结随机数生成器的状态，运行模型并获取结果
        with freeze_rng_state():
            results = func(*inputs, **kwargs)
        return results

    # 获取模型的导出和导入副本
    def getExportImportCopy(self, m, also_test_file=True, map_location=None):
        # 创建一个字节流对象
        buffer = io.BytesIO()
        # 将模型 m 序列化保存到字节流中
        torch.jit.save(m, buffer)
        buffer.seek(0)
        # 从字节流中加载模型，得到导入的模型对象
        imported = torch.jit.load(buffer, map_location=map_location)

        # 如果不需要测试文件，则直接返回导入的模型对象
        if not also_test_file:
            return imported

        # 使用临时文件保存导入的模型，并从文件中加载返回
        with TemporaryFileName() as fname:
            torch.jit.save(imported, fname)
            return torch.jit.load(fname, map_location=map_location)
    # 在给定的图中断言自动微分节点是否存在，检查非可融合节点和可融合节点的情况

    # 找到所有包含 'prim::DifferentiableGraph' 的节点
    diff_nodes = graph.findAllNodes('prim::DifferentiableGraph')
    # 获取这些节点的子图列表
    diff_subgraphs = [node.g('Subgraph') for node in diff_nodes]

    # 注意：目前没有测试包含可融合节点的情况
    # 查找所有 'prim::FusionGroup' 节点并展开为列表
    fusion_nodes = list(chain.from_iterable([g.findAllNodes('prim::FusionGroup') for g in diff_subgraphs]))
    # 获取这些融合节点的子图列表
    fusion_subgraphs = [node.g('Subgraph') for node in fusion_nodes]

    # 对于任何非可融合节点，它必须出现在至少一个 DifferentiableGraph 的子图中
    nodes_in_diff_graph = []
    nodes_not_in_diff_graph = []
    non_fusible_nodes_being_fused = []
    for node in nonfusible_nodes:
        if any(g.findNode(node) is not None for g in diff_subgraphs):
            nodes_in_diff_graph.append(node)
        else:
            nodes_not_in_diff_graph.append(node)
        if any(g.findNode(node) is not None for g in fusion_subgraphs):
            non_fusible_nodes_being_fused.append(node)
    
    # 检查是否找到了所有非可融合节点
    found_all_nonfusible_nodes = len(nodes_in_diff_graph) == len(nonfusible_nodes)

    # 对于任何可融合节点，它必须出现在至少一个 DifferentiableGraph 的 FusionGroup 中
    fusion_nodes_found = []
    fusion_nodes_not_found = []
    for node in fusible_nodes:
        if any(g.findNode(node) is not None for g in fusion_subgraphs):
            fusion_nodes_found.append(node)
        else:
            fusion_nodes_not_found.append(node)
    
    # 检查是否找到了所有可融合节点
    found_all_fusible_nodes = len(fusion_nodes_found) == len(fusible_nodes)

    # 如果应该存在自动微分节点，则进行断言检查
    if should_autodiff_node is not None:
        # 生成自动微分错误消息，包括相关节点的详细信息
        err_msg = self.autoDiffErrorMessage(should_autodiff_node,
                                            nodes_not_in_diff_graph,
                                            fusion_nodes_not_found,
                                            non_fusible_nodes_being_fused,
                                            fusion_nodes_found,
                                            nodes_in_diff_graph)
        # 使用断言检查是否找到了所有非可融合节点和可融合节点
        self.assertEqual(should_autodiff_node,
                         found_all_nonfusible_nodes and found_all_fusible_nodes, err_msg)
    # 定义一个方法，用于检查形状分析的结果
    def checkShapeAnalysis(self, out_sizes: Union[List[int], List[List[int]]],
                           traced_graph, assert_propagation, constant_prop=True):
        # 保存之前的符号化形状测试模式状态
        prev_symbolic_shapes_test_enabled = torch._C._jit_symbolic_shapes_test_mode_enabled()
        # 遍历两种测试模式：开启和关闭
        for enable_test_mode in [True, False]:
            # 设置当前的符号化形状测试模式
            torch._C._jit_set_symbolic_shapes_test_mode(enable_test_mode)
            # 擦除跟踪图中非输入形状信息
            torch._C._jit_erase_non_input_shape_information(traced_graph)
            # 如果启用常量传播，则进行常量传播优化
            if constant_prop:
                torch._C._jit_pass_constant_propagation(traced_graph)
            # 在图上执行形状传播
            torch._C._jit_pass_propagate_shapes_on_graph(traced_graph)
            # 获取跟踪图的输出类型
            output = next(traced_graph.outputs()).type()

            # 定义一个函数，用于测试类型和实际大小
            def test_type(type, actual_size):
                # 获取符号化大小
                sizes = type.symbolic_sizes()
                # 创建具有符号化大小的输出类型和实际类型
                out_type = TensorType.get().with_sizes(sizes)
                actual_type = TensorType.get().with_sizes(actual_size)

                # 断言实际类型是输出类型的子类型
                self.assertTrue(actual_type.isSubtypeOf(out_type))

                # 如果断言标志为真，则检查形状分析是否成功
                if assert_propagation:
                    self.assertEqual(out_type.sizes(), actual_size)

            # 如果输出类型是 torch._C.TensorType 的子类型
            if output.isSubtypeOf(torch._C.TensorType.get()):
                test_type(output, out_sizes)
            else:
                # 否则，获取元组元素并逐个测试
                tuple_elements = output.elements()
                for i in range(len(tuple_elements)):
                    test_type(tuple_elements[i], out_sizes[i])

        # 恢复之前的符号化形状测试模式状态
        torch._C._jit_set_symbolic_shapes_test_mode(prev_symbolic_shapes_test_enabled)
```