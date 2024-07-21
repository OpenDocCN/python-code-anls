# `.\pytorch\torch\testing\_internal\jit_utils.py`

```
# 忽略类型检查错误，这个是为了特定的检查工具
# Torch
from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401  # 禁止导入警告

from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
import zipfile
import functools

# Testing utils
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, \
    freeze_rng_state, enable_profiling_mode_for_profiling_tests, ProfilingMode, TEST_BAILOUTS, \
    is_iterable_of_tensors
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401  # 禁止导入警告

# Standard library
from contextlib import contextmanager
from functools import reduce
from io import StringIO
from collections import defaultdict

import importlib.util
import inspect
import io
import math
import os
import pickle
import sys
import tempfile
import textwrap
from importlib.abc import Loader
from typing import Any, Dict, List, Tuple, Union

# 检查是否支持 CUDA 并设置相关标志
RUN_CUDA = torch.cuda.is_available()
RUN_CUDA_MULTI_GPU = RUN_CUDA and torch.cuda.device_count() > 1
RUN_CUDA_HALF = RUN_CUDA

# 如果支持 CUDA 且不是 HIP 环境，则获取 CUDA 版本并检查是否支持半精度运算
if torch.cuda.is_available() and not torch.version.hip:
    CUDA_VERSION = torch._C._cuda_getCompiledVersion()
    for d in range(torch.cuda.device_count()):
        major = torch.cuda.get_device_capability(d)[0]
        if major < 6:
            RUN_CUDA_HALF = False

# 执行代码包装函数，用于在给定的全局和局部命名空间中执行代码
def execWrapper(code, glob, loc):
    exec(code, glob, loc)

# 对输入应用函数 fn，并且仅对 torch.Tensor 类型进行映射
def do_input_map(fn, input):
    return _nested_map(lambda t: isinstance(t, torch.Tensor), fn)(input)

# 清除 JIT 类注册表和相关状态
def clear_class_registry():
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

# 获取执行计划，假设只有一个执行计划，否则抛出运行时错误
def get_execution_plan(graph_executor_state):
    execution_plans = list(graph_executor_state.execution_plans.values())
    num_plans = len(execution_plans)
    if num_plans != 1:
        raise RuntimeError('This test assumes this GraphExecutor should '
                           f'only have one execution plan, got: {num_plans}')
    return execution_plans[0]

# 用于上下文管理的断言正则表达式，用于检查错误消息是否正确地突出显示源代码的部分
class _AssertRaisesRegexWithHighlightContext:
    """
    A context manager that is useful for checking that error messages highlight
    the correct part of the source code.
    """

    def __init__(self, test_case, exception, regex, highlight):
        self.test_case = test_case
        self.exception_type = exception
        self.regex = regex
        self.highlight = highlight

    def __enter__(self):
        return self
    # 定义上下文管理器的退出方法，用于处理资源释放或异常处理
    def __exit__(self, type, value, traceback):
        # 使用断言验证是否抛出了指定类型和匹配正则表达式的异常
        with self.test_case.assertRaisesRegex(self.exception_type, self.regex):
            if type:
                raise value  # 如果有异常类型，重新引发异常

        # 如果需要进行语法高亮检查
        if self.highlight:
            # 创建 FileCheck 对象，检查源代码是否正确高亮
            FileCheck().check_source_highlighted(self.highlight).run(str(value))

        # 返回 True 表示成功处理异常
        return True
# 定义一个常量，表示融合组的名称
FUSION_GROUP = "prim::TensorExprGroup"

# 定义一个测试类 JitTestCase，继承自 JitCommonTestCase
class JitTestCase(JitCommonTestCase):
    # 开启对 CUDA 内存泄漏的检查
    _do_cuda_memory_leak_check = True
    # 是否已经恢复警告状态的标志
    _restored_warnings = False

    # 内部类 capture_stdout，继承自 list
    class capture_stdout(list):
        """
        用临时的 StringIO 替换 sys.stdout
        """
        def __enter__(self):
            # 保存当前的 sys.stdout 和创建一个 StringIO 对象
            self.sys_stdout = sys.stdout
            self.stringio = StringIO()
            sys.stdout = self.stringio
            return self

        def __exit__(self, *args):
            # 将 StringIO 对象的内容转换为字符串并存储在列表中，恢复 sys.stdout
            self.append(str(self.stringio.getvalue()))
            del self.stringio
            sys.stdout = self.sys_stdout

    # 内部类 capture_stderr，继承自 list
    class capture_stderr(list):
        """
        用临时的 StringIO 替换 sys.stderr
        """
        def __enter__(self):
            # 保存当前的 sys.stderr 和创建一个 StringIO 对象
            self.sys_stderr = sys.stderr
            self.stringio = StringIO()
            sys.stderr = self.stringio
            return self

        def __exit__(self, *args):
            # 将 StringIO 对象的内容转换为字符串并存储在列表中，恢复 sys.stderr
            self.append(str(self.stringio.getvalue()))
            del self.stringio
            sys.stderr = self.sys_stderr

    # 设置自定义的 JIT hooks
    def setHooks(self):
        torch._C._jit_set_emit_hooks(self.emitModuleHook, self.emitFunctionHook)

    # 清除 JIT hooks
    def clearHooks(self):
        torch._C._jit_set_emit_hooks(None, None)

    # 测试初始化方法
    def setUp(self):
        super().setUp()
        # 在安装自己的警告过滤器前，确保 unittest 不会覆盖所有警告过滤器并强制显示所有警告
        if not JitTestCase._restored_warnings:
            torch.jit.TracerWarning.ignore_lib_warnings()
            JitTestCase._restored_warnings = True
        # 设置 JIT hooks
        self.setHooks()

    # 测试清理方法
    def tearDown(self):
        super().tearDown()
        # 需要清除 JIT hooks，因为在回调被销毁前可能会卸载 Python
        self.clearHooks()
        # 清除类注册表中的内容
        clear_class_registry()
    def assertAllFused(self, graph, except_for=()):

        # note this helper collects nodes on 'fast path' only
        # i.e. the true blocks of specialized checks
        # 这个辅助函数仅收集“快速路径”上的节点
        # 即专门检查的真实块

        def get_nodes_and_parents_recursively(block, kind, acc):
            # Recursively collect nodes of a specific kind within a block and its subblocks
            # 在一个块及其子块中递归收集特定类型的节点
            for node in block.nodes():
                if node.kind() == kind:
                    acc[block].append(node)
                elif node.kind() == 'prim::DifferentiableGraph':
                    get_nodes_and_parents_recursively(node.g('Subgraph'), kind, acc)
                elif node.kind() == 'prim::If' and (node.inputs().__next__().node().kind() == 'aten::all' or
                                                    node.inputs().__next__().node().kind() == 'prim::TypeCheck' or
                                                    node.inputs().__next__().node().kind() == 'prim::RequiresGradCheck'):
                    get_nodes_and_parents_recursively(node.blocks().__next__(), kind, acc)
                else:
                    for inner_block in node.blocks():
                        get_nodes_and_parents_recursively(inner_block, kind, acc)
            # 在块的节点中进行递归收集，并将结果存储在给定的累加器中

        allowed_nodes = {'prim::Constant', FUSION_GROUP, 'prim::BailoutTemplate',
                         'prim::TupleConstruct', 'prim::If', 'prim::TypeCheck', 'prim::RequiresGradCheck'} | set(except_for)

        # fusion_groups is a dictionary mapping blocks to lists of fusion nodes
        # fusion_groups 是一个字典，将块映射到融合节点列表
        fusion_groups: Dict[torch._C.Block, List[torch._C.Node]] = defaultdict(list)
        get_nodes_and_parents_recursively(graph, FUSION_GROUP, fusion_groups)
        # Ensure there is exactly one block containing fusion nodes
        # 确保有且仅有一个包含融合节点的块
        self.assertTrue(len(fusion_groups) == 1, f'got {graph}')
        (graph, fusion_nodes) = next(iter(fusion_groups.items()))
        # Ensure there is exactly one fusion node within the identified block
        # 确保在确定的块内有且仅有一个融合节点
        self.assertTrue(len(fusion_nodes) == 1, f'got {graph}')
        # Ensure all nodes in the graph are among the allowed node types
        # 确保图中所有节点都属于允许的节点类型
        self.assertTrue(all(node.kind() in allowed_nodes for node in graph.nodes()),
                        f'got {graph}')

    def _isHookExceptionOk(self, e):
        se = str(e)
        allowed = ("Could not export Python function",
                   "closures are not exportable")
        for a in allowed:
            if a in se:
                return True
        return False
    def _compared_saved_loaded(self, m):
        def extract_files(buffer):
            # 使用 zipfile 模块打开给定的缓冲区，解析 ZIP 格式以获取主模块代码
            archive = zipfile.ZipFile(buffer)
            # 检查文件名列表是否有重复项
            self.assertEqual(len(set(archive.namelist())), len(archive.namelist()))
            # 过滤出以 'archive/code/' 开头的文件名列表
            files = list(filter(lambda x: x.startswith('archive/code/'), archive.namelist()))
            # 筛选出以 '.py' 结尾的代码文件名列表
            code_files_str = filter(lambda x: x.endswith('.py'), files)
            # 使用生成器打开代码文件，将其内容解码为字符串
            code_files_stream = (archive.open(f) for f in code_files_str)
            code_files = ("".join([line.decode() for line in file]) for file in code_files_stream)

            # 筛选出以 '.debug_pkl' 结尾的调试文件名列表
            debug_files_str = filter(lambda f: f.endswith('.debug_pkl'), files)
            # 使用生成器打开调试文件，使用 pickle 加载文件内容
            debug_files_stream = (archive.open(f) for f in debug_files_str)
            debug_files = (pickle.load(f) for f in debug_files_stream)
            return code_files, debug_files

        # 在解析代码时禁用钩子，以免重新进入钩子
        with torch._jit_internal._disable_emit_hooks():
            try:
                # 如果模块代码为空，则立即返回
                if len(m.code) == 0:
                    return
                # 如果模块是 torch._C.ScriptModule 类型且没有方法名，则立即返回
                if isinstance(m, torch._C.ScriptModule):
                    if len(m._method_names()) == 0:
                        return

                # 将模块保存到缓冲区
                buffer = io.BytesIO()
                torch.jit.save(m, buffer)
                # 复制缓冲区数据，以便稍后恢复。因为 py2 和 py3 在 zipfile 语义上有差异，
                # 所以最好每次使用一个新的副本来处理。
                buffer_copy = buffer.getvalue()

                # 提取缓冲区中的文件内容
                code_files, debug_files = extract_files(buffer)

            except RuntimeError as e:
                # 如果遇到运行时错误并且不是预期的钩子异常，则抛出异常
                if not self._isHookExceptionOk(e):
                    raise
                else:
                    return

            # 再次导入模型（使用原始模型的副本）
            buffer2 = io.BytesIO(buffer_copy)
            imported = torch.jit.load(buffer2)

            # 再次保存模型
            saved_module_buffer_2 = io.BytesIO()
            torch.jit.save(imported, saved_module_buffer_2)

            saved_module_buffer_2.seek(0)
            # 提取保存后的模型的文件内容
            code_files_2, debug_files_2 = extract_files(saved_module_buffer_2)

            # 比较两个模型的代码文件内容是否一致
            for a, b in zip(code_files, code_files_2):
                self.assertMultiLineEqual(a, b)

            # 如果模块是 torch._C.ScriptModule 类型，则检查其 IValue 标签是否匹配
            if isinstance(m, torch._C.ScriptModule):
                self.assertTrue(torch._C._ivalue_tags_match(m, imported._c))
    # 发射函数钩子，用于处理函数对象
    def emitFunctionHook(self, func):
        # 如果函数名为 "<lambda>" 或者包含 "aten::" 字符串，则跳过检查
        if func.name == "<lambda>" or "aten::" in func.name:
            return
        # 对传入的函数对象进行保存和加载比较
        self._compared_saved_loaded(func)

    # 发射模块钩子，用于处理模块对象
    def emitModuleHook(self, module):
        # 对传入的模块对象进行保存和加载比较
        self._compared_saved_loaded(module)


    # 获取带有打包功能的导出与导入副本
    def getExportImportCopyWithPacking(self, m, also_test_file=True, map_location=None):
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 应用模块中所有支持打包操作的方法
        m.apply(lambda s: s._pack() if s._c._has_method('_pack') else None)
        # 将模型 m 保存到字节流缓冲区中
        torch.jit.save(m, buffer)
        # 对模块 m 应用所有支持解包操作的方法
        m.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)
        buffer.seek(0)
        # 从字节流缓冲区中加载模型，指定映射位置
        imported = torch.jit.load(buffer, map_location=map_location)
        # 对加载后的模型应用所有支持解包操作的方法
        imported.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)

        # 如果不需要同时测试文件，则直接返回加载后的模型
        if not also_test_file:
            return imported

        # 在理想情况下，我们希望不必手动删除文件，但是 NamedTemporaryFile 在 Windows 下会打开文件，
        # 并且文件不能在 Windows 中多次打开。为了支持 Windows，创建文件后立即关闭并手动删除
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.close()
            # 将导入的模型保存到临时文件中
            imported.save(f.name)
            # 从临时文件加载模型，指定映射位置
            result = torch.jit.load(f.name, map_location=map_location)
        finally:
            # 最终无论如何删除临时文件
            os.unlink(f.name)

        # 对加载后的结果模型应用所有支持解包操作的方法
        result.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)
        return result

    # 断言图中包含指定类型的节点
    def assertGraphContains(self, graph, kind, consider_subgraphs=False):

        # 如果考虑子图，则在图的字符串表示中统计指定类型节点的数量
        if consider_subgraphs:
            strgraph = str(graph)
            count = strgraph.count(kind) - strgraph.count(f'with {kind}')
            self.assertTrue(count > 0)
            return

        # 定义用于遍历图中节点的函数
        def nodes(block):
            out = []
            for node in block.nodes():
                # 如果节点类型与指定类型相符，则加入输出列表
                if node.kind() == kind:
                    out.append(node)
                # 递归处理节点的子块
                for block in node.blocks():
                    out += nodes(block)
            return out

        # 获取图中所有指定类型的节点
        out_nodes = nodes(graph)
        # 断言至少存在一个指定类型的节点
        self.assertTrue(len(out_nodes) > 0)
    # 定义一个方法来断言图包含特定类型和数量的节点
    def assertGraphContainsExactly(self, graph, kind, num_kind_nodes, consider_subgraphs=False):
        # 定义内部函数来执行断言操作
        def perform_assert(graph, kind, actual, expected, consider_subgraphs):
            # 如果实际节点数与期望节点数相等，则直接返回
            if actual == expected:
                return
            # 根据 consider_subgraphs 参数选择子图包含还是不包含的文本描述
            subgraph = 'including' if consider_subgraphs else 'excluding'
            # 抛出断言错误，显示当前图中的节点数与期望节点数不符
            raise AssertionError(
                f'{graph}\nError: graph contains {actual} {kind} nodes ({subgraph} subgraphs) but expected {expected}')

        # 如果考虑子图，则将图转换为字符串，计算特定类型节点的数量（排除特定类型的节点）
        if consider_subgraphs:
            strgraph = str(graph)
            count = strgraph.count(kind) - strgraph.count(f'with {kind}')
            # 执行断言
            perform_assert(graph, kind, count, num_kind_nodes,
                           consider_subgraphs)
            return

        # 定义递归函数来获取所有符合特定类型的节点
        def nodes(block):
            out = []
            for node in block.nodes():
                # 如果节点类型与指定类型相符，则加入输出列表
                if node.kind() == kind:
                    out.append(node)
                # 递归调用处理节点的子块
                for block in node.blocks():
                    out += nodes(block)
            return out

        # 获取所有符合特定类型的节点
        out_nodes = nodes(graph)
        # 执行断言
        perform_assert(graph, kind, len(out_nodes), num_kind_nodes,
                       consider_subgraphs)

    # 断言经过优化的 ONNX 图符合预期
    def assertExpectedONNXGraph(self, g, *args, **kwargs):
        g = torch.onnx._optimize_trace(g, operator_export_type=OperatorExportTypes.ONNX)
        self.assertExpectedGraph(g, *args, **kwargs)

    # 断言预期的图与给定的图匹配
    def assertExpectedGraph(self, trace, *args, **kwargs):
        # 如果 trace 是 torch._C.Graph 类型，则直接使用，否则获取其图形表示
        if isinstance(trace, torch._C.Graph):
            graph = trace
        else:
            graph = trace.graph()

        # 对图进行 lint 检查、死代码消除和规范化处理
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        # 调用 assertExpected 方法来进行最终断言
        self.assertExpected(str(graph), *args, **kwargs)

    # 运行指定名称的 JIT 传递（优化、转换）并返回处理后的图
    def run_pass(self, name, trace):
        # 如果 trace 是 torch._C.Graph 类型，则直接使用，否则获取其图形表示
        if isinstance(trace, torch._C.Graph):
            graph = trace
            set_graph = False
        else:
            set_graph = True
            graph = trace.graph()

        # 对图进行 lint 检查，运行指定的 JIT 传递，并根据返回结果更新图
        torch._C._jit_pass_lint(graph)
        result = getattr(torch._C, '_jit_pass_' + name)(graph)
        if result is not None and not isinstance(result, bool):
            graph = result
        torch._C._jit_pass_lint(graph)

        # 如果需要重新设置 trace 的图形表示，则更新
        if set_graph:
            trace.set_graph(graph)
        # 返回处理后的图
        return graph

    # 获取调用栈指定层级的帧变量（局部和全局变量）
    def get_frame_vars(self, frames_up):
        # 获取当前帧对象
        frame = inspect.currentframe()
        if not frame:
            raise RuntimeError("failed to inspect frame")
        i = 0
        # 循环向上遍历帧对象，直到达到指定的 frames_up 层级
        while i < frames_up + 1:
            frame = frame.f_back
            if not frame:
                raise RuntimeError("failed to get frame")
            i += 1
        # 创建空字典来存储定义的局部和全局变量
        defined_vars: Dict[str, Any] = {}
        # 更新字典，将局部变量和全局变量添加到字典中
        defined_vars.update(frame.f_locals)
        defined_vars.update(frame.f_globals)
        # 返回包含所有帧变量的字典
        return defined_vars

    # 返回一个 _AssertRaisesRegexWithHighlightContext 上下文管理器，用于检查异常和正则表达式
    def assertRaisesRegexWithHighlight(self, exception, regex, highlight):
        return _AssertRaisesRegexWithHighlightContext(self, exception, regex, highlight)
    def checkScriptRaisesRegex(self, script, inputs, exception, regex,
                               name=None, outputs=None, capture_output=False,
                               frames_up=1, profiling=ProfilingMode.PROFILING):
        """
        Checks that a given function will throw the correct exception,
        when executed with normal python, the string frontend, and the
        AST frontend. Logic taken from `checkScript` (see comments there
        for details)
        """
        # 使用上下文管理器，启用性能分析模式以进行性能分析测试
        with enable_profiling_mode_for_profiling_tests():
            # 在普通 Python 中
            with self.assertRaisesRegex(exception, regex):
                if isinstance(script, str):
                    # 获取当前调用栈上的变量框架
                    frame = self.get_frame_vars(frames_up)
                    the_locals: Dict[str, Any] = {}
                    # 执行包装函数，将全局变量设置为 frame，局部变量设置为 the_locals
                    execWrapper(script, glob=frame, loc=the_locals)
                    # 更新 frame 中的局部变量
                    frame.update(the_locals)
                    # 获取 Python 函数对象
                    python_fn = frame[name]
                else:
                    python_fn = script

                # 调用 Python 函数，并传入 inputs
                python_fn(*inputs)

            # 在字符串前端
            with self.assertRaisesRegex(exception, regex):
                if isinstance(script, str):
                    # 使用 torch.jit.CompilationUnit 将脚本编译为计算图单元
                    cu = torch.jit.CompilationUnit(script, _frames_up=frames_up)
                    # 获取脚本中的函数对象
                    string_frontend = getattr(cu, name)
                else:
                    # 获取函数对象的源代码，并使用 torch.jit.CompilationUnit 编译为计算图单元
                    source = textwrap.dedent(inspect.getsource(script))
                    cu = torch.jit.CompilationUnit(source, _frames_up=frames_up)
                    string_frontend = getattr(cu, script.__name__)

                # 调用字符串前端函数，并传入 inputs
                string_frontend(*inputs)

            # 在 Python AST 前端
            if not isinstance(script, str):
                with self.assertRaisesRegex(exception, regex):
                    # 使用 torch.jit.script 将 Python 函数编译为脚本
                    ge = torch.jit.script(python_fn)
                    # 调用脚本化的函数，并传入 inputs
                    ge(*inputs)

    def checkBailouts(self, model, inputs, expected):
        # 获取模型的调试状态
        state = model.get_debug_state()
        # 获取执行计划
        plan = get_execution_plan(state)
        # 获取需要执行的 bailout 数量
        num_bailouts = plan.code.num_bailouts()
        # 遍历所有 bailout
        for i in range(0, num_bailouts):
            # 请求执行特定编号的 bailout
            plan.code.request_bailout(i)
            # 执行模型，并获取输出结果
            bailout_outputs = model(*inputs)
            # 断言 bailout 后的输出结果与预期值相等
            self.assertEqual(bailout_outputs, expected)

    def checkModule(self, nn_module, args):
        """
        Check that a nn.Module's results in Script mode match eager and that it
        can be exported
        """
        # 使用 torch.jit.script 将 nn.Module 编译为脚本模式
        sm = torch.jit.script(nn_module)

        # 冻结随机数生成器状态，确保结果的可重现性
        with freeze_rng_state():
            # 在 eager 模式下执行 nn.Module，并获取输出结果
            eager_out = nn_module(*args)

        # 冻结随机数生成器状态，确保结果的可重现性
        with freeze_rng_state():
            # 在脚本模式下执行 nn.Module，并获取输出结果
            script_out = sm(*args)

        # 断言 eager 模式下的输出结果与脚本模式下的输出结果相等
        self.assertEqual(eager_out, script_out)
        # 断言能够成功导出并重新导入模块
        self.assertExportImportModule(sm, args)

        # 返回编译后的脚本模块对象
        return sm
# 定义一个上下文管理器类 NoTracerWarnContextManager，用于禁用 Torch 的追踪器警告
class NoTracerWarnContextManager:
    def __enter__(self):
        # 获取当前追踪器状态并保存到 prev 变量中
        self.prev = torch._C._jit_get_tracer_state_warn()
        # 设置追踪器状态为 False，禁用追踪器警告
        torch._C._jit_set_tracer_state_warn(False)

    def __exit__(self, *args):
        # 恢复之前保存的追踪器状态
        torch._C._jit_set_tracer_state_warn(self.prev)

# 定义一个上下文管理器函数 inline_everything_mode，用于控制 Torch 的内联模式
@contextmanager
def inline_everything_mode(should_inline):
    # 获取当前内联模式并保存到 old 变量中
    old = torch._C._jit_get_inline_everything_mode()
    # 设置新的内联模式
    torch._C._jit_set_inline_everything_mode(should_inline)
    try:
        yield  # 执行被装饰函数体
    finally:
        # 恢复之前保存的内联模式
        torch._C._jit_set_inline_everything_mode(old)

# 定义一个上下文管理器函数 set_fusion_group_inlining，用于设置 Torch 的融合组内联选项
@contextmanager
def set_fusion_group_inlining(inlining):
    # 获取当前融合组内联选项并保存到 old 变量中
    old = torch._C._debug_get_fusion_group_inlining()
    # 设置新的融合组内联选项
    torch._C._debug_set_fusion_group_inlining(inlining)
    try:
        yield  # 执行被装饰函数体
    finally:
        # 恢复之前保存的融合组内联选项
        torch._C._debug_set_fusion_group_inlining(old)

# 定义一个上下文管理器函数 disable_autodiff_subgraph_inlining，用于禁用 Torch 的自动微分子图内联
# 注意: 不可重入，只能嵌套使用
@contextmanager
def disable_autodiff_subgraph_inlining(enabled=True):
    # 设置自动微分子图内联状态
    torch._C._debug_set_autodiff_subgraph_inlining(not enabled)
    try:
        yield  # 执行被装饰函数体
    finally:
        # 恢复自动微分子图内联状态
        torch._C._debug_set_autodiff_subgraph_inlining(True)

# 定义一个函数 _inline_everything，用于内联 Torch 函数
def _inline_everything(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with inline_everything_mode(True):
            fn(*args, **kwargs)
    return wrapper

# 临时函数，为了向前兼容，标记为不使用
# TODO: (suo) 移除
def _tmp_donotuse_dont_inline_everything(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with inline_everything_mode(False):
            fn(*args, **kwargs)
    return wrapper

# 定义一个函数 _trace，用于追踪函数以进行测试
def _trace(*args, **kwargs):
    def wrapper(func):
        return torch.jit.trace(func, args, **kwargs)
    return wrapper

# 定义一个装饰器函数 enable_cpu_fuser，用于启用 CPU 融合器
def enable_cpu_fuser(fn):
    def wrapper(*args, **kwargs):
        # 临时修改 CPU 融合器相关设置
        torch._C._jit_override_can_fuse_on_cpu_legacy(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_set_te_must_use_llvm_cpu(False)
        try:
            fn(*args, **kwargs)  # 执行被装饰函数体
        finally:
            # 恢复 CPU 融合器相关设置
            torch._C._jit_override_can_fuse_on_cpu_legacy(False)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_set_te_must_use_llvm_cpu(True)
    return wrapper

# 定义一个条件装饰器函数 enable_cpu_fuser_if，根据条件决定是否启用 CPU 融合器
def enable_cpu_fuser_if(cond):
    if cond:
        return enable_cpu_fuser
    else:
        def noop_fuser(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return noop_fuser

# 定义函数 get_forward，用于获取对象的 forward 方法
def get_forward(c):
    return c._get_method('forward')

# 定义函数 get_forward_graph，用于获取对象 forward 方法的计算图
def get_forward_graph(c):
    return c._get_method('forward').graph

# 定义函数 get_module_method，用于获取模块中指定方法的方法对象
def get_module_method(m, module, method):
    return m._c.getattr(module)._get_method(method)

# 定义函数 attrs_with_prefix，用于获取模块中以指定前缀开头的属性名列表
def attrs_with_prefix(module, prefix):
    return [x for x, _ in module._modules._c.items()
            if x.startswith(prefix)]

# 定义函数 warmup_backward，用于对反向传播函数进行预热
def warmup_backward(f, *args):
    profiling_count = 3
    results = []
    # 对于给定的 profiling_count 次数，循环执行下列操作
    for i in range(profiling_count):
        # 检查参数 args 是否非空
        if len(args) > 0:
            # 使用 PyTorch 的自动求导功能计算函数 f 对参数 args 的梯度
            r = torch.autograd.grad(f, *args)
            # 将计算得到的梯度结果 r 添加到结果列表 results 中
            results.append(r)
        else:
            # 如果参数 args 为空，则使用反向传播方法计算函数 f 的梯度，并保留计算图
            f.backward(retain_graph=True)
    
    # 返回包含所有结果的列表 results
    return results
# 删除此函数一旦 https://bugs.python.org/issue42666 得到解决
def make_global(*args):
    # 遍历参数列表中的每个函数或对象
    for arg in args:
        # 将每个函数或对象设置为其所在模块的全局变量
        setattr(sys.modules[arg.__module__], arg.__name__, arg)

# 辅助函数，用于在 Python 3 下评估代码，避免在 Py2 中导致语法错误
def _get_py3_code(code, fn_name):
    # 创建一个临时目录来存放脚本文件
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = os.path.join(tmp_dir, 'script.py')
        # 将代码写入脚本文件
        with open(script_path, 'w') as f:
            f.write(code)
        # 根据脚本文件创建模块规范
        spec = importlib.util.spec_from_file_location(fn_name, script_path)
        # 根据规范加载模块
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        # 断言 loader 类型符合 MyPy 的要求
        assert isinstance(loader, Loader)
        # 执行模块中的代码
        loader.exec_module(module)
        # 获取模块中的指定函数
        fn = getattr(module, fn_name)
        return fn

class TensorExprTestOptions:
    def __init__(self):
        # 设置 torch 的性能分析执行器为启用状态，并保存旧状态
        self.old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        # 设置 torch 图执行优化模式为启用状态，并保存旧状态
        self.old_profiling_mode = torch._C._get_graph_executor_optimize(True)

        # 保存旧的 CPU 融合状态，并设置为可以在 CPU 上进行融合
        self.old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        # 保存旧的 GPU 融合状态，并设置为可以在 GPU 上进行融合
        self.old_gpu_fuser_state = torch._C._jit_can_fuse_on_gpu()
        # 设置允许在 CPU 上进行融合
        torch._C._jit_override_can_fuse_on_cpu(True)
        # 设置允许在 GPU 上进行融合
        torch._C._jit_override_can_fuse_on_gpu(True)
        # 保存旧的 tensor 表达式融合状态，并设置为启用状态
        self.texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        torch._C._jit_set_texpr_fuser_enabled(True)
        # 保存旧的融合组内联状态，并设置为禁用
        self.old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        # 保存旧的 LLVM CPU 必须使用状态，并设置为禁用
        self.old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

    def restore(self):
        # 恢复 torch 的性能分析执行器状态为原来的状态
        torch._C._jit_set_profiling_executor(self.old_profiling_executor)
        # 恢复 torch 图执行优化模式状态为原来的状态
        torch._C._get_graph_executor_optimize(self.old_profiling_mode)

        # 恢复 tensor 表达式融合状态为原来的状态
        torch._C._jit_set_texpr_fuser_enabled(self.texpr_fuser_state)
        # 恢复 GPU 融合状态为原来的状态
        torch._C._jit_override_can_fuse_on_gpu(self.old_gpu_fuser_state)
        # 恢复 CPU 融合状态为原来的状态
        torch._C._jit_override_can_fuse_on_cpu(self.old_cpu_fuser_state)
        # 恢复融合组内联状态为原来的状态
        torch._C._debug_set_fusion_group_inlining(self.old_fusion_inlining)
        # 恢复 LLVM CPU 必须使用状态为原来的状态
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_te_must_use_llvm_cpu)

def clone_inputs(args):
    # 初始化输入列表
    inputs: List[Union[torch.Tensor, List[torch.Tensor]]] = []

    # 遍历参数列表中的每个参数
    for arg in args:
        # 如果参数是 torch.Tensor 类型，将其复制到 inputs 列表中
        if isinstance(arg, torch.Tensor):
            inputs.append(arg.detach().clone())
        # 如果参数是张量的可迭代对象，将每个张量都复制到 inputs 列表中
        elif is_iterable_of_tensors(arg):
            inputs.append([t.detach().clone() for t in arg])
        # 对于其他类型的参数，直接添加到 inputs 列表中
        else:
            inputs.append(arg)

    return inputs

def get_traced_sample_variant_pairs(device, dtype, op):
    # 初始化输出列表，包含元组 (variant, sample)
    outputs: List[Tuple[Any, Any]] = []

    # 使用操作对象的方法获取样本输入
    samples = op.sample_inputs(device, dtype)

    # 获取要测试的不同变体
    func = op.get_op()
    method = op.get_method()
    variants = {
        # TODO: 原地测试目前失败，修复并添加原地变体
        'function': func, 'method': method,
    }
    # 检查操作名是否属于假操作函数列表
    has_fake_function = op.name in ["resize_", 'resize_as_']

    if has_fake_function:
        # 如果存在假操作函数，创建对应的操作变体字典
        variants = {'method': getattr(torch.Tensor, op.name)}

    # 在即时执行模式（JIT）中，某些操作函数接受的参数为 (Tensor, Scalar)；
    # 在非即时执行模式下（eager mode），这些操作可以接受 (Tensor, bool) 类型的参数。
    # 为了在 JIT 模式下测试这些操作，将 bool 类型的参数转换为 int 类型。
    ops_with_unsupported_bool_args = [
        {
            "name": "div_floor_rounding",
            "arg_idx": [0],
        },
        {
            "name": "div_no_rounding_mode",
            "arg_idx": [0],
        },
        {
            "name": "div_trunc_rounding",
            "arg_idx": [0],
        },
        {
            "name": "index_fill",
            "arg_idx": [2],
        },
        {
            "name": "full_like",
            "arg_idx": [0],
        },
        {
            "name": "mul",
            "arg_idx": [0],
        },
        {
            "name": "new_full",
            "arg_idx": [1],
        },
    ]

    # 如果存在假操作函数，直接返回已有输出
    if has_fake_function:
        return outputs

    # 遍历每个样本
    for sample in samples:
        # 遍历每个操作变体
        for variant in variants.values():
            # 跳过空操作变体
            if variant is None:
                continue

            # 如果操作变体是 lambda 函数，跳过
            if is_lambda(variant):
                continue

            # 查找与当前操作匹配的不支持 bool 类型参数的操作数据
            matching_ops = filter(lambda x: op.formatted_name == x["name"], ops_with_unsupported_bool_args)
            # 对每个匹配的操作数据进行处理
            for op_data in matching_ops:
                # 遍历需要转换参数的索引
                for idx in op_data["arg_idx"]:
                    args = list(sample.args)
                    # 如果参数列表中的参数类型为 bool，则转换为 int
                    if len(sample.args) > idx and isinstance(sample.args[idx], bool):
                        args[idx] = int(args[idx])
                    sample.args = tuple(args)

            # 将处理后的样本添加到输出列表中
            outputs.append((variant, sample))

    # 返回处理后的输出列表
    return outputs
# 检查输入对象是否为 Lambda 函数
def is_lambda(lamb):
    # 创建一个简单的 Lambda 函数 LAMBDA，并标记忽略 E731 规则（关于无参数 lambda 表达式的警告）
    LAMBDA = lambda: 0  # noqa: E731
    # 判断输入对象是否为 LAMBDA 类型，并且其名称与 LAMBDA 的名称相同
    return isinstance(lamb, type(LAMBDA)) and lamb.__name__ == LAMBDA.__name__
```