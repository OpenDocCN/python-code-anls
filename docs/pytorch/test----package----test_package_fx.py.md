# `.\pytorch\test\package\test_package_fx.py`

```
# Owner(s): ["oncall: package/deploy"]

# 导入所需模块和函数
from io import BytesIO
import torch
from torch.fx import Graph, GraphModule, symbolic_trace
from torch.package import (
    ObjMismatchError,
    PackageExporter,
    PackageImporter,
    sys_importer,
)
from torch.testing._internal.common_utils import run_tests

try:
    # 尝试从当前包中导入 PackageTestCase
    from .common import PackageTestCase
except ImportError:
    # 如果导入失败，支持直接运行此文件的情况，从 common 模块中导入 PackageTestCase
    from common import PackageTestCase

# 对内置函数 len 进行 FX 封装，以便进行跟踪
torch.fx.wrap("len")
# 再次封装，确保不会影响其他内容
torch.fx.wrap("len")

class TestPackageFX(PackageTestCase):
    """Tests for compatibility with FX."""

    def test_package_fx_simple(self):
        # 定义一个简单的 torch.nn.Module
        class SimpleTest(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 3.0)

        # 创建 SimpleTest 实例
        st = SimpleTest()
        # 对其进行符号化跟踪
        traced = symbolic_trace(st)

        # 创建一个 BytesIO 对象
        f = BytesIO()
        # 使用 PackageExporter 将 traced 对象保存为 pickle 文件
        with PackageExporter(f) as pe:
            pe.save_pickle("model", "model.pkl", traced)

        f.seek(0)
        # 创建 PackageImporter 对象，用于导入之前保存的模型
        pi = PackageImporter(f)
        loaded_traced = pi.load_pickle("model", "model.pkl")
        # 创建输入数据
        input = torch.rand(2, 3)
        # 断言加载的模型和原始模型在相同输入下产生相同输出
        self.assertEqual(loaded_traced(input), traced(input))

    def test_package_then_fx(self):
        # 从 package_a.test_module 中导入 SimpleTest 类
        from package_a.test_module import SimpleTest

        # 创建 SimpleTest 实例
        model = SimpleTest()
        # 创建一个 BytesIO 对象
        f = BytesIO()
        # 使用 PackageExporter 将 model 对象保存为 pickle 文件
        with PackageExporter(f) as pe:
            pe.intern("**")
            pe.save_pickle("model", "model.pkl", model)

        f.seek(0)
        # 创建 PackageImporter 对象，用于导入之前保存的模型
        pi = PackageImporter(f)
        loaded = pi.load_pickle("model", "model.pkl")
        # 对加载的模型进行符号化跟踪
        traced = symbolic_trace(loaded)
        # 创建输入数据
        input = torch.rand(2, 3)
        # 断言加载的模型和符号化跟踪后的模型在相同输入下产生相同输出
        self.assertEqual(loaded(input), traced(input))

    def test_package_fx_package(self):
        # 从 package_a.test_module 中导入 SimpleTest 类
        from package_a.test_module import SimpleTest

        # 创建 SimpleTest 实例
        model = SimpleTest()
        # 创建一个 BytesIO 对象
        f = BytesIO()
        # 使用 PackageExporter 将 model 对象保存为 pickle 文件
        with PackageExporter(f) as pe:
            pe.intern("**")
            pe.save_pickle("model", "model.pkl", model)

        f.seek(0)
        # 创建 PackageImporter 对象，用于导入之前保存的模型
        pi = PackageImporter(f)
        loaded = pi.load_pickle("model", "model.pkl")
        # 对加载的模型进行符号化跟踪
        traced = symbolic_trace(loaded)

        # 创建一个新的 BytesIO 对象
        f2 = BytesIO()
        # 尝试重新保存之前保存的模型，预期会引发 ObjMismatchError
        with self.assertRaises(ObjMismatchError):
            with PackageExporter(f2) as pe:
                pe.intern("**")
                pe.save_pickle("model", "model.pkl", traced)

        f2.seek(0)
        # 创建一个带有自定义导入器的 PackageExporter 对象
        with PackageExporter(f2, importer=(pi, sys_importer)) as pe:
            # 将当前环境中的所有内容都包含在导出的包中
            pe.intern("**")
            pe.save_pickle("model", "model.pkl", traced)
        f2.seek(0)
        # 创建新的 PackageImporter 对象，用于导入重新保存的模型
        pi2 = PackageImporter(f2)
        loaded2 = pi2.load_pickle("model", "model.pkl")

        input = torch.rand(2, 3)
        # 断言两次导入的模型在相同输入下产生相同输出
        self.assertEqual(loaded(input), loaded2(input))
    def test_package_fx_with_imports(self):
        import package_a.subpackage
        # 手动构建一个调用叶子函数的图形结构

        graph = Graph()  # 创建一个空的图对象
        a = graph.placeholder("x")  # 创建一个图的占位符节点 'x'
        b = graph.placeholder("y")  # 创建一个图的占位符节点 'y'
        c = graph.call_function(package_a.subpackage.leaf_function, (a, b))
        # 在图中调用 package_a.subpackage.leaf_function 函数，并将 a 和 b 作为参数传递
        d = graph.call_function(torch.sin, (c,))
        # 在图中调用 torch.sin 函数，并将 c 作为参数传递
        graph.output(d)  # 将 d 设置为图的输出节点
        gm = GraphModule(torch.nn.Module(), graph)
        # 创建一个图模块，使用 torch.nn.Module 作为根模块，graph 作为图结构

        f = BytesIO()  # 创建一个字节流对象
        with PackageExporter(f) as pe:
            pe.intern("**")  # 在打包过程中包含所有依赖
            pe.save_pickle("model", "model.pkl", gm)
            # 将 gm 对象以 "model.pkl" 的文件名保存为 pickle 格式
        f.seek(0)  # 将文件指针移动到文件开头

        pi = PackageImporter(f)  # 创建一个包导入器对象，使用前面创建的字节流对象 f
        loaded_gm = pi.load_pickle("model", "model.pkl")
        # 从 pickle 文件 "model.pkl" 中加载对象，预期为 GraphModule 类型的 gm

        input_x = torch.rand(2, 3)  # 创建一个随机张量作为输入
        input_y = torch.rand(2, 3)  # 创建另一个随机张量作为输入

        self.assertTrue(
            torch.allclose(loaded_gm(input_x, input_y), gm(input_x, input_y))
        )
        # 断言：加载的 gm 对象在输入 input_x 和 input_y 下的输出与原始 gm 对象的输出相近

        # 检查打包的 leaf_function 依赖版本与外部环境中的版本不同
        packaged_dependency = pi.import_module("package_a.subpackage")
        self.assertTrue(packaged_dependency is not package_a.subpackage)
        # 断言：打包的 leaf_function 依赖与外部环境中的 package_a.subpackage 不同

    def test_package_fx_custom_tracer(self):
        from package_a.test_all_leaf_modules_tracer import TestAllLeafModulesTracer
        from package_a.test_module import ModWithTwoSubmodsAndTensor, SimpleTest

        class SpecialGraphModule(torch.fx.GraphModule):
            def __init__(self, root, graph, info):
                super().__init__(root, graph)
                self.info = info

        sub_module = SimpleTest()  # 创建 SimpleTest 类的实例对象 sub_module
        module = ModWithTwoSubmodsAndTensor(
            torch.ones(3),
            sub_module,
            sub_module,
        )  # 创建 ModWithTwoSubmodsAndTensor 类的实例对象 module，并传入参数

        tracer = TestAllLeafModulesTracer()  # 创建 TestAllLeafModulesTracer 类的实例对象 tracer
        graph = tracer.trace(module)  # 使用 tracer 对象追踪 module 对象的图形结构

        self.assertEqual(graph._tracer_cls, TestAllLeafModulesTracer)
        # 断言：graph 对象的追踪类为 TestAllLeafModulesTracer

        gm = SpecialGraphModule(module, graph, "secret")
        # 创建 SpecialGraphModule 类的实例对象 gm，传入 module, graph 和字符串 "secret" 作为参数

        self.assertEqual(gm._tracer_cls, TestAllLeafModulesTracer)
        # 断言：gm 对象的追踪类为 TestAllLeafModulesTracer

        f = BytesIO()  # 创建一个字节流对象
        with PackageExporter(f) as pe:
            pe.intern("**")  # 在打包过程中包含所有依赖
            pe.save_pickle("model", "model.pkl", gm)
            # 将 gm 对象以 "model.pkl" 的文件名保存为 pickle 格式
        f.seek(0)  # 将文件指针移动到文件开头

        pi = PackageImporter(f)  # 创建一个包导入器对象，使用前面创建的字节流对象 f
        loaded_gm = pi.load_pickle("model", "model.pkl")
        # 从 pickle 文件 "model.pkl" 中加载对象，预期为 SpecialGraphModule 类型的 gm

        self.assertEqual(
            type(loaded_gm).__class__.__name__, SpecialGraphModule.__class__.__name__
        )
        # 断言：加载的 gm 对象的类名与 SpecialGraphModule 类的类名相同
        self.assertEqual(loaded_gm.info, "secret")
        # 断言：加载的 gm 对象的 info 属性为字符串 "secret"

        input_x = torch.randn(3)  # 创建一个正态分布的随机张量作为输入

        self.assertEqual(loaded_gm(input_x), gm(input_x))
        # 断言：加载的 gm 对象在输入 input_x 下的输出与原始 gm 对象的输出相等
    # 定义一个测试函数 test_package_fx_wrap，用于测试 Torch 模型的打包与加载功能
    def test_package_fx_wrap(self):
        # 定义一个内部测试模块 TestModule，继承自 torch.nn.Module
        class TestModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()

            # 前向传播方法，返回输入张量的长度
            def forward(self, a):
                return len(a)

        # 使用 Torch 的符号化跟踪功能，对 TestModule 进行符号化跟踪
        traced = torch.fx.symbolic_trace(TestModule())

        # 创建一个字节流对象 f
        f = BytesIO()
        # 使用 torch.package 包中的 PackageExporter 类，将 traced 对象保存为 pickle 格式到 f 中
        with torch.package.PackageExporter(f) as pe:
            pe.save_pickle("model", "model.pkl", traced)
        # 将文件指针移动到流的开头
        f.seek(0)

        # 使用 PackageImporter 类，从 f 中加载名为 "model.pkl" 的 pickle 文件
        pi = PackageImporter(f)
        loaded_traced = pi.load_pickle("model", "model.pkl")
        # 创建一个形状为 (2, 3) 的随机张量作为输入
        input = torch.rand(2, 3)
        # 断言 loaded_traced 对象对输入 input 的运行结果与 traced 对象的运行结果相同
        self.assertEqual(loaded_traced(input), traced(input))
# 如果当前模块被直接执行（而非被导入到其他模块），则执行下面的代码
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试
    run_tests()
```