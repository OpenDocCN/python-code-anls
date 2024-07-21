# `.\pytorch\test\fx\test_lazy_graph_module.py`

```
# Owner(s): ["oncall: fx"]

# 引入上下文管理、序列化、字节流等必要库
import contextlib
import pickle
from io import BytesIO
from unittest.mock import patch

# 引入 PyTorch 相关库和模块
import torch
import torch._export
from torch import fx
from torch.fx._lazy_graph_module import (
    _LazyGraphModule,
    _make_graph_module,
    _use_lazy_graph_module,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import run_tests, TestCase

# 测试用例类，继承自 TestCase
class TestLazyGraphModule(TestCase):
    exit_stack = None

    @classmethod
    def setUpClass(cls):
        # 设置类级别的退出堆栈，用于管理资源
        cls.exit_stack = contextlib.ExitStack()
        cls.exit_stack.enter_context(_use_lazy_graph_module(True))

    @classmethod
    def tearDownClass(cls):
        # 在类销毁时关闭退出堆栈
        cls.exit_stack.close()

    @staticmethod
    def replace_sin_with_cos(gm):
        # 替换计算图中所有 sin 函数为 cos 函数
        for n in gm.graph.nodes:
            if n.target == "sin":
                n.target = "cos"

    def test_replace_sin_with_cos(self):
        # 定义一个简单的函数，对输入 x 进行 sin 函数处理
        def f(x):
            return x.sin()

        x = torch.randn(2, 3)
        # 对函数 f 进行符号跟踪，生成计算图
        gm = fx.symbolic_trace(f)
        # 调用替换函数，将计算图中的 sin 替换为 cos
        self.replace_sin_with_cos(gm)

        # 重新编译计算图
        gm.recompile()
        expected = x.cos()
        # 对输入 x 应用修改后的计算图进行计算
        actual = gm(x)

        # 断言期望输出与实际输出的接近程度
        self.assertTrue(torch.allclose(expected, actual))
        # 获取计算图的可读表示，验证是否包含了 cos 函数
        code = gm.print_readable(False)
        self.assertTrue("cos()" in code)
        # 断言 gm 是 _LazyGraphModule 类的实例
        self.assertTrue(isinstance(gm, _LazyGraphModule))

    def test_call_forward_directly(self):
        # 定义一个简单的函数，对输入 x 进行 sin 函数处理
        def f(x):
            return x.sin()

        x = torch.randn(2, 3)
        # 对函数 f 进行符号跟踪，生成计算图
        gm = fx.symbolic_trace(f)
        # 断言 gm 是 _LazyGraphModule 类的实例
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        # 调用替换函数，将计算图中的 sin 替换为 cos
        self.replace_sin_with_cos(gm)
        # 重新编译计算图
        gm.recompile()
        expected = x.cos()
        # 直接调用 forward 方法进行前向传播计算
        actual = gm.forward(x)

        # 断言期望输出与实际输出的接近程度
        self.assertTrue(torch.allclose(expected, actual))

    def test_needs_recompile(self):
        """
        Make sure needs_recompile() return the corrent state.
        """

        # 定义一个简单的函数，对输入 x 进行 sin 函数处理
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        # 断言 gm 是 _LazyGraphModule 类的实例
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        # 判断计算图是否需要重新编译
        self.assertTrue(gm._needs_recompile())
        # 对计算图进行一次前向传播
        gm(torch.randn(2, 3))
        # 再次判断计算图是否需要重新编译
        self.assertFalse(gm._needs_recompile())

    def test_multi_recompile(self):
        """
        Cover the case that multiple recompilation happens.
        """

        # 定义一个简单的函数，对输入 x 进行 sin 函数处理
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        # 断言 gm 是 _LazyGraphModule 类的实例
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        # 判断计算图是否需要重新编译
        self.assertTrue(gm._needs_recompile())
        x = torch.randn(2, 3)
        # 触发第一次重新编译
        self.assertTrue(torch.allclose(x.sin(), gm(x)))
        # 再次判断计算图是否需要重新编译
        self.assertFalse(gm._needs_recompile())

        # 替换计算图中的 sin 函数为 cos 函数
        self.replace_sin_with_cos(gm)
        # 再次判断计算图是否需要重新编译
        self.assertFalse(gm._needs_recompile())
        # 再次进行重新编译
        gm.recompile()
        # 再次判断计算图是否需要重新编译
        self.assertTrue(gm._needs_recompile())
        # 触发第二次重新编译
        self.assertTrue(torch.allclose(x.cos(), gm(x)))
        # 最后判断计算图是否需要重新编译
        self.assertFalse(gm._needs_recompile())
    # 定义测试函数，用于验证访问 GraphModule 的 code 属性时是否会重新编译
    def test_accessing_code_cause_recompiling(self):
        """
        Make sure we recompile if we have not done that yet when we access the code
        property of a GraphModule.
        """

        # 定义简单的函数 f(x)，其中包含一个数学函数 sin 的调用
        def f(x):
            return x.sin()

        # 对函数 f 进行符号化跟踪，返回一个 LazyGraphModule 对象 gm
        gm = fx.symbolic_trace(f)
        # 断言 gm 是 _LazyGraphModule 的实例
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        # 断言 gm 需要重新编译
        self.assertTrue(gm._needs_recompile())
        # 访问 gm 的 code 属性，应该触发重新编译
        code = gm.code
        # 断言生成的 code 中包含 "sin" 函数
        self.assertTrue("sin" in code)
        # 断言 gm 不再需要重新编译
        self.assertFalse(gm._needs_recompile())

    # 定义测试函数，用于验证 GraphModule 对象的字符串表示中是否包含 "sin"
    def test_graph_module_str(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        # 断言 gm 是 _LazyGraphModule 的实例
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        # 断言 gm 的字符串表示中包含 "sin"
        self.assertTrue("sin" in str(gm))

    # 定义测试函数，验证使用 make_fx 对象再次对 LazyGraphModule 进行符号化跟踪时的行为
    def test_recapture_with_make_fx(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        # 断言 gm 是 _LazyGraphModule 的实例
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        # 断言 gm 需要重新编译
        self.assertTrue(gm._needs_recompile())
        # 使用 make_fx 对 gm 进行处理，生成新的 LazyGraphModule 对象 gm2
        gm2 = make_fx(gm)(torch.randn(2, 3))
        # 断言 gm2 是 _LazyGraphModule 的实例
        self.assertTrue(isinstance(gm2, _LazyGraphModule))
        # 断言 gm2 需要重新编译
        self.assertTrue(gm2._needs_recompile())

        # make_fx 将调用 gm 的 forward 方法，这会清除 _needs_recompile() 标志
        self.assertFalse(gm._needs_recompile())

    # 定义测试函数，验证使用 symbolic_trace 对 LazyGraphModule 进行再次符号化跟踪的行为
    def test_recapture_with_symbolic_trace(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        # 断言 gm 是 _LazyGraphModule 的实例
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        # 断言 gm 需要重新编译
        self.assertTrue(gm._needs_recompile())
        # 使用 symbolic_trace 对 gm 进行再次符号化跟踪，生成新的 LazyGraphModule 对象 gm2
        gm2 = fx.symbolic_trace(gm)

        # lazy recompilation 已经被实现，我们在 symbolic_trace 开始时实现了重新编译
        self.assertFalse(gm._needs_recompile())
        self.assertTrue(gm2._needs_recompile())

    # 定义测试函数，验证使用 dynamo 进行重新捕获的行为
    def test_recapture_with_dynamo(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        # 断言 gm 是 _LazyGraphModule 的实例
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        # 断言 gm 需要重新编译
        self.assertTrue(gm._needs_recompile())
        # 使用 torch.compile 调用 gm 的 forward 方法，并安装 eval hook，这将触发真正的重新编译
        torch.compile(gm)(torch.rand(2, 3))

        # dynamo 调用 gm.forward 并安装 eval hook，这将触发真正的重新编译
        self.assertFalse(gm._needs_recompile())
    def test_save_lazy_foward(self):
        """
        Save the lazy forward method and call it repeatly. Make sure we
        don't recompile for each such call.
        """

        def f(x):
            return x.sin()  # 定义一个函数 f，计算输入张量 x 的正弦值

        orig_gm_recompile = fx.GraphModule.recompile  # 保存原始的 GraphModule 类的 recompile 方法
        recompile_count = 0  # 初始化重新编译计数器

        def mock_gm_recompile(self):
            nonlocal recompile_count  # 使用 nonlocal 关键字引用外部函数中的 recompile_count
            recompile_count += 1  # 每调用一次 mock_gm_recompile 方法，重新编译计数器加一
            return orig_gm_recompile(self)  # 调用原始的 GraphModule 类的 recompile 方法

        with patch.object(fx.GraphModule, "recompile", mock_gm_recompile):  # 使用 mock_gm_recompile 替换 GraphModule 类的 recompile 方法
            gm = fx.symbolic_trace(f)  # 对函数 f 进行符号跟踪，返回一个 LazyGraphModule 对象
            self.assertTrue(isinstance(gm, _LazyGraphModule))  # 断言 gm 是 _LazyGraphModule 类的实例
            saved_fwd = gm.forward  # 保存 LazyGraphModule 对象的 forward 方法

            x = torch.rand(2, 3)  # 创建一个大小为 (2, 3) 的随机张量
            for _ in range(10):
                saved_fwd(x)  # 多次调用保存的 forward 方法

        self.assertEqual(recompile_count, 1)  # 断言重新编译计数器的值为 1

    def test_pickle(self):
        """
        Fx graph cache need the ability to pickle GraphModule/_LazyGraphModule.
        """

        def f(x):
            return x.sin()  # 定义一个函数 f，计算输入张量 x 的正弦值

        gm = fx.symbolic_trace(f)  # 对函数 f 进行符号跟踪，返回一个 LazyGraphModule 对象
        self.assertTrue(isinstance(gm, _LazyGraphModule))  # 断言 gm 是 _LazyGraphModule 类的实例
        serialized = pickle.dumps(gm)  # 序列化 LazyGraphModule 对象
        gm2 = pickle.loads(serialized)  # 反序列化得到一个新的 LazyGraphModule 对象
        self.assertTrue(isinstance(gm2, _LazyGraphModule))  # 断言 gm2 是 _LazyGraphModule 类的实例
        self.assertTrue("sin" in gm2.code)  # 断言 gm2 的代码中包含 "sin" 字符串

    def test_make_graph_module(self):
        gm = fx.symbolic_trace(lambda x: x.sin())  # 对 lambda 函数进行符号跟踪，返回一个 LazyGraphModule 对象
        self.assertTrue(isinstance(gm, _LazyGraphModule))  # 断言 gm 是 _LazyGraphModule 类的实例

        gm1 = _make_graph_module(
            gm, gm.graph, class_name="MyGraphModule", graph_module_cls=fx.GraphModule
        )  # 使用 _make_graph_module 函数创建一个新的 GraphModule 对象 gm1
        self.assertFalse(isinstance(gm1, _LazyGraphModule))  # 断言 gm1 不是 _LazyGraphModule 类的实例
        self.assertTrue(gm1.__class__.__name__ == "MyGraphModule")  # 断言 gm1 的类名为 "MyGraphModule"

        gm2 = _make_graph_module(gm, gm.graph)  # 使用 _make_graph_module 函数创建一个新的 LazyGraphModule 对象 gm2
        self.assertTrue(isinstance(gm2, _LazyGraphModule))  # 断言 gm2 是 _LazyGraphModule 类的实例
        self.assertTrue(gm2.__class__.__name__ == "GraphModule")  # 断言 gm2 的类名为 "GraphModule"

    def test_package_fx_simple(self):
        """
        Copied from test/package/test_package_fx.py to make sure LazyGraphModule
        works with torch.package.
        """

        class SimpleTest(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 3.0)  # 定义一个简单的前向传播方法

        st = SimpleTest()  # 创建 SimpleTest 类的实例
        traced = fx.symbolic_trace(st)  # 对 SimpleTest 实例进行符号跟踪，返回一个 LazyGraphModule 对象

        f = BytesIO()  # 创建一个 BytesIO 对象
        with PackageExporter(f) as pe:  # 使用 PackageExporter 导出器
            pe.save_pickle("model", "model.pkl", traced)  # 将 traced 对象保存为 pickle 文件 "model.pkl"

        f.seek(0)  # 将文件指针移动到文件开头
        pi = PackageImporter(f)  # 创建 PackageImporter 对象
        loaded_traced = pi.load_pickle("model", "model.pkl")  # 从 pickle 文件中加载 traced 对象
        input = torch.rand(2, 3)  # 创建一个大小为 (2, 3) 的随机张量
        self.assertEqual(loaded_traced(input), traced(input))  # 断言加载的 traced 对象与原始 traced 对象在给定输入下的输出相等
    def test_dynamo_innermost_fn(self):
        """
        Repro for https://github.com/pytorch/pytorch/issues/121198 .
        """
        # 定义内部函数 f，实现对输入 x 的乘以 2 的操作
        def f(x):
            return x * 2
        
        # 对函数 f 进行符号化追踪，得到图模块 gm
        gm = torch.fx.symbolic_trace(f)
        # 将符号化追踪得到的图模块 gm 转换为懒加载图模块 lazy_gm
        lazy_gm = torch.fx._lazy_graph_module._LazyGraphModule.from_graphmodule(gm)
        
        # 使用 torch._dynamo.disable 方法禁用 gm 的 forward 方法并封装
        wrapped_forward = torch._dynamo.disable(gm.forward)
        # 获取禁用后的 forward 方法的最内层函数
        got_inner_forward = torch._dynamo.eval_frame.innermost_fn(wrapped_forward)
        # 断言得到的最内层函数对象具有 "__self__" 属性
        assert hasattr(got_inner_forward, "__self__")
        
        # 使用 torch._dynamo.disable 方法禁用 lazy_gm 的 forward 方法并封装
        wrapped_lazy_forward = torch._dynamo.disable(lazy_gm.forward)
        # 获取禁用后的 lazy_gm 的 forward 方法的最内层函数
        got_lazy_inner_forward = torch._dynamo.eval_frame.innermost_fn(
            wrapped_lazy_forward
        )
        # 断言得到的最内层函数对象具有 "__self__" 属性
        assert hasattr(got_lazy_inner_forward, "__self__")
# 如果当前脚本作为主程序执行（而不是作为模块被导入），则执行run_tests()函数
if __name__ == "__main__":
    run_tests()
```