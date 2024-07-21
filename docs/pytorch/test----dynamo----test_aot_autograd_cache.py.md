# `.\pytorch\test\dynamo\test_aot_autograd_cache.py`

```py
# Owner(s): ["module: dynamo"]

# 导入所需的库和模块
import os
import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case

# 导入 Functorch 相关模块和配置
import torch._functorch._aot_autograd
from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import (
    AOTAutogradCache,
    autograd_cache_key,
    BypassAOTAutogradCache,
)
from torch._functorch._aot_autograd.schemas import AOTConfig

# 导入 Inductor 相关配置
from torch._inductor import config as inductor_config
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


# 使用 parametrized_tests 装饰器来自动化实例化参数化测试
@instantiate_parametrized_tests
class AOTAutogradCacheTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        """
        在每个单元测试之前重置所有计数器和缓存
        """
        super().setUp()
        counters.clear()  # 清空计数器
        self._clear_all_caches()  # 清空所有缓存

    def _clear_all_caches(self):
        """
        清空所有缓存，包括 AOTAutogradCache 和 FXCache
        """
        torch._inductor.codecache.FxGraphCache.clear()  # 清空 FXGraphCache 缓存
        AOTAutogradCache.clear()  # 清空 AOTAutogradCache 缓存
        self._clear_dynamo_and_codecache()  # 清空 dynamo 和 codecache

    def _clear_dynamo_and_codecache(self):
        """
        清空不相关的缓存，如 dynamo 和 PyCodeCache
        """
        torch._dynamo.reset()  # 重置 dynamo
        # 删除 PyCodeCache 中所有缓存的模块文件
        for m in torch._inductor.codecache.PyCodeCache.cache.values():
            os.remove(m.__file__)
        torch._inductor.codecache.PyCodeCache.cache_clear()  # 清空 PyCodeCache 缓存

    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_basic(self):
        """
        验证 FXGraphCache 和 AOTAutogradCache 之间的交互。
        """

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # 第一次调用应该在缓存中未命中。
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # 第二次调用应该命中缓存。（首先重置以避免内存中的保护阻止编译）
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a, b))

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    # 应用functorch配置，启用自动求导缓存功能
    def test_clear_fx_graph_cache(self):
        """
        Verify the interactions between FXGraphCache and AOTAutogradCache.
        """
        
        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        # 第一次调用应未命中缓存
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear FX graph cache: second call should also be a miss
        # 清除FX图缓存：第二次调用也应未命中
        self._clear_dynamo_and_codecache()
        torch._inductor.codecache.FxGraphCache.clear()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        # We save again into the cache
        # 再次保存到缓存中
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

    @inductor_config.patch("fx_graph_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    # 应用functorch配置，启用自动求导缓存功能
    def test_fx_graph_cache_off(self):
        """
        Should not use cache if FXGraphCache is not enabled
        """
        
        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        # 第一次调用应未命中缓存
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Clear FX graph cache: second call should also be a miss
        # 清除FX图缓存：第二次调用也应未命中
        self._clear_dynamo_and_codecache()

        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @dynamo_config.patch("compiled_autograd", True)
    # 应用inductor、functorch和dynamo的配置，启用FX图缓存和自动求导缓存
    # 定义一个测试方法，用于测试编译后自动求导的绕过情况
    def test_compiled_autograd_bypass(self):
        # 定义一个函数 fn，接受两个张量 a 和 b 作为输入
        def fn(a, b):
            # 计算张量 a 的余弦值，加上张量 b，得到 out
            out = a.cos() + b
            # 计算 out 的和作为损失
            loss = out.sum()
            # 对损失进行自动求导，得到梯度 ga 和 gb
            ga, gb = torch.autograd.grad(loss, inputs=[a, b])

        # 创建两个形状为 (25,) 的随机张量 a 和 b，并标记为需要计算梯度
        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        # 分别对张量 a 和 b 进行去除梯度信息后克隆，并重新标记为需要计算梯度
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)
        # 使用 Inductor 后端编译函数 fn
        compiled_fn = torch.compile(fn, backend="inductor")
        # 断言编译后的函数和原函数在给定输入下的输出相同
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        # 断言计数器中自动求导缓存的缺失次数为 1，来自编译后的前向计算
        self.assertEqual(
            counters["aot_autograd"]["autograd_cache_miss"], 1
        )
        # 断言计数器中自动求导缓存的绕过次数为 1，来自编译后的自动求导
        self.assertEqual(
            counters["aot_autograd"]["autograd_cache_bypass"], 1
        )

    # 使用修饰器配置 Inductor 后端，启用 FX 图缓存
    @inductor_config.patch("fx_graph_cache", True)
    # 使用修饰器配置 Functorch，启用自动求导缓存
    @functorch_config.patch({"enable_autograd_cache": True})
    # 使用修饰器配置 Dynamo，启用编译后自动求导
    @dynamo_config.patch("compiled_autograd", True)
    # 定义测试推断图缓存在启用编译后自动求导时的命中情况
    def test_inference_graph_cache_hit_with_compiled_autograd_enabled(self):
        # 定义一个函数 fn，接受两个张量 a 和 b 作为输入，返回它们的余弦和
        def fn(a, b):
            out = a.cos() + b
            return out.sum()

        # 创建两个形状为 (25,) 的随机张量 a 和 b
        a = torch.randn(25)
        b = torch.randn(25)
        # 使用 Inductor 后端编译函数 fn
        compiled_fn = torch.compile(fn, backend="inductor")
        # 断言编译后的函数和原函数在给定输入下的输出相同
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        # 断言计数器中自动求导缓存的缺失次数为 1
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        # 断言计数器中自动求导缓存的保存次数为 1，来自编译后的自动求导
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # 清除 Dynamo 和代码缓存后再次运行，应命中缓存
        counters.clear()
        self._clear_dynamo_and_codecache()
        # 断言原函数在相同输入下的输出仍然相同
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        # 断言计数器中自动求导缓存的缺失次数为 0
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        # 断言计数器中自动求导缓存的命中次数为 1，来自缓存的命中
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        # 断言计数器中自动求导缓存的保存次数为 0
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    # 使用修饰器配置 Inductor 后端，启用 FX 图缓存
    @inductor_config.patch({"fx_graph_cache": True})
    # 使用修饰器配置 Functorch，启用自动求导缓存
    @functorch_config.patch({"enable_autograd_cache": True})
    # 定义一个测试方法，用于测试自动求导的延迟编译和缓存保存功能
    def test_autograd_lazy_backward(self):
        """
        Lazily compile the backward, and lazily save to cache
        """

        # 定义一个简单的函数，对输入的张量进行操作
        def fn(a, b):
            return a.cos() + b

        # 创建两个随机张量 a 和 b，并指定需要计算梯度
        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        
        # 克隆张量 a 和 b，并将其从计算图中分离后再重新指定需要计算梯度
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)
        
        # 编译函数 fn，使用特定的后端 "inductor"
        compiled_fn = torch.compile(fn, backend="inductor")
        
        # 断言原始函数和编译后函数对相同输入的输出结果相等
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        
        # 断言自动求导统计信息中的缓存未命中次数等于 1
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        # 断言自动求导统计信息中的缓存命中次数为 0
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        # 断言自动求导统计信息中的缓存保存次数为 0
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # 清除动态编译器和代码缓存，再次运行，应该仍然是缓存未命中，因为尚未运行反向传播
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # 现在运行反向传播
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        
        # 断言张量 a 和 a2 的梯度相等
        self.assertEqual(a.grad, a2.grad)
        # 断言张量 b 和 b2 的梯度相等
        self.assertEqual(b.grad, b2.grad)
        # 断言自动求导统计信息中的缓存保存次数为 1
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # 再次清除动态编译器和代码缓存，重新运行所有步骤，现在应该是缓存命中
        self._clear_dynamo_and_codecache()
        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)
    def test_autograd_function(self):
        """
        Tests autograd cache hits
        """

        # 定义一个测试自动微分缓存命中的函数
        def fn(a, b):
            return a.sin() + b

        # 创建两个需要梯度计算的随机张量 a 和 b
        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        # 分别从 a 和 b 创建副本，并设置需要梯度计算
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)

        # 编译 fn 函数，选择后端为 "inductor"
        compiled_fn = torch.compile(fn, backend="inductor")

        # 第一次调用应该会缓存未命中
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        # 对 fn(a, b) 的结果求和并进行反向传播
        fn(a, b).sum().backward()
        # 对 compiled_fn(a2, b2) 的结果求和并进行反向传播
        compiled_fn(a2, b2).sum().backward()
        # 检查梯度是否相等
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)

        # 检查自动微分缓存的统计信息
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # 重置所有张量
        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)

        # 第二次调用应该命中缓存（第一次重置以确保内存保护不阻止编译）
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)

        # 再次检查自动微分缓存的统计信息
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
# 使用装饰器设置特定配置（在这里是 "fx_graph_cache"）
@inductor_config.patch("fx_graph_cache", True)
# 定义一个测试类 AOTAutogradCachePicklerTests，继承自 torch._dynamo.test_case.TestCase
class AOTAutogradCachePicklerTests(torch._dynamo.test_case.TestCase):

    # 返回当前设备类型的属性，如果有 CUDA 则返回 "cuda"，否则返回 "cpu"
    @property
    def device_type(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    # 返回默认的 AOTConfig 对象，包含多个配置项的默认值
    def default_config(self):
        return AOTConfig(
            fw_compiler=None,
            bw_compiler=None,
            inference_compiler=None,
            partition_fn=None,
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
            dynamic_shapes=True,
            aot_autograd_arg_pos_to_source=None,
            is_export=False,
            no_tangents=False,
            enable_log=False,
        )

    # 获取 dynamo 输出的函数，重置 dynamo 状态，然后编译函数 fn，并返回结果、fx_graph 和 example_inputs
    def _get_dynamo_output(self, fn, *args, **kwargs):
        # 重置 dynamo 状态
        torch._dynamo.reset()
        fx_graph = None
        example_inputs = None

        def compiler(gm, inputs, **kwargs):
            nonlocal fx_graph
            nonlocal example_inputs
            fx_graph = gm
            example_inputs = inputs
            return gm

        # 编译函数 fn，使用自定义的编译器 compiler，并且获取完整的图形表示（fullgraph=True）
        g = torch.compile(fn, backend=compiler, fullgraph=True)
        # 运行编译后的图形表示，并返回结果
        result = g(*args, **kwargs)
        return (result, fx_graph, example_inputs)

    # 生成缓存键的方法，使用函数 f、配置 config 和可选输入 inputs（默认为 torch.ones(3)）
    def gen_cache_key(self, f, config, inputs=None):
        if inputs is None:
            inputs = [torch.ones(3)]
        # 获取函数 f 的 dynamo 输出，并返回其自动求导缓存键
        _, fx_g, example_inputs = self._get_dynamo_output(f, *inputs)
        return autograd_cache_key(fx_g, example_inputs, config)

    # 测试基本的哈希键是否稳定的方法
    def test_basic_hash_key(self):
        # 定义一个简单的函数 fn，计算输入张量 x 的 sin 和 cos 的复合函数
        def fn(x):
            return x.sin().cos()

        # 获取默认配置
        config = self.default_config()
        # 检查在多次运行中哈希键是否稳定
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config)
        self.assertEqual(c1, c2)

    # 测试相同图形和配置是否生成相同的缓存键
    def test_identical_graphs_and_configs(self):
        # 定义两个相同的函数 fn 和 fn2，只是 fn2 有更多的中间变量
        def fn(x):
            return x.sin().cos()

        def fn2(x):
            y = x.sin()
            z = y.cos()
            return z

        # 获取默认配置
        config = self.default_config()
        # 创建另一个相同的配置 config2，并设置不同的 aot_id
        config2 = self.default_config()
        config2.aot_id = 1

        # 获取两个函数 fn 和 fn2 的缓存键
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config2)
        # 检查两个缓存键是否相等
        self.assertEqual(c1, c2)

    # 测试不同的图形是否生成不同的缓存键
    def test_different_graphs(self):
        # 定义两个不同的函数 fn 和 fn2
        def fn(x):
            return x.cos().sin()

        def fn2(x):
            return x.sin().cos()

        # 获取默认配置
        config = self.default_config()
        # 获取两个函数 fn 和 fn2 的缓存键
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn2, config)
        # 检查两个缓存键是否不相等
        self.assertNotEqual(c1, c2)

    # 测试不同配置是否生成不同的缓存键
    def test_different_configs(self):
        # 定义一个函数 fn，计算输入张量 x 的 cos 和 sin 的复合函数
        def fn(x):
            return x.cos().sin()

        # 获取默认配置
        config = self.default_config()
        # 创建另一个相同的配置 config2，并设置 dynamic_shapes=False
        config2 = self.default_config()
        config2.dynamic_shapes = False

        # 获取两个函数 fn 在不同配置下的缓存键
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config2)
        # 检查两个缓存键是否不相等
        self.assertNotEqual(c1, c2)
    # 测试不同的输入情况
    def test_different_inputs(self):
        # 定义一个函数 fn，对输入 x 先求余弦再求正弦
        def fn(x):
            return x.cos().sin()

        # 获取默认配置
        config = self.default_config()
        # 生成缓存键 c1，使用函数 fn、配置 config，输入为包含三个元素的 torch.ones 张量
        c1 = self.gen_cache_key(fn, config, inputs=[torch.ones(3)])
        # 生成缓存键 c2，使用函数 fn、配置 config，输入为包含两个元素的 torch.ones 张量
        c2 = self.gen_cache_key(fn, config, inputs=[torch.ones(2)])
        # 断言 c1 和 c2 不相等
        self.assertNotEqual(c1, c2)

    # 测试相同的全局配置
    def test_different_global_configs(self):
        # 定义一个函数 fn，对输入 x 先求余弦再求正弦
        def fn(x):
            return x.cos().sin()

        # 获取默认配置
        config = self.default_config()

        # 生成缓存键 c1，使用函数 fn、配置 config
        c1 = self.gen_cache_key(fn, config)
        # 再次生成缓存键 c2，使用函数 fn、配置 config
        c2 = self.gen_cache_key(fn, config)
        # 断言 c1 和 c2 相等
        self.assertEqual(c1, c2)

        # 再次生成缓存键 c1，使用函数 fn、配置 config
        c1 = self.gen_cache_key(fn, config)

        # 修改 functorch 配置
        with functorch_config.patch(
            {"debug_assert": not functorch_config.debug_assert}
        ):
            # 生成缓存键 c2，使用函数 fn、配置 config
            c2 = self.gen_cache_key(fn, config)

        # 断言 c1 和 c2 不相等
        self.assertNotEqual(c1, c2)

        # 再次生成缓存键 c1，使用函数 fn、配置 config
        c1 = self.gen_cache_key(fn, config)

        # 修改 inductor 配置
        with inductor_config.patch({"debug": not inductor_config.debug}):
            # 生成缓存键 c2，使用函数 fn、配置 config
            c2 = self.gen_cache_key(fn, config)

        # 断言 c1 和 c2 不相等
        self.assertNotEqual(c1, c2)

        # 再次生成缓存键 c1，使用函数 fn、配置 config
        c1 = self.gen_cache_key(fn, config)

        # 禁用 torch 的梯度计算
        with torch.no_grad():
            # 生成缓存键 c2，使用函数 fn、配置 config
            c2 = self.gen_cache_key(fn, config)
        # 断言 c1 和 c2 不相等
        self.assertNotEqual(c1, c2)

    # 测试不兼容的函数
    def test_incompatible_function(self):
        # 使用 torch._dynamo.allow_in_graph 装饰器定义一个允许在计算图中的函数
        @torch._dynamo.allow_in_graph
        class AllowInGraphFunc(torch.autograd.Function):
            @staticmethod
            def forward(_, x):
                # 中断计算图
                torch._dynamo.graph_break()
                return x.sin()

        # 定义一个函数 fn，应用 AllowInGraphFunc 到输入 x 上
        def fn(x):
            return AllowInGraphFunc.apply(x)

        # 获取默认配置
        config = self.default_config()
        # 断言生成缓存键时会抛出 BypassAOTAutogradCache 异常
        self.assertRaises(
            BypassAOTAutogradCache, lambda: self.gen_cache_key(fn, config)
        )

    # 测试正常的 torch 函数
    def test_normal_torch_function(self):
        # 使用 torch._dynamo.allow_in_graph 装饰器定义一个函数 fn
        @torch._dynamo.allow_in_graph
        def fn(x):
            # 计算输入 x 的正弦和余弦，然后相加取绝对值
            y = torch.sin(x)
            z = torch.cos(x)
            w = y + z
            w.abs()
            return w

        # 获取默认配置
        config = self.default_config()
        # 生成缓存键，使用函数 fn、配置 config
        self.gen_cache_key(fn, config)
# 如果这个脚本是直接被执行的主程序
if __name__ == "__main__":
    # 导入并运行 torch._dynamo.test_case 模块中的 run_tests 函数
    from torch._dynamo.test_case import run_tests
    run_tests()
```