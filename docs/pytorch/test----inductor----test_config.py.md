# `.\pytorch\test\inductor\test_config.py`

```py
# Owner(s): ["module: inductor"]

# 导入数学库
import math
# 导入单元测试框架
import unittest

# 导入 PyTorch 库
import torch

# 导入自定义的配置模块
from torch._inductor import config

# 导入自定义的测试运行函数和测试用例类
from torch._inductor.test_case import run_tests, TestCase

# 导入内部工具函数，检查是否有 CPU
from torch.testing._internal.inductor_utils import HAS_CPU


# 定义一个简单的函数，对输入 x 进行操作并返回结果
def dummy_fn(x):
    return torch.sigmoid(x + math.pi) / 10.0


# 定义一个继承自 torch.nn.Module 的简单模块类
class DummyModule(torch.nn.Module):
    def forward(self, x):
        return dummy_fn(x)


# 定义一个用于测试配置模块的测试类
class TestInductorConfig(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # 保存当前的配置状态
        cls._saved_config = config.save_config()

    def tearDown(self):
        super().tearDown()
        # 恢复之前保存的配置状态
        config.load_config(self._saved_config)

    # 测试设置配置参数的方法
    def test_set(self):
        # 设置最大融合大小为 13337
        config.max_fusion_size = 13337
        self.assertEqual(config.max_fusion_size, 13337)
        # 使用浅拷贝方法检查配置参数的值
        self.assertEqual(config.shallow_copy_dict()["max_fusion_size"], 13337)
        # 再次设置最大融合大小为 32
        config.max_fusion_size = 32
        self.assertEqual(config.max_fusion_size, 32)

        # 测试嵌套配置参数的设置
        prior = config.triton.cudagraphs
        config.triton.cudagraphs = not prior
        self.assertEqual(config.triton.cudagraphs, not prior)
        self.assertEqual(config.shallow_copy_dict()["triton.cudagraphs"], not prior)

    # 测试保存和加载配置的方法
    def test_save_load(self):
        config.max_fusion_size = 123
        config.triton.cudagraphs = True
        saved1 = config.save_config()
        config.max_fusion_size = 321
        config.triton.cudagraphs = False
        saved2 = config.save_config()

        self.assertEqual(config.max_fusion_size, 321)
        self.assertEqual(config.triton.cudagraphs, False)
        config.load_config(saved1)
        self.assertEqual(config.max_fusion_size, 123)
        self.assertEqual(config.triton.cudagraphs, True)
        config.load_config(saved2)
        self.assertEqual(config.max_fusion_size, 321)
        self.assertEqual(config.triton.cudagraphs, False)

    # 测试检查是否存在指定属性的方法
    def test_hasattr(self):
        self.assertTrue(hasattr(config, "max_fusion_size"))
        self.assertFalse(hasattr(config, "missing_name"))

    # 测试试图访问不存在属性时是否会引发异常的方法
    def test_invalid_names(self):
        self.assertRaises(AttributeError, lambda: config.does_not_exist)
        self.assertRaises(AttributeError, lambda: config.triton.does_not_exist)

        def store1():
            config.does_not_exist = True

        def store2():
            config.triton.does_not_exist = True

        self.assertRaises(AttributeError, store1)
        self.assertRaises(AttributeError, store2)
    # 定义一个测试方法 test_patch，用于测试配置的 patch 功能
    def test_patch(self):
        # 使用配置 patch 设置最大融合大小为 456
        with config.patch(max_fusion_size=456):
            # 断言当前配置的最大融合大小为 456
            self.assertEqual(config.max_fusion_size, 456)
            # 在内部 patch 中设置最大融合大小为 789，验证是否生效
            with config.patch(max_fusion_size=789):
                # 断言当前配置的最大融合大小为 789
                self.assertEqual(config.max_fusion_size, 789)
            # 内部 patch 结束后，再次验证最大融合大小是否回到外部的设置值 456
            self.assertEqual(config.max_fusion_size, 456)

        # 使用配置 patch 设置多个参数，包括 cpp.threads 和 max_fusion_size
        with config.patch({"cpp.threads": 9000, "max_fusion_size": 9001}):
            # 断言配置中 cpp.threads 的值为 9000
            self.assertEqual(config.cpp.threads, 9000)
            # 断言配置中 max_fusion_size 的值为 9001
            self.assertEqual(config.max_fusion_size, 9001)
            # 在内部 patch 中修改 cpp.threads 的值为 8999，验证是否生效
            with config.patch("cpp.threads", 8999):
                # 断言当前配置的 cpp.threads 值为 8999
                self.assertEqual(config.cpp.threads, 8999)
            # 内部 patch 结束后，再次验证 cpp.threads 是否回到外部的设置值 9000
            self.assertEqual(config.cpp.threads, 9000)

    # 定义一个测试方法 test_compile_api，用于测试编译 API 的行为
    @unittest.skipIf(not HAS_CPU, "requires C++ compiler")
    def test_compile_api(self):
        # 这些检查主要用于验证配置处理时不会因异常而崩溃
        x = torch.randn(8)
        y = dummy_fn(x)
        # 定义多个配置字典，用于不同的编译参数测试
        checks = [
            {},
            {"mode": "default"},
            {"mode": "reduce-overhead"},
            {"mode": "max-autotune"},
            {
                "options": {
                    "max-fusion-size": 128,
                    "unroll_reductions_threshold": 32,
                    "triton.cudagraphs": False,
                }
            },
            {"dynamic": True},
            {"fullgraph": True, "backend": "inductor"},
            {"disable": True},
        ]

        # 遍历不同的配置参数字典，进行编译函数的测试
        for kwargs in checks:
            # 重置 Torch 的动态图状态
            torch._dynamo.reset()
            # 调用 torch.compile 函数进行编译，传入不同的参数字典 kwargs
            opt_fn = torch.compile(dummy_fn, **kwargs)
            # 使用测试断言验证编译函数的输出是否正确，如果不正确则抛出异常
            torch.testing.assert_allclose(
                opt_fn(x), y, msg=f"torch.compile(..., **{kwargs!r}) failed"
            )
    # 测试获取编译器配置的方法
    def test_get_compiler_config(self):
        # 导入 torch._inductor 中的配置作为 inductor_default_config
        from torch._inductor import config as inductor_default_config

        # 从 inductor_default_config 中获取 triton.cudagraphs 的默认值
        default_cudagraphs = inductor_default_config._default["triton.cudagraphs"]

        # 使用 DummyModule 创建一个模型实例 model
        model = DummyModule()
        
        # 编译模型 model，并使用给定的选项更新 triton.cudagraphs 的默认配置
        optimized_module = torch.compile(
            model, options={"triton.cudagraphs": not default_cudagraphs}
        )
        
        # 获取优化后模型的编译器配置
        compiler_config = optimized_module.get_compiler_config()
        
        # 断言编译器配置中 triton.cudagraphs 的值与预期相符
        self.assertEqual(compiler_config["triton.cudagraphs"], not default_cudagraphs)

        # 使用 DummyModule 创建另一个模型实例 model
        model = DummyModule()
        
        # 编译模型 model，并保持默认配置不变
        optimized_module = torch.compile(model)
        
        # 再次获取优化后模型的编译器配置
        compiler_config = optimized_module.get_compiler_config()
        
        # 断言编译器配置中 triton.cudagraphs 的值与默认配置相符
        self.assertEqual(
            compiler_config["triton.cudagraphs"],
            default_cudagraphs,
        )

        # 编译用户定义的函数 dummy_fn，使用给定的选项更新 triton.cudagraphs 的默认配置
        optimized_module = torch.compile(
            dummy_fn, options={"triton.cudagraphs": not default_cudagraphs}
        )
        
        # 获取优化后模块的编译器配置
        compiler_config = optimized_module.get_compiler_config()
        
        # 断言编译器配置中 triton.cudagraphs 的值与预期相符
        self.assertEqual(compiler_config["triton.cudagraphs"], not default_cudagraphs)

        # 编译用户定义的函数 dummy_fn，并保持默认配置不变
        optimized_module = torch.compile(dummy_fn)
        
        # 获取优化后模块的编译器配置
        compiler_config = optimized_module.get_compiler_config()
        
        # 断言编译器配置中 triton.cudagraphs 的值与默认配置相符
        self.assertEqual(
            compiler_config["triton.cudagraphs"],
            default_cudagraphs,
        )

        # 使用 eager 后端编译用户定义的函数 dummy_fn，预期编译器配置为 None
        optimized_module = torch.compile(dummy_fn, backend="eager")
        compiler_config = optimized_module.get_compiler_config()
        
        # 断言编译器配置为 None
        self.assertTrue(compiler_config is None)

    # 测试编译 API 是否正确传递配置
    def test_compile_api_passes_config(self):
        # 确保配置实际上被传递到 inductor
        self.assertRaises(
            # 如果编译失败，期望抛出 torch._dynamo.exc.BackendCompilerFailed 异常
            torch._dynamo.exc.BackendCompilerFailed,
            # 使用 options={"_raise_error_for_testing": True} 来编译 dummy_fn，然后执行 torch.randn(10)
            lambda: torch.compile(dummy_fn, options={"_raise_error_for_testing": True})(
                torch.randn(10)
            ),
        )
    # 测试 API 选项函数，验证不同模式选项的返回结果

    # 获取 "reduce-overhead" 模式的选项字典
    reduce_overhead_opts = torch._inductor.list_mode_options("reduce-overhead")
    # 断言选项中是否包含 "triton.cudagraphs" 为 True
    self.assertEqual(reduce_overhead_opts["triton.cudagraphs"], True)
    # 断言选项中 "max_autotune" 不存在或者为 False
    self.assertEqual(reduce_overhead_opts.get("max_autotune", False), False)

    # 获取 "max-autotune" 模式的选项字典
    max_autotune_opts = torch._inductor.list_mode_options("max-autotune")
    # 断言选项中 "max_autotune" 存在且为 True
    self.assertEqual(max_autotune_opts["max_autotune"], True)
    # 断言选项中 "triton.cudagraphs" 存在且为 True
    self.assertEqual(max_autotune_opts["triton.cudagraphs"], True)

    # 获取 "max-autotune" 模式的选项字典，动态选项为 True
    max_autotune_opts = torch._inductor.list_mode_options(
        "max-autotune", dynamic=True
    )
    # 断言选项中 "max_autotune" 存在且为 True
    self.assertEqual(max_autotune_opts["max_autotune"], True)
    # 断言选项中 "triton.cudagraphs" 存在且为 True
    self.assertEqual(max_autotune_opts["triton.cudagraphs"], True)

    # 获取 "max-autotune-no-cudagraphs" 模式的选项字典
    max_autotune_no_cudagraphs_opts = torch._inductor.list_mode_options(
        "max-autotune-no-cudagraphs"
    )
    # 断言选项中 "max_autotune" 存在且为 True
    self.assertEqual(max_autotune_no_cudagraphs_opts["max_autotune"], True)
    # 断言选项中 "triton.cudagraphs" 不存在或者为 False
    self.assertEqual(
        max_autotune_no_cudagraphs_opts.get("triton.cudagraphs", False), False
    )

# 测试无效后端时是否能够触发 InvalidBackend 异常
def test_invalid_backend(self):
    self.assertRaises(
        torch._dynamo.exc.InvalidBackend,
        lambda: torch.compile(dummy_fn, backend="does_not_exist")(torch.randn(10)),
    )

# 测试非 Inductor 后端的函数
def test_non_inductor_backend(self):
    # 定义一个函数，用于断言期望的模式和选项
    def assert_options(expected_mode=None, expected_options=None):
        # 定义一个后端函数
        def backend(gm, _, *, mode=None, options=None):
            nonlocal call_count
            # 断言模式是否符合预期
            self.assertEqual(mode, expected_mode)
            # 断言选项是否符合预期
            self.assertEqual(options, expected_options)
            call_count += 1
            return gm

        return backend

    inp = torch.randn(8)

    def fn(x):
        return x + 1

    # 遍历不同模式和选项的组合，进行测试
    for mode, options in [
        (None, None),
        ("fast-mode", None),
        (None, {"foo": "bar"}),
    ]:
        call_count = 0
        # 编译函数 fn，使用自定义的后端函数作为 backend，同时传入模式和选项
        torch.compile(
            fn, backend=assert_options(mode, options), mode=mode, options=options
        )(inp)
        # 重置动态计算图编译器的状态
        torch._dynamo.reset()
        # 断言调用次数为 1
        self.assertEqual(call_count, 1)

    # 测试触发 BackendCompilerFailed 异常，使用不支持的关键字参数 'mode'
    # TypeError: eager() got an unexpected keyword argument 'mode'
    self.assertRaises(
        torch._dynamo.exc.BackendCompilerFailed,
        lambda: torch.compile(fn, backend="eager", mode="nope")(inp),
    )
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```