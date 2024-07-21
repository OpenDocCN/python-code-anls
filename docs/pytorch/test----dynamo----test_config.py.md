# `.\pytorch\test\dynamo\test_config.py`

```py
# Owner(s): ["module: dynamo"]

# 导入 PyTorch 库
import torch
# 导入 Dynamo 模块中的测试用例
import torch._dynamo.test_case
# 导入 Dynamo 模块中的测试工具
import torch._dynamo.testing
# 从 Dynamo 工具包中导入禁用缓存限制的函数
from torch._dynamo.utils import disable_cache_limit

# NB: do NOT include this test class in test_dynamic_shapes.py
# 注意：不要在 test_dynamic_shapes.py 中包含这个测试类

# 定义配置测试类，继承于 torch._dynamo.test_case.TestCase
class ConfigTests(torch._dynamo.test_case.TestCase):
    
    # 装饰器：禁用缓存限制
    @disable_cache_limit()
    # 定义测试函数 test_no_automatic_dynamic
    def test_no_automatic_dynamic(self):
        # 定义一个简单的函数 fn(a, b)，计算 a - b * 10
        def fn(a, b):
            return a - b * 10
        
        # 重置 Dynamo 状态
        torch._dynamo.reset()
        # 创建静态计数器 cnt_static
        cnt_static = torch._dynamo.testing.CompileCounter()
        
        # 进入 Dynamo 配置的上下文环境
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            # 优化函数 fn，并用计数器进行统计
            opt_fn = torch._dynamo.optimize(cnt_static)(fn)
            # 对范围为 [2, 11] 的整数循环
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # 断言静态计数器的帧数为 10
        self.assertEqual(cnt_static.frame_count, 10)

    # 装饰器：禁用缓存限制
    @disable_cache_limit()
    # 定义测试函数 test_automatic_dynamic
    def test_automatic_dynamic(self):
        # 定义一个简单的函数 fn(a, b)，计算 a - b * 10
        def fn(a, b):
            return a - b * 10
        
        # 重置 Dynamo 状态
        torch._dynamo.reset()
        # 创建动态计数器 cnt_dynamic
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        
        # 进入 Dynamo 配置的上下文环境
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=True
        ):
            # 优化函数 fn，并用计数器进行统计
            opt_fn = torch._dynamo.optimize(cnt_dynamic)(fn)
            # 对范围为 [2, 11] 的整数循环
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # 断言动态计数器的帧数为 2
        # 现在有两个图，而不是 10 个
        self.assertEqual(cnt_dynamic.frame_count, 2)

    # 装饰器：禁用缓存限制
    @disable_cache_limit()
    # 定义测试函数 test_no_assume_static_by_default
    def test_no_assume_static_by_default(self):
        # 定义一个简单的函数 fn(a, b)，计算 a - b * 10
        def fn(a, b):
            return a - b * 10
        
        # 重置 Dynamo 状态
        torch._dynamo.reset()
        # 创建动态计数器 cnt_dynamic
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        
        # 进入 Dynamo 配置的上下文环境
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            # 优化函数 fn，并用计数器进行统计
            opt_fn = torch._dynamo.optimize(cnt_dynamic)(fn)
            # 对范围为 [2, 11] 的整数循环
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # 断言动态计数器的帧数为 1
        # 现在只有一个图，因为我们没有等待重新编译
        self.assertEqual(cnt_dynamic.frame_count, 1)
    # 测试配置编译忽略功能
    def test_config_compile_ignored(self):
        # 如果不再相关，请从此列表中删除
        dynamo_guarded_config_ignorelist = {
            "log_file_name",  # 日志文件名
            "verbose",  # 详细模式
            "verify_correctness",  # 不会影响模型，但会引发运行时错误
            # （不会对编译行为造成静默更改）
            "cache_size_limit",  # 缓存大小限制
            "accumulated_cache_size_limit",  # 累积缓存大小限制
            "replay_record_enabled",  # 回放记录是否启用
            "cprofile",  # 只包装 _compile，不包括图形
            "repro_after",  # 重现后
            "repro_level",  # 重现级别
            "repro_forward_only",  # 仅前向重现
            "repro_tolerance",  # 重现容差
            "same_two_models_use_fp64",  # 相同的两个模型是否使用 fp64
            "error_on_recompile",  # 安全，因为：会抛出错误
            "report_guard_failures",  # 报告守护失败
            "base_dir",  # 用于最小化 / 记录日志
            "DEBUG_DIR_VAR_NAME",  # 调试目录变量名
            "debug_dir_root",  # 调试目录根路径
        }
        # 对于忽略列表中的每个键，确保其存在于编译忽略的键集合中
        for k in dynamo_guarded_config_ignorelist:
            assert k in torch._dynamo.config._compile_ignored_keys, k

    # 测试配置哈希功能
    def test_config_hash(self):
        config = torch._dynamo.config
        starting_hash = config.get_hash()

        # 使用不同配置参数修补配置
        with config.patch({"verbose": not config.verbose}):
            new_hash = config.get_hash()
            assert "verbose" in config._compile_ignored_keys
            assert new_hash == starting_hash

        new_hash = config.get_hash()
        assert new_hash == starting_hash

        # 再次使用不同的配置参数修补配置
        with config.patch({"dead_code_elimination": not config.dead_code_elimination}):
            changed_hash = config.get_hash()
            assert "dead_code_elimination" not in config._compile_ignored_keys
            assert changed_hash != starting_hash

            # 测试嵌套修补
            with config.patch({"verbose": not config.verbose}):
                inner_changed_hash = config.get_hash()
                assert inner_changed_hash == changed_hash
                assert inner_changed_hash != starting_hash

        newest_hash = config.get_hash()
        assert changed_hash != newest_hash
        assert newest_hash == starting_hash
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从torch._dynamo.test_case模块中导入run_tests函数
    from torch._dynamo.test_case import run_tests

    # 运行导入的run_tests函数，用于执行测试用例
    run_tests()
```