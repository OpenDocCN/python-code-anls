# `.\pytorch\test\inductor\test_codecache.py`

```
# Owner(s): ["module: inductor"]

# 导入所需的标准库和第三方库
import base64  # 提供对base64编解码的支持
import functools  # 提供高阶函数的支持，如partial函数
import json  # 提供JSON数据的编码和解码功能
import os  # 提供与操作系统交互的功能，如路径操作等
import pickle  # 提供对象序列化和反序列化的支持
import unittest  # 提供单元测试框架的支持
from typing import List  # 引入类型提示支持

import torch  # 引入PyTorch深度学习框架
from torch._dynamo import reset  # 导入torch._dynamo模块中的reset函数
from torch._dynamo.utils import counters  # 导入torch._dynamo.utils模块中的counters对象
from torch._inductor import config, metrics  # 导入torch._inductor模块中的config和metrics
from torch._inductor.async_compile import AsyncCompile  # 导入异步编译相关的AsyncCompile类
from torch._inductor.codecache import (  # 导入代码缓存相关的模块和类
    cuda_compile_command,
    CUDACodeCache,
    FxGraphCachePickler,
    FxGraphHashDetails,
    PyCodeCache,
    TensorMetadata,
    TensorMetadataAndValues,
)
from torch._inductor.runtime.runtime_utils import cache_dir  # 导入运行时工具中的cache_dir函数
from torch._inductor.test_case import run_tests, TestCase  # 导入测试相关的模块和类
from torch._inductor.utils import (  # 导入工具函数和清理函数
    clear_inductor_caches,
    fresh_inductor_cache,
)
from torch.testing._internal.common_cuda import SM80OrLater  # 导入CUDA相关的常量
from torch.testing._internal.common_device_type import largeTensorTest  # 导入设备类型相关的测试
from torch.testing._internal.common_utils import (  # 导入内部测试的工具函数
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (  # 导入inductor相关的测试工具
    GPU_TYPE,
    HAS_CUDA,
    HAS_GPU,
    HAS_MULTIGPU,
    requires_gpu,
)
from torch.utils._triton import has_triton  # 导入triton相关的工具函数和类

# 检查是否有triton库可用，并设置相应的标志
HAS_TRITON = has_triton()

# 如果有triton库可用，则导入triton模块
if HAS_TRITON:
    import triton  # 导入triton库

    from torch.testing._internal.triton_utils import add_kernel  # 导入triton测试相关的工具函数

# 定义一个装饰器函数，用于在没有triton库时跳过测试用例
requires_triton = functools.partial(unittest.skipIf, not HAS_TRITON, "requires triton")

# 设置torch._dynamo.config中的两个全局变量
torch._dynamo.config.fake_tensor_cache_enabled = True  # 启用假的张量缓存
torch._dynamo.config.fake_tensor_cache_crosscheck_enabled = True  # 启用假的张量缓存交叉检查

# 定义一个简单的神经网络模型类
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)  # 创建一个线性层

    def forward(self, inp):
        return self.fc1(inp)  # 执行前向传播


# 定义一个函数，用于执行代码缓存相关的测试
def _run_codecache_test(start_method):
    # 使用指定的启动方法修改torch._inductor.config中的worker_start_method和compile_threads参数
    with torch._inductor.config.patch(
        worker_start_method=start_method, compile_threads=16
    ):
        AsyncCompile.warm_pool()  # 预热编译池

        model = MyModel().to(device=GPU_TYPE)  # 创建一个MyModel实例并移动到指定设备
        model = torch.compile(model)  # 编译模型
        inp = torch.rand(10, 10).to(device=GPU_TYPE)  # 创建一个随机张量并移动到指定设备
        model(inp).sum().backward()  # 对模型输出执行反向传播求和


# 定义一个装饰器函数，用于声明需要GPU支持的测试用例
@requires_gpu()
def test_codecache_spawn():
    _run_codecache_test("spawn")  # 使用spawn启动方法运行代码缓存测试


# 定义一个装饰器函数，用于声明需要GPU支持的测试用例
@requires_gpu()
def test_codecache_fork():
    _run_codecache_test("fork")  # 使用fork启动方法运行代码缓存测试


# 定义一个简单的卷积神经网络模型类
class MyModelConv2d(torch.nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)  # 创建第一个卷积层
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)  # 创建第二个卷积层

    def forward(self, x):
        x = self.conv1(x)  # 执行第一个卷积操作
        torch._dynamo.graph_break()  # 断开动态图
        x = self.conv2(x)  # 执行第二个卷积操作
        return x  # 返回结果张量


# 使用instantiate_parametrized_tests装饰器声明一个参数化测试类
@instantiate_parametrized_tests
class TestFxGraphCache(TestCase):
    def setUp(self):
        super().setUp()  # 执行父类的setUp方法，准备测试环境
        counters.clear()  # 清空计数器

    def reset(self):
        torch._dynamo.reset()  # 重置动态图
        clear_inductor_caches()  # 清空inductor缓存

    @requires_triton()  # 声明需要triton支持的测试用例
    @config.patch({"fx_graph_cache": True})  # 启用fx_graph_cache配置
    @config.patch({"fx_graph_remote_cache": False})  # 禁用fx_graph_remote_cache配置
    @parametrize("device", (GPU_TYPE, "cpu"))  # 参数化测试，设备可以是GPU_TYPE或者"cpu"
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    def test_cache_load_function(self, device, dtype, dynamic):
        """
        Verify that we can populate and load functions from the cache.
        """
        # 如果设备为 GPU 类型但是没有 GPU，则跳过测试
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        # 如果设备为 cuda 并且数据类型为 torch.bfloat16 但不支持 SM80 或更新的 GPU，则跳过测试
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        # 定义一个简单的函数 fn，对输入进行操作并返回结果
        def fn(x, y):
            return (x * 2, y @ y)

        # 创建两个随机张量 a 和 b，根据设备和数据类型进行定义
        a = torch.rand(25, dtype=dtype, device=device)
        b = torch.rand(5, 5, dtype=dtype, device=device)

        # 使用 torch.compile 编译函数 fn，根据 dynamic 参数选择是否动态编译
        compiled_fn = torch.compile(fn, dynamic=dynamic)

        # 第一次调用应该未命中缓存
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        # 检查缓存未命中次数
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        # 检查缓存命中次数
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        # 检查查找写文件次数
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        # 第二次调用应该命中缓存。（首先重置内存保护以防止编译）
        # 删除所有缓存的编译模块文件
        for m in torch._inductor.codecache.PyCodeCache.cache.values():
            os.remove(m.__file__)
        self.reset()
        # 再次调用函数，此时应该命中缓存
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        # 检查缓存未命中次数仍然为 1
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        # 检查缓存命中次数增加到 1
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        # 检查查找写文件次数增加到 1
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)
    # 定义一个测试函数，用于测试远程缓存加载功能，接受设备类型、数据类型和动态编译参数
    def test_remote_cache_load_function(self, device, dtype, dynamic):
        # 导入 unittest.mock 模块中的 patch 函数
        from unittest.mock import patch

        # 如果设备要求 GPU 但没有 GPU 可用，则跳过测试
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        # 如果设备要求 CUDA 且数据类型为 torch.bfloat16 但不支持 SM80 或更高版本，则跳过测试
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        # 定义一个简单的函数 fn，用于对输入参数进行操作并返回结果
        def fn(x, y):
            return (x * 2, y @ y)

        # 创建具有指定设备和数据类型的随机张量 a 和 b
        a = torch.rand(25, dtype=dtype, device=device)
        b = torch.rand(5, 5, dtype=dtype, device=device)

        # 初始化一个空字典作为缓存
        cache = {}
        # 初始化获取和放置操作的计数器
        num_get = 0
        num_put = 0

        # 定义一个 MyCache 类，用于模拟缓存操作
        class MyCache:
            # 初始化方法，接受一个键和一个是否自动调整的标志位
            def __init__(self, key, is_autotune=False):
                pass

            # 获取缓存中指定文件名的数据
            def get(self, filename):
                nonlocal cache
                nonlocal num_get
                # 如果文件名不在缓存中，返回 None
                if filename not in cache:
                    return None
                # 从缓存中取出数据并解码为 JSON 格式
                ret = json.loads(cache[filename])
                num_get += 1
                # 如果配置是在 FB Code 环境下，则解码数据为 base64 编码形式
                if config.is_fbcode():
                    return base64.b64decode(ret["data"]) if ret is not None else ret
                else:
                    return base64.b64decode(ret) if ret is not None else ret

            # 将数据放入缓存中
            def put(self, filename, data):
                nonlocal cache
                nonlocal num_put
                # 如果配置是在 FB Code 环境下，则对数据进行特殊处理并编码为 base64 形式
                if config.is_fbcode():
                    data["data"] = base64.b64encode(data["data"]).decode("ascii")
                else:
                    data = base64.b64encode(data).decode("ascii")
                # 将处理后的数据以 JSON 格式存入缓存
                cache[filename] = json.dumps(data)
                num_put += 1

        # 根据当前环境选择远程缓存模块
        cache_module = (
            "triton.fb.fb_memcache.FbMemcacheRemoteFxGraphCacheBackend"
            if config.is_fbcode()
            else "torch._inductor.remote_cache.RedisRemoteCacheBackend"
        )

        # 使用 patch 函数修改配置和缓存模块，确保测试运行在指定环境下
        with config.patch(
            {
                "fx_graph_cache": False,
                "fx_graph_remote_cache": True,
            }
        ), patch.dict(os.environ), patch(cache_module, MyCache, create=True):
            # 移除环境变量中的 TRITON_CACHE_MANAGER 键
            os.environ.pop("TRITON_CACHE_MANAGER", None)
            # 执行循环，多次测试缓存相关功能
            for _ in range(4):
                # 使用 fresh_inductor_cache 上下文管理器重置缓存
                with fresh_inductor_cache():
                    # 编译函数 fn 并执行，比较执行结果
                    compiled_fn = torch.compile(fn, dynamic=dynamic)
                    self.assertEqual(fn(a, b), compiled_fn(a, b))
                # 重置环境状态
                reset()
            # 断言获取操作次数为 3
            self.assertEqual(num_get, 3)
            # 断言放置操作次数为 1
            self.assertEqual(num_put, 1)
    # 定义一个测试方法，用于验证能否从缓存中填充和加载模型
    def test_cache_load_model(self, device, dtype, dynamic):
        """
        Verify that we can populate and load models from the cache.
        """
        # 如果设备要求 GPU 但当前环境没有 GPU，则跳过测试
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # 定义一个函数 fn，用于执行模型的反向传播并返回参数的梯度
        def fn(mod, x):
            # 清空模型的梯度
            mod.zero_grad()
            # 执行模型的前向传播、反向传播并返回各参数的梯度
            mod(x).sum().backward()
            return [p.grad for p in mod.parameters()]

        # 使用 torch.compile 对函数 fn 进行编译，支持动态计算图
        compiled_fn = torch.compile(fn, dynamic=dynamic)

        # 创建一个 MyModelConv2d 的实例 mod，并将其移动到指定设备和数据类型上
        mod = MyModelConv2d().to(device=device, dtype=dtype)
        # 创建一个指定设备和数据类型的随机张量作为输入
        inp = torch.randn(2, 3, 16, 16, device=device, dtype=dtype)

        # 第一次调用应该看到所有的缓存未命中
        counters.clear()
        # 执行编译后的函数 compiled_fn，记录梯度信息到 grads1
        grads1 = compiled_fn(mod, inp)
        # 断言确保在 fxgraph_cache 中有未命中的计数
        self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
        # 断言确保在 fxgraph_cache 中命中的计数为 0
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # 第二次调用应该看到所有的命中（首先重置内存保护，以防止重新编译）
        counters.clear()
        self.reset()
        # 再次执行编译后的函数 compiled_fn，记录梯度信息到 grads2
        grads2 = compiled_fn(mod, inp)
        # 断言确保在 fxgraph_cache 中的未命中计数为 0
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        # 断言确保在 fxgraph_cache 中有命中的计数大于 0
        self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

        # 断言 grads1 和 grads2 的结果应该是相同的
        self.assertEqual(grads1, grads2)

    # 使用装饰器指定测试函数要求的大型张量条件，设备为 GPU_TYPE
    @largeTensorTest("64GB", device=GPU_TYPE)
    # 使用 config.patch 设置 fx_graph_cache 为 True
    @config.patch({"fx_graph_cache": True})
    # 使用 config.patch 设置 fx_graph_remote_cache 为 False
    @config.patch({"fx_graph_remote_cache": False})
    # 使用 parametrize 注入设备参数，设备为 GPU_TYPE
    @parametrize("device", (GPU_TYPE,))
    # 使用 parametrize 注入数据类型参数，数据类型为 torch.float16 和 torch.bfloat16
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    # 定义测试函数，测试在引入小于 int32 大小的张量条件下缓存相同图形。
    def test_cache_load_with_guards_int32_bounds(self, device, dtype):
        """
        Test caching the same graph, but under conditions that introduce guards
        for tensor sizes < int32.
        """
        
        # 如果设备是 GPU 类型但当前环境没有 GPU，跳过测试
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        
        # 如果设备是 CUDA 并且数据类型是 torch.bfloat16，但CUDA架构小于SM80，则跳过测试
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires CUDA SM80 or later")

        # 定义一个简单的函数 fn，返回输入的每个值加倍的元组
        def fn(x, y):
            return (x + x, y + y)

        # 使用 torch.compile 编译函数 fn，使其支持动态计算图
        compiled_fn = torch.compile(fn, dynamic=True)

        # 定义不同形状的张量对，测试是否在不同 int32 限制条件下缓存命中或未命中
        shapes = (
            ((5, 6), (7, 8)),
            ((5, 6), (47000, 47001)),
            ((47000, 47001), (5, 6)),
        )
        
        # 对每对形状进行迭代测试
        for a_shape, b_shape in shapes:
            # 创建随机张量 a 和 b，指定设备和数据类型
            a = torch.rand(a_shape, device=device, dtype=dtype)
            b = torch.rand(b_shape, device=device, dtype=dtype)

            # 清空计数器，用于记录缓存命中和未命中次数
            counters.clear()
            
            # 调用编译后的函数计算结果 res1
            res1 = compiled_fn(a, b)
            
            # 断言缓存未命中计数大于 0，表示预期缓存未命中
            self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
            
            # 断言缓存命中计数等于 0，表示预期缓存未命中
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            # 第二次调用相同函数应命中缓存（在此处强制重新编译）
            counters.clear()
            self.reset()  # 重置状态以强制重新编译
            res2 = compiled_fn(a, b)
            
            # 断言缓存未命中计数等于 0，表示预期缓存命中
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            
            # 断言缓存命中计数大于 0，表示预期缓存命中
            self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

            # 断言两次计算结果相等
            self.assertEqual(res1, res2)

    # 应用配置补丁，启用 fx_graph_cache，禁用 fx_graph_remote_cache
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    # 参数化测试设备，支持 GPU 类型和 CPU
    @parametrize("device", (GPU_TYPE, "cpu"))
    # 参数化数据类型，支持 torch.float32 和 torch.bfloat16
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_cache_load_with_guards_static_bounds(self, device, dtype):
        """
        Test caching the same graph, but under conditions that introduce guards
        for static bounds.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        # See lowering; for all of the pooling operators, we always guard and
        # make the height/width static.
        
        # 定义一个函数 fn，用于执行 adaptive_avg_pool2d 操作
        def fn(x):
            return torch.nn.functional.adaptive_avg_pool2d(x, [5, 7])

        # 编译函数 fn，以动态模式进行编译
        compiled_fn = torch.compile(fn, dynamic=True)

        # Iterate over different input shapes. Each new shape should cause
        # a cache miss.
        
        # 不同的输入形状进行迭代，每个新形状都应该导致缓存未命中
        shapes = ((1, 64, 8, 9), (1, 64, 9, 10), (1, 64, 10, 11))
        for shape in shapes:
            x = torch.rand(shape, device=device, dtype=dtype)

            # AVOID a dynamo reset here. For each cache hit, we expect guards
            # to have been added that will be violated with each new shape.
            
            # 避免在这里进行 Dynamo 重置。对于每次缓存命中，我们预期会添加保护措施，
            # 并且每个新形状都会违反这些保护措施。我们应该看到重新编译（并且缓存未命中）。
            counters.clear()
            res1 = compiled_fn(x)
            self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            # A second call should hit.
            
            # 第二次调用应该命中缓存
            counters.clear()
            self.reset()
            res2 = compiled_fn(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

            self.assertEqual(res1, res2)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_constant_handling(self, device):
        """
        Test that different constants are recognized correctly.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # 定义函数 fn1 和 fn2，用于不同的常量操作
        def fn1(x):
            return x + torch.tensor(list(range(0, 12)), device=device)

        def fn2(x):
            return x + torch.tensor(list(range(1, 13)), device=device)

        a = torch.rand(12, device=device)

        # 编译函数 fn1 和 fn2
        compiled_fn1 = torch.compile(fn1)
        compiled_fn2 = torch.compile(fn2)

        # A call to fn1 should miss in the cache.
        
        # 对 fn1 的调用应该导致缓存未命中
        self.assertEqual(fn1(a), compiled_fn1(a))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # A call to fn2 should also miss (the constant is different)
        
        # 对 fn2 的调用也应该导致缓存未命中（因为常量不同）
        self.assertEqual(fn2(a), compiled_fn2(a))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

    @requires_gpu()
    @requires_triton()
    # 应用配置补丁以启用本地 fx_graph 缓存
    @config.patch({"fx_graph_cache": True})
    # 应用配置补丁以禁用远程 fx_graph 缓存
    @config.patch({"fx_graph_remote_cache": False})
    # 定义测试函数 test_higher_order_op_bypass，用于验证在存在高阶操作时绕过缓存的行为
    def test_higher_order_op_bypass(self):
        """
        Verify that we bypass the cache when we have higher order ops.
        """

        # 定义内部函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 创建与 x 相同形状的零张量 output
            output = torch.zeros_like(x)
            # 计算 output 中元素的数量
            n_elements = output.numel()
            # 定义 lambda 函数 grid，用于生成计算网格
            grid = lambda meta: (
                # 使用 triton.cdiv 函数计算块数，n_elements 除以 meta["BLOCK_SIZE"]
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            # 调用 add_kernel 函数，传入 grid 计算结果和其他参数，计算结果存入 output
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)
            # 返回计算结果 output
            return output

        # 编译函数 fn，生成一个优化过的版本 compiled_fn
        compiled_fn = torch.compile(fn, fullgraph=True)

        # 生成随机张量 x 和 y，并传入 compiled_fn 中进行计算
        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        compiled_fn(x, y)

        # 断言缓存未命中的计数为 0
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        # 断言缓存命中的计数为 0
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        # 断言绕过缓存的计数大于 0
        self.assertGreater(counters["inductor"]["fxgraph_cache_bypass"], 0)

    # 应用配置补丁以启用本地 fx_graph 缓存
    @config.patch({"fx_graph_cache": True})
    # 应用配置补丁以禁用远程 fx_graph 缓存
    @config.patch({"fx_graph_remote_cache": False})
    # 定义测试函数 test_generated_kernel_count，测试在缓存命中时是否增加 generated_kernel_count 指标
    def test_generated_kernel_count(self):
        """
        Test that we bump the generated_kernel_count metric on a cache hit.
        """

        # 定义函数 fn，接受两个参数 x 和 y，并返回它们的乘积加上 y 的元组
        def fn(x, y):
            return (x * y + y,)

        # 创建两个随机张量 a 和 b
        a = torch.rand(5, 5)
        b = torch.rand(5, 5)

        # 编译函数 fn，生成一个优化过的版本 compiled_fn
        compiled_fn = torch.compile(fn)

        # 重置度量指标
        metrics.reset()
        # 断言 generated_kernel_count 指标为 0
        self.assertEqual(metrics.generated_kernel_count, 0)

        # 验证缓存未命中的情况
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        # 断言缓存命中的计数为 0
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        # 断言 generated_kernel_count 指标为 1
        self.assertEqual(metrics.generated_kernel_count, 1)

        # 验证缓存命中的情况
        self.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        # 断言缓存命中的计数为 1
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        # 断言 generated_kernel_count 指标为 2
        self.assertEqual(metrics.generated_kernel_count, 2)

    # 应用配置补丁以启用本地 fx_graph 缓存
    @config.patch({"fx_graph_cache": True})
    # 应用配置补丁以禁用远程 fx_graph 缓存
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_clear(self):
        """
        Test clearing the cache.
        """

        def fn(x, y):
            return (x * y,)

        a = torch.rand(5, 5)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn)

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        # 检查缓存未命中的计数器是否增加
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        # 检查缓存命中的计数器是否为零
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # A second call should hit.
        counters.clear()
        self.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        # 检查缓存未命中的计数器是否为零
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        # 检查缓存命中的计数器是否增加
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        # Clear the cache; now we should miss.
        counters.clear()
        self.reset()
        # 清空缓存
        torch._inductor.codecache.FxGraphCache.clear()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        # 检查缓存未命中的计数器是否增加
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        # 检查缓存命中的计数器是否为零

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_with_nt(self):
        def gen_nt(r):
            values = torch.randn(r, 16)
            offsets = torch.tensor([0, 2, 3, 6, 13, r])
            return torch.nested.nested_tensor_from_jagged(values, offsets)

        def fn(nt):
            if nt.values().size(0) % 16 == 0:
                return nt.sin()
            return nt.cos()

        inp1 = gen_nt(19)
        inp2 = gen_nt(20)

        counters.clear()
        # 编译函数 fn 并应用于输入 inp1 和 inp2
        torch.compile(fn)(inp1)
        torch.compile(fn)(inp2)
        # 检查缓存未命中的计数器是否增加
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        # 检查缓存命中的计数器是否为零
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.reset()
        counters.clear()
        torch.compile(fn)(inp1)
        torch.compile(fn)(inp2)
        # 检查缓存未命中的计数器是否为零
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        # 检查缓存命中的计数器是否增加
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_with_symint_non_arg_guard(self):
        def fn(x, ref_id):
            self_id = 22
            if self_id == ref_id:
                x = torch.mul(x, 1.0)
            else:
                x = torch.mul(x, 0)
            return x

        x = torch.ones(2)

        counters.clear()
        # 编译函数 fn 并应用于输入 x 和 ref_id=2
        torch.compile(fn, fullgraph=True, dynamic=True)(x, 2)
        # 检查缓存未命中的计数器是否增加
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        # 检查缓存命中的计数器是否为零
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.reset()
        counters.clear()
        torch.compile(fn, fullgraph=True, dynamic=True)(x, 2)
        # 检查缓存未命中的计数器是否为零
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        # 检查缓存命中的计数器是否增加
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
    # 将装饰器应用于测试方法，配置 fx_graph_cache 为 True
    @config.patch({"fx_graph_cache": True})
    # 将装饰器应用于测试方法，配置 fx_graph_remote_cache 为 False
    @config.patch({"fx_graph_remote_cache": False})
    # 定义测试方法 test_cache_guard
    def test_cache_guard(self):
        # 定义函数 f，接受参数 x 和 val
        def f(x, val):
            # 如果 val 大于 5，则调用 x 的 sin 方法并返回结果
            if val > 5:
                return x.sin()
            # 否则调用 x 的 cos 方法并返回结果
            else:
                return x.cos()

        # 创建一个包含两个元素的张量 x，元素值为 1
        x = torch.ones(2)
        # 使用动态编译模式编译函数 f，并传入参数 x 和 6，结果赋给 a
        a = torch.compile(f, dynamic=True)(x, 6)
        # 断言 fxgraph_cache_miss 计数为 1
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        # 断言 fxgraph_cache_hit 计数为 0
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # 重置测试环境
        self.reset()
        # 清空计数器
        counters.clear()
        # 使用动态编译模式编译函数 f，并传入参数 x 和 4，结果赋给 b
        b = torch.compile(f, dynamic=True)(x, 4)
        # 断言 fxgraph_cache_miss 计数为 1
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        # 断言 fxgraph_cache_hit 计数为 0
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # 断言 a 不等于 b
        self.assertNotEqual(a, b)
class TestFxGraphCacheHashing(TestCase):
    # 测试 FxGraphCacheHashing 类，用于测试张量常量的哈希值生成
    def test_tensor_constants(self):
        """
        Test the hashing of tensor constants.
        """
        # 序列化并获取张量常量的数据
        data = FxGraphCachePickler.dumps(torch.tensor(list(range(9))))
        # 确保反序列化后的数据类型为 TensorMetadataAndValues 类型
        self.assertIsInstance(pickle.loads(data), TensorMetadataAndValues)

    # 测试关键字参数在哈希时的特殊处理
    def test_hash_kwargs(self):
        """
        Test the special handling of the kwargs when hashing, i.e.,
        ordering of the kwargs dict and any set arguments.
        """
        # 测试关键字参数字典的顺序不影响哈希值
        details1 = FxGraphHashDetails(None, [], {"a": 0, "z": 1}, [])
        details2 = FxGraphHashDetails(None, [], {"z": 1, "a": 0}, [])
        self.assertEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

        # 不同的关键字参数值应影响哈希值
        details1 = FxGraphHashDetails(None, [], {"a": 0}, [])
        details2 = FxGraphHashDetails(None, [], {"a": 1}, [])
        self.assertNotEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

        # 集合的顺序不影响哈希值。集合是无序的，但排序和创建新集合可能会改变顺序。
        set1 = {"a", "b", "c", "d", "e", "f", "g"}
        set2 = set(sorted(set1))  # noqa: C414
        details1 = FxGraphHashDetails(None, [], {"a": set1}, [])
        details2 = FxGraphHashDetails(None, [], {"a": set2}, [])
        self.assertEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

        # 但不同的集合内容应影响哈希值。
        details1 = FxGraphHashDetails(None, [], {"a": {1, 2, 3}}, [])
        details2 = FxGraphHashDetails(None, [], {"a": {1, 2}}, [])
        self.assertNotEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

    # 测试不同的配置设置是否会影响哈希值
    def test_hash_config_changes(self):
        """
        Test that different config settings affect hashes.
        """
        # 使用 config.patch 上下文管理器设置不同的配置
        with config.patch({"max_autotune": False}):
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

        with config.patch({"max_autotune": True}):
            details3 = FxGraphHashDetails(None, [], {}, [])

        # 确保相同配置下生成的哈希值相同
        self.assertEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )
        # 确保不同配置下生成的哈希值不同
        self.assertNotEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details3),
        )

    # 如果没有 CUDA，则跳过测试；需要在 fbcode 环境下设置不同的 CUTLASS 路径
    @unittest.skipIf(not HAS_CUDA, "Requires CUDA")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    # 定义一个测试方法，用于测试 CUDA 编译命令生成函数 cuda_compile_command 的行为
    def test_cuda_compile_command(self):
        # 调用 cuda_compile_command 生成 CUDA 编译命令，不带额外参数
        cmd_no_extra_args: str = cuda_compile_command(
            ["abc.cu", "def.cu"], "output", "so"
        )
        # 断言生成的命令字符串中包含 "nvcc "
        assert "nvcc " in cmd_no_extra_args, cmd_no_extra_args
        # 断言生成的命令字符串中包含 "abc.cu"
        assert "abc.cu" in cmd_no_extra_args, cmd_no_extra_args
        # 断言生成的命令字符串中包含 "def.cu"
        assert "def.cu" in cmd_no_extra_args, cmd_no_extra_args
        # 断言生成的命令字符串中包含 "output"
        assert "output" in cmd_no_extra_args, cmd_no_extra_args
        
        # 调用 cuda_compile_command 生成 CUDA 编译命令，带额外参数
        cmd_extra_args: str = cuda_compile_command(
            ["abc.cu", "def.cu"], "output", "so", ["-Wwhatever", "-nothing"]
        )
        # 断言生成的命令字符串中包含 "nvcc "
        assert "nvcc " in cmd_extra_args, cmd_extra_args
        # 断言生成的命令字符串中包含 " -Wwhatever"
        assert " -Wwhatever" in cmd_extra_args, cmd_extra_args
        # 断言生成的命令字符串中包含 " -nothing"
        assert " -nothing" in cmd_extra_args, cmd_extra_args
        # 断言生成的命令字符串中包含 "abc.cu"
        assert "abc.cu" in cmd_extra_args, cmd_extra_args
        # 断言生成的命令字符串中包含 "def.cu"
        assert "def.cu" in cmd_extra_args, cmd_extra_args
        # 断言生成的命令字符串中包含 "output "
        assert "output " in cmd_extra_args, cmd_extra_args
        
        # 使用模拟对象 mock.patch 拦截 subprocess.check_output 的调用
        with mock.patch("subprocess.check_output") as check_output_mock:
            # 调用 CUDACodeCache.compile 编译 CUDA 代码 "test123.cu" 成为共享对象 "so"，带参数 ["-Wsomething"]
            CUDACodeCache.compile("test123.cu", "so", ["-Wsomething"])
            # 断言 check_output_mock 被调用
            check_output_mock.assert_called()
            # 获取 check_output_mock 调用时的参数列表 cmd_parts
            cmd_parts: List[str] = check_output_mock.call_args[0][0]
            # 断言 cmd_parts 的第一个元素是 "nvcc"
            assert cmd_parts[0] == "nvcc", cmd_parts
            # 断言 "-Wsomething" 在 cmd_parts 中
            assert "-Wsomething" in cmd_parts, cmd_parts
            # 断言 "-DNDEBUG" 在 cmd_parts 中
            assert "-DNDEBUG" in cmd_parts, cmd_parts
class TestUtils(TestCase):
    @config.patch({"fx_graph_remote_cache": False})
    def test_fresh_inductor_cache(self):
        # 定义一个简单的函数 fn，计算两个张量的和
        def fn(x, y):
            return x + y

        # 创建两个大小为 10 的随机张量
        a = torch.rand(10)
        b = torch.rand(10)

        # 使用 fresh_inductor_cache 上下文管理器，确保测试环境下的缓存为空
        with fresh_inductor_cache():
            # 断言 PyCodeCache 缓存中的键数量为 0
            self.assertEqual(len(PyCodeCache.cache.keys()), 0)
            # 编译函数 fn，并计算结果 res1
            res1 = torch.compile(fn)(a, b)
            # 获取当前的缓存目录路径
            cache_dir1 = cache_dir()

        # 重置 Torch 的一些状态
        torch._dynamo.reset()

        # 再次使用 fresh_inductor_cache 上下文管理器，确保缓存为空
        with fresh_inductor_cache():
            # 再次断言 PyCodeCache 缓存中的键数量为 0
            self.assertEqual(len(PyCodeCache.cache.keys()), 0)
            # 编译函数 fn，并计算结果 res2
            res2 = torch.compile(fn)(a, b)
            # 获取当前的缓存目录路径
            cache_dir2 = cache_dir()

        # 断言两次计算的结果应该相等
        self.assertEqual(res1, res2)
        # 断言两次获取的缓存目录路径不相等
        self.assertNotEqual(cache_dir1, cache_dir2)


if __name__ == "__main__":
    run_tests()
```