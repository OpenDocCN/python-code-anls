# `.\pytorch\test\dynamo\test_cudagraphs.py`

```
# Owner(s): ["module: cuda graphs"]

import functools  # 导入 functools 模块，用于函数装饰器
import unittest  # 导入 unittest 模块，用于编写和运行单元测试

import torch  # 导入 PyTorch 库

import torch._dynamo  # 导入 PyTorch 私有模块
import torch._dynamo.config  # 导入 PyTorch 私有模块中的 config 子模块
import torch._dynamo.test_case  # 导入 PyTorch 私有模块中的 test_case 子模块
import torch._dynamo.testing  # 导入 PyTorch 私有模块中的 testing 子模块
from torch._dynamo.testing import same  # 从 PyTorch 私有模块中的 testing 子模块导入 same 函数
from torch.testing._internal.common_utils import TEST_CUDA_GRAPH  # 导入 PyTorch 测试工具中的 TEST_CUDA_GRAPH 变量

# 定义一个装饰器函数 composed，接受多个装饰器，并将它们应用到被装饰函数上
def composed(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco

# 定义一个装饰器函数 assert_aot_autograd_counter，用于检查 AOT 自动求导计数器是否满足预期条件
def assert_aot_autograd_counter(ok=True):
    def deco(f):
        @functools.wraps(f)
        def wrap(self, *args, **kwargs):
            # 清除 AOT 自动求导计数器
            torch._dynamo.utils.counters.clear()
            r = f(self, *args, **kwargs)
            # 获取 AOT 自动求导计数器中的 ok 和 not_ok 计数
            c_ok = torch._dynamo.utils.counters["aot_autograd"]["ok"]
            c_not_ok = torch._dynamo.utils.counters["aot_autograd"]["not_ok"]
            # 根据 ok 参数判断计数器是否符合预期
            if ok:
                self.assertGreater(c_ok, 0)  # 断言 ok 计数大于 0
                self.assertEqual(c_not_ok, 0)  # 断言 not_ok 计数等于 0
            else:
                self.assertEqual(c_ok, 0)  # 断言 ok 计数等于 0
                self.assertGreater(c_not_ok, 0)  # 断言 not_ok 计数大于 0
            return r
        return wrap
    return deco

# 定义一个装饰器函数 patch_all，组合了 torch._dynamo.config.patch 和 assert_aot_autograd_counter 两个装饰器
def patch_all(ok=True):
    return composed(
        torch._dynamo.config.patch(
            verify_correctness=True, automatic_dynamic_shapes=True
        ),
        assert_aot_autograd_counter(ok),
    )

N_ITERS = 5  # 定义迭代次数常量

# 使用 unittest.skipIf 装饰器，如果 CUDA 不可用则跳过测试
@unittest.skipIf(not torch.cuda.is_available(), "these tests require cuda")
class TestAotCudagraphs(torch._dynamo.test_case.TestCase):
    # 使用 patch_all 装饰器修饰 test_basic 方法，应用于测试基本功能
    @patch_all()
    def test_basic(self):
        # 定义一个简单的模型函数 model，对输入进行简单的计算
        def model(x, y):
            return (x + y) * y
        
        # 使用 torch._dynamo.optimize 装饰器优化 fn 函数，采用 cudagraphs 优化策略
        @torch._dynamo.optimize("cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()
        
        x = torch.randn(3, device="cuda", requires_grad=True)  # 在 CUDA 设备上生成随机张量 x，并要求梯度计算
        y = torch.randn(3, device="cuda")  # 在 CUDA 设备上生成随机张量 y
        fn(x, y)  # 执行优化后的 fn 函数

    # 使用 patch_all 装饰器修饰 test_dtoh 方法，应用于测试数据从设备到主机的情况
    @patch_all()
    def test_dtoh(self):
        # 定义一个模型函数 model，对输入进行计算并将结果传输到 CPU 上进行处理
        def model(x, y):
            a = x + y
            b = a.cpu() * 3
            return b
        
        # 使用 torch._dynamo.optimize 装饰器优化 fn 函数，采用 cudagraphs 优化策略
        @torch._dynamo.optimize("cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()
        
        x = torch.randn(3, device="cuda", requires_grad=True)  # 在 CUDA 设备上生成随机张量 x，并要求梯度计算
        y = torch.randn(3, device="cuda")  # 在 CUDA 设备上生成随机张量 y
        fn(x, y)  # 执行优化后的 fn 函数

    # 使用 patch_all 装饰器修饰 test_htod 方法，应用于测试数据从主机到设备的情况
    @patch_all()
    def test_htod(self):
        # 定义一个模型函数 model，对输入进行计算并将结果传输到设备上
        def model(x, y):
            a = x + y
            return a * 3
        
        # 使用 torch._dynamo.optimize 装饰器优化 fn 函数，采用 cudagraphs 优化策略
        @torch._dynamo.optimize("cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()
        
        x = torch.randn(3, device="cuda", requires_grad=True)  # 在 CUDA 设备上生成随机张量 x，并要求梯度计算
        y = torch.randn((), device="cpu")  # 在 CPU 设备上生成随机张量 y
        fn(x, y)  # 执行优化后的 fn 函数
    @patch_all()
    # 使用装饰器 `patch_all()`，用于在测试执行前进行必要的全局补丁操作

    def test_mutate_input(self):
        # 定义测试方法 `test_mutate_input`，测试输入参数的变异性质

        def model(x, y):
            # 定义模型函数 `model`，接受输入参数 x 和 y
            y.add_(3)
            # 在 y 上执行就地加法操作，增加3
            return x * y
            # 返回 x 与 y 的乘积

        @torch._dynamo.optimize("cudagraphs")
        # 使用 PyTorch 的优化装饰器 `_dynamo.optimize("cudagraphs")`

        def fn(x, y):
            # 定义优化函数 `fn`，接受输入参数 x 和 y
            for i in range(N_ITERS):
                # 迭代 N_ITERS 次数
                with self.subTest(i):
                    # 使用单元测试的子测试 `subTest`，参数为 i
                    y_orig = y.clone()
                    # 克隆 y，保存原始值到 y_orig
                    loss = model(x, y).sum()
                    # 计算模型的损失值，对其求和
                    self.assertTrue(same(y, y_orig + 3))
                    # 断言 y 的值与 y_orig 加 3 相同
                    loss.backward()
                    # 反向传播计算梯度

        x = torch.randn(3, device="cuda", requires_grad=True)
        # 生成一个形状为 (3,) 的随机张量 x，位于 CUDA 设备上，需要梯度
        y = torch.randn(3, device="cuda")
        # 生成一个形状为 (3,) 的随机张量 y，位于 CUDA 设备上
        fn(x, y)
        # 调用 fn 函数，传入 x 和 y 进行测试

    @patch_all()
    # 使用装饰器 `patch_all()`，用于在测试执行前进行必要的全局补丁操作

    def test_mutate_constant(self):
        # 定义测试方法 `test_mutate_constant`，测试常量的变异性质

        def model(x, y):
            # 定义模型函数 `model`，接受输入参数 x 和 y
            c = torch.tensor(1)
            # 创建张量 c，值为 1
            c.add_(2)
            # 在 c 上执行就地加法操作，增加2（这里实际上没有对 c 造成影响）
            return x * y * 0 + c
            # 返回 x 与 y 的乘积再加上 c

        @torch._dynamo.optimize("cudagraphs")
        # 使用 PyTorch 的优化装饰器 `_dynamo.optimize("cudagraphs")`

        def fn(x, y):
            # 定义优化函数 `fn`，接受输入参数 x 和 y
            for i in range(N_ITERS):
                # 迭代 N_ITERS 次数
                with self.subTest(i):
                    # 使用单元测试的子测试 `subTest`，参数为 i
                    loss = model(x, y).sum()
                    # 计算模型的损失值，对其求和
                    self.assertTrue(same(loss, torch.tensor(3.0, device="cuda")))
                    # 断言 loss 的值与 torch.tensor(3.0, device="cuda") 相同
                    loss.backward()
                    # 反向传播计算梯度

        x = torch.randn(1, device="cuda", requires_grad=True)
        # 生成一个形状为 (1,) 的随机张量 x，位于 CUDA 设备上，需要梯度
        y = torch.randn(1, device="cuda")
        # 生成一个形状为 (1,) 的随机张量 y，位于 CUDA 设备上
        fn(x, y)
        # 调用 fn 函数，传入 x 和 y 进行测试

    @patch_all()
    # 使用装饰器 `patch_all()`，用于在测试执行前进行必要的全局补丁操作

    def test_factory(self):
        # 定义测试方法 `test_factory`，测试工厂函数的行为

        def model(y):
            # 定义模型函数 `model`，接受输入参数 y
            x = torch.zeros(3, device="cuda:0")
            # 创建一个形状为 (3,) 的全零张量 x，位于 CUDA 设备 "cuda:0"
            x.add_(3)
            # 在 x 上执行就地加法操作，增加3
            return x * y
            # 返回 x 与 y 的乘积

        @torch._dynamo.optimize("cudagraphs")
        # 使用 PyTorch 的优化装饰器 `_dynamo.optimize("cudagraphs")`

        def fn(y):
            # 定义优化函数 `fn`，接受输入参数 y
            for i in range(N_ITERS):
                # 迭代 N_ITERS 次数
                with self.subTest(i):
                    # 使用单元测试的子测试 `subTest`，参数为 i
                    loss = model(y).sum()
                    # 计算模型的损失值，对其求和
                    loss.backward()
                    # 反向传播计算梯度

        y = torch.randn(3, device="cuda:0", requires_grad=True)
        # 生成一个形状为 (3,) 的随机张量 y，位于 CUDA 设备 "cuda:0"，需要梯度
        fn(y)
        # 调用 fn 函数，传入 y 进行测试

    @patch_all()
    # 使用装饰器 `patch_all()`，用于在测试执行前进行必要的全局补丁操作

    def test_mutated_metadata(self):
        # 定义测试方法 `test_mutated_metadata`，测试元数据的变异行为

        # more tortured example at
        # https://github.com/pytorch/pytorch/issues/81385
        # 更复杂的示例，请参考上述链接

        def model(x):
            # 定义模型函数 `model`，接受输入参数 x
            x = x.clone()
            # 克隆输入的张量 x
            x.resize_(20)
            # 重新设置张量 x 的大小为 (20,)
            x.fill_(2)
            # 用值 2 填充张量 x
            return x

        @torch._dynamo.optimize("cudagraphs")
        # 使用 PyTorch 的优化装饰器 `_dynamo.optimize("cudagraphs")`

        def fn(x):
            # 定义优化函数 `fn`，接受输入参数 x
            for i in range(N_ITERS):
                # 迭代 N_ITERS 次数
                with self.subTest(i):
                    # 使用单元测试的子测试 `subTest`，参数为 i
                    rx = model(x)
                    # 调用模型函数，传入 x，获取结果 rx
                    self.assertTrue(same(rx, torch.full((20,), 2.0, device="cuda:0")))
                    # 断言 rx 的值与形状为 (20,)、值为 2.0 的张量相同

        x = torch.empty(0, device="cuda:0")
        # 创建一个形状为 (0,) 的空张量 x，位于 CUDA 设备 "cuda:0"
        fn(x)
        # 调用 fn 函数，传入 x 进行测试

    @patch_all()
    # 使用装饰器 `patch_all()`，用于在测试执行前进行必要的全局补丁操作

    def test_dead_fill(self):
        # 定义测试方法 `test_dead_fill`，测试填充操作对不同部分的影响

        def model(x):
            # 定义模型函数 `model`，接受输入参数 x
            x = x.clone()
            # 克隆输入的张量 x
            y = x[0:0]
            # 创建一个形状为 (0,) 的子张量 y，从 x 中取出
            x.fill_(2)
            # 用值 2 填充张量 x
            y.fill_(3)
            # 用值 3 填充子张量 y
            return x, y
            # 返回填充后的张量 x 和子张量 y

        @torch._dynamo.optimize("cudagraphs")
        # 使用 PyTorch 的优化装饰器 `_d
# 如果当前脚本作为主程序运行（而不是被导入作为模块）
if __name__ == "__main__":
    # 从torch._dynamo.test_case模块导入run_tests函数
    from torch._dynamo.test_case import run_tests

    # 如果未设置TEST_CUDA_GRAPH标志为True，则跳过CUDA图测试
    if not TEST_CUDA_GRAPH:
        # 如果当前脚本再次作为主程序运行
        if __name__ == "__main__":
            # 导入sys模块
            import sys

            # 终止程序并返回退出码0（正常退出）
            sys.exit(0)
        
        # 抛出unittest.SkipTest异常，提示跳过CUDA图测试
        raise unittest.SkipTest("cuda graph test is skipped")

    # 运行测试函数run_tests()
    run_tests()
```