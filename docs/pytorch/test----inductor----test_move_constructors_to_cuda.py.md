# `.\pytorch\test\inductor\test_move_constructors_to_cuda.py`

```py
# Owner(s): ["module: inductor"]

import functools  # 导入 functools 模块，用于支持部分函数应用
import unittest  # 导入 unittest 模块，用于编写和运行单元测试

import torch  # 导入 PyTorch 深度学习框架
from torch._inductor.test_case import run_tests, TestCase  # 导入自定义模块中的函数和类
from torch._inductor.utils import run_and_get_code  # 导入自定义模块中的函数
from torch.testing import FileCheck  # 导入 PyTorch 的测试模块
from torch.testing._internal.common_cuda import TEST_MULTIGPU  # 导入多 GPU 测试标志
from torch.testing._internal.common_utils import IS_LINUX  # 导入操作系统类型标志
from torch.testing._internal.inductor_utils import HAS_CUDA  # 导入 CUDA 可用性标志

requires_multigpu = functools.partial(  # 定义一个装饰器，用于标记需要多个 CUDA 设备的测试用例
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)

aten = torch.ops.aten  # 设置 torch 的 aten 操作

class TestMoveConstructorsToCuda(TestCase):  # 定义测试类 TestMoveConstructorsToCuda，继承自 unittest.TestCase
    def _check_fn(self, func, expect_cpu, *args):
        out_eager = func(*args)  # 调用函数 func，并获取其返回结果

        out_compiled, code = run_and_get_code(torch.compile(func), *args)  # 编译函数 func，并获取编译后的结果和代码
        self.assertEqual(out_eager, out_compiled)  # 断言直接执行和编译执行的结果相等

        assert len(code) == 1  # 断言生成的代码只有一段
        if expect_cpu:
            FileCheck().check("cpp_fused").run(code[0])  # 如果期望在 CPU 上执行，使用 FileCheck 验证是否包含 "cpp_fused"
        else:
            FileCheck().check_not("cpp_fused").run(code[0])  # 如果不期望在 CPU 上执行，使用 FileCheck 验证是否不包含 "cpp_fused"

    def test_simple(self):
        def foo(x):
            return x[torch.arange(x.shape[0])]  # 返回输入张量 x 按行索引的结果

        inp = torch.rand(32, 77, 512, device="cuda")  # 创建在 CUDA 设备上的随机张量

        self._check_fn(foo, False, inp)  # 调用 _check_fn 进行测试

    def test_output_failure(self):
        def foo(x):
            tmp1 = torch.arange(x.shape[0])
            return tmp1, x[tmp1]  # 返回索引张量和对应索引的结果张量

        inp = torch.rand(32, 77, 512, device="cuda")  # 创建在 CUDA 设备上的随机张量

        self._check_fn(foo, True, inp)  # 调用 _check_fn 进行测试

    def test_non_convertable_op_failure(self):
        def foo(x):
            y = torch.arange(x.shape[0])
            return x + y, torch.ones([4], device="cuda")  # 返回张量 x 与索引张量 y 相加的结果和在 CUDA 设备上的全 1 张量

        inp = torch.rand([100])  # 创建形状为 [100] 的随机张量

        self._check_fn(foo, True, inp)  # 调用 _check_fn 进行测试

    def test_multiple_constructors(self):
        def foo(x):
            tmp1 = torch.arange(x.shape[0])
            o1 = x[tmp1]  # 使用索引张量 tmp1 从输入张量 x 中获取结果张量 o1
            tmp2 = torch.arange(x.shape[1]).view([1, x.shape[1]])
            o2 = x[tmp2]  # 使用索引张量 tmp2 从输入张量 x 中获取结果张量 o2
            return o1, o2, o1 + o2  # 返回 o1、o2 和它们的和

        inp = torch.rand([200, 200])  # 创建形状为 [200, 200] 的随机张量
        self._check_fn(foo, True, inp)  # 调用 _check_fn 进行测试

    def test_sets_equiv(self):
        @torch.compile()  # 定义一个装饰器，用于编译函数
        def foo(x):
            c1 = torch.ones([4], dtype=torch.long)  # 创建一个值全为 1 的长整型张量 c1
            c2 = torch.arange(-1, 3)  # 创建一个从 -1 到 2 的张量 c2
            return x[c1 + c2], c2 - 4 * 2  # 返回使用 c1 + c2 索引的结果张量和 c2 - 8 的结果张量

        inp = torch.rand([4]).cuda()  # 创建在 CUDA 设备上的形状为 [4] 的随机张量
        out, code = run_and_get_code(foo, inp)  # 运行并获取函数 foo 的输出和生成的代码
        FileCheck().check_not("triton.jit").run(code[0])  # 使用 FileCheck 验证代码中不包含 "triton.jit"

        @torch.compile()  # 定义一个装饰器，用于编译函数
        def foo(x):
            c2 = torch.arange(-1, 3)  # 创建一个从 -1 到 2 的张量 c2
            c1 = torch.ones([4], dtype=torch.long)  # 创建一个值全为 1 的长整型张量 c1
            return x[c1 + c2], c2 - 4 * 2  # 返回使用 c1 + c2 索引的结果张量和 c2 - 8 的结果张量

        out, code = run_and_get_code(foo, inp)  # 运行并获取函数 foo 的输出和生成的代码
        FileCheck().check_not("triton.jit").run(code[0])  # 使用 FileCheck 验证代码中不包含 "triton.jit"

    @requires_multigpu()  # 标记此测试用例需要多个 CUDA 设备
    def test_multi_gpu(self):
        def foo(x):
            return (
                x[torch.arange(x.shape[0])],  # 返回按行索引的输入张量 x
                torch.ones([4], device="cuda:0"),  # 在第一个 CUDA 设备上创建一个全 1 张量
                torch.ones([4], device="cuda:1"),  # 在第二个 CUDA 设备上创建一个全 1 张量
            )

        # nyi, multi-gpu  # 暂未实现，多 GPU
        inp = torch.rand([100], device="cuda")  # 创建在 CUDA 设备上的形状为 [100] 的随机张量
        self._check_fn(foo, True, inp)  # 调用 _check_fn 进行测试
    # 定义一个名为 test_no_gpu 的测试方法，用于测试在无 GPU 环境下的情况
    def test_no_gpu(self):
        # 定义一个内部函数 foo，接受参数 x，并返回 x 中对应索引的元素
        def foo(x):
            return x[torch.arange(x.shape[0])]
        
        # 创建一个包含 100 个随机数的张量 inp
        inp = torch.rand([100])
        # 调用类的内部方法 _check_fn，验证函数 foo 的输出是否符合预期
        self._check_fn(foo, True, inp)
# 如果当前脚本是作为主程序运行
if __name__ == "__main__":
    # 如果操作系统是Linux并且有CUDA支持
    if IS_LINUX and HAS_CUDA:
        # 运行测试函数
        run_tests()
```