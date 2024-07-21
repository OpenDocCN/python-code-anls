# `.\pytorch\test\dynamo\test_base_output.py`

```
# 导入模块 unittest.mock，用于模拟 unittest 中的对象
# 导入 torch 库
# 导入 torch._dynamo.test_case 模块
# 导入 torch._dynamo.testing 模块，以及其中的 same 函数

try:
    # 尝试导入 diffusers.models 中的 unet_2d 模块
    from diffusers.models import unet_2d
except ImportError:
    # 如果导入失败，则将 unet_2d 设置为 None
    unet_2d = None

# 定义一个装饰器函数 maybe_skip，用于根据 unet_2d 是否为 None 来跳过测试
def maybe_skip(fn):
    if unet_2d is None:
        # 如果 unet_2d 为 None，则返回一个跳过测试的装饰器
        return unittest.skip("requires diffusers")(fn)
    # 否则直接返回原函数
    return fn

# 定义一个测试类 TestBaseOutput，继承自 torch._dynamo.test_case.TestCase 类
class TestBaseOutput(torch._dynamo.test_case.TestCase):
    # 装饰器 maybe_skip 应用于 test_create 方法
    @maybe_skip
    def test_create(self):
        # 定义一个内部函数 fn，接收参数 a
        def fn(a):
            # 使用 unet_2d.UNet2DOutput 类创建一个对象 tmp，参数为 a + 1
            tmp = unet_2d.UNet2DOutput(a + 1)
            return tmp
        
        # 调用 torch._dynamo.testing.standard_test 函数进行标准化测试
        torch._dynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=1)

    # 装饰器 maybe_skip 应用于 test_assign 方法
    @maybe_skip
    def test_assign(self):
        # 定义一个内部函数 fn，接收参数 a
        def fn(a):
            # 使用 unet_2d.UNet2DOutput 类创建一个对象 tmp，参数为 a + 1
            tmp = unet_2d.UNet2DOutput(a + 1)
            # 设置 tmp 对象的 sample 属性为 a + 2
            tmp.sample = a + 2
            return tmp
        
        # 定义参数 args，包含一个长度为 10 的随机张量
        args = [torch.randn(10)]
        # 调用 torch._dynamo.testing.CompileCounter() 创建一个计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize_assert(cnts) 装饰 fn 函数，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        # 分别用 args 调用原始函数 fn 和优化后的函数 opt_fn，得到 obj1 和 obj2
        obj1 = fn(*args)
        obj2 = opt_fn(*args)
        # 断言 obj1.sample 和 obj2.sample 相等
        self.assertTrue(same(obj1.sample, obj2.sample))
        # 断言 cnts.frame_count 等于 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言 cnts.op_count 等于 2
        self.assertEqual(cnts.op_count, 2)

    # 定义一个内部方法 _common，接收 fn 和 op_count 两个参数
    def _common(self, fn, op_count):
        # 定义参数 args，包含一个使用 unet_2d.UNet2DOutput 创建的对象，sample 是一个长度为 10 的随机张量
        args = [
            unet_2d.UNet2DOutput(
                sample=torch.randn(10),
            )
        ]
        # 用 args 调用 fn 函数，得到 obj1
        obj1 = fn(*args)
        # 创建一个计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize_assert(cnts) 装饰 fn 函数，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        # 分别用 args 调用原始函数 fn 和优化后的函数 opt_fn，得到 obj2
        obj2 = opt_fn(*args)
        # 断言 obj1 和 obj2 相等
        self.assertTrue(same(obj1, obj2))
        # 断言 cnts.frame_count 等于 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言 cnts.op_count 等于 op_count
        self.assertEqual(cnts.op_count, op_count)

    # 装饰器 maybe_skip 应用于 test_getattr 方法
    @maybe_skip
    def test_getattr(self):
        # 定义一个内部函数 fn，接收参数 obj，类型为 unet_2d.UNet2DOutput
        def fn(obj: unet_2d.UNet2DOutput):
            # 计算 x = obj.sample * 10
            x = obj.sample * 10
            return x
        
        # 调用 self._common 方法进行通用测试，传入 fn 函数和 op_count = 1
        self._common(fn, 1)

    # 装饰器 maybe_skip 应用于 test_getitem 方法
    @maybe_skip
    def test_getitem(self):
        # 定义一个内部函数 fn，接收参数 obj，类型为 unet_2d.UNet2DOutput
        def fn(obj: unet_2d.UNet2DOutput):
            # 计算 x = obj["sample"] * 10
            x = obj["sample"] * 10
            return x
        
        # 调用 self._common 方法进行通用测试，传入 fn 函数和 op_count = 1
        self._common(fn, 1)

    # 装饰器 maybe_skip 应用于 test_tuple 方法
    @maybe_skip
    def test_tuple(self):
        # 定义一个内部函数 fn，接收参数 obj，类型为 unet_2d.UNet2DOutput
        def fn(obj: unet_2d.UNet2DOutput):
            # 调用 obj 的 to_tuple 方法，赋值给 a
            a = obj.to_tuple()
            # 返回 a[0] * 10
            return a[0] * 10
        
        # 调用 self._common 方法进行通用测试，传入 fn 函数和 op_count = 1
        self._common(fn, 1)

    # 装饰器 maybe_skip 应用于 test_index 方法
    @maybe_skip
    def test_index(self):
        # 定义一个内部函数 fn，接收参数 obj，类型为 unet_2d.UNet2DOutput
        def fn(obj: unet_2d.UNet2DOutput):
            # 返回 obj[0] * 10
            return obj[0] * 10
        
        # 调用 self._common 方法进行通用测试，传入 fn 函数和 op_count = 1
        self._common(fn, 1)

# 如果当前脚本作为主程序运行，则执行 from torch._dynamo.test_case import run_tests 运行测试
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
```