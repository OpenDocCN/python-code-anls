# `.\pytorch\test\dynamo\test_pre_dispatch.py`

```py
# Owner(s): ["module: dynamo"]
# 导入PyTorch库
import torch

# 导入私有模块及其测试用例
import torch._dynamo
import torch._dynamo.test_case

# 定义测试类PreDispatchTests，继承自torch._dynamo.test_case.TestCase
class PreDispatchTests(torch._dynamo.test_case.TestCase):

    # 测试函数：测试torch.no_grad()在简单函数中的使用
    def test_no_grad_simple(self):
        # 定义函数f，接受参数a
        def f(a):
            # 计算a的正弦值
            b = a.sin()
            # 进入torch.no_grad()上下文
            with torch.no_grad():
                # 计算b的余弦值
                c = b.cos()
            # 返回b乘以c的正弦值
            return b * c.sin()

        # 编译函数f，使用预调度执行后端
        f_compiled = torch.compile(f, backend="pre_dispatch_eager")

        # 创建随机张量a_ref，并保留其梯度
        a_ref = torch.randn(4, requires_grad=True)
        # 克隆张量a_ref，并分离计算图，同时保留梯度信息
        a_test = a_ref.clone().detach().requires_grad_(True)

        # 计算函数f在a_ref上的输出
        out_ref = f(a_ref)
        # 计算编译后的函数f在a_test上的输出
        out_test = f_compiled(a_test)

        # 断言两个输出张量的值相等
        self.assertEqual(out_ref, out_test)

        # 对out_ref的所有元素求和并反向传播梯度
        out_ref.sum().backward()
        # 对out_test的所有元素求和并反向传播梯度
        out_test.sum().backward()

        # 断言a_ref和a_test的梯度相等
        self.assertEqual(a_ref.grad, a_test.grad)

    # 测试函数：测试torch.no_grad()和torch.enable_grad()的嵌套使用
    def test_enable_grad_and_no_grad(self):
        # 定义函数f，接受参数a
        def f(a):
            # 计算a乘以2
            b = a * 2
            # 进入torch.no_grad()上下文
            with torch.no_grad():
                # 计算b乘以3
                c = b * 3
                # 进入torch.enable_grad()上下文
                with torch.enable_grad():
                    # 计算c乘以4
                    d = c * 4
                # 计算d乘以5
                e = d * 5
            # 返回b、c、d、e的和
            return b + c + d + e

        # 编译函数f，使用预调度执行后端
        f_compiled = torch.compile(f, backend="pre_dispatch_eager")

        # 创建随机张量a_ref，并保留其梯度
        a_ref = torch.randn(4, requires_grad=True)
        # 克隆张量a_ref，并分离计算图，同时保留梯度信息
        a_test = a_ref.clone().detach().requires_grad_(True)

        # 计算函数f在a_ref上的输出
        out_ref = f(a_ref)
        # 计算编译后的函数f在a_test上的输出
        out_test = f_compiled(a_test)

        # 断言两个输出张量的值相等
        self.assertEqual(out_ref, out_test)

        # 对out_ref的所有元素求和并反向传播梯度
        out_ref.sum().backward()
        # 对out_test的所有元素求和并反向传播梯度
        out_test.sum().backward()

        # 断言a_ref和a_test的梯度相等
        self.assertEqual(a_ref.grad, a_test.grad)

    # 测试函数：测试torch.amp.autocast()的简单使用
    def test_autocast_simple(self):
        # 定义函数f，接受参数a
        def f(a):
            # 计算a乘以2
            b = a * 2
            # 进入torch.amp.autocast()上下文，设备类型为CPU
            with torch.amp.autocast(device_type="cpu"):
                # 计算b的平方矩阵乘积
                c = torch.matmul(b, b)
            # 返回b与c的和
            return b + c

        # 编译函数f，使用预调度执行后端
        f_compiled = torch.compile(f, backend="pre_dispatch_eager")

        # 创建随机张量a_ref，在CPU上，保留其梯度
        a_ref = torch.randn(4, device="cpu", requires_grad=True)
        # 克隆张量a_ref，并分离计算图，同时保留梯度信息
        a_test = a_ref.clone().detach().requires_grad_(True)

        # 计算函数f在a_ref上的输出
        out_ref = f(a_ref)
        # 计算编译后的函数f在a_test上的输出
        out_test = f_compiled(a_test)

        # 断言两个输出张量的值相等
        self.assertEqual(out_ref, out_test)

        # 对out_ref的所有元素求和并反向传播梯度
        out_ref.sum().backward()
        # 对out_test的所有元素求和并反向传播梯度
        out_test.sum().backward()

        # 断言a_ref和a_test的梯度相等
        self.assertEqual(a_ref.grad, a_test.grad)


# 如果当前脚本为主程序，则运行测试
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
```