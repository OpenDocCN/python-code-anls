# `.\pytorch\test\inductor\test_layout_optim.py`

```
# Owner(s): ["module: inductor"]
# 导入所需的库和模块
import copy  # 导入 copy 模块，用于复制对象
import os    # 导入 os 模块，用于操作系统相关功能
import random  # 导入 random 模块，用于生成随机数

import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块
from torch._dynamo.utils import same  # 导入 torch._dynamo.utils 模块中的 same 函数
from torch._inductor import config  # 导入 torch._inductor 中的 config 模块
from torch._inductor.test_case import run_tests, TestCase  # 导入测试相关的模块和类
from torch.testing._internal.common_cuda import tf32_off  # 导入 CUDA 相关的模块
from torch.testing._internal.inductor_utils import HAS_CUDA  # 导入测试工具相关的 CUDA 支持判断函数

USE_DDP_WRAPPER = os.environ.get("USE_DDP_WRAPPER", "1") == "1"

class Model2Conv(nn.Module):
    def __init__(self, dim=512, manual_graph_break=False):
        super().__init__()
        # 初始化两个卷积层，输入通道数为3，输出通道数为dim，卷积核大小为3x3，步长为2，无偏置
        self.conv1 = nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)
        self.manual_graph_break = manual_graph_break  # 是否手动中断计算图

    def forward(self, x):
        x = self.conv1(x)  # 第一个卷积层的前向传播
        if self.manual_graph_break:
            torch._dynamo.graph_break()  # 如果设置了手动中断计算图，调用中断函数
        x = self.conv2(x)  # 第二个卷积层的前向传播
        return x

    def get_example_inputs(self):
        return (torch.rand(2, 3, 16, 16),)  # 返回一个示例输入的张量元组


class TestLayoutOptim(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        import torch.distributed as dist  # 导入分布式训练模块

        # 尝试多次初始化分布式训练进程组，直到成功或达到尝试次数上限
        tot_retry = 5
        for retry_no in range(tot_retry):
            try:
                port = random.randint(10000, 60000)  # 随机生成一个端口号
                dist.init_process_group(
                    backend="nccl",  # 使用 NCCL 后端
                    init_method=f"tcp://localhost:{port}",  # 使用 TCP 协议初始化，端口号动态生成
                    world_size=1,  # 总的训练进程数为1
                    rank=0,  # 当前进程的排名为0
                )
                break  # 成功初始化，退出循环
            except RuntimeError:
                if retry_no == tot_retry - 1:
                    raise  # 如果达到尝试次数上限仍然失败，则抛出异常
                else:
                    continue  # 继续尝试下一个端口号

    def verify_accuracy(
        self, model_class, use_ddp_wrapper=USE_DDP_WRAPPER, is_train=False
        # 验证模型精度的方法，接受模型类和其他参数作为输入
    ):
        # 如果要引入图断点有两种潜在的方法
        # 1. 手动
        # 2. 使用 DDP
        # 如果我们不使用 DDP 来引入图断点，则手动进行
        def wrap_mod(m):
            if is_train:
                # 定义一个函数 f，接受任意数量的输入
                def f(*inp):
                    # 调用模型 m 处理输入 inp，计算输出并进行反向传播
                    x = m(*inp)
                    x.sum().backward()

                    # 收集所有参数的梯度
                    grads = []
                    for name, param in m.named_parameters():
                        grad = param.grad
                        if param.grad is None:
                            grad = torch.zeros_like(param)
                        grads.append(grad)
                    return grads

                return f
            else:
                return m

        # 根据是否使用 DDP 包装模型
        manual_graph_break = not use_ddp_wrapper
        # 创建模型对象 mod，并将其移动到 GPU 上
        mod = model_class(manual_graph_break=manual_graph_break).cuda()
        # 准备模型的示例输入，并将其移动到 GPU 上
        inp = [t.cuda() for t in mod.get_example_inputs()]
        # 对模型进行包装处理，预期输出与包装后的模型处理输入的结果
        expected_out = wrap_mod(mod)(*inp)

        # 创建模型的深拷贝，并转换为 float64 类型
        fp64_mod = copy.deepcopy(mod).to(torch.float64)
        # 准备 float64 类型的示例输入
        fp64_inp = [t.to(torch.float64) for t in copy.deepcopy(inp)]
        # 对 float64 类型的模型进行包装处理，预期输出与包装后的模型处理输入的结果
        fp64_out = wrap_mod(fp64_mod)(*fp64_inp)

        # 如果使用 DDP 包装模型
        if use_ddp_wrapper:
            from torch.nn.parallel import DistributedDataParallel as DDP

            # 使用 DDP 包装模型 mod
            ddp_wrapped_mod = DDP(mod)
            # 对包装后的模型进行优化
            opt_mod = torch.compile(wrap_mod(ddp_wrapped_mod))
        else:
            # 对模型进行优化
            opt_mod = torch.compile(wrap_mod(mod))
        
        # 使用原始输入进行模型优化，计算实际输出
        actual_out = opt_mod(*inp)

        # 如果是训练模式
        if is_train:
            # 断言预期输出与实际输出的一致性
            self.assertTrue(same(expected_out, actual_out, fp64_ref=fp64_out))
        else:
            # 计算预期输出与实际输出的总和
            expected_sum = expected_out.sum()
            actual_sum = actual_out.sum()
            # 打印预期总和和实际总和
            print(f"Expected sum {expected_sum}, actual sum {actual_sum}")
            # 断言预期输出与实际输出的一致性
            self.assertTrue(same(expected_out, actual_out, fp64_ref=fp64_out))

    # 为推断验证准确性定义一个方法，调用 verify_accuracy 方法
    def verify_accuracy_for_infer(self, *args, **kwargs):
        self.verify_accuracy(*args, **kwargs, is_train=False)

    # 为训练验证准确性定义一个方法，调用 verify_accuracy 方法
    def verify_accuracy_for_train(self, *args, **kwargs):
        self.verify_accuracy(*args, **kwargs, is_train=True)

    # 测试带有图断点的 2 层卷积模型
    def test_2conv_with_graph_break(self):
        """
        确保图断点不会导致任何准确性问题。
        """
        # 调用 verify_accuracy_for_infer 方法验证 2 层卷积模型的准确性
        self.verify_accuracy_for_infer(Model2Conv)
    def test_3conv_with_graph_break(self):
        # 定义一个名为 Model 的神经网络模型类
        class Model(nn.Module):
            # 初始化函数，设置模型参数和层
            def __init__(
                self, dim=512, patch_size=7, kernel_size=7, manual_graph_break=False
            ):
                super().__init__()
                # 构建一个序列模块，包括两个二维卷积层
                self.seq = nn.Sequential(
                    nn.Conv2d(
                        3, dim, kernel_size=patch_size, stride=patch_size, bias=False
                    ),
                    nn.Conv2d(
                        dim, dim, kernel_size, groups=dim, padding="same", bias=False
                    ),
                )
                # 单独定义一个二维卷积层
                self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
                # 是否手动中断计算图的标志
                self.manual_graph_break = manual_graph_break

            # 前向传播函数
            def forward(self, x):
                x = self.seq(x)  # 应用序列模块
                if self.manual_graph_break:
                    torch._dynamo.graph_break()  # 如果设置了手动中断计算图，则调用特定函数
                x = self.conv(x)  # 应用单独的卷积层
                return x

            # 获取输入示例的函数
            def get_example_inputs(self):
                return (torch.randn(2, 3, 16, 16),)

        # 使用 verify_accuracy_for_infer 函数验证推断过程中的模型准确性
        self.verify_accuracy_for_infer(Model)

    @torch.no_grad()
    def test_keep_output_layout_infer(self):
        # 定义一个名为 Model 的神经网络模型类
        class Model(nn.Module):
            # 初始化函数，定义一个二维卷积层
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            # 前向传播函数
            def forward(self, x):
                x = self.conv(x)  # 应用卷积层
                return x

            # 获取输入示例的函数
            def get_example_inputs(self):
                return (torch.randn(2, 3, 5, 5),)

        # 创建 Model 类的实例并部署到 GPU 上
        mod = Model().cuda()
        # 获取模型的示例输入并传输到 GPU 上
        inp = [t.cuda() for t in mod.get_example_inputs()]
        # 在优化后的模块上执行前向传播
        out = mod(*inp)

        # 从优化后的模块生成优化后的模型
        opt_mod = torch.compile(mod)
        # 在优化后的模块上执行前向传播
        opt_out = opt_mod(*inp)

        # 我们应该能够在即时执行输出上执行 view 操作
        out.view(5, -1)

        # 我们应该能够在优化后模块的输出上执行 view 操作
        # 注意，如果输出是通道优先（channels last），view 操作将失败。
        opt_out.view(5, -1)

    def test_keep_output_layout_with_freezing(self):
        # 使用 freezing 参数设置来测试保持输出布局的功能
        with config.patch(
            {
                "freezing": True,
            }
        ):
            self.test_keep_output_layout_infer()

    def test_training_acc(self):
        # 使用 verify_accuracy_for_train 函数验证训练过程中的模型准确性
        self.verify_accuracy_for_train(Model2Conv)

    def test_mutate_view(self):
        """
        The GraphModule passed to GraphLowering init method is like:
        https://gist.github.com/shunting314/07228313fd017e2267101ff32edc6d64

        It shows that we will call copy_ to update the argument in the end. This
        guarantees the correctnesss.
        """
        # 定义一个使用 torch.compile 装饰器的函数 f
        @torch.compile
        def f(x):
            # 对输入 x 进行 view 操作，将其变形为 3x2 的张量
            y = x.view(3, 2)
            # 对 y 张量进行原地乘法运算
            y.mul_(2)

        # 创建一个大小为 2x3 的张量 x，并将其部署到 GPU 上
        x = torch.ones(2, 3).cuda()
        # 调用函数 f，并传入 x 作为参数
        f(x)
        # 断言，验证 x 是否等于原始张量的每个元素乘以 2
        self.assertTrue(torch.equal(x, torch.ones(2, 3).cuda() * 2))
    def test_mutate_base(self):
        """
        The GraphModule passed to GraphLowering init method is like:
        https://gist.github.com/shunting314/fd60fe11d1f844c6db76aba7b06811bc

        It shows that the output of the graph is the mul node which contains
        the update we applied to the base tensor.
        """

        @torch.compile
        def f(x):
            # Reshape tensor x into a 3x2 view
            y = x.view(3, 2)
            # Multiply tensor x by 2 in-place
            x.mul_(2)
            return y

        # Create a tensor of ones with shape (2, 3) on GPU
        x = torch.ones(2, 3).cuda()
        # Apply function f to tensor x
        y = f(x)
        # Assert that y is equal to a tensor of ones with shape (3, 2) multiplied by 2 on GPU
        self.assertTrue(torch.equal(y, torch.ones(3, 2).cuda() * 2))

    @tf32_off()
    def test_mutate_base_for_conv_output(self):
        """
        Test case for mutating base tensor in convolution output.
        """

        class Model(nn.Module):
            def __init__(self, manual_graph_break=False):
                super().__init__()
                # Define a convolutional layer
                self.conv = nn.Conv2d(3, 512, kernel_size=3, stride=2, bias=False)

            def forward(self, x):
                # Pass input x through convolutional layer
                x = self.conv(x)
                # Flatten output x into a 1D tensor
                y = x.view(-1)
                # Multiply flattened tensor y by 2 in-place
                x.mul_(2)
                return y

            def get_example_inputs(self):
                # Return example input tensor
                return (torch.rand(2, 3, 16, 16),)

        # Verify model accuracy during inference
        self.verify_accuracy_for_infer(Model)

    @tf32_off()
    def test_mutate_view_for_conv_output(self):
        """
        Test case for mutating view tensor in convolution output.
        """

        class Model(nn.Module):
            def __init__(self, manual_graph_break=False):
                super().__init__()
                # Define a convolutional layer
                self.conv = nn.Conv2d(3, 512, kernel_size=3, stride=2, bias=False)

            def forward(self, x):
                # Pass input x through convolutional layer
                x = self.conv(x)
                # Flatten output x into a 1D tensor
                y = x.view(-1)
                # Multiply flattened tensor y by 2 in-place
                y.mul_(2)
                return x

            def get_example_inputs(self):
                # Return example input tensor
                return (torch.rand(2, 3, 16, 16),)

        # Verify model accuracy during inference
        self.verify_accuracy_for_infer(Model)

    def test_dynamic_shape_specialization(self):
        """
        Test case for dynamic shape specialization in tensor operations.
        """

        def f(a, b):
            # Compute sin of tensor a
            x = a.sin()
            # Compute cos of tensor b
            y = b.cos()
            # Add tensors x and y
            z = x + y
            return z

        # Perform the test for different sizes
        for size in [4, 8, 16]:
            # Create random tensor a with requires_grad enabled, on GPU
            a = torch.randn(2, size, requires_grad=True).cuda()
            # Create random tensor b, on GPU
            b = torch.randn(2, size).cuda()
            # Compile function f with dynamic shape support
            actual = torch.compile(f, dynamic=True)(a, b)
            # Assert that compiled result matches the direct computation
            self.assertTrue(torch.allclose(f(a, b), actual))

            # Trigger the compiling of the backward graph
            actual.sum().backward()
    def test_nll_loss_backward(self):
        """
        Repro for issue https://github.com/pytorch/pytorch/issues/120759

        The CUDA implementation of aten.nll_loss2d_backward.default requires
        the self tensor (whose layout will be used to create grad_input)
        to be contiguous. Layout optimization may change the self tensor's layout
        and cause failure. We fix that by adding layout constaints to the
        fallback of aten.nll_loss2d_backward.default .
        """

        # 定义一个自定义的神经网络模型类
        class MyModel(torch.nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                # 定义卷积层，输入通道数为1，输出通道数为num_classes，卷积核大小为3x3，步长为1，填充为"same"
                self.conv = torch.nn.Conv2d(1, num_classes, 3, 1, padding="same")
                # 定义全连接层，输入维度为input_dim * num_classes，输出维度为num_classes
                self.out = torch.nn.Linear(input_dim * num_classes, num_classes)

            def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                # 执行前向传播
                x = self.conv(x)  # 卷积层处理输入x
                b, c, t, f = x.size()  # 获取x的维度信息
                x = self.out(x.reshape(b, t, c * f))  # 对卷积结果进行reshape并传入全连接层
                logits = x.reshape(x.size(0), x.size(2), x.size(1))  # 调整输出的形状为(batch_size, seq_len, num_classes)
                # 计算交叉熵损失
                loss = torch.nn.functional.cross_entropy(logits, targets)
                return loss

        device = "cuda"  # 设定使用CUDA加速
        batch_size = 48  # 设置批量大小为48
        seq_len = 144  # 序列长度为144
        input_dim = 39  # 输入维度为39
        num_classes = 111  # 类别数为111

        model = MyModel(input_dim, num_classes)  # 创建模型实例
        model.to(device)  # 将模型移动到指定的设备（GPU）

        opt_model = torch.compile(model)  # 编译模型（此处代码存在错误，应为optimizer实例化，而非torch.compile）

        x = torch.ones((batch_size, 1, seq_len, input_dim), device=device)  # 创建输入数据张量x，全为1
        targets = torch.randint(
            0, num_classes - 1, (batch_size, seq_len), device=device, dtype=torch.int64
        )  # 随机生成目标张量targets，数据范围为0到num_classes-1，大小为(batch_size, seq_len)，数据类型为torch.int64

        loss = model(x, targets)  # 计算模型的损失
        loss.backward()  # 执行反向传播，计算梯度

        ref = model(x, targets)  # 再次计算模型的损失（此处存在错误，应为计算损失值，而非直接用于断言）
        self.assertTrue(torch.allclose(ref, loss))  # 使用断言检查ref和loss之间的数值接近度
# 如果这个脚本是作为主程序运行（而不是被导入为模块），则执行以下操作
if __name__ == "__main__":
    # 如果系统中有 CUDA 加速可用
    if HAS_CUDA:
        # 运行测试函数
        run_tests()
```