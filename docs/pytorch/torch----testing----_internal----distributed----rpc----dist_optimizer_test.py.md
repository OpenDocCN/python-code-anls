# `.\pytorch\torch\testing\_internal\distributed\rpc\dist_optimizer_test.py`

```
# 忽略 mypy 类型检查中的错误
# 导入多线程模块
import threading

# 导入 PyTorch 库及分布式自动求导、RPC 模块
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
# 导入优化器及分布式优化器模块
from torch import optim
from torch.distributed.optim import DistributedOptimizer
# 导入用于分布式测试的工具函数和类
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


# 示例类 MyModule
class MyModule:
    # 类级别的线程锁，用于确保线程安全
    lock = threading.Lock()

    # 构造函数，初始化模型参数
    def __init__(self, requires_grad=True):
        # 由于所有线程共享默认生成器，不能直接使用 torch.manual_seed(0)
        # 多个 RPC 线程的竞争可能会影响默认 RNG 实例的生成顺序，导致非确定性行为
        # 因此，在这里创建一个专用的 RNG 实例
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        # 创建随机初始化的模型参数 w
        self.w = torch.rand((3, 3), requires_grad=requires_grad, generator=g_cpu)

    # 前向传播方法
    def forward(self, t1):
        return torch.mm(self.w, t1)

    # 获取模型参数 w
    def get_w(self):
        return self.w


# 示例优化器类 FailingOptimizer
class FailingOptimizer(optim.Optimizer):
    # 构造函数，继承自父类 Optimizer
    def __init__(self, params):
        super().__init__(params, {})

    # 优化步骤，抛出异常以模拟失败情况
    def step(self, closure=None):
        raise ValueError("Error running optimizer.")


# 示例优化器类 OptimizerFailingOnConstructor
class OptimizerFailingOnConstructor(optim.Optimizer):
    # 构造函数，继承自父类 Optimizer
    def __init__(self, params):
        super().__init__(params, {})
        # 抛出异常，模拟在构造函数中出错的情况
        raise ValueError("Error creating optimizer.")

    # 优化步骤，抛出异常以模拟失败情况
    def step(self, closure=None):
        raise NotImplementedError


# 辅助函数 _call_method
def _call_method(method, obj_rref, *args, **kwargs):
    return method(obj_rref.local_value(), *args, **kwargs)


# 远程方法调用函数 remote_method
def remote_method(method, obj_rref, *args, **kwargs):
    """
    在远程对象上调用 rpc.remote 中的方法。

    Args:
        method: 方法（例如，Class.method）
        obj_rref (RRef): 对对象的远程引用
        args: 传递给方法的位置参数
        kwargs: 传递给方法的关键字参数

    返回一个 RRef，指向远程方法调用的结果。
    """
    return rpc.remote(
        obj_rref.owner(),
        _call_method,
        args=[method, obj_rref] + list(args),
        kwargs=kwargs,
    )


# 异步 RPC 方法调用函数 rpc_async_method
def rpc_async_method(method, obj_rref, *args, **kwargs):
    """
    在远程对象上调用 rpc.rpc_async 中的方法。

    Args:
        method: 方法（例如，Class.method）
        obj_rref (RRef): 对对象的远程引用
        args: 传递给方法的位置参数
        kwargs: 传递给方法的关键字参数

    返回一个 Future，指向方法调用的结果。
    """
    return rpc.rpc_async(
        obj_rref.owner(),
        _call_method,
        args=[method, obj_rref] + list(args),
        kwargs=kwargs,
    )


# 示例测试类 DistOptimizerTest，继承自 RpcAgentTestFixture
class DistOptimizerTest(RpcAgentTestFixture):
    @dist_init()
    def test_dist_optim_exception(self):
        # 分布式版本
        # 计算 owner1 和 owner2 的名称，用于远程调用
        owner1 = "worker%d" % ((self.rank + 1) % self.world_size)
        owner2 = "worker%d" % ((self.rank + 2) % self.world_size)

        # 在 owner1 和 owner2 上创建远程模块
        remote_module1 = rpc.remote(owner1, MyModule)
        remote_module2 = rpc.remote(owner2, MyModule)
        
        # 获取远程模块的参数
        remote_param1 = remote_method(MyModule.get_w, remote_module1)
        remote_param2 = remote_method(MyModule.get_w, remote_module2)

        # 创建分布式优化器，用于远程参数
        dist_optim = DistributedOptimizer(
            FailingOptimizer, [remote_param1, remote_param2]
        )

        # 使用分布式自动求导上下文
        with dist_autograd.context() as context_id:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(0)
            t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            
            # 异步调用远程模块的前向传播
            output1 = rpc_async_method(MyModule.forward, remote_module1, t2)
            output2 = rpc_async_method(MyModule.forward, remote_module2, output1.wait())
            
            # 计算损失
            loss = torch.add(output2.wait(), t1).sum()

            # 在分布式自动求导上下文中进行反向传播
            dist_autograd.backward(context_id, [loss])
            
            # 确保在优化器运行中捕获到期望的异常
            with self.assertRaisesRegex(Exception, "Error running optimizer"):
                dist_optim.step(context_id)

    @dist_init()
    def test_dist_optim_exception_on_constructor(self):
        # 分布式版本
        # 计算 owner1 和 owner2 的名称，用于远程调用
        owner1 = "worker%d" % ((self.rank + 1) % self.world_size)
        owner2 = "worker%d" % ((self.rank + 2) % self.world_size)

        # 在 owner1 和 owner2 上创建远程模块
        remote_module1 = rpc.remote(owner1, MyModule)
        remote_module2 = rpc.remote(owner2, MyModule)
        
        # 获取远程模块的参数
        remote_param1 = remote_method(MyModule.get_w, remote_module1)
        remote_param2 = remote_method(MyModule.get_w, remote_module2)

        # 在构造分布式优化器时确保捕获到期望的构造函数异常
        with self.assertRaisesRegex(Exception, "Error creating optimizer."):
            dist_optim = DistributedOptimizer(
                OptimizerFailingOnConstructor, [remote_param1, remote_param2]
            )
    # 定义测试函数 _test_dist_optim_base，接受优化器类 optim_cls 和其他参数
    def _test_dist_optim_base(self, optim_cls, *args, **kwargs):
        # 创建本地版本的 MyModule 实例
        module1 = MyModule()
        module2 = MyModule()
        # 获取两个模块的权重参数
        params = [module1.get_w(), module2.get_w()]
        # 使用传入的优化器类实例化本地优化器
        local_optim = optim_cls(params, *args, **kwargs)

        # 复制并分离两个模块的权重作为旧权重
        old_w1 = module1.w.clone().detach()
        old_w2 = module2.w.clone().detach()

        # 使用 torch.Generator 创建 CPU 随机数生成器 g_cpu，并设置种子
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        # 创建两个 3x3 的随机张量，需要梯度，使用上面的生成器
        t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        # 计算模块1的前向传播结果
        output1 = module1.forward(t2)
        # 计算模块2的前向传播结果
        output2 = module2.forward(output1)
        # 计算输出2与t1的和并求和，作为损失值
        loss = torch.add(output2, t1).sum()

        # 反向传播求梯度
        loss.backward()
        # 执行本地优化器的一步优化
        local_optim.step()

        # 分布式版本
        owner1 = "worker%d" % ((self.rank + 1) % self.world_size)
        owner2 = "worker%d" % ((self.rank + 2) % self.world_size)

        # 在远程节点owner1上创建MyModule的远程实例
        remote_module1 = rpc.remote(owner1, MyModule)
        # 在远程节点owner2上创建MyModule的远程实例
        remote_module2 = rpc.remote(owner2, MyModule)
        # 调用远程方法获取remote_module1的权重参数
        remote_param1 = remote_method(MyModule.get_w, remote_module1)
        # 调用远程方法获取remote_module2的权重参数
        remote_param2 = remote_method(MyModule.get_w, remote_module2)

        # 获取remote_param1的当前值作为远程初始权重
        old_w1_remote = remote_param1.to_here()

        # 检查：本地和远程初始权重应该相等
        self.assertEqual(old_w1, remote_param1.to_here())
        self.assertEqual(old_w2, remote_param2.to_here())

        # 实例化分布式优化器，传入远程参数和其他参数
        dist_optim = DistributedOptimizer(
            optim_cls, [remote_param1, remote_param2], *args, **kwargs
        )

        # 使用分布式自动求导上下文
        with dist_autograd.context() as context_id:
            # 重新设置 CPU 随机数生成器的种子
            g_cpu.manual_seed(0)
            # 创建两个 3x3 的随机张量，需要梯度，使用上面的生成器
            t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            # 在remote_module1上异步调用前向传播方法
            output1 = rpc_async_method(MyModule.forward, remote_module1, t2)
            # 在remote_module2上异步调用前向传播方法，并等待结果
            output2 = rpc_async_method(MyModule.forward, remote_module2, output1.wait())
            # 计算输出2和t1的和，并等待结果
            loss = torch.add(output2.wait(), t1)

            # 在分布式自动求导上下文中进行反向传播
            dist_autograd.backward(context_id, [loss.sum()])
            # 执行分布式优化器的一步优化
            dist_optim.step(context_id)

            # 异步获取remote_module1和remote_module2的新权重参数
            new_w1 = rpc_async_method(MyModule.get_w, remote_module1).wait()
            new_w2 = rpc_async_method(MyModule.get_w, remote_module2).wait()

            # 确保优化器已更新权重
            self.assertNotEqual(old_w1, new_w1)
            self.assertNotEqual(old_w2, new_w2)
            # 确保本地模块和远程模块的权重一致
            self.assertEqual(new_w1, module1.get_w())
            self.assertEqual(new_w2, module2.get_w())

    @dist_init()
    def test_dist_optim(self):
        # 测试不同优化器的分布式优化功能

        # 测试 Adagrad 优化器，学习率为 0.05
        self._test_dist_optim_base(optim.Adagrad, lr=0.05)
        
        # 测试 Adam 优化器，学习率为 0.01，启用 amsgrad
        self._test_dist_optim_base(optim.Adam, lr=1e-2, amsgrad=True)
        
        # 测试 AdamW 优化器，学习率为 0.05，启用 amsgrad
        self._test_dist_optim_base(optim.AdamW, lr=0.05, amsgrad=True)
        
        # 测试 SGD 优化器，学习率为 0.05
        self._test_dist_optim_base(optim.SGD, lr=0.05)
        
        # 测试 SGD 优化器，学习率为 0.001，动量为 1，权重衰减为 1，启用 nesterov
        self._test_dist_optim_base(optim.SGD, lr=1e-3, momentum=1, weight_decay=1, nesterov=True)
        
        # 测试 Adadelta 优化器，rho 参数为 0.95
        self._test_dist_optim_base(optim.Adadelta, rho=0.95)
        
        # 测试 RMSprop 优化器，学习率为 0.05
        self._test_dist_optim_base(optim.RMSprop, lr=0.05)
        
        # 测试 Adamax 优化器，学习率为 0.05
        self._test_dist_optim_base(optim.Adamax, lr=0.05)
        
        # 测试 Rprop 优化器，学习率为 0.05
        self._test_dist_optim_base(optim.Rprop, lr=0.05)
    # 定义测试函数，用于测试分布式优化器在梯度为None的情况下的行为
    def _test_dist_optim_none_grads(self, optim_cls, *args, **kwargs):
        # 创建本地模块1，所有参数都需要梯度
        module1 = MyModule()
        # 创建本地模块2，所有参数不需要梯度
        module2 = MyModule(requires_grad=False)
        # 获取模块1和模块2的权重参数列表
        params = [module1.get_w(), module2.get_w()]
        # 使用给定的优化器类初始化本地优化器
        local_optim = optim_cls(params, *args, **kwargs)

        # 记录模块1和模块2的旧权重
        old_w1 = module1.w.clone().detach()
        old_w2 = module2.w.clone().detach()

        # 创建一个CPU生成器，设置种子为0
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        # 创建两个3x3的张量，要求计算梯度
        t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        # 对模块1进行前向传播
        output1 = module1.forward(t2)
        # 对模块2进行前向传播，输入为模块1的输出
        output2 = module2.forward(output1)
        # 计算损失：模块2的输出与t1的和的总和
        loss = torch.add(output2, t1).sum()

        # 反向传播损失
        loss.backward()
        # 执行本地优化步骤
        local_optim.step()

        # 分布式版本
        # 确定远程模块的所有者
        owner1 = "worker%d" % ((self.rank + 1) % self.world_size)
        owner2 = "worker%d" % ((self.rank + 2) % self.world_size)
        # 在远程worker1上创建远程模块1
        remote_module1 = rpc.remote(owner1, MyModule)
        # 在远程worker2上创建远程模块2，参数不需要梯度
        remote_module2 = rpc.remote(owner2, MyModule, args=(False,))
        # 获取远程模块1的远程权重参数
        remote_param1 = remote_module1.remote().get_w()
        # 获取远程模块2的远程权重参数
        remote_param2 = remote_module2.remote().get_w()

        # 检查点：本地和远程初始权重应该匹配
        self.assertEqual(old_w1, remote_param1.to_here())
        self.assertEqual(old_w2, remote_param2.to_here())

        # 初始化分布式优化器，优化远程参数1和参数2
        dist_optim = DistributedOptimizer(
            optim_cls, [remote_param1, remote_param2], *args, **kwargs
        )

        # 在上下文中执行分布式自动求导
        with dist_autograd.context() as context_id:
            # 重新设置CPU生成器种子
            g_cpu.manual_seed(0)
            # 创建两个3x3的张量，要求计算梯度
            t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            # 异步调用远程模块1的前向传播
            output1 = remote_module1.rpc_async().forward(t2)
            # 异步调用远程模块2的前向传播，等待output1的完成
            output2 = remote_module2.rpc_async().forward(output1.wait())
            # 计算损失：output2与t1的和
            loss = torch.add(output2.wait(), t1)

            # 分布式自动求导，计算损失的总和的梯度
            dist_autograd.backward(context_id, [loss.sum()])
            # 执行分布式优化器步骤
            dist_optim.step(context_id)

            # 异步获取远程模块1和模块2的新权重
            new_w1 = remote_module1.rpc_async().get_w().wait()
            new_w2 = remote_module2.rpc_async().get_w().wait()

            # 确保优化器修改了模块1的权重
            self.assertNotEqual(old_w1, new_w1)

            # 确保优化器未修改模块2的权重
            self.assertEqual(old_w2, new_w2)
            # 确保本地模块的权重与远程模块相等
            self.assertEqual(new_w1, module1.get_w())
            self.assertEqual(new_w2, module2.get_w())
```