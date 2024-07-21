# `.\pytorch\test\distributed\test_c10d_pypg.py`

```
# Owner(s): ["oncall: distributed"]

# 导入所需的模块和类
import os  # 导入操作系统接口模块
import weakref  # 导入弱引用模块，用于创建弱引用对象

import test_c10d_common  # 导入测试通用模块

import torch  # 导入PyTorch模块
import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch._C._distributed_c10d import _create_work_from_future  # 导入用于创建工作的特定C++接口
from torch.futures import Future  # 导入PyTorch Future模块
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入PyTorch分布式数据并行模块
from torch.testing._internal.common_distributed import MultiProcessTestCase  # 导入多进程测试用例
from torch.testing._internal.common_utils import run_tests  # 导入运行测试的实用工具函数


def create_work(result):
    # 创建一个Future对象并设置结果为给定的result
    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)  # 根据Future对象创建一个工作对象


class MyWork(dist._Work):
    def __init__(self, result, pg):
        super().__init__()
        self.result_ = result  # 存储工作的结果数据
        self.future_ = torch.futures.Future()  # 创建一个PyTorch Future对象
        self.future_.set_result(result)  # 将结果设置到Future对象中
        self.pg_ = weakref.ref(pg)  # 创建对ProcessGroup的弱引用对象

    def wait(self, timeout):
        self.pg_().wait_count += 1  # 增加ProcessGroup中等待计数
        return True  # 模拟等待操作成功完成

    def get_future(self):
        self.pg_().get_future_count += 1  # 增加ProcessGroup中获取Future计数
        return self.future_  # 返回存储的Future对象


class LonelyRankProcessGroup(dist.ProcessGroup):
    """
    This PG only supports world_size of 1
    """

    def __init__(self, rank, world, use_wrapper):
        super().__init__(rank, world)
        assert rank == 0  # 断言rank必须为0，即只支持单进程
        assert world == 1  # 断言world必须为1，即只支持单进程

        self._rank = rank  # 存储进程的rank
        self._world = world  # 存储进程组的world size
        self.wait_count = 0  # 初始化等待计数
        self.get_future_count = 0  # 初始化获取Future计数
        self.use_wrapper = use_wrapper  # 存储是否使用包装器的标志
        self._work = []  # 初始化工作列表

    def broadcast(self, tensor_list, opts):
        if self.use_wrapper:
            return create_work(tensor_list)  # 如果使用包装器，创建并返回工作对象
        res = MyWork(tensor_list, self)  # 否则创建自定义的工作对象
        self._work.append(res)  # 将工作对象添加到工作列表中
        return res  # 返回创建的工作对象

    def allgather(self, output_tensors, input_tensor, opts):
        for o, i in zip(output_tensors[0], input_tensor):
            o.copy_(i)  # 将输入张量的数据复制到输出张量中
        if self.use_wrapper:
            return create_work(output_tensors)  # 如果使用包装器，创建并返回工作对象

        res = MyWork(output_tensors, self)  # 否则创建自定义的工作对象
        self._work.append(res)  # 将工作对象添加到工作列表中

        return res  # 返回创建的工作对象

    def allreduce(self, tensors, opts):
        if self.use_wrapper:
            return create_work(tensors)  # 如果使用包装器，创建并返回工作对象
        res = MyWork(tensors, self)  # 否则创建自定义的工作对象
        self._work.append(res)  # 将工作对象添加到工作列表中
        return res  # 返回创建的工作对象

    def size(self):
        return self._world  # 返回进程组的world size

    def getBackendName(self):
        return "lonely-pg"  # 返回进程组的后端名称

    def __repr__(self):
        return f"PLG w:{self._world} r:{self._rank}"  # 返回进程组的字符串表示形式


# We cannot use parametrize as some tests are defined on the base class and use _get_process_group
class AbstractDDPSingleRank(test_c10d_common.CommonDistributedDataParallelTest):
    def setUp(self):
        super().setUp()
        self._spawn_processes()  # 启动测试进程

    @property
    def world_size(self):
        return 1  # 返回世界大小为1

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)  # 尝试移除文件，用于清理测试环境
        except OSError:
            pass  # 若移除失败则忽略

    def _get_process_group(self):
        return LonelyRankProcessGroup(self.rank, self.world_size, self.use_wrapper)  # 返回一个LonelyRankProcessGroup对象，用于测试
    # 测试分布式数据并行（DDP）调用工作对象的方法
    def test_ddp_invoke_work_object(self):
        # 获取进程组对象
        pg = self._get_process_group()

        # 设置随机种子
        torch.manual_seed(123)
        # 创建一个简单的神经网络模型，包含一个线性层和ReLU激活函数
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        # 将模型包装起来，备份原始模型
        wrapped_model = model
        # 创建输入张量
        input_tensor = torch.rand(2)
        # 使用DDP将模型分布到进程组中
        model = DDP(model, process_group=pg)
        # 对模型进行前向传播、求和、反向传播
        model(input_tensor).sum().backward()

        # 复制DDP模型中第一个线性层的偏置项的梯度
        ddp_grad = wrapped_model[0].bias.grad.clone()

        # 清零原始模型的梯度
        wrapped_model.zero_grad()
        # 对原始模型进行前向传播、求和、反向传播
        wrapped_model(input_tensor).sum().backward()
        # 断言原始模型的第一个线性层的偏置项梯度与DDP模型中的一致
        self.assertEqual(wrapped_model[0].bias.grad, ddp_grad)
        # 如果不使用包装器，则检查进程组的等待计数和未来计数
        if not self.use_wrapper:
            self.assertTrue(pg.wait_count > 0)
            self.assertTrue(pg.get_future_count > 0)

    # 测试使用PyTorch进程组的DDP
    def test_ddp_with_pypg(self):
        # 获取进程组对象
        pg = self._get_process_group()

        # 使用CPU设备测试DDP与指定的进程组
        self._test_ddp_with_process_group(pg, [torch.device("cpu")], device_ids=None)

    # 测试使用PyTorch进程组和梯度视图的DDP
    def test_ddp_with_pypg_with_grad_views(self):
        # 获取进程组对象
        pg = self._get_process_group()

        # 使用CPU设备测试DDP与指定的进程组，并开启梯度作为桶视图的选项
        self._test_ddp_with_process_group(
            pg, [torch.device("cpu")], device_ids=None, gradient_as_bucket_view=True
        )
# 定义一个测试类 TestDDPWithWorkSubclass，继承自 AbstractDDPSingleRank 和 MultiProcessTestCase
class TestDDPWithWorkSubclass(AbstractDDPSingleRank, MultiProcessTestCase):
    # 定义一个属性 use_wrapper，返回 False，表示不使用包装器
    @property
    def use_wrapper(self):
        return False

# 定义一个测试类 TestDDPWithWorkWrapper，继承自 AbstractDDPSingleRank 和 MultiProcessTestCase
class TestDDPWithWorkWrapper(AbstractDDPSingleRank, MultiProcessTestCase):
    # 定义一个属性 use_wrapper，返回 True，表示使用包装器
    @property
    def use_wrapper(self):
        return True

# 如果该脚本作为主程序执行，则运行测试函数
if __name__ == "__main__":
    run_tests()
```