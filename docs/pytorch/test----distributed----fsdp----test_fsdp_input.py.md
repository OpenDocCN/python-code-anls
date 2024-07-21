# `.\pytorch\test\distributed\fsdp\test_fsdp_input.py`

```
# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch import distributed as dist  # 导入 torch 分布式模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入 FSDP 模块
from torch.nn import Linear, Module  # 导入 Linear 层和 Module 基类
from torch.optim import SGD  # 导入 SGD 优化器
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入 GPU 数量检测装饰器
from torch.testing._internal.common_fsdp import FSDPTest  # 导入 FSDP 测试基类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入实例化参数化测试函数
    parametrize,  # 导入参数化装饰器
    run_tests,  # 导入运行测试函数
    subtest,  # 导入子测试装饰器
    TEST_WITH_DEV_DBG_ASAN,  # 导入是否使用 dev-asan 环境的标志
)

if not dist.is_available():  # 如果分布式不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 输出提示信息到标准错误流
    sys.exit(0)  # 退出程序

if TEST_WITH_DEV_DBG_ASAN:  # 如果在 dev-asan 环境下
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",  # 输出提示信息到标准错误流
        file=sys.stderr,
    )
    sys.exit(0)  # 退出程序


class TestInput(FSDPTest):  # 定义测试类 TestInput，继承自 FSDPTest 类
    @property
    def world_size(self):  # 定义属性 world_size 方法
        return 1  # 返回值为 1

    @skip_if_lt_x_gpu(1)  # 使用 GPU 数量不小于 1 的装饰器
    @parametrize("input_cls", [subtest(dict, name="dict"), subtest(list, name="list")])  # 参数化测试，测试输入为 dict 或 list
    def test_input_type(self, input_cls):  # 定义测试输入类型的方法
        """Test FSDP with input being a list or a dict, only single GPU."""
        # 测试 FSDP 在输入为列表或字典时的行为，仅使用单个 GPU。

        class Model(Module):  # 定义模型类，继承自 Module
            def __init__(self):  # 初始化方法
                super().__init__()  # 调用父类初始化方法
                self.layer = Linear(4, 4)  # 定义一个 Linear 层

            def forward(self, input):  # 前向传播方法
                if isinstance(input, list):  # 如果输入是列表
                    input = input[0]  # 取列表的第一个元素
                else:  # 否则
                    assert isinstance(input, dict), input  # 断言输入是字典类型
                    input = input["in"]  # 取字典中键为 'in' 的值
                return self.layer(input)  # 返回 Linear 层的输出

        model = FSDP(Model()).cuda()  # 使用 FSDP 对模型进行分片，并移到 GPU 上
        optim = SGD(model.parameters(), lr=0.1)  # 定义 SGD 优化器

        for _ in range(5):  # 执行 5 次循环
            in_data = torch.rand(64, 4).cuda()  # 生成随机数据，并移到 GPU 上
            in_data.requires_grad = True  # 设置数据需要梯度计算

            if input_cls is list:  # 如果输入类型为列表
                in_data = [in_data]  # 将数据放入列表中
            else:  # 否则
                self.assertTrue(input_cls is dict)  # 断言输入类型为字典
                in_data = {"in": in_data}  # 将数据放入字典中

            out = model(in_data)  # 模型前向计算
            out.sum().backward()  # 计算梯度
            optim.step()  # 更新模型参数
            optim.zero_grad()  # 梯度清零


instantiate_parametrized_tests(TestInput)  # 实例化参数化测试类

if __name__ == "__main__":  # 如果作为主程序运行
    run_tests()  # 运行测试
```