# `.\pytorch\test\distributed\fsdp\test_fsdp_exec_order.py`

```
# Owner(s): ["oncall: distributed"]

# 导入系统模块
import sys
# 导入警告模块
import warnings
# 导入上下文管理模块，用于创建一个空的上下文
from contextlib import nullcontext

# 导入PyTorch库
import torch
# 导入PyTorch分布式模块
from torch import distributed as dist
# 导入FullyShardedDataParallel类，简称为FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# 导入分片策略类
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
# 导入测试工具模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
# 导入FSDP测试工具模块
from torch.testing._internal.common_fsdp import FSDPTest
# 导入通用工具模块
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

# 如果分布式不可用，则打印跳过测试的消息，并退出程序
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果配置中标记了使用dev-asan，则打印相关问题的已知问题，并退出程序
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class Model(torch.nn.Module):
    """
    Model that supports two computation paths: `layer0` -> `layer1` and
    `layer0` -> `layer2`. Notably, both `layer1` and `layer2` have 36 elements
    when flattened, which means that their corresponding all-gathers and
    reduce-scatters may be silently matched if we do not perform any checks.
    """

    def __init__(self) -> None:
        super().__init__()
        # 定义模型的层
        self.layer0 = torch.nn.Linear(5, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6, bias=False),
        )
        self.relu = torch.nn.ReLU()
        self.use_alt_path = False
        # 冻结layer2的参数
        for param in self.layer2.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 前向传播函数，根据self.use_alt_path决定计算路径
        # 如果self.use_alt_path为True，计算路径为layer0 -> layer2 -> relu
        # 如果self.use_alt_path为False，计算路径为layer0 -> layer1 -> relu
        z = self.relu(self.layer0(x))
        z = (
            self.relu(self.layer2(z))
            if self.use_alt_path
            else self.relu(self.layer1(z))
        )
        return z

    def get_input(self, device: torch.device):
        # 获取输入数据，返回一个包含随机数据的元组，该数据已移动到指定设备上
        return (torch.randn((8, 5)).to(device),)

    def get_loss(self, input, output):
        # 计算损失，这里简单地返回输出张量的和作为损失
        return output.sum()

    def run_backward(self, loss):
        # 执行反向传播，计算梯度
        loss.backward()

    def flip_path(self):
        # 切换计算路径
        # 冻结当前计算路径的参数，解冻另一个计算路径的参数
        params_to_freeze = (
            self.layer2.parameters() if self.use_alt_path else self.layer1.parameters()
        )
        params_to_unfreeze = (
            self.layer1.parameters() if self.use_alt_path else self.layer2.parameters()
        )
        for param in params_to_freeze:
            param.requires_grad = False
        for param in params_to_unfreeze:
            param.requires_grad = True
        # 切换计算路径标志
        self.use_alt_path = not self.use_alt_path

    @staticmethod
    # 定义一个名为 wrap 的函数，用于将模型包装成 FSDP 模型
    def wrap(sharding_strategy: ShardingStrategy, device: torch.device):
        # 创建一个新的 Model 实例
        model = Model()
        # 将 model 的 layer1 属性替换为 FSDP 包装后的 layer1，使用给定的分片策略
        model.layer1 = FSDP(model.layer1, sharding_strategy=sharding_strategy)
        # 将 model 的 layer2 属性替换为 FSDP 包装后的 layer2，使用给定的分片策略
        model.layer2 = FSDP(model.layer2, sharding_strategy=sharding_strategy)
        # 将整个 model 包装为 FSDP 模型，使用给定的分片策略
        fsdp_model = FSDP(model, sharding_strategy=sharding_strategy)
        # 将包装好的 FSDP 模型移动到指定的设备上
        return fsdp_model.to(device)
class TestFSDPExecOrder(FSDPTest):
    def setUp(self):
        super().setUp()

    @property
    def device(self):
        return torch.device("cuda")

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    def test_invalid_first_iter_order(
        self,
        sharding_strategy: ShardingStrategy,
    ):
        """Tests that FSDP errors if the all-gather order differs across ranks
        in the first iteration."""
        # 设置分布式调试级别为详细
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        # 使用给定的分片策略和设备包装模型
        fsdp_model = Model.wrap(sharding_strategy, self.device)
        # 如果当前进程的 rank 不是 0，则翻转路径
        if self.rank != 0:
            fsdp_model.flip_path()
        # 获取模型输入数据
        inp = fsdp_model.module.get_input(self.device)
        # 匹配错误消息的正则表达式前缀
        error_regex = "^(Forward order differs across ranks)"
        # 断言捕获到指定的 RuntimeError 并且错误消息符合预期的正则表达式
        with self.assertRaisesRegex(RuntimeError, error_regex):
            fsdp_model(*inp)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    @parametrize("iters_before_path_change", [1, 3])
    def test_invalid_later_iter_order(
        self,
        sharding_strategy: ShardingStrategy,
        iters_before_path_change: int,
    ):
        """Tests that FSDP warns the user if the all-gather order changes after
        the first iteration."""
        # 设置调试级别为 DETAIL
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        # 在第一次迭代中，所有进程运行相同的顺序，而在接下来的迭代中，除了 rank 0 外的所有进程将以不同的顺序运行
        fsdp_model = Model.wrap(sharding_strategy, self.device)
        for _ in range(iters_before_path_change):
            # 获取模型输入
            inp = fsdp_model.module.get_input(self.device)
            # 模型前向传播
            output = fsdp_model(*inp)
            # 计算损失
            loss = fsdp_model.module.get_loss(inp, output).to(self.device)
            # 模型反向传播
            fsdp_model.module.run_backward(loss)
        # 警告消息应与以下前缀匹配
        regex = (
            "^(Forward order differs from that of the first iteration "
            f"on rank {self.rank}. Collectives are unchecked and may give "
            "incorrect results or hang)"
        )
        # 如果 rank 不为 0，则期望发出 UserWarning 警告，匹配给定的正则表达式
        context = (
            self.assertWarnsRegex(
                expected_warning=UserWarning,
                expected_regex=regex,
            )
            if self.rank != 0
            else nullcontext()
        )
        if self.rank != 0:
            # 翻转路径以改变后续迭代的执行顺序
            fsdp_model.flip_path()
        # 获取模型输入
        inp = fsdp_model.module.get_input(self.device)
        # 在上下文中期望收到前向传播的警告
        with context:  # warning for forward pass all-gather
            output = fsdp_model(*inp)
        # 计算损失
        loss = fsdp_model.module.get_loss(inp, output).to(self.device)
        # 模型反向传播
        fsdp_model.module.run_backward(loss)
        # 执行额外的迭代，以检查是否没有更多警告
        inp = fsdp_model.module.get_input(self.device)
        output = fsdp_model(*inp)
        loss = fsdp_model.module.get_loss(inp, output).to(self.device)
        fsdp_model.module.run_backward(loss)
    # 定义一个测试训练和评估的方法，接受一个分片策略参数 `sharding_strategy`
    def test_train_eval(self, sharding_strategy: ShardingStrategy):
        # 设置分布式训练的调试级别为详细模式
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        
        # 使用给定的分片策略和设备包装模型，得到一个分布式的模型对象 `fsdp_model`
        fsdp_model = Model.wrap(sharding_strategy, self.device)
        
        # 定义训练的迭代次数和训练周期数
        NUM_ITERS = 3
        NUM_EPOCHS = 2
        
        # 使用 `warnings.catch_warnings` 上下文管理器记录警告信息到 `w`
        with warnings.catch_warnings(record=True) as w:
            # 循环执行指定次数的训练周期
            for _ in range(NUM_EPOCHS):
                # 将模型设置为训练模式
                fsdp_model.train()
                
                # 循环执行指定次数的训练迭代
                for _ in range(NUM_ITERS):
                    # 获取模型输入数据，从模块中获取，并使用指定设备处理
                    inp = fsdp_model.module.get_input(self.device)
                    # 对模型进行前向传播，生成输出
                    output = fsdp_model(*inp)
                    # 计算损失，并将其传送到指定设备
                    loss = fsdp_model.module.get_loss(inp, output).to(self.device)
                    # 执行反向传播
                    fsdp_model.module.run_backward(loss)
                
                # 将模型设置为评估模式
                fsdp_model.eval()
                
                # 再次循环执行指定次数的评估迭代
                for _ in range(NUM_ITERS):
                    # 获取模型输入数据，从模块中获取，并使用指定设备处理
                    inp = fsdp_model.module.get_input(self.device)
                    # 对模型进行前向传播，生成输出
                    output = fsdp_model(*inp)
                    # 获取评估损失，并将其传送到指定设备，但不使用该损失
                    fsdp_model.module.get_loss(inp, output).to(self.device)
        
        # 检查警告列表 `w` 中是否包含指定前缀的警告信息
        warning_prefix = "Forward order differs"
        for warning in w:
            if str(warning.message).startswith(warning_prefix):
                # 如果找到警告信息，抛出断言错误，指出警告错误发生
                raise AssertionError(
                    f"Warning was incorrectly issued: {warning.message}"
                )
        
        # 如果依然验证评估模式下的前向执行顺序，则上面的 `AssertionError` 会报告两种分片策略的错误
# 实例化带参数的测试用例，使用 TestFSDPExecOrder 类来创建测试实例
instantiate_parametrized_tests(TestFSDPExecOrder)

# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```