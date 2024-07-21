# `.\pytorch\test\distributed\fsdp\test_fsdp_uneven.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear
from torch.optim import SGD
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

# 如果分布式不可用，则输出消息并退出程序
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果使用 dev-asan 测试，跳过，因为 torch + multiprocessing spawn 存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 测试类，继承自 FSDPTest 类
class TestUnevenParamShard(FSDPTest):
    
    # 获取参考结果的方法
    def _get_ref_results(self, model, input, my_lr):
        with torch.no_grad():
            # 计算本地输出的一次迭代
            weight = model.weight.T.clone().to(self.rank)
            v = torch.Tensor(input[self.rank]).to(self.rank)
            ref_forward_output_my_rank = torch.matmul(v, weight)
            # 计算全局权重更新的一次迭代
            v = torch.Tensor(input[: self.world_size]).to(self.rank)
            grad = v.float().sum(0).repeat(weight.shape[0], 1).div(self.world_size)
            ref_weight_out = weight - grad.T * my_lr

        return ref_forward_output_my_rank, ref_weight_out

    # 标记为 GPU 小于 2 时跳过测试
    @skip_if_lt_x_gpu(2)
    def test_one_iteration(self):
        """Test FSDP with uneven divide of parameter shards."""
        # 创建一个线性模型
        model = Linear(3, 3, bias=False)
        input = torch.rand(8, 3)  # 创建随机输入数据
        my_lr = 0.1  # 设置学习率

        # 获取参考的前向输出和权重更新结果
        ref_forward_output_my_rank, ref_weight_out = self._get_ref_results(
            model, input, my_lr
        )

        # 将模型移动到当前 GPU rank
        model.to(self.rank)
        # 使用 FSDP 对象封装模型
        model = FSDP(model)
        # 定义优化器
        optim = SGD(model.parameters(), lr=my_lr)
        # 确保输入数据长度大于等于世界大小
        self.assertTrue(len(input) >= self.world_size)
        # 将输入数据移动到当前 GPU rank
        in_data = torch.Tensor(input[self.rank]).to(self.rank)
        # 对模型进行前向传播
        out = model(in_data)
        # 计算损失并反向传播
        out.float().sum().backward()
        # 执行优化步骤
        optim.step()
        # 清空梯度
        optim.zero_grad()

        # 使用 summon_full_params 方法获取完整的模型参数
        with model.summon_full_params(model):
            weight_out = model.module.weight.T.clone()
            # 断言验证前向输出和权重更新结果是否与参考一致
            self.assertEqual(ref_forward_output_my_rank, out)
            self.assertEqual(ref_weight_out, weight_out)

# 如果是主程序，则运行测试
if __name__ == "__main__":
    run_tests()
```