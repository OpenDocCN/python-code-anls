# `.\pytorch\test\distributed\pipelining\test_backward.py`

```
# 引入必要的库和模块
import copy  # 用于深拷贝对象

from model_registry import MLPModule  # 导入自定义的MLP模块

import torch  # 导入PyTorch库
from torch.distributed.pipelining._backward import stage_backward  # 导入分布式训练相关的反向传播函数
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试相关的函数和类


d_hid = 512  # 定义隐藏层维度
batch_size = 256  # 定义批量大小


class StageBackwardTests(TestCase):
    def test_stage_backward(self):
        # 创建一个MLP模块作为阶段模块
        mod = MLPModule(d_hid)
        x = torch.randn(batch_size, d_hid)  # 创建随机输入张量x
        x.requires_grad_(True)  # 设置x需要计算梯度
        target = torch.randn(batch_size, d_hid)  # 创建随机目标张量target
        loss_fn = torch.nn.MSELoss(reduction="sum")  # 定义均方误差损失函数

        # 创建一个mod的深拷贝作为参考模块
        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().requires_grad_(x.requires_grad)  # 对x进行detach并设置是否需要计算梯度
        ref_target = target.detach()  # 对target进行detach

        # 阶段式前向和后向传播
        out = mod(x)  # 模块mod进行前向传播
        loss = loss_fn(out, target)  # 计算模型输出与目标之间的损失
        grad_inputs = stage_backward(
            stage_output=loss,  # 当前阶段的输出作为损失
            output_grads=None,
            input_values=(x,),  # 输入值为x的元组形式
        )

        # 运行参考模块
        ref_out = ref_mod(ref_x)  # 参考模块进行前向传播
        ref_loss = loss_fn(ref_out, ref_target)  # 计算参考模型输出与目标之间的损失
        ref_loss.backward()  # 参考模型进行反向传播

        # 断言当前模块的输入梯度与参考模块的输入梯度相近
        torch.testing.assert_close(grad_inputs[0], ref_x.grad)

        # 检查每个参数的梯度是否一致
        for name, p in mod.named_parameters():
            ref_p = ref_mod.get_parameter(name)  # 获取参考模块中的同名参数
            try:
                torch.testing.assert_close(p.grad, ref_p.grad)  # 断言当前模块参数的梯度与参考模块参数的梯度相近
            except AssertionError:
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise


if __name__ == "__main__":
    run_tests()  # 执行测试函数
```