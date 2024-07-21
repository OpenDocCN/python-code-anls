# `.\pytorch\test\distributed\_tensor\test_experimental_ops.py`

```
# 导入PyTorch库
import torch

# 导入PyTorch分布式模块
import torch.distributed as dist

# 导入相关分布式张量功能和测试工具
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# 设置迭代次数和学习率
ITER_TIME = 10
LR = 0.001

# 继承自DTensorTestBase的DistOtherOpsTest类，用于测试分布式张量操作
class DistOtherOpsTest(DTensorTestBase):
    
    # 返回当前的世界大小（硬编码为2）
    @property
    def world_size(self) -> int:
        # hard code world size to 2
        return 2

    # 使用装饰器with_comms进行通信相关的测试
    @with_comms
    # 测试切片操作
    def test_slice(self):
        # 创建设备网格，用于指定设备类型和设备编号列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义分片规范
        shard_spec = [Replicate()]

        # 创建输入数据列表和梯度输出数据列表
        input_list = torch.rand(ITER_TIME, 1024, 10)
        grad_output_list = torch.rand(ITER_TIME, 1024, 5) * 1e-3

        # 迭代处理每个时间步
        for i in range(ITER_TIME):
            # 将输入数据移动到指定设备类型，并标记为需要梯度计算
            inp = input_list[i].to(self.device_type).requires_grad_()
            # 将梯度输出数据移动到指定设备类型
            grad_output = grad_output_list[i].to(self.device_type)

            # 使用分布式张量进行输入数据的分发
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            # 使用分布式张量进行梯度输出数据的分发
            grad_output_dtensor = distribute_tensor(grad_output, device_mesh, shard_spec)

            # 对分布式张量进行切片操作
            output = inp_dtensor[:, :5]
            # 对切片后的张量进行反向传播
            output.backward(grad_output_dtensor)

            # 对普通张量进行切片操作
            output_gt = inp[:, :5]
            # 对切片后的普通张量进行反向传播
            output_gt.backward(grad_output)

            # 计算输出张量的本地值与普通张量的差异
            output_diff_abs = output.to_local() - output_gt
            output_diff_rel = output_diff_abs / (torch.abs(output_gt) + 1e-8)
            # 计算输出张量的均方误差（绝对值和相对值）
            output_mse_abs = torch.mean(output_diff_abs * output_diff_abs).item()
            output_mse_rel = torch.mean(output_diff_rel * output_diff_rel).item()

            # 计算输入张量梯度的本地值与普通张量梯度的差异
            grad_diff_abs = inp_dtensor.grad.to_local() - inp.grad
            grad_diff_rel = grad_diff_abs / (torch.abs(inp.grad) + 1e-8)
            # 计算输入张量梯度的均方误差（绝对值和相对值）
            grad_mse_abs = torch.mean(grad_diff_abs * grad_diff_abs).item()
            grad_mse_rel = torch.mean(grad_diff_rel * grad_diff_rel).item()

            # 断言均方误差在允许范围内，否则输出错误信息
            self.assertTrue(
                output_mse_abs <= 1e-6,
                f"Too large absolute mse for output, expected less equal 1e-6, got {output_mse_abs}",
            )
            self.assertTrue(
                output_mse_rel <= 1e-6,
                f"Too large relative mse for output, expected less equal 1e-6, got {output_mse_rel}",
            )
            self.assertTrue(
                grad_mse_abs <= 1e-6,
                f"Too large absolute mse for gradient, expected less equal 1e-6, got {grad_mse_abs}",
            )
            self.assertTrue(
                grad_mse_rel <= 1e-6,
                f"Too large relative mse for gradient, expected less equal 1e-6, got {grad_mse_rel}",
            )
    # 定义一个名为 test_bernoulli 的测试函数，用于测试 Bernoulli 操作的分布式计算
    def test_bernoulli(self):
        # 获取当前进程的排名
        rank = dist.get_rank()
        # 创建一个设备网格对象，包含所有设备类型和全局设备列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义数据分片规范，这里使用 Replicate() 表示数据复制到所有设备

        # 生成输入数据列表，形状为 ITER_TIME × 1024 × 10 的随机张量
        input_list = torch.rand(ITER_TIME, 1024, 10)
        # 生成梯度输出数据列表，形状与 input_list 相同，乘以 1e-3 作为小幅度扰动
        grad_output_list = torch.rand(ITER_TIME, 1024, 10) * 1e-3

        # 迭代处理每个时间步
        for i in range(ITER_TIME):
            # 获取当前时间步的输入数据，并将其移动到指定的设备类型上，并要求计算梯度
            inp = input_list[i].to(self.device_type).requires_grad_()
            # 获取当前时间步的梯度输出数据，并将其移动到指定的设备类型上
            grad_output = grad_output_list[i].to(self.device_type)

            # 使用 distribute_tensor 函数将输入数据在设备网格上分发
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            # 使用 distribute_tensor 函数将梯度输出数据在设备网格上分发
            grad_output_dtensor = distribute_tensor(grad_output, device_mesh, shard_spec)

            # 对分布式输入数据应用 Bernoulli 操作
            output = torch.bernoulli(inp_dtensor)
            # 反向传播梯度到分布式输入数据上
            output.backward(grad_output_dtensor)

            # 将输出数据发送到本地节点
            send_output_tensor = output.to_local()
            # 创建一个与 send_output_tensor 形状相同的零张量，用于接收数据
            recv_output_tensor = torch.zeros_like(send_output_tensor)

            # 将输入数据的梯度发送到本地节点
            send_grad_tensor = inp_dtensor.grad.to_local()
            # 创建一个与 send_grad_tensor 形状相同的零张量，用于接收梯度数据
            recv_grad_tensor = torch.zeros_like(send_grad_tensor)

            # 创建发送输出数据的 P2POp 操作，发送到 1 ^ rank 的节点
            send_op_1 = dist.P2POp(dist.isend, send_output_tensor, 1 ^ rank)
            # 创建发送梯度数据的 P2POp 操作，发送到 1 ^ rank 的节点
            send_op_2 = dist.P2POp(dist.isend, send_grad_tensor, 1 ^ rank)
            # 创建接收输出数据的 P2POp 操作，从 1 ^ rank 的节点接收数据
            recv_op_1 = dist.P2POp(dist.irecv, recv_output_tensor, 1 ^ rank)
            # 创建接收梯度数据的 P2POp 操作，从 1 ^ rank 的节点接收数据
            recv_op_2 = dist.P2POp(dist.irecv, recv_grad_tensor, 1 ^ rank)

            # 批量执行发送和接收操作，返回所有请求的列表
            reqs = dist.batch_isend_irecv([send_op_1, send_op_2, recv_op_1, recv_op_2])
            # 等待所有请求完成
            for req in reqs:
                req.wait()

            # 计算发送和接收输出数据之间的绝对差值
            output_diff_abs = send_output_tensor - recv_output_tensor
            # 计算发送和接收输出数据之间的相对差值
            output_diff_rel = output_diff_abs / (torch.abs(recv_output_tensor) + 1e-8)
            # 计算发送和接收输出数据之间的绝对均方误差
            output_mse_abs = torch.mean(output_diff_abs * output_diff_abs).item()
            # 计算发送和接收输出数据之间的相对均方误差
            output_mse_rel = torch.mean(output_diff_rel * output_diff_rel).item()

            # 计算发送和接收梯度数据之间的绝对差值
            grad_diff_abs = send_grad_tensor - recv_grad_tensor
            # 计算发送和接收梯度数据之间的相对差值
            grad_diff_rel = grad_diff_abs / (torch.abs(recv_grad_tensor) + 1e-8)
            # 计算发送和接收梯度数据之间的绝对均方误差
            grad_mse_abs = torch.mean(grad_diff_abs * grad_diff_abs).item()
            # 计算发送和接收梯度数据之间的相对均方误差
            grad_mse_rel = torch.mean(grad_diff_rel * grad_diff_rel).item()

            # 断言输出数据的绝对均方误差小于等于 1e-6，否则输出错误信息
            self.assertTrue(
                output_mse_abs <= 1e-6,
                f"Too large absolute mse for output, expected less equal 1e-6, got {output_mse_abs}",
            )
            # 断言输出数据的相对均方误差小于等于 1e-6，否则输出错误信息
            self.assertTrue(
                output_mse_rel <= 1e-6,
                f"Too large relative mse for output, expected less equal 1e-6, got {output_mse_rel}",
            )
            # 断言梯度数据的绝对均方误差小于等于 1e-6，否则输出错误信息
            self.assertTrue(
                grad_mse_abs <= 1e-6,
                f"Too large absolute mse for gradient, expected less equal 1e-6, got {grad_mse_abs}",
            )
            # 断言梯度数据的相对均方误差小于等于 1e-6，否则输出错误信息
            self.assertTrue(
                grad_mse_rel <= 1e-6,
                f"Too large relative mse for gradient, expected less equal 1e-6, got {grad_mse_rel}",
            )

    @with_comms
    # 定义一个测试方法，用于测试某些功能
    def test_nll(self):
        # 创建一个设备网格对象，使用给定的设备类型和全局大小
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义一个数据分片规范列表，目前只包含一个复制规范对象
        shard_spec = [Replicate()]

        # 创建一个预测结果列表，包含随机生成的数据，形状为 (ITER_TIME, 1024, 10)
        pred_list = torch.rand(ITER_TIME, 1024, 10)
        # 创建一个目标值列表，包含随机生成的整数数据，形状为 (ITER_TIME, 1024)
        target_list = torch.randint(0, 10, (ITER_TIME, 1024), dtype=torch.long)

        # 定义损失函数为交叉熵损失函数
        criterion = torch.nn.CrossEntropyLoss()

        # 循环迭代 ITER_TIME 次
        for i in range(ITER_TIME):
            # 选择当前迭代的预测结果，并将其移动到指定的设备类型，并设置为需要梯度计算
            pred = pred_list[i].to(self.device_type).requires_grad_()
            # 选择当前迭代的目标值，并将其移动到指定的设备类型
            target = target_list[i].to(self.device_type)

            # 使用分布式张量函数将预测结果分布到设备网格上
            pred_dtensor = distribute_tensor(pred, device_mesh, shard_spec)
            # 使用分布式张量函数将目标值分布到设备网格上
            target_dtensor = distribute_tensor(target, device_mesh, shard_spec)

            # 计算分布式张量的交叉熵损失
            loss = criterion(pred_dtensor, target_dtensor)
            # 反向传播计算梯度
            loss.backward()

            # 计算普通张量的交叉熵损失
            loss_gt = criterion(pred, target)
            # 反向传播计算梯度
            loss_gt.backward()

            # 计算损失值的绝对差值
            loss_diff_abs = loss.to_local() - loss_gt
            # 计算损失值的相对差值
            loss_diff_rel = loss_diff_abs / (torch.abs(loss_gt) + 1e-8)
            # 计算损失值的均方误差（绝对值）
            loss_mse_abs = torch.mean(loss_diff_abs * loss_diff_abs).item()
            # 计算损失值的均方误差（相对值）
            loss_mse_rel = torch.mean(loss_diff_rel * loss_diff_rel).item()

            # 计算梯度的绝对差值
            grad_diff_abs = pred_dtensor.grad.to_local() - pred.grad
            # 计算梯度的相对差值
            grad_diff_rel = grad_diff_abs / (torch.abs(pred.grad) + 1e-8)
            # 计算梯度的均方误差（绝对值）
            grad_mse_abs = torch.mean(grad_diff_abs * grad_diff_abs).item()
            # 计算梯度的均方误差（相对值）
            grad_mse_rel = torch.mean(grad_diff_rel * grad_diff_rel).item()

            # 使用断言检查损失值的绝对均方误差是否小于等于 1e-6
            self.assertTrue(
                loss_mse_abs <= 1e-6,
                f"Too large absolute mse for loss, expected less equal 1e-6, got {loss_mse_abs}",
            )
            # 使用断言检查损失值的相对均方误差是否小于等于 1e-6
            self.assertTrue(
                loss_mse_rel <= 1e-6,
                f"Too large relative mse for loss, expected less equal 1e-6, got {loss_mse_rel}",
            )
            # 使用断言检查梯度的绝对均方误差是否小于等于 1e-6
            self.assertTrue(
                grad_mse_abs <= 1e-6,
                f"Too large absolute mse for gradient, expected less equal 1e-6, got {grad_mse_abs}",
            )
            # 使用断言检查梯度的相对均方误差是否小于等于 1e-6
            self.assertTrue(
                grad_mse_rel <= 1e-6,
                f"Too large relative mse for gradient, expected less equal 1e-6, got {grad_mse_rel}",
            )
# 如果当前脚本作为主程序运行（而非被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试
    run_tests()
```