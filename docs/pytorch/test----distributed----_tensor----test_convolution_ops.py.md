# `.\pytorch\test\distributed\_tensor\test_convolution_ops.py`

```
# 导入必要的库和模块
import copy  # 导入copy模块，用于深拷贝操作

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._tensor import (  # 导入分布式相关模块和类
    DeviceMesh,  # 设备网格类
    distribute_module,  # 分布式模块分发函数
    distribute_tensor,  # 分布式张量分发函数
    Replicate,  # 复制分布策略类
    Shard,  # 分片分布策略类
)
from torch.testing._internal.common_utils import run_tests  # 导入测试相关函数
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入分布式张量测试基类和相关函数
    DTensorTestBase,  # 分布式张量测试基类
    skip_if_lt_x_gpu,  # 如果GPU数量小于指定数量则跳过测试的装饰器
    with_comms,  # 测试通信相关装饰器
)

ITER_TIME = 10  # 迭代次数
LR = 0.001  # 学习率


def _conv_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    # 遍历模块的所有参数，并根据设备网格进行分布式参数初始化
    for name, param in module.named_parameters():
        dist_spec = [Replicate()]  # 使用复制分布策略
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, dist_spec)
        )
        name = "_".join(name.split("."))  # 将参数名中的点号替换为下划线
        module.register_parameter(name, dist_param)  # 注册分布式参数


class DistConvolutionOpsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # 将全局世界大小设为2
        return 2

    @with_comms
    # 定义测试函数，用于测试下采样卷积
    def test_downsampling_convolution(self):
        # 创建设备网格对象，指定设备类型和世界大小
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义分片规格，包含一个具有3个元素的Shard对象列表
        shard_spec = [Shard(3)]

        # 创建输入数据列表，形状为(ITER_TIME, 7, 3, 512, 1024)，并随机初始化
        input_list = torch.rand(ITER_TIME, 7, 3, 512, 1024)
        # 创建梯度输出数据列表，形状为(ITER_TIME, 7, 256, 128, 256)，并乘以1e-3
        grad_output_list = torch.rand(ITER_TIME, 7, 256, 128, 256) * 1e-3

        # 创建一个卷积神经网络模型，输入通道数为3，输出通道数为256，核大小为4x4，步长为4，无填充，放置在指定设备上
        model = nn.Conv2d(3, 256, kernel_size=4, stride=4, padding=0).to(
            self.device_type
        )
        # 初始化模型权重为全1
        nn.init.ones_(model.weight)
        # 初始化模型偏置为全0
        nn.init.zeros_(model.bias)
        # 深度复制模型，用于后续对比
        model_gt = copy.deepcopy(model).to(self.device_type)

        # 使用分布式模块分布式训练模型，传入设备网格、卷积函数、输入函数和输出函数
        model = distribute_module(
            model, device_mesh, _conv_fn, input_fn=None, output_fn=None
        )
        # 使用随机梯度下降优化器，学习率为LR，优化模型参数
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        # 迭代ITER_TIME次进行训练
        for i in range(ITER_TIME):
            # 梯度清零
            optimizer.zero_grad()
            # 将输入数据转移到指定设备，并设置需要计算梯度
            inp = input_list[i].to(self.device_type).requires_grad_()
            # 将输入数据分布到设备网格上的特定分片
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            # 使用模型进行前向传播
            output = model(inp_dtensor)
            # 获取当前迭代的梯度输出数据，并转移到指定设备
            grad_output = grad_output_list[i].to(self.device_type)
            # 将梯度输出数据分布到设备网格上的特定分片
            grad_output_dtensor = distribute_tensor(
                grad_output, device_mesh, shard_spec
            )
            # 反向传播梯度
            output.backward(grad_output_dtensor)
            # 执行优化步骤
            optimizer.step()

        # 使用普通张量进行训练
        optimizer_gt = torch.optim.SGD(model_gt.parameters(), lr=LR)
        # 迭代ITER_TIME次进行训练
        for i in range(ITER_TIME):
            # 梯度清零
            optimizer_gt.zero_grad()
            # 将输入数据转移到指定设备，并设置需要计算梯度
            inp = input_list[i].to(self.device_type).requires_grad_()
            # 使用模型进行前向传播
            output = model_gt(inp)
            # 获取当前迭代的梯度输出数据，并转移到指定设备
            grad_output = grad_output_list[i].to(self.device_type)
            # 反向传播梯度
            output.backward(grad_output)
            # 执行优化步骤
            optimizer_gt.step()

        # 计算模型权重的绝对差值
        weight_diff_abs = model.weight.to_local() - model_gt.weight
        # 计算模型偏置的绝对差值
        bias_diff_abs = model.bias.to_local() - model_gt.bias
        # 计算模型权重的相对差值
        weight_diff_rel = weight_diff_abs / (torch.abs(model_gt.weight) + 1e-8)
        # 计算模型偏置的相对差值
        bias_diff_rel = bias_diff_abs / (torch.abs(model_gt.bias) + 1e-8)
        # 计算模型权重的绝对均方误差
        weight_mse_abs = torch.mean(weight_diff_abs * weight_diff_abs).item()
        # 计算模型偏置的绝对均方误差
        bias_mse_abs = torch.mean(bias_diff_abs * bias_diff_abs).item()
        # 计算模型权重的相对均方误差
        weight_mse_rel = torch.mean(weight_diff_rel * weight_diff_rel).item()
        # 计算模型偏置的相对均方误差
        bias_mse_rel = torch.mean(bias_diff_rel * bias_diff_rel).item()

        # 断言检查模型权重的绝对均方误差是否小于等于1e-6
        self.assertTrue(
            weight_mse_abs <= 1e-6,
            f"Too large absolute mse for weight tensor, expected less equal 1e-6, got {weight_mse_abs}",
        )
        # 断言检查模型偏置的绝对均方误差是否小于等于1e-6
        self.assertTrue(
            bias_mse_abs <= 1e-6,
            f"Too large absolute mse for bias tensor, expected less equal 1e-6, got {bias_mse_abs}",
        )
        # 断言检查模型权重的相对均方误差是否小于等于1e-6
        self.assertTrue(
            weight_mse_rel <= 1e-6,
            f"Too large relative mse for weight tensor, expected less equal 1e-6, got {weight_mse_rel}",
        )
        # 断言检查模型偏置的相对均方误差是否小于等于1e-6
        self.assertTrue(
            bias_mse_rel <= 1e-6,
            f"Too large relative mse for bias tensor, expected less equal 1e-6, got {bias_mse_rel}",
        )
    # 在 CI 中，使用 gloo 后端时，发现 test_depthwise_convolution 存在问题，暂时禁用该测试以解除 CI 阻塞。
    # 使用装饰器 @with_comms，确保测试在通信环境下运行。
    # 使用装饰器 @skip_if_lt_x_gpu(2)，跳过不具备至少两个 GPU 的环境下的测试。
    @with_comms
    @skip_if_lt_x_gpu(2)
    # 定义测试深度卷积的函数，该函数是一个测试用例，属于测试框架的一部分
    def test_depthwise_convolution(self):
        # 创建设备网格对象，用于分布式计算，包括设备类型和设备编号列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义分片规范，这里只包含一个分片对象
        shard_spec = [Shard(3)]

        # 创建随机输入数据列表，形状为 ITER_TIME * 7 * 256 * 128 * 256
        input_list = torch.rand(ITER_TIME, 7, 256, 128, 256)
        # 创建随机梯度输出数据列表，形状与输入数据列表相同，乘以 1e-3
        grad_output_list = torch.rand(ITER_TIME, 7, 256, 128, 256) * 1e-3

        # 创建标准的深度卷积模型，256通道输入，256通道输出，7x7卷积核，填充3，组数为256，并移动到指定设备类型
        model = nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256).to(
            self.device_type
        )
        # 初始化权重为全1，偏置为全0
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)
        # 深度复制模型，用于后续对比
        model_gt = copy.deepcopy(model).to(self.device_type)

        # 使用分布式模块对模型进行分布式训练，使用指定的设备网格和卷积函数
        model = distribute_module(
            model, device_mesh, _conv_fn, input_fn=None, output_fn=None
        )
        # 使用随机梯度下降优化器进行优化
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        # 迭代训练 ITER_TIME 次
        for i in range(ITER_TIME):
            optimizer.zero_grad()
            # 将输入数据转移到指定设备并要求计算梯度
            inp = input_list[i].to(self.device_type).requires_grad_()
            # 使用分布式张量将输入数据分发到设备网格上指定的分片
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            # 计算模型的输出
            output = model(inp_dtensor)
            # 获取当前迭代的梯度输出数据
            grad_output = grad_output_list[i].to(self.device_type)
            # 使用分布式张量将梯度输出数据分发到设备网格上指定的分片
            grad_output_dtensor = distribute_tensor(
                grad_output, device_mesh, shard_spec
            )
            # 反向传播梯度
            output.backward(grad_output_dtensor)
            # 执行优化步骤
            optimizer.step()

        # 使用普通张量进行训练
        optimizer_gt = torch.optim.SGD(model_gt.parameters(), lr=LR)
        # 再次迭代训练 ITER_TIME 次
        for i in range(ITER_TIME):
            optimizer_gt.zero_grad()
            # 将输入数据转移到指定设备并要求计算梯度
            inp = input_list[i].to(self.device_type).requires_grad_()
            # 计算模型的输出
            output = model_gt(inp)
            # 获取当前迭代的梯度输出数据
            grad_output = grad_output_list[i].to(self.device_type)
            # 反向传播梯度
            output.backward(grad_output)
            # 执行优化步骤
            optimizer_gt.step()

        # 计算模型权重和偏置的绝对差值
        weight_diff_abs = model.weight.to_local() - model_gt.weight
        bias_diff_abs = model.bias.to_local() - model_gt.bias
        # 计算模型权重和偏置的相对差值
        weight_diff_rel = weight_diff_abs / (torch.abs(model_gt.weight) + 1e-8)
        bias_diff_rel = bias_diff_abs / (torch.abs(model_gt.bias) + 1e-8)
        # 计算模型权重和偏置的绝对均方误差
        weight_mse_abs = torch.mean(weight_diff_abs * weight_diff_abs).item()
        bias_mse_abs = torch.mean(bias_diff_abs * bias_diff_abs).item()
        # 计算模型权重和偏置的相对均方误差
        weight_mse_rel = torch.mean(weight_diff_rel * weight_diff_rel).item()
        bias_mse_rel = torch.mean(bias_diff_rel * bias_diff_rel).item()
        # 断言绝对均方误差小于等于1e-6，用于验证权重张量的精度
        self.assertTrue(
            weight_mse_abs <= 1e-6,
            f"Too large absolute mse for weight tensor, expected less equal 1e-6, got {weight_mse_abs}",
        )
        # 断言绝对均方误差小于等于1e-6，用于验证偏置张量的精度
        self.assertTrue(
            bias_mse_abs <= 1e-6,
            f"Too large absolute mse for bias tensor, expected less equal 1e-6, got {bias_mse_abs}",
        )
        # 断言相对均方误差小于等于1e-6，用于验证权重张量的相对精度
        self.assertTrue(
            weight_mse_rel <= 1e-6,
            f"Too large relative mse for weight tensor, expected less equal 1e-6, got {weight_mse_rel}",
        )
        # 断言相对均方误差小于等于1e-6，用于验证偏置张量的相对精度
        self.assertTrue(
            bias_mse_rel <= 1e-6,
            f"Too large relative mse for bias tensor, expected less equal 1e-6, got {bias_mse_rel}",
        )
# 如果这个模块是直接被运行的（而不是被导入到其它模块中执行），那么执行下面的代码
if __name__ == "__main__":
    # 调用运行测试函数
    run_tests()
```