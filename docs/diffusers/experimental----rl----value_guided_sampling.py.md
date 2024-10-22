# `.\diffusers\experimental\rl\value_guided_sampling.py`

```py
# 版权声明，2024年HuggingFace团队版权所有
# 
# 根据Apache许可证第2.0版（"许可证"）许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，
# 否则根据许可证分发的软件按“原样”提供，
# 不附带任何明示或暗示的担保或条件。
# 请参阅许可证以了解有关权限和限制的具体语言。

# 导入numpy库以进行数值计算
import numpy as np
# 导入PyTorch库以进行深度学习
import torch
# 导入tqdm库以显示进度条
import tqdm

# 从自定义模块中导入UNet1DModel
from ...models.unets.unet_1d import UNet1DModel
# 从自定义模块中导入DiffusionPipeline
from ...pipelines import DiffusionPipeline
# 从自定义模块中导入DDPMScheduler
from ...utils.dummy_pt_objects import DDPMScheduler
# 从自定义模块中导入randn_tensor函数
from ...utils.torch_utils import randn_tensor


# 定义用于值引导采样的管道类
class ValueGuidedRLPipeline(DiffusionPipeline):
    r"""
    用于从训练的扩散模型中进行值引导采样的管道，模型预测状态序列。

    该模型继承自[`DiffusionPipeline`]。请查阅超类文档以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    参数：
        value_function ([`UNet1DModel`]):
            一个专门用于基于奖励微调轨迹的UNet。
        unet ([`UNet1DModel`]):
            用于去噪编码轨迹的UNet架构。
        scheduler ([`SchedulerMixin`]):
            用于与`unet`结合去噪编码轨迹的调度器。此应用程序的默认调度器为[`DDPMScheduler`]。
        env ():
            一个遵循OpenAI gym API的环境进行交互。目前仅Hopper有预训练模型。
    """

    # 初始化方法，接受各个组件作为参数
    def __init__(
        self,
        value_function: UNet1DModel,  # 值函数UNet模型
        unet: UNet1DModel,             # 去噪UNet模型
        scheduler: DDPMScheduler,      # 调度器
        env,                           # 环境
    ):
        super().__init__()  # 调用父类的初始化方法

        # 注册模型和调度器模块
        self.register_modules(value_function=value_function, unet=unet, scheduler=scheduler, env=env)

        # 从环境获取数据集
        self.data = env.get_dataset()
        self.means = {}  # 初始化均值字典
        # 遍历数据集的每个键
        for key in self.data.keys():
            try:
                # 计算并存储每个键的均值
                self.means[key] = self.data[key].mean()
            except:  # 捕获异常
                pass
        self.stds = {}  # 初始化标准差字典
        # 再次遍历数据集的每个键
        for key in self.data.keys():
            try:
                # 计算并存储每个键的标准差
                self.stds[key] = self.data[key].std()
            except:  # 捕获异常
                pass
        # 获取状态维度
        self.state_dim = env.observation_space.shape[0]
        # 获取动作维度
        self.action_dim = env.action_space.shape[0]

    # 归一化输入数据
    def normalize(self, x_in, key):
        return (x_in - self.means[key]) / self.stds[key]  # 根据均值和标准差归一化

    # 反归一化输入数据
    def de_normalize(self, x_in, key):
        return x_in * self.stds[key] + self.means[key]  # 根据均值和标准差反归一化
    # 定义将输入转换为 Torch 张量的方法
        def to_torch(self, x_in):
            # 检查输入是否为字典类型
            if isinstance(x_in, dict):
                # 递归地将字典中的每个值转换为 Torch 张量
                return {k: self.to_torch(v) for k, v in x_in.items()}
            # 检查输入是否为 Torch 张量
            elif torch.is_tensor(x_in):
                # 将张量移动到指定设备
                return x_in.to(self.unet.device)
            # 将输入转换为 Torch 张量，并移动到指定设备
            return torch.tensor(x_in, device=self.unet.device)
    
    # 定义重置输入状态的方法
        def reset_x0(self, x_in, cond, act_dim):
            # 遍历条件字典中的每个键值对
            for key, val in cond.items():
                # 用条件值的克隆来更新输入的特定部分
                x_in[:, key, act_dim:] = val.clone()
            # 返回更新后的输入
            return x_in
    
    # 定义运行扩散过程的方法
        def run_diffusion(self, x, conditions, n_guide_steps, scale):
            # 获取输入的批次大小
            batch_size = x.shape[0]
            # 初始化输出
            y = None
            # 遍历调度器的每个时间步
            for i in tqdm.tqdm(self.scheduler.timesteps):
                # 创建用于传递给模型的时间步批次
                timesteps = torch.full((batch_size,), i, device=self.unet.device, dtype=torch.long)
                # 对于每个引导步骤
                for _ in range(n_guide_steps):
                    # 启用梯度计算
                    with torch.enable_grad():
                        # 设置输入张量为需要梯度计算
                        x.requires_grad_()
    
                        # 变换维度以匹配预训练模型的输入格式
                        y = self.value_function(x.permute(0, 2, 1), timesteps).sample
                        # 计算损失的梯度
                        grad = torch.autograd.grad([y.sum()], [x])[0]
    
                        # 获取当前时间步的后验方差
                        posterior_variance = self.scheduler._get_variance(i)
                        # 计算模型的标准差
                        model_std = torch.exp(0.5 * posterior_variance)
                        # 根据标准差缩放梯度
                        grad = model_std * grad
    
                    # 对于前两个时间步，设置梯度为零
                    grad[timesteps < 2] = 0
                    # 分离计算图，防止反向传播
                    x = x.detach()
                    # 更新输入张量，增加缩放后的梯度
                    x = x + scale * grad
                    # 使用条件重置输入张量
                    x = self.reset_x0(x, conditions, self.action_dim)
    
                # 使用 UNet 模型生成前一步的样本
                prev_x = self.unet(x.permute(0, 2, 1), timesteps).sample.permute(0, 2, 1)
    
                # TODO: 验证此关键字参数的弃用情况
                # 根据调度器步骤更新输入张量
                x = self.scheduler.step(prev_x, i, x)["prev_sample"]
    
                # 将条件应用于轨迹（设置初始状态）
                x = self.reset_x0(x, conditions, self.action_dim)
                # 将输入转换为 Torch 张量
                x = self.to_torch(x)
            # 返回最终输出和生成的样本
            return x, y
    # 定义调用方法，接收观测值及其他参数
    def __call__(self, obs, batch_size=64, planning_horizon=32, n_guide_steps=2, scale=0.1):
        # 归一化观测值并创建批次维度
        obs = self.normalize(obs, "observations")
        # 在第一个维度上重复观测值以形成批次
        obs = obs[None].repeat(batch_size, axis=0)
    
        # 将观测值转换为 PyTorch 张量，并创建条件字典
        conditions = {0: self.to_torch(obs)}
        # 定义输出张量的形状
        shape = (batch_size, planning_horizon, self.state_dim + self.action_dim)
    
        # 生成初始噪声并应用条件，使轨迹从当前状态开始
        x1 = randn_tensor(shape, device=self.unet.device)
        # 重置噪声张量，使其符合条件
        x = self.reset_x0(x1, conditions, self.action_dim)
        # 将张量转换为 PyTorch 格式
        x = self.to_torch(x)
    
        # 运行扩散过程以生成轨迹
        x, y = self.run_diffusion(x, conditions, n_guide_steps, scale)
    
        # 按值对输出轨迹进行排序
        sorted_idx = y.argsort(0, descending=True).squeeze()
        # 根据排序索引获取对应的值
        sorted_values = x[sorted_idx]
        # 提取行动部分
        actions = sorted_values[:, :, : self.action_dim]
        # 将张量转换为 NumPy 数组并分离
        actions = actions.detach().cpu().numpy()
        # 反归一化行动
        denorm_actions = self.de_normalize(actions, key="actions")
    
        # 选择具有最高值的行动
        if y is not None:
            # 如果存在值，引导选择索引为 0
            selected_index = 0
        else:
            # 如果没有运行值引导，随机选择一个行动
            selected_index = np.random.randint(0, batch_size)
    
        # 获取选中的反归一化行动
        denorm_actions = denorm_actions[selected_index, 0]
        # 返回最终选定的行动
        return denorm_actions
```