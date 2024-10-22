# `.\cogview3-finetune\sat\sgm\modules\distributions\distributions.py`

```py
# 导入 NumPy 库，通常用于数值计算
import numpy as np
# 导入 PyTorch 库，主要用于深度学习计算
import torch


# 定义抽象分布类，继承自基础类
class AbstractDistribution:
    # 抽象方法，样本生成
    def sample(self):
        # 抛出未实现错误，子类需实现此方法
        raise NotImplementedError()

    # 抽象方法，返回分布的众数
    def mode(self):
        # 抛出未实现错误，子类需实现此方法
        raise NotImplementedError()


# 定义 Dirac 分布类，继承自抽象分布类
class DiracDistribution(AbstractDistribution):
    # 初始化方法，接收一个值作为分布的值
    def __init__(self, value):
        # 将传入的值存储为实例变量
        self.value = value

    # 实现样本生成方法
    def sample(self):
        # 返回 Dirac 分布的值
        return self.value

    # 实现众数方法
    def mode(self):
        # 返回 Dirac 分布的值
        return self.value


# 定义对角高斯分布类
class DiagonalGaussianDistribution(object):
    # 初始化方法，接收参数和一个决定是否确定性的标志
    def __init__(self, parameters, deterministic=False):
        # 将参数存储为实例变量
        self.parameters = parameters
        # 将参数分割成均值和对数方差
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # 限制对数方差在 -30 到 20 之间
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        # 存储确定性标志
        self.deterministic = deterministic
        # 计算标准差
        self.std = torch.exp(0.5 * self.logvar)
        # 计算方差
        self.var = torch.exp(self.logvar)
        # 如果是确定性，方差和标准差设为零
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    # 实现样本生成方法
    def sample(self):
        # x = self.mean + self.std * torch.randn(self.mean.shape).to(
        #     device=self.parameters.device
        # )
        # 生成样本，遵循均值和标准差的分布
        x = self.mean + self.std * torch.randn_like(self.mean).to(
            device=self.parameters.device
        )
        # 返回生成的样本
        return x

    # 实现 KL 散度计算方法
    def kl(self, other=None):
        # 如果是确定性，返回 0.0 的张量
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            # 如果没有其他分布，计算与标准正态分布的 KL 散度
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                # 计算与另一分布的 KL 散度
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    # 实现负对数似然计算方法
    def nll(self, sample, dims=[1, 2, 3]):
        # 如果是确定性，返回 0.0 的张量
        if self.deterministic:
            return torch.Tensor([0.0])
        # 计算 2π 的对数值
        logtwopi = np.log(2.0 * np.pi)
        # 计算负对数似然并返回
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    # 实现众数计算方法
    def mode(self):
        # 返回均值作为众数
        return self.mean


# 定义计算两个高斯分布之间 KL 散度的函数
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    来源: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    计算两个高斯分布之间的 KL 散度。
    形状会自动广播，因此批次可以与标量等进行比较，
    适用于其他用例。
    """
    # 初始化张量变量
    tensor = None
    # 遍历四个输入对象，找到第一个张量
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    # 确保至少一个输入是张量
    assert tensor is not None, "at least one argument must be a Tensor"

    # 强制方差为张量类型。广播帮助将标量转换为
    # 张量，但对 torch.exp() 不起作用。
    # 将 logvar1 和 logvar2 进行处理，确保它们都是张量类型
        logvar1, logvar2 = [
            # 如果 x 是张量，保持不变；否则，将 x 转换为张量并移动到指定设备
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)  # 遍历 logvar1 和 logvar2
        ]
    
        # 计算并返回一个值，公式由多个部分组成
        return 0.5 * (
            # -1.0 表示计算的偏移量
            -1.0
            + logvar2  # 加上 logvar2 的值
            - logvar1  # 减去 logvar1 的值
            + torch.exp(logvar1 - logvar2)  # 加上 logvar1 和 logvar2 之差的指数
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)  # 加上均值差的平方乘以 logvar2 的负指数
        )
```