# `.\cogvideo-finetune\sat\sgm\modules\distributions\distributions.py`

```py
# 导入 numpy 库并简化为 np
import numpy as np
# 导入 PyTorch 库
import torch


# 定义抽象类 AbstractDistribution
class AbstractDistribution:
    # 抽象方法，生成样本
    def sample(self):
        raise NotImplementedError()

    # 抽象方法，返回分布的众数
    def mode(self):
        raise NotImplementedError()


# 定义 DiracDistribution 类，继承自 AbstractDistribution
class DiracDistribution(AbstractDistribution):
    # 初始化 DiracDistribution，接受一个值
    def __init__(self, value):
        self.value = value

    # 重写 sample 方法，返回固定值
    def sample(self):
        return self.value

    # 重写 mode 方法，返回固定值
    def mode(self):
        return self.value


# 定义 DiagonalGaussianDistribution 类
class DiagonalGaussianDistribution(object):
    # 初始化，接受参数和一个确定性标志
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        # 将参数分为均值和对数方差
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # 将对数方差限制在指定范围内
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        # 计算标准差和方差
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        # 如果是确定性模式，将方差和标准差设为零
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    # 生成样本
    def sample(self):
        # 使用均值和标准差生成随机样本
        x = self.mean + self.std * torch.randn_like(self.mean).to(device=self.parameters.device)
        return x

    # 计算 KL 散度
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])  # 确定性时 KL 散度为 0
        else:
            if other is None:
                # 计算与标准正态分布的 KL 散度
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                # 计算两个分布间的 KL 散度
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    # 计算负对数似然
    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])  # 确定性时 NLL 为 0
        logtwopi = np.log(2.0 * np.pi)  # 计算 2π 的对数
        # 计算负对数似然值
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    # 返回均值
    def mode(self):
        return self.mean


# 定义 normal_kl 函数，计算两个高斯分布之间的 KL 散度
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    计算两个高斯分布之间的 KL 散度。
    形状会自动广播，支持批量比较和标量等用例。
    """
    tensor = None
    # 找到第一个是 Tensor 的对象
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    # 确保至少有一个参数是 Tensor
    assert tensor is not None, "at least one argument must be a Tensor"

    # 强制将方差转换为 Tensor
    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]
    # 计算并返回某种损失值，具体公式由多个部分组成
        return 0.5 * (  # 返回整个计算结果的一半
            -1.0 + logvar2 - logvar1 +  # 计算对数方差差值，并减去常数项
            torch.exp(logvar1 - logvar2) +  # 计算指数项，表示方差的相对差异
            ((mean1 - mean2) ** 2) * torch.exp(-logvar2)  # 计算均值差的平方乘以方差的倒数
        )
```