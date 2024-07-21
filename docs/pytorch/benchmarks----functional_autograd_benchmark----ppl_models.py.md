# `.\pytorch\benchmarks\functional_autograd_benchmark\ppl_models.py`

```py
# 从自定义模块 utils 中导入 GetterReturnType 类型
from utils import GetterReturnType

# 导入 PyTorch 库
import torch
import torch.distributions as dist
from torch import Tensor

# 定义一个返回类型为 GetterReturnType 的函数，用于简单回归
def get_simple_regression(device: torch.device) -> GetterReturnType:
    # 设置样本数 N 和特征数 K
    N = 10
    K = 10

    # 设定 beta 先验的均值和标准差
    loc_beta = 0.0
    scale_beta = 1.0

    # 创建 beta 先验分布对象，正态分布
    beta_prior = dist.Normal(loc_beta, scale_beta)

    # 生成随机输入特征 X 和随机目标值 Y，均在指定设备上
    X = torch.rand(N, K + 1, device=device)
    Y = torch.rand(N, 1, device=device)

    # 从 beta_prior 中采样得到 beta_value，需要梯度追踪
    beta_value = beta_prior.sample((K + 1, 1))
    beta_value.requires_grad_(True)

    # 定义内部函数 forward，接受 beta_value 并返回评分 score
    def forward(beta_value: Tensor) -> Tensor:
        # 计算预测值 mu，X 与 beta_value 的矩阵乘积
        mu = X.mm(beta_value)

        # 计算分数 score，包括 Bernoulli 分布对 Y 的对数概率和 beta_prior 对 beta_value 的对数概率
        # 关闭 Bernoulli 分布的参数验证，因为 Y 是放松值
        score = (
            dist.Bernoulli(logits=mu, validate_args=False).log_prob(Y).sum()
            + beta_prior.log_prob(beta_value).sum()
        )
        return score

    # 返回 forward 函数和包含 beta_value 的元组，表示此函数的输出
    return forward, (beta_value.to(device),)


# 定义一个返回类型为 GetterReturnType 的函数，用于鲁棒回归
def get_robust_regression(device: torch.device) -> GetterReturnType:
    # 设置样本数 N 和特征数 K
    N = 10
    K = 10

    # 生成随机输入特征 X 和随机目标值 Y，均在指定设备上
    X = torch.rand(N, K + 1, device=device)
    Y = torch.rand(N, 1, device=device)

    # 预定义参数 nu_alpha 和 nu_beta，用于 Gamma 分布
    nu_alpha = torch.rand(1, 1, device=device)
    nu_beta = torch.rand(1, 1, device=device)
    nu = dist.Gamma(nu_alpha, nu_beta)

    # 预定义参数 sigma_rate，用于指数分布
    sigma_rate = torch.rand(N, 1, device=device)
    sigma = dist.Exponential(sigma_rate)

    # 预定义参数 beta_mean 和 beta_sigma，用于正态分布
    beta_mean = torch.rand(K + 1, 1, device=device)
    beta_sigma = torch.rand(K + 1, 1, device=device)
    beta = dist.Normal(beta_mean, beta_sigma)

    # 从 nu 中采样得到 nu_value，需要梯度追踪
    nu_value = nu.sample()
    nu_value.requires_grad_(True)

    # 从 sigma 中采样得到 sigma_value，并转换为未约束的对数值，需要梯度追踪
    sigma_value = sigma.sample()
    sigma_unconstrained_value = sigma_value.log()
    sigma_unconstrained_value.requires_grad_(True)

    # 从 beta 中采样得到 beta_value，需要梯度追踪
    beta_value = beta.sample()
    beta_value.requires_grad_(True)

    # 定义内部函数 forward，接受 nu_value、sigma_unconstrained_value 和 beta_value，并返回评分 score
    def forward(
        nu_value: Tensor, sigma_unconstrained_value: Tensor, beta_value: Tensor
    ) -> Tensor:
        # 此处未提供完整的函数定义，需在后续代码中继续完善
        pass
    # 定义一个函数，返回类型为 Tensor
    ) -> Tensor:
        # 将未约束的 sigma 值取指数，得到约束后的值
        sigma_constrained_value = sigma_unconstrained_value.exp()
        # 计算 mu，即 X 与 beta_value 的矩阵乘积
        mu = X.mm(beta_value)

        # 计算 nu_score，表示对 nu_value 求梯度的得分
        nu_score = dist.StudentT(nu_value, mu, sigma_constrained_value).log_prob(
            Y
        ).sum() + nu.log_prob(nu_value)

        # 计算 sigma_score，表示对 sigma_unconstrained_value 求梯度的得分
        sigma_score = (
            dist.StudentT(nu_value, mu, sigma_constrained_value).log_prob(Y).sum()
            + sigma.log_prob(sigma_constrained_value)
            + sigma_unconstrained_value
        )

        # 计算 beta_score，表示对 beta_value 求梯度的得分
        beta_score = dist.StudentT(nu_value, mu, sigma_constrained_value).log_prob(
            Y
        ).sum() + beta.log_prob(beta_value)

        # 返回 nu_score、sigma_score 和 beta_score 的总和
        return nu_score.sum() + sigma_score.sum() + beta_score.sum()

    # 返回 forward 函数及其参数元组，将这些参数移动到指定设备上
    return forward, (
        nu_value.to(device),
        sigma_unconstrained_value.to(device),
        beta_value.to(device),
    )
```