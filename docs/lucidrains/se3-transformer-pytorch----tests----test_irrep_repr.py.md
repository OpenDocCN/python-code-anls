# `.\lucidrains\se3-transformer-pytorch\tests\test_irrep_repr.py`

```py
# 导入 torch 库
import torch
# 从 se3_transformer_pytorch.spherical_harmonics 模块中导入 clear_spherical_harmonics_cache 函数
from se3_transformer_pytorch.spherical_harmonics import clear_spherical_harmonics_cache
# 从 se3_transformer_pytorch.irr_repr 模块中导入 spherical_harmonics, irr_repr, compose 函数
from se3_transformer_pytorch.irr_repr import spherical_harmonics, irr_repr, compose
# 从 se3_transformer_pytorch.utils 模块中导入 torch_default_dtype 函数
from se3_transformer_pytorch.utils import torch_default_dtype

# 使用 torch.float64 作为默认数据类型
@torch_default_dtype(torch.float64)
# 定义测试函数 test_irr_repr
def test_irr_repr():
    """
    This test tests that
    - irr_repr
    - compose
    - spherical_harmonics
    are compatible

    Y(Z(alpha) Y(beta) Z(gamma) x) = D(alpha, beta, gamma) Y(x)
    with x = Z(a) Y(b) eta
    """
    # 循环遍历阶数范围为 0 到 6
    for order in range(7):
        # 生成两个随机数 a, b
        a, b = torch.rand(2)
        # 生成三个随机数 alpha, beta, gamma
        alpha, beta, gamma = torch.rand(3)

        # 计算 compose(alpha, beta, gamma, a, b, 0) 的结果
        ra, rb, _ = compose(alpha, beta, gamma, a, b, 0)
        # 计算 spherical_harmonics(order, ra, rb) 的结果
        Yrx = spherical_harmonics(order, ra, rb)
        # 清除球谐函数缓存
        clear_spherical_harmonics_cache()

        # 计算 spherical_harmonics(order, a, b) 的结果
        Y = spherical_harmonics(order, a, b)
        # 清除球谐函数缓存
        clear_spherical_harmonics_cache()

        # 计算 irr_repr(order, alpha, beta, gamma) @ Y 的结果
        DrY = irr_repr(order, alpha, beta, gamma) @ Y

        # 计算 (Yrx - DrY).abs().max() 和 Y.abs().max() 的最大值
        d, r = (Yrx - DrY).abs().max(), Y.abs().max()
        # 打印结果
        print(d.item(), r.item())
        # 断言 d < 1e-10 * r，如果不成立则抛出异常
        assert d < 1e-10 * r, d / r
```