# `.\lucidrains\lie-transformer-pytorch\tests.py`

```py
# 导入 torch 库
import torch
# 从 lie_transformer_pytorch 库中导入 LieTransformer 类
from lie_transformer_pytorch import LieTransformer

# 定义测试 LieTransformer 类的函数
def test_transformer():
    # 创建 LieTransformer 模型对象，设置维度为 512，深度为 1
    model = LieTransformer(
        dim = 512,
        depth = 1
    )

    # 生成一个形状为 (1, 64, 512) 的随机张量 feats
    feats = torch.randn(1, 64, 512)
    # 生成一个形状为 (1, 64, 3) 的随机张量 coors
    coors = torch.randn(1, 64, 3)
    # 生成一个形状为 (1, 64) 的全为 True 的布尔张量 mask
    mask = torch.ones(1, 64).bool()

    # 使用 LieTransformer 模型处理 feats, coors 和 mask，得到输出 out
    out = model(feats, coors, mask = mask)
    # 断言输出 out 的形状为 (1, 256, 512)，如果不是则输出 'transformer runs'
    assert out.shape == (1, 256, 512), 'transformer runs'
```