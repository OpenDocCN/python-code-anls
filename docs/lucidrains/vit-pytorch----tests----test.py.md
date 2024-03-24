# `.\lucidrains\vit-pytorch\tests\test.py`

```py
# 导入 torch 库
import torch
# 从 vit_pytorch 库中导入 ViT 类
from vit_pytorch import ViT

# 定义测试函数
def test():
    # 创建 ViT 模型对象，设置参数：图像大小为 256，patch 大小为 32，类别数为 1000，特征维度为 1024，深度为 6，注意力头数为 16，MLP 隐藏层维度为 2048，dropout 概率为 0.1，嵌入层 dropout 概率为 0.1
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    # 生成一个形状为 (1, 3, 256, 256) 的随机张量作为输入图像
    img = torch.randn(1, 3, 256, 256)

    # 将输入图像传入 ViT 模型进行预测
    preds = v(img)
    # 断言预测结果的形状为 (1, 1000)，如果不符合则抛出异常信息 'correct logits outputted'
    assert preds.shape == (1, 1000), 'correct logits outputted'
```