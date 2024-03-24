# `.\lucidrains\enformer-pytorch\test_pretrained.py`

```
# 导入 torch 库
import torch
# 从 enformer_pytorch 库中导入 from_pretrained 函数
from enformer_pytorch import from_pretrained

# 从预训练模型 'EleutherAI/enformer-official-rough' 中加载模型，不使用 TF Gamma 参数，将模型放在 GPU 上
enformer = from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma = False).cuda()
# 将模型设置为评估模式
enformer.eval()

# 从文件 './data/test-sample.pt' 中加载数据
data = torch.load('./data/test-sample.pt')
# 将数据中的 'sequence' 和 'target' 转移到 GPU 上
seq, target = data['sequence'].cuda(), data['target'].cuda()

# 禁用梯度计算
with torch.no_grad():
    # 使用 enformer 模型进行推理，计算相关系数
    corr_coef = enformer(
        seq,
        target = target,
        return_corr_coef = True,
        head = 'human'
    )

# 打印相关系数
print(corr_coef)
# 断言相关系数大于 0.1
assert corr_coef > 0.1
```