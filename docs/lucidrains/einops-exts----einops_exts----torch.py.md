# `.\lucidrains\einops-exts\einops_exts\torch.py`

```
# 导入 torch 中的 nn 模块
# 导入 einops 中的 rearrange 函数
from torch import nn
from einops import rearrange

# 定义一个用于转换和重组数据的类 EinopsToAndFrom
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        # 初始化类的属性
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

        # 检查 from_einops 中是否包含 '...'
        if '...' in from_einops:
            # 如果包含 '...'，则将其分割成 before 和 after 两部分
            before, after = [part.strip().split() for part in from_einops.split('...')]
            # 生成重组键值对，包括 before 和 after 部分
            self.reconstitute_keys = tuple(zip(before, range(len(before)))) + tuple(zip(after, range(-len(after), 0)))
        else:
            # 如果不包含 '...'，则直接按空格分割成键值对
            split = from_einops.strip().split()
            self.reconstitute_keys = tuple(zip(split, range(len(split)))

    # 定义前向传播函数
    def forward(self, x, **kwargs):
        # 获取输入 x 的形状
        shape = x.shape
        # 根据 reconstitute_keys 生成重组参数字典
        reconstitute_kwargs = {key: shape[position] for key, position in self.reconstitute_keys}
        # 对输入 x 进行从 from_einops 到 to_einops 的重组
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        # 对重组后的 x 进行处理
        x = self.fn(x, **kwargs)
        # 将处理后的 x 重新从 to_einops 重组回 from_einops
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        # 返回处理后的 x
        return x
```