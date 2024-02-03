# `.\PaddleOCR\ppocr\modeling\backbones\rec_vitstr.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用的代码来源于：
# https://github.com/roatienza/deep-text-recognition-benchmark/blob/master/modules/vitstr.py

import numpy as np
import paddle
import paddle.nn as nn
from ppocr.modeling.backbones.rec_svtrnet import Block, PatchEmbed, zeros_, trunc_normal_, ones_

# 定义不同规模的 ViTSTR 模型的头部维度和头数
scale_dim_heads = {'tiny': [192, 3], 'small': [384, 6], 'base': [768, 12]}

# 定义 ViTSTR 类
class ViTSTR(nn.Layer):
    # 初始化权重
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    # 前向特征提取
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = paddle.tile(self.cls_token, repeat_times=[B, 1, 1])
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    # 完整的前向传播
    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, :self.seqlen]
        return x.transpose([0, 2, 1]).unsqueeze(2)
```