# `.\lucidrains\vit-pytorch\vit_pytorch\mae.py`

```
# 导入 PyTorch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn.functional 模块中导入 F
import torch.nn.functional as F
# 从 einops 库中导入 repeat 函数
from einops import repeat

# 从 vit_pytorch.vit 模块中导入 Transformer 类
from vit_pytorch.vit import Transformer

# 定义一个名为 MAE 的 nn.Module 类
class MAE(nn.Module):
    # 初始化函数，接收一系列参数
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        # 断言确保 masking_ratio 在 0 和 1 之间
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        # 将 masking_ratio 存储在对象中
        self.masking_ratio = masking_ratio

        # 从编码器中提取一些超参数和函数（待训练的视觉变换器）

        # 存储编码器对象
        self.encoder = encoder
        # 获取补丁数量和编码器维度
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        # 获取从图像到补丁的转换函数
        self.to_patch = encoder.to_patch_embedding[0]
        # 获取从补丁到嵌入的序列
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        # 获取每个补丁的像素值
        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # 解码器参数
        # 存储解码器维度
        self.decoder_dim = decoder_dim
        # 如果编码器维度与解码器维度不同，则使用 nn.Linear 进行映射，否则使用 nn.Identity
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        # 初始化一个可学习的遮罩令牌
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        # 创建一个 Transformer 解码器
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        # 创建一个嵌入层用于解码器位置编码
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        # 创建一个线性层用于将解码器输出映射回像素值
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
    # 定义一个前向传播函数，接收输入图像
    def forward(self, img):
        # 获取输入图像所在设备
        device = img.device

        # 获取图像的补丁

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # 将补丁转换为编码器标记并添加位置信息

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # 计算需要屏蔽的补丁数量，并获取随机索引，将其分为屏蔽和未屏蔽的部分

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # 获取要编码的未屏蔽标记

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # 获取用于最终重建损失的要屏蔽的补丁

        masked_patches = patches[batch_range, masked_indices]

        # 使用视觉变换器进行注意力

        encoded_tokens = self.encoder.transformer(tokens)

        # 投影编码器到解码器维度，如果它们不相等 - 论文中说可以使用较小的维度进行解码器

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # 重新应用解码器位置嵌入到未屏蔽标记

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # 重复屏蔽标记以匹配屏蔽数量，并使用上面得到的屏蔽索引添加位置

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # 将屏蔽标记连接到解码器标记并使用解码器进行注意力

        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # 剪切出屏蔽标记并投影到像素值

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # 计算重建损失

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss
```