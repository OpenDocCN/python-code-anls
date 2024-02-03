# `stable-diffusion-webui\modules\codeformer\vqgan_arch.py`

```py
# 从 CodeFormer 仓库复制的文件，请查看 modules/codeformer_model.py 中的注释

'''
VQGAN 代码，改编自 Unleashing Transformers 作者创建的原始代码:
https://github.com/samb-t/unleashing-transformers/blob/master/models/vqgan.py

'''
# 导入所需的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

# 定义归一化函数
def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# 定义激活函数 swish
@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)

# 定义 VQVAE 类
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # 嵌入的数量
        self.emb_dim = emb_dim  # 嵌入的维度
        self.beta = beta  # 在损失项中使用的承诺成本，beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)  # 嵌入层
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)  # 初始化嵌入权重
    # 前向传播函数，接受输入 z
    def forward(self, z):
        # 重新排列 z 的维度，将其变为 (batch, height, width, channel) 并展平
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        # 计算 z 到嵌入 e_j 的距离 (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        mean_distance = torch.mean(d)
        # 找到最接近的编码
        # min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encoding_scores, min_encoding_indices = torch.topk(d, 1, dim=1, largest=False)
        # [0-1], 得分越高，置信度越高
        min_encoding_scores = torch.exp(-min_encoding_scores/10)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # 获取量化后的潜在向量
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # 计算嵌入的损失
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # 保留梯度
        z_q = z + (z_q - z).detach()

        # 困惑度
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # 将形状重新排列以匹配原始输入形状
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "min_encoding_scores": min_encoding_scores,
            "mean_distance": mean_distance
            }
    # 获取编码簿特征，根据给定的索引和形状
    def get_codebook_feat(self, indices, shape):
        # 输入的索引是 batch*token_num 的形状，将其转换为 (batch*token_num)*1 的形状
        indices = indices.view(-1,1)
        # 创建一个全零张量，形状为 (batch*token_num, self.codebook_size)，并将其移到与 indices 相同的设备上
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        # 使用索引在 min_encodings 中进行填充，将对应位置的值设为 1
        min_encodings.scatter_(1, indices, 1)
        # 获取量化后的潜在向量
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # 如果给定了形状，则将 z_q 重新调整为原始输入形状
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        # 返回 z_q
        return z_q
# 定义 GumbelQuantizer 类，用于量化编码
class GumbelQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, straight_through=False, kl_weight=5e-4, temp_init=1.0):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings，嵌入的数量
        self.emb_dim = emb_dim  # dimension of embedding，嵌入的维度
        self.straight_through = straight_through  # flag for straight through Gumbel softmax
        self.temperature = temp_init  # initial temperature for Gumbel softmax
        self.kl_weight = kl_weight  # weight for KL divergence loss
        self.proj = nn.Conv2d(num_hiddens, codebook_size, 1)  # projects last encoder layer to quantized logits，将最后一个编码器层投影到量化的对数概率
        self.embed = nn.Embedding(codebook_size, emb_dim)  # embedding layer for quantized codes

    def forward(self, z):
        hard = self.straight_through if self.training else True  # use straight through Gumbel softmax during training

        logits = self.proj(z)  # project input to logits

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)  # apply Gumbel softmax to get soft one-hot encoding

        z_q = torch.einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)  # quantize input based on soft one-hot encoding

        # calculate KL divergence loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()

        min_encoding_indices = soft_one_hot.argmax(dim=1)  # get the index of the minimum encoding

        return z_q, diff, {
            "min_encoding_indices": min_encoding_indices
        }

# 定义 Downsample 类，用于下采样
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)  # convolutional layer for downsampling

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)  # zero-padding
        x = self.conv(x)  # apply convolution for downsampling
        return x

# 定义 Upsample 类，用于上采样
class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)  # convolutional layer for upsampling

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")  # interpolate to upsample by a factor of 2
        x = self.conv(x)  # apply convolution for upsampling

        return x

# 定义 ResBlock 类，用于残差块
class ResBlock(nn.Module):
    # 初始化 ResBlock 类，设置输入通道数和输出通道数
    def __init__(self, in_channels, out_channels=None):
        # 调用父类的初始化方法
        super(ResBlock, self).__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 如果未指定输出通道数，则输出通道数与输入通道数相同
        self.out_channels = in_channels if out_channels is None else out_channels
        # 初始化第一个归一化层
        self.norm1 = normalize(in_channels)
        # 初始化第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 初始化第二个归一化层
        self.norm2 = normalize(out_channels)
        # 初始化第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入通道数不等于输出通道数，则初始化输出通道的卷积层
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播函数
    def forward(self, x_in):
        # 将输入赋值给 x
        x = x_in
        # 对 x 进行归一化
        x = self.norm1(x)
        # 对 x 进行激活函数处理
        x = swish(x)
        # 对 x 进行第一个卷积操作
        x = self.conv1(x)
        # 对 x 进行归一化
        x = self.norm2(x)
        # 对 x 进行激活函数处理
        x = swish(x)
        # 对 x 进行第二个卷积操作
        x = self.conv2(x)
        # 如果输入通道数不等于输出通道数，则对输入进行输出通道的卷积操作
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        # 返回 x 与 x_in 的和作为输出
        return x + x_in
# 定义注意力块的类，继承自 nn.Module
class AttnBlock(nn.Module):
    # 初始化函数，接受输入通道数
    def __init__(self, in_channels):
        # 调用父类的初始化函数
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 初始化归一化层
        self.norm = normalize(in_channels)
        # 初始化查询卷积层
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # 初始化键卷积层
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # 初始化值卷积层
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # 初始化输出投影卷积层
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    # 前向传播函数，接受输入张量 x
    def forward(self, x):
        # 保存输入张量 x
        h_ = x
        # 对输入张量进行归一化
        h_ = self.norm(h_)
        # 计算查询张量 q
        q = self.q(h_)
        # 计算键张量 k
        k = self.k(h_)
        # 计算值张量 v

        # 计算注意力权重
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # 对值进行注意力聚合
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        # 使用输出投影层处理注意力聚合结果
        h_ = self.proj_out(h_)

        # 返回输入张量与注意力聚合结果的和
        return x+h_


class Encoder(nn.Module):
    # 初始化函数，设置模型的参数和结构
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的参数
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # 添加初始卷积层
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # 添加残差块和下采样块，对较小分辨率（16x16）进行注意力处理
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # 添加非局部注意力块
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # 归一化并转换为潜在空间大小
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    # 前向传播函数
    def forward(self, x):
        # 逐个应用模型中的块
        for block in self.blocks:
            x = block(x)

        return x
# 定义生成器类，继承自 nn.Module
class Generator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions-1)

        blocks = []
        # 初始卷积层
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # 非局部注意力块
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)


    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


# 注册 VQAutoEncoder 类到 ARCH_REGISTRY
@ARCH_REGISTRY.register()
class VQAutoEncoder(nn.Module):
    def forward(self, x):
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        x = self.generator(quant)
        return x, codebook_loss, quant_stats



# 基于补丁的鉴别器
# 注册 VQGANDiscriminator 类到 ARCH_REGISTRY
@ARCH_REGISTRY.register()
class VQGANDiscriminator(nn.Module):
    # 初始化函数，定义模型结构和参数
    def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):
        super().__init__()

        # 定义初始卷积层和激活函数
        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        # 逐渐增加滤波器数量的卷积层
        for n in range(1, n_layers):
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        # 添加最后一层卷积层和激活函数
        layers += [
            nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # 添加输出层，输出1通道预测图
        layers += [
            nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]
        # 将所有层组合成一个序列
        self.main = nn.Sequential(*layers)

        # 如果提供了模型路径，则加载模型参数
        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_d' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
            else:
                raise ValueError('Wrong params!')

    # 前向传播函数，返回模型输出
    def forward(self, x):
        return self.main(x)
```