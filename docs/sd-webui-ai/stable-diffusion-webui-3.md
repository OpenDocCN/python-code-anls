# SDWebUI源码解析 3

# `modules/codeformer/codeformer_arch.py`

这段代码的作用是定义了一个名为`calc_mean_std`的函数，它接受一个4D张量作为输入参数，并计算出该张量的均值和方差。

这个函数是用来支持在代码中使用`adaptive_instance_normalization`的，这个功能会在网络的训练过程中实现一些数据增强的操作，帮助改善模型的表现。

具体来说，函数首先通过`math.cast`将输入张量的数据类型从long long to float，然后使用一个小的值`eps`来对计算结果进行四舍五入。接下来，函数创建了一个包含大小为`(batch_size, sequence_length, feature_dim)`的张量，将这个张量中的每个位置的值计算出来，并对这些值进行平方后再求均值和方差。最后，函数返回计算出的均值和方差。


```py
# this file is copied from CodeFormer repository. Please see comment in modules/codeformer_model.py

import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

from modules.codeformer.vqgan_arch import *
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


```

这段代码定义了一个名为 `adaptive_instance_normalization` 的函数，用于对内容特征（`content_feat`）和风格特征（`style_feat`）进行实例归一化处理。

具体来说，函数接受两个输入参数：内容特征和风格特征。函数内部首先计算内容特征和风格特征的均值和标准差，然后使用这些均值和标准差计算内容特征和风格特征的归一化特征。最后，函数将归一化特征与风格特征的均值和标准差相乘，然后加上风格特征的均值和标准差，从而得到一个新的特征，这个新特征可以帮助更好地适应不同的风格。


```py
def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


```

This is a PyTorch implementation of a 2D self-attention model for image data. The model takes in image data in the form of a tensor of shape `(batch_size, image_size, num_channels)` and outputs a tensor of shape `(batch_size, image_size, num_channels, num_pos_feats)`.

The model has a self-attention mechanism that computes attention between the input image and other images in the same batch. The attention mechanism is computed using a dot-product attention mechanism, where the query (received image) is multiplied by a key ( other images in the batch) and then added to a weight (e-value) to compute the attention. The dims of the input and output should be broadcasted across the channels dimension.

The model also has a normalization layer that normalizes the input image to have zero mean and unit variance, and a scale layer that multiplies the input image by a scale factor.

Note that this model is not implemented for continuous data, it is intended for discrete data (images). You may want to use this code as a starting point for implementing a custom image data积水层。


```py
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

```

This is a implementation of a neural network model for natural language processing (NLP) tasks, specifically a multi-head attention mechanism for pre-training language models on流转 data. The model is based on the transformers architecture and uses 2048 hidden layers with dropout rate 0.0, and an activation function of the form "gelu".

The attention mechanism uses a multi-head attention, where each head of the attention mechanism has 8 fixed-size parameters. The attention mechanism is applied to the input tensor "tgt" and the query key-padding-mask "tgt\_key\_padding\_mask" is an optional tensor that should be used for the attention computation.

The function `forward` is the main function for the model, which computes the forward pass of the model given the input tensor "tgt" and the optional key-padding-mask "tgt\_key\_padding\_mask".

The model also includes some custom components such as the normalization layers, which are used to transform the input data to have zero mean and unit variance.


```py
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

```

这段代码定义了一个名为 "Fuse_sft_block" 的自定义类，继承自 PyTorch 的nn.Module类。这个类在网络中的作用是实现数据块的 forward 方法。

在自定义类的初始化方法 "__init__" 中，首先调用父类的初始化方法，确保创建的实例包含了两个 ResBlock 模块。然后，创建一个从输入通道到输出通道的 encoder，以及一个从输出通道到输入通道的 decoder。最后，将 encoder 和 decoder 连接起来，组成一个完整的数据块。

数据块在 forward 方法中进行前向传播。首先将传入的 encoder 和 decoder 特征图拼接起来，然后将拼接好的前向特征图传递给第一个数据块中的一个 ResBlock 模块。接着，将 ResBlock 模块的输出送入第二个数据块中的一个 ResUnit（类似于 Group 中的一个成员）中。最后，对于每个数据包，都会执行一个 concatenate 操作，将输入的 decoder 部分和拼接好的部分拼接在一起，得到一个输出。最终的结果返回给目标变量。


```py
class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


```

This is a code snippet for a neural network model that uses generator and discriminator networks. It is written in PyTorch and appears to be in the training stage II of the cross-entropy loss.

The generator network takes input features and an image, and produces an output feature map. The generator has a list of connect blocks that include the convolutional and normalization blocks. The fuse\_list variable is a list of fused blocks that are activated after each concatenation block.

The discriminator network takes the output of the generator and produces a logit. It also takes an image and produces an output feature map. The discriminator has a list of connect blocks that include the convolutional and normalization blocks.

The training stage II loss for the generator is defined as the negative log loss over the input features and the output feature map. The loss for the discriminator is defined as the negative log loss over the input features and the output logit.

It appears that the code also performs quantization of the input and output features to reduce the memory usage for the training process.


```py
@ARCH_REGISTRY.register()
class CodeFormer(VQAutoEncoder):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9, 
                codebook_size=1024, latent_size=256,
                connect_list=['32', '64', '128', '256'],
                fix_modules=['quantize','generator']):
        super(CodeFormer, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest',2, [16], codebook_size)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) 
                                    for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))
        
        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'512':2, '256':5, '128':8, '64':11, '32':14, '16':18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {'16':6, '32': 9, '64':12, '128':15, '256':18, '512':21}

        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        # ################### Encoder #####################
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x) 
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()

        lq_feat = x
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1,x.shape[0],1)
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2,0,1))
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb) # (hw)bn
        logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only: # for training stage II
          # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0],16,16,256])
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()

        if detach_16:
            quant_feat = quant_feat.detach() # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]

        for i, block in enumerate(self.generator.blocks):
            x = block(x) 
            if i in fuse_list: # fuse after i-th block
                f_size = str(x.shape[-1])
                if w>0:
                    x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        return out, logits, lq_feat
```

# `modules/codeformer/vqgan_arch.py`

这段代码是一个基于Leunghishing VAE模型的修改版。它的主要目的是实现一个将文本转换为图像的模型。这个模型使用了来自CodeFormer项目的源代码，并在其基础上进行了修改。它主要用于研究和工作，以便更好地处理自然语言文本和图像之间的关系。

具体来说，这段代码包含以下几个主要部分：

1. 导入所需的可视化库（如torchvision和torchtext），以及一个用于处理文本数据的函数。

2. 通过创建一个自定义的类来表示输入数据，包括文本数据、图像尺寸等。

3. 实现了一个文本到图像的模型，该模型基于Leunghishing VAE的架构，并使用了一种称为“VQGAN”的技术，允许对文本数据进行无监督的图像生成。

4. 通过训练和优化器来训练模型，并将其保存到一个文件中。

5. 在代码的最后，作者提供了一些开发指导，以便其他人可以理解和使用这个模型。


```py
# this file is copied from CodeFormer repository. Please see comment in modules/codeformer_model.py

'''
VQGAN code, adapted from the original created by the Unleashing Transformers authors:
https://github.com/samb-t/unleashing-transformers/blob/master/models/vqgan.py

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

```

This is a PyTorch implementation of a model called "MinEncoder". It takes a batch of input tokens and an integer sequence, and outputs a tensor of quantized latent vectors.

The model has an input layer that maps the input tokens to a tensor of minencoded latent vectors. This layer uses a linear transform that computes the mean distance between each input token and its corresponding quantized latent vector.

The model also has an output layer that maps the quantized latent vectors to a tensor that outputs the perplexity of each input token.

The model takes a parameter `self.codebook_size` which is the size of the codebook, i.e. the vocabulary, of the model. This parameter is set in the constructor of the model.

In the `get_codebook_feat` function, the minencoded latent vectors are retrieved for each input token by querying the codebook using the `index` of the input token. This function returns the quantized latent vectors, as well as some additional information that includes the perplexity of each input token.


```py
def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    

@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)


#  Define VQVAE classes
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        mean_distance = torch.mean(d)
        # find closest encodings
        # min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encoding_scores, min_encoding_indices = torch.topk(d, 1, dim=1, largest=False)
        # [0-1], higher score, higher confidence
        min_encoding_scores = torch.exp(-min_encoding_scores/10)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "min_encoding_scores": min_encoding_scores,
            "mean_distance": mean_distance
            }

    def get_codebook_feat(self, indices, shape):
        # input indices: batch*token_num -> (batch*token_num)*1
        # shape: batch, height, width, channel
        indices = indices.view(-1,1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q


```

这段代码定义了一个名为 GumbelQuantizer 的类，继承自 PyTorch 的 nn.Module 类。GumbelQuantizer 旨在对文本数据进行量化编码，以便于神经网络中处理。它实现了 Gumbel 算法，一种基于约束的神经网络，用于自然语言处理中的编码问题。

具体来说，这段代码的作用如下：

1. 初始化：在 GumbelQuantizer 的初始化方法中，设置了嵌入维度（emb_dim）、隐层维度（num_hiddens）、是否使用 Stranglong Through 算法、kl 正权重（kl_weight）和 temp_init 常量。这些参数是为了在训练时使用 Gumbel 算法。

2. 准备神经网络：创建了 GumbelQuantizer 继承类的实例。其中包括将最后一个编码层投影到量化编码的 proj 层，以及一个嵌入层（nn.Embedding）用于将文本数据转换为 Gumbel 数组。

3. 前向传播：实现了 GumbelQuantizer 的 forward 方法，用于前向传播文本数据到神经网络。这个方法中提取了编码层的 logits，通过 Stranglong Through 算法计算了 soft one-hot 向量，然后将其与提前计算好的kl 正权重相乘，得到了一个带有时效信息的 z_q 向量。最后，将这个 z_q 向量与输入的 z 数据一起输入到隐藏层中，输出一个包含 min_encoding_indices 的元组。

4. 计算损失：实现了 GumbelQuantizer 的 loss 方法，用于计算损失函数。这个方法中计算了与真实标签的差距的平方，然后通过对 min_encoding_indices 的约束来对 soft one-hot 向量进行惩罚。

5. 保存模型配置：在 GumbelQuantizer 的继承类的实例中，保存了上述设置中的参数。

GumbelQuantizer 通过对自然语言文本数据进行 Gumbel 编码，实现对文本数据的标准化编码。这种编码方法在神经网络中有着广泛的应用，特别是在自然语言处理领域。通过将原始数据映射到密集向量，Gumbel 编码可以提高模型的容量，同时 Stranglong Through 算法可以有效地提高模型的安全性。


```py
class GumbelQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, straight_through=False, kl_weight=5e-4, temp_init=1.0):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = nn.Conv2d(num_hiddens, codebook_size, 1)  # projects last encoder layer to quantized logits
        self.embed = nn.Embedding(codebook_size, emb_dim)

    def forward(self, z):
        hard = self.straight_through if self.training else True

        logits = self.proj(z)

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)

        z_q = torch.einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()
        min_encoding_indices = soft_one_hot.argmax(dim=1)

        return z_q, diff, {
            "min_encoding_indices": min_encoding_indices
        }


```

这段代码定义了两个类，Downsample 和 Upsample。这两个类继承自 PyTorch 的 nn.Module 类，用于实现图像插值。

具体来说，Downsample 类在 `__init__` 方法中定义了一个包含一个卷积层的函数，这个卷积层使用了两个参数，in_channels 和 ksize。在 `forward` 方法中，对输入的 x 进行处理，包括在 x 的前后添加了大小为 3x2 的卷积核，并执行了 stride=2,padding=0 的操作。最后，返回处理后的 x。

Upsample 类在 `__init__` 方法中定义了一个包含一个卷积层的函数，这个卷积层使用了三个参数，in_channels 和 ksize。在 `forward` 方法中，对输入的 x 进行处理，包括在 x 的前后添加了大小为 3x2 的卷积核，并执行了 stride=1,padding=1 的操作。最后，返回处理后的 x。

这两个类的 Downsample 和 Upsample 分别对输入图像 x 进行了不同的处理，使得输出图像可以插值到更大的尺寸上。其中，Downsample 的处理方式是对图像进行了等比例的缩放，而 Upsample 的处理方式是对图像进行了等比例的插值。


```py
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


```

该代码定义了一个名为ResBlock的类，继承自PyTorch中的nn.Module类。ResBlock用于在图像块（ResBlock）中执行卷积操作。该类包含了一个初始化函数__init__，该函数在__init__函数中指定了输入通道（in_channels）和输出通道（out_channels）。如果未提供out_channels参数，则默认为输入通道。在__init__函数之后，定义了一系列变量，包括输入通道的归一化（normalization）、第一个卷积层、第二个卷积层以及一个输出卷积层（如果输出通道未指定）。

在forward函数中，从前面的输入x_in开始，通过一系列卷积层和池化层，最终返回对输入图像块的加权和。在该函数中，还添加了一个输出卷积层，用于在图像块之间传递信息。该输出卷积层的参数是对于输入图像块的in_channels，而不是通过ResBlock计算得出的out_channels。因此，如果ResBlock初始化时未指定out_channels，该函数将使用默认的in_channels作为输出通道。


```py
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


```

This is a class definition for a neural network model, called "ProjectionModel". It appears to be a multi-layer neural network that takes a 2D input of shape `(batch_size, input_shape, input_shape)` and returns a 4D output of shape `(batch_size, input_shape, input_shape, h, w)`. It also has an additional `proj_out` parameter, which is a 2D parameter of shape `(batch_size, input_shape, h, w)`.

The model uses a combination of convolutional and attention mechanisms to produce the output. The convolutional layers have a kernel size of 1x1 and a stride of 1, and all the convolutional layers have a batch dimension of 0 to add a batch dimension to the output. The attention mechanism is a multi-head self-attention mechanism, which computes the attention between the input and the query (attention) and context (context) vectors.

The output of the last convolutional layer is passed through a projection layer to produce the final output. The shape of the output is (batch\_size, input\_shape, h, w).


```py
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k) 
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1) 
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


```

This is a Python implementation of the YOLOv5 model architecture, specifically the Multi-scale Faster R-CNN (MF-RCNN) model. It includes the multi-scale feature pyramid exposition, residual and downsampling blocks, and non-local attention mechanism. The model takes as input an input feature tensor `x`, and the output is another tensor `y` representing the object detection predictions.

It is worth noting that this implementation is not optimized or corrected for production use, and it may use some assumptions or experimental techniques.


```py
class Encoder(nn.Module):
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # initial convultion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
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

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        return x


```

This is a PyTorch implementation of a multi-scale feature fusion module, which can be used to improve the resolution of a low-resolution image. The module has a number of inline blocks with different feature resolvers, which allows it to capture features at different scales.

The module takes an input of size (batch\_size, image\_size, feature\_size) and returns the output. It uses a combination of attention mechanisms to ensure that the low-resolution image is resolved to the same level of the high-resolution image, which is determined by the number of in-channels.

The module also includes a normalization step, which applies the desired resolution to the output of each block. This allows the image to be resolved to a fixed resolution, which is determined by the number of out-channels.

Overall, this implementation is designed to improve the resolution of low-resolution images by providing a hierarchical fusion of features at different scales, and by ensuring that the low-resolution image is resolved to the same level of the high-resolution image.


```py
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
        # initial conv
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
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

  
```

This is a PyTorch implementation of a Variational Quantum Generator (VQG) model. It uses a Gumbel-type approach for quantization and a Nearest志算法作为进一步的量子化。VQG 模型主要包括编码器和解码器，以及参数中隐藏的参数。

该模型的主要结构包括：

- 编码器（Encoder）: 负责将输入 x 进行处理，包括多头注意力机制 (Multi-Head Attention)、通道注意力机制 (Channels Attention) 和位置编码 (Positional Encoding)。
- 解码器 (Decoder): 负责从编码器的输出结果开始重构输入 x。
- 量化器（Quantizer）: 负责将编码器的输出结果进行量化。
- 损失函数（Loss Function）: 负责评估模型。

此外，还包括一些辅助的代码来对模型进行初始化、评估和日志记录等操作。

这个模型是训练基于博弈论策略的，所以使用的量子化技术是 Gumbel 量子化技术。


```py
@ARCH_REGISTRY.register()
class VQAutoEncoder(nn.Module):
    def __init__(self, img_size, nf, ch_mult, quantizer="nearest", res_blocks=2, attn_resolutions=[16], codebook_size=1024, emb_dim=256,
                beta=0.25, gumbel_straight_through=False, gumbel_kl_weight=1e-8, model_path=None):
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 
        self.nf = nf 
        self.n_blocks = res_blocks 
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.quantizer_type = quantizer
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions
        )
        if self.quantizer_type == "nearest":
            self.beta = beta #0.25
            self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        elif self.quantizer_type == "gumbel":
            self.gumbel_num_hiddens = emb_dim
            self.straight_through = gumbel_straight_through
            self.kl_weight = gumbel_kl_weight
            self.quantize = GumbelQuantizer(
                self.codebook_size,
                self.embed_dim,
                self.gumbel_num_hiddens,
                self.straight_through,
                self.kl_weight
            )
        self.generator = Generator(
            self.nf, 
            self.embed_dim,
            self.ch_mult, 
            self.n_blocks, 
            self.resolution, 
            self.attn_resolutions
        )

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_ema' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'])
                logger.info(f'vqgan is loaded from: {model_path} [params_ema]')
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
                logger.info(f'vqgan is loaded from: {model_path} [params]')
            else:
                raise ValueError(f'Wrong params!')


    def forward(self, x):
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        x = self.generator(quant)
        return x, codebook_loss, quant_stats



```

This is a Python implementation of a neural network model using the LeakyReLU activation function. The model consists of a sequence of convolutional layers, batch normalization, and LeakyReLU activation.

The `LeakyReLU` function is defined with a parameter `theta` which controls the leak rate. When `theta=0`, the function returns a linear function that simply adds the input to the expected output. When `theta=1`, the function returns the input.

The `nn.LeakyReLU` function returns a LeakyReLU activation function with the specified `theta`.

The `nn.Conv2d` function is the `LeakyReLU` activation function used for the weights in the convolutional layers. The `stride` parameter specifies the stride of the convolution operation, and the `padding` parameter adds a small border to the convolution output.

The `nn.BatchNorm2d` function is the batch normalization layer.

The `nn.Conv2d` and `nn.BatchNorm2d` functions are repeated in the `layers` list for each block in the convolutional layers.

The `self.main` function returns the output of the last convolutional layer.

The model can be saved to a file using the `torch.save` function. The state dictionary can also be loaded using the similar `torch.load` function.


```py
# patch based discriminator
@ARCH_REGISTRY.register()
class VQGANDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):
        super().__init__()

        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        layers += [
            nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*layers)

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_d' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
            else:
                raise ValueError(f'Wrong params!')

    def forward(self, x):
        return self.main(x)
```

# `scripts/custom_code.py`

这段代码是一个Python脚本，它使用了Gradio库来创建一个交互式的图形界面。在这个脚本中，我们导入了两个模块，一个是Processed，另一个是shared。其中，Processed是一个处理图像数据的模块，shared是一个共同声明了几个变量的模块。我们还导入了两个函数，一个是Show函数，用来控制是否在界面上显示图像，另一个是run函数，用来执行实际的计算操作。

在脚本的run函数中，我们首先检查是否启用了--allow-code选项。如果不启用这个选项，我们会输出一个错误消息。如果启用了这个选项，我们编译了代码，创建了一个ModuleType对象，并将ModuleType对象的属性映射到了Globals对象上。然后，我们通过compile函数将代码编译成机器码，并执行这个机器码。

在执行机器码的过程中，我们会创建一个Processed对象，并将Processed对象传递给run函数。run函数的参数是p参数和一个或多个数字，它们会在运行脚本时传递给compiled函数。compiled函数会在运行脚本时执行代码，并返回一个Processed对象。


```py
import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):
    def title(self):
        return "Custom code"


    def show(self, is_img2img):
        return cmd_opts.allow_code

    def ui(self, is_img2img):
        code = gr.Textbox(label="Python code", visible=False, lines=1)

        return [code]

    def run(self, p, code):
        assert cmd_opts.allow_code, '--allow-code option must be enabled'

        display_result_data = [[], -1, ""]

        def display(imgs, s=display_result_data[1], i=display_result_data[2]):
            display_result_data[0] = imgs
            display_result_data[1] = s
            display_result_data[2] = i

        from types import ModuleType
        compiled = compile(code, '', 'exec')
        module = ModuleType("testmodule")
        module.__dict__.update(globals())
        module.p = p
        module.display = display
        exec(compiled, module.__dict__)

        return Processed(p, *display_result_data)


```

# `scripts/poor_mans_outpainting.py`

This is a Python function that performs an outpainting algorithm on a set of tiles. The function takes in a list of images, a list of tile data (RGB data), and a list of tile mask data (RGB data). The images and tile data are processed in parallel and the output is saved in a single combined image.

The function starts by initializing variables such as the number of images in the work pool, the number of tile layers, the number of tile tiles per layer, the number of images in a batch, and the index of the current batch.

The function then loops through the images and performs the outpainting using the `process_images()` function. The outpainting is done in parallel and the current batch is processed after each batch.

After all the images have been processed, the function combines the images using the `images.combine_grid()` function and saves the combined image if the `opts.samples_save` option is True.

The function also initializes a variable called `initial_seed` to 0 and a variable called `initial_info` to the empty dictionary.

The function then returns the `Processed` class object, which contains the processed images and additional metadata information.


```py
import math

import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw

from modules import images, processing
from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state



class Script(scripts.Script):
    def title(self):
        return "Poor man's outpainting"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=256, step=8, value=128)
        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, visible=False)
        inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='fill', type="index", visible=False)
        direction = gr.CheckboxGroup(label="Outpainting direction", choices=['left', 'right', 'up', 'down'], value=['left', 'right', 'up', 'down'])

        return [pixels, mask_blur, inpainting_fill, direction]

    def run(self, p, pixels, mask_blur, inpainting_fill, direction):
        initial_seed = None
        initial_info = None

        p.mask_blur = mask_blur * 2
        p.inpainting_fill = inpainting_fill
        p.inpaint_full_res = False

        left = pixels if "left" in direction else 0
        right = pixels if "right" in direction else 0
        up = pixels if "up" in direction else 0
        down = pixels if "down" in direction else 0

        init_img = p.init_images[0]
        target_w = math.ceil((init_img.width + left + right) / 64) * 64
        target_h = math.ceil((init_img.height + up + down) / 64) * 64

        if left > 0:
            left = left * (target_w - init_img.width) // (left + right)
        if right > 0:
            right = target_w - init_img.width - left

        if up > 0:
            up = up * (target_h - init_img.height) // (up + down)

        if down > 0:
            down = target_h - init_img.height - up

        img = Image.new("RGB", (target_w, target_h))
        img.paste(init_img, (left, up))

        mask = Image.new("L", (img.width, img.height), "white")
        draw = ImageDraw.Draw(mask)
        draw.rectangle((
            left + (mask_blur * 2 if left > 0 else 0),
            up + (mask_blur * 2 if up > 0 else 0),
            mask.width - right - (mask_blur * 2 if right > 0 else 0),
            mask.height - down - (mask_blur * 2 if down > 0 else 0)
        ), fill="black")

        latent_mask = Image.new("L", (img.width, img.height), "white")
        latent_draw = ImageDraw.Draw(latent_mask)
        latent_draw.rectangle((
             left + (mask_blur//2 if left > 0 else 0),
             up + (mask_blur//2 if up > 0 else 0),
             mask.width - right - (mask_blur//2 if right > 0 else 0),
             mask.height - down - (mask_blur//2 if down > 0 else 0)
        ), fill="black")

        processing.torch_gc()

        grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=pixels)
        grid_mask = images.split_grid(mask, tile_w=p.width, tile_h=p.height, overlap=pixels)
        grid_latent_mask = images.split_grid(latent_mask, tile_w=p.width, tile_h=p.height, overlap=pixels)

        p.n_iter = 1
        p.batch_size = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []
        work_mask = []
        work_latent_mask = []
        work_results = []

        for (y, h, row), (_, _, row_mask), (_, _, row_latent_mask) in zip(grid.tiles, grid_mask.tiles, grid_latent_mask.tiles):
            for tiledata, tiledata_mask, tiledata_latent_mask in zip(row, row_mask, row_latent_mask):
                x, w = tiledata[0:2]

                if x >= left and x+w <= img.width - right and y >= up and y+h <= img.height - down:
                    continue

                work.append(tiledata[2])
                work_mask.append(tiledata_mask[2])
                work_latent_mask.append(tiledata_latent_mask[2])

        batch_count = len(work)
        print(f"Poor man's outpainting will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)}.")

        state.job_count = batch_count

        for i in range(batch_count):
            p.init_images = [work[i]]
            p.image_mask = work_mask[i]
            p.latent_mask = work_latent_mask[i]

            state.job = f"Batch {i + 1} out of {batch_count}"
            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            p.seed = processed.seed + 1
            work_results += processed.images


        image_index = 0
        for y, h, row in grid.tiles:
            for tiledata in row:
                x, w = tiledata[0:2]

                if x >= left and x+w <= img.width - right and y >= up and y+h <= img.height - down:
                    continue

                tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                image_index += 1

        combined_image = images.combine_grid(grid)

        if opts.samples_save:
            images.save_image(combined_image, p.outpath_samples, "", initial_seed, p.prompt, opts.grid_format, info=initial_info)

        processed = Processed(p, [combined_image], initial_seed, initial_info)

        return processed


```

# `scripts/prompt_matrix.py`

这段代码的主要作用是定义了一个名为 `draw_xy_grid` 的函数，用于在二维坐标系中绘制一个二倍角网格(也就是坐标轴)。该函数需要输入四个参数：`xs` 和 `ys` 分别表示网格中行数和列数，`x_label` 和 `y_label` 分别表示网格中每个子网格的标签，`cell` 是一个二元组，表示子网格的起始位置和结束位置(也就是左下角)。

函数实现中，首先定义了一个名为 `res` 的列表，用于存储所有绘制过的图像。接着定义了两个列表 `ver_texts` 和 `hor_texts`，用于存储每个子网格中的网格注释文本。其中 `ver_texts` 和 `hor_texts` 中的每个元素都是一个含有 `(x, y)` 位置的列表，表示该位置对应于网格中的哪个子网格。

然后定义了一个名为 `first_pocessed` 的布尔值，用于指示是否已经捕获了第一个位置。接着定义了一个名为 `state` 的类，其中包含一些与网格绘制相关的变量和函数。在函数中，首先更新了 `state.job_count` 变量，表示该函数将绘制多少个子网格。接着，定义了一个 `for` 循环，遍历 `xs` 和 `ys` 中的所有元素，计算并更新了 `state.job` 变量，用于在函数中记录要绘制的每个子网格的索引号。

接下来，定义了一个 `draw_grid` 函数，该函数接收一个已经绘制的图像，以及一些参数，用于绘制网格。该函数将这个图像作为参数，并将其复制一份作为子图像，然后使用 `images.draw_grid_annotations` 函数在子图像上绘制网格注释。最后，将 `res` 和 `first_pocessed` 中的值返回给调用者。


```py
import math
from collections import namedtuple
from copy import copy
import random

import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import modules.sd_samplers


def draw_xy_grid(xs, ys, x_label, y_label, cell):
    res = []

    ver_texts = [[images.GridAnnotation(y_label(y))] for y in ys]
    hor_texts = [[images.GridAnnotation(x_label(x))] for x in xs]

    first_pocessed = None

    state.job_count = len(xs) * len(ys)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            state.job = f"{ix + iy * len(xs) + 1} out of {len(xs) * len(ys)}"

            processed = cell(x, y)
            if first_pocessed is None:
                first_pocessed = processed

            res.append(processed.images[0])

    grid = images.image_grid(res, rows=len(ys))
    grid = images.draw_grid_annotations(grid, res[0].width, res[0].height, hor_texts, ver_texts)

    first_pocessed.images = [grid]

    return first_pocessed


```



This is a function that creates an image of prompt matrices using the `PromptMatrix` class. The `PromptMatrix` class is a part of the `tensorflow-modelsim`, a library for creating and working with neural network models and their images.

The function takes in prompt data (text data) in the form of a list of prompts, and options for the image (batch size, grid size, and image save options).

The function first splits the prompt data into a list of prompts, then combines them into a matrix of prompts.

The function then creates a batch of images based on the specified batch size and grid size.

Finally, the function displays or saves the image of the prompt matrix.

Note that this function assumes that the `PromptMatrix` class has been defined and initialized before being called. Also, the `image_grid` and `draw_prompt_matrix` methods are from the `images` module of the `tensorflow-modelsim` library, and need to be imported before being used.


```py
class Script(scripts.Script):
    def title(self):
        return "Prompt matrix"

    def ui(self, is_img2img):
        put_at_start = gr.Checkbox(label='Put variable parts at start of prompt', value=False)

        return [put_at_start]

    def run(self, p, put_at_start):
        seed = modules.processing.set_seed(p.seed)

        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

        all_prompts = []
        prompt_matrix_parts = original_prompt.split("|")
        combination_count = 2 ** (len(prompt_matrix_parts) - 1)
        for combination_num in range(combination_count):
            selected_prompts = [text.strip().strip(',') for n, text in enumerate(prompt_matrix_parts[1:]) if combination_num & (1 << n)]

            if put_at_start:
                selected_prompts = selected_prompts + [prompt_matrix_parts[0]]
            else:
                selected_prompts = [prompt_matrix_parts[0]] + selected_prompts

            all_prompts.append(", ".join(selected_prompts))

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        p.do_not_save_grid = True

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

        p.prompt = all_prompts
        p.prompt_for_display = original_prompt
        p.seed = len(all_prompts) * [seed]
        processed = process_images(p)

        grid = images.image_grid(processed.images, p.batch_size, rows=1 << ((len(prompt_matrix_parts) - 1) // 2))
        grid = images.draw_prompt_matrix(grid, p.width, p.height, prompt_matrix_parts)
        processed.images.insert(0, grid)

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "prompt_matrix", prompt=original_prompt, seed=seed)

        return processed

```

# `scripts/xy_grid.py`

这段代码的作用是定义了一个名为 `apply_field` 的函数，用于对一个名为 `field` 的元组进行修改。

首先，从 `collections` 模块中导入 `namedtuple` 类，从 `copy` 模块中导入 `copy` 函数，从 `random` 模块中导入 `random` 函数。

然后，从 `scripts` 模块中导入 `process_images` 函数，从 `gradio` 模块中导入 `gr` 函数，从 `images` 模块中导入 `images` 函数。从 `Processed` 和 `opts` 模块中导入 `opts` 和 `cmd_opts` 函数，从 `shared` 模块中导入 `state` 函数，从 `sd_samplers` 模块中导入 `sd_samplers` 函数。

接着，定义了一个名为 `apply_field` 的函数，它接受一个名为 `field` 的元组，并从 `copy` 函数中克隆一份元组，然后使用 `apply_field` 函数对元组中的每个元素进行修改，最后将修改后的元组返回。

最后，在 `scripts.py` 和 `gradio.py` 中定义了一些函数和类，用于将修改后的元组返回给客户端，并在客户端将修改后的元组保存。


```py
from collections import namedtuple
from copy import copy
import random

import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import modules.sd_samplers
import re


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun


```

这段代码定义了一个名为 `apply_prompt` 的函数，它接受一个 `Prompt` 对象 `p`、一个 `Terminal` 对象 `x` 和一个或多个 `Sequence` 对象 `xs`。这个函数的作用是替换 `x` 和 `xs[0]` 在 `p.prompt` 和 `p.negative_prompt` 属性中的值，使得 `p` 在响应输入时更加鲁棒。

接下来是一个更大的代码块，定义了一个包含多个子任务的 `samplers_dict` 字典，这个字典中包含了所有 SD 采样器的名称和它们相应的索引。这个字典的目的是在 `apply_sampler` 函数中使用 SD 采样器，通过 `samplers_dict` 获取正确的采样器并将其索引存储在 `p.sampler_index` 属性中。

`apply_sampler` 函数接收一个 `Terminal` 对象 `p`、一个 `Sequence` 对象 `xs` 和一个或多个 `Sequence` 对象 `xs`，然后它尝试从 `samplers_dict` 中获取指定的采样器，并将其索引存储在 `p.sampler_index` 属性中。如果 `samplers_dict` 中没有指定的采样器，函数将引发 `RuntimeError`，并错误地捕获到这个错误。


```py
def apply_prompt(p, x, xs):
    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)


samplers_dict = {}
for i, sampler in enumerate(modules.sd_samplers.samplers):
    samplers_dict[sampler.name.lower()] = i
    for alias in sampler.aliases:
        samplers_dict[alias.lower()] = i


def apply_sampler(p, x, xs):
    sampler_index = samplers_dict.get(x.lower(), None)
    if sampler_index is None:
        raise RuntimeError(f"Unknown sampler: {x}")

    p.sampler_index = sampler_index


```

这段代码定义了两个函数，一个是`format_value`，另一个是`format_value_add_label`。这两个函数都是接受一个参数`p`，一个选项`opt`和一个参数`x`。

`format_value`函数的实现是直接将参数`x`保留，不做任何转换。

`format_value_add_label`函数的实现是使用给定的选项`opt`中的`label`和参数`x`，生成一个带有标签的格式字符串，并将格式字符串插入到字符串的末尾。

给定的轴选项是一个名为`AxisOption`的命名元组，它包含标签`label`，类型`int`，应用域`apply_field`和格式化函数`format_value_add_label`。这个轴选项列表是一个包含多个选项的列表，其中每个选项都是在上面的AxisOption中定义的。

给定的`AxisOptionImg2Img`是一个命名元组，它包含标签`label`，类型`float`，应用域`apply_field`和格式化函数`format_value`。这个轴选项列表中有一个选项是`Denoising`，这是一个图像预处理选项，它的格式化函数是上面定义的`format_value_add_label`。

另外，给定的`format_value_add_label`函数还被定义为`f"{opt.label}: {x}"`，这个格式字符串中的`{opt.label}`是上面定义的`AxisOption`中定义的标签，`{x}`是要格式化的参数。这个格式字符串将被插入到给定的字符串的末尾，生成一个新的字符串，其中`{opt.label}`和`{x}`都是带有标签的格式字符串。


```py
def format_value_add_label(p, opt, x):
    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    return x


AxisOption = namedtuple("AxisOption", ["label", "type", "apply", "format_value"])
AxisOptionImg2Img = namedtuple("AxisOptionImg2Img", ["label", "type", "apply", "format_value"])


axis_options = [
    AxisOption("Seed", int, apply_field("seed"), format_value_add_label),
    AxisOption("Steps", int, apply_field("steps"), format_value_add_label),
    AxisOption("CFG Scale", float, apply_field("cfg_scale"), format_value_add_label),
    AxisOption("Prompt S/R", str, apply_prompt, format_value),
    AxisOption("Sampler", str, apply_sampler, format_value),
    AxisOptionImg2Img("Denoising", float, apply_field("denoising_strength"), format_value_add_label) #  as it is now all AxisOptionImg2Img items must go after AxisOption ones
]


```

这段代码定义了一个名为 `draw_xy_grid` 的函数，它接受四个参数：`xs` 和 `ys` 是二维列表，分别表示 x 和 y 坐标，`x_label` 和 `y_label` 是文本标签，用于显示 x 和 y 轴的标签。`cell` 是一个计算单元格宽度和高度的函数，返回一个单位为细胞宽度的列表。

函数的作用是处理 `xs` 和 `ys` 中的每个元素，并将它们转化为 `images.GridAnnotation` 类中的 `grid` 对象。首先，对于每个 `x` 和 `y` 坐标，函数都会执行 `cell` 函数，并将处理结果添加到 `res` 列表中。然后，对于每个 `grid` 对象，函数将其添加到 `grid_res` 列表中。最后，函数将 `res` 列表中的所有元素返回，并将 `first_pocessed` 的图像添加到它上面。


```py
def draw_xy_grid(xs, ys, x_label, y_label, cell):
    res = []

    ver_texts = [[images.GridAnnotation(y_label(y))] for y in ys]
    hor_texts = [[images.GridAnnotation(x_label(x))] for x in xs]

    first_pocessed = None

    state.job_count = len(xs) * len(ys)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            state.job = f"{ix + iy * len(xs) + 1} out of {len(xs) * len(ys)}"

            processed = cell(x, y)
            if first_pocessed is None:
                first_pocessed = processed

            res.append(processed.images[0])

    grid = images.image_grid(res, rows=len(ys))
    grid = images.draw_grid_annotations(grid, res[0].width, res[0].height, hor_texts, ver_texts)

    first_pocessed.images = [grid]

    return first_pocessed


```

这是一个Python module，通过对MXNet模型的进行处理，实现对模型的训练和评估。

The module takes in two arguments, `opts` and `vals`, where `opts` is an object that contains the options for the model, and `vals` is a list of values corresponding to the input data.

The `set_seed` method sets the random seed for the module to ensure reproducibility.

The `batch_size` and `batch_count` methods determine the batch size for the input data and the number of batches in the data, respectively.

The `process_axis` function takes in the options for the axis and the input data, and applies any necessary processing to the data before returning it.

The `process_images` function takes in the processed data and applies the appropriate methods to convert it to an image format.

The `draw_xy_grid` function takes in the processed data and displays a grid of processed data, with the `x` and `y` coordinates labeled according to the `x_label` and `y_label` methods, respectively.

The `save_image` method saves the image to disk, with the filename being generated by the `filename` method and the image being saved in the `outpath_grids` directory specified by the `opts.grid_save` parameter.

The `saved_images` list contains all the images that have been saved to disk.


```py
re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")

class Script(scripts.Script):
    def title(self):
        return "X/Y plot"

    def ui(self, is_img2img):
        current_axis_options = [x for x in axis_options if type(x) == AxisOption or type(x) == AxisOptionImg2Img and is_img2img]

        with gr.Row():
            x_type = gr.Dropdown(label="X type", choices=[x.label for x in current_axis_options], value=current_axis_options[0].label, visible=False, type="index", elem_id="x_type")
            x_values = gr.Textbox(label="X values", visible=False, lines=1)

        with gr.Row():
            y_type = gr.Dropdown(label="Y type", choices=[x.label for x in current_axis_options], value=current_axis_options[1].label, visible=False, type="index", elem_id="y_type")
            y_values = gr.Textbox(label="Y values", visible=False, lines=1)

        return [x_type, x_values, y_type, y_values]

    def run(self, p, x_type, x_values, y_type, y_values):
        p.seed = modules.processing.set_seed(p.seed)
        p.batch_size = 1
        p.batch_count = 1

        def process_axis(opt, vals):
            valslist = [x.strip() for x in vals.split(",")]

            if opt.type == int:
                valslist_ext = []

                for val in valslist:
                    m = re_range.fullmatch(val)
                    if m is not None:

                        start = int(m.group(1))
                        end = int(m.group(2))+1
                        step = int(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += list(range(start, end, step))
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext

            valslist = [opt.type(x) for x in valslist]

            return valslist

        x_opt = axis_options[x_type]
        xs = process_axis(x_opt, x_values)

        y_opt = axis_options[y_type]
        ys = process_axis(y_opt, y_values)

        def cell(x, y):
            pc = copy(p)
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)

            return process_images(pc)

        processed = draw_xy_grid(
            xs=xs,
            ys=ys,
            x_label=lambda x: x_opt.format_value(p, x_opt, x),
            y_label=lambda y: y_opt.format_value(p, y_opt, y),
            cell=cell
        )

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "xy_grid", prompt=p.prompt, seed=processed.seed)

        return processed

```