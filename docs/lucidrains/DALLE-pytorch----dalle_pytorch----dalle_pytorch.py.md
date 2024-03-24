# `.\lucidrains\DALLE-pytorch\dalle_pytorch\dalle_pytorch.py`

```
# 从 math 模块中导入 log2 和 sqrt 函数
from math import log2, sqrt
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 和 einsum 模块
from torch import nn, einsum
# 从 torch.nn.functional 模块中导入 F
import torch.nn.functional as F
# 导入 numpy 库
import numpy as np

# 导入自定义模块
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

# 从 dalle_pytorch 库中导入 distributed_utils 模块
from dalle_pytorch import distributed_utils
# 从 dalle_pytorch.vae 模块中导入 OpenAIDiscreteVAE 和 VQGanVAE 类
from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE
# 从 dalle_pytorch.transformer 模块中导入 Transformer 和 DivideMax 类

# helpers

# 定义函数，判断变量是否存在
def exists(val):
    return val is not None

# 定义函数，返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义类，始终返回指定值
class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return self.val

# 判断张量是否为空
def is_empty(t):
    return t.nelement() == 0

# 计算带掩码的平均值
def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

# 生成与给定形状相同的概率掩码
def prob_mask_like(shape, prob, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# 设置模型参数是否需要梯度
def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

# 评估装饰器
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

# 计算对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps)

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# Gumbel 采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

# Top-k 采样
def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 共享嵌入层
class SharedEmbedding(nn.Embedding):
    def __init__(self, linear, start_index, end_index, **kwargs):
        super().__init__(end_index - start_index, linear.weight.shape[1], **kwargs)
        del self.weight

        self.linear = linear
        self.start_index = start_index
        self.end_index = end_index

    def forward(self, input):
        return F.embedding(
            input, self.linear.weight[self.start_index:self.end_index], self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

# 离散 VAE 类

# ResNet 块
class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

# 离散 VAE 类
class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        reinmax = False,
        kl_div_loss_weight = 0.,
        normalization = ((*((0.5,) * 3), 0), (*((0.5,) * 3), 1))
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言图片大小必须是2的幂次方
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        # 断言层数必须大于等于1
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        # 判断是否有残差块
        has_resblocks = num_resnet_blocks > 0

        # 初始化各种参数
        self.channels = channels
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.reinmax = reinmax

        # 创建编码簿
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        # 初始化编码器和解码器通道数
        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        # 创建编码器和解码器的层
        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        # 添加残差块
        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        # 创建编码器和解码器
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        # 设置损失函数和 KL 散度损失权重
        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # 处理类内的归一化
        self.normalization = tuple(map(lambda t: t[:channels], normalization))

        # 注册外部参数
        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        temp = None
        ):
        # 从输入参数中获取图像、标记数量、图像大小和 KL 散度损失权重
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        # 断言输入图像的形状符合要求
        assert img.shape[-1] == image_size and img.shape[-2] == image_size, f'input must have the correct image size {image_size}'

        # 对输入图像进行归一化处理
        img = self.norm(img)

        # 将归一化后的图像输入编码器获取 logits
        logits = self.encoder(img)

        # 如果需要返回 logits，则直接返回，用于 DALL-E 训练中获取硬图像索引
        if return_logits:
            return logits

        # 获取温度参数，默认为 self.temperature
        temp = default(temp, self.temperature)

        # 使用 Gumbel Softmax 采样生成 one-hot 编码
        one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=self.straight_through)

        # 如果使用 straight-through 和 reinmax
        if self.straight_through and self.reinmax:
            # 使用 reinmax 提高二阶精度 - https://arxiv.org/abs/2304.08612
            # 算法 2
            one_hot = one_hot.detach()
            π0 = logits.softmax(dim=1)
            π1 = (one_hot + (logits / temp).softmax(dim=1)) / 2
            π1 = ((log(π1) - logits).detach() + logits).softmax(dim=1)
            π2 = 2 * π1 - 0.5 * π0
            one_hot = π2 - π2.detach() + one_hot

        # 使用 one-hot 编码和 codebook 权重进行采样
        sampled = einsum('b n h w, n d -> b d h w', one_hot, self.codebook.weight)
        # 将采样结果输入解码器获取输出
        out = self.decoder(sampled)

        # 如果不需要返回损失，则直接返回输出
        if not return_loss:
            return out

        # 重构损失
        recon_loss = self.loss_fn(img, out)

        # KL 散度
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device=device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)

        # 计算总损失
        loss = recon_loss + (kl_div * kl_div_loss_weight)

        # 如果不需要返回重构图像，则直接返回总损失
        if not return_recons:
            return loss

        # 返回总损失和输出图像
        return loss, out
# 主要的 CLIP 类
class CLIP(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim_text = 512,  # 文本维度
        dim_image = 512,  # 图像维度
        dim_latent = 512,  # 潜在维度
        num_text_tokens = 10000,  # 文本标记数量
        text_enc_depth = 6,  # 文本编码器深度
        text_seq_len = 256,  # 文本序列长度
        text_heads = 8,  # 文本注意力头数
        num_visual_tokens = 512,  # 视觉标记数量
        visual_enc_depth = 6,  # 视觉编码器深度
        visual_heads = 8,  # 视觉注意力头数
        visual_image_size = 256,  # 视觉图像大小
        visual_patch_size = 32,  # 视觉图像块大小
        channels = 3  # 通道数
    ):
        super().__init__()
        # 创建文本嵌入层
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        # 创建文本位置嵌入层
        self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
        # 创建文本变换器
        self.text_transformer = Transformer(causal = False, seq_len = text_seq_len, dim = dim_text, depth = text_enc_depth, heads = text_heads, rotary_emb = False)
        # 创建文本到潜在空间的线性层
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        # 确保图像尺寸能够被图像块大小整除
        assert visual_image_size % visual_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (visual_image_size // visual_patch_size) ** 2
        patch_dim = channels * visual_patch_size ** 2

        self.visual_patch_size = visual_patch_size
        # 创建图像块到嵌入空间的线性层
        self.to_visual_embedding = nn.Linear(patch_dim, dim_image)
        # 创建图像位置嵌入层
        self.visual_pos_emb = nn.Embedding(num_patches, dim_image)
        # 创建视觉变换器
        self.visual_transformer = Transformer(causal = False, seq_len = num_patches, dim = dim_image, depth = visual_enc_depth, heads = visual_heads, rotary_emb = False)
        # 创建图像到潜在空间的线性层
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(1.))

    # 前向传播函数
    def forward(
        self,
        text,
        image,
        text_mask = None,
        return_loss = False
    ):
        b, device, p = text.shape[0], text.device, self.visual_patch_size

        # 文本嵌入
        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        # 图像块提取
        image_patches = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        image_emb = self.to_visual_embedding(image_patches)
        image_emb += self.visual_pos_emb(torch.arange(image_emb.shape[1], device = device))

        # 文本编码
        enc_text = self.text_transformer(text_emb, mask = text_mask)
        # 图像编码
        enc_image = self.visual_transformer(image_emb)

        # 计算文本潜在空间表示
        if exists(text_mask):
            text_latents = masked_mean(enc_text, text_mask, dim = 1)
        else:
            text_latents = enc_text.mean(dim = 1)

        # 计算图像潜在空间表示
        image_latents = enc_image.mean(dim = 1)

        # 线性变换
        text_latents = self.to_text_latent(text_latents)
        image_latents = self.to_visual_latent(image_latents)

        # 归一化
        text_latents, image_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (text_latents, image_latents))

        temp = self.temperature.exp()

        # 如果不需要计算损失，则返回相似度
        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, image_latents) * temp
            return sim

        # 计算损失
        sim = einsum('i d, j d -> i j', text_latents, image_latents) * temp
        labels = torch.arange(b, device = device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss

# 主要的 DALL-E 类
class DALLE(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,
        vae,
        num_text_tokens = 10000,
        text_seq_len = 256,
        depth,
        heads = 8,
        dim_head = 64,
        reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0,
        sparse_attn = False,
        attn_types = None,
        loss_img_weight = 7,
        stable = False,
        sandwich_norm = False,
        shift_tokens = True,
        rotary_emb = True,
        shared_attn_ids = None,
        shared_ff_ids = None,
        share_input_output_emb = False,
        optimize_for_inference = False,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言确保 vae 是 DiscreteVAE、OpenAIDiscreteVAE 或 VQGanVAE 的实例
        assert isinstance(vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE)), 'vae must be an instance of DiscreteVAE'

        # 获取图像大小、图像标记数量、图像特征图大小和图像序列长度
        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2

        # 为每个位置（文本序列长度）保留唯一的填充标记
        num_text_tokens = num_text_tokens + text_seq_len
        # 创建文本位置嵌入和图像位置嵌入
        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) if not rotary_emb else always(0) # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_fmap_size, image_fmap_size)) if not rotary_emb else always(0)

        # 设置文本标记数量和图像标记数量
        self.num_text_tokens = num_text_tokens
        self.num_image_tokens = num_image_tokens

        # 设置文本序列长度和图像序列长度
        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        # 计算总序列长度和总标记数量
        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        # 冻结 VAE 不参与训练
        self.vae = vae
        set_requires_grad(self.vae, False)

        # 创建 Transformer 模型
        self.transformer = Transformer(
            dim = dim,
            causal = True,
            seq_len = seq_len,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            reversible = reversible,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_fmap_size = image_fmap_size,
            sparse_attn = sparse_attn,
            stable = stable,
            sandwich_norm = sandwich_norm,
            shift_tokens = shift_tokens,
            rotary_emb = rotary_emb,
            shared_attn_ids = shared_attn_ids,
            shared_ff_ids = shared_ff_ids,
            optimize_for_inference = optimize_for_inference,
        )

        # 设置稳定性参数
        self.stable = stable

        # 如果稳定性为真，使用 DivideMax 进行归一化
        if stable:
            self.norm_by_max = DivideMax(dim = -1)

        # 转换为 logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        # 如果共享输入输出嵌入，创建共享嵌入层，否则创建独立嵌入层
        if share_input_output_emb:
            self.text_emb = SharedEmbedding(self.to_logits[1], 0, num_text_tokens)
            self.image_emb = SharedEmbedding(self.to_logits[1], num_text_tokens, total_tokens)
        else:
            self.text_emb = nn.Embedding(num_text_tokens, dim)
            self.image_emb = nn.Embedding(num_image_tokens, dim)

        # 创建序列范围和 logits 范围
        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')

        # 创建 logits 掩码
        logits_mask = (
            ((seq_range >= text_seq_len) & (logits_range < num_text_tokens)) |
            ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
        )

        # 注册 logits 掩码为缓冲区
        self.register_buffer('logits_mask', logits_mask, persistent=False)
        self.loss_img_weight = loss_img_weight


    @torch.no_grad()
    @eval_decorator
    def generate_texts(
        self,
        tokenizer,
        text = None,
        *,
        filter_thres = 0.5,
        temperature = 1.
        ):
        # 获取文本序列长度
        text_seq_len = self.text_seq_len
        # 如果文本为空或者为None，则将文本tokens设置为0，并移至GPU
        if text is None or text == "":
            text_tokens = torch.tensor([[0]]).cuda()
        else:
            # 将文本编码为tokens，并移至GPU
            text_tokens = torch.tensor(tokenizer.tokenizer.encode(text)).cuda().unsqueeze(0)

        # 循环直到文本tokens长度达到指定长度
        for _ in range(text_tokens.shape[1], text_seq_len):
            # 获取当前设备
            device = text_tokens.device

            # 获取文本tokens的嵌入
            tokens = self.text_emb(text_tokens)
            # 添加文本位置嵌入
            tokens += self.text_pos_emb(torch.arange(text_tokens.shape[1], device=device))

            # 获取tokens序列长度
            seq_len = tokens.shape[1]

            # 使用transformer处理tokens
            output_transf = self.transformer(tokens)

            # 如果启用了稳定性，对输出进行归一化
            if self.stable:
                output_transf = self.norm_by_max(output_transf)

            # 获取logits
            logits = self.to_logits(output_transf)

            # 对logits进行掩码，确保文本预测文本（除了最后一个token），图像预测图像
            logits_mask = self.logits_mask[:, :seq_len]
            max_neg_value = -torch.finfo(logits.dtype).max
            logits.masked_fill_(logits_mask, max_neg_value)
            logits = logits[:, -1, :]

            # 从logits中筛选出top k的token
            filtered_logits = top_k(logits, thres=filter_thres)
            # 使用Gumbel采样获取样本
            sample = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            # 将新样本添加到文本tokens中
            text_tokens = torch.cat((text_tokens, sample[:, None]), dim=-1)

        # 创建填充tokens集合
        padding_tokens = set(np.arange(self.text_seq_len) + (self.num_text_tokens - self.text_seq_len))
        # 解码文本tokens，获取文本列表
        texts = [tokenizer.tokenizer.decode(text_token, pad_tokens=padding_tokens) for text_token in text_tokens]
        return text_tokens, texts

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        *,
        clip=None,
        filter_thres=0.5,
        temperature=1.,
        img=None,
        num_init_img_tokens=None,
        cond_scale=1.,
        use_cache=False,
    ):
        # 获取VAE模型、文��序列长度、图像序列长度、文本tokens数量
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        # 计算总长度
        total_len = text_seq_len + image_seq_len

        # 确保文本在指定范围内
        text = text[:, :text_seq_len]
        out = text

        # 如果存在图像输入
        if exists(img):
            # 获取图像大小
            image_size = vae.image_size
            assert img.shape[1] == 3 and img.shape[2] == image_size and img.shape[3] == image_size, f'input image must have the correct image size {image_size}'

            # 获取图像的codebook索引
            indices = vae.get_codebook_indices(img)
            # 设置初始图像tokens数量
            num_img_tokens = default(num_init_img_tokens, int(0.4375 * image_seq_len))  # OpenAI used 14 * 32 initial tokens to prime
            assert num_img_tokens < image_seq_len, 'number of initial image tokens for priming must be less than the total image token sequence length'

            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim=-1)

        prev_cache = None
        cache = {} if use_cache else None
        # 循环直到out的长度达到总长度
        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            # 使用条件缩放处理文本和图像
            logits = self.forward_with_cond_scale(text, image, cond_scale=cond_scale, cache=cache)
            logits = logits[:, -1, :]

            # 从logits中筛选出top k的token
            filtered_logits = top_k(logits, thres=filter_thres)
            # 使用Gumbel采样获取样本
            sample = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            # 如果是图像token，减去num_text_tokens的偏移量
            sample -= (num_text_tokens if is_image else 0)
            out = torch.cat((out, sample[:, None]), dim=-1)

        # 获取文本序列和图像序列
        text_seq = out[:, :text_seq_len]
        img_seq = out[:, -image_seq_len:]
        # 解码图像序列
        images = vae.decode(img_seq)

        # 如果存在clip模型
        if exists(clip):
            # 使用clip模型评分
            scores = clip(text_seq, images, return_loss=False)
            return images, scores

        return images
    # 定义一个带有条件缩放参数的前向传播函数
    def forward_with_cond_scale(self, *args, cond_scale = 1, cache = None, **kwargs):
        # 如果条件缩放参数为1，则直接调用原始的前向传播函数
        if cond_scale == 1:
            return self(*args, **kwargs)

        # 如果缓存存在，则复制缓存，否则设为None
        prev_cache = cache.copy() if exists(cache) else None
        # 调用原始的前向传播函数，传入缓存参数
        logits = self(*args, cache = cache, **kwargs)

        # Katherine Crowson的发现
        # https://twitter.com/RiversHaveWings/status/1478093658716966912
        # 传入空条件概率为1的参数，调用原始的前向传播函数
        null_cond_logits = self(*args, null_cond_prob = 1., cache = prev_cache, **kwargs)
        # 返回空条件logits加上（原始logits减去空条件logits）乘以条件缩放参数的结果
        return null_cond_logits + (logits - null_cond_logits) * cond_scale

    # 定义一个前向传播函数，接受文本、图像、是否返回损失、空条件概率和缓存等参数
    def forward(
        self,
        text,
        image = None,
        return_loss = False,
        null_cond_prob = 0.,
        cache = None,
    ):
        # 检查传入的文本张量是否与指定的文本序列长度相匹配
        assert text.shape[-1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
        # 获取文本张量的批次大小、设备信息和总序列长度
        batch, device, total_seq_len = text.shape[0], text.device, self.total_seq_len

        # 以 <null_cond_prob> 的概率随机移除文本条件

        if null_cond_prob > 0:
            # 创建一个与文本张量形状相同的概率掩码，用于随机移除文本条件
            null_mask = prob_mask_like((batch,), null_cond_prob, device=device)
            # 将文本张量中的部分内容根据概率掩码置零
            text *= rearrange(~null_mask, 'b -> b 1')

        # 确保文本标记中的填充获得唯一的填充标记ID

        # 生成文本范围，用于替换文本张量中的填充标记
        text_range = torch.arange(self.text_seq_len, device=device) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        # 添加 <bos> 标记

        # 在文本张量的开头添加一个零值填充
        text = F.pad(text, (1, 0), value=0)

        # 对文本进行嵌入处理
        tokens = self.text_emb(text)
        # 添加文本位置编码
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device=device))

        seq_len = tokens.shape[1]

        # 如果存在图像且图像不为空
        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4

            if is_raw_image:
                # 获取图像的代码簿索引
                image_size = self.vae.image_size
                channels = self.vae.channels
                assert tuple(image.shape[1:]) == (channels, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'

                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            image_emb = self.image_emb(image)

            # 添加图像位置编码
            image_emb += self.image_pos_emb(image_emb)

            # 将文本和图像嵌入连接起来
            tokens = torch.cat((tokens, image_emb), dim=1)

            seq_len += image_len

        # 在训练时，如果长度超过总文本+图像长度，则移除最后一个标记，因为不需要对其进行训练

        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        # ���果启用稳定性训练
        if self.stable:
            alpha = 0.1
            # 对 tokens 进行稳定性训练
            tokens = tokens * alpha + tokens.detach() * (1 - alpha)

        # 如果存在缓存且缓存中有 'offset' 键
        if exists(cache) and cache.get('offset'):
            # 仅保留 tokens 的最后一个标记
            tokens = tokens[:, -1:]
        # 使用 transformer 进行处理，传入缓存信息
        out = self.transformer(tokens, cache=cache)

        # 如果启用稳定性训练
        if self.stable:
            # 对输出进行最大归一化
            out = self.norm_by_max(out)

        # 将输出转换为 logits
        logits = self.to_logits(out)

        # 对 logits 进行掩码处理，确保文本预测文本（除最后一个标记），图像预测图像

        logits_mask = self.logits_mask[:, :seq_len]
        if exists(cache) and cache.get('offset'):
            logits_mask = logits_mask[:, -1:]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)

        # 如果存在缓存
        if exists(cache):
            # 更新缓存中的 'offset' 键
            cache['offset'] = cache.get('offset', 0) + logits.shape[1]

        # 如果不需要返回损失值
        if not return_loss:
            return logits

        # 断言在训练时必须提供图像
        assert exists(image), 'when training, image must be supplied'

        # 对图像进行偏移处理
        offsetted_image = image + self.num_text_tokens
        # 创建标签，用于计算损失
        labels = torch.cat((text[:, 1:], offsetted_image), dim=1)

        # 重新排列 logits 的维度
        logits = rearrange(logits, 'b n c -> b c n')

        # 计算文本损失和图像损失
        loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len], labels[:, :self.text_seq_len])
        loss_img = F.cross_entropy(logits[:, :, self.text_seq_len:], labels[:, self.text_seq_len:])

        # 计算总损失
        loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
        return loss
```