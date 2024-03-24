# `.\lucidrains\muse-maskgit-pytorch\muse_maskgit_pytorch\muse_maskgit_pytorch.py`

```
        # 定义一个注意力机制模块
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        cross_attend = False,
        scale = 8,
        flash = True,
        dropout = 0.
    ):
        super().__init__()
        # 缩放因子
        self.scale = scale
        # 头数
        self.heads =  heads
        # 内部维度
        inner_dim = dim_head * heads

        # 是否进行跨注意力
        self.cross_attend = cross_attend
        # 归一化层
        self.norm = LayerNorm(dim)

        # 注意力机制
        self.attend = Attend(
            flash = flash,
            dropout = dropout,
            scale = scale
        )

        # 空键值对
        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        # 转换查询
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # 转换键值对
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        # 查询缩放
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        # 键缩放
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        # 输出转换
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
        ):
        # 断言条件：如果存在上下文信息，则不应该使用交叉注意力，反之亦然
        assert not (exists(context) ^ self.cross_attend)

        # 获取输入张量 x 的倒数第二维度的大小
        n = x.shape[-2]
        # 获取头数 h 和是否使用交叉注意力 is_cross_attn
        h, is_cross_attn = self.heads, exists(context)

        # 对输入张量 x 进行归一化处理
        x = self.norm(x)

        # 根据是否使用交叉注意力选择键值对输入
        kv_input = context if self.cross_attend else x

        # 分别计算查询 q、键 k、值 v，并根据最后一维度拆分成三部分
        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        # 将查询 q、键 k、值 v 重排维度，使得头数 h 在第二维度
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 获取空键值对 nk、nv，并根据头数 h 和批次大小重复扩展
        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        # 将键 k 和值 v 连接空键值对 nk、nv
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # 对查询 q、键 k 进行 L2 归一化处理
        q, k = map(l2norm, (q, k))
        # 对查询 q、键 k 进行缩放
        q = q * self.q_scale
        k = k * self.k_scale

        # 如果存在上下文掩码，则重复扩展到匹配注意力矩阵的维度，并进行填充
        if exists(context_mask):
            context_mask = repeat(context_mask, 'b j -> b h i j', h = h, i = n)
            context_mask = F.pad(context_mask, (1, 0), value = True)

        # 进行注意力计算
        out = self.attend(q, k, v, mask = context_mask)

        # 重排输出维度，使得头数 h 在第二维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 返回输出结果
        return self.to_out(out)
# 定义 TransformerBlocks 类，用于堆叠多个 Transformer 模块
class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 输入维度
        depth,  # 堆叠的 Transformer 模块数量
        dim_head = 64,  # 注意力头的维度
        heads = 8,  # 注意力头的数量
        ff_mult = 4,  # FeedForward 层的倍数
        flash = True  # 是否使用 Flash
    ):
        super().__init__()
        self.layers = nn.ModuleList([])  # 初始化空的模块列表

        for _ in range(depth):  # 根据 depth 循环堆叠 Transformer 模块
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, flash = flash),  # 添加注意力模块
                Attention(dim = dim, dim_head = dim_head, heads = heads, cross_attend = True, flash = flash),  # 添加交叉注意力模块
                FeedForward(dim = dim, mult = ff_mult)  # 添加 FeedForward 模块
            ]))

        self.norm = LayerNorm(dim)  # 初始化 LayerNorm 模块

    def forward(self, x, context = None, context_mask = None):  # 前向传播函数
        for attn, cross_attn, ff in self.layers:  # 遍历每个 Transformer 模块
            x = attn(x) + x  # 执行注意力模块并加上残差连接

            x = cross_attn(x, context = context, context_mask = context_mask) + x  # 执行交叉注意力模块并加上残差连接

            x = ff(x) + x  # 执行 FeedForward 模块并加上残差连接

        return self.norm(x)  # 返回 LayerNorm 后的结果

# 定义 Transformer 类，用于处理文本数据
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,  # 标记的数量
        dim,  # 输入维度
        seq_len,  # 序列长度
        dim_out = None,  # 输出维度
        t5_name = DEFAULT_T5_NAME,  # T5 模型名称
        self_cond = False,  # 是否自我条件
        add_mask_id = False,  # 是否添加 mask 标记
        **kwargs
    ):
        super().__init__()
        self.dim = dim  # 初始化输入维度
        self.mask_id = num_tokens if add_mask_id else None  # 初始化 mask 标记

        self.num_tokens = num_tokens  # 初始化标记数量
        self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)  # 初始化标记嵌入层
        self.pos_emb = nn.Embedding(seq_len, dim)  # 初始化位置嵌入层
        self.seq_len = seq_len  # 初始化序列长度

        self.transformer_blocks = TransformerBlocks(dim = dim, **kwargs)  # 初始化 TransformerBlocks 模块
        self.norm = LayerNorm(dim)  # 初始化 LayerNorm 模块

        self.dim_out = default(dim_out, num_tokens)  # 初始化输出维度
        self.to_logits = nn.Linear(dim, self.dim_out, bias = False)  # 初始化线性层

        # 文本条件

        self.encode_text = partial(t5_encode_text, name = t5_name)  # 编码文本

        text_embed_dim = get_encoded_dim(t5_name)  # 获取编码后的文本维度

        self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias = False) if text_embed_dim != dim else nn.Identity()  # 初始化文本嵌入层

        # 可选的自我条件

        self.self_cond = self_cond  # 初始化自我条件
        self.self_cond_to_init_embed = FeedForward(dim)  # 初始化 FeedForward 模块

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3.,  # 条件缩放因子
        return_embed = False,  # 是否返回嵌入
        **kwargs
    ):
        if cond_scale == 1:  # 如果条件缩放因子为1
            return self.forward(*args, return_embed = return_embed, cond_drop_prob = 0., **kwargs)  # 执行前向传播

        logits, embed = self.forward(*args, return_embed = True, cond_drop_prob = 0., **kwargs)  # 执行前向传播并返回嵌入

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)  # 执行前向传播，使用条件丢弃

        scaled_logits = null_logits + (logits - null_logits) * cond_scale  # 计算缩放后的 logits

        if return_embed:  # 如果需要返回嵌入
            return scaled_logits, embed  # 返回缩放后的 logits 和嵌入

        return scaled_logits  # 返回缩放后的 logits

    def forward_with_neg_prompt(
        self,
        text_embed: torch.Tensor,
        neg_text_embed: torch.Tensor,
        cond_scale = 3.,  # 条件缩放因子
        return_embed = False,
        **kwargs
    ):
        neg_logits = self.forward(*args, neg_text_embed = neg_text_embed, cond_drop_prob = 0., **kwargs)  # 执行前向传播，使用负面文本嵌入
        pos_logits, embed = self.forward(*args, return_embed = True, text_embed = text_embed, cond_drop_prob = 0., **kwargs)  # 执行前向传播，使用正面文本嵌入

        logits = neg_logits + (pos_logits - neg_logits) * cond_scale  # 计算缩放后的 logits

        if return_embed:  # 如果需要返回嵌入
            return scaled_logits, embed  # 返回缩放后的 logits 和嵌入

        return scaled_logits  # 返回缩放后的 logits

    def forward(
        self,
        x,
        return_embed = False,
        return_logits = False,
        labels = None,
        ignore_index = 0,
        self_cond_embed = None,
        cond_drop_prob = 0.,
        conditioning_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None
        ):
        # 获取输入张量的设备、维度和长度
        device, b, n = x.device, *x.shape
        # 断言序列长度不超过self.seq_len

        # 准备文本数据

        # 断言texts和text_embeds中只有一个存在
        assert exists(texts) ^ exists(text_embeds)

        # 如果texts存在，则使用self.encode_text方法对texts进行编码得到text_embeds
        if exists(texts):
            text_embeds = self.encode_text(texts)

        # 对text_embeds进行线性变换得到context
        context = self.text_embed_proj(text_embeds)

        # 生成context_mask，用于指示哪些位置有文本数据
        context_mask = (text_embeds != 0).any(dim=-1)

        # 如果cond_drop_prob大于0，则进行条件性的dropout
        if cond_drop_prob > 0.:
            mask = prob_mask_like((b, 1), 1. - cond_drop_prob, device)
            context_mask = context_mask & mask

        # 如果conditioning_token_ids存在，则将其与context拼接起来
        if exists(conditioning_token_ids):
            conditioning_token_ids = rearrange(conditioning_token_ids, 'b ... -> b (...)')
            cond_token_emb = self.token_emb(conditioning_token_ids)
            context = torch.cat((context, cond_token_emb), dim=-2)
            context_mask = F.pad(context_mask, (0, conditioning_token_ids.shape[-1]), value=True)

        # 对输入的token进行嵌入
        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device=device))

        # 如果self.self_cond为True，则对self_cond_embed进行初始化
        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        # 使用transformer_blocks进行编码
        embed = self.transformer_blocks(x, context=context, context_mask=context_mask)

        # 将编码结果转换为logits
        logits = self.to_logits(embed)

        # 如果return_embed为True，则返回logits和embed
        if return_embed:
            return logits, embed

        # 如果labels不存在，则返回logits
        if not exists(labels):
            return logits

        # 根据self.dim_out的值计算损失
        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, '... 1 -> ...'), labels)
        else:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index=ignore_index)

        # 如果return_logits为False，则返回损失
        if not return_logits:
            return loss

        # 返回损失和logits
        return loss, logits
# 定义一个自我批评的包装器类
class SelfCritic(nn.Module):
    # 初始化方法，接受一个网络对象作为参数
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Linear(net.dim, 1)

    # 带有条件缩放的前向传播方法
    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(x, *args, return_embed=True, **kwargs)
        return self.to_pred(embeds)

    # 带有负面提示的前向传播方法
    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(x, *args, return_embed=True, **kwargs)
        return self.to_pred(embeds)

    # 前向传播方法
    def forward(self, x, *args, labels=None, **kwargs):
        _, embeds = self.net(x, *args, return_embed=True, **kwargs)
        logits = self.to_pred(embeds)

        # 如果没有标签，则返回logits
        if not exists(labels):
            return logits

        # 重新排列logits并计算二元交叉熵损失
        logits = rearrange(logits, '... 1 -> ...')
        return F.binary_cross_entropy_with_logits(logits, labels)

# 特殊化的transformers类

# MaskGitTransformer类继承自Transformer类
class MaskGitTransformer(Transformer):
    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 断言'add_mask_id'不在关键字参数中
        assert 'add_mask_id' not in kwargs
        super().__init__(*args, add_mask_id=True, **kwargs)

# TokenCritic类继承自Transformer类
class TokenCritic(Transformer):
    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 断言'dim_out'不在关键字参数中
        assert 'dim_out' not in kwargs
        super().__init__(*args, dim_out=1, **kwargs)

# 无分类器指导函数

# 创建一个均匀分布的张量
def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

# 根据概率创建掩码张量
def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob

# 采样辅助函数

# 计算张量的自然对数
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

# 生成Gumbel噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 从Gumbel分布中采样
def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

# 保留top-k概率的值，其余设为负无穷
def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

# 噪声调度

# 余弦调度函数
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

# 主MaskGit类

# MaskGit类继承自nn.Module类
@beartype
class MaskGit(nn.Module):
    # 初始化方法，接受多个参数和关键字参数
    def __init__(
        self,
        image_size,
        transformer: MaskGitTransformer,
        noise_schedule: Callable = cosine_schedule,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic=False,
        vae: Optional[VQGanVAE] = None,
        cond_vae: Optional[VQGanVAE] = None,
        cond_image_size=None,
        cond_drop_prob=0.5,
        self_cond_prob=0.9,
        no_mask_token_prob=0.,
        critic_loss_weight=1.
        ):
        # 调用父类的构造函数
        super().__init__()
        # 如果存在 VAE 模型，则复制一个用于评估的副本，否则设为 None
        self.vae = vae.copy_for_eval() if exists(vae) else None

        # 如果存在条件 VAE 模型，则将其设为评估模式，否则设为与 VAE 模型相同
        if exists(cond_vae):
            self.cond_vae = cond_vae.eval()
        else:
            self.cond_vae = self.vae

        # 断言条件：如果存在条件 VAE 模型，则条件图像大小必须指定
        assert not (exists(cond_vae) and not exists(cond_image_size)), 'cond_image_size must be specified if conditioning'

        # 初始化图像大小和条件图像大小等属性
        self.image_size = image_size
        self.cond_image_size = cond_image_size
        self.resize_image_for_cond_image = exists(cond_image_size)

        # 设置条件丢弃概率
        self.cond_drop_prob = cond_drop_prob

        # 设置变换器和是否自我条件
        self.transformer = transformer
        self.self_cond = transformer.self_cond
        # 断言条件：VAE 和条件 VAE 的码书大小必须与变换器的标记数相等
        assert self.vae.codebook_size == self.cond_vae.codebook_size == transformer.num_tokens, 'transformer num_tokens must be set to be equal to the vae codebook size'

        # 设置掩码 ID 和噪声计划
        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

        # 断言条件：自我令牌评论和令牌评论不能同时存在
        assert not (self_token_critic and exists(token_critic))
        self.token_critic = token_critic

        # 如果存在自我令牌评论，则将其设置为 SelfCritic 类的实例
        if self_token_critic:
            self.token_critic = SelfCritic(transformer)

        # 设置评论损失权重
        self.critic_loss_weight = critic_loss_weight

        # 设置自我条件概率
        self.self_cond_prob = self_cond_prob

        # 设置不掩码令牌的概率，以保持相同令牌，以便变换器在所有令牌上产生更好的嵌入，如原始 BERT 论文中所做
        # 可能需要用于自我条件
        self.no_mask_token_prob = no_mask_token_prob

    # 保存模型参数到指定路径
    def save(self, path):
        torch.save(self.state_dict(), path)

    # 从指定路径加载模型参数
    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    # 生成方法，用于生成文本
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        texts: List[str],
        negative_texts: Optional[List[str]] = None,
        cond_images: Optional[torch.Tensor] = None,
        fmap_size = None,
        temperature = 1.,
        topk_filter_thres = 0.9,
        can_remask_prev_masked = False,
        force_not_use_token_critic = False,
        timesteps = 18,  # 理想的步数是 18，参考 maskgit 论文
        cond_scale = 3,
        critic_noise_scale = 1
    # 前向传播方法，用于模型推理
    def forward(
        self,
        images_or_ids: torch.Tensor,
        ignore_index = -1,
        cond_images: Optional[torch.Tensor] = None,
        cond_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cond_drop_prob = None,
        train_only_generator = False,
        sample_temperature = None
        ):
            # 如果需要进行标记化

            if images_or_ids.dtype == torch.float:
                assert exists(self.vae), 'vqgan vae must be passed in if training from raw images'
                assert all([height_or_width == self.image_size for height_or_width in images_or_ids.shape[-2:]]), 'the image you passed in is not of the correct dimensions'

                with torch.no_grad():
                    _, ids, _ = self.vae.encode(images_or_ids)
            else:
                assert not self.resize_image_for_cond_image, 'you cannot pass in raw image token ids if you want the framework to autoresize image for conditioning super res transformer'
                ids = images_or_ids

            # 处理指定的条件图像

            if self.resize_image_for_cond_image:
                cond_images_or_ids = F.interpolate(images_or_ids, self.cond_image_size, mode='nearest')

            # 获取一些基本变量

            ids = rearrange(ids, 'b ... -> b (...)')

            batch, seq_len, device, cond_drop_prob = *ids.shape, ids.device, default(cond_drop_prob, self.cond_drop_prob)

            # 如果需要对条件图像进行标记化

            assert not (exists(cond_images) and exists(cond_token_ids)), 'if conditioning on low resolution, cannot pass in both images and token ids'

            if exists(cond_images):
                assert exists(self.cond_vae), 'cond vqgan vae must be passed in'
                assert all([height_or_width == self.cond_image_size for height_or_width in cond_images.shape[-2:]])

                with torch.no_grad():
                    _, cond_token_ids, _ = self.cond_vae.encode(cond_images)

            # 准备掩码

            rand_time = uniform((batch,), device=device)
            rand_mask_probs = self.noise_schedule(rand_time)
            num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

            mask_id = self.mask_id
            batch_randperm = torch.rand((batch, seq_len), device=device).argsort(dim=-1)
            mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

            mask_id = self.transformer.mask_id
            labels = torch.where(mask, ids, ignore_index)

            if self.no_mask_token_prob > 0.:
                no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
                mask &= ~no_mask_mask

            x = torch.where(mask, mask_id, ids)

            # 获取文本嵌入

            if exists(texts):
                text_embeds = self.transformer.encode_text(texts)
                texts = None

            # 自我条件

            self_cond_embed = None

            if self.transformer.self_cond and random() < self.self_cond_prob:
                with torch.no_grad():
                    _, self_cond_embed = self.transformer(
                        x,
                        text_embeds=text_embeds,
                        conditioning_token_ids=cond_token_ids,
                        cond_drop_prob=0.,
                        return_embed=True
                    )

                    self_cond_embed.detach_()

            # 获取损失

            ce_loss, logits = self.transformer(
                x,
                text_embeds=text_embeds,
                self_cond_embed=self_cond_embed,
                conditioning_token_ids=cond_token_ids,
                labels=labels,
                cond_drop_prob=cond_drop_prob,
                ignore_index=ignore_index,
                return_logits=True
            )

            if not exists(self.token_critic) or train_only_generator:
                return ce_loss

            # 令牌评论家损失

            sampled_ids = gumbel_sample(logits, temperature=default(sample_temperature, random()))

            critic_input = torch.where(mask, sampled_ids, x)
            critic_labels = (ids != critic_input).float()

            bce_loss = self.token_critic(
                critic_input,
                text_embeds=text_embeds,
                conditioning_token_ids=cond_token_ids,
                labels=critic_labels,
                cond_drop_prob=cond_drop_prob
            )

            return ce_loss + self.critic_loss_weight * bce_loss
# 定义 Muse 类，继承自 nn.Module
@beartype
class Muse(nn.Module):
    # 初始化方法
    def __init__(
        self,
        base: MaskGit,  # 接收一个 MaskGit 类型的参数作为基础模型
        superres: MaskGit  # 接收一个 MaskGit 类型的参数作为超分辨率模型
    ):
        super().__init__()  # 调用父类的初始化方法
        self.base_maskgit = base.eval()  # 将传入的基础模型设为只读模式并赋值给实例变量

        assert superres.resize_image_for_cond_image  # 断言超分辨率模型具有 resize_image_for_cond_image 属性
        self.superres_maskgit = superres.eval()  # 将传入的超分辨率模型设为只读模式并赋值给实例变量

    # 前向传播方法，使用 torch.no_grad() 装饰器
    @torch.no_grad()
    def forward(
        self,
        texts: List[str],  # 接收一个字符串列表作为输入文本
        cond_scale = 3.,  # 设置默认条件尺度为 3
        temperature = 1.,  # 设置默认温度为 1
        timesteps = 18,  # 设置默认时间步数为 18
        superres_timesteps = None,  # 超分辨率时间步数，默认为 None
        return_lowres = False,  # 是否返回低分辨率图像，默认为 False
        return_pil_images = True  # 是否返回 PIL 图像，默认为 True
    ):
        # 使用基础模型生成低分辨率图像
        lowres_image = self.base_maskgit.generate(
            texts = texts,
            cond_scale = cond_scale,
            temperature = temperature,
            timesteps = timesteps
        )

        # 使用超分辨率模型生成高分辨率图像
        superres_image = self.superres_maskgit.generate(
            texts = texts,
            cond_scale = cond_scale,
            cond_images = lowres_image,
            temperature = temperature,
            timesteps = default(superres_timesteps, timesteps)  # 使用默认的超分辨率时间步数
        )
        
        # 如果需要返回 PIL 图像
        if return_pil_images:
            # 将低分辨率图像转换为 PIL 图像列表
            lowres_image = list(map(T.ToPILImage(), lowres_image))
            # 将高分辨率图像转换为 PIL 图像列表
            superres_image = list(map(T.ToPILImage(), superres_image))            

        # 如果不需要返回低分辨率图像，则返回高分辨率图像
        if not return_lowres:
            return superres_image

        # ��回高分辨率图像和低分辨率图像
        return superres_image, lowres_image
```