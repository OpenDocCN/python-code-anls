# `.\lucidrains\DALLE-pytorch\dalle_pytorch\attention.py`

```
# 从 inspect 模块中导入 isfunction 函数
# 从 math 模块中导入 ceil 函数
# 导入 torch 库
# 从 torch 模块中导入 nn、einsum
# 从 torch.nn 模块中导入 functional 模块，并重命名为 F
# 从 einops 库中导入 rearrange、repeat 函数
# 导入 rotary_embedding_torch 库中的 apply_rotary_emb 函数

def exists(val):
    # 判断值是否存在
    return val is not None

def uniq(arr):
    # 返回数组中唯一的元素
    return{el: True for el in arr}.keys()

def default(val, d):
    # 如果值存在，则返回该值；否则返回默认值
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    # 返回给定张量的最大负值
    return -torch.finfo(t.dtype).max

def stable_softmax(t, dim = -1, alpha = 32 ** 2):
    # 计算稳定的 softmax 函数
    t = t / alpha
    t = t - torch.amax(t, dim = dim, keepdim = True).detach()
    return (t * alpha).softmax(dim = dim)

def apply_pos_emb(pos_emb, qkv):
    # 应用位置编码到查询、键、值张量中
    n = qkv[0].shape[-2]
    pos_emb = pos_emb[..., :n, :]
    return tuple(map(lambda t: apply_rotary_emb(pos_emb, t), qkv))

# 定义 Attention 类
class Attention(nn.Module):
    def __init__(self, dim, seq_len, causal = True, heads = 8, dim_head = 64, dropout = 0., stable = False,
                 static_mask = None):
        # 初始化 Attention 类
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.stable = stable
        self.causal = causal
        self.register_buffer('static_mask', static_mask, persistent=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, rotary_pos_emb = None, cache = None, cache_key = None):
        # 前向传播函数
        b, n, _, h, device = *x.shape, self.heads, x.device
        softmax = torch.softmax if not self.stable else stable_softmax
        offset = cache.get('offset', 0) if exists(cache) else 0

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb[..., offset:, :], (q, k, v))

        q = q * self.scale

        if offset > 0:
            k_top, v_top = cache[cache_key]
            k = torch.cat([k_top, k], dim=-2)
            v = torch.cat([v_top, v], dim=-2)
        if exists(cache):
            cache[cache_key] = k, v

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal and offset == 0:  # causality is naturally enforced for the cached inference
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        if exists(self.static_mask):
            dots.masked_fill_(~self.static_mask[offset:offset + n, :offset + n], mask_value)

        attn = softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

# 定义 SparseConvCausalAttention 类，实现稀疏注意力机制
class SparseConvCausalAttention(nn.Module):
    # 初始化函数，设置模型参数和超参数
    def __init__(self, dim, seq_len, image_size = 32, kernel_size = 5, dilation = 1, heads = 8, dim_head = 64, dropout = 0., stable = False, **kwargs):
        # 调用父类的初始化函数
        super().__init__()
        # 断言核大小必须为奇数
        assert kernel_size % 2 == 1, 'kernel size must be odd'

        # 计算内部维度
        inner_dim = dim_head *  heads
        # 设置序列长度
        self.seq_len = seq_len
        # 设置头数
        self.heads = heads
        # 设置缩放因子
        self.scale = dim_head ** -0.5
        # 设置图像大小
        self.image_size = image_size
        # 设置核大小
        self.kernel_size = kernel_size
        # 设置膨胀率
        self.dilation = dilation

        # 设置是否稳定
        self.stable = stable

        # 创建线性层，用于将输入转换为查询、键和值
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # 创建输出层，包含线性层和dropout层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    # 定义前向传播函数，接受输入 x，mask 和旋转位置嵌入 rotary_pos_emb
    def forward(self, x, mask = None, rotary_pos_emb = None):
        # 解包 x 的形状信息，包括 batch 大小 b，序列长度 n，头数 h，图像大小 img_size，卷积核大小 kernel_size，膨胀率 dilation，序列长度 seq_len，设备信息 device
        b, n, _, h, img_size, kernel_size, dilation, seq_len, device = *x.shape, self.heads, self.image_size, self.kernel_size, self.dilation, self.seq_len, x.device
        # 根据是否稳定计算 softmax 函数
        softmax = torch.softmax if not self.stable else stable_softmax

        # 计算图像序列长度
        img_seq_len = img_size ** 2
        # 计算文本长度
        text_len = seq_len + 1 - img_seq_len

        # 填充

        # 计算填充长度
        padding = seq_len - n + 1
        # 如果 mask 为 None，则创建全为 True 的 mask 张量
        mask = default(mask, lambda: torch.ones(b, text_len, device = device).bool())

        # 对输入 x 进行填充
        x = F.pad(x, (0, 0, 0, padding), value = 0)
        # 裁剪 mask 的长度
        mask = mask[:, :text_len]

        # 求解查询 / 键 / 值

        # 将输入 x 转换为查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 重排查询、键、值的维度
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        # 如果存在旋转位置嵌入，则应用到查询、键、值上
        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))

        # 缩放查询
        q *= self.scale

        # 分离文本查询、图像查询、文本键、图像键、文本值、图像值
        ((q_text, q_img), (k_text, k_img), (v_text, v_img)) = map(lambda t: (t[:, :-img_seq_len], t[:, -img_seq_len:]), (q, k, v))

        # 文本注意力

        # 计算点积注意力得分
        dots_text = einsum('b i d, b j d -> b i j', q_text, k_text)
        # 计算 mask 的值
        mask_value = max_neg_value(dots_text)

        i, j = dots_text.shape[-2:]
        # 创建文本因果 mask
        text_causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
        dots_text.masked_fill_(text_causal_mask, mask_value)

        # 计算文本注意力权重
        attn_text = softmax(dots_text, dim = -1)
        out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)

        # 图像注意力

        # 计算有效卷积核大小
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        same_padding = effective_kernel_size // 2
        causal_padding = (same_padding * 2, 0, same_padding * 2, 0)

        # 重排图像键、值的维度
        k_img, v_img = map(lambda t: rearrange(t, 'b (h w) c -> b c h w', h = img_size), (k_img, v_img))
        # 对图像键、值进行填充
        k_img, v_img = map(lambda t: F.pad(t, causal_padding), (k_img, v_img))
        k_img, v_img = map(lambda t: F.unfold(t, kernel_size, dilation = dilation), (k_img, v_img))
        k_img, v_img = map(lambda t: rearrange(t, 'b (d j) i -> b i j d', j = kernel_size ** 2), (k_img, v_img))

        # 让图像关注所有文本

        dots_image = einsum('b i d, b i j d -> b i j', q_img, k_img)
        dots_image_to_text = einsum('b i d, b j d -> b i j', q_img, k_text)

        # 使用填充 mask 对张量进行填充和展开
        i, j = dots_image.shape[-2:]
        ones = torch.ones((img_seq_len,), device = device)
        ones = rearrange(ones, '(h w) -> () () h w', h = img_size)
        ones = F.pad(ones, causal_padding, value = 0.)
        ones = F.unfold(ones, kernel_size, dilation = dilation)
        ones = rearrange(ones, 'b j i -> b i j')

        # 对图像注意力进行 mask
        padding_mask = ones == 0.

        # 将文本 mask 与图像因果 mask 连接起来
        padding_mask = repeat(padding_mask, '() i j -> b i j', b = b * h)
        mask = repeat(mask, 'b j -> (b h) i j', i = i, h = h)
        mask = torch.cat((~mask, padding_mask), dim = -1)

        # 图像可以关注所有文本

        dots = torch.cat((dots_image_to_text, dots_image), dim = -1)
        dots.masked_fill_(mask, mask_value)

        attn = softmax(dots, dim = -1)

        # 聚合

        attn_image_to_text, attn_image = attn[..., :text_len], attn[..., text_len:]

        out_image_to_image = einsum('b i j, b i j d -> b i d', attn_image, v_img)
        out_image_to_text = einsum('b i j, b j d -> b i d', attn_image_to_text, v_text)

        out_image = out_image_to_image + out_image_to_text

        # 合并文本和图像的注意力值

        out = torch.cat((out_text, out_image), dim = 1)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out =  self.to_out(out)
        return out[:, :n]
# 稀疏轴向因果注意力机制

class SparseAxialCausalAttention(nn.Module):
    # 初始化函数，定义稀疏轴向因果注意力机制的参数
    def __init__(self, dim, seq_len, image_size = 32, axis = 0, heads = 8, dim_head = 64, dropout = 0., stable = False, **kwargs):
        super().__init__()
        # 断言轴向参数只能是0（沿高度）或1（沿宽度）
        assert axis in {0, 1}, 'axis must be either 0 (along height) or 1 (along width)'
        self.axis = axis

        # 计算内部维度
        inner_dim = dim_head *  heads
        self.seq_len = seq_len
        self.heads = heads
        # 缩放因子
        self.scale = dim_head ** -0.5
        self.image_size = image_size

        # 是否稳定
        self.stable = stable

        # 线性变换，将输入维度映射到内部维度的3倍（用于查询、键、值）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # 输出层，包含线性变换和dropout
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    # 定义前向传播函数，接受输入 x，mask 和旋转位置嵌入 rotary_pos_emb
    def forward(self, x, mask = None, rotary_pos_emb = None):
        # 解包 x 的形状信息，包括 batch 大小 b，序列长度 n，头数 h，图像大小 img_size，轴 axis，序列长度 seq_len，设备 device
        b, n, _, h, img_size, axis, seq_len, device = *x.shape, self.heads, self.image_size, self.axis, self.seq_len, x.device
        # 根据是否稳定计算 softmax 函数
        softmax = torch.softmax if not self.stable else stable_softmax

        # 计算图像序列长度和文本序列长度
        img_seq_len = img_size ** 2
        text_len = seq_len + 1 - img_seq_len

        # 填充

        # 计算需要填充的长度
        padding = seq_len - n + 1
        # 如果 mask 为 None，则创建全为 True 的 mask 张量
        mask = default(mask, lambda: torch.ones(b, text_len, device = device).bool())

        # 对输入 x 进行填充
        x = F.pad(x, (0, 0, 0, padding), value = 0)
        mask = mask[:, :text_len]

        # 求解查询 / 键 / 值

        # 将输入 x 转换为查询、键、值，并按维度 -1 切分
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        # 如果存在旋转位置嵌入，则应用到查询、键、值上
        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))

        # 缩放查询
        q *= self.scale

        # 拆分文本查询、图像查询、文本键、图像键、文本值、图像值
        ((q_text, q_img), (k_text, k_img), (v_text, v_img)) = map(lambda t: (t[:, :-img_seq_len], t[:, -img_seq_len:]), (q, k, v))

        # 文本注意力

        # 计算文本查询和文本键的点积
        dots_text = einsum('b i d, b j d -> b i j', q_text, k_text)
        mask_value = max_neg_value(dots_text)

        i, j = dots_text.shape[-2:]
        # 创建文本因果 mask
        text_causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
        dots_text.masked_fill_(text_causal_mask, mask_value)

        # 计算文本注意力权重
        attn_text = softmax(dots_text, dim = -1)
        out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)

        # 图像注意力

        # 根据轴 axis 拆分图像查询、图像键、图像值
        split_axis_einops = 'b (h w) c -> b h w c' if axis == 0 else 'b (h w) c -> b w h c'
        merge_axis_einops = 'b x n d -> b (x n) d' if axis == 0 else 'b x n d -> b (n x) d'

        # 拆分轴

        q_img, k_img, v_img = map(lambda t: rearrange(t, split_axis_einops, h = img_size), (q_img, k_img, v_img))

        # 相似度

        dots_image_to_image = einsum('b x i d, b x j d -> b x i j', q_img, k_img)
        dots_image_to_text = einsum('b x i d, b j d -> b x i j', q_img, k_text)

        dots = torch.cat((dots_image_to_text, dots_image_to_image), dim = -1)

        # mask 以使图像对文本有完全注意力，但沿轴是因果的

        bh, x, i, j = dots.shape
        causal_mask = torch.ones(i, img_size, device = device).triu_(img_size - i + 1).bool()
        causal_mask = repeat(causal_mask, 'i j -> b x i j', b = bh, x = x)

        mask = repeat(mask, 'b j -> (b h) x i j', h = h, x = x, i = i)
        mask = torch.cat((~mask, causal_mask), dim = -1)

        dots.masked_fill_(mask, mask_value)

        # 注意力

        attn = softmax(dots, dim = -1)

        # 聚合

        attn_image_to_text, attn_image_to_image = attn[..., :text_len], attn[..., text_len:]

        out_image_to_image = einsum('b x i j, b x j d -> b x i d', attn_image_to_image, v_img)
        out_image_to_text = einsum('b x i j, b j d -> b x i d', attn_image_to_text, v_text)

        out_image = out_image_to_image + out_image_to_text

        # 合并轴

        out_image = rearrange(out_image, merge_axis_einops, x = img_size)

        # 合并文本和图像的注意力值

        out = torch.cat((out_text, out_image), dim = 1)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out =  self.to_out(out)
        return out[:, :n]
# 定义 SparseAttention 类，继承自 Attention 类
class SparseAttention(Attention):
    # 初始化函数
    def __init__(
        self,
        *args,
        block_size = 16,  # 定义块大小，默认为16
        text_seq_len = 256,  # 定义文本序列长度，默认为256
        num_random_blocks = None,  # 定义随机块数，默认为None
        **kwargs
    ):
        super().__init__(*args, **kwargs)  # 调用父类的初始化函数
        from deepspeed.ops.sparse_attention import SparseSelfAttention, VariableSparsityConfig  # 导入相关模块
        self.block_size = block_size  # 设置块大小

        num_random_blocks = default(num_random_blocks, self.seq_len // block_size // 4)  # 计算随机块数
        global_block_indices = list(range(ceil(text_seq_len / block_size)))  # 计算全局块索引

        # 初始化稀疏自注意力机制
        self.attn_fn = SparseSelfAttention(
            sparsity_config = VariableSparsityConfig(
                num_heads = self.heads,
                block = self.block_size,
                num_random_blocks = num_random_blocks,
                global_block_indices = global_block_indices,
                attention = 'unidirectional' if self.causal else 'bidirectional'
            ),
            max_seq_length = self.seq_len,
            attn_mask_mode = 'add'
        )

    # 前向传播函数
    def forward(self, x, mask = None, rotary_pos_emb = None):
        b, n, _, h, device = *x.shape, self.heads, x.device  # 获取输入张量的形状和设备信息
        remainder = n % self.block_size  # 计算余数
        mask = default(mask, lambda: torch.ones(b, n, device = device).bool())  # 设置默认掩码

        if remainder > 0:
            padding = self.block_size - remainder  # 计算填充大小
            x = F.pad(x, (0, 0, 0, padding), value = 0)  # 对输入张量进行填充
            mask = F.pad(mask, (0, padding), value = False)  # 对掩码进行填充

        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 将输入张量转换为查询、键、值
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # 重排查询、键、值的维度

        if exists(rotary_pos_emb):  # 如果存在旋转位置编码
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))  # 应用位置编码

        key_pad_mask = None  # 初始化键掩码
        if exists(mask):  # 如果存在掩码
            key_pad_mask = ~mask  # 生成键掩码

        attn_mask = None  # 初始化注意力掩码
        if self.causal:  # 如果是因果注意力
            i, j = q.shape[-2], k.shape[-2]  # 获取查询和键的长度
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()  # 生成上三角掩码
            attn_mask = torch.zeros(i, j, device = device).to(q)  # 初始化注意力掩码
            mask_value = max_neg_value(q) / 2  # 计算掩码值
            attn_mask.masked_fill_(mask, mask_value)  # 填充注意力掩码

        # 使用稀疏自注意力机制进行计算
        out = self.attn_fn(q, k, v, attn_mask = attn_mask, key_padding_mask = key_pad_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重排输出维度
        out = self.to_out(out)  # 输出层处理
        return out[:, :n]  # 返回结果
```