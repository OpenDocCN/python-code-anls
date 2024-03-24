# `.\lucidrains\nuwa-pytorch\nuwa_pytorch\reversible_video_audio.py`

```
import torch
import torch.nn as nn
from torch.autograd.function import Function
from contextlib import contextmanager

from nuwa_pytorch.reversible import Deterministic

from einops import reduce

# helpers

# 检查值是否存在
def exists(val):
    return val is not None

# 上下文管理器，不执行任何操作
@contextmanager
def null_context():
    yield

# 在指定维度上按索引分割张量
def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# reversible self attention block

class ReversibleSelfAttnBlock(nn.Module):
    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)        

    def forward(self, x, m, _reverse = True, **kwargs):
        x1, x2 = torch.chunk(x, 2, dim = 2)
        m1, m2 = torch.chunk(m, 2, dim = 2)
        y1, y2, n1, n2 = None, None, None, None

        fn_context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse

        with fn_context():
            y1 = x1 + self.f(x2, record_rng = record_rng)
            y2 = x2 + self.g(y1, record_rng = record_rng)
            n1 = m1 + self.j(m2, record_rng = record_rng)
            n2 = m2 + self.k(n1, record_rng = record_rng)

        return torch.cat((y1, y2), dim = 2), torch.cat((n1, n2), dim = 2)

    def backward_pass(self, y, n, dy, dn, **kwargs):
        y1, y2 = torch.chunk(y, 2, dim = 2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim = 2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng = True)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng = True)
            torch.autograd.backward(fx2, dx1, retain_graph = True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim = 2)
            dx = torch.cat([dx1, dx2], dim = 2)

        n1, n2 = torch.chunk(n, 2, dim = 2)
        del n

        dn1, dn2 = torch.chunk(dn, 2, dim = 2)
        del dn

        with torch.enable_grad():
            n1.requires_grad = True
            gn1 = self.k(n1, set_rng = True)
            torch.autograd.backward(gn1, dn2)

        with torch.no_grad():
            m2 = n2 - gn1
            del n2, gn1

            dm1 = dn1 + n1.grad
            del dn1
            n1.grad = None

        with torch.enable_grad():
            m2.requires_grad = True
            fm2 = self.j(m2, set_rng = True)
            torch.autograd.backward(fm2, dm1, retain_graph=True)

        with torch.no_grad():
            m1 = n1 - fm2
            del n1, fm2

            dm2 = dn2 + m2.grad
            del dn2
            m2.grad = None

            m = torch.cat([m1, m2.detach()], dim = 2)
            dm = torch.cat([dm1, dm2], dim = 2)

        return x, m, dx, dm

class ReversibleCrossAttnBlock(nn.Module):
    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)        
    # 前向传播函数，接受输入 x 和 m，以及一系列参数，返回处理后的结果
    def forward(self, x, m, *, context, context_mask, video_mask = None, audio_mask = None, _reverse = True, **kwargs):
        # 将输入 x 和 m 按照第二维度分成两部分
        x1, x2 = torch.chunk(x, 2, dim = 2)
        m1, m2 = torch.chunk(m, 2, dim = 2)
        y1, y2, n1, n2 = None, None, None, None

        # 根据 _reverse 参数选择是否启用梯度记录
        fn_context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse

        # 使用 fn_context 上下文管理器，根据 _reverse 参数选择是否启用梯度记录
        with fn_context():
            # 计算 y1 和 y2
            y1 = x1 + self.f(x2, context = context, context_mask = context_mask, mask = video_mask, record_rng = record_rng)
            y2 = x2 + self.g(y1, record_rng = record_rng)
            # 计算 n1 和 n2
            n1 = m1 + self.j(m2, context = context, context_mask = context_mask, mask = audio_mask, record_rng = record_rng)
            n2 = m2 + self.k(n1, record_rng = record_rng)

        # 返回拼接后的结果
        return torch.cat((y1, y2), dim = 2), torch.cat((n1, n2), dim = 2)

    # 反向传播函数，接受输入 y, n, dy, dn，以及一系列参数，返回处理后的结果
    def backward_pass(self, y, n, dy, dn, *, context, context_mask, video_mask = None, audio_mask = None, **kwargs):
        # 将输入 y 和 n 按照第二维度分成两部分
        y1, y2 = torch.chunk(y, 2, dim = 2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim = 2)
        del dy

        # 启用梯度记录
        with torch.enable_grad():
            y1.requires_grad = True
            # 计算 gy1
            gy1 = self.g(y1, set_rng = True)
            # 反向传播计算 dy2
            torch.autograd.backward(gy1, dy2)

        # 使用 torch.no_grad 上下文管理器，计算中间结果
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        # 启用梯度记录
        with torch.enable_grad():
            x2.requires_grad = True
            # 计算 fx2
            fx2 = self.f(x2, set_rng = True, context = context, context_mask = context_mask, mask = video_mask)
            # 反向传播计算 dx1
            torch.autograd.backward(fx2, dx1, retain_graph = True)

        # 使用 torch.no_grad 上下文管理器，计算中间结果
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim = 2)
            dx = torch.cat([dx1, dx2], dim = 2)

        # 将输入 n 按照第二维度分成两部分
        n1, n2 = torch.chunk(n, 2, dim = 2)
        del n

        dn1, dn2 = torch.chunk(dn, 2, dim = 2)
        del dn

        # 启用梯度记录
        with torch.enable_grad():
            n1.requires_grad = True
            # 计算 gn1
            gn1 = self.k(n1, set_rng = True)
            # 反向传播计算 dn2
            torch.autograd.backward(gn1, dn2)

        # 使用 torch.no_grad 上下文管理器，计算中间结果
        with torch.no_grad():
            m2 = n2 - gn1
            del n2, gn1

            dm1 = dn1 + n1.grad
            del dn1
            n1.grad = None

        # 启用梯度记录
        with torch.enable_grad():
            m2.requires_grad = True
            # 计算 fm2
            fm2 = self.j(m2, set_rng = True, context = context, context_mask = context_mask, mask = audio_mask)
            # 反向传播计算 dm1
            torch.autograd.backward(fm2, dm1, retain_graph=True)

        # 使用 torch.no_grad 上下文管理器，计算中间结果
        with torch.no_grad():
            m1 = n1 - fm2
            del n1, fm2

            dm2 = dn2 + m2.grad
            del dn2
            m2.grad = None

            m = torch.cat([m1, m2.detach()], dim = 2)
            dm = torch.cat([dm1, dm2], dim = 2)

        # 返回结果
        return x, m, dx, dm
# 可逆交叉模态注意力块

class ReversibleCrossModalityAttnBlock(nn.Module):
    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)  # 初始化可逆函数 f
        self.g = Deterministic(g)  # 初始化可逆函数 g
        self.j = Deterministic(j)  # 初始化可逆函数 j
        self.k = Deterministic(k)  # 初始化可逆函数 k

    def forward(self, x, m, *, video_mask = None, audio_mask = None, _reverse = True, **kwargs):
        x1, x2 = torch.chunk(x, 2, dim = 2)  # 将输入 x 沿着第二维度分成两部分 x1 和 x2
        m1, m2 = torch.chunk(m, 2, dim = 2)  # 将输入 m 沿着第二维度分成两部分 m1 和 m2
        y1, y2, n1, n2 = None, None, None, None

        fn_context = torch.no_grad if _reverse else null_context  # 根据 _reverse 的值选择上下文管理器
        record_rng = self.training and _reverse

        with fn_context():
            y1 = x1 + self.f(x2, m2, record_rng = record_rng, mask = video_mask, context_mask = audio_mask)  # 计算 y1
            y2 = x2 + self.k(y1, record_rng = record_rng)  # 计算 y2
            n1 = m1 + self.j(m2, y2, record_rng = record_rng, mask = audio_mask, context_mask = video_mask)  # 计算 n1
            n2 = m2 + self.g(n1, record_rng = record_rng)  # 计算 n2

        return torch.cat((y1, y2), dim = 2), torch.cat((n1, n2), dim = 2)  # 返回拼接后的结果

    def backward_pass(self, y, n, dy, dn, video_mask = None, audio_mask = None, **kwargs):
        n1, n2 = torch.chunk(n, 2, dim = 2)  # 将输入 n 沿着第二维度分成两部分 n1 和 n2
        del n

        dn1, dn2 = torch.chunk(dn, 2, dim = 2)  # 将输入 dn 沿着第二维度分成两部分 dn1 和 dn2
        del dn

        y1, y2 = torch.chunk(y, 2, dim = 2)  # 将输入 y 沿着第二维度分成两部分 y1 和 y2
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim = 2)  # 将输入 dy 沿着第二维度分成两部分 dy1 和 dy2
        del dy

        with torch.enable_grad():
            n1.requires_grad = True
            gn1 = self.g(n1, set_rng = True)  # 计算 gn1
            torch.autograd.backward(gn1, dn2)  # 反向传播计算梯度

        with torch.no_grad():
            m2 = n2 - gn1  # 计算 m2
            del n2, gn1

            dm1 = dn1 + n1.grad  # 计算 dm1
            del dn1
            n1.grad = None

        with torch.enable_grad():
            m2.requires_grad = True
            y2.requires_grad = True
            fm2 = self.j(m2, y2, set_rng=True, mask = audio_mask, context_mask = video_mask)  # 计算 fm2
            torch.autograd.backward(fm2, dm1)  # 反向传播计算梯度

        with torch.no_grad():
            m1 = n1 - fm2  # 计算 m1
            del n1, fm2

            dm2 = dn2 + m2.grad  # 计算 dm2
            dx2 = dy2 + y2.grad  # 计算 dx2
            del dn2
            del dy2
            m2.grad = None
            y2.grad = None

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.k(y1, set_rng = True)  # 计算 gy1
            torch.autograd.backward(gy1, dx2)  # 反向传播计算梯度

        with torch.no_grad():
            x2 = y2 - gy1  # 计算 x2
            del y2, gy1

            dx1 = dy1 + y1.grad  # 计算 dx1
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            m2.requires_grad = True
            fx2 = self.f(x2, m2, set_rng = True, mask = video_mask, context_mask = audio_mask)  # 计算 fx2
            torch.autograd.backward(fx2, dx1)  # 反向传播计算梯度

        with torch.no_grad():
            x1 = y1 - fx2  # 计算 x1
            del y1, fx2

            dx2 = dx2 + x2.grad  # 计算 dx2
            dm2 = dm2 + m2.grad  # 计算 dm2
            x2.grad = None
            m2.grad = None

        with torch.no_grad():
            m = torch.cat([m1, m2.detach()], dim = 2)  # 拼接 m1 和 m2
            dm = torch.cat([dm1, dm2], dim = 2)  # 拼接 dm1 和 dm2

            x = torch.cat([x1, x2.detach()], dim = 2)  # 拼接 x1 和 x2
            dx = torch.cat([dx1, dx2], dim = 2)  # 拼接 dx1 和 dx2

        return x, m, dx, dm

# 反向和非反向函数

class ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, inp, ind, blocks, kwargs):
        x, m = split_at_index(1, ind, inp)  # 在指定索引处分割输入

        for block in blocks:
            x, m = block(x, m, _reverse = True, **kwargs)  # 对每个块进行前向传播

        ctx.blocks = blocks
        ctx.kwargs = kwargs
        ctx.ind = ind
        ctx.save_for_backward(x.detach(), m.detach())
        return torch.cat((x, m), dim = 1)  # 拼接结果

    @staticmethod
    # 定义一个反向传播函数，接受上下文和梯度作为参数
    def backward(ctx, d):
        # 从上下文中获取索引、块和关键字参数
        ind = ctx.ind
        blocks = ctx.blocks
        kwargs = ctx.kwargs
        # 将梯度按照索引分割成两部分
        dy, dn = split_at_index(1, ind, d)
        # 从上下文中获取保存的张量 y 和 n
        y, n = ctx.saved_tensors

        # 对块列表进行反向遍历
        for block in blocks[::-1]:
            # 调用每个块的反向传播函数，更新 y、n、dy 和 dn
            y, n, dy, dn = block.backward_pass(y, n, dy, dn, **kwargs)

        # 将分割后的梯度拼接在一起
        d = torch.cat((dy, dn), dim=1)
        # 返回更新后的梯度和 None（因为没有额外的返回值）
        return d, None, None, None
# 将 ReversibleFunction.apply 赋值给 reversible_apply
reversible_apply = ReversibleFunction.apply

# 定义不可逆应用函数，接受输入、索引、块和关键字参数
def irreversible_apply(inputs, ind, blocks, kwargs):
    # 在索引处将输入分割为 x 和 m
    x, m = split_at_index(1, ind, inputs)
    # 对每个块应用，更新 x 和 m
    for block in blocks:
        x, m = block(x, m, _reverse = False, **kwargs)
    # 拼接 x 和 m，返回结果
    return torch.cat((x, m), dim = 1)

# 主要的可逆序列类
class DualModalityReversibleSequence(nn.Module):
    # 初始化函数，接受输入块和块类型
    def __init__(self, input_blocks, block_types):
        super().__init__()
        self.block_types = block_types
        blocks = nn.ModuleList([])

        # 遍历输入块和块类型，根据类型选择可逆类别
        for block, block_type in zip(input_blocks, block_types):
            if block_type == 'intra_modality_self_attn':
                reversible_klass = ReversibleSelfAttnBlock
            elif block_type == 'intra_modality_cross_attn':
                reversible_klass = ReversibleCrossAttnBlock
            elif block_type == 'inter_modality_cross_attn':
                reversible_klass = ReversibleCrossModalityAttnBlock
            else:                
                raise ValueError(f'unknown layer type {block_type}')

            blocks.append(reversible_klass(*block))

        self.blocks = blocks

    # 前向传播函数，接受视频、音频、上下文和掩码等参数
    def forward(
        self,
        video,
        audio,
        *,
        context,
        context_mask = None,
        video_mask = None,
        audio_mask = None,
        reverse = True
    ):  
        blocks = self.blocks
        # 将视频和音频拼接起来
        video, audio = list(map(lambda t: torch.cat((t, t), dim = -1), (video, audio)))
        kwargs = {'context': context, 'context_mask': context_mask, 'video_mask': video_mask, 'audio_mask': audio_mask}

        # 根据是否可逆选择应用函数
        fn = reversible_apply if reverse else irreversible_apply
        ind = video.shape[1]
        inp = torch.cat((video, audio), dim = 1)
        out = fn(inp, ind, blocks, kwargs)
        # 将输出拆分为视频和音频
        video, audio  = split_at_index(1, ind, out)
        # 对视频和音频应用 reduce 函数，返回结果
        return list(map(lambda t: reduce(t, 'b n (c d) -> b n d', 'mean', c = 2), (video, audio)))
```