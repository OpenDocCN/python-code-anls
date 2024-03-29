# `.\lucidrains\alphafold2\alphafold2_pytorch\reversible.py`

```py
import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from contextlib import contextmanager

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

# 用于反向传播确定性的函数包装器

class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    # 记录随机数生成器状态
    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# 可逆自注意力块

class ReversibleSelfAttnBlock(nn.Module):
    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)        

    def forward(self, x, m, mask = None, msa_mask = None, seq_shape = None, msa_shape = None, seq_pos_emb = None, msa_pos_emb = None, _reverse = True, **kwargs):
        x1, x2 = torch.chunk(x, 2, dim = 2)
        m1, m2 = torch.chunk(m, 2, dim = 2)
        y1, y2, n1, n2 = None, None, None, None

        context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse

        with context():
            y1 = x1 + self.f(x2, shape = seq_shape, record_rng = record_rng, mask = mask, rotary_emb = seq_pos_emb)
            y2 = x2 + self.g(y1, shape = seq_shape, record_rng = record_rng)
            n1 = m1 + self.j(m2, shape = msa_shape, record_rng = record_rng, mask = msa_mask, rotary_emb = msa_pos_emb)
            n2 = m2 + self.k(n1, record_rng = record_rng)

        return torch.cat((y1, y2), dim = 2), torch.cat((n1, n2), dim = 2)

    def backward_pass(self, y, n, dy, dn, mask = None, msa_mask = None, seq_shape = None, msa_shape = None, seq_pos_emb = None, msa_pos_emb = None, **kwargs):
        y1, y2 = torch.chunk(y, 2, dim = 2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim = 2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, shape = seq_shape, set_rng = True)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, shape = seq_shape, set_rng = True, mask = mask, rotary_emb = seq_pos_emb)
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
            fm2 = self.j(m2, shape = msa_shape, set_rng = True, mask = msa_mask, rotary_emb = msa_pos_emb)
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

# 可逆交叉注意力块

class ReversibleCrossAttnBlock(nn.Module):
    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)

    def forward(self, x, m, mask = None, msa_mask = None, seq_shape = None, msa_shape = None, seq_to_msa_pos_emb = None, msa_to_seq_pos_emb = None, _reverse = True, **kwargs):
        x1, x2 = torch.chunk(x, 2, dim = 2)
        m1, m2 = torch.chunk(m, 2, dim = 2)
        y1, y2, n1, n2 = None, None, None, None

        context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse

        with context():
            y1 = x1 + self.f(x2, m2, record_rng = record_rng, mask = mask, context_mask = msa_mask, shape = seq_shape, context_shape = msa_shape, rotary_emb = seq_to_msa_pos_emb)
            y2 = x2 + self.k(y1, shape = seq_shape, record_rng = record_rng)
            n1 = m1 + self.j(m2, y2, record_rng = record_rng, mask = msa_mask, context_mask = mask, shape = msa_shape, context_shape = seq_shape, rotary_emb = msa_to_seq_pos_emb)
            n2 = m2 + self.g(n1, record_rng = record_rng)

        return torch.cat((y1, y2), dim = 2), torch.cat((n1, n2), dim = 2)
    # 反向传播函数，计算梯度并更新参数
    def backward_pass(self, y, n, dy, dn, mask = None, msa_mask = None, seq_shape = None, msa_shape = None, seq_to_msa_pos_emb = None, msa_to_seq_pos_emb = None, **kwargs):
        # 将输入张量 n 按照第二维度分成两部分
        n1, n2 = torch.chunk(n, 2, dim = 2)
        # 释放 n 张量的内存
        del n

        # 将输入张量 dn 按照第二维度分成两部分
        dn1, dn2 = torch.chunk(dn, 2, dim = 2)
        # 释放 dn 张量的内存
        del dn

        # 将输入张量 y 按照第二维度分成两部分
        y1, y2 = torch.chunk(y, 2, dim = 2)
        # 释放 y 张量的内存
        del y

        # 将输入张量 dy 按照第二维度分成两部分
        dy1, dy2 = torch.chunk(dy, 2, dim = 2)
        # 释放 dy 张量的内存
        del dy

        # 开启梯度计算
        with torch.enable_grad():
            # 设置 n1 张量需要计算梯度
            n1.requires_grad = True
            # 使用函数 g 计算 gn1，并进行反向传播
            gn1 = self.g(n1, set_rng = True)
            torch.autograd.backward(gn1, dn2)

        # 关闭梯度计算
        with torch.no_grad():
            # 计算 m2，并释放 n2 和 gn1 张量的内存
            m2 = n2 - gn1
            del n2, gn1

            # 计算 dm1，并释放 dn1 张量的内存
            dm1 = dn1 + n1.grad
            del dn1
            n1.grad = None

        # 开启梯度计算
        with torch.enable_grad():
            # 设置 m2 和 y2 张量需要计算梯度
            m2.requires_grad = True
            y2.requires_grad = True
            # 使用函数 j 计算 fm2，并进行反向传播
            fm2 = self.j(m2, y2, set_rng=True, mask = msa_mask, context_mask = mask, shape = msa_shape, context_shape = seq_shape, rotary_emb = msa_to_seq_pos_emb)
            torch.autograd.backward(fm2, dm1)

        # 关闭梯度计算
        with torch.no_grad():
            # 计算 m1，并释放 n1 和 fm2 张量的内存
            m1 = n1 - fm2
            del n1, fm2

            # 计算 dm2 和 dx2，并释放 dn2 和 dy2 张量的内存
            dm2 = dn2 + m2.grad
            dx2 = dy2 + y2.grad
            del dn2
            del dy2
            m2.grad = None
            y2.grad = None

        # 开启梯度计算
        with torch.enable_grad():
            # 设置 y1 需要计算梯度
            y1.requires_grad = True
            # 使用函数 k 计算 gy1，并进行反向传播
            gy1 = self.k(y1, shape = seq_shape, set_rng = True)
            torch.autograd.backward(gy1, dx2)

        # 关闭梯度计算
        with torch.no_grad():
            # 计算 x2，并释放 y2 和 gy1 张量的内存
            x2 = y2 - gy1
            del y2, gy1

            # 计算 dx1，并释放 dy1 张量的内存
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        # 开启梯度计算
        with torch.enable_grad():
            # 设置 x2 和 m2 需要计算梯度
            x2.requires_grad = True
            m2.requires_grad = True
            # 使用函数 f 计算 fx2，并进行反向传播
            fx2 = self.f(x2, m2, set_rng = True, mask = mask, context_mask = msa_mask, shape = seq_shape, context_shape = msa_shape, rotary_emb = seq_to_msa_pos_emb)
            torch.autograd.backward(fx2, dx1)

        # 关闭梯度计算
        with torch.no_grad():
            # 计算 x1，并释放 y1 和 fx2 张量的内存
            x1 = y1 - fx2
            del y1, fx2

            # 更新 dx2 和 dm2，并释放 x2 和 m2 的梯度
            dx2 = dx2 + x2.grad
            dm2 = dm2 + m2.grad
            x2.grad = None
            m2.grad = None

        # 关闭梯度计算
        with torch.no_grad():
            # 拼接 m1 和 m2，释放 m1 和 m2 的梯度
            m = torch.cat([m1, m2.detach()], dim = 2)
            dm = torch.cat([dm1, dm2], dim = 2)

            # 拼接 x1 和 x2，释放 x1 和 x2 的梯度
            x = torch.cat([x1, x2.detach()], dim = 2)
            dx = torch.cat([dx1, dx2], dim = 2)

        # 返回更新后的张量和梯度
        return x, m, dx, dm
# 定义可逆和不可逆函数

class ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, inp, ind, blocks, kwargs):
        # 将输入按照指定索引分割成两部分
        x, m = split_at_index(1, ind, inp)

        # 对每个块进行反向操作
        for block in blocks:
            x, m = block(x, m, _reverse = True, **kwargs)

        # 保存上下文信息
        ctx.blocks = blocks
        ctx.kwargs = kwargs
        ctx.ind = ind
        ctx.save_for_backward(x.detach(), m.detach())
        return torch.cat((x, m), dim = 1)

    @staticmethod
    def backward(ctx, d):
        ind = ctx.ind
        blocks = ctx.blocks
        kwargs = ctx.kwargs
        dy, dn = split_at_index(1, ind, d)
        y, n = ctx.saved_tensors

        # 对每个块进行反向传播
        for block in blocks[::-1]:
            y, n, dy, dn = block.backward_pass(y, n, dy, dn, **kwargs)

        d = torch.cat((dy, dn), dim = 1)
        return d, None, None, None

reversible_apply = ReversibleFunction.apply

def irreversible_apply(inputs, ind, blocks, kwargs):
    # 将输入按照指定索引分割成两部分
    x, m = split_at_index(1, ind, inputs)
    for block in blocks:
        x, m = block(x, m, _reverse = False, **kwargs)
    return torch.cat((x, m), dim = 1)

# 主要的可逆序列类

class ReversibleSequence(nn.Module):
    def __init__(self, input_blocks, block_types):
        super().__init__()
        self.block_types = block_types

        blocks = nn.ModuleList([])

        for block, block_type in zip(input_blocks, block_types):
            if block_type == 'self':
                reversible_klass = ReversibleSelfAttnBlock
            elif block_type == 'cross':
                reversible_klass = ReversibleCrossAttnBlock
            elif block_type == 'conv':
                reversible_klass = ReversibleSelfAttnBlock

            blocks.append(reversible_klass(*block))

        self.blocks = blocks

    def forward(
        self,
        seq,
        msa,
        seq_shape = None,
        msa_shape = None,
        mask = None,
        msa_mask = None,
        seq_pos_emb = None,
        msa_pos_emb = None,
        seq_to_msa_pos_emb = None,
        msa_to_seq_pos_emb = None,
        reverse = True
    ):
        assert exists(msa), 'reversibility does not work with no MSA sequences yet'
        
        blocks = self.blocks
        # 将序列和多序列对齐数据拼接在一起
        seq, msa = list(map(lambda t: torch.cat((t, t), dim = -1), (seq, msa)))
        kwargs = {'mask': mask, 'msa_mask': msa_mask, 'seq_shape': seq_shape, 'msa_shape': msa_shape, 'seq_pos_emb': seq_pos_emb, 'msa_pos_emb': msa_pos_emb, 'seq_to_msa_pos_emb': seq_to_msa_pos_emb, 'msa_to_seq_pos_emb': msa_to_seq_pos_emb}

        fn = reversible_apply if reverse else irreversible_apply
        ind = seq.shape[1]
        inp = torch.cat((seq, msa), dim = 1)
        out = fn(inp, ind, blocks, kwargs)
        seq, msa  = split_at_index(1, ind, out)
        return list(map(lambda t: reduce(t, 'b n (c d) -> b n d', 'mean', c = 2), (seq, msa)))
```