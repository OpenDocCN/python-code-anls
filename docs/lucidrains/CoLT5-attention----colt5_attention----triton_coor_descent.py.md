# `.\lucidrains\CoLT5-attention\colt5_attention\triton_coor_descent.py`

```
# 从 math 模块中导入 log 函数
from math import log

# 导入 torch 模块及相关类和函数
import torch
from torch import Tensor
from torch import autograd
import torch.nn.functional as F
from torch.cuda.amp import autocast, custom_fwd, custom_bwd

# 从 colt5_attention 模块中导入 coor_descent 函数
from colt5_attention.coor_descent import coor_descent
# 从 einops 模块中导入 pack、unpack、repeat 函数
from einops import pack, unpack, repeat

# 尝试导入 triton 模块及相关类和函数
try:
    import triton
    import triton.language as tl
except ImportError as e:
    # 如果导入失败，则打印提示信息
    print('triton is not installed, please install by running `pip install triton -U --pre`')
    # 退出程序
    exit()

# 确保使用的是最新版本的 triton

# 导入版本模块，用于比较 triton 版本
from packaging import version
# 断言 triton 版本大于等于 '2.0'
assert version.parse(triton.__version__) >= version.parse('2.0')

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 计算块大小对应的 warp 数量
def calc_num_warps(block_size):
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    return num_warps

# 将张量按照指定模式进行打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包后的张量按照指定模式进行解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]    

# 将数字分成指定组数
def num_to_groups(num, groups):
    assert 0 < groups <= num
    floor = num // groups
    remainder = num % groups
    out = []
    for ind in range(groups):
        out.append(floor + int(ind < remainder))
    assert sum(out) == num
    return out

# 前向传播

# 定义前向传播的 Triton 内核函数
@triton.jit
def coor_descent_kernel_forward(
    a_ptr,
    b_ptr,
    input_ptr,
    mask_ptr,
    k_ptr,
    a_iter_stride,
    b_row_stride,
    b_iter_stride,
    input_row_stride,
    mask_row_stride,
    n_iters,
    current_eps,
    eps_decay,
    eps,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    # 加载 mask 作为整数（因为布尔值会导致 Triton 出错）

    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets

    mask_ints = tl.load(mask_ptrs, mask = col_mask, other = 0)
    mask = mask_ints == 1

    # 加载 a 和 b

    a_ptr = a_ptr + row_idx
    a = tl.load(a_ptr)

    b_start_ptr = b_ptr + row_idx * b_row_stride
    b_ptrs = b_start_ptr + col_offsets
    b = tl.load(b_ptrs, mask = col_mask, other = 0)

    # 加载得分 s

    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    s = tl.load(input_ptrs, mask = mask, other = -float('inf'))

    # 加载 k - 控制输出的稀疏性

    k_ptr = k_ptr + row_idx
    k = tl.load(k_ptr)

    # 初始化一些常数

    logk = tl.log(k)

    for _ in range(n_iters):        

        a = (s + b) / current_eps
        a = tl.where(mask, a, -float('inf'))

        # 稳定的对数求和指数

        a_max = tl.max(a, axis = 0)
        a_minus_max = tl.where(mask, a - a_max, -float('inf'))
        exp = tl.exp(a_minus_max)
        sum_exp = tl.sum(exp, axis = 0)
        log_sum_exp = tl.log(sum_exp) + a_max

        a = current_eps * (logk - log_sum_exp)

        # 更新 b

        b = s + a
        b = tl.where(b >= 0., -b, 0.)

        # 衰减 epsilon，从 epsilon 缩放

        current_eps *= eps_decay

        if current_eps < eps:
            current_eps = eps

    # 存储 a 和 b 以备下一轮使用

    next_a_ptrs = a_ptr + a_iter_stride
    next_b_ptrs = b_ptrs + b_iter_stride

    tl.store(next_a_ptrs, a)
    tl.store(next_b_ptrs, b, mask = col_mask)

# 反向传播

# 定义反向传播的 Triton 内核函数
@triton.jit
def coor_descent_kernel_backward(
    dk_ptr,
    input_ptr,
    a_ptr,
    b_ptr,
    mask_ptr,
    ds_ptr,
    db_ptr,
    k_ptr,
    last_da_ptr,
    input_row_stride,
    b_row_stride,
    mask_row_stride,
    ds_row_stride,
    db_row_stride,
    n_iters,
    eps_init,
    eps_decay,
    eps,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 加载和生成 mask

    col_mask = col_offsets < n_cols

    # 加载 mask 作为整数（因为布尔值会导致 Triton 出错）

    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    # 计算掩码指针
    mask_ptrs = mask_start_ptr + col_offsets

    # 从指定位置加载整数值
    mask_ints = tl.load(mask_ptrs, mask = col_mask, other = 0)
    # 创建布尔掩码
    mask = mask_ints == 1

     # 加载 a 和 b

    # 更新 a 指针
    a_ptr = a_ptr + row_idx
    # 加载初始值 a
    init_a = tl.load(a_ptr)

    # 更新 b 起始指针
    b_start_ptr = b_ptr + row_idx * b_row_stride
    # 计算 b 指针
    b_ptrs = b_start_ptr + col_offsets
    # 加载初始值 b
    init_b = tl.load(b_ptrs, mask = mask, other = 0)

    # 加载输入

    # 更新行起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # 计算输入指针
    input_ptrs = row_start_ptr + col_offsets
    # 加载输入值
    s = tl.load(input_ptrs, mask = mask, other = -float('inf'))

    # 加载 k - 控制输出的稀疏性

    # 更新 k 指针
    k_ptr = k_ptr + row_idx
    # 加载 k 值
    k = tl.load(k_ptr)
    # 计算 k 的自然对数
    logk = tl.log(k)

    # 加载上一个 da

    # 更新上一个 da 指针
    last_da_ptr = last_da_ptr + row_idx
    # 加载上一个 da 值
    last_da = tl.load(last_da_ptr)

    # 加载初始 ds

    # 更新 ds 行起始指针
    ds_row_start_ptr = ds_ptr + row_idx * ds_row_stride
    # 计算 ds 指针
    ds_ptrs = ds_row_start_ptr + col_offsets
    # 加载初始 ds 值
    ds = tl.load(ds_ptrs, mask = mask, other = 0.)

    # 加载初始 db

    # 更新 db 行起始指针
    db_row_start_ptr = db_ptr + row_idx * db_row_stride
    # 计算 db 指针
    db_ptrs = db_row_start_ptr + col_offsets
    # 加载初始 db 值
    db = tl.load(db_ptrs, mask = mask, other = 0.)

    # 加载初始 dk

    # 更新 dk 指针
    dk_ptr = dk_ptr + row_idx
    # 加载 dk 值
    dk = tl.load(dk_ptr)

    # 反向传播

    for ind in range(n_iters):
        a = init_a
        b = init_b

        sa = s * 0
        softmax = s * 0

        # 计算 epsilon

        current_eps = eps_init / eps_decay

        # 重新计算

        for _ in range(n_iters - ind):
            # 更新 epsilon

            current_eps *= eps_decay

            if current_eps < eps:
                current_eps = eps

            # 更新 a

            sb = (s + b) / current_eps
            sb = tl.where(mask, sb, -float('inf'))

            # 稳定的对数求和指数

            sb_max = tl.max(sb, axis = 0)
            sb_minus_max = tl.where(mask, sb - sb_max, -float('inf'))
            exp = tl.exp(sb_minus_max)
            sum_exp = tl.sum(exp, axis = 0)
            softmax = exp / sum_exp
            log_sum_exp = tl.log(sum_exp) + sb_max

            a = current_eps * (logk - log_sum_exp)

            # 更新 b

            sa = s + a
            b = tl.where(sa > 0., -sa, 0.)

        # 向后传播

        dsa = db * tl.where(sa > 0, -1., 0.)

        ds += dsa

        da = tl.sum(dsa, axis = 0) + last_da

        dk += da * current_eps

        dsb = da * -softmax

        ds += dsb
        db = dsb

        last_da *= 0.

    # 存储 dk

    tl.store(dk_ptr, dk)

    # 存储 ds

    tl.store(ds_ptrs, ds, mask = col_mask)

    # 存储 db

    tl.store(db_ptrs, db, mask = col_mask)
# 定义一个继承自autograd.Function的类_coor_descent，用于实现坐标下降算法
class _coor_descent(autograd.Function):
    # 前向传播函数
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x,
        n_iters,
        k,
        eps,
        eps_init,
        eps_decay,
        mask,
        checkpoint_segments
    ):
        # 断言迭代次数大于0
        assert n_iters > 0
        # 断言输入张量在CUDA上
        assert x.is_cuda, 'triton coordinate descent must be on cuda'

        # 获取输入张量的批大小、是否需要梯度、设备和数据类型
        batch, requires_grad, device, dtype = x.shape[0], x.requires_grad, x.device, x.dtype

        # 如果mask不存在，则创建一个与x相同形状的全1张量
        if not exists(mask):
            mask = torch.ones_like(x, dtype=torch.bool, device=x.device)

        # 将x和mask打包成一维张量
        x, shape = pack_one(x, '* n')
        mask, _ = pack_one(mask, '* n')

        # 将x中mask为False的元素替换为最小值
        x = x.masked_fill(~mask, -torch.finfo(x.dtype).max)
        mask_ints = mask.int()

        epsilons = []
        eps_init = default(eps_init, eps)
        current_eps = float(max(eps_init, eps))

        n_rows, n_cols = x.shape

        # 如果k是整数或浮点数，则创建一个全为k的张量
        if isinstance(k, (int, float)):
            k = torch.full((n_rows,), k)

        # 断言k的元素数量与行数相同
        assert k.numel() == n_rows

        k = k.to(x)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        # 断言BLOCK_SIZE小于等于131072
        assert BLOCK_SIZE <= 131072, 'the maximum block size allowed is 131072 for triton cuda kernel - set the `route_block_size` for the CoordinateDescentRouter to be this value or less in order to uniformly route to get around this limitation'

        num_warps = calc_num_warps(BLOCK_SIZE)

        checkpointed_a = torch.empty((checkpoint_segments + 1, n_rows), device=device, dtype=dtype)
        checkpointed_b = torch.empty((checkpoint_segments + 1, n_rows, n_cols), device=device, dtype=dtype)

        checkpointed_a[0] = torch.zeros_like(k)
        checkpointed_b[0] = -x

        for ind, segment_iters in enumerate(num_to_groups(n_iters, checkpoint_segments)):
            is_last = ind == (checkpoint_segments - 1)

            epsilons.append(current_eps)

            # 调用CUDA核函数进行坐标下降计算
            coor_descent_kernel_forward[(n_rows,)](
                checkpointed_a[ind],
                checkpointed_b[ind],
                x,
                mask_ints,
                k,
                checkpointed_a.stride(0),
                n_cols,
                checkpointed_b.stride(0),
                x.stride(0),
                mask_ints.stride(0),
                segment_iters,
                current_eps,
                eps_decay,
                eps,
                n_cols,
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            current_eps *= (eps_decay ** segment_iters)
            current_eps = max(current_eps, eps)

        last_a, last_b = map(lambda t: t[-1], (checkpointed_a, checkpointed_b))
        y = torch.exp((last_a[..., None] + last_b + x) / current_eps)

        epsilons.append(current_eps)

        if requires_grad:
            checkpointed_a = checkpointed_a[:-1]
            checkpointed_b = checkpointed_b[:-1]

            ctx.args = (n_iters, checkpoint_segments, epsilons, eps_decay, eps)
            ctx.save_for_backward(x, y, k, mask, checkpointed_a, checkpointed_b)

        y = unpack_one(y, shape, '* n')

        return y

    # 反向传播函数
    @staticmethod
    @custom_bwd
    def backward(
        ctx,
        grad_probs
    ):
        # 断言梯度概率是否在 GPU 上
        assert grad_probs.is_cuda

        # 获取批量大小
        batch = grad_probs.shape[0]

        # 从上下文中获取参数
        n_iters, checkpoint_segments, epsilons, eps_decay, eps = ctx.args
        x, y, k, mask, checkpointed_a, checkpointed_b = ctx.saved_tensors

        # 将梯度概率打包成指定形状
        grad_probs, shape = pack_one(grad_probs, '* n')

        # 如果存在掩码，则将梯度概率中的非掩码部分置零
        if exists(mask):
            grad_probs = grad_probs.masked_fill(~mask, 0.)

        # 获取梯度概率的行数和列数
        n_rows, n_cols = grad_probs.shape

        # 计算块大小
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        # 解包 epsilon 值
        *epsilons, last_eps = epsilons

        # 计算 ds, db, dk, last_da
        ds = grad_probs * y / last_eps
        db = ds.clone()
        dk = torch.zeros_like(k)
        last_da = ds.sum(dim=-1)

        # 将掩码转换为整数类型
        mask_int = mask.int()

        # 使用 zip 函数将多个迭代器的元素打包成元组
        items = zip(
            reversed(checkpointed_a.unbind(dim=0)),
            reversed(checkpointed_b.unbind(dim=0)),
            reversed(num_to_groups(n_iters, checkpoint_segments)),
            reversed(epsilons)
        )

        # 遍历 items 中的元素
        for ind, (init_a, init_b, segment_iters, eps_init) in enumerate(items):
            is_first = ind == 0

            # 调用 coor_descent_kernel_backward 函数
            coor_descent_kernel_backward[(n_rows,)](
                dk,
                x,
                init_a,
                init_b,
                mask_int,
                ds,
                db,
                k,
                last_da if is_first else torch.zeros_like(last_da),
                x.stride(0),
                init_b.stride(0),
                mask_int.stride(0),
                ds.stride(0),
                db.stride(0),
                segment_iters,
                eps_init,
                eps_decay,
                eps,
                n_cols,
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE
            )

        # 更新 ds
        ds += -db
        ds = unpack_one(ds, shape, '* n')

        # 如果 k 不需要梯度，则将 dk 置为 None
        if not k.requires_grad:
            dk = None
        else:
            dk /= k

        # 返回结果
        return ds, None, dk, None, None, None, None, None
# 禁用自动类型转换的装饰器
@autocast(enabled = False)
# Triton 坐标下降算法
def triton_coor_descent(
    s,  # 输入张量
    *,
    n_iters,  # 迭代次数
    k,  # 参数 k
    eps = 1e-1,  # 精度参数，默认为 0.1
    eps_init = None,  # 初始精度参数
    eps_decay = 1.,  # 精度参数衰减率
    mask = None,  # 掩码
    checkpoint_segments = 1  # 检查点段数
):
    # 如果输入张量不在 CUDA 上，则使用普通的坐标下降算法
    if not s.is_cuda:
        return coor_descent(s, n_iters = n_iters, k = k, eps = eps, eps_init = eps_init, eps_decay = eps_decay, mask = mask)

    # 在 CUDA 上使用自定义的坐标下降算法
    return _coor_descent.apply(s, n_iters, k, eps, eps_init, eps_decay, mask, checkpoint_segments)
```