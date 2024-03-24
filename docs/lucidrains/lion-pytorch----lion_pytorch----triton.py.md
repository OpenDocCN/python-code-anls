# `.\lucidrains\lion-pytorch\lion_pytorch\triton.py`

```py
import torch
# 导入 torch 库

try:
    import triton
    import triton.language as tl
except ImportError as e:
    print('triton is not installed, please install by running `pip install triton -U --pre`')
    exit()
# 尝试导入 triton 库，如果导入失败则打印错误信息并退出程序

# clone param and exp_avg before autotuning takes place
# as those are updated in-place
# 在自动调整参数之前克隆参数和 exp_avg，因为它们是原地更新的

def clone_inplace_updated_params(nargs):
    nargs['p_ptr'] = nargs['p_ptr'].clone()
    nargs['exp_avg_ptr'] = nargs['exp_avg_ptr'].clone()
# 克隆原地更新的参数和 exp_avg

# triton cuda kernel

@triton.autotune(configs = [
    triton.Config({'BLOCK_SIZE': 128}, num_warps = 4, pre_hook = clone_inplace_updated_params),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps = 8, pre_hook = clone_inplace_updated_params),
], key = ['n_elements'])
@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)
    # 获取程序 ID

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 计算块的起始位置和偏移量

    mask = offsets < n_elements
    # 创建掩码以确保偏移量不超过元素数量

    # offsetted pointers

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets
    # 计算偏移后的指针位置

    # load

    p = tl.load(offset_p_ptr, mask = mask)
    grad = tl.load(offset_grad_ptr, mask = mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask = mask)
    # 从指定位置加载数据

    # stepweight decay

    p = p * (1 - lr * wd)
    # 更新参数

    # diff between momentum running average and grad

    diff = exp_avg - grad
    # 计算动量的运行平均值和梯度之间的差异

    # weight update

    update = diff * beta1 + grad
    # 更新权重

    # torch.sign

    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)
    # 计算更新的符号

    p = p + update_sign * can_update
    # 更新参数

    # decay the momentum running average coefficient

    exp_avg = diff * beta2 + grad
    # 更新动量的运行平均系数

    # store new params and momentum running average coefficient

    tl.store(offset_p_ptr, p, mask = mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask = mask)
    # 存储新的参数和动量的运行平均系数

def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float
):
    assert all([t.is_cuda for t in (p, grad, exp_avg)])
    n_elements = p.numel()
    # 确保参数在 GPU 上，并获取参数数量

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)    
    # 定义网格大小

    update_fn_kernel[grid](
        p,
        grad,
        exp_avg,
        lr,
        wd,
        beta1,
        beta2,
        n_elements
    )
    # 调用 triton 内核函数进行参数更新
```