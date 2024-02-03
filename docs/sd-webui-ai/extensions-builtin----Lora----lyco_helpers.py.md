# `stable-diffusion-webui\extensions-builtin\Lora\lyco_helpers.py`

```
import torch

# 根据输入的张量 t, wa, wb，通过张量乘法生成新的权重张量
def make_weight_cp(t, wa, wb):
    # 使用 torch.einsum 函数进行张量乘法操作，生成临时张量
    temp = torch.einsum('i j k l, j r -> i r k l', t, wb)
    # 再次使用 torch.einsum 函数进行张量乘法操作，生成最终的权重张量
    return torch.einsum('i j k l, i r -> r j k l', temp, wa)

# 重新构建传统的矩阵乘法，输入 up, down 为张量，shape 为形状，dyn_dim 为动态维度
def rebuild_conventional(up, down, shape, dyn_dim=None):
    # 将输入张量 reshape 成二维张量
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    # 如果指定了动态维度，则对 up 和 down 进行切片操作
    if dyn_dim is not None:
        up = up[:, :dyn_dim]
        down = down[:dyn_dim, :]
    # 返回矩阵乘法结果，并将结果 reshape 成指定形状
    return (up @ down).reshape(shape)

# 重新构建 CP 分解，输入 up, down, mid 为张量
def rebuild_cp_decomposition(up, down, mid):
    # 将输入张量 reshape 成二维张量
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    # 使用 torch.einsum 函数进行张量乘法操作，生成新的张量
    return torch.einsum('n m k l, i n, m j -> i j k l', mid, up, down)

# 从给定的维度和因子计算分解后的维度，返回一个元组
def factorization(dimension: int, factor:int=-1) -> tuple[int, int]:
    '''
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale.
    secon value is a value for weight.

    Becuase of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 1, 127   127 -> 1, 127    127 -> 1, 127   127 -> 1, 127   127 -> 1, 127
    128 -> 8, 16    128 -> 2, 64     128 -> 4, 32    128 -> 8, 16    128 -> 8, 16
    250 -> 10, 25   250 -> 2, 125    250 -> 2, 125   250 -> 5, 50    250 -> 10, 25
    360 -> 8, 45    360 -> 2, 180    360 -> 4, 90    360 -> 8, 45    360 -> 12, 30
    512 -> 16, 32   512 -> 2, 256    512 -> 4, 128   512 -> 8, 64    512 -> 16, 32
    1024 -> 32, 32  1024 -> 2, 512   1024 -> 4, 256  1024 -> 8, 128  1024 -> 16, 64
    '''

    # 如果指定了因子且能整除，则返回分解后的维度
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    # 如果未指定因子或无法整除，则使用输入维度作为因子
    if factor < 0:
        factor = dimension
    # 初始化m为1，n为dimension
    m, n = 1, dimension
    # 计算m和n的和
    length = m + n
    # 当m小于n时循环执行以下操作
    while m < n:
        # 将m增加1
        new_m = m + 1
        # 当dimension不能被new_m整除时，继续增加new_m
        while dimension % new_m != 0:
            new_m += 1
        # 计算new_n
        new_n = dimension // new_m
        # 如果new_m和new_n的和大于length或者new_m大于factor，则跳出循环
        if new_m + new_n > length or new_m > factor:
            break
        else:
            # 更新m和n为new_m和new_n
            m, n = new_m, new_n
    # 如果m大于n，则交换m和n的值
    if m > n:
        n, m = m, n
    # 返回m和n的值
    return m, n
```