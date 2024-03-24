# `.\lucidrains\triton-transformer\triton_transformer\utils.py`

```
# 检查值是否不为 None
def exists(val):
    return val is not None

# 如果值存在，则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 根据块大小计算 warp 数量
def calc_num_warps(block_size):
    # 默认 warp 数量为 4
    num_warps = 4
    # 如果块大小大于等于 2048，则 warp 数量为 8
    if block_size >= 2048:
        num_warps = 8
    # 如果块大小大于等于 4096，则 warp 数量为 16
    if block_size >= 4096:
        num_warps = 16
    # 返回 warp 数量
    return num_warps
```