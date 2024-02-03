# `bitsandbytes\bitsandbytes\optim\lion.py`

```py
# 导入必要的模块
from bitsandbytes.optim.optimizer import Optimizer1State

# Lion 类，继承自 Optimizer1State 类
class Lion(Optimizer1State):
    # Lion 类的初始化方法
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0, optim_bits=32, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 调用父类的初始化方法
        super().__init__("lion", params, lr, betas, 0., weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged)

# Lion8bit 类，继承自 Optimizer1State 类
class Lion8bit(Optimizer1State):
    # Lion8bit 类的初始化方法
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 调用父类的初始化方法，指定优化位数为 8
        super().__init__("lion", params, lr, betas, 0., weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged)

# Lion32bit 类，继承自 Optimizer1State 类
class Lion32bit(Optimizer1State):
    # Lion32bit 类的初始化方法
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 调用父类的初始化方法，指定优化位数为 32
        super().__init__("lion", params, lr, betas, 0., weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged)

# PagedLion 类，继承自 Optimizer1State 类
class PagedLion(Optimizer1State):
    # PagedLion 类的初始化方法
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0, optim_bits=32, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        # 调用父类的初始化方法，指定启用分页
        super().__init__("lion", params, lr, betas, 0., weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True)

# PagedLion8bit 类，继承自 Optimizer1State 类
class PagedLion8bit(Optimizer1State):
    # PagedLion8bit 类的初始化方法
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        # 调用父类的初始化方法，指定优化位数为 8，并启用分页
        super().__init__("lion", params, lr, betas, 0., weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True)
# 定义一个名为 PagedLion32bit 的类，继承自 Optimizer1State 类
class PagedLion32bit(Optimizer1State):
    # 初始化方法，接受一系列参数
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        # 调用父类的初始化方法，传入参数 "lion", params, lr, betas, 0., weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True
        super().__init__("lion", params, lr, betas, 0., weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True)
```