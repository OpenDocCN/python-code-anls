# `bitsandbytes\bitsandbytes\optim\adamw.py`

```py
# 导入必要的模块
from bitsandbytes.optim.optimizer import Optimizer2State

# 定义 AdamW 类，继承自 Optimizer2State 类
class AdamW(Optimizer2State):
    # 初始化方法，接受一系列参数
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 调用父类的初始化方法，传入参数
        super().__init__( "adam", params, lr, betas, eps, weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged )

# 定义 AdamW8bit 类，继承自 Optimizer2State 类
class AdamW8bit(Optimizer2State):
    # 初始化方法，接受一系列参数
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 调用父类的初始化方法，传入参数
        super().__init__( "adam", params, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged )

# 定义 AdamW32bit 类，继承自 Optimizer2State 类
class AdamW32bit(Optimizer2State):
    # 初始化方法，接受一系列参数
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 调用父类的初始化方法，传入参数
        super().__init__( "adam", params, lr, betas, eps, weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged)

# 定义 PagedAdamW 类，继承自 Optimizer2State 类
class PagedAdamW(Optimizer2State):
    # 初始化方法，接受一系列参数
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        # 调用父类的初始化方法，传入参数，并设置 is_paged 为 True
        super().__init__( "adam", params, lr, betas, eps, weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True)

# 定义 PagedAdamW8bit 类，继承自 Optimizer2State 类
class PagedAdamW8bit(Optimizer2State):
    # 初始化 Adam 优化器对象
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        # 调用父类的初始化方法，传入参数和默认参数值
        super().__init__( "adam", params, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True)
# 定义一个名为 PagedAdamW32bit 的类，继承自 Optimizer2State 类
class PagedAdamW32bit(Optimizer2State):
    # 初始化方法，接受一系列参数
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True):
        # 调用父类的初始化方法，传入参数 "adam", params, lr, betas, eps, weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True
        super().__init__( "adam", params, lr, betas, eps, weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True)
```