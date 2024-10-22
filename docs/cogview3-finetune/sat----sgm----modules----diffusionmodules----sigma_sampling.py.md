# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\sigma_sampling.py`

```py
# 导入 PyTorch 库
import torch

# 从相对路径导入 default 和 instantiate_from_config 函数
from ...util import default, instantiate_from_config


# 定义 EDMSampling 类
class EDMSampling:
    # 初始化方法，设置均值和标准差
    def __init__(self, p_mean=-1.2, p_std=1.2):
        # 保存均值到实例变量
        self.p_mean = p_mean
        # 保存标准差到实例变量
        self.p_std = p_std

    # 定义调用方法，允许类实例像函数一样被调用
    def __call__(self, n_samples, rand=None):
        # 计算对数标准差，根据随机数生成 log_sigma
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        # 返回 log_sigma 的指数值
        return log_sigma.exp()


# 定义 DiscreteSampling 类
class DiscreteSampling:
    # 初始化方法，设置离散化配置和其他参数
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, low_bound=0, up_bound=1):
        # 保存索引数量到实例变量
        self.num_idx = num_idx
        # 根据配置实例化 sigma 对象
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        # 计算并保存下界和上界
        self.low_bound = int(low_bound * num_idx)
        self.up_bound = int(up_bound * num_idx)
        # 打印采样范围
        print(f'sigma sampling from {self.low_bound} to {self.up_bound}')

    # 将索引转换为 sigma 值
    def idx_to_sigma(self, idx):
        # 返回对应索引的 sigma
        return self.sigmas[idx]

    # 定义调用方法
    def __call__(self, n_samples, rand=None, return_idx=False):
        # 生成随机索引，如果没有提供随机数则使用默认随机数
        idx = default(
            rand,
            torch.randint(self.low_bound, self.up_bound, (n_samples,)),
        )
        # 根据 return_idx 参数决定返回 sigma 值或索引
        if return_idx:
            return self.idx_to_sigma(idx), idx
        else:
            return self.idx_to_sigma(idx)

# 定义 PartialDiscreteSampling 类
class PartialDiscreteSampling:
    # 初始化方法，设置完整和部分索引数量
    def __init__(self, discretization_config, total_num_idx, partial_num_idx, do_append_zero=False, flip=True):
        # 保存总索引数量到实例变量
        self.total_num_idx = total_num_idx
        # 保存部分索引数量到实例变量
        self.partial_num_idx = partial_num_idx
        # 根据配置实例化 sigma 对象
        self.sigmas = instantiate_from_config(discretization_config)(
            total_num_idx, do_append_zero=do_append_zero, flip=flip
        )

    # 将索引转换为 sigma 值
    def idx_to_sigma(self, idx):
        # 返回对应索引的 sigma
        return self.sigmas[idx]

    # 定义调用方法
    def __call__(self, n_samples, rand=None):
        # 生成随机索引，根据部分索引数量限制随机范围
        idx = default(
            rand,
            # torch.randint(self.total_num_idx-self.partial_num_idx, self.total_num_idx, (n_samples,)),
            torch.randint(0, self.partial_num_idx, (n_samples,)),
        )
        # 返回对应的 sigma 值
        return self.idx_to_sigma(idx)
```