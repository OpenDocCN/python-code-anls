# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\sigma_sampling.py`

```py
import torch
import torch.distributed

from sat import mpu

from ...util import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        # 初始化函数，设置默认的 p_mean 和 p_std 值
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        # 调用函数，生成 n_samples 个样本
        # 通过随机数生成 log_sigma，其均值为 p_mean，标准差为 p_std
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        # 返回 log_sigma 的指数形式
        return log_sigma.exp()


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, uniform_sampling=False):
        # 初始化函数，设置离散化配置、索引数量、是否追加零、是否翻转和是否均匀采样
        self.num_idx = num_idx
        # 实例化离散化配置，得到 sigmas
        self.sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        # 获取数据并行的世界大小
        world_size = mpu.get_data_parallel_world_size()
        self.uniform_sampling = uniform_sampling
        if self.uniform_sampling:
            # 如果进行均匀采样
            i = 1
            while True:
                if world_size % i != 0 or num_idx % (world_size // i) != 0:
                    i += 1
                else:
                    # 计算组数
                    self.group_num = world_size // i
                    break

            assert self.group_num > 0
            assert world_size % self.group_num == 0
            # 计算每组的宽度，即每组中的排名数量
            self.group_width = world_size // self.group_num  # the number of rank in one group
            # 计算每个组的 sigma 区间
            self.sigma_interval = self.num_idx // self.group_num

    def idx_to_sigma(self, idx):
        # 根据索引获取 sigma
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None, return_idx=False):
        # 调用函数，生成 n_samples 个样本
        if self.uniform_sampling:
            # 如果进行均匀采样
            rank = mpu.get_data_parallel_rank()
            # 计算组索引
            group_index = rank // self.group_width
            # 生成索引，范围为 group_index * sigma_interval 到 (group_index + 1) * sigma_interval
            idx = default(
                rand,
                torch.randint(
                    group_index * self.sigma_interval, (group_index + 1) * self.sigma_interval, (n_samples,)
                ),
            )
        else:
            # 如果不进行均匀采样
            # 生成索引，范围为 0 到 num_idx
            idx = default(
                rand,
                torch.randint(0, self.num_idx, (n_samples,)),
            )
        if return_idx:
            # 如果返回索引
            # 返回索引对应的 sigma 和索引
            return self.idx_to_sigma(idx), idx
        else:
            # 如果不返回索引
            # 返回索引对应的 sigma
            return self.idx_to_sigma(idx)


class PartialDiscreteSampling:
    def __init__(self, discretization_config, total_num_idx, partial_num_idx, do_append_zero=False, flip=True):
        # 初始化函数，设置离散化配置、总索引数量、部分索引数量、是否追加零和是否翻转
        self.total_num_idx = total_num_idx
        self.partial_num_idx = partial_num_idx
        # 实例化离散化配置，得到 sigmas
        self.sigmas = instantiate_from_config(discretization_config)(
            total_num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        # 根据索引获取 sigma
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        # 调用函数，生成 n_samples 个样本
        # 生成索引，范围为 0 到 partial_num_idx
        idx = default(
            rand,
            torch.randint(0, self.partial_num_idx, (n_samples,)),
        )
        # 返回索引对应的 sigma
        return self.idx_to_sigma(idx)
```