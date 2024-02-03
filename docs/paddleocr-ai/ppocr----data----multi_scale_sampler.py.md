# `.\PaddleOCR\ppocr\data\multi_scale_sampler.py`

```
# 导入需要的模块
from paddle.io import Sampler
import paddle.distributed as dist

import numpy as np
import random
import math

# 定义一个继承自Sampler的MultiScaleSampler类
class MultiScaleSampler(Sampler):
    # 迭代器方法，用于生成数据集中的索引
    def __iter__(self):
        # 如果种子为空，则使用epoch作为随机数种子
        if self.seed is None:
            random.seed(self.epoch)
            self.epoch += 1
        else:
            random.seed(self.seed)
        # 打乱batchs_in_one_epoch_id列表中的元素顺序
        random.shuffle(self.batchs_in_one_epoch_id)
        # 遍历打乱后的batchs_in_one_epoch_id列表，生成索引
        for batch_tuple_id in self.batchs_in_one_epoch_id:
            # 返回索引对应的元素
            yield self.batchs_in_one_epoch[batch_tuple_id]
    # 定义一个迭代方法，用于生成每个批次的数据
    def iter(self):
        # 如果需要打乱数据集
        if self.shuffle:
            # 如果设置了随机种子，则使用该种子
            if self.seed is not None:
                random.seed(self.seed)
            else:
                random.seed(self.epoch)
            # 如果数据集宽度为0，则打乱图片索引
            if not self.ds_width:
                random.shuffle(self.img_indices)
            # 打乱图片批次对
            random.shuffle(self.img_batch_pairs)
            # 计算当前进程需要处理的图片索引
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):
                                              self.num_replicas]
        else:
            # 计算当前进程需要处理的图片索引
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):
                                              self.num_replicas]

        # 初始化起始索引和存储每个批次数据的列表
        start_index = 0
        batchs_in_one_epoch = []
        # 遍历每个批次
        for batch_tuple in self.batch_list:
            curr_w, curr_h, curr_bsz = batch_tuple
            # 计算结束索引
            end_index = min(start_index + curr_bsz, self.n_samples_per_replica)
            # 获取当前批次的图片索引
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            # 如果当前批次样本数不足，则从头部补充
            if n_batch_samples != curr_bsz:
                batch_ids += indices_rank_i[:(curr_bsz - n_batch_samples)]
            start_index += curr_bsz

            # 如果当前批次有数据
            if len(batch_ids) > 0:
                # 如果需要按照数据集宽度处理
                if self.ds_width:
                    # 计算当前批次图片的宽高比
                    wh_ratio_current = self.wh_ratio[self.wh_ratio_sort[
                        batch_ids]]
                    ratio_current = wh_ratio_current.mean()
                    ratio_current = ratio_current if ratio_current * curr_h < self.max_w else self.max_w / curr_h
                else:
                    ratio_current = None
                # 组装批次数据
                batch = [(curr_w, curr_h, b_id, ratio_current)
                         for b_id in batch_ids]
                # 将批次数据添加到列表中
                batchs_in_one_epoch.append(batch)
        # 返回存储每个批次数据的列表
        return batchs_in_one_epoch

    # 设置当前 epoch
    def set_epoch(self, epoch: int):
        self.epoch = epoch

    # 返回数据集的长度
    def __len__(self):
        return self.length
```