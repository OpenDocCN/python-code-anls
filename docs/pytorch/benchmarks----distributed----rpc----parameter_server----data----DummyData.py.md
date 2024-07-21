# `.\pytorch\benchmarks\distributed\rpc\parameter_server\data\DummyData.py`

```
# 导入随机数模块
import random

# 导入NumPy模块并使用np作为别名
import numpy as np

# 导入PyTorch模块
import torch
# 从torch.utils.data模块中导入Dataset类
from torch.utils.data import Dataset

# 定义DummyData类，继承自Dataset类
class DummyData(Dataset):
    def __init__(
        self,
        max_val: int,
        sample_count: int,
        sample_length: int,
        sparsity_percentage: int,
    ):
        """
        A data class that generates random data.
        Args:
            max_val (int): the maximum value for an element
            sample_count (int): count of training samples
            sample_length (int): number of elements in a sample
            sparsity_percentage (int): the percentage of
                embeddings used by the input data in each iteration
        """
        # 初始化类的实例变量
        self.max_val = max_val
        self.input_samples = sample_count
        self.input_dim = sample_length
        self.sparsity_percentage = sparsity_percentage

        # 内部函数，生成随机输入数据
        def generate_input():
            # 计算非稀疏元素的百分比
            precentage_of_elements = (100 - self.sparsity_percentage) / float(100)
            # 计算生成数据的索引数量
            index_count = int(self.max_val * precentage_of_elements)
            # 生成包含所有元素的列表，并随机打乱
            elements = list(range(self.max_val))
            random.shuffle(elements)
            # 选择一部分元素，作为数据的非稀疏元素
            elements = elements[:index_count]
            # 生成随机数据，每个样本包含指定数量的元素
            data = [
                [
                    elements[random.randint(0, index_count - 1)]
                    for _ in range(self.input_dim)
                ]
                for _ in range(self.input_samples)
            ]
            # 将数据转换为NumPy数组，然后转换为PyTorch张量返回
            return torch.from_numpy(np.array(data))

        # 使用内部函数生成输入数据，并赋值给self.input
        self.input = generate_input()
        # 生成随机目标数据，范围为[0, max_val)，并赋值给self.target
        self.target = torch.randint(0, max_val, [sample_count])

    # 返回输入数据的长度
    def __len__(self):
        return len(self.input)

    # 获取指定索引处的输入数据和目标数据
    def __getitem__(self, index):
        return self.input[index], self.target[index]
```