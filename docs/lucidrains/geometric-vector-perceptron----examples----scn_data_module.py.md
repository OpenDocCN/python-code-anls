# `.\lucidrains\geometric-vector-perceptron\examples\scn_data_module.py`

```py
# 导入必要的模块
from argparse import ArgumentParser
from typing import List, Optional
from typing import Union

import numpy as np
import pytorch_lightning as pl
import sidechainnet
from sidechainnet.dataloaders.collate import get_collate_fn
from sidechainnet.utils.sequence import ProteinVocabulary
from torch.utils.data import DataLoader, Dataset

# 定义自定义数据集类
class ScnDataset(Dataset):
    def __init__(self, dataset, max_len: int):
        super(ScnDataset, self).__init__()
        self.dataset = dataset

        self.max_len = max_len
        self.scn_collate_fn = get_collate_fn(False)
        self.vocab = ProteinVocabulary()

    # 定义数据集的拼接函数
    def collate_fn(self, batch):
        batch = self.scn_collate_fn(batch)
        real_seqs = [
            "".join([self.vocab.int2char(aa) for aa in seq])
            for seq in batch.int_seqs.numpy()
        ]
        seq = real_seqs[0][: self.max_len]
        true_coords = batch.crds[0].view(-1, 14, 3)[: self.max_len].view(-1, 3)
        angles = batch.angs[0, : self.max_len]
        mask = batch.msks[0, : self.max_len]

        # 计算填充序列的长度
        padding_seq = (np.array([*seq]) == "_").sum()
        return {
            "seq": seq,
            "true_coords": true_coords,
            "angles": angles,
            "padding_seq": padding_seq,
            "mask": mask,
        }

    # 获取数据集中指定索引的数据
    def __getitem__(self, index: int):
        return self.dataset[index]

    # 返回数据集的长度
    def __len__(self) -> int:
        return len(self.dataset)

# 定义数据模块类
class ScnDataModule(pl.LightningDataModule):
    # 添加数据特定参数
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--casp_version", type=int, default=7)
        parser.add_argument("--scn_dir", type=str, default="./sidechainnet_data")
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--eval_batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--train_max_len", type=int, default=256)
        parser.add_argument("--eval_max_len", type=int, default=256)

        return parser

    # 初始化数据模块
    def __init__(
        self,
        casp_version: int = 7,
        scn_dir: str = "./sidechainnet_data",
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        num_workers: int = 1,
        train_max_len: int = 256,
        eval_max_len: int = 256,
        **kwargs,
    ):
        super().__init__()

        assert train_batch_size == eval_batch_size == 1, "batch size must be 1 for now"

        self.casp_version = casp_version
        self.scn_dir = scn_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.train_max_len = train_max_len
        self.eval_max_len = eval_max_len

    # 设置数据模块
    def setup(self, stage: Optional[str] = None):
        dataloaders = sidechainnet.load(
            casp_version=self.casp_version,
            scn_dir=self.scn_dir,
            with_pytorch="dataloaders",
        )
        print(
            dataloaders.keys()
        )  # ['train', 'train_eval', 'valid-10', ..., 'valid-90', 'test']

        self.train = ScnDataset(dataloaders["train"].dataset, self.train_max_len)
        self.val = ScnDataset(dataloaders["valid-90"].dataset, self.eval_max_len)
        self.test = ScnDataset(dataloaders["test"].dataset, self.eval_max_len)

    # 获取训练数据加载器
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.train.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    # 定义用于验证数据集的数据加载器函数，返回一个数据加载器对象或对象列表
    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        # 创建一个数据加载器对象，用于加载验证数据集
        return DataLoader(
            self.val,  # 使用验证数据集
            batch_size=self.eval_batch_size,  # 指定批量大小
            shuffle=False,  # 不打乱数据集顺序
            collate_fn=self.val.collate_fn,  # 使用验证数据集的数据整理函数
            num_workers=self.num_workers,  # 指定数据加载器的工作进程数
            pin_memory=True,  # 将数据加载到 CUDA 固定内存中
        )

    # 定义用于测试数据集的数据加载器函数，返回一个数据加载器对象或对象列表
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        # 创建一个数据加载器对象，用于加载测试数据集
        return DataLoader(
            self.test,  # 使用测试数据集
            batch_size=self.eval_batch_size,  # 指定批量大小
            shuffle=False,  # 不打乱数据集顺序
            collate_fn=self.test.collate_fn,  # 使用测试数据集的数据整理函数
            num_workers=self.num_workers,  # 指定数据加载器的工作进程数
            pin_memory=True,  # 将数据加载到 CUDA 固定内存中
        )
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建一个 ScnDataModule 的实例对象
    dm = ScnDataModule()
    # 设置数据模块
    dm.setup()

    # 获取训练数据加载器
    train = dm.train_dataloader()
    # 打印训练数据加载器的长度
    print("train length", len(train))

    # 获取验证数据加载器
    valid = dm.val_dataloader()
    # 打印验证数据加载器的长度
    print("valid length", len(valid))

    # 获取测试数据加载器
    test = dm.test_dataloader()
    # 打印测试数据加载器的长度
    print("test length", len(test))

    # 遍历测试数据加载器
    for batch in test:
        # 打印当前批次的数据
        print(batch)
        # 跳出循环，只打印第一个批次的数据
        break
```