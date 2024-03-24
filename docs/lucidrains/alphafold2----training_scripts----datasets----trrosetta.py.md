# `.\lucidrains\alphafold2\training_scripts\datasets\trrosetta.py`

```
import pickle
import string
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.linalg as LA
import prody
import torch
from Bio import SeqIO
from einops import repeat
from sidechainnet.utils.measure import get_seq_coords_and_angles
from sidechainnet.utils.sequence import ProteinVocabulary
from torch.utils.data import DataLoader, Dataset
from alphafold2_pytorch.constants import DISTOGRAM_BUCKETS
from tqdm import tqdm

try:
    import pytorch_lightning as pl

    LightningDataModule = pl.LightningDataModule
except ImportError:
    LightningDataModule = object

CACHE_PATH = Path("~/.cache/alphafold2_pytorch").expanduser()
DATA_DIR = CACHE_PATH / "trrosetta" / "trrosetta"
URL = "http://s3.amazonaws.com/proteindata/data_pytorch/trrosetta.tar.gz"

REMOVE_KEYS = dict.fromkeys(string.ascii_lowercase)
REMOVE_KEYS["."] = None
REMOVE_KEYS["*"] = None
translation = str.maketrans(REMOVE_KEYS)

DEFAULT_VOCAB = ProteinVocabulary()


def default_tokenize(seq: str) -> List[int]:
    return [DEFAULT_VOCAB[ch] for ch in seq]


def read_fasta(filename: str) -> List[Tuple[str, str]]:
    def remove_insertions(sequence: str) -> str:
        return sequence.translate(translation)

    return [
        (record.description, remove_insertions(str(record.seq)))
        for record in SeqIO.parse(filename, "fasta")
    ]


def read_pdb(pdb: str):
    ag = prody.parsePDB(pdb)
    for chain in ag.iterChains():
        angles, coords, seq = get_seq_coords_and_angles(chain)
        return angles, coords, seq


def download_file(url, filename=None, root=CACHE_PATH):
    import os
    import urllib

    root.mkdir(exist_ok=True, parents=True)
    filename = filename or os.path.basename(url)

    download_target = root / filename
    download_target_tmp = root / f"tmp.{filename}"

    if download_target.exists() and not download_target.is_file():
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if download_target.is_file():
        return download_target

    with urllib.request.urlopen(url) as source, open(
        download_target_tmp, "wb"
    ) as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    download_target_tmp.rename(download_target)
    return download_target


def get_or_download(url: str = URL):
    """
    download and extract trrosetta data
    """
    import tarfile

    file = CACHE_PATH / "trrosetta.tar.gz"
    dir = CACHE_PATH / "trrosetta"
    dir_temp = CACHE_PATH / "trrosetta_tmp"
    if dir.is_dir():
        print(f"Load cached data from {dir}")
        return dir

    if not file.is_file():
        print(f"Cache not found, download from {url} to {file}")
        download_file(url)

    print(f"Extract data from {file} to {dir}")
    with tarfile.open(file, "r:gz") as tar:
        tar.extractall(dir_temp)

    dir_temp.rename(dir)

    return dir


def pad_sequences(sequences, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class TrRosettaDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        list_path: Path,
        tokenize: Callable[[str], List[int]],
        seq_pad_value: int = 20,
        random_sample_msa: bool = False,
        max_seq_len: int = 300,
        max_msa_num: int = 300,
        overwrite: bool = False,
    ):
        self.data_dir = data_dir
        self.file_list: List[Path] = self.read_file_list(data_dir, list_path)

        self.tokenize = tokenize
        self.seq_pad_value = seq_pad_value

        self.random_sample_msa = random_sample_msa
        self.max_seq_len = max_seq_len
        self.max_msa_num = max_msa_num

        self.overwrite = overwrite

    def __len__(self) -> int:
        return len(self.file_list)

    def read_file_list(self, data_dir: Path, list_path: Path):
        file_glob = (data_dir / "npz").glob("*.npz")
        files = set(list_path.read_text().split())
        if len(files) == 0:
            raise ValueError("Passed an empty split file set")

        file_list = [f for f in file_glob if f.name in files]
        if len(file_list) != len(files):
            num_missing = len(files) - len(file_list)
            raise FileNotFoundError(
                f"{num_missing} specified split files not found in directory"
            )

        return file_list

    def has_cache(self, index):
        if self.overwrite:
            return False

        path = (self.data_dir / "cache" / self.file_list[index].stem).with_suffix(
            ".pkl"
        )
        return path.is_file()

    def write_cache(self, index, data):
        path = (self.data_dir / "cache" / self.file_list[index].stem).with_suffix(
            ".pkl"
        )
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as file:
            pickle.dump(data, file)

    def read_cache(self, index):
        path = (self.data_dir / "cache" / self.file_list[index].stem).with_suffix(
            ".pkl"
        )
        with open(path, "rb") as file:
            return pickle.load(file)

    def __getitem__(self, index):
        if self.has_cache(index):
            item = self.read_cache(index)
        else:
            id = self.file_list[index].stem
            pdb_path = self.data_dir / "pdb" / f"{id}.pdb"
            msa_path = self.data_dir / "a3m" / f"{id}.a3m"
            _, msa = zip(*read_fasta(str(msa_path)))
            msa = np.array([np.array(list(seq)) for seq in msa])
            angles, coords, seq = read_pdb(str(pdb_path))
            seq = np.array(list(seq))
            coords = coords.reshape((coords.shape[0] // 14, 14, 3))
            dist = self.get_bucketed_distance(seq, coords, subset="ca")
            item = {
                "id": id,
                "seq": seq,
                "msa": msa,
                "coords": coords,
                "angles": angles,
                "dist": dist
            }
            self.write_cache(index, item)

        item["msa"] = self.sample(item["msa"], self.max_msa_num, self.random_sample_msa)
        item = self.crop(item, self.max_seq_len)
        return item

    def calc_cb(self, coord):
        N = coord[0]
        CA = coord[1]
        C = coord[2]

        b = CA - N
        c = C - CA
        a = np.cross(b, c)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        return CB

    def get_bucketed_distance(
        self, seq, coords, subset="ca", start=2, bins=DISTOGRAM_BUCKETS-1, step=0.5
        assert subset in ("ca", "cb")
        # 检查 subset 是否为 "ca" 或 "cb"
        
        if subset == "ca":
            coords = coords[:, 1, :]
            # 如果 subset 为 "ca"，则只保留坐标的第二列数据
        
        elif subset == "cb":
            cb_coords = []
            # 创建空列表用于存储 cb 坐标数据
            for res, coord in zip(seq, coords):
                # 遍历序列和坐标数据
                if res == "G":
                    # 如果氨基酸为 "G"
                    cb = self.calc_cb(coord)
                    # 计算 cb 坐标
                    cb_coords.append(cb)
                    # 将计算得到的 cb 坐标添加到列表中
                else:
                    cb_coords.append(coord[4, :])
                    # 如果氨基酸不是 "G"，则将坐标的第五行数据添加到列表中
            coords = np.array(cb_coords)
            # 将列表转换为 NumPy 数组，更新坐标数据
        
        vcs = coords + np.zeros([coords.shape[0]] + list(coords.shape))
        # 创建与 coords 形状相同的全零数组，并与 coords 相加，得到 vcs
        vcs = vcs - np.swapaxes(vcs, 0, 1)
        # 将 vcs 与其转置矩阵相减，更新 vcs
        distance_map = LA.norm(vcs, axis=2)
        # 计算 vcs 的二范数，得到距离矩阵
        mask = np.ones(distance_map.shape) - np.eye(distance_map.shape[0])
        # 创建与距离矩阵形状相同的全一数组，减去单位矩阵，得到 mask
        low_pos = np.where(distance_map < start)
        # 找出距离矩阵中小于 start 的位置
        high_pos = np.where(distance_map >= start + step * bins)
        # 找出距离矩阵中大于等于 start + step * bins 的位置

        mask[low_pos] = 0
        # 将低于 start 的位置在 mask 中置为 0
        distance_map = (distance_map - start) // step
        # 对距离矩阵进行归一化处理
        distance_map[high_pos] = bins
        # 将高于 start + step * bins 的位置在距离矩阵中置为 bins
        dist = (distance_map * mask).astype(int)
        # 将归一化后的距离矩阵乘以 mask，并转换为整数类型，得到最终距离矩阵
        return dist
        # 返回距离矩阵

    def crop(self, item, max_seq_len: int):
        # 截取序列数据，使其长度不超过 max_seq_len
        seq_len = len(item["seq"])

        if seq_len <= max_seq_len or max_seq_len <= 0:
            return item
            # 如果序列长度小于等于 max_seq_len 或 max_seq_len 小于等于 0，则直接返回原始数据

        start = 0
        end = start + max_seq_len
        # 计算截取的起始位置和结束位置

        item["seq"] = item["seq"][start:end]
        item["msa"] = item["msa"][:, start:end]
        item["coords"] = item["coords"][start:end]
        item["angles"] = item["angles"][start:end]
        item["dist"] = item["dist"][start:end, start:end]
        # 对 item 中的各项数据进行截取操作
        return item
        # 返回截取后的数据

    def sample(self, msa, max_msa_num: int, random: bool):
        # 对多序列进行采样，使其数量不超过 max_msa_num
        num_msa, seq_len = len(msa), len(msa[0])

        if num_msa <= max_msa_num or max_msa_num <= 0:
            return msa
            # 如果多序列数量小于等于 max_msa_num 或 max_msa_num 小于等于 0，则直接返回原始数据

        if random:
            # 如果需要随机采样
            num_sample = max_msa_num - 1
            # 计算需要采样的数量
            indices = np.random.choice(num_msa - 1, size=num_sample, replace=False) + 1
            # 随机选择索引进行采样
            indices = np.pad(indices, [1, 0], "constant")
            # 在索引数组前面添加一个元素
            return msa[indices]
            # 返回采样后的多序列数据
        else:
            return msa[:max_msa_num]
            # 如果不需要随机采样，则直接返回前 max_msa_num 个多序列数据

    def collate_fn(self, batch):
        # 对批量数据进行整理
        b = len(batch)
        # 获取批量数据的长度
        batch = {k: [item[k] for item in batch] for k in batch[0]}
        # 将批量数据转换为字典形式，按照键值进行整理

        id = batch["id"]
        seq = batch["seq"]
        msa = batch["msa"]
        coords = batch["coords"]
        angles = batch["angles"]
        dist = batch["dist"]
        # 获取批量数据中的各项内容

        lengths = torch.LongTensor([len(x[0]) for x in msa])
        depths = torch.LongTensor([len(x) for x in msa])
        max_len = lengths.max()
        max_depth = depths.max()
        # 计算多序列数据的长度和深度信息

        seq = pad_sequences(
            [torch.LongTensor(self.tokenize(seq_)) for seq_ in seq], self.seq_pad_value,
        )
        # 对序列数据进行填充处理

        msa = pad_sequences(
            [torch.LongTensor([self.tokenize(seq_) for seq_ in msa_]) for msa_ in msa],
            self.seq_pad_value,
        )
        # 对多序列数据进行填充处理

        coords = pad_sequences([torch.FloatTensor(x) for x in coords], 0.0)
        # 对坐标数据进行填充处理

        angles = pad_sequences([torch.FloatTensor(x) for x in angles], 0.0)
        # 对角度数据进行填充处理

        dist = pad_sequences([torch.LongTensor(x) for x in dist], -100)
        # 对距离数据进行填充处理

        mask = repeat(torch.arange(max_len), "l -> b l", b=b) < repeat(
            lengths, "b -> b l", l=max_len
        )
        # 生成序列数据的掩码

        msa_seq_mask = repeat(
            torch.arange(max_len), "l -> b s l", b=b, s=max_depth
        ) < repeat(lengths, "b -> b s l", s=max_depth, l=max_len)
        # 生成多序列数据的序列掩码

        msa_depth_mask = repeat(
            torch.arange(max_depth), "s -> b s l", b=b, l=max_len
        ) < repeat(depths, "b -> b s l", s=max_depth, l=max_len)
        # 生成多序列数据的深度掩码

        msa_mask = msa_seq_mask & msa_depth_mask
        # 组合多序列数据的掩码

        return {
            "id": id,
            "seq": seq,
            "msa": msa,
            "coords": coords,
            "angles": angles,
            "mask": mask,
            "msa_mask": msa_mask,
            "dist": dist,
        }
        # 返回整理后的批量��据
class TrRosettaDataModule(LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        # 创建参数解析器
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # 添加数据目录参数
        parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
        # 添加训练批量大小参数
        parser.add_argument("--train_batch_size", type=int, default=1)
        # 添加评估批量大小参数
        parser.add_argument("--eval_batch_size", type=int, default=1)
        # 添加测试批量大小参数
        parser.add_argument("--test_batch_size", type=int, default=1)
        # 添加工作进程数参数
        parser.add_argument("--num_workers", type=int, default=0)
        # 添加训练最大序列长度参数
        parser.add_argument("--train_max_seq_len", type=int, default=256)
        # 添加评估最大序列长度参数
        parser.add_argument("--eval_max_seq_len", type=int, default=256)
        # 添加测试最大序列长度参数
        parser.add_argument("--test_max_seq_len", type=int, default=-1)
        # 添加训练最大 MSA 数量参数
        parser.add_argument("--train_max_msa_num", type=int, default=256)
        # 添加评估最大 MSA 数量参数
        parser.add_argument("--eval_max_msa_num", type=int, default=256)
        # 添加测试最大 MSA 数量参数
        parser.add_argument("--test_max_msa_num", type=int, default=1000)
        # 添加覆盖参数
        parser.add_argument("--overwrite", dest="overwrite", action="store_true")
        # 返回参数解析器
        return parser

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        test_batch_size: int = 1,
        num_workers: int = 0,
        train_max_seq_len: int = 256,
        eval_max_seq_len: int = 256,
        test_max_seq_len: int = -1,
        train_max_msa_num: int = 32,
        eval_max_msa_num: int = 32,
        test_max_msa_num: int = 64,
        tokenize: Callable[[str], List[int]] = default_tokenize,
        seq_pad_value: int = 20,
        overwrite: bool = False,
        **kwargs,
    ):
        # 调用父类构造函数
        super(TrRosettaDataModule, self).__init__()
        # 解析数据目录
        self.data_dir = Path(data_dir).expanduser().resolve()
        # 初始化各参数
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.train_max_seq_len = train_max_seq_len
        self.eval_max_seq_len = eval_max_seq_len
        self.test_max_seq_len = test_max_seq_len
        self.train_max_msa_num = train_max_msa_num
        self.eval_max_msa_num = eval_max_msa_num
        self.test_max_msa_num = test_max_msa_num
        self.tokenize = tokenize
        self.seq_pad_value = seq_pad_value
        self.overwrite = overwrite
        # 获取或下载数据
        get_or_download()

    def setup(self, stage: Optional[str] = None):
        # 设置训练数据集
        self.train = TrRosettaDataset(
            self.data_dir,
            self.data_dir / "train_files.txt",
            self.tokenize,
            self.seq_pad_value,
            random_sample_msa=True,
            max_seq_len=self.train_max_seq_len,
            max_msa_num=self.train_max_msa_num,
            overwrite=self.overwrite,
        )
        # 设置验证数据集
        self.val = TrRosettaDataset(
            self.data_dir,
            self.data_dir / "valid_files.txt",
            self.tokenize,
            self.seq_pad_value,
            random_sample_msa=False,
            max_seq_len=self.eval_max_seq_len,
            max_msa_num=self.eval_max_msa_num,
            overwrite=self.overwrite,
        )
        # 设置测试数据集
        self.test = TrRosettaDataset(
            self.data_dir,
            self.data_dir / "valid_files.txt",
            self.tokenize,
            self.seq_pad_value,
            random_sample_msa=False,
            max_seq_len=self.test_max_seq_len,
            max_msa_num=self.test_max_msa_num,
            overwrite=self.overwrite,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        # 返回训练数据加载器
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.train.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        # 返回验证数据加载器
        return DataLoader(
            self.val,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.val.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        # 返回测试数据加载器
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test.collate_fn,
            num_workers=self.num_workers,
        )


def test():
    # 创建数据模块实例
    dm = TrRosettaDataModule(train_batch_size=1, num_workers=4)
    # 设置数据
    dm.setup()

    # 遍历训练数据加载器
    for batch in dm.train_dataloader():
        print("id", batch["id"])
        print("seq", batch["seq"].shape, batch["seq"])
        print("msa", batch["msa"].shape, batch["msa"][..., :20])
        print("msa", batch["msa"].shape, batch["msa"][..., -20:])
        print("coords", batch["coords"].shape)
        print("angles", batch["angles"].shape)
        print("mask", batch["mask"].shape)
        print("msa_mask", batch["msa_mask"].shape)
        print("dist", batch["dist"].shape, batch["dist"])
        break


if __name__ == "__main__":
    test()
```