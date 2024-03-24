# `.\lucidrains\enformer-pytorch\enformer_pytorch\data.py`

```py
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 torch.utils.data 中导入 Dataset 类
from torch.utils.data import Dataset

# 导入 polars 库并重命名为 pl
import polars as pl
# 导入 numpy 库并重命名为 np
import numpy as np
# 从 random 中导入 randrange 和 random 函数
from random import randrange, random
# 从 pathlib 中导入 Path 类
from pathlib import Path
# 从 pyfaidx 中导入 Fasta 类

import pyfaidx.Fasta

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回输入值
def identity(t):
    return t

# 将输入值转换为列表
def cast_list(t):
    return t if isinstance(t, list) else [t]

# 返回一个随机布尔值
def coin_flip():
    return random() > 0.5

# 基因组函数转换

# 创建一个包含 ASCII 码对应索引的张量
seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord('a')] = 0
seq_indices_embed[ord('c')] = 1
seq_indices_embed[ord('g')] = 2
seq_indices_embed[ord('t')] = 3
seq_indices_embed[ord('n')] = 4
seq_indices_embed[ord('A')] = 0
seq_indices_embed[ord('C')] = 1
seq_indices_embed[ord('G')] = 2
seq_indices_embed[ord('T')] = 3
seq_indices_embed[ord('N')] = 4
seq_indices_embed[ord('.')] = -1

# 创建一个包含 one-hot 编码的张量
one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord('a')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('c')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('g')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('t')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('n')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('A')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('C')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('G')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('T')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('N')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('.')] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

# 创建一个用于反向互补的映射张量
reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()

# 将字符串转换为张量
def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.fromstring(t, dtype = np.uint8), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]

# 将字符串转换为序列索引
def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]

# 将字符串转换为 one-hot 编码
def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]

# 将序列索引转换为 one-hot 编码
def seq_indices_to_one_hot(t, padding = -1):
    is_padding = t == padding
    t = t.clamp(min = 0)
    one_hot = F.one_hot(t, num_classes = 5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out

# 数据增强

# 反向互补序列索引
def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims = (-1,))

# 反向互补 one-hot 编码
def one_hot_reverse_complement(one_hot):
    *_, n, d = one_hot.shape
    assert d == 4, 'must be one hot encoding with last dimension equal to 4'
    return torch.flip(one_hot, (-1, -2))

# 处理 bed 文件

# 定义 FastaInterval 类
class FastaInterval():
    def __init__(
        self,
        *,
        fasta_file,
        context_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
    # 定义一个方法，用于生成指定染色体上指定区间的序列
    def __call__(self, chr_name, start, end, return_augs = False):
        # 计算区间长度
        interval_length = end - start
        # 获取染色体序列
        chromosome = self.seqs[chr_name]
        # 获取染色体序列长度
        chromosome_length = len(chromosome)

        # 如果存在平移增强参数
        if exists(self.shift_augs):
            # 获取最小和最大平移值
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            # 计算实际的最小和最大平移值
            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            # 随机选择平移值
            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        # 初始化左右填充值
        left_padding = right_padding = 0

        # 如果存在上下文长度参数且区间长度小于上下文长度
        if exists(self.context_length) and interval_length < self.context_length:
            # 计算额外的序列长度
            extra_seq = self.context_length - interval_length

            # 计算左右额外序列长度
            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        # 处理左边界溢出
        if start < 0:
            left_padding = -start
            start = 0

        # 处理右边界溢出
        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        # 生成序列并进行填充
        seq = ('.' * left_padding) + str(chromosome[start:end]) + ('.' * right_padding)

        # 判断是否需要进行反向互补增强
        should_rc_aug = self.rc_aug and coin_flip()

        # 如果需要返回序列索引
        if self.return_seq_indices:
            # 将序列转换为索引
            seq = str_to_seq_indices(seq)

            # 如果需要反向互补增强
            if should_rc_aug:
                seq = seq_indices_reverse_complement(seq)

            return seq

        # 将序列转换为独热编码
        one_hot = str_to_one_hot(seq)

        # 如果需要反向互补增强
        if should_rc_aug:
            one_hot = one_hot_reverse_complement(one_hot)

        # 如果不需要返回增强数据
        if not return_augs:
            return one_hot

        # 返回平移整数以及是否激活反向互补的布尔值
        rand_shift_tensor = torch.tensor([rand_shift])
        rand_aug_bool_tensor = torch.tensor([should_rc_aug])

        return one_hot, rand_shift_tensor, rand_aug_bool_tensor
# 定义一个继承自 Dataset 的 GenomeIntervalDataset 类
class GenomeIntervalDataset(Dataset):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        bed_file,
        fasta_file,
        filter_df_fn = identity,
        chr_bed_to_fasta_map = dict(),
        context_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False,
        return_augs = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 将 bed_file 转换为 Path 对象
        bed_path = Path(bed_file)
        # 断言 bed 文件路径存在
        assert bed_path.exists(), 'path to .bed file must exist'

        # 读取 bed 文件内容到 DataFrame
        df = pl.read_csv(str(bed_path), separator = '\t', has_header = False)
        # 对 DataFrame 应用过滤函数
        df = filter_df_fn(df)
        # 将过滤后的 DataFrame 赋值给实例变量 df
        self.df = df

        # 如果 bed 文件中的染色体名称与 fasta 文件中的键名不同，可以在运行时重新映射
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map

        # 创建 FastaInterval 对象，传入 fasta 文件路径和其他参数
        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            context_length = context_length,
            return_seq_indices = return_seq_indices,
            shift_augs = shift_augs,
            rc_aug = rc_aug
        )

        # 设置是否返回增强数据的标志
        self.return_augs = return_augs

    # 返回数据集的长度
    def __len__(self):
        return len(self.df)

    # 根据索引获取数据
    def __getitem__(self, ind):
        # 获取指定索引处的区间信息
        interval = self.df.row(ind)
        # 解析区间信息中的染色体名称、起始位置和结束位置
        chr_name, start, end = (interval[0], interval[1], interval[2])
        # 如果染色体名称需要重新映射，则进行映射
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        # 调用 FastaInterval 对象的方法，返回指定区间的数据
        return self.fasta(chr_name, start, end, return_augs = self.return_augs)
```